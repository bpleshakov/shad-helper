"""App-owned Sign in with ChatGPT credentials; no API-key or Codex fallback."""

import base64
import fcntl
import hashlib
import json
import os
import secrets
import tempfile
import time
import uuid
import webbrowser
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt

from .network import openai_http_client, resolve_openai_proxy

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
AUTHORIZE = f"{ISSUER}/api/accounts/authorize"
TOKEN = f"{ISSUER}/api/accounts/oauth/token"
DIRECT_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = f"openid profile email offline_access resource.invoke {DIRECT_SCOPE}"
TERMINAL_REFRESH_ERRORS = {
    "invalid_grant", "invalid_refresh_token", "token_expired", "refresh_token_expired",
    "refresh_token_invalidated", "refresh_token_reused",
}


class PlanAccessError(RuntimeError):
    def __init__(self, message, *, code=None, status=None, request_id=None):
        super().__init__(message)
        self.code = code
        self.status = status
        self.request_id = request_id


def http_error(response):
    """Keep status/code/request ID without printing credentials or raw bodies."""
    try:
        body = response.json()
    except ValueError:
        body = {}
    error = body.get("error", {}) if isinstance(body, dict) else {}
    code = error.get("code") if isinstance(error, dict) else error
    request_id = response.headers.get("x-request-id") or response.headers.get("openai-request-id")
    message = f"ChatGPT plan request failed (HTTP {response.status_code}"
    message += f", {code}" if code else ""
    message += "). Check account eligibility or ChatGPT Settings → Usage."
    return PlanAccessError(message, code=code, status=response.status_code, request_id=request_id)


def _read(path):
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PlanAccessError("Cannot read app-owned authentication state; sign in again.") from exc
    if not isinstance(value, dict):
        raise PlanAccessError("Invalid app-owned authentication state; sign in again.")
    return value


def _save(path, value):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".auth-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _locked(state_dir):
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_dir.chmod(0o700)
    fd = os.open(state_dir / ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "w") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _json_request(client, method, url, **kwargs):
    try:
        response = client.request(method, url, **kwargs)
    except httpx.HTTPError as exc:
        raise PlanAccessError("Cannot reach Sign in with ChatGPT; retry later.") from exc
    if not response.is_success:
        raise http_error(response)
    try:
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Expected an object")
        return body
    except ValueError as exc:
        raise PlanAccessError("Sign in with ChatGPT returned invalid JSON.") from exc


def _verify_identity(client, token, client_id, nonce=None):
    discovery = _json_request(client, "GET", f"{ISSUER}/.well-known/openid-configuration")
    jwks_uri = discovery.get("jwks_uri", "")
    if discovery.get("issuer") != ISSUER or urlsplit(jwks_uri).scheme != "https" or urlsplit(jwks_uri).netloc != "auth.openai.com":
        raise PlanAccessError("Unexpected OpenAI identity discovery configuration.")
    jwks = _json_request(client, "GET", jwks_uri)
    try:
        header = jwt.get_unverified_header(token)
        if header.get("alg") not in {"RS256", "ES256"}:
            raise ValueError("Unsupported signing algorithm")
        key = next(key for key in jwks["keys"] if key.get("kid") == header.get("kid"))
        claims = jwt.decode(
            token, jwt.PyJWK.from_dict(key).key, algorithms=[header["alg"]],
            audience=client_id, issuer=ISSUER, leeway=5,
            options={"require": ["sub", "exp", "iat", "iss", "aud"]},
        )
        if not claims["sub"] or (nonce is not None and not secrets.compare_digest(str(claims.get("nonce", "")), nonce)):
            raise ValueError("Invalid identity or nonce")
        return claims
    except (jwt.PyJWTError, KeyError, ValueError, StopIteration, TypeError) as exc:
        raise PlanAccessError("OpenAI identity-token verification failed; sign in again.") from exc


def _browser_callback(parameters, opener, timeout):
    result = {}

    class Callback(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            values = parse_qs(parsed.query)
            valid = parsed.path == "/auth/callback" and values.get("state") == [parameters["state"]]
            self.send_response(200 if valid else 400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b"Return to shad-helper." if valid else b"Invalid callback.")
            if valid:
                result.update({key: value[0] for key, value in values.items() if len(value) == 1})

        def log_message(self, *args):
            pass  # The callback URL contains an authorization code.

    with HTTPServer(("127.0.0.1", 0), Callback) as server:
        parameters["redirect_uri"] = f"http://127.0.0.1:{server.server_port}/auth/callback"
        if not opener(f"{AUTHORIZE}?{urlencode(parameters)}"):
            raise PlanAccessError("Could not open the system browser for Sign in with ChatGPT.")
        deadline = time.monotonic() + timeout
        while not result and time.monotonic() < deadline:
            server.timeout = min(1, max(0, deadline - time.monotonic()))
            server.handle_request()
    if not result:
        raise PlanAccessError("Sign in with ChatGPT timed out; run login again.")
    return result


def _token_record(tokens, previous):
    required = ("access_token", "refresh_token")
    if any(not isinstance(tokens.get(key), str) or not tokens[key] for key in required):
        raise PlanAccessError("ChatGPT did not grant renewable plan access; sign in again.")
    try:
        expires_in = int(tokens["expires_in"])
        if expires_in <= 0 or tokens.get("token_type", "").lower() != "bearer":
            raise ValueError("Invalid expiry or token type")
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanAccessError("Invalid ChatGPT token response.") from exc
    record = dict(previous)
    record.update({key: tokens[key] for key in required})
    if tokens.get("id_token"):
        record["id_token"] = tokens["id_token"]
    record.update(token_type="Bearer", expires_at=time.time() + expires_in,
                  scopes=tokens.get("scope", " ".join(previous.get("scopes", []))).split())
    if "earliest_refresh_at" in tokens:
        record["earliest_refresh_at"] = tokens["earliest_refresh_at"]
    return record


def login(state_dir: Path, *, proxy=None, transport=None, browser_opener=None, callback_timeout=300) -> dict:
    """Register/sign in this local app; return safe account metadata only."""
    proxy = resolve_openai_proxy(proxy)
    state_dir = Path(state_dir)
    with _locked(state_dir), openai_http_client(proxy=proxy, transport=transport, timeout=30) as client:
        registration_path = state_dir / "registration.json"
        registration = _read(registration_path)
        if "ext_agent_host_id" not in registration:
            registration["ext_agent_host_id"] = f"urn:uuid:{uuid.uuid4()}"
            _save(registration_path, registration)
        previous = _read(state_dir / "credentials.json")
        client_id = registration.get("client_id")
        verifier = secrets.token_urlsafe(64)
        parameters = {
            "client_id": client_id or "dynamic_agent_client", "response_type": "code",
            "scope": SCOPES, "resource": RESOURCE, "state": secrets.token_urlsafe(32),
            "nonce": secrets.token_urlsafe(32), "code_challenge_method": "S256",
            "code_challenge": base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("="),
            "ext_agent_host_id": registration["ext_agent_host_id"],
        }
        if not client_id:
            parameters["agent_name_hint"] = "shad-helper"
        elif previous.get("id_token"):
            parameters["id_token_hint"] = previous["id_token"]
        if client_id and previous and DIRECT_SCOPE not in previous.get("scopes", []):
            parameters["prompt"] = "consent"
        if browser_opener is None:
            print("Use the same OpenAI proxy in your browser for sign-in. "
                  "The callback to 127.0.0.1 stays local.")
        callback = _browser_callback(parameters, browser_opener or webbrowser.open, callback_timeout)
        if not secrets.compare_digest(callback.get("state", ""), parameters["state"]):
            raise PlanAccessError("Sign-in callback state did not match.")
        if callback.get("error"):
            raise PlanAccessError("ChatGPT sign-in or consent was declined.", code=callback["error"])
        issued_id = callback.get("client_id", client_id)
        if not issued_id or issued_id == "dynamic_agent_client" or (client_id and issued_id != client_id):
            raise PlanAccessError("Sign-in callback returned an invalid client registration.")
        registration["client_id"] = issued_id
        _save(registration_path, registration)
        if not callback.get("code"):
            raise PlanAccessError("Sign-in callback did not contain an authorization code.")
        tokens = _json_request(client, "POST", TOKEN, data={
            "grant_type": "authorization_code", "client_id": issued_id, "code": callback["code"],
            "code_verifier": verifier, "redirect_uri": parameters["redirect_uri"], "resource": RESOURCE,
        })
        identity = _verify_identity(client, tokens.get("id_token", ""), issued_id, parameters["nonce"])
        if previous.get("subject") and previous["subject"] != identity["sub"]:
            raise PlanAccessError("Sign-in account differs from this app's saved account.")
        record = _token_record(tokens, {**registration, "issuer": ISSUER, "subject": identity["sub"], "email": identity.get("email")})
        _save(state_dir / "credentials.json", record)
        return {key: record[key] for key in ("email", "subject", "client_id", "scopes")}


def access_token(state_dir: Path, *, proxy=None, transport=None) -> str:
    """Refresh near expiry under a process lock; never initiate browser sign-in."""
    proxy = resolve_openai_proxy(proxy)
    state_dir = Path(state_dir)
    with _locked(state_dir), openai_http_client(proxy=proxy, transport=transport, timeout=30) as client:
        path = state_dir / "credentials.json"
        record = _read(path)
        if DIRECT_SCOPE not in record.get("scopes", []) or not record.get("access_token"):
            raise PlanAccessError("Run shad-helper login and enable ChatGPT plan usage first.")
        if record.get("expires_at", 0) > time.time() + 120:
            return record["access_token"]
        if not record.get("refresh_token") or not record.get("client_id"):
            raise PlanAccessError("ChatGPT session cannot be renewed; run login again.")
        try:
            tokens = _json_request(client, "POST", TOKEN, data={
                "grant_type": "refresh_token", "client_id": record["client_id"],
                "refresh_token": record["refresh_token"], "resource": RESOURCE,
            })
        except PlanAccessError as exc:
            if exc.code in TERMINAL_REFRESH_ERRORS:
                for key in ("access_token", "refresh_token", "id_token"):
                    record.pop(key, None)
                _save(path, record)
            raise
        if tokens.get("id_token"):
            identity = _verify_identity(client, tokens["id_token"], record["client_id"])
            if identity["sub"] != record["subject"]:
                raise PlanAccessError("Refreshed ChatGPT identity differs from the saved account.")
        updated = _token_record(tokens, record)
        _save(path, updated)
        if DIRECT_SCOPE not in updated["scopes"]:
            raise PlanAccessError("ChatGPT plan permission was removed; run login again.")
        return updated["access_token"]
