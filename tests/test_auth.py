"""Offline unittest checks plus full OAuth/JWT pytest tests when dependencies exist."""

import ast
import base64
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import secrets
import tempfile
import time
import unittest
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlsplit


def _auth_module():
    path = Path(__file__).resolve().parents[1] / "shad_helper/auth.py"
    module = ModuleType("auth_offline_checks")
    module.__dict__.update({name: globals()[name] for name in (
        "base64", "contextmanager", "fcntl", "hashlib", "json", "os", "Path",
        "secrets", "tempfile", "time", "parse_qs", "urlencode", "urlsplit", "uuid")})
    module.httpx = SimpleNamespace(Client=lambda **kwargs: nullcontext(None), HTTPError=OSError)
    module.resolve_openai_proxy = lambda proxy=None: proxy or "http://127.0.0.1:8080"
    module.openai_http_client = lambda **kwargs: nullcontext(None)
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


class AuthStdlibTests(unittest.TestCase):
    def test_proxy_validation_precedes_state_and_browser(self):
        module = _auth_module()

        def reject_proxy(proxy):
            raise ValueError("Configure an OpenAI proxy")

        module.resolve_openai_proxy = reject_proxy
        module.openai_http_client = lambda **kwargs: self.fail("Client created without valid proxy")
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "new-state"
            for operation in (module.login, module.access_token):
                with self.assertRaisesRegex(ValueError, "proxy"):
                    operation(state)
                self.assertFalse(state.exists())

    def test_login_and_refresh_share_proxied_client_for_token_and_identity(self):
        module = _auth_module()
        configured_proxy = "http://127.0.0.1:8080"
        transport = object()
        options, requests = [], []
        claims = {"sub": "test-subject", "email": "example@example.test"}
        module.jwt = SimpleNamespace(
            get_unverified_header=lambda token: {"alg": "RS256", "kid": "key"},
            PyJWK=SimpleNamespace(from_dict=lambda key: SimpleNamespace(key="public-key")),
            decode=lambda *args, **kwargs: claims,
            PyJWTError=RuntimeError,
        )

        class Client:
            def request(self, method, url, **kwargs):
                requests.append((method, url))
                if url == module.TOKEN:
                    body = {"access_token": "test-access", "refresh_token": "test-refresh",
                            "id_token": "mock-identity", "expires_in": 3600,
                            "token_type": "Bearer", "scope": module.SCOPES}
                elif url.endswith("openid-configuration"):
                    body = {"issuer": module.ISSUER, "jwks_uri": module.ISSUER + "/jwks"}
                elif url.endswith("/jwks"):
                    body = {"keys": [{"kid": "key"}]}
                else:
                    raise AssertionError("Unexpected request")
                return SimpleNamespace(is_success=True, json=lambda: body)

        def client(**kwargs):
            options.append(kwargs)
            return nullcontext(Client())

        def callback(parameters, *args):
            claims["nonce"] = parameters["nonce"]
            parameters["redirect_uri"] = "http://127.0.0.1:54321/auth/callback"
            return {"state": parameters["state"], "code": "test-code", "client_id": "oaiapp_test"}

        module.openai_http_client = client
        module._browser_callback = callback
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            module.login(state, proxy=configured_proxy, transport=transport, browser_opener=lambda _: False)
            record = module._read(state / "credentials.json")
            record["expires_at"] = 0
            module._save(state / "credentials.json", record)
            self.assertEqual(module.access_token(state, proxy=configured_proxy, transport=transport), "test-access")
        self.assertEqual(options, [{"proxy": configured_proxy, "transport": transport, "timeout": 30}] * 2)
        self.assertEqual(requests, [("POST", module.TOKEN),
            ("GET", module.ISSUER + "/.well-known/openid-configuration"),
            ("GET", module.ISSUER + "/jwks")] * 2)

    def test_app_registration_pkce_and_identity_metadata(self):
        module = _auth_module()
        attempts, forms = [], []

        def callback(parameters, *args):
            parameters["redirect_uri"] = "http://127.0.0.1:54321/auth/callback"
            attempts.append(parameters.copy())
            return {"state": parameters["state"], "code": "test-code", "client_id": "oaiapp_test"}

        def exchange(client, method, url, **kwargs):
            forms.append(kwargs["data"])
            return {"access_token": "test-access", "refresh_token": "test-refresh",
                    "id_token": "mock-identity", "expires_in": 3600,
                    "token_type": "Bearer", "scope": module.SCOPES}

        def identity(client, token, client_id, nonce):
            self.assertEqual(client_id, "oaiapp_test")
            self.assertEqual(nonce, attempts[-1]["nonce"])
            return {"sub": "test-subject", "email": "example@example.test"}

        module._browser_callback, module._json_request, module._verify_identity = callback, exchange, identity
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            account = module.login(state, browser_opener=lambda _: False)
            module.login(state, browser_opener=lambda _: False)
            self.assertEqual(set(account), {"email", "subject", "client_id", "scopes"})
            self.assertEqual((state / "credentials.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual(state.stat().st_mode & 0o777, 0o700)
        self.assertEqual(attempts[0]["client_id"], "dynamic_agent_client")
        self.assertEqual(attempts[1]["client_id"], "oaiapp_test")
        self.assertEqual(attempts[0]["ext_agent_host_id"], attempts[1]["ext_agent_host_id"])
        self.assertNotEqual(attempts[0]["state"], attempts[1]["state"])
        self.assertNotEqual(attempts[0]["nonce"], attempts[1]["nonce"])
        for attempt, form in zip(attempts, forms):
            digest = hashlib.sha256(form["code_verifier"].encode()).digest()
            self.assertEqual(attempt["code_challenge"], base64.urlsafe_b64encode(digest).decode().rstrip("="))
            self.assertEqual(form["redirect_uri"], attempt["redirect_uri"])
            self.assertEqual(form["client_id"], "oaiapp_test")
            self.assertNotIn("client_secret", form)

    def test_wrong_callback_state_never_exchanges_code(self):
        module = _auth_module()
        module._browser_callback = lambda *args: {"state": "wrong", "code": "test-code", "client_id": "oaiapp_test"}
        module._json_request = lambda *args, **kwargs: self.fail("Invalid state exchanged a code")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(module.PlanAccessError):
                module.login(Path(directory), browser_opener=lambda _: False)
            self.assertFalse((Path(directory) / "credentials.json").exists())

    def test_atomic_token_storage_and_rotation(self):
        module = _auth_module()
        previous = {"client_id": "oaiapp_test", "scopes": [module.DIRECT_SCOPE]}
        record = module._token_record({"access_token": "new", "refresh_token": "rotated",
            "expires_in": 3600, "token_type": "Bearer"}, previous)
        self.assertEqual(record["refresh_token"], "rotated")
        self.assertEqual(record["scopes"], previous["scopes"])
        self.assertGreater(record["expires_at"], time.time() + 3500)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials.json"
            module._save(path, record)
            self.assertEqual(module._read(path), record)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(len(list(path.parent.iterdir())), 1)

    def test_loopback_handler_state_validation(self):
        module = _auth_module()
        tree = ast.parse(Path(__file__).read_text())
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                        and node.name == "test_loopback_callback_rejects_wrong_state_before_accepting_result")
        namespace = dict(auth=module, io=io, parse_qs=parse_qs, urlencode=urlencode, urlsplit=urlsplit)
        exec(compile(ast.Module(body=[function], type_ignores=[]), __file__, "exec"), namespace)
        namespace[function.name](SimpleNamespace(setattr=setattr))


_FULL_DEPS = all(importlib.util.find_spec(name) for name in ("httpx", "jwt", "cryptography", "pytest"))
if _FULL_DEPS:
    import base64
    import hashlib
    import io
    import json
    import stat
    import time
    from urllib.parse import parse_qs, urlencode, urlsplit

    import httpx
    import jwt
    import pytest
    from cryptography.hazmat.primitives.asymmetric import rsa

    from shad_helper import auth


    @pytest.fixture(autouse=True)
    def configured_openai_proxy(monkeypatch):
        monkeypatch.setenv("SHAD_HELPER_OPENAI_PROXY", "http://127.0.0.1:8080")


    @pytest.fixture
    def identity_key():
        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
        public.update(kid="test-key", alg="RS256", use="sig")
        return private, public


    def signed_identity(private, nonce, *, subject="subject-1", audience="oaiapp_test", issuer=auth.ISSUER):
        return jwt.encode({"sub": subject, "iss": issuer, "aud": audience, "iat": int(time.time()),
                           "exp": int(time.time()) + 3600, "nonce": nonce, "email": "test@example.test"},
                          private, algorithm="RS256", headers={"kid": "test-key"})


    def test_login_registers_validates_and_reuses_app_identity(tmp_path, monkeypatch, identity_key):
        private, public = identity_key
        pending = {}
        forms = []
        attempts = []
        client_options, request_urls = [], []
        proxy = "http://127.0.0.1:8081"
        original_client = auth.openai_http_client
        monkeypatch.setenv("NO_PROXY", "*")

        def proxied_client(**kwargs):
            client_options.append(kwargs)
            return original_client(**kwargs)

        def callback(parameters, opener, timeout):
            parameters["redirect_uri"] = "http://127.0.0.1:54321/auth/callback"
            pending.update(parameters)
            attempts.append(parameters.copy())
            return {"state": parameters["state"], "code": "test-code", "client_id": "oaiapp_test"}

        def request(request):
            request_urls.append(str(request.url))
            if str(request.url) == auth.TOKEN:
                form = parse_qs(request.content.decode())
                forms.append(form)
                return httpx.Response(200, json={"access_token": "access-1", "refresh_token": "refresh-1",
                    "id_token": signed_identity(private, pending["nonce"]), "expires_in": 3600,
                    "token_type": "Bearer", "scope": auth.SCOPES})
            if request.url.path.endswith("openid-configuration"):
                return httpx.Response(200, json={"issuer": auth.ISSUER, "jwks_uri": auth.ISSUER + "/.well-known/jwks.json"})
            if request.url.path.endswith("jwks.json"):
                return httpx.Response(200, json={"keys": [public]})
            pytest.fail(f"Unexpected request: {request.method} {request.url}")

        monkeypatch.setattr(auth, "_browser_callback", callback)
        monkeypatch.setattr(auth, "openai_http_client", proxied_client)
        transport = httpx.MockTransport(request)
        account = auth.login(tmp_path, proxy=proxy, transport=transport)
        auth.login(tmp_path, proxy=proxy, transport=transport)
        assert client_options == [{"proxy": proxy, "transport": transport, "timeout": 30}] * 2
        assert request_urls == [auth.TOKEN, auth.ISSUER + "/.well-known/openid-configuration",
                                auth.ISSUER + "/.well-known/jwks.json"] * 2
        assert account == {"email": "test@example.test", "subject": "subject-1", "client_id": "oaiapp_test", "scopes": auth.SCOPES.split()}
        assert "access_token" not in account
        assert attempts[0]["client_id"] == "dynamic_agent_client"
        assert attempts[0]["agent_name_hint"] == "shad-helper"
        assert attempts[1]["client_id"] == "oaiapp_test"
        assert "agent_name_hint" not in attempts[1]
        assert attempts[1]["id_token_hint"]
        assert attempts[0]["ext_agent_host_id"] == attempts[1]["ext_agent_host_id"]
        assert attempts[0]["state"] != attempts[1]["state"]
        assert attempts[0]["nonce"] != attempts[1]["nonce"]
        for parameters, form in zip(attempts, forms):
            assert form["client_id"] == ["oaiapp_test"]
            assert "client_secret" not in form
            assert form["redirect_uri"] == [parameters["redirect_uri"]]
            digest = hashlib.sha256(form["code_verifier"][0].encode()).digest()
            assert parameters["code_challenge"] == base64.urlsafe_b64encode(digest).decode().rstrip("=")
        assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
        assert stat.S_IMODE((tmp_path / "credentials.json").stat().st_mode) == 0o600


    @pytest.mark.parametrize("callback", [
        {"state": "wrong", "code": "code", "client_id": "oaiapp_test"},
        {"error": "access_denied"},
        {"code": "code"},
        {"code": "code", "client_id": "dynamic_agent_client"},
    ])
    def test_invalid_callback_never_exchanges_code(tmp_path, monkeypatch, callback):
        def receive(parameters, *args):
            return {"state": parameters["state"], **callback}

        def no_network(request):
            pytest.fail("Invalid callback must not make a token request")

        monkeypatch.setattr(auth, "_browser_callback", receive)
        with pytest.raises(auth.PlanAccessError):
            auth.login(tmp_path, transport=httpx.MockTransport(no_network))
        assert not (tmp_path / "credentials.json").exists()


    def test_loopback_callback_rejects_wrong_state_before_accepting_result(monkeypatch):
        observed = {"statuses": []}

        class Handler:
            def send_response(self, status):
                observed["statuses"].append(status)

            def send_header(self, *args):
                pass

            def end_headers(self):
                pass

        class Server:
            server_port = 54321

            def __init__(self, address, handler):
                assert address == ("127.0.0.1", 0)
                self.handler = handler

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def handle_request(self):
                parameters = parse_qs(urlsplit(observed["url"]).query)
                callback = self.handler()
                callback.path = "/auth/callback?" + urlencode({
                    "state": "wrong" if not observed["statuses"] else parameters["state"][0],
                    "code": "test-code", "client_id": "oaiapp_test"})
                callback.wfile = io.BytesIO()
                callback.do_GET()

        def opener(url):
            observed["url"] = url
            return True

        monkeypatch.setattr(auth, "BaseHTTPRequestHandler", Handler)
        monkeypatch.setattr(auth, "HTTPServer", Server)
        parameters = {"state": "expected", "client_id": "dynamic_agent_client"}
        result = auth._browser_callback(parameters, opener, 1)
        assert result == {"state": "expected", "code": "test-code", "client_id": "oaiapp_test"}
        assert observed["statuses"] == [400, 200]
        assert parameters["redirect_uri"] == "http://127.0.0.1:54321/auth/callback"


    @pytest.mark.parametrize("change", [{"audience": "another-client"}, {"issuer": "https://example.test"}, {"nonce": "wrong"}])
    def test_identity_validation_rejects_wrong_claims(identity_key, change):
        private, public = identity_key
        claims = {"nonce": "expected", **change}
        token = signed_identity(private, **claims)

        def request(request):
            return httpx.Response(200, json={"issuer": auth.ISSUER, "jwks_uri": auth.ISSUER + "/jwks"} if request.url.path.endswith("openid-configuration") else {"keys": [public]})

        with httpx.Client(transport=httpx.MockTransport(request)) as client:
            with pytest.raises(auth.PlanAccessError, match="verification"):
                auth._verify_identity(client, token, "oaiapp_test", "expected")


    def test_identity_validation_rejects_wrong_signature(identity_key):
        _, public = identity_key
        other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        token = signed_identity(other_key, "expected")

        def request(request):
            return httpx.Response(200, json={"issuer": auth.ISSUER, "jwks_uri": auth.ISSUER + "/jwks"} if request.url.path.endswith("openid-configuration") else {"keys": [public]})

        with httpx.Client(transport=httpx.MockTransport(request)) as client:
            with pytest.raises(auth.PlanAccessError, match="verification"):
                auth._verify_identity(client, token, "oaiapp_test", "expected")


    def write_credentials(tmp_path, *, expires_at=0):
        value = {"client_id": "oaiapp_test", "subject": "subject-1", "scopes": auth.SCOPES.split(),
                 "access_token": "old-access", "refresh_token": "old-refresh", "expires_at": expires_at}
        auth._save(tmp_path / "credentials.json", value)
        return value


    def test_refresh_atomically_replaces_rotating_tokens(tmp_path):
        write_credentials(tmp_path)

        def request(request):
            assert str(request.url) == auth.TOKEN
            assert parse_qs(request.content.decode()) == {"grant_type": ["refresh_token"], "client_id": ["oaiapp_test"],
                                                         "refresh_token": ["old-refresh"], "resource": [auth.RESOURCE]}
            return httpx.Response(200, json={"access_token": "new-access", "refresh_token": "new-refresh",
                                            "expires_in": 3600, "token_type": "Bearer", "scope": auth.SCOPES})

        assert auth.access_token(tmp_path, transport=httpx.MockTransport(request)) == "new-access"
        record = json.loads((tmp_path / "credentials.json").read_text())
        assert record["refresh_token"] == "new-refresh"
        assert record["expires_at"] > time.time() + 3500
        assert stat.S_IMODE((tmp_path / "credentials.json").stat().st_mode) == 0o600


    def test_valid_token_needs_no_network(tmp_path):
        write_credentials(tmp_path, expires_at=time.time() + 1000)
        assert auth.access_token(tmp_path, transport=httpx.MockTransport(lambda _: pytest.fail("Unexpected network"))) == "old-access"


    @pytest.mark.parametrize("code,cleared", [("invalid_grant", True), ("server_error", False)])
    def test_refresh_error_preserves_registration_and_only_clears_terminal_tokens(tmp_path, code, cleared):
        write_credentials(tmp_path)
        with pytest.raises(auth.PlanAccessError) as caught:
            auth.access_token(tmp_path, transport=httpx.MockTransport(lambda _: httpx.Response(400 if cleared else 503, json={"error": code})))
        assert caught.value.code == code
        record = json.loads((tmp_path / "credentials.json").read_text())
        assert record["client_id"] == "oaiapp_test"
        assert ("refresh_token" not in record) == cleared


    def test_plan_permission_required_even_with_access_token(tmp_path):
        record = write_credentials(tmp_path, expires_at=time.time() + 1000)
        record["scopes"] = ["openid"]
        auth._save(tmp_path / "credentials.json", record)
        with pytest.raises(auth.PlanAccessError, match="enable ChatGPT plan usage"):
            auth.access_token(tmp_path)


if __name__ == "__main__":
    unittest.main()
