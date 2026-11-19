"""Streaming Responses API, preferring an API key with ChatGPT-login fallback."""

import base64
import json
import sys
from pathlib import Path

import httpx

from shad_helper.auth import RESOURCE, PlanAccessError, access_token, http_error
from shad_helper.chapter_progress import check_cancelled, notes_connection
from shad_helper.network import openai_http_client, resolve_openai_proxy
from shad_helper.openai_config import load_api_key
from shad_helper.openai_usage import track_request

DEFAULT_MODEL = "gpt-6.1-sol"
DEFAULT_REASONING_EFFORT = "low"
MAX_PDF_BYTES = 500_000_000
MAX_IMAGE_ENCODED_BYTES = 400_000_000
CREDIT_EXHAUSTED_CODES = {"credit_balance_exhausted", "insufficient_quota"}


def _image_data(images):
    """Validate bounded PNG attachments before accessing credentials or the network."""
    result = []
    encoded_size = 0
    for label, path in images:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Image attachments require a non-empty label.")
        with Path(path).open("rb") as stream:
            data = stream.read(MAX_IMAGE_ENCODED_BYTES * 3 // 4 + 1)
        encoded_size += 4 * ((len(data) + 2) // 3)
        if encoded_size > MAX_IMAGE_ENCODED_BYTES:
            raise ValueError("Encoded image attachments must not exceed 40 MB.")
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Image attachments must be valid PNG files.")
        import pymupdf
        try:
            pixels = pymupdf.Pixmap(data)
            if pixels.width <= 0 or pixels.height <= 0:
                raise ValueError("Empty PNG")
        except (RuntimeError, ValueError, pymupdf.mupdf.FzErrorBase) as error:
            raise ValueError("Image attachments must be valid PNG files.") from error
        result.append((label, data))
    return result, encoded_size


def image_attachments_size(images: list[tuple[str, Path]]) -> int:
    """Return base64 bytes after validating the request's attachment budget."""
    return _image_data(images)[1]

class ResponseError(RuntimeError):
    def __init__(self, message, *, code=None):
        super().__init__(message)
        self.code = code


def _strict_schema(value):
    """Normalize Pydantic defaults to the Responses strict-schema contract."""
    if isinstance(value, list):
        return [_strict_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key == "default":
            continue
        if key in {"properties", "$defs", "definitions"} and isinstance(item, dict):
            result[key] = {name: _strict_schema(schema) for name, schema in item.items()}
        else:
            result[key] = _strict_schema(item)
    if result.get("type") == "object" or "properties" in result:
        result["required"] = list(result.get("properties", {}))
        result["additionalProperties"] = False
    return result


def _serialize(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _events(response):
    data = []
    for line in response.iter_lines():
        check_cancelled()
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line and data:
            raw = "\n".join(data)
            data = []
            if raw != "[DONE]":
                try:
                    yield json.loads(raw)
                except ValueError as exc:
                    raise ResponseError("OpenAI returned a malformed streaming event.") from exc
    if data:
        raw = "\n".join(data)
        if raw != "[DONE]":
            try:
                yield json.loads(raw)
            except ValueError as exc:
                raise ResponseError("OpenAI returned a malformed streaming event.") from exc


def _stream_text(response, *, api_key=False, usage_recorder=None):
    deltas = []
    for event in _events(response):
        if not isinstance(event, dict):
            raise ResponseError("OpenAI returned an invalid streaming event.")
        kind = event.get("type")
        if usage_recorder is not None and kind in {
                "response.completed", "response.failed", "response.incomplete", "error"}:
            usage_recorder(event.get("response"))
        if kind == "response.output_text.delta":
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise ResponseError("OpenAI returned an invalid text delta.")
            deltas.append(delta)
        elif kind in {"error", "response.failed"}:
            error = event.get("error") or event.get("response", {}).get("error") or event
            if not isinstance(error, dict) or not isinstance(error.get("code", "unknown_error"), str):
                raise ResponseError("OpenAI returned an invalid failure event.")
            code = error.get("code", "unknown_error")
            if code.startswith(("subscription_sharing_", "chatpass_")) or code in CREDIT_EXHAUSTED_CODES | {"server_error", "rate_limit_exceeded", "invalid_api_key", "authentication_error", "permission_denied"}:
                request_id = response.headers.get("x-request-id") or response.headers.get("openai-request-id")
                source = "OpenAI API" if api_key else "ChatGPT plan"
                raise PlanAccessError(f"{source} request failed: {code}.", code=code, request_id=request_id)
            raise ResponseError(f"OpenAI response failed: {code}.", code=code)
        elif kind == "response.incomplete":
            raise ResponseError("OpenAI response was incomplete; partial output was discarded.")
        elif kind == "response.completed":
            result = event.get("response", {})
            if result.get("status") != "completed":
                raise ResponseError("OpenAI completion event did not confirm a completed response.")
            output = []
            for item in result.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "refusal":
                        raise ResponseError("OpenAI refused the generation request.")
                    if content.get("type") == "output_text":
                        output.append(content.get("text", ""))
            return "".join(output or deltas)
    raise ResponseError("OpenAI stream ended without response.completed; partial output was discarded.")


class PlanClient:
    def __init__(self, state_dir: Path, model=DEFAULT_MODEL, *, proxy=None, transport=None, token_provider=None, request_recorder=None, api_key=None):
        self.state_dir = Path(state_dir)
        self.model = model
        self.proxy = resolve_openai_proxy(proxy)
        self.transport = transport
        self._api_key = load_api_key() if api_key is None else api_key.strip() or None
        self.token_provider = token_provider or (lambda: access_token(
            self.state_dir, proxy=self.proxy, transport=self.transport))
        self.request_recorder = request_recorder
        self._available_models = None

    def _fallback_to_login(self, error):
        if (not self._api_key or not isinstance(error.code, str)
                or error.code not in CREDIT_EXHAUSTED_CODES):
            return False
        self._api_key = None
        self._available_models = None
        print("OpenAI API credits exhausted; switching to saved ChatGPT login.", file=sys.stderr)
        return True

    def models(self, *, allow_fallback: bool = True) -> list[str]:
        """List models; disable fallback to check the current credentials independently."""
        try:
            return self._models()
        except PlanAccessError as exc:
            if not allow_fallback or not self._fallback_to_login(exc):
                raise
            return self._models()

    def _models(self) -> list[str]:
        url = f"{RESOURCE}/models"
        source = "OpenAI API" if self._api_key else "ChatGPT"
        headers = {"Authorization": f"Bearer {self._api_key or self.token_provider()}"}
        if self.request_recorder is not None:
            self.request_recorder("GET", url, None)
        try:
            with (openai_http_client(proxy=self.proxy, transport=self.transport, timeout=30) as client,
                  notes_connection(client)):
                response = client.get(url, headers=headers)
            if not response.is_success:
                raise http_error(response, api_key=bool(self._api_key))
            body = response.json()
            if self._api_key:
                self._available_models = [item["id"] for item in body["data"]]
            else:
                self._available_models = [item["slug"] for item in body["models"] if item.get("visibility") == "list"]
            return self._available_models.copy()
        except httpx.HTTPError as exc:
            raise PlanAccessError(f"Cannot reach the {source} model catalog; retry later.") from exc
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ResponseError(f"{source} returned an invalid account-specific model catalog.") from exc

    def complete(self, instructions: str, text: str, schema: dict | None = None,
                 *, pdf: Path | None = None,
                 images: list[tuple[str, Path]] | None = None) -> dict:
        content = [{"type": "input_text", "text": text}]
        if pdf is not None:
            pdf = Path(pdf)
            with pdf.open("rb") as stream:
                data = stream.read(MAX_PDF_BYTES)
            if len(data) >= MAX_PDF_BYTES:
                raise ValueError("PDF input must be smaller than 50 MB.")
            if not data.startswith(b"%PDF-"):
                raise ValueError("PDF input has an invalid PDF signature.")
            content.append({"type": "input_file", "filename": pdf.name,
                            "file_data": "data:application/pdf;base64,"
                            + base64.b64encode(data).decode("ascii")})
        for label, data in _image_data(images or [])[0]:
            content.extend((
                {"type": "input_text", "text": label},
                {"type": "input_image", "detail": "high", "image_url":
                 "data:image/png;base64," + base64.b64encode(data).decode("ascii")},
            ))
        body = {"model": self.model, "reasoning": {"effort": DEFAULT_REASONING_EFFORT}, "instructions": instructions, "input": [{"role": "user", "content": content}], "store": False, "stream": True}
        if schema is not None:
            body["text"] = {"format": {"type": "json_schema", "name": "lecture_notes", "schema": _strict_schema(schema), "strict": True}}
        payload = _serialize(body)
        try:
            return self._complete(payload)
        except PlanAccessError as exc:
            if not self._fallback_to_login(exc):
                raise
            return self._complete(payload)

    def _complete(self, payload):
        available = self.models() if self._available_models is None else self._available_models
        if self.model not in available:
            account = "configured OpenAI API key" if self._api_key else "signed-in ChatGPT account"
            raise PlanAccessError(f"{self.model} is unavailable for the {account}.", code="model_unavailable")
        url = f"{RESOURCE}/responses"
        headers = {"Authorization": f"Bearer {self._api_key or self.token_provider()}", "Content-Type": "application/json"}
        if self.request_recorder is not None:
            self.request_recorder("POST", url, payload)
        try:
            with (
                openai_http_client(proxy=self.proxy, transport=self.transport,
                                   timeout=httpx.Timeout(600, connect=30)) as client,
                notes_connection(client),
                track_request(api_key=bool(self._api_key), model=self.model) as usage_recorder,
                client.stream("POST", url, headers=headers, content=payload) as response,
            ):
                if not response.is_success:
                    if (usage_recorder is not None and 400 <= response.status_code < 500
                            and response.status_code != 408):
                        usage_recorder(rejected=True)
                    response.read()
                    raise http_error(response, api_key=bool(self._api_key))
                try:
                    raw = _stream_text(response, api_key=bool(self._api_key),
                                       usage_recorder=usage_recorder)
                except (KeyError, TypeError, AttributeError) as exc:
                    raise ResponseError("OpenAI returned malformed response content.") from exc
        except httpx.HTTPError as exc:
            source = "OpenAI API" if self._api_key else "ChatGPT"
            raise PlanAccessError(f"{source} connection was interrupted; partial output was discarded.") from exc
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise ResponseError("OpenAI returned invalid JSON.") from exc
        if not isinstance(value, dict):
            raise ResponseError("OpenAI output must be a JSON object.")
        return value
