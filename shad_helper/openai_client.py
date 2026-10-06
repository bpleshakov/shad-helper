"""Streaming Responses API through the user's authorized ChatGPT plan."""

import base64
import json
import mimetypes
from pathlib import Path

import httpx

from shad_helper.auth import RESOURCE, PlanAccessError, access_token, http_error
from shad_helper.network import openai_http_client, resolve_openai_proxy

# https://developers.openai.com/api/docs/guides/images-vision
MAX_REQUEST_BYTES = 512_000_000
MAX_IMAGES = 1_500


class ResponseError(RuntimeError):
    pass


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


def _image_payload_error():
    return ResponseError(
        "Evidence exceeds the image count or 512 MB request limit. "
        "Split the chapter or use the manual handoff; no evidence images were omitted."
    )


def _events(response):
    data = []
    for line in response.iter_lines():
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


def _stream_text(response):
    deltas = []
    for event in _events(response):
        if not isinstance(event, dict):
            raise ResponseError("OpenAI returned an invalid streaming event.")
        kind = event.get("type")
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
            if code.startswith(("subscription_sharing_", "chatpass_")) or code in {"server_error", "rate_limit_exceeded", "invalid_api_key", "authentication_error", "permission_denied"}:
                request_id = response.headers.get("x-request-id") or response.headers.get("openai-request-id")
                raise PlanAccessError(f"ChatGPT plan request failed: {code}.", code=code, request_id=request_id)
            raise ResponseError(f"OpenAI response failed: {code}.")
        elif kind == "response.incomplete":
            raise ResponseError("OpenAI response was incomplete; partial notes were discarded.")
        elif kind == "response.completed":
            result = event.get("response", {})
            if result.get("status") != "completed":
                raise ResponseError("OpenAI completion event did not confirm a completed response.")
            output = []
            for item in result.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "refusal":
                        raise ResponseError("OpenAI refused the note-generation request.")
                    if content.get("type") == "output_text":
                        output.append(content.get("text", ""))
            return "".join(output or deltas)
    raise ResponseError("OpenAI stream ended without response.completed; partial notes were discarded.")


class PlanClient:
    def __init__(self, state_dir: Path, model="gpt-6.1-sol", *, proxy=None, transport=None, token_provider=None):
        self.state_dir = Path(state_dir)
        self.model = model
        self.proxy = resolve_openai_proxy(proxy)
        self.transport = transport
        self.token_provider = token_provider or (lambda: access_token(
            self.state_dir, proxy=self.proxy, transport=self.transport))
        self._available_models = None

    def models(self) -> list[str]:
        try:
            with openai_http_client(proxy=self.proxy, transport=self.transport, timeout=30) as client:
                response = client.get(f"{RESOURCE}/models", headers={"Authorization": f"Bearer {self.token_provider()}"})
            if not response.is_success:
                raise http_error(response)
            body = response.json()
            self._available_models = [item["slug"] for item in body["models"] if item.get("visibility") == "list"]
            return self._available_models.copy()
        except httpx.HTTPError as exc:
            raise PlanAccessError("Cannot reach the ChatGPT model catalog; retry later.") from exc
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ResponseError("ChatGPT returned an invalid account-specific model catalog.") from exc

    def complete(self, instructions: str, text: str, images: list[Path] | None = None, schema: dict | None = None) -> dict:
        images = images or []
        if len(images) > MAX_IMAGES:
            raise _image_payload_error()
        content = [{"type": "input_text", "text": text}]
        body = {"model": self.model, "instructions": instructions, "input": [{"role": "user", "content": content}], "store": False, "stream": True}
        if schema is not None:
            body["text"] = {"format": {"type": "json_schema", "name": "lecture_notes", "schema": _strict_schema(schema), "strict": True}}
        estimated_size = len(_serialize(body))
        if estimated_size > MAX_REQUEST_BYTES:
            raise _image_payload_error()
        for image in images:
            image = Path(image)
            mime = mimetypes.guess_type(image.name)[0]
            if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                raise ResponseError(f"Unsupported evidence image format: {image.suffix}.")
            # Vision does not receive filenames; make the provenance mapping explicit.
            label = {"type": "input_text", "text": f"Evidence image: {image.name}"}
            item = {"type": "input_image", "image_url": f"data:{mime};base64,", "detail": "high"}
            estimated_size += len(_serialize([label, item])) - 1 + 4 * ((image.stat().st_size + 2) // 3)
            if estimated_size > MAX_REQUEST_BYTES:
                raise _image_payload_error()
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            item["image_url"] += encoded
            content.extend([label, item])
        payload = _serialize(body)
        if len(payload) > MAX_REQUEST_BYTES:
            raise _image_payload_error()
        available = self.models() if self._available_models is None else self._available_models
        if self.model not in available:
            raise PlanAccessError(f"{self.model} is unavailable for the signed-in ChatGPT account.", code="model_unavailable")
        try:
            with openai_http_client(proxy=self.proxy, transport=self.transport, timeout=httpx.Timeout(600, connect=30)) as client:
                with client.stream("POST", f"{RESOURCE}/responses", headers={"Authorization": f"Bearer {self.token_provider()}", "Content-Type": "application/json"}, content=payload) as response:
                    if not response.is_success:
                        response.read()
                        raise http_error(response)
                    try:
                        raw = _stream_text(response)
                    except (KeyError, TypeError, AttributeError) as exc:
                        raise ResponseError("OpenAI returned malformed response content.") from exc
        except httpx.HTTPError as exc:
            raise PlanAccessError("ChatGPT connection was interrupted; partial notes were discarded.") from exc
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise ResponseError("OpenAI returned invalid notes JSON.") from exc
        if not isinstance(value, dict):
            raise ResponseError("OpenAI notes output must be a JSON object.")
        return value
