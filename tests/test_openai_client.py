"""Offline unittest checks plus full mocked HTTP tests when dependencies exist."""

import ast
import base64
import importlib.util
import json
import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def _completed(text="{}"):
    return {"type": "response.completed", "response": {"status": "completed", "output": [
        {"content": [{"type": "output_text", "text": text}]}]}}


class _Response:
    is_success = True
    headers = {"x-request-id": "offline-request"}

    def __init__(self, events=None, raw=None, catalog=None):
        self.events = events if events is not None else [_completed()]
        self.raw = raw
        self.catalog = catalog

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_lines(self):
        if self.raw is not None:
            return iter(self.raw.splitlines())
        return iter(line for event in self.events for line in ("data: " + json.dumps(event), ""))

    def json(self):
        return self.catalog


def _client_module():
    """Load actual pure client logic without importing unavailable packages."""
    module = ModuleType("openai_client_offline_checks")
    calls = []
    module.client_options = []
    module.token_requests = []
    module.proxy_resolutions = []
    module.network_error = None

    class Client:
        def __init__(self, **kwargs):
            module.client_options.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, **kwargs):
            calls.append(("GET", url, kwargs))
            if module.network_error is not None:
                raise module.network_error
            return _Response(catalog={"models": [{"slug": "gpt-6-astra", "visibility": "list"},
                {"slug": "hidden", "visibility": "hide"}]})

        def stream(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            if module.network_error is not None:
                raise module.network_error
            return _Response(events=module.response_events)

    def resolve_proxy(proxy=None):
        module.proxy_resolutions.append(proxy)
        return proxy if proxy is not None else "http://offline-proxy:8080"

    def token_provider(state_dir, **kwargs):
        module.token_requests.append((state_dir, kwargs))
        return "offline-token"

    auth_path = Path(__file__).parents[1] / "shad_helper" / "auth.py"
    auth_tree = ast.parse(auth_path.read_text())
    auth_class = next(node for node in auth_tree.body if isinstance(node, ast.ClassDef) and node.name == "PlanAccessError")
    exec(compile(ast.Module(body=[auth_class], type_ignores=[]), str(auth_path), "exec"), module.__dict__)
    module.__dict__.update(base64=base64, json=json, sys=sys, Path=Path,
        RESOURCE="https://api.openai.com/v1", access_token=token_provider,
        resolve_openai_proxy=resolve_proxy, openai_http_client=Client,
        http_error=lambda _, **kwargs: module.PlanAccessError("offline HTTP error"),
        load_api_key=lambda: None,
        track_request=lambda **kwargs: nullcontext(None),
        notes_connection=lambda _: nullcontext(), check_cancelled=lambda: None,
        httpx=SimpleNamespace(Client=Client, Timeout=lambda *args, **kwargs: None, HTTPError=OSError))
    source_path = Path(__file__).parents[1] / "shad_helper" / "openai_client.py"
    source_tree = ast.parse(source_path.read_text())
    source_tree.body = [node for node in source_tree.body if isinstance(node, (ast.Assign, ast.FunctionDef, ast.ClassDef))]
    exec(compile(source_tree, str(source_path), "exec"), module.__dict__)
    module.response_events = [_completed()]
    return module, calls


class ClientStdlibTests(unittest.TestCase):
    def test_explicit_model_override_is_preserved(self):
        module, calls = _client_module()
        client = module.PlanClient(Path("auth-state"), model="custom-model")
        client._available_models = ["custom-model"]

        self.assertEqual(client.complete("Instructions", "Text"), {})

        self.assertEqual(len(calls), 1)
        body = json.loads(calls[0][2]["content"])
        self.assertEqual(body["model"], "custom-model")
        self.assertEqual(body["reasoning"], {"effort": "max"})

    def test_request_recorder_receives_the_sent_body_without_credentials(self):
        module, calls = _client_module()
        records = []

        def record(method, url, payload):
            self.assertEqual(len(calls), len(records))
            self.assertEqual(len(module.token_requests), len(records) + 1)
            records.append((method, url, payload))

        client = module.PlanClient(Path("auth-state"), request_recorder=record)
        self.assertEqual(client.complete("Instructions", "Лекция"), {})
        self.assertEqual(records, [
            ("GET", "https://api.openai.com/v1/models", None),
            ("POST", "https://api.openai.com/v1/responses", calls[-1][2]["content"])])
        self.assertNotIn("offline-token", repr(records))
        self.assertNotIn("offline-proxy", repr(records))

    def test_same_proxy_reaches_catalog_stream_and_token_refresh(self):
        module, calls = _client_module()
        transport = object()
        proxy = "http://lecture-proxy:8080"
        with patch.dict(os.environ, {"NO_PROXY": "*", "HTTPS_PROXY": "http://other-proxy:8080"}):
            client = module.PlanClient(Path("auth-state"), proxy=proxy, transport=transport)
            self.assertEqual(client.complete("Instructions", "Text"), {})
        self.assertEqual(module.proxy_resolutions, [proxy])
        self.assertEqual([item[0] for item in calls], ["GET", "POST"])
        self.assertEqual(len(module.client_options), 2)
        for options in module.client_options:
            self.assertEqual(options["proxy"], proxy)
            self.assertIs(options["transport"], transport)
        self.assertEqual(module.token_requests, [
            (Path("auth-state"), {"proxy": proxy, "transport": transport}),
            (Path("auth-state"), {"proxy": proxy, "transport": transport})])

    def test_invalid_configuration_blocks_client_before_network(self):
        module, calls = _client_module()

        def missing_proxy(_):
            raise ValueError("An OpenAI proxy is required.")

        module.resolve_openai_proxy = missing_proxy
        with self.assertRaisesRegex(ValueError, "proxy is required"):
            module.PlanClient(Path("auth-state"))
        self.assertEqual(module.client_options, [])
        self.assertEqual(module.token_requests, [])
        self.assertEqual(calls, [])

    def test_failed_proxy_has_no_direct_retry(self):
        module, calls = _client_module()
        proxy = "http://lecture-proxy:8080"
        client = module.PlanClient(Path("auth-state"), proxy=proxy)
        module.network_error = OSError("proxy unreachable")
        with self.assertRaisesRegex(module.PlanAccessError, "Cannot reach"):
            client.models()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(module.client_options), 1)
        self.assertEqual(module.client_options[0]["proxy"], proxy)
        client._available_models = ["gpt-6-astra"]
        with self.assertRaisesRegex(module.PlanAccessError, "connection was interrupted"):
            client.complete("Instructions", "Text")
        self.assertEqual([item[0] for item in calls], ["GET", "POST"])
        self.assertEqual(len(module.client_options), 2)
        self.assertTrue(all(item["proxy"] == proxy for item in module.client_options))

    def test_strict_schema_requires_nested_properties_and_preserves_source(self):
        module, _ = _client_module()
        schema = {"type": "object", "properties": {"default": {"type": "string", "default": "x"},
            "sections": {"type": "array", "items": {"$ref": "#/$defs/Section"}, "default": []}},
            "$defs": {"Section": {"type": "object", "properties": {
                "body": {"type": "string"}, "figure": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}}}}}
        original = json.loads(json.dumps(schema))
        strict = module._strict_schema(schema)
        self.assertEqual(schema, original)
        self.assertEqual(strict["required"], ["default", "sections"])
        self.assertFalse(strict["additionalProperties"])
        self.assertEqual(strict["properties"]["default"], {"type": "string"})
        self.assertEqual(strict["$defs"]["Section"]["required"], ["body", "figure"])
        self.assertFalse(strict["$defs"]["Section"]["additionalProperties"])
        self.assertNotIn("default", strict["properties"]["sections"])
        self.assertNotIn("default", strict["$defs"]["Section"]["properties"]["figure"])

    def test_stream_requires_successful_terminal_event(self):
        module, _ = _client_module()
        events = [{"type": "response.output_text.delta", "delta": "discarded"}, _completed('{"title":"Лекция"}')]
        self.assertEqual(module._stream_text(_Response(events)), '{"title":"Лекция"}')
        failures = [
            ([{"type": "response.output_text.delta", "delta": "{}"}], "without response.completed"),
            ([{"type": "response.incomplete"}], "incomplete"),
            ([{"type": "response.completed", "response": {"status": "incomplete"}}], "did not confirm"),
            ([{"type": "response.failed", "response": {"error": {"code": "bad_output"}}}], "bad_output"),
            ([{"type": "response.completed", "response": {"status": "completed", "output": [
                {"content": [{"type": "refusal"}]}]}}], "refused"),
        ]
        for events, message in failures:
            with self.subTest(message=message), self.assertRaisesRegex(module.ResponseError, message):
                module._stream_text(_Response(events))
        for raw, message in [("data: [DONE]\n\n", "without response.completed"), ("data: malformed\n\n", "malformed")]:
            with self.subTest(raw=raw), self.assertRaisesRegex(module.ResponseError, message):
                module._stream_text(_Response(raw=raw))

    def test_stream_plan_errors_preserve_code_and_request_id(self):
        module, _ = _client_module()
        code = "subscription_sharing_usage_limit_exceeded"
        events = [{"type": "response.output_text.delta", "delta": "{}"},
            {"type": "response.failed", "response": {"error": {"code": code}}}]
        with self.assertRaises(module.PlanAccessError) as caught:
            module._stream_text(_Response(events))
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.request_id, "offline-request")


    def test_unavailable_model_and_invalid_json_never_return_notes(self):
        module, calls = _client_module()
        client = module.PlanClient(Path("."), token_provider=lambda: "offline-token")
        client._available_models = ["unavailable-alternative"]
        with self.assertRaisesRegex(module.PlanAccessError, "unavailable"):
            client.complete("Instructions", "Text")
        self.assertEqual(calls, [])
        client._available_models = ["gpt-6-astra"]
        for output, message in [("not JSON", "invalid JSON"), ("[]", "JSON object")]:
            module.response_events = [_completed(output)]
            with self.subTest(output=output), self.assertRaisesRegex(module.ResponseError, message):
                client.complete("Instructions", "Text")


_FULL_DEPS = all(importlib.util.find_spec(name) for name in ("httpx", "jwt", "cryptography", "pytest"))
if _FULL_DEPS:
    import json

    import httpx
    import pytest

    from shad_helper import openai_client
    from shad_helper.auth import PlanAccessError
    from shad_helper.openai_client import PlanClient, ResponseError, _strict_schema


    @pytest.fixture(autouse=True)
    def configured_proxy(monkeypatch):
        monkeypatch.setenv("SHAD_HELPER_OPENAI_PROXY", "http://offline-proxy:8080")


    def sse(*events):
        return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


    def completed(text):
        return {"type": "response.completed", "response": {"status": "completed", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": text}]}]}}


    def client_for(tmp_path, stream, *, models=None, status=200):
        requests = []

        def handle(request):
            requests.append(request)
            assert request.headers["Authorization"] == "Bearer test-plan-token"
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"models": models if models is not None else [
                    {"slug": "gpt-6-astra", "visibility": "list"}, {"slug": "hidden", "visibility": "hide"}]})
            assert str(request.url) == "https://api.openai.com/v1/responses"
            return httpx.Response(status, text=stream, headers={"content-type": "text/event-stream"})

        return PlanClient(tmp_path, transport=httpx.MockTransport(handle), token_provider=lambda: "test-plan-token"), requests


    def test_proxy_configuration_reaches_catalog_responses_and_token_refresh(tmp_path, monkeypatch):
        existing_client, requests = client_for(tmp_path, sse(completed("{}")))
        proxy = "http://lecture-proxy:8080"
        monkeypatch.setenv("NO_PROXY", "*")
        monkeypatch.setenv("HTTPS_PROXY", "http://other-proxy:8080")
        client_options = []
        token_requests = []
        original_factory = openai_client.openai_http_client

        def factory(**kwargs):
            client_options.append(kwargs)
            return original_factory(**kwargs)

        def token(state_dir, **kwargs):
            token_requests.append((state_dir, kwargs))
            return "test-plan-token"

        monkeypatch.setattr(openai_client, "openai_http_client", factory)
        monkeypatch.setattr(openai_client, "access_token", token)
        client = PlanClient(tmp_path, proxy=proxy, transport=existing_client.transport)
        assert client.complete("Instructions", "Text") == {}
        assert [request.url.path for request in requests] == ["/v1/models", "/v1/responses"]
        assert len(client_options) == 2
        assert all(options["proxy"] == proxy for options in client_options)
        assert all(options["transport"] is existing_client.transport for options in client_options)
        assert client_options[0]["timeout"] == 30
        assert client_options[1]["timeout"].connect == 30
        assert client_options[1]["timeout"].read == 600
        assert token_requests == [
            (tmp_path, {"proxy": proxy, "transport": existing_client.transport}),
            (tmp_path, {"proxy": proxy, "transport": existing_client.transport})]


    def test_missing_proxy_configuration_blocks_before_token_or_network(tmp_path, monkeypatch):
        monkeypatch.delenv("SHAD_HELPER_OPENAI_PROXY")
        monkeypatch.setenv("HTTPS_PROXY", "http://unrelated-proxy:8080")

        def unexpected_network(_):
            pytest.fail("Network must not be used without an OpenAI proxy.")

        with pytest.raises(ValueError, match="proxy"):
            PlanClient(tmp_path, transport=httpx.MockTransport(unexpected_network), token_provider=lambda: pytest.fail("Unexpected token access"))


    def test_plan_request_contract_is_text_only(tmp_path):
        schema = {"type": "object", "properties": {"notes": {"type": "string"}}, "required": ["notes"], "additionalProperties": False}
        client, requests = client_for(tmp_path, sse({"type": "response.output_text.delta", "delta": '{"notes":'}, completed('{"notes":"Привет"}')))
        assert client.complete("Return notes JSON.", "Lecture text.", schema) == {"notes": "Привет"}
        body = json.loads(requests[-1].content)
        assert body["model"] == "gpt-6-astra"
        assert body["reasoning"] == {"effort": "max"}
        assert body["store"] is False and body["stream"] is True
        assert body["instructions"] == "Return notes JSON."
        assert isinstance(body["input"], list)
        assert body["input"][0]["role"] == "user"
        assert body["input"][0]["content"] == [{"type": "input_text", "text": "Lecture text."}]
        assert body["text"]["format"]["schema"] == schema
        assert not set(body) & {"tools", "max_output_tokens", "previous_response_id", "background", "temperature"}
        assert client.models() == ["gpt-6-astra"]


    @pytest.mark.parametrize("size", [12, 63])
    def test_pdf_request_contains_exact_bytes_filename_and_strict_schema(
            tmp_path, monkeypatch, size):
        monkeypatch.setattr(openai_client, "MAX_PDF_BYTES", 64)
        data = b"%PDF-1.7\n" + b"\x00\xff" + b"x" * (size - 11)
        pdf = tmp_path / "Конспект.pdf"
        pdf.write_bytes(data)
        schema = {"type": "object", "properties": {"cards": {
            "type": "array", "items": {"type": "string"}, "default": []}}}
        client, requests = client_for(tmp_path, sse(completed('{"cards":[]}')))
        records = []
        client.request_recorder = lambda *args: records.append(args)

        assert client.complete("Create cards.", "PDF order.", schema, pdf=pdf) == {"cards": []}

        expected = {"model": "gpt-6-astra", "reasoning": {"effort": "max"},
            "instructions": "Create cards.", "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "PDF order."},
                {"type": "input_file", "filename": pdf.name,
                 "file_data": "data:application/pdf;base64,"
                 + base64.b64encode(data).decode("ascii")}]}],
            "store": False, "stream": True, "text": {"format": {"type": "json_schema",
                "name": "lecture_notes", "schema": _strict_schema(schema), "strict": True}}}
        assert requests[-1].content == json.dumps(
            expected, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        assert records[-1] == ("POST", str(requests[-1].url), requests[-1].content)
        assert str(tmp_path).encode() not in requests[-1].content
        assert [request.url.path for request in requests] == ["/v1/models", "/v1/responses"]


    @pytest.mark.parametrize("kind", [
        "missing", "directory", "unreadable", "empty", "invalid", "at_limit", "over_limit",
    ])
    def test_invalid_pdf_fails_before_tokens_recording_or_network(tmp_path, monkeypatch, kind):
        monkeypatch.setattr(openai_client, "MAX_PDF_BYTES", 64)
        pdf = tmp_path / "notes.pdf"
        error = OSError if kind in {"missing", "directory", "unreadable"} else ValueError
        if kind == "directory":
            pdf.mkdir()
        elif kind != "missing":
            data = {"empty": b"", "invalid": b"not a PDF", "unreadable": b"%PDF-1.7\n",
                    "at_limit": b"%PDF-" + b"x" * 59,
                    "over_limit": b"%PDF-" + b"x" * 60}[kind]
            pdf.write_bytes(data)
        if kind == "unreadable":
            original_open = Path.open

            def unreadable(path, *args, **kwargs):
                if path == pdf:
                    raise PermissionError("PDF cannot be read")
                return original_open(path, *args, **kwargs)

            monkeypatch.setattr(Path, "open", unreadable)
        client, requests = client_for(tmp_path, sse(completed("{}")))
        client.token_provider = lambda: pytest.fail("Invalid PDF must not access credentials")
        client.request_recorder = lambda *args: pytest.fail("Invalid PDF must not record requests")

        with pytest.raises(error):
            client.complete("Create cards.", "PDF order.", pdf=pdf)

        assert requests == []


    def test_recorder_runs_before_transport_with_exact_request_bytes(tmp_path):
        events = []
        records = []
        schema = {"type": "object", "properties": {"notes": {"type": "string", "default": ""}}}

        def record(method, url, payload):
            events.append(("record", method))
            records.append((method, url, payload))

        def token():
            events.append(("token", None))
            return "private-plan-token"

        def handle(request):
            assert records[-1] == (request.method, str(request.url), request.content or None)
            assert request.headers["Authorization"] == "Bearer private-plan-token"
            events.append(("transport", request.method))
            if request.method == "GET":
                return httpx.Response(200, json={"models": [{"slug": "gpt-6-astra", "visibility": "list"}]})
            return httpx.Response(200, text=sse(completed("{}")))

        client = PlanClient(tmp_path, proxy="http://private-proxy:8080", transport=httpx.MockTransport(handle),
            token_provider=token, request_recorder=record)
        assert client.complete("Return notes JSON.", "Лекция.", schema) == {}
        assert events == [("token", None), ("record", "GET"), ("transport", "GET"),
            ("token", None), ("record", "POST"), ("transport", "POST")]
        body = json.loads(records[-1][2])
        assert body == {"model": "gpt-6-astra", "reasoning": {"effort": "max"},
            "instructions": "Return notes JSON.",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Лекция."}]}],
            "store": False, "stream": True, "text": {"format": {"type": "json_schema",
                "name": "lecture_notes", "schema": _strict_schema(schema), "strict": True}}}
        assert "private-plan-token" not in repr(records)
        assert "private-proxy" not in repr(records)


    @pytest.mark.parametrize("method", ["GET", "POST"])
    @pytest.mark.parametrize("error", [OSError("disk full"), ValueError("invalid archive"), httpx.HTTPError("recorder failed")])
    def test_recorder_failure_blocks_unrecorded_network_request(tmp_path, method, error):
        client, requests = client_for(tmp_path, sse(completed("{}")))
        if method == "POST":
            client._available_models = ["gpt-6-astra"]

        def fail_record(recorded_method, url, payload):
            assert recorded_method == method
            raise error

        client.request_recorder = fail_record
        with pytest.raises(type(error)) as caught:
            client.complete("Instructions", "Lecture text")
        assert caught.value is error
        assert requests == []


    @pytest.mark.parametrize("method", ["GET", "POST"])
    def test_failed_http_requests_are_recorded(tmp_path, method):
        records = []

        def fail(request):
            assert records[-1] == (request.method, str(request.url), request.content or None)
            raise httpx.ConnectError("Offline", request=request)

        client = PlanClient(tmp_path, transport=httpx.MockTransport(fail), token_provider=lambda: "token",
            request_recorder=lambda *args: records.append(args))
        if method == "POST":
            client._available_models = ["gpt-6-astra"]
        with pytest.raises(PlanAccessError):
            client.complete("Instructions", "Lecture text")
        assert len(records) == 1
        assert records[0][0] == method


    def test_strict_schema_recursively_requires_properties_without_mutating_input():
        schema = {"type": "object", "properties": {"title": {"type": "string", "default": ""},
            "sections": {"type": "array", "items": {"$ref": "#/$defs/Section"}, "default": []}},
            "required": ["title"], "$defs": {"Section": {"type": "object", "properties": {
                "body": {"type": "string"}, "figure": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}},
                "required": ["body"]}}}
        original = json.loads(json.dumps(schema))
        strict = _strict_schema(schema)
        assert schema == original
        assert strict["required"] == ["title", "sections"]
        assert strict["additionalProperties"] is False
        assert strict["$defs"]["Section"]["required"] == ["body", "figure"]
        assert strict["$defs"]["Section"]["additionalProperties"] is False
        assert "default" not in json.dumps(strict)
        assert strict["$defs"]["Section"]["properties"]["figure"]["anyOf"][-1] == {"type": "null"}


    def test_strict_schema_preserves_properties_named_default():
        schema = {"type": "object", "properties": {"default": {"type": "string", "default": "value"}}}
        assert _strict_schema(schema)["properties"] == {"default": {"type": "string"}}


    def test_unavailable_account_model_does_not_call_inference(tmp_path):
        client, requests = client_for(tmp_path, "", models=[{"slug": "unavailable-alternative", "visibility": "list"}])
        with pytest.raises(PlanAccessError, match="unavailable"):
            client.complete("Instructions", "Lecture text")
        assert len(requests) == 1


    @pytest.mark.parametrize("events, error, message", [
        ([{"type": "response.output_text.delta", "delta": '{"partial":true}'}], ResponseError, "without response.completed"),
        ([{"type": "response.output_text.delta", "delta": '{}'}, {"type": "response.incomplete"}], ResponseError, "incomplete"),
        ([{"type": "response.output_text.delta", "delta": '{}'}, {"type": "response.failed", "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded"}}}], PlanAccessError, "usage_limit"),
        ([{"type": "error", "code": "subscription_sharing_user_not_eligible"}], PlanAccessError, "not_eligible"),
        ([{"type": "response.failed", "response": {"error": {"code": "bad_output"}}}], ResponseError, "bad_output"),
        ([{"type": "response.completed", "response": {"status": "incomplete"}}], ResponseError, "did not confirm"),
        ([completed("not JSON")], ResponseError, "invalid JSON"),
        ([completed("[]")], ResponseError, "JSON object"),
    ])
    def test_stream_failures_never_return_partial_notes(tmp_path, events, error, message):
        client, _ = client_for(tmp_path, sse(*events))
        with pytest.raises(error, match=message):
            client.complete("Instructions", "Lecture text")


    @pytest.mark.parametrize("event", [
        {"type": "error", "code": "context_length_exceeded"},
        {"type": "response.failed", "response": {
            "error": {"code": "context_length_exceeded"}}},
    ])
    def test_stream_context_limit_preserves_code_for_chapter_retry(tmp_path, event):
        partial = {"type": "response.output_text.delta", "delta": '{"partial":true}'}
        client, requests = client_for(tmp_path, sse(partial, event))
        with pytest.raises(ResponseError, match="context_length_exceeded") as caught:
            client.complete("Instructions", "Lecture text")
        assert caught.value.code == "context_length_exceeded"
        assert [request.url.path for request in requests] == ["/v1/models", "/v1/responses"]


    def test_done_marker_is_not_success(tmp_path):
        client, _ = client_for(tmp_path, "data: [DONE]\n\n")
        with pytest.raises(ResponseError, match="without response.completed"):
            client.complete("Instructions", "Lecture text")


    def test_malformed_event_rejected(tmp_path):
        client, _ = client_for(tmp_path, "data: malformed\n\n")
        with pytest.raises(ResponseError, match="malformed"):
            client.complete("Instructions", "Lecture text")


    def test_refusal_rejected(tmp_path):
        event = {"type": "response.completed", "response": {"status": "completed", "output": [{"content": [{"type": "refusal", "refusal": "Declined"}]}]}}
        client, _ = client_for(tmp_path, sse(event))
        with pytest.raises(ResponseError, match="refused"):
            client.complete("Instructions", "Lecture text")


    def test_http_eligibility_error_retains_code_and_request_id(tmp_path):
        def handle(request):
            if request.url.path.endswith("models"):
                return httpx.Response(200, json={"models": [{"slug": "gpt-6-astra", "visibility": "list"}]})
            return httpx.Response(403, json={"error": {"code": "subscription_sharing_user_not_eligible"}}, headers={"x-request-id": "request-1"})

        client = PlanClient(tmp_path, transport=httpx.MockTransport(handle), token_provider=lambda: "token")
        with pytest.raises(PlanAccessError) as caught:
            client.complete("Instructions", "Lecture text")
        assert caught.value.code == "subscription_sharing_user_not_eligible"
        assert caught.value.status == 403
        assert caught.value.request_id == "request-1"


    def test_transport_error_has_no_billing_fallback(tmp_path):
        requests = []

        def fail(request):
            requests.append(request)
            raise httpx.ConnectError("Offline", request=request)

        client = PlanClient(tmp_path, transport=httpx.MockTransport(fail), token_provider=lambda: "token")
        with pytest.raises(PlanAccessError, match="Cannot reach"):
            client.complete("Instructions", "Lecture text")
        assert len(requests) == 1
        assert requests[0].url.path == "/v1/models"


    def png_attachment(tmp_path, name="drawing.png"):
        import pymupdf

        path = tmp_path / name
        pixels = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 12, 8), False)
        pixels.clear_with(220)
        pixels.save(path)
        return path


    def test_labeled_images_preserve_exact_png_bytes_and_high_detail(tmp_path):
        image = png_attachment(tmp_path)
        client, requests = client_for(tmp_path, sse(completed('{}')))
        assert client.complete("Compare", "Lecture", images=[("Рисунок ill01", image)]) == {}
        body = json.loads(requests[-1].content)
        assert body["input"][0]["content"] == [
            {"type": "input_text", "text": "Lecture"},
            {"type": "input_text", "text": "Рисунок ill01"},
            {"type": "input_image", "detail": "high", "image_url":
             "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode("ascii")},
        ]
        assert str(tmp_path) not in requests[-1].content.decode()


    def test_images_and_pdf_preserve_attachment_order(tmp_path):
        image = png_attachment(tmp_path)
        pdf = tmp_path / "notes.pdf"
        pdf.write_bytes(b"%PDF-1.7\n")
        client, requests = client_for(tmp_path, sse(completed('{}')))
        client.complete("Compare", "Lecture", pdf=pdf, images=[("ill01", image)])
        content = json.loads(requests[-1].content)["input"][0]["content"]
        assert [item["type"] for item in content] == [
            "input_text", "input_file", "input_text", "input_image"]


    def test_more_than_24_images_reach_request_without_truncation(tmp_path):
        image = png_attachment(tmp_path)
        images = [(f"ill{number:03d}", image) for number in range(100)]
        client, requests = client_for(tmp_path, sse(completed('{}')))
        assert client.complete("Compare", "Lecture", images=images) == {}
        content = json.loads(requests[-1].content)["input"][0]["content"]
        assert [item["text"] for item in content[1::2]] == [label for label, _ in images]
        assert [item["image_url"] for item in content[2::2]] == [
            "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode("ascii")
        ] * len(images)


    @pytest.mark.parametrize("kind", ["missing", "directory", "empty", "wrong_format",
                                      "malformed", "truncated", "blank_label", "bytes"])
    def test_invalid_image_fails_before_credentials_or_network(tmp_path, monkeypatch, kind):
        image = png_attachment(tmp_path)
        label = "ill01"
        if kind == "missing":
            image.unlink()
        elif kind == "directory":
            image.unlink()
            image.mkdir()
        elif kind in ("empty", "wrong_format", "malformed", "truncated"):
            image.write_bytes({"empty": b"", "wrong_format": b"%PDF-1.7",
                               "malformed": b"\x89PNG\r\n\x1a\nnot png",
                               "truncated": image.read_bytes()[:30]}[kind])
        elif kind == "blank_label":
            label = " "
        images = [(label, image)]
        if kind == "bytes":
            monkeypatch.setattr(openai_client, "MAX_IMAGE_ENCODED_BYTES", 12)
        client, requests = client_for(tmp_path, sse(completed('{}')))
        client.token_provider = lambda: pytest.fail("Invalid images must not access credentials")
        client.request_recorder = lambda *args: pytest.fail("Invalid images must not record requests")
        with pytest.raises((ValueError, OSError)):
            client.complete("Compare", "Lecture", images=images)
        assert requests == []


    def test_image_budget_counts_base64_and_allows_exact_limit(tmp_path, monkeypatch):
        image = png_attachment(tmp_path)
        size = len(base64.b64encode(image.read_bytes()))
        monkeypatch.setattr(openai_client, "MAX_IMAGE_ENCODED_BYTES", size * 2)
        assert openai_client.image_attachments_size([("first", image), ("second", image)]) == size * 2
        with pytest.raises(ValueError, match="40 MB"):
            openai_client.image_attachments_size([("image", image)] * 3)


if __name__ == "__main__":
    unittest.main()
