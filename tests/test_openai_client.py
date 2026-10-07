"""Offline unittest checks plus full mocked HTTP tests when dependencies exist."""

import ast
import importlib.util
import json
import os
import unittest
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
    module.__dict__.update(json=json, Path=Path,
        RESOURCE="https://api.openai.com/v1", access_token=token_provider,
        resolve_openai_proxy=resolve_proxy, openai_http_client=Client,
        http_error=lambda _: module.PlanAccessError("offline HTTP error"),
        httpx=SimpleNamespace(Client=Client, Timeout=lambda *args, **kwargs: None, HTTPError=OSError))
    source_path = Path(__file__).parents[1] / "shad_helper" / "openai_client.py"
    source_tree = ast.parse(source_path.read_text())
    source_tree.body = [node for node in source_tree.body if isinstance(node, (ast.Assign, ast.FunctionDef, ast.ClassDef))]
    exec(compile(source_tree, str(source_path), "exec"), module.__dict__)
    module.response_events = [_completed()]
    return module, calls


class ClientStdlibTests(unittest.TestCase):
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
        client._available_models = ["gpt-6.1-sol"]
        with self.assertRaisesRegex(module.PlanAccessError, "unavailable"):
            client.complete("Instructions", "Text")
        self.assertEqual(calls, [])
        client._available_models = ["gpt-6-astra"]
        for output, message in [("not JSON", "invalid notes JSON"), ("[]", "JSON object")]:
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
        assert body["reasoning"] == {"effort": "high"}
        assert body["store"] is False and body["stream"] is True
        assert body["instructions"] == "Return notes JSON."
        assert isinstance(body["input"], list)
        assert body["input"][0]["role"] == "user"
        assert body["input"][0]["content"] == [{"type": "input_text", "text": "Lecture text."}]
        assert body["text"]["format"]["schema"] == schema
        assert not set(body) & {"tools", "max_output_tokens", "previous_response_id", "background", "temperature"}
        assert client.models() == ["gpt-6-astra"]


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
        client, requests = client_for(tmp_path, "", models=[{"slug": "gpt-6.1-sol", "visibility": "list"}])
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
        ([completed("not JSON")], ResponseError, "invalid notes JSON"),
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


if __name__ == "__main__":
    unittest.main()
