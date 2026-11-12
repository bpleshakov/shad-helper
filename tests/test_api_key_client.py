"""API-key priority and credit-exhaustion fallback, using mocked HTTP only."""

import json
from unittest.mock import Mock

import httpx
import pytest

from shad_helper import auth
from shad_helper.auth import PlanAccessError
from shad_helper.cli import main
from shad_helper.openai_client import DEFAULT_MODEL, PlanClient, ResponseError
from shad_helper.openai_config import load_api_key


@pytest.fixture(autouse=True)
def configuration(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("SHAD_HELPER_OPENAI_PROXY", "http://offline-proxy:8080")


def stream(*events):
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


def completed(text='{"source":"complete"}'):
    return {"type": "response.completed", "response": {"status": "completed", "output": [
        {"content": [{"type": "output_text", "text": text}]}]}}


def catalog(request, *, plan_model=DEFAULT_MODEL):
    if request.headers["Authorization"] == "Bearer api-key":
        return httpx.Response(200, json={"data": [{"id": DEFAULT_MODEL}]})
    return httpx.Response(200, json={"models": [
        {"slug": plan_model, "visibility": "list"},
        {"slug": "hidden", "visibility": "hide"}]})


@pytest.mark.parametrize("assignment", [
    "OPENAI_API_KEY=api-key", 'OPENAI_API_KEY="api-key"',
    "export OPENAI_API_KEY='api-key' # comment", " OPENAI_API_KEY = api-key ",
])
def test_key_file_supports_env_assignments(tmp_path, assignment):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/openai.env").write_text(f"# Private configuration\n{assignment}\n")
    assert load_api_key() == "api-key"


@pytest.mark.parametrize("key", ["environment-key", "", "  "])
def test_environment_overrides_file_including_explicit_empty_key(tmp_path, monkeypatch, key):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/openai.env").write_text("OPENAI_API_KEY=file-key\n")
    monkeypatch.setenv("OPENAI_API_KEY", key)
    assert load_api_key() == (key.strip() or None)


def test_absent_key_uses_login_configuration(tmp_path):
    assert load_api_key() is None
    (tmp_path / "config").mkdir()
    (tmp_path / "config/openai.env").write_text("# No key\nUNRELATED=value\n")
    assert load_api_key() is None


@pytest.mark.parametrize("value", ['"private-key', "private-key extra-value"])
def test_invalid_key_file_has_safe_error(tmp_path, value):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/openai.env").write_text(f"OPENAI_API_KEY={value}\n")
    with pytest.raises(ValueError, match="OPENAI_API_KEY") as caught:
        load_api_key()
    assert "private-key" not in str(caught.value)


@pytest.mark.parametrize("source", ["environment", "file", "argument"])
def test_api_key_priority_without_login(tmp_path, monkeypatch, source):
    options = {}
    if source == "environment":
        monkeypatch.setenv("OPENAI_API_KEY", "api-key")
    elif source == "file":
        (tmp_path / "config").mkdir()
        (tmp_path / "config/openai.env").write_text('OPENAI_API_KEY="api-key"\n')
    else:
        options["api_key"] = "api-key"
    token = Mock(side_effect=AssertionError("Login must stay unused"))
    requests, records = [], []

    def handle(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer api-key"
        if request.method == "GET":
            return catalog(request)
        return httpx.Response(200, text=stream(completed()))

    client = PlanClient(tmp_path / "auth", transport=httpx.MockTransport(handle),
                        token_provider=token, request_recorder=lambda *args: records.append(args),
                        **options)
    assert client.complete("Instructions", "Lecture") == {"source": "complete"}
    assert [request.method for request in requests] == ["GET", "POST"]
    assert records == [(request.method, str(request.url), request.content or None)
                       for request in requests]
    assert "api-key" not in repr(records)
    token.assert_not_called()
    assert not (tmp_path / "auth").exists()


@pytest.mark.parametrize("code", ["credit_balance_exhausted", "insufficient_quota"])
@pytest.mark.parametrize("failure", ["http", "stream_failed", "stream_error", "catalog"])
def test_exhaustion_retries_same_payload_and_stays_on_login(tmp_path, code, failure, capsys):
    requests, records = [], []
    token = Mock(return_value="plan-token")
    pdf = tmp_path / "lecture.pdf"
    pdf.write_bytes(b"%PDF-1.7\noriginal")

    def handle(request):
        requests.append(request)
        if request.headers["Authorization"] == "Bearer api-key":
            if request.method == "GET" and failure != "catalog":
                return catalog(request)
            error = {"code": code, "message": "private billing information"}
            if failure in {"http", "catalog"}:
                return httpx.Response(429, json={"error": error})
            # Retrying must reuse the original bytes, even if a file changes.
            pdf.write_bytes(b"changed during the failed request")
            event = ({"type": "response.failed", "response": {"error": error}}
                     if failure == "stream_failed" else {"type": "error", **error})
            return httpx.Response(200, text=stream(
                {"type": "response.output_text.delta", "delta": '{"partial":true}'}, event))
        assert request.headers["Authorization"] == "Bearer plan-token"
        if request.method == "GET":
            return catalog(request)
        return httpx.Response(200, text=stream(completed()))

    client = PlanClient(tmp_path / "auth", api_key="api-key",
                        transport=httpx.MockTransport(handle), token_provider=token,
                        request_recorder=lambda *args: records.append(args))
    assert client.complete("Instructions", "Lecture", pdf=pdf) == {"source": "complete"}
    assert client.complete("Next", "Chapter") == {"source": "complete"}
    expected = [("GET", "api-key")]
    if failure != "catalog":
        expected.append(("POST", "api-key"))
    expected += [("GET", "plan-token"), ("POST", "plan-token"), ("POST", "plan-token")]
    assert [(r.method, r.headers["Authorization"].removeprefix("Bearer "))
            for r in requests] == expected
    if failure != "catalog":
        posts = [r.content for r in requests if r.method == "POST"]
        assert posts[0] == posts[1]
    assert records == [(r.method, str(r.url), r.content or None) for r in requests]
    output = capsys.readouterr()
    assert output.out == ""
    assert "ChatGPT" in output.err
    assert "api-key" not in repr(records) + output.err
    assert "plan-token" not in repr(records) + output.err
    assert "private billing information" not in output.err


@pytest.mark.parametrize("status,code", [
    (401, "invalid_api_key"), (403, "permission_denied"),
    (429, "rate_limit_exceeded"), (429, "slow_down"),
    (429, "project_spend_limit_exceeded"), (429, "organization_spend_limit_exceeded"),
    (429, "organization_usage_limit_exceeded"), (500, "server_error"),
])
@pytest.mark.parametrize("failure", ["http", "stream"])
def test_other_errors_do_not_switch_credentials(tmp_path, status, code, failure):
    token = Mock(side_effect=AssertionError("Unexpected login fallback"))

    def handle(request):
        if request.method == "GET":
            return catalog(request)
        # A generic quota type must not override a specific, different error code.
        error = {"code": code, "type": "insufficient_quota", "message": "private detail"}
        if failure == "http":
            return httpx.Response(status, json={"error": error},
                                  headers={"x-request-id": "failed-request"})
        return httpx.Response(200, text=stream({"type": "response.failed",
                                               "response": {"error": error}}))

    client = PlanClient(tmp_path, api_key="api-key", transport=httpx.MockTransport(handle),
                        token_provider=token)
    with pytest.raises((PlanAccessError, ResponseError)) as caught:
        client.complete("Instructions", "Lecture")
    assert caught.value.code == code
    assert "private detail" not in str(caught.value)
    assert "ChatGPT plan" not in str(caught.value)
    if failure == "http":
        assert caught.value.status == status
        assert caught.value.request_id == "failed-request"
    token.assert_not_called()


def test_fallback_requires_saved_login_and_never_opens_browser(tmp_path, monkeypatch):
    browser = Mock(side_effect=AssertionError("Unexpected browser login"))
    monkeypatch.setattr(auth.webbrowser, "open", browser)

    def handle(request):
        if request.method == "GET":
            return catalog(request)
        return httpx.Response(429, json={"error": {"code": "credit_balance_exhausted"}})

    client = PlanClient(tmp_path, api_key="api-key", transport=httpx.MockTransport(handle))
    with pytest.raises(PlanAccessError, match="shad-helper login"):
        client.complete("Instructions", "Lecture")
    browser.assert_not_called()


def test_login_catalog_is_rechecked_after_exhaustion(tmp_path):
    requests = []

    def handle(request):
        requests.append(request)
        if request.method == "GET":
            return catalog(request, plan_model="different-model")
        return httpx.Response(429, json={"error": {"code": "credit_balance_exhausted"}})

    client = PlanClient(tmp_path, api_key="api-key", token_provider=lambda: "plan-token",
                        transport=httpx.MockTransport(handle))
    with pytest.raises(PlanAccessError) as caught:
        client.complete("Instructions", "Lecture")
    assert caught.value.code == "model_unavailable"
    assert [r.method for r in requests] == ["GET", "POST", "GET"]


def test_login_failure_is_propagated_without_another_fallback(tmp_path):
    requests = []

    def handle(request):
        requests.append(request)
        if request.method == "GET":
            return catalog(request)
        return httpx.Response(429, json={"error": {"code": "credit_balance_exhausted"}})

    client = PlanClient(tmp_path, api_key="api-key", token_provider=lambda: "plan-token",
                        transport=httpx.MockTransport(handle))
    with pytest.raises(PlanAccessError, match="ChatGPT plan"):
        client.complete("Instructions", "Lecture")
    assert [r.method for r in requests] == ["GET", "POST", "GET", "POST"]


@pytest.mark.parametrize("mode", ["network", "incomplete", "invalid_json", "unavailable"])
def test_non_billing_failures_do_not_use_login(tmp_path, mode):
    token = Mock(side_effect=AssertionError("Unexpected login fallback"))

    def handle(request):
        if request.method == "GET":
            if mode == "unavailable":
                return httpx.Response(200, json={"data": []})
            return catalog(request)
        if mode == "network":
            raise httpx.ReadError("Connection interrupted")
        if mode == "incomplete":
            return httpx.Response(200, text=stream({"type": "response.incomplete"}))
        return httpx.Response(200, text=stream(completed("invalid JSON")))

    client = PlanClient(tmp_path, api_key="api-key", transport=httpx.MockTransport(handle),
                        token_provider=token)
    with pytest.raises((PlanAccessError, ResponseError)):
        client.complete("Instructions", "Lecture")
    token.assert_not_called()


@pytest.mark.parametrize(("failed_source", "code"), [
    (None, None),
    ("api_key", "credit_balance_exhausted"),
    ("api_key", "insufficient_quota"),
    ("api_key", "invalid_api_key"),
    ("chatgpt_login", "authentication_error"),
])
def test_doctor_keeps_auth_catalogs_and_failures_separate(
        tmp_path, monkeypatch, capsys, failed_source, code):
    from shad_helper import cli, openai_client

    monkeypatch.setenv("OPENAI_API_KEY", "api-key")
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: True)
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/bin/{name}")
    state_dir = tmp_path / "auth"
    state_dir.mkdir()
    (state_dir / "credentials.json").write_text("{}")
    token = Mock(return_value="plan-token")
    monkeypatch.setattr(openai_client, "access_token", token)
    browser = Mock(side_effect=AssertionError("Unexpected browser login"))
    monkeypatch.setattr(auth.webbrowser, "open", browser)
    requests = []

    def handle(request):
        assert request.method == "GET"
        assert str(request.url) == "https://api.openai.com/v1/models"
        authorization = request.headers["Authorization"]
        requests.append(authorization)
        source = "api_key" if authorization == "Bearer api-key" else "chatgpt_login"
        if source == failed_source:
            return httpx.Response(429, json={"error": {"code": code, "message": "private detail"}})
        if source == "api_key":
            return httpx.Response(200, json={"data": [{"id": DEFAULT_MODEL}, {"id": "api-only"}]})
        return httpx.Response(200, json={"models": [
            {"slug": "login-only", "visibility": "list"},
            {"slug": "hidden", "visibility": "hide"}]})

    monkeypatch.setattr(openai_client, "openai_http_client",
                        lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handle)))
    assert main(["--state-dir", str(state_dir), "doctor"]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)["openai"]
    assert set(report) == {"api_key", "chatgpt_login"}
    for source, available in {"api_key": [DEFAULT_MODEL, "api-only"],
                              "chatgpt_login": ["login-only"]}.items():
        result = report[source]
        failed = source == failed_source
        assert result["status"] == ("error" if failed else "available")
        assert result["models"] == ([] if failed else available)
        assert result["default_model"] == DEFAULT_MODEL
        assert result["default_model_available"] is (None if failed else DEFAULT_MODEL in available)
    assert requests == ["Bearer api-key", "Bearer plan-token"]
    token.assert_called_once_with(state_dir, proxy="http://offline-proxy:8080", transport=None)
    browser.assert_not_called()
    for secret in ("api-key", "plan-token", "private detail", "switching"):
        assert secret not in captured.out + captured.err
    assert "API requests: 0; ChatGPT-plan requests: 0." in captured.err
