"""Command spending counts only this command's API attempts and reported usage."""

import json
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from shad_helper import cli
from shad_helper.auth import PlanAccessError
from shad_helper.openai_client import DEFAULT_MODEL, PlanClient, ResponseError
from shad_helper.openai_usage import command_usage, response_cost


def response(*, inputs=1000, cached=200, outputs=100, **overrides):
    return {"model": DEFAULT_MODEL, "service_tier": "default", "status": "completed",
            "usage": {"input_tokens": inputs, "input_tokens_details": {
                "cached_tokens": cached, "cache_write_tokens": 0},
                "output_tokens": outputs, "output_tokens_details": {"reasoning_tokens": 80}},
            "output": [{"content": [{"type": "output_text", "text": "{}"}]}], **overrides}


def sse(result, kind="response.completed"):
    return "data: " + json.dumps({"type": kind, "response": result}) + "\n\n"


def client(tmp_path, results, *, api_key="test-key"):
    responses = iter(results)
    requests = []

    def handle(request):
        requests.append(request)
        if request.method == "GET":
            if request.headers["Authorization"] == "Bearer test-key":
                return httpx.Response(200, json={"data": [{"id": DEFAULT_MODEL}]})
            return httpx.Response(200, json={"models": [
                {"slug": DEFAULT_MODEL, "visibility": "list"}]})
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, httpx.Response):
            return result
        return httpx.Response(200, text=result)

    return PlanClient(tmp_path, proxy="http://offline-proxy:8080", api_key=api_key,
                      token_provider=lambda: "plan-token", transport=httpx.MockTransport(handle)), requests


def test_price_uses_cached_input_and_counts_reasoning_once():
    assert response_cost(response(), DEFAULT_MODEL) == Decimal("0.0132")


def test_price_separates_both_cache_write_lifetimes():
    result = response(inputs=12000, cached=4000)
    result["usage"]["input_tokens_details"].update(cache_write_tokens=7000,
        cache_write_tokens_details={"ttl_30m_tokens": 3000, "ttl_12h_tokens": 4000})
    assert response_cost(result, DEFAULT_MODEL) == Decimal("0.1565")


@pytest.mark.parametrize("inputs,expected", [(272000, "2.725"), (272001, "5.44752")])
def test_long_context_boundary(inputs, expected):
    assert response_cost(response(inputs=inputs, cached=0), DEFAULT_MODEL) == Decimal(expected)


@pytest.mark.parametrize("tier,multiplier", [
    ("default", "1"), ("flex", "0.5"), ("priority", "2"), ("fast", "2"),
])
def test_actual_processing_tier_changes_the_estimate(tier, multiplier):
    assert response_cost(response(service_tier=tier), DEFAULT_MODEL) == (
        Decimal("0.0132") * Decimal(multiplier))


@pytest.mark.parametrize("change", [
    {"usage": None}, {"usage": {}}, {"model": "unknown-model"},
    {"service_tier": "unknown-tier"},
])
def test_missing_usage_or_unknown_pricing_is_not_zero(change):
    assert response_cost(response(**change), DEFAULT_MODEL) is None


def test_image_tool_cost_is_unpriced_even_when_text_model_has_rates():
    result = response(model="gpt-6-astra", output=[{"type": "image_generation_call"}])
    assert response_cost(result, "gpt-6-astra") is None


@pytest.mark.parametrize("field,value", [
    ("input_tokens", -1), ("input_tokens", True), ("output_tokens", "100"),
    ("output_tokens", float("nan")), ("input_tokens_details", None),
])
def test_malformed_usage_is_unpriced(field, value):
    result = response()
    result["usage"][field] = value
    assert response_cost(result, DEFAULT_MODEL) is None


@pytest.mark.parametrize("details", [
    {"cached_tokens": 1001}, {"cached_tokens": -1}, {"cached_tokens": 0, "cache_write_tokens": 10},
    {"cached_tokens": 0, "cache_write_tokens": 10,
     "cache_write_tokens_details": {"ttl_30m_tokens": 8, "ttl_12h_tokens": 3}},
])
def test_inconsistent_cache_counts_are_unpriced(details):
    result = response()
    result["usage"]["input_tokens_details"] = details
    assert response_cost(result, DEFAULT_MODEL) is None


def test_completed_requests_from_multiple_clients_share_command_total(tmp_path, capsys):
    first, _ = client(tmp_path, [sse(response())])
    second, _ = client(tmp_path, [sse(response())])
    with command_usage("run"):
        first.complete("Instructions", "Chapter")
        second.complete("Instructions", "Cards")
    output = capsys.readouterr()
    assert output.out == ""
    assert "estimated API cost $0.026400" in output.err
    assert "API requests: 2; ChatGPT-plan requests: 0" in output.err


@pytest.mark.parametrize("failure", ["invalid_json", "incomplete", "failed", "refusal"])
def test_failed_generation_and_retry_both_contribute_cost(tmp_path, capsys, failure):
    result = response()
    kind = "response.completed"
    if failure == "invalid_json":
        result["output"][0]["content"][0]["text"] = "not JSON"
    elif failure == "refusal":
        result["output"][0]["content"] = [{"type": "refusal"}]
    else:
        result["status"] = failure
        result["error"] = {"code": "server_error"}
        kind = "response." + failure
    model, _ = client(tmp_path, [sse(result, kind), sse(response())])
    with command_usage("notes"):
        with pytest.raises((PlanAccessError, ResponseError)):
            model.complete("Instructions", "Chapter")
        model.complete("Instructions", "Retry")
    assert "estimated API cost $0.026400" in capsys.readouterr().err


@pytest.mark.parametrize("failure", [
    sse(response(usage=None)), "data: [DONE]\n\n", httpx.ReadError("Interrupted"),
    httpx.Response(500),
])
def test_unreported_usage_marks_total_unknown(tmp_path, capsys, failure):
    model, _ = client(tmp_path, [sse(response()), failure])
    with command_usage("notes"):
        model.complete("Instructions", "Chapter")
        try:
            model.complete("Instructions", "Next chapter")
        except (PlanAccessError, ResponseError):
            pass
    output = capsys.readouterr().err
    assert "API cost unknown" in output
    assert "known subtotal $0.013200 (estimated)" in output
    assert "unpriced API requests: 1" in output


def test_credit_rejection_and_login_fallback_do_not_add_api_cost(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "unused-admin-secret")
    model, requests = client(tmp_path, [
        httpx.Response(429, json={"error": {"code": "credit_balance_exhausted"}}), sse(response())])
    with command_usage("notes"):
        model.complete("Instructions", "Chapter")
    output = capsys.readouterr().err
    assert "estimated API cost $0.000000" in output
    assert "API requests: 1; ChatGPT-plan requests: 1" in output
    assert "unknown" not in output
    assert {r.url.path for r in requests} == {"/v1/models", "/v1/responses"}
    assert all("unused-admin-secret" not in r.headers["Authorization"] for r in requests)


def test_invalid_input_makes_no_chargeable_attempt(tmp_path, capsys):
    model, requests = client(tmp_path, [])
    with command_usage("anki"), pytest.raises(FileNotFoundError):
        model.complete("Instructions", "Cards", pdf=tmp_path / "missing.pdf")
    assert requests == []
    assert "API requests: 0; ChatGPT-plan requests: 0" in capsys.readouterr().err


@pytest.mark.parametrize("failure", [None, RuntimeError("Failed after generation"), KeyboardInterrupt()])
def test_cli_reports_spending_after_success_failure_and_interrupt(
        tmp_path, monkeypatch, capsys, failure):
    model, _ = client(tmp_path, [sse(response())])

    def generate(job, *args, **kwargs):
        model.complete("Instructions", "Chapter")
        if failure is not None:
            raise failure
        return job / "notes/main.tex"

    monkeypatch.setitem(sys.modules, "shad_helper.notes", SimpleNamespace(generate_notes=generate))
    if isinstance(failure, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            cli.main(["notes", str(tmp_path)])
    else:
        assert cli.main(["notes", str(tmp_path)]) == (1 if failure else 0)
    output = capsys.readouterr().err
    assert output.count("OpenAI spending [notes]") == 1
    assert "estimated API cost $0.013200" in output

    # A second command in the same process must start at zero.
    monkeypatch.setitem(sys.modules, "shad_helper.rendering", SimpleNamespace(
        render=Mock(return_value=Path("main.pdf"))))
    assert cli.main(["render", str(tmp_path)]) == 0
    assert "estimated API cost $0.000000" in capsys.readouterr().err


def test_cli_batch_combines_projects_and_counts_plan_usage_separately(
        tmp_path, monkeypatch, capsys):
    models = [client(tmp_path, [sse(response())])[0],
              client(tmp_path, [sse(response())])[0],
              client(tmp_path, [sse(response())], api_key="")[0]]
    clients = iter(models)

    def generate(job, *args, **kwargs):
        next(clients).complete("Instructions", "Chapter")
        return job / "notes/main.tex"

    monkeypatch.setitem(sys.modules, "shad_helper.notes", SimpleNamespace(generate_notes=generate))
    assert cli.main(["notes", *(str(tmp_path / str(index)) for index in range(3))]) == 0
    output = capsys.readouterr()
    assert "3 completed, 0 failed" in output.out
    assert output.err == ("OpenAI spending [notes]: estimated API cost $0.026400; "
                          "API requests: 2; ChatGPT-plan requests: 1.\n")
