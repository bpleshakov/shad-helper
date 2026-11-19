"""Command-local API spending estimates from response usage, without billing credentials."""

import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal

_command_usage = ContextVar("openai_command_usage", default=None)


def response_cost(response, requested_model) -> Decimal | None:
    """Price this app's text/PDF/image requests; missing or unsupported usage is unknown."""
    # GPT-6 Astra public rates, checked 2026-11-08:
    # https://developers.openai.com/api/docs/models/gpt-6-astra
    # https://developers.openai.com/api/docs/guides/prompt-caching
    try:
        if response.get("model", requested_model) != "gpt-6-astra":
            return None
        if any(item.get("type") == "image_generation_call" for item in response.get("output", [])):
            return None  # Image tool charges are additional to the text model's tokens.
        multiplier = {"default": Decimal(1), "flex": Decimal("0.5"),
                      "priority": Decimal(2), "fast": Decimal(2)}.get(
                          response.get("service_tier") or "default")
        if multiplier is None:
            return None
        usage = response["usage"]
        inputs, outputs = usage["input_tokens"], usage["output_tokens"]
        details = usage["input_tokens_details"]
        cached, writes = details["cached_tokens"], details.get("cache_write_tokens", 0)
        lifetime = details.get("cache_write_tokens_details")
        if lifetime is None and writes == 0:
            write_30m, write_12h = 0, 0
        else:
            write_30m, write_12h = lifetime["ttl_30m_tokens"], lifetime["ttl_12h_tokens"]
        counts = (inputs, outputs, cached, writes, write_30m, write_12h)
        if any(type(count) is not int or count < 0 for count in counts):
            return None
        if cached + writes > inputs or write_30m + write_12h != writes:
            return None
    except (AttributeError, KeyError, TypeError):
        return None
    input_cost = ((inputs - cached - writes) * Decimal(10) + cached
                  + write_30m * Decimal("12.5") + write_12h * Decimal(25))
    # Reasoning tokens are already included in output_tokens.
    output_cost = outputs * Decimal(50)
    if inputs > 272_000:
        input_cost *= 2
        output_cost *= Decimal("1.5")
    return (input_cost + output_cost) * multiplier / Decimal(1_000_000)


@dataclass
class CommandUsage:
    cost: Decimal = Decimal(0)
    api_requests: int = 0
    plan_requests: int = 0
    unknown_requests: int = 0

    def summary(self, command):
        if self.unknown_requests:
            amount = (f"API cost unknown; known subtotal ${self.cost:.6f} (estimated); "
                      f"unpriced API requests: {self.unknown_requests}")
        else:
            amount = f"estimated API cost ${self.cost:.6f}"
        return (f"OpenAI spending [{command}]: {amount}; API requests: {self.api_requests}; "
                f"ChatGPT-plan requests: {self.plan_requests}.")


@contextmanager
def command_usage(command):
    usage = CommandUsage()
    token = _command_usage.set(usage)
    try:
        yield usage
    finally:
        _command_usage.reset(token)
        print(usage.summary(command), file=sys.stderr, flush=True)


@contextmanager
def track_request(*, api_key, model):
    """Count each sent generation attempt, including retries and interrupted streams."""
    from .run_diagnostics import record_generation_request

    record_generation_request()
    usage = _command_usage.get()
    if usage is None:
        yield None
        return
    if not api_key:
        usage.plan_requests += 1
        yield None
        return
    usage.api_requests += 1
    recorded = False

    def record(response=None, *, rejected=False):
        nonlocal recorded
        if recorded:
            return
        recorded = True
        cost = Decimal(0) if rejected else response_cost(response, model)
        if cost is None:
            usage.unknown_requests += 1
        else:
            usage.cost += cost

    try:
        yield record
    finally:
        if not recorded:
            usage.unknown_requests += 1
