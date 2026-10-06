"""One mandatory proxy configuration for all app-owned OpenAI HTTP clients."""

import os
from urllib.parse import urlsplit

import httpx


def resolve_openai_proxy(proxy: str | None = None) -> str:
    value = os.environ.get("SHAD_HELPER_OPENAI_PROXY") if proxy is None else proxy
    if not value:
        raise ValueError("Configure an OpenAI proxy with --openai-proxy or SHAD_HELPER_OPENAI_PROXY.")
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
                or parsed.port == 0
                or any(character.isspace() or ord(character) < 32 or ord(character) == 127
                       for character in value)):
            raise ValueError
    except (ValueError, TypeError):
        # A malformed URL can contain proxy credentials; never include it in errors.
        raise ValueError("OpenAI proxy must be a valid http:// or https:// proxy URL.") from None
    return value


def openai_http_client(*, proxy: str | None = None, transport=None, timeout=30) -> httpx.Client:
    selected = resolve_openai_proxy(proxy)
    # Explicit transport injection is an offline test seam. Proxy mounts would
    # otherwise take precedence over MockTransport and unexpectedly use the network.
    try:
        return httpx.Client(proxy=selected if transport is None else None,
                            transport=transport, timeout=timeout, trust_env=False)
    except (ValueError, httpx.InvalidURL):
        raise ValueError("Cannot configure the OpenAI proxy; check its URL.") from None
