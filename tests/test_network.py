import socket
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import httpcore
import httpx
import pytest

from shad_helper import network


@pytest.mark.parametrize("proxy", [
    "http://127.0.0.1:8080",
    "https://proxy.example:443",
    "http://username:password@proxy.example:3128/",
    "https://name:p%40ss%3Aword@proxy.example",
    "http://[::1]:8080",
])
def test_valid_http_proxy_urls_preserve_configuration(proxy):
    assert network.resolve_openai_proxy(proxy) == proxy


def test_explicit_proxy_takes_precedence_over_environment(monkeypatch):
    monkeypatch.setenv("SHAD_HELPER_OPENAI_PROXY", "http://environment.example:8080")
    assert network.resolve_openai_proxy("https://explicit.example:443") == "https://explicit.example:443"
    assert network.resolve_openai_proxy() == "http://environment.example:8080"


def test_explicit_blank_proxy_is_rejected_instead_of_using_environment(monkeypatch):
    monkeypatch.setenv("SHAD_HELPER_OPENAI_PROXY", "http://environment.example:8080")
    with pytest.raises(ValueError, match="Configure an OpenAI proxy"):
        network.resolve_openai_proxy("")


@pytest.mark.parametrize("proxy", [
    "proxy.example:8080",
    "not a URL",
    "socks5://username:DO_NOT_ECHO@proxy.example:8080",
    "ftp://username:DO_NOT_ECHO@proxy.example:8080",
    "http://username:DO_NOT_ECHO@proxy.example/path",
    "http://username:DO_NOT_ECHO@proxy.example?query=value",
    "http://username:DO_NOT_ECHO@proxy.example#fragment",
    "http://username:DO_NOT_ECHO@proxy.example:invalid",
    "http://username:DO_NOT_ECHO@proxy.example:65536",
    "http://username:DO_NOT_ECHO@proxy.example:0",
    "http://username:DO_NOT_ECHO@proxy.example:-1",
    "http://username:DO_NOT_ECHO@",
    "http:///missing-host",
    "http://[invalid",
    " http://username:DO_NOT_ECHO@proxy.example:8080",
    "http://username:DO_NOT_ECHO@proxy.example:8080\n",
    "http://username:DO_NOT_ECHO@proxy .example:8080",
])
def test_malformed_proxy_urls_are_rejected_without_echoing_credentials(proxy):
    with pytest.raises(ValueError) as raised:
        network.resolve_openai_proxy(proxy)
    assert "DO_NOT_ECHO" not in str(raised.value)
    assert proxy not in str(raised.value)


def test_missing_proxy_blocks_client_creation_even_with_standard_proxy_environment(monkeypatch):
    monkeypatch.delenv("SHAD_HELPER_OPENAI_PROXY", raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://ordinary.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://ordinary.example:8080")
    with patch.object(network.httpx, "Client") as constructor, \
            pytest.raises(ValueError, match="SHAD_HELPER_OPENAI_PROXY"):
        network.openai_http_client()
    constructor.assert_not_called()


def test_constructor_uses_selected_proxy_and_disables_environment_overrides(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://wrong.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://wrong.example:8080")
    monkeypatch.setenv("NO_PROXY", "*")
    with patch.object(network.httpx, "Client") as constructor:
        result = network.openai_http_client(proxy="https://selected.example:443", timeout=12)
    constructor.assert_called_once_with(
        proxy="https://selected.example:443", transport=None, timeout=12, trust_env=False,
    )
    assert result is constructor.return_value
    # Omitting verify preserves HTTPX's certificate verification default.
    assert "verify" not in constructor.call_args.kwargs


def test_client_keeps_certificate_verification_enabled():
    with network.openai_http_client(proxy="http://127.0.0.1:8080") as client:
        assert client._transport._pool._ssl_context.verify_mode == ssl.CERT_REQUIRED
        assert client._transport._pool._ssl_context.check_hostname is True


def test_invalid_proxy_blocks_creation_before_httpx_and_redacts_constructor_errors():
    with patch.object(network.httpx, "Client") as constructor, pytest.raises(ValueError):
        network.openai_http_client(proxy="http://name:DO_NOT_ECHO@proxy.example:bad")
    constructor.assert_not_called()
    with patch.object(network.httpx, "Client", side_effect=ValueError("DO_NOT_ECHO")), \
            pytest.raises(ValueError) as raised:
        network.openai_http_client(proxy="http://name:DO_NOT_ECHO@proxy.example:8080")
    assert "DO_NOT_ECHO" not in str(raised.value)


def test_injected_transport_still_requires_a_valid_proxy_and_never_uses_network(monkeypatch):
    monkeypatch.delenv("SHAD_HELPER_OPENAI_PROXY", raising=False)
    targets = []

    def respond(request):
        targets.append(str(request.url))
        return httpx.Response(200, json={"offline": True})

    transport = httpx.MockTransport(respond)
    with pytest.raises(ValueError, match="Configure an OpenAI proxy"):
        network.openai_http_client(transport=transport)
    with pytest.raises(ValueError):
        network.openai_http_client(proxy="socks5://localhost:8080", transport=transport)
    with network.openai_http_client(proxy="http://127.0.0.1:1", transport=transport) as client:
        for target in ("https://api.openai.com/v1/models", "https://auth.openai.com/.well-known/jwks.json"):
            assert client.get(target).json() == {"offline": True}
    assert targets == ["https://api.openai.com/v1/models", "https://auth.openai.com/.well-known/jwks.json"]


def test_real_httpx_mounts_route_both_openai_hosts_to_the_configured_proxy(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("HTTP_PROXY", "http://wrong.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://wrong.example:8080")
    targets = []

    def request_boundary(pool, request):
        # HTTPX chooses its actual mount before reaching this socket-free boundary.
        assert isinstance(pool, httpcore.HTTPProxy), "The direct connection pool was selected"
        assert pool._proxy_url.scheme == b"http"
        assert pool._proxy_url.host == b"configured.example"
        assert pool._proxy_url.port == 3128
        targets.append(request.url.host)
        return httpcore.Response(200, content=b"proxied")

    with patch.object(httpcore.ConnectionPool, "handle_request", request_boundary), \
            patch.object(socket, "create_connection", side_effect=AssertionError("No socket is allowed")), \
            patch.object(socket, "getaddrinfo", side_effect=AssertionError("No DNS is allowed")), \
            network.openai_http_client(proxy="http://configured.example:3128") as client:
        for target in ("https://api.openai.com/v1/models", "https://auth.openai.com/.well-known/jwks.json"):
            response = client.get(target)
            assert response.status_code == 200
            assert response.text == "proxied"
    assert targets == [b"api.openai.com", b"auth.openai.com"]


def test_no_proxy_cannot_bypass_configured_proxy_for_openai_hosts(monkeypatch):
    targets = []

    class ProxyHandler(BaseHTTPRequestHandler):
        def do_CONNECT(self):
            targets.append(self.path)
            # The proxy records the tunnel destination and never opens a remote socket.
            self.send_response(502)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format, *args):
            pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
    except OSError as error:
        pytest.skip(f"Local proxy integration cannot bind loopback in this environment: {error}")
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    worker.start()
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    connect = socket.create_connection

    def local_only_connection(address, *args, **kwargs):
        assert address[0] == "127.0.0.1", "The client attempted to bypass the local proxy"
        return connect(address, *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", local_only_connection)
    proxy = f"http://127.0.0.1:{server.server_port}"
    try:
        with network.openai_http_client(proxy=proxy, timeout=2) as client:
            for target in ("https://api.openai.com/v1/models", "https://auth.openai.com/.well-known/jwks.json"):
                with pytest.raises(httpx.ProxyError):
                    client.get(target)
        assert targets == ["api.openai.com:443", "auth.openai.com:443"]
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
