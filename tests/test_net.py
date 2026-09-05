import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from immich_pbn.config import ResolverSpec, ServerProfile
from immich_pbn.net.resolver import (
    ResolverError,
    StaticResolver,
    SystemResolver,
    available_resolvers,
    build_resolver,
    register_resolver,
)
from immich_pbn.net.transport import HTTPStatusError, Transport, TransportError

UNRESOLVABLE = "immich.invalid-tld-used-by-tests"


def test_system_resolver_defers_to_the_os():
    assert SystemResolver().resolve("anything") is None


def test_static_resolver_is_case_insensitive():
    resolver = StaticResolver(hosts={"Immich.Local": "192.168.1.50"})
    assert resolver.resolve("immich.LOCAL") == ["192.168.1.50"]
    assert resolver.resolve("other.host") is None


def test_planned_resolvers_are_registered_but_fail_loudly():
    for kind in ("dns", "mdns"):
        assert kind in available_resolvers()
        with pytest.raises(ResolverError, match="scaffolding"):
            build_resolver(kind).resolve("immich.local")


def test_unknown_resolver_lists_what_exists():
    with pytest.raises(ResolverError, match="static"):
        build_resolver("telepathy")


def test_a_new_strategy_needs_only_a_registration():
    register_resolver("test-only", lambda **kw: StaticResolver(hosts={"a": "1.2.3.4"}))
    assert build_resolver("test-only").resolve("a") == ["1.2.3.4"]


@pytest.fixture
def echo_server():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen["host"] = self.headers.get("Host")
            seen["path"] = self.path
            seen["api_key"] = self.headers.get("x-api-key")
            if self.path.endswith("/boom"):
                body = b'{"message":"nope"}'
                self.send_response(401)
            elif self.path.endswith("/notjson"):
                body = b"<html>proxy login page</html>"
                self.send_response(200)
            else:
                body = json.dumps({"res": "pong"}).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], seen
    server.shutdown()


def _profile(port, **kwargs):
    return ServerProfile(
        name="test",
        base_url=f"http://{UNRESOLVABLE}:{port}",
        api_key="secret",
        retries=0,
        resolver=ResolverSpec(kind="static", options={"hosts": {UNRESOLVABLE: "127.0.0.1"}}),
        **kwargs,
    )


def test_resolver_redirects_the_connection_but_keeps_the_hostname(echo_server):
    port, seen = echo_server
    with Transport(_profile(port)) as transport:
        assert transport.get_json("/server/ping") == {"res": "pong"}
    # Host header and TLS identity must stay the configured name, not the IP,
    # or virtual hosts stop routing and certificates stop matching.
    assert seen["host"] == f"{UNRESOLVABLE}:{port}"
    assert seen["path"] == "/api/server/ping"
    assert seen["api_key"] == "secret"


def test_without_a_resolver_the_same_name_does_not_resolve(echo_server):
    port, _ = echo_server
    plain = ServerProfile(name="p", base_url=f"http://{UNRESOLVABLE}:{port}", retries=0)
    with pytest.raises(TransportError):
        Transport(plain).get_json("/server/ping")


def test_auth_failures_say_where_the_key_comes_from(echo_server):
    port, _ = echo_server
    with Transport(_profile(port)) as transport:
        with pytest.raises(HTTPStatusError, match="API Key") as caught:
            transport.get_json("/boom")
    assert caught.value.status == 401


def test_html_where_json_was_expected_is_a_clear_error(echo_server):
    port, _ = echo_server
    with Transport(_profile(port)) as transport:
        with pytest.raises(TransportError, match="expected JSON"):
            transport.get_json("/notjson")


def test_timeout_pair_never_exceeds_the_overall_budget():
    profile = ServerProfile(base_url="http://h", timeout=4.0)
    connect, read = Transport(profile).timeout
    assert connect <= read == 4.0
