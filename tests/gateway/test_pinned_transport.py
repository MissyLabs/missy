"""SR-1.9b: live tests for the DNS-rebinding check/connect TOCTOU fix.

SR-1.9a (earlier this session) closed the "allowlisted hostname skips the
IP check entirely" gap in NetworkPolicyEngine. But even with that fix, the
policy check and the actual HTTP connection were two independent DNS
resolutions: PolicyHTTPClient._check_url() validates an IP via
getaddrinfo(), then discards it -- the real request goes through httpx,
which lets httpcore do its own, separate resolution when it actually
opens the socket. A low-TTL DNS record (attacker-controlled or otherwise)
can return a different address between the two, bypassing every check
that was just performed.

These tests exercise the real fix end-to-end: a real socket server, a
real PolicyHTTPClient, and a monkeypatched socket.getaddrinfo that would
raise if the target hostname were ever resolved a second time -- proving
the actual TCP connection uses the IP the policy check validated, not a
fresh resolution.
"""

from __future__ import annotations

import http.server
import socket
import threading
from collections.abc import Generator

import httpx
import pytest

from missy.config.settings import (
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ShellPolicy,
)
from missy.gateway import client as gateway_module
from missy.gateway.client import PolicyHTTPClient
from missy.gateway.pinned_transport import (
    PinnedAsyncHTTPTransport,
    PinnedHTTPTransport,
    clear_pin,
    pin_host,
)
from missy.policy import engine as engine_module
from missy.policy.engine import init_policy_engine


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(f"ok Host={self.headers.get('Host', '')}".encode())

    def log_message(self, *args: object) -> None:  # noqa: D102 - silence test noise
        pass


@pytest.fixture
def real_server() -> Generator[int, None, None]:
    """A real local HTTP server bound to 127.0.0.1 on an ephemeral port."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _EchoHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()
        thread.join(timeout=2)


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    original_engine = engine_module._engine
    original_approval = gateway_module._interactive_approval
    yield
    engine_module._engine = original_engine
    gateway_module._interactive_approval = original_approval


def _config(allowed_hosts=None, allowed_cidrs=None, default_deny=True) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(
            default_deny=default_deny,
            allowed_hosts=allowed_hosts or [],
            allowed_cidrs=allowed_cidrs or [],
        ),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(),
        plugins=PluginPolicy(),
        providers={},
        workspace_path="/tmp",
        audit_log_path="/tmp/audit-sr19b.log",
    )


class TestPinnedConnectionUsesValidatedIP:
    def test_bare_ip_target_connects_for_real(self, real_server: int):
        init_policy_engine(_config(allowed_cidrs=["127.0.0.1/32"]))
        client = PolicyHTTPClient(session_id="s1", task_id="t1", category="tool")
        resp = client.get(f"http://127.0.0.1:{real_server}/")
        assert resp.status_code == 200
        assert "ok" in resp.text

    def test_hostname_target_resolved_exactly_once_not_at_connect_time(
        self, real_server: int, monkeypatch: pytest.MonkeyPatch
    ):
        """The core SR-1.9b property: getaddrinfo("pinned-test-host", ...)
        must be called exactly once (by the policy check), never again by
        the actual connection -- proving the connection reuses the
        validated IP rather than re-resolving."""
        init_policy_engine(
            _config(
                allowed_hosts=[f"pinned-test-host:{real_server}"],
                allowed_cidrs=["127.0.0.1/32"],
            )
        )
        real_getaddrinfo = socket.getaddrinfo
        calls_for_target = {"n": 0}

        def _tracking_getaddrinfo(host, *args, **kwargs):
            if host == "pinned-test-host":
                calls_for_target["n"] += 1
                if calls_for_target["n"] > 1:
                    raise AssertionError(
                        "pinned-test-host was resolved a second time -- "
                        "the connection re-resolved instead of using the "
                        "policy-validated pinned IP (TOCTOU gap is back)"
                    )
                return real_getaddrinfo("127.0.0.1", *args, **kwargs)
            return real_getaddrinfo(host, *args, **kwargs)

        monkeypatch.setattr(socket, "getaddrinfo", _tracking_getaddrinfo)

        client = PolicyHTTPClient(session_id="s1", task_id="t1", category="tool")
        resp = client.get(f"http://pinned-test-host:{real_server}/")

        assert resp.status_code == 200
        assert "pinned-test-host" in resp.text  # Host header preserved correctly
        assert calls_for_target["n"] == 1

    def test_simulated_rebinding_attack_does_not_reach_the_new_address(
        self, real_server: int, monkeypatch: pytest.MonkeyPatch
    ):
        """Directly simulate the review's attack: a hostname that would
        resolve to the real server at check time, but to an unreachable
        "attacker" address if resolved again at connect time. Confirm the
        request still succeeds -- proving it never used the second
        (attacker-controlled) resolution."""
        init_policy_engine(
            _config(
                allowed_hosts=[f"rebind-attack-host:{real_server}"],
                allowed_cidrs=["127.0.0.1/32"],
            )
        )
        real_getaddrinfo = socket.getaddrinfo
        state = {"calls": 0}

        def _rebinding_getaddrinfo(host, *args, **kwargs):
            if host == "rebind-attack-host":
                state["calls"] += 1
                if state["calls"] == 1:
                    return real_getaddrinfo("127.0.0.1", *args, **kwargs)
                # A "connect-time" resolution would now point somewhere
                # that can never serve this request (TEST-NET-1, RFC 5737 --
                # guaranteed non-routable, connection will fail/hang).
                return real_getaddrinfo("192.0.2.1", *args, **kwargs)
            return real_getaddrinfo(host, *args, **kwargs)

        monkeypatch.setattr(socket, "getaddrinfo", _rebinding_getaddrinfo)

        client = PolicyHTTPClient(session_id="s1", task_id="t1", category="tool")
        resp = client.get(f"http://rebind-attack-host:{real_server}/", timeout=3.0)

        assert resp.status_code == 200
        assert state["calls"] == 1


class TestFailClosedWhenUnpinned:
    def test_pinned_transport_refuses_connection_with_no_pin(self):
        """If _check_url() were ever bypassed, the transport must not
        silently fall back to an unvalidated DNS resolution."""
        transport = PinnedHTTPTransport()
        client = httpx.Client(transport=transport)
        clear_pin("example-not-pinned.invalid")
        with pytest.raises(httpx.ConnectError, match="no policy-validated IP pinned"):
            client.get("http://example-not-pinned.invalid/")

    @pytest.mark.asyncio
    async def test_async_pinned_transport_refuses_connection_with_no_pin(self):
        transport = PinnedAsyncHTTPTransport()
        client = httpx.AsyncClient(transport=transport)
        clear_pin("example-not-pinned-async.invalid")
        with pytest.raises(httpx.ConnectError, match="no policy-validated IP pinned"):
            await client.get("http://example-not-pinned-async.invalid/")

    def test_pinned_transport_connects_when_pin_present(self, real_server: int):
        transport = PinnedHTTPTransport()
        client = httpx.Client(transport=transport)
        pin_host("pin-direct-test", "127.0.0.1")
        try:
            resp = client.get(f"http://pin-direct-test:{real_server}/")
            assert resp.status_code == 200
        finally:
            clear_pin("pin-direct-test")


class TestAsyncPinnedConnection:
    @pytest.mark.asyncio
    async def test_async_request_resolves_hostname_exactly_once(
        self, real_server: int, monkeypatch: pytest.MonkeyPatch
    ):
        init_policy_engine(
            _config(
                allowed_hosts=[f"pinned-async-host:{real_server}"],
                allowed_cidrs=["127.0.0.1/32"],
            )
        )
        real_getaddrinfo = socket.getaddrinfo
        calls = {"n": 0}

        def _tracking_getaddrinfo(host, *args, **kwargs):
            if host == "pinned-async-host":
                calls["n"] += 1
                if calls["n"] > 1:
                    raise AssertionError("re-resolved at connect time")
                return real_getaddrinfo("127.0.0.1", *args, **kwargs)
            return real_getaddrinfo(host, *args, **kwargs)

        monkeypatch.setattr(socket, "getaddrinfo", _tracking_getaddrinfo)

        client = PolicyHTTPClient(session_id="s1", task_id="t1", category="tool")
        resp = await client.aget(f"http://pinned-async-host:{real_server}/")

        assert resp.status_code == 200
        assert calls["n"] == 1
        await client.aclose()


class TestOperatorOverridePinning:
    def test_operator_override_still_pins_something(self, real_server: int):
        """The interactive-approval override path raises past the policy
        check (before any pin would normally be set) -- confirm it still
        pins a best-effort IP so the transport doesn't fail-closed on a
        request the operator explicitly approved."""
        from missy.agent.interactive_approval import InteractiveApproval
        from missy.gateway.client import set_interactive_approval

        init_policy_engine(_config(default_deny=True))  # nothing allowed by default

        class _AlwaysApprove(InteractiveApproval):
            def __init__(self):  # noqa: D107 - test double, no real init needed
                pass

            def prompt_user(self, action: str, detail: str, session_id: str = "") -> bool:
                return True

        set_interactive_approval(_AlwaysApprove())
        try:
            client = PolicyHTTPClient(session_id="s1", task_id="t1", category="tool")
            resp = client.get(f"http://127.0.0.1:{real_server}/")
            assert resp.status_code == 200
        finally:
            set_interactive_approval(None)
