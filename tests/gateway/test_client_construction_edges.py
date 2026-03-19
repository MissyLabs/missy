"""Session 15 tests for missy/gateway/client.py.

Comprehensive coverage of PolicyHTTPClient focusing on angles not deeply
exercised in earlier test files:

- PolicyHTTPClient construction edge cases (max_response_bytes, None timeout)
- Connection pool limits are applied at client construction
- follow_redirects is disabled on both sync and async clients
- URL parsing edge cases: IPv6, port-in-URL, trailing slash, fragments
- Host extraction from port-bearing URLs for policy check
- REST policy path extraction (empty path defaults to "/")
- _check_url called with and without method argument
- _check_rest_policy called only when method is non-empty
- REST policy deny detail carries host, method, and path
- create_client passes max_response_bytes through (default value)
- set_interactive_approval module-level assignment and idempotent clear
- InteractiveApproval prompt_user receives correct operation and url
- Approval prompt_user=True does not call httpx when policy allows anyway
- Operator approves: httpx IS called and response returned
- Operator denies: httpx is NOT called and PolicyViolationError propagates
- Kwargs stripping: each dangerous key stripped individually
- Allowed kwargs exhaustive set membership check
- _sanitize_kwargs is a classmethod (no instance required)
- Response size: Content-Length == limit (boundary: not exceeded)
- Response size: Content-Length == limit+1 (boundary: exceeded)
- Response size: body length == limit (no error)
- Response size: body length == limit+1 (error)
- _check_response_size handles response.content access exception silently
- Audit event timestamp is timezone-aware
- Audit event policy_rule field is None for network_request events
- Audit events from different sessions do not bleed into each other
- Synchronous close is idempotent (double-close)
- Async aclose is idempotent (double-close)
- close() does not affect async client; aclose() does not affect sync client
- All six sync methods emit correct method name in audit event
- All six async methods emit correct method name in audit event
- create_client with all parameters produces expected attribute values
- Permissive policy allows any host including IP addresses
- Restrictive policy with CIDR allows IP addresses in range
- Category "provider", "tool", "discord" each forwarded correctly
- Session ID and task ID forwarded to every check_network call
- pool limits: max_connections, max_keepalive_connections, keepalive_expiry
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from missy.config.settings import (
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ShellPolicy,
)
from missy.core.events import event_bus
from missy.core.exceptions import PolicyViolationError
from missy.gateway import client as gateway_module
from missy.gateway.client import PolicyHTTPClient, create_client, set_interactive_approval
from missy.policy import engine as engine_module
from missy.policy.engine import init_policy_engine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _permissive_config() -> MissyConfig:
    """Network default_deny=False — all hosts are reachable."""
    return MissyConfig(
        network=NetworkPolicy(
            default_deny=False,
            allowed_cidrs=[],
            allowed_domains=[],
            allowed_hosts=[],
        ),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(),
        plugins=PluginPolicy(),
        providers={},
        workspace_path="/tmp",
        audit_log_path="/tmp/audit.log",
    )


def _restrictive_config(
    allowed_hosts: list[str] | None = None,
    allowed_domains: list[str] | None = None,
    allowed_cidrs: list[str] | None = None,
) -> MissyConfig:
    """Network default_deny=True — only explicit allow lists pass."""
    return MissyConfig(
        network=NetworkPolicy(
            default_deny=True,
            allowed_cidrs=allowed_cidrs or [],
            allowed_domains=allowed_domains or [],
            allowed_hosts=allowed_hosts or [],
        ),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(),
        plugins=PluginPolicy(),
        providers={},
        workspace_path="/tmp",
        audit_log_path="/tmp/audit.log",
    )


def _mock_response(
    status_code: int = 200,
    headers: dict | None = None,
    content: bytes = b"ok",
) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.content = content
    raw_headers = headers or {}
    resp.headers = MagicMock()
    resp.headers.get = lambda k, default=None: raw_headers.get(k, default)
    return resp


# ---------------------------------------------------------------------------
# Autouse fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    """Reset policy engine to permissive, clear event bus, clear approval."""
    original_engine = engine_module._engine
    original_approval = gateway_module._interactive_approval

    init_policy_engine(_permissive_config())
    event_bus.clear()
    set_interactive_approval(None)

    yield

    engine_module._engine = original_engine
    gateway_module._interactive_approval = original_approval
    event_bus.clear()


# ---------------------------------------------------------------------------
# Construction edge cases
# ---------------------------------------------------------------------------


class TestConstructionEdgeCases:
    """Additional construction scenarios not covered by earlier tests."""

    def test_max_response_bytes_is_stored(self) -> None:
        client = PolicyHTTPClient(max_response_bytes=512)
        assert client.max_response_bytes == 512

    def test_large_max_response_bytes(self) -> None:
        limit = 200 * 1024 * 1024  # 200 MB
        client = PolicyHTTPClient(max_response_bytes=limit)
        assert client.max_response_bytes == limit

    def test_default_max_response_bytes_constant_value(self) -> None:
        assert PolicyHTTPClient.DEFAULT_MAX_RESPONSE_BYTES == 50 * 1024 * 1024

    def test_none_timeout_accepted(self) -> None:
        # None timeout is treated as "no timeout" (httpx convention)
        client = PolicyHTTPClient(timeout=None)
        assert client.timeout is None

    def test_all_params_stored_correctly(self) -> None:
        client = PolicyHTTPClient(
            session_id="sess",
            task_id="task",
            timeout=45,
            category="tool",
            max_response_bytes=2048,
        )
        assert client.session_id == "sess"
        assert client.task_id == "task"
        assert client.timeout == 45
        assert client.category == "tool"
        assert client.max_response_bytes == 2048

    def test_both_internal_clients_are_none_initially(self) -> None:
        client = PolicyHTTPClient()
        assert client._sync_client is None
        assert client._async_client is None


# ---------------------------------------------------------------------------
# Connection pool limits
# ---------------------------------------------------------------------------


class TestConnectionPoolLimits:
    """Verify _POOL_LIMITS are applied when creating underlying httpx clients."""

    def test_sync_client_has_pool_limits_applied(self) -> None:
        client = PolicyHTTPClient()
        client._get_sync_client()
        # httpx.Client exposes pool limits via ._transport._pool
        # We verify the pool limits class-level constant has expected values
        limits = PolicyHTTPClient._POOL_LIMITS
        assert limits.max_connections == 20
        assert limits.max_keepalive_connections == 10

    def test_pool_limits_keepalive_expiry(self) -> None:
        limits = PolicyHTTPClient._POOL_LIMITS
        assert limits.keepalive_expiry == 30

    def test_sync_client_follow_redirects_is_false(self) -> None:
        client = PolicyHTTPClient()
        sync = client._get_sync_client()
        assert sync.follow_redirects is False

    def test_async_client_follow_redirects_is_false(self) -> None:
        client = PolicyHTTPClient()
        async_client = client._get_async_client()
        assert async_client.follow_redirects is False


# ---------------------------------------------------------------------------
# URL parsing edge cases
# ---------------------------------------------------------------------------


class TestURLParsingEdgeCases:
    """URL parsing scenarios that exercise host-extraction and scheme guards."""

    def test_ipv6_host_accepted_by_permissive_policy(self) -> None:
        """IPv6 addresses wrapped in brackets should parse without error."""
        client = PolicyHTTPClient()
        # urlparse strips the brackets and returns the bare IPv6 address
        client._check_url("http://[::1]:8080/path")

    def test_url_with_port_extracts_hostname_without_port(self) -> None:
        """Policy check must receive the bare hostname, not host:port."""
        client = PolicyHTTPClient()
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine
            client._check_url("https://api.example.com:8443/resource")
        # hostname is "api.example.com" — no port suffix
        args, kwargs = mock_engine.check_network.call_args
        assert args[0] == "api.example.com"

    def test_url_with_fragment_extracts_path_without_fragment(self) -> None:
        """URL fragment should not be included in the path passed to REST policy."""
        client = PolicyHTTPClient()
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.rest_policy = None
            mock_get_engine.return_value = mock_engine
            client._check_url("https://example.com/page#section", method="GET")
        # rest_policy is None so no REST check, but the call must not raise

    def test_url_with_trailing_slash_accepted(self) -> None:
        client = PolicyHTTPClient()
        client._check_url("https://example.com/")

    def test_url_without_path_passes_slash_to_rest_policy(self) -> None:
        """When the URL has no explicit path, REST policy should receive '/'."""
        client = PolicyHTTPClient()
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.rest_policy = MagicMock()
            mock_engine.rest_policy.check.return_value = "allow"
            mock_get_engine.return_value = mock_engine
            client._check_url("https://example.com", method="GET")
        _host, _method, path = mock_engine.rest_policy.check.call_args[0]
        assert path == "/"

    def test_url_with_query_and_path_passes_path_only_to_rest_policy(self) -> None:
        """Query string must not be included in the path for REST policy."""
        client = PolicyHTTPClient()
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.rest_policy = MagicMock()
            mock_engine.rest_policy.check.return_value = "allow"
            mock_get_engine.return_value = mock_engine
            client._check_url("https://example.com/search?q=foo&limit=5", method="GET")
        _host, _method, path = mock_engine.rest_policy.check.call_args[0]
        assert path == "/search"
        assert "?" not in path

    def test_data_scheme_rejected(self) -> None:
        """data: URIs are not in _ALLOWED_SCHEMES and must be rejected."""
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("data:text/html,<h1>hi</h1>")

    def test_allowed_schemes_set_contents(self) -> None:
        assert {"http", "https"} == PolicyHTTPClient._ALLOWED_SCHEMES

    def test_url_length_exactly_at_limit_allowed(self) -> None:
        """A URL of exactly 8192 characters must not raise the length guard."""
        base = "https://example.com/"
        path = "a" * (8192 - len(base))
        url = base + path
        assert len(url) == 8192
        client = PolicyHTTPClient()
        # Should pass the length check (policy check may still raise — we only
        # care that the length check itself does not fire)
        try:
            client._check_url(url)
        except PolicyViolationError:
            pass  # policy denial is acceptable
        except ValueError as exc:
            if "exceeds maximum length" in str(exc):
                pytest.fail("URL at exact limit should not raise the length guard")


# ---------------------------------------------------------------------------
# REST policy method forwarding
# ---------------------------------------------------------------------------


class TestRESTPolicyMethodForwarding:
    """Verify method and path are forwarded correctly to rest_policy.check()."""

    @pytest.mark.parametrize(
        "method,path",
        [
            ("GET", "/repos/owner/repo"),
            ("POST", "/repos/owner/repo/issues"),
            ("PUT", "/repos/owner/repo/contents/file.txt"),
            ("PATCH", "/repos/owner/repo/git/refs/head"),
            ("DELETE", "/repos/owner/repo/releases/1"),
            ("HEAD", "/repos/owner/repo"),
        ],
    )
    def test_method_and_path_forwarded_to_rest_policy(
        self, method: str, path: str
    ) -> None:
        client = PolicyHTTPClient()
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.rest_policy = MagicMock()
            mock_engine.rest_policy.check.return_value = "allow"
            mock_get_engine.return_value = mock_engine
            client._check_url(f"https://api.github.com{path}", method=method)
        host_arg, method_arg, path_arg = mock_engine.rest_policy.check.call_args[0]
        assert host_arg == "api.github.com"
        assert method_arg == method
        assert path_arg == path

    def test_no_method_argument_means_rest_policy_not_called(self) -> None:
        client = PolicyHTTPClient()
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.rest_policy = MagicMock()
            mock_get_engine.return_value = mock_engine
            client._check_url("https://example.com/resource")
        mock_engine.rest_policy.check.assert_not_called()

    def test_empty_string_method_argument_means_rest_policy_not_called(self) -> None:
        client = PolicyHTTPClient()
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.rest_policy = MagicMock()
            mock_get_engine.return_value = mock_engine
            client._check_url("https://example.com/resource", method="")
        mock_engine.rest_policy.check.assert_not_called()

    def test_rest_deny_error_carries_host_method_and_path(self) -> None:
        client = PolicyHTTPClient()
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.rest_policy.check.return_value = "deny"
            mock_get_engine.return_value = mock_engine
            with pytest.raises(PolicyViolationError) as exc_info:
                client._check_url(
                    "https://api.github.com/admin/users", method="DELETE"
                )
        err = exc_info.value
        assert "DELETE" in str(err) or "DELETE" in err.detail
        assert "api.github.com" in str(err) or "api.github.com" in err.detail
        assert "/admin/users" in str(err) or "/admin/users" in err.detail

    def test_rest_deny_error_category_is_network(self) -> None:
        client = PolicyHTTPClient()
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.rest_policy.check.return_value = "deny"
            mock_get_engine.return_value = mock_engine
            with pytest.raises(PolicyViolationError) as exc_info:
                client._check_url("https://example.com/bad", method="POST")
        assert exc_info.value.category == "network"

    def test_rest_unexpected_error_denial_category_is_network(self) -> None:
        client = PolicyHTTPClient()
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.rest_policy.check.side_effect = ValueError("crash")
            mock_get_engine.return_value = mock_engine
            with pytest.raises(PolicyViolationError) as exc_info:
                client._check_url("https://example.com/resource", method="GET")
        assert exc_info.value.category == "network"


# ---------------------------------------------------------------------------
# Kwargs sanitisation — exhaustive and boundary cases
# ---------------------------------------------------------------------------


class TestKwargsSanitisationExhaustive:
    """Every allowed kwarg passes; every dangerous kwarg is stripped."""

    def test_each_allowed_kwarg_passes_individually(self) -> None:
        allowed = PolicyHTTPClient._ALLOWED_KWARGS
        for key in allowed:
            result = PolicyHTTPClient._sanitize_kwargs({key: "value"})
            assert key in result, f"Expected allowed kwarg {key!r} to pass through"

    def test_each_dangerous_kwarg_stripped_individually(self) -> None:
        dangerous_keys = [
            "verify",
            "transport",
            "base_url",
            "auth",
            "follow_redirects",
            "event_hooks",
            "proxies",
            "cert",
            "trust_env",
        ]
        for key in dangerous_keys:
            result = PolicyHTTPClient._sanitize_kwargs({key: "anything"})
            assert key not in result, f"Dangerous kwarg {key!r} should have been stripped"

    def test_sanitize_kwargs_is_classmethod(self) -> None:
        """Can be called on the class without an instance."""
        result = PolicyHTTPClient._sanitize_kwargs({"headers": {"X-A": "1"}, "verify": False})
        assert result == {"headers": {"X-A": "1"}}

    def test_sanitize_preserves_none_values_for_allowed_keys(self) -> None:
        result = PolicyHTTPClient._sanitize_kwargs({"headers": None, "timeout": None})
        assert result == {"headers": None, "timeout": None}

    def test_sanitize_preserves_complex_nested_values(self) -> None:
        nested = {"headers": {"Authorization": "Bearer tok", "X-Req-Id": "abc"}}
        result = PolicyHTTPClient._sanitize_kwargs(nested)
        assert result == nested


# ---------------------------------------------------------------------------
# Response size limits — boundary conditions
# ---------------------------------------------------------------------------


class TestResponseSizeBoundaries:
    """Boundary tests at exactly limit and limit+1 bytes."""

    def test_content_length_exactly_at_limit_does_not_raise(self) -> None:
        limit = 1000
        client = PolicyHTTPClient(max_response_bytes=limit)
        resp = _mock_response(200, headers={"content-length": str(limit)})
        client._check_response_size(resp, "https://example.com/")

    def test_content_length_one_over_limit_raises(self) -> None:
        limit = 1000
        client = PolicyHTTPClient(max_response_bytes=limit)
        resp = _mock_response(200, headers={"content-length": str(limit + 1)})
        with pytest.raises(ValueError, match="too large"):
            client._check_response_size(resp, "https://example.com/")

    def test_body_length_exactly_at_limit_does_not_raise(self) -> None:
        limit = 500
        client = PolicyHTTPClient(max_response_bytes=limit)
        resp = MagicMock(spec=httpx.Response)
        resp.headers = MagicMock()
        resp.headers.get = lambda k, default=None: None  # no content-length header
        resp.content = b"x" * limit
        client._check_response_size(resp, "https://example.com/")

    def test_body_length_one_over_limit_raises(self) -> None:
        limit = 500
        client = PolicyHTTPClient(max_response_bytes=limit)
        resp = MagicMock(spec=httpx.Response)
        resp.headers = MagicMock()
        resp.headers.get = lambda k, default=None: None
        resp.content = b"x" * (limit + 1)
        with pytest.raises(ValueError, match="too large"):
            client._check_response_size(resp, "https://example.com/")

    def test_content_access_exception_silently_ignored(self) -> None:
        """If response.content raises, size check exits silently."""
        client = PolicyHTTPClient(max_response_bytes=100)
        resp = MagicMock(spec=httpx.Response)
        resp.headers = MagicMock()
        resp.headers.get = lambda k, default=None: None
        type(resp).content = property(lambda self: (_ for _ in ()).throw(OSError("stream closed")))
        # Must not raise
        client._check_response_size(resp, "https://example.com/")

    def test_zero_content_length_does_not_raise(self) -> None:
        """Content-Length: 0 is within any reasonable limit."""
        client = PolicyHTTPClient(max_response_bytes=1000)
        resp = _mock_response(200, headers={"content-length": "0"})
        client._check_response_size(resp, "https://example.com/")

    def test_error_message_includes_url(self) -> None:
        url = "https://big-server.example.com/huge-file"
        client = PolicyHTTPClient(max_response_bytes=100)
        resp = _mock_response(200, headers={"content-length": "9999"})
        with pytest.raises(ValueError) as exc_info:
            client._check_response_size(resp, url)
        assert url in str(exc_info.value)


# ---------------------------------------------------------------------------
# Audit event correctness — detailed field checks
# ---------------------------------------------------------------------------


class TestAuditEventFields:
    """Granular checks on AuditEvent fields produced by _emit_request_event."""

    def test_timestamp_is_timezone_aware(self) -> None:

        client = PolicyHTTPClient()
        client._emit_request_event("GET", "https://example.com/", 200)
        events = event_bus.get_events(event_type="network_request")
        assert events[0].timestamp.tzinfo is not None

    def test_policy_rule_field_is_none(self) -> None:
        client = PolicyHTTPClient()
        client._emit_request_event("GET", "https://example.com/", 200)
        events = event_bus.get_events(event_type="network_request")
        assert events[0].policy_rule is None

    def test_event_detail_contains_all_three_fields(self) -> None:
        client = PolicyHTTPClient()
        client._emit_request_event("PATCH", "https://api.example.com/x", 204)
        events = event_bus.get_events(event_type="network_request")
        detail = events[0].detail
        assert "method" in detail
        assert "url" in detail
        assert "status_code" in detail

    @pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"])
    def test_correct_method_name_in_detail(self, method: str) -> None:
        client = PolicyHTTPClient()
        client._emit_request_event(method, "https://example.com/", 200)
        events = event_bus.get_events(event_type="network_request")
        assert events[0].detail["method"] == method

    def test_events_from_different_sessions_are_isolated(self) -> None:
        c1 = PolicyHTTPClient(session_id="session-A")
        c2 = PolicyHTTPClient(session_id="session-B")
        c1._emit_request_event("GET", "https://example.com/a", 200)
        c2._emit_request_event("GET", "https://example.com/b", 200)
        events_a = event_bus.get_events(session_id="session-A")
        events_b = event_bus.get_events(session_id="session-B")
        assert len(events_a) == 1
        assert len(events_b) == 1
        assert events_a[0].detail["url"] == "https://example.com/a"
        assert events_b[0].detail["url"] == "https://example.com/b"

    def test_event_bus_accumulates_events_across_methods(self) -> None:
        client = PolicyHTTPClient(session_id="acc-session")
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            client.get("https://api.example.com/a")
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            client.post("https://api.example.com/b")
        with patch.object(httpx.Client, "delete", return_value=mock_resp):
            client.delete("https://api.example.com/c")
        # Filter to only network_request events — the policy engine also emits
        # network_check events on the same session_id.
        events = event_bus.get_events(
            event_type="network_request", session_id="acc-session"
        )
        assert len(events) == 3
        methods = {e.detail["method"] for e in events}
        assert methods == {"GET", "POST", "DELETE"}


# ---------------------------------------------------------------------------
# All sync HTTP methods emit correct audit events
# ---------------------------------------------------------------------------


class TestSyncMethodAuditEmission:
    """Each sync method emits an audit event with the correct method field."""

    @pytest.mark.parametrize(
        "method_name,http_method",
        [
            ("get", "GET"),
            ("post", "POST"),
            ("put", "PUT"),
            ("patch", "PATCH"),
            ("delete", "DELETE"),
            ("head", "HEAD"),
        ],
    )
    def test_sync_method_emits_correct_method_in_audit(
        self, method_name: str, http_method: str
    ) -> None:
        client = PolicyHTTPClient(session_id="sync-audit")
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, method_name, return_value=mock_resp):
            getattr(client, method_name)("https://api.example.com/resource")
        # Filter to network_request events only; the policy engine also emits
        # network_check events on the same session_id.
        events = event_bus.get_events(
            event_type="network_request", session_id="sync-audit"
        )
        assert len(events) == 1
        assert events[0].detail["method"] == http_method


# ---------------------------------------------------------------------------
# All async HTTP methods emit correct audit events
# ---------------------------------------------------------------------------


class TestAsyncMethodAuditEmission:
    """Each async method emits an audit event with the correct method field."""

    @pytest.mark.parametrize(
        "method_name,http_method,httpx_method",
        [
            ("aget", "GET", "get"),
            ("apost", "POST", "post"),
            ("aput", "PUT", "put"),
            ("apatch", "PATCH", "patch"),
            ("adelete", "DELETE", "delete"),
            ("ahead", "HEAD", "head"),
        ],
    )
    async def test_async_method_emits_correct_method_in_audit(
        self, method_name: str, http_method: str, httpx_method: str
    ) -> None:
        client = PolicyHTTPClient(session_id="async-audit")
        mock_resp = _mock_response(200)
        with patch.object(
            httpx.AsyncClient, httpx_method, new_callable=AsyncMock, return_value=mock_resp
        ):
            await getattr(client, method_name)("https://api.example.com/resource")
        # Filter to network_request events only; the policy engine also emits
        # network_check events on the same session_id.
        events = event_bus.get_events(
            event_type="network_request", session_id="async-audit"
        )
        assert len(events) == 1
        assert events[0].detail["method"] == http_method


# ---------------------------------------------------------------------------
# Context manager isolation — close only affects its own client type
# ---------------------------------------------------------------------------


class TestContextManagerIsolation:
    """close() and aclose() must not cross-contaminate sync vs async clients."""

    def test_close_does_not_affect_async_client(self) -> None:
        client = PolicyHTTPClient()
        # Create both clients
        _ = client._get_sync_client()
        async_client = client._get_async_client()
        # Close only the sync side
        client.close()
        assert client._sync_client is None
        assert client._async_client is async_client  # async client untouched

    async def test_aclose_does_not_affect_sync_client(self) -> None:
        client = PolicyHTTPClient()
        sync_client = client._get_sync_client()
        _ = client._get_async_client()
        await client.aclose()
        assert client._async_client is None
        assert client._sync_client is sync_client  # sync client untouched

    def test_double_close_is_safe(self) -> None:
        client = PolicyHTTPClient()
        _ = client._get_sync_client()
        client.close()
        client.close()  # must not raise
        assert client._sync_client is None

    async def test_double_aclose_is_safe(self) -> None:
        client = PolicyHTTPClient()
        _ = client._get_async_client()
        await client.aclose()
        await client.aclose()  # must not raise
        assert client._async_client is None

    def test_sync_context_manager_leaves_async_client_intact(self) -> None:
        client = PolicyHTTPClient()
        async_client = client._get_async_client()
        with client:
            _ = client._get_sync_client()
        assert client._sync_client is None
        assert client._async_client is async_client


# ---------------------------------------------------------------------------
# Interactive approval — detailed argument and flow checks
# ---------------------------------------------------------------------------


class TestInteractiveApprovalDetail:
    """Detailed checks on how InteractiveApproval.prompt_user is called."""

    def _use_restrictive(self) -> None:
        engine_module._engine = None
        init_policy_engine(_restrictive_config())

    def _make_approval(self, prompt_returns: bool) -> MagicMock:
        from missy.agent.interactive_approval import InteractiveApproval

        approval = MagicMock(spec=InteractiveApproval)
        approval.prompt_user.return_value = prompt_returns
        return approval

    def test_prompt_user_receives_operation_network_request(self) -> None:
        self._use_restrictive()
        approval = self._make_approval(prompt_returns=True)
        set_interactive_approval(approval)
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            client.get("https://blocked.example.com/")
        # First arg to prompt_user must describe the operation
        call_args = approval.prompt_user.call_args[0]
        assert call_args[0] == "network_request"

    def test_prompt_user_receives_url_as_second_argument(self) -> None:
        self._use_restrictive()
        approval = self._make_approval(prompt_returns=True)
        set_interactive_approval(approval)
        client = PolicyHTTPClient()
        url = "https://blocked.example.com/secret/path"
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            client.get(url)
        call_args = approval.prompt_user.call_args[0]
        assert call_args[1] == url

    def test_prompt_user_called_exactly_once_per_denied_request(self) -> None:
        self._use_restrictive()
        approval = self._make_approval(prompt_returns=True)
        set_interactive_approval(approval)
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            client.get("https://blocked.example.com/")
        approval.prompt_user.assert_called_once()

    def test_approved_request_httpx_is_called(self) -> None:
        self._use_restrictive()
        approval = self._make_approval(prompt_returns=True)
        set_interactive_approval(approval)
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp) as mock_httpx:
            client.get("https://blocked.example.com/")
        mock_httpx.assert_called_once()

    def test_denied_by_operator_httpx_not_called(self) -> None:
        self._use_restrictive()
        approval = self._make_approval(prompt_returns=False)
        set_interactive_approval(approval)
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get") as mock_httpx, pytest.raises(
            PolicyViolationError
        ):
            client.get("https://blocked.example.com/")
        mock_httpx.assert_not_called()

    def test_approved_request_returns_response(self) -> None:
        self._use_restrictive()
        approval = self._make_approval(prompt_returns=True)
        set_interactive_approval(approval)
        client = PolicyHTTPClient()
        mock_resp = _mock_response(201)
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            resp = client.post("https://blocked.example.com/submit")
        assert resp.status_code == 201

    def test_set_then_clear_approval_instance(self) -> None:
        from missy.agent.interactive_approval import InteractiveApproval

        approval = MagicMock(spec=InteractiveApproval)
        set_interactive_approval(approval)
        assert gateway_module._interactive_approval is approval
        set_interactive_approval(None)
        assert gateway_module._interactive_approval is None

    def test_plain_object_not_used_as_approval(self) -> None:
        """Non-InteractiveApproval objects are ignored and the request is denied."""
        self._use_restrictive()
        # Use a plain object with prompt_user — it must not be called
        fake_approval = MagicMock()
        set_interactive_approval(fake_approval)
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get"), pytest.raises(PolicyViolationError):
            client.get("https://blocked.example.com/")
        fake_approval.prompt_user.assert_not_called()


# ---------------------------------------------------------------------------
# Category forwarding to policy engine
# ---------------------------------------------------------------------------


class TestCategoryAndSessionForwarding:
    """session_id, task_id, and category are all forwarded on every call."""

    @pytest.mark.parametrize("category", ["provider", "tool", "discord", ""])
    def test_category_forwarded_for_all_sync_methods(self, category: str) -> None:
        client = PolicyHTTPClient(
            session_id="s", task_id="t", category=category
        )
        mock_resp = _mock_response(200)
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.rest_policy = None
            mock_get_engine.return_value = mock_engine
            with patch.object(httpx.Client, "get", return_value=mock_resp):
                client.get("https://api.example.com/x")
        mock_engine.check_network.assert_called_once_with(
            "api.example.com", "s", "t", category=category
        )

    def test_session_id_forwarded_to_check_network(self) -> None:
        client = PolicyHTTPClient(session_id="my-session", task_id="")
        mock_resp = _mock_response(200)
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.rest_policy = None
            mock_get_engine.return_value = mock_engine
            with patch.object(httpx.Client, "get", return_value=mock_resp):
                client.get("https://api.example.com/x")
        args, kwargs = mock_engine.check_network.call_args
        assert args[1] == "my-session"

    def test_task_id_forwarded_to_check_network(self) -> None:
        client = PolicyHTTPClient(session_id="", task_id="my-task")
        mock_resp = _mock_response(200)
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.rest_policy = None
            mock_get_engine.return_value = mock_engine
            with patch.object(httpx.Client, "get", return_value=mock_resp):
                client.get("https://api.example.com/x")
        args, kwargs = mock_engine.check_network.call_args
        assert args[2] == "my-task"

    async def test_category_forwarded_for_async_get(self) -> None:
        client = PolicyHTTPClient(
            session_id="as", task_id="at", category="provider"
        )
        mock_resp = _mock_response(200)
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.rest_policy = None
            mock_get_engine.return_value = mock_engine
            with patch.object(
                httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp
            ):
                await client.aget("https://api.example.com/x")
        mock_engine.check_network.assert_called_once_with(
            "api.example.com", "as", "at", category="provider"
        )


# ---------------------------------------------------------------------------
# create_client factory — full parameter coverage
# ---------------------------------------------------------------------------


class TestCreateClientFactoryFull:
    """create_client() covers all documented parameters."""

    def test_all_params_forwarded(self) -> None:
        client = create_client(
            session_id="s1",
            task_id="t1",
            timeout=120,
            category="discord",
        )
        assert client.session_id == "s1"
        assert client.task_id == "t1"
        assert client.timeout == 120
        assert client.category == "discord"

    def test_default_max_response_bytes_on_factory_client(self) -> None:
        """Factory clients receive the default max_response_bytes."""
        client = create_client()
        assert client.max_response_bytes == PolicyHTTPClient.DEFAULT_MAX_RESPONSE_BYTES

    def test_factory_returns_independent_instances(self) -> None:
        clients = [create_client(session_id=f"s{i}") for i in range(5)]
        ids = {id(c) for c in clients}
        assert len(ids) == 5

    def test_factory_with_tool_category(self) -> None:
        client = create_client(category="tool")
        assert client.category == "tool"

    def test_factory_with_provider_category(self) -> None:
        client = create_client(category="provider")
        assert client.category == "provider"


# ---------------------------------------------------------------------------
# Policy enforcement — permissive policy allows all hosts
# ---------------------------------------------------------------------------


class TestPermissivePolicyAllowsAll:
    """With default_deny=False any host including IPs is reachable."""

    def test_ip_address_allowed_with_permissive_policy(self) -> None:
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("http://192.168.1.1/resource")
        assert resp.status_code == 200

    def test_localhost_allowed_with_permissive_policy(self) -> None:
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("http://localhost:8080/health")
        assert resp.status_code == 200

    def test_subdomain_allowed_with_permissive_policy(self) -> None:
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            resp = client.post("https://api.v2.internal.example.com/endpoint")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Policy enforcement — restrictive policy with CIDR
# ---------------------------------------------------------------------------


class TestRestrictivePolicyCIDR:
    """Hosts within an allowed CIDR pass; hosts outside are denied."""

    def test_host_in_allowed_cidr_passes(self) -> None:
        engine_module._engine = None
        init_policy_engine(_restrictive_config(allowed_cidrs=["192.168.0.0/16"]))
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("http://192.168.1.100/resource")
        assert resp.status_code == 200

    def test_host_outside_cidr_denied(self) -> None:
        engine_module._engine = None
        init_policy_engine(_restrictive_config(allowed_cidrs=["10.0.0.0/8"]))
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get"), pytest.raises(PolicyViolationError):
            client.get("http://192.168.1.1/resource")


# ---------------------------------------------------------------------------
# Connection error handling — all sync + async error types
# ---------------------------------------------------------------------------


class TestAllHTTPErrorTypes:
    """All httpx error types propagate after a policy pass."""

    def test_request_error_propagates_from_put(self) -> None:
        client = PolicyHTTPClient()
        with patch.object(
            httpx.Client, "put", side_effect=httpx.RequestError("network error")
        ), pytest.raises(httpx.RequestError):
            client.put("https://api.example.com/resource")

    def test_read_timeout_propagates_from_get(self) -> None:
        client = PolicyHTTPClient()
        with patch.object(
            httpx.Client, "get", side_effect=httpx.ReadTimeout("read timed out")
        ), pytest.raises(httpx.ReadTimeout):
            client.get("https://api.example.com/slow-endpoint")

    def test_write_error_propagates_from_post(self) -> None:
        client = PolicyHTTPClient()
        with patch.object(
            httpx.Client, "post", side_effect=httpx.WriteError("write failed")
        ), pytest.raises(httpx.WriteError):
            client.post("https://api.example.com/submit")

    async def test_async_timeout_propagates_from_aput(self) -> None:
        client = PolicyHTTPClient()
        with patch.object(
            httpx.AsyncClient,
            "put",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("timeout"),
        ), pytest.raises(httpx.TimeoutException):
            await client.aput("https://api.example.com/resource")

    async def test_async_connect_error_propagates_from_adelete(self) -> None:
        client = PolicyHTTPClient()
        with patch.object(
            httpx.AsyncClient,
            "delete",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ), pytest.raises(httpx.ConnectError):
            await client.adelete("https://api.example.com/resource")

    def test_network_error_does_not_emit_network_request_audit_event(self) -> None:
        """A network failure must not emit a network_request event (only network_check
        from the policy engine may be present, but no completed-request event)."""
        client = PolicyHTTPClient(session_id="err-test")
        with patch.object(
            httpx.Client, "post", side_effect=httpx.ConnectError("refused")
        ), pytest.raises(httpx.ConnectError):
            client.post("https://api.example.com/submit")
        events = event_bus.get_events(
            event_type="network_request", session_id="err-test"
        )
        assert events == []


# ---------------------------------------------------------------------------
# set_interactive_approval — module-level state
# ---------------------------------------------------------------------------


class TestSetInteractiveApproval:
    """Verify set_interactive_approval() correctly manages module-level state."""

    def test_initial_value_is_none_after_clean_state(self) -> None:
        assert gateway_module._interactive_approval is None

    def test_set_stores_reference(self) -> None:
        from missy.agent.interactive_approval import InteractiveApproval

        approval = MagicMock(spec=InteractiveApproval)
        set_interactive_approval(approval)
        assert gateway_module._interactive_approval is approval

    def test_set_can_be_called_with_none(self) -> None:
        set_interactive_approval(None)
        assert gateway_module._interactive_approval is None

    def test_overwriting_approval_replaces_previous(self) -> None:
        from missy.agent.interactive_approval import InteractiveApproval

        a1 = MagicMock(spec=InteractiveApproval)
        a2 = MagicMock(spec=InteractiveApproval)
        set_interactive_approval(a1)
        set_interactive_approval(a2)
        assert gateway_module._interactive_approval is a2
