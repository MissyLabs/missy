"""Deep unit tests for missy/gateway/client.py.

Covers PolicyHTTPClient construction, policy enforcement on every HTTP
method (sync and async), REST policy integration, URL validation, response
size limiting, audit event emission, lazy client creation, context manager
lifecycle, the interactive approval flow, and the create_client factory.
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
# Config helpers
# ---------------------------------------------------------------------------


def _permissive_config() -> MissyConfig:
    """Network default_deny=False — every host is reachable."""
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
    rest_policies: list[dict] | None = None,
) -> MissyConfig:
    """Network default_deny=True — only explicit allow lists pass."""
    return MissyConfig(
        network=NetworkPolicy(
            default_deny=True,
            allowed_cidrs=[],
            allowed_domains=allowed_domains or [],
            allowed_hosts=allowed_hosts or [],
            rest_policies=rest_policies or [],
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
    text: str = "ok",
    headers: dict | None = None,
    content: bytes = b"ok",
) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    resp.content = content
    # Build a real-ish headers mapping
    raw_headers = headers or {}
    resp.headers = MagicMock()
    resp.headers.get = lambda k, default=None: raw_headers.get(k, default)
    return resp


# ---------------------------------------------------------------------------
# Autouse fixture: clean policy engine + event bus state for every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    """Reset policy engine to permissive, clear event bus, clear interactive approval."""
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
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for PolicyHTTPClient.__init__()."""

    def test_default_construction(self) -> None:
        client = PolicyHTTPClient()
        assert client.session_id == ""
        assert client.task_id == ""
        assert client.timeout == 30
        assert client.category == ""
        assert client.max_response_bytes == PolicyHTTPClient.DEFAULT_MAX_RESPONSE_BYTES

    def test_custom_session_and_task_id(self) -> None:
        client = PolicyHTTPClient(session_id="my-session", task_id="my-task")
        assert client.session_id == "my-session"
        assert client.task_id == "my-task"

    def test_custom_timeout(self) -> None:
        client = PolicyHTTPClient(timeout=60)
        assert client.timeout == 60

    def test_custom_category(self) -> None:
        client = PolicyHTTPClient(category="provider")
        assert client.category == "provider"

    def test_custom_max_response_bytes(self) -> None:
        client = PolicyHTTPClient(max_response_bytes=1024)
        assert client.max_response_bytes == 1024

    def test_zero_max_response_bytes_uses_default(self) -> None:
        client = PolicyHTTPClient(max_response_bytes=0)
        assert client.max_response_bytes == PolicyHTTPClient.DEFAULT_MAX_RESPONSE_BYTES

    def test_negative_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout must be positive"):
            PolicyHTTPClient(timeout=-1)

    def test_zero_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout must be positive"):
            PolicyHTTPClient(timeout=0)

    def test_sync_client_starts_as_none(self) -> None:
        client = PolicyHTTPClient()
        assert client._sync_client is None

    def test_async_client_starts_as_none(self) -> None:
        client = PolicyHTTPClient()
        assert client._async_client is None

    def test_default_max_response_bytes_is_50mb(self) -> None:
        assert PolicyHTTPClient.DEFAULT_MAX_RESPONSE_BYTES == 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Lazy client creation
# ---------------------------------------------------------------------------


class TestLazyClientCreation:
    def test_get_sync_client_creates_on_first_call(self) -> None:
        client = PolicyHTTPClient()
        assert client._sync_client is None
        sync = client._get_sync_client()
        assert isinstance(sync, httpx.Client)
        assert client._sync_client is sync

    def test_get_sync_client_returns_same_instance(self) -> None:
        client = PolicyHTTPClient()
        a = client._get_sync_client()
        b = client._get_sync_client()
        assert a is b

    def test_get_async_client_creates_on_first_call(self) -> None:
        client = PolicyHTTPClient()
        assert client._async_client is None
        ac = client._get_async_client()
        assert isinstance(ac, httpx.AsyncClient)
        assert client._async_client is ac

    def test_get_async_client_returns_same_instance(self) -> None:
        client = PolicyHTTPClient()
        a = client._get_async_client()
        b = client._get_async_client()
        assert a is b


# ---------------------------------------------------------------------------
# URL validation (_check_url)
# ---------------------------------------------------------------------------


class TestURLValidation:
    """Tests for _check_url covering scheme, host, and length guards."""

    def test_empty_url_raises_value_error(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("")

    def test_ftp_scheme_raises(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("ftp://example.com/file")

    def test_file_scheme_raises(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("file:///etc/passwd")

    def test_javascript_scheme_raises(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("javascript:alert(1)")

    def test_no_host_http_raises(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Cannot determine host"):
            client._check_url("http://")

    def test_https_without_host_raises(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Cannot determine host"):
            client._check_url("https://")

    def test_relative_path_raises(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("/relative/path")

    def test_url_exceeding_max_length_raises(self) -> None:
        client = PolicyHTTPClient()
        long_url = "https://example.com/" + "a" * 8200
        with pytest.raises(ValueError, match="exceeds maximum length"):
            client._check_url(long_url)

    def test_url_at_exact_max_length_raises(self) -> None:
        client = PolicyHTTPClient()
        # Build a URL whose total length is exactly 8193 — one over the limit
        base = "https://x.com/"
        padding = "a" * (8193 - len(base))
        with pytest.raises(ValueError, match="exceeds maximum length"):
            client._check_url(base + padding)

    def test_valid_http_url_does_not_raise_with_permissive_policy(self) -> None:
        client = PolicyHTTPClient()
        # Should complete without error (permissive policy engine)
        client._check_url("http://example.com/path")

    def test_valid_https_url_does_not_raise_with_permissive_policy(self) -> None:
        client = PolicyHTTPClient()
        client._check_url("https://api.example.com/v1/resource")

    def test_url_with_query_string_is_accepted(self) -> None:
        client = PolicyHTTPClient()
        client._check_url("https://example.com/search?q=hello&limit=10")

    def test_url_with_port_is_accepted(self) -> None:
        client = PolicyHTTPClient()
        client._check_url("https://example.com:8443/api")


# ---------------------------------------------------------------------------
# Policy enforcement — synchronous methods
# ---------------------------------------------------------------------------


class TestSyncPolicyEnforcement:
    """Network policy is checked before any httpx I/O for sync methods."""

    def _use_restrictive(
        self,
        allowed_hosts: list[str] | None = None,
        allowed_domains: list[str] | None = None,
    ) -> None:
        engine_module._engine = None
        init_policy_engine(_restrictive_config(allowed_hosts=allowed_hosts, allowed_domains=allowed_domains))

    def test_get_blocked_by_default_deny(self) -> None:
        self._use_restrictive()
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get") as mock_get:
            with pytest.raises(PolicyViolationError):
                client.get("https://blocked.example.com/")
            mock_get.assert_not_called()

    def test_get_allowed_when_host_in_allowlist(self) -> None:
        self._use_restrictive(allowed_hosts=["allowed.example.com"])
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp) as mock_get:
            resp = client.get("https://allowed.example.com/resource")
        assert resp.status_code == 200
        mock_get.assert_called_once()

    def test_post_blocked_by_default_deny(self) -> None:
        self._use_restrictive()
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "post") as mock_post:
            with pytest.raises(PolicyViolationError):
                client.post("https://blocked.example.com/submit")
            mock_post.assert_not_called()

    def test_post_allowed_when_domain_in_allowlist(self) -> None:
        # Wildcard pattern "*.example.com" matches "api.example.com" subdomains.
        self._use_restrictive(allowed_domains=["*.example.com"])
        client = PolicyHTTPClient()
        mock_resp = _mock_response(201)
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            resp = client.post("https://api.example.com/items", json={"x": 1})
        assert resp.status_code == 201

    def test_put_blocked_by_default_deny(self) -> None:
        self._use_restrictive()
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "put") as mock_put:
            with pytest.raises(PolicyViolationError):
                client.put("https://blocked.example.com/resource/1")
            mock_put.assert_not_called()

    def test_put_allowed(self) -> None:
        self._use_restrictive(allowed_hosts=["api.example.com"])
        client = PolicyHTTPClient()
        mock_resp = _mock_response(204)
        with patch.object(httpx.Client, "put", return_value=mock_resp):
            resp = client.put("https://api.example.com/resource/1")
        assert resp.status_code == 204

    def test_delete_blocked_by_default_deny(self) -> None:
        self._use_restrictive()
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "delete") as mock_del:
            with pytest.raises(PolicyViolationError):
                client.delete("https://blocked.example.com/resource/1")
            mock_del.assert_not_called()

    def test_delete_allowed(self) -> None:
        self._use_restrictive(allowed_hosts=["api.example.com"])
        client = PolicyHTTPClient()
        mock_resp = _mock_response(204)
        with patch.object(httpx.Client, "delete", return_value=mock_resp):
            resp = client.delete("https://api.example.com/resource/1")
        assert resp.status_code == 204

    def test_patch_blocked_by_default_deny(self) -> None:
        self._use_restrictive()
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "patch") as mock_patch:
            with pytest.raises(PolicyViolationError):
                client.patch("https://blocked.example.com/resource/1")
            mock_patch.assert_not_called()

    def test_patch_allowed(self) -> None:
        self._use_restrictive(allowed_hosts=["api.example.com"])
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "patch", return_value=mock_resp):
            resp = client.patch("https://api.example.com/resource/1")
        assert resp.status_code == 200

    def test_head_blocked_by_default_deny(self) -> None:
        self._use_restrictive()
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "head") as mock_head:
            with pytest.raises(PolicyViolationError):
                client.head("https://blocked.example.com/")
            mock_head.assert_not_called()

    def test_head_allowed(self) -> None:
        self._use_restrictive(allowed_hosts=["api.example.com"])
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "head", return_value=mock_resp):
            resp = client.head("https://api.example.com/resource")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Policy enforcement — asynchronous methods
# ---------------------------------------------------------------------------


class TestAsyncPolicyEnforcement:
    """Network policy is checked before any httpx I/O for async methods."""

    def _use_restrictive(self, allowed_hosts: list[str] | None = None) -> None:
        engine_module._engine = None
        init_policy_engine(_restrictive_config(allowed_hosts=allowed_hosts))

    async def test_aget_blocked(self) -> None:
        self._use_restrictive()
        client = PolicyHTTPClient()
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            with pytest.raises(PolicyViolationError):
                await client.aget("https://blocked.example.com/")
            mock_get.assert_not_called()

    async def test_aget_allowed(self) -> None:
        self._use_restrictive(allowed_hosts=["api.example.com"])
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp):
            resp = await client.aget("https://api.example.com/")
        assert resp.status_code == 200

    async def test_apost_blocked(self) -> None:
        self._use_restrictive()
        client = PolicyHTTPClient()
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            with pytest.raises(PolicyViolationError):
                await client.apost("https://blocked.example.com/submit")
            mock_post.assert_not_called()

    async def test_apost_allowed(self) -> None:
        self._use_restrictive(allowed_hosts=["api.example.com"])
        client = PolicyHTTPClient()
        mock_resp = _mock_response(201)
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
            resp = await client.apost("https://api.example.com/items")
        assert resp.status_code == 201

    async def test_aput_blocked(self) -> None:
        self._use_restrictive()
        client = PolicyHTTPClient()
        with patch.object(httpx.AsyncClient, "put", new_callable=AsyncMock) as mock_put:
            with pytest.raises(PolicyViolationError):
                await client.aput("https://blocked.example.com/resource")
            mock_put.assert_not_called()

    async def test_aput_allowed(self) -> None:
        self._use_restrictive(allowed_hosts=["api.example.com"])
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.AsyncClient, "put", new_callable=AsyncMock, return_value=mock_resp):
            resp = await client.aput("https://api.example.com/resource")
        assert resp.status_code == 200

    async def test_adelete_blocked(self) -> None:
        self._use_restrictive()
        client = PolicyHTTPClient()
        with patch.object(httpx.AsyncClient, "delete", new_callable=AsyncMock) as mock_del:
            with pytest.raises(PolicyViolationError):
                await client.adelete("https://blocked.example.com/resource")
            mock_del.assert_not_called()

    async def test_adelete_allowed(self) -> None:
        self._use_restrictive(allowed_hosts=["api.example.com"])
        client = PolicyHTTPClient()
        mock_resp = _mock_response(204)
        with patch.object(httpx.AsyncClient, "delete", new_callable=AsyncMock, return_value=mock_resp):
            resp = await client.adelete("https://api.example.com/resource")
        assert resp.status_code == 204

    async def test_apatch_blocked(self) -> None:
        self._use_restrictive()
        client = PolicyHTTPClient()
        with patch.object(httpx.AsyncClient, "patch", new_callable=AsyncMock) as mock_patch:
            with pytest.raises(PolicyViolationError):
                await client.apatch("https://blocked.example.com/resource")
            mock_patch.assert_not_called()

    async def test_apatch_allowed(self) -> None:
        self._use_restrictive(allowed_hosts=["api.example.com"])
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.AsyncClient, "patch", new_callable=AsyncMock, return_value=mock_resp):
            resp = await client.apatch("https://api.example.com/resource")
        assert resp.status_code == 200

    async def test_ahead_blocked(self) -> None:
        self._use_restrictive()
        client = PolicyHTTPClient()
        with patch.object(httpx.AsyncClient, "head", new_callable=AsyncMock) as mock_head:
            with pytest.raises(PolicyViolationError):
                await client.ahead("https://blocked.example.com/resource")
            mock_head.assert_not_called()

    async def test_ahead_allowed(self) -> None:
        self._use_restrictive(allowed_hosts=["api.example.com"])
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.AsyncClient, "head", new_callable=AsyncMock, return_value=mock_resp):
            resp = await client.ahead("https://api.example.com/resource")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# REST policy enforcement
# ---------------------------------------------------------------------------


class TestRESTPolicyEnforcement:
    """Tests for _check_rest_policy() — L7 method+path control."""

    @patch("missy.gateway.client.get_policy_engine")
    def test_rest_policy_deny_raises(self, mock_get_engine: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_engine.check_network.return_value = True
        mock_engine.rest_policy.check.return_value = "deny"
        mock_get_engine.return_value = mock_engine

        client = PolicyHTTPClient()
        with pytest.raises(PolicyViolationError, match="REST policy denied"):
            client._check_url("https://api.example.com/secret", method="DELETE")

    @patch("missy.gateway.client.get_policy_engine")
    def test_rest_policy_allow_does_not_raise(self, mock_get_engine: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_engine.check_network.return_value = True
        mock_engine.rest_policy.check.return_value = "allow"
        mock_get_engine.return_value = mock_engine

        client = PolicyHTTPClient()
        # Must not raise
        client._check_url("https://api.example.com/repos/foo", method="GET")

    @patch("missy.gateway.client.get_policy_engine")
    def test_no_rest_policy_attribute_does_not_raise(self, mock_get_engine: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_engine.check_network.return_value = True
        mock_engine.rest_policy = None
        mock_get_engine.return_value = mock_engine

        client = PolicyHTTPClient()
        # Must not raise when engine has no REST policy configured
        client._check_url("https://api.example.com/anything", method="POST")

    @patch("missy.gateway.client.get_policy_engine")
    def test_rest_policy_exception_denies_request(self, mock_get_engine: MagicMock) -> None:
        """Non-PolicyViolationError from REST policy check denies request (fail-closed)."""
        mock_engine = MagicMock()
        mock_engine.check_network.return_value = True
        mock_engine.rest_policy.check.side_effect = RuntimeError("parser broken")
        mock_get_engine.return_value = mock_engine

        client = PolicyHTTPClient()
        # Must raise — fail-closed denies the request
        with pytest.raises(PolicyViolationError):
            client._check_url("https://api.example.com/resource", method="GET")

    @patch("missy.gateway.client.get_policy_engine")
    def test_rest_policy_error_detail_includes_method_and_path(
        self, mock_get_engine: MagicMock
    ) -> None:
        mock_engine = MagicMock()
        mock_engine.check_network.return_value = True
        mock_engine.rest_policy.check.return_value = "deny"
        mock_get_engine.return_value = mock_engine

        client = PolicyHTTPClient()
        with pytest.raises(PolicyViolationError) as exc_info:
            client._check_url("https://api.example.com/admin/keys", method="DELETE")

        err = exc_info.value
        assert "DELETE" in err.detail
        assert "/admin/keys" in err.detail

    @patch("missy.gateway.client.get_policy_engine")
    def test_no_method_skips_rest_policy(self, mock_get_engine: MagicMock) -> None:
        """When method is empty string, REST policy check is bypassed."""
        mock_engine = MagicMock()
        mock_engine.check_network.return_value = True
        mock_get_engine.return_value = mock_engine

        client = PolicyHTTPClient()
        client._check_url("https://api.example.com/resource")  # no method arg
        mock_engine.rest_policy.check.assert_not_called()


# ---------------------------------------------------------------------------
# Kwargs sanitisation
# ---------------------------------------------------------------------------


class TestKwargsSanitisation:
    """Tests for _sanitize_kwargs() — only safe kwargs pass through to httpx."""

    def test_allowed_kwargs_pass_through(self) -> None:
        safe = {
            "headers": {"X-Token": "abc"},
            "params": {"q": "test"},
            "data": b"raw",
            "json": {"key": "val"},
            "content": b"bytes",
            "cookies": {"session": "xyz"},
            "timeout": 10,
            "files": {"upload": b"data"},
            "extensions": {},
        }
        result = PolicyHTTPClient._sanitize_kwargs(safe)
        assert result == safe

    def test_dangerous_kwargs_are_stripped(self) -> None:
        dangerous = {
            "verify": False,          # would bypass TLS
            "transport": MagicMock(), # would bypass policy layer
            "base_url": "http://evil.com",
            "auth": ("user", "pass"),
            "follow_redirects": True,
        }
        result = PolicyHTTPClient._sanitize_kwargs(dangerous)
        assert result == {}

    def test_mixed_kwargs_only_safe_ones_pass(self) -> None:
        kwargs = {
            "headers": {"Accept": "application/json"},
            "verify": False,  # dangerous
            "json": {"payload": 42},
            "transport": MagicMock(),  # dangerous
        }
        result = PolicyHTTPClient._sanitize_kwargs(kwargs)
        assert set(result.keys()) == {"headers", "json"}

    def test_empty_kwargs_returns_empty(self) -> None:
        assert PolicyHTTPClient._sanitize_kwargs({}) == {}

    def test_get_forwards_safe_kwargs_to_httpx(self) -> None:
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp) as mock_get:
            client.get(
                "https://api.example.com/resource",
                headers={"X-Custom": "val"},
                params={"limit": 10},
                verify=False,      # should be stripped
                transport=None,    # should be stripped
            )
        _, call_kwargs = mock_get.call_args
        assert "headers" in call_kwargs
        assert "params" in call_kwargs
        assert "verify" not in call_kwargs
        assert "transport" not in call_kwargs


# ---------------------------------------------------------------------------
# Response size limits
# ---------------------------------------------------------------------------


class TestResponseSizeLimits:
    """Tests for _check_response_size()."""

    def test_response_within_limit_does_not_raise(self) -> None:
        client = PolicyHTTPClient(max_response_bytes=1000)
        resp = _mock_response(200, headers={"content-length": "500"})
        client._check_response_size(resp, "https://example.com/")  # must not raise

    def test_content_length_header_exceeds_limit_raises(self) -> None:
        client = PolicyHTTPClient(max_response_bytes=1000)
        resp = _mock_response(200, headers={"content-length": "2000"})
        with pytest.raises(ValueError, match="too large"):
            client._check_response_size(resp, "https://example.com/")

    def test_no_content_length_checks_body_within_limit(self) -> None:
        client = PolicyHTTPClient(max_response_bytes=1000)
        resp = MagicMock(spec=httpx.Response)
        resp.headers = MagicMock()
        resp.headers.get = lambda k, default=None: None  # no content-length
        resp.content = b"x" * 500
        client._check_response_size(resp, "https://example.com/")  # must not raise

    def test_no_content_length_body_exceeds_limit_raises(self) -> None:
        client = PolicyHTTPClient(max_response_bytes=100)
        resp = MagicMock(spec=httpx.Response)
        resp.headers = MagicMock()
        resp.headers.get = lambda k, default=None: None
        resp.content = b"x" * 200
        with pytest.raises(ValueError, match="too large"):
            client._check_response_size(resp, "https://example.com/")

    def test_none_headers_is_safe(self) -> None:
        client = PolicyHTTPClient()
        resp = MagicMock()
        resp.headers = None
        client._check_response_size(resp, "https://example.com/")  # must not raise

    def test_malformed_content_length_treated_as_zero(self) -> None:
        """Non-integer Content-Length header is treated as 0 — no size error."""
        client = PolicyHTTPClient(max_response_bytes=1000)
        resp = _mock_response(200, headers={"content-length": "not-a-number"})
        client._check_response_size(resp, "https://example.com/")  # must not raise

    def test_get_blocks_oversized_response_via_content_length(self) -> None:
        client = PolicyHTTPClient(max_response_bytes=100)
        resp = _mock_response(200, headers={"content-length": "200"})
        with patch.object(httpx.Client, "get", return_value=resp), pytest.raises(ValueError, match="too large"):
            client.get("https://api.example.com/big-resource")


# ---------------------------------------------------------------------------
# Audit event emission
# ---------------------------------------------------------------------------


class TestAuditEventEmission:
    """Tests for _emit_request_event() and its integration with the event bus."""

    def test_emit_stores_event_with_correct_type(self) -> None:
        client = PolicyHTTPClient()
        client._emit_request_event("GET", "https://example.com/", 200)
        events = event_bus.get_events(event_type="network_request")
        assert len(events) == 1

    def test_emit_stores_method_in_detail(self) -> None:
        client = PolicyHTTPClient()
        client._emit_request_event("DELETE", "https://example.com/resource", 204)
        events = event_bus.get_events(event_type="network_request")
        assert events[0].detail["method"] == "DELETE"

    def test_emit_stores_url_in_detail(self) -> None:
        client = PolicyHTTPClient()
        url = "https://api.example.com/v1/items"
        client._emit_request_event("POST", url, 201)
        events = event_bus.get_events(event_type="network_request")
        assert events[0].detail["url"] == url

    def test_emit_stores_status_code_in_detail(self) -> None:
        client = PolicyHTTPClient()
        client._emit_request_event("GET", "https://example.com/", 404)
        events = event_bus.get_events(event_type="network_request")
        assert events[0].detail["status_code"] == 404

    def test_emit_carries_session_id(self) -> None:
        client = PolicyHTTPClient(session_id="my-session")
        client._emit_request_event("GET", "https://example.com/", 200)
        events = event_bus.get_events(event_type="network_request")
        assert events[0].session_id == "my-session"

    def test_emit_carries_task_id(self) -> None:
        client = PolicyHTTPClient(task_id="my-task")
        client._emit_request_event("GET", "https://example.com/", 200)
        events = event_bus.get_events(event_type="network_request")
        assert events[0].task_id == "my-task"

    def test_emit_category_is_network(self) -> None:
        client = PolicyHTTPClient()
        client._emit_request_event("GET", "https://example.com/", 200)
        events = event_bus.get_events(event_type="network_request")
        assert events[0].category == "network"

    def test_emit_result_is_allow(self) -> None:
        client = PolicyHTTPClient()
        client._emit_request_event("GET", "https://example.com/", 200)
        events = event_bus.get_events(event_type="network_request")
        assert events[0].result == "allow"

    def test_successful_get_emits_event(self) -> None:
        client = PolicyHTTPClient(session_id="s1", task_id="t1")
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            client.get("https://api.example.com/resource")
        events = event_bus.get_events(event_type="network_request")
        assert len(events) == 1
        assert events[0].detail["method"] == "GET"
        assert events[0].detail["status_code"] == 200

    def test_successful_post_emits_event(self) -> None:
        client = PolicyHTTPClient()
        mock_resp = _mock_response(201)
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            client.post("https://api.example.com/items")
        events = event_bus.get_events(event_type="network_request")
        assert events[0].detail["method"] == "POST"

    def test_denied_request_does_not_emit_network_request_event(self) -> None:
        """Policy-denied requests raise before emitting an audit event."""
        engine_module._engine = None
        init_policy_engine(_restrictive_config())
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get"), pytest.raises(PolicyViolationError):
            client.get("https://denied.example.com/")
        events = event_bus.get_events(event_type="network_request")
        assert events == []

    def test_multiple_requests_emit_multiple_events(self) -> None:
        client = PolicyHTTPClient(session_id="multi")
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            client.get("https://api.example.com/a")
            client.get("https://api.example.com/b")
            client.get("https://api.example.com/c")
        events = event_bus.get_events(event_type="network_request", session_id="multi")
        assert len(events) == 3

    async def test_async_get_emits_event(self) -> None:
        client = PolicyHTTPClient(session_id="async-s")
        mock_resp = _mock_response(200)
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp):
            await client.aget("https://api.example.com/resource")
        events = event_bus.get_events(event_type="network_request")
        assert len(events) == 1
        assert events[0].detail["method"] == "GET"
        assert events[0].session_id == "async-s"


# ---------------------------------------------------------------------------
# Connection error handling
# ---------------------------------------------------------------------------


class TestConnectionErrorHandling:
    """httpx exceptions propagate unmodified after a successful policy check."""

    def test_connect_error_propagates_from_get(self) -> None:
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get", side_effect=httpx.ConnectError("refused")), pytest.raises(httpx.ConnectError):
            client.get("https://api.example.com/")

    def test_timeout_error_propagates_from_post(self) -> None:
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "post", side_effect=httpx.TimeoutException("timed out")), pytest.raises(httpx.TimeoutException):
            client.post("https://api.example.com/items")

    def test_http_status_error_propagates_from_delete(self) -> None:
        client = PolicyHTTPClient()
        exc = httpx.HTTPStatusError("500 error", request=MagicMock(), response=MagicMock())
        with patch.object(httpx.Client, "delete", side_effect=exc), pytest.raises(httpx.HTTPStatusError):
            client.delete("https://api.example.com/resource/1")

    def test_connection_error_does_not_emit_audit_event(self) -> None:
        """No network_request event should be emitted when the request never completes."""
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get", side_effect=httpx.ConnectError("refused")), pytest.raises(httpx.ConnectError):
            client.get("https://api.example.com/")
        assert event_bus.get_events(event_type="network_request") == []

    async def test_async_connect_error_propagates(self) -> None:
        client = PolicyHTTPClient()
        with patch.object(
            httpx.AsyncClient, "get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("async refused"),
        ), pytest.raises(httpx.ConnectError):
            await client.aget("https://api.example.com/")


# ---------------------------------------------------------------------------
# Context manager lifecycle
# ---------------------------------------------------------------------------


class TestSyncContextManager:
    """Tests for PolicyHTTPClient.__enter__ / __exit__."""

    def test_enter_returns_self(self) -> None:
        client = PolicyHTTPClient()
        assert client.__enter__() is client
        client.__exit__(None, None, None)

    def test_exit_closes_sync_client(self) -> None:
        with PolicyHTTPClient() as client:
            _ = client._get_sync_client()
            assert client._sync_client is not None
        assert client._sync_client is None

    def test_close_idempotent_when_client_never_created(self) -> None:
        client = PolicyHTTPClient()
        client.close()  # must not raise
        assert client._sync_client is None

    def test_close_after_request_sets_sync_client_to_none(self) -> None:
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            client.get("https://api.example.com/")
        assert client._sync_client is not None
        client.close()
        assert client._sync_client is None

    def test_full_request_inside_sync_context_manager(self) -> None:
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp), PolicyHTTPClient() as client:
            resp = client.get("https://api.example.com/ping")
        assert resp.status_code == 200
        assert client._sync_client is None


class TestAsyncContextManager:
    """Tests for PolicyHTTPClient.__aenter__ / __aexit__."""

    async def test_aenter_returns_self(self) -> None:
        client = PolicyHTTPClient()
        result = await client.__aenter__()
        assert result is client
        await client.__aexit__(None, None, None)

    async def test_aexit_closes_async_client(self) -> None:
        async with PolicyHTTPClient() as client:
            _ = client._get_async_client()
            assert client._async_client is not None
        assert client._async_client is None

    async def test_aclose_idempotent_when_client_never_created(self) -> None:
        client = PolicyHTTPClient()
        await client.aclose()  # must not raise
        assert client._async_client is None

    async def test_aclose_after_request_sets_async_client_to_none(self) -> None:
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp):
            await client.aget("https://api.example.com/")
        inner = client._async_client
        assert inner is not None
        await client.aclose()
        assert client._async_client is None

    async def test_full_request_inside_async_context_manager(self) -> None:
        mock_resp = _mock_response(200)
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp):
            async with PolicyHTTPClient() as client:
                resp = await client.aget("https://api.example.com/ping")
        assert resp.status_code == 200
        assert client._async_client is None


# ---------------------------------------------------------------------------
# Interactive approval flow
# ---------------------------------------------------------------------------


class TestInteractiveApprovalFlow:
    """Tests for the interactive approval integration in _check_url."""

    def _make_approval(self, prompt_returns: bool) -> MagicMock:
        from missy.agent.interactive_approval import InteractiveApproval

        approval = MagicMock(spec=InteractiveApproval)
        approval.prompt_user.return_value = prompt_returns
        return approval

    def _use_restrictive(self) -> None:
        engine_module._engine = None
        init_policy_engine(_restrictive_config())

    def test_operator_approves_allows_request(self) -> None:
        self._use_restrictive()
        approval = self._make_approval(prompt_returns=True)
        set_interactive_approval(approval)

        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            # With approval granted, must not raise
            resp = client.get("https://operator-approved.example.com/")
        assert resp.status_code == 200
        approval.prompt_user.assert_called_once()

    def test_operator_denies_raises_policy_violation(self) -> None:
        self._use_restrictive()
        approval = self._make_approval(prompt_returns=False)
        set_interactive_approval(approval)

        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get"), pytest.raises(PolicyViolationError):
            client.get("https://denied.example.com/")
        approval.prompt_user.assert_called_once()

    def test_no_interactive_approval_instance_raises_immediately(self) -> None:
        self._use_restrictive()
        # _interactive_approval is already None from fixture
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get") as mock_get, pytest.raises(PolicyViolationError):
            client.get("https://denied.example.com/")
        mock_get.assert_not_called()

    def test_non_interactive_approval_object_raises(self) -> None:
        """A non-InteractiveApproval object in _interactive_approval is not used."""
        self._use_restrictive()
        # Set a plain MagicMock (not an InteractiveApproval instance)
        set_interactive_approval(MagicMock())

        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get") as mock_get, pytest.raises(PolicyViolationError):
            client.get("https://denied.example.com/")
        mock_get.assert_not_called()

    def test_set_interactive_approval_stores_instance(self) -> None:
        from missy.agent.interactive_approval import InteractiveApproval

        approval = MagicMock(spec=InteractiveApproval)
        set_interactive_approval(approval)
        assert gateway_module._interactive_approval is approval

    def test_set_interactive_approval_none_clears_instance(self) -> None:
        from missy.agent.interactive_approval import InteractiveApproval

        set_interactive_approval(MagicMock(spec=InteractiveApproval))
        set_interactive_approval(None)
        assert gateway_module._interactive_approval is None


# ---------------------------------------------------------------------------
# Category forwarding to the policy engine
# ---------------------------------------------------------------------------


class TestCategoryForwarding:
    """The category param is forwarded to get_policy_engine().check_network()."""

    @pytest.mark.parametrize("category", ["provider", "tool", "discord", ""])
    def test_category_forwarded_on_sync_get(self, category: str) -> None:
        client = PolicyHTTPClient(category=category)
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine
            mock_resp = _mock_response(200)
            with patch.object(httpx.Client, "get", return_value=mock_resp):
                client.get("https://api.example.com/")
        mock_engine.check_network.assert_called_once_with(
            "api.example.com", "", "", category=category
        )

    async def test_category_forwarded_on_async_post(self) -> None:
        client = PolicyHTTPClient(category="discord")
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine
            mock_resp = _mock_response(200)
            with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp):
                await client.apost("https://discord.com/api/webhooks/x")
        mock_engine.check_network.assert_called_once_with(
            "discord.com", "", "", category="discord"
        )

    def test_session_and_task_forwarded_to_policy_engine(self) -> None:
        client = PolicyHTTPClient(session_id="sess-xyz", task_id="task-abc")
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine
            mock_resp = _mock_response(200)
            with patch.object(httpx.Client, "get", return_value=mock_resp):
                client.get("https://api.example.com/")
        mock_engine.check_network.assert_called_once_with(
            "api.example.com", "sess-xyz", "task-abc", category=""
        )


# ---------------------------------------------------------------------------
# create_client factory
# ---------------------------------------------------------------------------


class TestCreateClientFactory:
    """Tests for the create_client() module-level factory."""

    def test_returns_policy_http_client_instance(self) -> None:
        client = create_client()
        assert isinstance(client, PolicyHTTPClient)

    def test_default_parameters(self) -> None:
        client = create_client()
        assert client.session_id == ""
        assert client.task_id == ""
        assert client.timeout == 30
        assert client.category == ""

    def test_custom_session_id(self) -> None:
        client = create_client(session_id="abc")
        assert client.session_id == "abc"

    def test_custom_task_id(self) -> None:
        client = create_client(task_id="xyz")
        assert client.task_id == "xyz"

    def test_custom_timeout(self) -> None:
        client = create_client(timeout=60)
        assert client.timeout == 60

    def test_custom_category(self) -> None:
        client = create_client(category="provider")
        assert client.category == "provider"

    def test_each_call_returns_distinct_instance(self) -> None:
        a = create_client()
        b = create_client()
        assert a is not b
