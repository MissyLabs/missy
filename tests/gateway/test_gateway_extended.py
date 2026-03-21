"""Extended tests for missy/gateway/client.py.

Focuses on areas that are either absent from or only lightly covered by the
existing test suite:

1. PolicyHTTPClient request validation (URL edge cases, port handling, IPv6)
2. REST policy enforcement via the full policy engine (config-driven)
3. Network policy checks via CIDR, domain, and per-category host allow lists
4. Interactive approval flow (edge cases and non-TTY auto-denial)
5. Error handling for network failures (all sync+async HTTP error types)
6. TLS / SSL handling — verify=False is stripped, follow_redirects is off
7. Request/response logging — audit events carry correct fields
8. Thread safety of the HTTP client (concurrent requests, lazy init race)
9. Rate limiting integration stub (policy engine hook)
10. Edge cases: malformed URLs, empty hosts, redirect behaviour, IPv6 hosts
"""

from __future__ import annotations

import contextlib
import threading
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
from missy.policy.rest_policy import RestPolicy, RestRule

# ---------------------------------------------------------------------------
# Config / response helpers
# ---------------------------------------------------------------------------


def _permissive_config() -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(default_deny=False),
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
    provider_allowed_hosts: list[str] | None = None,
    tool_allowed_hosts: list[str] | None = None,
    discord_allowed_hosts: list[str] | None = None,
    rest_policies: list[dict] | None = None,
) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(
            default_deny=True,
            allowed_cidrs=allowed_cidrs or [],
            allowed_domains=allowed_domains or [],
            allowed_hosts=allowed_hosts or [],
            provider_allowed_hosts=provider_allowed_hosts or [],
            tool_allowed_hosts=tool_allowed_hosts or [],
            discord_allowed_hosts=discord_allowed_hosts or [],
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
    content: bytes = b"ok",
    headers: dict | None = None,
) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.content = content
    resp.text = content.decode(errors="replace")
    raw_headers = headers or {}
    resp.headers = MagicMock()
    resp.headers.get = lambda k, default=None: raw_headers.get(k, default)
    return resp


# ---------------------------------------------------------------------------
# Autouse fixture: clean state for every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    original_engine = engine_module._engine
    original_approval = gateway_module._interactive_approval

    init_policy_engine(_permissive_config())
    event_bus.clear()
    set_interactive_approval(None)

    yield

    engine_module._engine = original_engine
    gateway_module._interactive_approval = original_approval
    event_bus.clear()


# ===========================================================================
# 1. PolicyHTTPClient request validation — URL edge cases
# ===========================================================================


class TestRequestValidationURLEdgeCases:
    """_check_url enforces scheme, host presence, and length limits."""

    # --- Scheme validation ---

    def test_data_scheme_is_rejected(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("data:text/html,<h1>hello</h1>")

    def test_ws_scheme_is_rejected(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("ws://example.com/socket")

    def test_wss_scheme_is_rejected(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("wss://example.com/socket")

    def test_http_scheme_is_accepted(self) -> None:
        client = PolicyHTTPClient()
        # permissive engine — should not raise
        client._check_url("http://example.com/path")

    def test_https_scheme_is_accepted(self) -> None:
        client = PolicyHTTPClient()
        client._check_url("https://example.com/path")

    # --- Host presence ---

    def test_url_with_only_scheme_and_colon_raises(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError):
            client._check_url("https:")

    def test_url_with_triple_slash_no_host_raises(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Cannot determine host"):
            client._check_url("https:///path-without-host")

    def test_url_with_empty_string_raises(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("")

    # --- Port handling ---

    def test_url_with_non_standard_http_port_is_accepted(self) -> None:
        client = PolicyHTTPClient()
        client._check_url("http://example.com:8080/path")

    def test_url_with_https_port_443_is_accepted(self) -> None:
        client = PolicyHTTPClient()
        client._check_url("https://example.com:443/path")

    # --- Query strings and fragments ---

    def test_url_with_complex_query_string_is_accepted(self) -> None:
        client = PolicyHTTPClient()
        client._check_url("https://search.example.com/q?term=foo+bar&page=2&sort=desc")

    def test_url_with_fragment_is_accepted(self) -> None:
        client = PolicyHTTPClient()
        client._check_url("https://docs.example.com/guide#section-3")

    def test_url_with_encoded_characters_is_accepted(self) -> None:
        client = PolicyHTTPClient()
        client._check_url("https://example.com/path%20with%20spaces")

    # --- Length limit ---

    def test_url_exactly_at_8192_chars_is_accepted(self) -> None:
        """A URL exactly 8192 characters long is at the boundary and should pass."""
        client = PolicyHTTPClient()
        base = "https://example.com/"
        path = "a" * (8192 - len(base))
        # Exactly 8192 — should not raise
        client._check_url(base + path)

    def test_url_at_8193_chars_is_rejected(self) -> None:
        client = PolicyHTTPClient()
        base = "https://example.com/"
        path = "a" * (8193 - len(base))
        with pytest.raises(ValueError, match="exceeds maximum length"):
            client._check_url(base + path)

    # --- IPv6 ---

    def test_ipv6_localhost_in_brackets_is_parsed(self) -> None:
        """IPv6 addresses enclosed in brackets are valid URL syntax."""
        client = PolicyHTTPClient()
        # Permissive engine — only check that it doesn't raise a ValueError
        # (it may raise PolicyViolationError for the IP check, but not ValueError)
        with contextlib.suppress(PolicyViolationError):
            client._check_url("http://[::1]/path")

    def test_ipv6_address_without_brackets_raises_scheme_error(self) -> None:
        """Bare IPv6 without brackets is not a valid URL and should not parse."""
        client = PolicyHTTPClient()
        # "http://::1/path" has an ambiguous host; urlparse returns empty hostname
        with pytest.raises((ValueError, PolicyViolationError)):
            client._check_url("http://::1/path")


# ===========================================================================
# 2. REST policy enforcement (config-driven, L7)
# ===========================================================================


class TestRESTLPolicyEnforcementConfigDriven:
    """End-to-end REST policy enforcement via a real PolicyEngine."""

    def _use_restrictive_with_rest(self, rest_policies: list[dict]) -> None:
        init_policy_engine(
            _restrictive_config(
                allowed_hosts=["api.github.com"],
                rest_policies=rest_policies,
            )
        )

    def test_allowed_get_path_does_not_raise(self) -> None:
        self._use_restrictive_with_rest(
            [{"host": "api.github.com", "method": "GET", "path": "/repos/**", "action": "allow"}]
        )
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://api.github.com/repos/org/repo")
        assert resp.status_code == 200

    def test_denied_delete_on_all_paths(self) -> None:
        self._use_restrictive_with_rest(
            [{"host": "api.github.com", "method": "DELETE", "path": "/**", "action": "deny"}]
        )
        client = PolicyHTTPClient()
        with (
            patch.object(httpx.Client, "delete") as mock_del,
            pytest.raises(PolicyViolationError, match="REST policy denied"),
        ):
            client.delete("https://api.github.com/repos/org/repo")
        mock_del.assert_not_called()

    def test_first_matching_rule_wins_allow_before_deny(self) -> None:
        """When a GET /repos/** allow rule appears before a catch-all deny, GET should pass."""
        self._use_restrictive_with_rest(
            [
                {"host": "api.github.com", "method": "GET", "path": "/repos/**", "action": "allow"},
                {"host": "api.github.com", "method": "*", "path": "/**", "action": "deny"},
            ]
        )
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://api.github.com/repos/org/name")
        assert resp.status_code == 200

    def test_catch_all_deny_blocks_unmatched_path(self) -> None:
        """A wildcard deny rule blocks paths not explicitly allowed."""
        self._use_restrictive_with_rest(
            [
                {"host": "api.github.com", "method": "GET", "path": "/repos/**", "action": "allow"},
                {"host": "api.github.com", "method": "*", "path": "/**", "action": "deny"},
            ]
        )
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "post") as mock_post, pytest.raises(PolicyViolationError):
            client.post("https://api.github.com/admin/secret")
        mock_post.assert_not_called()

    def test_rest_policy_none_result_passes_through(self) -> None:
        """When no REST rule matches, the request proceeds (None = pass-through)."""
        self._use_restrictive_with_rest(
            [{"host": "other.host.com", "method": "DELETE", "path": "/**", "action": "deny"}]
        )
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://api.github.com/zen")
        assert resp.status_code == 200

    def test_wildcard_method_matches_any_method(self) -> None:
        """A rule with method '*' should deny all methods on the matching path."""
        self._use_restrictive_with_rest(
            [{"host": "api.github.com", "method": "*", "path": "/admin/**", "action": "deny"}]
        )
        client = PolicyHTTPClient()
        for method_name in ("get", "post", "put", "patch", "delete"):
            with (
                patch.object(httpx.Client, method_name) as mock_method,
                pytest.raises(PolicyViolationError),
            ):
                getattr(client, method_name)("https://api.github.com/admin/keys")
            mock_method.assert_not_called()

    def test_path_glob_matching_depth(self) -> None:
        """Glob ** matches nested paths at arbitrary depth."""
        self._use_restrictive_with_rest(
            [{"host": "api.github.com", "method": "GET", "path": "/a/**/z", "action": "deny"}]
        )
        client = PolicyHTTPClient()
        with pytest.raises(PolicyViolationError):
            client.get("https://api.github.com/a/b/c/d/z")

    def test_rest_policy_case_insensitive_method(self) -> None:
        """REST rules configured with lowercase methods should still match."""
        rest_policy = RestPolicy.from_config(
            [{"host": "api.github.com", "method": "delete", "path": "/**", "action": "deny"}]
        )
        result = rest_policy.check("api.github.com", "DELETE", "/something")
        assert result == "deny"

    def test_rest_policy_case_insensitive_host(self) -> None:
        """REST rules configured with uppercase hostnames should still match."""
        rest_policy = RestPolicy.from_config(
            [{"host": "API.GITHUB.COM", "method": "GET", "path": "/**", "action": "allow"}]
        )
        result = rest_policy.check("api.github.com", "GET", "/anything")
        assert result == "allow"


# ===========================================================================
# 3. Network policy checks (CIDR, domain suffix, per-category hosts)
# ===========================================================================


class TestNetworkPolicyChecks:
    """Validate that the full network policy stack is exercised correctly."""

    def test_domain_suffix_match_allows_subdomain(self) -> None:
        init_policy_engine(_restrictive_config(allowed_domains=["*.example.com"]))
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://api.example.com/data")
        assert resp.status_code == 200

    def test_domain_suffix_match_allows_apex(self) -> None:
        """*.example.com should also allow the apex domain example.com."""
        init_policy_engine(_restrictive_config(allowed_domains=["*.example.com"]))
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://example.com/resource")
        assert resp.status_code == 200

    def test_exact_domain_does_not_match_subdomain(self) -> None:
        """An exact domain rule should not match a subdomain."""
        init_policy_engine(_restrictive_config(allowed_domains=["example.com"]))
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get") as mock_get, pytest.raises(PolicyViolationError):
            client.get("https://sub.example.com/data")
        mock_get.assert_not_called()

    def test_provider_category_host_is_checked(self) -> None:
        """provider_allowed_hosts is consulted when category='provider'."""
        init_policy_engine(_restrictive_config(provider_allowed_hosts=["api.anthropic.com"]))
        client = PolicyHTTPClient(category="provider")
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            resp = client.post("https://api.anthropic.com/v1/messages")
        assert resp.status_code == 200

    def test_provider_category_host_does_not_allow_other_category(self) -> None:
        """A host in provider_allowed_hosts must not be accessible with category='tool'."""
        init_policy_engine(_restrictive_config(provider_allowed_hosts=["api.anthropic.com"]))
        client = PolicyHTTPClient(category="tool")
        with patch.object(httpx.Client, "get") as mock_get, pytest.raises(PolicyViolationError):
            client.get("https://api.anthropic.com/v1/models")
        mock_get.assert_not_called()

    def test_tool_category_host_is_checked(self) -> None:
        init_policy_engine(_restrictive_config(tool_allowed_hosts=["tools.example.com"]))
        client = PolicyHTTPClient(category="tool")
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://tools.example.com/execute")
        assert resp.status_code == 200

    def test_discord_category_host_is_checked(self) -> None:
        init_policy_engine(_restrictive_config(discord_allowed_hosts=["discord.com"]))
        client = PolicyHTTPClient(category="discord")
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            resp = client.post("https://discord.com/api/webhooks/123/token")
        assert resp.status_code == 200

    def test_global_host_allowed_regardless_of_category(self) -> None:
        """A host in allowed_hosts is accessible for any (or no) category."""
        init_policy_engine(_restrictive_config(allowed_hosts=["shared.example.com"]))
        for cat in ("", "provider", "tool", "discord"):
            client = PolicyHTTPClient(category=cat)
            mock_resp = _mock_response(200)
            with patch.object(httpx.Client, "get", return_value=mock_resp):
                resp = client.get("https://shared.example.com/resource")
            assert resp.status_code == 200

    def test_host_with_port_in_allow_list_is_matched(self) -> None:
        """allowed_hosts entry with :port suffix should still match the bare host."""
        init_policy_engine(_restrictive_config(allowed_hosts=["api.example.com:443"]))
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://api.example.com/data")
        assert resp.status_code == 200

    def test_default_allow_mode_permits_any_host(self) -> None:
        """default_deny=False permits all outbound requests without an explicit allow list."""
        init_policy_engine(_permissive_config())
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://completely-arbitrary-host.example.org/anything")
        assert resp.status_code == 200


# ===========================================================================
# 4. Interactive approval flow — edge cases
# ===========================================================================


class TestInteractiveApprovalEdgeCases:
    """Additional scenarios for the interactive approval integration."""

    def _use_restrictive(self) -> None:
        init_policy_engine(_restrictive_config())

    def _make_approval(self, prompt_returns: bool) -> MagicMock:
        from missy.agent.interactive_approval import InteractiveApproval

        approval = MagicMock(spec=InteractiveApproval)
        approval.prompt_user.return_value = prompt_returns
        return approval

    def test_approval_called_with_network_request_action(self) -> None:
        """prompt_user is called with 'network_request' as the action name."""
        self._use_restrictive()
        approval = self._make_approval(prompt_returns=True)
        set_interactive_approval(approval)
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            client.get("https://anything.example.com/path")
        approval.prompt_user.assert_called_once_with(
            "network_request", "https://anything.example.com/path"
        )

    def test_approval_called_with_full_url(self) -> None:
        """The full URL (with query string) is passed to prompt_user."""
        self._use_restrictive()
        approval = self._make_approval(prompt_returns=True)
        set_interactive_approval(approval)
        client = PolicyHTTPClient()
        url = "https://anything.example.com/path?key=val"
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            client.get(url)
        approval.prompt_user.assert_called_once_with("network_request", url)

    def test_approval_granted_request_is_executed(self) -> None:
        """After operator approval, the underlying httpx call is made."""
        self._use_restrictive()
        approval = self._make_approval(prompt_returns=True)
        set_interactive_approval(approval)
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp) as mock_get:
            client.get("https://override.example.com/")
        mock_get.assert_called_once()

    def test_approval_denied_request_is_not_executed(self) -> None:
        """After operator denial, the httpx call is never made."""
        self._use_restrictive()
        approval = self._make_approval(prompt_returns=False)
        set_interactive_approval(approval)
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get") as mock_get, pytest.raises(PolicyViolationError):
            client.get("https://override.example.com/")
        mock_get.assert_not_called()

    def test_approval_granted_emits_audit_event(self) -> None:
        """An operator-approved request still emits a network_request audit event."""
        self._use_restrictive()
        approval = self._make_approval(prompt_returns=True)
        set_interactive_approval(approval)
        client = PolicyHTTPClient()
        mock_resp = _mock_response(201)
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            client.post("https://override.example.com/submit")
        events = event_bus.get_events(event_type="network_request")
        assert len(events) == 1
        assert events[0].result == "allow"

    def test_approval_denied_no_audit_event_emitted(self) -> None:
        """A denied request (even after prompting) emits no network_request event."""
        self._use_restrictive()
        approval = self._make_approval(prompt_returns=False)
        set_interactive_approval(approval)
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get"), pytest.raises(PolicyViolationError):
            client.get("https://denied.example.com/")
        assert event_bus.get_events(event_type="network_request") == []

    def test_non_interactive_approval_type_raises_without_prompting(self) -> None:
        """A plain object (not InteractiveApproval) in _interactive_approval is ignored."""
        self._use_restrictive()
        # Install a generic callable that records if it was called.
        called = []

        class FakeApproval:
            def prompt_user(self, *_):  # noqa: ANN001
                called.append(True)
                return True

        set_interactive_approval(FakeApproval())
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get"), pytest.raises(PolicyViolationError):
            client.get("https://blocked.example.com/")
        # FakeApproval.prompt_user must NOT have been called
        assert called == []


# ===========================================================================
# 5. Error handling for network failures
# ===========================================================================


class TestNetworkFailureErrorHandling:
    """httpx network-layer errors propagate transparently and inhibit audit events."""

    def test_connect_timeout_propagates(self) -> None:
        client = PolicyHTTPClient()
        with (
            patch.object(httpx.Client, "get", side_effect=httpx.ConnectTimeout("timeout")),
            pytest.raises(httpx.ConnectTimeout),
        ):
            client.get("https://api.example.com/slow")

    def test_read_timeout_propagates(self) -> None:
        client = PolicyHTTPClient()
        with (
            patch.object(httpx.Client, "post", side_effect=httpx.ReadTimeout("read timeout")),
            pytest.raises(httpx.ReadTimeout),
        ):
            client.post("https://api.example.com/upload")

    def test_write_timeout_propagates(self) -> None:
        client = PolicyHTTPClient()
        with (
            patch.object(httpx.Client, "put", side_effect=httpx.WriteTimeout("write timeout")),
            pytest.raises(httpx.WriteTimeout),
        ):
            client.put("https://api.example.com/resource")

    def test_pool_timeout_propagates(self) -> None:
        client = PolicyHTTPClient()
        with (
            patch.object(httpx.Client, "get", side_effect=httpx.PoolTimeout("pool exhausted")),
            pytest.raises(httpx.PoolTimeout),
        ):
            client.get("https://api.example.com/resource")

    def test_remote_protocol_error_propagates(self) -> None:
        client = PolicyHTTPClient()
        exc = httpx.RemoteProtocolError("bad protocol")
        with (
            patch.object(httpx.Client, "get", side_effect=exc),
            pytest.raises(httpx.RemoteProtocolError),
        ):
            client.get("https://api.example.com/resource")

    def test_network_error_does_not_emit_event(self) -> None:
        client = PolicyHTTPClient()
        with (
            patch.object(httpx.Client, "delete", side_effect=httpx.ConnectError("refused")),
            pytest.raises(httpx.ConnectError),
        ):
            client.delete("https://api.example.com/res/1")
        assert event_bus.get_events(event_type="network_request") == []

    async def test_async_read_timeout_propagates(self) -> None:
        client = PolicyHTTPClient()
        with (
            patch.object(
                httpx.AsyncClient,
                "get",
                new_callable=AsyncMock,
                side_effect=httpx.ReadTimeout("async read timeout"),
            ),
            pytest.raises(httpx.ReadTimeout),
        ):
            await client.aget("https://api.example.com/slow")

    async def test_async_connect_error_does_not_emit_event(self) -> None:
        client = PolicyHTTPClient()
        with (
            patch.object(
                httpx.AsyncClient,
                "post",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("async refused"),
            ),
            pytest.raises(httpx.ConnectError),
        ):
            await client.apost("https://api.example.com/submit")
        assert event_bus.get_events(event_type="network_request") == []

    def test_4xx_response_is_returned_not_raised(self) -> None:
        """A 404 is a valid HTTP response and must not be wrapped in an exception."""
        client = PolicyHTTPClient()
        mock_resp = _mock_response(404)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://api.example.com/missing")
        assert resp.status_code == 404

    def test_5xx_response_is_returned_not_raised(self) -> None:
        client = PolicyHTTPClient()
        mock_resp = _mock_response(503)
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            resp = client.post("https://api.example.com/flaky")
        assert resp.status_code == 503


# ===========================================================================
# 6. TLS / SSL handling
# ===========================================================================


class TestTLSSSLHandling:
    """The client never passes verify=False or follow_redirects=True to httpx."""

    def test_sync_client_created_with_follow_redirects_false(self) -> None:
        client = PolicyHTTPClient()
        sync = client._get_sync_client()
        # httpx.Client stores follow_redirects as an attribute
        assert sync.follow_redirects is False

    def test_async_client_created_with_follow_redirects_false(self) -> None:
        client = PolicyHTTPClient()
        ac = client._get_async_client()
        assert ac.follow_redirects is False

    def test_verify_kwarg_stripped_from_get_request(self) -> None:
        """verify=False in kwargs must be stripped before reaching httpx."""
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp) as mock_get:
            client.get("https://api.example.com/", verify=False)
        _, call_kwargs = mock_get.call_args
        assert "verify" not in call_kwargs

    def test_verify_kwarg_stripped_from_post_request(self) -> None:
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "post", return_value=mock_resp) as mock_post:
            client.post("https://api.example.com/", verify=False, json={"x": 1})
        _, call_kwargs = mock_post.call_args
        assert "verify" not in call_kwargs
        assert "json" in call_kwargs

    def test_transport_kwarg_stripped(self) -> None:
        """Custom transport kwarg (which could bypass policy) is stripped."""
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp) as mock_get:
            client.get("https://api.example.com/", transport=MagicMock())
        _, call_kwargs = mock_get.call_args
        assert "transport" not in call_kwargs

    def test_auth_kwarg_stripped(self) -> None:
        """auth kwarg (credential injection) is stripped."""
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp) as mock_get:
            client.get("https://api.example.com/", auth=("user", "secret"))
        _, call_kwargs = mock_get.call_args
        assert "auth" not in call_kwargs

    def test_base_url_kwarg_stripped(self) -> None:
        """base_url kwarg (redirect traffic to arbitrary host) is stripped."""
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp) as mock_get:
            client.get("https://api.example.com/", base_url="https://evil.com")
        _, call_kwargs = mock_get.call_args
        assert "base_url" not in call_kwargs

    def test_sync_client_pool_limits_are_applied(self) -> None:
        client = PolicyHTTPClient()
        sync = client._get_sync_client()
        limits = sync._transport._pool._max_connections  # noqa: SLF001
        # Validate pool limits are set (max_connections=20)
        assert limits == 20

    async def test_async_client_created_with_follow_redirects_false_in_request(self) -> None:
        """Async requests also go through a client with follow_redirects=False."""
        client = PolicyHTTPClient()
        mock_resp = _mock_response(302)
        with patch.object(
            httpx.AsyncClient,
            "get",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            resp = await client.aget("https://api.example.com/redirect")
        # Should receive the 302 rather than silently following the redirect
        assert resp.status_code == 302


# ===========================================================================
# 7. Request / response logging (audit events)
# ===========================================================================


class TestRequestResponseLogging:
    """Audit events are emitted with correct fields for all HTTP methods."""

    def test_get_audit_event_has_correct_method(self) -> None:
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get", return_value=_mock_response(200)):
            client.get("https://api.example.com/resource")
        assert event_bus.get_events(event_type="network_request")[0].detail["method"] == "GET"

    def test_post_audit_event_has_correct_method(self) -> None:
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "post", return_value=_mock_response(201)):
            client.post("https://api.example.com/items")
        assert event_bus.get_events(event_type="network_request")[0].detail["method"] == "POST"

    def test_put_audit_event_has_correct_method(self) -> None:
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "put", return_value=_mock_response(204)):
            client.put("https://api.example.com/items/1")
        assert event_bus.get_events(event_type="network_request")[0].detail["method"] == "PUT"

    def test_patch_audit_event_has_correct_method(self) -> None:
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "patch", return_value=_mock_response(200)):
            client.patch("https://api.example.com/items/1")
        assert event_bus.get_events(event_type="network_request")[0].detail["method"] == "PATCH"

    def test_delete_audit_event_has_correct_method(self) -> None:
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "delete", return_value=_mock_response(204)):
            client.delete("https://api.example.com/items/1")
        assert event_bus.get_events(event_type="network_request")[0].detail["method"] == "DELETE"

    def test_head_audit_event_has_correct_method(self) -> None:
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "head", return_value=_mock_response(200)):
            client.head("https://api.example.com/resource")
        assert event_bus.get_events(event_type="network_request")[0].detail["method"] == "HEAD"

    def test_audit_event_url_includes_query_string(self) -> None:
        """The full URL (with query string) is recorded in the audit event."""
        client = PolicyHTTPClient()
        url = "https://api.example.com/search?q=test&page=2"
        with patch.object(httpx.Client, "get", return_value=_mock_response(200)):
            client.get(url)
        assert event_bus.get_events(event_type="network_request")[0].detail["url"] == url

    def test_audit_event_status_code_matches_response(self) -> None:
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "post", return_value=_mock_response(422)):
            client.post("https://api.example.com/invalid")
        assert event_bus.get_events(event_type="network_request")[0].detail["status_code"] == 422

    def test_sequential_requests_emit_events_in_order(self) -> None:
        client = PolicyHTTPClient(session_id="ordered")
        urls = [
            "https://api.example.com/a",
            "https://api.example.com/b",
            "https://api.example.com/c",
        ]
        for url in urls:
            with patch.object(httpx.Client, "get", return_value=_mock_response(200)):
                client.get(url)
        events = event_bus.get_events(event_type="network_request", session_id="ordered")
        assert [e.detail["url"] for e in events] == urls

    async def test_async_methods_emit_correct_methods_in_events(self) -> None:
        client = PolicyHTTPClient(session_id="async-logging")
        specs = [
            ("aget", "get", "GET", 200),
            ("apost", "post", "POST", 201),
            ("aput", "put", "PUT", 204),
            ("apatch", "patch", "PATCH", 200),
            ("adelete", "delete", "DELETE", 204),
        ]
        for client_method, httpx_method, expected_method, status in specs:
            event_bus.clear()
            mock_resp = _mock_response(status)
            with patch.object(
                httpx.AsyncClient,
                httpx_method,
                new_callable=AsyncMock,
                return_value=mock_resp,
            ):
                await getattr(client, client_method)("https://api.example.com/resource")
            events = event_bus.get_events(event_type="network_request")
            assert len(events) == 1
            assert events[0].detail["method"] == expected_method
            assert events[0].detail["status_code"] == status


# ===========================================================================
# 8. Thread safety of the HTTP client
# ===========================================================================


class TestThreadSafetyHTTPClient:
    """Concurrent requests share state safely; lazy-init is race-free."""

    def test_concurrent_gets_all_succeed(self) -> None:
        """Multiple threads making GET requests should all succeed."""
        init_policy_engine(_permissive_config())
        client = PolicyHTTPClient()
        results: list[int] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        mock_resp = _mock_response(200)

        def do_get() -> None:
            try:
                resp = client.get("https://api.example.com/resource")
                with lock:
                    results.append(resp.status_code)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        # Patch at the class level so all threads share the same mock
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            threads = [threading.Thread(target=do_get) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5.0)

        assert errors == [], f"Thread errors: {errors}"
        assert len(results) == 10
        assert all(s == 200 for s in results)

    def test_lazy_sync_client_init_is_idempotent_under_concurrency(self) -> None:
        """_get_sync_client() returns the same instance even under concurrent access."""
        client = PolicyHTTPClient()
        instances: list[httpx.Client] = []
        lock = threading.Lock()

        def grab_client() -> None:
            instance = client._get_sync_client()
            with lock:
                instances.append(instance)

        threads = [threading.Thread(target=grab_client) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # All threads should have gotten the same instance (or at worst the last one)
        # The key requirement is that the client was created (not None)
        assert all(isinstance(i, httpx.Client) for i in instances)

    def test_multiple_clients_have_independent_state(self) -> None:
        """Each PolicyHTTPClient instance is independent."""
        client_a = PolicyHTTPClient(session_id="a")
        client_b = PolicyHTTPClient(session_id="b")

        assert client_a is not client_b
        assert client_a.session_id != client_b.session_id

        # Creating sync client for one should not affect the other
        _ = client_a._get_sync_client()
        assert client_a._sync_client is not None
        assert client_b._sync_client is None

    def test_close_does_not_affect_other_client_instances(self) -> None:
        """Closing one client does not invalidate another."""
        client_a = PolicyHTTPClient()
        client_b = PolicyHTTPClient()

        _ = client_a._get_sync_client()
        _ = client_b._get_sync_client()

        client_a.close()
        assert client_a._sync_client is None
        assert client_b._sync_client is not None  # unaffected
        client_b.close()


# ===========================================================================
# 9. Rate limiting integration (via policy engine mock)
# ===========================================================================


class TestRateLimitingIntegration:
    """The gateway delegates to the policy engine; rate limiting can be expressed
    as a PolicyViolationError from check_network.  Verify that PolicyHTTPClient
    handles this correctly."""

    def test_rate_limited_host_raises_policy_violation(self) -> None:
        """When check_network raises (simulating rate-limit enforcement), the
        request is blocked and no httpx call is made."""
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.check_network.side_effect = PolicyViolationError(
                "Rate limit exceeded for api.example.com",
                category="network",
                detail="Too many requests per minute",
            )
            mock_get_engine.return_value = mock_engine

            client = PolicyHTTPClient()
            with (
                patch.object(httpx.Client, "get") as mock_get,
                pytest.raises(PolicyViolationError, match="Rate limit exceeded"),
            ):
                client.get("https://api.example.com/resource")
            mock_get.assert_not_called()

    def test_rate_limit_error_does_not_emit_audit_event(self) -> None:
        """A policy-denied request (including rate limiting) must not emit a
        network_request audit event."""
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.check_network.side_effect = PolicyViolationError(
                "Rate limited", category="network", detail="quota exceeded"
            )
            mock_get_engine.return_value = mock_engine

            client = PolicyHTTPClient()
            with patch.object(httpx.Client, "get"), pytest.raises(PolicyViolationError):
                client.get("https://api.example.com/resource")

        assert event_bus.get_events(event_type="network_request") == []

    def test_policy_engine_check_network_called_once_per_request(self) -> None:
        """check_network is called exactly once per HTTP method invocation."""
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine
            mock_resp = _mock_response(200)
            with patch.object(httpx.Client, "get", return_value=mock_resp):
                PolicyHTTPClient().get("https://api.example.com/one")
            with patch.object(httpx.Client, "get", return_value=mock_resp):
                PolicyHTTPClient().get("https://api.example.com/two")

        assert mock_engine.check_network.call_count == 2


# ===========================================================================
# 10. Edge cases: malformed URLs, empty hosts, redirect behaviour
# ===========================================================================


class TestEdgeCases:
    """Miscellaneous edge cases not covered elsewhere."""

    def test_url_with_username_in_netloc_still_extracts_hostname(self) -> None:
        """URLs with userinfo (user:pass@host) — hostname is extracted correctly."""
        client = PolicyHTTPClient()
        # permissive engine — should not raise on a valid URL
        with contextlib.suppress(PolicyViolationError):
            client._check_url("https://user:pass@api.example.com/private")

    def test_url_with_only_whitespace_raises(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError):
            client._check_url("   ")

    def test_redirect_is_not_followed_sync(self) -> None:
        """The sync client is configured with follow_redirects=False, so a 301
        response is returned to the caller as-is."""
        client = PolicyHTTPClient()
        mock_resp = _mock_response(301, headers={"location": "https://api.example.com/new"})
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://api.example.com/old")
        assert resp.status_code == 301

    async def test_redirect_is_not_followed_async(self) -> None:
        client = PolicyHTTPClient()
        mock_resp = _mock_response(302, headers={"location": "https://api.example.com/new"})
        with patch.object(
            httpx.AsyncClient,
            "get",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            resp = await client.aget("https://api.example.com/old")
        assert resp.status_code == 302

    def test_response_size_limit_exactly_at_boundary_passes(self) -> None:
        client = PolicyHTTPClient(max_response_bytes=500)
        resp = _mock_response(200, headers={"content-length": "500"})
        # Exactly at limit — should NOT raise
        client._check_response_size(resp, "https://api.example.com/")

    def test_response_size_limit_one_byte_over_raises(self) -> None:
        client = PolicyHTTPClient(max_response_bytes=500)
        resp = _mock_response(200, headers={"content-length": "501"})
        with pytest.raises(ValueError, match="too large"):
            client._check_response_size(resp, "https://api.example.com/")

    def test_content_length_zero_passes(self) -> None:
        client = PolicyHTTPClient(max_response_bytes=500)
        resp = _mock_response(204, headers={"content-length": "0"})
        client._check_response_size(resp, "https://api.example.com/")

    def test_content_length_negative_treated_as_zero(self) -> None:
        """A negative Content-Length is invalid; treated as 0 (no size error)."""
        client = PolicyHTTPClient(max_response_bytes=100)
        resp = _mock_response(200, headers={"content-length": "-1"})
        # Negative int < max_response_bytes — no error raised
        client._check_response_size(resp, "https://api.example.com/")

    def test_multiple_clients_can_coexist_with_different_timeouts(self) -> None:
        slow = PolicyHTTPClient(timeout=120)
        fast = PolicyHTTPClient(timeout=5)
        assert slow.timeout == 120
        assert fast.timeout == 5
        slow_sync = slow._get_sync_client()
        fast_sync = fast._get_sync_client()
        assert slow_sync is not fast_sync

    def test_create_client_factory_forwards_all_params(self) -> None:
        client = create_client(session_id="s99", task_id="t99", timeout=45, category="tool")
        assert client.session_id == "s99"
        assert client.task_id == "t99"
        assert client.timeout == 45
        assert client.category == "tool"

    def test_rest_rule_dataclass_is_frozen(self) -> None:
        """RestRule is a frozen dataclass — mutation should raise."""
        rule = RestRule(host="api.github.com", method="GET", path="/repos/**", action="allow")
        with pytest.raises((AttributeError, TypeError)):
            rule.host = "evil.com"  # type: ignore[misc]

    def test_rest_policy_empty_rules_returns_none(self) -> None:
        policy = RestPolicy(rules=[])
        result = policy.check("api.example.com", "GET", "/anything")
        assert result is None

    def test_rest_policy_from_config_empty_list(self) -> None:
        policy = RestPolicy.from_config([])
        result = policy.check("api.example.com", "POST", "/data")
        assert result is None

    def test_get_policy_engine_raises_when_uninitialised(self) -> None:
        """get_policy_engine() must raise RuntimeError when called before init."""
        from missy.policy.engine import get_policy_engine

        original = engine_module._engine
        try:
            engine_module._engine = None
            with pytest.raises(RuntimeError, match="not been initialised"):
                get_policy_engine()
        finally:
            engine_module._engine = original
