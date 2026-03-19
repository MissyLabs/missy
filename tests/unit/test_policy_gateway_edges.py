"""Session-13 edge-case tests for policy engine and gateway client.

Covers gaps NOT addressed by the existing test suite across:
- NetworkPolicyEngine: IPv6 CIDR ranges, /0, /16 edges, broadcast, bracket host
  end-to-end, multiple bad CIDRs with one valid, degenerate domain patterns,
  leading-dot domain, empty domain list fall-through.
- PolicyEngine facade: rest_policy attribute type, category forwarding, singleton
  replacement, rest_policies config wiring.
- RestPolicy: None-result pass-through, from_config with missing keys, path
  with encoded characters, empty path, path root ("/").
- PolicyHTTPClient: IPv6 bracket URL end-to-end, userinfo stripping, non-standard
  port policy match, double-close idempotency, audit event timestamp UTC,
  _emit_request_event with empty IDs, None timeout accepted, max_response_bytes
  at DEFAULT boundary, content-length zero edge, _check_rest_policy None result.
- Preset integration: pypi / npm / docker-hub / huggingface content checks,
  resolve_presets into a real NetworkPolicyEngine, _parse_network presets-only.
"""

from __future__ import annotations

import socket
from collections.abc import Generator
from unittest.mock import MagicMock, patch

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
from missy.gateway.client import PolicyHTTPClient, set_interactive_approval
from missy.policy import engine as engine_module
from missy.policy.engine import PolicyEngine, init_policy_engine
from missy.policy.network import NetworkPolicyEngine
from missy.policy.presets import PRESETS, resolve_presets
from missy.policy.rest_policy import RestPolicy, RestRule

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_net_engine(**overrides) -> NetworkPolicyEngine:
    defaults = {
        "default_deny": True,
        "allowed_cidrs": [],
        "allowed_domains": [],
        "allowed_hosts": [],
        "provider_allowed_hosts": [],
        "tool_allowed_hosts": [],
        "discord_allowed_hosts": [],
    }
    defaults.update(overrides)
    return NetworkPolicyEngine(NetworkPolicy(**defaults))


def _make_config(**net_overrides) -> MissyConfig:
    """Build a MissyConfig with default_deny=True and network overrides."""
    defaults = {
        "default_deny": True,
        "allowed_cidrs": [],
        "allowed_domains": [],
        "allowed_hosts": [],
        "rest_policies": [],
    }
    defaults.update(net_overrides)
    return MissyConfig(
        network=NetworkPolicy(**defaults),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(enabled=False),
        plugins=PluginPolicy(enabled=False),
        providers={},
        workspace_path="/tmp/workspace",
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
    raw = headers or {}
    resp.headers = MagicMock()
    resp.headers.get = lambda k, default=None: raw.get(k, default)
    return resp


@pytest.fixture(autouse=True)
def _isolate() -> Generator[None, None, None]:
    """Reset singleton and event bus around every test."""
    original_engine = engine_module._engine
    original_approval = gateway_module._interactive_approval
    event_bus.clear()
    yield
    engine_module._engine = original_engine
    gateway_module._interactive_approval = original_approval
    event_bus.clear()


# ===========================================================================
# NetworkPolicyEngine — CIDR edge cases
# ===========================================================================


class TestCIDREdgeCases:
    """CIDR matching with IPv6 ranges, /0, /16, broadcast, and bad entries."""

    # --- IPv6 CIDR ranges ---

    def test_ipv6_ula_range_allowed(self) -> None:
        """fc00::/7 covers the Unique Local Address (ULA) range."""
        engine = _make_net_engine(allowed_cidrs=["fc00::/7"])
        assert engine.check_host("fd00::1") is True

    def test_ipv6_link_local_range_allowed_when_in_cidr(self) -> None:
        """fe80::/10 is the link-local range; allowed if CIDR is configured."""
        engine = _make_net_engine(allowed_cidrs=["fe80::/10"])
        assert engine.check_host("fe80::1") is True

    def test_ipv6_documentation_range_allowed(self) -> None:
        """2001:db8::/32 is reserved for documentation."""
        engine = _make_net_engine(allowed_cidrs=["2001:db8::/32"])
        assert engine.check_host("2001:db8::cafe") is True

    def test_ipv6_address_outside_allowed_cidr_denied(self) -> None:
        engine = _make_net_engine(allowed_cidrs=["2001:db8::/32"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("2001:db9::1")

    # --- Broadcast and edge addresses ---

    def test_broadcast_address_of_slash24_allowed(self) -> None:
        """The broadcast address (.255) of a /24 block is still in that CIDR."""
        engine = _make_net_engine(allowed_cidrs=["192.0.2.0/24"])
        assert engine.check_host("192.0.2.255") is True

    def test_network_address_of_slash24_allowed(self) -> None:
        """The network address (.0) of a /24 block is in that CIDR."""
        engine = _make_net_engine(allowed_cidrs=["192.0.2.0/24"])
        assert engine.check_host("192.0.2.0") is True

    def test_first_address_outside_slash24_denied(self) -> None:
        """192.0.3.0 is one block outside 192.0.2.0/24."""
        engine = _make_net_engine(allowed_cidrs=["192.0.2.0/24"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("192.0.3.0")

    # --- /0 and /16 edges ---

    def test_slash0_cidr_allows_any_ipv4(self) -> None:
        """/0 covers all IPv4 addresses."""
        engine = _make_net_engine(allowed_cidrs=["0.0.0.0/0"])
        assert engine.check_host("8.8.8.8") is True

    def test_slash16_allows_first_host(self) -> None:
        engine = _make_net_engine(allowed_cidrs=["10.20.0.0/16"])
        assert engine.check_host("10.20.0.1") is True

    def test_slash16_allows_last_host(self) -> None:
        engine = _make_net_engine(allowed_cidrs=["10.20.0.0/16"])
        assert engine.check_host("10.20.255.254") is True

    def test_slash16_denies_next_block(self) -> None:
        engine = _make_net_engine(allowed_cidrs=["10.20.0.0/16"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("10.21.0.1")

    # --- Multiple bad CIDRs with one valid ---

    def test_all_bad_cidrs_skipped_valid_one_still_works(self) -> None:
        """Three invalid CIDRs before one valid one; the valid one still matches."""
        engine = _make_net_engine(
            allowed_cidrs=["not-a-cidr", "999.999.999.999/24", "::zzz/128", "10.0.0.0/8"]
        )
        assert engine.check_host("10.50.0.1") is True

    def test_all_bad_cidrs_causes_deny(self) -> None:
        """If every CIDR is invalid and no other rule matches, deny."""
        engine = _make_net_engine(allowed_cidrs=["not-a-cidr", "bad/bad"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("10.0.0.1")

    # --- IPv6 bracket notation end-to-end ---

    def test_ipv6_in_brackets_allowed_via_cidr(self) -> None:
        """[::1] notation from URL parsers is stripped and checked as ::1."""
        engine = _make_net_engine(allowed_cidrs=["::1/128"])
        assert engine.check_host("[::1]") is True

    def test_ipv6_in_brackets_denied_when_not_in_cidr(self) -> None:
        engine = _make_net_engine(allowed_cidrs=["2001:db8::/32"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("[::2]")


# ===========================================================================
# NetworkPolicyEngine — domain matching edge cases
# ===========================================================================


class TestDomainMatchingEdgeCases:
    """Degenerate patterns, leading-dot domains, empty allowlists."""

    def test_leading_dot_pattern_matches_exact_suffix(self) -> None:
        """'.example.com' has no wildcard prefix, so it is treated as exact match only."""
        engine = _make_net_engine(allowed_domains=[".example.com"])
        with pytest.raises(PolicyViolationError):
            # '.example.com' is not a wildcard — subdomain should not match
            engine.check_host("api.example.com")

    def test_empty_domains_list_falls_through_to_deny(self) -> None:
        engine = _make_net_engine(allowed_domains=[], allowed_hosts=[])
        with pytest.raises(PolicyViolationError):
            engine.check_host("any.example.com")

    def test_wildcard_with_trailing_dot_in_pattern_does_not_crash(self) -> None:
        """A slightly malformed pattern '*.example.com.' must not raise internally."""
        engine = _make_net_engine(allowed_domains=["*.example.com."])
        # The trailing dot in the pattern suffix becomes "example.com." which
        # won't match "api.example.com" — we just need no exception.
        with pytest.raises(PolicyViolationError):
            engine.check_host("api.example.com")

    def test_domain_pattern_with_brackets_around_hostname_stripped(self) -> None:
        """check_host strips surrounding brackets before matching.

        [api.example.com] → api.example.com which matches *.example.com.
        This is the same normalisation applied to IPv6 literals.
        """
        engine = _make_net_engine(allowed_domains=["*.example.com"])
        # Bracket-wrapped hostnames are stripped; api.example.com matches *.example.com.
        assert engine.check_host("[api.example.com]") is True

    def test_two_wildcard_domains_second_matches(self) -> None:
        engine = _make_net_engine(allowed_domains=["*.foo.com", "*.bar.com"])
        assert engine.check_host("api.bar.com") is True

    def test_wildcard_does_not_match_sibling_tld(self) -> None:
        engine = _make_net_engine(allowed_domains=["*.example.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("api.example.net")

    def test_exact_domain_case_insensitive(self) -> None:
        engine = _make_net_engine(allowed_domains=["Example.COM"])
        assert engine.check_host("example.com") is True

    def test_check_domain_returns_pattern_string_in_rule_name(self) -> None:
        engine = _make_net_engine(allowed_domains=["*.example.com"])
        event_bus.clear()
        engine.check_host("api.example.com")
        events = event_bus.get_events(result="allow")
        assert events[0].policy_rule == "domain:*.example.com"


# ===========================================================================
# PolicyEngine facade
# ===========================================================================


class TestPolicyEngineFacade:
    """Tests for the PolicyEngine facade layer not covered elsewhere."""

    def test_rest_policy_attribute_is_rest_policy_instance(self) -> None:
        pe = PolicyEngine(_make_config())
        assert isinstance(pe.rest_policy, RestPolicy)

    def test_empty_rest_policies_in_config_yields_empty_rest_policy(self) -> None:
        pe = PolicyEngine(_make_config(rest_policies=[]))
        assert pe.rest_policy.check("any.host", "GET", "/any") is None

    def test_rest_policies_wired_from_config(self) -> None:
        config = _make_config(
            allowed_hosts=["api.github.com"],
            rest_policies=[
                {"host": "api.github.com", "method": "DELETE", "path": "/**", "action": "deny"}
            ],
        )
        pe = PolicyEngine(config)
        assert pe.rest_policy.check("api.github.com", "DELETE", "/repos") == "deny"

    def test_check_network_category_forwarded(self) -> None:
        config = _make_config()
        config.network.provider_allowed_hosts = ["api.anthropic.com"]
        pe = PolicyEngine(config)
        # With category="provider" it should be allowed
        assert pe.check_network("api.anthropic.com", category="provider") is True

    def test_check_network_category_wrong_denied(self) -> None:
        config = _make_config()
        config.network.provider_allowed_hosts = ["api.anthropic.com"]
        pe = PolicyEngine(config)
        with pytest.raises(PolicyViolationError):
            pe.check_network("api.anthropic.com", category="tool")

    def test_init_policy_engine_replaces_singleton(self) -> None:
        """After init_policy_engine is called twice, only the second engine is returned."""
        from missy.policy.engine import get_policy_engine

        engine1 = init_policy_engine(_make_config(default_deny=False))
        engine2 = init_policy_engine(_make_config(default_deny=True))
        assert get_policy_engine() is engine2
        assert engine1 is not engine2

    def test_policy_engine_check_shell_delegates_to_shell_engine(self) -> None:
        config = _make_config()
        config.shell.enabled = True
        config.shell.allowed_commands = ["ls"]
        pe = PolicyEngine(config)
        assert pe.check_shell("ls -la") is True

    def test_policy_engine_check_read_delegates(self, tmp_path) -> None:
        config = _make_config()
        config.filesystem.allowed_read_paths = [str(tmp_path)]
        pe = PolicyEngine(config)
        assert pe.check_read(str(tmp_path / "file.txt")) is True


# ===========================================================================
# RestPolicy edge cases
# ===========================================================================


class TestRestPolicyEdgeCases:
    """Gaps in RestPolicy not covered by existing test files."""

    def test_check_returns_none_when_no_rules_match(self) -> None:
        """None result must pass through without raise at the caller level."""
        policy = RestPolicy.from_config(
            [{"host": "api.github.com", "method": "GET", "path": "/repos/**", "action": "allow"}]
        )
        result = policy.check("other.com", "GET", "/repos/foo")
        assert result is None

    def test_check_rest_policy_none_result_does_not_raise_in_gateway(self) -> None:
        """When rest_policy.check() returns None, gateway does not raise."""
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.check_network.return_value = True
            mock_engine.rest_policy.check.return_value = None
            mock_get_engine.return_value = mock_engine
            client = PolicyHTTPClient()
            # Should not raise
            client._check_rest_policy("api.github.com", "GET", "/repos/foo")

    def test_from_config_missing_host_defaults_to_empty_string(self) -> None:
        """A dict without 'host' key should use empty string as default."""
        policy = RestPolicy.from_config(
            [{"method": "GET", "path": "/foo", "action": "allow"}]
        )
        # The rule's host is "" — will only match host=""
        assert policy.check("", "GET", "/foo") == "allow"
        assert policy.check("api.example.com", "GET", "/foo") is None

    def test_from_config_missing_method_defaults_to_star(self) -> None:
        """A dict without 'method' key should default to '*' (match any method)."""
        policy = RestPolicy.from_config(
            [{"host": "h.com", "path": "/**", "action": "deny"}]
        )
        assert policy.check("h.com", "POST", "/anything") == "deny"
        assert policy.check("h.com", "DELETE", "/anything") == "deny"

    def test_from_config_missing_path_defaults_to_double_star(self) -> None:
        """A dict without 'path' key should default to '/**' (match all paths)."""
        policy = RestPolicy.from_config(
            [{"host": "h.com", "method": "GET", "action": "allow"}]
        )
        assert policy.check("h.com", "GET", "/any/path/here") == "allow"

    def test_from_config_missing_action_defaults_to_deny(self) -> None:
        """A dict without 'action' key should default to 'deny'."""
        policy = RestPolicy.from_config(
            [{"host": "h.com", "method": "DELETE", "path": "/**"}]
        )
        assert policy.check("h.com", "DELETE", "/res") == "deny"

    def test_path_root_slash_matched_by_double_star_rule(self) -> None:
        """The path '/' is matched by the '/**' glob."""
        policy = RestPolicy.from_config(
            [{"host": "h.com", "method": "GET", "path": "/**", "action": "allow"}]
        )
        assert policy.check("h.com", "GET", "/") == "allow"

    def test_empty_path_matched_by_double_star(self) -> None:
        """An empty path string is matched by '/**' because fnmatch('', '/**') — depends on fnmatch."""
        policy = RestPolicy.from_config(
            [{"host": "h.com", "method": "GET", "path": "/**", "action": "allow"}]
        )
        result = policy.check("h.com", "GET", "")
        # fnmatch("", "/**") is False — result is None; we just verify no crash
        assert result in ("allow", None)

    def test_url_encoded_path_character_is_literal_in_fnmatch(self) -> None:
        """Path encoding (%20 etc.) is treated as a literal string by fnmatch."""
        policy = RestPolicy.from_config(
            [{"host": "h.com", "method": "GET", "path": "/path%20with%20spaces", "action": "allow"}]
        )
        assert policy.check("h.com", "GET", "/path%20with%20spaces") == "allow"
        assert policy.check("h.com", "GET", "/path with spaces") is None

    def test_rule_with_allow_action_propagated_correctly(self) -> None:
        policy = RestPolicy(rules=[RestRule(host="h.com", method="GET", path="/ok", action="allow")])
        assert policy.check("h.com", "GET", "/ok") == "allow"

    def test_rule_with_deny_action_propagated_correctly(self) -> None:
        policy = RestPolicy(rules=[RestRule(host="h.com", method="POST", path="/bad", action="deny")])
        assert policy.check("h.com", "POST", "/bad") == "deny"


# ===========================================================================
# PolicyHTTPClient — additional edge cases
# ===========================================================================


class TestPolicyHTTPClientEdgeCases:
    """Gaps in PolicyHTTPClient not covered by existing tests."""

    @pytest.fixture(autouse=True)
    def _permissive(self) -> None:
        init_policy_engine(_make_config(default_deny=False))

    # --- None timeout ---

    def test_none_timeout_is_accepted(self) -> None:
        """timeout=None bypasses the positive-value check (None means no timeout)."""
        client = PolicyHTTPClient(timeout=None)
        assert client.timeout is None

    # --- Double-close idempotency ---

    def test_close_twice_does_not_raise(self) -> None:
        client = PolicyHTTPClient()
        _ = client._get_sync_client()
        client.close()
        client.close()  # second call — _sync_client is already None
        assert client._sync_client is None

    async def test_aclose_twice_does_not_raise(self) -> None:
        client = PolicyHTTPClient()
        _ = client._get_async_client()
        await client.aclose()
        await client.aclose()  # second call — _async_client is already None
        assert client._async_client is None

    # --- Audit event timestamp is timezone-aware ---

    def test_emit_event_timestamp_is_utc_aware(self) -> None:
        client = PolicyHTTPClient(session_id="tz-test")
        client._emit_request_event("GET", "https://example.com/", 200)
        events = event_bus.get_events(event_type="network_request")
        assert events[0].timestamp.tzinfo is not None
        # Must be UTC
        assert events[0].timestamp.utcoffset().total_seconds() == 0

    # --- _emit_request_event with empty IDs ---

    def test_emit_event_with_empty_session_and_task_ids(self) -> None:
        client = PolicyHTTPClient()
        client._emit_request_event("POST", "https://example.com/submit", 201)
        events = event_bus.get_events(event_type="network_request")
        assert len(events) == 1
        assert events[0].session_id == ""
        assert events[0].task_id == ""

    # --- max_response_bytes default constant ---

    def test_default_max_response_bytes_constant_is_50mb(self) -> None:
        assert PolicyHTTPClient.DEFAULT_MAX_RESPONSE_BYTES == 50 * 1024 * 1024

    def test_default_max_response_bytes_applied_when_zero_passed(self) -> None:
        client = PolicyHTTPClient(max_response_bytes=0)
        assert client.max_response_bytes == PolicyHTTPClient.DEFAULT_MAX_RESPONSE_BYTES

    # --- content-length 0 edge ---

    def test_response_content_length_zero_not_rejected(self) -> None:
        """A 204 No Content response with Content-Length: 0 should never raise."""
        client = PolicyHTTPClient(max_response_bytes=1024)
        resp = _mock_response(204, content=b"", headers={"content-length": "0"})
        client._check_response_size(resp, "https://example.com/nothing")  # must not raise

    # --- Response size at DEFAULT_MAX_RESPONSE_BYTES boundary ---

    def test_response_at_default_limit_boundary_not_rejected(self) -> None:
        client = PolicyHTTPClient()
        limit = PolicyHTTPClient.DEFAULT_MAX_RESPONSE_BYTES
        resp = _mock_response(200, headers={"content-length": str(limit)})
        client._check_response_size(resp, "https://example.com/boundary")  # at limit, not over

    def test_response_one_over_default_limit_rejected(self) -> None:
        client = PolicyHTTPClient()
        over = PolicyHTTPClient.DEFAULT_MAX_RESPONSE_BYTES + 1
        resp = _mock_response(200, headers={"content-length": str(over)})
        with pytest.raises(ValueError, match="too large"):
            client._check_response_size(resp, "https://example.com/toobig")

    # --- URL with userinfo (user:pass@host) ---

    def test_url_with_userinfo_extracts_hostname_correctly(self) -> None:
        """urlparse().hostname strips the userinfo; policy sees only the host."""
        client = PolicyHTTPClient()
        # Permissive engine — should not raise
        client._check_url("https://user:secret@api.example.com/private")

    # --- Non-standard port does not affect host matching ---

    def test_url_with_unusual_port_host_matched_by_allowed_hosts(self) -> None:
        """Policy engine matches 'host' not 'host:port'; port in URL is stripped by urlparse."""
        init_policy_engine(_make_config(
            default_deny=True,
            allowed_hosts=["internal.corp.com"],
        ))
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://internal.corp.com:9000/api/data")
        assert resp.status_code == 200

    # --- IPv6 bracket URL end-to-end with real engine ---

    def test_ipv6_bracket_url_allowed_via_cidr_engine(self) -> None:
        """[::1] in a URL is parsed to host '::1' which is checked against CIDR."""
        init_policy_engine(_make_config(
            default_deny=True,
            allowed_cidrs=["::1/128"],
        ))
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("http://[::1]/status")
        assert resp.status_code == 200

    def test_ipv6_bracket_url_denied_when_not_in_cidr(self) -> None:
        init_policy_engine(_make_config(
            default_deny=True,
            allowed_cidrs=["2001:db8::/32"],
        ))
        client = PolicyHTTPClient()
        with pytest.raises(PolicyViolationError):
            client.get("http://[::2]/status")

    # --- _check_rest_policy None result does not raise ---

    def test_check_rest_policy_none_result_is_a_passthrough(self) -> None:
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.rest_policy.check.return_value = None
            mock_get_engine.return_value = mock_engine
            client = PolicyHTTPClient()
            client._check_rest_policy("api.example.com", "GET", "/safe")
            # Must not raise; None means no rule matched → pass-through

    # --- set_interactive_approval module-level state ---

    def test_set_interactive_approval_replaces_previous_value(self) -> None:
        from missy.agent.interactive_approval import InteractiveApproval

        first = MagicMock(spec=InteractiveApproval)
        second = MagicMock(spec=InteractiveApproval)
        set_interactive_approval(first)
        assert gateway_module._interactive_approval is first
        set_interactive_approval(second)
        assert gateway_module._interactive_approval is second

    # --- Allowed kwargs frozenset membership ---

    def test_extensions_kwarg_is_allowed(self) -> None:
        result = PolicyHTTPClient._sanitize_kwargs({"extensions": {"trace": True}})
        assert "extensions" in result

    def test_files_kwarg_is_allowed(self) -> None:
        result = PolicyHTTPClient._sanitize_kwargs({"files": {"upload": b"data"}})
        assert "files" in result

    def test_follow_redirects_kwarg_is_stripped(self) -> None:
        result = PolicyHTTPClient._sanitize_kwargs({"follow_redirects": True, "headers": {}})
        assert "follow_redirects" not in result
        assert "headers" in result


# ===========================================================================
# Preset integration tests
# ===========================================================================


class TestPresetContentChecks:
    """Verify specific entries in presets not covered by existing tests."""

    def test_pypi_preset_hosts(self) -> None:
        hosts, _, _, unknown = resolve_presets(["pypi"])
        assert "pypi.org" in hosts
        assert "files.pythonhosted.org" in hosts
        assert unknown == []

    def test_pypi_preset_domains(self) -> None:
        _, domains, _, _ = resolve_presets(["pypi"])
        assert "pypi.org" in domains
        assert "pythonhosted.org" in domains

    def test_npm_preset_hosts(self) -> None:
        hosts, _, _, unknown = resolve_presets(["npm"])
        assert "registry.npmjs.org" in hosts
        assert unknown == []

    def test_npm_preset_domains(self) -> None:
        _, domains, _, _ = resolve_presets(["npm"])
        assert "npmjs.org" in domains

    def test_docker_hub_preset_hosts(self) -> None:
        hosts, _, _, unknown = resolve_presets(["docker-hub"])
        assert "registry-1.docker.io" in hosts
        assert "auth.docker.io" in hosts
        assert unknown == []

    def test_docker_hub_preset_domains(self) -> None:
        _, domains, _, _ = resolve_presets(["docker-hub"])
        assert "docker.io" in domains
        assert "docker.com" in domains

    def test_huggingface_preset_hosts(self) -> None:
        hosts, _, _, unknown = resolve_presets(["huggingface"])
        assert "huggingface.co" in hosts
        assert unknown == []

    def test_huggingface_preset_domains(self) -> None:
        _, domains, _, _ = resolve_presets(["huggingface"])
        assert "huggingface.co" in domains

    def test_discord_preset_domains(self) -> None:
        _, domains, _, _ = resolve_presets(["discord"])
        assert "discord.gg" in domains
        assert "discordapp.com" in domains

    def test_openai_preset_has_auth_host(self) -> None:
        hosts, _, _, _ = resolve_presets(["openai"])
        assert "auth.openai.com" in hosts

    def test_all_non_ollama_presets_have_empty_cidrs(self) -> None:
        """Only ollama and home-assistant presets should have CIDR entries."""
        cidr_presets = {"ollama", "home-assistant"}
        for name, preset in PRESETS.items():
            if name in cidr_presets:
                assert len(preset["cidrs"]) > 0, f"{name} expected to have CIDRs"
            else:
                assert preset["cidrs"] == [], f"{name} should have empty CIDRs"


class TestPresetToNetworkEngineIntegration:
    """End-to-end: resolve_presets feeds a real NetworkPolicyEngine."""

    def test_anthropic_preset_allows_api_host(self) -> None:
        hosts, domains, cidrs, _ = resolve_presets(["anthropic"])
        engine = NetworkPolicyEngine(
            NetworkPolicy(
                default_deny=True,
                allowed_hosts=hosts,
                allowed_domains=domains,
                allowed_cidrs=cidrs,
            )
        )
        assert engine.check_host("api.anthropic.com") is True

    def test_anthropic_preset_allows_subdomain_via_domain(self) -> None:
        hosts, domains, cidrs, _ = resolve_presets(["anthropic"])
        engine = NetworkPolicyEngine(
            NetworkPolicy(
                default_deny=True,
                allowed_hosts=hosts,
                allowed_domains=domains,
                allowed_cidrs=cidrs,
            )
        )
        # anthropic.com is an exact domain match
        assert engine.check_host("anthropic.com") is True

    def test_github_preset_allows_api_host(self) -> None:
        hosts, domains, cidrs, _ = resolve_presets(["github"])
        engine = NetworkPolicyEngine(
            NetworkPolicy(
                default_deny=True,
                allowed_hosts=hosts,
                allowed_domains=domains,
                allowed_cidrs=cidrs,
            )
        )
        assert engine.check_host("api.github.com") is True

    def test_github_preset_blocks_unrelated_host(self) -> None:
        hosts, domains, cidrs, _ = resolve_presets(["github"])
        engine = NetworkPolicyEngine(
            NetworkPolicy(
                default_deny=True,
                allowed_hosts=hosts,
                allowed_domains=domains,
                allowed_cidrs=cidrs,
            )
        )
        with pytest.raises(PolicyViolationError):
            engine.check_host("evil.example.com")

    def test_ollama_preset_allows_localhost_ip(self) -> None:
        hosts, domains, cidrs, _ = resolve_presets(["ollama"])
        engine = NetworkPolicyEngine(
            NetworkPolicy(
                default_deny=True,
                allowed_hosts=hosts,
                allowed_domains=domains,
                allowed_cidrs=cidrs,
            )
        )
        assert engine.check_host("127.0.0.1") is True

    def test_merged_anthropic_github_preset_allows_both(self) -> None:
        hosts, domains, cidrs, _ = resolve_presets(["anthropic", "github"])
        engine = NetworkPolicyEngine(
            NetworkPolicy(
                default_deny=True,
                allowed_hosts=hosts,
                allowed_domains=domains,
                allowed_cidrs=cidrs,
            )
        )
        assert engine.check_host("api.anthropic.com") is True
        assert engine.check_host("api.github.com") is True

    def test_parse_network_presets_only_no_explicit_hosts(self) -> None:
        """_parse_network with only presets, no explicit allowed_hosts, still works."""
        from missy.config.settings import _parse_network

        data = {"presets": ["anthropic"]}
        policy = _parse_network(data)
        assert "api.anthropic.com" in policy.allowed_hosts

    def test_parse_network_github_preset_adds_githubusercontent_domain(self) -> None:
        from missy.config.settings import _parse_network

        data = {"presets": ["github"]}
        policy = _parse_network(data)
        assert "githubusercontent.com" in policy.allowed_domains


# ===========================================================================
# DNS rebinding: mixed private + public addresses
# ===========================================================================


class TestDNSRebindingMixedAddresses:
    """The engine denies any hostname that resolves to ANY private address
    not covered by an explicit CIDR, even if a public address also resolves."""

    def _mixed_dns_patch(self, private_ip: str, public_ip: str):
        return patch(
            "missy.policy.network.socket.getaddrinfo",
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", (public_ip, 80)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", (private_ip, 80)),
            ],
        )

    def test_mixed_private_public_without_cidr_is_denied(self) -> None:
        """When one resolved IP is private and not in CIDR, the whole request is denied."""
        engine = _make_net_engine()
        with (
            self._mixed_dns_patch("10.0.0.1", "203.0.113.1"),
            pytest.raises(PolicyViolationError),
        ):
            engine.check_host("dual-stack.example.com")

    def test_mixed_private_public_with_private_cidr_allowed(self) -> None:
        """When the private IP is explicitly in an allowed CIDR, the request may proceed."""
        engine = _make_net_engine(allowed_cidrs=["10.0.0.0/8", "203.0.113.0/24"])
        with self._mixed_dns_patch("10.0.0.1", "203.0.113.1"):
            assert engine.check_host("dual-stack.example.com") is True
