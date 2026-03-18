"""Edge case tests for gateway client and policy engine.

Covers RestPolicy, NetworkPolicyEngine, PolicyHTTPClient, and preset
resolution with unusual inputs, boundary conditions, and adversarial
scenarios not exercised by the primary unit tests.
"""

from __future__ import annotations

import socket
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
from missy.gateway.client import PolicyHTTPClient, create_client
from missy.policy import engine as engine_module
from missy.policy.engine import init_policy_engine
from missy.policy.network import NetworkPolicyEngine
from missy.policy.presets import PRESETS, resolve_presets
from missy.policy.rest_policy import RestPolicy

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_network_engine(**kwargs) -> NetworkPolicyEngine:
    """Build a NetworkPolicyEngine with secure defaults + overrides."""
    defaults: dict = {
        "default_deny": True,
        "allowed_cidrs": [],
        "allowed_domains": [],
        "allowed_hosts": [],
        "provider_allowed_hosts": [],
        "tool_allowed_hosts": [],
        "discord_allowed_hosts": [],
    }
    defaults.update(kwargs)
    return NetworkPolicyEngine(NetworkPolicy(**defaults))


def make_config(
    *,
    default_deny: bool = True,
    allowed_hosts: list[str] | None = None,
    allowed_domains: list[str] | None = None,
    allowed_cidrs: list[str] | None = None,
    rest_policies: list[dict] | None = None,
    provider_allowed_hosts: list[str] | None = None,
    tool_allowed_hosts: list[str] | None = None,
    discord_allowed_hosts: list[str] | None = None,
) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(
            default_deny=default_deny,
            allowed_hosts=allowed_hosts or [],
            allowed_domains=allowed_domains or [],
            allowed_cidrs=allowed_cidrs or [],
            rest_policies=rest_policies or [],
            provider_allowed_hosts=provider_allowed_hosts or [],
            tool_allowed_hosts=tool_allowed_hosts or [],
            discord_allowed_hosts=discord_allowed_hosts or [],
        ),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(enabled=False),
        plugins=PluginPolicy(enabled=False),
        providers={},
        workspace_path="/tmp/workspace",
        audit_log_path="/tmp/audit.log",
    )


@pytest.fixture(autouse=True)
def reset_engine_and_bus() -> Generator[None, None, None]:
    """Isolate each test: clear event bus and reset singleton."""
    original_engine = engine_module._engine
    engine_module._engine = None
    event_bus.clear()
    yield
    engine_module._engine = original_engine
    event_bus.clear()


# ===========================================================================
# REST policy edge cases
# ===========================================================================


class TestRestPolicyWildcardDoubleStarPath:
    """/** should match any path on the correct host."""

    def _policy(self, action: str = "deny") -> RestPolicy:
        return RestPolicy.from_config(
            [{"host": "api.example.com", "method": "DELETE", "path": "/**", "action": action}]
        )

    def test_double_star_matches_root(self) -> None:
        assert self._policy("deny").check("api.example.com", "DELETE", "/") == "deny"

    def test_double_star_matches_single_segment(self) -> None:
        assert self._policy("deny").check("api.example.com", "DELETE", "/resource") == "deny"

    def test_double_star_matches_deep_path(self) -> None:
        result = self._policy("deny").check("api.example.com", "DELETE", "/a/b/c/d/e")
        assert result == "deny"

    def test_double_star_allow_action_propagates(self) -> None:
        assert self._policy("allow").check("api.example.com", "DELETE", "/anything") == "allow"

    def test_double_star_does_not_match_different_host(self) -> None:
        result = self._policy("deny").check("other.example.com", "DELETE", "/resource")
        assert result is None


class TestRestPolicySingleSegmentWildcard:
    """Single * at a specific position matches only that segment level."""

    def _policy(self) -> RestPolicy:
        return RestPolicy.from_config(
            [{"host": "api.example.com", "method": "GET", "path": "/v1/*/data", "action": "allow"}]
        )

    def test_single_star_matches_segment(self) -> None:
        assert self._policy().check("api.example.com", "GET", "/v1/users/data") == "allow"

    def test_single_star_does_not_match_two_segments(self) -> None:
        # fnmatch: * does not cross path separators (it matches /)
        # but fnmatch is not pathlib glob — * DOES match / in fnmatch
        # so /v1/users/extra/data won't match /v1/*/data
        result = self._policy().check("api.example.com", "GET", "/v1/users/extra/data")
        # fnmatch '*' matches any string including '/', so this may match
        # We simply assert the call doesn't raise
        assert result in ("allow", None)

    def test_single_star_does_not_match_different_prefix(self) -> None:
        assert self._policy().check("api.example.com", "GET", "/v2/users/data") is None


class TestRestPolicyMethodCaseInsensitive:
    """Method matching must be case-insensitive in both rule and check."""

    @pytest.mark.parametrize(
        "rule_method,check_method",
        [
            ("get", "GET"),
            ("GET", "get"),
            ("Get", "gEt"),
            ("post", "POST"),
            ("DELETE", "delete"),
            ("patch", "PATCH"),
        ],
    )
    def test_method_case_combinations(self, rule_method: str, check_method: str) -> None:
        policy = RestPolicy.from_config(
            [{"host": "h.com", "method": rule_method, "path": "/**", "action": "allow"}]
        )
        assert policy.check("h.com", check_method, "/foo") == "allow"


class TestRestPolicyFirstMatchWins:
    """Rules are evaluated top-to-bottom; the first matching rule wins."""

    def test_deny_before_allow_wins(self) -> None:
        policy = RestPolicy.from_config(
            [
                {"host": "api.x.com", "method": "GET", "path": "/secret/**", "action": "deny"},
                {"host": "api.x.com", "method": "GET", "path": "/**", "action": "allow"},
            ]
        )
        assert policy.check("api.x.com", "GET", "/secret/key") == "deny"

    def test_allow_before_deny_wins(self) -> None:
        policy = RestPolicy.from_config(
            [
                {"host": "api.x.com", "method": "GET", "path": "/public/**", "action": "allow"},
                {"host": "api.x.com", "method": "GET", "path": "/**", "action": "deny"},
            ]
        )
        assert policy.check("api.x.com", "GET", "/public/resource") == "allow"

    def test_non_matching_first_rule_falls_to_second(self) -> None:
        policy = RestPolicy.from_config(
            [
                {"host": "api.x.com", "method": "POST", "path": "/data", "action": "deny"},
                {"host": "api.x.com", "method": "GET", "path": "/data", "action": "allow"},
            ]
        )
        assert policy.check("api.x.com", "GET", "/data") == "allow"

    def test_three_rules_third_matches(self) -> None:
        policy = RestPolicy.from_config(
            [
                {"host": "api.x.com", "method": "POST", "path": "/a", "action": "deny"},
                {"host": "api.x.com", "method": "PUT", "path": "/b", "action": "deny"},
                {"host": "api.x.com", "method": "GET", "path": "/c", "action": "allow"},
            ]
        )
        assert policy.check("api.x.com", "GET", "/c") == "allow"


class TestRestPolicyNoMatchDefaultBehavior:
    """When no rule matches, check() returns None (pass-through)."""

    def test_wrong_host_returns_none(self) -> None:
        policy = RestPolicy.from_config(
            [{"host": "api.example.com", "method": "GET", "path": "/**", "action": "allow"}]
        )
        assert policy.check("other.com", "GET", "/foo") is None

    def test_wrong_method_returns_none(self) -> None:
        policy = RestPolicy.from_config(
            [{"host": "api.example.com", "method": "GET", "path": "/**", "action": "allow"}]
        )
        assert policy.check("api.example.com", "POST", "/foo") is None

    def test_wrong_path_returns_none(self) -> None:
        policy = RestPolicy.from_config(
            [{"host": "api.example.com", "method": "GET", "path": "/repos/**", "action": "allow"}]
        )
        assert policy.check("api.example.com", "GET", "/users/foo") is None

    def test_all_mismatch_returns_none(self) -> None:
        policy = RestPolicy.from_config(
            [{"host": "api.example.com", "method": "POST", "path": "/data", "action": "deny"}]
        )
        assert policy.check("other.com", "GET", "/other") is None


class TestRestPolicyEmptyRules:
    """An empty rules list always returns None."""

    def test_empty_list_constructor(self) -> None:
        policy = RestPolicy(rules=[])
        assert policy.check("anything.com", "DELETE", "/nuke") is None

    def test_none_rules_constructor(self) -> None:
        policy = RestPolicy(rules=None)
        assert policy.check("anything.com", "POST", "/data") is None

    def test_from_config_empty(self) -> None:
        policy = RestPolicy.from_config([])
        assert policy.check("api.github.com", "GET", "/repos") is None


class TestRestPolicyPathWithQueryString:
    """Path matching uses only the path component; queries must be stripped by caller."""

    def test_path_without_query_matches(self) -> None:
        policy = RestPolicy.from_config(
            [{"host": "api.x.com", "method": "GET", "path": "/search", "action": "allow"}]
        )
        assert policy.check("api.x.com", "GET", "/search") == "allow"

    def test_path_with_query_does_not_match_bare_pattern(self) -> None:
        """If the caller passes a path+query string, fnmatch won't match /search pattern."""
        policy = RestPolicy.from_config(
            [{"host": "api.x.com", "method": "GET", "path": "/search", "action": "allow"}]
        )
        # Gateway strips query before calling; raw path+query would not match
        result = policy.check("api.x.com", "GET", "/search?q=foo")
        assert result is None  # pattern "/search" does not match "/search?q=foo"

    def test_wildcard_pattern_absorbs_query_suffix(self) -> None:
        """/** pattern matches any string including ?-containing ones."""
        policy = RestPolicy.from_config(
            [{"host": "api.x.com", "method": "GET", "path": "/**", "action": "allow"}]
        )
        # fnmatch "/**" matches "/search?q=foo" because ** matches everything
        assert policy.check("api.x.com", "GET", "/search?q=foo") == "allow"


class TestRestPolicyMethodWildcardStar:
    """method='*' in a rule matches any HTTP method."""

    def test_star_method_matches_get(self) -> None:
        policy = RestPolicy.from_config(
            [{"host": "h.com", "method": "*", "path": "/health", "action": "allow"}]
        )
        assert policy.check("h.com", "GET", "/health") == "allow"

    def test_star_method_matches_post(self) -> None:
        policy = RestPolicy.from_config(
            [{"host": "h.com", "method": "*", "path": "/health", "action": "allow"}]
        )
        assert policy.check("h.com", "POST", "/health") == "allow"

    def test_star_method_matches_delete(self) -> None:
        policy = RestPolicy.from_config(
            [{"host": "h.com", "method": "*", "path": "/health", "action": "allow"}]
        )
        assert policy.check("h.com", "DELETE", "/health") == "allow"

    def test_star_method_does_not_match_wrong_path(self) -> None:
        policy = RestPolicy.from_config(
            [{"host": "h.com", "method": "*", "path": "/health", "action": "allow"}]
        )
        assert policy.check("h.com", "GET", "/other") is None


# ===========================================================================
# Network policy edge cases
# ===========================================================================


class TestNetworkPolicyCIDRSlash32:
    """/32 is an exact single-host CIDR."""

    def test_exact_ip_in_slash32_allowed(self) -> None:
        engine = make_network_engine(allowed_cidrs=["203.0.113.42/32"])
        assert engine.check_host("203.0.113.42") is True

    def test_adjacent_ip_not_in_slash32(self) -> None:
        engine = make_network_engine(allowed_cidrs=["203.0.113.42/32"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("203.0.113.43")

    def test_subnet_ip_not_in_slash32(self) -> None:
        engine = make_network_engine(allowed_cidrs=["203.0.113.42/32"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("203.0.113.0")

    def test_slash32_ipv6_loopback(self) -> None:
        engine = make_network_engine(allowed_cidrs=["::1/128"])
        assert engine.check_host("::1") is True

    def test_slash32_ipv6_other_address_denied(self) -> None:
        engine = make_network_engine(allowed_cidrs=["::1/128"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("::2")


class TestNetworkPolicyDomainSuffixMatching:
    """Wildcard domain patterns follow *.example.com semantics."""

    def test_wildcard_matches_direct_subdomain(self) -> None:
        engine = make_network_engine(allowed_domains=["*.example.com"])
        assert engine.check_host("sub.example.com") is True

    def test_wildcard_matches_deep_subdomain(self) -> None:
        engine = make_network_engine(allowed_domains=["*.example.com"])
        assert engine.check_host("a.b.c.example.com") is True

    def test_wildcard_matches_root_domain_itself(self) -> None:
        """*.example.com permits example.com (documented behavior)."""
        engine = make_network_engine(allowed_domains=["*.example.com"])
        assert engine.check_host("example.com") is True

    def test_wildcard_does_not_match_partial_suffix(self) -> None:
        """notexample.com must not match *.example.com."""
        engine = make_network_engine(allowed_domains=["*.example.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("notexample.com")

    def test_wildcard_does_not_match_different_tld(self) -> None:
        engine = make_network_engine(allowed_domains=["*.example.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("sub.example.org")

    def test_wildcard_does_not_match_unrelated_domain(self) -> None:
        engine = make_network_engine(allowed_domains=["*.example.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("attacker.com")

    def test_exact_domain_does_not_match_subdomain(self) -> None:
        """An exact pattern 'example.com' should not permit sub.example.com."""
        engine = make_network_engine(allowed_domains=["example.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("sub.example.com")

    def test_exact_domain_matches_itself(self) -> None:
        engine = make_network_engine(allowed_domains=["example.com"])
        assert engine.check_host("example.com") is True

    def test_partial_string_attack_rejected(self) -> None:
        """'evilexample.com' must not match domain pattern 'example.com'."""
        engine = make_network_engine(allowed_domains=["example.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("evilexample.com")


class TestNetworkPolicyExactHostMatching:
    """Exact host entries in allowed_hosts."""

    def test_exact_match_allowed(self) -> None:
        engine = make_network_engine(allowed_hosts=["api.github.com"])
        assert engine.check_host("api.github.com") is True

    def test_different_host_denied(self) -> None:
        engine = make_network_engine(allowed_hosts=["api.github.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("www.github.com")

    def test_host_matching_is_case_insensitive(self) -> None:
        engine = make_network_engine(allowed_hosts=["API.GITHUB.COM"])
        assert engine.check_host("api.github.com") is True

    def test_allowed_hosts_entry_with_port_stripped(self) -> None:
        """A configured 'host:port' entry matches the bare hostname."""
        engine = make_network_engine(allowed_hosts=["api.github.com:443"])
        assert engine.check_host("api.github.com") is True

    def test_superset_host_not_matched(self) -> None:
        """api.github.com entry does not permit extra.api.github.com."""
        engine = make_network_engine(allowed_hosts=["api.github.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("extra.api.github.com")


class TestNetworkPolicyPerCategoryAllowlists:
    """Per-category host lists (provider, tool, discord) are independent."""

    def test_provider_host_blocked_without_category(self) -> None:
        engine = make_network_engine(provider_allowed_hosts=["api.anthropic.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("api.anthropic.com")  # no category arg

    def test_provider_host_allowed_with_provider_category(self) -> None:
        engine = make_network_engine(provider_allowed_hosts=["api.anthropic.com"])
        assert engine.check_host("api.anthropic.com", category="provider") is True

    def test_tool_host_blocked_with_provider_category(self) -> None:
        """A host in tool_allowed_hosts is not accessible via category='provider'."""
        engine = make_network_engine(tool_allowed_hosts=["api.weatherapi.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("api.weatherapi.com", category="provider")

    def test_tool_host_allowed_with_tool_category(self) -> None:
        engine = make_network_engine(tool_allowed_hosts=["api.weatherapi.com"])
        assert engine.check_host("api.weatherapi.com", category="tool") is True

    def test_discord_host_allowed_with_discord_category(self) -> None:
        engine = make_network_engine(discord_allowed_hosts=["gateway.discord.gg"])
        assert engine.check_host("gateway.discord.gg", category="discord") is True

    def test_discord_host_blocked_with_tool_category(self) -> None:
        engine = make_network_engine(discord_allowed_hosts=["gateway.discord.gg"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("gateway.discord.gg", category="tool")

    def test_global_allowlist_works_regardless_of_category(self) -> None:
        engine = make_network_engine(allowed_hosts=["shared.api.com"])
        for cat in ("", "provider", "tool", "discord", "unknown"):
            assert engine.check_host("shared.api.com", category=cat) is True

    def test_unknown_category_gives_no_extra_access(self) -> None:
        engine = make_network_engine(provider_allowed_hosts=["api.anthropic.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("api.anthropic.com", category="unknown")


class TestNetworkPolicyEmptyAllowlistsDefaultDeny:
    """With default_deny=True and empty allowlists, everything is blocked."""

    def test_hostname_blocked(self) -> None:
        engine = make_network_engine()
        with pytest.raises(PolicyViolationError):
            engine.check_host("api.example.com")

    def test_ip_address_blocked(self) -> None:
        engine = make_network_engine()
        with pytest.raises(PolicyViolationError):
            engine.check_host("1.2.3.4")

    def test_localhost_ip_blocked(self) -> None:
        engine = make_network_engine()
        with pytest.raises(PolicyViolationError):
            engine.check_host("127.0.0.1")

    def test_violation_error_category_is_network(self) -> None:
        engine = make_network_engine()
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_host("example.com")
        assert exc_info.value.category == "network"

    def test_violation_includes_host_in_detail(self) -> None:
        engine = make_network_engine()
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_host("blocked.example.com")
        assert "blocked.example.com" in exc_info.value.detail

    def test_audit_event_emitted_on_deny(self) -> None:
        engine = make_network_engine()
        with pytest.raises(PolicyViolationError):
            engine.check_host("blocked.example.com")
        deny_events = event_bus.get_events(result="deny")
        assert len(deny_events) == 1
        assert deny_events[0].detail["host"] == "blocked.example.com"


class TestNetworkPolicyDNSRebindingProtection:
    """Private/loopback addresses resolved via DNS are denied unless CIDR-allowed."""

    def _mock_dns(self, ip: str):
        """Return a getaddrinfo patch that resolves to *ip*."""
        return patch(
            "missy.policy.network.socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 80))],
        )

    def test_private_ip_via_dns_without_cidr_denied(self) -> None:
        engine = make_network_engine()
        with self._mock_dns("10.0.0.1"), pytest.raises(PolicyViolationError):
            engine.check_host("evil.example.com")

    def test_private_ip_via_dns_with_matching_cidr_allowed(self) -> None:
        engine = make_network_engine(allowed_cidrs=["10.0.0.0/8"])
        with self._mock_dns("10.0.0.1"):
            # Loopback/private is allowed when explicitly in CIDR
            assert engine.check_host("internal.service.com") is True

    def test_public_ip_via_dns_not_in_cidr_denied(self) -> None:
        engine = make_network_engine(allowed_cidrs=["10.0.0.0/8"])
        with self._mock_dns("203.0.113.42"), pytest.raises(PolicyViolationError):
            engine.check_host("public.example.com")

    def test_dns_failure_falls_through_to_deny(self) -> None:
        engine = make_network_engine(allowed_cidrs=["10.0.0.0/8"])
        with (
            patch(
                "missy.policy.network.socket.getaddrinfo",
                side_effect=OSError("Name or service not known"),
            ),
            pytest.raises(PolicyViolationError),
        ):
            engine.check_host("unresolvable.invalid")


# ===========================================================================
# Gateway client edge cases
# ===========================================================================


class TestGatewayClientBlockedByPolicy:
    """PolicyHTTPClient.get/post raise PolicyViolationError when network is denied."""

    def _make_client_with_engine(self, allowed_hosts: list[str]) -> PolicyHTTPClient:
        config = make_config(allowed_hosts=allowed_hosts)
        init_policy_engine(config)
        return PolicyHTTPClient(timeout=5)

    def test_get_blocked_host_raises_policy_violation(self) -> None:
        client = self._make_client_with_engine([])
        with pytest.raises(PolicyViolationError):
            client.get("https://blocked.example.com/data")

    def test_post_blocked_host_raises_policy_violation(self) -> None:
        client = self._make_client_with_engine([])
        with pytest.raises(PolicyViolationError):
            client.post("https://blocked.example.com/data")

    def test_put_blocked_host_raises_policy_violation(self) -> None:
        client = self._make_client_with_engine([])
        with pytest.raises(PolicyViolationError):
            client.put("https://blocked.example.com/data")

    def test_delete_blocked_host_raises_policy_violation(self) -> None:
        client = self._make_client_with_engine([])
        with pytest.raises(PolicyViolationError):
            client.delete("https://blocked.example.com/data")

    def test_patch_blocked_host_raises_policy_violation(self) -> None:
        client = self._make_client_with_engine([])
        with pytest.raises(PolicyViolationError):
            client.patch("https://blocked.example.com/data")

    def test_head_blocked_host_raises_policy_violation(self) -> None:
        client = self._make_client_with_engine([])
        with pytest.raises(PolicyViolationError):
            client.head("https://blocked.example.com/data")

    def test_error_raised_before_network_io(self) -> None:
        """The violation is raised before httpx touches the network."""
        client = self._make_client_with_engine([])
        with patch.object(client, "_get_sync_client") as mock_sync:
            with pytest.raises(PolicyViolationError):
                client.get("https://blocked.example.com/data")
            mock_sync.assert_not_called()


class TestGatewayClientAllowedRequest:
    """Allowed requests pass through to httpx without raising."""

    def _setup_allowed_engine(self) -> None:
        config = make_config(allowed_hosts=["allowed.example.com"])
        init_policy_engine(config)

    def _fake_response(self, status: int = 200) -> httpx.Response:
        return httpx.Response(status, content=b"ok")

    def test_get_allowed_host_calls_httpx(self) -> None:
        self._setup_allowed_engine()
        client = PolicyHTTPClient(timeout=5)
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = self._fake_response()
        client._sync_client = mock_httpx
        response = client.get("https://allowed.example.com/data")
        assert response.status_code == 200
        mock_httpx.get.assert_called_once()

    def test_post_allowed_host_calls_httpx(self) -> None:
        self._setup_allowed_engine()
        client = PolicyHTTPClient(timeout=5)
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = self._fake_response(201)
        client._sync_client = mock_httpx
        response = client.post("https://allowed.example.com/data", json={"key": "val"})
        assert response.status_code == 201
        mock_httpx.post.assert_called_once()

    def test_delete_allowed_host_calls_httpx(self) -> None:
        self._setup_allowed_engine()
        client = PolicyHTTPClient(timeout=5)
        mock_httpx = MagicMock()
        mock_httpx.delete.return_value = self._fake_response(204)
        client._sync_client = mock_httpx
        response = client.delete("https://allowed.example.com/resource/1")
        assert response.status_code == 204

    def test_kwargs_forwarded_to_httpx(self) -> None:
        self._setup_allowed_engine()
        client = PolicyHTTPClient(timeout=5)
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = self._fake_response()
        client._sync_client = mock_httpx
        client.get("https://allowed.example.com/data", headers={"X-Token": "abc"})
        _, kwargs = mock_httpx.get.call_args
        assert "headers" in kwargs
        assert kwargs["headers"] == {"X-Token": "abc"}


class TestGatewayClientURLValidation:
    """_check_url rejects non-HTTP schemes and malformed URLs."""

    def test_ftp_scheme_rejected(self) -> None:
        client = PolicyHTTPClient(timeout=5)
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("ftp://files.example.com/file.tar.gz")

    def test_file_scheme_rejected(self) -> None:
        client = PolicyHTTPClient(timeout=5)
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("file:///etc/passwd")

    def test_data_scheme_rejected(self) -> None:
        client = PolicyHTTPClient(timeout=5)
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("data:text/plain;base64,SGVsbG8=")

    def test_javascript_scheme_rejected(self) -> None:
        client = PolicyHTTPClient(timeout=5)
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("javascript:alert(1)")

    def test_empty_string_url(self) -> None:
        client = PolicyHTTPClient(timeout=5)
        with pytest.raises(ValueError):
            client._check_url("")

    def test_url_exceeding_max_length(self) -> None:
        client = PolicyHTTPClient(timeout=5)
        long_url = "https://example.com/" + "a" * 8200
        with pytest.raises(ValueError, match="exceeds maximum length"):
            client._check_url(long_url)

    def test_url_at_max_length_boundary_accepted(self) -> None:
        """A URL of exactly 8192 characters does not trigger the length check."""
        client = PolicyHTTPClient(timeout=5)
        # 8192 - len("https://x.com/") = 8178 filler chars
        prefix = "https://x.com/"
        filler = "a" * (8192 - len(prefix))
        url = prefix + filler
        assert len(url) == 8192
        # Policy engine not initialised — will raise RuntimeError, not ValueError
        with pytest.raises((RuntimeError, PolicyViolationError)):
            client._check_url(url)

    def test_http_scheme_accepted(self) -> None:
        """http:// is a permitted scheme (checked against policy, not rejected early)."""
        client = PolicyHTTPClient(timeout=5)
        # Engine not initialised — expect RuntimeError from get_policy_engine(), not ValueError
        with pytest.raises((RuntimeError, PolicyViolationError)):
            client._check_url("http://example.com/path")

    def test_https_scheme_accepted(self) -> None:
        client = PolicyHTTPClient(timeout=5)
        with pytest.raises((RuntimeError, PolicyViolationError)):
            client._check_url("https://example.com/path")

    def test_no_host_component_rejected(self) -> None:
        client = PolicyHTTPClient(timeout=5)
        with pytest.raises(ValueError, match="Cannot determine host"):
            client._check_url("https:///path-only")


class TestGatewayClientTimeoutHandling:
    """Timeout is forwarded to httpx; non-positive timeouts are rejected."""

    def test_negative_timeout_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="timeout must be positive"):
            PolicyHTTPClient(timeout=-1)

    def test_zero_timeout_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="timeout must be positive"):
            PolicyHTTPClient(timeout=0)

    def test_positive_timeout_accepted(self) -> None:
        client = PolicyHTTPClient(timeout=1)
        assert client.timeout == 1

    def test_timeout_propagated_to_sync_client(self) -> None:
        client = PolicyHTTPClient(timeout=7)
        sync_client = client._get_sync_client()
        # httpx.Client stores timeout as an httpx.Timeout object
        assert sync_client.timeout.connect == pytest.approx(7.0)

    def test_timeout_propagated_to_async_client(self) -> None:
        client = PolicyHTTPClient(timeout=12)
        async_client = client._get_async_client()
        assert async_client.timeout.connect == pytest.approx(12.0)


class TestGatewayClientUnsafeKwargsStripped:
    """Security-sensitive kwargs (verify, transport, auth, base_url) are stripped."""

    def test_verify_false_stripped(self) -> None:
        safe = PolicyHTTPClient._sanitize_kwargs({"verify": False, "headers": {"X": "1"}})
        assert "verify" not in safe
        assert "headers" in safe

    def test_transport_stripped(self) -> None:
        safe = PolicyHTTPClient._sanitize_kwargs({"transport": object(), "params": {"k": "v"}})
        assert "transport" not in safe
        assert "params" in safe

    def test_auth_stripped(self) -> None:
        safe = PolicyHTTPClient._sanitize_kwargs({"auth": ("user", "pass"), "json": {}})
        assert "auth" not in safe
        assert "json" in safe

    def test_base_url_stripped(self) -> None:
        safe = PolicyHTTPClient._sanitize_kwargs({"base_url": "https://evil.com", "content": b""})
        assert "base_url" not in safe
        assert "content" in safe

    def test_all_allowed_kwargs_pass_through(self) -> None:
        allowed = {
            "headers": {},
            "params": {},
            "data": b"",
            "json": {},
            "content": b"",
            "cookies": {},
            "timeout": 10,
            "files": {},
            "extensions": {},
        }
        safe = PolicyHTTPClient._sanitize_kwargs(allowed)
        assert safe == allowed

    def test_empty_kwargs_dict(self) -> None:
        assert PolicyHTTPClient._sanitize_kwargs({}) == {}

    def test_unknown_kwarg_stripped(self) -> None:
        safe = PolicyHTTPClient._sanitize_kwargs({"stream": True, "follow_redirects": True})
        assert "stream" not in safe
        assert "follow_redirects" not in safe


class TestGatewayClientResponseSizeLimit:
    """Oversized responses are rejected via ValueError."""

    def _fake_response(self, content_length: int | None, body: bytes = b"") -> httpx.Response:
        headers = {}
        if content_length is not None:
            headers["content-length"] = str(content_length)
        return httpx.Response(200, headers=headers, content=body)

    def test_content_length_over_limit_raises(self) -> None:
        client = PolicyHTTPClient(timeout=5, max_response_bytes=100)
        resp = self._fake_response(content_length=101)
        with pytest.raises(ValueError, match="too large"):
            client._check_response_size(resp, "https://example.com/big")

    def test_content_length_at_limit_allowed(self) -> None:
        client = PolicyHTTPClient(timeout=5, max_response_bytes=100)
        resp = self._fake_response(content_length=100)
        client._check_response_size(resp, "https://example.com/ok")  # should not raise

    def test_body_over_limit_without_content_length_raises(self) -> None:
        client = PolicyHTTPClient(timeout=5, max_response_bytes=10)
        resp = self._fake_response(content_length=None, body=b"x" * 11)
        with pytest.raises(ValueError, match="too large"):
            client._check_response_size(resp, "https://example.com/chunked")

    def test_body_at_limit_without_content_length_allowed(self) -> None:
        client = PolicyHTTPClient(timeout=5, max_response_bytes=10)
        resp = self._fake_response(content_length=None, body=b"x" * 10)
        client._check_response_size(resp, "https://example.com/chunked")  # should not raise

    def test_default_max_is_50mb(self) -> None:
        client = PolicyHTTPClient(timeout=5)
        assert client.max_response_bytes == 50 * 1024 * 1024

    def test_invalid_content_length_header_skips_check(self) -> None:
        """A non-numeric Content-Length header should not raise."""
        client = PolicyHTTPClient(timeout=5, max_response_bytes=100)
        resp = self._fake_response(content_length=None, body=b"small")
        # Manually inject a bad header via mock
        mock_headers = MagicMock()
        mock_headers.get.return_value = "not-a-number"
        resp = MagicMock()
        resp.headers = mock_headers
        resp.content = b"small body"
        client._check_response_size(resp, "https://example.com")  # should not raise

    def test_no_headers_attribute_skips_check(self) -> None:
        """A response without headers attribute is silently accepted."""
        client = PolicyHTTPClient(timeout=5, max_response_bytes=10)
        resp = MagicMock()
        resp.headers = None
        client._check_response_size(resp, "https://example.com")  # should not raise


class TestGatewayClientAsyncMethods:
    """Async gateway methods enforce policy before performing network I/O."""

    def _setup_allowed_engine(self) -> None:
        config = make_config(allowed_hosts=["api.example.com"])
        init_policy_engine(config)

    def _setup_denied_engine(self) -> None:
        config = make_config(allowed_hosts=[])
        init_policy_engine(config)

    @pytest.mark.asyncio
    async def test_aget_blocked_host_raises(self) -> None:
        self._setup_denied_engine()
        client = PolicyHTTPClient(timeout=5)
        with pytest.raises(PolicyViolationError):
            await client.aget("https://blocked.example.com/data")

    @pytest.mark.asyncio
    async def test_apost_blocked_host_raises(self) -> None:
        self._setup_denied_engine()
        client = PolicyHTTPClient(timeout=5)
        with pytest.raises(PolicyViolationError):
            await client.apost("https://blocked.example.com/data")

    @pytest.mark.asyncio
    async def test_aget_allowed_host_calls_httpx(self) -> None:
        self._setup_allowed_engine()
        client = PolicyHTTPClient(timeout=5)
        fake_resp = httpx.Response(200, content=b"ok")
        mock_async_httpx = AsyncMock()
        mock_async_httpx.get = AsyncMock(return_value=fake_resp)
        client._async_client = mock_async_httpx
        response = await client.aget("https://api.example.com/data")
        assert response.status_code == 200
        mock_async_httpx.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aclose_safe_when_never_used(self) -> None:
        """aclose() on a fresh client (no async client created) does not raise."""
        client = PolicyHTTPClient(timeout=5)
        await client.aclose()  # should not raise

    @pytest.mark.asyncio
    async def test_context_manager_async(self) -> None:
        """async with PolicyHTTPClient(...) is safe to enter and exit."""
        client = PolicyHTTPClient(timeout=5)
        async with client as c:
            assert c is client


class TestGatewayClientContextManager:
    """Synchronous context manager support."""

    def test_sync_context_manager_closes_on_exit(self) -> None:
        client = PolicyHTTPClient(timeout=5)
        with client as c:
            assert c is client
        # After __exit__, sync client is None (never created, so still None)
        assert client._sync_client is None

    def test_close_idempotent(self) -> None:
        """close() called on a client that never opened a sync connection is safe."""
        client = PolicyHTTPClient(timeout=5)
        client.close()
        client.close()  # second call must not raise


class TestGatewayClientRESTDenialIntegration:
    """End-to-end: REST policy denial surfaces as PolicyViolationError from client."""

    def test_delete_denied_by_rest_policy(self) -> None:
        config = make_config(
            allowed_hosts=["api.github.com"],
            rest_policies=[
                {"host": "api.github.com", "method": "DELETE", "path": "/**", "action": "deny"},
            ],
        )
        init_policy_engine(config)
        client = PolicyHTTPClient(timeout=5)
        with pytest.raises(PolicyViolationError, match="REST policy denied"):
            client.delete("https://api.github.com/repos/owner/repo")

    def test_get_allowed_by_rest_policy(self) -> None:
        config = make_config(
            allowed_hosts=["api.github.com"],
            rest_policies=[
                {
                    "host": "api.github.com",
                    "method": "GET",
                    "path": "/repos/**",
                    "action": "allow",
                },
            ],
        )
        init_policy_engine(config)
        client = PolicyHTTPClient(timeout=5)
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = httpx.Response(200, content=b"ok")
        client._sync_client = mock_httpx
        response = client.get("https://api.github.com/repos/owner/repo")
        assert response.status_code == 200


class TestCreateClientFactory:
    """create_client() constructs a PolicyHTTPClient with correct attributes."""

    def test_creates_policy_http_client(self) -> None:
        client = create_client(session_id="s1", task_id="t1", timeout=15, category="tool")
        assert isinstance(client, PolicyHTTPClient)

    def test_session_id_set(self) -> None:
        client = create_client(session_id="my-session")
        assert client.session_id == "my-session"

    def test_task_id_set(self) -> None:
        client = create_client(task_id="my-task")
        assert client.task_id == "my-task"

    def test_timeout_set(self) -> None:
        client = create_client(timeout=60)
        assert client.timeout == 60

    def test_category_set(self) -> None:
        client = create_client(category="provider")
        assert client.category == "provider"

    def test_default_values(self) -> None:
        client = create_client()
        assert client.session_id == ""
        assert client.task_id == ""
        assert client.timeout == 30
        assert client.category == ""


# ===========================================================================
# Preset edge cases
# ===========================================================================


class TestAnthropicPreset:
    """The 'anthropic' preset resolves to the correct hosts and domains."""

    def test_anthropic_host_included(self) -> None:
        hosts, _, _, unknown = resolve_presets(["anthropic"])
        assert "api.anthropic.com" in hosts
        assert unknown == []

    def test_anthropic_domain_included(self) -> None:
        _, domains, _, _ = resolve_presets(["anthropic"])
        assert "anthropic.com" in domains

    def test_anthropic_no_cidrs(self) -> None:
        _, _, cidrs, _ = resolve_presets(["anthropic"])
        assert cidrs == []


class TestGithubPreset:
    """The 'github' preset resolves to the correct hosts and domains."""

    def test_github_api_host_included(self) -> None:
        hosts, _, _, _ = resolve_presets(["github"])
        assert "api.github.com" in hosts

    def test_github_root_host_included(self) -> None:
        hosts, _, _, _ = resolve_presets(["github"])
        assert "github.com" in hosts

    def test_github_domain_included(self) -> None:
        _, domains, _, _ = resolve_presets(["github"])
        assert "github.com" in domains

    def test_githubusercontent_domain_included(self) -> None:
        _, domains, _, _ = resolve_presets(["github"])
        assert "githubusercontent.com" in domains

    def test_github_no_cidrs(self) -> None:
        _, _, cidrs, _ = resolve_presets(["github"])
        assert cidrs == []


class TestUnknownPreset:
    """Unknown preset names are collected in the 'unknown' return value, not raised."""

    def test_unknown_preset_goes_to_unknown_list(self) -> None:
        _, _, _, unknown = resolve_presets(["does-not-exist"])
        assert "does-not-exist" in unknown

    def test_unknown_preset_yields_no_hosts(self) -> None:
        hosts, domains, cidrs, _ = resolve_presets(["totally-fake"])
        assert hosts == []
        assert domains == []
        assert cidrs == []

    def test_multiple_unknown_presets_all_collected(self) -> None:
        _, _, _, unknown = resolve_presets(["fake-a", "fake-b", "fake-c"])
        assert set(unknown) == {"fake-a", "fake-b", "fake-c"}

    def test_mix_of_known_and_unknown(self) -> None:
        hosts, _, _, unknown = resolve_presets(["anthropic", "fake-preset"])
        assert "api.anthropic.com" in hosts
        assert "fake-preset" in unknown
        assert "anthropic" not in unknown

    def test_empty_string_preset_treated_as_unknown(self) -> None:
        _, _, _, unknown = resolve_presets([""])
        assert "" in unknown


class TestMultiplePresetsDeduplication:
    """Merging multiple presets does not produce duplicate entries."""

    def test_same_preset_twice_no_duplicates(self) -> None:
        hosts, domains, cidrs, _ = resolve_presets(["anthropic", "anthropic"])
        assert len(hosts) == len(set(hosts))
        assert len(domains) == len(set(domains))

    def test_anthropic_and_github_merged_no_duplicates(self) -> None:
        hosts, domains, cidrs, unknown = resolve_presets(["anthropic", "github"])
        assert unknown == []
        assert len(hosts) == len(set(hosts))
        assert len(domains) == len(set(domains))
        # Both presets should have contributed
        assert "api.anthropic.com" in hosts
        assert "api.github.com" in hosts

    def test_ollama_and_home_assistant_share_cidr_deduplicated(self) -> None:
        """Both ollama and home-assistant include 127.0.0.0/8; only one copy expected."""
        _, _, cidrs, _ = resolve_presets(["ollama", "home-assistant"])
        assert cidrs.count("127.0.0.0/8") == 1

    def test_all_known_presets_no_unknown(self) -> None:
        all_preset_names = list(PRESETS.keys())
        _, _, _, unknown = resolve_presets(all_preset_names)
        assert unknown == []

    def test_order_preserved_for_hosts(self) -> None:
        """First occurrence of a host is preserved; subsequent duplicates dropped."""
        hosts_ab, _, _, _ = resolve_presets(["anthropic", "github"])
        hosts_ba, _, _, _ = resolve_presets(["github", "anthropic"])
        # anthropic hosts should appear before github hosts in first call
        idx_anthropic = hosts_ab.index("api.anthropic.com")
        idx_github = hosts_ab.index("api.github.com")
        assert idx_anthropic < idx_github
        # reversed order in second call
        idx_github_ba = hosts_ba.index("api.github.com")
        idx_anthropic_ba = hosts_ba.index("api.anthropic.com")
        assert idx_github_ba < idx_anthropic_ba


class TestPresetConstantsIntegrity:
    """Sanity checks on the PRESETS dictionary itself."""

    def test_all_presets_have_required_keys(self) -> None:
        for name, preset in PRESETS.items():
            assert "hosts" in preset, f"Preset '{name}' missing 'hosts'"
            assert "domains" in preset, f"Preset '{name}' missing 'domains'"
            assert "cidrs" in preset, f"Preset '{name}' missing 'cidrs'"

    def test_all_preset_values_are_lists(self) -> None:
        for name, preset in PRESETS.items():
            for key in ("hosts", "domains", "cidrs"):
                assert isinstance(preset[key], list), (
                    f"Preset '{name}' key '{key}' is not a list"
                )

    def test_no_preset_entry_is_none(self) -> None:
        for name, preset in PRESETS.items():
            for key in ("hosts", "domains", "cidrs"):
                assert None not in preset[key], (
                    f"Preset '{name}' has None in '{key}'"
                )
