"""Deep integration tests for the gateway and policy systems working together.

These tests instantiate *real* policy engine objects (no mocks for the engines
themselves) and wire them to a real :class:`PolicyHTTPClient`.  Network I/O is
fully mocked via ``unittest.mock`` patches on ``httpx.Client`` /
``httpx.AsyncClient``, so the suite runs entirely offline.

Coverage targets
----------------
1.  PolicyHTTPClient respects network policy — allowed hosts pass, denied blocked.
2.  REST policy method filtering — GET allowed, DELETE denied on same host.
3.  REST policy path globbing — /repos/** allowed, /admin/** denied.
4.  Multiple preset expansion — anthropic + github presets expand correctly.
5.  Default deny with no presets — all requests blocked.
6.  CIDR allowlist — IP-based requests allowed/denied per CIDR.
7.  Domain suffix matching — *.example.com allows sub.example.com.
8.  PolicyEngine facade — check_network delegates to NetworkPolicyEngine.
9.  InteractiveApproval session memory — "allow always" remembered per session.
10. Rate limiter integration — RateLimiter gates calls to the gateway client.
11. Gateway error handling — httpx timeout, connection error, and HTTP errors.
12. Policy audit events — network violations generate deny audit events.
"""

from __future__ import annotations

import hashlib
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from missy.agent.interactive_approval import InteractiveApproval
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
from missy.policy.engine import PolicyEngine, get_policy_engine, init_policy_engine
from missy.policy.network import NetworkPolicyEngine
from missy.policy.presets import PRESETS, resolve_presets
from missy.policy.rest_policy import RestPolicy, RestRule
from missy.providers.rate_limiter import RateLimiter, RateLimitExceeded

# ---------------------------------------------------------------------------
# Test-wide fixtures and helpers
# ---------------------------------------------------------------------------


def _build_config(
    *,
    default_deny: bool = True,
    allowed_hosts: list[str] | None = None,
    allowed_domains: list[str] | None = None,
    allowed_cidrs: list[str] | None = None,
    provider_allowed_hosts: list[str] | None = None,
    tool_allowed_hosts: list[str] | None = None,
    discord_allowed_hosts: list[str] | None = None,
    rest_policies: list[dict] | None = None,
) -> MissyConfig:
    """Build a minimal :class:`MissyConfig` with the supplied network settings."""
    return MissyConfig(
        network=NetworkPolicy(
            default_deny=default_deny,
            allowed_hosts=allowed_hosts or [],
            allowed_domains=allowed_domains or [],
            allowed_cidrs=allowed_cidrs or [],
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


def _mock_http_response(status_code: int = 200, body: str = "ok") -> MagicMock:
    """Return a minimal fake :class:`httpx.Response`."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = body
    resp.headers = {}
    resp.content = body.encode()
    return resp


@pytest.fixture(autouse=True)
def _reset_engine_and_bus() -> Generator[None, None, None]:
    """Isolate every test: restore original engine + clear the event bus."""
    original_engine = engine_module._engine
    original_approval = gateway_module._interactive_approval
    event_bus.clear()
    yield
    engine_module._engine = original_engine
    gateway_module._interactive_approval = original_approval
    event_bus.clear()


# ===========================================================================
# 1. PolicyHTTPClient respects network policy
# ===========================================================================


class TestNetworkPolicyEnforcement:
    """The gateway client enforces host-level allow/deny before any I/O."""

    def test_allowed_host_get_succeeds(self) -> None:
        """A host in allowed_hosts must reach the underlying httpx.Client."""
        init_policy_engine(_build_config(allowed_hosts=["api.example.com"]))
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp) as mock_get:
            resp = client.get("https://api.example.com/data")
        assert resp.status_code == 200
        mock_get.assert_called_once()

    def test_denied_host_raises_before_network_io(self) -> None:
        """A host not in any allowlist must raise PolicyViolationError; httpx
        must never be invoked."""
        init_policy_engine(_build_config(default_deny=True))
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get") as mock_get:
            with pytest.raises(PolicyViolationError) as exc_info:
                client.get("https://evil.attacker.io/steal")
            mock_get.assert_not_called()
        assert exc_info.value.category == "network"

    def test_post_to_allowed_host_succeeds(self) -> None:
        """POST on an allowed host must go through to httpx."""
        init_policy_engine(_build_config(allowed_hosts=["api.example.com"]))
        client = PolicyHTTPClient(session_id="s1", task_id="t1")
        mock_resp = _mock_http_response(201)
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            resp = client.post("https://api.example.com/items", json={"x": 1})
        assert resp.status_code == 201

    def test_denied_host_post_blocked(self) -> None:
        """POST to a denied host must raise PolicyViolationError."""
        init_policy_engine(_build_config(default_deny=True))
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "post") as mock_post:
            with pytest.raises(PolicyViolationError):
                client.post("https://nope.example.com/data")
            mock_post.assert_not_called()

    def test_delete_to_allowed_host_succeeds(self) -> None:
        """DELETE on an allowed host (without REST restrictions) passes."""
        init_policy_engine(_build_config(allowed_hosts=["api.example.com"]))
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(204)
        with patch.object(httpx.Client, "delete", return_value=mock_resp):
            resp = client.delete("https://api.example.com/items/1")
        assert resp.status_code == 204

    def test_patch_to_denied_host_blocked(self) -> None:
        """PATCH to a host not in any allowlist must be blocked."""
        init_policy_engine(_build_config(default_deny=True))
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "patch") as mock_patch:
            with pytest.raises(PolicyViolationError):
                client.patch("https://denied.example.com/item/1")
            mock_patch.assert_not_called()

    async def test_async_get_allowed_host_succeeds(self) -> None:
        """Async GET on an allowed host must reach httpx.AsyncClient."""
        init_policy_engine(_build_config(allowed_hosts=["api.example.com"]))
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(200)
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp):
            resp = await client.aget("https://api.example.com/status")
        assert resp.status_code == 200

    async def test_async_get_denied_host_blocked(self) -> None:
        """Async GET on a denied host raises before any coroutine is awaited."""
        init_policy_engine(_build_config(default_deny=True))
        client = PolicyHTTPClient()
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            with pytest.raises(PolicyViolationError):
                await client.aget("https://evil.example.com/")
            mock_get.assert_not_called()

    def test_create_client_factory_respects_policy(self) -> None:
        """create_client() must produce a client that honours the installed engine."""
        init_policy_engine(_build_config(default_deny=True))
        client = create_client(session_id="s", task_id="t")
        with patch.object(httpx.Client, "get"), pytest.raises(PolicyViolationError):
            client.get("https://blocked.example.com/")


# ===========================================================================
# 2. REST policy method filtering
# ===========================================================================


class TestRestPolicyMethodFiltering:
    """GET is allowed; DELETE is denied on the same host via REST rules."""

    def _github_rest_config(self) -> MissyConfig:
        return _build_config(
            allowed_hosts=["api.github.com"],
            rest_policies=[
                {"host": "api.github.com", "method": "GET", "path": "/**", "action": "allow"},
                {"host": "api.github.com", "method": "DELETE", "path": "/**", "action": "deny"},
            ],
        )

    def test_get_allowed_by_rest_rule(self) -> None:
        """GET /repos/foo passes because the allow-GET rule matches first."""
        init_policy_engine(self._github_rest_config())
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://api.github.com/repos/foo/bar")
        assert resp.status_code == 200

    def test_delete_denied_by_rest_rule(self) -> None:
        """DELETE on the same host is blocked by the deny-DELETE rule."""
        init_policy_engine(self._github_rest_config())
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "delete") as mock_del:
            with pytest.raises(PolicyViolationError) as exc_info:
                client.delete("https://api.github.com/repos/foo/bar")
            mock_del.assert_not_called()
        assert "REST policy denied" in str(exc_info.value)

    def test_post_no_matching_rest_rule_passes_through(self) -> None:
        """POST has no explicit REST rule — network policy already allows host,
        so no REST denial is produced and the request is forwarded."""
        init_policy_engine(self._github_rest_config())
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(201)
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            resp = client.post("https://api.github.com/repos/foo/issues", json={})
        assert resp.status_code == 201

    def test_rest_deny_raises_policy_violation_error(self) -> None:
        """The raised exception must carry category='network'."""
        init_policy_engine(self._github_rest_config())
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "delete"), pytest.raises(PolicyViolationError) as exc_info:
            client.delete("https://api.github.com/repos/x/y")
        assert exc_info.value.category == "network"

    def test_method_comparison_is_case_insensitive(self) -> None:
        """RestPolicy normalises methods to uppercase; client must send uppercase."""
        # Build a policy that allows GET explicitly
        rest_policy = RestPolicy.from_config(
            [{"host": "api.example.com", "method": "get", "path": "/**", "action": "deny"}]
        )
        # Method "get" in config becomes "GET" after normalisation
        result = rest_policy.check("api.example.com", "GET", "/anything")
        assert result == "deny"

    async def test_async_delete_denied_by_rest_rule(self) -> None:
        """Async DELETE must also be blocked by a deny-DELETE REST rule."""
        init_policy_engine(self._github_rest_config())
        client = PolicyHTTPClient()
        with patch.object(httpx.AsyncClient, "delete", new_callable=AsyncMock) as mock_del:
            with pytest.raises(PolicyViolationError):
                await client.adelete("https://api.github.com/repos/foo/bar")
            mock_del.assert_not_called()


# ===========================================================================
# 3. REST policy path globbing
# ===========================================================================


class TestRestPolicyPathGlobbing:
    """/repos/** allowed, /admin/** denied on the same host."""

    def _path_glob_config(self) -> MissyConfig:
        return _build_config(
            allowed_hosts=["api.github.com"],
            rest_policies=[
                {
                    "host": "api.github.com",
                    "method": "*",
                    "path": "/repos/**",
                    "action": "allow",
                },
                {
                    "host": "api.github.com",
                    "method": "*",
                    "path": "/admin/**",
                    "action": "deny",
                },
            ],
        )

    def test_repos_path_allowed(self) -> None:
        """GET /repos/org/repo returns 200."""
        init_policy_engine(self._path_glob_config())
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://api.github.com/repos/myorg/myrepo")
        assert resp.status_code == 200

    def test_admin_path_denied(self) -> None:
        """GET /admin/settings must be blocked by the deny rule."""
        init_policy_engine(self._path_glob_config())
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get") as mock_get:
            with pytest.raises(PolicyViolationError):
                client.get("https://api.github.com/admin/settings")
            mock_get.assert_not_called()

    def test_repos_deep_path_allowed(self) -> None:
        """GET /repos/org/repo/issues/42 matches /repos/**."""
        init_policy_engine(self._path_glob_config())
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://api.github.com/repos/org/repo/issues/42")
        assert resp.status_code == 200

    def test_admin_deep_path_denied(self) -> None:
        """POST /admin/users/delete also matches /admin/**."""
        init_policy_engine(self._path_glob_config())
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "post") as mock_post:
            with pytest.raises(PolicyViolationError):
                client.post("https://api.github.com/admin/users/delete")
            mock_post.assert_not_called()

    def test_unmatched_path_falls_through_network_policy(self) -> None:
        """A path that matches no REST rule falls through to the (passing)
        network policy and the request is forwarded."""
        init_policy_engine(self._path_glob_config())
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://api.github.com/zen")
        assert resp.status_code == 200

    def test_rest_policy_path_glob_unit(self) -> None:
        """RestPolicy.check correctly matches fnmatch globs."""
        policy = RestPolicy.from_config(
            [
                {"host": "api.example.com", "method": "*", "path": "/data/**", "action": "allow"},
                {"host": "api.example.com", "method": "*", "path": "/secret/**", "action": "deny"},
            ]
        )
        assert policy.check("api.example.com", "GET", "/data/record/1") == "allow"
        assert policy.check("api.example.com", "DELETE", "/secret/token") == "deny"
        assert policy.check("api.example.com", "GET", "/public/info") is None

    def test_first_matching_rest_rule_wins(self) -> None:
        """Rules are evaluated top-to-bottom; the first match wins."""
        policy = RestPolicy(
            rules=[
                RestRule(host="api.example.com", method="*", path="/v1/**", action="allow"),
                RestRule(host="api.example.com", method="*", path="/v1/admin/**", action="deny"),
            ]
        )
        # /v1/admin/secret matches the first rule (allow) before reaching deny.
        assert policy.check("api.example.com", "GET", "/v1/admin/secret") == "allow"


# ===========================================================================
# 4. Multiple preset expansion
# ===========================================================================


class TestPresetExpansion:
    """anthropic + github presets expand to their documented host/domain lists."""

    def test_anthropic_preset_contains_expected_host(self) -> None:
        """The anthropic preset must always include api.anthropic.com."""
        hosts, domains, cidrs, unknown = resolve_presets(["anthropic"])
        assert "api.anthropic.com" in hosts
        assert not unknown

    def test_github_preset_contains_expected_hosts(self) -> None:
        """The github preset must include api.github.com and github.com."""
        hosts, domains, cidrs, unknown = resolve_presets(["github"])
        assert "api.github.com" in hosts
        assert "github.com" in hosts
        assert not unknown

    def test_combined_presets_are_deduplicated(self) -> None:
        """Overlapping entries across presets must appear only once."""
        hosts, domains, cidrs, unknown = resolve_presets(["anthropic", "github"])
        assert len(hosts) == len(set(hosts))
        assert len(domains) == len(set(domains))

    def test_combined_presets_include_both_services(self) -> None:
        """The merged result must cover hosts from both services."""
        hosts, domains, cidrs, unknown = resolve_presets(["anthropic", "github"])
        assert "api.anthropic.com" in hosts
        assert "api.github.com" in hosts

    def test_unknown_preset_reported(self) -> None:
        """An unrecognised preset name must appear in the unknown list."""
        _, _, _, unknown = resolve_presets(["anthropic", "made_up_preset"])
        assert "made_up_preset" in unknown

    def test_anthropic_preset_allows_api_via_network_engine(self) -> None:
        """When the anthropic preset is used, api.anthropic.com must pass."""
        hosts, domains, cidrs, _ = resolve_presets(["anthropic"])
        policy = NetworkPolicy(
            default_deny=True,
            allowed_hosts=hosts,
            allowed_domains=domains,
            allowed_cidrs=cidrs,
        )
        engine = NetworkPolicyEngine(policy)
        assert engine.check_host("api.anthropic.com") is True

    def test_anthropic_preset_blocks_random_host(self) -> None:
        """Only the preset's listed hosts are allowed; others remain blocked."""
        hosts, domains, cidrs, _ = resolve_presets(["anthropic"])
        policy = NetworkPolicy(
            default_deny=True,
            allowed_hosts=hosts,
            allowed_domains=domains,
            allowed_cidrs=cidrs,
        )
        engine = NetworkPolicyEngine(policy)
        with pytest.raises(PolicyViolationError):
            engine.check_host("api.evil.com")

    def test_github_preset_allows_api_github_via_gateway_client(self) -> None:
        """Integration: github preset wired into engine + gateway passes GET."""
        hosts, domains, cidrs, _ = resolve_presets(["github"])
        init_policy_engine(
            _build_config(
                allowed_hosts=hosts,
                allowed_domains=domains,
                allowed_cidrs=cidrs,
            )
        )
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://api.github.com/repos/octocat/hello-world")
        assert resp.status_code == 200

    def test_all_known_presets_resolve_without_error(self) -> None:
        """Every key in PRESETS must resolve without producing unknowns."""
        all_names = list(PRESETS.keys())
        _, _, _, unknown = resolve_presets(all_names)
        assert unknown == []

    def test_empty_preset_list_produces_empty_results(self) -> None:
        """An empty preset list must produce empty hosts/domains/cidrs."""
        hosts, domains, cidrs, unknown = resolve_presets([])
        assert hosts == []
        assert domains == []
        assert cidrs == []
        assert unknown == []


# ===========================================================================
# 5. Default deny with no presets
# ===========================================================================


class TestDefaultDenyNoPresets:
    """With default_deny=True and no allowlists, every request is blocked."""

    def test_get_to_any_host_blocked(self) -> None:
        """Even a request to a well-known host must raise when nothing is allowed."""
        init_policy_engine(_build_config(default_deny=True))
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get") as mock_get:
            with pytest.raises(PolicyViolationError):
                client.get("https://google.com/")
            mock_get.assert_not_called()

    def test_post_blocked(self) -> None:
        init_policy_engine(_build_config(default_deny=True))
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "post") as mock_post:
            with pytest.raises(PolicyViolationError):
                client.post("https://api.example.com/v1/data")
            mock_post.assert_not_called()

    async def test_async_get_blocked(self) -> None:
        init_policy_engine(_build_config(default_deny=True))
        client = PolicyHTTPClient()
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            with pytest.raises(PolicyViolationError):
                await client.aget("https://example.com/")
            mock_get.assert_not_called()

    def test_violation_category_is_network(self) -> None:
        init_policy_engine(_build_config(default_deny=True))
        client = PolicyHTTPClient()
        with pytest.raises(PolicyViolationError) as exc_info:
            client.get("https://blocked.example.com/")
        assert exc_info.value.category == "network"

    def test_default_allow_mode_permits_any_host(self) -> None:
        """Sanity: default_deny=False must let every host through."""
        init_policy_engine(_build_config(default_deny=False))
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://anything-at-all.example.com/")
        assert resp.status_code == 200


# ===========================================================================
# 6. CIDR allowlist
# ===========================================================================


class TestCidrAllowlist:
    """IP-based allowlists work end-to-end through the gateway."""

    def test_ip_in_cidr_allowed_via_network_engine(self) -> None:
        """10.5.0.1 is inside 10.0.0.0/8 and must be allowed."""
        policy = NetworkPolicy(
            default_deny=True,
            allowed_cidrs=["10.0.0.0/8"],
        )
        engine = NetworkPolicyEngine(policy)
        assert engine.check_host("10.5.0.1") is True

    def test_ip_outside_cidr_denied_via_network_engine(self) -> None:
        """8.8.8.8 is not inside 10.0.0.0/8 and must be denied."""
        policy = NetworkPolicy(
            default_deny=True,
            allowed_cidrs=["10.0.0.0/8"],
        )
        engine = NetworkPolicyEngine(policy)
        with pytest.raises(PolicyViolationError):
            engine.check_host("8.8.8.8")

    def test_ip_in_cidr_passes_gateway_client(self) -> None:
        """A request to a CIDR-allowed IP must reach httpx."""
        init_policy_engine(_build_config(allowed_cidrs=["192.168.1.0/24"]))
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("http://192.168.1.100/api/data")
        assert resp.status_code == 200

    def test_ip_outside_cidr_blocked_by_gateway(self) -> None:
        """A request to an IP not in any CIDR must be blocked."""
        init_policy_engine(_build_config(allowed_cidrs=["192.168.1.0/24"]))
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get") as mock_get:
            with pytest.raises(PolicyViolationError):
                client.get("http://172.16.0.1/api/data")
            mock_get.assert_not_called()

    def test_multiple_cidrs_second_matches(self) -> None:
        """When the first CIDR misses, the second must still allow the IP."""
        init_policy_engine(
            _build_config(allowed_cidrs=["10.0.0.0/8", "172.16.0.0/12"])
        )
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("http://172.20.5.1/ping")
        assert resp.status_code == 200

    def test_ipv6_loopback_in_cidr_allowed(self) -> None:
        """::1 is inside ::1/128 and must be permitted."""
        policy = NetworkPolicy(
            default_deny=True,
            allowed_cidrs=["::1/128"],
        )
        engine = NetworkPolicyEngine(policy)
        assert engine.check_host("::1") is True

    def test_cidr_audit_event_contains_matching_cidr(self) -> None:
        """The allow audit event must carry the matching CIDR in policy_rule."""
        policy = NetworkPolicy(default_deny=True, allowed_cidrs=["10.0.0.0/8"])
        engine = NetworkPolicyEngine(policy)
        engine.check_host("10.1.2.3")
        events = event_bus.get_events(result="allow")
        assert len(events) == 1
        assert events[0].policy_rule == "cidr:10.0.0.0/8"


# ===========================================================================
# 7. Domain suffix matching
# ===========================================================================


class TestDomainSuffixMatching:
    """*.example.com must allow sub.example.com and example.com itself."""

    def test_wildcard_allows_direct_subdomain(self) -> None:
        policy = NetworkPolicy(default_deny=True, allowed_domains=["*.example.com"])
        engine = NetworkPolicyEngine(policy)
        assert engine.check_host("sub.example.com") is True

    def test_wildcard_allows_root_domain(self) -> None:
        """*.example.com also covers example.com itself."""
        policy = NetworkPolicy(default_deny=True, allowed_domains=["*.example.com"])
        engine = NetworkPolicyEngine(policy)
        assert engine.check_host("example.com") is True

    def test_wildcard_allows_deep_subdomain(self) -> None:
        policy = NetworkPolicy(default_deny=True, allowed_domains=["*.example.com"])
        engine = NetworkPolicyEngine(policy)
        assert engine.check_host("a.b.example.com") is True

    def test_wildcard_blocks_different_root(self) -> None:
        """*.example.com must not cover notexample.com."""
        policy = NetworkPolicy(default_deny=True, allowed_domains=["*.example.com"])
        engine = NetworkPolicyEngine(policy)
        with pytest.raises(PolicyViolationError):
            engine.check_host("notexample.com")

    def test_exact_domain_does_not_cover_subdomain(self) -> None:
        """An exact domain entry must not implicitly cover subdomains."""
        policy = NetworkPolicy(default_deny=True, allowed_domains=["example.com"])
        engine = NetworkPolicyEngine(policy)
        with pytest.raises(PolicyViolationError):
            engine.check_host("sub.example.com")

    def test_domain_suffix_gateway_integration(self) -> None:
        """End-to-end: wildcard domain in engine allows request via gateway."""
        init_policy_engine(_build_config(allowed_domains=["*.example.com"]))
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://api.example.com/v1/health")
        assert resp.status_code == 200

    def test_domain_suffix_blocks_other_domain_via_gateway(self) -> None:
        """Only *.example.com is allowed; requests to other domains are blocked."""
        init_policy_engine(_build_config(allowed_domains=["*.example.com"]))
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get") as mock_get:
            with pytest.raises(PolicyViolationError):
                client.get("https://other-domain.io/data")
            mock_get.assert_not_called()

    def test_domain_allow_event_carries_rule(self) -> None:
        """Allow events produced by domain matching must record the rule."""
        policy = NetworkPolicy(default_deny=True, allowed_domains=["*.example.com"])
        engine = NetworkPolicyEngine(policy)
        engine.check_host("api.example.com")
        events = event_bus.get_events(result="allow")
        assert events[0].policy_rule == "domain:*.example.com"


# ===========================================================================
# 8. PolicyEngine facade delegates correctly
# ===========================================================================


class TestPolicyEngineFacade:
    """PolicyEngine.check_network must delegate to the NetworkPolicyEngine."""

    def _make_engine(
        self,
        allowed_hosts: list[str] | None = None,
        allowed_domains: list[str] | None = None,
    ) -> PolicyEngine:
        config = _build_config(
            allowed_hosts=allowed_hosts or [],
            allowed_domains=allowed_domains or [],
        )
        return PolicyEngine(config)

    def test_check_network_allows_listed_host(self) -> None:
        engine = self._make_engine(allowed_hosts=["api.example.com"])
        assert engine.check_network("api.example.com") is True

    def test_check_network_denies_unlisted_host(self) -> None:
        engine = self._make_engine()
        with pytest.raises(PolicyViolationError):
            engine.check_network("evil.com")

    def test_check_network_propagates_session_and_task_ids(self) -> None:
        """session_id and task_id must appear in the emitted audit event."""
        engine = self._make_engine(allowed_hosts=["api.example.com"])
        engine.check_network("api.example.com", session_id="s-99", task_id="t-42")
        events = event_bus.get_events()
        assert any(e.session_id == "s-99" and e.task_id == "t-42" for e in events)

    def test_check_network_propagates_category(self) -> None:
        """When a category matches per-category hosts the request is allowed."""
        config = _build_config(
            provider_allowed_hosts=["api.special.com"],
        )
        engine = PolicyEngine(config)
        assert engine.check_network("api.special.com", category="provider") is True

    def test_facade_exposes_rest_policy(self) -> None:
        """The engine must expose a .rest_policy attribute."""
        engine = self._make_engine(allowed_hosts=["api.example.com"])
        assert hasattr(engine, "rest_policy")
        assert isinstance(engine.rest_policy, RestPolicy)

    def test_facade_rest_policy_deny_integration(self) -> None:
        """check_network passes but REST policy fires afterwards inside _check_url."""
        config = _build_config(
            allowed_hosts=["api.github.com"],
            rest_policies=[
                {"host": "api.github.com", "method": "DELETE", "path": "/**", "action": "deny"},
            ],
        )
        init_policy_engine(config)
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "delete") as mock_del:
            with pytest.raises(PolicyViolationError) as exc_info:
                client.delete("https://api.github.com/repos/x")
            mock_del.assert_not_called()
        assert "REST policy denied" in str(exc_info.value)

    def test_singleton_init_get_roundtrip(self) -> None:
        """init_policy_engine followed by get_policy_engine must return the
        same object."""
        config = _build_config(allowed_hosts=["api.example.com"])
        installed = init_policy_engine(config)
        assert get_policy_engine() is installed

    def test_second_init_replaces_singleton(self) -> None:
        """Calling init_policy_engine a second time must replace the engine."""
        config_a = _build_config(allowed_hosts=["a.example.com"])
        config_b = _build_config(allowed_hosts=["b.example.com"])
        engine_a = init_policy_engine(config_a)
        engine_b = init_policy_engine(config_b)
        assert get_policy_engine() is engine_b
        assert engine_b is not engine_a


# ===========================================================================
# 9. InteractiveApproval session memory
# ===========================================================================


class TestInteractiveApprovalSessionMemory:
    """allow-always decisions are remembered for the session; deny is not."""

    def test_check_remembered_returns_none_initially(self) -> None:
        approval = InteractiveApproval()
        result = approval.check_remembered("network_request", "https://example.com")
        assert result is None

    def test_allow_always_decision_is_remembered(self) -> None:
        """After an 'a' response, the same action+detail must return True."""
        approval = InteractiveApproval()
        # Simulate the 'allow always' path by calling the internal method.
        key = approval._make_key("network_request", "https://example.com")
        with approval._lock:
            approval._remembered[key] = True
        assert approval.check_remembered("network_request", "https://example.com") is True

    def test_remembered_decision_prevents_new_prompt(self) -> None:
        """prompt_user must return the remembered True without re-prompting."""
        approval = InteractiveApproval()
        key = approval._make_key("network_request", "https://example.com")
        with approval._lock:
            approval._remembered[key] = True
        with patch.object(approval, "_do_prompt") as mock_prompt:
            result = approval.prompt_user("network_request", "https://example.com")
        mock_prompt.assert_not_called()
        assert result is True

    def test_non_tty_auto_denies(self) -> None:
        """In a non-TTY environment, prompt_user must return False without
        interacting with the terminal."""
        approval = InteractiveApproval()
        with patch.object(InteractiveApproval, "_is_tty", return_value=False):
            result = approval.prompt_user("network_request", "https://example.com")
        assert result is False

    def test_different_action_detail_pairs_have_different_keys(self) -> None:
        """Distinct action+detail combinations must not share a memory slot."""
        approval = InteractiveApproval()
        key1 = approval._make_key("network_request", "https://a.example.com")
        key2 = approval._make_key("network_request", "https://b.example.com")
        assert key1 != key2

    def test_same_action_detail_produces_stable_key(self) -> None:
        """The same action+detail must always produce the same key (SHA-256)."""
        approval = InteractiveApproval()
        k1 = approval._make_key("network_request", "https://example.com")
        k2 = approval._make_key("network_request", "https://example.com")
        assert k1 == k2
        expected = hashlib.sha256(b"network_request:https://example.com").hexdigest()
        assert k1 == expected

    def test_interactive_approval_overrides_gateway_denial(self) -> None:
        """When the interactive approval instance returns True, the gateway
        must NOT raise even for a host blocked by network policy.

        _check_url passes the full URL to prompt_user, so the remembered key
        must be seeded with the exact URL string used in the request.
        """
        init_policy_engine(_build_config(default_deny=True))
        approval = InteractiveApproval()
        # The gateway calls prompt_user("network_request", url) where url is
        # the full URL string including path.
        request_url = "https://blocked.example.com/data"
        key = approval._make_key("network_request", request_url)
        with approval._lock:
            approval._remembered[key] = True
        set_interactive_approval(approval)
        # prompt_user will return True from the remembered decision,
        # so _check_url skips the raise.
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get(request_url)
        assert resp.status_code == 200

    def test_interactive_approval_none_does_not_override_denial(self) -> None:
        """With no approval instance set, a denied host must still raise."""
        init_policy_engine(_build_config(default_deny=True))
        set_interactive_approval(None)
        client = PolicyHTTPClient()
        with pytest.raises(PolicyViolationError):
            client.get("https://blocked.example.com/data")

    def test_allow_always_response_triggers_remember(self) -> None:
        """Simulating an 'a' response in _do_prompt must persist the decision.

        Console is imported inside _do_prompt from rich.console, so we patch
        it at the rich.console source rather than on the module namespace.
        """
        approval = InteractiveApproval()
        mock_console = MagicMock()
        mock_console.input.return_value = "a"
        with (
            patch.object(InteractiveApproval, "_is_tty", return_value=True),
            patch("rich.console.Console", return_value=mock_console),
        ):
            result = approval.prompt_user("network_request", "https://example.com")
        assert result is True
        assert approval.check_remembered("network_request", "https://example.com") is True


# ===========================================================================
# 10. Rate limiter integration
# ===========================================================================


class TestRateLimiterIntegration:
    """RateLimiter gates calls; exceeded budget raises RateLimitExceeded."""

    def test_rate_limiter_allows_first_request(self) -> None:
        """A fresh limiter with capacity must allow the first acquire()."""
        limiter = RateLimiter(requests_per_minute=60)
        # Should not raise
        limiter.acquire()

    def test_rate_limiter_raises_when_exhausted(self) -> None:
        """A limiter with 1 RPM and 0 max wait must raise on the second call."""
        limiter = RateLimiter(requests_per_minute=1, max_wait_seconds=0.0)
        limiter.acquire()  # Drains the bucket
        with pytest.raises(RateLimitExceeded):
            limiter.acquire()

    def test_rate_limiter_gates_gateway_requests(self) -> None:
        """Callers that respect the limiter must have their requests forwarded."""
        init_policy_engine(_build_config(allowed_hosts=["api.example.com"]))
        limiter = RateLimiter(requests_per_minute=100)
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            limiter.acquire()
            resp = client.get("https://api.example.com/v1/resource")
        assert resp.status_code == 200

    def test_exhausted_limiter_prevents_gateway_call(self) -> None:
        """When the limiter is exhausted the gateway is never reached."""
        init_policy_engine(_build_config(allowed_hosts=["api.example.com"]))
        limiter = RateLimiter(requests_per_minute=1, max_wait_seconds=0.0)
        client = PolicyHTTPClient()
        limiter.acquire()  # Use the only token
        with patch.object(httpx.Client, "get") as mock_get:
            with pytest.raises(RateLimitExceeded):
                limiter.acquire()
                client.get("https://api.example.com/v1/resource")
            mock_get.assert_not_called()

    def test_unlimited_rate_limiter_never_raises(self) -> None:
        """A limiter with RPM=0 (unlimited) must never raise."""
        limiter = RateLimiter(requests_per_minute=0, tokens_per_minute=0)
        for _ in range(20):
            limiter.acquire()

    def test_on_rate_limit_response_drains_bucket(self) -> None:
        """After on_rate_limit_response the request capacity drops to zero."""
        limiter = RateLimiter(requests_per_minute=60, max_wait_seconds=0.0)
        limiter.on_rate_limit_response(retry_after=0.0)
        with pytest.raises(RateLimitExceeded):
            limiter.acquire()

    def test_request_capacity_decrements_after_acquire(self) -> None:
        """Each acquire() must reduce request_capacity by 1."""
        limiter = RateLimiter(requests_per_minute=10, max_wait_seconds=5.0)
        before = limiter.request_capacity
        limiter.acquire()
        after = limiter.request_capacity
        assert after < before

    def test_token_capacity_decrements_with_token_acquire(self) -> None:
        """acquire(tokens=500) must reduce token_capacity by approximately 500.

        A small tolerance is necessary because the bucket refills continuously
        between the two ``token_capacity`` property reads.
        """
        limiter = RateLimiter(requests_per_minute=60, tokens_per_minute=10_000)
        before = limiter.token_capacity
        limiter.acquire(tokens=500)
        after = limiter.token_capacity
        # Allow up to 1 token of refill noise between the two reads.
        assert after <= before - 500 + 1


# ===========================================================================
# 11. Gateway error handling
# ===========================================================================


class TestGatewayErrorHandling:
    """httpx errors surface correctly through the gateway client."""

    def test_timeout_error_propagates(self) -> None:
        """httpx.TimeoutException raised by the underlying client must bubble up."""
        init_policy_engine(_build_config(allowed_hosts=["api.example.com"]))
        client = PolicyHTTPClient(timeout=1)
        with patch.object(
            httpx.Client,
            "get",
            side_effect=httpx.TimeoutException("timed out"),
        ), pytest.raises(httpx.TimeoutException):
            client.get("https://api.example.com/slow")

    def test_connect_error_propagates(self) -> None:
        """httpx.ConnectError must propagate after the policy check passes."""
        init_policy_engine(_build_config(allowed_hosts=["api.example.com"]))
        client = PolicyHTTPClient()
        with patch.object(
            httpx.Client,
            "get",
            side_effect=httpx.ConnectError("refused"),
        ), pytest.raises(httpx.ConnectError):
            client.get("https://api.example.com/unreachable")

    def test_http_4xx_returned_as_response(self) -> None:
        """A 404 HTTP response must be returned rather than raised."""
        init_policy_engine(_build_config(allowed_hosts=["api.example.com"]))
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(404, "not found")
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://api.example.com/missing")
        assert resp.status_code == 404

    def test_http_5xx_returned_as_response(self) -> None:
        """A 500 HTTP response must be returned rather than raised."""
        init_policy_engine(_build_config(allowed_hosts=["api.example.com"]))
        client = PolicyHTTPClient()
        mock_resp = _mock_http_response(500, "internal error")
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://api.example.com/broken")
        assert resp.status_code == 500

    def test_response_too_large_raises_value_error(self) -> None:
        """A response whose Content-Length exceeds the cap must raise ValueError."""
        init_policy_engine(_build_config(allowed_hosts=["api.example.com"]))
        client = PolicyHTTPClient(max_response_bytes=100)
        mock_resp = _mock_http_response(200)
        # Override headers to report a large size.
        mock_resp.headers = {"content-length": "9999999"}
        with patch.object(httpx.Client, "get", return_value=mock_resp), pytest.raises(ValueError, match="too large"):
            client.get("https://api.example.com/huge")

    def test_invalid_timeout_raises_on_construction(self) -> None:
        """A non-positive timeout must be rejected at construction time."""
        with pytest.raises(ValueError):
            PolicyHTTPClient(timeout=0)

    def test_unsupported_url_scheme_raises_value_error(self) -> None:
        """Non-http(s) schemes must raise ValueError before any policy check."""
        init_policy_engine(_build_config(default_deny=False))
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client.get("ftp://files.example.com/archive.tar.gz")

    async def test_async_timeout_propagates(self) -> None:
        """Async variant: httpx.TimeoutException must bubble up from aget."""
        init_policy_engine(_build_config(allowed_hosts=["api.example.com"]))
        client = PolicyHTTPClient()
        with patch.object(
            httpx.AsyncClient,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("async timeout"),
        ), pytest.raises(httpx.TimeoutException):
            await client.aget("https://api.example.com/slow")

    def test_policy_check_before_httpx_on_denied_host(self) -> None:
        """The policy guard must fire before httpx is touched — confirmed by
        verifying httpx is never called on a blocked host."""
        init_policy_engine(_build_config(default_deny=True))
        client = PolicyHTTPClient()
        call_log: list[str] = []

        def _httpx_get_called(*_: Any, **__: Any) -> None:
            call_log.append("httpx_called")
            return _mock_http_response()

        with patch.object(httpx.Client, "get", side_effect=_httpx_get_called), pytest.raises(PolicyViolationError):
            client.get("https://blocked.example.com/")
        assert call_log == []


# ===========================================================================
# 12. Policy audit events
# ===========================================================================


class TestPolicyAuditEvents:
    """Network violations must produce audit events on the event bus."""

    def test_denied_host_emits_deny_event(self) -> None:
        """A blocked host must produce exactly one 'deny' audit event."""
        policy = NetworkPolicy(default_deny=True)
        engine = NetworkPolicyEngine(policy)
        with pytest.raises(PolicyViolationError):
            engine.check_host("evil.example.com")
        deny_events = event_bus.get_events(event_type="network_check", result="deny")
        assert len(deny_events) == 1
        assert deny_events[0].detail["host"] == "evil.example.com"

    def test_allowed_host_emits_allow_event(self) -> None:
        """An allowed host must produce exactly one 'allow' audit event."""
        policy = NetworkPolicy(
            default_deny=True,
            allowed_hosts=["api.example.com"],
        )
        engine = NetworkPolicyEngine(policy)
        engine.check_host("api.example.com")
        allow_events = event_bus.get_events(event_type="network_check", result="allow")
        assert len(allow_events) == 1
        assert allow_events[0].detail["host"] == "api.example.com"

    def test_successful_request_emits_network_request_event(self) -> None:
        """A completed request must emit a 'network_request' event with
        method, url, and status_code."""
        init_policy_engine(_build_config(allowed_hosts=["api.example.com"]))
        client = PolicyHTTPClient(session_id="sess-audit", task_id="task-audit")
        mock_resp = _mock_http_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            client.get("https://api.example.com/ping")
        req_events = event_bus.get_events(event_type="network_request")
        assert len(req_events) == 1
        ev = req_events[0]
        assert ev.detail["method"] == "GET"
        assert ev.detail["url"] == "https://api.example.com/ping"
        assert ev.detail["status_code"] == 200
        assert ev.session_id == "sess-audit"
        assert ev.task_id == "task-audit"
        assert ev.result == "allow"
        assert ev.category == "network"

    def test_denied_request_event_category_is_network(self) -> None:
        """Deny events raised from network violations carry category='network'."""
        policy = NetworkPolicy(default_deny=True)
        engine = NetworkPolicyEngine(policy)
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_host("blocked.example.com")
        assert exc_info.value.category == "network"
        events = event_bus.get_events(event_type="network_check", result="deny")
        assert events[0].category == "network"

    def test_multiple_requests_each_emit_event(self) -> None:
        """Each successful request must emit its own audit event."""
        init_policy_engine(_build_config(allowed_hosts=["api.example.com"]))
        client = PolicyHTTPClient(session_id="multi")
        mock_resp = _mock_http_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            for _ in range(3):
                client.get("https://api.example.com/endpoint")
        req_events = event_bus.get_events(event_type="network_request", session_id="multi")
        assert len(req_events) == 3

    def test_rest_policy_violation_raises_without_request_event(self) -> None:
        """When a REST rule denies, no 'network_request' event (success) is
        emitted because httpx is never invoked."""
        init_policy_engine(
            _build_config(
                allowed_hosts=["api.github.com"],
                rest_policies=[
                    {
                        "host": "api.github.com",
                        "method": "DELETE",
                        "path": "/**",
                        "action": "deny",
                    }
                ],
            )
        )
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "delete"), pytest.raises(PolicyViolationError):
            client.delete("https://api.github.com/repos/foo")
        req_events = event_bus.get_events(event_type="network_request")
        assert len(req_events) == 0

    def test_session_and_task_ids_in_deny_event(self) -> None:
        """session_id and task_id must propagate into deny audit events."""
        policy = NetworkPolicy(
            default_deny=True,
            allowed_hosts=["allowed.example.com"],
        )
        engine = NetworkPolicyEngine(policy)
        with pytest.raises(PolicyViolationError):
            engine.check_host("denied.example.com", session_id="s42", task_id="t7")
        deny_event = event_bus.get_events(result="deny")[0]
        assert deny_event.session_id == "s42"
        assert deny_event.task_id == "t7"

    def test_event_bus_cleared_between_tests(self) -> None:
        """Sanity check that the autouse fixture leaves the bus clean."""
        # At the start of every test the bus is empty.
        events = event_bus.get_events()
        assert events == []
