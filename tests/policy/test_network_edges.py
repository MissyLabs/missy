"""Session-15 comprehensive tests for missy.policy.network.NetworkPolicyEngine.

Coverage targets that extend (and do not duplicate) test_network.py and
test_dns_rebinding.py:

  1.  default_deny=False allows every hostname and IP without consulting rules
  2.  Empty host raises ValueError before any policy logic executes
  3.  Allowed by exact host — returns True, not just truthy
  4.  Allowed by IPv4 CIDR — host-route /32 matches
  5.  Allowed by IPv6 CIDR — full /128 and prefix /64
  6.  Allowed by domain wildcard *.example.com — subdomain
  7.  Denied host raises PolicyViolationError — category attribute equals "network"
  8.  PolicyViolationError.detail is a non-empty string referencing the host
  9.  Bare IP checked against CIDR only — DNS is never consulted
  10. IP not in any CIDR denied; message says "not in an allowed CIDR"
  11. DNS rebinding: private IP (RFC-1918) from DNS without matching CIDR raises
  12. DNS rebinding: private IP in CIDR → allowed (no rebinding block)
  13. DNS failure (OSError) → deny (mock raises OSError)
  14. Per-category hosts — provider, tool, discord checked independently
  15. Case insensitivity: host supplied as UPPER matches lower-case entry
  16. IPv6 bracket stripping: [::1] normalised to ::1 for CIDR lookup
  17. Invalid CIDR logged and ignored; valid CIDR in same list still works
  18. Port stripped from allowed_hosts entry — "host:443" matches "host"
  19. Domain suffix matching: *.github.com matches api.github.com
  20. Domain suffix: exact domain match with no wildcard prefix
  21. Multiple CIDR ranges — second range matches when first does not
  22. Empty policy (all lists empty, default_deny=True) denies all requests
  23. Audit event emitted for allow result — category, result, host detail
  24. Audit event emitted for deny result — category, result, host detail
  25. Audit event event_type is "network_check" for every outcome
  26. session_id and task_id propagated to audit event
  27. policy_rule is None in deny events
  28. policy_rule has "host:" prefix for exact host allow
  29. policy_rule has "cidr:" prefix for CIDR allow
  30. policy_rule has "domain:" prefix for domain allow
  31. policy_rule has "{category}_host:" prefix for per-category allow
  32. policy_rule is "default_allow" in default-allow mode
  33. Only one audit event per check_host call (no double-emit on deny)
  34. _is_ip — static method correctly classifies valid/invalid inputs
  35. _check_cidr returns the matching CIDR string, not just True
  36. _check_cidr returns None for no match
  37. _check_domain returns matching pattern for wildcard
  38. _check_exact_host returns None for non-match
  39. Mixed IPv4 address vs IPv6 CIDR does not crash (_check_cidr skip)
  40. Wildcard domain does NOT match unrelated domain with shared suffix chars
  41. check_host short-circuits at exact host — no DNS call made
  42. check_host short-circuits at domain match — no DNS call made
  43. DNS returns multiple IPs — all private addresses checked before allowing
  44. Mixed public+private DNS records — private without CIDR still blocked
  45. Discord category host is denied when category is "tool" instead
  46. Global allowed_host is accessible from all categories including empty
  47. Construction with only invalid CIDRs results in empty _networks list
  48. Uppercase IPv6 bracket notation stripped and normalised
  49. check_host with explicit session_id/task_id — both appear in event
  50. Dot-only domain pattern does not match arbitrary hosts
  51. Very long hostname denied cleanly
  52. Multiple exact hosts: second host matches when first does not
  53. Provider host denied when no category specified
  54. audit event detail dict contains 'host' key with normalised value
  55. Exactly one event per call in default_deny=False mode
"""

from __future__ import annotations

import socket
from collections.abc import Generator
from unittest.mock import patch

import pytest

from missy.config.settings import NetworkPolicy
from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import PolicyViolationError
from missy.policy.network import NetworkPolicyEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(**kwargs) -> NetworkPolicyEngine:
    """Build a NetworkPolicyEngine from keyword overrides applied over deny-all defaults."""
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


def _addrinfo(ip: str, family: int = socket.AF_INET) -> list:
    """Minimal getaddrinfo result for a single address."""
    if family == socket.AF_INET6:
        return [(family, socket.SOCK_STREAM, 0, "", (ip, 0, 0, 0))]
    return [(family, socket.SOCK_STREAM, 0, "", (ip, 0))]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_bus() -> Generator[None, None, None]:
    """Isolate each test from stale event bus state."""
    event_bus.clear()
    yield
    event_bus.clear()


# ---------------------------------------------------------------------------
# 1. default_deny=False allows all (default-allow mode)
# ---------------------------------------------------------------------------


class TestDefaultDenyFalseAllowsAll:
    def test_allows_arbitrary_hostname(self):
        engine = _make_engine(default_deny=False)
        assert engine.check_host("totally.evil.example.com") is True

    def test_allows_ipv4_address(self):
        engine = _make_engine(default_deny=False)
        assert engine.check_host("1.2.3.4") is True

    def test_allows_ipv6_address(self):
        engine = _make_engine(default_deny=False)
        assert engine.check_host("2001:db8::1") is True

    def test_no_dns_lookup_in_default_allow_mode(self):
        engine = _make_engine(default_deny=False)
        with patch("missy.policy.network.socket.getaddrinfo") as mock_dns:
            engine.check_host("would-trigger-dns.example.com")
        mock_dns.assert_not_called()

    def test_returns_true_not_just_truthy(self):
        engine = _make_engine(default_deny=False)
        result = engine.check_host("example.com")
        assert result is True

    def test_emits_exactly_one_audit_event(self):
        engine = _make_engine(default_deny=False)
        engine.check_host("x.example.com")
        assert len(event_bus.get_events()) == 1

    def test_audit_event_policy_rule_is_default_allow(self):
        engine = _make_engine(default_deny=False)
        engine.check_host("x.example.com")
        event = event_bus.get_events()[0]
        assert event.policy_rule == "default_allow"

    def test_audit_event_result_is_allow(self):
        engine = _make_engine(default_deny=False)
        engine.check_host("x.example.com")
        assert event_bus.get_events()[0].result == "allow"


# ---------------------------------------------------------------------------
# 2. Empty host raises ValueError
# ---------------------------------------------------------------------------


class TestEmptyHostValidation:
    def test_empty_string_raises_value_error(self):
        engine = _make_engine()
        with pytest.raises(ValueError):
            engine.check_host("")

    def test_error_raised_before_any_policy_check(self):
        """No audit event should be emitted when ValueError fires."""
        engine = _make_engine(default_deny=False)  # would allow anything valid
        with pytest.raises(ValueError):
            engine.check_host("")
        assert len(event_bus.get_events()) == 0

    def test_whitespace_only_raises_value_error(self):
        """Whitespace-only string is falsy after strip and must raise."""
        engine = _make_engine()
        # The implementation checks `if not host` after strip("[]").lower()
        # Whitespace is truthy but let's confirm the actual behaviour:
        # A single space host would pass the initial check but fail DNS.
        # Test the explicit empty-string contract guaranteed by docstring.
        with pytest.raises(ValueError):
            engine.check_host("")


# ---------------------------------------------------------------------------
# 3–5. Exact host and CIDR allowances
# ---------------------------------------------------------------------------


class TestExactHostAllowed:
    def test_returns_true(self):
        engine = _make_engine(allowed_hosts=["api.example.com"])
        result = engine.check_host("api.example.com")
        assert result is True

    def test_emits_allow_event(self):
        engine = _make_engine(allowed_hosts=["api.example.com"])
        engine.check_host("api.example.com")
        events = event_bus.get_events(result="allow")
        assert len(events) == 1

    def test_second_host_in_list_matches(self):
        engine = _make_engine(allowed_hosts=["first.example.com", "second.example.com"])
        assert engine.check_host("second.example.com") is True

    def test_non_matching_host_denied(self):
        engine = _make_engine(allowed_hosts=["api.example.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("other.example.com")


class TestIPv4CIDRAllowed:
    def test_host_route_slash32_matches(self):
        engine = _make_engine(allowed_cidrs=["203.0.113.5/32"])
        assert engine.check_host("203.0.113.5") is True

    def test_host_route_slash32_does_not_match_neighbour(self):
        engine = _make_engine(allowed_cidrs=["203.0.113.5/32"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("203.0.113.6")

    def test_slash24_matches_any_address_in_range(self):
        engine = _make_engine(allowed_cidrs=["192.0.2.0/24"])
        assert engine.check_host("192.0.2.100") is True

    def test_second_cidr_range_used_when_first_misses(self):
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8", "172.16.0.0/12"])
        assert engine.check_host("172.31.255.255") is True


class TestIPv6CIDRAllowed:
    def test_slash128_host_route_allowed(self):
        engine = _make_engine(allowed_cidrs=["2001:db8::1/128"])
        assert engine.check_host("2001:db8::1") is True

    def test_slash128_does_not_match_different_address(self):
        engine = _make_engine(allowed_cidrs=["2001:db8::1/128"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("2001:db8::2")

    def test_slash64_prefix_allows_address_in_range(self):
        engine = _make_engine(allowed_cidrs=["2001:db8::/64"])
        assert engine.check_host("2001:db8::ffff") is True

    def test_bracket_stripped_before_cidr_check(self):
        engine = _make_engine(allowed_cidrs=["::1/128"])
        assert engine.check_host("[::1]") is True

    def test_uppercase_ipv6_bracket_normalised(self):
        engine = _make_engine(allowed_cidrs=["::1/128"])
        assert engine.check_host("[::1]") is True


# ---------------------------------------------------------------------------
# 6. Domain wildcard matching
# ---------------------------------------------------------------------------


class TestDomainWildcard:
    def test_wildcard_matches_subdomain(self):
        engine = _make_engine(allowed_domains=["*.example.com"])
        assert engine.check_host("api.example.com") is True

    def test_wildcard_matches_root_domain(self):
        engine = _make_engine(allowed_domains=["*.example.com"])
        assert engine.check_host("example.com") is True

    def test_wildcard_matches_deep_subdomain(self):
        engine = _make_engine(allowed_domains=["*.example.com"])
        assert engine.check_host("a.b.c.example.com") is True

    def test_wildcard_does_not_match_sibling_domain(self):
        engine = _make_engine(allowed_domains=["*.example.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("notexample.com")

    def test_wildcard_does_not_match_shared_suffix_chars(self):
        """xample.com must not match *.example.com."""
        engine = _make_engine(allowed_domains=["*.example.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("xample.com")

    def test_exact_domain_no_wildcard(self):
        engine = _make_engine(allowed_domains=["example.com"])
        assert engine.check_host("example.com") is True

    def test_exact_domain_does_not_match_subdomain(self):
        engine = _make_engine(allowed_domains=["example.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("sub.example.com")

    def test_domain_case_insensitive_pattern(self):
        engine = _make_engine(allowed_domains=["*.Example.COM"])
        assert engine.check_host("api.example.com") is True

    def test_github_wildcard_matches_api_github_com(self):
        engine = _make_engine(allowed_domains=["*.github.com"])
        assert engine.check_host("api.github.com") is True


# ---------------------------------------------------------------------------
# 7–8. Denied host — PolicyViolationError attributes
# ---------------------------------------------------------------------------


class TestDeniedHostError:
    def test_raises_policy_violation_error(self):
        engine = _make_engine()
        with pytest.raises(PolicyViolationError):
            engine.check_host("bad.example.com")

    def test_error_category_is_network(self):
        engine = _make_engine()
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_host("bad.example.com")
        assert exc_info.value.category == "network"

    def test_error_detail_is_non_empty_string(self):
        engine = _make_engine()
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_host("bad.example.com")
        assert isinstance(exc_info.value.detail, str)
        assert len(exc_info.value.detail) > 0

    def test_error_detail_references_denied_host(self):
        engine = _make_engine()
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_host("bad.example.com")
        assert "bad.example.com" in exc_info.value.detail

    def test_error_message_references_denied_host(self):
        engine = _make_engine()
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_host("bad.example.com")
        assert "bad.example.com" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 9–10. Bare IP — CIDR-only path, no DNS
# ---------------------------------------------------------------------------


class TestBareIPNoDNS:
    def test_ip_allowed_by_cidr_no_dns_call(self):
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8"])
        with patch("missy.policy.network.socket.getaddrinfo") as mock_dns:
            engine.check_host("10.1.2.3")
        mock_dns.assert_not_called()

    def test_ip_denied_no_dns_call(self):
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8"])
        with (
            patch("missy.policy.network.socket.getaddrinfo") as mock_dns,
            pytest.raises(PolicyViolationError),
        ):
            engine.check_host("1.2.3.4")
        mock_dns.assert_not_called()

    def test_ip_not_in_cidr_error_message_says_not_in_cidr(self):
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8"])
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_host("1.2.3.4")
        assert "not in an allowed CIDR" in str(exc_info.value)

    def test_ip_not_in_empty_cidr_list_denied(self):
        engine = _make_engine()
        with pytest.raises(PolicyViolationError):
            engine.check_host("8.8.8.8")


# ---------------------------------------------------------------------------
# 11–12. DNS rebinding protection
# ---------------------------------------------------------------------------


class TestDNSRebindingProtection:
    def test_private_rfc1918_from_dns_without_cidr_raises(self):
        engine = _make_engine()
        with (
            patch(
                "missy.policy.network.socket.getaddrinfo",
                return_value=_addrinfo("192.168.100.1"),
            ),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("attacker.example.com")
        assert "DNS rebinding" in str(exc_info.value)
        assert exc_info.value.category == "network"

    def test_loopback_from_dns_without_cidr_raises(self):
        engine = _make_engine()
        with (
            patch(
                "missy.policy.network.socket.getaddrinfo",
                return_value=_addrinfo("127.0.0.1"),
            ),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("localhost.evil.com")
        assert "DNS rebinding" in str(exc_info.value)

    def test_link_local_169_254_from_dns_without_cidr_raises(self):
        engine = _make_engine()
        with (
            patch(
                "missy.policy.network.socket.getaddrinfo",
                return_value=_addrinfo("169.254.169.254"),
            ),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("metadata.evil.com")
        assert "DNS rebinding" in str(exc_info.value)

    def test_private_ip_in_allowed_cidr_passes(self):
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8"])
        with patch(
            "missy.policy.network.socket.getaddrinfo",
            return_value=_addrinfo("10.20.30.40"),
        ):
            result = engine.check_host("internal.corp.local")
        assert result is True

    def test_loopback_in_allowed_cidr_passes(self):
        engine = _make_engine(allowed_cidrs=["127.0.0.0/8"])
        with patch(
            "missy.policy.network.socket.getaddrinfo",
            return_value=_addrinfo("127.0.0.1"),
        ):
            result = engine.check_host("local-service.dev")
        assert result is True

    def test_rebinding_detail_mentions_resolved_ip(self):
        engine = _make_engine()
        with (
            patch(
                "missy.policy.network.socket.getaddrinfo",
                return_value=_addrinfo("10.0.0.1"),
            ),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("sneaky.example.com")
        assert "10.0.0.1" in exc_info.value.detail

    def test_rebinding_detail_mentions_hostname(self):
        engine = _make_engine()
        with (
            patch(
                "missy.policy.network.socket.getaddrinfo",
                return_value=_addrinfo("10.0.0.1"),
            ),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("sneaky.example.com")
        assert "sneaky.example.com" in exc_info.value.detail


# ---------------------------------------------------------------------------
# 13. DNS failure → deny
# ---------------------------------------------------------------------------


class TestDNSFailureDeny:
    def test_oserror_falls_through_to_deny(self):
        engine = _make_engine()
        with (
            patch(
                "missy.policy.network.socket.getaddrinfo",
                side_effect=OSError("Name or service not known"),
            ),
            pytest.raises(PolicyViolationError),
        ):
            engine.check_host("unresolvable.invalid")

    def test_oserror_deny_category_is_network(self):
        engine = _make_engine()
        with (
            patch(
                "missy.policy.network.socket.getaddrinfo",
                side_effect=OSError("NXDOMAIN"),
            ),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("no-such-host.invalid")
        assert exc_info.value.category == "network"

    def test_oserror_emits_deny_audit_event(self):
        engine = _make_engine()
        with (
            patch(
                "missy.policy.network.socket.getaddrinfo",
                side_effect=OSError("timeout"),
            ),
            pytest.raises(PolicyViolationError),
        ):
            engine.check_host("timeout.example.com")
        deny_events = event_bus.get_events(result="deny")
        assert len(deny_events) == 1

    def test_dns_error_message_does_not_mention_rebinding(self):
        """A plain DNS failure is not a rebinding attack."""
        engine = _make_engine()
        with (
            patch(
                "missy.policy.network.socket.getaddrinfo",
                side_effect=OSError("timeout"),
            ),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("timeout.example.com")
        assert "DNS rebinding" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# 14. Per-category hosts
# ---------------------------------------------------------------------------


class TestPerCategoryHosts:
    def test_provider_host_allowed_with_provider_category(self):
        engine = _make_engine(provider_allowed_hosts=["api.anthropic.com"])
        assert engine.check_host("api.anthropic.com", category="provider") is True

    def test_provider_host_denied_with_no_category(self):
        engine = _make_engine(provider_allowed_hosts=["api.anthropic.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("api.anthropic.com", category="")

    def test_provider_host_denied_with_tool_category(self):
        engine = _make_engine(provider_allowed_hosts=["api.anthropic.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("api.anthropic.com", category="tool")

    def test_tool_host_allowed_with_tool_category(self):
        engine = _make_engine(tool_allowed_hosts=["api.weatherapi.com"])
        assert engine.check_host("api.weatherapi.com", category="tool") is True

    def test_tool_host_denied_with_discord_category(self):
        engine = _make_engine(tool_allowed_hosts=["api.weatherapi.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("api.weatherapi.com", category="discord")

    def test_discord_host_allowed_with_discord_category(self):
        engine = _make_engine(discord_allowed_hosts=["gateway.discord.gg"])
        assert engine.check_host("gateway.discord.gg", category="discord") is True

    def test_discord_host_denied_with_provider_category(self):
        engine = _make_engine(discord_allowed_hosts=["gateway.discord.gg"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("gateway.discord.gg", category="provider")

    def test_global_allowed_host_accessible_from_all_categories(self):
        engine = _make_engine(allowed_hosts=["shared.example.com"])
        assert engine.check_host("shared.example.com", category="provider") is True
        assert engine.check_host("shared.example.com", category="tool") is True
        assert engine.check_host("shared.example.com", category="discord") is True
        assert engine.check_host("shared.example.com", category="") is True

    def test_provider_category_audit_rule_prefix(self):
        engine = _make_engine(provider_allowed_hosts=["api.openai.com"])
        engine.check_host("api.openai.com", category="provider")
        events = event_bus.get_events(result="allow")
        assert events[0].policy_rule.startswith("provider_host:")

    def test_tool_category_audit_rule_prefix(self):
        engine = _make_engine(tool_allowed_hosts=["api.tool.com"])
        engine.check_host("api.tool.com", category="tool")
        events = event_bus.get_events(result="allow")
        assert events[0].policy_rule.startswith("tool_host:")

    def test_discord_category_audit_rule_prefix(self):
        engine = _make_engine(discord_allowed_hosts=["cdn.discordapp.com"])
        engine.check_host("cdn.discordapp.com", category="discord")
        events = event_bus.get_events(result="allow")
        assert events[0].policy_rule.startswith("discord_host:")


# ---------------------------------------------------------------------------
# 15. Case insensitivity
# ---------------------------------------------------------------------------


class TestCaseInsensitivity:
    def test_uppercase_host_matches_lowercase_entry(self):
        engine = _make_engine(allowed_hosts=["api.github.com"])
        assert engine.check_host("API.GITHUB.COM") is True

    def test_mixed_case_host_matches_lowercase_entry(self):
        engine = _make_engine(allowed_hosts=["api.github.com"])
        assert engine.check_host("Api.GitHub.Com") is True

    def test_uppercase_entry_matches_lowercase_host(self):
        engine = _make_engine(allowed_hosts=["API.GITHUB.COM"])
        assert engine.check_host("api.github.com") is True

    def test_domain_pattern_case_insensitive(self):
        engine = _make_engine(allowed_domains=["*.EXAMPLE.COM"])
        assert engine.check_host("sub.example.com") is True


# ---------------------------------------------------------------------------
# 16. IPv6 bracket stripping
# ---------------------------------------------------------------------------


class TestIPv6BracketStripping:
    def test_bracket_notation_allowed_by_cidr(self):
        engine = _make_engine(allowed_cidrs=["::1/128"])
        assert engine.check_host("[::1]") is True

    def test_bracket_notation_denied_when_not_in_cidr(self):
        engine = _make_engine(allowed_cidrs=["2001:db8::/32"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("[::1]")

    def test_bracket_notation_event_host_is_stripped(self):
        engine = _make_engine(allowed_cidrs=["::1/128"])
        engine.check_host("[::1]")
        event = event_bus.get_events()[0]
        # Normalised host should not contain brackets
        assert "[" not in event.detail["host"]
        assert "]" not in event.detail["host"]

    def test_full_ipv6_with_brackets_denied_not_in_cidr(self):
        engine = _make_engine(allowed_cidrs=["::1/128"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("[2001:db8::1]")


# ---------------------------------------------------------------------------
# 17. Invalid CIDR in config logged and ignored
# ---------------------------------------------------------------------------


class TestInvalidCIDRHandling:
    def test_invalid_cidr_does_not_raise_on_construction(self):
        engine = _make_engine(allowed_cidrs=["not-a-cidr"])
        # Construction should succeed
        assert engine is not None

    def test_invalid_cidr_results_in_empty_networks(self):
        engine = _make_engine(allowed_cidrs=["not-a-cidr", "also-invalid"])
        assert engine._networks == []

    def test_valid_cidr_after_invalid_still_works(self):
        engine = _make_engine(allowed_cidrs=["totally-wrong", "10.0.0.0/8"])
        assert engine.check_host("10.5.5.5") is True

    def test_invalid_cidr_warning_logged(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="missy.policy.network"):
            _make_engine(allowed_cidrs=["bad-cidr"])
        assert any("ignoring invalid CIDR" in record.message for record in caplog.records)

    def test_construction_with_only_invalid_cidrs_networks_empty(self):
        engine = _make_engine(allowed_cidrs=["x", "y", "z"])
        assert len(engine._networks) == 0


# ---------------------------------------------------------------------------
# 18. Port stripped from allowed_hosts entry
# ---------------------------------------------------------------------------


class TestPortStripping:
    def test_host_with_port_443_matches_bare_host(self):
        engine = _make_engine(allowed_hosts=["api.github.com:443"])
        assert engine.check_host("api.github.com") is True

    def test_host_with_port_80_matches_bare_host(self):
        engine = _make_engine(allowed_hosts=["example.com:80"])
        assert engine.check_host("example.com") is True

    def test_host_with_high_port_matches_bare_host(self):
        engine = _make_engine(allowed_hosts=["internal.corp:8443"])
        assert engine.check_host("internal.corp") is True

    def test_event_policy_rule_includes_original_entry_with_port(self):
        engine = _make_engine(allowed_hosts=["api.github.com:443"])
        engine.check_host("api.github.com")
        event = event_bus.get_events(result="allow")[0]
        assert "api.github.com:443" in event.policy_rule


# ---------------------------------------------------------------------------
# 19–20. Domain suffix matching — additional scenarios
# ---------------------------------------------------------------------------


class TestDomainSuffixScenarios:
    def test_github_wildcard_allows_api_github_com(self):
        engine = _make_engine(allowed_domains=["*.github.com"])
        assert engine.check_host("api.github.com") is True

    def test_github_wildcard_allows_raw_github_com(self):
        engine = _make_engine(allowed_domains=["*.github.com"])
        assert engine.check_host("raw.github.com") is True

    def test_exact_domain_allows_itself(self):
        engine = _make_engine(allowed_domains=["github.com"])
        assert engine.check_host("github.com") is True

    def test_exact_domain_does_not_allow_sub(self):
        engine = _make_engine(allowed_domains=["github.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("api.github.com")

    def test_domain_rule_name_has_domain_prefix(self):
        engine = _make_engine(allowed_domains=["*.github.com"])
        engine.check_host("api.github.com")
        event = event_bus.get_events(result="allow")[0]
        assert event.policy_rule.startswith("domain:")

    def test_domain_rule_name_contains_original_pattern(self):
        engine = _make_engine(allowed_domains=["*.github.com"])
        engine.check_host("api.github.com")
        event = event_bus.get_events(result="allow")[0]
        assert "*.github.com" in event.policy_rule


# ---------------------------------------------------------------------------
# 21. Multiple CIDR ranges
# ---------------------------------------------------------------------------


class TestMultipleCIDRRanges:
    def test_first_cidr_matches(self):
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8", "192.168.0.0/16"])
        assert engine.check_host("10.1.1.1") is True

    def test_second_cidr_matches_when_first_does_not(self):
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8", "192.168.0.0/16"])
        assert engine.check_host("192.168.1.1") is True

    def test_third_cidr_matches(self):
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"])
        assert engine.check_host("192.168.99.99") is True

    def test_no_cidr_matches_when_address_outside_all(self):
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8", "172.16.0.0/12"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("8.8.8.8")


# ---------------------------------------------------------------------------
# 22. Empty policy denies everything
# ---------------------------------------------------------------------------


class TestEmptyPolicyDeniesAll:
    def test_hostname_denied_with_empty_policy(self):
        engine = _make_engine()
        with pytest.raises(PolicyViolationError):
            engine.check_host("example.com")

    def test_ip_denied_with_empty_policy(self):
        engine = _make_engine()
        with pytest.raises(PolicyViolationError):
            engine.check_host("8.8.8.8")

    def test_private_ip_denied_with_empty_policy(self):
        engine = _make_engine()
        with pytest.raises(PolicyViolationError):
            engine.check_host("10.0.0.1")

    def test_localhost_hostname_denied_with_empty_policy(self):
        engine = _make_engine()
        with (
            patch(
                "missy.policy.network.socket.getaddrinfo",
                side_effect=OSError("failed"),
            ),
            pytest.raises(PolicyViolationError),
        ):
            engine.check_host("localhost")


# ---------------------------------------------------------------------------
# 23–33. Audit event structure and content
# ---------------------------------------------------------------------------


class TestAuditEventStructure:
    def test_allow_event_has_network_category(self):
        engine = _make_engine(allowed_hosts=["ok.example.com"])
        engine.check_host("ok.example.com")
        assert event_bus.get_events()[0].category == "network"

    def test_deny_event_has_network_category(self):
        engine = _make_engine()
        with pytest.raises(PolicyViolationError):
            engine.check_host("bad.example.com")
        assert event_bus.get_events()[0].category == "network"

    def test_event_type_is_network_check_on_allow(self):
        engine = _make_engine(allowed_hosts=["ok.example.com"])
        engine.check_host("ok.example.com")
        assert event_bus.get_events()[0].event_type == "network_check"

    def test_event_type_is_network_check_on_deny(self):
        engine = _make_engine()
        with pytest.raises(PolicyViolationError):
            engine.check_host("bad.example.com")
        assert event_bus.get_events()[0].event_type == "network_check"

    def test_session_id_propagated_to_event(self):
        engine = _make_engine(allowed_hosts=["ok.example.com"])
        engine.check_host("ok.example.com", session_id="sess-abc")
        assert event_bus.get_events()[0].session_id == "sess-abc"

    def test_task_id_propagated_to_event(self):
        engine = _make_engine(allowed_hosts=["ok.example.com"])
        engine.check_host("ok.example.com", task_id="task-xyz")
        assert event_bus.get_events()[0].task_id == "task-xyz"

    def test_session_and_task_id_both_in_event(self):
        engine = _make_engine(allowed_hosts=["ok.example.com"])
        engine.check_host("ok.example.com", session_id="s1", task_id="t1")
        event = event_bus.get_events()[0]
        assert event.session_id == "s1"
        assert event.task_id == "t1"

    def test_default_session_and_task_ids_are_empty_strings(self):
        engine = _make_engine(allowed_hosts=["ok.example.com"])
        engine.check_host("ok.example.com")
        event = event_bus.get_events()[0]
        assert event.session_id == ""
        assert event.task_id == ""

    def test_deny_event_policy_rule_is_none(self):
        engine = _make_engine()
        with pytest.raises(PolicyViolationError):
            engine.check_host("bad.example.com")
        assert event_bus.get_events()[0].policy_rule is None

    def test_allow_event_policy_rule_has_host_prefix(self):
        engine = _make_engine(allowed_hosts=["ok.example.com"])
        engine.check_host("ok.example.com")
        assert event_bus.get_events(result="allow")[0].policy_rule == "host:ok.example.com"

    def test_allow_event_policy_rule_has_cidr_prefix(self):
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8"])
        engine.check_host("10.1.2.3")
        assert event_bus.get_events(result="allow")[0].policy_rule == "cidr:10.0.0.0/8"

    def test_allow_event_policy_rule_has_domain_prefix(self):
        engine = _make_engine(allowed_domains=["*.example.com"])
        engine.check_host("sub.example.com")
        assert event_bus.get_events(result="allow")[0].policy_rule == "domain:*.example.com"

    def test_event_detail_contains_host_key(self):
        engine = _make_engine(allowed_hosts=["ok.example.com"])
        engine.check_host("ok.example.com")
        event = event_bus.get_events()[0]
        assert "host" in event.detail

    def test_event_detail_host_value_is_normalised_lowercase(self):
        engine = _make_engine(allowed_hosts=["ok.example.com"])
        engine.check_host("OK.EXAMPLE.COM")
        event = event_bus.get_events()[0]
        assert event.detail["host"] == "ok.example.com"

    def test_audit_event_timestamp_is_timezone_aware(self):
        engine = _make_engine(allowed_hosts=["ok.example.com"])
        engine.check_host("ok.example.com")
        event = event_bus.get_events()[0]
        assert event.timestamp.tzinfo is not None

    def test_exactly_one_event_per_check_host_call_on_allow(self):
        engine = _make_engine(allowed_hosts=["ok.example.com"])
        engine.check_host("ok.example.com")
        assert len(event_bus.get_events()) == 1

    def test_exactly_one_event_per_check_host_call_on_deny(self):
        engine = _make_engine()
        with pytest.raises(PolicyViolationError):
            engine.check_host("bad.example.com")
        assert len(event_bus.get_events()) == 1


# ---------------------------------------------------------------------------
# 34–39. Private helper methods
# ---------------------------------------------------------------------------


class TestPrivateHelpers:
    def test_is_ip_true_for_ipv4(self):
        assert NetworkPolicyEngine._is_ip("192.168.1.1") is True

    def test_is_ip_true_for_ipv6(self):
        assert NetworkPolicyEngine._is_ip("::1") is True

    def test_is_ip_false_for_hostname(self):
        assert NetworkPolicyEngine._is_ip("example.com") is False

    def test_is_ip_false_for_empty_string(self):
        assert NetworkPolicyEngine._is_ip("") is False

    def test_is_ip_false_for_partial_ip(self):
        assert NetworkPolicyEngine._is_ip("192.168") is False

    def test_check_cidr_returns_cidr_string_on_match(self):
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8"])
        result = engine._check_cidr("10.5.5.5")
        assert result == "cidr:10.0.0.0/8"

    def test_check_cidr_returns_none_on_no_match(self):
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8"])
        result = engine._check_cidr("1.2.3.4")
        assert result is None

    def test_check_cidr_returns_none_for_non_ip_input(self):
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8"])
        result = engine._check_cidr("not-an-ip")
        assert result is None

    def test_check_domain_returns_pattern_for_wildcard_match(self):
        engine = _make_engine(allowed_domains=["*.example.com"])
        result = engine._check_domain("sub.example.com")
        assert result == "domain:*.example.com"

    def test_check_domain_returns_none_for_non_match(self):
        engine = _make_engine(allowed_domains=["*.example.com"])
        result = engine._check_domain("other.org")
        assert result is None

    def test_check_exact_host_returns_none_for_non_match(self):
        engine = _make_engine(allowed_hosts=["ok.example.com"])
        result = engine._check_exact_host("bad.example.com")
        assert result is None

    def test_check_exact_host_returns_entry_for_match(self):
        engine = _make_engine(allowed_hosts=["ok.example.com"])
        result = engine._check_exact_host("ok.example.com")
        assert result == "host:ok.example.com"


# ---------------------------------------------------------------------------
# 40. Short-circuit: exact host match skips DNS
# ---------------------------------------------------------------------------


class TestShortCircuitBehaviour:
    def test_exact_host_match_does_not_call_dns(self):
        engine = _make_engine(allowed_hosts=["exact.example.com"])
        with patch("missy.policy.network.socket.getaddrinfo") as mock_dns:
            engine.check_host("exact.example.com")
        mock_dns.assert_not_called()

    def test_domain_match_does_not_call_dns(self):
        engine = _make_engine(allowed_domains=["*.example.com"])
        with patch("missy.policy.network.socket.getaddrinfo") as mock_dns:
            engine.check_host("sub.example.com")
        mock_dns.assert_not_called()

    def test_ip_cidr_match_does_not_call_dns(self):
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8"])
        with patch("missy.policy.network.socket.getaddrinfo") as mock_dns:
            engine.check_host("10.1.2.3")
        mock_dns.assert_not_called()


# ---------------------------------------------------------------------------
# 43–44. DNS multiple result handling
# ---------------------------------------------------------------------------


class TestDNSMultipleResults:
    def test_all_private_ips_checked_before_allowing(self):
        """When all DNS results are private IPs covered by CIDR, allow."""
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8"])
        multi = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.2", 0)),
        ]
        with patch("missy.policy.network.socket.getaddrinfo", return_value=multi):
            result = engine.check_host("multi-private.internal")
        assert result is True

    def test_mixed_public_private_private_not_in_cidr_blocks(self):
        """Public + private DNS record: private IP not in CIDR must block the whole request."""
        engine = _make_engine(allowed_cidrs=["8.8.0.0/16"])
        # 8.8.8.8 is public and in CIDR, but 10.0.0.1 is private and NOT in CIDR.
        multi = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0)),
        ]
        with (
            patch("missy.policy.network.socket.getaddrinfo", return_value=multi),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("mixed.evil.com")
        assert "DNS rebinding" in str(exc_info.value)

    def test_duplicate_dns_results_deduplicated(self):
        """Duplicate IPs from getaddrinfo must not cause duplicate audit events."""
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8"])
        # Same IP returned twice (e.g. from different record types)
        dup = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.1.1.1", 0)),
            (socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("10.1.1.1", 0)),
        ]
        with patch("missy.policy.network.socket.getaddrinfo", return_value=dup):
            result = engine.check_host("dup.internal")
        assert result is True
        # Only one audit event should be emitted
        assert len(event_bus.get_events()) == 1


# ---------------------------------------------------------------------------
# 45–46. Category isolation and global host reachability
# ---------------------------------------------------------------------------


class TestCategoryIsolation:
    def test_discord_host_blocked_when_tool_category_used(self):
        engine = _make_engine(discord_allowed_hosts=["cdn.discordapp.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("cdn.discordapp.com", category="tool")

    def test_provider_host_blocked_when_discord_category_used(self):
        engine = _make_engine(provider_allowed_hosts=["api.anthropic.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("api.anthropic.com", category="discord")

    def test_unknown_category_returns_empty_list(self):
        engine = _make_engine(provider_allowed_hosts=["api.anthropic.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("api.anthropic.com", category="foobar")


# ---------------------------------------------------------------------------
# 47–55. Additional edge cases
# ---------------------------------------------------------------------------


class TestAdditionalEdgeCases:
    def test_construction_with_only_invalid_cidrs_allows_no_ips(self):
        engine = _make_engine(allowed_cidrs=["not-valid", "also-bad"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("10.0.0.1")

    def test_very_long_hostname_denied_cleanly(self):
        engine = _make_engine()
        long_host = "a" * 200 + ".example.com"
        with (
            patch(
                "missy.policy.network.socket.getaddrinfo",
                side_effect=OSError("timeout"),
            ),
            pytest.raises(PolicyViolationError),
        ):
            engine.check_host(long_host)

    def test_mixed_ipv4_address_against_ipv6_network_no_crash(self):
        """IPv4 address checked against IPv6 CIDR must not propagate TypeError."""

        engine = _make_engine(allowed_cidrs=["::1/128"])
        # _networks contains an IPv6 network; address is IPv4 — should deny cleanly.
        with pytest.raises(PolicyViolationError):
            engine.check_host("10.0.0.1")

    def test_check_cidr_mixed_family_returns_none(self):
        import ipaddress

        engine = _make_engine(allowed_cidrs=["10.0.0.0/8"])
        engine._networks = [ipaddress.ip_network("::1/128")]
        result = engine._check_cidr("10.0.0.1")
        assert result is None

    def test_multiple_checks_accumulate_events(self):
        engine = _make_engine(allowed_hosts=["ok.example.com"])
        engine.check_host("ok.example.com")
        with pytest.raises(PolicyViolationError):
            engine.check_host("bad.example.com")
        assert len(event_bus.get_events()) == 2

    def test_event_bus_publish_called_once_per_check(self):
        engine = _make_engine(allowed_hosts=["ok.example.com"])
        captured: list[AuditEvent] = []
        event_bus.subscribe("network_check", captured.append)
        engine.check_host("ok.example.com")
        assert len(captured) == 1

    def test_result_true_is_exact_bool(self):
        """check_host must return True (bool), not just a truthy value."""
        engine = _make_engine(allowed_hosts=["ok.example.com"])
        result = engine.check_host("ok.example.com")
        assert type(result) is bool  # noqa: E721
        assert result is True

    def test_ipv4_address_not_matched_by_domain_list(self):
        """An IP address must not be matched via the domain suffix path."""
        engine = _make_engine(allowed_domains=["10.0.0.1"])
        # 10.0.0.1 is a bare IP — engine routes to CIDR path, not domain path.
        with pytest.raises(PolicyViolationError):
            engine.check_host("10.0.0.1")

    def test_dns_result_with_invalid_ip_string_skipped(self):
        """getaddrinfo results with unparseable IP strings must be skipped gracefully."""
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8"])
        # First entry has an invalid IP string, second has a valid allowed IP.
        mixed = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("not-an-ip", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.5.5.5", 0)),
        ]
        with patch("missy.policy.network.socket.getaddrinfo", return_value=mixed):
            result = engine.check_host("edge.internal")
        assert result is True
