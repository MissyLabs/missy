"""Tests for missy.policy.network.NetworkPolicyEngine."""

from __future__ import annotations

import socket
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import NetworkPolicy
from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import PolicyViolationError
from missy.policy.network import NetworkPolicyEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_event_bus() -> Generator[None, None, None]:
    """Ensure each test starts with a clean event bus."""
    event_bus.clear()
    yield
    event_bus.clear()


def make_engine(**kwargs) -> NetworkPolicyEngine:
    """Build a NetworkPolicyEngine from keyword overrides."""
    defaults = dict(
        default_deny=True,
        allowed_cidrs=[],
        allowed_domains=[],
        allowed_hosts=[],
    )
    defaults.update(kwargs)
    return NetworkPolicyEngine(NetworkPolicy(**defaults))


# ---------------------------------------------------------------------------
# default_deny=False (allow-all mode)
# ---------------------------------------------------------------------------


class TestDefaultAllow:
    def test_allows_any_hostname(self):
        engine = make_engine(default_deny=False)
        assert engine.check_host("evil.example.com") is True

    def test_allows_ip_address(self):
        engine = make_engine(default_deny=False)
        assert engine.check_host("192.168.1.1") is True

    def test_emits_allow_event_with_rule(self):
        engine = make_engine(default_deny=False)
        engine.check_host("example.com", session_id="s1", task_id="t1")
        events = event_bus.get_events(event_type="network_check")
        assert len(events) == 1
        assert events[0].result == "allow"
        assert events[0].policy_rule == "default_allow"
        assert events[0].detail["host"] == "example.com"

    def test_session_task_ids_in_event(self):
        engine = make_engine(default_deny=False)
        engine.check_host("x.com", session_id="sess-42", task_id="task-7")
        event = event_bus.get_events()[0]
        assert event.session_id == "sess-42"
        assert event.task_id == "task-7"


# ---------------------------------------------------------------------------
# IP address – CIDR checks
# ---------------------------------------------------------------------------


class TestCIDRChecks:
    def test_ipv4_in_cidr_allowed(self):
        engine = make_engine(allowed_cidrs=["10.0.0.0/8"])
        assert engine.check_host("10.1.2.3") is True

    def test_ipv4_not_in_cidr_denied(self):
        engine = make_engine(allowed_cidrs=["10.0.0.0/8"])
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_host("192.168.1.1")
        assert exc_info.value.category == "network"

    def test_ipv6_in_cidr_allowed(self):
        engine = make_engine(allowed_cidrs=["::1/128"])
        assert engine.check_host("::1") is True

    def test_ipv6_bracket_notation_stripped(self):
        """URL parsers emit IPv6 in brackets; engine must strip them."""
        engine = make_engine(allowed_cidrs=["::1/128"])
        assert engine.check_host("[::1]") is True

    def test_multiple_cidrs_second_matches(self):
        engine = make_engine(allowed_cidrs=["10.0.0.0/8", "172.16.0.0/12"])
        assert engine.check_host("172.20.0.1") is True

    def test_cidr_rule_name_in_event(self):
        engine = make_engine(allowed_cidrs=["10.0.0.0/8"])
        engine.check_host("10.0.0.1")
        event = event_bus.get_events(result="allow")[0]
        assert event.policy_rule == "cidr:10.0.0.0/8"

    def test_ip_not_in_cidr_emits_deny_event(self):
        engine = make_engine(allowed_cidrs=["10.0.0.0/8"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("1.2.3.4")
        events = event_bus.get_events(result="deny")
        assert len(events) == 1
        assert events[0].detail["host"] == "1.2.3.4"

    def test_invalid_cidr_in_policy_ignored(self):
        """A bad CIDR string should log a warning and be skipped."""
        engine = make_engine(allowed_cidrs=["not-a-cidr", "10.0.0.0/8"])
        assert engine.check_host("10.0.0.1") is True

    def test_mixed_ipv4_ipv6_cidr_no_crash(self):
        """IPv4 address against IPv6 network should not raise."""
        engine = make_engine(allowed_cidrs=["::1/128"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("10.0.0.1")


# ---------------------------------------------------------------------------
# Exact host matching
# ---------------------------------------------------------------------------


class TestExactHostMatching:
    def test_exact_host_match_allowed(self):
        engine = make_engine(allowed_hosts=["api.github.com"])
        assert engine.check_host("api.github.com") is True

    def test_case_insensitive_host_match(self):
        engine = make_engine(allowed_hosts=["API.GitHub.COM"])
        assert engine.check_host("api.github.com") is True

    def test_host_with_port_in_allowlist(self):
        """Port suffix in the configured entry should be ignored."""
        engine = make_engine(allowed_hosts=["api.github.com:443"])
        assert engine.check_host("api.github.com") is True

    def test_different_host_not_matched(self):
        engine = make_engine(allowed_hosts=["api.github.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("evil.example.com")

    def test_host_rule_name_in_event(self):
        engine = make_engine(allowed_hosts=["api.github.com"])
        engine.check_host("api.github.com")
        event = event_bus.get_events(result="allow")[0]
        assert event.policy_rule == "host:api.github.com"


# ---------------------------------------------------------------------------
# Domain / wildcard matching
# ---------------------------------------------------------------------------


class TestDomainMatching:
    def test_exact_domain_match(self):
        engine = make_engine(allowed_domains=["github.com"])
        assert engine.check_host("github.com") is True

    def test_wildcard_subdomain_match(self):
        engine = make_engine(allowed_domains=["*.github.com"])
        assert engine.check_host("api.github.com") is True

    def test_wildcard_matches_root_domain(self):
        """*.github.com should also permit github.com itself."""
        engine = make_engine(allowed_domains=["*.github.com"])
        assert engine.check_host("github.com") is True

    def test_wildcard_deep_subdomain(self):
        engine = make_engine(allowed_domains=["*.github.com"])
        assert engine.check_host("releases.api.github.com") is True

    def test_wildcard_does_not_match_different_root(self):
        engine = make_engine(allowed_domains=["*.github.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("notgithub.com")

    def test_domain_case_insensitive(self):
        engine = make_engine(allowed_domains=["*.GitHub.COM"])
        assert engine.check_host("api.github.com") is True

    def test_domain_rule_name_in_event(self):
        engine = make_engine(allowed_domains=["*.github.com"])
        engine.check_host("api.github.com")
        event = event_bus.get_events(result="allow")[0]
        assert event.policy_rule == "domain:*.github.com"

    def test_no_false_suffix_match(self):
        """'notgithub.com' must not match domain 'github.com'."""
        engine = make_engine(allowed_domains=["github.com"])
        with pytest.raises(PolicyViolationError):
            engine.check_host("notgithub.com")


# ---------------------------------------------------------------------------
# DNS resolution fallback
# ---------------------------------------------------------------------------


class TestDNSFallback:
    def test_hostname_resolved_to_allowed_ip(self):
        engine = make_engine(allowed_cidrs=["93.184.216.0/24"])
        # Mock getaddrinfo to return 93.184.216.34 (example.com's IP)
        mock_result = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 80))]
        with patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result):
            assert engine.check_host("example.com") is True

    def test_hostname_resolved_but_ip_not_in_cidr(self):
        engine = make_engine(allowed_cidrs=["10.0.0.0/8"])
        mock_result = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 80))]
        with patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result):
            with pytest.raises(PolicyViolationError):
                engine.check_host("example.com")

    def test_dns_failure_falls_through_to_deny(self):
        engine = make_engine(allowed_cidrs=["10.0.0.0/8"])
        with patch(
            "missy.policy.network.socket.getaddrinfo",
            side_effect=OSError("Name or service not known"),
        ):
            with pytest.raises(PolicyViolationError):
                engine.check_host("unresolvable.invalid")

    def test_multiple_dns_results_first_match_wins(self):
        engine = make_engine(allowed_cidrs=["10.0.0.0/8"])
        mock_result = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 80)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 80)),
        ]
        with patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result):
            assert engine.check_host("multi.example.com") is True


# ---------------------------------------------------------------------------
# Edge cases and input validation
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_host_raises_value_error(self):
        engine = make_engine()
        with pytest.raises(ValueError):
            engine.check_host("")

    def test_none_like_host_raises_value_error(self):
        engine = make_engine()
        with pytest.raises(ValueError):
            engine.check_host("")

    def test_uppercase_host_normalised(self):
        engine = make_engine(allowed_hosts=["api.github.com"])
        assert engine.check_host("API.GITHUB.COM") is True

    def test_deny_emits_single_event(self):
        engine = make_engine()
        with pytest.raises(PolicyViolationError):
            engine.check_host("bad.example.com")
        assert len(event_bus.get_events()) == 1

    def test_policy_violation_error_category(self):
        engine = make_engine()
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.check_host("bad.example.com")
        assert exc_info.value.category == "network"
        assert "bad.example.com" in exc_info.value.detail

    def test_check_host_priority_cidr_before_domain(self):
        """An IP address should be checked against CIDRs, not domain rules."""
        engine = make_engine(
            allowed_cidrs=["10.0.0.0/8"],
            allowed_domains=["10.0.0.1"],  # nonsensical but tests priority
        )
        # 10.0.0.1 is an IP – should use CIDR path, not domain path
        assert engine.check_host("10.0.0.1") is True

    def test_audit_event_has_utc_timestamp(self):
        engine = make_engine(allowed_hosts=["x.com"])
        engine.check_host("x.com")
        event = event_bus.get_events()[0]
        assert event.timestamp.tzinfo is not None

    def test_audit_event_category_is_network(self):
        engine = make_engine(default_deny=False)
        engine.check_host("anything.com")
        assert event_bus.get_events()[0].category == "network"

    def test_check_cidr_with_invalid_ip_string_returns_none(self):
        """_check_cidr must return None rather than raising for a non-IP string."""
        engine = make_engine(allowed_cidrs=["10.0.0.0/8"])
        result = engine._check_cidr("not-an-ip")
        assert result is None

    def test_check_cidr_mixed_family_no_type_error(self):
        """IPv4 address against an IPv6 network (and vice versa) must not
        propagate TypeError; the CIDR entry is silently skipped."""
        import ipaddress as _ip

        engine = make_engine(allowed_cidrs=["10.0.0.0/8"])
        # Monkeypatch the pre-parsed network to an IPv6 one so that the
        # mixed-family TypeError branch inside _check_cidr is exercised.
        engine._networks = [_ip.ip_network("::1/128")]
        # 10.0.0.1 is IPv4 vs IPv6 network -> TypeError caught, returns None.
        result = engine._check_cidr("10.0.0.1")
        assert result is None


# ---------------------------------------------------------------------------
# Per-category network allowlists
# ---------------------------------------------------------------------------


class TestPerCategoryAllowlists:
    """Tests for per-category host lists (provider, tool, discord)."""

    def test_provider_host_allowed_with_category(self):
        engine = NetworkPolicyEngine(
            NetworkPolicy(
                default_deny=True,
                allowed_hosts=[],
                provider_allowed_hosts=["api.anthropic.com"],
            )
        )
        # Allowed when category matches
        assert engine.check_host("api.anthropic.com", category="provider") is True
        # Denied without the matching category
        with pytest.raises(PolicyViolationError):
            engine.check_host("api.anthropic.com", category="")

    def test_tool_host_allowed_with_category(self):
        engine = NetworkPolicyEngine(
            NetworkPolicy(
                default_deny=True,
                allowed_hosts=[],
                tool_allowed_hosts=["api.weatherapi.com"],
            )
        )
        assert engine.check_host("api.weatherapi.com", category="tool") is True
        with pytest.raises(PolicyViolationError):
            engine.check_host("api.weatherapi.com", category="provider")

    def test_discord_host_allowed_with_category(self):
        engine = NetworkPolicyEngine(
            NetworkPolicy(
                default_deny=True,
                allowed_hosts=[],
                discord_allowed_hosts=["discord.com", "gateway.discord.gg"],
            )
        )
        assert engine.check_host("discord.com", category="discord") is True
        assert engine.check_host("gateway.discord.gg", category="discord") is True

    def test_global_host_still_works_with_category(self):
        engine = NetworkPolicyEngine(
            NetworkPolicy(
                default_deny=True,
                allowed_hosts=["api.anthropic.com"],
                provider_allowed_hosts=[],
            )
        )
        # Global list works regardless of category
        assert engine.check_host("api.anthropic.com", category="provider") is True
        assert engine.check_host("api.anthropic.com", category="tool") is True
        assert engine.check_host("api.anthropic.com", category="") is True

    def test_category_rule_prefix_in_audit(self):
        engine = NetworkPolicyEngine(
            NetworkPolicy(
                default_deny=True,
                tool_allowed_hosts=["api.example.com"],
            )
        )
        event_bus.clear()
        engine.check_host("api.example.com", category="tool")
        events = event_bus.get_events(event_type="network_check", result="allow")
        assert len(events) == 1
        assert events[0].policy_rule.startswith("tool_host:")

    def test_unknown_category_falls_through(self):
        engine = NetworkPolicyEngine(
            NetworkPolicy(
                default_deny=True,
                provider_allowed_hosts=["api.anthropic.com"],
            )
        )
        with pytest.raises(PolicyViolationError):
            engine.check_host("api.anthropic.com", category="unknown")
