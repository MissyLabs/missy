"""Tests for DNS rebinding protection in NetworkPolicyEngine.check_host (step 5).

When a hostname resolves via DNS to a private, loopback, or link-local address
the engine must:
  - Allow the request when that IP range is explicitly covered by allowed_cidrs.
  - Raise PolicyViolationError with "DNS rebinding" in the message otherwise.

Public IPs obtained via DNS are checked against allowed_cidrs normally; they
never trigger the rebinding path.
"""

from __future__ import annotations

import socket
from collections.abc import Generator
from unittest.mock import patch

import pytest

from missy.config.settings import NetworkPolicy
from missy.core.events import event_bus
from missy.core.exceptions import PolicyViolationError
from missy.policy.network import NetworkPolicyEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(**kwargs) -> NetworkPolicyEngine:
    """Build a NetworkPolicyEngine with sensible defaults plus keyword overrides."""
    defaults: dict = {
        "default_deny": True,
        "allowed_cidrs": [],
        "allowed_domains": [],
        "allowed_hosts": [],
    }
    defaults.update(kwargs)
    return NetworkPolicyEngine(NetworkPolicy(**defaults))


def _addrinfo_ipv4(ip: str) -> list:
    """Return a minimal getaddrinfo result for a single IPv4 address."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]


def _addrinfo_ipv6(ip: str) -> list:
    """Return a minimal getaddrinfo result for a single IPv6 address."""
    return [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", (ip, 0, 0, 0))]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_event_bus() -> Generator[None, None, None]:
    """Ensure each test starts with a clean event bus."""
    event_bus.clear()
    yield
    event_bus.clear()


# ---------------------------------------------------------------------------
# DNS rebinding — private IPv4 addresses denied
# ---------------------------------------------------------------------------


class TestDNSRebindingPrivateIPv4Denied:
    """Hostnames that resolve to RFC-1918 addresses are blocked unless the
    matching CIDR is explicitly listed in allowed_cidrs."""

    def test_dns_rebinding_to_private_ip_denied(self):
        """10.x resolves via DNS without a matching CIDR — must be denied."""
        engine = _make_engine()  # no allowed_cidrs
        mock_result = _addrinfo_ipv4("10.0.0.1")
        with (
            patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("evil.example.com")

        msg = str(exc_info.value)
        assert "DNS rebinding" in msg
        assert exc_info.value.category == "network"

    def test_dns_rebinding_to_private_ip_detail_contains_resolved_ip(self):
        """The violation detail should mention both the hostname and the IP."""
        engine = _make_engine()
        mock_result = _addrinfo_ipv4("192.168.1.100")
        with (
            patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("attacker.example.com")

        # Both the hostname and the resolved IP should appear in the exception
        detail = exc_info.value.detail
        assert "attacker.example.com" in detail
        assert "192.168.1.100" in detail

    def test_dns_rebinding_emits_deny_audit_event(self):
        """A DNS rebinding block must emit a deny audit event."""
        engine = _make_engine()
        mock_result = _addrinfo_ipv4("172.16.0.1")
        with (
            patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result),
            pytest.raises(PolicyViolationError),
        ):
            engine.check_host("internal.evil.com")

        deny_events = event_bus.get_events(result="deny")
        assert len(deny_events) == 1
        assert deny_events[0].detail["host"] == "internal.evil.com"


# ---------------------------------------------------------------------------
# DNS rebinding — loopback denied
# ---------------------------------------------------------------------------


class TestDNSRebindingLoopbackDenied:
    def test_dns_rebinding_to_loopback_denied(self):
        """127.0.0.1 via DNS must be blocked with DNS rebinding message."""
        engine = _make_engine()
        mock_result = _addrinfo_ipv4("127.0.0.1")
        with (
            patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("localhost.evil.com")

        assert "DNS rebinding" in str(exc_info.value)

    def test_dns_rebinding_to_loopback_range_denied(self):
        """Any address in 127.0.0.0/8 is loopback and must be blocked."""
        engine = _make_engine()
        mock_result = _addrinfo_ipv4("127.0.0.2")
        with (
            patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("loop.evil.com")

        assert "DNS rebinding" in str(exc_info.value)


# ---------------------------------------------------------------------------
# DNS rebinding — link-local denied
# ---------------------------------------------------------------------------


class TestDNSRebindingLinkLocalDenied:
    def test_dns_rebinding_to_link_local_denied(self):
        """169.254.169.254 (cloud metadata endpoint) via DNS must be blocked."""
        engine = _make_engine()
        mock_result = _addrinfo_ipv4("169.254.169.254")
        with (
            patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("metadata.evil.com")

        assert "DNS rebinding" in str(exc_info.value)
        assert exc_info.value.category == "network"

    def test_dns_rebinding_to_link_local_169_254_range_denied(self):
        """Any 169.254.x.x address (APIPA) is link-local and must be blocked."""
        engine = _make_engine()
        mock_result = _addrinfo_ipv4("169.254.1.1")
        with (
            patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("linklocal.evil.com")

        assert "DNS rebinding" in str(exc_info.value)


# ---------------------------------------------------------------------------
# DNS to private IP — allowed when CIDR matches
# ---------------------------------------------------------------------------


class TestDNSToPrivateIPAllowedWhenCIDRMatches:
    def test_dns_to_private_ip_allowed_when_cidr_matches(self):
        """10.0.0.1 via DNS is allowed when 10.0.0.0/8 is in allowed_cidrs."""
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8"])
        mock_result = _addrinfo_ipv4("10.0.0.1")
        with patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result):
            result = engine.check_host("internal.company.com")

        assert result is True

    def test_dns_to_loopback_allowed_when_cidr_matches(self):
        """127.0.0.1 via DNS is allowed when 127.0.0.0/8 is in allowed_cidrs."""
        engine = _make_engine(allowed_cidrs=["127.0.0.0/8"])
        mock_result = _addrinfo_ipv4("127.0.0.1")
        with patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result):
            result = engine.check_host("loopback-service.local")

        assert result is True

    def test_dns_to_link_local_allowed_when_cidr_matches(self):
        """169.254.169.254 is allowed when 169.254.0.0/16 is in allowed_cidrs."""
        engine = _make_engine(allowed_cidrs=["169.254.0.0/16"])
        mock_result = _addrinfo_ipv4("169.254.169.254")
        with patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result):
            result = engine.check_host("metadata.internal")

        assert result is True

    def test_dns_to_private_allowed_emits_allow_event(self):
        """An allowed private-IP result must emit an allow audit event."""
        engine = _make_engine(allowed_cidrs=["10.0.0.0/8"])
        mock_result = _addrinfo_ipv4("10.5.5.5")
        with patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result):
            engine.check_host("db.internal.corp")

        allow_events = event_bus.get_events(result="allow")
        assert len(allow_events) == 1
        assert allow_events[0].policy_rule == "cidr:10.0.0.0/8"

    def test_private_ip_not_matching_cidr_still_denied(self):
        """10.0.0.1 from DNS is denied even when 192.168.0.0/16 is allowed."""
        engine = _make_engine(allowed_cidrs=["192.168.0.0/16"])
        mock_result = _addrinfo_ipv4("10.0.0.1")
        with (
            patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("wrong-range.evil.com")

        assert "DNS rebinding" in str(exc_info.value)


# ---------------------------------------------------------------------------
# DNS to public IP — checked against allowed_cidrs normally
# ---------------------------------------------------------------------------


class TestDNSToPublicIP:
    def test_dns_to_public_ip_checked_against_cidrs(self):
        """8.8.8.8 via DNS is allowed when 8.8.0.0/16 is in allowed_cidrs."""
        engine = _make_engine(allowed_cidrs=["8.8.0.0/16"])
        mock_result = _addrinfo_ipv4("8.8.8.8")
        with patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result):
            result = engine.check_host("dns.google")

        assert result is True

    def test_dns_to_public_ip_not_in_cidrs_denied(self):
        """8.8.8.8 via DNS is denied when no matching CIDR is configured.

        The error must NOT mention DNS rebinding — this is a normal policy deny.
        """
        engine = _make_engine()  # no allowed_cidrs
        mock_result = _addrinfo_ipv4("8.8.8.8")
        with (
            patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("dns.google")

        msg = str(exc_info.value)
        assert "DNS rebinding" not in msg
        assert exc_info.value.category == "network"

    def test_dns_to_public_ip_denied_emits_deny_event(self):
        """A public-IP denial (non-rebinding) still emits one deny event."""
        engine = _make_engine()
        mock_result = _addrinfo_ipv4("8.8.8.8")
        with (
            patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result),
            pytest.raises(PolicyViolationError),
        ):
            engine.check_host("dns.google")

        deny_events = event_bus.get_events(result="deny")
        assert len(deny_events) == 1


# ---------------------------------------------------------------------------
# IPv6 rebinding protection
# ---------------------------------------------------------------------------


class TestDNSRebindingIPv6:
    def test_dns_rebinding_to_ipv6_loopback_denied(self):
        """::1 via DNS must be blocked with DNS rebinding message."""
        engine = _make_engine()
        mock_result = _addrinfo_ipv6("::1")
        with (
            patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("evil-ipv6.example.com")

        assert "DNS rebinding" in str(exc_info.value)
        assert exc_info.value.category == "network"

    def test_dns_rebinding_to_ipv6_link_local_denied(self):
        """fe80::1 via DNS must be blocked with DNS rebinding message."""
        engine = _make_engine()
        mock_result = _addrinfo_ipv6("fe80::1")
        with (
            patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("linklocal-ipv6.evil.com")

        assert "DNS rebinding" in str(exc_info.value)
        assert exc_info.value.category == "network"

    def test_dns_rebinding_to_ipv6_private_ula_denied(self):
        """fc00::/7 (Unique Local Addresses) via DNS must be blocked."""
        engine = _make_engine()
        mock_result = _addrinfo_ipv6("fd00::1")
        with (
            patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("ula.evil.com")

        assert "DNS rebinding" in str(exc_info.value)

    def test_dns_to_ipv6_loopback_allowed_when_cidr_matches(self):
        """::1 via DNS is allowed when ::1/128 is in allowed_cidrs."""
        engine = _make_engine(allowed_cidrs=["::1/128"])
        mock_result = _addrinfo_ipv6("::1")
        with patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result):
            result = engine.check_host("localhost-v6.internal")

        assert result is True


# ---------------------------------------------------------------------------
# PolicyViolationError is re-raised (not swallowed by OSError handler)
# ---------------------------------------------------------------------------


class TestPolicyViolationErrorPropagation:
    """Verify that PolicyViolationError raised inside the DNS block escapes
    rather than being caught by the OSError handler."""

    def test_policy_violation_propagates_not_caught_as_os_error(self):
        """The OSError handler must not suppress a PolicyViolationError."""
        engine = _make_engine()
        mock_result = _addrinfo_ipv4("10.0.0.1")

        # We expect exactly PolicyViolationError, not a generic deny from step 6.
        with (
            patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result),
            pytest.raises(PolicyViolationError) as exc_info,
        ):
            engine.check_host("sneaky.evil.com")

        # The exception must carry rebinding details, not the generic step-6 message
        assert "DNS rebinding" in str(exc_info.value)

    def test_only_one_deny_event_emitted_on_rebinding_block(self):
        """Exactly one audit event should fire when DNS rebinding is blocked."""
        engine = _make_engine()
        mock_result = _addrinfo_ipv4("10.0.0.1")
        with (
            patch("missy.policy.network.socket.getaddrinfo", return_value=mock_result),
            pytest.raises(PolicyViolationError),
        ):
            engine.check_host("once.evil.com")

        assert len(event_bus.get_events()) == 1
