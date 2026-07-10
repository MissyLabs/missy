"""Network policy enforcement for the Missy framework.

Outbound network access is evaluated against a :class:`NetworkPolicy` instance.
Every check emits an :class:`~missy.core.events.AuditEvent` through the
module-level event bus regardless of the outcome, giving consumers a complete
audit trail.

Example::

    from missy.config.settings import NetworkPolicy
    from missy.policy.network import NetworkPolicyEngine

    policy = NetworkPolicy(
        default_deny=True,
        allowed_domains=["*.github.com"],
        allowed_cidrs=["10.0.0.0/8"],
    )
    engine = NetworkPolicyEngine(policy)
    engine.check_host("api.github.com")   # -> True
    engine.check_host("evil.example.com") # -> raises PolicyViolationError
"""

from __future__ import annotations

import ipaddress
import logging
import socket

from missy.config.settings import NetworkPolicy
from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import PolicyViolationError

logger = logging.getLogger(__name__)


class NetworkPolicyEngine:
    """Evaluates outbound network requests against a :class:`NetworkPolicy`.

    The engine is intentionally stateless between calls: it holds only the
    policy configuration supplied at construction time.

    Args:
        policy: The network policy to enforce.
    """

    def __init__(self, policy: NetworkPolicy) -> None:
        self._policy = policy
        # Pre-parse CIDR networks once so repeated checks stay O(n) not O(n*parse).
        self._networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        for cidr in policy.allowed_cidrs:
            try:
                self._networks.append(ipaddress.ip_network(cidr, strict=False))
            except ValueError:
                logger.warning("NetworkPolicyEngine: ignoring invalid CIDR %r", cidr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_host(
        self,
        host: str,
        session_id: str = "",
        task_id: str = "",
        category: str = "",
    ) -> bool:
        """Return ``True`` if *host* is reachable under the current policy.

        The check short-circuits as soon as a match is found, following this
        order:

        1. If ``policy.default_deny`` is ``False``, allow everything.
        2. If *host* is a bare IP address, check CIDR allow-lists.
        3. Check for an exact match in ``policy.allowed_hosts`` (and any
           per-category hosts when *category* is set).
        4. Check wildcard / suffix match in ``policy.allowed_domains``.
        5. Attempt a DNS resolution of *host* and re-check the resulting IPs
           against the CIDR allow-lists.
        6. Deny.

        Args:
            host: Hostname or IP address (without port or scheme).
            session_id: Optional identifier of the calling session.
            task_id: Optional identifier of the calling task.
            category: Request category (``"provider"``, ``"tool"``,
                ``"discord"``).  When set, the corresponding per-category
                host list is merged into the check.

        Returns:
            ``True`` when the host is explicitly allowed.

        Raises:
            PolicyViolationError: When the host is denied by policy.
            ValueError: When *host* is ``None`` or an empty string.
        """
        if not host:
            raise ValueError("host must be a non-empty string")

        # Normalise: strip surrounding brackets from IPv6 literals such as
        # [::1] that arrive from URL parsers.
        host = host.strip("[]").lower()

        # Step 1 – default-allow mode bypasses all remaining checks.
        if not self._policy.default_deny:
            self._emit_event(host, "allow", "default_allow", session_id, task_id)
            return True

        # Step 2 – bare IP: check CIDR lists only (no DNS).
        if self._is_ip(host):
            rule = self._check_cidr(host)
            if rule:
                self._emit_event(host, "allow", rule, session_id, task_id)
                return True
            # IP not in any CIDR – deny immediately without DNS.
            self._emit_event(host, "deny", None, session_id, task_id)
            raise PolicyViolationError(
                f"Network access denied: {host!r} is not in an allowed CIDR block.",
                category="network",
                detail=f"IP address {host!r} is not covered by any allowed_cidrs entry.",
            )

        # Step 3 – exact hostname match (global + per-category).
        #
        # SR-1.9a: a name match alone is not sufficient — still verify the
        # resolved IP isn't private/rebound (see _resolve_and_check_rebinding).
        # Previously this step allowed immediately with zero IP check, so an
        # explicitly allowlisted hostname (e.g. "build.corp.example.com")
        # that resolves to internal infrastructure (10.0.0.5) connected with
        # no verification at all — the opposite of every other host, which
        # gets this same check via step 5 below. Trusting a *name* is not the
        # same as trusting whatever IP it happens to resolve to right now.
        rule = self._check_exact_host(host, category=category)
        if rule:
            self._resolve_and_check_rebinding(host, session_id, task_id)
            self._emit_event(host, "allow", rule, session_id, task_id)
            return True

        # Step 4 – wildcard / suffix domain match. Same SR-1.9a rebinding
        # check as step 3.
        rule = self._check_domain(host)
        if rule:
            self._resolve_and_check_rebinding(host, session_id, task_id)
            self._emit_event(host, "allow", rule, session_id, task_id)
            return True

        # Step 5 – DNS resolution + CIDR re-check for hostnames not matched
        # by name at all.
        try:
            resolved = self._resolve_and_check_rebinding(host, session_id, task_id)
            if resolved:
                for ip_str, _addr in resolved:
                    rule = self._check_cidr(ip_str)
                    if rule:
                        self._emit_event(host, "allow", rule, session_id, task_id)
                        return True
        except PolicyViolationError:
            raise

        # Step 6 – deny.
        self._emit_event(host, "deny", None, session_id, task_id)
        raise PolicyViolationError(
            f"Network access denied: {host!r} is not permitted by policy.",
            category="network",
            detail=(
                f"Host {host!r} did not match any allowed_hosts, allowed_domains, "
                "or allowed_cidrs entry."
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_and_check_rebinding(
        self,
        host: str,
        session_id: str,
        task_id: str,
    ) -> list[tuple[str, ipaddress.IPv4Address | ipaddress.IPv6Address]] | None:
        """Resolve *host* and deny if any resolved IP is an unallowed private address.

        Applied uniformly to every hostname match — exact, domain, and the
        no-name-match DNS fallback — so that an explicitly allowlisted
        hostname gets the same DNS-rebinding protection as any other host
        (SR-1.9a). This prevents an attacker (or a hostname whose DNS record
        later changes) from pointing an allowlisted name at
        ``169.254.169.254`` (cloud metadata) or ``10.0.0.1`` (internal
        infrastructure) and bypassing the name-based checks entirely.

        We check ALL resolved addresses before allowing. If ANY address is
        private/loopback/link-local and not covered by ``allowed_cidrs``,
        the entire request is denied — this prevents mixed-record attacks
        where a hostname resolves to both a public and a private IP.

        Args:
            host: Normalised (lowercase, no brackets) hostname.
            session_id: Calling session identifier, for the audit event.
            task_id: Calling task identifier, for the audit event.

        Returns:
            De-duplicated ``(ip_str, addr)`` pairs for every resolved
            address, or ``None`` if DNS resolution failed (nothing to rebind
            if the name doesn't resolve — the caller's own name-based match,
            if any, still stands).

        Raises:
            PolicyViolationError: When a resolved IP is private/reserved and
                not covered by ``allowed_cidrs``.
        """
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            logger.debug("NetworkPolicyEngine: DNS resolution failed for %r", host)
            return None

        seen_ips: set[str] = set()
        resolved: list[tuple[str, ipaddress.IPv4Address | ipaddress.IPv6Address]] = []
        for info in infos:
            ip_str = info[4][0]
            if ip_str in seen_ips:
                continue
            seen_ips.add(ip_str)
            try:
                addr = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            resolved.append((ip_str, addr))

        for ip_str, addr in resolved:
            if addr.is_private or addr.is_loopback or addr.is_link_local:
                rule = self._check_cidr(ip_str)
                if not rule:
                    logger.warning(
                        "NetworkPolicyEngine: DNS rebinding blocked — %r resolved to "
                        "private address %s which is not in allowed_cidrs",
                        host,
                        ip_str,
                    )
                    self._emit_event(host, "deny", None, session_id, task_id)
                    raise PolicyViolationError(
                        f"Network access denied: {host!r} resolved to private "
                        f"address {ip_str} (possible DNS rebinding attack).",
                        category="network",
                        detail=(
                            f"Hostname {host!r} resolved to {ip_str} which is a "
                            "private/reserved address not explicitly allowed by policy."
                        ),
                    )

        return resolved

    @staticmethod
    def _is_ip(host: str) -> bool:
        """Return ``True`` when *host* is a valid IPv4 or IPv6 address."""
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            return False

    def _check_cidr(self, ip_str: str) -> str | None:
        """Return the first matching CIDR rule string, or ``None``.

        Args:
            ip_str: A string representation of an IPv4 or IPv6 address.

        Returns:
            The CIDR string that contains *ip_str*, or ``None`` if no match.
        """
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            return None

        for network, cidr_str in zip(self._networks, self._policy.allowed_cidrs, strict=False):
            try:
                if addr in network:
                    return f"cidr:{cidr_str}"
            except TypeError:
                # Mixed IPv4/IPv6 comparison – skip.
                continue
        return None

    def _check_exact_host(self, host: str, category: str = "") -> str | None:
        """Return the matching ``allowed_hosts`` entry, or ``None``.

        The comparison is case-insensitive and strips any port suffix from the
        configured entry so that ``"api.github.com:443"`` still matches the
        plain hostname ``"api.github.com"``.

        When *category* is set, the per-category host list is also checked.

        Args:
            host: Normalised (lowercase, no brackets) hostname.
            category: Request category (``"provider"``, ``"tool"``,
                ``"discord"``).

        Returns:
            The matching entry string, or ``None``.
        """
        # Build combined host list: global + per-category
        hosts = list(self._policy.allowed_hosts)
        cat_hosts = self._category_hosts(category)
        for entry in hosts:
            configured_host = entry.lower().rsplit(":", 1)[0].strip("[]")
            if host == configured_host:
                return f"host:{entry}"
        for entry in cat_hosts:
            configured_host = entry.lower().rsplit(":", 1)[0].strip("[]")
            if host == configured_host:
                return f"{category}_host:{entry}"
        return None

    def _category_hosts(self, category: str) -> list[str]:
        """Return the per-category host list for *category*.

        Args:
            category: One of ``"provider"``, ``"tool"``, ``"discord"``, or
                empty string.

        Returns:
            The relevant per-category host list, or an empty list.
        """
        if category == "provider":
            return self._policy.provider_allowed_hosts
        if category == "tool":
            return self._policy.tool_allowed_hosts
        if category == "discord":
            return self._policy.discord_allowed_hosts
        return []

    def _check_domain(self, host: str) -> str | None:
        """Return the matching ``allowed_domains`` entry, or ``None``.

        Pattern semantics:

        * ``"github.com"`` – exact match only.
        * ``"*.github.com"`` – matches ``"api.github.com"``, ``"github.com"``,
          and any subdomain.

        Args:
            host: Normalised (lowercase, no brackets) hostname.

        Returns:
            The matching domain pattern string, or ``None``.
        """
        for pattern in self._policy.allowed_domains:
            pattern_lower = pattern.lower()

            if pattern_lower.startswith("*."):
                # Wildcard pattern: *.github.com
                suffix = pattern_lower[2:]  # "github.com"
                if host == suffix or host.endswith("." + suffix):
                    return f"domain:{pattern}"
            else:
                # Exact domain match.
                if host == pattern_lower:
                    return f"domain:{pattern}"

        return None

    def _emit_event(
        self,
        host: str,
        result: str,
        rule: str | None,
        session_id: str,
        task_id: str,
    ) -> None:
        """Publish a ``network_check`` audit event.

        Args:
            host: The hostname that was checked.
            result: ``"allow"`` or ``"deny"``.
            rule: The policy rule that produced the result, or ``None``.
            session_id: Calling session identifier.
            task_id: Calling task identifier.
        """
        event = AuditEvent.now(
            session_id=session_id,
            task_id=task_id,
            event_type="network_check",
            category="network",
            result=result,  # type: ignore[arg-type]
            policy_rule=rule,
            detail={"host": host},
        )
        event_bus.publish(event)
