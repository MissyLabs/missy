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
        allowed, _ip = self.check_host_resolved(
            host, session_id=session_id, task_id=task_id, category=category
        )
        return allowed

    def check_host_resolved(
        self,
        host: str,
        session_id: str = "",
        task_id: str = "",
        category: str = "",
    ) -> tuple[bool, str | None]:
        """Like :meth:`check_host`, but also returns the validated IP.

        SR-1.9b: ``check_host`` alone tells a caller *that* a host is
        allowed, but the IP it validated is discarded — if the caller
        then makes its own, independent DNS resolution to actually
        connect (as ``httpx``/``httpcore`` does by default), a low-TTL
        DNS record can return a different address between the check and
        the connect (DNS rebinding), silently bypassing every check this
        method just performed. Callers that go on to make the actual
        connection (:class:`~missy.gateway.client.PolicyHTTPClient`)
        must use the returned IP to pin that connection, not re-resolve.

        Returns:
            ``(True, ip)`` when allowed and a concrete IP was resolved
            and validated; ``(True, None)`` only in ``default_deny=False``
            mode, which never performs DNS resolution at all (an
            established, deliberately-tested property of that mode) and
            so has nothing to pin -- but also has no security boundary
            to protect, since everything is already allowed. Raises on
            denial, same as :meth:`check_host`.

        Raises:
            PolicyViolationError: When the host is denied by policy.
            ValueError: When *host* is ``None`` or an empty string.
        """
        if not host:
            raise ValueError("host must be a non-empty string")

        # Normalise: strip surrounding brackets from IPv6 literals such as
        # [::1] that arrive from URL parsers.
        host = host.strip("[]").lower()

        # Step 1 – default-allow mode bypasses all remaining checks,
        # including DNS resolution (an established, deliberately-tested
        # property: default_deny=False must never trigger a DNS lookup,
        # e.g. for pure offline/local-only setups). No security boundary
        # exists to protect in this mode, so there is nothing to pin the
        # connection against -- the transport falls back to normal,
        # unpinned resolution for this one request (see
        # missy.gateway.pinned_transport: a ``None`` pin is treated as
        # "resolve normally," not "deny").
        if not self._policy.default_deny:
            self._emit_event(host, "allow", "default_allow", session_id, task_id)
            return True, None

        # Step 2 – bare IP: check CIDR lists only (no DNS) — the "resolved"
        # IP is simply the literal itself.
        if self._is_ip(host):
            rule = self._check_cidr(host)
            if rule:
                self._emit_event(host, "allow", rule, session_id, task_id)
                return True, host
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
            resolved = self._resolve_and_check_rebinding(host, session_id, task_id)
            self._emit_event(host, "allow", rule, session_id, task_id)
            return True, resolved[0][0]

        # Step 4 – wildcard / suffix domain match. Same SR-1.9a rebinding
        # check as step 3.
        rule = self._check_domain(host)
        if rule:
            resolved = self._resolve_and_check_rebinding(host, session_id, task_id)
            self._emit_event(host, "allow", rule, session_id, task_id)
            return True, resolved[0][0]

        # Step 5 – DNS resolution + CIDR re-check for hostnames not matched
        # by name at all.
        try:
            resolved = self._resolve_and_check_rebinding(host, session_id, task_id)
            if resolved:
                for ip_str, _addr in resolved:
                    rule = self._check_cidr(ip_str)
                    if rule:
                        self._emit_event(host, "allow", rule, session_id, task_id)
                        return True, ip_str
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

    @staticmethod
    def _resolve_best_effort(
        host: str,
    ) -> list[tuple[str, ipaddress.IPv4Address | ipaddress.IPv6Address]]:
        """Resolve *host* with no rebinding/CIDR checks.

        Used both for ``default_deny=False`` mode (no security boundary
        to enforce, but still worth pinning to *some* validated-at-
        lookup-time address) and by :mod:`missy.gateway.client`'s
        interactive-approval override path (an explicit human override
        of a policy denial still needs a best-effort pin so the
        connection doesn't fail closed against the SR-1.9b transport).
        """
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return []
        seen_ips: set[str] = set()
        resolved: list[tuple[str, ipaddress.IPv4Address | ipaddress.IPv6Address]] = []
        for info in infos:
            ip_str = str(info[4][0])
            if ip_str in seen_ips:
                continue
            seen_ips.add(ip_str)
            try:
                addr = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            resolved.append((ip_str, addr))
        return resolved

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_and_check_rebinding(
        self,
        host: str,
        session_id: str,
        task_id: str,
    ) -> list[tuple[str, ipaddress.IPv4Address | ipaddress.IPv6Address]]:
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
            address. At least one validated address is always returned.

        Raises:
            PolicyViolationError: When DNS resolution fails, produces no
                usable addresses, or a resolved IP is private/reserved and
                not covered by ``allowed_cidrs``.
        """
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            logger.warning(
                "NetworkPolicyEngine: DNS resolution failed for %r; denying because "
                "the destination could not be validated",
                host,
            )
            self._emit_event(host, "deny", None, session_id, task_id)
            raise PolicyViolationError(
                f"Network access denied: DNS resolution failed for {host!r} during "
                "policy validation.",
                category="network",
                detail=(
                    f"Hostname {host!r} could not be resolved to a policy-validated "
                    "address. Default-deny mode does not permit a later unvalidated "
                    "DNS lookup."
                ),
            ) from None

        seen_ips: set[str] = set()
        resolved: list[tuple[str, ipaddress.IPv4Address | ipaddress.IPv6Address]] = []
        for info in infos:
            ip_str = str(info[4][0])
            if ip_str in seen_ips:
                continue
            seen_ips.add(ip_str)
            try:
                addr = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            resolved.append((ip_str, addr))

        if not resolved:
            logger.warning(
                "NetworkPolicyEngine: DNS resolution returned no usable addresses for %r; denying",
                host,
            )
            self._emit_event(host, "deny", None, session_id, task_id)
            raise PolicyViolationError(
                f"Network access denied: DNS resolution returned no valid addresses for {host!r}.",
                category="network",
                detail=(
                    f"Hostname {host!r} did not resolve to any valid IPv4 or IPv6 "
                    "address that could be validated by policy."
                ),
            )

        for ip_str, addr in resolved:
            # Hostname allowlists express trust in a DNS name, not blanket
            # access to every address class that name might later target.
            # ``is_private`` alone misses important SSRF destinations such
            # as carrier-grade NAT (100.64.0.0/10) and multicast; require a
            # globally routable unicast address unless the operator
            # explicitly allowed the exact address range via allowed_cidrs.
            disallowed = (
                not addr.is_global
                or addr.is_multicast
                or addr.is_unspecified
                or addr.is_reserved
                or addr.is_loopback
                or addr.is_link_local
            )
            if disallowed:
                rule = self._check_cidr(ip_str)
                if not rule:
                    logger.warning(
                        "NetworkPolicyEngine: DNS rebinding blocked — %r resolved to "
                        "non-public address %s which is not in allowed_cidrs",
                        host,
                        ip_str,
                    )
                    self._emit_event(host, "deny", None, session_id, task_id)
                    raise PolicyViolationError(
                        f"Network access denied: {host!r} resolved to private, reserved, "
                        f"or otherwise non-public address {ip_str} "
                        "(possible DNS rebinding attack).",
                        category="network",
                        detail=(
                            f"Hostname {host!r} resolved to {ip_str}, a private, reserved, "
                            "multicast, local, or otherwise non-public address not explicitly "
                            "allowed by policy."
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
