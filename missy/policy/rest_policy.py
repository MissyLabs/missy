"""L7 REST policy enforcement for the Missy framework.

:class:`RestPolicy` evaluates HTTP method + path rules on a per-host basis,
providing fine-grained control beyond host/domain-level network policy.

Rules are evaluated top-to-bottom; the first matching rule wins.  When no
rule matches, ``None`` is returned so the caller can fall through to the
existing network policy.

Example config::

    rest_policies:
      - host: "api.github.com"
        method: "GET"
        path: "/repos/**"
        action: "allow"
      - host: "api.github.com"
        method: "DELETE"
        path: "/**"
        action: "deny"
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RestRule:
    """A single REST policy rule.

    Attributes:
        host: Hostname to match (exact match, case-insensitive).
        method: HTTP method to match (case-insensitive), or ``"*"`` for any.
        path: URL path glob pattern (fnmatch).
        action: ``"allow"`` or ``"deny"``.
    """

    host: str
    method: str
    path: str
    action: str


class RestPolicy:
    """Evaluates HTTP method + path rules against a list of REST rules.

    Rules are evaluated top-to-bottom; first match wins.  If no rule matches,
    :meth:`check` returns ``None`` (pass-through to network policy).

    Args:
        rules: Ordered list of :class:`RestRule` instances.
    """

    def __init__(self, rules: list[RestRule] | None = None) -> None:
        self._rules: list[RestRule] = list(rules) if rules else []

    @classmethod
    def from_config(cls, raw_rules: list[dict]) -> RestPolicy:
        """Build a :class:`RestPolicy` from a list of config dicts.

        Each dict must contain ``host``, ``method``, ``path``, and ``action``.
        """
        rules: list[RestRule] = []
        for entry in raw_rules:
            rules.append(
                RestRule(
                    host=str(entry.get("host", "")).lower(),
                    method=str(entry.get("method", "*")).upper(),
                    path=str(entry.get("path", "/**")),
                    action=str(entry.get("action", "deny")).lower(),
                )
            )
        return cls(rules)

    def check(self, host: str, method: str, path: str) -> str | None:
        """Evaluate a request against REST rules.

        Args:
            host: Target hostname (case-insensitive).
            method: HTTP method (e.g. ``"GET"``).
            path: URL path (e.g. ``"/repos/foo/bar"``).

        Returns:
            ``"allow"`` or ``"deny"`` if a rule matches, ``None`` otherwise.
        """
        if not self._rules:
            return None

        host_lower = host.lower()
        method_upper = method.upper()

        for rule in self._rules:
            if rule.host != host_lower:
                continue
            if rule.method != "*" and rule.method != method_upper:
                continue
            if not fnmatch.fnmatch(path, rule.path):
                continue
            logger.debug(
                "REST policy match: %s %s%s -> %s (rule: %s %s %s)",
                method_upper,
                host_lower,
                path,
                rule.action,
                rule.host,
                rule.method,
                rule.path,
            )
            return rule.action

        return None
