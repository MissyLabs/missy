"""Generic smooth weighted round-robin selector (provider-preference hierarchy).

Lifted out as its own small, name-keyed structure rather than folded into
:mod:`~missy.providers.round_robin` because it operates one level up:
:class:`~missy.providers.round_robin.RoundRobinAccounts` balances
*accounts* (credentials) within a single provider, each with its own SDK
client and rate limiter; this balances across whole, already-constructed
*providers* (e.g. ``anthropic`` vs ``openai`` vs ``ollama``), which need
none of that per-account bookkeeping. Both use the same smooth weighted
round-robin shape -- equal weights degrade to plain round-robin, verified
by :mod:`tests.providers.test_weighted_selector` -- so an operator reasons
about one balancing algorithm across the whole codebase, not two.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class _WeightedEntry:
    """Per-name smooth-WRR bookkeeping. Never exposed outside this module."""

    name: str
    weight: float
    current_weight: float = field(default=0.0, repr=False)


class WeightedRoundRobin:
    """Thread-safe smooth weighted round-robin over named, weighted entries.

    Unlike :class:`~missy.providers.round_robin.RoundRobinAccounts`, this
    selector has no fixed entry list: :meth:`select` is given the current
    *candidates* and their *weights* on every call, so it stays correct
    across a changing set of eligible providers (availability, runtime
    enable/disable, circuit-breaker state) without needing to be rebuilt.
    Rotation state (``current_weight``) persists per name across calls so
    balancing remains proportional over many selections even though the
    candidate set fluctuates.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _WeightedEntry] = {}
        self._lock = threading.Lock()

    def select(self, candidates: list[str], weights: dict[str, float]) -> str | None:
        """Return the next weighted pick among *candidates*.

        Args:
            candidates: Names eligible for this selection. Order does not
                affect the weighted outcome, but ties in ``current_weight``
                (e.g. every entry's first-ever selection) are broken by
                first occurrence in this list, matching plain round-robin
                for equal weights.
            weights: Weight for each candidate name. A name missing from
                this mapping defaults to weight ``1.0``. A weight ``<= 0``
                excludes that name from this selection entirely (it can
                still be reached by explicit name elsewhere) but its
                rotation state is preserved in case it becomes eligible
                again later with a positive weight.

        Returns:
            The selected name, or ``None`` when no candidate has a
            positive weight.
        """
        with self._lock:
            pool: list[_WeightedEntry] = []
            for name in candidates:
                weight = max(0.0, weights.get(name, 1.0))
                if weight <= 0:
                    continue
                entry = self._entries.setdefault(name, _WeightedEntry(name=name, weight=weight))
                entry.weight = weight
                pool.append(entry)
            if not pool:
                return None
            total = sum(entry.weight for entry in pool)
            for entry in pool:
                entry.current_weight += entry.weight
            picked = max(pool, key=lambda entry: entry.current_weight)
            picked.current_weight -= total
            return picked.name
