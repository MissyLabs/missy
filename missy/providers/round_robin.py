"""Reusable multi-account round-robin balancing for providers (F15).

The per-account balancing that makes two credentials double a provider's
effective throughput (each account gets its own SDK client and its own
:class:`~missy.providers.rate_limiter.RateLimiter`, not one shared budget) was
implemented only inside ``OpenAIProvider``. This module lifts the generic part —
the account list, the thread-safe round-robin selector, and the per-account
rate limiters — into a small, provider-agnostic helper so any provider
(Anthropic, an OpenAI-compatible endpoint, …) can adopt round-robin by supplying
its keys plus a rate-limiter factory, while keeping its own (provider-specific)
client construction.

``OpenAIProvider`` uses this helper; the client-building and thread-local
"current account" tracking stay in the provider (a client is SDK-specific).

Per-account health tracking
----------------------------
Plain round-robin is blind to account health: an account whose credential has
hit a real upstream limit (quota exhausted, suspended, etc.) keeps getting
selected on schedule and fails on every one of its turns, degrading roughly
1-in-N calls indefinitely rather than routing around the bad account. Callers
report each call's outcome via :meth:`RoundRobinAccounts.record_success`/
:meth:`record_failure`; after *failure_threshold* consecutive failures an
account is skipped for a backoff window (doubling on a failed post-cooldown
retry, capped at *max_backoff_seconds* — the same closed/open/half-open shape
as :class:`~missy.agent.circuit_breaker.CircuitBreaker`, reimplemented here
rather than imported so this provider-layer module has no dependency on the
higher-level ``agent`` package). If every account is currently in backoff,
selection fails open (returns the one recovering soonest) rather than
refusing to serve at all -- a temporarily degraded provider should still be
usable, just imperfectly balanced.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Account:
    """One configured credential a provider round-robins across.

    Attributes:
        index: Stable 0-based position in the account list (surfaced in audit /
            diagnostics — never the key itself).
        api_key: The credential for this account.
        rate_limiter: This account's own rate limiter, so it has an independent
            budget rather than sharing one with the other accounts.
        client: The provider-built SDK client for this account, cached lazily by
            the provider (``None`` until first use).
        consecutive_failures: Failures recorded since the last success (or
            since construction). Reset to 0 by :meth:`RoundRobinAccounts.record_success`.
        unhealthy_until: ``time.monotonic()`` timestamp before which this
            account is skipped by selection. ``0.0`` (the default) means
            healthy/never opened.
        backoff_seconds: The backoff duration used the last time this account
            opened; ``0.0`` until the first time it does. Doubles (up to the
            selector's ``max_backoff_seconds``) each time a post-cooldown
            probe also fails, mirroring exponential backoff elsewhere in the
            codebase.
    """

    index: int
    api_key: str
    rate_limiter: Any
    client: Any | None = None
    consecutive_failures: int = field(default=0, repr=False)
    unhealthy_until: float = field(default=0.0, repr=False)
    backoff_seconds: float = field(default=0.0, repr=False)

    def __repr__(self) -> str:
        return f"Account(index={self.index}, api_key=<redacted>, client_ready={self.client is not None})"


@dataclass(frozen=True)
class AccountView:
    """Credential-free immutable public account state."""

    index: int
    client_ready: bool
    healthy: bool = True
    consecutive_failures: int = 0


class RoundRobinAccounts:
    """Thread-safe round-robin selector over a list of :class:`Account`.

    Args:
        keys: The configured credentials. Round-robin is only activated when at
            least *min_accounts* are supplied (a single key needs no balancing).
        make_rate_limiter: Zero-arg factory building a fresh, independent
            rate limiter for each account.
        min_accounts: Minimum keys required to enable balancing (default 2).
        failure_threshold: Consecutive failures on one account before it's
            skipped for a backoff window (default 5, matching
            :class:`~missy.agent.circuit_breaker.CircuitBreaker`'s default).
        base_backoff_seconds: Initial backoff window once an account opens
            (default 60.0).
        max_backoff_seconds: Cap on the doubling backoff window (default 300.0).
    """

    def __init__(
        self,
        keys: list[str] | None,
        make_rate_limiter: Callable[[], Any],
        *,
        min_accounts: int = 2,
        failure_threshold: int = 5,
        base_backoff_seconds: float = 60.0,
        max_backoff_seconds: float = 300.0,
    ) -> None:
        key_list = list(keys or [])
        self._accounts: list[Account] = (
            [
                Account(index=i, api_key=key, rate_limiter=make_rate_limiter())
                for i, key in enumerate(key_list)
            ]
            if len(key_list) >= min_accounts
            else []
        )
        self._index = 0
        self._lock = threading.Lock()
        self._failure_threshold = failure_threshold
        self._base_backoff_seconds = base_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds

    @property
    def accounts(self) -> tuple[AccountView, ...]:
        """Return credential-free immutable snapshots of account state."""
        with self._lock:
            now = time.monotonic()
            return tuple(
                AccountView(
                    index=account.index,
                    client_ready=account.client is not None,
                    healthy=account.unhealthy_until <= now,
                    consecutive_failures=account.consecutive_failures,
                )
                for account in self._accounts
            )

    @property
    def _live_accounts(self) -> list[Account]:
        """Private provider integration surface; never expose in diagnostics."""
        return self._accounts

    @property
    def is_multi_account(self) -> bool:
        """True when 2+ accounts are configured for balancing."""
        return bool(self._accounts)

    @property
    def count(self) -> int:
        """How many accounts are balanced across (0 when inactive)."""
        return len(self._accounts)

    def select(self) -> AccountView | None:
        """Return a credential-free view of the next account in rotation.

        Returns ``None`` when balancing is inactive, so the caller falls back to
        its single-credential path. Concurrent callers are assigned distinct
        accounts in rotation with no lost or duplicated turns.
        """
        account = self._select_live()
        if account is None:
            return None
        return AccountView(
            index=account.index,
            client_ready=account.client is not None,
            healthy=account.unhealthy_until <= time.monotonic(),
            consecutive_failures=account.consecutive_failures,
        )

    def _select_live(self) -> Account | None:
        """Private provider integration selector returning credential state.

        Skips accounts currently in their failure backoff window, continuing
        the rotation among whichever accounts are healthy. When every account
        is unhealthy, fails open and returns the one recovering soonest
        rather than refusing to select at all.
        """
        if not self._accounts:
            return None
        with self._lock:
            n = len(self._accounts)
            start = self._index
            self._index = (start + 1) % n
            now = time.monotonic()
            for offset in range(n):
                candidate = self._accounts[(start + offset) % n]
                if candidate.unhealthy_until <= now:
                    return candidate
            return min(self._accounts, key=lambda a: a.unhealthy_until)

    def record_success(self, account: Account) -> None:
        """Report that a call on *account* succeeded, clearing its backoff state."""
        with self._lock:
            account.consecutive_failures = 0
            account.backoff_seconds = 0.0
            account.unhealthy_until = 0.0

    def record_failure(self, account: Account) -> None:
        """Report that a call on *account* failed.

        After *failure_threshold* consecutive failures, *account* is skipped
        by :meth:`_select_live` until its backoff window elapses. A failure
        that happens once that window has already elapsed (i.e. this was a
        post-cooldown probe) doubles the backoff instead of restarting it at
        the base duration, up to *max_backoff_seconds* -- the same shape as
        :class:`~missy.agent.circuit_breaker.CircuitBreaker`'s HALF_OPEN
        probe-failed case.
        """
        with self._lock:
            account.consecutive_failures += 1
            if account.consecutive_failures < self._failure_threshold:
                return
            account.backoff_seconds = (
                min(account.backoff_seconds * 2, self._max_backoff_seconds)
                if account.backoff_seconds
                else self._base_backoff_seconds
            )
            account.unhealthy_until = time.monotonic() + account.backoff_seconds
