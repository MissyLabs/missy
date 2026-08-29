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
        weight: Relative weight (> 0) for this account in the smooth
            weighted round-robin (provider-preference hierarchy,
            ``ProviderConfig.account_weights``). Defaults to ``1.0``, in
            which case selection is identical to plain round-robin.
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
        current_weight: Smooth weighted round-robin bookkeeping (nginx-style
            "current effective weight"); private selection state, not part
            of any public view.
    """

    index: int
    api_key: str
    rate_limiter: Any
    weight: float = 1.0
    client: Any | None = None
    consecutive_failures: int = field(default=0, repr=False)
    unhealthy_until: float = field(default=0.0, repr=False)
    backoff_seconds: float = field(default=0.0, repr=False)
    current_weight: float = field(default=0.0, repr=False)

    def __repr__(self) -> str:
        return f"Account(index={self.index}, api_key=<redacted>, client_ready={self.client is not None})"


@dataclass(frozen=True)
class AccountView:
    """Credential-free immutable public account state."""

    index: int
    client_ready: bool
    healthy: bool = True
    consecutive_failures: int = 0
    weight: float = 1.0


class RoundRobinAccounts:
    """Thread-safe round-robin selector over a list of :class:`Account`.

    Args:
        keys: The configured credentials. Round-robin is only activated when at
            least *min_accounts* are supplied (a single key needs no balancing).
        make_rate_limiter: Zero-arg factory building a fresh, independent
            rate limiter for each account.
        weights: Optional per-account weights parallel to *keys*
            (``ProviderConfig.account_weights``). ``None`` or an empty list
            gives every account weight ``1.0`` -- identical to plain
            round-robin, the original behavior. When provided, must be the
            same length as *keys*.
        min_accounts: Minimum keys required to enable balancing (default 2).
        failure_threshold: Consecutive failures on one account before it's
            skipped for a backoff window (default 5, matching
            :class:`~missy.agent.circuit_breaker.CircuitBreaker`'s default).
        base_backoff_seconds: Initial backoff window once an account opens
            (default 60.0).
        max_backoff_seconds: Cap on the doubling backoff window (default 300.0).

    Raises:
        ValueError: If *weights* is non-empty and its length does not
            match *keys*.
    """

    def __init__(
        self,
        keys: list[str] | None,
        make_rate_limiter: Callable[[], Any],
        *,
        weights: list[float] | None = None,
        min_accounts: int = 2,
        failure_threshold: int = 5,
        base_backoff_seconds: float = 60.0,
        max_backoff_seconds: float = 300.0,
    ) -> None:
        key_list = list(keys or [])
        weight_list = list(weights) if weights else [1.0] * len(key_list)
        if len(weight_list) != len(key_list):
            raise ValueError(
                f"weights must be the same length as keys ({len(key_list)}), "
                f"got {len(weight_list)}."
            )
        self._accounts: list[Account] = (
            [
                Account(
                    index=i,
                    api_key=key,
                    rate_limiter=make_rate_limiter(),
                    weight=weight_list[i],
                )
                for i, key in enumerate(key_list)
            ]
            if len(key_list) >= min_accounts
            else []
        )
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
                    weight=account.weight,
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

    def capacity_summary(self) -> dict[str, float]:
        """Aggregate rate-limit budget across every account (credential-free).

        Each account has its own independent :class:`RateLimiter`; this sums
        their configured limits and current live capacity so a caller (the
        Web TUI's provider usage view) can show "how much usage is
        available" for a multi-account provider as one combined bar, the
        same shape as a single-account provider's own rate limiter exposes.
        Unlimited (0) on any one account makes the whole sum unlimited,
        matching :class:`RateLimiter`'s own 0-means-unlimited convention.
        """
        with self._lock:
            accounts = list(self._accounts)
        if not accounts:
            return {
                "requests_per_minute": 0,
                "request_capacity": float("inf"),
                "tokens_per_minute": 0,
                "token_capacity": float("inf"),
            }
        rpm_values = [a.rate_limiter.requests_per_minute for a in accounts]
        tpm_values = [a.rate_limiter.tokens_per_minute for a in accounts]
        unlimited_rpm = any(v == 0 for v in rpm_values)
        unlimited_tpm = any(v == 0 for v in tpm_values)
        return {
            "requests_per_minute": 0 if unlimited_rpm else sum(rpm_values),
            "request_capacity": (
                float("inf") if unlimited_rpm else sum(a.rate_limiter.request_capacity for a in accounts)
            ),
            "tokens_per_minute": 0 if unlimited_tpm else sum(tpm_values),
            "token_capacity": (
                float("inf") if unlimited_tpm else sum(a.rate_limiter.token_capacity for a in accounts)
            ),
        }

    @property
    def count(self) -> int:
        """How many accounts are balanced across (0 when inactive)."""
        return len(self._accounts)

    def per_account_capacity(self) -> list[dict]:
        """Return each account's health + rate-limit budget, credential-free.

        Deliberately omits any account *name* -- ``Account.api_key`` holds
        an actual secret for some providers (e.g. raw OpenAI API keys) and
        only a safe display name for others (e.g. Codex OAuth account
        slugs), so this module can't decide which is safe to expose.
        Callers (each provider's own ``list_accounts()``) zip this
        credential-free capacity/health data with whatever name they know
        is safe to show for *their* account type, by matching on ``index``.
        """
        with self._lock:
            accounts = list(self._accounts)
        now = time.monotonic()
        return [
            {
                "index": account.index,
                "healthy": account.unhealthy_until <= now,
                "consecutive_failures": account.consecutive_failures,
                "weight": account.weight,
                "client_ready": account.client is not None,
                "rate_limit": account.rate_limiter.capacity_dict(),
            }
            for account in accounts
        ]

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
            weight=account.weight,
        )

    def _select_live(self) -> Account | None:
        """Private provider integration selector returning credential state.

        Skips accounts currently in their failure backoff window, continuing
        the rotation among whichever accounts are healthy. When every account
        is unhealthy, fails open and returns the one recovering soonest
        rather than refusing to select at all.

        Healthy accounts are picked via smooth weighted round-robin
        (nginx-style: each account's ``current_weight`` accumulates its
        configured weight every round; the highest is picked and then
        debited by the round's total weight). With every account's weight
        equal to ``1.0`` (the default) this produces the exact same strict
        0, 1, 2, ... rotation as plain round-robin.
        """
        if not self._accounts:
            return None
        with self._lock:
            now = time.monotonic()
            healthy = [a for a in self._accounts if a.unhealthy_until <= now]
            if not healthy:
                return min(self._accounts, key=lambda a: a.unhealthy_until)
            total = sum(a.weight for a in healthy)
            for a in healthy:
                a.current_weight += a.weight
            picked = max(healthy, key=lambda a: a.current_weight)
            picked.current_weight -= total
            return picked

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
