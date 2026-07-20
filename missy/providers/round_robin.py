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
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
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
    """

    index: int
    api_key: str
    rate_limiter: Any
    client: Any | None = None

    def __repr__(self) -> str:
        return f"Account(index={self.index}, api_key=<redacted>, client_ready={self.client is not None})"


@dataclass(frozen=True)
class AccountView:
    """Credential-free immutable public account state."""

    index: int
    client_ready: bool


class RoundRobinAccounts:
    """Thread-safe round-robin selector over a list of :class:`Account`.

    Args:
        keys: The configured credentials. Round-robin is only activated when at
            least *min_accounts* are supplied (a single key needs no balancing).
        make_rate_limiter: Zero-arg factory building a fresh, independent
            rate limiter for each account.
        min_accounts: Minimum keys required to enable balancing (default 2).
    """

    def __init__(
        self,
        keys: list[str] | None,
        make_rate_limiter: Callable[[], Any],
        *,
        min_accounts: int = 2,
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

    @property
    def accounts(self) -> tuple[AccountView, ...]:
        """Return credential-free immutable snapshots of account state."""
        with self._lock:
            return tuple(
                AccountView(index=account.index, client_ready=account.client is not None)
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
        return AccountView(index=account.index, client_ready=account.client is not None)

    def _select_live(self) -> Account | None:
        """Private provider integration selector returning credential state."""
        if not self._accounts:
            return None
        with self._lock:
            idx = self._index
            self._index = (idx + 1) % len(self._accounts)
        return self._accounts[idx]
