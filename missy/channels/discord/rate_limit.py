"""Per-user command rate limiting for the Discord channel.

DISC-CMD-008 (task #10 validation): no dedicated rate limiter existed
for incoming Discord commands (slash or natural-language), so a single
user could spam paid LLM calls with only the overall session
``CostTracker`` budget as a backstop. This module adds a real,
non-blocking, per-author token bucket that the channel checks before
dispatching any command-producing message or interaction.

Unlike :class:`missy.providers.rate_limiter.RateLimiter` (a single
global, blocking bucket meant for outbound provider API calls), this is
keyed per Discord user ID, never blocks the event loop, and evicts
idle buckets so memory stays bounded regardless of how many distinct
users the bot has ever seen.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

#: Buckets untouched for longer than this are evicted on the next check
#: to keep memory bounded for long-running bots with many distinct users.
_IDLE_EVICTION_SECONDS = 3600.0


@dataclass(frozen=True)
class RateLimitResult:
    """Outcome of a rate-limit check for one user."""

    allowed: bool
    retry_after_seconds: float = 0.0


class _UserBucket:
    __slots__ = ("tokens", "last_refill", "last_touched")

    def __init__(self, capacity: float, now: float) -> None:
        # `now` must be the exact same timestamp the caller uses for its
        # own elapsed-time computation immediately after construction --
        # calling time.monotonic() again here would let this bucket's
        # last_refill land microseconds *after* that "now", producing a
        # negative elapsed time on the very first check() and starting a
        # brand new user one token short of full capacity.
        self.tokens = capacity
        self.last_refill = now
        self.last_touched = now


class DiscordUserRateLimiter:
    """Per-user token bucket rate limiter for Discord command dispatch.

    Thread-safe. ``check()`` never blocks — it immediately returns
    whether the request is allowed and, if not, how long the caller
    should tell the user to wait.

    Args:
        requests_per_minute: Maximum commands per user per minute.
            ``0`` disables rate limiting entirely (always allowed).
    """

    def __init__(self, requests_per_minute: int = 10) -> None:
        self._rpm = max(0, int(requests_per_minute))
        self._lock = threading.Lock()
        self._buckets: dict[str, _UserBucket] = {}

    def check(self, user_id: str) -> RateLimitResult:
        """Check and, if allowed, consume one token for *user_id*.

        Args:
            user_id: Discord snowflake ID of the invoking user.

        Returns:
            A :class:`RateLimitResult` — ``allowed=True`` also consumes
            one token from that user's bucket; ``allowed=False`` leaves
            the bucket untouched so the user isn't further penalized
            for requests that were already refused.
        """
        if self._rpm <= 0 or not user_id:
            return RateLimitResult(allowed=True)

        now = time.monotonic()
        with self._lock:
            self._evict_idle_locked(now)

            bucket = self._buckets.get(user_id)
            if bucket is None:
                bucket = _UserBucket(capacity=float(self._rpm), now=now)
                self._buckets[user_id] = bucket

            elapsed = now - bucket.last_refill
            refill = elapsed * (self._rpm / 60.0)
            bucket.tokens = min(float(self._rpm), bucket.tokens + refill)
            bucket.last_refill = now
            bucket.last_touched = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return RateLimitResult(allowed=True)

            retry_after = (1.0 - bucket.tokens) / (self._rpm / 60.0)
            return RateLimitResult(allowed=False, retry_after_seconds=retry_after)

    def _evict_idle_locked(self, now: float) -> None:
        """Remove buckets untouched for longer than the idle window.

        Caller must hold ``self._lock``.
        """
        stale = [
            uid
            for uid, bucket in self._buckets.items()
            if now - bucket.last_touched > _IDLE_EVICTION_SECONDS
        ]
        for uid in stale:
            del self._buckets[uid]

    @property
    def tracked_user_count(self) -> int:
        """Number of users with an active (non-evicted) bucket."""
        with self._lock:
            return len(self._buckets)
