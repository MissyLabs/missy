"""Token bucket rate limiter for provider API calls.

Prevents burst requests from exceeding provider rate limits.  Each
provider instance can have its own :class:`RateLimiter` that enforces
requests-per-minute and tokens-per-minute constraints.

The implementation uses a sliding-window token bucket that refills
continuously.  When the bucket is empty, :meth:`acquire` blocks (up to
a configurable timeout) until capacity is available.

Example::

    from missy.providers.rate_limiter import RateLimiter

    limiter = RateLimiter(requests_per_minute=60, tokens_per_minute=100_000)
    limiter.acquire()          # blocks if rate exceeded
    limiter.acquire(tokens=500)  # deducts from token budget too
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

logger = logging.getLogger(__name__)


def parse_retry_after(
    value: Any,
    *,
    default: float = 5.0,
    maximum: float = 300.0,
    now: datetime | None = None,
) -> float:
    """Parse Retry-After seconds or an HTTP-date into a finite bounded delay."""
    safe_default = min(max(float(default), 0.0), max(float(maximum), 0.0))
    bound = max(float(maximum), 0.0)
    if value is None:
        return safe_default
    text = str(value).strip()
    if not text:
        return safe_default
    try:
        seconds = float(text)
        if not math.isfinite(seconds) or seconds < 0:
            return safe_default
        return min(seconds, bound)
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        parsed = parsedate_to_datetime(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        current = now or datetime.now(UTC)
        seconds = (parsed.astimezone(UTC) - current.astimezone(UTC)).total_seconds()
        if not math.isfinite(seconds):
            return safe_default
        return min(max(seconds, 0.0), bound)
    except (TypeError, ValueError, OverflowError):
        return safe_default


class RateLimitExceeded(Exception):
    """Raised when a request cannot be serviced within the wait timeout."""

    def __init__(self, wait_seconds: float) -> None:
        self.wait_seconds = wait_seconds
        super().__init__(f"Rate limit exceeded; would need to wait {wait_seconds:.1f}s")


class RateLimitRequestTooLarge(RateLimitExceeded, ValueError):
    """Raised when one request can never fit in the configured token bucket."""

    def __init__(self, requested_tokens: int, capacity: int) -> None:
        self.requested_tokens = requested_tokens
        self.capacity = capacity
        RateLimitExceeded.__init__(self, float("inf"))
        self.args = (
            f"Token estimate {requested_tokens} exceeds the configured "
            f"tokens-per-minute capacity {capacity}",
        )


@dataclass(frozen=True, slots=True)
class RateLimitReservation:
    """Opaque identity for one tracked token estimate.

    Passing the reservation back to :meth:`RateLimiter.record_usage` makes
    reconciliation idempotent and lets the limiter settle concurrent responses
    in acquisition order, even when responses arrive out of order.
    """

    _owner: object
    sequence: int
    estimated_tokens: int


@dataclass(slots=True)
class _PendingReconciliation:
    estimate: int
    actual: int | None = None


class RateLimiter:
    """Token bucket rate limiter with request and token budgets.

    Thread-safe: all mutations are guarded by an internal lock.

    Args:
        requests_per_minute: Maximum API requests per minute (0 = unlimited).
        tokens_per_minute: Maximum tokens per minute (0 = unlimited).
        max_wait_seconds: Maximum blocking wait time in :meth:`acquire`.
    """

    def __init__(
        self,
        requests_per_minute: int | None = 60,
        tokens_per_minute: int | None = 100_000,
        max_wait_seconds: float = 30.0,
        *,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self._rpm = self._validate_limit("requests_per_minute", requests_per_minute)
        self._tpm = self._validate_limit("tokens_per_minute", tokens_per_minute)
        self._max_wait = self._validate_wait(max_wait_seconds)
        # Keep defaults dynamic so runtime instrumentation and tests may
        # replace the module clock/sleeper after construction. Explicit fake
        # functions remain fixed and deterministic.
        self._clock = clock if clock is not None else lambda: time.monotonic()
        self._sleep = sleeper if sleeper is not None else lambda delay: time.sleep(delay)
        self._lock = threading.Lock()

        self._reservation_owner = object()
        self._next_reservation = 1
        self._reservation_order: deque[int] = deque()
        self._pending_reconciliations: dict[int, _PendingReconciliation] = {}

        # Request bucket
        self._req_tokens = float(self._rpm) if self._rpm > 0 else 0.0
        self._req_last_refill = self._clock()

        # Token bucket
        self._tok_tokens = float(self._tpm) if self._tpm > 0 else 0.0
        self._tok_last_refill = self._clock()

    @staticmethod
    def _validate_limit(name: str, value: int | None) -> int:
        """Normalize ``None`` to unlimited and reject ambiguous limits."""
        if value is None:
            return 0
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer or None")
        return value

    @staticmethod
    def _validate_wait(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("max_wait_seconds must be a finite non-negative number")
        result = float(value)
        if not math.isfinite(result) or result < 0:
            raise ValueError("max_wait_seconds must be a finite non-negative number")
        return result

    @staticmethod
    def _validate_token_count(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    def _refill(self) -> None:
        """Refill both buckets based on elapsed time (must hold lock)."""
        now = self._clock()

        if self._rpm > 0:
            elapsed = now - self._req_last_refill
            if elapsed > 0:
                refill = elapsed * (self._rpm / 60.0)
                self._req_tokens = min(float(self._rpm), self._req_tokens + refill)
                self._req_last_refill = now

        if self._tpm > 0:
            elapsed = now - self._tok_last_refill
            if elapsed > 0:
                refill = elapsed * (self._tpm / 60.0)
                self._tok_tokens = min(float(self._tpm), self._tok_tokens + refill)
                self._tok_last_refill = now

    def acquire(self, tokens: int = 0, *, reconcile: bool = False) -> RateLimitReservation | None:
        """Acquire permission for one API request, blocking if necessary.

        Args:
            tokens: Estimated token count for this request (used for
                token-per-minute limiting).

        Raises:
            RateLimitExceeded: If the request cannot be serviced within
                ``max_wait_seconds``.
        """
        tokens = self._validate_token_count("tokens", tokens)
        if self._tpm > 0 and tokens > self._tpm:
            raise RateLimitRequestTooLarge(tokens, self._tpm)
        if self._rpm <= 0 and self._tpm <= 0:
            return None  # No limits configured

        deadline = self._clock() + self._max_wait

        while True:
            with self._lock:
                self._refill()

                req_ok = self._rpm <= 0 or self._req_tokens >= 1.0
                tok_ok = self._tpm <= 0 or tokens <= 0 or self._tok_tokens >= tokens

                if req_ok and tok_ok:
                    if self._rpm > 0:
                        self._req_tokens -= 1.0
                    if self._tpm > 0 and tokens > 0:
                        self._tok_tokens -= max(0.0, float(tokens))
                    if reconcile and self._tpm > 0:
                        sequence = self._next_reservation
                        self._next_reservation += 1
                        self._reservation_order.append(sequence)
                        self._pending_reconciliations[sequence] = _PendingReconciliation(tokens)
                        return RateLimitReservation(
                            self._reservation_owner,
                            sequence,
                            tokens,
                        )
                    return None

                # Calculate wait time
                waits = []
                if not req_ok and self._rpm > 0:
                    waits.append((1.0 - self._req_tokens) / (self._rpm / 60.0))
                if not tok_ok and self._tpm > 0 and tokens > 0:
                    needed = float(tokens) - self._tok_tokens
                    waits.append(needed / (self._tpm / 60.0))
                wait_needed = max(waits) if waits else 0.1

            remaining = deadline - self._clock()
            if remaining <= 0:
                raise RateLimitExceeded(wait_needed)

            sleep_time = min(wait_needed, remaining, 0.5)
            self._sleep(sleep_time)

    def record_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        estimated_tokens: int = 0,
        *,
        reservation: RateLimitReservation | None = None,
    ) -> bool:
        """Reconcile the token bucket against actual usage after a response.

        The ``acquire()`` call already deducted *estimated_tokens* from the
        bucket before the request was sent. This method must first credit
        that estimate back, then deduct the *actual* consumption -- a net
        adjustment of ``estimated_tokens - actual_total``. Deducting the
        actual total on top of the already-deducted estimate (without ever
        crediting the estimate back) would double-charge every single
        call, exhausting the configured ``tokens_per_minute`` budget at
        roughly half the intended rate and causing spurious blocking/
        `RateLimitExceeded` under normal, correctly-configured load.

        Args:
            prompt_tokens: Actual input tokens consumed.
            completion_tokens: Actual output tokens generated.
            estimated_tokens: The estimate originally passed to the
                :meth:`acquire` call this response corresponds to.
        """
        prompt_tokens = self._validate_token_count("prompt_tokens", prompt_tokens)
        completion_tokens = self._validate_token_count("completion_tokens", completion_tokens)
        estimated_tokens = self._validate_token_count("estimated_tokens", estimated_tokens)
        if self._tpm <= 0:
            return False
        total = prompt_tokens + completion_tokens

        if reservation is not None:
            if not isinstance(reservation, RateLimitReservation):
                raise ValueError("reservation must be returned by acquire(reconcile=True)")
            if reservation._owner is not self._reservation_owner:
                raise ValueError("reservation belongs to a different rate limiter")
            with self._lock:
                pending = self._pending_reconciliations.get(reservation.sequence)
                if pending is None or pending.actual is not None:
                    return False
                pending.actual = total
                self._settle_reconciliations()
            return True

        net_adjustment = float(estimated_tokens) - float(total)
        if net_adjustment == 0:
            return False
        with self._lock:
            self._tok_tokens = max(0.0, min(float(self._tpm), self._tok_tokens + net_adjustment))
        return True

    def cancel_reservation(self, reservation: RateLimitReservation) -> bool:
        """Finalize a failed request while retaining its conservative estimate.

        Cancellation applies a zero reconciliation adjustment: the estimate
        already consumed at acquisition remains charged, but the request no
        longer blocks deterministic settlement of later responses.
        """
        if not isinstance(reservation, RateLimitReservation):
            raise ValueError("reservation must be returned by acquire(reconcile=True)")
        if reservation._owner is not self._reservation_owner:
            raise ValueError("reservation belongs to a different rate limiter")
        with self._lock:
            pending = self._pending_reconciliations.get(reservation.sequence)
            if pending is None or pending.actual is not None:
                return False
            pending.actual = pending.estimate
            self._settle_reconciliations()
        return True

    def _settle_reconciliations(self) -> None:
        """Apply completed tracked requests in acquisition order (lock held)."""
        while self._reservation_order:
            sequence = self._reservation_order[0]
            pending = self._pending_reconciliations[sequence]
            if pending.actual is None:
                break
            self._reservation_order.popleft()
            del self._pending_reconciliations[sequence]
            adjustment = float(pending.estimate - pending.actual)
            self._tok_tokens = max(
                0.0,
                min(float(self._tpm), self._tok_tokens + adjustment),
            )

    def on_rate_limit_response(self, retry_after: float = 0.0) -> None:
        """Handle a 429 response from the API.

        Drains the request bucket to zero and waits for the specified
        retry-after period.

        Args:
            retry_after: Seconds to wait (from the API's Retry-After header).
        """
        retry_after = self._validate_wait(retry_after)
        with self._lock:
            self._req_tokens = 0.0
            if self._tpm > 0:
                self._tok_tokens = 0.0
        if retry_after > 0:
            logger.info("Rate limited by API; waiting %.1fs", retry_after)
            self._sleep(min(retry_after, self._max_wait))

    @property
    def request_capacity(self) -> float:
        """Current request bucket capacity (0.0 to rpm)."""
        with self._lock:
            self._refill()
            return self._req_tokens if self._rpm > 0 else float("inf")

    @property
    def token_capacity(self) -> float:
        """Current token bucket capacity (0.0 to tpm)."""
        with self._lock:
            self._refill()
            return self._tok_tokens if self._tpm > 0 else float("inf")
