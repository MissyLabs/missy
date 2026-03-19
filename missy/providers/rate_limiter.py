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
import threading
import time

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when a request cannot be serviced within the wait timeout."""

    def __init__(self, wait_seconds: float) -> None:
        self.wait_seconds = wait_seconds
        super().__init__(f"Rate limit exceeded; would need to wait {wait_seconds:.1f}s")


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
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100_000,
        max_wait_seconds: float = 30.0,
    ) -> None:
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        self._max_wait = max_wait_seconds
        self._lock = threading.Lock()

        # Request bucket
        self._req_tokens = float(requests_per_minute) if requests_per_minute > 0 else 0.0
        self._req_last_refill = time.monotonic()

        # Token bucket
        self._tok_tokens = float(tokens_per_minute) if tokens_per_minute > 0 else 0.0
        self._tok_last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill both buckets based on elapsed time (must hold lock)."""
        now = time.monotonic()

        if self._rpm > 0:
            elapsed = now - self._req_last_refill
            refill = elapsed * (self._rpm / 60.0)
            self._req_tokens = min(float(self._rpm), self._req_tokens + refill)
            self._req_last_refill = now

        if self._tpm > 0:
            elapsed = now - self._tok_last_refill
            refill = elapsed * (self._tpm / 60.0)
            self._tok_tokens = min(float(self._tpm), self._tok_tokens + refill)
            self._tok_last_refill = now

    def acquire(self, tokens: int = 0) -> None:
        """Acquire permission for one API request, blocking if necessary.

        Args:
            tokens: Estimated token count for this request (used for
                token-per-minute limiting).

        Raises:
            RateLimitExceeded: If the request cannot be serviced within
                ``max_wait_seconds``.
        """
        if self._rpm <= 0 and self._tpm <= 0:
            return  # No limits configured

        deadline = time.monotonic() + self._max_wait

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
                    return

                # Calculate wait time
                waits = []
                if not req_ok and self._rpm > 0:
                    waits.append((1.0 - self._req_tokens) / (self._rpm / 60.0))
                if not tok_ok and self._tpm > 0 and tokens > 0:
                    needed = float(tokens) - self._tok_tokens
                    waits.append(needed / (self._tpm / 60.0))
                wait_needed = max(waits) if waits else 0.1

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RateLimitExceeded(wait_needed)

            sleep_time = min(wait_needed, remaining, 0.5)
            time.sleep(sleep_time)

    def record_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Deduct actual token usage after a response is received.

        Call this after getting the actual token counts from the API response
        to adjust the token budget.  The ``acquire()`` call already deducted
        the *estimated* tokens; this method corrects the bucket to reflect
        the *actual* consumption.

        Args:
            prompt_tokens: Actual input tokens consumed.
            completion_tokens: Actual output tokens generated.
        """
        if self._tpm <= 0:
            return
        total = prompt_tokens + completion_tokens
        if total <= 0:
            return
        with self._lock:
            self._tok_tokens = max(0.0, self._tok_tokens - float(total))

    def on_rate_limit_response(self, retry_after: float = 0.0) -> None:
        """Handle a 429 response from the API.

        Drains the request bucket to zero and waits for the specified
        retry-after period.

        Args:
            retry_after: Seconds to wait (from the API's Retry-After header).
        """
        with self._lock:
            self._req_tokens = 0.0
            if self._tpm > 0:
                self._tok_tokens = 0.0
        if retry_after > 0:
            logger.info("Rate limited by API; waiting %.1fs", retry_after)
            time.sleep(min(retry_after, self._max_wait))

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
