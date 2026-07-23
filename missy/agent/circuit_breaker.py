"""Circuit breaker for provider failure isolation.

Implements the classic Closed → Open → HalfOpen state machine with
exponential backoff on the recovery timeout.

Example::

    from missy.agent.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker("anthropic", threshold=3, base_timeout=30.0)

    def call_provider():
        ...

    result = breaker.call(call_provider)
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import StrEnum


class CircuitState(StrEnum):
    """Possible states of a :class:`CircuitBreaker`."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Closed → Open → HalfOpen state machine with exponential backoff.

    The circuit starts CLOSED (normal operation).  After *threshold*
    consecutive failures it moves to OPEN and rejects calls immediately.
    After *base_timeout* seconds it transitions to HALF_OPEN to allow a
    single probe call.  A successful probe resets to CLOSED; a failed probe
    doubles the recovery timeout (up to *max_timeout*) and returns to OPEN.

    This class is thread-safe.

    Args:
        name: Identifier for logging.
        threshold: Consecutive failures before opening (default 5).
        base_timeout: Initial recovery timeout in seconds (default 60).
        max_timeout: Maximum recovery timeout in seconds (default 300).
    """

    def __init__(
        self,
        name: str,
        threshold: int = 5,
        base_timeout: float = 60.0,
        max_timeout: float = 300.0,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.name = name
        self._threshold = threshold
        self._base_timeout = base_timeout
        self._max_timeout = max_timeout
        # Keep the default lookup dynamic so existing operators/tests that
        # monkeypatch ``time.monotonic`` after construction still work.
        self._clock = clock if clock is not None else lambda: time.monotonic()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._recovery_timeout: float = base_timeout
        self._lock = threading.Lock()
        # Availability-hardening: True while exactly one HALF_OPEN probe
        # call is in flight. See call()'s HALF_OPEN branch.
        self._probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state.

        Transitions OPEN → HALF_OPEN automatically when the recovery timeout
        has elapsed since the last failure.

        Returns:
            The current :class:`CircuitState`.
        """
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and self._clock() - self._last_failure_time >= self._recovery_timeout
            ):
                self._state = CircuitState.HALF_OPEN
            return self._state

    def call(self, func, *args, **kwargs):
        """Execute *func* and track success/failure against the circuit state.

        Args:
            func: Callable to execute.
            *args: Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            The return value of *func*.

        Raises:
            MissyError: When the circuit is OPEN (call is rejected).
            Exception: Re-raises any exception raised by *func* after
                recording the failure.
        """
        # Atomically check-and-transition to prevent TOCTOU races. This
        # covers two distinct races, not just one:
        #   1. OPEN -> HALF_OPEN: only the thread that observes the
        #      recovery timeout has elapsed makes the transition.
        #   2. Concurrent calls arriving *while already HALF_OPEN* (e.g.
        #      one thread just transitioned it a moment ago): only one
        #      such call may proceed as "the" probe. Every other
        #      concurrent caller is rejected until that probe resolves --
        #      HALF_OPEN means "allow a single probe call," not "allow
        #      every call that happens to see this state." Without this,
        #      an arbitrary number of threads can all slip through the
        #      instant the state flips, hammering a backend that just
        #      started recovering (confirmed via a live concurrency
        #      reproduction: 5 threads racing a freshly-HALF_OPEN breaker
        #      all executed func() before this fix).
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._clock() - self._last_failure_time >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._probe_in_flight = True
                else:
                    from missy.core.exceptions import MissyError

                    raise MissyError(f"Circuit breaker '{self.name}' is OPEN; skipping call")
            elif self._state == CircuitState.HALF_OPEN:
                if self._probe_in_flight:
                    from missy.core.exceptions import MissyError

                    raise MissyError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN; "
                        "a probe call is already in flight"
                    )
                self._probe_in_flight = True
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Reset failure tracking and close the circuit."""
        with self._lock:
            self._failure_count = 0
            self._recovery_timeout = self._base_timeout
            self._state = CircuitState.CLOSED
            self._probe_in_flight = False

    def _on_failure(self) -> None:
        """Record a failure; open the circuit when threshold is exceeded."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = self._clock()
            if self._state == CircuitState.HALF_OPEN:
                # Probe failed: back off further and re-open
                self._recovery_timeout = min(self._recovery_timeout * 2, self._max_timeout)
                self._state = CircuitState.OPEN
                self._probe_in_flight = False
            elif self._failure_count >= self._threshold:
                self._state = CircuitState.OPEN
