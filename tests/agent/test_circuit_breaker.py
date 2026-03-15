"""Tests for missy.agent.circuit_breaker.CircuitBreaker."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, call, patch

import pytest

from missy.agent.circuit_breaker import CircuitBreaker, CircuitState
from missy.core.exceptions import MissyError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_breaker(threshold: int = 3, base_timeout: float = 60.0, max_timeout: float = 300.0) -> CircuitBreaker:
    """Return a breaker with test-friendly defaults."""
    return CircuitBreaker("test", threshold=threshold, base_timeout=base_timeout, max_timeout=max_timeout)


def _trip(breaker: CircuitBreaker, n: int | None = None) -> None:
    """Drive *n* (default: threshold) consecutive failures through the breaker."""
    count = n if n is not None else breaker._threshold
    for _ in range(count):
        with pytest.raises(Exception):
            breaker.call(_raise)


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Construction / initial state
# ---------------------------------------------------------------------------


class TestInit:
    def test_initial_state_is_closed(self):
        breaker = _make_breaker()
        assert breaker.state == CircuitState.CLOSED

    def test_name_stored(self):
        breaker = CircuitBreaker("my-provider")
        assert breaker.name == "my-provider"

    def test_default_threshold(self):
        breaker = CircuitBreaker("x")
        assert breaker._threshold == 5

    def test_default_base_timeout(self):
        breaker = CircuitBreaker("x")
        assert breaker._base_timeout == 60.0

    def test_default_max_timeout(self):
        breaker = CircuitBreaker("x")
        assert breaker._max_timeout == 300.0

    def test_custom_parameters(self):
        breaker = CircuitBreaker("x", threshold=2, base_timeout=10.0, max_timeout=50.0)
        assert breaker._threshold == 2
        assert breaker._base_timeout == 10.0
        assert breaker._max_timeout == 50.0

    def test_initial_failure_count_is_zero(self):
        breaker = _make_breaker()
        assert breaker._failure_count == 0

    def test_initial_recovery_timeout_equals_base(self):
        breaker = _make_breaker(base_timeout=30.0)
        assert breaker._recovery_timeout == 30.0


# ---------------------------------------------------------------------------
# CLOSED state — normal operation
# ---------------------------------------------------------------------------


class TestClosedState:
    def test_call_succeeds_and_returns_value(self):
        breaker = _make_breaker()
        result = breaker.call(lambda: 42)
        assert result == 42

    def test_call_forwards_positional_args(self):
        breaker = _make_breaker()
        result = breaker.call(lambda a, b: a + b, 3, 4)
        assert result == 7

    def test_call_forwards_keyword_args(self):
        breaker = _make_breaker()
        result = breaker.call(lambda x, y: x * y, x=3, y=5)
        assert result == 15

    def test_failures_below_threshold_do_not_open(self):
        breaker = _make_breaker(threshold=3)
        _trip(breaker, n=2)
        assert breaker.state == CircuitState.CLOSED

    def test_single_failure_stays_closed(self):
        breaker = _make_breaker(threshold=5)
        _trip(breaker, n=1)
        assert breaker.state == CircuitState.CLOSED

    def test_success_resets_failure_count(self):
        breaker = _make_breaker(threshold=3)
        _trip(breaker, n=2)
        breaker.call(lambda: None)  # success
        assert breaker._failure_count == 0

    def test_success_after_partial_failures_prevents_open(self):
        breaker = _make_breaker(threshold=3)
        _trip(breaker, n=2)
        breaker.call(lambda: "ok")
        _trip(breaker, n=2)  # only 2 more — still below threshold
        assert breaker.state == CircuitState.CLOSED

    def test_exception_is_re_raised(self):
        breaker = _make_breaker(threshold=5)
        with pytest.raises(ValueError, match="sentinel"):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("sentinel")))

    def test_callable_exception_propagates_type(self):
        breaker = _make_breaker()

        def kaboom():
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            breaker.call(kaboom)


# ---------------------------------------------------------------------------
# Transition: CLOSED → OPEN
# ---------------------------------------------------------------------------


class TestClosedToOpen:
    def test_opens_at_threshold(self):
        breaker = _make_breaker(threshold=3)
        _trip(breaker, n=3)
        assert breaker.state == CircuitState.OPEN

    def test_opens_exactly_at_threshold_not_before(self):
        breaker = _make_breaker(threshold=4)
        _trip(breaker, n=3)
        assert breaker.state == CircuitState.CLOSED
        _trip(breaker, n=1)
        assert breaker.state == CircuitState.OPEN

    def test_threshold_of_one_opens_on_first_failure(self):
        breaker = _make_breaker(threshold=1)
        _trip(breaker, n=1)
        assert breaker.state == CircuitState.OPEN

    def test_failure_count_tracks_correctly_before_open(self):
        breaker = _make_breaker(threshold=5)
        _trip(breaker, n=3)
        assert breaker._failure_count == 3

    def test_last_failure_time_is_set_on_open(self):
        breaker = _make_breaker(threshold=2)
        before = time.monotonic()
        _trip(breaker, n=2)
        after = time.monotonic()
        assert before <= breaker._last_failure_time <= after


# ---------------------------------------------------------------------------
# OPEN state — calls rejected immediately
# ---------------------------------------------------------------------------


class TestOpenState:
    def test_open_rejects_call_with_missy_error(self):
        breaker = _make_breaker(threshold=2)
        _trip(breaker, n=2)
        with pytest.raises(MissyError):
            breaker.call(lambda: "should not run")

    def test_open_error_message_contains_name(self):
        breaker = CircuitBreaker("my-service", threshold=1)
        _trip(breaker, n=1)
        with pytest.raises(MissyError, match="my-service"):
            breaker.call(lambda: None)

    def test_open_error_message_mentions_open(self):
        breaker = _make_breaker(threshold=1)
        _trip(breaker, n=1)
        with pytest.raises(MissyError, match="OPEN"):
            breaker.call(lambda: None)

    def test_func_is_never_called_when_open(self):
        breaker = _make_breaker(threshold=1)
        _trip(breaker, n=1)
        mock_fn = MagicMock()
        with pytest.raises(MissyError):
            breaker.call(mock_fn)
        mock_fn.assert_not_called()

    def test_multiple_open_rejections_all_raise(self):
        breaker = _make_breaker(threshold=2)
        _trip(breaker, n=2)
        for _ in range(5):
            with pytest.raises(MissyError):
                breaker.call(lambda: None)


# ---------------------------------------------------------------------------
# Transition: OPEN → HALF_OPEN (timeout elapsed)
# ---------------------------------------------------------------------------


class TestOpenToHalfOpen:
    def test_transitions_to_half_open_after_timeout(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01)
        _trip(breaker, n=1)
        assert breaker.state == CircuitState.OPEN
        time.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN

    def test_does_not_transition_before_timeout(self):
        breaker = _make_breaker(threshold=1, base_timeout=60.0)
        _trip(breaker, n=1)
        assert breaker.state == CircuitState.OPEN

    def test_state_property_drives_transition(self):
        """Accessing .state is the trigger for OPEN → HALF_OPEN."""
        breaker = _make_breaker(threshold=1, base_timeout=0.01)
        _trip(breaker, n=1)
        time.sleep(0.02)
        # Internal state is still OPEN; the property must flip it.
        assert breaker._state == CircuitState.OPEN
        _ = breaker.state  # trigger the transition
        assert breaker._state == CircuitState.HALF_OPEN

    def test_monkeypatched_monotonic(self):
        """State transition works with a monkeypatched clock."""
        breaker = _make_breaker(threshold=1, base_timeout=30.0)

        with patch("missy.agent.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 1000.0
            _trip(breaker, n=1)

            # Still within timeout window
            mock_time.return_value = 1020.0
            assert breaker.state == CircuitState.OPEN

            # Jump past timeout
            mock_time.return_value = 1031.0
            assert breaker.state == CircuitState.HALF_OPEN


# ---------------------------------------------------------------------------
# HALF_OPEN: probe success → CLOSED
# ---------------------------------------------------------------------------


class TestHalfOpenSuccess:
    def _open_then_expire(self, breaker: CircuitBreaker) -> None:
        _trip(breaker, n=breaker._threshold)
        assert breaker.state == CircuitState.OPEN
        time.sleep(breaker._base_timeout * 1.1)

    def test_successful_probe_closes_circuit(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01)
        self._open_then_expire(breaker)
        assert breaker.state == CircuitState.HALF_OPEN
        breaker.call(lambda: "ok")
        assert breaker.state == CircuitState.CLOSED

    def test_success_resets_failure_count(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01)
        self._open_then_expire(breaker)
        breaker.call(lambda: None)
        assert breaker._failure_count == 0

    def test_success_resets_recovery_timeout_to_base(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01, max_timeout=1.0)
        _trip(breaker, n=1)
        time.sleep(0.02)
        # Force a failed probe to double the timeout first
        with pytest.raises(RuntimeError):
            breaker.call(_raise)
        # Now recover properly
        time.sleep(breaker._recovery_timeout * 1.1)
        breaker.call(lambda: None)
        assert breaker._recovery_timeout == breaker._base_timeout

    def test_closed_after_probe_allows_normal_calls(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01)
        self._open_then_expire(breaker)
        breaker.call(lambda: None)  # probe succeeds → CLOSED
        result = breaker.call(lambda: 99)
        assert result == 99

    def test_probe_return_value_is_forwarded(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01)
        self._open_then_expire(breaker)
        result = breaker.call(lambda: "probe_result")
        assert result == "probe_result"


# ---------------------------------------------------------------------------
# HALF_OPEN: probe failure → OPEN (with backoff)
# ---------------------------------------------------------------------------


class TestHalfOpenFailure:
    def _open_and_expire(self, breaker: CircuitBreaker) -> None:
        _trip(breaker, n=breaker._threshold)
        time.sleep(breaker._base_timeout * 1.1)

    def test_failed_probe_returns_to_open(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01)
        self._open_and_expire(breaker)
        assert breaker.state == CircuitState.HALF_OPEN
        with pytest.raises(RuntimeError):
            breaker.call(_raise)
        assert breaker.state == CircuitState.OPEN

    def test_failed_probe_doubles_recovery_timeout(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01, max_timeout=10.0)
        _trip(breaker, n=1)
        time.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN
        original_timeout = breaker._recovery_timeout
        with pytest.raises(RuntimeError):
            breaker.call(_raise)
        assert breaker._recovery_timeout == original_timeout * 2

    def test_failed_probe_exception_is_re_raised(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01)
        self._open_and_expire(breaker)

        with pytest.raises(ValueError, match="probe_error"):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("probe_error")))

    def test_open_after_failed_probe_rejects_immediately(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01)
        self._open_and_expire(breaker)
        with pytest.raises(RuntimeError):
            breaker.call(_raise)
        # Must be OPEN again and reject without calling the function
        mock_fn = MagicMock()
        with pytest.raises(MissyError):
            breaker.call(mock_fn)
        mock_fn.assert_not_called()


# ---------------------------------------------------------------------------
# Exponential backoff of recovery timeout
# ---------------------------------------------------------------------------


class TestExponentialBackoff:
    def _cycle_open_halfopen_fail(self, breaker: CircuitBreaker) -> None:
        """Drive one OPEN → HALF_OPEN → OPEN cycle via a failed probe."""
        assert breaker.state == CircuitState.OPEN
        time.sleep(breaker._recovery_timeout * 1.1)
        assert breaker.state == CircuitState.HALF_OPEN
        with pytest.raises(RuntimeError):
            breaker.call(_raise)
        assert breaker.state == CircuitState.OPEN

    def test_first_backoff_doubles_base(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01, max_timeout=1.0)
        _trip(breaker, n=1)
        self._cycle_open_halfopen_fail(breaker)
        assert breaker._recovery_timeout == pytest.approx(0.02)

    def test_second_backoff_doubles_again(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01, max_timeout=1.0)
        _trip(breaker, n=1)
        self._cycle_open_halfopen_fail(breaker)
        self._cycle_open_halfopen_fail(breaker)
        assert breaker._recovery_timeout == pytest.approx(0.04)

    def test_backoff_caps_at_max_timeout(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01, max_timeout=0.03)
        _trip(breaker, n=1)
        # First fail: 0.01 * 2 = 0.02
        self._cycle_open_halfopen_fail(breaker)
        # Second fail: 0.02 * 2 = 0.04, capped to 0.03
        self._cycle_open_halfopen_fail(breaker)
        assert breaker._recovery_timeout == pytest.approx(0.03)

    def test_backoff_does_not_exceed_max_timeout_after_many_cycles(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01, max_timeout=0.05)
        _trip(breaker, n=1)
        for _ in range(10):
            self._cycle_open_halfopen_fail(breaker)
        assert breaker._recovery_timeout <= 0.05

    def test_success_resets_backoff_to_base(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01, max_timeout=1.0)
        _trip(breaker, n=1)
        # Fail a probe to grow the timeout
        self._cycle_open_halfopen_fail(breaker)
        assert breaker._recovery_timeout > breaker._base_timeout
        # Now let the next probe succeed
        time.sleep(breaker._recovery_timeout * 1.1)
        breaker.call(lambda: None)  # success probe
        assert breaker._recovery_timeout == breaker._base_timeout


# ---------------------------------------------------------------------------
# Max timeout cap
# ---------------------------------------------------------------------------


class TestMaxTimeoutCap:
    def test_max_timeout_is_respected_on_failure(self):
        breaker = _make_breaker(threshold=1, base_timeout=200.0, max_timeout=300.0)
        _trip(breaker, n=1)
        # Manually set recovery_timeout close to max before the probe
        breaker._recovery_timeout = 200.0
        breaker._state = CircuitState.HALF_OPEN

        with pytest.raises(RuntimeError):
            breaker.call(_raise)

        assert breaker._recovery_timeout == 300.0  # 200 * 2 = 400, capped to 300

    def test_max_timeout_equal_to_base(self):
        """When max == base the timeout must not grow at all."""
        breaker = _make_breaker(threshold=1, base_timeout=0.01, max_timeout=0.01)
        _trip(breaker, n=1)
        time.sleep(0.02)
        with pytest.raises(RuntimeError):
            breaker.call(_raise)
        assert breaker._recovery_timeout == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Reset method
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_from_open_returns_to_closed(self):
        breaker = _make_breaker(threshold=1)
        _trip(breaker, n=1)
        assert breaker.state == CircuitState.OPEN
        breaker._on_success()  # equivalent of a reset via success path
        assert breaker.state == CircuitState.CLOSED

    def test_on_success_resets_failure_count(self):
        breaker = _make_breaker(threshold=5)
        _trip(breaker, n=3)
        breaker._on_success()
        assert breaker._failure_count == 0

    def test_on_success_resets_recovery_timeout(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01, max_timeout=1.0)
        _trip(breaker, n=1)
        time.sleep(0.02)
        with pytest.raises(RuntimeError):
            breaker.call(_raise)
        assert breaker._recovery_timeout > 0.01
        breaker._on_success()
        assert breaker._recovery_timeout == pytest.approx(0.01)

    def test_call_success_after_open_then_halfopen_resets_everything(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01)
        _trip(breaker, n=1)
        time.sleep(0.02)
        breaker.call(lambda: None)
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0
        assert breaker._recovery_timeout == breaker._base_timeout

    def test_closed_breaker_can_be_used_normally_after_success_reset(self):
        breaker = _make_breaker(threshold=2)
        _trip(breaker, n=1)
        breaker.call(lambda: None)
        # Another failure should start fresh count
        _trip(breaker, n=1)
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_failures_open_exactly_once(self):
        """Many threads hammering failures must open the circuit exactly once."""
        breaker = _make_breaker(threshold=10, base_timeout=60.0)
        errors = []
        barrier = threading.Barrier(20)

        def do_fail():
            barrier.wait()
            try:
                breaker.call(_raise)
            except (RuntimeError, MissyError):
                pass
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=do_fail) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert breaker.state == CircuitState.OPEN

    def test_concurrent_reads_of_state_are_safe(self):
        """Concurrent reads must not corrupt internal state."""
        breaker = _make_breaker(threshold=5, base_timeout=60.0)
        results = []
        errors = []

        def read_state():
            try:
                for _ in range(50):
                    results.append(breaker.state)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=read_state) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(s in CircuitState for s in results)

    def test_concurrent_success_and_failure_does_not_deadlock(self):
        breaker = _make_breaker(threshold=100, base_timeout=60.0)
        stop = threading.Event()
        errors = []

        def succeed():
            while not stop.is_set():
                try:
                    breaker.call(lambda: None)
                except Exception:
                    pass

        def fail():
            while not stop.is_set():
                try:
                    breaker.call(_raise)
                except Exception:
                    pass

        workers = [threading.Thread(target=succeed) for _ in range(3)]
        workers += [threading.Thread(target=fail) for _ in range(3)]
        for w in workers:
            w.start()

        time.sleep(0.1)
        stop.set()

        for w in workers:
            w.join(timeout=2.0)

        for w in workers:
            assert not w.is_alive(), "Thread is still alive — possible deadlock"

        assert not errors

    def test_state_lock_is_reentrant_safe(self):
        """Accessing .state from multiple threads simultaneously is safe."""
        breaker = _make_breaker(threshold=1, base_timeout=0.01)
        _trip(breaker, n=1)
        time.sleep(0.02)

        seen_states = []
        barrier = threading.Barrier(10)

        def check():
            barrier.wait()
            seen_states.append(breaker.state)

        threads = [threading.Thread(target=check) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should see HALF_OPEN (first access) or CLOSED (impossible
        # here since no success probe fired); the key assertion is no crash and
        # the state is a valid CircuitState value.
        assert all(s in CircuitState for s in seen_states)


# ---------------------------------------------------------------------------
# CircuitState enum
# ---------------------------------------------------------------------------


class TestCircuitStateEnum:
    def test_values_are_strings(self):
        assert CircuitState.CLOSED == "closed"
        assert CircuitState.OPEN == "open"
        assert CircuitState.HALF_OPEN == "half_open"

    def test_all_three_members_exist(self):
        members = {s for s in CircuitState}
        assert members == {CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN}

    def test_str_representation(self):
        assert str(CircuitState.CLOSED) == "closed"
        assert str(CircuitState.OPEN) == "open"
        assert str(CircuitState.HALF_OPEN) == "half_open"


# ---------------------------------------------------------------------------
# Edge cases / miscellaneous
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_call_with_no_failures_stays_closed_indefinitely(self):
        breaker = _make_breaker(threshold=3)
        for _ in range(100):
            breaker.call(lambda: None)
        assert breaker.state == CircuitState.CLOSED

    def test_alternating_success_failure_never_opens(self):
        breaker = _make_breaker(threshold=3)
        for _ in range(20):
            _trip(breaker, n=1)            # one failure
            breaker.call(lambda: None)     # one success (resets count)
        assert breaker.state == CircuitState.CLOSED

    def test_breaker_with_threshold_zero_still_functions(self):
        """Threshold=0 means the circuit opens immediately on construction — or
        whichever the first call is.  We only verify it does not raise during
        construction; actual threshold-of-zero semantics are implementation-defined."""
        # The implementation does not guard against threshold=0.
        # Just verify no AttributeError / TypeError on construction.
        breaker = CircuitBreaker("x", threshold=0)
        assert breaker._threshold == 0

    def test_recovery_timeout_does_not_go_below_base_via_success(self):
        breaker = _make_breaker(threshold=1, base_timeout=10.0)
        # Artificially set a lower-than-base recovery timeout
        breaker._recovery_timeout = 1.0
        breaker._on_success()
        # On success, recovery_timeout is reset to base_timeout
        assert breaker._recovery_timeout == 10.0

    def test_call_passes_through_return_none(self):
        breaker = _make_breaker()
        result = breaker.call(lambda: None)
        assert result is None

    def test_multiple_named_breakers_are_independent(self):
        a = CircuitBreaker("service-a", threshold=2)
        b = CircuitBreaker("service-b", threshold=2)
        _trip(a, n=2)
        assert a.state == CircuitState.OPEN
        assert b.state == CircuitState.CLOSED

    def test_open_rejection_does_not_modify_failure_count(self):
        breaker = _make_breaker(threshold=2)
        _trip(breaker, n=2)
        count_before = breaker._failure_count
        with pytest.raises(MissyError):
            breaker.call(lambda: None)
        # Rejected call should not increment failure_count further
        assert breaker._failure_count == count_before

    def test_func_raising_base_exception_subclass_is_tracked(self):
        """Any Exception subclass (not BaseException) is caught and counted."""
        breaker = _make_breaker(threshold=2)

        class CustomError(Exception):
            pass

        def raise_custom():
            raise CustomError("custom")

        with pytest.raises(CustomError):
            breaker.call(raise_custom)
        assert breaker._failure_count == 1
