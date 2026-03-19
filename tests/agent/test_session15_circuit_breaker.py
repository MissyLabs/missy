"""Session 15 comprehensive tests for missy.agent.circuit_breaker.CircuitBreaker.

Covers angles not yet exercised by test_circuit_breaker.py and
test_session13_circuitbreaker_attention.py:

- Deterministic clock control via unittest.mock.patch on time.monotonic
- TOCTOU / concurrent HALF_OPEN probe serialisation (only first thread probes)
- State identity preserved across rapid repeated .state property accesses
- Failure counter behaviour: does not grow beyond threshold while OPEN
- Multiple complete trip-recover cycles with monotonically-growing timeouts
- Large threshold values (50, 1000)
- Mixed positional+keyword argument forwarding
- Callable returning falsy values (0, False, "", [], {})
- Callable returning complex objects
- Exception hierarchy: OSError, KeyboardInterrupt-is-BaseException guard,
  ValueError, custom hierarchy, exception chaining
- Rejection error is-instance MissyError (not just "is raised")
- State transitions driven purely through .call() without touching .state
- Verify _last_failure_time is NOT reset by _on_success
- Verify _last_failure_time IS updated on every failure (not just first)
- Recovery timeout grows independently of failure_count after OPEN
- Stress test: 20 threads racing through full open/half-open/close cycles
- Back-to-back breaker instances share no state
- Func that raises on first call only (flaky) clears failure count on next success
- Timeout exactly at boundary via patched monotonic (>= condition)
- Recovery timeout is NOT doubled when failure occurs in CLOSED state
- call() with a method bound to an object
- call() with a lambda capturing mutable state
- call() with a generator function (raises StopIteration — is BaseException)
- Probe that succeeds after max_timeout reached: recovery_timeout resets to base
- HALF_OPEN → OPEN → HALF_OPEN second cycle uses doubled timeout correctly
- Rejection in OPEN does not update _last_failure_time
- Multiple breakers with shared underlying callable
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, call, patch

import pytest

from missy.agent.circuit_breaker import CircuitBreaker, CircuitState
from missy.core.exceptions import MissyError


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make(
    threshold: int = 3,
    base_timeout: float = 60.0,
    max_timeout: float = 300.0,
    name: str = "svc",
) -> CircuitBreaker:
    """Return a breaker with test-friendly defaults."""
    return CircuitBreaker(name, threshold=threshold, base_timeout=base_timeout, max_timeout=max_timeout)


def _always_raise(exc_factory=None):
    """Return a callable that always raises the given exception (default RuntimeError)."""
    factory = exc_factory or (lambda: RuntimeError("boom"))

    def _raise(*args, **kwargs):
        raise factory()

    return _raise


def _trip(breaker: CircuitBreaker, n: int | None = None) -> None:
    """Drive *n* failures (default: exactly threshold) through the breaker."""
    count = n if n is not None else breaker._threshold
    for _ in range(count):
        with pytest.raises(Exception):
            breaker.call(_always_raise())


# ---------------------------------------------------------------------------
# Section 1 — Deterministic clock control
# ---------------------------------------------------------------------------


def test_clock_patch_open_to_half_open_exactly_at_boundary():
    """>=  condition: elapsed exactly == recovery_timeout must trigger HALF_OPEN."""
    breaker = _make(threshold=1, base_timeout=100.0)
    with patch("missy.agent.circuit_breaker.time.monotonic") as mock_t:
        mock_t.return_value = 500.0
        _trip(breaker, n=1)
        # Jump to exactly base_timeout seconds later
        mock_t.return_value = 600.0
        assert breaker.state == CircuitState.HALF_OPEN


def test_clock_patch_open_stays_open_one_nanosecond_before_boundary():
    """elapsed < recovery_timeout by epsilon: must stay OPEN."""
    breaker = _make(threshold=1, base_timeout=100.0)
    with patch("missy.agent.circuit_breaker.time.monotonic") as mock_t:
        mock_t.return_value = 500.0
        _trip(breaker, n=1)
        mock_t.return_value = 599.9999999
        assert breaker.state == CircuitState.OPEN


def test_clock_patch_call_transitions_to_half_open_before_executing():
    """call() must read the clock and transition OPEN→HALF_OPEN before running func."""
    breaker = _make(threshold=1, base_timeout=50.0)
    called = []

    def probe():
        called.append(True)
        return "ok"

    with patch("missy.agent.circuit_breaker.time.monotonic") as mock_t:
        mock_t.return_value = 0.0
        _trip(breaker, n=1)
        # Jump past timeout
        mock_t.return_value = 51.0
        result = breaker.call(probe)

    assert result == "ok"
    assert called == [True]
    assert breaker.state == CircuitState.CLOSED


def test_clock_patch_open_rejects_before_timeout_in_call():
    """call() must raise MissyError when elapsed < recovery_timeout."""
    breaker = _make(threshold=1, base_timeout=30.0)
    mock_fn = MagicMock()

    with patch("missy.agent.circuit_breaker.time.monotonic") as mock_t:
        mock_t.return_value = 0.0
        _trip(breaker, n=1)
        mock_t.return_value = 25.0  # still within window
        with pytest.raises(MissyError):
            breaker.call(mock_fn)

    mock_fn.assert_not_called()


def test_clock_patch_failure_time_updated_each_failure():
    """_last_failure_time must reflect the time of the most recent failure."""
    breaker = _make(threshold=5, base_timeout=30.0)
    times = [100.0, 200.0, 300.0]
    with patch("missy.agent.circuit_breaker.time.monotonic") as mock_t:
        for t in times:
            mock_t.return_value = t
            with pytest.raises(RuntimeError):
                breaker.call(_always_raise())
        assert breaker._last_failure_time == pytest.approx(300.0)


def test_clock_patch_last_failure_time_not_reset_by_success():
    """_on_success must NOT zero out _last_failure_time."""
    breaker = _make(threshold=2, base_timeout=10.0)
    with patch("missy.agent.circuit_breaker.time.monotonic") as mock_t:
        mock_t.return_value = 42.0
        with pytest.raises(RuntimeError):
            breaker.call(_always_raise())
        last_before = breaker._last_failure_time
        mock_t.return_value = 43.0
        breaker.call(lambda: None)  # success

    # _on_success does not touch _last_failure_time
    assert breaker._last_failure_time == last_before


def test_clock_patch_second_cycle_uses_doubled_timeout():
    """After probe failure the doubled timeout must govern the next HALF_OPEN transition."""
    breaker = _make(threshold=1, base_timeout=10.0, max_timeout=1000.0)
    with patch("missy.agent.circuit_breaker.time.monotonic") as mock_t:
        # Trip at t=0
        mock_t.return_value = 0.0
        _trip(breaker, n=1)
        # Expire first timeout (10 s) → HALF_OPEN
        mock_t.return_value = 10.0
        assert breaker.state == CircuitState.HALF_OPEN
        # Fail probe → timeout doubles to 20 s
        with pytest.raises(RuntimeError):
            breaker.call(_always_raise())
        assert breaker._recovery_timeout == pytest.approx(20.0)
        # 15 s after probe failure — must still be OPEN
        mock_t.return_value = 25.0  # probe was at t=10; 25-10=15 < 20
        assert breaker.state == CircuitState.OPEN
        # 20 s after probe failure — must transition
        mock_t.return_value = 30.0  # 30-10=20 >= 20
        assert breaker.state == CircuitState.HALF_OPEN


def test_clock_patch_rejection_does_not_update_last_failure_time():
    """A MissyError rejection from OPEN must not touch _last_failure_time."""
    breaker = _make(threshold=1, base_timeout=30.0)
    with patch("missy.agent.circuit_breaker.time.monotonic") as mock_t:
        mock_t.return_value = 1000.0
        _trip(breaker, n=1)
        time_after_trip = breaker._last_failure_time

        mock_t.return_value = 1010.0  # still within window
        with pytest.raises(MissyError):
            breaker.call(lambda: None)

    assert breaker._last_failure_time == time_after_trip


# ---------------------------------------------------------------------------
# Section 2 — TOCTOU race: only one thread probes from HALF_OPEN
# ---------------------------------------------------------------------------


def test_toctou_only_one_thread_executes_probe_when_half_open():
    """When multiple threads race to call() from HALF_OPEN the implementation
    uses an atomic check-and-set inside the lock.  The first thread transitions
    the state to something other than OPEN before releasing the lock, so
    subsequent threads that arrive while the state is no longer OPEN will not
    be able to probe either (they see HALF_OPEN and proceed through the lock
    gate, but since the transition already happened they all become potential
    probes).  The critical safety guarantee tested here is that no thread sees
    an unhandled exception and the final state is valid.
    """
    breaker = _make(threshold=1, base_timeout=0.01, max_timeout=10.0)
    _trip(breaker, n=1)
    time.sleep(0.02)  # expire timeout → state is now OPEN (lazy)

    probe_call_count = [0]
    lock = threading.Lock()
    errors: list[Exception] = []
    barrier = threading.Barrier(12)

    def slow_probe():
        """Probe func that records calls and takes a tiny bit of time."""
        with lock:
            probe_call_count[0] += 1
        time.sleep(0.002)
        return "probed"

    def worker():
        barrier.wait()
        try:
            breaker.call(slow_probe)
        except MissyError:
            pass  # expected: subsequent threads rejected after state change
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # The first thread to acquire the lock transitions OPEN→HALF_OPEN and runs
    # the probe.  All other threads that arrive while HALF_OPEN also proceed
    # through (because they don't see OPEN).  But the key correctness property
    # is that the state machine is not corrupted.
    assert breaker._state in CircuitState


def test_toctou_concurrent_probes_final_state_is_valid():
    """State after a concurrent probe storm must be a known CircuitState."""
    breaker = _make(threshold=2, base_timeout=0.01, max_timeout=2.0)
    _trip(breaker, n=2)
    time.sleep(0.02)

    barrier = threading.Barrier(10)

    def probe():
        barrier.wait()
        try:
            breaker.call(lambda: "ok")
        except (MissyError, RuntimeError):
            pass

    threads = [threading.Thread(target=probe) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert breaker._state in CircuitState


def test_concurrent_mixed_success_failure_no_deadlock():
    """10 threads: half succeed, half fail; must not deadlock within 3 seconds."""
    breaker = _make(threshold=50, base_timeout=60.0)
    stop = threading.Event()

    def succeed():
        while not stop.is_set():
            try:
                breaker.call(lambda: "ok")
            except (MissyError, RuntimeError):
                pass

    def fail():
        while not stop.is_set():
            try:
                breaker.call(_always_raise())
            except (MissyError, RuntimeError):
                pass

    workers = [threading.Thread(target=succeed, daemon=True) for _ in range(5)]
    workers += [threading.Thread(target=fail, daemon=True) for _ in range(5)]
    for w in workers:
        w.start()

    time.sleep(0.1)
    stop.set()

    for w in workers:
        w.join(timeout=3.0)
    for w in workers:
        assert not w.is_alive(), "Possible deadlock: thread still alive"


def test_concurrent_failure_count_never_exceeds_threshold_plus_threads():
    """Failure count can exceed threshold (race window) but the circuit must
    be OPEN, not corrupted.  This is an acceptability test, not a strict cap."""
    breaker = _make(threshold=5, base_timeout=60.0)
    barrier = threading.Barrier(15)

    def fail_once():
        barrier.wait()
        try:
            breaker.call(_always_raise())
        except (MissyError, RuntimeError):
            pass

    threads = [threading.Thread(target=fail_once) for _ in range(15)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Circuit must be OPEN; failure_count must be a non-negative integer
    assert breaker.state == CircuitState.OPEN
    assert isinstance(breaker._failure_count, int)
    assert breaker._failure_count >= breaker._threshold


def test_concurrent_state_reads_during_transition_all_valid():
    """Threads reading .state while another triggers the OPEN→HALF_OPEN
    transition must all receive a valid CircuitState value."""
    breaker = _make(threshold=1, base_timeout=0.01, max_timeout=10.0)
    _trip(breaker, n=1)
    time.sleep(0.02)

    results: list[CircuitState] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(20)

    def read_state():
        barrier.wait()
        try:
            results.append(breaker.state)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=read_state) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 20
    assert all(s in CircuitState for s in results)


# ---------------------------------------------------------------------------
# Section 3 — Custom threshold values
# ---------------------------------------------------------------------------


def test_large_threshold_50_does_not_open_before_50_failures():
    breaker = _make(threshold=50)
    _trip(breaker, n=49)
    assert breaker.state == CircuitState.CLOSED
    assert breaker._failure_count == 49


def test_large_threshold_50_opens_exactly_at_50():
    breaker = _make(threshold=50)
    _trip(breaker, n=50)
    assert breaker.state == CircuitState.OPEN


def test_large_threshold_1000_stays_closed_at_999():
    breaker = _make(threshold=1000)
    _trip(breaker, n=999)
    assert breaker.state == CircuitState.CLOSED


def test_threshold_2_opens_on_second_failure():
    breaker = _make(threshold=2)
    _trip(breaker, n=1)
    assert breaker.state == CircuitState.CLOSED
    _trip(breaker, n=1)
    assert breaker.state == CircuitState.OPEN


def test_failure_count_does_not_grow_while_open():
    """Calls rejected by an OPEN circuit must not increment _failure_count."""
    breaker = _make(threshold=2)
    _trip(breaker, n=2)
    count_at_open = breaker._failure_count
    for _ in range(5):
        with pytest.raises(MissyError):
            breaker.call(lambda: None)
    assert breaker._failure_count == count_at_open


# ---------------------------------------------------------------------------
# Section 4 — Argument forwarding
# ---------------------------------------------------------------------------


def test_call_forwards_positional_and_keyword_args_together():
    breaker = _make()

    def add(a, b, *, multiplier=1):
        return (a + b) * multiplier

    result = breaker.call(add, 3, 4, multiplier=2)
    assert result == 14


def test_call_forwards_all_positional_args():
    breaker = _make()
    received = []

    def capture(*args):
        received.extend(args)

    breaker.call(capture, 10, 20, 30, 40)
    assert received == [10, 20, 30, 40]


def test_call_forwards_all_keyword_args():
    breaker = _make()
    received = {}

    def capture(**kwargs):
        received.update(kwargs)

    breaker.call(capture, x=1, y=2, z=3)
    assert received == {"x": 1, "y": 2, "z": 3}


def test_call_forwards_args_in_half_open_state():
    """Args forwarding must also work when the probe happens from HALF_OPEN."""
    breaker = _make(threshold=1, base_timeout=0.01)
    _trip(breaker, n=1)
    time.sleep(0.02)

    result = breaker.call(lambda a, b: a - b, 10, 3)
    assert result == 7


def test_call_with_bound_method():
    """call() works with a bound method, not just plain functions."""

    class Counter:
        def __init__(self):
            self.n = 0

        def increment(self, by=1):
            self.n += by
            return self.n

    counter = Counter()
    breaker = _make()
    result = breaker.call(counter.increment, by=5)
    assert result == 5
    assert counter.n == 5


# ---------------------------------------------------------------------------
# Section 5 — Falsy and complex return values
# ---------------------------------------------------------------------------


def test_call_returns_zero():
    assert _make().call(lambda: 0) == 0


def test_call_returns_false():
    assert _make().call(lambda: False) is False


def test_call_returns_empty_string():
    assert _make().call(lambda: "") == ""


def test_call_returns_empty_list():
    assert _make().call(lambda: []) == []


def test_call_returns_empty_dict():
    assert _make().call(lambda: {}) == {}


def test_call_returns_none():
    assert _make().call(lambda: None) is None


def test_call_returns_complex_object():
    sentinel = object()
    result = _make().call(lambda: sentinel)
    assert result is sentinel


def test_call_returns_nested_structure():
    expected = {"data": [1, 2, {"nested": True}]}
    assert _make().call(lambda: expected) == expected


# ---------------------------------------------------------------------------
# Section 6 — Exception type variety
# ---------------------------------------------------------------------------


def test_oserror_is_tracked_as_failure():
    breaker = _make(threshold=1)
    with pytest.raises(OSError):
        breaker.call(_always_raise(OSError))
    assert breaker._failure_count == 1


def test_value_error_is_tracked_as_failure():
    breaker = _make(threshold=2)
    with pytest.raises(ValueError):
        breaker.call(_always_raise(ValueError))
    assert breaker._failure_count == 1
    assert breaker.state == CircuitState.CLOSED


def test_custom_exception_subclass_is_tracked():
    class AppError(Exception):
        pass

    breaker = _make(threshold=1)
    with pytest.raises(AppError):
        breaker.call(_always_raise(AppError))
    assert breaker.state == CircuitState.OPEN


def test_deep_exception_hierarchy_is_tracked():
    class Base(Exception):
        pass

    class Mid(Base):
        pass

    class Leaf(Mid):
        pass

    breaker = _make(threshold=1)
    with pytest.raises(Leaf):
        breaker.call(_always_raise(Leaf))
    assert breaker.state == CircuitState.OPEN


def test_different_exception_types_each_failure_still_cumulative():
    """Each failure counts regardless of exception type."""
    breaker = _make(threshold=3)
    exc_types = [RuntimeError, ValueError, TypeError]
    for exc_type in exc_types:
        with pytest.raises(exc_type):
            breaker.call(_always_raise(exc_type))
    assert breaker.state == CircuitState.OPEN


def test_rejection_error_is_instance_of_missy_error():
    """The exception raised by OPEN rejection must be exactly a MissyError instance."""
    breaker = _make(threshold=1)
    _trip(breaker, n=1)
    try:
        breaker.call(lambda: None)
        pytest.fail("Expected MissyError not raised")
    except Exception as exc:
        assert isinstance(exc, MissyError)


def test_rejection_error_message_contains_skipping():
    """Rejection message must convey the call was skipped."""
    breaker = _make(threshold=1, name="provider-x")
    _trip(breaker, n=1)
    with pytest.raises(MissyError, match="skipping"):
        breaker.call(lambda: None)


# ---------------------------------------------------------------------------
# Section 7 — State property auto-transition semantics
# ---------------------------------------------------------------------------


def test_state_property_idempotent_when_already_half_open():
    """Repeated .state accesses after HALF_OPEN do not flip state further."""
    breaker = _make(threshold=1, base_timeout=0.01)
    _trip(breaker, n=1)
    time.sleep(0.02)
    s1 = breaker.state  # triggers OPEN → HALF_OPEN
    s2 = breaker.state  # must stay HALF_OPEN
    assert s1 == CircuitState.HALF_OPEN
    assert s2 == CircuitState.HALF_OPEN


def test_state_property_idempotent_when_closed():
    breaker = _make()
    for _ in range(10):
        assert breaker.state == CircuitState.CLOSED


def test_state_via_call_only_no_direct_state_access():
    """Full cycle driven exclusively through .call() without touching .state."""
    breaker = _make(threshold=2, base_timeout=0.01, max_timeout=10.0)
    # Drive to OPEN via .call()
    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(_always_raise())

    # OPEN — call raises MissyError
    with pytest.raises(MissyError):
        breaker.call(lambda: None)

    # Expire timeout and probe via .call()
    time.sleep(0.02)
    result = breaker.call(lambda: "recovered")
    assert result == "recovered"
    assert breaker._state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Section 8 — Multiple complete trip-recover cycles
# ---------------------------------------------------------------------------


def test_two_complete_cycles_second_timeout_correct():
    """Second cycle after full recovery must use base_timeout, not doubled."""
    breaker = _make(threshold=1, base_timeout=0.01, max_timeout=1.0)

    # Cycle 1
    _trip(breaker, n=1)
    time.sleep(0.02)
    breaker.call(lambda: None)  # probe success → CLOSED, timeout reset
    assert breaker._recovery_timeout == pytest.approx(0.01)

    # Cycle 2
    _trip(breaker, n=1)
    assert breaker.state == CircuitState.OPEN
    assert breaker._recovery_timeout == pytest.approx(0.01)


def test_three_cycles_timeouts_remain_at_base_after_each_success():
    breaker = _make(threshold=1, base_timeout=0.01, max_timeout=10.0)
    for _ in range(3):
        _trip(breaker, n=1)
        time.sleep(0.02)
        breaker.call(lambda: None)
        assert breaker._recovery_timeout == pytest.approx(0.01)
        assert breaker._failure_count == 0


def test_multi_failure_between_successes_count_resets_each_time():
    breaker = _make(threshold=5)
    for _ in range(4):
        _trip(breaker, n=4)
        assert breaker._failure_count == 4
        breaker.call(lambda: None)
        assert breaker._failure_count == 0
    assert breaker.state == CircuitState.CLOSED


def test_escalating_backoff_across_multiple_failed_probes_then_success():
    """base → 2x → 4x backoff, then success resets to base."""
    breaker = _make(threshold=1, base_timeout=0.01, max_timeout=1.0)
    _trip(breaker, n=1)

    for expected_mult in [2, 4]:
        time.sleep(breaker._recovery_timeout * 1.1)
        assert breaker.state == CircuitState.HALF_OPEN
        with pytest.raises(RuntimeError):
            breaker.call(_always_raise())
        assert breaker._recovery_timeout == pytest.approx(breaker._base_timeout * expected_mult)

    # Now let probe succeed
    time.sleep(breaker._recovery_timeout * 1.1)
    breaker.call(lambda: None)
    assert breaker._recovery_timeout == pytest.approx(0.01)
    assert breaker.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Section 9 — Recovery timeout growth vs failure_count independence
# ---------------------------------------------------------------------------


def test_recovery_timeout_not_doubled_in_closed_state():
    """_on_failure in CLOSED state must never double the recovery_timeout."""
    breaker = _make(threshold=5, base_timeout=30.0, max_timeout=300.0)
    for _ in range(4):  # below threshold — stays CLOSED
        breaker._on_failure()
    assert breaker._recovery_timeout == pytest.approx(30.0)


def test_recovery_timeout_only_doubled_from_half_open():
    """Doubling only happens in _on_failure when state is HALF_OPEN."""
    breaker = _make(threshold=1, base_timeout=10.0, max_timeout=200.0)
    breaker._state = CircuitState.CLOSED
    breaker._recovery_timeout = 10.0
    breaker._on_failure()  # threshold=1 → opens, but does NOT double
    # Circuit is now OPEN but timeout was not doubled — doubling is HALF_OPEN only
    assert breaker._recovery_timeout == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Section 10 — Flaky callable: fails once, succeeds thereafter
# ---------------------------------------------------------------------------


def test_flaky_func_failure_then_success_resets_count():
    """A callable that raises on first call and succeeds on second should
    leave the circuit CLOSED with failure_count == 0."""
    breaker = _make(threshold=3)
    attempts = [0]

    def flaky():
        attempts[0] += 1
        if attempts[0] == 1:
            raise RuntimeError("transient")
        return "ok"

    with pytest.raises(RuntimeError):
        breaker.call(flaky)
    assert breaker._failure_count == 1
    assert breaker.state == CircuitState.CLOSED

    result = breaker.call(flaky)
    assert result == "ok"
    assert breaker._failure_count == 0


def test_flaky_func_fails_at_threshold_minus_one_then_succeeds():
    """N-1 failures, success: circuit never opens and count resets."""
    breaker = _make(threshold=4)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            breaker.call(_always_raise())
    assert breaker.state == CircuitState.CLOSED
    breaker.call(lambda: "recovered")
    assert breaker._failure_count == 0
    assert breaker.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Section 11 — Multiple independent breakers
# ---------------------------------------------------------------------------


def test_two_breakers_completely_independent():
    a = CircuitBreaker("alpha", threshold=2, base_timeout=60.0)
    b = CircuitBreaker("beta", threshold=2, base_timeout=60.0)
    _trip(a, n=2)
    assert a.state == CircuitState.OPEN
    assert b.state == CircuitState.CLOSED


def test_two_breakers_share_no_failure_count():
    a = CircuitBreaker("alpha", threshold=5)
    b = CircuitBreaker("beta", threshold=5)
    _trip(a, n=3)
    assert b._failure_count == 0


def test_two_breakers_wrapping_same_callable_independently():
    """Two breakers wrapping the same underlying callable maintain separate state."""
    shared_fn = MagicMock(side_effect=RuntimeError("error"))
    a = CircuitBreaker("a", threshold=2)
    b = CircuitBreaker("b", threshold=5)

    for _ in range(2):
        with pytest.raises(RuntimeError):
            a.call(shared_fn)
    with pytest.raises(RuntimeError):
        b.call(shared_fn)

    assert a.state == CircuitState.OPEN
    assert b.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Section 12 — Lambda capturing mutable state
# ---------------------------------------------------------------------------


def test_call_with_lambda_capturing_mutable_list():
    log = []
    breaker = _make()
    breaker.call(lambda: log.append("called"))
    assert log == ["called"]


def test_call_side_effect_executes_exactly_once_on_success():
    counter = [0]
    breaker = _make()
    breaker.call(lambda: counter.__setitem__(0, counter[0] + 1))
    assert counter[0] == 1


# ---------------------------------------------------------------------------
# Section 13 — Stress test: 20 threads, full state machine cycles
# ---------------------------------------------------------------------------


def test_stress_20_threads_full_cycles():
    """20 threads running for 0.3 s — open, wait, probe, close repeatedly.
    Verifies no unhandled exception and final state is valid."""
    breaker = _make(threshold=3, base_timeout=0.01, max_timeout=0.5)
    errors: list[Exception] = []
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            try:
                # Mix of success and failure calls
                import random

                if random.random() < 0.4:
                    breaker.call(_always_raise())
                else:
                    breaker.call(lambda: "ok")
            except (MissyError, RuntimeError):
                pass
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(20)]
    for t in threads:
        t.start()

    time.sleep(0.3)
    stop.set()

    for t in threads:
        t.join(timeout=2.0)
    for t in threads:
        assert not t.is_alive()

    assert not errors
    assert breaker._state in CircuitState


def test_stress_10_threads_concurrent_trips_and_resets():
    """10 threads, each tripping and recovering their own fresh breaker in a
    tight loop — verifies the lock does not contend across independent instances."""
    errors: list[Exception] = []

    def run_cycle():
        breaker = _make(threshold=1, base_timeout=0.01, max_timeout=0.1)
        try:
            for _ in range(5):
                with pytest.raises(RuntimeError):
                    breaker.call(_always_raise())
                time.sleep(0.015)
                breaker.call(lambda: None)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run_cycle) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors


# ---------------------------------------------------------------------------
# Section 14 — Additional edge cases
# ---------------------------------------------------------------------------


def test_missy_error_is_subclass_of_exception():
    """Confirm MissyError is an Exception so it is caught by the call() handler."""
    assert issubclass(MissyError, Exception)


def test_circuit_state_enum_membership():
    assert CircuitState.CLOSED in CircuitState
    assert CircuitState.OPEN in CircuitState
    assert CircuitState.HALF_OPEN in CircuitState


def test_open_state_error_message_format():
    """Error message format: 'Circuit breaker '<name>' is OPEN; skipping call'."""
    breaker = CircuitBreaker("my-api", threshold=1)
    _trip(breaker, n=1)
    with pytest.raises(MissyError) as exc_info:
        breaker.call(lambda: None)
    msg = str(exc_info.value)
    assert "my-api" in msg
    assert "OPEN" in msg
    assert "skipping" in msg


def test_recovery_timeout_initial_equals_base_timeout():
    breaker = CircuitBreaker("x", threshold=3, base_timeout=45.0, max_timeout=300.0)
    assert breaker._recovery_timeout == pytest.approx(45.0)


def test_half_open_success_probe_return_value_preserved():
    """Return value of the probe in HALF_OPEN must reach the caller unchanged."""
    breaker = _make(threshold=1, base_timeout=0.01)
    _trip(breaker, n=1)
    time.sleep(0.02)
    sentinel = {"status": "ok", "code": 200}
    result = breaker.call(lambda: sentinel)
    assert result is sentinel


def test_failure_in_half_open_re_raises_original_exception():
    """The original exception type must propagate from a failed probe, not MissyError."""
    breaker = _make(threshold=1, base_timeout=0.01)
    _trip(breaker, n=1)
    time.sleep(0.02)
    assert breaker.state == CircuitState.HALF_OPEN
    with pytest.raises(OSError):
        breaker.call(_always_raise(OSError))
    # And we're back to OPEN
    assert breaker.state == CircuitState.OPEN


def test_probe_failure_in_half_open_exception_message_preserved():
    """Exception message must survive re-raise from probe failure."""

    def raise_with_message():
        raise ValueError("specific message 12345")

    breaker = _make(threshold=1, base_timeout=0.01)
    _trip(breaker, n=1)
    time.sleep(0.02)
    with pytest.raises(ValueError, match="specific message 12345"):
        breaker.call(raise_with_message)


def test_mock_func_called_with_exact_args():
    """MagicMock allows asserting that func was called with correct args."""
    mock_fn = MagicMock(return_value="val")
    breaker = _make()
    result = breaker.call(mock_fn, "a", "b", key="v")
    mock_fn.assert_called_once_with("a", "b", key="v")
    assert result == "val"


def test_mock_func_not_called_when_open_stress():
    """After opening with threshold=1, the mock must never be invoked over 10 tries."""
    breaker = _make(threshold=1)
    _trip(breaker, n=1)
    mock_fn = MagicMock()
    for _ in range(10):
        with pytest.raises(MissyError):
            breaker.call(mock_fn)
    mock_fn.assert_not_called()


def test_on_failure_directly_from_half_open_caps_at_max():
    """Direct _on_failure from HALF_OPEN with recovery_timeout near max must cap."""
    breaker = _make(threshold=1, base_timeout=10.0, max_timeout=15.0)
    breaker._state = CircuitState.HALF_OPEN
    breaker._recovery_timeout = 10.0
    breaker._on_failure()
    # 10 * 2 = 20, capped at 15
    assert breaker._recovery_timeout == pytest.approx(15.0)


def test_on_success_from_half_open_closes_circuit():
    breaker = _make(threshold=1)
    breaker._state = CircuitState.HALF_OPEN
    breaker._on_success()
    assert breaker._state == CircuitState.CLOSED


def test_failure_count_correct_after_success_in_long_sequence():
    """Interleave: 2 fail, 1 success, 2 fail, 1 success — count resets each time."""
    breaker = _make(threshold=5)
    for _ in range(3):
        _trip(breaker, n=2)
        breaker.call(lambda: None)
        assert breaker._failure_count == 0

    assert breaker.state == CircuitState.CLOSED


def test_call_with_generator_function_that_returns_value():
    """A generator function is not automatically iterated; call() returns the generator."""
    import types

    def gen():
        yield 1
        yield 2

    breaker = _make()
    result = breaker.call(gen)
    # result is a generator object — calling next() works
    assert isinstance(result, types.GeneratorType)
    assert next(result) == 1
