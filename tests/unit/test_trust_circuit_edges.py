"""Edge-case tests for TrustScorer and CircuitBreaker.

These tests deliberately target gaps not covered by the primary test files:
  - tests/security/test_trust.py
  - tests/agent/test_circuit_breaker.py

Coverage targets
----------------
TrustScorer:
  - get_scores() returns a copy (mutation-safe)
  - is_trusted() uses strict greater-than, so score == threshold is NOT trusted
  - Default weights for all three record_* methods
  - Zero-weight calls are no-ops
  - Multiple entities are tracked independently in one instance
  - reset() on a previously-unseen entity forces an explicit 500 entry
  - Cumulative success from the floor (0) boundary
  - Cumulative failure from the ceiling (1000) boundary
  - Success/failure interleaving preserves correctness per entity

CircuitBreaker:
  - BaseException subclasses (KeyboardInterrupt, SystemExit) bypass the
    except-Exception clause and are NOT counted as failures
  - _on_failure() called directly while CLOSED but below threshold does not
    open the circuit (only the call() path gates on >= threshold)
  - Failure count increments one-by-one up to threshold-1 then opens on N
  - _recovery_timeout is NOT changed when CLOSED → OPEN transition occurs
    (it is only mutated on a failed half-open probe)
  - call() in HALF_OPEN state (state property drives transition) actually
    executes the callable — the function IS invoked
  - Rejected OPEN calls do not modify _last_failure_time
  - Multiple independent CircuitBreaker instances share no state
  - A zero-argument callable works correctly (no args/kwargs needed)
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from missy.agent.circuit_breaker import CircuitBreaker, CircuitState
from missy.core.exceptions import MissyError
from missy.security.trust import DEFAULT_SCORE, MAX_SCORE, MIN_SCORE, TrustScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_breaker(
    name: str = "test",
    threshold: int = 3,
    base_timeout: float = 60.0,
    max_timeout: float = 300.0,
) -> CircuitBreaker:
    return CircuitBreaker(name, threshold=threshold, base_timeout=base_timeout, max_timeout=max_timeout)


def _trip(breaker: CircuitBreaker, n: int) -> None:
    """Drive *n* failures through the breaker (all raise RuntimeError)."""
    for _ in range(n):
        with pytest.raises(RuntimeError):
            breaker.call(_raise)


def _raise(*_args, **_kwargs):
    raise RuntimeError("boom")


# ===========================================================================
# TrustScorer edge cases
# ===========================================================================


class TestTrustScorerGetScores:
    """Tests for get_scores() — not covered by the primary test file."""

    def test_get_scores_empty_on_new_instance(self):
        """A brand-new scorer has no recorded entities."""
        scorer = TrustScorer()
        assert scorer.get_scores() == {}

    def test_get_scores_returns_recorded_entities(self):
        scorer = TrustScorer()
        scorer.record_success("tool_a")
        scorer.record_failure("tool_b")
        scores = scorer.get_scores()
        assert "tool_a" in scores
        assert "tool_b" in scores

    def test_get_scores_returns_copy_not_reference(self):
        """Mutating the returned dict must not affect internal state."""
        scorer = TrustScorer()
        scorer.record_success("tool_a")
        snapshot = scorer.get_scores()
        snapshot["tool_a"] = 0  # mutate the copy
        assert scorer.score("tool_a") == 510  # internal state unchanged

    def test_get_scores_values_match_individual_score_calls(self):
        scorer = TrustScorer()
        scorer.record_success("x", weight=25)
        scorer.record_failure("y", weight=75)
        scores = scorer.get_scores()
        assert scores["x"] == scorer.score("x")
        assert scores["y"] == scorer.score("y")

    def test_get_scores_does_not_include_unrecorded_entities(self):
        """score() for unseen entities returns DEFAULT_SCORE but does not persist."""
        scorer = TrustScorer()
        _ = scorer.score("ghost")  # read without writing
        assert "ghost" not in scorer.get_scores()


class TestTrustScorerIsTrustedBoundary:
    """is_trusted uses strict >, so score == threshold must return False."""

    def test_score_equal_to_threshold_is_not_trusted(self):
        scorer = TrustScorer()
        # Drop from 500 to exactly 200
        scorer.record_failure("e", weight=300)
        assert scorer.score("e") == 200
        assert scorer.is_trusted("e", threshold=200) is False

    def test_score_one_above_threshold_is_trusted(self):
        scorer = TrustScorer()
        scorer.record_failure("e", weight=299)
        assert scorer.score("e") == 201
        assert scorer.is_trusted("e", threshold=200) is True

    def test_score_one_below_threshold_is_not_trusted(self):
        scorer = TrustScorer()
        scorer.record_failure("e", weight=301)
        assert scorer.score("e") == 199
        assert scorer.is_trusted("e", threshold=200) is False

    def test_default_score_trusted_with_default_threshold(self):
        """Fresh entity at 500 is trusted with default threshold of 200."""
        scorer = TrustScorer()
        assert scorer.is_trusted("new") is True

    def test_score_at_zero_is_never_trusted(self):
        scorer = TrustScorer()
        scorer.record_violation("e", weight=1000)
        assert scorer.score("e") == MIN_SCORE
        assert scorer.is_trusted("e", threshold=0) is False

    def test_score_at_max_is_always_trusted(self):
        scorer = TrustScorer()
        for _ in range(60):
            scorer.record_success("e", weight=10)
        assert scorer.score("e") == MAX_SCORE
        assert scorer.is_trusted("e", threshold=999) is True


class TestTrustScorerDefaultWeights:
    """Verify each record_* method uses the correct default weight."""

    def test_record_success_default_weight_is_10(self):
        scorer = TrustScorer()
        scorer.record_success("e")  # no weight argument
        assert scorer.score("e") == DEFAULT_SCORE + 10

    def test_record_failure_default_weight_is_50(self):
        scorer = TrustScorer()
        scorer.record_failure("e")
        assert scorer.score("e") == DEFAULT_SCORE - 50

    def test_record_violation_default_weight_is_200(self):
        scorer = TrustScorer()
        scorer.record_violation("e")
        assert scorer.score("e") == DEFAULT_SCORE - 200


class TestTrustScorerZeroWeight:
    """Zero-weight calls must be no-ops for all three record_* methods."""

    def test_success_weight_zero_does_not_change_score(self):
        scorer = TrustScorer()
        scorer.record_success("e", weight=0)
        # After a zero-weight success the entity is written into _scores at 500
        assert scorer.score("e") == DEFAULT_SCORE

    def test_failure_weight_zero_does_not_change_score(self):
        scorer = TrustScorer()
        scorer.record_failure("e", weight=0)
        assert scorer.score("e") == DEFAULT_SCORE

    def test_violation_weight_zero_does_not_change_score(self):
        scorer = TrustScorer()
        scorer.record_violation("e", weight=0)
        assert scorer.score("e") == DEFAULT_SCORE


class TestTrustScorerMultipleEntities:
    """Multiple entities are tracked independently in one TrustScorer instance."""

    def test_two_entities_have_independent_scores(self):
        scorer = TrustScorer()
        scorer.record_success("good", weight=100)
        scorer.record_failure("bad", weight=100)
        assert scorer.score("good") == 600
        assert scorer.score("bad") == 400

    def test_violation_on_one_does_not_affect_other(self):
        scorer = TrustScorer()
        scorer.record_violation("rogue")
        assert scorer.score("innocent") == DEFAULT_SCORE

    def test_reset_on_one_entity_leaves_others_unchanged(self):
        scorer = TrustScorer()
        scorer.record_success("a", weight=50)
        scorer.record_success("b", weight=50)
        scorer.reset("a")
        assert scorer.score("a") == DEFAULT_SCORE
        assert scorer.score("b") == DEFAULT_SCORE + 50

    def test_many_entities_all_independent(self):
        scorer = TrustScorer()
        entities = [f"entity_{i}" for i in range(10)]
        for i, eid in enumerate(entities):
            for _ in range(i):
                scorer.record_success(eid, weight=10)
        for i, eid in enumerate(entities):
            assert scorer.score(eid) == min(DEFAULT_SCORE + i * 10, MAX_SCORE)


class TestTrustScorerResetEdges:
    """reset() edge cases."""

    def test_reset_unseen_entity_creates_explicit_entry(self):
        """reset() on an entity never touched before writes it into _scores."""
        scorer = TrustScorer()
        scorer.reset("ghost")
        # The entity must now be explicitly stored (not just a default fallback)
        assert "ghost" in scorer.get_scores()
        assert scorer.score("ghost") == DEFAULT_SCORE

    def test_reset_after_max_score_returns_to_default(self):
        scorer = TrustScorer()
        for _ in range(100):
            scorer.record_success("e", weight=100)
        assert scorer.score("e") == MAX_SCORE
        scorer.reset("e")
        assert scorer.score("e") == DEFAULT_SCORE

    def test_reset_is_idempotent(self):
        scorer = TrustScorer()
        scorer.record_failure("e", weight=200)
        scorer.reset("e")
        scorer.reset("e")
        assert scorer.score("e") == DEFAULT_SCORE


class TestTrustScorerBoundaryArithmetic:
    """Clamping at both ends of the 0–1000 range."""

    def test_success_from_zero_increments_correctly(self):
        """Once floored at 0, a success should bring the score to weight value."""
        scorer = TrustScorer()
        scorer.record_violation("e", weight=1000)  # clamp to 0
        assert scorer.score("e") == MIN_SCORE
        scorer.record_success("e", weight=10)
        assert scorer.score("e") == 10

    def test_failure_from_max_decrements_correctly(self):
        """Once capped at 1000, a failure should subtract normally."""
        scorer = TrustScorer()
        for _ in range(100):
            scorer.record_success("e", weight=100)
        assert scorer.score("e") == MAX_SCORE
        scorer.record_failure("e", weight=50)
        assert scorer.score("e") == MAX_SCORE - 50

    def test_success_at_max_stays_at_max(self):
        scorer = TrustScorer()
        for _ in range(100):
            scorer.record_success("e", weight=100)
        scorer.record_success("e", weight=999)
        assert scorer.score("e") == MAX_SCORE

    def test_failure_at_zero_stays_at_zero(self):
        scorer = TrustScorer()
        scorer.record_violation("e", weight=1000)
        scorer.record_failure("e", weight=999)
        assert scorer.score("e") == MIN_SCORE

    def test_violation_from_low_score_clamps_at_zero(self):
        scorer = TrustScorer()
        scorer.record_failure("e", weight=490)  # score → 10
        assert scorer.score("e") == 10
        scorer.record_violation("e", weight=200)  # 10 - 200 → clamp to 0
        assert scorer.score("e") == MIN_SCORE

    def test_interleaved_ops_maintain_correct_running_total(self):
        scorer = TrustScorer()
        scorer.record_success("e", weight=100)   # 500 + 100 = 600
        scorer.record_failure("e", weight=50)    # 600 - 50  = 550
        scorer.record_violation("e", weight=200) # 550 - 200 = 350
        scorer.record_success("e", weight=10)    # 350 + 10  = 360
        assert scorer.score("e") == 360


# ===========================================================================
# CircuitBreaker edge cases
# ===========================================================================


class TestCircuitBreakerBaseException:
    """BaseException subclasses must not be caught by the except-Exception clause."""

    def test_keyboard_interrupt_is_not_caught(self):
        """KeyboardInterrupt (BaseException, not Exception) must propagate unchanged
        and must NOT increment the failure count."""
        breaker = _make_breaker(threshold=5)

        def raise_keyboard():
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            breaker.call(raise_keyboard)

        # The failure was not recorded — count stays at 0
        assert breaker._failure_count == 0

    def test_system_exit_is_not_caught(self):
        """SystemExit must propagate without being counted as a failure."""
        breaker = _make_breaker(threshold=5)

        def raise_sys_exit():
            raise SystemExit(1)

        with pytest.raises(SystemExit):
            breaker.call(raise_sys_exit)

        assert breaker._failure_count == 0

    def test_base_exception_does_not_open_circuit(self):
        """Repeated BaseException raises must not open the circuit."""
        breaker = _make_breaker(threshold=2)

        for _ in range(5):
            with pytest.raises(KeyboardInterrupt):
                breaker.call(lambda: (_ for _ in ()).throw(KeyboardInterrupt()))

        assert breaker.state == CircuitState.CLOSED

    def test_exception_after_base_exception_is_still_counted(self):
        """A regular Exception after a BaseException still increments the count."""
        breaker = _make_breaker(threshold=5)

        with pytest.raises(KeyboardInterrupt):
            breaker.call(lambda: (_ for _ in ()).throw(KeyboardInterrupt()))

        with pytest.raises(RuntimeError):
            breaker.call(_raise)

        assert breaker._failure_count == 1


class TestCircuitBreakerFailureCountGranularity:
    """Failure count increments are exactly 1 per call, threshold opens at N."""

    def test_failure_count_increments_one_at_a_time(self):
        breaker = _make_breaker(threshold=5)
        for expected_count in range(1, 5):
            _trip(breaker, n=1)
            assert breaker._failure_count == expected_count
            assert breaker.state == CircuitState.CLOSED

    def test_exactly_at_threshold_opens(self):
        breaker = _make_breaker(threshold=4)
        _trip(breaker, n=3)
        assert breaker.state == CircuitState.CLOSED
        _trip(breaker, n=1)
        assert breaker.state == CircuitState.OPEN

    def test_failure_count_does_not_reset_on_open_rejection(self):
        """When the circuit is OPEN, rejected calls must not change _failure_count."""
        breaker = _make_breaker(threshold=2)
        _trip(breaker, n=2)
        count_when_opened = breaker._failure_count
        for _ in range(3):
            with pytest.raises(MissyError):
                breaker.call(lambda: None)
        assert breaker._failure_count == count_when_opened


class TestCircuitBreakerRecoveryTimeoutInvariant:
    """_recovery_timeout must only change on a failed half-open probe or success."""

    def test_closed_to_open_does_not_change_recovery_timeout(self):
        """Tripping the breaker CLOSED → OPEN leaves _recovery_timeout at base."""
        breaker = _make_breaker(threshold=2, base_timeout=60.0)
        _trip(breaker, n=2)
        assert breaker.state == CircuitState.OPEN
        assert breaker._recovery_timeout == 60.0

    def test_open_rejection_does_not_change_recovery_timeout(self):
        """Calls rejected while OPEN do not alter _recovery_timeout."""
        breaker = _make_breaker(threshold=1, base_timeout=60.0)
        _trip(breaker, n=1)
        with pytest.raises(MissyError):
            breaker.call(lambda: None)
        assert breaker._recovery_timeout == 60.0

    def test_open_rejection_does_not_change_last_failure_time(self):
        """A rejected OPEN call must not overwrite _last_failure_time."""
        breaker = _make_breaker(threshold=1, base_timeout=60.0)
        _trip(breaker, n=1)
        recorded_time = breaker._last_failure_time
        time.sleep(0.01)
        with pytest.raises(MissyError):
            breaker.call(lambda: None)
        assert breaker._last_failure_time == recorded_time


class TestCircuitBreakerHalfOpenCallExecuted:
    """In HALF_OPEN state the callable must actually be invoked."""

    def test_callable_is_invoked_in_half_open_state(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01)
        _trip(breaker, n=1)
        time.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN

        invoked = []

        def probe():
            invoked.append(True)
            return "done"

        result = breaker.call(probe)
        assert result == "done"
        assert invoked == [True]

    def test_half_open_success_clears_state_before_next_call(self):
        """After a successful probe, a brand-new failure run must start from 0."""
        breaker = _make_breaker(threshold=2, base_timeout=0.01)
        _trip(breaker, n=2)
        time.sleep(0.02)
        breaker.call(lambda: None)  # probe succeeds → CLOSED
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0
        # One failure below threshold should not re-open
        _trip(breaker, n=1)
        assert breaker.state == CircuitState.CLOSED


class TestCircuitBreakerIndependentInstances:
    """Two CircuitBreaker instances must share no state whatsoever."""

    def test_tripping_one_does_not_affect_other(self):
        a = _make_breaker(name="service-a", threshold=2)
        b = _make_breaker(name="service-b", threshold=2)
        _trip(a, n=2)
        assert a.state == CircuitState.OPEN
        assert b.state == CircuitState.CLOSED

    def test_recovery_timeout_changes_are_isolated(self):
        a = _make_breaker(name="a", threshold=1, base_timeout=0.01, max_timeout=1.0)
        b = _make_breaker(name="b", threshold=1, base_timeout=0.01, max_timeout=1.0)
        _trip(a, n=1)
        time.sleep(0.02)
        # Fail the probe on 'a' to double its recovery timeout
        with pytest.raises(RuntimeError):
            a.call(_raise)
        # 'b' must still have the base timeout
        assert b._recovery_timeout == 0.01
        assert a._recovery_timeout == pytest.approx(0.02)

    def test_same_name_instances_are_still_independent(self):
        """Name is just a label; same name does not mean shared state."""
        x = CircuitBreaker("shared-name", threshold=1)
        y = CircuitBreaker("shared-name", threshold=1)
        _trip(x, n=1)
        assert x.state == CircuitState.OPEN
        assert y.state == CircuitState.CLOSED


class TestCircuitBreakerZeroArgCallable:
    """Callables that take no arguments work correctly through call()."""

    def test_zero_arg_callable_succeeds(self):
        breaker = _make_breaker()
        result = breaker.call(lambda: "hello")
        assert result == "hello"

    def test_zero_arg_callable_failure_is_counted(self):
        breaker = _make_breaker(threshold=3)

        def boom():
            raise ValueError("no args needed")

        with pytest.raises(ValueError):
            breaker.call(boom)
        assert breaker._failure_count == 1


class TestCircuitBreakerDirectOnFailure:
    """_on_failure() called directly (bypassing call()) behaves correctly."""

    def test_direct_on_failure_below_threshold_does_not_open(self):
        """Directly calling _on_failure once on a threshold-3 breaker keeps it CLOSED."""
        breaker = _make_breaker(threshold=3)
        breaker._on_failure()
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 1

    def test_direct_on_failure_at_threshold_opens(self):
        breaker = _make_breaker(threshold=2)
        breaker._on_failure()
        breaker._on_failure()
        assert breaker.state == CircuitState.OPEN

    def test_direct_on_failure_from_half_open_doubles_timeout(self):
        """Directly invoking _on_failure while in HALF_OPEN state doubles the timeout."""
        breaker = _make_breaker(threshold=1, base_timeout=30.0, max_timeout=300.0)
        breaker._state = CircuitState.HALF_OPEN
        breaker._recovery_timeout = 30.0
        breaker._on_failure()
        assert breaker._recovery_timeout == pytest.approx(60.0)
        assert breaker._state == CircuitState.OPEN


class TestCircuitBreakerClockPatching:
    """Verify state transitions using a patched monotonic clock (no real sleeping)."""

    def test_open_to_half_open_via_patched_clock(self):
        """The full open→half_open transition driven purely through time.monotonic mock."""
        breaker = _make_breaker(threshold=1, base_timeout=100.0)

        with patch("missy.agent.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 5000.0
            _trip(breaker, n=1)
            assert breaker._state == CircuitState.OPEN

            mock_time.return_value = 5099.0  # 99 s elapsed — still open
            assert breaker.state == CircuitState.OPEN

            mock_time.return_value = 5101.0  # 101 s elapsed — timeout passed
            assert breaker.state == CircuitState.HALF_OPEN

    def test_failed_probe_backoff_sequence_via_patched_clock(self):
        """Verify 60 → 120 → 240 → 300 (max) backoff without sleeping."""
        breaker = CircuitBreaker("test", threshold=1, base_timeout=60.0, max_timeout=300.0)

        def advance_past_timeout_and_fail(mock_time, current_t):
            """Move the clock past recovery_timeout and fail the probe."""
            new_t = current_t + breaker._recovery_timeout + 1.0
            mock_time.return_value = new_t
            assert breaker.state == CircuitState.HALF_OPEN
            with pytest.raises(RuntimeError):
                breaker.call(_raise)
            assert breaker.state == CircuitState.OPEN
            return new_t

        with patch("missy.agent.circuit_breaker.time.monotonic") as mock_time:
            t = 0.0
            mock_time.return_value = t
            _trip(breaker, n=1)  # trip → OPEN, recovery=60

            t = advance_past_timeout_and_fail(mock_time, t)
            assert breaker._recovery_timeout == pytest.approx(120.0)

            t = advance_past_timeout_and_fail(mock_time, t)
            assert breaker._recovery_timeout == pytest.approx(240.0)

            t = advance_past_timeout_and_fail(mock_time, t)
            assert breaker._recovery_timeout == pytest.approx(300.0)  # capped

            t = advance_past_timeout_and_fail(mock_time, t)
            assert breaker._recovery_timeout == pytest.approx(300.0)  # stays at max


class TestCircuitBreakerConcurrentOpenRejection:
    """Threads racing to read state while the circuit is OPEN all get MissyError."""

    def test_all_threads_receive_missy_error_when_open(self):
        breaker = _make_breaker(threshold=1, base_timeout=60.0)
        _trip(breaker, n=1)

        errors: list[Exception] = []
        unexpected: list[Exception] = []
        barrier = threading.Barrier(8)

        def try_call():
            barrier.wait()
            try:
                breaker.call(lambda: None)
            except MissyError as e:
                errors.append(e)
            except Exception as e:
                unexpected.append(e)

        threads = [threading.Thread(target=try_call) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not unexpected, f"Unexpected exceptions: {unexpected}"
        assert len(errors) == 8, f"Expected 8 MissyError, got {len(errors)}"
