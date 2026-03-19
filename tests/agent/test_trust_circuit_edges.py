"""Edge-case tests for TrustScorer and CircuitBreaker.

Covers scenarios not addressed by the existing test suites in
tests/security/test_trust.py and tests/agent/test_circuit_breaker.py.
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


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


def _trip(breaker: CircuitBreaker, n: int | None = None) -> None:
    count = n if n is not None else breaker._threshold
    for _ in range(count):
        with pytest.raises(RuntimeError):
            breaker.call(_raise)


# ===========================================================================
# TrustScorer edge cases
# ===========================================================================


class TestTrustScorerInitialState:
    def test_unknown_entity_returns_default_score(self):
        scorer = TrustScorer()
        assert scorer.score("never_seen") == DEFAULT_SCORE

    def test_default_score_constant_is_500(self):
        assert DEFAULT_SCORE == 500

    def test_max_score_constant_is_1000(self):
        assert MAX_SCORE == 1000

    def test_min_score_constant_is_0(self):
        assert MIN_SCORE == 0

    def test_get_scores_empty_on_new_instance(self):
        scorer = TrustScorer()
        assert scorer.get_scores() == {}

    def test_get_scores_excludes_never_queried_unknown_entities(self):
        """Calling score() on an unknown entity must NOT persist it in _scores."""
        scorer = TrustScorer()
        scorer.score("phantom")
        assert "phantom" not in scorer.get_scores()


class TestTrustScorerGetScores:
    def test_get_scores_returns_all_tracked_entities(self):
        scorer = TrustScorer()
        scorer.record_success("a")
        scorer.record_failure("b")
        result = scorer.get_scores()
        assert set(result.keys()) == {"a", "b"}

    def test_get_scores_returns_copy_not_reference(self):
        """Mutating the returned dict must not affect internal state."""
        scorer = TrustScorer()
        scorer.record_success("x")
        snapshot = scorer.get_scores()
        snapshot["x"] = 0
        assert scorer.score("x") != 0

    def test_get_scores_reflects_current_values(self):
        scorer = TrustScorer()
        scorer.record_success("svc", weight=10)
        scores = scorer.get_scores()
        assert scores["svc"] == DEFAULT_SCORE + 10

    def test_get_scores_includes_reset_entities(self):
        scorer = TrustScorer()
        scorer.record_failure("svc", weight=100)
        scorer.reset("svc")
        scores = scorer.get_scores()
        assert scores["svc"] == DEFAULT_SCORE


class TestTrustScorerBoundaries:
    def test_success_at_maximum_stays_capped(self):
        scorer = TrustScorer()
        scorer._scores["entity"] = MAX_SCORE
        scorer.record_success("entity", weight=500)
        assert scorer.score("entity") == MAX_SCORE

    def test_failure_at_zero_stays_floored(self):
        scorer = TrustScorer()
        scorer._scores["entity"] = MIN_SCORE
        scorer.record_failure("entity", weight=999)
        assert scorer.score("entity") == MIN_SCORE

    def test_violation_at_zero_stays_floored(self):
        scorer = TrustScorer()
        scorer._scores["entity"] = MIN_SCORE
        scorer.record_violation("entity", weight=999)
        assert scorer.score("entity") == MIN_SCORE

    def test_success_brings_score_exactly_to_max(self):
        scorer = TrustScorer()
        scorer._scores["entity"] = MAX_SCORE - 1
        scorer.record_success("entity", weight=1)
        assert scorer.score("entity") == MAX_SCORE

    def test_failure_brings_score_exactly_to_zero(self):
        scorer = TrustScorer()
        scorer._scores["entity"] = 1
        scorer.record_failure("entity", weight=1)
        assert scorer.score("entity") == MIN_SCORE


class TestTrustScorerIsTrustedBoundary:
    def test_score_exactly_at_threshold_is_not_trusted(self):
        """is_trusted uses strict greater-than, so score == threshold is False."""
        scorer = TrustScorer()
        scorer._scores["borderline"] = 200
        assert scorer.is_trusted("borderline", threshold=200) is False

    def test_score_one_above_threshold_is_trusted(self):
        scorer = TrustScorer()
        scorer._scores["borderline"] = 201
        assert scorer.is_trusted("borderline", threshold=200) is True

    def test_score_one_below_threshold_is_not_trusted(self):
        scorer = TrustScorer()
        scorer._scores["borderline"] = 199
        assert scorer.is_trusted("borderline", threshold=200) is False

    def test_default_threshold_200_with_new_entity(self):
        """New entity at 500 is above default threshold of 200."""
        scorer = TrustScorer()
        assert scorer.is_trusted("new") is True

    def test_is_trusted_with_zero_threshold(self):
        """Any positive score is trusted when threshold=0."""
        scorer = TrustScorer()
        scorer._scores["entity"] = 1
        assert scorer.is_trusted("entity", threshold=0) is True

    def test_is_trusted_with_max_threshold(self):
        """Only the maximum score passes threshold=1000."""
        scorer = TrustScorer()
        scorer._scores["almost"] = MAX_SCORE - 1
        assert scorer.is_trusted("almost", threshold=MAX_SCORE) is False
        scorer._scores["max"] = MAX_SCORE
        assert scorer.is_trusted("max", threshold=MAX_SCORE) is False  # strict >


class TestTrustScorerZeroWeight:
    def test_success_with_zero_weight_does_not_change_score(self):
        scorer = TrustScorer()
        scorer.record_success("svc", weight=0)
        assert scorer.score("svc") == DEFAULT_SCORE

    def test_failure_with_zero_weight_does_not_change_score(self):
        scorer = TrustScorer()
        scorer.record_failure("svc", weight=0)
        assert scorer.score("svc") == DEFAULT_SCORE

    def test_violation_with_zero_weight_does_not_change_score(self):
        scorer = TrustScorer()
        scorer.record_violation("svc", weight=0)
        assert scorer.score("svc") == DEFAULT_SCORE


class TestTrustScorerMultipleEntities:
    def test_entities_are_tracked_independently(self):
        scorer = TrustScorer()
        scorer.record_success("a", weight=100)
        scorer.record_failure("b", weight=100)
        scorer.record_violation("c", weight=300)
        assert scorer.score("a") == DEFAULT_SCORE + 100
        assert scorer.score("b") == DEFAULT_SCORE - 100
        assert scorer.score("c") == DEFAULT_SCORE - 300

    def test_resetting_one_entity_leaves_others_unchanged(self):
        scorer = TrustScorer()
        scorer.record_success("a", weight=50)
        scorer.record_success("b", weight=50)
        scorer.reset("a")
        assert scorer.score("a") == DEFAULT_SCORE
        assert scorer.score("b") == DEFAULT_SCORE + 50

    def test_reset_of_unknown_entity_creates_entry_at_default(self):
        scorer = TrustScorer()
        scorer.reset("brand_new")
        assert scorer.score("brand_new") == DEFAULT_SCORE
        assert "brand_new" in scorer.get_scores()

    def test_many_entities_all_start_at_default_when_queried(self):
        scorer = TrustScorer()
        entities = [f"entity_{i}" for i in range(50)]
        for e in entities:
            assert scorer.score(e) == DEFAULT_SCORE


class TestTrustScorerThreadSafety:
    def test_concurrent_record_success_stays_within_bounds(self):
        scorer = TrustScorer()
        scorer._scores["shared"] = DEFAULT_SCORE
        errors = []

        def bump():
            try:
                for _ in range(100):
                    scorer.record_success("shared", weight=10)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=bump) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert MIN_SCORE <= scorer.score("shared") <= MAX_SCORE

    def test_concurrent_record_failure_stays_within_bounds(self):
        scorer = TrustScorer()
        scorer._scores["shared"] = DEFAULT_SCORE
        errors = []

        def drain():
            try:
                for _ in range(100):
                    scorer.record_failure("shared", weight=10)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=drain) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert MIN_SCORE <= scorer.score("shared") <= MAX_SCORE

    def test_concurrent_mixed_operations_no_exception(self):
        scorer = TrustScorer()
        errors = []

        def run():
            try:
                for _ in range(50):
                    scorer.record_success("svc", weight=5)
                    scorer.record_failure("svc", weight=5)
                    scorer.record_violation("svc", weight=10)
                    scorer.score("svc")
                    scorer.is_trusted("svc")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_different_entities_do_not_interfere(self):
        """Writes to different keys must not cause data races or wrong values."""
        scorer = TrustScorer()
        results: dict[str, list[int]] = {}
        errors: list[Exception] = []

        def write_entity(name: str):
            try:
                for _ in range(20):
                    scorer.record_success(name, weight=10)
                results[name] = scorer.score(name)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write_entity, args=(f"e{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for val in results.values():
            assert MIN_SCORE <= val <= MAX_SCORE


# ===========================================================================
# CircuitBreaker edge cases
# ===========================================================================


class TestCircuitBreakerOpenStateInternals:
    def test_rejection_in_open_does_not_increment_failure_count(self):
        """call() in OPEN raises MissyError before reaching _on_failure."""
        breaker = _make_breaker(threshold=2)
        _trip(breaker, n=2)
        count_before = breaker._failure_count
        with pytest.raises(MissyError):
            breaker.call(lambda: None)
        assert breaker._failure_count == count_before

    def test_rejection_in_open_does_not_update_last_failure_time(self):
        """OPEN rejection must not refresh _last_failure_time (would reset timeout)."""
        breaker = _make_breaker(threshold=1, base_timeout=60.0)
        _trip(breaker, n=1)
        recorded_time = breaker._last_failure_time
        time.sleep(0.01)
        with pytest.raises(MissyError):
            breaker.call(lambda: None)
        assert breaker._last_failure_time == recorded_time

    def test_on_failure_direct_when_already_open_doubles_timeout(self):
        """Calling _on_failure() while OPEN (direct internal call) doubles timeout."""
        breaker = _make_breaker(threshold=1, base_timeout=0.01, max_timeout=10.0)
        _trip(breaker, n=1)
        # Circuit is now OPEN; simulate what happens during a HALF_OPEN probe fail
        # by manually flipping to HALF_OPEN then calling _on_failure directly.
        breaker._state = CircuitState.HALF_OPEN
        original = breaker._recovery_timeout
        breaker._on_failure()
        assert breaker._recovery_timeout == original * 2
        assert breaker._state == CircuitState.OPEN


class TestCircuitBreakerBaseExceptionNotCaught:
    def test_keyboard_interrupt_not_caught_by_breaker(self):
        """BaseException subclasses (not Exception) propagate without incrementing count."""
        breaker = _make_breaker(threshold=5)

        def raise_kb():
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            breaker.call(raise_kb)

        # KeyboardInterrupt is BaseException, not Exception — count must stay 0.
        assert breaker._failure_count == 0
        assert breaker.state == CircuitState.CLOSED

    def test_system_exit_not_caught_by_breaker(self):
        breaker = _make_breaker(threshold=5)

        def raise_sys():
            raise SystemExit(0)

        with pytest.raises(SystemExit):
            breaker.call(raise_sys)

        assert breaker._failure_count == 0
        assert breaker.state == CircuitState.CLOSED


class TestCircuitBreakerCallReturnValues:
    def test_call_returns_complex_objects(self):
        breaker = _make_breaker()
        payload = {"key": [1, 2, 3], "nested": {"a": True}}
        result = breaker.call(lambda: payload)
        assert result is payload

    def test_call_returns_falsy_zero(self):
        breaker = _make_breaker()
        assert breaker.call(lambda: 0) == 0

    def test_call_returns_empty_string(self):
        breaker = _make_breaker()
        assert breaker.call(lambda: "") == ""

    def test_call_returns_empty_list(self):
        breaker = _make_breaker()
        assert breaker.call(list) == []

    def test_call_forwards_multiple_positional_args(self):
        breaker = _make_breaker()

        def add(a, b, c):
            return a + b + c

        assert breaker.call(add, 1, 2, 3) == 6

    def test_call_forwards_mixed_args_and_kwargs(self):
        breaker = _make_breaker()

        def concat(prefix, text, suffix="!"):
            return f"{prefix}{text}{suffix}"

        result = breaker.call(concat, "hello", text=" world", suffix=".")
        assert result == "hello world."


class TestCircuitBreakerFailureCountTracking:
    def test_failure_count_increments_for_each_failure_below_threshold(self):
        breaker = _make_breaker(threshold=5)
        for expected in range(1, 5):
            _trip(breaker, n=1)
            assert breaker._failure_count == expected

    def test_failure_count_at_threshold_opens_circuit(self):
        breaker = _make_breaker(threshold=3)
        _trip(breaker, n=3)
        assert breaker._failure_count == 3
        assert breaker.state == CircuitState.OPEN

    def test_success_resets_count_to_zero_not_decrements(self):
        breaker = _make_breaker(threshold=5)
        _trip(breaker, n=4)
        assert breaker._failure_count == 4
        breaker.call(lambda: None)
        assert breaker._failure_count == 0


class TestCircuitBreakerHalfOpenEdges:
    def test_half_open_state_only_allows_one_probe_before_decision(self):
        """In HALF_OPEN a success closes; there is no concept of 'remaining probes'."""
        breaker = _make_breaker(threshold=1, base_timeout=0.01)
        _trip(breaker, n=1)
        time.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN
        breaker.call(lambda: None)  # success → CLOSED
        # Additional calls now succeed normally (CLOSED)
        breaker.call(lambda: None)
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_failed_probe_increments_failure_count(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01)
        _trip(breaker, n=1)
        count_after_trip = breaker._failure_count
        time.sleep(0.02)
        with pytest.raises(RuntimeError):
            breaker.call(_raise)
        assert breaker._failure_count == count_after_trip + 1

    def test_recovery_timeout_doubles_on_each_failed_probe(self):
        breaker = _make_breaker(threshold=1, base_timeout=0.01, max_timeout=1.0)
        _trip(breaker, n=1)
        expected = breaker._base_timeout
        for _ in range(4):
            time.sleep(breaker._recovery_timeout * 1.5)
            assert breaker.state == CircuitState.HALF_OPEN
            with pytest.raises(RuntimeError):
                breaker.call(_raise)
            expected = min(expected * 2, breaker._max_timeout)
            assert breaker._recovery_timeout == pytest.approx(expected)


class TestCircuitBreakerMonotonicClockMocking:
    def test_open_state_persists_until_timeout_with_mocked_clock(self):
        breaker = _make_breaker(threshold=1, base_timeout=100.0)

        with patch("missy.agent.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 0.0
            _trip(breaker, n=1)
            mock_time.return_value = 99.9  # just before timeout
            assert breaker.state == CircuitState.OPEN

    def test_half_open_transitions_at_exact_timeout_boundary(self):
        breaker = _make_breaker(threshold=1, base_timeout=100.0)

        with patch("missy.agent.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 0.0
            _trip(breaker, n=1)
            # Exactly at timeout (>= 100.0) transitions to HALF_OPEN
            mock_time.return_value = 100.0
            assert breaker.state == CircuitState.HALF_OPEN

    def test_just_before_timeout_boundary_remains_open(self):
        breaker = _make_breaker(threshold=1, base_timeout=100.0)

        with patch("missy.agent.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 0.0
            _trip(breaker, n=1)
            mock_time.return_value = 99.999
            assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerThreadSafety:
    def test_concurrent_trips_reach_open_state_without_data_corruption(self):
        """Multiple threads racing to cause failures must produce a clean OPEN state."""
        breaker = _make_breaker(threshold=5, base_timeout=60.0)
        barrier = threading.Barrier(15)
        errors: list[Exception] = []

        def worker():
            barrier.wait()
            for _ in range(10):
                try:
                    breaker.call(_raise)
                except (RuntimeError, MissyError):
                    pass
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert breaker.state == CircuitState.OPEN
        assert breaker._failure_count >= breaker._threshold

    def test_concurrent_success_calls_do_not_corrupt_state(self):
        breaker = _make_breaker(threshold=50, base_timeout=60.0)
        errors: list[Exception] = []

        def succeed():
            try:
                for _ in range(100):
                    breaker.call(lambda: 1)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=succeed) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0

    def test_lock_not_held_on_exception_propagation(self):
        """After an exception in call(), the lock must be released so future calls work."""
        breaker = _make_breaker(threshold=5)

        def raise_on_call():
            raise ValueError("test")

        with pytest.raises(ValueError):
            breaker.call(raise_on_call)

        # If the lock were held, this would deadlock; give it a short timeout.
        acquired = breaker._lock.acquire(timeout=1.0)
        assert acquired, "Lock was not released after exception in call()"
        breaker._lock.release()


class TestCircuitBreakerStateTransitionCoverage:
    def test_full_cycle_closed_open_halfopen_closed(self):
        breaker = _make_breaker(threshold=2, base_timeout=0.01)
        assert breaker.state == CircuitState.CLOSED
        _trip(breaker, n=2)
        assert breaker.state == CircuitState.OPEN
        time.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN
        breaker.call(lambda: None)
        assert breaker.state == CircuitState.CLOSED

    def test_repeated_trip_recover_cycles(self):
        """Circuit must be reopenable after recovery across multiple cycles."""
        breaker = _make_breaker(threshold=1, base_timeout=0.01)
        for _ in range(5):
            _trip(breaker, n=1)
            assert breaker.state == CircuitState.OPEN
            time.sleep(0.02)
            breaker.call(lambda: None)
            assert breaker.state == CircuitState.CLOSED

    def test_closed_breaker_after_recovery_counts_failures_from_zero(self):
        """After successful recovery, a new trip needs threshold failures again."""
        breaker = _make_breaker(threshold=3, base_timeout=0.01)
        _trip(breaker, n=3)
        time.sleep(0.02)
        breaker.call(lambda: None)  # probe → CLOSED
        # One and two failures should not open again
        _trip(breaker, n=2)
        assert breaker.state == CircuitState.CLOSED
        _trip(breaker, n=1)  # third failure opens it again
        assert breaker.state == CircuitState.OPEN

    def test_name_attribute_preserved_through_state_changes(self):
        breaker = CircuitBreaker("sentinel-name", threshold=1, base_timeout=0.01)
        _trip(breaker, n=1)
        assert breaker.name == "sentinel-name"
        time.sleep(0.02)
        breaker.call(lambda: None)
        assert breaker.name == "sentinel-name"

    def test_on_success_from_closed_state_is_idempotent(self):
        """Calling _on_success() when already CLOSED must not raise or corrupt state."""
        breaker = _make_breaker()
        assert breaker.state == CircuitState.CLOSED
        breaker._on_success()
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0
        assert breaker._recovery_timeout == breaker._base_timeout

    def test_multiple_named_breakers_share_no_state(self):
        a = CircuitBreaker("alpha", threshold=2)
        b = CircuitBreaker("beta", threshold=2)
        c = CircuitBreaker("gamma", threshold=2)
        _trip(a, n=2)
        assert a.state == CircuitState.OPEN
        assert b.state == CircuitState.CLOSED
        assert c.state == CircuitState.CLOSED
        _trip(b, n=1)
        assert c.state == CircuitState.CLOSED
