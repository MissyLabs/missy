"""Session 15 comprehensive tests for TrustScorer.

Covers areas not yet exercised by test_trust.py or test_session13_vault_trust.py:

Default / initialisation
------------------------
* score() for unknown entity returns exactly DEFAULT_SCORE (500)
* get_scores() on a fresh scorer returns an empty dict
* Two independent TrustScorer instances do not share state

record_success
--------------
* Default weight (+10) from DEFAULT baseline
* Custom weight increments correctly
* First call on unknown entity starts from 500 then adds weight
* Successive successes accumulate
* Single large weight clamps to MAX_SCORE without going over
* Many small successes clamp to MAX_SCORE

record_failure
--------------
* Default weight (−50) from DEFAULT baseline
* Custom weight decrements correctly
* First call on unknown entity starts from 500 then subtracts weight
* Successive failures accumulate
* Single large weight floors to MIN_SCORE without going negative
* Many small failures floor to MIN_SCORE

record_violation
----------------
* Default weight (−200) from DEFAULT baseline
* First call on unknown entity starts from 500 then subtracts weight
* Custom weight (e.g. weight=50) decrements by exactly that amount
* Violation on an already-low score floors at 0

Score sequencing
----------------
* Multiple successes followed by a violation
* Alternating success and failure converges within bounds
* Reset after a chain restores to DEFAULT_SCORE
* Negative weight on record_success behaves as a decrease (no exception)
* Very large positive weight on record_success clamps to MAX_SCORE
* Very large positive weight on record_failure floors to MIN_SCORE

is_trusted
----------
* Unknown entity with default threshold (200) returns True (500 > 200)
* Custom threshold above DEFAULT returns False
* Custom threshold below DEFAULT returns True
* Score equal to threshold returns False (strict >)
* Score one above threshold returns True
* After flooring at 0, threshold=0 returns False (0 is not > 0)
* After maxing at 1000, threshold=1000 returns False (1000 is not > 1000)

get_scores
----------
* Returns empty dict initially
* Reflects entities after writes
* Snapshot is a copy — mutating it leaves scorer intact
* Returns consistent snapshot under concurrent writes

reset
-----
* reset on a modified entity restores to DEFAULT_SCORE
* reset on an unknown entity sets it to DEFAULT_SCORE
* reset on an already-default entity leaves it at DEFAULT_SCORE
* reset does not affect other entities

Multiple entities
-----------------
* 100 independent entities all default to 500
* Success on entity A does not affect entity B
* Violation on entity B does not affect entity A

Thread safety
-------------
* 20 threads concurrently calling record_success — final score within bounds
* 20 threads concurrently calling record_failure — final score within bounds
* Concurrent reads via score() during concurrent writes raise no exceptions
* Stress: 20 threads recording both success and failure simultaneously
* Concurrent get_scores() calls during writes always return valid copies

Weight edge cases
-----------------
* weight=0 success is a no-op
* weight=0 failure is a no-op
* weight=0 violation is a no-op
* Very large weight (sys.maxsize) clamps correctly for success
* Very large weight (sys.maxsize) floors correctly for failure
"""

from __future__ import annotations

import sys
import threading

from missy.security.trust import DEFAULT_SCORE, MAX_SCORE, MIN_SCORE, TrustScorer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fresh() -> TrustScorer:
    """Return a new TrustScorer instance."""
    return TrustScorer()


# ===========================================================================
# Default / initialisation
# ===========================================================================


class TestTrustScorerInitialisation:
    """Verify the initial state of a freshly constructed TrustScorer."""

    def test_unknown_entity_returns_default_score(self):
        """Querying a never-seen entity returns exactly DEFAULT_SCORE (500)."""
        scorer = fresh()
        assert scorer.score("never_seen") == DEFAULT_SCORE

    def test_unknown_entity_default_score_equals_500(self):
        """The literal value returned for an unknown entity is 500."""
        scorer = fresh()
        assert scorer.score("x") == 500

    def test_get_scores_empty_initially(self):
        """A freshly constructed scorer has no tracked entities."""
        scorer = fresh()
        assert scorer.get_scores() == {}

    def test_two_instances_do_not_share_state(self):
        """Mutations on one scorer do not bleed into a separate instance."""
        s1 = TrustScorer()
        s2 = TrustScorer()
        s1.record_failure("shared_id", weight=200)
        assert s2.score("shared_id") == DEFAULT_SCORE


# ===========================================================================
# record_success
# ===========================================================================


class TestRecordSuccess:
    """Behaviour of record_success across a range of inputs."""

    def test_default_weight_increments_by_10(self):
        """record_success with no explicit weight adds exactly 10."""
        scorer = fresh()
        scorer.record_success("tool")
        assert scorer.score("tool") == DEFAULT_SCORE + 10

    def test_custom_weight_increments_correctly(self):
        """record_success with weight=25 adds exactly 25."""
        scorer = fresh()
        scorer.record_success("tool", weight=25)
        assert scorer.score("tool") == DEFAULT_SCORE + 25

    def test_first_call_on_unknown_entity_starts_from_default(self):
        """First record_success uses DEFAULT_SCORE as the base, not 0."""
        scorer = fresh()
        scorer.record_success("brand_new", weight=10)
        assert scorer.score("brand_new") == DEFAULT_SCORE + 10

    def test_successive_successes_accumulate(self):
        """Three consecutive successes of weight 10 add 30 total."""
        scorer = fresh()
        for _ in range(3):
            scorer.record_success("tool", weight=10)
        assert scorer.score("tool") == DEFAULT_SCORE + 30

    def test_single_large_weight_clamps_to_max(self):
        """A weight larger than MAX_SCORE - DEFAULT_SCORE clamps to MAX_SCORE."""
        scorer = fresh()
        scorer.record_success("tool", weight=MAX_SCORE)  # 500 + 1000 > 1000
        assert scorer.score("tool") == MAX_SCORE

    def test_many_small_successes_clamp_to_max(self):
        """Many small increments converge to exactly MAX_SCORE, not beyond."""
        scorer = fresh()
        for _ in range(200):
            scorer.record_success("tool", weight=10)
        assert scorer.score("tool") == MAX_SCORE

    def test_score_never_exceeds_max_after_extreme_weight(self):
        """weight=sys.maxsize must still produce MAX_SCORE, not overflow."""
        scorer = fresh()
        scorer.record_success("tool", weight=sys.maxsize)
        assert scorer.score("tool") == MAX_SCORE


# ===========================================================================
# record_failure
# ===========================================================================


class TestRecordFailure:
    """Behaviour of record_failure across a range of inputs."""

    def test_default_weight_decrements_by_50(self):
        """record_failure with no explicit weight subtracts exactly 50."""
        scorer = fresh()
        scorer.record_failure("tool")
        assert scorer.score("tool") == DEFAULT_SCORE - 50

    def test_custom_weight_decrements_correctly(self):
        """record_failure with weight=75 subtracts exactly 75."""
        scorer = fresh()
        scorer.record_failure("tool", weight=75)
        assert scorer.score("tool") == DEFAULT_SCORE - 75

    def test_first_call_on_unknown_entity_starts_from_default(self):
        """First record_failure uses DEFAULT_SCORE as the base, not 0."""
        scorer = fresh()
        scorer.record_failure("brand_new", weight=50)
        assert scorer.score("brand_new") == DEFAULT_SCORE - 50

    def test_successive_failures_accumulate(self):
        """Three consecutive failures of weight 50 subtract 150 total."""
        scorer = fresh()
        for _ in range(3):
            scorer.record_failure("tool", weight=50)
        assert scorer.score("tool") == DEFAULT_SCORE - 150

    def test_single_large_weight_floors_to_min(self):
        """A weight larger than DEFAULT_SCORE floors to MIN_SCORE."""
        scorer = fresh()
        scorer.record_failure("tool", weight=MAX_SCORE)  # 500 - 1000 < 0
        assert scorer.score("tool") == MIN_SCORE

    def test_many_small_failures_floor_to_min(self):
        """Many small decrements converge to exactly MIN_SCORE, not below."""
        scorer = fresh()
        for _ in range(200):
            scorer.record_failure("tool", weight=10)
        assert scorer.score("tool") == MIN_SCORE

    def test_score_never_below_min_after_extreme_weight(self):
        """weight=sys.maxsize must still produce MIN_SCORE, not underflow."""
        scorer = fresh()
        scorer.record_failure("tool", weight=sys.maxsize)
        assert scorer.score("tool") == MIN_SCORE


# ===========================================================================
# record_violation
# ===========================================================================


class TestRecordViolation:
    """Behaviour of record_violation across a range of inputs."""

    def test_default_weight_decrements_by_200(self):
        """record_violation with no explicit weight subtracts exactly 200."""
        scorer = fresh()
        scorer.record_violation("server")
        assert scorer.score("server") == DEFAULT_SCORE - 200

    def test_first_call_on_unknown_entity_starts_from_default(self):
        """First record_violation uses DEFAULT_SCORE as the base."""
        scorer = fresh()
        scorer.record_violation("brand_new")
        assert scorer.score("brand_new") == DEFAULT_SCORE - 200

    def test_custom_violation_weight_50(self):
        """record_violation(weight=50) subtracts exactly 50."""
        scorer = fresh()
        scorer.record_violation("server", weight=50)
        assert scorer.score("server") == DEFAULT_SCORE - 50

    def test_violation_on_low_score_floors_at_zero(self):
        """A violation on a score that is already near zero floors at MIN_SCORE."""
        scorer = fresh()
        # Bring score to 50 first.
        scorer.record_failure("server", weight=DEFAULT_SCORE - 50)
        assert scorer.score("server") == 50
        scorer.record_violation("server", weight=200)
        assert scorer.score("server") == MIN_SCORE


# ===========================================================================
# Score sequencing
# ===========================================================================


class TestScoreSequencing:
    """Tests that combine multiple operations in sequence."""

    def test_successes_then_violation(self):
        """Several successes followed by a violation produce the expected score."""
        scorer = fresh()
        for _ in range(5):
            scorer.record_success("tool", weight=20)  # 500 + 100 = 600
        scorer.record_violation("tool")  # 600 - 200 = 400
        assert scorer.score("tool") == 400

    def test_alternating_success_and_failure_stays_in_bounds(self):
        """Alternating success/failure of equal weight keeps score near DEFAULT."""
        scorer = fresh()
        for _ in range(30):
            scorer.record_success("tool", weight=10)
            scorer.record_failure("tool", weight=10)
        final = scorer.score("tool")
        assert MIN_SCORE <= final <= MAX_SCORE
        assert final == DEFAULT_SCORE

    def test_reset_after_chain_restores_default(self):
        """After arbitrary operations, reset() brings the score back to DEFAULT."""
        scorer = fresh()
        scorer.record_success("tool", weight=300)
        scorer.record_violation("tool")
        scorer.record_failure("tool", weight=100)
        scorer.reset("tool")
        assert scorer.score("tool") == DEFAULT_SCORE

    def test_negative_weight_on_success_decreases_score(self):
        """Passing a negative weight to record_success acts as a decrease."""
        scorer = fresh()
        scorer.record_success("tool", weight=-50)
        # min(500 + (-50), 1000) = 450; no exception should be raised
        assert scorer.score("tool") == DEFAULT_SCORE - 50

    def test_very_large_positive_weight_success_clamps(self):
        """weight=10_000_000 on record_success clamps to MAX_SCORE."""
        scorer = fresh()
        scorer.record_success("tool", weight=10_000_000)
        assert scorer.score("tool") == MAX_SCORE

    def test_very_large_positive_weight_failure_floors(self):
        """weight=10_000_000 on record_failure floors to MIN_SCORE."""
        scorer = fresh()
        scorer.record_failure("tool", weight=10_000_000)
        assert scorer.score("tool") == MIN_SCORE


# ===========================================================================
# is_trusted
# ===========================================================================


class TestIsTrusted:
    """Boundary behaviour of the is_trusted predicate."""

    def test_unknown_entity_with_default_threshold_is_trusted(self):
        """An unknown entity defaults to 500, which is > default threshold 200."""
        scorer = fresh()
        assert scorer.is_trusted("new") is True

    def test_custom_threshold_above_default_returns_false(self):
        """threshold=600 > DEFAULT_SCORE(500) → unknown entity not trusted."""
        scorer = fresh()
        assert scorer.is_trusted("new", threshold=600) is False

    def test_custom_threshold_below_default_returns_true(self):
        """threshold=100 < DEFAULT_SCORE(500) → unknown entity trusted."""
        scorer = fresh()
        assert scorer.is_trusted("new", threshold=100) is True

    def test_score_equal_to_threshold_returns_false(self):
        """is_trusted uses strict >, so score == threshold is False."""
        scorer = fresh()
        # Bring score to exactly 300.
        scorer.record_failure("e", weight=DEFAULT_SCORE - 300)
        assert scorer.score("e") == 300
        assert scorer.is_trusted("e", threshold=300) is False

    def test_score_one_above_threshold_returns_true(self):
        """Score exactly one point above threshold → True."""
        scorer = fresh()
        scorer.record_failure("e", weight=DEFAULT_SCORE - 301)
        assert scorer.score("e") == 301
        assert scorer.is_trusted("e", threshold=300) is True

    def test_score_zero_threshold_zero_returns_false(self):
        """is_trusted(threshold=0) is False when score is exactly 0."""
        scorer = fresh()
        scorer.record_failure("e", weight=MAX_SCORE)  # floor at 0
        assert scorer.score("e") == MIN_SCORE
        assert scorer.is_trusted("e", threshold=0) is False

    def test_score_max_threshold_max_returns_false(self):
        """is_trusted(threshold=1000) is always False — score cannot exceed 1000."""
        scorer = fresh()
        for _ in range(200):
            scorer.record_success("e", weight=100)
        assert scorer.score("e") == MAX_SCORE
        assert scorer.is_trusted("e", threshold=MAX_SCORE) is False


# ===========================================================================
# get_scores
# ===========================================================================


class TestGetScores:
    """Verify the snapshot semantics of get_scores()."""

    def test_get_scores_empty_on_fresh_scorer(self):
        """get_scores() returns {} before any entity has been modified."""
        scorer = fresh()
        assert scorer.get_scores() == {}

    def test_get_scores_reflects_recorded_entities(self):
        """get_scores() includes all entities that have been written to."""
        scorer = fresh()
        scorer.record_success("a", weight=10)
        scorer.record_failure("b", weight=20)
        snap = scorer.get_scores()
        assert snap == {"a": DEFAULT_SCORE + 10, "b": DEFAULT_SCORE - 20}

    def test_get_scores_snapshot_mutation_does_not_affect_scorer(self):
        """Mutating the returned dict does not change the scorer's internal state."""
        scorer = fresh()
        scorer.record_success("e", weight=10)
        snap = scorer.get_scores()
        snap["e"] = 0
        snap["injected"] = 999
        assert scorer.score("e") == DEFAULT_SCORE + 10
        assert "injected" not in scorer.get_scores()

    def test_get_scores_consistent_during_concurrent_writes(self):
        """Snapshots taken during concurrent writes are always valid dicts."""
        scorer = fresh()
        snapshots: list[dict] = []
        errors: list[Exception] = []

        def writer() -> None:
            for i in range(50):
                scorer.record_success(f"entity_{i}", weight=1)

        def reader() -> None:
            try:
                for _ in range(20):
                    snap = scorer.get_scores()
                    assert isinstance(snap, dict)
                    snapshots.append(snap)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(5)] + [
            threading.Thread(target=reader) for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Every snapshot must be a dict with int values in bounds.
        for snap in snapshots:
            for val in snap.values():
                assert MIN_SCORE <= val <= MAX_SCORE


# ===========================================================================
# reset
# ===========================================================================


class TestReset:
    """Verify the reset() method under various conditions."""

    def test_reset_modified_entity_restores_default(self):
        """After modifications, reset() brings the entity back to DEFAULT_SCORE."""
        scorer = fresh()
        scorer.record_failure("e", weight=200)
        assert scorer.score("e") != DEFAULT_SCORE
        scorer.reset("e")
        assert scorer.score("e") == DEFAULT_SCORE

    def test_reset_unknown_entity_sets_to_default(self):
        """reset() on a never-seen entity sets its score to DEFAULT_SCORE."""
        scorer = fresh()
        scorer.reset("new_entity")
        assert scorer.score("new_entity") == DEFAULT_SCORE

    def test_reset_already_default_entity_stays_at_default(self):
        """reset() on an entity already at DEFAULT_SCORE has no observable effect."""
        scorer = fresh()
        scorer.record_success("e", weight=0)  # score stays at 500
        scorer.reset("e")
        assert scorer.score("e") == DEFAULT_SCORE

    def test_reset_does_not_affect_other_entities(self):
        """Resetting entity A must not change entity B's score."""
        scorer = fresh()
        scorer.record_success("a", weight=100)
        scorer.record_failure("b", weight=100)
        scorer.reset("a")
        assert scorer.score("b") == DEFAULT_SCORE - 100


# ===========================================================================
# Multiple entities
# ===========================================================================


class TestMultipleEntities:
    """Ensure different entities are tracked independently."""

    def test_100_entities_all_default_to_500(self):
        """Scoring 100 different untouched entities all returns DEFAULT_SCORE."""
        scorer = fresh()
        for i in range(100):
            assert scorer.score(f"entity_{i}") == DEFAULT_SCORE

    def test_success_on_a_does_not_affect_b(self):
        """record_success on entity A leaves entity B at DEFAULT_SCORE."""
        scorer = fresh()
        scorer.record_success("a", weight=10)
        assert scorer.score("b") == DEFAULT_SCORE

    def test_violation_on_b_does_not_affect_a(self):
        """record_violation on entity B leaves entity A at DEFAULT_SCORE."""
        scorer = fresh()
        scorer.record_violation("b")
        assert scorer.score("a") == DEFAULT_SCORE

    def test_independent_trajectories_tracked_simultaneously(self):
        """Ten entities undergoing different operations all end at correct scores."""
        scorer = fresh()
        for i in range(10):
            scorer.record_success(f"e{i}", weight=i * 10)
        for i in range(10):
            expected = min(DEFAULT_SCORE + i * 10, MAX_SCORE)
            assert scorer.score(f"e{i}") == expected


# ===========================================================================
# Thread safety
# ===========================================================================


class TestThreadSafety:
    """Concurrent access must not corrupt scores or raise exceptions."""

    def test_20_threads_concurrent_success_stays_in_bounds(self):
        """20 threads each recording success never push score above MAX_SCORE."""
        scorer = fresh()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    scorer.record_success("shared", weight=10)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert scorer.score("shared") == MAX_SCORE

    def test_20_threads_concurrent_failure_stays_in_bounds(self):
        """20 threads each recording failure never push score below MIN_SCORE."""
        scorer = fresh()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    scorer.record_failure("shared", weight=10)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert scorer.score("shared") == MIN_SCORE

    def test_concurrent_reads_during_writes_raise_no_exceptions(self):
        """score() called concurrently with record_success must never throw."""
        scorer = fresh()
        errors: list[Exception] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                try:
                    _ = scorer.score("entity")
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        def writer() -> None:
            for _ in range(200):
                scorer.record_success("entity", weight=1)
            stop.set()

        readers = [threading.Thread(target=reader) for _ in range(5)]
        writer_thread = threading.Thread(target=writer)

        for r in readers:
            r.start()
        writer_thread.start()
        writer_thread.join()
        stop.set()
        for r in readers:
            r.join()

        assert not errors

    def test_stress_20_threads_success_and_failure_simultaneously(self):
        """20 threads each doing record_success and record_failure concurrently."""
        scorer = fresh()
        errors: list[Exception] = []

        def mixed_worker() -> None:
            try:
                for _ in range(25):
                    scorer.record_success("entity", weight=10)
                    scorer.record_failure("entity", weight=10)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=mixed_worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        final = scorer.score("entity")
        assert MIN_SCORE <= final <= MAX_SCORE

    def test_concurrent_get_scores_returns_valid_copies(self):
        """Concurrent get_scores() calls return dicts with only valid scores."""
        scorer = fresh()
        results: list[dict] = []
        errors: list[Exception] = []

        def modifier() -> None:
            for i in range(40):
                scorer.record_success(f"e{i}", weight=5)
                scorer.record_failure(f"f{i}", weight=5)

        def snapshot_taker() -> None:
            try:
                for _ in range(10):
                    snap = scorer.get_scores()
                    results.append(snap)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=modifier) for _ in range(3)] + [
            threading.Thread(target=snapshot_taker) for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for snap in results:
            for v in snap.values():
                assert isinstance(v, int)
                assert MIN_SCORE <= v <= MAX_SCORE


# ===========================================================================
# Weight edge cases
# ===========================================================================


class TestWeightEdgeCases:
    """Degenerate weight values must be handled gracefully."""

    def test_zero_weight_success_is_noop(self):
        """record_success(weight=0) leaves the score unchanged."""
        scorer = fresh()
        scorer.record_success("e", weight=0)
        assert scorer.score("e") == DEFAULT_SCORE

    def test_zero_weight_failure_is_noop(self):
        """record_failure(weight=0) leaves the score unchanged."""
        scorer = fresh()
        scorer.record_failure("e", weight=0)
        assert scorer.score("e") == DEFAULT_SCORE

    def test_zero_weight_violation_is_noop(self):
        """record_violation(weight=0) leaves the score unchanged."""
        scorer = fresh()
        scorer.record_violation("e", weight=0)
        assert scorer.score("e") == DEFAULT_SCORE

    def test_maxsize_weight_success_clamps_to_max(self):
        """weight=sys.maxsize on record_success clamps to MAX_SCORE."""
        scorer = fresh()
        scorer.record_success("e", weight=sys.maxsize)
        assert scorer.score("e") == MAX_SCORE

    def test_maxsize_weight_failure_floors_to_min(self):
        """weight=sys.maxsize on record_failure floors to MIN_SCORE."""
        scorer = fresh()
        scorer.record_failure("e", weight=sys.maxsize)
        assert scorer.score("e") == MIN_SCORE
