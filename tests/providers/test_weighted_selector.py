"""Tests for the generic smooth weighted round-robin provider selector."""

from __future__ import annotations

from missy.providers.weighted_selector import WeightedRoundRobin


class TestEqualWeightsDegradeToRoundRobin:
    def test_equal_weights_rotate_in_order(self) -> None:
        wrr = WeightedRoundRobin()
        weights = {"a": 1.0, "b": 1.0, "c": 1.0}
        picks = [wrr.select(["a", "b", "c"], weights) for _ in range(6)]
        assert picks == ["a", "b", "c", "a", "b", "c"]

    def test_missing_weight_defaults_to_one(self) -> None:
        wrr = WeightedRoundRobin()
        picks = [wrr.select(["a", "b"], {}) for _ in range(4)]
        assert picks == ["a", "b", "a", "b"]


class TestWeightedBalancing:
    def test_higher_weight_selected_more_often(self) -> None:
        wrr = WeightedRoundRobin()
        weights = {"a": 3.0, "b": 1.0}
        picks = [wrr.select(["a", "b"], weights) for _ in range(8)]
        assert picks.count("a") == 6
        assert picks.count("b") == 2

    def test_zero_weight_never_selected(self) -> None:
        wrr = WeightedRoundRobin()
        weights = {"a": 0.0, "b": 1.0}
        picks = [wrr.select(["a", "b"], weights) for _ in range(5)]
        assert picks == ["b"] * 5

    def test_all_zero_weight_returns_none(self) -> None:
        wrr = WeightedRoundRobin()
        assert wrr.select(["a", "b"], {"a": 0.0, "b": 0.0}) is None

    def test_negative_weight_treated_as_zero(self) -> None:
        wrr = WeightedRoundRobin()
        weights = {"a": -5.0, "b": 1.0}
        picks = [wrr.select(["a", "b"], weights) for _ in range(3)]
        assert picks == ["b"] * 3

    def test_empty_candidates_returns_none(self) -> None:
        wrr = WeightedRoundRobin()
        assert wrr.select([], {"a": 1.0}) is None


class TestFluctuatingCandidateSet:
    def test_excluded_name_keeps_its_rotation_state(self) -> None:
        """A name temporarily missing from candidates (e.g. its breaker
        opened) doesn't lose its accumulated weight -- it resumes fairly
        once it's eligible again rather than restarting from scratch."""
        wrr = WeightedRoundRobin()
        weights = {"a": 1.0, "b": 1.0, "c": 1.0}
        first = wrr.select(["a", "b", "c"], weights)
        assert first == "a"
        # "a" excluded this round (as if its breaker opened).
        second = wrr.select(["b", "c"], weights)
        assert second == "b"
        # "a" becomes eligible again -- rotation continues fairly rather
        # than re-picking "a" immediately just because it "hasn't gone yet".
        third = wrr.select(["a", "b", "c"], weights)
        assert third == "c"

    def test_weight_change_takes_effect_on_next_select(self) -> None:
        wrr = WeightedRoundRobin()
        assert wrr.select(["a", "b"], {"a": 1.0, "b": 1.0}) == "a"
        # Re-weight "b" heavily; subsequent picks should favor it.
        picks = [wrr.select(["a", "b"], {"a": 1.0, "b": 9.0}) for _ in range(10)]
        assert picks.count("b") > picks.count("a")


class TestThreadSafety:
    def test_concurrent_selection_distributes_by_weight(self) -> None:
        import threading

        wrr = WeightedRoundRobin()
        weights = {"a": 1.0, "b": 1.0}
        counts = {"a": 0, "b": 0}
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(250):
                picked = wrr.select(["a", "b"], weights)
                with lock:
                    counts[picked] += 1

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counts["a"] == 500
        assert counts["b"] == 500
