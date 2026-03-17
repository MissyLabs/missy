"""Tests for missy.security.trust — Trust scoring system."""

from __future__ import annotations

from missy.security.trust import TrustScorer


class TestTrustScorer:
    """Tests for TrustScorer entity reliability tracking."""

    def test_default_score(self):
        """New entity starts at 500."""
        scorer = TrustScorer()
        assert scorer.score("new_entity") == 500

    def test_success_increases(self):
        """Recording a success increases the score."""
        scorer = TrustScorer()
        scorer.record_success("tool_a", weight=10)
        assert scorer.score("tool_a") == 510

    def test_failure_decreases(self):
        """Recording a failure decreases the score."""
        scorer = TrustScorer()
        scorer.record_failure("tool_b", weight=50)
        assert scorer.score("tool_b") == 450

    def test_violation_major_decrease(self):
        """Policy violation causes a large drop."""
        scorer = TrustScorer()
        scorer.record_violation("bad_server", weight=200)
        assert scorer.score("bad_server") == 300

    def test_score_capped_at_1000(self):
        """Score must not exceed 1000."""
        scorer = TrustScorer()
        # Push score to max
        for _ in range(100):
            scorer.record_success("reliable", weight=100)
        assert scorer.score("reliable") == 1000

    def test_score_floored_at_0(self):
        """Score must not go below 0."""
        scorer = TrustScorer()
        # Push score to minimum
        for _ in range(20):
            scorer.record_failure("unreliable", weight=100)
        assert scorer.score("unreliable") == 0

    def test_is_trusted_threshold(self):
        """Entity below threshold is untrusted."""
        scorer = TrustScorer()
        # Default score 500 is above threshold 200
        assert scorer.is_trusted("entity", threshold=200) is True
        # Drop below threshold
        scorer.record_violation("entity", weight=400)
        # Score is now 100, below threshold 200
        assert scorer.is_trusted("entity", threshold=200) is False

    def test_reset(self):
        """Reset restores entity to default score 500."""
        scorer = TrustScorer()
        scorer.record_failure("entity", weight=100)
        assert scorer.score("entity") == 400
        scorer.reset("entity")
        assert scorer.score("entity") == 500
