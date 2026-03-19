"""Tests for benchmark and intent classifier edge cases.

Covers:
- CaptureBenchmark recording and reporting
- Benchmark statistics (percentiles, averages)
- Intent classification for various utterances
- Intent confidence and decision thresholds
- Activation decisions (activate, ask, skip)
"""

from __future__ import annotations

import threading

from missy.vision.benchmark import CaptureBenchmark
from missy.vision.intent import (
    ActivationDecision,
    VisionIntent,
    VisionIntentClassifier,
)

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class TestCaptureBenchmark:
    """Test CaptureBenchmark recording and reporting."""

    def test_record_and_report(self):
        bench = CaptureBenchmark()
        bench.record("capture", 50.0)
        bench.record("capture", 60.0)
        bench.record("capture", 40.0)

        report = bench.report()
        cats = report["categories"]
        assert "capture" in cats
        stats = cats["capture"]
        assert stats["count"] == 3
        assert stats["mean_ms"] == 50.0
        assert stats["min_ms"] == 40.0
        assert stats["max_ms"] == 60.0

    def test_empty_report(self):
        bench = CaptureBenchmark()
        report = bench.report()
        assert report["categories"] == {}

    def test_multiple_categories(self):
        bench = CaptureBenchmark()
        bench.record("capture", 10.0)
        bench.record("pipeline", 20.0)
        bench.record("save", 30.0)

        cats = bench.report()["categories"]
        assert len(cats) == 3
        assert "capture" in cats
        assert "pipeline" in cats
        assert "save" in cats

    def test_thread_safe_recording(self):
        bench = CaptureBenchmark()
        errors: list[Exception] = []

        def record_batch(category: str, count: int):
            try:
                for i in range(count):
                    bench.record(category, float(i))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_batch, args=(f"cat-{i}", 100))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        cats = bench.report()["categories"]
        assert len(cats) == 4
        for cat in cats.values():
            assert cat["count"] == 100

    def test_max_samples_limit(self):
        bench = CaptureBenchmark(max_samples=10)
        for i in range(50):
            bench.record("test", float(i))

        cats = bench.report()["categories"]
        # Should only retain max_samples
        assert cats["test"]["count"] == 10

    def test_record_capture_convenience(self):
        bench = CaptureBenchmark()
        bench.record_capture(latency_ms=45.0, quality=0.87)
        assert "capture" in bench.report()["categories"]

    def test_record_pipeline_convenience(self):
        bench = CaptureBenchmark()
        bench.record_pipeline(processing_ms=12.3)
        assert "pipeline" in bench.report()["categories"]


# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------


class TestVisionIntentClassifier:
    """Test vision intent classification."""

    def test_look_at_this(self):
        classifier = VisionIntentClassifier()
        result = classifier.classify("Hey Missy, look at this")
        assert result.intent == VisionIntent.LOOK
        assert result.decision != ActivationDecision.SKIP

    def test_puzzle_piece(self):
        classifier = VisionIntentClassifier()
        result = classifier.classify("Can you tell where this puzzle piece goes?")
        assert result.intent == VisionIntent.PUZZLE
        assert result.decision == ActivationDecision.ACTIVATE

    def test_painting_feedback(self):
        classifier = VisionIntentClassifier()
        result = classifier.classify("What do you think of this painting?")
        assert result.intent == VisionIntent.PAINTING

    def test_no_vision_needed(self):
        classifier = VisionIntentClassifier()
        result = classifier.classify("What's the weather today?")
        assert result.intent == VisionIntent.NONE
        assert result.decision == ActivationDecision.SKIP

    def test_check_table(self):
        classifier = VisionIntentClassifier()
        result = classifier.classify("Can you check what is on the table?")
        assert result.intent != VisionIntent.NONE

    def test_screenshot_request(self):
        classifier = VisionIntentClassifier()
        result = classifier.classify("Take a screenshot of my desktop")
        assert result.intent == VisionIntent.SCREENSHOT

    def test_read_text(self):
        classifier = VisionIntentClassifier()
        result = classifier.classify("Can you read what this sign says?")
        # "read" may classify as INSPECT or READ depending on patterns
        assert result.intent != VisionIntent.NONE

    def test_compare_request(self):
        classifier = VisionIntentClassifier()
        result = classifier.classify("Does this piece match the sky section?")
        assert result.intent != VisionIntent.NONE

    def test_empty_text_returns_none(self):
        classifier = VisionIntentClassifier()
        result = classifier.classify("")
        assert result.intent == VisionIntent.NONE
        assert result.decision == ActivationDecision.SKIP

    def test_confidence_range(self):
        classifier = VisionIntentClassifier()
        result = classifier.classify("Look at this painting please")
        assert 0.0 <= result.confidence <= 1.0

    def test_audit_dict_serializable(self):
        classifier = VisionIntentClassifier()
        result = classifier.classify("Check this puzzle board")
        audit = result.to_audit_dict()
        assert "intent" in audit
        assert "confidence" in audit
        assert "decision" in audit
        assert isinstance(audit["confidence"], float)

    def test_improve_painting(self):
        classifier = VisionIntentClassifier()
        result = classifier.classify("How can I improve this painting?")
        assert result.intent == VisionIntent.PAINTING

    def test_inspect_request(self):
        classifier = VisionIntentClassifier()
        result = classifier.classify("Look at this and inspect it carefully for damage")
        assert result.intent != VisionIntent.NONE

    def test_custom_threshold(self):
        classifier = VisionIntentClassifier(auto_threshold=0.99)
        result = classifier.classify("Look at this")
        # With very high threshold, should ASK instead of ACTIVATE
        # (depending on confidence)
        assert result.intent == VisionIntent.LOOK
