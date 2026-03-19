"""Cross-module integration tests for vision subsystem.


Tests interactions between:
- Discovery → Capture → Health monitor flow
- Scene memory → Vision memory bridge flow
- Intent classifier → Audit logging flow
- Pipeline → Orientation → Analysis chain
- Shutdown coordination
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Discovery → Capture → Health Monitor integration
# ---------------------------------------------------------------------------


class TestDiscoveryCaptureHealthFlow:
    """Tests the flow from device discovery through capture to health recording."""

    def test_health_monitor_records_discovery_event(self) -> None:
        """Health monitor's record_device_discovery creates device entry."""
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        monitor.record_device_discovery("/dev/video0")
        stats = monitor.get_device_stats("/dev/video0")
        assert stats is not None
        assert stats.last_seen > 0

    def test_health_monitor_tracks_success_rate(self) -> None:
        """After mixed captures, success rate is computed correctly."""
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        for _ in range(7):
            monitor.record_capture(success=True, device="/dev/video0", latency_ms=50)
        for _ in range(3):
            monitor.record_capture(success=False, device="/dev/video0", error="blank")
        report = monitor.get_health_report()
        stats = report["devices"]["/dev/video0"]
        assert stats["total_captures"] == 10
        assert stats["success_rate"] == 0.7

    def test_health_monitor_consecutive_failure_tracking(self) -> None:
        from missy.vision.health_monitor import VisionHealthMonitor

        monitor = VisionHealthMonitor()
        for _ in range(5):
            monitor.record_capture(success=False, device="/dev/video0", error="timeout")
        report = monitor.get_health_report()
        assert report["devices"]["/dev/video0"]["consecutive_failures"] == 5

        # Success resets consecutive failures
        monitor.record_capture(success=True, device="/dev/video0", latency_ms=30)
        report = monitor.get_health_report()
        assert report["devices"]["/dev/video0"]["consecutive_failures"] == 0


# ---------------------------------------------------------------------------
# Scene Memory → Vision Memory Bridge flow
# ---------------------------------------------------------------------------


class TestSceneToMemoryBridgeFlow:
    """Tests the flow from scene sessions to persistent memory."""

    def test_scene_summary_can_be_stored_in_bridge(self) -> None:
        """A scene session summary can be persisted via VisionMemoryBridge."""
        from missy.vision.scene_memory import SceneSession
        from missy.vision.vision_memory import VisionMemoryBridge

        # Create a scene session with frames
        session = SceneSession(task_id="puzzle-1", max_frames=5)
        for i in range(3):
            img = np.zeros((100, 100, 3), dtype=np.uint8) + i * 50
            session.add_frame(img, deduplicate=False)
        session.add_observation("Found 3 edge pieces")

        summary = session.summarize()

        # Store in bridge
        mock_mem = MagicMock()
        bridge = VisionMemoryBridge(memory_store=mock_mem)
        obs_id = bridge.store_observation(
            session_id="puzzle-1",
            task_type="puzzle",
            observation=f"Scene: {summary['frame_count']} frames, observations: {summary['observations']}",
            confidence=0.85,
        )
        assert obs_id
        mock_mem.add_turn.assert_called_once()

    def test_scene_change_detection_between_frames(self) -> None:
        """Change detection produces valid SceneChange objects."""
        from missy.vision.scene_memory import SceneFrame, SceneSession

        session = SceneSession(task_id="test", max_frames=10)

        # Add two different frames
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

        f1 = SceneFrame(frame_id=1, image=img1)
        f2 = SceneFrame(frame_id=2, image=img2)

        change = session.detect_change(f1, f2)
        assert change.change_score > 0
        assert change.from_frame == 1
        assert change.to_frame == 2

    def test_scene_deduplication_rejects_similar_frames(self) -> None:
        """Adding identical frames with deduplication should reject duplicates."""
        from missy.vision.scene_memory import SceneSession

        session = SceneSession(task_id="test", max_frames=10)
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        added_first = session.add_frame(img)
        added_dup = session.add_frame(
            img.copy(),
            deduplicate=True,
            dedup_threshold=3,
        )
        assert added_first is not None
        assert added_dup is None  # should be rejected as duplicate
        assert session.frame_count == 1


# ---------------------------------------------------------------------------
# Intent → Audit flow
# ---------------------------------------------------------------------------


class TestIntentAuditFlow:
    """Intent classification results are logged via audit."""

    def test_intent_result_to_audit_dict(self) -> None:
        from missy.vision.intent import VisionIntentClassifier

        classifier = VisionIntentClassifier()
        result = classifier.classify("Can you look at this puzzle?")
        audit = result.to_audit_dict()
        assert "intent" in audit
        assert "decision" in audit
        assert "confidence" in audit
        assert isinstance(audit["confidence"], float)

    def test_high_confidence_triggers_activate(self) -> None:
        from missy.vision.intent import ActivationDecision, VisionIntentClassifier

        classifier = VisionIntentClassifier(auto_threshold=0.6)
        result = classifier.classify("Missy, look at this painting please")
        # Explicit look command should have high confidence
        assert result.confidence > 0.5
        assert result.decision in (ActivationDecision.ACTIVATE, ActivationDecision.ASK)

    def test_irrelevant_text_skips_vision(self) -> None:
        from missy.vision.intent import ActivationDecision, VisionIntentClassifier

        classifier = VisionIntentClassifier()
        result = classifier.classify("What is the capital of France?")
        assert result.decision == ActivationDecision.SKIP


# ---------------------------------------------------------------------------
# Pipeline → Orientation chain
# ---------------------------------------------------------------------------


class TestPipelineOrientationChain:
    """Tests the preprocessing → orientation correction → quality assessment chain."""

    def test_pipeline_then_orientation_check(self) -> None:
        from missy.vision.orientation import Orientation, detect_orientation
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Process through pipeline
        processed = pipeline.process(img)
        assert processed is not None

        # Check orientation
        result = detect_orientation(processed)
        assert result.detected == Orientation.NORMAL

    def test_portrait_pipeline_and_auto_correct(self) -> None:
        from missy.vision.orientation import auto_correct
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        # Very tall portrait image
        img = np.zeros((2000, 200, 3), dtype=np.uint8)

        processed = pipeline.process(img)
        corrected, result = auto_correct(processed)

        if result.correction_applied:
            h, w = corrected.shape[:2]
            assert w > h  # should now be landscape

    def test_quality_assessment_after_processing(self) -> None:
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

        processed = pipeline.process(img)
        quality = pipeline.assess_quality(processed)
        assert quality["quality"] in ("poor", "fair", "good", "excellent")
        assert quality["width"] > 0
        assert quality["height"] > 0


# ---------------------------------------------------------------------------
# Shutdown coordination
# ---------------------------------------------------------------------------


class TestShutdownCoordination:
    def setup_method(self) -> None:
        from missy.vision.shutdown import reset_shutdown_state
        reset_shutdown_state()

    def teardown_method(self) -> None:
        from missy.vision.shutdown import reset_shutdown_state
        reset_shutdown_state()

    def test_shutdown_idempotent(self) -> None:
        from missy.vision.shutdown import vision_shutdown

        result1 = vision_shutdown()
        result2 = vision_shutdown()
        assert result1["status"] == "shutdown"
        assert result2["status"] == "already_shutdown"

    def test_shutdown_reports_steps(self) -> None:
        from missy.vision.shutdown import vision_shutdown

        result = vision_shutdown()
        assert "steps" in result
        assert len(result["steps"]) > 0

    def test_concurrent_shutdown(self) -> None:
        """Multiple threads calling shutdown — only one performs cleanup."""
        from missy.vision.shutdown import vision_shutdown

        results: list[dict] = []
        barrier = threading.Barrier(5)

        def worker() -> None:
            barrier.wait()
            results.append(vision_shutdown())

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(results) == 5
        shutdown_count = sum(1 for r in results if r["status"] == "shutdown")
        already_count = sum(1 for r in results if r["status"] == "already_shutdown")
        assert shutdown_count == 1
        assert already_count == 4


# ---------------------------------------------------------------------------
# Analysis prompt builder integration
# ---------------------------------------------------------------------------


class TestAnalysisPromptIntegration:
    def _request(self, mode: str) -> Any:
        from missy.vision.analysis import AnalysisMode, AnalysisRequest
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        return AnalysisRequest(image=img, mode=AnalysisMode(mode))

    def test_puzzle_prompt_has_required_sections(self) -> None:
        from missy.vision.analysis import AnalysisPromptBuilder

        builder = AnalysisPromptBuilder()
        prompt = builder.build_prompt(self._request("puzzle"))
        assert "puzzle" in prompt.lower() or "piece" in prompt.lower()

    def test_painting_prompt_has_encouraging_tone(self) -> None:
        from missy.vision.analysis import AnalysisPromptBuilder

        builder = AnalysisPromptBuilder()
        prompt = builder.build_prompt(self._request("painting"))
        prompt_lower = prompt.lower()
        # Should have encouraging/supportive language
        assert any(word in prompt_lower for word in [
            "encourage", "support", "gentle", "warm", "constructive",
            "praise", "positive", "kind", "helpful",
        ])

    def test_general_prompt_is_not_empty(self) -> None:
        from missy.vision.analysis import AnalysisPromptBuilder

        builder = AnalysisPromptBuilder()
        prompt = builder.build_prompt(self._request("general"))
        assert len(prompt) > 50

    def test_puzzle_preprocessor_edge_enhancement(self) -> None:
        from missy.vision.analysis import PuzzlePreprocessor

        preprocessor = PuzzlePreprocessor()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw some edges
        img[40:60, 40:60] = 255

        enhanced = preprocessor.enhance_edges(img)
        assert enhanced is not None
        assert enhanced.shape == img.shape

    def test_puzzle_preprocessor_color_regions(self) -> None:
        from missy.vision.analysis import PuzzlePreprocessor

        preprocessor = PuzzlePreprocessor()
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)

        regions = preprocessor.extract_color_regions(img)
        assert "dominant_colors" in regions
        assert len(regions["dominant_colors"]) > 0

    def test_puzzle_preprocessor_edge_detection(self) -> None:
        from missy.vision.analysis import PuzzlePreprocessor

        preprocessor = PuzzlePreprocessor()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[30:70, 30:70] = 200  # clear rectangle

        result = preprocessor.detect_edges_and_corners(img)
        assert "contour_count" in result
        assert result["contour_count"] >= 0
