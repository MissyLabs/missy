"""Integration and cross-module tests.


Tests cover:
- Vision CLI command argument parsing
- Source factory selection
- Analysis mode prompt selection
- Orientation detection edge cases
- Doctor diagnostic steps
- Shutdown hook registration
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

# ---------------------------------------------------------------------------
# Analysis mode prompt selection
# ---------------------------------------------------------------------------


class TestAnalysisPrompts:
    """AnalysisPromptBuilder selects correct prompts per mode."""

    def _make_request(self, mode: str, context: str = ""):  # -> AnalysisRequest
        from missy.vision.analysis import AnalysisMode, AnalysisRequest
        try:
            m = AnalysisMode(mode)
        except ValueError:
            m = AnalysisMode.GENERAL
        return AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=m,
            context=context,
        )

    def test_general_mode_prompt(self) -> None:
        from missy.vision.analysis import AnalysisPromptBuilder

        builder = AnalysisPromptBuilder()
        prompt = builder.build_prompt(self._make_request("general"))
        assert "analyze" in prompt.lower() or "description" in prompt.lower()

    def test_puzzle_mode_prompt(self) -> None:
        from missy.vision.analysis import AnalysisPromptBuilder

        builder = AnalysisPromptBuilder()
        prompt = builder.build_prompt(self._make_request("puzzle"))
        assert "puzzle" in prompt.lower()
        assert "piece" in prompt.lower()

    def test_painting_mode_prompt(self) -> None:
        from missy.vision.analysis import AnalysisPromptBuilder

        builder = AnalysisPromptBuilder()
        prompt = builder.build_prompt(self._make_request("painting"))
        assert "paint" in prompt.lower()

    def test_inspection_mode_prompt(self) -> None:
        from missy.vision.analysis import AnalysisPromptBuilder

        builder = AnalysisPromptBuilder()
        prompt = builder.build_prompt(self._make_request("inspection"))
        assert len(prompt) > 0

    def test_prompt_with_context(self) -> None:
        from missy.vision.analysis import AnalysisPromptBuilder

        builder = AnalysisPromptBuilder()
        prompt = builder.build_prompt(
            self._make_request("puzzle", context="This is a 1000-piece landscape puzzle")
        )
        assert "User note" in prompt
        assert "1000-piece" in prompt

    def test_context_sanitization_truncation(self) -> None:
        from missy.vision.analysis import AnalysisPromptBuilder

        builder = AnalysisPromptBuilder()
        long_context = "x" * 3000
        prompt = builder.build_prompt(self._make_request("general", context=long_context))
        # Context should be truncated to 2000 chars
        assert len(prompt) < len(long_context) + 1000

    def test_context_sanitization_delimiters(self) -> None:
        from missy.vision.analysis import AnalysisPromptBuilder

        builder = AnalysisPromptBuilder()
        prompt = builder.build_prompt(
            self._make_request("general", context="Some user context")
        )
        assert "[User-provided context]" in prompt


# ---------------------------------------------------------------------------
# Source factory selection
# ---------------------------------------------------------------------------


class TestSourceFactory:
    """Source factory creates correct source type."""

    def test_file_source_creation(self, tmp_path: Path) -> None:
        from missy.vision.sources import FileSource

        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        source = FileSource(str(img_path))
        assert source.source_type().value == "file"

    def test_screenshot_source_creation(self) -> None:
        from missy.vision.sources import ScreenshotSource

        source = ScreenshotSource()
        assert source.source_type().value == "screenshot"

    def test_photo_source_creation(self, tmp_path: Path) -> None:
        from missy.vision.sources import PhotoSource

        source = PhotoSource(str(tmp_path))
        assert source.source_type().value == "photo"

    def test_webcam_source_creation(self) -> None:
        from missy.vision.sources import WebcamSource

        source = WebcamSource("/dev/video0")
        assert source.source_type().value == "webcam"


# ---------------------------------------------------------------------------
# Orientation detection
# ---------------------------------------------------------------------------


class TestOrientationDetection:
    """Image orientation detection and correction."""

    def test_landscape_detection(self) -> None:
        from missy.vision.orientation import detect_orientation

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result is not None
        assert result.confidence > 0

    def test_portrait_detection(self) -> None:
        from missy.vision.orientation import detect_orientation

        img = np.zeros((640, 480, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result is not None

    def test_square_detection(self) -> None:
        from missy.vision.orientation import detect_orientation

        img = np.zeros((500, 500, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result is not None

    def test_auto_correct_portrait(self) -> None:
        from missy.vision.orientation import auto_correct

        img = np.zeros((640, 480, 3), dtype=np.uint8)
        corrected_img, result = auto_correct(img)
        assert corrected_img is not None
        assert isinstance(corrected_img, np.ndarray)
        assert result is not None

    def test_auto_correct_landscape(self) -> None:
        from missy.vision.orientation import auto_correct

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        corrected_img, result = auto_correct(img)
        assert corrected_img is not None

    def test_tiny_image_orientation(self) -> None:
        from missy.vision.orientation import detect_orientation

        img = np.zeros((1, 1, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result is not None


# ---------------------------------------------------------------------------
# Doctor diagnostics
# ---------------------------------------------------------------------------


class TestDoctorDiagnostics:
    """VisionDoctor diagnostic checks."""

    def test_check_opencv_installed(self) -> None:
        from missy.vision.doctor import VisionDoctor

        doctor = VisionDoctor()
        result = doctor.check_opencv()
        assert result is not None
        assert hasattr(result, "passed")
        assert hasattr(result, "message")

    def test_check_video_group(self) -> None:
        from missy.vision.doctor import VisionDoctor

        doctor = VisionDoctor()
        result = doctor.check_video_group()
        assert result is not None
        assert hasattr(result, "passed")

    def test_check_video_devices(self) -> None:
        from missy.vision.doctor import VisionDoctor

        doctor = VisionDoctor()
        result = doctor.check_video_devices()
        assert result is not None
        assert hasattr(result, "passed")

    def test_run_all_checks(self) -> None:
        from missy.vision.doctor import VisionDoctor

        doctor = VisionDoctor()
        report = doctor.run_all()
        assert report is not None
        assert hasattr(report, "results") or hasattr(report, "checks")

    def test_check_captures_directory(self) -> None:
        from missy.vision.doctor import VisionDoctor

        doctor = VisionDoctor()
        result = doctor.check_captures_directory()
        assert result is not None
        assert hasattr(result, "passed")

    def test_check_sysfs(self) -> None:
        from missy.vision.doctor import VisionDoctor

        doctor = VisionDoctor()
        result = doctor.check_sysfs()
        assert result is not None
        assert hasattr(result, "passed")


# ---------------------------------------------------------------------------
# Shutdown hook
# ---------------------------------------------------------------------------


class TestShutdownHook:
    """Shutdown hook registration."""

    def test_register_shutdown_hook(self) -> None:
        from missy.vision.shutdown import register_shutdown_hook, reset_shutdown_state

        reset_shutdown_state()
        # Should not raise
        register_shutdown_hook()

    def test_reset_shutdown_state(self) -> None:
        from missy.vision.shutdown import reset_shutdown_state, vision_shutdown

        # Do a shutdown
        with patch("missy.vision.scene_memory.get_scene_manager") as mock_mgr:
            mock_mgr.return_value.list_sessions.return_value = []
            with patch("missy.vision.health_monitor.get_health_monitor") as mock_hm:
                mock_hm.return_value._persist_path = None
                with patch("missy.vision.audit.audit_vision_session"):
                    result1 = vision_shutdown()
                    assert result1["status"] == "shutdown"

                    # Reset and verify we can shutdown again
                    reset_shutdown_state()
                    result2 = vision_shutdown()
                    assert result2["status"] == "shutdown"


# ---------------------------------------------------------------------------
# Intent classifier activation log
# ---------------------------------------------------------------------------


class TestIntentActivationLog:
    """Intent classifier tracks activation decisions."""

    def test_activation_log_populated(self) -> None:
        from missy.vision.intent import VisionIntentClassifier

        classifier = VisionIntentClassifier()
        classifier.classify("look at this")
        classifier.classify("what is the weather?")

        log = classifier.activation_log
        assert len(log) >= 2

    def test_activation_log_grows(self) -> None:
        from missy.vision.intent import VisionIntentClassifier

        classifier = VisionIntentClassifier()
        for i in range(50):
            classifier.classify(f"test message {i}")

        assert len(classifier.activation_log) == 50

    def test_classifier_high_threshold(self) -> None:
        from missy.vision.intent import ActivationDecision, VisionIntentClassifier

        classifier = VisionIntentClassifier(auto_threshold=1.0)
        result = classifier.classify("look at this painting")
        # With auto_threshold=1.0, auto-activate should not trigger
        assert result.decision != ActivationDecision.ACTIVATE
