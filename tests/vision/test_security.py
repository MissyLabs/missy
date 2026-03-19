"""Security tests for the vision subsystem.

Ensures vision operations don't leak secrets, audit logging works,
and access controls are respected.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from missy.vision.audit import (
    audit_vision_burst,
    audit_vision_capture,
    audit_vision_error,
    audit_vision_intent,
    audit_vision_session,
)
from missy.vision.intent import VisionIntentClassifier

# ---------------------------------------------------------------------------
# Audit event tests
# ---------------------------------------------------------------------------


class TestVisionAuditEvents:
    """Verify audit events are emitted correctly."""

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_capture_audit_logs_device(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        audit_vision_capture(
            device="/dev/video0",
            source_type="webcam",
            success=True,
            width=1920,
            height=1080,
        )
        mock_logger.log.assert_called_once()
        event = mock_logger.log.call_args[0][0]
        assert event["device"] == "/dev/video0"
        assert event["success"] is True

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_capture_audit_logs_failure(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        audit_vision_capture(
            device="/dev/video0",
            success=False,
            error="Permission denied",
        )
        event = mock_logger.log.call_args[0][0]
        assert event["success"] is False
        assert "Permission denied" in event["error"]

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_burst_audit(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        audit_vision_burst(
            device="/dev/video0",
            count=5,
            successful=4,
            best_only=True,
        )
        event = mock_logger.log.call_args[0][0]
        assert event["requested_count"] == 5
        assert event["successful_count"] == 4
        assert event["best_only"] is True

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_error_audit(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        audit_vision_error(
            operation="capture",
            error="Device busy",
            device="/dev/video0",
            recoverable=True,
        )
        event = mock_logger.log.call_args[0][0]
        assert event["operation"] == "capture"
        assert event["recoverable"] is True

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_intent_audit_does_not_log_full_text(self, mock_get_logger):
        """Intent audit should log text_length, not the full user text."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        audit_vision_intent(
            text="Look at this secret document about my password",
            intent="look",
            confidence=0.95,
            decision="activate",
        )
        event = mock_logger.log.call_args[0][0]
        # Should have text_length, NOT the raw text
        assert "text_length" in event
        assert "text" not in event  # raw text must not be logged
        assert "secret" not in str(event)
        assert "password" not in str(event)

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_session_audit(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        audit_vision_session(
            action="create",
            task_id="puzzle_1",
            task_type="puzzle",
        )
        event = mock_logger.log.call_args[0][0]
        assert event["action"] == "session_create"
        assert event["task_id"] == "puzzle_1"

    def test_audit_never_raises(self):
        """Audit failures must never crash the vision pipeline."""
        with patch(
            "missy.observability.audit_logger.get_audit_logger",
            side_effect=RuntimeError("audit broken"),
        ):
            # These must all succeed silently
            audit_vision_capture(device="/dev/video0")
            audit_vision_burst(count=3)
            audit_vision_error(operation="test")
            audit_vision_intent(text="hello")
            audit_vision_session(action="create")

    def test_audit_with_none_logger(self):
        """Audit should handle None logger gracefully."""
        with patch("missy.observability.audit_logger.get_audit_logger", return_value=None):
            audit_vision_capture(device="/dev/video0")  # should not raise


# ---------------------------------------------------------------------------
# Intent classification security
# ---------------------------------------------------------------------------


class TestIntentSecurity:
    """Ensure intent classification doesn't create always-on surveillance."""

    def test_normal_speech_does_not_activate(self):
        """Normal conversation should not trigger vision activation."""
        classifier = VisionIntentClassifier()
        normal_phrases = [
            "What's the weather today?",
            "Set a timer for 10 minutes",
            "Tell me a joke",
            "Play some music",
            "What time is it?",
            "Remind me to buy groceries",
        ]
        for phrase in normal_phrases:
            result = classifier.classify(phrase)
            assert result.decision.value != "activate", (
                f"Normal phrase triggered activation: {phrase!r}"
            )

    def test_empty_input_does_not_activate(self):
        classifier = VisionIntentClassifier()
        for text in ["", "   ", "\n", "\t"]:
            result = classifier.classify(text)
            assert result.confidence == 0.0

    def test_activation_requires_strong_confidence(self):
        """Auto-activation should only happen with high confidence."""
        classifier = VisionIntentClassifier(auto_threshold=0.80)
        # Weak vision signals should not auto-activate
        result = classifier.classify("something about colors maybe")
        if result.decision.value == "activate":
            assert result.confidence >= 0.80

    def test_activation_log_is_bounded(self):
        """Activation log should not grow unbounded in long sessions."""
        classifier = VisionIntentClassifier()
        for i in range(1000):
            classifier.classify(f"look at this {i}")

        log = classifier.activation_log
        assert len(log) == 500  # capped at 500 entries
        classifier.clear_log()
        assert len(classifier.activation_log) == 0


# ---------------------------------------------------------------------------
# Scene memory security
# ---------------------------------------------------------------------------


class TestSceneMemorySecurity:
    def test_session_close_releases_image_data(self):
        """Closing a session should clear image data to prevent leaks."""
        from missy.vision.scene_memory import SceneSession, TaskType

        session = SceneSession("test", TaskType.GENERAL)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        session.add_frame(img, deduplicate=False)
        session.add_frame(img, deduplicate=False)

        assert session.frame_count == 2

        session.close()
        assert not session.is_active

        # Image data should be cleared
        for frame in session._frames:
            assert frame.image.size <= 1  # empty or minimal
