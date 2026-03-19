"""Tests for missy.vision.audit — vision audit event logging."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.vision.audit import (
    audit_vision_capture,
    audit_vision_analysis,
    audit_vision_device_discovery,
    audit_vision_intent,
    audit_vision_session,
)


class TestAuditVisionCapture:
    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_emits_event(self, mock_get_logger):
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
        assert event["category"] == "vision"
        assert event["action"] == "capture"
        assert event["device"] == "/dev/video0"
        assert event["success"] is True

    @patch("missy.observability.audit_logger.get_audit_logger", return_value=None)
    def test_no_logger_no_crash(self, mock_get_logger):
        """Should not crash when audit logger is not initialized."""
        audit_vision_capture(device="/dev/video0")  # Should not raise

    def test_exception_swallowed(self):
        """Audit failures should never break the vision pipeline."""
        # With no audit logger initialized, should silently handle the error
        audit_vision_capture(device="/dev/video0")  # Should not raise


class TestAuditVisionAnalysis:
    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_emits_event(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        audit_vision_analysis(mode="puzzle", success=True)

        mock_logger.log.assert_called_once()
        event = mock_logger.log.call_args[0][0]
        assert event["mode"] == "puzzle"


class TestAuditVisionIntent:
    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_emits_event(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        audit_vision_intent(
            text="Look at this puzzle",
            intent="puzzle",
            confidence=0.92,
            decision="activate",
            trigger_phrase="puzzle",
        )

        mock_logger.log.assert_called_once()
        event = mock_logger.log.call_args[0][0]
        assert event["intent"] == "puzzle"
        assert event["confidence"] == 0.92
        # Should not include the full text for privacy
        assert event["text_length"] == len("Look at this puzzle")


class TestAuditVisionDeviceDiscovery:
    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_emits_event(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        audit_vision_device_discovery(
            camera_count=2,
            preferred_device="/dev/video0",
            preferred_name="Logitech C922x",
        )

        mock_logger.log.assert_called_once()
        event = mock_logger.log.call_args[0][0]
        assert event["camera_count"] == 2
        assert event["preferred_name"] == "Logitech C922x"


class TestAuditVisionSession:
    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_emits_event(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        audit_vision_session(
            action="create",
            task_id="puzzle-1",
            task_type="puzzle",
            frame_count=0,
        )

        mock_logger.log.assert_called_once()
        event = mock_logger.log.call_args[0][0]
        assert event["action"] == "session_create"
        assert event["task_id"] == "puzzle-1"
