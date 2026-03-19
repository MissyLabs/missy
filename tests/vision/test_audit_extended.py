"""Extended tests for missy.vision.audit — all audit functions and edge cases."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from missy.vision.audit import (
    _emit_audit_event,
    audit_vision_analysis,
    audit_vision_burst,
    audit_vision_capture,
    audit_vision_device_discovery,
    audit_vision_error,
    audit_vision_intent,
    audit_vision_session,
)


class TestEmitAuditEvent:
    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_emits_event(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        _emit_audit_event("vision", "test_action", {"key": "value"})

        mock_logger.log.assert_called_once()
        event = mock_logger.log.call_args[0][0]
        assert event["category"] == "vision"
        assert event["action"] == "test_action"
        assert event["key"] == "value"
        assert "timestamp" in event

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_no_logger_no_crash(self, mock_get_logger):
        mock_get_logger.return_value = None
        # Should not raise
        _emit_audit_event("vision", "test", {})

    def test_import_error_no_crash(self):
        """If audit logger module cannot be imported, should not crash."""
        with patch.dict("sys.modules", {"missy.observability.audit_logger": None}):
            # Should not raise
            _emit_audit_event("vision", "test", {})

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_logger_exception_no_crash(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_logger.log.side_effect = RuntimeError("disk full")
        mock_get_logger.return_value = mock_logger

        # Should not raise
        _emit_audit_event("vision", "test", {})


class TestAuditVisionCapture:
    @patch("missy.vision.audit._emit_audit_event")
    def test_capture_success(self, mock_emit):
        audit_vision_capture(
            device="/dev/video0",
            source_type="webcam",
            success=True,
            width=1920,
            height=1080,
        )
        mock_emit.assert_called_once()
        args = mock_emit.call_args
        assert args[0][1] == "capture"
        assert args[0][2]["success"] is True
        assert args[0][2]["device"] == "/dev/video0"

    @patch("missy.vision.audit._emit_audit_event")
    def test_capture_failure(self, mock_emit):
        audit_vision_capture(
            device="/dev/video0",
            success=False,
            error="Device busy",
        )
        details = mock_emit.call_args[0][2]
        assert details["success"] is False
        assert details["error"] == "Device busy"

    @patch("missy.vision.audit._emit_audit_event")
    def test_capture_default_trigger(self, mock_emit):
        audit_vision_capture()
        details = mock_emit.call_args[0][2]
        assert details["trigger_reason"] == "user_command"


class TestAuditVisionAnalysis:
    @patch("missy.vision.audit._emit_audit_event")
    def test_analysis_puzzle_mode(self, mock_emit):
        audit_vision_analysis(mode="puzzle", source_type="webcam", success=True)
        details = mock_emit.call_args[0][2]
        assert details["mode"] == "puzzle"

    @patch("missy.vision.audit._emit_audit_event")
    def test_analysis_failure(self, mock_emit):
        audit_vision_analysis(success=False, error="Provider error")
        details = mock_emit.call_args[0][2]
        assert details["success"] is False


class TestAuditVisionIntent:
    @patch("missy.vision.audit._emit_audit_event")
    def test_intent_logged(self, mock_emit):
        audit_vision_intent(
            text="look at this puzzle piece",
            intent="puzzle",
            confidence=0.95,
            decision="activate",
            trigger_phrase="puzzle piece",
        )
        details = mock_emit.call_args[0][2]
        assert details["intent"] == "puzzle"
        assert details["confidence"] == 0.95
        assert details["text_length"] == len("look at this puzzle piece")
        # Raw text should NOT be logged (privacy)
        assert "text" not in details or details.get("text") is None

    @patch("missy.vision.audit._emit_audit_event")
    def test_intent_confidence_rounded(self, mock_emit):
        audit_vision_intent(confidence=0.123456789)
        details = mock_emit.call_args[0][2]
        assert details["confidence"] == 0.123


class TestAuditVisionDeviceDiscovery:
    @patch("missy.vision.audit._emit_audit_event")
    def test_discovery_logged(self, mock_emit):
        audit_vision_device_discovery(
            camera_count=2,
            preferred_device="/dev/video0",
            preferred_name="C922x",
        )
        details = mock_emit.call_args[0][2]
        assert details["camera_count"] == 2
        assert details["preferred_device"] == "/dev/video0"


class TestAuditVisionSession:
    @patch("missy.vision.audit._emit_audit_event")
    def test_session_create(self, mock_emit):
        audit_vision_session(
            action="create",
            task_id="puzzle-1",
            task_type="puzzle",
            frame_count=0,
        )
        assert mock_emit.call_args[0][1] == "session_create"

    @patch("missy.vision.audit._emit_audit_event")
    def test_session_close(self, mock_emit):
        audit_vision_session(
            action="close",
            task_id="puzzle-1",
            task_type="puzzle",
            frame_count=12,
        )
        assert mock_emit.call_args[0][1] == "session_close"


class TestAuditVisionBurst:
    @patch("missy.vision.audit._emit_audit_event")
    def test_burst_logged(self, mock_emit):
        audit_vision_burst(
            device="/dev/video0",
            count=5,
            successful=4,
            best_only=True,
        )
        details = mock_emit.call_args[0][2]
        assert details["requested_count"] == 5
        assert details["successful_count"] == 4
        assert details["best_only"] is True


class TestAuditVisionError:
    @patch("missy.vision.audit._emit_audit_event")
    def test_error_logged(self, mock_emit):
        audit_vision_error(
            operation="capture",
            error="Permission denied",
            device="/dev/video0",
            recoverable=True,
        )
        details = mock_emit.call_args[0][2]
        assert details["operation"] == "capture"
        assert details["recoverable"] is True

    @patch("missy.vision.audit._emit_audit_event")
    def test_non_recoverable_error(self, mock_emit):
        audit_vision_error(
            operation="open",
            error="Device not found",
            recoverable=False,
        )
        details = mock_emit.call_args[0][2]
        assert details["recoverable"] is False
