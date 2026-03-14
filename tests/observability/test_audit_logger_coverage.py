"""Coverage gap tests for missy.observability.audit_logger.AuditLogger.

Targets uncovered lines:
  82-83  : _patched_publish wrapper catches exception from _handle_event
  123-124: _handle_event catches write exception and logs error
  151-153: get_recent_events reads file, catches read exception
  162-163: get_recent_events skips malformed JSON lines
  183-185: get_policy_violations reads file, catches read exception
  191    : get_policy_violations skips empty lines
  194-195: get_policy_violations skips malformed JSON lines
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.core.events import AuditEvent, EventBus
from missy.observability.audit_logger import AuditLogger


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


def _make_event(event_type: str = "test.event", result: str = "allow") -> AuditEvent:
    return AuditEvent.now(
        session_id="s1",
        task_id="t1",
        event_type=event_type,
        category="network",
        result=result,
    )


# ---------------------------------------------------------------------------
# Lines 82-83: _patched_publish catches exception from _handle_event
# ---------------------------------------------------------------------------


class TestSubscribePatchedPublishExceptionHandling:
    """When _handle_event raises, the patched publish method catches it and logs."""

    def test_publish_does_not_propagate_handle_event_exception(self, bus: EventBus, tmp_path: Path):
        """Exception in _handle_event is silently swallowed by the wrapper."""
        al = AuditLogger(log_path=str(tmp_path / "audit.jsonl"), bus=bus)

        with patch.object(al, "_handle_event", side_effect=RuntimeError("crash")):
            # publish should NOT raise even though _handle_event raises
            bus.publish(_make_event())

    def test_module_logger_exception_called_on_handle_event_failure(
        self, bus: EventBus, tmp_path: Path
    ):
        """When _handle_event raises, the module logger.exception is called."""
        al = AuditLogger(log_path=str(tmp_path / "audit.jsonl"), bus=bus)

        with patch.object(al, "_handle_event", side_effect=ValueError("bad event")):
            with patch("missy.observability.audit_logger._module_logger") as mock_logger:
                bus.publish(_make_event())
                mock_logger.exception.assert_called_once()


# ---------------------------------------------------------------------------
# Lines 123-124: _handle_event catches write exception
# ---------------------------------------------------------------------------


class TestHandleEventWriteFailure:
    """When writing the log line fails, the error is logged but not raised."""

    def test_write_exception_does_not_propagate(self, bus: EventBus, tmp_path: Path):
        """IOError during file write is caught; no exception escapes."""
        al = AuditLogger(log_path=str(tmp_path / "audit.jsonl"), bus=bus)

        # AuditLogger uses Path.open (pathlib), not builtins.open
        with patch.object(Path, "open", side_effect=IOError("disk full")):
            # Should not raise
            al._handle_event(_make_event())

    def test_write_exception_logged_as_error(self, bus: EventBus, tmp_path: Path):
        """logger.error is called when the write fails."""
        al = AuditLogger(log_path=str(tmp_path / "audit.jsonl"), bus=bus)

        with patch.object(Path, "open", side_effect=IOError("no space")):
            with patch("missy.observability.audit_logger._module_logger") as mock_logger:
                al._handle_event(_make_event())
                mock_logger.error.assert_called_once()


# ---------------------------------------------------------------------------
# Lines 151-153: get_recent_events catches read exception
# ---------------------------------------------------------------------------


class TestGetRecentEventsReadFailure:
    """When read_text raises, get_recent_events returns [] and logs an error."""

    def test_read_error_returns_empty_list(self, bus: EventBus, tmp_path: Path):
        log_path = tmp_path / "audit.jsonl"
        log_path.write_text("")  # file exists so the .exists() check passes
        al = AuditLogger(log_path=str(log_path), bus=bus)

        with patch.object(Path, "read_text", side_effect=IOError("read error")):
            result = al.get_recent_events()

        assert result == []

    def test_read_error_logs_message(self, bus: EventBus, tmp_path: Path):
        log_path = tmp_path / "audit.jsonl"
        log_path.write_text("")
        al = AuditLogger(log_path=str(log_path), bus=bus)

        with patch.object(Path, "read_text", side_effect=IOError("io err")):
            with patch("missy.observability.audit_logger._module_logger") as mock_logger:
                al.get_recent_events()
                mock_logger.error.assert_called_once()


# ---------------------------------------------------------------------------
# Lines 162-163: get_recent_events skips malformed JSON
# ---------------------------------------------------------------------------


class TestGetRecentEventsMalformedJSON:
    """Malformed JSON lines are skipped with a warning; valid lines still returned."""

    def test_malformed_lines_skipped(self, bus: EventBus, tmp_path: Path):
        log_path = tmp_path / "audit.jsonl"
        good = json.dumps({"event_type": "ok", "result": "allow"})
        log_path.write_text("not json at all\n" + good + "\n")
        al = AuditLogger(log_path=str(log_path), bus=bus)

        events = al.get_recent_events()

        assert len(events) == 1
        assert events[0]["event_type"] == "ok"

    def test_malformed_line_logged_as_warning(self, bus: EventBus, tmp_path: Path):
        log_path = tmp_path / "audit.jsonl"
        log_path.write_text("{broken json\n")
        al = AuditLogger(log_path=str(log_path), bus=bus)

        with patch("missy.observability.audit_logger._module_logger") as mock_logger:
            al.get_recent_events()
            mock_logger.warning.assert_called_once()

    def test_all_malformed_returns_empty(self, bus: EventBus, tmp_path: Path):
        log_path = tmp_path / "audit.jsonl"
        log_path.write_text("garbage\nmore garbage\n")
        al = AuditLogger(log_path=str(log_path), bus=bus)

        events = al.get_recent_events()
        assert events == []


# ---------------------------------------------------------------------------
# Lines 183-185: get_policy_violations catches read exception
# ---------------------------------------------------------------------------


class TestGetPolicyViolationsReadFailure:
    """When read_text raises, get_policy_violations returns [] and logs an error."""

    def test_read_error_returns_empty_list(self, bus: EventBus, tmp_path: Path):
        log_path = tmp_path / "audit.jsonl"
        log_path.write_text("")  # must exist
        al = AuditLogger(log_path=str(log_path), bus=bus)

        with patch.object(Path, "read_text", side_effect=IOError("disk error")):
            result = al.get_policy_violations()

        assert result == []

    def test_read_error_logs_message(self, bus: EventBus, tmp_path: Path):
        log_path = tmp_path / "audit.jsonl"
        log_path.write_text("")
        al = AuditLogger(log_path=str(log_path), bus=bus)

        with patch.object(Path, "read_text", side_effect=IOError("disk err")):
            with patch("missy.observability.audit_logger._module_logger") as mock_logger:
                al.get_policy_violations()
                mock_logger.error.assert_called_once()


# ---------------------------------------------------------------------------
# Line 191: get_policy_violations skips empty/blank lines
# ---------------------------------------------------------------------------


class TestGetPolicyViolationsEmptyLines:
    """Blank lines within the log file are ignored."""

    def test_empty_lines_skipped(self, bus: EventBus, tmp_path: Path):
        log_path = tmp_path / "audit.jsonl"
        deny_record = json.dumps({"result": "deny", "event_type": "shell.exec"})
        # Mix empty lines and a valid deny
        log_path.write_text("\n\n" + deny_record + "\n\n")
        al = AuditLogger(log_path=str(log_path), bus=bus)

        violations = al.get_policy_violations()
        assert len(violations) == 1
        assert violations[0]["result"] == "deny"

    def test_only_empty_lines_returns_no_violations(self, bus: EventBus, tmp_path: Path):
        log_path = tmp_path / "audit.jsonl"
        log_path.write_text("\n   \n\t\n")
        al = AuditLogger(log_path=str(log_path), bus=bus)

        assert al.get_policy_violations() == []


# ---------------------------------------------------------------------------
# Lines 194-195: get_policy_violations skips malformed JSON
# ---------------------------------------------------------------------------


class TestGetPolicyViolationsMalformedJSON:
    """Malformed JSON lines are skipped; valid deny lines still returned."""

    def test_malformed_lines_skipped_in_violations(self, bus: EventBus, tmp_path: Path):
        log_path = tmp_path / "audit.jsonl"
        deny_record = json.dumps({"result": "deny", "event_type": "net.block"})
        log_path.write_text("broken{{json\n" + deny_record + "\n")
        al = AuditLogger(log_path=str(log_path), bus=bus)

        violations = al.get_policy_violations()
        # Only the valid deny line should be returned
        assert len(violations) == 1
        assert violations[0]["result"] == "deny"

    def test_non_deny_malformed_lines_both_skipped(self, bus: EventBus, tmp_path: Path):
        log_path = tmp_path / "audit.jsonl"
        # all bad JSON lines
        log_path.write_text("notjson\nalsonotjson\n")
        al = AuditLogger(log_path=str(log_path), bus=bus)

        assert al.get_policy_violations() == []

    def test_mix_of_allow_deny_and_malformed(self, bus: EventBus, tmp_path: Path):
        log_path = tmp_path / "audit.jsonl"
        allow = json.dumps({"result": "allow", "event_type": "net.req"})
        deny = json.dumps({"result": "deny", "event_type": "shell.block"})
        log_path.write_text(allow + "\nbad json\n" + deny + "\n")
        al = AuditLogger(log_path=str(log_path), bus=bus)

        violations = al.get_policy_violations()
        assert len(violations) == 1
        assert violations[0]["event_type"] == "shell.block"
