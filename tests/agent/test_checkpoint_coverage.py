"""Coverage-gap tests for missy/agent/checkpoint.py.

Targets uncovered lines (checkpoint.py is 129 lines but coverage report references
lines based on the full 451-line file):
  358-359: _row_to_dict — JSONDecodeError/TypeError for a malformed loop_messages column
  397-399: scan_for_recovery — CheckpointManager() constructor raises
  405-406: scan_for_recovery — abandon_old() raises
  410-412: scan_for_recovery — get_incomplete() raises
  436-437: scan_for_recovery — event_bus.publish() raises (audit emit failure)
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.checkpoint import CheckpointManager, RecoveryResult, scan_for_recovery
from missy.core.events import event_bus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path):
    return str(tmp_path / "checkpoints.db")


@pytest.fixture()
def cm(tmp_db):
    return CheckpointManager(db_path=tmp_db)


@pytest.fixture(autouse=True)
def clear_event_bus():
    event_bus.clear()
    yield
    event_bus.clear()


# ---------------------------------------------------------------------------
# _row_to_dict — malformed JSON columns (lines 358-359)
# ---------------------------------------------------------------------------


class TestRowToDictMalformedJSON:
    """Lines 358-359: JSONDecodeError and TypeError branch in _row_to_dict."""

    def test_bad_json_in_loop_messages_falls_back_to_empty_list(self, cm, tmp_db):
        """When loop_messages contains invalid JSON, _row_to_dict returns []."""
        cid = cm.create("s", "t", "p")
        # Directly corrupt the stored JSON value.
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "UPDATE checkpoints SET loop_messages = ? WHERE id = ?",
            ("not-valid-json{{", cid),
        )
        conn.commit()
        conn.close()

        rows = cm.get_incomplete()
        row = next(r for r in rows if r["id"] == cid)
        # Should fall back gracefully rather than raising.
        assert row["loop_messages"] == []

    def test_bad_json_in_tool_names_used_falls_back_to_empty_list(self, cm, tmp_db):
        """When tool_names_used contains invalid JSON, _row_to_dict returns []."""
        cid = cm.create("s", "t", "p")
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "UPDATE checkpoints SET tool_names_used = ? WHERE id = ?",
            ("{broken", cid),
        )
        conn.commit()
        conn.close()

        rows = cm.get_incomplete()
        row = next(r for r in rows if r["id"] == cid)
        assert row["tool_names_used"] == []

    def test_json_loads_type_error_falls_back_to_empty_list(self):
        """TypeError branch in _row_to_dict: when json.loads raises TypeError, return []."""
        from missy.agent.checkpoint import CheckpointManager

        # Build a fake row with a syntactically valid string but patch json.loads
        # to raise TypeError to simulate the exceptional code path.
        class FakeRow(dict):
            pass

        fake_row = FakeRow({
            "id": "fake-id",
            "session_id": "s",
            "task_id": "t",
            "prompt": "p",
            "state": "RUNNING",
            "loop_messages": "[]",  # Valid string so isinstance check passes
            "tool_names_used": "[]",
            "iteration": 0,
            "created_at": 0.0,
            "updated_at": 0.0,
        })

        with patch("missy.agent.checkpoint.json.loads", side_effect=TypeError("unexpected type")):
            result = CheckpointManager._row_to_dict(fake_row)

        assert result["loop_messages"] == []
        assert result["tool_names_used"] == []


# ---------------------------------------------------------------------------
# scan_for_recovery — CheckpointManager constructor failure (lines 397-399)
# ---------------------------------------------------------------------------


class TestScanForRecoveryConstructorFailure:
    """Lines 397-399: CheckpointManager() raises → return []."""

    def test_returns_empty_list_when_db_open_fails(self):
        """If CheckpointManager raises during construction, scan returns []."""
        with patch(
            "missy.agent.checkpoint.CheckpointManager",
            side_effect=OSError("cannot open database"),
        ):
            results = scan_for_recovery(db_path="/nonexistent/path/cp.db")

        assert results == []

    def test_returns_empty_list_on_generic_exception(self):
        """Any exception from CheckpointManager constructor is caught gracefully."""
        with patch(
            "missy.agent.checkpoint.CheckpointManager",
            side_effect=RuntimeError("unexpected"),
        ):
            results = scan_for_recovery(db_path="/tmp/dummy.db")

        assert results == []


# ---------------------------------------------------------------------------
# scan_for_recovery — abandon_old() failure (lines 405-406)
# ---------------------------------------------------------------------------


class TestScanForRecoveryAbandonOldFailure:
    """Lines 405-406: abandon_old() raises → warning logged, scan continues."""

    def test_continues_when_abandon_old_raises(self, tmp_db):
        """abandon_old() failure should not abort the scan."""
        cm_real = CheckpointManager(db_path=tmp_db)
        cm_real.create("s", "t", "p")

        mock_cm = MagicMock(spec=CheckpointManager)
        mock_cm.abandon_old.side_effect = RuntimeError("abandon blew up")
        # get_incomplete must return the real records so we can verify continuation.
        mock_cm.get_incomplete.return_value = cm_real.get_incomplete()
        mock_cm.classify.return_value = "resume"

        with patch("missy.agent.checkpoint.CheckpointManager", return_value=mock_cm):
            results = scan_for_recovery(db_path=tmp_db)

        # Scan continued past the abandon_old failure.
        assert isinstance(results, list)
        # get_incomplete was still called.
        mock_cm.get_incomplete.assert_called_once()


# ---------------------------------------------------------------------------
# scan_for_recovery — get_incomplete() failure (lines 410-412)
# ---------------------------------------------------------------------------


class TestScanForRecoveryGetIncompleteFailure:
    """Lines 410-412: get_incomplete() raises → warning logged, return []."""

    def test_returns_empty_list_when_get_incomplete_raises(self, tmp_db):
        """get_incomplete() failure causes scan to return []."""
        mock_cm = MagicMock(spec=CheckpointManager)
        mock_cm.abandon_old.return_value = 0
        mock_cm.get_incomplete.side_effect = sqlite3.OperationalError("table missing")

        with patch("missy.agent.checkpoint.CheckpointManager", return_value=mock_cm):
            results = scan_for_recovery(db_path=tmp_db)

        assert results == []


# ---------------------------------------------------------------------------
# scan_for_recovery — audit event emission failure (lines 436-437)
# ---------------------------------------------------------------------------


class TestScanForRecoveryAuditEmitFailure:
    """Lines 436-437: event_bus.publish() raises → debug logged, scan continues."""

    def test_audit_failure_does_not_abort_scan(self, tmp_db):
        """When AuditEvent.now() or event_bus.publish() raises, scan still returns results."""
        cm_real = CheckpointManager(db_path=tmp_db)
        cm_real.create("sess-1", "task-1", "my prompt")
        cm_real.create("sess-2", "task-2", "other prompt")

        with patch("missy.core.events.event_bus.publish", side_effect=RuntimeError("bus error")):
            results = scan_for_recovery(db_path=tmp_db)

        # Despite audit failures, both checkpoints are returned.
        assert len(results) == 2
        assert all(isinstance(r, RecoveryResult) for r in results)

    def test_audit_emit_exception_does_not_affect_result_fields(self, tmp_db):
        """RecoveryResult fields are correct even when audit emission fails."""
        cm_real = CheckpointManager(db_path=tmp_db)
        cm_real.create("my-session", "my-task", "the prompt text")

        with patch("missy.core.events.event_bus.publish", side_effect=Exception("boom")):
            results = scan_for_recovery(db_path=tmp_db)

        assert len(results) == 1
        r = results[0]
        assert r.session_id == "my-session"
        assert r.prompt == "the prompt text"
        assert r.action in ("resume", "restart", "abandon")
