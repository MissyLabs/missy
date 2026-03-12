"""Tests for missy.agent.checkpoint.CheckpointManager and scan_for_recovery."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from unittest.mock import patch

import pytest

from missy.agent.checkpoint import CheckpointManager, RecoveryResult, scan_for_recovery
from missy.core.events import event_bus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path):
    """Return a path string for a temporary SQLite database."""
    return str(tmp_path / "test_checkpoints.db")


@pytest.fixture()
def cm(tmp_db):
    """Return a CheckpointManager backed by a temp database."""
    return CheckpointManager(db_path=tmp_db)


@pytest.fixture(autouse=True)
def clear_event_bus():
    event_bus.clear()
    yield
    event_bus.clear()


# ---------------------------------------------------------------------------
# CheckpointManager: construction and schema
# ---------------------------------------------------------------------------


class TestCheckpointManagerInit:
    def test_creates_db_file(self, tmp_db):
        CheckpointManager(db_path=tmp_db)
        assert os.path.exists(tmp_db)

    def test_creates_parent_directories(self, tmp_path):
        deep_path = str(tmp_path / "a" / "b" / "c" / "checkpoints.db")
        CheckpointManager(db_path=deep_path)
        assert os.path.exists(deep_path)

    def test_schema_has_checkpoints_table(self, tmp_db):
        CheckpointManager(db_path=tmp_db)
        conn = sqlite3.connect(tmp_db)
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
        ).fetchall()
        conn.close()
        assert rows, "checkpoints table should exist"

    def test_wal_mode_enabled(self, tmp_db):
        CheckpointManager(db_path=tmp_db)
        conn = sqlite3.connect(tmp_db)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_tilde_expansion(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        db_path = "~/.missy/test_checkpoints.db"
        cm = CheckpointManager(db_path=db_path)
        expanded = os.path.expanduser(db_path)
        assert os.path.exists(expanded)


# ---------------------------------------------------------------------------
# CheckpointManager: create
# ---------------------------------------------------------------------------


class TestCreate:
    def test_returns_uuid_string(self, cm):
        cid = cm.create("sess-1", "task-1", "do something")
        assert isinstance(cid, str)
        assert len(cid) == 36  # UUID4 hyphenated

    def test_inserted_with_running_state(self, cm, tmp_db):
        cid = cm.create("sess-1", "task-1", "prompt")
        conn = sqlite3.connect(tmp_db)
        row = conn.execute("SELECT state FROM checkpoints WHERE id=?", (cid,)).fetchone()
        conn.close()
        assert row[0] == "RUNNING"

    def test_stores_session_task_and_prompt(self, cm, tmp_db):
        cid = cm.create("my-session", "my-task", "do the thing")
        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT session_id, task_id, prompt FROM checkpoints WHERE id=?", (cid,)
        ).fetchone()
        conn.close()
        assert row == ("my-session", "my-task", "do the thing")

    def test_timestamps_are_set(self, cm, tmp_db):
        before = time.time()
        cid = cm.create("s", "t", "p")
        after = time.time()
        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT created_at, updated_at FROM checkpoints WHERE id=?", (cid,)
        ).fetchone()
        conn.close()
        assert before <= row[0] <= after
        assert before <= row[1] <= after

    def test_multiple_creates_produce_unique_ids(self, cm):
        ids = {cm.create("s", "t", "p") for _ in range(20)}
        assert len(ids) == 20


# ---------------------------------------------------------------------------
# CheckpointManager: update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_persists_loop_messages(self, cm, tmp_db):
        cid = cm.create("s", "t", "p")
        messages = [{"role": "user", "content": "hello"}]
        cm.update(cid, messages, ["tool_a"], iteration=1)
        conn = sqlite3.connect(tmp_db)
        raw = conn.execute("SELECT loop_messages FROM checkpoints WHERE id=?", (cid,)).fetchone()[0]
        conn.close()
        assert json.loads(raw) == messages

    def test_update_persists_tool_names(self, cm, tmp_db):
        cid = cm.create("s", "t", "p")
        cm.update(cid, [], ["tool_x", "tool_y"], iteration=0)
        conn = sqlite3.connect(tmp_db)
        raw = conn.execute(
            "SELECT tool_names_used FROM checkpoints WHERE id=?", (cid,)
        ).fetchone()[0]
        conn.close()
        assert json.loads(raw) == ["tool_x", "tool_y"]

    def test_update_persists_iteration(self, cm, tmp_db):
        cid = cm.create("s", "t", "p")
        cm.update(cid, [], [], iteration=7)
        conn = sqlite3.connect(tmp_db)
        row = conn.execute("SELECT iteration FROM checkpoints WHERE id=?", (cid,)).fetchone()
        conn.close()
        assert row[0] == 7

    def test_update_advances_updated_at(self, cm, tmp_db):
        cid = cm.create("s", "t", "p")
        conn = sqlite3.connect(tmp_db)
        before_update = conn.execute(
            "SELECT updated_at FROM checkpoints WHERE id=?", (cid,)
        ).fetchone()[0]
        conn.close()
        time.sleep(0.01)
        cm.update(cid, [], [], iteration=0)
        conn = sqlite3.connect(tmp_db)
        after_update = conn.execute(
            "SELECT updated_at FROM checkpoints WHERE id=?", (cid,)
        ).fetchone()[0]
        conn.close()
        assert after_update >= before_update

    def test_update_does_not_change_state(self, cm, tmp_db):
        cid = cm.create("s", "t", "p")
        cm.update(cid, [], [], iteration=2)
        conn = sqlite3.connect(tmp_db)
        state = conn.execute("SELECT state FROM checkpoints WHERE id=?", (cid,)).fetchone()[0]
        conn.close()
        assert state == "RUNNING"


# ---------------------------------------------------------------------------
# CheckpointManager: complete / fail
# ---------------------------------------------------------------------------


class TestComplete:
    def test_sets_state_to_complete(self, cm, tmp_db):
        cid = cm.create("s", "t", "p")
        cm.complete(cid)
        conn = sqlite3.connect(tmp_db)
        state = conn.execute("SELECT state FROM checkpoints WHERE id=?", (cid,)).fetchone()[0]
        conn.close()
        assert state == "COMPLETE"

    def test_complete_not_in_get_incomplete(self, cm):
        cid = cm.create("s", "t", "p")
        cm.complete(cid)
        incomplete = cm.get_incomplete()
        assert all(c["id"] != cid for c in incomplete)


class TestFail:
    def test_sets_state_to_failed(self, cm, tmp_db):
        cid = cm.create("s", "t", "p")
        cm.fail(cid)
        conn = sqlite3.connect(tmp_db)
        state = conn.execute("SELECT state FROM checkpoints WHERE id=?", (cid,)).fetchone()[0]
        conn.close()
        assert state == "FAILED"

    def test_fail_with_error_appends_to_prompt(self, cm, tmp_db):
        cid = cm.create("s", "t", "original prompt")
        cm.fail(cid, error="something went wrong")
        conn = sqlite3.connect(tmp_db)
        prompt = conn.execute("SELECT prompt FROM checkpoints WHERE id=?", (cid,)).fetchone()[0]
        conn.close()
        assert "something went wrong" in prompt

    def test_fail_without_error_does_not_change_prompt(self, cm, tmp_db):
        cid = cm.create("s", "t", "original prompt")
        cm.fail(cid)
        conn = sqlite3.connect(tmp_db)
        prompt = conn.execute("SELECT prompt FROM checkpoints WHERE id=?", (cid,)).fetchone()[0]
        conn.close()
        assert prompt == "original prompt"


# ---------------------------------------------------------------------------
# CheckpointManager: get_incomplete
# ---------------------------------------------------------------------------


class TestGetIncomplete:
    def test_returns_running_checkpoints(self, cm):
        cid = cm.create("s", "t", "p")
        incomplete = cm.get_incomplete()
        ids = [c["id"] for c in incomplete]
        assert cid in ids

    def test_excludes_complete_checkpoints(self, cm):
        cid = cm.create("s", "t", "p")
        cm.complete(cid)
        incomplete = cm.get_incomplete()
        assert all(c["id"] != cid for c in incomplete)

    def test_excludes_failed_checkpoints(self, cm):
        cid = cm.create("s", "t", "p")
        cm.fail(cid)
        incomplete = cm.get_incomplete()
        assert all(c["id"] != cid for c in incomplete)

    def test_deserialises_loop_messages(self, cm):
        cid = cm.create("s", "t", "p")
        msgs = [{"role": "user", "content": "hi"}]
        cm.update(cid, msgs, [], 0)
        row = next(c for c in cm.get_incomplete() if c["id"] == cid)
        assert row["loop_messages"] == msgs

    def test_deserialises_tool_names_used(self, cm):
        cid = cm.create("s", "t", "p")
        cm.update(cid, [], ["tool_alpha"], 0)
        row = next(c for c in cm.get_incomplete() if c["id"] == cid)
        assert row["tool_names_used"] == ["tool_alpha"]

    def test_returns_empty_when_no_running(self, cm):
        assert cm.get_incomplete() == []


# ---------------------------------------------------------------------------
# CheckpointManager: classify
# ---------------------------------------------------------------------------


class TestClassify:
    def _make_checkpoint(self, age_seconds: float) -> dict:
        return {"created_at": time.time() - age_seconds}

    def test_recent_checkpoint_is_resume(self, cm):
        cp = self._make_checkpoint(age_seconds=60)  # 1 minute old
        assert cm.classify(cp) == "resume"

    def test_boundary_at_one_hour_is_restart(self, cm):
        cp = self._make_checkpoint(age_seconds=3601)  # just over 1 hour
        assert cm.classify(cp) == "restart"

    def test_middle_age_is_restart(self, cm):
        cp = self._make_checkpoint(age_seconds=7200)  # 2 hours
        assert cm.classify(cp) == "restart"

    def test_boundary_at_24_hours_is_abandon(self, cm):
        cp = self._make_checkpoint(age_seconds=86401)  # just over 24 hours
        assert cm.classify(cp) == "abandon"

    def test_very_old_is_abandon(self, cm):
        cp = self._make_checkpoint(age_seconds=7 * 86400)
        assert cm.classify(cp) == "abandon"

    def test_very_fresh_is_resume(self, cm):
        cp = self._make_checkpoint(age_seconds=1)
        assert cm.classify(cp) == "resume"


# ---------------------------------------------------------------------------
# CheckpointManager: abandon_old
# ---------------------------------------------------------------------------


class TestAbandonOld:
    def test_old_running_becomes_abandoned(self, cm, tmp_db):
        cid = cm.create("s", "t", "p")
        # Manually backdate the created_at timestamp
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "UPDATE checkpoints SET created_at=? WHERE id=?",
            (time.time() - 90000, cid),
        )
        conn.commit()
        conn.close()
        count = cm.abandon_old(max_age_seconds=86400)
        assert count == 1
        conn = sqlite3.connect(tmp_db)
        state = conn.execute("SELECT state FROM checkpoints WHERE id=?", (cid,)).fetchone()[0]
        conn.close()
        assert state == "ABANDONED"

    def test_recent_running_not_abandoned(self, cm):
        cid = cm.create("s", "t", "p")
        count = cm.abandon_old(max_age_seconds=86400)
        assert count == 0
        row = next(c for c in cm.get_incomplete() if c["id"] == cid)
        assert row["id"] == cid

    def test_returns_count_of_abandoned(self, cm, tmp_db):
        ids = [cm.create("s", "t", "p") for _ in range(3)]
        conn = sqlite3.connect(tmp_db)
        for cid in ids:
            conn.execute(
                "UPDATE checkpoints SET created_at=? WHERE id=?",
                (time.time() - 90000, cid),
            )
        conn.commit()
        conn.close()
        count = cm.abandon_old()
        assert count == 3


# ---------------------------------------------------------------------------
# CheckpointManager: delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_removes_record(self, cm, tmp_db):
        cid = cm.create("s", "t", "p")
        cm.delete(cid)
        conn = sqlite3.connect(tmp_db)
        row = conn.execute("SELECT id FROM checkpoints WHERE id=?", (cid,)).fetchone()
        conn.close()
        assert row is None

    def test_delete_nonexistent_is_noop(self, cm):
        cm.delete("00000000-0000-0000-0000-000000000000")  # should not raise


# ---------------------------------------------------------------------------
# CheckpointManager: cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def _backdate(self, tmp_db: str, cid: str, days: int) -> None:
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "UPDATE checkpoints SET updated_at=? WHERE id=?",
            (time.time() - days * 86400, cid),
        )
        conn.commit()
        conn.close()

    def test_cleanup_removes_old_terminal_records(self, cm, tmp_db):
        cid = cm.create("s", "t", "p")
        cm.complete(cid)
        self._backdate(tmp_db, cid, days=10)
        count = cm.cleanup(older_than_days=7)
        assert count == 1
        conn = sqlite3.connect(tmp_db)
        row = conn.execute("SELECT id FROM checkpoints WHERE id=?", (cid,)).fetchone()
        conn.close()
        assert row is None

    def test_cleanup_preserves_recent_terminal_records(self, cm, tmp_db):
        cid = cm.create("s", "t", "p")
        cm.complete(cid)
        count = cm.cleanup(older_than_days=7)
        assert count == 0

    def test_cleanup_does_not_remove_running(self, cm, tmp_db):
        cid = cm.create("s", "t", "p")
        self._backdate(tmp_db, cid, days=10)
        count = cm.cleanup(older_than_days=7)
        assert count == 0
        incomplete = cm.get_incomplete()
        assert any(c["id"] == cid for c in incomplete)

    def test_cleanup_handles_failed_state(self, cm, tmp_db):
        cid = cm.create("s", "t", "p")
        cm.fail(cid)
        self._backdate(tmp_db, cid, days=10)
        count = cm.cleanup(older_than_days=7)
        assert count == 1

    def test_cleanup_handles_abandoned_state(self, cm, tmp_db):
        cid = cm.create("s", "t", "p")
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "UPDATE checkpoints SET state='ABANDONED', updated_at=? WHERE id=?",
            (time.time() - 10 * 86400, cid),
        )
        conn.commit()
        conn.close()
        count = cm.cleanup(older_than_days=7)
        assert count == 1


# ---------------------------------------------------------------------------
# scan_for_recovery
# ---------------------------------------------------------------------------


class TestScanForRecovery:
    def test_returns_list(self, tmp_db):
        results = scan_for_recovery(db_path=tmp_db)
        assert isinstance(results, list)

    def test_empty_when_no_checkpoints(self, tmp_db):
        results = scan_for_recovery(db_path=tmp_db)
        assert results == []

    def test_returns_recovery_result_objects(self, tmp_db):
        cm = CheckpointManager(db_path=tmp_db)
        cm.create("sess-abc", "task-123", "my prompt")
        results = scan_for_recovery(db_path=tmp_db)
        assert len(results) == 1
        assert isinstance(results[0], RecoveryResult)

    def test_result_fields_populated(self, tmp_db):
        cm = CheckpointManager(db_path=tmp_db)
        cm.create("sess-abc", "task-123", "my prompt")
        results = scan_for_recovery(db_path=tmp_db)
        r = results[0]
        assert r.session_id == "sess-abc"
        assert r.prompt == "my prompt"
        assert r.action in ("resume", "restart", "abandon")

    def test_recent_checkpoint_classified_as_resume(self, tmp_db):
        cm = CheckpointManager(db_path=tmp_db)
        cm.create("s", "t", "p")
        results = scan_for_recovery(db_path=tmp_db)
        assert results[0].action == "resume"

    def test_old_checkpoint_is_abandoned_before_scan(self, tmp_db):
        cm = CheckpointManager(db_path=tmp_db)
        cid = cm.create("s", "t", "p")
        # Backdate to 2 days ago
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "UPDATE checkpoints SET created_at=? WHERE id=?",
            (time.time() - 2 * 86400, cid),
        )
        conn.commit()
        conn.close()
        results = scan_for_recovery(db_path=tmp_db)
        # abandon_old should have transitioned it before get_incomplete is called
        assert len(results) == 0

    def test_complete_checkpoints_not_returned(self, tmp_db):
        cm = CheckpointManager(db_path=tmp_db)
        cid = cm.create("s", "t", "p")
        cm.complete(cid)
        results = scan_for_recovery(db_path=tmp_db)
        assert results == []

    def test_emits_audit_event_per_checkpoint(self, tmp_db):
        event_bus.clear()
        cm = CheckpointManager(db_path=tmp_db)
        cm.create("sess-1", "task-1", "prompt 1")
        cm.create("sess-2", "task-2", "prompt 2")
        scan_for_recovery(db_path=tmp_db)
        events = event_bus.get_events(event_type="agent.checkpoint.recovery_scan")
        assert len(events) == 2

    def test_audit_event_has_correct_category(self, tmp_db):
        cm = CheckpointManager(db_path=tmp_db)
        cm.create("sess-1", "task-1", "prompt")
        event_bus.clear()
        scan_for_recovery(db_path=tmp_db)
        events = event_bus.get_events(event_type="agent.checkpoint.recovery_scan")
        assert events[0].category == "plugin"

    def test_audit_event_detail_contains_action(self, tmp_db):
        cm = CheckpointManager(db_path=tmp_db)
        cm.create("sess-1", "task-1", "prompt")
        event_bus.clear()
        scan_for_recovery(db_path=tmp_db)
        events = event_bus.get_events(event_type="agent.checkpoint.recovery_scan")
        assert "action" in events[0].detail

    def test_graceful_on_missing_db(self, tmp_path):
        # Path that does not yet exist; scan_for_recovery should not raise
        results = scan_for_recovery(db_path=str(tmp_path / "nonexistent" / "cp.db"))
        assert isinstance(results, list)

    def test_loop_messages_propagated(self, tmp_db):
        cm = CheckpointManager(db_path=tmp_db)
        cid = cm.create("s", "t", "p")
        msgs = [{"role": "user", "content": "step 1"}]
        cm.update(cid, msgs, ["tool_a"], iteration=3)
        results = scan_for_recovery(db_path=tmp_db)
        assert results[0].loop_messages == msgs
        assert results[0].iteration == 3


# ---------------------------------------------------------------------------
# RecoveryResult dataclass
# ---------------------------------------------------------------------------


class TestRecoveryResult:
    def test_fields_accessible(self):
        r = RecoveryResult(
            checkpoint_id="abc",
            session_id="sess",
            prompt="do things",
            action="resume",
            loop_messages=[],
            iteration=2,
        )
        assert r.checkpoint_id == "abc"
        assert r.session_id == "sess"
        assert r.prompt == "do things"
        assert r.action == "resume"
        assert r.loop_messages == []
        assert r.iteration == 2

    def test_default_loop_messages_is_empty_list(self):
        r = RecoveryResult(
            checkpoint_id="x",
            session_id="s",
            prompt="p",
            action="restart",
        )
        assert r.loop_messages == []

    def test_default_iteration_is_zero(self):
        r = RecoveryResult(
            checkpoint_id="x",
            session_id="s",
            prompt="p",
            action="restart",
        )
        assert r.iteration == 0
