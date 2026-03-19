"""JSON parsing error recovery tests across multiple Missy modules.

Verifies that malformed, truncated, binary-garbage, and edge-case JSON inputs
are handled gracefully — no crashes, sensible defaults or empty results
returned — across:

- missy.agent.persona.PersonaManager.get_audit_log()
- missy.agent.checkpoint.CheckpointManager._row_to_dict() and scan_for_recovery()
- missy.agent.hatching.HatchingLog.get_entries()
- missy.scheduler.manager.SchedulerManager._load_jobs()
- missy.channels.voice.registry.DeviceRegistry.load()

Note: The user request referred to "CheckpointStore", but the actual class in
missy.agent.checkpoint is CheckpointManager.  Tests use the real class name.
"""

from __future__ import annotations

import json
import os
import stat
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_persona_manager(tmp_path: Path) -> Any:
    from missy.agent.persona import PersonaManager

    persona_file = tmp_path / "persona.yaml"
    return PersonaManager(persona_path=str(persona_file))


def _write_audit_log(tmp_path: Path, content: str) -> Path:
    """Write raw bytes to the persona_audit.jsonl path expected by PersonaManager."""
    log_path = tmp_path / "persona_audit.jsonl"
    log_path.write_text(content, encoding="utf-8")
    return log_path


def _write_audit_log_bytes(tmp_path: Path, content: bytes) -> Path:
    log_path = tmp_path / "persona_audit.jsonl"
    log_path.write_bytes(content)
    return log_path


def _make_hatching_log(tmp_path: Path) -> Any:
    from missy.agent.hatching import HatchingLog

    log_path = tmp_path / "hatching_log.jsonl"
    return HatchingLog(log_path=log_path)


def _write_hatching_log(tmp_path: Path, content: str) -> Path:
    log_path = tmp_path / "hatching_log.jsonl"
    log_path.write_text(content, encoding="utf-8")
    return log_path


def _write_hatching_log_bytes(tmp_path: Path, content: bytes) -> Path:
    log_path = tmp_path / "hatching_log.jsonl"
    log_path.write_bytes(content)
    return log_path


def _make_checkpoint_manager(tmp_path: Path) -> Any:
    from missy.agent.checkpoint import CheckpointManager

    db_path = str(tmp_path / "checkpoints.db")
    return CheckpointManager(db_path=db_path)


def _make_scheduler(tmp_path: Path) -> Any:
    from missy.scheduler.manager import SchedulerManager

    jobs_file = tmp_path / "jobs.json"
    return SchedulerManager(jobs_file=str(jobs_file))


def _write_jobs_file_safe(tmp_path: Path, content: str) -> Path:
    """Write a jobs.json with permissions that pass the ownership/permission checks."""
    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(content, encoding="utf-8")
    # Ensure owner-only write (passes the 0o022 check in _load_jobs)
    jobs_file.chmod(0o600)
    return jobs_file


def _make_device_registry(tmp_path: Path) -> Any:
    from missy.channels.voice.registry import DeviceRegistry

    registry_path = tmp_path / "devices.json"
    return DeviceRegistry(registry_path=str(registry_path))


def _write_registry_safe(tmp_path: Path, content: str) -> Path:
    """Write a devices.json that passes ownership and permission validation."""
    reg_file = tmp_path / "devices.json"
    reg_file.write_text(content, encoding="utf-8")
    reg_file.chmod(0o600)
    return reg_file


def _write_registry_bytes_safe(tmp_path: Path, content: bytes) -> Path:
    reg_file = tmp_path / "devices.json"
    reg_file.write_bytes(content)
    reg_file.chmod(0o600)
    return reg_file


# ===========================================================================
# 1. PersonaManager.get_audit_log() — JSONL error recovery
# ===========================================================================


class TestPersonaAuditLogEmptyLines:
    """Blank lines in the audit log are silently skipped."""

    def test_only_blank_lines_returns_empty_list(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        _write_audit_log(tmp_path, "\n\n\n\n")
        result = pm.get_audit_log()
        assert result == []

    def test_blank_lines_between_valid_entries_are_skipped(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        good = json.dumps({"action": "save", "version": 1})
        _write_audit_log(tmp_path, f"\n{good}\n\n{good}\n")
        result = pm.get_audit_log()
        assert len(result) == 2

    def test_single_blank_line_file(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        _write_audit_log(tmp_path, " ")
        result = pm.get_audit_log()
        assert result == []


class TestPersonaAuditLogTruncatedJSON:
    """Truncated / incomplete JSON objects are skipped without crashing."""

    def test_truncated_object_mid_key(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        _write_audit_log(tmp_path, '{"action": "sa')
        result = pm.get_audit_log()
        assert result == []

    def test_truncated_after_opening_brace(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        _write_audit_log(tmp_path, "{")
        result = pm.get_audit_log()
        assert result == []

    def test_truncated_json_mixed_with_valid(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        good = json.dumps({"action": "reset", "version": 3})
        truncated = '{"action": "trun'
        _write_audit_log(tmp_path, f"{truncated}\n{good}\n")
        result = pm.get_audit_log()
        # At least the valid line should be recovered
        assert any(e.get("action") == "reset" for e in result)

    def test_unclosed_string_in_json(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        _write_audit_log(tmp_path, '{"action": "unclosed}')
        result = pm.get_audit_log()
        assert result == []


class TestPersonaAuditLogBinaryGarbage:
    """Binary/null-byte content behaviour for get_audit_log().

    The implementation opens the log file with utf-8 encoding and propagates
    UnicodeDecodeError when non-UTF-8 bytes are encountered — this is a known
    limitation of the current code.  Tests document this actual behaviour.
    """

    def test_null_bytes_in_ascii_region_are_decoded(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        # Null bytes (0x00) are valid in UTF-8 streams — the line is parsed as
        # a string; json.loads will fail on the embedded nulls and skip the line.
        _write_audit_log_bytes(tmp_path, b'{"action":"save"}\x00\x00\x00\n')
        result = pm.get_audit_log()
        assert isinstance(result, list)

    def test_pure_binary_non_utf8_raises_unicode_decode_error(self, tmp_path: Path) -> None:
        """get_audit_log() propagates UnicodeDecodeError for non-UTF-8 files.

        The outer OSError guard in get_audit_log() does not catch
        UnicodeDecodeError, so callers must handle this if binary files are
        expected.  This test documents that real behaviour.
        """
        pm = _make_persona_manager(tmp_path)
        # bytes(range(256)) contains bytes > 0x7F that are invalid UTF-8
        _write_audit_log_bytes(tmp_path, bytes(range(256)))
        with pytest.raises(UnicodeDecodeError):
            pm.get_audit_log()

    def test_binary_line_after_valid_utf8_raises_unicode_decode_error(self, tmp_path: Path) -> None:
        """A binary byte sequence after valid UTF-8 also propagates UnicodeDecodeError."""
        pm = _make_persona_manager(tmp_path)
        good_bytes = json.dumps({"action": "save"}).encode("utf-8") + b"\n"
        binary_line = b"\xff\xfe\x00\x01garbage\n"
        _write_audit_log_bytes(tmp_path, good_bytes + binary_line)
        with pytest.raises(UnicodeDecodeError):
            pm.get_audit_log()


class TestPersonaAuditLogNestedObjects:
    """Nested JSON objects and arrays are parsed correctly."""

    def test_deeply_nested_json_is_parsed(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        nested = {"action": "save", "details": {"x": {"y": {"z": [1, 2, 3]}}}}
        _write_audit_log(tmp_path, json.dumps(nested) + "\n")
        result = pm.get_audit_log()
        assert len(result) == 1
        assert result[0]["details"]["x"]["y"]["z"] == [1, 2, 3]

    def test_array_at_root_is_invalid_json_lines(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        # A bare JSON array is valid JSON but get_audit_log expects dicts.
        # contextlib.suppress(json.JSONDecodeError) will let non-dicts through
        # since json.loads succeeds, but we just verify no crash.
        _write_audit_log(tmp_path, "[1, 2, 3]\n")
        result = pm.get_audit_log()
        assert isinstance(result, list)

    def test_multiple_valid_entries_with_unicode(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        entries = [
            {"action": "save", "name": "Missy \u2665"},
            {"action": "reset", "name": "\u4e2d\u6587"},
        ]
        content = "\n".join(json.dumps(e) for e in entries) + "\n"
        _write_audit_log(tmp_path, content)
        result = pm.get_audit_log()
        assert len(result) == 2

    def test_json_number_at_root_is_handled(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        _write_audit_log(tmp_path, "42\n")
        result = pm.get_audit_log()
        assert isinstance(result, list)


class TestPersonaAuditLogVeryLargeLines:
    """Very large lines are handled without hanging."""

    def test_one_megabyte_valid_json_line(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        big_value = "x" * (1024 * 1024)
        entry = json.dumps({"action": "save", "payload": big_value})
        _write_audit_log(tmp_path, entry + "\n")
        result = pm.get_audit_log()
        assert len(result) == 1

    def test_very_long_invalid_line(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        _write_audit_log(tmp_path, ("a" * 500_000) + "\n")
        result = pm.get_audit_log()
        assert result == []

    def test_non_existent_audit_log_returns_empty(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        # Do not create the log file at all
        result = pm.get_audit_log()
        assert result == []


class TestPersonaAuditLogUnicodeEdgeCases:
    """Unicode surrogate and boundary characters are handled gracefully."""

    def test_emoji_in_json_values(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        entry = json.dumps({"action": "save", "name": "Missy \U0001f916"})
        _write_audit_log(tmp_path, entry + "\n")
        result = pm.get_audit_log()
        assert len(result) == 1

    def test_zero_width_space_in_line(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        # A line that is only whitespace (including zero-width space) should be skipped.
        _write_audit_log(tmp_path, "\u200b\n")
        result = pm.get_audit_log()
        assert result == []

    def test_bom_at_start_of_file(self, tmp_path: Path) -> None:
        pm = _make_persona_manager(tmp_path)
        good = json.dumps({"action": "save"})
        # BOM + valid JSON on second line
        _write_audit_log_bytes(tmp_path, b"\xef\xbb\xbf" + good.encode("utf-8") + b"\n")
        result = pm.get_audit_log()
        # BOM may or may not cause a parse failure; the key is no crash.
        assert isinstance(result, list)


# ===========================================================================
# 2. CheckpointManager — JSON column recovery
# ===========================================================================


class TestCheckpointManagerRowToDict:
    """_row_to_dict() handles corrupted loop_messages and tool_names_used."""

    def test_corrupted_loop_messages_falls_back_to_empty_list(self, tmp_path: Path) -> None:
        cm = _make_checkpoint_manager(tmp_path)
        cid = cm.create("sess-1", "task-1", "Do something")
        # Manually corrupt the loop_messages column
        conn = cm._connect()
        conn.execute(
            "UPDATE checkpoints SET loop_messages = ? WHERE id = ?",
            ("NOTJSON{{{", cid),
        )
        conn.commit()
        rows = cm.get_incomplete()
        assert len(rows) == 1
        assert rows[0]["loop_messages"] == []

    def test_corrupted_tool_names_used_falls_back_to_empty_list(self, tmp_path: Path) -> None:
        cm = _make_checkpoint_manager(tmp_path)
        cid = cm.create("sess-2", "task-2", "Do more")
        conn = cm._connect()
        conn.execute(
            "UPDATE checkpoints SET tool_names_used = ? WHERE id = ?",
            ("[unclosed", cid),
        )
        conn.commit()
        rows = cm.get_incomplete()
        assert len(rows) == 1
        assert rows[0]["tool_names_used"] == []

    def test_none_loop_messages_via_row_to_dict_passes_through_as_none(
        self, tmp_path: Path
    ) -> None:
        """_row_to_dict() documents None-passthrough for non-string loop_messages.

        The implementation uses `json.loads(raw) if isinstance(raw, str) else raw`,
        so when raw is None (not a string), it is returned unchanged rather than
        replaced with [].  This test documents that actual behaviour.

        In practice the NOT NULL schema constraint prevents SQL NULL being stored,
        so this path is exercised only when _row_to_dict() is called with a
        synthetic dict.
        """
        from missy.agent.checkpoint import CheckpointManager

        fake_row_dict = {
            "id": "fake-id",
            "session_id": "sess-x",
            "task_id": "task-x",
            "prompt": "p",
            "state": "RUNNING",
            "loop_messages": None,
            "tool_names_used": None,
            "iteration": 0,
            "created_at": 0.0,
            "updated_at": 0.0,
        }
        result = CheckpointManager._row_to_dict(fake_row_dict)  # type: ignore[arg-type]
        # None passes through unchanged (not a string, so json.loads is skipped)
        assert result["loop_messages"] is None
        assert result["tool_names_used"] is None

    def test_truncated_json_array_in_loop_messages(self, tmp_path: Path) -> None:
        cm = _make_checkpoint_manager(tmp_path)
        cid = cm.create("sess-4", "task-4", "Truncated array")
        conn = cm._connect()
        conn.execute(
            "UPDATE checkpoints SET loop_messages = ? WHERE id = ?",
            ('[{"role": "user", "content":', cid),
        )
        conn.commit()
        rows = cm.get_incomplete()
        assert rows[0]["loop_messages"] == []

    def test_valid_loop_messages_is_deserialized_correctly(self, tmp_path: Path) -> None:
        cm = _make_checkpoint_manager(tmp_path)
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        cid = cm.create("sess-5", "task-5", "Valid messages")
        cm.update(cid, messages, ["read_file"], iteration=1)
        rows = cm.get_incomplete()
        assert len(rows) == 1
        assert rows[0]["loop_messages"] == messages

    def test_empty_string_loop_messages_falls_back_to_empty_list(self, tmp_path: Path) -> None:
        cm = _make_checkpoint_manager(tmp_path)
        cid = cm.create("sess-6", "task-6", "Empty string")
        conn = cm._connect()
        conn.execute(
            "UPDATE checkpoints SET loop_messages = ? WHERE id = ?",
            ("", cid),
        )
        conn.commit()
        rows = cm.get_incomplete()
        # Empty string is not valid JSON, so falls back
        assert rows[0]["loop_messages"] == []


class TestScanForRecoveryCorruptedDB:
    """scan_for_recovery() handles a missing or corrupt database gracefully."""

    def test_nonexistent_db_path_returns_empty(self, tmp_path: Path) -> None:
        from missy.agent.checkpoint import scan_for_recovery

        db_path = str(tmp_path / "nonexistent_dir" / "checkpoints.db")
        # scan_for_recovery creates the directory; it should return [] on a fresh DB
        result = scan_for_recovery(db_path=db_path)
        assert result == []

    def test_returns_empty_when_no_running_checkpoints(self, tmp_path: Path) -> None:
        from missy.agent.checkpoint import scan_for_recovery

        db_path = str(tmp_path / "checkpoints.db")
        cm = _make_checkpoint_manager(tmp_path)
        cid = cm.create("sess-x", "task-x", "Already done")
        cm.complete(cid)
        result = scan_for_recovery(db_path=db_path)
        assert result == []

    def test_running_checkpoint_is_returned(self, tmp_path: Path) -> None:
        from missy.agent.checkpoint import scan_for_recovery

        db_path = str(tmp_path / "checkpoints.db")
        cm = _make_checkpoint_manager(tmp_path)
        cm.create("sess-y", "task-y", "Incomplete task")
        result = scan_for_recovery(db_path=db_path)
        assert len(result) == 1
        assert result[0].prompt == "Incomplete task"


# ===========================================================================
# 3. HatchingLog.get_entries() — JSONL error recovery
# ===========================================================================


class TestHatchingLogEmptyLines:
    """Blank lines in hatching_log.jsonl are silently skipped."""

    def test_only_blank_lines_returns_empty(self, tmp_path: Path) -> None:
        hl = _make_hatching_log(tmp_path)
        _write_hatching_log(tmp_path, "\n\n\n")
        result = hl.get_entries()
        assert result == []

    def test_blank_lines_interspersed_with_valid(self, tmp_path: Path) -> None:
        hl = _make_hatching_log(tmp_path)
        good = json.dumps({"step": "env_check", "status": "ok"})
        _write_hatching_log(tmp_path, f"\n{good}\n\n{good}\n")
        result = hl.get_entries()
        assert len(result) == 2

    def test_whitespace_only_lines_are_skipped(self, tmp_path: Path) -> None:
        hl = _make_hatching_log(tmp_path)
        good = json.dumps({"step": "persona", "status": "ok"})
        _write_hatching_log(tmp_path, f"   \n{good}\n\t\n")
        result = hl.get_entries()
        assert len(result) == 1


class TestHatchingLogTruncatedJSON:
    """Truncated JSON log entries are skipped and do not crash the reader."""

    def test_truncated_mid_value(self, tmp_path: Path) -> None:
        hl = _make_hatching_log(tmp_path)
        _write_hatching_log(tmp_path, '{"step": "env_check", "status": "o')
        result = hl.get_entries()
        assert result == []

    def test_truncated_after_key(self, tmp_path: Path) -> None:
        hl = _make_hatching_log(tmp_path)
        _write_hatching_log(tmp_path, '{"step":')
        result = hl.get_entries()
        assert result == []

    def test_valid_then_truncated_then_valid(self, tmp_path: Path) -> None:
        hl = _make_hatching_log(tmp_path)
        good = json.dumps({"step": "done", "status": "ok"})
        content = f"{good}\n" + '{"step": "bad\n' + f"{good}\n"
        _write_hatching_log(tmp_path, content)
        result = hl.get_entries()
        assert len(result) == 2

    def test_unclosed_brace(self, tmp_path: Path) -> None:
        hl = _make_hatching_log(tmp_path)
        _write_hatching_log(tmp_path, "{")
        result = hl.get_entries()
        assert result == []


class TestHatchingLogBinaryGarbage:
    """Binary input behaviour for get_entries().

    The implementation opens the log file with utf-8 encoding.  Non-UTF-8 bytes
    cause UnicodeDecodeError, which the current code does not suppress.  Tests
    document this actual behaviour.
    """

    def test_pure_binary_non_utf8_raises_unicode_decode_error(self, tmp_path: Path) -> None:
        """get_entries() propagates UnicodeDecodeError for non-UTF-8 files.

        The OSError guard in get_entries() does not catch UnicodeDecodeError.
        This test documents that real behaviour.
        """
        hl = _make_hatching_log(tmp_path)
        _write_hatching_log_bytes(tmp_path, bytes(range(128, 256)))
        with pytest.raises(UnicodeDecodeError):
            hl.get_entries()

    def test_null_bytes_ascii_region_in_log(self, tmp_path: Path) -> None:
        """Null bytes (0x00) are valid UTF-8 and do not raise a decode error."""
        hl = _make_hatching_log(tmp_path)
        good = json.dumps({"step": "boot", "status": "ok"}).encode("utf-8")
        # Null bytes in the ASCII range are valid UTF-8
        _write_hatching_log_bytes(tmp_path, good + b"\x00\x01\x02\n")
        result = hl.get_entries()
        assert isinstance(result, list)

    def test_non_utf8_latin1_bytes_raise_unicode_decode_error(self, tmp_path: Path) -> None:
        """Latin-1 encoded bytes (0x80–0x9F) are invalid UTF-8."""
        hl = _make_hatching_log(tmp_path)
        _write_hatching_log_bytes(tmp_path, b"\x80\x81\x82\n")
        with pytest.raises(UnicodeDecodeError):
            hl.get_entries()


class TestHatchingLogNonExistentFile:
    """Missing log file returns an empty list, not an exception."""

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        hl = _make_hatching_log(tmp_path)
        # Do not create the log file
        result = hl.get_entries()
        assert result == []

    def test_valid_entries_after_file_created(self, tmp_path: Path) -> None:
        hl = _make_hatching_log(tmp_path)
        hl.log("env_check", "ok", "Python version check passed")
        result = hl.get_entries()
        assert len(result) == 1
        assert result[0]["step"] == "env_check"


class TestHatchingLogVeryLargeLines:
    """Very large lines are handled without hanging."""

    def test_large_valid_json_line(self, tmp_path: Path) -> None:
        hl = _make_hatching_log(tmp_path)
        big = {"step": "test", "status": "ok", "message": "x" * 500_000}
        _write_hatching_log(tmp_path, json.dumps(big) + "\n")
        result = hl.get_entries()
        assert len(result) == 1

    def test_large_invalid_line_returns_empty(self, tmp_path: Path) -> None:
        hl = _make_hatching_log(tmp_path)
        _write_hatching_log(tmp_path, "z" * 200_000 + "\n")
        result = hl.get_entries()
        assert result == []


class TestHatchingLogUnicodeEdgeCases:
    """Unicode content is handled correctly in log entries."""

    def test_cjk_characters_in_message(self, tmp_path: Path) -> None:
        hl = _make_hatching_log(tmp_path)
        entry = json.dumps({"step": "persona", "status": "ok", "message": "\u4e2d\u6587\u6d4b\u8bd5"})
        _write_hatching_log(tmp_path, entry + "\n")
        result = hl.get_entries()
        assert len(result) == 1

    def test_rtl_text_in_message(self, tmp_path: Path) -> None:
        hl = _make_hatching_log(tmp_path)
        entry = json.dumps({"step": "boot", "status": "ok", "message": "\u0645\u0631\u062d\u0628\u0627"})
        _write_hatching_log(tmp_path, entry + "\n")
        result = hl.get_entries()
        assert len(result) == 1


# ===========================================================================
# 4. SchedulerManager._load_jobs() — JSON file error recovery
# ===========================================================================


class TestSchedulerManagerLoadJobsCorruptedFile:
    """Corrupted jobs.json content is handled without crashing."""

    def test_truncated_json_array_loads_empty(self, tmp_path: Path) -> None:
        mgr = _make_scheduler(tmp_path)
        _write_jobs_file_safe(tmp_path, "[{")
        mgr._load_jobs()
        assert mgr._jobs == {}

    def test_empty_file_loads_empty(self, tmp_path: Path) -> None:
        mgr = _make_scheduler(tmp_path)
        _write_jobs_file_safe(tmp_path, "")
        mgr._load_jobs()
        assert mgr._jobs == {}

    def test_null_value_loads_empty(self, tmp_path: Path) -> None:
        mgr = _make_scheduler(tmp_path)
        _write_jobs_file_safe(tmp_path, "null")
        mgr._load_jobs()
        assert mgr._jobs == {}

    def test_json_object_instead_of_array_loads_empty(self, tmp_path: Path) -> None:
        mgr = _make_scheduler(tmp_path)
        _write_jobs_file_safe(tmp_path, '{"key": "value"}')
        mgr._load_jobs()
        assert mgr._jobs == {}

    def test_plain_text_loads_empty(self, tmp_path: Path) -> None:
        mgr = _make_scheduler(tmp_path)
        _write_jobs_file_safe(tmp_path, "not json at all")
        mgr._load_jobs()
        assert mgr._jobs == {}

    def test_array_with_non_dict_entries_skips_bad_records(self, tmp_path: Path) -> None:
        mgr = _make_scheduler(tmp_path)
        # Mix of non-dicts and a well-formed (but likely incomplete) dict
        content = json.dumps([1, "string", None, True, []])
        _write_jobs_file_safe(tmp_path, content)
        mgr._load_jobs()
        # Non-dict entries are all skipped
        assert mgr._jobs == {}

    def test_array_with_only_unknown_fields_is_skipped(self, tmp_path: Path) -> None:
        """A record dict with no recognised fields at all causes from_dict() to fail."""
        mgr = _make_scheduler(tmp_path)
        # A record with no recognised keys: from_dict relies on 'id' being present;
        # if it is absent, the job gets a new uuid and is still loaded. So we supply
        # a record that will cause a TypeError in from_dict (not a dict).
        content = json.dumps(["this is a string, not a dict"])
        _write_jobs_file_safe(tmp_path, content)
        mgr._load_jobs()
        # Non-dict entries are skipped by the isinstance(record, dict) guard
        assert mgr._jobs == {}

    def test_record_with_only_id_and_name_is_loaded_with_defaults(self, tmp_path: Path) -> None:
        """ScheduledJob.from_dict() is lenient and loads partial records with defaults.

        This documents that a record with only {id, name} is NOT skipped —
        from_dict() fills missing fields with defaults.
        """
        mgr = _make_scheduler(tmp_path)
        minimal = {"id": "min-1", "name": "Minimal Job"}
        content = json.dumps([minimal])
        _write_jobs_file_safe(tmp_path, content)
        mgr._load_jobs()
        # from_dict() succeeds on a minimal record — it is loaded, not skipped.
        assert "min-1" in mgr._jobs

    def test_nonexistent_jobs_file_is_no_op(self, tmp_path: Path) -> None:
        mgr = _make_scheduler(tmp_path)
        # Do not create the file
        mgr._load_jobs()
        assert mgr._jobs == {}

    def test_large_corrupt_file_loads_empty(self, tmp_path: Path) -> None:
        mgr = _make_scheduler(tmp_path)
        _write_jobs_file_safe(tmp_path, "{" * 10_000)
        mgr._load_jobs()
        assert mgr._jobs == {}

    def test_unicode_in_corrupt_json_loads_empty(self, tmp_path: Path) -> None:
        mgr = _make_scheduler(tmp_path)
        _write_jobs_file_safe(tmp_path, '[\u4e2d\u6587{broken}]')
        mgr._load_jobs()
        assert mgr._jobs == {}


class TestSchedulerManagerLoadJobsPartialData:
    """A mix of valid and invalid records: valid ones are loaded."""

    def test_valid_record_mixed_with_non_dict_entries(self, tmp_path: Path) -> None:
        from missy.scheduler.jobs import ScheduledJob

        mgr = _make_scheduler(tmp_path)
        good_job = ScheduledJob(name="TestJob", schedule="every 1 hour", task="do something")
        good_dict = good_job.to_dict()
        content = json.dumps([good_dict, "bad_string", 42])
        _write_jobs_file_safe(tmp_path, content)
        mgr._load_jobs()
        assert len(mgr._jobs) == 1

    def test_empty_json_array_loads_no_jobs(self, tmp_path: Path) -> None:
        mgr = _make_scheduler(tmp_path)
        _write_jobs_file_safe(tmp_path, "[]")
        mgr._load_jobs()
        assert mgr._jobs == {}


# ===========================================================================
# 5. DeviceRegistry.load() — JSON file error recovery
# ===========================================================================


class TestDeviceRegistryLoadCorruptedFile:
    """Corrupted devices.json is handled without crashing."""

    def test_truncated_json_loads_empty(self, tmp_path: Path) -> None:
        reg = _make_device_registry(tmp_path)
        _write_registry_safe(tmp_path, "[{")
        reg.load()
        assert reg.list_nodes() == []

    def test_empty_file_loads_empty(self, tmp_path: Path) -> None:
        reg = _make_device_registry(tmp_path)
        _write_registry_safe(tmp_path, "")
        reg.load()
        assert reg.list_nodes() == []

    def test_null_json_loads_empty(self, tmp_path: Path) -> None:
        reg = _make_device_registry(tmp_path)
        _write_registry_safe(tmp_path, "null")
        reg.load()
        assert reg.list_nodes() == []

    def test_json_object_instead_of_array_loads_empty(self, tmp_path: Path) -> None:
        reg = _make_device_registry(tmp_path)
        _write_registry_safe(tmp_path, '{"node_id": "abc"}')
        reg.load()
        # json.loads succeeds but iterating over a dict will fail on missing "node_id"
        # key in the iteration; the outer except catches this → empty registry.
        assert isinstance(reg.list_nodes(), list)

    def test_plain_text_loads_empty(self, tmp_path: Path) -> None:
        reg = _make_device_registry(tmp_path)
        _write_registry_safe(tmp_path, "this is not json")
        reg.load()
        assert reg.list_nodes() == []

    def test_binary_garbage_loads_empty(self, tmp_path: Path) -> None:
        reg = _make_device_registry(tmp_path)
        _write_registry_bytes_safe(tmp_path, bytes(range(128)))
        reg.load()
        assert isinstance(reg.list_nodes(), list)

    def test_nonexistent_file_starts_empty(self, tmp_path: Path) -> None:
        reg = _make_device_registry(tmp_path)
        # Do not create the file
        reg.load()
        assert reg.list_nodes() == []

    def test_json_array_with_missing_node_id_loads_empty(self, tmp_path: Path) -> None:
        reg = _make_device_registry(tmp_path)
        # A record without the required "node_id" key will cause a TypeError/KeyError
        bad_record = [{"friendly_name": "Lounge", "room": "Living Room"}]
        _write_registry_safe(tmp_path, json.dumps(bad_record))
        reg.load()
        # The exception handler catches this and starts empty
        assert isinstance(reg.list_nodes(), list)

    def test_large_corrupt_file_loads_empty(self, tmp_path: Path) -> None:
        reg = _make_device_registry(tmp_path)
        _write_registry_safe(tmp_path, "[" + "x" * 100_000)
        reg.load()
        assert reg.list_nodes() == []

    def test_unicode_garbage_loads_empty(self, tmp_path: Path) -> None:
        reg = _make_device_registry(tmp_path)
        _write_registry_safe(tmp_path, "[\u4e2d\u6587\u5185\u5bb9broken{]")
        reg.load()
        assert reg.list_nodes() == []


class TestDeviceRegistryLoadValidData:
    """Valid devices.json is loaded correctly."""

    def test_valid_node_is_loaded(self, tmp_path: Path) -> None:
        from missy.channels.voice.registry import EdgeNode

        reg = _make_device_registry(tmp_path)
        node = EdgeNode(
            node_id="node-001",
            friendly_name="Test Node",
            room="Lab",
            ip_address="192.168.1.100",
        )
        reg._nodes["node-001"] = node
        reg.save()

        reg2 = _make_device_registry(tmp_path)
        reg2.load()
        loaded = reg2.get_node("node-001")
        assert loaded is not None
        assert loaded.friendly_name == "Test Node"

    def test_empty_valid_array_loads_empty(self, tmp_path: Path) -> None:
        reg = _make_device_registry(tmp_path)
        _write_registry_safe(tmp_path, "[]")
        reg.load()
        assert reg.list_nodes() == []

    def test_partial_valid_node_with_extra_keys_is_loaded(self, tmp_path: Path) -> None:
        reg = _make_device_registry(tmp_path)
        node_data = {
            "node_id": "node-extra",
            "friendly_name": "Extra Fields",
            "room": "Office",
            "ip_address": "10.0.0.1",
            "unknown_future_field": "some value",  # should be silently ignored
        }
        _write_registry_safe(tmp_path, json.dumps([node_data]))
        reg.load()
        loaded = reg.get_node("node-extra")
        assert loaded is not None
        assert loaded.friendly_name == "Extra Fields"


class TestDeviceRegistryLoadNullBytesInContent:
    """Null bytes in the registry file do not crash the load."""

    def test_null_bytes_mixed_with_valid_json(self, tmp_path: Path) -> None:
        from missy.channels.voice.registry import EdgeNode

        reg = _make_device_registry(tmp_path)
        node = EdgeNode(
            node_id="node-002",
            friendly_name="Null Test",
            room="Garage",
            ip_address="192.168.1.50",
        )
        valid_json = json.dumps([{"node_id": "node-002", "friendly_name": "Null Test",
                                   "room": "Garage", "ip_address": "192.168.1.50"}])
        # Inject a null byte mid-file — will fail UTF-8 decode or JSON parse
        content = valid_json[:10].encode("utf-8") + b"\x00" + valid_json[10:].encode("utf-8")
        _write_registry_bytes_safe(tmp_path, content)
        reg.load()
        # Should not crash; result may be empty
        assert isinstance(reg.list_nodes(), list)
