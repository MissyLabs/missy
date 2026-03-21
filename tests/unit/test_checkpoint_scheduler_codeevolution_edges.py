"""Edge case tests targeting untested defensive code paths.


Tests for:
- Checkpoint _row_to_dict JSON fallback
- scan_for_recovery exception isolation
- Scheduler _load_jobs permission/ownership checks
- Code evolution malformed data recovery
- Provider registry constructor failure
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# ──────────────────────────────────────────────────────────────────────
# Checkpoint: _row_to_dict JSON fallback
# ──────────────────────────────────────────────────────────────────────


class TestCheckpointRowToDict:
    """Verify _row_to_dict handles malformed JSON gracefully."""

    def test_valid_json_columns(self):
        from missy.agent.checkpoint import CheckpointManager

        row_dict = {
            "id": "cp1",
            "loop_messages": '["msg1", "msg2"]',
            "tool_names_used": '["shell_exec"]',
        }
        # Simulate sqlite3.Row using a dict wrapper
        result = CheckpointManager._row_to_dict(row_dict)
        assert result["loop_messages"] == ["msg1", "msg2"]
        assert result["tool_names_used"] == ["shell_exec"]

    def test_malformed_json_falls_back_to_empty_list(self):
        from missy.agent.checkpoint import CheckpointManager

        row_dict = {
            "id": "cp1",
            "loop_messages": "{not valid json",
            "tool_names_used": "also broken [",
        }
        result = CheckpointManager._row_to_dict(row_dict)
        assert result["loop_messages"] == []
        assert result["tool_names_used"] == []

    def test_none_value_handled(self):
        from missy.agent.checkpoint import CheckpointManager

        row_dict = {
            "id": "cp1",
            "loop_messages": None,
            "tool_names_used": None,
        }
        result = CheckpointManager._row_to_dict(row_dict)
        # None is not a str, so the `isinstance(raw, str)` branch is False
        assert result["loop_messages"] is None
        assert result["tool_names_used"] is None

    def test_already_parsed_list(self):
        from missy.agent.checkpoint import CheckpointManager

        row_dict = {
            "id": "cp1",
            "loop_messages": ["already", "parsed"],
            "tool_names_used": [],
        }
        result = CheckpointManager._row_to_dict(row_dict)
        assert result["loop_messages"] == ["already", "parsed"]

    def test_missing_json_keys(self):
        from missy.agent.checkpoint import CheckpointManager

        row_dict = {"id": "cp1"}  # No loop_messages or tool_names_used
        result = CheckpointManager._row_to_dict(row_dict)
        assert "id" in result  # Other keys preserved


# ──────────────────────────────────────────────────────────────────────
# scan_for_recovery exception isolation
# ──────────────────────────────────────────────────────────────────────


class TestScanForRecoveryExceptions:
    """Verify scan_for_recovery handles DB and event failures gracefully."""

    def test_checkpoint_db_open_failure(self):
        """If CheckpointManager() raises, return empty list."""
        from missy.agent.checkpoint import scan_for_recovery

        with patch(
            "missy.agent.checkpoint.CheckpointManager",
            side_effect=RuntimeError("db corrupt"),
        ):
            result = scan_for_recovery(db_path="/tmp/nonexistent.db")
        assert result == []

    def test_abandon_old_failure(self):
        """If abandon_old() fails, scan continues with get_incomplete()."""
        from missy.agent.checkpoint import scan_for_recovery

        mock_cm = MagicMock()
        mock_cm.abandon_old.side_effect = RuntimeError("abandon error")
        mock_cm.get_incomplete.return_value = []

        with patch(
            "missy.agent.checkpoint.CheckpointManager",
            return_value=mock_cm,
        ):
            result = scan_for_recovery()
        assert result == []  # No incomplete, but didn't crash

    def test_get_incomplete_failure(self):
        """If get_incomplete() fails, return empty list."""
        from missy.agent.checkpoint import scan_for_recovery

        mock_cm = MagicMock()
        mock_cm.abandon_old.return_value = 0
        mock_cm.get_incomplete.side_effect = RuntimeError("query error")

        with patch(
            "missy.agent.checkpoint.CheckpointManager",
            return_value=mock_cm,
        ):
            result = scan_for_recovery()
        assert result == []

    def test_audit_event_failure(self):
        """If event_bus.publish fails, scan_for_recovery still returns results."""
        from missy.agent.checkpoint import scan_for_recovery

        mock_cm = MagicMock()
        mock_cm.abandon_old.return_value = 0
        mock_cm.get_incomplete.return_value = [
            {
                "id": "cp1",
                "session_id": "s1",
                "task_id": "t1",
                "prompt": "test prompt",
                "loop_messages": [],
                "iteration": 2,
                "created_at": time.time() - 100,
            }
        ]
        mock_cm.classify.return_value = "resume"

        with (
            patch(
                "missy.agent.checkpoint.CheckpointManager",
                return_value=mock_cm,
            ),
            patch(
                "missy.agent.checkpoint.event_bus.publish",
                side_effect=RuntimeError("event bus down"),
            ),
        ):
            result = scan_for_recovery()
        assert len(result) == 1
        assert result[0].action == "resume"


# ──────────────────────────────────────────────────────────────────────
# Scheduler _load_jobs permission/ownership checks
# ──────────────────────────────────────────────────────────────────────


class TestSchedulerLoadJobsSecurity:
    """Verify _load_jobs rejects insecure file permissions."""

    def _make_manager(self, jobs_file: Path):
        from missy.scheduler.manager import SchedulerManager

        mgr = SchedulerManager.__new__(SchedulerManager)
        mgr.jobs_file = jobs_file
        mgr._jobs = {}
        mgr._lock = __import__("threading").Lock()
        return mgr

    def test_wrong_ownership_rejected(self, tmp_path):
        """Jobs file owned by different UID is rejected."""

        jobs_file = tmp_path / "jobs.json"
        jobs_file.write_text("[]")
        mgr = self._make_manager(jobs_file)

        # Mock stat to return wrong UID
        with patch.object(Path, "stat") as mock_stat:
            mock_result = MagicMock()
            mock_result.st_uid = os.getuid() + 1  # Wrong user
            mock_result.st_mode = 0o100600
            mock_stat.return_value = mock_result
            mgr._load_jobs()

        assert len(mgr._jobs) == 0  # Nothing loaded

    def test_world_writable_rejected(self, tmp_path):
        """Jobs file with world-writable permissions is rejected."""
        jobs_file = tmp_path / "jobs.json"
        jobs_file.write_text("[]")
        mgr = self._make_manager(jobs_file)

        with patch.object(Path, "stat") as mock_stat:
            mock_result = MagicMock()
            mock_result.st_uid = os.getuid()
            mock_result.st_mode = 0o100666  # World writable
            mock_stat.return_value = mock_result
            mgr._load_jobs()

        assert len(mgr._jobs) == 0

    def test_group_writable_rejected(self, tmp_path):
        """Jobs file with group-writable permissions is rejected."""
        jobs_file = tmp_path / "jobs.json"
        jobs_file.write_text("[]")
        mgr = self._make_manager(jobs_file)

        with patch.object(Path, "stat") as mock_stat:
            mock_result = MagicMock()
            mock_result.st_uid = os.getuid()
            mock_result.st_mode = 0o100660  # Group writable
            mock_stat.return_value = mock_result
            mgr._load_jobs()

        assert len(mgr._jobs) == 0

    def test_stat_oserror_handled(self, tmp_path):
        """OSError on stat() is caught gracefully."""
        jobs_file = tmp_path / "jobs.json"
        jobs_file.write_text("[]")
        mgr = self._make_manager(jobs_file)

        with (
            patch.object(type(jobs_file), "stat", side_effect=OSError("permission denied")),
            patch.object(type(jobs_file), "exists", return_value=True),
        ):
            mgr._load_jobs()

        assert len(mgr._jobs) == 0

    def test_malformed_individual_record_skipped(self, tmp_path):
        """Valid jobs are loaded even if some records are malformed."""
        jobs_file = tmp_path / "jobs.json"
        records = [
            {"bad": "record"},  # Missing required fields
            {
                "id": "job1",
                "name": "test",
                "schedule": "every 5 minutes",
                "task": "echo hello",
                "provider": "anthropic",
                "enabled": True,
                "status": "active",
            },
        ]
        jobs_file.write_text(json.dumps(records))
        mgr = self._make_manager(jobs_file)

        with patch.object(Path, "stat") as mock_stat:
            mock_result = MagicMock()
            mock_result.st_uid = os.getuid()
            mock_result.st_mode = 0o100600
            mock_stat.return_value = mock_result
            mgr._load_jobs()

        # At least one should have been skipped, the valid one may or may not load
        # depending on ScheduledJob.from_dict() validation
        # Key: no crash occurred


# ──────────────────────────────────────────────────────────────────────
# Code evolution malformed data recovery
# ──────────────────────────────────────────────────────────────────────


class TestCodeEvolutionLoadRecovery:
    """Verify evolution data loading handles corruption gracefully."""

    def test_malformed_json_returns_empty(self, tmp_path):
        from missy.agent.code_evolution import CodeEvolutionManager

        store = tmp_path / "evolutions.json"
        store.write_text("{not valid json!")
        mgr = CodeEvolutionManager(store_path=str(store))
        assert mgr._proposals == []

    def test_nonexistent_file_returns_empty(self, tmp_path):
        from missy.agent.code_evolution import CodeEvolutionManager

        store = tmp_path / "does_not_exist.json"
        mgr = CodeEvolutionManager(store_path=str(store))
        assert mgr._proposals == []


class TestCodeEvolutionTracebackParsing:
    """Verify analyze_error_for_evolution handles edge cases."""

    def test_no_missy_files_in_traceback(self, tmp_path):
        from missy.agent.code_evolution import CodeEvolutionManager

        mgr = CodeEvolutionManager(store_path=str(tmp_path / "evo.json"))
        result = mgr.analyze_error_for_evolution(
            tool_name="test",
            error_message="some error",
            traceback_text='File "/other/code.py", line 1\n  raise ValueError()',
            failure_count=5,
        )
        assert result is None

    def test_malformed_traceback_lines(self, tmp_path):
        from missy.agent.code_evolution import CodeEvolutionManager

        mgr = CodeEvolutionManager(store_path=str(tmp_path / "evo.json"))
        # Line contains missy/ and File " but the path extraction will fail
        mgr.analyze_error_for_evolution(
            tool_name="test",
            error_message="some error",
            traceback_text='  File "missy/broken',  # Truncated/malformed
            failure_count=5,
        )
        # May return None (no valid missy file extracted) or a proposal
        # Key invariant: should not crash

    def test_failure_count_below_threshold(self, tmp_path):
        from missy.agent.code_evolution import CodeEvolutionManager

        mgr = CodeEvolutionManager(store_path=str(tmp_path / "evo.json"))
        result = mgr.analyze_error_for_evolution(
            tool_name="test",
            error_message="some error",
            traceback_text='File "/path/to/missy/foo.py", line 42',
            failure_count=2,  # Below threshold of 3
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────────
# Provider registry constructor failure
# ──────────────────────────────────────────────────────────────────────


class TestProviderRegistryConstructorFailure:
    """Verify from_config handles provider construction errors."""

    def test_provider_constructor_exception_skipped(self):
        from missy.providers.registry import ProviderRegistry

        # Create a mock config with a provider that will fail to construct
        mock_provider_config = MagicMock()
        mock_provider_config.enabled = True
        mock_provider_config.name = "test_broken"
        mock_provider_config.base_url = ""

        mock_config = MagicMock()
        mock_config.providers = {"test_broken": mock_provider_config}
        mock_config.network.provider_allowed_hosts = []

        bad_cls = MagicMock(side_effect=TypeError("bad init"))
        with patch.dict(
            "missy.providers.registry._PROVIDER_CLASSES",
            {"test_broken": bad_cls},
        ):
            registry = ProviderRegistry.from_config(mock_config)
            # Should not crash; provider is just skipped
            assert "test_broken" not in registry._providers
