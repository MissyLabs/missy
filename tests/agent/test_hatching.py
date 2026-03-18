"""Tests for missy.agent.hatching — first-run bootstrapping."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from missy.agent.hatching import (
    HatchingLog,
    HatchingManager,
    HatchingState,
    HatchingStatus,
    _HatchingStepWarning,
)


# ---------------------------------------------------------------------------
# HatchingStatus enum
# ---------------------------------------------------------------------------


class TestHatchingStatusEnum:
    def test_all_four_states_exist(self):
        names = {s.name for s in HatchingStatus}
        assert names == {"UNHATCHED", "IN_PROGRESS", "HATCHED", "FAILED"}

    def test_values_are_strings(self):
        assert HatchingStatus.UNHATCHED.value == "unhatched"
        assert HatchingStatus.IN_PROGRESS.value == "in_progress"
        assert HatchingStatus.HATCHED.value == "hatched"
        assert HatchingStatus.FAILED.value == "failed"


# ---------------------------------------------------------------------------
# HatchingState dataclass
# ---------------------------------------------------------------------------


class TestHatchingStateDefaults:
    def test_default_status_is_unhatched(self):
        state = HatchingState()
        assert state.status is HatchingStatus.UNHATCHED

    def test_default_timestamps_are_none(self):
        state = HatchingState()
        assert state.started_at is None
        assert state.completed_at is None

    def test_default_flags_are_false(self):
        state = HatchingState()
        assert state.persona_generated is False
        assert state.environment_validated is False
        assert state.provider_verified is False
        assert state.security_initialized is False
        assert state.memory_seeded is False

    def test_default_steps_completed_is_empty_list(self):
        state = HatchingState()
        assert state.steps_completed == []

    def test_default_error_is_none(self):
        state = HatchingState()
        assert state.error is None


class TestHatchingStateToDictFromDict:
    def test_to_dict_serialises_status_as_string(self):
        state = HatchingState(status=HatchingStatus.HATCHED)
        d = state.to_dict()
        assert d["status"] == "hatched"

    def test_to_dict_from_dict_round_trips_default_state(self):
        original = HatchingState()
        restored = HatchingState.from_dict(original.to_dict())
        assert restored.status is original.status
        assert restored.started_at == original.started_at
        assert restored.completed_at == original.completed_at
        assert restored.steps_completed == original.steps_completed
        assert restored.persona_generated == original.persona_generated
        assert restored.environment_validated == original.environment_validated
        assert restored.provider_verified == original.provider_verified
        assert restored.security_initialized == original.security_initialized
        assert restored.memory_seeded == original.memory_seeded
        assert restored.error == original.error

    def test_to_dict_from_dict_round_trips_populated_state(self):
        original = HatchingState(
            status=HatchingStatus.HATCHED,
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:01:00+00:00",
            steps_completed=["validate_environment", "finalize"],
            persona_generated=True,
            environment_validated=True,
            provider_verified=True,
            security_initialized=True,
            memory_seeded=True,
            error=None,
        )
        restored = HatchingState.from_dict(original.to_dict())
        assert restored.status is HatchingStatus.HATCHED
        assert restored.started_at == "2026-01-01T00:00:00+00:00"
        assert restored.completed_at == "2026-01-01T00:01:00+00:00"
        assert restored.steps_completed == ["validate_environment", "finalize"]
        assert restored.persona_generated is True
        assert restored.memory_seeded is True

    def test_from_dict_with_unknown_status_falls_back_to_unhatched(self):
        state = HatchingState.from_dict({"status": "totally_unknown_value"})
        assert state.status is HatchingStatus.UNHATCHED

    def test_from_dict_with_missing_keys_uses_defaults(self):
        state = HatchingState.from_dict({})
        assert state.status is HatchingStatus.UNHATCHED
        assert state.steps_completed == []
        assert state.error is None


# ---------------------------------------------------------------------------
# HatchingLog
# ---------------------------------------------------------------------------


class TestHatchingLogWriteAndRead:
    def test_log_writes_entry(self, tmp_path):
        log = HatchingLog(log_path=tmp_path / "test.jsonl")
        log.log("my_step", "ok", "everything fine")
        entries = log.get_entries()
        assert len(entries) == 1
        assert entries[0]["step"] == "my_step"
        assert entries[0]["status"] == "ok"
        assert entries[0]["message"] == "everything fine"

    def test_log_appends_multiple_entries_in_order(self, tmp_path):
        log = HatchingLog(log_path=tmp_path / "test.jsonl")
        log.log("step_a", "ok", "first")
        log.log("step_b", "warn", "second")
        log.log("step_c", "error", "third")
        entries = log.get_entries()
        assert len(entries) == 3
        assert [e["step"] for e in entries] == ["step_a", "step_b", "step_c"]

    def test_log_includes_details_when_provided(self, tmp_path):
        log = HatchingLog(log_path=tmp_path / "test.jsonl")
        log.log("some_step", "ok", "msg", details={"key": "value"})
        entries = log.get_entries()
        assert entries[0]["details"] == {"key": "value"}

    def test_log_details_defaults_to_empty_dict(self, tmp_path):
        log = HatchingLog(log_path=tmp_path / "test.jsonl")
        log.log("s", "ok", "m")
        entries = log.get_entries()
        assert entries[0]["details"] == {}

    def test_get_entries_returns_empty_list_when_file_missing(self, tmp_path):
        log = HatchingLog(log_path=tmp_path / "nonexistent.jsonl")
        assert log.get_entries() == []

    def test_get_entries_skips_corrupt_lines(self, tmp_path):
        log_path = tmp_path / "mixed.jsonl"
        log_path.write_text(
            '{"step": "good", "status": "ok", "message": "m", "details": {}, "timestamp": "t"}\n'
            "THIS IS NOT JSON\n"
            '{"step": "also_good", "status": "ok", "message": "m2", "details": {}, "timestamp": "t"}\n',
            encoding="utf-8",
        )
        log = HatchingLog(log_path=log_path)
        entries = log.get_entries()
        assert len(entries) == 2
        assert entries[0]["step"] == "good"
        assert entries[1]["step"] == "also_good"

    def test_get_entries_handles_blank_lines_gracefully(self, tmp_path):
        log_path = tmp_path / "blanks.jsonl"
        log_path.write_text(
            "\n"
            '{"step": "s", "status": "ok", "message": "m", "details": {}, "timestamp": "t"}\n'
            "\n",
            encoding="utf-8",
        )
        log = HatchingLog(log_path=log_path)
        assert len(log.get_entries()) == 1

    def test_log_creates_parent_directories(self, tmp_path):
        nested_path = tmp_path / "deep" / "nested" / "hatching.jsonl"
        log = HatchingLog(log_path=nested_path)
        log.log("s", "ok", "m")
        assert nested_path.exists()


# ---------------------------------------------------------------------------
# HatchingManager — state queries
# ---------------------------------------------------------------------------


class TestHatchingManagerStateQueries:
    def test_needs_hatching_true_when_no_state_file(self, tmp_path):
        mgr = HatchingManager(
            state_path=tmp_path / "hatching.yaml",
            log_path=tmp_path / "log.jsonl",
        )
        assert mgr.needs_hatching() is True

    def test_is_hatched_false_when_no_state_file(self, tmp_path):
        mgr = HatchingManager(
            state_path=tmp_path / "hatching.yaml",
            log_path=tmp_path / "log.jsonl",
        )
        assert mgr.is_hatched() is False

    def test_get_state_returns_unhatched_default_when_no_file(self, tmp_path):
        mgr = HatchingManager(
            state_path=tmp_path / "hatching.yaml",
            log_path=tmp_path / "log.jsonl",
        )
        state = mgr.get_state()
        assert state.status is HatchingStatus.UNHATCHED

    def test_needs_hatching_true_when_status_is_failed(self, tmp_path):
        state_path = tmp_path / "hatching.yaml"
        state = HatchingState(status=HatchingStatus.FAILED)
        state_path.write_text(yaml.safe_dump(state.to_dict()), encoding="utf-8")
        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "log.jsonl")
        assert mgr.needs_hatching() is True

    def test_needs_hatching_false_when_already_hatched(self, tmp_path):
        state_path = tmp_path / "hatching.yaml"
        state = HatchingState(status=HatchingStatus.HATCHED)
        state_path.write_text(yaml.safe_dump(state.to_dict()), encoding="utf-8")
        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "log.jsonl")
        assert mgr.needs_hatching() is False

    def test_is_hatched_true_when_status_is_hatched(self, tmp_path):
        state_path = tmp_path / "hatching.yaml"
        state = HatchingState(status=HatchingStatus.HATCHED)
        state_path.write_text(yaml.safe_dump(state.to_dict()), encoding="utf-8")
        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "log.jsonl")
        assert mgr.is_hatched() is True

    def test_get_state_returns_default_when_yaml_is_corrupt(self, tmp_path):
        state_path = tmp_path / "hatching.yaml"
        state_path.write_text(": this: is: not: valid yaml :::", encoding="utf-8")
        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "log.jsonl")
        state = mgr.get_state()
        assert state.status is HatchingStatus.UNHATCHED


# ---------------------------------------------------------------------------
# HatchingManager — run_hatching
# ---------------------------------------------------------------------------


def _make_manager(tmp_path: Path) -> HatchingManager:
    """Create a HatchingManager that writes all paths inside tmp_path."""
    return HatchingManager(
        state_path=tmp_path / "hatching.yaml",
        log_path=tmp_path / "log.jsonl",
    )


def _patch_module_paths(monkeypatch, tmp_path: Path):
    """Redirect all module-level _*_PATH constants inside hatching.py to tmp_path."""
    import missy.agent.hatching as hatching_mod

    monkeypatch.setattr(hatching_mod, "_MISSY_DIR", tmp_path)
    monkeypatch.setattr(hatching_mod, "_CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(hatching_mod, "_IDENTITY_PATH", tmp_path / "identity.pem")
    monkeypatch.setattr(hatching_mod, "_SECRETS_DIR", tmp_path / "secrets")
    monkeypatch.setattr(hatching_mod, "_PERSONA_PATH", tmp_path / "persona.yaml")
    monkeypatch.setattr(hatching_mod, "_MEMORY_DB_PATH", tmp_path / "memory.db")


class TestHatchingManagerRunHatching:
    def test_run_hatching_succeeds_and_returns_hatched_state(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)

        mock_persona_manager = MagicMock()
        mock_store = MagicMock()
        mock_turn = MagicMock()
        mock_turn.id = "turn-1"

        with patch.dict(
            "sys.modules",
            {
                "missy.agent.persona": MagicMock(PersonaManager=MagicMock(return_value=mock_persona_manager)),
                "missy.memory.sqlite_store": MagicMock(
                    SQLiteMemoryStore=MagicMock(return_value=mock_store),
                    ConversationTurn=MagicMock(new=MagicMock(return_value=mock_turn)),
                ),
            },
        ):
            mgr = _make_manager(tmp_path)
            monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
            state = mgr.run_hatching(interactive=False)

        assert state.status is HatchingStatus.HATCHED

    def test_run_hatching_records_all_steps_completed(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)

        mock_turn = MagicMock()
        mock_turn.id = "turn-1"
        mock_store = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "missy.agent.persona": MagicMock(PersonaManager=MagicMock(return_value=MagicMock())),
                "missy.memory.sqlite_store": MagicMock(
                    SQLiteMemoryStore=MagicMock(return_value=mock_store),
                    ConversationTurn=MagicMock(new=MagicMock(return_value=mock_turn)),
                ),
            },
        ):
            mgr = _make_manager(tmp_path)
            monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
            state = mgr.run_hatching(interactive=False)

        expected_steps = {
            "validate_environment",
            "initialize_config",
            "verify_providers",
            "initialize_security",
            "generate_persona",
            "seed_memory",
            "finalize",
        }
        assert expected_steps.issubset(set(state.steps_completed))

    def test_run_hatching_returns_immediately_when_already_hatched(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        state_path = tmp_path / "hatching.yaml"
        hatched_state = HatchingState(
            status=HatchingStatus.HATCHED,
            completed_at="2026-01-01T00:00:00+00:00",
        )
        state_path.write_text(yaml.safe_dump(hatched_state.to_dict()), encoding="utf-8")

        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "log.jsonl")
        state = mgr.run_hatching(interactive=False)

        assert state.status is HatchingStatus.HATCHED
        # State file should not have been overwritten with new started_at
        assert state.completed_at == "2026-01-01T00:00:00+00:00"

    def test_run_hatching_skips_already_completed_steps(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)

        # Pre-populate state with some steps done
        state_path = tmp_path / "hatching.yaml"
        partial_state = HatchingState(
            status=HatchingStatus.IN_PROGRESS,
            started_at="2026-01-01T00:00:00+00:00",
            steps_completed=["validate_environment", "initialize_config"],
            environment_validated=True,
        )
        state_path.write_text(yaml.safe_dump(partial_state.to_dict()), encoding="utf-8")

        mock_turn = MagicMock()
        mock_turn.id = "t"
        mock_store = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "missy.agent.persona": MagicMock(PersonaManager=MagicMock(return_value=MagicMock())),
                "missy.memory.sqlite_store": MagicMock(
                    SQLiteMemoryStore=MagicMock(return_value=mock_store),
                    ConversationTurn=MagicMock(new=MagicMock(return_value=mock_turn)),
                ),
            },
        ):
            mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "log.jsonl")
            monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
            state = mgr.run_hatching(interactive=False)

        assert state.status is HatchingStatus.HATCHED
        # validate_environment counted once; the partial state listed it already done
        assert state.steps_completed.count("validate_environment") == 1

    def test_run_hatching_from_failed_state_clears_error_and_retries(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)

        state_path = tmp_path / "hatching.yaml"
        failed_state = HatchingState(
            status=HatchingStatus.FAILED,
            error="Step 'some_step' failed: something went wrong",
        )
        state_path.write_text(yaml.safe_dump(failed_state.to_dict()), encoding="utf-8")

        mock_turn = MagicMock()
        mock_turn.id = "t"
        mock_store = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "missy.agent.persona": MagicMock(PersonaManager=MagicMock(return_value=MagicMock())),
                "missy.memory.sqlite_store": MagicMock(
                    SQLiteMemoryStore=MagicMock(return_value=mock_store),
                    ConversationTurn=MagicMock(new=MagicMock(return_value=mock_turn)),
                ),
            },
        ):
            mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "log.jsonl")
            monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
            state = mgr.run_hatching(interactive=False)

        assert state.status is HatchingStatus.HATCHED
        assert state.error is None

    def test_run_hatching_sets_started_at_timestamp(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)

        mock_turn = MagicMock()
        mock_turn.id = "t"
        mock_store = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "missy.agent.persona": MagicMock(PersonaManager=MagicMock(return_value=MagicMock())),
                "missy.memory.sqlite_store": MagicMock(
                    SQLiteMemoryStore=MagicMock(return_value=mock_store),
                    ConversationTurn=MagicMock(new=MagicMock(return_value=mock_turn)),
                ),
            },
        ):
            mgr = _make_manager(tmp_path)
            monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
            state = mgr.run_hatching(interactive=False)

        assert state.started_at is not None
        assert state.completed_at is not None

    def test_run_hatching_fails_state_on_unrecoverable_step_error(self, tmp_path, monkeypatch):
        """A step raising a plain Exception (not _HatchingStepWarning) marks state FAILED."""
        _patch_module_paths(monkeypatch, tmp_path)

        import missy.agent.hatching as hatching_mod

        def _bad_step(state, *, interactive):
            raise RuntimeError("disk exploded")

        mgr = _make_manager(tmp_path)
        # Inject a broken step directly
        original_finalize = mgr._finalize
        mgr._finalize = _bad_step  # type: ignore[method-assign]

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        mock_turn = MagicMock()
        mock_turn.id = "t"
        mock_store = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "missy.agent.persona": MagicMock(PersonaManager=MagicMock(return_value=MagicMock())),
                "missy.memory.sqlite_store": MagicMock(
                    SQLiteMemoryStore=MagicMock(return_value=mock_store),
                    ConversationTurn=MagicMock(new=MagicMock(return_value=mock_turn)),
                ),
            },
        ):
            state = mgr.run_hatching(interactive=False)

        assert state.status is HatchingStatus.FAILED
        assert state.error is not None
        assert "disk exploded" in state.error


# ---------------------------------------------------------------------------
# HatchingManager — reset
# ---------------------------------------------------------------------------


class TestHatchingManagerReset:
    def test_reset_removes_state_file(self, tmp_path):
        state_path = tmp_path / "hatching.yaml"
        state_path.write_text(yaml.safe_dump(HatchingState().to_dict()), encoding="utf-8")

        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "log.jsonl")
        assert state_path.exists()
        mgr.reset()
        assert not state_path.exists()

    def test_reset_when_no_state_file_does_not_raise(self, tmp_path):
        mgr = HatchingManager(
            state_path=tmp_path / "hatching.yaml",
            log_path=tmp_path / "log.jsonl",
        )
        mgr.reset()  # should not raise

    def test_reset_writes_log_entry(self, tmp_path):
        log_path = tmp_path / "log.jsonl"
        mgr = HatchingManager(state_path=tmp_path / "hatching.yaml", log_path=log_path)
        mgr.reset()
        log = HatchingLog(log_path=log_path)
        entries = log.get_entries()
        assert any(e["step"] == "reset" for e in entries)

    def test_reset_preserves_log_file(self, tmp_path):
        log_path = tmp_path / "log.jsonl"
        state_path = tmp_path / "hatching.yaml"
        state_path.write_text(yaml.safe_dump(HatchingState().to_dict()), encoding="utf-8")

        mgr = HatchingManager(state_path=state_path, log_path=log_path)
        # Write a log entry before reset
        mgr.get_hatching_log()  # no-op but exercises the path
        log = HatchingLog(log_path=log_path)
        log.log("pre_reset", "ok", "entry before reset")

        mgr.reset()

        # Log file must still exist with prior entry intact
        assert log_path.exists()
        entries = log.get_entries()
        assert any(e["step"] == "pre_reset" for e in entries)

    def test_needs_hatching_true_after_reset(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)

        state_path = tmp_path / "hatching.yaml"
        state = HatchingState(status=HatchingStatus.HATCHED)
        state_path.write_text(yaml.safe_dump(state.to_dict()), encoding="utf-8")

        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "log.jsonl")
        assert not mgr.needs_hatching()

        mgr.reset()
        assert mgr.needs_hatching()


# ---------------------------------------------------------------------------
# HatchingManager — get_hatching_log
# ---------------------------------------------------------------------------


class TestHatchingManagerGetHatchingLog:
    def test_get_hatching_log_returns_empty_list_initially(self, tmp_path):
        mgr = HatchingManager(
            state_path=tmp_path / "hatching.yaml",
            log_path=tmp_path / "log.jsonl",
        )
        assert mgr.get_hatching_log() == []

    def test_get_hatching_log_reflects_entries_after_run(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)

        mock_turn = MagicMock()
        mock_turn.id = "t"
        mock_store = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "missy.agent.persona": MagicMock(PersonaManager=MagicMock(return_value=MagicMock())),
                "missy.memory.sqlite_store": MagicMock(
                    SQLiteMemoryStore=MagicMock(return_value=mock_store),
                    ConversationTurn=MagicMock(new=MagicMock(return_value=mock_turn)),
                ),
            },
        ):
            mgr = _make_manager(tmp_path)
            monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
            mgr.run_hatching(interactive=False)

        entries = mgr.get_hatching_log()
        assert len(entries) > 0
        steps_logged = {e["step"] for e in entries}
        assert "hatching" in steps_logged
        assert "finalize" in steps_logged
