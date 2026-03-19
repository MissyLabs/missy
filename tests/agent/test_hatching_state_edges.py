"""Comprehensive tests for HatchingManager, HatchingState,

HatchingStatus, and HatchingLog from missy/agent/hatching.py.

Coverage targets:
- HatchingStatus enum completeness and values
- HatchingState defaults, mutations, to_dict/from_dict serialisation
- HatchingLog write/read, append semantics, error resilience, timestamps
- HatchingManager path isolation, needs_hatching logic, get_state / save_state
- run_hatching: happy path, step skipping (resume), failure propagation,
  step warning (non-fatal), interactive vs. non-interactive modes
- Individual private step methods via direct injection or monkeypatching
- check_environment Python version enforcement and disk-space warning
- verify_providers env var vs config key detection
- _check_config_for_provider_key config parsing logic
- _initialize_config creates default when absent, skips when present
- _initialize_security creates secrets dir, handles missing identity key
- _generate_persona happy path, import-error warning, OS-error warning
- _seed_memory happy path, import-error warning, generic exception warning
- _finalize sets HATCHED, records completed_at, clears error
- reset / get_hatching_log integration
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from missy.agent.hatching import (
    _MIN_FREE_BYTES,
    HatchingLog,
    HatchingManager,
    HatchingState,
    HatchingStatus,
    _HatchingStepWarning,
)

# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _make_manager(tmp_path: Path) -> HatchingManager:
    """Isolated HatchingManager with all paths inside *tmp_path*."""
    return HatchingManager(
        state_path=tmp_path / "hatching.yaml",
        log_path=tmp_path / "log.jsonl",
    )


def _patch_module_paths(monkeypatch, tmp_path: Path) -> None:
    """Redirect all module-level _*_PATH constants to *tmp_path*."""
    import missy.agent.hatching as m

    monkeypatch.setattr(m, "_MISSY_DIR", tmp_path)
    monkeypatch.setattr(m, "_CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(m, "_IDENTITY_PATH", tmp_path / "identity.pem")
    monkeypatch.setattr(m, "_SECRETS_DIR", tmp_path / "secrets")
    monkeypatch.setattr(m, "_PERSONA_PATH", tmp_path / "persona.yaml")
    monkeypatch.setattr(m, "_MEMORY_DB_PATH", tmp_path / "memory.db")


def _stub_external_deps(monkeypatch):
    """Stub PersonaManager and memory store so individual step tests don't
    need real implementations installed."""
    mock_turn = MagicMock()
    mock_turn.id = "t-1"
    mock_store = MagicMock()
    mock_pm = MagicMock()

    monkeypatch.setattr("missy.agent.persona.PersonaManager", mock_pm)
    monkeypatch.setattr("missy.memory.sqlite_store.SQLiteMemoryStore", mock_store)
    monkeypatch.setattr(
        "missy.memory.sqlite_store.ConversationTurn",
        MagicMock(new=MagicMock(return_value=mock_turn)),
    )
    return mock_pm, mock_store, mock_turn


def _write_state(path: Path, state: HatchingState) -> None:
    path.write_text(yaml.safe_dump(state.to_dict()), encoding="utf-8")


# ===========================================================================
# 1. HatchingStatus enum
# ===========================================================================


class TestHatchingStatusEnum:
    """Verify every enum member exists with the correct string value."""

    def test_unhatched_value(self):
        assert HatchingStatus.UNHATCHED.value == "unhatched"

    def test_in_progress_value(self):
        assert HatchingStatus.IN_PROGRESS.value == "in_progress"

    def test_hatched_value(self):
        assert HatchingStatus.HATCHED.value == "hatched"

    def test_failed_value(self):
        assert HatchingStatus.FAILED.value == "failed"

    def test_exactly_four_members(self):
        assert len(list(HatchingStatus)) == 4

    def test_membership_by_value(self):
        assert HatchingStatus("hatched") is HatchingStatus.HATCHED

    def test_invalid_value_raises_value_error(self):
        with pytest.raises(ValueError):
            HatchingStatus("not_a_real_status")


# ===========================================================================
# 2. HatchingState defaults
# ===========================================================================


class TestHatchingStateDefaults:
    """HatchingState should use safe defaults on construction."""

    def test_status_defaults_to_unhatched(self):
        assert HatchingState().status is HatchingStatus.UNHATCHED

    def test_started_at_defaults_to_none(self):
        assert HatchingState().started_at is None

    def test_completed_at_defaults_to_none(self):
        assert HatchingState().completed_at is None

    def test_steps_completed_defaults_to_empty_list(self):
        assert HatchingState().steps_completed == []

    def test_persona_generated_defaults_false(self):
        assert HatchingState().persona_generated is False

    def test_environment_validated_defaults_false(self):
        assert HatchingState().environment_validated is False

    def test_provider_verified_defaults_false(self):
        assert HatchingState().provider_verified is False

    def test_security_initialized_defaults_false(self):
        assert HatchingState().security_initialized is False

    def test_memory_seeded_defaults_false(self):
        assert HatchingState().memory_seeded is False

    def test_error_defaults_to_none(self):
        assert HatchingState().error is None

    def test_steps_completed_instances_are_independent(self):
        """Each instance must have its own list, not share one."""
        a = HatchingState()
        b = HatchingState()
        a.steps_completed.append("x")
        assert "x" not in b.steps_completed


# ===========================================================================
# 3. HatchingState with custom values
# ===========================================================================


class TestHatchingStateCustomValues:
    """to_dict / from_dict round-trips and field mutations."""

    def test_custom_status_preserved(self):
        state = HatchingState(status=HatchingStatus.HATCHED)
        assert state.status is HatchingStatus.HATCHED

    def test_to_dict_encodes_status_as_string(self):
        d = HatchingState(status=HatchingStatus.IN_PROGRESS).to_dict()
        assert d["status"] == "in_progress"

    def test_to_dict_includes_all_fields(self):
        state = HatchingState(
            status=HatchingStatus.HATCHED,
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:01:00+00:00",
            steps_completed=["validate_environment"],
            persona_generated=True,
            environment_validated=True,
            provider_verified=True,
            security_initialized=True,
            memory_seeded=True,
            error=None,
        )
        d = state.to_dict()
        assert d["started_at"] == "2026-01-01T00:00:00+00:00"
        assert d["completed_at"] == "2026-01-01T00:01:00+00:00"
        assert d["steps_completed"] == ["validate_environment"]
        assert d["persona_generated"] is True
        assert d["memory_seeded"] is True

    def test_from_dict_round_trip_full_state(self):
        original = HatchingState(
            status=HatchingStatus.HATCHED,
            started_at="2026-03-19T10:00:00+00:00",
            completed_at="2026-03-19T10:01:00+00:00",
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
        assert restored.started_at == original.started_at
        assert restored.completed_at == original.completed_at
        assert restored.steps_completed == original.steps_completed
        assert restored.persona_generated is True
        assert restored.memory_seeded is True
        assert restored.error is None

    def test_from_dict_unknown_status_falls_back_to_unhatched(self):
        state = HatchingState.from_dict({"status": "definitely_unknown"})
        assert state.status is HatchingStatus.UNHATCHED

    def test_from_dict_empty_dict_uses_defaults(self):
        state = HatchingState.from_dict({})
        assert state.status is HatchingStatus.UNHATCHED
        assert state.steps_completed == []

    def test_from_dict_none_steps_becomes_empty_list(self):
        state = HatchingState.from_dict({"steps_completed": None})
        assert state.steps_completed == []

    def test_from_dict_coerces_truthy_int_flag(self):
        state = HatchingState.from_dict({"environment_validated": 1})
        assert state.environment_validated is True

    def test_from_dict_coerces_empty_string_flag_to_false(self):
        state = HatchingState.from_dict({"provider_verified": ""})
        assert state.provider_verified is False

    def test_from_dict_preserves_error_string(self):
        state = HatchingState.from_dict({"error": "oops something went wrong"})
        assert state.error == "oops something went wrong"


# ===========================================================================
# 4. HatchingLog
# ===========================================================================


class TestHatchingLogWrite:
    """Log entries should be persisted as valid JSONL."""

    def test_single_entry_readable_back(self, tmp_path):
        log = HatchingLog(log_path=tmp_path / "h.jsonl")
        log.log("step_a", "ok", "all good")
        entries = log.get_entries()
        assert len(entries) == 1
        entry = entries[0]
        assert entry["step"] == "step_a"
        assert entry["status"] == "ok"
        assert entry["message"] == "all good"

    def test_entry_contains_timestamp(self, tmp_path):
        log = HatchingLog(log_path=tmp_path / "h.jsonl")
        log.log("s", "ok", "m")
        entry = log.get_entries()[0]
        assert "timestamp" in entry
        assert entry["timestamp"]  # non-empty

    def test_entry_with_details(self, tmp_path):
        log = HatchingLog(log_path=tmp_path / "h.jsonl")
        log.log("s", "ok", "m", details={"provider": "anthropic"})
        assert log.get_entries()[0]["details"] == {"provider": "anthropic"}

    def test_entry_without_details_defaults_to_empty_dict(self, tmp_path):
        log = HatchingLog(log_path=tmp_path / "h.jsonl")
        log.log("s", "ok", "m")
        assert log.get_entries()[0]["details"] == {}

    def test_multiple_entries_appended_in_order(self, tmp_path):
        log = HatchingLog(log_path=tmp_path / "h.jsonl")
        for i in range(5):
            log.log(f"step_{i}", "ok", f"msg {i}")
        steps = [e["step"] for e in log.get_entries()]
        assert steps == [f"step_{i}" for i in range(5)]

    def test_get_entries_returns_empty_list_when_file_absent(self, tmp_path):
        log = HatchingLog(log_path=tmp_path / "nonexistent.jsonl")
        assert log.get_entries() == []

    def test_get_entries_skips_malformed_lines(self, tmp_path):
        p = tmp_path / "mixed.jsonl"
        p.write_text(
            '{"step":"good","status":"ok","message":"m","details":{},"timestamp":"t"}\n'
            "NOT JSON AT ALL\n"
            '{"step":"also_good","status":"ok","message":"m2","details":{},"timestamp":"t"}\n',
            encoding="utf-8",
        )
        log = HatchingLog(log_path=p)
        entries = log.get_entries()
        assert len(entries) == 2
        assert entries[0]["step"] == "good"
        assert entries[1]["step"] == "also_good"

    def test_get_entries_ignores_blank_lines(self, tmp_path):
        p = tmp_path / "blanks.jsonl"
        p.write_text(
            "\n"
            '{"step":"s","status":"ok","message":"m","details":{},"timestamp":"t"}\n'
            "\n",
            encoding="utf-8",
        )
        log = HatchingLog(log_path=p)
        assert len(log.get_entries()) == 1

    def test_log_creates_parent_directories(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "log.jsonl"
        log = HatchingLog(log_path=deep)
        log.log("s", "ok", "m")
        assert deep.exists()

    def test_log_survives_oserror_on_open(self, tmp_path):
        log = HatchingLog(log_path=tmp_path / "log.jsonl")
        with patch("missy.agent.hatching.os.open", side_effect=OSError("disk full")):
            log.log("s", "ok", "m")  # must not raise

    def test_get_entries_survives_oserror_on_read(self, tmp_path):
        p = tmp_path / "log.jsonl"
        p.write_text('{"step":"s","status":"ok","message":"m","details":{},"timestamp":"t"}\n')
        log = HatchingLog(log_path=p)
        with patch.object(Path, "open", side_effect=OSError("permission denied")):
            result = log.get_entries()
        assert result == []


# ===========================================================================
# 5. HatchingManager — file and state operations
# ===========================================================================


class TestHatchingManagerStateFile:
    """get_state / save via run_hatching / persistence round-trip."""

    def test_get_state_returns_unhatched_when_no_file(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.get_state().status is HatchingStatus.UNHATCHED

    def test_get_state_returns_persisted_status(self, tmp_path):
        state_path = tmp_path / "hatching.yaml"
        _write_state(state_path, HatchingState(status=HatchingStatus.HATCHED))
        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "l.jsonl")
        assert mgr.get_state().status is HatchingStatus.HATCHED

    def test_get_state_returns_default_for_empty_yaml(self, tmp_path):
        state_path = tmp_path / "hatching.yaml"
        state_path.write_text("", encoding="utf-8")
        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "l.jsonl")
        assert mgr.get_state().status is HatchingStatus.UNHATCHED

    def test_get_state_returns_default_for_corrupt_yaml(self, tmp_path):
        state_path = tmp_path / "hatching.yaml"
        state_path.write_text(": bad: yaml :::", encoding="utf-8")
        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "l.jsonl")
        assert mgr.get_state().status is HatchingStatus.UNHATCHED

    def test_get_state_returns_default_for_non_dict_yaml(self, tmp_path):
        state_path = tmp_path / "hatching.yaml"
        state_path.write_text("- item1\n- item2\n", encoding="utf-8")
        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "l.jsonl")
        assert mgr.get_state().status is HatchingStatus.UNHATCHED

    def test_get_state_ignores_unknown_yaml_keys(self, tmp_path):
        state_path = tmp_path / "hatching.yaml"
        data = {"status": "hatched", "future_key": "future_value"}
        state_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "l.jsonl")
        assert mgr.get_state().status is HatchingStatus.HATCHED

    def test_save_state_produces_readable_yaml(self, tmp_path):
        """_save_state (exercised via run_hatching) must write parseable YAML."""

        state_path = tmp_path / "hatching.yaml"
        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "l.jsonl")
        test_state = HatchingState(
            status=HatchingStatus.IN_PROGRESS,
            started_at="2026-03-19T10:00:00+00:00",
            steps_completed=["validate_environment"],
        )
        mgr._save_state(test_state)

        with state_path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        assert loaded["status"] == "in_progress"
        assert "validate_environment" in loaded["steps_completed"]

    def test_save_state_round_trip_preserves_all_flags(self, tmp_path):

        state_path = tmp_path / "hatching.yaml"
        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "l.jsonl")
        original = HatchingState(
            status=HatchingStatus.HATCHED,
            persona_generated=True,
            environment_validated=True,
            provider_verified=True,
            security_initialized=True,
            memory_seeded=True,
        )
        mgr._save_state(original)
        restored = mgr.get_state()

        assert restored.persona_generated is True
        assert restored.environment_validated is True
        assert restored.provider_verified is True
        assert restored.security_initialized is True
        assert restored.memory_seeded is True


# ===========================================================================
# 6. needs_hatching / is_hatched
# ===========================================================================


class TestNeedsHatching:
    """needs_hatching and is_hatched reflect file / status correctly."""

    def test_needs_hatching_true_when_no_file(self, tmp_path):
        assert _make_manager(tmp_path).needs_hatching() is True

    def test_needs_hatching_true_for_unhatched_status(self, tmp_path):
        sp = tmp_path / "hatching.yaml"
        _write_state(sp, HatchingState(status=HatchingStatus.UNHATCHED))
        mgr = HatchingManager(state_path=sp, log_path=tmp_path / "l.jsonl")
        assert mgr.needs_hatching() is True

    def test_needs_hatching_true_for_failed_status(self, tmp_path):
        sp = tmp_path / "hatching.yaml"
        _write_state(sp, HatchingState(status=HatchingStatus.FAILED))
        mgr = HatchingManager(state_path=sp, log_path=tmp_path / "l.jsonl")
        assert mgr.needs_hatching() is True

    def test_needs_hatching_false_for_hatched_status(self, tmp_path):
        sp = tmp_path / "hatching.yaml"
        _write_state(sp, HatchingState(status=HatchingStatus.HATCHED))
        mgr = HatchingManager(state_path=sp, log_path=tmp_path / "l.jsonl")
        assert mgr.needs_hatching() is False

    def test_needs_hatching_false_for_in_progress(self, tmp_path):
        """IN_PROGRESS is not in the retry set — it means hatching is underway."""
        sp = tmp_path / "hatching.yaml"
        _write_state(sp, HatchingState(status=HatchingStatus.IN_PROGRESS))
        mgr = HatchingManager(state_path=sp, log_path=tmp_path / "l.jsonl")
        assert mgr.needs_hatching() is False

    def test_is_hatched_false_when_no_file(self, tmp_path):
        assert _make_manager(tmp_path).is_hatched() is False

    def test_is_hatched_true_when_status_hatched(self, tmp_path):
        sp = tmp_path / "hatching.yaml"
        _write_state(sp, HatchingState(status=HatchingStatus.HATCHED))
        mgr = HatchingManager(state_path=sp, log_path=tmp_path / "l.jsonl")
        assert mgr.is_hatched() is True

    def test_is_hatched_false_for_in_progress(self, tmp_path):
        sp = tmp_path / "hatching.yaml"
        _write_state(sp, HatchingState(status=HatchingStatus.IN_PROGRESS))
        mgr = HatchingManager(state_path=sp, log_path=tmp_path / "l.jsonl")
        assert mgr.is_hatched() is False


# ===========================================================================
# 7. run_hatching — happy path, resume, failure
# ===========================================================================


class TestRunHatching:
    """Full run_hatching flow with mocked subsystems."""

    def _run(self, tmp_path, monkeypatch, *, interactive=False) -> HatchingState:
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        _stub_external_deps(monkeypatch)

        mgr = _make_manager(tmp_path)
        return mgr.run_hatching(interactive=interactive)

    def test_returns_hatched_on_success(self, tmp_path, monkeypatch):
        state = self._run(tmp_path, monkeypatch)
        assert state.status is HatchingStatus.HATCHED

    def test_all_eight_steps_completed(self, tmp_path, monkeypatch):
        state = self._run(tmp_path, monkeypatch)
        expected = {
            "validate_environment",
            "initialize_config",
            "verify_providers",
            "initialize_security",
            "generate_persona",
            "check_vision",
            "seed_memory",
            "finalize",
        }
        assert expected.issubset(set(state.steps_completed))

    def test_started_at_and_completed_at_set(self, tmp_path, monkeypatch):
        state = self._run(tmp_path, monkeypatch)
        assert state.started_at is not None
        assert state.completed_at is not None

    def test_state_file_written_after_run(self, tmp_path, monkeypatch):
        self._run(tmp_path, monkeypatch)
        assert (tmp_path / "hatching.yaml").exists()

    def test_log_file_written_after_run(self, tmp_path, monkeypatch):
        self._run(tmp_path, monkeypatch)
        assert (tmp_path / "log.jsonl").exists()

    def test_short_circuits_when_already_hatched(self, tmp_path):
        sp = tmp_path / "hatching.yaml"
        _write_state(
            sp,
            HatchingState(
                status=HatchingStatus.HATCHED,
                completed_at="2026-01-01T00:00:00+00:00",
            ),
        )
        mgr = HatchingManager(state_path=sp, log_path=tmp_path / "l.jsonl")
        state = mgr.run_hatching(interactive=False)
        assert state.status is HatchingStatus.HATCHED
        assert state.completed_at == "2026-01-01T00:00:00+00:00"

    def test_resume_skips_already_completed_steps(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        _stub_external_deps(monkeypatch)

        sp = tmp_path / "hatching.yaml"
        _write_state(
            sp,
            HatchingState(
                status=HatchingStatus.IN_PROGRESS,
                started_at="2026-01-01T00:00:00+00:00",
                steps_completed=["validate_environment", "initialize_config"],
                environment_validated=True,
            ),
        )
        mgr = HatchingManager(state_path=sp, log_path=tmp_path / "l.jsonl")
        state = mgr.run_hatching(interactive=False)

        assert state.status is HatchingStatus.HATCHED
        # Step should appear exactly once in the final list.
        assert state.steps_completed.count("validate_environment") == 1

    def test_failed_state_cleared_on_retry(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        _stub_external_deps(monkeypatch)

        sp = tmp_path / "hatching.yaml"
        _write_state(
            sp,
            HatchingState(
                status=HatchingStatus.FAILED,
                error="Step 'seed_memory' failed: timeout",
            ),
        )
        mgr = HatchingManager(state_path=sp, log_path=tmp_path / "l.jsonl")
        state = mgr.run_hatching(interactive=False)

        assert state.status is HatchingStatus.HATCHED
        assert state.error is None

    def test_unrecoverable_step_exception_sets_failed(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        _stub_external_deps(monkeypatch)

        mgr = _make_manager(tmp_path)

        def _boom(state, *, interactive):
            raise RuntimeError("unexpected catastrophe")

        mgr._finalize = _boom  # type: ignore[method-assign]
        state = mgr.run_hatching(interactive=False)

        assert state.status is HatchingStatus.FAILED
        assert state.error is not None
        assert "unexpected catastrophe" in state.error

    def test_step_warning_is_non_fatal(self, tmp_path, monkeypatch):
        """A step raising _HatchingStepWarning must NOT abort the entire flow."""
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        _stub_external_deps(monkeypatch)

        mgr = _make_manager(tmp_path)

        def _warn_step(state, *, interactive):
            raise _HatchingStepWarning("just a heads-up")

        # Replace check_vision (already non-fatal by design) with our warning step.
        mgr._check_vision = _warn_step  # type: ignore[method-assign]
        state = mgr.run_hatching(interactive=False)

        # Hatching must still succeed overall.
        assert state.status is HatchingStatus.HATCHED
        # The step must still appear in steps_completed despite the warning.
        assert "check_vision" in state.steps_completed

    def test_interactive_mode_does_not_raise(self, tmp_path, monkeypatch, capsys):
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        _stub_external_deps(monkeypatch)

        mgr = _make_manager(tmp_path)
        state = mgr.run_hatching(interactive=True)
        assert state.status is HatchingStatus.HATCHED


# ===========================================================================
# 8. Individual step methods
# ===========================================================================


class TestValidateEnvironment:
    """_validate_environment: Python version check and disk-space warning."""

    def _call(self, mgr, *, interactive=False) -> None:
        state = HatchingState()
        mgr._validate_environment(state, interactive=interactive)

    def test_passes_on_current_python(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mgr = _make_manager(tmp_path)
        # Must not raise — we are running Python 3.11+.
        self._call(mgr)

    def test_sets_environment_validated_flag(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        mgr._validate_environment(state, interactive=False)
        assert state.environment_validated is True

    def test_raises_for_old_python(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        with patch.object(sys, "version_info", SimpleNamespace(major=3, minor=10, micro=0)), pytest.raises(RuntimeError, match="Python 3.11\\+"):
            mgr._validate_environment(state, interactive=False)

    def test_warns_on_low_disk_space(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        # Return a stat result that reports less than _MIN_FREE_BYTES free.
        fake_stat = SimpleNamespace(f_bavail=1, f_frsize=1)
        with patch("missy.agent.hatching.os.statvfs", return_value=fake_stat), pytest.raises(_HatchingStepWarning, match="Low disk space"):
            mgr._validate_environment(state, interactive=False)

    def test_sufficient_disk_space_does_not_warn(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        # Simulate plenty of free space.
        free = _MIN_FREE_BYTES * 10
        fake_stat = SimpleNamespace(f_bavail=free, f_frsize=1)
        with patch("missy.agent.hatching.os.statvfs", return_value=fake_stat):
            mgr._validate_environment(state, interactive=False)  # should not raise


class TestInitializeConfig:
    """_initialize_config: creates config only when absent."""

    def test_creates_default_config_when_absent(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mgr = _make_manager(tmp_path)
        config_path = tmp_path / "config.yaml"
        assert not config_path.exists()

        state = HatchingState()
        mgr._initialize_config(state, interactive=False)
        assert config_path.exists()

    def test_written_config_is_valid_yaml(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mgr = _make_manager(tmp_path)
        mgr._initialize_config(HatchingState(), interactive=False)
        config_path = tmp_path / "config.yaml"
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "providers" in data
        assert data.get("config_version") == 2

    def test_skips_when_config_already_exists(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        config_path = tmp_path / "config.yaml"
        config_path.write_text("# custom config\n", encoding="utf-8")

        mgr = _make_manager(tmp_path)
        mgr._initialize_config(HatchingState(), interactive=False)
        # Original content must be preserved.
        assert config_path.read_text(encoding="utf-8") == "# custom config\n"


class TestVerifyProviders:
    """_verify_providers: env vars and config file detection."""

    def test_sets_provider_verified_with_anthropic_env(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic-test")
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        mgr._verify_providers(state, interactive=False)
        assert state.provider_verified is True

    def test_sets_provider_verified_with_openai_env(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        mgr._verify_providers(state, interactive=False)
        assert state.provider_verified is True

    def test_warns_when_no_provider_key_found(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        with pytest.raises(_HatchingStepWarning, match="No provider API key"):
            mgr._verify_providers(state, interactive=False)
        assert state.provider_verified is False

    def test_reads_api_key_from_config_file(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config_path = tmp_path / "config.yaml"
        config_data = {
            "providers": {
                "anthropic": {"api_key": "sk-from-config"},
            }
        }
        config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        mgr._verify_providers(state, interactive=False)
        assert state.provider_verified is True

    def test_reads_api_keys_list_from_config(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config_path = tmp_path / "config.yaml"
        config_data = {
            "providers": {
                "anthropic": {"api_keys": ["sk-key1", "sk-key2"]},
            }
        }
        config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        mgr._verify_providers(state, interactive=False)
        assert state.provider_verified is True

    def test_check_config_no_file_returns_none(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mgr = _make_manager(tmp_path)
        assert mgr._check_config_for_provider_key() is None

    def test_check_config_empty_api_key_returns_none(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        config_path = tmp_path / "config.yaml"
        config_data = {"providers": {"anthropic": {"api_key": None}}}
        config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
        mgr = _make_manager(tmp_path)
        assert mgr._check_config_for_provider_key() is None


class TestInitializeSecurity:
    """_initialize_security: secrets directory creation."""

    def test_creates_secrets_directory(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        mgr._initialize_security(state, interactive=False)
        assert (tmp_path / "secrets").is_dir()

    def test_sets_security_initialized_flag(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        mgr._initialize_security(state, interactive=False)
        assert state.security_initialized is True

    def test_handles_missing_identity_key_non_fatally(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        # identity.pem not created — should not raise, just log warning
        mgr._initialize_security(state, interactive=False)
        assert state.security_initialized is True

    def test_identity_present_is_logged(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        (tmp_path / "identity.pem").write_text("FAKE PEM", encoding="utf-8")
        log_path = tmp_path / "log.jsonl"
        mgr = HatchingManager(state_path=tmp_path / "hatching.yaml", log_path=log_path)
        state = HatchingState()
        mgr._initialize_security(state, interactive=False)
        entries = HatchingLog(log_path=log_path).get_entries()
        details_list = [e.get("details", {}) for e in entries]
        assert any(d.get("identity_present") is True for d in details_list)


class TestGeneratePersona:
    """_generate_persona: delegates to PersonaManager, handles errors."""

    def test_sets_persona_generated_flag(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mock_pm = MagicMock()
        monkeypatch.setattr("missy.agent.persona.PersonaManager", mock_pm)
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        mgr._generate_persona(state, interactive=False)
        assert state.persona_generated is True

    def test_skips_if_persona_file_already_exists(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        persona_path = tmp_path / "persona.yaml"
        persona_path.write_text("name: Missy\n", encoding="utf-8")
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        # PersonaManager should NOT be imported/called if file already exists.
        with patch("missy.agent.hatching.Path.exists", return_value=True):
            mgr._generate_persona(state, interactive=False)
        assert state.persona_generated is True

    def test_import_error_raises_step_warning(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        # Simulate ImportError for PersonaManager.
        with patch("builtins.__import__", side_effect=ImportError("no module")), pytest.raises(_HatchingStepWarning, match="Could not import PersonaManager"):
            mgr._generate_persona(state, interactive=False)

    def test_oserror_on_write_raises_step_warning(self, tmp_path, monkeypatch):
        import missy.agent.hatching as _h

        _patch_module_paths(monkeypatch, tmp_path)
        # Ensure persona file does not exist so the early-return is skipped.
        if _h._PERSONA_PATH.exists():
            _h._PERSONA_PATH.unlink()
        mock_pm_instance = MagicMock()
        mock_pm_instance.save.side_effect = OSError("permission denied")
        mock_pm_class = MagicMock(return_value=mock_pm_instance)
        # Patch the PersonaManager at the point of local import inside
        # _generate_persona.  We wrap the real __import__ so only the
        # specific import statement is intercepted.
        _real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        def _fake_import(name, *args, **kwargs):
            if name == "missy.agent.persona":
                import types
                mod = types.ModuleType("missy.agent.persona")
                mod.PersonaManager = mock_pm_class
                return mod
            return _real_import(name, *args, **kwargs)
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        with patch("builtins.__import__", side_effect=_fake_import), \
             pytest.raises(_HatchingStepWarning, match="Could not write persona"):
            mgr._generate_persona(state, interactive=False)


class TestSeedMemory:
    """_seed_memory: delegates to SQLiteMemoryStore, handles errors."""

    def test_sets_memory_seeded_flag(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mock_turn = MagicMock()
        mock_turn.id = "welcome-id"
        mock_store = MagicMock()
        monkeypatch.setattr("missy.memory.sqlite_store.SQLiteMemoryStore", MagicMock(return_value=mock_store))
        monkeypatch.setattr(
            "missy.memory.sqlite_store.ConversationTurn",
            MagicMock(new=MagicMock(return_value=mock_turn)),
        )
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        mgr._seed_memory(state, interactive=False)
        assert state.memory_seeded is True

    def test_import_error_raises_step_warning(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        with patch("builtins.__import__", side_effect=ImportError("no sqlite")), pytest.raises(_HatchingStepWarning, match="Could not import SQLiteMemoryStore"):
            mgr._seed_memory(state, interactive=False)

    def test_generic_exception_raises_step_warning(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mock_store = MagicMock()
        mock_store.add_turn.side_effect = RuntimeError("DB locked")
        monkeypatch.setattr(
            "missy.memory.sqlite_store.SQLiteMemoryStore", MagicMock(return_value=mock_store)
        )
        mock_turn = MagicMock()
        mock_turn.id = "t"
        monkeypatch.setattr(
            "missy.memory.sqlite_store.ConversationTurn",
            MagicMock(new=MagicMock(return_value=mock_turn)),
        )
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        with pytest.raises(_HatchingStepWarning, match="Memory seeding failed"):
            mgr._seed_memory(state, interactive=False)


class TestFinalize:
    """_finalize: marks state HATCHED and sets timestamps."""

    def test_sets_status_to_hatched(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mgr = _make_manager(tmp_path)
        state = HatchingState(status=HatchingStatus.IN_PROGRESS)
        mgr._finalize(state, interactive=False)
        assert state.status is HatchingStatus.HATCHED

    def test_sets_completed_at(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mgr = _make_manager(tmp_path)
        state = HatchingState()
        mgr._finalize(state, interactive=False)
        assert state.completed_at is not None

    def test_clears_error_field(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        mgr = _make_manager(tmp_path)
        state = HatchingState(error="previous error message")
        mgr._finalize(state, interactive=False)
        assert state.error is None

    def test_persists_state_after_finalize(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        state_path = tmp_path / "hatching.yaml"
        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "l.jsonl")
        state = HatchingState(status=HatchingStatus.IN_PROGRESS)
        mgr._finalize(state, interactive=False)
        on_disk = mgr.get_state()
        assert on_disk.status is HatchingStatus.HATCHED


# ===========================================================================
# 9. reset and get_hatching_log
# ===========================================================================


class TestReset:
    """reset() removes state, preserves log."""

    def test_reset_removes_state_file(self, tmp_path):
        sp = tmp_path / "hatching.yaml"
        _write_state(sp, HatchingState())
        mgr = HatchingManager(state_path=sp, log_path=tmp_path / "l.jsonl")
        mgr.reset()
        assert not sp.exists()

    def test_reset_when_file_absent_does_not_raise(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.reset()  # must not raise

    def test_needs_hatching_true_after_reset(self, tmp_path):
        sp = tmp_path / "hatching.yaml"
        _write_state(sp, HatchingState(status=HatchingStatus.HATCHED))
        mgr = HatchingManager(state_path=sp, log_path=tmp_path / "l.jsonl")
        assert not mgr.needs_hatching()
        mgr.reset()
        assert mgr.needs_hatching()

    def test_reset_writes_log_entry(self, tmp_path):
        log_path = tmp_path / "l.jsonl"
        mgr = HatchingManager(state_path=tmp_path / "h.yaml", log_path=log_path)
        mgr.reset()
        entries = HatchingLog(log_path=log_path).get_entries()
        assert any(e["step"] == "reset" for e in entries)

    def test_reset_preserves_prior_log_entries(self, tmp_path):
        log_path = tmp_path / "l.jsonl"
        log = HatchingLog(log_path=log_path)
        log.log("prior_step", "ok", "before reset")

        sp = tmp_path / "h.yaml"
        _write_state(sp, HatchingState())
        mgr = HatchingManager(state_path=sp, log_path=log_path)
        mgr.reset()

        entries = HatchingLog(log_path=log_path).get_entries()
        assert any(e["step"] == "prior_step" for e in entries)


class TestGetHatchingLog:
    """get_hatching_log delegates to HatchingLog."""

    def test_empty_initially(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.get_hatching_log() == []

    def test_reflects_entries_after_run(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        _stub_external_deps(monkeypatch)
        mgr = _make_manager(tmp_path)
        mgr.run_hatching(interactive=False)
        entries = mgr.get_hatching_log()
        assert len(entries) > 0
        assert any(e["step"] == "hatching" for e in entries)
        assert any(e["step"] == "finalize" for e in entries)


# ===========================================================================
# 10. _HatchingStepWarning sentinel
# ===========================================================================


class TestHatchingStepWarning:
    """Verify the sentinel exception behaves like a normal Exception."""

    def test_is_exception_subclass(self):
        assert issubclass(_HatchingStepWarning, Exception)

    def test_str_contains_message(self):
        w = _HatchingStepWarning("test warning text")
        assert "test warning text" in str(w)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(_HatchingStepWarning):
            raise _HatchingStepWarning("raised")
