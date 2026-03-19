"""Session 13 — edge-case tests for hatching and checkpoint subsystems.

Covers gaps not addressed by the existing test suites:
- HatchingManager: first-run detection logic, step completion tracking,
  vision readiness, provider setup verification, security baseline init,
  non-interactive mode, state persistence, resume from partial hatching,
  missing/corrupt state file edge cases.
- CheckpointManager: serialisation round-trips, ordering guarantees,
  expiration/cleanup boundaries, corrupt-data resilience, concurrent
  access from multiple threads, and full scan_for_recovery scenarios.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from missy.agent.checkpoint import (
    _RESTART_THRESHOLD_SECS,
    _RESUME_THRESHOLD_SECS,
    CheckpointManager,
    scan_for_recovery,
)
from missy.agent.hatching import (
    HatchingLog,
    HatchingManager,
    HatchingState,
    HatchingStatus,
    _HatchingStepWarning,
)
from missy.core.events import event_bus

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_manager(tmp_path: Path) -> HatchingManager:
    return HatchingManager(
        state_path=tmp_path / "hatching.yaml",
        log_path=tmp_path / "log.jsonl",
    )


def _patch_module_paths(monkeypatch, tmp_path: Path) -> None:
    import missy.agent.hatching as m

    monkeypatch.setattr(m, "_MISSY_DIR", tmp_path)
    monkeypatch.setattr(m, "_CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(m, "_IDENTITY_PATH", tmp_path / "identity.pem")
    monkeypatch.setattr(m, "_SECRETS_DIR", tmp_path / "secrets")
    monkeypatch.setattr(m, "_PERSONA_PATH", tmp_path / "persona.yaml")
    monkeypatch.setattr(m, "_MEMORY_DB_PATH", tmp_path / "memory.db")


def _mock_persona_and_memory():
    """Context-manager that stubs out PersonaManager and SQLiteMemoryStore."""
    mock_turn = MagicMock()
    mock_turn.id = "turn-hatching"
    mock_store = MagicMock()
    mock_pm = MagicMock()

    return (
        patch("missy.agent.persona.PersonaManager", return_value=mock_pm),
        patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
        patch(
            "missy.memory.sqlite_store.ConversationTurn",
            MagicMock(new=MagicMock(return_value=mock_turn)),
        ),
    )


@pytest.fixture(autouse=True)
def clear_event_bus_fixture():
    event_bus.clear()
    yield
    event_bus.clear()


@pytest.fixture()
def tmp_db(tmp_path):
    return str(tmp_path / "checkpoints.db")


@pytest.fixture()
def cm(tmp_db):
    return CheckpointManager(db_path=tmp_db)


def _raw_query(tmp_db: str, sql: str, params: tuple = ()):
    with sqlite3.connect(tmp_db) as conn:
        return conn.execute(sql, params).fetchone()


def _raw_exec(tmp_db: str, sql: str, params: tuple = ()) -> None:
    with sqlite3.connect(tmp_db) as conn:
        conn.execute(sql, params)
        conn.commit()


# ===========================================================================
# HATCHING — first-run detection logic
# ===========================================================================


class TestFirstRunDetection:
    def test_no_state_file_triggers_needs_hatching(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.needs_hatching() is True

    def test_in_progress_status_does_not_satisfy_needs_hatching(self, tmp_path):
        """IN_PROGRESS is treated as incomplete; needs_hatching should be False
        (only UNHATCHED / FAILED require re-entry), but agent is not done."""
        state_path = tmp_path / "hatching.yaml"
        state_path.write_text(
            yaml.safe_dump({"status": "in_progress"}), encoding="utf-8"
        )
        mgr = _make_manager(tmp_path)
        # IN_PROGRESS is not UNHATCHED/FAILED so needs_hatching returns False
        assert mgr.needs_hatching() is False
        assert mgr.is_hatched() is False

    def test_hatched_status_clears_needs_hatching(self, tmp_path):
        state_path = tmp_path / "hatching.yaml"
        state = HatchingState(status=HatchingStatus.HATCHED)
        state_path.write_text(yaml.safe_dump(state.to_dict()), encoding="utf-8")
        mgr = _make_manager(tmp_path)
        assert mgr.needs_hatching() is False
        assert mgr.is_hatched() is True

    def test_failed_status_requires_rehatching(self, tmp_path):
        state_path = tmp_path / "hatching.yaml"
        state = HatchingState(status=HatchingStatus.FAILED, error="boom")
        state_path.write_text(yaml.safe_dump(state.to_dict()), encoding="utf-8")
        mgr = _make_manager(tmp_path)
        assert mgr.needs_hatching() is True

    def test_state_file_with_zero_byte_content_returns_unhatched(self, tmp_path):
        state_path = tmp_path / "hatching.yaml"
        state_path.write_bytes(b"")
        mgr = _make_manager(tmp_path)
        assert mgr.get_state().status is HatchingStatus.UNHATCHED

    def test_state_file_containing_only_whitespace_returns_unhatched(self, tmp_path):
        state_path = tmp_path / "hatching.yaml"
        state_path.write_text("   \n\t\n", encoding="utf-8")
        mgr = _make_manager(tmp_path)
        assert mgr.get_state().status is HatchingStatus.UNHATCHED


# ===========================================================================
# HATCHING — step completion tracking
# ===========================================================================


class TestStepCompletionTracking:
    def test_completed_step_not_duplicated_in_list(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        state_path = tmp_path / "hatching.yaml"
        partial = HatchingState(
            status=HatchingStatus.IN_PROGRESS,
            started_at="2026-01-01T00:00:00+00:00",
            steps_completed=["validate_environment"],
            environment_validated=True,
        )
        state_path.write_text(yaml.safe_dump(partial.to_dict()), encoding="utf-8")

        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "key-x")
            mgr = HatchingManager(
                state_path=state_path, log_path=tmp_path / "log.jsonl"
            )
            state = mgr.run_hatching(interactive=False)

        assert state.steps_completed.count("validate_environment") == 1

    def test_all_eight_canonical_steps_appear_in_completed(
        self, tmp_path, monkeypatch
    ):
        _patch_module_paths(monkeypatch, tmp_path)
        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "key-y")
            mgr = _make_manager(tmp_path)
            state = mgr.run_hatching(interactive=False)

        canonical = {
            "validate_environment",
            "initialize_config",
            "verify_providers",
            "initialize_security",
            "generate_persona",
            "check_vision",
            "seed_memory",
            "finalize",
        }
        assert canonical.issubset(set(state.steps_completed))

    def test_warning_step_still_marked_completed(self, tmp_path, monkeypatch):
        """A step that raises _HatchingStepWarning must still land in steps_completed."""
        _patch_module_paths(monkeypatch, tmp_path)

        def _warn_step(state, *, interactive):
            raise _HatchingStepWarning("non-fatal warning from test")

        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "key-z")
            mgr = _make_manager(tmp_path)
            mgr._check_vision = _warn_step  # type: ignore[method-assign]
            state = mgr.run_hatching(interactive=False)

        assert "check_vision" in state.steps_completed
        assert state.status is HatchingStatus.HATCHED

    def test_fatal_step_stops_subsequent_steps(self, tmp_path, monkeypatch):
        """RuntimeError in a step must prevent all later steps from running."""
        _patch_module_paths(monkeypatch, tmp_path)

        executed: list[str] = []

        def _fail_security(state, *, interactive):
            raise RuntimeError("security init exploded")

        def _track_persona(state, *, interactive):
            executed.append("generate_persona")

        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "key-z")
            mgr = _make_manager(tmp_path)
            mgr._initialize_security = _fail_security  # type: ignore[method-assign]
            mgr._generate_persona = _track_persona  # type: ignore[method-assign]
            state = mgr.run_hatching(interactive=False)

        assert state.status is HatchingStatus.FAILED
        assert "generate_persona" not in executed

    def test_state_persisted_after_each_step(self, tmp_path, monkeypatch):
        """State file on disk must reflect the step just completed, not wait for all."""
        _patch_module_paths(monkeypatch, tmp_path)
        state_path = tmp_path / "hatching.yaml"

        seen_on_disk: list[list[str]] = []

        original_save = HatchingManager._save_state

        def _spy_save(self_mgr, state):
            original_save(self_mgr, state)
            seen_on_disk.append(list(state.steps_completed))

        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3, patch.object(HatchingManager, "_save_state", _spy_save):
            monkeypatch.setenv("ANTHROPIC_API_KEY", "key-persist")
            mgr = HatchingManager(
                state_path=state_path, log_path=tmp_path / "log.jsonl"
            )
            mgr.run_hatching(interactive=False)

        # Must have been saved multiple times (at least once per step + initial)
        assert len(seen_on_disk) > 2


# ===========================================================================
# HATCHING — vision readiness check
# ===========================================================================


class TestVisionReadinessCheck:
    def test_vision_step_warning_does_not_fail_hatching(self, tmp_path, monkeypatch):
        """check_vision raises a warning; hatching must still complete as HATCHED."""
        _patch_module_paths(monkeypatch, tmp_path)

        def _warn_vision(state, *, interactive):
            raise _HatchingStepWarning("no cameras detected")

        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
            mgr = _make_manager(tmp_path)
            mgr._check_vision = _warn_vision  # type: ignore[method-assign]
            state = mgr.run_hatching(interactive=False)

        assert state.status is HatchingStatus.HATCHED
        assert "check_vision" in state.steps_completed

    def test_vision_opencv_missing_produces_warning_entry_in_log(
        self, tmp_path, monkeypatch
    ):
        """When cv2 is not importable, the log must record a 'warn' entry for check_vision."""
        _patch_module_paths(monkeypatch, tmp_path)
        log_path = tmp_path / "log.jsonl"

        import sys

        saved = sys.modules.get("cv2")
        sys.modules["cv2"] = None  # type: ignore[assignment]
        try:
            p1, p2, p3 = _mock_persona_and_memory()
            # Also stub out discover_cameras so we don't need real vision module
            with (
                p1,
                p2,
                p3,
                patch(
                    "missy.vision.discovery.discover_cameras",
                    side_effect=ImportError("no cv2"),
                ),
                patch(
                    "missy.vision.doctor.VisionDoctor",
                    side_effect=Exception("no doctor"),
                ),
            ):
                monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
                mgr = HatchingManager(
                    state_path=tmp_path / "hatching.yaml", log_path=log_path
                )
                mgr.run_hatching(interactive=False)
        finally:
            if saved is None:
                sys.modules.pop("cv2", None)
            else:
                sys.modules["cv2"] = saved

        log = HatchingLog(log_path=log_path)
        vision_entries = [e for e in log.get_entries() if e["step"] == "check_vision"]
        # There should be at least a warn entry for check_vision
        statuses = {e["status"] for e in vision_entries}
        assert "warn" in statuses


# ===========================================================================
# HATCHING — provider setup verification
# ===========================================================================


class TestProviderVerification:
    def test_anthropic_env_var_satisfies_provider_check(
        self, tmp_path, monkeypatch
    ):
        _patch_module_paths(monkeypatch, tmp_path)
        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
            monkeypatch.delenv("OPENAI_API_KEY", raising=False)
            mgr = _make_manager(tmp_path)
            state = mgr.run_hatching(interactive=False)

        assert state.provider_verified is True

    def test_openai_env_var_satisfies_provider_check(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
            monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
            mgr = _make_manager(tmp_path)
            state = mgr.run_hatching(interactive=False)

        assert state.provider_verified is True

    def test_no_api_key_produces_warn_step_and_provider_not_verified(
        self, tmp_path, monkeypatch
    ):
        _patch_module_paths(monkeypatch, tmp_path)
        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
            monkeypatch.delenv("OPENAI_API_KEY", raising=False)
            mgr = _make_manager(tmp_path)
            state = mgr.run_hatching(interactive=False)

        # Warning treated as non-fatal; hatching should still succeed
        assert state.status is HatchingStatus.HATCHED
        assert state.provider_verified is False
        assert "verify_providers" in state.steps_completed

    def test_config_file_api_key_satisfies_provider_check(
        self, tmp_path, monkeypatch
    ):
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Write a config with an explicit api_key
        config_data = {
            "providers": {
                "anthropic": {"api_key": "from-config-file", "model": "claude-sonnet-4-6"}
            }
        }
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(config_data), encoding="utf-8"
        )

        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            mgr = _make_manager(tmp_path)
            state = mgr.run_hatching(interactive=False)

        assert state.provider_verified is True

    def test_config_api_keys_list_satisfies_provider_check(
        self, tmp_path, monkeypatch
    ):
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        config_data = {
            "providers": {
                "anthropic": {
                    "api_keys": ["key-one", "key-two"],
                    "model": "claude-sonnet-4-6",
                }
            }
        }
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(config_data), encoding="utf-8"
        )

        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            mgr = _make_manager(tmp_path)
            state = mgr.run_hatching(interactive=False)

        assert state.provider_verified is True

    def test_config_with_null_api_key_not_satisfied(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        config_data = {
            "providers": {"anthropic": {"api_key": None, "model": "claude-sonnet-4-6"}}
        }
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(config_data), encoding="utf-8"
        )

        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            mgr = _make_manager(tmp_path)
            state = mgr.run_hatching(interactive=False)

        assert state.provider_verified is False


# ===========================================================================
# HATCHING — security baseline initialisation
# ===========================================================================


class TestSecurityBaselineInit:
    def test_secrets_directory_created_with_restricted_mode(
        self, tmp_path, monkeypatch
    ):
        _patch_module_paths(monkeypatch, tmp_path)
        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
            mgr = _make_manager(tmp_path)
            mgr.run_hatching(interactive=False)

        secrets_dir = tmp_path / "secrets"
        assert secrets_dir.exists()
        mode = oct(secrets_dir.stat().st_mode & 0o777)
        assert mode == oct(0o700)

    def test_security_initialized_flag_set_after_step(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
            mgr = _make_manager(tmp_path)
            state = mgr.run_hatching(interactive=False)

        assert state.security_initialized is True

    def test_log_records_security_step_ok(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        log_path = tmp_path / "log.jsonl"
        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
            mgr = HatchingManager(
                state_path=tmp_path / "hatching.yaml", log_path=log_path
            )
            mgr.run_hatching(interactive=False)

        entries = HatchingLog(log_path=log_path).get_entries()
        security_entries = [
            e for e in entries if e["step"] == "initialize_security"
        ]
        assert any(e["status"] == "ok" for e in security_entries)


# ===========================================================================
# HATCHING — non-interactive mode
# ===========================================================================


class TestNonInteractiveMode:
    def test_non_interactive_run_does_not_print(
        self, tmp_path, monkeypatch, capsys
    ):
        _patch_module_paths(monkeypatch, tmp_path)
        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
            mgr = _make_manager(tmp_path)
            mgr.run_hatching(interactive=False)

        out, _ = capsys.readouterr()
        assert out == ""

    def test_interactive_run_prints_step_labels(
        self, tmp_path, monkeypatch, capsys
    ):
        _patch_module_paths(monkeypatch, tmp_path)
        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
            mgr = _make_manager(tmp_path)
            mgr.run_hatching(interactive=True)

        out, _ = capsys.readouterr()
        assert "validate_environment" in out

    def test_skipped_steps_print_in_interactive_mode(
        self, tmp_path, monkeypatch, capsys
    ):
        _patch_module_paths(monkeypatch, tmp_path)
        state_path = tmp_path / "hatching.yaml"
        partial = HatchingState(
            status=HatchingStatus.IN_PROGRESS,
            started_at="2026-01-01T00:00:00+00:00",
            steps_completed=["validate_environment"],
            environment_validated=True,
        )
        state_path.write_text(yaml.safe_dump(partial.to_dict()), encoding="utf-8")

        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
            mgr = HatchingManager(
                state_path=state_path, log_path=tmp_path / "log.jsonl"
            )
            mgr.run_hatching(interactive=True)

        out, _ = capsys.readouterr()
        assert "skip" in out.lower()


# ===========================================================================
# HATCHING — state persistence and resume
# ===========================================================================


class TestStatePersistenceAndResume:
    def test_state_file_written_atomically_via_tmp(self, tmp_path, monkeypatch):
        """_save_state must use a .tmp file that is then renamed."""
        _patch_module_paths(monkeypatch, tmp_path)
        state_path = tmp_path / "hatching.yaml"

        writes: list[str] = []
        original_replace = Path.replace

        def _spy_replace(self_path, target):
            writes.append(str(self_path))
            return original_replace(self_path, target)

        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3, patch.object(Path, "replace", _spy_replace):
            monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
            mgr = HatchingManager(
                state_path=state_path, log_path=tmp_path / "log.jsonl"
            )
            mgr.run_hatching(interactive=False)

        # Every write should have come from the .yaml.tmp temp file
        assert all(".yaml.tmp" in w for w in writes), writes

    def test_resume_from_partial_preserves_already_done_flags(
        self, tmp_path, monkeypatch
    ):
        _patch_module_paths(monkeypatch, tmp_path)
        state_path = tmp_path / "hatching.yaml"
        partial = HatchingState(
            status=HatchingStatus.IN_PROGRESS,
            started_at="2026-01-01T00:00:00+00:00",
            steps_completed=[
                "validate_environment",
                "initialize_config",
                "verify_providers",
                "initialize_security",
            ],
            environment_validated=True,
            provider_verified=True,
            security_initialized=True,
        )
        state_path.write_text(yaml.safe_dump(partial.to_dict()), encoding="utf-8")

        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
            mgr = HatchingManager(
                state_path=state_path, log_path=tmp_path / "log.jsonl"
            )
            state = mgr.run_hatching(interactive=False)

        assert state.status is HatchingStatus.HATCHED
        # Original started_at must be preserved (not reset)
        assert state.started_at == "2026-01-01T00:00:00+00:00"

    def test_completed_at_set_on_successful_hatching(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
            mgr = _make_manager(tmp_path)
            state = mgr.run_hatching(interactive=False)

        assert state.completed_at is not None

    def test_error_field_cleared_on_retry_after_failure(self, tmp_path, monkeypatch):
        _patch_module_paths(monkeypatch, tmp_path)
        state_path = tmp_path / "hatching.yaml"
        failed = HatchingState(
            status=HatchingStatus.FAILED,
            error="Step 'initialize_security' failed: OSError",
        )
        state_path.write_text(yaml.safe_dump(failed.to_dict()), encoding="utf-8")

        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
            mgr = HatchingManager(
                state_path=state_path, log_path=tmp_path / "log.jsonl"
            )
            state = mgr.run_hatching(interactive=False)

        assert state.error is None
        assert state.status is HatchingStatus.HATCHED

    def test_corrupt_state_file_triggers_fresh_hatching(
        self, tmp_path, monkeypatch
    ):
        """A state file with invalid YAML (but valid UTF-8) must fall back to
        a fresh UNHATCHED state, allowing hatching to run from scratch."""
        _patch_module_paths(monkeypatch, tmp_path)
        state_path = tmp_path / "hatching.yaml"
        # This is valid UTF-8 but structurally invalid YAML (duplicate mapping keys
        # with malformed syntax causes yaml.safe_load to raise YAMLError).
        state_path.write_text(
            ": not: valid: yaml: {unclosed bracket",
            encoding="utf-8",
        )

        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
            mgr = HatchingManager(
                state_path=state_path, log_path=tmp_path / "log.jsonl"
            )
            state = mgr.run_hatching(interactive=False)

        # Fresh hatching from corrupt state should succeed
        assert state.status is HatchingStatus.HATCHED

    def test_missing_config_creates_default_config_file(
        self, tmp_path, monkeypatch
    ):
        _patch_module_paths(monkeypatch, tmp_path)
        config_path = tmp_path / "config.yaml"
        assert not config_path.exists()

        p1, p2, p3 = _mock_persona_and_memory()
        with p1, p2, p3:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
            mgr = _make_manager(tmp_path)
            mgr.run_hatching(interactive=False)

        assert config_path.exists()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config.get("config_version") == 2


# ===========================================================================
# HATCHING — HatchingState from_dict edge cases
# ===========================================================================


class TestHatchingStateFromDictEdgeCases:
    def test_steps_completed_with_integer_items_coerced(self):
        """Non-string items in steps_completed list should be preserved."""
        state = HatchingState.from_dict({"steps_completed": [1, 2, "three"]})
        assert 1 in state.steps_completed
        assert "three" in state.steps_completed

    def test_all_status_values_parse_correctly(self):
        for status in HatchingStatus:
            state = HatchingState.from_dict({"status": status.value})
            assert state.status is status

    def test_error_field_preserved_round_trip(self):
        original = HatchingState(
            status=HatchingStatus.FAILED, error="some detailed error"
        )
        restored = HatchingState.from_dict(original.to_dict())
        assert restored.error == "some detailed error"

    def test_to_dict_includes_all_fields(self):
        state = HatchingState()
        d = state.to_dict()
        expected_keys = {
            "status",
            "started_at",
            "completed_at",
            "steps_completed",
            "persona_generated",
            "environment_validated",
            "provider_verified",
            "security_initialized",
            "memory_seeded",
            "error",
        }
        assert expected_keys.issubset(set(d.keys()))


# ===========================================================================
# CHECKPOINT — creation and serialisation
# ===========================================================================


class TestCheckpointCreationAndSerialisation:
    def test_initial_loop_messages_defaults_to_empty_json_array(
        self, cm, tmp_db
    ):
        cid = cm.create("s", "t", "p")
        raw = _raw_query(
            tmp_db, "SELECT loop_messages FROM checkpoints WHERE id=?", (cid,)
        )[0]
        assert json.loads(raw) == []

    def test_initial_tool_names_defaults_to_empty_json_array(
        self, cm, tmp_db
    ):
        cid = cm.create("s", "t", "p")
        raw = _raw_query(
            tmp_db, "SELECT tool_names_used FROM checkpoints WHERE id=?", (cid,)
        )[0]
        assert json.loads(raw) == []

    def test_initial_iteration_defaults_to_zero(self, cm, tmp_db):
        cid = cm.create("s", "t", "p")
        row = _raw_query(
            tmp_db, "SELECT iteration FROM checkpoints WHERE id=?", (cid,)
        )
        assert row[0] == 0

    def test_update_then_get_incomplete_round_trips_messages(self, cm):
        cid = cm.create("session", "task", "prompt text")
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        cm.update(cid, messages, ["read_file", "write_file"], iteration=5)
        incomplete = cm.get_incomplete()
        row = next(r for r in incomplete if r["id"] == cid)
        assert row["loop_messages"] == messages
        assert row["tool_names_used"] == ["read_file", "write_file"]
        assert row["iteration"] == 5

    def test_update_multiple_times_keeps_last_state(self, cm):
        cid = cm.create("s", "t", "p")
        cm.update(cid, [{"role": "user", "content": "v1"}], ["t1"], iteration=1)
        cm.update(cid, [{"role": "user", "content": "v2"}], ["t2"], iteration=2)
        row = next(r for r in cm.get_incomplete() if r["id"] == cid)
        assert row["loop_messages"][0]["content"] == "v2"
        assert row["iteration"] == 2


# ===========================================================================
# CHECKPOINT — multiple checkpoints ordering
# ===========================================================================


class TestMultipleCheckpointsOrdering:
    def test_get_incomplete_ordered_by_created_at_ascending(
        self, cm, tmp_db
    ):
        ids = []
        for i in range(5):
            cid = cm.create(f"session-{i}", f"task-{i}", f"prompt {i}")
            ids.append(cid)
            # Stagger timestamps slightly to ensure deterministic order
            _raw_exec(
                tmp_db,
                "UPDATE checkpoints SET created_at=? WHERE id=?",
                (time.time() - (5 - i) * 10, cid),
            )

        incomplete = cm.get_incomplete()
        returned_ids = [r["id"] for r in incomplete]
        # The checkpoint created earliest (smallest created_at) should come first
        assert returned_ids == sorted(
            ids, key=lambda c: _raw_query(tmp_db, "SELECT created_at FROM checkpoints WHERE id=?", (c,))[0]
        )

    def test_mixed_states_only_running_returned(self, cm):
        running_id = cm.create("s", "t", "running")
        complete_id = cm.create("s", "t", "complete")
        failed_id = cm.create("s", "t", "failed")

        cm.complete(complete_id)
        cm.fail(failed_id)

        incomplete = cm.get_incomplete()
        ids = [r["id"] for r in incomplete]
        assert running_id in ids
        assert complete_id not in ids
        assert failed_id not in ids


# ===========================================================================
# CHECKPOINT — expiration and cleanup
# ===========================================================================


class TestCheckpointExpirationAndCleanup:
    def test_classify_exactly_at_resume_boundary_is_restart(self, cm):
        """At exactly 3600 seconds (not strictly less than), action should be 'restart'."""
        cp = {"created_at": time.time() - _RESUME_THRESHOLD_SECS}
        # age == threshold => not < threshold => restart
        assert cm.classify(cp) == "restart"

    def test_classify_exactly_at_restart_boundary_is_abandon(self, cm):
        """At exactly 86400 seconds, action should be 'abandon'."""
        cp = {"created_at": time.time() - _RESTART_THRESHOLD_SECS}
        assert cm.classify(cp) == "abandon"

    def test_abandon_old_only_affects_running_state(self, cm, tmp_db):
        """COMPLETE checkpoints older than max_age must NOT be transitioned to ABANDONED."""
        cid = cm.create("s", "t", "p")
        cm.complete(cid)
        _raw_exec(
            tmp_db,
            "UPDATE checkpoints SET created_at=? WHERE id=?",
            (time.time() - 90_000, cid),
        )
        count = cm.abandon_old()
        assert count == 0
        state = _raw_query(
            tmp_db, "SELECT state FROM checkpoints WHERE id=?", (cid,)
        )[0]
        assert state == "COMPLETE"

    def test_cleanup_with_zero_days_removes_all_terminal(self, cm, tmp_db):
        for _ in range(3):
            cid = cm.create("s", "t", "p")
            cm.complete(cid)
        # older_than_days=0 uses cutoff = now; all completed rows are "older"
        count = cm.cleanup(older_than_days=0)
        assert count == 3

    def test_cleanup_returns_zero_when_nothing_to_clean(self, cm):
        cm.create("s", "t", "p")
        # Running state must not be cleaned
        count = cm.cleanup(older_than_days=0)
        assert count == 0

    def test_cleanup_mixed_terminal_states_all_removed(self, cm, tmp_db):
        for state_fn in (cm.complete, cm.fail):
            cid = cm.create("s", "t", "p")
            state_fn(cid)
            _raw_exec(
                tmp_db,
                "UPDATE checkpoints SET updated_at=? WHERE id=?",
                (time.time() - 10 * 86400, cid),
            )
        count = cm.cleanup(older_than_days=7)
        assert count == 2


# ===========================================================================
# CHECKPOINT — corrupt data resilience
# ===========================================================================


class TestCorruptDataResilience:
    def test_corrupt_loop_messages_json_returns_empty_list(
        self, cm, tmp_db
    ):
        """Mangled JSON in loop_messages must degrade gracefully to []."""
        cid = cm.create("s", "t", "p")
        _raw_exec(
            tmp_db,
            "UPDATE checkpoints SET loop_messages=? WHERE id=?",
            ("NOT_VALID_JSON!!!", cid),
        )
        incomplete = cm.get_incomplete()
        row = next(r for r in incomplete if r["id"] == cid)
        assert row["loop_messages"] == []

    def test_corrupt_tool_names_json_returns_empty_list(self, cm, tmp_db):
        cid = cm.create("s", "t", "p")
        _raw_exec(
            tmp_db,
            "UPDATE checkpoints SET tool_names_used=? WHERE id=?",
            ("{bad json}", cid),
        )
        incomplete = cm.get_incomplete()
        row = next(r for r in incomplete if r["id"] == cid)
        assert row["tool_names_used"] == []

    def test_fail_with_long_error_appended_to_prompt(self, cm, tmp_db):
        long_error = "E" * 2000
        cid = cm.create("s", "t", "original")
        cm.fail(cid, error=long_error)
        prompt = _raw_query(
            tmp_db, "SELECT prompt FROM checkpoints WHERE id=?", (cid,)
        )[0]
        assert long_error in prompt
        assert "original" in prompt


# ===========================================================================
# CHECKPOINT — concurrent access
# ===========================================================================


class TestConcurrentCheckpointAccess:
    def test_multiple_threads_can_create_checkpoints_without_deadlock(
        self, tmp_db
    ):
        """Each thread creates 5 checkpoints; total must be all 25."""
        cm = CheckpointManager(db_path=tmp_db)
        created: list[str] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def _worker(thread_id: int) -> None:
            try:
                for i in range(5):
                    cid = cm.create(
                        f"session-{thread_id}",
                        f"task-{thread_id}-{i}",
                        f"prompt {thread_id}:{i}",
                    )
                    with lock:
                        created.append(cid)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert len(created) == 25
        assert len(set(created)) == 25  # all unique UUIDs

    def test_concurrent_complete_and_read_does_not_corrupt(self, tmp_db):
        """One thread completes checkpoints while another reads incomplete list."""
        cm = CheckpointManager(db_path=tmp_db)
        ids = [cm.create("s", f"task-{i}", "p") for i in range(20)]
        errors: list[Exception] = []

        def _completer():
            try:
                for cid in ids[:10]:
                    cm.complete(cid)
            except Exception as exc:
                errors.append(exc)

        def _reader():
            try:
                for _ in range(10):
                    cm.get_incomplete()
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_completer)
        t2 = threading.Thread(target=_reader)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors


# ===========================================================================
# CHECKPOINT — scan_for_recovery edge cases
# ===========================================================================


class TestScanForRecoveryEdgeCases:
    def test_stale_checkpoint_abandoned_then_not_in_results(self, tmp_db):
        cm = CheckpointManager(db_path=tmp_db)
        cid = cm.create("s", "t", "p")
        _raw_exec(
            tmp_db,
            "UPDATE checkpoints SET created_at=? WHERE id=?",
            (time.time() - 2 * _RESTART_THRESHOLD_SECS, cid),
        )
        results = scan_for_recovery(db_path=tmp_db)
        ids_in_results = [r.checkpoint_id for r in results]
        assert cid not in ids_in_results

    def test_fresh_and_stale_mixed_only_fresh_returned(self, tmp_db):
        cm = CheckpointManager(db_path=tmp_db)
        fresh_id = cm.create("s-fresh", "t-fresh", "fresh prompt")
        stale_id = cm.create("s-stale", "t-stale", "stale prompt")
        _raw_exec(
            tmp_db,
            "UPDATE checkpoints SET created_at=? WHERE id=?",
            (time.time() - 2 * _RESTART_THRESHOLD_SECS, stale_id),
        )
        results = scan_for_recovery(db_path=tmp_db)
        ids = [r.checkpoint_id for r in results]
        assert fresh_id in ids
        assert stale_id not in ids

    def test_recovery_result_action_values_are_valid(self, tmp_db):
        cm = CheckpointManager(db_path=tmp_db)
        # Recent checkpoint
        cm.create("s", "t", "recent prompt")
        # Middle-aged checkpoint (between 1h and 24h)
        cid_mid = cm.create("s2", "t2", "mid prompt")
        _raw_exec(
            tmp_db,
            "UPDATE checkpoints SET created_at=? WHERE id=?",
            (time.time() - 7200, cid_mid),
        )
        results = scan_for_recovery(db_path=tmp_db)
        valid_actions = {"resume", "restart", "abandon"}
        for r in results:
            assert r.action in valid_actions

    def test_scan_does_not_raise_on_inaccessible_db_path(self, tmp_path):
        """A path inside a non-existent nested directory must return []."""
        deep = str(tmp_path / "does" / "not" / "exist" / "cp.db")
        results = scan_for_recovery(db_path=deep)
        assert isinstance(results, list)

    def test_scan_emits_no_events_when_no_checkpoints(self, tmp_db):
        event_bus.clear()
        scan_for_recovery(db_path=tmp_db)
        events = event_bus.get_events(event_type="agent.checkpoint.recovery_scan")
        assert len(events) == 0

    def test_scan_loop_messages_and_iteration_populated_in_result(self, tmp_db):
        cm = CheckpointManager(db_path=tmp_db)
        cid = cm.create("s", "t", "p")
        msgs = [{"role": "assistant", "content": "progress update"}]
        cm.update(cid, msgs, ["shell_exec"], iteration=4)
        results = scan_for_recovery(db_path=tmp_db)
        assert len(results) == 1
        assert results[0].loop_messages == msgs
        assert results[0].iteration == 4
