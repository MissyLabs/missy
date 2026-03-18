"""Coverage gap tests for missy.agent.hatching.

Targets uncovered lines: 181-182, 204-205, 375, 379, 386, 393, 402, 417-418,
458, 470-471, 477, 486-491, 515-517, 562, 566-567, 571, 575, 579, 596-597,
617, 657-663, 695-700, 717.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
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
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(tmp_path: Path) -> HatchingManager:
    return HatchingManager(
        state_path=tmp_path / "hatching.yaml",
        log_path=tmp_path / "log.jsonl",
    )


def _patch_module_paths(monkeypatch, tmp_path: Path) -> None:
    import missy.agent.hatching as hatching_mod

    monkeypatch.setattr(hatching_mod, "_MISSY_DIR", tmp_path)
    monkeypatch.setattr(hatching_mod, "_CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(hatching_mod, "_IDENTITY_PATH", tmp_path / "identity.pem")
    monkeypatch.setattr(hatching_mod, "_SECRETS_DIR", tmp_path / "secrets")
    monkeypatch.setattr(hatching_mod, "_PERSONA_PATH", tmp_path / "persona.yaml")
    monkeypatch.setattr(hatching_mod, "_MEMORY_DB_PATH", tmp_path / "memory.db")


# ---------------------------------------------------------------------------
# HatchingLog.log() — lines 181-182 (OSError swallowed)
# ---------------------------------------------------------------------------


class TestHatchingLogOSErrorSwallowed:
    def test_log_swallows_oserror_on_write(self, tmp_path):
        """An OSError during log write must be silently swallowed (lines 181-182)."""
        log = HatchingLog(log_path=tmp_path / "hatching_log.jsonl")

        # Patch os.open to raise an OSError so the write path explodes.
        with patch("missy.agent.hatching.os.open", side_effect=OSError("disk full")):
            # Must not raise; the exception is logged and discarded.
            log.log("step", "ok", "message")

        # No entries should be present because the write failed.
        assert log.get_entries() == []

    def test_log_swallows_oserror_on_mkdir(self, tmp_path):
        """An OSError from mkdir during log write is also swallowed."""
        log = HatchingLog(log_path=tmp_path / "sub" / "hatching_log.jsonl")

        with patch("missy.agent.hatching.Path.mkdir", side_effect=OSError("read-only fs")):
            log.log("step", "ok", "message")  # must not raise


# ---------------------------------------------------------------------------
# HatchingLog.get_entries() — lines 204-205 (OSError returns empty list)
# ---------------------------------------------------------------------------


class TestHatchingLogGetEntriesOSError:
    def test_get_entries_returns_empty_list_on_oserror(self, tmp_path):
        """An OSError while reading the log file must return [] (lines 204-205)."""
        log_path = tmp_path / "hatching_log.jsonl"
        # Create the file so the exists() check passes.
        log_path.write_text('{"step":"s","status":"ok","message":"m","details":{},"timestamp":"t"}\n')

        with patch("missy.agent.hatching.Path.open", side_effect=OSError("permission denied")):
            result = log_path  # confirm path exists before patching
            log = HatchingLog(log_path=log_path)
            entries = log.get_entries()

        assert entries == []


# ---------------------------------------------------------------------------
# HatchingManager.run_hatching() interactive print paths
# Lines 375, 379, 386, 393, 402
# ---------------------------------------------------------------------------


class TestRunHatchingInteractivePrintPaths:
    def test_run_hatching_interactive_prints_skip_for_completed_step(
        self, tmp_path, monkeypatch, capsys
    ):
        """Already-completed steps print '[skip] ...' when interactive=True (line 375)."""
        _patch_module_paths(monkeypatch, tmp_path)

        state_path = tmp_path / "hatching.yaml"
        # Pre-mark all steps except finalize as done.
        partial_state = HatchingState(
            status=HatchingStatus.IN_PROGRESS,
            started_at="2026-01-01T00:00:00+00:00",
            steps_completed=[
                "validate_environment",
                "initialize_config",
                "verify_providers",
                "initialize_security",
                "generate_persona",
                "seed_memory",
            ],
            environment_validated=True,
            provider_verified=True,
            security_initialized=True,
            persona_generated=True,
            memory_seeded=True,
        )
        state_path.write_text(yaml.safe_dump(partial_state.to_dict()), encoding="utf-8")

        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "log.jsonl")
        state = mgr.run_hatching(interactive=True)

        captured = capsys.readouterr()
        assert "[skip]" in captured.out
        assert "validate_environment" in captured.out

    def test_run_hatching_interactive_prints_ok_for_successful_step(
        self, tmp_path, monkeypatch, capsys
    ):
        """Successful steps print '[ ok ] ...' when interactive=True (line 386)."""
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        mock_turn = MagicMock()
        mock_turn.id = "turn-1"
        mock_store = MagicMock()

        with (
            patch("missy.agent.persona.PersonaManager", return_value=MagicMock()),
            patch(
                "missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store
            ),
            patch(
                "missy.memory.sqlite_store.ConversationTurn",
                MagicMock(new=MagicMock(return_value=mock_turn)),
            ),
        ):
            mgr = _make_manager(tmp_path)
            state = mgr.run_hatching(interactive=True)

        captured = capsys.readouterr()
        assert "[ ok ]" in captured.out
        assert state.status is HatchingStatus.HATCHED

    def test_run_hatching_interactive_prints_warn_for_step_warning(
        self, tmp_path, monkeypatch, capsys
    ):
        """Steps raising _HatchingStepWarning print '[warn] ...' (line 393)."""
        _patch_module_paths(monkeypatch, tmp_path)

        mgr = _make_manager(tmp_path)

        def _warn_step(state, *, interactive):
            raise _HatchingStepWarning("low disk space")

        # Replace all steps with a single warning step plus a finalize stub so
        # the loop completes.
        def _ok_step(state, *, interactive):
            pass

        # Patch the steps list via run_hatching by monkey-patching one step method.
        original_validate = mgr._validate_environment
        mgr._validate_environment = _warn_step  # type: ignore[method-assign]

        # Remaining steps need the env to be valid; skip them.
        partial_state = HatchingState(
            status=HatchingStatus.IN_PROGRESS,
            started_at="2026-01-01T00:00:00+00:00",
            steps_completed=[
                "initialize_config",
                "verify_providers",
                "initialize_security",
                "generate_persona",
                "seed_memory",
                "finalize",
            ],
        )
        (tmp_path / "hatching.yaml").write_text(
            yaml.safe_dump(partial_state.to_dict()), encoding="utf-8"
        )

        state = mgr.run_hatching(interactive=True)

        captured = capsys.readouterr()
        assert "[warn]" in captured.out
        assert "low disk space" in captured.out

    def test_run_hatching_interactive_prints_fail_for_generic_exception(
        self, tmp_path, monkeypatch, capsys
    ):
        """Steps raising a generic Exception print '[FAIL] ...' (line 402)."""
        _patch_module_paths(monkeypatch, tmp_path)

        mgr = _make_manager(tmp_path)

        def _fail_step(state, *, interactive):
            raise RuntimeError("catastrophic failure")

        mgr._validate_environment = _fail_step  # type: ignore[method-assign]

        state = mgr.run_hatching(interactive=True)

        captured = capsys.readouterr()
        assert "[FAIL]" in captured.out
        assert "catastrophic failure" in captured.out
        assert state.status is HatchingStatus.FAILED

    def test_run_hatching_interactive_prints_step_name_before_executing(
        self, tmp_path, monkeypatch, capsys
    ):
        """The '[....] step_name' line is printed before the step runs (line 379)."""
        _patch_module_paths(monkeypatch, tmp_path)

        printed_lines: list[str] = []
        original_print = print

        def _capturing_print(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            printed_lines.append(text)
            original_print(*args, **kwargs)

        # Only run validate_environment; pre-complete the others.
        partial_state = HatchingState(
            status=HatchingStatus.IN_PROGRESS,
            started_at="2026-01-01T00:00:00+00:00",
            steps_completed=[
                "initialize_config",
                "verify_providers",
                "initialize_security",
                "generate_persona",
                "seed_memory",
                "finalize",
            ],
        )
        (tmp_path / "hatching.yaml").write_text(
            yaml.safe_dump(partial_state.to_dict()), encoding="utf-8"
        )

        mgr = _make_manager(tmp_path)

        with patch("builtins.print", side_effect=_capturing_print):
            mgr.run_hatching(interactive=True)

        # At least one line should contain the pending indicator.
        assert any("[....]" in line for line in printed_lines)


# ---------------------------------------------------------------------------
# HatchingManager.reset() — lines 417-418 (OSError swallowed)
# ---------------------------------------------------------------------------


class TestHatchingManagerResetOSError:
    def test_reset_swallows_oserror_when_unlink_fails(self, tmp_path):
        """An OSError from unlink() during reset is silently swallowed (lines 417-418)."""
        state_path = tmp_path / "hatching.yaml"
        state_path.write_text(yaml.safe_dump(HatchingState().to_dict()), encoding="utf-8")

        mgr = HatchingManager(state_path=state_path, log_path=tmp_path / "log.jsonl")

        with patch.object(Path, "unlink", side_effect=OSError("permission denied")):
            mgr.reset()  # must not raise


# ---------------------------------------------------------------------------
# _validate_environment() — lines 458, 470-471, 477, 486-491
# ---------------------------------------------------------------------------


class TestValidateEnvironment:
    def test_raises_runtime_error_for_old_python(self, tmp_path, monkeypatch):
        """Python < 3.11 must raise RuntimeError (line 458)."""
        _patch_module_paths(monkeypatch, tmp_path)

        old_version = sys.version_info
        fake_version = SimpleNamespace(major=3, minor=10, micro=5)

        with patch("missy.agent.hatching.sys") as mock_sys:
            mock_sys.version_info = fake_version
            mgr = _make_manager(tmp_path)
            state = HatchingState()

            with pytest.raises(RuntimeError, match="Python 3.11\\+ is required"):
                mgr._validate_environment(state, interactive=False)

    def test_raises_runtime_error_when_missy_dir_cannot_be_created(
        self, tmp_path, monkeypatch
    ):
        """OSError from mkdir must be re-raised as RuntimeError (lines 470-471)."""
        _patch_module_paths(monkeypatch, tmp_path)

        with patch("missy.agent.hatching.Path.mkdir", side_effect=OSError("read-only")):
            mgr = _make_manager(tmp_path)
            state = HatchingState()

            with pytest.raises(RuntimeError, match="Cannot create Missy data directory"):
                mgr._validate_environment(state, interactive=False)

    def test_raises_runtime_error_when_missy_dir_not_writable(
        self, tmp_path, monkeypatch
    ):
        """A non-writable _MISSY_DIR must raise RuntimeError (line 477)."""
        _patch_module_paths(monkeypatch, tmp_path)

        # mkdir succeeds (tmp_path already exists); os.access returns False.
        with (
            patch("missy.agent.hatching.Path.mkdir"),
            patch("missy.agent.hatching.os.access", return_value=False),
        ):
            mgr = _make_manager(tmp_path)
            state = HatchingState()

            with pytest.raises(RuntimeError, match="is not writable"):
                mgr._validate_environment(state, interactive=False)

    def test_raises_step_warning_on_low_disk_space(self, tmp_path, monkeypatch):
        """statvfs returning tiny free bytes raises _HatchingStepWarning (lines 486-489)."""
        _patch_module_paths(monkeypatch, tmp_path)

        # Construct a fake statvfs result: 1 block of 4096 bytes free = 4 KiB.
        fake_stat = SimpleNamespace(f_bavail=1, f_frsize=4096)

        with (
            patch("missy.agent.hatching.Path.mkdir"),
            patch("missy.agent.hatching.os.access", return_value=True),
            patch("missy.agent.hatching.os.statvfs", return_value=fake_stat),
        ):
            mgr = _make_manager(tmp_path)
            state = HatchingState()

            with pytest.raises(_HatchingStepWarning, match="Low disk space"):
                mgr._validate_environment(state, interactive=False)

    def test_skips_gracefully_when_statvfs_raises_oserror(self, tmp_path, monkeypatch):
        """An OSError from statvfs must be silently ignored (line 490-491)."""
        _patch_module_paths(monkeypatch, tmp_path)

        with (
            patch("missy.agent.hatching.Path.mkdir"),
            patch("missy.agent.hatching.os.access", return_value=True),
            patch("missy.agent.hatching.os.statvfs", side_effect=OSError("not supported")),
        ):
            mgr = _make_manager(tmp_path)
            state = HatchingState()
            # Should not raise; statvfs failure is silently skipped.
            mgr._validate_environment(state, interactive=False)

        assert state.environment_validated is True


# ---------------------------------------------------------------------------
# _initialize_config() — lines 515-517
# ---------------------------------------------------------------------------


class TestInitializeConfig:
    def test_skips_when_config_already_exists(self, tmp_path, monkeypatch):
        """When config.yaml already exists the step returns without rewriting it (line 515 region)."""
        _patch_module_paths(monkeypatch, tmp_path)
        config_path = tmp_path / "config.yaml"
        config_path.write_text("existing: true\n", encoding="utf-8")

        mgr = _make_manager(tmp_path)
        state = HatchingState()
        mgr._initialize_config(state, interactive=False)

        # File must be unchanged.
        assert config_path.read_text() == "existing: true\n"

    def test_prints_message_when_interactive_and_config_written(
        self, tmp_path, monkeypatch, capsys
    ):
        """interactive=True prints the created path after writing config (lines 514-515)."""
        _patch_module_paths(monkeypatch, tmp_path)
        # Ensure config does not exist so the write branch executes.
        config_path = tmp_path / "config.yaml"
        assert not config_path.exists()

        mgr = _make_manager(tmp_path)
        state = HatchingState()
        mgr._initialize_config(state, interactive=True)

        captured = capsys.readouterr()
        assert "config" in captured.out.lower() or str(config_path) in captured.out
        assert config_path.exists()

    def test_raises_runtime_error_on_oserror_writing_config(
        self, tmp_path, monkeypatch
    ):
        """OSError while writing config must raise RuntimeError (lines 516-517)."""
        _patch_module_paths(monkeypatch, tmp_path)

        with (
            patch("missy.agent.hatching.Path.mkdir"),
            patch("missy.agent.hatching.Path.write_text", side_effect=OSError("disk full")),
            patch("missy.agent.hatching.Path.exists", return_value=False),
        ):
            mgr = _make_manager(tmp_path)
            state = HatchingState()

            with pytest.raises(RuntimeError, match="Cannot write default config"):
                mgr._initialize_config(state, interactive=False)


# ---------------------------------------------------------------------------
# _verify_providers() — line 562
# ---------------------------------------------------------------------------


class TestVerifyProviders:
    def test_raises_step_warning_when_no_provider_key_found(
        self, tmp_path, monkeypatch
    ):
        """No env var and no config key raises _HatchingStepWarning (line 562)."""
        _patch_module_paths(monkeypatch, tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Config does not exist — _check_config_for_provider_key returns None.
        mgr = _make_manager(tmp_path)
        state = HatchingState()

        with pytest.raises(_HatchingStepWarning, match="No provider API key found"):
            mgr._verify_providers(state, interactive=False)

        assert state.provider_verified is False


# ---------------------------------------------------------------------------
# _check_config_for_provider_key() — lines 566-567, 571, 575, 579
# ---------------------------------------------------------------------------


class TestCheckConfigForProviderKey:
    def test_returns_none_when_config_does_not_exist(self, tmp_path, monkeypatch):
        """Missing config.yaml returns None (line 566-567 branch via existence check)."""
        _patch_module_paths(monkeypatch, tmp_path)
        # config.yaml was not created, so _CONFIG_PATH.exists() is False.
        mgr = _make_manager(tmp_path)
        result = mgr._check_config_for_provider_key()
        assert result is None

    def test_returns_none_on_oserror_reading_config(self, tmp_path, monkeypatch):
        """OSError while reading config returns None (lines 566-567)."""
        _patch_module_paths(monkeypatch, tmp_path)
        config_path = tmp_path / "config.yaml"
        config_path.write_text("providers:\n  anthropic:\n    api_key: sk-test\n")

        with patch("missy.agent.hatching.Path.open", side_effect=OSError("permission")):
            mgr = _make_manager(tmp_path)
            result = mgr._check_config_for_provider_key()

        assert result is None

    def test_returns_none_when_providers_is_not_a_dict(self, tmp_path, monkeypatch):
        """providers not being a dict must return None (line 571)."""
        _patch_module_paths(monkeypatch, tmp_path)
        config_path = tmp_path / "config.yaml"
        config_path.write_text("providers:\n  - list_item\n", encoding="utf-8")

        mgr = _make_manager(tmp_path)
        result = mgr._check_config_for_provider_key()
        assert result is None

    def test_returns_none_when_provider_cfg_is_not_a_dict(
        self, tmp_path, monkeypatch
    ):
        """A provider_cfg that is not a dict is skipped (line 575)."""
        _patch_module_paths(monkeypatch, tmp_path)
        config_path = tmp_path / "config.yaml"
        # provider value is a plain string, not a mapping.
        config_path.write_text(
            "providers:\n  anthropic: just_a_string\n", encoding="utf-8"
        )

        mgr = _make_manager(tmp_path)
        result = mgr._check_config_for_provider_key()
        assert result is None

    def test_returns_provider_name_when_api_key_is_present(
        self, tmp_path, monkeypatch
    ):
        """A non-empty api_key in config must return the provider name (line 579)."""
        _patch_module_paths(monkeypatch, tmp_path)
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "providers:\n  anthropic:\n    api_key: sk-real-key\n",
            encoding="utf-8",
        )

        mgr = _make_manager(tmp_path)
        result = mgr._check_config_for_provider_key()
        assert result == "anthropic"

    def test_returns_provider_name_when_api_keys_list_is_present(
        self, tmp_path, monkeypatch
    ):
        """A non-empty api_keys list is also accepted (line 578-579)."""
        _patch_module_paths(monkeypatch, tmp_path)
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "providers:\n  openai:\n    api_keys:\n      - sk-key-one\n",
            encoding="utf-8",
        )

        mgr = _make_manager(tmp_path)
        result = mgr._check_config_for_provider_key()
        assert result == "openai"


# ---------------------------------------------------------------------------
# _initialize_security() — lines 596-597, 617
# ---------------------------------------------------------------------------


class TestInitializeSecurity:
    def test_raises_runtime_error_when_secrets_dir_cannot_be_created(
        self, tmp_path, monkeypatch
    ):
        """OSError from mkdir raises RuntimeError (lines 596-597)."""
        _patch_module_paths(monkeypatch, tmp_path)

        with patch("missy.agent.hatching.Path.mkdir", side_effect=OSError("read-only")):
            mgr = _make_manager(tmp_path)
            state = HatchingState()

            with pytest.raises(RuntimeError, match="Cannot create secrets directory"):
                mgr._initialize_security(state, interactive=False)

    def test_prints_note_when_identity_absent_and_interactive(
        self, tmp_path, monkeypatch, capsys
    ):
        """When identity key is absent and interactive=True, a note is printed (line 617)."""
        _patch_module_paths(monkeypatch, tmp_path)

        # Ensure secrets dir can be created (use real tmp_path).
        secrets_dir = tmp_path / "secrets"
        # Do NOT create identity.pem so the absent-key branch is taken.

        mgr = _make_manager(tmp_path)
        state = HatchingState()
        mgr._initialize_security(state, interactive=True)

        captured = capsys.readouterr()
        assert "identity" in captured.out.lower() or "key" in captured.out.lower()
        assert state.security_initialized is True

    def test_does_not_print_note_when_identity_present(
        self, tmp_path, monkeypatch, capsys
    ):
        """When identity key exists the note is NOT printed."""
        _patch_module_paths(monkeypatch, tmp_path)

        # Create a fake identity key file.
        identity_path = tmp_path / "identity.pem"
        identity_path.write_text("---fake pem---", encoding="utf-8")

        mgr = _make_manager(tmp_path)
        state = HatchingState()
        mgr._initialize_security(state, interactive=True)

        captured = capsys.readouterr()
        # The interactive note about generating the key must NOT appear.
        assert "will be generated" not in captured.out


# ---------------------------------------------------------------------------
# _generate_persona() — lines 657-663
# ---------------------------------------------------------------------------


class TestGeneratePersona:
    def test_raises_step_warning_on_import_error(self, tmp_path, monkeypatch):
        """ImportError for PersonaManager raises _HatchingStepWarning (lines 658-661)."""
        _patch_module_paths(monkeypatch, tmp_path)

        with patch(
            "missy.agent.hatching.HatchingManager._generate_persona",
            wraps=None,
        ):
            # Use a real call but make the internal import fail.
            pass

        # Directly exercise the step by making the import inside it raise.
        mgr = _make_manager(tmp_path)
        state = HatchingState()

        # Simulate ImportError by replacing the module lookup for PersonaManager.
        import missy.agent.persona as persona_mod

        with patch.object(
            persona_mod,
            "PersonaManager",
            side_effect=ImportError("no module"),
        ):
            # We need to cause the `from missy.agent.persona import PersonaManager`
            # line inside _generate_persona to raise.  We do that by temporarily
            # removing the module from sys.modules so the import re-executes.
            original_module = sys.modules.pop("missy.agent.persona", None)
            sys.modules["missy.agent.persona"] = MagicMock(
                **{"PersonaManager": MagicMock(side_effect=ImportError("injected"))}
            )
            # Actually the import uses `from missy.agent.persona import PersonaManager`
            # inside the function body; patching sys.modules is the reliable way.
            try:
                with pytest.raises(_HatchingStepWarning, match="Could not import PersonaManager"):
                    mgr._generate_persona(state, interactive=False)
            finally:
                if original_module is not None:
                    sys.modules["missy.agent.persona"] = original_module
                else:
                    sys.modules.pop("missy.agent.persona", None)

    def test_raises_step_warning_on_oserror_writing_persona(
        self, tmp_path, monkeypatch
    ):
        """OSError while writing persona file raises _HatchingStepWarning (lines 662-665)."""
        _patch_module_paths(monkeypatch, tmp_path)

        mock_pm = MagicMock()
        mock_pm.save.side_effect = OSError("disk full")

        with patch(
            "missy.agent.persona.PersonaManager", return_value=mock_pm
        ):
            mgr = _make_manager(tmp_path)
            state = HatchingState()

            with pytest.raises(_HatchingStepWarning, match="Could not write persona file"):
                mgr._generate_persona(state, interactive=False)

    def test_prints_message_when_persona_written_and_interactive(
        self, tmp_path, monkeypatch, capsys
    ):
        """interactive=True prints a message after persona is written (line 657)."""
        _patch_module_paths(monkeypatch, tmp_path)

        mock_pm = MagicMock()

        with patch("missy.agent.persona.PersonaManager", return_value=mock_pm):
            mgr = _make_manager(tmp_path)
            state = HatchingState()
            mgr._generate_persona(state, interactive=True)

        captured = capsys.readouterr()
        assert "persona" in captured.out.lower() or str(tmp_path) in captured.out


# ---------------------------------------------------------------------------
# _seed_memory() — lines 695-700
# ---------------------------------------------------------------------------


class TestSeedMemory:
    def test_raises_step_warning_on_import_error(self, tmp_path, monkeypatch):
        """ImportError for SQLiteMemoryStore raises _HatchingStepWarning (lines 695-698)."""
        _patch_module_paths(monkeypatch, tmp_path)

        mgr = _make_manager(tmp_path)
        state = HatchingState()

        # Remove the memory module from sys.modules temporarily so the import fails.
        original_module = sys.modules.pop("missy.memory.sqlite_store", None)
        sys.modules["missy.memory.sqlite_store"] = MagicMock(
            **{
                "SQLiteMemoryStore": MagicMock(
                    side_effect=ImportError("no sqlite_store")
                ),
                "ConversationTurn": MagicMock(),
            }
        )
        try:
            with pytest.raises(_HatchingStepWarning, match="Could not import SQLiteMemoryStore"):
                mgr._seed_memory(state, interactive=False)
        finally:
            if original_module is not None:
                sys.modules["missy.memory.sqlite_store"] = original_module
            else:
                sys.modules.pop("missy.memory.sqlite_store", None)

    def test_raises_step_warning_on_generic_exception_from_store(
        self, tmp_path, monkeypatch
    ):
        """A generic exception from store.add_turn raises _HatchingStepWarning (lines 699-702)."""
        _patch_module_paths(monkeypatch, tmp_path)

        mock_turn = MagicMock()
        mock_turn.id = "turn-1"
        mock_store = MagicMock()
        mock_store.add_turn.side_effect = RuntimeError("database locked")

        with (
            patch(
                "missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store
            ),
            patch(
                "missy.memory.sqlite_store.ConversationTurn",
                MagicMock(new=MagicMock(return_value=mock_turn)),
            ),
        ):
            mgr = _make_manager(tmp_path)
            state = HatchingState()

            with pytest.raises(
                _HatchingStepWarning,
                match="Memory seeding failed",
            ):
                mgr._seed_memory(state, interactive=False)


# ---------------------------------------------------------------------------
# _finalize() — line 717
# ---------------------------------------------------------------------------


class TestFinalize:
    def test_prints_message_when_interactive(self, tmp_path, capsys):
        """interactive=True prints the 'hatched and ready' message (line 717)."""
        mgr = _make_manager(tmp_path)
        state = HatchingState(status=HatchingStatus.IN_PROGRESS)

        mgr._finalize(state, interactive=True)

        captured = capsys.readouterr()
        assert "hatched" in captured.out.lower() or "ready" in captured.out.lower()
        assert state.status is HatchingStatus.HATCHED

    def test_does_not_print_when_not_interactive(self, tmp_path, capsys):
        """interactive=False produces no stdout output."""
        mgr = _make_manager(tmp_path)
        state = HatchingState(status=HatchingStatus.IN_PROGRESS)

        mgr._finalize(state, interactive=False)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert state.status is HatchingStatus.HATCHED
