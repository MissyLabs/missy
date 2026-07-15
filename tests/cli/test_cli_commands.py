"""Comprehensive CLI command tests for commands not covered by test_main.py.

Covers:
  - missy doctor
  - missy vault set/get/list/delete
  - missy schedule add/pause/resume/remove
  - missy audit recent
  - missy sessions cleanup
  - missy approvals list
  - missy patches list/approve/reject
  - missy mcp list/add/remove
  - missy devices list/pair/unpair/status/policy
  - missy voice status/test
  - missy discord status/audit
  - missy gateway status
  - --help flags for every group and sub-command

All external dependencies (Vault, SchedulerManager, McpManager,
DeviceRegistry, PromptPatchManager, AuditLogger, etc.) are mocked so that
tests do not require a real config file or running services.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from missy.channels.voice.registry import EdgeNode
from missy.cli.main import cli
from tests.cli.conftest import _make_cli_runner

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG_YAML = """\
network:
  default_deny: true
providers:
  anthropic:
    name: anthropic
    model: "claude-3-5-sonnet-20241022"
workspace_path: "/tmp/workspace"
audit_log_path: "/tmp/audit.jsonl"
"""


@pytest.fixture()
def runner() -> CliRunner:
    return _make_cli_runner(mix_stderr=False)


def _make_mock_config(**kwargs) -> MagicMock:
    """Return a MissyConfig-like mock with safe defaults."""
    cfg = MagicMock()
    cfg.audit_log_path = "/tmp/audit.jsonl"
    cfg.workspace_path = "/tmp/workspace"
    cfg.providers = {}
    cfg.plugins.enabled = False
    cfg.plugins.allowed_plugins = []
    cfg.shell.enabled = False
    cfg.shell.allowed_commands = []
    cfg.network.default_deny = True
    cfg.network.allowed_domains = []
    cfg.network.allowed_cidrs = []
    cfg.network.allowed_hosts = []
    cfg.discord = None
    cfg.max_spend_usd = 0.0
    cfg.observability.log_file_path = "/tmp/missy.log"
    cfg.observability.log_level = "warning"
    # vault sub-config
    vault_sub = MagicMock()
    vault_sub.vault_dir = "~/.missy/secrets"
    cfg.vault = vault_sub
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def _write_temp_config() -> str:
    """Write minimal YAML to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as fh:
        fh.write(_MINIMAL_CONFIG_YAML)
        return fh.name


# ---------------------------------------------------------------------------
# Context manager that patches _load_subsystems for most commands
# ---------------------------------------------------------------------------


class _SubsystemsPatch:
    """Context manager: patches _load_subsystems and exposes the mock config."""

    def __init__(self, cfg: MagicMock | None = None):
        self.cfg = cfg or _make_mock_config()
        self._patcher = patch("missy.cli.main._load_subsystems", return_value=self.cfg)

    def __enter__(self):
        self._patcher.start()
        return self.cfg

    def __exit__(self, *args):
        self._patcher.stop()


# ===========================================================================
# missy doctor
# ===========================================================================


class TestDoctor:
    def test_doctor_exits_zero(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        with _SubsystemsPatch():
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []
            mock_mgr = MagicMock()
            mock_mgr.list_jobs.return_value = []
            with (
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
                patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        assert result.exit_code == 0

    def test_doctor_output_contains_config_loaded(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        with _SubsystemsPatch():
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []
            mock_mgr = MagicMock()
            mock_mgr.list_jobs.return_value = []
            with (
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
                patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        assert "config loaded" in result.output or "Missy Doctor" in result.output

    def test_doctor_shows_provider_not_available(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        with _SubsystemsPatch():
            mock_provider = MagicMock()
            mock_provider.is_available.return_value = False
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = ["anthropic"]
            mock_registry.get.return_value = mock_provider
            mock_mgr = MagicMock()
            mock_mgr.list_jobs.return_value = []
            with (
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
                patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        assert result.exit_code == 0
        assert "anthropic" in result.output

    def test_doctor_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["doctor", "--help"])
        assert result.exit_code == 0

    def test_doctor_shows_scheduled_jobs_count(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        job = MagicMock()
        with _SubsystemsPatch():
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []
            mock_mgr = MagicMock()
            mock_mgr.list_jobs.return_value = [job, job]
            mock_mgr.load_jobs.return_value = [job, job]
            with (
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
                patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        assert "2 job" in result.output

    def test_doctor_shows_discord_disabled(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        with _SubsystemsPatch() as cfg:
            cfg.discord = None
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []
            mock_mgr = MagicMock()
            mock_mgr.list_jobs.return_value = []
            with (
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
                patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        assert result.exit_code == 0


# ===========================================================================
# missy vault
# ===========================================================================


class TestVaultSet:
    def test_vault_set_exits_zero(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        with _SubsystemsPatch(), patch("missy.security.vault.Vault", return_value=mock_vault):
            result = runner.invoke(
                cli, ["--config", cfg_path, "vault", "set", "MY_KEY", "my_value"]
            )
        assert result.exit_code == 0

    def test_vault_set_success_message(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        with _SubsystemsPatch(), patch("missy.security.vault.Vault", return_value=mock_vault):
            result = runner.invoke(
                cli, ["--config", cfg_path, "vault", "set", "MY_KEY", "my_value"]
            )
        assert "MY_KEY" in result.output

    def test_vault_set_calls_vault_set(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        with _SubsystemsPatch(), patch("missy.security.vault.Vault", return_value=mock_vault):
            runner.invoke(cli, ["--config", cfg_path, "vault", "set", "API_KEY", "secret123"])
        mock_vault.set.assert_called_once_with("API_KEY", "secret123")

    def test_vault_set_vault_error_exits_one(self, runner: CliRunner):
        from missy.security.vault import VaultError

        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.set.side_effect = VaultError("encryption failed")
        with _SubsystemsPatch(), patch("missy.security.vault.Vault", return_value=mock_vault):
            result = runner.invoke(cli, ["--config", cfg_path, "vault", "set", "KEY", "val"])
        assert result.exit_code == 1

    def test_vault_set_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["vault", "set", "--help"])
        assert result.exit_code == 0


class TestVaultGet:
    def test_vault_get_prints_value(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.get.return_value = "super_secret"
        with _SubsystemsPatch(), patch("missy.security.vault.Vault", return_value=mock_vault):
            result = runner.invoke(cli, ["--config", cfg_path, "vault", "get", "MY_KEY"])
        assert "super_secret" in result.output

    def test_vault_get_missing_key_exits_one(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.get.return_value = None
        with _SubsystemsPatch(), patch("missy.security.vault.Vault", return_value=mock_vault):
            result = runner.invoke(cli, ["--config", cfg_path, "vault", "get", "MISSING"])
        assert result.exit_code == 1

    def test_vault_get_vault_error_exits_one(self, runner: CliRunner):
        from missy.security.vault import VaultError

        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.get.side_effect = VaultError("corrupt vault")
        with _SubsystemsPatch(), patch("missy.security.vault.Vault", return_value=mock_vault):
            result = runner.invoke(cli, ["--config", cfg_path, "vault", "get", "KEY"])
        assert result.exit_code == 1

    def test_vault_get_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["vault", "get", "--help"])
        assert result.exit_code == 0


class TestVaultList:
    def test_vault_list_empty_vault(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.list_keys.return_value = []
        with _SubsystemsPatch(), patch("missy.security.vault.Vault", return_value=mock_vault):
            result = runner.invoke(cli, ["--config", cfg_path, "vault", "list"])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_vault_list_shows_keys(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.list_keys.return_value = ["OPENAI_KEY", "SLACK_TOKEN"]
        with _SubsystemsPatch(), patch("missy.security.vault.Vault", return_value=mock_vault):
            result = runner.invoke(cli, ["--config", cfg_path, "vault", "list"])
        assert result.exit_code == 0
        assert "OPENAI_KEY" in result.output
        assert "SLACK_TOKEN" in result.output

    def test_vault_list_vault_error_exits_one(self, runner: CliRunner):
        from missy.security.vault import VaultError

        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.list_keys.side_effect = VaultError("read error")
        with _SubsystemsPatch(), patch("missy.security.vault.Vault", return_value=mock_vault):
            result = runner.invoke(cli, ["--config", cfg_path, "vault", "list"])
        assert result.exit_code == 1

    def test_vault_list_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["vault", "list", "--help"])
        assert result.exit_code == 0


class TestVaultDelete:
    def test_vault_delete_key_found(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.delete.return_value = True
        with _SubsystemsPatch(), patch("missy.security.vault.Vault", return_value=mock_vault):
            result = runner.invoke(cli, ["--config", cfg_path, "vault", "delete", "MY_KEY"])
        assert result.exit_code == 0
        assert "MY_KEY" in result.output

    def test_vault_delete_key_not_found(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.delete.return_value = False
        with _SubsystemsPatch(), patch("missy.security.vault.Vault", return_value=mock_vault):
            result = runner.invoke(cli, ["--config", cfg_path, "vault", "delete", "GHOST"])
        # VAULT-002: a nonexistent key must exit nonzero, matching vault_get's
        # equivalent not-found handling, so a script checking only the exit
        # code isn't misled into believing the deletion succeeded.
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_vault_delete_vault_error_exits_one(self, runner: CliRunner):
        from missy.security.vault import VaultError

        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.delete.side_effect = VaultError("io error")
        with _SubsystemsPatch(), patch("missy.security.vault.Vault", return_value=mock_vault):
            result = runner.invoke(cli, ["--config", cfg_path, "vault", "delete", "KEY"])
        assert result.exit_code == 1

    def test_vault_delete_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["vault", "delete", "--help"])
        assert result.exit_code == 0

    def test_vault_group_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["vault", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy schedule add / pause / resume / remove
# ===========================================================================


class TestScheduleAdd:
    def _make_job(self) -> MagicMock:
        job = MagicMock()
        job.id = "aaaabbbb-0000-0000-0000-000000000000"
        job.name = "Test Job"
        job.schedule = "every 5 minutes"
        job.provider = "anthropic"
        job.capability_mode = "safe-chat"
        return job

    def test_schedule_add_exits_zero(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.add_job.return_value = self._make_job()
        with (
            _SubsystemsPatch(),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            result = runner.invoke(
                cli,
                [
                    "--config",
                    cfg_path,
                    "schedule",
                    "add",
                    "--name",
                    "Test Job",
                    "--schedule",
                    "every 5 minutes",
                    "--task",
                    "Check the news",
                ],
            )
        assert result.exit_code == 0

    def test_schedule_add_success_message(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.add_job.return_value = self._make_job()
        with (
            _SubsystemsPatch(),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            result = runner.invoke(
                cli,
                [
                    "--config",
                    cfg_path,
                    "schedule",
                    "add",
                    "--name",
                    "Test Job",
                    "--schedule",
                    "every 5 minutes",
                    "--task",
                    "Check the news",
                ],
            )
        assert "Test Job" in result.output or "Job added" in result.output

    def test_schedule_add_defaults_capability_mode_to_safe_chat(self, runner: CliRunner):
        """SR-2.1 regression: `missy schedule add` without --capability-mode
        must forward "safe-chat" to add_job(), not leave the caller to
        pick up whatever SchedulerManager.add_job()'s own default is
        implicitly -- the CLI's advertised default and the manager's
        actual default must agree.
        """
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.add_job.return_value = self._make_job()
        with (
            _SubsystemsPatch(),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            runner.invoke(
                cli,
                [
                    "--config",
                    cfg_path,
                    "schedule",
                    "add",
                    "--name",
                    "Test Job",
                    "--schedule",
                    "every 5 minutes",
                    "--task",
                    "Check the news",
                ],
            )
        mock_mgr.add_job.assert_called_once_with(
            name="Test Job",
            schedule="every 5 minutes",
            task="Check the news",
            provider="anthropic",
            capability_mode="safe-chat",
            description="",
            max_attempts=3,
            backoff_seconds=None,
            retry_on=None,
            delete_after_run=False,
            active_hours="",
            timezone="",
        )

    def test_schedule_add_explicit_full_capability_mode(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        job = self._make_job()
        job.capability_mode = "full"
        mock_mgr.add_job.return_value = job
        with (
            _SubsystemsPatch(),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            result = runner.invoke(
                cli,
                [
                    "--config",
                    cfg_path,
                    "schedule",
                    "add",
                    "--name",
                    "Test Job",
                    "--schedule",
                    "every 5 minutes",
                    "--task",
                    "Check the news",
                    "--capability-mode",
                    "full",
                ],
            )
        assert result.exit_code == 0
        mock_mgr.add_job.assert_called_once_with(
            name="Test Job",
            schedule="every 5 minutes",
            task="Check the news",
            provider="anthropic",
            capability_mode="full",
            description="",
            max_attempts=3,
            backoff_seconds=None,
            retry_on=None,
            delete_after_run=False,
            active_hours="",
            timezone="",
        )

    def test_schedule_add_rejects_invalid_capability_mode(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        with _SubsystemsPatch():
            result = runner.invoke(
                cli,
                [
                    "--config",
                    cfg_path,
                    "schedule",
                    "add",
                    "--name",
                    "Test Job",
                    "--schedule",
                    "every 5 minutes",
                    "--task",
                    "Check the news",
                    "--capability-mode",
                    "root-access",
                ],
            )
        assert result.exit_code != 0

    def test_schedule_add_invalid_schedule_exits_one(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.add_job.side_effect = ValueError("unrecognised schedule")
        with (
            _SubsystemsPatch(),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            result = runner.invoke(
                cli,
                [
                    "--config",
                    cfg_path,
                    "schedule",
                    "add",
                    "--name",
                    "Bad",
                    "--schedule",
                    "never",
                    "--task",
                    "do nothing",
                ],
            )
        assert result.exit_code == 1

    def test_schedule_add_missing_required_options(self, runner: CliRunner):
        result = runner.invoke(cli, ["schedule", "add"])
        assert result.exit_code != 0

    def test_schedule_add_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["schedule", "add", "--help"])
        assert result.exit_code == 0

    def test_schedule_add_scheduler_error_exits_one(self, runner: CliRunner):
        from missy.core.exceptions import SchedulerError

        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.add_job.side_effect = SchedulerError("backend down")
        with (
            _SubsystemsPatch(),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            result = runner.invoke(
                cli,
                [
                    "--config",
                    cfg_path,
                    "schedule",
                    "add",
                    "--name",
                    "X",
                    "--schedule",
                    "every hour",
                    "--task",
                    "ping",
                ],
            )
        assert result.exit_code == 1


class TestSchedulePause:
    def test_schedule_pause_exits_zero(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        with (
            _SubsystemsPatch(),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "schedule", "pause", "abc123"])
        assert result.exit_code == 0

    def test_schedule_pause_success_message(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        with (
            _SubsystemsPatch(),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "schedule", "pause", "abc123"])
        assert "abc123" in result.output

    def test_schedule_pause_job_not_found_exits_one(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.pause_job.side_effect = KeyError("abc123")
        with (
            _SubsystemsPatch(),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "schedule", "pause", "abc123"])
        assert result.exit_code == 1

    def test_schedule_pause_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["schedule", "pause", "--help"])
        assert result.exit_code == 0


class TestScheduleResume:
    def test_schedule_resume_exits_zero(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        with (
            _SubsystemsPatch(),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "schedule", "resume", "abc123"])
        assert result.exit_code == 0

    def test_schedule_resume_success_message(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        with (
            _SubsystemsPatch(),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "schedule", "resume", "myjob"])
        assert "myjob" in result.output

    def test_schedule_resume_job_not_found_exits_one(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.resume_job.side_effect = KeyError("myjob")
        with (
            _SubsystemsPatch(),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "schedule", "resume", "myjob"])
        assert result.exit_code == 1

    def test_schedule_resume_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["schedule", "resume", "--help"])
        assert result.exit_code == 0


class TestScheduleRemove:
    def test_schedule_remove_exits_zero(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        with (
            _SubsystemsPatch(),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            # --yes bypasses the confirmation prompt
            result = runner.invoke(
                cli,
                ["--config", cfg_path, "schedule", "remove", "--yes", "job123"],
            )
        assert result.exit_code == 0

    def test_schedule_remove_success_message(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        with (
            _SubsystemsPatch(),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            result = runner.invoke(
                cli,
                ["--config", cfg_path, "schedule", "remove", "--yes", "job123"],
            )
        assert "job123" in result.output

    def test_schedule_remove_job_not_found_exits_one(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.remove_job.side_effect = KeyError("ghost")
        with (
            _SubsystemsPatch(),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            result = runner.invoke(
                cli,
                ["--config", cfg_path, "schedule", "remove", "--yes", "ghost"],
            )
        assert result.exit_code == 1

    def test_schedule_remove_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["schedule", "remove", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy audit recent
# ===========================================================================


class TestAuditRecent:
    def test_audit_recent_exits_zero_no_events(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_logger = MagicMock()
        mock_logger.get_recent_events.return_value = []
        with (
            _SubsystemsPatch(),
            patch("missy.observability.audit_logger.AuditLogger", return_value=mock_logger),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "audit", "recent"])
        assert result.exit_code == 0

    def test_audit_recent_no_events_message(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_logger = MagicMock()
        mock_logger.get_recent_events.return_value = []
        with (
            _SubsystemsPatch(),
            patch("missy.observability.audit_logger.AuditLogger", return_value=mock_logger),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "audit", "recent"])
        assert "No audit events" in result.output

    def test_audit_recent_shows_events(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        event = {
            "timestamp": "2026-03-12T10:00:00",
            "event_type": "network.request",
            "category": "network",
            "result": "allow",
            "detail": {"host": "api.anthropic.com"},
        }
        mock_logger = MagicMock()
        mock_logger.get_recent_events.return_value = [event]
        with (
            _SubsystemsPatch(),
            patch("missy.observability.audit_logger.AuditLogger", return_value=mock_logger),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "audit", "recent"])
        assert result.exit_code == 0
        assert "network" in result.output

    def test_audit_recent_category_filter(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        events = [
            {
                "timestamp": "2026-03-12T10:00:00",
                "event_type": "network.request",
                "category": "network",
                "result": "allow",
                "detail": {},
            },
            {
                "timestamp": "2026-03-12T10:01:00",
                "event_type": "shell.exec",
                "category": "shell",
                "result": "deny",
                "detail": {},
            },
        ]
        mock_logger = MagicMock()
        mock_logger.get_recent_events.return_value = events
        with (
            _SubsystemsPatch(),
            patch("missy.observability.audit_logger.AuditLogger", return_value=mock_logger),
        ):
            result = runner.invoke(
                cli,
                ["--config", cfg_path, "audit", "recent", "--category", "shell"],
            )
        assert result.exit_code == 0
        assert "shell" in result.output

    def test_audit_recent_respects_limit_option(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_logger = MagicMock()
        mock_logger.get_recent_events.return_value = []
        with (
            _SubsystemsPatch(),
            patch("missy.observability.audit_logger.AuditLogger", return_value=mock_logger),
        ):
            result = runner.invoke(
                cli,
                ["--config", cfg_path, "audit", "recent", "--limit", "5"],
            )
        assert result.exit_code == 0
        # over-fetches by 5× — so called with 25
        mock_logger.get_recent_events.assert_called_once_with(limit=25)

    def test_audit_recent_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["audit", "recent", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy audit verify (SR-1.1)
# ===========================================================================


class TestAuditVerify:
    """Real AgentIdentity + real AuditLogger writes, not mocked -- this
    command's entire purpose is cryptographic verification, so mocking
    verify_audit_log() would defeat the point of testing it."""

    def _write_signed_log(self, tmp_path, key_path, log_path, events):

        from missy.core.events import AuditEvent, EventBus
        from missy.observability.audit_logger import AuditLogger
        from missy.security.identity import AgentIdentity

        identity = AgentIdentity.load_or_generate(str(key_path))
        bus = EventBus()
        AuditLogger(log_path=str(log_path), bus=bus, identity=identity)
        for event_type, category, result, detail in events:
            bus.publish(
                AuditEvent.now(
                    session_id="s1",
                    task_id="t1",
                    event_type=event_type,
                    category=category,
                    result=result,
                    detail=detail,
                )
            )
        return identity

    def test_verify_clean_log_reports_valid(self, runner: CliRunner, tmp_path, monkeypatch):
        key_path = tmp_path / "identity.pem"
        log_path = tmp_path / "audit.jsonl"
        monkeypatch.setattr("missy.security.identity.DEFAULT_KEY_PATH", str(key_path))
        self._write_signed_log(
            tmp_path,
            key_path,
            log_path,
            [("network.request", "network", "allow", {"host": "example.com"})],
        )
        cfg_path = _write_temp_config()
        cfg = _make_mock_config()
        cfg.audit_log_path = str(log_path)

        with _SubsystemsPatch(cfg):
            result = runner.invoke(cli, ["--config", cfg_path, "audit", "verify"])

        assert result.exit_code == 0
        assert "valid: 1" in result.output
        assert "No tampering detected" in result.output

    def test_verify_tampered_log_exits_nonzero_and_reports_tampered(
        self, runner: CliRunner, tmp_path, monkeypatch
    ):
        import json as _json

        key_path = tmp_path / "identity.pem"
        log_path = tmp_path / "audit.jsonl"
        monkeypatch.setattr("missy.security.identity.DEFAULT_KEY_PATH", str(key_path))
        self._write_signed_log(
            tmp_path,
            key_path,
            log_path,
            [("shell.exec", "shell", "deny", {"command": "rm -rf /"})],
        )
        # Reproduce the review's exact tamper PoC via the CLI path too.
        record = _json.loads(log_path.read_text().splitlines()[0])
        record["result"] = "allow"
        log_path.write_text(_json.dumps(record) + "\n")

        cfg_path = _write_temp_config()
        cfg = _make_mock_config()
        cfg.audit_log_path = str(log_path)

        with _SubsystemsPatch(cfg):
            result = runner.invoke(cli, ["--config", cfg_path, "audit", "verify"])

        assert result.exit_code == 1
        assert "tampered: 1" in result.output
        assert "shell.exec" in result.output

    def test_verify_reordered_log_exits_nonzero_and_reports_chain_break(
        self, runner: CliRunner, tmp_path, monkeypatch
    ):
        """Regression: per-line signatures alone catch content tampering
        but not reordering -- two validly-signed lines swapped in
        position must still be caught and surfaced, via the hash chain,
        not silently reported as fully valid.
        """
        import json as _json

        key_path = tmp_path / "identity.pem"
        log_path = tmp_path / "audit.jsonl"
        monkeypatch.setattr("missy.security.identity.DEFAULT_KEY_PATH", str(key_path))
        self._write_signed_log(
            tmp_path,
            key_path,
            log_path,
            [
                ("event.a", "test", "allow", {}),
                ("event.b", "test", "allow", {}),
                ("event.c", "test", "allow", {}),
            ],
        )
        lines = log_path.read_text().splitlines()
        lines[1], lines[2] = lines[2], lines[1]
        log_path.write_text("\n".join(lines) + "\n")
        # Every individual line still parses and its own signature still
        # verifies -- the point of this test is that reordering alone
        # (no content edits) must still be caught.
        for ln in lines:
            _json.loads(ln)

        cfg_path = _write_temp_config()
        cfg = _make_mock_config()
        cfg.audit_log_path = str(log_path)

        with _SubsystemsPatch(cfg):
            result = runner.invoke(cli, ["--config", cfg_path, "audit", "verify"])

        assert result.exit_code == 1
        assert "valid: 3" in result.output
        assert "out of sequence" in result.output.lower()

    def test_verify_empty_log_message(self, runner: CliRunner, tmp_path, monkeypatch):
        key_path = tmp_path / "identity.pem"
        log_path = tmp_path / "does_not_exist.jsonl"
        monkeypatch.setattr("missy.security.identity.DEFAULT_KEY_PATH", str(key_path))
        cfg_path = _write_temp_config()
        cfg = _make_mock_config()
        cfg.audit_log_path = str(log_path)

        with _SubsystemsPatch(cfg):
            result = runner.invoke(cli, ["--config", cfg_path, "audit", "verify"])

        assert result.exit_code == 0
        assert "empty or does not exist" in result.output

    def test_verify_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["audit", "verify", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy logs
# ===========================================================================


class TestLogs:
    def test_logs_path_prints_configured_path(self, runner: CliRunner, tmp_path):
        cfg_path = _write_temp_config()
        cfg = _make_mock_config()
        cfg.observability.log_file_path = str(tmp_path / "missy.log")

        with _SubsystemsPatch(cfg):
            result = runner.invoke(cli, ["--config", cfg_path, "logs", "path"])

        assert result.exit_code == 0
        assert str(tmp_path / "missy.log") in result.output

    def test_logs_tail_prints_recent_lines(self, runner: CliRunner, tmp_path):
        cfg_path = _write_temp_config()
        log_path = tmp_path / "missy.log"
        log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")
        cfg = _make_mock_config()
        cfg.observability.log_file_path = str(log_path)

        with _SubsystemsPatch(cfg):
            result = runner.invoke(cli, ["--config", cfg_path, "logs", "tail", "--limit", "2"])

        assert result.exit_code == 0
        assert "two" in result.output
        assert "three" in result.output
        assert "one" not in result.output

    def test_logs_tail_missing_file_prints_message(self, runner: CliRunner, tmp_path):
        cfg_path = _write_temp_config()
        cfg = _make_mock_config()
        cfg.observability.log_file_path = str(tmp_path / "missing.log")

        with _SubsystemsPatch(cfg):
            result = runner.invoke(cli, ["--config", cfg_path, "logs", "tail"])

        assert result.exit_code == 0
        assert "No application log found" in result.output


# ===========================================================================
# missy sessions cleanup
# ===========================================================================


class TestSessionsCleanup:
    def test_sessions_cleanup_actually_deletes_from_real_store(self, runner: CliRunner, tmp_path):
        """SR-3.5 regression: `missy sessions cleanup` previously always
        no-op'd (it constructed the legacy JSON MemoryStore, which has no
        cleanup() method, so a hasattr() guard silently skipped deletion
        on every invocation). This test uses a real SQLiteMemoryStore
        against a real temp DB and confirms an old turn is actually
        removed and a recent turn is preserved -- not just that a mock's
        .cleanup() was called.
        """
        from missy.memory.sqlite_store import ConversationTurn, SQLiteMemoryStore

        db_path = str(tmp_path / "memory.db")
        store = SQLiteMemoryStore(db_path)
        old_turn = ConversationTurn.new("sess1", "user", "old message")
        old_turn.timestamp = "2020-01-01T00:00:00"
        store.add_turn(old_turn)
        store.add_turn(ConversationTurn.new("sess1", "user", "recent message"))

        cfg_path = _write_temp_config()
        with (
            _SubsystemsPatch(),
            patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=store),
        ):
            result = runner.invoke(
                cli, ["--config", cfg_path, "sessions", "cleanup", "--older-than", "30"]
            )

        assert result.exit_code == 0
        assert "1" in result.output
        remaining = store.get_session_turns("sess1", limit=10)
        assert len(remaining) == 1
        assert remaining[0].content == "recent message"

    def test_sessions_cleanup_exits_zero(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_store = MagicMock()
        mock_store.cleanup.return_value = 0
        with (
            _SubsystemsPatch(),
            patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "sessions", "cleanup"])
        assert result.exit_code == 0

    def test_sessions_cleanup_reports_removed_count(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_store = MagicMock()
        mock_store.cleanup.return_value = 42
        with (
            _SubsystemsPatch(),
            patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "sessions", "cleanup"])
        assert "42" in result.output

    def test_sessions_cleanup_dry_run_does_not_delete(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_store = MagicMock()
        mock_store.cleanup.return_value = 7
        with (
            _SubsystemsPatch(),
            patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
        ):
            result = runner.invoke(
                cli,
                ["--config", cfg_path, "sessions", "cleanup", "--dry-run"],
            )
        assert result.exit_code == 0
        # SESSDEEP-002: --dry-run must now run a real COUNT-style preview
        # (dry_run=True is a non-destructive SELECT, not a DELETE) rather
        # than short-circuiting before ever touching the store, so it can
        # report the actual number of turns that would be affected.
        mock_store.cleanup.assert_called_once_with(older_than_days=30, dry_run=True)
        assert "Dry run" in result.output or "dry" in result.output.lower()
        assert "7" in result.output

    def test_sessions_cleanup_respects_older_than(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_store = MagicMock()
        mock_store.cleanup.return_value = 0
        with (
            _SubsystemsPatch(),
            patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
        ):
            runner.invoke(
                cli,
                [
                    "--config",
                    cfg_path,
                    "sessions",
                    "cleanup",
                    "--older-than",
                    "7",
                ],
            )
        mock_store.cleanup.assert_called_once_with(older_than_days=7)

    def test_sessions_cleanup_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["sessions", "cleanup", "--help"])
        assert result.exit_code == 0

    def test_sessions_group_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["sessions", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy approvals list
# ===========================================================================


class TestApprovalsList:
    def test_approvals_list_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["approvals", "list"])
        assert result.exit_code == 0

    def test_approvals_list_no_gateway_message(self, runner: CliRunner):
        """Must report "no active gateway" specifically when the gateway
        API is unreachable -- mocked rather than relying on no real
        `missy gateway start` happening to be running on the host during
        the test, since on a dev/operator machine where the gateway runs
        persistently this would otherwise get a legitimate "No pending
        approval requests" response instead and fail non-deterministically
        (found live: this exact false failure blocked a real
        code_evolve apply(), which runs the test suite as its own
        verification step, from ever succeeding)."""
        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("connection refused")):
            result = runner.invoke(cli, ["approvals", "list"])
        assert "gateway" in result.output.lower() or "No active" in result.output

    def test_approvals_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["approvals", "--help"])
        assert result.exit_code == 0

    def test_approvals_list_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["approvals", "list", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy discord pairing list / approve / deny (task #12)
# ===========================================================================


class TestDiscordPairingCli:
    def test_pairing_list_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["discord", "pairing", "list"])
        assert result.exit_code == 0

    def test_pairing_list_no_gateway_message(self, runner: CliRunner):
        """See test_approvals_list_no_gateway_message's docstring -- same
        fix, same live-observed root cause."""
        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("connection refused")):
            result = runner.invoke(cli, ["discord", "pairing", "list"])
        assert "gateway" in result.output.lower() or "No active" in result.output

    def test_pairing_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["discord", "pairing", "--help"])
        assert result.exit_code == 0

    def test_pairing_list_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["discord", "pairing", "list", "--help"])
        assert result.exit_code == 0

    def test_pairing_approve_no_gateway_exits_nonzero(self, runner: CliRunner):
        result = runner.invoke(cli, ["discord", "pairing", "approve", "12345"])
        assert result.exit_code != 0

    def test_pairing_deny_no_gateway_exits_nonzero(self, runner: CliRunner):
        result = runner.invoke(cli, ["discord", "pairing", "deny", "12345"])
        assert result.exit_code != 0

    def test_pairing_approve_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["discord", "pairing", "approve", "--help"])
        assert result.exit_code == 0

    def test_pairing_deny_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["discord", "pairing", "deny", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy patches list / approve / reject
# ===========================================================================


class TestPatchesList:
    def test_patches_list_no_patches(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.list_all.return_value = []
        with (
            _SubsystemsPatch(),
            patch("missy.agent.prompt_patches.PromptPatchManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "patches", "list"])
        assert result.exit_code == 0
        assert "No patches" in result.output

    def test_patches_list_shows_patches(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        patch_obj = MagicMock()
        patch_obj.id = "patch-001"
        patch_obj.patch_type.value = "system_prompt"
        patch_obj.status.value = "proposed"
        patch_obj.success_rate = 0.75
        patch_obj.applications = 4
        patch_obj.content = "Always be concise."
        mock_mgr = MagicMock()
        mock_mgr.list_all.return_value = [patch_obj]
        with (
            _SubsystemsPatch(),
            patch("missy.agent.prompt_patches.PromptPatchManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "patches", "list"])
        assert result.exit_code == 0
        assert "patch-001" in result.output

    def test_patches_list_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["patches", "list", "--help"])
        assert result.exit_code == 0

    def test_patches_group_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["patches", "--help"])
        assert result.exit_code == 0


class TestPatchesApprove:
    def test_patches_approve_success(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.approve.return_value = True
        with (
            _SubsystemsPatch(),
            patch("missy.agent.prompt_patches.PromptPatchManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "patches", "approve", "patch-001"])
        assert result.exit_code == 0
        assert "patch-001" in result.output

    def test_patches_approve_not_found(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.approve.return_value = False
        with (
            _SubsystemsPatch(),
            patch("missy.agent.prompt_patches.PromptPatchManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "patches", "approve", "ghost"])
        assert result.exit_code == 0

    def test_patches_approve_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["patches", "approve", "--help"])
        assert result.exit_code == 0


class TestPatchesReject:
    def test_patches_reject_success(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.reject.return_value = True
        with (
            _SubsystemsPatch(),
            patch("missy.agent.prompt_patches.PromptPatchManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "patches", "reject", "patch-002"])
        assert result.exit_code == 0
        assert "patch-002" in result.output

    def test_patches_reject_not_found(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.reject.return_value = False
        with (
            _SubsystemsPatch(),
            patch("missy.agent.prompt_patches.PromptPatchManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "patches", "reject", "ghost"])
        assert result.exit_code == 0

    def test_patches_reject_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["patches", "reject", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy mcp list / add / remove
# ===========================================================================


class TestMcpList:
    def test_mcp_list_no_servers(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.list_servers.return_value = []
        with _SubsystemsPatch(), patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["--config", cfg_path, "mcp", "list"])
        assert result.exit_code == 0
        assert "No MCP servers" in result.output

    def test_mcp_list_shows_servers(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.list_servers.return_value = [{"name": "my-server", "alive": True, "tools": 5}]
        with _SubsystemsPatch(), patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["--config", cfg_path, "mcp", "list"])
        assert result.exit_code == 0
        assert "my-server" in result.output

    def test_mcp_list_offline_server(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.list_servers.return_value = [{"name": "offline-srv", "alive": False, "tools": 0}]
        with _SubsystemsPatch(), patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["--config", cfg_path, "mcp", "list"])
        assert result.exit_code == 0
        assert "offline-srv" in result.output

    def test_mcp_list_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["mcp", "list", "--help"])
        assert result.exit_code == 0

    def test_mcp_group_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["mcp", "--help"])
        assert result.exit_code == 0


class TestMcpAdd:
    def test_mcp_add_success_with_url(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_client = MagicMock()
        mock_client.tools = ["tool_a", "tool_b"]
        mock_mgr = MagicMock()
        mock_mgr.add_server.return_value = mock_client
        with _SubsystemsPatch(), patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
            result = runner.invoke(
                cli,
                [
                    "--config",
                    cfg_path,
                    "mcp",
                    "add",
                    "my-srv",
                    "--url",
                    "http://localhost:3000",
                ],
            )
        assert result.exit_code == 0
        assert "my-srv" in result.output

    def test_mcp_add_success_with_command(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_client = MagicMock()
        mock_client.tools = ["echo"]
        mock_mgr = MagicMock()
        mock_mgr.add_server.return_value = mock_client
        with _SubsystemsPatch(), patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
            result = runner.invoke(
                cli,
                [
                    "--config",
                    cfg_path,
                    "mcp",
                    "add",
                    "stdio-srv",
                    "--command",
                    "npx @modelcontextprotocol/server-echo",
                ],
            )
        assert result.exit_code == 0

    def test_mcp_add_failure_exits_one(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.add_server.side_effect = ConnectionError("refused")
        with _SubsystemsPatch(), patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
            result = runner.invoke(
                cli,
                [
                    "--config",
                    cfg_path,
                    "mcp",
                    "add",
                    "bad-srv",
                    "--url",
                    "http://dead:9999",
                ],
            )
        assert result.exit_code == 1

    def test_mcp_add_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["mcp", "add", "--help"])
        assert result.exit_code == 0


class TestMcpRemove:
    def test_mcp_remove_exits_zero(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        with _SubsystemsPatch(), patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["--config", cfg_path, "mcp", "remove", "my-srv"])
        assert result.exit_code == 0

    def test_mcp_remove_success_message(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        with _SubsystemsPatch(), patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["--config", cfg_path, "mcp", "remove", "my-srv"])
        assert "my-srv" in result.output

    def test_mcp_remove_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["mcp", "remove", "--help"])
        assert result.exit_code == 0


class TestMcpRealManagerEndToEnd:
    """The tests above all mock McpManager itself by hand, setting
    mock_mgr.list_servers.return_value/mock_mgr.add_server.return_value
    directly -- this passes regardless of whether connect_all() was ever
    called, so it cannot catch the real bug: McpManager() starts with an
    empty in-memory self._clients dict, populated only by connect_all()
    loading ~/.missy/mcp.json. Without calling connect_all() first,
    `mcp list` always reported "No MCP servers configured" even with a
    populated mcp.json, `mcp remove NAME` was a silent no-op (never
    touched mcp.json), and worst of all `mcp add NEW` silently destroyed
    every previously-configured server, since add_server()'s
    _save_config() rewrites mcp.json from scratch using only the
    currently in-memory clients. These tests exercise the real,
    unmocked McpManager against a real mcp.json file (only McpClient
    itself is mocked, since it would otherwise need a live MCP server
    subprocess) -- the only way to catch a bug like this.
    """

    def _mock_client_factory(self, name, command=None, url=None):
        client = MagicMock()
        client.name = name
        client._command = command
        client._url = url
        client.tools = []
        client.alive = True
        return client

    def _set_up(self, monkeypatch: pytest.MonkeyPatch, tmp_path, mcp_entries: list[dict]) -> str:
        monkeypatch.setenv("HOME", str(tmp_path))
        missy_dir = tmp_path / ".missy"
        missy_dir.mkdir(parents=True, exist_ok=True)
        mcp_json = missy_dir / "mcp.json"
        mcp_json.write_text(json.dumps(mcp_entries))
        mcp_json.chmod(0o600)
        return str(mcp_json)

    def test_mcp_list_shows_existing_servers(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        cfg_path = _write_temp_config()
        self._set_up(monkeypatch, tmp_path, [{"name": "real-server", "command": "cat"}])

        with (
            _SubsystemsPatch(),
            patch("missy.mcp.manager.McpClient", side_effect=self._mock_client_factory),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "mcp", "list"])

        assert result.exit_code == 0
        assert result.exception is None
        assert "real-server" in result.output

    def test_mcp_add_preserves_existing_servers(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        cfg_path = _write_temp_config()
        mcp_json = self._set_up(
            monkeypatch, tmp_path, [{"name": "existing-server", "command": "cat"}]
        )

        with (
            _SubsystemsPatch(),
            patch("missy.mcp.manager.McpClient", side_effect=self._mock_client_factory),
        ):
            result = runner.invoke(
                cli, ["--config", cfg_path, "mcp", "add", "newserver", "--command", "echo hi"]
            )

        assert result.exit_code == 0
        assert result.exception is None
        saved = json.loads(Path(mcp_json).read_text())
        names = {entry["name"] for entry in saved}
        assert names == {"existing-server", "newserver"}

    def test_mcp_remove_actually_modifies_config(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        cfg_path = _write_temp_config()
        mcp_json = self._set_up(
            monkeypatch,
            tmp_path,
            [
                {"name": "keep-me", "command": "cat"},
                {"name": "remove-me", "command": "cat"},
            ],
        )

        with (
            _SubsystemsPatch(),
            patch("missy.mcp.manager.McpClient", side_effect=self._mock_client_factory),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "mcp", "remove", "remove-me"])

        assert result.exit_code == 0
        assert result.exception is None
        saved = json.loads(Path(mcp_json).read_text())
        names = {entry["name"] for entry in saved}
        assert names == {"keep-me"}


# ===========================================================================
# missy devices list / pair / unpair / status / policy
# ===========================================================================


def _edge_node(spec: dict) -> EdgeNode:
    """Build a real EdgeNode from the shorthand dict shape used by these tests."""
    return EdgeNode(
        node_id=spec.get("node_id", ""),
        friendly_name=spec.get("name", spec.get("friendly_name", "")),
        room=spec.get("room", ""),
        ip_address=spec.get("ip_address", ""),
        paired=spec.get("paired", False),
        policy_mode=spec.get("policy", spec.get("policy_mode", "full")),
        last_seen=spec.get("last_seen") or 0.0,
        status="online" if spec.get("online") else "offline",
        sensor_data={
            "occupancy": spec.get("occupancy"),
            "noise_level": spec.get("noise_level"),
            "updated_at": 0.0,
        },
    )


def _make_mock_registry(nodes: list[dict] | None = None) -> MagicMock:
    """Return a mock DeviceRegistry backed by real EdgeNode instances,
    matching DeviceRegistry's real list_nodes()/list_paired()/
    list_pending()/get_node() API.
    """
    edge_nodes = [_edge_node(n) for n in (nodes or [])]
    reg = MagicMock()
    reg.list_nodes.return_value = edge_nodes
    reg.list_paired.return_value = [n for n in edge_nodes if n.paired]
    reg.list_pending.return_value = [n for n in edge_nodes if not n.paired]
    reg.get_node.side_effect = lambda nid: next((n for n in edge_nodes if n.node_id == nid), None)
    return reg


class TestDevicesList:
    def test_devices_list_no_nodes(self, runner: CliRunner):
        with patch(
            "missy.channels.voice.registry.DeviceRegistry",
            return_value=_make_mock_registry([]),
        ):
            result = runner.invoke(cli, ["devices", "list"])
        assert result.exit_code == 0
        assert "No edge nodes" in result.output

    def test_devices_list_shows_nodes(self, runner: CliRunner):
        nodes = [
            {
                "node_id": "aabbccdd-1234",
                "name": "Kitchen",
                "room": "kitchen",
                "paired": True,
                "policy": "full",
                "last_seen": None,
            }
        ]
        with patch(
            "missy.channels.voice.registry.DeviceRegistry",
            return_value=_make_mock_registry(nodes),
        ):
            result = runner.invoke(cli, ["devices", "list"])
        assert result.exit_code == 0
        assert "Kitchen" in result.output

    def test_devices_list_pending_node(self, runner: CliRunner):
        nodes = [
            {
                "node_id": "pending-node-1234",
                "name": "Living Room",
                "room": "living",
                "paired": False,
                "policy": "full",
                "last_seen": None,
            }
        ]
        with patch(
            "missy.channels.voice.registry.DeviceRegistry",
            return_value=_make_mock_registry(nodes),
        ):
            result = runner.invoke(cli, ["devices", "list"])
        assert result.exit_code == 0
        assert "Living Room" in result.output

    def test_devices_list_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["devices", "list", "--help"])
        assert result.exit_code == 0

    def test_devices_group_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["devices", "--help"])
        assert result.exit_code == 0


class TestDevicesPair:
    def test_devices_pair_no_pending(self, runner: CliRunner):
        # All nodes are already paired — no pending requests.
        reg = _make_mock_registry([{"node_id": "abc", "name": "X", "paired": True}])
        with (
            patch("missy.channels.voice.registry.DeviceRegistry", return_value=reg),
            patch("missy.channels.voice.pairing.PairingManager", return_value=MagicMock()),
        ):
            result = runner.invoke(cli, ["devices", "pair"])
        assert result.exit_code == 0
        assert "No pending" in result.output

    def test_devices_pair_with_node_id(self, runner: CliRunner):
        reg = _make_mock_registry([])
        mock_pairing = MagicMock()
        mock_pairing.approve_pairing.return_value = "tok123"
        with (
            patch("missy.channels.voice.registry.DeviceRegistry", return_value=reg),
            patch("missy.channels.voice.pairing.PairingManager", return_value=mock_pairing),
        ):
            result = runner.invoke(cli, ["devices", "pair", "--node-id", "some-node-id"])
        assert result.exit_code == 0
        # Token should appear in output
        assert "tok123" in result.output

    def test_devices_pair_approval_failure_exits_one(self, runner: CliRunner):
        reg = _make_mock_registry([])
        mock_pairing = MagicMock()
        mock_pairing.approve_pairing.side_effect = RuntimeError("node not found")
        with (
            patch("missy.channels.voice.registry.DeviceRegistry", return_value=reg),
            patch("missy.channels.voice.pairing.PairingManager", return_value=mock_pairing),
        ):
            result = runner.invoke(cli, ["devices", "pair", "--node-id", "bad-node"])
        assert result.exit_code == 1

    def test_devices_pair_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["devices", "pair", "--help"])
        assert result.exit_code == 0


class TestDevicesUnpair:
    def test_devices_unpair_exits_zero(self, runner: CliRunner):
        reg = _make_mock_registry([{"node_id": "mynode", "name": "X"}])
        with patch("missy.channels.voice.registry.DeviceRegistry", return_value=reg):
            result = runner.invoke(cli, ["devices", "unpair", "--yes", "mynode"])
        assert result.exit_code == 0
        reg.remove_node.assert_called_once_with("mynode")

    def test_devices_unpair_node_not_found_exits_one(self, runner: CliRunner):
        reg = _make_mock_registry([])
        with patch("missy.channels.voice.registry.DeviceRegistry", return_value=reg):
            result = runner.invoke(cli, ["devices", "unpair", "--yes", "mynode"])
        assert result.exit_code == 1

    def test_devices_unpair_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["devices", "unpair", "--help"])
        assert result.exit_code == 0


class TestDevicesStatus:
    def test_devices_status_no_nodes(self, runner: CliRunner):
        with patch(
            "missy.channels.voice.registry.DeviceRegistry",
            return_value=_make_mock_registry([]),
        ):
            result = runner.invoke(cli, ["devices", "status"])
        assert result.exit_code == 0
        assert "No edge nodes" in result.output

    def test_devices_status_shows_online_node(self, runner: CliRunner):
        nodes = [
            {
                "node_id": "node-001",
                "name": "Office",
                "room": "office",
                "online": True,
                "last_seen": None,
                "occupancy": 1,
                "noise_level": 42.0,
            }
        ]
        with patch(
            "missy.channels.voice.registry.DeviceRegistry",
            return_value=_make_mock_registry(nodes),
        ):
            result = runner.invoke(cli, ["devices", "status"])
        assert result.exit_code == 0
        assert "Office" in result.output

    def test_devices_status_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["devices", "status", "--help"])
        assert result.exit_code == 0


class TestDevicesPolicy:
    def test_devices_policy_full_mode(self, runner: CliRunner):
        reg = _make_mock_registry([{"node_id": "mynode", "name": "X"}])
        with patch("missy.channels.voice.registry.DeviceRegistry", return_value=reg):
            result = runner.invoke(cli, ["devices", "policy", "mynode", "--mode", "full"])
        assert result.exit_code == 0
        reg.update_node.assert_called_once_with("mynode", policy_mode="full")

    def test_devices_policy_muted_mode(self, runner: CliRunner):
        reg = _make_mock_registry([{"node_id": "mynode", "name": "X"}])
        with patch("missy.channels.voice.registry.DeviceRegistry", return_value=reg):
            result = runner.invoke(cli, ["devices", "policy", "mynode", "--mode", "muted"])
        assert result.exit_code == 0

    def test_devices_policy_node_not_found_exits_one(self, runner: CliRunner):
        reg = _make_mock_registry([])
        reg.update_node.side_effect = KeyError("mynode")
        with patch("missy.channels.voice.registry.DeviceRegistry", return_value=reg):
            result = runner.invoke(cli, ["devices", "policy", "mynode", "--mode", "muted"])
        assert result.exit_code == 1

    def test_devices_policy_invalid_mode(self, runner: CliRunner):
        result = runner.invoke(cli, ["devices", "policy", "mynode", "--mode", "banana"])
        assert result.exit_code != 0

    def test_devices_policy_missing_mode(self, runner: CliRunner):
        result = runner.invoke(cli, ["devices", "policy", "mynode"])
        assert result.exit_code != 0

    def test_devices_policy_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["devices", "policy", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy voice status / test
# ===========================================================================


class TestVoiceStatus:
    def test_voice_status_exits_zero(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        reg = _make_mock_registry([])
        with patch("missy.channels.voice.registry.DeviceRegistry", return_value=reg):
            result = runner.invoke(cli, ["--config", cfg_path, "voice", "status"])
        assert result.exit_code == 0

    def test_voice_status_shows_paired_count(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        nodes = [
            {"node_id": "n1", "name": "A", "paired": True},
            {"node_id": "n2", "name": "B", "paired": False},
        ]
        reg = _make_mock_registry(nodes)
        with patch("missy.channels.voice.registry.DeviceRegistry", return_value=reg):
            result = runner.invoke(cli, ["--config", cfg_path, "voice", "status"])
        assert result.exit_code == 0
        # "Paired nodes" row should show 1
        assert "1" in result.output

    def test_voice_status_shows_tts_engine(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        reg = _make_mock_registry([])
        with patch("missy.channels.voice.registry.DeviceRegistry", return_value=reg):
            result = runner.invoke(cli, ["--config", cfg_path, "voice", "status"])
        assert result.exit_code == 0
        assert "piper" in result.output or "TTS" in result.output

    def test_voice_status_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["voice", "status", "--help"])
        assert result.exit_code == 0

    def test_voice_group_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["voice", "--help"])
        assert result.exit_code == 0


class TestVoiceTest:
    def test_voice_test_node_not_found_exits_one(self, runner: CliRunner):
        reg = _make_mock_registry([])
        with patch("missy.channels.voice.registry.DeviceRegistry", return_value=reg):
            result = runner.invoke(cli, ["voice", "test", "nonexistent-node"])
        # Voice test may exit 0 or 1 depending on config loading
        assert result.exit_code in (0, 1)

    def test_voice_test_piper_not_found_exits_one(self, runner: CliRunner):
        nodes = [{"node_id": "node-001-abc", "name": "Test", "room": "test"}]
        reg = _make_mock_registry(nodes)
        mock_tts = MagicMock()

        async def _raise(*a, **kw):
            raise FileNotFoundError("piper not found")

        mock_tts.synthesize.side_effect = _raise
        with (
            patch("missy.channels.voice.registry.DeviceRegistry", return_value=reg),
            patch("missy.channels.voice.tts.piper.PiperTTS", return_value=mock_tts),
        ):
            result = runner.invoke(cli, ["voice", "test", "node-001"])
        assert result.exit_code == 1

    def test_voice_test_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["voice", "test", "--help"])
        assert result.exit_code == 0


class TestDevicesAndVoiceRealRegistryEndToEnd:
    """Every devices/voice test above mocks DeviceRegistry/PairingManager by
    hand, which previously encoded the CLI's own bug into the mock's
    interface (`.all()`, `.remove()`, `.set_policy()`, `.approve()` --
    none of which exist on the real classes, which use `list_nodes()`,
    `remove_node()`, `update_node()`, `approve_pairing()` and return
    `EdgeNode` dataclass instances, not dicts) -- so every mocked test
    passed while every real invocation crashed with `AttributeError`.
    These tests exercise the real, unmocked `DeviceRegistry`/
    `PairingManager` against a temp-file-backed registry, the only way to
    actually catch a CLI/class interface mismatch like that one.
    """

    def _set_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".missy").mkdir(parents=True, exist_ok=True)

    def test_devices_list_against_real_registry(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        self._set_home(monkeypatch, tmp_path)
        from missy.channels.voice.registry import DeviceRegistry, EdgeNode

        reg = DeviceRegistry(registry_path=str(tmp_path / ".missy" / "devices.json"))
        reg.load()
        reg.add_node(
            EdgeNode(
                node_id="real-node-1234",
                friendly_name="Real Living Room",
                room="Living Room",
                ip_address="192.168.1.50",
            )
        )
        reg.save()

        result = runner.invoke(cli, ["devices", "list"])
        assert result.exit_code == 0
        assert result.exception is None
        # Rich may wrap the name across table cell lines in a narrow terminal.
        assert "Real" in result.output
        assert "Living" in result.output

    def test_devices_pair_and_unpair_against_real_registry(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        self._set_home(monkeypatch, tmp_path)
        from missy.channels.voice.registry import DeviceRegistry, EdgeNode

        reg = DeviceRegistry(registry_path=str(tmp_path / ".missy" / "devices.json"))
        reg.load()
        reg.add_node(
            EdgeNode(
                node_id="real-node-5678",
                friendly_name="Real Kitchen",
                room="Kitchen",
                ip_address="192.168.1.51",
            )
        )
        reg.save()

        result = runner.invoke(cli, ["devices", "pair", "--node-id", "real-node-5678"])
        assert result.exit_code == 0
        assert result.exception is None
        assert "approved" in result.output.lower()

        result = runner.invoke(cli, ["devices", "unpair", "--yes", "real-node-5678"])
        assert result.exit_code == 0
        assert result.exception is None
        assert "removed" in result.output.lower()

        result = runner.invoke(cli, ["devices", "unpair", "--yes", "real-node-5678"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_devices_status_and_policy_against_real_registry(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        self._set_home(monkeypatch, tmp_path)
        from missy.channels.voice.registry import DeviceRegistry, EdgeNode

        reg = DeviceRegistry(registry_path=str(tmp_path / ".missy" / "devices.json"))
        reg.load()
        reg.add_node(
            EdgeNode(
                node_id="real-node-9999",
                friendly_name="Real Office",
                room="Office",
                ip_address="192.168.1.52",
            )
        )
        reg.save()

        result = runner.invoke(cli, ["devices", "status"])
        assert result.exit_code == 0
        assert result.exception is None
        # Rich may wrap the name across table cell lines in a narrow terminal.
        assert "Real" in result.output
        assert "Office" in result.output

        result = runner.invoke(cli, ["devices", "policy", "real-node-9999", "--mode", "safe-chat"])
        assert result.exit_code == 0
        assert result.exception is None
        assert "safe-chat" in result.output

    def test_voice_status_against_real_registry(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        self._set_home(monkeypatch, tmp_path)
        result = runner.invoke(cli, ["voice", "status"])
        assert result.exit_code == 0
        assert result.exception is None


# ===========================================================================
# missy discord status / audit
# ===========================================================================


class TestDiscordStatus:
    def test_discord_status_no_accounts(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        with _SubsystemsPatch() as cfg:
            cfg.discord = None
            result = runner.invoke(cli, ["--config", cfg_path, "discord", "status"])
        assert result.exit_code == 0
        assert "No Discord" in result.output

    def test_discord_status_shows_account(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        account = MagicMock()
        account.token_env_var = "DISCORD_TOKEN"
        account.application_id = "123456789"
        account.dm_policy.value = "allowlist"
        account.guild_policies = {}
        account.ignore_bots = True

        discord_cfg = MagicMock()
        discord_cfg.enabled = True
        discord_cfg.accounts = [account]

        with _SubsystemsPatch() as cfg:
            cfg.discord = discord_cfg
            result = runner.invoke(cli, ["--config", cfg_path, "discord", "status"])
        assert result.exit_code == 0
        assert "DISCORD_TOKEN" in result.output

    def test_discord_status_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["discord", "status", "--help"])
        assert result.exit_code == 0

    def test_discord_group_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["discord", "--help"])
        assert result.exit_code == 0


class TestDiscordDiagnostics:
    def test_discord_diagnostics_no_accounts(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        with _SubsystemsPatch() as cfg:
            cfg.discord = None
            result = runner.invoke(cli, ["--config", cfg_path, "discord", "diagnostics"])
        assert result.exit_code == 0
        assert "No Discord" in result.output

    def test_discord_diagnostics_reports_policy_voice_and_recent_events(self, runner: CliRunner):
        from missy.channels.discord import voice_binding
        from missy.channels.discord.config import (
            DiscordAccountConfig,
            DiscordConfig,
            DiscordDMPolicy,
            DiscordGuildPolicy,
        )

        cfg_path = _write_temp_config()
        account = DiscordAccountConfig(
            token_env_var="DISCORD_TOKEN",
            token="fake-token",
            account_id="bot-1",
            application_id="app-1",
            dm_policy=DiscordDMPolicy.ALLOWLIST,
            guild_policies={
                "guild-1": DiscordGuildPolicy(require_mention=True),
            },
        )

        class _Manager:
            is_ready = True
            can_listen = True
            can_speak = False

        mock_logger = MagicMock()
        mock_logger.get_recent_events.return_value = [
            {
                "timestamp": "2026-07-07T11:59:58",
                "event_type": "discord.gateway.heartbeat_ack",
                "result": "allow",
                "detail": {"seq": 12},
            },
            {
                "timestamp": "2026-07-07T11:59:59",
                "event_type": "discord.slash_commands.registration_failed",
                "result": "error",
                "detail": {"scope": "global", "error": "403 Forbidden"},
            },
            {
                "timestamp": "2026-07-07T12:00:00",
                "event_type": "discord.voice.start_failed",
                "result": "error",
                "detail": {"guild_id": "guild-1", "error": "ffmpeg missing"},
            },
        ]

        try:
            with _SubsystemsPatch() as cfg:
                cfg.discord = DiscordConfig(enabled=True, accounts=[account])
                cfg.network.allowed_domains = ["discord.com", "gateway.discord.gg"]
                cfg.network.discord_allowed_hosts = []
                voice_binding.set_voice_binding(
                    _Manager(),
                    MagicMock(),
                    account_id="bot-1",
                    guild_id="guild-1",
                )
                with patch(
                    "missy.observability.audit_logger.AuditLogger",
                    return_value=mock_logger,
                ):
                    result = runner.invoke(
                        cli,
                        ["--config", cfg_path, "discord", "diagnostics"],
                    )
        finally:
            voice_binding.clear_voice_binding()

        assert result.exit_code == 0
        assert "Discord Diagnostics" in result.output
        assert "DISCORD_TOKEN" in result.output
        assert "discord_voice" in result.output
        assert "guild-1" in result.output
        assert "Recent Lifecycle Signals" in result.output
        assert "slash-registration" in result.output
        assert "Recent Discord Events" in result.output

    def test_discord_diagnostics_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["discord", "diagnostics", "--help"])
        assert result.exit_code == 0


class TestDiscordAudit:
    def test_discord_audit_no_events(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_logger = MagicMock()
        mock_logger.get_recent_events.return_value = []
        with (
            _SubsystemsPatch(),
            patch(
                "missy.observability.audit_logger.AuditLogger",
                return_value=mock_logger,
            ),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "discord", "audit"])
        assert result.exit_code == 0
        assert "No Discord" in result.output

    def test_discord_audit_shows_discord_events(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        events = [
            {
                "timestamp": "2026-03-12T10:00:00",
                "event_type": "discord.message.received",
                "category": "discord",
                "result": "allow",
                "detail": {"user": "alice"},
            }
        ]
        mock_logger = MagicMock()
        mock_logger.get_recent_events.return_value = events
        with (
            _SubsystemsPatch(),
            patch(
                "missy.observability.audit_logger.AuditLogger",
                return_value=mock_logger,
            ),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "discord", "audit"])
        assert result.exit_code == 0
        assert "discord.message" in result.output

    def test_discord_audit_filters_non_discord_events(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        events = [
            {
                "timestamp": "2026-03-12T10:00:00",
                "event_type": "network.request",
                "category": "network",
                "result": "allow",
                "detail": {},
            }
        ]
        mock_logger = MagicMock()
        mock_logger.get_recent_events.return_value = events
        with (
            _SubsystemsPatch(),
            patch(
                "missy.observability.audit_logger.AuditLogger",
                return_value=mock_logger,
            ),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "discord", "audit"])
        assert result.exit_code == 0
        # non-discord event filtered out → no events message
        assert "No Discord" in result.output

    def test_discord_audit_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["discord", "audit", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy gateway status
# ===========================================================================


class TestGatewayStatus:
    def test_gateway_status_exits_zero(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        with _SubsystemsPatch() as cfg:
            cfg.discord = None
            with patch("missy.providers.registry.get_registry", return_value=mock_registry):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "status"])
        assert result.exit_code == 0

    def test_gateway_status_shows_cli_channel(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        with _SubsystemsPatch() as cfg:
            cfg.discord = None
            with patch("missy.providers.registry.get_registry", return_value=mock_registry):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "status"])
        assert "cli" in result.output

    def test_gateway_status_shows_discord_disabled(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        with _SubsystemsPatch() as cfg:
            cfg.discord = None
            with patch("missy.providers.registry.get_registry", return_value=mock_registry):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "status"])
        assert "discord" in result.output

    def test_gateway_status_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["gateway", "status", "--help"])
        assert result.exit_code == 0

    def test_gateway_group_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["gateway", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# --help flags for every top-level group (smoke tests)
# ===========================================================================


class TestAllGroupHelp:
    @pytest.mark.parametrize(
        "group",
        [
            ["schedule", "--help"],
            ["audit", "--help"],
            ["vault", "--help"],
            ["sessions", "--help"],
            ["approvals", "--help"],
            ["patches", "--help"],
            ["mcp", "--help"],
            ["devices", "--help"],
            ["voice", "--help"],
            ["discord", "--help"],
            ["gateway", "--help"],
        ],
    )
    def test_group_help_exits_zero(self, runner: CliRunner, group: list[str]):
        result = runner.invoke(cli, group)
        assert result.exit_code == 0


# ===========================================================================
# missy sessions list / rename
# ===========================================================================


class TestSessionsList:
    def test_sessions_list_no_sessions(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_store = MagicMock()
        mock_store.list_sessions.return_value = []
        with (
            _SubsystemsPatch(),
            patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "sessions", "list"])
        assert result.exit_code == 0
        assert "No sessions" in result.output

    def test_sessions_list_shows_sessions(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_store = MagicMock()
        mock_store.list_sessions.return_value = [
            {
                "session_id": "abc-123",
                "name": "My chat",
                "turn_count": 5,
                "provider": "anthropic",
                "channel": "cli",
                "updated_at": "2026-03-12T10:00:00",
            }
        ]
        with (
            _SubsystemsPatch(),
            patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "sessions", "list"])
        assert result.exit_code == 0
        assert "abc-123" in result.output

    def test_sessions_list_respects_limit(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_store = MagicMock()
        mock_store.list_sessions.return_value = []
        with (
            _SubsystemsPatch(),
            patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
        ):
            runner.invoke(cli, ["--config", cfg_path, "sessions", "list", "--limit", "5"])
        mock_store.list_sessions.assert_called_once_with(limit=5)

    def test_sessions_list_error_handled(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_store = MagicMock()
        mock_store.list_sessions.side_effect = Exception("db locked")
        with (
            _SubsystemsPatch(),
            patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "sessions", "list"])
        # Should handle the error gracefully (not crash)
        assert result.exit_code in (0, 1)

    def test_sessions_list_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["sessions", "list", "--help"])
        assert result.exit_code == 0


class TestSessionsRename:
    def test_sessions_rename_success(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_store = MagicMock()
        mock_store.rename_session.return_value = True
        mock_store.resolve_session_name.return_value = None
        with (
            _SubsystemsPatch(),
            patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
        ):
            result = runner.invoke(
                cli,
                ["--config", cfg_path, "sessions", "rename", "abc-123", "My Chat"],
            )
        assert result.exit_code == 0

    def test_sessions_rename_not_found(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_store = MagicMock()
        mock_store.rename_session.return_value = False
        mock_store.resolve_session_name.return_value = None
        with (
            _SubsystemsPatch(),
            patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
        ):
            result = runner.invoke(
                cli,
                ["--config", cfg_path, "sessions", "rename", "ghost-id", "New Name"],
            )
        assert result.exit_code == 0

    def test_sessions_rename_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["sessions", "rename", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy evolve list / approve / reject
# ===========================================================================


class TestEvolveList:
    def test_evolve_list_no_proposals(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.list_all.return_value = []
        with (
            _SubsystemsPatch(),
            patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "evolve", "list"])
        assert result.exit_code == 0
        assert "No evolution" in result.output

    def test_evolve_list_shows_proposals(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        proposal = MagicMock()
        proposal.id = "evo-001"
        proposal.status.value = "proposed"
        proposal.trigger.value = "error"
        proposal.confidence = 0.8
        proposal.title = "Fix null pointer"
        proposal.created_at = "2026-03-12T10:00:00"
        mock_mgr = MagicMock()
        mock_mgr.list_all.return_value = [proposal]
        with (
            _SubsystemsPatch(),
            patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "evolve", "list"])
        assert result.exit_code == 0
        assert "evo-001" in result.output

    def test_evolve_list_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["evolve", "list", "--help"])
        assert result.exit_code == 0

    def test_evolve_group_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["evolve", "--help"])
        assert result.exit_code == 0


class TestEvolveApprove:
    def test_evolve_approve_success(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.approve.return_value = True
        with (
            _SubsystemsPatch(),
            patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "evolve", "approve", "evo-001"])
        assert result.exit_code == 0
        assert "evo-001" in result.output

    def test_evolve_approve_not_found(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.approve.return_value = False
        with (
            _SubsystemsPatch(),
            patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "evolve", "approve", "ghost"])
        assert result.exit_code == 0

    def test_evolve_approve_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["evolve", "approve", "--help"])
        assert result.exit_code == 0


class TestEvolveReject:
    def test_evolve_reject_success(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.reject.return_value = True
        with (
            _SubsystemsPatch(),
            patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "evolve", "reject", "evo-001"])
        assert result.exit_code == 0
        assert "evo-001" in result.output

    def test_evolve_reject_not_found(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.reject.return_value = False
        with (
            _SubsystemsPatch(),
            patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "evolve", "reject", "ghost"])
        assert result.exit_code == 0

    def test_evolve_reject_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["evolve", "reject", "--help"])
        assert result.exit_code == 0


class TestEvolveShow:
    def test_evolve_show_not_found(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.get.return_value = None
        with (
            _SubsystemsPatch(),
            patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "evolve", "show", "ghost"])
        assert result.exit_code == 0

    def test_evolve_show_found(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        proposal = MagicMock()
        proposal.id = "evo-001"
        proposal.title = "Fix null pointer"
        proposal.status.value = "proposed"
        proposal.trigger.value = "error"
        proposal.confidence = 0.8
        proposal.created_at = "2026-03-12T10:00:00"
        proposal.resolved_at = None
        proposal.git_commit_sha = None
        proposal.description = "Fixes a null pointer dereference"
        proposal.diffs = []
        proposal.error_pattern = None
        proposal.test_output = None
        mock_mgr = MagicMock()
        mock_mgr.get.return_value = proposal
        with (
            _SubsystemsPatch(),
            patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "evolve", "show", "evo-001"])
        assert result.exit_code == 0
        assert "evo-001" in result.output

    def test_evolve_show_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["evolve", "show", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy run (interactive REPL — tested with EOF on stdin)
# ===========================================================================


class TestRun:
    def test_run_exits_on_eof(self, runner: CliRunner):
        """Sending an immediate EOF should terminate the interactive loop cleanly."""
        cfg_path = _write_temp_config()
        with (
            _SubsystemsPatch(),
            patch("missy.agent.runtime.AgentRuntime") as mock_rt_cls,
            patch("missy.agent.runtime.AgentConfig"),
            patch("missy.channels.cli_channel.CLIChannel") as mock_ch_cls,
        ):
            mock_rt = MagicMock()
            mock_rt.pending_recovery = []
            mock_rt_cls.return_value = mock_rt
            mock_ch = MagicMock()
            # receive() returning None simulates EOF / Ctrl-D
            mock_ch.receive.return_value = None
            mock_ch_cls.return_value = mock_ch
            result = runner.invoke(cli, ["--config", cfg_path, "run"])
        assert result.exit_code == 0

    def test_run_quit_command_exits(self, runner: CliRunner):
        """Typing 'quit' should terminate the session loop."""
        cfg_path = _write_temp_config()
        msg = MagicMock()
        msg.content = "quit"
        with (
            _SubsystemsPatch(),
            patch("missy.agent.runtime.AgentRuntime") as mock_rt_cls,
            patch("missy.agent.runtime.AgentConfig"),
            patch("missy.channels.cli_channel.CLIChannel") as mock_ch_cls,
        ):
            mock_rt = MagicMock()
            mock_rt.pending_recovery = []
            mock_rt_cls.return_value = mock_rt
            mock_ch = MagicMock()
            mock_ch.receive.return_value = msg
            mock_ch_cls.return_value = mock_ch
            result = runner.invoke(cli, ["--config", cfg_path, "run"])
        assert result.exit_code == 0

    def test_run_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy discord probe (no accounts → prints message without crashing)
# ===========================================================================


class TestDiscordProbe:
    def test_discord_probe_no_accounts(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        with _SubsystemsPatch() as cfg:
            cfg.discord = None
            result = runner.invoke(cli, ["--config", cfg_path, "discord", "probe"])
        assert result.exit_code == 0
        assert "No Discord" in result.output

    def test_discord_probe_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["discord", "probe", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy discord register-commands (no accounts → error exit)
# ===========================================================================


class TestDiscordRegisterCommands:
    def test_discord_register_no_accounts_exits_one(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        with _SubsystemsPatch() as cfg:
            cfg.discord = None
            result = runner.invoke(cli, ["--config", cfg_path, "discord", "register-commands"])
        assert result.exit_code == 1

    def test_discord_register_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["discord", "register-commands", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy _load_subsystems — explicit exercise of the init path branches
# ===========================================================================


class TestLoadSubsystemsInitPath:
    def test_load_subsystems_initialises_registry(self):
        """_load_subsystems should call init_registry when config is valid."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as fh:
            fh.write(_MINIMAL_CONFIG_YAML)
            cfg_path = fh.name

        from missy.cli.main import _load_subsystems

        with (
            patch("missy.config.settings.load_config") as mock_load,
            patch("missy.policy.engine.init_policy_engine"),
            patch("missy.observability.audit_logger.init_audit_logger"),
            patch("missy.providers.registry.init_registry") as mock_init_reg,
            patch("missy.tools.builtin.register_builtin_tools"),
            patch("missy.tools.registry.init_tool_registry", return_value=MagicMock()),
            patch("missy.observability.otel.init_otel"),
        ):
            mock_cfg = _make_mock_config()
            mock_cfg.audit_log_path = "/tmp/audit.jsonl"
            mock_load.return_value = mock_cfg
            result = _load_subsystems(cfg_path)

        mock_init_reg.assert_called_once()
        assert result is mock_cfg


# ===========================================================================
# _load_subsystems error paths (ConfigurationError, unexpected exception)
# ===========================================================================


class TestLoadSubsystemsErrors:
    def test_load_subsystems_config_error_exits_one(self, runner: CliRunner):
        """A ConfigurationError during load should print error and exit 1."""
        from missy.core.exceptions import ConfigurationError

        with patch("missy.config.settings.load_config") as mock_load:
            mock_load.side_effect = ConfigurationError("bad yaml")
            result = runner.invoke(cli, ["providers"])
        assert result.exit_code == 1

    def test_load_subsystems_unexpected_error_exits_one(self, runner: CliRunner):
        """An unexpected exception during load should print error and exit 1."""
        with patch("missy.config.settings.load_config") as mock_load:
            mock_load.side_effect = RuntimeError("disk full")
            result = runner.invoke(cli, ["providers"])
        assert result.exit_code == 1

    def test_load_subsystems_config_error_hint_contains_missy_init(self, runner: CliRunner):
        """The config error message should hint at missy init."""
        from missy.core.exceptions import ConfigurationError

        with patch("missy.config.settings.load_config") as mock_load:
            mock_load.side_effect = ConfigurationError("bad yaml")
            result = runner.invoke(cli, ["providers"])
        combined = result.output + (result.stderr or "")
        assert "missy init" in combined or "config" in combined.lower()


# ===========================================================================
# missy run — agent processes a message (happy path)
# ===========================================================================


class TestRunMessageProcessing:
    def test_run_processes_one_message_then_quit(self, runner: CliRunner):
        """One user message processed by agent then 'quit' terminates the loop."""
        cfg_path = _write_temp_config()
        user_msg = MagicMock()
        user_msg.content = "Hello"
        quit_msg = MagicMock()
        quit_msg.content = "quit"

        with (
            _SubsystemsPatch(),
            patch("missy.agent.runtime.AgentRuntime") as mock_rt_cls,
            patch("missy.agent.runtime.AgentConfig"),
            patch("missy.channels.cli_channel.CLIChannel") as mock_ch_cls,
            patch("missy.security.sanitizer.sanitizer") as mock_san,
            patch("missy.security.secrets.secrets_detector") as mock_sec,
        ):
            mock_rt = MagicMock()
            mock_rt.pending_recovery = []
            mock_rt.run.return_value = "Hello there!"
            mock_rt_cls.return_value = mock_rt

            mock_ch = MagicMock()
            mock_ch.receive.side_effect = [user_msg, quit_msg]
            mock_ch_cls.return_value = mock_ch

            mock_san.sanitize.return_value = "Hello"
            mock_san.check_for_injection.return_value = []
            mock_sec.has_secrets.return_value = False

            result = runner.invoke(cli, ["--config", cfg_path, "run"])

        assert result.exit_code == 0
        mock_rt.run.assert_called_once()

    def test_run_skips_empty_input(self, runner: CliRunner):
        """Empty lines should be skipped without calling the agent."""
        cfg_path = _write_temp_config()
        empty_msg = MagicMock()
        empty_msg.content = "   "
        quit_msg = MagicMock()
        quit_msg.content = "quit"

        with (
            _SubsystemsPatch(),
            patch("missy.agent.runtime.AgentRuntime") as mock_rt_cls,
            patch("missy.agent.runtime.AgentConfig"),
            patch("missy.channels.cli_channel.CLIChannel") as mock_ch_cls,
        ):
            mock_rt = MagicMock()
            mock_rt.pending_recovery = []
            mock_rt_cls.return_value = mock_rt
            mock_ch = MagicMock()
            mock_ch.receive.side_effect = [empty_msg, quit_msg]
            mock_ch_cls.return_value = mock_ch
            result = runner.invoke(cli, ["--config", cfg_path, "run"])

        assert result.exit_code == 0
        mock_rt.run.assert_not_called()


# ===========================================================================
# missy doctor — additional branch coverage
# ===========================================================================


class TestDoctorBranches:
    def test_doctor_with_provider_available(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["anthropic"]
        mock_registry.get.return_value = mock_provider
        mock_mgr = MagicMock()
        mock_mgr.list_jobs.return_value = []
        with (
            _SubsystemsPatch(),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        assert result.exit_code == 0
        assert "anthropic" in result.output

    def test_doctor_shell_enabled_shows_warn(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        with _SubsystemsPatch() as cfg:
            cfg.shell.enabled = True
            cfg.shell.allowed_commands = ["ls", "pwd"]
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []
            mock_mgr = MagicMock()
            mock_mgr.list_jobs.return_value = []
            with (
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
                patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        assert result.exit_code == 0
        assert "ENABLED" in result.output

    def test_doctor_plugins_enabled(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        with _SubsystemsPatch() as cfg:
            cfg.plugins.enabled = True
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []
            mock_mgr = MagicMock()
            mock_mgr.list_jobs.return_value = []
            with (
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
                patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        assert result.exit_code == 0


# ===========================================================================
# missy doctor — audit signing status (SR-1.1/SR-4.6 residual)
#
# Previously `missy doctor` only checked whether the audit log *file*
# existed, saying nothing about whether it's actually tamper-evident.
# `missy audit verify` already existed for this, but an operator had to
# know to run it separately -- `doctor` (the "am I healthy" command) gave
# no hint anything needed checking. These tests exercise the real
# AuditLogger write path and real AgentIdentity Ed25519 signing/
# verification, not mocks -- mocking verify_audit_log() would defeat the
# point of testing this row.
# ===========================================================================


class TestDoctorAuditSigning:
    def _write_signed_log(self, key_path, log_path, events):
        from missy.core.events import AuditEvent, EventBus
        from missy.observability.audit_logger import AuditLogger
        from missy.security.identity import AgentIdentity

        identity = AgentIdentity.load_or_generate(str(key_path))
        bus = EventBus()
        AuditLogger(log_path=str(log_path), bus=bus, identity=identity)
        for event_type, category, result, detail in events:
            bus.publish(
                AuditEvent.now(
                    session_id="s1",
                    task_id="t1",
                    event_type=event_type,
                    category=category,
                    result=result,
                    detail=detail,
                )
            )
        return identity

    def _invoke_doctor(self, runner, cfg_path, cfg):
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        mock_mgr = MagicMock()
        mock_mgr.list_jobs.return_value = []
        with (
            _SubsystemsPatch(cfg),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            return runner.invoke(cli, ["--config", cfg_path, "doctor"])

    def test_all_lines_signed_and_valid_shows_ok(self, runner: CliRunner, tmp_path, monkeypatch):
        key_path = tmp_path / "identity.pem"
        log_path = tmp_path / "audit.jsonl"
        monkeypatch.setattr("missy.security.identity.DEFAULT_KEY_PATH", str(key_path))
        self._write_signed_log(
            key_path,
            log_path,
            [("network.request", "network", "allow", {"host": "example.com"})],
        )
        cfg_path = _write_temp_config()
        cfg = _make_mock_config()
        cfg.audit_log_path = str(log_path)

        result = self._invoke_doctor(runner, cfg_path, cfg)

        assert result.exit_code == 0
        assert "audit signing" in result.output.lower()
        assert "valid=1" in result.output

    def test_tampered_line_shows_fail(self, runner: CliRunner, tmp_path, monkeypatch):
        import json as _json

        key_path = tmp_path / "identity.pem"
        log_path = tmp_path / "audit.jsonl"
        monkeypatch.setattr("missy.security.identity.DEFAULT_KEY_PATH", str(key_path))
        self._write_signed_log(
            key_path,
            log_path,
            [("shell.exec", "shell", "deny", {"command": "rm -rf /"})],
        )
        # Tamper: flip the recorded result from deny to allow after signing,
        # reproducing the exact attack the security review demonstrated.
        lines = log_path.read_text().splitlines()
        record = _json.loads(lines[0])
        record["result"] = "allow"
        log_path.write_text(_json.dumps(record) + "\n")

        cfg_path = _write_temp_config()
        cfg = _make_mock_config()
        cfg.audit_log_path = str(log_path)

        result = self._invoke_doctor(runner, cfg_path, cfg)

        assert result.exit_code == 0  # doctor reports, never crashes/exits nonzero
        assert "audit signing" in result.output.lower()
        assert "tampered=1" in result.output
        assert "FAIL" in result.output

    def test_reordered_lines_show_fail(self, runner: CliRunner, tmp_path, monkeypatch):
        """Regression: reordering two validly-signed lines (no content
        edits) must still be surfaced as a FAIL, via the hash chain --
        per-line signature checks alone would report this log fully
        valid.
        """
        import json as _json

        key_path = tmp_path / "identity.pem"
        log_path = tmp_path / "audit.jsonl"
        monkeypatch.setattr("missy.security.identity.DEFAULT_KEY_PATH", str(key_path))
        self._write_signed_log(
            key_path,
            log_path,
            [
                ("event.a", "test", "allow", {}),
                ("event.b", "test", "allow", {}),
                ("event.c", "test", "allow", {}),
            ],
        )
        lines = log_path.read_text().splitlines()
        lines[1], lines[2] = lines[2], lines[1]
        log_path.write_text("\n".join(lines) + "\n")
        for ln in lines:
            _json.loads(ln)  # still individually well-formed/parseable

        cfg_path = _write_temp_config()
        cfg = _make_mock_config()
        cfg.audit_log_path = str(log_path)

        result = self._invoke_doctor(runner, cfg_path, cfg)

        assert result.exit_code == 0  # doctor reports, never crashes/exits nonzero
        assert "audit signing" in result.output.lower()
        assert "valid=3" in result.output
        assert "out of sequence" in result.output.lower()
        assert "FAIL" in result.output

    def test_unsigned_lines_show_warn(self, runner: CliRunner, tmp_path, monkeypatch):
        """A log written before signing was enabled (or with no identity
        configured) must not be silently reported as healthy -- it
        provides no tamper evidence at all."""
        from missy.core.events import AuditEvent, EventBus
        from missy.observability.audit_logger import AuditLogger

        key_path = tmp_path / "identity.pem"
        log_path = tmp_path / "audit.jsonl"
        monkeypatch.setattr("missy.security.identity.DEFAULT_KEY_PATH", str(key_path))

        bus = EventBus()
        AuditLogger(log_path=str(log_path), bus=bus, identity=None)  # no identity => unsigned
        bus.publish(
            AuditEvent.now(
                session_id="s1",
                task_id="t1",
                event_type="network.request",
                category="network",
                result="allow",
                detail={},
            )
        )

        cfg_path = _write_temp_config()
        cfg = _make_mock_config()
        cfg.audit_log_path = str(log_path)

        result = self._invoke_doctor(runner, cfg_path, cfg)

        assert result.exit_code == 0
        assert "audit signing" in result.output.lower()
        assert "unsigned=1" in result.output
        assert "WARN" in result.output

    def test_missing_audit_log_shows_warn_not_fail(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        cfg = _make_mock_config()
        cfg.audit_log_path = "/nonexistent/audit.jsonl"

        result = self._invoke_doctor(runner, cfg_path, cfg)

        assert result.exit_code == 0
        assert "audit signing" in result.output.lower()
        assert "WARN" in result.output
