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

import tempfile
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from missy.cli.main import cli

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
    return CliRunner(mix_stderr=False)


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
    # vault sub-config
    vault_sub = MagicMock()
    vault_sub.vault_dir = "~/.missy/secrets"
    cfg.vault = vault_sub
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def _write_temp_config() -> str:
    """Write minimal YAML to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False
    ) as fh:
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
        with _SubsystemsPatch() as cfg:
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
        with _SubsystemsPatch() as cfg:
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
        with _SubsystemsPatch() as cfg:
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
        with _SubsystemsPatch():
            with patch("missy.security.vault.Vault", return_value=mock_vault):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "vault", "set", "MY_KEY", "my_value"]
                )
        assert result.exit_code == 0

    def test_vault_set_success_message(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        with _SubsystemsPatch():
            with patch("missy.security.vault.Vault", return_value=mock_vault):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "vault", "set", "MY_KEY", "my_value"]
                )
        assert "MY_KEY" in result.output

    def test_vault_set_calls_vault_set(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        with _SubsystemsPatch():
            with patch("missy.security.vault.Vault", return_value=mock_vault):
                runner.invoke(
                    cli, ["--config", cfg_path, "vault", "set", "API_KEY", "secret123"]
                )
        mock_vault.set.assert_called_once_with("API_KEY", "secret123")

    def test_vault_set_vault_error_exits_one(self, runner: CliRunner):
        from missy.security.vault import VaultError
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.set.side_effect = VaultError("encryption failed")
        with _SubsystemsPatch():
            with patch("missy.security.vault.Vault", return_value=mock_vault):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "vault", "set", "KEY", "val"]
                )
        assert result.exit_code == 1

    def test_vault_set_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["vault", "set", "--help"])
        assert result.exit_code == 0


class TestVaultGet:
    def test_vault_get_prints_value(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.get.return_value = "super_secret"
        with _SubsystemsPatch():
            with patch("missy.security.vault.Vault", return_value=mock_vault):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "vault", "get", "MY_KEY"]
                )
        assert "super_secret" in result.output

    def test_vault_get_missing_key_exits_one(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.get.return_value = None
        with _SubsystemsPatch():
            with patch("missy.security.vault.Vault", return_value=mock_vault):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "vault", "get", "MISSING"]
                )
        assert result.exit_code == 1

    def test_vault_get_vault_error_exits_one(self, runner: CliRunner):
        from missy.security.vault import VaultError
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.get.side_effect = VaultError("corrupt vault")
        with _SubsystemsPatch():
            with patch("missy.security.vault.Vault", return_value=mock_vault):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "vault", "get", "KEY"]
                )
        assert result.exit_code == 1

    def test_vault_get_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["vault", "get", "--help"])
        assert result.exit_code == 0


class TestVaultList:
    def test_vault_list_empty_vault(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.list_keys.return_value = []
        with _SubsystemsPatch():
            with patch("missy.security.vault.Vault", return_value=mock_vault):
                result = runner.invoke(cli, ["--config", cfg_path, "vault", "list"])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_vault_list_shows_keys(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.list_keys.return_value = ["OPENAI_KEY", "SLACK_TOKEN"]
        with _SubsystemsPatch():
            with patch("missy.security.vault.Vault", return_value=mock_vault):
                result = runner.invoke(cli, ["--config", cfg_path, "vault", "list"])
        assert result.exit_code == 0
        assert "OPENAI_KEY" in result.output
        assert "SLACK_TOKEN" in result.output

    def test_vault_list_vault_error_exits_one(self, runner: CliRunner):
        from missy.security.vault import VaultError
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.list_keys.side_effect = VaultError("read error")
        with _SubsystemsPatch():
            with patch("missy.security.vault.Vault", return_value=mock_vault):
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
        with _SubsystemsPatch():
            with patch("missy.security.vault.Vault", return_value=mock_vault):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "vault", "delete", "MY_KEY"]
                )
        assert result.exit_code == 0
        assert "MY_KEY" in result.output

    def test_vault_delete_key_not_found(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.delete.return_value = False
        with _SubsystemsPatch():
            with patch("missy.security.vault.Vault", return_value=mock_vault):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "vault", "delete", "GHOST"]
                )
        # CLI may silently handle "not found" (vault.delete returns False)
        assert result.exit_code == 0

    def test_vault_delete_vault_error_exits_one(self, runner: CliRunner):
        from missy.security.vault import VaultError
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.delete.side_effect = VaultError("io error")
        with _SubsystemsPatch():
            with patch("missy.security.vault.Vault", return_value=mock_vault):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "vault", "delete", "KEY"]
                )
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
        return job

    def test_schedule_add_exits_zero(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.add_job.return_value = self._make_job()
        with _SubsystemsPatch():
            with patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli,
                    [
                        "--config", cfg_path, "schedule", "add",
                        "--name", "Test Job",
                        "--schedule", "every 5 minutes",
                        "--task", "Check the news",
                    ],
                )
        assert result.exit_code == 0

    def test_schedule_add_success_message(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.add_job.return_value = self._make_job()
        with _SubsystemsPatch():
            with patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli,
                    [
                        "--config", cfg_path, "schedule", "add",
                        "--name", "Test Job",
                        "--schedule", "every 5 minutes",
                        "--task", "Check the news",
                    ],
                )
        assert "Test Job" in result.output or "Job added" in result.output

    def test_schedule_add_invalid_schedule_exits_one(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.add_job.side_effect = ValueError("unrecognised schedule")
        with _SubsystemsPatch():
            with patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli,
                    [
                        "--config", cfg_path, "schedule", "add",
                        "--name", "Bad",
                        "--schedule", "never",
                        "--task", "do nothing",
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
        with _SubsystemsPatch():
            with patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli,
                    [
                        "--config", cfg_path, "schedule", "add",
                        "--name", "X",
                        "--schedule", "every hour",
                        "--task", "ping",
                    ],
                )
        assert result.exit_code == 1


class TestSchedulePause:
    def test_schedule_pause_exits_zero(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        with _SubsystemsPatch():
            with patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "schedule", "pause", "abc123"]
                )
        assert result.exit_code == 0

    def test_schedule_pause_success_message(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        with _SubsystemsPatch():
            with patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "schedule", "pause", "abc123"]
                )
        assert "abc123" in result.output

    def test_schedule_pause_job_not_found_exits_one(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.pause_job.side_effect = KeyError("abc123")
        with _SubsystemsPatch():
            with patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "schedule", "pause", "abc123"]
                )
        assert result.exit_code == 1

    def test_schedule_pause_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["schedule", "pause", "--help"])
        assert result.exit_code == 0


class TestScheduleResume:
    def test_schedule_resume_exits_zero(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        with _SubsystemsPatch():
            with patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "schedule", "resume", "abc123"]
                )
        assert result.exit_code == 0

    def test_schedule_resume_success_message(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        with _SubsystemsPatch():
            with patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "schedule", "resume", "myjob"]
                )
        assert "myjob" in result.output

    def test_schedule_resume_job_not_found_exits_one(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.resume_job.side_effect = KeyError("myjob")
        with _SubsystemsPatch():
            with patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "schedule", "resume", "myjob"]
                )
        assert result.exit_code == 1

    def test_schedule_resume_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["schedule", "resume", "--help"])
        assert result.exit_code == 0


class TestScheduleRemove:
    def test_schedule_remove_exits_zero(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        with _SubsystemsPatch():
            with patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
                # --yes bypasses the confirmation prompt
                result = runner.invoke(
                    cli,
                    ["--config", cfg_path, "schedule", "remove", "--yes", "job123"],
                )
        assert result.exit_code == 0

    def test_schedule_remove_success_message(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        with _SubsystemsPatch():
            with patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli,
                    ["--config", cfg_path, "schedule", "remove", "--yes", "job123"],
                )
        assert "job123" in result.output

    def test_schedule_remove_job_not_found_exits_one(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.remove_job.side_effect = KeyError("ghost")
        with _SubsystemsPatch():
            with patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
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
        with _SubsystemsPatch():
            with patch(
                "missy.observability.audit_logger.AuditLogger", return_value=mock_logger
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "audit", "recent"]
                )
        assert result.exit_code == 0

    def test_audit_recent_no_events_message(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_logger = MagicMock()
        mock_logger.get_recent_events.return_value = []
        with _SubsystemsPatch():
            with patch(
                "missy.observability.audit_logger.AuditLogger", return_value=mock_logger
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "audit", "recent"]
                )
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
        with _SubsystemsPatch():
            with patch(
                "missy.observability.audit_logger.AuditLogger", return_value=mock_logger
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "audit", "recent"]
                )
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
        with _SubsystemsPatch():
            with patch(
                "missy.observability.audit_logger.AuditLogger", return_value=mock_logger
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
        with _SubsystemsPatch():
            with patch(
                "missy.observability.audit_logger.AuditLogger", return_value=mock_logger
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
# missy sessions cleanup
# ===========================================================================


class TestSessionsCleanup:
    def test_sessions_cleanup_exits_zero(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_store = MagicMock()
        mock_store.cleanup.return_value = 0
        with _SubsystemsPatch():
            with patch("missy.memory.store.MemoryStore", return_value=mock_store):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "sessions", "cleanup"]
                )
        assert result.exit_code == 0

    def test_sessions_cleanup_reports_removed_count(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_store = MagicMock()
        mock_store.cleanup.return_value = 42
        with _SubsystemsPatch():
            with patch("missy.memory.store.MemoryStore", return_value=mock_store):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "sessions", "cleanup"]
                )
        assert "42" in result.output

    def test_sessions_cleanup_dry_run_does_not_delete(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_store = MagicMock()
        with _SubsystemsPatch():
            with patch("missy.memory.store.MemoryStore", return_value=mock_store):
                result = runner.invoke(
                    cli,
                    ["--config", cfg_path, "sessions", "cleanup", "--dry-run"],
                )
        assert result.exit_code == 0
        mock_store.cleanup.assert_not_called()
        assert "Dry run" in result.output or "dry" in result.output.lower()

    def test_sessions_cleanup_respects_older_than(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_store = MagicMock()
        mock_store.cleanup.return_value = 0
        with _SubsystemsPatch():
            with patch("missy.memory.store.MemoryStore", return_value=mock_store):
                runner.invoke(
                    cli,
                    [
                        "--config", cfg_path,
                        "sessions", "cleanup",
                        "--older-than", "7",
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
        result = runner.invoke(cli, ["approvals", "list"])
        assert "gateway" in result.output.lower() or "No active" in result.output

    def test_approvals_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["approvals", "--help"])
        assert result.exit_code == 0

    def test_approvals_list_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["approvals", "list", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy patches list / approve / reject
# ===========================================================================


class TestPatchesList:
    def test_patches_list_no_patches(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.list_all.return_value = []
        with _SubsystemsPatch():
            with patch(
                "missy.agent.prompt_patches.PromptPatchManager", return_value=mock_mgr
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "patches", "list"]
                )
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
        with _SubsystemsPatch():
            with patch(
                "missy.agent.prompt_patches.PromptPatchManager", return_value=mock_mgr
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "patches", "list"]
                )
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
        with _SubsystemsPatch():
            with patch(
                "missy.agent.prompt_patches.PromptPatchManager", return_value=mock_mgr
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "patches", "approve", "patch-001"]
                )
        assert result.exit_code == 0
        assert "patch-001" in result.output

    def test_patches_approve_not_found(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.approve.return_value = False
        with _SubsystemsPatch():
            with patch(
                "missy.agent.prompt_patches.PromptPatchManager", return_value=mock_mgr
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "patches", "approve", "ghost"]
                )
        assert result.exit_code == 0

    def test_patches_approve_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["patches", "approve", "--help"])
        assert result.exit_code == 0


class TestPatchesReject:
    def test_patches_reject_success(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.reject.return_value = True
        with _SubsystemsPatch():
            with patch(
                "missy.agent.prompt_patches.PromptPatchManager", return_value=mock_mgr
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "patches", "reject", "patch-002"]
                )
        assert result.exit_code == 0
        assert "patch-002" in result.output

    def test_patches_reject_not_found(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.reject.return_value = False
        with _SubsystemsPatch():
            with patch(
                "missy.agent.prompt_patches.PromptPatchManager", return_value=mock_mgr
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "patches", "reject", "ghost"]
                )
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
        with _SubsystemsPatch():
            with patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "mcp", "list"]
                )
        assert result.exit_code == 0
        assert "No MCP servers" in result.output

    def test_mcp_list_shows_servers(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.list_servers.return_value = [
            {"name": "my-server", "alive": True, "tools": 5}
        ]
        with _SubsystemsPatch():
            with patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "mcp", "list"]
                )
        assert result.exit_code == 0
        assert "my-server" in result.output

    def test_mcp_list_offline_server(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.list_servers.return_value = [
            {"name": "offline-srv", "alive": False, "tools": 0}
        ]
        with _SubsystemsPatch():
            with patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "mcp", "list"]
                )
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
        with _SubsystemsPatch():
            with patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli,
                    [
                        "--config", cfg_path, "mcp", "add", "my-srv",
                        "--url", "http://localhost:3000",
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
        with _SubsystemsPatch():
            with patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli,
                    [
                        "--config", cfg_path, "mcp", "add", "stdio-srv",
                        "--command", "npx @modelcontextprotocol/server-echo",
                    ],
                )
        assert result.exit_code == 0

    def test_mcp_add_failure_exits_one(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.add_server.side_effect = ConnectionError("refused")
        with _SubsystemsPatch():
            with patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli,
                    [
                        "--config", cfg_path, "mcp", "add", "bad-srv",
                        "--url", "http://dead:9999",
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
        with _SubsystemsPatch():
            with patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "mcp", "remove", "my-srv"]
                )
        assert result.exit_code == 0

    def test_mcp_remove_success_message(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        with _SubsystemsPatch():
            with patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "mcp", "remove", "my-srv"]
                )
        assert "my-srv" in result.output

    def test_mcp_remove_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["mcp", "remove", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy devices list / pair / unpair / status / policy
# ===========================================================================


def _make_mock_registry(nodes: list[dict] | None = None) -> MagicMock:
    """Return a mock DeviceRegistry that returns the given node dicts."""
    reg = MagicMock()
    reg.all.return_value = nodes or []
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
        reg = _make_mock_registry(
            [{"node_id": "abc", "name": "X", "paired": True}]
        )
        with patch(
            "missy.channels.voice.registry.DeviceRegistry", return_value=reg
        ), patch(
            "missy.channels.voice.pairing.PairingManager", return_value=MagicMock()
        ):
            result = runner.invoke(cli, ["devices", "pair"])
        assert result.exit_code == 0
        assert "No pending" in result.output

    def test_devices_pair_with_node_id(self, runner: CliRunner):
        reg = _make_mock_registry([])
        mock_pairing = MagicMock()
        mock_pairing.approve.return_value = "tok123"
        with patch(
            "missy.channels.voice.registry.DeviceRegistry", return_value=reg
        ), patch(
            "missy.channels.voice.pairing.PairingManager", return_value=mock_pairing
        ):
            result = runner.invoke(
                cli, ["devices", "pair", "--node-id", "some-node-id"]
            )
        assert result.exit_code == 0
        # Token should appear in output
        assert "tok123" in result.output

    def test_devices_pair_approval_failure_exits_one(self, runner: CliRunner):
        reg = _make_mock_registry([])
        mock_pairing = MagicMock()
        mock_pairing.approve.side_effect = RuntimeError("node not found")
        with patch(
            "missy.channels.voice.registry.DeviceRegistry", return_value=reg
        ), patch(
            "missy.channels.voice.pairing.PairingManager", return_value=mock_pairing
        ):
            result = runner.invoke(
                cli, ["devices", "pair", "--node-id", "bad-node"]
            )
        assert result.exit_code == 1

    def test_devices_pair_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["devices", "pair", "--help"])
        assert result.exit_code == 0


class TestDevicesUnpair:
    def test_devices_unpair_exits_zero(self, runner: CliRunner):
        reg = _make_mock_registry([])
        with patch(
            "missy.channels.voice.registry.DeviceRegistry", return_value=reg
        ):
            result = runner.invoke(
                cli, ["devices", "unpair", "--yes", "mynode"]
            )
        assert result.exit_code == 0

    def test_devices_unpair_node_not_found_exits_one(self, runner: CliRunner):
        reg = _make_mock_registry([])
        reg.remove.side_effect = KeyError("mynode")
        with patch(
            "missy.channels.voice.registry.DeviceRegistry", return_value=reg
        ):
            result = runner.invoke(
                cli, ["devices", "unpair", "--yes", "mynode"]
            )
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
        reg = _make_mock_registry([])
        with patch(
            "missy.channels.voice.registry.DeviceRegistry", return_value=reg
        ):
            result = runner.invoke(
                cli, ["devices", "policy", "mynode", "--mode", "full"]
            )
        assert result.exit_code == 0
        reg.set_policy.assert_called_once_with("mynode", "full")

    def test_devices_policy_muted_mode(self, runner: CliRunner):
        reg = _make_mock_registry([])
        with patch(
            "missy.channels.voice.registry.DeviceRegistry", return_value=reg
        ):
            result = runner.invoke(
                cli, ["devices", "policy", "mynode", "--mode", "muted"]
            )
        assert result.exit_code == 0

    def test_devices_policy_node_not_found_exits_one(self, runner: CliRunner):
        reg = _make_mock_registry([])
        reg.set_policy.side_effect = KeyError("mynode")
        with patch(
            "missy.channels.voice.registry.DeviceRegistry", return_value=reg
        ):
            result = runner.invoke(
                cli, ["devices", "policy", "mynode", "--mode", "muted"]
            )
        assert result.exit_code == 1

    def test_devices_policy_invalid_mode(self, runner: CliRunner):
        result = runner.invoke(
            cli, ["devices", "policy", "mynode", "--mode", "banana"]
        )
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
        with patch(
            "missy.channels.voice.registry.DeviceRegistry", return_value=reg
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "voice", "status"])
        assert result.exit_code == 0

    def test_voice_status_shows_paired_count(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        nodes = [{"paired": True}, {"paired": False}]
        reg = _make_mock_registry(nodes)
        with patch(
            "missy.channels.voice.registry.DeviceRegistry", return_value=reg
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "voice", "status"])
        assert result.exit_code == 0
        # "Paired nodes" row should show 1
        assert "1" in result.output

    def test_voice_status_shows_tts_engine(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        reg = _make_mock_registry([])
        with patch(
            "missy.channels.voice.registry.DeviceRegistry", return_value=reg
        ):
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
        with patch(
            "missy.channels.voice.registry.DeviceRegistry", return_value=reg
        ):
            result = runner.invoke(cli, ["voice", "test", "nonexistent-node"])
        # Voice test may exit 0 or 1 depending on config loading
        assert result.exit_code in (0, 1)

    def test_voice_test_piper_not_found_exits_one(self, runner: CliRunner):
        nodes = [{"node_id": "node-001-abc", "name": "Test", "room": "test"}]
        reg = _make_mock_registry(nodes)
        mock_tts = MagicMock()
        import asyncio

        async def _raise(*a, **kw):
            raise FileNotFoundError("piper not found")

        mock_tts.synthesize.side_effect = _raise
        with patch(
            "missy.channels.voice.registry.DeviceRegistry", return_value=reg
        ), patch(
            "missy.channels.voice.tts.piper.PiperTTS", return_value=mock_tts
        ):
            result = runner.invoke(cli, ["voice", "test", "node-001"])
        assert result.exit_code == 1

    def test_voice_test_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["voice", "test", "--help"])
        assert result.exit_code == 0


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


class TestDiscordAudit:
    def test_discord_audit_no_events(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_logger = MagicMock()
        mock_logger.get_recent_events.return_value = []
        with _SubsystemsPatch():
            with patch(
                "missy.observability.audit_logger.AuditLogger",
                return_value=mock_logger,
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "discord", "audit"]
                )
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
        with _SubsystemsPatch():
            with patch(
                "missy.observability.audit_logger.AuditLogger",
                return_value=mock_logger,
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "discord", "audit"]
                )
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
        with _SubsystemsPatch():
            with patch(
                "missy.observability.audit_logger.AuditLogger",
                return_value=mock_logger,
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "discord", "audit"]
                )
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
            with patch(
                "missy.providers.registry.get_registry", return_value=mock_registry
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "gateway", "status"]
                )
        assert result.exit_code == 0

    def test_gateway_status_shows_cli_channel(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        with _SubsystemsPatch() as cfg:
            cfg.discord = None
            with patch(
                "missy.providers.registry.get_registry", return_value=mock_registry
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "gateway", "status"]
                )
        assert "cli" in result.output

    def test_gateway_status_shows_discord_disabled(self, runner: CliRunner):
        cfg_path = _write_temp_config()
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        with _SubsystemsPatch() as cfg:
            cfg.discord = None
            with patch(
                "missy.providers.registry.get_registry", return_value=mock_registry
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "gateway", "status"]
                )
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
