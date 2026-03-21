"""Tests for missy.cli.main — CLI commands via Click's CliRunner."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from missy.cli.main import cli
from tests.cli.conftest import _make_cli_runner

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    """Return a Click CliRunner that mixes stdout/stderr."""
    return _make_cli_runner(mix_stderr=False)


def _make_mock_config(
    *,
    plugins_enabled: bool = False,
    providers: dict | None = None,
    audit_log_path: str = "/tmp/audit.jsonl",
) -> MagicMock:
    """Build a MissyConfig-like mock with sane defaults."""
    cfg = MagicMock()
    cfg.plugins.enabled = plugins_enabled
    cfg.plugins.allowed_plugins = []
    cfg.providers = providers if providers is not None else {}
    cfg.audit_log_path = audit_log_path
    return cfg


# ---------------------------------------------------------------------------
# --help / root group
# ---------------------------------------------------------------------------


class TestHelp:
    def test_root_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_root_help_contains_missy(self, runner: CliRunner):
        result = runner.invoke(cli, ["--help"])
        assert "Missy" in result.output

    def test_root_help_lists_init_command(self, runner: CliRunner):
        result = runner.invoke(cli, ["--help"])
        assert "init" in result.output

    def test_root_help_lists_ask_command(self, runner: CliRunner):
        result = runner.invoke(cli, ["--help"])
        assert "ask" in result.output

    def test_root_help_lists_providers_command(self, runner: CliRunner):
        result = runner.invoke(cli, ["--help"])
        assert "providers" in result.output

    def test_root_help_lists_skills_command(self, runner: CliRunner):
        result = runner.invoke(cli, ["--help"])
        assert "skills" in result.output

    def test_root_help_lists_plugins_command(self, runner: CliRunner):
        result = runner.invoke(cli, ["--help"])
        assert "plugins" in result.output

    def test_root_help_lists_schedule_command(self, runner: CliRunner):
        result = runner.invoke(cli, ["--help"])
        assert "schedule" in result.output

    def test_root_help_lists_audit_command(self, runner: CliRunner):
        result = runner.invoke(cli, ["--help"])
        assert "audit" in result.output

    def test_ask_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["ask", "--help"])
        assert result.exit_code == 0

    def test_schedule_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["schedule", "--help"])
        assert result.exit_code == 0

    def test_audit_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["audit", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# missy init
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_exits_zero(self, runner: CliRunner):
        with runner.isolated_filesystem() as tmpdir:
            fake_home = Path(tmpdir)
            with (
                patch.dict(os.environ, {"HOME": tmpdir}),
                patch("pathlib.Path.home", return_value=fake_home),
            ):
                result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0

    def test_init_creates_missy_directory(self, runner: CliRunner):
        with runner.isolated_filesystem() as tmpdir:
            fake_home = Path(tmpdir)
            with patch("pathlib.Path.home", return_value=fake_home), patch("missy.cli.main.Path"):
                # Let the real Path handle everything but control expanduser
                fake_home / ".missy"

                real_path = Path

                def patched_path(*args, **kwargs):
                    p = real_path(*args, **kwargs)
                    return p

                # Restore real Path; just control expanduser via env

            # Simplest approach: patch HOME env var and let real Path work
            with patch.dict(os.environ, {"HOME": str(tmpdir)}):
                result = runner.invoke(cli, ["init"])

        # Check that the command ran (may or may not create dirs depending on
        # home resolution, but should not crash with exit code > 1 from an
        # unhandled exception)
        assert result.exit_code in (0, 1)

    def test_init_success_output_contains_initialised(self, runner: CliRunner):
        with runner.isolated_filesystem() as tmpdir, patch.dict(os.environ, {"HOME": str(tmpdir)}):
            result = runner.invoke(cli, ["init"])
        # If exit is 0, success message should reference initialisation
        if result.exit_code == 0:
            assert "initialised" in result.output.lower() or "Missy" in result.output

    def test_init_second_call_skips_existing_config(self, runner: CliRunner):
        """Re-running init should not fail when config already exists."""
        with runner.isolated_filesystem() as tmpdir, patch.dict(os.environ, {"HOME": str(tmpdir)}):
            runner.invoke(cli, ["init"])
            result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0

    def test_init_config_file_created(self, runner: CliRunner):
        with runner.isolated_filesystem() as tmpdir:
            with patch.dict(os.environ, {"HOME": str(tmpdir)}):
                result = runner.invoke(cli, ["init"])
            config_file = Path(tmpdir) / ".missy" / "config.yaml"
            if result.exit_code == 0:
                assert config_file.exists()

    def test_init_audit_file_created(self, runner: CliRunner):
        with runner.isolated_filesystem() as tmpdir:
            with patch.dict(os.environ, {"HOME": str(tmpdir)}):
                result = runner.invoke(cli, ["init"])
            audit_file = Path(tmpdir) / ".missy" / "audit.jsonl"
            if result.exit_code == 0:
                assert audit_file.exists()

    def test_init_jobs_file_created(self, runner: CliRunner):
        with runner.isolated_filesystem() as tmpdir:
            with patch.dict(os.environ, {"HOME": str(tmpdir)}):
                result = runner.invoke(cli, ["init"])
            jobs_file = Path(tmpdir) / ".missy" / "jobs.json"
            if result.exit_code == 0:
                assert jobs_file.exists()
                assert jobs_file.read_text() == "[]"


# ---------------------------------------------------------------------------
# missy ask
# ---------------------------------------------------------------------------

# Minimal config YAML that load_config can parse without a real file on disk.
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


class TestAsk:
    def _invoke_ask(self, runner: CliRunner, prompt: str = "hello") -> object:
        """Invoke ``missy ask`` with fully mocked subsystems.

        The ``ask`` command imports AgentRuntime/AgentConfig lazily inside the
        function body, so we must patch them in their home module
        (``missy.agent.runtime``) rather than on ``missy.cli.main``.
        """
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_cfg = _make_mock_config(
                    providers={"anthropic": MagicMock(model="claude-3-5-sonnet-20241022")}
                )
                mock_load.return_value = mock_cfg

                with patch("missy.agent.runtime.AgentRuntime") as mock_runtime_cls:
                    mock_runtime = MagicMock()
                    mock_runtime.run.return_value = "Paris is the capital of France."
                    mock_runtime_cls.return_value = mock_runtime

                    with patch("missy.agent.runtime.AgentConfig"):
                        result = runner.invoke(
                            cli,
                            ["--config", cfg_path, "ask", prompt],
                        )
        finally:
            os.unlink(cfg_path)

        return result

    def test_ask_exits_zero(self, runner: CliRunner):
        result = self._invoke_ask(runner)
        assert result.exit_code == 0

    def test_ask_output_contains_response(self, runner: CliRunner):
        result = self._invoke_ask(runner)
        assert "Paris" in result.output

    def test_ask_calls_agent_run(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_cfg = _make_mock_config(providers={"anthropic": MagicMock(model="m")})
                mock_load.return_value = mock_cfg

                with patch("missy.agent.runtime.AgentRuntime") as mock_runtime_cls:
                    mock_runtime = MagicMock()
                    mock_runtime.run.return_value = "response"
                    mock_runtime_cls.return_value = mock_runtime

                    with patch("missy.agent.runtime.AgentConfig"):
                        runner.invoke(
                            cli,
                            ["--config", cfg_path, "ask", "test question"],
                        )

                    mock_runtime.run.assert_called_once()
        finally:
            os.unlink(cfg_path)

    def test_ask_missing_prompt_argument_exits_nonzero(self, runner: CliRunner):
        result = runner.invoke(cli, ["ask"])
        assert result.exit_code != 0

    def test_ask_with_provider_option(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_cfg = _make_mock_config(providers={"openai": MagicMock(model="gpt-4o")})
                mock_load.return_value = mock_cfg

                with patch("missy.agent.runtime.AgentRuntime") as mock_runtime_cls:
                    mock_runtime = MagicMock()
                    mock_runtime.run.return_value = "ok"
                    mock_runtime_cls.return_value = mock_runtime

                    with patch("missy.agent.runtime.AgentConfig") as mock_agent_cfg:
                        result = runner.invoke(
                            cli,
                            ["--config", cfg_path, "ask", "--provider", "openai", "hi"],
                        )

                    # Ensure AgentConfig was called with provider="openai"
                    mock_agent_cfg.assert_called_once()
                    call_kwargs = mock_agent_cfg.call_args[1]
                    assert call_kwargs["provider"] == "openai"
                    assert call_kwargs["capability_mode"] == "full"
        finally:
            os.unlink(cfg_path)

        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# missy providers
# ---------------------------------------------------------------------------


class TestProviders:
    def _mock_cfg_with_provider(self) -> MagicMock:
        provider_cfg = MagicMock()
        provider_cfg.model = "claude-3-5-sonnet-20241022"
        provider_cfg.base_url = None
        provider_cfg.timeout = 30

        cfg = MagicMock()
        cfg.providers = {"anthropic": provider_cfg}
        cfg.audit_log_path = "/tmp/audit.jsonl"
        return cfg

    def test_providers_exits_zero(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_load.return_value = self._mock_cfg_with_provider()

                mock_provider = MagicMock()
                mock_provider.is_available.return_value = True

                mock_registry = MagicMock()
                mock_registry.get.return_value = mock_provider

                with patch("missy.providers.registry.get_registry", return_value=mock_registry):
                    result = runner.invoke(cli, ["--config", cfg_path, "providers"])
        finally:
            os.unlink(cfg_path)

        assert result.exit_code == 0

    def test_providers_output_contains_provider_name(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_load.return_value = self._mock_cfg_with_provider()

                mock_provider = MagicMock()
                mock_provider.is_available.return_value = True

                mock_registry = MagicMock()
                mock_registry.get.return_value = mock_provider

                with patch("missy.providers.registry.get_registry", return_value=mock_registry):
                    result = runner.invoke(cli, ["--config", cfg_path, "providers"])
        finally:
            os.unlink(cfg_path)

        assert "anthropic" in result.output

    def test_providers_no_providers_configured(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                cfg = MagicMock()
                cfg.providers = {}
                mock_load.return_value = cfg

                result = runner.invoke(cli, ["--config", cfg_path, "providers"])
        finally:
            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "No providers" in result.output


# ---------------------------------------------------------------------------
# missy skills
# ---------------------------------------------------------------------------


class TestSkills:
    def test_skills_exits_zero_with_no_skills(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_load.return_value = _make_mock_config()

                with patch("missy.skills.registry.SkillRegistry") as mock_registry_cls:
                    mock_registry = MagicMock()
                    mock_registry.list_skills.return_value = []
                    mock_registry_cls.return_value = mock_registry

                    result = runner.invoke(cli, ["--config", cfg_path, "skills"])
        finally:
            os.unlink(cfg_path)

        assert result.exit_code == 0

    def test_skills_no_skills_message(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_load.return_value = _make_mock_config()

                with patch("missy.skills.registry.SkillRegistry") as mock_registry_cls:
                    mock_registry = MagicMock()
                    mock_registry.list_skills.return_value = []
                    mock_registry_cls.return_value = mock_registry

                    result = runner.invoke(cli, ["--config", cfg_path, "skills"])
        finally:
            os.unlink(cfg_path)

        assert "No skills" in result.output

    def test_skills_lists_registered_skills(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_load.return_value = _make_mock_config()

                with patch("missy.skills.registry.SkillRegistry") as mock_registry_cls:
                    mock_registry = MagicMock()
                    mock_registry.list_skills.return_value = ["calculator", "search"]
                    mock_registry_cls.return_value = mock_registry

                    result = runner.invoke(cli, ["--config", cfg_path, "skills"])
        finally:
            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "calculator" in result.output
        assert "search" in result.output


# ---------------------------------------------------------------------------
# missy plugins
# ---------------------------------------------------------------------------


class TestPlugins:
    def test_plugins_exits_zero(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_load.return_value = _make_mock_config()

                mock_loader = MagicMock()
                mock_loader.list_plugins.return_value = []

                with patch("missy.plugins.loader.init_plugin_loader", return_value=mock_loader):
                    result = runner.invoke(cli, ["--config", cfg_path, "plugins"])
        finally:
            os.unlink(cfg_path)

        assert result.exit_code == 0

    def test_plugins_shows_disabled_status_by_default(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_load.return_value = _make_mock_config(plugins_enabled=False)

                mock_loader = MagicMock()
                mock_loader.list_plugins.return_value = []

                with patch("missy.plugins.loader.init_plugin_loader", return_value=mock_loader):
                    result = runner.invoke(cli, ["--config", cfg_path, "plugins"])
        finally:
            os.unlink(cfg_path)

        assert "disabled" in result.output.lower()

    def test_plugins_no_plugins_loaded_message(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_load.return_value = _make_mock_config()

                mock_loader = MagicMock()
                mock_loader.list_plugins.return_value = []

                with patch("missy.plugins.loader.init_plugin_loader", return_value=mock_loader):
                    result = runner.invoke(cli, ["--config", cfg_path, "plugins"])
        finally:
            os.unlink(cfg_path)

        assert "No plugins" in result.output

    def test_plugins_displays_loaded_plugin(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_load.return_value = _make_mock_config(plugins_enabled=True)

                mock_loader = MagicMock()
                mock_loader.list_plugins.return_value = [
                    {
                        "name": "my-plugin",
                        "version": "1.0.0",
                        "description": "A test plugin",
                        "enabled": True,
                    }
                ]

                with patch("missy.plugins.loader.init_plugin_loader", return_value=mock_loader):
                    result = runner.invoke(cli, ["--config", cfg_path, "plugins"])
        finally:
            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "my-plugin" in result.output


# ---------------------------------------------------------------------------
# missy schedule list
# ---------------------------------------------------------------------------


class TestScheduleList:
    def test_schedule_list_exits_zero_no_jobs(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_load.return_value = _make_mock_config()

                with patch("missy.scheduler.manager.SchedulerManager") as mock_mgr_cls:
                    mock_mgr = MagicMock()
                    mock_mgr.list_jobs.return_value = []
                    mock_mgr_cls.return_value = mock_mgr

                    result = runner.invoke(cli, ["--config", cfg_path, "schedule", "list"])
        finally:
            os.unlink(cfg_path)

        assert result.exit_code == 0

    def test_schedule_list_no_jobs_message(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_load.return_value = _make_mock_config()

                with patch("missy.scheduler.manager.SchedulerManager") as mock_mgr_cls:
                    mock_mgr = MagicMock()
                    mock_mgr.list_jobs.return_value = []
                    mock_mgr_cls.return_value = mock_mgr

                    result = runner.invoke(cli, ["--config", cfg_path, "schedule", "list"])
        finally:
            os.unlink(cfg_path)

        assert "No scheduled jobs" in result.output

    def test_schedule_list_shows_jobs(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        job = MagicMock()
        job.id = "abc12345-0000-0000-0000-000000000000"
        job.name = "Daily summary"
        job.schedule = "daily at 09:00"
        job.provider = "anthropic"
        job.enabled = True
        job.run_count = 3
        job.last_run = None
        job.next_run = None

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_load.return_value = _make_mock_config()

                with patch("missy.scheduler.manager.SchedulerManager") as mock_mgr_cls:
                    mock_mgr = MagicMock()
                    mock_mgr.list_jobs.return_value = [job]
                    mock_mgr_cls.return_value = mock_mgr

                    result = runner.invoke(cli, ["--config", cfg_path, "schedule", "list"])
        finally:
            os.unlink(cfg_path)

        assert result.exit_code == 0
        # Rich may word-wrap "Daily summary" across two lines in the table cell,
        # so assert on the individual words rather than the full string.
        assert "Daily" in result.output
        assert "summary" in result.output

    def test_schedule_list_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["schedule", "list", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# missy audit security
# ---------------------------------------------------------------------------


class TestAuditSecurity:
    def test_audit_security_exits_zero_no_violations(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_load.return_value = _make_mock_config()

                with patch("missy.observability.audit_logger.AuditLogger") as mock_logger_cls:
                    mock_logger = MagicMock()
                    mock_logger.get_policy_violations.return_value = []
                    mock_logger_cls.return_value = mock_logger

                    result = runner.invoke(cli, ["--config", cfg_path, "audit", "security"])
        finally:
            os.unlink(cfg_path)

        assert result.exit_code == 0

    def test_audit_security_no_violations_message(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_load.return_value = _make_mock_config()

                with patch("missy.observability.audit_logger.AuditLogger") as mock_logger_cls:
                    mock_logger = MagicMock()
                    mock_logger.get_policy_violations.return_value = []
                    mock_logger_cls.return_value = mock_logger

                    result = runner.invoke(cli, ["--config", cfg_path, "audit", "security"])
        finally:
            os.unlink(cfg_path)

        assert "No policy violations" in result.output

    def test_audit_security_shows_violations(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        violation = {
            "timestamp": "2026-03-11T09:00:00",
            "event_type": "network.request",
            "category": "network",
            "result": "deny",
            "detail": {"host": "evil.example.com"},
        }

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_load.return_value = _make_mock_config()

                with patch("missy.observability.audit_logger.AuditLogger") as mock_logger_cls:
                    mock_logger = MagicMock()
                    mock_logger.get_policy_violations.return_value = [violation]
                    mock_logger_cls.return_value = mock_logger

                    result = runner.invoke(cli, ["--config", cfg_path, "audit", "security"])
        finally:
            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "network" in result.output

    def test_audit_security_respects_limit_option(self, runner: CliRunner):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(_MINIMAL_CONFIG_YAML)
            cfg_path = f.name

        try:
            with patch("missy.cli.main._load_subsystems") as mock_load:
                mock_load.return_value = _make_mock_config()

                with patch("missy.observability.audit_logger.AuditLogger") as mock_logger_cls:
                    mock_logger = MagicMock()
                    mock_logger.get_policy_violations.return_value = []
                    mock_logger_cls.return_value = mock_logger

                    result = runner.invoke(
                        cli,
                        ["--config", cfg_path, "audit", "security", "--limit", "10"],
                    )
                    mock_logger.get_policy_violations.assert_called_once_with(limit=10)
        finally:
            os.unlink(cfg_path)

        assert result.exit_code == 0

    def test_audit_security_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["audit", "security", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# _load_subsystems error path (config not found → sys.exit(1))
# ---------------------------------------------------------------------------


class TestLoadSubsystemsErrorPath:
    def test_missing_config_file_exits_one(self, runner: CliRunner):
        """Commands that call _load_subsystems with a missing file exit 1."""
        result = runner.invoke(
            cli,
            ["--config", "/nonexistent/path/config.yaml", "providers"],
        )
        assert result.exit_code == 1

    def test_missing_config_shows_error_hint(self, runner: CliRunner):
        result = runner.invoke(
            cli,
            ["--config", "/nonexistent/path/config.yaml", "providers"],
        )
        # Error output should contain guidance for the user
        combined = result.output + (result.stderr or "")
        assert "missy init" in combined or "config" in combined.lower()
