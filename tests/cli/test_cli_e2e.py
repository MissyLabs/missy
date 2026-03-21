"""Comprehensive end-to-end CLI integration tests for Missy.

Covers the following commands via Click's CliRunner:
  - missy init (first run and idempotent second run)
  - missy ask (mocked provider response)
  - missy providers list
  - missy skills / missy skills list
  - missy plugins
  - missy presets list
  - missy audit recent
  - missy audit security
  - missy doctor
  - missy hatch --non-interactive
  - missy persona show
  - missy cost
  - missy recover
  - missy sandbox status
  - missy config backups

All provider calls, subsystem initialisation, and file-system-sensitive
operations are mocked so no network activity or real config file on disk
is required.  Tests use isolated temporary directories via CliRunner and
patched HOME where needed to avoid touching the real ~/.missy directory.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from missy.cli.main import cli

# ---------------------------------------------------------------------------
# Module-level helpers and constants
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG_YAML = """\
config_version: 2

network:
  default_deny: true
  presets: []
  allowed_cidrs: []
  allowed_domains: []
  allowed_hosts: []

filesystem:
  allowed_write_paths: []
  allowed_read_paths: []

shell:
  enabled: false
  allowed_commands: []

plugins:
  enabled: false
  allowed_plugins: []

providers:
  anthropic:
    name: anthropic
    model: "claude-3-5-sonnet-20241022"
    timeout: 30

workspace_path: "/tmp/workspace"
audit_log_path: "/tmp/audit.jsonl"
max_spend_usd: 0.0
"""


def _write_temp_config(content: str = _MINIMAL_CONFIG_YAML) -> str:
    """Write *content* to a temporary YAML file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as fh:
        fh.write(content)
        return fh.name


def _make_mock_config(**kwargs: Any) -> MagicMock:
    """Return a MissyConfig-like MagicMock with sensible defaults.

    Pass keyword arguments to override individual attributes.
    """
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
    vault_sub = MagicMock()
    vault_sub.vault_dir = "~/.missy/secrets"
    cfg.vault = vault_sub
    container_sub = MagicMock()
    container_sub.enabled = False
    container_sub.image = "python:3.12-slim"
    container_sub.memory_limit = "256m"
    container_sub.cpu_quota = 0.5
    container_sub.network_mode = "none"
    cfg.container = container_sub
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    """Return a CliRunner with separate stdout/stderr streams."""
    return CliRunner(mix_stderr=False)


@pytest.fixture()
def mock_config() -> MagicMock:
    """Return a default mock config object."""
    return _make_mock_config()


@pytest.fixture()
def temp_config_path():
    """Write a minimal YAML config file and yield its path; clean up after."""
    path = _write_temp_config()
    yield path
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke_with_mock_subsystems(
    runner: CliRunner,
    args: list[str],
    cfg: MagicMock | None = None,
    extra_patches: list | None = None,
) -> Any:
    """Invoke *cli* with *args* after patching ``_load_subsystems``.

    Parameters
    ----------
    runner:
        The CliRunner to use.
    args:
        CLI argument list to pass to ``runner.invoke``.
    cfg:
        MagicMock config object. Defaults to ``_make_mock_config()``.
    extra_patches:
        List of ``unittest.mock.patch`` patch objects to apply in addition
        to ``_load_subsystems``.  Callers are responsible for starting /
        stopping them, or pass them as context managers.
    """
    mock_cfg = cfg or _make_mock_config()
    with patch("missy.cli.main._load_subsystems", return_value=mock_cfg):
        result = runner.invoke(cli, args)
    return result


# ===========================================================================
# 1. missy init
# ===========================================================================


class TestInit:
    """End-to-end tests for ``missy init``."""

    def test_init_exits_zero_in_fresh_directory(self, runner: CliRunner) -> None:
        """``missy init`` exits 0 in a directory with no pre-existing config."""
        with runner.isolated_filesystem() as tmpdir, patch.dict(os.environ, {"HOME": tmpdir}):
            result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0

    def test_init_creates_config_file(self, runner: CliRunner) -> None:
        """``missy init`` creates ``~/.missy/config.yaml``."""
        with runner.isolated_filesystem() as tmpdir:
            with patch.dict(os.environ, {"HOME": tmpdir}):
                result = runner.invoke(cli, ["init"])
            config_file = Path(tmpdir) / ".missy" / "config.yaml"
            if result.exit_code == 0:
                assert config_file.exists()

    def test_init_creates_audit_log(self, runner: CliRunner) -> None:
        """``missy init`` creates the audit log placeholder file."""
        with runner.isolated_filesystem() as tmpdir:
            with patch.dict(os.environ, {"HOME": tmpdir}):
                result = runner.invoke(cli, ["init"])
            if result.exit_code == 0:
                assert (Path(tmpdir) / ".missy" / "audit.jsonl").exists()

    def test_init_creates_jobs_file_with_empty_list(self, runner: CliRunner) -> None:
        """``missy init`` creates jobs.json containing ``[]``."""
        with runner.isolated_filesystem() as tmpdir:
            with patch.dict(os.environ, {"HOME": tmpdir}):
                result = runner.invoke(cli, ["init"])
            jobs_file = Path(tmpdir) / ".missy" / "jobs.json"
            if result.exit_code == 0:
                assert jobs_file.exists()
                assert jobs_file.read_text().strip() == "[]"

    def test_init_output_mentions_initialised(self, runner: CliRunner) -> None:
        """Success output contains an initialisation confirmation message."""
        with runner.isolated_filesystem() as tmpdir, patch.dict(os.environ, {"HOME": tmpdir}):
            result = runner.invoke(cli, ["init"])
        if result.exit_code == 0:
            lower = result.output.lower()
            assert "initialised" in lower or "missy" in lower

    def test_init_is_idempotent_on_second_run(self, runner: CliRunner) -> None:
        """Running ``missy init`` twice exits 0 both times."""
        with runner.isolated_filesystem() as tmpdir, patch.dict(os.environ, {"HOME": tmpdir}):
            first = runner.invoke(cli, ["init"])
            second = runner.invoke(cli, ["init"])
        assert first.exit_code == 0
        assert second.exit_code == 0

    def test_init_second_run_skips_existing_config(self, runner: CliRunner) -> None:
        """Second run reports that config already exists rather than overwriting."""
        with runner.isolated_filesystem() as tmpdir, patch.dict(os.environ, {"HOME": tmpdir}):
            runner.invoke(cli, ["init"])
            result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0
        assert "already exists" in result.output or "skipping" in result.output.lower()

    def test_init_creates_secrets_directory(self, runner: CliRunner) -> None:
        """``missy init`` creates the secrets subdirectory."""
        with runner.isolated_filesystem() as tmpdir:
            with patch.dict(os.environ, {"HOME": tmpdir}):
                result = runner.invoke(cli, ["init"])
            secrets_dir = Path(tmpdir) / ".missy" / "secrets"
            if result.exit_code == 0:
                assert secrets_dir.is_dir()


# ===========================================================================
# 2. missy ask
# ===========================================================================


class TestAsk:
    """End-to-end tests for ``missy ask``."""

    def _invoke_ask(
        self,
        runner: CliRunner,
        prompt: str = "What is the capital of France?",
        response_text: str = "The capital of France is Paris.",
    ):
        """Invoke ``missy ask`` with fully mocked agent runtime."""
        cfg_path = _write_temp_config()
        try:
            with (
                patch("missy.cli.main._load_subsystems") as mock_load,
                patch("missy.agent.runtime.AgentRuntime") as mock_rt_cls,
                patch("missy.agent.runtime.AgentConfig"),
                patch("missy.agent.hatching.HatchingManager") as mock_hatch,
            ):
                mock_cfg = _make_mock_config(
                    providers={"anthropic": MagicMock(model="claude-3-5-sonnet-20241022")}
                )
                mock_load.return_value = mock_cfg
                mock_rt = MagicMock()
                mock_rt.run.return_value = response_text
                mock_rt_cls.return_value = mock_rt
                mock_hatch.return_value.needs_hatching.return_value = False
                result = runner.invoke(cli, ["--config", cfg_path, "ask", prompt])
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(cfg_path)
        return result

    def test_ask_exits_zero(self, runner: CliRunner) -> None:
        """``missy ask`` exits with code 0 when the provider succeeds."""
        result = self._invoke_ask(runner)
        assert result.exit_code == 0

    def test_ask_output_contains_response_text(self, runner: CliRunner) -> None:
        """The provider response is included in the command's output."""
        result = self._invoke_ask(runner, response_text="Paris is the capital of France.")
        assert "Paris" in result.output

    def test_ask_with_custom_prompt(self, runner: CliRunner) -> None:
        """A custom prompt string is accepted without error."""
        result = self._invoke_ask(runner, prompt="Hello Missy", response_text="Hello!")
        assert result.exit_code == 0

    def test_ask_provider_error_exits_nonzero(self, runner: CliRunner) -> None:
        """A ``ProviderError`` from the agent results in a non-zero exit code."""
        from missy.core.exceptions import ProviderError

        cfg_path = _write_temp_config()
        try:
            with (
                patch("missy.cli.main._load_subsystems") as mock_load,
                patch("missy.agent.runtime.AgentRuntime") as mock_rt_cls,
                patch("missy.agent.runtime.AgentConfig"),
                patch("missy.agent.hatching.HatchingManager") as mock_hatch,
            ):
                mock_cfg = _make_mock_config(
                    providers={"anthropic": MagicMock(model="claude-3-5-sonnet-20241022")}
                )
                mock_load.return_value = mock_cfg
                mock_rt = MagicMock()
                mock_rt.run.side_effect = ProviderError("API unavailable")
                mock_rt_cls.return_value = mock_rt
                mock_hatch.return_value.needs_hatching.return_value = False
                result = runner.invoke(cli, ["--config", cfg_path, "ask", "hello"])
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(cfg_path)

        assert result.exit_code != 0

    def test_ask_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["ask", "--help"])
        assert result.exit_code == 0

    def test_ask_help_shows_prompt_argument(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["ask", "--help"])
        assert "PROMPT" in result.output.upper() or "prompt" in result.output.lower()


# ===========================================================================
# 3. missy providers list
# ===========================================================================


class TestProvidersList:
    """End-to-end tests for ``missy providers list``."""

    def test_providers_list_exits_zero(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """``missy providers list`` exits 0."""
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        mock_registry.get.return_value = None

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
        ):
            result = runner.invoke(cli, ["providers", "list"])

        assert result.exit_code == 0

    def test_providers_list_no_providers_message(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """When no providers are configured the output says so."""
        mock_config.providers = {}
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
        ):
            result = runner.invoke(cli, ["providers", "list"])

        assert result.exit_code == 0
        assert "no providers" in result.output.lower()

    def test_providers_list_shows_configured_provider(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """A configured provider name appears in the table output."""
        provider_cfg = MagicMock()
        provider_cfg.model = "claude-3-5-sonnet-20241022"
        provider_cfg.base_url = None
        provider_cfg.timeout = 30
        mock_config.providers = {"anthropic": provider_cfg}

        mock_provider = MagicMock()
        mock_provider.is_available.return_value = False

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_provider

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
        ):
            result = runner.invoke(cli, ["providers", "list"])

        assert result.exit_code == 0
        assert "anthropic" in result.output

    def test_providers_bare_group_also_lists(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """``missy providers`` (without sub-command) also lists providers."""
        mock_config.providers = {}
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
        ):
            result = runner.invoke(cli, ["providers"])

        assert result.exit_code == 0

    def test_providers_list_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["providers", "list", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# 4. missy skills
# ===========================================================================


class TestSkills:
    """End-to-end tests for ``missy skills`` and ``missy skills list``."""

    def test_skills_list_exits_zero_when_empty(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """``missy skills list`` exits 0 when no skills are registered."""
        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.skills.registry.SkillRegistry", return_value=mock_registry),
        ):
            result = runner.invoke(cli, ["skills", "list"])

        assert result.exit_code == 0

    def test_skills_list_shows_empty_message(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """When no skills are registered a descriptive message is printed."""
        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.skills.registry.SkillRegistry", return_value=mock_registry),
        ):
            result = runner.invoke(cli, ["skills", "list"])

        assert "no skills" in result.output.lower() or "not currently" in result.output.lower()

    def test_skills_bare_group_also_lists(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """``missy skills`` (no sub-command) delegates to list."""
        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.skills.registry.SkillRegistry", return_value=mock_registry),
        ):
            result = runner.invoke(cli, ["skills"])

        assert result.exit_code == 0

    def test_skills_list_shows_skill_names(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """Skill names returned by the registry appear in the table."""
        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = ["greet", "summarise"]

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.skills.registry.SkillRegistry", return_value=mock_registry),
        ):
            result = runner.invoke(cli, ["skills", "list"])

        assert result.exit_code == 0
        assert "greet" in result.output
        assert "summarise" in result.output

    def test_skills_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["skills", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# 5. missy plugins
# ===========================================================================


class TestPlugins:
    """End-to-end tests for ``missy plugins``."""

    def test_plugins_exits_zero(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """``missy plugins`` exits 0."""
        mock_loader = MagicMock()
        mock_loader.list_plugins.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.plugins.loader.init_plugin_loader", return_value=mock_loader),
        ):
            result = runner.invoke(cli, ["plugins"])

        assert result.exit_code == 0

    def test_plugins_shows_disabled_status_by_default(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """The 'disabled' label is present when plugins are not enabled."""
        mock_config.plugins.enabled = False
        mock_loader = MagicMock()
        mock_loader.list_plugins.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.plugins.loader.init_plugin_loader", return_value=mock_loader),
        ):
            result = runner.invoke(cli, ["plugins"])

        assert result.exit_code == 0
        assert "disabled" in result.output.lower()

    def test_plugins_shows_no_plugins_message(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """When no plugins are loaded a descriptive message is shown."""
        mock_loader = MagicMock()
        mock_loader.list_plugins.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.plugins.loader.init_plugin_loader", return_value=mock_loader),
        ):
            result = runner.invoke(cli, ["plugins"])

        assert "no plugins" in result.output.lower()

    def test_plugins_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["plugins", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# 6. missy presets list
# ===========================================================================


class TestPresetsList:
    """End-to-end tests for ``missy presets list``."""

    def test_presets_list_exits_zero(self, runner: CliRunner) -> None:
        """``missy presets list`` exits 0 without any mocking (no config needed)."""
        result = runner.invoke(cli, ["presets", "list"])
        assert result.exit_code == 0

    def test_presets_list_shows_anthropic_preset(self, runner: CliRunner) -> None:
        """The ``anthropic`` preset is included in the output table."""
        result = runner.invoke(cli, ["presets", "list"])
        assert "anthropic" in result.output.lower()

    def test_presets_list_shows_github_preset(self, runner: CliRunner) -> None:
        """The ``github`` preset is included in the output table."""
        result = runner.invoke(cli, ["presets", "list"])
        assert "github" in result.output.lower()

    def test_presets_list_output_contains_table_columns(self, runner: CliRunner) -> None:
        """The output table contains the expected column headers."""
        result = runner.invoke(cli, ["presets", "list"])
        assert result.exit_code == 0
        # At least one of the column headers or table title is present
        assert any(
            token in result.output for token in ("Hosts", "Domains", "CIDRs", "Presets", "Name")
        )

    def test_presets_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["presets", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# 7. missy audit recent
# ===========================================================================


class TestAuditRecent:
    """End-to-end tests for ``missy audit recent``."""

    def test_audit_recent_exits_zero_when_empty(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """``missy audit recent`` exits 0 when the audit log has no events."""
        mock_al = MagicMock()
        mock_al.get_recent_events.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.observability.audit_logger.AuditLogger", return_value=mock_al),
        ):
            result = runner.invoke(cli, ["audit", "recent"])

        assert result.exit_code == 0

    def test_audit_recent_shows_no_events_message(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """An empty audit log produces a 'no events' message."""
        mock_al = MagicMock()
        mock_al.get_recent_events.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.observability.audit_logger.AuditLogger", return_value=mock_al),
        ):
            result = runner.invoke(cli, ["audit", "recent"])

        assert "no audit events" in result.output.lower()

    def test_audit_recent_renders_events(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """When events are present they appear in the table output."""
        event = {
            "timestamp": "2026-03-18T12:00:00",
            "event_type": "network.request",
            "category": "network",
            "result": "allow",
            "detail": {"url": "https://api.anthropic.com"},
        }
        mock_al = MagicMock()
        mock_al.get_recent_events.return_value = [event]

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.observability.audit_logger.AuditLogger", return_value=mock_al),
        ):
            result = runner.invoke(cli, ["audit", "recent"])

        assert result.exit_code == 0
        assert "network" in result.output

    def test_audit_recent_respects_limit_option(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """The ``--limit`` option is forwarded without error."""
        mock_al = MagicMock()
        mock_al.get_recent_events.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.observability.audit_logger.AuditLogger", return_value=mock_al),
        ):
            result = runner.invoke(cli, ["audit", "recent", "--limit", "10"])

        assert result.exit_code == 0

    def test_audit_recent_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["audit", "recent", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# 8. missy audit security
# ===========================================================================


class TestAuditSecurity:
    """End-to-end tests for ``missy audit security``."""

    def test_audit_security_exits_zero_when_empty(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """``missy audit security`` exits 0 when there are no violations."""
        mock_al = MagicMock()
        mock_al.get_policy_violations.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.observability.audit_logger.AuditLogger", return_value=mock_al),
        ):
            result = runner.invoke(cli, ["audit", "security"])

        assert result.exit_code == 0

    def test_audit_security_shows_no_violations_message(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """An audit log with no violations prints a descriptive message."""
        mock_al = MagicMock()
        mock_al.get_policy_violations.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.observability.audit_logger.AuditLogger", return_value=mock_al),
        ):
            result = runner.invoke(cli, ["audit", "security"])

        assert "no policy violations" in result.output.lower()

    def test_audit_security_renders_violations(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Violations in the log are rendered as a table row."""
        violation = {
            "timestamp": "2026-03-18T12:00:00",
            "event_type": "network.policy_deny",
            "category": "network",
            "result": "deny",
            "detail": {"url": "https://evil.example.com"},
        }
        mock_al = MagicMock()
        mock_al.get_policy_violations.return_value = [violation]

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.observability.audit_logger.AuditLogger", return_value=mock_al),
        ):
            result = runner.invoke(cli, ["audit", "security"])

        assert result.exit_code == 0
        assert "network" in result.output

    def test_audit_security_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["audit", "security", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# 9. missy doctor
# ===========================================================================


class TestDoctor:
    """End-to-end tests for ``missy doctor``."""

    def test_doctor_exits_zero(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """``missy doctor`` exits 0 under nominal conditions."""
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        mock_registry.get.return_value = None

        mock_scheduler_mgr = MagicMock()
        mock_scheduler_mgr.list_jobs.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler_mgr),
        ):
            result = runner.invoke(cli, ["doctor"])

        assert result.exit_code == 0

    def test_doctor_output_contains_missy_doctor_title(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """The doctor output includes a recognisable table title or heading."""
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        mock_registry.get.return_value = None

        mock_scheduler_mgr = MagicMock()
        mock_scheduler_mgr.list_jobs.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler_mgr),
        ):
            result = runner.invoke(cli, ["doctor"])

        assert "doctor" in result.output.lower() or "missy" in result.output.lower()

    def test_doctor_checks_config_loaded(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """The doctor output includes a 'config loaded' check row."""
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        mock_registry.get.return_value = None

        mock_scheduler_mgr = MagicMock()
        mock_scheduler_mgr.list_jobs.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler_mgr),
        ):
            result = runner.invoke(cli, ["doctor"])

        assert "config" in result.output.lower()

    def test_doctor_shows_network_policy_status(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """The doctor output includes the network policy check."""
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        mock_registry.get.return_value = None

        mock_scheduler_mgr = MagicMock()
        mock_scheduler_mgr.list_jobs.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler_mgr),
        ):
            result = runner.invoke(cli, ["doctor"])

        assert "network" in result.output.lower()

    def test_doctor_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["doctor", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# 10. missy hatch --non-interactive
# ===========================================================================


class TestHatchNonInteractive:
    """End-to-end tests for ``missy hatch --non-interactive``."""

    def test_hatch_non_interactive_exits_zero_on_success(self, runner: CliRunner) -> None:
        """``missy hatch --non-interactive`` exits 0 when hatching succeeds."""
        from missy.agent.hatching import HatchingState, HatchingStatus

        mock_mgr = MagicMock()
        mock_mgr.is_hatched.return_value = False

        state = HatchingState(
            status=HatchingStatus.HATCHED,
            completed_at="2026-03-18T12:00:00+00:00",
            steps_completed=["validate_environment", "finalize"],
            persona_generated=True,
            memory_seeded=True,
        )
        mock_mgr.run_hatching.return_value = state

        with patch("missy.agent.hatching.HatchingManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["hatch", "--non-interactive"])

        assert result.exit_code == 0

    def test_hatch_non_interactive_success_message(self, runner: CliRunner) -> None:
        """Successful hatching prints a success message."""
        from missy.agent.hatching import HatchingState, HatchingStatus

        mock_mgr = MagicMock()
        mock_mgr.is_hatched.return_value = False

        state = HatchingState(
            status=HatchingStatus.HATCHED,
            completed_at="2026-03-18T12:00:00+00:00",
            steps_completed=["validate_environment", "finalize"],
            persona_generated=True,
            memory_seeded=True,
        )
        mock_mgr.run_hatching.return_value = state

        with patch("missy.agent.hatching.HatchingManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["hatch", "--non-interactive"])

        assert "hatched" in result.output.lower()

    def test_hatch_already_hatched_exits_zero(self, runner: CliRunner) -> None:
        """When Missy is already hatched the command exits 0."""
        from missy.agent.hatching import HatchingState, HatchingStatus

        mock_mgr = MagicMock()
        mock_mgr.is_hatched.return_value = True

        state = HatchingState(
            status=HatchingStatus.HATCHED,
            completed_at="2026-03-18T12:00:00+00:00",
            steps_completed=["validate_environment", "finalize"],
            persona_generated=True,
            memory_seeded=True,
        )
        mock_mgr.get_state.return_value = state

        with patch("missy.agent.hatching.HatchingManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["hatch", "--non-interactive"])

        assert result.exit_code == 0

    def test_hatch_already_hatched_shows_already_message(self, runner: CliRunner) -> None:
        """The 'already hatched' path prints an informational message."""
        from missy.agent.hatching import HatchingState, HatchingStatus

        mock_mgr = MagicMock()
        mock_mgr.is_hatched.return_value = True

        state = HatchingState(
            status=HatchingStatus.HATCHED,
            completed_at="2026-03-18T12:00:00+00:00",
            steps_completed=["validate_environment"],
            persona_generated=True,
            memory_seeded=True,
        )
        mock_mgr.get_state.return_value = state

        with patch("missy.agent.hatching.HatchingManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["hatch"])

        assert "already hatched" in result.output.lower()

    def test_hatch_failed_status_exits_nonzero(self, runner: CliRunner) -> None:
        """A ``FAILED`` hatching status results in a non-zero exit code."""
        from missy.agent.hatching import HatchingState, HatchingStatus

        mock_mgr = MagicMock()
        mock_mgr.is_hatched.return_value = False

        state = HatchingState(
            status=HatchingStatus.FAILED,
            completed_at=None,
            steps_completed=[],
            persona_generated=False,
            memory_seeded=False,
            error="Environment check failed",
        )
        mock_mgr.run_hatching.return_value = state

        with patch("missy.agent.hatching.HatchingManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["hatch", "--non-interactive"])

        assert result.exit_code != 0

    def test_hatch_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["hatch", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# 11. missy persona show
# ===========================================================================


class TestPersonaShow:
    """End-to-end tests for ``missy persona show``."""

    def test_persona_show_exits_zero(self, runner: CliRunner) -> None:
        """``missy persona show`` exits 0."""
        from missy.agent.persona import PersonaConfig

        mock_mgr = MagicMock()
        mock_mgr.get_persona.return_value = PersonaConfig(version=1)

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "show"])

        assert result.exit_code == 0

    def test_persona_show_contains_persona_name(self, runner: CliRunner) -> None:
        """The persona name is present in the output table."""
        from missy.agent.persona import PersonaConfig

        persona_cfg = PersonaConfig(name="Missy", version=1)
        mock_mgr = MagicMock()
        mock_mgr.get_persona.return_value = persona_cfg

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "show"])

        assert result.exit_code == 0
        assert "Missy" in result.output

    def test_persona_show_contains_tone_field(self, runner: CliRunner) -> None:
        """The output table includes the 'Tone' field."""
        from missy.agent.persona import PersonaConfig

        mock_mgr = MagicMock()
        mock_mgr.get_persona.return_value = PersonaConfig(version=1)

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "show"])

        assert "Tone" in result.output or "tone" in result.output.lower()

    def test_persona_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["persona", "--help"])
        assert result.exit_code == 0

    def test_persona_show_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["persona", "show", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# 12. missy cost
# ===========================================================================


class TestCost:
    """End-to-end tests for ``missy cost``."""

    def test_cost_exits_zero(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """``missy cost`` exits 0."""
        with patch("missy.cli.main._load_subsystems", return_value=mock_config):
            result = runner.invoke(cli, ["cost"])
        assert result.exit_code == 0

    def test_cost_shows_unlimited_when_budget_is_zero(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """When max_spend_usd is 0 the output says 'unlimited'."""
        mock_config.max_spend_usd = 0.0

        with patch("missy.cli.main._load_subsystems", return_value=mock_config):
            result = runner.invoke(cli, ["cost"])

        assert result.exit_code == 0
        assert "unlimited" in result.output.lower()

    def test_cost_shows_configured_budget(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """When max_spend_usd is set the formatted value appears in the output."""
        mock_config.max_spend_usd = 10.0

        with patch("missy.cli.main._load_subsystems", return_value=mock_config):
            result = runner.invoke(cli, ["cost"])

        assert result.exit_code == 0
        assert "$10.00" in result.output

    def test_cost_output_mentions_config_yaml(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """The output instructs the user how to set a budget via config.yaml."""
        with patch("missy.cli.main._load_subsystems", return_value=mock_config):
            result = runner.invoke(cli, ["cost"])

        assert "config.yaml" in result.output or "max_spend_usd" in result.output

    def test_cost_with_session_no_data(
        self, runner: CliRunner, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """``--session`` with no matching records prints 'No cost records'."""
        mock_store = MagicMock()
        mock_store.get_session_turns.return_value = []
        mock_store.get_session_costs.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
        ):
            result = runner.invoke(cli, ["cost", "--session", "nonexistent-session"])

        assert result.exit_code == 0
        assert "No cost records" in result.output

    def test_cost_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["cost", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# 13. missy recover
# ===========================================================================


class TestRecover:
    """End-to-end tests for ``missy recover``."""

    def test_recover_exits_zero_when_no_checkpoints(self, runner: CliRunner) -> None:
        """``missy recover`` exits 0 when no incomplete checkpoints exist."""
        with patch("missy.agent.checkpoint.scan_for_recovery", return_value=[]):
            result = runner.invoke(cli, ["recover"])
        assert result.exit_code == 0

    def test_recover_shows_no_checkpoints_message(self, runner: CliRunner) -> None:
        """An empty checkpoint list produces a 'no incomplete' message."""
        with patch("missy.agent.checkpoint.scan_for_recovery", return_value=[]):
            result = runner.invoke(cli, ["recover"])
        assert "no incomplete checkpoints" in result.output.lower()

    def test_recover_shows_checkpoint_data(self, runner: CliRunner) -> None:
        """Existing checkpoints are rendered in a table."""
        from missy.agent.checkpoint import RecoveryResult

        checkpoints = [
            RecoveryResult(
                checkpoint_id="aabbccdd-0001-0002-0003-000000000001",
                session_id="sess-xyz",
                prompt="Summarise this document",
                action="resume",
                iteration=2,
            ),
        ]
        with patch("missy.agent.checkpoint.scan_for_recovery", return_value=checkpoints):
            result = runner.invoke(cli, ["recover"])

        assert result.exit_code == 0
        assert "sess-xyz" in result.output or "Summarise" in result.output

    def test_recover_abandon_all_exits_zero(self, runner: CliRunner) -> None:
        """``missy recover --abandon-all`` exits 0 when abandonment succeeds."""
        mock_cm = MagicMock()
        mock_cm.abandon_old.return_value = 3

        with patch("missy.agent.checkpoint.CheckpointManager", return_value=mock_cm):
            result = runner.invoke(cli, ["recover", "--abandon-all"])

        assert result.exit_code == 0

    def test_recover_abandon_all_shows_count(self, runner: CliRunner) -> None:
        """``--abandon-all`` output mentions the number of abandoned checkpoints."""
        mock_cm = MagicMock()
        mock_cm.abandon_old.return_value = 5

        with patch("missy.agent.checkpoint.CheckpointManager", return_value=mock_cm):
            result = runner.invoke(cli, ["recover", "--abandon-all"])

        assert "5" in result.output or "Abandoned" in result.output

    def test_recover_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["recover", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# 14. missy sandbox status
# ===========================================================================


class TestSandboxStatus:
    """End-to-end tests for ``missy sandbox status``."""

    def test_sandbox_status_exits_zero_when_docker_unavailable(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """``missy sandbox status`` exits 0 even when Docker is not present."""
        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch(
                "missy.security.container.ContainerSandbox.is_available",
                return_value=False,
            ),
        ):
            result = runner.invoke(cli, ["sandbox", "status"])

        assert result.exit_code == 0

    def test_sandbox_status_shows_docker_row(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """The output table includes a 'Docker' row."""
        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch(
                "missy.security.container.ContainerSandbox.is_available",
                return_value=False,
            ),
        ):
            result = runner.invoke(cli, ["sandbox", "status"])

        assert result.exit_code == 0
        assert "Docker" in result.output or "docker" in result.output.lower()

    def test_sandbox_status_docker_available_shows_available(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """When Docker is available the output includes 'available'."""
        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch(
                "missy.security.container.ContainerSandbox.is_available",
                return_value=True,
            ),
        ):
            result = runner.invoke(cli, ["sandbox", "status"])

        assert result.exit_code == 0
        assert "available" in result.output.lower()

    def test_sandbox_status_shows_container_config(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Config values from the container section are present in the output."""
        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch(
                "missy.security.container.ContainerSandbox.is_available",
                return_value=False,
            ),
        ):
            result = runner.invoke(cli, ["sandbox", "status"])

        assert result.exit_code == 0
        # enabled, image, memory_limit, cpu_quota, or network_mode should appear
        assert any(
            token in result.output for token in ("enabled", "image", "memory", "cpu", "network")
        )

    def test_sandbox_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["sandbox", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# 15. missy config backups
# ===========================================================================


class TestConfigBackups:
    """End-to-end tests for ``missy config backups``."""

    def test_config_backups_exits_zero_when_empty(self, runner: CliRunner) -> None:
        """``missy config backups`` exits 0 when no backups exist."""
        with patch("missy.config.plan.list_backups", return_value=[]):
            result = runner.invoke(cli, ["config", "backups"])
        assert result.exit_code == 0

    def test_config_backups_shows_no_backups_message(self, runner: CliRunner) -> None:
        """An empty backup list produces a 'no backups' message."""
        with patch("missy.config.plan.list_backups", return_value=[]):
            result = runner.invoke(cli, ["config", "backups"])
        assert "no config backups" in result.output.lower()

    def test_config_backups_shows_backup_files(self, runner: CliRunner) -> None:
        """Backup files returned by list_backups appear in the table."""
        backup_path = MagicMock()
        backup_path.name = "config.yaml.20260318T120000"
        backup_path.stat.return_value = MagicMock(st_size=512)

        with patch("missy.config.plan.list_backups", return_value=[backup_path]):
            result = runner.invoke(cli, ["config", "backups"])

        assert result.exit_code == 0
        assert "config.yaml.20260318T120000" in result.output

    def test_config_backups_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["config", "backups", "--help"])
        assert result.exit_code == 0

    def test_config_backups_shows_file_sizes(self, runner: CliRunner) -> None:
        """The backup table includes file size information."""
        backup_path = MagicMock()
        backup_path.name = "config.yaml.20260318T120000"
        backup_path.stat.return_value = MagicMock(st_size=1024)

        with patch("missy.config.plan.list_backups", return_value=[backup_path]):
            result = runner.invoke(cli, ["config", "backups"])

        assert result.exit_code == 0
        assert "1,024" in result.output or "bytes" in result.output.lower()


# ===========================================================================
# Cross-cutting: --help flags for all tested commands
# ===========================================================================


class TestHelpFlags:
    """Verify that --help exits 0 and produces output for every tested group."""

    @pytest.mark.parametrize(
        "args",
        [
            ["--help"],
            ["init", "--help"],
            ["ask", "--help"],
            ["providers", "--help"],
            ["providers", "list", "--help"],
            ["skills", "--help"],
            ["skills", "list", "--help"],
            ["plugins", "--help"],
            ["presets", "--help"],
            ["presets", "list", "--help"],
            ["audit", "--help"],
            ["audit", "recent", "--help"],
            ["audit", "security", "--help"],
            ["doctor", "--help"],
            ["hatch", "--help"],
            ["persona", "--help"],
            ["persona", "show", "--help"],
            ["cost", "--help"],
            ["recover", "--help"],
            ["sandbox", "--help"],
            ["sandbox", "status", "--help"],
            ["config", "--help"],
            ["config", "backups", "--help"],
        ],
    )
    def test_help_exits_zero(self, runner: CliRunner, args: list[str]) -> None:
        result = runner.invoke(cli, args)
        assert result.exit_code == 0, (
            f"--help for {args!r} exited {result.exit_code}:\n{result.output}"
        )

    @pytest.mark.parametrize(
        "args",
        [
            ["--help"],
            ["audit", "--help"],
            ["providers", "--help"],
            ["skills", "--help"],
        ],
    )
    def test_help_output_is_nonempty(self, runner: CliRunner, args: list[str]) -> None:
        result = runner.invoke(cli, args)
        assert len(result.output.strip()) > 0
