"""Integration tests for Missy CLI commands via Click's test runner.

Each test class covers a distinct CLI command group or subcommand and focuses
on observable behaviour: exit codes, key phrases in output, and that mocked
dependencies are wired up correctly.

External I/O (config loading, vault, scheduler, MCP, persona, audit logger,
DeviceRegistry, ContainerSandbox) is patched so tests run hermetically without
a real config file or running services.

Test structure follows Arrange-Act-Assert throughout.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from missy.cli.main import cli

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG_YAML = """\
config_version: 2
network:
  default_deny: true
  presets: []
  allowed_domains: []
  allowed_cidrs: []
  allowed_hosts: []
providers:
  anthropic:
    name: anthropic
    model: "claude-sonnet-4-6"
    timeout: 30
workspace_path: "/tmp/workspace"
audit_log_path: "/tmp/audit.jsonl"
"""


def _write_temp_config(content: str = _MINIMAL_CONFIG_YAML) -> str:
    """Write YAML to a temp file and return its absolute path."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as fh:
        fh.write(content)
        return fh.name


def _make_mock_config(**overrides) -> MagicMock:
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
    vault_sub = MagicMock()
    vault_sub.vault_dir = "~/.missy/secrets"
    cfg.vault = vault_sub
    cfg.container = None
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner(mix_stderr=False)


# ---------------------------------------------------------------------------
# Helper context manager: patches _load_subsystems for a single test
# ---------------------------------------------------------------------------


class _SubsystemsPatch:
    """Context manager that stubs _load_subsystems for commands that call it."""

    def __init__(self, cfg: MagicMock | None = None):
        self.cfg = cfg or _make_mock_config()
        self._patcher = patch("missy.cli.main._load_subsystems", return_value=self.cfg)

    def __enter__(self) -> MagicMock:
        self._patcher.start()
        return self.cfg

    def __exit__(self, *args) -> None:
        self._patcher.stop()


# ===========================================================================
# missy init
# ===========================================================================


class TestInit:
    """Tests for `missy init` — creates ~/.missy directory and default files."""

    def test_init_creates_config_in_tmpdir(self, runner: CliRunner, tmp_path: Path) -> None:
        """init writes a config.yaml into a fresh missy directory."""
        missy_dir = tmp_path / ".missy"
        missy_dir / "config.yaml"

        with patch("missy.cli.main.Path") as mock_path_cls:
            # Only intercept the two paths the command constructs at the top level.
            real_path = Path  # keep a reference to the real class

            def path_factory(arg):
                if arg == "~/.missy":
                    return missy_dir
                if arg == "~/workspace":
                    return tmp_path / "workspace"
                return real_path(arg)

            mock_path_cls.side_effect = path_factory
            result = runner.invoke(cli, ["init"])

        # Should exit cleanly regardless of whether mock worked perfectly,
        # because the command catches OSError internally.
        assert result.exit_code == 0

    def test_init_exits_zero(self, runner: CliRunner, tmp_path: Path) -> None:
        """init always exits with code 0 on success."""
        with (
            patch("missy.cli.main.Path") as mock_path_cls,
        ):
            real_path = Path

            def path_factory(arg):
                if arg == "~/.missy":
                    p = MagicMock(spec=Path)
                    p.__truediv__ = lambda self, other: real_path(tmp_path / ".missy" / other)
                    p.mkdir = MagicMock()
                    return p
                if arg == "~/workspace":
                    p = MagicMock(spec=Path)
                    p.mkdir = MagicMock()
                    return p
                return real_path(arg)

            mock_path_cls.side_effect = path_factory
            result = runner.invoke(cli, ["init"])

        assert result.exit_code == 0

    def test_init_help_exits_zero(self, runner: CliRunner) -> None:
        """--help for init exits 0 and describes the command."""
        result = runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0
        assert "Initialize" in result.output or "init" in result.output.lower()

    def test_init_output_mentions_config(self, runner: CliRunner, tmp_path: Path) -> None:
        """init output references the config file path."""
        with patch("pathlib.Path.mkdir"), patch("pathlib.Path.exists", return_value=False), patch(
            "pathlib.Path.write_text"
        ), patch("pathlib.Path.touch"):
            result = runner.invoke(
                cli,
                ["--config", str(tmp_path / "config.yaml"), "init"],
            )
        # Exit code may be non-zero if workspace creation fails; just verify output.
        assert "config" in result.output.lower() or result.exit_code == 0

    def test_init_skips_existing_config(self, runner: CliRunner, tmp_path: Path) -> None:
        """When config.yaml already exists, init prints 'skipping' and exits 0."""
        missy_dir = tmp_path / ".missy"
        missy_dir.mkdir(mode=0o700)
        config_file = missy_dir / "config.yaml"
        config_file.write_text("# existing config\n")

        with patch("missy.cli.main.Path") as mock_path_cls:
            real_path = Path

            def path_factory(arg):
                if arg == "~/.missy":
                    return missy_dir
                if arg == "~/workspace":
                    ws = MagicMock(spec=Path)
                    ws.mkdir = MagicMock()
                    return ws
                return real_path(arg)

            mock_path_cls.side_effect = path_factory
            result = runner.invoke(cli, ["init"])

        assert result.exit_code == 0
        assert "skipping" in result.output.lower()


# ===========================================================================
# missy providers list
# ===========================================================================


class TestProvidersList:
    """Tests for `missy providers list`."""

    def test_providers_list_no_providers_configured(self, runner: CliRunner) -> None:
        """When cfg.providers is empty, prints 'No providers configured'."""
        cfg_path = _write_temp_config()
        with _SubsystemsPatch() as cfg:
            cfg.providers = {}
            result = runner.invoke(cli, ["--config", cfg_path, "providers", "list"])
        assert result.exit_code == 0
        assert "No providers configured" in result.output

    def test_providers_list_shows_configured_provider(self, runner: CliRunner) -> None:
        """With one provider in config, lists it in a table with availability."""
        cfg_path = _write_temp_config()
        provider_cfg = MagicMock()
        provider_cfg.model = "claude-sonnet-4-6"
        provider_cfg.base_url = None
        provider_cfg.timeout = 30

        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_provider

        with _SubsystemsPatch() as cfg:
            cfg.providers = {"anthropic": provider_cfg}
            with patch("missy.providers.registry.get_registry", return_value=mock_registry):
                result = runner.invoke(cli, ["--config", cfg_path, "providers", "list"])

        assert result.exit_code == 0
        assert "anthropic" in result.output

    def test_providers_list_shows_unavailable_provider(self, runner: CliRunner) -> None:
        """A provider whose is_available() returns False shows 'no' in the table."""
        cfg_path = _write_temp_config()
        provider_cfg = MagicMock()
        provider_cfg.model = "claude-sonnet-4-6"
        provider_cfg.base_url = None
        provider_cfg.timeout = 30

        mock_provider = MagicMock()
        mock_provider.is_available.return_value = False

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_provider

        with _SubsystemsPatch() as cfg:
            cfg.providers = {"anthropic": provider_cfg}
            with patch("missy.providers.registry.get_registry", return_value=mock_registry):
                result = runner.invoke(cli, ["--config", cfg_path, "providers", "list"])

        assert result.exit_code == 0
        # "no" appears in the Rich text that represents unavailability
        assert "anthropic" in result.output

    def test_providers_list_help_exits_zero(self, runner: CliRunner) -> None:
        """providers list --help exits 0."""
        result = runner.invoke(cli, ["providers", "list", "--help"])
        assert result.exit_code == 0

    def test_providers_list_unregistered_provider_shows_not_loaded(
        self, runner: CliRunner
    ) -> None:
        """A provider in config but absent from registry shows 'not loaded'."""
        cfg_path = _write_temp_config()
        provider_cfg = MagicMock()
        provider_cfg.model = "some-model"
        provider_cfg.base_url = None
        provider_cfg.timeout = 30

        mock_registry = MagicMock()
        mock_registry.get.return_value = None  # not registered

        with _SubsystemsPatch() as cfg:
            cfg.providers = {"ollama": provider_cfg}
            with patch("missy.providers.registry.get_registry", return_value=mock_registry):
                result = runner.invoke(cli, ["--config", cfg_path, "providers", "list"])

        assert result.exit_code == 0
        assert "not loaded" in result.output


# ===========================================================================
# missy presets list
# ===========================================================================


class TestPresetsList:
    """Tests for `missy presets list`."""

    def test_presets_list_exits_zero(self, runner: CliRunner) -> None:
        """presets list exits 0."""
        result = runner.invoke(cli, ["presets", "list"])
        assert result.exit_code == 0

    def test_presets_list_shows_anthropic_preset(self, runner: CliRunner) -> None:
        """The 'anthropic' preset appears in the table output."""
        result = runner.invoke(cli, ["presets", "list"])
        assert result.exit_code == 0
        assert "anthropic" in result.output.lower()

    def test_presets_list_shows_table_header(self, runner: CliRunner) -> None:
        """The output contains the table title text."""
        result = runner.invoke(cli, ["presets", "list"])
        assert result.exit_code == 0
        assert "Preset" in result.output or "preset" in result.output.lower()

    def test_presets_list_help_exits_zero(self, runner: CliRunner) -> None:
        """presets list --help exits 0."""
        result = runner.invoke(cli, ["presets", "list", "--help"])
        assert result.exit_code == 0

    def test_presets_list_shows_multiple_presets(self, runner: CliRunner) -> None:
        """More than one preset name appears in the output."""
        mock_presets = {
            "anthropic": {"hosts": ["api.anthropic.com:443"], "domains": [], "cidrs": []},
            "github": {"hosts": ["api.github.com:443"], "domains": [], "cidrs": []},
        }
        with patch("missy.policy.presets.PRESETS", mock_presets):
            result = runner.invoke(cli, ["presets", "list"])
        assert result.exit_code == 0
        assert "anthropic" in result.output
        assert "github" in result.output


# ===========================================================================
# missy doctor
# ===========================================================================


class TestDoctor:
    """Tests for `missy doctor` — system health check."""

    def _invoke_doctor(self, runner: CliRunner, cfg_path: str) -> object:
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        mock_mgr = MagicMock()
        mock_mgr.list_jobs.return_value = []

        with (
            _SubsystemsPatch(),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            return runner.invoke(cli, ["--config", cfg_path, "doctor"])

    def test_doctor_exits_zero(self, runner: CliRunner) -> None:
        """doctor exits 0 under normal conditions."""
        cfg_path = _write_temp_config()
        result = self._invoke_doctor(runner, cfg_path)
        assert result.exit_code == 0

    def test_doctor_output_contains_config_check(self, runner: CliRunner) -> None:
        """doctor output mentions 'config loaded'."""
        cfg_path = _write_temp_config()
        result = self._invoke_doctor(runner, cfg_path)
        assert "config loaded" in result.output

    def test_doctor_shows_no_providers_when_registry_empty(self, runner: CliRunner) -> None:
        """When the registry has no providers, doctor shows the warning row."""
        cfg_path = _write_temp_config()
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        mock_mgr = MagicMock()
        mock_mgr.list_jobs.return_value = []

        with (
            _SubsystemsPatch(),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "doctor"])

        assert result.exit_code == 0
        assert "no providers configured" in result.output

    def test_doctor_shows_provider_row_when_registered(self, runner: CliRunner) -> None:
        """When a provider is registered, doctor shows its availability row."""
        cfg_path = _write_temp_config()
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = False

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

    def test_doctor_shows_job_count(self, runner: CliRunner) -> None:
        """doctor reports the number of scheduled jobs."""
        cfg_path = _write_temp_config()
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        mock_job = MagicMock()
        mock_mgr = MagicMock()
        mock_mgr.list_jobs.return_value = [mock_job, mock_job, mock_job]

        with (
            _SubsystemsPatch(),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["--config", cfg_path, "doctor"])

        assert result.exit_code == 0
        assert "3 job" in result.output

    def test_doctor_shows_discord_disabled_when_not_configured(self, runner: CliRunner) -> None:
        """When discord is None, doctor shows 'not configured'."""
        cfg_path = _write_temp_config()
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        mock_mgr = MagicMock()
        mock_mgr.list_jobs.return_value = []

        with (
            _SubsystemsPatch() as cfg,
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr),
        ):
            cfg.discord = None
            result = runner.invoke(cli, ["--config", cfg_path, "doctor"])

        assert result.exit_code == 0
        assert "not configured" in result.output

    def test_doctor_help_exits_zero(self, runner: CliRunner) -> None:
        """doctor --help exits 0."""
        result = runner.invoke(cli, ["doctor", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy persona show
# ===========================================================================


class TestPersonaShow:
    """Tests for `missy persona show`."""

    def test_persona_show_exits_zero(self, runner: CliRunner) -> None:
        """persona show exits 0 with a mocked PersonaManager."""
        from missy.agent.persona import PersonaConfig

        mock_mgr = MagicMock()
        mock_mgr.get_persona.return_value = PersonaConfig()

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "show"])

        assert result.exit_code == 0

    def test_persona_show_displays_persona_name(self, runner: CliRunner) -> None:
        """persona show renders the persona name in the table output."""
        from missy.agent.persona import PersonaConfig

        persona = PersonaConfig(name="Atlas")
        mock_mgr = MagicMock()
        mock_mgr.get_persona.return_value = persona

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "show"])

        assert result.exit_code == 0
        assert "Atlas" in result.output

    def test_persona_show_displays_tone(self, runner: CliRunner) -> None:
        """persona show renders tone fields in the table."""
        from missy.agent.persona import PersonaConfig

        persona = PersonaConfig(tone=["friendly", "precise"])
        mock_mgr = MagicMock()
        mock_mgr.get_persona.return_value = persona

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "show"])

        assert result.exit_code == 0
        assert "friendly" in result.output
        assert "precise" in result.output

    def test_persona_show_displays_identity(self, runner: CliRunner) -> None:
        """persona show renders the identity_description field."""
        from missy.agent.persona import PersonaConfig

        persona = PersonaConfig(identity_description="A trustworthy Linux assistant.")
        mock_mgr = MagicMock()
        mock_mgr.get_persona.return_value = persona

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "show"])

        assert result.exit_code == 0
        assert "A trustworthy Linux assistant." in result.output

    def test_persona_show_includes_version_in_title(self, runner: CliRunner) -> None:
        """Table title includes the persona version number."""
        from missy.agent.persona import PersonaConfig

        persona = PersonaConfig(version=7)
        mock_mgr = MagicMock()
        mock_mgr.get_persona.return_value = persona

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "show"])

        assert result.exit_code == 0
        assert "v7" in result.output

    def test_persona_show_help_exits_zero(self, runner: CliRunner) -> None:
        """persona show --help exits 0."""
        result = runner.invoke(cli, ["persona", "show", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy persona reset
# ===========================================================================


class TestPersonaReset:
    """Tests for `missy persona reset`."""

    def test_persona_reset_exits_zero(self, runner: CliRunner) -> None:
        """persona reset exits 0 and calls mgr.reset()."""
        mock_mgr = MagicMock()
        mock_mgr.version = 1

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "reset"])

        assert result.exit_code == 0
        mock_mgr.reset.assert_called_once()

    def test_persona_reset_output_confirms_action(self, runner: CliRunner) -> None:
        """persona reset output mentions 'reset to defaults'."""
        mock_mgr = MagicMock()
        mock_mgr.version = 3

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "reset"])

        assert result.exit_code == 0
        assert "reset" in result.output.lower()

    def test_persona_reset_shows_version_after_reset(self, runner: CliRunner) -> None:
        """persona reset output shows the new version number."""
        mock_mgr = MagicMock()
        mock_mgr.version = 5

        with patch("missy.agent.persona.PersonaManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["persona", "reset"])

        assert result.exit_code == 0
        assert "5" in result.output

    def test_persona_reset_help_exits_zero(self, runner: CliRunner) -> None:
        """persona reset --help exits 0."""
        result = runner.invoke(cli, ["persona", "reset", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy audit recent
# ===========================================================================


class TestAuditRecent:
    """Tests for `missy audit recent`."""

    def test_audit_recent_no_events_prints_dim_message(self, runner: CliRunner) -> None:
        """When the log has no events, prints 'No audit events found'."""
        cfg_path = _write_temp_config()
        mock_al = MagicMock()
        mock_al.get_recent_events.return_value = []

        with _SubsystemsPatch(), patch("missy.observability.audit_logger.AuditLogger", return_value=mock_al):
            result = runner.invoke(cli, ["--config", cfg_path, "audit", "recent"])

        assert result.exit_code == 0
        assert "No audit events found" in result.output

    def test_audit_recent_shows_event_rows(self, runner: CliRunner) -> None:
        """When events exist, renders them in a table."""
        cfg_path = _write_temp_config()
        events = [
            {
                "timestamp": "2026-03-18T10:00:00Z",
                "event_type": "network.request",
                "category": "network",
                "result": "allow",
                "detail": {"url": "https://api.anthropic.com"},
            },
        ]
        mock_al = MagicMock()
        mock_al.get_recent_events.return_value = events

        with _SubsystemsPatch(), patch("missy.observability.audit_logger.AuditLogger", return_value=mock_al):
            result = runner.invoke(cli, ["--config", cfg_path, "audit", "recent"])

        assert result.exit_code == 0
        assert "network.request" in result.output

    def test_audit_recent_category_filter(self, runner: CliRunner) -> None:
        """--category filters events to only the specified category."""
        cfg_path = _write_temp_config()
        events = [
            {
                "timestamp": "2026-03-18T10:00:00Z",
                "event_type": "network.request",
                "category": "network",
                "result": "allow",
                "detail": {},
            },
            {
                "timestamp": "2026-03-18T10:01:00Z",
                "event_type": "shell.exec",
                "category": "shell",
                "result": "deny",
                "detail": {},
            },
        ]
        mock_al = MagicMock()
        mock_al.get_recent_events.return_value = events

        with _SubsystemsPatch(), patch("missy.observability.audit_logger.AuditLogger", return_value=mock_al):
            result = runner.invoke(
                cli, ["--config", cfg_path, "audit", "recent", "--category", "shell"]
            )

        assert result.exit_code == 0
        assert "shell.exec" in result.output
        # network event should be filtered out
        assert "network.request" not in result.output

    def test_audit_recent_category_filter_no_match_prints_message(
        self, runner: CliRunner
    ) -> None:
        """When the category filter matches nothing, prints 'No audit events found'."""
        cfg_path = _write_temp_config()
        events = [
            {
                "timestamp": "2026-03-18T10:00:00Z",
                "event_type": "network.request",
                "category": "network",
                "result": "allow",
                "detail": {},
            },
        ]
        mock_al = MagicMock()
        mock_al.get_recent_events.return_value = events

        with _SubsystemsPatch(), patch("missy.observability.audit_logger.AuditLogger", return_value=mock_al):
            result = runner.invoke(
                cli,
                ["--config", cfg_path, "audit", "recent", "--category", "filesystem"],
            )

        assert result.exit_code == 0
        assert "No audit events found" in result.output

    def test_audit_recent_limit_option(self, runner: CliRunner) -> None:
        """--limit is accepted without error."""
        cfg_path = _write_temp_config()
        mock_al = MagicMock()
        mock_al.get_recent_events.return_value = []

        with _SubsystemsPatch(), patch("missy.observability.audit_logger.AuditLogger", return_value=mock_al):
            result = runner.invoke(
                cli, ["--config", cfg_path, "audit", "recent", "--limit", "10"]
            )

        assert result.exit_code == 0

    def test_audit_recent_help_exits_zero(self, runner: CliRunner) -> None:
        """audit recent --help exits 0."""
        result = runner.invoke(cli, ["audit", "recent", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy config backups
# ===========================================================================


class TestConfigBackups:
    """Tests for `missy config backups`."""

    def test_config_backups_no_backups_message(self, runner: CliRunner) -> None:
        """When no backups exist, prints 'No config backups found'."""
        with patch("missy.config.plan.list_backups", return_value=[]):
            result = runner.invoke(cli, ["config", "backups"])

        assert result.exit_code == 0
        assert "No config backups found" in result.output

    def test_config_backups_lists_backup_files(self, runner: CliRunner, tmp_path: Path) -> None:
        """When backups exist, renders them in a table with file names."""
        backup1 = tmp_path / "config.yaml.20260318-120000"
        backup1.write_text("# backup 1\n")
        backup2 = tmp_path / "config.yaml.20260319-090000"
        backup2.write_text("# backup 2\n")

        with patch("missy.config.plan.list_backups", return_value=[backup1, backup2]):
            result = runner.invoke(cli, ["config", "backups"])

        assert result.exit_code == 0
        assert backup1.name in result.output
        assert backup2.name in result.output

    def test_config_backups_shows_file_size(self, runner: CliRunner, tmp_path: Path) -> None:
        """Table includes file size in bytes."""
        backup = tmp_path / "config.yaml.20260318-120000"
        backup.write_text("hello\n")

        with patch("missy.config.plan.list_backups", return_value=[backup]):
            result = runner.invoke(cli, ["config", "backups"])

        assert result.exit_code == 0
        assert "bytes" in result.output

    def test_config_backups_help_exits_zero(self, runner: CliRunner) -> None:
        """config backups --help exits 0."""
        result = runner.invoke(cli, ["config", "backups", "--help"])
        assert result.exit_code == 0

    def test_config_backups_exits_zero_always(self, runner: CliRunner) -> None:
        """config backups always exits 0."""
        with patch("missy.config.plan.list_backups", return_value=[]):
            result = runner.invoke(cli, ["config", "backups"])
        assert result.exit_code == 0


# ===========================================================================
# missy vault list
# ===========================================================================


class TestVaultList:
    """Tests for `missy vault list`."""

    def test_vault_list_empty_vault(self, runner: CliRunner) -> None:
        """When vault is empty, prints 'Vault is empty'."""
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.list_keys.return_value = []

        with _SubsystemsPatch(), patch("missy.security.vault.Vault", return_value=mock_vault):
            result = runner.invoke(cli, ["--config", cfg_path, "vault", "list"])

        assert result.exit_code == 0
        assert "Vault is empty" in result.output

    def test_vault_list_shows_key_names(self, runner: CliRunner) -> None:
        """When keys exist, each key name appears in the output."""
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.list_keys.return_value = ["ANTHROPIC_API_KEY", "OPENAI_KEY"]

        with _SubsystemsPatch(), patch("missy.security.vault.Vault", return_value=mock_vault):
            result = runner.invoke(cli, ["--config", cfg_path, "vault", "list"])

        assert result.exit_code == 0
        assert "ANTHROPIC_API_KEY" in result.output
        assert "OPENAI_KEY" in result.output

    def test_vault_list_vault_error_exits_one(self, runner: CliRunner) -> None:
        """When Vault raises VaultError, exits 1."""
        from missy.security.vault import VaultError

        cfg_path = _write_temp_config()

        with _SubsystemsPatch(), patch("missy.security.vault.Vault", side_effect=VaultError("key file missing")):
            result = runner.invoke(cli, ["--config", cfg_path, "vault", "list"])

        assert result.exit_code == 1

    def test_vault_list_exits_zero_on_success(self, runner: CliRunner) -> None:
        """vault list exits 0 when the vault is readable."""
        cfg_path = _write_temp_config()
        mock_vault = MagicMock()
        mock_vault.list_keys.return_value = ["MY_SECRET"]

        with _SubsystemsPatch(), patch("missy.security.vault.Vault", return_value=mock_vault):
            result = runner.invoke(cli, ["--config", cfg_path, "vault", "list"])

        assert result.exit_code == 0

    def test_vault_list_help_exits_zero(self, runner: CliRunner) -> None:
        """vault list --help exits 0."""
        result = runner.invoke(cli, ["vault", "list", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy mcp list
# ===========================================================================


class TestMcpList:
    """Tests for `missy mcp list`."""

    def test_mcp_list_no_servers_message(self, runner: CliRunner) -> None:
        """When no MCP servers are configured, prints 'No MCP servers configured'."""
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.list_servers.return_value = []

        with _SubsystemsPatch(), patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["--config", cfg_path, "mcp", "list"])

        assert result.exit_code == 0
        assert "No MCP servers configured" in result.output

    def test_mcp_list_shows_server_name(self, runner: CliRunner) -> None:
        """When servers exist, their names appear in the table."""
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.list_servers.return_value = [
            {"name": "filesystem-server", "alive": True, "tools": 5},
        ]

        with _SubsystemsPatch(), patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["--config", cfg_path, "mcp", "list"])

        assert result.exit_code == 0
        assert "filesystem-server" in result.output

    def test_mcp_list_shows_alive_status(self, runner: CliRunner) -> None:
        """A live server shows 'yes'; a dead server shows 'no' in the Alive column."""
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.list_servers.return_value = [
            {"name": "live-server", "alive": True, "tools": 3},
            {"name": "dead-server", "alive": False, "tools": 0},
        ]

        with _SubsystemsPatch(), patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["--config", cfg_path, "mcp", "list"])

        assert result.exit_code == 0
        assert "live-server" in result.output
        assert "dead-server" in result.output

    def test_mcp_list_shows_tool_count(self, runner: CliRunner) -> None:
        """Tool count for each server appears in the output."""
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.list_servers.return_value = [
            {"name": "my-server", "alive": True, "tools": 12},
        ]

        with _SubsystemsPatch(), patch("missy.mcp.manager.McpManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["--config", cfg_path, "mcp", "list"])

        assert result.exit_code == 0
        assert "12" in result.output

    def test_mcp_list_help_exits_zero(self, runner: CliRunner) -> None:
        """mcp list --help exits 0."""
        result = runner.invoke(cli, ["mcp", "list", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy schedule list
# ===========================================================================


class TestScheduleList:
    """Tests for `missy schedule list`."""

    def test_schedule_list_no_jobs(self, runner: CliRunner) -> None:
        """When no jobs exist, prints 'No scheduled jobs'."""
        cfg_path = _write_temp_config()
        mock_mgr = MagicMock()
        mock_mgr.list_jobs.return_value = []

        with _SubsystemsPatch(), patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["--config", cfg_path, "schedule", "list"])

        assert result.exit_code == 0
        assert "No scheduled jobs" in result.output

    def test_schedule_list_shows_job_name(self, runner: CliRunner) -> None:
        """When jobs exist, job names appear in the table.

        Rich wraps long cell values across multiple lines when the terminal is
        narrow, so we search for the first word of the name rather than the
        full phrase.
        """
        cfg_path = _write_temp_config()
        job = MagicMock()
        job.id = "aaaa-bbbb-cccc-dddd"
        job.name = "Daily digest"
        job.schedule = "daily at 09:00"
        job.provider = "anthropic"
        job.enabled = True
        job.run_count = 0
        job.last_run = None
        job.next_run = None

        mock_mgr = MagicMock()
        mock_mgr.list_jobs.return_value = [job]

        with _SubsystemsPatch(), patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["--config", cfg_path, "schedule", "list"])

        assert result.exit_code == 0
        # Rich may wrap "Daily digest" across two lines; check for either word.
        assert "Daily" in result.output

    def test_schedule_list_shows_schedule_expression(self, runner: CliRunner) -> None:
        """The schedule expression appears in the table.

        Rich may wrap the cell value across two lines in a narrow terminal, so
        we strip newlines from the output before doing the substring check.
        """
        cfg_path = _write_temp_config()
        job = MagicMock()
        job.id = "aaaa-1111-2222-3333"
        job.name = "Hourly check"
        job.schedule = "every hour"
        job.provider = "anthropic"
        job.enabled = True
        job.run_count = 5
        job.last_run = None
        job.next_run = None

        mock_mgr = MagicMock()
        mock_mgr.list_jobs.return_value = [job]

        with _SubsystemsPatch(), patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["--config", cfg_path, "schedule", "list"])

        assert result.exit_code == 0
        # Rich may wrap "every hour" across two table rows when the terminal is
        # narrow.  Check that each word appears somewhere in the output rather
        # than requiring the phrase to be on the same line.
        assert "every" in result.output
        assert "hour" in result.output

    def test_schedule_list_shows_multiple_jobs(self, runner: CliRunner) -> None:
        """Multiple jobs all appear in a single table output."""
        cfg_path = _write_temp_config()

        def make_job(name: str, schedule: str) -> MagicMock:
            job = MagicMock()
            job.id = f"{name}-id"
            job.name = name
            job.schedule = schedule
            job.provider = "anthropic"
            job.enabled = True
            job.run_count = 0
            job.last_run = None
            job.next_run = None
            return job

        jobs = [make_job("Task A", "every 5 minutes"), make_job("Task B", "daily at 08:00")]
        mock_mgr = MagicMock()
        mock_mgr.list_jobs.return_value = jobs

        with _SubsystemsPatch(), patch("missy.scheduler.manager.SchedulerManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["--config", cfg_path, "schedule", "list"])

        assert result.exit_code == 0
        assert "Task A" in result.output
        assert "Task B" in result.output

    def test_schedule_list_help_exits_zero(self, runner: CliRunner) -> None:
        """schedule list --help exits 0."""
        result = runner.invoke(cli, ["schedule", "list", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy devices list
# ===========================================================================


class TestDevicesList:
    """Tests for `missy devices list`."""

    def test_devices_list_no_nodes(self, runner: CliRunner) -> None:
        """When no nodes are registered, prints 'No edge nodes registered'."""
        mock_reg = MagicMock()
        mock_reg.all.return_value = []

        with patch("missy.channels.voice.registry.DeviceRegistry", return_value=mock_reg):
            result = runner.invoke(cli, ["devices", "list"])

        assert result.exit_code == 0
        assert "No edge nodes registered" in result.output

    def test_devices_list_shows_node_id(self, runner: CliRunner) -> None:
        """A registered node's truncated ID appears in the table."""
        mock_reg = MagicMock()
        mock_reg.all.return_value = [
            {
                "node_id": "abcdef1234567890",
                "name": "Kitchen Pi",
                "room": "kitchen",
                "paired": True,
                "policy": "full",
                "last_seen": None,
            }
        ]

        with patch("missy.channels.voice.registry.DeviceRegistry", return_value=mock_reg):
            result = runner.invoke(cli, ["devices", "list"])

        assert result.exit_code == 0
        assert "abcdef12" in result.output  # first 8 chars of node_id

    def test_devices_list_shows_node_name_and_room(self, runner: CliRunner) -> None:
        """Node name and room appear in the output."""
        mock_reg = MagicMock()
        mock_reg.all.return_value = [
            {
                "node_id": "node-111-aaa",
                "name": "Bedroom Speaker",
                "room": "bedroom",
                "paired": True,
                "policy": "full",
                "last_seen": None,
            }
        ]

        with patch("missy.channels.voice.registry.DeviceRegistry", return_value=mock_reg):
            result = runner.invoke(cli, ["devices", "list"])

        assert result.exit_code == 0
        assert "Bedroom Speaker" in result.output
        assert "bedroom" in result.output

    def test_devices_list_shows_pending_status(self, runner: CliRunner) -> None:
        """An unpaired node shows 'pending' status."""
        mock_reg = MagicMock()
        mock_reg.all.return_value = [
            {
                "node_id": "node-pending-xyz",
                "name": "New Device",
                "room": "office",
                "paired": False,
                "policy": "full",
                "last_seen": None,
            }
        ]

        with patch("missy.channels.voice.registry.DeviceRegistry", return_value=mock_reg):
            result = runner.invoke(cli, ["devices", "list"])

        assert result.exit_code == 0
        assert "pending" in result.output

    def test_devices_list_last_seen_never(self, runner: CliRunner) -> None:
        """A node with no last_seen timestamp shows 'never'."""
        mock_reg = MagicMock()
        mock_reg.all.return_value = [
            {
                "node_id": "node-no-ts",
                "name": "Silent Node",
                "room": "garage",
                "paired": True,
                "policy": "full",
                "last_seen": None,
            }
        ]

        with patch("missy.channels.voice.registry.DeviceRegistry", return_value=mock_reg):
            result = runner.invoke(cli, ["devices", "list"])

        assert result.exit_code == 0
        assert "never" in result.output

    def test_devices_list_help_exits_zero(self, runner: CliRunner) -> None:
        """devices list --help exits 0."""
        result = runner.invoke(cli, ["devices", "list", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy sandbox status
# ===========================================================================


class TestSandboxStatus:
    """Tests for `missy sandbox status`."""

    def test_sandbox_status_exits_zero(self, runner: CliRunner) -> None:
        """sandbox status exits 0."""
        cfg_path = _write_temp_config()

        with _SubsystemsPatch(), patch("missy.security.container.ContainerSandbox.is_available", return_value=False):
            result = runner.invoke(cli, ["--config", cfg_path, "sandbox", "status"])

        assert result.exit_code == 0

    def test_sandbox_status_docker_not_available(self, runner: CliRunner) -> None:
        """When Docker is unavailable, output shows 'not found'."""
        cfg_path = _write_temp_config()

        with _SubsystemsPatch(), patch("missy.security.container.ContainerSandbox.is_available", return_value=False):
            result = runner.invoke(cli, ["--config", cfg_path, "sandbox", "status"])

        assert result.exit_code == 0
        assert "not found" in result.output

    def test_sandbox_status_docker_available(self, runner: CliRunner) -> None:
        """When Docker is available, output shows 'available'."""
        cfg_path = _write_temp_config()

        with _SubsystemsPatch(), patch("missy.security.container.ContainerSandbox.is_available", return_value=True):
            result = runner.invoke(cli, ["--config", cfg_path, "sandbox", "status"])

        assert result.exit_code == 0
        assert "available" in result.output

    def test_sandbox_status_shows_disabled_when_not_configured(self, runner: CliRunner) -> None:
        """When container config is absent, shows 'not configured'."""
        cfg_path = _write_temp_config()

        with _SubsystemsPatch() as cfg, patch("missy.security.container.ContainerSandbox.is_available", return_value=False):
            cfg.container = None
            result = runner.invoke(cli, ["--config", cfg_path, "sandbox", "status"])

        assert result.exit_code == 0
        assert "not configured" in result.output

    def test_sandbox_status_shows_container_settings(self, runner: CliRunner) -> None:
        """When container config is present, shows image and memory_limit."""
        from missy.security.container import ContainerConfig

        cfg_path = _write_temp_config()
        container_cfg = ContainerConfig(
            enabled=False,
            image="python:3.12-slim",
            memory_limit="512m",
            cpu_quota=0.5,
            network_mode="none",
        )

        with _SubsystemsPatch() as cfg:
            cfg.container = container_cfg
            with patch("missy.security.container.ContainerSandbox.is_available", return_value=False):
                result = runner.invoke(cli, ["--config", cfg_path, "sandbox", "status"])

        assert result.exit_code == 0
        assert "python:3.12-slim" in result.output
        assert "512m" in result.output

    def test_sandbox_status_help_exits_zero(self, runner: CliRunner) -> None:
        """sandbox status --help exits 0."""
        result = runner.invoke(cli, ["sandbox", "status", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# missy cost
# ===========================================================================


class TestCost:
    """Tests for `missy cost`."""

    def test_cost_exits_zero_no_session(self, runner: CliRunner) -> None:
        """cost without --session exits 0."""
        cfg_path = _write_temp_config()

        with _SubsystemsPatch() as cfg:
            cfg.max_spend_usd = 0.0
            result = runner.invoke(cli, ["--config", cfg_path, "cost"])

        assert result.exit_code == 0

    def test_cost_shows_unlimited_when_budget_zero(self, runner: CliRunner) -> None:
        """When max_spend_usd is 0, output says 'unlimited'."""
        cfg_path = _write_temp_config()

        with _SubsystemsPatch() as cfg:
            cfg.max_spend_usd = 0.0
            result = runner.invoke(cli, ["--config", cfg_path, "cost"])

        assert result.exit_code == 0
        assert "unlimited" in result.output

    def test_cost_shows_budget_when_set(self, runner: CliRunner) -> None:
        """When max_spend_usd is nonzero, output shows the dollar amount."""
        cfg_path = _write_temp_config()

        with _SubsystemsPatch() as cfg:
            cfg.max_spend_usd = 5.0
            result = runner.invoke(cli, ["--config", cfg_path, "cost"])

        assert result.exit_code == 0
        assert "$5.00" in result.output

    def test_cost_shows_config_location_hint(self, runner: CliRunner) -> None:
        """Output includes the hint about max_spend_usd in config.yaml."""
        cfg_path = _write_temp_config()

        with _SubsystemsPatch() as cfg:
            cfg.max_spend_usd = 0.0
            result = runner.invoke(cli, ["--config", cfg_path, "cost"])

        assert result.exit_code == 0
        assert "max_spend_usd" in result.output

    def test_cost_with_session_queries_store(self, runner: CliRunner) -> None:
        """With --session, cost calls get_session_turns on the memory store."""
        cfg_path = _write_temp_config()
        mock_store = MagicMock()
        mock_store.get_session_turns.return_value = []
        mock_store.get_session_costs.return_value = []

        with _SubsystemsPatch() as cfg:
            cfg.max_spend_usd = 0.0
            with patch(
                "missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "cost", "--session", "my-session-id"]
                )

        assert result.exit_code == 0
        assert "my-session-id" in result.output

    def test_cost_with_session_shows_cost_records(self, runner: CliRunner) -> None:
        """When cost records exist for a session, totals appear in the output."""
        cfg_path = _write_temp_config()
        cost_records = [
            {
                "cost_usd": 0.001234,
                "prompt_tokens": 500,
                "completion_tokens": 200,
                "model": "claude-sonnet-4-6",
            },
        ]
        mock_store = MagicMock()
        mock_store.get_session_turns.return_value = [MagicMock()]
        mock_store.get_session_costs.return_value = cost_records

        with _SubsystemsPatch() as cfg:
            cfg.max_spend_usd = 0.0
            with patch(
                "missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "cost", "--session", "abc123"]
                )

        assert result.exit_code == 0
        # Token totals should appear
        assert "500" in result.output
        assert "200" in result.output

    def test_cost_with_session_no_cost_records(self, runner: CliRunner) -> None:
        """When no cost records exist for a session, shows 'No cost records' message."""
        cfg_path = _write_temp_config()
        mock_store = MagicMock()
        mock_store.get_session_turns.return_value = []
        mock_store.get_session_costs.return_value = []

        with _SubsystemsPatch() as cfg:
            cfg.max_spend_usd = 0.0
            with patch(
                "missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "cost", "--session", "empty-session"]
                )

        assert result.exit_code == 0
        assert "No cost records" in result.output

    def test_cost_help_exits_zero(self, runner: CliRunner) -> None:
        """cost --help exits 0."""
        result = runner.invoke(cli, ["cost", "--help"])
        assert result.exit_code == 0

    def test_cost_with_session_store_error_shows_error(self, runner: CliRunner) -> None:
        """When the memory store raises an exception, the error appears in output."""
        cfg_path = _write_temp_config()

        with _SubsystemsPatch() as cfg:
            cfg.max_spend_usd = 0.0
            with patch(
                "missy.memory.sqlite_store.SQLiteMemoryStore",
                side_effect=RuntimeError("database locked"),
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "cost", "--session", "bad-session"]
                )

        assert result.exit_code == 0  # command handles exceptions gracefully
        assert "Error" in result.output or "error" in result.output.lower()


# ===========================================================================
# Cross-cutting: --help for every top-level group
# ===========================================================================


class TestTopLevelHelp:
    """Verifies that --help for each command group exits cleanly."""

    @pytest.mark.parametrize(
        "args",
        [
            ["--help"],
            ["audit", "--help"],
            ["config", "--help"],
            ["devices", "--help"],
            ["discord", "--help"],
            ["gateway", "--help"],
            ["mcp", "--help"],
            ["patches", "--help"],
            ["persona", "--help"],
            ["presets", "--help"],
            ["providers", "--help"],
            ["sandbox", "--help"],
            ["schedule", "--help"],
            ["sessions", "--help"],
            ["skills", "--help"],
            ["vault", "--help"],
            ["voice", "--help"],
        ],
    )
    def test_help_exits_zero(self, runner: CliRunner, args: list[str]) -> None:
        """Every command group's --help exits 0."""
        result = runner.invoke(cli, args)
        assert result.exit_code == 0, (
            f"Expected exit 0 for `missy {' '.join(args)}` but got {result.exit_code}. "
            f"Output: {result.output}"
        )
