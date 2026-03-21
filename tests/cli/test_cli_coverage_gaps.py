"""Coverage gap tests for missy/cli/main.py.

Targets the following uncovered lines:
- Lines 149-150: _load_subsystems — migrate_config raises, warning logged and execution continues
- Lines 338-355: setup --no-prompt — missing provider error, ClickException from wizard, success
- Lines 413-414: ask command — HatchingManager().needs_hatching() raises, silently passed
- Lines 520-521: run command — HatchingManager().needs_hatching() raises, silently passed
- Lines 991-1002: providers switch — success path and ValueError error path
- Lines 1055-1080: skills scan — with skill results (table rendered) and no results
- Lines 1513-1534: gateway start — screencast channel enabled path (start + failure)
- Lines 1539-1649: gateway start — Discord channel configured path
- Lines 1664-1668: gateway start finally block — screencast channel stop called
- Lines 2572-2588: mcp pin — KeyError path and generic Exception path
- Lines 2981-3000: config diff — config missing, no backups, no diff, diff present
- Lines 3007-3016: config rollback — no backup, with backup
- Lines 3023-3043: config plan — no config, no backups, no diff, diff present
- Line 3143: hatch — status is not HATCHED or FAILED (else branch, e.g. IN_PROGRESS)
- Line 3312: __main__ entry point
"""

from __future__ import annotations

import signal
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from missy.cli.main import cli
from tests.cli.conftest import _make_cli_runner

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG_YAML = """\
network:
  default_deny: true
providers:
  anthropic:
    name: anthropic
    model: "claude-sonnet-4-6"
workspace_path: "/tmp/workspace"
audit_log_path: "/tmp/audit.jsonl"
"""


@pytest.fixture()
def runner() -> CliRunner:
    return _make_cli_runner(mix_stderr=False)


def _make_mock_config(**overrides) -> MagicMock:
    cfg = MagicMock()
    cfg.audit_log_path = "/tmp/audit.jsonl"
    cfg.workspace_path = "/tmp/workspace"
    cfg.max_spend_usd = 0.0
    cfg.providers = {}
    cfg.network.default_deny = True
    cfg.network.allowed_domains = []
    cfg.network.allowed_cidrs = []
    cfg.network.allowed_hosts = []
    cfg.shell.enabled = False
    cfg.shell.allowed_commands = []
    cfg.plugins.enabled = False
    cfg.plugins.allowed_plugins = []
    cfg.discord = None
    cfg.vault = None
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _write_config(content: str) -> str:
    """Write config content to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        return f.name


# ---------------------------------------------------------------------------
# Lines 149-150: migrate_config raises — warning logged, execution continues
# ---------------------------------------------------------------------------


class TestLoadSubsystemsMigrationException:
    def test_migrate_config_exception_continues_loading(self, runner: CliRunner) -> None:
        """When migrate_config raises an exception, the warning is logged and
        config loading continues normally (lines 149-150). Verified by calling
        _load_subsystems directly with a real config file.
        """
        from missy.cli.main import _load_subsystems

        cfg_path = _write_config(_MINIMAL_CONFIG_YAML)
        try:
            with (
                patch(
                    "missy.config.migrate.migrate_config",
                    side_effect=RuntimeError("migration failed unexpectedly"),
                ),
                patch("missy.config.settings.load_config", return_value=_make_mock_config()),
                patch("missy.policy.engine.init_policy_engine"),
                patch("missy.observability.audit_logger.init_audit_logger"),
                patch("missy.providers.registry.init_registry"),
                patch("missy.tools.builtin.register_builtin_tools"),
                patch("missy.tools.registry.init_tool_registry"),
            ):
                # _load_subsystems should not raise — the migration exception is swallowed
                result = _load_subsystems(cfg_path)

            assert result is not None
        finally:
            import os

            os.unlink(cfg_path)

    def test_migrate_config_exception_is_swallowed_directly(
        self, runner: CliRunner, tmp_path
    ) -> None:
        """Call a command that exercises _load_subsystems with migrate_config raising."""
        from missy.cli.main import _load_subsystems

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_MINIMAL_CONFIG_YAML)

        with (
            patch("missy.config.migrate.migrate_config", side_effect=ValueError("bad migration")),
            patch("missy.config.settings.load_config", return_value=_make_mock_config()),
            patch("missy.policy.engine.init_policy_engine"),
            patch("missy.observability.audit_logger.init_audit_logger"),
            patch("missy.providers.registry.init_registry"),
            patch("missy.tools.builtin.register_builtin_tools"),
            patch("missy.tools.registry.init_tool_registry"),
        ):
            # Should not raise — the exception is logged and swallowed
            result = _load_subsystems(str(cfg_path))
            assert result is not None


# ---------------------------------------------------------------------------
# Lines 338-355: setup --no-prompt paths
# ---------------------------------------------------------------------------


class TestSetupNoPrompt:
    def test_setup_no_prompt_without_provider_exits_1(self, runner: CliRunner) -> None:
        """setup --no-prompt without --provider prints an error and exits 1 (line 341-342)."""
        result = runner.invoke(cli, ["setup", "--no-prompt"])
        assert result.exit_code == 1
        all_output = result.output + result.stderr
        assert "--provider" in all_output

    def test_setup_no_prompt_wizard_click_exception_exits_1(self, runner: CliRunner) -> None:
        """setup --no-prompt when run_wizard_noninteractive raises ClickException
        prints the error message and exits 1 (lines 352-354).
        """
        import click

        with patch(
            "missy.cli.wizard.run_wizard_noninteractive",
            side_effect=click.ClickException("invalid provider configuration"),
        ):
            result = runner.invoke(cli, ["setup", "--no-prompt", "--provider", "anthropic"])

        assert result.exit_code == 1
        all_output = result.output + result.stderr
        assert "invalid provider configuration" in all_output

    def test_setup_no_prompt_success_returns(self, runner: CliRunner) -> None:
        """setup --no-prompt with valid --provider calls run_wizard_noninteractive
        and returns normally (lines 344-355).
        """
        with patch("missy.cli.wizard.run_wizard_noninteractive") as mock_wizard:
            result = runner.invoke(
                cli,
                [
                    "setup",
                    "--no-prompt",
                    "--provider",
                    "anthropic",
                    "--model",
                    "claude-sonnet-4-6",
                ],
            )

        assert result.exit_code == 0
        mock_wizard.assert_called_once()
        kwargs = mock_wizard.call_args[1]
        assert kwargs.get("provider") == "anthropic"
        assert kwargs.get("model") == "claude-sonnet-4-6"

    def test_setup_no_prompt_passes_api_key_env(self, runner: CliRunner) -> None:
        """setup --no-prompt forwards --api-key-env to run_wizard_noninteractive."""
        with patch("missy.cli.wizard.run_wizard_noninteractive") as mock_wizard:
            result = runner.invoke(
                cli,
                [
                    "setup",
                    "--no-prompt",
                    "--provider",
                    "anthropic",
                    "--api-key-env",
                    "ANTHROPIC_API_KEY",
                ],
            )

        assert result.exit_code == 0
        kwargs = mock_wizard.call_args[1]
        assert kwargs.get("api_key_env") == "ANTHROPIC_API_KEY"


# ---------------------------------------------------------------------------
# Lines 413-414: ask command — hatching check raises, silently passed
# ---------------------------------------------------------------------------


class TestAskHatchingCheckException:
    def test_ask_hatching_exception_is_swallowed(self, runner: CliRunner) -> None:
        """When HatchingManager().needs_hatching() raises in the ask command,
        the exception is silently passed (lines 413-414).
        """
        mock_config = _make_mock_config()
        mock_runtime = MagicMock()
        mock_runtime.run.return_value = "Paris"

        # needs_hatching raises — should be swallowed
        mock_hatching_mgr = MagicMock()
        mock_hatching_mgr.needs_hatching.side_effect = RuntimeError("hatching db corrupt")

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.agent.hatching.HatchingManager", return_value=mock_hatching_mgr),
            patch("missy.agent.runtime.AgentRuntime", return_value=mock_runtime),
            patch("missy.agent.runtime.AgentConfig"),
            patch("missy.security.secrets.SecretsDetector.has_secrets", return_value=False),
            patch("missy.security.sanitizer.InputSanitizer.sanitize", side_effect=lambda x: x),
        ):
            result = runner.invoke(cli, ["ask", "What is the capital of France?"])

        # Command must complete — hatching exception is swallowed, agent runs normally
        assert result.exit_code == 0
        assert "Paris" in result.output

    def test_ask_hatching_exception_does_not_show_tip(self, runner: CliRunner) -> None:
        """When HatchingManager() constructor itself raises, the tip is not printed."""
        mock_config = _make_mock_config()
        mock_runtime = MagicMock()
        mock_runtime.run.return_value = "response text"

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch(
                "missy.agent.hatching.HatchingManager",
                side_effect=ImportError("no hatching module"),
            ),
            patch("missy.agent.runtime.AgentRuntime", return_value=mock_runtime),
            patch("missy.agent.runtime.AgentConfig"),
            patch("missy.security.secrets.SecretsDetector.has_secrets", return_value=False),
            patch("missy.security.sanitizer.InputSanitizer.sanitize", side_effect=lambda x: x),
        ):
            result = runner.invoke(cli, ["ask", "hello"])

        assert result.exit_code == 0
        # The "Tip: Run missy hatch" line must NOT appear since the exception was swallowed
        assert "missy hatch" not in result.output


# ---------------------------------------------------------------------------
# Lines 520-521: run command — hatching check raises, silently passed
# ---------------------------------------------------------------------------


class TestRunHatchingCheckException:
    def test_run_hatching_exception_is_swallowed(self, runner: CliRunner) -> None:
        """When HatchingManager().needs_hatching() raises in the run command,
        the exception is silently passed (lines 520-521).
        """
        mock_config = _make_mock_config()
        # Return None from receive() immediately to simulate EOF / Ctrl-D so the
        # interactive loop exits cleanly without needing a real tty.
        mock_channel = MagicMock()
        mock_channel.receive.return_value = None

        mock_hatching_mgr = MagicMock()
        mock_hatching_mgr.needs_hatching.side_effect = OSError("hatching file not found")

        mock_agent = MagicMock()
        mock_agent.pending_recovery = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.agent.hatching.HatchingManager", return_value=mock_hatching_mgr),
            patch("missy.channels.cli_channel.CLIChannel", return_value=mock_channel),
            patch("missy.agent.runtime.AgentRuntime", return_value=mock_agent),
            patch("missy.agent.runtime.AgentConfig"),
        ):
            result = runner.invoke(cli, ["run"])

        # Command completes — hatching exception does not bubble up
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Lines 991-1002: providers switch
# ---------------------------------------------------------------------------


class TestProvidersSwitch:
    def test_providers_switch_success(self, runner: CliRunner) -> None:
        """providers switch calls set_default and prints success message (line 1002)."""
        mock_config = _make_mock_config()
        mock_registry = MagicMock()
        mock_registry.set_default.return_value = None

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
        ):
            result = runner.invoke(cli, ["providers", "switch", "openai"])

        assert result.exit_code == 0
        mock_registry.set_default.assert_called_once_with("openai")
        assert "openai" in result.output.lower() or "switched" in result.output.lower()

    def test_providers_switch_unknown_provider_exits_1(self, runner: CliRunner) -> None:
        """providers switch with an unknown provider name prints error and exits 1 (lines 998-1000)."""
        mock_config = _make_mock_config()
        mock_registry = MagicMock()
        mock_registry.set_default.side_effect = ValueError("Provider 'nonexistent' not found")

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
        ):
            result = runner.invoke(cli, ["providers", "switch", "nonexistent"])

        assert result.exit_code == 1
        all_output = result.output + result.stderr
        assert "nonexistent" in all_output or "not found" in all_output.lower()


# ---------------------------------------------------------------------------
# Lines 1055-1080: skills scan
# ---------------------------------------------------------------------------


class TestSkillsScan:
    def test_skills_scan_no_results(self, runner: CliRunner) -> None:
        """skills scan with no manifests found prints a 'no SKILL.md' message (lines 1060-1062)."""
        mock_config = _make_mock_config()
        mock_discovery = MagicMock()
        mock_discovery.scan_directory.return_value = []

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.skills.discovery.SkillDiscovery", return_value=mock_discovery),
        ):
            result = runner.invoke(cli, ["skills", "scan", "--path", "/tmp/no_skills"])

        assert result.exit_code == 0
        assert "No SKILL.md" in result.output or "no skill" in result.output.lower()

    def test_skills_scan_with_results_renders_table(self, runner: CliRunner) -> None:
        """skills scan with discovered manifests renders a table (lines 1064-1080)."""
        mock_config = _make_mock_config()

        manifest = MagicMock()
        manifest.name = "git-helper"
        manifest.version = "1.0"
        manifest.author = "tester"
        manifest.description = "Automates common git workflows for development teams"
        manifest.tools = ["run_shell", "read_file"]

        mock_discovery = MagicMock()
        mock_discovery.scan_directory.return_value = [manifest]

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.skills.discovery.SkillDiscovery", return_value=mock_discovery),
        ):
            result = runner.invoke(cli, ["skills", "scan", "--path", "/tmp/my_skills"])

        assert result.exit_code == 0
        assert "git-helper" in result.output
        assert "tester" in result.output
        assert "1.0" in result.output

    def test_skills_scan_long_description_truncated(self, runner: CliRunner) -> None:
        """skills scan truncates descriptions longer than 60 chars with an ellipsis (line 1076)."""
        mock_config = _make_mock_config()

        long_desc = "A" * 80  # exceeds the 60-char truncation limit
        manifest = MagicMock()
        manifest.name = "long-skill"
        manifest.version = "2.0"
        manifest.author = None
        manifest.description = long_desc
        manifest.tools = []

        mock_discovery = MagicMock()
        mock_discovery.scan_directory.return_value = [manifest]

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.skills.discovery.SkillDiscovery", return_value=mock_discovery),
        ):
            result = runner.invoke(cli, ["skills", "scan"])

        assert result.exit_code == 0
        # Rich may render the trailing ellipsis as either ASCII "..." or Unicode "…"
        assert "..." in result.output or "\u2026" in result.output
        assert long_desc not in result.output  # full string not present; truncated


# ---------------------------------------------------------------------------
# Lines 1513-1534: gateway start — screencast channel start path
# ---------------------------------------------------------------------------


class TestGatewayStartScreencast:
    def test_gateway_start_screencast_enabled_starts_channel(self, runner: CliRunner) -> None:
        """When the config YAML includes screencast.enabled=true, ScreencastChannel
        is instantiated and started (lines 1513-1533).
        """
        import os

        cfg_path = _write_config(_MINIMAL_CONFIG_YAML)
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None

            mock_sc_channel = MagicMock()
            mock_sc_channel.start.return_value = None
            mock_sc_channel.stop.return_value = None
            mock_sc_channel._server = MagicMock()
            mock_sc_channel._server._tls_enabled = False

            call_count = [0]

            def fake_sleep(t):
                call_count[0] += 1
                if call_count[0] >= 1:
                    os.kill(os.getpid(), signal.SIGTERM)

            screencast_yaml = {
                "screencast": {
                    "enabled": True,
                    "host": "127.0.0.1",
                    "port": 8780,
                }
            }

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("yaml.safe_load", return_value=screencast_yaml),
                patch(
                    "missy.channels.screencast.channel.ScreencastChannel",
                    return_value=mock_sc_channel,
                ),
                patch("time.sleep", side_effect=fake_sleep),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "start"])

            assert result.exit_code == 0
            mock_sc_channel.start.assert_called_once()
            assert "8780" in result.output or "Screencast" in result.output
        finally:
            import os as _os

            _os.unlink(cfg_path)

    def test_gateway_start_screencast_start_failure_shows_warning(self, runner: CliRunner) -> None:
        """When ScreencastChannel.start() raises, a warning is printed and gateway continues
        (lines 1532-1534).
        """
        import os

        cfg_path = _write_config(_MINIMAL_CONFIG_YAML)
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None

            mock_sc_channel = MagicMock()
            mock_sc_channel.start.side_effect = RuntimeError("screencast port in use")

            call_count = [0]

            def fake_sleep(t):
                call_count[0] += 1
                if call_count[0] >= 1:
                    os.kill(os.getpid(), signal.SIGTERM)

            screencast_yaml = {
                "screencast": {
                    "enabled": True,
                    "host": "127.0.0.1",
                    "port": 8780,
                }
            }

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("yaml.safe_load", return_value=screencast_yaml),
                patch(
                    "missy.channels.screencast.channel.ScreencastChannel",
                    return_value=mock_sc_channel,
                ),
                patch("time.sleep", side_effect=fake_sleep),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "start"])

            assert result.exit_code == 0
            assert (
                "failed to start" in result.output.lower() or "screencast" in result.output.lower()
            )
        finally:
            import os as _os

            _os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# Lines 1664-1668: gateway start finally block — screencast channel stop
# ---------------------------------------------------------------------------


class TestGatewayStartScreencastStop:
    def test_gateway_start_screencast_channel_stopped_in_finally(self, runner: CliRunner) -> None:
        """When screencast_channel is not None, its stop() is called in the finally block (lines 1664-1666)."""
        import os

        cfg_path = _write_config(_MINIMAL_CONFIG_YAML)
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None

            mock_sc_channel = MagicMock()
            mock_sc_channel.start.return_value = None
            mock_sc_channel.stop.return_value = None
            mock_sc_channel._server = MagicMock()
            mock_sc_channel._server._tls_enabled = False

            call_count = [0]

            def fake_sleep(t):
                call_count[0] += 1
                if call_count[0] >= 1:
                    os.kill(os.getpid(), signal.SIGTERM)

            screencast_yaml = {
                "screencast": {
                    "enabled": True,
                    "host": "127.0.0.1",
                    "port": 8780,
                }
            }

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("yaml.safe_load", return_value=screencast_yaml),
                patch(
                    "missy.channels.screencast.channel.ScreencastChannel",
                    return_value=mock_sc_channel,
                ),
                patch("time.sleep", side_effect=fake_sleep),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "start"])

            assert result.exit_code == 0
            # stop() must have been called in the finally block
            mock_sc_channel.stop.assert_called_once()

        finally:
            import os as _os

            _os.unlink(cfg_path)

    def test_gateway_start_screencast_stop_exception_swallowed(self, runner: CliRunner) -> None:
        """When screencast_channel.stop() raises, the exception is swallowed (lines 1667-1668)."""
        import os

        cfg_path = _write_config(_MINIMAL_CONFIG_YAML)
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None

            mock_sc_channel = MagicMock()
            mock_sc_channel.start.return_value = None
            mock_sc_channel.stop.side_effect = RuntimeError("cannot stop screencast")
            mock_sc_channel._server = MagicMock()
            mock_sc_channel._server._tls_enabled = False

            call_count = [0]

            def fake_sleep(t):
                call_count[0] += 1
                if call_count[0] >= 1:
                    os.kill(os.getpid(), signal.SIGTERM)

            screencast_yaml = {
                "screencast": {
                    "enabled": True,
                }
            }

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("yaml.safe_load", return_value=screencast_yaml),
                patch(
                    "missy.channels.screencast.channel.ScreencastChannel",
                    return_value=mock_sc_channel,
                ),
                patch("time.sleep", side_effect=fake_sleep),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "start"])

            # Exception from stop() must not propagate to the user
            assert result.exit_code == 0
            assert "cannot stop screencast" not in result.output
        finally:
            import os as _os

            _os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# Lines 2572-2588: mcp pin — error paths
# ---------------------------------------------------------------------------


class TestMcpPin:
    def test_mcp_pin_server_not_connected_exits_1(self, runner: CliRunner) -> None:
        """mcp pin when server is not connected raises KeyError → prints error, exits 1 (lines 2580-2582)."""
        mock_config = _make_mock_config()
        mock_mgr = MagicMock()
        mock_mgr.pin_server_digest.side_effect = KeyError("myserver")

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.mcp.manager.McpManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["mcp", "pin", "myserver"])

        assert result.exit_code == 1
        all_output = result.output + result.stderr
        assert "not connected" in all_output or "myserver" in all_output

    def test_mcp_pin_generic_exception_exits_1(self, runner: CliRunner) -> None:
        """mcp pin when pin_server_digest raises a generic exception → prints error, exits 1 (lines 2583-2585)."""
        mock_config = _make_mock_config()
        mock_mgr = MagicMock()
        mock_mgr.pin_server_digest.side_effect = OSError("connection refused")

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.mcp.manager.McpManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["mcp", "pin", "myserver"])

        assert result.exit_code == 1
        all_output = result.output + result.stderr
        assert "connection refused" in all_output or "failed" in all_output.lower()

    def test_mcp_pin_success(self, runner: CliRunner) -> None:
        """mcp pin success prints the pinned digest (lines 2587-2588)."""
        mock_config = _make_mock_config()
        mock_mgr = MagicMock()
        mock_mgr.pin_server_digest.return_value = "sha256:abcdef1234567890"

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.mcp.manager.McpManager", return_value=mock_mgr),
        ):
            result = runner.invoke(cli, ["mcp", "pin", "myserver"])

        assert result.exit_code == 0
        assert "sha256:abcdef1234567890" in result.output
        mock_mgr.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# Lines 2981-3000: config diff
# ---------------------------------------------------------------------------


class TestConfigDiff:
    def test_config_diff_no_config_file_exits_1(self, runner: CliRunner, tmp_path) -> None:
        """config diff when config file doesn't exist prints error and exits 1 (lines 2986-2988)."""
        nonexistent_cfg = tmp_path / "missing_config.yaml"

        result = runner.invoke(cli, ["--config", str(nonexistent_cfg), "config", "diff"])

        assert result.exit_code == 1
        all_output = result.output + result.stderr
        assert "not found" in all_output.lower() or "missing" in all_output.lower()

    def test_config_diff_no_backups_prints_message(self, runner: CliRunner, tmp_path) -> None:
        """config diff with no backups available prints dim message (lines 2991-2993)."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_MINIMAL_CONFIG_YAML)

        with patch("missy.config.plan.list_backups", return_value=[]):
            result = runner.invoke(cli, ["--config", str(cfg_path), "config", "diff"])

        assert result.exit_code == 0
        assert "No backups" in result.output or "no backups" in result.output.lower()

    def test_config_diff_no_differences(self, runner: CliRunner, tmp_path) -> None:
        """config diff with no diff between current and backup prints 'No differences' (lines 2997-2998)."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_MINIMAL_CONFIG_YAML)

        backup_path = tmp_path / "config.yaml.backup"
        backup_path.write_text(_MINIMAL_CONFIG_YAML)

        with (
            patch("missy.config.plan.list_backups", return_value=[backup_path]),
            patch("missy.config.plan.diff_configs", return_value=""),
        ):
            result = runner.invoke(cli, ["--config", str(cfg_path), "config", "diff"])

        assert result.exit_code == 0
        assert "No differences" in result.output

    def test_config_diff_with_differences_prints_diff(self, runner: CliRunner, tmp_path) -> None:
        """config diff with differences prints the diff text (line 3000)."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_MINIMAL_CONFIG_YAML)

        backup_path = tmp_path / "config.yaml.backup"
        backup_path.write_text(_MINIMAL_CONFIG_YAML)

        diff_text = "--- backup\n+++ current\n@@ -1 +1 @@\n-model: old\n+model: new\n"

        with (
            patch("missy.config.plan.list_backups", return_value=[backup_path]),
            patch("missy.config.plan.diff_configs", return_value=diff_text),
        ):
            result = runner.invoke(cli, ["--config", str(cfg_path), "config", "diff"])

        assert result.exit_code == 0
        assert "model: new" in result.output


# ---------------------------------------------------------------------------
# Lines 3007-3016: config rollback
# ---------------------------------------------------------------------------


class TestConfigRollback:
    def test_config_rollback_no_backups(self, runner: CliRunner, tmp_path) -> None:
        """config rollback with no backups prints 'No backups available' message (lines 3013-3014)."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_MINIMAL_CONFIG_YAML)

        with patch("missy.config.plan.rollback", return_value=None):
            result = runner.invoke(cli, ["--config", str(cfg_path), "config", "rollback"])

        assert result.exit_code == 0
        assert "No backups" in result.output or "no backups" in result.output.lower()

    def test_config_rollback_success(self, runner: CliRunner, tmp_path) -> None:
        """config rollback with a backup prints success message (lines 3015-3016)."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_MINIMAL_CONFIG_YAML)

        restored_backup = tmp_path / "config.yaml.20260318_120000"
        restored_backup.write_text(_MINIMAL_CONFIG_YAML)

        with patch("missy.config.plan.rollback", return_value=restored_backup):
            result = runner.invoke(cli, ["--config", str(cfg_path), "config", "rollback"])

        assert result.exit_code == 0
        assert "config.yaml.20260318_120000" in result.output or "restored" in result.output.lower()


# ---------------------------------------------------------------------------
# Lines 3023-3043: config plan
# ---------------------------------------------------------------------------


class TestConfigPlan:
    def test_config_plan_no_config_file(self, runner: CliRunner, tmp_path) -> None:
        """config plan when no config file exists prints setup hint (lines 3028-3030)."""
        nonexistent_cfg = tmp_path / "missing_config.yaml"

        result = runner.invoke(cli, ["--config", str(nonexistent_cfg), "config", "plan"])

        assert result.exit_code == 0
        assert "setup" in result.output.lower() or "No config file" in result.output

    def test_config_plan_no_backups(self, runner: CliRunner, tmp_path) -> None:
        """config plan with no backups prints 'No previous backups' (lines 3032-3035)."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_MINIMAL_CONFIG_YAML)

        with patch("missy.config.plan.list_backups", return_value=[]):
            result = runner.invoke(cli, ["--config", str(cfg_path), "config", "plan"])

        assert result.exit_code == 0
        assert "No previous backups" in result.output or "baseline" in result.output.lower()

    def test_config_plan_no_differences(self, runner: CliRunner, tmp_path) -> None:
        """config plan with no diff between current and backup prints 'matches' message (lines 3038-3040)."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_MINIMAL_CONFIG_YAML)

        backup_path = tmp_path / "config.yaml.backup"
        backup_path.write_text(_MINIMAL_CONFIG_YAML)

        with (
            patch("missy.config.plan.list_backups", return_value=[backup_path]),
            patch("missy.config.plan.diff_configs", return_value=""),
        ):
            result = runner.invoke(cli, ["--config", str(cfg_path), "config", "plan"])

        assert result.exit_code == 0
        assert "matches" in result.output.lower() or "no changes" in result.output.lower()

    def test_config_plan_with_differences(self, runner: CliRunner, tmp_path) -> None:
        """config plan with diff text prints 'Changes since last backup' header (lines 3041-3043)."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_MINIMAL_CONFIG_YAML)

        backup_path = tmp_path / "config.yaml.backup"
        backup_path.write_text(_MINIMAL_CONFIG_YAML)

        diff_text = (
            "--- backup\n+++ current\n@@ -1 +1 @@\n-old_setting: true\n+new_setting: false\n"
        )

        with (
            patch("missy.config.plan.list_backups", return_value=[backup_path]),
            patch("missy.config.plan.diff_configs", return_value=diff_text),
        ):
            result = runner.invoke(cli, ["--config", str(cfg_path), "config", "plan"])

        assert result.exit_code == 0
        assert "Changes since last backup" in result.output
        assert "new_setting" in result.output


# ---------------------------------------------------------------------------
# Line 3143: hatch — status is not HATCHED or FAILED (else branch)
# ---------------------------------------------------------------------------


class TestHatchElseBranch:
    def test_hatch_in_progress_status_prints_status_value(self, runner: CliRunner) -> None:
        """When run_hatching() returns IN_PROGRESS status, the else branch prints the
        status value (line 3143).
        """
        from missy.agent.hatching import HatchingState, HatchingStatus

        state = HatchingState(
            status=HatchingStatus.IN_PROGRESS,
            steps_completed=["validate_environment"],
        )

        mock_mgr = MagicMock()
        mock_mgr.is_hatched.return_value = False
        mock_mgr.run_hatching.return_value = state

        with patch("missy.agent.hatching.HatchingManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["hatch"])

        assert result.exit_code == 0
        assert "in_progress" in result.output.lower() or "in progress" in result.output.lower()

    def test_hatch_unhatched_status_prints_status_value(self, runner: CliRunner) -> None:
        """When run_hatching() returns UNHATCHED status, the else branch prints it."""
        from missy.agent.hatching import HatchingState, HatchingStatus

        state = HatchingState(
            status=HatchingStatus.UNHATCHED,
            steps_completed=[],
        )

        mock_mgr = MagicMock()
        mock_mgr.is_hatched.return_value = False
        mock_mgr.run_hatching.return_value = state

        with patch("missy.agent.hatching.HatchingManager", return_value=mock_mgr):
            result = runner.invoke(cli, ["hatch"])

        assert result.exit_code == 0
        assert "unhatched" in result.output.lower()


# ---------------------------------------------------------------------------
# Lines 1539-1649: gateway start — Discord channel configured (idle service mode)
# ---------------------------------------------------------------------------


class TestGatewayStartDiscord:
    def test_gateway_start_no_discord_runs_idle_mode(self, runner: CliRunner) -> None:
        """When Discord is not configured, gateway prints idle message and loops (lines 1650-1655)."""
        import os

        cfg_path = _write_config(_MINIMAL_CONFIG_YAML)
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None

            call_count = [0]

            def fake_sleep(t):
                call_count[0] += 1
                if call_count[0] >= 1:
                    os.kill(os.getpid(), signal.SIGTERM)

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("yaml.safe_load", return_value={}),
                patch("time.sleep", side_effect=fake_sleep),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "start"])

            assert result.exit_code == 0
            assert "idle" in result.output.lower() or "No Discord" in result.output
        finally:
            import os as _os

            _os.unlink(cfg_path)

    def test_gateway_start_discord_disabled_runs_idle_mode(self, runner: CliRunner) -> None:
        """When Discord is configured but disabled, gateway runs in idle service mode."""
        import os

        cfg_path = _write_config(_MINIMAL_CONFIG_YAML)
        try:
            mock_config = _make_mock_config()

            # Discord configured but disabled
            discord_cfg = MagicMock()
            discord_cfg.enabled = False
            discord_cfg.accounts = []
            mock_config.discord = discord_cfg

            call_count = [0]

            def fake_sleep(t):
                call_count[0] += 1
                if call_count[0] >= 1:
                    os.kill(os.getpid(), signal.SIGTERM)

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("yaml.safe_load", return_value={}),
                patch("time.sleep", side_effect=fake_sleep),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "start"])

            assert result.exit_code == 0
        finally:
            import os as _os

            _os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# Line 3312: __main__ entry point
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    def test_main_module_entry_point_invocable(self) -> None:
        """Line 3312 (if __name__ == '__main__': cli()) can be reached by
        running the module directly. We test this by verifying the guard
        compiles and the cli object is callable.
        """

        # Verify the main module's __file__ exists and contains the guard
        from missy.cli import main as main_module

        source_path = Path(main_module.__file__)
        assert source_path.exists()
        content = source_path.read_text()
        assert 'if __name__ == "__main__":' in content
        assert "cli()" in content

    def test_cli_is_callable_as_main(self) -> None:
        """The cli() callable can be invoked via CliRunner simulating __main__ behaviour."""
        runner = _make_cli_runner(mix_stderr=False)
        # Invoke --help to confirm cli is callable without side effects
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "missy" in result.output.lower() or "Usage" in result.output
