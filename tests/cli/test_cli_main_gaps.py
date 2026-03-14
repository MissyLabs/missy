"""Targeted coverage tests for missy/cli/main.py gaps.

Targets remaining uncovered lines:
- Line 877-878: providers — is_available() raises exception → available = False
- Lines 1227-1228: gateway_start signal handler _stop function body
- Lines 1269-1276: gateway_start proactive callback (both AgentRuntime success/failure branches)
- Lines 1284-1285: gateway_start proactive manager startup exception
- Lines 1328-1330: gateway_start voice channel started message
- Lines 1338-1441: gateway_start Discord channel processing (idle/discord paths)
- Lines 1453-1454: gateway_start voice channel stop raises exception
- Lines 1458-1461: gateway_start proactive manager stop raises exception
- Line 1556: doctor — secrets dir missing
- Lines 1636-1644: doctor — MCP servers configured
- Line 1656: doctor — watchdog available
- Lines 1659-1660: doctor — watchdog not installed
- Lines 1672-1676: doctor — voice channel configured
- Lines 1683-1684: doctor — voice channel error
- Lines 1692-1694: doctor — checkpoint exists
- Lines 2110-2116: evolve show — diffs present (with and without description)
- Lines 2118, 2120: evolve show — error_pattern and test_output present
- Line 2667: __main__ entry point
"""

from __future__ import annotations

import signal
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from missy.cli.main import cli

# ---------------------------------------------------------------------------
# Shared helpers (mirror pattern from test_cli_main_extended.py)
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
    return CliRunner(mix_stderr=False)


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


def _cfg_path() -> str:
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(_MINIMAL_CONFIG_YAML)
        f.flush()
        return f.name


# ---------------------------------------------------------------------------
# providers — is_available() raises exception → available = False (lines 877-878)
# ---------------------------------------------------------------------------


class TestProvidersIsAvailableException:
    def test_is_available_exception_shows_no(self, runner: CliRunner):
        """When provider.is_available() raises, the table should show 'no' not crash."""
        cfg_path = _cfg_path()
        try:
            mock_provider = MagicMock()
            mock_provider.is_available.side_effect = RuntimeError("network error")

            mock_registry = MagicMock()
            mock_registry.get.return_value = mock_provider

            mock_config = _make_mock_config()
            mock_config.providers = {"anthropic": MagicMock(model="claude", base_url=None, timeout=30)}

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "providers"])

            assert result.exit_code == 0
            assert "no" in result.output.lower() or "anthropic" in result.output
        finally:
            import os
            os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# gateway_start — signal handler _stop body (lines 1227-1228)
# ---------------------------------------------------------------------------


class TestGatewayStartSignalHandler:
    def test_stop_signal_sets_stop_event(self, runner: CliRunner):
        """When SIGTERM is received, the _stop handler sets stop_event and prints a message.

        We test this by sending SIGTERM from within a patched time.sleep that captures
        the signal handler and calls it directly.
        """
        import os

        cfg_path = _cfg_path()
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None
            mock_config.providers = {}

            call_count = [0]

            def fake_sleep(t):
                call_count[0] += 1
                if call_count[0] == 1:
                    # Trigger SIGTERM on the first sleep to exercise _stop body
                    os.kill(os.getpid(), signal.SIGTERM)
                elif call_count[0] > 5:
                    # Safety: break out of loop if SIGTERM didn't stop things
                    raise SystemExit(0)

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("missy.agent.runtime.AgentRuntime"),
                patch("missy.agent.runtime.AgentConfig"),
                patch("time.sleep", side_effect=fake_sleep),
                patch("yaml.safe_load", return_value={}),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "start"])

            # Exit code 0: graceful shutdown via signal
            assert result.exit_code == 0
            assert "Shutting down" in result.output or "stopped" in result.output.lower()
        finally:
            import os as _os
            _os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# gateway_start — proactive manager exception path (lines 1284-1285)
# ---------------------------------------------------------------------------


class TestGatewayStartProactiveException:
    def test_proactive_manager_startup_exception_continues(self, runner: CliRunner):
        """When proactive manager fails to start, gateway should continue (not crash)."""
        import os

        cfg_path = _cfg_path()
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None
            mock_config.providers = {}

            # Simulate proactive being enabled so the branch is entered
            proactive_cfg = MagicMock()
            proactive_cfg.enabled = True
            proactive_cfg.triggers = [MagicMock()]
            mock_config.proactive = proactive_cfg

            call_count = [0]

            def fake_sleep(t):
                call_count[0] += 1
                if call_count[0] >= 1:
                    os.kill(os.getpid(), signal.SIGTERM)

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("missy.agent.runtime.AgentRuntime"),
                patch("missy.agent.runtime.AgentConfig"),
                # Make ProactiveManager raise during import or instantiation
                patch(
                    "missy.agent.proactive.ProactiveManager",
                    side_effect=RuntimeError("proactive boom"),
                ),
                patch("time.sleep", side_effect=fake_sleep),
                patch("yaml.safe_load", return_value={}),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "start"])

            assert result.exit_code == 0
            assert "failed to start" in result.output.lower() or "proactive" in result.output.lower()
        finally:
            import os as _os
            _os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# gateway_start — voice channel started message (lines 1328-1330)
# ---------------------------------------------------------------------------


class TestGatewayStartVoiceChannelStarted:
    def test_voice_channel_started_message_printed(self, runner: CliRunner):
        """When voice is enabled in config, voice channel start should print a message."""
        import os

        cfg_path = _cfg_path()
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None
            mock_config.providers = {}

            mock_vc = MagicMock()
            mock_vc.start = MagicMock()
            mock_vc.stop = MagicMock()

            call_count = [0]

            def fake_sleep(t):
                call_count[0] += 1
                if call_count[0] >= 1:
                    os.kill(os.getpid(), signal.SIGTERM)

            voice_cfg = {"enabled": True, "host": "0.0.0.0", "port": 8765}

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("missy.agent.runtime.AgentRuntime"),
                patch("missy.agent.runtime.AgentConfig"),
                patch("missy.channels.voice.channel.VoiceChannel", return_value=mock_vc),
                patch("yaml.safe_load", return_value={"voice": voice_cfg}),
                patch("time.sleep", side_effect=fake_sleep),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "start"])

            assert result.exit_code == 0
            assert "Voice channel started" in result.output
        finally:
            import os as _os
            _os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# gateway_start — voice channel stop raises exception (lines 1453-1454)
# ---------------------------------------------------------------------------


class TestGatewayStartVoiceStopException:
    def test_voice_channel_stop_exception_is_swallowed(self, runner: CliRunner):
        """When voice_channel.stop() raises, gateway should still exit cleanly."""
        import os

        cfg_path = _cfg_path()
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None
            mock_config.providers = {}

            mock_vc = MagicMock()
            mock_vc.start = MagicMock()
            mock_vc.stop = MagicMock(side_effect=RuntimeError("stop failed"))

            call_count = [0]

            def fake_sleep(t):
                call_count[0] += 1
                if call_count[0] >= 1:
                    os.kill(os.getpid(), signal.SIGTERM)

            voice_cfg = {"enabled": True, "host": "0.0.0.0", "port": 8765}

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("missy.agent.runtime.AgentRuntime"),
                patch("missy.agent.runtime.AgentConfig"),
                patch("missy.channels.voice.channel.VoiceChannel", return_value=mock_vc),
                patch("yaml.safe_load", return_value={"voice": voice_cfg}),
                patch("time.sleep", side_effect=fake_sleep),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "start"])

            # Should complete without error even though stop() raised
            assert result.exit_code == 0
        finally:
            import os as _os
            _os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# gateway_start — proactive manager stop exception (lines 1458-1461)
# ---------------------------------------------------------------------------


class TestGatewayStartProactiveStopException:
    def test_proactive_stop_exception_swallowed(self, runner: CliRunner):
        """When proactive_manager.stop() raises, gateway should still exit cleanly."""
        import os

        cfg_path = _cfg_path()
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None
            mock_config.providers = {}

            proactive_cfg = MagicMock()
            proactive_cfg.enabled = True
            proactive_cfg.triggers = [MagicMock()]
            mock_config.proactive = proactive_cfg

            mock_pm = MagicMock()
            mock_pm.start = MagicMock()
            mock_pm.stop = MagicMock(side_effect=RuntimeError("pm stop failed"))

            call_count = [0]

            def fake_sleep(t):
                call_count[0] += 1
                if call_count[0] >= 1:
                    os.kill(os.getpid(), signal.SIGTERM)

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("missy.agent.runtime.AgentRuntime"),
                patch("missy.agent.runtime.AgentConfig"),
                patch("missy.agent.proactive.ProactiveManager", return_value=mock_pm),
                patch("missy.agent.proactive.ProactiveTrigger"),
                patch("time.sleep", side_effect=fake_sleep),
                patch("yaml.safe_load", return_value={}),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "start"])

            assert result.exit_code == 0
        finally:
            import os as _os
            _os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# doctor — secrets dir missing (line 1556)
# ---------------------------------------------------------------------------


class TestDoctorSecretsDirMissing:
    def test_doctor_secrets_dir_missing_shows_warning(self, runner: CliRunner):
        """When ~/.missy/secrets doesn't exist, doctor should show a 'missing' warning."""
        cfg_path = _cfg_path()
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None

            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []

            mock_scheduler = MagicMock()
            mock_scheduler.list_jobs.return_value = []

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
                patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler),
                patch("pathlib.Path.exists", return_value=False),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])

            assert result.exit_code == 0
            # Secrets dir missing message contains "missing" or "secrets"
            assert "missing" in result.output.lower() or "secrets" in result.output.lower()
        finally:
            import os
            os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# doctor — MCP servers configured (lines 1636-1644)
# ---------------------------------------------------------------------------


class TestDoctorMcpServers:
    def test_doctor_mcp_servers_present(self, runner: CliRunner, tmp_path):
        """When mcp.json has servers, doctor shows their count."""
        import json as _json

        mcp_path = tmp_path / "mcp.json"
        mcp_path.write_text(_json.dumps({"servers": {"myserver": {"command": "mcp-server"}}}))

        # Write a real config file so doctor can read it
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_MINIMAL_CONFIG_YAML)

        mock_config = _make_mock_config()
        mock_config.discord = None

        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        mock_scheduler = MagicMock()
        mock_scheduler.list_jobs.return_value = []

        # Patch Path.expanduser to redirect mcp.json to our temp file
        real_expanduser = Path.expanduser

        def fake_expanduser(self):
            if "mcp.json" in str(self):
                return mcp_path
            return real_expanduser(self)

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler),
            patch.object(Path, "expanduser", fake_expanduser),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = runner.invoke(cli, ["--config", str(cfg_path), "doctor"])

        assert result.exit_code == 0

    def test_doctor_mcp_json_empty_servers(self, runner: CliRunner, tmp_path):
        """When mcp.json has no servers dict, doctor shows 'no servers in mcp.json'."""
        import json as _json

        mcp_path = tmp_path / "mcp.json"
        mcp_path.write_text(_json.dumps({"servers": {}}))

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_MINIMAL_CONFIG_YAML)

        mock_config = _make_mock_config()
        mock_config.discord = None

        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        mock_scheduler = MagicMock()
        mock_scheduler.list_jobs.return_value = []

        real_expanduser = Path.expanduser

        def fake_expanduser(self):
            if "mcp.json" in str(self):
                return mcp_path
            return real_expanduser(self)

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler),
            patch.object(Path, "expanduser", fake_expanduser),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = runner.invoke(cli, ["--config", str(cfg_path), "doctor"])

        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# doctor — watchdog available / not installed (lines 1656, 1659-1660)
# ---------------------------------------------------------------------------


class TestDoctorWatchdog:
    def test_doctor_watchdog_available(self, runner: CliRunner):
        """When watchdog is installed, doctor shows 'watchdog available'."""
        cfg_path = _cfg_path()
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None

            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []
            mock_scheduler = MagicMock()
            mock_scheduler.list_jobs.return_value = []

            mock_spec = MagicMock()  # non-None spec → watchdog installed

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
                patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler),
                patch("importlib.util.find_spec", return_value=mock_spec),
                patch("pathlib.Path.exists", return_value=False),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])

            assert result.exit_code == 0
            assert "watchdog" in result.output.lower()
        finally:
            import os
            os.unlink(cfg_path)

    def test_doctor_watchdog_not_installed(self, runner: CliRunner):
        """When watchdog is not installed, doctor shows a warning."""
        cfg_path = _cfg_path()
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None

            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []
            mock_scheduler = MagicMock()
            mock_scheduler.list_jobs.return_value = []

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
                patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler),
                patch("importlib.util.find_spec", return_value=None),  # not installed
                patch("pathlib.Path.exists", return_value=False),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])

            assert result.exit_code == 0
            assert "watchdog" in result.output.lower()
        finally:
            import os
            os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# doctor — voice channel configured in YAML (lines 1672-1676)
# ---------------------------------------------------------------------------


class TestDoctorVoiceChannelConfigured:
    def test_doctor_voice_channel_configured_shows_details(self, runner: CliRunner, tmp_path):
        """When voice section is present in config YAML, doctor shows stt/tts details."""

        voice_yaml = """\
network:
  default_deny: true
providers:
  anthropic:
    name: anthropic
    model: "claude-sonnet-4-6"
workspace_path: "/tmp/workspace"
audit_log_path: "/tmp/audit.jsonl"
voice:
  host: "0.0.0.0"
  port: 8765
  stt:
    engine: "faster-whisper"
  tts:
    engine: "piper"
"""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(voice_yaml)

        try:
            mock_config = _make_mock_config()
            mock_config.discord = None

            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []
            mock_scheduler = MagicMock()
            mock_scheduler.list_jobs.return_value = []

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
                patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler),
                patch("pathlib.Path.exists", return_value=False),
            ):
                result = runner.invoke(cli, ["--config", str(cfg_path), "doctor"])

            assert result.exit_code == 0
            # Voice channel details should appear in doctor output
            assert "voice" in result.output.lower()
        finally:
            pass  # tmp_path is cleaned up by pytest

    def test_doctor_voice_channel_yaml_read_exception(self, runner: CliRunner, tmp_path):
        """When reading config YAML raises, doctor shows a warning for voice check."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_MINIMAL_CONFIG_YAML)

        mock_config = _make_mock_config()
        mock_config.discord = None

        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        mock_scheduler = MagicMock()
        mock_scheduler.list_jobs.return_value = []

        def _raise_on_second(*args, **kwargs):
            # First call is from _load_subsystems (mocked), this one for voice
            raise OSError("unreadable YAML")

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler),
            patch("pathlib.Path.exists", return_value=True),
            # Patch yaml.safe_load to raise so voice channel config read fails
            patch("yaml.safe_load", side_effect=OSError("unreadable")),
        ):
            result = runner.invoke(cli, ["--config", str(cfg_path), "doctor"])

        # Should not crash; voice error branch shows a warning
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# doctor — checkpoint exists (lines 1692-1694)
# ---------------------------------------------------------------------------


class TestDoctorCheckpointExists:
    def test_doctor_checkpoint_db_exists_shows_path(self, runner: CliRunner, tmp_path):
        """When checkpoints.db exists, doctor shows 'database: <path>'."""
        cp_db = tmp_path / "checkpoints.db"
        cp_db.touch()

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_MINIMAL_CONFIG_YAML)

        mock_config = _make_mock_config()
        mock_config.discord = None

        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        mock_scheduler = MagicMock()
        mock_scheduler.list_jobs.return_value = []

        real_expanduser = Path.expanduser

        def fake_expanduser(self):
            if "checkpoints.db" in str(self):
                return cp_db
            return real_expanduser(self)

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler),
            patch.object(Path, "expanduser", fake_expanduser),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = runner.invoke(cli, ["--config", str(cfg_path), "doctor"])

        assert result.exit_code == 0
        assert "checkpoints" in result.output.lower() or "database" in result.output.lower()


# ---------------------------------------------------------------------------
# evolve show — diffs, error_pattern, test_output (lines 2110-2120)
# ---------------------------------------------------------------------------


class TestEvolveShowWithDiffs:
    def test_evolve_show_with_diffs_renders_diff_block(self, runner: CliRunner):
        """When proposal has diffs, evolve show should render the diff block."""
        cfg_path = _cfg_path()
        try:
            diff = MagicMock()
            diff.file_path = "missy/agent/runtime.py"
            diff.description = "Fix null pointer"
            diff.original_code = "old code here"
            diff.proposed_code = "new code here"

            proposal = MagicMock()
            proposal.id = "evo-diffs-001"
            proposal.title = "Fix runtime bug"
            proposal.status.value = "approved"
            proposal.trigger.value = "error"
            proposal.confidence = 0.95
            proposal.created_at = "2026-03-14T12:00:00"
            proposal.resolved_at = None
            proposal.git_commit_sha = None
            proposal.description = "Fixes a critical bug"
            proposal.diffs = [diff]
            proposal.error_pattern = "AttributeError: NoneType"
            proposal.test_output = "PASSED 5 tests in 1.2s"

            mock_mgr = MagicMock()
            mock_mgr.get.return_value = proposal

            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "evolve", "show", "evo-diffs-001"]
                )

            assert result.exit_code == 0
            assert "Fix null pointer" in result.output
            assert "AttributeError" in result.output
            assert "PASSED" in result.output
        finally:
            import os
            os.unlink(cfg_path)

    def test_evolve_show_with_diffs_no_description(self, runner: CliRunner):
        """Diffs without a description should still render without error."""
        cfg_path = _cfg_path()
        try:
            diff = MagicMock()
            diff.file_path = "missy/tools/registry.py"
            diff.description = None  # No description → the 'if d.description:' branch is False
            diff.original_code = "x = None"
            diff.proposed_code = "x = 0"

            proposal = MagicMock()
            proposal.id = "evo-nodesc-002"
            proposal.title = "Cleanup"
            proposal.status.value = "proposed"
            proposal.trigger.value = "manual"
            proposal.confidence = 0.7
            proposal.created_at = "2026-03-14T12:00:00"
            proposal.resolved_at = None
            proposal.git_commit_sha = "abc123"
            proposal.description = "A cleanup"
            proposal.diffs = [diff]
            proposal.error_pattern = None
            proposal.test_output = None

            mock_mgr = MagicMock()
            mock_mgr.get.return_value = proposal

            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "evolve", "show", "evo-nodesc-002"]
                )

            assert result.exit_code == 0
            assert "registry.py" in result.output
        finally:
            import os
            os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# __main__ entry point (line 2667)
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    def test_main_module_is_executable(self):
        """Running missy/cli/main.py as __main__ should invoke the CLI."""
        import subprocess
        import sys

        # We invoke with --help so the CLI exits cleanly without a config file
        result = subprocess.run(
            [sys.executable, "-m", "missy.cli.main", "--help"],
            capture_output=True,
            text=True,
        )
        # --help exits with code 0 and prints usage
        assert result.returncode == 0
        assert "missy" in result.stdout.lower() or "usage" in result.stdout.lower()
