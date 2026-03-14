"""Coverage gap tests for missy/cli/main.py.

Targets the following uncovered lines:
- Lines 1269-1276: proactive callback bodies (success and fallback paths)
- Lines 1659-1660: doctor watchdog check raises an exception
- Lines 1672-1676: doctor voice channel config present in YAML
- Lines 1693-1694: doctor checkpoint path raises an exception
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

_VOICE_CONFIG_YAML = """\
network:
  default_deny: true
providers:
  anthropic:
    name: anthropic
    model: "claude-sonnet-4-6"
workspace_path: "/tmp/workspace"
audit_log_path: "/tmp/audit.jsonl"
voice:
  host: "127.0.0.1"
  port: 9000
  stt:
    engine: "faster-whisper"
  tts:
    engine: "piper"
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


def _write_config(content: str) -> str:
    """Write config content to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        return f.name


# ---------------------------------------------------------------------------
# Lines 1268-1269: successful _proactive_callback body is invoked
# ---------------------------------------------------------------------------


class TestProactiveCallbackSuccess:
    def test_proactive_callback_success_body_is_called(self, runner: CliRunner):
        """The success-path _proactive_callback (line 1269) must be invoked.

        The callback is defined as a closure inside the try block when
        AgentRuntime is created successfully.  The only way to execute line
        1269 is to make ProactiveManager actually call the callback.  We do
        that by capturing the callback from the ProactiveManager constructor
        call and invoking it ourselves before the gateway exits.
        """
        import os

        cfg_path = _write_config(_MINIMAL_CONFIG_YAML)
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None
            mock_config.providers = {"anthropic": MagicMock()}

            proactive_cfg = MagicMock()
            proactive_cfg.enabled = True
            proactive_cfg.triggers = [MagicMock()]
            mock_config.proactive = proactive_cfg

            # Capture the agent_callback passed to ProactiveManager so we can
            # call it directly and cover line 1269.
            captured_callback: list = []

            class FakeProactiveManager:
                def __init__(self, triggers, agent_callback):
                    captured_callback.append(agent_callback)
                    self._pm = MagicMock()

                def start(self):
                    # Invoke the success-path callback so line 1269 executes.
                    if captured_callback:
                        captured_callback[0]("test prompt", "session-001")

                def stop(self):
                    pass

            mock_runtime = MagicMock()
            mock_runtime.run.return_value = "mocked response"

            call_count = [0]

            def fake_sleep(t):
                call_count[0] += 1
                if call_count[0] >= 1:
                    os.kill(os.getpid(), signal.SIGTERM)

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("missy.agent.runtime.AgentRuntime", return_value=mock_runtime),
                patch("missy.agent.runtime.AgentConfig"),
                patch(
                    "missy.agent.proactive.ProactiveManager",
                    side_effect=FakeProactiveManager,
                ),
                patch("missy.agent.proactive.ProactiveTrigger"),
                patch("time.sleep", side_effect=fake_sleep),
                patch("yaml.safe_load", return_value={}),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "start"])

            assert result.exit_code == 0
            # The callback was captured and invoked; runtime.run should have been called.
            assert mock_runtime.run.call_count >= 1
        finally:
            import os as _os

            _os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# Lines 1271-1276: fallback _proactive_callback body is invoked
# ---------------------------------------------------------------------------


class TestProactiveCallbackFallback:
    def test_proactive_callback_fallback_body_is_called(self, runner: CliRunner):
        """The fallback _proactive_callback (lines 1274-1276) executes when
        AgentRuntime construction fails.

        We make AgentRuntime raise inside the proactive try-block, which causes
        the fallback closure to be defined.  Then we capture it from the
        ProactiveManager call and invoke it so lines 1274-1276 execute.
        """
        import os

        cfg_path = _write_config(_MINIMAL_CONFIG_YAML)
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None
            mock_config.providers = {"anthropic": MagicMock()}

            proactive_cfg = MagicMock()
            proactive_cfg.enabled = True
            proactive_cfg.triggers = [MagicMock()]
            mock_config.proactive = proactive_cfg

            captured_callback: list = []

            class FakeProactiveManager:
                def __init__(self, triggers, agent_callback):
                    captured_callback.append(agent_callback)

                def start(self):
                    # Invoke the fallback callback so lines 1274-1276 execute.
                    if captured_callback:
                        result = captured_callback[0]("hello world", "session-fallback")
                        # The fallback always returns an empty string.
                        assert result == ""

                def stop(self):
                    pass

            # We need two different AgentRuntime behaviors:
            # - First call (inside proactive try) → raises so fallback callback is defined
            # - Second call (shared agent runtime) → succeeds normally
            runtime_call_count = [0]
            mock_outer_runtime = MagicMock()

            def side_effect_runtime(config):
                runtime_call_count[0] += 1
                if runtime_call_count[0] == 1:
                    # First call is inside the proactive block — make it fail.
                    raise RuntimeError("proactive runtime init failed")
                return mock_outer_runtime

            call_count = [0]

            def fake_sleep(t):
                call_count[0] += 1
                if call_count[0] >= 1:
                    os.kill(os.getpid(), signal.SIGTERM)

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("missy.agent.runtime.AgentRuntime", side_effect=side_effect_runtime),
                patch("missy.agent.runtime.AgentConfig"),
                patch(
                    "missy.agent.proactive.ProactiveManager",
                    side_effect=FakeProactiveManager,
                ),
                patch("missy.agent.proactive.ProactiveTrigger"),
                patch("time.sleep", side_effect=fake_sleep),
                patch("yaml.safe_load", return_value={}),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "start"])

            assert result.exit_code == 0
            # The fallback callback was captured and invoked.
            assert len(captured_callback) == 1
        finally:
            import os as _os

            _os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# Lines 1659-1660: doctor watchdog check raises an exception
# ---------------------------------------------------------------------------


class TestDoctorWatchdogException:
    def test_doctor_watchdog_check_exception_shows_warning(self, runner: CliRunner):
        """When importlib.util.find_spec raises inside the watchdog check,
        doctor should add 'could not check watchdog' to the table (lines 1659-1660).
        """
        cfg_path = _write_config(_MINIMAL_CONFIG_YAML)
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None

            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []
            mock_scheduler = MagicMock()
            mock_scheduler.list_jobs.return_value = []

            # find_spec raises to trigger the except branch at lines 1659-1660.
            def raise_on_watchdog(name, *args, **kwargs):
                if name == "watchdog":
                    raise RuntimeError("importlib broken")
                return None

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
                patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler),
                patch("importlib.util.find_spec", side_effect=raise_on_watchdog),
                patch("pathlib.Path.exists", return_value=False),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])

            assert result.exit_code == 0
            assert "watchdog" in result.output.lower()
            assert "could not check" in result.output.lower()
        finally:
            import os

            os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# Lines 1672-1676: doctor voice channel config present in YAML
# ---------------------------------------------------------------------------


class TestDoctorVoiceChannelYaml:
    def test_doctor_voice_config_present_shows_host_port_stt_tts(self, runner: CliRunner, tmp_path):
        """When the config YAML contains a 'voice' section, doctor reads host/port/stt/tts
        and adds a row with those details (lines 1672-1676).

        The existing test_cli_main_gaps.py::TestDoctorVoiceChannelConfigured test writes a
        config with voice keys but patches pathlib.Path.exists to False, which causes the YAML
        to not be read at all.  Here we use a real config file with voice keys and do NOT patch
        Path.exists so the YAML is actually read.
        """
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(_VOICE_CONFIG_YAML)

        mock_config = _make_mock_config()
        mock_config.discord = None

        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        mock_scheduler = MagicMock()
        mock_scheduler.list_jobs.return_value = []

        # Redirect only the mcp.json and checkpoints.db expanduser calls so that
        # those branches resolve to non-existent paths without disrupting the
        # config file read for the voice section.
        real_expanduser = Path.expanduser

        def selective_expanduser(self):
            path_str = str(self)
            if "mcp.json" in path_str or "checkpoints.db" in path_str or "secrets" in path_str:
                return tmp_path / "nonexistent_placeholder"
            return real_expanduser(self)

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler),
            patch.object(Path, "expanduser", selective_expanduser),
        ):
            result = runner.invoke(cli, ["--config", str(cfg_path), "doctor"])

        assert result.exit_code == 0
        output = result.output.lower()
        assert "voice" in output
        # The row should contain the host:port details from the YAML.
        assert "127.0.0.1" in result.output
        assert "9000" in result.output
        assert "faster-whisper" in result.output
        assert "piper" in result.output

    def test_doctor_voice_config_uses_defaults_when_keys_absent(self, runner: CliRunner, tmp_path):
        """When the voice section is present but has no host/port/stt/tts keys,
        doctor falls back to default values (lines 1672-1676 default branches).

        An empty dict ``{}`` is falsy so we supply a non-empty voice section
        that omits host/port/stt/tts, forcing each `.get()` call to use its
        default argument.
        """
        minimal_voice_yaml = """\
network:
  default_deny: true
providers:
  anthropic:
    name: anthropic
    model: "claude-sonnet-4-6"
workspace_path: "/tmp/workspace"
audit_log_path: "/tmp/audit.jsonl"
voice:
  enabled: true
"""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(minimal_voice_yaml)

        mock_config = _make_mock_config()
        mock_config.discord = None

        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = []
        mock_scheduler = MagicMock()
        mock_scheduler.list_jobs.return_value = []

        real_expanduser = Path.expanduser

        def selective_expanduser(self):
            path_str = str(self)
            if "mcp.json" in path_str or "checkpoints.db" in path_str or "secrets" in path_str:
                return tmp_path / "nonexistent_placeholder"
            return real_expanduser(self)

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_config),
            patch("missy.providers.registry.get_registry", return_value=mock_registry),
            patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler),
            patch.object(Path, "expanduser", selective_expanduser),
        ):
            result = runner.invoke(cli, ["--config", str(cfg_path), "doctor"])

        assert result.exit_code == 0
        output = result.output.lower()
        assert "voice" in output
        # Defaults: 0.0.0.0:8765
        assert "0.0.0.0" in result.output
        assert "8765" in result.output


# ---------------------------------------------------------------------------
# Lines 1693-1694: doctor checkpoint path raises an exception
# ---------------------------------------------------------------------------


class TestDoctorCheckpointException:
    def test_doctor_checkpoint_exception_is_silently_swallowed(self, runner: CliRunner):
        """When Path('~/.missy/checkpoints.db').expanduser() raises, doctor should
        silently pass (lines 1693-1694) and the command should still complete.
        """
        cfg_path = _write_config(_MINIMAL_CONFIG_YAML)
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None

            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []
            mock_scheduler = MagicMock()
            mock_scheduler.list_jobs.return_value = []

            real_expanduser = Path.expanduser

            def raise_on_checkpoints(self):
                if "checkpoints.db" in str(self):
                    raise RuntimeError("cannot expand checkpoints path")
                return real_expanduser(self)

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
                patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler),
                patch.object(Path, "expanduser", raise_on_checkpoints),
                patch("pathlib.Path.exists", return_value=False),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])

            # The exception is silently swallowed; doctor completes successfully.
            assert result.exit_code == 0
            # The exception must NOT appear in the output since the except just passes.
            assert "cannot expand checkpoints path" not in result.output
        finally:
            import os

            os.unlink(cfg_path)

    def test_doctor_checkpoint_exists_path_read_raises(self, runner: CliRunner):
        """When cp_path.exists() itself raises, lines 1693-1694 swallow the error."""
        cfg_path = _write_config(_MINIMAL_CONFIG_YAML)
        try:
            mock_config = _make_mock_config()
            mock_config.discord = None

            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []
            mock_scheduler = MagicMock()
            mock_scheduler.list_jobs.return_value = []

            real_exists = Path.exists

            def raise_on_checkpoints(self):
                if "checkpoints.db" in str(self):
                    raise OSError("permission denied on checkpoints.db")
                return real_exists(self)

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_config),
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
                patch("missy.scheduler.manager.SchedulerManager", return_value=mock_scheduler),
                patch.object(Path, "exists", raise_on_checkpoints),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])

            assert result.exit_code == 0
            assert "permission denied" not in result.output
        finally:
            import os

            os.unlink(cfg_path)
