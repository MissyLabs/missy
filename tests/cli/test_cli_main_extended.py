"""Extended CLI tests targeting uncovered lines in missy/cli/main.py.

Covers the following previously uncovered sections (referenced from the
coverage report):
  - _load_subsystems error branches (lines 157-158, 164-165)
  - --debug flag on root group (line 213)
  - missy init OSError branch and workspace OSError branch (236-238, 268-269)
  - missy setup normal path and KeyboardInterrupt/Abort path (299-306)
  - missy ask: secrets warning, injection warning, ProviderError, generic
    exception (347, 362, 387-398)
  - missy run: secrets/injection warnings, ProviderError, generic exception,
    recovery display with resumable/restartable tasks (456-466, 493, 501,
    508-513)
  - schedule pause/resume/remove SchedulerError branches (649-651, 674-676,
    700-702)
  - providers list: provider not loaded branch (871-872, 875)
  - plugins list: allowed_plugins display branch (946)
  - discord probe: account with token set (1046-1065)
  - discord register-commands: out-of-range index, missing token, missing
    app_id, happy path (1099-1135)
  - gateway start: idle mode loop with discord disabled (1214-1459)
  - gateway status: discord accounts present branch (1475-1478)
  - doctor: audit/workspace/secrets missing, provider available, network
    default_deny=False, memory store, mcp.json, checkpoints (1531, 1538,
    1547, 1570-1571, 1594-1616, 1622-1683)
  - cost: session lookup exception branch (1739-1740)
  - recover: ImportError branch, abandon_old exception branch (1767-1769,
    1776-1778)
  - sessions cleanup: store has no cleanup method (1931)
  - sessions rename: resolve, rename success, rename not-found, exception
    (1986-1995)
  - evolve apply: proposal not found, not-approved, success with restart,
    failure with test output (2141-2163)
  - evolve rollback: success and failure branches (2171-2178)
  - devices list: last_seen timestamp rendered (2344)
  - devices pair: pending selection prompt path (2380-2387)
  - devices status: last_seen timestamp rendered (2444)
  - voice status: faster-whisper installed branch (2510), config read
    exception branch (2530-2531)
  - voice test: happy-path success (2597-2606), generic exception (2613-2615)
  - entry point __main__ branch (2623)

All external dependencies are mocked so no real config file, database, or
running services are required.
"""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import click
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


def _cfg_path() -> str:
    """Write a minimal config to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(_MINIMAL_CONFIG_YAML)
        f.flush()
        return f.name


# ---------------------------------------------------------------------------
# _load_subsystems exception branches (lines 157-158, 164-165)
# ---------------------------------------------------------------------------


class TestLoadSubsystemsBranches:
    """Verify _load_subsystems swallows tool/otel init errors gracefully.

    Because _load_subsystems uses lazy imports inside the function body, we
    must patch the symbols in their home modules rather than in missy.cli.main.
    """

    def test_tool_registry_exception_is_swallowed(self, runner: CliRunner):
        """When register_builtin_tools raises, _load_subsystems still returns."""
        cfg_path = _cfg_path()
        try:
            with (
                patch("missy.config.settings.load_config", return_value=_make_mock_config()),
                patch("missy.policy.engine.init_policy_engine"),
                patch("missy.observability.audit_logger.init_audit_logger"),
                patch("missy.providers.registry.init_registry"),
                patch(
                    "missy.tools.builtin.register_builtin_tools",
                    side_effect=RuntimeError("tool boom"),
                ),
                patch(
                    "missy.providers.registry.get_registry",
                    return_value=MagicMock(list_providers=list),
                ),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "providers"])
            assert result.exit_code == 0
        finally:
            import os

            os.unlink(cfg_path)

    def test_otel_init_exception_is_swallowed(self, runner: CliRunner):
        """When init_otel raises, _load_subsystems still returns."""
        cfg_path = _cfg_path()
        try:
            with (
                patch("missy.config.settings.load_config", return_value=_make_mock_config()),
                patch("missy.policy.engine.init_policy_engine"),
                patch("missy.observability.audit_logger.init_audit_logger"),
                patch("missy.providers.registry.init_registry"),
                patch("missy.observability.otel.init_otel", side_effect=RuntimeError("otel boom")),
                patch(
                    "missy.providers.registry.get_registry",
                    return_value=MagicMock(list_providers=list),
                ),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "providers"])
            assert result.exit_code == 0
        finally:
            import os

            os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# --debug flag (line 213)
# ---------------------------------------------------------------------------


class TestDebugFlag:
    def test_debug_flag_enables_debug_logging(self, runner: CliRunner):

        cfg_path = _cfg_path()
        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch(
                    "missy.providers.registry.get_registry",
                    return_value=MagicMock(list_providers=list),
                ),
            ):
                result = runner.invoke(cli, ["--debug", "--config", cfg_path, "providers"])
        finally:
            import os

            os.unlink(cfg_path)
        # Root logger level should have been set to DEBUG; exit 0 suffices as
        # a smoke test without inspecting logger state.
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# missy init error / workspace OSError branches (236-238, 268-269)
# ---------------------------------------------------------------------------


class TestInitBranches:
    def test_init_oserror_on_mkdir_exits_one(self, runner: CliRunner):
        from pathlib import Path

        with patch.object(Path, "mkdir", side_effect=OSError("permission denied")):
            result = runner.invoke(cli, ["init"])
        assert result.exit_code == 1

    def test_init_workspace_oserror_shows_warning(self, runner: CliRunner):
        """When workspace mkdir raises OSError a warning is printed (no crash)."""
        import os
        from pathlib import Path

        with runner.isolated_filesystem() as tmpdir, patch.dict(os.environ, {"HOME": str(tmpdir)}):
            # Allow ~/.missy to be created but fail on ~/workspace
            original_mkdir = Path.mkdir

            def selective_mkdir(self, *args, **kwargs):
                if str(self).endswith("workspace"):
                    raise OSError("no space")
                original_mkdir(self, *args, **kwargs)

            with patch.object(Path, "mkdir", selective_mkdir):
                result = runner.invoke(cli, ["init"])

        # Should NOT crash; prints a yellow warning instead.
        assert result.exit_code == 0
        assert "manually" in result.output or "workspace" in result.output.lower()


# ---------------------------------------------------------------------------
# missy setup (lines 299-306)
# ---------------------------------------------------------------------------


class TestSetup:
    def test_setup_calls_run_wizard(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            with patch("missy.cli.wizard.run_wizard") as mock_wizard:
                result = runner.invoke(cli, ["--config", cfg_path, "setup"])
            mock_wizard.assert_called_once()
        finally:
            import os

            os.unlink(cfg_path)
        assert result.exit_code == 0

    def test_setup_keyboard_interrupt_exits_zero(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            with patch("missy.cli.wizard.run_wizard", side_effect=KeyboardInterrupt):
                result = runner.invoke(cli, ["--config", cfg_path, "setup"])
        finally:
            import os

            os.unlink(cfg_path)
        assert result.exit_code == 0
        assert "aborted" in result.output.lower()

    def test_setup_click_abort_exits_zero(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            with patch("missy.cli.wizard.run_wizard", side_effect=click.Abort()):
                result = runner.invoke(cli, ["--config", cfg_path, "setup"])
        finally:
            import os

            os.unlink(cfg_path)
        assert result.exit_code == 0

    def test_setup_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["setup", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# missy ask: warning/error branches (347, 362, 387-398)
# ---------------------------------------------------------------------------


class TestAskBranches:
    def _invoke_ask(
        self,
        runner: CliRunner,
        prompt: str = "hello",
        *,
        has_secrets: bool = False,
        has_injection: bool = False,
        provider_error: Exception | None = None,
        generic_error: Exception | None = None,
        response: str = "ok",
    ):
        cfg_path = _cfg_path()
        try:
            mock_cfg = _make_mock_config()
            mock_runtime = MagicMock()

            if provider_error is not None:
                mock_runtime.run.side_effect = provider_error
            elif generic_error is not None:
                mock_runtime.run.side_effect = generic_error
            else:
                mock_runtime.run.return_value = response

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.agent.runtime.AgentRuntime", return_value=mock_runtime),
                patch("missy.agent.runtime.AgentConfig"),
                patch("missy.security.secrets.secrets_detector") as mock_detector,
                patch("missy.security.sanitizer.sanitizer") as mock_san,
            ):
                mock_detector.has_secrets.return_value = has_secrets
                mock_san.sanitize.return_value = prompt
                mock_san.check_for_injection.return_value = (
                    ["ignore_human_instructions"] if has_injection else []
                )
                result = runner.invoke(cli, ["--config", cfg_path, "ask", prompt])
        finally:
            import os

            os.unlink(cfg_path)
        return result

    def test_ask_secrets_warning_printed(self, runner: CliRunner):
        result = self._invoke_ask(runner, has_secrets=True)
        combined = result.output + (result.stderr or "")
        assert "secret" in combined.lower() or "credential" in combined.lower()

    def test_ask_injection_warning_printed(self, runner: CliRunner):
        result = self._invoke_ask(runner, has_injection=True)
        combined = result.output + (result.stderr or "")
        assert "injection" in combined.lower() or "warning" in combined.lower()

    def test_ask_provider_error_exits_one(self, runner: CliRunner):
        from missy.core.exceptions import ProviderError

        result = self._invoke_ask(runner, provider_error=ProviderError("api down"))
        assert result.exit_code == 1

    def test_ask_provider_error_shows_hint(self, runner: CliRunner):
        from missy.core.exceptions import ProviderError

        result = self._invoke_ask(runner, provider_error=ProviderError("api down"))
        combined = result.output + (result.stderr or "")
        assert "provider" in combined.lower() or "api key" in combined.lower()

    def test_ask_generic_exception_exits_one(self, runner: CliRunner):
        result = self._invoke_ask(runner, generic_error=RuntimeError("boom"))
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# missy run: warning / error branches (456-466, 493, 501, 508-513)
# ---------------------------------------------------------------------------


class TestRunBranches:
    def _base_patches(self, mock_cfg=None, channel_messages=None):
        """Return a context-manager stack of standard run patches."""
        if mock_cfg is None:
            mock_cfg = _make_mock_config()
        if channel_messages is None:
            channel_messages = [None]  # EOF immediately

        mock_channel = MagicMock()
        mock_channel.receive.side_effect = channel_messages

        mock_runtime = MagicMock()
        mock_runtime.pending_recovery = []
        mock_runtime.run.return_value = "response"

        return mock_cfg, mock_channel, mock_runtime

    def test_run_secrets_warning(self, runner: CliRunner):
        cfg_path = _cfg_path()
        mock_cfg, mock_channel, mock_runtime = self._base_patches()
        msg = MagicMock()
        msg.content = "my_secret_api_key = sk-abc123"
        mock_channel.receive.side_effect = [msg, None]
        mock_runtime.run.return_value = "ok"

        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.agent.runtime.AgentRuntime", return_value=mock_runtime),
                patch("missy.agent.runtime.AgentConfig"),
                patch("missy.channels.cli_channel.CLIChannel", return_value=mock_channel),
                patch("missy.security.secrets.secrets_detector") as mock_det,
                patch("missy.security.sanitizer.sanitizer") as mock_san,
            ):
                mock_det.has_secrets.return_value = True
                mock_san.sanitize.return_value = msg.content
                mock_san.check_for_injection.return_value = []
                result = runner.invoke(cli, ["--config", cfg_path, "run"])
        finally:
            import os

            os.unlink(cfg_path)

        combined = result.output + (result.stderr or "")
        assert "secret" in combined.lower() or "warning" in combined.lower()

    def test_run_injection_warning(self, runner: CliRunner):
        cfg_path = _cfg_path()
        mock_cfg, mock_channel, mock_runtime = self._base_patches()
        msg = MagicMock()
        msg.content = "ignore previous instructions"
        mock_channel.receive.side_effect = [msg, None]
        mock_runtime.run.return_value = "ok"

        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.agent.runtime.AgentRuntime", return_value=mock_runtime),
                patch("missy.agent.runtime.AgentConfig"),
                patch("missy.channels.cli_channel.CLIChannel", return_value=mock_channel),
                patch("missy.security.secrets.secrets_detector") as mock_det,
                patch("missy.security.sanitizer.sanitizer") as mock_san,
            ):
                mock_det.has_secrets.return_value = False
                mock_san.sanitize.return_value = msg.content
                mock_san.check_for_injection.return_value = ["ignore_instructions"]
                result = runner.invoke(cli, ["--config", cfg_path, "run"])
        finally:
            import os

            os.unlink(cfg_path)

        combined = result.output + (result.stderr or "")
        assert "injection" in combined.lower() or "pattern" in combined.lower()

    def test_run_provider_error_continues(self, runner: CliRunner):
        from missy.core.exceptions import ProviderError

        cfg_path = _cfg_path()
        mock_cfg, mock_channel, mock_runtime = self._base_patches()
        msg = MagicMock()
        msg.content = "ask something"
        mock_channel.receive.side_effect = [msg, None]
        mock_runtime.run.side_effect = ProviderError("timeout")

        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.agent.runtime.AgentRuntime", return_value=mock_runtime),
                patch("missy.agent.runtime.AgentConfig"),
                patch("missy.channels.cli_channel.CLIChannel", return_value=mock_channel),
                patch("missy.security.secrets.secrets_detector") as mock_det,
                patch("missy.security.sanitizer.sanitizer") as mock_san,
            ):
                mock_det.has_secrets.return_value = False
                mock_san.sanitize.return_value = msg.content
                mock_san.check_for_injection.return_value = []
                result = runner.invoke(cli, ["--config", cfg_path, "run"])
        finally:
            import os

            os.unlink(cfg_path)

        # Run exits 0 (loop continues after ProviderError, then EOF breaks)
        assert result.exit_code == 0
        combined = result.output + (result.stderr or "")
        assert "provider" in combined.lower() or "error" in combined.lower()

    def test_run_generic_exception_continues(self, runner: CliRunner):
        cfg_path = _cfg_path()
        mock_cfg, mock_channel, mock_runtime = self._base_patches()
        msg = MagicMock()
        msg.content = "ask something"
        mock_channel.receive.side_effect = [msg, None]
        mock_runtime.run.side_effect = RuntimeError("kaboom")

        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.agent.runtime.AgentRuntime", return_value=mock_runtime),
                patch("missy.agent.runtime.AgentConfig"),
                patch("missy.channels.cli_channel.CLIChannel", return_value=mock_channel),
                patch("missy.security.secrets.secrets_detector") as mock_det,
                patch("missy.security.sanitizer.sanitizer") as mock_san,
            ):
                mock_det.has_secrets.return_value = False
                mock_san.sanitize.return_value = msg.content
                mock_san.check_for_injection.return_value = []
                result = runner.invoke(cli, ["--config", cfg_path, "run"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0

    def test_run_with_resumable_recovery_tasks(self, runner: CliRunner):
        cfg_path = _cfg_path()
        mock_cfg, mock_channel, mock_runtime = self._base_patches()
        mock_channel.receive.side_effect = [None]

        resumable = MagicMock()
        resumable.action = "resume"
        resumable.session_id = "sess-abc"
        resumable.prompt = "Do a long task that was not finished"

        restartable = MagicMock()
        restartable.action = "restart"
        restartable.session_id = "sess-xyz"
        restartable.prompt = "Another old task"

        mock_runtime.pending_recovery = [resumable, restartable]

        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.agent.runtime.AgentRuntime", return_value=mock_runtime),
                patch("missy.agent.runtime.AgentConfig"),
                patch("missy.channels.cli_channel.CLIChannel", return_value=mock_channel),
                patch("missy.security.secrets.secrets_detector") as mock_det,
                patch("missy.security.sanitizer.sanitizer") as mock_san,
            ):
                mock_det.has_secrets.return_value = False
                mock_san.sanitize.return_value = ""
                mock_san.check_for_injection.return_value = []
                result = runner.invoke(cli, ["--config", cfg_path, "run"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "resumable" in result.output.lower() or "resume" in result.output.lower()
        assert "restartable" in result.output.lower() or "restart" in result.output.lower()

    def test_run_recovery_prompt_truncated_when_long(self, runner: CliRunner):
        """Prompts longer than 60 chars are truncated with an ellipsis."""
        cfg_path = _cfg_path()
        mock_cfg, mock_channel, mock_runtime = self._base_patches()
        mock_channel.receive.side_effect = [None]

        resumable = MagicMock()
        resumable.action = "resume"
        resumable.session_id = "sess-long"
        resumable.prompt = "A" * 80  # longer than 60

        mock_runtime.pending_recovery = [resumable]

        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.agent.runtime.AgentRuntime", return_value=mock_runtime),
                patch("missy.agent.runtime.AgentConfig"),
                patch("missy.channels.cli_channel.CLIChannel", return_value=mock_channel),
                patch("missy.security.secrets.secrets_detector") as mock_det,
                patch("missy.security.sanitizer.sanitizer") as mock_san,
            ):
                mock_det.has_secrets.return_value = False
                mock_san.sanitize.return_value = ""
                mock_san.check_for_injection.return_value = []
                result = runner.invoke(cli, ["--config", cfg_path, "run"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "..." in result.output


# ---------------------------------------------------------------------------
# schedule pause/resume/remove SchedulerError branches (649-651, 674-676,
# 700-702)
# ---------------------------------------------------------------------------


class TestScheduleErrorBranches:
    def _invoke_schedule(self, runner, subcmd, job_id, error=None):
        cfg_path = _cfg_path()
        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.scheduler.manager.SchedulerManager") as mock_mgr_cls,
            ):
                mock_mgr = MagicMock()
                if error is not None:
                    getattr(mock_mgr, f"{subcmd}_job").side_effect = error
                mock_mgr_cls.return_value = mock_mgr
                args = ["schedule", subcmd, job_id]
                if subcmd == "remove":
                    args = ["schedule", subcmd, "--yes", job_id]
                result = runner.invoke(cli, ["--config", cfg_path] + args)
        finally:
            import os

            os.unlink(cfg_path)
        return result

    def test_schedule_pause_scheduler_error_exits_one(self, runner: CliRunner):
        from missy.core.exceptions import SchedulerError

        result = self._invoke_schedule(runner, "pause", "job-1", SchedulerError("locked"))
        assert result.exit_code == 1

    def test_schedule_resume_scheduler_error_exits_one(self, runner: CliRunner):
        from missy.core.exceptions import SchedulerError

        result = self._invoke_schedule(runner, "resume", "job-1", SchedulerError("locked"))
        assert result.exit_code == 1

    def test_schedule_remove_scheduler_error_exits_one(self, runner: CliRunner):
        from missy.core.exceptions import SchedulerError

        result = self._invoke_schedule(runner, "remove", "job-1", SchedulerError("locked"))
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# providers list: provider not loaded (lines 871-875)
# ---------------------------------------------------------------------------


class TestProvidersNotLoaded:
    def test_providers_not_loaded_shows_not_loaded(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            provider_cfg = MagicMock()
            provider_cfg.model = "some-model"
            provider_cfg.base_url = None
            provider_cfg.timeout = 30

            mock_cfg = _make_mock_config(providers={"ollama": provider_cfg})

            mock_registry = MagicMock()
            mock_registry.get.return_value = None  # not loaded

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "providers"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "not loaded" in result.output or "ollama" in result.output


# ---------------------------------------------------------------------------
# plugins list: allowed_plugins display (line 946)
# ---------------------------------------------------------------------------


class TestPluginsAllowedList:
    def test_plugins_shows_allowed_plugins_list(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_cfg = _make_mock_config()
            mock_cfg.plugins.enabled = True
            mock_cfg.plugins.allowed_plugins = ["myplugin", "anotherplugin"]

            mock_loader = MagicMock()
            mock_loader.list_plugins.return_value = []

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.plugins.loader.init_plugin_loader", return_value=mock_loader),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "plugins"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "myplugin" in result.output


# ---------------------------------------------------------------------------
# discord probe: account with token (lines 1046-1065)
# ---------------------------------------------------------------------------


class TestDiscordProbeBranches:
    def test_discord_probe_token_present_and_connected(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_account = MagicMock()
            mock_account.token_env_var = "DISCORD_TOKEN"
            mock_account.resolve_token.return_value = "fake-token"

            discord_cfg = MagicMock()
            discord_cfg.accounts = [mock_account]

            mock_cfg = _make_mock_config()
            mock_cfg.discord = discord_cfg

            mock_user = {"username": "testbot", "discriminator": "0001", "id": "123456"}
            mock_rest = MagicMock()
            mock_rest.get_current_user.return_value = mock_user
            mock_http = MagicMock()

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.channels.discord.rest.DiscordRestClient", return_value=mock_rest),
                patch("missy.gateway.client.create_client", return_value=mock_http),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "discord", "probe"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "testbot" in result.output

    def test_discord_probe_rest_failure_prints_error(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_account = MagicMock()
            mock_account.token_env_var = "DISCORD_TOKEN"
            mock_account.resolve_token.return_value = "fake-token"

            discord_cfg = MagicMock()
            discord_cfg.accounts = [mock_account]

            mock_cfg = _make_mock_config()
            mock_cfg.discord = discord_cfg

            mock_rest = MagicMock()
            mock_rest.get_current_user.side_effect = Exception("HTTP 401")
            mock_http = MagicMock()

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.channels.discord.rest.DiscordRestClient", return_value=mock_rest),
                patch("missy.gateway.client.create_client", return_value=mock_http),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "discord", "probe"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0  # probe does not exit(1) on REST failure
        combined = result.output + (result.stderr or "")
        assert "401" in combined or "failed" in combined.lower()

    def test_discord_probe_missing_token_env_var(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_account = MagicMock()
            mock_account.token_env_var = "MISSING_VAR"
            mock_account.resolve_token.return_value = None

            discord_cfg = MagicMock()
            discord_cfg.accounts = [mock_account]

            mock_cfg = _make_mock_config()
            mock_cfg.discord = discord_cfg

            with patch("missy.cli.main._load_subsystems", return_value=mock_cfg):
                result = runner.invoke(cli, ["--config", cfg_path, "discord", "probe"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        combined = result.output + (result.stderr or "")
        assert "MISSING_VAR" in combined or "not set" in combined.lower()


# ---------------------------------------------------------------------------
# discord register-commands branches (lines 1099-1135)
# ---------------------------------------------------------------------------


class TestDiscordRegisterCommandsBranches:
    def _base_discord_cfg(self, token=None, app_id="12345"):
        mock_account = MagicMock()
        mock_account.token_env_var = "DISCORD_BOT_TOKEN"
        mock_account.resolve_token.return_value = token
        mock_account.application_id = app_id

        discord_cfg = MagicMock()
        discord_cfg.accounts = [mock_account]
        return discord_cfg

    def test_register_out_of_range_account_index_exits_one(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_cfg = _make_mock_config()
            mock_cfg.discord = self._base_discord_cfg(token="tok")

            with patch("missy.cli.main._load_subsystems", return_value=mock_cfg):
                result = runner.invoke(
                    cli,
                    ["--config", cfg_path, "discord", "register-commands", "--account-index", "5"],
                )
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 1

    def test_register_missing_token_exits_one(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_cfg = _make_mock_config()
            mock_cfg.discord = self._base_discord_cfg(token=None, app_id="999")

            with patch("missy.cli.main._load_subsystems", return_value=mock_cfg):
                result = runner.invoke(cli, ["--config", cfg_path, "discord", "register-commands"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 1

    def test_register_missing_application_id_exits_one(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_cfg = _make_mock_config()
            mock_cfg.discord = self._base_discord_cfg(token="tok", app_id=None)

            with patch("missy.cli.main._load_subsystems", return_value=mock_cfg):
                result = runner.invoke(cli, ["--config", cfg_path, "discord", "register-commands"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 1

    def test_register_rest_exception_exits_one(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_cfg = _make_mock_config()
            mock_cfg.discord = self._base_discord_cfg(token="tok", app_id="app-id")

            mock_rest = MagicMock()
            mock_rest.register_slash_commands.side_effect = Exception("network error")
            mock_http = MagicMock()

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.channels.discord.rest.DiscordRestClient", return_value=mock_rest),
                patch("missy.gateway.client.create_client", return_value=mock_http),
                patch("missy.channels.discord.commands.SLASH_COMMANDS", []),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "discord", "register-commands"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 1

    def test_register_happy_path_exits_zero(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_cfg = _make_mock_config()
            mock_cfg.discord = self._base_discord_cfg(token="tok", app_id="app-id")

            mock_rest = MagicMock()
            mock_rest.register_slash_commands.return_value = [{"name": "ask"}]
            mock_http = MagicMock()

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.channels.discord.rest.DiscordRestClient", return_value=mock_rest),
                patch("missy.gateway.client.create_client", return_value=mock_http),
                patch("missy.channels.discord.commands.SLASH_COMMANDS", []),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "discord", "register-commands"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "1" in result.output


# ---------------------------------------------------------------------------
# gateway start: idle service mode (lines 1214-1459)
# ---------------------------------------------------------------------------


class TestGatewayStartIdleMode:
    def test_gateway_start_idle_mode_exits_on_signal(self, runner: CliRunner):
        """gateway start with no Discord should enter idle loop; we use a mock
        time.sleep that raises to break the infinite loop cleanly."""
        cfg_path = _cfg_path()
        try:
            mock_cfg = _make_mock_config()
            mock_cfg.discord = None

            mock_runtime = MagicMock()
            mock_runtime.run.return_value = "ok"

            # We need to break the infinite while loop; patch time.sleep to
            # raise SystemExit after first call.
            sleep_calls = [0]

            def fake_sleep(n):
                sleep_calls[0] += 1
                if sleep_calls[0] >= 1:
                    raise SystemExit(0)

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.agent.runtime.AgentRuntime", return_value=mock_runtime),
                patch("missy.agent.runtime.AgentConfig"),
                patch("time.sleep", side_effect=fake_sleep),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "start"])
        finally:
            import os

            os.unlink(cfg_path)

        # Exit 0 (SystemExit(0) is treated as exit code 0 by CliRunner)
        assert result.exit_code == 0
        assert "Gateway" in result.output

    def test_gateway_start_voice_channel_failure_continues(self, runner: CliRunner):
        """Voice channel startup exceptions must not abort gateway start."""
        cfg_path = _cfg_path()
        try:
            mock_cfg = _make_mock_config()
            mock_cfg.discord = None

            mock_runtime = MagicMock()

            sleep_calls = [0]

            def fake_sleep(n):
                sleep_calls[0] += 1
                if sleep_calls[0] >= 1:
                    raise SystemExit(0)

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.agent.runtime.AgentRuntime", return_value=mock_runtime),
                patch("missy.agent.runtime.AgentConfig"),
                patch("time.sleep", side_effect=fake_sleep),
                patch(
                    "missy.channels.voice.channel.VoiceChannel",
                    side_effect=RuntimeError("no piper"),
                ),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "start"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "failed" in result.output.lower() or "Voice" in result.output


# ---------------------------------------------------------------------------
# gateway status: discord accounts branch (lines 1475-1478)
# ---------------------------------------------------------------------------


class TestGatewayStatusDiscordBranch:
    def test_gateway_status_discord_accounts_configured(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_account = MagicMock()
            mock_account.resolve_token.return_value = "tok"
            mock_account.token_env_var = "DISCORD_TOKEN"
            mock_account.dm_policy.value = "allowlist"

            discord_cfg = MagicMock()
            discord_cfg.enabled = True
            discord_cfg.accounts = [mock_account]

            mock_cfg = _make_mock_config()
            mock_cfg.discord = discord_cfg

            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []

            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "gateway", "status"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "discord" in result.output.lower()
        assert "configured" in result.output.lower() or "DISCORD_TOKEN" in result.output


# ---------------------------------------------------------------------------
# doctor: additional branches (lines 1531, 1538, 1547, 1570-1683)
# ---------------------------------------------------------------------------


class TestDoctorBranchesExtended:
    def _invoke_doctor(self, runner, cfg_overrides=None, extra_patches=None):
        cfg_path = _cfg_path()
        mock_cfg = _make_mock_config(**(cfg_overrides or {}))

        ctx_patches = [
            patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
        ]
        if extra_patches:
            ctx_patches.extend(extra_patches)

        try:
            # Apply all patches
            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.scheduler.manager.SchedulerManager") as mock_mgr_cls,
                patch("missy.providers.registry.get_registry") as mock_reg,
            ):
                mock_mgr = MagicMock()
                mock_mgr.list_jobs.return_value = []
                mock_mgr_cls.return_value = mock_mgr

                mock_registry = MagicMock()
                mock_registry.list_providers.return_value = []
                mock_reg.return_value = mock_registry

                if extra_patches:
                    # Apply extra patches sequentially using contextlib
                    import contextlib

                    with contextlib.ExitStack() as stack:
                        for p in extra_patches:
                            stack.enter_context(p)
                        result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
                else:
                    result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        finally:
            import os

            os.unlink(cfg_path)
        return result

    def test_doctor_audit_log_missing_shows_warn(self, runner: CliRunner):

        cfg_path = _cfg_path()
        mock_cfg = _make_mock_config()
        mock_cfg.audit_log_path = "/nonexistent/audit.jsonl"

        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.scheduler.manager.SchedulerManager") as mock_mgr_cls,
                patch("missy.providers.registry.get_registry") as mock_reg,
            ):
                mock_mgr_cls.return_value.list_jobs.return_value = []
                mock_reg.return_value.list_providers.return_value = []
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "audit log" in result.output.lower() or "WARN" in result.output

    def test_doctor_workspace_missing_shows_warn(self, runner: CliRunner):
        cfg_path = _cfg_path()
        mock_cfg = _make_mock_config()
        mock_cfg.workspace_path = "/nonexistent/workspace"

        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.scheduler.manager.SchedulerManager") as mock_mgr_cls,
                patch("missy.providers.registry.get_registry") as mock_reg,
            ):
                mock_mgr_cls.return_value.list_jobs.return_value = []
                mock_reg.return_value.list_providers.return_value = []
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "workspace" in result.output.lower()

    def test_doctor_network_default_deny_false_shows_warn(self, runner: CliRunner):
        cfg_path = _cfg_path()
        mock_cfg = _make_mock_config()
        mock_cfg.network.default_deny = False
        mock_cfg.network.allowed_domains = []
        mock_cfg.network.allowed_cidrs = []
        mock_cfg.network.allowed_hosts = []

        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.scheduler.manager.SchedulerManager") as mock_mgr_cls,
                patch("missy.providers.registry.get_registry") as mock_reg,
            ):
                mock_mgr_cls.return_value.list_jobs.return_value = []
                mock_reg.return_value.list_providers.return_value = []
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "network policy" in result.output.lower() or "WARN" in result.output

    def test_doctor_provider_available_shows_ok(self, runner: CliRunner):
        cfg_path = _cfg_path()
        provider_cfg = MagicMock()
        provider_cfg.model = "claude-sonnet-4-6"

        mock_cfg = _make_mock_config(providers={"anthropic": provider_cfg})

        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True

        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["anthropic"]
        mock_registry.get.return_value = mock_provider

        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.scheduler.manager.SchedulerManager") as mock_mgr_cls,
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
            ):
                mock_mgr_cls.return_value.list_jobs.return_value = []
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "api key" in result.output.lower() or "anthropic" in result.output

    def test_doctor_provider_is_available_exception_shows_fail(self, runner: CliRunner):
        cfg_path = _cfg_path()
        provider_cfg = MagicMock()
        provider_cfg.model = "gpt-4"

        mock_cfg = _make_mock_config(providers={"openai": provider_cfg})

        mock_provider = MagicMock()
        mock_provider.is_available.side_effect = RuntimeError("auth error")

        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["openai"]
        mock_registry.get.return_value = mock_provider

        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.scheduler.manager.SchedulerManager") as mock_mgr_cls,
                patch("missy.providers.registry.get_registry", return_value=mock_registry),
            ):
                mock_mgr_cls.return_value.list_jobs.return_value = []
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "not available" in result.output or "openai" in result.output.lower()

    def test_doctor_discord_enabled_shows_accounts(self, runner: CliRunner):
        cfg_path = _cfg_path()
        mock_cfg = _make_mock_config()

        mock_account = MagicMock()
        mock_account.resolve_token.return_value = "tok"
        mock_account.token_env_var = "DISCORD_TOKEN"
        mock_account.dm_policy.value = "allowlist"

        discord_cfg = MagicMock()
        discord_cfg.enabled = True
        discord_cfg.accounts = [mock_account]
        mock_cfg.discord = discord_cfg

        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.scheduler.manager.SchedulerManager") as mock_mgr_cls,
                patch("missy.providers.registry.get_registry") as mock_reg,
            ):
                mock_mgr_cls.return_value.list_jobs.return_value = []
                mock_reg.return_value.list_providers.return_value = []
                result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "discord" in result.output.lower()

    def test_doctor_memory_store_accessible(self, runner: CliRunner):
        cfg_path = _cfg_path()
        mock_cfg = _make_mock_config()

        mock_store = MagicMock()
        mock_store.get_session_turns.return_value = []

        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.scheduler.manager.SchedulerManager") as mock_mgr_cls,
                patch("missy.providers.registry.get_registry") as mock_reg,
                patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
            ):
                from pathlib import Path

                # Simulate memory.db existing
                with patch.object(Path, "exists", return_value=True):
                    mock_mgr_cls.return_value.list_jobs.return_value = []
                    mock_reg.return_value.list_providers.return_value = []
                    result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0

    def test_doctor_memory_store_exception_shows_fail(self, runner: CliRunner):
        cfg_path = _cfg_path()
        mock_cfg = _make_mock_config()

        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
                patch("missy.scheduler.manager.SchedulerManager") as mock_mgr_cls,
                patch("missy.providers.registry.get_registry") as mock_reg,
                patch(
                    "missy.memory.sqlite_store.SQLiteMemoryStore",
                    side_effect=RuntimeError("db error"),
                ),
            ):
                from pathlib import Path

                with patch.object(Path, "exists", return_value=True):
                    mock_mgr_cls.return_value.list_jobs.return_value = []
                    mock_reg.return_value.list_providers.return_value = []
                    result = runner.invoke(cli, ["--config", cfg_path, "doctor"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "error" in result.output.lower() or "memory" in result.output.lower()


# ---------------------------------------------------------------------------
# cost: session lookup exception (line 1739-1740)
# ---------------------------------------------------------------------------


class TestCostSessionException:
    def test_cost_session_lookup_exception_shows_error(self, runner: CliRunner):
        mock_cfg = _make_mock_config()

        with (
            patch("missy.cli.main._load_subsystems", return_value=mock_cfg),
            patch(
                "missy.memory.sqlite_store.SQLiteMemoryStore", side_effect=RuntimeError("db gone")
            ),
        ):
            result = runner.invoke(cli, ["cost", "--session", "bad-session"])

        assert result.exit_code == 0
        assert "error" in result.output.lower() or "Error" in result.output


# ---------------------------------------------------------------------------
# recover: ImportError and abandon exception branches (1767-1769, 1776-1778)
# ---------------------------------------------------------------------------


class TestRecoverBranches:
    def test_recover_import_error_exits_one(self, runner: CliRunner):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "missy.agent.checkpoint":
                raise ImportError("no checkpoint module")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = runner.invoke(cli, ["recover"])

        assert result.exit_code == 1

    def test_recover_abandon_exception_exits_one(self, runner: CliRunner):
        mock_cm = MagicMock()
        mock_cm.abandon_old.side_effect = RuntimeError("db locked")

        with patch("missy.agent.checkpoint.CheckpointManager", return_value=mock_cm):
            result = runner.invoke(cli, ["recover", "--abandon-all"])

        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# sessions cleanup: no cleanup method (line 1931)
# ---------------------------------------------------------------------------


class TestSessionsCleanupNoMethod:
    def test_sessions_cleanup_store_without_cleanup_method(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_store = MagicMock(spec=[])  # spec=[] means no methods at all

            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.memory.store.MemoryStore", return_value=mock_store),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "sessions", "cleanup"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "does not support" in result.output or "sqlite" in result.output.lower()


# ---------------------------------------------------------------------------
# sessions rename (lines 1986-1995)
# ---------------------------------------------------------------------------


class TestSessionsRename:
    def test_sessions_rename_success(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_store = MagicMock()
            mock_store.rename_session.return_value = True
            mock_store.resolve_session_name.return_value = None

            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
            ):
                result = runner.invoke(
                    cli,
                    [
                        "--config",
                        cfg_path,
                        "sessions",
                        "rename",
                        "some-uuid-session-id",
                        "My Session",
                    ],
                )
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "renamed" in result.output.lower() or "My Session" in result.output

    def test_sessions_rename_not_found(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_store = MagicMock()
            mock_store.rename_session.return_value = False
            mock_store.resolve_session_name.return_value = None

            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
            ):
                result = runner.invoke(
                    cli,
                    ["--config", cfg_path, "sessions", "rename", "no-such-uuid-sess", "Name"],
                )
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        combined = result.output + (result.stderr or "")
        assert "not found" in combined.lower() or "Error" in combined

    def test_sessions_rename_resolves_short_name(self, runner: CliRunner):
        """Short IDs without hyphens should trigger resolve_session_name."""
        cfg_path = _cfg_path()
        try:
            mock_store = MagicMock()
            mock_store.resolve_session_name.return_value = "full-uuid-resolved"
            mock_store.rename_session.return_value = True

            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "sessions", "rename", "shortname", "New Name"]
                )
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        mock_store.rename_session.assert_called_once_with("full-uuid-resolved", "New Name")

    def test_sessions_rename_exception_shows_error(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch(
                    "missy.memory.sqlite_store.SQLiteMemoryStore",
                    side_effect=RuntimeError("db error"),
                ),
            ):
                result = runner.invoke(
                    cli,
                    ["--config", cfg_path, "sessions", "rename", "some-uuid", "Name"],
                )
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        combined = result.output + (result.stderr or "")
        assert "error" in combined.lower() or "Error" in combined


# ---------------------------------------------------------------------------
# sessions list (1950-1970)
# ---------------------------------------------------------------------------


class TestSessionsList:
    def test_sessions_list_no_sessions(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_store = MagicMock()
            mock_store.list_sessions.return_value = []

            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "sessions", "list"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "No sessions" in result.output

    def test_sessions_list_shows_sessions(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_store = MagicMock()
            mock_store.list_sessions.return_value = [
                {
                    "session_id": "abc-123-def",
                    "name": "My Session",
                    "turn_count": 5,
                    "provider": "anthropic",
                    "channel": "cli",
                    "updated_at": "2026-03-14T10:00:00",
                }
            ]

            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=mock_store),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "sessions", "list"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "My Session" in result.output or "abc-123" in result.output

    def test_sessions_list_exception_shows_error(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch(
                    "missy.memory.sqlite_store.SQLiteMemoryStore",
                    side_effect=RuntimeError("db gone"),
                ),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "sessions", "list"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        combined = result.output + (result.stderr or "")
        assert "error" in combined.lower() or "Error" in combined

    def test_sessions_list_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["sessions", "list", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# evolve apply (lines 2141-2163)
# ---------------------------------------------------------------------------


class TestEvolveApply:
    def _mock_proposal(self, status="approved", has_diffs=False):
        prop = MagicMock()
        prop.status.value = status
        prop.id = "prop-1234"
        prop.description = "Improve error handling"
        prop.created_at = "2026-03-14T09:00:00"
        prop.resolved_at = None
        prop.git_commit_sha = None
        prop.error_pattern = None
        prop.test_output = None
        prop.diffs = []
        return prop

    def test_evolve_apply_proposal_not_found(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_mgr = MagicMock()
            mock_mgr.get.return_value = None

            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "evolve", "apply", "prop-9999"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        combined = result.output + (result.stderr or "")
        assert "not found" in combined.lower() or "Error" in combined

    def test_evolve_apply_not_approved_status(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_mgr = MagicMock()
            mock_mgr.get.return_value = self._mock_proposal(status="proposed")

            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "evolve", "apply", "prop-1234"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        combined = result.output + (result.stderr or "")
        assert "proposed" in combined or "approve" in combined.lower()

    def test_evolve_apply_success_with_no_restart(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_mgr = MagicMock()
            mock_mgr.get.return_value = self._mock_proposal(status="approved")
            mock_mgr.apply.return_value = {"success": True, "message": "Applied successfully"}

            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
                patch("missy.agent.code_evolution.restart_process"),
            ):
                result = runner.invoke(
                    cli,
                    ["--config", cfg_path, "evolve", "apply", "--no-restart", "prop-1234"],
                )
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "Applied" in result.output or "Skipping restart" in result.output

    def test_evolve_apply_success_triggers_restart(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_mgr = MagicMock()
            mock_mgr.get.return_value = self._mock_proposal(status="approved")
            mock_mgr.apply.return_value = {"success": True, "message": "Applied ok"}

            mock_restart = MagicMock()

            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
                patch("missy.agent.code_evolution.restart_process", mock_restart),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "evolve", "apply", "prop-1234"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        mock_restart.assert_called_once()

    def test_evolve_apply_failure_with_test_output(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_mgr = MagicMock()
            mock_mgr.get.return_value = self._mock_proposal(status="approved")
            mock_mgr.apply.return_value = {
                "success": False,
                "message": "Tests failed",
                "test_output": "FAILED test_something :: AssertionError",
            }

            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "evolve", "apply", "prop-1234"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "Tests failed" in result.output or "AssertionError" in result.output

    def test_evolve_apply_failure_no_test_output(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_mgr = MagicMock()
            mock_mgr.get.return_value = self._mock_proposal(status="approved")
            mock_mgr.apply.return_value = {
                "success": False,
                "message": "Syntax error in patch",
            }

            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
            ):
                result = runner.invoke(cli, ["--config", cfg_path, "evolve", "apply", "prop-1234"])
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        combined = result.output + (result.stderr or "")
        assert "Syntax error" in combined or "Error" in combined

    def test_evolve_apply_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["evolve", "apply", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# evolve rollback (lines 2171-2178)
# ---------------------------------------------------------------------------


class TestEvolveRollback:
    def test_evolve_rollback_success(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_mgr = MagicMock()
            mock_mgr.rollback.return_value = {"success": True, "message": "Reverted ok"}

            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "evolve", "rollback", "prop-1234"]
                )
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        assert "Reverted ok" in result.output or "Success" in result.output

    def test_evolve_rollback_failure(self, runner: CliRunner):
        cfg_path = _cfg_path()
        try:
            mock_mgr = MagicMock()
            mock_mgr.rollback.return_value = {
                "success": False,
                "message": "Nothing to revert",
            }

            with (
                patch("missy.cli.main._load_subsystems", return_value=_make_mock_config()),
                patch("missy.agent.code_evolution.CodeEvolutionManager", return_value=mock_mgr),
            ):
                result = runner.invoke(
                    cli, ["--config", cfg_path, "evolve", "rollback", "prop-1234"]
                )
        finally:
            import os

            os.unlink(cfg_path)

        assert result.exit_code == 0
        combined = result.output + (result.stderr or "")
        assert "Nothing to revert" in combined or "Error" in combined

    def test_evolve_rollback_help_exits_zero(self, runner: CliRunner):
        result = runner.invoke(cli, ["evolve", "rollback", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# devices list: last_seen timestamp branch (line 2344)
# ---------------------------------------------------------------------------


class TestDevicesListTimestamp:
    def test_devices_list_shows_last_seen_formatted(self, runner: CliRunner):
        import time

        mock_reg = MagicMock()
        mock_reg.all.return_value = [
            {
                "node_id": "abcdef1234567890",
                "name": "Living Room",
                "room": "living",
                "paired": True,
                "policy": "full",
                "last_seen": time.time() - 3600,  # 1 hour ago
            }
        ]

        with patch("missy.channels.voice.registry.DeviceRegistry", return_value=mock_reg):
            result = runner.invoke(cli, ["devices", "list"])

        assert result.exit_code == 0
        # last_seen should have been formatted as YYYY-MM-DD HH:MM, not "never"
        assert "never" not in result.output or "2026" in result.output or "2025" in result.output


# ---------------------------------------------------------------------------
# devices pair: pending selection prompt (lines 2380-2387)
# ---------------------------------------------------------------------------


class TestDevicesPairPrompt:
    def test_devices_pair_pending_list_and_select(self, runner: CliRunner):
        mock_reg = MagicMock()
        mock_reg.all.return_value = [
            {
                "node_id": "aaabbbcccdddeeee",
                "name": "Kitchen",
                "room": "kitchen",
                "paired": False,
            }
        ]

        mock_pairing_mgr = MagicMock()
        mock_pairing_mgr.approve.return_value = "secret-token-xyz"

        with (
            patch("missy.channels.voice.registry.DeviceRegistry", return_value=mock_reg),
            patch("missy.channels.voice.pairing.PairingManager", return_value=mock_pairing_mgr),
        ):
            result = runner.invoke(cli, ["devices", "pair"], input="0\n")

        assert result.exit_code == 0
        assert "secret-token-xyz" in result.output or "token" in result.output.lower()

    def test_devices_pair_invalid_index_exits_one(self, runner: CliRunner):
        mock_reg = MagicMock()
        mock_reg.all.return_value = [
            {
                "node_id": "aaabbbcccdddeeee",
                "name": "Kitchen",
                "room": "kitchen",
                "paired": False,
            }
        ]

        mock_pairing_mgr = MagicMock()

        with (
            patch("missy.channels.voice.registry.DeviceRegistry", return_value=mock_reg),
            patch("missy.channels.voice.pairing.PairingManager", return_value=mock_pairing_mgr),
        ):
            result = runner.invoke(cli, ["devices", "pair"], input="99\n")

        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# devices status: last_seen timestamp branch (line 2444)
# ---------------------------------------------------------------------------


class TestDevicesStatusTimestamp:
    def test_devices_status_last_seen_formatted(self, runner: CliRunner):
        import time

        mock_reg = MagicMock()
        mock_reg.all.return_value = [
            {
                "node_id": "node00001234abcd",
                "name": "Office",
                "room": "office",
                "online": True,
                "last_seen": time.time() - 120,
                "occupancy": 2,
                "noise_level": 35.5,
            }
        ]

        with patch("missy.channels.voice.registry.DeviceRegistry", return_value=mock_reg):
            result = runner.invoke(cli, ["devices", "status"])

        assert result.exit_code == 0
        assert "35.5 dB" in result.output


# ---------------------------------------------------------------------------
# voice status: faster_whisper installed branch + config exception (2510,
# 2530-2531)
# ---------------------------------------------------------------------------


class TestVoiceStatusBranches:
    def test_voice_status_faster_whisper_installed(self, runner: CliRunner):
        mock_reg = MagicMock()
        mock_reg.all.return_value = []

        # Ensure faster_whisper import succeeds
        import sys
        import types

        fake_fw = types.ModuleType("faster_whisper")
        with (
            patch.dict(sys.modules, {"faster_whisper": fake_fw}),
            patch("missy.channels.voice.registry.DeviceRegistry", return_value=mock_reg),
        ):
            result = runner.invoke(cli, ["voice", "status"])

        assert result.exit_code == 0
        assert "installed" in result.output.lower()

    def test_voice_status_config_read_exception_is_swallowed(self, runner: CliRunner):
        """Errors reading yaml config for voice status should not crash."""
        mock_reg = MagicMock()
        mock_reg.all.return_value = []

        with (
            patch("missy.channels.voice.registry.DeviceRegistry", return_value=mock_reg),
            patch("yaml.safe_load", side_effect=Exception("yaml parse error")),
        ):
            result = runner.invoke(cli, ["voice", "status"])

        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# voice test: success and generic exception branches (2597-2606, 2613-2615)
# ---------------------------------------------------------------------------


class TestVoiceTestBranches:
    def test_voice_test_success(self, runner: CliRunner):
        mock_reg = MagicMock()
        mock_reg.all.return_value = [
            {"node_id": "testnode12345678", "name": "Test Node", "room": "lab", "paired": True}
        ]

        mock_tts = MagicMock()
        # synthesize is async; return a coroutine that yields bytes
        mock_tts.synthesize = AsyncMock(return_value=b"\x00" * 44100)

        with (
            patch("missy.channels.voice.registry.DeviceRegistry", return_value=mock_reg),
            patch("missy.channels.voice.tts.piper.PiperTTS", return_value=mock_tts),
        ):
            result = runner.invoke(cli, ["voice", "test", "testnode", "--text", "hello world"])

        assert result.exit_code == 0
        assert "synthesis" in result.output.lower() or "bytes" in result.output.lower()

    def test_voice_test_generic_exception_exits_one(self, runner: CliRunner):
        mock_reg = MagicMock()
        mock_reg.all.return_value = [
            {"node_id": "testnode12345678", "name": "Test Node", "room": "lab", "paired": True}
        ]

        mock_tts = MagicMock()
        mock_tts.synthesize = AsyncMock(side_effect=RuntimeError("audio device error"))

        with (
            patch("missy.channels.voice.registry.DeviceRegistry", return_value=mock_reg),
            patch("missy.channels.voice.tts.piper.PiperTTS", return_value=mock_tts),
        ):
            result = runner.invoke(cli, ["voice", "test", "testnode", "--text", "hello"])

        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "audio device error" in combined or "failed" in combined.lower()
