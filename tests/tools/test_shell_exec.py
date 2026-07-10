"""Tests for missy.tools.builtin.shell_exec.ShellExecTool."""

from __future__ import annotations

from unittest.mock import MagicMock

from missy.tools.builtin.shell_exec import ShellExecTool

# ---------------------------------------------------------------------------
# Direct execution (no sandbox)
# ---------------------------------------------------------------------------


class TestDirectExecution:
    """Tests for ShellExecTool without sandbox."""

    def test_simple_command(self):
        tool = ShellExecTool()
        result = tool.execute(command="echo hello")
        assert result.success is True
        assert "hello" in result.output

    def test_empty_command_fails(self):
        tool = ShellExecTool()
        result = tool.execute(command="   ")
        assert result.success is False
        assert "empty" in result.error

    def test_nonexistent_command(self):
        tool = ShellExecTool()
        result = tool.execute(command="__nonexistent_binary_xyz__")
        assert result.success is False

    def test_timeout_respected(self):
        tool = ShellExecTool()
        result = tool.execute(command="sleep 60", timeout=1)
        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_cwd_parameter(self):
        tool = ShellExecTool()
        result = tool.execute(command="pwd", cwd="/tmp")
        assert result.success is True
        assert "/tmp" in result.output

    def test_timeout_capped_at_max(self):
        tool = ShellExecTool()
        # Should not raise even with absurd timeout — gets capped to 300
        result = tool.execute(command="echo ok", timeout=99999)
        assert result.success is True

    def test_background_pid_var_rejected(self):
        # $! never refers to anything real: each call is a fresh subprocess
        # with no job-control history, so a command like this would silently
        # no-op instead of killing whatever was started in a prior call.
        tool = ShellExecTool()
        result = tool.execute(command="kill $!")
        assert result.success is False
        assert "$!" in result.error
        assert "fresh" not in result.error  # sanity: not asserting exact wording elsewhere
        assert "subprocess" in result.error

    def test_background_pid_var_rejected_when_chained(self):
        # The whole point: a naive exit-code check on the compound command
        # would see success (echo masks kill's failure), so this must be
        # caught before execution, not left to the model to notice.
        tool = ShellExecTool()
        result = tool.execute(command="kill $! ; echo done")
        assert result.success is False
        assert "$!" in result.error

    def test_background_pid_var_braced_form_rejected(self):
        tool = ShellExecTool()
        result = tool.execute(command="kill ${!}")
        assert result.success is False
        assert "$!" in result.error or "${!}" in result.error

    def test_port_based_kill_still_allowed(self):
        # The recommended alternative pattern must not be caught by the
        # $! rejection — it doesn't reference the job-control variable at all.
        tool = ShellExecTool()
        result = tool.execute(command="lsof -ti:59999 | xargs -r kill")
        assert result.success is True

    def test_literal_bang_without_dollar_still_allowed(self):
        # Only the $! expansion is rejected, not any bare '!' character.
        tool = ShellExecTool()
        result = tool.execute(command="echo 'hello!'")
        assert result.success is True
        assert "hello!" in result.output


# ---------------------------------------------------------------------------
# Sandbox routing
# ---------------------------------------------------------------------------


class TestSandboxRouting:
    """Tests for ShellExecTool routing to sandbox when configured."""

    def test_no_sandbox_uses_direct(self):
        tool = ShellExecTool(sandbox_config=None)
        assert tool._sandbox is None
        result = tool.execute(command="echo direct")
        assert result.success is True
        assert "direct" in result.output

    def test_sandbox_execute_called_when_available(self):
        """When sandbox is set, execute should delegate to it."""
        mock_sandbox = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "sandboxed output"
        mock_result.error = None
        mock_sandbox.execute.return_value = mock_result

        tool = ShellExecTool()
        tool._sandbox = mock_sandbox

        result = tool.execute(command="echo test", timeout=10)
        assert result.success is True
        assert result.output == "sandboxed output"
        mock_sandbox.execute.assert_called_once_with("echo test", cwd=None, timeout=10)

    def test_sandbox_failure_returns_error(self):
        """When sandbox raises, tool returns error ToolResult."""
        mock_sandbox = MagicMock()
        mock_sandbox.execute.side_effect = RuntimeError("Docker exploded")

        tool = ShellExecTool()
        tool._sandbox = mock_sandbox

        result = tool.execute(command="echo test")
        assert result.success is False
        assert "Docker exploded" in result.error

    def test_sandbox_cwd_passed_through(self):
        mock_sandbox = MagicMock()
        mock_result = MagicMock(success=True, output="ok", error=None)
        mock_sandbox.execute.return_value = mock_result

        tool = ShellExecTool()
        tool._sandbox = mock_sandbox

        tool.execute(command="ls", cwd="/workspace")
        mock_sandbox.execute.assert_called_once_with("ls", cwd="/workspace", timeout=30)

    def test_refusing_sandbox_surfaces_failed_result(self):
        """Fail-closed: when Docker is required but unavailable the tool
        surfaces a clean failed ToolResult and does not run the command."""
        from unittest.mock import patch

        from missy.security.sandbox import SandboxConfig

        cfg = SandboxConfig(enabled=True, require_isolation=True)
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError  # docker info fails
            tool = ShellExecTool(sandbox_config=cfg)

        # The command must NOT run on the host.
        with patch("subprocess.run") as mock_run:
            result = tool.execute(command="echo test")
            mock_run.assert_not_called()
        assert result.success is False
        assert "Docker" in result.error


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_schema_has_required_fields(self):
        tool = ShellExecTool()
        schema = tool.get_schema()
        assert schema["name"] == "shell_exec"
        assert "command" in schema["parameters"]["properties"]
        assert "command" in schema["parameters"]["required"]
