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
