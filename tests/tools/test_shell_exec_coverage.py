"""Coverage gap tests for missy.tools.builtin.shell_exec.ShellExecTool.

Targets uncovered lines:
  54-59  : sandbox_config provided but import/init raises → _sandbox stays None
  140    : output truncation when combined bytes exceed _MAX_OUTPUT_BYTES
  151-160: FileNotFoundError, PermissionError, generic Exception in _execute_direct
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from missy.tools.builtin.shell_exec import ShellExecTool, _MAX_OUTPUT_BYTES


class TestSandboxConfigImportFails:
    """Lines 54-59: sandbox_config given but get_sandbox import/call raises."""

    def test_sandbox_import_error_leaves_sandbox_none(self):
        """If importing missy.security.sandbox raises, _sandbox stays None."""
        with patch(
            "missy.tools.builtin.shell_exec.ShellExecTool.__init__",
            wraps=None,
        ):
            pass  # just to ensure import path is available

        # Patch the inner import so it raises ImportError
        with patch.dict(
            "sys.modules",
            {"missy.security.sandbox": None},  # causes ImportError on from … import
        ):
            tool = ShellExecTool(sandbox_config={"enabled": True})
            # Exception is caught silently; _sandbox should remain None
            assert tool._sandbox is None

    def test_get_sandbox_raises_leaves_sandbox_none(self):
        """If get_sandbox() itself raises, _sandbox stays None."""
        mock_module = MagicMock()
        mock_module.get_sandbox.side_effect = RuntimeError("docker not available")

        with patch.dict("sys.modules", {"missy.security.sandbox": mock_module}):
            tool = ShellExecTool(sandbox_config={"enabled": True})
            assert tool._sandbox is None


class TestOutputTruncation:
    """Line 140: combined stdout+stderr exceeds _MAX_OUTPUT_BYTES."""

    def test_large_output_is_truncated(self):
        """Output that exceeds 32 KB is truncated with a notice appended."""
        # Generate a command that produces a large byte payload by generating
        # more than _MAX_OUTPUT_BYTES worth of output via subprocess mock.
        big_bytes = b"x" * (_MAX_OUTPUT_BYTES + 1000)

        mock_proc = MagicMock()
        mock_proc.stdout = big_bytes
        mock_proc.stderr = b""
        mock_proc.returncode = 0

        with patch("subprocess.run", return_value=mock_proc):
            tool = ShellExecTool()
            result = tool._execute_direct(command="echo big", cwd=None, timeout=30)

        assert result.success is True
        # Output should contain the truncation notice
        assert "[Output truncated]" in result.output
        # The raw output length should be bounded
        assert len(result.output.encode("utf-8")) <= _MAX_OUTPUT_BYTES + 100

    def test_output_exactly_at_limit_not_truncated(self):
        """Output exactly at _MAX_OUTPUT_BYTES is not truncated."""
        exact_bytes = b"y" * _MAX_OUTPUT_BYTES

        mock_proc = MagicMock()
        mock_proc.stdout = exact_bytes
        mock_proc.stderr = b""
        mock_proc.returncode = 0

        with patch("subprocess.run", return_value=mock_proc):
            tool = ShellExecTool()
            result = tool._execute_direct(command="echo exact", cwd=None, timeout=30)

        assert result.success is True
        assert "[Output truncated]" not in result.output

    def test_stderr_included_in_truncation_check(self):
        """stdout + stderr combined exceeding limit triggers truncation."""
        half = _MAX_OUTPUT_BYTES // 2 + 500
        mock_proc = MagicMock()
        mock_proc.stdout = b"s" * half
        mock_proc.stderr = b"e" * half
        mock_proc.returncode = 0

        with patch("subprocess.run", return_value=mock_proc):
            tool = ShellExecTool()
            result = tool._execute_direct(command="cmd", cwd=None, timeout=30)

        assert result.success is True
        assert "[Output truncated]" in result.output


class TestDirectExecuteExceptions:
    """Lines 151-160: FileNotFoundError, PermissionError, generic Exception."""

    def test_file_not_found_error(self):
        """Lines 151-156: FileNotFoundError → 'Command not found'."""
        with patch("subprocess.run", side_effect=FileNotFoundError("no such file")):
            tool = ShellExecTool()
            result = tool._execute_direct(command="nonexistent_binary", cwd=None, timeout=30)

        assert result.success is False
        assert result.output is None
        assert result.error == "Command not found"

    def test_permission_error(self):
        """Lines 157-158: PermissionError → 'Permission denied: …'."""
        exc = PermissionError("access denied to /usr/sbin/example")
        with patch("subprocess.run", side_effect=exc):
            tool = ShellExecTool()
            result = tool._execute_direct(command="/usr/sbin/example", cwd=None, timeout=30)

        assert result.success is False
        assert result.output is None
        assert "Permission denied" in result.error
        assert "access denied" in result.error

    def test_generic_exception(self):
        """Lines 159-160: Any other exception → error=str(exc)."""
        exc = OSError("kernel panic")
        with patch("subprocess.run", side_effect=exc):
            tool = ShellExecTool()
            result = tool._execute_direct(command="bad", cwd=None, timeout=30)

        assert result.success is False
        assert result.output is None
        assert "kernel panic" in result.error

    def test_nonzero_exit_code_produces_error_message(self):
        """Exit code != 0 sets success=False with 'Exit code: N' error."""
        mock_proc = MagicMock()
        mock_proc.stdout = b"some output\n"
        mock_proc.stderr = b""
        mock_proc.returncode = 2

        with patch("subprocess.run", return_value=mock_proc):
            tool = ShellExecTool()
            result = tool._execute_direct(command="false", cwd=None, timeout=30)

        assert result.success is False
        assert "Exit code: 2" in result.error
        assert "some output" in result.output
