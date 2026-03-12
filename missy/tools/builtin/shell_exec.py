"""Built-in tool: execute a shell command.

Requires shell policy approval AND the command executable must be permitted
by the policy engine's allowed_commands list.  Commands are executed via
:mod:`subprocess` with ``shell=False`` to prevent shell injection.

Example::

    from missy.tools.builtin.shell_exec import ShellExecTool

    tool = ShellExecTool()
    result = tool.execute(command="echo hello")
    assert result.success
    assert "hello" in result.output
"""
from __future__ import annotations

import shlex
import subprocess
from typing import Any, Optional

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

_MAX_OUTPUT_BYTES = 32_768  # 32 KB
_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 300


class ShellExecTool(BaseTool):
    """Execute a whitelisted shell command as a subprocess.

    Commands are split with :func:`shlex.split` and passed directly to
    :func:`subprocess.run` with ``shell=False``, preventing shell
    metacharacter injection.  Both stdout and stderr are captured and
    combined in the result output.

    Attributes:
        name: ``"shell_exec"``
        description: One-line description for function-calling schemas.
        permissions: ``shell=True``; all other flags ``False``.
    """

    name = "shell_exec"
    description = (
        "Execute a shell command. Pass the full command string in the 'command' parameter, "
        "e.g. command='ls -la /home' or command='sudo systemctl status cups'. "
        "Supports pipes, redirection, and compound commands with && or ;. "
        "Always provide a non-empty command string."
    )
    permissions = ToolPermissions(shell=True)

    def execute(
        self,
        *,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT,
        **_kwargs: Any,
    ) -> ToolResult:
        """Run *command* as a subprocess.

        Args:
            command: The command string to execute, e.g. ``"ls -la /tmp"``.
                Parsed with :func:`shlex.split`; shell metacharacters are
                not interpreted.
            cwd: Optional working directory for the subprocess.
            timeout: Maximum wall-clock seconds before the process is killed
                (default: 30, capped at 300).

        Returns:
            :class:`~missy.tools.base.ToolResult` with:

            * ``success=True`` when the process exits with code 0.
            * ``success=False`` with ``error`` describing the exit code or
              timeout otherwise.
            * ``output`` contains combined stdout + stderr, truncated to
              32 KB when necessary.
        """
        timeout = min(int(timeout), _MAX_TIMEOUT)

        try:
            args = shlex.split(command)
        except ValueError as exc:
            return ToolResult(success=False, output=None, error=f"Invalid command syntax: {exc}")

        if not args:
            return ToolResult(success=False, output=None, error="command must not be empty")

        try:
            proc = subprocess.run(
                args,
                shell=False,
                capture_output=True,
                cwd=cwd,
                timeout=timeout,
            )
            combined: bytes = proc.stdout + proc.stderr
            if len(combined) > _MAX_OUTPUT_BYTES:
                combined = combined[:_MAX_OUTPUT_BYTES] + b"\n[Output truncated]"
            output = combined.decode("utf-8", errors="replace")
            success = proc.returncode == 0
            error = f"Exit code: {proc.returncode}" if not success else None
            return ToolResult(success=success, output=output, error=error)
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command timed out after {timeout}s",
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command not found: {args[0]!r}",
            )
        except PermissionError as exc:
            return ToolResult(success=False, output=None, error=f"Permission denied: {exc}")
        except Exception as exc:
            return ToolResult(success=False, output=None, error=str(exc))

    def get_schema(self) -> dict[str, Any]:
        """Return the JSON Schema for this tool's parameters."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute, e.g. 'ls -la /tmp'.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for the subprocess (optional).",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            f"Timeout in seconds (default: {_DEFAULT_TIMEOUT}, "
                            f"max: {_MAX_TIMEOUT})."
                        ),
                    },
                },
                "required": ["command"],
            },
        }
