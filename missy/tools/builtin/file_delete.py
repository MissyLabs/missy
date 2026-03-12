"""Built-in tool: delete a file.

Requires filesystem_write policy approval before execution.
Directory deletion is intentionally not supported to prevent accidental
recursive removal.

Example::

    from missy.tools.builtin.file_delete import FileDeleteTool

    tool = FileDeleteTool()
    result = tool.execute(path="/tmp/scratch.txt")
    assert result.success
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult


class FileDeleteTool(BaseTool):
    """Delete a single file within a policy-allowed write path.

    Only regular files may be deleted; directories are explicitly rejected
    to prevent accidental recursive removal.

    Attributes:
        name: ``"file_delete"``
        description: One-line description for function-calling schemas.
        permissions: ``filesystem_write=True``; all other flags ``False``.
    """

    name = "file_delete"
    description = (
        "Delete a file. "
        "Only regular files are supported; directories will not be deleted. "
        "The file must be within a policy-allowed write path."
    )
    permissions = ToolPermissions(filesystem_write=True)

    def execute(self, *, path: str, **_kwargs: Any) -> ToolResult:
        """Delete the file at *path*.

        Args:
            path: Absolute or ``~``-prefixed path to the file to delete.

        Returns:
            :class:`~missy.tools.base.ToolResult` with:

            * ``success=True`` and ``output`` confirming deletion.
            * ``success=False`` and ``error`` describing the problem.
        """
        try:
            p = Path(path).expanduser()
        except Exception as exc:
            return ToolResult(success=False, output=None, error=f"Invalid path: {exc}")

        try:
            if not p.exists():
                return ToolResult(success=False, output=None, error=f"File not found: {path}")
            if not p.is_file():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Not a file (directories will not be deleted): {path}",
                )
            p.unlink()
            return ToolResult(success=True, output=f"Deleted: {path}")
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
                    "path": {
                        "type": "string",
                        "description": "Path to the file to delete.",
                    },
                },
                "required": ["path"],
            },
        }
