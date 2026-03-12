"""Built-in tool: write content to a file.

Requires filesystem_write policy approval before execution.

Example::

    from missy.tools.builtin.file_write import FileWriteTool

    tool = FileWriteTool()
    result = tool.execute(path="/tmp/hello.txt", content="hello world\\n")
    assert result.success
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

_DEFAULT_ENCODING = "utf-8"
_VALID_MODES = frozenset({"overwrite", "append"})


class FileWriteTool(BaseTool):
    """Write text content to a file within a policy-allowed write path.

    Parent directories are created automatically when they do not exist.

    Attributes:
        name: ``"file_write"``
        description: One-line description for function-calling schemas.
        permissions: ``filesystem_write=True``; all other flags ``False``.
    """

    name = "file_write"
    description = (
        "Write content to a file, creating it and any missing parent directories "
        "if they do not exist. "
        "The file must be within a policy-allowed write path."
    )
    permissions = ToolPermissions(filesystem_write=True)

    def execute(
        self,
        *,
        path: str,
        content: str,
        mode: str = "overwrite",
        encoding: str = _DEFAULT_ENCODING,
        **_kwargs: Any,
    ) -> ToolResult:
        """Write *content* to *path*.

        Args:
            path: Absolute or ``~``-prefixed destination path.
            content: Text to write.
            mode: ``"overwrite"`` (default) replaces existing content;
                ``"append"`` adds content after any existing bytes.
            encoding: Text encoding (default: utf-8).

        Returns:
            :class:`~missy.tools.base.ToolResult` with:

            * ``success=True`` and ``output`` summarising bytes written.
            * ``success=False`` and ``error`` describing the problem.
        """
        if mode not in _VALID_MODES:
            return ToolResult(
                success=False,
                output=None,
                error=f"Invalid mode {mode!r}. Must be one of: {sorted(_VALID_MODES)}",
            )

        try:
            p = Path(path).expanduser()
        except Exception as exc:
            return ToolResult(success=False, output=None, error=f"Invalid path: {exc}")

        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            write_mode = "a" if mode == "append" else "w"
            with p.open(write_mode, encoding=encoding) as fh:
                fh.write(content)
            return ToolResult(
                success=True,
                output=f"Written {len(content):,} chars to {path}",
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
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "description": (
                            "'overwrite' replaces existing content (default); "
                            "'append' adds after existing content."
                        ),
                    },
                    "encoding": {
                        "type": "string",
                        "description": f"File encoding (default: {_DEFAULT_ENCODING!r}).",
                    },
                },
                "required": ["path", "content"],
            },
        }
