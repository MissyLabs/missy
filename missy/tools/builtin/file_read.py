"""Built-in tool: read a file from the filesystem.

Requires filesystem_read policy approval before execution.

Example::

    from missy.tools.builtin.file_read import FileReadTool

    tool = FileReadTool()
    result = tool.execute(path="/etc/hostname")
    assert result.success
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

_DEFAULT_ENCODING = "utf-8"
_DEFAULT_MAX_BYTES = 65_536  # 64 KB


class FileReadTool(BaseTool):
    """Read the text contents of a file within a policy-allowed path.

    Attributes:
        name: ``"file_read"``
        description: One-line description for function-calling schemas.
        permissions: ``filesystem_read=True``; all other flags ``False``.
    """

    name = "file_read"
    description = "Read the contents of a file. The file must be within a policy-allowed read path."
    permissions = ToolPermissions(filesystem_read=True)

    def execute(
        self,
        *,
        path: str,
        encoding: str = _DEFAULT_ENCODING,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        **_kwargs: Any,
    ) -> ToolResult:
        """Read *path* and return its text content.

        Args:
            path: Absolute or ``~``-prefixed path to the file.
            encoding: Text encoding passed to :func:`open` (default: utf-8).
                Decoding errors are replaced with the Unicode replacement
                character rather than raised.
            max_bytes: Maximum number of bytes to read.  Content beyond
                this limit is dropped and a truncation notice is appended
                (default: 65536).

        Returns:
            :class:`~missy.tools.base.ToolResult` with:

            * ``success=True`` and ``output`` set to the file text on success.
            * ``success=False`` and ``error`` describing the problem on failure.
        """
        try:
            p = Path(path).expanduser()
        except Exception as exc:
            return ToolResult(success=False, output=None, error=f"Invalid path: {exc}")

        try:
            if not p.exists():
                return ToolResult(success=False, output=None, error=f"File not found: {path}")
            if not p.is_file():
                return ToolResult(success=False, output=None, error=f"Not a file: {path}")

            size = p.stat().st_size
            with p.open("r", encoding=encoding, errors="replace") as fh:
                content = fh.read(max_bytes)

            truncated = size > max_bytes
            suffix = (
                f"\n[Truncated: {size:,} bytes total, showing first {max_bytes:,}]"
                if truncated
                else ""
            )
            return ToolResult(success=True, output=content + suffix)
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
                        "description": "Path to the file to read.",
                    },
                    "encoding": {
                        "type": "string",
                        "description": f"File encoding (default: {_DEFAULT_ENCODING!r}).",
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": (
                            f"Maximum bytes to read (default: {_DEFAULT_MAX_BYTES}). "
                            "Content beyond this limit is truncated."
                        ),
                    },
                },
                "required": ["path"],
            },
        }
