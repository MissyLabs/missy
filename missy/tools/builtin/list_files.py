"""Built-in tool: list files in a directory.

Requires filesystem_read policy approval before execution.

Example::

    from missy.tools.builtin.list_files import ListFilesTool

    tool = ListFilesTool()
    result = tool.execute(path="/tmp")
    assert result.success
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

_DEFAULT_MAX_ENTRIES = 200


class ListFilesTool(BaseTool):
    """List entries in a directory within a policy-allowed read path.

    Non-recursive by default; pass ``recursive=True`` to walk the entire tree.
    Results are capped at *max_entries* to prevent runaway output on large trees.

    Attributes:
        name: ``"list_files"``
        description: One-line description for function-calling schemas.
        permissions: ``filesystem_read=True``; all other flags ``False``.
    """

    name = "list_files"
    description = (
        "List files and directories at a given path. "
        "The path must be within a policy-allowed read path."
    )
    permissions = ToolPermissions(filesystem_read=True)

    def execute(
        self,
        *,
        path: str,
        recursive: bool = False,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        **_kwargs: Any,
    ) -> ToolResult:
        """List the contents of the directory at *path*.

        Args:
            path: Absolute or ``~``-prefixed directory path.
            recursive: When ``True``, walk subdirectories with
                :meth:`~pathlib.Path.rglob` (default: ``False``).
            max_entries: Maximum number of entries to include in the output.
                A notice is appended when the real count exceeds this value
                (default: 200).

        Returns:
            :class:`~missy.tools.base.ToolResult` with:

            * ``success=True`` and ``output`` containing a formatted listing.
            * ``success=False`` and ``error`` describing the problem.
        """
        try:
            p = Path(path).expanduser().resolve(strict=False)
        except Exception as exc:
            return ToolResult(success=False, output=None, error=f"Invalid path: {exc}")

        try:
            if not p.exists():
                return ToolResult(success=False, output=None, error=f"Path not found: {path}")
            if not p.is_dir():
                return ToolResult(success=False, output=None, error=f"Not a directory: {path}")

            if recursive:
                # Sort directories before files at each level by converting the
                # path to a string for a stable, predictable order.
                all_entries = sorted(p.rglob("*"), key=lambda x: str(x))
            else:
                # Show directories first, then files, both alphabetically.
                all_entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))

            lines: list[str] = []
            for entry in all_entries[:max_entries]:
                rel = entry.relative_to(p)
                if entry.is_dir():
                    lines.append(f"[dir]  {rel}/")
                else:
                    try:
                        size = entry.stat().st_size
                        lines.append(f"[file] {rel}  ({size:,} bytes)")
                    except OSError:
                        lines.append(f"[file] {rel}")

            overflow = len(all_entries) - max_entries
            if overflow > 0:
                lines.append(f"... and {overflow:,} more entries (increase max_entries to see all)")

            output = "\n".join(lines) if lines else "(empty directory)"
            return ToolResult(success=True, output=output)
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
                        "description": "Directory path to list.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively (default: false).",
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": (
                            f"Maximum entries to return (default: {_DEFAULT_MAX_ENTRIES}). "
                            "A truncation notice is shown when the actual count exceeds this."
                        ),
                    },
                },
                "required": ["path"],
            },
        }
