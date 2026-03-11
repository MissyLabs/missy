"""Built-in skill: list workspace files.

Lists files in the configured workspace directory.
Requires filesystem read access to the workspace (enforced by policy).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from missy.skills.base import BaseSkill, SkillPermissions, SkillResult


class WorkspaceListSkill(BaseSkill):
    """Lists files in the workspace directory."""

    name = "workspace_list"
    description = "List files in the Missy workspace directory."
    version = "1.0.0"
    permissions = SkillPermissions(filesystem_read=True)

    def execute(self, workspace_path: str = "~/workspace", **kwargs: Any) -> SkillResult:
        """Return a newline-separated list of files in *workspace_path*."""
        path = Path(workspace_path).expanduser()
        if not path.exists():
            return SkillResult(success=False, output=None, error=f"Workspace not found: {path}")
        if not path.is_dir():
            return SkillResult(success=False, output=None, error=f"Not a directory: {path}")
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        if not entries:
            return SkillResult(success=True, output=f"Workspace is empty: {path}")
        lines = []
        for entry in entries:
            prefix = "[dir] " if entry.is_dir() else "[file] "
            lines.append(f"{prefix}{entry.name}")
        return SkillResult(success=True, output="\n".join(lines))
