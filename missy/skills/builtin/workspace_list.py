"""Built-in skill: list workspace files.

Lists files in the configured workspace directory.
Requires filesystem read access to the workspace (enforced by policy).
"""

from __future__ import annotations

import heapq
import json
import os
import stat
from pathlib import Path
from typing import Any

from missy.skills.base import BaseSkill, SkillPermissions, SkillResult, reject_unknown_arguments


class WorkspaceListSkill(BaseSkill):
    """Lists files in the workspace directory."""

    name = "workspace_list"
    description = "List files in the Missy workspace directory."
    version = "1.0.0"
    permissions = SkillPermissions(filesystem_read=True)

    def __init__(self, workspace_root: str = "~/workspace", max_entries: int = 1_000) -> None:
        self._workspace_root = Path(workspace_root).expanduser().resolve(strict=False)
        self._max_entries = max(1, min(int(max_entries), 10_000))

    def execute(self, workspace_path: str = ".", **kwargs: Any) -> SkillResult:
        """Return a bounded listing below the configured workspace root."""
        if error := reject_unknown_arguments(kwargs):
            return error
        if not isinstance(workspace_path, str):
            return SkillResult(success=False, output=None, error="Workspace path must be a string")

        requested = Path(workspace_path).expanduser()
        if requested.is_absolute():
            try:
                relative = requested.resolve(strict=False).relative_to(self._workspace_root)
            except ValueError:
                return SkillResult(
                    success=False,
                    output=None,
                    error="Workspace path is outside the configured workspace",
                )
        else:
            relative = requested
        if any(part == ".." for part in relative.parts):
            return SkillResult(
                success=False,
                output=None,
                error="Workspace path is outside the configured workspace",
            )

        directory_fd, error = self._open_directory_beneath(relative)
        if directory_fd is None:
            return SkillResult(success=False, output=None, error=error)

        display_path = self._workspace_root / relative
        try:
            with os.scandir(directory_fd) as iterator:
                entries = heapq.nsmallest(
                    self._max_entries + 1,
                    iterator,
                    key=lambda entry: (
                        0 if entry.is_dir(follow_symlinks=False) else 1,
                        entry.name,
                    ),
                )
        except OSError as exc:
            return SkillResult(success=False, output=None, error=f"Workspace listing failed: {exc}")
        finally:
            os.close(directory_fd)

        if not entries:
            return SkillResult(success=True, output=f"Workspace is empty: {display_path}")

        truncated = len(entries) > self._max_entries
        lines = []
        for entry in entries[: self._max_entries]:
            if entry.is_symlink():
                prefix = "[symlink] "
            elif entry.is_dir(follow_symlinks=False):
                prefix = "[dir] "
            else:
                prefix = "[file] "
            safe_name = json.dumps(entry.name, ensure_ascii=True)[1:-1]
            lines.append(f"{prefix}{safe_name}")
        if truncated:
            lines.append(f"[truncated] showing first {self._max_entries} entries")
        return SkillResult(success=True, output="\n".join(lines))

    def _open_directory_beneath(self, relative: Path) -> tuple[int | None, str]:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        try:
            directory_fd = os.open(self._workspace_root, flags | nofollow)
        except FileNotFoundError:
            return None, f"Workspace not found: {self._workspace_root}"
        except NotADirectoryError:
            return None, f"Not a directory: {self._workspace_root}"
        except OSError as exc:
            return None, f"Workspace cannot be opened safely: {exc}"

        try:
            for part in relative.parts:
                if part in {"", "."}:
                    continue
                try:
                    metadata = os.stat(part, dir_fd=directory_fd, follow_symlinks=False)
                except FileNotFoundError:
                    return None, f"Workspace not found: {self._workspace_root / relative}"
                if stat.S_ISLNK(metadata.st_mode):
                    return None, "Workspace symlink paths are not allowed"
                if not stat.S_ISDIR(metadata.st_mode):
                    return None, f"Not a directory: {self._workspace_root / relative}"
                next_fd = os.open(part, flags | nofollow, dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            result_fd = directory_fd
            directory_fd = -1
            return result_fd, ""
        except OSError as exc:
            return None, f"Workspace cannot be opened safely: {exc}"
        finally:
            if directory_fd >= 0:
                os.close(directory_fd)
