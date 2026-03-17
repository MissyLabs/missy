"""Config backup, rollback, and diff utilities.

Provides automatic backup creation before config overwrites, rollback to
the latest backup, and unified-diff comparison between config versions.
"""

from __future__ import annotations

import difflib
import shutil
import time
from pathlib import Path

DEFAULT_BACKUP_DIR = "~/.missy/config.d"
MAX_BACKUPS = 5


def _backup_dir(path: Path | None = None) -> Path:
    """Return the resolved backup directory path."""
    return (path or Path(DEFAULT_BACKUP_DIR)).expanduser()


def backup_config(config_path: str | Path, backup_dir: str | Path | None = None) -> Path:
    """Create a timestamped backup of the config file.

    Args:
        config_path: Path to the current config file.
        backup_dir: Directory to store backups (default ``~/.missy/config.d``).

    Returns:
        Path to the newly created backup file.
    """
    src = Path(config_path).expanduser()
    dest_dir = _backup_dir(Path(backup_dir) if backup_dir else None)
    dest_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = dest_dir / f"config.yaml.{timestamp}"
    shutil.copy2(str(src), str(backup_path))

    _prune_backups(dest_dir)
    return backup_path


def _prune_backups(backup_dir: Path, max_keep: int = MAX_BACKUPS) -> None:
    """Remove oldest backups so that at most *max_keep* remain."""
    backups = sorted(list_backups(backup_dir), key=lambda p: p.stat().st_mtime)
    while len(backups) > max_keep:
        oldest = backups.pop(0)
        oldest.unlink()


def rollback(config_path: str | Path, backup_dir: str | Path | None = None) -> Path | None:
    """Restore the latest backup, backing up the current config first.

    Args:
        config_path: Path to the current config file.
        backup_dir: Directory containing backups.

    Returns:
        Path to the restored backup, or ``None`` if no backups exist.
    """
    dest_dir = _backup_dir(Path(backup_dir) if backup_dir else None)
    backups = sorted(list_backups(dest_dir), key=lambda p: p.stat().st_mtime)
    if not backups:
        return None

    latest = backups[-1]
    cfg = Path(config_path).expanduser()

    # Read the content to restore before any backup operations
    restore_content = latest.read_text(encoding="utf-8")

    # Back up current config before overwriting
    if cfg.exists():
        backup_config(cfg, dest_dir)

    cfg.write_text(restore_content, encoding="utf-8")
    return latest


def list_backups(backup_dir: str | Path | None = None) -> list[Path]:
    """Return all backup files sorted by modification time (oldest first).

    Args:
        backup_dir: Directory to scan (default ``~/.missy/config.d``).

    Returns:
        List of :class:`Path` objects for each backup file.
    """
    dest_dir = _backup_dir(Path(backup_dir) if backup_dir else None)
    if not dest_dir.exists():
        return []
    return sorted(
        [p for p in dest_dir.iterdir() if p.name.startswith("config.yaml.")],
        key=lambda p: p.stat().st_mtime,
    )


def diff_configs(path_a: str | Path, path_b: str | Path) -> str:
    """Return a unified diff between two config files.

    Args:
        path_a: Path to the first config file.
        path_b: Path to the second config file.

    Returns:
        A unified diff string, or an empty string if files are identical.
    """
    a_lines = Path(path_a).expanduser().read_text(encoding="utf-8").splitlines(keepends=True)
    b_lines = Path(path_b).expanduser().read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            a_lines,
            b_lines,
            fromfile=str(path_a),
            tofile=str(path_b),
        )
    )
