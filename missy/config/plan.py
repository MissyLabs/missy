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


def _backup_dir(path: Path | None = None, config_path: str | Path | None = None) -> Path:
    """Return the resolved backup directory path.

    Args:
        path: An explicit backup directory, if the caller has one.
        config_path: The config file this backup directory is *for*, used
            to derive a default when *path* is not given. CFGPLAN-001
            (6th tool-specific validation run): this used to default to
            the hardcoded, absolute ``~/.missy/config.d`` regardless of
            *config_path* -- so even a fully test-isolated ``tmp_path``
            config file's backup still landed in the real, live
            ``~/.missy/config.d``, silently polluting the operator's own
            config backup history with fake fixture content (and, since
            backups are pruned to a max of 5, potentially evicting
            genuine backups out of the retained window). Deriving from
            *config_path*'s own parent directory when it's provided
            preserves the exact same real-world default (a real config
            at ``~/.missy/config.yaml`` still backs up to
            ``~/.missy/config.d``) while making a differently-located
            config file (test fixture or otherwise) keep its backups
            alongside itself instead.

    Returns:
        The resolved, expanded backup directory path.
    """
    if path is not None:
        return path.expanduser()
    if config_path is not None:
        return (Path(config_path).expanduser().parent / "config.d").expanduser()
    return Path(DEFAULT_BACKUP_DIR).expanduser()


def backup_config(config_path: str | Path, backup_dir: str | Path | None = None) -> Path:
    """Create a timestamped backup of the config file.

    Args:
        config_path: Path to the current config file.
        backup_dir: Directory to store backups. Defaults to a
            ``config.d`` directory alongside *config_path* itself (see
            :func:`_backup_dir`) rather than an absolute, hardcoded
            default -- for the real ``~/.missy/config.yaml`` this is
            still ``~/.missy/config.d``, unchanged from before.

    Returns:
        Path to the newly created backup file.
    """
    src = Path(config_path).expanduser()
    dest_dir = _backup_dir(Path(backup_dir) if backup_dir else None, config_path=src)
    dest_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    # Name the backup after the *source* file, not a hardcoded "config.yaml".
    # For the real ~/.missy/config.yaml this is unchanged; for a
    # differently-named config (a migration fixture, a test tmp file) the
    # backup no longer masquerades as a config.yaml backup and no longer
    # shares one flat namespace with unrelated configs in the same dir.
    prefix = src.name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = dest_dir / f"{prefix}.{timestamp}"
    # Two backup_config() calls within the same wall-clock second (the
    # timestamp's resolution) previously produced the identical filename,
    # and shutil.copy2() overwrites an existing file with no collision
    # check -- the second call silently destroyed the first backup's
    # content, with no error raised. Disambiguate with a numeric suffix so
    # rapid successive backups (e.g. configuring two providers back-to-back,
    # or migrate_config() immediately followed by another write) are never
    # lost.
    suffix = 1
    while backup_path.exists():
        backup_path = dest_dir / f"{prefix}.{timestamp}_{suffix}"
        suffix += 1
    shutil.copy2(str(src), str(backup_path))

    _prune_backups(dest_dir, prefix=prefix)
    return backup_path


def _prune_backups(backup_dir: Path, max_keep: int = MAX_BACKUPS, prefix: str = "config.yaml") -> None:
    """Remove oldest backups (for *prefix*) so that at most *max_keep* remain."""
    backups = list_backups(backup_dir, prefix=prefix)
    while len(backups) > max_keep:
        oldest = backups.pop(0)
        oldest.unlink()


def rollback(config_path: str | Path, backup_dir: str | Path | None = None) -> Path | None:
    """Restore the latest backup, backing up the current config first.

    Args:
        config_path: Path to the current config file.
        backup_dir: Directory containing backups. Defaults to a
            ``config.d`` directory alongside *config_path* itself (see
            :func:`_backup_dir`), matching :func:`backup_config`'s
            default.

    Returns:
        Path to the restored backup, or ``None`` if no backups exist.
    """
    dest_dir = _backup_dir(Path(backup_dir) if backup_dir else None, config_path=config_path)
    backups = list_backups(dest_dir, prefix=Path(config_path).name)
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


def list_backups(
    backup_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    prefix: str | None = None,
) -> list[Path]:
    """Return all backup files sorted by true backup creation order (oldest first).

    Sorted by filename rather than ``stat().st_mtime``: ``shutil.copy2()``
    (used by :func:`backup_config`) preserves the *source* config file's
    mtime on the copy, not the time the backup was actually made -- two
    backups of an unchanged source file get identical mtimes, and even
    across edits, mtime reflects "when the config content was last
    written," not "when this backup was created." The filename's
    ``YYYYMMDD_HHMMSS[_N]`` timestamp (with the ``_N`` disambiguating
    suffix :func:`backup_config` already adds for same-second collisions)
    sorts lexicographically in true chronological order, which
    ``_prune_backups()``/:func:`rollback` rely on to correctly identify
    the oldest/latest backup.

    Args:
        backup_dir: Directory to scan (default derived from *config_path*,
            or ``~/.missy/config.d`` if that's not given either).
        config_path: The config file these backups are for, used to
            derive the default *backup_dir* the same way
            :func:`backup_config`/:func:`rollback` do (see
            :func:`_backup_dir`) when *backup_dir* isn't given directly.

    Returns:
        List of :class:`Path` objects for each backup file.
    """
    dest_dir = _backup_dir(Path(backup_dir) if backup_dir else None, config_path=config_path)
    if not dest_dir.exists():
        return []
    # Match backups for a specific source config: explicit *prefix* wins,
    # else derive it from *config_path*'s filename, else fall back to the
    # historical "config.yaml" prefix (the real config, and any pre-existing
    # backups written before source-named backups landed).
    if prefix is None:
        prefix = Path(config_path).name if config_path else "config.yaml"
    return sorted(
        (p for p in dest_dir.iterdir() if p.name.startswith(f"{prefix}.")),
        key=lambda p: p.name,
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
