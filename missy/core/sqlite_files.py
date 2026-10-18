"""Owner-only creation of SQLite database files without disturbing locks.

SQLite relies on POSIX advisory (fcntl) locks, and on Linux closing *any*
file descriptor for a file releases every such lock the process holds on it,
including locks held by other open SQLite connections in the same process.
Opening and closing a live database file just to fix its mode therefore
silently unlocks sibling connections, the documented "how to corrupt an
SQLite database" case. This helper never opens an existing file.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


def ensure_private_db_file(path: str | os.PathLike[str], *, label: str = "database") -> None:
    """Create *path* with mode 0600 if it is new; otherwise only tighten its mode.

    Raises:
        ValueError: If *path* is a symlink or not a regular file.
    """
    db_path = Path(path)
    try:
        info = os.lstat(db_path)
    except FileNotFoundError:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(db_path, flags, 0o600)
        except FileExistsError:
            # Another creator won the race; treat it as an existing file.
            info = os.lstat(db_path)
        else:
            # Nothing else can hold a lock on a file created this instant.
            os.close(fd)
            return
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"Refusing symlinked {label}: {db_path}")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Refusing non-regular {label}: {db_path}")
    if stat.S_IMODE(info.st_mode) != 0o600:
        os.chmod(db_path, 0o600)


#: SQLite stores Missy keeps under ~/.missy, checked by doctor and the watchdog.
KNOWN_DATABASES = (
    "memory.db",
    "checkpoints.db",
    "graph_memory.db",
    "request_tracker.db",
)


def quick_check(path: str | os.PathLike[str]) -> tuple[bool, str]:
    """Run a read-only ``PRAGMA quick_check`` and return ``(ok, detail)``.

    Opens the file through SQLite (read-only URI), never through a raw
    descriptor, so it cannot disturb locks held by other connections.
    """
    import sqlite3

    uri = f"file:{Path(path).as_posix()}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True, timeout=5)
        try:
            rows = con.execute("PRAGMA quick_check(1)").fetchall()
        finally:
            con.close()
    except sqlite3.Error as exc:
        return False, str(exc)
    detail = rows[0][0] if rows else "no result"
    return detail == "ok", detail.splitlines()[0] if detail else detail


def check_known_databases(state_dir: str | os.PathLike[str] = "~/.missy") -> dict[str, str]:
    """Return ``{name: "ok" | problem}`` for every known database that exists."""
    base = Path(state_dir).expanduser()
    results: dict[str, str] = {}
    for name in KNOWN_DATABASES:
        db_path = base / name
        if db_path.exists():
            ok, detail = quick_check(db_path)
            results[name] = "ok" if ok else detail
    return results
