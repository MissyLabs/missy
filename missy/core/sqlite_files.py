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
