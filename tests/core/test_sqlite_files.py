"""ensure_private_db_file must never release SQLite's POSIX locks (premortem item 2)."""

from __future__ import annotations

import os
import sqlite3
import stat
import subprocess
import sys

import pytest

from missy.core.sqlite_files import ensure_private_db_file

_TRY_WRITE_LOCK = """
import sqlite3, sys
con = sqlite3.connect(sys.argv[1], timeout=0)
try:
    con.execute("BEGIN IMMEDIATE")
except sqlite3.OperationalError as exc:
    print("locked" if "locked" in str(exc) else f"error: {exc}")
else:
    print("acquired")
"""


def _other_process_can_write(path: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _TRY_WRITE_LOCK, path],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout.strip()


def test_new_file_is_created_owner_only(tmp_path) -> None:
    db = tmp_path / "new.db"
    ensure_private_db_file(db)
    assert stat.S_IMODE(os.stat(db).st_mode) == 0o600


def test_existing_file_mode_is_tightened(tmp_path) -> None:
    db = tmp_path / "loose.db"
    db.write_bytes(b"")
    os.chmod(db, 0o644)
    ensure_private_db_file(db)
    assert stat.S_IMODE(os.stat(db).st_mode) == 0o600


def test_symlink_is_refused(tmp_path) -> None:
    target = tmp_path / "real.db"
    target.write_bytes(b"")
    link = tmp_path / "link.db"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlinked"):
        ensure_private_db_file(link)


def test_existing_database_keeps_its_write_lock(tmp_path) -> None:
    db = str(tmp_path / "live.db")
    holder = sqlite3.connect(db)
    holder.execute("CREATE TABLE t (x)")
    holder.commit()
    holder.execute("BEGIN IMMEDIATE")  # this process now holds the write lock
    assert _other_process_can_write(db) == "locked"

    ensure_private_db_file(db)

    # Opening and closing the file here used to drop the lock silently.
    assert _other_process_can_write(db) == "locked"
    holder.rollback()
    holder.close()
