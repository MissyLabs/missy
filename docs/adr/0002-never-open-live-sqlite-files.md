# ADR-0002: Never open or close a live SQLite file outside SQLite

**Status:** Accepted, 2026-10-18 (`6219a7b`)

**Context.** `CheckpointManager` and `SQLiteMemoryStore` opened and closed the
database file with `os.open`/`os.close` on every construction to enforce mode
0600. On Linux, closing any descriptor releases every POSIX lock the process
holds on that file, including SQLite's own locks held by other connections in
the same process. A test reproduced it: after the open/close, another process
acquired a write lock this process still believed it held. `checkpoints.db`
was found corrupt (failing `quick_check` since 2026-09-28), and a parallel test
run that wrote into the live file corrupted a fresh copy within seconds.

**Decision.** Database files are created and mode-fixed only through
`missy.core.sqlite_files.ensure_private_db_file`: create exclusively with 0600
when new; otherwise only `lstat`/`chmod`, never open. Integrity checks open
the file through SQLite in read-only mode (SQLite's VFS defers closing
descriptors while locks are held). One `CheckpointManager` per runtime. Tests
run with `HOME` redirected (`tests/conftest.py`).

**Consequences.** `missy doctor` and the watchdog run `PRAGMA quick_check` on
every known store, and the retention job reports failures as errors. Any new
SQLite store must use the helper and be added to
`sqlite_files.KNOWN_DATABASES`.
