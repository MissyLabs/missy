"""Task checkpointing and startup recovery.

CheckpointManager writes a checkpoint to SQLite after every tool round
in the agent loop.  On startup (AgentRuntime.__init__), incomplete
checkpoints are scanned and classified for recovery.

The SQLite database is opened in WAL mode so that concurrent readers and
the single writer do not block each other.  Each thread gets its own
connection via :mod:`threading.local` to satisfy SQLite's thread-safety
requirements.

Example::

    from missy.agent.checkpoint import CheckpointManager, scan_for_recovery

    # At startup
    pending = scan_for_recovery()
    for result in pending:
        print(result.action, result.session_id, result.prompt)

    # Inside the tool loop
    cm = CheckpointManager()
    cid = cm.create("session-abc", "task-123", "Summarise this document")
    cm.update(cid, loop_messages, ["read_file"], iteration=2)
    cm.complete(cid)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field

from missy.core.events import AuditEvent, event_bus

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "~/.missy/checkpoints.db"

#: Age thresholds (in seconds) for checkpoint recovery classification.
_RESUME_THRESHOLD_SECS = 3600  # 1 hour — checkpoint is "fresh"
_RESTART_THRESHOLD_SECS = 86400  # 24 hours — checkpoint is "stale"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    state           TEXT NOT NULL DEFAULT 'RUNNING',
    loop_messages   TEXT NOT NULL DEFAULT '[]',
    tool_names_used TEXT NOT NULL DEFAULT '[]',
    iteration       INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_state ON checkpoints (state);
CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON checkpoints (session_id);
"""

# ---------------------------------------------------------------------------
# Recovery result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RecoveryResult:
    """Describes a single incomplete checkpoint and the recommended action.

    Attributes:
        checkpoint_id: Primary key of the checkpoint record.
        session_id: Session that created the checkpoint.
        prompt: Original user prompt that started the task.
        action: One of ``"resume"``, ``"restart"``, or ``"abandon"``.
        loop_messages: The serialised message history at the time of the
            last successful update.
        iteration: Loop iteration counter at the last update.
    """

    checkpoint_id: str
    session_id: str
    prompt: str
    action: str  # "resume" | "restart" | "abandon"
    loop_messages: list[dict] = field(default_factory=list)
    iteration: int = 0


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Persist and query agent task checkpoints in a local SQLite database.

    A single :class:`CheckpointManager` instance may be shared across threads.
    Each thread opens its own SQLite connection (thread-local) and WAL mode is
    enabled so that concurrent access does not cause ``database is locked``
    errors.

    Args:
        db_path: Path to the SQLite database file.  Tilde expansion is
            performed automatically.  The parent directory is created if it
            does not exist.  Defaults to ``~/.missy/checkpoints.db``.
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self.db_path = os.path.expanduser(db_path)
        self._local = threading.local()
        # Ensure the parent directory exists before any connection is opened
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True, mode=0o700)
        # Eagerly initialise the schema on the calling thread
        conn = self._connect()
        conn.executescript(_DDL)
        conn.commit()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return a thread-local WAL-mode SQLite connection.

        Creates the connection on first access for each thread and caches it
        for subsequent calls on the same thread.

        Returns:
            An open :class:`sqlite3.Connection` in WAL journal mode.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def create(self, session_id: str, task_id: str, prompt: str) -> str:
        """Insert a new RUNNING checkpoint and return its UUID.

        Args:
            session_id: The active session identifier.
            task_id: The task identifier generated by the runtime.
            prompt: The original user prompt for this task.

        Returns:
            The UUID4 string that serves as the checkpoint primary key.
        """
        checkpoint_id = str(uuid.uuid4())
        now = time.time()
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO checkpoints
                (id, session_id, task_id, prompt, state,
                 loop_messages, tool_names_used, iteration,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, 'RUNNING', '[]', '[]', 0, ?, ?)
            """,
            (checkpoint_id, session_id, task_id, prompt, now, now),
        )
        conn.commit()
        logger.debug("Checkpoint created: id=%s session=%s", checkpoint_id, session_id)
        return checkpoint_id

    def update(
        self,
        checkpoint_id: str,
        loop_messages: list[dict],
        tool_names_used: list[str],
        iteration: int,
    ) -> None:
        """Persist the current loop state for an existing checkpoint.

        Args:
            checkpoint_id: The checkpoint to update.
            loop_messages: Current message history (will be JSON-serialised).
            tool_names_used: Tool names invoked so far (JSON-serialised list).
            iteration: Current loop iteration counter.
        """
        now = time.time()
        conn = self._connect()
        conn.execute(
            """
            UPDATE checkpoints
               SET loop_messages   = ?,
                   tool_names_used = ?,
                   iteration       = ?,
                   updated_at      = ?
             WHERE id = ?
            """,
            (
                json.dumps(loop_messages),
                json.dumps(tool_names_used),
                iteration,
                now,
                checkpoint_id,
            ),
        )
        conn.commit()

    def complete(self, checkpoint_id: str) -> None:
        """Mark a checkpoint as COMPLETE.

        Args:
            checkpoint_id: The checkpoint to finalise.
        """
        self._set_state(checkpoint_id, "COMPLETE")

    def fail(self, checkpoint_id: str, error: str = "") -> None:
        """Mark a checkpoint as FAILED.

        Args:
            checkpoint_id: The checkpoint to mark failed.
            error: Optional error description stored in the prompt field as
                a suffix (not persisted separately to keep the schema minimal).
        """
        self._set_state(checkpoint_id, "FAILED")
        if error:
            # Append the error to the prompt for post-mortem visibility
            conn = self._connect()
            conn.execute(
                "UPDATE checkpoints SET prompt = prompt || ? WHERE id = ?",
                (f"\n[ERROR] {error}", checkpoint_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_incomplete(self) -> list[dict]:
        """Return all checkpoints with state ``RUNNING``.

        Returns:
            A list of dicts with keys matching the ``checkpoints`` table
            columns.  ``loop_messages`` and ``tool_names_used`` are
            already deserialised from JSON.
        """
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM checkpoints WHERE state = 'RUNNING' ORDER BY created_at"
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def classify(self, checkpoint: dict) -> str:
        """Classify an incomplete checkpoint by age.

        Args:
            checkpoint: A checkpoint dict as returned by :meth:`get_incomplete`.

        Returns:
            ``"resume"`` if the checkpoint is less than 1 hour old,
            ``"restart"`` if it is between 1 hour and 24 hours old, or
            ``"abandon"`` if it is older than 24 hours.
        """
        age = time.time() - checkpoint["created_at"]
        if age < _RESUME_THRESHOLD_SECS:
            return "resume"
        if age < _RESTART_THRESHOLD_SECS:
            return "restart"
        return "abandon"

    # ------------------------------------------------------------------
    # Bulk / maintenance operations
    # ------------------------------------------------------------------

    def abandon_old(self, max_age_seconds: int = _RESTART_THRESHOLD_SECS) -> int:
        """Set state=ABANDONED for RUNNING checkpoints older than *max_age_seconds*.

        Args:
            max_age_seconds: Age cutoff in seconds.  Checkpoints whose
                ``created_at`` timestamp is older than this are abandoned.
                Defaults to 86400 (24 hours).

        Returns:
            The number of rows updated.
        """
        cutoff = time.time() - max_age_seconds
        conn = self._connect()
        cursor = conn.execute(
            """
            UPDATE checkpoints
               SET state      = 'ABANDONED',
                   updated_at = ?
             WHERE state = 'RUNNING'
               AND created_at < ?
            """,
            (time.time(), cutoff),
        )
        conn.commit()
        return cursor.rowcount

    def delete(self, checkpoint_id: str) -> None:
        """Delete a single checkpoint record.

        Args:
            checkpoint_id: Primary key of the record to delete.
        """
        conn = self._connect()
        conn.execute("DELETE FROM checkpoints WHERE id = ?", (checkpoint_id,))
        conn.commit()

    def cleanup(self, older_than_days: int = 7) -> int:
        """Delete terminal (non-RUNNING) checkpoints older than *older_than_days*.

        Args:
            older_than_days: Records with ``updated_at`` older than this many
                days and in state COMPLETE, FAILED, or ABANDONED are deleted.
                Defaults to 7 days.

        Returns:
            The number of rows deleted.
        """
        cutoff = time.time() - older_than_days * _RESTART_THRESHOLD_SECS
        conn = self._connect()
        cursor = conn.execute(
            """
            DELETE FROM checkpoints
             WHERE state IN ('COMPLETE', 'FAILED', 'ABANDONED')
               AND updated_at < ?
            """,
            (cutoff,),
        )
        conn.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _set_state(self, checkpoint_id: str, state: str) -> None:
        conn = self._connect()
        conn.execute(
            "UPDATE checkpoints SET state = ?, updated_at = ? WHERE id = ?",
            (state, time.time(), checkpoint_id),
        )
        conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        # Deserialise JSON columns for callers' convenience
        for key in ("loop_messages", "tool_names_used"):
            raw = d.get(key, "[]")
            try:
                d[key] = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                d[key] = []
        return d


# ---------------------------------------------------------------------------
# Module-level scan function
# ---------------------------------------------------------------------------


def scan_for_recovery(db_path: str = _DEFAULT_DB_PATH) -> list[RecoveryResult]:
    """Scan for incomplete checkpoints and classify them for recovery.

    This function is intended to be called once during
    :class:`~missy.agent.runtime.AgentRuntime` initialisation to discover
    tasks that were interrupted by a crash or restart.

    The function:

    1. Instantiates a :class:`CheckpointManager` at *db_path*.
    2. Abandons any RUNNING checkpoints older than 24 hours.
    3. Fetches all remaining RUNNING checkpoints.
    4. Classifies each as ``"resume"``, ``"restart"``, or ``"abandon"``.
    5. Emits an ``"agent.checkpoint.recovery_scan"`` audit event for each.
    6. Returns the list of :class:`RecoveryResult` instances.

    Callers are responsible for deciding what to do with resume/restart
    candidates (e.g. re-queuing them or notifying the user).

    Args:
        db_path: Path to the checkpoints database.  Tilde expansion is
            performed inside :class:`CheckpointManager`.

    Returns:
        A list of :class:`RecoveryResult` objects, one per incomplete
        checkpoint.  May be empty when no incomplete checkpoints exist.
    """
    try:
        cm = CheckpointManager(db_path=db_path)
    except Exception as exc:
        logger.warning("scan_for_recovery: could not open checkpoint DB: %s", exc)
        return []

    try:
        abandoned = cm.abandon_old()
        if abandoned:
            logger.info("scan_for_recovery: abandoned %d stale checkpoint(s).", abandoned)
    except Exception as exc:
        logger.warning("scan_for_recovery: abandon_old() failed: %s", exc)

    try:
        incomplete = cm.get_incomplete()
    except Exception as exc:
        logger.warning("scan_for_recovery: get_incomplete() failed: %s", exc)
        return []

    results: list[RecoveryResult] = []
    now = time.time()

    for checkpoint in incomplete:
        action = cm.classify(checkpoint)
        age_seconds = now - checkpoint["created_at"]

        try:
            event = AuditEvent.now(
                session_id=checkpoint.get("session_id", ""),
                task_id=checkpoint.get("task_id", ""),
                event_type="agent.checkpoint.recovery_scan",
                category="plugin",
                result="allow",
                detail={
                    "action": action,
                    "session_id": checkpoint.get("session_id", ""),
                    "age_seconds": age_seconds,
                    "checkpoint_id": checkpoint.get("id", ""),
                },
            )
            event_bus.publish(event)
        except Exception as exc:
            logger.debug("scan_for_recovery: failed to emit audit event: %s", exc)

        results.append(
            RecoveryResult(
                checkpoint_id=checkpoint["id"],
                session_id=checkpoint.get("session_id", ""),
                prompt=checkpoint.get("prompt", ""),
                action=action,
                loop_messages=checkpoint.get("loop_messages", []),
                iteration=checkpoint.get("iteration", 0),
            )
        )

    return results
