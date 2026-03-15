"""SQLite-backed memory store with full-text search."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in a conversation, used by SQLiteMemoryStore.

    Attributes:
        id: Unique turn identifier (UUID string).
        session_id: Identifier of the session this turn belongs to.
        timestamp: UTC timestamp as an ISO-8601 string.
        role: Speaker role — ``"user"``, ``"assistant"``, or ``"tool"``.
        content: The message content.
        provider: Name of the AI provider that generated this turn.
        metadata: Arbitrary extra data attached to the turn.
    """

    id: str
    session_id: str
    timestamp: str
    role: str  # "user" | "assistant" | "tool"
    content: str
    provider: str = ""
    metadata: dict = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        session_id: str,
        role: str,
        content: str,
        provider: str = "",
    ) -> ConversationTurn:
        """Construct a new turn with a generated id and current UTC timestamp."""
        return cls(
            id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.now(UTC).isoformat(),
            role=role,
            content=content,
            provider=provider,
        )

    def to_dict(self) -> dict:
        """Serialise the turn to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "role": self.role,
            "content": self.content,
            "provider": self.provider,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConversationTurn:
        """Deserialise a turn from a dictionary produced by :meth:`to_dict`."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            session_id=data.get("session_id", ""),
            timestamp=data.get("timestamp", ""),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            provider=data.get("provider", ""),
            metadata=data.get("metadata", {}),
        )


class SQLiteMemoryStore:
    """SQLite-backed conversation memory with FTS search.

    Provides the same interface as the original JSON :class:`MemoryStore`
    plus a full-text :meth:`search` method backed by SQLite FTS5.

    Args:
        db_path: Path to the SQLite database file.  Tilde expansion is
            performed automatically.

    Example::

        store = SQLiteMemoryStore()
        turn = ConversationTurn.new("sess-1", "user", "Hello!")
        store.add_turn(turn)
        results = store.search("Hello")
    """

    def __init__(self, db_path: str = "~/.missy/memory.db") -> None:
        self._path = Path(db_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection, creating it if needed."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        """Create tables, indexes, and FTS triggers on first use."""
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS turns (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                provider    TEXT DEFAULT '',
                metadata    TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_turns_session
                ON turns(session_id);
            CREATE INDEX IF NOT EXISTS idx_turns_timestamp
                ON turns(timestamp);

            CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                id        UNINDEXED,
                session_id UNINDEXED,
                content,
                role      UNINDEXED,
                content='turns',
                content_rowid='rowid'
            );

            CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
                INSERT INTO turns_fts(rowid, id, session_id, content)
                VALUES (new.rowid, new.id, new.session_id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
                INSERT INTO turns_fts(turns_fts, rowid, id, session_id, content)
                VALUES ('delete', old.rowid, old.id, old.session_id, old.content);
            END;

            CREATE TABLE IF NOT EXISTS learnings (
                id         TEXT PRIMARY KEY,
                task_type  TEXT,
                outcome    TEXT,
                lesson     TEXT NOT NULL,
                approach   TEXT DEFAULT '[]',
                timestamp  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                name        TEXT DEFAULT '',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                turn_count  INTEGER DEFAULT 0,
                provider    TEXT DEFAULT '',
                channel     TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(updated_at);

            CREATE TABLE IF NOT EXISTS costs (
                id                TEXT PRIMARY KEY,
                session_id        TEXT NOT NULL,
                model             TEXT NOT NULL,
                prompt_tokens     INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                cost_usd          REAL NOT NULL DEFAULT 0.0,
                timestamp         TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_costs_session
                ON costs(session_id);
            CREATE INDEX IF NOT EXISTS idx_costs_timestamp
                ON costs(timestamp);
        """)
        conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_turn(self, turn: ConversationTurn) -> None:
        """Persist a :class:`ConversationTurn` to the database.

        Uses ``INSERT OR REPLACE`` so duplicate ids are handled gracefully.

        Args:
            turn: The turn to persist.
        """
        conn = self._conn()
        conn.execute(
            """INSERT OR REPLACE INTO turns
               (id, session_id, timestamp, role, content, provider, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                turn.id,
                turn.session_id,
                turn.timestamp,
                turn.role,
                turn.content,
                turn.provider,
                json.dumps(turn.metadata),
            ),
        )
        conn.commit()

    def clear_session(self, session_id: str) -> None:
        """Delete all turns for *session_id*.

        Args:
            session_id: The session whose turns should be deleted.
        """
        conn = self._conn()
        conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_session_turns(self, session_id: str, limit: int = 100) -> list[ConversationTurn]:
        """Return up to *limit* turns for *session_id* in chronological order.

        Args:
            session_id: Session identifier to filter by.
            limit: Maximum number of turns to return.

        Returns:
            A list of :class:`ConversationTurn` objects ordered oldest-first.
        """
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        turns = [self._row_to_turn(r) for r in rows]
        turns.reverse()
        return turns

    def get_recent_turns(self, limit: int = 50) -> list[ConversationTurn]:
        """Return up to *limit* most recent turns across all sessions.

        Args:
            limit: Maximum number of turns to return.

        Returns:
            A list of :class:`ConversationTurn` objects ordered oldest-first.
        """
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM turns ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        turns = [self._row_to_turn(r) for r in rows]
        turns.reverse()
        return turns

    def search(
        self,
        query: str,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[ConversationTurn]:
        """Full-text search across conversation history.

        Uses the FTS5 ``turns_fts`` virtual table for fast ranked search.

        Args:
            query: FTS5 query string (supports prefix, phrase, and boolean
                operators, e.g. ``"python AND async"``).
            limit: Maximum number of results to return.
            session_id: When given, restrict results to this session.

        Returns:
            Matching :class:`ConversationTurn` objects ordered by relevance.
        """
        conn = self._conn()
        # Sanitize FTS5 query to prevent syntax injection.  Wrap in double
        # quotes to treat input as a phrase search and escape embedded quotes.

        safe_query = '"' + query.replace('"', '""') + '"'
        try:
            if session_id:
                rows = conn.execute(
                    """SELECT t.* FROM turns t
                       JOIN turns_fts f ON t.rowid = f.rowid
                       WHERE turns_fts MATCH ? AND t.session_id = ?
                       ORDER BY rank LIMIT ?""",
                    (safe_query, session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT t.* FROM turns t
                       JOIN turns_fts f ON t.rowid = f.rowid
                       WHERE turns_fts MATCH ?
                       ORDER BY rank LIMIT ?""",
                    (safe_query, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            logger.warning("FTS5 query failed for %r — returning empty", query[:100])
            return []
        return [self._row_to_turn(r) for r in rows]

    # ------------------------------------------------------------------
    # Learnings
    # ------------------------------------------------------------------

    def save_learning(self, learning) -> None:
        """Store a learning object in the ``learnings`` table.

        Accepts any object with ``task_type``, ``outcome``, ``lesson``,
        ``approach``, and ``timestamp`` attributes (duck-typed).

        Args:
            learning: An object carrying the learning fields.
        """
        conn = self._conn()
        conn.execute(
            """INSERT INTO learnings
               (id, task_type, outcome, lesson, approach, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                getattr(learning, "task_type", ""),
                getattr(learning, "outcome", ""),
                getattr(learning, "lesson", ""),
                json.dumps(getattr(learning, "approach", [])),
                getattr(learning, "timestamp", datetime.now(UTC).isoformat()),
            ),
        )
        conn.commit()

    def get_learnings(
        self,
        task_type: str | None = None,
        limit: int = 5,
    ) -> list[str]:
        """Retrieve recent learning lessons for context injection.

        Args:
            task_type: When given, filter to this task type only.
            limit: Maximum number of lessons to return.

        Returns:
            A list of lesson strings, most recent first.
        """
        conn = self._conn()
        if task_type:
            rows = conn.execute(
                "SELECT lesson FROM learnings WHERE task_type = ? ORDER BY timestamp DESC LIMIT ?",
                (task_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT lesson FROM learnings ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Session metadata
    # ------------------------------------------------------------------

    def register_session(
        self,
        session_id: str,
        name: str = "",
        provider: str = "",
        channel: str = "",
    ) -> None:
        """Register or update a session in the sessions table.

        Args:
            session_id: The session identifier.
            name: Optional human-friendly name.
            provider: Provider used for this session.
            channel: Channel this session originates from.
        """
        now = datetime.now(UTC).isoformat()
        conn = self._conn()
        conn.execute(
            """INSERT INTO sessions (session_id, name, created_at, updated_at, provider, channel)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                   updated_at = excluded.updated_at,
                   provider = CASE WHEN excluded.provider != '' THEN excluded.provider ELSE sessions.provider END,
                   channel = CASE WHEN excluded.channel != '' THEN excluded.channel ELSE sessions.channel END,
                   name = CASE WHEN excluded.name != '' THEN excluded.name ELSE sessions.name END""",
            (session_id, name, now, now, provider, channel),
        )
        conn.commit()

    def update_session_turn_count(self, session_id: str) -> None:
        """Recalculate the turn count for a session from the turns table."""
        conn = self._conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        now = datetime.now(UTC).isoformat()
        conn.execute(
            "UPDATE sessions SET turn_count = ?, updated_at = ? WHERE session_id = ?",
            (count, now, session_id),
        )
        conn.commit()

    def rename_session(self, session_id: str, name: str) -> bool:
        """Set a human-friendly name for a session.

        Args:
            session_id: The session to rename.
            name: The new name.

        Returns:
            True if the session was found and renamed.
        """
        conn = self._conn()
        cur = conn.execute(
            "UPDATE sessions SET name = ? WHERE session_id = ?",
            (name, session_id),
        )
        conn.commit()
        return cur.rowcount > 0

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """List sessions ordered by most recently updated.

        Returns:
            List of dicts with session_id, name, created_at, updated_at,
            turn_count, provider, channel.
        """
        conn = self._conn()
        rows = conn.execute(
            """SELECT session_id, name, created_at, updated_at,
                      turn_count, provider, channel
               FROM sessions
               ORDER BY updated_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "name": r["name"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "turn_count": r["turn_count"],
                "provider": r["provider"],
                "channel": r["channel"],
            }
            for r in rows
        ]

    def resolve_session_name(self, name: str) -> str | None:
        """Look up a session ID by its friendly name.

        Args:
            name: The friendly name to search for.

        Returns:
            The session_id if found, None otherwise.
        """
        conn = self._conn()
        row = conn.execute(
            "SELECT session_id FROM sessions WHERE name = ? LIMIT 1",
            (name,),
        ).fetchone()
        return row["session_id"] if row else None

    # ------------------------------------------------------------------
    # Cost tracking
    # ------------------------------------------------------------------

    def record_cost(
        self,
        session_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        """Persist a cost record for a provider call.

        Args:
            session_id: Session the call belongs to.
            model: Model identifier used for the call.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
            cost_usd: Computed cost in USD.
        """
        conn = self._conn()
        conn.execute(
            """INSERT INTO costs
               (id, session_id, model, prompt_tokens, completion_tokens, cost_usd, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                session_id,
                model,
                prompt_tokens,
                completion_tokens,
                cost_usd,
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()

    def get_session_costs(self, session_id: str) -> list[dict]:
        """Return all cost records for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List of dicts with model, prompt_tokens, completion_tokens,
            cost_usd, and timestamp fields.
        """
        conn = self._conn()
        rows = conn.execute(
            """SELECT model, prompt_tokens, completion_tokens, cost_usd, timestamp
               FROM costs WHERE session_id = ? ORDER BY timestamp""",
            (session_id,),
        ).fetchall()
        return [
            {
                "model": r["model"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "cost_usd": r["cost_usd"],
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

    def get_total_costs(self, limit: int = 50) -> list[dict]:
        """Return per-session cost summaries, most recent first.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of dicts with session_id, call_count, total_prompt_tokens,
            total_completion_tokens, total_cost_usd.
        """
        conn = self._conn()
        rows = conn.execute(
            """SELECT session_id,
                      COUNT(*) as call_count,
                      SUM(prompt_tokens) as total_prompt_tokens,
                      SUM(completion_tokens) as total_completion_tokens,
                      SUM(cost_usd) as total_cost_usd,
                      MAX(timestamp) as last_call
               FROM costs
               GROUP BY session_id
               ORDER BY last_call DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "call_count": r["call_count"],
                "total_prompt_tokens": r["total_prompt_tokens"],
                "total_completion_tokens": r["total_completion_tokens"],
                "total_cost_usd": r["total_cost_usd"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup(self, older_than_days: int = 30) -> int:
        """Delete turns older than *older_than_days* days.

        Args:
            older_than_days: Age threshold in days.

        Returns:
            Number of rows deleted.
        """
        conn = self._conn()
        cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()
        cur = conn.execute("DELETE FROM turns WHERE timestamp < ?", (cutoff,))
        conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_turn(row: sqlite3.Row) -> ConversationTurn:
        return ConversationTurn(
            id=row["id"],
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            role=row["role"],
            content=row["content"],
            provider=row["provider"] or "",
            metadata=json.loads(row["metadata"] or "{}"),
        )
