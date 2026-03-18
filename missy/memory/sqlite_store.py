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


@dataclass
class SummaryRecord:
    """A DAG node summarising a chunk of conversation history.

    Leaf summaries (depth 0) compress raw turns.  Condensed summaries
    (depth 1+) compress groups of same-depth summaries.
    """

    id: str
    session_id: str
    depth: int
    content: str
    token_estimate: int = 0
    source_turn_ids: list[str] = field(default_factory=list)
    source_summary_ids: list[str] = field(default_factory=list)
    parent_id: str | None = None
    time_range_start: str | None = None
    time_range_end: str | None = None
    descendant_count: int = 0
    file_refs: list[str] = field(default_factory=list)
    created_at: str = ""

    @classmethod
    def new(
        cls,
        session_id: str,
        depth: int,
        content: str,
        *,
        token_estimate: int = 0,
        source_turn_ids: list[str] | None = None,
        source_summary_ids: list[str] | None = None,
        time_range_start: str | None = None,
        time_range_end: str | None = None,
        descendant_count: int = 0,
    ) -> SummaryRecord:
        return cls(
            id=f"sum_{uuid.uuid4().hex[:16]}",
            session_id=session_id,
            depth=depth,
            content=content,
            token_estimate=token_estimate or max(1, len(content) // 4),
            source_turn_ids=source_turn_ids or [],
            source_summary_ids=source_summary_ids or [],
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            descendant_count=descendant_count,
            created_at=datetime.now(UTC).isoformat(),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "depth": self.depth,
            "content": self.content,
            "token_estimate": self.token_estimate,
            "source_turn_ids": self.source_turn_ids,
            "source_summary_ids": self.source_summary_ids,
            "parent_id": self.parent_id,
            "time_range_start": self.time_range_start,
            "time_range_end": self.time_range_end,
            "descendant_count": self.descendant_count,
            "file_refs": self.file_refs,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SummaryRecord:
        return cls(
            id=data.get("id", f"sum_{uuid.uuid4().hex[:16]}"),
            session_id=data.get("session_id", ""),
            depth=data.get("depth", 0),
            content=data.get("content", ""),
            token_estimate=data.get("token_estimate", 0),
            source_turn_ids=data.get("source_turn_ids", []),
            source_summary_ids=data.get("source_summary_ids", []),
            parent_id=data.get("parent_id"),
            time_range_start=data.get("time_range_start"),
            time_range_end=data.get("time_range_end"),
            descendant_count=data.get("descendant_count", 0),
            file_refs=data.get("file_refs", []),
            created_at=data.get("created_at", ""),
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> SummaryRecord:
        return cls(
            id=row["id"],
            session_id=row["session_id"],
            depth=row["depth"],
            content=row["content"],
            token_estimate=row["token_estimate"],
            source_turn_ids=json.loads(row["source_turn_ids"] or "[]"),
            source_summary_ids=json.loads(row["source_summary_ids"] or "[]"),
            parent_id=row["parent_id"],
            time_range_start=row["time_range_start"],
            time_range_end=row["time_range_end"],
            descendant_count=row["descendant_count"],
            file_refs=json.loads(row["file_refs"] or "[]"),
            created_at=row["created_at"],
        )


@dataclass
class LargeContentRecord:
    """A stored large tool result that was too big for inline context."""

    id: str
    session_id: str
    turn_id: str | None
    tool_name: str
    original_chars: int
    content: str
    summary: str = ""
    created_at: str = ""

    @classmethod
    def new(
        cls,
        session_id: str,
        tool_name: str,
        content: str,
        *,
        turn_id: str | None = None,
        summary: str = "",
    ) -> LargeContentRecord:
        return cls(
            id=f"ref_{uuid.uuid4().hex[:16]}",
            session_id=session_id,
            turn_id=turn_id,
            tool_name=tool_name,
            original_chars=len(content),
            content=content,
            summary=summary,
            created_at=datetime.now(UTC).isoformat(),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "tool_name": self.tool_name,
            "original_chars": self.original_chars,
            "content": self.content,
            "summary": self.summary,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> LargeContentRecord:
        return cls(
            id=row["id"],
            session_id=row["session_id"],
            turn_id=row["turn_id"],
            tool_name=row["tool_name"],
            original_chars=row["original_chars"],
            content=row["content"],
            summary=row["summary"] or "",
            created_at=row["created_at"],
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
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
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

            CREATE TABLE IF NOT EXISTS summaries (
                id                 TEXT PRIMARY KEY,
                session_id         TEXT NOT NULL,
                depth              INTEGER NOT NULL DEFAULT 0,
                content            TEXT NOT NULL,
                token_estimate     INTEGER NOT NULL DEFAULT 0,
                source_turn_ids    TEXT NOT NULL DEFAULT '[]',
                source_summary_ids TEXT NOT NULL DEFAULT '[]',
                parent_id          TEXT,
                time_range_start   TEXT,
                time_range_end     TEXT,
                descendant_count   INTEGER NOT NULL DEFAULT 0,
                file_refs          TEXT NOT NULL DEFAULT '[]',
                created_at         TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_summaries_session
                ON summaries(session_id);
            CREATE INDEX IF NOT EXISTS idx_summaries_depth
                ON summaries(session_id, depth);

            CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
                id         UNINDEXED,
                session_id UNINDEXED,
                content,
                content='summaries',
                content_rowid='rowid'
            );

            CREATE TRIGGER IF NOT EXISTS summaries_ai AFTER INSERT ON summaries BEGIN
                INSERT INTO summaries_fts(rowid, id, session_id, content)
                VALUES (new.rowid, new.id, new.session_id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS summaries_ad AFTER DELETE ON summaries BEGIN
                INSERT INTO summaries_fts(summaries_fts, rowid, id, session_id, content)
                VALUES ('delete', old.rowid, old.id, old.session_id, old.content);
            END;

            CREATE TABLE IF NOT EXISTS large_content (
                id              TEXT PRIMARY KEY,
                session_id      TEXT NOT NULL,
                turn_id         TEXT,
                tool_name       TEXT NOT NULL DEFAULT '',
                original_chars  INTEGER NOT NULL,
                content         TEXT NOT NULL,
                summary         TEXT NOT NULL DEFAULT '',
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_large_content_session
                ON large_content(session_id);
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
    # Summary DAG operations
    # ------------------------------------------------------------------

    def add_summary(self, summary: SummaryRecord) -> None:
        """Persist a summary record."""
        conn = self._conn()
        conn.execute(
            """INSERT OR REPLACE INTO summaries
               (id, session_id, depth, content, token_estimate,
                source_turn_ids, source_summary_ids, parent_id,
                time_range_start, time_range_end, descendant_count,
                file_refs, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                summary.id,
                summary.session_id,
                summary.depth,
                summary.content,
                summary.token_estimate,
                json.dumps(summary.source_turn_ids),
                json.dumps(summary.source_summary_ids),
                summary.parent_id,
                summary.time_range_start,
                summary.time_range_end,
                summary.descendant_count,
                json.dumps(summary.file_refs),
                summary.created_at,
            ),
        )
        conn.commit()

    def get_summaries(
        self,
        session_id: str,
        depth: int | None = None,
        limit: int = 50,
    ) -> list[SummaryRecord]:
        """Return summaries for a session, optionally filtered by depth."""
        conn = self._conn()
        if depth is not None:
            rows = conn.execute(
                """SELECT * FROM summaries
                   WHERE session_id = ? AND depth = ?
                   ORDER BY created_at
                   LIMIT ?""",
                (session_id, depth, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM summaries
                   WHERE session_id = ?
                   ORDER BY depth, created_at
                   LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [SummaryRecord.from_row(r) for r in rows]

    def get_summary_by_id(self, summary_id: str) -> SummaryRecord | None:
        """Return a single summary by ID, or None."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM summaries WHERE id = ?", (summary_id,)
        ).fetchone()
        return SummaryRecord.from_row(row) if row else None

    def get_uncompacted_summaries(
        self, session_id: str, depth: int
    ) -> list[SummaryRecord]:
        """Return summaries at *depth* with no parent (eligible for condensation)."""
        conn = self._conn()
        rows = conn.execute(
            """SELECT * FROM summaries
               WHERE session_id = ? AND depth = ? AND parent_id IS NULL
               ORDER BY created_at""",
            (session_id, depth),
        ).fetchall()
        return [SummaryRecord.from_row(r) for r in rows]

    def mark_summary_compacted(
        self, summary_ids: list[str], parent_id: str
    ) -> None:
        """Set parent_id on consumed summaries after condensation."""
        if not summary_ids:
            return
        conn = self._conn()
        placeholders = ",".join("?" for _ in summary_ids)
        conn.execute(
            f"UPDATE summaries SET parent_id = ? WHERE id IN ({placeholders})",
            [parent_id] + summary_ids,
        )
        conn.commit()

    def get_source_turns(self, summary_id: str) -> list[ConversationTurn]:
        """Resolve source_turn_ids for a summary into full turn objects."""
        summary = self.get_summary_by_id(summary_id)
        if not summary or not summary.source_turn_ids:
            return []
        conn = self._conn()
        placeholders = ",".join("?" for _ in summary.source_turn_ids)
        rows = conn.execute(
            f"SELECT * FROM turns WHERE id IN ({placeholders}) ORDER BY timestamp",
            summary.source_turn_ids,
        ).fetchall()
        return [self._row_to_turn(r) for r in rows]

    def get_child_summaries(self, parent_id: str) -> list[SummaryRecord]:
        """Return summaries whose parent_id matches."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM summaries WHERE parent_id = ? ORDER BY created_at",
            (parent_id,),
        ).fetchall()
        return [SummaryRecord.from_row(r) for r in rows]

    def search_summaries(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[SummaryRecord]:
        """Full-text search across summaries."""
        conn = self._conn()
        safe_query = '"' + query.replace('"', '""') + '"'
        try:
            if session_id:
                rows = conn.execute(
                    """SELECT s.* FROM summaries s
                       JOIN summaries_fts f ON s.rowid = f.rowid
                       WHERE summaries_fts MATCH ? AND s.session_id = ?
                       ORDER BY rank LIMIT ?""",
                    (safe_query, session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT s.* FROM summaries s
                       JOIN summaries_fts f ON s.rowid = f.rowid
                       WHERE summaries_fts MATCH ?
                       ORDER BY rank LIMIT ?""",
                    (safe_query, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            logger.warning("Summary FTS5 query failed for %r", query[:100])
            return []
        return [SummaryRecord.from_row(r) for r in rows]

    def get_session_token_count(self, session_id: str) -> int:
        """Estimate total tokens for a session (turns + top-level summaries)."""
        conn = self._conn()
        turn_chars = conn.execute(
            "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        summary_chars = conn.execute(
            """SELECT COALESCE(SUM(LENGTH(content)), 0) FROM summaries
               WHERE session_id = ? AND parent_id IS NULL""",
            (session_id,),
        ).fetchone()[0]
        return max(1, (turn_chars + summary_chars) // 4)

    # ------------------------------------------------------------------
    # Large content operations
    # ------------------------------------------------------------------

    def store_large_content(self, record: LargeContentRecord) -> str:
        """Persist a large content record. Returns the record ID."""
        conn = self._conn()
        conn.execute(
            """INSERT OR REPLACE INTO large_content
               (id, session_id, turn_id, tool_name, original_chars,
                content, summary, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.session_id,
                record.turn_id,
                record.tool_name,
                record.original_chars,
                record.content,
                record.summary,
                record.created_at,
            ),
        )
        conn.commit()
        return record.id

    def get_large_content(self, content_id: str) -> LargeContentRecord | None:
        """Return a large content record by ID, or None."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM large_content WHERE id = ?", (content_id,)
        ).fetchone()
        return LargeContentRecord.from_row(row) if row else None

    def search_large_content(
        self, query: str, session_id: str, limit: int = 5
    ) -> list[LargeContentRecord]:
        """Simple LIKE search across large content summaries and content."""
        conn = self._conn()
        pattern = f"%{query}%"
        rows = conn.execute(
            """SELECT * FROM large_content
               WHERE session_id = ? AND (summary LIKE ? OR content LIKE ?)
               ORDER BY created_at DESC LIMIT ?""",
            (session_id, pattern, pattern, limit),
        ).fetchall()
        return [LargeContentRecord.from_row(r) for r in rows]

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
