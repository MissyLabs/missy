"""Simple JSON-based conversation memory store.

:class:`MemoryStore` persists conversation turns to a JSONL-like JSON file at
``~/.missy/memory.json`` and provides query methods for retrieving turns by
session or recency.

:class:`~missy.memory.sqlite_store.SQLiteMemoryStore` is also re-exported from
this module for convenience.

Example::

    from missy.memory.store import MemoryStore

    store = MemoryStore()
    turn = store.add_turn(
        session_id="session-abc",
        role="user",
        content="What is the weather today?",
        provider="anthropic",
    )
    history = store.get_session_turns("session-abc", limit=20)
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from missy.memory.sqlite_store import SQLiteMemoryStore  # re-export

__all__ = ["ConversationTurn", "MemoryStore", "SQLiteMemoryStore"]

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in a conversation.

    Attributes:
        id: Unique turn identifier (UUID string).
        session_id: Identifier of the session this turn belongs to.
        timestamp: UTC timestamp of the turn.
        role: Speaker role, typically ``"user"`` or ``"assistant"``.
        content: The message content.
        provider: Name of the AI provider that generated this turn (empty
            for user turns).
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    role: str = ""
    content: str = ""
    provider: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise the turn to a JSON-compatible dictionary.

        Returns:
            A mapping with all fields; the timestamp is represented as an
            ISO-8601 string.
        """
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "role": self.role,
            "content": self.content,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationTurn:
        """Deserialise a turn from a dictionary produced by :meth:`to_dict`.

        Args:
            data: Mapping with turn fields.  The ``timestamp`` value must be
                an ISO-8601 string or ``None``.

        Returns:
            A new :class:`ConversationTurn` instance.
        """
        raw_ts = data.get("timestamp")
        timestamp = datetime.fromisoformat(raw_ts) if raw_ts else datetime.now(tz=UTC)
        return cls(
            id=str(data.get("id", str(uuid.uuid4()))),
            session_id=str(data.get("session_id", "")),
            timestamp=timestamp,
            role=str(data.get("role", "")),
            content=str(data.get("content", "")),
            provider=str(data.get("provider", "")),
        )


class MemoryStore:
    """Persists and retrieves conversation turns using a JSON file.

    All turns are kept in memory as a list of :class:`ConversationTurn`
    objects and flushed to *store_path* on every write operation.

    Args:
        store_path: Path to the JSON file used for persistence.  Tilde
            expansion is performed automatically.
    """

    def __init__(self, store_path: str = "~/.missy/memory.json") -> None:
        self.store_path = Path(store_path).expanduser()
        self._turns: list[ConversationTurn] = []
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        provider: str = "",
    ) -> ConversationTurn:
        """Append a new conversation turn and persist it.

        Args:
            session_id: Session the turn belongs to.
            role: Speaker role (e.g. ``"user"`` or ``"assistant"``).
            content: Message text.
            provider: AI provider name; pass ``""`` for user turns.

        Returns:
            The newly created :class:`ConversationTurn`.
        """
        turn = ConversationTurn(
            session_id=session_id,
            role=role,
            content=content,
            provider=provider,
        )
        with self._lock:
            self._turns.append(turn)
            self._save()
        return turn

    def clear_session(self, session_id: str) -> None:
        """Remove all turns for a given session and persist the change.

        Args:
            session_id: The session whose turns should be deleted.
        """
        with self._lock:
            original_count = len(self._turns)
            self._turns = [t for t in self._turns if t.session_id != session_id]
            removed = original_count - len(self._turns)
            if removed:
                self._save()
        logger.debug("Cleared %d turn(s) for session %r.", removed, session_id)

    def compact_session(self, session_id: str, keep_recent: int = 10) -> int:
        """Summarise and remove old turns for a session, keeping the most recent *keep_recent*.

        Old turns are replaced by a single synthetic ``"assistant"`` turn that
        contains a brief excerpt of each compacted message.  The summary is
        inserted at the chronological position of the oldest removed turn.

        Args:
            session_id: The session to compact.
            keep_recent: Number of most-recent turns to preserve verbatim.

        Returns:
            Number of turns removed (0 when nothing was compacted).
        """
        session_turns = self.get_session_turns(session_id)
        if len(session_turns) <= keep_recent:
            return 0

        to_remove = session_turns[:-keep_recent]
        removed_count = len(to_remove)
        to_remove_ids = {t.id for t in to_remove}

        # Build a compact summary of the removed turns
        summary_lines: list[str] = []
        for t in to_remove:
            prefix = "User" if t.role == "user" else "Assistant"
            summary_lines.append(f"{prefix}: {t.content[:100]}")
        summary = "[Compacted history]\n" + "\n".join(summary_lines)

        summary_turn = ConversationTurn(
            id=f"compact-{session_id}",
            session_id=session_id,
            timestamp=to_remove[0].timestamp,
            role="assistant",
            content=summary,
            provider="compaction",
        )

        # Rebuild the global turn list: drop removed turns, prepend summary
        with self._lock:
            remaining = [
                t for t in self._turns if not (t.session_id == session_id and t.id in to_remove_ids)
            ]
            # Insert the summary turn before all other turns for this session so
            # it appears first in chronological order.
            insert_pos = next(
                (i for i, t in enumerate(remaining) if t.session_id == session_id),
                0,
            )
            remaining.insert(insert_pos, summary_turn)
            self._turns = remaining
            self._save()

        return removed_count

    def search(
        self,
        query: str,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[ConversationTurn]:
        """Basic case-insensitive keyword search across conversation history.

        This is a simple linear scan suitable for the JSON store.  For
        full-text ranked search use
        :class:`~missy.memory.sqlite_store.SQLiteMemoryStore`.

        Args:
            query: Substring to search for (case-insensitive).
            limit: Maximum number of results to return.
            session_id: When given, restrict results to this session.

        Returns:
            Matching :class:`ConversationTurn` objects in the order they
            appear in the store, capped at *limit*.
        """
        query_lower = query.lower()
        matches: list[ConversationTurn] = []
        for turn in self._turns:
            if session_id and turn.session_id != session_id:
                continue
            if query_lower in turn.content.lower():
                matches.append(turn)
        return matches[:limit]

    def get_summaries(
        self,
        session_id: str,  # noqa: ARG002
        depth: int | None = None,  # noqa: ARG002
        limit: int = 50,  # noqa: ARG002
    ) -> list:
        """No-op stub — summaries require :class:`~missy.memory.sqlite_store.SQLiteMemoryStore`.

        Returns:
            An empty list.
        """
        return []

    def get_session_token_count(self, session_id: str) -> int:
        """Estimate total tokens for a session by character count / 4.

        Args:
            session_id: Session identifier.

        Returns:
            Approximate token count.
        """
        total_chars = sum(len(t.content) for t in self._turns if t.session_id == session_id)
        return max(total_chars // 4, 0)

    def save_learning(self, learning) -> None:  # noqa: ARG002
        """No-op stub — learnings require :class:`~missy.memory.sqlite_store.SQLiteMemoryStore`.

        Args:
            learning: Ignored.
        """

    def get_learnings(
        self,
        task_type: str | None = None,  # noqa: ARG002
        limit: int = 5,  # noqa: ARG002
    ) -> list:
        """No-op stub — learnings require :class:`~missy.memory.sqlite_store.SQLiteMemoryStore`.

        Returns:
            An empty list.
        """
        return []

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_session_turns(self, session_id: str, limit: int = 50) -> list[ConversationTurn]:
        """Return the most recent turns for a specific session.

        Args:
            session_id: Session identifier to filter by.
            limit: Maximum number of turns to return.  The *most recent*
                turns are returned when the result set is truncated.

        Returns:
            A list of :class:`ConversationTurn` objects in chronological
            order, capped at *limit* entries.
        """
        session_turns = [t for t in self._turns if t.session_id == session_id]
        return session_turns[-limit:]

    def get_recent_turns(self, limit: int = 10) -> list[ConversationTurn]:
        """Return the most recent turns across all sessions.

        Args:
            limit: Maximum number of turns to return.

        Returns:
            A list of the *limit* most recent :class:`ConversationTurn`
            objects in chronological order.
        """
        return self._turns[-limit:]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Populate :attr:`_turns` from the JSON persistence file.

        Malformed records are skipped with a warning.  If the file does not
        exist the store starts empty without error.
        """
        if not self.store_path.exists():
            return

        try:
            raw_text = self.store_path.read_text(encoding="utf-8")
            records = json.loads(raw_text)
        except Exception as exc:
            logger.error("Failed to read memory store %s: %s", self.store_path, exc)
            return

        if not isinstance(records, list):
            logger.error(
                "Memory store file %s must contain a JSON array; found %s.",
                self.store_path,
                type(records).__name__,
            )
            return

        loaded = 0
        for record in records:
            if not isinstance(record, dict):
                logger.warning("Skipping non-dict memory record: %r", record)
                continue
            try:
                self._turns.append(ConversationTurn.from_dict(record))
                loaded += 1
            except Exception as exc:
                logger.warning("Skipping malformed memory record: %s", exc)

        logger.debug("Loaded %d turn(s) from %s.", loaded, self.store_path)

    def _save(self) -> None:
        """Persist all in-memory turns to the JSON file.

        The parent directory is created if it does not exist.  Errors are
        logged but not re-raised so callers are not disrupted.
        """
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            payload = [t.to_dict() for t in self._turns]
            self.store_path.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Failed to save memory store to %s: %s", self.store_path, exc)
