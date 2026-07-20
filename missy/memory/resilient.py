"""Resilient memory store with in-memory fallback on primary failure."""

from __future__ import annotations

import logging
import threading
import unicodedata
from datetime import datetime

logger = logging.getLogger(__name__)


class ResilientMemoryStore:
    """Wraps a primary store with an in-memory dict fallback.

    Behaviour:
    - All writes go to both the in-memory cache *and* the primary store.
    - On primary failure, operations transparently fall back to the cache.
    - After *max_failures* consecutive failures the primary is marked
      unhealthy and every subsequent success attempt logs the recovery.
    - On recovery the cached entries are synced back to the primary.

    Args:
        primary: Any store object that implements the MemoryStore interface
            (``add_turn``, ``get_session_turns``, ``get_recent_turns``,
            ``clear_session``, ``search``, ``save_learning``,
            ``get_learnings``, ``cleanup``).
        max_failures: Number of consecutive failures before the store is
            considered unhealthy.

    Example::

        from missy.memory.sqlite_store import SQLiteMemoryStore
        from missy.memory.resilient import ResilientMemoryStore

        store = ResilientMemoryStore(SQLiteMemoryStore())
        store.add_turn(turn)
        print(store.is_healthy)
    """

    def __init__(self, primary, max_failures: int = 3) -> None:
        if isinstance(max_failures, bool) or not isinstance(max_failures, int):
            raise TypeError("max_failures must be an integer")
        if not 1 <= max_failures <= 100:
            raise ValueError("max_failures must be between 1 and 100")
        self._primary = primary
        self._max_failures = max_failures
        self._failures = 0
        self._cache: dict[str, list] = {}  # session_id -> list of turns
        self._lock = threading.Lock()
        self._recovery_lock = threading.Lock()
        self._pending_ops: list[tuple[str, tuple]] = []
        self._healthy = True
        self._read_status: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Health tracking
    # ------------------------------------------------------------------

    def _on_success(self) -> None:
        if not self._replay_pending():
            return
        with self._lock:
            if not self._healthy:
                logger.info(
                    "ResilientMemory: primary recovered after %d failure(s)",
                    self._failures,
                )
            self._failures = 0
            self._healthy = True

    def _on_failure(self, exc: Exception) -> None:
        with self._lock:
            self._failures += 1
            self._healthy = False
            logger.warning(
                "ResilientMemory: primary failure #%d: %s",
                self._failures,
                exc,
            )
            if self._failures >= self._max_failures:
                logger.error(
                    "ResilientMemory: primary unhealthy after %d consecutive failures",
                    self._failures,
                )

    def _queue_pending(self, operation: str, *args) -> None:
        with self._lock:
            self._pending_ops.append((operation, args))

    def _replay_pending(self) -> bool:
        """Replay only failed mutations, in original order, exactly once per success.

        Successful writes are retained in the read cache but never enter this
        journal, preventing recovery from duplicating every previously
        persisted turn. SQLite's stable turn IDs make an interrupted replay
        idempotent if the process loses confirmation after commit.
        """
        with self._recovery_lock:
            while True:
                with self._lock:
                    if not self._pending_ops:
                        return True
                    operation, args = self._pending_ops[0]
                try:
                    getattr(self._primary, operation)(*args)
                except Exception as exc:
                    self._on_failure(exc)
                    logger.warning("ResilientMemory: pending %s replay failed", operation)
                    return False
                with self._lock:
                    if self._pending_ops and self._pending_ops[0] == (operation, args):
                        self._pending_ops.pop(0)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_turn(self, turn) -> None:
        """Append *turn* to the cache and attempt to write to the primary.

        Args:
            turn: A turn object with at least a ``session_id`` attribute.
        """
        with self._lock:
            sid = turn.session_id
            if sid not in self._cache:
                self._cache[sid] = []
            self._cache[sid].append(turn)

        if not self._replay_pending():
            self._queue_pending("add_turn", turn)
            return
        try:
            self._primary.add_turn(turn)
            self._on_success()
        except Exception as exc:
            self._queue_pending("add_turn", turn)
            self._on_failure(exc)

    def clear_session(self, session_id: str) -> None:
        """Remove all turns for *session_id* from cache and primary.

        Note this does not clear persisted summaries -- see
        :meth:`clear_session_full` for a genuine full reset.

        Args:
            session_id: The session whose turns should be deleted.
        """
        with self._lock:
            self._cache.pop(session_id, None)
        if not self._replay_pending():
            self._queue_pending("clear_session", session_id)
            return
        try:
            self._primary.clear_session(session_id)
            self._on_success()
        except Exception as exc:
            self._queue_pending("clear_session", session_id)
            self._on_failure(exc)

    def clear_session_full(self, session_id: str) -> None:
        """Remove all turns AND all summaries for *session_id*.

        Falls back to :meth:`clear_session` when the primary store has no
        ``clear_session_full`` of its own (e.g. the plain JSON-backed
        :class:`~missy.memory.store.MemoryStore`, which has no summaries
        table to begin with).

        Args:
            session_id: The session to fully reset.
        """
        with self._lock:
            self._cache.pop(session_id, None)
        operation = (
            "clear_session_full"
            if getattr(self._primary, "clear_session_full", None) is not None
            else "clear_session"
        )
        if not self._replay_pending():
            self._queue_pending(operation, session_id)
            return
        try:
            full_reset = getattr(self._primary, "clear_session_full", None)
            if full_reset is not None:
                full_reset(session_id)
            else:
                self._primary.clear_session(session_id)
            self._on_success()
        except Exception as exc:
            self._queue_pending(operation, session_id)
            self._on_failure(exc)

    def delete_turn(self, turn_id: str) -> bool:
        """Delete a single turn from the cache and the primary store.

        Args:
            turn_id: The turn's unique id.

        Returns:
            ``True`` if the primary store deleted a matching row.
        """
        with self._lock:
            for turns in self._cache.values():
                turns[:] = [t for t in turns if getattr(t, "id", None) != turn_id]
        if not self._replay_pending():
            self._queue_pending("delete_turn", turn_id)
            return False
        try:
            result = self._primary.delete_turn(turn_id)
            self._on_success()
            return result
        except Exception as exc:
            self._queue_pending("delete_turn", turn_id)
            self._on_failure(exc)
            return False

    def set_turn_pinned(self, turn_id: str, pinned: bool) -> bool:
        """Forward a pin/unpin request to the primary store.

        Args:
            turn_id: The turn's unique id.
            pinned: Whether the turn should be marked pinned.

        Returns:
            ``True`` if the primary store updated a matching row.
        """
        if not self._replay_pending():
            self._queue_pending("set_turn_pinned", turn_id, pinned)
            self._set_cached_pin(turn_id, pinned)
            return False
        try:
            result = self._primary.set_turn_pinned(turn_id, pinned)
            if result:
                self._set_cached_pin(turn_id, pinned)
            self._on_success()
            return result
        except Exception as exc:
            self._queue_pending("set_turn_pinned", turn_id, pinned)
            self._set_cached_pin(turn_id, pinned)
            self._on_failure(exc)
            return False

    def save_learning(self, learning) -> None:
        """Forward a learning object to the primary store.

        Args:
            learning: An object carrying learning fields (duck-typed).
        """
        try:
            self._primary.save_learning(learning)
            self._on_success()
        except Exception as exc:
            self._on_failure(exc)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_session_turns(self, session_id: str, limit: int = 100) -> list:
        """Return up to *limit* turns for *session_id*, falling back to cache.

        Args:
            session_id: Session identifier to filter by.
            limit: Maximum number of turns to return.

        Returns:
            A list of turn objects in chronological order.
        """
        limit = _validated_limit(limit)
        if not isinstance(session_id, str) or not session_id or len(session_id) > 512:
            raise ValueError("session_id must be a non-empty bounded string")
        try:
            result = self._primary.get_session_turns(session_id, limit)
            self._on_success()
            self._mark_read("get_session_turns", "primary")
            return result
        except Exception as exc:
            self._on_failure(exc)
            with self._lock:
                cached = list(self._cache.get(session_id, []))
            self._mark_read("get_session_turns", "fallback")
            return sorted(cached, key=_turn_sort_key)[-limit:]

    def get_recent_turns(self, limit: int = 50) -> list:
        """Return up to *limit* most recent turns, falling back to cache.

        Args:
            limit: Maximum number of turns to return.

        Returns:
            A list of turn objects in chronological order.
        """
        limit = _validated_limit(limit)
        try:
            result = self._primary.get_recent_turns(limit)
            self._on_success()
            self._mark_read("get_recent_turns", "primary")
            return result
        except Exception as exc:
            self._on_failure(exc)
            with self._lock:
                all_turns = [turn for turns in self._cache.values() for turn in turns]
            all_turns.sort(key=_turn_sort_key)
            self._mark_read("get_recent_turns", "fallback")
            return all_turns[-limit:]

    def search(
        self,
        query: str,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list:
        """Search conversation history, falling back to keyword scan of cache.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.
            session_id: When given, restrict results to this session.

        Returns:
            A list of matching turn objects.
        """
        limit = _validated_limit(limit)
        if not isinstance(query, str) or not query.strip() or len(query) > 10_000:
            raise ValueError("query must be non-empty bounded text")
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id or len(session_id) > 512
        ):
            raise ValueError("session_id must be a non-empty bounded string")
        try:
            result = self._primary.search(query, limit=limit, session_id=session_id)
            self._on_success()
            self._mark_read("search", "primary")
            return result
        except Exception as exc:
            self._on_failure(exc)
            # Unicode-normalized, session-scoped substring scan over one
            # lock-protected cache snapshot with stable tie ordering.
            matches: list = []
            query_normalized = unicodedata.normalize("NFKC", query).casefold()
            with self._lock:
                snapshot = {sid: list(turns) for sid, turns in self._cache.items()}
            for sid, turns in snapshot.items():
                if session_id and sid != session_id:
                    continue
                for t in turns:
                    content = unicodedata.normalize("NFKC", str(t.content)).casefold()
                    if query_normalized in content:
                        matches.append(t)
            matches.sort(key=_turn_sort_key)
            self._mark_read("search", "fallback")
            return matches[:limit]

    def get_learnings(
        self,
        task_type: str | None = None,
        limit: int = 5,
    ) -> list:
        """Retrieve recent learning lessons, returning empty list on failure.

        Args:
            task_type: When given, filter to this task type only.
            limit: Maximum number of lessons to return.

        Returns:
            A list of lesson strings, or an empty list when the primary
            is unavailable.
        """
        limit = _validated_limit(limit)
        try:
            result = self._primary.get_learnings(task_type=task_type, limit=limit)
            self._on_success()
            self._mark_read("get_learnings", "primary")
            return result
        except Exception as exc:
            self._on_failure(exc)
            self._mark_read("get_learnings", "unavailable")
            return []

    def get_total_costs(self, limit: int = 50) -> list:
        """Return per-session cost summaries from the primary store.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            A list of per-session cost dicts, or ``[]`` on primary failure
            (cost records are not held in the in-memory fallback cache).
        """
        limit = _validated_limit(limit)
        try:
            result = self._primary.get_total_costs(limit=limit)
            self._on_success()
            self._mark_read("get_total_costs", "primary")
            return result
        except Exception as exc:
            self._on_failure(exc)
            self._mark_read("get_total_costs", "unavailable")
            return []

    def get_cost_totals(self) -> dict:
        """Return dashboard-wide aggregate spend from the primary store.

        Returns:
            A dict of aggregate totals, or a zeroed dict on primary failure.
        """
        try:
            result = self._primary.get_cost_totals()
            self._on_success()
            self._mark_read("get_cost_totals", "primary")
            return result
        except Exception as exc:
            self._on_failure(exc)
            self._mark_read("get_cost_totals", "unavailable")
            return {
                "call_count": 0,
                "session_count": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_cost_usd": 0.0,
            }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup(self, older_than_days: int = 30, dry_run: bool = False) -> int:
        """Delegate cleanup to the primary store.

        Args:
            older_than_days: Age threshold in days.
            dry_run: When ``True``, count matching rows without deleting
                them (see :meth:`SQLiteMemoryStore.cleanup`).

        Returns:
            Number of rows deleted (or that would be deleted, if
            *dry_run*), or 0 on failure.
        """
        try:
            result = self._primary.cleanup(older_than_days=older_than_days, dry_run=dry_run)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure(exc)
            return 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        """``True`` when the primary store has not exceeded *max_failures*."""
        with self._lock:
            return self._healthy

    @property
    def read_status(self) -> dict[str, str]:
        """Return whether recent read surfaces used primary/fallback/unavailable data."""
        with self._lock:
            return dict(self._read_status)

    def _mark_read(self, operation: str, status: str) -> None:
        with self._lock:
            self._read_status[operation] = status

    def _set_cached_pin(self, turn_id: str, pinned: bool) -> None:
        with self._lock:
            for turns in self._cache.values():
                for turn in turns:
                    if getattr(turn, "id", None) == turn_id:
                        try:
                            turn.pinned = pinned
                        except Exception:
                            return


def _validated_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer between 1 and 1000")
    return limit


def _turn_sort_key(turn: object) -> tuple[str, str]:
    timestamp = getattr(turn, "timestamp", "")
    if isinstance(timestamp, datetime):
        timestamp_key = timestamp.isoformat()
    elif isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
        timestamp_key = f"{float(timestamp):030.9f}"
    else:
        timestamp_key = str(timestamp)
    return timestamp_key, str(getattr(turn, "id", ""))
