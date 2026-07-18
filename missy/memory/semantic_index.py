"""Semantic conversation memory (F12).

``VectorMemoryStore`` (FAISS) was only wired into the vision scene-memory path;
general conversation recall was FTS5 keyword search only. ``ConversationSemanticIndex``
wires the vector store to conversation turns so the agent can recall a *paraphrase*
of an old message, not just a keyword match.

It is a thin, defensive adapter: index turns as they're processed (the
``SleeptimeWorker`` does this in the background when enabled), or bulk-``reindex``
from the SQLite store, then ``search`` semantically with an optional per-session
filter. When FAISS (the ``[vector]`` extra) is unavailable every method degrades
to a no-op / empty result, so the base install is unaffected.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SEMANTIC_INDEX_PATH = "~/.missy/memory_semantic.faiss"


class ConversationSemanticIndex:
    """FAISS-backed semantic index over conversation turns.

    Args:
        index_path: Where the FAISS index + metadata persist.
        dimension: Embedding dimensionality (must match the vectorizer).
    """

    def __init__(
        self,
        index_path: str | None = None,
        dimension: int = 384,
    ) -> None:
        # Resolve the default at call-time (not a bound default arg) so the path
        # is patchable in tests / overridable at runtime.
        path = index_path if index_path is not None else DEFAULT_SEMANTIC_INDEX_PATH
        self._store: Any | None = None
        try:
            from missy.memory.vector_store import VectorMemoryStore

            self._store = VectorMemoryStore(dimension=dimension, index_path=path)
            # VectorMemoryStore doesn't auto-load; pull any persisted index so a
            # fresh process (e.g. `missy memory semantic-search`) sees prior
            # `reindex` output.
            with contextlib.suppress(Exception):
                self._store.load()
        except Exception:
            logger.debug("VectorMemoryStore unavailable; semantic index disabled.", exc_info=True)
            self._store = None

    def flush(self) -> None:
        """Persist the in-memory index to disk. No-op when unavailable."""
        if self._store is None:
            return
        with contextlib.suppress(Exception):
            self._store.save()

    @property
    def available(self) -> bool:
        """True when a vector store backend is usable."""
        return self._store is not None

    def index_turn(self, turn: Any) -> bool:
        """Index a single conversation turn. Returns True if indexed.

        Skips blank content and de-dups by turn id (so re-indexing the same
        turn doesn't add duplicate vectors). Fully defensive.
        """
        if self._store is None:
            return False
        content = (getattr(turn, "content", "") or "").strip()
        if not content:
            return False
        turn_id = str(getattr(turn, "id", "") or "")
        # De-dup: skip if this turn id is already indexed.
        if turn_id and self._is_indexed(turn_id):
            return False
        try:
            self._store.add(
                content,
                metadata={
                    "turn_id": turn_id,
                    "session_id": str(getattr(turn, "session_id", "") or ""),
                    "role": str(getattr(turn, "role", "") or ""),
                    "timestamp": str(getattr(turn, "timestamp", "") or ""),
                },
            )
            return True
        except Exception:
            logger.debug("Semantic index_turn failed; skipping.", exc_info=True)
            return False

    def _is_indexed(self, turn_id: str) -> bool:
        entries = getattr(self._store, "_entries", None)
        if not entries:
            return False
        return any((e.get("metadata") or {}).get("turn_id") == turn_id for e in entries)

    def reindex(
        self, memory_store: Any, *, session_id: str | None = None, limit: int = 1000
    ) -> int:
        """Bulk-index turns from a SQLite memory store. Returns count indexed."""
        if self._store is None:
            return 0
        try:
            if session_id:
                turns = memory_store.get_session_turns(session_id, limit=limit)
            else:
                turns = memory_store.get_recent_turns(limit=limit)
        except Exception:
            logger.debug("Semantic reindex: could not load turns.", exc_info=True)
            return 0
        count = 0
        for turn in turns or []:
            if self.index_turn(turn):
                count += 1
        if count:
            self.flush()
        return count

    def search(self, query: str, top_k: int = 5, *, session_id: str | None = None) -> list[dict]:
        """Semantic search over indexed turns, optionally filtered by session.

        Returns a list of ``{text, metadata, score}`` dicts (best first). Empty
        when the backend is unavailable or nothing is indexed.
        """
        if self._store is None or not query.strip():
            return []
        try:
            # Over-fetch when filtering by session so the filter still yields top_k.
            raw = self._store.search(query, top_k=top_k * 4 if session_id else top_k)
        except Exception:
            logger.debug("Semantic search failed.", exc_info=True)
            return []
        if session_id:
            raw = [r for r in raw if (r.get("metadata") or {}).get("session_id") == session_id]
        return raw[:top_k]
