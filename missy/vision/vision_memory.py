"""Integration between vision observations and Missy's memory subsystem.

Persists significant visual observations (scene analysis results,
puzzle state, painting feedback) to the memory store so they can
influence future conversations and provide context continuity
across sessions.

This bridges the ephemeral in-process ``SceneSession`` (which holds
raw frames) with durable ``SQLiteMemoryStore`` and optional
``VectorMemoryStore`` for semantic retrieval.

Example::

    from missy.vision.vision_memory import VisionMemoryBridge

    bridge = VisionMemoryBridge()
    bridge.store_observation(
        session_id="session-123",
        task_type="puzzle",
        observation="Found 3 edge pieces matching the sky region",
        confidence=0.85,
    )
    results = bridge.recall_observations(query="sky pieces", limit=5)
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


class VisionMemoryBridge:
    """Bridge between vision observations and the memory subsystem.

    Stores observations as special memory entries tagged with vision
    metadata, enabling semantic recall of past visual analysis results.

    Parameters
    ----------
    memory_store:
        A SQLiteMemoryStore instance.  If None, creates one.
    vector_store:
        An optional VectorMemoryStore for semantic search.
    """

    # Category prefix for vision memories
    CATEGORY = "vision_observation"

    def __init__(
        self,
        memory_store: Any = None,
        vector_store: Any = None,
    ) -> None:
        self._memory = memory_store
        self._vector = vector_store
        self._initialized = False

    def _ensure_init(self) -> None:
        """Lazy-initialize stores if not provided."""
        if self._initialized:
            return
        if self._memory is None:
            try:
                from missy.memory.sqlite_store import SQLiteMemoryStore
                self._memory = SQLiteMemoryStore()
            except Exception as exc:
                logger.warning("Cannot init SQLiteMemoryStore: %s", exc)
        if self._vector is None:
            try:
                from missy.memory.vector_store import VectorMemoryStore
                self._vector = VectorMemoryStore()
            except Exception:
                logger.debug("VectorMemoryStore not available (faiss not installed)")
        self._initialized = True

    def store_observation(
        self,
        session_id: str,
        task_type: str,
        observation: str,
        confidence: float = 0.0,
        source: str = "",
        frame_id: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist a visual observation to the memory store.

        Parameters
        ----------
        session_id:
            Active conversation/task session ID.
        task_type:
            Vision task type (puzzle, painting, general, inspection).
        observation:
            The textual observation or analysis result.
        confidence:
            Confidence score (0.0 to 1.0) of the observation.
        source:
            Image source description (e.g. "webcam:/dev/video0").
        frame_id:
            Frame ID within the scene session.
        metadata:
            Additional metadata to store.

        Returns
        -------
        str
            The observation ID (UUID).
        """
        self._ensure_init()
        obs_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        entry = {
            "observation_id": obs_id,
            "session_id": session_id,
            "task_type": task_type,
            "observation": observation,
            "confidence": confidence,
            "source": source,
            "frame_id": frame_id,
            "timestamp": now,
            **(metadata or {}),
        }

        # Store in SQLite memory as a "vision" role turn
        if self._memory is not None:
            try:
                self._memory.add_turn(
                    session_id=session_id,
                    role="vision",
                    content=observation,
                    provider="vision",
                    metadata=entry,
                )
            except Exception as exc:
                logger.warning("Failed to store vision observation in SQLite: %s", exc)

        # Index in vector store for semantic recall
        if self._vector is not None:
            try:
                text = f"[{task_type}] {observation}"
                self._vector.add(text, entry)
            except Exception as exc:
                logger.debug("Failed to index vision observation in vector store: %s", exc)

        logger.info(
            "Stored vision observation %s: [%s] %s (confidence=%.2f)",
            obs_id,
            task_type,
            observation[:80],
            confidence,
        )
        return obs_id

    def recall_observations(
        self,
        query: str = "",
        task_type: str = "",
        session_id: str = "",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Recall past vision observations.

        Uses semantic search (vector store) if a query is provided,
        otherwise falls back to chronological retrieval.

        Parameters
        ----------
        query:
            Natural language query for semantic search.
        task_type:
            Filter to observations of this task type.
        session_id:
            Filter to observations from this session.
        limit:
            Maximum results to return.

        Returns
        -------
        list[dict]
            Matching observations with metadata.
        """
        self._ensure_init()
        results: list[dict[str, Any]] = []

        # Try vector search first if query given
        if query and self._vector is not None:
            try:
                search_query = f"[{task_type}] {query}" if task_type else query
                vector_results = self._vector.search(search_query, top_k=limit)
                for score, meta in vector_results:
                    if task_type and meta.get("task_type") != task_type:
                        continue
                    if session_id and meta.get("session_id") != session_id:
                        continue
                    meta["relevance_score"] = score
                    results.append(meta)
                if results:
                    return results[:limit]
            except Exception as exc:
                logger.debug("Vector search failed, falling back: %s", exc)

        # Fall back to SQLite search
        if self._memory is not None:
            try:
                if session_id:
                    turns = self._memory.get_session_turns(session_id, limit=limit * 2)
                elif query:
                    turns = self._memory.search(query, limit=limit * 2)
                else:
                    turns = self._memory.get_recent(limit=limit * 2)

                for turn in turns:
                    if getattr(turn, "role", "") != "vision":
                        continue
                    meta = getattr(turn, "metadata", {}) or {}
                    if task_type and meta.get("task_type") != task_type:
                        continue
                    results.append({
                        "observation": getattr(turn, "content", ""),
                        "session_id": getattr(turn, "session_id", ""),
                        **meta,
                    })
                    if len(results) >= limit:
                        break
            except Exception as exc:
                logger.debug("SQLite recall failed: %s", exc)

        return results

    def get_session_context(self, session_id: str) -> str:
        """Build a context string from all vision observations in a session.

        Suitable for injecting into prompts to provide visual context.
        """
        observations = self.recall_observations(session_id=session_id, limit=20)
        if not observations:
            return ""

        lines = ["## Visual Observations"]
        for obs in observations:
            task = obs.get("task_type", "general")
            text = obs.get("observation", "")
            conf = obs.get("confidence", 0)
            ts = obs.get("timestamp", "")
            lines.append(f"- [{task}] {text} (confidence: {conf:.0%}, {ts})")

        return "\n".join(lines)

    def clear_session(self, session_id: str) -> int:
        """Remove all vision observations for a session.

        Returns the number of observations removed.
        """
        self._ensure_init()
        count = 0
        if self._memory is not None:
            try:
                turns = self._memory.get_session_turns(session_id, limit=1000)
                for turn in turns:
                    if getattr(turn, "role", "") == "vision":
                        try:
                            self._memory.delete_turn(getattr(turn, "id", ""))
                            count += 1
                        except Exception:
                            pass
            except Exception as exc:
                logger.debug("Failed to clear session: %s", exc)
        return count
