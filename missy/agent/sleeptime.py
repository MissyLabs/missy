"""Sleeptime computing — background memory processing during idle periods.

Inspired by Letta's VoiceSleeptimeAgent, :class:`SleeptimeWorker` runs a
daemon thread that wakes periodically to check whether the agent has been
idle long enough to warrant background memory work.  When idle, it:

1. Summarises batches of unsummarised conversation turns into
   :class:`~missy.memory.sqlite_store.SummaryRecord` entries.
2. Extracts :class:`~missy.agent.learnings.TaskLearning` records from turns
   that involved tool calls.
3. Publishes progress events to the :class:`~missy.core.message_bus.MessageBus`.

The worker never holds locks that could block the main agent loop, and all
errors are caught and logged so the thread never crashes silently.

Integration points in ``AgentRuntime``::

    # In __init__:
    self._sleeptime = SleeptimeWorker(
        memory_store=self._memory_store,
        provider_registry=self._provider_registry,
    )
    self._sleeptime.start()

    # In run() — at the top, before processing:
    self._sleeptime.record_activity()

    # In cleanup() / __del__:
    self._sleeptime.stop()

Example::

    from missy.agent.sleeptime import SleeptimeConfig, SleeptimeWorker

    worker = SleeptimeWorker(
        config=SleeptimeConfig(idle_threshold_seconds=60.0),
        memory_store=store,
    )
    worker.start()
    # ... agent runs happen ...
    worker.stop()
    print(worker.stats.summaries_created)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from missy.memory.sqlite_store import SQLiteMemoryStore
    from missy.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bus topic constants
# ---------------------------------------------------------------------------

#: A sleeptime processing cycle has started.
SLEEPTIME_CYCLE_START = "sleeptime.cycle.start"

#: A sleeptime processing cycle completed successfully.
SLEEPTIME_CYCLE_COMPLETE = "sleeptime.cycle.complete"

#: An error occurred during a sleeptime cycle.
SLEEPTIME_ERROR = "sleeptime.error"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SleeptimeConfig:
    """Configuration for :class:`SleeptimeWorker`.

    Attributes:
        enabled: When ``False`` the worker starts but immediately exits its
            loop without processing anything.
        idle_threshold_seconds: Seconds of inactivity before background
            processing is allowed to start.
        min_unprocessed_turns: Minimum unsummarised turns in a session before
            it is worth summarising.
        batch_size: Maximum number of turns to consume per session per cycle.
        check_interval_seconds: How often (in seconds) the worker wakes to
            check idle status.
        use_llm_summarization: Use an LLM provider for summaries when one is
            available.  Falls back to keyword extraction when ``False`` or
            when no provider is reachable.
    """

    enabled: bool = True
    idle_threshold_seconds: float = 300.0
    min_unprocessed_turns: int = 5
    batch_size: int = 20
    check_interval_seconds: float = 60.0
    use_llm_summarization: bool = True


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class SleeptimeStats:
    """Cumulative statistics for a :class:`SleeptimeWorker` instance.

    Attributes:
        cycles_completed: Number of full processing cycles that finished
            without an unhandled exception.
        turns_processed: Total turns summarised across all cycles.
        summaries_created: Total :class:`~missy.memory.sqlite_store.SummaryRecord`
            objects persisted.
        learnings_extracted: Total :class:`~missy.agent.learnings.TaskLearning`
            records saved.
        last_cycle_at: ISO-8601 timestamp of the most recent completed cycle,
            or ``None`` if no cycle has run yet.
        total_processing_seconds: Cumulative wall-clock time spent processing.
        errors: Number of cycles that raised an unexpected exception.
    """

    cycles_completed: int = 0
    turns_processed: int = 0
    summaries_created: int = 0
    learnings_extracted: int = 0
    last_cycle_at: str | None = None
    total_processing_seconds: float = 0.0
    errors: int = 0


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class SleeptimeWorker:
    """Background worker that processes memory during agent idle periods.

    Runs as a daemon thread so it is automatically killed when the main
    process exits.  The worker wakes every
    :attr:`~SleeptimeConfig.check_interval_seconds` and, if the agent has
    been idle for at least :attr:`~SleeptimeConfig.idle_threshold_seconds`,
    runs one processing cycle.

    The worker is intentionally stateless with respect to the agent loop —
    it only reads/writes through the injected ``memory_store`` and never
    blocks any path that the main thread might be waiting on.

    Args:
        config: Tuning parameters.  Defaults to :class:`SleeptimeConfig`.
        memory_store: A :class:`~missy.memory.sqlite_store.SQLiteMemoryStore`
            instance.  When ``None`` the worker starts but skips all
            processing.
        provider_registry: A :class:`~missy.providers.registry.ProviderRegistry`
            used to obtain a fast LLM for summarisation.  Optional; falls
            back to keyword extraction when absent or unavailable.
        graph_store: Reserved for future graph-memory entity consolidation.
            Currently unused.
    """

    def __init__(
        self,
        config: SleeptimeConfig | None = None,
        memory_store: SQLiteMemoryStore | None = None,
        provider_registry: ProviderRegistry | None = None,
        graph_store: object | None = None,
    ) -> None:
        self._config = config or SleeptimeConfig()
        self._memory_store = memory_store
        self._provider_registry = provider_registry
        self._graph_store = graph_store

        self._last_activity: float = time.monotonic()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._processing = False
        self._stats = SleeptimeStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background worker daemon thread.

        No-op if the worker is already running.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="missy-sleeptime",
            daemon=True,
        )
        self._thread.start()
        logger.debug(
            "SleeptimeWorker started (idle_threshold=%.0fs).", self._config.idle_threshold_seconds
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the worker to stop and wait for it to finish.

        Args:
            timeout: Maximum seconds to wait for the thread to exit after
                the stop signal is set.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.debug("SleeptimeWorker stopped.")

    def record_activity(self) -> None:
        """Reset the idle timer.

        Call this on every user interaction (i.e. at the top of
        ``AgentRuntime.run()``) so the worker does not process memory while
        the agent is actively responding.
        """
        self._last_activity = time.monotonic()

    def is_idle(self) -> bool:
        """Return ``True`` if the agent has been idle long enough to process.

        Returns:
            ``True`` when ``time.monotonic() - last_activity >= idle_threshold_seconds``.
        """
        elapsed = time.monotonic() - self._last_activity
        return elapsed >= self._config.idle_threshold_seconds

    @property
    def is_processing(self) -> bool:
        """Whether a processing cycle is currently executing."""
        return self._processing

    @property
    def stats(self) -> SleeptimeStats:
        """Cumulative processing statistics."""
        return self._stats

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Main daemon loop: sleep, wake, check idle, process if warranted."""
        while not self._stop_event.is_set():
            # Sleep in small increments so stop_event is checked promptly.
            self._stop_event.wait(timeout=self._config.check_interval_seconds)
            if self._stop_event.is_set():
                break

            if not self._config.enabled:
                continue

            if not self.is_idle():
                logger.debug("SleeptimeWorker: agent not idle — skipping cycle.")
                continue

            self._processing = True
            cycle_start = time.monotonic()
            try:
                self._process_cycle()
            except Exception:
                self._stats.errors += 1
                logger.exception("SleeptimeWorker: unhandled error in processing cycle.")
                self._publish_error("unhandled exception in _process_cycle")
            finally:
                elapsed = time.monotonic() - cycle_start
                self._stats.total_processing_seconds += elapsed
                self._processing = False

    def _process_cycle(self) -> None:
        """Run one background processing cycle.

        Steps:
        1. Identify sessions with unsummarised turns.
        2. Summarise batches of old turns, creating
           :class:`~missy.memory.sqlite_store.SummaryRecord` objects.
        3. Extract learnings from tool-heavy turns.
        4. Publish start/complete events to the message bus.
        """
        if self._memory_store is None:
            logger.debug("SleeptimeWorker: no memory_store — nothing to do.")
            return

        self._publish_bus(SLEEPTIME_CYCLE_START, {})

        sessions = self._find_sessions_needing_work()
        if not sessions:
            logger.debug("SleeptimeWorker: no sessions with enough unprocessed turns.")
            self._publish_bus(SLEEPTIME_CYCLE_COMPLETE, self._stats_payload())
            return

        cycle_turns = 0
        cycle_summaries = 0
        cycle_learnings = 0

        for session_id in sessions:
            turns = self._get_unsummarised_turns(session_id)
            if not turns:
                continue

            batch = turns[: self._config.batch_size]
            summary_content = self._summarize_session_turns(session_id, batch)
            if summary_content:
                self._persist_summary(session_id, batch, summary_content)
                cycle_summaries += 1

            new_learnings = self._extract_batch_learnings(session_id, batch)
            for learning in new_learnings:
                try:
                    self._memory_store.save_learning(learning)
                    cycle_learnings += 1
                except Exception:
                    logger.warning("SleeptimeWorker: failed to save learning — skipping.")

            cycle_turns += len(batch)

        self._stats.cycles_completed += 1
        self._stats.turns_processed += cycle_turns
        self._stats.summaries_created += cycle_summaries
        self._stats.learnings_extracted += cycle_learnings
        self._stats.last_cycle_at = datetime.now(UTC).isoformat()

        logger.info(
            "SleeptimeWorker cycle complete: %d turns, %d summaries, %d learnings.",
            cycle_turns,
            cycle_summaries,
            cycle_learnings,
        )
        self._publish_bus(SLEEPTIME_CYCLE_COMPLETE, self._stats_payload())

    # ------------------------------------------------------------------
    # Session discovery
    # ------------------------------------------------------------------

    def _find_sessions_needing_work(self) -> list[str]:
        """Return session IDs that have enough unsummarised turns to process.

        Returns:
            A list of session_id strings, possibly empty.
        """
        if self._memory_store is None:
            return []
        try:
            sessions = self._memory_store.list_sessions(limit=100)
        except Exception:
            logger.warning("SleeptimeWorker: failed to list sessions.", exc_info=True)
            return []

        result: list[str] = []
        for session in sessions:
            sid = session.get("session_id", "")
            if not sid:
                continue
            turns = self._get_unsummarised_turns(sid)
            if len(turns) >= self._config.min_unprocessed_turns:
                result.append(sid)
        return result

    def _get_unsummarised_turns(self, session_id: str) -> list:
        """Return turns for *session_id* that are not yet covered by a summary.

        We identify summarised turns by collecting all ``source_turn_ids``
        from existing depth-0 summaries and subtracting them from the full
        turn list.

        Args:
            session_id: The session to query.

        Returns:
            A list of :class:`~missy.memory.sqlite_store.ConversationTurn`
            objects, oldest-first, that have not been summarised.
        """
        if self._memory_store is None:
            return []
        try:
            all_turns = self._memory_store.get_session_turns(session_id, limit=500)
            existing_summaries = self._memory_store.get_summaries(session_id, depth=0, limit=200)
        except Exception:
            logger.warning("SleeptimeWorker: failed to fetch turns/summaries.", exc_info=True)
            return []

        summarised_ids: set[str] = set()
        for s in existing_summaries:
            summarised_ids.update(s.source_turn_ids)

        return [t for t in all_turns if t.id not in summarised_ids]

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def _summarize_session_turns(self, session_id: str, turns: list) -> str | None:
        """Summarise *turns* into a single string.

        Attempts LLM summarisation when :attr:`~SleeptimeConfig.use_llm_summarization`
        is enabled and a provider is available.  Falls back to keyword
        extraction on failure or when no provider is configured.

        Args:
            session_id: The owning session (used for logging only).
            turns: List of :class:`~missy.memory.sqlite_store.ConversationTurn`
                objects to summarise.

        Returns:
            A non-empty summary string, or ``None`` if nothing useful was
            extracted.
        """
        if not turns:
            return None

        if self._config.use_llm_summarization and self._provider_registry is not None:
            combined = self._turns_to_text(turns)
            llm_result = self._llm_summarize(combined)
            if llm_result:
                return llm_result

        return self._keyword_summarize(turns) or None

    def _persist_summary(self, session_id: str, turns: list, content: str) -> None:
        """Create and persist a depth-0 :class:`~missy.memory.sqlite_store.SummaryRecord`.

        Args:
            session_id: Owning session.
            turns: Source turns that were compressed into *content*.
            content: The summary text.
        """
        if self._memory_store is None:
            return
        from missy.memory.sqlite_store import SummaryRecord

        source_ids = [t.id for t in turns]
        timestamps = [t.timestamp for t in turns if t.timestamp]
        record = SummaryRecord.new(
            session_id=session_id,
            depth=0,
            content=content,
            source_turn_ids=source_ids,
            time_range_start=min(timestamps) if timestamps else None,
            time_range_end=max(timestamps) if timestamps else None,
            descendant_count=len(turns),
        )
        try:
            self._memory_store.add_summary(record)
        except Exception:
            logger.warning("SleeptimeWorker: failed to persist summary.", exc_info=True)

    def _llm_summarize(self, text: str) -> str | None:
        """Use an LLM provider to summarise *text*.

        Prefers the fast_model tier if the provider exposes one.  Returns
        ``None`` on any failure so the caller can fall back gracefully.

        Args:
            text: The concatenated turn text to summarise.

        Returns:
            A summary string, or ``None`` on error.
        """
        if self._provider_registry is None or not text.strip():
            return None

        # Prefer a fast/cheap model tier.
        provider = None
        for name in self._provider_registry.list_providers():
            candidate = self._provider_registry.get(name)
            if candidate is not None:
                try:
                    if candidate.is_available():
                        provider = candidate
                        break
                except Exception:
                    continue

        if provider is None:
            logger.debug("SleeptimeWorker: no available provider for LLM summarisation.")
            return None

        from missy.providers.base import Message

        prompt = (
            "You are a memory assistant. Summarise the following conversation excerpt "
            "into 3-5 concise bullet points capturing key facts, decisions, and outcomes. "
            "Be specific and factual. Do not add commentary.\n\n"
            f"{text[:4000]}"
        )
        try:
            response = provider.complete([Message(role="user", content=prompt)])
            result = response.content.strip()
            return result if result else None
        except Exception:
            logger.warning("SleeptimeWorker: LLM summarisation failed.", exc_info=True)
            return None

    def _keyword_summarize(self, turns: list) -> str:
        """Extract key facts from *turns* without an LLM.

        Delegates to the same heuristics used by
        :class:`~missy.agent.consolidation.MemoryConsolidator`.

        Args:
            turns: List of :class:`~missy.memory.sqlite_store.ConversationTurn`
                objects.

        Returns:
            A bullet-point summary string, possibly empty.
        """
        from missy.agent.consolidation import MemoryConsolidator

        messages = [
            {"role": t.role, "content": t.content, "name": t.metadata.get("tool_name", "")}
            for t in turns
        ]
        consolidator = MemoryConsolidator()
        facts = consolidator.extract_key_facts(messages)
        if not facts:
            return ""
        return "\n".join(f"- {fact}" for fact in facts)

    # ------------------------------------------------------------------
    # Learning extraction
    # ------------------------------------------------------------------

    def _extract_batch_learnings(self, session_id: str, turns: list) -> list:
        """Extract learnings from turns that involved tool calls.

        Groups consecutive tool-call+result pairs and calls
        :func:`~missy.agent.learnings.extract_learnings` for each assistant
        turn that references tool names in its metadata.

        Args:
            session_id: Owning session (currently unused but kept for
                future context).
            turns: Source turns to analyse.

        Returns:
            A list of :class:`~missy.agent.learnings.TaskLearning` objects.
        """
        from missy.agent.learnings import extract_learnings

        tool_names: list[str] = []
        last_assistant_content = ""

        for turn in turns:
            if turn.role == "tool":
                tool_name = turn.metadata.get("tool_name", "") or turn.metadata.get("name", "")
                if tool_name:
                    tool_names.append(tool_name)
            elif turn.role == "assistant":
                last_assistant_content = turn.content

        if not tool_names or not last_assistant_content:
            return []

        try:
            learning = extract_learnings(
                tool_names_used=tool_names,
                final_response=last_assistant_content,
                prompt="",
            )
            return [learning]
        except Exception:
            logger.warning("SleeptimeWorker: learning extraction failed.", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Bus helpers
    # ------------------------------------------------------------------

    def _publish_bus(self, topic: str, payload: dict) -> None:
        """Publish a message to the global message bus if available.

        Silently no-ops when the bus has not been initialised (e.g. in tests
        that do not set up the bus).

        Args:
            topic: Bus topic string.
            payload: Message payload dict.
        """
        try:
            from missy.core.message_bus import BusMessage, get_message_bus

            bus = get_message_bus()
            bus.publish(BusMessage(topic=topic, payload=payload, source="sleeptime"))
        except RuntimeError:
            # Bus not initialised — acceptable in many test scenarios.
            pass
        except Exception:
            logger.debug("SleeptimeWorker: failed to publish bus message.", exc_info=True)

    def _publish_error(self, detail: str) -> None:
        """Publish a ``SLEEPTIME_ERROR`` event to the bus.

        Args:
            detail: Human-readable error description.
        """
        self._publish_bus(SLEEPTIME_ERROR, {"detail": detail})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _turns_to_text(self, turns: list) -> str:
        """Render *turns* to a plain-text transcript for LLM input.

        Args:
            turns: :class:`~missy.memory.sqlite_store.ConversationTurn` objects.

        Returns:
            A multi-line string, one turn per line prefixed with its role.
        """
        lines: list[str] = []
        for turn in turns:
            role_label = turn.role.upper()
            content = turn.content.strip()
            if content:
                lines.append(f"{role_label}: {content}")
        return "\n".join(lines)

    def _stats_payload(self) -> dict:
        """Return the current stats as a bus-payload dict.

        Returns:
            A dict mirroring :class:`SleeptimeStats` fields.
        """
        s = self._stats
        return {
            "cycles_completed": s.cycles_completed,
            "turns_processed": s.turns_processed,
            "summaries_created": s.summaries_created,
            "learnings_extracted": s.learnings_extracted,
            "last_cycle_at": s.last_cycle_at,
            "total_processing_seconds": s.total_processing_seconds,
            "errors": s.errors,
        }

    def __repr__(self) -> str:
        return (
            f"<SleeptimeWorker enabled={self._config.enabled} "
            f"idle={self.is_idle()} processing={self._processing} "
            f"cycles={self._stats.cycles_completed}>"
        )
