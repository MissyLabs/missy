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
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

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

    # Provider-backed background work is deliberately opt-in.  This is a
    # safety boundary: constructing an AgentRuntime must never spend quota.
    enabled: bool = False
    idle_threshold_seconds: float = 300.0
    min_unprocessed_turns: int = 5
    batch_size: int = 20
    check_interval_seconds: float = 60.0
    use_llm_summarization: bool = False
    provider: str | None = None
    backoff_base_seconds: float = 60.0
    backoff_max_seconds: float = 3600.0


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
    provider_calls: int = 0
    provider_failures: int = 0
    last_provider_call_at: str | None = None
    last_provider_failure_at: str | None = None
    next_retry_at: str | None = None


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
        graph_store: Optional :class:`~missy.memory.graph_store.GraphMemoryStore`
            (F04). When provided, each processed turn is fed into the graph via
            :meth:`_ingest_graph_entities` so the ``graph_query`` tool / ``missy
            graph`` CLI have populated data. ``None`` (the default) disables
            ingestion entirely — no extra write load.
    """

    _owners_lock = threading.Lock()
    _owners: dict[str, SleeptimeWorker] = {}
    _provider_lock = threading.Lock()
    _provider_backoff: dict[str, tuple[float, float]] = {}

    def __init__(
        self,
        config: SleeptimeConfig | None = None,
        memory_store: SQLiteMemoryStore | None = None,
        provider_registry: ProviderRegistry | None = None,
        graph_store: object | None = None,
        semantic_index: object | None = None,
        completion_runner: Callable[..., Any] | None = None,
    ) -> None:
        self._config = config or SleeptimeConfig()
        self._memory_store = memory_store
        self._provider_registry = provider_registry
        self._graph_store = graph_store
        # F12: optional ConversationSemanticIndex — when set, processed turns are
        # indexed into FAISS so memory_search / `missy memory semantic-search`
        # can do paraphrase recall. None (default) disables it (no write load).
        self._semantic_index = semantic_index
        self._completion_runner = completion_runner

        self._last_activity: float = time.monotonic()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._processing = False
        self._stats = SleeptimeStats()
        self._owner_key: str | None = None
        self._abort_cycle = False

    def _memory_store_key(self) -> str:
        store = self._memory_store
        if hasattr(store, "_primary"):
            store = store._primary
        path = getattr(store, "_path", None)
        if path is not None:
            try:
                return f"sqlite:{path.resolve()}"
            except Exception:
                return f"sqlite:{path}"
        return f"object:{id(store)}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the background worker daemon thread.

        No-op if the worker is already running.
        """
        if self._thread is not None and self._thread.is_alive():
            return True
        if not self._config.enabled or os.environ.get("MISSY_DISABLE_SLEEPTIME", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            logger.info("SleeptimeWorker disabled; no background thread started.")
            return False
        key = self._memory_store_key()
        with self._owners_lock:
            owner = self._owners.get(key)
            if owner is not None and owner is not self:
                logger.warning(
                    "SleeptimeWorker owner already exists for %s; refusing duplicate.", key
                )
                return False
            self._owners[key] = self
            self._owner_key = key
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
        return True

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the worker to stop and wait for it to finish.

        Args:
            timeout: Maximum seconds to wait for the thread to exit after
                the stop signal is set.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # A provider call may be blocked in ACPX.  Kill only child
                # process groups launched by this background thread, then
                # give it one final chance to unwind.
                try:
                    from missy.providers.acpx_provider import cancel_sleeptime_processes

                    cancel_sleeptime_processes()
                except Exception:
                    logger.debug("SleeptimeWorker: ACPX cancellation failed", exc_info=True)
                self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.error("SleeptimeWorker did not stop within %.1fs", timeout * 2)
                return
            self._thread = None
        if self._owner_key is not None:
            with self._owners_lock:
                if self._owners.get(self._owner_key) is self:
                    del self._owners[self._owner_key]
            self._owner_key = None
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

    @classmethod
    def diagnostics(cls) -> dict[str, Any]:
        """Return process-wide, operator-safe worker state."""
        with cls._owners_lock:
            workers = list(cls._owners.values())
        return {
            "worker_count": sum(bool(w._thread and w._thread.is_alive()) for w in workers),
            "workers": [
                {
                    "store": w._owner_key,
                    "provider": w._config.provider or "deterministic",
                    "processing": w._processing,
                    "provider_calls": w._stats.provider_calls,
                    "provider_failures": w._stats.provider_failures,
                    "last_provider_call_at": w._stats.last_provider_call_at,
                    "last_provider_failure_at": w._stats.last_provider_failure_at,
                    "next_retry_at": w._stats.next_retry_at,
                }
                for w in workers
            ],
        }

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

    def _process_playbook_promotions(self, threshold: int = 3) -> None:
        """Check Playbook for patterns eligible for promotion and mark them.

        5th tool-specific validation run's OPS-014 finding:
        :meth:`~missy.agent.playbook.Playbook.get_promotable` and
        :meth:`~missy.agent.playbook.Playbook.mark_promoted` had zero
        production callers anywhere -- ``Playbook.record()`` (the write
        side) is genuinely wired into :meth:`AgentRuntime.run`'s
        learnings-extraction path and real patterns do accumulate
        genuine success counts, but nothing ever checked whether any of
        them crossed the promotion threshold. This makes that check
        actually run, emitting an audit event per promotable pattern
        (so an operator can see it via ``missy audit recent``) and
        marking it promoted so it isn't re-reported every cycle.

        Independent of ``self._memory_store`` -- ``Playbook`` is its own
        JSON-backed store (``~/.missy/playbook.json``), not the
        conversation memory store this worker otherwise processes.

        Args:
            threshold: Minimum ``success_count`` for promotion, matching
                :meth:`~missy.agent.playbook.Playbook.get_promotable`'s
                own default.
        """
        try:
            from missy.agent.playbook import Playbook

            playbook = Playbook()
            promotable = playbook.get_promotable(threshold=threshold)
            for entry in promotable:
                # Each entry is handled independently so one bad entry
                # (a publish failure or a mark_promoted error) can't block
                # the rest of the batch.
                try:
                    from missy.core.events import AuditEvent, event_bus

                    event_bus.publish(
                        AuditEvent.now(
                            session_id="sleeptime",
                            task_id=entry.pattern_id,
                            event_type="playbook.pattern_promotable",
                            category="plugin",
                            result="allow",
                            detail={
                                "pattern_id": entry.pattern_id,
                                "task_type": entry.task_type,
                                "description": entry.description,
                                "success_count": entry.success_count,
                            },
                        )
                    )
                except Exception:
                    logger.debug(
                        "SleeptimeWorker: failed to publish playbook promotion audit event.",
                        exc_info=True,
                    )
                try:
                    playbook.mark_promoted(entry.pattern_id)
                except Exception:
                    logger.debug(
                        "SleeptimeWorker: failed to mark playbook pattern promoted.",
                        exc_info=True,
                    )
        except Exception:
            logger.debug("SleeptimeWorker: playbook promotion check failed.", exc_info=True)

    def _process_cycle(self) -> None:
        """Run one background processing cycle.

        Steps:
        0. Check Playbook for patterns eligible for promotion.
        1. Identify sessions with unsummarised turns.
        2. Summarise batches of old turns, creating
           :class:`~missy.memory.sqlite_store.SummaryRecord` objects.
        3. Extract learnings from tool-heavy turns.
        4. Publish start/complete events to the message bus.
        """
        self._process_playbook_promotions()

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

        self._abort_cycle = False
        for session_id in sessions:
            turns = self._get_unsummarised_turns(session_id)
            if not turns:
                continue

            batch = turns[: self._config.batch_size]
            summary_content = self._summarize_session_turns(session_id, batch)
            if summary_content:
                self._persist_summary(session_id, batch, summary_content)
                cycle_summaries += 1

            if self._abort_cycle:
                logger.warning("SleeptimeWorker: provider circuit opened; aborting cycle.")
                break

            new_learnings = self._extract_batch_learnings(session_id, batch)
            for learning in new_learnings:
                try:
                    self._memory_store.save_learning(learning)
                    cycle_learnings += 1
                except Exception:
                    logger.warning("SleeptimeWorker: failed to save learning — skipping.")

            self._ingest_graph_entities(session_id, batch)
            self._index_semantic(batch)

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

    def _ingest_graph_entities(self, session_id: str, batch: list) -> int:
        """Feed a batch of turns into the knowledge graph (F04 ingestion side).

        No-ops when no ``graph_store`` was injected (the default), so existing
        deployments are unaffected. Fully defensive: a graph failure must never
        break the sleeptime cycle. Returns the number of entities ingested
        (for observability/testing).

        Args:
            session_id: The session the turns belong to.
            batch: Turn objects with ``content``/``role`` attributes.

        Returns:
            Count of entities the graph reported ingesting across the batch.
        """
        if self._graph_store is None:
            return 0
        ingest = getattr(self._graph_store, "ingest_turn", None)
        if not callable(ingest):
            return 0

        ingested = 0
        for turn in batch:
            content = getattr(turn, "content", "") or ""
            role = getattr(turn, "role", "") or "user"
            if not content.strip():
                continue
            try:
                entities, _rels = ingest(content, role, session_id)
                ingested += len(entities or [])
            except Exception:
                logger.debug("SleeptimeWorker: graph ingest_turn failed — skipping.", exc_info=True)
        if ingested:
            logger.info(
                "SleeptimeWorker: ingested %d graph entit(y/ies) for session %s.",
                ingested,
                session_id,
            )
        return ingested

    def _index_semantic(self, batch: list) -> int:
        """Feed a batch of turns into the semantic index (F12 ingestion side).

        No-op when no ``semantic_index`` was injected. Fully defensive — an
        indexing failure never breaks the sleeptime cycle. Returns the number
        of turns indexed (for observability/testing).
        """
        if self._semantic_index is None:
            return 0
        index_turn = getattr(self._semantic_index, "index_turn", None)
        if not callable(index_turn):
            return 0
        indexed = 0
        for turn in batch:
            try:
                if index_turn(turn):
                    indexed += 1
            except Exception:
                logger.debug("SleeptimeWorker: semantic index_turn failed.", exc_info=True)
        if indexed:
            # Persist the batch so a separate reader process sees it.
            flush = getattr(self._semantic_index, "flush", None)
            if callable(flush):
                try:
                    flush()
                except Exception:
                    logger.debug("SleeptimeWorker: semantic flush failed.", exc_info=True)
            logger.info("SleeptimeWorker: semantically indexed %d turn(s).", indexed)
        return indexed

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
            # is_idle() only reflects *this* SleeptimeWorker instance's own
            # activity timer -- a multi-channel deployment (e.g. `missy run`
            # constructing a separate AgentRuntime, and therefore a separate
            # SleeptimeWorker, per channel) commonly has several workers
            # sharing one SQLiteMemoryStore. A session actively being
            # written to by a DIFFERENT runtime's in-flight run() call still
            # shows up here (list_sessions() has no per-worker ownership
            # concept), and this worker's own idle timer has nothing to do
            # with whether *that* session is actually idle -- summarizing it
            # anyway races the concurrent turn-append and can miss a
            # just-written turn from the summary's source_turn_ids boundary.
            # Guard directly against the session's own last-activity
            # timestamp rather than relying solely on this instance's timer.
            if self._session_recently_active(session.get("updated_at")):
                continue
            turns = self._get_unsummarised_turns(sid)
            if len(turns) >= self._config.min_unprocessed_turns:
                result.append(sid)
        return result

    def _session_recently_active(self, updated_at: str | None) -> bool:
        """Return ``True`` if *updated_at* is within the idle threshold.

        Args:
            updated_at: ISO-8601 timestamp string from ``list_sessions()``,
                or ``None``/empty when unavailable.

        Returns:
            ``True`` when the session was updated too recently to be
            considered idle (or when the timestamp can't be parsed, to
            fail closed rather than summarise a session we can't confirm
            is actually idle).
        """
        if not updated_at:
            return False
        try:
            last_active = datetime.fromisoformat(updated_at)
            if last_active.tzinfo is None:
                last_active = last_active.replace(tzinfo=UTC)
            elapsed = (datetime.now(UTC) - last_active).total_seconds()
        except (ValueError, TypeError):
            return True
        return elapsed < self._config.idle_threshold_seconds

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
            llm_result = self._llm_summarize(combined, session_id=session_id)
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

    def _llm_summarize(self, text: str, *, session_id: str = "sleeptime") -> str | None:
        """Use an LLM provider to summarise *text*.

        Uses only the explicitly configured provider. Returns ``None`` on
        failure so the caller can use deterministic summarisation.

        Args:
            text: The concatenated turn text to summarise.

        Returns:
            A summary string, or ``None`` on error.
        """
        if self._provider_registry is None or not text.strip():
            return None

        provider_name = (self._config.provider or "").strip()
        if not provider_name:
            return None

        provider = self._provider_registry.get(provider_name)

        if provider is None:
            logger.warning(
                "SleeptimeWorker: configured provider %r is not registered.", provider_name
            )
            return None

        now = time.monotonic()
        with self._provider_lock:
            retry_at, _delay = self._provider_backoff.get(provider_name, (0.0, 0.0))
        if retry_at > now:
            self._abort_cycle = True
            self._stats.next_retry_at = datetime.fromtimestamp(
                time.time() + (retry_at - now), UTC
            ).isoformat()
            return None

        from missy.providers.base import Message

        prompt = (
            "You are a memory assistant. Summarise the following conversation excerpt "
            "into 3-5 concise bullet points capturing key facts, decisions, and outcomes. "
            "Be specific and factual. Do not add commentary.\n\n"
            f"{text[:4000]}"
        )
        try:
            task_id = f"sleeptime-summary:{uuid.uuid4()}"
            messages = [Message(role="user", content=prompt)]
            if self._completion_runner is not None:
                response = self._completion_runner(
                    provider_name, messages, session_id=session_id, task_id=task_id
                )
            else:
                response = provider.complete(messages, session_id=session_id, task_id=task_id)
            self._stats.provider_calls += 1
            self._stats.last_provider_call_at = datetime.now(UTC).isoformat()
            with self._provider_lock:
                self._provider_backoff.pop(provider_name, None)
            self._stats.next_retry_at = None
            result = response.content.strip()
            return result if result else None
        except Exception as exc:
            self._stats.provider_calls += 1
            self._stats.provider_failures += 1
            self._stats.last_provider_call_at = datetime.now(UTC).isoformat()
            self._stats.last_provider_failure_at = self._stats.last_provider_call_at
            from missy.providers.health import ProviderFailureClass, classify_provider_error

            failure_class = classify_provider_error(exc)
            if failure_class in {
                ProviderFailureClass.AUTH,
                ProviderFailureClass.RATE_LIMIT,
                ProviderFailureClass.TIMEOUT,
            }:
                with self._provider_lock:
                    _old_retry, old_delay = self._provider_backoff.get(provider_name, (0.0, 0.0))
                    delay = min(
                        old_delay * 2 if old_delay else self._config.backoff_base_seconds,
                        self._config.backoff_max_seconds,
                    )
                    retry_at = time.monotonic() + delay
                    self._provider_backoff[provider_name] = (retry_at, delay)
                self._stats.next_retry_at = datetime.fromtimestamp(
                    time.time() + delay, UTC
                ).isoformat()
                self._abort_cycle = True
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
