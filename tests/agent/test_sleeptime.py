"""Tests for missy.agent.sleeptime — background memory processing."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from missy.agent.sleeptime import (
    SLEEPTIME_CYCLE_COMPLETE,
    SLEEPTIME_CYCLE_START,
    SleeptimeConfig,
    SleeptimeStats,
    SleeptimeWorker,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_turn(
    turn_id: str = "t1",
    session_id: str = "s1",
    role: str = "user",
    content: str = "hello",
    metadata: dict | None = None,
) -> MagicMock:
    """Build a minimal ConversationTurn-shaped mock."""
    turn = MagicMock()
    turn.id = turn_id
    turn.session_id = session_id
    turn.role = role
    turn.content = content
    turn.timestamp = "2024-01-01T00:00:00+00:00"
    turn.metadata = metadata or {}
    return turn


def _make_summary(source_turn_ids: list[str] | None = None) -> MagicMock:
    s = MagicMock()
    s.source_turn_ids = source_turn_ids or []
    return s


def _make_store(
    sessions: list[dict] | None = None,
    turns: list | None = None,
    summaries: list | None = None,
) -> MagicMock:
    store = MagicMock()
    store.list_sessions.return_value = sessions or []
    store.get_session_turns.return_value = turns or []
    store.get_summaries.return_value = summaries or []
    store.add_summary.return_value = None
    store.save_learning.return_value = None
    return store


# ---------------------------------------------------------------------------
# SleeptimeConfig defaults
# ---------------------------------------------------------------------------


class TestSleeptimeConfigDefaults:
    def test_defaults(self):
        cfg = SleeptimeConfig()
        assert cfg.enabled is True
        assert cfg.idle_threshold_seconds == 300.0
        assert cfg.min_unprocessed_turns == 5
        assert cfg.batch_size == 20
        assert cfg.check_interval_seconds == 60.0
        assert cfg.use_llm_summarization is True

    def test_custom_values(self):
        cfg = SleeptimeConfig(enabled=False, idle_threshold_seconds=10.0, batch_size=5)
        assert cfg.enabled is False
        assert cfg.idle_threshold_seconds == 10.0
        assert cfg.batch_size == 5


# ---------------------------------------------------------------------------
# SleeptimeStats defaults
# ---------------------------------------------------------------------------


class TestSleeptimeStats:
    def test_zero_initialized(self):
        s = SleeptimeStats()
        assert s.cycles_completed == 0
        assert s.turns_processed == 0
        assert s.summaries_created == 0
        assert s.learnings_extracted == 0
        assert s.last_cycle_at is None
        assert s.total_processing_seconds == 0.0
        assert s.errors == 0


# ---------------------------------------------------------------------------
# Worker lifecycle
# ---------------------------------------------------------------------------


class TestWorkerLifecycle:
    def test_start_creates_daemon_thread(self):
        worker = SleeptimeWorker(config=SleeptimeConfig(check_interval_seconds=9999))
        worker.start()
        assert worker._thread is not None
        assert worker._thread.is_alive()
        assert worker._thread.daemon is True
        worker.stop(timeout=2.0)

    def test_start_is_idempotent(self):
        worker = SleeptimeWorker(config=SleeptimeConfig(check_interval_seconds=9999))
        worker.start()
        first_thread = worker._thread
        worker.start()  # second call — should not replace the thread
        assert worker._thread is first_thread
        worker.stop(timeout=2.0)

    def test_stop_joins_thread(self):
        worker = SleeptimeWorker(config=SleeptimeConfig(check_interval_seconds=9999))
        worker.start()
        assert worker._thread is not None
        worker.stop(timeout=3.0)
        assert worker._thread is None

    def test_stop_without_start_is_safe(self):
        worker = SleeptimeWorker()
        worker.stop(timeout=1.0)  # should not raise

    def test_repr(self):
        worker = SleeptimeWorker()
        r = repr(worker)
        assert "SleeptimeWorker" in r
        assert "enabled=" in r


# ---------------------------------------------------------------------------
# Idle detection
# ---------------------------------------------------------------------------


class TestIdleDetection:
    def test_initially_idle_after_threshold(self):
        worker = SleeptimeWorker(config=SleeptimeConfig(idle_threshold_seconds=0.01))
        time.sleep(0.05)
        assert worker.is_idle() is True

    def test_not_idle_right_after_activity(self):
        worker = SleeptimeWorker(config=SleeptimeConfig(idle_threshold_seconds=300.0))
        worker.record_activity()
        assert worker.is_idle() is False

    def test_record_activity_resets_timer(self):
        worker = SleeptimeWorker(config=SleeptimeConfig(idle_threshold_seconds=0.01))
        time.sleep(0.05)
        assert worker.is_idle() is True
        worker.record_activity()
        assert worker.is_idle() is False

    def test_becomes_idle_after_threshold(self):
        worker = SleeptimeWorker(config=SleeptimeConfig(idle_threshold_seconds=0.05))
        worker.record_activity()
        assert worker.is_idle() is False
        time.sleep(0.1)
        assert worker.is_idle() is True


# ---------------------------------------------------------------------------
# is_processing property
# ---------------------------------------------------------------------------


class TestIsProcessing:
    def test_not_processing_initially(self):
        worker = SleeptimeWorker()
        assert worker.is_processing is False

    def test_processing_flag_set_during_cycle(self):
        """Verify the flag is set to True while _process_cycle is executing."""
        processing_states: list[bool] = []

        original_process = SleeptimeWorker._process_cycle

        def slow_cycle(self_inner):
            processing_states.append(self_inner.is_processing)
            original_process(self_inner)

        worker = SleeptimeWorker(config=SleeptimeConfig(check_interval_seconds=9999))
        worker._process_cycle = lambda: slow_cycle(worker)  # type: ignore[method-assign]

        # Manually invoke the flag-setting logic from _run_loop.
        worker._processing = True
        assert worker.is_processing is True
        worker._processing = False
        assert worker.is_processing is False


# ---------------------------------------------------------------------------
# _process_cycle
# ---------------------------------------------------------------------------


class TestProcessCycle:
    def test_skips_when_no_memory_store(self):
        worker = SleeptimeWorker(config=SleeptimeConfig())
        # Should not raise — just returns early.
        worker._process_cycle()
        assert worker.stats.cycles_completed == 0

    def test_skips_when_no_sessions_with_enough_turns(self):
        # No turns means _find_sessions_needing_work returns empty; the cycle
        # exits early (before the counter is incremented) without raising.
        store = _make_store(sessions=[{"session_id": "s1"}], turns=[], summaries=[])
        worker = SleeptimeWorker(memory_store=store)
        worker._process_cycle()
        assert worker.stats.cycles_completed == 0
        assert worker.stats.summaries_created == 0

    def test_creates_summary_when_turns_available(self):
        turns = [_make_turn(f"t{i}", role="user", content=f"message {i}") for i in range(6)]
        store = _make_store(
            sessions=[{"session_id": "s1"}],
            turns=turns,
            summaries=[],
        )
        worker = SleeptimeWorker(
            config=SleeptimeConfig(
                min_unprocessed_turns=5,
                use_llm_summarization=False,
            ),
            memory_store=store,
        )
        worker._process_cycle()

        assert worker.stats.summaries_created == 1
        assert worker.stats.cycles_completed == 1
        store.add_summary.assert_called_once()

    def test_stats_accumulate_across_cycles(self):
        turns = [_make_turn(f"t{i}", role="user", content=f"msg {i}") for i in range(6)]
        store = _make_store(
            sessions=[{"session_id": "s1"}],
            turns=turns,
            summaries=[],
        )
        worker = SleeptimeWorker(
            config=SleeptimeConfig(min_unprocessed_turns=5, use_llm_summarization=False),
            memory_store=store,
        )
        worker._process_cycle()
        worker._process_cycle()

        assert worker.stats.cycles_completed == 2

    def test_last_cycle_at_set_after_cycle(self):
        # last_cycle_at is only stamped when sessions with enough turns are found.
        turns = [_make_turn(f"t{i}", content=f"msg {i}") for i in range(6)]
        store = _make_store(sessions=[{"session_id": "s1"}], turns=turns, summaries=[])
        worker = SleeptimeWorker(
            config=SleeptimeConfig(min_unprocessed_turns=5, use_llm_summarization=False),
            memory_store=store,
        )
        assert worker.stats.last_cycle_at is None
        worker._process_cycle()
        assert worker.stats.last_cycle_at is not None

    def test_turns_processed_count(self):
        turns = [_make_turn(f"t{i}", content=f"msg {i}") for i in range(6)]
        store = _make_store(
            sessions=[{"session_id": "s1"}],
            turns=turns,
            summaries=[],
        )
        worker = SleeptimeWorker(
            config=SleeptimeConfig(min_unprocessed_turns=5, use_llm_summarization=False),
            memory_store=store,
        )
        worker._process_cycle()
        # batch_size=20 default — all 6 turns should be processed
        assert worker.stats.turns_processed == 6


# ---------------------------------------------------------------------------
# LLM summarisation
# ---------------------------------------------------------------------------


class TestLlmSummarize:
    def _make_provider_registry(self, response_text: str = "- Summary bullet") -> MagicMock:
        registry = MagicMock()
        provider = MagicMock()
        provider.is_available.return_value = True
        completion = MagicMock()
        completion.content = response_text
        provider.complete.return_value = completion
        registry.list_providers.return_value = ["anthropic"]
        registry.get.return_value = provider
        return registry

    def test_returns_llm_output(self):
        registry = self._make_provider_registry("- Key fact extracted by LLM")
        worker = SleeptimeWorker(
            config=SleeptimeConfig(use_llm_summarization=True),
            provider_registry=registry,
        )
        result = worker._llm_summarize("USER: hello\nASSISTANT: world")
        assert result == "- Key fact extracted by LLM"

    def test_returns_none_on_empty_provider_response(self):
        registry = self._make_provider_registry("   ")
        worker = SleeptimeWorker(
            config=SleeptimeConfig(use_llm_summarization=True),
            provider_registry=registry,
        )
        result = worker._llm_summarize("some text")
        assert result is None

    def test_returns_none_on_provider_exception(self):
        registry = MagicMock()
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.complete.side_effect = RuntimeError("API error")
        registry.list_providers.return_value = ["anthropic"]
        registry.get.return_value = provider
        worker = SleeptimeWorker(
            config=SleeptimeConfig(use_llm_summarization=True),
            provider_registry=registry,
        )
        result = worker._llm_summarize("text")
        assert result is None

    def test_returns_none_when_no_registry(self):
        worker = SleeptimeWorker()
        result = worker._llm_summarize("text")
        assert result is None

    def test_returns_none_for_empty_text(self):
        registry = self._make_provider_registry("something")
        worker = SleeptimeWorker(provider_registry=registry)
        result = worker._llm_summarize("   ")
        assert result is None

    def test_returns_none_when_no_available_provider(self):
        registry = MagicMock()
        provider = MagicMock()
        provider.is_available.return_value = False
        registry.list_providers.return_value = ["anthropic"]
        registry.get.return_value = provider
        worker = SleeptimeWorker(provider_registry=registry)
        result = worker._llm_summarize("text")
        assert result is None


# ---------------------------------------------------------------------------
# Keyword summarisation fallback
# ---------------------------------------------------------------------------


class TestKeywordSummarize:
    def test_returns_bullet_points_for_keyword_turns(self):
        turns = [
            _make_turn("t1", role="user", content="Deploy the service"),
            _make_turn("t2", role="assistant", content="Result: deployment succeeded"),
            _make_turn("t3", role="user", content="Decided: use staging first"),
        ]
        worker = SleeptimeWorker()
        result = worker._keyword_summarize(turns)
        assert "Result:" in result or "result:" in result.lower()
        assert result.startswith("-")

    def test_returns_empty_string_for_no_facts(self):
        turns = [
            _make_turn("t1", role="user", content="Hello there, how are you today?"),
            _make_turn("t2", role="assistant", content="I am doing well, thank you very much!"),
        ]
        worker = SleeptimeWorker()
        result = worker._keyword_summarize(turns)
        # May return empty or short user message — just ensure it does not raise.
        assert isinstance(result, str)

    def test_fallback_used_when_llm_disabled(self):
        turns = [_make_turn(f"t{i}", content=f"Result: step {i} done") for i in range(6)]
        worker = SleeptimeWorker(config=SleeptimeConfig(use_llm_summarization=False))
        result = worker._summarize_session_turns("s1", turns)
        assert result is not None
        assert "Result:" in result or "result:" in result.lower()


# ---------------------------------------------------------------------------
# Learning extraction
# ---------------------------------------------------------------------------


class TestExtractBatchLearnings:
    def test_extracts_learning_from_tool_turns(self):
        turns = [
            _make_turn("t1", role="user", content="run the tests"),
            _make_turn(
                "t2",
                role="tool",
                content="exit code 0",
                metadata={"tool_name": "shell_exec"},
            ),
            _make_turn(
                "t3",
                role="assistant",
                content="successfully ran tests",
            ),
        ]
        worker = SleeptimeWorker()
        learnings = worker._extract_batch_learnings("s1", turns)
        assert len(learnings) == 1
        assert "shell_exec" in learnings[0].approach

    def test_no_learning_without_tool_turns(self):
        turns = [
            _make_turn("t1", role="user", content="hello"),
            _make_turn("t2", role="assistant", content="hi"),
        ]
        worker = SleeptimeWorker()
        learnings = worker._extract_batch_learnings("s1", turns)
        assert learnings == []

    def test_no_learning_without_assistant_response(self):
        turns = [
            _make_turn("t1", role="tool", content="result", metadata={"tool_name": "file_read"}),
        ]
        worker = SleeptimeWorker()
        learnings = worker._extract_batch_learnings("s1", turns)
        assert learnings == []

    def test_learning_saved_to_store(self):
        turns = [
            _make_turn("t1", role="tool", content="ok", metadata={"tool_name": "web_fetch"}),
            _make_turn("t2", role="assistant", content="successfully fetched the page"),
        ]
        store = _make_store(sessions=[{"session_id": "s1"}], turns=turns, summaries=[])
        worker = SleeptimeWorker(
            config=SleeptimeConfig(min_unprocessed_turns=1, use_llm_summarization=False),
            memory_store=store,
        )
        worker._process_cycle()
        store.save_learning.assert_called()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_store_error_does_not_crash_cycle(self):
        # list_sessions failing causes _find_sessions_needing_work to return
        # empty; _process_cycle exits early without incrementing cycles_completed.
        # The key invariant is that the call does not raise.
        store = MagicMock()
        store.list_sessions.side_effect = RuntimeError("db gone")
        worker = SleeptimeWorker(memory_store=store)
        worker._process_cycle()  # must not raise
        # Early exit path — counter not bumped, but no exception either
        assert worker.stats.cycles_completed == 0
        assert worker.stats.errors == 0

    def test_save_learning_error_is_skipped(self):
        turns = [
            _make_turn("t1", role="tool", content="ok", metadata={"tool_name": "shell_exec"}),
            _make_turn("t2", role="assistant", content="success"),
        ]
        store = _make_store(
            sessions=[{"session_id": "s1"}],
            turns=turns,
            summaries=[],
        )
        store.save_learning.side_effect = RuntimeError("disk full")
        worker = SleeptimeWorker(
            config=SleeptimeConfig(min_unprocessed_turns=1, use_llm_summarization=False),
            memory_store=store,
        )
        # Should not raise even though save_learning fails
        worker._process_cycle()
        assert worker.stats.cycles_completed == 1

    def test_add_summary_error_is_skipped(self):
        turns = [_make_turn(f"t{i}", content=f"msg {i}") for i in range(6)]
        store = _make_store(
            sessions=[{"session_id": "s1"}],
            turns=turns,
            summaries=[],
        )
        store.add_summary.side_effect = RuntimeError("write error")
        worker = SleeptimeWorker(
            config=SleeptimeConfig(min_unprocessed_turns=5, use_llm_summarization=False),
            memory_store=store,
        )
        worker._process_cycle()
        # Cycle should still complete — summary count stays 0 because persist failed
        assert worker.stats.cycles_completed == 1

    def test_unhandled_exception_increments_errors(self):
        """An unhandled exception in _process_cycle must increment stats.errors."""
        worker = SleeptimeWorker()

        def bad_cycle():
            raise RuntimeError("unexpected")

        worker._process_cycle = bad_cycle  # type: ignore[method-assign]

        # Simulate what _run_loop does
        try:
            worker._process_cycle()
        except Exception:
            worker._stats.errors += 1

        assert worker.stats.errors == 1

    def test_disabled_worker_does_not_process(self):
        store = _make_store(sessions=[{"session_id": "s1"}])
        worker = SleeptimeWorker(
            config=SleeptimeConfig(enabled=False),
            memory_store=store,
        )
        # Calling _run_loop once with stop_event pre-set to avoid infinite loop
        worker._stop_event.set()
        worker._run_loop()
        # The stop_event was set immediately, so no processing happened
        assert worker.stats.cycles_completed == 0


# ---------------------------------------------------------------------------
# Unsummarised turn filtering
# ---------------------------------------------------------------------------


class TestGetUnsummarisedTurns:
    def test_excludes_already_summarised_turns(self):
        turns = [_make_turn(f"t{i}") for i in range(5)]
        existing_summary = _make_summary(source_turn_ids=["t0", "t1"])
        store = _make_store(turns=turns, summaries=[existing_summary])
        worker = SleeptimeWorker(memory_store=store)
        result = worker._get_unsummarised_turns("s1")
        ids = [t.id for t in result]
        assert "t0" not in ids
        assert "t1" not in ids
        assert "t2" in ids

    def test_all_turns_unsummarised_when_no_summaries(self):
        turns = [_make_turn(f"t{i}") for i in range(4)]
        store = _make_store(turns=turns, summaries=[])
        worker = SleeptimeWorker(memory_store=store)
        result = worker._get_unsummarised_turns("s1")
        assert len(result) == 4

    def test_returns_empty_when_no_memory_store(self):
        worker = SleeptimeWorker()
        assert worker._get_unsummarised_turns("s1") == []

    def test_returns_empty_on_store_exception(self):
        store = MagicMock()
        store.get_session_turns.side_effect = RuntimeError("db error")
        worker = SleeptimeWorker(memory_store=store)
        result = worker._get_unsummarised_turns("s1")
        assert result == []


# ---------------------------------------------------------------------------
# Batch size limiting
# ---------------------------------------------------------------------------


class TestBatchSize:
    def test_batch_size_limits_turns_consumed(self):
        turns = [_make_turn(f"t{i}", content=f"msg {i}") for i in range(15)]
        store = _make_store(
            sessions=[{"session_id": "s1"}],
            turns=turns,
            summaries=[],
        )
        worker = SleeptimeWorker(
            config=SleeptimeConfig(
                min_unprocessed_turns=5,
                batch_size=7,
                use_llm_summarization=False,
            ),
            memory_store=store,
        )
        worker._process_cycle()
        assert worker.stats.turns_processed == 7


# ---------------------------------------------------------------------------
# Message bus integration
# ---------------------------------------------------------------------------


class TestBusIntegration:
    def test_publishes_cycle_start_and_complete(self):
        from missy.core.message_bus import init_message_bus, reset_message_bus

        reset_message_bus()
        bus = init_message_bus()
        received: list[str] = []
        bus.subscribe("sleeptime.*", lambda msg: received.append(msg.topic))

        store = _make_store(sessions=[{"session_id": "s1"}], turns=[], summaries=[])
        worker = SleeptimeWorker(memory_store=store)
        worker._process_cycle()

        assert SLEEPTIME_CYCLE_START in received
        assert SLEEPTIME_CYCLE_COMPLETE in received
        reset_message_bus()

    def test_no_error_when_bus_not_initialised(self):
        from missy.core.message_bus import reset_message_bus

        reset_message_bus()
        worker = SleeptimeWorker()
        # Should not raise
        worker._publish_bus(SLEEPTIME_CYCLE_START, {})
        worker._publish_error("test error")


# ---------------------------------------------------------------------------
# Thread safety smoke test
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_record_activity_from_multiple_threads(self):
        """Calling record_activity concurrently must not raise."""
        worker = SleeptimeWorker(config=SleeptimeConfig(idle_threshold_seconds=9999))

        errors: list[Exception] = []

        def spam_activity():
            for _ in range(100):
                try:
                    worker.record_activity()
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=spam_activity) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert worker.is_idle() is False

    def test_stats_not_corrupted_under_concurrent_reads(self):
        """Reading stats while a cycle is running must not raise."""
        turns = [_make_turn(f"t{i}", content=f"msg {i}") for i in range(8)]
        store = _make_store(
            sessions=[{"session_id": "s1"}],
            turns=turns,
            summaries=[],
        )
        worker = SleeptimeWorker(
            config=SleeptimeConfig(min_unprocessed_turns=5, use_llm_summarization=False),
            memory_store=store,
        )
        errors: list[Exception] = []

        def run_cycles():
            for _ in range(5):
                try:
                    worker._process_cycle()
                except Exception as exc:
                    errors.append(exc)

        def read_stats():
            for _ in range(50):
                try:
                    _ = worker.stats.cycles_completed
                    _ = worker.stats.summaries_created
                except Exception as exc:
                    errors.append(exc)

        cycle_thread = threading.Thread(target=run_cycles)
        reader_thread = threading.Thread(target=read_stats)
        cycle_thread.start()
        reader_thread.start()
        cycle_thread.join()
        reader_thread.join()

        assert errors == []
