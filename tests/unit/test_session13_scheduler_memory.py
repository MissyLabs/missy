"""Session 13: Scheduler and memory store edge case tests.

Covers gaps not addressed by any existing test suite:

Scheduler (missy.scheduler.jobs / parser / manager):
- should_retry boundary logic (consecutive_failures vs max_attempts)
- active_hours with malformed format defaults to True
- active_hours zero-width window (start == end)
- from_dict with max_attempts exactly 10 (boundary, not clamped)
- from_dict with negative consecutive_failures coerced to int
- to_dict / from_dict round-trip for retry state fields
- _run_job with missing job_id logs and returns without raising
- cleanup_memory delegates to MemoryStore when available
- list_jobs_with_details is a true alias for list_jobs
- JSON persistence: empty array loads as zero jobs
- Parser: "at" trigger normalises space separator to T
- Parser: raw cron with step notation passes through intact
- Parser: interval with tz kwarg does NOT include timezone key
- Job ordering maintained after pause/resume cycle
- add_job with custom max_attempts exactly 1 (boundary accepted)

Memory store (missy.memory.store.MemoryStore JSON):
- compact_session with exactly keep_recent turns is a no-op
- compact_session isolates target session, leaves others intact
- compact_session summary includes User/Assistant prefixes
- compact_session insert position is before remaining session turns
- get_session_token_count basic character-to-token ratio
- get_summaries stub always returns empty list
- get_learnings stub always returns empty list
- save_learning stub is a no-op
- search on empty store returns empty list
- search with limit=0 returns empty list
- ConversationTurn.to_dict with None timestamp emits None
- JSON file is pretty-printed (indent=2)
- Multiple reload cycles do not duplicate turns
- add_turn with special/unicode characters round-trips correctly
- clear_session on already-empty session does not write extra data
- from_dict with all string fields coerced to str correctly
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from missy.memory.store import ConversationTurn, MemoryStore
from missy.scheduler.jobs import ScheduledJob
from missy.scheduler.manager import SchedulerManager
from missy.scheduler.parser import parse_schedule

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def jobs_path(tmp_path: Path) -> str:
    return str(tmp_path / "jobs.json")


@pytest.fixture()
def mgr(jobs_path: str) -> SchedulerManager:
    m = SchedulerManager(jobs_file=jobs_path)
    m.start()
    yield m
    m.stop()


@pytest.fixture()
def mem_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(store_path=str(tmp_path / "memory.json"))


# ===========================================================================
# 1. ScheduledJob.should_retry boundary conditions
# ===========================================================================


class TestShouldRetryBoundary:
    """should_retry returns False when consecutive_failures >= max_attempts."""

    def test_zero_failures_always_retries(self) -> None:
        job = ScheduledJob(max_attempts=3, consecutive_failures=0)
        assert job.should_retry("any error") is True

    def test_failures_below_max_retries(self) -> None:
        job = ScheduledJob(max_attempts=3, consecutive_failures=2)
        assert job.should_retry("transient") is True

    def test_failures_equal_max_does_not_retry(self) -> None:
        job = ScheduledJob(max_attempts=3, consecutive_failures=3)
        assert job.should_retry("fatal") is False

    def test_failures_above_max_does_not_retry(self) -> None:
        job = ScheduledJob(max_attempts=3, consecutive_failures=99)
        assert job.should_retry("fatal") is False

    def test_max_attempts_1_retries_on_first_failure_only(self) -> None:
        job = ScheduledJob(max_attempts=1, consecutive_failures=0)
        assert job.should_retry("first") is True
        job.consecutive_failures = 1
        assert job.should_retry("second") is False

    def test_error_string_is_ignored(self) -> None:
        """The error argument does not gate retry decisions."""
        job = ScheduledJob(max_attempts=5, consecutive_failures=1)
        for error in ["network", "provider_error", "unknown_category", ""]:
            assert job.should_retry(error) is True


# ===========================================================================
# 2. ScheduledJob.should_run_now with malformed active_hours
# ===========================================================================


class TestActiveHoursMalformed:
    def test_empty_active_hours_always_runs(self) -> None:
        job = ScheduledJob(active_hours="")
        assert job.should_run_now() is True

    def test_malformed_format_defaults_to_true(self) -> None:
        """A string that does not match HH:MM-HH:MM should fall back to True."""
        for bad in ["always", "9am-5pm", "09:00", "not-a-time", "00:00:00-23:59:59"]:
            job = ScheduledJob(active_hours=bad)
            assert job.should_run_now() is True, f"Expected True for active_hours={bad!r}"

    def test_zero_width_window_same_start_and_end(self) -> None:
        """When start == end, the window is a single instant (inclusive on both ends)."""
        job = ScheduledJob(active_hours="12:00-12:00")
        with patch("missy.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 12, 0, 0)
            assert job.should_run_now() is True

    def test_zero_width_window_one_minute_after_is_outside(self) -> None:
        job = ScheduledJob(active_hours="12:00-12:00")
        with patch("missy.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 12, 1, 0)
            assert job.should_run_now() is False


# ===========================================================================
# 3. ScheduledJob.from_dict boundary / coercion cases
# ===========================================================================


class TestFromDictBoundary:
    def test_max_attempts_exactly_10_not_clamped(self) -> None:
        job = ScheduledJob.from_dict({"max_attempts": 10})
        assert job.max_attempts == 10

    def test_max_attempts_11_clamped_to_10(self) -> None:
        job = ScheduledJob.from_dict({"max_attempts": 11})
        assert job.max_attempts == 10

    def test_consecutive_failures_preserved(self) -> None:
        job = ScheduledJob.from_dict({"consecutive_failures": 7})
        assert job.consecutive_failures == 7

    def test_last_error_preserved_on_round_trip(self) -> None:
        original = ScheduledJob(last_error="connection refused")
        restored = ScheduledJob.from_dict(original.to_dict())
        assert restored.last_error == "connection refused"

    def test_enabled_false_from_dict(self) -> None:
        job = ScheduledJob.from_dict({"enabled": False})
        assert job.enabled is False

    def test_enabled_0_coerced_to_false(self) -> None:
        job = ScheduledJob.from_dict({"enabled": 0})
        assert job.enabled is False

    def test_delete_after_run_true_from_dict(self) -> None:
        job = ScheduledJob.from_dict({"delete_after_run": True})
        assert job.delete_after_run is True

    def test_run_count_string_coerced_to_int(self) -> None:
        """Legacy files may have stored run_count as a string."""
        job = ScheduledJob.from_dict({"run_count": "42"})
        assert job.run_count == 42

    def test_all_retry_fields_round_trip(self) -> None:
        original = ScheduledJob(
            max_attempts=4,
            backoff_seconds=[5, 15, 60],
            retry_on=["timeout", "rate_limit"],
            consecutive_failures=2,
            last_error="HTTP 429",
        )
        restored = ScheduledJob.from_dict(original.to_dict())
        assert restored.max_attempts == 4
        assert restored.backoff_seconds == [5, 15, 60]
        assert restored.retry_on == ["timeout", "rate_limit"]
        assert restored.consecutive_failures == 2
        assert restored.last_error == "HTTP 429"


# ===========================================================================
# 4. SchedulerManager._run_job with missing job
# ===========================================================================


class TestRunJobMissingId:
    def test_run_job_with_unknown_id_does_not_raise(self, mgr: SchedulerManager) -> None:
        """_run_job must log and return cleanly when the job no longer exists."""
        mgr._run_job("completely-fake-id")  # must not raise

    def test_run_job_with_unknown_id_does_not_call_agent(self, mgr: SchedulerManager) -> None:
        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            mgr._run_job("ghost-id")
            MockRuntime.assert_not_called()


# ===========================================================================
# 5. SchedulerManager.cleanup_memory
# ===========================================================================


class TestCleanupMemory:
    def test_cleanup_memory_returns_int(self, mgr: SchedulerManager) -> None:
        result = mgr.cleanup_memory(older_than_days=30)
        assert isinstance(result, int)
        assert result >= 0

    def test_cleanup_memory_returns_zero_on_exception(self, mgr: SchedulerManager) -> None:
        # MemoryStore is imported inside cleanup_memory, so patch the source module
        with patch("missy.memory.store.MemoryStore", side_effect=RuntimeError("boom")):
            result = mgr.cleanup_memory(older_than_days=7)
        assert result == 0


# ===========================================================================
# 6. SchedulerManager.list_jobs_with_details is an alias
# ===========================================================================


class TestListJobsWithDetails:
    def test_list_jobs_with_details_returns_same_ids(self, mgr: SchedulerManager) -> None:
        mgr.add_job("a", "every 5 minutes", "task a")
        mgr.add_job("b", "daily at 09:00", "task b")
        by_list = [j.id for j in mgr.list_jobs()]
        by_details = [j.id for j in mgr.list_jobs_with_details()]
        assert by_list == by_details

    def test_list_jobs_with_details_empty_when_no_jobs(self, mgr: SchedulerManager) -> None:
        assert mgr.list_jobs_with_details() == []


# ===========================================================================
# 7. JSON persistence: empty array and zero-job state
# ===========================================================================


class TestEmptyJobsFilePersistence:
    def test_empty_json_array_loads_zero_jobs(self, jobs_path: str) -> None:
        path = Path(jobs_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")
        path.chmod(0o600)

        m = SchedulerManager(jobs_file=jobs_path)
        m.start()
        assert m.list_jobs() == []
        m.stop()

    def test_manager_starts_cleanly_with_no_file(self, tmp_path: Path) -> None:
        path = str(tmp_path / "nonexistent_jobs.json")
        m = SchedulerManager(jobs_file=path)
        m.start()
        assert m.list_jobs() == []
        m.stop()

    def test_add_job_with_max_attempts_1_accepted(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job("one-shot-retry", "every 5 minutes", "t", max_attempts=1)
        assert job.max_attempts == 1


# ===========================================================================
# 8. Parser: at-trigger, cron passthrough, interval+tz
# ===========================================================================


class TestParserEdgeCasesSession13:
    def test_at_trigger_space_separator_normalised(self) -> None:
        result = parse_schedule("at 2099-07-04 15:30")
        assert result["trigger"] == "date"
        assert result["run_date"] == "2099-07-04T15:30"

    def test_at_trigger_t_separator_preserved(self) -> None:
        result = parse_schedule("at 2099-12-31T23:59")
        assert result["run_date"] == "2099-12-31T23:59"

    def test_raw_cron_step_notation_passes_through(self) -> None:
        result = parse_schedule("*/10 6-22 * * 1-5")
        assert result["trigger"] == "cron"
        assert result["_cron_expression"] == "*/10 6-22 * * 1-5"

    def test_raw_cron_comma_list_passes_through(self) -> None:
        result = parse_schedule("0 8,12,18 * * *")
        assert result["_cron_expression"] == "0 8,12,18 * * *"

    def test_interval_with_tz_kwarg_has_no_timezone_key(self) -> None:
        """Interval triggers must not attach timezone even when tz is supplied."""
        result = parse_schedule("every 15 minutes", tz="America/New_York")
        assert "timezone" not in result
        assert result == {"trigger": "interval", "minutes": 15}

    def test_weekly_at_keyword_optional(self) -> None:
        with_at = parse_schedule("weekly on Monday at 08:00")
        without_at = parse_schedule("weekly on Monday 08:00")
        # Both must produce the same result
        assert with_at == without_at

    def test_daily_midnight_boundary(self) -> None:
        result = parse_schedule("daily at 00:00")
        assert result["hour"] == 0
        assert result["minute"] == 0


# ===========================================================================
# 9. Job ordering after pause/resume cycle
# ===========================================================================


class TestJobOrderingAfterPauseResume:
    def test_job_order_preserved_after_pause_resume(self, mgr: SchedulerManager) -> None:
        names = ["first", "second", "third"]
        jobs = [mgr.add_job(n, "every 5 minutes", "t") for n in names]

        mgr.pause_job(jobs[1].id)
        mgr.resume_job(jobs[1].id)

        listed_names = [j.name for j in mgr.list_jobs()]
        assert listed_names == names


# ===========================================================================
# 10. MemoryStore.compact_session edge cases
# ===========================================================================


class TestCompactSessionEdgeCases:
    def test_compact_exactly_at_keep_recent_is_noop(self, mem_store: MemoryStore) -> None:
        for i in range(5):
            mem_store.add_turn("s1", "user", f"msg {i}")
        removed = mem_store.compact_session("s1", keep_recent=5)
        assert removed == 0
        # No summary turn should be inserted
        turns = mem_store.get_session_turns("s1")
        assert len(turns) == 5

    def test_compact_does_not_affect_other_sessions(self, mem_store: MemoryStore) -> None:
        for i in range(10):
            mem_store.add_turn("target", "user", f"target-{i}")
        for i in range(3):
            mem_store.add_turn("bystander", "user", f"by-{i}")

        mem_store.compact_session("target", keep_recent=3)

        bystander_turns = mem_store.get_session_turns("bystander")
        assert len(bystander_turns) == 3
        assert all("by-" in t.content for t in bystander_turns)

    def test_compact_summary_contains_user_prefix(self, mem_store: MemoryStore) -> None:
        mem_store.add_turn("s1", "user", "user message content")
        mem_store.add_turn("s1", "assistant", "assistant response")
        for _ in range(8):
            mem_store.add_turn("s1", "user", "filler")

        mem_store.compact_session("s1", keep_recent=3)

        all_turns = mem_store.get_session_turns("s1", limit=100)
        summary_turns = [t for t in all_turns if t.role == "assistant" and "[Compacted history]" in t.content]
        assert len(summary_turns) == 1
        summary = summary_turns[0].content
        assert "User:" in summary

    def test_compact_summary_contains_assistant_prefix(self, mem_store: MemoryStore) -> None:
        mem_store.add_turn("s1", "assistant", "assistant first response")
        for _ in range(9):
            mem_store.add_turn("s1", "user", "filler")

        mem_store.compact_session("s1", keep_recent=3)

        all_turns = mem_store.get_session_turns("s1", limit=100)
        summary_turns = [t for t in all_turns if "[Compacted history]" in t.content]
        assert len(summary_turns) == 1
        assert "Assistant:" in summary_turns[0].content

    def test_compact_summary_turn_appears_before_retained_turns(self, mem_store: MemoryStore) -> None:
        for i in range(10):
            mem_store.add_turn("ordered", "user", f"msg-{i}")

        mem_store.compact_session("ordered", keep_recent=3)

        turns = mem_store.get_session_turns("ordered", limit=100)
        # First turn should be the compaction summary
        assert "[Compacted history]" in turns[0].content
        # Last 3 turns should be the most recent original messages
        assert turns[-1].content == "msg-9"
        assert turns[-2].content == "msg-8"
        assert turns[-3].content == "msg-7"

    def test_compact_then_reload_preserves_compacted_state(
        self, tmp_path: Path
    ) -> None:
        path = str(tmp_path / "compact_reload.json")
        s1 = MemoryStore(store_path=path)
        for i in range(8):
            s1.add_turn("sess", "user", f"old-{i}")

        s1.compact_session("sess", keep_recent=2)
        original_count = len(s1.get_session_turns("sess", limit=100))

        s2 = MemoryStore(store_path=path)
        reloaded_count = len(s2.get_session_turns("sess", limit=100))
        assert reloaded_count == original_count


# ===========================================================================
# 11. MemoryStore stub methods
# ===========================================================================


class TestMemoryStoreStubs:
    def test_get_summaries_returns_empty_list(self, mem_store: MemoryStore) -> None:
        mem_store.add_turn("s1", "user", "content")
        result = mem_store.get_summaries("s1")
        assert result == []

    def test_get_summaries_accepts_depth_param(self, mem_store: MemoryStore) -> None:
        result = mem_store.get_summaries("s1", depth=2, limit=10)
        assert result == []

    def test_get_learnings_returns_empty_list(self, mem_store: MemoryStore) -> None:
        assert mem_store.get_learnings() == []

    def test_get_learnings_accepts_task_type_param(self, mem_store: MemoryStore) -> None:
        assert mem_store.get_learnings(task_type="coding", limit=5) == []

    def test_save_learning_does_not_raise(self, mem_store: MemoryStore) -> None:
        mem_store.save_learning({"task_type": "coding", "lesson": "use generators"})

    def test_save_learning_does_not_affect_turns(self, mem_store: MemoryStore) -> None:
        mem_store.add_turn("s1", "user", "before")
        mem_store.save_learning({"lesson": "irrelevant"})
        turns = mem_store.get_session_turns("s1")
        assert len(turns) == 1


# ===========================================================================
# 12. MemoryStore.search edge cases
# ===========================================================================


class TestMemoryStoreSearchEdgeCases:
    def test_search_empty_store_returns_empty_list(self, mem_store: MemoryStore) -> None:
        assert mem_store.search("anything") == []

    def test_search_with_limit_zero_returns_empty(self, mem_store: MemoryStore) -> None:
        mem_store.add_turn("s1", "user", "hello world")
        results = mem_store.search("hello", limit=0)
        assert results == []

    def test_search_with_no_match_returns_empty(self, mem_store: MemoryStore) -> None:
        mem_store.add_turn("s1", "user", "completely different")
        assert mem_store.search("xyzzy_not_here") == []

    def test_search_special_characters_does_not_raise(self, mem_store: MemoryStore) -> None:
        mem_store.add_turn("s1", "user", "hello world")
        # These patterns look like SQL or regex injection
        for query in ["'; DROP TABLE turns; --", ".*", "[a-z]+", "%hello%", "\\"]:
            results = mem_store.search(query)
            assert isinstance(results, list)

    def test_search_very_long_query_does_not_raise(self, mem_store: MemoryStore) -> None:
        mem_store.add_turn("s1", "user", "hello")
        long_query = "word " * 5000
        results = mem_store.search(long_query)
        assert isinstance(results, list)

    def test_search_unicode_query_does_not_raise(self, mem_store: MemoryStore) -> None:
        mem_store.add_turn("s1", "user", "日本語テスト")
        results = mem_store.search("テスト")
        assert isinstance(results, list)


# ===========================================================================
# 13. ConversationTurn.to_dict with None timestamp
# ===========================================================================


class TestConversationTurnNoneTimestamp:
    def test_to_dict_none_timestamp_emits_none(self) -> None:
        turn = ConversationTurn(session_id="s", role="user", content="hi", timestamp=None)
        d = turn.to_dict()
        assert d["timestamp"] is None

    def test_from_dict_none_timestamp_uses_default(self) -> None:
        turn = ConversationTurn.from_dict({"timestamp": None})
        assert isinstance(turn.timestamp, datetime)


# ===========================================================================
# 14. MemoryStore JSON file format and persistence details
# ===========================================================================


class TestMemoryStorePersistenceDetails:
    def test_json_file_is_pretty_printed(self, tmp_path: Path) -> None:
        path = str(tmp_path / "pretty.json")
        store = MemoryStore(store_path=path)
        store.add_turn("s1", "user", "content")
        raw = Path(path).read_text(encoding="utf-8")
        # indent=2 means the file contains newlines and leading spaces
        assert "\n" in raw
        assert "  " in raw

    def test_multiple_reload_cycles_do_not_duplicate_turns(self, tmp_path: Path) -> None:
        path = str(tmp_path / "reload_cycle.json")
        s1 = MemoryStore(store_path=path)
        s1.add_turn("sess", "user", "original")

        # Reload and add another turn
        s2 = MemoryStore(store_path=path)
        s2.add_turn("sess", "user", "second")

        # Reload again
        s3 = MemoryStore(store_path=path)
        turns = s3.get_session_turns("sess")
        assert len(turns) == 2
        contents = [t.content for t in turns]
        assert "original" in contents
        assert "second" in contents

    def test_add_turn_special_characters_round_trip(self, mem_store: MemoryStore) -> None:
        content = 'He said "hello" & she replied <world> \n\t with a \\backslash'
        mem_store.add_turn("s1", "user", content)
        turns = mem_store.get_session_turns("s1")
        assert turns[0].content == content

    def test_add_turn_unicode_content_round_trip(self, mem_store: MemoryStore) -> None:
        content = "Emoji: rocket-launch-time, CJK: 你好世界, Arabic: مرحبا"
        mem_store.add_turn("s1", "user", content)
        turns = mem_store.get_session_turns("s1")
        assert turns[0].content == content

    def test_clear_empty_session_does_not_corrupt_file(self, tmp_path: Path) -> None:
        path = str(tmp_path / "clear_empty.json")
        store = MemoryStore(store_path=path)
        store.add_turn("s1", "user", "real turn")
        store.clear_session("s2")  # s2 does not exist

        reloaded = MemoryStore(store_path=path)
        turns = reloaded.get_session_turns("s1")
        assert len(turns) == 1

    def test_get_session_token_count_approx_chars_over_four(
        self, mem_store: MemoryStore
    ) -> None:
        mem_store.add_turn("s1", "user", "a" * 400)
        count = mem_store.get_session_token_count("s1")
        # 400 chars // 4 == 100
        assert count == 100

    def test_get_session_token_count_empty_session_returns_zero(
        self, mem_store: MemoryStore
    ) -> None:
        count = mem_store.get_session_token_count("ghost-session")
        assert count == 0

    def test_get_session_token_count_multiple_turns_summed(
        self, mem_store: MemoryStore
    ) -> None:
        for _ in range(4):
            mem_store.add_turn("s1", "user", "a" * 400)
        count = mem_store.get_session_token_count("s1")
        # 4 x 400 = 1600 chars // 4 == 400
        assert count == 400


# ===========================================================================
# 15. MemoryStore.get_recent_turns ordering invariant
# ===========================================================================


class TestMemoryStoreRecentTurnsOrdering:
    def test_recent_turns_preserves_insertion_order(self, mem_store: MemoryStore) -> None:
        """get_recent_turns returns turns in insertion order (FIFO), not reversed."""
        for i in range(5):
            mem_store.add_turn("s1", "user", f"msg-{i}")

        recent = mem_store.get_recent_turns(limit=5)
        contents = [t.content for t in recent]
        assert contents == [f"msg-{i}" for i in range(5)]

    def test_get_session_turns_limit_returns_tail(self, mem_store: MemoryStore) -> None:
        for i in range(10):
            mem_store.add_turn("s1", "user", f"item-{i}")

        turns = mem_store.get_session_turns("s1", limit=3)
        contents = [t.content for t in turns]
        # The most recent 3 turns
        assert contents == ["item-7", "item-8", "item-9"]

    def test_get_recent_turns_across_sessions_returns_tail(
        self, mem_store: MemoryStore
    ) -> None:
        for i in range(8):
            mem_store.add_turn(f"session-{i % 2}", "user", f"msg-{i}")

        recent = mem_store.get_recent_turns(limit=4)
        contents = [t.content for t in recent]
        assert contents == ["msg-4", "msg-5", "msg-6", "msg-7"]


# ===========================================================================
# 16. MemoryStore concurrent safety (JSON store uses a threading.Lock)
# ===========================================================================


class TestMemoryStoreConcurrentWritesSafe:
    def test_concurrent_add_turns_no_data_loss(self, tmp_path: Path) -> None:
        store = MemoryStore(store_path=str(tmp_path / "concurrent.json"))
        errors: list[Exception] = []

        def add_batch(session_id: str, count: int) -> None:
            try:
                for i in range(count):
                    store.add_turn(session_id, "user", f"msg-{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=add_batch, args=(f"sess-{j}", 20)) for j in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert errors == [], f"Unexpected errors: {errors}"
        total = sum(len(store.get_session_turns(f"sess-{j}")) for j in range(5))
        assert total == 100
