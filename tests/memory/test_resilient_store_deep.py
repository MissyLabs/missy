"""Deep behavioural tests for ResilientMemoryStore.

Focuses on areas not covered by test_resilient_extended.py:
- Exact fallback mechanics and multi-session cache interaction
- Sync semantics: which turns are replayed, partial sync failures
- Search edge cases: empty cache, query with no matches, session filter across
  multi-session caches
- Learning operations: fallback returns empty, primary call args forwarded
- Cleanup: default arg, failure handling, return value propagation
- Health state transitions: exact failure counting, recovery resets counter
- Cache integrity: turns are never duplicated in the in-memory dict
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from missy.memory.resilient import ResilientMemoryStore

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _turn(session_id: str = "s1", content: str = "hello", ts: datetime | None = None) -> MagicMock:
    t = MagicMock()
    t.session_id = session_id
    t.content = content
    t.timestamp = ts or datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    t.id = f"turn-{id(t)}"
    return t


def _healthy() -> MagicMock:
    p = MagicMock()
    p.get_session_turns.return_value = []
    p.get_recent_turns.return_value = []
    p.search.return_value = []
    p.get_learnings.return_value = []
    p.cleanup.return_value = 0
    return p


def _broken(exc: Exception | None = None) -> MagicMock:
    err = exc or RuntimeError("db unavailable")
    p = MagicMock()
    for method in (
        "add_turn",
        "clear_session",
        "save_learning",
        "get_session_turns",
        "get_recent_turns",
        "search",
        "get_learnings",
        "cleanup",
    ):
        getattr(p, method).side_effect = err
    return p


# ---------------------------------------------------------------------------
# 1. Fallback behaviour — primary store failure
# ---------------------------------------------------------------------------


class TestFallbackBehaviour:
    """Primary store failure falls back to the in-memory cache."""

    def test_get_session_turns_returns_cached_when_primary_raises(self):
        store = ResilientMemoryStore(_broken())
        t = _turn(session_id="s1")
        store._cache["s1"] = [t]

        result = store.get_session_turns("s1")
        assert result == [t]

    def test_get_session_turns_empty_cache_for_unknown_session(self):
        store = ResilientMemoryStore(_broken())
        result = store.get_session_turns("no-such-session")
        assert result == []

    def test_get_recent_turns_aggregates_all_sessions_on_failure(self):
        store = ResilientMemoryStore(_broken())
        t1 = _turn("s1", ts=datetime(2026, 1, 1, 10, 0, tzinfo=UTC))
        t2 = _turn("s2", ts=datetime(2026, 1, 1, 11, 0, tzinfo=UTC))
        t3 = _turn("s3", ts=datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        store._cache.update({"s1": [t1], "s2": [t2], "s3": [t3]})

        result = store.get_recent_turns(limit=10)
        # Must be chronologically sorted
        assert result == [t1, t2, t3]

    def test_get_recent_turns_limit_applied_after_sort(self):
        store = ResilientMemoryStore(_broken())
        turns = [_turn("s1", ts=datetime(2026, 1, 1, i, 0, tzinfo=UTC)) for i in range(10)]
        store._cache["s1"] = turns

        result = store.get_recent_turns(limit=3)
        assert len(result) == 3
        # Latest three
        assert result == turns[-3:]

    def test_add_turn_still_stores_in_cache_when_primary_fails(self):
        store = ResilientMemoryStore(_broken())
        t = _turn(session_id="sess-x")
        store.add_turn(t)
        assert t in store._cache["sess-x"]

    def test_multiple_failed_adds_accumulate_in_cache(self):
        store = ResilientMemoryStore(_broken())
        turns = [_turn(session_id="s1") for _ in range(5)]
        for t in turns:
            store.add_turn(t)
        assert len(store._cache["s1"]) == 5

    def test_adds_to_different_sessions_partitioned_correctly(self):
        store = ResilientMemoryStore(_broken())
        ta = _turn(session_id="a")
        tb = _turn(session_id="b")
        store.add_turn(ta)
        store.add_turn(tb)
        assert store._cache["a"] == [ta]
        assert store._cache["b"] == [tb]

    def test_clear_session_removes_cache_even_when_primary_fails(self):
        store = ResilientMemoryStore(_broken())
        store._cache["s1"] = [_turn()]
        store.clear_session("s1")
        assert "s1" not in store._cache

    def test_clear_nonexistent_session_does_not_raise(self):
        store = ResilientMemoryStore(_broken())
        store.clear_session("ghost")  # must not raise

    def test_save_learning_failure_marked_unhealthy(self):
        store = ResilientMemoryStore(_broken(), max_failures=1)
        store.save_learning(MagicMock())
        assert store.is_healthy is False

    def test_cleanup_returns_zero_on_failure(self):
        store = ResilientMemoryStore(_broken())
        assert store.cleanup(older_than_days=7) == 0

    def test_cleanup_default_arg_passes_through(self):
        primary = _healthy()
        primary.cleanup.return_value = 4
        store = ResilientMemoryStore(primary)
        result = store.cleanup()
        primary.cleanup.assert_called_once_with(older_than_days=30)
        assert result == 4

    def test_get_learnings_returns_empty_list_on_failure(self):
        store = ResilientMemoryStore(_broken())
        result = store.get_learnings(task_type="coding", limit=3)
        assert result == []


# ---------------------------------------------------------------------------
# 2. Sync: backup to primary after recovery
# ---------------------------------------------------------------------------


class TestSyncAfterRecovery:
    """After the primary recovers, cached turns are replayed into it."""

    def test_recovery_replays_all_cached_sessions(self):
        primary = MagicMock()
        primary.add_turn.side_effect = RuntimeError("down")

        store = ResilientMemoryStore(primary, max_failures=1)
        ta = _turn(session_id="a")
        tb = _turn(session_id="b")
        store.add_turn(ta)
        store.add_turn(tb)
        assert not store.is_healthy

        # Fix the primary and trigger recovery via a successful write
        primary.add_turn.side_effect = None
        tc = _turn(session_id="c")
        store.add_turn(tc)

        # _sync_cache_to_primary calls add_turn for ta and tb (in cache),
        # then the normal path calls it for tc.
        calls = [c[0][0] for c in primary.add_turn.call_args_list]
        assert ta in calls
        assert tb in calls
        assert tc in calls

    def test_partial_sync_failure_does_not_crash(self):
        """A sync error for one turn should not abort recovery."""
        primary = MagicMock()
        primary.add_turn.side_effect = RuntimeError("down")

        store = ResilientMemoryStore(primary, max_failures=1)
        turns = [_turn() for _ in range(3)]
        for t in turns:
            store.add_turn(t)

        # Fix primary but make it fail on the first sync call only
        call_count = {"n": 0}
        list(turns)

        def flaky_add(t):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("sync hiccup")

        primary.add_turn.side_effect = flaky_add

        # Should not raise even though first sync turn fails
        trigger = _turn(session_id="new")
        try:
            store.add_turn(trigger)
        except Exception:
            pytest.fail("add_turn raised unexpectedly during partial sync failure")

    def test_recovery_resets_failure_counter(self):
        primary = MagicMock()
        primary.add_turn.side_effect = RuntimeError("fail")

        store = ResilientMemoryStore(primary, max_failures=2)
        store.add_turn(_turn())
        store.add_turn(_turn())
        assert store._failures == 2

        primary.add_turn.side_effect = None
        store.add_turn(_turn())
        assert store._failures == 0

    def test_is_healthy_true_after_recovery(self):
        primary = MagicMock()
        primary.add_turn.side_effect = RuntimeError("fail")

        store = ResilientMemoryStore(primary, max_failures=1)
        store.add_turn(_turn())
        assert store.is_healthy is False

        primary.add_turn.side_effect = None
        store.add_turn(_turn())
        assert store.is_healthy is True

    def test_multiple_recoveries_possible(self):
        """A store can go unhealthy, recover, go unhealthy again, and recover."""
        primary = MagicMock()
        store = ResilientMemoryStore(primary, max_failures=1)

        primary.add_turn.side_effect = RuntimeError("fail")
        store.add_turn(_turn())
        assert not store.is_healthy

        primary.add_turn.side_effect = None
        store.add_turn(_turn())
        assert store.is_healthy

        primary.add_turn.side_effect = RuntimeError("fail again")
        store.add_turn(_turn())
        assert not store.is_healthy

        primary.add_turn.side_effect = None
        store.add_turn(_turn())
        assert store.is_healthy


# ---------------------------------------------------------------------------
# 3. Search
# ---------------------------------------------------------------------------


class TestSearch:
    """Search uses primary when healthy; falls back to keyword scan of cache."""

    def test_primary_search_result_returned_unchanged(self):
        primary = _healthy()
        primary.search.return_value = ["r1", "r2"]
        store = ResilientMemoryStore(primary)
        assert store.search("term") == ["r1", "r2"]

    def test_search_passes_args_to_primary(self):
        primary = _healthy()
        store = ResilientMemoryStore(primary)
        store.search("q", limit=7, session_id="s9")
        primary.search.assert_called_once_with("q", limit=7, session_id="s9")

    def test_fallback_case_insensitive_substring_match(self):
        # The implementation lowercases both sides and does a single `in` check.
        # "quick brown" is a contiguous substring of "the quick brown fox", so it
        # must match regardless of the original casing.
        store = ResilientMemoryStore(_broken())
        t = _turn(content="The Quick Brown Fox")
        store._cache["s1"] = [t]
        result = store.search("quick brown")
        assert result == [t]

    def test_fallback_non_contiguous_words_do_not_match(self):
        # "quick fox" is NOT a contiguous substring of "the quick brown fox",
        # so a plain `in` check must not return it.
        store = ResilientMemoryStore(_broken())
        t = _turn(content="The Quick Brown Fox")
        store._cache["s1"] = [t]
        result = store.search("quick fox")
        assert result == []

    def test_fallback_exact_substring_matches(self):
        store = ResilientMemoryStore(_broken())
        t = _turn(content="error: permission denied")
        store._cache["s1"] = [t]
        result = store.search("permission denied")
        assert result == [t]

    def test_fallback_no_match_returns_empty_list(self):
        store = ResilientMemoryStore(_broken())
        store._cache["s1"] = [_turn(content="unrelated message")]
        result = store.search("xyz-not-found")
        assert result == []

    def test_fallback_session_filter_excludes_other_sessions(self):
        store = ResilientMemoryStore(_broken())
        ta = _turn(session_id="a", content="needle")
        tb = _turn(session_id="b", content="needle")
        store._cache["a"] = [ta]
        store._cache["b"] = [tb]
        result = store.search("needle", session_id="a")
        assert result == [ta]

    def test_fallback_limit_respected(self):
        store = ResilientMemoryStore(_broken())
        store._cache["s1"] = [_turn(content="hit") for _ in range(15)]
        result = store.search("hit", limit=4)
        assert len(result) == 4

    def test_fallback_empty_cache_returns_empty_list(self):
        store = ResilientMemoryStore(_broken())
        assert store.search("anything") == []

    def test_primary_failure_marks_store_unhealthy(self):
        store = ResilientMemoryStore(_broken(), max_failures=1)
        store.search("q")
        assert not store.is_healthy


# ---------------------------------------------------------------------------
# 4. Learning operations
# ---------------------------------------------------------------------------


class TestLearningOperations:
    """save_learning and get_learnings with healthy and failing primary."""

    def test_save_learning_delegates_to_primary(self):
        primary = _healthy()
        store = ResilientMemoryStore(primary)
        learning = MagicMock()
        store.save_learning(learning)
        primary.save_learning.assert_called_once_with(learning)

    def test_save_learning_marks_healthy_on_success(self):
        primary = MagicMock()
        primary.save_learning.side_effect = RuntimeError("fail")
        store = ResilientMemoryStore(primary, max_failures=1)
        store.save_learning(MagicMock())
        assert not store.is_healthy

        primary.save_learning.side_effect = None
        store.save_learning(MagicMock())
        assert store.is_healthy

    def test_get_learnings_passes_kwargs_to_primary(self):
        primary = _healthy()
        primary.get_learnings.return_value = ["lesson"]
        store = ResilientMemoryStore(primary)
        result = store.get_learnings(task_type="refactor", limit=3)
        primary.get_learnings.assert_called_once_with(task_type="refactor", limit=3)
        assert result == ["lesson"]

    def test_get_learnings_default_args_forwarded(self):
        primary = _healthy()
        store = ResilientMemoryStore(primary)
        store.get_learnings()
        primary.get_learnings.assert_called_once_with(task_type=None, limit=5)

    def test_get_learnings_empty_list_when_primary_fails(self):
        store = ResilientMemoryStore(_broken())
        assert store.get_learnings() == []
        assert store.get_learnings(task_type="x") == []
        assert store.get_learnings(limit=100) == []

    def test_save_learning_does_not_cache_in_memory(self):
        """Learnings have no in-memory fallback — cache must stay empty."""
        store = ResilientMemoryStore(_broken())
        store.save_learning(MagicMock())
        assert store._cache == {}

    def test_get_learnings_marks_healthy_after_recovery(self):
        primary = MagicMock()
        primary.get_learnings.side_effect = RuntimeError("fail")
        store = ResilientMemoryStore(primary, max_failures=1)
        store.get_learnings()
        assert not store.is_healthy

        primary.get_learnings.side_effect = None
        primary.get_learnings.return_value = []
        store.get_learnings()
        assert store.is_healthy


# ---------------------------------------------------------------------------
# 5. Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    """cleanup delegates to primary and returns 0 on failure."""

    def test_cleanup_default_older_than_is_30(self):
        primary = _healthy()
        primary.cleanup.return_value = 0
        store = ResilientMemoryStore(primary)
        store.cleanup()
        primary.cleanup.assert_called_once_with(older_than_days=30)

    def test_cleanup_custom_age_forwarded(self):
        primary = _healthy()
        primary.cleanup.return_value = 7
        store = ResilientMemoryStore(primary)
        result = store.cleanup(older_than_days=14)
        primary.cleanup.assert_called_once_with(older_than_days=14)
        assert result == 7

    def test_cleanup_returns_row_count_from_primary(self):
        primary = _healthy()
        primary.cleanup.return_value = 42
        store = ResilientMemoryStore(primary)
        assert store.cleanup() == 42

    def test_cleanup_returns_zero_when_primary_raises(self):
        store = ResilientMemoryStore(_broken())
        assert store.cleanup(older_than_days=3) == 0

    def test_cleanup_marks_unhealthy_on_failure(self):
        store = ResilientMemoryStore(_broken(), max_failures=1)
        store.cleanup()
        assert not store.is_healthy

    def test_cleanup_recovery(self):
        primary = MagicMock()
        primary.cleanup.side_effect = RuntimeError("fail")
        store = ResilientMemoryStore(primary, max_failures=1)
        store.cleanup()
        assert not store.is_healthy

        primary.cleanup.side_effect = None
        primary.cleanup.return_value = 0
        store.cleanup()
        assert store.is_healthy

    def test_cleanup_does_not_alter_in_memory_cache(self):
        primary = _healthy()
        store = ResilientMemoryStore(primary)
        store._cache["s1"] = [_turn()]
        store.cleanup()
        assert "s1" in store._cache  # cleanup must not touch cache

    def test_cleanup_zero_age_threshold(self):
        primary = _healthy()
        primary.cleanup.return_value = 99
        store = ResilientMemoryStore(primary)
        result = store.cleanup(older_than_days=0)
        assert result == 99
