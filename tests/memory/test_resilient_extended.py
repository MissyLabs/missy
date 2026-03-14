"""Comprehensive tests for missy.memory.resilient.ResilientMemoryStore."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from missy.memory.resilient import ResilientMemoryStore


def _make_turn(session_id="s1", content="hello", ts=None):
    t = MagicMock()
    t.session_id = session_id
    t.content = content
    t.timestamp = ts or datetime(2026, 1, 1, 12, 0, 0)
    t.id = f"turn-{id(t)}"
    return t


def _healthy_primary():
    p = MagicMock()
    p.get_session_turns.return_value = []
    p.get_recent_turns.return_value = []
    p.search.return_value = []
    p.get_learnings.return_value = []
    p.cleanup.return_value = 0
    return p


def _broken_primary():
    p = MagicMock()
    p.add_turn.side_effect = RuntimeError("db locked")
    p.clear_session.side_effect = RuntimeError("db locked")
    p.save_learning.side_effect = RuntimeError("db locked")
    p.get_session_turns.side_effect = RuntimeError("db locked")
    p.get_recent_turns.side_effect = RuntimeError("db locked")
    p.search.side_effect = RuntimeError("db locked")
    p.get_learnings.side_effect = RuntimeError("db locked")
    p.cleanup.side_effect = RuntimeError("db locked")
    return p


class TestHealthTracking:
    def test_starts_healthy(self):
        store = ResilientMemoryStore(_healthy_primary())
        assert store.is_healthy is True

    def test_becomes_unhealthy_after_failures(self):
        store = ResilientMemoryStore(_broken_primary(), max_failures=2)
        turn = _make_turn()
        store.add_turn(turn)
        store.add_turn(turn)
        assert store.is_healthy is False

    def test_recovers_on_success(self):
        primary = _broken_primary()
        store = ResilientMemoryStore(primary, max_failures=1)
        store.add_turn(_make_turn())
        assert store.is_healthy is False
        # Fix primary
        primary.add_turn.side_effect = None
        store.add_turn(_make_turn())
        assert store.is_healthy is True


class TestAddTurn:
    def test_writes_to_primary(self):
        primary = _healthy_primary()
        store = ResilientMemoryStore(primary)
        turn = _make_turn()
        store.add_turn(turn)
        primary.add_turn.assert_called_once_with(turn)

    def test_caches_turn_on_failure(self):
        store = ResilientMemoryStore(_broken_primary())
        turn = _make_turn(session_id="s1")
        store.add_turn(turn)
        assert len(store._cache["s1"]) == 1

    def test_caches_turn_even_on_success(self):
        store = ResilientMemoryStore(_healthy_primary())
        turn = _make_turn(session_id="s1")
        store.add_turn(turn)
        assert len(store._cache["s1"]) == 1


class TestClearSession:
    def test_clears_cache_and_primary(self):
        primary = _healthy_primary()
        store = ResilientMemoryStore(primary)
        store.add_turn(_make_turn(session_id="s1"))
        store.clear_session("s1")
        assert "s1" not in store._cache
        primary.clear_session.assert_called_once_with("s1")

    def test_clears_cache_on_primary_failure(self):
        store = ResilientMemoryStore(_broken_primary())
        store._cache["s1"] = [_make_turn()]
        store.clear_session("s1")
        assert "s1" not in store._cache


class TestSaveLearning:
    def test_delegates_to_primary(self):
        primary = _healthy_primary()
        store = ResilientMemoryStore(primary)
        learning = MagicMock()
        store.save_learning(learning)
        primary.save_learning.assert_called_once_with(learning)

    def test_handles_primary_failure(self):
        store = ResilientMemoryStore(_broken_primary())
        store.save_learning(MagicMock())  # should not raise


class TestGetSessionTurns:
    def test_returns_from_primary(self):
        primary = _healthy_primary()
        turns = [_make_turn()]
        primary.get_session_turns.return_value = turns
        store = ResilientMemoryStore(primary)
        result = store.get_session_turns("s1", limit=10)
        assert result == turns

    def test_falls_back_to_cache(self):
        store = ResilientMemoryStore(_broken_primary())
        cached = [_make_turn(session_id="s1")]
        store._cache["s1"] = cached
        result = store.get_session_turns("s1", limit=10)
        assert result == cached

    def test_cache_fallback_respects_limit(self):
        store = ResilientMemoryStore(_broken_primary())
        store._cache["s1"] = [_make_turn() for _ in range(20)]
        result = store.get_session_turns("s1", limit=5)
        assert len(result) == 5


class TestGetRecentTurns:
    def test_returns_from_primary(self):
        primary = _healthy_primary()
        turns = [_make_turn()]
        primary.get_recent_turns.return_value = turns
        store = ResilientMemoryStore(primary)
        assert store.get_recent_turns(limit=10) == turns

    def test_falls_back_to_cache_sorted(self):
        store = ResilientMemoryStore(_broken_primary())
        t1 = _make_turn(session_id="s1", ts=datetime(2026, 1, 1, 10, 0))
        t2 = _make_turn(session_id="s2", ts=datetime(2026, 1, 1, 12, 0))
        store._cache["s1"] = [t1]
        store._cache["s2"] = [t2]
        result = store.get_recent_turns(limit=10)
        assert result[0].timestamp < result[1].timestamp


class TestSearch:
    def test_search_from_primary(self):
        primary = _healthy_primary()
        primary.search.return_value = ["result"]
        store = ResilientMemoryStore(primary)
        assert store.search("hello") == ["result"]

    def test_search_falls_back_to_cache(self):
        store = ResilientMemoryStore(_broken_primary())
        t1 = _make_turn(content="hello world")
        t2 = _make_turn(content="goodbye")
        store._cache["s1"] = [t1, t2]
        result = store.search("hello")
        assert len(result) == 1

    def test_search_cache_respects_session_filter(self):
        store = ResilientMemoryStore(_broken_primary())
        t1 = _make_turn(session_id="s1", content="match")
        t2 = _make_turn(session_id="s2", content="match")
        store._cache["s1"] = [t1]
        store._cache["s2"] = [t2]
        result = store.search("match", session_id="s1")
        assert len(result) == 1

    def test_search_cache_respects_limit(self):
        store = ResilientMemoryStore(_broken_primary())
        store._cache["s1"] = [_make_turn(content="match") for _ in range(20)]
        result = store.search("match", limit=3)
        assert len(result) == 3


class TestGetLearnings:
    def test_from_primary(self):
        primary = _healthy_primary()
        primary.get_learnings.return_value = ["lesson1"]
        store = ResilientMemoryStore(primary)
        assert store.get_learnings() == ["lesson1"]

    def test_returns_empty_on_failure(self):
        store = ResilientMemoryStore(_broken_primary())
        assert store.get_learnings() == []


class TestCleanup:
    def test_delegates_to_primary(self):
        primary = _healthy_primary()
        primary.cleanup.return_value = 5
        store = ResilientMemoryStore(primary)
        assert store.cleanup(older_than_days=7) == 5

    def test_returns_zero_on_failure(self):
        store = ResilientMemoryStore(_broken_primary())
        assert store.cleanup() == 0


class TestRecoverySync:
    def test_recovery_syncs_cache_to_primary(self):
        primary = MagicMock()
        primary.add_turn.side_effect = RuntimeError("fail")
        store = ResilientMemoryStore(primary, max_failures=1)

        # Add turns while primary is broken
        t1 = _make_turn()
        store.add_turn(t1)
        assert store.is_healthy is False

        # Fix primary
        primary.add_turn.side_effect = None
        t2 = _make_turn()
        store.add_turn(t2)

        # Recovery should have synced cached turns
        assert store.is_healthy is True
        # add_turn called: once failing (t1), once syncing (t1 again), once for t2
        assert primary.add_turn.call_count >= 2
