"""DATA-03: ResilientMemoryStore's fallback cache is bounded and consistent
with edits, pins and retention."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from missy.memory.resilient import ResilientMemoryStore
from missy.memory.sqlite_store import ConversationTurn


def _failing_primary() -> MagicMock:
    p = MagicMock()
    p.get_session_turns.side_effect = RuntimeError("db down")
    return p


def test_cache_is_bounded_per_session_and_in_total():
    store = ResilientMemoryStore(MagicMock(), max_cached_turns_per_session=5, max_cached_sessions=3)
    for i in range(10_000):
        store.add_turn(ConversationTurn.new(f"s{i % 4}", "user", f"m{i}"))
    assert len(store._cache) == 3
    assert all(len(turns) <= 5 for turns in store._cache.values())
    assert "s0" not in store._cache  # least recently written session evicted


def test_edit_is_reflected_in_fallback_reads():
    primary = MagicMock()
    store = ResilientMemoryStore(primary)
    turn = ConversationTurn.new("s1", "user", "my key is sk-secret")
    store.add_turn(turn)
    primary.update_turn_content.return_value = True
    assert store.update_turn_content(turn.id, "my key is [redacted]") is True
    primary.get_session_turns.side_effect = RuntimeError("db down")
    assert store.get_session_turns("s1")[0].content == "my key is [redacted]"


def test_edit_queued_when_primary_down():
    primary = MagicMock()
    primary.update_turn_content.side_effect = RuntimeError("db down")
    store = ResilientMemoryStore(primary)
    store.update_turn_content("t1", "x")
    assert ("update_turn_content", ("t1", "x")) in store._pending_ops


def test_pinned_turn_survives_cache_pruning():
    store = ResilientMemoryStore(MagicMock())
    old = ConversationTurn.new("s1", "user", "keep me")
    old.timestamp = datetime(2020, 1, 1, tzinfo=UTC).isoformat()
    old.metadata["pinned"] = True
    store.add_turn(old)
    store.cleanup(older_than_days=30)
    assert store._cache["s1"] == [old]


def test_clear_session_full_returns_primary_counts():
    primary = MagicMock()
    primary.clear_session_full.return_value = {"turns": 3, "summaries": 1}
    store = ResilientMemoryStore(primary)
    assert store.clear_session_full("s1") == {"turns": 3, "summaries": 1}
