"""Tests for SQLiteMemoryStore.delete_turn / set_turn_pinned and the
ResilientMemoryStore delegation of both, plus pinned-turn cleanup exemption.
"""

from __future__ import annotations

import pytest

from missy.memory.resilient import ResilientMemoryStore
from missy.memory.sqlite_store import ConversationTurn, SQLiteMemoryStore


@pytest.fixture
def store(tmp_path):
    return SQLiteMemoryStore(str(tmp_path / "memory.db"))


class TestDeleteTurn:
    def test_delete_existing_turn_returns_true(self, store):
        turn = ConversationTurn.new("sess1", "user", "hello")
        store.add_turn(turn)
        assert store.delete_turn(turn.id) is True
        assert store.get_session_turns("sess1") == []

    def test_delete_unknown_turn_returns_false(self, store):
        assert store.delete_turn("does-not-exist") is False

    def test_delete_removes_from_search_index(self, store):
        turn = ConversationTurn.new("sess1", "user", "unique-marker-text")
        store.add_turn(turn)
        assert store.search("unique-marker-text")
        store.delete_turn(turn.id)
        assert store.search("unique-marker-text") == []


class TestSetTurnPinned:
    def test_pin_existing_turn_returns_true(self, store):
        turn = ConversationTurn.new("sess1", "user", "hello")
        store.add_turn(turn)
        assert store.set_turn_pinned(turn.id, True) is True
        [reloaded] = store.get_session_turns("sess1")
        assert reloaded.metadata.get("pinned") is True

    def test_unpin_clears_flag(self, store):
        turn = ConversationTurn.new("sess1", "user", "hello")
        store.add_turn(turn)
        store.set_turn_pinned(turn.id, True)
        store.set_turn_pinned(turn.id, False)
        [reloaded] = store.get_session_turns("sess1")
        assert "pinned" not in reloaded.metadata

    def test_pin_unknown_turn_returns_false(self, store):
        assert store.set_turn_pinned("does-not-exist", True) is False

    def test_pin_preserves_other_metadata(self, store):
        turn = ConversationTurn.new("sess1", "user", "hello")
        turn.metadata = {"other": "value"}
        store.add_turn(turn)
        store.set_turn_pinned(turn.id, True)
        [reloaded] = store.get_session_turns("sess1")
        assert reloaded.metadata == {"other": "value", "pinned": True}


class TestCleanupRespectsPinned:
    def test_pinned_turn_survives_cleanup(self, store):
        import json
        from datetime import UTC, datetime, timedelta

        old_ts = (datetime.now(UTC) - timedelta(days=90)).isoformat()
        turn = ConversationTurn.new("sess1", "user", "old but pinned")
        conn = store._conn()
        conn.execute(
            "INSERT INTO turns (id, session_id, timestamp, role, content, provider, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (turn.id, "sess1", old_ts, "user", "old but pinned", "", json.dumps({"pinned": True})),
        )
        conn.commit()

        deleted = store.cleanup(older_than_days=30)

        assert deleted == 0
        assert len(store.get_session_turns("sess1")) == 1

    def test_unpinned_old_turn_is_deleted(self, store):
        import json
        from datetime import UTC, datetime, timedelta

        old_ts = (datetime.now(UTC) - timedelta(days=90)).isoformat()
        turn = ConversationTurn.new("sess1", "user", "old and unpinned")
        conn = store._conn()
        conn.execute(
            "INSERT INTO turns (id, session_id, timestamp, role, content, provider, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (turn.id, "sess1", old_ts, "user", "old and unpinned", "", json.dumps({})),
        )
        conn.commit()

        deleted = store.cleanup(older_than_days=30)

        assert deleted == 1
        assert store.get_session_turns("sess1") == []


class TestResilientDelegation:
    def test_delete_turn_delegates_to_primary(self, store):
        resilient = ResilientMemoryStore(store)
        turn = ConversationTurn.new("sess1", "user", "hello")
        resilient.add_turn(turn)
        assert resilient.delete_turn(turn.id) is True
        assert store.get_session_turns("sess1") == []

    def test_delete_turn_removes_from_cache_even_on_primary_failure(self):
        class FailingStore:
            def add_turn(self, turn):
                pass

            def delete_turn(self, turn_id):
                raise RuntimeError("db unavailable")

        resilient = ResilientMemoryStore(FailingStore())
        turn = ConversationTurn.new("sess1", "user", "hello")
        resilient.add_turn(turn)
        assert resilient.delete_turn(turn.id) is False
        assert resilient._cache["sess1"] == []

    def test_set_turn_pinned_delegates_to_primary(self, store):
        resilient = ResilientMemoryStore(store)
        turn = ConversationTurn.new("sess1", "user", "hello")
        resilient.add_turn(turn)
        assert resilient.set_turn_pinned(turn.id, True) is True
        [reloaded] = store.get_session_turns("sess1")
        assert reloaded.metadata.get("pinned") is True

    def test_set_turn_pinned_returns_false_on_primary_failure(self):
        class FailingStore:
            def set_turn_pinned(self, turn_id, pinned):
                raise RuntimeError("db unavailable")

        resilient = ResilientMemoryStore(FailingStore())
        assert resilient.set_turn_pinned("t1", True) is False
