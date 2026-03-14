"""Tests for missy.memory.store."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from missy.memory.store import ConversationTurn, MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(store_path=str(tmp_path / "memory.json"))


class TestConversationTurnToDict:
    def test_all_keys_present(self):
        turn = ConversationTurn(session_id="s", role="user", content="hi")
        d = turn.to_dict()
        assert {"id", "session_id", "timestamp", "role", "content", "provider"} == set(d.keys())

    def test_timestamp_is_isoformat_string(self):
        ts = datetime(2025, 6, 1, 12, 0, 0)
        turn = ConversationTurn(timestamp=ts)
        assert turn.to_dict()["timestamp"] == ts.isoformat()

    def test_values_match_attributes(self):
        turn = ConversationTurn(
            session_id="abc", role="assistant", content="hello", provider="openai"
        )
        d = turn.to_dict()
        assert d["session_id"] == "abc"
        assert d["role"] == "assistant"
        assert d["content"] == "hello"
        assert d["provider"] == "openai"


class TestConversationTurnFromDict:
    def test_round_trip(self):
        original = ConversationTurn(session_id="s1", role="user", content="hi", provider="")
        restored = ConversationTurn.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.session_id == original.session_id
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.timestamp == original.timestamp

    def test_missing_timestamp_defaults_gracefully(self):
        turn = ConversationTurn.from_dict({"session_id": "s", "role": "user", "content": "x"})
        assert isinstance(turn.timestamp, datetime)

    def test_missing_id_generates_new_uuid(self):
        turn = ConversationTurn.from_dict({"session_id": "s"})
        assert isinstance(turn.id, str)
        assert len(turn.id) == 36


class TestMemoryStoreAddTurn:
    def test_add_turn_returns_conversation_turn(self, store: MemoryStore):
        turn = store.add_turn("s1", "user", "Hello")
        assert isinstance(turn, ConversationTurn)
        assert turn.session_id == "s1"
        assert turn.role == "user"
        assert turn.content == "Hello"

    def test_add_turn_persists_to_file(self, store: MemoryStore):
        store.add_turn("s1", "user", "Hello")
        data = json.loads(store.store_path.read_text())
        assert len(data) == 1
        assert data[0]["content"] == "Hello"

    def test_add_turn_accumulates(self, store: MemoryStore):
        store.add_turn("s1", "user", "A")
        store.add_turn("s1", "assistant", "B")
        assert len(store.get_recent_turns()) == 2


class TestMemoryStoreGetSessionTurns:
    def test_returns_only_matching_session(self, store: MemoryStore):
        store.add_turn("s1", "user", "A")
        store.add_turn("s2", "user", "B")
        turns = store.get_session_turns("s1")
        assert len(turns) == 1
        assert turns[0].content == "A"

    def test_limit_is_respected(self, store: MemoryStore):
        for i in range(10):
            store.add_turn("s1", "user", str(i))
        turns = store.get_session_turns("s1", limit=3)
        assert len(turns) == 3
        # Last 3 are returned
        assert turns[-1].content == "9"

    def test_returns_empty_for_unknown_session(self, store: MemoryStore):
        assert store.get_session_turns("unknown") == []

    def test_default_limit_is_50(self, store: MemoryStore):
        for i in range(60):
            store.add_turn("s1", "user", str(i))
        turns = store.get_session_turns("s1")
        assert len(turns) == 50


class TestMemoryStoreGetRecentTurns:
    def test_returns_last_n_turns_across_sessions(self, store: MemoryStore):
        store.add_turn("s1", "user", "A")
        store.add_turn("s2", "user", "B")
        store.add_turn("s1", "assistant", "C")
        turns = store.get_recent_turns(limit=2)
        assert len(turns) == 2
        assert turns[-1].content == "C"

    def test_default_limit_is_10(self, store: MemoryStore):
        for i in range(15):
            store.add_turn("s1", "user", str(i))
        turns = store.get_recent_turns()
        assert len(turns) == 10

    def test_fewer_turns_than_limit_returns_all(self, store: MemoryStore):
        store.add_turn("s1", "user", "only")
        turns = store.get_recent_turns(limit=5)
        assert len(turns) == 1


class TestMemoryStoreClearSession:
    def test_clear_removes_session_turns(self, store: MemoryStore):
        store.add_turn("s1", "user", "A")
        store.add_turn("s1", "assistant", "B")
        store.clear_session("s1")
        assert store.get_session_turns("s1") == []

    def test_clear_preserves_other_sessions(self, store: MemoryStore):
        store.add_turn("s1", "user", "A")
        store.add_turn("s2", "user", "B")
        store.clear_session("s1")
        assert len(store.get_session_turns("s2")) == 1

    def test_clear_nonexistent_session_is_noop(self, store: MemoryStore):
        store.add_turn("s1", "user", "A")
        store.clear_session("s2")  # Should not raise
        assert len(store.get_recent_turns()) == 1

    def test_clear_session_persists(self, store: MemoryStore):
        store.add_turn("s1", "user", "A")
        store.clear_session("s1")
        reloaded = MemoryStore(store_path=str(store.store_path))
        assert reloaded.get_session_turns("s1") == []


class TestMemoryStorePersistence:
    def test_data_survives_reload(self, tmp_path: Path):
        path = str(tmp_path / "m.json")
        s1 = MemoryStore(store_path=path)
        s1.add_turn("sess", "user", "remember me")

        s2 = MemoryStore(store_path=path)
        turns = s2.get_session_turns("sess")
        assert len(turns) == 1
        assert turns[0].content == "remember me"

    def test_missing_file_starts_empty(self, tmp_path: Path):
        store = MemoryStore(store_path=str(tmp_path / "nonexistent.json"))
        assert store.get_recent_turns() == []

    def test_malformed_file_starts_empty(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("NOT JSON", encoding="utf-8")
        store = MemoryStore(store_path=str(path))
        assert store.get_recent_turns() == []
