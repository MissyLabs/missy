"""Tests for SQLiteMemoryStore session metadata features."""

from __future__ import annotations

from pathlib import Path

import pytest

from missy.memory.sqlite_store import SQLiteMemoryStore, ConversationTurn


@pytest.fixture
def store(tmp_path: Path) -> SQLiteMemoryStore:
    return SQLiteMemoryStore(db_path=str(tmp_path / "test_memory.db"))


class TestRegisterSession:
    def test_register_new_session(self, store: SQLiteMemoryStore) -> None:
        store.register_session("sess-1", name="my session", provider="anthropic")
        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "sess-1"
        assert sessions[0]["name"] == "my session"
        assert sessions[0]["provider"] == "anthropic"

    def test_register_updates_existing(self, store: SQLiteMemoryStore) -> None:
        store.register_session("sess-1", name="first")
        store.register_session("sess-1", name="second")
        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["name"] == "second"

    def test_register_preserves_fields_on_empty_update(
        self, store: SQLiteMemoryStore
    ) -> None:
        store.register_session("sess-1", name="keep", provider="anthropic", channel="cli")
        store.register_session("sess-1")  # empty update
        sessions = store.list_sessions()
        assert sessions[0]["name"] == "keep"
        assert sessions[0]["provider"] == "anthropic"
        assert sessions[0]["channel"] == "cli"


class TestRenameSession:
    def test_rename_existing(self, store: SQLiteMemoryStore) -> None:
        store.register_session("sess-1", name="old")
        assert store.rename_session("sess-1", "new name")
        sessions = store.list_sessions()
        assert sessions[0]["name"] == "new name"

    def test_rename_nonexistent(self, store: SQLiteMemoryStore) -> None:
        assert not store.rename_session("no-such-session", "name")


class TestListSessions:
    def test_empty_store(self, store: SQLiteMemoryStore) -> None:
        assert store.list_sessions() == []

    def test_ordered_by_updated_at(self, store: SQLiteMemoryStore) -> None:
        store.register_session("sess-1", name="first")
        store.register_session("sess-2", name="second")
        # Re-register first to update its updated_at
        store.register_session("sess-1", name="first-updated")
        sessions = store.list_sessions()
        assert sessions[0]["session_id"] == "sess-1"
        assert sessions[1]["session_id"] == "sess-2"

    def test_limit(self, store: SQLiteMemoryStore) -> None:
        for i in range(10):
            store.register_session(f"sess-{i}")
        sessions = store.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_all_fields_present(self, store: SQLiteMemoryStore) -> None:
        store.register_session("sess-1", name="test", provider="openai", channel="discord")
        s = store.list_sessions()[0]
        assert "session_id" in s
        assert "name" in s
        assert "created_at" in s
        assert "updated_at" in s
        assert "turn_count" in s
        assert "provider" in s
        assert "channel" in s


class TestResolveSessionName:
    def test_resolve_existing(self, store: SQLiteMemoryStore) -> None:
        store.register_session("sess-abc-123", name="my-debug-session")
        assert store.resolve_session_name("my-debug-session") == "sess-abc-123"

    def test_resolve_nonexistent(self, store: SQLiteMemoryStore) -> None:
        assert store.resolve_session_name("nonexistent") is None


class TestUpdateTurnCount:
    def test_count_reflects_turns(self, store: SQLiteMemoryStore) -> None:
        store.register_session("sess-1")
        store.add_turn(ConversationTurn.new("sess-1", "user", "hello"))
        store.add_turn(ConversationTurn.new("sess-1", "assistant", "hi"))
        store.update_session_turn_count("sess-1")
        sessions = store.list_sessions()
        assert sessions[0]["turn_count"] == 2

    def test_count_zero_with_no_turns(self, store: SQLiteMemoryStore) -> None:
        store.register_session("sess-1")
        store.update_session_turn_count("sess-1")
        sessions = store.list_sessions()
        assert sessions[0]["turn_count"] == 0
