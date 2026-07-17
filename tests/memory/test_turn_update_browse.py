"""Tests for SQLiteMemoryStore.update_turn_content and browse_turns.

update_turn_content backs the Web TUI memory browser's edit action; the
FTS-reindex assertions are the load-bearing ones — ``turns_fts`` is an
external-content FTS5 table, so a plain UPDATE without the ``turns_au``
trigger silently desyncs the search index.
"""

from __future__ import annotations

import pytest

from missy.memory.sqlite_store import ConversationTurn, SQLiteMemoryStore


@pytest.fixture
def store(tmp_path):
    return SQLiteMemoryStore(str(tmp_path / "memory.db"))


class TestUpdateTurnContent:
    def test_update_existing_turn_returns_true_and_persists(self, store):
        turn = ConversationTurn.new("sess1", "user", "original text")
        store.add_turn(turn)
        assert store.update_turn_content(turn.id, "revised text") is True
        [reloaded] = store.get_session_turns("sess1")
        assert reloaded.content == "revised text"

    def test_update_unknown_turn_returns_false(self, store):
        assert store.update_turn_content("does-not-exist", "text") is False

    def test_update_records_edited_at_metadata(self, store):
        turn = ConversationTurn.new("sess1", "user", "original")
        store.add_turn(turn)
        store.update_turn_content(turn.id, "revised")
        [reloaded] = store.get_session_turns("sess1")
        assert reloaded.metadata.get("edited_at")

    def test_update_preserves_other_metadata(self, store):
        turn = ConversationTurn.new("sess1", "user", "original")
        turn.metadata = {"other": "value"}
        store.add_turn(turn)
        store.set_turn_pinned(turn.id, True)
        store.update_turn_content(turn.id, "revised")
        [reloaded] = store.get_session_turns("sess1")
        assert reloaded.metadata.get("other") == "value"
        assert reloaded.metadata.get("pinned") is True

    def test_update_reindexes_fts_old_content_unfindable(self, store):
        turn = ConversationTurn.new("sess1", "user", "unique-original-marker")
        store.add_turn(turn)
        assert store.search("unique-original-marker")
        store.update_turn_content(turn.id, "unique-replacement-marker")
        assert store.search("unique-original-marker") == []
        results = store.search("unique-replacement-marker")
        assert [t.id for t in results] == [turn.id]

    def test_pin_after_edit_keeps_fts_consistent(self, store):
        """set_turn_pinned also UPDATEs the row; the turns_au trigger must
        not corrupt the index when content is unchanged."""
        turn = ConversationTurn.new("sess1", "user", "stable-content-marker")
        store.add_turn(turn)
        store.set_turn_pinned(turn.id, True)
        assert [t.id for t in store.search("stable-content-marker")] == [turn.id]


class TestBrowseTurns:
    def _seed(self, store, count: int, session_id: str = "sess1") -> list[ConversationTurn]:
        turns = []
        for index in range(count):
            turn = ConversationTurn.new(session_id, "user", f"message {index}")
            # Deterministic ordering: timestamps strictly increasing.
            turn.timestamp = f"2026-07-17T00:00:{index:02d}+00:00"
            store.add_turn(turn)
            turns.append(turn)
        return turns

    def test_browse_empty_store(self, store):
        turns, total = store.browse_turns()
        assert turns == []
        assert total == 0

    def test_browse_returns_newest_first_with_total(self, store):
        seeded = self._seed(store, 5)
        turns, total = store.browse_turns(limit=2, offset=0)
        assert total == 5
        assert [t.id for t in turns] == [seeded[4].id, seeded[3].id]

    def test_browse_offset_pages_through(self, store):
        seeded = self._seed(store, 5)
        turns, total = store.browse_turns(limit=2, offset=4)
        assert total == 5
        assert [t.id for t in turns] == [seeded[0].id]

    def test_browse_filters_by_session(self, store):
        self._seed(store, 3, session_id="sess1")
        other = self._seed(store, 2, session_id="sess2")
        turns, total = store.browse_turns(session_id="sess2")
        assert total == 2
        assert {t.id for t in turns} == {t.id for t in other}
