"""Coverage gap tests for missy.memory.sqlite_store.SQLiteMemoryStore.

Targets uncovered lines:
  55     : ConversationTurn.to_dict — metadata field included
  68     : ConversationTurn.from_dict — metadata field deserialized
  225-227: clear_session — turn deletion confirmed
  263-270: get_recent_turns — cross-session ordering
  291-308: search — FTS5 with and without session_id filter
  323-337: save_learning — persists learning object
  353-364: get_learnings — with and without task_type filter
  588-594: cleanup — deletes old turns, returns count
  602     : _row_to_turn — metadata parsing from None value
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from missy.memory.sqlite_store import ConversationTurn, SQLiteMemoryStore


@pytest.fixture
def store(tmp_path: Path) -> SQLiteMemoryStore:
    return SQLiteMemoryStore(db_path=str(tmp_path / "test_coverage.db"))


# ---------------------------------------------------------------------------
# ConversationTurn.to_dict — line 55: metadata field
# ---------------------------------------------------------------------------


class TestConversationTurnToDict:
    def test_metadata_included_in_dict(self):
        turn = ConversationTurn(
            id="abc",
            session_id="s1",
            timestamp="2026-01-01T00:00:00",
            role="user",
            content="hello",
            provider="anthropic",
            metadata={"key": "value", "num": 42},
        )
        d = turn.to_dict()
        assert "metadata" in d
        assert d["metadata"] == {"key": "value", "num": 42}

    def test_metadata_empty_dict_in_dict(self):
        turn = ConversationTurn(
            id="x",
            session_id="s",
            timestamp="t",
            role="user",
            content="c",
        )
        d = turn.to_dict()
        assert d["metadata"] == {}


# ---------------------------------------------------------------------------
# ConversationTurn.from_dict — line 68: metadata field
# ---------------------------------------------------------------------------


class TestConversationTurnFromDict:
    def test_metadata_deserialized(self):
        data = {
            "id": "abc",
            "session_id": "s1",
            "timestamp": "2026-01-01T00:00:00",
            "role": "user",
            "content": "hi",
            "provider": "",
            "metadata": {"foo": "bar"},
        }
        turn = ConversationTurn.from_dict(data)
        assert turn.metadata == {"foo": "bar"}

    def test_metadata_defaults_to_empty_dict_when_missing(self):
        turn = ConversationTurn.from_dict({"session_id": "s", "role": "user", "content": "x"})
        assert turn.metadata == {}


# ---------------------------------------------------------------------------
# clear_session — lines 225-227
# ---------------------------------------------------------------------------


class TestClearSession:
    def test_clear_session_removes_turns(self, store: SQLiteMemoryStore):
        turn = ConversationTurn.new("sess-clear", "user", "hi")
        store.add_turn(turn)
        assert len(store.get_session_turns("sess-clear")) == 1

        store.clear_session("sess-clear")

        assert store.get_session_turns("sess-clear") == []

    def test_clear_session_does_not_affect_other_sessions(self, store: SQLiteMemoryStore):
        store.add_turn(ConversationTurn.new("sess-a", "user", "keep"))
        store.add_turn(ConversationTurn.new("sess-b", "user", "delete"))
        store.clear_session("sess-b")

        remaining = store.get_session_turns("sess-a")
        assert len(remaining) == 1
        assert remaining[0].content == "keep"

    def test_clear_nonexistent_session_is_noop(self, store: SQLiteMemoryStore):
        # Should not raise
        store.clear_session("no-such-session")


# ---------------------------------------------------------------------------
# get_recent_turns — lines 263-270
# ---------------------------------------------------------------------------


class TestGetRecentTurns:
    def test_returns_turns_across_sessions(self, store: SQLiteMemoryStore):
        store.add_turn(ConversationTurn.new("sess-1", "user", "msg-a"))
        store.add_turn(ConversationTurn.new("sess-2", "user", "msg-b"))
        turns = store.get_recent_turns(limit=10)
        contents = {t.content for t in turns}
        assert "msg-a" in contents
        assert "msg-b" in contents

    def test_returns_at_most_limit_turns(self, store: SQLiteMemoryStore):
        for i in range(10):
            store.add_turn(ConversationTurn.new("sess-1", "user", f"msg-{i}"))
        turns = store.get_recent_turns(limit=3)
        assert len(turns) == 3

    def test_returns_empty_when_no_turns(self, store: SQLiteMemoryStore):
        assert store.get_recent_turns() == []

    def test_turns_returned_oldest_first(self, store: SQLiteMemoryStore):
        """Reversed after DESC fetch → chronological order."""
        for i in range(5):
            store.add_turn(ConversationTurn.new("sess-1", "user", f"msg-{i}"))
        turns = store.get_recent_turns(limit=5)
        # The first returned turn should have the earliest content
        assert turns[0].content == "msg-0"
        assert turns[-1].content == "msg-4"


# ---------------------------------------------------------------------------
# search — lines 291-308
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_finds_matching_content(self, store: SQLiteMemoryStore):
        store.add_turn(ConversationTurn.new("sess-1", "user", "python programming language"))
        store.add_turn(ConversationTurn.new("sess-1", "user", "javascript frameworks"))
        results = store.search("python")
        assert len(results) == 1
        assert "python" in results[0].content

    def test_search_returns_empty_for_no_match(self, store: SQLiteMemoryStore):
        store.add_turn(ConversationTurn.new("sess-1", "user", "hello world"))
        results = store.search("nonexistent_term_xyz")
        assert results == []

    def test_search_with_session_id_filter(self, store: SQLiteMemoryStore):
        store.add_turn(ConversationTurn.new("sess-1", "user", "async python code"))
        store.add_turn(ConversationTurn.new("sess-2", "user", "async javascript code"))

        # Only return matches from sess-1
        results = store.search("async", session_id="sess-1")
        assert len(results) == 1
        assert results[0].session_id == "sess-1"

    def test_search_without_session_id_returns_all_sessions(self, store: SQLiteMemoryStore):
        store.add_turn(ConversationTurn.new("sess-1", "user", "shared topic alpha"))
        store.add_turn(ConversationTurn.new("sess-2", "user", "shared topic beta"))

        results = store.search("shared")
        assert len(results) == 2

    def test_search_respects_limit(self, store: SQLiteMemoryStore):
        for i in range(10):
            store.add_turn(ConversationTurn.new("sess-1", "user", f"target content item {i}"))
        results = store.search("target", limit=3)
        assert len(results) <= 3

    def test_search_returns_conversation_turn_objects(self, store: SQLiteMemoryStore):
        store.add_turn(ConversationTurn.new("sess-1", "user", "findable term here"))
        results = store.search("findable")
        assert all(isinstance(r, ConversationTurn) for r in results)


# ---------------------------------------------------------------------------
# save_learning — lines 323-337
# ---------------------------------------------------------------------------


@dataclass
class _FakeLearning:
    task_type: str = "coding"
    outcome: str = "success"
    lesson: str = "Always test edge cases"
    approach: list = None
    timestamp: str = ""

    def __post_init__(self):
        if self.approach is None:
            self.approach = ["write tests", "review code"]
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


class TestSaveLearning:
    def test_save_learning_persists(self, store: SQLiteMemoryStore):
        learning = _FakeLearning(lesson="Test every edge case")
        store.save_learning(learning)

        lessons = store.get_learnings()
        assert "Test every edge case" in lessons

    def test_save_multiple_learnings(self, store: SQLiteMemoryStore):
        store.save_learning(_FakeLearning(lesson="Lesson One"))
        store.save_learning(_FakeLearning(lesson="Lesson Two"))
        lessons = store.get_learnings()
        assert len(lessons) == 2

    def test_save_learning_with_missing_attributes(self, store: SQLiteMemoryStore):
        """save_learning uses getattr with defaults for all fields."""

        class MinimalLearning:
            lesson = "Minimal lesson"

        store.save_learning(MinimalLearning())
        lessons = store.get_learnings()
        assert "Minimal lesson" in lessons


# ---------------------------------------------------------------------------
# get_learnings — lines 353-364
# ---------------------------------------------------------------------------


class TestGetLearnings:
    def test_get_learnings_no_filter(self, store: SQLiteMemoryStore):
        store.save_learning(_FakeLearning(task_type="coding", lesson="Code lesson"))
        store.save_learning(_FakeLearning(task_type="writing", lesson="Write lesson"))

        lessons = store.get_learnings()
        assert "Code lesson" in lessons
        assert "Write lesson" in lessons

    def test_get_learnings_with_task_type_filter(self, store: SQLiteMemoryStore):
        store.save_learning(_FakeLearning(task_type="coding", lesson="Code lesson"))
        store.save_learning(_FakeLearning(task_type="writing", lesson="Write lesson"))

        coding_lessons = store.get_learnings(task_type="coding")
        assert "Code lesson" in coding_lessons
        assert "Write lesson" not in coding_lessons

    def test_get_learnings_respects_limit(self, store: SQLiteMemoryStore):
        for i in range(10):
            store.save_learning(_FakeLearning(lesson=f"Lesson {i}"))

        lessons = store.get_learnings(limit=3)
        assert len(lessons) == 3

    def test_get_learnings_returns_strings(self, store: SQLiteMemoryStore):
        store.save_learning(_FakeLearning(lesson="A string lesson"))
        lessons = store.get_learnings()
        assert all(isinstance(ln, str) for ln in lessons)

    def test_get_learnings_empty_store(self, store: SQLiteMemoryStore):
        assert store.get_learnings() == []

    def test_get_learnings_most_recent_first(self, store: SQLiteMemoryStore):
        """Most recent lesson comes first (ORDER BY timestamp DESC)."""
        store.save_learning(_FakeLearning(lesson="Older lesson"))
        store.save_learning(_FakeLearning(lesson="Newer lesson"))

        # With limit=1, should get the most recent
        lessons = store.get_learnings(limit=1)
        assert lessons[0] == "Newer lesson"


# ---------------------------------------------------------------------------
# cleanup — lines 588-594
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_deletes_old_turns(self, store: SQLiteMemoryStore):
        # Insert a turn with an old timestamp directly
        conn = store._conn()
        old_time = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        conn.execute(
            "INSERT INTO turns (id, session_id, timestamp, role, content, provider, metadata) "
            "VALUES ('old-1', 'sess-cleanup', ?, 'user', 'old message', '', '{}')",
            (old_time,),
        )
        conn.commit()

        deleted = store.cleanup(older_than_days=30)
        assert deleted == 1
        assert store.get_session_turns("sess-cleanup") == []

    def test_cleanup_preserves_recent_turns(self, store: SQLiteMemoryStore):
        store.add_turn(ConversationTurn.new("sess-1", "user", "recent message"))
        deleted = store.cleanup(older_than_days=30)
        assert deleted == 0
        assert len(store.get_session_turns("sess-1")) == 1

    def test_cleanup_returns_zero_when_nothing_deleted(self, store: SQLiteMemoryStore):
        store.add_turn(ConversationTurn.new("sess-1", "user", "fresh"))
        result = store.cleanup(older_than_days=365)
        assert result == 0

    def test_cleanup_returns_correct_count(self, store: SQLiteMemoryStore):
        conn = store._conn()
        old_time = (datetime.now(UTC) - timedelta(days=90)).isoformat()
        for i in range(5):
            conn.execute(
                "INSERT INTO turns (id, session_id, timestamp, role, content, provider, metadata) "
                "VALUES (?, 'sess-old', ?, 'user', 'old', '', '{}')",
                (f"old-{i}", old_time),
            )
        conn.commit()

        deleted = store.cleanup(older_than_days=30)
        assert deleted == 5


# ---------------------------------------------------------------------------
# _row_to_turn — line 602: metadata=None handled
# ---------------------------------------------------------------------------


class TestRowToTurn:
    def test_row_to_turn_with_null_metadata(self, store: SQLiteMemoryStore):
        """Rows inserted without metadata (NULL) should produce empty dict."""
        conn = store._conn()
        # Insert a row with NULL metadata to test the `or '{}'` fallback
        conn.execute(
            "INSERT INTO turns (id, session_id, timestamp, role, content, provider, metadata) "
            "VALUES ('null-meta', 'sess-null', '2026-01-01T00:00:00', 'user', 'test', '', NULL)"
        )
        conn.commit()

        turns = store.get_session_turns("sess-null")
        assert len(turns) == 1
        assert turns[0].metadata == {}

    def test_row_to_turn_with_valid_metadata(self, store: SQLiteMemoryStore):
        """Metadata JSON is correctly deserialized."""
        turn = ConversationTurn(
            id="meta-test",
            session_id="sess-meta",
            timestamp=datetime.now(UTC).isoformat(),
            role="user",
            content="test",
            provider="",
            metadata={"tool": "shell_exec", "duration_ms": 42},
        )
        store.add_turn(turn)

        retrieved = store.get_session_turns("sess-meta")
        assert retrieved[0].metadata == {"tool": "shell_exec", "duration_ms": 42}
