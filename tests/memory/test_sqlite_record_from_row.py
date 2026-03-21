"""Tests for SummaryRecord.from_row, LargeContentRecord.from_row, and
SQLiteMemoryStore.get_source_turns."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from missy.memory.sqlite_store import (
    ConversationTurn,
    LargeContentRecord,
    SQLiteMemoryStore,
    SummaryRecord,
)

# ---------------------------------------------------------------------------
# Helper: sqlite3.Row-compatible fake
# ---------------------------------------------------------------------------


class FakeRow:
    """Dict-backed stand-in for sqlite3.Row with key-based access."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getitem__(self, key: str):  # noqa: ANN001
        return self._data[key]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> SQLiteMemoryStore:
    return SQLiteMemoryStore(db_path=str(tmp_path / "test_from_row.db"))


# ---------------------------------------------------------------------------
# SummaryRecord.from_row
# ---------------------------------------------------------------------------


class TestSummaryRecordFromRow:
    def test_all_fields_populated(self):
        source_turns = ["turn-1", "turn-2"]
        source_summaries = ["sum-a"]
        file_refs = ["file.txt"]
        row = FakeRow(
            {
                "id": "sum_abc123",
                "session_id": "sess-1",
                "depth": 2,
                "content": "A summary of events.",
                "token_estimate": 42,
                "source_turn_ids": json.dumps(source_turns),
                "source_summary_ids": json.dumps(source_summaries),
                "parent_id": "sum_parent",
                "time_range_start": "2026-01-01T00:00:00+00:00",
                "time_range_end": "2026-01-02T00:00:00+00:00",
                "descendant_count": 5,
                "file_refs": json.dumps(file_refs),
                "created_at": "2026-01-01T12:00:00+00:00",
            }
        )
        record = SummaryRecord.from_row(row)

        assert record.id == "sum_abc123"
        assert record.session_id == "sess-1"
        assert record.depth == 2
        assert record.content == "A summary of events."
        assert record.token_estimate == 42
        assert record.source_turn_ids == source_turns
        assert record.source_summary_ids == source_summaries
        assert record.parent_id == "sum_parent"
        assert record.time_range_start == "2026-01-01T00:00:00+00:00"
        assert record.time_range_end == "2026-01-02T00:00:00+00:00"
        assert record.descendant_count == 5
        assert record.file_refs == file_refs
        assert record.created_at == "2026-01-01T12:00:00+00:00"

    def test_null_json_fields_become_empty_lists(self):
        """None values for JSON list columns fall back to empty lists."""
        row = FakeRow(
            {
                "id": "sum_null_json",
                "session_id": "sess-2",
                "depth": 0,
                "content": "Sparse summary.",
                "token_estimate": 10,
                "source_turn_ids": None,
                "source_summary_ids": None,
                "parent_id": None,
                "time_range_start": None,
                "time_range_end": None,
                "descendant_count": 0,
                "file_refs": None,
                "created_at": "2026-01-01T12:00:00+00:00",
            }
        )
        record = SummaryRecord.from_row(row)

        assert record.source_turn_ids == []
        assert record.source_summary_ids == []
        assert record.file_refs == []
        assert record.parent_id is None

    def test_returns_summary_record_instance(self):
        row = FakeRow(
            {
                "id": "sum_type_check",
                "session_id": "sess-3",
                "depth": 1,
                "content": "Type check.",
                "token_estimate": 5,
                "source_turn_ids": "[]",
                "source_summary_ids": "[]",
                "parent_id": None,
                "time_range_start": None,
                "time_range_end": None,
                "descendant_count": 0,
                "file_refs": "[]",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
        assert isinstance(SummaryRecord.from_row(row), SummaryRecord)


# ---------------------------------------------------------------------------
# LargeContentRecord.from_row
# ---------------------------------------------------------------------------


class TestLargeContentRecordFromRow:
    def test_all_fields_populated(self):
        row = FakeRow(
            {
                "id": "ref_abcdef01",
                "session_id": "sess-10",
                "turn_id": "turn-99",
                "tool_name": "bash",
                "original_chars": 9000,
                "content": "x" * 9000,
                "summary": "A big bash result.",
                "created_at": "2026-03-01T08:00:00+00:00",
            }
        )
        record = LargeContentRecord.from_row(row)

        assert record.id == "ref_abcdef01"
        assert record.session_id == "sess-10"
        assert record.turn_id == "turn-99"
        assert record.tool_name == "bash"
        assert record.original_chars == 9000
        assert len(record.content) == 9000
        assert record.summary == "A big bash result."
        assert record.created_at == "2026-03-01T08:00:00+00:00"

    def test_summary_none_becomes_empty_string(self):
        """A NULL summary column must be coerced to an empty string."""
        row = FakeRow(
            {
                "id": "ref_no_summary",
                "session_id": "sess-11",
                "turn_id": "turn-1",
                "tool_name": "read_file",
                "original_chars": 500,
                "content": "file content",
                "summary": None,
                "created_at": "2026-03-01T09:00:00+00:00",
            }
        )
        record = LargeContentRecord.from_row(row)
        assert record.summary == ""

    def test_turn_id_none_is_preserved(self):
        """turn_id is nullable and None must round-trip correctly."""
        row = FakeRow(
            {
                "id": "ref_no_turn",
                "session_id": "sess-12",
                "turn_id": None,
                "tool_name": "web_search",
                "original_chars": 200,
                "content": "search results",
                "summary": "Search summary.",
                "created_at": "2026-03-01T10:00:00+00:00",
            }
        )
        record = LargeContentRecord.from_row(row)
        assert record.turn_id is None

    def test_returns_large_content_record_instance(self):
        row = FakeRow(
            {
                "id": "ref_type",
                "session_id": "sess-13",
                "turn_id": "t1",
                "tool_name": "tool",
                "original_chars": 1,
                "content": "x",
                "summary": "",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
        assert isinstance(LargeContentRecord.from_row(row), LargeContentRecord)


# ---------------------------------------------------------------------------
# SQLiteMemoryStore.get_source_turns
# ---------------------------------------------------------------------------


class TestGetSourceTurns:
    def test_returns_empty_list_for_nonexistent_summary(self, store: SQLiteMemoryStore):
        result = store.get_source_turns("summary-does-not-exist")
        assert result == []

    def test_returns_empty_list_for_summary_with_no_source_turns(self, store: SQLiteMemoryStore):
        summary = SummaryRecord.new(
            session_id="sess-empty",
            depth=0,
            content="Summary with no source turns.",
            source_turn_ids=[],
        )
        store.add_summary(summary)
        result = store.get_source_turns(summary.id)
        assert result == []

    def test_returns_turns_matching_source_turn_ids(self, store: SQLiteMemoryStore):
        session_id = "sess-with-turns"
        t1 = ConversationTurn.new(session_id, "user", "Hello")
        t2 = ConversationTurn.new(session_id, "assistant", "Hi there")
        t3 = ConversationTurn.new(session_id, "user", "Goodbye")
        for turn in (t1, t2, t3):
            store.add_turn(turn)

        # Summary references only t1 and t2
        summary = SummaryRecord.new(
            session_id=session_id,
            depth=0,
            content="Compressed first exchange.",
            source_turn_ids=[t1.id, t2.id],
        )
        store.add_summary(summary)

        result = store.get_source_turns(summary.id)
        result_ids = {t.id for t in result}

        assert result_ids == {t1.id, t2.id}
        assert t3.id not in result_ids

    def test_returned_objects_are_conversation_turns(self, store: SQLiteMemoryStore):
        session_id = "sess-type-check"
        turn = ConversationTurn.new(session_id, "user", "Type check turn.")
        store.add_turn(turn)

        summary = SummaryRecord.new(
            session_id=session_id,
            depth=0,
            content="Single-turn summary.",
            source_turn_ids=[turn.id],
        )
        store.add_summary(summary)

        result = store.get_source_turns(summary.id)
        assert len(result) == 1
        assert isinstance(result[0], ConversationTurn)
        assert result[0].content == "Type check turn."

    def test_results_ordered_by_timestamp(self, store: SQLiteMemoryStore):
        """Turns are returned in chronological order (ORDER BY timestamp)."""
        session_id = "sess-order"
        turns = [ConversationTurn.new(session_id, "user", f"Message {i}") for i in range(4)]
        for t in turns:
            store.add_turn(t)

        summary = SummaryRecord.new(
            session_id=session_id,
            depth=0,
            content="Multi-turn summary.",
            source_turn_ids=[t.id for t in turns],
        )
        store.add_summary(summary)

        result = store.get_source_turns(summary.id)
        timestamps = [r.timestamp for r in result]
        assert timestamps == sorted(timestamps)
