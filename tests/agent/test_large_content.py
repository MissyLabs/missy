"""Tests for large content interception (Feature 5)."""

from __future__ import annotations

import os
import tempfile

import pytest

from missy.memory.sqlite_store import (
    ConversationTurn,
    LargeContentRecord,
    SQLiteMemoryStore,
    SummaryRecord,
)


@pytest.fixture
def memory_store():
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    return SQLiteMemoryStore(db)


class TestLargeContentRecord:
    def test_new(self):
        r = LargeContentRecord.new("sess1", "shell_exec", "x" * 100)
        assert r.id.startswith("ref_")
        assert r.original_chars == 100
        assert r.session_id == "sess1"
        assert r.tool_name == "shell_exec"

    def test_to_dict(self):
        r = LargeContentRecord.new("sess1", "tool", "content")
        d = r.to_dict()
        assert d["id"] == r.id
        assert d["tool_name"] == "tool"
        assert d["content"] == "content"


class TestLargeContentStore:
    def test_store_and_retrieve(self, memory_store):
        r = LargeContentRecord.new("sess1", "shell_exec", "x" * 100000, summary="big output")
        cid = memory_store.store_large_content(r)
        assert cid == r.id

        got = memory_store.get_large_content(cid)
        assert got is not None
        assert got.original_chars == 100000
        assert got.summary == "big output"
        assert len(got.content) == 100000

    def test_get_missing(self, memory_store):
        assert memory_store.get_large_content("ref_nonexistent") is None

    def test_search_large_content(self, memory_store):
        r = LargeContentRecord.new("sess1", "tool", "hello world content", summary="test output")
        memory_store.store_large_content(r)

        results = memory_store.search_large_content("hello", "sess1")
        assert len(results) == 1
        assert results[0].id == r.id

    def test_search_no_match(self, memory_store):
        r = LargeContentRecord.new("sess1", "tool", "something")
        memory_store.store_large_content(r)
        results = memory_store.search_large_content("zzzzz", "sess1")
        assert len(results) == 0

    def test_multiple_records(self, memory_store):
        for i in range(5):
            r = LargeContentRecord.new("sess1", f"tool_{i}", f"content_{i}")
            memory_store.store_large_content(r)
        results = memory_store.search_large_content("content", "sess1")
        assert len(results) == 5


class TestSummaryStore:
    """Test summary CRUD alongside large content (Feature 2A)."""

    def test_add_and_get_summary(self, memory_store):
        s = SummaryRecord.new("sess1", depth=0, content="Test summary")
        memory_store.add_summary(s)
        got = memory_store.get_summary_by_id(s.id)
        assert got is not None
        assert got.content == "Test summary"
        assert got.depth == 0

    def test_get_uncompacted(self, memory_store):
        s1 = SummaryRecord.new("sess1", depth=0, content="A")
        s2 = SummaryRecord.new("sess1", depth=0, content="B")
        memory_store.add_summary(s1)
        memory_store.add_summary(s2)

        unc = memory_store.get_uncompacted_summaries("sess1", 0)
        assert len(unc) == 2

    def test_mark_compacted(self, memory_store):
        s1 = SummaryRecord.new("sess1", depth=0, content="A")
        s2 = SummaryRecord.new("sess1", depth=0, content="B")
        memory_store.add_summary(s1)
        memory_store.add_summary(s2)

        parent = SummaryRecord.new("sess1", depth=1, content="AB combined")
        memory_store.add_summary(parent)
        memory_store.mark_summary_compacted([s1.id, s2.id], parent.id)

        unc = memory_store.get_uncompacted_summaries("sess1", 0)
        assert len(unc) == 0

    def test_get_child_summaries(self, memory_store):
        child = SummaryRecord.new("sess1", depth=0, content="child")
        parent = SummaryRecord.new("sess1", depth=1, content="parent")
        memory_store.add_summary(child)
        memory_store.add_summary(parent)
        memory_store.mark_summary_compacted([child.id], parent.id)

        children = memory_store.get_child_summaries(parent.id)
        assert len(children) == 1
        assert children[0].id == child.id

    def test_search_summaries(self, memory_store):
        s = SummaryRecord.new("sess1", depth=0, content="kubernetes deployment failed")
        memory_store.add_summary(s)

        results = memory_store.search_summaries("kubernetes", session_id="sess1")
        assert len(results) == 1
        assert results[0].id == s.id

    def test_get_session_token_count(self, memory_store):
        t = ConversationTurn.new("sess1", "user", "x" * 400)
        memory_store.add_turn(t)

        s = SummaryRecord.new("sess1", depth=0, content="y" * 200)
        memory_store.add_summary(s)

        count = memory_store.get_session_token_count("sess1")
        assert count > 0

    def test_summary_record_serialization(self):
        s = SummaryRecord.new(
            "sess1",
            depth=1,
            content="test",
            source_turn_ids=["t1", "t2"],
            source_summary_ids=["s1"],
            time_range_start="2026-01-01",
            time_range_end="2026-01-02",
        )
        d = s.to_dict()
        restored = SummaryRecord.from_dict(d)
        assert restored.id == s.id
        assert restored.source_turn_ids == ["t1", "t2"]
        assert restored.source_summary_ids == ["s1"]
        assert restored.depth == 1
