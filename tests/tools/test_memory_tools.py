"""Tests for missy.tools.builtin.memory_tools — agent retrieval tools."""

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
from missy.tools.builtin.memory_tools import (
    MemoryDescribeTool,
    MemoryExpandTool,
    MemorySearchTool,
)


@pytest.fixture
def memory_store():
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    return SQLiteMemoryStore(db)


@pytest.fixture
def populated_store(memory_store):
    """Store with turns, summaries, and large content."""
    for i in range(5):
        t = ConversationTurn.new("sess1", "user", f"Message about kubernetes deployment {i}")
        memory_store.add_turn(t)

    s = SummaryRecord.new(
        "sess1", depth=0, content="Discussion about kubernetes deployment issues",
        source_turn_ids=["t1", "t2"],
        time_range_start="2026-01-01T00:00:00",
        time_range_end="2026-01-01T01:00:00",
        descendant_count=5,
    )
    memory_store.add_summary(s)

    lc = LargeContentRecord.new(
        "sess1", "shell_exec", "x" * 100000,
        summary="Large kubectl output",
    )
    memory_store.store_large_content(lc)

    return memory_store, s.id, lc.id


def _text(result):
    """Extract readable text from a ToolResult."""
    return result.output if result.success else (result.error or "")


class TestMemorySearchTool:
    def test_search_messages(self, populated_store):
        store, _, _ = populated_store
        tool = MemorySearchTool()
        result = tool.execute(
            query="kubernetes", scope="messages",
            _memory_store=store, _session_id="sess1",
        )
        assert result.success
        assert "Messages" in result.output
        assert "kubernetes" in result.output.lower()

    def test_search_summaries(self, populated_store):
        store, _, _ = populated_store
        tool = MemorySearchTool()
        result = tool.execute(
            query="kubernetes", scope="summaries", _memory_store=store,
        )
        assert result.success
        assert "Summaries" in result.output

    def test_search_both(self, populated_store):
        store, _, _ = populated_store
        tool = MemorySearchTool()
        result = tool.execute(query="kubernetes", scope="both", _memory_store=store)
        assert result.success

    def test_no_results(self, populated_store):
        store, _, _ = populated_store
        tool = MemorySearchTool()
        result = tool.execute(query="zzznonexistent", scope="both", _memory_store=store)
        assert result.success
        assert "No results" in result.output

    def test_missing_query(self, memory_store):
        tool = MemorySearchTool()
        result = tool.execute(query="", _memory_store=memory_store)
        assert not result.success

    def test_no_store(self):
        tool = MemorySearchTool()
        result = tool.execute(query="test")
        assert not result.success
        assert "not available" in result.error

    def test_session_filter(self, populated_store):
        store, _, _ = populated_store
        tool = MemorySearchTool()
        result = tool.execute(
            query="kubernetes", session_id="nonexistent_session", _memory_store=store,
        )
        assert "No results" in result.output


class TestMemoryDescribeTool:
    def test_describe_summary(self, populated_store):
        store, sum_id, _ = populated_store
        tool = MemoryDescribeTool()
        result = tool.execute(item_id=sum_id, _memory_store=store)
        assert result.success
        assert "Summary:" in result.output
        assert "Depth:" in result.output
        assert "kubernetes" in result.output

    def test_describe_large_content(self, populated_store):
        store, _, ref_id = populated_store
        tool = MemoryDescribeTool()
        result = tool.execute(item_id=ref_id, _memory_store=store)
        assert result.success
        assert "Large Content:" in result.output
        assert "shell_exec" in result.output

    def test_invalid_id(self, memory_store):
        tool = MemoryDescribeTool()
        result = tool.execute(item_id="bad_id", _memory_store=memory_store)
        assert not result.success
        assert "Unknown ID format" in result.error

    def test_missing_summary(self, memory_store):
        tool = MemoryDescribeTool()
        result = tool.execute(item_id="sum_nonexistent", _memory_store=memory_store)
        assert not result.success
        assert "not found" in result.error

    def test_missing_ref(self, memory_store):
        tool = MemoryDescribeTool()
        result = tool.execute(item_id="ref_nonexistent", _memory_store=memory_store)
        assert not result.success
        assert "not found" in result.error


class TestMemoryExpandTool:
    def test_expand_large_content(self, populated_store):
        store, _, ref_id = populated_store
        tool = MemoryExpandTool()
        result = tool.execute(item_id=ref_id, max_tokens=1000, _memory_store=store)
        assert result.success
        assert "Expanded:" in result.output

    def test_expand_large_content_truncated(self, populated_store):
        store, _, ref_id = populated_store
        tool = MemoryExpandTool()
        result = tool.execute(item_id=ref_id, max_tokens=100, _memory_store=store)
        assert "TRUNCATED" in result.output

    def test_expand_summary(self, populated_store):
        store, sum_id, _ = populated_store
        tool = MemoryExpandTool()
        result = tool.execute(item_id=sum_id, _memory_store=store)
        assert result.success
        assert "Expanded:" in result.output

    def test_expand_missing(self, memory_store):
        tool = MemoryExpandTool()
        result = tool.execute(item_id="sum_nonexistent", _memory_store=memory_store)
        assert not result.success
        assert "not found" in result.error

    def test_expand_no_store(self):
        tool = MemoryExpandTool()
        result = tool.execute(item_id="sum_foo")
        assert not result.success

    def test_max_tokens_capped(self, populated_store):
        store, _, ref_id = populated_store
        tool = MemoryExpandTool()
        result = tool.execute(item_id=ref_id, max_tokens=999999, _memory_store=store)
        assert result.success

    def test_empty_db(self, memory_store):
        tool = MemoryExpandTool()
        result = tool.execute(item_id="sum_anything", _memory_store=memory_store)
        assert not result.success
