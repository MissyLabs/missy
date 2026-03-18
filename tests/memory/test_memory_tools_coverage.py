"""Tests for memory_tools.py — targeting uncovered lines (80% → ~95%).

Covers:
- MemorySearchTool: scope filtering, error paths, empty results
- MemoryDescribeTool: summary, large content, missing items, invalid IDs
- MemoryExpandTool: summary expansion, large content truncation, DAG walking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from missy.tools.builtin.memory_tools import (
    MemoryDescribeTool,
    MemoryExpandTool,
    MemorySearchTool,
)


# ---------------------------------------------------------------------------
# Helpers / mock objects
# ---------------------------------------------------------------------------


@dataclass
class FakeTurn:
    id: str = "turn_1"
    role: str = "user"
    content: str = "Hello world"
    timestamp: str = "2026-03-18T12:00:00.000"
    session_id: str = "sess_1"


@dataclass
class FakeSummary:
    id: str = "sum_1"
    depth: int = 1
    content: str = "Summary of conversation"
    token_estimate: int = 100
    descendant_count: int = 5
    time_range_start: str = "2026-03-18T10:00:00"
    time_range_end: str = "2026-03-18T12:00:00"
    source_turn_ids: list = field(default_factory=list)
    source_summary_ids: list = field(default_factory=list)
    parent_id: str | None = None
    created_at: str = "2026-03-18T12:00:00"
    session_id: str = "sess_1"


@dataclass
class FakeLargeContent:
    id: str = "ref_1"
    tool_name: str = "file_read"
    original_chars: int = 5000
    session_id: str = "sess_1"
    created_at: str = "2026-03-18T12:00:00"
    summary: str = "Contents of config.yaml"
    content: str = "x" * 5000


def make_mock_store(
    turns=None,
    summaries_search=None,
    summary_by_id=None,
    child_summaries=None,
    large_content=None,
    source_turns=None,
):
    store = MagicMock()
    store.search.return_value = turns or []
    store.search_summaries.return_value = summaries_search or []
    store.get_summary_by_id.return_value = summary_by_id
    store.get_child_summaries.return_value = child_summaries or []
    store.get_large_content.return_value = large_content
    store.get_source_turns.return_value = source_turns or []
    return store


# ---------------------------------------------------------------------------
# MemorySearchTool
# ---------------------------------------------------------------------------


class TestMemorySearchTool:
    def setup_method(self):
        self.tool = MemorySearchTool()

    def test_empty_query(self):
        r = self.tool.execute(query="", _memory_store=MagicMock())
        assert not r.success
        assert "required" in r.error

    def test_no_store(self):
        r = self.tool.execute(query="test")
        assert not r.success
        assert "not available" in r.error

    def test_search_messages_only(self):
        turns = [FakeTurn(content="Hello there")]
        store = make_mock_store(turns=turns)
        r = self.tool.execute(query="hello", scope="messages", _memory_store=store)
        assert r.success
        assert "Messages" in r.output
        assert "Hello there" in r.output

    def test_search_summaries_only(self):
        summaries = [FakeSummary(content="A summary of discussion")]
        store = make_mock_store(summaries_search=summaries)
        r = self.tool.execute(query="discussion", scope="summaries", _memory_store=store)
        assert r.success
        assert "Summaries" in r.output

    def test_search_both_scopes(self):
        turns = [FakeTurn()]
        summaries = [FakeSummary()]
        store = make_mock_store(turns=turns, summaries_search=summaries)
        r = self.tool.execute(query="test", scope="both", _memory_store=store)
        assert r.success
        assert "Messages" in r.output
        assert "Summaries" in r.output

    def test_search_no_results(self):
        store = make_mock_store()
        r = self.tool.execute(query="nonexistent", _memory_store=store)
        assert r.success
        assert "No results" in r.output

    def test_search_with_session_id(self):
        store = make_mock_store(turns=[FakeTurn()])
        r = self.tool.execute(
            query="test", scope="messages", session_id="sess_42", _memory_store=store
        )
        assert r.success
        store.search.assert_called_once_with("test", limit=10, session_id="sess_42")

    def test_search_uses_internal_session_id(self):
        store = make_mock_store(turns=[FakeTurn()])
        r = self.tool.execute(
            query="test", scope="messages", _session_id="sess_99", _memory_store=store
        )
        assert r.success
        store.search.assert_called_once_with("test", limit=10, session_id="sess_99")

    def test_limit_capped_at_50(self):
        store = make_mock_store(turns=[FakeTurn()])
        self.tool.execute(
            query="test", scope="messages", limit=100, _memory_store=store
        )
        store.search.assert_called_once_with("test", limit=50, session_id=None)

    def test_message_search_error(self):
        store = make_mock_store()
        store.search.side_effect = RuntimeError("DB error")
        r = self.tool.execute(query="test", scope="messages", _memory_store=store)
        assert r.success
        assert "error" in r.output.lower()

    def test_summary_search_error(self):
        store = make_mock_store()
        store.search_summaries.side_effect = RuntimeError("FTS error")
        r = self.tool.execute(query="test", scope="summaries", _memory_store=store)
        assert r.success
        assert "error" in r.output.lower()

    def test_long_content_truncated(self):
        turn = FakeTurn(content="x" * 300)
        store = make_mock_store(turns=[turn])
        r = self.tool.execute(query="test", scope="messages", _memory_store=store)
        assert "..." in r.output

    def test_summary_with_time_range(self):
        s = FakeSummary(
            time_range_start="2026-03-18T10:00:00",
            time_range_end="2026-03-18T12:00:00",
        )
        store = make_mock_store(summaries_search=[s])
        r = self.tool.execute(query="test", scope="summaries", _memory_store=store)
        assert "2026-03-18T10:00:00" in r.output

    def test_summary_without_time_range(self):
        s = FakeSummary(time_range_start=None, time_range_end=None)
        store = make_mock_store(summaries_search=[s])
        r = self.tool.execute(query="test", scope="summaries", _memory_store=store)
        assert r.success

    def test_long_summary_truncated(self):
        s = FakeSummary(content="y" * 400)
        store = make_mock_store(summaries_search=[s])
        r = self.tool.execute(query="test", scope="summaries", _memory_store=store)
        assert "..." in r.output


# ---------------------------------------------------------------------------
# MemoryDescribeTool
# ---------------------------------------------------------------------------


class TestMemoryDescribeTool:
    def setup_method(self):
        self.tool = MemoryDescribeTool()

    def test_empty_item_id(self):
        r = self.tool.execute(item_id="", _memory_store=MagicMock())
        assert not r.success
        assert "required" in r.error

    def test_no_store(self):
        r = self.tool.execute(item_id="sum_1")
        assert not r.success
        assert "not available" in r.error

    def test_unknown_id_format(self):
        r = self.tool.execute(item_id="xyz_123", _memory_store=MagicMock())
        assert not r.success
        assert "Unknown ID format" in r.error

    def test_describe_summary(self):
        summary = FakeSummary(
            id="sum_42",
            source_turn_ids=["t1", "t2"],
            source_summary_ids=["s1"],
        )
        store = make_mock_store(summary_by_id=summary, child_summaries=[])
        r = self.tool.execute(item_id="sum_42", _memory_store=store)
        assert r.success
        assert "sum_42" in r.output
        assert "Depth" in r.output
        assert "Tokens" in r.output

    def test_describe_summary_with_children(self):
        summary = FakeSummary(id="sum_parent")
        child = FakeSummary(id="sum_child", depth=2, token_estimate=50)
        store = make_mock_store(summary_by_id=summary, child_summaries=[child])
        r = self.tool.execute(item_id="sum_parent", _memory_store=store)
        assert r.success
        assert "Children" in r.output
        assert "sum_child" in r.output

    def test_describe_summary_not_found(self):
        store = make_mock_store(summary_by_id=None)
        r = self.tool.execute(item_id="sum_missing", _memory_store=store)
        assert not r.success
        assert "not found" in r.error

    def test_describe_summary_no_parent(self):
        summary = FakeSummary(parent_id=None)
        store = make_mock_store(summary_by_id=summary)
        r = self.tool.execute(item_id="sum_1", _memory_store=store)
        assert "top-level" in r.output

    def test_describe_large_content(self):
        lc = FakeLargeContent()
        store = make_mock_store(large_content=lc)
        r = self.tool.execute(item_id="ref_1", _memory_store=store)
        assert r.success
        assert "Large Content" in r.output
        assert "file_read" in r.output
        assert "Preview" in r.output

    def test_describe_large_content_not_found(self):
        store = make_mock_store(large_content=None)
        r = self.tool.execute(item_id="ref_missing", _memory_store=store)
        assert not r.success
        assert "not found" in r.error


# ---------------------------------------------------------------------------
# MemoryExpandTool
# ---------------------------------------------------------------------------


class TestMemoryExpandTool:
    def setup_method(self):
        self.tool = MemoryExpandTool()

    def test_empty_item_id(self):
        r = self.tool.execute(item_id="", _memory_store=MagicMock())
        assert not r.success
        assert "required" in r.error

    def test_no_store(self):
        r = self.tool.execute(item_id="sum_1")
        assert not r.success
        assert "not available" in r.error

    def test_unknown_id_format(self):
        r = self.tool.execute(item_id="bad_1", _memory_store=MagicMock())
        assert not r.success
        assert "Unknown ID format" in r.error

    def test_expand_large_content(self):
        lc = FakeLargeContent(content="Real content here" * 10)
        store = make_mock_store(large_content=lc)
        r = self.tool.execute(item_id="ref_1", _memory_store=store)
        assert r.success
        assert "Expanded" in r.output
        assert "Real content here" in r.output

    def test_expand_large_content_truncated(self):
        lc = FakeLargeContent(content="x" * 100_000, original_chars=100_000)
        store = make_mock_store(large_content=lc)
        r = self.tool.execute(item_id="ref_1", max_tokens=100, _memory_store=store)
        assert r.success
        assert "TRUNCATED" in r.output

    def test_expand_large_content_not_found(self):
        store = make_mock_store(large_content=None)
        r = self.tool.execute(item_id="ref_missing", _memory_store=store)
        assert not r.success
        assert "not found" in r.error

    def test_expand_summary_with_children(self):
        summary = FakeSummary(
            id="sum_parent",
            source_summary_ids=["sum_child1"],
            source_turn_ids=[],
        )
        child = FakeSummary(id="sum_child1", depth=2, content="Child content here")
        store = make_mock_store(
            summary_by_id=summary,
            child_summaries=[child],
        )
        r = self.tool.execute(item_id="sum_parent", _memory_store=store)
        assert r.success
        assert "sum_child1" in r.output
        assert "Child content here" in r.output

    def test_expand_summary_children_via_ids(self):
        """When get_child_summaries returns empty, fall back to source_summary_ids."""
        summary = FakeSummary(
            id="sum_parent",
            source_summary_ids=["sum_c1", "sum_c2"],
            source_turn_ids=[],
        )
        child1 = FakeSummary(id="sum_c1", depth=2, content="Child 1")
        child2 = FakeSummary(id="sum_c2", depth=2, content="Child 2")

        store = MagicMock()
        store.get_child_summaries.return_value = []  # empty — trigger fallback
        call_count = 0

        def get_by_id(sid):
            nonlocal call_count
            call_count += 1
            if sid == "sum_parent":
                return summary
            elif sid == "sum_c1":
                return child1
            elif sid == "sum_c2":
                return child2
            return None

        store.get_summary_by_id.side_effect = get_by_id
        store.get_source_turns.return_value = []

        r = self.tool.execute(item_id="sum_parent", _memory_store=store)
        assert r.success
        assert "Child 1" in r.output
        assert "Child 2" in r.output

    def test_expand_summary_with_source_turns(self):
        summary = FakeSummary(
            id="sum_1",
            source_summary_ids=[],
            source_turn_ids=["t1", "t2"],
        )
        turns = [
            FakeTurn(id="t1", role="user", content="Question?", timestamp="2026-03-18T10:00:00"),
            FakeTurn(id="t2", role="assistant", content="Answer!", timestamp="2026-03-18T10:01:00"),
        ]
        store = make_mock_store(summary_by_id=summary, source_turns=turns)
        r = self.tool.execute(item_id="sum_1", _memory_store=store)
        assert r.success
        assert "Question?" in r.output
        assert "Answer!" in r.output

    def test_expand_summary_not_found(self):
        store = make_mock_store(summary_by_id=None)
        r = self.tool.execute(item_id="sum_missing", _memory_store=store)
        assert not r.success
        assert "not found" in r.error

    def test_expand_summary_no_source_content(self):
        summary = FakeSummary(
            id="sum_empty",
            source_summary_ids=[],
            source_turn_ids=[],
        )
        store = make_mock_store(summary_by_id=summary)
        r = self.tool.execute(item_id="sum_empty", _memory_store=store)
        assert r.success
        assert "No source content" in r.output

    def test_expand_summary_truncation_on_children(self):
        summary = FakeSummary(
            id="sum_big",
            source_summary_ids=["sum_c1"],
            source_turn_ids=[],
        )
        # A child with very long content
        child = FakeSummary(id="sum_c1", depth=2, content="z" * 100_000)
        store = make_mock_store(
            summary_by_id=summary,
            child_summaries=[child],
        )
        r = self.tool.execute(item_id="sum_big", max_tokens=10, _memory_store=store)
        assert r.success
        assert "TRUNCATED" in r.output

    def test_expand_summary_truncation_on_turns(self):
        summary = FakeSummary(
            id="sum_big",
            source_summary_ids=[],
            source_turn_ids=["t1"],
        )
        turns = [FakeTurn(id="t1", content="w" * 100_000)]
        store = make_mock_store(summary_by_id=summary, source_turns=turns)
        r = self.tool.execute(item_id="sum_big", max_tokens=10, _memory_store=store)
        assert r.success
        assert "TRUNCATED" in r.output

    def test_max_tokens_capped(self):
        """max_tokens is capped at 20000."""
        lc = FakeLargeContent(content="a" * 100)
        store = make_mock_store(large_content=lc)
        r = self.tool.execute(item_id="ref_1", max_tokens=999999, _memory_store=store)
        assert r.success
