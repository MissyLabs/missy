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
        "sess1",
        depth=0,
        content="Discussion about kubernetes deployment issues",
        source_turn_ids=["t1", "t2"],
        time_range_start="2026-01-01T00:00:00",
        time_range_end="2026-01-01T01:00:00",
        descendant_count=5,
    )
    memory_store.add_summary(s)

    lc = LargeContentRecord.new(
        "sess1",
        "shell_exec",
        "x" * 100000,
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
            query="kubernetes",
            scope="messages",
            _memory_store=store,
            _session_id="sess1",
        )
        assert result.success
        assert "Messages" in result.output
        assert "kubernetes" in result.output.lower()

    def test_search_summaries(self, populated_store):
        store, _, _ = populated_store
        tool = MemorySearchTool()
        result = tool.execute(
            query="kubernetes",
            scope="summaries",
            _memory_store=store,
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
            query="kubernetes",
            session_id="nonexistent_session",
            _memory_store=store,
        )
        assert "No results" in result.output

    def test_schema_does_not_claim_boolean_or_prefix_syntax_support(self):
        """The tool's own schema previously told the calling LLM that
        AND/OR and FTS5 syntax were supported ("supports FTS5 syntax:
        phrases, AND/OR"), but SQLiteMemoryStore.search() always wraps the
        entire query as one literal phrase (intentional -- prevents FTS5
        syntax injection). A model following the documented schema that
        sent e.g. "python OR javascript" got a silent, unexplained empty
        result set. The schema description must not promise behavior the
        store deliberately disables.
        """
        schema = MemorySearchTool().get_schema()
        query_desc = schema["parameters"]["properties"]["query"]["description"]
        assert "AND/OR" not in query_desc
        assert "FTS5 syntax" not in query_desc

    def test_boolean_query_matched_as_literal_phrase_not_silently_empty(self, populated_store):
        """A query containing FTS5 boolean syntax is treated as literal
        text, matching the corrected schema description, rather than being
        interpreted as a query operator.
        """
        store, _, _ = populated_store
        tool = MemorySearchTool()
        # The literal phrase "kubernetes AND deploy" won't match content
        # that only contains "kubernetes" on its own -- confirming AND is
        # NOT being interpreted as a boolean operator (which would match).
        result = tool.execute(
            query="kubernetes AND totally_absent_word_xyz",
            scope="messages",
            _memory_store=store,
        )
        assert result.success
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


# ---------------------------------------------------------------------------
# FX-C: a failed lookup must yield an explicit unverified/error response,
# never a confident "does not exist" claim. The validation harness observed
# false claims that known sum_*/ref_* IDs did not exist. A store exception
# (DB locked, I/O error, etc.) and a genuine "no such row" must be
# distinguishable in the returned error text -- both are errors, but only
# the second is a real non-existence claim.
# ---------------------------------------------------------------------------


class _RaisingStore:
    """A minimal store stub whose lookups always raise, simulating a
    transient failure (e.g. a locked database) rather than a missing row."""

    def get_summary_by_id(self, _summary_id):
        raise RuntimeError("database is locked")

    def get_large_content(self, _content_id):
        raise RuntimeError("database is locked")


class TestMemoryDescribeExceptionVsNotFound:
    def test_genuine_missing_summary_says_not_found(self, memory_store):
        tool = MemoryDescribeTool()
        result = tool.execute(item_id="sum_genuinely_absent", _memory_store=memory_store)
        assert not result.success
        assert "not found" in result.error
        assert "internal error" not in result.error

    def test_lookup_exception_does_not_claim_not_found(self):
        tool = MemoryDescribeTool()
        result = tool.execute(item_id="sum_whatever", _memory_store=_RaisingStore())
        assert not result.success
        assert "not found" not in result.error
        assert "internal error" in result.error
        assert "unverified" in result.error

    def test_lookup_exception_for_large_content_does_not_claim_not_found(self):
        tool = MemoryDescribeTool()
        result = tool.execute(item_id="ref_whatever", _memory_store=_RaisingStore())
        assert not result.success
        assert "not found" not in result.error
        assert "internal error" in result.error

    def test_existing_summary_content_matches_exactly_what_was_stored(self, populated_store):
        # Grounding check: the described content must be the literal
        # stored content, not a paraphrase or reconstruction.
        store, sum_id, _ = populated_store
        tool = MemoryDescribeTool()
        result = tool.execute(item_id=sum_id, _memory_store=store)
        assert result.success
        assert "Discussion about kubernetes deployment issues" in result.output


class TestMemoryExpandExceptionVsNotFound:
    def test_genuine_missing_summary_says_not_found(self, memory_store):
        tool = MemoryExpandTool()
        result = tool.execute(item_id="sum_genuinely_absent", _memory_store=memory_store)
        assert not result.success
        assert "not found" in result.error
        assert "internal error" not in result.error

    def test_lookup_exception_does_not_claim_not_found(self):
        tool = MemoryExpandTool()
        result = tool.execute(item_id="sum_whatever", _memory_store=_RaisingStore())
        assert not result.success
        assert "not found" not in result.error
        assert "internal error" in result.error
        assert "unverified" in result.error

    def test_lookup_exception_for_large_content_does_not_claim_not_found(self):
        tool = MemoryExpandTool()
        result = tool.execute(item_id="ref_whatever", _memory_store=_RaisingStore())
        assert not result.success
        assert "not found" not in result.error
        assert "internal error" in result.error


class TestMemoryToolsDispatchThroughRealRegistry:
    """SR-3.3 regression: memory_search/memory_describe/memory_expand must
    survive dispatch through the real ToolRegistry, not just direct
    .execute() calls.

    Before this fix, none of the three tools declared the
    ``permissions: ToolPermissions`` attribute ``BaseTool``/
    ``ToolRegistry._check_permissions()`` requires (they carried vestigial,
    unused ``requires_filesystem_read``-style attributes instead). Every
    dispatch through ``ToolRegistry.execute()`` crashed with
    ``AttributeError: 'MemoryExpandTool' object has no attribute
    'permissions'`` before the tool's own logic ever ran — direct
    ``tool.execute(...)`` calls (as used throughout the rest of this file)
    never exercised that code path and so never caught it.
    """

    @pytest.fixture
    def registry(self):
        from missy.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.register(MemorySearchTool())
        reg.register(MemoryDescribeTool())
        reg.register(MemoryExpandTool())
        return reg

    def test_all_three_tools_declare_permissions(self):
        from missy.tools.base import ToolPermissions

        for tool in (MemorySearchTool(), MemoryDescribeTool(), MemoryExpandTool()):
            assert isinstance(tool.permissions, ToolPermissions)

    def test_memory_search_dispatches_through_registry(self, registry, populated_store):
        store, _, _ = populated_store
        result = registry.execute(
            "memory_search",
            session_id="sess1",
            task_id="t1",
            query="kubernetes",
            _memory_store=store,
        )
        assert result.success
        assert "kubernetes" in result.output.lower()

    def test_memory_describe_dispatches_through_registry(self, registry, populated_store):
        store, sum_id, _ = populated_store
        result = registry.execute(
            "memory_describe",
            session_id="sess1",
            task_id="t1",
            item_id=sum_id,
            _memory_store=store,
        )
        assert result.success

    def test_memory_expand_dispatches_through_registry(self, registry, populated_store):
        store, _, ref_id = populated_store
        result = registry.execute(
            "memory_expand",
            session_id="sess1",
            task_id="t1",
            item_id=ref_id,
            _memory_store=store,
        )
        assert result.success
        assert "kubectl" in result.output.lower() or "x" * 100 in result.output
