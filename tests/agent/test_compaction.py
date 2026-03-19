"""Tests for missy.agent.compaction — compaction engine."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from missy.agent.compaction import (
    _chunk_turns,
    compact_if_needed,
    compact_session,
    should_compact,
)
from missy.agent.context import TokenBudget
from missy.memory.sqlite_store import ConversationTurn, SQLiteMemoryStore


@pytest.fixture
def memory_store():
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    return SQLiteMemoryStore(db)


@pytest.fixture
def summarizer():
    provider = MagicMock()
    resp = MagicMock()
    resp.content = "Summary of the conversation."
    provider.chat.return_value = resp
    from missy.agent.summarizer import Summarizer
    return Summarizer(provider)


def _add_turns(store, session_id, count, content_size=100):
    """Add `count` turns to the store."""
    for i in range(count):
        turn = ConversationTurn.new(session_id, "user", f"Turn {i}: {'x' * content_size}")
        store.add_turn(turn)


class TestChunkTurns:
    def test_single_chunk(self):
        turns = [ConversationTurn.new("s", "user", "a" * 100) for _ in range(3)]
        chunks = _chunk_turns(turns, max_tokens=1000)
        assert len(chunks) == 1
        assert len(chunks[0]) == 3

    def test_multiple_chunks(self):
        # Each turn is ~25 tokens (100 chars / 4), max_tokens=30 → ~1 per chunk
        turns = [ConversationTurn.new("s", "user", "a" * 100) for _ in range(5)]
        chunks = _chunk_turns(turns, max_tokens=30)
        assert len(chunks) == 5

    def test_empty_input(self):
        assert _chunk_turns([], max_tokens=100) == []


class TestCompactSession:
    def test_skips_when_few_turns(self, memory_store, summarizer):
        _add_turns(memory_store, "sess1", 5)
        stats = compact_session("sess1", memory_store, summarizer, fresh_tail_count=16)
        assert stats["leaf_summaries_created"] == 0
        assert stats["turns_compacted"] == 0

    def test_creates_leaf_summaries(self, memory_store, summarizer):
        _add_turns(memory_store, "sess1", 30, content_size=200)
        stats = compact_session("sess1", memory_store, summarizer, fresh_tail_count=16)
        assert stats["leaf_summaries_created"] > 0
        assert stats["turns_compacted"] > 0
        # Verify summaries were stored
        summaries = memory_store.get_summaries("sess1", depth=0)
        assert len(summaries) > 0

    def test_does_not_resummarize_existing(self, memory_store, summarizer):
        _add_turns(memory_store, "sess1", 30, content_size=200)
        compact_session("sess1", memory_store, summarizer, fresh_tail_count=16)
        stats2 = compact_session("sess1", memory_store, summarizer, fresh_tail_count=16)
        # Second pass should find nothing new to summarize
        assert stats2["leaf_summaries_created"] == 0

    def test_condensation_triggers_at_fanout(self, memory_store, summarizer):
        # Add enough turns to create >= 4 leaf summaries
        _add_turns(memory_store, "sess1", 100, content_size=2000)
        stats = compact_session(
            "sess1", memory_store, summarizer,
            fresh_tail_count=16,
            leaf_chunk_tokens=500,     # small chunks to force many leaves
            condensed_min_fanout=4,
        )
        assert stats["leaf_summaries_created"] >= 4
        assert stats["condensed_summaries_created"] >= 1


class TestShouldCompact:
    def test_below_threshold(self, memory_store):
        _add_turns(memory_store, "sess1", 5, content_size=10)
        assert should_compact("sess1", memory_store, 30_000) is False

    def test_above_threshold(self, memory_store):
        _add_turns(memory_store, "sess1", 100, content_size=1000)
        assert should_compact("sess1", memory_store, 100, threshold=0.5) is True


class TestCompactIfNeeded:
    def test_no_compaction_when_small(self, memory_store, summarizer):
        _add_turns(memory_store, "sess1", 5)
        budget = TokenBudget(total=100_000)
        result = compact_if_needed("sess1", memory_store, summarizer, budget)
        assert result is None

    def test_compacts_when_large(self, memory_store, summarizer):
        _add_turns(memory_store, "sess1", 50, content_size=1000)
        budget = TokenBudget(total=100, system_reserve=0, tool_definitions_reserve=0)  # very small budget forces compaction
        result = compact_if_needed("sess1", memory_store, summarizer, budget)
        assert result is not None
        assert result["leaf_summaries_created"] > 0
