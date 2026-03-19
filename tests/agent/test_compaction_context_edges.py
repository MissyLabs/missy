"""Session 14: Edge case tests for compaction and context modules.

Covers:
- _chunk_turns: empty list, single large turn, boundary splitting
- compact_session: no turns, fewer than fresh_tail, existing summaries
- should_compact: below/above threshold
- compact_if_needed: triggers/skips based on token count
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Fake objects
# ---------------------------------------------------------------------------


@dataclass
class FakeTurn:
    id: str = "t1"
    timestamp: str = "2026-03-19T12:00:00"
    role: str = "user"
    content: str = "Hello"
    session_id: str = "sess-1"


@dataclass
class FakeSummaryRecord:
    id: str = "s1"
    session_id: str = "sess-1"
    depth: int = 0
    content: str = "Summary"
    source_turn_ids: list = field(default_factory=list)
    source_summary_ids: list = field(default_factory=list)
    time_range_start: str = ""
    time_range_end: str = ""
    descendant_count: int = 0
    compacted_into: str | None = None

    @classmethod
    def new(cls, **kwargs) -> FakeSummaryRecord:
        return cls(**kwargs)


class FakeMemoryStore:
    def __init__(self, turns=None, summaries=None, token_count=0):
        self._turns = turns or []
        self._summaries = summaries or []
        self._token_count = token_count
        self._added_summaries = []

    def get_session_turns(self, session_id, limit=10000):
        return self._turns[:limit]

    def get_summaries(self, session_id, depth=0, limit=10000):
        return [s for s in self._summaries if s.depth == depth][:limit]

    def get_uncompacted_summaries(self, session_id, depth):
        return [s for s in self._summaries if s.depth == depth and s.compacted_into is None]

    def add_summary(self, record):
        self._added_summaries.append(record)

    def mark_summary_compacted(self, ids, parent_id):
        for s in self._summaries:
            if s.id in ids:
                s.compacted_into = parent_id

    def get_session_token_count(self, session_id):
        return self._token_count


class FakeSummarizer:
    def __init__(self, response="Summarized text"):
        self._response = response
        self.calls = []

    def summarize_turns(self, turns, prior_summary="", target_tokens=1200):
        self.calls.append(("turns", len(turns)))
        return self._response, "normal"

    def summarize_summaries(self, summaries, target_tokens=2000):
        self.calls.append(("summaries", len(summaries)))
        return self._response, "normal"


# ---------------------------------------------------------------------------
# _chunk_turns tests
# ---------------------------------------------------------------------------


class TestChunkTurns:
    """Tests for _chunk_turns helper."""

    def test_empty_turns(self):
        from missy.agent.compaction import _chunk_turns
        assert _chunk_turns([], 1000) == []

    def test_single_small_turn(self):
        from missy.agent.compaction import _chunk_turns
        turns = [FakeTurn(content="hi")]
        chunks = _chunk_turns(turns, 1000)
        assert len(chunks) == 1
        assert len(chunks[0]) == 1

    def test_single_large_turn(self):
        """A single turn larger than max_tokens should still be in one chunk."""
        from missy.agent.compaction import _chunk_turns
        turns = [FakeTurn(content="x" * 10000)]  # ~2500 tokens
        chunks = _chunk_turns(turns, 100)
        assert len(chunks) == 1  # Can't split a single turn

    def test_multiple_turns_split(self):
        from missy.agent.compaction import _chunk_turns
        turns = [FakeTurn(id=f"t{i}", content="word " * 100) for i in range(10)]
        chunks = _chunk_turns(turns, 200)
        assert len(chunks) > 1
        # All turns accounted for
        total = sum(len(c) for c in chunks)
        assert total == 10

    def test_exact_boundary(self):
        """Turns exactly at the token limit should not split."""
        from missy.agent.compaction import _chunk_turns
        # Each turn is ~25 tokens (100 chars / 4)
        turns = [FakeTurn(id=f"t{i}", content="x" * 100) for i in range(4)]
        chunks = _chunk_turns(turns, 100)  # 100 tokens = 4 turns
        assert len(chunks) == 1

    def test_turns_with_empty_content(self):
        from missy.agent.compaction import _chunk_turns
        turns = [FakeTurn(id=f"t{i}", content="") for i in range(5)]
        chunks = _chunk_turns(turns, 10)
        # Empty content → ~1 token each, 5 total fits in chunk
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# compact_session tests
# ---------------------------------------------------------------------------


class TestCompactSession:
    """Tests for compact_session."""

    def test_no_turns_returns_empty_stats(self):
        from missy.agent.compaction import compact_session

        store = FakeMemoryStore(turns=[])
        summarizer = FakeSummarizer()
        with patch("missy.memory.sqlite_store.SummaryRecord", FakeSummaryRecord):
            stats = compact_session("sess-1", store, summarizer)
        assert stats["leaf_summaries_created"] == 0
        assert stats["turns_compacted"] == 0

    def test_fewer_than_fresh_tail(self):
        from missy.agent.compaction import compact_session

        turns = [FakeTurn(id=f"t{i}") for i in range(5)]
        store = FakeMemoryStore(turns=turns)
        summarizer = FakeSummarizer()
        with patch("missy.memory.sqlite_store.SummaryRecord", FakeSummaryRecord):
            stats = compact_session("sess-1", store, summarizer, fresh_tail_count=16)
        assert stats["leaf_summaries_created"] == 0

    def test_exactly_at_fresh_tail(self):
        from missy.agent.compaction import compact_session

        turns = [FakeTurn(id=f"t{i}") for i in range(16)]
        store = FakeMemoryStore(turns=turns)
        summarizer = FakeSummarizer()
        with patch("missy.memory.sqlite_store.SummaryRecord", FakeSummaryRecord):
            stats = compact_session("sess-1", store, summarizer, fresh_tail_count=16)
        assert stats["leaf_summaries_created"] == 0  # Equal, not greater

    def test_compacts_turns_beyond_fresh_tail(self):
        from missy.agent.compaction import compact_session

        turns = [FakeTurn(id=f"t{i}", content="word " * 100) for i in range(20)]
        store = FakeMemoryStore(turns=turns)
        summarizer = FakeSummarizer()
        with patch("missy.memory.sqlite_store.SummaryRecord", FakeSummaryRecord):
            stats = compact_session("sess-1", store, summarizer, fresh_tail_count=4)
        assert stats["leaf_summaries_created"] >= 1
        assert stats["turns_compacted"] == 16  # 20 - 4 fresh tail

    def test_skips_already_summarized_turns(self):
        from missy.agent.compaction import compact_session

        turns = [FakeTurn(id=f"t{i}", content="word " * 100) for i in range(20)]
        # Mark first 8 turns as already summarized
        existing_summary = FakeSummaryRecord(
            id="s1", depth=0,
            source_turn_ids=[f"t{i}" for i in range(8)],
        )
        store = FakeMemoryStore(turns=turns, summaries=[existing_summary])
        summarizer = FakeSummarizer()
        with patch("missy.memory.sqlite_store.SummaryRecord", FakeSummaryRecord):
            stats = compact_session("sess-1", store, summarizer, fresh_tail_count=4)
        # Only 8 turns need summarizing (turns 8-15; 16-19 are fresh tail)
        assert stats["turns_compacted"] == 8


# ---------------------------------------------------------------------------
# should_compact tests
# ---------------------------------------------------------------------------


class TestShouldCompact:
    """Tests for should_compact."""

    def test_below_threshold(self):
        from missy.agent.compaction import should_compact

        store = FakeMemoryStore(token_count=5000)
        assert should_compact("sess-1", store, 10000, threshold=0.75) is False

    def test_above_threshold(self):
        from missy.agent.compaction import should_compact

        store = FakeMemoryStore(token_count=8000)
        assert should_compact("sess-1", store, 10000, threshold=0.75) is True

    def test_exactly_at_threshold(self):
        from missy.agent.compaction import should_compact

        store = FakeMemoryStore(token_count=7500)
        # int(10000 * 0.75) = 7500 → 7500 > 7500 is False
        assert should_compact("sess-1", store, 10000, threshold=0.75) is False

    def test_just_above_threshold(self):
        from missy.agent.compaction import should_compact

        store = FakeMemoryStore(token_count=7501)
        assert should_compact("sess-1", store, 10000, threshold=0.75) is True

    def test_zero_budget(self):
        from missy.agent.compaction import should_compact

        store = FakeMemoryStore(token_count=100)
        # int(0 * 0.75) = 0 → 100 > 0 is True
        assert should_compact("sess-1", store, 0, threshold=0.75) is True

    def test_zero_token_count(self):
        from missy.agent.compaction import should_compact

        store = FakeMemoryStore(token_count=0)
        assert should_compact("sess-1", store, 10000, threshold=0.75) is False


# ---------------------------------------------------------------------------
# compact_if_needed tests
# ---------------------------------------------------------------------------


class TestCompactIfNeeded:
    """Tests for compact_if_needed."""

    def test_no_compaction_needed(self):
        from missy.agent.compaction import compact_if_needed

        store = FakeMemoryStore(token_count=100)
        budget = MagicMock()
        budget.total = 10000
        budget.context_threshold = 0.75
        result = compact_if_needed("sess-1", store, FakeSummarizer(), budget)
        assert result is None

    def test_compaction_triggered(self):
        from missy.agent.compaction import compact_if_needed

        turns = [FakeTurn(id=f"t{i}", content="word " * 100) for i in range(30)]
        store = FakeMemoryStore(turns=turns, token_count=9000)
        budget = MagicMock()
        budget.total = 10000
        budget.context_threshold = 0.75
        budget.fresh_tail_count = 4
        budget.leaf_chunk_tokens = 20000
        budget.condensed_min_fanout = 4
        summarizer = FakeSummarizer()
        with patch("missy.memory.sqlite_store.SummaryRecord", FakeSummaryRecord):
            result = compact_if_needed("sess-1", store, summarizer, budget)
        assert result is not None
        assert result["leaf_summaries_created"] >= 1

    def test_budget_without_threshold_uses_default(self):
        """Budget without context_threshold attr should use default 0.75."""
        from missy.agent.compaction import compact_if_needed

        store = FakeMemoryStore(token_count=100)
        budget = MagicMock(spec=[])
        budget.total = 10000
        # No context_threshold attribute → getattr falls back to default
        del budget.context_threshold
        result = compact_if_needed("sess-1", store, FakeSummarizer(), budget)
        assert result is None
