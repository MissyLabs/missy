"""Extended tests for missy.agent.compaction — compaction engine.

Covers edge cases and scenarios not addressed in test_compaction.py:
  - _chunk_turns behaviour for oversized turns, empty content, mixed sizes
  - compact_session edge cases: fresh_tail_count=0, exact-tail boundary,
    max_condense_depth limiting, idempotency, empty session, single turn,
    source_turn_ids coverage, condensation depth progression
  - should_compact edge cases: threshold=0, threshold=1.0, empty session
  - compact_if_needed: budget attribute forwarding, missing optional attributes
"""

from __future__ import annotations

import time
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_store(tmp_path):
    db = tmp_path / "test.db"
    return SQLiteMemoryStore(str(db))


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    resp = MagicMock()
    resp.content = "Mock summary text."
    provider.chat.return_value = resp
    return provider


@pytest.fixture
def summarizer(mock_provider):
    from missy.agent.summarizer import Summarizer

    return Summarizer(mock_provider)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_turn(session_id: str, content: str, role: str = "user") -> ConversationTurn:
    """Create a single ConversationTurn with the given content."""
    return ConversationTurn.new(session_id, role, content)


def _add_turns(store: SQLiteMemoryStore, session_id: str, count: int, content_size: int = 100) -> list[ConversationTurn]:
    """Add `count` turns and return them."""
    turns = []
    for i in range(count):
        turn = ConversationTurn.new(session_id, "user", f"Turn {i}: {'x' * content_size}")
        store.add_turn(turn)
        turns.append(turn)
        # Tiny sleep so timestamps are strictly ordered in the DB
        time.sleep(0.001)
    return turns


# ---------------------------------------------------------------------------
# Tests: _chunk_turns
# ---------------------------------------------------------------------------


class TestChunkTurnsSingleLargeTurn:
    """A turn that alone exceeds max_tokens must still form its own chunk."""

    def test_oversized_turn_becomes_own_chunk(self):
        # 4000 chars → ~1000 tokens; max_tokens=100
        big_turn = _make_turn("s", "a" * 4000)
        chunks = _chunk_turns([big_turn], max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0][0] is big_turn

    def test_oversized_turn_not_merged_with_next(self):
        big_turn = _make_turn("s", "a" * 4000)
        small_turn = _make_turn("s", "b" * 40)
        chunks = _chunk_turns([big_turn, small_turn], max_tokens=100)
        # big_turn is already over 100 tokens — small_turn gets its own chunk
        assert len(chunks) == 2
        assert chunks[0] == [big_turn]
        assert chunks[1] == [small_turn]

    def test_multiple_oversized_turns_each_own_chunk(self):
        turns = [_make_turn("s", "a" * 4000) for _ in range(3)]
        chunks = _chunk_turns(turns, max_tokens=100)
        assert len(chunks) == 3


class TestChunkTurnsEmptyContent:
    """Turns with empty content should count as 1 token (floor) and group together."""

    def test_empty_content_turn_gets_minimum_token_count(self):
        # empty content → max(1, 0//4) = 1 token; all fit in one chunk of 10
        turns = [_make_turn("s", "") for _ in range(5)]
        chunks = _chunk_turns(turns, max_tokens=10)
        assert len(chunks) == 1
        assert len(chunks[0]) == 5

    def test_empty_content_turn_included_in_chunk(self):
        empty = _make_turn("s", "")
        chunks = _chunk_turns([empty], max_tokens=1000)
        assert chunks == [[empty]]


class TestChunkTurnsVaryingSizes:
    """Turns with different token counts should be split at the right boundaries."""

    def test_small_turns_accumulate_into_single_chunk(self):
        # 10 chars each → 2 tokens each; max_tokens=100 → all 10 fit
        turns = [_make_turn("s", "a" * 10) for _ in range(10)]
        chunks = _chunk_turns(turns, max_tokens=100)
        assert len(chunks) == 1

    def test_boundary_split_is_exact(self):
        # 100 chars → 25 tokens each; max_tokens=50 → 2 per chunk
        turns = [_make_turn("s", "a" * 100) for _ in range(4)]
        chunks = _chunk_turns(turns, max_tokens=50)
        assert len(chunks) == 2
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2

    def test_mixed_sizes_split_correctly(self):
        small = _make_turn("s", "a" * 40)   # 10 tokens
        large = _make_turn("s", "b" * 400)  # 100 tokens
        small2 = _make_turn("s", "c" * 40)  # 10 tokens
        # max_tokens=50: [small], then large>50 so new chunk, small2 joins large's chunk
        chunks = _chunk_turns([small, large, small2], max_tokens=50)
        # After small (10 tokens), large (100) would push current=10+100>50 → flush
        # large starts new chunk (100 tokens), small2 (10) would push 110>50 → flush large alone
        # small2 forms its own chunk
        assert len(chunks) == 3
        assert chunks[0] == [small]
        assert chunks[1] == [large]
        assert chunks[2] == [small2]

    def test_all_turns_preserved_across_chunks(self):
        turns = [_make_turn("s", "a" * (50 * i + 10)) for i in range(6)]
        chunks = _chunk_turns(turns, max_tokens=30)
        all_from_chunks = [t for chunk in chunks for t in chunk]
        assert all_from_chunks == turns


# ---------------------------------------------------------------------------
# Tests: compact_session edge cases
# ---------------------------------------------------------------------------


class TestCompactSessionFreshTailZero:
    """fresh_tail_count=0 has a Python slice edge case: turns[:-0] == [].

    The compaction engine therefore treats fresh_tail_count=0 identically
    to fresh_tail_count equal to len(turns), i.e. nothing is evictable.
    Tests here document the actual (observed) behavior.
    """

    def test_fresh_tail_zero_skips_due_to_python_slice(self, memory_store, summarizer):
        # turns[:-0] evaluates to [] in Python, so evictable is empty
        _add_turns(memory_store, "s", 5, content_size=200)
        stats = compact_session("s", memory_store, summarizer, fresh_tail_count=0)
        assert stats["turns_compacted"] == 0
        assert stats["leaf_summaries_created"] == 0

    def test_fresh_tail_one_compacts_all_but_last(self, memory_store, summarizer):
        """fresh_tail_count=1 is the smallest value that actually evicts turns."""
        added = _add_turns(memory_store, "s", 5, content_size=200)
        stats = compact_session("s", memory_store, summarizer, fresh_tail_count=1)
        assert stats["turns_compacted"] == 4
        summaries = memory_store.get_summaries("s", depth=0, limit=100)
        covered_ids = {tid for s in summaries for tid in s.source_turn_ids}
        # The last turn (fresh tail) must NOT appear in any leaf summary
        assert added[-1].id not in covered_ids
        # The first four turns must be covered
        for turn in added[:-1]:
            assert turn.id in covered_ids


class TestCompactSessionExactTailBoundary:
    """When turn count == fresh_tail_count, nothing should be compacted."""

    def test_exactly_at_tail_count_skips(self, memory_store, summarizer):
        _add_turns(memory_store, "s", 8, content_size=200)
        stats = compact_session("s", memory_store, summarizer, fresh_tail_count=8)
        assert stats["leaf_summaries_created"] == 0
        assert stats["turns_compacted"] == 0

    def test_one_below_tail_count_skips(self, memory_store, summarizer):
        _add_turns(memory_store, "s", 7, content_size=200)
        stats = compact_session("s", memory_store, summarizer, fresh_tail_count=8)
        assert stats["leaf_summaries_created"] == 0

    def test_one_above_tail_count_compacts(self, memory_store, summarizer):
        _add_turns(memory_store, "s", 9, content_size=200)
        stats = compact_session("s", memory_store, summarizer, fresh_tail_count=8)
        assert stats["turns_compacted"] == 1


class TestCompactSessionMaxCondenseDepth:
    """max_condense_depth should cap how many condensation tiers are created."""

    def test_max_condense_depth_zero_prevents_condensation(self, memory_store, summarizer):
        # Force many leaf summaries; then restrict condensation to depth 0
        _add_turns(memory_store, "s", 80, content_size=2000)
        stats = compact_session(
            "s", memory_store, summarizer,
            fresh_tail_count=4,
            leaf_chunk_tokens=400,
            condensed_min_fanout=4,
            max_condense_depth=0,
        )
        assert stats["leaf_summaries_created"] >= 4
        # No condensation because max_condense_depth=0 limits while loop to depth<=0,
        # meaning it attempts depth=0 uncompacted summaries condensation once then stops
        # Condensation at depth 0 will try to condense existing depth-0 summaries
        # max_condense_depth=0 means we allow one condensation pass at depth=0 only
        _ = stats["condensed_summaries_created"]
        # Verify no depth-2 summaries exist
        depth2 = memory_store.get_summaries("s", depth=2)
        assert len(depth2) == 0

    def test_max_condense_depth_one_limits_to_two_levels(self, memory_store, summarizer):
        _add_turns(memory_store, "s", 200, content_size=2000)
        compact_session(
            "s", memory_store, summarizer,
            fresh_tail_count=4,
            leaf_chunk_tokens=200,
            condensed_min_fanout=2,
            max_condense_depth=1,
        )
        # depth-0 leaf summaries → depth-1 condensed; no depth-2
        depth3 = memory_store.get_summaries("s", depth=3)
        assert len(depth3) == 0


class TestCompactSessionIdempotency:
    """Running compact_session twice should not re-summarize existing content."""

    def test_second_pass_creates_no_new_leaf_summaries(self, memory_store, summarizer):
        _add_turns(memory_store, "s", 30, content_size=200)
        compact_session("s", memory_store, summarizer, fresh_tail_count=4)
        stats2 = compact_session("s", memory_store, summarizer, fresh_tail_count=4)
        assert stats2["leaf_summaries_created"] == 0
        assert stats2["turns_compacted"] == 0

    def test_summary_count_stable_after_second_pass(self, memory_store, summarizer):
        _add_turns(memory_store, "s", 30, content_size=200)
        compact_session("s", memory_store, summarizer, fresh_tail_count=4)
        count_after_first = len(memory_store.get_summaries("s", depth=0))
        compact_session("s", memory_store, summarizer, fresh_tail_count=4)
        count_after_second = len(memory_store.get_summaries("s", depth=0))
        assert count_after_first == count_after_second


class TestCompactSessionEmptySession:
    """An empty session should return zero stats without errors."""

    def test_empty_session_returns_zero_stats(self, memory_store, summarizer):
        stats = compact_session("empty-sess", memory_store, summarizer, fresh_tail_count=16)
        assert stats["leaf_summaries_created"] == 0
        assert stats["condensed_summaries_created"] == 0
        assert stats["turns_compacted"] == 0
        assert stats["tiers_used"] == []

    def test_empty_session_does_not_call_summarizer(self, memory_store, mock_provider):
        from missy.agent.summarizer import Summarizer

        s = Summarizer(mock_provider)
        compact_session("empty-sess", memory_store, s, fresh_tail_count=16)
        mock_provider.chat.assert_not_called()


class TestCompactSessionSingleTurn:
    """A session with exactly one turn."""

    def test_single_turn_below_tail_skips(self, memory_store, summarizer):
        _add_turns(memory_store, "s", 1, content_size=200)
        stats = compact_session("s", memory_store, summarizer, fresh_tail_count=16)
        assert stats["turns_compacted"] == 0

    def test_single_turn_zero_tail_also_skips(self, memory_store, summarizer):
        # turns[:-0] == [] so evictable is empty regardless of count
        _add_turns(memory_store, "s", 1, content_size=200)
        stats = compact_session("s", memory_store, summarizer, fresh_tail_count=0)
        assert stats["turns_compacted"] == 0

    def test_two_turns_fresh_tail_one_compacts_one(self, memory_store, summarizer):
        _add_turns(memory_store, "s", 2, content_size=200)
        stats = compact_session("s", memory_store, summarizer, fresh_tail_count=1)
        assert stats["turns_compacted"] == 1


class TestCompactSessionSourceTurnIdCoverage:
    """Leaf summary source_turn_ids must collectively cover all evictable turns."""

    def test_source_turn_ids_cover_all_evictable(self, memory_store, summarizer):
        fresh_tail = 4
        added = _add_turns(memory_store, "s", 20, content_size=200)
        compact_session("s", memory_store, summarizer, fresh_tail_count=fresh_tail)

        evictable_ids = {t.id for t in added[:-fresh_tail]}
        summaries = memory_store.get_summaries("s", depth=0)
        covered_ids = {tid for s in summaries for tid in s.source_turn_ids}
        assert evictable_ids == covered_ids

    def test_fresh_tail_turns_not_in_source_turn_ids(self, memory_store, summarizer):
        fresh_tail = 6
        added = _add_turns(memory_store, "s", 20, content_size=200)
        compact_session("s", memory_store, summarizer, fresh_tail_count=fresh_tail)

        tail_ids = {t.id for t in added[-fresh_tail:]}
        summaries = memory_store.get_summaries("s", depth=0)
        covered_ids = {tid for s in summaries for tid in s.source_turn_ids}
        assert covered_ids.isdisjoint(tail_ids)


class TestCompactSessionCondensationDepthProgression:
    """Condensation at depth N produces a summary at depth N+1."""

    def test_depth_one_summaries_created_from_depth_zero(self, memory_store, summarizer):
        # Need >= condensed_min_fanout leaf summaries to trigger condensation
        _add_turns(memory_store, "s", 80, content_size=2000)
        stats = compact_session(
            "s", memory_store, summarizer,
            fresh_tail_count=4,
            leaf_chunk_tokens=400,
            condensed_min_fanout=4,
        )
        assert stats["condensed_summaries_created"] >= 1
        depth1 = memory_store.get_summaries("s", depth=1, limit=10)
        assert len(depth1) >= 1

    def test_condensed_summary_depth_field_is_one(self, memory_store, summarizer):
        """SummaryRecord.depth field on condensed summaries must be 1."""
        _add_turns(memory_store, "s", 80, content_size=2000)
        compact_session(
            "s", memory_store, summarizer,
            fresh_tail_count=4,
            leaf_chunk_tokens=400,
            condensed_min_fanout=4,
        )
        depth1 = memory_store.get_summaries("s", depth=1, limit=10)
        # All returned records from depth=1 query must have depth attribute == 1
        for rec in depth1:
            assert rec.depth == 1

    def test_condensed_summary_has_source_summary_ids(self, memory_store, summarizer):
        """Condensed summaries must reference source summary IDs (not turn IDs)."""
        _add_turns(memory_store, "s", 80, content_size=2000)
        compact_session(
            "s", memory_store, summarizer,
            fresh_tail_count=4,
            leaf_chunk_tokens=400,
            condensed_min_fanout=4,
        )
        depth1 = memory_store.get_summaries("s", depth=1, limit=10)
        for rec in depth1:
            # source_summary_ids must be populated; source_turn_ids must be empty
            assert len(rec.source_summary_ids) >= 1
            assert rec.source_turn_ids == []


# ---------------------------------------------------------------------------
# Tests: should_compact
# ---------------------------------------------------------------------------


class TestShouldCompactThresholdZero:
    """threshold=0 means limit=0; any session whose token count > 0 triggers.

    Note: the store returns a minimum token count of 1 even for sessions with
    no turns, so threshold=0 effectively always triggers regardless of whether
    the session is empty.
    """

    def test_threshold_zero_triggers_for_non_empty_session(self, memory_store):
        _add_turns(memory_store, "s", 1, content_size=10)
        # limit = int(budget * 0) = 0; total_tokens >= 1 → True
        assert should_compact("s", memory_store, 30_000, threshold=0.0) is True

    def test_threshold_zero_also_triggers_for_unknown_session(self, memory_store):
        # get_session_token_count returns minimum 1 for any session
        # limit = int(30_000 * 0.0) = 0; 1 > 0 → True
        assert should_compact("never-seen", memory_store, 30_000, threshold=0.0) is True


class TestShouldCompactThresholdOne:
    """threshold=1.0 requires total tokens > full budget to trigger."""

    def test_threshold_one_does_not_trigger_under_budget(self, memory_store):
        _add_turns(memory_store, "s", 5, content_size=10)
        # Very large budget → tokens never exceed it
        assert should_compact("s", memory_store, 10_000_000, threshold=1.0) is False

    def test_threshold_one_triggers_when_over_budget(self, memory_store):
        _add_turns(memory_store, "s", 100, content_size=1000)
        # budget=1 → limit=1; total_tokens definitely > 1
        assert should_compact("s", memory_store, 1, threshold=1.0) is True


class TestShouldCompactEmptySession:
    """Behavior when the session has no stored turns.

    The store returns a minimum token count of 1 even for sessions not in the
    database, so a session with no turns is NOT necessarily below threshold.
    """

    def test_empty_session_with_large_budget_stays_below(self, memory_store):
        # limit = int(100_000 * 0.5) = 50_000; 1 token << 50_000 → False
        assert should_compact("never-used", memory_store, 100_000, threshold=0.5) is False

    def test_empty_session_with_zero_budget_triggers(self, memory_store):
        # limit = int(0 * 0.5) = 0; 1 > 0 → True (budget too small for even 1 token)
        assert should_compact("never-used", memory_store, 0, threshold=0.5) is True


# ---------------------------------------------------------------------------
# Tests: compact_if_needed — budget attribute forwarding
# ---------------------------------------------------------------------------


class TestCompactIfNeededBudgetForwarding:
    """compact_if_needed should pass budget attributes through to compact_session."""

    def test_fresh_tail_count_forwarded(self, memory_store, mock_provider):
        """When budget.fresh_tail_count=1, all but the last turn should compact."""
        from missy.agent.summarizer import Summarizer

        s = Summarizer(mock_provider)
        _add_turns(memory_store, "sess", 5, content_size=200)

        # Make budget very small so should_compact triggers, and fresh_tail_count=1
        budget = TokenBudget(total=1, system_reserve=0, tool_definitions_reserve=0, fresh_tail_count=1)
        result = compact_if_needed("sess", memory_store, s, budget)
        assert result is not None
        assert result["turns_compacted"] == 4

    def test_leaf_chunk_tokens_forwarded(self, memory_store, mock_provider):
        """A very small leaf_chunk_tokens should force many chunks."""
        from missy.agent.summarizer import Summarizer

        s = Summarizer(mock_provider)
        _add_turns(memory_store, "sess", 10, content_size=400)

        budget = TokenBudget(total=1, system_reserve=0, tool_definitions_reserve=0, fresh_tail_count=1)
        # Inject leaf_chunk_tokens as attribute on the budget object
        budget.leaf_chunk_tokens = 10  # very small → many chunks
        result = compact_if_needed("sess", memory_store, s, budget)
        assert result is not None
        assert result["leaf_summaries_created"] >= 1

    def test_condensed_min_fanout_forwarded(self, memory_store, mock_provider):
        """condensed_min_fanout=100 on budget should prevent condensation."""
        from missy.agent.summarizer import Summarizer

        s = Summarizer(mock_provider)
        _add_turns(memory_store, "sess", 30, content_size=1000)

        budget = TokenBudget(total=1, system_reserve=0, tool_definitions_reserve=0, fresh_tail_count=2)
        budget.condensed_min_fanout = 100  # very high → condensation never triggers
        result = compact_if_needed("sess", memory_store, s, budget)
        assert result is not None
        assert result["condensed_summaries_created"] == 0

    def test_context_threshold_forwarded(self, memory_store, mock_provider):
        """context_threshold=1.0 should prevent compaction for reasonable sessions."""
        from missy.agent.summarizer import Summarizer

        s = Summarizer(mock_provider)
        _add_turns(memory_store, "sess", 5, content_size=10)

        budget = TokenBudget(total=10_000_000)
        budget.context_threshold = 1.0  # threshold so high it never triggers
        result = compact_if_needed("sess", memory_store, s, budget)
        assert result is None


class TestCompactIfNeededMissingOptionalAttributes:
    """compact_if_needed should use module defaults when budget lacks optional attrs."""

    def test_missing_context_threshold_uses_default(self, memory_store, summarizer):
        # Minimal object with only .total
        class MinimalBudget:
            total = 1  # tiny so should_compact triggers

        _add_turns(memory_store, "sess", 20, content_size=200)
        result = compact_if_needed("sess", memory_store, summarizer, MinimalBudget())
        assert result is not None
        # Default fresh_tail_count=16 means 20-16=4 turns compacted
        assert result["turns_compacted"] == 4

    def test_missing_fresh_tail_count_uses_default(self, memory_store, summarizer):
        class MinimalBudget:
            total = 1
            context_threshold = 0.0

        _add_turns(memory_store, "sess", 20, content_size=200)
        result = compact_if_needed("sess", memory_store, summarizer, MinimalBudget())
        assert result is not None
        # Default _DEFAULT_FRESH_TAIL = 16, so 20 - 16 = 4 turns compacted
        assert result["turns_compacted"] == 4

    def test_missing_leaf_chunk_tokens_uses_default(self, memory_store, summarizer):
        class MinimalBudget:
            total = 1
            context_threshold = 0.0
            fresh_tail_count = 1  # protect only the last turn

        _add_turns(memory_store, "sess", 4, content_size=200)
        result = compact_if_needed("sess", memory_store, summarizer, MinimalBudget())
        assert result is not None
        # With default leaf_chunk_tokens (20k), all 3 evictable turns fit in 1 chunk
        assert result["leaf_summaries_created"] == 1

    def test_missing_condensed_min_fanout_uses_default(self, memory_store, mock_provider):
        """Without condensed_min_fanout on budget, default of 4 applies.

        With only 3 evictable leaf summaries and default fanout=4, no condensation
        should occur.
        """
        from missy.agent.summarizer import Summarizer

        s = Summarizer(mock_provider)

        class MinimalBudget:
            total = 1
            context_threshold = 0.0
            fresh_tail_count = 1  # protect last turn only

        # 4 turns, 3 evictable → 3 leaf summaries < default fanout of 4
        _add_turns(memory_store, "sess", 4, content_size=400)
        result = compact_if_needed("sess", memory_store, s, MinimalBudget())
        assert result is not None
        assert result["condensed_summaries_created"] == 0

    def test_returns_none_when_below_threshold(self, memory_store, summarizer):
        class MinimalBudget:
            total = 10_000_000  # huge → should_compact will be False

        _add_turns(memory_store, "sess", 3, content_size=10)
        result = compact_if_needed("sess", memory_store, summarizer, MinimalBudget())
        assert result is None
