"""Session 12: Context manager edge cases and vision memory bridge robustness.

Covers:
- ContextManager with malformed history entries
- ContextManager with very large budgets and empty inputs
- ContextManager summary formatting edge cases
- VisionMemoryBridge recall with non-vision turns
- VisionMemoryBridge session context formatting
- _approx_tokens boundary behavior
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# _approx_tokens
# ---------------------------------------------------------------------------


class TestApproxTokens:
    def test_empty_string(self) -> None:
        from missy.agent.context import _approx_tokens
        assert _approx_tokens("") == 1  # minimum 1

    def test_short_string(self) -> None:
        from missy.agent.context import _approx_tokens
        assert _approx_tokens("hi") == 1  # 2 chars // 4 = 0 → max(1, 0) = 1

    def test_exact_boundary(self) -> None:
        from missy.agent.context import _approx_tokens
        assert _approx_tokens("abcd") == 1  # 4 chars // 4 = 1

    def test_long_string(self) -> None:
        from missy.agent.context import _approx_tokens
        assert _approx_tokens("a" * 400) == 100


# ---------------------------------------------------------------------------
# TokenBudget validation
# ---------------------------------------------------------------------------


class TestTokenBudgetValidation:
    def test_negative_total_raises(self) -> None:
        from missy.agent.context import TokenBudget
        with pytest.raises(ValueError, match="total"):
            TokenBudget(total=-1)

    def test_memory_fraction_out_of_range(self) -> None:
        from missy.agent.context import TokenBudget
        with pytest.raises(ValueError, match="memory_fraction"):
            TokenBudget(memory_fraction=1.5)

    def test_learnings_fraction_out_of_range(self) -> None:
        from missy.agent.context import TokenBudget
        with pytest.raises(ValueError, match="learnings_fraction"):
            TokenBudget(learnings_fraction=-0.1)

    def test_negative_fresh_tail_count(self) -> None:
        from missy.agent.context import TokenBudget
        with pytest.raises(ValueError, match="fresh_tail_count"):
            TokenBudget(fresh_tail_count=-1)

    def test_reserves_exceed_total(self) -> None:
        from missy.agent.context import TokenBudget
        with pytest.raises(ValueError, match="exceeds total"):
            TokenBudget(total=100, system_reserve=60, tool_definitions_reserve=60)

    def test_zero_total_with_zero_reserves(self) -> None:
        from missy.agent.context import TokenBudget
        budget = TokenBudget(total=0, system_reserve=0, tool_definitions_reserve=0)
        assert budget.total == 0

    def test_fractions_near_one(self) -> None:
        from missy.agent.context import TokenBudget
        # Both fractions at edge of range should still work
        budget = TokenBudget(memory_fraction=1.0, learnings_fraction=1.0)
        assert budget.memory_fraction == 1.0


# ---------------------------------------------------------------------------
# ContextManager.build_messages
# ---------------------------------------------------------------------------


class TestContextManagerBuildMessages:
    def _mgr(self, **kwargs: Any) -> Any:
        from missy.agent.context import ContextManager, TokenBudget
        return ContextManager(TokenBudget(**kwargs))

    def test_empty_history(self) -> None:
        mgr = self._mgr()
        system, msgs = mgr.build_messages("System.", "Hello", [])
        assert system.startswith("System.")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Hello"

    def test_history_with_non_string_content(self) -> None:
        """History entries with list content (multi-modal) should not crash."""
        mgr = self._mgr()
        history = [
            {"role": "user", "content": ["image", "text"]},
            {"role": "assistant", "content": None},
        ]
        system, msgs = mgr.build_messages("System.", "New", history)
        assert msgs[-1]["content"] == "New"

    def test_memory_results_truncated(self) -> None:
        """Memory results exceeding budget get truncated."""
        mgr = self._mgr(total=1000, system_reserve=100, tool_definitions_reserve=100)
        long_memory = ["x" * 10000]
        system, msgs = mgr.build_messages("S.", "M.", [], memory_results=long_memory)
        assert "Relevant Memory" in system
        # Should be truncated
        assert len(system) < 15000

    def test_learnings_exceed_budget_are_skipped(self) -> None:
        """Learnings that exceed their budget fraction are omitted."""
        mgr = self._mgr(total=500, system_reserve=100, tool_definitions_reserve=100, learnings_fraction=0.01)
        long_learnings = ["x" * 1000]
        system, msgs = mgr.build_messages("S.", "M.", [], learnings=long_learnings)
        assert "Past Learnings" not in system

    def test_learnings_within_budget_are_included(self) -> None:
        mgr = self._mgr(total=30000, learnings_fraction=0.1)
        learnings = ["Use pytest -x for faster iteration"]
        system, _ = mgr.build_messages("S.", "M.", [], learnings=learnings)
        assert "Past Learnings" in system
        assert "pytest -x" in system

    def test_learnings_limited_to_five(self) -> None:
        mgr = self._mgr()
        learnings = [f"Learning {i}" for i in range(10)]
        system, _ = mgr.build_messages("S.", "M.", [], learnings=learnings)
        assert "Learning 4" in system
        assert "Learning 5" not in system

    def test_fresh_tail_count_zero(self) -> None:
        """When fresh_tail_count is 0, all history is evictable."""
        mgr = self._mgr(fresh_tail_count=0, total=50000)
        history = [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        _, msgs = mgr.build_messages("S.", "New", history)
        # Should include some history plus new message
        assert msgs[-1]["content"] == "New"

    def test_history_exceeds_budget_oldest_dropped(self) -> None:
        """When history is too large, oldest entries are dropped.
        Fresh tail is always kept, but evictable entries are pruned."""
        mgr = self._mgr(total=500, system_reserve=50, tool_definitions_reserve=50, fresh_tail_count=2)
        history = [{"role": "user", "content": "x" * 200} for _ in range(10)]
        _, msgs = mgr.build_messages("S.", "New", history)
        # Fresh tail (2 entries) + new message should always be present
        assert msgs[-1]["content"] == "New"
        # Should have fewer than all 10 original entries + new message
        assert len(msgs) < 11

    def test_summaries_included_before_history(self) -> None:
        mgr = self._mgr(total=50000, fresh_tail_count=2)
        summary = SimpleNamespace(
            depth=1,
            descendant_count=5,
            content="Previous conversation about Python.",
            time_range_start="10:00",
            time_range_end="10:30",
        )
        history = [{"role": "user", "content": "recent msg"}]
        _, msgs = mgr.build_messages("S.", "New", history, summaries=[summary])
        # Summary should be first
        assert "Conversation Summary" in msgs[0]["content"]
        assert "Python" in msgs[0]["content"]

    def test_summary_without_time_range(self) -> None:
        from missy.agent.context import _format_summary
        summary = SimpleNamespace(depth=2, descendant_count=10, content="Some summary")
        text = _format_summary(summary)
        assert "depth 2" in text
        assert "10 messages" in text
        assert "covers" not in text

    def test_summary_with_time_range(self) -> None:
        from missy.agent.context import _format_summary
        summary = SimpleNamespace(
            depth=1, descendant_count=3, content="Chat",
            time_range_start="09:00", time_range_end="09:30",
        )
        text = _format_summary(summary)
        assert "covers 09:00 to 09:30" in text

    def test_summary_exceeding_budget_is_dropped(self) -> None:
        mgr = self._mgr(total=500, system_reserve=50, tool_definitions_reserve=50, fresh_tail_count=0)
        summary = SimpleNamespace(
            depth=1, descendant_count=100, content="x" * 5000,
        )
        _, msgs = mgr.build_messages("S.", "New", [], summaries=[summary])
        # Summary too large — should be skipped, only new message
        summary_msgs = [m for m in msgs if "Conversation Summary" in m.get("content", "")]
        assert len(summary_msgs) == 0


# ---------------------------------------------------------------------------
# VisionMemoryBridge recall edge cases
# ---------------------------------------------------------------------------


class TestVisionMemoryRecallEdgeCases:
    def test_recall_filters_non_vision_turns(self) -> None:
        """Only turns with role='vision' are returned."""
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_mem = MagicMock()
        turns = [
            SimpleNamespace(role="user", content="hello", session_id="s1", metadata={}),
            SimpleNamespace(role="vision", content="sky piece found", session_id="s1",
                          metadata={"task_type": "puzzle", "observation": "sky piece found"}),
            SimpleNamespace(role="assistant", content="ok", session_id="s1", metadata={}),
        ]
        mock_mem.get_session_turns.return_value = turns

        bridge = VisionMemoryBridge(memory_store=mock_mem)
        results = bridge.recall_observations(session_id="s1")
        assert len(results) == 1
        assert results[0]["observation"] == "sky piece found"

    def test_recall_task_type_filter(self) -> None:
        """task_type filter applied correctly."""
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_mem = MagicMock()
        turns = [
            SimpleNamespace(role="vision", content="a", session_id="s1",
                          metadata={"task_type": "puzzle"}),
            SimpleNamespace(role="vision", content="b", session_id="s1",
                          metadata={"task_type": "painting"}),
        ]
        mock_mem.get_session_turns.return_value = turns

        bridge = VisionMemoryBridge(memory_store=mock_mem)
        results = bridge.recall_observations(session_id="s1", task_type="painting")
        assert len(results) == 1
        assert results[0]["observation"] == "b"

    def test_recall_with_query_no_vector(self) -> None:
        """Query search uses SQLite when vector store is None."""
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_mem = MagicMock()
        turn = SimpleNamespace(role="vision", content="found edge", session_id="s1",
                              metadata={"task_type": "puzzle", "observation": "found edge"})
        mock_mem.search.return_value = [turn]

        bridge = VisionMemoryBridge(memory_store=mock_mem, vector_store=None)
        results = bridge.recall_observations(query="edge pieces")
        assert len(results) == 1

    def test_recall_no_stores_returns_empty(self) -> None:
        """When no stores are initialized and init fails, returns empty."""
        from missy.vision.vision_memory import VisionMemoryBridge

        bridge = VisionMemoryBridge()
        bridge._initialized = True
        bridge._memory = None
        bridge._vector = None
        results = bridge.recall_observations(query="test")
        assert results == []

    def test_recall_respects_limit(self) -> None:
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_mem = MagicMock()
        turns = [
            SimpleNamespace(role="vision", content=f"obs{i}", session_id="s1",
                          metadata={"task_type": "general", "observation": f"obs{i}"})
            for i in range(20)
        ]
        mock_mem.get_session_turns.return_value = turns

        bridge = VisionMemoryBridge(memory_store=mock_mem)
        results = bridge.recall_observations(session_id="s1", limit=3)
        assert len(results) == 3

    def test_store_with_none_metadata(self) -> None:
        """metadata=None should not crash."""
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_mem = MagicMock()
        bridge = VisionMemoryBridge(memory_store=mock_mem)
        obs_id = bridge.store_observation(
            session_id="s1", task_type="general", observation="test", metadata=None
        )
        assert obs_id
        mock_mem.add_turn.assert_called_once()

    def test_store_confidence_and_source_in_metadata(self) -> None:
        """Confidence and source are stored in the metadata dict."""
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_mem = MagicMock()
        bridge = VisionMemoryBridge(memory_store=mock_mem)
        bridge.store_observation(
            session_id="s1", task_type="puzzle", observation="test",
            confidence=0.92, source="webcam:/dev/video0", frame_id=3,
        )

        call_args = mock_mem.add_turn.call_args
        meta = call_args.kwargs.get("metadata") or call_args[1].get("metadata")
        assert meta["confidence"] == 0.92
        assert meta["source"] == "webcam:/dev/video0"
        assert meta["frame_id"] == 3

    def test_clear_session_no_vision_turns(self) -> None:
        """Clearing a session with no vision turns returns 0."""
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_mem = MagicMock()
        mock_mem.get_session_turns.return_value = [
            SimpleNamespace(role="user", id="t1"),
            SimpleNamespace(role="assistant", id="t2"),
        ]
        bridge = VisionMemoryBridge(memory_store=mock_mem)
        assert bridge.clear_session("s1") == 0

    def test_clear_session_get_turns_fails(self) -> None:
        """If get_session_turns raises, returns 0."""
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_mem = MagicMock()
        mock_mem.get_session_turns.side_effect = RuntimeError("DB locked")
        bridge = VisionMemoryBridge(memory_store=mock_mem)
        assert bridge.clear_session("s1") == 0


# ---------------------------------------------------------------------------
# Session context formatting
# ---------------------------------------------------------------------------


class TestSessionContextFormatting:
    def test_multiple_observations(self) -> None:
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_mem = MagicMock()
        turns = [
            SimpleNamespace(role="vision", content=f"obs{i}", session_id="s1",
                          metadata={"task_type": "puzzle", "observation": f"obs{i}",
                                   "confidence": 0.8, "timestamp": f"2026-03-19T{10+i}:00:00"})
            for i in range(3)
        ]
        mock_mem.get_session_turns.return_value = turns

        bridge = VisionMemoryBridge(memory_store=mock_mem)
        ctx = bridge.get_session_context("s1")
        assert ctx.count("[puzzle]") == 3
        assert "obs0" in ctx
        assert "obs2" in ctx

    def test_context_with_missing_metadata_fields(self) -> None:
        """Observations with incomplete metadata still format correctly."""
        from missy.vision.vision_memory import VisionMemoryBridge

        mock_mem = MagicMock()
        turn = SimpleNamespace(
            role="vision", content="partial", session_id="s1",
            metadata={}  # no task_type, confidence, timestamp
        )
        mock_mem.get_session_turns.return_value = [turn]

        bridge = VisionMemoryBridge(memory_store=mock_mem)
        ctx = bridge.get_session_context("s1")
        assert "[general]" in ctx  # default task_type
        assert "partial" in ctx
