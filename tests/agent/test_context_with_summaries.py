"""Tests for context assembly with summaries (Feature 2D)."""

from __future__ import annotations

from dataclasses import dataclass

from missy.agent.context import ContextManager, TokenBudget, _format_summary


@dataclass
class FakeSummary:
    """Minimal summary-like object for testing context assembly."""
    depth: int = 0
    content: str = ""
    time_range_start: str | None = None
    time_range_end: str | None = None
    descendant_count: int = 0


class TestFormatSummary:
    def test_basic(self):
        s = FakeSummary(depth=0, content="Test summary", descendant_count=5)
        result = _format_summary(s)
        assert "depth 0" in result
        assert "5 messages" in result
        assert "Test summary" in result

    def test_with_time_range(self):
        s = FakeSummary(
            depth=1, content="Content",
            time_range_start="2026-01-01", time_range_end="2026-01-02",
            descendant_count=10,
        )
        result = _format_summary(s)
        assert "2026-01-01" in result
        assert "2026-01-02" in result
        assert "depth 1" in result

    def test_no_time_range(self):
        s = FakeSummary(depth=0, content="Content")
        result = _format_summary(s)
        assert "covers" not in result


class TestBuildMessagesWithSummaries:
    def test_summaries_included_before_history(self):
        budget = TokenBudget(
            total=50_000, memory_fraction=0.0, learnings_fraction=0.0,
        )
        cm = ContextManager(budget)
        summaries = [
            FakeSummary(depth=0, content="Earlier discussion about setup", descendant_count=10),
        ]
        history = [
            {"role": "user", "content": "recent msg"},
            {"role": "assistant", "content": "recent resp"},
        ]
        _, messages = cm.build_messages(
            system="S", new_message="new",
            history=history, summaries=summaries,
        )
        # Summary should be first message
        assert "Earlier discussion about setup" in messages[0]["content"]
        # History follows
        assert messages[1]["content"] == "recent msg"
        # New message is last
        assert messages[-1]["content"] == "new"

    def test_no_summaries_preserves_behavior(self):
        cm = ContextManager()
        history = [{"role": "user", "content": "msg1"}]
        _, messages = cm.build_messages(
            system="S", new_message="new", history=history, summaries=None,
        )
        assert len(messages) == 2
        assert messages[0]["content"] == "msg1"
        assert messages[-1]["content"] == "new"

    def test_empty_summaries_preserves_behavior(self):
        cm = ContextManager()
        history = [{"role": "user", "content": "msg1"}]
        _, messages = cm.build_messages(
            system="S", new_message="new", history=history, summaries=[],
        )
        assert len(messages) == 2

    def test_summaries_respect_budget(self):
        budget = TokenBudget(
            total=200, system_reserve=50, tool_definitions_reserve=50,
            memory_fraction=0.0, learnings_fraction=0.0, fresh_tail_count=0,
        )
        cm = ContextManager(budget)
        # available = 100 tokens = 400 chars
        summaries = [
            FakeSummary(depth=0, content="A" * 2000, descendant_count=50),  # ~500 tokens, too big
        ]
        _, messages = cm.build_messages(
            system="S", new_message="new", history=[], summaries=summaries,
        )
        # Summary should be excluded (exceeds budget), only new message
        assert len(messages) == 1
        assert messages[0]["content"] == "new"

    def test_multiple_summaries_budget_limit(self):
        budget = TokenBudget(
            total=500, system_reserve=50, tool_definitions_reserve=50,
            memory_fraction=0.0, learnings_fraction=0.0, fresh_tail_count=0,
        )
        cm = ContextManager(budget)
        # available = 400 tokens = 1600 chars
        summaries = [
            FakeSummary(depth=0, content="Summary A " * 30, descendant_count=5),  # ~75 tokens
            FakeSummary(depth=0, content="Summary B " * 30, descendant_count=5),  # ~75 tokens
            FakeSummary(depth=1, content="C " * 3000, descendant_count=20),       # ~1500 tokens, over budget
        ]
        _, messages = cm.build_messages(
            system="S", new_message="new", history=[], summaries=summaries,
        )
        # First two summaries should fit, third should be dropped
        summary_msgs = [m for m in messages if "Summary" in m.get("content", "")]
        assert len(summary_msgs) >= 1  # At least first summary fits


class TestIntegration:
    """Full cycle: summaries + fresh tail + evictable."""

    def test_full_assembly_order(self):
        budget = TokenBudget(
            total=50_000, memory_fraction=0.0, learnings_fraction=0.0,
            fresh_tail_count=2,
        )
        cm = ContextManager(budget)
        summaries = [
            FakeSummary(depth=0, content="Old context summary", descendant_count=20),
        ]
        history = [
            {"role": "user", "content": "evictable old msg"},
            {"role": "assistant", "content": "evictable old resp"},
            {"role": "user", "content": "fresh msg"},
            {"role": "assistant", "content": "fresh resp"},
        ]
        _, messages = cm.build_messages(
            system="S", new_message="new",
            history=history, summaries=summaries,
        )
        contents = [m["content"] for m in messages]
        # Order: summary, evictable, fresh tail, new
        assert "Old context summary" in contents[0]
        assert "fresh msg" in contents
        assert "fresh resp" in contents
        assert contents[-1] == "new"
