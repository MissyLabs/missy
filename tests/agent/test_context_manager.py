"""Tests for missy.agent.context — context window management with token budget."""

from __future__ import annotations

import pytest

from missy.agent.context import ContextManager, TokenBudget, _approx_tokens


class TestApproxTokens:
    def test_empty_string(self):
        assert _approx_tokens("") == 1  # minimum 1

    def test_short_string(self):
        assert _approx_tokens("hi") == 1  # 2 chars // 4 = 0, clamped to 1

    def test_typical_string(self):
        assert _approx_tokens("hello world test") == 4  # 16 chars // 4

    def test_long_string(self):
        text = "a" * 400
        assert _approx_tokens(text) == 100


class TestTokenBudget:
    def test_defaults(self):
        b = TokenBudget()
        assert b.total == 30_000
        assert b.system_reserve == 2_000
        assert b.tool_definitions_reserve == 2_000
        assert b.memory_fraction == pytest.approx(0.15)
        assert b.learnings_fraction == pytest.approx(0.05)

    def test_custom(self):
        b = TokenBudget(total=10_000, system_reserve=500, memory_fraction=0.2)
        assert b.total == 10_000
        assert b.system_reserve == 500
        assert b.memory_fraction == pytest.approx(0.2)


class TestContextManagerInit:
    def test_default_budget(self):
        cm = ContextManager()
        assert cm._budget.total == 30_000

    def test_custom_budget(self):
        budget = TokenBudget(total=5000)
        cm = ContextManager(budget=budget)
        assert cm._budget.total == 5000


class TestBuildMessages:
    def test_simple_message_no_history(self):
        cm = ContextManager()
        system, messages = cm.build_messages(
            system="You are helpful.",
            new_message="Hello",
            history=[],
        )
        assert system == "You are helpful."
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}

    def test_history_preserved_when_within_budget(self):
        cm = ContextManager(TokenBudget(total=50_000))
        history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "msg2"},
        ]
        system, messages = cm.build_messages(
            system="System",
            new_message="msg3",
            history=history,
        )
        assert len(messages) == 4  # 3 history + 1 new
        assert messages[-1]["content"] == "msg3"

    def test_oldest_history_pruned_first(self):
        # Use a tiny budget so only a few messages fit.
        # fresh_tail_count=0 disables fresh-tail protection to test pure pruning.
        budget = TokenBudget(
            total=200,
            system_reserve=50,
            tool_definitions_reserve=50,
            memory_fraction=0.0,
            learnings_fraction=0.0,
            fresh_tail_count=0,
        )
        cm = ContextManager(budget=budget)
        # available = 200 - 50 - 50 = 100 tokens = ~400 chars
        history = [
            {"role": "user", "content": "A" * 200},  # ~50 tokens
            {"role": "assistant", "content": "B" * 200},  # ~50 tokens
            {"role": "user", "content": "C" * 40},  # ~10 tokens
        ]
        system, messages = cm.build_messages(
            system="S",
            new_message="new",
            history=history,
        )
        # The oldest message(s) should be pruned
        assert messages[-1]["content"] == "new"
        # Should have kept some but not all history
        assert len(messages) < 4

    def test_memory_injected_into_system(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base system.",
            new_message="Hello",
            history=[],
            memory_results=["User prefers Python.", "User works on security."],
        )
        assert "## Relevant Memory" in system
        assert "User prefers Python." in system
        assert "User works on security." in system

    def test_memory_truncated_when_over_budget(self):
        budget = TokenBudget(
            total=200,
            system_reserve=50,
            tool_definitions_reserve=50,
            memory_fraction=0.1,  # ~10 tokens = 40 chars of memory allowed
        )
        cm = ContextManager(budget=budget)
        long_memory = ["A" * 500]  # Way over budget
        system, _ = cm.build_messages(
            system="Base",
            new_message="Hi",
            history=[],
            memory_results=long_memory,
        )
        assert "## Relevant Memory" in system
        # The memory text should be truncated
        memory_section = system.split("## Relevant Memory\n")[1]
        assert len(memory_section) < 500

    def test_learnings_injected(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="Hi",
            history=[],
            learnings=["Always verify paths.", "Check permissions first."],
        )
        assert "## Past Learnings" in system
        assert "- Always verify paths." in system
        assert "- Check permissions first." in system

    def test_learnings_capped_at_five(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="Hi",
            history=[],
            learnings=[f"Learning {i}" for i in range(10)],
        )
        # Only first 5 should appear
        assert "Learning 4" in system
        assert "Learning 5" not in system

    def test_learnings_skipped_when_over_budget(self):
        budget = TokenBudget(
            total=200,
            system_reserve=50,
            tool_definitions_reserve=50,
            learnings_fraction=0.001,  # ~0.1 tokens = almost nothing
        )
        cm = ContextManager(budget=budget)
        system, _ = cm.build_messages(
            system="Base",
            new_message="Hi",
            history=[],
            learnings=["A long learning that exceeds the tiny budget " * 10],
        )
        assert "## Past Learnings" not in system

    def test_no_memory_no_learnings(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base prompt",
            new_message="Hi",
            history=[],
            memory_results=None,
            learnings=None,
        )
        assert system == "Base prompt"

    def test_empty_memory_list(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="Hi",
            history=[],
            memory_results=[],
        )
        assert "## Relevant Memory" not in system

    def test_empty_learnings_list(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="Hi",
            history=[],
            learnings=[],
        )
        assert "## Past Learnings" not in system

    def test_tool_definitions_ignored(self):
        cm = ContextManager()
        system, messages = cm.build_messages(
            system="Base",
            new_message="Hi",
            history=[],
            tool_definitions=[{"name": "tool1"}, {"name": "tool2"}],
        )
        assert system == "Base"
        assert len(messages) == 1

    def test_history_with_missing_content(self):
        cm = ContextManager()
        history = [
            {"role": "user"},  # no content key
            {"role": "assistant", "content": "response"},
        ]
        system, messages = cm.build_messages(
            system="Base",
            new_message="Hi",
            history=history,
        )
        # Should not crash; get("content", "") handles missing key
        assert len(messages) >= 1

    def test_combined_memory_and_learnings(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="Hi",
            history=[],
            memory_results=["Memory item"],
            learnings=["Learning item"],
        )
        assert "## Relevant Memory" in system
        assert "## Past Learnings" in system
        # Memory comes before learnings
        mem_pos = system.index("## Relevant Memory")
        learn_pos = system.index("## Past Learnings")
        assert mem_pos < learn_pos

    def test_new_message_always_included(self):
        """Even with zero budget, the new message should be in the output."""
        budget = TokenBudget(
            total=100,
            system_reserve=50,
            tool_definitions_reserve=50,
        )
        cm = ContextManager(budget=budget)
        _, messages = cm.build_messages(
            system="S",
            new_message="This is the new message",
            history=[{"role": "user", "content": "old msg"}],
        )
        assert messages[-1]["content"] == "This is the new message"


class TestFreshTailProtection:
    """Tests for fresh-tail protection (Feature 1)."""

    def test_fresh_tail_preserved_under_tight_budget(self):
        """Fresh tail messages survive even when budget is tight."""
        budget = TokenBudget(
            total=200,
            system_reserve=50,
            tool_definitions_reserve=50,
            memory_fraction=0.0,
            learnings_fraction=0.0,
            fresh_tail_count=2,
        )
        cm = ContextManager(budget=budget)
        history = [
            {"role": "user", "content": "A" * 400},  # old, ~100 tokens
            {"role": "assistant", "content": "B" * 400},  # old, ~100 tokens
            {"role": "user", "content": "C" * 40},  # fresh tail
            {"role": "assistant", "content": "D" * 40},  # fresh tail
        ]
        _, messages = cm.build_messages(
            system="S",
            new_message="new",
            history=history,
        )
        contents = [m["content"] for m in messages]
        # Both fresh tail entries must be present
        assert "C" * 40 in contents
        assert "D" * 40 in contents
        assert "new" in contents
        # Old evictable entries should be pruned (budget too tight)
        assert "A" * 400 not in contents

    def test_evictable_prefix_pruned_correctly(self):
        """Evictable prefix fills remaining budget newest-first."""
        budget = TokenBudget(
            total=500,
            system_reserve=50,
            tool_definitions_reserve=50,
            memory_fraction=0.0,
            learnings_fraction=0.0,
            fresh_tail_count=1,
        )
        cm = ContextManager(budget=budget)
        # available = 400 tokens = 1600 chars
        history = [
            {"role": "user", "content": "old1-" + "x" * 395},  # ~100 tokens
            {"role": "assistant", "content": "old2-" + "y" * 395},  # ~100 tokens
            {"role": "user", "content": "old3-" + "z" * 395},  # ~100 tokens
            {"role": "assistant", "content": "tail"},  # fresh tail, ~2 tokens
        ]
        _, messages = cm.build_messages(
            system="S",
            new_message="new",
            history=history,
        )
        contents = [m["content"] for m in messages]
        # Fresh tail always present
        assert "tail" in contents
        assert "new" in contents
        # old3 is newest evictable, should be included if budget allows
        assert any("old3-" in c for c in contents)

    def test_huge_message_in_evictable_doesnt_kill_fresh_tail(self):
        """A single massive evictable message doesn't block fresh tail."""
        budget = TokenBudget(
            total=200,
            system_reserve=50,
            tool_definitions_reserve=50,
            memory_fraction=0.0,
            learnings_fraction=0.0,
            fresh_tail_count=2,
        )
        cm = ContextManager(budget=budget)
        history = [
            {"role": "user", "content": "X" * 10000},  # huge, evictable
            {"role": "user", "content": "recent1"},  # fresh tail
            {"role": "assistant", "content": "recent2"},  # fresh tail
        ]
        _, messages = cm.build_messages(
            system="S",
            new_message="new",
            history=history,
        )
        contents = [m["content"] for m in messages]
        assert "recent1" in contents
        assert "recent2" in contents
        assert "new" in contents
        # Huge message should be dropped
        assert "X" * 10000 not in contents

    def test_fresh_tail_zero_preserves_old_behavior(self):
        """fresh_tail_count=0 means all history is evictable (backward compat)."""
        budget = TokenBudget(
            total=200,
            system_reserve=50,
            tool_definitions_reserve=50,
            memory_fraction=0.0,
            learnings_fraction=0.0,
            fresh_tail_count=0,
        )
        cm = ContextManager(budget=budget)
        history = [
            {"role": "user", "content": "A" * 200},
            {"role": "assistant", "content": "B" * 200},
            {"role": "user", "content": "C" * 40},
        ]
        _, messages = cm.build_messages(
            system="S",
            new_message="new",
            history=history,
        )
        # With fresh_tail_count=0, everything is evictable and oldest pruned
        assert messages[-1]["content"] == "new"
        assert len(messages) < 4

    def test_fresh_tail_larger_than_history(self):
        """When fresh_tail_count > len(history), all history is fresh."""
        budget = TokenBudget(
            total=200,
            system_reserve=50,
            tool_definitions_reserve=50,
            memory_fraction=0.0,
            learnings_fraction=0.0,
            fresh_tail_count=100,  # way more than 2 history items
        )
        cm = ContextManager(budget=budget)
        history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
        ]
        _, messages = cm.build_messages(
            system="S",
            new_message="new",
            history=history,
        )
        # All history preserved as fresh tail
        assert len(messages) == 3
        assert messages[0]["content"] == "msg1"
        assert messages[-1]["content"] == "new"

    def test_fresh_tail_exceeds_budget_still_included(self):
        """Even when fresh tail alone exceeds budget, it's still included."""
        budget = TokenBudget(
            total=100,
            system_reserve=50,
            tool_definitions_reserve=50,
            memory_fraction=0.0,
            learnings_fraction=0.0,
            fresh_tail_count=3,
        )
        cm = ContextManager(budget=budget)
        # available = 0 tokens, but fresh tail is protected
        history = [
            {"role": "user", "content": "A" * 200},
            {"role": "assistant", "content": "B" * 200},
            {"role": "user", "content": "C" * 200},
        ]
        _, messages = cm.build_messages(
            system="S",
            new_message="new",
            history=history,
        )
        # All 3 fresh tail + new message = 4
        assert len(messages) == 4
        assert messages[-1]["content"] == "new"

    def test_default_fresh_tail_count(self):
        """Default fresh_tail_count is 16."""
        b = TokenBudget()
        assert b.fresh_tail_count == 16
