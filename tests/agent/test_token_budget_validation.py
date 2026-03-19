"""Tests for TokenBudget input validation."""

import pytest

from missy.agent.context import TokenBudget


class TestTokenBudgetValidation:
    """Tests for TokenBudget.__post_init__ validation."""

    def test_negative_total_raises(self):
        with pytest.raises(ValueError, match="total must be >= 0"):
            TokenBudget(total=-1, system_reserve=0, tool_definitions_reserve=0)

    def test_zero_total_accepted(self):
        b = TokenBudget(total=0, system_reserve=0, tool_definitions_reserve=0)
        assert b.total == 0

    def test_memory_fraction_below_zero_raises(self):
        with pytest.raises(ValueError, match="memory_fraction"):
            TokenBudget(memory_fraction=-0.1)

    def test_memory_fraction_above_one_raises(self):
        with pytest.raises(ValueError, match="memory_fraction"):
            TokenBudget(memory_fraction=1.5)

    def test_learnings_fraction_below_zero_raises(self):
        with pytest.raises(ValueError, match="learnings_fraction"):
            TokenBudget(learnings_fraction=-0.01)

    def test_learnings_fraction_above_one_raises(self):
        with pytest.raises(ValueError, match="learnings_fraction"):
            TokenBudget(learnings_fraction=2.0)

    def test_negative_fresh_tail_count_raises(self):
        with pytest.raises(ValueError, match="fresh_tail_count"):
            TokenBudget(fresh_tail_count=-1)

    def test_reserves_exceed_total_raises(self):
        with pytest.raises(ValueError, match="exceeds total"):
            TokenBudget(total=100, system_reserve=80, tool_definitions_reserve=80)

    def test_reserves_equal_total_accepted(self):
        b = TokenBudget(total=100, system_reserve=50, tool_definitions_reserve=50)
        assert b.total == 100

    def test_default_budget_valid(self):
        b = TokenBudget()
        assert b.total == 30_000
        assert b.memory_fraction == 0.15
        assert b.learnings_fraction == 0.05

    def test_boundary_fractions_accepted(self):
        b = TokenBudget(memory_fraction=0.0, learnings_fraction=1.0)
        assert b.memory_fraction == 0.0
        assert b.learnings_fraction == 1.0

    def test_zero_fresh_tail_count_accepted(self):
        b = TokenBudget(fresh_tail_count=0)
        assert b.fresh_tail_count == 0
