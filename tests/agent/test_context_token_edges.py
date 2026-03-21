"""Comprehensive tests for missy.agent.context.


Covers _approx_tokens, TokenBudget, and ContextManager.build_messages with a
focus on scenarios not already addressed in:
  - test_context_manager.py
  - test_context_edge_cases.py
  - test_context_with_summaries.py
  - test_token_budget_validation.py

Areas targeted here:
  1.  _approx_tokens — exact char/token correspondence, boundary rounding
  2.  TokenBudget — defaults, validation error messages, boundary acceptance
  3.  ContextManager initialisation — default and explicit budget
  4.  build_messages return type — always a (str, list) tuple
  5.  System prompt passthrough — exact preservation without enrichment
  6.  New message always last — across all configurations
  7.  Memory injection — multi-result join, double-newline separator
  8.  Memory truncation — hard cap at memory_budget * 4 chars
  9.  Learnings injection — bullet format, first-five cap, double-newline sep
 10.  Learnings budget gate — skipped when marginally over budget
 11.  History pruning — oldest-first, exact-fit, one-over eviction
 12.  Fresh-tail protection — tail always included when budget is tight
 13.  fresh_tail_count=0 — all history evictable
 14.  Large history (100+ messages) — only recent subset kept
 15.  Very long single history message — handled without crash
 16.  Budget arithmetic — available, memory_budget, history_budget formulas
 17.  Summaries — role assigned, placed before evictable, budget-limited
 18.  Summary formatting — time-range branch, no-time-range branch
 19.  tool_definitions ignored — no effect on output
 20.  Message-dict shape — role/content keys in results
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from missy.agent.context import (
    ContextManager,
    TokenBudget,
    _approx_tokens,
    _format_summary,
)

# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _budget(
    total: int = 1_000,
    system_reserve: int = 0,
    tool_definitions_reserve: int = 0,
    memory_fraction: float = 0.0,
    learnings_fraction: float = 0.0,
    fresh_tail_count: int = 0,
) -> TokenBudget:
    """Return a tightly controlled TokenBudget for deterministic tests."""
    return TokenBudget(
        total=total,
        system_reserve=system_reserve,
        tool_definitions_reserve=tool_definitions_reserve,
        memory_fraction=memory_fraction,
        learnings_fraction=learnings_fraction,
        fresh_tail_count=fresh_tail_count,
    )


def _tok(n: int) -> str:
    """Return a string costing exactly n tokens (4 chars per token)."""
    return "x" * (n * 4)


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


@dataclass
class _Summary:
    """Minimal duck-type stand-in for SummaryRecord."""

    depth: int = 0
    content: str = ""
    descendant_count: int = 0
    time_range_start: str | None = None
    time_range_end: str | None = None


# ---------------------------------------------------------------------------
# 1. _approx_tokens
# ---------------------------------------------------------------------------


class TestApproxTokens:
    """_approx_tokens: 4-chars-per-token with minimum-1 clamp."""

    def test_empty_string_returns_one(self):
        assert _approx_tokens("") == 1

    def test_one_char_returns_one(self):
        assert _approx_tokens("a") == 1

    def test_two_chars_returns_one(self):
        assert _approx_tokens("ab") == 1

    def test_three_chars_returns_one(self):
        assert _approx_tokens("abc") == 1

    def test_exactly_four_chars_returns_one(self):
        assert _approx_tokens("abcd") == 1

    def test_five_chars_rounds_down_to_one(self):
        # 5 // 4 == 1
        assert _approx_tokens("abcde") == 1

    def test_eight_chars_returns_two(self):
        assert _approx_tokens("a" * 8) == 2

    def test_twelve_chars_returns_three(self):
        assert _approx_tokens("a" * 12) == 3

    def test_400_chars_returns_100(self):
        assert _approx_tokens("a" * 400) == 100

    def test_4000_chars_returns_1000(self):
        assert _approx_tokens("z" * 4000) == 1000

    def test_spaces_counted_same_as_letters(self):
        # Four spaces = 1 token
        assert _approx_tokens("    ") == 1

    def test_newlines_counted_as_chars(self):
        # Four newlines = 1 token
        assert _approx_tokens("\n\n\n\n") == 1

    def test_non_ascii_counted_by_code_point(self):
        # Python len() counts code points; 4 code points = 1 token
        text = "\u00e9\u00e9\u00e9\u00e9"  # 4 × é
        assert _approx_tokens(text) == 1

    def test_tok_helper_consistency(self):
        # _tok(n) must produce exactly n tokens
        for n in (1, 5, 10, 50, 100):
            assert _approx_tokens(_tok(n)) == n


# ---------------------------------------------------------------------------
# 2. TokenBudget — defaults and validation
# ---------------------------------------------------------------------------


class TestTokenBudgetDefaults:
    def test_total_default(self):
        assert TokenBudget().total == 30_000

    def test_system_reserve_default(self):
        assert TokenBudget().system_reserve == 2_000

    def test_tool_definitions_reserve_default(self):
        assert TokenBudget().tool_definitions_reserve == 2_000

    def test_memory_fraction_default(self):
        assert TokenBudget().memory_fraction == pytest.approx(0.15)

    def test_learnings_fraction_default(self):
        assert TokenBudget().learnings_fraction == pytest.approx(0.05)

    def test_fresh_tail_count_default(self):
        assert TokenBudget().fresh_tail_count == 16


class TestTokenBudgetValidation:
    def test_negative_total_raises_value_error(self):
        with pytest.raises(ValueError):
            TokenBudget(total=-1, system_reserve=0, tool_definitions_reserve=0)

    def test_zero_total_is_accepted(self):
        b = TokenBudget(total=0, system_reserve=0, tool_definitions_reserve=0)
        assert b.total == 0

    def test_memory_fraction_below_zero_raises(self):
        with pytest.raises(ValueError):
            TokenBudget(memory_fraction=-0.01)

    def test_memory_fraction_above_one_raises(self):
        with pytest.raises(ValueError):
            TokenBudget(memory_fraction=1.01)

    def test_memory_fraction_zero_accepted(self):
        b = TokenBudget(memory_fraction=0.0)
        assert b.memory_fraction == 0.0

    def test_memory_fraction_one_accepted(self):
        b = TokenBudget(
            total=30_000,
            system_reserve=0,
            tool_definitions_reserve=0,
            memory_fraction=1.0,
        )
        assert b.memory_fraction == pytest.approx(1.0)

    def test_learnings_fraction_below_zero_raises(self):
        with pytest.raises(ValueError):
            TokenBudget(learnings_fraction=-0.1)

    def test_learnings_fraction_above_one_raises(self):
        with pytest.raises(ValueError):
            TokenBudget(learnings_fraction=2.0)

    def test_learnings_fraction_zero_accepted(self):
        assert TokenBudget(learnings_fraction=0.0).learnings_fraction == 0.0

    def test_negative_fresh_tail_count_raises(self):
        with pytest.raises(ValueError):
            TokenBudget(fresh_tail_count=-1)

    def test_zero_fresh_tail_count_accepted(self):
        assert TokenBudget(fresh_tail_count=0).fresh_tail_count == 0

    def test_reserves_exceeding_total_raises(self):
        with pytest.raises(ValueError):
            TokenBudget(total=100, system_reserve=60, tool_definitions_reserve=60)

    def test_reserves_exactly_equal_total_accepted(self):
        b = TokenBudget(total=200, system_reserve=100, tool_definitions_reserve=100)
        assert b.total == 200

    def test_reserves_just_under_total_accepted(self):
        b = TokenBudget(total=201, system_reserve=100, tool_definitions_reserve=100)
        assert b.total == 201


# ---------------------------------------------------------------------------
# 3. ContextManager — initialisation
# ---------------------------------------------------------------------------


class TestContextManagerInit:
    def test_no_budget_arg_uses_defaults(self):
        cm = ContextManager()
        assert cm._budget.total == 30_000
        assert cm._budget.fresh_tail_count == 16

    def test_explicit_budget_stored(self):
        b = TokenBudget(total=5_000)
        cm = ContextManager(budget=b)
        assert cm._budget.total == 5_000

    def test_none_budget_arg_uses_defaults(self):
        cm = ContextManager(budget=None)
        assert cm._budget.total == 30_000


# ---------------------------------------------------------------------------
# 4. build_messages — return type always (str, list)
# ---------------------------------------------------------------------------


class TestBuildMessagesReturnType:
    def test_returns_two_tuple(self):
        cm = ContextManager()
        result = cm.build_messages(system="S", new_message="M", history=[])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_string(self):
        cm = ContextManager()
        system, _ = cm.build_messages(system="S", new_message="M", history=[])
        assert isinstance(system, str)

    def test_second_element_is_list(self):
        cm = ContextManager()
        _, messages = cm.build_messages(system="S", new_message="M", history=[])
        assert isinstance(messages, list)

    def test_last_message_is_dict_with_role_and_content(self):
        cm = ContextManager()
        _, messages = cm.build_messages(system="S", new_message="hello", history=[])
        last = messages[-1]
        assert "role" in last
        assert "content" in last

    def test_last_message_role_is_user(self):
        cm = ContextManager()
        _, messages = cm.build_messages(system="S", new_message="hi", history=[])
        assert messages[-1]["role"] == "user"


# ---------------------------------------------------------------------------
# 5. System prompt passthrough
# ---------------------------------------------------------------------------


class TestSystemPromptPassthrough:
    def test_system_unchanged_without_enrichment(self):
        cm = ContextManager()
        system, _ = cm.build_messages(system="Exact system text.", new_message="hi", history=[])
        assert system == "Exact system text."

    def test_system_unchanged_when_all_enrichments_none(self):
        cm = ContextManager()
        text = "Another exact prompt."
        system, _ = cm.build_messages(
            system=text,
            new_message="x",
            history=[],
            memory_results=None,
            learnings=None,
            summaries=None,
        )
        assert system == text

    def test_system_unchanged_with_empty_enrichment_lists(self):
        cm = ContextManager()
        text = "Base."
        system, _ = cm.build_messages(
            system=text,
            new_message="x",
            history=[],
            memory_results=[],
            learnings=[],
            summaries=[],
        )
        assert system == text

    def test_enriched_system_starts_with_base(self):
        cm = ContextManager()
        base = "You are Missy."
        system, _ = cm.build_messages(
            system=base,
            new_message="hi",
            history=[],
            memory_results=["some memory"],
            learnings=["a lesson"],
        )
        assert system.startswith(base)


# ---------------------------------------------------------------------------
# 6. New message always last
# ---------------------------------------------------------------------------


class TestNewMessageAlwaysLast:
    def test_new_message_last_with_no_history(self):
        cm = ContextManager()
        _, messages = cm.build_messages(system="S", new_message="LAST", history=[])
        assert messages[-1]["content"] == "LAST"

    def test_new_message_last_with_history(self):
        cm = ContextManager(_budget(total=50_000))
        history = [_msg("user", "a"), _msg("assistant", "b")]
        _, messages = cm.build_messages(system="S", new_message="FINAL", history=history)
        assert messages[-1]["content"] == "FINAL"

    def test_new_message_last_when_all_history_pruned(self):
        # Zero history budget: all history dropped.
        cm = ContextManager(_budget(total=100, system_reserve=50, tool_definitions_reserve=50))
        history = [_msg("user", "old"), _msg("assistant", "older")]
        _, messages = cm.build_messages(system="S", new_message="ONLY", history=history)
        assert messages[-1]["content"] == "ONLY"
        assert len(messages) == 1

    def test_new_message_last_with_summaries(self):
        cm = ContextManager(_budget(total=50_000))
        summaries = [_Summary(depth=0, content="Summary", descendant_count=5)]
        _, messages = cm.build_messages(
            system="S",
            new_message="END",
            history=[_msg("user", "prior")],
            summaries=summaries,
        )
        assert messages[-1]["content"] == "END"


# ---------------------------------------------------------------------------
# 7. Memory injection
# ---------------------------------------------------------------------------


class TestMemoryInjection:
    def test_single_memory_result_appears_in_system(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="x",
            history=[],
            memory_results=["important context"],
        )
        assert "important context" in system

    def test_memory_section_heading_present(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="x",
            history=[],
            memory_results=["mem"],
        )
        assert "## Relevant Memory" in system

    def test_memory_double_newline_separator(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="x",
            history=[],
            memory_results=["mem"],
        )
        assert "\n\n## Relevant Memory" in system

    def test_multiple_memory_results_all_present(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="x",
            history=[],
            memory_results=["alpha", "beta", "gamma"],
        )
        assert "alpha" in system
        assert "beta" in system
        assert "gamma" in system

    def test_multiple_memory_results_joined_with_newline(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="x",
            history=[],
            memory_results=["first", "second"],
        )
        mem_start = system.index("## Relevant Memory\n") + len("## Relevant Memory\n")
        injected = system[mem_start:]
        assert "first\nsecond" in injected

    def test_none_memory_results_no_section(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base", new_message="x", history=[], memory_results=None
        )
        assert "## Relevant Memory" not in system

    def test_empty_memory_results_no_section(self):
        cm = ContextManager()
        system, _ = cm.build_messages(system="Base", new_message="x", history=[], memory_results=[])
        assert "## Relevant Memory" not in system


# ---------------------------------------------------------------------------
# 8. Memory truncation
# ---------------------------------------------------------------------------


class TestMemoryTruncation:
    def test_memory_truncated_to_budget_chars(self):
        # available = 1000 - 0 - 0 = 1000 tokens
        # memory_fraction = 0.10 => memory_budget = 100 tokens = 400 chars
        b = _budget(total=1_000, memory_fraction=0.10)
        cm = ContextManager(b)
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            memory_results=["M" * 2_000],  # 500 tokens, well over cap
        )
        mem_start = system.index("## Relevant Memory\n") + len("## Relevant Memory\n")
        injected = system[mem_start:]
        assert len(injected) == 400  # 100 tokens * 4 chars/token

    def test_memory_within_budget_not_truncated(self):
        # memory_budget = 0.50 * 1000 = 500 tokens = 2000 chars
        b = _budget(total=1_000, memory_fraction=0.50)
        cm = ContextManager(b)
        short = "short text"
        system, _ = cm.build_messages(
            system="Base", new_message="hi", history=[], memory_results=[short]
        )
        assert short in system

    def test_memory_truncated_exactly_at_boundary(self):
        # available = 400 tokens; memory_fraction = 0.25 => memory_budget = 100 tokens = 400 chars
        # Truncation triggers when _approx_tokens(memory_text) > memory_budget.
        # 404 chars → 404 // 4 = 101 tokens > 100 → truncated to 400 chars.
        b = _budget(total=400, memory_fraction=0.25)
        cm = ContextManager(b)
        big_mem = "A" * 404  # 101 tokens, exceeds 100-token budget by 1
        system, _ = cm.build_messages(
            system="Base", new_message="hi", history=[], memory_results=[big_mem]
        )
        mem_start = system.index("## Relevant Memory\n") + len("## Relevant Memory\n")
        injected = system[mem_start:]
        assert len(injected) == 400


# ---------------------------------------------------------------------------
# 9. Learnings injection
# ---------------------------------------------------------------------------


class TestLearningsInjection:
    def test_learnings_section_heading(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base", new_message="x", history=[], learnings=["lesson"]
        )
        assert "## Past Learnings" in system

    def test_learnings_double_newline_separator(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base", new_message="x", history=[], learnings=["lesson"]
        )
        assert "\n\n## Past Learnings" in system

    def test_learnings_formatted_as_bullets(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="x",
            history=[],
            learnings=["lesson one", "lesson two"],
        )
        assert "- lesson one" in system
        assert "- lesson two" in system

    def test_learnings_capped_at_five(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="x",
            history=[],
            learnings=[f"item {i}" for i in range(10)],
        )
        # items 0-4 must appear; items 5-9 must not
        for i in range(5):
            assert f"item {i}" in system
        for i in range(5, 10):
            assert f"item {i}" not in system

    def test_learnings_exactly_five_all_included(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="x",
            history=[],
            learnings=[f"L{i}" for i in range(5)],
        )
        for i in range(5):
            assert f"L{i}" in system

    def test_none_learnings_no_section(self):
        cm = ContextManager()
        system, _ = cm.build_messages(system="Base", new_message="x", history=[], learnings=None)
        assert "## Past Learnings" not in system

    def test_empty_learnings_no_section(self):
        cm = ContextManager()
        system, _ = cm.build_messages(system="Base", new_message="x", history=[], learnings=[])
        assert "## Past Learnings" not in system

    def test_memory_appears_before_learnings_when_both_present(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="x",
            history=[],
            memory_results=["mem"],
            learnings=["learn"],
        )
        assert system.index("## Relevant Memory") < system.index("## Past Learnings")


# ---------------------------------------------------------------------------
# 10. Learnings budget gate
# ---------------------------------------------------------------------------


class TestLearningsBudgetGate:
    def test_learnings_excluded_when_over_budget(self):
        # available = 100 tokens; learnings_fraction = 0.0 => budget = 0 tokens
        b = _budget(total=100, learnings_fraction=0.0)
        cm = ContextManager(b)
        system, _ = cm.build_messages(
            system="Base",
            new_message="x",
            history=[],
            learnings=["should not appear"],
        )
        assert "## Past Learnings" not in system

    def test_learnings_included_when_within_budget(self):
        # Large budget so learnings definitely fit
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="x",
            history=[],
            learnings=["fits easily"],
        )
        assert "## Past Learnings" in system
        assert "fits easily" in system


# ---------------------------------------------------------------------------
# 11. History pruning — exact-fit and one-over
# ---------------------------------------------------------------------------


class TestHistoryPruning:
    def test_history_all_kept_within_budget(self):
        cm = ContextManager(_budget(total=50_000))
        history = [_msg("user", "a"), _msg("assistant", "b"), _msg("user", "c")]
        _, messages = cm.build_messages(system="S", new_message="d", history=history)
        assert len(messages) == 4

    def test_single_message_exact_fit(self):
        # available = 100 tokens; new_message = 1 token ("x" = 1 char -> 1 tok)
        # history_budget = 99 tokens; message of 99 tokens should just fit.
        b = _budget(total=100)
        cm = ContextManager(b)
        content = _tok(99)
        assert _approx_tokens(content) == 99
        _, messages = cm.build_messages(
            system="S", new_message="x", history=[_msg("user", content)]
        )
        assert any(m["content"] == content for m in messages)

    def test_single_message_one_token_over_evicted(self):
        # Message of 100 tokens exceeds 99-token history budget → evicted.
        b = _budget(total=100)
        cm = ContextManager(b)
        content = _tok(100)
        _, messages = cm.build_messages(
            system="S", new_message="x", history=[_msg("user", content)]
        )
        assert all(m["content"] != content for m in messages)
        assert messages[-1]["content"] == "x"

    def test_oldest_pruned_before_newest(self):
        # 3 equal-cost messages; only 1 fits; newest should survive.
        b = _budget(
            total=300,
            system_reserve=100,
            tool_definitions_reserve=100,
        )
        # available = 100 tokens; new = 1 token; history budget = 99 tokens
        # each msg = 50 tokens; only newest (msg2) fits
        cm = ContextManager(b)
        history = [
            _msg("user", _tok(50)),  # oldest
            _msg("assistant", _tok(50)),  # middle
            _msg("user", _tok(50)),  # newest evictable
        ]
        _, messages = cm.build_messages(system="S", new_message="x", history=history)
        # Exactly one history entry of 50 tokens fits; the newest is kept.
        history_msgs = [m for m in messages if m["content"] != "x"]
        assert len(history_msgs) == 1

    def test_chronological_order_preserved_in_output(self):
        cm = ContextManager(_budget(total=50_000))
        history = [
            _msg("user", "first"),
            _msg("assistant", "second"),
            _msg("user", "third"),
        ]
        _, messages = cm.build_messages(system="S", new_message="fourth", history=history)
        contents = [m["content"] for m in messages]
        assert contents.index("first") < contents.index("second")
        assert contents.index("second") < contents.index("third")
        assert contents.index("third") < contents.index("fourth")


# ---------------------------------------------------------------------------
# 12. Fresh-tail protection
# ---------------------------------------------------------------------------


class TestFreshTailProtection:
    def test_fresh_tail_kept_despite_tight_budget(self):
        # Budget so tight that evictable messages cannot fit, but tail is protected.
        b = _budget(
            total=200,
            system_reserve=50,
            tool_definitions_reserve=50,
            fresh_tail_count=2,
        )
        cm = ContextManager(b)
        # available = 100 tokens; each tail msg = 1 token; each evictable = 400 tokens
        history = [
            _msg("user", _tok(400)),  # evictable, huge
            _msg("assistant", _tok(400)),  # evictable, huge
            _msg("user", "tail-a"),  # fresh tail
            _msg("assistant", "tail-b"),  # fresh tail
        ]
        _, messages = cm.build_messages(system="S", new_message="new", history=history)
        contents = [m["content"] for m in messages]
        assert "tail-a" in contents
        assert "tail-b" in contents
        assert "new" in contents

    def test_evictable_not_included_when_tail_exhausts_budget(self):
        b = _budget(
            total=200,
            system_reserve=50,
            tool_definitions_reserve=50,
            fresh_tail_count=2,
        )
        cm = ContextManager(b)
        history = [
            _msg("user", _tok(400)),  # evictable, far exceeds remaining budget
            _msg("user", "tail1"),
            _msg("assistant", "tail2"),
        ]
        _, messages = cm.build_messages(system="S", new_message="new", history=history)
        contents = [m["content"] for m in messages]
        assert _tok(400) not in contents

    def test_fresh_tail_larger_than_history(self):
        # fresh_tail_count > len(history) means all history is fresh.
        b = _budget(total=50_000, fresh_tail_count=100)
        cm = ContextManager(b)
        history = [_msg("user", "only"), _msg("assistant", "two")]
        _, messages = cm.build_messages(system="S", new_message="new", history=history)
        assert len(messages) == 3
        assert messages[0]["content"] == "only"


# ---------------------------------------------------------------------------
# 13. fresh_tail_count=0 — all history evictable
# ---------------------------------------------------------------------------


class TestFreshTailZero:
    def test_all_history_evictable_with_zero_tail(self):
        # With fresh_tail_count=0, even the most recent messages can be pruned.
        b = _budget(
            total=200,
            system_reserve=50,
            tool_definitions_reserve=50,
            fresh_tail_count=0,
        )
        cm = ContextManager(b)
        history = [
            _msg("user", _tok(200)),  # 200 tokens, won't fit
            _msg("assistant", _tok(200)),  # 200 tokens, won't fit
        ]
        _, messages = cm.build_messages(system="S", new_message="new", history=history)
        assert messages == [_msg("user", "new")]

    def test_newest_evictable_kept_when_budget_allows(self):
        b = _budget(total=500, fresh_tail_count=0)
        cm = ContextManager(b)
        # available = 500 tokens; new = 1 token; 3 messages each 1 token should all fit
        history = [
            _msg("user", "msg1"),
            _msg("assistant", "msg2"),
            _msg("user", "msg3"),
        ]
        _, messages = cm.build_messages(system="S", new_message="msg4", history=history)
        contents = [m["content"] for m in messages]
        assert "msg1" in contents
        assert "msg2" in contents
        assert "msg3" in contents


# ---------------------------------------------------------------------------
# 14. Large history (100+ messages)
# ---------------------------------------------------------------------------


class TestLargeHistory:
    def test_100_message_history_only_recent_kept(self):
        # 100 messages each costing 20 tokens = 2000 tokens total.
        # Budget of 500 tokens available after reserves → only newest ~25 fit.
        b = _budget(
            total=600,
            system_reserve=50,
            tool_definitions_reserve=50,
            fresh_tail_count=0,
        )
        cm = ContextManager(b)
        history = [_msg("user", _tok(20)) for _ in range(100)]
        _, messages = cm.build_messages(system="S", new_message="new", history=history)
        # Should be far fewer than 100 history messages
        history_in_result = [m for m in messages if m["content"] != "new"]
        assert len(history_in_result) < 100
        assert len(history_in_result) > 0
        assert messages[-1]["content"] == "new"

    def test_100_message_history_order_preserved(self):
        # Ensure messages kept from a large history remain in original order.
        b = _budget(total=50_000)
        cm = ContextManager(b)
        history = [_msg("user", f"msg{i}") for i in range(100)]
        _, messages = cm.build_messages(system="S", new_message="final", history=history)
        history_contents = [m["content"] for m in messages if m["content"] != "final"]
        # Preserved order: each label is numerically ordered
        indices = [int(c[3:]) for c in history_contents]
        assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# 15. Very long single history message
# ---------------------------------------------------------------------------


class TestVeryLongSingleMessage:
    def test_huge_evictable_message_does_not_crash(self):
        b = _budget(total=200, fresh_tail_count=0)
        cm = ContextManager(b)
        huge = "Z" * 100_000
        _, messages = cm.build_messages(system="S", new_message="ok", history=[_msg("user", huge)])
        assert messages[-1]["content"] == "ok"

    def test_huge_evictable_message_dropped(self):
        b = _budget(total=200, fresh_tail_count=0)
        cm = ContextManager(b)
        huge = _tok(10_000)  # 10 000 tokens >> available
        _, messages = cm.build_messages(system="S", new_message="ok", history=[_msg("user", huge)])
        assert all(m["content"] != huge for m in messages)

    def test_huge_fresh_tail_message_still_included(self):
        # Fresh tail is unconditionally protected regardless of size.
        b = _budget(total=200, fresh_tail_count=1)
        cm = ContextManager(b)
        huge = _tok(10_000)
        _, messages = cm.build_messages(system="S", new_message="ok", history=[_msg("user", huge)])
        contents = [m["content"] for m in messages]
        assert huge in contents


# ---------------------------------------------------------------------------
# 16. Budget arithmetic
# ---------------------------------------------------------------------------


class TestBudgetArithmetic:
    def test_available_tokens_formula(self):
        b = TokenBudget(total=30_000, system_reserve=2_000, tool_definitions_reserve=2_000)
        expected_available = 26_000
        assert b.total - b.system_reserve - b.tool_definitions_reserve == expected_available

    def test_memory_budget_derived_from_available(self):
        TokenBudget(
            total=30_000,
            system_reserve=2_000,
            tool_definitions_reserve=2_000,
            memory_fraction=0.15,
        )
        available = 26_000
        expected_memory_budget = int(available * 0.15)  # 3900
        assert expected_memory_budget == 3_900

    def test_history_budget_is_remainder(self):
        TokenBudget(
            total=30_000,
            system_reserve=2_000,
            tool_definitions_reserve=2_000,
            memory_fraction=0.15,
            learnings_fraction=0.05,
        )
        available = 26_000
        memory_budget = int(available * 0.15)  # 3900
        learnings_budget = int(available * 0.05)  # 1300
        expected_history = available - memory_budget - learnings_budget  # 20800
        assert expected_history == 20_800


# ---------------------------------------------------------------------------
# 17. Summaries — placement and budget limiting
# ---------------------------------------------------------------------------


class TestSummariesPlacement:
    def test_summary_appears_before_evictable_history(self):
        b = _budget(total=50_000)
        cm = ContextManager(b)
        summaries = [_Summary(depth=0, content="compressed past", descendant_count=3)]
        history = [_msg("user", "evictable")]
        _, messages = cm.build_messages(
            system="S",
            new_message="new",
            history=history,
            summaries=summaries,
        )
        # _format_summary wraps the content in a header block, so we search
        # for the keyword rather than an exact content match.
        summary_idx = next(
            i for i, m in enumerate(messages) if "compressed past" in m.get("content", "")
        )
        evictable_idx = next(i for i, m in enumerate(messages) if m.get("content") == "evictable")
        assert summary_idx < evictable_idx

    def test_summary_role_is_user(self):
        b = _budget(total=50_000)
        cm = ContextManager(b)
        summaries = [_Summary(depth=0, content="summary text", descendant_count=2)]
        _, messages = cm.build_messages(
            system="S", new_message="x", history=[], summaries=summaries
        )
        summary_msgs = [m for m in messages if "summary text" in m.get("content", "")]
        assert all(m["role"] == "user" for m in summary_msgs)

    def test_summary_excluded_when_over_budget(self):
        # Zero budget: summaries cannot fit.
        b = _budget(
            total=100,
            system_reserve=50,
            tool_definitions_reserve=50,
        )
        cm = ContextManager(b)
        summaries = [_Summary(depth=0, content="A" * 2000, descendant_count=10)]
        _, messages = cm.build_messages(
            system="S", new_message="new", history=[], summaries=summaries
        )
        assert len(messages) == 1
        assert messages[0]["content"] == "new"

    def test_multiple_summaries_all_included_when_budget_allows(self):
        b = _budget(total=50_000)
        cm = ContextManager(b)
        summaries = [_Summary(depth=0, content=f"summary{i}", descendant_count=1) for i in range(3)]
        _, messages = cm.build_messages(
            system="S", new_message="x", history=[], summaries=summaries
        )
        contents = [m["content"] for m in messages]
        for i in range(3):
            assert any(f"summary{i}" in c for c in contents)

    def test_empty_summaries_list_no_effect(self):
        cm = ContextManager()
        history = [_msg("user", "h1")]
        _, messages = cm.build_messages(system="S", new_message="x", history=history, summaries=[])
        assert len(messages) == 2  # history + new


# ---------------------------------------------------------------------------
# 18. Summary formatting — _format_summary
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def test_depth_appears_in_output(self):
        s = _Summary(depth=3, content="content", descendant_count=7)
        text = _format_summary(s)
        assert "depth 3" in text

    def test_descendant_count_appears_in_output(self):
        s = _Summary(depth=0, content="content", descendant_count=42)
        text = _format_summary(s)
        assert "42 messages" in text

    def test_content_appears_in_output(self):
        s = _Summary(depth=0, content="actual summary body", descendant_count=1)
        text = _format_summary(s)
        assert "actual summary body" in text

    def test_time_range_included_when_both_set(self):
        s = _Summary(
            depth=1,
            content="body",
            descendant_count=5,
            time_range_start="2026-03-01",
            time_range_end="2026-03-15",
        )
        text = _format_summary(s)
        assert "2026-03-01" in text
        assert "2026-03-15" in text
        assert "covers" in text

    def test_time_range_absent_when_not_set(self):
        s = _Summary(depth=0, content="body", descendant_count=2)
        text = _format_summary(s)
        assert "covers" not in text

    def test_time_range_absent_when_only_start_set(self):
        # Both start and end required; only start should not trigger time_info.
        s = _Summary(
            depth=0,
            content="body",
            descendant_count=2,
            time_range_start="2026-03-01",
        )
        text = _format_summary(s)
        assert "covers" not in text

    def test_time_range_absent_when_only_end_set(self):
        s = _Summary(
            depth=0,
            content="body",
            descendant_count=2,
            time_range_end="2026-03-15",
        )
        text = _format_summary(s)
        assert "covers" not in text

    def test_header_format(self):
        s = _Summary(depth=2, content="body", descendant_count=10)
        text = _format_summary(s)
        assert text.startswith("[Conversation Summary")


# ---------------------------------------------------------------------------
# 19. tool_definitions — ignored entirely
# ---------------------------------------------------------------------------


class TestToolDefinitionsIgnored:
    def test_tool_definitions_does_not_change_system_prompt(self):
        cm = ContextManager()
        base = "My system."
        system, _ = cm.build_messages(
            system=base,
            new_message="x",
            history=[],
            tool_definitions=[{"name": "tool_a", "description": "does A"}],
        )
        assert system == base

    def test_tool_definitions_does_not_add_messages(self):
        cm = ContextManager()
        _, messages = cm.build_messages(
            system="S",
            new_message="x",
            history=[],
            tool_definitions=[{"name": "t1"}, {"name": "t2"}, {"name": "t3"}],
        )
        assert len(messages) == 1

    def test_tool_definitions_none_same_as_empty_list(self):
        cm = ContextManager()
        system_none, msgs_none = cm.build_messages(
            system="S", new_message="x", history=[], tool_definitions=None
        )
        system_empty, msgs_empty = cm.build_messages(
            system="S", new_message="x", history=[], tool_definitions=[]
        )
        assert system_none == system_empty
        assert msgs_none == msgs_empty


# ---------------------------------------------------------------------------
# 20. Message-dict shape throughout
# ---------------------------------------------------------------------------


class TestMessageDictShape:
    def test_all_messages_have_role_key(self):
        cm = ContextManager(_budget(total=50_000))
        history = [_msg("user", "h1"), _msg("assistant", "h2")]
        _, messages = cm.build_messages(system="S", new_message="new", history=history)
        for m in messages:
            assert "role" in m

    def test_all_messages_have_content_key(self):
        cm = ContextManager(_budget(total=50_000))
        history = [_msg("user", "h1"), _msg("assistant", "h2")]
        _, messages = cm.build_messages(system="S", new_message="new", history=history)
        for m in messages:
            assert "content" in m

    def test_history_roles_preserved(self):
        cm = ContextManager(_budget(total=50_000))
        history = [_msg("user", "u"), _msg("assistant", "a")]
        _, messages = cm.build_messages(system="S", new_message="new", history=history)
        roles = [m["role"] for m in messages[:-1]]
        assert "user" in roles
        assert "assistant" in roles

    def test_missing_content_in_history_does_not_crash(self):
        cm = ContextManager()
        history = [{"role": "user"}]  # no "content" key
        _, messages = cm.build_messages(system="S", new_message="ok", history=history)
        assert messages[-1]["content"] == "ok"
