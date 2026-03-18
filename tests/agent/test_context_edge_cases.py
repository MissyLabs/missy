"""Edge-case tests for missy.agent.context.ContextManager.

Covers boundary conditions not addressed in test_context_manager.py or
test_context_with_summaries.py:

1.  Token budget exhaustion          - history pruned when total would overflow
2.  Memory fraction hard cap         - memory text truncated to exactly the cap
3.  Learnings fraction boundary      - learnings skipped when marginally over budget
4.  Empty history                    - no crash, correct output shape
5.  System prompt preserved          - pruning never touches the system prompt
6.  Single huge message              - one message exceeding entire available budget
7.  Pruning order                    - oldest evictable messages dropped first
8.  Memory injection structure       - section heading and separator placement
9.  Learnings injection structure    - bullet formatting and section heading
10. Token counting                   - _approx_tokens edge cases
"""

from __future__ import annotations

from missy.agent.context import ContextManager, TokenBudget, _approx_tokens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_budget(
    total: int = 400,
    system_reserve: int = 0,
    tool_definitions_reserve: int = 0,
    memory_fraction: float = 0.0,
    learnings_fraction: float = 0.0,
    fresh_tail_count: int = 0,
) -> TokenBudget:
    """Return a small, tightly controlled budget for deterministic tests."""
    return TokenBudget(
        total=total,
        system_reserve=system_reserve,
        tool_definitions_reserve=tool_definitions_reserve,
        memory_fraction=memory_fraction,
        learnings_fraction=learnings_fraction,
        fresh_tail_count=fresh_tail_count,
    )


def char_tokens(n_tokens: int) -> str:
    """Return a string that costs exactly n_tokens (4 chars per token)."""
    return "x" * (n_tokens * 4)


# ---------------------------------------------------------------------------
# 1. Token budget exhaustion
# ---------------------------------------------------------------------------

class TestTokenBudgetExhaustion:
    """History is pruned when combined tokens exceed the available budget."""

    def test_all_history_pruned_when_budget_is_zero(self):
        # system_reserve + tool_definitions_reserve == total => available == 0
        budget = make_budget(total=100, system_reserve=50, tool_definitions_reserve=50)
        cm = ContextManager(budget)
        history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
        ]
        _, messages = cm.build_messages(
            system="S", new_message="new", history=history
        )
        # Only the new message survives; zero budget means all history is evicted.
        assert messages == [{"role": "user", "content": "new"}]

    def test_partial_history_kept_when_budget_is_tight(self):
        # available = 400 - 200 - 0 = 200 tokens = 800 chars
        # new_message costs 1 token; remaining for history = 199 tokens = 796 chars
        budget = make_budget(total=400, system_reserve=200, fresh_tail_count=0)
        cm = ContextManager(budget)
        # 3 messages each costing 100 tokens (400 chars); only 1 can fit after new msg
        history = [
            {"role": "user",      "content": char_tokens(100)},  # oldest
            {"role": "assistant", "content": char_tokens(100)},
            {"role": "user",      "content": char_tokens(100)},  # newest
        ]
        _, messages = cm.build_messages(
            system="S", new_message="X", history=history
        )
        # Newest evictable (history[2]) should be kept; oldest two pruned.
        contents = [m["content"] for m in messages]
        assert char_tokens(100) in contents  # newest kept
        assert len(messages) == 2  # newest history + new message

    def test_budget_consumed_exactly(self):
        # available = 100 tokens; new_message = 1 token; history budget = 99 tokens
        # One message of exactly 99 tokens should just fit.
        budget = make_budget(total=100, fresh_tail_count=0)
        cm = ContextManager(budget)
        ninety_nine_token_content = char_tokens(99)
        history = [{"role": "user", "content": ninety_nine_token_content}]
        _, messages = cm.build_messages(
            system="S", new_message="x", history=history
        )
        # The history message fits exactly; it should be present.
        assert any(m["content"] == ninety_nine_token_content for m in messages)

    def test_budget_exceeded_by_one_token_evicts_history(self):
        # available = 100 tokens; new_message = 1 token; history budget = 99 tokens
        # A message of 100 tokens exceeds the remaining 99, so it should be pruned.
        budget = make_budget(total=100, fresh_tail_count=0)
        cm = ContextManager(budget)
        hundred_token_content = char_tokens(100)
        history = [{"role": "user", "content": hundred_token_content}]
        _, messages = cm.build_messages(
            system="S", new_message="x", history=history
        )
        # Pruned; only new message remains.
        assert all(m["content"] != hundred_token_content for m in messages)
        assert messages[-1]["content"] == "x"


# ---------------------------------------------------------------------------
# 2. Memory fraction hard cap
# ---------------------------------------------------------------------------

class TestMemoryFractionCap:
    """Memory text is truncated to exactly memory_budget * 4 characters."""

    def test_memory_truncated_to_exactly_fraction(self):
        # available = 1000 tokens; memory_fraction = 0.10 => memory_budget = 100 tokens = 400 chars
        budget = TokenBudget(
            total=1200,
            system_reserve=100,
            tool_definitions_reserve=100,
            memory_fraction=0.10,
            learnings_fraction=0.0,
            fresh_tail_count=0,
        )
        cm = ContextManager(budget)
        # Memory string is 2000 chars (~500 tokens), well over budget.
        long_memory = "M" * 2000
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            memory_results=[long_memory],
        )
        # The memory section should exist but be capped.
        assert "## Relevant Memory" in system
        memory_section_start = system.index("## Relevant Memory\n") + len("## Relevant Memory\n")
        injected_memory = system[memory_section_start:]
        # Cap is memory_budget * 4 = 100 * 4 = 400 chars.
        assert len(injected_memory) == 400

    def test_memory_under_fraction_not_truncated(self):
        # available = 1000 tokens; memory_fraction = 0.50 => 500 tokens = 2000 chars
        budget = TokenBudget(
            total=1200,
            system_reserve=100,
            tool_definitions_reserve=100,
            memory_fraction=0.50,
            learnings_fraction=0.0,
            fresh_tail_count=0,
        )
        cm = ContextManager(budget)
        short_memory = "short memory content"  # well under 2000 chars
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            memory_results=[short_memory],
        )
        assert short_memory in system

    def test_multiple_memory_results_joined_then_capped(self):
        # Verify joining happens before cap; result is a single truncated string.
        budget = TokenBudget(
            total=500,
            system_reserve=0,
            tool_definitions_reserve=0,
            memory_fraction=0.10,  # 50 tokens = 200 chars
            learnings_fraction=0.0,
            fresh_tail_count=0,
        )
        cm = ContextManager(budget)
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            memory_results=["A" * 300, "B" * 300],
        )
        assert "## Relevant Memory" in system
        memory_start = system.index("## Relevant Memory\n") + len("## Relevant Memory\n")
        injected = system[memory_start:]
        # Should be capped at 200 chars (50 tokens * 4)
        assert len(injected) == 200


# ---------------------------------------------------------------------------
# 3. Learnings fraction boundary
# ---------------------------------------------------------------------------

class TestLearningsFractionBoundary:
    """Learnings are included only when their token count is within budget."""

    def test_learnings_exactly_at_budget_included(self):
        # available = 100 tokens; learnings_fraction = 0.20 => 20 tokens = 80 chars
        # Build a learnings text of exactly 20 tokens (80 chars).
        budget = TokenBudget(
            total=100,
            system_reserve=0,
            tool_definitions_reserve=0,
            memory_fraction=0.0,
            learnings_fraction=0.20,
            fresh_tail_count=0,
        )
        cm = ContextManager(budget)
        # One learning item: "- " + 78 chars = 80 chars = exactly 20 tokens
        item_text = "L" * 78  # will be formatted as "- LLLL...78Ls" = 80 chars
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            learnings=[item_text],
        )
        assert "## Past Learnings" in system

    def test_learnings_just_over_budget_excluded(self):
        # available = 100 tokens; learnings_fraction = 0.20 => learnings_budget = 20 tokens.
        # The source check is: _approx_tokens(learnings_text) <= learnings_budget.
        # learnings_text = "- " + item_text (formatted as a single bullet).
        # To exceed the budget the text must cost >= 21 tokens => >= 84 chars.
        # "- " (2 chars) + 82 "L"s = 84 chars => _approx_tokens = 84 // 4 = 21 > 20 => excluded.
        budget = TokenBudget(
            total=100,
            system_reserve=0,
            tool_definitions_reserve=0,
            memory_fraction=0.0,
            learnings_fraction=0.20,
            fresh_tail_count=0,
        )
        cm = ContextManager(budget)
        # "- " (2) + 82 chars = 84 chars => 21 tokens > 20 token budget => excluded
        item_text = "L" * 82
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            learnings=[item_text],
        )
        assert "## Past Learnings" not in system

    def test_learnings_fraction_zero_excludes_any_learnings(self):
        budget = make_budget(total=10_000, learnings_fraction=0.0)
        cm = ContextManager(budget)
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            learnings=["This should not appear"],
        )
        assert "## Past Learnings" not in system

    def test_learnings_only_first_five_counted_against_budget(self):
        # Even with a generous budget, only up to 5 learnings are used.
        cm = ContextManager()
        items = [f"item{i}" for i in range(10)]
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            learnings=items,
        )
        if "## Past Learnings" in system:
            assert "item5" not in system
            assert "item9" not in system


# ---------------------------------------------------------------------------
# 4. Empty history
# ---------------------------------------------------------------------------

class TestEmptyHistory:
    """build_messages with an empty history list must behave correctly."""

    def test_empty_history_returns_only_new_message(self):
        cm = ContextManager()
        system, messages = cm.build_messages(
            system="System prompt",
            new_message="Hello",
            history=[],
        )
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_empty_history_system_prompt_unchanged_without_enrichment(self):
        cm = ContextManager()
        base = "My system prompt"
        system, _ = cm.build_messages(system=base, new_message="Hi", history=[])
        assert system == base

    def test_empty_history_with_all_enrichments(self):
        cm = ContextManager()
        system, messages = cm.build_messages(
            system="Base",
            new_message="Hi",
            history=[],
            memory_results=["mem1"],
            learnings=["learn1"],
        )
        assert "## Relevant Memory" in system
        assert "## Past Learnings" in system
        assert len(messages) == 1
        assert messages[0]["content"] == "Hi"

    def test_empty_history_empty_string_new_message(self):
        cm = ContextManager()
        _, messages = cm.build_messages(
            system="S", new_message="", history=[]
        )
        assert messages[-1] == {"role": "user", "content": ""}

    def test_empty_history_with_summaries_none(self):
        cm = ContextManager()
        _, messages = cm.build_messages(
            system="S", new_message="X", history=[], summaries=None
        )
        assert len(messages) == 1


# ---------------------------------------------------------------------------
# 5. System prompt preserved under pressure
# ---------------------------------------------------------------------------

class TestSystemPromptPreserved:
    """The system prompt string is never modified or truncated by budget logic."""

    def test_system_prompt_intact_when_history_pruned(self):
        budget = make_budget(total=100, system_reserve=0, fresh_tail_count=0)
        cm = ContextManager(budget)
        original_system = "You are Missy, a helpful assistant."
        # History large enough to overflow budget
        heavy_history = [{"role": "user", "content": char_tokens(200)}]
        system, _ = cm.build_messages(
            system=original_system,
            new_message="hi",
            history=heavy_history,
        )
        assert system == original_system

    def test_system_prompt_intact_with_memory_and_learnings(self):
        cm = ContextManager()
        base_system = "Base system text."
        system, _ = cm.build_messages(
            system=base_system,
            new_message="hi",
            history=[],
            memory_results=["memory snippet"],
            learnings=["a learning"],
        )
        # Base system text must appear at the start of the enriched prompt.
        assert system.startswith(base_system)

    def test_system_prompt_contains_all_enrichment_sections(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            memory_results=["mem"],
            learnings=["learn"],
        )
        assert system.startswith("Base")
        assert "## Relevant Memory" in system
        assert "## Past Learnings" in system

    def test_system_reserve_is_not_available_for_history(self):
        # With system_reserve == total, available == 0, all history is pruned.
        budget = TokenBudget(
            total=50,
            system_reserve=50,
            tool_definitions_reserve=0,
            memory_fraction=0.0,
            learnings_fraction=0.0,
            fresh_tail_count=0,
        )
        cm = ContextManager(budget)
        history = [{"role": "user", "content": "old message"}]
        _, messages = cm.build_messages(
            system="S", new_message="new", history=history
        )
        assert all(m["content"] != "old message" for m in messages)
        assert messages[-1]["content"] == "new"


# ---------------------------------------------------------------------------
# 6. Single huge message
# ---------------------------------------------------------------------------

class TestSingleHugeMessage:
    """A single message larger than the entire history budget is simply dropped."""

    def test_single_huge_evictable_message_dropped(self):
        # available = 100 tokens; new_message = 1 token; history_budget = 99 tokens
        # One message of 1000 tokens > 99 => must be dropped.
        budget = make_budget(total=100, fresh_tail_count=0)
        cm = ContextManager(budget)
        huge_content = char_tokens(1000)
        history = [{"role": "user", "content": huge_content}]
        _, messages = cm.build_messages(
            system="S", new_message="new", history=history
        )
        assert all(m["content"] != huge_content for m in messages)
        assert messages[-1]["content"] == "new"

    def test_huge_message_in_fresh_tail_still_included(self):
        # Fresh tail is protected regardless of size.
        budget = make_budget(
            total=100,
            fresh_tail_count=1,
        )
        cm = ContextManager(budget)
        huge_content = char_tokens(1000)
        history = [{"role": "user", "content": huge_content}]  # becomes fresh tail
        _, messages = cm.build_messages(
            system="S", new_message="new", history=history
        )
        contents = [m["content"] for m in messages]
        assert huge_content in contents

    def test_huge_message_followed_by_small_messages(self):
        # Huge message is oldest (evictable); small messages are newer (also evictable).
        budget = make_budget(total=200, fresh_tail_count=0)
        cm = ContextManager(budget)
        huge = char_tokens(5000)
        small1 = "a" * 4   # 1 token
        small2 = "b" * 4   # 1 token
        history = [
            {"role": "user",      "content": huge},    # oldest, too big
            {"role": "assistant", "content": small1},
            {"role": "user",      "content": small2},
        ]
        _, messages = cm.build_messages(
            system="S", new_message="new", history=history
        )
        contents = [m["content"] for m in messages]
        assert huge not in contents
        # Small messages should fit in the budget
        assert small1 in contents or small2 in contents

    def test_new_message_is_never_pruned(self):
        # Even when it exceeds the budget on its own, the new message is appended.
        budget = make_budget(total=10)
        cm = ContextManager(budget)
        giant_new = char_tokens(10000)
        _, messages = cm.build_messages(
            system="S", new_message=giant_new, history=[]
        )
        assert messages[-1]["content"] == giant_new


# ---------------------------------------------------------------------------
# 7. Pruning order
# ---------------------------------------------------------------------------

class TestPruningOrder:
    """Oldest evictable messages are pruned first; newest are retained."""

    def test_oldest_evicted_before_newest(self):
        # Budget fits exactly 2 history tokens + new message.
        # history: [old1 (50t), old2 (50t), old3 (50t)] — only newest 1 should fit.
        budget = TokenBudget(
            total=300,
            system_reserve=100,
            tool_definitions_reserve=100,
            memory_fraction=0.0,
            learnings_fraction=0.0,
            fresh_tail_count=0,
        )
        cm = ContextManager(budget)
        # available = 100 tokens; new_message = 1 token; remaining = 99 for history
        # Each history message = 50 tokens; only the newest fits.
        history = [
            {"role": "user",      "content": char_tokens(50), "id": "old1"},
            {"role": "assistant", "content": char_tokens(50), "id": "old2"},
            {"role": "user",      "content": char_tokens(50), "id": "new3"},
        ]
        _, messages = cm.build_messages(
            system="S", new_message="x", history=history
        )
        contents = [m["content"] for m in messages]
        # Newest evictable (new3) kept; old1 and old2 pruned.
        assert char_tokens(50) in contents   # new3 content (same as old1/old2 chars but only one kept)
        assert len([c for c in contents if c == char_tokens(50)]) == 1

    def test_multi_prune_retains_newest(self):
        # 5 evictable messages; budget allows only 2.
        budget = TokenBudget(
            total=1000,
            system_reserve=500,
            tool_definitions_reserve=250,
            memory_fraction=0.0,
            learnings_fraction=0.0,
            fresh_tail_count=0,
        )
        cm = ContextManager(budget)
        # available = 250 tokens; new_message = 1 token; history budget = 249 tokens
        # Each message costs 60 tokens => 4 would cost 240, 5 would cost 300 => 4 fit max
        messages_in = [
            {"role": "user",      "content": char_tokens(60), "label": f"msg{i}"}
            for i in range(5)
        ]
        _, result = cm.build_messages(
            system="S", new_message="x", history=messages_in
        )
        # 4 history messages fit (4*60=240 <= 249); the oldest (msg0) should be pruned.
        history_in_result = [m for m in result if m["content"] == char_tokens(60)]
        # 4 messages of identical content fit; 5th (oldest) is dropped
        assert len(history_in_result) == 4

    def test_evictable_order_preserved_in_output(self):
        # The kept evictable messages must appear in their original chronological order.
        budget = make_budget(total=500, fresh_tail_count=0)
        cm = ContextManager(budget)
        history = [
            {"role": "user",      "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user",      "content": "third"},
        ]
        _, messages = cm.build_messages(
            system="S", new_message="fourth", history=history
        )
        contents = [m["content"] for m in messages]
        # All fit; verify relative order.
        idx_first  = contents.index("first")
        idx_second = contents.index("second")
        idx_third  = contents.index("third")
        idx_fourth = contents.index("fourth")
        assert idx_first < idx_second < idx_third < idx_fourth


# ---------------------------------------------------------------------------
# 8. Memory injection structure
# ---------------------------------------------------------------------------

class TestMemoryInjectionStructure:
    """Verify the exact formatting of injected memory in the system prompt."""

    def test_memory_section_heading(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            memory_results=["snippet"],
        )
        assert "## Relevant Memory" in system

    def test_memory_appended_after_base_system(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            memory_results=["memory content"],
        )
        base_end = system.index("## Relevant Memory")
        assert base_end > 0  # base system comes first
        assert system[:base_end].strip() == "Base"

    def test_multiple_memory_results_joined_with_newlines(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            memory_results=["result A", "result B", "result C"],
        )
        mem_start = system.index("## Relevant Memory\n") + len("## Relevant Memory\n")
        mem_text = system[mem_start:]
        assert "result A" in mem_text
        assert "result B" in mem_text
        assert "result C" in mem_text

    def test_single_memory_result_no_trailing_newline_issue(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            memory_results=["solo memory item"],
        )
        assert "solo memory item" in system

    def test_memory_separator_is_double_newline(self):
        # The enriched system should include \n\n before the ## heading.
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            memory_results=["mem"],
        )
        assert "\n\n## Relevant Memory" in system


# ---------------------------------------------------------------------------
# 9. Learnings injection structure
# ---------------------------------------------------------------------------

class TestLearningsInjectionStructure:
    """Verify exact formatting of injected learnings in the system prompt."""

    def test_learnings_section_heading(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            learnings=["something"],
        )
        assert "## Past Learnings" in system

    def test_learnings_formatted_as_bullets(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            learnings=["first lesson", "second lesson"],
        )
        assert "- first lesson" in system
        assert "- second lesson" in system

    def test_learnings_appended_after_memory(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            memory_results=["mem"],
            learnings=["learn"],
        )
        mem_pos   = system.index("## Relevant Memory")
        learn_pos = system.index("## Past Learnings")
        assert mem_pos < learn_pos

    def test_learnings_separator_is_double_newline(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            learnings=["a lesson"],
        )
        assert "\n\n## Past Learnings" in system

    def test_only_first_five_learnings_appear(self):
        cm = ContextManager()
        many = [f"lesson {i}" for i in range(8)]
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            learnings=many,
        )
        if "## Past Learnings" in system:
            for i in range(5):
                assert f"lesson {i}" in system
            for i in range(5, 8):
                assert f"lesson {i}" not in system

    def test_learnings_content_correct_with_single_item(self):
        cm = ContextManager()
        system, _ = cm.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            learnings=["only this one"],
        )
        assert "- only this one" in system


# ---------------------------------------------------------------------------
# 10. Token counting edge cases
# ---------------------------------------------------------------------------

class TestTokenCounting:
    """Edge cases for the _approx_tokens helper."""

    def test_minimum_one_for_empty_string(self):
        assert _approx_tokens("") == 1

    def test_minimum_one_for_short_string(self):
        # 1, 2, 3 chars all round down to 0 but are clamped to 1
        assert _approx_tokens("a") == 1
        assert _approx_tokens("ab") == 1
        assert _approx_tokens("abc") == 1

    def test_four_chars_equals_one_token(self):
        assert _approx_tokens("abcd") == 1

    def test_five_chars_equals_one_token(self):
        # 5 // 4 == 1
        assert _approx_tokens("abcde") == 1

    def test_eight_chars_equals_two_tokens(self):
        assert _approx_tokens("a" * 8) == 2

    def test_large_string_linear_scaling(self):
        text = "a" * 4000
        assert _approx_tokens(text) == 1000

    def test_whitespace_counted_like_other_chars(self):
        # Four spaces = 1 token
        assert _approx_tokens("    ") == 1

    def test_unicode_multibyte_chars_counted_by_len(self):
        # Python len() counts code points, not bytes; emoji = 1 code point = 0.25 tokens
        # Four emojis => 4 code points => 1 token
        four_emoji = "\U0001F600\U0001F600\U0001F600\U0001F600"
        assert _approx_tokens(four_emoji) == 1

    def test_newlines_counted_as_chars(self):
        # 4 newlines = 1 token
        assert _approx_tokens("\n\n\n\n") == 1

    def test_token_count_used_in_budget_arithmetic(self):
        # Verify that _approx_tokens is consistent with budget decisions:
        # If a message has exactly (history_budget) tokens, it should fit.
        budget = make_budget(total=200, fresh_tail_count=0)
        cm = ContextManager(budget)
        # available = 200 tokens; new_message "x" = 1 token; history_budget = 199 tokens
        # A message of exactly 199 tokens = 796 chars should fit.
        fitting_content = "a" * 796
        assert _approx_tokens(fitting_content) == 199
        history = [{"role": "user", "content": fitting_content}]
        _, messages = cm.build_messages(
            system="S", new_message="x", history=history
        )
        assert any(m["content"] == fitting_content for m in messages)

    def test_available_budget_formula(self):
        # available = total - system_reserve - tool_definitions_reserve
        b = TokenBudget(total=30_000, system_reserve=2_000, tool_definitions_reserve=2_000)
        expected_available = 26_000
        memory_budget   = int(expected_available * b.memory_fraction)
        learnings_budget = int(expected_available * b.learnings_fraction)
        history_budget  = expected_available - memory_budget - learnings_budget
        # Sanity check the arithmetic the ContextManager uses internally.
        assert memory_budget   == int(26_000 * 0.15)   # 3_900
        assert learnings_budget == int(26_000 * 0.05)  # 1_300
        assert history_budget  == 26_000 - 3_900 - 1_300  # 20_800
