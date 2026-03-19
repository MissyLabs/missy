"""Comprehensive tests for MemoryConsolidator.

Covers the full behavioural spec for missy.agent.consolidation.MemoryConsolidator
with focus on boundary conditions, keyword detection, deduplication, summary
message structure, and static token estimation.

Tests in this file are distinct from those already exercised in:
  - tests/agent/test_consolidation.py
  - tests/agent/test_consolidation_approval_edges.py
  - tests/agent/test_attention_consolidation_edges.py
"""

from __future__ import annotations

import pytest

from missy.agent.consolidation import (
    _FACT_KEYWORDS,
    _RECENT_KEEP,
    MemoryConsolidator,
)

# ===========================================================================
# should_consolidate — threshold logic
# ===========================================================================


class TestShouldConsolidateThreshold:
    """Precise boundary checks for the >= comparison in should_consolidate."""

    def test_default_params_below_threshold(self):
        """21_000 / 30_000 == 0.7 — strictly below 0.8, must return False."""
        mc = MemoryConsolidator()
        assert mc.should_consolidate(21_000) is False

    def test_default_params_at_threshold(self):
        """24_000 / 30_000 == 0.8 exactly — must return True (>= comparison)."""
        mc = MemoryConsolidator()
        assert mc.should_consolidate(24_000) is True

    def test_default_params_above_threshold(self):
        """25_000 / 30_000 > 0.8 — must return True."""
        mc = MemoryConsolidator()
        assert mc.should_consolidate(25_000) is True

    def test_default_params_at_full_capacity(self):
        """30_000 / 30_000 == 1.0 — always above any sub-1.0 threshold."""
        mc = MemoryConsolidator()
        assert mc.should_consolidate(30_000) is True

    def test_default_params_over_capacity(self):
        """Token usage exceeding max_tokens still triggers consolidation."""
        mc = MemoryConsolidator()
        assert mc.should_consolidate(35_000) is True

    def test_zero_current_tokens_never_triggers(self):
        mc = MemoryConsolidator()
        assert mc.should_consolidate(0) is False

    def test_max_tokens_zero_returns_false_regardless_of_current(self):
        """Guard against division-by-zero: max_tokens <= 0 always returns False."""
        mc = MemoryConsolidator(max_tokens=0)
        assert mc.should_consolidate(0) is False
        assert mc.should_consolidate(1_000_000) is False

    def test_negative_max_tokens_returns_false(self):
        """Negative max_tokens satisfies the <= 0 guard."""
        mc = MemoryConsolidator(max_tokens=-100)
        assert mc.should_consolidate(50) is False

    def test_custom_threshold_pct_0_5_at_boundary(self):
        """threshold_pct=0.5 with max_tokens=10_000: exactly 5_000 triggers."""
        mc = MemoryConsolidator(threshold_pct=0.5, max_tokens=10_000)
        assert mc.should_consolidate(4_999) is False
        assert mc.should_consolidate(5_000) is True

    def test_custom_threshold_pct_1_0_requires_full_capacity(self):
        """threshold_pct=1.0 fires only when current == max."""
        mc = MemoryConsolidator(threshold_pct=1.0, max_tokens=500)
        assert mc.should_consolidate(499) is False
        assert mc.should_consolidate(500) is True

    def test_custom_small_max_tokens_exact_boundary(self):
        """Small budgets still use the same >= comparison."""
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=10)
        # 8 / 10 == 0.8
        assert mc.should_consolidate(7) is False
        assert mc.should_consolidate(8) is True

    def test_result_is_bool_not_float(self):
        """should_consolidate must return an actual bool, not a float."""
        mc = MemoryConsolidator()
        result = mc.should_consolidate(24_000)
        assert type(result) is bool  # noqa: E721


# ===========================================================================
# consolidate — empty and small message lists
# ===========================================================================


class TestConsolidateEmpty:
    def test_empty_list_returns_empty_list_and_empty_string(self):
        mc = MemoryConsolidator()
        result, summary = mc.consolidate([], "system")
        assert result == []
        assert summary == ""

    def test_empty_list_result_is_list_not_other_type(self):
        mc = MemoryConsolidator()
        result, _ = mc.consolidate([], "system")
        assert isinstance(result, list)


class TestConsolidateFewerThanRecentKeep:
    """Lists with len <= _RECENT_KEEP must pass through untouched."""

    def test_one_message_returned_as_is(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": "hi"}]
        result, summary = mc.consolidate(msgs, "sys")
        assert result == msgs
        assert summary == ""

    def test_two_messages_returned_as_is(self):
        mc = MemoryConsolidator()
        msgs = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        result, summary = mc.consolidate(msgs, "sys")
        assert result == msgs
        assert summary == ""

    def test_three_messages_returned_as_is(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(3)]
        result, summary = mc.consolidate(msgs, "sys")
        assert result == msgs
        assert summary == ""

    def test_exactly_recent_keep_messages_returned_as_is(self):
        """Exactly _RECENT_KEEP messages: boundary condition, no consolidation."""
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(_RECENT_KEEP)]
        result, summary = mc.consolidate(msgs, "sys")
        assert result == msgs
        assert summary == ""

    def test_fewer_than_recent_keep_result_is_new_list(self):
        """consolidate() returns list(messages), so the object identity differs."""
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": "only"}]
        result, _ = mc.consolidate(msgs, "sys")
        assert result is not msgs

    def test_fewer_than_recent_keep_elements_equal(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        result, _ = mc.consolidate(msgs, "sys")
        assert result[0] == msgs[0]
        assert result[1] == msgs[1]


# ===========================================================================
# consolidate — more than _RECENT_KEEP messages
# ===========================================================================


class TestConsolidateAboveThreshold:
    """Behaviour when len(messages) > _RECENT_KEEP."""

    def test_five_messages_yields_five_result_items(self):
        """1 summary + _RECENT_KEEP recent = 5 items when 5 are given."""
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        result, _ = mc.consolidate(msgs, "sys")
        assert len(result) == 5

    def test_ten_messages_yields_five_result_items(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(10)]
        result, _ = mc.consolidate(msgs, "sys")
        assert len(result) == _RECENT_KEEP + 1

    def test_hundred_messages_yields_five_result_items(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(100)]
        result, _ = mc.consolidate(msgs, "sys")
        assert len(result) == _RECENT_KEEP + 1

    def test_recent_four_preserved_exactly(self):
        """The last _RECENT_KEEP messages must appear verbatim at the end."""
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(8)]
        result, _ = mc.consolidate(msgs, "sys")
        assert result[-_RECENT_KEEP:] == msgs[-_RECENT_KEEP:]

    def test_summary_message_is_first_element(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(6)]
        result, _ = mc.consolidate(msgs, "sys")
        assert result[0]["role"] == "user"
        assert "[Session context consolidated]" in result[0]["content"]

    def test_summary_message_content_contains_summary_text(self):
        """The returned summary string must be embedded in the summary message."""
        mc = MemoryConsolidator()
        msgs = [
            {"role": "assistant", "content": "Result: deployment complete"},
            {"role": "user", "content": "r1"},
            {"role": "user", "content": "r2"},
            {"role": "user", "content": "r3"},
            {"role": "user", "content": "r4"},
        ]
        result, summary = mc.consolidate(msgs, "sys")
        assert summary in result[0]["content"]

    def test_old_messages_not_in_consolidated_output(self):
        """Messages that were consolidated must not appear as standalone entries."""
        mc = MemoryConsolidator()
        old_content = "this is the old message that must be gone"
        msgs = [{"role": "user", "content": old_content}] + [
            {"role": "user", "content": f"recent{i}"} for i in range(_RECENT_KEEP)
        ]
        result, _ = mc.consolidate(msgs, "sys")
        direct_contents = [m["content"] for m in result]
        assert old_content not in direct_contents

    def test_original_list_not_mutated(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(8)]
        original_copy = list(msgs)
        mc.consolidate(msgs, "sys")
        assert msgs == original_copy

    def test_result_is_new_list_object(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(6)]
        result, _ = mc.consolidate(msgs, "sys")
        assert result is not msgs

    def test_system_prompt_parameter_accepted_and_ignored(self):
        """system_prompt is accepted for interface symmetry but not required in output."""
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(6)]
        result1, _ = mc.consolidate(msgs, "prompt A")
        result2, _ = mc.consolidate(msgs, "prompt B")
        # Both should produce the same structure regardless of system_prompt value.
        assert len(result1) == len(result2)

    def test_no_facts_fallback_text_in_summary(self):
        """When no facts are extractable, a placeholder appears in the summary."""
        mc = MemoryConsolidator()
        # Old messages: all are long assistant prose with no keywords.
        old = [
            {"role": "assistant", "content": "Here is a long explanation that contains no actionable keywords. " * 3},
            {"role": "assistant", "content": "Another verbose paragraph with no specific decisions mentioned. " * 3},
        ]
        recent = [{"role": "user", "content": f"r{i}"} for i in range(_RECENT_KEEP)]
        _, summary = mc.consolidate(old + recent, "sys")
        assert "no key facts extracted" in summary

    def test_no_facts_fallback_text_in_message_content(self):
        mc = MemoryConsolidator()
        old = [
            {"role": "assistant", "content": "A very verbose message with no facts. " * 5},
        ]
        recent = [{"role": "user", "content": f"r{i}"} for i in range(_RECENT_KEEP)]
        result, _ = mc.consolidate(old + recent, "sys")
        assert "no key facts extracted" in result[0]["content"]

    def test_facts_summary_uses_bullet_format(self):
        """Each extracted fact should appear as a '- fact' bullet in the summary."""
        mc = MemoryConsolidator()
        msgs = [
            {"role": "assistant", "content": "result: step completed"},
            {"role": "user", "content": "r1"},
            {"role": "user", "content": "r2"},
            {"role": "user", "content": "r3"},
            {"role": "user", "content": "r4"},
        ]
        _, summary = mc.consolidate(msgs, "sys")
        assert "- " in summary

    def test_multiple_consolidation_rounds_reduce_tokens(self):
        """Calling consolidate twice on the growing context further compresses it."""
        mc = MemoryConsolidator()
        # First round: 8 messages -> 5
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(8)]
        round1, _ = mc.consolidate(msgs, "sys")
        assert len(round1) == 5

        # Second round: add 4 more to the result -> now 9 total -> still compresses to 5
        msgs2 = round1 + [{"role": "user", "content": f"new{i}"} for i in range(4)]
        round2, _ = mc.consolidate(msgs2, "sys")
        assert len(round2) == 5


# ===========================================================================
# extract_key_facts — tool messages
# ===========================================================================


class TestExtractKeyFactsToolMessages:
    def test_tool_message_prefixed_with_name_in_brackets(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "tool", "name": "shell_exec", "content": "exit 0"}]
        facts = mc.extract_key_facts(msgs)
        assert facts == ["[shell_exec] exit 0"]

    def test_tool_message_content_truncated_at_200_chars(self):
        mc = MemoryConsolidator()
        long = "A" * 300
        msgs = [{"role": "tool", "name": "my_tool", "content": long}]
        facts = mc.extract_key_facts(msgs)
        assert len(facts) == 1
        # Label "[my_tool] " is 10 chars; content portion capped at 200.
        assert facts[0] == f"[my_tool] {'A' * 200}"

    def test_tool_message_exactly_200_chars_not_truncated(self):
        mc = MemoryConsolidator()
        content_200 = "B" * 200
        msgs = [{"role": "tool", "name": "t", "content": content_200}]
        facts = mc.extract_key_facts(msgs)
        assert f"[t] {'B' * 200}" in facts

    def test_tool_message_empty_content_excluded(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "tool", "name": "noop", "content": ""}]
        assert mc.extract_key_facts(msgs) == []

    def test_tool_message_whitespace_only_excluded(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "tool", "name": "noop", "content": "   \t\n"}]
        assert mc.extract_key_facts(msgs) == []

    def test_tool_message_missing_name_defaults_to_tool(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "tool", "content": "output value"}]
        facts = mc.extract_key_facts(msgs)
        assert len(facts) == 1
        assert facts[0].startswith("[tool]")

    def test_duplicate_tool_messages_deduplicated(self):
        mc = MemoryConsolidator()
        msg = {"role": "tool", "name": "cmd", "content": "ok"}
        facts = mc.extract_key_facts([msg, msg, msg])
        assert facts.count("[cmd] ok") == 1

    def test_two_different_tool_messages_both_included(self):
        mc = MemoryConsolidator()
        msgs = [
            {"role": "tool", "name": "tool_a", "content": "result a"},
            {"role": "tool", "name": "tool_b", "content": "result b"},
        ]
        facts = mc.extract_key_facts(msgs)
        assert "[tool_a] result a" in facts
        assert "[tool_b] result b" in facts


# ===========================================================================
# extract_key_facts — keyword detection
# ===========================================================================


class TestExtractKeyFactsKeywords:
    @pytest.mark.parametrize("keyword", list(_FACT_KEYWORDS))
    def test_each_keyword_triggers_line_extraction(self, keyword: str):
        """Every keyword in _FACT_KEYWORDS must cause the containing line to be extracted."""
        mc = MemoryConsolidator()
        line = f"{keyword} some important information"
        msgs = [{"role": "assistant", "content": line}]
        facts = mc.extract_key_facts(msgs)
        assert any(line in f for f in facts), f"Keyword {keyword!r} not detected"

    def test_keyword_matching_is_case_insensitive(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "assistant", "content": "ERROR: disk full"}]
        facts = mc.extract_key_facts(msgs)
        assert any("ERROR: disk full" in f for f in facts)

    def test_keyword_in_uppercase_line_preserved_verbatim(self):
        """The line is stored as-is even though matching is case-insensitive."""
        mc = MemoryConsolidator()
        line = "SUCCESS: all checks passed"
        msgs = [{"role": "assistant", "content": line}]
        facts = mc.extract_key_facts(msgs)
        assert line in facts

    def test_multiline_content_all_keyword_lines_extracted(self):
        mc = MemoryConsolidator()
        content = "result: step 1 ok\nsome prose without keywords\nfound: the bug"
        msgs = [{"role": "assistant", "content": content}]
        facts = mc.extract_key_facts(msgs)
        assert any("result: step 1 ok" in f for f in facts)
        assert any("found: the bug" in f for f in facts)

    def test_prose_lines_without_keywords_excluded(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "assistant", "content": "Here is a general observation about the system."}]
        facts = mc.extract_key_facts(msgs)
        assert facts == []

    def test_duplicate_keyword_lines_across_messages_deduplicated(self):
        mc = MemoryConsolidator()
        msgs = [
            {"role": "assistant", "content": "result: done"},
            {"role": "assistant", "content": "result: done"},
        ]
        facts = mc.extract_key_facts(msgs)
        assert facts.count("result: done") == 1

    def test_blank_lines_in_content_skipped(self):
        """Empty and whitespace-only lines must not produce empty fact strings."""
        mc = MemoryConsolidator()
        content = "\n\n   \nresult: the job finished\n\n"
        msgs = [{"role": "assistant", "content": content}]
        facts = mc.extract_key_facts(msgs)
        assert "" not in facts
        assert any("result: the job finished" in f for f in facts)


# ===========================================================================
# extract_key_facts — short user messages
# ===========================================================================


class TestExtractKeyFactsShortUserMessages:
    def test_short_user_message_under_120_chars_included(self):
        mc = MemoryConsolidator()
        short = "Deploy the app to staging"
        msgs = [{"role": "user", "content": short}]
        facts = mc.extract_key_facts(msgs)
        assert short in facts

    def test_user_message_exactly_120_chars_included(self):
        mc = MemoryConsolidator()
        msg_120 = "X" * 120
        msgs = [{"role": "user", "content": msg_120}]
        facts = mc.extract_key_facts(msgs)
        assert msg_120 in facts

    def test_user_message_121_chars_excluded_without_keyword(self):
        mc = MemoryConsolidator()
        msg_121 = "Y" * 121
        msgs = [{"role": "user", "content": msg_121}]
        facts = mc.extract_key_facts(msgs)
        assert msg_121 not in facts

    def test_long_user_message_with_keyword_still_extracts_the_keyword_line(self):
        """Even a >120-char user message produces a fact for its keyword-bearing lines."""
        mc = MemoryConsolidator()
        long_prefix = "context " * 20
        content = long_prefix + "\nresult: the upload succeeded"
        msgs = [{"role": "user", "content": content}]
        facts = mc.extract_key_facts(msgs)
        assert any("result: the upload succeeded" in f for f in facts)

    def test_empty_user_message_excluded(self):
        """An empty user message must not be added (0 < len guard)."""
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": ""}]
        facts = mc.extract_key_facts(msgs)
        assert "" not in facts

    def test_short_user_message_with_keyword_not_doubled(self):
        """A short user message that also matches a keyword appears exactly once."""
        mc = MemoryConsolidator()
        msg = "result: ok"  # short (<= 120) AND contains a keyword
        msgs = [{"role": "user", "content": msg}]
        facts = mc.extract_key_facts(msgs)
        assert facts.count("result: ok") == 1

    def test_duplicate_short_user_messages_deduplicated(self):
        mc = MemoryConsolidator()
        msg = "Run the tests"
        msgs = [{"role": "user", "content": msg}, {"role": "user", "content": msg}]
        facts = mc.extract_key_facts(msgs)
        assert facts.count(msg) == 1


# ===========================================================================
# extract_key_facts — empty input and no-facts path
# ===========================================================================


class TestExtractKeyFactsEdgeCases:
    def test_empty_messages_list_returns_empty(self):
        mc = MemoryConsolidator()
        assert mc.extract_key_facts([]) == []

    def test_messages_with_missing_role_key_do_not_raise(self):
        """Messages lacking 'role' should not raise; they default to empty string."""
        mc = MemoryConsolidator()
        msgs = [{"content": "result: something happened"}]
        facts = mc.extract_key_facts(msgs)
        # The keyword line should still be extracted via the keyword scan.
        assert any("result: something happened" in f for f in facts)

    def test_messages_with_missing_content_key_do_not_raise(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "user"}]
        facts = mc.extract_key_facts(msgs)
        # Empty content -> nothing to extract, but no exception.
        assert isinstance(facts, list)

    def test_messages_with_none_content_do_not_raise(self):
        """content=None is coerced to the string 'None'; must not crash."""
        mc = MemoryConsolidator()
        msgs = [{"role": "assistant", "content": None}]
        # Should not raise; the stringified content won't match any keyword.
        facts = mc.extract_key_facts(msgs)
        assert isinstance(facts, list)

    def test_return_type_is_list(self):
        mc = MemoryConsolidator()
        result = mc.extract_key_facts([])
        assert isinstance(result, list)

    def test_no_facts_leads_to_fallback_in_consolidate(self):
        """When no facts are extracted, consolidate embeds the fallback placeholder."""
        mc = MemoryConsolidator()
        old = [{"role": "assistant", "content": "Some lengthy non-factual prose paragraph. " * 4}]
        recent = [{"role": "user", "content": f"keep{i}"} for i in range(_RECENT_KEEP)]
        result, summary = mc.consolidate(old + recent, "sys")
        assert "(previous conversation context" in result[0]["content"]
        assert "(previous conversation context" in summary


# ===========================================================================
# estimate_tokens — static method
# ===========================================================================


class TestEstimateTokens:
    def test_empty_list_returns_zero(self):
        assert MemoryConsolidator.estimate_tokens([]) == 0

    def test_400_chars_yields_100_tokens(self):
        msgs = [{"role": "user", "content": "a" * 400}]
        assert MemoryConsolidator.estimate_tokens(msgs) == 100

    def test_800_chars_across_two_messages_yields_200_tokens(self):
        msgs = [
            {"role": "user", "content": "a" * 400},
            {"role": "assistant", "content": "b" * 400},
        ]
        assert MemoryConsolidator.estimate_tokens(msgs) == 200

    def test_three_chars_rounds_down_to_zero(self):
        msgs = [{"role": "user", "content": "abc"}]
        assert MemoryConsolidator.estimate_tokens(msgs) == 0

    def test_four_chars_exactly_one_token(self):
        msgs = [{"role": "user", "content": "abcd"}]
        assert MemoryConsolidator.estimate_tokens(msgs) == 1

    def test_five_chars_rounds_down_to_one_token(self):
        msgs = [{"role": "user", "content": "abcde"}]
        assert MemoryConsolidator.estimate_tokens(msgs) == 1

    def test_empty_content_contributes_zero(self):
        msgs = [{"role": "user", "content": ""}]
        assert MemoryConsolidator.estimate_tokens(msgs) == 0

    def test_missing_content_key_treated_as_empty(self):
        msgs = [{"role": "user"}]
        assert MemoryConsolidator.estimate_tokens(msgs) == 0

    def test_non_string_content_coerced_via_str(self):
        """Non-string content must be str()-coerced before measuring length."""
        # str(12345678) == "12345678" → 8 chars → 2 tokens
        msgs = [{"role": "tool", "content": 12_345_678}]
        assert MemoryConsolidator.estimate_tokens(msgs) == 2

    def test_callable_on_class_without_instance(self):
        """estimate_tokens is a @staticmethod and can be called on the class."""
        msgs = [{"role": "user", "content": "a" * 40}]
        assert MemoryConsolidator.estimate_tokens(msgs) == 10

    def test_callable_on_instance(self):
        """estimate_tokens is also accessible on an instance."""
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": "a" * 40}]
        assert mc.estimate_tokens(msgs) == 10

    def test_result_is_int(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = MemoryConsolidator.estimate_tokens(msgs)
        assert isinstance(result, int)


# ===========================================================================
# constructor — custom parameters
# ===========================================================================


class TestConstructorParameters:
    def test_default_threshold_pct_is_0_8(self):
        mc = MemoryConsolidator()
        # Verify via behaviour: 24_000 / 30_000 == 0.8 triggers
        assert mc.should_consolidate(24_000) is True
        assert mc.should_consolidate(23_999) is False

    def test_default_max_tokens_is_30000(self):
        mc = MemoryConsolidator()
        # 30_000 / 30_000 == 1.0 triggers (any positive threshold)
        assert mc.should_consolidate(30_000) is True

    def test_custom_threshold_stored_correctly(self):
        mc = MemoryConsolidator(threshold_pct=0.6, max_tokens=1_000)
        # 600 / 1000 == 0.6 triggers
        assert mc.should_consolidate(599) is False
        assert mc.should_consolidate(600) is True

    def test_custom_max_tokens_stored_correctly(self):
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=5_000)
        # 4_000 / 5_000 == 0.8 triggers
        assert mc.should_consolidate(3_999) is False
        assert mc.should_consolidate(4_000) is True


# ===========================================================================
# _RECENT_KEEP and _FACT_KEYWORDS constants
# ===========================================================================


class TestModuleConstants:
    def test_recent_keep_is_four(self):
        assert _RECENT_KEEP == 4

    def test_fact_keywords_count(self):
        assert len(_FACT_KEYWORDS) == 10

    def test_fact_keywords_all_lowercase_and_end_with_colon(self):
        for kw in _FACT_KEYWORDS:
            assert kw == kw.lower(), f"Keyword not lowercase: {kw!r}"
            assert kw.endswith(":"), f"Keyword does not end with colon: {kw!r}"

    def test_fact_keywords_contains_result(self):
        assert "result:" in _FACT_KEYWORDS

    def test_fact_keywords_contains_error(self):
        assert "error:" in _FACT_KEYWORDS
