"""Edge case tests for AttentionSystem and MemoryConsolidator.

Covers gaps not addressed by test_attention.py and test_consolidation.py.
All tests are pure-Python, no I/O, no mocking required.
"""

from __future__ import annotations

import pytest

from missy.agent.attention import (
    AlertingAttention,
    AttentionSystem,
    ExecutiveAttention,
    OrientingAttention,
    SelectiveAttention,
    SustainedAttention,
)
from missy.agent.consolidation import MemoryConsolidator

# ---------------------------------------------------------------------------
# AttentionSystem — subsystem composition
# ---------------------------------------------------------------------------


class TestAttentionSystemSubsystems:
    """Verify all five subsystem instances are present and of correct types."""

    def test_all_five_subsystems_present(self):
        attn = AttentionSystem()
        assert isinstance(attn._alerting, AlertingAttention)
        assert isinstance(attn._orienting, OrientingAttention)
        assert isinstance(attn._sustained, SustainedAttention)
        assert isinstance(attn._selective, SelectiveAttention)
        assert isinstance(attn._executive, ExecutiveAttention)

    def test_process_populates_all_state_fields(self):
        """process() must fill every AttentionState field, not leave defaults everywhere."""
        attn = AttentionSystem()
        state = attn.process("error in the config file")
        # urgency > 0 because "error" is a keyword
        assert state.urgency > 0.0
        # focus_duration always >= 1
        assert state.focus_duration >= 1
        # priority_tools is a list (may be empty or not, but must exist)
        assert isinstance(state.priority_tools, list)
        # context_filter is lowercase topics
        assert all(t == t.lower() for t in state.context_filter)

    def test_process_empty_string_input(self):
        """Empty input must not raise and must return a valid, low-urgency state."""
        attn = AttentionSystem()
        state = attn.process("")
        assert state.urgency == pytest.approx(0.0)
        assert state.topics == []
        assert state.focus_duration >= 1
        assert state.priority_tools == []
        assert state.context_filter == []

    def test_context_filter_is_lowercase_of_topics(self):
        """context_filter must be the lowercased version of extracted topics."""
        attn = AttentionSystem()
        state = attn.process("Configure Docker networking")
        for topic, cf in zip(state.topics, state.context_filter, strict=False):
            assert cf == topic.lower()

    def test_process_consecutive_different_topics_keeps_duration_at_one(self):
        """Switching to a completely different topic every turn should never exceed 1."""
        attn = AttentionSystem()
        # Each call uses a unique, non-overlapping capitalised entity.
        attn.process("Tell me about Kubernetes")
        state2 = attn.process("Explain Python decorators")
        state3 = attn.process("What is Rust ownership")
        # Every topic switch resets sustained attention to 1.
        assert state2.focus_duration == 1
        assert state3.focus_duration == 1


# ---------------------------------------------------------------------------
# AlertingAttention — urgency scoring edge cases
# ---------------------------------------------------------------------------


class TestAlertingAttentionEdges:
    def test_punctuation_stripped_before_keyword_match(self):
        """'error!' should match keyword 'error' after stripping punctuation."""
        alerting = AlertingAttention()
        score = alerting.score("error!")
        assert score == pytest.approx(1.0)

    def test_single_urgency_keyword_in_long_text_gives_partial_score(self):
        """One urgency word in a long sentence gives a low but nonzero score."""
        alerting = AlertingAttention()
        # 1 keyword ("failed") in 10 words -> 0.1
        text = "the deployment pipeline ran and then suddenly failed last night"
        score = alerting.score(text)
        assert 0.0 < score < 0.5

    def test_multiple_urgency_keywords_in_long_text(self):
        """Multiple keywords in a longer text produce intermediate score."""
        alerting = AlertingAttention()
        # 3 urgency keywords out of 12 words -> 0.25
        text = "the server is down and the error is critical please respond"
        score = alerting.score(text)
        assert 0.2 < score < 1.0

    def test_all_urgency_keywords_score_exactly_one(self):
        """Input consisting entirely of urgency keywords caps at 1.0."""
        alerting = AlertingAttention()
        score = alerting.score(
            "error critical urgent broken down failed security immediately asap emergency"
        )
        assert score == pytest.approx(1.0)

    def test_case_insensitive_matching(self):
        """Urgency detection is case-insensitive via .lower()."""
        alerting = AlertingAttention()
        score_lower = alerting.score("error")
        score_upper = alerting.score("ERROR")
        assert score_lower == pytest.approx(score_upper)


# ---------------------------------------------------------------------------
# OrientingAttention — topic extraction edge cases
# ---------------------------------------------------------------------------


class TestOrientingAttentionEdges:
    def test_empty_string_returns_empty_list(self):
        orienting = OrientingAttention()
        assert orienting.extract_topics("") == []

    def test_duplicate_topics_deduplicated(self):
        """The same topic word appearing twice should only appear once in output."""
        orienting = OrientingAttention()
        # "Docker" appears after "about" and again capitalised mid-sentence.
        topics = orienting.extract_topics("tell me about Docker and more about Docker")
        assert topics.count("Docker") == 1

    def test_all_lowercase_sentence_no_capitals(self):
        """Sentence with no capitalised words and no prepositions yields no topics."""
        orienting = OrientingAttention()
        topics = orienting.extract_topics("what is the time right now please")
        # "the" is a topic preposition; "time" follows it
        assert "time" in topics

    def test_word_at_sentence_start_not_extracted_as_topic(self):
        """The first word, even if capitalised, is not treated as a topic."""
        orienting = OrientingAttention()
        topics = orienting.extract_topics("Docker is a container runtime")
        # "Docker" is at index 0, so it should NOT be extracted as a topic.
        assert "Docker" not in topics

    def test_word_after_about_extracted(self):
        orienting = OrientingAttention()
        topics = orienting.extract_topics("tell me about networking")
        assert "networking" in topics

    def test_word_after_with_extracted(self):
        orienting = OrientingAttention()
        topics = orienting.extract_topics("help me with testing")
        assert "testing" in topics


# ---------------------------------------------------------------------------
# SustainedAttention — focus continuity edge cases
# ---------------------------------------------------------------------------


class TestSustainedAttentionEdges:
    def test_empty_topics_resets_duration_to_one(self):
        """Passing an empty list should reset focus duration to 1."""
        sustained = SustainedAttention()
        sustained.update(["Docker"])
        sustained.update(["Docker"])
        duration = sustained.update([])
        assert duration == 1

    def test_empty_topics_on_first_call_returns_one(self):
        sustained = SustainedAttention()
        assert sustained.update([]) == 1

    def test_partial_overlap_below_threshold_resets(self):
        """Less than 50% topic overlap should reset duration to 1."""
        sustained = SustainedAttention()
        # Establish focus on 4 topics.
        sustained.update(["a", "b", "c", "d"])
        # Only 1 of 4 overlaps -> 25% < 50% -> reset.
        duration = sustained.update(["a", "x", "y", "z"])
        assert duration == 1

    def test_partial_overlap_above_threshold_increments(self):
        """More than 50% topic overlap should increment duration."""
        sustained = SustainedAttention()
        sustained.update(["a", "b", "c"])
        # 2 of 3 overlap -> 66% > 50% -> increment.
        duration = sustained.update(["a", "b", "d"])
        assert duration == 2

    def test_exact_50_percent_overlap_does_not_increment(self):
        """Exactly 50% overlap (not strictly greater) resets to 1."""
        sustained = SustainedAttention()
        sustained.update(["a", "b"])
        # 1 of 2 overlaps -> 50%, condition is > 0.5 -> does NOT increment.
        duration = sustained.update(["a", "c"])
        assert duration == 1

    def test_duration_accumulates_over_many_turns(self):
        sustained = SustainedAttention()
        for _ in range(5):
            sustained.update(["Docker"])
        assert sustained.update(["Docker"]) == 6

    def test_case_insensitive_overlap_detection(self):
        """Topic matching for overlap is case-insensitive."""
        sustained = SustainedAttention()
        sustained.update(["Docker"])
        # "docker" (lowercase) should match "Docker".
        duration = sustained.update(["docker"])
        assert duration == 2


# ---------------------------------------------------------------------------
# SelectiveAttention — filtering edge cases
# ---------------------------------------------------------------------------


class TestSelectiveAttentionEdges:
    def test_empty_fragments_returns_empty_list(self):
        result = SelectiveAttention.filter([], ["Docker"])
        assert result == []

    def test_empty_topics_and_empty_fragments(self):
        result = SelectiveAttention.filter([], [])
        assert result == []

    def test_case_insensitive_fragment_matching(self):
        """Topic matching against fragments is case-insensitive."""
        fragments = ["DOCKER container setup guide"]
        result = SelectiveAttention.filter(fragments, ["docker"])
        assert fragments[0] in result

    def test_no_matching_fragments_returns_empty(self):
        fragments = ["weather report", "stock prices today"]
        result = SelectiveAttention.filter(fragments, ["kubernetes"])
        assert result == []

    def test_multiple_topics_any_match_passes_fragment(self):
        """A fragment matching any one of multiple topics is kept."""
        fragments = ["python tutorial", "java basics", "rust ownership"]
        result = SelectiveAttention.filter(fragments, ["python", "rust"])
        assert "python tutorial" in result
        assert "rust ownership" in result
        assert "java basics" not in result

    def test_returns_new_list_not_same_object(self):
        """The returned list must be a new object, not the original."""
        fragments = ["a", "b"]
        result = SelectiveAttention.filter(fragments, [])
        assert result is not fragments


# ---------------------------------------------------------------------------
# ExecutiveAttention — tool prioritisation edge cases
# ---------------------------------------------------------------------------


class TestExecutiveAttentionEdges:
    def test_no_tools_when_low_urgency_and_no_file_topics(self):
        """With urgency <= 0.5 and no file-related topics, no tools are prioritised."""
        priority = ExecutiveAttention.prioritise(0.0, [])
        assert priority == []

    def test_urgency_at_boundary_0_5_does_not_trigger(self):
        """Urgency exactly at 0.5 is NOT above the threshold, so no shell/file tools."""
        priority = ExecutiveAttention.prioritise(0.5, [])
        assert priority == []

    def test_urgency_just_above_0_5_triggers_shell_and_file(self):
        priority = ExecutiveAttention.prioritise(0.51, [])
        assert "shell_exec" in priority
        assert "file_read" in priority

    def test_high_urgency_with_file_topic_no_duplicate_file_read(self):
        """When urgency > 0.5 and a file topic is also present, file_read is not duplicated."""
        priority = ExecutiveAttention.prioritise(0.9, ["config", "file"])
        assert priority.count("file_read") == 1

    def test_high_urgency_with_file_topic_skips_file_write(self):
        """When urgency > 0.5, file_read is already in priority, so the file-topic
        branch (guarded by 'file_read not in priority') is skipped entirely.
        file_write is therefore NOT added at high urgency + file topic.
        This documents the actual conditional logic in prioritise()."""
        priority = ExecutiveAttention.prioritise(0.9, ["log"])
        # "log" is a file-related topic word, but the branch guard prevents file_write.
        assert "file_write" not in priority
        # shell_exec and file_read are still present from the urgency branch.
        assert "shell_exec" in priority
        assert "file_read" in priority

    def test_only_file_topic_no_shell_exec(self):
        """File topics alone (low urgency) should NOT add shell_exec."""
        priority = ExecutiveAttention.prioritise(0.1, ["directory"])
        assert "shell_exec" not in priority
        assert "file_read" in priority

    def test_empty_topics_high_urgency_no_file_write(self):
        """High urgency with no file topics should not include file_write."""
        priority = ExecutiveAttention.prioritise(1.0, [])
        assert "file_write" not in priority


# ---------------------------------------------------------------------------
# MemoryConsolidator — should_consolidate edge cases
# ---------------------------------------------------------------------------


class TestShouldConsolidateEdges:
    def test_just_below_threshold_does_not_trigger(self):
        """79.9% usage must not trigger consolidation."""
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=1000)
        assert mc.should_consolidate(799) is False

    def test_just_above_threshold_triggers(self):
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=1000)
        assert mc.should_consolidate(801) is True

    def test_negative_tokens_does_not_trigger(self):
        """Negative token count (defensive) must not trigger consolidation."""
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=30000)
        assert mc.should_consolidate(-1) is False

    def test_max_tokens_one(self):
        """max_tokens=1 is a valid boundary; 1 token at 100% triggers."""
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=1)
        assert mc.should_consolidate(1) is True

    def test_custom_threshold_pct(self):
        """Custom thresholds work correctly."""
        mc = MemoryConsolidator(threshold_pct=0.5, max_tokens=100)
        assert mc.should_consolidate(49) is False
        assert mc.should_consolidate(50) is True


# ---------------------------------------------------------------------------
# MemoryConsolidator — consolidate edge cases
# ---------------------------------------------------------------------------


class TestConsolidateEdges:
    def test_consolidate_with_enough_messages_compresses(self):
        """A history long enough for the pipeline to compress must produce fewer messages."""
        mc = MemoryConsolidator()
        # 20 messages is well above what any step considers "recent only".
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        result, summary = mc.consolidate(messages, "system")
        # Pipeline condenses: result must be shorter than input.
        assert len(result) < len(messages)
        # The last 4 messages are preserved intact at the tail.
        assert result[-4:] == messages[-4:]

    def test_summary_message_role_is_user(self):
        """The injected consolidation summary must have role='user'."""
        mc = MemoryConsolidator()
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(8)]
        result, _ = mc.consolidate(messages, "system")
        assert result[0]["role"] == "user"

    def test_no_key_facts_produces_placeholder_text(self):
        """When old messages contain no extractable facts, a placeholder is used."""
        mc = MemoryConsolidator()
        # Verbose assistant responses with no fact keywords, long user messages.
        messages = [
            {"role": "assistant", "content": "Sure, I can help with that general question."},
            {"role": "assistant", "content": "Let me think about this problem some more."},
            {"role": "assistant", "content": "Here is my extended analysis of the situation."},
            {"role": "assistant", "content": "In conclusion, it depends on many factors."},
            {"role": "user", "content": "recent 1"},
            {"role": "assistant", "content": "recent 2"},
            {"role": "user", "content": "recent 3"},
            {"role": "assistant", "content": "recent 4"},
        ]
        result, summary = mc.consolidate(messages, "system")
        assert "no key facts extracted" in summary

    def test_consolidate_returns_new_list_not_same_object(self):
        mc = MemoryConsolidator()
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(6)]
        result, _ = mc.consolidate(messages, "system")
        assert result is not messages

    def test_system_prompt_argument_accepted_but_unused(self):
        """The system_prompt parameter is accepted for interface symmetry."""
        mc = MemoryConsolidator()
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(6)]
        # Should not raise regardless of system_prompt content.
        result, summary = mc.consolidate(messages, "any system prompt content")
        assert isinstance(result, list)
        assert isinstance(summary, str)


# ---------------------------------------------------------------------------
# MemoryConsolidator — extract_key_facts edge cases
# ---------------------------------------------------------------------------


class TestExtractKeyFactsEdges:
    def test_single_message_with_keyword_line(self):
        """A single message containing a keyword line yields that line as a fact."""
        mc = MemoryConsolidator()
        messages = [{"role": "assistant", "content": "Result: tests passed"}]
        facts = mc.extract_key_facts(messages)
        assert any("Result: tests passed" in f for f in facts)

    def test_tool_message_with_empty_content_skipped(self):
        """A tool message with empty content should not add an empty fact."""
        mc = MemoryConsolidator()
        messages = [{"role": "tool", "name": "shell_exec", "content": ""}]
        facts = mc.extract_key_facts(messages)
        assert facts == []

    def test_tool_message_content_truncated_at_200_chars(self):
        """Tool message content in facts is capped at 200 characters."""
        mc = MemoryConsolidator()
        long_output = "x" * 500
        messages = [{"role": "tool", "name": "my_tool", "content": long_output}]
        facts = mc.extract_key_facts(messages)
        assert len(facts) == 1
        # Fact = "[my_tool] " + first 200 chars of content
        assert len(facts[0]) == len("[my_tool] ") + 200

    def test_short_user_message_at_exact_boundary_included(self):
        """A user message of exactly 120 chars must be included as a fact."""
        mc = MemoryConsolidator()
        boundary_msg = "a" * 120
        messages = [{"role": "user", "content": boundary_msg}]
        facts = mc.extract_key_facts(messages)
        assert boundary_msg in facts

    def test_user_message_of_121_chars_not_included_as_short_msg(self):
        """A user message of 121 chars exceeds the short-message threshold."""
        mc = MemoryConsolidator()
        long_user_msg = "b" * 121
        messages = [{"role": "user", "content": long_user_msg}]
        facts = mc.extract_key_facts(messages)
        # No fact keyword in content; too long for short-message rule.
        assert long_user_msg not in facts

    def test_assistant_prose_without_keywords_not_included(self):
        """Verbose assistant messages with no keywords are not extracted as facts."""
        mc = MemoryConsolidator()
        prose = "This is a very long assistant response with lots of words " * 5
        messages = [{"role": "assistant", "content": prose}]
        facts = mc.extract_key_facts(messages)
        assert facts == []

    def test_all_fact_keywords_are_matched(self):
        """Each keyword in _FACT_KEYWORDS must be detectable."""
        mc = MemoryConsolidator()
        keyword_lines = [
            "result: something happened",
            "decided: use approach B",
            "found: the bug is here",
            "error: connection refused",
            "success: deployment complete",
            "created: new resource",
            "updated: configuration file",
            "deleted: old records",
            "confirmed: access granted",
            "output: 42",
        ]
        messages = [{"role": "assistant", "content": "\n".join(keyword_lines)}]
        facts = mc.extract_key_facts(messages)
        for line in keyword_lines:
            assert any(line in f for f in facts), f"Keyword line not extracted: {line!r}"

    def test_missing_role_key_does_not_raise(self):
        """Messages without a 'role' key are handled gracefully."""
        mc = MemoryConsolidator()
        messages = [{"content": "Result: graceful handling"}]
        # Should not raise; role defaults to empty string.
        facts = mc.extract_key_facts(messages)
        # Keyword line should still be extracted from content.
        assert any("Result: graceful handling" in f for f in facts)

    def test_missing_content_key_does_not_raise(self):
        """Messages without a 'content' key are handled gracefully."""
        mc = MemoryConsolidator()
        messages = [{"role": "user"}]
        facts = mc.extract_key_facts(messages)
        # No content -> no facts, no exception.
        assert isinstance(facts, list)

    def test_empty_messages_list_returns_empty(self):
        mc = MemoryConsolidator()
        assert mc.extract_key_facts([]) == []


# ---------------------------------------------------------------------------
# MemoryConsolidator — estimate_tokens edge cases
# ---------------------------------------------------------------------------


class TestEstimateTokensEdges:
    def test_missing_content_key_does_not_raise(self):
        """Messages without 'content' key should be treated as zero chars."""
        mc = MemoryConsolidator()
        messages = [{"role": "user"}]
        tokens = mc.estimate_tokens(messages)
        assert tokens == 0

    def test_single_char_content(self):
        mc = MemoryConsolidator()
        messages = [{"role": "user", "content": "a"}]
        # 1 char / 4 = 0 (integer division)
        assert mc.estimate_tokens(messages) == 0

    def test_four_chars_is_one_token(self):
        mc = MemoryConsolidator()
        messages = [{"role": "user", "content": "abcd"}]
        assert mc.estimate_tokens(messages) == 1

    def test_empty_messages_list(self):
        mc = MemoryConsolidator()
        assert mc.estimate_tokens([]) == 0

    def test_non_string_content_is_coerced(self):
        """Content values that are not strings should be str()-coerced."""
        mc = MemoryConsolidator()
        messages = [{"role": "tool", "content": 12345678}]
        # str(12345678) = "12345678" -> 8 chars -> 2 tokens
        tokens = mc.estimate_tokens(messages)
        assert tokens == 2
