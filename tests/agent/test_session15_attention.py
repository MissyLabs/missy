"""Comprehensive tests for missy.agent.attention — session 15.

Covers AttentionState, AlertingAttention, OrientingAttention,
SustainedAttention, SelectiveAttention, ExecutiveAttention, and
AttentionSystem with 50+ targeted cases including happy paths, edge
cases, boundary conditions, and keyword-set membership checks.

All tests are pure-Python, no I/O, no mocking required.
"""

from __future__ import annotations

import dataclasses

import pytest

from missy.agent.attention import (
    AlertingAttention,
    AttentionState,
    AttentionSystem,
    ExecutiveAttention,
    OrientingAttention,
    SelectiveAttention,
    SustainedAttention,
    _FILE_TOPIC_WORDS,
    _TOPIC_PREPOSITIONS,
    _URGENCY_KEYWORDS,
)


# ---------------------------------------------------------------------------
# AttentionState — dataclass defaults and independence
# ---------------------------------------------------------------------------


class TestAttentionStateDefaults:
    def test_urgency_default_is_zero(self):
        state = AttentionState()
        assert state.urgency == 0.0

    def test_topics_default_is_empty_list(self):
        state = AttentionState()
        assert state.topics == []

    def test_focus_duration_default_is_one(self):
        state = AttentionState()
        assert state.focus_duration == 1

    def test_priority_tools_default_is_empty_list(self):
        state = AttentionState()
        assert state.priority_tools == []

    def test_context_filter_default_is_empty_list(self):
        state = AttentionState()
        assert state.context_filter == []

    def test_is_a_dataclass(self):
        assert dataclasses.is_dataclass(AttentionState)

    def test_mutable_list_fields_are_independent_between_instances(self):
        """Each instance must get its own list, not a shared default."""
        s1 = AttentionState()
        s2 = AttentionState()
        s1.topics.append("leaked")
        assert s2.topics == [], "topics list leaked across instances"

    def test_priority_tools_independent_between_instances(self):
        s1 = AttentionState()
        s2 = AttentionState()
        s1.priority_tools.append("leaked_tool")
        assert s2.priority_tools == []

    def test_context_filter_independent_between_instances(self):
        s1 = AttentionState()
        s2 = AttentionState()
        s1.context_filter.append("leaked")
        assert s2.context_filter == []

    def test_explicit_field_assignment(self):
        state = AttentionState(
            urgency=0.75,
            topics=["Docker"],
            focus_duration=3,
            priority_tools=["shell_exec"],
            context_filter=["docker"],
        )
        assert state.urgency == pytest.approx(0.75)
        assert state.topics == ["Docker"]
        assert state.focus_duration == 3
        assert state.priority_tools == ["shell_exec"]
        assert state.context_filter == ["docker"]


# ---------------------------------------------------------------------------
# AlertingAttention — urgency scoring
# ---------------------------------------------------------------------------


class TestAlertingAttentionEmptyAndZero:
    def test_empty_string_returns_zero(self):
        assert AlertingAttention().score("") == pytest.approx(0.0)

    def test_whitespace_only_returns_zero(self):
        # "   ".split() == [] -> no words -> returns 0.0
        assert AlertingAttention().score("   ") == pytest.approx(0.0)

    def test_tab_only_returns_zero(self):
        assert AlertingAttention().score("\t") == pytest.approx(0.0)

    def test_no_urgency_words_returns_zero(self):
        assert AlertingAttention().score("what time is it today") == pytest.approx(0.0)

    def test_single_non_urgency_word_returns_zero(self):
        assert AlertingAttention().score("hello") == pytest.approx(0.0)


class TestAlertingAttentionFullMatch:
    def test_single_urgency_keyword_alone_returns_one(self):
        """One word that matches is 1/1 = 1.0."""
        assert AlertingAttention().score("error") == pytest.approx(1.0)

    def test_all_ten_urgency_keywords_returns_one(self):
        text = "error critical urgent broken down failed security immediately asap emergency"
        assert AlertingAttention().score(text) == pytest.approx(1.0)

    def test_duplicate_urgency_keywords_returns_one(self):
        """Repeating a keyword many times cannot exceed 1.0."""
        text = "error error error error error"
        assert AlertingAttention().score(text) == pytest.approx(1.0)


class TestAlertingAttentionPartialMatch:
    def test_one_keyword_in_three_words(self):
        # "server is down" -> 1 match / 3 words = 0.333...
        score = AlertingAttention().score("server is down")
        assert pytest.approx(score, abs=1e-6) == 1 / 3

    def test_two_keywords_in_four_words(self):
        # "critical error please respond" -> 2/4 = 0.5
        score = AlertingAttention().score("critical error please respond")
        assert score == pytest.approx(0.5)

    def test_one_keyword_in_long_sentence(self):
        # exactly 11 words, "failed" is keyword -> 1/11
        text = "the deployment pipeline ran and then suddenly failed last night ok"
        words = text.split()
        assert len(words) == 11
        score = AlertingAttention().score(text)
        assert score == pytest.approx(1 / 11)

    def test_score_stays_below_one_for_partial_match(self):
        score = AlertingAttention().score("there is an error in the system")
        assert 0.0 < score < 1.0


class TestAlertingAttentionPunctuationStripping:
    """The alerting scorer strips '!.,?;:' from word boundaries."""

    @pytest.mark.parametrize("suffix", ["!", ".", ",", "?", ";", ":"])
    def test_urgency_keyword_with_trailing_punctuation_matches(self, suffix):
        word = f"error{suffix}"
        score = AlertingAttention().score(word)
        assert score == pytest.approx(1.0), f"Failed for suffix {suffix!r}"

    @pytest.mark.parametrize("prefix", ["!", ".", ",", "?", ";", ":"])
    def test_urgency_keyword_with_leading_punctuation_matches(self, prefix):
        word = f"{prefix}error"
        score = AlertingAttention().score(word)
        assert score == pytest.approx(1.0), f"Failed for prefix {prefix!r}"

    def test_double_sided_punctuation_matches(self):
        # "!error!" stripped to "error"
        assert AlertingAttention().score("!error!") == pytest.approx(1.0)

    def test_quoted_keyword_does_not_match(self):
        # Quotes are NOT in the strip set for alerting; '"error"' does not match
        assert AlertingAttention().score('"error"') == pytest.approx(0.0)

    def test_compound_hyphenated_word_does_not_match(self):
        # Hyphen is not stripped, so "error-critical" is not in keyword set
        assert AlertingAttention().score("error-critical") == pytest.approx(0.0)


class TestAlertingAttentionCaseInsensitive:
    def test_uppercase_matches(self):
        assert AlertingAttention().score("ERROR") == pytest.approx(1.0)

    def test_mixedcase_matches(self):
        assert AlertingAttention().score("Critical") == pytest.approx(1.0)

    def test_score_same_regardless_of_case(self):
        score_low = AlertingAttention().score("error critical")
        score_high = AlertingAttention().score("ERROR CRITICAL")
        assert score_low == pytest.approx(score_high)


# ---------------------------------------------------------------------------
# OrientingAttention — topic extraction
# ---------------------------------------------------------------------------


class TestOrientingAttentionEmptyAndSimple:
    def test_empty_string_returns_empty_list(self):
        assert OrientingAttention().extract_topics("") == []

    def test_single_word_at_index_zero_not_extracted(self):
        # First word at index 0 is never a topic (even if capitalised)
        assert OrientingAttention().extract_topics("Docker") == []

    def test_single_lowercase_word_returns_empty(self):
        assert OrientingAttention().extract_topics("hello") == []


class TestOrientingAttentionPrepositions:
    """Words following topic prepositions are extracted."""

    def test_word_after_about(self):
        assert "networking" in OrientingAttention().extract_topics("tell me about networking")

    def test_word_after_with(self):
        assert "python" in OrientingAttention().extract_topics("help me with python")

    def test_word_after_for(self):
        assert "answers" in OrientingAttention().extract_topics("looking for answers")

    def test_word_after_the(self):
        assert "database" in OrientingAttention().extract_topics("fix the database")

    def test_preposition_itself_not_included_as_topic(self):
        topics = OrientingAttention().extract_topics("tell me about networking")
        assert "about" not in topics

    def test_chained_prepositions_about_the_extracts_both(self):
        # "about" fires "the", "the" fires "Docker"
        topics = OrientingAttention().extract_topics("tell me about the Docker")
        assert "Docker" in topics


class TestOrientingAttentionCapitalisedWords:
    def test_capitalised_word_after_first_position_extracted(self):
        topics = OrientingAttention().extract_topics("configure Docker networking")
        assert "Docker" in topics

    def test_capital_at_index_zero_not_extracted(self):
        topics = OrientingAttention().extract_topics("Docker is a container runtime")
        assert "Docker" not in topics

    def test_capital_after_comma_is_extracted(self):
        # Comma is not a sentence-ending character in the check
        topics = OrientingAttention().extract_topics("hello, Docker is running")
        assert "Docker" in topics

    def test_capital_after_period_not_extracted(self):
        # Previous word ends with '.' -> treated as sentence start
        topics = OrientingAttention().extract_topics("sentence ends. Python is great")
        assert "Python" not in topics

    def test_capital_after_exclamation_not_extracted(self):
        topics = OrientingAttention().extract_topics("Stop! Docker must restart")
        assert "Docker" not in topics

    def test_capital_after_question_mark_not_extracted(self):
        topics = OrientingAttention().extract_topics("Why? Python should work")
        assert "Python" not in topics


class TestOrientingAttentionDeduplification:
    def test_duplicate_topic_words_appear_once(self):
        topics = OrientingAttention().extract_topics("tell me about Docker and more about Docker")
        assert topics.count("Docker") == 1

    def test_order_preserved_on_deduplication(self):
        topics = OrientingAttention().extract_topics("help with Python and about Python")
        assert topics[0] == "Python"

    def test_parenthesised_word_stripped_and_extracted(self):
        # OrientingAttention strips '()' from words
        topics = OrientingAttention().extract_topics("run (Docker) container")
        assert "Docker" in topics

    def test_quoted_capitalised_word_stripped_and_extracted(self):
        # OrientingAttention strips '"' and "'"
        topics = OrientingAttention().extract_topics('use "Docker" container')
        assert "Docker" in topics


class TestOrientingAttentionFiltersPrepositions:
    def test_preposition_following_preposition_is_still_captured(self):
        # "about" fires "with", then "with" fires "for"; prepositions are
        # extracted as topics when they follow other prepositions
        topics = OrientingAttention().extract_topics("tell me about with for")
        # "with" follows "about" -> extracted; "for" follows "with" -> extracted
        assert "with" in topics
        assert "for" in topics


# ---------------------------------------------------------------------------
# SustainedAttention — focus continuity
# ---------------------------------------------------------------------------


class TestSustainedAttentionInitialState:
    def test_first_call_returns_one_regardless_of_topics(self):
        s = SustainedAttention()
        assert s.update(["Docker"]) == 1

    def test_first_call_empty_topics_returns_one(self):
        s = SustainedAttention()
        assert s.update([]) == 1


class TestSustainedAttentionIncrement:
    def test_same_single_topic_increments_duration(self):
        s = SustainedAttention()
        s.update(["Docker"])
        assert s.update(["Docker"]) == 2

    def test_same_topic_over_many_turns_accumulates(self):
        s = SustainedAttention()
        for _ in range(5):
            s.update(["Docker"])
        # 5 calls -> duration = 5; one more -> 6
        assert s.update(["Docker"]) == 6

    def test_majority_overlap_increments_not_resets(self):
        # prev: {a, b, c}, curr: {a, b, d} -> 2/3 = 0.67 > 0.5 -> increment
        s = SustainedAttention()
        s.update(["a", "b", "c"])
        assert s.update(["a", "b", "d"]) == 2

    def test_case_insensitive_overlap_increments(self):
        s = SustainedAttention()
        s.update(["Docker"])
        # "docker" lowercase still matches "Docker" after lowering
        assert s.update(["docker"]) == 2


class TestSustainedAttentionReset:
    def test_new_topic_resets_to_one(self):
        s = SustainedAttention()
        s.update(["Docker"])
        s.update(["Docker"])
        assert s.update(["Python", "Flask"]) == 1

    def test_empty_topics_resets_to_one_after_focus(self):
        s = SustainedAttention()
        s.update(["Docker"])
        s.update(["Docker"])
        assert s.update([]) == 1

    def test_empty_topics_after_reset_still_returns_one(self):
        s = SustainedAttention()
        s.update(["Docker"])
        s.update([])
        # After reset, calling again with empty topics -> still 1
        assert s.update([]) == 1


class TestSustainedAttentionOverlapThreshold:
    def test_exactly_50_percent_overlap_does_not_increment(self):
        # prev: {a, b}, curr: {a, c} -> 1/2 = 0.5; condition is > 0.5 -> reset
        s = SustainedAttention()
        s.update(["a", "b"])
        assert s.update(["a", "c"]) == 1

    def test_just_above_50_percent_increments(self):
        # prev: {a, b, c}, curr: {a, b, d} -> 2/3 = 0.667 > 0.5 -> increment
        s = SustainedAttention()
        s.update(["a", "b", "c"])
        assert s.update(["a", "b", "d"]) == 2

    def test_below_50_percent_resets(self):
        # prev: {a, b, c, d}, curr: {a, x, y, z} -> 1/4 = 0.25 -> reset
        s = SustainedAttention()
        s.update(["a", "b", "c", "d"])
        assert s.update(["a", "x", "y", "z"]) == 1


# ---------------------------------------------------------------------------
# SelectiveAttention — context filtering
# ---------------------------------------------------------------------------


class TestSelectiveAttentionBasic:
    def test_empty_topics_returns_all_fragments(self):
        fragments = ["alpha", "beta", "gamma"]
        result = SelectiveAttention.filter(fragments, [])
        assert result == fragments

    def test_empty_fragments_returns_empty_list(self):
        assert SelectiveAttention.filter([], ["Docker"]) == []

    def test_both_empty_returns_empty_list(self):
        assert SelectiveAttention.filter([], []) == []

    def test_matching_fragment_included(self):
        result = SelectiveAttention.filter(["Docker guide"], ["Docker"])
        assert result == ["Docker guide"]

    def test_non_matching_fragment_excluded(self):
        result = SelectiveAttention.filter(["weather today"], ["Docker"])
        assert result == []

    def test_returns_new_list_not_same_object(self):
        fragments = ["alpha", "beta"]
        result = SelectiveAttention.filter(fragments, [])
        assert result is not fragments


class TestSelectiveAttentionCaseInsensitive:
    def test_uppercase_fragment_matches_lowercase_topic(self):
        result = SelectiveAttention.filter(["DOCKER container"], ["docker"])
        assert len(result) == 1

    def test_lowercase_fragment_matches_uppercase_topic(self):
        result = SelectiveAttention.filter(["docker setup"], ["Docker"])
        assert len(result) == 1


class TestSelectiveAttentionMultipleTopics:
    def test_any_topic_match_passes_fragment(self):
        fragments = ["python tutorial", "java basics", "rust ownership"]
        result = SelectiveAttention.filter(fragments, ["python", "rust"])
        assert "python tutorial" in result
        assert "rust ownership" in result
        assert "java basics" not in result

    def test_no_topic_matches_any_fragment_returns_empty(self):
        fragments = ["weather report", "stock prices today"]
        result = SelectiveAttention.filter(fragments, ["kubernetes"])
        assert result == []

    def test_all_fragments_match_all_returned(self):
        fragments = ["Docker networking", "Docker volumes", "Docker compose"]
        result = SelectiveAttention.filter(fragments, ["Docker"])
        assert result == fragments


# ---------------------------------------------------------------------------
# ExecutiveAttention — tool prioritisation
# ---------------------------------------------------------------------------


class TestExecutiveAttentionNoUrgencyNoFileTopics:
    def test_zero_urgency_empty_topics_returns_empty(self):
        assert ExecutiveAttention.prioritise(0.0, []) == []

    def test_low_urgency_non_file_topic_returns_empty(self):
        assert ExecutiveAttention.prioritise(0.3, ["kubernetes"]) == []

    def test_urgency_at_boundary_0_5_does_not_trigger(self):
        # Condition is strictly > 0.5
        assert ExecutiveAttention.prioritise(0.5, []) == []


class TestExecutiveAttentionUrgencyTrigger:
    def test_urgency_just_above_threshold_adds_shell_and_file_read(self):
        priority = ExecutiveAttention.prioritise(0.51, [])
        assert "shell_exec" in priority
        assert "file_read" in priority

    def test_urgency_at_one_adds_shell_and_file_read(self):
        priority = ExecutiveAttention.prioritise(1.0, [])
        assert "shell_exec" in priority
        assert "file_read" in priority

    def test_high_urgency_does_not_add_file_write(self):
        # The file-topic branch is guarded by 'file_read not in priority',
        # which is already False when urgency branch fires first.
        priority = ExecutiveAttention.prioritise(0.9, [])
        assert "file_write" not in priority


class TestExecutiveAttentionFileTopics:
    def test_file_word_config_adds_file_read_and_write(self):
        priority = ExecutiveAttention.prioritise(0.1, ["config"])
        assert "file_read" in priority
        assert "file_write" in priority

    def test_file_word_log_adds_file_tools(self):
        priority = ExecutiveAttention.prioritise(0.1, ["log"])
        assert "file_read" in priority
        assert "file_write" in priority

    def test_file_topics_alone_do_not_add_shell_exec(self):
        priority = ExecutiveAttention.prioritise(0.1, ["directory"])
        assert "shell_exec" not in priority

    @pytest.mark.parametrize("file_word", sorted(_FILE_TOPIC_WORDS))
    def test_each_file_topic_word_triggers_file_tools(self, file_word):
        priority = ExecutiveAttention.prioritise(0.1, [file_word])
        assert "file_read" in priority, f"file_read missing for topic {file_word!r}"
        assert "file_write" in priority, f"file_write missing for topic {file_word!r}"


class TestExecutiveAttentionHighUrgencyWithFileTopics:
    def test_no_duplicate_file_read_when_urgency_and_file_topic_both_fire(self):
        # High urgency already adds file_read; file-topic branch is guarded
        # against adding it again.
        priority = ExecutiveAttention.prioritise(0.9, ["config", "file"])
        assert priority.count("file_read") == 1

    def test_file_write_not_added_when_urgency_fires_first(self):
        # Once urgency branch runs, 'file_read not in priority' is False,
        # so the file-topic branch does not add file_write.
        priority = ExecutiveAttention.prioritise(0.8, ["log"])
        assert "file_write" not in priority
        assert "shell_exec" in priority
        assert "file_read" in priority


# ---------------------------------------------------------------------------
# AttentionSystem — full pipeline
# ---------------------------------------------------------------------------


class TestAttentionSystemSubsystemTypes:
    def test_all_five_subsystems_instantiated_with_correct_types(self):
        attn = AttentionSystem()
        assert isinstance(attn._alerting, AlertingAttention)
        assert isinstance(attn._orienting, OrientingAttention)
        assert isinstance(attn._sustained, SustainedAttention)
        assert isinstance(attn._selective, SelectiveAttention)
        assert isinstance(attn._executive, ExecutiveAttention)


class TestAttentionSystemProcessReturnType:
    def test_process_returns_attention_state(self):
        state = AttentionSystem().process("error!")
        assert isinstance(state, AttentionState)

    def test_all_state_fields_populated(self):
        state = AttentionSystem().process("error in the config file")
        assert isinstance(state.urgency, float)
        assert isinstance(state.topics, list)
        assert isinstance(state.focus_duration, int)
        assert isinstance(state.priority_tools, list)
        assert isinstance(state.context_filter, list)


class TestAttentionSystemUrgencyPropagated:
    def test_urgent_text_propagates_nonzero_urgency(self):
        state = AttentionSystem().process("critical error failed immediately")
        assert state.urgency > 0.0

    def test_non_urgent_text_has_zero_urgency(self):
        state = AttentionSystem().process("what is two plus two")
        assert state.urgency == pytest.approx(0.0)

    def test_urgency_score_capped_at_one(self):
        state = AttentionSystem().process(
            "error critical urgent broken down failed security immediately asap emergency"
        )
        assert state.urgency == pytest.approx(1.0)


class TestAttentionSystemTopicsPropagated:
    def test_topics_extracted_from_capitalised_word(self):
        state = AttentionSystem().process("configure Docker networking")
        assert "Docker" in state.topics

    def test_topics_extracted_via_preposition(self):
        state = AttentionSystem().process("tell me about networking")
        assert "networking" in state.topics

    def test_context_filter_is_lowercase_version_of_topics(self):
        state = AttentionSystem().process("configure Docker networking")
        for topic, cf in zip(state.topics, state.context_filter, strict=False):
            assert cf == topic.lower()


class TestAttentionSystemEmptyInput:
    def test_empty_string_does_not_raise(self):
        state = AttentionSystem().process("")
        assert state.urgency == pytest.approx(0.0)
        assert state.topics == []
        assert state.focus_duration >= 1
        assert state.priority_tools == []
        assert state.context_filter == []

    def test_whitespace_only_input_does_not_raise(self):
        state = AttentionSystem().process("   ")
        assert state.urgency == pytest.approx(0.0)


class TestAttentionSystemFocusContinuity:
    def test_first_call_focus_duration_is_one(self):
        state = AttentionSystem().process("configure Docker networking")
        assert state.focus_duration == 1

    def test_same_topic_across_calls_increments_duration(self):
        attn = AttentionSystem()
        attn.process("tell me about Docker")
        state2 = attn.process("more about Docker please")
        assert state2.focus_duration >= 2

    def test_topic_change_resets_focus_duration(self):
        attn = AttentionSystem()
        attn.process("configure Docker networking")
        attn.process("configure Docker networking")
        # Completely different topic
        state3 = attn.process("explain quantum physics")
        assert state3.focus_duration == 1

    def test_three_identical_topics_accumulates_to_three(self):
        attn = AttentionSystem()
        attn.process("tell me about Docker")
        attn.process("tell me about Docker")
        state3 = attn.process("tell me about Docker")
        assert state3.focus_duration == 3

    def test_multiple_instances_do_not_share_sustained_state(self):
        attn1 = AttentionSystem()
        attn2 = AttentionSystem()
        for _ in range(5):
            attn1.process("tell me about Docker")
        # attn2 has seen nothing; its first call must return duration 1
        state = attn2.process("tell me about Docker")
        assert state.focus_duration == 1


class TestAttentionSystemHistoryParameter:
    def test_process_with_none_history_does_not_raise(self):
        state = AttentionSystem().process("error", None)
        assert isinstance(state, AttentionState)

    def test_process_with_empty_history_does_not_raise(self):
        state = AttentionSystem().process("error", [])
        assert isinstance(state, AttentionState)

    def test_process_with_populated_history_does_not_raise(self):
        history = [{"role": "user", "content": "hello"}]
        state = AttentionSystem().process("critical failure", history)
        assert isinstance(state, AttentionState)


class TestAttentionSystemToolPriority:
    def test_urgent_input_adds_shell_and_file_tools(self):
        # 3 urgency keywords in 5 words -> urgency 0.6 > 0.5 threshold
        state = AttentionSystem().process("server is down critical error")
        assert "shell_exec" in state.priority_tools
        assert "file_read" in state.priority_tools

    def test_file_topic_input_adds_file_tools(self):
        state = AttentionSystem().process("read the config file please")
        assert "file_read" in state.priority_tools

    def test_non_urgent_non_file_input_has_no_priority_tools(self):
        state = AttentionSystem().process("what is the weather today")
        assert state.priority_tools == []


# ---------------------------------------------------------------------------
# Edge cases: special characters, unicode, very long text
# ---------------------------------------------------------------------------


class TestEdgeCasesSpecialInputs:
    def test_very_long_text_all_urgency_words_caps_at_one(self):
        text = "error " * 1000
        score = AlertingAttention().score(text.strip())
        assert score == pytest.approx(1.0)

    def test_very_long_text_one_urgency_word_gives_tiny_score(self):
        # "error" + 999 non-urgency words -> 1/1000 = 0.001
        filler = " ".join(["hello"] * 999)
        text = "error " + filler
        score = AlertingAttention().score(text)
        assert score == pytest.approx(1 / 1000)

    def test_unicode_text_does_not_raise(self):
        # Non-Latin text is split by whitespace; no urgency keywords
        score = AlertingAttention().score("сервер недоступен")
        assert score == pytest.approx(0.0)

    def test_unicode_mixed_with_urgency_keyword(self):
        # "СРОЧНО error" -> ["СРОЧНО", "error"]; "error" matches
        score = AlertingAttention().score("СРОЧНО error")
        assert score == pytest.approx(0.5)

    def test_orientating_empty_string_does_not_raise(self):
        assert OrientingAttention().extract_topics("") == []

    def test_selective_filter_with_unicode_topics(self):
        fragments = ["Docker networking", "kubernetes setup"]
        result = SelectiveAttention.filter(fragments, ["Docker"])
        assert result == ["Docker networking"]

    def test_alerting_tab_separated_urgency_words(self):
        # Tab is a whitespace separator -> ["error", "tab"] -> "error" matches
        score = AlertingAttention().score("error\ttab")
        assert score == pytest.approx(0.5)

    def test_alerting_newline_separated_urgency_words(self):
        # Newline separates words -> "urgent" matches, "newline" does not
        score = AlertingAttention().score("urgent\nnewline")
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Keyword set membership verification
# ---------------------------------------------------------------------------


class TestKeywordSetMembership:
    def test_all_ten_urgency_keywords_present(self):
        expected = {
            "error",
            "critical",
            "urgent",
            "broken",
            "down",
            "failed",
            "security",
            "immediately",
            "asap",
            "emergency",
        }
        assert expected == _URGENCY_KEYWORDS

    def test_all_four_topic_prepositions_present(self):
        expected = {"about", "with", "for", "the"}
        assert expected == _TOPIC_PREPOSITIONS

    def test_all_eleven_file_topic_words_present(self):
        expected = {
            "file",
            "files",
            "directory",
            "folder",
            "path",
            "read",
            "write",
            "edit",
            "config",
            "log",
            "logs",
        }
        assert expected == _FILE_TOPIC_WORDS

    @pytest.mark.parametrize(
        "keyword",
        [
            "error",
            "critical",
            "urgent",
            "broken",
            "down",
            "failed",
            "security",
            "immediately",
            "asap",
            "emergency",
        ],
    )
    def test_each_urgency_keyword_scores_one_in_isolation(self, keyword):
        score = AlertingAttention().score(keyword)
        assert score == pytest.approx(1.0), f"Keyword {keyword!r} did not score 1.0"

    @pytest.mark.parametrize("prep", ["about", "with", "for", "the"])
    def test_each_preposition_extracts_following_word(self, prep):
        text = f"please {prep} testing"
        topics = OrientingAttention().extract_topics(text)
        assert "testing" in topics, f"Preposition {prep!r} did not extract following word"
