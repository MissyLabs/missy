"""Tests for missy.agent.attention — brain-inspired attention system."""

from __future__ import annotations

import pytest

from missy.agent.attention import (
    AlertingAttention,
    AttentionState,
    AttentionSystem,
    ExecutiveAttention,
    OrientingAttention,
    SelectiveAttention,
    SustainedAttention,
)


class TestAlertingDetectsUrgency:
    def test_server_down_scores_high(self):
        alerting = AlertingAttention()
        score = alerting.score("server is down!")
        # "down" is an urgency keyword, 1 match in 3 words -> 0.33+
        assert score > 0.2

    def test_emergency_scores_high(self):
        alerting = AlertingAttention()
        score = alerting.score("critical error failed immediately")
        # 4 urgency keywords in 4 words -> 1.0
        assert score == pytest.approx(1.0)


class TestAlertingNormalMessage:
    def test_normal_message_scores_low(self):
        alerting = AlertingAttention()
        score = alerting.score("what time is it")
        assert score == pytest.approx(0.0)

    def test_empty_input(self):
        alerting = AlertingAttention()
        assert alerting.score("") == pytest.approx(0.0)


class TestOrientingExtractsTopics:
    def test_extracts_capitalised_words(self):
        orienting = OrientingAttention()
        topics = orienting.extract_topics("Tell me about Docker containers")
        assert "Docker" in topics

    def test_extracts_after_prepositions(self):
        orienting = OrientingAttention()
        topics = orienting.extract_topics("Tell me about the networking stack")
        # "the" is a topic preposition, so "networking" should be extracted
        assert "networking" in topics

    def test_no_topics_from_simple_query(self):
        orienting = OrientingAttention()
        topics = orienting.extract_topics("hello")
        assert topics == []


class TestSustainedIncrementsOnSameTopic:
    def test_repeated_topic_increases_duration(self):
        sustained = SustainedAttention()
        sustained.update(["Docker"])
        d2 = sustained.update(["Docker"])
        d3 = sustained.update(["Docker"])
        assert d2 == 2
        assert d3 == 3


class TestSustainedResetsOnTopicChange:
    def test_new_topic_resets_to_one(self):
        sustained = SustainedAttention()
        sustained.update(["Docker"])
        sustained.update(["Docker"])
        d = sustained.update(["Python", "Flask"])
        assert d == 1


class TestSelectiveFiltersRelevant:
    def test_only_matching_fragments_pass(self):
        fragments = [
            "Docker container networking guide",
            "weather forecast for tomorrow",
            "Docker image build process",
        ]
        result = SelectiveAttention.filter(fragments, ["Docker"])
        assert len(result) == 2
        assert all("Docker" in f for f in result)

    def test_empty_topics_returns_all(self):
        fragments = ["a", "b", "c"]
        result = SelectiveAttention.filter(fragments, [])
        assert len(result) == 3


class TestExecutiveUrgencyPriority:
    def test_urgent_input_prioritises_shell_file(self):
        priority = ExecutiveAttention.prioritise(0.8, ["server"])
        assert "shell_exec" in priority
        assert "file_read" in priority

    def test_low_urgency_no_priority(self):
        priority = ExecutiveAttention.prioritise(0.1, ["general"])
        assert priority == []

    def test_file_topics_add_file_tools(self):
        priority = ExecutiveAttention.prioritise(0.1, ["config", "file"])
        assert "file_read" in priority
        assert "file_write" in priority


class TestFullPipeline:
    def test_process_returns_valid_attention_state(self):
        attn = AttentionSystem()
        state = attn.process("The server is down! Fix it immediately!")
        assert isinstance(state, AttentionState)
        assert state.urgency > 0.0
        assert isinstance(state.topics, list)
        assert state.focus_duration >= 1
        assert isinstance(state.priority_tools, list)
        assert isinstance(state.context_filter, list)

    def test_process_with_history(self):
        attn = AttentionSystem()
        history = [{"role": "user", "content": "hello"}]
        state = attn.process("Tell me about Docker", history)
        assert isinstance(state, AttentionState)

    def test_sustained_across_multiple_calls(self):
        attn = AttentionSystem()
        state1 = attn.process("Tell me about Docker containers")
        state2 = attn.process("More about Docker networking")
        # Both mention Docker, so focus should increase
        assert state2.focus_duration >= state1.focus_duration
