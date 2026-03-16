"""Tests for missy.agent.summarizer — LLM-based summarization engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.agent.summarizer import Summarizer, _approx_tokens
from missy.memory.sqlite_store import ConversationTurn, SummaryRecord


class TestApproxTokens:
    def test_empty(self):
        assert _approx_tokens("") == 1

    def test_normal(self):
        assert _approx_tokens("a" * 400) == 100


class TestSummarizeTurns:
    def _make_summarizer(self, response_text="Summary text"):
        provider = MagicMock()
        resp = MagicMock()
        resp.content = response_text
        provider.chat.return_value = resp
        return Summarizer(provider)

    def _make_turns(self, n=3):
        return [
            ConversationTurn.new(f"sess", role="user", content=f"Turn {i}")
            for i in range(n)
        ]

    def test_normal_summarization(self):
        s = self._make_summarizer("Short summary")
        turns = self._make_turns(5)
        text, tier = s.summarize_turns(turns)
        assert text == "Short summary"
        assert tier == "normal"
        assert s.tier_counts["normal"] == 1

    def test_with_prior_summary(self):
        s = self._make_summarizer("New summary")
        turns = self._make_turns(2)
        text, tier = s.summarize_turns(turns, prior_summary="Old context")
        assert text == "New summary"
        # Verify prior_summary was included in the prompt
        call_args = s._provider.chat.call_args
        prompt = call_args.kwargs["messages"][0].content
        assert "Old context" in prompt

    def test_escalation_tier2_when_output_exceeds_input(self):
        """When tier 1 output is bigger than input, escalate to tier 2."""
        provider = MagicMock()
        # First call returns output longer than input
        resp1 = MagicMock()
        resp1.content = "x" * 10000  # way too long
        resp2 = MagicMock()
        resp2.content = "Compressed result"
        provider.chat.side_effect = [resp1, resp2]

        s = Summarizer(provider)
        turns = self._make_turns(2)  # very short input
        text, tier = s.summarize_turns(turns)
        assert tier == "aggressive"
        assert text == "Compressed result"
        assert s.tier_counts["aggressive"] == 1

    def test_escalation_tier3_on_llm_failure(self):
        """When LLM fails entirely, fall back to deterministic truncation."""
        provider = MagicMock()
        provider.chat.side_effect = RuntimeError("LLM down")

        s = Summarizer(provider)
        turns = self._make_turns(3)
        text, tier = s.summarize_turns(turns)
        assert tier == "fallback"
        assert "[TRUNCATED" in text
        assert s.tier_counts["fallback"] == 1

    def test_format_turns(self):
        turns = self._make_turns(2)
        result = Summarizer._format_turns(turns)
        assert "user: Turn 0" in result
        assert "user: Turn 1" in result


class TestSummarizeSummaries:
    def test_condensation(self):
        provider = MagicMock()
        resp = MagicMock()
        resp.content = "Condensed summary"
        provider.chat.return_value = resp

        s = Summarizer(provider)
        summaries = [
            SummaryRecord.new("sess", depth=0, content="Summary A"),
            SummaryRecord.new("sess", depth=0, content="Summary B"),
        ]
        text, tier = s.summarize_summaries(summaries)
        assert text == "Condensed summary"
        assert tier == "normal"


class TestTierCounts:
    def test_counts_accumulate(self):
        provider = MagicMock()
        resp = MagicMock()
        resp.content = "ok"
        provider.chat.return_value = resp

        s = Summarizer(provider)
        turns = [ConversationTurn.new("s", "user", "hi")]
        s.summarize_turns(turns)
        s.summarize_turns(turns)
        assert s.tier_counts["normal"] == 2
