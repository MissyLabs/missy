"""Tests for missy.agent.summarizer — LLM-based summarization engine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from missy.agent.summarizer import Summarizer, _approx_tokens
from missy.memory.sqlite_store import ConversationTurn, SummaryRecord
from missy.providers.base import BaseProvider, CompletionResponse


def _mock_provider() -> MagicMock:
    """A provider mock constrained to the real BaseProvider interface.

    Using ``spec=BaseProvider`` means calling a nonexistent method (e.g. the
    old ``provider.chat(...)`` bug) raises ``AttributeError`` here exactly as
    it would against a real provider, instead of silently auto-vivifying a
    mock attribute and hiding the bug.
    """
    return MagicMock(spec=BaseProvider)


def _completion(text: str) -> CompletionResponse:
    return CompletionResponse(content=text, model="test-model", provider="test", usage={}, raw={})


class TestApproxTokens:
    def test_empty(self):
        assert _approx_tokens("") == 1

    def test_normal(self):
        assert _approx_tokens("a" * 400) == 100


class TestSummarizeTurns:
    def _make_summarizer(self, response_text="Summary text"):
        provider = _mock_provider()
        provider.complete.return_value = _completion(response_text)
        return Summarizer(provider)

    def _make_turns(self, n=3):
        return [ConversationTurn.new("sess", role="user", content=f"Turn {i}") for i in range(n)]

    def test_normal_summarization(self):
        s = self._make_summarizer("Short summary")
        turns = self._make_turns(5)
        text, tier = s.summarize_turns(turns)
        assert text == "Short summary"
        assert tier == "normal"
        assert s.tier_counts["normal"] == 1

    def test_calls_provider_complete_not_chat(self):
        """Regression test for SR-3.2: Summarizer must call BaseProvider.complete(),
        not a nonexistent chat() method — a bare MagicMock() previously masked
        this because it auto-creates any attribute, including .chat.
        """
        s = self._make_summarizer("Short summary")
        turns = self._make_turns(2)
        s.summarize_turns(turns)
        assert s._provider.complete.called
        assert s.tier_counts["fallback"] == 0

    def test_real_provider_interface_rejects_chat_call(self):
        """Sanity check that spec=BaseProvider actually enforces the contract."""
        provider = _mock_provider()
        with pytest.raises(AttributeError):
            provider.chat(messages=[])

    def test_with_prior_summary(self):
        s = self._make_summarizer("New summary")
        turns = self._make_turns(2)
        text, tier = s.summarize_turns(turns, prior_summary="Old context")
        assert text == "New summary"
        # Verify prior_summary was included in the prompt
        call_args = s._provider.complete.call_args
        prompt = call_args.args[0][0].content
        assert "Old context" in prompt

    def test_escalation_tier2_when_output_exceeds_input(self):
        """When tier 1 output is bigger than input, escalate to tier 2."""
        provider = _mock_provider()
        # First call returns output longer than input
        resp1 = _completion("x" * 10000)  # way too long
        resp2 = _completion("Compressed result")
        provider.complete.side_effect = [resp1, resp2]

        s = Summarizer(provider)
        turns = self._make_turns(2)  # very short input
        text, tier = s.summarize_turns(turns)
        assert tier == "aggressive"
        assert text == "Compressed result"
        assert s.tier_counts["aggressive"] == 1

    def test_escalation_tier3_on_llm_failure(self):
        """When LLM fails entirely, fall back to deterministic truncation."""
        provider = _mock_provider()
        provider.complete.side_effect = RuntimeError("LLM down")

        s = Summarizer(provider)
        turns = self._make_turns(3)
        text, tier = s.summarize_turns(turns)
        assert tier == "fallback"
        assert "[TRUNCATED" in text
        assert s.tier_counts["fallback"] == 1

    def test_escalation_tier3_preserves_new_content_over_large_prior_summary(self):
        """Regression: Tier 3 used to truncate the *entire assembled prompt*
        (fixed instructional header + prior_context + transcript) from the
        front. A real prior summary threaded forward across a chain of
        compaction passes can legitimately be tens of thousands of
        characters -- easily larger than the whole target_tokens*4 budget
        on its own -- so the truncated result could end up being 100%
        stale header/prior-summary boilerplate with zero characters of the
        actual new conversation this call exists to summarize, while still
        being tagged "[TRUNCATED -- summarization failed]" as if it were a
        normal (if abbreviated) summary. Must preserve at least some of the
        new content instead.
        """
        provider = _mock_provider()
        provider.complete.side_effect = RuntimeError("LLM down")

        s = Summarizer(provider)
        turns = [ConversationTurn.new("sess", role="user", content="X" * 20000)]
        prior_summary = "P" * 4900

        text, tier = s.summarize_turns(turns, prior_summary=prior_summary, target_tokens=1200)

        assert tier == "fallback"
        assert "X" in text, "fallback dropped 100% of the new conversation content"

    def test_format_turns(self):
        turns = self._make_turns(2)
        result = Summarizer._format_turns(turns)
        assert "user: Turn 0" in result
        assert "user: Turn 1" in result


class TestSummarizeSummaries:
    def test_condensation(self):
        provider = _mock_provider()
        provider.complete.return_value = _completion("Condensed summary")

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
        provider = _mock_provider()
        provider.complete.return_value = _completion("ok")

        s = Summarizer(provider)
        turns = [ConversationTurn.new("s", "user", "hi")]
        s.summarize_turns(turns)
        s.summarize_turns(turns)
        assert s.tier_counts["normal"] == 2
