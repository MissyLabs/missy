"""Tests for missy.memory.synthesizer — unified memory synthesis."""

from __future__ import annotations

import pytest

from missy.memory.synthesizer import MemoryFragment, MemorySynthesizer


class TestAddFragments:
    def test_fragments_stored_correctly(self):
        synth = MemorySynthesizer()
        synth.add_fragments("conversation", ["hello world", "how are you"])
        assert len(synth._fragments) == 2
        assert synth._fragments[0].source == "conversation"
        assert synth._fragments[0].content == "hello world"
        assert synth._fragments[1].content == "how are you"

    def test_base_relevance_applied(self):
        synth = MemorySynthesizer()
        synth.add_fragments("learnings", ["lesson one"], base_relevance=0.8)
        assert synth._fragments[0].relevance == pytest.approx(0.8)

    def test_default_relevance(self):
        synth = MemorySynthesizer()
        synth.add_fragments("conversation", ["item"])
        assert synth._fragments[0].relevance == pytest.approx(0.5)

    def test_multiple_sources(self):
        synth = MemorySynthesizer()
        synth.add_fragments("conversation", ["a"])
        synth.add_fragments("learnings", ["b"])
        assert len(synth._fragments) == 2
        assert synth._fragments[0].source == "conversation"
        assert synth._fragments[1].source == "learnings"


class TestScoreRelevance:
    def test_query_relevant_fragments_score_higher(self):
        synth = MemorySynthesizer()
        relevant = MemoryFragment(source="test", content="Docker container setup guide")
        irrelevant = MemoryFragment(source="test", content="weather forecast today")

        score_relevant = synth.score_relevance(relevant, "How do I setup Docker?")
        score_irrelevant = synth.score_relevance(irrelevant, "How do I setup Docker?")

        assert score_relevant > score_irrelevant

    def test_empty_query_returns_base_relevance(self):
        synth = MemorySynthesizer()
        frag = MemoryFragment(source="test", content="some content", relevance=0.7)
        assert synth.score_relevance(frag, "") == pytest.approx(0.7)

    def test_full_overlap_gives_high_score(self):
        synth = MemorySynthesizer()
        frag = MemoryFragment(source="test", content="docker setup", relevance=1.0)
        score = synth.score_relevance(frag, "docker setup")
        # base * 0.5 + overlap * 0.5 = 1.0 * 0.5 + 1.0 * 0.5 = 1.0
        assert score == pytest.approx(1.0)

    def test_no_overlap_halves_base(self):
        synth = MemorySynthesizer()
        frag = MemoryFragment(source="test", content="alpha beta", relevance=0.6)
        score = synth.score_relevance(frag, "gamma delta")
        # base * 0.5 + 0 * 0.5 = 0.3
        assert score == pytest.approx(0.3)


class TestSynthesizeRanksByRelevance:
    def test_most_relevant_first(self):
        synth = MemorySynthesizer()
        synth.add_fragments("low", ["weather forecast today"], base_relevance=0.1)
        synth.add_fragments("high", ["Docker container networking guide"], base_relevance=0.9)
        synth.add_fragments("mid", ["some other topic here"], base_relevance=0.5)

        result = synth.synthesize("Docker networking")
        lines = result.strip().split("\n")
        # The Docker-related fragment should be first
        assert "Docker" in lines[0]


class TestSynthesizeRespectsTokenLimit:
    def test_output_stays_within_budget(self):
        synth = MemorySynthesizer(max_tokens=10)
        # Each fragment is about 10+ tokens, so only 1-2 should fit
        synth.add_fragments("a", ["a " * 50], base_relevance=0.9)
        synth.add_fragments("b", ["b " * 50], base_relevance=0.8)
        synth.add_fragments("c", ["c " * 50], base_relevance=0.7)

        result = synth.synthesize("anything")
        # With max_tokens=10 and ~4 chars per token, nothing large should fit
        assert len(result) <= 40 or result == ""

        synth2 = MemorySynthesizer(max_tokens=20)
        synth2.add_fragments("a", ["word " * 40], base_relevance=0.9)
        synth2.add_fragments("b", ["more " * 40], base_relevance=0.8)
        result2 = synth2.synthesize("anything")
        # Should have at most 1 fragment since each is ~200 chars = ~50 tokens
        lines = [line for line in result2.strip().split("\n") if line]
        assert len(lines) <= 1


class TestDeduplicateRemovesSimilar:
    def test_near_identical_removed(self):
        synth = MemorySynthesizer()
        frags = [
            MemoryFragment(source="a", content="docker container setup guide", relevance=0.5),
            MemoryFragment(source="b", content="docker container setup guide help", relevance=0.6),
        ]
        result = synth.deduplicate(frags, threshold=0.7)
        # These share most words, so one should be removed
        assert len(result) == 1
        # The higher relevance one should be kept
        assert result[0].relevance == pytest.approx(0.6)


class TestDeduplicateKeepsDifferent:
    def test_different_fragments_kept(self):
        synth = MemorySynthesizer()
        frags = [
            MemoryFragment(source="a", content="docker container networking", relevance=0.5),
            MemoryFragment(source="b", content="python flask web application", relevance=0.6),
        ]
        result = synth.deduplicate(frags, threshold=0.8)
        assert len(result) == 2


class TestSynthesizeEmpty:
    def test_no_fragments_returns_empty(self):
        synth = MemorySynthesizer()
        assert synth.synthesize("any query") == ""


class TestSourceLabelsInOutput:
    def test_output_includes_source_attribution(self):
        synth = MemorySynthesizer()
        synth.add_fragments("learnings", ["always check ports"])
        synth.add_fragments("conversation", ["discussed Docker setup"])

        result = synth.synthesize("Docker setup")
        assert "[learnings]" in result
        assert "[conversation]" in result
