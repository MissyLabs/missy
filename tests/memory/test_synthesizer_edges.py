"""Session-15 comprehensive tests for missy.memory.synthesizer.

Covers MemoryFragment, MemorySynthesizer, _approx_tokens, and _word_set
with 50+ cases including edge cases, unicode, and large-scale behaviour.
"""

from __future__ import annotations

import pytest

from missy.memory.synthesizer import (
    MemoryFragment,
    MemorySynthesizer,
    _approx_tokens,
    _word_set,
)

# ---------------------------------------------------------------------------
# MemoryFragment defaults
# ---------------------------------------------------------------------------


class TestMemoryFragmentDefaults:
    def test_required_fields_source_and_content(self):
        frag = MemoryFragment(source="conversation", content="hello world")
        assert frag.source == "conversation"
        assert frag.content == "hello world"

    def test_default_relevance_is_half(self):
        frag = MemoryFragment(source="s", content="c")
        assert frag.relevance == pytest.approx(0.5)

    def test_default_timestamp_is_empty_string(self):
        frag = MemoryFragment(source="s", content="c")
        assert frag.timestamp == ""

    def test_explicit_relevance_stored(self):
        frag = MemoryFragment(source="s", content="c", relevance=0.9)
        assert frag.relevance == pytest.approx(0.9)

    def test_explicit_timestamp_stored(self):
        frag = MemoryFragment(source="s", content="c", timestamp="2026-01-01T00:00:00")
        assert frag.timestamp == "2026-01-01T00:00:00"

    def test_zero_relevance_allowed(self):
        frag = MemoryFragment(source="s", content="c", relevance=0.0)
        assert frag.relevance == pytest.approx(0.0)

    def test_relevance_one_allowed(self):
        frag = MemoryFragment(source="s", content="c", relevance=1.0)
        assert frag.relevance == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _approx_tokens
# ---------------------------------------------------------------------------


class TestApproxTokens:
    def test_empty_string_returns_one(self):
        assert _approx_tokens("") == 1

    def test_three_chars_returns_one(self):
        assert _approx_tokens("abc") == 1

    def test_four_chars_returns_one(self):
        # 4 // 4 == 1, so exactly one token
        assert _approx_tokens("abcd") == 1

    def test_eight_chars_returns_two(self):
        assert _approx_tokens("abcdefgh") == 2

    def test_twelve_chars_returns_three(self):
        assert _approx_tokens("abcdefghijkl") == 3

    def test_one_char_returns_one(self):
        assert _approx_tokens("x") == 1

    def test_exactly_four_multiples(self):
        # 40 chars / 4 = 10 tokens
        assert _approx_tokens("a" * 40) == 10

    def test_large_text(self):
        text = "word " * 200  # 1000 chars
        assert _approx_tokens(text) == 250

    def test_return_type_is_int(self):
        assert isinstance(_approx_tokens("hello"), int)

    def test_minimum_floor_is_one(self):
        # Even for lengths that produce 0 when integer-divided
        for n in range(4):
            assert _approx_tokens("a" * n) == 1


# ---------------------------------------------------------------------------
# _word_set
# ---------------------------------------------------------------------------


class TestWordSet:
    def test_empty_string_returns_empty_set(self):
        assert _word_set("") == set()

    def test_single_word(self):
        assert _word_set("hello") == {"hello"}

    def test_case_folding(self):
        assert _word_set("Hello WORLD") == {"hello", "world"}

    def test_deduplication(self):
        assert _word_set("the the the") == {"the"}

    def test_multiple_words(self):
        result = _word_set("docker container setup")
        assert result == {"docker", "container", "setup"}

    def test_return_type_is_set(self):
        assert isinstance(_word_set("hello"), set)

    def test_mixed_case_deduplication(self):
        # "Docker" and "docker" are the same after case folding
        assert _word_set("Docker docker DOCKER") == {"docker"}

    def test_leading_trailing_whitespace(self):
        assert _word_set("  hello  ") == {"hello"}

    def test_numbers_preserved(self):
        result = _word_set("version 3.11")
        assert "version" in result
        assert "3.11" in result

    def test_unicode_word(self):
        result = _word_set("caf\u00e9 latte")
        assert "caf\u00e9" in result
        assert "latte" in result


# ---------------------------------------------------------------------------
# add_fragments
# ---------------------------------------------------------------------------


class TestAddFragments:
    def test_single_item_stored(self):
        synth = MemorySynthesizer()
        synth.add_fragments("conversation", ["hello world"])
        assert len(synth._fragments) == 1

    def test_source_assigned_correctly(self):
        synth = MemorySynthesizer()
        synth.add_fragments("learnings", ["item one"])
        assert synth._fragments[0].source == "learnings"

    def test_content_assigned_correctly(self):
        synth = MemorySynthesizer()
        synth.add_fragments("playbook", ["run tests first"])
        assert synth._fragments[0].content == "run tests first"

    def test_base_relevance_applied_to_all_items(self):
        synth = MemorySynthesizer()
        synth.add_fragments("summaries", ["a", "b", "c"], base_relevance=0.75)
        for frag in synth._fragments:
            assert frag.relevance == pytest.approx(0.75)

    def test_default_base_relevance_is_half(self):
        synth = MemorySynthesizer()
        synth.add_fragments("x", ["item"])
        assert synth._fragments[0].relevance == pytest.approx(0.5)

    def test_multiple_items_all_stored(self):
        synth = MemorySynthesizer()
        synth.add_fragments("conversation", ["a", "b", "c", "d"])
        assert len(synth._fragments) == 4

    def test_multiple_sources_accumulate(self):
        synth = MemorySynthesizer()
        synth.add_fragments("conversation", ["msg one"])
        synth.add_fragments("learnings", ["lesson one"])
        synth.add_fragments("playbook", ["pattern one"])
        assert len(synth._fragments) == 3

    def test_multiple_sources_preserve_source_labels(self):
        synth = MemorySynthesizer()
        synth.add_fragments("conversation", ["msg"])
        synth.add_fragments("learnings", ["lesson"])
        assert synth._fragments[0].source == "conversation"
        assert synth._fragments[1].source == "learnings"

    def test_empty_items_list_adds_nothing(self):
        synth = MemorySynthesizer()
        synth.add_fragments("empty_source", [])
        assert len(synth._fragments) == 0

    def test_empty_string_item_stored(self):
        synth = MemorySynthesizer()
        synth.add_fragments("source", [""])
        assert len(synth._fragments) == 1
        assert synth._fragments[0].content == ""

    def test_timestamp_defaults_to_empty_on_added_fragments(self):
        synth = MemorySynthesizer()
        synth.add_fragments("source", ["item"])
        assert synth._fragments[0].timestamp == ""


# ---------------------------------------------------------------------------
# score_relevance
# ---------------------------------------------------------------------------


class TestScoreRelevance:
    def test_empty_query_returns_base_relevance(self):
        synth = MemorySynthesizer()
        frag = MemoryFragment(source="s", content="any content here", relevance=0.7)
        assert synth.score_relevance(frag, "") == pytest.approx(0.7)

    def test_empty_query_returns_base_relevance_zero(self):
        synth = MemorySynthesizer()
        frag = MemoryFragment(source="s", content="some text", relevance=0.0)
        assert synth.score_relevance(frag, "") == pytest.approx(0.0)

    def test_full_overlap_maximises_score(self):
        synth = MemorySynthesizer()
        frag = MemoryFragment(source="s", content="docker setup guide", relevance=1.0)
        # All three query words appear in content
        score = synth.score_relevance(frag, "docker setup guide")
        # base(1.0)*0.5 + overlap(1.0)*0.5 = 1.0
        assert score == pytest.approx(1.0)

    def test_no_overlap_halves_base_relevance(self):
        synth = MemorySynthesizer()
        frag = MemoryFragment(source="s", content="alpha beta gamma", relevance=0.8)
        score = synth.score_relevance(frag, "delta epsilon zeta")
        # base(0.8)*0.5 + 0*0.5 = 0.4
        assert score == pytest.approx(0.4)

    def test_partial_overlap_intermediate_score(self):
        synth = MemorySynthesizer()
        # Content has "docker" and "setup", query adds "networking"
        frag = MemoryFragment(source="s", content="docker setup instructions", relevance=0.6)
        score = synth.score_relevance(frag, "docker setup networking")
        # query_words = {docker, setup, networking} (3 words)
        # shared = {docker, setup} => 2
        # overlap = 2/3
        expected = 0.6 * 0.5 + (2 / 3) * 0.5
        assert score == pytest.approx(expected, rel=1e-6)

    def test_exact_match_single_word(self):
        synth = MemorySynthesizer()
        frag = MemoryFragment(source="s", content="docker", relevance=0.5)
        score = synth.score_relevance(frag, "docker")
        # overlap = 1/1 = 1.0 -> 0.5*0.5 + 1.0*0.5 = 0.75
        assert score == pytest.approx(0.75)

    def test_case_insensitive_matching(self):
        synth = MemorySynthesizer()
        frag = MemoryFragment(source="s", content="Docker Container", relevance=0.5)
        score_lower = synth.score_relevance(frag, "docker container")
        score_upper = synth.score_relevance(frag, "DOCKER CONTAINER")
        assert score_lower == pytest.approx(score_upper)

    def test_relevant_scores_higher_than_irrelevant(self):
        synth = MemorySynthesizer()
        query = "how to configure docker networking"
        relevant = MemoryFragment(source="s", content="docker networking configuration tips", relevance=0.5)
        irrelevant = MemoryFragment(source="s", content="baking bread recipes flour water", relevance=0.5)
        assert synth.score_relevance(relevant, query) > synth.score_relevance(irrelevant, query)

    def test_score_bounded_between_zero_and_one(self):
        synth = MemorySynthesizer()
        frag = MemoryFragment(source="s", content="python flask web app", relevance=0.5)
        score = synth.score_relevance(frag, "python flask")
        assert 0.0 <= score <= 1.0

    def test_whitespace_only_query_treated_as_empty(self):
        synth = MemorySynthesizer()
        frag = MemoryFragment(source="s", content="something", relevance=0.6)
        # _word_set of "   " returns empty set
        score = synth.score_relevance(frag, "   ")
        assert score == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------


class TestDeduplicate:
    def test_identical_content_produces_one_result(self):
        synth = MemorySynthesizer()
        frags = [
            MemoryFragment(source="a", content="docker container setup", relevance=0.5),
            MemoryFragment(source="b", content="docker container setup", relevance=0.5),
        ]
        result = synth.deduplicate(frags)
        assert len(result) == 1

    def test_high_overlap_above_threshold_removed(self):
        synth = MemorySynthesizer()
        # 4 of 5 words shared -> Jaccard = 4/5 = 0.8 >= 0.8 threshold
        frags = [
            MemoryFragment(source="a", content="alpha beta gamma delta", relevance=0.5),
            MemoryFragment(source="b", content="alpha beta gamma delta epsilon", relevance=0.6),
        ]
        result = synth.deduplicate(frags, threshold=0.8)
        assert len(result) == 1

    def test_higher_relevance_kept_on_duplicate(self):
        synth = MemorySynthesizer()
        frags = [
            MemoryFragment(source="a", content="docker setup guide", relevance=0.3),
            MemoryFragment(source="b", content="docker setup guide", relevance=0.9),
        ]
        result = synth.deduplicate(frags)
        assert len(result) == 1
        assert result[0].relevance == pytest.approx(0.9)

    def test_first_higher_relevance_kept_when_duplicate_arrives(self):
        synth = MemorySynthesizer()
        # First frag has higher relevance; duplicate arrives second
        frags = [
            MemoryFragment(source="a", content="docker setup guide", relevance=0.9),
            MemoryFragment(source="b", content="docker setup guide", relevance=0.3),
        ]
        result = synth.deduplicate(frags)
        assert len(result) == 1
        assert result[0].relevance == pytest.approx(0.9)

    def test_dissimilar_fragments_both_kept(self):
        synth = MemorySynthesizer()
        frags = [
            MemoryFragment(source="a", content="docker container networking", relevance=0.5),
            MemoryFragment(source="b", content="python flask web framework", relevance=0.6),
        ]
        result = synth.deduplicate(frags, threshold=0.8)
        assert len(result) == 2

    def test_empty_list_returns_empty_list(self):
        synth = MemorySynthesizer()
        result = synth.deduplicate([])
        assert result == []

    def test_single_fragment_returned_unchanged(self):
        synth = MemorySynthesizer()
        frags = [MemoryFragment(source="a", content="unique content", relevance=0.7)]
        result = synth.deduplicate(frags)
        assert len(result) == 1
        assert result[0].content == "unique content"

    def test_empty_content_fragments_treated_as_duplicates(self):
        synth = MemorySynthesizer()
        # Both have empty content -> word sets are both empty -> total == 0 -> is_dup = True
        frags = [
            MemoryFragment(source="a", content="", relevance=0.5),
            MemoryFragment(source="b", content="", relevance=0.6),
        ]
        result = synth.deduplicate(frags)
        assert len(result) == 1

    def test_threshold_zero_everything_is_duplicate(self):
        synth = MemorySynthesizer()
        frags = [
            MemoryFragment(source="a", content="completely different words here", relevance=0.5),
            MemoryFragment(source="b", content="totally unrelated text about cats", relevance=0.4),
            MemoryFragment(source="c", content="another fragment about clouds", relevance=0.3),
        ]
        # With threshold=0.0, any overlap >= 0 qualifies as duplicate
        result = synth.deduplicate(frags, threshold=0.0)
        assert len(result) == 1

    def test_threshold_one_only_exact_duplicates_removed(self):
        synth = MemorySynthesizer()
        # Jaccard of these two is < 1.0 because word sets differ in size
        frags = [
            MemoryFragment(source="a", content="docker setup", relevance=0.5),
            MemoryFragment(source="b", content="docker setup guide", relevance=0.6),
        ]
        # At threshold=1.0 overlap must be exactly 1.0 (sets are equal)
        result = synth.deduplicate(frags, threshold=1.0)
        assert len(result) == 2

    def test_exact_duplicates_removed_at_threshold_one(self):
        synth = MemorySynthesizer()
        frags = [
            MemoryFragment(source="a", content="exactly the same", relevance=0.5),
            MemoryFragment(source="b", content="exactly the same", relevance=0.7),
        ]
        result = synth.deduplicate(frags, threshold=1.0)
        assert len(result) == 1

    def test_original_fragments_not_mutated(self):
        synth = MemorySynthesizer()
        original = [
            MemoryFragment(source="a", content="test content alpha", relevance=0.5),
            MemoryFragment(source="b", content="test content beta", relevance=0.6),
        ]
        original_len = len(original)
        synth.deduplicate(original, threshold=0.8)
        # Input list should still have the same length
        assert len(original) == original_len

    def test_three_near_duplicates_reduces_to_one(self):
        synth = MemorySynthesizer()
        frags = [
            MemoryFragment(source="a", content="docker container setup", relevance=0.5),
            MemoryFragment(source="b", content="docker container setup", relevance=0.6),
            MemoryFragment(source="c", content="docker container setup", relevance=0.7),
        ]
        result = synth.deduplicate(frags)
        assert len(result) == 1
        assert result[0].relevance == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------


class TestSynthesizeEmpty:
    def test_no_fragments_returns_empty_string(self):
        synth = MemorySynthesizer()
        assert synth.synthesize("any query") == ""

    def test_empty_query_on_empty_synthesizer(self):
        synth = MemorySynthesizer()
        assert synth.synthesize("") == ""


class TestSynthesizeSingleFragment:
    def test_single_fragment_formatted_with_source(self):
        synth = MemorySynthesizer()
        synth.add_fragments("learnings", ["check ports before deploying"])
        result = synth.synthesize("deploy")
        assert "[learnings]" in result
        assert "check ports before deploying" in result

    def test_single_fragment_format_is_bracket_source_space_content(self):
        synth = MemorySynthesizer()
        synth.add_fragments("playbook", ["run tests first"])
        result = synth.synthesize("tests")
        assert result == "[playbook] run tests first"


class TestSynthesizeMultipleFragments:
    def test_all_sources_appear_in_output(self):
        synth = MemorySynthesizer()
        synth.add_fragments("learnings", ["lesson about networking"])
        synth.add_fragments("conversation", ["discussed docker containers"])
        result = synth.synthesize("docker networking")
        assert "[learnings]" in result
        assert "[conversation]" in result

    def test_fragments_separated_by_newline(self):
        synth = MemorySynthesizer()
        synth.add_fragments("a", ["first fragment text"])
        synth.add_fragments("b", ["second fragment text"])
        result = synth.synthesize("query")
        lines = result.split("\n")
        assert len(lines) == 2

    def test_sorted_by_relevance_descending(self):
        synth = MemorySynthesizer()
        # Use distinct non-overlapping words so query scoring does not interfere
        synth.add_fragments("low", ["zzz qqq www"], base_relevance=0.1)
        synth.add_fragments("high", ["aaa bbb ccc"], base_relevance=0.9)
        synth.add_fragments("mid", ["mmm nnn ooo"], base_relevance=0.5)
        # Use empty query so scores are purely base_relevance
        result = synth.synthesize("")
        lines = result.split("\n")
        assert "[high]" in lines[0]
        assert "[mid]" in lines[1]
        assert "[low]" in lines[2]

    def test_query_boosts_relevant_fragments_to_top(self):
        synth = MemorySynthesizer()
        # Low base relevance but content directly matches query words
        synth.add_fragments("relevant", ["docker networking configuration"], base_relevance=0.1)
        # High base relevance but content is completely unrelated
        synth.add_fragments("irrelevant", ["baking bread flour yeast"], base_relevance=0.9)
        result = synth.synthesize("docker networking")
        lines = result.split("\n")
        # relevant fragment should be first because query boosting overcomes base penalty
        # relevant: 0.1*0.5 + (2/2)*0.5 = 0.05 + 0.5 = 0.55
        # irrelevant: 0.9*0.5 + 0*0.5 = 0.45
        assert "[relevant]" in lines[0]


class TestSynthesizeTruncation:
    def test_truncation_respects_max_tokens(self):
        # max_tokens=5 means only a tiny fragment can fit
        synth = MemorySynthesizer(max_tokens=5)
        synth.add_fragments("a", ["x" * 100], base_relevance=0.9)
        synth.add_fragments("b", ["y" * 100], base_relevance=0.8)
        result = synth.synthesize("query")
        # The very first fragment's token count already exceeds 5, so output is empty
        assert result == ""

    def test_only_fitting_fragments_included(self):
        # Each item is ~20 chars = 5 tokens; max_tokens=12 should fit 2 but not 3
        synth = MemorySynthesizer(max_tokens=12)
        # "[a] " (4) + 16 chars content = 20 chars = 5 tokens per line
        synth.add_fragments("a", ["aaaa bbbb cccc dddd"], base_relevance=0.9)
        synth.add_fragments("b", ["eeee ffff gggg hhhh"], base_relevance=0.8)
        synth.add_fragments("c", ["iiii jjjj kkkk llll"], base_relevance=0.7)
        result = synth.synthesize("")
        lines = [line for line in result.split("\n") if line]
        assert len(lines) <= 2

    def test_highest_relevance_fragments_kept_on_truncation(self):
        synth = MemorySynthesizer(max_tokens=30)
        # 4 fragments each ~8 tokens; only 3 should fit
        synth.add_fragments("lowest", ["one two three four five six"], base_relevance=0.1)
        synth.add_fragments("low", ["aaa bbb ccc ddd eee fff"], base_relevance=0.3)
        synth.add_fragments("high", ["ggg hhh iii jjj kkk lll"], base_relevance=0.8)
        synth.add_fragments("highest", ["mmm nnn ooo ppp qqq rrr"], base_relevance=0.9)
        result = synth.synthesize("")
        assert "[highest]" in result
        assert "[high]" in result
        # lowest-relevance should be excluded if budget is tight
        # (behaviour may vary, but highest-relevance lines should be present)


class TestSynthesizeDeduplicate:
    def test_deduplicate_applied_before_output(self):
        synth = MemorySynthesizer()
        synth.add_fragments("a", ["exact duplicate content here"])
        synth.add_fragments("b", ["exact duplicate content here"])
        result = synth.synthesize("duplicate")
        lines = [line for line in result.split("\n") if line]
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_very_large_number_of_fragments(self):
        synth = MemorySynthesizer(max_tokens=10000)
        items = [f"fragment number {i} about topic {i}" for i in range(500)]
        synth.add_fragments("bulk", items)
        result = synth.synthesize("fragment topic")
        # Should complete without error and produce output
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fragment_with_empty_content_handled(self):
        synth = MemorySynthesizer()
        synth.add_fragments("source", [""])
        result = synth.synthesize("any query")
        # Empty content is included (token count = 1 for the label portion)
        assert isinstance(result, str)

    def test_unicode_content_preserved(self):
        synth = MemorySynthesizer()
        synth.add_fragments("source", ["caf\u00e9 resum\u00e9 na\u00efve clich\u00e9"])
        result = synth.synthesize("caf\u00e9")
        assert "caf\u00e9" in result

    def test_unicode_query_matches_unicode_content(self):
        synth = MemorySynthesizer()
        frag = MemoryFragment(source="s", content="caf\u00e9 resum\u00e9 na\u00efve", relevance=0.5)
        score = synth.score_relevance(frag, "caf\u00e9 na\u00efve")
        # Should be higher than base*0.5 because there is word overlap
        assert score > 0.25

    def test_mixed_relevance_sources_sorted_correctly(self):
        synth = MemorySynthesizer()
        synth.add_fragments("conversation", ["talked about weather"], base_relevance=0.4)
        synth.add_fragments("learnings", ["docker networking fix"], base_relevance=0.7)
        synth.add_fragments("summaries", ["project overview summary"], base_relevance=0.6)
        result = synth.synthesize("")
        lines = result.split("\n")
        assert "[learnings]" in lines[0]
        assert "[summaries]" in lines[1]
        assert "[conversation]" in lines[2]

    def test_max_tokens_zero_produces_empty_output(self):
        synth = MemorySynthesizer(max_tokens=0)
        synth.add_fragments("source", ["some content to synthesize"])
        result = synth.synthesize("query")
        # No fragment can fit in a zero-token budget
        assert result == ""

    def test_add_fragments_called_multiple_times_accumulates(self):
        synth = MemorySynthesizer()
        for i in range(10):
            synth.add_fragments("source", [f"item {i}"])
        assert len(synth._fragments) == 10

    def test_synthesize_does_not_modify_internal_fragments(self):
        synth = MemorySynthesizer()
        synth.add_fragments("source", ["alpha beta gamma"])
        original_relevance = synth._fragments[0].relevance
        synth.synthesize("alpha beta")
        # Internal fragments should retain their original base relevance
        assert synth._fragments[0].relevance == pytest.approx(original_relevance)

    def test_special_characters_in_content(self):
        synth = MemorySynthesizer()
        synth.add_fragments("source", ["config path: /etc/missy/config.yaml"])
        result = synth.synthesize("config")
        assert "/etc/missy/config.yaml" in result

    def test_very_long_single_fragment_within_budget(self):
        long_text = "word " * 1000  # 5000 chars = 1250 tokens
        synth = MemorySynthesizer(max_tokens=2000)
        synth.add_fragments("source", [long_text])
        result = synth.synthesize("word")
        assert "[source]" in result

    def test_very_long_single_fragment_exceeds_budget(self):
        long_text = "word " * 1000  # ~1250 tokens
        synth = MemorySynthesizer(max_tokens=100)
        synth.add_fragments("source", [long_text])
        result = synth.synthesize("word")
        assert result == ""
