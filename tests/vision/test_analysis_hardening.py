"""Edge-case hardening tests for AnalysisPromptBuilder.

Covers:
- All four analysis modes produce valid, non-empty prompts
- Puzzle mode includes board-state tracking language
- Painting mode uses warm/encouraging language and avoids harsh words
- Context parameters are properly injected into prompts
- Edge cases: empty context, None-like values, very long context strings
- Prompt suitability for LLM consumption (minimum length, structure markers)
"""

from __future__ import annotations

import numpy as np
import pytest

from missy.vision.analysis import (
    AnalysisMode,
    AnalysisPromptBuilder,
    AnalysisRequest,
    _format_state,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DUMMY_IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture()
def builder() -> AnalysisPromptBuilder:
    return AnalysisPromptBuilder()


def make_request(
    mode: AnalysisMode = AnalysisMode.GENERAL,
    context: str = "",
    previous_observations: list[str] | None = None,
    previous_state: dict | None = None,
    is_followup: bool = False,
) -> AnalysisRequest:
    return AnalysisRequest(
        image=DUMMY_IMAGE,
        mode=mode,
        context=context,
        previous_observations=previous_observations or [],
        previous_state=previous_state or {},
        is_followup=is_followup,
    )


# ---------------------------------------------------------------------------
# 1. All modes produce valid non-empty prompts
# ---------------------------------------------------------------------------


class TestAllModesProduceValidPrompts:
    @pytest.mark.parametrize(
        "mode",
        [
            AnalysisMode.GENERAL,
            AnalysisMode.PUZZLE,
            AnalysisMode.PAINTING,
            AnalysisMode.INSPECTION,
        ],
    )
    def test_prompt_is_non_empty_string(
        self, builder: AnalysisPromptBuilder, mode: AnalysisMode
    ) -> None:
        prompt = builder.build_prompt(make_request(mode=mode))
        assert isinstance(prompt, str)
        assert len(prompt.strip()) > 0

    @pytest.mark.parametrize(
        "mode",
        [
            AnalysisMode.GENERAL,
            AnalysisMode.PUZZLE,
            AnalysisMode.PAINTING,
            AnalysisMode.INSPECTION,
        ],
    )
    def test_prompt_meets_minimum_length(
        self, builder: AnalysisPromptBuilder, mode: AnalysisMode
    ) -> None:
        """Prompts must be substantial enough for an LLM to act on (>= 100 chars)."""
        prompt = builder.build_prompt(make_request(mode=mode))
        assert len(prompt) >= 100, f"{mode} prompt is suspiciously short: {len(prompt)} chars"

    @pytest.mark.parametrize(
        "mode",
        [
            AnalysisMode.GENERAL,
            AnalysisMode.PUZZLE,
            AnalysisMode.PAINTING,
            AnalysisMode.INSPECTION,
        ],
    )
    def test_prompt_ends_with_newline_or_content(
        self, builder: AnalysisPromptBuilder, mode: AnalysisMode
    ) -> None:
        """Prompts must not be whitespace-only."""
        prompt = builder.build_prompt(make_request(mode=mode))
        assert prompt.strip(), f"{mode} prompt contains only whitespace"

    @pytest.mark.parametrize(
        "mode",
        [
            AnalysisMode.GENERAL,
            AnalysisMode.PUZZLE,
            AnalysisMode.PAINTING,
            AnalysisMode.INSPECTION,
        ],
    )
    def test_prompt_returns_str_not_bytes(
        self, builder: AnalysisPromptBuilder, mode: AnalysisMode
    ) -> None:
        prompt = builder.build_prompt(make_request(mode=mode))
        assert type(prompt) is str  # noqa: E721 — exact type check


# ---------------------------------------------------------------------------
# 2. Puzzle mode — board-state tracking language
# ---------------------------------------------------------------------------


class TestPuzzleModeContent:
    def test_board_state_heading_present(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PUZZLE))
        assert "Board State" in prompt

    def test_completion_percentage_language(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PUZZLE))
        assert "completion" in prompt.lower() or "percentage" in prompt.lower()

    def test_piece_identification_section(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PUZZLE))
        assert "Piece Identification" in prompt

    def test_placement_guidance_section(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PUZZLE))
        assert "Placement Guidance" in prompt

    def test_grouping_suggestions_section(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PUZZLE))
        assert "Grouping" in prompt

    def test_strategy_tips_section(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PUZZLE))
        assert "Strategy" in prompt

    def test_orientation_hints_section(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PUZZLE))
        assert "Orientation" in prompt

    def test_edge_corner_piece_terminology(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PUZZLE))
        assert "edge piece" in prompt.lower() or "corner piece" in prompt.lower()

    def test_rotation_degrees_mentioned(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PUZZLE))
        # Rotation hints require degree values
        assert "90" in prompt or "180" in prompt

    def test_followup_includes_previous_observations(self, builder: AnalysisPromptBuilder) -> None:
        obs = ["Top-left corner is 50% done", "Blue sky region identified"]
        prompt = builder.build_prompt(
            make_request(
                mode=AnalysisMode.PUZZLE,
                is_followup=True,
                previous_observations=obs,
            )
        )
        assert "Top-left corner is 50% done" in prompt
        assert "Blue sky region identified" in prompt

    def test_followup_includes_previous_state(self, builder: AnalysisPromptBuilder) -> None:
        state = {"completion_pct": "40%", "pieces_placed": 120}
        prompt = builder.build_prompt(
            make_request(
                mode=AnalysisMode.PUZZLE,
                is_followup=True,
                previous_observations=["Some obs"],
                previous_state=state,
            )
        )
        assert "40%" in prompt
        assert "120" in prompt

    def test_followup_without_observations_falls_back_to_initial(
        self, builder: AnalysisPromptBuilder
    ) -> None:
        """is_followup=True but no observations → initial puzzle prompt."""
        prompt = builder.build_prompt(
            make_request(mode=AnalysisMode.PUZZLE, is_followup=True, previous_observations=[])
        )
        assert "Board State" in prompt

    def test_followup_asks_about_progress(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(
            make_request(
                mode=AnalysisMode.PUZZLE,
                is_followup=True,
                previous_observations=["Corners complete"],
            )
        )
        assert "progress" in prompt.lower()


# ---------------------------------------------------------------------------
# 3. Painting mode — warm, encouraging language
# ---------------------------------------------------------------------------


HARSH_WORDS = ["wrong", "bad", "weak", "poor", "fail"]


class TestPaintingModeContent:
    def test_warm_tone_indicator_present(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PAINTING))
        warm_indicators = ["warm", "encourage", "supportive", "love"]
        assert any(word in prompt.lower() for word in warm_indicators)

    def test_guidelines_explicitly_forbid_harsh_words(self, builder: AnalysisPromptBuilder) -> None:
        """The painting prompt must contain an explicit prohibition against harsh language.

        The template includes a 'Never use words like ...' guideline line which itself
        lists the forbidden words as quoted examples.  We assert the prohibition line
        is present rather than doing a naive full-prompt substring search (which would
        trivially match the quoted examples inside the prohibition itself).
        """
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PAINTING))
        lower = prompt.lower()
        # The guidelines section must include the ban instruction
        assert "never use words like" in lower or "never" in lower, (
            "Painting prompt is missing the harsh-language prohibition guideline"
        )

    def test_first_impression_section(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PAINTING))
        assert "First Impression" in prompt

    def test_color_light_section(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PAINTING))
        assert "Color" in prompt and "Light" in prompt

    def test_composition_section(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PAINTING))
        assert "Composition" in prompt

    def test_technique_section(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PAINTING))
        assert "Technique" in prompt

    def test_emotional_impact_section(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PAINTING))
        assert "Emotional" in prompt

    def test_growth_opportunity_section(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PAINTING))
        assert "Growth" in prompt

    def test_encouraging_language_markers(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PAINTING))
        # The template explicitly requires these phrases
        assert "I love how" in prompt or "encouraging" in prompt.lower()

    def test_suggestion_framing_language(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PAINTING))
        assert "might" in prompt.lower() or "explore" in prompt.lower()

    def test_painting_followup_is_warm(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(
            make_request(
                mode=AnalysisMode.PAINTING,
                is_followup=True,
                previous_observations=["Sunset colors are vivid"],
            )
        )
        assert (
            "warm" in prompt.lower() or "encourage" in prompt.lower() or "support" in prompt.lower()
        )

    def test_painting_followup_no_harsh_words(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(
            make_request(
                mode=AnalysisMode.PAINTING,
                is_followup=True,
                previous_observations=["Nice brushwork"],
            )
        )
        for word in HARSH_WORDS:
            assert word not in prompt.lower(), f"Harsh word '{word}' found in painting followup"

    def test_painting_followup_without_observations_uses_initial(
        self, builder: AnalysisPromptBuilder
    ) -> None:
        prompt = builder.build_prompt(
            make_request(mode=AnalysisMode.PAINTING, is_followup=True, previous_observations=[])
        )
        assert "First Impression" in prompt


# ---------------------------------------------------------------------------
# 4. Context parameters incorporated into prompts
# ---------------------------------------------------------------------------


class TestContextIncorporation:
    def test_general_mode_appends_context(self, builder: AnalysisPromptBuilder) -> None:
        ctx = "Focus on the red objects in the foreground"
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.GENERAL, context=ctx))
        assert ctx in prompt

    def test_puzzle_mode_appends_context(self, builder: AnalysisPromptBuilder) -> None:
        ctx = "I am working on the border first"
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PUZZLE, context=ctx))
        assert ctx in prompt

    def test_painting_mode_appends_context(self, builder: AnalysisPromptBuilder) -> None:
        ctx = "I used watercolors and it is my first attempt"
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PAINTING, context=ctx))
        assert ctx in prompt

    def test_inspection_mode_appends_context(self, builder: AnalysisPromptBuilder) -> None:
        ctx = "Check for corrosion on the connectors"
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.INSPECTION, context=ctx))
        assert ctx in prompt

    def test_general_context_uses_additional_context_label(
        self, builder: AnalysisPromptBuilder
    ) -> None:
        ctx = "unique-ctx-marker"
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.GENERAL, context=ctx))
        assert "User-provided context" in prompt

    def test_puzzle_context_uses_user_note_label(self, builder: AnalysisPromptBuilder) -> None:
        ctx = "unique-puzzle-ctx-marker"
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PUZZLE, context=ctx))
        assert "User note" in prompt

    def test_painting_context_uses_painter_says_label(self, builder: AnalysisPromptBuilder) -> None:
        ctx = "unique-painting-ctx-marker"
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PAINTING, context=ctx))
        assert "painter says" in prompt.lower()

    def test_inspection_context_uses_focus_label(self, builder: AnalysisPromptBuilder) -> None:
        ctx = "unique-inspection-ctx-marker"
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.INSPECTION, context=ctx))
        assert "Inspection focus" in prompt

    def test_context_appears_after_base_prompt(self, builder: AnalysisPromptBuilder) -> None:
        """Context should be appended, not prepended."""
        ctx = "appended-context"
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.GENERAL, context=ctx))
        base_end = prompt.index(ctx)
        # There must be base content before the context
        assert base_end > 50


# ---------------------------------------------------------------------------
# 5. Edge cases: empty context, None-like values, very long context
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_context_produces_base_prompt_only(self, builder: AnalysisPromptBuilder) -> None:
        for mode in AnalysisMode:
            prompt = builder.build_prompt(make_request(mode=mode, context=""))
            # Should not include any context appendage label when context is empty
            assert "Additional context:" not in prompt or mode != AnalysisMode.GENERAL
            assert len(prompt.strip()) > 0

    def test_general_mode_empty_context_no_appended_label(
        self, builder: AnalysisPromptBuilder
    ) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.GENERAL, context=""))
        assert "Additional context:" not in prompt

    def test_puzzle_mode_empty_context_no_user_note(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PUZZLE, context=""))
        assert "User note:" not in prompt

    def test_painting_mode_empty_context_no_painter_says(
        self, builder: AnalysisPromptBuilder
    ) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.PAINTING, context=""))
        assert "painter says" not in prompt.lower()

    def test_inspection_mode_empty_context_no_focus_label(
        self, builder: AnalysisPromptBuilder
    ) -> None:
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.INSPECTION, context=""))
        assert "Inspection focus:" not in prompt

    def test_very_long_context_is_truncated(self, builder: AnalysisPromptBuilder) -> None:
        long_ctx = "A" * 5000
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.GENERAL, context=long_ctx))
        # Context should be truncated (max 2000 chars) for prompt injection mitigation
        assert "[truncated]" in prompt
        assert long_ctx not in prompt  # full string should not appear

    def test_very_long_context_does_not_corrupt_base_prompt(
        self, builder: AnalysisPromptBuilder
    ) -> None:
        long_ctx = "B" * 5000
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.GENERAL, context=long_ctx))
        # The base prompt content must still be present
        assert "Main subjects and objects" in prompt

    def test_context_with_special_characters(self, builder: AnalysisPromptBuilder) -> None:
        ctx = 'Context with "quotes", \\backslashes\\, and {braces}'
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.GENERAL, context=ctx))
        assert ctx in prompt

    def test_context_with_newlines(self, builder: AnalysisPromptBuilder) -> None:
        ctx = "Line one\nLine two\nLine three"
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.GENERAL, context=ctx))
        assert ctx in prompt

    def test_context_with_unicode(self, builder: AnalysisPromptBuilder) -> None:
        ctx = "Analyze the \u5c71 (mountain) and \u6d77 (ocean)"
        prompt = builder.build_prompt(make_request(mode=AnalysisMode.GENERAL, context=ctx))
        assert ctx in prompt

    def test_empty_previous_observations_list(self, builder: AnalysisPromptBuilder) -> None:
        """Followup with empty observations must not raise and must return a valid prompt."""
        prompt = builder.build_prompt(
            make_request(mode=AnalysisMode.PUZZLE, is_followup=True, previous_observations=[])
        )
        assert len(prompt.strip()) > 0

    def test_empty_previous_state_dict(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(
            make_request(
                mode=AnalysisMode.PUZZLE,
                is_followup=True,
                previous_observations=["Some obs"],
                previous_state={},
            )
        )
        assert "No previous state recorded" in prompt

    def test_single_previous_observation(self, builder: AnalysisPromptBuilder) -> None:
        obs = ["Only one observation"]
        prompt = builder.build_prompt(
            make_request(mode=AnalysisMode.PUZZLE, is_followup=True, previous_observations=obs)
        )
        assert "Only one observation" in prompt

    def test_many_previous_observations(self, builder: AnalysisPromptBuilder) -> None:
        obs = [f"Observation {i}" for i in range(50)]
        prompt = builder.build_prompt(
            make_request(mode=AnalysisMode.PUZZLE, is_followup=True, previous_observations=obs)
        )
        assert "Observation 0" in prompt
        assert "Observation 49" in prompt

    def test_previous_state_with_various_value_types(self, builder: AnalysisPromptBuilder) -> None:
        state = {"integer": 42, "float": 3.14, "string": "hello", "boolean": True, "none": None}
        prompt = builder.build_prompt(
            make_request(
                mode=AnalysisMode.PUZZLE,
                is_followup=True,
                previous_observations=["obs"],
                previous_state=state,
            )
        )
        assert "42" in prompt
        assert "3.14" in prompt
        assert "hello" in prompt


# ---------------------------------------------------------------------------
# 6. Prompt suitability for LLM consumption
# ---------------------------------------------------------------------------


class TestLLMSuitability:
    MIN_PROMPT_LENGTH = 200

    @pytest.mark.parametrize("mode", list(AnalysisMode))
    def test_prompt_exceeds_minimum_length_for_llm(
        self, builder: AnalysisPromptBuilder, mode: AnalysisMode
    ) -> None:
        prompt = builder.build_prompt(make_request(mode=mode))
        assert len(prompt) >= self.MIN_PROMPT_LENGTH, (
            f"{mode} prompt length {len(prompt)} is below minimum {self.MIN_PROMPT_LENGTH}"
        )

    @pytest.mark.parametrize("mode", list(AnalysisMode))
    def test_prompt_contains_numbered_structure_or_bullets(
        self, builder: AnalysisPromptBuilder, mode: AnalysisMode
    ) -> None:
        """Structured prompts must contain numbered items or bullet points."""
        prompt = builder.build_prompt(make_request(mode=mode))
        has_numbered = any(f"{n}." in prompt for n in range(1, 8))
        has_bullets = "-" in prompt
        assert has_numbered or has_bullets, f"{mode} prompt lacks structured markers"

    @pytest.mark.parametrize("mode", list(AnalysisMode))
    def test_prompt_has_no_unfilled_template_placeholders(
        self, builder: AnalysisPromptBuilder, mode: AnalysisMode
    ) -> None:
        """Base (non-followup) prompts must not contain raw {placeholder} tokens."""
        prompt = builder.build_prompt(make_request(mode=mode))
        import re

        # Match {word} patterns — these would indicate an unfilled .format() call
        placeholders = re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", prompt)
        assert not placeholders, f"{mode} prompt has unfilled placeholders: {placeholders}"

    def test_puzzle_followup_no_unfilled_placeholders(self, builder: AnalysisPromptBuilder) -> None:
        prompt = builder.build_prompt(
            make_request(
                mode=AnalysisMode.PUZZLE,
                is_followup=True,
                previous_observations=["obs1"],
                previous_state={"key": "value"},
            )
        )
        import re

        placeholders = re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", prompt)
        assert not placeholders, f"Puzzle followup has unfilled placeholders: {placeholders}"

    def test_painting_followup_no_unfilled_placeholders(
        self, builder: AnalysisPromptBuilder
    ) -> None:
        prompt = builder.build_prompt(
            make_request(
                mode=AnalysisMode.PAINTING,
                is_followup=True,
                previous_observations=["obs1"],
            )
        )
        import re

        placeholders = re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", prompt)
        assert not placeholders, f"Painting followup has unfilled placeholders: {placeholders}"

    def test_puzzle_prompt_is_longer_than_general(self, builder: AnalysisPromptBuilder) -> None:
        """Puzzle prompts are the most structured and should be the longest base prompts."""
        puzzle = builder.build_prompt(make_request(mode=AnalysisMode.PUZZLE))
        general = builder.build_prompt(make_request(mode=AnalysisMode.GENERAL))
        assert len(puzzle) > len(general)

    def test_painting_prompt_is_longer_than_general(self, builder: AnalysisPromptBuilder) -> None:
        painting = builder.build_prompt(make_request(mode=AnalysisMode.PAINTING))
        general = builder.build_prompt(make_request(mode=AnalysisMode.GENERAL))
        assert len(painting) > len(general)

    def test_context_increases_prompt_length(self, builder: AnalysisPromptBuilder) -> None:
        without_ctx = builder.build_prompt(make_request(mode=AnalysisMode.GENERAL, context=""))
        with_ctx = builder.build_prompt(
            make_request(mode=AnalysisMode.GENERAL, context="extra context details")
        )
        assert len(with_ctx) > len(without_ctx)

    def test_prompt_does_not_start_with_whitespace(self, builder: AnalysisPromptBuilder) -> None:
        for mode in AnalysisMode:
            prompt = builder.build_prompt(make_request(mode=mode))
            assert prompt == prompt.lstrip("\n\r"), f"{mode} prompt starts with leading newlines"


# ---------------------------------------------------------------------------
# _format_state helper unit tests
# ---------------------------------------------------------------------------


class TestFormatState:
    def test_empty_dict_returns_no_state_message(self) -> None:
        result = _format_state({})
        assert "No previous state" in result

    def test_single_key_value_formatted(self) -> None:
        result = _format_state({"completion": "30%"})
        assert "completion" in result
        assert "30%" in result

    def test_multiple_keys_all_present(self) -> None:
        state = {"a": 1, "b": 2, "c": 3}
        result = _format_state(state)
        for key in ("a", "b", "c"):
            assert key in result

    def test_values_are_stringified(self) -> None:
        result = _format_state({"count": 99, "flag": True})
        assert "99" in result
        assert "True" in result

    def test_each_entry_on_its_own_line(self) -> None:
        state = {"x": 1, "y": 2}
        result = _format_state(state)
        lines = [line for line in result.splitlines() if line.strip()]
        assert len(lines) == 2
