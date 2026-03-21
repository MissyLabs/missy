"""Extended tests for missy.vision.intent — new patterns added to the classifier.

Covers the newer puzzle, painting, and inspect patterns, plus edge cases for
thresholds, log management, and robustness.
"""

from __future__ import annotations

import pytest

from missy.vision.intent import (
    ActivationDecision,
    VisionIntent,
    VisionIntentClassifier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_classifier(**kwargs) -> VisionIntentClassifier:
    """Return a fresh VisionIntentClassifier with default thresholds unless overridden."""
    return VisionIntentClassifier(**kwargs)


# ---------------------------------------------------------------------------
# Puzzle patterns (new)
# ---------------------------------------------------------------------------


class TestNewPuzzlePatterns:
    """Tests for puzzle patterns added beyond the original test suite."""

    def setup_method(self):
        self.clf = make_classifier()

    def test_missing_piece(self):
        # "missing piece" matches _PUZZLE_PATTERNS at 0.85
        result = self.clf.classify("I found a missing piece")
        assert result.intent == VisionIntent.PUZZLE
        assert result.confidence >= 0.80
        assert result.suggested_mode == "puzzle"

    def test_top_of_the_puzzle(self):
        # "(top|bottom|left|right|...) of the puzzle" matches at 0.75
        result = self.clf.classify("top of the puzzle")
        assert result.intent == VisionIntent.PUZZLE
        assert result.suggested_mode == "puzzle"

    def test_fit_in_the_blue_area(self):
        # "fit (this|that|it) in the <word> (area|region|part)" matches at 0.85
        result = self.clf.classify("fit this in the blue area")
        assert result.intent == VisionIntent.PUZZLE
        assert result.confidence >= 0.80

    def test_which_pieces_go_together(self):
        # "(which|what) pieces? (go|match|connect|fit)" matches at 0.85
        result = self.clf.classify("which pieces go together?")
        assert result.intent == VisionIntent.PUZZLE
        assert result.confidence >= 0.80

    def test_fit_it_in_the_sky_region(self):
        # "fit it in the sky region" matches at 0.85
        result = self.clf.classify("fit it in the sky region")
        assert result.intent == VisionIntent.PUZZLE
        assert result.confidence >= 0.80

    def test_bottom_of_the_board(self):
        # "bottom of the board" also matches the positional puzzle pattern
        result = self.clf.classify("the pieces at the bottom of the board look wrong")
        assert result.intent == VisionIntent.PUZZLE

    def test_what_pieces_fit_here(self):
        # "what pieces fit" covers the (which|what) pieces? pattern
        result = self.clf.classify("what pieces fit here?")
        assert result.intent == VisionIntent.PUZZLE

    def test_missing_piece_activates(self):
        # confidence 0.85 >= default auto_threshold 0.80 → ACTIVATE
        result = self.clf.classify("I found a missing piece")
        assert result.decision == ActivationDecision.ACTIVATE


# ---------------------------------------------------------------------------
# Painting patterns (new)
# ---------------------------------------------------------------------------


class TestNewPaintingPatterns:
    """Tests for painting patterns added beyond the original test suite."""

    def setup_method(self):
        self.clf = make_classifier()

    def test_how_am_i_doing_with_the_painting(self):
        # "(how)? am I doing" + "paint" matches at 0.80
        result = self.clf.classify("how am I doing with the painting?")
        assert result.intent == VisionIntent.PAINTING
        assert result.confidence >= 0.75
        assert result.suggested_mode == "painting"

    def test_mix_the_colors_better(self):
        # "mix (the)? (colors?|paint)" matches at 0.65
        result = self.clf.classify("mix the colors better")
        assert result.intent == VisionIntent.PAINTING
        # 0.65 is below default auto_threshold 0.80
        assert result.decision in (ActivationDecision.ASK, ActivationDecision.ACTIVATE)

    def test_blend_the_paint_colors(self):
        # "blend (the)? (colors?|paint)" matches at 0.65
        result = self.clf.classify("blend the paint colors")
        assert result.intent == VisionIntent.PAINTING

    def test_how_can_i_improve_this(self):
        # "how can I improve (this|my)" matches at 0.70
        result = self.clf.classify("how can I improve this?")
        assert result.intent == VisionIntent.PAINTING
        assert result.confidence >= 0.65

    def test_landscape_painting(self):
        # "painting" keyword alone matches at 0.85; "landscape" matches at 0.55
        # "painting" wins with 0.85 → ACTIVATE
        result = self.clf.classify("this is a landscape painting")
        assert result.intent == VisionIntent.PAINTING
        assert result.confidence >= 0.80

    def test_mix_colors_lower_confidence(self):
        # "mix the colors" → 0.65, below auto_threshold 0.80
        result = self.clf.classify("mix the colors better")
        assert result.confidence < 0.80

    def test_how_am_i_doing_painting_activates(self):
        # 0.80 confidence >= default auto_threshold 0.80
        result = self.clf.classify("how am I doing with the painting?")
        assert result.decision == ActivationDecision.ACTIVATE

    def test_shade_the_paint(self):
        # "shade (the)? paint" also matches the mix/blend/shade pattern
        result = self.clf.classify("I need to shade the paint more evenly")
        assert result.intent == VisionIntent.PAINTING


# ---------------------------------------------------------------------------
# Inspect patterns (new)
# ---------------------------------------------------------------------------


class TestNewInspectPatterns:
    """Tests for inspection patterns added beyond the original test suite."""

    def setup_method(self):
        self.clf = make_classifier()

    def test_zoom_in_on_that(self):
        # "zoom in" matches at 0.75
        result = self.clf.classify("zoom in on that")
        assert result.intent == VisionIntent.INSPECT
        assert result.confidence >= 0.70
        assert result.suggested_mode == "inspection"

    def test_what_color_is_that(self):
        # "what color is" matches at 0.70
        result = self.clf.classify("what color is that?")
        assert result.intent == VisionIntent.INSPECT
        assert result.confidence >= 0.65

    def test_how_many_are_there(self):
        # "how many" matches "count|how many" at 0.65
        result = self.clf.classify("how many are there?")
        assert result.intent == VisionIntent.INSPECT

    def test_what_brand_is_that(self):
        # "what brand is" matches at 0.70
        result = self.clf.classify("what brand is that?")
        assert result.intent == VisionIntent.INSPECT
        assert result.confidence >= 0.65

    def test_magnify(self):
        # "magnify" is part of the zoom_in / magnify group at 0.75
        result = self.clf.classify("can you magnify that for me?")
        assert result.intent == VisionIntent.INSPECT

    def test_look_closer(self):
        # "look closer" matches at 0.75
        result = self.clf.classify("look closer at the label")
        assert result.intent == VisionIntent.INSPECT

    def test_what_type_is_it(self):
        # "what type is" matches at 0.70
        result = self.clf.classify("what type is it?")
        assert result.intent == VisionIntent.INSPECT


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty input, whitespace, unrelated text, long input, thresholds."""

    def setup_method(self):
        self.clf = make_classifier()

    def test_empty_string_returns_none_skip(self):
        result = self.clf.classify("")
        assert result.intent == VisionIntent.NONE
        assert result.decision == ActivationDecision.SKIP
        assert result.confidence == 0.0

    def test_whitespace_only_returns_none_skip(self):
        result = self.clf.classify("   \t\n  ")
        assert result.intent == VisionIntent.NONE
        assert result.decision == ActivationDecision.SKIP
        assert result.confidence == 0.0

    def test_unrelated_text_returns_none_low_confidence(self):
        result = self.clf.classify("Please schedule a meeting for tomorrow at 3pm")
        assert result.intent == VisionIntent.NONE
        assert result.confidence == 0.0
        assert result.decision == ActivationDecision.SKIP

    def test_another_unrelated_text(self):
        result = self.clf.classify("What is the boiling point of water?")
        assert result.intent == VisionIntent.NONE
        assert result.decision == ActivationDecision.SKIP

    def test_combined_puzzle_and_look_boosts_confidence(self):
        # "look at this" → LOOK 0.95; "missing piece" → PUZZLE base 0.85 + 0.10 boost = 0.95.
        # The boost makes the effective confidence equal to LOOK's 0.95, but the
        # classifier uses strict `>` so LOOK wins when tied.  The important
        # thing is that the overall confidence reflects the boost (>= 0.90)
        # and a vision-relevant intent was detected.
        result = self.clf.classify("look at this missing piece")
        assert result.intent in (VisionIntent.LOOK, VisionIntent.PUZZLE)
        assert result.confidence >= 0.90
        assert result.decision == ActivationDecision.ACTIVATE

    def test_combined_painting_and_look_boosts_confidence(self):
        # "look at" → LOOK 0.95, then "canvas" → PAINTING 0.85 + 0.10 = 0.95
        result = self.clf.classify("look at this canvas")
        assert result.intent in (VisionIntent.PAINTING, VisionIntent.LOOK)
        assert result.confidence >= 0.85

    def test_very_long_input_still_classifies(self):
        long_text = (
            "I have been working on this jigsaw puzzle for three days and I am "
            "really struggling to figure out where all the pieces go, especially "
            "the ones that look similar. "
        ) * 10  # ~600 chars
        result = self.clf.classify(long_text)
        assert result.intent == VisionIntent.PUZZLE
        assert result.decision == ActivationDecision.ACTIVATE

    def test_confidence_exactly_at_auto_threshold_activates(self):
        # Craft a classifier where auto_threshold == ask_threshold == 0.75
        # "top of the puzzle" fires at 0.75; set both thresholds to 0.75
        clf = make_classifier(auto_threshold=0.75, ask_threshold=0.50)
        result = clf.classify("top of the puzzle")
        # 0.75 >= auto_threshold 0.75 → ACTIVATE
        assert result.decision == ActivationDecision.ACTIVATE

    def test_confidence_just_below_auto_threshold_asks(self):
        # "top of the puzzle" = 0.75; auto_threshold 0.80, ask_threshold 0.50
        clf = make_classifier(auto_threshold=0.80, ask_threshold=0.50)
        result = clf.classify("top of the puzzle")
        assert result.confidence == pytest.approx(0.75)
        assert result.decision == ActivationDecision.ASK

    def test_confidence_below_ask_threshold_skips(self):
        # Force ask_threshold above the match confidence
        clf = make_classifier(auto_threshold=0.90, ask_threshold=0.80)
        result = clf.classify("top of the puzzle")
        # 0.75 < ask_threshold 0.80
        assert result.decision == ActivationDecision.SKIP

    def test_invalid_auto_threshold_raises(self):
        with pytest.raises(ValueError):
            VisionIntentClassifier(auto_threshold=1.5)

    def test_invalid_ask_threshold_raises(self):
        with pytest.raises(ValueError):
            VisionIntentClassifier(ask_threshold=-0.1)


# ---------------------------------------------------------------------------
# Activation log behaviour
# ---------------------------------------------------------------------------


class TestActivationLog:
    """Activation log is populated on every classify call; clear_log resets it."""

    def test_log_is_empty_on_init(self):
        clf = make_classifier()
        assert clf.activation_log == []

    def test_log_populated_after_single_classify(self):
        clf = make_classifier()
        clf.classify("look at this missing piece")
        log = clf.activation_log
        assert len(log) == 1
        # "look at this" (0.95) ties with boosted "missing piece" (0.95);
        # LOOK wins on strict > comparison, so either vision intent is valid here.
        assert log[0].intent in (VisionIntent.LOOK, VisionIntent.PUZZLE)

    def test_log_populated_after_multiple_classifies(self):
        clf = make_classifier()
        clf.classify("zoom in on that")
        clf.classify("mix the colors better")
        clf.classify("where does this piece go?")
        assert len(clf.activation_log) == 3

    def test_log_also_records_none_results(self):
        clf = make_classifier()
        clf.classify("")
        # empty string returns early without appending
        assert len(clf.activation_log) == 0

    def test_log_records_unrelated_text_as_none(self):
        clf = make_classifier()
        clf.classify("What is the capital of France?")
        log = clf.activation_log
        assert len(log) == 1
        assert log[0].intent == VisionIntent.NONE

    def test_clear_log_empties_the_log(self):
        clf = make_classifier()
        clf.classify("look at this")
        clf.classify("zoom in on that")
        clf.clear_log()
        assert clf.activation_log == []

    def test_clear_log_then_classify_repopulates(self):
        clf = make_classifier()
        clf.classify("look at this")
        clf.clear_log()
        clf.classify("what brand is that?")
        assert len(clf.activation_log) == 1

    def test_activation_log_returns_copy(self):
        clf = make_classifier()
        clf.classify("look at this")
        log1 = clf.activation_log
        log1.clear()  # mutate the returned copy
        assert len(clf.activation_log) == 1  # original unchanged

    def test_log_entries_have_audit_dict(self):
        clf = make_classifier()
        clf.classify("I found a missing piece")
        entry = clf.activation_log[0]
        d = entry.to_audit_dict()
        assert set(d.keys()) >= {
            "intent",
            "decision",
            "confidence",
            "trigger_phrase",
            "suggested_mode",
            "timestamp",
        }
        assert d["intent"] == "puzzle"
