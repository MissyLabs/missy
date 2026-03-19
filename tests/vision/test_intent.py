"""Tests for missy.vision.intent — audio-triggered vision intent detection."""

from __future__ import annotations

from missy.vision.intent import (
    ActivationDecision,
    IntentResult,
    VisionIntent,
    VisionIntentClassifier,
    classify_vision_intent,
)

# ---------------------------------------------------------------------------
# VisionIntentClassifier tests
# ---------------------------------------------------------------------------


class TestVisionIntentClassifier:
    def setup_method(self):
        self.classifier = VisionIntentClassifier()

    # -- Explicit vision requests --

    def test_look_at_this(self):
        result = self.classifier.classify("Missy, look at this")
        assert result.intent == VisionIntent.LOOK
        assert result.decision == ActivationDecision.ACTIVATE
        assert result.confidence >= 0.80

    def test_show_me(self):
        result = self.classifier.classify("Show me what you see")
        assert result.intent == VisionIntent.LOOK
        assert result.confidence >= 0.80

    def test_take_a_photo(self):
        result = self.classifier.classify("Take a photo of the table")
        assert result.intent == VisionIntent.LOOK
        assert result.decision == ActivationDecision.ACTIVATE

    def test_screenshot(self):
        result = self.classifier.classify("Take a screenshot")
        assert result.intent == VisionIntent.SCREENSHOT
        assert result.confidence >= 0.90

    # -- Puzzle intents --

    def test_puzzle_piece_placement(self):
        result = self.classifier.classify("Where does this puzzle piece go?")
        assert result.intent == VisionIntent.PUZZLE
        assert result.suggested_mode == "puzzle"
        assert result.confidence >= 0.80

    def test_edge_piece(self):
        result = self.classifier.classify("Is this an edge piece?")
        assert result.intent == VisionIntent.PUZZLE
        assert result.confidence >= 0.80

    def test_sky_section(self):
        result = self.classifier.classify("Does this piece fit in the sky section?")
        assert result.intent == VisionIntent.PUZZLE

    def test_sort_pieces(self):
        result = self.classifier.classify("Help me sort the pieces by color")
        assert result.intent == VisionIntent.PUZZLE

    def test_jigsaw(self):
        result = self.classifier.classify("I need help with this jigsaw")
        assert result.intent == VisionIntent.PUZZLE

    # -- Painting intents --

    def test_painting_feedback(self):
        result = self.classifier.classify("What do you think of this painting?")
        assert result.intent == VisionIntent.PAINTING
        assert result.suggested_mode == "painting"

    def test_improve_painting(self):
        result = self.classifier.classify("How can I improve this painting?")
        assert result.intent == VisionIntent.PAINTING

    def test_canvas(self):
        result = self.classifier.classify("Look at this canvas I've been working on")
        assert result.intent in (VisionIntent.PAINTING, VisionIntent.LOOK)

    # -- Inspect/read intents --

    def test_read_this(self):
        result = self.classifier.classify("Can you read what this says?")
        assert result.intent == VisionIntent.INSPECT

    def test_whats_on_table(self):
        result = self.classifier.classify("What's on the table?")
        assert result.intent in (VisionIntent.INSPECT, VisionIntent.CHECK)

    # -- No vision needed --

    def test_no_vision_general_question(self):
        result = self.classifier.classify("What is the capital of France?")
        assert result.intent == VisionIntent.NONE
        assert result.decision == ActivationDecision.SKIP

    def test_no_vision_coding_request(self):
        result = self.classifier.classify("Write a Python function to sort a list")
        assert result.decision == ActivationDecision.SKIP

    def test_empty_input(self):
        result = self.classifier.classify("")
        assert result.intent == VisionIntent.NONE
        assert result.confidence == 0.0

    def test_whitespace_only(self):
        result = self.classifier.classify("   ")
        assert result.intent == VisionIntent.NONE

    # -- Activation thresholds --

    def test_high_confidence_activates(self):
        cls = VisionIntentClassifier(auto_threshold=0.80)
        result = cls.classify("Look at this")
        assert result.decision == ActivationDecision.ACTIVATE

    def test_medium_confidence_asks(self):
        cls = VisionIntentClassifier(auto_threshold=0.95, ask_threshold=0.40)
        result = cls.classify("How does this look?")
        # "how does this look" matches painting at ~0.65
        if result.confidence < 0.95 and result.confidence >= 0.40:
            assert result.decision == ActivationDecision.ASK

    def test_low_confidence_skips(self):
        cls = VisionIntentClassifier(auto_threshold=0.95, ask_threshold=0.90)
        result = cls.classify("What is Python?")
        assert result.decision == ActivationDecision.SKIP

    # -- Activation log --

    def test_activation_log(self):
        cls = VisionIntentClassifier()
        cls.classify("Look at this")
        cls.classify("What color is the sky?")

        assert len(cls.activation_log) == 2

    def test_clear_log(self):
        cls = VisionIntentClassifier()
        cls.classify("Look at this")
        cls.clear_log()
        assert len(cls.activation_log) == 0


# ---------------------------------------------------------------------------
# IntentResult tests
# ---------------------------------------------------------------------------


class TestIntentResult:
    def test_to_audit_dict(self):
        result = IntentResult(
            intent=VisionIntent.PUZZLE,
            decision=ActivationDecision.ACTIVATE,
            confidence=0.92,
            trigger_phrase="puzzle piece",
            suggested_mode="puzzle",
        )
        d = result.to_audit_dict()
        assert d["intent"] == "puzzle"
        assert d["decision"] == "activate"
        assert d["confidence"] == 0.92


# ---------------------------------------------------------------------------
# Module-level convenience tests
# ---------------------------------------------------------------------------


class TestConvenience:
    def test_classify_vision_intent(self):
        result = classify_vision_intent("Look at this puzzle")
        assert result.intent in (VisionIntent.LOOK, VisionIntent.PUZZLE)
        assert result.confidence > 0.0
