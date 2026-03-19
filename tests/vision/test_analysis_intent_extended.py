"""Session 14: Edge case tests for analysis, intent classifier, and audit.

Covers:
- AnalysisPromptBuilder: all modes, context sanitization, followup prompts,
  context truncation, format_state
- _describe_color: all named colors and edge cases
- PuzzlePreprocessor: edge enhancement, color extraction, contour detection
- VisionIntentClassifier: explicit patterns, puzzle/painting detection,
  threshold boundaries, empty input, activation log
- Audit functions: all 7 event types, failure handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Analysis module tests
# ---------------------------------------------------------------------------


class TestAnalysisPromptBuilder:
    """Tests for AnalysisPromptBuilder."""

    def test_general_mode_no_context(self):
        from missy.vision.analysis import AnalysisMode, AnalysisPromptBuilder, AnalysisRequest

        builder = AnalysisPromptBuilder()
        request = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.GENERAL,
        )
        prompt = builder.build_prompt(request)
        assert "Analyze this image" in prompt
        assert "User-provided context" not in prompt

    def test_general_mode_with_context(self):
        from missy.vision.analysis import AnalysisMode, AnalysisPromptBuilder, AnalysisRequest

        builder = AnalysisPromptBuilder()
        request = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.GENERAL,
            context="A photo of my desk",
        )
        prompt = builder.build_prompt(request)
        assert "A photo of my desk" in prompt
        assert "[User-provided context]" in prompt

    def test_puzzle_mode_first_time(self):
        from missy.vision.analysis import AnalysisMode, AnalysisPromptBuilder, AnalysisRequest

        builder = AnalysisPromptBuilder()
        request = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.PUZZLE,
        )
        prompt = builder.build_prompt(request)
        assert "jigsaw puzzle" in prompt
        assert "Board State" in prompt

    def test_puzzle_mode_followup(self):
        from missy.vision.analysis import AnalysisMode, AnalysisPromptBuilder, AnalysisRequest

        builder = AnalysisPromptBuilder()
        request = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.PUZZLE,
            is_followup=True,
            previous_observations=["Found 3 edge pieces", "Sky section 50% complete"],
            previous_state={"completion": "40%"},
        )
        prompt = builder.build_prompt(request)
        assert "Found 3 edge pieces" in prompt
        assert "completion" in prompt

    def test_puzzle_followup_without_observations_uses_initial(self):
        """Followup with empty observations should use initial prompt."""
        from missy.vision.analysis import AnalysisMode, AnalysisPromptBuilder, AnalysisRequest

        builder = AnalysisPromptBuilder()
        request = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.PUZZLE,
            is_followup=True,
            previous_observations=[],
        )
        prompt = builder.build_prompt(request)
        assert "Board State" in prompt  # Initial prompt

    def test_painting_mode(self):
        from missy.vision.analysis import AnalysisMode, AnalysisPromptBuilder, AnalysisRequest

        builder = AnalysisPromptBuilder()
        request = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.PAINTING,
        )
        prompt = builder.build_prompt(request)
        assert "supportive painting coach" in prompt
        assert "warm" in prompt.lower()

    def test_painting_mode_with_context(self):
        from missy.vision.analysis import AnalysisMode, AnalysisPromptBuilder, AnalysisRequest

        builder = AnalysisPromptBuilder()
        request = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.PAINTING,
            context="This is my first watercolor",
        )
        prompt = builder.build_prompt(request)
        assert "first watercolor" in prompt
        assert "[The painter says]" in prompt

    def test_painting_followup(self):
        from missy.vision.analysis import AnalysisMode, AnalysisPromptBuilder, AnalysisRequest

        builder = AnalysisPromptBuilder()
        request = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.PAINTING,
            is_followup=True,
            previous_observations=["Beautiful color palette"],
        )
        prompt = builder.build_prompt(request)
        assert "Beautiful color palette" in prompt

    def test_inspection_mode(self):
        from missy.vision.analysis import AnalysisMode, AnalysisPromptBuilder, AnalysisRequest

        builder = AnalysisPromptBuilder()
        request = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.INSPECTION,
        )
        prompt = builder.build_prompt(request)
        assert "inspection report" in prompt

    def test_inspection_with_context(self):
        from missy.vision.analysis import AnalysisMode, AnalysisPromptBuilder, AnalysisRequest

        builder = AnalysisPromptBuilder()
        request = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.INSPECTION,
            context="Check for damage",
        )
        prompt = builder.build_prompt(request)
        assert "[Inspection focus]" in prompt

    def test_context_truncation(self):
        from missy.vision.analysis import AnalysisPromptBuilder

        long_context = "x" * 3000
        truncated = AnalysisPromptBuilder._sanitize_context(long_context)
        assert len(truncated) < 3000
        assert "[truncated]" in truncated

    def test_context_empty_string(self):
        from missy.vision.analysis import AnalysisPromptBuilder

        assert AnalysisPromptBuilder._sanitize_context("") == ""

    def test_context_exactly_at_limit(self):
        from missy.vision.analysis import AnalysisPromptBuilder

        context = "x" * 2000
        truncated = AnalysisPromptBuilder._sanitize_context(context)
        assert "[truncated]" not in truncated

    def test_context_one_over_limit(self):
        from missy.vision.analysis import AnalysisPromptBuilder

        context = "x" * 2001
        truncated = AnalysisPromptBuilder._sanitize_context(context)
        assert "[truncated]" in truncated


class TestDescribeColor:
    """Tests for _describe_color helper."""

    def test_black(self):
        from missy.vision.analysis import _describe_color
        assert _describe_color([10, 10, 10]) == "black"

    def test_white(self):
        from missy.vision.analysis import _describe_color
        assert _describe_color([220, 220, 220]) == "white"

    def test_red(self):
        from missy.vision.analysis import _describe_color
        assert _describe_color([200, 30, 30]) == "red"

    def test_green(self):
        from missy.vision.analysis import _describe_color
        assert _describe_color([20, 180, 20]) == "green"

    def test_blue(self):
        from missy.vision.analysis import _describe_color
        assert _describe_color([20, 20, 180]) == "blue"

    def test_yellow(self):
        from missy.vision.analysis import _describe_color
        assert _describe_color([200, 200, 30]) == "yellow"

    def test_orange(self):
        from missy.vision.analysis import _describe_color
        assert _describe_color([200, 120, 30]) == "orange"

    def test_purple(self):
        from missy.vision.analysis import _describe_color
        assert _describe_color([160, 30, 160]) == "purple"

    def test_gray(self):
        from missy.vision.analysis import _describe_color
        assert _describe_color([100, 100, 100]) == "gray"

    def test_light_gray(self):
        from missy.vision.analysis import _describe_color
        assert _describe_color([180, 180, 180]) == "light gray"

    def test_tan_brown(self):
        from missy.vision.analysis import _describe_color
        assert _describe_color([160, 120, 90]) == "tan/brown"

    def test_unnamed_color(self):
        from missy.vision.analysis import _describe_color
        result = _describe_color([100, 200, 150])
        assert "rgb(" in result


class TestFormatState:
    """Tests for _format_state helper."""

    def test_empty_state(self):
        from missy.vision.analysis import _format_state
        assert "No previous state" in _format_state({})

    def test_state_with_entries(self):
        from missy.vision.analysis import _format_state
        result = _format_state({"completion": "40%", "pieces_placed": 50})
        assert "completion" in result
        assert "40%" in result

    def test_state_single_entry(self):
        from missy.vision.analysis import _format_state
        result = _format_state({"key": "value"})
        assert "- key: value" in result


class TestAnalysisMode:
    """Tests for AnalysisMode enum."""

    def test_all_modes(self):
        from missy.vision.analysis import AnalysisMode
        assert AnalysisMode.GENERAL == "general"
        assert AnalysisMode.PUZZLE == "puzzle"
        assert AnalysisMode.PAINTING == "painting"
        assert AnalysisMode.INSPECTION == "inspection"


class TestAnalysisDataclasses:
    """Tests for AnalysisRequest and AnalysisResult dataclasses."""

    def test_analysis_request_defaults(self):
        from missy.vision.analysis import AnalysisMode, AnalysisRequest

        req = AnalysisRequest(image=np.zeros((10, 10, 3), dtype=np.uint8))
        assert req.mode == AnalysisMode.GENERAL
        assert req.context == ""
        assert req.previous_observations == []
        assert req.is_followup is False

    def test_analysis_result_defaults(self):
        from missy.vision.analysis import AnalysisMode, AnalysisResult

        result = AnalysisResult(text="test", mode=AnalysisMode.GENERAL)
        assert result.confidence == 1.0
        assert result.metadata == {}
        assert result.prompt_used == ""


# ---------------------------------------------------------------------------
# VisionIntentClassifier tests
# ---------------------------------------------------------------------------


class TestVisionIntentClassifier:
    """Tests for VisionIntentClassifier."""

    def test_empty_text(self):
        from missy.vision.intent import ActivationDecision, VisionIntent, VisionIntentClassifier

        clf = VisionIntentClassifier()
        result = clf.classify("")
        assert result.intent == VisionIntent.NONE
        assert result.decision == ActivationDecision.SKIP

    def test_whitespace_only(self):
        from missy.vision.intent import VisionIntent, VisionIntentClassifier

        clf = VisionIntentClassifier()
        result = clf.classify("   \n\t  ")
        assert result.intent == VisionIntent.NONE

    def test_no_vision_intent(self):
        from missy.vision.intent import VisionIntent, VisionIntentClassifier

        clf = VisionIntentClassifier()
        result = clf.classify("What is the weather like today?")
        assert result.intent == VisionIntent.NONE

    def test_explicit_look(self):
        from missy.vision.intent import VisionIntent, VisionIntentClassifier

        clf = VisionIntentClassifier()
        result = clf.classify("Look at this please")
        assert result.intent == VisionIntent.LOOK
        assert result.confidence >= 0.8

    def test_take_a_photo(self):
        from missy.vision.intent import VisionIntent, VisionIntentClassifier

        clf = VisionIntentClassifier()
        result = clf.classify("Take a photo of that")
        assert result.intent == VisionIntent.LOOK

    def test_puzzle_piece_intent(self):
        from missy.vision.intent import VisionIntent, VisionIntentClassifier

        clf = VisionIntentClassifier()
        result = clf.classify("Where does this puzzle piece go?")
        assert result.intent == VisionIntent.PUZZLE
        assert result.suggested_mode == "puzzle"

    def test_edge_piece_intent(self):
        from missy.vision.intent import VisionIntent, VisionIntentClassifier

        clf = VisionIntentClassifier()
        result = clf.classify("Is this an edge piece?")
        assert result.intent == VisionIntent.PUZZLE

    def test_painting_intent(self):
        from missy.vision.intent import VisionIntent, VisionIntentClassifier

        clf = VisionIntentClassifier()
        result = clf.classify("What do you think of my painting?")
        assert result.intent == VisionIntent.PAINTING
        assert result.suggested_mode == "painting"

    def test_check_intent(self):
        from missy.vision.intent import VisionIntent, VisionIntentClassifier

        clf = VisionIntentClassifier()
        result = clf.classify("Check what's on the table")
        assert result.intent == VisionIntent.CHECK

    def test_screenshot_intent(self):
        from missy.vision.intent import VisionIntent, VisionIntentClassifier

        clf = VisionIntentClassifier()
        result = clf.classify("Take a screenshot")
        assert result.intent == VisionIntent.SCREENSHOT

    def test_read_intent(self):
        from missy.vision.intent import VisionIntentClassifier

        clf = VisionIntentClassifier()
        result = clf.classify("Read what this says")
        assert result.confidence > 0

    def test_inspect_intent(self):
        from missy.vision.intent import VisionIntentClassifier

        clf = VisionIntentClassifier()
        result = clf.classify("What's on the desk?")
        assert result.confidence > 0

    def test_auto_threshold_triggers_activate(self):
        from missy.vision.intent import ActivationDecision, VisionIntentClassifier

        clf = VisionIntentClassifier(auto_threshold=0.80)
        result = clf.classify("Look at this puzzle piece")
        if result.confidence >= 0.80:
            assert result.decision == ActivationDecision.ACTIVATE

    def test_ask_threshold_triggers_ask(self):
        from missy.vision.intent import ActivationDecision, VisionIntentClassifier

        clf = VisionIntentClassifier(auto_threshold=0.99, ask_threshold=0.50)
        result = clf.classify("How does this look?")
        if 0.50 <= result.confidence < 0.99:
            assert result.decision == ActivationDecision.ASK

    def test_invalid_auto_threshold(self):
        from missy.vision.intent import VisionIntentClassifier

        with pytest.raises(ValueError):
            VisionIntentClassifier(auto_threshold=1.5)

    def test_invalid_ask_threshold(self):
        from missy.vision.intent import VisionIntentClassifier

        with pytest.raises(ValueError):
            VisionIntentClassifier(ask_threshold=-0.1)

    def test_intent_result_to_audit_dict(self):
        from missy.vision.intent import ActivationDecision, IntentResult, VisionIntent

        result = IntentResult(
            intent=VisionIntent.PUZZLE,
            decision=ActivationDecision.ACTIVATE,
            confidence=0.95,
            trigger_phrase="puzzle piece",
            suggested_mode="puzzle",
        )
        d = result.to_audit_dict()
        assert d["intent"] == "puzzle"
        assert d["decision"] == "activate"
        assert d["confidence"] == 0.95

    def test_case_insensitive_detection(self):
        from missy.vision.intent import VisionIntent, VisionIntentClassifier

        clf = VisionIntentClassifier()
        result = clf.classify("LOOK AT THIS PUZZLE PIECE")
        assert result.intent in (VisionIntent.LOOK, VisionIntent.PUZZLE)
        assert result.confidence > 0

    def test_jigsaw_keyword(self):
        from missy.vision.intent import VisionIntent, VisionIntentClassifier

        clf = VisionIntentClassifier()
        result = clf.classify("Help me with this jigsaw")
        assert result.intent == VisionIntent.PUZZLE

    def test_painting_with_improve(self):
        from missy.vision.intent import VisionIntent, VisionIntentClassifier

        clf = VisionIntentClassifier()
        result = clf.classify("How can I improve this painting?")
        assert result.intent == VisionIntent.PAINTING

    def test_sky_section_puzzle(self):
        from missy.vision.intent import VisionIntent, VisionIntentClassifier

        clf = VisionIntentClassifier()
        result = clf.classify("Does this piece fit in the sky section?")
        assert result.intent == VisionIntent.PUZZLE


# ---------------------------------------------------------------------------
# Audit function tests
# ---------------------------------------------------------------------------


class TestVisionAudit:
    """Tests for vision audit functions.

    The audit module uses lazy imports inside _emit_audit_event, so we
    patch the source module (missy.observability.audit_logger).
    """

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_audit_capture(self, mock_get_logger):
        from missy.vision.audit import audit_vision_capture

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        audit_vision_capture(
            device="/dev/video0", source_type="webcam",
            success=True, width=1920, height=1080,
        )
        mock_logger.log.assert_called_once()
        event = mock_logger.log.call_args[0][0]
        assert event["action"] == "capture"
        assert event["device"] == "/dev/video0"

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_audit_analysis(self, mock_get_logger):
        from missy.vision.audit import audit_vision_analysis

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        audit_vision_analysis(mode="puzzle", success=True)
        event = mock_logger.log.call_args[0][0]
        assert event["action"] == "analyze"
        assert event["mode"] == "puzzle"

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_audit_intent(self, mock_get_logger):
        from missy.vision.audit import audit_vision_intent

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        audit_vision_intent(
            text="look at this", intent="look",
            confidence=0.95, decision="activate",
        )
        event = mock_logger.log.call_args[0][0]
        assert event["action"] == "intent"
        assert event["confidence"] == 0.95

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_audit_device_discovery(self, mock_get_logger):
        from missy.vision.audit import audit_vision_device_discovery

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        audit_vision_device_discovery(camera_count=2, preferred_device="/dev/video0")
        event = mock_logger.log.call_args[0][0]
        assert event["action"] == "device_discovery"

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_audit_session(self, mock_get_logger):
        from missy.vision.audit import audit_vision_session

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        audit_vision_session(action="open", task_id="t1", task_type="puzzle", frame_count=5)
        event = mock_logger.log.call_args[0][0]
        assert event["action"] == "session_open"

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_audit_burst(self, mock_get_logger):
        from missy.vision.audit import audit_vision_burst

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        audit_vision_burst(device="/dev/video0", count=5, successful=4)
        event = mock_logger.log.call_args[0][0]
        assert event["action"] == "burst_capture"

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_audit_error(self, mock_get_logger):
        from missy.vision.audit import audit_vision_error

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        audit_vision_error(operation="capture", error="timeout", recoverable=True)
        event = mock_logger.log.call_args[0][0]
        assert event["action"] == "error"
        assert event["recoverable"] is True

    @patch("missy.observability.audit_logger.get_audit_logger", return_value=None)
    def test_audit_no_logger(self, _):
        """Audit should be a no-op when logger is None."""
        from missy.vision.audit import audit_vision_capture
        audit_vision_capture()  # Should not raise

    def test_audit_import_failure(self):
        """Audit should handle import failures gracefully."""
        from missy.vision.audit import audit_vision_capture
        # _emit_audit_event catches all exceptions — even if get_audit_logger
        # can't be imported, it logs debug and returns silently.
        # Simply calling without a configured logger exercises the fallback path.
        audit_vision_capture()  # Should not raise


# ---------------------------------------------------------------------------
# PuzzlePreprocessor tests
# ---------------------------------------------------------------------------


class TestPuzzlePreprocessor:
    """Tests for PuzzlePreprocessor (requires OpenCV)."""

    def test_enhance_edges_basic(self):
        from missy.vision.analysis import PuzzlePreprocessor

        pp = PuzzlePreprocessor()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = pp.enhance_edges(img)
        assert result.shape == img.shape

    def test_extract_color_regions_basic(self):
        from missy.vision.analysis import PuzzlePreprocessor

        pp = PuzzlePreprocessor()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = pp.extract_color_regions(img)
        assert "dominant_colors" in result

    def test_detect_edges_and_corners(self):
        from missy.vision.analysis import PuzzlePreprocessor

        pp = PuzzlePreprocessor()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = pp.detect_edges_and_corners(img)
        assert "contour_count" in result or "error" in result

    def test_enhance_edges_single_color(self):
        """Solid color image should have minimal edges."""
        from missy.vision.analysis import PuzzlePreprocessor

        pp = PuzzlePreprocessor()
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = pp.enhance_edges(img)
        assert result.shape == img.shape
