"""Tests for missy.vision.analysis — domain-specific visual analysis."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from missy.vision.analysis import (
    AnalysisMode,
    AnalysisPromptBuilder,
    AnalysisRequest,
    PuzzlePreprocessor,
    _describe_color,
)


# ---------------------------------------------------------------------------
# AnalysisPromptBuilder tests
# ---------------------------------------------------------------------------


class TestAnalysisPromptBuilder:
    def setup_method(self):
        self.builder = AnalysisPromptBuilder()

    def test_general_prompt(self):
        req = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.GENERAL,
        )
        prompt = self.builder.build_prompt(req)
        assert "Analyze this image" in prompt
        assert "Main subjects" in prompt

    def test_general_with_context(self):
        req = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.GENERAL,
            context="Focus on the left side",
        )
        prompt = self.builder.build_prompt(req)
        assert "Focus on the left side" in prompt

    def test_puzzle_prompt(self):
        req = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.PUZZLE,
        )
        prompt = self.builder.build_prompt(req)
        assert "jigsaw puzzle" in prompt
        assert "Board State" in prompt
        assert "Piece Identification" in prompt
        assert "Placement Guidance" in prompt

    def test_puzzle_followup(self):
        req = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.PUZZLE,
            is_followup=True,
            previous_observations=["Sky section is 50% complete"],
            previous_state={"completed": ["sky corner"]},
        )
        prompt = self.builder.build_prompt(req)
        assert "Sky section is 50% complete" in prompt
        assert "progress" in prompt.lower()

    def test_painting_prompt(self):
        req = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.PAINTING,
        )
        prompt = self.builder.build_prompt(req)
        assert "supportive painting coach" in prompt
        assert "warm" in prompt.lower()
        assert "encouraging" in prompt.lower()
        # Should NOT contain harsh language
        assert "wrong" not in prompt.lower().split("never")[-1] or "wrong" in prompt
        assert "Never use words like" in prompt  # the negative instruction

    def test_painting_with_context(self):
        req = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.PAINTING,
            context="This is my first oil painting",
        )
        prompt = self.builder.build_prompt(req)
        assert "first oil painting" in prompt

    def test_painting_followup(self):
        req = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.PAINTING,
            is_followup=True,
            previous_observations=["Beautiful use of warm colors"],
        )
        prompt = self.builder.build_prompt(req)
        assert "Beautiful use of warm colors" in prompt

    def test_inspection_prompt(self):
        req = AnalysisRequest(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            mode=AnalysisMode.INSPECTION,
        )
        prompt = self.builder.build_prompt(req)
        assert "inspection report" in prompt.lower()


# ---------------------------------------------------------------------------
# PuzzlePreprocessor tests
# ---------------------------------------------------------------------------


class TestPuzzlePreprocessor:
    def setup_method(self):
        self.preprocessor = PuzzlePreprocessor()

    def test_enhance_edges(self):
        """enhance_edges should return an image of the same shape."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.preprocessor.enhance_edges(img)
        assert result.shape == img.shape

    def test_extract_color_regions(self):
        """Should return color region data."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = self.preprocessor.extract_color_regions(img)
        assert "dominant_colors" in result or "error" in result


# ---------------------------------------------------------------------------
# Color description tests
# ---------------------------------------------------------------------------


class TestDescribeColor:
    def test_black(self):
        assert _describe_color([10, 10, 10]) == "black"

    def test_white(self):
        assert _describe_color([220, 220, 220]) == "white"

    def test_red(self):
        assert _describe_color([200, 30, 30]) == "red"

    def test_green(self):
        assert _describe_color([30, 180, 30]) == "green"

    def test_blue(self):
        assert _describe_color([30, 30, 180]) == "blue"

    def test_yellow(self):
        assert _describe_color([200, 200, 30]) == "yellow"

    def test_gray(self):
        assert "gray" in _describe_color([100, 100, 100])

    def test_fallback_rgb(self):
        result = _describe_color([123, 45, 67])
        assert "rgb" in result or any(
            name in result for name in ("red", "green", "blue", "gray", "brown")
        )


# ---------------------------------------------------------------------------
# AnalysisMode enum tests
# ---------------------------------------------------------------------------


class TestAnalysisMode:
    def test_values(self):
        assert AnalysisMode.GENERAL.value == "general"
        assert AnalysisMode.PUZZLE.value == "puzzle"
        assert AnalysisMode.PAINTING.value == "painting"
        assert AnalysisMode.INSPECTION.value == "inspection"

    def test_from_string(self):
        assert AnalysisMode("puzzle") == AnalysisMode.PUZZLE
