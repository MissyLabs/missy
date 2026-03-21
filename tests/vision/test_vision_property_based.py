"""Property-based tests for vision subsystem.


Uses Hypothesis to find edge cases in:
- Orientation detection (any aspect ratio → valid result)
- Pipeline quality assessment (any image → valid dict)
- Change detection constants (weights always sum to 1)
- Intent classification (any string → valid IntentResult)
- Color description (any RGB → non-empty string)
- Scene memory hash (deterministic for same input)
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Orientation detection properties
# ---------------------------------------------------------------------------


class TestOrientationProperties:
    @given(
        h=st.integers(min_value=1, max_value=10000),
        w=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=100)
    def test_detect_always_returns_valid_result(self, h: int, w: int) -> None:
        from missy.vision.orientation import Orientation, detect_orientation

        img = np.zeros((h, w, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert isinstance(result.detected, Orientation)
        assert 0.0 <= result.confidence <= 1.0
        assert result.method != ""

    @given(
        h=st.integers(min_value=1, max_value=5000),
        w=st.integers(min_value=1, max_value=5000),
    )
    @settings(max_examples=50)
    def test_auto_correct_preserves_pixel_count(self, h: int, w: int) -> None:
        """auto_correct may transpose dimensions but pixel count stays same."""
        from missy.vision.orientation import auto_correct

        img = np.zeros((h, w, 3), dtype=np.uint8)
        corrected, result = auto_correct(img)
        # Total pixels should be preserved (rotation doesn't change count)
        orig_pixels = h * w
        new_h, new_w = corrected.shape[:2]
        assert new_h * new_w == orig_pixels

    @given(
        h=st.integers(min_value=1, max_value=5000),
        w=st.integers(min_value=1, max_value=5000),
    )
    @settings(max_examples=50)
    def test_landscape_always_detected_as_normal(self, h: int, w: int) -> None:
        """Images with aspect ratio > 1.25 should be NORMAL."""
        assume(w / h > 1.25)
        from missy.vision.orientation import Orientation, detect_orientation

        img = np.zeros((h, w, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL


# ---------------------------------------------------------------------------
# Pipeline quality assessment properties
# ---------------------------------------------------------------------------


class TestPipelineQualityProperties:
    @given(
        h=st.integers(min_value=10, max_value=500),
        w=st.integers(min_value=10, max_value=500),
    )
    @settings(max_examples=50)
    def test_assess_quality_returns_valid_dict(self, h: int, w: int) -> None:
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.zeros((h, w, 3), dtype=np.uint8)
        result = pipeline.assess_quality(img)
        assert isinstance(result, dict)
        assert "quality" in result
        assert result["quality"] in ("poor", "fair", "good", "excellent")
        assert "brightness" in result
        assert "contrast" in result
        assert "sharpness" in result
        assert result["width"] == w
        assert result["height"] == h


# ---------------------------------------------------------------------------
# Intent classification properties
# ---------------------------------------------------------------------------


class TestIntentClassificationProperties:
    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=100)
    def test_classify_returns_valid_result(self, text: str) -> None:
        from missy.vision.intent import ActivationDecision, VisionIntent, VisionIntentClassifier

        classifier = VisionIntentClassifier()
        result = classifier.classify(text)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.intent, VisionIntent)
        assert isinstance(result.decision, ActivationDecision)

    @given(text=st.from_regex(r"^\s*$", fullmatch=True))
    @settings(max_examples=50)
    def test_empty_or_whitespace_skips_vision(self, text: str) -> None:
        """Pure whitespace or empty strings should not trigger vision."""
        from missy.vision.intent import ActivationDecision, VisionIntentClassifier

        classifier = VisionIntentClassifier()
        result = classifier.classify(text)
        assert result.decision == ActivationDecision.SKIP


# ---------------------------------------------------------------------------
# Color description properties
# ---------------------------------------------------------------------------


class TestColorDescriptionProperties:
    @given(
        r=st.integers(min_value=0, max_value=255),
        g=st.integers(min_value=0, max_value=255),
        b=st.integers(min_value=0, max_value=255),
    )
    @settings(max_examples=200)
    def test_describe_color_always_returns_string(self, r: int, g: int, b: int) -> None:
        from missy.vision.analysis import _describe_color

        result = _describe_color([r, g, b])
        assert isinstance(result, str)
        assert len(result) > 0

    @given(
        r=st.integers(min_value=0, max_value=255),
        g=st.integers(min_value=0, max_value=255),
        b=st.integers(min_value=0, max_value=255),
    )
    @settings(max_examples=100)
    def test_describe_color_deterministic(self, r: int, g: int, b: int) -> None:
        from missy.vision.analysis import _describe_color

        a = _describe_color([r, g, b])
        b_result = _describe_color([r, g, b])
        assert a == b_result


# ---------------------------------------------------------------------------
# Scene memory hash properties
# ---------------------------------------------------------------------------


class TestSceneMemoryHashProperties:
    @given(
        h=st.integers(min_value=8, max_value=200),
        w=st.integers(min_value=8, max_value=200),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(max_examples=30)
    def test_thumbnail_hash_deterministic(self, h: int, w: int, seed: int) -> None:
        from missy.vision.scene_memory import SceneFrame

        rng = np.random.default_rng(seed)
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        f1 = SceneFrame(frame_id=1, image=img)
        f2 = SceneFrame(frame_id=2, image=img.copy())
        assert f1.thumbnail_hash == f2.thumbnail_hash

    @given(
        h=st.integers(min_value=8, max_value=200),
        w=st.integers(min_value=8, max_value=200),
    )
    @settings(max_examples=30)
    def test_thumbnail_hash_length(self, h: int, w: int) -> None:
        from missy.vision.scene_memory import SceneFrame

        img = np.zeros((h, w, 3), dtype=np.uint8)
        frame = SceneFrame(frame_id=1, image=img)
        assert isinstance(frame.thumbnail_hash, str)
        assert len(frame.thumbnail_hash) == 16  # 64-bit hash → 16 hex chars


# ---------------------------------------------------------------------------
# Token approximation properties
# ---------------------------------------------------------------------------


class TestTokenApproxProperties:
    @given(text=st.text(min_size=0, max_size=10000))
    @settings(max_examples=100, deadline=1000)
    def test_approx_tokens_always_positive(self, text: str) -> None:
        from missy.agent.context import _approx_tokens

        result = _approx_tokens(text)
        assert result >= 1

    @given(text=st.text(min_size=1, max_size=10000))
    @settings(max_examples=50, deadline=1000)
    def test_approx_tokens_monotonic(self, text: str) -> None:
        """Longer strings should produce equal or more tokens."""
        from missy.agent.context import _approx_tokens

        full = _approx_tokens(text)
        half = _approx_tokens(text[: len(text) // 2])
        assert full >= half
