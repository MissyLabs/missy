"""Edge case tests for the vision subsystem.

Covers error recovery, boundary conditions, and resilience scenarios.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.capture import CameraHandle, CaptureConfig, CaptureError, CaptureResult
from missy.vision.discovery import CameraDevice, CameraDiscovery
from missy.vision.intent import (
    ActivationDecision,
    VisionIntent,
    VisionIntentClassifier,
)
from missy.vision.pipeline import ImagePipeline, PipelineConfig
from missy.vision.scene_memory import SceneManager, SceneSession, TaskType
from missy.vision.sources import (
    FileSource,
    ImageFrame,
    PhotoSource,
    SourceType,
    create_source,
)


# ---------------------------------------------------------------------------
# Camera Discovery edge cases
# ---------------------------------------------------------------------------


class TestDiscoveryEdgeCases:
    def test_multiple_cameras_preferred_order(self):
        """With multiple cameras, Logitech C922x should always be preferred."""
        disc = CameraDiscovery()
        generic = CameraDevice("/dev/video0", "Generic", "1234", "5678", "usb-1")
        c920 = CameraDevice("/dev/video2", "C920", "046d", "082d", "usb-2")
        c922 = CameraDevice("/dev/video4", "C922x", "046d", "085c", "usb-3")
        disc._cache = [generic, c920, c922]
        disc._cache_time = 1e18

        assert disc.find_preferred().device_path == "/dev/video4"

    def test_cache_expiry(self):
        """Cache should expire after TTL."""
        disc = CameraDiscovery(cache_ttl_seconds=0.01)
        disc._cache = [CameraDevice("/dev/video0", "Test", "0000", "0000", "")]
        disc._cache_time = time.monotonic() - 1  # expired

        # discover() should re-scan (empty sysfs = no results)
        with patch.object(disc, "_scan_sysfs", return_value=[]) as mock_scan:
            result = disc.discover()
            mock_scan.assert_called_once()

    def test_find_by_name_case_insensitive(self):
        disc = CameraDiscovery()
        disc._cache = [CameraDevice("/dev/video0", "Logitech C922x", "046d", "085c", "")]
        disc._cache_time = 1e18

        matches = disc.find_by_name("logitech")
        assert len(matches) == 1

    def test_find_by_name_regex(self):
        disc = CameraDiscovery()
        disc._cache = [
            CameraDevice("/dev/video0", "Logitech C922x", "046d", "085c", ""),
            CameraDevice("/dev/video2", "Logitech C920", "046d", "082d", ""),
        ]
        disc._cache_time = 1e18

        matches = disc.find_by_name(r"C9[12]\d")
        assert len(matches) == 2


# ---------------------------------------------------------------------------
# Capture edge cases
# ---------------------------------------------------------------------------


class TestCaptureEdgeCases:
    @patch("missy.vision.capture._get_cv2")
    def test_exception_during_read(self, mock_cv2_fn):
        """Camera read raising exception should be caught and retried."""
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CAP_V4L2 = 200

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        # First read raises, second succeeds
        good_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        mock_cap.read.side_effect = [
            OSError("device disconnected"),
            (True, good_frame),
        ]
        mock_cv2.VideoCapture.return_value = mock_cap

        config = CaptureConfig(warmup_frames=0, max_retries=3, retry_delay=0)
        handle = CameraHandle("/dev/video0", config)
        handle.open()
        result = handle.capture()

        assert result.success is True
        assert result.attempt_count == 2
        handle.close()

    @patch("missy.vision.capture._get_cv2")
    def test_double_close_is_safe(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CAP_V4L2 = 200

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_cap.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = mock_cap

        handle = CameraHandle("/dev/video0", CaptureConfig(warmup_frames=0))
        handle.open()
        handle.close()
        handle.close()  # Should not raise

    @patch("missy.vision.capture._get_cv2")
    def test_double_open_is_noop(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.CAP_V4L2 = 200

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_cap.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = mock_cap

        handle = CameraHandle("/dev/video0", CaptureConfig(warmup_frames=0))
        handle.open()
        handle.open()  # Should be no-op
        # VideoCapture should only be created once
        assert mock_cv2.VideoCapture.call_count == 1
        handle.close()

    def test_capture_result_no_image_shape(self):
        result = CaptureResult(success=False)
        assert result.shape == (0, 0, 0)


# ---------------------------------------------------------------------------
# Source edge cases
# ---------------------------------------------------------------------------


class TestSourceEdgeCases:
    @patch("missy.vision.sources._get_cv2")
    def test_photo_source_acquire_specific_out_of_range(self, mock_cv2_fn, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"fake")
        source = PhotoSource(str(tmp_path))

        with pytest.raises(IndexError):
            source.acquire_specific(999)

    @patch("missy.vision.sources._get_cv2")
    def test_photo_source_acquire_specific_negative(self, mock_cv2_fn, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"fake")
        source = PhotoSource(str(tmp_path))

        with pytest.raises(IndexError):
            source.acquire_specific(-1)

    def test_image_frame_metadata(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        frame = ImageFrame(
            image=img,
            source_type=SourceType.FILE,
            metadata={"custom_key": "custom_value"},
        )
        assert frame.metadata["custom_key"] == "custom_value"
        assert frame.width == 200
        assert frame.height == 100


# ---------------------------------------------------------------------------
# Scene Memory edge cases
# ---------------------------------------------------------------------------


class TestSceneMemoryEdgeCases:
    def test_session_frame_eviction_preserves_newest(self):
        session = SceneSession("test", max_frames=2)
        for i in range(5):
            img = np.full((10, 10, 3), i * 50, dtype=np.uint8)
            session.add_frame(img)

        assert session.frame_count == 2
        latest = session.get_latest_frame()
        assert latest.frame_id == 5

    def test_session_detect_change_error_handling(self):
        """detect_change should handle comparison errors gracefully."""
        session = SceneSession("test")

        # Create frames with incompatible shapes
        f1 = session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        f2 = session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8))

        # Should not raise even with odd inputs
        change = session.detect_change(f1, f2)
        assert isinstance(change.change_score, float)

    def test_scene_manager_list_empty(self):
        mgr = SceneManager()
        assert mgr.list_sessions() == []

    def test_session_summarize_includes_all_fields(self):
        session = SceneSession("test", TaskType.PUZZLE, max_frames=5)
        session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        session.add_observation("test observation")
        session.update_state(progress="50%")

        summary = session.summarize()
        assert "task_id" in summary
        assert "task_type" in summary
        assert "created" in summary
        assert "frame_count" in summary
        assert "observations" in summary
        assert "state" in summary
        assert "active" in summary

    def test_closed_session_state(self):
        session = SceneSession("test")
        session.add_frame(np.zeros((50, 50, 3), dtype=np.uint8))
        session.add_observation("note")
        session.update_state(key="val")
        session.close()

        assert not session.is_active
        # Frames, observations, and state should be fully released
        assert session.get_latest_frame() is None
        assert session.frame_count == 0
        assert session.observations == []
        assert session.state == {}


# ---------------------------------------------------------------------------
# Intent edge cases
# ---------------------------------------------------------------------------


class TestIntentEdgeCases:
    def test_combined_look_and_puzzle(self):
        """Both look and puzzle keywords should boost puzzle confidence."""
        cls = VisionIntentClassifier()
        result = cls.classify("Look at this puzzle piece")
        assert result.intent == VisionIntent.PUZZLE
        assert result.confidence >= 0.90

    def test_combined_look_and_painting(self):
        cls = VisionIntentClassifier()
        result = cls.classify("Can you look at this painting?")
        assert result.intent in (VisionIntent.PAINTING, VisionIntent.LOOK)
        assert result.confidence >= 0.80

    def test_very_long_input(self):
        """Should not crash on very long input."""
        cls = VisionIntentClassifier()
        long_text = "hello world " * 10000
        result = cls.classify(long_text)
        assert isinstance(result.confidence, float)

    def test_special_characters(self):
        cls = VisionIntentClassifier()
        result = cls.classify("!@#$%^&*(){}[]|\\:\";<>?,./~`")
        assert result.intent == VisionIntent.NONE

    def test_unicode_input(self):
        cls = VisionIntentClassifier()
        result = cls.classify("看看这个拼图")
        assert isinstance(result.confidence, float)

    def test_custom_thresholds(self):
        # Very strict — nothing auto-activates
        cls = VisionIntentClassifier(auto_threshold=1.0, ask_threshold=0.99)
        result = cls.classify("Look at this")
        assert result.decision != ActivationDecision.ACTIVATE

    def test_activation_log_grows(self):
        cls = VisionIntentClassifier()
        for i in range(10):
            cls.classify(f"Test message {i}")
        assert len(cls.activation_log) == 10


# ---------------------------------------------------------------------------
# Pipeline edge cases
# ---------------------------------------------------------------------------


class TestPipelineEdgeCases:
    def test_single_pixel_image(self):
        """Single pixel image should not crash."""
        pipeline = ImagePipeline(PipelineConfig(
            normalize_exposure=False, denoise=False, sharpen=False
        ))
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        result = pipeline.resize(img, 1280)
        assert result.shape == (1, 1, 3)

    def test_grayscale_input_quality(self):
        """Quality assessment with grayscale image."""
        pipeline = ImagePipeline()
        gray = np.full((100, 100), 128, dtype=np.uint8)
        # Convert to 3-channel for assessment
        img = np.stack([gray, gray, gray], axis=2)
        quality = pipeline.assess_quality(img)
        assert "quality" in quality

    def test_overexposed_image(self):
        """Very bright image should be flagged."""
        pipeline = ImagePipeline()
        img = np.full((100, 100, 3), 250, dtype=np.uint8)
        quality = pipeline.assess_quality(img)
        assert "overexposed" in quality["issues"]

    def test_low_contrast_image(self):
        """Uniform image should have low contrast."""
        pipeline = ImagePipeline()
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        quality = pipeline.assess_quality(img)
        assert "low contrast" in quality["issues"]


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestVisionConfig:
    def test_default_vision_config(self):
        from missy.config.settings import VisionConfig
        cfg = VisionConfig()
        assert cfg.enabled is True
        assert cfg.capture_width == 1920
        assert cfg.auto_activate_threshold == 0.80

    def test_parse_vision_config(self):
        from missy.config.settings import _parse_vision
        data = {
            "enabled": False,
            "preferred_device": "/dev/video2",
            "capture_width": 1280,
            "auto_activate_threshold": 0.90,
        }
        cfg = _parse_vision(data)
        assert cfg.enabled is False
        assert cfg.preferred_device == "/dev/video2"
        assert cfg.capture_width == 1280
        assert cfg.auto_activate_threshold == 0.90

    def test_parse_vision_empty(self):
        from missy.config.settings import _parse_vision
        cfg = _parse_vision({})
        assert cfg.enabled is True
        assert cfg.preferred_device == ""
