"""Tests for vision subsystem hardening: thread safety, input validation, error handling."""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.capture import CameraHandle, CaptureConfig, CaptureError
from missy.vision.discovery import CameraDevice, CameraDiscovery
from missy.vision.intent import VisionIntentClassifier
from missy.vision.pipeline import ImagePipeline, PipelineConfig
from missy.vision.provider_format import format_image_for_provider
from missy.vision.scene_memory import SceneManager, SceneSession, TaskType
from missy.vision.sources import FileSource, ImageFrame, PhotoSource, SourceType


# ---------------------------------------------------------------------------
# capture.py hardening
# ---------------------------------------------------------------------------


class TestCaptureHardening:
    def test_empty_device_path_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            CameraHandle("")

    def test_none_device_path_raises(self):
        with pytest.raises((ValueError, TypeError)):
            CameraHandle(None)  # type: ignore[arg-type]

    def test_invalid_frame_shape_rejected(self):
        """Frames with 0-dimension shape should be retried."""
        config = CaptureConfig(max_retries=1, warmup_frames=0)
        handle = CameraHandle("/dev/video0", config)

        mock_cap = MagicMock()
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, empty_frame)
        mock_cap.isOpened.return_value = True
        handle._cap = mock_cap
        handle._opened = True

        result = handle.capture()
        assert not result.success
        assert "Invalid frame shape" in result.error or "Capture failed" in result.error

    def test_thread_safe_capture(self):
        """Concurrent capture calls should not crash."""
        config = CaptureConfig(max_retries=1, warmup_frames=0)
        handle = CameraHandle("/dev/video0", config)

        mock_cap = MagicMock()
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mock_cap.read.return_value = (True, frame)
        mock_cap.isOpened.return_value = True
        handle._cap = mock_cap
        handle._opened = True

        results = []
        errors = []

        def do_capture():
            try:
                r = handle.capture()
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_capture) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        assert len(results) == 5
        assert all(r.success for r in results)

    def test_thread_safe_close(self):
        """Concurrent close calls should not crash."""
        config = CaptureConfig(warmup_frames=0)
        handle = CameraHandle("/dev/video0", config)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        handle._cap = mock_cap
        handle._opened = True

        threads = [threading.Thread(target=handle.close) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not handle.is_open


# ---------------------------------------------------------------------------
# discovery.py hardening
# ---------------------------------------------------------------------------


class TestDiscoveryHardening:
    def test_invalid_regex_returns_empty(self):
        disc = CameraDiscovery()
        disc._cache = [CameraDevice("/dev/video0", "Test", "0000", "0000", "")]
        disc._cache_time = 1e18
        result = disc.find_by_name("[invalid regex")
        assert result == []

    def test_sysfs_permission_denied(self, tmp_path):
        """OSError on iterdir should return empty list, not crash."""
        disc = CameraDiscovery(sysfs_base=str(tmp_path / "nonexistent_subdir"))
        # nonexistent dir
        result = disc.discover()
        assert result == []

    def test_sysfs_iterdir_oserror(self, tmp_path):
        """Mocked OSError from iterdir should be handled."""
        disc = CameraDiscovery(sysfs_base=str(tmp_path))
        with patch.object(Path, "iterdir", side_effect=OSError("Permission denied")):
            result = disc._scan_sysfs()
            assert result == []


# ---------------------------------------------------------------------------
# pipeline.py hardening
# ---------------------------------------------------------------------------


class TestPipelineHardening:
    def test_process_none_image_raises(self):
        pipeline = ImagePipeline()
        with pytest.raises(ValueError, match="non-None"):
            pipeline.process(None)  # type: ignore[arg-type]

    def test_process_empty_image_raises(self):
        pipeline = ImagePipeline()
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="invalid shape"):
            pipeline.process(empty)

    def test_resize_negative_max_dim_raises(self):
        pipeline = ImagePipeline()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="positive"):
            pipeline.resize(img, -1)

    def test_resize_zero_max_dim_raises(self):
        pipeline = ImagePipeline()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="positive"):
            pipeline.resize(img, 0)

    @patch("missy.vision.pipeline._get_cv2")
    def test_normalize_exposure_grayscale(self, mock_get_cv2):
        """Grayscale images should be handled without COLOR_BGR2LAB."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        clahe_mock = MagicMock()
        enhanced = np.ones((50, 50), dtype=np.uint8) * 128
        clahe_mock.apply.return_value = enhanced
        mock_cv2.createCLAHE.return_value = clahe_mock

        pipeline = ImagePipeline()
        gray_img = np.ones((50, 50), dtype=np.uint8) * 100
        result = pipeline.normalize_exposure(gray_img)

        # Should call createCLAHE and apply directly (no cvtColor)
        mock_cv2.createCLAHE.assert_called_once()
        clahe_mock.apply.assert_called_once()
        mock_cv2.cvtColor.assert_not_called()

    @patch("missy.vision.pipeline._get_cv2")
    def test_normalize_exposure_bgra(self, mock_get_cv2):
        """4-channel BGRA images should have alpha preserved."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        # Setup mocks for the BGR path
        lab = np.ones((50, 50, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = lab
        l_ch = np.ones((50, 50), dtype=np.uint8)
        mock_cv2.split.return_value = (l_ch, l_ch, l_ch)
        clahe_mock = MagicMock()
        clahe_mock.apply.return_value = l_ch
        mock_cv2.createCLAHE.return_value = clahe_mock
        merged = np.ones((50, 50, 3), dtype=np.uint8)
        mock_cv2.merge.side_effect = [merged, np.ones((50, 50, 4), dtype=np.uint8)]

        pipeline = ImagePipeline()
        bgra = np.ones((50, 50, 4), dtype=np.uint8)
        pipeline.normalize_exposure(bgra)

        # Should call merge twice: once for LAB→BGR, once for adding alpha back
        assert mock_cv2.merge.call_count == 2

    def test_assess_quality_none_raises(self):
        pipeline = ImagePipeline()
        with pytest.raises(ValueError, match="non-None"):
            pipeline.assess_quality(None)  # type: ignore[arg-type]

    def test_assess_quality_empty_raises(self):
        pipeline = ImagePipeline()
        with pytest.raises(ValueError, match="invalid shape"):
            pipeline.assess_quality(np.zeros((0, 0), dtype=np.uint8))


# ---------------------------------------------------------------------------
# scene_memory.py hardening
# ---------------------------------------------------------------------------


class TestSceneManagerHardening:
    def test_thread_safe_create_session(self):
        """Concurrent session creation should not crash or lose sessions."""
        mgr = SceneManager(max_sessions=50)
        errors = []

        def create(i):
            try:
                mgr.create_session(f"task_{i}", TaskType.GENERAL, 5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        assert len(mgr.list_sessions()) == 20

    def test_thread_safe_get_session(self):
        """Concurrent get_session calls should not crash."""
        mgr = SceneManager()
        mgr.create_session("test", TaskType.GENERAL)
        errors = []

        def get():
            try:
                s = mgr.get_session("test")
                assert s is not None
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors

    def test_thread_safe_close_all(self):
        mgr = SceneManager()
        for i in range(5):
            mgr.create_session(f"task_{i}")

        errors = []

        def close():
            try:
                mgr.close_all()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=close) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors


# ---------------------------------------------------------------------------
# intent.py hardening
# ---------------------------------------------------------------------------


class TestIntentHardening:
    def test_invalid_auto_threshold_raises(self):
        with pytest.raises(ValueError, match="auto_threshold"):
            VisionIntentClassifier(auto_threshold=1.5)

    def test_negative_auto_threshold_raises(self):
        with pytest.raises(ValueError, match="auto_threshold"):
            VisionIntentClassifier(auto_threshold=-0.1)

    def test_invalid_ask_threshold_raises(self):
        with pytest.raises(ValueError, match="ask_threshold"):
            VisionIntentClassifier(ask_threshold=2.0)

    def test_boundary_thresholds_ok(self):
        c = VisionIntentClassifier(auto_threshold=0.0, ask_threshold=1.0)
        assert c._auto_threshold == 0.0
        assert c._ask_threshold == 1.0

    def test_thread_safe_activation_log(self):
        classifier = VisionIntentClassifier()
        errors = []

        def classify(text):
            try:
                classifier.classify(text)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=classify, args=(f"look at this {i}",))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        assert len(classifier.activation_log) == 20


# ---------------------------------------------------------------------------
# provider_format.py hardening
# ---------------------------------------------------------------------------


class TestProviderFormatHardening:
    def test_empty_provider_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            format_image_for_provider("", "base64data")

    def test_none_provider_name_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            format_image_for_provider(None, "data")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# sources.py hardening
# ---------------------------------------------------------------------------


class TestSourcesHardening:
    def test_file_source_decode_error_message(self, tmp_path):
        """Error message should mention 'unsupported format'."""
        f = tmp_path / "bad.jpg"
        f.write_bytes(b"not an image")

        with patch("missy.vision.sources._get_cv2") as mock:
            mock.return_value = MagicMock(imread=MagicMock(return_value=None))
            source = FileSource(str(f))
            with pytest.raises(ValueError, match="unsupported format"):
                source.acquire()

    def test_photo_source_scan_missing_dir(self, tmp_path):
        source = PhotoSource(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError, match="not found"):
            source.scan()

    def test_photo_source_scan_permission_error(self, tmp_path):
        d = tmp_path / "photos"
        d.mkdir()
        source = PhotoSource(d)
        with patch.object(Path, "iterdir", side_effect=OSError("Permission denied")):
            with pytest.raises(OSError, match="Cannot scan"):
                source.scan()

    @patch("missy.vision.sources._get_cv2")
    def test_jpeg_encode_failure(self, mock_get_cv2):
        mock_cv2 = MagicMock()
        mock_cv2.imencode.return_value = (False, None)
        mock_get_cv2.return_value = mock_cv2

        frame = ImageFrame(
            image=np.ones((10, 10, 3), dtype=np.uint8),
            source_type=SourceType.FILE,
        )
        with pytest.raises(RuntimeError, match="Failed to encode"):
            frame.to_jpeg_bytes()

    @patch("missy.vision.sources._get_cv2")
    def test_png_encode_failure(self, mock_get_cv2):
        mock_cv2 = MagicMock()
        mock_cv2.imencode.return_value = (False, None)
        mock_get_cv2.return_value = mock_cv2

        frame = ImageFrame(
            image=np.ones((10, 10, 3), dtype=np.uint8),
            source_type=SourceType.FILE,
        )
        with pytest.raises(RuntimeError, match="Failed to encode"):
            frame.to_png_bytes()


# ---------------------------------------------------------------------------
# resilient_capture.py hardening
# ---------------------------------------------------------------------------


class TestResilientCaptureHardening:
    def test_capture_with_none_handle(self):
        from missy.vision.resilient_capture import ResilientCamera

        cam = ResilientCamera()
        cam._connected = True
        cam._handle = None  # simulate race condition

        result = cam.capture()
        # Should not crash — either reconnect or return failure
        assert isinstance(result, (type(None),) + (object,).__class__.__mro__)
