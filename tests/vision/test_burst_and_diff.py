"""Tests for burst capture mode and image diff visualization."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.capture import CameraHandle, CaptureConfig, CaptureResult
from missy.vision.scene_memory import SceneFrame, SceneSession, TaskType


# ---------------------------------------------------------------------------
# Burst capture tests
# ---------------------------------------------------------------------------


class TestBurstCapture:
    def _make_handle(self) -> CameraHandle:
        config = CaptureConfig(max_retries=1, warmup_frames=0)
        handle = CameraHandle("/dev/video0", config)
        mock_cap = MagicMock()
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mock_cap.read.return_value = (True, frame)
        mock_cap.isOpened.return_value = True
        handle._cap = mock_cap
        handle._opened = True
        return handle

    def test_burst_capture_returns_correct_count(self):
        handle = self._make_handle()
        results = handle.capture_burst(count=5, interval=0)
        assert len(results) == 5
        assert all(r.success for r in results)

    def test_burst_single_frame(self):
        handle = self._make_handle()
        results = handle.capture_burst(count=1, interval=0)
        assert len(results) == 1
        assert results[0].success

    def test_burst_zero_count_raises(self):
        handle = self._make_handle()
        with pytest.raises(ValueError, match="count must be >= 1"):
            handle.capture_burst(count=0)

    def test_burst_negative_count_raises(self):
        handle = self._make_handle()
        with pytest.raises(ValueError, match="count must be >= 1"):
            handle.capture_burst(count=-1)

    def test_burst_clamped_to_20(self):
        handle = self._make_handle()
        results = handle.capture_burst(count=50, interval=0)
        assert len(results) == 20

    def test_burst_with_some_failures(self):
        config = CaptureConfig(max_retries=1, warmup_frames=0)
        handle = CameraHandle("/dev/video0", config)
        mock_cap = MagicMock()
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128

        call_count = [0]

        def mock_read():
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return (False, None)
            return (True, frame)

        mock_cap.read.side_effect = mock_read
        mock_cap.isOpened.return_value = True
        handle._cap = mock_cap
        handle._opened = True

        results = handle.capture_burst(count=4, interval=0)
        assert len(results) == 4
        assert any(r.success for r in results)


class TestBestCapture:
    def _make_handle_varying_sharpness(self) -> CameraHandle:
        config = CaptureConfig(max_retries=1, warmup_frames=0)
        handle = CameraHandle("/dev/video0", config)
        mock_cap = MagicMock()

        # Simulate frames with varying sharpness
        call_count = [0]

        def mock_read():
            call_count[0] += 1
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            if call_count[0] == 2:
                # Add sharp edges to frame 2
                frame[40:60, :] = 255
                frame[:, 40:60] = 0
            return (True, frame)

        mock_cap.read.side_effect = mock_read
        mock_cap.isOpened.return_value = True
        handle._cap = mock_cap
        handle._opened = True
        return handle

    @patch("missy.vision.capture._get_cv2")
    def test_best_returns_single_result(self, mock_get_cv2):
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        config = CaptureConfig(max_retries=1, warmup_frames=0)
        handle = CameraHandle("/dev/video0", config)
        mock_cap = MagicMock()
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mock_cap.read.return_value = (True, frame)
        mock_cap.isOpened.return_value = True
        handle._cap = mock_cap
        handle._opened = True

        gray = np.ones((100, 100), dtype=np.uint8) * 128
        mock_cv2.cvtColor.return_value = gray
        laplacian = MagicMock()
        laplacian.var.return_value = 50.0
        mock_cv2.Laplacian.return_value = laplacian
        mock_cv2.CV_64F = 6

        result = handle.capture_best(burst_count=3)
        assert isinstance(result, CaptureResult)
        assert result.success

    def test_best_with_all_failures(self):
        config = CaptureConfig(max_retries=1, warmup_frames=0)
        handle = CameraHandle("/dev/video0", config)
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)
        mock_cap.isOpened.return_value = True
        handle._cap = mock_cap
        handle._opened = True

        result = handle.capture_best(burst_count=3)
        assert not result.success
        assert "No successful frames" in result.error


# ---------------------------------------------------------------------------
# Scene diff visualization tests
# ---------------------------------------------------------------------------


class TestSceneDiffVisualization:
    def _make_session(self) -> SceneSession:
        return SceneSession("test", TaskType.GENERAL, max_frames=10)

    def _make_frame(self, value: int, frame_id: int) -> SceneFrame:
        img = np.ones((100, 100, 3), dtype=np.uint8) * value
        return SceneFrame(frame_id=frame_id, image=img)

    def test_visualize_identical_frames(self):
        session = self._make_session()
        fa = self._make_frame(100, 1)
        fb = self._make_frame(100, 2)
        result = session.visualize_change(fa, fb)
        # Result may be None if cv2 not available, or an ndarray if mocked
        assert result is None or isinstance(result, np.ndarray)

    def test_visualize_different_frames(self):
        session = self._make_session()
        fa = self._make_frame(0, 1)
        fb = self._make_frame(255, 2)
        result = session.visualize_change(fa, fb)
        assert result is None or isinstance(result, np.ndarray)

    def test_visualize_handles_exception(self):
        """Should return None on error, not raise."""
        session = self._make_session()
        fa = SceneFrame(frame_id=1, image=np.array([]))  # invalid shape
        fb = SceneFrame(frame_id=2, image=np.array([]))
        result = session.visualize_change(fa, fb)
        assert result is None


# ---------------------------------------------------------------------------
# VisionBurstCaptureTool tests
# ---------------------------------------------------------------------------


class TestVisionBurstCaptureTool:
    def test_tool_instantiation(self):
        from missy.tools.builtin.vision_tools import VisionBurstCaptureTool

        tool = VisionBurstCaptureTool()
        assert tool.name == "vision_burst"
        assert "burst" in tool.description.lower()

    def test_tool_schema(self):
        from missy.tools.builtin.vision_tools import VisionBurstCaptureTool

        tool = VisionBurstCaptureTool()
        schema = tool.get_schema()
        assert "count" in schema["parameters"]["properties"]
        assert "best_only" in schema["parameters"]["properties"]

    @patch("missy.vision.discovery.find_preferred_camera", return_value=None)
    def test_tool_no_camera(self, mock_find):
        from missy.tools.builtin.vision_tools import VisionBurstCaptureTool

        tool = VisionBurstCaptureTool()
        result = tool.execute(count=3)
        assert not result.success
        assert "No camera" in result.error
