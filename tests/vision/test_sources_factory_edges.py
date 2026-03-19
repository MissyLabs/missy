"""Tests for source factory, shutdown, and miscellaneous edge cases.

Covers:
- Source factory with all types
- Source factory with invalid types
- WebcamSource device path validation
- FileSource MAX_FILE_SIZE and MAX_DIMENSION
- Shutdown idempotency
- Shutdown reset
- SceneManager session eviction
- Orientation detection edge cases
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Source factory
# ---------------------------------------------------------------------------


class TestSourceFactory:
    """Test create_source() factory function."""

    def test_create_webcam_source(self):
        from missy.vision.sources import SourceType, create_source

        source = create_source(SourceType.WEBCAM, device_path="/dev/video0")
        assert source.source_type() == SourceType.WEBCAM

    def test_create_file_source(self):
        from missy.vision.sources import SourceType, create_source

        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            source = create_source(SourceType.FILE, file_path=f.name)
            assert source.source_type() == SourceType.FILE

    def test_create_screenshot_source(self):
        from missy.vision.sources import SourceType, create_source

        source = create_source(SourceType.SCREENSHOT)
        assert source.source_type() == SourceType.SCREENSHOT

    def test_create_photo_source(self):
        from missy.vision.sources import SourceType, create_source

        with tempfile.TemporaryDirectory() as d:
            source = create_source(SourceType.PHOTO, directory=d)
            assert source.source_type() == SourceType.PHOTO

    def test_create_file_source_requires_path(self):
        from missy.vision.sources import SourceType, create_source

        with pytest.raises(ValueError, match="file_path required"):
            create_source(SourceType.FILE)

    def test_create_photo_source_requires_directory(self):
        from missy.vision.sources import SourceType, create_source

        with pytest.raises(ValueError, match="directory required"):
            create_source(SourceType.PHOTO)

    def test_create_source_from_string(self):
        from missy.vision.sources import SourceType, create_source

        source = create_source("webcam", device_path="/dev/video0")
        assert source.source_type() == SourceType.WEBCAM

    def test_create_source_invalid_type(self):
        from missy.vision.sources import create_source

        with pytest.raises(ValueError):
            create_source("nonexistent")


class TestWebcamSourceValidation:
    """Test WebcamSource device path validation."""

    def test_valid_device_path(self):
        from missy.vision.sources import WebcamSource

        ws = WebcamSource("/dev/video0")
        assert ws._device_path == "/dev/video0"

    def test_invalid_device_path(self):
        from missy.vision.sources import WebcamSource

        with pytest.raises(ValueError, match="Invalid device path"):
            WebcamSource("/dev/sda1")

    def test_invalid_device_path_injection(self):
        from missy.vision.sources import WebcamSource

        with pytest.raises(ValueError, match="Invalid device path"):
            WebcamSource("/dev/video0; rm -rf /")

    def test_relative_path_rejected(self):
        from missy.vision.sources import WebcamSource

        with pytest.raises(ValueError, match="Invalid device path"):
            WebcamSource("video0")


class TestFileSourceLimits:
    """Test FileSource file size and dimension checks."""

    def test_empty_file_rejected(self):
        from missy.vision.sources import FileSource

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"")
            path = f.name

        try:
            source = FileSource(path)
            with pytest.raises(ValueError, match="empty"):
                source.acquire()
        finally:
            Path(path).unlink()

    def test_nonexistent_file_raises(self):
        from missy.vision.sources import FileSource

        source = FileSource("/nonexistent/file.jpg")
        with pytest.raises(FileNotFoundError):
            source.acquire()

    def test_is_available_false_for_missing(self):
        from missy.vision.sources import FileSource

        source = FileSource("/nonexistent/file.jpg")
        assert not source.is_available()


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    """Test vision shutdown idempotency and reset."""

    def test_shutdown_idempotent(self):
        from missy.vision.shutdown import reset_shutdown_state, vision_shutdown

        reset_shutdown_state()
        result1 = vision_shutdown()
        assert result1["status"] == "shutdown"

        result2 = vision_shutdown()
        assert result2["status"] == "already_shutdown"

        reset_shutdown_state()  # cleanup

    def test_shutdown_reset(self):
        from missy.vision.shutdown import reset_shutdown_state, vision_shutdown

        reset_shutdown_state()
        vision_shutdown()

        reset_shutdown_state()
        result = vision_shutdown()
        assert result["status"] == "shutdown"

        reset_shutdown_state()  # cleanup

    def test_shutdown_returns_steps(self):
        """Shutdown should return steps in the summary."""
        from missy.vision.shutdown import reset_shutdown_state, vision_shutdown

        reset_shutdown_state()
        result = vision_shutdown()
        assert result["status"] == "shutdown"
        assert "steps" in result
        assert isinstance(result["steps"], list)

        reset_shutdown_state()


# ---------------------------------------------------------------------------
# SceneManager eviction
# ---------------------------------------------------------------------------


class TestSceneManagerEviction:
    """Test scene session eviction when at capacity."""

    def test_evicts_inactive_first(self):
        from missy.vision.scene_memory import SceneManager, TaskType

        mgr = SceneManager(max_sessions=2)
        s1 = mgr.create_session("task-1", TaskType.GENERAL)
        s1.close()  # now inactive
        mgr.create_session("task-2", TaskType.GENERAL)

        # Creating a third should evict the inactive one
        mgr.create_session("task-3", TaskType.GENERAL)
        assert mgr.get_session("task-1") is None
        assert mgr.get_session("task-2") is not None
        assert mgr.get_session("task-3") is not None

    def test_evicts_oldest_active_when_all_active(self):
        from missy.vision.scene_memory import SceneManager, TaskType

        mgr = SceneManager(max_sessions=2)
        mgr.create_session("task-1", TaskType.GENERAL)
        mgr.create_session("task-2", TaskType.GENERAL)

        # Both are active; creating a third should evict oldest
        mgr.create_session("task-3", TaskType.GENERAL)
        assert mgr.get_session("task-1") is None

    def test_get_active_session_returns_most_recent(self):
        from missy.vision.scene_memory import SceneManager, TaskType

        mgr = SceneManager(max_sessions=5)
        mgr.create_session("task-1", TaskType.GENERAL)
        mgr.create_session("task-2", TaskType.PUZZLE)
        mgr.create_session("task-3", TaskType.PAINTING)

        active = mgr.get_active_session()
        assert active is not None
        assert active.task_id == "task-3"

    def test_get_active_session_skips_closed(self):
        from missy.vision.scene_memory import SceneManager, TaskType

        mgr = SceneManager(max_sessions=5)
        mgr.create_session("task-1", TaskType.GENERAL)
        s2 = mgr.create_session("task-2", TaskType.GENERAL)
        s2.close()

        active = mgr.get_active_session()
        assert active is not None
        assert active.task_id == "task-1"

    def test_close_all_sessions(self):
        from missy.vision.scene_memory import SceneManager, TaskType

        mgr = SceneManager(max_sessions=5)
        mgr.create_session("task-1", TaskType.GENERAL)
        mgr.create_session("task-2", TaskType.GENERAL)
        mgr.close_all()

        assert mgr.get_active_session() is None


# ---------------------------------------------------------------------------
# Orientation edge cases
# ---------------------------------------------------------------------------


class TestOrientationEdgeCases:
    """Test orientation detection edge cases."""

    def test_square_image_normal(self):
        from missy.vision.orientation import Orientation, detect_orientation

        img = np.ones((100, 100, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL
        assert result.method == "aspect_ratio"

    def test_landscape_image_normal(self):
        from missy.vision.orientation import Orientation, detect_orientation

        img = np.ones((100, 200, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.NORMAL

    def test_portrait_image_rotated(self):
        from missy.vision.orientation import Orientation, detect_orientation

        img = np.ones((200, 100, 3), dtype=np.uint8)
        result = detect_orientation(img)
        assert result.detected == Orientation.ROTATED_90_CW

    def test_auto_correct_landscape(self):
        from missy.vision.orientation import auto_correct

        img = np.ones((100, 200, 3), dtype=np.uint8)
        result, info = auto_correct(img)
        assert result.shape == img.shape  # landscape stays landscape

    def test_correct_orientation_no_change_needed(self):
        from missy.vision.orientation import Orientation, correct_orientation

        img = np.ones((100, 200, 3), dtype=np.uint8)
        result = correct_orientation(img, Orientation.NORMAL)
        assert result.shape == img.shape

    def test_correct_orientation_90cw(self):
        from missy.vision.orientation import Orientation, correct_orientation

        img = np.ones((200, 100, 3), dtype=np.uint8)
        result = correct_orientation(img, Orientation.ROTATED_90_CW)
        # Correcting 90CW means rotating 90CCW → dimensions swap
        assert result.shape[0] == 100
        assert result.shape[1] == 200

    def test_none_image_returns_normal(self):
        from missy.vision.orientation import Orientation, detect_orientation

        result = detect_orientation(None)
        assert result.detected == Orientation.NORMAL
        assert result.confidence == 0.0


class TestPhotoSourceSequencing:
    """Test PhotoSource sequential photo iteration."""

    def test_photo_source_wraps_around(self):
        """PhotoSource should wrap around when reaching the end."""
        import cv2

        from missy.vision.sources import PhotoSource

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 2 small valid image files
            for name in ("a.jpg", "b.jpg"):
                img = np.ones((10, 10, 3), dtype=np.uint8) * 128
                cv2.imwrite(str(Path(tmpdir) / name), img)

            source = PhotoSource(tmpdir)
            source.scan()
            assert source.file_count == 2

            # Acquire all files + 1 more (wrap)
            frame1 = source.acquire()
            frame2 = source.acquire()
            frame3 = source.acquire()  # wraps to first

            assert frame1.metadata["photo_index"] == 0
            assert frame2.metadata["photo_index"] == 1
            assert frame3.metadata["photo_index"] == 0  # wrapped

    def test_photo_source_empty_directory(self):
        from missy.vision.sources import PhotoSource

        with tempfile.TemporaryDirectory() as tmpdir:
            source = PhotoSource(tmpdir)
            with pytest.raises(FileNotFoundError, match="No images found"):
                source.acquire()

    def test_photo_source_specific_index(self):
        import cv2

        from missy.vision.sources import PhotoSource

        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                img = np.ones((10, 10, 3), dtype=np.uint8) * 128
                cv2.imwrite(str(Path(tmpdir) / name), img)

            source = PhotoSource(tmpdir)
            frame = source.acquire_specific(1)
            assert frame.metadata["photo_index"] == 1

    def test_photo_source_invalid_index(self):
        import cv2

        from missy.vision.sources import PhotoSource

        with tempfile.TemporaryDirectory() as tmpdir:
            img = np.ones((10, 10, 3), dtype=np.uint8) * 128
            cv2.imwrite(str(Path(tmpdir) / "a.jpg"), img)

            source = PhotoSource(tmpdir)
            with pytest.raises(IndexError):
                source.acquire_specific(5)
