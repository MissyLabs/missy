"""Tests for input validation in FileSource and ScreenshotSource.

Covers the size/dimension guards added to FileSource.acquire() and the
tool-name propagation in ScreenshotSource.acquire()/_take_screenshot().
"""

from __future__ import annotations

import logging
import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.sources import FileSource, ScreenshotSource, SourceType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_FILE_SIZE = FileSource.MAX_FILE_SIZE  # 100 * 1024 * 1024
MAX_DIMENSION = FileSource.MAX_DIMENSION  # 16384


def _make_mock_cv2(img: np.ndarray | None = None) -> MagicMock:
    """Return a mock cv2 module whose imread() returns *img*."""
    mock_cv2 = MagicMock()
    mock_cv2.imread.return_value = img
    return mock_cv2


def _normal_image(width: int = 640, height: int = 480) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# FileSource — size validation
# ---------------------------------------------------------------------------


class TestFileSourceSizeValidation:
    def test_rejects_empty_file(self, tmp_path):
        """FileSource.acquire() raises ValueError for a zero-byte file."""
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")

        source = FileSource(str(empty))
        with pytest.raises(ValueError, match="empty"):
            source.acquire()

    @patch("missy.vision.sources._get_cv2")
    def test_rejects_oversized_file(self, mock_cv2_fn, tmp_path):
        """FileSource.acquire() raises ValueError when st_size exceeds MAX_FILE_SIZE."""
        mock_cv2_fn.return_value = _make_mock_cv2(_normal_image())

        large = tmp_path / "large.png"
        # Write a single byte then patch stat so size appears enormous
        large.write_bytes(b"x")

        with patch.object(large.stat().__class__, "st_size", new_callable=lambda: property(lambda self: MAX_FILE_SIZE + 1)):
            pass  # property approach is awkward; use os.stat mock instead

        # Simpler: patch the Path.stat() return value
        mock_stat = MagicMock()
        mock_stat.st_mode = 0o100644  # regular file
        mock_stat.st_size = MAX_FILE_SIZE + 1
        with patch("pathlib.Path.stat", return_value=mock_stat):
            source = FileSource(str(large))
            with pytest.raises(ValueError, match="too large"):
                source.acquire()

    @patch("missy.vision.sources._get_cv2")
    def test_accepts_file_at_exact_max_size_boundary(self, mock_cv2_fn, tmp_path):
        """FileSource.acquire() accepts a file whose size equals MAX_FILE_SIZE exactly."""
        img = _normal_image()
        mock_cv2_fn.return_value = _make_mock_cv2(img)

        boundary = tmp_path / "boundary.png"
        boundary.write_bytes(b"x")

        mock_stat = MagicMock()
        mock_stat.st_mode = 0o100644  # regular file
        mock_stat.st_size = MAX_FILE_SIZE  # exactly at the limit — allowed
        with patch("pathlib.Path.stat", return_value=mock_stat):
            source = FileSource(str(boundary))
            frame = source.acquire()

        assert frame.source_type == SourceType.FILE


# ---------------------------------------------------------------------------
# FileSource — dimension validation
# ---------------------------------------------------------------------------


class TestFileSourceDimensionValidation:
    @patch("missy.vision.sources._get_cv2")
    def test_rejects_zero_height_image(self, mock_cv2_fn, tmp_path):
        """FileSource.acquire() raises ValueError when imread returns a zero-height array."""
        # A numpy array with shape (0, 100, 3) simulates a degenerate decode result
        zero_height_img = np.zeros((0, 100, 3), dtype=np.uint8)
        mock_cv2_fn.return_value = _make_mock_cv2(zero_height_img)

        f = tmp_path / "degenerate.png"
        f.write_bytes(b"fake image data")

        source = FileSource(str(f))
        with pytest.raises(ValueError, match="zero dimension"):
            source.acquire()

    @patch("missy.vision.sources._get_cv2")
    def test_rejects_zero_width_image(self, mock_cv2_fn, tmp_path):
        """FileSource.acquire() raises ValueError when imread returns a zero-width array."""
        zero_width_img = np.zeros((100, 0, 3), dtype=np.uint8)
        mock_cv2_fn.return_value = _make_mock_cv2(zero_width_img)

        f = tmp_path / "degenerate.png"
        f.write_bytes(b"fake image data")

        source = FileSource(str(f))
        with pytest.raises(ValueError, match="zero dimension"):
            source.acquire()

    @patch("missy.vision.sources._get_cv2")
    def test_warns_on_oversized_dimensions(self, mock_cv2_fn, tmp_path, caplog):
        """FileSource.acquire() emits a warning but still succeeds for images wider than MAX_DIMENSION."""
        # Image larger than MAX_DIMENSION in width
        huge_img = np.zeros((100, MAX_DIMENSION + 1, 3), dtype=np.uint8)
        mock_cv2_fn.return_value = _make_mock_cv2(huge_img)

        f = tmp_path / "huge.png"
        f.write_bytes(b"fake image data")

        source = FileSource(str(f))
        with caplog.at_level(logging.WARNING, logger="missy.vision.sources"):
            frame = source.acquire()

        # Acquire should succeed — the warning is advisory only
        assert frame.source_type == SourceType.FILE
        # At least one warning record should mention large dimensions
        assert any("large" in record.message.lower() for record in caplog.records)

    @patch("missy.vision.sources._get_cv2")
    def test_warns_on_oversized_height(self, mock_cv2_fn, tmp_path, caplog):
        """FileSource.acquire() warns when image height exceeds MAX_DIMENSION."""
        huge_img = np.zeros((MAX_DIMENSION + 1, 100, 3), dtype=np.uint8)
        mock_cv2_fn.return_value = _make_mock_cv2(huge_img)

        f = tmp_path / "tall.png"
        f.write_bytes(b"fake image data")

        source = FileSource(str(f))
        with caplog.at_level(logging.WARNING, logger="missy.vision.sources"):
            frame = source.acquire()

        assert frame.source_type == SourceType.FILE
        assert any("large" in record.message.lower() for record in caplog.records)


# ---------------------------------------------------------------------------
# FileSource — happy path and format coverage
# ---------------------------------------------------------------------------


class TestFileSourceNormalOperation:
    @patch("missy.vision.sources._get_cv2")
    def test_accepts_normal_sized_file_and_image(self, mock_cv2_fn, tmp_path):
        """FileSource.acquire() returns a correctly populated ImageFrame for a valid file."""
        img = _normal_image(640, 480)
        mock_cv2_fn.return_value = _make_mock_cv2(img)

        f = tmp_path / "photo.jpg"
        f.write_bytes(b"fake jpeg content")

        source = FileSource(str(f))
        frame = source.acquire()

        assert frame.source_type == SourceType.FILE
        assert frame.width == 640
        assert frame.height == 480
        assert str(f) == frame.source_path

    def test_raises_file_not_found_for_nonexistent_path(self):
        """FileSource.acquire() raises FileNotFoundError for a path that does not exist."""
        source = FileSource("/tmp/missy_test_nonexistent_image_xyzzy.jpg")
        with pytest.raises(FileNotFoundError):
            source.acquire()

    @pytest.mark.parametrize("extension", [".jpg", ".png", ".bmp"])
    @patch("missy.vision.sources._get_cv2")
    def test_accepts_various_image_formats(self, mock_cv2_fn, tmp_path, extension):
        """FileSource.acquire() succeeds for common image file extensions."""
        img = _normal_image(320, 240)
        mock_cv2_fn.return_value = _make_mock_cv2(img)

        f = tmp_path / f"image{extension}"
        f.write_bytes(b"fake content")

        source = FileSource(str(f))
        frame = source.acquire()

        assert frame.source_type == SourceType.FILE
        assert frame.width == 320
        assert frame.height == 240


# ---------------------------------------------------------------------------
# ScreenshotSource — tool name propagation
# ---------------------------------------------------------------------------


class TestScreenshotSourceToolName:
    def test_take_screenshot_returns_tool_name_on_success(self, tmp_path):
        """_take_screenshot() returns the name of the tool that succeeded."""
        output_file = tmp_path / "shot.png"

        # Patch subprocess.run so that the first tool (scrot) reports success
        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            # Create the output file to simulate a real tool writing it
            output_file.write_bytes(b"fake png")
            return result

        source = ScreenshotSource()
        with patch("subprocess.run", side_effect=fake_run):
            tool_name = source._take_screenshot(str(output_file))

        assert tool_name == "scrot"

    def test_take_screenshot_returns_second_tool_name_when_first_fails(self, tmp_path):
        """_take_screenshot() returns the name of the first tool that succeeds."""
        output_file = tmp_path / "shot.png"
        call_count = {"n": 0}

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First tool (scrot) fails
                result.returncode = 1
            else:
                # Second tool (gnome-screenshot) succeeds
                result.returncode = 0
                output_file.write_bytes(b"fake png")
            return result

        source = ScreenshotSource()
        with patch("subprocess.run", side_effect=fake_run):
            tool_name = source._take_screenshot(str(output_file))

        assert tool_name == "gnome-screenshot"

    @patch("missy.vision.sources._get_cv2")
    def test_acquire_error_message_includes_tool_name_when_image_unreadable(
        self, mock_cv2_fn, tmp_path
    ):
        """acquire() error message names the screenshot tool when imread fails to decode."""
        # imread returns None — the file exists but is unreadable
        mock_cv2_fn.return_value = _make_mock_cv2(None)

        output_file = tmp_path / "shot.png"

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            output_file.write_bytes(b"corrupt data")
            return result

        source = ScreenshotSource()

        # Patch tempfile.NamedTemporaryFile so we control the tmp path
        import tempfile

        real_ntf = tempfile.NamedTemporaryFile

        def fake_ntf(**kwargs):
            obj = real_ntf(suffix=".png", delete=False, dir=str(tmp_path))
            # Make the name predictable by overwriting
            return obj

        with patch("subprocess.run", side_effect=fake_run), pytest.raises(RuntimeError) as exc_info:
            source.acquire()

        # The error message must name the tool that was used
        assert "scrot" in str(exc_info.value)

    def test_take_screenshot_raises_when_all_tools_absent(self, tmp_path):
        """_take_screenshot() raises RuntimeError when every tool is missing."""
        output_file = tmp_path / "shot.png"

        source = ScreenshotSource()
        with patch("subprocess.run", side_effect=FileNotFoundError("not found")), pytest.raises(RuntimeError, match="No screenshot tool succeeded"):
            source._take_screenshot(str(output_file))

    def test_take_screenshot_raises_when_all_tools_timeout(self, tmp_path):
        """_take_screenshot() raises RuntimeError when every tool times out."""
        output_file = tmp_path / "shot.png"

        source = ScreenshotSource()
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="scrot", timeout=10),
        ), pytest.raises(RuntimeError, match="No screenshot tool succeeded"):
            source._take_screenshot(str(output_file))
