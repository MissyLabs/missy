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

        with patch.object(
            large.stat().__class__,
            "st_size",
            new_callable=lambda: property(lambda self: MAX_FILE_SIZE + 1),
        ):
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
    def test_rejects_oversized_width_post_decode_safety_net(self, mock_cv2_fn, tmp_path):
        """Availability hardening: FileSource.acquire() must reject (not just
        warn about) images wider than MAX_DIMENSION. This exercises the
        post-decode safety net specifically -- the file's bytes aren't a
        real image PIL's header peek can parse (see
        TestFileSourceDecompressionBombGuard below for the primary,
        pre-decode defense), so this path is what catches it here."""
        huge_img = np.zeros((100, MAX_DIMENSION + 1, 3), dtype=np.uint8)
        mock_cv2_fn.return_value = _make_mock_cv2(huge_img)

        f = tmp_path / "huge.png"
        f.write_bytes(b"fake image data")

        source = FileSource(str(f))
        with pytest.raises(ValueError, match="exceed the maximum allowed"):
            source.acquire()

    @patch("missy.vision.sources._get_cv2")
    def test_rejects_oversized_height_post_decode_safety_net(self, mock_cv2_fn, tmp_path):
        """Availability hardening: same as above, for height."""
        huge_img = np.zeros((MAX_DIMENSION + 1, 100, 3), dtype=np.uint8)
        mock_cv2_fn.return_value = _make_mock_cv2(huge_img)

        f = tmp_path / "tall.png"
        f.write_bytes(b"fake image data")

        source = FileSource(str(f))
        with pytest.raises(ValueError, match="exceed the maximum allowed"):
            source.acquire()


# ---------------------------------------------------------------------------
# FileSource — decompression-bomb guard (pre-decode, availability hardening)
# ---------------------------------------------------------------------------


class TestFileSourceDecompressionBombGuard:
    """Live-reproduced before this fix: an 11 MB PNG declaring 30000x30000
    pixels was fully decoded by cv2.imread() (allocating a ~2.5 GB numpy
    array, taking ~2.8s) before MAX_DIMENSION was ever consulted -- which
    at the time only logged a warning and let the oversized image through
    regardless. The fix peeks the image header via PIL (cheap, no pixel
    decode) and rejects before OpenCV is ever invoked."""

    def test_rejects_declared_oversized_dimensions_before_cv2_runs(self, tmp_path):
        from PIL import Image

        # Wide-but-thin so the total pixel count stays under Pillow's own
        # MAX_IMAGE_PIXELS decompression-bomb threshold -- this exercises
        # *this* fix's MAX_DIMENSION check specifically, not Pillow's own
        # separate guard (see the test below for that one).
        img = Image.new("RGB", (MAX_DIMENSION + 1, 100), color=(0, 0, 0))
        f = tmp_path / "wide.png"
        img.save(f, compress_level=1)

        with (
            patch("missy.vision.sources._get_cv2") as mock_cv2_fn,
            pytest.raises(ValueError, match="refusing to decode"),
        ):
            FileSource(str(f)).acquire()
        # The whole point: cv2.imread() must never be reached for a
        # rejected image (cv2 itself may still be lazily resolved, but
        # the actual decode call is what's expensive and must be skipped).
        mock_cv2_fn.return_value.imread.assert_not_called()

    def test_pillow_decompression_bomb_guard_is_propagated_as_rejection(self, tmp_path):
        """A file whose declared pixel count trips Pillow's own built-in
        MAX_IMAGE_PIXELS guard must be rejected, not silently treated as
        'dimensions unknown, fall through to the slow path' -- that would
        defeat the entire point of checking the header first."""
        from PIL import Image

        img = Image.new("RGB", (20000, 20000), color=(0, 0, 0))
        f = tmp_path / "bomb.png"
        img.save(f, compress_level=1)

        with (
            patch("missy.vision.sources._get_cv2") as mock_cv2_fn,
            pytest.raises(ValueError, match="decompression-bomb"),
        ):
            FileSource(str(f)).acquire()
        mock_cv2_fn.return_value.imread.assert_not_called()

    def test_normal_image_still_decodes_successfully(self, tmp_path):
        from PIL import Image

        img = Image.new("RGB", (800, 600), color=(255, 0, 0))
        f = tmp_path / "normal.png"
        img.save(f)

        frame = FileSource(str(f)).acquire()
        assert frame.source_type == SourceType.FILE
        assert frame.width == 800
        assert frame.height == 600

    def test_corrupt_file_falls_through_to_post_decode_check(self, tmp_path):
        """A file PIL can't parse at all (corrupt / not actually an image)
        must not crash the pre-check -- it should fall through and let
        the existing decode-failure handling take over."""
        f = tmp_path / "corrupt.png"
        f.write_bytes(b"not a real image")

        with pytest.raises(ValueError, match="Failed to decode"):
            FileSource(str(f)).acquire()


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
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("not found")),
            pytest.raises(RuntimeError, match="No screenshot tool succeeded"),
        ):
            source._take_screenshot(str(output_file))

    def test_take_screenshot_raises_when_all_tools_timeout(self, tmp_path):
        """_take_screenshot() raises RuntimeError when every tool times out."""
        output_file = tmp_path / "shot.png"

        source = ScreenshotSource()
        with (
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="scrot", timeout=10),
            ),
            pytest.raises(RuntimeError, match="No screenshot tool succeeded"),
        ):
            source._take_screenshot(str(output_file))
