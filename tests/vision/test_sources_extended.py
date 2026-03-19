"""Extended tests for missy.vision.sources — pattern filtering and edge cases."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.sources import (
    FileSource,
    ImageFrame,
    PhotoSource,
    ScreenshotSource,
    SourceType,
    WebcamSource,
    create_source,
)


class TestPhotoSourcePattern:
    """Tests for the pattern parameter in PhotoSource.scan()."""

    def test_default_pattern_matches_all(self, tmp_path: Path):
        """Default '*' pattern should match all image files."""
        (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
        (tmp_path / "b.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
        (tmp_path / "c.txt").write_text("not an image")

        source = PhotoSource(tmp_path)
        files = source.scan()
        assert len(files) == 2  # a.jpg, b.png — c.txt excluded by extension

    def test_pattern_filters_by_glob(self, tmp_path: Path):
        """Pattern '*.jpg' should only match JPEG files."""
        (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
        (tmp_path / "b.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
        (tmp_path / "c.jpeg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

        source = PhotoSource(tmp_path, pattern="*.jpg")
        files = source.scan()
        assert len(files) == 1
        assert files[0].name == "a.jpg"

    def test_pattern_recursive_glob(self, tmp_path: Path):
        """Pattern '**/*.png' should find PNG files in subdirectories."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
        (tmp_path / "top.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
        (tmp_path / "ignore.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 100)

        source = PhotoSource(tmp_path, pattern="**/*.png")
        files = source.scan()
        names = {f.name for f in files}
        assert "nested.png" in names
        assert "top.png" in names
        assert "ignore.jpg" not in names

    def test_empty_directory(self, tmp_path: Path):
        source = PhotoSource(tmp_path)
        files = source.scan()
        assert files == []

    def test_file_count_property(self, tmp_path: Path):
        (tmp_path / "img.jpg").write_bytes(b"\xff" * 100)
        source = PhotoSource(tmp_path)
        assert source.file_count == 1

    def test_acquire_wraps_around(self, tmp_path: Path):
        """acquire() should wrap around after last photo."""
        (tmp_path / "a.jpg").write_bytes(b"\xff" * 100)

        with patch("missy.vision.sources.FileSource") as MockFS:
            mock_frame = MagicMock()
            mock_frame.source_type = SourceType.FILE
            mock_frame.metadata = {}
            MockFS.return_value.acquire.return_value = mock_frame

            source = PhotoSource(tmp_path)
            source.scan()

            # First call: index goes 0→1
            source.acquire()
            assert source._index == 1
            # Second call: wraps 1→0→1
            source.acquire()
            assert source._index == 1  # wrapped back and incremented


class TestSourceFactory:
    def test_create_webcam(self):
        src = create_source("webcam", device_path="/dev/video0")
        assert isinstance(src, WebcamSource)

    def test_create_file(self, tmp_path: Path):
        src = create_source("file", file_path=str(tmp_path / "test.jpg"))
        assert isinstance(src, FileSource)

    def test_create_screenshot(self):
        src = create_source("screenshot")
        assert isinstance(src, ScreenshotSource)

    def test_create_photo(self, tmp_path: Path):
        src = create_source("photo", directory=str(tmp_path))
        assert isinstance(src, PhotoSource)

    def test_create_file_requires_path(self):
        with pytest.raises(ValueError, match="file_path required"):
            create_source("file")

    def test_create_photo_requires_directory(self):
        with pytest.raises(ValueError, match="directory required"):
            create_source("photo")

    def test_create_invalid_type(self):
        with pytest.raises(ValueError):
            create_source("invalid_source_type")


class TestFileSource:
    def test_is_available_nonexistent(self, tmp_path: Path):
        src = FileSource(tmp_path / "nonexistent.jpg")
        assert not src.is_available()

    def test_is_available_directory(self, tmp_path: Path):
        src = FileSource(tmp_path)
        assert not src.is_available()

    def test_acquire_nonexistent(self, tmp_path: Path):
        src = FileSource(tmp_path / "nonexistent.jpg")
        with pytest.raises(FileNotFoundError):
            src.acquire()


class TestWebcamSource:
    def test_is_available_no_device(self):
        src = WebcamSource("/dev/video999")
        assert not src.is_available()

    def test_source_type(self):
        src = WebcamSource()
        assert src.source_type() == SourceType.WEBCAM


class TestScreenshotSource:
    def test_source_type(self):
        src = ScreenshotSource()
        assert src.source_type() == SourceType.SCREENSHOT

    @patch("subprocess.run")
    def test_is_available_with_scrot(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        src = ScreenshotSource()
        assert src.is_available()

    @patch("subprocess.run")
    def test_is_available_no_tools(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        src = ScreenshotSource()
        assert not src.is_available()


class TestImageFrame:
    def test_auto_dimensions(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = ImageFrame(image=img, source_type=SourceType.FILE)
        assert frame.width == 640
        assert frame.height == 480

    @patch("missy.vision.sources._get_cv2")
    def test_to_jpeg_bytes(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2.imencode.return_value = (True, MagicMock(tobytes=MagicMock(return_value=b"jpeg")))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1
        mock_cv2_fn.return_value = mock_cv2

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        frame = ImageFrame(image=img, source_type=SourceType.FILE)
        result = frame.to_jpeg_bytes(quality=90)
        assert result == b"jpeg"

    @patch("missy.vision.sources._get_cv2")
    def test_to_jpeg_bytes_failure(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2.imencode.return_value = (False, None)
        mock_cv2.IMWRITE_JPEG_QUALITY = 1
        mock_cv2_fn.return_value = mock_cv2

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        frame = ImageFrame(image=img, source_type=SourceType.FILE)
        with pytest.raises(RuntimeError, match="Failed to encode"):
            frame.to_jpeg_bytes()

    @patch("missy.vision.sources._get_cv2")
    def test_to_png_bytes(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2.imencode.return_value = (True, MagicMock(tobytes=MagicMock(return_value=b"png")))
        mock_cv2_fn.return_value = mock_cv2

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        frame = ImageFrame(image=img, source_type=SourceType.FILE)
        result = frame.to_png_bytes()
        assert result == b"png"

    @patch("missy.vision.sources._get_cv2")
    def test_to_base64(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2.imencode.return_value = (True, MagicMock(tobytes=MagicMock(return_value=b"jpeg")))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1
        mock_cv2_fn.return_value = mock_cv2

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        frame = ImageFrame(image=img, source_type=SourceType.FILE)
        result = frame.to_base64()
        assert isinstance(result, str)
        # Should be valid base64
        import base64
        decoded = base64.b64decode(result)
        assert decoded == b"jpeg"
