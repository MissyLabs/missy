"""Tests for missy.vision.sources — unified image source abstraction."""

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


# ---------------------------------------------------------------------------
# ImageFrame tests
# ---------------------------------------------------------------------------


class TestImageFrame:
    def test_auto_dimensions(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = ImageFrame(image=img, source_type=SourceType.FILE)
        assert frame.width == 640
        assert frame.height == 480

    @patch("missy.vision.sources._get_cv2")
    def test_to_jpeg_bytes(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.IMWRITE_JPEG_QUALITY = 1
        mock_cv2.imencode.return_value = (True, np.array([0xFF, 0xD8], dtype=np.uint8))

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = ImageFrame(image=img, source_type=SourceType.FILE)
        data = frame.to_jpeg_bytes()

        assert isinstance(data, bytes)
        assert len(data) > 0

    @patch("missy.vision.sources._get_cv2")
    def test_to_base64(self, mock_cv2_fn):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.IMWRITE_JPEG_QUALITY = 1
        mock_cv2.imencode.return_value = (True, np.array([0x41, 0x42], dtype=np.uint8))

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = ImageFrame(image=img, source_type=SourceType.FILE)
        b64 = frame.to_base64()

        assert isinstance(b64, str)
        assert len(b64) > 0


# ---------------------------------------------------------------------------
# FileSource tests
# ---------------------------------------------------------------------------


class TestFileSource:
    def test_source_type(self):
        source = FileSource("/tmp/test.jpg")
        assert source.source_type() == SourceType.FILE

    def test_is_available_missing_file(self):
        source = FileSource("/nonexistent/path.jpg")
        assert source.is_available() is False

    def test_is_available_existing_file(self, tmp_path):
        f = tmp_path / "test.jpg"
        f.write_bytes(b"fake")
        source = FileSource(str(f))
        assert source.is_available() is True

    @patch("missy.vision.sources._get_cv2")
    def test_acquire_success(self, mock_cv2_fn, tmp_path):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2

        f = tmp_path / "test.jpg"
        f.write_bytes(b"fake")

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = img

        source = FileSource(str(f))
        frame = source.acquire()

        assert frame.source_type == SourceType.FILE
        assert frame.width == 100
        assert frame.height == 100

    def test_acquire_missing_file(self):
        source = FileSource("/nonexistent/image.jpg")
        with pytest.raises(FileNotFoundError):
            source.acquire()

    @patch("missy.vision.sources._get_cv2")
    def test_acquire_unreadable(self, mock_cv2_fn, tmp_path):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2
        mock_cv2.imread.return_value = None

        f = tmp_path / "bad.jpg"
        f.write_bytes(b"not an image")

        source = FileSource(str(f))
        with pytest.raises(ValueError, match="Failed to decode image"):
            source.acquire()


# ---------------------------------------------------------------------------
# PhotoSource tests
# ---------------------------------------------------------------------------


class TestPhotoSource:
    def test_source_type(self, tmp_path):
        source = PhotoSource(str(tmp_path))
        assert source.source_type() == SourceType.PHOTO

    def test_is_available(self, tmp_path):
        source = PhotoSource(str(tmp_path))
        assert source.is_available() is True

    def test_not_available(self):
        source = PhotoSource("/nonexistent/dir")
        assert source.is_available() is False

    def test_scan_finds_images(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"fake")
        (tmp_path / "b.png").write_bytes(b"fake")
        (tmp_path / "c.txt").write_bytes(b"not image")

        source = PhotoSource(str(tmp_path))
        files = source.scan()
        assert len(files) == 2
        assert all(f.suffix in (".jpg", ".png") for f in files)

    def test_file_count(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"fake")
        (tmp_path / "b.jpg").write_bytes(b"fake")

        source = PhotoSource(str(tmp_path))
        assert source.file_count == 2

    def test_scan_empty_dir(self, tmp_path):
        source = PhotoSource(str(tmp_path))
        files = source.scan()
        assert files == []

    @patch("missy.vision.sources._get_cv2")
    def test_acquire_wraps_around(self, mock_cv2_fn, tmp_path):
        mock_cv2 = MagicMock()
        mock_cv2_fn.return_value = mock_cv2

        (tmp_path / "a.jpg").write_bytes(b"fake")
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = img

        source = PhotoSource(str(tmp_path))
        f1 = source.acquire()
        assert f1.metadata["photo_index"] == 0

        # Wrap around
        f2 = source.acquire()
        assert f2.metadata["photo_index"] == 0  # only 1 file, wraps

    def test_acquire_empty_dir_raises(self, tmp_path):
        source = PhotoSource(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="No images"):
            source.acquire()


# ---------------------------------------------------------------------------
# WebcamSource tests
# ---------------------------------------------------------------------------


class TestWebcamSource:
    def test_source_type(self):
        source = WebcamSource("/dev/video0")
        assert source.source_type() == SourceType.WEBCAM

    def test_is_available_no_device(self):
        source = WebcamSource("/dev/video999")
        assert source.is_available() is False


# ---------------------------------------------------------------------------
# ScreenshotSource tests
# ---------------------------------------------------------------------------


class TestScreenshotSource:
    def test_source_type(self):
        source = ScreenshotSource()
        assert source.source_type() == SourceType.SCREENSHOT


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestCreateSource:
    def test_create_webcam(self):
        source = create_source(SourceType.WEBCAM, device_path="/dev/video0")
        assert isinstance(source, WebcamSource)

    def test_create_file(self):
        source = create_source("file", file_path="/tmp/test.jpg")
        assert isinstance(source, FileSource)

    def test_create_screenshot(self):
        source = create_source(SourceType.SCREENSHOT)
        assert isinstance(source, ScreenshotSource)

    def test_create_photo(self, tmp_path):
        source = create_source("photo", directory=str(tmp_path))
        assert isinstance(source, PhotoSource)

    def test_create_file_no_path_raises(self):
        with pytest.raises(ValueError, match="file_path required"):
            create_source("file")

    def test_create_photo_no_dir_raises(self):
        with pytest.raises(ValueError, match="directory required"):
            create_source("photo")

    def test_create_unknown_type_raises(self):
        with pytest.raises(ValueError):
            create_source("unknown_type")
