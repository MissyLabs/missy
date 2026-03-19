"""Source abstraction and provider format edge case tests.


Covers:
- FileSource: symlink rejection, empty file, oversized file, zero-dimension image
- PhotoSource: empty directory, wrap-around, acquire_specific bounds
- ScreenshotSource: tool fallback, availability check
- WebcamSource: device path validation
- Provider format: all providers, edge cases, Ollama format
- Source factory: all source types, invalid type
"""

from __future__ import annotations

import stat
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# FileSource edge cases
# ---------------------------------------------------------------------------


class TestFileSourceEdgeCases:
    """Edge cases for FileSource validation and security."""

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        """Empty file should raise ValueError."""
        from missy.vision.sources import FileSource

        empty = tmp_path / "empty.jpg"
        empty.touch()
        source = FileSource(empty)
        with pytest.raises(ValueError, match="empty"):
            source.acquire()

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Missing file should raise FileNotFoundError."""
        from missy.vision.sources import FileSource

        source = FileSource(tmp_path / "missing.jpg")
        with pytest.raises(FileNotFoundError):
            source.acquire()

    def test_oversized_file_raises(self, tmp_path: Path) -> None:
        """File exceeding MAX_FILE_SIZE should raise ValueError."""
        from missy.vision.sources import FileSource

        large = tmp_path / "large.jpg"
        large.touch()

        source = FileSource(large)
        # Mock stat to pretend file is huge

        class FakeStat:
            st_mode = stat.S_IFREG | 0o644
            st_size = 200 * 1024 * 1024  # 200 MB

        with (
            patch.object(Path, "stat", return_value=FakeStat()),
            pytest.raises(ValueError, match="too large"),
        ):
            source.acquire()

    def test_source_type_is_file(self) -> None:
        """FileSource.source_type() should return FILE."""
        from missy.vision.sources import FileSource, SourceType

        source = FileSource("/some/path.jpg")
        assert source.source_type() == SourceType.FILE

    def test_is_available_for_existing_file(self, tmp_path: Path) -> None:
        """is_available() should return True for existing regular file."""
        from missy.vision.sources import FileSource

        f = tmp_path / "test.jpg"
        f.write_bytes(b"fake")
        source = FileSource(f)
        assert source.is_available() is True

    def test_is_available_for_missing_file(self) -> None:
        """is_available() should return False for missing file."""
        from missy.vision.sources import FileSource

        source = FileSource("/nonexistent/file.jpg")
        assert source.is_available() is False

    def test_valid_image_read(self, tmp_path: Path) -> None:
        """Reading a valid image file produces an ImageFrame."""
        import cv2

        from missy.vision.sources import FileSource, SourceType

        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)

        source = FileSource(path)
        frame = source.acquire()
        assert frame.source_type == SourceType.FILE
        assert frame.width == 50
        assert frame.height == 50


# ---------------------------------------------------------------------------
# PhotoSource edge cases
# ---------------------------------------------------------------------------


class TestPhotoSourceEdgeCases:
    """PhotoSource directory scanning and navigation."""

    def test_empty_directory_raises_on_acquire(self, tmp_path: Path) -> None:
        """Acquiring from empty directory should raise FileNotFoundError."""
        from missy.vision.sources import PhotoSource

        source = PhotoSource(tmp_path)
        with pytest.raises(FileNotFoundError, match="No images"):
            source.acquire()

    def test_scan_filters_non_image_files(self, tmp_path: Path) -> None:
        """scan() should skip non-image extensions."""
        from missy.vision.sources import PhotoSource

        (tmp_path / "data.csv").write_text("x,y")
        (tmp_path / "readme.md").write_text("hello")
        (tmp_path / "script.py").write_text("print(1)")

        source = PhotoSource(tmp_path)
        files = source.scan()
        assert files == []

    def test_scan_finds_image_files(self, tmp_path: Path) -> None:
        """scan() should find supported image extensions."""
        import cv2

        from missy.vision.sources import PhotoSource

        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "a.jpg"), img)
        cv2.imwrite(str(tmp_path / "b.png"), img)

        source = PhotoSource(tmp_path)
        files = source.scan()
        assert len(files) == 2

    def test_wrap_around_on_acquire(self, tmp_path: Path) -> None:
        """acquire() should wrap around after reaching the last image."""
        import cv2

        from missy.vision.sources import PhotoSource

        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "a.jpg"), img)
        cv2.imwrite(str(tmp_path / "b.jpg"), img)

        source = PhotoSource(tmp_path)
        source.scan()

        # Acquire all photos
        source.acquire()
        source.acquire()
        # Wrap around
        f3 = source.acquire()
        assert f3.metadata["photo_index"] == 0

    def test_acquire_specific_valid_index(self, tmp_path: Path) -> None:
        """acquire_specific() with valid index returns correct photo."""
        import cv2

        from missy.vision.sources import PhotoSource

        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "a.jpg"), img)
        cv2.imwrite(str(tmp_path / "b.jpg"), img)

        source = PhotoSource(tmp_path)
        frame = source.acquire_specific(1)
        assert frame.metadata["photo_index"] == 1

    def test_acquire_specific_invalid_index(self, tmp_path: Path) -> None:
        """acquire_specific() with invalid index raises IndexError."""
        import cv2

        from missy.vision.sources import PhotoSource

        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "a.jpg"), img)

        source = PhotoSource(tmp_path)
        with pytest.raises(IndexError):
            source.acquire_specific(5)

    def test_file_count_triggers_scan(self, tmp_path: Path) -> None:
        """file_count should trigger a scan if not already done."""
        from missy.vision.sources import PhotoSource

        source = PhotoSource(tmp_path)
        assert source.file_count == 0  # triggers scan, finds nothing

    def test_source_type_is_photo(self, tmp_path: Path) -> None:
        from missy.vision.sources import PhotoSource, SourceType

        source = PhotoSource(tmp_path)
        assert source.source_type() == SourceType.PHOTO

    def test_nonexistent_directory_raises(self) -> None:
        from missy.vision.sources import PhotoSource

        source = PhotoSource("/nonexistent/dir")
        with pytest.raises(FileNotFoundError):
            source.scan()


# ---------------------------------------------------------------------------
# WebcamSource validation
# ---------------------------------------------------------------------------


class TestWebcamSourceValidation:
    """WebcamSource device path validation."""

    def test_valid_device_path_accepted(self) -> None:
        from missy.vision.sources import WebcamSource

        source = WebcamSource("/dev/video0")
        assert source._device_path == "/dev/video0"

    def test_invalid_device_path_rejected(self) -> None:
        from missy.vision.sources import WebcamSource

        with pytest.raises(ValueError, match="Invalid device path"):
            WebcamSource("/dev/sda1")

    def test_traversal_path_rejected(self) -> None:
        from missy.vision.sources import WebcamSource

        with pytest.raises(ValueError, match="Invalid device path"):
            WebcamSource("/dev/video0/../sda")

    def test_command_injection_rejected(self) -> None:
        from missy.vision.sources import WebcamSource

        with pytest.raises(ValueError, match="Invalid device path"):
            WebcamSource("/dev/video0; rm -rf /")

    def test_source_type_is_webcam(self) -> None:
        from missy.vision.sources import SourceType, WebcamSource

        source = WebcamSource("/dev/video0")
        assert source.source_type() == SourceType.WEBCAM

    def test_is_available_false_when_no_device(self) -> None:
        from missy.vision.sources import WebcamSource

        source = WebcamSource("/dev/video99")
        assert source.is_available() is False


# ---------------------------------------------------------------------------
# ScreenshotSource
# ---------------------------------------------------------------------------


class TestScreenshotSourceEdgeCases:
    """ScreenshotSource availability and fallback."""

    def test_source_type_is_screenshot(self) -> None:
        from missy.vision.sources import ScreenshotSource, SourceType

        source = ScreenshotSource()
        assert source.source_type() == SourceType.SCREENSHOT

    def test_is_available_no_tools(self) -> None:
        """is_available() returns False when no screenshot tools exist."""
        from missy.vision.sources import ScreenshotSource

        source = ScreenshotSource()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert source.is_available() is False

    def test_acquire_all_tools_fail(self) -> None:
        """acquire() raises RuntimeError when all tools fail."""
        from missy.vision.sources import ScreenshotSource

        source = ScreenshotSource()
        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            pytest.raises(RuntimeError, match="No screenshot tool"),
        ):
            source.acquire()


# ---------------------------------------------------------------------------
# Source factory
# ---------------------------------------------------------------------------


class TestSourceFactory:
    """create_source() factory function."""

    def test_create_webcam_source(self) -> None:
        from missy.vision.sources import SourceType, WebcamSource, create_source

        source = create_source(SourceType.WEBCAM, device_path="/dev/video0")
        assert isinstance(source, WebcamSource)

    def test_create_file_source(self) -> None:
        from missy.vision.sources import FileSource, SourceType, create_source

        source = create_source(SourceType.FILE, file_path="/tmp/test.jpg")
        assert isinstance(source, FileSource)

    def test_create_screenshot_source(self) -> None:
        from missy.vision.sources import ScreenshotSource, SourceType, create_source

        source = create_source(SourceType.SCREENSHOT)
        assert isinstance(source, ScreenshotSource)

    def test_create_photo_source(self) -> None:
        from missy.vision.sources import PhotoSource, SourceType, create_source

        source = create_source(SourceType.PHOTO, directory="/tmp")
        assert isinstance(source, PhotoSource)

    def test_create_file_without_path_raises(self) -> None:
        from missy.vision.sources import SourceType, create_source

        with pytest.raises(ValueError, match="file_path"):
            create_source(SourceType.FILE)

    def test_create_photo_without_directory_raises(self) -> None:
        from missy.vision.sources import SourceType, create_source

        with pytest.raises(ValueError, match="directory"):
            create_source(SourceType.PHOTO)

    def test_create_from_string_type(self) -> None:
        from missy.vision.sources import WebcamSource, create_source

        source = create_source("webcam", device_path="/dev/video0")
        assert isinstance(source, WebcamSource)

    def test_invalid_source_type_raises(self) -> None:
        from missy.vision.sources import create_source

        with pytest.raises(ValueError):
            create_source("invalid_type")


# ---------------------------------------------------------------------------
# Provider format extended
# ---------------------------------------------------------------------------


class TestProviderFormatExtended:
    """Extended provider format tests."""

    _DUMMY_B64 = "iVBORw0KGgo="

    def test_ollama_uses_openai_format(self) -> None:
        """Ollama should produce the same format as OpenAI."""
        from missy.vision.provider_format import format_image_for_provider

        ollama = format_image_for_provider("ollama", self._DUMMY_B64)
        openai = format_image_for_provider("openai", self._DUMMY_B64)
        assert ollama["type"] == openai["type"]
        assert ollama["type"] == "image_url"

    def test_gpt_alias_works(self) -> None:
        """'gpt' should be treated as OpenAI."""
        from missy.vision.provider_format import format_image_for_provider

        result = format_image_for_provider("gpt", self._DUMMY_B64)
        assert result["type"] == "image_url"

    def test_empty_provider_raises(self) -> None:
        from missy.vision.provider_format import format_image_for_provider

        with pytest.raises(ValueError, match="provider_name"):
            format_image_for_provider("", self._DUMMY_B64)

    def test_whitespace_provider_raises(self) -> None:
        from missy.vision.provider_format import format_image_for_provider

        with pytest.raises(ValueError, match="provider_name"):
            format_image_for_provider("  ", self._DUMMY_B64)

    def test_empty_image_raises(self) -> None:
        from missy.vision.provider_format import format_image_for_provider

        with pytest.raises(ValueError, match="image_base64"):
            format_image_for_provider("anthropic", "")

    def test_empty_media_type_raises(self) -> None:
        from missy.vision.provider_format import format_image_for_provider

        with pytest.raises(ValueError, match="media_type"):
            format_image_for_provider("anthropic", self._DUMMY_B64, "")

    def test_case_insensitive_provider(self) -> None:
        from missy.vision.provider_format import format_image_for_provider

        result = format_image_for_provider("ANTHROPIC", self._DUMMY_B64)
        assert result["type"] == "image"

    def test_anthropic_media_type_in_source(self) -> None:
        from missy.vision.provider_format import format_image_for_anthropic

        result = format_image_for_anthropic(self._DUMMY_B64, "image/png")
        assert result["source"]["media_type"] == "image/png"

    def test_openai_detail_parameter(self) -> None:
        from missy.vision.provider_format import format_image_for_openai

        result = format_image_for_openai(self._DUMMY_B64, detail="high")
        assert result["image_url"]["detail"] == "high"

    def test_build_vision_message_structure(self) -> None:
        from missy.vision.provider_format import build_vision_message

        msg = build_vision_message("anthropic", self._DUMMY_B64, "Describe this")
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "image"
        assert msg["content"][1]["type"] == "text"
        assert msg["content"][1]["text"] == "Describe this"

    def test_build_vision_message_whitespace_prompt_raises(self) -> None:
        """Empty/whitespace-only prompt should raise."""
        from missy.vision.provider_format import build_vision_message

        with pytest.raises(ValueError, match="prompt"):
            build_vision_message("anthropic", self._DUMMY_B64, "")


# ---------------------------------------------------------------------------
# ImageFrame encoding
# ---------------------------------------------------------------------------


class TestImageFrameEncoding:
    """ImageFrame to_jpeg_bytes, to_base64, to_png_bytes."""

    def test_to_jpeg_bytes(self) -> None:
        from missy.vision.sources import ImageFrame, SourceType

        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        frame = ImageFrame(image=img, source_type=SourceType.FILE)
        data = frame.to_jpeg_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0
        # JPEG magic bytes
        assert data[:2] == b"\xff\xd8"

    def test_to_png_bytes(self) -> None:
        from missy.vision.sources import ImageFrame, SourceType

        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        frame = ImageFrame(image=img, source_type=SourceType.FILE)
        data = frame.to_png_bytes()
        assert isinstance(data, bytes)
        # PNG magic bytes
        assert data[:4] == b"\x89PNG"

    def test_to_base64(self) -> None:
        import base64

        from missy.vision.sources import ImageFrame, SourceType

        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        frame = ImageFrame(image=img, source_type=SourceType.FILE)
        b64 = frame.to_base64()
        assert isinstance(b64, str)
        # Should be valid base64
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_auto_dimensions(self) -> None:
        from missy.vision.sources import ImageFrame, SourceType

        img = np.full((120, 160, 3), 128, dtype=np.uint8)
        frame = ImageFrame(image=img, source_type=SourceType.WEBCAM)
        assert frame.width == 160
        assert frame.height == 120
