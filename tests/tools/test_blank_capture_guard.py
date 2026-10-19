"""Blank/uniform screen captures must never reach the vision model.

Live validation: the display returned all-black 640x480 frames (identical
987-byte PNGs) and x11_read_screen sent them to Ollama, which hallucinated
"Microsoft PowerPoint", a fake "Run Test Success" and a nonexistent Submit
button.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

from missy.tools.builtin.x11_tools import X11ReadScreenTool, X11ScreenshotTool  # noqa: E402
from missy.vision.blank_frame import uniform_image_array, uniform_image_file  # noqa: E402


def _black_png(path: str) -> None:
    Image.new("RGB", (640, 480), (0, 0, 0)).save(path)


def _real_png(path: str) -> None:
    img = Image.new("RGB", (640, 480), (0, 0, 0))
    ImageDraw.Draw(img).text((10, 10), "user@host:~$ ls", fill=(220, 220, 220))
    img.save(path)


@pytest.fixture(autouse=True)
def _no_rate_limit():
    with patch("missy.tools.builtin.x11_tools._check_rate_limit", return_value=None):
        yield


class TestUniformDetection:
    def test_black_png_is_uniform(self, tmp_path):
        p = str(tmp_path / "b.png")
        _black_png(p)
        assert uniform_image_file(p) is not None

    def test_solid_colour_png_is_uniform(self, tmp_path):
        p = str(tmp_path / "w.png")
        Image.new("RGB", (64, 64), (30, 90, 200)).save(p)
        assert uniform_image_file(p) is not None

    def test_terminal_with_one_line_of_text_is_not_uniform(self, tmp_path):
        p = str(tmp_path / "t.png")
        _real_png(p)
        assert uniform_image_file(p) is None

    def test_unreadable_file_fails_open(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(b"not an image")
        assert uniform_image_file(str(p)) is None

    def test_missing_pil_fails_open(self, tmp_path):
        p = str(tmp_path / "b.png")
        _black_png(p)
        with patch.dict("sys.modules", {"PIL": None}):
            assert uniform_image_file(p) is None

    def test_array_check(self):
        assert uniform_image_array(np.zeros((480, 640, 3), dtype=np.uint8)) is not None
        grad = np.tile(np.arange(100, dtype=np.uint8)[:, None, None], (1, 100, 3))
        assert uniform_image_array(grad) is None


class TestReadScreenBlankGuard:
    def test_blank_capture_not_sent_to_vision(self, tmp_path):
        path = str(tmp_path / "screen.png")
        tool = X11ReadScreenTool()
        with (
            patch.object(tool, "_take_screenshot", side_effect=lambda p, r: _black_png(p)),
            patch.object(tool, "_call_ollama_vision") as vision,
        ):
            result = tool.execute(path=path)
        vision.assert_not_called()
        assert result.success is False
        assert "blank" in result.error.lower()
        assert result.output["blank_capture"] is True

    def test_real_capture_still_sent_to_vision(self, tmp_path):
        path = str(tmp_path / "screen.png")
        tool = X11ReadScreenTool()
        with (
            patch.object(tool, "_take_screenshot", side_effect=lambda p, r: _real_png(p)),
            patch.object(tool, "_call_ollama_vision", return_value="a terminal") as vision,
            patch(
                "missy.tools.builtin.x11_tools._extract_native_ocr_coordinates",
                return_value={},
            ),
            patch(
                "missy.tools.builtin.x11_tools._redact_screenshot_secrets",
                return_value={},
            ),
        ):
            result = tool.execute(path=path)
        vision.assert_called_once()
        assert result.success is True


class TestScreenshotBlankGuard:
    def test_blank_screenshot_flagged(self, tmp_path):
        path = str(tmp_path / "shot.png")

        def fake_run(cmd, **_):
            _black_png(path)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch("missy.tools.builtin.x11_tools._run", side_effect=fake_run):
            result = X11ScreenshotTool().execute(path=path, redact=False)
        assert result.success is False
        assert "blank" in result.error.lower()


class TestVisionToolsBlankGuard:
    def test_vision_capture_screenshot_blank(self):
        from missy.tools.builtin.vision_tools import VisionCaptureTool

        frame = MagicMock()
        frame.image = np.zeros((480, 640, 3), dtype=np.uint8)
        with (
            patch("missy.vision.sources.ScreenshotSource.is_available", return_value=True),
            patch("missy.vision.sources.ScreenshotSource.acquire", return_value=frame),
        ):
            result = VisionCaptureTool().execute(source="screenshot")
        assert result.success is False
        assert "blank" in result.error.lower()

    def test_vision_analyze_blank_image_skips_provider(self, tmp_path):
        from missy.tools.builtin.vision_tools import VisionAnalyzeTool

        frame = MagicMock()
        frame.image = np.zeros((480, 640, 3), dtype=np.uint8)
        with (
            patch("missy.vision.sources.FileSource.is_available", return_value=True),
            patch("missy.vision.sources.FileSource.acquire", return_value=frame),
            patch("missy.vision.provider_call.analyze_image_with_provider_fallback") as analyze,
        ):
            result = VisionAnalyzeTool().execute(source=str(tmp_path / "shot.png"))
        analyze.assert_not_called()
        assert result.success is False
        assert "blank" in result.error.lower()
