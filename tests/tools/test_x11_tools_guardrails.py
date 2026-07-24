"""Tests for the guardrails added to missy/tools/builtin/x11_tools.py:
rate limiting, window allowlist gating, and best-effort OCR-based
screenshot secret redaction.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.tools.builtin.x11_tools import (
    X11ClickTool,
    X11KeyTool,
    X11ReadScreenTool,
    X11ScreenshotTool,
    X11TypeTool,
    X11WindowListTool,
    _redact_screenshot_secrets,
)


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestX11RateLimiting:
    def test_click_denied_when_rate_limited(self):
        with patch(
            "missy.tools.builtin.x11_tools._check_rate_limit",
            return_value="Rate limit exceeded",
        ):
            result = X11ClickTool().execute(x=0, y=0)
        assert result.success is False
        assert "Rate limit" in result.error

    def test_type_denied_when_rate_limited(self):
        with patch(
            "missy.tools.builtin.x11_tools._check_rate_limit",
            return_value="Rate limit exceeded",
        ):
            result = X11TypeTool().execute(text="hi")
        assert result.success is False

    def test_key_denied_when_rate_limited(self):
        with patch(
            "missy.tools.builtin.x11_tools._check_rate_limit",
            return_value="Rate limit exceeded",
        ):
            result = X11KeyTool().execute(key="Return")
        assert result.success is False

    def test_window_list_denied_when_rate_limited(self):
        with patch(
            "missy.tools.builtin.x11_tools._check_rate_limit",
            return_value="Rate limit exceeded",
        ):
            result = X11WindowListTool().execute()
        assert result.success is False

    def test_screenshot_denied_when_rate_limited(self):
        with patch(
            "missy.tools.builtin.x11_tools._check_rate_limit",
            return_value="Rate limit exceeded",
        ):
            result = X11ScreenshotTool().execute()
        assert result.success is False

    def test_click_not_rate_limited_by_default(self):
        """A single call within the default 30/min budget must succeed."""
        with patch("missy.tools.builtin.x11_tools._run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            result = X11ClickTool().execute(x=1, y=1)
        assert result.success is True


# ---------------------------------------------------------------------------
# Window allowlist (only enforced once desktop.enabled is set; see
# _desktop_shared.check_window_allowed's backward-compatibility note)
# ---------------------------------------------------------------------------


class TestX11WindowAllowlist:
    def test_click_without_window_name_never_checks_allowlist(self):
        with (
            patch("missy.tools.builtin.x11_tools._check_window_allowed") as mock_check,
            patch("missy.tools.builtin.x11_tools._run") as mock_run,
        ):
            mock_run.return_value = _completed(returncode=0)
            X11ClickTool().execute(x=1, y=1)
        mock_check.assert_called_once_with("")

    def test_click_with_window_name_denied_by_allowlist(self):
        with patch(
            "missy.tools.builtin.x11_tools._check_window_allowed",
            return_value="requires approval",
        ):
            result = X11ClickTool().execute(x=1, y=1, window_name="Secret")
        assert result.success is False
        assert "requires approval" in result.error

    def test_type_with_window_name_denied_by_allowlist(self):
        with patch(
            "missy.tools.builtin.x11_tools._check_window_allowed",
            return_value="requires approval",
        ):
            result = X11TypeTool().execute(text="hi", window_name="Secret")
        assert result.success is False

    def test_key_with_window_name_denied_by_allowlist(self):
        with patch(
            "missy.tools.builtin.x11_tools._check_window_allowed",
            return_value="requires approval",
        ):
            result = X11KeyTool().execute(key="Return", window_name="Secret")
        assert result.success is False

    def test_click_with_allowed_window_name_proceeds(self):
        with (
            patch("missy.tools.builtin.x11_tools._check_window_allowed", return_value=None),
            patch("missy.tools.builtin.x11_tools._run") as mock_run,
        ):
            mock_run.return_value = _completed(returncode=0)
            result = X11ClickTool().execute(x=1, y=1, window_name="Firefox")
        assert result.success is True

    def test_backward_compatible_without_any_config(self, monkeypatch):
        """End-to-end (no mocking of _check_window_allowed itself): with no
        config file at all, window_name targeting must still work exactly
        as it did before this guardrail existed."""
        monkeypatch.setenv("MISSY_CONFIG", "/nonexistent/config.yaml")
        with patch("missy.tools.builtin.x11_tools._run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            result = X11ClickTool().execute(x=1, y=1, window_name="AnyApp")
        assert result.success is True


# ---------------------------------------------------------------------------
# _redact_screenshot_secrets
# ---------------------------------------------------------------------------


class TestRedactScreenshotSecrets:
    def test_reports_unavailable_when_pytesseract_missing(self):
        with patch.dict("sys.modules", {"pytesseract": None}):
            result = _redact_screenshot_secrets("/tmp/whatever.png")
        assert result["redaction_available"] is False
        assert result["redaction_applied"] is False
        assert "pytesseract" in result["note"]

    def test_redacts_matching_word_and_saves(self, tmp_path):
        img_path = tmp_path / "shot.png"
        from PIL import Image

        Image.new("RGB", (200, 100), color="white").save(img_path)

        fake_pytesseract = MagicMock()
        fake_pytesseract.Output.DICT = "dict"
        fake_pytesseract.image_to_data.return_value = {
            "text": ["hello", "AKIAABCDEFGHIJKLMNOP", "world"],
            "left": [0, 50, 150],
            "top": [0, 0, 0],
            "width": [40, 90, 40],
            "height": [20, 20, 20],
        }

        with patch.dict("sys.modules", {"pytesseract": fake_pytesseract}):
            result = _redact_screenshot_secrets(str(img_path))

        assert result["redaction_available"] is True
        assert result["redaction_applied"] is True
        assert result["redacted_regions"] == 1

    def test_no_secrets_found_does_not_rewrite_file(self, tmp_path):
        img_path = tmp_path / "shot.png"
        from PIL import Image

        Image.new("RGB", (200, 100), color="white").save(img_path)
        original_mtime = img_path.stat().st_mtime_ns

        fake_pytesseract = MagicMock()
        fake_pytesseract.Output.DICT = "dict"
        fake_pytesseract.image_to_data.return_value = {
            "text": ["hello", "world"],
            "left": [0, 50],
            "top": [0, 0],
            "width": [40, 40],
            "height": [20, 20],
        }

        with patch.dict("sys.modules", {"pytesseract": fake_pytesseract}):
            result = _redact_screenshot_secrets(str(img_path))

        assert result["redaction_applied"] is False
        assert result["redacted_regions"] == 0
        # File untouched -- no unnecessary rewrite when nothing was found.
        assert img_path.stat().st_mtime_ns == original_mtime

    def test_ocr_failure_is_reported_not_raised(self, tmp_path):
        fake_pytesseract = MagicMock()
        fake_pytesseract.Output.DICT = "dict"

        with patch.dict("sys.modules", {"pytesseract": fake_pytesseract}):
            result = _redact_screenshot_secrets(str(tmp_path / "nonexistent.png"))

        assert result["redaction_available"] is True
        assert result["redaction_applied"] is False
        assert result["note"] is not None

    def test_blank_ocr_words_are_skipped(self, tmp_path):
        img_path = tmp_path / "shot.png"
        from PIL import Image

        Image.new("RGB", (200, 100), color="white").save(img_path)

        fake_pytesseract = MagicMock()
        fake_pytesseract.Output.DICT = "dict"
        fake_pytesseract.image_to_data.return_value = {
            "text": ["", "   ", ""],
            "left": [0, 0, 0],
            "top": [0, 0, 0],
            "width": [1, 1, 1],
            "height": [1, 1, 1],
        }

        with patch.dict("sys.modules", {"pytesseract": fake_pytesseract}):
            result = _redact_screenshot_secrets(str(img_path))

        assert result["redacted_regions"] == 0


# ---------------------------------------------------------------------------
# X11ScreenshotTool redact parameter wiring
# ---------------------------------------------------------------------------


class TestX11ScreenshotToolRedaction:
    def test_redact_true_calls_redaction_helper(self, tmp_path):
        path = str(tmp_path / "shot.png")
        with (
            patch("missy.tools.builtin.x11_tools._run") as mock_run,
            patch("missy.tools.builtin.x11_tools._redact_screenshot_secrets") as mock_redact,
            patch("os.path.getsize", return_value=100),
        ):
            mock_run.return_value = _completed(returncode=0)
            mock_redact.return_value = {
                "redaction_available": True,
                "redaction_applied": False,
                "redacted_regions": 0,
                "note": None,
            }
            result = X11ScreenshotTool().execute(path=path, redact=True)

        mock_redact.assert_called_once_with(path)
        assert result.output["redaction_available"] is True

    def test_redact_false_skips_scan(self, tmp_path):
        path = str(tmp_path / "shot.png")
        with (
            patch("missy.tools.builtin.x11_tools._run") as mock_run,
            patch("missy.tools.builtin.x11_tools._redact_screenshot_secrets") as mock_redact,
            patch("os.path.getsize", return_value=100),
        ):
            mock_run.return_value = _completed(returncode=0)
            result = X11ScreenshotTool().execute(path=path, redact=False)

        mock_redact.assert_not_called()
        assert result.output["redaction_applied"] is False
        assert "redact=False" in result.output["note"]

    def test_redact_defaults_to_true(self, tmp_path):
        path = str(tmp_path / "shot.png")
        with (
            patch("missy.tools.builtin.x11_tools._run") as mock_run,
            patch("missy.tools.builtin.x11_tools._redact_screenshot_secrets") as mock_redact,
            patch("os.path.getsize", return_value=100),
        ):
            mock_run.return_value = _completed(returncode=0)
            mock_redact.return_value = {
                "redaction_available": False,
                "redaction_applied": False,
                "redacted_regions": 0,
                "note": "x",
            }
            X11ScreenshotTool().execute(path=path)

        mock_redact.assert_called_once()


# ---------------------------------------------------------------------------
# X11ReadScreenTool text redaction of the vision description
# ---------------------------------------------------------------------------


class TestX11ReadScreenToolRedaction:
    def test_native_ocr_box_text_is_censored_and_low_confidence_is_dropped(self, tmp_path):
        pytest.importorskip("pytesseract")
        from PIL import Image

        from missy.tools.builtin.x11_tools import _extract_native_ocr_coordinates

        path = str(tmp_path / "screen.png")
        Image.new("RGB", (800, 600), "white").save(path)
        ocr_data = {
            "text": ["AKIAABCDEFGHIJKLMNOP", "noise"],
            "conf": ["99", "12"],
            "left": [10, 20],
            "top": [30, 40],
            "width": [200, 10],
            "height": [20, 10],
        }
        with patch("pytesseract.image_to_data", return_value=ocr_data):
            metadata = _extract_native_ocr_coordinates(path)

        assert metadata["ocr_coordinates_available"] is True
        assert len(metadata["ocr_text_boxes"]) == 1
        assert "AKIAABCDEFGHIJKLMNOP" not in metadata["ocr_text_boxes"][0]["text"]

    def test_secret_in_vision_description_is_censored(self, tmp_path):
        path = str(tmp_path / "screen.png")
        tool = X11ReadScreenTool()

        with (
            patch.object(tool, "_take_screenshot", return_value=None),
            patch("builtins.open", MagicMock()),
            patch("base64.b64encode", return_value=b"ZmFrZQ=="),
            patch.object(
                tool,
                "_call_ollama_vision",
                return_value="I see an API key: AKIAABCDEFGHIJKLMNOP on screen.",
            ),
            patch(
                "missy.tools.builtin.x11_tools._redact_screenshot_secrets",
                return_value={
                    "redaction_available": True,
                    "redaction_applied": False,
                    "redacted_regions": 0,
                    "note": None,
                },
            ),
        ):
            result = tool.execute(path=path)

        assert result.success is True
        assert "AKIAABCDEFGHIJKLMNOP" not in result.output["description"]

    def test_also_redacts_the_saved_screenshot_file(self, tmp_path):
        path = str(tmp_path / "screen.png")
        tool = X11ReadScreenTool()

        with (
            patch.object(tool, "_take_screenshot", return_value=None),
            patch("builtins.open", MagicMock()),
            patch("base64.b64encode", return_value=b"ZmFrZQ=="),
            patch.object(tool, "_call_ollama_vision", return_value="Nothing sensitive here."),
            patch("missy.tools.builtin.x11_tools._redact_screenshot_secrets") as mock_redact,
        ):
            mock_redact.return_value = {
                "redaction_available": True,
                "redaction_applied": True,
                "redacted_regions": 2,
                "note": None,
            }
            result = tool.execute(path=path)

        mock_redact.assert_called_once_with(path)
        assert result.output["redacted_regions"] == 2
