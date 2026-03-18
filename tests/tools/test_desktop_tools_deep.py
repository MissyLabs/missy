"""Deep tests for desktop automation tools: browser, TTS, X11, AT-SPI.

All external dependencies (playwright, subprocess, pyatspi) are mocked
so tests run without a display server or audio stack.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest


# =========================================================================
# X11 tools helpers
# =========================================================================


class TestDisplayEnv:
    """Tests for x11_tools._display_env()."""

    def test_display_from_environment(self, monkeypatch):
        monkeypatch.setenv("DISPLAY", ":42")
        monkeypatch.setenv("PATH", "/usr/bin")
        from missy.tools.builtin.x11_tools import _display_env

        env = _display_env()
        assert env["DISPLAY"] == ":42"
        assert env["PATH"] == "/usr/bin"

    def test_display_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("DISPLAY", raising=False)
        from missy.tools.builtin.x11_tools import _display_env

        env = _display_env()
        assert env["DISPLAY"] == ":0"

    def test_only_safe_vars_passed(self, monkeypatch):
        """Env must not leak API keys to subprocesses."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        monkeypatch.setenv("PATH", "/usr/bin")
        from missy.tools.builtin.x11_tools import _display_env

        env = _display_env()
        assert "ANTHROPIC_API_KEY" not in env
        assert "OPENAI_API_KEY" not in env
        assert "PATH" in env


class TestButtonNum:
    """Tests for x11_tools._button_num()."""

    def test_left(self):
        from missy.tools.builtin.x11_tools import _button_num

        assert _button_num("left") == "1"

    def test_middle(self):
        from missy.tools.builtin.x11_tools import _button_num

        assert _button_num("middle") == "2"

    def test_right(self):
        from missy.tools.builtin.x11_tools import _button_num

        assert _button_num("right") == "3"

    def test_case_insensitive(self):
        from missy.tools.builtin.x11_tools import _button_num

        assert _button_num("LEFT") == "1"
        assert _button_num("Right") == "3"

    def test_unknown_defaults_to_left(self):
        from missy.tools.builtin.x11_tools import _button_num

        assert _button_num("unknown") == "1"


class TestX11ExtractAccountId:
    """Tests for x11_tools._extract_account_id()."""

    def test_valid_jwt_with_account_id(self):
        import base64
        from missy.tools.builtin.x11_tools import _extract_account_id

        payload = {
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"},
            "sub": "user_456",
        }
        b64_payload = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
        token = f"header.{b64_payload}.signature"
        assert _extract_account_id(token) == "acct_123"

    def test_fallback_to_sub(self):
        import base64
        from missy.tools.builtin.x11_tools import _extract_account_id

        payload = {"sub": "user_789"}
        b64_payload = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
        token = f"header.{b64_payload}.signature"
        assert _extract_account_id(token) == "user_789"

    def test_invalid_jwt(self):
        from missy.tools.builtin.x11_tools import _extract_account_id

        assert _extract_account_id("not-a-jwt") == ""

    def test_empty_token(self):
        from missy.tools.builtin.x11_tools import _extract_account_id

        assert _extract_account_id("") == ""


class TestX11ScreenshotTool:
    """Tests for X11ScreenshotTool.execute()."""

    @patch("missy.tools.builtin.x11_tools._run")
    def test_screenshot_success(self, mock_run, tmp_path):
        from missy.tools.builtin.x11_tools import X11ScreenshotTool

        out_path = str(tmp_path / "shot.png")
        # Create the file to simulate scrot writing it
        Path(out_path).write_bytes(b"\x89PNG fake")

        mock_run.return_value = subprocess.CompletedProcess(args="scrot", returncode=0, stdout="", stderr="")
        tool = X11ScreenshotTool()
        result = tool.execute(path=out_path)
        assert result.success is True
        assert result.output["path"] == out_path

    @patch("missy.tools.builtin.x11_tools._run")
    def test_screenshot_scrot_not_installed(self, mock_run):
        from missy.tools.builtin.x11_tools import X11ScreenshotTool

        mock_run.return_value = subprocess.CompletedProcess(
            args="scrot", returncode=127, stdout="", stderr="command not found"
        )
        tool = X11ScreenshotTool()
        result = tool.execute(path="/tmp/test.png")
        assert result.success is False
        assert "scrot is not installed" in result.error

    @patch("missy.tools.builtin.x11_tools._run")
    def test_screenshot_with_region(self, mock_run, tmp_path):
        from missy.tools.builtin.x11_tools import X11ScreenshotTool

        out_path = str(tmp_path / "region.png")
        Path(out_path).write_bytes(b"\x89PNG")
        mock_run.return_value = subprocess.CompletedProcess(args="", returncode=0, stdout="", stderr="")
        tool = X11ScreenshotTool()
        result = tool.execute(path=out_path, region="100,100,200,200")
        assert result.success is True
        # Check that region was included in command
        call_args = mock_run.call_args[0][0]
        assert "100,100,200,200" in call_args


class TestX11ClickTool:
    """Tests for X11ClickTool.execute()."""

    @patch("missy.tools.builtin.x11_tools._run")
    def test_left_click(self, mock_run):
        from missy.tools.builtin.x11_tools import X11ClickTool

        mock_run.return_value = subprocess.CompletedProcess(args="", returncode=0, stdout="", stderr="")
        tool = X11ClickTool()
        result = tool.execute(x=100, y=200, button="left")
        assert result.success is True
        assert result.output["x"] == 100
        assert result.output["y"] == 200

    @patch("missy.tools.builtin.x11_tools._run")
    def test_double_click(self, mock_run):
        from missy.tools.builtin.x11_tools import X11ClickTool

        mock_run.return_value = subprocess.CompletedProcess(args="", returncode=0, stdout="", stderr="")
        tool = X11ClickTool()
        result = tool.execute(x=50, y=50, button="double")
        assert result.success is True
        call_cmd = mock_run.call_args[0][0]
        assert "--repeat 2" in call_cmd

    @patch("missy.tools.builtin.x11_tools._run")
    def test_click_with_window_focus(self, mock_run):
        from missy.tools.builtin.x11_tools import X11ClickTool

        mock_run.return_value = subprocess.CompletedProcess(args="", returncode=0, stdout="", stderr="")
        tool = X11ClickTool()
        result = tool.execute(x=10, y=20, window_name="Firefox")
        assert result.success is True
        # Should have been called twice: focus + click
        assert mock_run.call_count == 2

    @patch("missy.tools.builtin.x11_tools._run")
    def test_click_xdotool_not_installed(self, mock_run):
        from missy.tools.builtin.x11_tools import X11ClickTool

        mock_run.return_value = subprocess.CompletedProcess(
            args="", returncode=127, stdout="", stderr="command not found"
        )
        tool = X11ClickTool()
        result = tool.execute(x=0, y=0)
        assert result.success is False
        assert "xdotool is not installed" in result.error

    @patch("missy.tools.builtin.x11_tools._run")
    def test_click_enforces_int_types(self, mock_run):
        """x/y must be coerced to int to prevent shell injection."""
        from missy.tools.builtin.x11_tools import X11ClickTool

        mock_run.return_value = subprocess.CompletedProcess(args="", returncode=0, stdout="", stderr="")
        tool = X11ClickTool()
        result = tool.execute(x="100", y="200")  # strings should be coerced
        assert result.success is True


class TestX11TypeTool:
    """Tests for X11TypeTool.execute()."""

    @patch("missy.tools.builtin.x11_tools._run")
    def test_type_text(self, mock_run):
        from missy.tools.builtin.x11_tools import X11TypeTool

        mock_run.return_value = subprocess.CompletedProcess(args="", returncode=0, stdout="", stderr="")
        tool = X11TypeTool()
        result = tool.execute(text="hello world")
        assert result.success is True


# =========================================================================
# TTS tools
# =========================================================================


class TestEnsureRuntimeDir:
    """Tests for tts_speak._ensure_runtime_dir()."""

    def test_returns_safe_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
        monkeypatch.setenv("PATH", "/usr/bin")
        from missy.tools.builtin.tts_speak import _ensure_runtime_dir

        env = _ensure_runtime_dir()
        assert "ANTHROPIC_API_KEY" not in env
        assert "PATH" in env

    def test_sets_xdg_runtime_dir_if_missing(self, monkeypatch):
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        from missy.tools.builtin.tts_speak import _ensure_runtime_dir

        env = _ensure_runtime_dir()
        assert "XDG_RUNTIME_DIR" in env
        assert env["XDG_RUNTIME_DIR"].startswith("/run/user/")

    def test_preserves_existing_xdg_runtime_dir(self, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/custom/run")
        from missy.tools.builtin.tts_speak import _ensure_runtime_dir

        env = _ensure_runtime_dir()
        assert env["XDG_RUNTIME_DIR"] == "/custom/run"


class TestPiperEnv:
    """Tests for tts_speak._piper_env()."""

    def test_adds_ld_library_path(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        from missy.tools.builtin.tts_speak import _piper_env, _PIPER_BIN

        env = _piper_env()
        assert str(_PIPER_BIN.parent) in env.get("LD_LIBRARY_PATH", "")


class TestFindPiperModel:
    """Tests for tts_speak._find_piper_model()."""

    @patch("missy.tools.builtin.tts_speak._PIPER_VOICES_DIR")
    def test_voices_dir_not_exists(self, mock_dir):
        mock_dir.is_dir.return_value = False
        from missy.tools.builtin.tts_speak import _find_piper_model

        assert _find_piper_model("en_US-lessac-medium") is None


class TestSynthPiper:
    """Tests for tts_speak._synth_piper()."""

    @patch("missy.tools.builtin.tts_speak._find_piper_model")
    @patch("missy.tools.builtin.tts_speak._PIPER_BIN")
    def test_piper_binary_not_found(self, mock_bin, mock_model):
        mock_bin.is_file.return_value = False
        from missy.tools.builtin.tts_speak import _synth_piper

        result = _synth_piper("hello", "/tmp/out.wav", "en_US-lessac-medium", 1.0)
        assert result == "piper binary not found"

    @patch("missy.tools.builtin.tts_speak._find_piper_model")
    @patch("missy.tools.builtin.tts_speak._PIPER_BIN")
    def test_piper_voice_not_found(self, mock_bin, mock_model):
        mock_bin.is_file.return_value = True
        mock_model.return_value = None
        from missy.tools.builtin.tts_speak import _synth_piper

        result = _synth_piper("hello", "/tmp/out.wav", "nonexistent", 1.0)
        assert "voice model not found" in result

    @patch("missy.tools.builtin.tts_speak._piper_env")
    @patch("missy.tools.builtin.tts_speak._find_piper_model")
    @patch("missy.tools.builtin.tts_speak._PIPER_BIN")
    @patch("subprocess.run")
    def test_piper_success(self, mock_subproc, mock_bin, mock_model, mock_env, tmp_path):
        mock_bin.is_file.return_value = True
        mock_bin.__str__ = lambda self: "/usr/bin/piper"
        mock_model.return_value = Path("/models/voice.onnx")
        mock_env.return_value = {"PATH": "/usr/bin"}

        wav = tmp_path / "out.wav"
        wav.write_bytes(b"RIFF fake wav data")

        mock_subproc.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")

        from missy.tools.builtin.tts_speak import _synth_piper

        result = _synth_piper("hello", str(wav), "en_US-lessac-medium", 1.0)
        assert result is None  # success

    @patch("missy.tools.builtin.tts_speak._piper_env")
    @patch("missy.tools.builtin.tts_speak._find_piper_model")
    @patch("missy.tools.builtin.tts_speak._PIPER_BIN")
    @patch("subprocess.run")
    def test_piper_timeout(self, mock_subproc, mock_bin, mock_model, mock_env):
        mock_bin.is_file.return_value = True
        mock_bin.__str__ = lambda self: "/usr/bin/piper"
        mock_model.return_value = Path("/models/voice.onnx")
        mock_env.return_value = {"PATH": "/usr/bin"}
        mock_subproc.side_effect = subprocess.TimeoutExpired(cmd="piper", timeout=60)

        from missy.tools.builtin.tts_speak import _synth_piper

        result = _synth_piper("hello", "/tmp/out.wav", "en_US-lessac-medium", 1.0)
        assert result == "piper timed out"


class TestSynthEspeak:
    """Tests for tts_speak._synth_espeak()."""

    @patch("subprocess.run")
    def test_espeak_success(self, mock_run, tmp_path):
        wav_path = str(tmp_path / "out.wav")
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"RIFF fake wav data", stderr=b""
        )
        from missy.tools.builtin.tts_speak import _synth_espeak

        result = _synth_espeak("hello", wav_path, 160, 50, "en", {"PATH": "/usr/bin"})
        assert result is None

    @patch("subprocess.run")
    def test_espeak_not_installed(self, mock_run):
        mock_run.side_effect = FileNotFoundError("espeak-ng")
        from missy.tools.builtin.tts_speak import _synth_espeak

        result = _synth_espeak("hello", "/tmp/out.wav", 160, 50, "en", {})
        assert result == "espeak-ng not installed"

    @patch("subprocess.run")
    def test_espeak_no_output(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
        from missy.tools.builtin.tts_speak import _synth_espeak

        result = _synth_espeak("hello", "/tmp/out.wav", 160, 50, "en", {})
        assert result == "espeak-ng produced no audio"

    @patch("subprocess.run")
    def test_espeak_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="espeak-ng", timeout=30)
        from missy.tools.builtin.tts_speak import _synth_espeak

        result = _synth_espeak("hello", "/tmp/out.wav", 160, 50, "en", {})
        assert result == "espeak-ng timed out"


class TestPlayWav:
    """Tests for tts_speak._play_wav()."""

    @patch("subprocess.run")
    def test_play_success(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
        from missy.tools.builtin.tts_speak import _play_wav

        result = _play_wav("/tmp/test.wav", {"PATH": "/usr/bin"})
        assert result is None

    @patch("subprocess.run")
    def test_gst_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("gst-launch-1.0")
        from missy.tools.builtin.tts_speak import _play_wav

        result = _play_wav("/tmp/test.wav", {})
        assert result == "gst-launch-1.0 not found"

    @patch("subprocess.run")
    def test_play_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="gst-launch", timeout=60)
        from missy.tools.builtin.tts_speak import _play_wav

        result = _play_wav("/tmp/test.wav", {})
        assert result == "audio playback timed out"

    @patch("subprocess.run")
    def test_gst_command_not_found_in_stderr(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=127, stdout=b"", stderr=b"command not found"
        )
        from missy.tools.builtin.tts_speak import _play_wav

        result = _play_wav("/tmp/test.wav", {})
        assert "gst-launch-1.0 not installed" in result


class TestTTSSpeakTool:
    """Tests for TTSSpeakTool.execute()."""

    def test_empty_text(self):
        from missy.tools.builtin.tts_speak import TTSSpeakTool

        tool = TTSSpeakTool()
        result = tool.execute(text="")
        assert result.success is False
        assert "No text" in result.error

    def test_whitespace_only(self):
        from missy.tools.builtin.tts_speak import TTSSpeakTool

        tool = TTSSpeakTool()
        result = tool.execute(text="   ")
        assert result.success is False

    @patch("missy.tools.builtin.tts_speak._play_wav")
    @patch("missy.tools.builtin.tts_speak._synth_piper")
    def test_piper_success_path(self, mock_piper, mock_play):
        mock_piper.return_value = None  # success
        mock_play.return_value = None  # success
        from missy.tools.builtin.tts_speak import TTSSpeakTool

        tool = TTSSpeakTool()
        result = tool.execute(text="Hello world")
        assert result.success is True
        assert "2 words" in result.output

    @patch("missy.tools.builtin.tts_speak._play_wav")
    @patch("missy.tools.builtin.tts_speak._synth_espeak")
    @patch("missy.tools.builtin.tts_speak._synth_piper")
    def test_piper_fallback_to_espeak(self, mock_piper, mock_espeak, mock_play):
        mock_piper.return_value = "piper binary not found"
        mock_espeak.return_value = None  # success
        mock_play.return_value = None
        from missy.tools.builtin.tts_speak import TTSSpeakTool

        tool = TTSSpeakTool()
        result = tool.execute(text="Hello")
        assert result.success is True
        assert "engine=espeak-ng" in result.output

    @patch("missy.tools.builtin.tts_speak._synth_espeak")
    @patch("missy.tools.builtin.tts_speak._synth_piper")
    def test_both_engines_fail(self, mock_piper, mock_espeak):
        mock_piper.return_value = "piper not found"
        mock_espeak.return_value = "espeak-ng not installed"
        from missy.tools.builtin.tts_speak import TTSSpeakTool

        tool = TTSSpeakTool()
        result = tool.execute(text="Hello")
        assert result.success is False
        assert "TTS synthesis failed" in result.error

    @patch("missy.tools.builtin.tts_speak._play_wav")
    @patch("missy.tools.builtin.tts_speak._synth_piper")
    def test_playback_failure(self, mock_piper, mock_play):
        mock_piper.return_value = None
        mock_play.return_value = "gst-launch-1.0 not found"
        from missy.tools.builtin.tts_speak import TTSSpeakTool

        tool = TTSSpeakTool()
        result = tool.execute(text="Hello")
        assert result.success is False
        assert "gst-launch" in result.error

    @patch("missy.tools.builtin.tts_speak._play_wav")
    @patch("missy.tools.builtin.tts_speak._synth_piper")
    def test_speed_clamping(self, mock_piper, mock_play):
        mock_piper.return_value = None
        mock_play.return_value = None
        from missy.tools.builtin.tts_speak import TTSSpeakTool

        tool = TTSSpeakTool()
        # Speed too high should be clamped to 4.0
        result = tool.execute(text="Hello", speed=100.0)
        assert result.success is True
        # Speed too low should be clamped to 0.25
        result = tool.execute(text="Hello", speed=0.01)
        assert result.success is True


# =========================================================================
# Browser tools
# =========================================================================


class TestBrowserSession:
    """Tests for browser_tools.BrowserSession."""

    def test_valid_session_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr("missy.tools.builtin.browser_tools._SESSIONS_DIR", tmp_path)
        from missy.tools.builtin.browser_tools import BrowserSession

        session = BrowserSession("test-session_123")
        assert session.session_id == "test-session_123"

    def test_invalid_session_id_special_chars(self, tmp_path, monkeypatch):
        monkeypatch.setattr("missy.tools.builtin.browser_tools._SESSIONS_DIR", tmp_path)
        from missy.tools.builtin.browser_tools import BrowserSession

        with pytest.raises(ValueError, match="must be alphanumeric"):
            BrowserSession("../../etc/passwd")

    def test_session_id_too_long(self, tmp_path, monkeypatch):
        monkeypatch.setattr("missy.tools.builtin.browser_tools._SESSIONS_DIR", tmp_path)
        from missy.tools.builtin.browser_tools import BrowserSession

        with pytest.raises(ValueError, match="exceeds maximum length"):
            BrowserSession("a" * 200)

    def test_session_id_with_spaces(self, tmp_path, monkeypatch):
        monkeypatch.setattr("missy.tools.builtin.browser_tools._SESSIONS_DIR", tmp_path)
        from missy.tools.builtin.browser_tools import BrowserSession

        with pytest.raises(ValueError, match="must be alphanumeric"):
            BrowserSession("session with spaces")

    def test_close_without_start(self, tmp_path, monkeypatch):
        monkeypatch.setattr("missy.tools.builtin.browser_tools._SESSIONS_DIR", tmp_path)
        from missy.tools.builtin.browser_tools import BrowserSession

        session = BrowserSession("test")
        session.close()  # should not raise


class TestSessionRegistry:
    """Tests for browser_tools._SessionRegistry."""

    def test_get_or_create_creates_new(self, tmp_path, monkeypatch):
        monkeypatch.setattr("missy.tools.builtin.browser_tools._SESSIONS_DIR", tmp_path)
        from missy.tools.builtin.browser_tools import _SessionRegistry

        reg = _SessionRegistry()
        s1 = reg.get_or_create("session1")
        s2 = reg.get_or_create("session1")
        assert s1 is s2  # same instance

    def test_has_active_session_initially_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr("missy.tools.builtin.browser_tools._SESSIONS_DIR", tmp_path)
        from missy.tools.builtin.browser_tools import _SessionRegistry

        reg = _SessionRegistry()
        assert reg.has_active_session() is False

    def test_close_session(self, tmp_path, monkeypatch):
        monkeypatch.setattr("missy.tools.builtin.browser_tools._SESSIONS_DIR", tmp_path)
        from missy.tools.builtin.browser_tools import _SessionRegistry

        reg = _SessionRegistry()
        reg.get_or_create("s1")
        reg.close("s1")
        # Should be removed from registry
        s_new = reg.get_or_create("s1")
        assert s_new is not None

    def test_close_nonexistent_session(self, tmp_path, monkeypatch):
        monkeypatch.setattr("missy.tools.builtin.browser_tools._SESSIONS_DIR", tmp_path)
        from missy.tools.builtin.browser_tools import _SessionRegistry

        reg = _SessionRegistry()
        reg.close("nonexistent")  # should not raise

    def test_screenshot_active_no_sessions(self, tmp_path, monkeypatch):
        monkeypatch.setattr("missy.tools.builtin.browser_tools._SESSIONS_DIR", tmp_path)
        from missy.tools.builtin.browser_tools import _SessionRegistry

        reg = _SessionRegistry()
        assert reg.screenshot_active("/tmp/shot.png") is False


class TestBrowserToolErr:
    """Tests for browser_tools._err()."""

    def test_err_helper(self):
        from missy.tools.builtin.browser_tools import _err

        result = _err(ValueError("test error"))
        assert result.success is False
        assert result.error == "test error"


class TestBrowserNavigateTool:
    """Tests for BrowserNavigateTool."""

    def test_tool_attributes(self):
        from missy.tools.builtin.browser_tools import BrowserNavigateTool

        tool = BrowserNavigateTool()
        assert tool.name == "browser_navigate"
        assert tool.permissions.network is True

    @patch("missy.tools.builtin.browser_tools._page")
    def test_navigate_success(self, mock_page):
        from missy.tools.builtin.browser_tools import BrowserNavigateTool

        page = MagicMock()
        page.url = "https://example.com"
        page.title.return_value = "Example"
        mock_page.return_value = page

        tool = BrowserNavigateTool()
        result = tool.execute(url="https://example.com")
        assert result.success is True
        assert "Example" in result.output

    @patch("missy.tools.builtin.browser_tools._page")
    def test_navigate_failure(self, mock_page):
        from missy.tools.builtin.browser_tools import BrowserNavigateTool

        mock_page.side_effect = RuntimeError("playwright not installed")
        tool = BrowserNavigateTool()
        result = tool.execute(url="https://example.com")
        assert result.success is False
        assert "playwright" in result.error


class TestBrowserClickTool:
    """Tests for BrowserClickTool."""

    def test_tool_attributes(self):
        from missy.tools.builtin.browser_tools import BrowserClickTool

        tool = BrowserClickTool()
        assert tool.name == "browser_click"


class TestBrowserCloseTool:
    """Tests for BrowserCloseTool."""

    def test_tool_exists(self):
        from missy.tools.builtin.browser_tools import BrowserCloseTool

        tool = BrowserCloseTool()
        assert tool.name == "browser_close"


class TestSafeBrowserEnvVars:
    """Tests for _SAFE_BROWSER_ENV_VARS filtering."""

    def test_safe_vars_contain_essentials(self):
        from missy.tools.builtin.browser_tools import _SAFE_BROWSER_ENV_VARS

        assert "PATH" in _SAFE_BROWSER_ENV_VARS
        assert "HOME" in _SAFE_BROWSER_ENV_VARS
        assert "DISPLAY" in _SAFE_BROWSER_ENV_VARS

    def test_safe_vars_exclude_secrets(self):
        from missy.tools.builtin.browser_tools import _SAFE_BROWSER_ENV_VARS

        assert "ANTHROPIC_API_KEY" not in _SAFE_BROWSER_ENV_VARS
        assert "OPENAI_API_KEY" not in _SAFE_BROWSER_ENV_VARS
        assert "AWS_SECRET_ACCESS_KEY" not in _SAFE_BROWSER_ENV_VARS


# =========================================================================
# AT-SPI tools helpers
# =========================================================================


class TestAtSpiFormatTree:
    """Tests for atspi_tools._format_tree()."""

    def test_format_empty(self):
        from missy.tools.builtin.atspi_tools import _format_tree

        assert _format_tree([]) == ""

    def test_format_single_node(self):
        from missy.tools.builtin.atspi_tools import _format_tree

        nodes = [{"depth": 0, "role": "frame", "name": "Main Window", "states": ["visible"]}]
        result = _format_tree(nodes)
        assert "[frame]" in result
        assert "'Main Window'" in result
        assert "(visible)" in result

    def test_format_nested_nodes(self):
        from missy.tools.builtin.atspi_tools import _format_tree

        nodes = [
            {"depth": 0, "role": "frame", "name": "App", "states": []},
            {"depth": 1, "role": "button", "name": "OK", "states": ["focusable"]},
            {"depth": 1, "role": "button", "name": "Cancel", "states": ["focusable"]},
        ]
        result = _format_tree(nodes)
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0].startswith("[frame]")
        assert lines[1].startswith("  [button]")

    def test_format_with_text(self):
        from missy.tools.builtin.atspi_tools import _format_tree

        nodes = [{"depth": 0, "role": "text", "name": "field", "states": [], "text": "Some value here"}]
        result = _format_tree(nodes)
        assert "text=" in result
        assert "Some value here" in result

    def test_format_text_same_as_name(self):
        """When text == name, text should not be shown to avoid duplication."""
        from missy.tools.builtin.atspi_tools import _format_tree

        nodes = [{"depth": 0, "role": "label", "name": "Hello", "states": [], "text": "Hello"}]
        result = _format_tree(nodes)
        assert "text=" not in result


class TestAtSpiFindApplication:
    """Tests for atspi_tools._find_application()."""

    def test_find_by_name(self):
        from missy.tools.builtin.atspi_tools import _find_application

        desktop = MagicMock()
        desktop.childCount = 2
        child1 = MagicMock()
        child1.name = "Terminal"
        child2 = MagicMock()
        child2.name = "Firefox"
        desktop.getChildAtIndex.side_effect = lambda i: [child1, child2][i]

        result = _find_application(desktop, "firefox")
        assert result is child2

    def test_find_not_found(self):
        from missy.tools.builtin.atspi_tools import _find_application

        desktop = MagicMock()
        desktop.childCount = 1
        child = MagicMock()
        child.name = "Terminal"
        desktop.getChildAtIndex.return_value = child

        result = _find_application(desktop, "Chrome")
        assert result is None

    def test_find_case_insensitive(self):
        from missy.tools.builtin.atspi_tools import _find_application

        desktop = MagicMock()
        desktop.childCount = 1
        child = MagicMock()
        child.name = "FIREFOX"
        desktop.getChildAtIndex.return_value = child

        result = _find_application(desktop, "firefox")
        assert result is child

    def test_find_handles_exception(self):
        from missy.tools.builtin.atspi_tools import _find_application

        desktop = MagicMock()
        desktop.childCount = 2
        desktop.getChildAtIndex.side_effect = Exception("AT-SPI error")

        result = _find_application(desktop, "anything")
        assert result is None


class TestAtSpiWalkTree:
    """Tests for atspi_tools._walk_tree()."""

    def test_walk_none_node(self):
        from missy.tools.builtin.atspi_tools import _walk_tree

        assert _walk_tree(None, max_depth=3) == []

    def test_walk_max_depth_exceeded(self):
        from missy.tools.builtin.atspi_tools import _walk_tree

        node = MagicMock()
        result = _walk_tree(node, max_depth=3, current_depth=4)
        assert result == []

    def test_walk_single_node(self):
        from missy.tools.builtin.atspi_tools import _walk_tree

        node = MagicMock()
        node.getRoleName.return_value = "button"
        node.name = "OK"
        node.childCount = 0
        node.queryText.side_effect = Exception("no text interface")
        node.getState.side_effect = Exception("no states")

        result = _walk_tree(node, max_depth=3, current_depth=0)
        assert len(result) == 1
        assert result[0]["role"] == "button"
        assert result[0]["name"] == "OK"


class TestAtSpiFindElement:
    """Tests for atspi_tools._find_element()."""

    def test_find_by_name_only(self):
        from missy.tools.builtin.atspi_tools import _find_element

        app = MagicMock()
        app.getRoleName.return_value = "frame"
        app.name = "Main"
        app.childCount = 1
        btn = MagicMock()
        btn.getRoleName.return_value = "push button"
        btn.name = "Submit"
        btn.childCount = 0
        app.getChildAtIndex.return_value = btn

        result = _find_element(app, name="Submit", role=None, max_depth=5)
        assert result is btn

    def test_find_by_role_only(self):
        from missy.tools.builtin.atspi_tools import _find_element

        app = MagicMock()
        app.getRoleName.return_value = "push button"
        app.name = "OK"
        app.childCount = 0

        result = _find_element(app, name=None, role="push button", max_depth=5)
        assert result is app

    def test_find_by_name_and_role(self):
        from missy.tools.builtin.atspi_tools import _find_element

        app = MagicMock()
        app.getRoleName.return_value = "frame"
        app.name = "App"
        app.childCount = 1
        btn = MagicMock()
        btn.getRoleName.return_value = "push button"
        btn.name = "OK"
        btn.childCount = 0
        app.getChildAtIndex.return_value = btn

        result = _find_element(app, name="OK", role="push button", max_depth=5)
        assert result is btn

    def test_find_not_found(self):
        from missy.tools.builtin.atspi_tools import _find_element

        app = MagicMock()
        app.getRoleName.return_value = "frame"
        app.name = "App"
        app.childCount = 0

        result = _find_element(app, name="Nonexistent", role=None, max_depth=5)
        assert result is None

    def test_find_neither_name_nor_role(self):
        """When both name and role are None, should not match anything."""
        from missy.tools.builtin.atspi_tools import _find_element

        app = MagicMock()
        app.getRoleName.return_value = "button"
        app.name = "OK"
        app.childCount = 0

        result = _find_element(app, name=None, role=None, max_depth=5)
        assert result is None


class TestAtSpiGetTreeTool:
    """Tests for AtSpiGetTreeTool."""

    def test_tool_attributes(self):
        from missy.tools.builtin.atspi_tools import AtSpiGetTreeTool

        tool = AtSpiGetTreeTool()
        assert tool.name == "atspi_get_tree"
        assert tool.permissions.shell is False

    @patch("missy.tools.builtin.atspi_tools._get_desktop")
    def test_pyatspi_not_installed(self, mock_desktop):
        mock_desktop.side_effect = ImportError("pyatspi not available")
        from missy.tools.builtin.atspi_tools import AtSpiGetTreeTool

        tool = AtSpiGetTreeTool()
        result = tool.execute()
        assert result.success is False
        assert "pyatspi" in result.error.lower() or "not installed" in result.error.lower()


class TestAtSpiClickTool:
    """Tests for AtSpiClickTool."""

    def test_tool_attributes(self):
        from missy.tools.builtin.atspi_tools import AtSpiClickTool

        tool = AtSpiClickTool()
        assert tool.name == "atspi_click"


class TestAtSpiGetTextTool:
    """Tests for AtSpiGetTextTool."""

    def test_tool_attributes(self):
        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool

        tool = AtSpiGetTextTool()
        assert tool.name == "atspi_get_text"


class TestAtSpiSetValueTool:
    """Tests for AtSpiSetValueTool."""

    def test_tool_attributes(self):
        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool

        tool = AtSpiSetValueTool()
        assert tool.name == "atspi_set_value"


# =========================================================================
# Audio tool tests
# =========================================================================


class TestAudioListDevicesTool:
    """Tests for AudioListDevicesTool."""

    def test_tool_attributes(self):
        from missy.tools.builtin.tts_speak import AudioListDevicesTool

        tool = AudioListDevicesTool()
        assert tool.name == "audio_list_devices"


class TestSafeTTSEnvVars:
    """Environment variable safety for TTS."""

    def test_safe_vars_include_audio(self):
        from missy.tools.builtin.tts_speak import _SAFE_TTS_ENV_VARS

        assert "PULSE_SERVER" in _SAFE_TTS_ENV_VARS
        assert "PIPEWIRE_REMOTE" in _SAFE_TTS_ENV_VARS
        assert "XDG_RUNTIME_DIR" in _SAFE_TTS_ENV_VARS

    def test_safe_vars_exclude_secrets(self):
        from missy.tools.builtin.tts_speak import _SAFE_TTS_ENV_VARS

        assert "ANTHROPIC_API_KEY" not in _SAFE_TTS_ENV_VARS
        assert "OPENAI_API_KEY" not in _SAFE_TTS_ENV_VARS
        assert "AWS_SECRET_ACCESS_KEY" not in _SAFE_TTS_ENV_VARS
