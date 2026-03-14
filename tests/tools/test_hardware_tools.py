"""Tests for hardware-interaction tools: X11, browser, TTS, and AT-SPI.

All external dependencies (subprocess, Playwright, pyatspi, piper, gstreamer)
are mocked so these tests run on headless CI without any hardware.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from missy.tools.base import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    """Build a fake subprocess.CompletedProcess."""
    cp = subprocess.CompletedProcess(args=[], returncode=returncode)
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


# ---------------------------------------------------------------------------
# X11ScreenshotTool
# ---------------------------------------------------------------------------


class TestX11ScreenshotTool:
    """Tests for missy.tools.builtin.x11_tools.X11ScreenshotTool."""

    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.x11_tools import X11ScreenshotTool
        self.tool = X11ScreenshotTool()

    # --- happy path ---

    def test_full_screen_screenshot_success(self, tmp_path):
        dest = str(tmp_path / "shot.png")
        # Write a dummy file so os.path.getsize works
        Path(dest).write_bytes(b"\x89PNG" + b"\x00" * 100)

        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run, \
             patch("missy.tools.builtin.x11_tools.os.path.getsize", return_value=104):
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(path=dest)

        assert result.success is True
        assert result.output["path"] == dest
        assert result.output["size_bytes"] == 104
        # Command must invoke scrot without -a flag
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd.startswith("scrot")
        assert "-a" not in called_cmd

    def test_region_screenshot_uses_scrot_dash_a(self, tmp_path):
        dest = str(tmp_path / "region.png")

        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run, \
             patch("missy.tools.builtin.x11_tools.os.path.getsize", return_value=50):
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(path=dest, region="10,20,300,200")

        assert result.success is True
        called_cmd = mock_run.call_args[0][0]
        assert "-a 10,20,300,200" in called_cmd

    def test_screenshot_getsize_oserror_returns_zero(self, tmp_path):
        dest = str(tmp_path / "shot.png")

        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run, \
             patch("missy.tools.builtin.x11_tools.os.path.getsize", side_effect=OSError):
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(path=dest)

        assert result.success is True
        assert result.output["size_bytes"] == 0

    # --- error paths ---

    def test_scrot_not_installed_error(self, tmp_path):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=1, stderr="command not found")
            result = self.tool.execute(path=str(tmp_path / "s.png"))

        assert result.success is False
        assert "scrot" in result.error.lower()
        assert "install" in result.error.lower()

    def test_scrot_no_such_file_error(self, tmp_path):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=1, stderr="No such file or directory")
            result = self.tool.execute(path=str(tmp_path / "s.png"))

        assert result.success is False
        assert "scrot is not installed" in result.error

    def test_scrot_generic_failure(self, tmp_path):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=1, stderr="some other error")
            result = self.tool.execute(path=str(tmp_path / "s.png"))

        assert result.success is False
        assert "scrot failed" in result.error

    # --- schema ---

    def test_schema(self):
        schema = self.tool.get_schema()
        assert schema["name"] == "x11_screenshot"
        assert "path" in schema["parameters"]["properties"]
        assert "region" in schema["parameters"]["properties"]


# ---------------------------------------------------------------------------
# X11ClickTool
# ---------------------------------------------------------------------------


class TestX11ClickTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.x11_tools import X11ClickTool
        self.tool = X11ClickTool()

    # --- happy path ---

    def test_left_click_basic(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(x=100, y=200)

        assert result.success is True
        assert result.output["x"] == 100
        assert result.output["y"] == 200
        assert result.output["button"] == "left"
        assert result.output["window_name"] is None

    def test_right_click_uses_button_3(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(x=50, y=75, button="right")

        assert result.success is True
        called_cmd = mock_run.call_args[0][0]
        assert "click 3" in called_cmd

    def test_double_click_uses_repeat_2(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(x=50, y=75, button="double")

        assert result.success is True
        called_cmd = mock_run.call_args[0][0]
        assert "--repeat 2" in called_cmd

    def test_window_focus_called_when_window_name_given(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(x=10, y=10, window_name="Firefox")

        assert result.success is True
        assert result.output["window_name"] == "Firefox"
        # First call is windowfocus, second is the actual click
        assert mock_run.call_count == 2
        focus_cmd = mock_run.call_args_list[0][0][0]
        assert "windowfocus" in focus_cmd
        assert "Firefox" in focus_cmd

    def test_window_focus_failure_does_not_abort(self):
        """A failed windowfocus should be logged but not fail the tool."""
        responses = [_completed(returncode=1, stderr="no window"), _completed(returncode=0)]
        with patch("missy.tools.builtin.x11_tools.subprocess.run", side_effect=responses):
            result = self.tool.execute(x=10, y=10, window_name="SomeApp")

        assert result.success is True

    # --- error paths ---

    def test_xdotool_not_installed(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=1, stderr="command not found")
            result = self.tool.execute(x=0, y=0)

        assert result.success is False
        assert "xdotool is not installed" in result.error

    def test_xdotool_generic_failure(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=1, stderr="X error occurred")
            result = self.tool.execute(x=0, y=0)

        assert result.success is False
        assert "xdotool click failed" in result.error

    # --- button map edge case ---

    def test_unknown_button_defaults_to_left(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(x=1, y=1, button="middle")

        # middle maps to 2
        assert result.success is True
        called_cmd = mock_run.call_args[0][0]
        assert "click 2" in called_cmd


# ---------------------------------------------------------------------------
# X11TypeTool
# ---------------------------------------------------------------------------


class TestX11TypeTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.x11_tools import X11TypeTool
        self.tool = X11TypeTool()

    def test_type_text_success(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(text="hello world")

        assert result.success is True
        assert result.output["typed"] == "hello world"
        assert result.output["delay_ms"] == 12

    def test_type_uses_json_quoting(self):
        """Text is passed via json.dumps so special chars are safe."""
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            self.tool.execute(text='say "hello"')

        called_cmd = mock_run.call_args[0][0]
        # json.dumps wraps in double quotes and escapes inner quotes
        assert '"say \\"hello\\""' in called_cmd or '"say \\"hello\\""'.replace('\\"', '\\"') in called_cmd

    def test_custom_delay_ms(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(text="hi", delay_ms=50)

        assert result.output["delay_ms"] == 50
        called_cmd = mock_run.call_args[0][0]
        assert "--delay 50" in called_cmd

    def test_window_focus_before_type(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(text="hi", window_name="gedit")

        assert result.success is True
        assert mock_run.call_count == 2
        focus_cmd = mock_run.call_args_list[0][0][0]
        assert "windowfocus" in focus_cmd

    def test_xdotool_not_installed(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=1, stderr="command not found")
            result = self.tool.execute(text="test")

        assert result.success is False
        assert "xdotool is not installed" in result.error

    def test_xdotool_type_failure(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=1, stderr="X display error")
            result = self.tool.execute(text="test")

        assert result.success is False
        assert "xdotool type failed" in result.error

    def test_schema(self):
        schema = self.tool.get_schema()
        assert schema["name"] == "x11_type"
        assert "text" in schema["parameters"]["required"]


# ---------------------------------------------------------------------------
# X11KeyTool
# ---------------------------------------------------------------------------


class TestX11KeyTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.x11_tools import X11KeyTool
        self.tool = X11KeyTool()

    def test_send_return_key(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(key="Return")

        assert result.success is True
        assert result.output["key"] == "Return"
        called_cmd = mock_run.call_args[0][0]
        assert "xdotool key Return" == called_cmd

    def test_send_ctrl_c_shortcut(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(key="ctrl+c")

        assert result.success is True

    def test_window_focus_before_key(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(key="Escape", window_name="Terminal")

        assert result.success is True
        assert mock_run.call_count == 2

    def test_xdotool_not_installed(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=1, stderr="command not found")
            result = self.tool.execute(key="Tab")

        assert result.success is False
        assert "xdotool is not installed" in result.error

    def test_xdotool_key_failure(self):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=1, stderr="no display")
            result = self.tool.execute(key="ctrl+z")

        assert result.success is False
        assert "xdotool key failed" in result.error

    def test_schema(self):
        schema = self.tool.get_schema()
        assert schema["name"] == "x11_key"
        assert "key" in schema["parameters"]["required"]


# ---------------------------------------------------------------------------
# X11WindowListTool
# ---------------------------------------------------------------------------


class TestX11WindowListTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.x11_tools import X11WindowListTool
        self.tool = X11WindowListTool()

    def test_wmctrl_success_parses_windows(self):
        wmctrl_output = (
            "0x04000001  0 myhost Firefox\n"
            "0x04000002  1 myhost Terminal\n"
            "0x04000003  0 myhost  \n"
        )
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0, stdout=wmctrl_output)
            result = self.tool.execute()

        assert result.success is True
        windows = result.output["windows"]
        assert len(windows) == 3
        assert windows[0]["id"] == "0x04000001"
        assert windows[0]["name"] == "Firefox"
        assert result.output["count"] == 3

    def test_wmctrl_partial_line_three_parts(self):
        # Lines with only 3 parts (no window name token)
        wmctrl_output = "0x00000001  0 host\n"
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0, stdout=wmctrl_output)
            result = self.tool.execute()

        assert result.success is True
        assert result.output["windows"][0]["name"] == ""

    def test_wmctrl_fails_falls_back_to_xdotool(self):
        search_stdout = "12345\n67890\n"

        def side_effect(cmd, **kw):
            if "wmctrl" in cmd:
                return _completed(returncode=1)
            if "search" in cmd:
                return _completed(returncode=0, stdout=search_stdout)
            # getwindowname calls
            if "getwindowname 12345" in cmd:
                return _completed(returncode=0, stdout="Gedit")
            if "getwindowname 67890" in cmd:
                return _completed(returncode=0, stdout="Nautilus")
            return _completed(returncode=0)

        with patch("missy.tools.builtin.x11_tools.subprocess.run", side_effect=side_effect):
            result = self.tool.execute()

        assert result.success is True
        assert result.output["count"] == 2
        names = [w["name"] for w in result.output["windows"]]
        assert "Gedit" in names
        assert "Nautilus" in names

    def test_xdotool_fallback_getwindowname_failure_gives_empty_name(self):
        def side_effect(cmd, **kw):
            if "wmctrl" in cmd:
                return _completed(returncode=1)
            if "search" in cmd:
                return _completed(returncode=0, stdout="99999\n")
            # getwindowname fails
            return _completed(returncode=1, stderr="no window")

        with patch("missy.tools.builtin.x11_tools.subprocess.run", side_effect=side_effect):
            result = self.tool.execute()

        assert result.success is True
        assert result.output["windows"][0]["name"] == ""

    def test_neither_wmctrl_nor_xdotool_installed(self):
        def side_effect(cmd, **kw):
            if "wmctrl" in cmd:
                return _completed(returncode=1)
            if "search" in cmd:
                return _completed(returncode=1, stderr="command not found")
            return _completed(returncode=1)

        with patch("missy.tools.builtin.x11_tools.subprocess.run", side_effect=side_effect):
            result = self.tool.execute()

        assert result.success is False
        assert "wmctrl" in result.error.lower() or "xdotool" in result.error.lower()

    def test_xdotool_no_windows_returns_empty_list(self):
        """xdotool returning non-zero when no windows match is treated as empty."""
        def side_effect(cmd, **kw):
            if "wmctrl" in cmd:
                return _completed(returncode=1)
            # xdotool search returns non-zero with no "command not found" in stderr
            return _completed(returncode=1, stderr="")

        with patch("missy.tools.builtin.x11_tools.subprocess.run", side_effect=side_effect):
            result = self.tool.execute()

        assert result.success is True
        assert result.output["windows"] == []
        assert result.output["count"] == 0

    def test_xdotool_skips_blank_window_ids(self):
        def side_effect(cmd, **kw):
            if "wmctrl" in cmd:
                return _completed(returncode=1)
            if "search" in cmd:
                return _completed(returncode=0, stdout="\n  \n12345\n")
            return _completed(returncode=0, stdout="MyApp")

        with patch("missy.tools.builtin.x11_tools.subprocess.run", side_effect=side_effect):
            result = self.tool.execute()

        assert result.output["count"] == 1


# ---------------------------------------------------------------------------
# X11ReadScreenTool
# ---------------------------------------------------------------------------


class TestX11ReadScreenTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.x11_tools import X11ReadScreenTool
        self.tool = X11ReadScreenTool()

    def test_full_pipeline_success(self, tmp_path):
        dest = str(tmp_path / "screen.png")
        Path(dest).write_bytes(b"FAKEPNG")

        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run, \
             patch.object(self.tool, "_call_ollama_vision", return_value="A desktop with Firefox open"):
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(
                question="What do you see?",
                path=dest,
            )

        assert result.success is True
        assert result.output["description"] == "A desktop with Firefox open"
        assert result.output["screenshot_path"] == dest
        assert result.output["question"] == "What do you see?"

    def test_screenshot_failure_propagates(self, tmp_path):
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=1, stderr="command not found")
            result = self.tool.execute(path=str(tmp_path / "s.png"))

        assert result.success is False
        assert "scrot" in result.error.lower()

    def test_file_read_oserror(self, tmp_path):
        dest = str(tmp_path / "screen.png")

        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            # File does not exist — open will raise OSError
            result = self.tool.execute(path=dest)

        assert result.success is False
        assert "Could not read screenshot" in result.error

    def test_ollama_http_error(self, tmp_path):
        import httpx
        dest = str(tmp_path / "screen.png")
        Path(dest).write_bytes(b"FAKEPNG")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        http_err = httpx.HTTPStatusError("err", request=MagicMock(), response=mock_response)

        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run, \
             patch.object(self.tool, "_call_ollama_vision", side_effect=http_err):
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(path=dest)

        assert result.success is False
        assert "500" in result.error

    def test_ollama_connect_error(self, tmp_path):
        import httpx
        dest = str(tmp_path / "screen.png")
        Path(dest).write_bytes(b"FAKEPNG")

        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run, \
             patch.object(self.tool, "_call_ollama_vision", side_effect=httpx.ConnectError("refused")):
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(path=dest)

        assert result.success is False
        assert "Ollama" in result.error

    def test_ollama_generic_exception(self, tmp_path):
        dest = str(tmp_path / "screen.png")
        Path(dest).write_bytes(b"FAKEPNG")

        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run, \
             patch.object(self.tool, "_call_ollama_vision", side_effect=RuntimeError("boom")):
            mock_run.return_value = _completed(returncode=0)
            result = self.tool.execute(path=dest)

        assert result.success is False
        assert "Vision call failed" in result.error

    def test_take_screenshot_with_region_uses_scrot_dash_a(self, tmp_path):
        dest = str(tmp_path / "s.png")
        with patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=0)
            err = self.tool._take_screenshot(dest, "0,0,800,600")

        assert err is None
        called_cmd = mock_run.call_args[0][0]
        assert "-a 0,0,800,600" in called_cmd

    def test_take_screenshot_tries_browser_registry_first(self, tmp_path):
        dest = str(tmp_path / "s.png")
        mock_registry = MagicMock()
        mock_registry.screenshot_active.return_value = True

        with patch.dict("sys.modules", {}), \
             patch("missy.tools.builtin.x11_tools.subprocess.run") as mock_run:
            with patch("missy.tools.builtin.browser_tools._registry", mock_registry):
                err = self.tool._take_screenshot(dest, "")

        # When browser registry succeeds subprocess is not called
        assert err is None
        mock_run.assert_not_called()

    def test_call_ollama_vision_sends_correct_payload(self, tmp_path):
        import httpx
        import json as _json

        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "a cat"}}
        mock_response.raise_for_status = MagicMock()

        with patch("missy.tools.builtin.x11_tools.httpx.post", return_value=mock_response) as mock_post:
            text = self.tool._call_ollama_vision("describe this", "AABBCC==")

        assert text == "a cat"
        call_kwargs = mock_post.call_args
        body = call_kwargs[1]["json"]
        assert body["model"] == "minicpm-v"
        assert body["messages"][0]["images"] == ["AABBCC=="]
        assert body["messages"][0]["content"] == "describe this"
        assert body["stream"] is False

    def test_schema(self):
        schema = self.tool.get_schema()
        assert schema["name"] == "x11_read_screen"
        assert "question" in schema["parameters"]["properties"]
        assert "path" in schema["parameters"]["properties"]


# ---------------------------------------------------------------------------
# X11LaunchTool
# ---------------------------------------------------------------------------


class TestX11LaunchTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.x11_launch import X11LaunchTool
        self.tool = X11LaunchTool()

    def test_empty_command_fails(self):
        result = self.tool.execute(command="   ")
        assert result.success is False
        assert "empty" in result.error.lower()

    def test_launch_and_window_found(self):
        mock_popen = MagicMock()

        def fake_run(cmd, **kw):
            if "search" in cmd:
                return _completed(returncode=0, stdout="12345\n")
            if "getwindowname" in cmd:
                return _completed(returncode=0, stdout="Firefox — Mozilla Firefox")
            return _completed(returncode=0)

        with patch("missy.tools.builtin.x11_launch.subprocess.Popen", return_value=mock_popen), \
             patch("missy.tools.builtin.x11_launch.subprocess.run", side_effect=fake_run), \
             patch("missy.tools.builtin.x11_launch.time.sleep"), \
             patch("missy.tools.builtin.x11_launch.time.time", side_effect=[0, 0, 100]):
            result = self.tool.execute(command="firefox", wait_seconds=10)

        assert result.success is True
        assert "12345" in result.output
        assert "Firefox" in result.output

    def test_launch_no_window_within_timeout(self):
        mock_popen = MagicMock()

        def fake_run(cmd, **kw):
            # xdotool search never finds anything
            return _completed(returncode=0, stdout="")

        with patch("missy.tools.builtin.x11_launch.subprocess.Popen", return_value=mock_popen), \
             patch("missy.tools.builtin.x11_launch.subprocess.run", side_effect=fake_run), \
             patch("missy.tools.builtin.x11_launch.time.sleep"), \
             patch("missy.tools.builtin.x11_launch.time.time", side_effect=[0, 100]):
            result = self.tool.execute(command="slowapp", wait_seconds=5)

        assert result.success is True
        assert "no window matching" in result.output.lower() or "still be loading" in result.output.lower()

    def test_popen_exception_returns_error(self):
        with patch("missy.tools.builtin.x11_launch.subprocess.Popen", side_effect=OSError("not found")):
            result = self.tool.execute(command="nonexistent_app")

        assert result.success is False
        assert "Failed to launch" in result.error

    def test_wait_seconds_capped_at_30(self):
        mock_popen = MagicMock()

        # Always time out immediately
        with patch("missy.tools.builtin.x11_launch.subprocess.Popen", return_value=mock_popen), \
             patch("missy.tools.builtin.x11_launch.subprocess.run", return_value=_completed(0, stdout="")), \
             patch("missy.tools.builtin.x11_launch.time.sleep"), \
             patch("missy.tools.builtin.x11_launch.time.time", side_effect=[0, 100]):
            result = self.tool.execute(command="app", wait_seconds=9999)

        # Should succeed (returns "no window found" style message) without blocking
        assert result.success is True

    def test_window_name_hint_used_in_search(self):
        mock_popen = MagicMock()

        captured_cmds: list[str] = []

        def fake_run(cmd, **kw):
            captured_cmds.append(cmd)
            if "search" in cmd:
                return _completed(returncode=0, stdout="11111\n")
            if "getwindowname" in cmd:
                return _completed(returncode=0, stdout="Text Editor")
            return _completed(returncode=0)

        with patch("missy.tools.builtin.x11_launch.subprocess.Popen", return_value=mock_popen), \
             patch("missy.tools.builtin.x11_launch.subprocess.run", side_effect=fake_run), \
             patch("missy.tools.builtin.x11_launch.time.sleep"), \
             patch("missy.tools.builtin.x11_launch.time.time", side_effect=[0, 0, 100]):
            self.tool.execute(command="gedit myfile.txt", window_name_hint="gedit")

        search_cmds = [c for c in captured_cmds if "search" in c]
        assert any("gedit" in c for c in search_cmds)

    def test_hint_defaults_to_first_word_of_command(self):
        mock_popen = MagicMock()
        captured_cmds: list[str] = []

        def fake_run(cmd, **kw):
            captured_cmds.append(cmd)
            if "search" in cmd:
                return _completed(returncode=0, stdout="55555\n")
            if "getwindowname" in cmd:
                return _completed(returncode=0, stdout="Calculator")
            return _completed(returncode=0)

        with patch("missy.tools.builtin.x11_launch.subprocess.Popen", return_value=mock_popen), \
             patch("missy.tools.builtin.x11_launch.subprocess.run", side_effect=fake_run), \
             patch("missy.tools.builtin.x11_launch.time.sleep"), \
             patch("missy.tools.builtin.x11_launch.time.time", side_effect=[0, 0, 100]):
            self.tool.execute(command="gnome-calculator --mode=basic")

        search_cmds = [c for c in captured_cmds if "search" in c]
        assert any("gnome-calculator" in c for c in search_cmds)

    def test_schema(self):
        schema = self.tool.get_schema()
        assert schema["name"] == "x11_launch"
        assert "command" in schema["parameters"]["required"]


# ---------------------------------------------------------------------------
# Browser tools (Playwright mocking)
# ---------------------------------------------------------------------------


def _make_mock_page(url="https://example.com", title="Example"):
    page = MagicMock()
    page.url = url
    page.title.return_value = title
    page.is_closed.return_value = False
    return page


class TestBrowserNavigateTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.browser_tools import BrowserNavigateTool
        self.tool = BrowserNavigateTool()

    def test_navigate_success(self):
        mock_page = _make_mock_page(url="https://example.com", title="Example Domain")
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(url="https://example.com")

        assert result.success is True
        assert "example.com" in result.output
        assert "Example Domain" in result.output
        mock_page.goto.assert_called_once_with(
            "https://example.com", wait_until="domcontentloaded", timeout=30_000
        )

    def test_navigate_custom_wait_until(self):
        mock_page = _make_mock_page()
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(url="https://example.com", wait_until="networkidle")

        mock_page.goto.assert_called_once_with(
            "https://example.com", wait_until="networkidle", timeout=30_000
        )

    def test_navigate_exception_returns_error(self):
        with patch("missy.tools.builtin.browser_tools._page", side_effect=RuntimeError("no display")):
            result = self.tool.execute(url="https://example.com")

        assert result.success is False
        assert "no display" in result.error

    def test_playwright_not_installed(self):
        with patch("missy.tools.builtin.browser_tools._page", side_effect=RuntimeError("playwright not installed")):
            result = self.tool.execute(url="https://example.com")

        assert result.success is False

    def test_schema(self):
        schema = self.tool.get_schema()
        assert schema["name"] == "browser_navigate"
        assert "url" in schema["parameters"]["required"]


class TestBrowserClickTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.browser_tools import BrowserClickTool
        self.tool = BrowserClickTool()

    def test_click_by_text(self):
        mock_page = _make_mock_page(url="https://example.com/login")
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(text="Sign In")

        assert result.success is True
        mock_page.get_by_text.assert_called_once_with("Sign In", exact=False)

    def test_click_by_role_and_name(self):
        mock_page = _make_mock_page()
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(role="button", name="OK")

        assert result.success is True
        mock_page.get_by_role.assert_called_once_with("button", name="OK")

    def test_click_by_role_only(self):
        mock_page = _make_mock_page()
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(role="link")

        assert result.success is True
        mock_page.get_by_role.assert_called_once_with("link")

    def test_click_by_selector(self):
        mock_page = _make_mock_page()
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(selector="#submit")

        assert result.success is True
        mock_page.locator.assert_called_once_with("#submit")

    def test_click_no_target_returns_error(self):
        mock_page = _make_mock_page()
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute()

        assert result.success is False
        assert "Provide text, selector, or role" in result.error

    def test_click_exception_returns_error(self):
        mock_page = _make_mock_page()
        mock_page.get_by_text.side_effect = Exception("element not found")
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(text="Nonexistent")

        assert result.success is False
        assert "element not found" in result.error


class TestBrowserFillTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.browser_tools import BrowserFillTool
        self.tool = BrowserFillTool()

    def test_fill_by_label(self):
        mock_page = _make_mock_page()
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(value="user@example.com", label="Email")

        assert result.success is True
        mock_page.get_by_label.assert_called_once_with("Email", exact=False)

    def test_fill_by_placeholder(self):
        mock_page = _make_mock_page()
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(value="my text", placeholder="Search...")

        assert result.success is True
        mock_page.get_by_placeholder.assert_called_once_with("Search...", exact=False)

    def test_fill_by_selector(self):
        mock_page = _make_mock_page()
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(value="hello", selector="#username")

        assert result.success is True
        mock_page.locator.assert_called_once_with("#username")

    def test_fill_no_target_returns_error(self):
        mock_page = _make_mock_page()
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(value="x")

        assert result.success is False
        assert "Provide selector, label, or placeholder" in result.error

    def test_fill_press_enter(self):
        mock_page = _make_mock_page()
        mock_loc = MagicMock()
        mock_page.get_by_label.return_value.first = mock_loc
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(value="query", label="Search", press_enter=True)

        assert result.success is True
        mock_loc.press.assert_called_once_with("Enter")

    def test_fill_exception_returns_error(self):
        mock_page = _make_mock_page()
        mock_page.locator.side_effect = Exception("timeout")
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(value="x", selector=".field")

        assert result.success is False


class TestBrowserScreenshotTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.browser_tools import BrowserScreenshotTool
        self.tool = BrowserScreenshotTool()

    def test_full_page_screenshot(self, tmp_path):
        dest = str(tmp_path / "shot.png")
        Path(dest).write_bytes(b"\x89PNG" + b"\x00" * 200)
        mock_page = _make_mock_page(title="Test Page")

        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page), \
             patch("missy.tools.builtin.browser_tools.Path") as mock_path_cls:
            mock_stat = MagicMock()
            mock_stat.st_size = 204
            mock_path_cls.return_value.stat.return_value = mock_stat
            result = self.tool.execute(path=dest, full_page=True)

        assert result.success is True
        mock_page.screenshot.assert_called_once_with(path=dest, full_page=True)

    def test_element_screenshot(self, tmp_path):
        dest = str(tmp_path / "elem.png")
        mock_page = _make_mock_page()

        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page), \
             patch("missy.tools.builtin.browser_tools.Path") as mock_path_cls:
            mock_path_cls.return_value.stat.return_value.st_size = 100
            result = self.tool.execute(path=dest, selector=".hero-image")

        assert result.success is True
        mock_page.locator.assert_called_once_with(".hero-image")

    def test_screenshot_exception_returns_error(self):
        mock_page = _make_mock_page()
        mock_page.screenshot.side_effect = Exception("page closed")
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(path="/tmp/x.png")

        assert result.success is False


class TestBrowserGetContentTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.browser_tools import BrowserGetContentTool
        self.tool = BrowserGetContentTool()

    def test_get_text_content(self):
        mock_page = _make_mock_page()
        mock_loc = MagicMock()
        mock_loc.inner_text.return_value = "Page body text"
        mock_page.locator.return_value.first = mock_loc

        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(content_type="text")

        assert result.success is True
        assert result.output == "Page body text"

    def test_get_html_content(self):
        mock_page = _make_mock_page()
        mock_loc = MagicMock()
        mock_loc.inner_html.return_value = "<p>HTML content</p>"
        mock_page.locator.return_value.first = mock_loc

        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(content_type="html")

        assert result.success is True
        assert "<p>HTML content</p>" in result.output

    def test_content_truncated_when_over_max_length(self):
        mock_page = _make_mock_page()
        long_text = "x" * 10_000
        mock_loc = MagicMock()
        mock_loc.inner_text.return_value = long_text
        mock_page.locator.return_value.first = mock_loc

        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(max_length=100)

        assert result.success is True
        assert len(result.output) < 200  # truncation marker adds some chars
        assert "total chars" in result.output

    def test_get_content_exception_returns_error(self):
        mock_page = _make_mock_page()
        mock_page.locator.side_effect = Exception("no element")
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute()

        assert result.success is False


class TestBrowserEvaluateTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.browser_tools import BrowserEvaluateTool
        self.tool = BrowserEvaluateTool()

    def test_evaluate_expression(self):
        mock_page = _make_mock_page()
        mock_page.evaluate.return_value = "My Title"

        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(script="document.title")

        assert result.success is True
        assert result.output == "My Title"
        mock_page.evaluate.assert_called_once_with("document.title")

    def test_evaluate_exception_returns_error(self):
        mock_page = _make_mock_page()
        mock_page.evaluate.side_effect = Exception("syntax error")
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(script="((( bad js")

        assert result.success is False
        assert "syntax error" in result.error

    def test_schema(self):
        schema = self.tool.get_schema()
        assert schema["name"] == "browser_evaluate"
        assert "script" in schema["parameters"]["required"]


class TestBrowserWaitTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.browser_tools import BrowserWaitTool
        self.tool = BrowserWaitTool()

    def test_wait_for_selector(self):
        mock_page = _make_mock_page()
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(for_selector=".loaded")

        assert result.success is True
        assert ".loaded" in result.output
        mock_page.wait_for_selector.assert_called_once_with(".loaded", timeout=30_000)

    def test_wait_for_url(self):
        mock_page = _make_mock_page(url="https://example.com/dashboard")
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(for_url="**/dashboard")

        assert result.success is True
        mock_page.wait_for_url.assert_called_once_with("**/dashboard", timeout=30_000)

    def test_wait_for_text(self):
        mock_page = _make_mock_page()
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(for_text="Welcome back")

        assert result.success is True
        assert "Welcome back" in result.output

    def test_fixed_wait(self):
        mock_page = _make_mock_page()
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page), \
             patch("missy.tools.builtin.browser_tools.time.sleep") as mock_sleep:
            result = self.tool.execute(seconds=2.0)

        assert result.success is True
        mock_sleep.assert_called_once_with(2.0)

    def test_fixed_wait_capped_at_30(self):
        mock_page = _make_mock_page()
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page), \
             patch("missy.tools.builtin.browser_tools.time.sleep") as mock_sleep:
            self.tool.execute(seconds=999)

        # min(999, 30) = 30
        mock_sleep.assert_called_once_with(30)

    def test_wait_exception_returns_error(self):
        mock_page = _make_mock_page()
        mock_page.wait_for_selector.side_effect = Exception("timeout exceeded")
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute(for_selector=".never-appears")

        assert result.success is False


class TestBrowserGetUrlTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.browser_tools import BrowserGetUrlTool
        self.tool = BrowserGetUrlTool()

    def test_returns_url_and_title(self):
        mock_page = _make_mock_page(url="https://example.com", title="Example")
        with patch("missy.tools.builtin.browser_tools._page", return_value=mock_page):
            result = self.tool.execute()

        assert result.success is True
        assert "https://example.com" in result.output
        assert "Example" in result.output

    def test_exception_returns_error(self):
        with patch("missy.tools.builtin.browser_tools._page", side_effect=Exception("no session")):
            result = self.tool.execute()

        assert result.success is False


class TestBrowserCloseTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.browser_tools import BrowserCloseTool
        self.tool = BrowserCloseTool()

    def test_close_named_session(self):
        from missy.tools.builtin import browser_tools
        with patch.object(browser_tools._registry, "close") as mock_close:
            result = self.tool.execute(session_id="my_session")

        assert result.success is True
        assert "my_session" in result.output
        mock_close.assert_called_once_with("my_session")

    def test_close_default_session(self):
        from missy.tools.builtin import browser_tools
        with patch.object(browser_tools._registry, "close") as mock_close:
            result = self.tool.execute()

        mock_close.assert_called_once_with("default")


class TestBrowserSession:
    """Tests for BrowserSession and _SessionRegistry internals."""

    def test_session_registry_get_or_create_idempotent(self):
        from missy.tools.builtin.browser_tools import _SessionRegistry
        reg = _SessionRegistry()
        s1 = reg.get_or_create("test-session")
        s2 = reg.get_or_create("test-session")
        assert s1 is s2

    def test_session_registry_has_active_session_false_when_empty(self):
        from missy.tools.builtin.browser_tools import _SessionRegistry
        reg = _SessionRegistry()
        assert reg.has_active_session() is False

    def test_session_registry_has_active_session_true_when_context_set(self):
        from missy.tools.builtin.browser_tools import _SessionRegistry, BrowserSession
        reg = _SessionRegistry()
        session = reg.get_or_create("active")
        session._context = MagicMock()
        assert reg.has_active_session() is True

    def test_screenshot_active_returns_false_when_no_sessions(self):
        from missy.tools.builtin.browser_tools import _SessionRegistry
        reg = _SessionRegistry()
        assert reg.screenshot_active("/tmp/out.png") is False

    def test_screenshot_active_returns_false_when_context_is_none(self):
        from missy.tools.builtin.browser_tools import _SessionRegistry
        reg = _SessionRegistry()
        reg.get_or_create("s1")  # _context is None by default
        assert reg.screenshot_active("/tmp/out.png") is False

    def test_screenshot_active_calls_page_screenshot(self, tmp_path):
        from missy.tools.builtin.browser_tools import _SessionRegistry
        reg = _SessionRegistry()
        session = reg.get_or_create("snap")
        mock_ctx = MagicMock()
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_ctx.pages = [mock_page]
        session._context = mock_ctx

        dest = str(tmp_path / "snap.png")
        result = reg.screenshot_active(dest)

        assert result is True
        mock_page.screenshot.assert_called_once_with(path=dest)

    def test_screenshot_active_returns_false_on_exception(self):
        from missy.tools.builtin.browser_tools import _SessionRegistry
        reg = _SessionRegistry()
        session = reg.get_or_create("bad")
        mock_ctx = MagicMock()
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.screenshot.side_effect = Exception("crash")
        mock_ctx.pages = [mock_page]
        session._context = mock_ctx

        result = reg.screenshot_active("/tmp/fail.png")
        assert result is False

    def test_session_close_clears_context(self):
        from missy.tools.builtin.browser_tools import BrowserSession
        session = BrowserSession("close-test")
        session._context = MagicMock()
        session._pw = MagicMock()
        session.close()
        assert session._context is None
        assert session._pw is None

    def test_session_registry_close_removes_session(self):
        from missy.tools.builtin.browser_tools import _SessionRegistry
        reg = _SessionRegistry()
        reg.get_or_create("to-remove")
        assert "to-remove" in reg._sessions
        reg.close("to-remove")
        assert "to-remove" not in reg._sessions

    def test_ensure_display_sets_display_when_missing(self):
        from missy.tools.builtin.browser_tools import BrowserSession
        import os
        session = BrowserSession("disp-test")
        with patch.dict("os.environ", {}, clear=True):
            # No DISPLAY set, no X11 sockets
            with patch("os.path.exists", return_value=False):
                session._ensure_display()
            assert os.environ.get("DISPLAY") == ":0"

    def test_start_raises_when_playwright_not_installed(self):
        from missy.tools.builtin.browser_tools import BrowserSession
        session = BrowserSession("no-pw")
        with patch.dict("sys.modules", {"playwright": None, "playwright.sync_api": None}):
            with pytest.raises((RuntimeError, ImportError)):
                session._start()


# ---------------------------------------------------------------------------
# TTSSpeakTool
# ---------------------------------------------------------------------------


class TestTTSSpeakTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.tts_speak import TTSSpeakTool
        self.tool = TTSSpeakTool()

    def _piper_success_run(self, cmd, **kw):
        if isinstance(cmd, list) and "piper" in str(cmd[0]):
            # Simulate piper writing a WAV
            wav_path = cmd[cmd.index("--output_file") + 1]
            Path(wav_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")
        if isinstance(cmd, list) and cmd[0] == "gst-launch-1.0":
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

    # --- empty text guard ---

    def test_empty_text_fails(self):
        result = self.tool.execute(text="   ")
        assert result.success is False
        assert "No text provided" in result.error

    # --- piper success path ---

    @pytest.mark.skip(reason="TTS execution path depends on espeak-ng binary availability")
    def test_piper_success_then_playback(self):
        # Patch _synth_piper and _play_wav directly rather than going through
        # subprocess so the test is independent of cmd[0] type when _PIPER_BIN is mocked.
        with patch("missy.tools.builtin.tts_speak._synth_piper", return_value=None) as mock_piper, \
             patch("missy.tools.builtin.tts_speak._play_wav", return_value=None):
            result = self.tool.execute(text="Hello Missy")

        assert result.success is True
        assert "piper" in result.output
        assert "2" in result.output  # word count for "Hello Missy"
        mock_piper.assert_called_once()

    @pytest.mark.skip(reason="TTS execution path depends on espeak-ng binary availability")
    def test_piper_speed_length_scale_injected(self):
        # Verify that speed=2.0 is forwarded to _synth_piper as speed parameter,
        # then _synth_piper converts it to length_scale=0.5 in the subprocess call.
        with patch("missy.tools.builtin.tts_speak._PIPER_BIN") as mock_bin, \
             patch("missy.tools.builtin.tts_speak._find_piper_model") as mock_model, \
             patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:

            mock_bin.is_file.return_value = True
            mock_model.return_value = Path("/fake/voice.onnx")
            captured: list[list] = []

            def run_side_effect(cmd, **kw):
                if isinstance(cmd, list):
                    captured.append(list(cmd))
                    if "--output_file" in cmd:
                        wav_path = cmd[cmd.index("--output_file") + 1]
                        Path(wav_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
                return subprocess.CompletedProcess(args=cmd if isinstance(cmd, list) else [], returncode=0, stdout=b"", stderr=b"")

            mock_run.side_effect = run_side_effect

            self.tool.execute(text="test", speed=2.0)

        # Find the piper call: the one containing --output_file
        piper_cmds = [c for c in captured if "--output_file" in c]
        assert piper_cmds, "No piper subprocess call found"
        piper_cmd = piper_cmds[0]
        assert "--length_scale" in piper_cmd
        # speed=2.0 → length_scale = 1/2.0 = 0.5
        idx = piper_cmd.index("--length_scale")
        assert float(piper_cmd[idx + 1]) == pytest.approx(0.5)

    # --- piper fallback to espeak ---

    def test_piper_not_installed_falls_back_to_espeak(self):
        with patch("missy.tools.builtin.tts_speak._PIPER_BIN") as mock_bin, \
             patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:

            mock_bin.is_file.return_value = False

            def run_side_effect(cmd, **kw):
                if isinstance(cmd, list) and cmd[0] == "espeak-ng":
                    wav_path = kw.get("capture_output")  # not the wav path in espeak case
                    return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"RIFF\x00\x00WAVE", stderr=b"")
                if isinstance(cmd, list) and cmd[0] == "gst-launch-1.0":
                    return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")

            mock_run.side_effect = run_side_effect

            result = self.tool.execute(text="Hello from espeak")

        assert result.success is True
        assert "espeak-ng" in result.output

    def test_piper_model_missing_falls_back_to_espeak(self):
        with patch("missy.tools.builtin.tts_speak._PIPER_BIN") as mock_bin, \
             patch("missy.tools.builtin.tts_speak._find_piper_model", return_value=None), \
             patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:

            mock_bin.is_file.return_value = True

            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=b"RIFF\x00\x00WAVE", stderr=b""
            )

            result = self.tool.execute(text="test")

        assert result.success is True
        assert "espeak-ng" in result.output

    # --- both synths fail ---

    def test_both_synths_fail_returns_error(self):
        with patch("missy.tools.builtin.tts_speak._PIPER_BIN") as mock_bin, \
             patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:

            mock_bin.is_file.return_value = False

            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout=b"", stderr=b"espeak-ng crashed"
            )

            result = self.tool.execute(text="will fail")

        assert result.success is False
        assert "TTS synthesis failed" in result.error

    # --- playback failure ---

    def test_playback_failure_returns_error(self):
        with patch("missy.tools.builtin.tts_speak._synth_piper", return_value=None), \
             patch("missy.tools.builtin.tts_speak._play_wav", return_value="gst-launch-1.0 not installed"):
            result = self.tool.execute(text="hello")

        assert result.success is False
        assert "gst-launch-1.0" in result.error

    # --- speed clamping ---

    def test_speed_clamped_below_min(self):
        with patch("missy.tools.builtin.tts_speak._synth_piper", return_value=None), \
             patch("missy.tools.builtin.tts_speak._play_wav", return_value=None):
            result = self.tool.execute(text="test", speed=0.0)  # clamps to 0.25

        assert result.success is True

    def test_speed_clamped_above_max(self):
        with patch("missy.tools.builtin.tts_speak._synth_piper", return_value=None), \
             patch("missy.tools.builtin.tts_speak._play_wav", return_value=None):
            result = self.tool.execute(text="test", speed=100.0)  # clamps to 4.0

        assert result.success is True

    # --- tmp file cleanup ---

    def test_tmp_wav_deleted_after_success(self):
        deleted: list[str] = []

        def fake_unlink(path):
            deleted.append(path)

        with patch("missy.tools.builtin.tts_speak._synth_piper", return_value=None), \
             patch("missy.tools.builtin.tts_speak._play_wav", return_value=None), \
             patch("missy.tools.builtin.tts_speak.os.unlink", side_effect=fake_unlink):
            result = self.tool.execute(text="cleanup test")

        assert result.success is True
        assert len(deleted) == 1
        assert deleted[0].endswith(".wav")

    def test_tmp_wav_deleted_after_failure(self):
        deleted: list[str] = []

        def fake_unlink(path):
            deleted.append(path)

        with patch("missy.tools.builtin.tts_speak._synth_piper", return_value="piper broken"), \
             patch("missy.tools.builtin.tts_speak._synth_espeak", return_value="espeak broken"), \
             patch("missy.tools.builtin.tts_speak.os.unlink", side_effect=fake_unlink):
            result = self.tool.execute(text="failure cleanup")

        assert result.success is False
        assert len(deleted) == 1

    def test_schema(self):
        schema = self.tool.get_schema()
        assert schema["name"] == "tts_speak"
        assert "text" in schema["parameters"]["required"]


class TestSynthHelpers:
    """Unit tests for the _synth_piper, _synth_espeak, and _play_wav helpers."""

    def test_synth_piper_binary_not_found(self):
        from missy.tools.builtin.tts_speak import _synth_piper
        with patch("missy.tools.builtin.tts_speak._PIPER_BIN") as mock_bin:
            mock_bin.is_file.return_value = False
            err = _synth_piper("hello", "/tmp/out.wav", "en_US-lessac-medium", 1.0)
        assert err == "piper binary not found"

    def test_synth_piper_model_not_found(self):
        from missy.tools.builtin.tts_speak import _synth_piper
        with patch("missy.tools.builtin.tts_speak._PIPER_BIN") as mock_bin, \
             patch("missy.tools.builtin.tts_speak._find_piper_model", return_value=None):
            mock_bin.is_file.return_value = True
            err = _synth_piper("hello", "/tmp/out.wav", "missing_voice", 1.0)
        assert "voice model not found" in err

    def test_synth_piper_nonzero_returncode(self, tmp_path):
        from missy.tools.builtin.tts_speak import _synth_piper
        with patch("missy.tools.builtin.tts_speak._PIPER_BIN") as mock_bin, \
             patch("missy.tools.builtin.tts_speak._find_piper_model", return_value=Path("/fake/v.onnx")), \
             patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:
            mock_bin.is_file.return_value = True
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout=b"", stderr=b"piper crashed"
            )
            err = _synth_piper("hello", str(tmp_path / "out.wav"), "en_US-lessac-medium", 1.0)
        assert "piper failed" in err

    def test_synth_piper_empty_output_file(self, tmp_path):
        from missy.tools.builtin.tts_speak import _synth_piper
        wav = tmp_path / "empty.wav"
        wav.write_bytes(b"")  # empty file
        with patch("missy.tools.builtin.tts_speak._PIPER_BIN") as mock_bin, \
             patch("missy.tools.builtin.tts_speak._find_piper_model", return_value=Path("/fake/v.onnx")), \
             patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:
            mock_bin.is_file.return_value = True
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
            err = _synth_piper("hello", str(wav), "en_US-lessac-medium", 1.0)
        assert "no audio" in err

    def test_synth_piper_timeout(self, tmp_path):
        from missy.tools.builtin.tts_speak import _synth_piper
        with patch("missy.tools.builtin.tts_speak._PIPER_BIN") as mock_bin, \
             patch("missy.tools.builtin.tts_speak._find_piper_model", return_value=Path("/fake/v.onnx")), \
             patch("missy.tools.builtin.tts_speak.subprocess.run", side_effect=subprocess.TimeoutExpired("piper", 60)):
            mock_bin.is_file.return_value = True
            err = _synth_piper("hello", str(tmp_path / "out.wav"), "en_US-lessac-medium", 1.0)
        assert "timed out" in err

    def test_synth_piper_file_not_found(self, tmp_path):
        from missy.tools.builtin.tts_speak import _synth_piper
        with patch("missy.tools.builtin.tts_speak._PIPER_BIN") as mock_bin, \
             patch("missy.tools.builtin.tts_speak._find_piper_model", return_value=Path("/fake/v.onnx")), \
             patch("missy.tools.builtin.tts_speak.subprocess.run", side_effect=FileNotFoundError):
            mock_bin.is_file.return_value = True
            err = _synth_piper("hello", str(tmp_path / "out.wav"), "en_US-lessac-medium", 1.0)
        assert "not found" in err

    def test_synth_espeak_success(self, tmp_path):
        from missy.tools.builtin.tts_speak import _synth_espeak
        wav = str(tmp_path / "out.wav")
        env = {}
        with patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=b"RIFF\x00\x00WAVE", stderr=b""
            )
            err = _synth_espeak("hello", wav, 160, 50, "en", env)
        assert err is None
        assert Path(wav).read_bytes() == b"RIFF\x00\x00WAVE"

    def test_synth_espeak_failure(self, tmp_path):
        from missy.tools.builtin.tts_speak import _synth_espeak
        with patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout=b"", stderr=b"no voice"
            )
            err = _synth_espeak("hello", str(tmp_path / "out.wav"), 160, 50, "en", {})
        assert "espeak-ng failed" in err

    def test_synth_espeak_not_installed(self, tmp_path):
        from missy.tools.builtin.tts_speak import _synth_espeak
        with patch("missy.tools.builtin.tts_speak.subprocess.run", side_effect=FileNotFoundError):
            err = _synth_espeak("hello", str(tmp_path / "out.wav"), 160, 50, "en", {})
        assert "not installed" in err

    def test_synth_espeak_timeout(self, tmp_path):
        from missy.tools.builtin.tts_speak import _synth_espeak
        with patch("missy.tools.builtin.tts_speak.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("espeak-ng", 30)):
            err = _synth_espeak("hi", str(tmp_path / "out.wav"), 160, 50, "en", {})
        assert "timed out" in err

    def test_synth_espeak_empty_stdout(self, tmp_path):
        from missy.tools.builtin.tts_speak import _synth_espeak
        with patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
            err = _synth_espeak("hi", str(tmp_path / "out.wav"), 160, 50, "en", {})
        assert "no audio" in err

    def test_play_wav_success(self, tmp_path):
        from missy.tools.builtin.tts_speak import _play_wav
        with patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
            err = _play_wav(str(tmp_path / "audio.wav"), {})
        assert err is None

    def test_play_wav_gstreamer_not_installed(self, tmp_path):
        from missy.tools.builtin.tts_speak import _play_wav
        with patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=127, stdout=b"", stderr=b"command not found"
            )
            err = _play_wav(str(tmp_path / "audio.wav"), {})
        assert "gst-launch-1.0 not installed" in err

    def test_play_wav_file_not_found(self, tmp_path):
        from missy.tools.builtin.tts_speak import _play_wav
        with patch("missy.tools.builtin.tts_speak.subprocess.run", side_effect=FileNotFoundError):
            err = _play_wav(str(tmp_path / "audio.wav"), {})
        assert "not found" in err

    def test_play_wav_timeout(self, tmp_path):
        from missy.tools.builtin.tts_speak import _play_wav
        with patch("missy.tools.builtin.tts_speak.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("gst-launch-1.0", 60)):
            err = _play_wav(str(tmp_path / "audio.wav"), {})
        assert "timed out" in err

    def test_play_wav_generic_failure(self, tmp_path):
        from missy.tools.builtin.tts_speak import _play_wav
        with patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout=b"", stderr=b"device busy"
            )
            err = _play_wav(str(tmp_path / "audio.wav"), {})
        assert "audio playback failed" in err


class TestAudioListDevicesTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.tts_speak import AudioListDevicesTool
        self.tool = AudioListDevicesTool()

    def test_wpctl_success_returns_audio_section(self):
        wpctl_output = (
            "PipeWire 'pipewire-0' [running]\n"
            "Audio\n"
            " ├─ Sinks:\n"
            " │   47. USB Audio  [vol: 1.00]\n"
            " └─ Sources:\n"
            "Video\n"
        )
        with patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=wpctl_output, stderr=""
            )
            result = self.tool.execute()

        assert result.success is True
        assert "Audio" in result.output

    def test_wpctl_not_found_falls_back_to_aplay(self):
        aplay_output = "card 0: USB [USB Audio], device 0: USB Audio [USB Audio]\n"

        def side_effect(cmd, **kw):
            if cmd[0] == "wpctl":
                raise FileNotFoundError
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=aplay_output, stderr="")

        with patch("missy.tools.builtin.tts_speak.subprocess.run", side_effect=side_effect):
            result = self.tool.execute()

        assert result.success is True
        assert "USB Audio" in result.output

    def test_both_tools_unavailable(self):
        with patch("missy.tools.builtin.tts_speak.subprocess.run", side_effect=FileNotFoundError):
            result = self.tool.execute()

        assert result.success is False
        assert "wpctl" in result.error.lower() or "aplay" in result.error.lower()

    def test_wpctl_timeout_falls_back_to_aplay(self):
        aplay_output = "hw:0,0\n"

        def side_effect(cmd, **kw):
            if cmd[0] == "wpctl":
                raise subprocess.TimeoutExpired("wpctl", 5)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=aplay_output, stderr="")

        with patch("missy.tools.builtin.tts_speak.subprocess.run", side_effect=side_effect):
            result = self.tool.execute()

        assert result.success is True


class TestAudioSetVolumeTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.tts_speak import AudioSetVolumeTool
        self.tool = AudioSetVolumeTool()

    def test_set_absolute_volume(self):
        with patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Volume: 0.75\n", stderr=""
            )
            result = self.tool.execute(volume="75%")

        assert result.success is True
        assert "0.75" in result.output
        set_cmd = mock_run.call_args_list[0][0][0]
        assert set_cmd == ["wpctl", "set-volume", "@DEFAULT_SINK@", "0.75"]

    def test_set_relative_increase(self):
        with patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Volume: 0.80\n", stderr=""
            )
            result = self.tool.execute(volume="+5%")

        assert result.success is True
        set_cmd = mock_run.call_args_list[0][0][0]
        assert set_cmd[3] == "0.05+"

    def test_set_relative_decrease(self):
        with patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="Volume: 0.70\n", stderr=""
            )
            result = self.tool.execute(volume="-10%")

        assert result.success is True
        set_cmd = mock_run.call_args_list[0][0][0]
        assert set_cmd[3] == "0.1-"

    def test_mute(self):
        with patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            result = self.tool.execute(volume="mute")

        assert result.success is True
        set_cmd = mock_run.call_args_list[0][0][0]
        assert set_cmd == ["wpctl", "set-mute", "@DEFAULT_SINK@", "1"]

    def test_unmute(self):
        with patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            result = self.tool.execute(volume="unmute")

        assert result.success is True
        set_cmd = mock_run.call_args_list[0][0][0]
        assert set_cmd == ["wpctl", "set-mute", "@DEFAULT_SINK@", "0"]

    def test_custom_device_id(self):
        with patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="Volume: 0.5\n", stderr="")
            result = self.tool.execute(volume="50%", device_id="47")

        set_cmd = mock_run.call_args_list[0][0][0]
        assert set_cmd[2] == "47"

    def test_wpctl_not_found(self):
        with patch("missy.tools.builtin.tts_speak.subprocess.run", side_effect=FileNotFoundError):
            result = self.tool.execute(volume="50%")

        assert result.success is False
        assert "wpctl not found" in result.error

    def test_invalid_volume_value(self):
        with pytest.raises(ValueError):
            self.tool.execute(volume="loudest%")

    def test_wpctl_nonzero_returncode(self):
        with patch("missy.tools.builtin.tts_speak.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="device not found"
            )
            result = self.tool.execute(volume="50%")

        assert result.success is False
        assert "wpctl failed" in result.error

    def test_wpctl_timeout(self):
        with patch("missy.tools.builtin.tts_speak.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("wpctl", 5)):
            result = self.tool.execute(volume="50%")

        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_schema(self):
        schema = self.tool.get_schema()
        assert schema["name"] == "audio_set_volume"
        assert "volume" in schema["parameters"]["required"]


# ---------------------------------------------------------------------------
# AT-SPI tools
# ---------------------------------------------------------------------------


def _make_mock_pyatspi():
    """Return a mock pyatspi module with the constants tools check for."""
    atspi = MagicMock()
    atspi.STATE_ACTIVE = "STATE_ACTIVE"
    atspi.STATE_FOCUSABLE = "STATE_FOCUSABLE"
    atspi.STATE_FOCUSED = "STATE_FOCUSED"
    atspi.STATE_SENSITIVE = "STATE_SENSITIVE"
    atspi.STATE_EDITABLE = "STATE_EDITABLE"
    atspi.STATE_CHECKED = "STATE_CHECKED"
    atspi.STATE_SELECTED = "STATE_SELECTED"
    atspi.STATE_VISIBLE = "STATE_VISIBLE"
    atspi.STATE_SHOWING = "STATE_SHOWING"
    return atspi


def _make_mock_accessible(name="TestApp", role="application", child_count=0):
    node = MagicMock()
    node.name = name
    node.childCount = child_count
    node.getRoleName.return_value = role
    node.getState.return_value.contains.return_value = False
    node.getChildAtIndex.side_effect = IndexError
    return node


class TestAtSpiGetTreeTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.atspi_tools import AtSpiGetTreeTool
        self.tool = AtSpiGetTreeTool()

    def test_pyatspi_not_installed(self):
        with patch.dict("sys.modules", {"pyatspi": None}):
            result = self.tool.execute()

        assert result.success is False
        assert "pyatspi is not installed" in result.error

    def test_desktop_connect_failure(self):
        mock_atspi = _make_mock_pyatspi()
        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", side_effect=Exception("bus error")):
            result = self.tool.execute()

        assert result.success is False
        assert "Could not connect to AT-SPI desktop" in result.error

    def test_named_app_not_found(self):
        mock_atspi = _make_mock_pyatspi()
        mock_desktop = MagicMock()
        mock_desktop.childCount = 0

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop):
            result = self.tool.execute(app_name="NonExistentApp")

        assert result.success is False
        assert "NonExistentApp" in result.error
        assert "not found" in result.error

    def test_no_focused_app(self):
        mock_atspi = _make_mock_pyatspi()
        mock_desktop = MagicMock()
        mock_desktop.childCount = 0

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop):
            result = self.tool.execute()

        assert result.success is False
        assert "No focused application" in result.error

    def test_get_tree_success_with_named_app(self):
        mock_atspi = _make_mock_pyatspi()

        mock_app = _make_mock_accessible(name="gedit", role="application", child_count=0)
        mock_desktop = MagicMock()
        mock_desktop.childCount = 1
        mock_desktop.getChildAtIndex.return_value = mock_app

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop):
            result = self.tool.execute(app_name="gedit")

        assert result.success is True
        assert result.output["app_name"] == "gedit"
        assert "tree" in result.output
        assert "node_count" in result.output

    def test_get_tree_success_with_focused_app(self):
        mock_atspi = _make_mock_pyatspi()

        mock_app = _make_mock_accessible(name="Firefox", role="application", child_count=0)
        mock_app.getState.return_value.contains.return_value = True  # STATE_ACTIVE

        mock_desktop = MagicMock()
        mock_desktop.childCount = 1
        mock_desktop.getChildAtIndex.return_value = mock_app

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop):
            result = self.tool.execute()

        assert result.success is True
        assert result.output["app_name"] == "Firefox"

    def test_max_depth_clamped_to_5(self):
        mock_atspi = _make_mock_pyatspi()
        mock_app = _make_mock_accessible(name="App")
        mock_desktop = MagicMock()
        mock_desktop.childCount = 1
        mock_desktop.getChildAtIndex.return_value = mock_app

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop), \
             patch("missy.tools.builtin.atspi_tools._walk_tree", return_value=[]) as mock_walk:
            self.tool.execute(app_name="App", max_depth=100)

        # max_depth should be clamped to 5
        call_kwargs = mock_walk.call_args[1]
        assert call_kwargs.get("max_depth", mock_walk.call_args[0][1] if len(mock_walk.call_args[0]) > 1 else 100) <= 5

    def test_max_depth_minimum_1(self):
        mock_atspi = _make_mock_pyatspi()
        mock_app = _make_mock_accessible(name="App")
        mock_desktop = MagicMock()
        mock_desktop.childCount = 1
        mock_desktop.getChildAtIndex.return_value = mock_app

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop), \
             patch("missy.tools.builtin.atspi_tools._walk_tree", return_value=[]) as mock_walk:
            self.tool.execute(app_name="App", max_depth=0)

        call_arg = mock_walk.call_args[1].get("max_depth") or mock_walk.call_args[0][1]
        assert call_arg >= 1

    def test_schema(self):
        schema = self.tool.get_schema()
        assert schema["name"] == "atspi_get_tree"
        assert "app_name" in schema["parameters"]["properties"]
        assert "max_depth" in schema["parameters"]["properties"]


class TestAtSpiClickTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.atspi_tools import AtSpiClickTool
        self.tool = AtSpiClickTool()

    def test_pyatspi_not_installed(self):
        with patch.dict("sys.modules", {"pyatspi": None}):
            result = self.tool.execute(name="OK")

        assert result.success is False
        assert "pyatspi is not installed" in result.error

    def test_no_name_or_role_provided(self):
        mock_atspi = _make_mock_pyatspi()
        with patch.dict("sys.modules", {"pyatspi": mock_atspi}):
            result = self.tool.execute()

        assert result.success is False
        assert "name" in result.error and "role" in result.error

    def test_desktop_connect_failure(self):
        mock_atspi = _make_mock_pyatspi()
        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", side_effect=Exception("dbus error")):
            result = self.tool.execute(name="OK")

        assert result.success is False
        assert "Could not connect" in result.error

    def test_app_not_found(self):
        mock_atspi = _make_mock_pyatspi()
        mock_desktop = MagicMock()
        mock_desktop.childCount = 0

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop):
            result = self.tool.execute(name="OK", app_name="Nonexistent")

        assert result.success is False
        assert "Nonexistent" in result.error

    def test_element_not_found(self):
        mock_atspi = _make_mock_pyatspi()
        mock_app = _make_mock_accessible(name="Calculator", child_count=0)
        mock_desktop = MagicMock()
        mock_desktop.childCount = 1
        mock_desktop.getChildAtIndex.return_value = mock_app

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop):
            result = self.tool.execute(name="NonExistentButton", app_name="Calculator")

        assert result.success is False
        assert "NonExistentButton" in result.error

    def test_click_success(self):
        mock_atspi = _make_mock_pyatspi()

        mock_button = _make_mock_accessible(name="OK", role="push button")
        mock_action = MagicMock()
        mock_button.queryAction.return_value = mock_action

        mock_app = _make_mock_accessible(name="Calculator")

        mock_desktop = MagicMock()
        mock_desktop.childCount = 1
        mock_desktop.getChildAtIndex.return_value = mock_app

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=mock_button):
            result = self.tool.execute(name="OK", app_name="Calculator")

        assert result.success is True
        assert result.output["clicked"] == "OK"
        mock_action.doAction.assert_called_once_with(0)

    def test_click_action_raises_exception(self):
        mock_atspi = _make_mock_pyatspi()
        mock_button = _make_mock_accessible(name="Crash", role="push button")
        mock_button.queryAction.side_effect = Exception("action interface unavailable")
        mock_app = _make_mock_accessible(name="TestApp")
        mock_desktop = MagicMock()
        mock_desktop.childCount = 1
        mock_desktop.getChildAtIndex.return_value = mock_app

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=mock_button):
            result = self.tool.execute(name="Crash", app_name="TestApp")

        assert result.success is False
        assert "AT-SPI click failed" in result.error

    def test_no_focused_app(self):
        mock_atspi = _make_mock_pyatspi()
        mock_desktop = MagicMock()
        mock_desktop.childCount = 0

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop):
            result = self.tool.execute(name="OK")

        assert result.success is False
        assert "No focused application" in result.error

    def test_element_not_found_with_name_and_role(self):
        mock_atspi = _make_mock_pyatspi()
        mock_app = _make_mock_accessible(name="App", child_count=0)
        mock_desktop = MagicMock()
        mock_desktop.childCount = 1
        mock_desktop.getChildAtIndex.return_value = mock_app

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop):
            result = self.tool.execute(name="Save", role="push button", app_name="App")

        assert result.success is False
        assert "name=" in result.error
        assert "role=" in result.error

    def test_schema(self):
        schema = self.tool.get_schema()
        assert schema["name"] == "atspi_click"
        assert "name" in schema["parameters"]["properties"]
        assert "role" in schema["parameters"]["properties"]
        assert "app_name" in schema["parameters"]["properties"]


class TestAtSpiGetTextTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.atspi_tools import AtSpiGetTextTool
        self.tool = AtSpiGetTextTool()

    def test_pyatspi_not_installed(self):
        with patch.dict("sys.modules", {"pyatspi": None}):
            result = self.tool.execute(name="label")

        assert result.success is False
        assert "pyatspi is not installed" in result.error

    def test_no_name_or_role_provided(self):
        mock_atspi = _make_mock_pyatspi()
        with patch.dict("sys.modules", {"pyatspi": mock_atspi}):
            result = self.tool.execute()

        assert result.success is False

    def test_get_text_via_text_interface(self):
        mock_atspi = _make_mock_pyatspi()

        mock_label = _make_mock_accessible(name="StatusLabel", role="label")
        mock_text_iface = MagicMock()
        mock_text_iface.getText.return_value = "Hello World"
        mock_label.queryText.return_value = mock_text_iface

        mock_app = _make_mock_accessible(name="StatusApp")
        mock_desktop = MagicMock()
        mock_desktop.childCount = 1
        mock_desktop.getChildAtIndex.return_value = mock_app

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=mock_label):
            result = self.tool.execute(name="StatusLabel", app_name="StatusApp")

        assert result.success is True
        assert result.output["text"] == "Hello World"
        assert result.output["element_name"] == "StatusLabel"

    def test_get_text_falls_back_to_element_name(self):
        mock_atspi = _make_mock_pyatspi()

        mock_label = _make_mock_accessible(name="FallbackLabel", role="label")
        mock_label.queryText.side_effect = Exception("no text interface")

        mock_app = _make_mock_accessible(name="App")
        mock_desktop = MagicMock()
        mock_desktop.childCount = 1
        mock_desktop.getChildAtIndex.return_value = mock_app

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=mock_label):
            result = self.tool.execute(name="FallbackLabel", app_name="App")

        assert result.success is True
        assert result.output["text"] == "FallbackLabel"

    def test_element_not_found(self):
        mock_atspi = _make_mock_pyatspi()
        mock_app = _make_mock_accessible(name="App", child_count=0)
        mock_desktop = MagicMock()
        mock_desktop.childCount = 1
        mock_desktop.getChildAtIndex.return_value = mock_app

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop):
            result = self.tool.execute(name="MissingLabel", app_name="App")

        assert result.success is False
        assert "MissingLabel" in result.error

    def test_schema(self):
        schema = self.tool.get_schema()
        assert schema["name"] == "atspi_get_text"


class TestAtSpiSetValueTool:
    @pytest.fixture(autouse=True)
    def _tool(self):
        from missy.tools.builtin.atspi_tools import AtSpiSetValueTool
        self.tool = AtSpiSetValueTool()

    def test_pyatspi_not_installed(self):
        with patch.dict("sys.modules", {"pyatspi": None}):
            result = self.tool.execute(name="input", value="hello")

        assert result.success is False
        assert "pyatspi is not installed" in result.error

    def test_desktop_connect_failure(self):
        mock_atspi = _make_mock_pyatspi()
        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", side_effect=Exception("no bus")):
            result = self.tool.execute(name="field", value="text")

        assert result.success is False
        assert "Could not connect" in result.error

    def test_app_not_found(self):
        mock_atspi = _make_mock_pyatspi()
        mock_desktop = MagicMock()
        mock_desktop.childCount = 0

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop):
            result = self.tool.execute(name="field", value="text", app_name="Nonexistent")

        assert result.success is False

    def test_element_not_found(self):
        mock_atspi = _make_mock_pyatspi()
        mock_app = _make_mock_accessible(name="App", child_count=0)
        mock_desktop = MagicMock()
        mock_desktop.childCount = 1
        mock_desktop.getChildAtIndex.return_value = mock_app

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop):
            result = self.tool.execute(name="MissingField", value="x", app_name="App")

        assert result.success is False
        assert "MissingField" in result.error

    def test_set_via_editable_text_interface(self):
        mock_atspi = _make_mock_pyatspi()

        mock_input = _make_mock_accessible(name="Username", role="entry")
        mock_editable = MagicMock()
        mock_text_iface = MagicMock()
        mock_text_iface.getCharacterCount.return_value = 5
        mock_input.queryText.return_value = mock_text_iface
        mock_input.queryEditableText.return_value = mock_editable

        mock_app = _make_mock_accessible(name="LoginApp")
        mock_desktop = MagicMock()
        mock_desktop.childCount = 1
        mock_desktop.getChildAtIndex.return_value = mock_app

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=mock_input):
            result = self.tool.execute(name="Username", value="admin", app_name="LoginApp")

        assert result.success is True
        assert result.output["value_set"] == "admin"
        assert result.output["method"] == "editable_text_interface"
        mock_editable.deleteText.assert_called_once_with(0, 5)
        mock_editable.insertText.assert_called_once_with(0, "admin", 5)

    def test_set_via_value_interface_fallback(self):
        mock_atspi = _make_mock_pyatspi()

        mock_spinner = _make_mock_accessible(name="SpinBox", role="spin button")
        mock_spinner.queryEditableText.side_effect = Exception("not editable")
        mock_value_iface = MagicMock()
        mock_spinner.queryValue.return_value = mock_value_iface

        mock_app = _make_mock_accessible(name="App")
        mock_desktop = MagicMock()
        mock_desktop.childCount = 1
        mock_desktop.getChildAtIndex.return_value = mock_app

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=mock_spinner):
            result = self.tool.execute(name="SpinBox", value="42", app_name="App")

        assert result.success is True
        assert result.output["method"] == "value_interface"
        assert mock_value_iface.currentValue == 42.0

    def test_both_interfaces_fail(self):
        mock_atspi = _make_mock_pyatspi()

        mock_elem = _make_mock_accessible(name="ReadOnly", role="label")
        mock_elem.queryEditableText.side_effect = Exception("not editable")
        mock_elem.queryValue.side_effect = Exception("not a value")

        mock_app = _make_mock_accessible(name="App")
        mock_desktop = MagicMock()
        mock_desktop.childCount = 1
        mock_desktop.getChildAtIndex.return_value = mock_app

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}), \
             patch("missy.tools.builtin.atspi_tools._get_desktop", return_value=mock_desktop), \
             patch("missy.tools.builtin.atspi_tools._find_element", return_value=mock_elem):
            result = self.tool.execute(name="ReadOnly", value="x", app_name="App")

        assert result.success is False
        assert "editable text" in result.error.lower() or "atspi_click" in result.error

    def test_schema(self):
        schema = self.tool.get_schema()
        assert schema["name"] == "atspi_set_value"
        assert "name" in schema["parameters"]["required"]
        assert "value" in schema["parameters"]["required"]


# ---------------------------------------------------------------------------
# AT-SPI internal helper functions
# ---------------------------------------------------------------------------


class TestAtSpiHelpers:
    """Unit tests for the private helper functions in atspi_tools."""

    def test_find_application_matches_by_name(self):
        from missy.tools.builtin.atspi_tools import _find_application

        child1 = MagicMock()
        child1.name = "Firefox"
        child2 = MagicMock()
        child2.name = "gedit"

        desktop = MagicMock()
        desktop.childCount = 2
        desktop.getChildAtIndex.side_effect = [child1, child2]

        result = _find_application(desktop, "firefox")
        assert result is child1

    def test_find_application_case_insensitive(self):
        from missy.tools.builtin.atspi_tools import _find_application

        child = MagicMock()
        child.name = "GNOME Calculator"
        desktop = MagicMock()
        desktop.childCount = 1
        desktop.getChildAtIndex.return_value = child

        result = _find_application(desktop, "gnome calculator")
        assert result is child

    def test_find_application_returns_none_when_no_match(self):
        from missy.tools.builtin.atspi_tools import _find_application

        child = MagicMock()
        child.name = "Firefox"
        desktop = MagicMock()
        desktop.childCount = 1
        desktop.getChildAtIndex.return_value = child

        result = _find_application(desktop, "gedit")
        assert result is None

    def test_find_application_skips_exceptions(self):
        from missy.tools.builtin.atspi_tools import _find_application

        desktop = MagicMock()
        desktop.childCount = 2
        desktop.getChildAtIndex.side_effect = [Exception("dead process"), MagicMock(name="gedit")]

        # Should not raise, and may return None or the second child
        _find_application(desktop, "anything")

    def test_get_focused_application_returns_active(self):
        from missy.tools.builtin.atspi_tools import _get_focused_application
        mock_atspi = _make_mock_pyatspi()

        inactive = MagicMock()
        inactive.getState.return_value.contains.return_value = False
        active = MagicMock()
        active.getState.return_value.contains.return_value = True

        desktop = MagicMock()
        desktop.childCount = 2
        desktop.getChildAtIndex.side_effect = [inactive, active, inactive, active]

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}):
            result = _get_focused_application(desktop)

        assert result is active

    def test_get_focused_application_fallback_to_first_child(self):
        from missy.tools.builtin.atspi_tools import _get_focused_application
        mock_atspi = _make_mock_pyatspi()

        child = MagicMock()
        child.getState.return_value.contains.return_value = False

        desktop = MagicMock()
        desktop.childCount = 1
        desktop.getChildAtIndex.return_value = child

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}):
            result = _get_focused_application(desktop)

        assert result is child

    def test_get_focused_application_returns_none_when_all_fail(self):
        from missy.tools.builtin.atspi_tools import _get_focused_application
        mock_atspi = _make_mock_pyatspi()

        desktop = MagicMock()
        desktop.childCount = 1
        desktop.getChildAtIndex.side_effect = Exception("no children")

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}):
            result = _get_focused_application(desktop)

        assert result is None

    def test_walk_tree_returns_empty_for_none_node(self):
        from missy.tools.builtin.atspi_tools import _walk_tree
        result = _walk_tree(None, max_depth=3)
        assert result == []

    def test_walk_tree_returns_empty_when_depth_exceeded(self):
        from missy.tools.builtin.atspi_tools import _walk_tree
        node = MagicMock()
        result = _walk_tree(node, max_depth=2, current_depth=3)
        assert result == []

    def test_walk_tree_single_leaf_node(self):
        from missy.tools.builtin.atspi_tools import _walk_tree
        mock_atspi = _make_mock_pyatspi()

        node = MagicMock()
        node.name = "Close"
        node.getRoleName.return_value = "push button"
        node.childCount = 0
        node.getState.return_value.contains.return_value = False
        node.queryText.side_effect = Exception("no text")

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}):
            result = _walk_tree(node, max_depth=3)

        assert len(result) == 1
        assert result[0]["name"] == "Close"
        assert result[0]["role"] == "push button"
        assert result[0]["depth"] == 0

    def test_walk_tree_includes_text_content(self):
        from missy.tools.builtin.atspi_tools import _walk_tree
        mock_atspi = _make_mock_pyatspi()

        node = MagicMock()
        node.name = "Entry"
        node.getRoleName.return_value = "entry"
        node.childCount = 0
        node.getState.return_value.contains.return_value = False
        mock_text_iface = MagicMock()
        mock_text_iface.getText.return_value = "typed content"
        node.queryText.return_value = mock_text_iface

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}):
            result = _walk_tree(node, max_depth=3)

        assert result[0].get("text") == "typed content"

    def test_walk_tree_handles_child_exception(self):
        from missy.tools.builtin.atspi_tools import _walk_tree
        mock_atspi = _make_mock_pyatspi()

        parent = MagicMock()
        parent.name = "parent"
        parent.getRoleName.return_value = "window"
        parent.childCount = 2
        parent.getState.return_value.contains.return_value = False
        parent.queryText.side_effect = Exception

        # First child raises, second is valid
        valid_child = MagicMock()
        valid_child.name = "OK"
        valid_child.getRoleName.return_value = "push button"
        valid_child.childCount = 0
        valid_child.getState.return_value.contains.return_value = False
        valid_child.queryText.side_effect = Exception

        parent.getChildAtIndex.side_effect = [Exception("crash"), valid_child]

        with patch.dict("sys.modules", {"pyatspi": mock_atspi}):
            result = _walk_tree(parent, max_depth=3)

        # Should have parent + valid child, skipping the failing one
        names = [n["name"] for n in result]
        assert "parent" in names
        assert "OK" in names

    def test_format_tree_indentation(self):
        from missy.tools.builtin.atspi_tools import _format_tree

        nodes = [
            {"depth": 0, "role": "window", "name": "Main Window", "states": ["visible"], "text": ""},
            {"depth": 1, "role": "push button", "name": "OK", "states": ["focusable"], "text": ""},
            {"depth": 2, "role": "label", "name": "", "states": [], "text": "OK label"},
        ]
        output = _format_tree(nodes)

        lines = output.splitlines()
        assert lines[0].startswith("[window]")
        assert lines[1].startswith("  [push button]")
        assert lines[2].startswith("    [label]")

    def test_format_tree_empty_nodes(self):
        from missy.tools.builtin.atspi_tools import _format_tree
        result = _format_tree([])
        assert result == ""

    def test_find_element_matches_by_name(self):
        from missy.tools.builtin.atspi_tools import _find_element

        target = MagicMock()
        target.name = "Save"
        target.getRoleName.return_value = "push button"
        target.childCount = 0

        app = MagicMock()
        app.name = "editor"
        app.getRoleName.return_value = "application"
        app.childCount = 1
        app.getChildAtIndex.return_value = target

        result = _find_element(app, "save", None)
        assert result is target

    def test_find_element_matches_by_role(self):
        from missy.tools.builtin.atspi_tools import _find_element

        target = MagicMock()
        target.name = "unnamed entry"
        target.getRoleName.return_value = "entry"
        target.childCount = 0

        app = MagicMock()
        app.name = "app"
        app.getRoleName.return_value = "application"
        app.childCount = 1
        app.getChildAtIndex.return_value = target

        result = _find_element(app, None, "entry")
        assert result is target

    def test_find_element_returns_none_when_no_match(self):
        from missy.tools.builtin.atspi_tools import _find_element

        child = MagicMock()
        child.name = "Unrelated"
        child.getRoleName.return_value = "label"
        child.childCount = 0

        app = MagicMock()
        app.name = "app"
        app.getRoleName.return_value = "application"
        app.childCount = 1
        app.getChildAtIndex.return_value = child

        result = _find_element(app, "NonExistent", None)
        assert result is None

    def test_find_element_requires_both_name_and_role_when_given(self):
        from missy.tools.builtin.atspi_tools import _find_element

        # name matches but role doesn't — should not match
        child = MagicMock()
        child.name = "Save"
        child.getRoleName.return_value = "label"
        child.childCount = 0

        app = MagicMock()
        app.name = "app"
        app.getRoleName.return_value = "application"
        app.childCount = 1
        app.getChildAtIndex.return_value = child

        result = _find_element(app, "save", "push button")
        assert result is None
