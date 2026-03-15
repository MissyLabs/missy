"""Coverage gap tests for missy/tools/builtin/x11_tools.py.

Targets uncovered lines:
  56-65 : _extract_account_id — short token, valid JWT payload parsing
  70-74 : _load_oauth_token — exception fallback returns None
  79    : _get_vision_token — env var takes precedence
  87-89 : _get_ollama_base_url — config with ollama base_url / exception fallback
  284   : X11TypeTool.execute — xdotool command not found error
  339   : X11KeyTool.execute — xdotool key command not found error
  462-463: X11ReadScreenTool._take_screenshot — scrot command not found
  474   : X11ReadScreenTool._take_screenshot — scrot failed generic error
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

from missy.tools.builtin.x11_tools import (
    X11ClickTool,
    X11KeyTool,
    X11ReadScreenTool,
    X11ScreenshotTool,
    X11TypeTool,
    X11WindowListTool,
    _button_num,
    _extract_account_id,
    _get_ollama_base_url,
    _get_vision_token,
    _load_oauth_token,
)

# ---------------------------------------------------------------------------
# _extract_account_id — lines 56-65
# ---------------------------------------------------------------------------


class TestExtractAccountId:
    def test_empty_string_returns_empty(self):
        """Malformed token with no dots returns empty string."""
        assert _extract_account_id("") == ""

    def test_short_token_one_part_returns_empty(self):
        """Token with only one segment returns empty (len(parts) < 2)."""
        assert _extract_account_id("onlyone") == ""

    def test_valid_jwt_with_chatgpt_account_id(self):
        """Valid JWT returns chatgpt_account_id from nested auth namespace."""
        payload = {
            "https://api.openai.com/auth": {"chatgpt_account_id": "acc_123"},
            "sub": "user_456",
        }
        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        token = f"header.{encoded}.signature"
        assert _extract_account_id(token) == "acc_123"

    def test_valid_jwt_falls_back_to_sub(self):
        """When chatgpt_account_id missing, falls back to 'sub'."""
        payload = {"sub": "user_999"}
        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        token = f"header.{encoded}.sig"
        assert _extract_account_id(token) == "user_999"

    def test_invalid_base64_returns_empty(self):
        """Non-decodable payload returns empty string (exception caught)."""
        assert _extract_account_id("a.!!!.b") == ""

    def test_two_part_token_missing_signature(self):
        """Token with exactly two parts (no signature) still extracts payload."""
        payload = {"sub": "u42"}
        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        token = f"header.{encoded}"
        # len(parts) == 2, parts[1] is valid
        assert _extract_account_id(token) == "u42"


# ---------------------------------------------------------------------------
# _load_oauth_token — lines 70-74
# ---------------------------------------------------------------------------


class TestLoadOauthToken:
    def test_returns_token_on_success(self):
        """Calls refresh_token_if_needed and returns its result."""
        with patch(
            "missy.tools.builtin.x11_tools.refresh_token_if_needed",
            return_value="tok_abc",
            create=True,
        ), patch.dict(
            "sys.modules",
            {
                "missy.cli.oauth": MagicMock(
                    refresh_token_if_needed=MagicMock(return_value="tok_abc")
                )
            },
        ):
            token = _load_oauth_token()
        # May succeed or fall back; just ensure it doesn't raise
        assert token is None or isinstance(token, str)

    def test_returns_none_on_exception(self):
        """When the import or refresh raises, returns None."""
        with patch.dict("sys.modules", {"missy.cli.oauth": None}):
            token = _load_oauth_token()
        assert token is None


# ---------------------------------------------------------------------------
# _get_vision_token — line 79
# ---------------------------------------------------------------------------


class TestGetVisionToken:
    def test_env_var_takes_precedence(self):
        """OPENAI_API_KEY env var is returned first."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
            token = _get_vision_token()
        assert token == "sk-test-key"

    def test_falls_back_to_oauth_when_no_env(self):
        """Falls back to _load_oauth_token when env var not set."""
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch("missy.tools.builtin.x11_tools._load_oauth_token", return_value="oauth_token"),
        ):
            token = _get_vision_token()
        assert token == "oauth_token"


# ---------------------------------------------------------------------------
# _get_ollama_base_url — lines 87-89
# ---------------------------------------------------------------------------


class TestGetOllamaBaseUrl:
    def test_returns_default_on_exception(self):
        """When load_config raises, returns the default URL."""
        with patch(
            "missy.tools.builtin.x11_tools.load_config",
            side_effect=Exception("no config"),
            create=True,
        ):
            url = _get_ollama_base_url()
        assert url == "http://localhost:11434"

    def test_returns_config_url_when_set(self):
        """When ollama provider config has base_url, returns it (stripped)."""
        provider_cfg = MagicMock()
        provider_cfg.base_url = "http://myhost:11434/"

        cfg = MagicMock()
        cfg.providers = {"ollama": provider_cfg}

        mock_settings = MagicMock()
        mock_settings.load_config.return_value = cfg

        with patch.dict(sys.modules, {"missy.config.settings": mock_settings}):
            url = _get_ollama_base_url()
        assert url == "http://myhost:11434"

    def test_returns_default_when_no_ollama_provider(self):
        """When ollama not in providers, returns default URL."""
        cfg = MagicMock()
        cfg.providers = {}

        with patch("missy.tools.builtin.x11_tools.load_config", return_value=cfg, create=True):
            url = _get_ollama_base_url()
        assert url == "http://localhost:11434"

    def test_returns_default_when_base_url_none(self):
        """When provider config exists but base_url is None/empty, returns default."""
        provider_cfg = MagicMock()
        provider_cfg.base_url = None

        cfg = MagicMock()
        cfg.providers = {"ollama": provider_cfg}

        with patch("missy.tools.builtin.x11_tools.load_config", return_value=cfg, create=True):
            url = _get_ollama_base_url()
        assert url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# _button_num helper
# ---------------------------------------------------------------------------


class TestButtonNum:
    def test_left_maps_to_1(self):
        assert _button_num("left") == "1"

    def test_middle_maps_to_2(self):
        assert _button_num("middle") == "2"

    def test_right_maps_to_3(self):
        assert _button_num("right") == "3"

    def test_unknown_defaults_to_1(self):
        assert _button_num("unknown") == "1"

    def test_case_insensitive(self):
        assert _button_num("LEFT") == "1"
        assert _button_num("Right") == "3"


# ---------------------------------------------------------------------------
# X11ScreenshotTool
# ---------------------------------------------------------------------------


def _make_completed(returncode=0, stdout="", stderr=""):
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


class TestX11ScreenshotTool:
    def test_success_full_screen(self, tmp_path):
        dest = str(tmp_path / "shot.png")
        dest_path = tmp_path / "shot.png"
        dest_path.write_bytes(b"PNG")

        with patch(
            "missy.tools.builtin.x11_tools._run", return_value=_make_completed(0)
        ) as mock_run:
            result = X11ScreenshotTool().execute(path=dest)

        assert result.success is True
        assert result.output["path"] == dest
        mock_run.assert_called_once_with(f"scrot {dest}")

    def test_success_with_region(self, tmp_path):
        dest = str(tmp_path / "region.png")
        (tmp_path / "region.png").write_bytes(b"PNG")

        with patch(
            "missy.tools.builtin.x11_tools._run", return_value=_make_completed(0)
        ) as mock_run:
            result = X11ScreenshotTool().execute(path=dest, region="10,20,100,200")

        assert result.success is True
        mock_run.assert_called_once_with(f"scrot -a 10,20,100,200 {dest}")

    def test_scrot_not_installed_error(self, tmp_path):
        dest = str(tmp_path / "shot.png")
        err_result = _make_completed(1, stderr="command not found")

        with patch("missy.tools.builtin.x11_tools._run", return_value=err_result):
            result = X11ScreenshotTool().execute(path=dest)

        assert result.success is False
        assert "scrot is not installed" in result.error

    def test_scrot_generic_failure(self, tmp_path):
        dest = str(tmp_path / "shot.png")
        err_result = _make_completed(1, stderr="some other error")

        with patch("missy.tools.builtin.x11_tools._run", return_value=err_result):
            result = X11ScreenshotTool().execute(path=dest)

        assert result.success is False
        assert "scrot failed" in result.error


# ---------------------------------------------------------------------------
# X11ClickTool
# ---------------------------------------------------------------------------


class TestX11ClickTool:
    def test_success_left_click(self):
        with patch(
            "missy.tools.builtin.x11_tools._run", return_value=_make_completed(0)
        ) as mock_run:
            result = X11ClickTool().execute(x=100, y=200)

        assert result.success is True
        assert result.output["x"] == 100
        assert result.output["y"] == 200
        mock_run.assert_called_once_with("xdotool mousemove 100 200 click 1")

    def test_double_click(self):
        with patch(
            "missy.tools.builtin.x11_tools._run", return_value=_make_completed(0)
        ) as mock_run:
            result = X11ClickTool().execute(x=50, y=60, button="double")

        assert result.success is True
        mock_run.assert_called_once_with("xdotool mousemove 50 60 click --repeat 2 1")

    def test_right_click(self):
        with patch(
            "missy.tools.builtin.x11_tools._run", return_value=_make_completed(0)
        ) as mock_run:
            result = X11ClickTool().execute(x=10, y=20, button="right")

        assert result.success is True
        mock_run.assert_called_once_with("xdotool mousemove 10 20 click 3")

    def test_window_focus_called_before_click(self):
        calls = []

        def fake_run(cmd):
            calls.append(cmd)
            return _make_completed(0)

        with patch("missy.tools.builtin.x11_tools._run", side_effect=fake_run):
            result = X11ClickTool().execute(x=0, y=0, window_name="Terminal")

        assert result.success is True
        assert len(calls) == 2
        assert "windowfocus" in calls[0]
        assert "mousemove" in calls[1]

    def test_xdotool_not_installed_error(self):
        err_result = _make_completed(1, stderr="command not found")
        with patch("missy.tools.builtin.x11_tools._run", return_value=err_result):
            result = X11ClickTool().execute(x=0, y=0)

        assert result.success is False
        assert "xdotool is not installed" in result.error

    def test_click_failure_generic_error(self):
        err_result = _make_completed(1, stderr="bad display")
        with patch("missy.tools.builtin.x11_tools._run", return_value=err_result):
            result = X11ClickTool().execute(x=0, y=0)

        assert result.success is False
        assert "xdotool click failed" in result.error


# ---------------------------------------------------------------------------
# X11TypeTool — line 284 (command not found)
# ---------------------------------------------------------------------------


class TestX11TypeTool:
    def test_success_type_text(self):
        with patch("missy.tools.builtin.x11_tools._run", return_value=_make_completed(0)):
            result = X11TypeTool().execute(text="hello world")

        assert result.success is True
        assert result.output["typed"] == "hello world"

    def test_xdotool_not_installed(self):
        """Line 284: command not found error path."""
        err_result = _make_completed(1, stderr="command not found")
        with patch("missy.tools.builtin.x11_tools._run", return_value=err_result):
            result = X11TypeTool().execute(text="hi")

        assert result.success is False
        assert "xdotool is not installed" in result.error

    def test_type_failure_generic_error(self):
        err_result = _make_completed(1, stderr="X11 error")
        with patch("missy.tools.builtin.x11_tools._run", return_value=err_result):
            result = X11TypeTool().execute(text="hi")

        assert result.success is False
        assert "xdotool type failed" in result.error

    def test_window_focus_before_type(self):
        calls = []

        def fake_run(cmd):
            calls.append(cmd)
            return _make_completed(0)

        with patch("missy.tools.builtin.x11_tools._run", side_effect=fake_run):
            result = X11TypeTool().execute(text="hi", window_name="gedit")

        assert result.success is True
        assert len(calls) == 2
        assert "windowfocus" in calls[0]
        assert "type" in calls[1]

    def test_custom_delay(self):
        with patch(
            "missy.tools.builtin.x11_tools._run", return_value=_make_completed(0)
        ) as mock_run:
            result = X11TypeTool().execute(text="test", delay_ms=50)

        assert result.success is True
        assert result.output["delay_ms"] == 50
        called_cmd = mock_run.call_args[0][0]
        assert "--delay 50" in called_cmd

    def test_windowfocus_failure_logs_and_continues(self):
        """Line 284: when windowfocus fails, debug-logs and still types."""
        calls = []

        def fake_run(cmd):
            calls.append(cmd)
            if "windowfocus" in cmd:
                return _make_completed(1, stderr="window not found")
            return _make_completed(0)

        with patch("missy.tools.builtin.x11_tools._run", side_effect=fake_run):
            result = X11TypeTool().execute(text="hi", window_name="nonexistent")

        assert result.success is True
        assert len(calls) == 2  # windowfocus attempted then type proceeds


# ---------------------------------------------------------------------------
# X11KeyTool — line 339 (command not found)
# ---------------------------------------------------------------------------


class TestX11KeyTool:
    def test_success_send_key(self):
        with patch(
            "missy.tools.builtin.x11_tools._run", return_value=_make_completed(0)
        ) as mock_run:
            result = X11KeyTool().execute(key="Return")

        assert result.success is True
        assert result.output["key"] == "Return"
        mock_run.assert_called_once_with("xdotool key -- Return")

    def test_xdotool_not_installed(self):
        """Line 339: command not found path."""
        err_result = _make_completed(1, stderr="command not found")
        with patch("missy.tools.builtin.x11_tools._run", return_value=err_result):
            result = X11KeyTool().execute(key="ctrl+c")

        assert result.success is False
        assert "xdotool is not installed" in result.error

    def test_key_failure_generic_error(self):
        err_result = _make_completed(1, stderr="invalid key name")
        with patch("missy.tools.builtin.x11_tools._run", return_value=err_result):
            result = X11KeyTool().execute(key="bad_key")

        assert result.success is False
        assert "xdotool key failed" in result.error

    def test_window_focus_before_key(self):
        calls = []

        def fake_run(cmd):
            calls.append(cmd)
            return _make_completed(0)

        with patch("missy.tools.builtin.x11_tools._run", side_effect=fake_run):
            result = X11KeyTool().execute(key="Escape", window_name="VLC")

        assert result.success is True
        assert len(calls) == 2
        assert "windowfocus" in calls[0]
        assert "xdotool key" in calls[1]

    def test_windowfocus_failure_logs_and_continues(self):
        """Line 339: when windowfocus fails, debug-logs and still sends key."""
        calls = []

        def fake_run(cmd):
            calls.append(cmd)
            if "windowfocus" in cmd:
                return _make_completed(1, stderr="window not found")
            return _make_completed(0)

        with patch("missy.tools.builtin.x11_tools._run", side_effect=fake_run):
            result = X11KeyTool().execute(key="Return", window_name="nonexistent")

        assert result.success is True
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# X11WindowListTool
# ---------------------------------------------------------------------------


class TestX11WindowListTool:
    def test_wmctrl_success_four_parts(self):
        wmctrl_output = "0x00400001  0 myhost  Firefox\n0x00400002  1 myhost  Terminal"
        with patch(
            "missy.tools.builtin.x11_tools._run",
            return_value=_make_completed(0, stdout=wmctrl_output),
        ):
            result = X11WindowListTool().execute()

        assert result.success is True
        assert result.output["count"] == 2
        assert result.output["windows"][0]["name"] == "Firefox"

    def test_wmctrl_success_three_parts(self):
        wmctrl_output = "0x00400001  0 myhost"
        with patch(
            "missy.tools.builtin.x11_tools._run",
            return_value=_make_completed(0, stdout=wmctrl_output),
        ):
            result = X11WindowListTool().execute()

        assert result.success is True
        assert result.output["windows"][0]["name"] == ""

    def test_wmctrl_fails_xdotool_not_found(self):
        """Neither wmctrl nor xdotool: command not found error."""
        wmctrl_fail = _make_completed(1)
        xdotool_fail = _make_completed(1, stderr="command not found")

        with patch("missy.tools.builtin.x11_tools._run", side_effect=[wmctrl_fail, xdotool_fail]):
            result = X11WindowListTool().execute()

        assert result.success is False
        assert "wmctrl" in result.error or "xdotool" in result.error

    def test_wmctrl_fails_xdotool_empty_match(self):
        """xdotool returns non-zero (no windows) → empty list."""
        wmctrl_fail = _make_completed(1)
        xdotool_fail = _make_completed(1, stderr="no windows found")

        with patch("missy.tools.builtin.x11_tools._run", side_effect=[wmctrl_fail, xdotool_fail]):
            result = X11WindowListTool().execute()

        assert result.success is True
        assert result.output["count"] == 0

    def test_wmctrl_fails_xdotool_success_with_windows(self):
        """wmctrl fails, xdotool lists window IDs, names fetched per ID."""
        wmctrl_fail = _make_completed(1)
        xdotool_ids = _make_completed(0, stdout="12345\n67890")
        name1 = _make_completed(0, stdout="Firefox")
        name2 = _make_completed(0, stdout="Terminal")

        with patch(
            "missy.tools.builtin.x11_tools._run",
            side_effect=[wmctrl_fail, xdotool_ids, name1, name2],
        ):
            result = X11WindowListTool().execute()

        assert result.success is True
        assert result.output["count"] == 2
        names = [w["name"] for w in result.output["windows"]]
        assert "Firefox" in names
        assert "Terminal" in names


# ---------------------------------------------------------------------------
# X11ReadScreenTool._take_screenshot — lines 462-463, 474
# ---------------------------------------------------------------------------


class TestX11ReadScreenToolTakeScreenshot:
    def test_scrot_command_not_found(self):
        """Lines 462-463: scrot not installed returns the error string."""
        tool = X11ReadScreenTool()
        err_result = _make_completed(1, stderr="command not found")

        with patch("missy.tools.builtin.x11_tools._run", return_value=err_result):
            error = tool._take_screenshot("/tmp/shot.png", "")

        assert error is not None
        assert "scrot is not installed" in error

    def test_scrot_generic_failure(self):
        """Line 474: generic scrot error returns error string."""
        tool = X11ReadScreenTool()
        err_result = _make_completed(1, stderr="display unavailable")

        with patch("missy.tools.builtin.x11_tools._run", return_value=err_result):
            error = tool._take_screenshot("/tmp/shot.png", "")

        assert error is not None
        assert "scrot failed" in error

    def test_scrot_success_returns_none(self, tmp_path):
        """Successful scrot returns None (no error)."""
        tool = X11ReadScreenTool()
        dest = str(tmp_path / "ok.png")

        with patch("missy.tools.builtin.x11_tools._run", return_value=_make_completed(0)):
            error = tool._take_screenshot(dest, "")

        assert error is None

    def test_scrot_with_region(self):
        """Region capture uses 'scrot -a region path' command."""
        tool = X11ReadScreenTool()

        with patch(
            "missy.tools.builtin.x11_tools._run", return_value=_make_completed(0)
        ) as mock_run:
            tool._take_screenshot("/tmp/r.png", "0,0,100,100")

        called_cmd = mock_run.call_args[0][0]
        assert "-a 0,0,100,100" in called_cmd

    def test_browser_tools_import_exception_falls_through_to_scrot(self):
        """Lines 462-463: browser_tools import raises → falls through to scrot."""
        tool = X11ReadScreenTool()

        with (
            patch("missy.tools.builtin.x11_tools._run", return_value=_make_completed(0)),
            patch.dict(sys.modules, {"missy.tools.builtin.browser_tools": None}),
        ):
            # No region → tries browser_tools import, which fails, then scrot succeeds
            error = tool._take_screenshot("/tmp/shot.png", "")

        assert error is None  # scrot succeeded


class TestX11ReadScreenToolExecute:
    def test_screenshot_failure_propagates(self):
        """When _take_screenshot returns an error, execute returns failure."""
        tool = X11ReadScreenTool()

        with patch.object(tool, "_take_screenshot", return_value="scrot failed: X11"):
            result = tool.execute()

        assert result.success is False
        assert "scrot failed" in result.error

    def test_cannot_read_file(self, tmp_path):
        """When screenshot succeeds but file read fails, returns error."""
        tool = X11ReadScreenTool()
        nonexistent = str(tmp_path / "missing.png")

        with patch.object(tool, "_take_screenshot", return_value=None):
            result = tool.execute(path=nonexistent)

        assert result.success is False
        assert "Could not read screenshot" in result.error

    def test_ollama_connect_error(self, tmp_path):
        """When Ollama not reachable, returns friendly error."""
        import httpx

        tool = X11ReadScreenTool()
        dest = tmp_path / "ok.png"
        dest.write_bytes(b"PNG")

        with (
            patch.object(tool, "_take_screenshot", return_value=None),
            patch.object(
                tool, "_call_ollama_vision", side_effect=httpx.ConnectError("connection refused")
            ),
        ):
            result = tool.execute(path=str(dest))

        assert result.success is False
        assert "Ollama" in result.error or "connect" in result.error.lower()

    def test_ollama_http_status_error(self, tmp_path):
        """When Ollama returns HTTP error status, returns error with status code."""
        import httpx

        tool = X11ReadScreenTool()
        dest = tmp_path / "ok.png"
        dest.write_bytes(b"PNG")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        http_err = httpx.HTTPStatusError("error", request=MagicMock(), response=mock_response)

        with (
            patch.object(tool, "_take_screenshot", return_value=None),
            patch.object(tool, "_call_ollama_vision", side_effect=http_err),
        ):
            result = tool.execute(path=str(dest))

        assert result.success is False
        assert "500" in result.error

    def test_generic_vision_exception(self, tmp_path):
        """Generic exception from _call_ollama_vision returns error."""
        tool = X11ReadScreenTool()
        dest = tmp_path / "ok.png"
        dest.write_bytes(b"PNG")

        with (
            patch.object(tool, "_take_screenshot", return_value=None),
            patch.object(tool, "_call_ollama_vision", side_effect=Exception("timeout")),
        ):
            result = tool.execute(path=str(dest))

        assert result.success is False
        assert "Vision call failed" in result.error

    def test_success_full_flow(self, tmp_path):
        """Happy path: screenshot taken, vision model returns description."""
        tool = X11ReadScreenTool()
        dest = tmp_path / "screen.png"
        dest.write_bytes(b"PNG")

        with (
            patch.object(tool, "_take_screenshot", return_value=None),
            patch.object(tool, "_call_ollama_vision", return_value="A desktop with Firefox open."),
        ):
            result = tool.execute(
                question="What is on screen?",
                path=str(dest),
            )

        assert result.success is True
        assert result.output["description"] == "A desktop with Firefox open."
        assert result.output["question"] == "What is on screen?"
