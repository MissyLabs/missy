"""X11 desktop automation tools for Missy.

Provides tools to interact with the X11 display: take screenshots,
click, type, send key presses, list windows, and interpret the screen
via AI vision.

All tools require ``shell=True`` permissions and inject ``DISPLAY=:0``
into the subprocess environment (falling back to the existing
``DISPLAY`` env var if set).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
from typing import Any

import httpx

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

logger = logging.getLogger(__name__)

_CODEX_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
_DEFAULT_MODEL = "gpt-5.2"

# Vision model defaults — Ollama is the primary backend for x11_read_screen.
_OLLAMA_VISION_MODEL = "minicpm-v"
_OLLAMA_DEFAULT_URL = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _display_env() -> dict[str, str]:
    """Return an environment dict with DISPLAY set."""
    return {**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")}


def _run(cmd: str) -> subprocess.CompletedProcess:
    """Run *cmd* via the shell and return the completed process."""
    return subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        env=_display_env(),
    )


def _extract_account_id(token: str) -> str:
    """Pull ``chatgpt_account_id`` from the JWT payload without verification."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return ""
        padding = 4 - len(parts[1]) % 4
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * padding))
        ns = payload.get("https://api.openai.com/auth", {})
        return ns.get("chatgpt_account_id", "") or payload.get("sub", "")
    except Exception as _jwt_exc:
        logger.debug("x11: JWT parse failed: %s", _jwt_exc)
        return ""


def _load_oauth_token() -> str | None:
    """Load the stored OAuth access token, refreshing if needed."""
    try:
        from missy.cli.oauth import refresh_token_if_needed

        return refresh_token_if_needed()
    except Exception as _oauth_exc:
        logger.debug("x11: OAuth token load failed: %s", _oauth_exc)
        return None


def _get_vision_token() -> str | None:
    """Return an API token suitable for vision calls."""
    return os.environ.get("OPENAI_API_KEY") or _load_oauth_token()


def _get_ollama_base_url() -> str:
    """Return the Ollama base URL from config or default."""
    try:
        from missy.config.settings import load_config

        cfg = load_config()
        provider_cfg = cfg.providers.get("ollama")
        if provider_cfg and provider_cfg.base_url:
            return provider_cfg.base_url.rstrip("/")
    except Exception as _cfg_exc:
        logger.debug("x11: Ollama config load failed: %s", _cfg_exc)
    return _OLLAMA_DEFAULT_URL


# ---------------------------------------------------------------------------
# Button map helper
# ---------------------------------------------------------------------------


def _button_num(button: str) -> str:
    """Map a human-friendly button name to the xdotool button number."""
    mapping = {"left": "1", "middle": "2", "right": "3"}
    return mapping.get(button.lower(), "1")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


class X11ScreenshotTool(BaseTool):
    """Take a screenshot of the X11 desktop and save it to a file.

    Returns the file path and file size in bytes.
    """

    name = "x11_screenshot"
    description = (
        "Take a screenshot of the X11 desktop and save it to a file. "
        "Returns the file path and image dimensions."
    )
    permissions = ToolPermissions(shell=True, filesystem_write=True)

    parameters: dict[str, Any] = {
        "path": {
            "type": "string",
            "description": "Destination file path for the PNG screenshot.",
            "default": "/tmp/screenshot.png",
        },
        "region": {
            "type": "string",
            "description": (
                "Optional region to capture as 'x,y,w,h'. When omitted the full screen is captured."
            ),
        },
    }

    def execute(
        self, *, path: str = "/tmp/screenshot.png", region: str = "", **_: Any
    ) -> ToolResult:
        cmd = f"scrot -a {region} {path}" if region else f"scrot {path}"

        result = _run(cmd)

        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            if "No such file or directory" in err or "command not found" in err:
                return ToolResult(
                    success=False,
                    output=None,
                    error="scrot is not installed. Install it with: sudo apt install scrot",
                )
            return ToolResult(success=False, output=None, error=f"scrot failed: {err}")

        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0

        return ToolResult(
            success=True,
            output={"path": path, "size_bytes": size},
        )


class X11ClickTool(BaseTool):
    """Click at an x,y coordinate on the screen.

    Optionally focuses a window by name before clicking.
    Supports left, right, and double click.
    """

    name = "x11_click"
    description = (
        "Click at an x,y coordinate on the screen, or click on a window by name. "
        "Supports left/right/double click."
    )
    permissions = ToolPermissions(shell=True)

    parameters: dict[str, Any] = {
        "x": {
            "type": "integer",
            "description": "X coordinate to click.",
            "required": True,
        },
        "y": {
            "type": "integer",
            "description": "Y coordinate to click.",
            "required": True,
        },
        "button": {
            "type": "string",
            "description": "Mouse button: 'left', 'right', or 'double'. Defaults to 'left'.",
            "default": "left",
        },
        "window_name": {
            "type": "string",
            "description": "Optional window name pattern to focus before clicking.",
        },
    }

    def execute(
        self,
        *,
        x: int,
        y: int,
        button: str = "left",
        window_name: str = "",
        **_: Any,
    ) -> ToolResult:

        if window_name:
            focus_cmd = f"xdotool search --name {json.dumps(window_name)} windowfocus"
            r = _run(focus_cmd)
            if r.returncode != 0:
                logger.debug("xdotool windowfocus stderr: %s", r.stderr.strip())

        if button.lower() == "double":
            cmd = f"xdotool mousemove {x} {y} click --repeat 2 1"
        else:
            btn_num = _button_num(button)
            cmd = f"xdotool mousemove {x} {y} click {btn_num}"

        result = _run(cmd)
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            if "command not found" in err:
                return ToolResult(
                    success=False,
                    output=None,
                    error="xdotool is not installed. Install it with: sudo apt install xdotool",
                )
            return ToolResult(success=False, output=None, error=f"xdotool click failed: {err}")

        return ToolResult(
            success=True,
            output={"x": x, "y": y, "button": button, "window_name": window_name or None},
        )


class X11TypeTool(BaseTool):
    """Type text into the currently focused window.

    Optionally focuses a window by name first.
    """

    name = "x11_type"
    description = (
        "Type text into the currently focused window. Optionally focus a window first by name."
    )
    permissions = ToolPermissions(shell=True)

    parameters: dict[str, Any] = {
        "text": {
            "type": "string",
            "description": "Text to type.",
            "required": True,
        },
        "window_name": {
            "type": "string",
            "description": "Optional window name pattern to focus before typing.",
        },
        "delay_ms": {
            "type": "integer",
            "description": "Delay in milliseconds between keystrokes. Defaults to 12.",
            "default": 12,
        },
    }

    def execute(
        self,
        *,
        text: str,
        window_name: str = "",
        delay_ms: int = 12,
        **_: Any,
    ) -> ToolResult:
        if window_name:
            focus_cmd = f"xdotool search --name {json.dumps(window_name)} windowfocus"
            r = _run(focus_cmd)
            if r.returncode != 0:
                logger.debug("xdotool windowfocus stderr: %s", r.stderr.strip())

        # Use -- to prevent text that starts with - from being interpreted as flags.
        # We pass text via shell quoting using json.dumps (which produces a valid
        # double-quoted string) to handle spaces and special characters safely.
        quoted_text = json.dumps(text)
        cmd = f"xdotool type --delay {delay_ms} -- {quoted_text}"

        result = _run(cmd)
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            if "command not found" in err:
                return ToolResult(
                    success=False,
                    output=None,
                    error="xdotool is not installed. Install it with: sudo apt install xdotool",
                )
            return ToolResult(success=False, output=None, error=f"xdotool type failed: {err}")

        return ToolResult(
            success=True,
            output={"typed": text, "delay_ms": delay_ms, "window_name": window_name or None},
        )


class X11KeyTool(BaseTool):
    """Send a key press or keyboard shortcut to the focused window.

    Examples of valid key names: 'Return', 'ctrl+c', 'alt+F4', 'Tab', 'Escape'.
    """

    name = "x11_key"
    description = (
        "Send a key press or keyboard shortcut to the focused window. "
        "Examples: 'Return', 'ctrl+c', 'alt+F4', 'Tab', 'Escape'."
    )
    permissions = ToolPermissions(shell=True)

    parameters: dict[str, Any] = {
        "key": {
            "type": "string",
            "description": "Key or shortcut to send, e.g. 'Return', 'ctrl+c', 'alt+F4'.",
            "required": True,
        },
        "window_name": {
            "type": "string",
            "description": "Optional window name pattern to focus before sending the key.",
        },
    }

    def execute(self, *, key: str, window_name: str = "", **_: Any) -> ToolResult:
        if window_name:
            focus_cmd = f"xdotool search --name {json.dumps(window_name)} windowfocus"
            r = _run(focus_cmd)
            if r.returncode != 0:
                logger.debug("xdotool windowfocus stderr: %s", r.stderr.strip())

        cmd = f"xdotool key {key}"
        result = _run(cmd)
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            if "command not found" in err:
                return ToolResult(
                    success=False,
                    output=None,
                    error="xdotool is not installed. Install it with: sudo apt install xdotool",
                )
            return ToolResult(success=False, output=None, error=f"xdotool key failed: {err}")

        return ToolResult(
            success=True,
            output={"key": key, "window_name": window_name or None},
        )


class X11WindowListTool(BaseTool):
    """List all open X11 windows with their names and IDs."""

    name = "x11_window_list"
    description = (
        "List all open X11 windows with their names and IDs. "
        "Useful to find window names for targeting clicks."
    )
    permissions = ToolPermissions(shell=True)

    parameters: dict[str, Any] = {}

    def execute(self, **_: Any) -> ToolResult:
        # Prefer wmctrl -l (gives cleaner output); fall back to xdotool.
        wmctrl_result = _run("wmctrl -l")
        if wmctrl_result.returncode == 0:
            windows: list[dict] = []
            for line in wmctrl_result.stdout.splitlines():
                parts = line.split(None, 3)
                if len(parts) >= 4:
                    windows.append(
                        {"id": parts[0], "desktop": parts[1], "host": parts[2], "name": parts[3]}
                    )
                elif len(parts) == 3:
                    windows.append(
                        {"id": parts[0], "desktop": parts[1], "host": parts[2], "name": ""}
                    )
            return ToolResult(success=True, output={"windows": windows, "count": len(windows)})

        # Fallback: xdotool search for all visible windows then get names.
        search_result = _run('xdotool search --onlyvisible --name ""')
        if search_result.returncode != 0:
            err = search_result.stderr.strip()
            if "command not found" in err:
                return ToolResult(
                    success=False,
                    output=None,
                    error=(
                        "Neither wmctrl nor xdotool is available. "
                        "Install one with: sudo apt install wmctrl"
                    ),
                )
            # xdotool returns non-zero when no windows match — treat as empty list.
            return ToolResult(success=True, output={"windows": [], "count": 0})

        win_ids = search_result.stdout.splitlines()
        windows = []
        for wid in win_ids:
            wid = wid.strip()
            if not wid:
                continue
            name_result = _run(f"xdotool getwindowname {wid}")
            name = name_result.stdout.strip() if name_result.returncode == 0 else ""
            windows.append({"id": wid, "name": name})

        return ToolResult(success=True, output={"windows": windows, "count": len(windows)})


class X11ReadScreenTool(BaseTool):
    """Take a screenshot and send it to the AI vision model to interpret.

    Returns a text description of what the AI sees. Useful for verifying
    UI state, reading on-screen text, or understanding application state.
    """

    name = "x11_read_screen"
    description = (
        "Take a screenshot and send it to the AI vision model to interpret what is on screen. "
        "Returns a text description of what the AI sees. Use this to verify UI state, "
        "read text on screen, or understand application state."
    )
    permissions = ToolPermissions(shell=True, network=True, filesystem_write=True)

    parameters: dict[str, Any] = {
        "question": {
            "type": "string",
            "description": "What to ask the vision model about the screen.",
            "default": "Describe what you see on screen in detail",
        },
        "path": {
            "type": "string",
            "description": "File path to save the screenshot.",
            "default": "/tmp/screen_read.png",
        },
        "region": {
            "type": "string",
            "description": "Optional partial capture region as 'x,y,w,h'.",
        },
    }

    # ------------------------------------------------------------------
    # Screenshot helper
    # ------------------------------------------------------------------

    def _take_screenshot(self, path: str, region: str) -> str | None:
        """Take a screenshot; return an error string or None on success.

        If a Playwright browser session is active and no specific region
        is requested, captures from the browser page directly — this is
        more reliable than scrot which captures the desktop and may show
        the wrong window.  Falls back to scrot for desktop/region captures.
        """
        if not region:
            try:
                from missy.tools.builtin.browser_tools import _registry as browser_registry

                if browser_registry.screenshot_active(path):
                    return None
            except Exception:
                pass

        cmd = f"scrot -a {region} {path}" if region else f"scrot {path}"
        result = _run(cmd)
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            if "command not found" in err:
                return "scrot is not installed. Install it with: sudo apt install scrot"
            return f"scrot failed: {err}"
        return None

    # ------------------------------------------------------------------
    # Vision call via Ollama
    # ------------------------------------------------------------------

    def _call_ollama_vision(self, question: str, b64_image: str) -> str:
        """Send screenshot to Ollama vision model for interpretation.

        Uses the ``/api/chat`` endpoint with the ``images`` field, which
        accepts a list of base64-encoded images alongside the text prompt.
        """
        base_url = _get_ollama_base_url()
        body: dict[str, Any] = {
            "model": _OLLAMA_VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": question,
                    "images": [b64_image],
                }
            ],
            "stream": False,
        }

        resp = httpx.post(
            f"{base_url}/api/chat",
            json=body,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(
        self,
        *,
        question: str = "Describe what you see on screen in detail",
        path: str = "/tmp/screen_read.png",
        region: str = "",
        **_: Any,
    ) -> ToolResult:
        # 1. Take screenshot.
        err = self._take_screenshot(path, region)
        if err:
            return ToolResult(success=False, output=None, error=err)

        # 2. Read and base64-encode.
        try:
            with open(path, "rb") as fh:
                b64_image = base64.b64encode(fh.read()).decode("ascii")
        except OSError as exc:
            return ToolResult(success=False, output=None, error=f"Could not read screenshot: {exc}")

        # 3. Call Ollama vision model.
        try:
            description = self._call_ollama_vision(question, b64_image)
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Ollama vision HTTP {exc.response.status_code}: {exc.response.text[:200]}",
            )
        except httpx.ConnectError:
            return ToolResult(
                success=False,
                output=None,
                error=(
                    f"Cannot connect to Ollama at {_get_ollama_base_url()}. "
                    "Is the Ollama server running?"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, output=None, error=f"Vision call failed: {exc}")

        return ToolResult(
            success=True,
            output={"description": description, "screenshot_path": path, "question": question},
        )
