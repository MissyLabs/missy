"""Playwright-based browser automation tools for Missy.

Uses launch_persistent_context() — the correct Playwright pattern for
agent use: cookies/localStorage persist across tool calls, no session
restore dialogs, proper per-session isolation.

Install: pip install playwright && playwright install firefox
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

logger = logging.getLogger(__name__)

_SESSIONS_DIR = Path("~/.missy/browser_sessions").expanduser()
_FIREFOX_PREFS = {
    "browser.sessionstore.resume_from_crash": 0,
    "browser.sessionstore.enabled": False,
    "browser.startup.page": 0,
    "browser.tabs.warnOnClose": False,
    "toolkit.startup.max_resumed_crashes": -1,
}


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class BrowserSession:
    def __init__(self, session_id: str, headless: bool = False) -> None:
        self.session_id = session_id
        self.headless = headless
        self._lock = threading.Lock()
        self._pw = None
        self._context = None
        self._page = None
        self._user_data_dir = _SESSIONS_DIR / session_id
        self._user_data_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_display(self) -> None:
        if not os.environ.get("DISPLAY"):
            for sock in ["/tmp/.X11-unix/X0", "/tmp/.X11-unix/X1"]:
                if os.path.exists(sock):
                    os.environ["DISPLAY"] = f":{sock.split('X')[-1]}"
                    return
            os.environ["DISPLAY"] = ":0"

    def _start(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise RuntimeError("playwright not installed — run: pip install playwright && playwright install firefox")
        self._ensure_display()
        self._pw = sync_playwright().start()
        self._context = self._pw.firefox.launch_persistent_context(
            user_data_dir=str(self._user_data_dir),
            headless=self.headless,
            args=["--no-remote"],
            firefox_user_prefs=_FIREFOX_PREFS,
            env={**os.environ},
        )

    def get_page(self):
        """Return the most recent open page in the context.

        Always checks the context for the latest page rather than
        returning a stale cached reference — Firefox may open new tabs
        via session restore, redirects, or target=_blank links.
        """
        with self._lock:
            if self._context is None:
                self._start()
            pages = self._context.pages
            # Filter out closed pages
            live = [p for p in pages if not p.is_closed()]
            if live:
                self._page = live[-1]
            else:
                self._page = self._context.new_page()
            return self._page

    def close(self) -> None:
        with self._lock:
            try:
                if self._context:
                    self._context.close()
            except Exception:
                pass
            try:
                if self._pw:
                    self._pw.stop()
            except Exception:
                pass
            self._context = None
            self._page = None
            self._pw = None


class _SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, BrowserSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str, headless: bool = False) -> BrowserSession:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = BrowserSession(session_id, headless=headless)
            return self._sessions[session_id]

    def close(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                self._sessions.pop(session_id).close()


_registry = _SessionRegistry()


def _page(session_id: str = "default", headless: bool = False):
    return _registry.get_or_create(session_id, headless=headless).get_page()


def _err(exc: Exception) -> ToolResult:
    return ToolResult(success=False, output=None, error=str(exc))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class BrowserNavigateTool(BaseTool):
    name = "browser_navigate"
    description = (
        "Open a URL in Firefox. Launches automatically on first call — no need to open Firefox manually. "
        "For searching use: url='https://duckduckgo.com/?q=your+query'. "
        "Cookies and login state persist across calls in the same session."
    )
    permissions = ToolPermissions(network=True)
    parameters = {
        "url": {"type": "string", "description": "Full URL to navigate to.", "required": True},
        "headless": {"type": "boolean", "description": "Hide browser window (default False)."},
        "wait_until": {"type": "string", "description": "'load', 'domcontentloaded', 'networkidle' (default 'domcontentloaded')."},
        "session_id": {"type": "string", "description": "Session name (default 'default')."},
    }

    def execute(self, *, url: str, headless: bool = False,
                wait_until: str = "domcontentloaded", session_id: str = "default", **_kw) -> ToolResult:
        try:
            pg = _page(session_id, headless=headless)
            pg.goto(url, wait_until=wait_until, timeout=30_000)
            return ToolResult(success=True, output=f"URL: {pg.url}\nTitle: {pg.title()}", error=None)
        except Exception as exc:
            return _err(exc)


class BrowserClickTool(BaseTool):
    name = "browser_click"
    description = (
        "Click an element on the page by visible text, CSS selector, or ARIA role. "
        "Examples: text='Sign In', selector='#submit', role='button' name='OK'."
    )
    permissions = ToolPermissions(network=True)
    parameters = {
        "text": {"type": "string", "description": "Visible text of element to click."},
        "selector": {"type": "string", "description": "CSS selector."},
        "role": {"type": "string", "description": "ARIA role (e.g. 'button', 'link')."},
        "name": {"type": "string", "description": "Accessible name, used with role."},
        "timeout_ms": {"type": "integer", "description": "Timeout ms (default 5000)."},
        "session_id": {"type": "string", "description": "Session name (default 'default')."},
    }

    def execute(self, *, text: str = "", selector: str = "", role: str = "",
                name: str = "", timeout_ms: int = 5000, session_id: str = "default", **_kw) -> ToolResult:
        try:
            pg = _page(session_id)
            t = timeout_ms
            if text:
                pg.get_by_text(text, exact=False).first.click(timeout=t)
            elif role and name:
                pg.get_by_role(role, name=name).click(timeout=t)
            elif role:
                pg.get_by_role(role).first.click(timeout=t)
            elif selector:
                pg.locator(selector).first.click(timeout=t)
            else:
                return ToolResult(success=False, output=None, error="Provide text, selector, or role.")
            return ToolResult(success=True, output=f"Clicked. URL: {pg.url}", error=None)
        except Exception as exc:
            return _err(exc)


class BrowserFillTool(BaseTool):
    name = "browser_fill"
    description = "Fill a text input by label, placeholder, or CSS selector."
    permissions = ToolPermissions(network=True)
    parameters = {
        "value": {"type": "string", "description": "Text to type.", "required": True},
        "selector": {"type": "string", "description": "CSS selector."},
        "label": {"type": "string", "description": "Associated label text."},
        "placeholder": {"type": "string", "description": "Placeholder text."},
        "press_enter": {"type": "boolean", "description": "Press Enter after filling (default False)."},
        "session_id": {"type": "string", "description": "Session name (default 'default')."},
    }

    def execute(self, *, value: str, selector: str = "", label: str = "",
                placeholder: str = "", press_enter: bool = False,
                session_id: str = "default", **_kw) -> ToolResult:
        try:
            pg = _page(session_id)
            if label:
                loc = pg.get_by_label(label, exact=False).first
            elif placeholder:
                loc = pg.get_by_placeholder(placeholder, exact=False).first
            elif selector:
                loc = pg.locator(selector).first
            else:
                return ToolResult(success=False, output=None, error="Provide selector, label, or placeholder.")
            loc.fill(value)
            if press_enter:
                loc.press("Enter")
            return ToolResult(success=True, output="Field filled.", error=None)
        except Exception as exc:
            return _err(exc)


class BrowserScreenshotTool(BaseTool):
    name = "browser_screenshot"
    description = "Take a screenshot of the current browser page."
    permissions = ToolPermissions(network=True)
    parameters = {
        "path": {"type": "string", "description": "Save path (default /tmp/browser_screenshot.png)."},
        "full_page": {"type": "boolean", "description": "Capture full scrollable page (default False)."},
        "selector": {"type": "string", "description": "CSS selector to screenshot a specific element."},
        "session_id": {"type": "string", "description": "Session name (default 'default')."},
    }

    def execute(self, *, path: str = "/tmp/browser_screenshot.png", full_page: bool = False,
                selector: str = "", session_id: str = "default", **_kw) -> ToolResult:
        try:
            pg = _page(session_id)
            if selector:
                pg.locator(selector).first.screenshot(path=path)
            else:
                pg.screenshot(path=path, full_page=full_page)
            size = Path(path).stat().st_size
            return ToolResult(success=True, output=f"Screenshot: {path} ({size:,} bytes) — {pg.title()}", error=None)
        except Exception as exc:
            return _err(exc)


class BrowserGetContentTool(BaseTool):
    name = "browser_get_content"
    description = "Get text or HTML from the current page or a specific element."
    permissions = ToolPermissions(network=True)
    parameters = {
        "selector": {"type": "string", "description": "CSS selector (default: body)."},
        "content_type": {"type": "string", "description": "'text' or 'html' (default 'text')."},
        "max_length": {"type": "integer", "description": "Max chars to return (default 5000)."},
        "session_id": {"type": "string", "description": "Session name (default 'default')."},
    }

    def execute(self, *, selector: str = "body", content_type: str = "text",
                max_length: int = 5000, session_id: str = "default", **_kw) -> ToolResult:
        try:
            pg = _page(session_id)
            loc = pg.locator(selector).first
            content = loc.inner_text() if content_type == "text" else loc.inner_html()
            if len(content) > max_length:
                content = content[:max_length] + f"\n[…{len(content):,} total chars]"
            return ToolResult(success=True, output=content, error=None)
        except Exception as exc:
            return _err(exc)


class BrowserEvaluateTool(BaseTool):
    name = "browser_evaluate"
    description = "Run JavaScript in the browser and return the result."
    permissions = ToolPermissions(network=True)
    parameters = {
        "script": {"type": "string", "description": "JS expression, e.g. 'document.title'.", "required": True},
        "session_id": {"type": "string", "description": "Session name (default 'default')."},
    }

    def execute(self, *, script: str, session_id: str = "default", **_kw) -> ToolResult:
        try:
            pg = _page(session_id)
            return ToolResult(success=True, output=str(pg.evaluate(script)), error=None)
        except Exception as exc:
            return _err(exc)


class BrowserWaitTool(BaseTool):
    name = "browser_wait"
    description = "Wait for an element, URL change, visible text, or a fixed time."
    permissions = ToolPermissions(network=True)
    parameters = {
        "for_selector": {"type": "string", "description": "CSS selector to wait to appear."},
        "for_url": {"type": "string", "description": "URL glob to wait for."},
        "for_text": {"type": "string", "description": "Visible text to wait for."},
        "seconds": {"type": "number", "description": "Fixed wait (max 30, default 2)."},
        "session_id": {"type": "string", "description": "Session name (default 'default')."},
    }

    def execute(self, *, for_selector: str = "", for_url: str = "", for_text: str = "",
                seconds: float = 2.0, session_id: str = "default", **_kw) -> ToolResult:
        try:
            pg = _page(session_id)
            if for_selector:
                pg.wait_for_selector(for_selector, timeout=30_000)
                return ToolResult(success=True, output=f"'{for_selector}' appeared.", error=None)
            if for_url:
                pg.wait_for_url(for_url, timeout=30_000)
                return ToolResult(success=True, output=f"URL: {pg.url}", error=None)
            if for_text:
                pg.get_by_text(for_text, exact=False).first.wait_for(timeout=30_000)
                return ToolResult(success=True, output=f"Text '{for_text}' appeared.", error=None)
            time.sleep(min(float(seconds), 30))
            return ToolResult(success=True, output=f"Waited {seconds}s.", error=None)
        except Exception as exc:
            return _err(exc)


class BrowserGetUrlTool(BaseTool):
    name = "browser_get_url"
    description = "Get the current browser URL and page title."
    permissions = ToolPermissions(network=True)
    parameters = {"session_id": {"type": "string", "description": "Session name (default 'default')."}}

    def execute(self, *, session_id: str = "default", **_kw) -> ToolResult:
        try:
            pg = _page(session_id)
            return ToolResult(success=True, output=f"URL: {pg.url}\nTitle: {pg.title()}", error=None)
        except Exception as exc:
            return _err(exc)


class BrowserCloseTool(BaseTool):
    name = "browser_close"
    description = "Close the browser session."
    permissions = ToolPermissions(network=True)
    parameters = {"session_id": {"type": "string", "description": "Session name (default 'default')."}}

    def execute(self, *, session_id: str = "default", **_kw) -> ToolResult:
        _registry.close(session_id)
        return ToolResult(success=True, output=f"Session '{session_id}' closed.", error=None)
