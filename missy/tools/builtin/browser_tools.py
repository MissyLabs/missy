"""Playwright-based browser automation tools for Missy.

Manages a single shared Firefox session across all tool calls within a
gateway session.  The module-level ``_state`` dict holds references to the
running playwright instance, browser, and active page so they survive
across individual tool invocations.

All tools require ``network=True`` permission because they drive a live
browser that makes outbound HTTP requests.

Install dependencies::

    pip install playwright
    playwright install firefox
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level browser state — shared across all tool instances
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "playwright": None,
    "browser": None,
    "page": None,
}

_PLAYWRIGHT_MISSING = (
    "playwright is not installed. "
    "Install it with: pip install playwright && playwright install firefox"
)
_NO_PAGE = "No browser open. Use browser_navigate first."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_display() -> None:
    """Set DISPLAY=:0 if not already set so headful Firefox can open."""
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"


def _get_page(headless: bool = False):
    """Return the active page, launching Firefox when not already running.

    Args:
        headless: When ``True`` run without a visible window.

    Returns:
        A Playwright ``Page`` object.

    Raises:
        ImportError: When playwright is not installed.
    """
    from playwright.sync_api import sync_playwright  # type: ignore[import]

    _ensure_display()

    if _state["page"] is not None:
        return _state["page"]

    if _state["playwright"] is None:
        _state["playwright"] = sync_playwright().start()

    pw = _state["playwright"]
    browser = pw.firefox.launch(
        headless=headless,
        args=["--no-remote"],
        firefox_user_prefs={
            "browser.sessionstore.resume_from_crash": 0,
            "browser.startup.page": 0,
            "browser.sessionstore.enabled": False,
        },
        env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")},
    )
    _state["browser"] = browser
    page = browser.new_page()
    _state["page"] = page
    logger.debug("Browser launched (headless=%s)", headless)
    return page


def _close_browser() -> None:
    """Shut down the browser and reset all module-level state."""
    try:
        if _state["browser"] is not None:
            _state["browser"].close()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Error closing browser: %s", exc)
    finally:
        _state["browser"] = None
        _state["page"] = None

    try:
        if _state["playwright"] is not None:
            _state["playwright"].stop()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Error stopping playwright: %s", exc)
    finally:
        _state["playwright"] = None


def _active_page():
    """Return the current page or ``None`` if no browser is open."""
    return _state.get("page")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


class BrowserNavigateTool(BaseTool):
    """Open a URL in the browser, launching Firefox if not already running.

    Handles session-restore dialogs automatically by dismissing them after
    navigation if they appear.
    """

    name = "browser_navigate"
    description = (
        "Open a URL in the browser. Launches Firefox if not already running. "
        "Handles session restore dialogs automatically."
    )
    permissions = ToolPermissions(network=True)

    parameters: dict[str, Any] = {
        "url": {
            "type": "string",
            "description": "The URL to navigate to.",
            "required": True,
        },
        "headless": {
            "type": "boolean",
            "description": "Run without a visible window (default False — shows the browser).",
            "default": False,
        },
        "wait_until": {
            "type": "string",
            "description": (
                "Navigation readiness event: 'load', 'domcontentloaded', or 'networkidle'. "
                "Defaults to 'domcontentloaded'."
            ),
            "default": "domcontentloaded",
        },
    }

    def execute(
        self,
        *,
        url: str,
        headless: bool = False,
        wait_until: str = "domcontentloaded",
        **_: Any,
    ) -> ToolResult:
        try:
            from playwright.sync_api import TimeoutError as PWTimeoutError  # type: ignore[import]
        except ImportError:
            return ToolResult(success=False, output=None, error=_PLAYWRIGHT_MISSING)

        try:
            page = _get_page(headless=headless)
            page.goto(url, wait_until=wait_until, timeout=30_000)

            # Auto-dismiss Firefox session restore dialog
            try:
                restore_btn = page.get_by_text("Start New Session", exact=False)
                if restore_btn.count() > 0:
                    restore_btn.first.click(timeout=2_000)
                    logger.debug("Dismissed session restore dialog")
            except PWTimeoutError:
                pass  # No restore dialog present
            except Exception:  # noqa: BLE001
                pass

            return ToolResult(
                success=True,
                output={"title": page.title(), "url": page.url},
            )
        except ImportError:
            return ToolResult(success=False, output=None, error=_PLAYWRIGHT_MISSING)
        except PWTimeoutError as exc:
            return ToolResult(success=False, output=None, error=f"Navigation timed out: {exc}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, output=None, error=f"Navigation failed: {exc}")


class BrowserClickTool(BaseTool):
    """Click a browser element by CSS selector, visible text, or ARIA role.

    At least one of ``selector``, ``text``, or ``role`` must be supplied.
    """

    name = "browser_click"
    description = (
        "Click an element in the browser by CSS selector, visible text, or role. "
        "Examples: selector='#submit-btn', text='Sign In', role='button' name='Accept'."
    )
    permissions = ToolPermissions(network=True)

    parameters: dict[str, Any] = {
        "selector": {
            "type": "string",
            "description": "CSS selector for the element to click.",
        },
        "text": {
            "type": "string",
            "description": "Visible text content of the element to click.",
        },
        "role": {
            "type": "string",
            "description": "ARIA role of the element, e.g. 'button', 'link', 'checkbox'.",
        },
        "name": {
            "type": "string",
            "description": "Accessible name used together with 'role' to narrow the match.",
        },
        "timeout_ms": {
            "type": "integer",
            "description": "Maximum time to wait for the element in milliseconds (default 5000).",
            "default": 5000,
        },
    }

    def execute(
        self,
        *,
        selector: str = "",
        text: str = "",
        role: str = "",
        name: str = "",
        timeout_ms: int = 5_000,
        **_: Any,
    ) -> ToolResult:
        try:
            from playwright.sync_api import TimeoutError as PWTimeoutError  # type: ignore[import]
        except ImportError:
            return ToolResult(success=False, output=None, error=_PLAYWRIGHT_MISSING)

        page = _active_page()
        if page is None:
            return ToolResult(success=False, output=None, error=_NO_PAGE)

        if not selector and not text and not role:
            return ToolResult(
                success=False,
                output=None,
                error="At least one of 'selector', 'text', or 'role' must be provided.",
            )

        try:
            if text:
                page.get_by_text(text).first.click(timeout=timeout_ms)
                clicked_by = f"text={text!r}"
            elif role and name:
                page.get_by_role(role, name=name).click(timeout=timeout_ms)  # type: ignore[call-arg]
                clicked_by = f"role={role!r} name={name!r}"
            elif role:
                page.get_by_role(role).click(timeout=timeout_ms)  # type: ignore[call-arg]
                clicked_by = f"role={role!r}"
            else:
                page.locator(selector).click(timeout=timeout_ms)
                clicked_by = f"selector={selector!r}"

            return ToolResult(
                success=True,
                output={"clicked_by": clicked_by, "url": page.url},
            )
        except PWTimeoutError as exc:
            return ToolResult(success=False, output=None, error=f"Element not found (timeout): {exc}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, output=None, error=f"Click failed: {exc}")


class BrowserFillTool(BaseTool):
    """Fill a text input or textarea using CSS selector, label, or placeholder."""

    name = "browser_fill"
    description = (
        "Fill a text input or textarea in the browser. "
        "Use selector (CSS), label text, or placeholder text to identify the field."
    )
    permissions = ToolPermissions(network=True)

    parameters: dict[str, Any] = {
        "value": {
            "type": "string",
            "description": "Text to fill into the field.",
            "required": True,
        },
        "selector": {
            "type": "string",
            "description": "CSS selector for the input element.",
        },
        "label": {
            "type": "string",
            "description": "Label text associated with the input.",
        },
        "placeholder": {
            "type": "string",
            "description": "Placeholder text of the input.",
        },
    }

    def execute(
        self,
        *,
        value: str,
        selector: str = "",
        label: str = "",
        placeholder: str = "",
        **_: Any,
    ) -> ToolResult:
        try:
            from playwright.sync_api import TimeoutError as PWTimeoutError  # type: ignore[import]
        except ImportError:
            return ToolResult(success=False, output=None, error=_PLAYWRIGHT_MISSING)

        page = _active_page()
        if page is None:
            return ToolResult(success=False, output=None, error=_NO_PAGE)

        if not selector and not label and not placeholder:
            return ToolResult(
                success=False,
                output=None,
                error="At least one of 'selector', 'label', or 'placeholder' must be provided.",
            )

        try:
            if label:
                locator = page.get_by_label(label)
                filled_by = f"label={label!r}"
            elif placeholder:
                locator = page.get_by_placeholder(placeholder)
                filled_by = f"placeholder={placeholder!r}"
            else:
                locator = page.locator(selector)
                filled_by = f"selector={selector!r}"

            locator.fill(value)
            return ToolResult(
                success=True,
                output={"filled_by": filled_by, "value_length": len(value)},
            )
        except PWTimeoutError as exc:
            return ToolResult(success=False, output=None, error=f"Field not found (timeout): {exc}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, output=None, error=f"Fill failed: {exc}")


class BrowserScreenshotTool(BaseTool):
    """Take a screenshot of the current browser page or a specific element."""

    name = "browser_screenshot"
    description = (
        "Take a screenshot of the current browser page. "
        "Optionally screenshot a specific element."
    )
    permissions = ToolPermissions(network=True)

    parameters: dict[str, Any] = {
        "path": {
            "type": "string",
            "description": "File path to save the screenshot PNG.",
            "default": "/tmp/browser_screenshot.png",
        },
        "full_page": {
            "type": "boolean",
            "description": "Capture the full scrollable page height (default False).",
            "default": False,
        },
        "selector": {
            "type": "string",
            "description": "Optional CSS selector of an element to screenshot instead of the full page.",
        },
    }

    def execute(
        self,
        *,
        path: str = "/tmp/browser_screenshot.png",
        full_page: bool = False,
        selector: str = "",
        **_: Any,
    ) -> ToolResult:
        try:
            from playwright.sync_api import TimeoutError as PWTimeoutError  # type: ignore[import]
        except ImportError:
            return ToolResult(success=False, output=None, error=_PLAYWRIGHT_MISSING)

        page = _active_page()
        if page is None:
            return ToolResult(success=False, output=None, error=_NO_PAGE)

        try:
            if selector:
                page.locator(selector).screenshot(path=path)
                captured = f"element {selector!r}"
            else:
                page.screenshot(path=path, full_page=full_page)
                captured = "full page" if full_page else "viewport"

            return ToolResult(
                success=True,
                output={"path": path, "captured": captured, "title": page.title()},
            )
        except PWTimeoutError as exc:
            return ToolResult(success=False, output=None, error=f"Screenshot timed out: {exc}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, output=None, error=f"Screenshot failed: {exc}")


class BrowserGetContentTool(BaseTool):
    """Read text or HTML content from the current page or a specific element."""

    name = "browser_get_content"
    description = (
        "Get the text content or HTML of the current page or a specific element. "
        "Useful for reading page content, checking form values, or verifying page state."
    )
    permissions = ToolPermissions(network=True)

    parameters: dict[str, Any] = {
        "selector": {
            "type": "string",
            "description": "CSS selector of the element to read (defaults to 'body').",
        },
        "content_type": {
            "type": "string",
            "description": "Return 'text' for inner text or 'html' for inner HTML (default 'text').",
            "default": "text",
        },
        "max_length": {
            "type": "integer",
            "description": "Maximum characters to return (default 5000).",
            "default": 5000,
        },
    }

    def execute(
        self,
        *,
        selector: str = "",
        content_type: str = "text",
        max_length: int = 5_000,
        **_: Any,
    ) -> ToolResult:
        try:
            from playwright.sync_api import TimeoutError as PWTimeoutError  # type: ignore[import]
        except ImportError:
            return ToolResult(success=False, output=None, error=_PLAYWRIGHT_MISSING)

        page = _active_page()
        if page is None:
            return ToolResult(success=False, output=None, error=_NO_PAGE)

        target = selector or "body"
        try:
            locator = page.locator(target)
            if content_type == "html":
                raw = locator.inner_html()
            else:
                raw = locator.inner_text()

            truncated = len(raw) > max_length
            content = raw[:max_length]

            return ToolResult(
                success=True,
                output={
                    "content": content,
                    "content_type": content_type,
                    "truncated": truncated,
                    "total_length": len(raw),
                    "selector": target,
                },
            )
        except PWTimeoutError as exc:
            return ToolResult(success=False, output=None, error=f"Element not found (timeout): {exc}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, output=None, error=f"Get content failed: {exc}")


class BrowserEvaluateTool(BaseTool):
    """Execute JavaScript in the browser page and return the result."""

    name = "browser_evaluate"
    description = (
        "Run JavaScript in the browser page and return the result. "
        "Useful for complex interactions, reading state, or manipulating the page."
    )
    permissions = ToolPermissions(network=True)

    parameters: dict[str, Any] = {
        "script": {
            "type": "string",
            "description": "JavaScript expression or function body to evaluate.",
            "required": True,
        },
    }

    def execute(self, *, script: str, **_: Any) -> ToolResult:
        try:
            from playwright.sync_api import TimeoutError as PWTimeoutError  # type: ignore[import]
        except ImportError:
            return ToolResult(success=False, output=None, error=_PLAYWRIGHT_MISSING)

        page = _active_page()
        if page is None:
            return ToolResult(success=False, output=None, error=_NO_PAGE)

        try:
            result = page.evaluate(script)
            return ToolResult(
                success=True,
                output={"result": str(result), "url": page.url},
            )
        except PWTimeoutError as exc:
            return ToolResult(success=False, output=None, error=f"Evaluate timed out: {exc}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, output=None, error=f"JavaScript evaluation failed: {exc}")


class BrowserWaitTool(BaseTool):
    """Wait for an element, URL pattern, visible text, or a fixed number of seconds."""

    name = "browser_wait"
    description = (
        "Wait for an element to appear, a URL pattern, or a fixed time. "
        "Use before interacting with dynamic content."
    )
    permissions = ToolPermissions(network=True)

    parameters: dict[str, Any] = {
        "for_selector": {
            "type": "string",
            "description": "CSS selector to wait for.",
        },
        "for_url": {
            "type": "string",
            "description": "URL pattern (string or glob) to wait for.",
        },
        "for_text": {
            "type": "string",
            "description": "Visible text to wait for on the page.",
        },
        "seconds": {
            "type": "number",
            "description": "Fallback fixed sleep in seconds (default 2, max 30).",
            "default": 2,
        },
    }

    def execute(
        self,
        *,
        for_selector: str = "",
        for_url: str = "",
        for_text: str = "",
        seconds: float = 2.0,
        **_: Any,
    ) -> ToolResult:
        try:
            from playwright.sync_api import TimeoutError as PWTimeoutError  # type: ignore[import]
        except ImportError:
            return ToolResult(success=False, output=None, error=_PLAYWRIGHT_MISSING)

        page = _active_page()
        if page is None:
            return ToolResult(success=False, output=None, error=_NO_PAGE)

        waited_for: Optional[str] = None
        try:
            if for_selector:
                page.wait_for_selector(for_selector, timeout=30_000)
                waited_for = f"selector={for_selector!r}"
            elif for_url:
                page.wait_for_url(for_url, timeout=30_000)
                waited_for = f"url={for_url!r}"
            elif for_text:
                page.get_by_text(for_text).wait_for(timeout=30_000)
                waited_for = f"text={for_text!r}"
            else:
                clamped = min(max(float(seconds), 0), 30)
                time.sleep(clamped)
                waited_for = f"{clamped:.1f}s sleep"

            return ToolResult(
                success=True,
                output={"waited_for": waited_for, "url": page.url},
            )
        except PWTimeoutError as exc:
            return ToolResult(success=False, output=None, error=f"Wait timed out: {exc}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, output=None, error=f"Wait failed: {exc}")


class BrowserCloseTool(BaseTool):
    """Close the browser and clean up the session."""

    name = "browser_close"
    description = "Close the browser and clean up the session."
    permissions = ToolPermissions(network=True)

    parameters: dict[str, Any] = {}

    def execute(self, **_: Any) -> ToolResult:
        try:
            from playwright.sync_api import TimeoutError  # noqa: F401  # type: ignore[import]
        except ImportError:
            return ToolResult(success=False, output=None, error=_PLAYWRIGHT_MISSING)

        _close_browser()
        return ToolResult(success=True, output="Browser closed.")


class BrowserGetUrlTool(BaseTool):
    """Return the current browser URL and page title."""

    name = "browser_get_url"
    description = "Get the current browser URL and page title."
    permissions = ToolPermissions(network=True)

    parameters: dict[str, Any] = {}

    def execute(self, **_: Any) -> ToolResult:
        try:
            from playwright.sync_api import TimeoutError  # noqa: F401  # type: ignore[import]
        except ImportError:
            return ToolResult(success=False, output=None, error=_PLAYWRIGHT_MISSING)

        page = _active_page()
        if page is None:
            return ToolResult(success=False, output=None, error=_NO_PAGE)

        try:
            return ToolResult(
                success=True,
                output={"url": page.url, "title": page.title()},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, output=None, error=f"Could not read URL: {exc}")
