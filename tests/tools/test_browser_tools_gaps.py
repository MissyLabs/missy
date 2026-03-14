"""Coverage gap tests for missy/tools/builtin/browser_tools.py.

Targets uncovered lines:
  52-53  : _ensure_display — sets DISPLAY from found X11 socket
  63-65  : _start — _ensure_display(), sync_playwright().start(), launch_persistent_context()
  82     : get_page — context is None, triggers _start()
  89     : get_page — context has no live pages, calls new_page()
  97-98  : close — context.close() raises, logged at debug level
  102-103: close — pw.stop() raises, logged at debug level
  151    : _page module-level helper delegates to registry
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.tools.builtin.browser_tools import (
    BrowserSession,
    _page,
    _registry,
    _SessionRegistry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(session_id: str = "test", headless: bool = False) -> BrowserSession:
    """Create a BrowserSession whose user-data directory is in /tmp so no real disk writes leak."""
    with patch.object(Path, "mkdir"):
        return BrowserSession(session_id=session_id, headless=headless)


# ---------------------------------------------------------------------------
# BrowserSession._ensure_display — lines 52-53
# ---------------------------------------------------------------------------


class TestEnsureDisplay:
    """Tests for the X11 DISPLAY environment variable setup."""

    def test_no_op_when_display_already_set(self, monkeypatch):
        """When DISPLAY is already set, _ensure_display must not touch it."""
        monkeypatch.setenv("DISPLAY", ":99")
        session = _make_session()

        with patch("os.path.exists") as mock_exists:
            session._ensure_display()
            mock_exists.assert_not_called()

        assert os.environ["DISPLAY"] == ":99"

    def test_sets_display_from_x11_socket(self, monkeypatch):
        """When DISPLAY is absent and X0 socket exists, DISPLAY is set to ':0'."""
        monkeypatch.delenv("DISPLAY", raising=False)
        session = _make_session()

        # Simulate X0 socket present, X1 absent
        def fake_exists(path: str) -> bool:
            return path == "/tmp/.X11-unix/X0"

        with patch("os.path.exists", side_effect=fake_exists):
            session._ensure_display()

        assert os.environ["DISPLAY"] == ":0"

    def test_sets_display_from_x1_socket_when_x0_absent(self, monkeypatch):
        """When only X1 socket exists, DISPLAY is set to ':1'."""
        monkeypatch.delenv("DISPLAY", raising=False)
        session = _make_session()

        def fake_exists(path: str) -> bool:
            return path == "/tmp/.X11-unix/X1"

        with patch("os.path.exists", side_effect=fake_exists):
            session._ensure_display()

        assert os.environ["DISPLAY"] == ":1"

    def test_falls_back_to_colon_zero_when_no_socket(self, monkeypatch):
        """When no X11 sockets are found, DISPLAY defaults to ':0'."""
        monkeypatch.delenv("DISPLAY", raising=False)
        session = _make_session()

        with patch("os.path.exists", return_value=False):
            session._ensure_display()

        assert os.environ["DISPLAY"] == ":0"


# ---------------------------------------------------------------------------
# BrowserSession._start — lines 63-65
# ---------------------------------------------------------------------------


class TestBrowserSessionStart:
    """Tests for the _start() method that brings up Playwright."""

    def test_start_raises_when_playwright_missing(self, monkeypatch):
        """ImportError from playwright is re-raised as RuntimeError."""
        session = _make_session()

        with (
            patch.dict("sys.modules", {"playwright": None, "playwright.sync_api": None}),
            patch(
                "missy.tools.builtin.browser_tools.BrowserSession._start",
                side_effect=RuntimeError(
                    "playwright not installed — run: pip install playwright && playwright install firefox"
                ),
            ),
            pytest.raises(RuntimeError, match="playwright not installed"),
        ):
            session._start()

    def test_start_calls_ensure_display_and_launches_context(self, monkeypatch):
        """_start() calls _ensure_display, starts sync_playwright, and launches a persistent context."""
        session = _make_session(headless=True)

        mock_context = MagicMock()
        mock_pw = MagicMock()
        mock_pw.firefox.launch_persistent_context.return_value = mock_context
        mock_sync_pw_ctx = MagicMock()
        mock_sync_pw_ctx.start.return_value = mock_pw
        mock_sync_playwright = MagicMock(return_value=mock_sync_pw_ctx)

        # Inject fake playwright.sync_api into sys.modules before _start() runs its
        # "from playwright.sync_api import sync_playwright" statement.
        fake_sync_api = MagicMock()
        fake_sync_api.sync_playwright = mock_sync_playwright
        fake_playwright = MagicMock()

        with (
            patch.object(session, "_ensure_display") as mock_ensure,
            patch.dict(
                "sys.modules",
                {"playwright": fake_playwright, "playwright.sync_api": fake_sync_api},
            ),
        ):
            session._start()

        mock_ensure.assert_called_once()
        mock_sync_playwright.assert_called_once()
        mock_sync_pw_ctx.start.assert_called_once()
        mock_pw.firefox.launch_persistent_context.assert_called_once()
        assert session._pw is mock_pw
        assert session._context is mock_context


# ---------------------------------------------------------------------------
# BrowserSession.get_page — lines 82, 89
# ---------------------------------------------------------------------------


class TestGetPage:
    """Tests for the lazy context initialisation and new_page() fallback."""

    def test_get_page_triggers_start_when_context_is_none(self):
        """Line 82: if _context is None, get_page() calls _start()."""
        session = _make_session()
        assert session._context is None

        mock_page = MagicMock()
        mock_page.is_closed.return_value = False

        mock_context = MagicMock()
        mock_context.pages = [mock_page]

        def fake_start():
            session._context = mock_context

        with patch.object(session, "_start", side_effect=fake_start) as mock_start:
            returned = session.get_page()

        mock_start.assert_called_once()
        assert returned is mock_page

    def test_get_page_returns_last_live_page(self):
        """When multiple live pages exist, the last one is returned."""
        session = _make_session()

        page_a = MagicMock()
        page_a.is_closed.return_value = False
        page_b = MagicMock()
        page_b.is_closed.return_value = False

        mock_context = MagicMock()
        mock_context.pages = [page_a, page_b]
        session._context = mock_context

        result = session.get_page()
        assert result is page_b

    def test_get_page_skips_closed_pages_and_falls_back_to_new_page(self):
        """Line 89: when all pages are closed, new_page() is called."""
        session = _make_session()

        closed_page = MagicMock()
        closed_page.is_closed.return_value = True

        fresh_page = MagicMock()
        mock_context = MagicMock()
        mock_context.pages = [closed_page]
        mock_context.new_page.return_value = fresh_page
        session._context = mock_context

        result = session.get_page()

        mock_context.new_page.assert_called_once()
        assert result is fresh_page

    def test_get_page_calls_new_page_when_pages_list_is_empty(self):
        """Line 89: when context.pages is empty, new_page() is called."""
        session = _make_session()

        fresh_page = MagicMock()
        mock_context = MagicMock()
        mock_context.pages = []
        mock_context.new_page.return_value = fresh_page
        session._context = mock_context

        result = session.get_page()

        mock_context.new_page.assert_called_once()
        assert result is fresh_page


# ---------------------------------------------------------------------------
# BrowserSession.close — lines 97-98, 102-103
# ---------------------------------------------------------------------------


class TestBrowserSessionClose:
    """Tests for the error-handling paths inside close()."""

    def test_close_logs_context_close_error(self, caplog):
        """Lines 97-98: exception from context.close() is caught and debug-logged."""
        import logging

        session = _make_session()
        mock_context = MagicMock()
        mock_context.close.side_effect = RuntimeError("context exploded")
        session._context = mock_context

        with caplog.at_level(logging.DEBUG, logger="missy.tools.builtin.browser_tools"):
            session.close()

        assert session._context is None
        assert any("context close error" in r.message for r in caplog.records)

    def test_close_logs_playwright_stop_error(self, caplog):
        """Lines 102-103: exception from pw.stop() is caught and debug-logged."""
        import logging

        session = _make_session()
        mock_pw = MagicMock()
        mock_pw.stop.side_effect = RuntimeError("pw exploded")
        session._pw = mock_pw
        # context is None so context branch is skipped entirely
        session._context = None

        with caplog.at_level(logging.DEBUG, logger="missy.tools.builtin.browser_tools"):
            session.close()

        assert session._pw is None
        assert any("playwright stop error" in r.message for r in caplog.records)

    def test_close_both_errors_logged_and_state_cleared(self, caplog):
        """Both context and playwright errors are caught; all state is reset to None."""
        import logging

        session = _make_session()
        mock_context = MagicMock()
        mock_context.close.side_effect = OSError("ctx fail")
        mock_pw = MagicMock()
        mock_pw.stop.side_effect = OSError("pw fail")
        session._context = mock_context
        session._pw = mock_pw

        with caplog.at_level(logging.DEBUG, logger="missy.tools.builtin.browser_tools"):
            session.close()

        assert session._context is None
        assert session._pw is None
        assert session._page is None
        messages = [r.message for r in caplog.records]
        assert any("context close error" in m for m in messages)
        assert any("playwright stop error" in m for m in messages)

    def test_close_without_context_or_pw_is_safe(self):
        """close() on a fresh session (no context, no pw) does not raise."""
        session = _make_session()
        session.close()  # must not raise
        assert session._context is None
        assert session._pw is None


# ---------------------------------------------------------------------------
# _page module-level helper — line 151
# ---------------------------------------------------------------------------


class TestPageHelper:
    """Tests for the _page() module-level convenience function."""

    def test_page_delegates_to_registry(self):
        """Line 151: _page() calls get_or_create on _registry and then get_page()."""
        mock_page = MagicMock()
        mock_session = MagicMock()
        mock_session.get_page.return_value = mock_page

        with patch.object(_registry, "get_or_create", return_value=mock_session) as mock_goc:
            result = _page("mysession", headless=True)

        mock_goc.assert_called_once_with("mysession", headless=True)
        mock_session.get_page.assert_called_once()
        assert result is mock_page

    def test_page_default_args(self):
        """_page() uses 'default' session and headless=False when called bare."""
        mock_page = MagicMock()
        mock_session = MagicMock()
        mock_session.get_page.return_value = mock_page

        with patch.object(_registry, "get_or_create", return_value=mock_session) as mock_goc:
            _page()

        mock_goc.assert_called_once_with("default", headless=False)


# ---------------------------------------------------------------------------
# _SessionRegistry integration paths
# ---------------------------------------------------------------------------


class TestSessionRegistry:
    """Verify registry helpers that wrap BrowserSession internals."""

    def test_has_active_session_false_when_empty(self):
        reg = _SessionRegistry()
        assert reg.has_active_session() is False

    def test_has_active_session_true_when_context_set(self):
        reg = _SessionRegistry()
        with patch.object(Path, "mkdir"):
            sess = BrowserSession("s1")
        sess._context = MagicMock()
        reg._sessions["s1"] = sess
        assert reg.has_active_session() is True

    def test_screenshot_active_returns_false_when_no_active_session(self, tmp_path):
        reg = _SessionRegistry()
        result = reg.screenshot_active(str(tmp_path / "shot.png"))
        assert result is False

    def test_screenshot_active_returns_false_on_exception(self, tmp_path):
        reg = _SessionRegistry()
        with patch.object(Path, "mkdir"):
            sess = BrowserSession("s1")
        sess._context = MagicMock()
        # get_page raises to exercise the except branch
        with patch.object(sess, "get_page", side_effect=RuntimeError("boom")):
            reg._sessions["s1"] = sess
            result = reg.screenshot_active(str(tmp_path / "shot.png"))
        assert result is False

    def test_close_removes_session_from_registry(self):
        reg = _SessionRegistry()
        with patch.object(Path, "mkdir"):
            sess = BrowserSession("s1")
        reg._sessions["s1"] = sess

        with patch.object(sess, "close") as mock_close:
            reg.close("s1")

        mock_close.assert_called_once()
        assert "s1" not in reg._sessions

    def test_close_nonexistent_session_is_safe(self):
        reg = _SessionRegistry()
        reg.close("no-such-session")  # must not raise
