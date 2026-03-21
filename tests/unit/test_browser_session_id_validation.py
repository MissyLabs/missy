"""Browser session_id validation to prevent directory traversal."""

from __future__ import annotations

import pytest


class TestBrowserSessionIdValidation:
    """Ensure session_id is validated to prevent path traversal."""

    def test_valid_session_id(self):
        import re

        session_id = "valid-session_123"
        assert re.match(r"^[a-zA-Z0-9_\-]+$", session_id)

    def test_traversal_attack_rejected(self):
        from missy.tools.builtin.browser_tools import BrowserSession

        with pytest.raises(ValueError, match="Invalid session_id"):
            BrowserSession("../../etc/passwd")

    def test_slash_rejected(self):
        from missy.tools.builtin.browser_tools import BrowserSession

        with pytest.raises(ValueError, match="Invalid session_id"):
            BrowserSession("session/traversal")

    def test_dot_dot_rejected(self):
        from missy.tools.builtin.browser_tools import BrowserSession

        with pytest.raises(ValueError, match="Invalid session_id"):
            BrowserSession("..session")

    def test_space_rejected(self):
        from missy.tools.builtin.browser_tools import BrowserSession

        with pytest.raises(ValueError, match="Invalid session_id"):
            BrowserSession("session with spaces")

    def test_empty_rejected(self):
        from missy.tools.builtin.browser_tools import BrowserSession

        with pytest.raises(ValueError, match="Invalid session_id"):
            BrowserSession("")

    def test_null_byte_rejected(self):
        from missy.tools.builtin.browser_tools import BrowserSession

        with pytest.raises(ValueError, match="Invalid session_id"):
            BrowserSession("session\x00id")
