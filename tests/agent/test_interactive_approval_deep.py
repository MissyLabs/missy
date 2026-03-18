"""Deep coverage tests for missy.agent.interactive_approval.

Targets the lines not exercised by test_interactive_approval.py:

  Lines 100-128  (_do_prompt — all branches including exception paths)
  Lines 139-142  (_is_tty — exception fallback)

Because Console and Panel are imported *inside* _do_prompt (local imports),
they must be patched at their source locations — rich.console.Console and
rich.panel.Panel — rather than as attributes of the interactive_approval module.

The ImportError path is covered by patching sys.modules so the import raises.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.interactive_approval import InteractiveApproval


@pytest.fixture
def approval() -> InteractiveApproval:
    return InteractiveApproval()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _console_returning(response: str) -> MagicMock:
    """Return a Console mock whose .input() returns *response*."""
    console = MagicMock()
    console.input.return_value = response
    return console


def _patched_rich(console: MagicMock):
    """Context manager: replace rich.console.Console and rich.panel.Panel."""
    return (
        patch("rich.console.Console", return_value=console),
        patch("rich.panel.Panel"),
    )


# ---------------------------------------------------------------------------
# _do_prompt — exception paths (lines 116-117)
# ---------------------------------------------------------------------------

class TestDoPromptExceptions:
    """_do_prompt must return False for ImportError, EOFError, and KeyboardInterrupt."""

    def test_import_error_returns_false(self, approval: InteractiveApproval) -> None:
        """When rich is unavailable, the ImportError is caught and False is returned."""
        # Setting the module slot to None forces 'from rich.console import Console'
        # to raise ImportError inside the function body.
        with patch.dict("sys.modules", {"rich.console": None, "rich.panel": None}):
            result = approval._do_prompt("net", "https://example.com")
        assert result is False

    def test_eoferror_returns_false(self, approval: InteractiveApproval) -> None:
        """When console.input() raises EOFError, _do_prompt returns False."""
        console = MagicMock()
        console.input.side_effect = EOFError

        console_patch, panel_patch = _patched_rich(console)
        with console_patch, panel_patch:
            result = approval._do_prompt("net", "https://example.com")

        assert result is False

    def test_keyboard_interrupt_returns_false(self, approval: InteractiveApproval) -> None:
        """When the user presses Ctrl-C during input, _do_prompt returns False."""
        console = MagicMock()
        console.input.side_effect = KeyboardInterrupt

        console_patch, panel_patch = _patched_rich(console)
        with console_patch, panel_patch:
            result = approval._do_prompt("net", "https://example.com")

        assert result is False


# ---------------------------------------------------------------------------
# _do_prompt — "y" response path (lines 119-120)
# ---------------------------------------------------------------------------

class TestDoPromptYesResponse:
    """_do_prompt returns True when the operator enters 'y'."""

    def test_y_returns_true(self, approval: InteractiveApproval) -> None:
        console = _console_returning("y")

        console_patch, panel_patch = _patched_rich(console)
        with console_patch, panel_patch:
            result = approval._do_prompt("tool_call", "run ls")

        assert result is True

    def test_y_does_not_store_in_remembered(self, approval: InteractiveApproval) -> None:
        """'y' (allow once) must not persist a remembered decision."""
        console = _console_returning("y")

        console_patch, panel_patch = _patched_rich(console)
        with console_patch, panel_patch:
            approval._do_prompt("tool_call", "run ls")

        assert approval.check_remembered("tool_call", "run ls") is None

    def test_y_uppercase_is_normalised_to_true(self, approval: InteractiveApproval) -> None:
        """Surrounding whitespace and uppercase are stripped/lowered; 'Y' → True."""
        console = _console_returning("  Y  ")

        console_patch, panel_patch = _patched_rich(console)
        with console_patch, panel_patch:
            result = approval._do_prompt("tool_call", "run ls")

        assert result is True


# ---------------------------------------------------------------------------
# _do_prompt — deny paths (line 128)
# ---------------------------------------------------------------------------

class TestDoPromptDenyResponse:
    """_do_prompt returns False for 'n', empty string, and any other non-approval input."""

    @pytest.mark.parametrize("raw_input", ["n", "N", "", "  ", "nope", "yes", "2", "!"])
    def test_non_approval_input_returns_false(
        self, approval: InteractiveApproval, raw_input: str
    ) -> None:
        console = _console_returning(raw_input)

        console_patch, panel_patch = _patched_rich(console)
        with console_patch, panel_patch:
            result = approval._do_prompt("net", "https://example.com")

        assert result is False

    @pytest.mark.parametrize("raw_input", ["n", "N", "", "nope"])
    def test_deny_does_not_store_in_remembered(
        self, approval: InteractiveApproval, raw_input: str
    ) -> None:
        """Deny responses must not leave a remembered entry."""
        console = _console_returning(raw_input)

        console_patch, panel_patch = _patched_rich(console)
        with console_patch, panel_patch:
            approval._do_prompt("net", "https://example.com")

        assert approval.check_remembered("net", "https://example.com") is None


# ---------------------------------------------------------------------------
# _is_tty — exception fallback (lines 141-142)
# ---------------------------------------------------------------------------

class TestIsTtyExceptionFallback:
    """_is_tty returns False when sys.stdin.isatty() raises any exception."""

    def test_attribute_error_returns_false(self) -> None:
        bad_stdin = MagicMock()
        bad_stdin.isatty.side_effect = AttributeError("no isatty")

        with patch.object(sys, "stdin", bad_stdin):
            result = InteractiveApproval._is_tty()

        assert result is False

    def test_os_error_returns_false(self) -> None:
        bad_stdin = MagicMock()
        bad_stdin.isatty.side_effect = OSError("bad fd")

        with patch.object(sys, "stdin", bad_stdin):
            result = InteractiveApproval._is_tty()

        assert result is False

    def test_runtime_error_returns_false(self) -> None:
        bad_stdin = MagicMock()
        bad_stdin.isatty.side_effect = RuntimeError("unexpected")

        with patch.object(sys, "stdin", bad_stdin):
            result = InteractiveApproval._is_tty()

        assert result is False

    def test_returns_true_when_stdin_is_tty(self) -> None:
        """Baseline: returns True when stdin.isatty() is True."""
        tty_stdin = MagicMock()
        tty_stdin.isatty.return_value = True

        with patch.object(sys, "stdin", tty_stdin):
            assert InteractiveApproval._is_tty() is True

    def test_returns_false_when_stdin_is_not_tty(self) -> None:
        """Baseline: returns False when stdin.isatty() is False (e.g. pipe)."""
        pipe_stdin = MagicMock()
        pipe_stdin.isatty.return_value = False

        with patch.object(sys, "stdin", pipe_stdin):
            assert InteractiveApproval._is_tty() is False


# ---------------------------------------------------------------------------
# Integration: prompt_user → _do_prompt with real rich mocks
# ---------------------------------------------------------------------------

class TestPromptUserIntegration:
    """End-to-end prompt_user flows that exercise _do_prompt via actual dispatch."""

    def test_prompt_user_tty_y_response_returns_true(
        self, approval: InteractiveApproval
    ) -> None:
        """prompt_user returns True when TTY is present and operator enters 'y'."""
        console = _console_returning("y")

        with patch.object(InteractiveApproval, "_is_tty", return_value=True):
            console_patch, panel_patch = _patched_rich(console)
            with console_patch, panel_patch:
                result = approval.prompt_user("tool_call", "run ls")

        assert result is True
        # "y" must NOT leave a remembered entry.
        assert approval.check_remembered("tool_call", "run ls") is None

    def test_prompt_user_tty_a_stores_remembered_and_skips_second_prompt(
        self, approval: InteractiveApproval
    ) -> None:
        """'a' stores the decision; subsequent calls return True without prompting."""
        console = _console_returning("a")

        with patch.object(InteractiveApproval, "_is_tty", return_value=True):
            console_patch, panel_patch = _patched_rich(console)
            with console_patch, panel_patch:
                first_result = approval.prompt_user("net", "https://example.com")

        assert first_result is True
        assert approval.check_remembered("net", "https://example.com") is True

        # Second call must use the cache; Console constructor must not be called again.
        with patch("rich.console.Console") as mock_console_cls:
            second_result = approval.prompt_user("net", "https://example.com")

        assert second_result is True
        mock_console_cls.assert_not_called()

    def test_prompt_user_tty_n_response_returns_false(
        self, approval: InteractiveApproval
    ) -> None:
        """prompt_user returns False and stores nothing when operator enters 'n'."""
        console = _console_returning("n")

        with patch.object(InteractiveApproval, "_is_tty", return_value=True):
            console_patch, panel_patch = _patched_rich(console)
            with console_patch, panel_patch:
                result = approval.prompt_user("net", "https://blocked.com")

        assert result is False
        assert approval.check_remembered("net", "https://blocked.com") is None

    def test_prompt_user_keyboard_interrupt_returns_false(
        self, approval: InteractiveApproval
    ) -> None:
        """prompt_user returns False when the operator hits Ctrl-C at the prompt."""
        console = MagicMock()
        console.input.side_effect = KeyboardInterrupt

        with patch.object(InteractiveApproval, "_is_tty", return_value=True):
            console_patch, panel_patch = _patched_rich(console)
            with console_patch, panel_patch:
                result = approval.prompt_user("net", "https://example.com")

        assert result is False
