"""Coverage gap tests for:

- missy/cli/anthropic_auth.py   — lines 119-121, 237
- missy/channels/discord/screen_commands.py — lines 95, 138-139, 170, 183
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# screen_commands — lines 95, 138-139, 170, 183
# ---------------------------------------------------------------------------
from missy.channels.discord.screen_commands import (
    ScreenCommandResult,
    maybe_handle_screen_command,
)
from missy.channels.screencast.session_manager import AnalysisResult

# ---------------------------------------------------------------------------
# anthropic_auth — lines 119-121, 237
# ---------------------------------------------------------------------------
from missy.cli.anthropic_auth import (
    load_token,
    store_token,
)

# ===========================================================================
# Module 1: anthropic_auth
# ===========================================================================


class TestStoreTokenWriteFailure:
    """Lines 119-121: BaseException cleanup path inside store_token.

    When json.dump raises mid-write the temporary file must be deleted
    and the exception must propagate to the caller.
    """

    @pytest.fixture(autouse=True)
    def _patch_token_file(self, tmp_path, monkeypatch):
        self.token_file = tmp_path / "secrets" / "anthropic-token.json"
        monkeypatch.setattr("missy.cli.anthropic_auth.TOKEN_FILE", self.token_file)

    def test_write_error_removes_tmp_file(self, tmp_path):
        """Tmp file is unlinked when json.dump raises so no partial file is left."""
        tmp_path_expected = self.token_file.with_suffix(".tmp")

        with patch("json.dump", side_effect=OSError("disk full")), pytest.raises(OSError, match="disk full"):
            store_token("sk-ant-api03-test", "api_key")

        # The tmp file must have been cleaned up on exception.
        assert not tmp_path_expected.exists()

    def test_write_error_does_not_leave_token_file(self, tmp_path):
        """The final token file must not exist after a failed write."""
        with patch("json.dump", side_effect=ValueError("serialise fail")), pytest.raises(ValueError):
            store_token("bad-token", "setup_token")

        assert not self.token_file.exists()

    def test_write_error_propagates_original_exception_type(self):
        """The exact exception type from the write is re-raised unmodified."""
        class _CustomError(Exception):
            pass

        with patch("json.dump", side_effect=_CustomError("custom")), pytest.raises(_CustomError, match="custom"):
            store_token("tok", "api_key")

    def test_successful_write_after_prior_failure_still_works(self):
        """A subsequent store_token call succeeds even after a prior failure."""
        with patch("json.dump", side_effect=OSError("transient")), pytest.raises(OSError):
            store_token("failing-token", "api_key")

        # Now a normal call should work.
        store_token("good-token", "api_key")
        data = load_token()
        assert data is not None
        assert data["token"] == "good-token"


class TestSetupTokenFlowEmptyPasteLoopContinue:
    """Line 237: empty paste + 'No' to abort → loop continues.

    When the user submits an empty paste and then declines the abort prompt
    (i.e. answers 'No' to "Abort setup-token flow?"), the loop must not
    return None but must instead iterate and prompt again.

    The second iteration provides a valid setup-token so we can observe
    the function returning a value rather than None.
    """

    @pytest.fixture(autouse=True)
    def _patch_token_file(self, tmp_path, monkeypatch):
        token_file = tmp_path / "secrets" / "anthropic-token.json"
        monkeypatch.setattr("missy.cli.anthropic_auth.TOKEN_FILE", token_file)

    def test_empty_paste_decline_abort_then_valid_token(self):
        """Empty → decline abort → valid token → function returns the token."""
        from missy.cli import anthropic_auth

        valid_token = "sk-ant-oat01-" + "B" * 70

        prompt_returns = ["", valid_token]

        def _fake_prompt(*args, **kwargs):
            return prompt_returns.pop(0)

        # confirm: ToS accept=True, "Abort?": False (don't abort), then (no more)
        confirm_returns = [True, False]

        def _fake_confirm(msg, **kwargs):
            return confirm_returns.pop(0)

        with (
            patch.object(anthropic_auth, "console"),
            patch("click.confirm", side_effect=_fake_confirm),
            patch("click.prompt", side_effect=_fake_prompt),
            patch("shutil.which", return_value=None),
        ):
            result = anthropic_auth.run_anthropic_setup_token_flow()

        assert result == valid_token

    def test_empty_paste_decline_abort_then_abort(self):
        """Empty → decline abort → empty again → accept abort → returns None."""
        from missy.cli import anthropic_auth

        prompt_returns = ["", ""]

        def _fake_prompt(*args, **kwargs):
            return prompt_returns.pop(0)

        # confirm: ToS=True, first abort decline=False, second abort accept=True
        confirm_returns = [True, False, True]

        def _fake_confirm(msg, **kwargs):
            return confirm_returns.pop(0)

        with (
            patch.object(anthropic_auth, "console"),
            patch("click.confirm", side_effect=_fake_confirm),
            patch("click.prompt", side_effect=_fake_prompt),
            patch("shutil.which", return_value=None),
        ):
            result = anthropic_auth.run_anthropic_setup_token_flow()

        assert result is None


# ===========================================================================
# Module 2: screen_commands
# ===========================================================================


async def _invoke(
    content: str,
    screencast: object | None = None,
    channel_id: str = "chan-1",
    author_id: str = "user-1",
) -> ScreenCommandResult:
    return await maybe_handle_screen_command(
        content=content,
        channel_id=channel_id,
        author_id=author_id,
        screencast=screencast,
    )


def _fake_session(
    session_id: str = "s-001",
    label: str = "test",
    created_by: str = "user1",
    frame_count: int = 5,
    analysis_count: int = 2,
    last_frame_at: float = 0.0,
) -> dict:
    return {
        "session_id": session_id,
        "label": label,
        "created_by": created_by,
        "created_at": time.time() - 60,
        "frame_count": frame_count,
        "analysis_count": analysis_count,
        "last_frame_at": last_frame_at,
    }


class TestScreenCommandFallthrough:
    """Line 95: final fallthrough ScreenCommandResult(False) after all subcmd branches.

    The only way to reach line 95 would be for a recognised subcmd token to
    slip through every preceding if-branch — which cannot happen with the
    current logic.  The closest observable equivalent is confirming that every
    listed valid subcommand is handled (returns handled=True with a screencast),
    ensuring coverage infra accounts for the branch.  We additionally verify
    the unknown-subcommand path (line 65) which is the *reachable* early
    return that guards line 95.
    """

    async def test_unknown_subcommand_returns_not_handled(self):
        """An unrecognised subcommand returns handled=False without reaching line 95."""
        result = await _invoke("!screen frobnicate")
        assert result.handled is False
        assert result.reply is None

    async def test_all_valid_subcommands_are_handled_with_screencast(self):
        """Every recognised subcommand produces handled=True when screencast is present."""
        sc = MagicMock()
        sc.create_session.return_value = ("s-1", "tok", "http://example.com/s-1")
        sc.get_active_sessions.return_value = []
        sc.get_status.return_value = {"running": False}
        sc.get_latest_analysis.return_value = None

        for subcmd in ("share", "list", "stop", "analyze", "status"):
            result = await _invoke(f"!screen {subcmd}", screencast=sc)
            assert result.handled is True, f"subcmd={subcmd!r} should be handled"


class TestListLastFrameAt:
    """Lines 138-139: last_frame_at branch in _handle_list.

    When a session has a non-zero last_frame_at timestamp the list output
    must include the 'last frame … ago' suffix.
    """

    async def test_last_frame_shown_when_nonzero(self):
        sc = MagicMock()
        # last_frame_at set to 30 seconds ago
        sc.get_active_sessions.return_value = [
            _fake_session("s-framed", last_frame_at=time.time() - 30)
        ]

        result = await _invoke("!screen list", screencast=sc)

        assert result.handled is True
        assert result.reply is not None
        assert "last frame" in result.reply

    async def test_last_frame_absent_when_zero(self):
        """When last_frame_at is falsy the suffix is omitted."""
        sc = MagicMock()
        sc.get_active_sessions.return_value = [
            _fake_session("s-noframe", last_frame_at=0.0)
        ]

        result = await _invoke("!screen list", screencast=sc)

        assert result.handled is True
        assert "last frame" not in (result.reply or "")


class TestAnalyzeAutoSelectSession:
    """Line 170: no session_id given but active sessions exist — uses last one."""

    async def test_analyze_no_id_uses_last_active_session(self):
        sc = MagicMock()
        sessions = [
            _fake_session("s-first"),
            _fake_session("s-last"),
        ]
        sc.get_active_sessions.return_value = sessions

        analysis = AnalysisResult(
            session_id="s-last",
            frame_number=1,
            analysis_text="brief result",
            model="llava",
            processing_ms=100,
        )
        sc.get_latest_analysis.return_value = analysis

        result = await _invoke("!screen analyze", screencast=sc)

        assert result.handled is True
        sc.get_latest_analysis.assert_called_once_with("s-last")
        assert "s-last" in (result.reply or "")

    async def test_analyze_explicit_id_skips_active_session_lookup(self):
        """When session_id is supplied, get_active_sessions is never called."""
        sc = MagicMock()
        sc.get_latest_analysis.return_value = None

        result = await _invoke("!screen analyze explicit-id", screencast=sc)

        sc.get_active_sessions.assert_not_called()
        assert result.handled is True


class TestAnalyzeLongTextTruncation:
    """Line 183: analysis text longer than max_body is truncated with '...'."""

    async def test_long_analysis_text_is_truncated(self):
        sc = MagicMock()
        # Generate text well over 2000 characters to guarantee truncation.
        long_text = "A" * 3000
        analysis = AnalysisResult(
            session_id="s-long",
            frame_number=7,
            analysis_text=long_text,
            model="llava",
            processing_ms=500,
        )
        sc.get_latest_analysis.return_value = analysis

        result = await _invoke("!screen analyze s-long", screencast=sc)

        assert result.handled is True
        assert result.reply is not None
        # Full text cannot fit; truncation marker must appear.
        assert "..." in result.reply
        # Total reply must be within Discord's 2000-character limit + footer overhead.
        assert len(result.reply) < 2100

    async def test_short_analysis_text_is_not_truncated(self):
        """Text that fits within the limit passes through unchanged."""
        sc = MagicMock()
        short_text = "Concise result."
        analysis = AnalysisResult(
            session_id="s-short",
            frame_number=3,
            analysis_text=short_text,
            model="llava",
            processing_ms=80,
        )
        sc.get_latest_analysis.return_value = analysis

        result = await _invoke("!screen analyze s-short", screencast=sc)

        assert result.handled is True
        assert short_text in (result.reply or "")
        assert "..." not in (result.reply or "").split(short_text)[0]
