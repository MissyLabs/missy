"""Coverage gap tests for missy.cli.oauth.

Targets uncovered lines: 124, 132-138, 146-153, 284, 359, 387-388,
397-404, 410.
"""

from __future__ import annotations

import threading
import time
from http.server import HTTPServer
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Line 124: _CallbackHandler.log_message silences access log
# ---------------------------------------------------------------------------


class TestCallbackHandlerLogMessage:
    def test_log_message_does_nothing(self):
        """log_message must be callable and return None (access-log suppressor)."""
        from missy.cli.oauth import _CallbackHandler

        handler = _CallbackHandler.__new__(_CallbackHandler)
        result = handler.log_message("GET /", "200", "OK")
        assert result is None


# ---------------------------------------------------------------------------
# Lines 132-138: _start_callback_server success and OSError fallback
# ---------------------------------------------------------------------------


class TestStartCallbackServer:
    def test_returns_server_on_success(self):
        from missy.cli.oauth import _start_callback_server

        mock_server = MagicMock(spec=HTTPServer)
        mock_thread = MagicMock()

        with (
            patch("missy.cli.oauth.HTTPServer", return_value=mock_server),
            patch("missy.cli.oauth.threading.Thread", return_value=mock_thread),
        ):
            result = _start_callback_server()

        assert result is mock_server
        mock_thread.start.assert_called_once()

    def test_returns_none_when_port_in_use(self):
        from missy.cli.oauth import _start_callback_server

        with patch("missy.cli.oauth.HTTPServer", side_effect=OSError("already in use")):
            result = _start_callback_server()

        assert result is None


# ---------------------------------------------------------------------------
# Lines 146-153: _wait_for_callback various outcomes
# ---------------------------------------------------------------------------


class TestWaitForCallback:
    def test_returns_code_when_callback_received(self):
        from missy.cli import oauth
        from missy.cli.oauth import _wait_for_callback

        mock_server = MagicMock()
        oauth._callback_event.clear()
        oauth._callback_result.clear()

        # Simulate a callback arriving before timeout
        oauth._callback_result["code"] = "auth-code-xyz"
        oauth._callback_result["state"] = "abc"
        oauth._callback_result["error"] = None
        oauth._callback_event.set()

        result = _wait_for_callback(mock_server, timeout=5)
        assert result == "auth-code-xyz"
        mock_server.shutdown.assert_called_once()

    def test_returns_none_on_timeout(self):
        from missy.cli import oauth
        from missy.cli.oauth import _wait_for_callback

        mock_server = MagicMock()
        oauth._callback_event.clear()
        oauth._callback_result.clear()

        # Use a very short timeout so the test finishes quickly
        result = _wait_for_callback(mock_server, timeout=0)
        assert result is None
        mock_server.shutdown.assert_called_once()

    def test_returns_none_when_error_in_callback(self):
        from missy.cli import oauth
        from missy.cli.oauth import _wait_for_callback

        mock_server = MagicMock()
        oauth._callback_event.clear()
        oauth._callback_result.clear()

        oauth._callback_result["error"] = "access_denied"
        oauth._callback_event.set()

        result = _wait_for_callback(mock_server, timeout=5)
        assert result is None

    def test_returns_none_when_no_code_in_result(self):
        from missy.cli import oauth
        from missy.cli.oauth import _wait_for_callback

        mock_server = MagicMock()
        oauth._callback_event.clear()
        oauth._callback_result.clear()

        # Event is set but code is not present
        oauth._callback_result["code"] = None
        oauth._callback_result["error"] = None
        oauth._callback_event.set()

        result = _wait_for_callback(mock_server, timeout=5)
        assert result is None


# ---------------------------------------------------------------------------
# Line 284: refresh_token_if_needed returns stale token when no refresh token
# ---------------------------------------------------------------------------


class TestRefreshTokenIfNeededStalePath:
    def test_returns_stale_token_when_no_refresh_token(self, tmp_path):
        from missy.cli.oauth import refresh_token_if_needed

        token_data = {
            "provider": "openai-oauth",
            "client_id": "test-client",
            "access_token": "stale-access-token",
            "refresh_token": "",  # empty — no refresh possible
            "expires_at": int(time.time()) - 3600,  # expired
        }

        with patch("missy.cli.oauth.load_token", return_value=token_data):
            result = refresh_token_if_needed()

        assert result == "stale-access-token"

    def test_returns_stale_token_when_no_client_id(self):
        token_data = {
            "provider": "openai-oauth",
            "client_id": "",
            "access_token": "my-stale-token",
            "refresh_token": "some-refresh",
            "expires_at": int(time.time()) - 3600,
        }

        with (
            patch("missy.cli.oauth.load_token", return_value=token_data),
            patch("missy.cli.oauth.DEFAULT_CLIENT_ID", ""),
        ):
            result = __import__(
                "missy.cli.oauth", fromlist=["refresh_token_if_needed"]
            ).refresh_token_if_needed(client_id="")
        # No client_id means we can't refresh — return stale
        assert result == "my-stale-token"


# ---------------------------------------------------------------------------
# Line 359: run_openai_oauth — server is None (port in use)
# ---------------------------------------------------------------------------


class TestRunOpenAIOAuthServerNone:
    def test_prints_port_in_use_message_when_server_none(self):
        """When _start_callback_server returns None, a warning is printed."""
        from missy.cli.oauth import run_openai_oauth

        with (
            patch("missy.cli.oauth._start_callback_server", return_value=None),
            patch("missy.cli.oauth.webbrowser.open"),
            patch("missy.cli.oauth._callback_event") as mock_event,
        ):
            mock_event.clear = MagicMock()
            with patch("missy.cli.oauth._callback_result") as mock_result:
                mock_result.clear = MagicMock()
                with (
                    patch("missy.cli.oauth.console") as mock_console,
                    patch("missy.cli.oauth.threading.Event") as MockEvent,
                ):
                    # Make the paste_done event fire immediately with empty input
                    mock_paste_done = MagicMock()
                    mock_paste_done.is_set.return_value = True
                    mock_paste_done.wait.return_value = None
                    MockEvent.return_value = mock_paste_done
                    with (
                        patch("missy.cli.oauth.threading.Thread"),
                        patch(
                            "missy.cli.oauth._extract_code_from_url",
                            return_value=None,
                        ),
                    ):
                        # No code → aborts
                        result = run_openai_oauth(client_id="test-client")

        assert result is None
        # Check the port-in-use warning was printed
        printed_messages = [str(call) for call in mock_console.print.call_args_list]
        assert any("1455" in msg or "unavailable" in msg for msg in printed_messages)


# ---------------------------------------------------------------------------
# Lines 387-388: _prompt_thread exception handler sets paste_done
# ---------------------------------------------------------------------------


class TestPromptThreadExceptionHandler:
    def test_prompt_thread_exception_appends_empty_string(self):
        """If click.prompt raises inside _prompt_thread, '' is appended."""
        paste_result: list[str] = []
        paste_done = threading.Event()

        def prompt_thread():
            try:
                raise RuntimeError("prompt failed")
            except Exception:
                paste_result.append("")
            finally:
                paste_done.set()

        t = threading.Thread(target=prompt_thread, daemon=True)
        t.start()
        paste_done.wait(timeout=2)

        assert paste_result == [""]
        assert paste_done.is_set()

    def test_run_oauth_prompt_thread_handles_exception(self):
        """Exercise the _prompt_thread exception path in run_openai_oauth."""
        from missy.cli.oauth import run_openai_oauth

        with (
            patch("missy.cli.oauth._start_callback_server", return_value=None),
            patch("missy.cli.oauth.webbrowser.open"),
            patch("missy.cli.oauth._callback_event"),
            patch("missy.cli.oauth._callback_result"),
            patch(
                "missy.cli.oauth.click.prompt",
                side_effect=Exception("terminal error"),
            ),
            patch("missy.cli.oauth.threading.Thread") as MockThread,
        ):
            # Simulate the thread finishing immediately
            def run_target(target=None, daemon=None):
                mock_t = MagicMock()

                def start():
                    if target:
                        target()

                mock_t.start = start
                return mock_t

            MockThread.side_effect = run_target
            result = run_openai_oauth(client_id="test-client")
        # No code → returns None
        assert result is None


# ---------------------------------------------------------------------------
# Lines 397-404: automatic callback wins the race
# ---------------------------------------------------------------------------


class TestAutomaticCallbackWins:
    def test_automatic_callback_code_used_over_paste(self):
        """When callback event fires before paste, the auto-captured code is used."""
        from missy.cli import oauth

        # Set up: callback delivers a code, exchange succeeds
        oauth._callback_event.clear()
        oauth._callback_result.clear()

        mock_server = MagicMock()
        token_resp = {
            "access_token": "auto-access-token",
            "refresh_token": "refresh",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

        with (
            patch("missy.cli.oauth._start_callback_server", return_value=mock_server),
            patch("missy.cli.oauth.webbrowser.open"),
            patch("missy.cli.oauth._exchange_code", return_value=token_resp),
            patch("missy.cli.oauth._save_token"),
            patch(
                "missy.cli.oauth._extract_account_metadata",
                return_value=("uid", "user@example.com"),
            ),
            patch("missy.cli.oauth.console"),
        ):
            # Make _callback_event.wait return True immediately
            # and set callback result
            def fake_wait(timeout=None):
                oauth._callback_result["code"] = "the-auth-code"
                oauth._callback_result["error"] = None
                # Set the state to match what run_openai_oauth generates.
                # We patch secrets.token_urlsafe below to control it.
                oauth._callback_result["state"] = "fixed-state-value"
                oauth._callback_event.set()
                return True

            with (
                patch.object(oauth._callback_event, "wait", side_effect=fake_wait),
                patch("missy.cli.oauth.threading.Thread") as MockThread,
                patch("missy.cli.oauth.secrets.token_urlsafe", return_value="fixed-state-value"),
            ):
                # Make paste_done.is_set() return True to break the loop
                paste_done_mock = MagicMock()
                paste_done_mock.is_set.return_value = False
                # Thread target stored so we can call it
                captured_thread = MagicMock()
                captured_thread.start = MagicMock()
                MockThread.return_value = captured_thread

                result = oauth.run_openai_oauth(client_id="test-client")

        assert result == "auto-access-token"


# ---------------------------------------------------------------------------
# Line 410: paste_result used when auto-capture did not provide code
# ---------------------------------------------------------------------------


class TestPasteResultFallback:
    def test_paste_result_used_when_no_auto_code(self):
        """When server captures nothing, paste_result[0] is used as the code source."""
        from missy.cli import oauth

        oauth._callback_event.clear()
        oauth._callback_result.clear()

        mock_server = MagicMock()
        token_resp = {
            "access_token": "pasted-token",
            "refresh_token": "",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        redirect_url = "http://localhost:1455/auth/callback?code=pasted-code&state=xyz"

        with (
            patch("missy.cli.oauth._start_callback_server", return_value=mock_server),
            patch("missy.cli.oauth.webbrowser.open"),
            patch("missy.cli.oauth._exchange_code", return_value=token_resp),
            patch("missy.cli.oauth._save_token"),
            patch("missy.cli.oauth._extract_account_metadata", return_value=("", "")),
            patch("missy.cli.oauth.console"),
            patch("missy.cli.oauth.threading.Thread") as MockThread,
        ):
            # Simulate paste thread: appends URL and sets paste_done
            def make_thread(target=None, daemon=None):
                m = MagicMock()

                def start_fn():
                    if target:
                        target()

                m.start = start_fn
                return m

            MockThread.side_effect = make_thread

            with (
                patch("missy.cli.oauth.click.prompt", return_value=redirect_url),
                patch.object(oauth._callback_event, "wait", return_value=False),
            ):
                result = oauth.run_openai_oauth(client_id="test-client")

        assert result == "pasted-token"
