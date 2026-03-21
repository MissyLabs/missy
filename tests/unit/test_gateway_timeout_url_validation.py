"""Tests for session 25: input validation, exception logging, magic number extraction."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Gateway: timeout validation
# ---------------------------------------------------------------------------


class TestGatewayTimeoutValidation:
    """PolicyHTTPClient rejects invalid timeout values."""

    def test_zero_timeout_rejected(self):
        from missy.gateway.client import PolicyHTTPClient

        with pytest.raises(ValueError, match="timeout must be positive"):
            PolicyHTTPClient(timeout=0)

    def test_negative_timeout_rejected(self):
        from missy.gateway.client import PolicyHTTPClient

        with pytest.raises(ValueError, match="timeout must be positive"):
            PolicyHTTPClient(timeout=-5)

    def test_positive_timeout_accepted(self):
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient(timeout=10)
        assert client.timeout == 10

    def test_default_timeout_accepted(self):
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient()
        assert client.timeout == 30


# ---------------------------------------------------------------------------
# Gateway: URL length validation
# ---------------------------------------------------------------------------


class TestGatewayURLLengthValidation:
    """_check_url rejects oversized URLs."""

    def test_url_exceeding_max_length_rejected(self):
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient()
        long_url = "https://example.com/" + "a" * 8200
        with pytest.raises(ValueError, match="exceeds maximum length"):
            client._check_url(long_url)

    def test_url_within_max_length_accepted(self):
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient()
        url = "https://example.com/short"
        # Should not raise ValueError for length (may raise for policy)
        try:
            client._check_url(url)
        except Exception as exc:
            # PolicyViolationError is OK (host denied), ValueError for length is not
            assert "exceeds maximum length" not in str(exc)

    def test_exactly_8192_url_accepted(self):
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient()
        # Exactly at the limit should be accepted
        url = "https://x.co/" + "a" * (8192 - len("https://x.co/"))
        assert len(url) == 8192
        try:
            client._check_url(url)
        except Exception as exc:
            assert "exceeds maximum length" not in str(exc)


# ---------------------------------------------------------------------------
# Runtime: empty input validation
# ---------------------------------------------------------------------------


class TestRuntimeEmptyInputValidation:
    """AgentRuntime.run() and run_stream() reject empty input."""

    def _make_runtime(self):
        from missy.agent.runtime import AgentRuntime

        with patch.object(AgentRuntime, "__init__", lambda self: None):
            rt = AgentRuntime.__new__(AgentRuntime)
        return rt

    def test_run_rejects_empty_string(self):
        rt = self._make_runtime()
        with pytest.raises(ValueError, match="non-empty"):
            rt.run("")

    def test_run_rejects_whitespace_only(self):
        rt = self._make_runtime()
        with pytest.raises(ValueError, match="non-empty"):
            rt.run("   \n\t  ")

    def test_run_stream_rejects_empty_string(self):
        rt = self._make_runtime()
        with pytest.raises(ValueError, match="non-empty"):
            # run_stream is a generator; must call next() to trigger
            gen = rt.run_stream("")
            next(gen)

    def test_run_stream_rejects_whitespace_only(self):
        rt = self._make_runtime()
        with pytest.raises(ValueError, match="non-empty"):
            gen = rt.run_stream("  \t  ")
            next(gen)


# ---------------------------------------------------------------------------
# Browser: session_id length limit
# ---------------------------------------------------------------------------


class TestBrowserSessionIdLength:
    """BrowserSession validates session_id length."""

    def test_session_id_over_128_chars_rejected(self):
        from missy.tools.builtin.browser_tools import BrowserSession

        long_id = "a" * 129
        with pytest.raises(ValueError, match="exceeds maximum length"):
            BrowserSession(long_id)

    def test_session_id_exactly_128_chars_accepted(self):
        from missy.tools.builtin.browser_tools import BrowserSession

        valid_id = "a" * 128
        # Should not raise for length; may raise for other reasons (playwright)
        try:
            BrowserSession(valid_id)
        except ValueError as exc:
            assert "exceeds maximum length" not in str(exc)
        except Exception:
            pass  # playwright unavailable is fine

    def test_short_session_id_accepted(self):
        from missy.tools.builtin.browser_tools import BrowserSession

        try:
            BrowserSession("default")
        except ValueError as exc:
            assert "exceeds maximum length" not in str(exc)
        except Exception:
            pass  # playwright unavailable is fine


# ---------------------------------------------------------------------------
# Checkpoint: named constants
# ---------------------------------------------------------------------------


class TestCheckpointConstants:
    """Checkpoint module uses named constants instead of magic numbers."""

    def test_constants_defined(self):
        from missy.agent.checkpoint import _RESTART_THRESHOLD_SECS, _RESUME_THRESHOLD_SECS

        assert _RESUME_THRESHOLD_SECS == 3600
        assert _RESTART_THRESHOLD_SECS == 86400

    def test_classify_uses_resume_threshold(self):
        import time

        from missy.agent.checkpoint import _RESUME_THRESHOLD_SECS, CheckpointManager

        cm = CheckpointManager.__new__(CheckpointManager)
        # Fresh checkpoint: just under the threshold
        checkpoint = {"created_at": time.time() - _RESUME_THRESHOLD_SECS + 10}
        assert cm.classify(checkpoint) == "resume"

    def test_classify_uses_restart_threshold(self):
        import time

        from missy.agent.checkpoint import _RESTART_THRESHOLD_SECS, CheckpointManager

        cm = CheckpointManager.__new__(CheckpointManager)
        # Between 1hr and 24hr: restart
        checkpoint = {"created_at": time.time() - _RESTART_THRESHOLD_SECS + 10}
        assert cm.classify(checkpoint) == "restart"

    def test_classify_abandon_after_restart_threshold(self):
        import time

        from missy.agent.checkpoint import _RESTART_THRESHOLD_SECS, CheckpointManager

        cm = CheckpointManager.__new__(CheckpointManager)
        # Beyond 24hr: abandon
        checkpoint = {"created_at": time.time() - _RESTART_THRESHOLD_SECS - 100}
        assert cm.classify(checkpoint) == "abandon"


# ---------------------------------------------------------------------------
# Runtime: streaming fallback logging
# ---------------------------------------------------------------------------


class TestStreamingFallbackLogging:
    """run_stream logs debug message when streaming fails."""

    def test_streaming_failure_logged(self, caplog):
        from missy.agent.runtime import AgentRuntime

        with patch.object(AgentRuntime, "__init__", lambda self: None):
            rt = AgentRuntime.__new__(AgentRuntime)

        # Set up minimal mocks
        rt._sanitizer = None
        rt._session_mgr = MagicMock()
        rt._session_mgr.get_active_session.return_value = MagicMock(id="test-sid")
        rt._cost_tracker = None
        rt.config = MagicMock()
        rt.config.max_iterations = 1

        mock_provider = MagicMock()
        mock_provider.name = "test"
        mock_provider.stream.side_effect = RuntimeError("stream broke")

        rt._sanitizer = None
        rt._resolve_session = MagicMock(return_value=MagicMock(id="test-sid"))
        rt._get_provider = MagicMock(return_value=mock_provider)
        rt._get_tools = MagicMock(return_value=None)
        rt._load_history = MagicMock(return_value=[])
        rt._build_context_messages = MagicMock(return_value=("sys", []))
        rt._dicts_to_messages = MagicMock(return_value=[])
        rt._acquire_rate_limit = MagicMock()
        rt._emit_event = MagicMock()
        rt._save_turn = MagicMock()

        # The _single_turn fallback
        mock_response = MagicMock()
        mock_response.content = "fallback response"
        rt._single_turn = MagicMock(return_value=mock_response)

        with caplog.at_level(logging.DEBUG, logger="missy.agent.runtime"):
            result = list(rt.run_stream("test input"))

        assert any("Streaming failed" in msg or "falling back" in msg for msg in caplog.messages)
        assert "fallback response" in result


# ---------------------------------------------------------------------------
# Code evolution: exc_info logging
# ---------------------------------------------------------------------------


class TestCodeEvolutionLogging:
    """code_evolution _load and _revert_diffs log with exc_info."""

    def test_load_logs_with_exc_info(self, caplog, tmp_path):
        from missy.agent.code_evolution import CodeEvolutionManager

        bad_file = tmp_path / "evolutions.json"
        bad_file.write_text("not valid json!!!")

        engine = CodeEvolutionManager.__new__(CodeEvolutionManager)
        engine._path = bad_file
        engine._proposals = []

        with caplog.at_level(logging.WARNING, logger="missy.agent.code_evolution"):
            result = engine._load()

        assert result == []
        assert any("Failed to load evolutions" in msg for msg in caplog.messages)
        # exc_info should attach the exception record
        assert any(
            record.exc_info for record in caplog.records if "Failed to load" in record.message
        )


# ---------------------------------------------------------------------------
# Sanitizer: decode fallback logging
# ---------------------------------------------------------------------------


class TestSanitizerDecodeLogging:
    """InputSanitizer logs debug when URL/HTML decode fails."""

    def test_url_decode_failure_logged(self, caplog):
        from missy.security.sanitizer import InputSanitizer

        san = InputSanitizer()
        # unquote rarely fails, but we can mock it
        with (
            patch("missy.security.sanitizer.unquote", side_effect=ValueError("bad")),
            caplog.at_level(logging.DEBUG, logger="missy.security.sanitizer"),
        ):
            # Should not raise, just log
            san.check_for_injection("normal text")

        assert any("URL-decode failed" in msg for msg in caplog.messages)

    def test_html_unescape_failure_logged(self, caplog):
        from missy.security.sanitizer import InputSanitizer

        san = InputSanitizer()
        with (
            patch("missy.security.sanitizer.html.unescape", side_effect=ValueError("bad")),
            caplog.at_level(logging.DEBUG, logger="missy.security.sanitizer"),
        ):
            san.check_for_injection("normal text")

        assert any("HTML-unescape failed" in msg for msg in caplog.messages)
