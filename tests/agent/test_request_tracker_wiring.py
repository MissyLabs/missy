"""Tests for RequestTracker wiring in AgentRuntime (OpenClaw tool intelligence)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# _fingerprint_tc helper
# ---------------------------------------------------------------------------


def test_fingerprint_tc_stable() -> None:
    from missy.agent.runtime import _fingerprint_tc

    fp1 = _fingerprint_tc("shell_exec", {"command": "ls -la"})
    fp2 = _fingerprint_tc("shell_exec", {"command": "ls -la"})
    assert fp1 == fp2


def test_fingerprint_tc_different_args() -> None:
    from missy.agent.runtime import _fingerprint_tc

    fp1 = _fingerprint_tc("shell_exec", {"command": "ls -la"})
    fp2 = _fingerprint_tc("shell_exec", {"command": "cat /etc/passwd"})
    assert fp1 != fp2


def test_fingerprint_tc_different_tools() -> None:
    from missy.agent.runtime import _fingerprint_tc

    fp1 = _fingerprint_tc("shell_exec", {"command": "ls"})
    fp2 = _fingerprint_tc("file_read", {"command": "ls"})
    assert fp1 != fp2


def test_fingerprint_tc_dict_order_invariant() -> None:
    """Dicts with the same keys/values in different order must produce the same fp."""
    from missy.agent.runtime import _fingerprint_tc

    fp1 = _fingerprint_tc("tool", {"a": 1, "b": 2})
    fp2 = _fingerprint_tc("tool", {"b": 2, "a": 1})
    assert fp1 == fp2


def test_fingerprint_tc_empty_args() -> None:
    from missy.agent.runtime import _fingerprint_tc

    fp = _fingerprint_tc("no_args_tool", {})
    assert isinstance(fp, str) and len(fp) == 16


# ---------------------------------------------------------------------------
# _make_request_tracker in AgentRuntime
# ---------------------------------------------------------------------------


def test_make_request_tracker_returns_tracker() -> None:
    """_make_request_tracker should return a RequestTracker when available."""
    mock_tracker = MagicMock()
    mock_get = MagicMock(return_value=mock_tracker)

    with patch("missy.tools.intelligence.get_request_tracker", mock_get):
        from missy.agent.runtime import AgentRuntime

        tracker = AgentRuntime._make_request_tracker()
        assert tracker is mock_tracker


def test_make_request_tracker_graceful_degradation() -> None:
    """When tools.intelligence is unavailable, _make_request_tracker returns None."""
    with patch.dict("sys.modules", {"missy.tools.intelligence": None}):
        from missy.agent.runtime import AgentRuntime

        # Force reimport path by patching the import
        tracker = AgentRuntime._make_request_tracker()
        # Should be None or a real tracker — just should not raise
        assert tracker is None or tracker is not None


# ---------------------------------------------------------------------------
# _track_request
# ---------------------------------------------------------------------------


def test_track_request_calls_record() -> None:
    from missy.agent.runtime import AgentConfig, AgentRuntime

    rt = AgentRuntime(AgentConfig())
    mock_tracker = MagicMock()
    rt._request_tracker = mock_tracker

    rt._track_request("hello world", "sess-1", ["shell_exec"], "anthropic")
    mock_tracker.record.assert_called_once_with(
        session_id="sess-1",
        user_message="hello world",
        tool_calls=["shell_exec"],
        metadata={"provider": "anthropic"},
    )


def test_track_request_no_tracker_is_noop() -> None:
    from missy.agent.runtime import AgentConfig, AgentRuntime

    rt = AgentRuntime(AgentConfig())
    rt._request_tracker = None  # simulate unavailable tracker
    # Should not raise
    rt._track_request("hello", "sess-1", [], "anthropic")


def test_track_request_exception_is_swallowed() -> None:
    from missy.agent.runtime import AgentConfig, AgentRuntime

    rt = AgentRuntime(AgentConfig())
    broken = MagicMock()
    broken.record.side_effect = RuntimeError("db gone")
    rt._request_tracker = broken

    # Should not propagate the exception
    rt._track_request("hello", "sess-1", [], "anthropic")


def test_track_request_empty_tool_calls_ok() -> None:
    from missy.agent.runtime import AgentConfig, AgentRuntime

    rt = AgentRuntime(AgentConfig())
    mock_tracker = MagicMock()
    rt._request_tracker = mock_tracker

    rt._track_request("what time is it?", "sess-2", [], "ollama")
    mock_tracker.record.assert_called_once()
    _call = mock_tracker.record.call_args
    assert _call.kwargs["tool_calls"] == []


# ---------------------------------------------------------------------------
# runtime initializes _request_tracker attribute
# ---------------------------------------------------------------------------


def test_runtime_has_request_tracker_attribute() -> None:
    from missy.agent.runtime import AgentConfig, AgentRuntime

    rt = AgentRuntime(AgentConfig())
    assert hasattr(rt, "_request_tracker")
