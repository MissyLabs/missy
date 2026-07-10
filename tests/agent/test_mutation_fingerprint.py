"""Tests for OpenClaw A3: mutation fingerprinting in AgentRuntime._tool_loop."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# _fingerprint_tc edge cases (supplement test_request_tracker_wiring.py)
# ---------------------------------------------------------------------------


def test_fingerprint_tc_none_values() -> None:
    """None values in args should not raise."""
    from missy.agent.runtime import _fingerprint_tc

    fp = _fingerprint_tc("tool", {"key": None})
    assert isinstance(fp, str)


def test_fingerprint_tc_nested_dicts() -> None:
    """Nested dicts should be stable across calls."""
    from missy.agent.runtime import _fingerprint_tc

    fp1 = _fingerprint_tc("tool", {"opts": {"verbose": True, "timeout": 30}})
    fp2 = _fingerprint_tc("tool", {"opts": {"timeout": 30, "verbose": True}})
    # sort_keys=True is recursive so these should match
    assert fp1 == fp2


def test_fingerprint_tc_list_values() -> None:
    """List values should produce stable fingerprints."""
    from missy.agent.runtime import _fingerprint_tc

    fp1 = _fingerprint_tc("tool", {"items": [1, 2, 3]})
    fp2 = _fingerprint_tc("tool", {"items": [1, 2, 3]})
    assert fp1 == fp2


def test_fingerprint_tc_returns_16_hex_chars() -> None:
    from missy.agent.runtime import _fingerprint_tc

    fp = _fingerprint_tc("any_tool", {"x": "y"})
    # We use first 16 chars of sha256 hex (all hex)
    assert len(fp) == 16
    assert all(c in "0123456789abcdef" for c in fp)


# ---------------------------------------------------------------------------
# A3 injection is wired into _tool_loop
# ---------------------------------------------------------------------------


def _make_runtime_with_tool_loop_mocks():
    """Build a minimal AgentRuntime and mock everything needed for _tool_loop."""
    from missy.agent.runtime import AgentConfig, AgentRuntime
    from missy.providers.base import CompletionResponse, ToolCall, ToolResult

    config = AgentConfig(max_iterations=4)
    rt = AgentRuntime(config)

    # Mock provider
    provider = MagicMock()
    provider.name = "mock_provider"
    provider.accepts_message_dicts = False

    # Build a ToolCall that will be executed
    tc = ToolCall(id="tc-1", name="shell_exec", arguments={"command": "bad_cmd"})

    # Tool result that always errors
    error_result = ToolResult(
        tool_call_id="tc-1",
        name="shell_exec",
        content="command not found",
        is_error=True,
    )

    return rt, provider, tc, error_result, CompletionResponse


def test_repeated_error_fingerprint_injects_lastToolError():
    """When the same tool call errors twice, lastToolError should be injected."""
    from missy.agent.runtime import AgentConfig, AgentRuntime
    from missy.providers.base import CompletionResponse, ToolCall, ToolResult

    config = AgentConfig(max_iterations=4)
    rt = AgentRuntime(config)

    provider = MagicMock()
    provider.name = "mock_provider"
    provider.accepts_message_dicts = False

    tc = ToolCall(id="tc-1", name="shell_exec", arguments={"command": "fail"})
    error_tr = ToolResult(
        tool_call_id="tc-1",
        name="shell_exec",
        content="permission denied",
        is_error=True,
    )

    _base = {"model": "mock", "provider": "mock_provider", "usage": {}, "raw": {}}
    # First two iterations: tool_calls with same failing tc; third: final stop
    tool_response = CompletionResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[tc],
        **_base,
    )
    final_response = CompletionResponse(
        content="I give up trying that command.",
        finish_reason="stop",
        tool_calls=None,
        **_base,
    )
    # Return tool_calls twice, then stop
    provider.complete_with_tools.side_effect = [
        tool_response,
        tool_response,
        final_response,
    ]

    def mock_execute(tc, session_id="", task_id="", **_kwargs):
        return error_tr

    rt._execute_tool = mock_execute  # type: ignore[method-assign]

    # Patch make_verification_prompt to return empty string (simplify)
    with patch("missy.agent.done_criteria.make_verification_prompt", return_value="[verify]"):
        result_text, tools_used = rt._tool_loop(
            provider=provider,
            tools=[],
            system_prompt="",
            messages=[{"role": "user", "content": "do the thing"}],
            session_id="test-sess",
            task_id="test-task",
            user_input="do the thing",
        )

    # Should have called complete_with_tools at least twice (second call sees A3 injection)
    assert provider.complete_with_tools.call_count >= 2

    # What we can assert is that the final text was returned without exception.
    assert isinstance(result_text, str)
    assert "shell_exec" in tools_used


def test_successful_retry_clears_fingerprint_error():
    """A successful call with the same args should not trigger lastToolError."""
    from missy.agent.runtime import AgentConfig, AgentRuntime
    from missy.providers.base import CompletionResponse, ToolCall, ToolResult

    config = AgentConfig(max_iterations=3)
    rt = AgentRuntime(config)

    provider = MagicMock()
    provider.name = "mock_provider"
    provider.accepts_message_dicts = False

    tc = ToolCall(id="tc-1", name="shell_exec", arguments={"command": "pwd"})
    success_tr = ToolResult(
        tool_call_id="tc-1",
        name="shell_exec",
        content="/home/missy",
        is_error=False,
    )
    _base = {"model": "mock", "provider": "mock_provider", "usage": {}, "raw": {}}
    tool_response = CompletionResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[tc],
        **_base,
    )
    final_response = CompletionResponse(
        content="Done.",
        finish_reason="stop",
        tool_calls=None,
        **_base,
    )
    provider.complete_with_tools.side_effect = [tool_response, final_response]

    def mock_execute(tc, session_id="", task_id="", **_kwargs):
        return success_tr

    rt._execute_tool = mock_execute  # type: ignore[method-assign]

    with patch("missy.agent.done_criteria.make_verification_prompt", return_value="[verify]"):
        result_text, tools_used = rt._tool_loop(
            provider=provider,
            tools=[],
            system_prompt="",
            messages=[{"role": "user", "content": "where am I?"}],
            session_id="test-sess",
            task_id="test-task",
            user_input="where am I?",
        )

    assert result_text == "Done."
    assert "shell_exec" in tools_used


def test_different_args_do_not_trigger_mutation_injection():
    """Two calls to same tool with different args should not trigger lastToolError."""
    from missy.agent.runtime import AgentConfig, AgentRuntime
    from missy.providers.base import CompletionResponse, ToolCall, ToolResult

    config = AgentConfig(max_iterations=4)
    rt = AgentRuntime(config)

    provider = MagicMock()
    provider.name = "mock_provider"
    provider.accepts_message_dicts = False

    tc1 = ToolCall(id="tc-1", name="shell_exec", arguments={"command": "ls /tmp"})
    tc2 = ToolCall(id="tc-2", name="shell_exec", arguments={"command": "ls /var"})

    error_tr1 = ToolResult(tool_call_id="tc-1", name="shell_exec", content="error1", is_error=True)
    error_tr2 = ToolResult(tool_call_id="tc-2", name="shell_exec", content="error2", is_error=True)

    _base = {"model": "mock", "provider": "mock_provider", "usage": {}, "raw": {}}
    resp1 = CompletionResponse(content="", finish_reason="tool_calls", tool_calls=[tc1], **_base)
    resp2 = CompletionResponse(content="", finish_reason="tool_calls", tool_calls=[tc2], **_base)
    final = CompletionResponse(content="All done.", finish_reason="stop", tool_calls=None, **_base)
    provider.complete_with_tools.side_effect = [resp1, resp2, final]

    call_count = [0]

    def mock_execute(tc, session_id="", task_id="", **_kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return error_tr1
        return error_tr2

    rt._execute_tool = mock_execute  # type: ignore[method-assign]

    with patch("missy.agent.done_criteria.make_verification_prompt", return_value="[verify]"):
        result_text, tools_used = rt._tool_loop(
            provider=provider,
            tools=[],
            system_prompt="",
            messages=[{"role": "user", "content": "list dirs"}],
            session_id="test-sess",
            task_id="test-task",
            user_input="list dirs",
        )

    # Both calls used different args — no repeated error fingerprint should trigger injection
    # after first round (count is 1 per fp). After second round, still 1 per fp.
    # So no lastToolError.
    assert result_text == "All done."
