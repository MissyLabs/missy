"""Comprehensive session-15 tests for missy/agent/runtime.py.

Covers:
  - AgentConfig dataclass defaults and custom values
  - _rewrite_heredoc_command: all heredoc patterns, passthrough, edge cases
  - AgentRuntime.__init__: subsystem wiring with mocked dependencies
  - _bus_publish: bus=None no-op, bus.publish raises, normal publish
  - _make_message_bus: graceful degradation when get_message_bus raises
  - switch_provider: updates config and rebuilds circuit breaker
  - Module-level constants: _MAX_TOOL_RESULT_CHARS, _LARGE_CONTENT_THRESHOLD
  - DISCORD_SYSTEM_PROMPT content and differences from default system prompt
  - capability_mode storage and _get_tools filtering
  - NullReporter default when no progress_reporter supplied
"""

from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.runtime import (
    _LARGE_CONTENT_THRESHOLD,
    _MAX_TOOL_RESULT_CHARS,
    DISCORD_SYSTEM_PROMPT,
    AgentConfig,
    AgentRuntime,
    _rewrite_heredoc_command,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(name: str = "fake", available: bool = True) -> MagicMock:
    from missy.providers.base import CompletionResponse

    p = MagicMock()
    p.name = name
    p.is_available.return_value = available
    p.complete.return_value = CompletionResponse(
        content="reply",
        model="test-model",
        provider=name,
        usage={"prompt_tokens": 5, "completion_tokens": 3},
        raw={},
        finish_reason="stop",
    )
    return p


def _make_mock_registry(provider: MagicMock) -> MagicMock:
    reg = MagicMock()
    reg.get.return_value = provider
    reg.get_available.return_value = [provider]
    return reg


def _make_runtime(provider: MagicMock | None = None, **config_kwargs) -> AgentRuntime:
    """Build an AgentRuntime with all subsystems mocked out for speed."""
    if provider is None:
        provider = _make_provider()
    reg = _make_mock_registry(provider)
    cfg = AgentConfig(**config_kwargs) if config_kwargs else AgentConfig(provider="fake")

    with (
        patch("missy.agent.runtime.get_registry", return_value=reg),
        patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError("no tools")),
        patch("missy.agent.runtime.get_message_bus", side_effect=RuntimeError("no bus")),
    ):
        runtime = AgentRuntime(cfg)

    # Disable subsystems that require filesystem / network for these unit tests
    runtime._rate_limiter = None
    runtime._memory_store = None
    runtime._cost_tracker = None
    runtime._context_manager = None
    runtime._drift_detector = None
    return runtime


# ---------------------------------------------------------------------------
# 1. AgentConfig defaults
# ---------------------------------------------------------------------------


class TestAgentConfigDefaults:
    def test_default_provider(self):
        assert AgentConfig().provider == "anthropic"

    def test_default_model_is_none(self):
        assert AgentConfig().model is None

    def test_default_max_iterations(self):
        assert AgentConfig().max_iterations == 10

    def test_default_temperature(self):
        assert AgentConfig().temperature == 0.7

    def test_default_capability_mode(self):
        assert AgentConfig().capability_mode == "full"

    def test_default_max_spend_usd(self):
        assert AgentConfig().max_spend_usd == 0.0

    def test_default_system_prompt_contains_missy(self):
        assert "Missy" in AgentConfig().system_prompt

    def test_default_system_prompt_mentions_tools(self):
        prompt = AgentConfig().system_prompt
        assert "file_read" in prompt or "tools" in prompt.lower()

    def test_default_system_prompt_is_nonempty_string(self):
        prompt = AgentConfig().system_prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 50


class TestAgentConfigCustomValues:
    def test_custom_provider(self):
        cfg = AgentConfig(provider="openai")
        assert cfg.provider == "openai"

    def test_custom_model(self):
        cfg = AgentConfig(model="gpt-4o")
        assert cfg.model == "gpt-4o"

    def test_custom_max_iterations(self):
        cfg = AgentConfig(max_iterations=3)
        assert cfg.max_iterations == 3

    def test_custom_temperature(self):
        cfg = AgentConfig(temperature=0.0)
        assert cfg.temperature == 0.0

    def test_custom_system_prompt(self):
        cfg = AgentConfig(system_prompt="Be brief.")
        assert cfg.system_prompt == "Be brief."

    def test_custom_capability_mode(self):
        cfg = AgentConfig(capability_mode="safe-chat")
        assert cfg.capability_mode == "safe-chat"

    def test_custom_max_spend_usd(self):
        cfg = AgentConfig(max_spend_usd=5.0)
        assert cfg.max_spend_usd == 5.0

    def test_capability_mode_discord(self):
        cfg = AgentConfig(capability_mode="discord")
        assert cfg.capability_mode == "discord"

    def test_capability_mode_no_tools(self):
        cfg = AgentConfig(capability_mode="no-tools")
        assert cfg.capability_mode == "no-tools"

    def test_all_fields_present(self):
        field_names = {f.name for f in fields(AgentConfig)}
        assert "provider" in field_names
        assert "model" in field_names
        assert "system_prompt" in field_names
        assert "max_iterations" in field_names
        assert "temperature" in field_names
        assert "capability_mode" in field_names
        assert "max_spend_usd" in field_names


# ---------------------------------------------------------------------------
# 2. _rewrite_heredoc_command
# ---------------------------------------------------------------------------


class TestRewriteHeredocCommandPassthrough:
    def test_no_heredoc_passthrough(self):
        args = {"command": "echo hello"}
        result = _rewrite_heredoc_command(args)
        assert result is args

    def test_no_command_key_passthrough(self):
        args = {"other": "value"}
        result = _rewrite_heredoc_command(args)
        assert result is args

    def test_empty_command_passthrough(self):
        args = {"command": ""}
        result = _rewrite_heredoc_command(args)
        assert result is args

    def test_unrelated_double_lt_in_filename_passthrough(self):
        # If "<<" not in command, returns unchanged — this tests the guard
        args = {"command": "cat file.txt | grep pattern"}
        result = _rewrite_heredoc_command(args)
        assert result is args


class TestRewriteHeredocCommandPatterns:
    def test_python3_heredoc_single_quote_delimiter(self):
        command = "python3 - <<'EOF'\nprint('hello')\nEOF"
        args = {"command": command}
        result = _rewrite_heredoc_command(args)
        assert result is not args
        new_cmd = result["command"]
        assert new_cmd.startswith("python3 ")
        assert "<<" not in new_cmd
        # Verify temp file contains the body
        tmppath = new_cmd.split(" ", 1)[1]
        assert os.path.exists(tmppath)
        content = Path(tmppath).read_text()
        assert "print('hello')" in content
        os.unlink(tmppath)

    def test_bash_heredoc(self):
        command = "bash - <<'SCRIPT'\necho hi\nSCRIPT"
        args = {"command": command}
        result = _rewrite_heredoc_command(args)
        new_cmd = result["command"]
        assert new_cmd.startswith("bash ")
        assert "<<" not in new_cmd
        tmppath = new_cmd.split(" ", 1)[1]
        assert tmppath.endswith(".sh")
        content = Path(tmppath).read_text()
        assert "echo hi" in content
        os.unlink(tmppath)

    def test_ruby_heredoc(self):
        command = "ruby - <<'RUBY'\nputs 'world'\nRUBY"
        args = {"command": command}
        result = _rewrite_heredoc_command(args)
        new_cmd = result["command"]
        assert new_cmd.startswith("ruby ")
        tmppath = new_cmd.split(" ", 1)[1]
        assert tmppath.endswith(".rb")
        os.unlink(tmppath)

    def test_heredoc_double_quote_delimiter(self):
        command = 'python3 - <<"EOF"\nprint(42)\nEOF'
        args = {"command": command}
        result = _rewrite_heredoc_command(args)
        new_cmd = result["command"]
        assert "<<" not in new_cmd
        tmppath = new_cmd.split(" ", 1)[1]
        content = Path(tmppath).read_text()
        assert "print(42)" in content
        os.unlink(tmppath)

    def test_heredoc_unquoted_delimiter(self):
        command = "python3 - <<EOF\nprint('unquoted')\nEOF"
        args = {"command": command}
        result = _rewrite_heredoc_command(args)
        new_cmd = result["command"]
        assert "<<" not in new_cmd
        tmppath = new_cmd.split(" ", 1)[1]
        content = Path(tmppath).read_text()
        assert "print('unquoted')" in content
        os.unlink(tmppath)

    def test_python3_heredoc_no_stdin_dash(self):
        # Some models omit the "-" marker: python3 <<'EOF'
        command = "python3 <<'EOF'\nprint('no dash')\nEOF"
        args = {"command": command}
        result = _rewrite_heredoc_command(args)
        new_cmd = result["command"]
        assert "<<" not in new_cmd
        tmppath = new_cmd.split(" ", 1)[1]
        content = Path(tmppath).read_text()
        assert "print('no dash')" in content
        os.unlink(tmppath)

    def test_temp_file_has_correct_python_extension(self):
        command = "python3 - <<'PY'\nx = 1\nPY"
        result = _rewrite_heredoc_command({"command": command})
        tmppath = result["command"].split(" ", 1)[1]
        assert tmppath.endswith(".py")
        os.unlink(tmppath)

    def test_heredoc_with_empty_body(self):
        command = "python3 - <<'EOF'\n\nEOF"
        args = {"command": command}
        result = _rewrite_heredoc_command(args)
        # Pattern requires (.*?) which can match empty — either rewrites or passes through
        # Just verify no exception is raised and result is a dict
        assert isinstance(result, dict)
        if "<<" not in result.get("command", ""):
            tmppath = result["command"].split(" ", 1)[1]
            if os.path.exists(tmppath):
                os.unlink(tmppath)

    def test_heredoc_with_unicode_content(self):
        command = "python3 - <<'EOF'\nprint('héllo wörld')\nEOF"
        args = {"command": command}
        result = _rewrite_heredoc_command(args)
        new_cmd = result["command"]
        assert "<<" not in new_cmd
        tmppath = new_cmd.split(" ", 1)[1]
        content = Path(tmppath).read_text(encoding="utf-8")
        assert "héllo" in content
        os.unlink(tmppath)

    def test_heredoc_with_special_chars_in_body(self):
        command = "python3 - <<'EOF'\nprint('$HOME and {braces}')\nEOF"
        args = {"command": command}
        result = _rewrite_heredoc_command(args)
        new_cmd = result["command"]
        tmppath = new_cmd.split(" ", 1)[1]
        content = Path(tmppath).read_text()
        assert "$HOME" in content
        assert "{braces}" in content
        os.unlink(tmppath)

    def test_rewrite_preserves_other_args(self):
        command = "python3 - <<'EOF'\nprint('x')\nEOF"
        args = {"command": command, "timeout": 30, "env": {"KEY": "val"}}
        result = _rewrite_heredoc_command(args)
        assert result["timeout"] == 30
        assert result["env"] == {"KEY": "val"}
        tmppath = result["command"].split(" ", 1)[1]
        os.unlink(tmppath)

    def test_rewrite_does_not_mutate_original_args(self):
        command = "python3 - <<'EOF'\nprint('x')\nEOF"
        original_command = command
        args = {"command": command}
        result = _rewrite_heredoc_command(args)
        # original dict command should be unchanged
        assert args["command"] == original_command
        tmppath = result["command"].split(" ", 1)[1]
        os.unlink(tmppath)


class TestRewriteHeredocCommandMalformed:
    def test_malformed_heredoc_passthrough(self):
        # Has "<<" but no closing delimiter — pattern won't match
        command = "python3 - << NODCLOSE\nsome code\n"
        args = {"command": command}
        result = _rewrite_heredoc_command(args)
        # Should return original (pattern didn't match)
        assert result is args

    def test_heredoc_mismatched_delimiters_passthrough(self):
        # Opening delimiter is OPEN, closing is CLOSE — mismatch
        command = "python3 - <<'OPEN'\ncode here\nCLOSE"
        args = {"command": command}
        result = _rewrite_heredoc_command(args)
        assert result is args

    def test_nested_heredoc_like_content(self):
        # Body itself contains << but pattern should still work for outer
        command = "python3 - <<'EOF'\nimport subprocess\nsubprocess.run(['cat', '<<'])\nEOF"
        args = {"command": command}
        # Should not raise — either rewrites or passes through
        result = _rewrite_heredoc_command(args)
        assert isinstance(result, dict)
        if result is not args:
            tmppath = result["command"].split(" ", 1)[1]
            if os.path.exists(tmppath):
                os.unlink(tmppath)


# ---------------------------------------------------------------------------
# 3. Module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_max_tool_result_chars_value(self):
        assert _MAX_TOOL_RESULT_CHARS == 200_000

    def test_large_content_threshold_value(self):
        assert _LARGE_CONTENT_THRESHOLD == 50_000

    def test_large_content_threshold_less_than_max(self):
        assert _LARGE_CONTENT_THRESHOLD <= _MAX_TOOL_RESULT_CHARS

    def test_max_tool_result_chars_is_int(self):
        assert isinstance(_MAX_TOOL_RESULT_CHARS, int)

    def test_large_content_threshold_is_int(self):
        assert isinstance(_LARGE_CONTENT_THRESHOLD, int)


# ---------------------------------------------------------------------------
# 4. DISCORD_SYSTEM_PROMPT
# ---------------------------------------------------------------------------


class TestDiscordSystemPrompt:
    def test_discord_prompt_is_string(self):
        assert isinstance(DISCORD_SYSTEM_PROMPT, str)

    def test_discord_prompt_nonempty(self):
        assert len(DISCORD_SYSTEM_PROMPT) > 50

    def test_discord_prompt_mentions_discord(self):
        assert "Discord" in DISCORD_SYSTEM_PROMPT

    def test_discord_prompt_disallows_x11(self):
        # Discord prompt mentions X11 only to deny access to it
        lower = DISCORD_SYSTEM_PROMPT.lower()
        if "x11" in lower:
            # Must be in a denial context — the prompt says "do NOT have access to ... X11"
            assert "not" in lower or "do not" in lower

    def test_discord_prompt_no_browser_reference(self):
        # Discord prompt explicitly says no browser/GUI access
        lower = DISCORD_SYSTEM_PROMPT.lower()
        if "browser" in lower:
            assert "not" in lower or "do not" in lower

    def test_discord_prompt_different_from_default(self):
        default = AgentConfig().system_prompt
        assert default != DISCORD_SYSTEM_PROMPT

    def test_discord_prompt_mentions_missy(self):
        assert "Missy" in DISCORD_SYSTEM_PROMPT

    def test_discord_prompt_no_gui_reference(self):
        # Explicitly disallows GUI/X11/browser
        assert "do NOT" in DISCORD_SYSTEM_PROMPT or "do not" in DISCORD_SYSTEM_PROMPT.lower()

    def test_discord_prompt_mentions_discord_upload_file(self):
        assert "discord_upload_file" in DISCORD_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 5. AgentRuntime.__init__ wiring
# ---------------------------------------------------------------------------


class TestAgentRuntimeInit:
    def test_init_stores_config(self):
        runtime = _make_runtime()
        assert isinstance(runtime.config, AgentConfig)

    def test_init_sets_provider_from_config(self):
        runtime = _make_runtime()
        assert runtime.config.provider == "fake"

    def test_init_creates_session_manager(self):
        from missy.core.session import SessionManager

        runtime = _make_runtime()
        assert isinstance(runtime._session_mgr, SessionManager)

    def test_init_creates_circuit_breaker(self):
        runtime = _make_runtime()
        # Either a real CircuitBreaker or the _NoOpCircuitBreaker stub
        assert runtime._circuit_breaker is not None
        assert hasattr(runtime._circuit_breaker, "call")

    def test_init_creates_trust_scorer(self):
        from missy.security.trust import TrustScorer

        runtime = _make_runtime()
        assert isinstance(runtime._trust_scorer, TrustScorer)

    def test_init_progress_reporter_default_is_null_reporter(self):
        from missy.agent.progress import NullReporter

        runtime = _make_runtime()
        assert isinstance(runtime._progress, NullReporter)

    def test_init_accepts_custom_progress_reporter(self):
        custom_reporter = MagicMock()
        provider = _make_provider()
        reg = _make_mock_registry(provider)
        cfg = AgentConfig(provider="fake")

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
            patch("missy.agent.runtime.get_message_bus", side_effect=RuntimeError),
        ):
            runtime = AgentRuntime(cfg, progress_reporter=custom_reporter)

        assert runtime._progress is custom_reporter

    def test_init_capability_mode_stored(self):
        provider = _make_provider()
        reg = _make_mock_registry(provider)
        cfg = AgentConfig(provider="fake", capability_mode="discord")

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
            patch("missy.agent.runtime.get_message_bus", side_effect=RuntimeError),
        ):
            runtime = AgentRuntime(cfg)

        assert runtime.config.capability_mode == "discord"

    def test_init_pending_recovery_is_list(self):
        runtime = _make_runtime()
        assert isinstance(runtime._pending_recovery, list)

    def test_pending_recovery_property(self):
        runtime = _make_runtime()
        result = runtime.pending_recovery
        assert isinstance(result, list)

    def test_pending_recovery_returns_copy(self):
        runtime = _make_runtime()
        r1 = runtime.pending_recovery
        r2 = runtime.pending_recovery
        assert r1 is not r2  # fresh copy each time


# ---------------------------------------------------------------------------
# 6. _bus_publish
# ---------------------------------------------------------------------------


class TestBusPublish:
    def test_bus_publish_no_op_when_bus_is_none(self):
        runtime = _make_runtime()
        runtime._message_bus = None
        # Must not raise
        runtime._bus_publish("some.topic", {"key": "value"})

    def test_bus_publish_calls_publish_on_bus(self):
        runtime = _make_runtime()
        mock_bus = MagicMock()
        runtime._message_bus = mock_bus
        runtime._bus_publish("agent.run.start", {"session_id": "s1"})
        mock_bus.publish.assert_called_once()

    def test_bus_publish_passes_correct_topic(self):
        runtime = _make_runtime()
        mock_bus = MagicMock()
        runtime._message_bus = mock_bus
        runtime._bus_publish("my.topic", {"x": 1})
        call_arg = mock_bus.publish.call_args[0][0]
        assert call_arg.topic == "my.topic"

    def test_bus_publish_passes_payload(self):
        runtime = _make_runtime()
        mock_bus = MagicMock()
        runtime._message_bus = mock_bus
        runtime._bus_publish("t", {"foo": "bar"}, source="test")
        call_arg = mock_bus.publish.call_args[0][0]
        assert call_arg.payload == {"foo": "bar"}

    def test_bus_publish_passes_source(self):
        runtime = _make_runtime()
        mock_bus = MagicMock()
        runtime._message_bus = mock_bus
        runtime._bus_publish("t", {}, source="agent-test")
        call_arg = mock_bus.publish.call_args[0][0]
        assert call_arg.source == "agent-test"

    def test_bus_publish_swallows_exception_from_bus(self):
        runtime = _make_runtime()
        mock_bus = MagicMock()
        mock_bus.publish.side_effect = RuntimeError("bus exploded")
        runtime._message_bus = mock_bus
        # Must not propagate
        runtime._bus_publish("t", {})

    def test_bus_publish_default_source_is_agent(self):
        runtime = _make_runtime()
        mock_bus = MagicMock()
        runtime._message_bus = mock_bus
        runtime._bus_publish("t", {})
        call_arg = mock_bus.publish.call_args[0][0]
        assert call_arg.source == "agent"


# ---------------------------------------------------------------------------
# 7. _make_message_bus
# ---------------------------------------------------------------------------


class TestMakeMessageBus:
    def test_make_message_bus_returns_none_on_exception(self):
        with patch("missy.agent.runtime.get_message_bus", side_effect=RuntimeError("no bus")):
            result = AgentRuntime._make_message_bus()
        assert result is None

    def test_make_message_bus_returns_bus_when_available(self):
        mock_bus = MagicMock()
        with patch("missy.agent.runtime.get_message_bus", return_value=mock_bus):
            result = AgentRuntime._make_message_bus()
        assert result is mock_bus

    def test_make_message_bus_returns_none_on_valueerror(self):
        with patch("missy.agent.runtime.get_message_bus", side_effect=ValueError("init failed")):
            result = AgentRuntime._make_message_bus()
        assert result is None

    def test_make_message_bus_returns_none_on_exception_subclass(self):
        class BusSetupError(Exception):
            pass

        with patch("missy.agent.runtime.get_message_bus", side_effect=BusSetupError):
            result = AgentRuntime._make_message_bus()
        assert result is None


# ---------------------------------------------------------------------------
# 8. switch_provider
# ---------------------------------------------------------------------------


class TestSwitchProvider:
    def test_switch_provider_updates_config(self):
        runtime = _make_runtime()
        mock_reg = MagicMock()

        with patch("missy.agent.runtime.get_registry", return_value=mock_reg):
            runtime.switch_provider("openai")

        assert runtime.config.provider == "openai"

    def test_switch_provider_calls_set_default(self):
        runtime = _make_runtime()
        mock_reg = MagicMock()

        with patch("missy.agent.runtime.get_registry", return_value=mock_reg):
            runtime.switch_provider("openai")

        mock_reg.set_default.assert_called_once_with("openai")

    def test_switch_provider_rebuilds_circuit_breaker(self):
        runtime = _make_runtime()
        old_cb = runtime._circuit_breaker
        mock_reg = MagicMock()

        with patch("missy.agent.runtime.get_registry", return_value=mock_reg):
            runtime.switch_provider("openai")

        # Circuit breaker should be a new object
        assert runtime._circuit_breaker is not old_cb

    def test_switch_provider_new_circuit_breaker_has_call(self):
        runtime = _make_runtime()
        mock_reg = MagicMock()

        with patch("missy.agent.runtime.get_registry", return_value=mock_reg):
            runtime.switch_provider("ollama")

        assert hasattr(runtime._circuit_breaker, "call")

    def test_switch_provider_propagates_registry_error(self):
        runtime = _make_runtime()

        with patch("missy.agent.runtime.get_registry", side_effect=ValueError("not found")), pytest.raises(ValueError):
            runtime.switch_provider("nonexistent")


# ---------------------------------------------------------------------------
# 9. capability_mode and _get_tools filtering
# ---------------------------------------------------------------------------


class TestCapabilityMode:
    def test_no_tools_mode_returns_empty(self):
        runtime = _make_runtime(capability_mode="no-tools")
        tools = runtime._get_tools()
        assert tools == []

    def test_full_mode_returns_all_tools(self):
        t1 = MagicMock()
        t1.name = "shell_exec"
        t2 = MagicMock()
        t2.name = "file_read"

        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["shell_exec", "file_read"]
        tool_reg.get.side_effect = {"shell_exec": t1, "file_read": t2}.get

        runtime = _make_runtime(capability_mode="full")
        with patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg):
            tools = runtime._get_tools()

        assert len(tools) == 2

    def test_safe_chat_mode_filters_tools(self):
        t_calc = MagicMock()
        t_calc.name = "calculator"
        t_shell = MagicMock()
        t_shell.name = "shell_exec"

        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calculator", "shell_exec"]
        tool_reg.get.side_effect = {"calculator": t_calc, "shell_exec": t_shell}.get

        runtime = _make_runtime(capability_mode="safe-chat")
        with patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg):
            tools = runtime._get_tools()

        tool_names = [t.name for t in tools]
        assert "calculator" in tool_names
        assert "shell_exec" not in tool_names

    def test_discord_mode_excludes_x11_tools(self):
        t_x11 = MagicMock()
        t_x11.name = "x11_screenshot"
        t_file = MagicMock()
        t_file.name = "file_read"

        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["x11_screenshot", "file_read"]
        tool_reg.get.side_effect = {"x11_screenshot": t_x11, "file_read": t_file}.get

        runtime = _make_runtime(capability_mode="discord")
        with patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg):
            tools = runtime._get_tools()

        tool_names = [t.name for t in tools]
        assert "x11_screenshot" not in tool_names
        assert "file_read" in tool_names

    def test_get_tools_returns_empty_list_when_registry_unavailable(self):
        runtime = _make_runtime()
        with patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError("no reg")):
            tools = runtime._get_tools()
        assert tools == []


# ---------------------------------------------------------------------------
# 10. _NoOpCircuitBreaker
# ---------------------------------------------------------------------------


class TestNoOpCircuitBreaker:
    def test_no_op_breaker_calls_func(self):
        from missy.agent.runtime import _NoOpCircuitBreaker

        breaker = _NoOpCircuitBreaker()
        result = breaker.call(lambda x: x * 2, 21)
        assert result == 42

    def test_no_op_breaker_passes_kwargs(self):
        from missy.agent.runtime import _NoOpCircuitBreaker

        breaker = _NoOpCircuitBreaker()
        result = breaker.call(lambda a, b=0: a + b, 10, b=5)
        assert result == 15

    def test_no_op_breaker_propagates_exceptions(self):
        from missy.agent.runtime import _NoOpCircuitBreaker

        breaker = _NoOpCircuitBreaker()
        with pytest.raises(ValueError, match="boom"):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("boom")))


# ---------------------------------------------------------------------------
# 11. _make_circuit_breaker static method
# ---------------------------------------------------------------------------


class TestMakeCircuitBreaker:
    def test_make_circuit_breaker_returns_object_with_call(self):
        cb = AgentRuntime._make_circuit_breaker("test_provider")
        assert hasattr(cb, "call")

    def test_make_circuit_breaker_different_names(self):
        cb1 = AgentRuntime._make_circuit_breaker("anthropic")
        cb2 = AgentRuntime._make_circuit_breaker("openai")
        # Both should have a call method; they are separate instances
        assert hasattr(cb1, "call")
        assert hasattr(cb2, "call")
        assert cb1 is not cb2


# ---------------------------------------------------------------------------
# 12. AgentConfig is a proper dataclass
# ---------------------------------------------------------------------------


class TestAgentConfigIsDataclass:
    def test_config_supports_equality(self):
        c1 = AgentConfig(provider="x", max_iterations=3)
        c2 = AgentConfig(provider="x", max_iterations=3)
        assert c1 == c2

    def test_config_inequality(self):
        c1 = AgentConfig(provider="x")
        c2 = AgentConfig(provider="y")
        assert c1 != c2

    def test_config_fields_are_mutable(self):
        cfg = AgentConfig()
        cfg.provider = "changed"
        assert cfg.provider == "changed"
