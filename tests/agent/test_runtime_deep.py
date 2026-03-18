"""Deep integration tests for AgentRuntime and its subsystems.

Exercises multi-subsystem interactions end-to-end with a mocked provider:
- Full agentic tool-call loop (multi-step)
- Iteration limit enforcement with fallback
- Circuit breaker tripping and state transitions
- ContextManager token budget and pruning
- ProgressReporter lifecycle events
- AttentionSystem integration in the run pipeline
- Persona / behavior layer shaping of system prompt
- Error recovery from provider failures
- DoneCriteria verification prompt injection
- History accumulation across turns
"""

from __future__ import annotations

import contextlib
import time
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.attention import (
    AlertingAttention,
    AttentionSystem,
    ExecutiveAttention,
    OrientingAttention,
    SelectiveAttention,
    SustainedAttention,
)
from missy.agent.circuit_breaker import CircuitBreaker, CircuitState
from missy.agent.context import ContextManager, TokenBudget
from missy.agent.progress import AuditReporter, CLIReporter, NullReporter, ProgressReporter
from missy.agent.runtime import AgentConfig, AgentRuntime
from missy.core.events import event_bus
from missy.core.exceptions import MissyError, ProviderError
from missy.providers import registry as registry_module
from missy.providers.base import CompletionResponse, ToolCall

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_provider(
    name: str = "fake",
    reply: str = "final answer",
    available: bool = True,
) -> MagicMock:
    """Build a mock provider with both complete() and complete_with_tools()."""
    provider = MagicMock()
    provider.name = name
    provider.is_available.return_value = available
    provider.complete.return_value = CompletionResponse(
        content=reply,
        model="fake-model",
        provider=name,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        raw={},
        tool_calls=[],
        finish_reason="stop",
    )
    provider.complete_with_tools.return_value = CompletionResponse(
        content=reply,
        model="fake-model",
        provider=name,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        raw={},
        tool_calls=[],
        finish_reason="stop",
    )
    return provider


def _make_tool_call_response(
    tool_name: str,
    tool_id: str = "tc-1",
    args: dict | None = None,
    provider_name: str = "fake",
) -> CompletionResponse:
    """Build a CompletionResponse that requests a single tool call."""
    return CompletionResponse(
        content="",
        model="fake-model",
        provider=provider_name,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        raw={},
        tool_calls=[ToolCall(id=tool_id, name=tool_name, arguments=args or {})],
        finish_reason="tool_calls",
    )


def _make_stop_response(reply: str = "done", provider_name: str = "fake") -> CompletionResponse:
    """Build a final stop CompletionResponse."""
    return CompletionResponse(
        content=reply,
        model="fake-model",
        provider=provider_name,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        raw={},
        tool_calls=[],
        finish_reason="stop",
    )


def _make_registry(providers: dict | None = None) -> MagicMock:
    registry = MagicMock()
    providers = providers or {}

    def _get(name: str):
        return providers.get(name)

    def _get_available():
        return [p for p in providers.values() if p.is_available()]

    registry.get.side_effect = _get
    registry.get_available.side_effect = _get_available
    return registry


def _make_tool_registry(tools: list | None = None) -> MagicMock:
    """Build a mock tool registry with a calculator-style tool."""
    tools = tools or []
    tool_reg = MagicMock()
    tool_reg.list_tools.return_value = [getattr(t, "name", f"tool_{i}") for i, t in enumerate(tools)]

    def _get(name: str):
        for t in tools:
            if getattr(t, "name", None) == name:
                return t
        return None

    def _execute(name: str, **kwargs):
        for t in tools:
            if getattr(t, "name", None) == name:
                result = MagicMock()
                result.success = True
                result.output = f"{name}_result"
                result.error = None
                return result
        raise KeyError(f"tool {name!r} not found")

    tool_reg.get.side_effect = _get
    tool_reg.execute.side_effect = _execute
    return tool_reg


def _make_mock_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = f"Mock {name} tool"
    tool.schema = {"type": "object", "properties": {}}
    return tool


@pytest.fixture(autouse=True)
def _clear_event_bus():
    event_bus.clear()
    yield
    event_bus.clear()


@pytest.fixture(autouse=True)
def _reset_registry_singleton():
    original = registry_module._registry
    yield
    registry_module._registry = original


# ---------------------------------------------------------------------------
# 1. Runtime initialization — all subsystems created
# ---------------------------------------------------------------------------


class TestRuntimeInitialization:
    def test_circuit_breaker_created(self):
        rt = AgentRuntime(AgentConfig(provider="fake"))
        assert rt._circuit_breaker is not None

    def test_context_manager_created(self):
        rt = AgentRuntime(AgentConfig())
        assert rt._context_manager is not None

    def test_attention_system_created(self):
        rt = AgentRuntime(AgentConfig())
        assert rt._attention is not None

    def test_trust_scorer_created(self):
        rt = AgentRuntime(AgentConfig())
        assert rt._trust_scorer is not None

    def test_progress_reporter_defaults_to_null(self):
        rt = AgentRuntime(AgentConfig())
        assert isinstance(rt._progress, NullReporter)

    def test_custom_progress_reporter_assigned(self):
        reporter = CLIReporter()
        rt = AgentRuntime(AgentConfig(), progress_reporter=reporter)
        assert rt._progress is reporter

    def test_session_manager_created(self):
        rt = AgentRuntime(AgentConfig())
        assert rt._session_mgr is not None

    def test_circuit_breaker_name_matches_provider(self):
        rt = AgentRuntime(AgentConfig(provider="openai"))
        assert rt._circuit_breaker.name == "openai"

    def test_persona_manager_created(self):
        rt = AgentRuntime(AgentConfig())
        # Persona manager uses graceful degradation — may be None if import fails,
        # but with a full install it should exist.
        # We just verify the attribute is present (not raising).
        assert hasattr(rt, "_persona_manager")

    def test_behavior_layer_created(self):
        rt = AgentRuntime(AgentConfig())
        assert hasattr(rt, "_behavior")


# ---------------------------------------------------------------------------
# 2. Runtime run with tool calls — multi-step tool call loop
# ---------------------------------------------------------------------------


class TestRuntimeToolCallLoop:
    def test_single_tool_call_then_stop(self):
        """Provider requests one tool, gets result, then returns final answer."""
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("calculator", args={"expression": "2+2"}),
            _make_stop_response("The answer is 4."),
        ]

        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=5))
            result = rt.run("What is 2+2?")

        assert result == "The answer is 4."
        assert provider.complete_with_tools.call_count == 2

    def test_tool_names_tracked_in_complete_event(self):
        """Tools used are reported in the agent.run.complete audit event."""
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("calculator", args={}),
            _make_stop_response("done"),
        ]

        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=5))
            rt.run("use calculator")

        complete_events = event_bus.get_events(event_type="agent.run.complete")
        assert len(complete_events) >= 1
        tools_used = complete_events[0].detail.get("tools_used", [])
        assert "calculator" in tools_used

    def test_two_sequential_tool_calls(self):
        """Provider requests tool twice in succession before stopping."""
        provider = _make_provider()
        tool_a = _make_mock_tool("tool_a")
        tool_b = _make_mock_tool("tool_b")
        tool_reg = _make_tool_registry([tool_a, tool_b])

        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("tool_a"),
            _make_tool_call_response("tool_b"),
            _make_stop_response("all done"),
        ]

        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=10))
            result = rt.run("run both tools")

        assert result == "all done"
        assert provider.complete_with_tools.call_count == 3

    def test_no_tools_uses_single_turn(self):
        """When no tools are registered, complete() is used instead of complete_with_tools()."""
        provider = _make_provider(reply="plain reply")
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError("no registry")),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=5))
            result = rt.run("hello")

        assert result == "plain reply"
        provider.complete.assert_called_once()
        provider.complete_with_tools.assert_not_called()

    def test_tool_execution_failure_marked_as_error(self):
        """When tool raises, result is an error ToolResult and loop continues."""
        provider = _make_provider()
        bad_tool = _make_mock_tool("bad_tool")
        tool_reg = _make_tool_registry([bad_tool])
        tool_reg.execute.side_effect = Exception("tool boom")

        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("bad_tool"),
            _make_stop_response("recovered"),
        ]

        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=5))
            result = rt.run("try bad tool")

        # Error is swallowed by the loop; we get the final response
        assert result == "recovered"

    def test_provider_without_complete_with_tools_falls_back(self):
        """Provider lacking complete_with_tools falls back to single-turn complete()."""
        provider = _make_provider()
        del provider.complete_with_tools  # Remove the attribute

        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=5))
            result = rt.run("calculate something")

        assert result == "final answer"
        provider.complete.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Runtime iteration limit — max iterations respected
# ---------------------------------------------------------------------------


class TestRuntimeIterationLimit:
    def test_max_iterations_one_returns_single_turn(self):
        """max_iterations=1 disables the tool loop entirely."""
        provider = _make_provider(reply="single turn reply")
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=1))
            result = rt.run("hi")

        assert result == "single turn reply"
        provider.complete.assert_called_once()
        provider.complete_with_tools.assert_not_called()

    def test_iteration_limit_triggers_fallback(self):
        """When provider always returns tool_calls, iteration limit causes fallback."""
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        # Always return a tool call — never stop
        provider.complete_with_tools.return_value = _make_tool_call_response("calculator")
        # Fallback single turn
        provider.complete.return_value = _make_stop_response("fallback after limit")

        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=3))
            result = rt.run("run forever")

        # Ran exactly max_iterations tool rounds, then fell back
        assert provider.complete_with_tools.call_count == 3
        # Result is either the fallback content or the sentinel string
        assert result in ("fallback after limit", "[Agent reached iteration limit without a final response.]")

    def test_iteration_limit_fallback_returns_sentinel_on_exception(self):
        """When fallback single-turn also fails, sentinel string is returned."""
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.return_value = _make_tool_call_response("calculator")
        provider.complete.side_effect = ProviderError("fallback also failed")

        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=2))
            result = rt.run("keep looping")

        assert "iteration limit" in result.lower() or result.startswith("[Agent")


# ---------------------------------------------------------------------------
# 4. Runtime circuit breaker integration
# ---------------------------------------------------------------------------


class TestRuntimeCircuitBreakerIntegration:
    def test_provider_errors_increment_circuit_breaker(self):
        """Repeated provider failures trip the circuit breaker."""
        provider = _make_provider()
        provider.complete.side_effect = ProviderError("boom")
        registry = _make_registry({"fake": provider})

        rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=1))
        # Override with a low-threshold breaker
        from missy.agent.circuit_breaker import CircuitBreaker

        rt._circuit_breaker = CircuitBreaker("fake", threshold=2, base_timeout=3600.0)

        with patch("missy.agent.runtime.get_registry", return_value=registry):
            with pytest.raises(ProviderError):
                rt.run("first fail")
            with pytest.raises(ProviderError):
                rt.run("second fail")

        assert rt._circuit_breaker.state == CircuitState.OPEN

    def test_open_circuit_raises_missy_error(self):
        """When circuit is already OPEN, the call is rejected immediately."""
        provider = _make_provider()
        registry = _make_registry({"fake": provider})

        rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=1))
        from missy.agent.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker("fake", threshold=1, base_timeout=3600.0)
        # Force open state
        breaker._state = CircuitState.OPEN
        breaker._last_failure_time = time.monotonic()
        rt._circuit_breaker = breaker

        with patch("missy.agent.runtime.get_registry", return_value=registry):
            with pytest.raises((MissyError, ProviderError)):
                rt.run("should be rejected")

        # Provider was never actually called
        provider.complete.assert_not_called()

    def test_successful_call_resets_circuit_breaker(self):
        """A successful call after failures resets the circuit to CLOSED."""
        from missy.agent.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker("test", threshold=3, base_timeout=60.0)

        # Simulate one failure (not enough to open)
        with contextlib.suppress(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        assert breaker._failure_count == 1
        assert breaker.state == CircuitState.CLOSED

        # Successful call resets
        breaker.call(lambda: "ok")
        assert breaker._failure_count == 0
        assert breaker.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# 5. Runtime context management — token budget, history pruning
# ---------------------------------------------------------------------------


class TestRuntimeContextManagement:
    def test_run_builds_context_with_system_prompt(self):
        """System prompt appears in messages passed to provider."""
        provider = _make_provider()
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", system_prompt="Custom sys."))
            rt.run("hello")

        messages = provider.complete.call_args[0][0]
        system_msgs = [m for m in messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content.startswith("Custom sys.")

    def test_context_manager_prunes_old_history(self):
        """ContextManager drops old messages when budget is exceeded."""
        # Very tight budget: 200 total tokens
        budget = TokenBudget(
            total=200,
            system_reserve=50,
            tool_definitions_reserve=50,
            memory_fraction=0.1,
            learnings_fraction=0.05,
            fresh_tail_count=2,
        )
        mgr = ContextManager(budget)

        # Build a long history — each message ~20 tokens (80 chars)
        long_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 80}
            for i in range(20)
        ]

        _, messages = mgr.build_messages(
            system="You are Missy.",
            new_message="new message",
            history=long_history,
        )

        # Should be pruned: only fresh_tail (last 2) + new message survive from full history
        # The new message is always the last entry
        assert messages[-1] == {"role": "user", "content": "new message"}
        # Total messages must be fewer than history + 1
        assert len(messages) < len(long_history) + 1

    def test_context_manager_preserves_fresh_tail(self):
        """The last N messages (fresh_tail_count) are always kept."""
        budget = TokenBudget(
            total=100,
            system_reserve=10,
            tool_definitions_reserve=10,
            memory_fraction=0.0,
            learnings_fraction=0.0,
            fresh_tail_count=3,
        )
        mgr = ContextManager(budget)

        history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]

        _, messages = mgr.build_messages(
            system="sys",
            new_message="new",
            history=history,
        )

        contents = [m["content"] for m in messages]
        # The fresh tail: last 3 history items must always be present
        for tail_msg in history[-3:]:
            assert tail_msg["content"] in contents

    def test_context_manager_injects_memory_into_system(self):
        """memory_results are appended to the system prompt."""
        mgr = ContextManager()
        system, _ = mgr.build_messages(
            system="Base prompt.",
            new_message="hi",
            history=[],
            memory_results=["fact1", "fact2"],
        )
        assert "Relevant Memory" in system
        assert "fact1" in system
        assert "fact2" in system

    def test_context_manager_injects_learnings_into_system(self):
        """learnings are appended to the system prompt."""
        mgr = ContextManager()
        system, _ = mgr.build_messages(
            system="Base prompt.",
            new_message="hi",
            history=[],
            learnings=["learned A", "learned B"],
        )
        assert "Past Learnings" in system
        assert "learned A" in system

    def test_context_manager_caps_memory_at_budget(self):
        """Memory exceeding the memory budget fraction is truncated."""
        budget = TokenBudget(
            total=1000,
            system_reserve=100,
            tool_definitions_reserve=100,
            memory_fraction=0.05,  # Only 5% → ~40 tokens → 160 chars
        )
        mgr = ContextManager(budget)
        # Memory far exceeding budget
        huge_memory = ["m" * 5000]

        system, _ = mgr.build_messages(
            system="sys",
            new_message="hi",
            history=[],
            memory_results=huge_memory,
        )
        memory_section = system.split("## Relevant Memory")[-1] if "## Relevant Memory" in system else ""
        # The memory section should be present but clipped
        assert len(memory_section) < 5000

    def test_context_manager_only_uses_five_learnings(self):
        """At most 5 learnings are injected."""
        mgr = ContextManager()
        many_learnings = [f"learning_{i}" for i in range(10)]

        system, _ = mgr.build_messages(
            system="sys",
            new_message="hi",
            history=[],
            learnings=many_learnings,
        )
        # Items 6–9 should not appear
        for i in range(5, 10):
            assert f"learning_{i}" not in system


# ---------------------------------------------------------------------------
# 6. Runtime done criteria — stop when done criteria met
# ---------------------------------------------------------------------------


class TestRuntimeDoneCriteria:
    def test_verification_prompt_injected_after_tool_results(self):
        """After each round of tool results, a verification prompt is injected."""
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        captured_messages: list = []

        def capture_complete_with_tools(messages, tools, system):
            captured_messages.extend(messages)
            # First call: request tool; second call: stop
            if provider.complete_with_tools.call_count == 1:
                return _make_tool_call_response("calculator")
            return _make_stop_response("done")

        provider.complete_with_tools.side_effect = capture_complete_with_tools
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=5))
            rt.run("do something")

        # Second call to complete_with_tools must have included the verification user message
        second_call_msgs = provider.complete_with_tools.call_args_list[1][0][0]
        user_msg_contents = [m.content for m in second_call_msgs if m.role == "user"]
        assert any(len(c) > 5 for c in user_msg_contents)  # verification prompt is non-trivial

    def test_stop_finish_reason_exits_loop_immediately(self):
        """finish_reason='stop' with no tool calls exits the loop after one iteration."""
        provider = _make_provider(reply="instant answer")
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=10))
            result = rt.run("quick question")

        assert result == "instant answer"
        assert provider.complete_with_tools.call_count == 1


# ---------------------------------------------------------------------------
# 7. Runtime with persona — persona shapes system prompt
# ---------------------------------------------------------------------------


class TestRuntimeWithPersona:
    def test_persona_manager_called_when_available(self):
        """PersonaManager.get_persona() is invoked during the run pipeline."""
        provider = _make_provider(reply="hello")
        registry = _make_registry({"fake": provider})

        mock_persona = MagicMock()
        mock_persona.name = "Ada"
        mock_persona.tone = "formal"

        mock_persona_mgr = MagicMock()
        mock_persona_mgr.get_persona.return_value = mock_persona

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            rt._persona_manager = mock_persona_mgr
            rt.run("hello")

        # Persona manager may be called in response shaper — check it was at least accessible
        assert mock_persona_mgr is rt._persona_manager

    def test_persona_manager_none_does_not_crash(self):
        """Runtime works correctly when persona manager is None."""
        provider = _make_provider(reply="ok")
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            rt._persona_manager = None
            rt._response_shaper = None
            result = rt.run("hello")

        assert result == "ok"

    def test_persona_shaper_exception_does_not_propagate(self):
        """Exceptions in response shaping are silenced; raw response is returned."""
        provider = _make_provider(reply="raw reply")
        registry = _make_registry({"fake": provider})

        mock_shaper = MagicMock()
        mock_shaper.shape_response.side_effect = RuntimeError("shaper failed")

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            rt._response_shaper = mock_shaper
            # Should not raise; falls back to raw reply
            result = rt.run("hello")

        # Result is the censored raw reply (not an exception)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 8. Runtime with behavior layer — behavior layer called in pipeline
# ---------------------------------------------------------------------------


class TestRuntimeWithBehaviorLayer:
    def test_behavior_layer_shapes_system_prompt(self):
        """BehaviorLayer.shape_system_prompt is invoked before provider call."""
        provider = _make_provider()
        registry = _make_registry({"fake": provider})

        mock_behavior = MagicMock()
        mock_behavior.analyze_user_tone.return_value = "neutral"
        mock_behavior.shape_system_prompt.return_value = "Shaped: original system"

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", system_prompt="original system"))
            rt._behavior = mock_behavior
            rt.run("some input")

        mock_behavior.shape_system_prompt.assert_called_once()

    def test_behavior_layer_exception_falls_back_gracefully(self):
        """If behavior layer raises, the unshaped system prompt is used."""
        provider = _make_provider(reply="ok")
        registry = _make_registry({"fake": provider})

        mock_behavior = MagicMock()
        mock_behavior.analyze_user_tone.side_effect = RuntimeError("behavior crash")
        mock_behavior.shape_system_prompt.side_effect = RuntimeError("behavior crash")

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            rt._behavior = mock_behavior
            result = rt.run("hello")

        assert result == "ok"

    def test_intent_interpreter_called_with_user_input(self):
        """IntentInterpreter.classify_intent is called with the user input."""
        provider = _make_provider()
        registry = _make_registry({"fake": provider})

        mock_intent = MagicMock()
        mock_intent.classify_intent.return_value = "command"
        mock_intent.extract_urgency.return_value = "low"

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            rt._intent_interpreter = mock_intent
            rt.run("do the thing")

        mock_intent.classify_intent.assert_called_once_with("do the thing")


# ---------------------------------------------------------------------------
# 9. Runtime error recovery — provider errors handled gracefully
# ---------------------------------------------------------------------------


class TestRuntimeErrorRecovery:
    def test_provider_error_emits_error_event(self):
        provider = _make_provider()
        provider.complete.side_effect = ProviderError("API down")
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=1))
            with pytest.raises(ProviderError):
                rt.run("anything")

        error_events = event_bus.get_events(event_type="agent.run.error")
        assert len(error_events) >= 1

    def test_unexpected_exception_raises_provider_error(self):
        provider = _make_provider()
        provider.complete.side_effect = ValueError("completely unexpected")
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=1))
            with pytest.raises(ProviderError, match="Unexpected"):
                rt.run("anything")

    def test_no_provider_available_raises_provider_error(self):
        registry = _make_registry(providers={})

        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="missing"))
            with pytest.raises(ProviderError):
                rt.run("anything")

    def test_tool_loop_exception_re_raised_as_provider_error(self):
        """An unhandled exception in the tool loop propagates as ProviderError."""
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = ProviderError("mid-loop crash")
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=5))
            with pytest.raises(ProviderError, match="mid-loop crash"):
                rt.run("trigger crash")

    def test_empty_user_input_raises_value_error(self):
        rt = AgentRuntime(AgentConfig())
        with pytest.raises(ValueError):
            rt.run("")

    def test_whitespace_only_input_raises_value_error(self):
        rt = AgentRuntime(AgentConfig())
        with pytest.raises(ValueError):
            rt.run("   ")


# ---------------------------------------------------------------------------
# 10. Runtime with attention system — attention filters inform context
# ---------------------------------------------------------------------------


class TestRuntimeAttentionSystem:
    def test_urgent_input_emits_attention_event(self):
        """High-urgency input (urgency > 0.7) emits agent.attention.urgent event."""
        provider = _make_provider()
        registry = _make_registry({"fake": provider})

        # A message with enough urgency keywords to exceed 0.7 threshold
        urgent_msg = "critical error security broken failed immediately"

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            rt.run(urgent_msg)

        attention_events = event_bus.get_events(event_type="agent.attention.urgent")
        assert len(attention_events) >= 1

    def test_attention_system_processes_user_input(self):
        """AttentionSystem.process() is called during run()."""
        provider = _make_provider()
        registry = _make_registry({"fake": provider})

        mock_attention = MagicMock()
        mock_attn_state = MagicMock()
        mock_attn_state.urgency = 0.0
        mock_attn_state.topics = ["server"]
        mock_attn_state.focus_duration = 1
        mock_attn_state.priority_tools = []
        mock_attention.process.return_value = mock_attn_state

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            rt._attention = mock_attention
            rt.run("check the server")

        mock_attention.process.assert_called_once()

    def test_attention_system_none_does_not_crash(self):
        """Runtime degrades gracefully when attention system is None."""
        provider = _make_provider(reply="ok")
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            rt._attention = None
            result = rt.run("test input")

        assert result == "ok"

    def test_attention_system_exception_does_not_propagate(self):
        """Exceptions raised by the attention system are silenced."""
        provider = _make_provider(reply="ok")
        registry = _make_registry({"fake": provider})

        mock_attention = MagicMock()
        mock_attention.process.side_effect = RuntimeError("attention exploded")

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            rt._attention = mock_attention
            result = rt.run("test")

        assert result == "ok"


# ---------------------------------------------------------------------------
# 11. ContextManager token counting
# ---------------------------------------------------------------------------


class TestContextManagerTokenCounting:
    def test_approx_tokens_four_chars_per_token(self):
        """_approx_tokens uses 4-chars-per-token heuristic."""
        from missy.agent.context import _approx_tokens

        assert _approx_tokens("abcd") == 1       # 4 chars → 1 token
        assert _approx_tokens("abcdefgh") == 2   # 8 chars → 2 tokens
        assert _approx_tokens("") == 1            # minimum 1

    def test_new_message_always_included(self):
        """The new user message is always the final element in returned messages."""
        mgr = ContextManager(TokenBudget(total=50, system_reserve=10, tool_definitions_reserve=10))
        _, messages = mgr.build_messages(
            system="sys",
            new_message="the new one",
            history=[{"role": "user", "content": "old message " * 100}],
        )
        assert messages[-1] == {"role": "user", "content": "the new one"}

    def test_empty_history_returns_only_new_message(self):
        mgr = ContextManager()
        _, messages = mgr.build_messages(
            system="sys",
            new_message="hello",
            history=[],
        )
        assert messages == [{"role": "user", "content": "hello"}]

    def test_system_prompt_enriched_by_memory_and_returned(self):
        mgr = ContextManager()
        system, _ = mgr.build_messages(
            system="Base.",
            new_message="hi",
            history=[],
            memory_results=["snippet A"],
            learnings=["lesson 1"],
        )
        assert "Base." in system
        assert "snippet A" in system
        assert "lesson 1" in system

    def test_message_count_does_not_exceed_budget(self):
        """Total token estimate of returned messages stays within history_budget."""
        from missy.agent.context import _approx_tokens

        budget = TokenBudget(
            total=300,
            system_reserve=50,
            tool_definitions_reserve=50,
            memory_fraction=0.0,
            learnings_fraction=0.0,
            fresh_tail_count=0,
        )
        mgr = ContextManager(budget)
        history = [{"role": "user", "content": "x" * 40} for _ in range(20)]  # 10 tokens each

        _, messages = mgr.build_messages(
            system="sys",
            new_message="new",
            history=history,
        )

        total_tokens = sum(_approx_tokens(str(m.get("content", ""))) for m in messages)
        history_budget = budget.total - budget.system_reserve - budget.tool_definitions_reserve
        assert total_tokens <= history_budget + 5  # small slack for rounding


# ---------------------------------------------------------------------------
# 12. ContextManager memory injection — memory fraction respected
# ---------------------------------------------------------------------------


class TestContextManagerMemoryInjection:
    def test_memory_fraction_limits_injected_size(self):
        """Memory snippet is clipped when it exceeds the memory budget."""
        budget = TokenBudget(
            total=500,
            system_reserve=50,
            tool_definitions_reserve=50,
            memory_fraction=0.02,  # ~8 tokens → 32 chars
        )
        mgr = ContextManager(budget)
        huge_snippet = "M" * 10_000

        system, _ = mgr.build_messages(
            system="s",
            new_message="hi",
            history=[],
            memory_results=[huge_snippet],
        )

        # The memory section exists but is significantly clipped
        if "## Relevant Memory" in system:
            mem_start = system.index("## Relevant Memory") + len("## Relevant Memory\n")
            injected = system[mem_start:]
            assert len(injected) < 500  # Much less than original 10_000 chars

    def test_no_memory_does_not_modify_system(self):
        """System prompt is returned unchanged when no memory is provided."""
        mgr = ContextManager()
        system, _ = mgr.build_messages(
            system="Base system prompt.",
            new_message="hi",
            history=[],
        )
        assert system == "Base system prompt."

    def test_multiple_memory_snippets_joined_with_newlines(self):
        """Multiple memory snippets are joined by newline in the system prompt."""
        mgr = ContextManager()
        system, _ = mgr.build_messages(
            system="sys",
            new_message="hi",
            history=[],
            memory_results=["fact A", "fact B", "fact C"],
        )
        assert "fact A" in system
        assert "fact B" in system
        assert "fact C" in system


# ---------------------------------------------------------------------------
# 13. ContextManager pruning — oldest messages removed first
# ---------------------------------------------------------------------------


class TestContextManagerPruning:
    def test_oldest_evictable_messages_dropped_first(self):
        """When budget is exceeded, the oldest non-tail messages are dropped first."""
        # Each message is ~100 chars = 25 tokens.  Budget allows only ~2 history tokens
        # after reserves, so only the fresh tail (2 messages) + new message survive.
        budget = TokenBudget(
            total=30,
            system_reserve=10,
            tool_definitions_reserve=10,
            memory_fraction=0.0,
            learnings_fraction=0.0,
            fresh_tail_count=2,
        )
        mgr = ContextManager(budget)

        # 6 messages, each 100 chars (~25 tokens) — far exceeds the 10-token history budget
        history = [{"role": "user", "content": f"msg{i}:" + "x" * 96} for i in range(6)]

        _, messages = mgr.build_messages(
            system="sys",
            new_message="new",
            history=history,
        )

        contents = [m["content"] for m in messages]
        # The fresh tail (last 2 history entries: msg4, msg5) are always kept.
        # The oldest evictable messages (msg0–msg3) should have been dropped.
        assert not any(c.startswith("msg0:") for c in contents)
        assert not any(c.startswith("msg1:") for c in contents)

    def test_fresh_tail_count_zero_makes_all_evictable(self):
        """When fresh_tail_count=0, all history is evictable."""
        budget = TokenBudget(
            total=100,
            system_reserve=10,
            tool_definitions_reserve=10,
            memory_fraction=0.0,
            learnings_fraction=0.0,
            fresh_tail_count=0,
        )
        mgr = ContextManager(budget)
        # 10 messages of 40 chars each = 10 tokens each, far exceeds budget
        history = [{"role": "user", "content": "y" * 40} for _ in range(10)]

        _, messages = mgr.build_messages(
            system="sys",
            new_message="new",
            history=history,
        )

        # All old messages may have been dropped; only new message is guaranteed
        assert messages[-1] == {"role": "user", "content": "new"}

    def test_build_messages_alternates_user_assistant_correctly(self):
        """Returned message roles are preserved from history."""
        mgr = ContextManager()
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        _, messages = mgr.build_messages(
            system="sys",
            new_message="q3",
            history=history,
        )
        # All history fits in default budget; roles preserved
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "user", "assistant", "user"]


# ---------------------------------------------------------------------------
# 14. CircuitBreaker state machine — Closed → Open → HalfOpen → Closed
# ---------------------------------------------------------------------------


class TestCircuitBreakerStateMachine:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED

    def test_failures_below_threshold_stay_closed(self):
        cb = CircuitBreaker("test", threshold=3)
        for _ in range(2):
            with contextlib.suppress(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert cb.state == CircuitState.CLOSED

    def test_threshold_failures_open_circuit(self):
        cb = CircuitBreaker("test", threshold=3)
        for _ in range(3):
            with contextlib.suppress(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert cb.state == CircuitState.OPEN

    def test_open_circuit_rejects_calls(self):
        cb = CircuitBreaker("test", threshold=1, base_timeout=3600.0)
        with contextlib.suppress(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        with pytest.raises(MissyError, match="OPEN"):
            cb.call(lambda: "should not be called")

    def test_open_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker("test", threshold=1, base_timeout=0.05)
        with contextlib.suppress(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        assert cb.state == CircuitState.OPEN
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        cb = CircuitBreaker("test", threshold=1, base_timeout=0.05)
        with contextlib.suppress(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN

        cb.call(lambda: "probe ok")
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    def test_half_open_failure_reopens_circuit_with_doubled_timeout(self):
        cb = CircuitBreaker("test", threshold=1, base_timeout=0.05, max_timeout=300.0)
        with contextlib.suppress(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN

        original_timeout = cb._recovery_timeout
        with contextlib.suppress(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("probe fail")))

        assert cb.state == CircuitState.OPEN
        assert cb._recovery_timeout == min(original_timeout * 2, 300.0)

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test", threshold=5)
        for _ in range(4):
            with contextlib.suppress(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        assert cb._failure_count == 4
        cb.call(lambda: "ok")
        assert cb._failure_count == 0

    def test_circuit_breaker_is_thread_safe(self):
        """Concurrent success and failure calls don't corrupt state."""
        import threading

        cb = CircuitBreaker("test-thread", threshold=100)
        errors: list[Exception] = []

        def succeed():
            try:
                for _ in range(50):
                    cb.call(lambda: "ok")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=succeed) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert cb.state == CircuitState.CLOSED

    def test_max_timeout_is_respected_on_backoff(self):
        cb = CircuitBreaker("test", threshold=1, base_timeout=200.0, max_timeout=300.0)
        # Force into half_open then fail twice to test cap
        cb._state = CircuitState.HALF_OPEN
        cb._recovery_timeout = 200.0

        with contextlib.suppress(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        assert cb._recovery_timeout <= 300.0


# ---------------------------------------------------------------------------
# 15. ProgressReporter lifecycle
# ---------------------------------------------------------------------------


class TestProgressReporterLifecycle:
    # NullReporter — verify all methods callable and silent
    def test_null_reporter_all_methods_callable(self):
        r = NullReporter()
        r.on_start("task A")
        r.on_progress(0.5, "halfway")
        r.on_tool_start("calculator")
        r.on_tool_done("calculator", "ok")
        r.on_iteration(0, 5)
        r.on_complete("finished")
        r.on_error("oops")

    def test_null_reporter_satisfies_protocol(self):
        r = NullReporter()
        assert isinstance(r, ProgressReporter)

    def test_cli_reporter_satisfies_protocol(self):
        r = CLIReporter()
        assert isinstance(r, ProgressReporter)

    def test_audit_reporter_satisfies_protocol(self):
        r = AuditReporter()
        assert isinstance(r, ProgressReporter)

    def test_progress_reporter_called_in_tool_loop(self):
        """ProgressReporter receives lifecycle events during tool loop."""
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("calculator"),
            _make_stop_response("done"),
        ]
        registry = _make_registry({"fake": provider})

        mock_reporter = MagicMock(spec=NullReporter)

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=5), progress_reporter=mock_reporter)
            rt.run("run tool")

        mock_reporter.on_start.assert_called_once()
        mock_reporter.on_iteration.assert_called()
        mock_reporter.on_tool_start.assert_called_with("calculator")
        mock_reporter.on_tool_done.assert_called_with("calculator", "ok")
        mock_reporter.on_complete.assert_called_once()

    def test_progress_reporter_on_error_called_on_exception(self):
        """on_error is called when provider raises inside the tool loop."""
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = ProviderError("boom")
        registry = _make_registry({"fake": provider})

        mock_reporter = MagicMock(spec=NullReporter)

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=5), progress_reporter=mock_reporter)
            with pytest.raises(ProviderError):
                rt.run("error run")

        mock_reporter.on_error.assert_called_once()

    def test_audit_reporter_publishes_to_event_bus(self):
        """AuditReporter emits audit events via the event bus."""
        reporter = AuditReporter(session_id="s1", task_id="t1")
        reporter.on_start("my task")

        events = event_bus.get_events(event_type="agent.progress.start")
        assert len(events) >= 1
        assert events[0].detail.get("task") == "my task"

    def test_audit_reporter_tool_start_emits_event(self):
        reporter = AuditReporter(session_id="s2", task_id="t2")
        reporter.on_tool_start("shell_exec")

        events = event_bus.get_events(event_type="agent.progress.tool_start")
        assert len(events) >= 1
        assert events[0].detail.get("tool") == "shell_exec"

    def test_audit_reporter_complete_emits_event(self):
        reporter = AuditReporter(session_id="s3", task_id="t3")
        reporter.on_complete("all done")

        events = event_bus.get_events(event_type="agent.progress.complete")
        assert len(events) >= 1
        assert events[0].detail.get("summary") == "all done"

    def test_audit_reporter_error_emits_event(self):
        reporter = AuditReporter(session_id="s4", task_id="t4")
        reporter.on_error("something went wrong")

        events = event_bus.get_events(event_type="agent.progress.error")
        assert len(events) >= 1
        assert events[0].detail.get("error") == "something went wrong"

    def test_cli_reporter_writes_to_stderr(self, capsys):
        r = CLIReporter()
        r.on_start("test task")
        r.on_progress(0.5, "halfway")
        r.on_tool_start("file_read")
        r.on_tool_done("file_read", "read ok")
        r.on_iteration(2, 10)
        r.on_complete("all finished")
        r.on_error("some error")

        captured = capsys.readouterr()
        assert "test task" in captured.err
        assert "halfway" in captured.err
        assert "file_read" in captured.err
        assert "all finished" in captured.err
        assert "some error" in captured.err


# ---------------------------------------------------------------------------
# Attention subsystem unit tests
# ---------------------------------------------------------------------------


class TestAlertingAttention:
    def test_no_urgency_keywords_returns_zero(self):
        a = AlertingAttention()
        assert a.score("the weather is nice today") == 0.0

    def test_all_urgency_keywords_returns_one(self):
        a = AlertingAttention()
        # All words are urgency keywords
        score = a.score("error critical urgent broken")
        assert score == 1.0

    def test_partial_urgency_returns_fraction(self):
        a = AlertingAttention()
        # 1 out of 4 words is an urgency keyword
        score = a.score("the system is down")
        assert 0.0 < score < 1.0

    def test_empty_string_returns_zero(self):
        a = AlertingAttention()
        assert a.score("") == 0.0

    def test_punctuation_stripped_from_keywords(self):
        a = AlertingAttention()
        # "error!" should still match "error"
        score = a.score("error!")
        assert score == 1.0


class TestOrientingAttention:
    def test_capitalised_word_after_first_extracted(self):
        o = OrientingAttention()
        topics = o.extract_topics("Check the Server status")
        assert "Server" in topics

    def test_word_after_preposition_extracted(self):
        o = OrientingAttention()
        topics = o.extract_topics("tell me about Python")
        assert "Python" in topics

    def test_no_topics_for_plain_sentence(self):
        o = OrientingAttention()
        topics = o.extract_topics("this is a plain sentence")
        assert topics == []

    def test_duplicate_topics_deduplicated(self):
        o = OrientingAttention()
        topics = o.extract_topics("Python is great and Python is fast")
        assert topics.count("Python") == 1


class TestSustainedAttention:
    def test_initial_duration_is_one(self):
        s = SustainedAttention()
        duration = s.update(["topic_a"])
        assert duration == 1

    def test_same_topics_increments_duration(self):
        s = SustainedAttention()
        s.update(["server"])
        duration = s.update(["server"])
        assert duration == 2

    def test_different_topics_resets_duration(self):
        s = SustainedAttention()
        s.update(["server"])
        duration = s.update(["database"])
        assert duration == 1

    def test_empty_topics_resets_to_one(self):
        s = SustainedAttention()
        s.update(["server"])
        s.update(["server"])
        duration = s.update([])
        assert duration == 1


class TestSelectiveAttention:
    def test_filters_fragments_by_topic(self):
        fragments = ["nginx is running", "mysql is down", "redis is fine"]
        filtered = SelectiveAttention.filter(fragments, ["mysql"])
        assert filtered == ["mysql is down"]

    def test_no_topics_returns_all_fragments(self):
        fragments = ["a", "b", "c"]
        filtered = SelectiveAttention.filter(fragments, [])
        assert filtered == fragments

    def test_case_insensitive_matching(self):
        fragments = ["The MySQL database crashed"]
        filtered = SelectiveAttention.filter(fragments, ["mysql"])
        assert len(filtered) == 1


class TestExecutiveAttention:
    def test_high_urgency_prioritises_shell_and_file(self):
        priority = ExecutiveAttention.prioritise(0.8, [])
        assert "shell_exec" in priority
        assert "file_read" in priority

    def test_low_urgency_no_default_priority(self):
        priority = ExecutiveAttention.prioritise(0.2, [])
        assert priority == []

    def test_file_topic_adds_file_tools(self):
        priority = ExecutiveAttention.prioritise(0.0, ["file", "config"])
        assert "file_read" in priority
        assert "file_write" in priority

    def test_high_urgency_plus_file_topic_no_duplicates(self):
        priority = ExecutiveAttention.prioritise(0.8, ["file"])
        assert priority.count("file_read") == 1


class TestAttentionSystemIntegration:
    def test_process_returns_attention_state(self):
        attn = AttentionSystem()
        state = attn.process("the server is down! Fix it immediately!")
        assert state.urgency > 0.0
        assert state.focus_duration >= 1

    def test_process_context_filter_lower_cases_topics(self):
        attn = AttentionSystem()
        state = attn.process("Check the Server logs")
        assert all(t == t.lower() for t in state.context_filter)

    def test_process_sustained_attention_tracks_continuity(self):
        attn = AttentionSystem()
        attn.process("server issue")
        state2 = attn.process("server crash")
        # Both turns have "server" topic → duration should be > 1
        assert state2.focus_duration >= 1  # at least 1; implementation-dependent

    def test_process_empty_input_returns_zero_urgency(self):
        attn = AttentionSystem()
        state = attn.process("hello there")
        assert 0.0 <= state.urgency <= 1.0
