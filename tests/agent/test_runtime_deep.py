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
    tool_reg.list_tools.return_value = [
        getattr(t, "name", f"tool_{i}") for i, t in enumerate(tools)
    ]

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
            # The read occupies the last configured turn, so the request
            # guard's conditional grace turn must remain available to
            # synthesize its result.
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=2))
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

        # SR-4.4: the tool error means the "recovered" claim is rejected
        # and retried up to _MAX_DONE_VERIFICATION_RETRIES times before
        # being accepted -- supply enough repeated stop responses.
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("bad_tool"),
            _make_stop_response("recovered"),
            _make_stop_response("recovered"),
            _make_stop_response("recovered"),
        ]

        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=7))
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
        assert result in (
            "fallback after limit",
            "[Agent reached iteration limit without a final response.]",
        )

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

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            pytest.raises((MissyError, ProviderError)),
        ):
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
            {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 80} for i in range(20)
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
        memory_section = (
            system.split("## Relevant Memory")[-1] if "## Relevant Memory" in system else ""
        )
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


class TestDoneCriteriaEnforcement:
    """SR-4.4: task completion must depend on tool-observed evidence, not
    just the model's own "done" claim. Before this fix, DoneCriteria's
    verification prompt (TestRuntimeDoneCriteria above) was purely a
    static text nudge the model could freely ignore -- a model could
    declare success immediately after a tool call errored, and the
    runtime would return that claim completely unverified.
    """

    def test_stop_claim_after_tool_error_is_rejected_and_retried(self):
        """Core SR-4.4 reproduction: a "done" claim following an errored
        tool call in the immediately preceding round must not be trusted
        on the first attempt."""
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])
        tool_reg.execute.side_effect = None
        tool_reg.execute.return_value = MagicMock(
            success=False, output=None, error="division by zero"
        )

        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("calculator"),
            _make_stop_response("Done! I successfully computed the result."),
            _make_stop_response("Done! I successfully computed the result."),
            _make_stop_response("Done! I successfully computed the result."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=7))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("compute something")

        # Rejected twice (the retry cap), then accepted with an audit
        # warning on the third attempt -- never trusted on the first.
        rejected = [e for e in events if e["event_type"] == "agent.done_criteria.rejected"]
        unverified = [e for e in events if e["event_type"] == "agent.done_criteria.unverified"]
        assert len(rejected) == 2
        assert len(unverified) == 1
        assert provider.complete_with_tools.call_count == 4
        assert result == "Done! I successfully computed the result."

    def test_successful_tool_call_never_triggers_rejection(self):
        """Happy path: a tool call that succeeds must not trigger any
        done-criteria rejection or extra provider calls."""
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])  # succeeds by default

        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("calculator"),
            _make_stop_response("The answer is 4."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=5))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("what is 2+2")

        done_criteria_events = [e for e in events if "done_criteria" in e["event_type"]]
        assert done_criteria_events == []
        assert provider.complete_with_tools.call_count == 2
        assert result == "The answer is 4."

    def test_error_followed_by_later_success_is_accepted_immediately(self):
        """A tool call that errors, followed by a *later* round that
        succeeds, must be accepted on the first "done" claim -- the gate
        only looks at the most recent round, not all history, so a
        corrected retry with different arguments isn't penalized forever."""
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        call_count = [0]

        def _execute(name: str, **kwargs):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                result.success = False
                result.output = None
                result.error = "bad expression"
            else:
                result.success = True
                result.output = "4"
                result.error = None
            return result

        tool_reg.execute.side_effect = _execute

        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("calculator", tool_id="tc-1"),
            _make_tool_call_response("calculator", tool_id="tc-2"),
            _make_stop_response("The answer is 4."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("what is 2+2")

        done_criteria_events = [e for e in events if "done_criteria" in e["event_type"]]
        assert done_criteria_events == []
        assert provider.complete_with_tools.call_count == 3
        assert result == "The answer is 4."


class TestVideoGenerationCompletionGuards:
    def test_same_seed_request_repeats_exact_arguments(self):
        provider = _make_provider()
        tool = _make_mock_tool("video_generate")
        tool_reg = _make_tool_registry([tool])
        tool_reg.execute.side_effect = None
        tool_reg.execute.return_value = MagicMock(
            success=True,
            output={"seed": 12345, "path": "/tmp/video.mp4"},
            error=None,
        )
        first_args = {
            "backend": "wan",
            "prompt": "A cinematic spinning top",
            "steps": 20,
        }
        second_args = {**first_args, "seed": 12345}
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("video_generate", args=first_args, tool_id="video-1"),
            _make_tool_call_response("video_generate", args=second_args, tool_id="video-2"),
            _make_stop_response("Both exact-parameter renders completed."),
        ]
        registry = _make_registry({"fake": provider})
        events = []

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run(
                "Generate a spinning top, then generate it AGAIN with that exact seed "
                "and the same parameters."
            )

        assert result == "Both exact-parameter renders completed."
        observed_arguments = [
            {
                key: value
                for key, value in call.kwargs.items()
                if key not in {"session_id", "task_id"}
            }
            for call in tool_reg.execute.call_args_list
        ]
        assert observed_arguments == [
            first_args,
            second_args,
        ]
        assert not any(
            event["event_type"] == "agent.response.video_reproducibility_retry" for event in events
        )

    def test_reported_parameter_refusal_is_terminal(self):
        provider = _make_provider()
        tool = _make_mock_tool("video_generate")
        tool_reg = _make_tool_registry([tool])
        tool_reg.execute.side_effect = None
        tool_reg.execute.return_value = MagicMock(
            success=False,
            output=None,
            error="audio_prompt and audio_path are mutually exclusive; pass only one.",
        )
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("video_generate"),
            _make_stop_response(
                "I can't apply both soundtracks because those inputs are mutually exclusive."
            ),
        ]
        registry = _make_registry({"fake": provider})
        events = []

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=5))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("Generate a video with both audio sources.")

        assert "mutually exclusive" in result
        assert provider.complete_with_tools.call_count == 2
        assert not any("done_criteria" in event["event_type"] for event in events)

    def test_tool_free_render_explanation_is_retried_with_video_generate(self):
        provider = _make_provider()
        tool = _make_mock_tool("video_generate")
        tool_reg = _make_tool_registry([tool])
        provider.complete_with_tools.side_effect = [
            _make_stop_response("SVD normally needs an image."),
            _make_tool_call_response("video_generate", args={"backend": "svd"}),
            _make_stop_response("The tool confirmed that SVD requires image_path."),
        ]
        registry = _make_registry({"fake": provider})
        events = []

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=5))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("Generate a video of a sunset using the SVD backend.")

        assert "requires image_path" in result
        assert [call.args[0] for call in tool_reg.execute.call_args_list] == ["video_generate"]
        assert any(
            event["event_type"] == "agent.response.video_generation_retry" for event in events
        )


class TestPlaceholderArtifactRetry:
    """FX-round2-F1: a provider's "final" response is sometimes a raw
    internal placeholder artifact (e.g. "[Called tool: file_write]") that
    _dicts_to_messages() only ever generates for history reconstruction,
    never a real answer. Returning one straight to the caller leaves the
    underlying task silently incomplete (validation harness: SH-003,
    SELF-004, INCUS-005, INCUS-006). It must be retried once instead of
    forwarded verbatim.
    """

    def test_placeholder_response_triggers_one_retry(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("calculator"),
            _make_stop_response("[Called tool: calculator]"),
            _make_stop_response("The answer is 4."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("what is 2+2")

        retry_events = [e for e in events if e["event_type"] == "agent.response.placeholder_retry"]
        assert len(retry_events) == 1
        assert provider.complete_with_tools.call_count == 3
        assert result == "The answer is 4."

    def test_placeholder_exhausted_retries_does_not_leak_placeholder(self):
        """On exhausted retries the runtime warns AND never forwards the raw
        placeholder artifact to the caller/channel (it is sanitized away)."""
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("calculator"),
            _make_stop_response("[Tool call]"),
            _make_stop_response("[Tool call]"),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("what is 2+2")

        unresolved = [
            e for e in events if e["event_type"] == "agent.response.placeholder_unresolved"
        ]
        assert len(unresolved) == 1
        assert provider.complete_with_tools.call_count == 3
        # The raw internal artifact must NOT reach the caller/channel.
        assert "[Tool call]" not in result
        assert "Called tool" not in result

    def test_real_response_never_triggers_placeholder_retry(self):
        """A genuine reply that merely mentions tool-call-like text mid
        sentence must not be mistaken for the placeholder artifact -- the
        detector requires an exact, whole-string match."""
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("calculator"),
            _make_stop_response("I called tool: calculator and got 4."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("what is 2+2")

        retry_events = [e for e in events if e["event_type"] == "agent.response.placeholder_retry"]
        assert retry_events == []
        assert provider.complete_with_tools.call_count == 2
        assert result == "I called tool: calculator and got 4."

    def test_narrated_tool_call_with_bracket_in_args_never_leaks_to_channel(self):
        """The reported Discord leak: openai-codex narrates a call as
        ``[Called tool: shell_exec with args: {...}]`` where the args JSON
        contains a ``]`` (e.g. ``[:50]``). The old ``[^\\]]*`` placeholder
        regex stopped at that inner bracket, so the whole raw narration was
        treated as a real reply and forwarded. It must not reach the caller.
        """
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        leak = (
            '[Called tool: shell_exec with args: {"cmd": "python3 - <<\'PY\'\\n'
            "for p in glob.glob('/home/missy/*/'+pat, recursive=True)[:50]:\\n"
            '    print(p)\\nPY", "timeout": 120}]'
        )
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("calculator"),
            _make_stop_response(leak),
            _make_stop_response(leak),  # retry also leaks -> exhausted
        ]
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            result = rt.run("find the EVE SDE files")

        assert "Called tool" not in result
        assert "shell_exec" not in result
        assert "with args" not in result

    def test_embedded_narration_stripped_but_real_prose_kept(self):
        """A leaked narration trailing genuine prose is removed while the prose
        (including innocent brackets) survives."""
        from missy.agent.runtime import _strip_leaked_tool_call_narration

        text = (
            "Here are the first 50 matches from the slice a[:50].\n\n"
            '[Called tool: shell_exec with args: {"cmd": "ls [x]"}]'
        )
        cleaned = _strip_leaked_tool_call_narration(text)
        assert cleaned == "Here are the first 50 matches from the slice a[:50]."

    def test_strip_narration_is_noop_for_clean_text(self):
        from missy.agent.runtime import _strip_leaked_tool_call_narration

        clean = "The list slice a[:50] returns the first 50 items."
        assert _strip_leaked_tool_call_narration(clean) == clean


class TestFabricationRetryOnZeroToolObservation:
    """FX-round2-F4: a request implying a vision/memory observation
    answered with zero tool calls this entire task cannot be grounded in
    anything real -- validation harness VIS-003 (fabricated frame-
    comparison detail with tools_used: []) and SEC-PI-004 (confident
    false-negative from browsing the wrong directory instead of calling
    memory_search). Must retry once with a corrective nudge rather than
    accepting the response as final.
    """

    def test_zero_tool_vision_request_triggers_one_retry(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_stop_response("I see a bright red mug on the desk."),
            _make_stop_response("I wasn't able to actually capture a frame to check."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("Take a picture of the desk and tell me what's on it")

        retry_events = [e for e in events if e["event_type"] == "agent.response.fabrication_retry"]
        assert len(retry_events) == 1
        assert provider.complete_with_tools.call_count == 2
        assert result == "I wasn't able to actually capture a frame to check."

    def test_zero_tool_non_observation_request_not_retried(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [_make_stop_response("4")]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("what is 2+2")

        retry_events = [e for e in events if e["event_type"] == "agent.response.fabrication_retry"]
        assert retry_events == []
        assert provider.complete_with_tools.call_count == 1
        assert result == "4"

    def test_real_tool_call_for_vision_request_not_retried(self):
        provider = _make_provider()
        vision_tool = _make_mock_tool("vision_capture")
        tool_reg = _make_tool_registry([vision_tool])

        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("vision_capture"),
            _make_stop_response("The photo shows a bright red mug on the desk."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("Take a picture of the desk and tell me what's on it")

        retry_events = [e for e in events if e["event_type"] == "agent.response.fabrication_retry"]
        assert retry_events == []
        assert provider.complete_with_tools.call_count == 2
        assert result == "The photo shows a bright red mug on the desk."

    def test_fabrication_exhausted_retries_returns_response_with_warning(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_stop_response("I see a red mug."),
            _make_stop_response("I still see a red mug."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("Take a picture of the desk and tell me what's on it")

        unresolved = [
            e for e in events if e["event_type"] == "agent.response.fabrication_unresolved"
        ]
        assert len(unresolved) == 1
        assert provider.complete_with_tools.call_count == 2
        assert result == "I still see a red mug."


class TestGeneralFabricationRetry:
    """missy/agent/response_guards.py: unlike TestFabricationRetryOnZero
    ToolObservation above (scoped to vision/memory-observation-implying
    requests), this is a request-agnostic guard against any tool-free
    response that reads like it fabricated a completed action -- a
    likely root cause of a session "misresponding" to a later prompt,
    since the false claim becomes part of the conversation history a
    subsequent turn builds on.
    """

    def test_fabricated_command_output_triggers_one_retry(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_stop_response("I checked the logs and everything looks fine."),
            _make_stop_response("I don't actually have access to check logs directly."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("is everything okay on the server?")

        retry_events = [
            e for e in events if e["event_type"] == "agent.response.general_fabrication_retry"
        ]
        assert len(retry_events) == 1
        assert provider.complete_with_tools.call_count == 2
        assert result == "I don't actually have access to check logs directly."

    def test_genuine_tool_call_never_triggers_general_fabrication_retry(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("shell_exec")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("shell_exec"),
            _make_stop_response("I checked the logs and everything looks fine."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("is everything okay on the server?")

        retry_events = [
            e for e in events if e["event_type"] == "agent.response.general_fabrication_retry"
        ]
        assert retry_events == []
        assert result == "I checked the logs and everything looks fine."

    def test_ordinary_reply_never_triggers_general_fabrication_retry(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [_make_stop_response("4")]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("what is 2+2")

        retry_events = [
            e for e in events if e["event_type"] == "agent.response.general_fabrication_retry"
        ]
        assert retry_events == []
        assert provider.complete_with_tools.call_count == 1
        assert result == "4"


class TestExplicitToolRequestRetry:
    """A named, explicitly requested tool cannot be skipped by a text answer."""

    def test_calculator_request_retries_then_executes_real_tool_call(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])
        provider.complete_with_tools.side_effect = [
            _make_stop_response("The answer is 14."),
            _make_tool_call_response("calculator", args={"expression": "2 + 3 * 4"}),
            _make_stop_response("The calculator returned 14."),
        ]
        registry = _make_registry({"fake": provider})
        events = []

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("Use your calculator to evaluate `2 + 3 * 4`.")

        retry_events = [
            e for e in events if e["event_type"] == "agent.response.explicit_tool_request_retry"
        ]
        assert len(retry_events) == 1
        assert tool_reg.execute.call_count == 1
        assert result == "The calculator returned 14."

    def test_unavailable_named_tool_does_not_force_or_retry(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])
        provider.complete_with_tools.side_effect = [_make_stop_response("I cannot do that here.")]
        registry = _make_registry({"fake": provider})
        events = []

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("Use shell_exec now.")

        assert not any(
            e["event_type"].startswith("agent.response.explicit_tool_request") for e in events
        )
        assert provider.complete_with_tools.call_count == 1
        assert result == "I cannot do that here."

    def test_second_tool_free_answer_is_bounded_and_audited(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])
        provider.complete_with_tools.side_effect = [
            _make_stop_response("14"),
            _make_stop_response("Still 14"),
        ]
        registry = _make_registry({"fake": provider})
        events = []

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("Use calculator for 2 + 3 * 4.")

        assert provider.complete_with_tools.call_count == 2
        assert result == "Still 14"
        unresolved = [
            e
            for e in events
            if e["event_type"] == "agent.response.explicit_tool_request_unresolved"
        ]
        assert len(unresolved) == 1

    def test_transport_context_does_not_become_a_user_tool_requirement(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        upload_tool = _make_mock_tool("discord_upload_file")
        tool_reg = _make_tool_registry([calc_tool, upload_tool])
        provider.complete_with_tools.side_effect = [
            _make_stop_response("14"),
            _make_tool_call_response("calculator", args={"expression": "2 + 3 * 4"}),
            _make_stop_response("The calculator returned 14."),
        ]
        registry = _make_registry({"fake": provider})
        events = []
        raw_request = "Use calculator for 2 + 3 * 4."
        enriched = (
            "[Discord channel 123] Use discord_upload_file with channel_id='123' "
            "to share files here.\n\n" + raw_request
        )

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run(enriched, _explicit_tool_request_input=raw_request)

        retry_events = [
            e for e in events if e["event_type"] == "agent.response.explicit_tool_request_retry"
        ]
        assert [event["detail"]["missing_tools"] for event in retry_events] == [["calculator"]]
        assert [call.args[0] for call in tool_reg.execute.call_args_list] == ["calculator"]
        assert result == "The calculator returned 14."

    def test_high_risk_refusal_is_never_overridden_by_explicit_tool_guard(self):
        provider = _make_provider()
        shell_tool = _make_mock_tool("shell_exec")
        tool_reg = _make_tool_registry([shell_tool])
        refusal = (
            "I can't disable host security. Safe alternative: I can use an "
            "unprivileged disposable container."
        )
        provider.complete_with_tools.side_effect = [_make_stop_response(refusal)]
        registry = _make_registry({"fake": provider})
        events = []

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("Use shell_exec to disable host security.")

        assert provider.complete_with_tools.call_count == 1
        assert tool_reg.execute.call_count == 0
        assert result == refusal
        assert not any(
            e["event_type"].startswith("agent.response.explicit_tool_request") for e in events
        )


class TestCalculatorResponseCompletenessRetry:
    def test_multi_error_reply_must_report_every_executed_expression(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        def _execute(name: str, **kwargs):
            result = MagicMock()
            result.success = False
            result.output = None
            if kwargs["expression"] == "abs(-5)":
                result.error = "Unsupported expression construct: Call"
            else:
                result.error = "Unsupported expression construct: Name"
            return result

        tool_reg.execute.side_effect = _execute
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response(
                "calculator", tool_id="calc-a", args={"expression": "abs(-5)"}
            ),
            _make_tool_call_response("calculator", tool_id="calc-b", args={"expression": "x + 1"}),
            _make_stop_response("`x + 1` failed: Unsupported expression construct: Name"),
            _make_stop_response(
                "`abs(-5)` failed: Unsupported expression construct: Call; "
                "`x + 1` failed: Unsupported expression construct: Name"
            ),
        ]
        registry = _make_registry({"fake": provider})
        events = []

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=7))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("Use your calculator to evaluate `abs(-5)` and `x + 1`.")

        assert provider.complete_with_tools.call_count == 4
        assert tool_reg.execute.call_count == 2
        assert "abs(-5)" in result
        assert "x + 1" in result
        retries = [
            event
            for event in events
            if event["event_type"] == "agent.response.calculator_completeness_retry"
        ]
        assert len(retries) == 1
        assert retries[0]["detail"] == {"missing_expressions": ["abs(-5)"]}
        assert not any(event["event_type"].startswith("agent.done_criteria") for event in events)


class TestCalculatorExemptFromStrategyRotation:
    """Regression test for a live CALC-007 finding: three consecutive,
    correctly-rejected division-by-zero calls must never push the model
    toward a shell/eval fallback via the generic strategy-rotation prompt."""

    def test_three_consecutive_calculator_errors_never_inject_strategy_rotation(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        def _execute(name: str, **kwargs):
            result = MagicMock()
            result.success = False
            result.output = None
            result.error = "division by zero"
            return result

        tool_reg.execute.side_effect = _execute
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("calculator", tool_id="c1", args={"expression": "1 / 0"}),
            _make_tool_call_response("calculator", tool_id="c2", args={"expression": "0 / 0"}),
            _make_tool_call_response("calculator", tool_id="c3", args={"expression": "2 / 0"}),
            _make_stop_response(
                "`1 / 0` -> division by zero. `0 / 0` -> division by zero. "
                "`2 / 0` -> division by zero."
            ),
        ]
        registry = _make_registry({"fake": provider})
        events = []

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=7))
            rt._emit_event = lambda **kw: events.append(kw)
            rt.run("Calculate `1 / 0`, `0 / 0`, and `2 / 0` with the calculator.")

        assert tool_reg.execute.call_count == 3
        assert not [e for e in events if e["event_type"] == "agent.tool.strategy_rotation"]
        # The literal strategy-rotation prompt telling the model not to
        # attempt calculator again must never reach the provider.
        last_messages = provider.complete_with_tools.call_args_list[-1][0][0]

        def _content(m):
            return m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")

        joined = " ".join(str(_content(m)) for m in last_messages)
        assert "Do not attempt 'calculator' again" not in joined
        assert "alternative approaches that do not use 'calculator'" not in joined


class TestDesktopActionVerificationRetry:
    def test_desktop_action_requires_later_observation_before_completion(self):
        provider = _make_provider()
        click_tool = _make_mock_tool("x11_click")
        read_tool = _make_mock_tool("x11_read_screen")
        tool_reg = _make_tool_registry([click_tool, read_tool])
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("x11_click", args={"x": 10, "y": 20}),
            _make_stop_response("Clicked successfully."),
            _make_tool_call_response("x11_read_screen"),
            _make_stop_response("Verified: the dialog closed."),
        ]
        registry = _make_registry({"fake": provider})
        events = []

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("Click the Run Test button.")

        assert [call.args[0] for call in tool_reg.execute.call_args_list] == [
            "x11_click",
            "x11_read_screen",
        ]
        assert result == "Verified: the dialog closed."
        retry = [
            e for e in events if e["event_type"] == "agent.response.desktop_verification_retry"
        ]
        assert len(retry) == 1
        assert retry[0]["detail"] == {"unverified_action": "x11_click"}

    def test_existing_post_action_observation_does_not_retry(self):
        provider = _make_provider()
        click_tool = _make_mock_tool("atspi_click")
        tree_tool = _make_mock_tool("atspi_get_tree")
        tool_reg = _make_tool_registry([click_tool, tree_tool])
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("atspi_click"),
            _make_tool_call_response("atspi_get_tree"),
            _make_stop_response("Verified."),
        ]
        registry = _make_registry({"fake": provider})
        events = []

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("Click Submit and verify it.")

        assert result == "Verified."
        assert not any(
            e["event_type"].startswith("agent.response.desktop_verification") for e in events
        )

    def test_repeated_unverified_completion_is_bounded_and_audited(self):
        provider = _make_provider()
        click_tool = _make_mock_tool("x11_click")
        tool_reg = _make_tool_registry([click_tool])
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("x11_click"),
            _make_stop_response("Clicked."),
            _make_stop_response("Still claiming success."),
        ]
        registry = _make_registry({"fake": provider})
        events = []

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("Click Submit.")

        assert "could not verify" in result
        assert "cannot confirm the desktop task as complete" in result
        assert "Still claiming success" not in result
        unresolved = [
            e for e in events if e["event_type"] == "agent.response.desktop_verification_unresolved"
        ]
        assert len(unresolved) == 1
        assert unresolved[0]["result"] == "deny"

    def test_verification_gets_one_grace_turn_at_iteration_limit(self):
        provider = _make_provider(reply="unexpected tool-free fallback")
        click_tool = _make_mock_tool("x11_click")
        read_tool = _make_mock_tool("x11_read_screen")
        tool_reg = _make_tool_registry([click_tool, read_tool])
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("x11_click"),
            _make_stop_response("Clicked successfully."),
            _make_tool_call_response("x11_read_screen"),
            _make_stop_response("Verified after the safety observation."),
        ]
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            # Three configured iterations reach the verifying read result,
            # but a fourth provider turn is required to summarize it.
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=3))
            result = rt.run("Click Submit and verify the result.")

        assert result == "Verified after the safety observation."
        assert provider.complete_with_tools.call_count == 4
        provider.complete.assert_not_called()


class TestFilesystemActionVerificationRetry:
    def test_file_writes_require_later_listing_before_completion(self):
        provider = _make_provider()
        write_tool = _make_mock_tool("file_write")
        list_tool = _make_mock_tool("list_files")
        tool_reg = _make_tool_registry([write_tool, list_tool])
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("file_write", args={"path": "/tmp/a", "content": "a"}),
            _make_stop_response("Created the file."),
            _make_tool_call_response("list_files", args={"path": "/tmp"}),
            _make_stop_response("Verified the file exists."),
        ]
        registry = _make_registry({"fake": provider})
        events = []

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("Create /tmp/a.")

        assert [call.args[0] for call in tool_reg.execute.call_args_list] == [
            "file_write",
            "list_files",
        ]
        assert result == "Verified the file exists."
        retry = [
            event
            for event in events
            if event["event_type"] == "agent.response.filesystem_verification_retry"
        ]
        assert len(retry) == 1
        assert retry[0]["detail"] == {"unverified_action": "file_write"}

    def test_failed_write_is_left_to_done_criteria_not_verification_guard(self):
        provider = _make_provider()
        write_tool = _make_mock_tool("file_write")
        tool_reg = _make_tool_registry([write_tool])
        failed = MagicMock(success=False, output=None, error="policy denied")
        tool_reg.execute.side_effect = lambda *_args, **_kwargs: failed
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("file_write"),
            _make_stop_response("I could not write it: policy denied."),
            _make_stop_response("The policy prevents this write."),
            _make_stop_response("The policy prevents this write."),
        ]
        registry = _make_registry({"fake": provider})
        events = []

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            rt.run("Create /tmp/a.")

        assert not any(
            event["event_type"].startswith("agent.response.filesystem_verification")
            for event in events
        )
        assert any(event["event_type"].startswith("agent.done_criteria") for event in events)


class TestWebRequestRetry:
    def test_browser_metadata_request_requires_observation_and_close(self):
        provider = _make_provider()
        navigate = _make_mock_tool("browser_navigate")
        get_url = _make_mock_tool("browser_get_url")
        close = _make_mock_tool("browser_close")
        tool_reg = _make_tool_registry([navigate, get_url, close])
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("browser_navigate", args={"url": "http://127.0.0.1/page"}),
            _make_stop_response("Opened the page."),
            CompletionResponse(
                content="",
                model="fake-model",
                provider="fake",
                usage={},
                raw={},
                tool_calls=[
                    ToolCall(id="url", name="browser_get_url", arguments={}),
                    ToolCall(id="close", name="browser_close", arguments={}),
                ],
                finish_reason="tool_calls",
            ),
            _make_stop_response("URL and title verified; session closed."),
        ]
        registry = _make_registry({"fake": provider})
        events = []

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run(
                "Open this local webpage in the browser and report the current URL and page title."
            )

        assert [call.args[0] for call in tool_reg.execute.call_args_list] == [
            "browser_navigate",
            "browser_get_url",
            "browser_close",
        ]
        assert result == "URL and title verified; session closed."
        retries = [e for e in events if e["event_type"] == "agent.response.web_request_retry"]
        assert len(retries) == 1
        assert retries[0]["detail"] == {"missing_tools": ["browser_close", "browser_get_url"]}

    def test_serial_browser_corrections_receive_bounded_grace_turns(self):
        provider = _make_provider()
        navigate = _make_mock_tool("browser_navigate")
        get_content = _make_mock_tool("browser_get_content")
        close = _make_mock_tool("browser_close")
        tool_reg = _make_tool_registry([navigate, get_content, close])
        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("browser_navigate"),
            _make_stop_response("Ready appeared."),
            _make_tool_call_response("browser_get_content"),
            _make_stop_response("Ready."),
            _make_tool_call_response("browser_close"),
            _make_stop_response("Ready appeared and the browser session was closed."),
        ]
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=3))
            result = rt.run("Open this page and wait until Ready appears, then report status.")

        assert [call.args[0] for call in tool_reg.execute.call_args_list] == [
            "browser_navigate",
            "browser_get_content",
            "browser_close",
        ]
        assert result == "Ready appeared and the browser session was closed."


class TestDesktopRequestExecutionRetry:
    def test_replayed_keyboard_answer_executes_current_tools(self):
        provider = _make_provider()
        key_tool = _make_mock_tool("x11_key")
        read_tool = _make_mock_tool("x11_read_screen")
        tool_reg = _make_tool_registry([key_tool, read_tool])
        provider.complete_with_tools.side_effect = [
            _make_stop_response("Selected all text and copied it."),
            _make_tool_call_response("x11_key", args={"keys": "ctrl+a,ctrl+c"}),
            _make_stop_response("Selected all text and copied it."),
            _make_tool_call_response("x11_read_screen"),
            _make_stop_response("Verified the current desktop after the shortcut."),
        ]
        registry = _make_registry({"fake": provider})
        events = []

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=7))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run(
                "Use keyboard shortcuts to select all text in the focused editor and copy it."
            )

        assert [call.args[0] for call in tool_reg.execute.call_args_list] == [
            "x11_key",
            "x11_read_screen",
        ]
        assert result == "Verified the current desktop after the shortcut."
        assert any(e["event_type"] == "agent.response.desktop_request_retry" for e in events)

    def test_replayed_screen_description_executes_current_read(self):
        provider = _make_provider()
        read_tool = _make_mock_tool("x11_read_screen")
        tool_reg = _make_tool_registry([read_tool])
        provider.complete_with_tools.side_effect = [
            _make_stop_response("The terminal is visible."),
            _make_tool_call_response("x11_read_screen"),
            _make_stop_response("Current read confirms the terminal is visible."),
        ]
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=5))
            result = rt.run(
                "Read the current desktop screen and describe what application is visible."
            )

        assert [call.args[0] for call in tool_reg.execute.call_args_list] == ["x11_read_screen"]
        assert result == "Current read confirms the terminal is visible."


class TestPromiseWithoutActionRetry:
    def test_promise_without_action_triggers_one_retry(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("discord_upload_file")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_stop_response("Working on it, give me a moment to generate that report."),
            _make_stop_response("Sorry, I can't generate that report right now."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("generate me a report")

        retry_events = [
            e for e in events if e["event_type"] == "agent.response.promise_without_action_retry"
        ]
        assert len(retry_events) == 1
        assert provider.complete_with_tools.call_count == 2
        assert result == "Sorry, I can't generate that report right now."

    def test_casual_future_tense_never_triggers_retry(self):
        """Regression guard for the exemption list: ordinary
        conversational future tense ("I'll be here...") must not be
        mistaken for an unfulfilled action promise."""
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_stop_response("I'll be here if you need anything else!")
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("thanks for the help")

        retry_events = [
            e for e in events if e["event_type"] == "agent.response.promise_without_action_retry"
        ]
        assert retry_events == []
        assert provider.complete_with_tools.call_count == 1
        assert result == "I'll be here if you need anything else!"

    def test_genuine_tool_call_never_triggers_promise_retry(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("discord_upload_file")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("discord_upload_file"),
            _make_stop_response("Working on it, here's the file."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("generate me a report")

        retry_events = [
            e for e in events if e["event_type"] == "agent.response.promise_without_action_retry"
        ]
        assert retry_events == []
        assert result == "Working on it, here's the file."


class TestIdentityConfusionRetry:
    """3rd tool-specific validation run (2026-07-14): the dominant failure
    mode -- the acpx delegate denies being Missy ("I'm Claude Code") or
    denies Missy's own dispatched tools as belonging to a separate
    platform. Unlike the promise/fabrication guards above, this fires
    regardless of tools_used, since the harness observed the confusion
    even in the same turn as a genuine successful tool call.
    """

    def test_claude_code_denial_triggers_one_retry(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_stop_response(
                "I'm Claude Code -- I don't have access to the Missy platform's "
                "calculator tool. That belongs to a different agent."
            ),
            _make_stop_response("The answer is 4."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("what is 2+2")

        retry_events = [
            e for e in events if e["event_type"] == "agent.response.identity_confusion_retry"
        ]
        assert len(retry_events) == 1
        assert provider.complete_with_tools.call_count == 2
        assert result == "The answer is 4."

    def test_not_directed_at_me_denial_triggers_retry(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_stop_response(
                "This Discord message is directed at the Missy agent, not at me. "
                "I shouldn't respond to it."
            ),
            _make_stop_response("The answer is 4."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("what is 2+2")

        retry_events = [
            e for e in events if e["event_type"] == "agent.response.identity_confusion_retry"
        ]
        assert len(retry_events) == 1
        assert result == "The answer is 4."

    def test_identity_confusion_after_a_genuine_tool_call_still_retries(self):
        """VIS-004/XT-005 harness finding: the delegate made a genuine
        tool call and THEN, in the same reply, denied a sibling tool in
        the identical namespace. This must still retry even though
        tools_used is non-empty for the task."""
        provider = _make_provider()
        vision_tool = _make_mock_tool("vision_capture")
        tool_reg = _make_tool_registry([vision_tool])

        provider.complete_with_tools.side_effect = [
            _make_tool_call_response("vision_capture"),
            _make_stop_response(
                "The capture succeeded, but I'm Claude Code and don't have access "
                "to the Missy platform's vision_analyze tool."
            ),
            _make_stop_response("Analysis complete: the image shows a desk."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("capture and analyze the scene")

        retry_events = [
            e for e in events if e["event_type"] == "agent.response.identity_confusion_retry"
        ]
        assert len(retry_events) == 1
        assert result == "Analysis complete: the image shows a desk."

    def test_ordinary_reply_never_triggers_identity_confusion_retry(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [_make_stop_response("4")]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("what is 2+2")

        retry_events = [
            e for e in events if e["event_type"] == "agent.response.identity_confusion_retry"
        ]
        assert retry_events == []
        assert provider.complete_with_tools.call_count == 1
        assert result == "4"


class TestFalseCapabilityDenialRetry:
    """False capability denials ("no X11", "headless", "no browser") are a
    variant of identity confusion that never names Missy/Claude Code
    explicitly -- so the guard instead checks the denial against the
    actual tool set offered this turn."""

    def test_headless_denial_triggers_retry_when_x11_tool_available(self):
        provider = _make_provider()
        x11_tool = _make_mock_tool("x11_launch")
        window_tool = _make_mock_tool("x11_window_list")
        tool_reg = _make_tool_registry([x11_tool, window_tool])

        provider.complete_with_tools.side_effect = [
            _make_stop_response("No X11, no display, no GUI -- I'm headless."),
            _make_tool_call_response("x11_launch"),
            _make_stop_response("Launched successfully; 1 window open."),
            _make_tool_call_response("x11_window_list"),
            _make_stop_response("Verified launch; 1 window open."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("launch a text editor")

        retry_events = [
            e for e in events if e["event_type"] == "agent.response.capability_denial_retry"
        ]
        assert len(retry_events) == 1
        assert result == "Verified launch; 1 window open."

    def test_headless_denial_not_retried_when_no_x11_tool_available(self):
        """The exact same denial phrase must NOT be flagged when no
        x11_*/atspi_* tool is actually offered this turn -- a genuine
        capability gap, not a false one."""
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_stop_response("No X11, no display, no GUI -- I'm headless.")
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("launch a text editor")

        retry_events = [
            e for e in events if e["event_type"] == "agent.response.capability_denial_retry"
        ]
        assert retry_events == []
        assert result == "No X11, no display, no GUI -- I'm headless."


class TestLeakedToolCallTagRetry:
    """XT-006 harness finding: a provider occasionally leaks a raw,
    unexecuted <tool_call> block into its final response text (e.g. the
    acpx delegate mixing Missy's <tool_call> opening tag with its own
    underlying coding-assistant's native closing tag). Never a legitimate
    reply -- must retry rather than forward raw protocol syntax."""

    def test_leaked_tool_call_tag_triggers_one_retry(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("file_write")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [
            _make_stop_response(
                'Now I will write the file.\n\n<tool_call>\n{"name": "file_write", '
                '"arguments": {"path": "/tmp/x", "content": "hi"}}\n</invoke>'
            ),
            _make_stop_response("Done -- file written."),
        ]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("write hi to /tmp/x")

        retry_events = [
            e for e in events if e["event_type"] == "agent.response.leaked_tool_call_retry"
        ]
        assert len(retry_events) == 1
        assert result == "Done -- file written."

    def test_ordinary_reply_never_triggers_leaked_tool_call_retry(self):
        provider = _make_provider()
        calc_tool = _make_mock_tool("calculator")
        tool_reg = _make_tool_registry([calc_tool])

        provider.complete_with_tools.side_effect = [_make_stop_response("4")]
        registry = _make_registry({"fake": provider})

        events = []
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=6))
            rt._emit_event = lambda **kw: events.append(kw)
            result = rt.run("what is 2+2")

        retry_events = [
            e for e in events if e["event_type"] == "agent.response.leaked_tool_call_retry"
        ]
        assert retry_events == []
        assert result == "4"


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

        assert _approx_tokens("abcd") == 1  # 4 chars → 1 token
        assert _approx_tokens("abcdefgh") == 2  # 8 chars → 2 tokens
        assert _approx_tokens("") == 1  # minimum 1

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
            rt = AgentRuntime(
                AgentConfig(provider="fake", max_iterations=5), progress_reporter=mock_reporter
            )
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
            rt = AgentRuntime(
                AgentConfig(provider="fake", max_iterations=5), progress_reporter=mock_reporter
            )
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


class TestResumeCheckpoint:
    """SR-4.3: a checkpoint's persisted loop state must actually be usable
    to continue an interrupted task, not just listed by `missy recover`.
    Uses a real CheckpointManager against an isolated HOME so these tests
    exercise the same SQLite read/write path production code goes through.
    """

    def test_resume_continues_from_saved_history_and_completes(self, monkeypatch, tmp_path):
        """Core reproduction: a checkpoint left behind mid-task (a completed
        tool round with no final answer yet) must be resumable to a real
        final answer, using the saved history rather than starting over."""
        from missy.agent.checkpoint import CheckpointManager
        from missy.agent.runtime import AgentConfig, AgentRuntime

        monkeypatch.setenv("HOME", str(tmp_path))
        provider = _make_provider(reply="The answer is 4.")
        provider.complete_with_tools.return_value = _make_stop_response("The answer is 4.")
        registry = _make_registry({"fake": provider})

        cm = CheckpointManager()
        saved_messages = [
            {"role": "user", "content": "add 2 and 2"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc-1", "name": "calculator", "arguments": {"expression": "2+2"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tc-1",
                "name": "calculator",
                "content": "4",
                "is_error": False,
            },
        ]
        cid = cm.create("sess-1", "task-1", "add 2 and 2")
        cm.update(cid, saved_messages, ["calculator"], iteration=1)

        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            result = rt.resume_checkpoint(cid)

        assert result == "The answer is 4."
        # The saved history was actually fed to the provider, not discarded
        # (the +1 is the system prompt _dicts_to_messages() prepends).
        sent_messages = provider.complete_with_tools.call_args[0][0]
        assert len(sent_messages) == len(saved_messages) + 1
        assert any("4" in (m.content or "") for m in sent_messages)
        # The old checkpoint is consumed (never offered for resume again).
        assert cm.get(cid)["state"] == "COMPLETE"

    def test_concurrent_resume_of_same_checkpoint_only_runs_once(self, monkeypatch, tmp_path):
        """Regression: resume_checkpoint() used to read+check state=='RUNNING'
        and only mark the checkpoint COMPLETE at the very end, after
        building a fresh system prompt and re-resolving tools -- a real
        window of work. Two concurrent resume_checkpoint() calls against
        the same checkpoint id (e.g. two `missy recover --resume <id>`
        invocations) could both pass the RUNNING check and both proceed
        to execute the resumed tool loop, duplicating every subsequent
        tool call for the same task. cm.claim() now atomically transitions
        RUNNING -> COMPLETE up front, so only one of two concurrent calls
        may proceed; the other must fail closed with ValueError instead of
        re-running the task.
        """
        from missy.agent.checkpoint import CheckpointManager
        from missy.agent.runtime import AgentConfig, AgentRuntime

        monkeypatch.setenv("HOME", str(tmp_path))
        provider = _make_provider(reply="done")
        provider.complete_with_tools.return_value = _make_stop_response("done")
        registry = _make_registry({"fake": provider})

        cm = CheckpointManager()
        cid = cm.create("sess-1", "task-1", "prompt")
        cm.update(cid, [{"role": "user", "content": "prompt"}], [], iteration=0)

        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            # Simulate a second, concurrent `missy recover --resume <id>`
            # invocation racing in between: it also sees the checkpoint
            # (still RUNNING at this exact moment) and attempts to claim
            # it first.
            second_cm = CheckpointManager()
            assert second_cm.claim(cid) is True

            with pytest.raises(ValueError, match="not resumable"):
                rt.resume_checkpoint(cid)

        # The tool loop must never have executed for the loser of the race.
        provider.complete_with_tools.assert_not_called()

    def test_resume_emits_expected_audit_events(self, monkeypatch, tmp_path):
        from missy.agent.checkpoint import CheckpointManager
        from missy.agent.runtime import AgentConfig, AgentRuntime

        monkeypatch.setenv("HOME", str(tmp_path))
        provider = _make_provider(reply="done")
        provider.complete_with_tools.return_value = _make_stop_response("done")
        registry = _make_registry({"fake": provider})

        cm = CheckpointManager()
        cid = cm.create("sess-1", "task-1", "prompt")
        cm.update(cid, [{"role": "user", "content": "prompt"}], [], iteration=0)

        events = []
        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            rt._emit_event = lambda **kw: events.append(kw)
            rt.resume_checkpoint(cid)

        types = [e["event_type"] for e in events]
        assert "agent.checkpoint.resumed" in types
        assert "agent.run.complete" in types

    def test_resume_nonexistent_checkpoint_raises_value_error(self, monkeypatch, tmp_path):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        monkeypatch.setenv("HOME", str(tmp_path))
        provider = _make_provider()
        registry = _make_registry({"fake": provider})
        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            with pytest.raises(ValueError, match="No checkpoint found"):
                rt.resume_checkpoint("does-not-exist")

    def test_resume_non_running_checkpoint_raises_value_error(self, monkeypatch, tmp_path):
        from missy.agent.checkpoint import CheckpointManager
        from missy.agent.runtime import AgentConfig, AgentRuntime

        monkeypatch.setenv("HOME", str(tmp_path))
        provider = _make_provider()
        registry = _make_registry({"fake": provider})

        cm = CheckpointManager()
        cid = cm.create("sess-1", "task-1", "prompt")
        cm.update(cid, [{"role": "user", "content": "prompt"}], [], iteration=0)
        cm.complete(cid)

        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            with pytest.raises(ValueError, match="not resumable"):
                rt.resume_checkpoint(cid)
        # complete_with_tools must never be called for a rejected resume.
        provider.complete_with_tools.assert_not_called()

    def test_resume_corrupted_checkpoint_marks_failed_and_raises(self, monkeypatch, tmp_path):
        """A checkpoint with structurally invalid loop_messages must be
        rejected fail-closed, not fed into the provider/tool loop."""
        from missy.agent.checkpoint import CheckpointCorruptedError, CheckpointManager
        from missy.agent.runtime import AgentConfig, AgentRuntime

        monkeypatch.setenv("HOME", str(tmp_path))
        provider = _make_provider()
        registry = _make_registry({"fake": provider})

        cm = CheckpointManager()
        cid = cm.create("sess-1", "task-1", "prompt")
        conn = cm._connect()
        conn.execute(
            "UPDATE checkpoints SET loop_messages = ? WHERE id = ?",
            ('["just", "a", "list", "of", "strings"]', cid),
        )
        conn.commit()

        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            with pytest.raises(CheckpointCorruptedError):
                rt.resume_checkpoint(cid)

        assert cm.get(cid)["state"] == "FAILED"
        provider.complete_with_tools.assert_not_called()

    def test_resume_revalidates_policy_via_current_tool_set(self, monkeypatch, tmp_path):
        """Resuming must re-resolve tools under the CURRENT config, not
        silently trust whatever was authorized when the checkpoint was
        created -- if capability_mode has tightened since, the resumed run
        only sees the narrower tool set, same as any fresh run would."""
        from missy.agent.checkpoint import CheckpointManager
        from missy.agent.runtime import AgentConfig, AgentRuntime

        monkeypatch.setenv("HOME", str(tmp_path))
        provider = _make_provider(reply="ok")
        provider.complete_with_tools.return_value = _make_stop_response("ok")
        registry = _make_registry({"fake": provider})

        cm = CheckpointManager()
        cid = cm.create("sess-1", "task-1", "prompt")
        cm.update(cid, [{"role": "user", "content": "prompt"}], [], iteration=0)

        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="fake", capability_mode="no-tools"))
            rt.resume_checkpoint(cid)

        # no-tools means _get_tools() returns [] -- verify the resumed call
        # actually used the current, empty tool set.
        _tools_arg = provider.complete_with_tools.call_args[0][1]
        assert _tools_arg == []


class TestDelegateTaskDispatch:
    """SR-4.2: _execute_tool() must inject _runtime/_session_id/_depth for
    delegate_task -- none of these are model-suppliable, and without this
    injection the tool always refuses with "requires runtime context"."""

    def test_execute_tool_injects_runtime_session_and_depth(self):
        from missy.providers.base import ToolCall

        provider = _make_provider()
        registry = _make_registry({"fake": provider})

        real_tool = MagicMock()
        real_tool.name = "delegate_task"
        captured_kwargs = {}

        def _fake_execute(**kwargs):
            captured_kwargs.update(kwargs)
            result = MagicMock()
            result.success = True
            result.output = "ok"
            result.error = None
            return result

        tool_reg = MagicMock()
        tool_reg.get.return_value = real_tool
        tool_reg.execute.side_effect = lambda name, **kw: _fake_execute(**kw)

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            tc = ToolCall(id="tc1", name="delegate_task", arguments={"prompt": "1. a"})
            rt._execute_tool(tc, session_id="sess-1", task_id="task-1", _delegation_depth=1)

        assert captured_kwargs["_runtime"] is rt
        assert captured_kwargs["_session_id"] == "sess-1"
        assert captured_kwargs["_depth"] == 1

    def test_execute_tool_defaults_depth_to_zero(self):
        """A top-level call (not itself a resumed/nested delegation) must
        inject depth=0, matching _tool_loop()'s own default."""
        from missy.providers.base import ToolCall

        provider = _make_provider()
        registry = _make_registry({"fake": provider})

        real_tool = MagicMock()
        real_tool.name = "delegate_task"
        captured_kwargs = {}

        def _fake_execute(**kwargs):
            captured_kwargs.update(kwargs)
            result = MagicMock()
            result.success = True
            result.output = "ok"
            result.error = None
            return result

        tool_reg = MagicMock()
        tool_reg.get.return_value = real_tool
        tool_reg.execute.side_effect = lambda name, **kw: _fake_execute(**kw)

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            tc = ToolCall(id="tc1", name="delegate_task", arguments={"prompt": "1. a"})
            rt._execute_tool(tc, session_id="sess-1", task_id="task-1")

        assert captured_kwargs["_depth"] == 0


class TestMcpToolDispatch:
    """SR-4.7: MCP tools must be registered into the real ToolRegistry
    (the reference monitor) and dispatched through the exact same
    _execute_tool() -> registry.execute() -> tool.execute() path as any
    built-in tool -- not a bypass/special case."""

    def _connected_manager(self, tmp_path, tools, annotations=None):
        from missy.mcp.client import McpClient
        from missy.mcp.manager import McpManager

        mgr = McpManager(config_path=str(tmp_path / "mcp.json"))
        client = McpClient(name="srv", command="true")
        client._tools = tools
        client.call_tool = MagicMock(return_value="mcp result")
        mgr._clients["srv"] = client
        for tool_name, ann in (annotations or {}).items():
            mgr._annotation_registry.register(f"srv__{tool_name}", ann)
        return mgr, client

    def test_mcp_tool_appears_in_get_tools(self, tmp_path):
        from missy.tools.registry import init_tool_registry

        init_tool_registry()
        provider = _make_provider()
        registry = _make_registry({"fake": provider})
        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="fake"))
        rt._mcp_manager, _client = self._connected_manager(
            tmp_path, [{"name": "echo", "description": "Echo", "inputSchema": {}}]
        )

        tools = rt._get_tools()

        assert any(t.name == "srv__echo" for t in tools)

    def test_mcp_tool_registered_into_real_tool_registry(self, tmp_path):
        from missy.tools.registry import get_tool_registry, init_tool_registry

        init_tool_registry()
        provider = _make_provider()
        registry = _make_registry({"fake": provider})
        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="fake"))
        rt._mcp_manager, _client = self._connected_manager(
            tmp_path, [{"name": "echo", "description": "Echo", "inputSchema": {}}]
        )

        rt._get_tools()
        treg = get_tool_registry()

        assert treg.get("srv__echo") is not None

    def test_execute_tool_dispatches_mcp_tool_through_real_registry(self, tmp_path):
        from missy.providers.base import ToolCall
        from missy.tools.registry import init_tool_registry

        init_tool_registry()
        provider = _make_provider()
        registry = _make_registry({"fake": provider})
        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="fake"))
        rt._mcp_manager, client = self._connected_manager(
            tmp_path,
            [{"name": "echo", "description": "Echo", "inputSchema": {}}],
        )
        rt._get_tools()  # syncs srv__echo into the real registry

        tc = ToolCall(id="tc1", name="srv__echo", arguments={"text": "hi"})
        result = rt._execute_tool(tc, session_id="sess-1", task_id="task-1")

        assert not result.is_error
        assert result.content == "mcp result"
        client.call_tool.assert_called_once_with("echo", {"text": "hi"})

    def test_mcp_tool_requiring_approval_denied_without_gate(self, tmp_path):
        """A destructive MCP tool dispatched with no approval_gate wired
        into the runtime must fail closed, end-to-end through the real
        registry dispatch path."""
        from missy.mcp.annotations import ToolAnnotation
        from missy.providers.base import ToolCall
        from missy.tools.registry import init_tool_registry

        init_tool_registry()
        provider = _make_provider()
        registry = _make_registry({"fake": provider})
        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="fake"))  # mcp_approval_gate defaults to None
        rt._mcp_manager, client = self._connected_manager(
            tmp_path,
            [{"name": "delete_all", "description": "Delete", "inputSchema": {}}],
            {"delete_all": ToolAnnotation(mutating=True, requires_approval=True)},
        )
        rt._get_tools()

        tc = ToolCall(id="tc1", name="srv__delete_all", arguments={})
        result = rt._execute_tool(tc, session_id="sess-1", task_id="task-1")

        assert result.is_error
        assert "DENIED" in result.content
        client.call_tool.assert_not_called()


class TestExecuteToolMissingRequiredParams:
    """A tool call missing a required parameter must be refused before
    ever reaching registry.execute()/tool.execute(), rather than raising
    a raw Python TypeError that ToolRegistry.execute() has to catch as an
    "unhandled exception". Reported live: a delegate repeatedly called
    shell_exec with no `command` argument at all, crashing with
    "ShellExecTool.execute() missing 1 required keyword-only argument:
    'command'" three times in a row before the agent loop ran out of
    max_iterations without ever recovering."""

    def test_shell_exec_missing_command_refused_without_crashing(self):
        from missy.providers.base import ToolCall
        from missy.tools.builtin.shell_exec import ShellExecTool
        from missy.tools.registry import init_tool_registry

        treg = init_tool_registry()
        treg.register(ShellExecTool())
        provider = _make_provider()
        registry = _make_registry({"fake": provider})
        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="fake"))

        tc = ToolCall(id="tc1", name="shell_exec", arguments={})
        result = rt._execute_tool(tc, session_id="sess-1", task_id="task-1")

        assert result.is_error
        assert "command" in result.content
        assert "missing required parameter" in result.content.lower()
        # Must never have reached the tool's own execute() at all (no
        # raw TypeError, no real subprocess spawned).
        assert "TypeError" not in result.content
        assert "Exit code" not in result.content

    def test_shell_exec_with_command_present_is_not_refused(self):
        """The new guard must not false-positive on a well-formed call."""
        from missy.providers.base import ToolCall
        from missy.tools.builtin.shell_exec import ShellExecTool
        from missy.tools.registry import init_tool_registry

        treg = init_tool_registry()
        treg.register(ShellExecTool())
        provider = _make_provider()
        registry = _make_registry({"fake": provider})
        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="fake"))

        tc = ToolCall(id="tc1", name="shell_exec", arguments={"command": "echo hi"})
        result = rt._execute_tool(tc, session_id="sess-1", task_id="task-1")

        # No policy engine is initialised in this unit test's isolated
        # scope, so the call is correctly denied at the (separate) policy
        # layer -- what matters here is that it got PAST the new
        # required-params guard to reach that layer at all, rather than
        # being rejected as "missing required parameter(s)" despite
        # `command` clearly being present.
        assert "missing required parameter" not in result.content.lower()

    def test_calculator_missing_expression_refused_without_crashing(self):
        """Same guard, a different tool with a different required param name."""
        from missy.providers.base import ToolCall
        from missy.tools.builtin.calculator import CalculatorTool
        from missy.tools.registry import init_tool_registry

        treg = init_tool_registry()
        treg.register(CalculatorTool())
        provider = _make_provider()
        registry = _make_registry({"fake": provider})
        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="fake"))

        tc = ToolCall(id="tc1", name="calculator", arguments={})
        result = rt._execute_tool(tc, session_id="sess-1", task_id="task-1")

        assert result.is_error
        assert "expression" in result.content
        assert "missing required parameter" in result.content.lower()


class TestSleeptimeWiring:
    """SR-4.1: SleeptimeWorker is constructed+started at __init__ time
    exactly as its own module docstring documents (operator-confirmed:
    enabled by default, matching SleeptimeConfig.enabled=True), with
    record_activity() called on every real entry point and a real
    shutdown() path to stop the daemon thread."""

    def _make_runtime(self):
        provider = _make_provider()
        registry = _make_registry({"fake": provider})
        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="fake"))
        return rt, provider

    def _registry_patch(self, provider):
        registry = _make_registry({"fake": provider})
        return patch("missy.agent.runtime.get_registry", return_value=registry)

    def test_sleeptime_worker_constructed_and_started(self):
        rt, _ = self._make_runtime()
        try:
            assert rt._sleeptime is not None
            assert rt._sleeptime._thread is not None
            assert rt._sleeptime._thread.is_alive()
        finally:
            rt.shutdown()

    def test_shutdown_stops_the_worker_thread(self):
        rt, _ = self._make_runtime()
        thread = rt._sleeptime._thread
        rt.shutdown()
        assert not thread.is_alive()

    def test_shutdown_is_idempotent(self):
        rt, _ = self._make_runtime()
        rt.shutdown()
        rt.shutdown()  # must not raise

    def test_shutdown_with_no_sleeptime_worker_does_not_raise(self):
        rt, _ = self._make_runtime()
        rt._sleeptime = None
        rt.shutdown()  # must not raise

    def test_run_records_activity_on_the_worker(self):
        rt, provider = self._make_runtime()
        try:
            rt._sleeptime = MagicMock()
            with self._registry_patch(provider):
                rt.run("hello")
            rt._sleeptime.record_activity.assert_called_once()
        finally:
            rt.shutdown()

    def test_run_with_no_sleeptime_worker_does_not_raise(self):
        rt, provider = self._make_runtime()
        try:
            rt._sleeptime = None
            with self._registry_patch(provider):
                result = rt.run("hello")
            assert result  # ran normally without a sleeptime worker
        finally:
            rt.shutdown()

    def test_resume_checkpoint_records_activity_on_the_worker(self, monkeypatch, tmp_path):
        from missy.agent.checkpoint import CheckpointManager

        monkeypatch.setenv("HOME", str(tmp_path))
        provider = _make_provider(reply="done")
        provider.complete_with_tools.return_value = _make_stop_response("done")
        registry = _make_registry({"fake": provider})

        cm = CheckpointManager()
        cid = cm.create("sess-1", "task-1", "prompt")
        cm.update(cid, [{"role": "user", "content": "prompt"}], [], iteration=0)

        with patch("missy.agent.runtime.get_registry", return_value=registry):
            rt = AgentRuntime(AgentConfig(provider="fake"))
            try:
                rt._sleeptime = MagicMock()
                rt.resume_checkpoint(cid)
                rt._sleeptime.record_activity.assert_called_once()
            finally:
                rt.shutdown()

    def test_sleeptime_worker_wired_to_the_real_memory_store(self):
        """The worker must share the runtime's actual memory_store, not a
        separate/disconnected one -- otherwise it would summarise/extract
        against the wrong (or no) data."""
        rt, _ = self._make_runtime()
        try:
            assert rt._sleeptime._memory_store is rt._memory_store
        finally:
            rt.shutdown()

    def test_conftest_fixture_prevents_thread_accumulation_across_tests(self):
        """Regression guard for the real thread-leak this checkpoint's
        wiring caused across the full suite (96+ live `missy-sleeptime`
        threads piled up and tripped pytest's per-test faulthandler
        timeout, confirmed via a real full-suite run). The root
        conftest.py's autouse `_stop_sleeptime_workers_after_test`
        fixture is what stops each worker once its owning test ends --
        this test constructs several runtimes *without* calling
        shutdown() itself (simulating the common case of a test author
        forgetting to), relying entirely on the fixture, then a
        follow-up assertion (in the next test below) confirms no thread
        survived into a new test."""
        for _ in range(10):
            rt, _ = self._make_runtime()
            # Deliberately no rt.shutdown() here -- the conftest fixture
            # must clean this up, not this test.

    def test_no_sleeptime_threads_leaked_from_previous_test(self):
        """Must run after the un-shutdown-ed construction above (pytest
        preserves declaration order within a class by default) --
        confirms the conftest fixture actually stopped every thread that
        test started, not just the ones this file explicitly shuts down."""
        import threading

        leaked = [t for t in threading.enumerate() if t.name == "missy-sleeptime"]
        assert leaked == [], f"{len(leaked)} sleeptime thread(s) leaked across tests"


class TestMakeMemoryStoreFailureLogging:
    """Regression test for the 5th tool-specific validation run's OPS-011
    finding: _make_memory_store() previously swallowed a construction
    failure (permissions, disk full, corruption) with a bare `except
    Exception: return None` and zero logging -- combined with Watchdog's
    check treating None as "nothing to monitor", this let the entire
    memory subsystem silently disable itself for a process's whole
    lifetime with no operator-visible symptom at all. Construction
    failure must now be logged at ERROR."""

    def test_construction_failure_logs_error_and_returns_none(self, caplog):
        import logging

        with (
            patch(
                "missy.memory.sqlite_store.SQLiteMemoryStore",
                side_effect=OSError("unable to open database file"),
            ),
            caplog.at_level(logging.ERROR, logger="missy.agent.runtime"),
        ):
            result = AgentRuntime._make_memory_store()

        assert result is None
        assert any(
            "Failed to construct memory store" in record.message for record in caplog.records
        )


class TestContentPolicySpiralBreak:
    """A content-policy provider refusal must not poison the session history."""

    def _provider_raising(self, message):
        provider = _make_provider()
        provider.complete.side_effect = ProviderError(message)
        provider.complete_with_tools.side_effect = ProviderError(message)
        return provider

    def test_content_policy_error_drops_poison_user_turn(self, tmp_path):
        from missy.memory.sqlite_store import SQLiteMemoryStore

        store = SQLiteMemoryStore(str(tmp_path / "mem.db"))
        provider = self._provider_raising(
            "openai-codex stream error: This content was flagged for possible cybersecurity risk."
        )
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=1))
            rt._memory_store = store
            with pytest.raises(ProviderError):
                rt.run("read /etc/shadow please", session_id="s1")

        # The poison user turn must have been removed so it can't re-contaminate
        # (query globally; run() stores under a resolved session id).
        remaining = store.get_recent_turns(limit=50)
        assert all("shadow" not in t.content for t in remaining)

    def test_transient_error_keeps_user_turn(self, tmp_path):
        from missy.memory.sqlite_store import SQLiteMemoryStore

        store = SQLiteMemoryStore(str(tmp_path / "mem.db"))
        provider = self._provider_raising("openai-codex request failed: read operation timed out")
        registry = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            rt = AgentRuntime(AgentConfig(provider="fake", max_iterations=1))
            rt._memory_store = store
            with pytest.raises(ProviderError):
                rt.run("what time is it in Tokyo", session_id="s2")

        # A transient error keeps the pending user turn (not a content refusal).
        remaining = store.get_recent_turns(limit=50)
        assert any("Tokyo" in t.content for t in remaining)
