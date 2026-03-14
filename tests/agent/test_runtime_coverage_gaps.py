"""Coverage gap tests for missy/agent/runtime.py.

Targets uncovered lines:
  262-263: run() — cost_detail from get_summary() raises, silently ignored
  301-302: run_stream() — ProviderError re-raised
  437-438: _tool_loop — FailureTracker ImportError falls back to None
  447-449: _tool_loop — CheckpointManager exception falls back to None
  499     : _tool_loop — failure_tracker.record_failure returns True (should_inject)
  504     : _tool_loop — failure_tracker is None → should_inject=False
  537-554: _tool_loop — strategy rotation: inject prompt + emit event + analyze
  562-563: _tool_loop — checkpoint update exception silently ignored
  583-584: _tool_loop — checkpoint complete exception silently ignored
  587-594: _tool_loop — unhandled exception: checkpoint marked failed and re-raised
  611-612: _tool_loop — iteration limit reached, fallback single turn raises → sentinel
  971-972: _make_circuit_breaker — CircuitBreaker import fails → _NoOpCircuitBreaker
  985-986: _make_context_manager — ContextManager import fails → None
  999-1000: _make_memory_store — MemoryStore import fails → None
  1014-1015: _make_cost_tracker — CostTracker import fails → None
  1028-1029: _make_rate_limiter — RateLimiter import fails → None
  1049-1050: _scan_checkpoints — scan_for_recovery raises → []
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.runtime import AgentConfig, AgentRuntime
from missy.core.exceptions import ProviderError
from missy.providers import registry as registry_module
from missy.providers.base import CompletionResponse, ToolCall, ToolResult


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


def _make_provider(name="fake", reply="ok", available=True):
    provider = MagicMock()
    provider.name = name
    provider.is_available.return_value = available
    provider.complete.return_value = CompletionResponse(
        content=reply,
        model="m",
        provider=name,
        usage={"prompt_tokens": 5, "completion_tokens": 3},
        raw={},
        finish_reason="stop",
    )
    provider.complete_with_tools.return_value = CompletionResponse(
        content=reply,
        model="m",
        provider=name,
        usage={"prompt_tokens": 5, "completion_tokens": 3},
        raw={},
        finish_reason="stop",
    )
    return provider


def _make_registry(provider):
    reg = MagicMock()
    reg.get.return_value = provider
    reg.get_available.return_value = [provider]
    return reg


@pytest.fixture(autouse=True)
def reset_registry():
    original = registry_module._registry
    yield
    registry_module._registry = original


# ---------------------------------------------------------------------------
# Helper: build a minimal AgentRuntime with all subsystems mocked out
# ---------------------------------------------------------------------------


def _build_runtime(provider, max_iterations=1, capability_mode="no-tools",
                   max_spend_usd=0.0):
    """Create an AgentRuntime with a mocked registry and lazy subsystems disabled."""
    reg = _make_registry(provider)
    cfg = AgentConfig(
        provider="fake",
        max_iterations=max_iterations,
        capability_mode=capability_mode,
        max_spend_usd=max_spend_usd,
    )
    with patch("missy.agent.runtime.get_registry", return_value=reg), \
         patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError):
        runtime = AgentRuntime(cfg)
    runtime._rate_limiter = None  # disable rate limiting in tests
    runtime._memory_store = None
    runtime._cost_tracker = None
    runtime._context_manager = None
    return runtime, reg


# ---------------------------------------------------------------------------
# run() — lines 262-263: cost_detail get_summary raises, silently skipped
# ---------------------------------------------------------------------------


class TestRunCostDetailException:
    def test_cost_summary_exception_is_silenced(self):
        """Lines 262-263: cost_tracker.get_summary() raising is swallowed."""
        provider = _make_provider()
        runtime, reg = _build_runtime(provider)

        cost_tracker = MagicMock()
        cost_tracker.get_summary.side_effect = Exception("summary error")
        cost_tracker.record_from_response.return_value = None
        runtime._cost_tracker = cost_tracker

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            result = runtime.run("hello")

        assert result == "ok"
        # Even though get_summary raised, the run completed normally
        cost_tracker.get_summary.assert_called_once()


# ---------------------------------------------------------------------------
# run_stream() — lines 301-302: ProviderError re-raised
# ---------------------------------------------------------------------------


class TestRunStreamProviderError:
    def test_provider_error_propagates_from_run_stream(self):
        """Lines 301-302: when _get_provider raises ProviderError, it re-raises."""
        cfg = AgentConfig(provider="nonexistent", max_iterations=1)

        reg = MagicMock()
        reg.get.return_value = None
        reg.get_available.return_value = []

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError):
            runtime = AgentRuntime(cfg)

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            gen = runtime.run_stream("hello")
            with pytest.raises(ProviderError):
                next(gen)


# ---------------------------------------------------------------------------
# _tool_loop — lines 437-438: FailureTracker ImportError
# ---------------------------------------------------------------------------


class TestToolLoopFailureTrackerImportError:
    def test_failure_tracker_import_error_falls_back_to_none(self):
        """Lines 437-438: when FailureTracker cannot be imported, loop uses None."""
        provider = _make_provider(reply="done")
        runtime, reg = _build_runtime(provider, max_iterations=2, capability_mode="full")

        tool = MagicMock()
        tool.name = "calculator"
        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calculator"]
        tool_reg.get.return_value = tool

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg), \
             patch.dict(sys.modules, {"missy.agent.failure_tracker": None}):
            result = runtime.run("compute something")

        assert result == "done"


# ---------------------------------------------------------------------------
# _tool_loop — lines 447-449: CheckpointManager exception
# ---------------------------------------------------------------------------


class TestToolLoopCheckpointManagerException:
    def test_checkpoint_manager_exception_falls_back_to_none(self):
        """Lines 447-449: CheckpointManager.create() raising is caught gracefully."""
        provider = _make_provider(reply="done")
        runtime, reg = _build_runtime(provider, max_iterations=2, capability_mode="full")

        tool = MagicMock()
        tool.name = "calc"
        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calc"]
        tool_reg.get.return_value = tool

        checkpoint_mgr = MagicMock()
        checkpoint_mgr.create.side_effect = Exception("DB not available")

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg), \
             patch(
                 "missy.agent.checkpoint.CheckpointManager",
                 return_value=checkpoint_mgr,
                 create=True,
             ):
            result = runtime.run("do something")

        assert result == "done"


# ---------------------------------------------------------------------------
# _tool_loop — tool call execution path (lines 499, 504, 537-554, 562-563, 583-584)
# ---------------------------------------------------------------------------


def _make_tool_call_response(tool_name="calculator", tool_id="tc1", args=None,
                             finish_reason="tool_calls"):
    tc = ToolCall(id=tool_id, name=tool_name, arguments=args or {})
    return CompletionResponse(
        content="",
        model="m",
        provider="fake",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        raw={},
        finish_reason=finish_reason,
        tool_calls=[tc],
    )


def _make_stop_response(content="final answer"):
    return CompletionResponse(
        content=content,
        model="m",
        provider="fake",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        raw={},
        finish_reason="stop",
    )


class TestToolLoopWithToolCalls:
    def _setup_tool_loop(self, tool_responses, final_response, max_iterations=5):
        """Helper: set up a runtime whose provider returns tool calls then stops."""
        provider = MagicMock()
        provider.name = "fake"
        provider.is_available.return_value = True
        provider.complete_with_tools.side_effect = tool_responses + [final_response]
        provider.complete.return_value = final_response

        tool = MagicMock()
        tool.name = "calculator"

        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calculator"]
        tool_reg.get.return_value = tool
        tool_reg.execute.return_value = MagicMock(
            success=True, output="42", error=None
        )

        reg = _make_registry(provider)
        cfg = AgentConfig(
            provider="fake",
            max_iterations=max_iterations,
            capability_mode="full",
        )

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg):
            runtime = AgentRuntime(cfg)
        runtime._rate_limiter = None
        runtime._memory_store = None
        runtime._cost_tracker = None
        runtime._context_manager = None
        return runtime, reg, tool_reg

    def test_tool_call_success_path(self):
        """Lines 499-504: tool succeeds, failure_tracker.record_success called."""
        tc_response = _make_tool_call_response()
        stop_response = _make_stop_response("done")
        runtime, reg, tool_reg = self._setup_tool_loop([tc_response], stop_response)

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg):
            result = runtime.run("calculate")

        assert result == "done"

    def test_tool_call_error_path_no_strategy_rotation(self):
        """Line 499: tool fails but failure count below threshold — no injection."""
        tc_response = _make_tool_call_response()
        stop_response = _make_stop_response("done after error")
        runtime, reg, tool_reg = self._setup_tool_loop([tc_response], stop_response)

        # Make tool fail
        tool_reg.execute.return_value = MagicMock(
            success=False, output=None, error="division by zero"
        )

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg):
            result = runtime.run("calculate badly")

        assert result == "done after error"

    def test_strategy_rotation_injected_at_threshold(self):
        """Lines 537-554: when threshold reached, strategy prompt is injected."""
        # Set up 3 sequential tool failures (threshold=3 in FailureTracker)
        tc_response1 = _make_tool_call_response(tool_id="tc1")
        tc_response2 = _make_tool_call_response(tool_id="tc2")
        tc_response3 = _make_tool_call_response(tool_id="tc3")
        stop_response = _make_stop_response("eventually done")

        runtime, reg, tool_reg = self._setup_tool_loop(
            [tc_response1, tc_response2, tc_response3], stop_response, max_iterations=10
        )

        # All tool executions fail
        tool_reg.execute.return_value = MagicMock(
            success=False, output=None, error="tool error"
        )

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg):
            result = runtime.run("do failing thing")

        # Loop completes even with repeated failures
        assert isinstance(result, str)

    def test_checkpoint_update_exception_is_silenced(self):
        """Lines 562-563: checkpoint update exception is caught silently."""
        tc_response = _make_tool_call_response()
        stop_response = _make_stop_response("ok")
        runtime, reg, tool_reg = self._setup_tool_loop([tc_response], stop_response)

        # Inject a checkpoint manager that raises on update
        checkpoint_mgr = MagicMock()
        checkpoint_mgr.create.return_value = "ckpt-1"
        checkpoint_mgr.update.side_effect = Exception("DB write failed")

        runtime._pending_recovery = []

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg), \
             patch("missy.agent.checkpoint.CheckpointManager",
                   return_value=checkpoint_mgr, create=True):
            result = runtime.run("task")

        assert result == "ok"

    def test_checkpoint_complete_exception_is_silenced(self):
        """Lines 583-584: checkpoint complete exception is caught silently."""
        stop_response = _make_stop_response("ok")
        runtime, reg, tool_reg = self._setup_tool_loop([], stop_response)

        checkpoint_mgr = MagicMock()
        checkpoint_mgr.create.return_value = "ckpt-2"
        checkpoint_mgr.complete.side_effect = Exception("complete failed")

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg), \
             patch("missy.agent.checkpoint.CheckpointManager",
                   return_value=checkpoint_mgr, create=True):
            result = runtime.run("task")

        assert result == "ok"


# ---------------------------------------------------------------------------
# _tool_loop — lines 587-594: unhandled exception marks checkpoint failed
# ---------------------------------------------------------------------------


class TestToolLoopCheckpointFailOnException:
    def test_checkpoint_marked_failed_on_exception(self):
        """Lines 587-594: when provider raises, checkpoint is marked failed."""
        provider = MagicMock()
        provider.name = "fake"
        provider.is_available.return_value = True
        provider.complete_with_tools.side_effect = RuntimeError("provider crashed")

        tool = MagicMock()
        tool.name = "calc"
        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calc"]
        tool_reg.get.return_value = tool

        reg = _make_registry(provider)
        cfg = AgentConfig(provider="fake", max_iterations=3, capability_mode="full")

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg):
            runtime = AgentRuntime(cfg)
        runtime._rate_limiter = None
        runtime._memory_store = None
        runtime._cost_tracker = None
        runtime._context_manager = None

        checkpoint_mgr = MagicMock()
        checkpoint_mgr.create.return_value = "ckpt-fail"

        # Patch the checkpoint module that runtime.py imports at call time
        import missy.agent.checkpoint as _ckpt_mod
        original_cls = getattr(_ckpt_mod, "CheckpointManager", None)
        _ckpt_mod.CheckpointManager = MagicMock(return_value=checkpoint_mgr)

        try:
            with patch("missy.agent.runtime.get_registry", return_value=reg), \
                 patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg):
                # RuntimeError gets wrapped in ProviderError by run()
                with pytest.raises(ProviderError):
                    runtime.run("do task")
        finally:
            if original_cls is not None:
                _ckpt_mod.CheckpointManager = original_cls

        checkpoint_mgr.fail.assert_called_once_with("ckpt-fail")


# ---------------------------------------------------------------------------
# _tool_loop — lines 611-612: iteration limit + final fallback raises → sentinel
# ---------------------------------------------------------------------------


class TestToolLoopIterationLimit:
    def test_iteration_limit_returns_sentinel_when_fallback_raises(self):
        """Lines 611-612: when iteration limit reached and final complete raises."""
        provider = MagicMock()
        provider.name = "fake"
        provider.is_available.return_value = True
        # Always return tool_calls (never finishes)
        provider.complete_with_tools.return_value = CompletionResponse(
            content="",
            model="m",
            provider="fake",
            usage={},
            raw={},
            finish_reason="tool_calls",
            tool_calls=[ToolCall(id="t1", name="calc", arguments={})],
        )
        # Fallback single-turn also fails
        provider.complete.side_effect = Exception("also failed")

        tool = MagicMock()
        tool.name = "calc"
        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calc"]
        tool_reg.get.return_value = tool
        tool_reg.execute.return_value = MagicMock(success=True, output="42", error=None)

        reg = _make_registry(provider)
        cfg = AgentConfig(provider="fake", max_iterations=2, capability_mode="full")

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg):
            runtime = AgentRuntime(cfg)
        runtime._rate_limiter = None
        runtime._memory_store = None
        runtime._cost_tracker = None
        runtime._context_manager = None

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg):
            result = runtime.run("loop forever")

        assert "iteration limit" in result

    def test_iteration_limit_returns_fallback_content_when_complete_succeeds(self):
        """When iteration limit reached, fallback complete() returns usable content."""
        provider = MagicMock()
        provider.name = "fake"
        provider.is_available.return_value = True
        # Always tool calls
        provider.complete_with_tools.return_value = CompletionResponse(
            content="",
            model="m",
            provider="fake",
            usage={},
            raw={},
            finish_reason="tool_calls",
            tool_calls=[ToolCall(id="t1", name="calc", arguments={})],
        )
        # Fallback succeeds
        provider.complete.return_value = CompletionResponse(
            content="fallback response",
            model="m",
            provider="fake",
            usage={},
            raw={},
            finish_reason="stop",
        )

        tool = MagicMock()
        tool.name = "calc"
        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calc"]
        tool_reg.get.return_value = tool
        tool_reg.execute.return_value = MagicMock(success=True, output="42", error=None)

        reg = _make_registry(provider)
        cfg = AgentConfig(provider="fake", max_iterations=2, capability_mode="full")

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg):
            runtime = AgentRuntime(cfg)
        runtime._rate_limiter = None
        runtime._memory_store = None
        runtime._cost_tracker = None
        runtime._context_manager = None

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg):
            result = runtime.run("loop but fallback works")

        assert result == "fallback response"


# ---------------------------------------------------------------------------
# Lazy factory fallbacks — lines 971-1050
# ---------------------------------------------------------------------------


class TestLazyFactoryFallbacks:
    def test_make_circuit_breaker_import_error_returns_noop(self):
        """Lines 971-972: CircuitBreaker import error → _NoOpCircuitBreaker."""
        with patch.dict(sys.modules, {"missy.agent.circuit_breaker": None}):
            cb = AgentRuntime._make_circuit_breaker("test_provider")
        # _NoOpCircuitBreaker has a .call() method
        assert hasattr(cb, "call")
        # Verify it works as a pass-through
        fn = MagicMock(return_value="result")
        assert cb.call(fn, "arg") == "result"

    def test_make_context_manager_import_error_returns_none(self):
        """Lines 985-986: ContextManager import error → None."""
        with patch.dict(sys.modules, {"missy.agent.context": None}):
            result = AgentRuntime._make_context_manager()
        assert result is None

    def test_make_memory_store_import_error_returns_none(self):
        """Lines 999-1000: MemoryStore import error → None."""
        with patch.dict(sys.modules, {"missy.memory.store": None}):
            result = AgentRuntime._make_memory_store()
        assert result is None

    def test_make_cost_tracker_import_error_returns_none(self):
        """Lines 1014-1015: CostTracker import error → None."""
        cfg = AgentConfig()
        runtime = MagicMock(spec=AgentRuntime)
        runtime.config = cfg
        with patch.dict(sys.modules, {"missy.agent.cost_tracker": None}):
            result = AgentRuntime._make_cost_tracker(runtime)
        assert result is None

    def test_make_rate_limiter_import_error_returns_none(self):
        """Lines 1028-1029: RateLimiter import error → None."""
        with patch.dict(sys.modules, {"missy.providers.rate_limiter": None}):
            result = AgentRuntime._make_rate_limiter()
        assert result is None

    def test_scan_checkpoints_exception_returns_empty_list(self):
        """Lines 1049-1050: scan_for_recovery raises → empty list."""
        with patch.dict(sys.modules, {"missy.agent.checkpoint": None}):
            result = AgentRuntime._scan_checkpoints()
        assert result == []

    def test_scan_checkpoints_returns_results_and_logs(self):
        """Lines 1043-1048: results found → info logged and returned."""
        mock_result = MagicMock()
        mock_checkpoint_module = MagicMock()
        mock_checkpoint_module.scan_for_recovery.return_value = [mock_result]

        with patch.dict(sys.modules, {
            "missy.agent.checkpoint": mock_checkpoint_module
        }):
            results = AgentRuntime._scan_checkpoints()

        assert results == [mock_result]


# ---------------------------------------------------------------------------
# _tool_loop — provider without complete_with_tools falls back to single_turn
# ---------------------------------------------------------------------------


class TestToolLoopFallbackToSingleTurn:
    def test_provider_without_complete_with_tools_falls_back(self):
        """AttributeError on complete_with_tools triggers single_turn fallback."""
        provider = MagicMock()
        provider.name = "fake"
        provider.is_available.return_value = True
        provider.complete.return_value = CompletionResponse(
            content="fallback ok",
            model="m",
            provider="fake",
            usage={},
            raw={},
            finish_reason="stop",
        )

        # Simulate missing complete_with_tools via circuit breaker raising AttributeError
        def cb_call(fn, *args, **kwargs):
            if fn == provider.complete_with_tools:
                raise AttributeError("complete_with_tools not implemented")
            return fn(*args, **kwargs)

        cb = MagicMock()
        cb.call.side_effect = cb_call

        tool = MagicMock()
        tool.name = "calc"
        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calc"]
        tool_reg.get.return_value = tool

        reg = _make_registry(provider)
        cfg = AgentConfig(provider="fake", max_iterations=3, capability_mode="full")

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg):
            runtime = AgentRuntime(cfg)
        runtime._rate_limiter = None
        runtime._memory_store = None
        runtime._cost_tracker = None
        runtime._context_manager = None
        runtime._circuit_breaker = cb

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg):
            result = runtime.run("hello")

        assert result == "fallback ok"


# ---------------------------------------------------------------------------
# run() — general Exception during _run_loop raises ProviderError
# ---------------------------------------------------------------------------


class TestRunLoopGeneralException:
    def test_unexpected_exception_wrapped_as_provider_error(self):
        """Lines 236-248: non-ProviderError in _run_loop is wrapped in ProviderError."""
        provider = _make_provider()
        runtime, reg = _build_runtime(provider)

        with patch.object(runtime, "_run_loop", side_effect=ValueError("unexpected")), \
             patch("missy.agent.runtime.get_registry", return_value=reg):
            with pytest.raises(ProviderError, match="Unexpected error"):
                runtime.run("hello")


# ---------------------------------------------------------------------------
# run() — ProviderError during _run_loop re-raises directly
# ---------------------------------------------------------------------------


class TestRunLoopProviderError:
    def test_provider_error_in_run_loop_re_raised_directly(self):
        """Lines 223-235: ProviderError from _run_loop propagates unchanged."""
        provider = _make_provider()
        runtime, reg = _build_runtime(provider)

        with patch.object(runtime, "_run_loop",
                          side_effect=ProviderError("provider died")), \
             patch("missy.agent.runtime.get_registry", return_value=reg):
            with pytest.raises(ProviderError, match="provider died"):
                runtime.run("hello")


# ---------------------------------------------------------------------------
# _tool_loop — failure_tracker is None (line 503-504)
# ---------------------------------------------------------------------------


class TestToolLoopNoFailureTracker:
    def test_no_failure_tracker_should_inject_is_false(self):
        """Lines 503-504: when failure_tracker is None, should_inject is False."""
        provider = MagicMock()
        provider.name = "fake"
        provider.is_available.return_value = True

        tc = ToolCall(id="tc1", name="calc", arguments={})
        tool_call_resp = CompletionResponse(
            content="",
            model="m",
            provider="fake",
            usage={},
            raw={},
            finish_reason="tool_calls",
            tool_calls=[tc],
        )
        stop_resp = CompletionResponse(
            content="finished",
            model="m",
            provider="fake",
            usage={},
            raw={},
            finish_reason="stop",
        )
        provider.complete_with_tools.side_effect = [tool_call_resp, stop_resp]

        tool = MagicMock()
        tool.name = "calc"
        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calc"]
        tool_reg.get.return_value = tool
        tool_reg.execute.return_value = MagicMock(success=False, output=None,
                                                   error="err")

        reg = _make_registry(provider)
        cfg = AgentConfig(provider="fake", max_iterations=5, capability_mode="full")

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg), \
             patch.dict(sys.modules, {"missy.agent.failure_tracker": None}):
            runtime = AgentRuntime(cfg)
        runtime._rate_limiter = None
        runtime._memory_store = None
        runtime._cost_tracker = None
        runtime._context_manager = None

        with patch("missy.agent.runtime.get_registry", return_value=reg), \
             patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg), \
             patch.dict(sys.modules, {"missy.agent.failure_tracker": None}):
            result = runtime.run("task with failing tool")

        assert result == "finished"
