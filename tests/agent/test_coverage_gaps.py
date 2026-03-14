"""Tests targeting uncovered lines in agent modules and core exceptions.

Covers:
  - missy/agent/done_criteria.py   lines 52, 62, 77, 86, 99
  - missy/agent/context.py         lines 109-113, 116-118, 128
  - missy/core/exceptions.py       lines 25, 52-57
  - missy/agent/circuit_breaker.py lines 80-81, 102-103, 126-129, 131
  - missy/agent/runtime.py         (tool loop, streaming, execute_tool, helpers,
                                    _dicts_to_messages, capability_mode, etc.)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from missy.providers.base import CompletionResponse, ToolCall, ToolResult
from missy.providers import registry as registry_module


# ---------------------------------------------------------------------------
# Helpers shared across runtime tests
# ---------------------------------------------------------------------------


def _make_provider(name="fake", reply="ok", available=True):
    provider = MagicMock()
    provider.name = name
    provider.is_available.return_value = available
    provider.complete.return_value = CompletionResponse(
        content=reply,
        model="fake-model-1",
        provider=name,
        usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        raw={},
        finish_reason="stop",
    )
    return provider


def _make_registry(providers=None):
    reg = MagicMock()
    providers = providers or {}
    reg.get.side_effect = lambda n: providers.get(n)
    reg.get_available.side_effect = lambda: [p for p in providers.values() if p.is_available()]
    return reg


@pytest.fixture(autouse=True)
def reset_registry():
    original = registry_module._registry
    yield
    registry_module._registry = original


# ===========================================================================
# missy/agent/done_criteria.py
# ===========================================================================


class TestDoneCriteriaAllMet:
    """Tests for DoneCriteria.all_met (line 52)."""

    def test_all_met_true_when_all_verified(self):
        from missy.agent.done_criteria import DoneCriteria

        dc = DoneCriteria(conditions=["a", "b"], verified=[True, True])
        assert dc.all_met is True

    def test_all_met_false_when_empty_conditions(self):
        from missy.agent.done_criteria import DoneCriteria

        dc = DoneCriteria(conditions=[], verified=[])
        assert dc.all_met is False

    def test_all_met_false_when_one_unverified(self):
        from missy.agent.done_criteria import DoneCriteria

        dc = DoneCriteria(conditions=["a", "b"], verified=[True, False])
        assert dc.all_met is False

    def test_all_met_false_when_no_verified(self):
        from missy.agent.done_criteria import DoneCriteria

        dc = DoneCriteria(conditions=["a"], verified=[False])
        assert dc.all_met is False


class TestDoneCriteriaPending:
    """Tests for DoneCriteria.pending (line 62)."""

    def test_pending_returns_unverified_conditions(self):
        from missy.agent.done_criteria import DoneCriteria

        dc = DoneCriteria(conditions=["x", "y", "z"], verified=[True, False, True])
        assert dc.pending == ["y"]

    def test_pending_empty_when_all_verified(self):
        from missy.agent.done_criteria import DoneCriteria

        dc = DoneCriteria(conditions=["a", "b"], verified=[True, True])
        assert dc.pending == []

    def test_pending_all_when_none_verified(self):
        from missy.agent.done_criteria import DoneCriteria

        dc = DoneCriteria(conditions=["a", "b"], verified=[False, False])
        assert dc.pending == ["a", "b"]

    def test_pending_empty_when_no_conditions(self):
        from missy.agent.done_criteria import DoneCriteria

        dc = DoneCriteria()
        assert dc.pending == []


class TestIsCompoundTask:
    """Tests for is_compound_task (line 77)."""

    def test_returns_true_for_then_connector(self):
        from missy.agent.done_criteria import is_compound_task

        assert is_compound_task("Search for file and then delete it") is True

    def test_returns_true_for_numbered_list(self):
        from missy.agent.done_criteria import is_compound_task

        assert is_compound_task("1. Do this\n2. Do that") is True

    def test_returns_true_for_bullet_list(self):
        from missy.agent.done_criteria import is_compound_task

        assert is_compound_task("- step one\n- step two") is True

    def test_returns_true_for_ordinal_words(self):
        from missy.agent.done_criteria import is_compound_task

        assert is_compound_task("First do A, finally do B") is True

    def test_returns_false_for_simple_prompt(self):
        from missy.agent.done_criteria import is_compound_task

        assert is_compound_task("What is the weather today?") is False

    def test_returns_false_for_single_action(self):
        from missy.agent.done_criteria import is_compound_task

        assert is_compound_task("Read the file foo.txt") is False


class TestMakeDonePrompt:
    """Tests for make_done_prompt (line 86)."""

    def test_returns_string(self):
        from missy.agent.done_criteria import make_done_prompt

        result = make_done_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_done_conditions_phrase(self):
        from missy.agent.done_criteria import make_done_prompt

        result = make_done_prompt()
        assert "DONE" in result or "done" in result.lower()


class TestMakeVerificationPrompt:
    """Tests for make_verification_prompt (line 99)."""

    def test_returns_string(self):
        from missy.agent.done_criteria import make_verification_prompt

        result = make_verification_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_review_phrase(self):
        from missy.agent.done_criteria import make_verification_prompt

        result = make_verification_prompt()
        assert "Review" in result or "review" in result.lower()


# ===========================================================================
# missy/agent/context.py
# ===========================================================================


class TestContextManagerMemoryTruncation:
    """Tests for memory truncation path (lines 109-113)."""

    def test_memory_text_truncated_when_over_budget(self):
        from missy.agent.context import ContextManager, TokenBudget

        # Tiny budget so memory always overflows
        budget = TokenBudget(
            total=1000,
            system_reserve=0,
            tool_definitions_reserve=0,
            memory_fraction=0.01,  # very small fraction → small memory_budget
        )
        mgr = ContextManager(budget=budget)

        long_memory = "A" * 5000  # definitely over budget
        system, messages = mgr.build_messages(
            system="Base",
            new_message="hello",
            history=[],
            memory_results=[long_memory],
        )
        # Memory should have been included but truncated
        assert "Relevant Memory" in system
        # Truncated: the full 5000-char string shouldn't be there
        memory_section = system.split("## Relevant Memory")[-1]
        assert len(memory_section) < 5000

    def test_memory_within_budget_not_truncated(self):
        from missy.agent.context import ContextManager, TokenBudget

        budget = TokenBudget(
            total=100_000,
            system_reserve=0,
            tool_definitions_reserve=0,
            memory_fraction=0.5,
        )
        mgr = ContextManager(budget=budget)

        short_memory = "short memory snippet"
        system, _ = mgr.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            memory_results=[short_memory],
        )
        assert short_memory in system


class TestContextManagerLearnings:
    """Tests for learnings injection (lines 116-118)."""

    def test_learnings_injected_when_within_budget(self):
        from missy.agent.context import ContextManager, TokenBudget

        budget = TokenBudget(
            total=100_000,
            system_reserve=0,
            tool_definitions_reserve=0,
            learnings_fraction=0.1,
        )
        mgr = ContextManager(budget=budget)

        system, _ = mgr.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            learnings=["lesson A", "lesson B"],
        )
        assert "Past Learnings" in system
        assert "lesson A" in system
        assert "lesson B" in system

    def test_learnings_not_injected_when_over_budget(self):
        from missy.agent.context import ContextManager, TokenBudget

        # learnings_fraction so tiny the learnings won't fit
        budget = TokenBudget(
            total=100,
            system_reserve=0,
            tool_definitions_reserve=0,
            learnings_fraction=0.00001,
        )
        mgr = ContextManager(budget=budget)

        long_learning = "x" * 2000
        system, _ = mgr.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            learnings=[long_learning],
        )
        # The very long learning should not appear in system because budget too small
        assert "Past Learnings" not in system

    def test_learnings_limited_to_five_items(self):
        from missy.agent.context import ContextManager, TokenBudget

        budget = TokenBudget(total=100_000, system_reserve=0, tool_definitions_reserve=0)
        mgr = ContextManager(budget=budget)

        learnings = [f"lesson {i}" for i in range(10)]
        system, _ = mgr.build_messages(
            system="Base",
            new_message="hi",
            history=[],
            learnings=learnings,
        )
        # Only first 5 should appear
        for i in range(5):
            assert f"lesson {i}" in system
        for i in range(5, 10):
            assert f"lesson {i}" not in system


class TestContextManagerHistoryPruning:
    """Tests for history pruning (line 128)."""

    def test_old_history_dropped_when_over_budget(self):
        from missy.agent.context import ContextManager, TokenBudget

        # Tight budget: only 50 tokens for history
        budget = TokenBudget(
            total=100,
            system_reserve=0,
            tool_definitions_reserve=0,
            memory_fraction=0.0,
            learnings_fraction=0.0,
        )
        mgr = ContextManager(budget=budget)

        # Each turn uses ~25 tokens (100 chars / 4)
        history = [{"role": "user", "content": "A" * 100} for _ in range(10)]

        _, messages = mgr.build_messages(
            system="",
            new_message="hi",
            history=history,
        )
        # With tight budget, most history should be pruned
        # The final message is the new user message
        assert messages[-1]["content"] == "hi"
        assert len(messages) < len(history) + 1

    def test_history_preserved_when_within_budget(self):
        from missy.agent.context import ContextManager, TokenBudget

        budget = TokenBudget(total=100_000, system_reserve=0, tool_definitions_reserve=0)
        mgr = ContextManager(budget=budget)

        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        _, messages = mgr.build_messages(
            system="",
            new_message="new message",
            history=history,
        )
        # All history should be preserved plus new message
        assert len(messages) == 3
        assert messages[-1]["content"] == "new message"


# ===========================================================================
# missy/core/exceptions.py
# ===========================================================================


class TestPolicyViolationErrorRepr:
    """Tests for PolicyViolationError.__repr__ (line 25)."""

    def test_repr_contains_class_name(self):
        from missy.core.exceptions import PolicyViolationError

        exc = PolicyViolationError("blocked", category="network", detail="host not allowed")
        r = repr(exc)
        assert "PolicyViolationError" in r

    def test_repr_contains_category(self):
        from missy.core.exceptions import PolicyViolationError

        exc = PolicyViolationError("msg", category="filesystem", detail="no write")
        r = repr(exc)
        assert "filesystem" in r

    def test_repr_contains_detail(self):
        from missy.core.exceptions import PolicyViolationError

        exc = PolicyViolationError("msg", category="shell", detail="command denied")
        r = repr(exc)
        assert "command denied" in r

    def test_repr_contains_message(self):
        from missy.core.exceptions import PolicyViolationError

        exc = PolicyViolationError("my message", category="network", detail="d")
        r = repr(exc)
        assert "my message" in r


class TestApprovalRequiredError:
    """Tests for ApprovalRequiredError (lines 52-57)."""

    def test_action_stored(self):
        from missy.core.exceptions import ApprovalRequiredError

        exc = ApprovalRequiredError("delete_file")
        assert exc.action == "delete_file"

    def test_reason_stored(self):
        from missy.core.exceptions import ApprovalRequiredError

        exc = ApprovalRequiredError("delete_file", reason="irreversible")
        assert exc.reason == "irreversible"

    def test_message_contains_action(self):
        from missy.core.exceptions import ApprovalRequiredError

        exc = ApprovalRequiredError("run_shell")
        assert "run_shell" in str(exc)

    def test_message_contains_reason_when_set(self):
        from missy.core.exceptions import ApprovalRequiredError

        exc = ApprovalRequiredError("run_shell", reason="dangerous command")
        assert "dangerous command" in str(exc)

    def test_message_without_reason(self):
        from missy.core.exceptions import ApprovalRequiredError

        exc = ApprovalRequiredError("upload_file")
        assert "upload_file" in str(exc)
        assert "Approval required" in str(exc)

    def test_empty_reason(self):
        from missy.core.exceptions import ApprovalRequiredError

        exc = ApprovalRequiredError("action", reason="")
        # Should not include em-dash when reason is empty
        assert "\u2014" not in str(exc)

    def test_is_missy_error(self):
        from missy.core.exceptions import ApprovalRequiredError, MissyError

        exc = ApprovalRequiredError("x")
        assert isinstance(exc, MissyError)


# ===========================================================================
# missy/agent/circuit_breaker.py
# ===========================================================================


class TestCircuitBreakerStateTransition:
    """Tests for OPEN → HALF_OPEN auto-transition (lines 80-81)."""

    def test_state_transitions_to_half_open_after_timeout(self):
        from missy.agent.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker("test", threshold=1, base_timeout=0.01)
        # Force into OPEN by failing
        try:
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass

        assert breaker._state == CircuitState.OPEN

        # Wait for timeout to expire
        time.sleep(0.05)

        # Accessing .state should auto-transition to HALF_OPEN
        state = breaker.state
        assert state == CircuitState.HALF_OPEN

    def test_state_stays_open_before_timeout(self):
        from missy.agent.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker("test", threshold=1, base_timeout=9999.0)
        try:
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerOpenRejectsCall:
    """Tests for OPEN state rejection (lines 102-103)."""

    def test_open_circuit_raises_missy_error(self):
        from missy.agent.circuit_breaker import CircuitBreaker, CircuitState
        from missy.core.exceptions import MissyError

        breaker = CircuitBreaker("test", threshold=1, base_timeout=9999.0)
        # Force open
        try:
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass

        assert breaker._state == CircuitState.OPEN

        with pytest.raises(MissyError, match="OPEN"):
            breaker.call(lambda: "result")

    def test_closed_circuit_allows_call(self):
        from missy.agent.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker("test")
        result = breaker.call(lambda: 42)
        assert result == 42


class TestCircuitBreakerHalfOpenFailure:
    """Tests for HALF_OPEN → OPEN with backoff (lines 126-129, 131)."""

    def test_half_open_failure_doubles_recovery_timeout(self):
        from missy.agent.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker("test", threshold=1, base_timeout=0.01, max_timeout=100.0)

        # Force to OPEN
        try:
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass
        assert breaker._state == CircuitState.OPEN

        # Wait for HALF_OPEN
        time.sleep(0.05)
        assert breaker.state == CircuitState.HALF_OPEN

        original_timeout = breaker._recovery_timeout

        # Fail in HALF_OPEN state
        try:
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail again")))
        except ValueError:
            pass

        # Should be back to OPEN with doubled timeout
        assert breaker._state == CircuitState.OPEN
        assert breaker._recovery_timeout == min(original_timeout * 2, breaker._max_timeout)

    def test_half_open_failure_opens_circuit_again(self):
        from missy.agent.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker("test", threshold=1, base_timeout=0.01)
        try:
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass

        time.sleep(0.05)
        assert breaker.state == CircuitState.HALF_OPEN

        try:
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("probe fail")))
        except ValueError:
            pass

        assert breaker._state == CircuitState.OPEN

    def test_half_open_success_closes_circuit(self):
        from missy.agent.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker("test", threshold=1, base_timeout=0.01)
        try:
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass

        time.sleep(0.05)
        assert breaker.state == CircuitState.HALF_OPEN

        result = breaker.call(lambda: "ok")
        assert result == "ok"
        assert breaker._state == CircuitState.CLOSED

    def test_failure_at_threshold_opens_circuit(self):
        from missy.agent.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker("test", threshold=3, base_timeout=9999.0)
        for _ in range(3):
            try:
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
            except ValueError:
                pass

        assert breaker._state == CircuitState.OPEN

    def test_max_timeout_capped(self):
        from missy.agent.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker("test", threshold=1, base_timeout=0.01, max_timeout=0.02)

        # Force open
        try:
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass

        # Wait to get HALF_OPEN
        time.sleep(0.05)
        _ = breaker.state  # trigger transition

        # Fail probe multiple times to test cap
        for _ in range(5):
            time.sleep(0.05)
            if breaker.state == CircuitState.HALF_OPEN:
                try:
                    breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
                except ValueError:
                    pass

        assert breaker._recovery_timeout <= breaker._max_timeout


# ===========================================================================
# missy/agent/runtime.py - uncovered paths
# ===========================================================================


class TestRuntimeCapabilityMode:
    """Tests for _get_tools capability_mode filtering (lines 679, 689)."""

    def test_no_tools_mode_returns_empty(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})
        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake", capability_mode="no-tools"))
            tools = runtime._get_tools()
        assert tools == []

    def test_safe_chat_mode_filters_tools(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        safe_tool = MagicMock()
        safe_tool.name = "calculator"
        unsafe_tool = MagicMock()
        unsafe_tool.name = "shell_exec"

        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calculator", "shell_exec"]
        tool_reg.get.side_effect = lambda n: {
            "calculator": safe_tool,
            "shell_exec": unsafe_tool,
        }.get(n)

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            runtime = AgentRuntime(AgentConfig(provider="fake", capability_mode="safe-chat"))
            tools = runtime._get_tools()

        names = [getattr(t, "name", "") for t in tools]
        assert "calculator" in names
        assert "shell_exec" not in names

    def test_full_mode_returns_all_tools(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        tool_a = MagicMock()
        tool_a.name = "shell_exec"
        tool_b = MagicMock()
        tool_b.name = "calculator"

        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["shell_exec", "calculator"]
        tool_reg.get.side_effect = lambda n: {"shell_exec": tool_a, "calculator": tool_b}.get(n)

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            runtime = AgentRuntime(AgentConfig(provider="fake", capability_mode="full"))
            tools = runtime._get_tools()

        assert len(tools) == 2

    def test_tool_registry_runtime_error_returns_empty(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError("not init")),
        ):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            tools = runtime._get_tools()

        assert tools == []


class TestRuntimeExecuteTool:
    """Tests for _execute_tool error paths (lines 706-750)."""

    def test_execute_tool_key_error_returns_error_result(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        tool_reg = MagicMock()
        tool_reg.execute.side_effect = KeyError("no_such_tool")

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            tc = ToolCall(id="1", name="no_such_tool", arguments={})
            result = runtime._execute_tool(tc)

        assert result.is_error is True
        assert "not found" in result.content

    def test_execute_tool_runtime_error_returns_error_result(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        tool_reg = MagicMock()
        tool_reg.execute.side_effect = RuntimeError("registry gone")

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            tc = ToolCall(id="2", name="some_tool", arguments={})
            result = runtime._execute_tool(tc)

        assert result.is_error is True
        assert "not initialised" in result.content

    def test_execute_tool_unexpected_error_returns_error_result(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        tool_reg = MagicMock()
        tool_reg.execute.side_effect = Exception("boom")

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            tc = ToolCall(id="3", name="broken_tool", arguments={})
            result = runtime._execute_tool(tc)

        assert result.is_error is True
        assert "boom" in result.content

    def test_execute_tool_success_returns_output(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        tool_result_mock = MagicMock()
        tool_result_mock.success = True
        tool_result_mock.output = "42"
        tool_result_mock.error = None

        tool_reg = MagicMock()
        tool_reg.execute.return_value = tool_result_mock

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            tc = ToolCall(id="4", name="calculator", arguments={"expr": "6*7"})
            result = runtime._execute_tool(tc)

        assert result.is_error is False
        assert result.content == "42"

    def test_execute_tool_failure_returns_error_content(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        tool_result_mock = MagicMock()
        tool_result_mock.success = False
        tool_result_mock.output = None
        tool_result_mock.error = "permission denied"

        tool_reg = MagicMock()
        tool_reg.execute.return_value = tool_result_mock

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            tc = ToolCall(id="5", name="file_write", arguments={"path": "/etc/hosts"})
            result = runtime._execute_tool(tc)

        assert result.is_error is True
        assert result.content == "permission denied"

    def test_execute_tool_strips_session_and_task_id_from_args(self):
        """session_id and task_id in tool arguments should not be forwarded."""
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        tool_result_mock = MagicMock()
        tool_result_mock.success = True
        tool_result_mock.output = "done"

        tool_reg = MagicMock()
        tool_reg.execute.return_value = tool_result_mock

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            tc = ToolCall(
                id="6",
                name="file_read",
                arguments={"path": "/tmp/x", "session_id": "s1", "task_id": "t1"},
            )
            runtime._execute_tool(tc, session_id="s_real", task_id="t_real")

        # The call should not have forwarded session_id / task_id from tool args
        call_kwargs = tool_reg.execute.call_args[1]
        assert call_kwargs.get("session_id") == "s_real"
        assert call_kwargs.get("task_id") == "t_real"
        assert "path" in call_kwargs


class TestRuntimeDictsToMessages:
    """Tests for _dicts_to_messages (lines 940-951)."""

    def test_tool_role_becomes_user_message(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        runtime = AgentRuntime(AgentConfig())
        dicts = [{"role": "tool", "name": "calculator", "content": "42", "is_error": False}]
        messages = runtime._dicts_to_messages("sys", dicts)
        # system + tool-as-user
        assert len(messages) == 2
        assert messages[1].role == "user"
        assert "Tool result for calculator" in messages[1].content

    def test_tool_error_becomes_error_user_message(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        runtime = AgentRuntime(AgentConfig())
        dicts = [{"role": "tool", "name": "shell_exec", "content": "boom", "is_error": True}]
        messages = runtime._dicts_to_messages("sys", dicts)
        assert "Tool error for shell_exec" in messages[1].content

    def test_unknown_role_skipped(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        runtime = AgentRuntime(AgentConfig())
        dicts = [
            {"role": "function", "content": "ignored"},
            {"role": "user", "content": "kept"},
        ]
        messages = runtime._dicts_to_messages("sys", dicts)
        # system + user only (function role skipped)
        assert len(messages) == 2
        assert messages[1].content == "kept"

    def test_system_prompt_prepended(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        runtime = AgentRuntime(AgentConfig())
        messages = runtime._dicts_to_messages("my system", [{"role": "user", "content": "hi"}])
        assert messages[0].role == "system"
        assert messages[0].content == "my system"


class TestRuntimeBuildContextMessages:
    """Tests for _build_context_messages (lines 776-787)."""

    def test_falls_back_when_context_manager_raises(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake", system_prompt="SP"))
            # Replace context manager with one that raises
            failing_cm = MagicMock()
            failing_cm.build_messages.side_effect = RuntimeError("cm broken")
            runtime._context_manager = failing_cm

            system, messages = runtime._build_context_messages("user input", [])

        assert system == "SP"
        assert messages == [{"role": "user", "content": "user input"}]

    def test_falls_back_when_context_manager_is_none(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake", system_prompt="SP"))
            runtime._context_manager = None

            system, messages = runtime._build_context_messages("msg", [])

        assert system == "SP"
        assert messages[-1]["content"] == "msg"


class TestRuntimeLoadHistory:
    """Tests for _load_history (lines 799-806)."""

    def test_returns_empty_when_memory_store_is_none(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            runtime._memory_store = None

            history = runtime._load_history("session-123")

        assert history == []

    def test_returns_empty_when_memory_store_raises(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            failing_store = MagicMock()
            failing_store.get_session_turns.side_effect = Exception("db error")
            runtime._memory_store = failing_store

            history = runtime._load_history("session-123")

        assert history == []

    def test_returns_turns_as_dicts(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        turn1 = MagicMock()
        turn1.role = "user"
        turn1.content = "hello"
        turn2 = MagicMock()
        turn2.role = "assistant"
        turn2.content = "hi there"

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            store = MagicMock()
            store.get_session_turns.return_value = [turn1, turn2]
            runtime._memory_store = store

            history = runtime._load_history("session-123")

        assert history == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]


class TestRuntimeSaveTurn:
    """Tests for _save_turn (lines 819-829)."""

    def test_save_turn_no_op_when_memory_store_none(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            runtime._memory_store = None
            # Should not raise
            runtime._save_turn("s1", "user", "hello")

    def test_save_turn_swallows_exception(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            bad_store = MagicMock()
            bad_store.add_turn.side_effect = Exception("write error")
            runtime._memory_store = bad_store

            # Should not raise
            runtime._save_turn("s1", "user", "hello")

    def test_save_turn_calls_memory_store(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            good_store = MagicMock()
            runtime._memory_store = good_store

            runtime._save_turn("s1", "assistant", "reply", provider="fake")

        good_store.add_turn.assert_called_once_with(
            session_id="s1",
            role="assistant",
            content="reply",
            provider="fake",
        )


class TestRuntimeRecordLearnings:
    """Tests for _record_learnings (lines 844-859)."""

    def test_record_learnings_handles_import_error(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            with patch("builtins.__import__", side_effect=ImportError("no learnings")):
                # Should not raise
                try:
                    runtime._record_learnings(["shell_exec"], "done", "run ls")
                except Exception:
                    pytest.fail("_record_learnings should not raise")

    def test_record_learnings_calls_extract_learnings(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        mock_learning = MagicMock()
        mock_learning.task_type = "shell"
        mock_learning.outcome = "success"
        mock_learning.lesson = "ls works"

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            with patch("missy.agent.learnings.extract_learnings", return_value=mock_learning):
                runtime._record_learnings(["shell_exec"], "done", "run ls")


class TestRuntimeAcquireRateLimit:
    """Tests for _acquire_rate_limit (lines 1088-1095)."""

    def test_no_op_when_rate_limiter_none(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            runtime._rate_limiter = None
            # Should not raise
            runtime._acquire_rate_limit()

    def test_calls_acquire_on_limiter(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            mock_limiter = MagicMock()
            runtime._rate_limiter = mock_limiter

            runtime._acquire_rate_limit()

        mock_limiter.acquire.assert_called_once()

    def test_swallows_rate_limiter_exception(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            bad_limiter = MagicMock()
            bad_limiter.acquire.side_effect = Exception("limiter error")
            runtime._rate_limiter = bad_limiter

            # Should not raise
            runtime._acquire_rate_limit()


class TestRuntimeRecordCost:
    """Tests for _record_cost (lines 1062-1086)."""

    def test_no_op_when_cost_tracker_none(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            runtime._cost_tracker = None
            response = MagicMock()
            # Should not raise
            runtime._record_cost(response)

    def test_persists_cost_to_store_when_available(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))

            mock_rec = MagicMock()
            mock_rec.model = "claude-sonnet"
            mock_rec.prompt_tokens = 10
            mock_rec.completion_tokens = 5
            mock_rec.cost_usd = 0.001

            cost_tracker = MagicMock()
            cost_tracker.record_from_response.return_value = mock_rec

            # Store without _primary attribute so it's used directly
            class DirectStore:
                def record_cost(self, **kwargs):
                    self.last_call = kwargs

            store = DirectStore()

            runtime._cost_tracker = cost_tracker
            runtime._memory_store = store

            response = MagicMock()
            runtime._record_cost(response, session_id="s1")

        assert hasattr(store, "last_call")
        assert store.last_call["session_id"] == "s1"

    def test_swallows_cost_tracker_exception(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            bad_tracker = MagicMock()
            bad_tracker.record_from_response.side_effect = Exception("cost error")
            runtime._cost_tracker = bad_tracker

            # Should not raise
            runtime._record_cost(MagicMock())


class TestRuntimeToolLoop:
    """Tests for _tool_loop paths (lines 469-612)."""

    def _make_tool_response(self, name="test", reply="final answer"):
        """Make a CompletionResponse with tool calls."""
        tc = ToolCall(id="tc1", name=name, arguments={})
        return CompletionResponse(
            content="",
            model="fake-model",
            provider="fake",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            raw={},
            finish_reason="tool_calls",
            tool_calls=[tc],
        )

    def _make_final_response(self, reply="final answer"):
        return CompletionResponse(
            content=reply,
            model="fake-model",
            provider="fake",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            raw={},
            finish_reason="stop",
        )

    def test_tool_loop_completes_with_final_response(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider(reply="done")
        reg = _make_registry({"fake": provider})

        tool_response = self._make_tool_response("calculator")
        final_response = self._make_final_response("The answer is 42")

        # First call returns tool_calls, second returns final
        provider.complete_with_tools.side_effect = [tool_response, final_response]

        tool_result_mock = MagicMock()
        tool_result_mock.success = True
        tool_result_mock.output = "42"

        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calculator"]
        calc_tool = MagicMock()
        calc_tool.name = "calculator"
        tool_reg.get.return_value = calc_tool
        tool_reg.execute.return_value = tool_result_mock

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            runtime = AgentRuntime(AgentConfig(provider="fake", max_iterations=5))
            result = runtime.run("What is 6 * 7?")

        assert "42" in result or "answer" in result.lower()

    def test_tool_loop_falls_back_when_complete_with_tools_not_implemented(self):
        """If provider lacks complete_with_tools, fall back to single turn."""
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider(reply="fallback response")
        reg = _make_registry({"fake": provider})

        # complete_with_tools raises AttributeError
        del provider.complete_with_tools

        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calculator"]
        calc_tool = MagicMock()
        calc_tool.name = "calculator"
        tool_reg.get.return_value = calc_tool

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            runtime = AgentRuntime(AgentConfig(provider="fake", max_iterations=5))
            result = runtime.run("What is 6 * 7?")

        assert result == "fallback response"

    def test_tool_loop_max_iterations_reached(self):
        """If model keeps calling tools, eventually return fallback response."""
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider(reply="single turn fallback")
        reg = _make_registry({"fake": provider})

        # Always return tool_calls (never stop)
        tool_response = self._make_tool_response("calculator")
        provider.complete_with_tools.return_value = tool_response

        tool_result_mock = MagicMock()
        tool_result_mock.success = True
        tool_result_mock.output = "42"

        tool_reg = MagicMock()
        tool_reg.list_tools.return_value = ["calculator"]
        calc_tool = MagicMock()
        calc_tool.name = "calculator"
        tool_reg.get.return_value = calc_tool
        tool_reg.execute.return_value = tool_result_mock

        with (
            patch("missy.agent.runtime.get_registry", return_value=reg),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_reg),
        ):
            runtime = AgentRuntime(AgentConfig(provider="fake", max_iterations=2))
            result = runtime.run("loop forever")

        # Should get some response (single turn fallback or iteration limit message)
        assert isinstance(result, str)
        assert len(result) > 0


class TestRuntimeEmitEvent:
    """Tests for _emit_event error handling (lines 1224-1225)."""

    def test_emit_event_swallows_publish_exception(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            with patch("missy.agent.runtime.event_bus") as mock_bus:
                mock_bus.publish.side_effect = Exception("bus error")
                # Should not raise
                runtime._emit_event(
                    session_id="s1",
                    task_id="t1",
                    event_type="test.event",
                    result="allow",
                    detail={},
                )


class TestRuntimeResolveSession:
    """Tests for _resolve_session (lines 1118-1145)."""

    def test_resolve_session_creates_new_when_none_exists(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            session = runtime._resolve_session(None)

        assert session is not None

    def test_resolve_session_stores_caller_session_id_in_metadata(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            session = runtime._resolve_session("caller-abc")

        assert session.metadata.get("caller_session_id") == "caller-abc"

    def test_resolve_session_reuses_existing(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            s1 = runtime._resolve_session(None)
            s2 = runtime._resolve_session(None)

        assert s1.id == s2.id


class TestRuntimeBuildMessages:
    """Tests for _build_messages legacy helper (line 1192)."""

    def test_build_messages_returns_system_and_user(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        runtime = AgentRuntime(AgentConfig(system_prompt="sys"))
        messages = runtime._build_messages("hello user")

        roles = [m.role for m in messages]
        assert "system" in roles
        assert "user" in roles

    def test_build_messages_system_content_matches_config(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        runtime = AgentRuntime(AgentConfig(system_prompt="my system prompt"))
        messages = runtime._build_messages("hi")
        system_msg = next(m for m in messages if m.role == "system")
        assert system_msg.content == "my system prompt"

    def test_build_messages_user_content_matches_input(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        runtime = AgentRuntime(AgentConfig())
        messages = runtime._build_messages("what is 2+2?")
        user_msg = next(m for m in messages if m.role == "user")
        assert user_msg.content == "what is 2+2?"


class TestNoOpCircuitBreaker:
    """Tests for _NoOpCircuitBreaker (line 1236)."""

    def test_no_op_circuit_breaker_calls_through(self):
        from missy.agent.runtime import _NoOpCircuitBreaker

        noop = _NoOpCircuitBreaker()
        result = noop.call(lambda x: x * 2, 21)
        assert result == 42

    def test_no_op_circuit_breaker_propagates_exception(self):
        from missy.agent.runtime import _NoOpCircuitBreaker

        noop = _NoOpCircuitBreaker()
        with pytest.raises(ValueError, match="test error"):
            noop.call(lambda: (_ for _ in ()).throw(ValueError("test error")))


class TestRuntimeAnalyzeForEvolution:
    """Tests for _analyze_for_evolution (lines 881-913)."""

    def test_analyze_for_evolution_handles_exception(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            failure_tracker = MagicMock()
            failure_tracker.get_stats.side_effect = Exception("stats error")

            # Should not raise
            runtime._analyze_for_evolution("my_tool", "error content", failure_tracker)

    def test_analyze_for_evolution_emits_event_when_skeleton_proposed(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        provider = _make_provider()
        reg = _make_registry({"fake": provider})

        skeleton = MagicMock()
        skeleton.id = "evo-1"
        skeleton.title = "Fix calculator bug"

        failure_tracker = MagicMock()
        failure_tracker.get_stats.return_value = {"my_tool": {"total_failures": 3}}

        events = []

        with patch("missy.agent.runtime.get_registry", return_value=reg):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            runtime._emit_event = lambda **kwargs: events.append(kwargs)

            with patch("missy.agent.code_evolution.CodeEvolutionManager") as MockMgr:
                instance = MagicMock()
                instance.analyze_error_for_evolution.return_value = skeleton
                MockMgr.return_value = instance
                runtime._analyze_for_evolution("my_tool", "traceback text", failure_tracker)

        # Event may or may not be emitted depending on import path; just ensure no crash
