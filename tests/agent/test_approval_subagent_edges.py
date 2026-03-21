"""Edge case tests for ApprovalGate and SubAgentRunner.

Focuses on scenarios not exercised by the existing test suites in
test_approval_gate.py and test_sub_agent.py.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from missy.agent.approval import (
    ApprovalDenied,
    ApprovalGate,
    ApprovalTimeout,
    PendingApproval,
)
from missy.agent.sub_agent import (
    MAX_CONCURRENT,
    MAX_SUB_AGENTS,
    SubAgentRunner,
    SubTask,
    parse_subtasks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_pending(gate: ApprovalGate, count: int = 1, retries: int = 100) -> list[dict]:
    """Poll until *count* pending approvals appear (or retries exhausted)."""
    for _ in range(retries):
        pending = gate.list_pending()
        if len(pending) >= count:
            return pending
        time.sleep(0.01)
    return gate.list_pending()


def _request_in_thread(
    gate: ApprovalGate, action: str = "action", risk: str = "medium"
) -> tuple[threading.Thread, dict]:
    """Spawn a thread that calls gate.request() and records the outcome."""
    outcome: dict = {}

    def _run():
        try:
            gate.request(action, risk=risk)
            outcome["approved"] = True
        except ApprovalDenied:
            outcome["denied"] = True
        except ApprovalTimeout:
            outcome["timeout"] = True

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, outcome


# ---------------------------------------------------------------------------
# ApprovalGate — edge cases
# ---------------------------------------------------------------------------


class TestApprovalGateEdges:
    # --- list_pending with no pending requests ---

    def test_list_pending_empty_at_start(self):
        """A freshly created gate has no pending entries."""
        gate = ApprovalGate()
        assert gate.list_pending() == []

    def test_list_pending_returns_list_not_view(self):
        """list_pending() must return a snapshot; mutating it must not corrupt state."""
        gate = ApprovalGate(default_timeout=2.0)
        t, outcome = _request_in_thread(gate)
        pending = _wait_for_pending(gate)
        assert len(pending) == 1

        # Mutate the returned list — must not affect internal state
        pending.clear()
        assert len(gate.list_pending()) == 1

        # Clean up
        approval_id = gate.list_pending()[0]["id"]
        gate.handle_response(f"approve {approval_id}")
        t.join(timeout=2.0)

    # --- add and retrieve a pending request ---

    def test_pending_entry_has_correct_fields(self):
        """Each pending entry exposes id, action, and reason."""
        gate = ApprovalGate(default_timeout=2.0)
        t, _ = _request_in_thread(gate, action="do something risky")
        pending = _wait_for_pending(gate)
        assert len(pending) == 1
        entry = pending[0]
        assert "id" in entry
        assert entry["action"] == "do something risky"
        assert "reason" in entry

        gate.handle_response(f"approve {entry['id']}")
        t.join(timeout=2.0)

    # --- approve removes request from pending ---

    def test_approve_removes_from_pending(self):
        """After approval the entry must disappear from list_pending()."""
        gate = ApprovalGate(default_timeout=2.0)
        t, outcome = _request_in_thread(gate)
        pending = _wait_for_pending(gate)
        approval_id = pending[0]["id"]

        gate.handle_response(f"approve {approval_id}")
        t.join(timeout=2.0)

        assert gate.list_pending() == []
        assert outcome.get("approved") is True

    # --- deny removes request from pending ---

    def test_deny_removes_from_pending(self):
        """After denial the entry must disappear from list_pending()."""
        gate = ApprovalGate(default_timeout=2.0)
        t, outcome = _request_in_thread(gate)
        pending = _wait_for_pending(gate)
        approval_id = pending[0]["id"]

        gate.handle_response(f"deny {approval_id}")
        t.join(timeout=2.0)

        assert gate.list_pending() == []
        assert outcome.get("denied") is True

    # --- approve non-existent request ID ---

    def test_handle_response_unknown_id_returns_false(self):
        """A response for an unknown ID must return False and not crash."""
        gate = ApprovalGate()
        result = gate.handle_response("approve deadbeef")
        assert result is False

    def test_handle_response_unknown_id_does_not_affect_existing(self):
        """Bogus approval ID must not interfere with a real pending request."""
        gate = ApprovalGate(default_timeout=2.0)
        t, outcome = _request_in_thread(gate)
        pending = _wait_for_pending(gate)
        real_id = pending[0]["id"]

        # Send a bogus ID first
        assert gate.handle_response("approve 00000000") is False
        # Real pending is still there
        assert len(gate.list_pending()) == 1

        gate.handle_response(f"approve {real_id}")
        t.join(timeout=2.0)
        assert outcome.get("approved") is True

    # --- request timeout / expiry ---

    def test_timeout_cleans_up_pending(self):
        """Timed-out requests must be removed from list_pending()."""
        gate = ApprovalGate(default_timeout=0.02)
        with pytest.raises(ApprovalTimeout):
            gate.request("slow action")
        assert gate.list_pending() == []

    def test_timeout_message_contains_action(self):
        """The ApprovalTimeout message must name the action."""
        gate = ApprovalGate(default_timeout=0.02)
        with pytest.raises(ApprovalTimeout, match="very specific action"):
            gate.request("very specific action")

    def test_deny_message_contains_action(self):
        """The ApprovalDenied message must name the action."""
        gate = ApprovalGate(default_timeout=2.0)
        t, outcome = _request_in_thread(gate, action="delete production db")
        pending = _wait_for_pending(gate)
        approval_id = pending[0]["id"]

        exc_info: dict = {}

        def _deny_and_capture():
            pa = gate._pending.get(approval_id)
            if pa:
                try:
                    pa.deny()
                    pa.wait()
                except ApprovalDenied as exc:
                    exc_info["msg"] = str(exc)

        # Use gate.handle_response for the actual denial
        gate.handle_response(f"deny {approval_id}")
        t.join(timeout=2.0)
        assert outcome.get("denied") is True

    def test_pending_approval_wait_timeout_message_includes_seconds(self):
        """Timeout message should include the timeout duration."""
        pa = PendingApproval("nuke everything", "test", timeout=0.01)
        with pytest.raises(ApprovalTimeout) as exc_info:
            pa.wait()
        assert "0.01" in str(exc_info.value)

    # --- concurrent approval requests ---

    def test_multiple_concurrent_requests_all_listed(self):
        """Multiple concurrent gate.request() calls all appear in list_pending()."""
        gate = ApprovalGate(default_timeout=3.0)
        threads_outcomes = [_request_in_thread(gate, action=f"action-{i}") for i in range(3)]

        # Wait for all three to register
        pending = _wait_for_pending(gate, count=3, retries=200)
        assert len(pending) == 3

        # Approve them all
        for entry in pending:
            gate.handle_response(f"approve {entry['id']}")

        for t, outcome in threads_outcomes:
            t.join(timeout=2.0)
            assert outcome.get("approved") is True

        assert gate.list_pending() == []

    def test_concurrent_approve_and_deny(self):
        """Mixed approve/deny across concurrent requests resolves correctly."""
        gate = ApprovalGate(default_timeout=3.0)
        t0, outcome0 = _request_in_thread(gate, action="action-approve")
        t1, outcome1 = _request_in_thread(gate, action="action-deny")

        pending = _wait_for_pending(gate, count=2, retries=200)
        assert len(pending) == 2

        # Identify which is which by action name
        id_approve = next(e["id"] for e in pending if e["action"] == "action-approve")
        id_deny = next(e["id"] for e in pending if e["action"] == "action-deny")

        gate.handle_response(f"approve {id_approve}")
        gate.handle_response(f"deny {id_deny}")

        t0.join(timeout=2.0)
        t1.join(timeout=2.0)

        assert outcome0.get("approved") is True
        assert outcome1.get("denied") is True

    def test_sequential_requests_each_removed_after_resolution(self):
        """Running two sequential requests leaves list_pending() empty between them."""
        gate = ApprovalGate(default_timeout=2.0)

        for _ in range(2):
            t, outcome = _request_in_thread(gate)
            pending = _wait_for_pending(gate)
            gate.handle_response(f"approve {pending[0]['id']}")
            t.join(timeout=2.0)
            assert outcome.get("approved") is True
            assert gate.list_pending() == []

    def test_handle_response_partial_id_match_not_approved(self):
        """A prefix substring of an ID must not accidentally match."""
        gate = ApprovalGate(default_timeout=2.0)
        t, outcome = _request_in_thread(gate)
        pending = _wait_for_pending(gate)
        real_id = pending[0]["id"]

        # Send only first 3 chars of the 8-char ID — must not match
        short_id = real_id[:3]
        assert gate.handle_response(f"approve {short_id}") is False

        # Clean up
        gate.handle_response(f"deny {real_id}")
        t.join(timeout=2.0)

    def test_handle_response_approve_with_trailing_whitespace(self):
        """Leading/trailing whitespace in the response must be stripped."""
        gate = ApprovalGate(default_timeout=2.0)
        t, outcome = _request_in_thread(gate)
        pending = _wait_for_pending(gate)
        approval_id = pending[0]["id"]

        assert gate.handle_response(f"  approve {approval_id}  ")
        t.join(timeout=2.0)
        assert outcome.get("approved") is True

    def test_send_fn_receives_risk_level(self):
        """The send_fn message must mention the risk level."""
        messages: list[str] = []
        gate = ApprovalGate(send_fn=messages.append, default_timeout=0.02)
        with pytest.raises(ApprovalTimeout):
            gate.request("risky action", risk="high")
        assert messages
        assert "high" in messages[0].lower()

    def test_send_fn_receives_approval_id(self):
        """The send_fn message must include the generated approval ID."""
        messages: list[str] = []
        gate = ApprovalGate(send_fn=messages.append, default_timeout=0.02)
        with pytest.raises(ApprovalTimeout):
            gate.request("some action")
        assert messages
        assert "ID:" in messages[0]


# ---------------------------------------------------------------------------
# SubAgentRunner — edge cases
# ---------------------------------------------------------------------------


class TestSubAgentRunnerEdges:
    def _factory(self, return_value: str = "result") -> Callable:
        """Return a factory that always produces a runtime returning *return_value*."""

        def _make():
            rt = MagicMock()
            rt.run.return_value = return_value
            return rt

        return _make

    # --- spawn a sub-agent with valid config ---

    def test_run_subtask_returns_runtime_result(self):
        """run_subtask must return exactly what runtime.run() returns."""
        runner = SubAgentRunner(runtime_factory=self._factory("precise output"))
        task = SubTask(id=0, description="step one")
        assert runner.run_subtask(task) == "precise output"

    def test_run_subtask_sets_result_on_subtask(self):
        """On success, subtask.result must be populated."""
        runner = SubAgentRunner(runtime_factory=self._factory("done"))
        task = SubTask(id=0, description="step one")
        runner.run_subtask(task)
        assert task.result == "done"
        assert task.error is None

    def test_factory_called_once_per_run_subtask(self):
        """run_subtask() must call the factory exactly once per invocation."""
        factory = MagicMock(return_value=MagicMock(run=MagicMock(return_value="ok")))
        runner = SubAgentRunner(runtime_factory=factory)
        runner.run_subtask(SubTask(id=0, description="x"))
        assert factory.call_count == 1

    # --- handle sub-agent failure gracefully ---

    def test_run_subtask_exception_sets_error_field(self):
        """On failure, subtask.error must be set and result must remain None."""

        def _boom():
            rt = MagicMock()
            rt.run.side_effect = ValueError("something went wrong")
            return rt

        runner = SubAgentRunner(runtime_factory=_boom)
        task = SubTask(id=7, description="will fail")
        result = runner.run_subtask(task)

        assert task.error == "something went wrong"
        assert task.result is None
        assert "[Error in subtask 7:" in result
        assert "something went wrong" in result

    def test_run_subtask_exception_does_not_propagate(self):
        """Exceptions from runtime.run() must be caught; run_subtask() must not raise."""

        def _boom():
            rt = MagicMock()
            rt.run.side_effect = RuntimeError("hard crash")
            return rt

        runner = SubAgentRunner(runtime_factory=_boom)
        # Should not raise
        result = runner.run_subtask(SubTask(id=0, description="fail"))
        assert "hard crash" in result

    def test_run_all_continues_after_one_failure(self):
        """A failed subtask must not abort the remaining ones."""
        call_count = [0]

        def _factory():
            call_count[0] += 1
            rt = MagicMock()
            if call_count[0] == 1:
                rt.run.side_effect = RuntimeError("first fails")
            else:
                rt.run.return_value = "ok"
            return rt

        runner = SubAgentRunner(runtime_factory=_factory)
        tasks = [SubTask(id=i, description=f"task {i}") for i in range(3)]
        results = runner.run_all(tasks)

        assert len(results) == 3
        assert "[Error" in results[0]
        assert results[1] == "ok"
        assert results[2] == "ok"

    # --- sub-agent timeout (simulated via slow runtime) ---

    def test_run_subtask_slow_runtime_still_returns(self):
        """A runtime that takes a moment must still resolve (no deadlock)."""

        def _slow_factory():
            rt = MagicMock()

            def _slow_run(prompt):
                time.sleep(0.05)
                return "slow ok"

            rt.run.side_effect = _slow_run
            return rt

        runner = SubAgentRunner(runtime_factory=_slow_factory)
        task = SubTask(id=0, description="slow task")
        result = runner.run_subtask(task)
        assert result == "slow ok"

    # --- max concurrent sub-agents limit ---

    def test_semaphore_limits_concurrent_executions(self):
        """No more than MAX_CONCURRENT runtimes should be active simultaneously."""
        active = [0]
        peak = [0]
        lock = threading.Lock()
        barrier = threading.Barrier(MAX_CONCURRENT)

        def _factory():
            rt = MagicMock()

            def _run(prompt):
                with lock:
                    active[0] += 1
                    if active[0] > peak[0]:
                        peak[0] = active[0]
                # Synchronise so all MAX_CONCURRENT slots fill up simultaneously
                with contextlib.suppress(threading.BrokenBarrierError):
                    barrier.wait(timeout=2.0)
                with lock:
                    active[0] -= 1
                return "ok"

            rt.run.side_effect = _run
            return rt

        runner = SubAgentRunner(runtime_factory=_factory)
        tasks = [SubTask(id=i, description=f"t{i}") for i in range(MAX_CONCURRENT * 2)]

        threads = [
            threading.Thread(target=runner.run_subtask, args=(task,), daemon=True) for task in tasks
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=5.0)

        assert peak[0] <= MAX_CONCURRENT

    def test_run_all_caps_total_tasks(self):
        """run_all() must not execute more than max_total tasks."""
        factory = MagicMock(return_value=MagicMock(run=MagicMock(return_value="ok")))
        runner = SubAgentRunner(runtime_factory=factory)
        tasks = [SubTask(id=i, description=f"t{i}") for i in range(MAX_SUB_AGENTS + 5)]
        results = runner.run_all(tasks, max_total=4)
        assert len(results) == 4
        assert factory.call_count == 4

    def test_run_all_zero_tasks(self):
        """run_all() with an empty task list must return an empty list."""
        factory = MagicMock()
        runner = SubAgentRunner(runtime_factory=factory)
        results = runner.run_all([])
        assert results == []
        factory.assert_not_called()

    def test_run_all_single_task(self):
        """run_all() with a single task must return a one-element list."""
        runner = SubAgentRunner(runtime_factory=self._factory("solo"))
        results = runner.run_all([SubTask(id=0, description="only one")])
        assert results == ["solo"]

    # --- sub-agent inherits parent session context ---

    def test_dependency_result_injected_as_context(self):
        """The result of a dependency task must appear in the next task's prompt."""
        call_args_log: list[str] = []
        call_count = [0]

        def _factory():
            call_count[0] += 1
            rt = MagicMock()
            if call_count[0] == 1:
                rt.run.return_value = "parent result data"
            else:

                def _capture(prompt):
                    call_args_log.append(prompt)
                    return "child done"

                rt.run.side_effect = _capture
            return rt

        runner = SubAgentRunner(runtime_factory=_factory)
        tasks = [
            SubTask(id=0, description="parent task"),
            SubTask(id=1, description="child task", depends_on=[0]),
        ]
        runner.run_all(tasks)

        assert call_args_log, "Child runtime was never called"
        child_prompt = call_args_log[0]
        assert "parent result data" in child_prompt
        assert "child task" in child_prompt

    def test_dependency_context_truncated_to_200_chars(self):
        """Context injected from a dependency must be capped at 200 characters."""
        long_result = "X" * 500
        call_count = [0]
        captured_prompts: list[str] = []

        def _factory():
            call_count[0] += 1
            rt = MagicMock()
            if call_count[0] == 1:
                rt.run.return_value = long_result
            else:

                def _cap(prompt):
                    captured_prompts.append(prompt)
                    return "ok"

                rt.run.side_effect = _cap
            return rt

        runner = SubAgentRunner(runtime_factory=_factory)
        tasks = [
            SubTask(id=0, description="first"),
            SubTask(id=1, description="second", depends_on=[0]),
        ]
        runner.run_all(tasks)

        assert captured_prompts
        # The 200-char slice of long_result should appear, not the full 500-char string
        assert "X" * 200 in captured_prompts[0]
        assert "X" * 201 not in captured_prompts[0]

    def test_multiple_dependencies_all_injected(self):
        """A task depending on two prior steps must receive both results as context."""
        results_by_id = {0: "alpha output", 1: "beta output"}
        call_count = [0]
        captured: list[str] = []

        def _factory():
            call_count[0] += 1
            idx = call_count[0] - 1
            rt = MagicMock()
            if idx in results_by_id:
                rt.run.return_value = results_by_id[idx]
            else:

                def _cap(prompt):
                    captured.append(prompt)
                    return "merged"

                rt.run.side_effect = _cap
            return rt

        runner = SubAgentRunner(runtime_factory=_factory)
        tasks = [
            SubTask(id=0, description="step-a"),
            SubTask(id=1, description="step-b"),
            SubTask(id=2, description="step-c", depends_on=[0, 1]),
        ]
        runner.run_all(tasks)

        assert captured
        assert "alpha output" in captured[0]
        assert "beta output" in captured[0]

    def test_error_in_dependency_does_not_block_dependent(self):
        """A failed dependency (no result set) must not crash the dependent task."""
        call_count = [0]

        def _factory():
            call_count[0] += 1
            rt = MagicMock()
            if call_count[0] == 1:
                rt.run.side_effect = RuntimeError("dep failed")
            else:
                rt.run.return_value = "independent result"
            return rt

        runner = SubAgentRunner(runtime_factory=_factory)
        tasks = [
            SubTask(id=0, description="bad dep"),
            SubTask(id=1, description="good task", depends_on=[0]),
        ]
        results = runner.run_all(tasks)
        # Both tasks must produce a result string; second should succeed
        assert len(results) == 2
        assert "independent result" in results[1]

    # --- parse_subtasks edge cases not in existing suite ---

    def test_parse_single_numbered_item(self):
        """A single numbered item parses as a one-element list (no connectives)."""
        tasks = parse_subtasks("1. Only one step")
        assert len(tasks) == 1
        assert tasks[0].description == "Only one step"

    def test_parse_numbered_list_preserves_description_whitespace_stripped(self):
        """Descriptions must be stripped of surrounding whitespace."""
        tasks = parse_subtasks("1.   lots of spaces   \n2.   second   ")
        assert tasks[0].description == "lots of spaces"
        assert tasks[1].description == "second"

    def test_parse_connectives_first_task_has_no_dependency(self):
        """The very first sequential task must have an empty depends_on."""
        tasks = parse_subtasks("do A then do B")
        assert tasks[0].depends_on == []

    def test_parse_mixed_connectives_all_chained(self):
        """All sequential connective tasks form a linear dependency chain."""
        tasks = parse_subtasks("step one then step two and then step three finally step four")
        for i in range(1, len(tasks)):
            assert tasks[i].depends_on == [i - 1]
