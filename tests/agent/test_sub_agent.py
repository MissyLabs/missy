"""Tests for missy.agent.sub_agent — sub-agent delegation."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from missy.agent.sub_agent import (
    MAX_CONCURRENT,
    MAX_SUB_AGENT_DEPTH,
    MAX_SUB_AGENTS,
    SubAgentRunner,
    SubTask,
    parse_subtasks,
)


class TestParseSubtasks:
    def test_numbered_list(self):
        prompt = "1. Search the web\n2. Summarise results\n3. Save to file"
        tasks = parse_subtasks(prompt)
        assert len(tasks) == 3
        assert tasks[0].description == "Search the web"
        assert tasks[1].description == "Summarise results"
        assert tasks[2].description == "Save to file"

    def test_numbered_list_parenthesis(self):
        prompt = "1) First task\n2) Second task"
        tasks = parse_subtasks(prompt)
        assert len(tasks) == 2

    def test_sequential_connectives(self):
        prompt = "Search the web then summarise results and then save to file"
        tasks = parse_subtasks(prompt)
        assert len(tasks) == 3
        assert tasks[1].depends_on == [0]
        assert tasks[2].depends_on == [1]
        assert tasks[0].depends_on == []

    def test_after_that_connective(self):
        prompt = "Do step one after that do step two"
        tasks = parse_subtasks(prompt)
        assert len(tasks) == 2
        assert tasks[1].depends_on == [0]

    def test_finally_connective(self):
        prompt = "Prepare data then process it finally report"
        tasks = parse_subtasks(prompt)
        assert len(tasks) >= 2

    def test_single_task_fallback(self):
        prompt = "Just do this simple task"
        tasks = parse_subtasks(prompt)
        assert len(tasks) == 1
        assert tasks[0].id == 0
        assert tasks[0].description == prompt

    def test_empty_prompt(self):
        tasks = parse_subtasks("")
        assert len(tasks) == 1
        assert tasks[0].description == ""

    def test_numbered_list_ids_sequential(self):
        prompt = "1. A\n2. B\n3. C"
        tasks = parse_subtasks(prompt)
        for i, t in enumerate(tasks):
            assert t.id == i

    def test_numbered_no_dependencies(self):
        """Numbered tasks have no auto-dependencies (unlike connectives)."""
        prompt = "1. A\n2. B"
        tasks = parse_subtasks(prompt)
        assert tasks[0].depends_on == []
        assert tasks[1].depends_on == []


class TestSubTask:
    def test_defaults(self):
        t = SubTask(id=0, description="test")
        assert t.tool_hints == []
        assert t.depends_on == []
        assert t.result is None
        assert t.error is None

    def test_with_tool_hints(self):
        t = SubTask(id=1, description="search", tool_hints=["web_fetch"])
        assert t.tool_hints == ["web_fetch"]


class TestSubAgentRunner:
    """SR-4.2: SubAgentRunner now reuses a single shared AgentRuntime and
    session_id (rather than a fresh runtime per subtask), so a sub-agent's
    spend aggregates against the exact same per-session CostTracker the
    parent call is bound by -- no separate cross-child budget-aggregation
    logic is needed, it falls out of reusing the parent's own session-
    scoped accounting. It also genuinely runs independent subtasks in
    parallel now (see TestRealConcurrency below), not sequentially.
    """

    def _mock_runtime(self, return_value="result"):
        runtime = MagicMock()
        runtime.run.return_value = return_value
        return runtime

    def test_run_subtask_basic(self):
        runtime = self._mock_runtime("done")
        runner = SubAgentRunner(runtime=runtime, session_id="sess-1", depth=1)
        task = SubTask(id=0, description="do it")
        result = runner.run_subtask(task)
        assert result == "done"
        assert task.result == "done"
        runtime.run.assert_called_once_with("do it", session_id="sess-1", _delegation_depth=1)

    def test_run_subtask_with_context(self):
        runtime = self._mock_runtime("done")
        runner = SubAgentRunner(runtime=runtime, session_id="sess-1", depth=0)
        task = SubTask(id=0, description="do it")
        runner.run_subtask(task, context="previous context")
        call_arg = runtime.run.call_args[0][0]
        assert "Context: previous context" in call_arg
        assert "Task: do it" in call_arg

    def test_run_subtask_error(self):
        runtime = MagicMock()
        runtime.run.side_effect = RuntimeError("boom")
        runner = SubAgentRunner(runtime=runtime, session_id="sess-1", depth=0)
        task = SubTask(id=0, description="fail")
        result = runner.run_subtask(task)
        assert "[Error" in result
        assert "boom" in result
        assert task.error == "boom"

    def test_run_all_simple(self):
        runtime = self._mock_runtime("ok")
        runner = SubAgentRunner(runtime=runtime, session_id="sess-1", depth=0)
        tasks = [SubTask(id=0, description="a"), SubTask(id=1, description="b")]
        results = runner.run_all(tasks)
        assert len(results) == 2
        assert results == ["ok", "ok"]

    def test_run_all_with_dependencies(self):
        call_count = 0
        lock = threading.Lock()

        def _run(prompt, session_id="", _delegation_depth=0):
            nonlocal call_count
            with lock:
                call_count += 1
                return f"result_{call_count}"

        runtime = MagicMock()
        runtime.run.side_effect = _run
        runner = SubAgentRunner(runtime=runtime, session_id="sess-1", depth=0)
        tasks = [
            SubTask(id=0, description="first"),
            SubTask(id=1, description="second", depends_on=[0]),
        ]
        results = runner.run_all(tasks)
        assert len(results) == 2
        # Second task should have received context from first
        assert tasks[0].result == "result_1"
        # The dependent task's prompt must include the first task's result.
        second_call_prompt = runtime.run.call_args_list[1][0][0]
        assert "result_1" in second_call_prompt

    def test_run_all_caps_at_max_total(self):
        runtime = self._mock_runtime("ok")
        runner = SubAgentRunner(runtime=runtime, session_id="sess-1", depth=0)
        tasks = [SubTask(id=i, description=f"task {i}") for i in range(20)]
        results = runner.run_all(tasks, max_total=3)
        assert len(results) == 3

    def test_run_all_default_max(self):
        runtime = self._mock_runtime("ok")
        runner = SubAgentRunner(runtime=runtime, session_id="sess-1", depth=0)
        tasks = [SubTask(id=i, description=f"task {i}") for i in range(MAX_SUB_AGENTS + 5)]
        results = runner.run_all(tasks)
        assert len(results) == MAX_SUB_AGENTS

    def test_shared_runtime_used_for_every_subtask(self):
        """Unlike the old factory-per-subtask design, every subtask must
        run through the exact same AgentRuntime instance (that's what
        makes budget aggregation and policy consistency work)."""
        runtime = self._mock_runtime("ok")
        runner = SubAgentRunner(runtime=runtime, session_id="sess-1", depth=0)
        tasks = [SubTask(id=0, description="a"), SubTask(id=1, description="b")]
        runner.run_all(tasks)
        assert runtime.run.call_count == 2
        # Every call used the same session_id -- this is what makes spend
        # aggregate against one shared CostTracker.
        for call in runtime.run.call_args_list:
            assert call.kwargs["session_id"] == "sess-1"

    def test_depth_forwarded_to_every_subtask(self):
        runtime = self._mock_runtime("ok")
        runner = SubAgentRunner(runtime=runtime, session_id="sess-1", depth=1)
        tasks = [SubTask(id=0, description="a")]
        runner.run_all(tasks)
        assert runtime.run.call_args.kwargs["_delegation_depth"] == 1


class TestRealConcurrency:
    """SR-4.2: independent subtasks must genuinely overlap in wall-clock
    time, not just carry a MAX_CONCURRENT constant that nothing enforces
    (the pre-fix bug: run_all() looped sequentially despite the semaphore)."""

    def test_independent_subtasks_run_in_parallel(self):
        start_times: list[float] = []
        lock = threading.Lock()

        def _run(prompt, session_id="", _delegation_depth=0):
            with lock:
                start_times.append(time.monotonic())
            time.sleep(0.2)
            return "done"

        runtime = MagicMock()
        runtime.run.side_effect = _run
        runner = SubAgentRunner(runtime=runtime, session_id="sess-1", depth=0)
        tasks = [SubTask(id=i, description=f"task {i}") for i in range(3)]

        t0 = time.monotonic()
        runner.run_all(tasks)
        elapsed = time.monotonic() - t0

        # Sequential would take >= 0.6s; genuine parallelism should finish
        # well under that even accounting for scheduling overhead.
        assert elapsed < 0.5
        # All three calls should have started within a tight window of
        # each other, not staggered ~0.2s apart.
        assert max(start_times) - min(start_times) < 0.15

    def test_concurrency_capped_at_max_concurrent(self):
        """No more than MAX_CONCURRENT subtasks should be in flight at once,
        even when more than MAX_CONCURRENT independent tasks are queued."""
        in_flight = 0
        max_observed = 0
        lock = threading.Lock()

        def _run(prompt, session_id="", _delegation_depth=0):
            nonlocal in_flight, max_observed
            with lock:
                in_flight += 1
                max_observed = max(max_observed, in_flight)
            time.sleep(0.1)
            with lock:
                in_flight -= 1
            return "done"

        runtime = MagicMock()
        runtime.run.side_effect = _run
        runner = SubAgentRunner(runtime=runtime, session_id="sess-1", depth=0)
        tasks = [SubTask(id=i, description=f"task {i}") for i in range(MAX_CONCURRENT + 3)]

        runner.run_all(tasks)

        assert max_observed <= MAX_CONCURRENT

    def test_dependent_waves_do_not_run_concurrently(self):
        """A task with an unmet dependency must not start before its
        dependency's result is available -- only independent tasks within
        the same wave run concurrently."""
        events: list[str] = []
        lock = threading.Lock()

        def _run(prompt, session_id="", _delegation_depth=0):
            with lock:
                events.append("start")
            time.sleep(0.1)
            with lock:
                events.append("end")
            return "ok"

        runtime = MagicMock()
        runtime.run.side_effect = _run
        runner = SubAgentRunner(runtime=runtime, session_id="sess-1", depth=0)
        tasks = [
            SubTask(id=0, description="first"),
            SubTask(id=1, description="second", depends_on=[0]),
        ]
        runner.run_all(tasks)

        # Strictly sequential: start, end, start, end (never two starts
        # before an end, since task 1 depends on task 0).
        assert events == ["start", "end", "start", "end"]


class TestMaxSubAgentDepth:
    def test_constant_is_a_small_positive_bound(self):
        """A sanity check that this hasn't been accidentally set to 0 or
        an unbounded value -- the whole point is a real, small cap."""
        assert 0 < MAX_SUB_AGENT_DEPTH <= 5
