"""Tests for missy.agent.sub_agent — sub-agent delegation."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.runtime import AgentConfig, AgentRuntime
from missy.agent.sub_agent import (
    MAX_CONCURRENT,
    MAX_SUB_AGENT_DEPTH,
    MAX_SUB_AGENTS,
    DelegationPlanError,
    SubAgentRunner,
    SubTask,
    build_diverse_subtasks,
    parse_subtasks,
    validate_subtasks,
)
from missy.providers.base import CompletionResponse


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


class TestDiversePlans:
    def test_builds_complementary_specialists_and_synthesis(self):
        tasks = build_diverse_subtasks("Design a resilient migration plan")

        assert [task.name for task in tasks] == [
            "Requirements Analyst",
            "Solution Architect",
            "Risk Critic",
            "Validation Specialist",
            "Lead Synthesizer",
        ]
        assert all(not task.depends_on for task in tasks[:-1])
        assert tasks[-1].depends_on == [0, 1, 2, 3]
        assert tasks[-1].is_synthesis is True
        assert len({task.focus for task in tasks[:-1]}) == 4

    def test_can_omit_synthesis(self):
        tasks = build_diverse_subtasks("Compare two approaches", include_synthesis=False)
        assert len(tasks) == 4
        assert not any(task.is_synthesis for task in tasks)

    def test_specialist_mandates_are_injected_into_runtime_prompts(self):
        prompts: list[str] = []

        def _run(prompt, session_id="", _delegation_depth=0):
            prompts.append(prompt)
            return "finding"

        runtime = MagicMock()
        runtime.run.side_effect = _run
        runner = SubAgentRunner(runtime=runtime, session_id="s", depth=1)
        runner.run_all(build_diverse_subtasks("Solve the advanced request"))

        specialist_prompts = [prompt for prompt in prompts if "Lead Synthesizer" not in prompt]
        assert len(specialist_prompts) == 4
        assert any("Requirements Analyst" in prompt for prompt in specialist_prompts)
        assert any("Risk Critic" in prompt for prompt in specialist_prompts)
        assert all("Success criteria:" in prompt for prompt in specialist_prompts)
        synthesis_prompt = next(prompt for prompt in prompts if "Lead Synthesizer" in prompt)
        assert all(f"Result of step {index}: finding" in synthesis_prompt for index in range(4))


class TestDelegationPlanValidation:
    def test_rejects_missing_dependency(self):
        with pytest.raises(DelegationPlanError, match="missing dependencies"):
            validate_subtasks([SubTask(id=0, description="bad", depends_on=[9])])

    def test_rejects_self_dependency(self):
        with pytest.raises(DelegationPlanError, match="depend on itself"):
            validate_subtasks([SubTask(id=0, description="bad", depends_on=[0])])

    def test_rejects_dependency_cycle(self):
        tasks = [
            SubTask(id=0, description="a", depends_on=[1]),
            SubTask(id=1, description="b", depends_on=[0]),
        ]
        with pytest.raises(DelegationPlanError, match="dependency cycle"):
            validate_subtasks(tasks)

    def test_runner_validates_before_executing(self):
        runtime = MagicMock()
        runner = SubAgentRunner(runtime=runtime, session_id="s", depth=1)

        with pytest.raises(DelegationPlanError):
            runner.run_all([SubTask(id=0, description="bad", depends_on=[4])])
        runtime.run.assert_not_called()


class TestFailurePolicies:
    @staticmethod
    def _failing_runtime():
        runtime = MagicMock()

        def _run(prompt, session_id="", _delegation_depth=0):
            if "upstream" in prompt:
                raise RuntimeError("upstream failed")
            return "ok"

        runtime.run.side_effect = _run
        return runtime

    def test_skip_dependents_prevents_unsafe_downstream_execution(self):
        runtime = self._failing_runtime()
        runner = SubAgentRunner(
            runtime=runtime,
            session_id="s",
            depth=1,
            failure_policy="skip_dependents",
        )
        tasks = [
            SubTask(id=0, description="upstream"),
            SubTask(id=1, description="destructive downstream", depends_on=[0]),
        ]

        results = runner.run_all(tasks)

        assert runtime.run.call_count == 1
        assert tasks[1].status == "skipped"
        assert "dependencies failed" in results[1]

    def test_fail_fast_skips_every_remaining_wave(self):
        runtime = self._failing_runtime()
        runner = SubAgentRunner(
            runtime=runtime,
            session_id="s",
            depth=1,
            failure_policy="fail_fast",
        )
        tasks = [
            SubTask(id=0, description="upstream"),
            SubTask(id=1, description="second", depends_on=[0]),
            SubTask(id=2, description="third", depends_on=[1]),
        ]

        runner.run_all(tasks)

        assert runtime.run.call_count == 1
        assert [task.status for task in tasks] == ["error", "skipped", "skipped"]

    def test_invalid_failure_policy_is_rejected(self):
        with pytest.raises(ValueError, match="failure_policy"):
            SubAgentRunner(
                runtime=MagicMock(),
                session_id="s",
                depth=1,
                failure_policy="invented",
            )


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

    def test_dependent_task_context_surfaces_failed_dependency(self):
        """Regression: a dependency's .result is only set on success (a
        failure sets .error instead), and the prior context builder's
        "if ... .result" filter silently omitted failed dependencies from
        context entirely -- not even an error placeholder. A dependent
        step then ran with no indication its dependency had failed,
        potentially taking action based on a false assumption that
        upstream work completed.
        """

        def _run(prompt, session_id="", _delegation_depth=0):
            if "first" in prompt:
                raise RuntimeError("could not find the file")
            return "ok"

        runtime = MagicMock()
        runtime.run.side_effect = _run
        runner = SubAgentRunner(runtime=runtime, session_id="sess-1", depth=0)
        tasks = [
            SubTask(id=0, description="first: search for file"),
            SubTask(id=1, description="second: delete the file found", depends_on=[0]),
        ]
        runner.run_all(tasks)

        assert tasks[0].error is not None
        assert tasks[0].result is None

        second_call_prompt = runtime.run.call_args_list[1][0][0]
        assert "FAILED" in second_call_prompt
        assert "could not find the file" in second_call_prompt

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

    def test_each_subtask_can_select_a_different_provider(self):
        runtime = self._mock_runtime("ok")
        runner = SubAgentRunner(runtime=runtime, session_id="sess-1", depth=0)
        tasks = [
            SubTask(id=0, description="delegate", provider="acpx"),
            SubTask(id=1, description="analyze", provider="openai"),
        ]

        runner.run_all(tasks)

        providers = {call.kwargs["_provider"] for call in runtime.run.call_args_list}
        assert providers == {"acpx", "openai"}

    def test_unassigned_subtask_inherits_parent_provider(self):
        runtime = self._mock_runtime("ok")
        runner = SubAgentRunner(
            runtime=runtime,
            session_id="sess-1",
            depth=1,
            default_provider="acpx",
        )

        runner.run_all([SubTask(id=0, description="nested")])

        assert runtime.run.call_args.kwargs["_provider"] == "acpx"


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

    def test_shared_runtime_dispatches_concurrent_agents_to_distinct_providers(self):
        providers = {}
        for name in ("parent", "acpx", "openai"):
            provider = MagicMock()
            provider.name = name
            provider.is_available.return_value = True
            provider.complete.return_value = CompletionResponse(
                content=f"{name} reply",
                model=f"{name}-model",
                provider=name,
                usage={},
                raw={},
            )
            providers[name] = provider

        registry = MagicMock()
        registry.get.side_effect = providers.get
        registry.get_config.return_value = None
        registry.key_for.side_effect = lambda candidate: next(
            (name for name, provider in providers.items() if provider is candidate), None
        )
        registry.list_providers.return_value = sorted(providers)

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError("no tools")),
            patch("missy.agent.runtime.get_message_bus", side_effect=RuntimeError("no bus")),
        ):
            runtime = AgentRuntime(AgentConfig(provider="parent"))
            runtime._memory_store = None
            runtime._request_tracker = None
            runner = SubAgentRunner(runtime=runtime, session_id="sess-1", depth=1)
            results = runner.run_all(
                [
                    SubTask(id=0, description="delegate", provider="acpx"),
                    SubTask(id=1, description="review", provider="openai"),
                ]
            )

        assert results == ["acpx reply", "openai reply"]
        providers["acpx"].complete.assert_called_once()
        providers["openai"].complete.assert_called_once()
        providers["parent"].complete.assert_not_called()
        assert runtime.config.provider == "parent"

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
