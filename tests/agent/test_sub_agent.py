"""Tests for missy.agent.sub_agent — sub-agent delegation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from missy.agent.sub_agent import (
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
    def _mock_runtime(self, return_value="result"):
        runtime = MagicMock()
        runtime.run.return_value = return_value
        return runtime

    def test_run_subtask_basic(self):
        runtime = self._mock_runtime("done")
        runner = SubAgentRunner(runtime_factory=lambda: runtime)
        task = SubTask(id=0, description="do it")
        result = runner.run_subtask(task)
        assert result == "done"
        assert task.result == "done"
        runtime.run.assert_called_once_with("do it")

    def test_run_subtask_with_context(self):
        runtime = self._mock_runtime("done")
        runner = SubAgentRunner(runtime_factory=lambda: runtime)
        task = SubTask(id=0, description="do it")
        runner.run_subtask(task, context="previous context")
        call_arg = runtime.run.call_args[0][0]
        assert "Context: previous context" in call_arg
        assert "Task: do it" in call_arg

    def test_run_subtask_error(self):
        runtime = MagicMock()
        runtime.run.side_effect = RuntimeError("boom")
        runner = SubAgentRunner(runtime_factory=lambda: runtime)
        task = SubTask(id=0, description="fail")
        result = runner.run_subtask(task)
        assert "[Error" in result
        assert "boom" in result
        assert task.error == "boom"

    def test_run_all_simple(self):
        runtime = self._mock_runtime("ok")
        runner = SubAgentRunner(runtime_factory=lambda: runtime)
        tasks = [SubTask(id=0, description="a"), SubTask(id=1, description="b")]
        results = runner.run_all(tasks)
        assert len(results) == 2
        assert results == ["ok", "ok"]

    def test_run_all_with_dependencies(self):
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            rt = MagicMock()
            rt.run.return_value = f"result_{call_count}"
            return rt

        runner = SubAgentRunner(runtime_factory=factory)
        tasks = [
            SubTask(id=0, description="first"),
            SubTask(id=1, description="second", depends_on=[0]),
        ]
        results = runner.run_all(tasks)
        assert len(results) == 2
        # Second task should have received context from first
        assert tasks[0].result == "result_1"

    def test_run_all_caps_at_max_total(self):
        runtime = self._mock_runtime("ok")
        runner = SubAgentRunner(runtime_factory=lambda: runtime)
        tasks = [SubTask(id=i, description=f"task {i}") for i in range(20)]
        results = runner.run_all(tasks, max_total=3)
        assert len(results) == 3

    def test_run_all_default_max(self):
        runtime = self._mock_runtime("ok")
        runner = SubAgentRunner(runtime_factory=lambda: runtime)
        tasks = [SubTask(id=i, description=f"task {i}") for i in range(MAX_SUB_AGENTS + 5)]
        results = runner.run_all(tasks)
        assert len(results) == MAX_SUB_AGENTS

    def test_factory_called_per_subtask(self):
        factory = MagicMock()
        rt = MagicMock()
        rt.run.return_value = "ok"
        factory.return_value = rt
        runner = SubAgentRunner(runtime_factory=factory)
        tasks = [SubTask(id=0, description="a"), SubTask(id=1, description="b")]
        runner.run_all(tasks)
        assert factory.call_count == 2
