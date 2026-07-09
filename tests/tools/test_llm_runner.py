"""Tests for missy.tools.benchmark.llm_runner."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from missy.providers.base import BaseProvider, CompletionResponse, Message, ToolCall
from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.benchmark.benchmark_store import BenchmarkStore
from missy.tools.benchmark.llm_runner import LLMBenchmarkRunner, LLMBenchmarkTask
from missy.tools.benchmark.mock_provider import MockToolProvider
from missy.tools.builtin.calculator import CalculatorTool
from missy.tools.registry import ToolRegistry


class _StubProvider(BaseProvider):
    """Provider double whose tool-call behavior is set per-test."""

    name = "stub"

    def __init__(self, tool_calls: list[ToolCall] | None = None, raises: Exception | None = None):
        self._tool_calls = tool_calls or []
        self._raises = raises

    def is_available(self) -> bool:
        return True

    def complete(self, messages, **kwargs) -> CompletionResponse:
        return CompletionResponse(content="", model="stub-1", provider=self.name, usage={}, raw={})

    def complete_with_tools(self, messages, tools, system: str = "") -> CompletionResponse:
        if self._raises:
            raise self._raises
        return CompletionResponse(
            content="",
            model="stub-1",
            provider=self.name,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            raw={},
            tool_calls=self._tool_calls,
            finish_reason="tool_calls" if self._tool_calls else "stop",
        )


class _AlwaysFailTool(BaseTool):
    name = "always_fail"
    description = "A tool that always fails, for execute_tool=True tests."
    permissions = ToolPermissions()

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=False, output=None, error="deliberately failed")

    def get_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": {}, "required": []},
        }


@pytest.fixture
def store(tmp_path: Path) -> BenchmarkStore:
    return BenchmarkStore(db_path=tmp_path / "benchmarks.db")


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(_AlwaysFailTool())
    return reg


class TestLLMBenchmarkTask:
    def test_create_derives_required_params_from_expected_args(self) -> None:
        task = LLMBenchmarkTask.create(
            tool_name="calculator", prompt="p", expected_args={"expression": "2+2"}
        )
        assert task.schema_required_params == ["expression"]
        assert task.id  # non-empty UUID


class TestLLMBenchmarkRunnerToolSelection:
    def test_provider_calls_tool_scores_high(self, store, registry) -> None:
        provider = MockToolProvider()
        runner = LLMBenchmarkRunner(provider=provider, store=store)
        task = LLMBenchmarkTask.create(tool_name="calculator", prompt="compute '2 + 2'")
        scored = runner.run_task(task, CalculatorTool(), registry=registry)
        assert scored.result.tool_call_made is True
        assert scored.tool_call_quality == 1.0
        assert scored.result.provider == "mock"

    def test_provider_ignores_tools_scores_failure(self, store, registry) -> None:
        provider = MockToolProvider(call_tool=False)
        runner = LLMBenchmarkRunner(provider=provider, store=store)
        task = LLMBenchmarkTask.create(tool_name="calculator", prompt="compute '2 + 2'")
        scored = runner.run_task(task, CalculatorTool(), registry=registry)
        assert scored.result.tool_call_made is False
        assert scored.result.success is False
        assert "did not call" in scored.result.error
        assert scored.tool_call_quality == 0.0

    def test_provider_calling_wrong_tool_flags_safety_violation(self, store, registry) -> None:
        wrong_call = ToolCall(id=str(uuid.uuid4()), name="not_calculator", arguments={})
        provider = _StubProvider(tool_calls=[wrong_call])
        runner = LLMBenchmarkRunner(provider=provider, store=store)
        task = LLMBenchmarkTask.create(tool_name="calculator", prompt="compute '2 + 2'")
        scored = runner.run_task(task, CalculatorTool(), registry=registry)
        assert scored.result.tool_call_made is False
        assert scored.result.safety_violation is True
        assert scored.safety == 0.0

    def test_provider_exception_recorded_as_error(self, store, registry) -> None:
        provider = _StubProvider(raises=RuntimeError("network down"))
        runner = LLMBenchmarkRunner(provider=provider, store=store)
        task = LLMBenchmarkTask.create(tool_name="calculator", prompt="compute '2 + 2'")
        scored = runner.run_task(task, CalculatorTool(), registry=registry)
        assert scored.result.success is False
        assert "network down" in scored.result.error


class TestLLMBenchmarkRunnerExecuteFlag:
    def test_default_does_not_execute_tool(self, store, registry) -> None:
        provider = MockToolProvider()
        runner = LLMBenchmarkRunner(provider=provider, store=store, execute_tool=False)
        task = LLMBenchmarkTask.create(
            tool_name="calculator",
            prompt="compute '2 + 2'",
            expected_args={"expression": "2 + 2"},
        )
        scored = runner.run_task(task, CalculatorTool(), registry=registry)
        # Not executed: actual_output reflects the normalised arg mapping,
        # not a real calculator result.
        assert scored.result.actual_output == "expression=2 + 2"
        assert scored.result.metadata["executed"] is False

    def test_execute_true_runs_the_real_tool(self, store, registry) -> None:
        provider = MockToolProvider()
        runner = LLMBenchmarkRunner(provider=provider, store=store, execute_tool=True)
        task = LLMBenchmarkTask.create(
            tool_name="calculator",
            prompt="compute '2 + 2'",
            expected_args={"expression": "2 + 2"},
            expected_output=4,
        )
        scored = runner.run_task(task, CalculatorTool(), registry=registry)
        assert scored.result.actual_output == 4
        assert scored.result.success is True
        assert scored.result.metadata["executed"] is True

    def test_execute_true_surfaces_tool_failure(self, store, registry) -> None:
        stub_call = ToolCall(id=str(uuid.uuid4()), name="always_fail", arguments={})
        provider = _StubProvider(tool_calls=[stub_call])
        runner = LLMBenchmarkRunner(provider=provider, store=store, execute_tool=True)
        task = LLMBenchmarkTask.create(tool_name="always_fail", prompt="please fail")
        scored = runner.run_task(task, _AlwaysFailTool(), registry=registry)
        assert scored.result.success is False
        assert "deliberately failed" in scored.result.error


class TestLLMBenchmarkRunnerPersistence:
    def test_persists_by_default(self, store, registry) -> None:
        provider = MockToolProvider()
        runner = LLMBenchmarkRunner(provider=provider, store=store)
        task = LLMBenchmarkTask.create(tool_name="calculator", prompt="compute '2 + 2'")
        runner.run_task(task, CalculatorTool(), registry=registry)
        assert store.count(tool_name="calculator", provider="mock") == 1

    def test_no_persist_skips_store(self, store, registry) -> None:
        provider = MockToolProvider()
        runner = LLMBenchmarkRunner(provider=provider, store=store)
        task = LLMBenchmarkTask.create(tool_name="calculator", prompt="compute '2 + 2'")
        runner.run_task(task, CalculatorTool(), registry=registry, persist=False)
        assert store.count(tool_name="calculator", provider="mock") == 0


class TestLLMBenchmarkRunnerSuite:
    def test_run_suite_aggregates(self, store, registry) -> None:
        provider = MockToolProvider()
        runner = LLMBenchmarkRunner(provider=provider, store=store)
        tasks = [
            LLMBenchmarkTask.create(tool_name="calculator", prompt="compute '1 + 1'"),
            LLMBenchmarkTask.create(tool_name="calculator", prompt="compute '2 + 2'"),
        ]
        report = runner.run_suite(tasks, CalculatorTool(), registry=registry)
        assert report.provider == "mock"
        assert report.tool_name == "calculator"
        assert len(report.scored_results) == 2
        assert report.error_count == 0
        assert "composite" in report.aggregate

    def test_run_suite_empty_tasks(self, store, registry) -> None:
        provider = MockToolProvider()
        runner = LLMBenchmarkRunner(provider=provider, store=store)
        report = runner.run_suite([], CalculatorTool(), registry=registry)
        assert report.scored_results == []
        assert report.tool_name == ""
