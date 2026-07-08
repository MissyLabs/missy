"""Tests for `missy tools benchmark run` CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _make_mock_tool(name: str = "calculator") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = "Test tool"
    tool.get_schema.return_value = {
        "name": name,
        "description": "Test tool",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "math expr"},
            },
            "required": ["expression"],
        },
    }
    return tool


# ---------------------------------------------------------------------------
# _build_suite_from_tool helper
# ---------------------------------------------------------------------------


def test_build_suite_from_examples() -> None:
    from missy.cli.main import _build_suite_from_tool

    tool = MagicMock()
    tool.name = "calculator"
    tool.get_schema.return_value = {
        "name": "calculator",
        "description": "Calc",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "examples": [
            {"input": {"expression": "2+2"}, "expected_output": "4"},
            {"input": {"expression": "10*5"}, "expected_output": "50"},
        ],
    }
    suite = _build_suite_from_tool(tool, "direct")
    assert suite.task_count() == 2
    assert suite.tool_name == "calculator"


def test_build_suite_fallback_to_smoke_task() -> None:
    from missy.cli.main import _build_suite_from_tool

    tool = _make_mock_tool("echo_tool")
    # No examples — falls back to smoke task built from required params
    suite = _build_suite_from_tool(tool, "direct")
    assert suite.task_count() == 1
    task = suite.tasks[0]
    assert task.tool_name == "echo_tool"
    # Required param 'expression' should be in input_args with a string default
    assert "expression" in task.input_args
    assert isinstance(task.input_args["expression"], str)


def test_build_suite_no_required_params_gets_empty_smoke_task() -> None:
    from missy.cli.main import _build_suite_from_tool

    tool = MagicMock()
    tool.name = "no_args"
    tool.get_schema.return_value = {
        "name": "no_args",
        "description": "No args",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    suite = _build_suite_from_tool(tool, "direct")
    assert suite.task_count() == 1
    assert suite.tasks[0].input_args == {}


def test_build_suite_malformed_examples_skipped() -> None:
    from missy.cli.main import _build_suite_from_tool

    tool = MagicMock()
    tool.name = "t"
    tool.get_schema.return_value = {
        "name": "t",
        "description": "T",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "examples": [
            "not a dict",  # malformed — skip
            {"no_input": True},  # missing 'input' key — skip
        ],
    }
    suite = _build_suite_from_tool(tool, "direct")
    # Falls back to smoke task since no valid examples
    assert suite.task_count() == 1


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


def test_benchmark_run_tool_not_found() -> None:
    from missy.cli.main import cli

    runner = CliRunner()
    with patch("missy.tools.registry.get_tool_registry") as mock_reg:
        mock_reg.return_value.get.return_value = None
        result = runner.invoke(cli, ["tools", "benchmark", "run", "nonexistent_tool"])

    assert result.exit_code != 0
    assert "not registered" in result.output


def test_benchmark_run_success() -> None:
    from missy.cli.main import cli
    from missy.tools.benchmark.runner import SuiteRunReport
    from missy.tools.benchmark.scoring import BenchmarkResult, ScoredResult

    runner_cli = CliRunner()
    mock_tool = _make_mock_tool("calculator")

    scored = ScoredResult(
        result=BenchmarkResult(
            task_id="t1",
            tool_name="calculator",
            provider="direct",
            success=True,
            latency_ms=12.5,
            cost_usd=0.0,
            actual_output="4",
            expected_output="4",
            tool_call_made=True,
            tool_call_args={"expression": "2+2"},
            schema_required_params=["expression"],
            safety_violation=False,
            error="",
        ),
        composite=0.9,
        correctness=1.0,
        latency_score=0.9,
        cost_score=1.0,
        reliability=1.0,
        safety=1.0,
        schema_score=1.0,
        tool_call_quality=1.0,
    )

    mock_report = SuiteRunReport(
        suite_name="calculator_auto_direct",
        tool_name="calculator",
        provider="direct",
        scored_results=[scored],
        aggregate={
            "composite": 0.9,
            "correctness": 1.0,
            "reliability": 1.0,
            "latency_ms": 12.5,
        },
        started_at="2026-07-07T00:00:00+00:00",
        finished_at="2026-07-07T00:00:01+00:00",
        error_count=0,
    )

    with (
        patch("missy.tools.registry.get_tool_registry") as mock_reg,
        patch("missy.tools.benchmark.runner.BenchmarkRunner.run_suite", return_value=mock_report),
        patch("missy.tools.benchmark.benchmark_store.BenchmarkStore.save_batch"),
    ):
        mock_reg.return_value.get.return_value = mock_tool
        result = runner_cli.invoke(cli, ["tools", "benchmark", "run", "calculator", "--no-persist"])

    assert result.exit_code == 0
    assert "calculator" in result.output
    assert "Aggregate" in result.output


def test_benchmark_run_shows_failure() -> None:
    from missy.cli.main import cli
    from missy.tools.benchmark.runner import SuiteRunReport
    from missy.tools.benchmark.scoring import BenchmarkResult, ScoredResult

    runner_cli = CliRunner()
    mock_tool = _make_mock_tool("failing_tool")
    mock_tool.get_schema.return_value = {
        "name": "failing_tool",
        "description": "Always fails",
        "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": []},
    }

    scored = ScoredResult(
        result=BenchmarkResult(
            task_id="t2",
            tool_name="failing_tool",
            provider="direct",
            success=False,
            latency_ms=5.0,
            cost_usd=0.0,
            actual_output=None,
            expected_output=None,
            tool_call_made=True,
            tool_call_args={},
            schema_required_params=[],
            safety_violation=False,
            error="something went wrong",
        ),
        composite=0.1,
        correctness=0.0,
        latency_score=0.9,
        cost_score=1.0,
        reliability=0.0,
        safety=1.0,
        schema_score=1.0,
        tool_call_quality=0.0,
    )

    mock_report = SuiteRunReport(
        suite_name="failing_tool_auto_direct",
        tool_name="failing_tool",
        provider="direct",
        scored_results=[scored],
        aggregate={"composite": 0.1, "correctness": 0.0, "reliability": 0.0, "latency_ms": 5.0},
        started_at="2026-07-07T00:00:00+00:00",
        finished_at="2026-07-07T00:00:01+00:00",
        error_count=1,
    )

    with (
        patch("missy.tools.registry.get_tool_registry") as mock_reg,
        patch("missy.tools.benchmark.runner.BenchmarkRunner.run_suite", return_value=mock_report),
    ):
        mock_reg.return_value.get.return_value = mock_tool
        result = runner_cli.invoke(
            cli, ["tools", "benchmark", "run", "failing_tool", "--no-persist"]
        )

    assert result.exit_code == 0
    assert "0.100" in result.output  # composite shown


def test_benchmark_run_persists_by_default() -> None:
    """Without --no-persist, results should be persisted via the store."""
    from missy.cli.main import cli
    from missy.tools.benchmark.runner import SuiteRunReport

    runner_cli = CliRunner()
    mock_tool = _make_mock_tool("my_tool")
    mock_tool.get_schema.return_value = {
        "name": "my_tool",
        "description": "A tool",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }

    mock_report = SuiteRunReport(
        suite_name="my_tool_auto_direct",
        tool_name="my_tool",
        provider="direct",
        scored_results=[],
        aggregate={"composite": 0.5, "correctness": 0.5, "reliability": 0.5, "latency_ms": 10.0},
        started_at="2026-07-07T00:00:00+00:00",
        finished_at="2026-07-07T00:00:01+00:00",
        error_count=0,
    )

    with (
        patch("missy.tools.registry.get_tool_registry") as mock_reg,
        patch(
            "missy.tools.benchmark.runner.BenchmarkRunner.run_suite", return_value=mock_report
        ) as mock_run,
    ):
        mock_reg.return_value.get.return_value = mock_tool
        result = runner_cli.invoke(cli, ["tools", "benchmark", "run", "my_tool"])

    assert result.exit_code == 0
    # run_suite called with persist=True (default)
    _, kwargs = mock_run.call_args
    assert kwargs.get("persist", True) is True
