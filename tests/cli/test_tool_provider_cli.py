"""Tests for `missy tools providers *` and `missy tools benchmark run-llm`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from missy.tools.benchmark.benchmark_store import BenchmarkStore
from missy.tools.intelligence.provider_gate import ProviderGateStore

# ---------------------------------------------------------------------------
# tools providers status/enable/disable/clear/recommend
# ---------------------------------------------------------------------------


def _make_mock_tool(name: str = "calculator") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.get_schema.return_value = {
        "name": name,
        "description": "Test tool",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    }
    return tool


def test_providers_status_no_data(tmp_path: Path) -> None:
    from missy.cli.main import cli

    bench_store = BenchmarkStore(db_path=tmp_path / "bench.db")
    overrides = ProviderGateStore(path=tmp_path / "overrides.json")

    runner = CliRunner()
    with (
        patch("missy.tools.benchmark.get_benchmark_store", return_value=bench_store),
        patch("missy.tools.intelligence.get_provider_gate_store", return_value=overrides),
    ):
        result = runner.invoke(cli, ["tools", "providers", "status", "calculator"])

    assert result.exit_code == 0
    assert "No benchmark data or overrides" in result.output


def test_providers_status_shows_override(tmp_path: Path) -> None:
    from missy.cli.main import cli

    bench_store = BenchmarkStore(db_path=tmp_path / "bench.db")
    overrides = ProviderGateStore(path=tmp_path / "overrides.json")
    overrides.set("calculator", "anthropic", False)

    runner = CliRunner()
    with (
        patch("missy.tools.benchmark.get_benchmark_store", return_value=bench_store),
        patch("missy.tools.intelligence.get_provider_gate_store", return_value=overrides),
    ):
        result = runner.invoke(cli, ["tools", "providers", "status", "calculator"])

    assert result.exit_code == 0
    assert "anthropic" in result.output
    assert "override" in result.output


def test_providers_enable_persists(tmp_path: Path) -> None:
    from missy.cli.main import cli

    overrides = ProviderGateStore(path=tmp_path / "overrides.json")
    runner = CliRunner()
    with patch("missy.tools.intelligence.get_provider_gate_store", return_value=overrides):
        result = runner.invoke(cli, ["tools", "providers", "enable", "calculator", "ollama"])

    assert result.exit_code == 0
    assert "Enabled" in result.output
    assert overrides.get("calculator", "ollama") is True


def test_providers_disable_persists(tmp_path: Path) -> None:
    from missy.cli.main import cli

    overrides = ProviderGateStore(path=tmp_path / "overrides.json")
    runner = CliRunner()
    with patch("missy.tools.intelligence.get_provider_gate_store", return_value=overrides):
        result = runner.invoke(cli, ["tools", "providers", "disable", "calculator", "ollama"])

    assert result.exit_code == 0
    assert "Disabled" in result.output
    assert overrides.get("calculator", "ollama") is False


def test_providers_clear_removes_override(tmp_path: Path) -> None:
    from missy.cli.main import cli

    overrides = ProviderGateStore(path=tmp_path / "overrides.json")
    overrides.set("calculator", "ollama", False)
    runner = CliRunner()
    with patch("missy.tools.intelligence.get_provider_gate_store", return_value=overrides):
        result = runner.invoke(cli, ["tools", "providers", "clear", "calculator", "ollama"])

    assert result.exit_code == 0
    assert "Cleared" in result.output
    assert overrides.get("calculator", "ollama") is None


def test_providers_clear_when_nothing_set(tmp_path: Path) -> None:
    from missy.cli.main import cli

    overrides = ProviderGateStore(path=tmp_path / "overrides.json")
    runner = CliRunner()
    with patch("missy.tools.intelligence.get_provider_gate_store", return_value=overrides):
        result = runner.invoke(cli, ["tools", "providers", "clear", "calculator", "ollama"])

    assert result.exit_code == 0
    assert "No override was set" in result.output


def test_providers_recommend_no_data(tmp_path: Path) -> None:
    from missy.cli.main import cli

    bench_store = BenchmarkStore(db_path=tmp_path / "bench.db")
    runner = CliRunner()
    with patch("missy.tools.benchmark.get_benchmark_store", return_value=bench_store):
        result = runner.invoke(cli, ["tools", "providers", "recommend", "calculator"])

    assert result.exit_code == 0
    assert "No benchmark data" in result.output


def test_providers_recommend_picks_best(tmp_path: Path) -> None:
    from missy.cli.main import cli
    from missy.tools.benchmark.scoring import BenchmarkResult, BenchmarkScorer

    bench_store = BenchmarkStore(db_path=tmp_path / "bench.db")
    scorer = BenchmarkScorer()
    for _ in range(3):
        good = BenchmarkResult(
            task_id="t",
            tool_name="calculator",
            provider="anthropic",
            success=True,
            latency_ms=5.0,
            cost_usd=0.0,
            actual_output="4",
            expected_output="4",
            tool_call_made=True,
            tool_call_args={"expression": "2+2"},
            schema_required_params=["expression"],
        )
        bench_store.save(scorer.score(good))
        bad = BenchmarkResult(
            task_id="t",
            tool_name="calculator",
            provider="ollama",
            success=False,
            latency_ms=20_000.0,
            cost_usd=1.0,
            actual_output="wrong",
            expected_output="4",
            tool_call_made=False,
            tool_call_args={},
            schema_required_params=["expression"],
            safety_violation=True,
            error="boom",
        )
        bench_store.save(scorer.score(bad))

    runner = CliRunner()
    with patch("missy.tools.benchmark.get_benchmark_store", return_value=bench_store):
        result = runner.invoke(cli, ["tools", "providers", "recommend", "calculator"])

    assert result.exit_code == 0
    assert "anthropic" in result.output


# ---------------------------------------------------------------------------
# tools benchmark run-llm
# ---------------------------------------------------------------------------


def test_benchmark_run_llm_tool_not_found() -> None:
    from missy.cli.main import cli

    runner = CliRunner()
    with patch("missy.tools.registry.get_tool_registry") as mock_reg:
        mock_reg.return_value.get.return_value = None
        result = runner.invoke(
            cli,
            ["tools", "benchmark", "run-llm", "nonexistent", "--prompt", "do it"],
        )

    assert result.exit_code != 0
    assert "not registered" in result.output


def test_benchmark_run_llm_mock_provider_success() -> None:
    from missy.cli.main import cli

    tool = _make_mock_tool("calculator")
    runner = CliRunner()
    with patch("missy.tools.registry.get_tool_registry") as mock_reg:
        mock_reg.return_value.get.return_value = tool
        result = runner.invoke(
            cli,
            [
                "tools",
                "benchmark",
                "run-llm",
                "calculator",
                "--prompt",
                "please compute '2 + 2'",
                "--no-persist",
            ],
        )

    assert result.exit_code == 0
    assert "LLM Benchmark: calculator (mock)" in result.output
    assert "Tool call made: yes" in result.output


def test_benchmark_run_llm_unconfigured_provider() -> None:
    from missy.cli.main import cli

    tool = _make_mock_tool("calculator")
    runner = CliRunner()
    with (
        patch("missy.tools.registry.get_tool_registry") as mock_reg,
        patch("missy.providers.registry.get_registry") as mock_preg,
    ):
        mock_reg.return_value.get.return_value = tool
        mock_preg.return_value.get.return_value = None
        result = runner.invoke(
            cli,
            [
                "tools",
                "benchmark",
                "run-llm",
                "calculator",
                "--prompt",
                "do it",
                "--provider",
                "anthropic",
            ],
        )

    assert result.exit_code != 0
    assert "not configured" in result.output


def test_benchmark_run_llm_execute_warns() -> None:
    from missy.cli.main import cli

    tool = _make_mock_tool("calculator")
    runner = CliRunner()
    with patch("missy.tools.registry.get_tool_registry") as mock_reg:
        mock_reg.return_value.get.return_value = tool
        mock_reg.return_value.execute.return_value = MagicMock(
            success=True, output=4, error=None
        )
        result = runner.invoke(
            cli,
            [
                "tools",
                "benchmark",
                "run-llm",
                "calculator",
                "--prompt",
                "please compute '2 + 2'",
                "--execute",
                "--no-persist",
            ],
        )

    assert result.exit_code == 0
    assert "Warning" in result.output


def test_benchmark_run_llm_invalid_expect_arg() -> None:
    from missy.cli.main import cli

    tool = _make_mock_tool("calculator")
    runner = CliRunner()
    with patch("missy.tools.registry.get_tool_registry") as mock_reg:
        mock_reg.return_value.get.return_value = tool
        result = runner.invoke(
            cli,
            [
                "tools",
                "benchmark",
                "run-llm",
                "calculator",
                "--prompt",
                "do it",
                "--expect-arg",
                "not-a-kv-pair",
            ],
        )

    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output
