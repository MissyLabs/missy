"""Tests for `missy budget status|reset` (F19)."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from missy.cli.main import cli


def _runner() -> CliRunner:
    sig = inspect.signature(CliRunner.__init__)
    kwargs = {"mix_stderr": False} if "mix_stderr" in sig.parameters else {}
    return CliRunner(**kwargs)


def _combined(result) -> str:
    out = result.output
    try:
        err = result.stderr
    except (ValueError, AttributeError):
        err = ""
    if err and err not in out:
        out += err
    return out


def _run(args, *, max_usd: float, period: str = "total", gb_path: Path):
    cfg = SimpleNamespace(global_max_spend_usd=max_usd, global_budget_period=period)
    with (
        patch("missy.cli.main._load_subsystems", return_value=cfg),
        patch("missy.agent.global_budget.DEFAULT_GLOBAL_BUDGET_PATH", str(gb_path)),
    ):
        return _runner().invoke(cli, args)


class TestBudgetStatus:
    def test_disabled_message(self, tmp_path: Path) -> None:
        r = _run(["budget", "status"], max_usd=0.0, gb_path=tmp_path / "gb.json")
        assert r.exit_code == 0, _combined(r)
        assert "No global budget configured" in _combined(r)

    def test_enabled_shows_totals(self, tmp_path: Path) -> None:
        from missy.agent.global_budget import GlobalBudget

        p = tmp_path / "gb.json"
        GlobalBudget(2.0, path=str(p)).record(0.5)
        r = _run(["budget", "status"], max_usd=2.0, gb_path=p)
        out = _combined(r)
        assert r.exit_code == 0, out
        assert "Global budget" in out
        assert "0.5000" in out  # spent
        assert "1.5000" in out  # remaining


class TestBudgetReset:
    def test_reset_zeroes_total(self, tmp_path: Path) -> None:
        from missy.agent.global_budget import GlobalBudget

        p = tmp_path / "gb.json"
        GlobalBudget(2.0, path=str(p)).record(1.0)
        r = _run(["budget", "reset", "--yes"], max_usd=2.0, gb_path=p)
        assert r.exit_code == 0, _combined(r)
        assert "reset to zero" in _combined(r)
        assert GlobalBudget(2.0, path=str(p)).total_spent() == 0.0
