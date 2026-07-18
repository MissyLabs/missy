"""Tests for the `missy tools trust` CLI (F11)."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from missy.cli.main import cli
from missy.security.trust import TrustScorer


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


def _seed(path: Path) -> None:
    scorer = TrustScorer(persist_path=path)
    scorer.record_success("calculator")  # 510 (ok)
    for _ in range(5):
        scorer.record_violation("shell_exec")  # floored low (LOW TRUST)


def _invoke(args, trust_path: Path):
    # The CLI builds TrustScorer(persist_path=DEFAULT_TRUST_PATH); patch the
    # constant so it reads our temp file.
    with (
        patch("missy.cli.main._load_subsystems", return_value=object()),
        patch("missy.security.trust.DEFAULT_TRUST_PATH", str(trust_path)),
    ):
        return _runner().invoke(cli, args)


class TestToolsTrust:
    def test_lists_scores_and_flags_low_trust(self, tmp_path: Path) -> None:
        p = tmp_path / "trust.json"
        _seed(p)
        result = _invoke(["tools", "trust"], p)
        out = _combined(result)
        assert result.exit_code == 0, out
        assert "calculator" in out
        assert "shell_exec" in out
        assert "LOW TRUST" in out

    def test_named_entity_shows_single_score(self, tmp_path: Path) -> None:
        p = tmp_path / "trust.json"
        _seed(p)
        result = _invoke(["tools", "trust", "calculator"], p)
        out = _combined(result)
        assert result.exit_code == 0, out
        assert "calculator" in out
        assert "510" in out

    def test_unseen_entity_reports_default(self, tmp_path: Path) -> None:
        p = tmp_path / "trust.json"
        _seed(p)
        result = _invoke(["tools", "trust", "never_scored_tool"], p)
        out = _combined(result)
        assert "500" in out
        assert "never scored" in out.lower()

    def test_empty_store_message(self, tmp_path: Path) -> None:
        p = tmp_path / "trust.json"  # not created
        result = _invoke(["tools", "trust"], p)
        assert "No trust scores recorded" in _combined(result)

    def test_threshold_option_changes_flagging(self, tmp_path: Path) -> None:
        p = tmp_path / "trust.json"
        scorer = TrustScorer(persist_path=p)
        scorer.record_success("midtool")  # 510
        # With a very high threshold, even 510 is "low trust".
        result = _invoke(["tools", "trust", "--threshold", "900"], p)
        assert "LOW TRUST" in _combined(result)
