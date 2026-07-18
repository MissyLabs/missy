"""Tests for `missy persona experiment` CLI (F24)."""

from __future__ import annotations

import inspect
from pathlib import Path
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


def _run(args, exp_path: Path):
    # PersonaExperiment() uses DEFAULT_EXPERIMENT_PATH; point it at a temp file.
    with patch("missy.agent.persona_experiment.DEFAULT_EXPERIMENT_PATH", str(exp_path)):
        return _runner().invoke(cli, args)


class TestPersonaExperimentCLI:
    def test_add_list_enable_flow(self, tmp_path: Path, monkeypatch) -> None:
        from missy.agent import persona_experiment as mod
        from missy.agent.persona import PersonaConfig

        monkeypatch.setattr(mod, "_snapshot_current_persona", lambda: PersonaConfig(name="Base"))
        p = tmp_path / "pexp.json"

        r = _run(["persona", "experiment", "add-variant", "A", "--tone", "direct"], p)
        assert r.exit_code == 0, _combined(r)
        assert "variant" in _combined(r).lower()

        _run(["persona", "experiment", "add-variant", "B", "--tone", "playful"], p)
        r = _run(["persona", "experiment", "list"], p)
        out = _combined(r)
        assert "A" in out and "B" in out
        assert "disabled" in out

        r = _run(["persona", "experiment", "enable"], p)
        assert "enabled" in _combined(r).lower()
        r = _run(["persona", "experiment", "list"], p)
        assert "enabled" in _combined(r)

    def test_enable_without_variants_errors(self, tmp_path: Path) -> None:
        r = _run(["persona", "experiment", "enable"], tmp_path / "pexp.json")
        assert "Register at least one variant" in _combined(r)

    def test_assign_and_record_and_results(self, tmp_path: Path, monkeypatch) -> None:
        from missy.agent import persona_experiment as mod
        from missy.agent.persona import PersonaConfig

        monkeypatch.setattr(mod, "_snapshot_current_persona", lambda: PersonaConfig(name="Base"))
        p = tmp_path / "pexp.json"
        _run(["persona", "experiment", "add-variant", "A"], p)
        _run(["persona", "experiment", "add-variant", "B"], p)

        r = _run(["persona", "experiment", "assign", "sess-1"], p)
        assert "variant" in _combined(r)

        _run(["persona", "experiment", "record", "A", "--success"], p)
        _run(["persona", "experiment", "record", "A", "--failure", "--refused"], p)
        r = _run(["persona", "experiment", "results"], p)
        out = _combined(r)
        assert "A" in out
        assert "%" in out  # rates rendered

    def test_disable_and_clear(self, tmp_path: Path, monkeypatch) -> None:
        from missy.agent import persona_experiment as mod
        from missy.agent.persona import PersonaConfig

        monkeypatch.setattr(mod, "_snapshot_current_persona", lambda: PersonaConfig(name="Base"))
        p = tmp_path / "pexp.json"
        _run(["persona", "experiment", "add-variant", "A"], p)
        _run(["persona", "experiment", "enable"], p)
        r = _run(["persona", "experiment", "disable"], p)
        assert "disabled" in _combined(r).lower()
        r = _run(["persona", "experiment", "clear", "--yes"], p)
        assert "cleared" in _combined(r).lower()
        r = _run(["persona", "experiment", "list"], p)
        assert "No variants" in _combined(r)
