"""Tests for the `missy skills promote` CLI (F20)."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from missy.agent.playbook import Playbook
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


@pytest.fixture
def playbook(tmp_path: Path) -> Playbook:
    pb = Playbook(store_path=str(tmp_path / "playbook.json"))
    for _ in range(4):
        pb.record("shell", "deploy the app", ["shell_exec", "file_write"], "use rsync")
    return pb


def _invoke(args, playbook_obj):
    with (
        patch("missy.cli.main._load_subsystems", return_value=object()),
        patch("missy.agent.playbook.Playbook", return_value=playbook_obj),
    ):
        return _runner().invoke(cli, args)


class TestSkillsPromote:
    def test_promotes_and_writes_proposal(self, playbook: Playbook, tmp_path: Path) -> None:
        prop = str(tmp_path / "proposals")
        result = _invoke(["skills", "promote", "--proposals-dir", prop], playbook)
        out = _combined(result)
        assert result.exit_code == 0, out
        assert "Promoted 1" in out
        assert "shell" in out
        assert os.path.isdir(prop)

    def test_dry_run_writes_nothing(self, playbook: Playbook, tmp_path: Path) -> None:
        prop = str(tmp_path / "proposals")
        result = _invoke(["skills", "promote", "--proposals-dir", prop, "--dry-run"], playbook)
        out = _combined(result)
        assert result.exit_code == 0, out
        assert "Would promote 1" in out
        assert not os.path.exists(prop)

    def test_nothing_to_promote(self, tmp_path: Path) -> None:
        empty = Playbook(store_path=str(tmp_path / "empty.json"))
        result = _invoke(["skills", "promote"], empty)
        assert "No playbook patterns" in _combined(result)

    def test_threshold_option(self, playbook: Playbook, tmp_path: Path) -> None:
        # success_count is 4; a threshold of 5 promotes nothing.
        result = _invoke(["skills", "promote", "--threshold", "5"], playbook)
        assert "No playbook patterns" in _combined(result)
