"""Tests for `missy memory reindex|semantic-search` (F12)."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from missy.cli.main import cli
from missy.memory.sqlite_store import ConversationTurn, SQLiteMemoryStore

pytest.importorskip("faiss")


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


def _run(args, *, store, index_path: Path):
    with (
        patch("missy.cli.main._load_subsystems", return_value=object()),
        patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=store),
        patch(
            "missy.memory.semantic_index.DEFAULT_SEMANTIC_INDEX_PATH",
            str(index_path),
        ),
    ):
        return _runner().invoke(cli, args)


class TestMemorySemanticCLI:
    def test_reindex_then_search(self, tmp_path: Path) -> None:
        store = SQLiteMemoryStore(db_path=str(tmp_path / "mem.db"))
        store.add_turn(ConversationTurn.new("s1", "user", "I really enjoy playing chess"))
        store.add_turn(ConversationTurn.new("s1", "user", "the sky is blue and clear"))
        idx_path = tmp_path / "sem.faiss"

        r = _run(["memory", "reindex"], store=store, index_path=idx_path)
        assert r.exit_code == 0, _combined(r)
        assert "Indexed 2" in _combined(r)

        r = _run(["memory", "semantic-search", "chess"], store=store, index_path=idx_path)
        out = _combined(r)
        assert r.exit_code == 0, out
        assert "chess" in out.lower()

    def test_search_empty_index(self, tmp_path: Path) -> None:
        store = SQLiteMemoryStore(db_path=str(tmp_path / "mem.db"))
        r = _run(
            ["memory", "semantic-search", "anything"],
            store=store,
            index_path=tmp_path / "empty.faiss",
        )
        assert "No semantic matches" in _combined(r)
