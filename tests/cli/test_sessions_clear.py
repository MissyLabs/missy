"""Tests for `missy sessions clear` (F14 operator surface for clear_session_full).

These drive the real Click command through CliRunner, pointing the CLI at a
temp SQLite store, and assert the full-reset behaviour, the confirmation
guard, name resolution, and not-found handling.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from missy.cli.main import cli
from missy.memory.sqlite_store import ConversationTurn, SQLiteMemoryStore, SummaryRecord


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
def store(tmp_path: Path) -> SQLiteMemoryStore:
    return SQLiteMemoryStore(db_path=str(tmp_path / "sessions_clear.db"))


@pytest.fixture
def seeded(store: SQLiteMemoryStore) -> SQLiteMemoryStore:
    store.add_turn(ConversationTurn.new("sess-target", "user", "hello"))
    store.add_turn(ConversationTurn.new("sess-target", "assistant", "hi there"))
    store.add_summary(
        SummaryRecord(id="sum-t", session_id="sess-target", depth=1, content="contam")
    )
    # A second, unrelated session that must never be touched.
    store.add_turn(ConversationTurn.new("sess-other", "user", "keep me"))
    return store


def _invoke(args, store_obj):
    """Run the CLI command with _load_subsystems stubbed and the CLI's
    SQLiteMemoryStore() pointed at our temp store."""
    with (
        patch("missy.cli.main._load_subsystems", return_value=object()),
        patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=store_obj),
    ):
        return _runner().invoke(cli, args)


class TestSessionsClear:
    def test_clears_turns_and_summaries_with_yes(self, seeded: SQLiteMemoryStore) -> None:
        result = _invoke(["sessions", "clear", "sess-target", "--yes"], seeded)
        assert result.exit_code == 0, _combined(result)
        assert seeded.session_exists("sess-target") is False
        assert seeded.get_session_turns("sess-target") == []
        assert seeded.get_summaries("sess-target") == []

    def test_reports_removed_counts(self, seeded: SQLiteMemoryStore) -> None:
        result = _invoke(["sessions", "clear", "sess-target", "-y"], seeded)
        out = _combined(result)
        assert "2 turn" in out
        assert "1 summary" in out
        assert "Restart the gateway" in out

    def test_does_not_touch_other_sessions(self, seeded: SQLiteMemoryStore) -> None:
        _invoke(["sessions", "clear", "sess-target", "--yes"], seeded)
        assert seeded.session_exists("sess-other") is True
        assert len(seeded.get_session_turns("sess-other")) == 1

    def test_unknown_session_reports_not_found(self, store: SQLiteMemoryStore) -> None:
        result = _invoke(["sessions", "clear", "does-not-exist", "--yes"], store)
        assert "not found" in _combined(result).lower()

    def test_confirmation_abort_leaves_data(self, seeded: SQLiteMemoryStore) -> None:
        # No --yes: feed "n" to the click.confirm prompt -> abort, nothing cleared.
        with (
            patch("missy.cli.main._load_subsystems", return_value=object()),
            patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=seeded),
        ):
            result = _runner().invoke(cli, ["sessions", "clear", "sess-target"], input="n\n")
        assert seeded.session_exists("sess-target") is True
        assert "Aborted" in _combined(result)

    def test_confirmation_yes_clears(self, seeded: SQLiteMemoryStore) -> None:
        with (
            patch("missy.cli.main._load_subsystems", return_value=object()),
            patch("missy.memory.sqlite_store.SQLiteMemoryStore", return_value=seeded),
        ):
            result = _runner().invoke(cli, ["sessions", "clear", "sess-target"], input="y\n")
        assert result.exit_code == 0, _combined(result)
        assert seeded.session_exists("sess-target") is False

    def test_resolves_friendly_name(self, seeded: SQLiteMemoryStore) -> None:
        # A session must be registered (as the gateway does per OPS-012)
        # before it can carry a friendly name / be resolved by one.
        seeded.register_session("sess-target")
        assert seeded.rename_session("sess-target", "myfriendly") is True
        result = _invoke(["sessions", "clear", "myfriendly", "--yes"], seeded)
        assert result.exit_code == 0, _combined(result)
        assert seeded.session_exists("sess-target") is False

    def test_clears_summary_only_session(self, store: SQLiteMemoryStore) -> None:
        # Session with only a summary (raw turns pruned) is still clearable.
        store.add_summary(SummaryRecord(id="so-1", session_id="sess-summ", depth=1, content="x"))
        assert store.session_exists("sess-summ") is True
        result = _invoke(["sessions", "clear", "sess-summ", "--yes"], store)
        assert result.exit_code == 0, _combined(result)
        assert store.session_exists("sess-summ") is False
