"""Targeted coverage for hotreload and plan config modules.


Focuses on gaps not addressed by existing test files:
- test_hotreload.py
- test_plan.py
- test_migrate_plan_edges.py
- test_config_extended.py

Covers constructor internals, default parameter values, string-path
interfaces, edge paths in _watch(), _do_reload(), diff_configs(), and
backup/rollback interactions not exercised by prior suites.
"""

from __future__ import annotations

import pathlib
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.config.hotreload import ConfigWatcher
from missy.config.plan import (
    MAX_BACKUPS,
    backup_config,
    diff_configs,
    list_backups,
    rollback,
)

# ---------------------------------------------------------------------------
# ConfigWatcher — constructor internals and defaults
# ---------------------------------------------------------------------------


class TestConfigWatcherConstructorDefaults:
    """Constructor stores every parameter with the correct default."""

    def test_debounce_default_is_two_seconds(self, tmp_path):
        """debounce_seconds defaults to 2.0 when not supplied."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        w = ConfigWatcher(str(cfg), lambda c: None)
        assert w._debounce == 2.0

    def test_poll_interval_default_is_one_second(self, tmp_path):
        """poll_interval defaults to 1.0 when not supplied."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        w = ConfigWatcher(str(cfg), lambda c: None)
        assert w._poll == 1.0

    def test_custom_debounce_stored(self, tmp_path):
        """A custom debounce_seconds value is preserved on the instance."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        w = ConfigWatcher(str(cfg), lambda c: None, debounce_seconds=7.5)
        assert w._debounce == 7.5

    def test_custom_poll_stored(self, tmp_path):
        """A custom poll_interval value is preserved on the instance."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        w = ConfigWatcher(str(cfg), lambda c: None, poll_interval=0.25)
        assert w._poll == 0.25

    def test_last_mtime_initialised_to_zero(self, tmp_path):
        """_last_mtime is 0.0 before start() is called."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        w = ConfigWatcher(str(cfg), lambda c: None)
        assert w._last_mtime == 0.0

    def test_last_change_time_initialised_to_zero(self, tmp_path):
        """_last_change_time is 0.0 before any change is detected."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        w = ConfigWatcher(str(cfg), lambda c: None)
        assert w._last_change_time == 0.0

    def test_thread_initially_none(self, tmp_path):
        """_thread is None before start() is called."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        w = ConfigWatcher(str(cfg), lambda c: None)
        assert w._thread is None

    def test_stop_event_is_threading_event(self, tmp_path):
        """_stop is a threading.Event instance."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        w = ConfigWatcher(str(cfg), lambda c: None)
        assert isinstance(w._stop, threading.Event)

    def test_reload_fn_stored(self, tmp_path):
        """The reload_fn callable is stored as _reload_fn."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        sentinel = lambda c: None  # noqa: E731
        w = ConfigWatcher(str(cfg), sentinel)
        assert w._reload_fn is sentinel

    def test_path_is_path_object(self, tmp_path):
        """_path is a pathlib.Path, not a raw string."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        w = ConfigWatcher(str(cfg), lambda c: None)
        assert isinstance(w._path, pathlib.Path)


# ---------------------------------------------------------------------------
# ConfigWatcher._check_file_safety — fine-grained edge cases
# ---------------------------------------------------------------------------


class TestCheckFileSafetyEdgeCases:
    """Targeted edge cases not covered by existing safety tests."""

    def test_group_readable_no_write_bit_passes(self, tmp_path):
        """0o640 has no write bits for group or world — must be accepted."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        cfg.chmod(0o640)
        w = ConfigWatcher(str(cfg), MagicMock())
        assert w._check_file_safety() is True

    def test_both_group_and_world_writable_returns_false(self, tmp_path):
        """0o666 has both group-write and world-write bits — must be rejected."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        cfg.chmod(0o666)
        w = ConfigWatcher(str(cfg), MagicMock())
        assert w._check_file_safety() is False

    def test_is_symlink_checked_before_stat(self, tmp_path):
        """is_symlink() is consulted before stat(); a symlink short-circuits to False."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        cfg.chmod(0o600)
        w = ConfigWatcher(str(cfg), MagicMock())
        with patch.object(pathlib.Path, "is_symlink", return_value=True):
            result = w._check_file_safety()
        assert result is False

    def test_oserror_on_is_symlink_returns_false(self, tmp_path):
        """If is_symlink() raises OSError, _check_file_safety returns False."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        w = ConfigWatcher(str(cfg), MagicMock())
        # is_symlink is called first; if it raises, the outer except catches it
        original_is_symlink = pathlib.Path.is_symlink

        def raising_is_symlink(self_path):
            if self_path == w._path:
                raise OSError("proc gone")
            return original_is_symlink(self_path)

        with patch.object(pathlib.Path, "is_symlink", raising_is_symlink):
            result = w._check_file_safety()
        assert result is False

    def test_only_world_write_bit_set_returns_false(self, tmp_path):
        """0o602 sets the world-write bit only — must be rejected."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        cfg.chmod(0o602)  # owner=rw, others=w (S_IWOTH)
        w = ConfigWatcher(str(cfg), MagicMock())
        assert w._check_file_safety() is False

    def test_only_group_write_bit_set_returns_false(self, tmp_path):
        """0o620 has only the group-write bit — must be rejected."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        cfg.chmod(0o620)
        w = ConfigWatcher(str(cfg), MagicMock())
        assert w._check_file_safety() is False

    def test_nonexistent_file_returns_false(self, tmp_path):
        """_check_file_safety returns False if the file no longer exists."""
        cfg = tmp_path / "does_not_exist.yaml"
        w = ConfigWatcher(str(cfg), MagicMock())
        assert w._check_file_safety() is False


# ---------------------------------------------------------------------------
# ConfigWatcher._do_reload — load_config called with str path
# ---------------------------------------------------------------------------


class TestDoReloadPathPassedAsString:
    """_do_reload must pass the config path as a string to load_config."""

    @patch("missy.config.settings.load_config")
    def test_load_config_receives_string_path(self, mock_load, tmp_path):
        """load_config is called with str(self._path), not a Path object."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        cfg.chmod(0o600)
        mock_load.return_value = MagicMock()
        w = ConfigWatcher(str(cfg), MagicMock())
        w._do_reload()
        args, _ = mock_load.call_args
        assert isinstance(args[0], str)

    @patch("missy.config.settings.load_config")
    def test_do_reload_noop_on_unsafe_file_no_load_config_call(self, mock_load, tmp_path):
        """When _check_file_safety() returns False, load_config is never called."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        cfg.chmod(0o666)  # group- and world-writable
        w = ConfigWatcher(str(cfg), MagicMock())
        w._do_reload()
        mock_load.assert_not_called()

    @patch("missy.config.settings.load_config")
    def test_do_reload_reload_fn_not_called_on_load_exception(self, mock_load, tmp_path):
        """If load_config raises, reload_fn must not be called."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        cfg.chmod(0o600)
        mock_load.side_effect = ValueError("bad YAML")
        cb = MagicMock()
        w = ConfigWatcher(str(cfg), cb)
        w._do_reload()
        cb.assert_not_called()


# ---------------------------------------------------------------------------
# ConfigWatcher start() — _last_mtime set from file stat
# ---------------------------------------------------------------------------


class TestStartSetsLastMtime:
    """start() seeds _last_mtime from the file's current mtime."""

    def test_start_reads_mtime_from_existing_file(self, tmp_path):
        """After start(), _last_mtime equals the file's actual mtime."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        expected_mtime = cfg.stat().st_mtime
        w = ConfigWatcher(str(cfg), MagicMock(), poll_interval=60)
        w.start()
        w.stop()
        assert w._last_mtime == pytest.approx(expected_mtime, rel=1e-6)

    def test_start_clears_stop_event(self, tmp_path):
        """start() clears the _stop event so the watch loop can run."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("k: v")
        w = ConfigWatcher(str(cfg), MagicMock(), poll_interval=60)
        w._stop.set()  # pre-set to simulate a previous stop
        w.start()
        assert not w._stop.is_set()
        w.stop()


# ---------------------------------------------------------------------------
# ConfigWatcher — no spurious reload when mtime is unchanged
# ---------------------------------------------------------------------------


class TestNoSpuriousReload:
    """The watcher must NOT fire reload_fn when the file has not changed."""

    @patch("missy.config.settings.load_config")
    def test_stable_file_never_triggers_reload(self, mock_load, tmp_path):
        """If the file is never modified, reload_fn is never called."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("stable: true")
        cfg.chmod(0o600)
        mock_load.return_value = MagicMock()
        cb = MagicMock()

        w = ConfigWatcher(str(cfg), cb, debounce_seconds=0.05, poll_interval=0.02)
        w.start()
        time.sleep(0.3)
        w.stop()

        cb.assert_not_called()


# ---------------------------------------------------------------------------
# backup_config — string vs Path arguments
# ---------------------------------------------------------------------------


class TestBackupConfigStringPaths:
    """backup_config accepts string paths as well as Path objects."""

    def test_string_config_path_accepted(self, tmp_path):
        """backup_config works when config_path is a plain str."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("v: 1\n")
        bd = tmp_path / "bk"
        result = backup_config(str(cfg), str(bd))
        assert result.exists()

    def test_string_backup_dir_accepted(self, tmp_path):
        """backup_config works when backup_dir is a plain str."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("v: 1\n")
        bd = str(tmp_path / "bk")
        result = backup_config(cfg, bd)
        assert result.exists()

    def test_returns_path_object(self, tmp_path):
        """backup_config always returns a pathlib.Path, not a str."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("v: 1\n")
        bd = tmp_path / "bk"
        result = backup_config(cfg, bd)
        assert isinstance(result, Path)

    def test_source_not_found_raises(self, tmp_path):
        """If the source file does not exist, backup_config propagates the error."""
        missing = tmp_path / "nonexistent.yaml"
        bd = tmp_path / "bk"
        with pytest.raises(OSError):
            backup_config(missing, bd)


# ---------------------------------------------------------------------------
# backup_config — backup content fidelity
# ---------------------------------------------------------------------------


class TestBackupContentFidelity:
    """Backup file must reproduce the source precisely."""

    def test_single_newline_file_preserved(self, tmp_path):
        """A file containing a single newline is preserved verbatim."""
        cfg = tmp_path / "config.yaml"
        cfg.write_bytes(b"\n")
        bd = tmp_path / "bk"
        bkp = backup_config(cfg, bd)
        assert bkp.read_bytes() == b"\n"

    def test_large_content_preserved(self, tmp_path):
        """A multi-kilobyte config file is reproduced byte-for-byte."""
        content = ("k: " + "v" * 200 + "\n") * 50
        cfg = tmp_path / "config.yaml"
        cfg.write_text(content, encoding="utf-8")
        bd = tmp_path / "bk"
        bkp = backup_config(cfg, bd)
        assert bkp.read_text(encoding="utf-8") == content


# ---------------------------------------------------------------------------
# list_backups — string path argument
# ---------------------------------------------------------------------------


class TestListBackupsStringPath:
    """list_backups accepts a string path."""

    def test_string_path_returns_list(self, tmp_path):
        """list_backups works when backup_dir is a plain str."""
        bd = tmp_path / "bk"
        bd.mkdir()
        result = list_backups(str(bd))
        assert isinstance(result, list)

    def test_nonexistent_dir_string_returns_empty(self, tmp_path):
        """list_backups with a non-existent string path returns []."""
        result = list_backups(str(tmp_path / "nowhere"))
        assert result == []

    def test_list_backups_returns_path_objects(self, tmp_path):
        """Every entry returned by list_backups is a pathlib.Path."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("v: 1\n")
        bd = tmp_path / "bk"
        backup_config(cfg, bd)
        for p in list_backups(bd):
            assert isinstance(p, Path)


# ---------------------------------------------------------------------------
# rollback — string path arguments and return value types
# ---------------------------------------------------------------------------


class TestRollbackStringPaths:
    """rollback accepts string paths for both arguments."""

    def test_string_config_and_backup_dir_accepted(self, tmp_path):
        """rollback works when both arguments are plain strings."""
        cfg = tmp_path / "config.yaml"
        bd = tmp_path / "bk"
        cfg.write_text("original: true\n")
        backup_config(cfg, bd)
        cfg.write_text("modified: true\n")
        result = rollback(str(cfg), str(bd))
        assert result is not None
        assert cfg.read_text() == "original: true\n"

    def test_rollback_returns_path_or_none(self, tmp_path):
        """rollback returns a Path (when backup exists) or None (when it does not)."""
        cfg = tmp_path / "config.yaml"
        bd = tmp_path / "bk"
        assert rollback(str(cfg), str(bd)) is None

        cfg.write_text("v: 1\n")
        backup_config(cfg, bd)
        cfg.write_text("v: 2\n")
        result = rollback(str(cfg), str(bd))
        assert isinstance(result, Path)

    def test_rollback_to_earliest_when_only_one_backup(self, tmp_path):
        """With a single backup, rollback restores that one backup exactly."""
        cfg = tmp_path / "config.yaml"
        bd = tmp_path / "bk"
        original = "single_backup: true\n"
        cfg.write_text(original)
        backup_config(cfg, bd)
        cfg.write_text("overwritten: true\n")
        rollback(cfg, bd)
        assert original in cfg.read_text()


# ---------------------------------------------------------------------------
# diff_configs — string path arguments and error behaviour
# ---------------------------------------------------------------------------


class TestDiffConfigsStringPaths:
    """diff_configs accepts string paths."""

    def test_string_paths_accepted(self, tmp_path):
        """diff_configs works when both paths are plain strings."""
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text("x: 1\n")
        b.write_text("x: 2\n")
        diff = diff_configs(str(a), str(b))
        assert "x" in diff

    def test_nonexistent_file_raises(self, tmp_path):
        """diff_configs raises FileNotFoundError when a path does not exist."""
        a = tmp_path / "a.yaml"
        a.write_text("x: 1\n")
        with pytest.raises(FileNotFoundError):
            diff_configs(a, tmp_path / "missing.yaml")

    def test_diff_output_contains_unified_diff_headers(self, tmp_path):
        """The diff output includes '---' and '+++' header lines."""
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text("key: old\n")
        b.write_text("key: new\n")
        diff = diff_configs(a, b)
        assert "---" in diff
        assert "+++" in diff

    def test_diff_reflects_actual_changed_value(self, tmp_path):
        """The diff body contains the old and new values."""
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text("timeout: 30\n")
        b.write_text("timeout: 60\n")
        diff = diff_configs(a, b)
        assert "30" in diff
        assert "60" in diff

    def test_diff_unicode_content_handled(self, tmp_path):
        """diff_configs handles files with non-ASCII (UTF-8) content."""
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text("name: résumé\n", encoding="utf-8")
        b.write_text("name: curriculum vitae\n", encoding="utf-8")
        diff = diff_configs(a, b)
        assert "résumé" in diff or "curriculum" in diff


# ---------------------------------------------------------------------------
# backup_config — pruning boundary at MAX_BACKUPS - 1
# ---------------------------------------------------------------------------


class TestPruningBoundary:
    """Boundary conditions around the MAX_BACKUPS threshold."""

    def test_one_below_max_no_pruning(self, tmp_path):
        """Creating MAX_BACKUPS - 1 backups must not prune any."""
        cfg = tmp_path / "config.yaml"
        bd = tmp_path / "bk"
        for i in range(MAX_BACKUPS - 1):
            cfg.write_text(f"v: {i}\n")
            with patch("missy.config.plan.time") as mt:
                mt.strftime.return_value = f"20260301_0000{i:02d}"
                backup_config(cfg, bd)
        assert len(list_backups(bd)) == MAX_BACKUPS - 1

    def test_two_above_max_prunes_two_oldest(self, tmp_path):
        """Creating MAX_BACKUPS + 2 backups must prune exactly 2, leaving MAX_BACKUPS."""
        cfg = tmp_path / "config.yaml"
        bd = tmp_path / "bk"
        timestamps = [f"20260302_{i:06d}" for i in range(MAX_BACKUPS + 2)]
        names = []
        for i, ts in enumerate(timestamps):
            cfg.write_text(f"v: {i}\n")
            with patch("missy.config.plan.time") as mt:
                mt.strftime.return_value = ts
                names.append(backup_config(cfg, bd).name)
        remaining = {p.name for p in list_backups(bd)}
        assert len(remaining) == MAX_BACKUPS
        # Two oldest must be gone
        assert names[0] not in remaining
        assert names[1] not in remaining


# ---------------------------------------------------------------------------
# rollback — interaction with backup pruning
# ---------------------------------------------------------------------------


class TestRollbackPruningInteraction:
    """rollback pre-backs-up the current config; this counts toward the MAX_BACKUPS limit."""

    def test_rollback_pre_backup_triggers_pruning_when_at_max(self, tmp_path):
        """When already at MAX_BACKUPS, the pre-rollback backup causes one prune."""
        cfg = tmp_path / "config.yaml"
        bd = tmp_path / "bk"

        # Fill up to exactly MAX_BACKUPS backups with distinct timestamps.
        for i in range(MAX_BACKUPS):
            cfg.write_text(f"v: {i}\n")
            with patch("missy.config.plan.time") as mt:
                mt.strftime.return_value = f"20260303_0000{i:02d}"
                backup_config(cfg, bd)

        assert len(list_backups(bd)) == MAX_BACKUPS

        # Modify config; rollback will create an extra backup (pre-restore snapshot),
        # pushing over MAX_BACKUPS and triggering a prune.
        cfg.write_text("v: current\n")
        with patch("missy.config.plan.time") as mt:
            mt.strftime.return_value = "20260303_000099"
            rollback(cfg, bd)

        # Total must remain capped at MAX_BACKUPS.
        assert len(list_backups(bd)) <= MAX_BACKUPS
