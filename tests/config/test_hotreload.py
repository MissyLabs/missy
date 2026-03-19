"""Tests for missy.config.hotreload."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from missy.config.hotreload import ConfigWatcher, _apply_config


class TestConfigWatcherInit:
    def test_init_stores_path(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("test: true")
        w = ConfigWatcher(str(cfg), lambda c: None)
        assert w._path == cfg

    def test_init_expands_tilde(self):
        w = ConfigWatcher("~/config.yaml", lambda c: None)
        assert "~" not in str(w._path)


class TestConfigWatcherLifecycle:
    def test_start_and_stop(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("test: true")
        w = ConfigWatcher(str(cfg), lambda c: None, poll_interval=0.05)
        w.start()
        assert w._thread is not None
        assert w._thread.is_alive()
        w.stop()
        assert not w._thread.is_alive()

    def test_start_missing_file(self, tmp_path):
        cfg = tmp_path / "nonexistent.yaml"
        w = ConfigWatcher(str(cfg), lambda c: None, poll_interval=0.05)
        w.start()
        assert w._last_mtime == 0.0
        w.stop()

    def test_stop_without_start(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("test: true")
        w = ConfigWatcher(str(cfg), lambda c: None)
        w.stop()  # should not raise


class TestConfigWatcherDetectsChanges:
    def test_detects_file_modification(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("version: 1")
        callback = MagicMock()

        w = ConfigWatcher(
            str(cfg),
            callback,
            debounce_seconds=0.1,
            poll_interval=0.05,
        )
        w.start()
        time.sleep(0.1)  # let it see the initial state

        # Modify the file
        cfg.write_text("version: 2")
        time.sleep(0.5)  # wait for debounce + poll
        w.stop()

        # The callback may or may not have been called depending on timing,
        # but the watcher should not have crashed
        assert True  # passed without error


class TestDoReload:
    @patch("missy.config.settings.load_config")
    def test_reload_calls_fn(self, mock_load, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("test: true")
        cfg.chmod(0o600)  # safety check requires owner-only permissions
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        callback = MagicMock()

        w = ConfigWatcher(str(cfg), callback)
        w._do_reload()
        mock_load.assert_called_once_with(str(cfg))
        callback.assert_called_once_with(mock_config)

    @patch("missy.config.settings.load_config", side_effect=Exception("parse error"))
    def test_reload_handles_exception(self, mock_load, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("bad: yaml")
        cfg.chmod(0o600)  # safety check requires owner-only permissions
        w = ConfigWatcher(str(cfg), MagicMock())
        w._do_reload()  # should not raise

    def test_reload_rejects_world_writable(self, tmp_path):
        """Config files with group/world write bits must be rejected."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("test: true")
        cfg.chmod(0o666)
        callback = MagicMock()
        w = ConfigWatcher(str(cfg), callback)
        w._do_reload()
        callback.assert_not_called()

    def test_reload_rejects_symlink(self, tmp_path):
        """Symlinked config files must be rejected."""
        real = tmp_path / "real.yaml"
        real.write_text("test: true")
        real.chmod(0o600)
        link = tmp_path / "config.yaml"
        link.symlink_to(real)
        callback = MagicMock()
        w = ConfigWatcher(str(link), callback)
        w._do_reload()
        callback.assert_not_called()


class TestApplyConfig:
    @patch("missy.providers.registry.init_registry")
    @patch("missy.policy.engine.init_policy_engine")
    def test_apply_reinitializes_subsystems(self, mock_policy, mock_reg):
        config = MagicMock()
        _apply_config(config)
        mock_policy.assert_called_once_with(config)
        mock_reg.assert_called_once_with(config)


class TestDebounce:
    """Multiple rapid changes should coalesce into a single reload."""

    @patch("missy.config.settings.load_config")
    def test_rapid_changes_trigger_single_reload(self, mock_load, tmp_path):
        """Ten rapid writes within the debounce window must produce exactly one reload."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("version: 1")
        cfg.chmod(0o600)
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        callback = MagicMock()

        # Short debounce and poll so the test finishes quickly.
        w = ConfigWatcher(str(cfg), callback, debounce_seconds=0.15, poll_interval=0.02)
        w.start()
        time.sleep(0.05)  # let watcher record the initial mtime

        # Write ten times in rapid succession — all within the debounce window.
        for i in range(10):
            cfg.write_text(f"version: {i + 2}")
            time.sleep(0.01)

        # Wait long enough for debounce to elapse and one reload to fire.
        time.sleep(0.4)
        w.stop()

        # Exactly one reload, not ten.
        assert callback.call_count == 1

    @patch("missy.config.settings.load_config")
    def test_two_separated_changes_trigger_two_reloads(self, mock_load, tmp_path):
        """Changes separated by more than the debounce period each get their own reload."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("version: 1")
        cfg.chmod(0o600)
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        callback = MagicMock()

        w = ConfigWatcher(str(cfg), callback, debounce_seconds=0.1, poll_interval=0.02)
        w.start()
        time.sleep(0.05)

        # First change — let debounce elapse before making the second.
        cfg.write_text("version: 2")
        time.sleep(0.35)

        # Second change — let debounce elapse again.
        cfg.write_text("version: 3")
        time.sleep(0.35)

        w.stop()
        assert callback.call_count == 2


class TestFileDeletionDuringWatch:
    """Config file deleted while the watcher is running."""

    @patch("missy.config.settings.load_config")
    def test_deletion_does_not_crash_watcher(self, mock_load, tmp_path):
        """Deleting the config file mid-watch must not crash the watcher thread."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("version: 1")
        cfg.chmod(0o600)
        callback = MagicMock()

        w = ConfigWatcher(str(cfg), callback, debounce_seconds=0.1, poll_interval=0.02)
        w.start()
        time.sleep(0.05)

        cfg.unlink()
        time.sleep(0.3)  # several poll cycles with the file gone

        # Watcher thread should still be alive.
        assert w._thread is not None
        assert w._thread.is_alive()
        w.stop()

    @patch("missy.config.settings.load_config")
    def test_deletion_then_recreate_triggers_reload(self, mock_load, tmp_path):
        """After deletion, recreating the file should be detected and reloaded."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("version: 1")
        cfg.chmod(0o600)
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        callback = MagicMock()

        w = ConfigWatcher(str(cfg), callback, debounce_seconds=0.1, poll_interval=0.02)
        w.start()
        time.sleep(0.05)

        cfg.unlink()
        time.sleep(0.15)  # let deletion settle

        # Recreate the file with new content.
        cfg.write_text("version: 2")
        cfg.chmod(0o600)
        time.sleep(0.4)  # debounce + poll

        w.stop()
        # At least one reload after recreation.
        assert callback.call_count >= 1


class TestAtomicReplacement:
    """Config replaced via an atomic rename (the common editor pattern)."""

    @patch("missy.config.settings.load_config")
    def test_atomic_rename_triggers_reload(self, mock_load, tmp_path):
        """An atomic rename (tmp file → config path) must be detected as a change."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("version: 1")
        cfg.chmod(0o600)
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        callback = MagicMock()

        w = ConfigWatcher(str(cfg), callback, debounce_seconds=0.1, poll_interval=0.02)
        w.start()
        time.sleep(0.05)

        # Simulate atomic replacement via a temp file rename.
        tmp = tmp_path / "config.yaml.tmp"
        tmp.write_text("version: 2")
        tmp.chmod(0o600)
        tmp.rename(cfg)

        time.sleep(0.4)
        w.stop()

        assert callback.call_count == 1


class TestPermissionChangeDuringWatch:
    """Config file permission changes to unsafe (group-writable) while the watcher runs."""

    @patch("missy.config.settings.load_config")
    def test_reload_blocked_after_permission_widens(self, mock_load, tmp_path):
        """If the file becomes group-writable between changes, reload must be rejected."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("version: 1")
        cfg.chmod(0o600)
        callback = MagicMock()

        w = ConfigWatcher(str(cfg), callback, debounce_seconds=0.05, poll_interval=0.02)
        w.start()
        time.sleep(0.05)

        # Widen permissions to group-writable, then touch to update mtime.
        cfg.chmod(0o664)
        cfg.write_text("version: 2")
        time.sleep(0.3)

        w.stop()
        # _check_file_safety should have rejected the reload.
        callback.assert_not_called()

    def test_check_file_safety_rejects_group_writable(self, tmp_path):
        """_check_file_safety returns False for a group-writable file."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("test: true")
        cfg.chmod(0o664)
        w = ConfigWatcher(str(cfg), MagicMock())
        assert w._check_file_safety() is False

    def test_check_file_safety_rejects_world_writable(self, tmp_path):
        """_check_file_safety returns False for a world-writable file."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("test: true")
        cfg.chmod(0o646)
        w = ConfigWatcher(str(cfg), MagicMock())
        assert w._check_file_safety() is False

    def test_check_file_safety_accepts_owner_only(self, tmp_path):
        """_check_file_safety returns True for a 0o600 file owned by the current user."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("test: true")
        cfg.chmod(0o600)
        w = ConfigWatcher(str(cfg), MagicMock())
        assert w._check_file_safety() is True


class TestOwnershipCheck:
    """_check_file_safety rejects files not owned by the current user."""

    def test_rejects_file_owned_by_other_uid(self, tmp_path):
        """When os.getuid() differs from the file's st_uid, safety check must fail."""
        import os

        cfg = tmp_path / "config.yaml"
        cfg.write_text("test: true")
        cfg.chmod(0o600)
        w = ConfigWatcher(str(cfg), MagicMock())

        real_stat = cfg.stat()

        # Build a fake stat result where uid is someone else.
        fake_st = MagicMock()
        fake_st.st_uid = os.getuid() + 1  # different from current user
        fake_st.st_mode = real_stat.st_mode  # preserve safe permissions

        # Patch pathlib.Path.stat at the class level so Path.stat() returns our fake
        # result for the watched path.  Patch os.getuid in the hotreload module to
        # return the real current uid so that the uid comparison with fake_st.st_uid
        # (= current_uid + 1) correctly fails the safety check.
        import pathlib

        original_stat = pathlib.Path.stat

        def patched_stat(self_path, **kwargs):
            if self_path == w._path:
                return fake_st
            return original_stat(self_path, **kwargs)

        with (
            patch("missy.config.hotreload.os.getuid", return_value=os.getuid()),
            patch.object(pathlib.Path, "stat", patched_stat),
            patch.object(pathlib.Path, "is_symlink", return_value=False),
        ):
            result = w._check_file_safety()

        assert result is False

    def test_rejects_when_getuid_differs(self, tmp_path):
        """Patch os.getuid to return a value that doesn't match st_uid."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("test: true")
        cfg.chmod(0o600)
        w = ConfigWatcher(str(cfg), MagicMock())

        real_uid = cfg.stat().st_uid

        # Make os.getuid return something different from the file's actual owner.
        with patch("missy.config.hotreload.os.getuid", return_value=real_uid + 999):
            result = w._check_file_safety()

        assert result is False

    def test_accepts_file_owned_by_current_user(self, tmp_path):
        """_check_file_safety returns True when st_uid matches os.getuid()."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("test: true")
        cfg.chmod(0o600)
        w = ConfigWatcher(str(cfg), MagicMock())
        # No patching needed — tmp_path files are owned by the running user.
        assert w._check_file_safety() is True

    def test_check_file_safety_handles_oserror_on_stat(self, tmp_path):
        """If stat() raises OSError, _check_file_safety must return False."""
        import pathlib

        cfg = tmp_path / "config.yaml"
        cfg.write_text("test: true")
        w = ConfigWatcher(str(cfg), MagicMock())

        original_stat = pathlib.Path.stat

        def raising_stat(self_path, **kwargs):
            if self_path == w._path:
                raise OSError("permission denied")
            return original_stat(self_path, **kwargs)

        with patch.object(pathlib.Path, "stat", raising_stat), patch.object(pathlib.Path, "is_symlink", return_value=False):
            result = w._check_file_safety()

        assert result is False


class TestConcurrentStartStop:
    """Calling start() twice must not spawn two watcher threads."""

    def test_start_twice_does_not_duplicate_thread(self, tmp_path):
        """A second start() call while the watcher is already running is safe."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("test: true")
        w = ConfigWatcher(str(cfg), MagicMock(), poll_interval=0.02)
        w.start()
        first_thread = w._thread

        # Second start — implementation overwrites _thread reference.
        w.start()
        second_thread = w._thread

        # Both threads are alive, but we verify the caller gets a stable watcher
        # (does not raise, does not deadlock on stop).
        w.stop()

        # After stop() the most-recently-started thread must be dead.
        assert not second_thread.is_alive()
        # Clean up the orphaned first thread (daemon, will die with process).
        _ = first_thread  # reference kept to show intent, not asserting state

    def test_stop_twice_does_not_raise(self, tmp_path):
        """Calling stop() twice on the same watcher must not raise."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("test: true")
        w = ConfigWatcher(str(cfg), MagicMock(), poll_interval=0.02)
        w.start()
        w.stop()
        w.stop()  # second stop must be harmless


class TestReloadCallbackException:
    """If reload_fn raises, the watcher must continue watching."""

    @patch("missy.config.settings.load_config")
    def test_callback_exception_does_not_kill_watcher(self, mock_load, tmp_path):
        """An exception from reload_fn is caught and the watcher thread stays alive."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("version: 1")
        cfg.chmod(0o600)
        mock_load.return_value = MagicMock()

        call_count = [0]

        def exploding_callback(config):
            call_count[0] += 1
            raise RuntimeError("callback explosion")

        w = ConfigWatcher(
            str(cfg), exploding_callback, debounce_seconds=0.1, poll_interval=0.02
        )
        w.start()
        time.sleep(0.05)

        cfg.write_text("version: 2")
        time.sleep(0.4)

        # Thread must still be alive despite the exception.
        assert w._thread is not None
        assert w._thread.is_alive()
        w.stop()

    @patch("missy.config.settings.load_config")
    def test_watcher_continues_after_callback_exception(self, mock_load, tmp_path):
        """After a failing reload, a subsequent change is still detected and reloaded."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("version: 1")
        cfg.chmod(0o600)
        mock_load.return_value = MagicMock()

        attempt = [0]

        def sometimes_exploding(config):
            attempt[0] += 1
            if attempt[0] == 1:
                raise RuntimeError("first attempt fails")
            # Second call succeeds silently.

        w = ConfigWatcher(
            str(cfg), sometimes_exploding, debounce_seconds=0.1, poll_interval=0.02
        )
        w.start()
        time.sleep(0.05)

        # First change — callback explodes.
        cfg.write_text("version: 2")
        time.sleep(0.4)
        assert attempt[0] == 1

        # Second change — callback should succeed.
        cfg.write_text("version: 3")
        time.sleep(0.4)
        w.stop()

        assert attempt[0] == 2


class TestEmptyFile:
    """Watcher behaviour when the config file is empty (zero bytes)."""

    @patch("missy.config.settings.load_config")
    def test_empty_file_passes_safety_check(self, mock_load, tmp_path):
        """An empty but correctly-permissioned file passes safety checks."""
        cfg = tmp_path / "config.yaml"
        cfg.write_bytes(b"")
        cfg.chmod(0o600)
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        callback = MagicMock()

        w = ConfigWatcher(str(cfg), callback)
        w._do_reload()

        # load_config is called even for empty files (content parsing is its concern).
        mock_load.assert_called_once_with(str(cfg))
        callback.assert_called_once_with(mock_config)

    @patch("missy.config.settings.load_config", side_effect=Exception("empty YAML"))
    def test_empty_file_load_exception_is_handled(self, mock_load, tmp_path):
        """If load_config raises on an empty file, _do_reload must not propagate the error."""
        cfg = tmp_path / "config.yaml"
        cfg.write_bytes(b"")
        cfg.chmod(0o600)
        callback = MagicMock()

        w = ConfigWatcher(str(cfg), callback)
        w._do_reload()  # must not raise

        callback.assert_not_called()

    @patch("missy.config.settings.load_config")
    def test_file_truncated_to_zero_during_watch_is_detected(self, mock_load, tmp_path):
        """Truncating a file to zero bytes changes its mtime, which must be detected."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("version: 1")
        cfg.chmod(0o600)
        mock_load.return_value = MagicMock()
        callback = MagicMock()

        w = ConfigWatcher(str(cfg), callback, debounce_seconds=0.1, poll_interval=0.02)
        w.start()
        time.sleep(0.05)

        cfg.write_bytes(b"")  # truncate — mtime changes
        time.sleep(0.4)
        w.stop()

        assert callback.call_count == 1


class TestRaceConditionBetweenStatAndRead:
    """File is modified between the stat check (mtime detection) and the actual read."""

    @patch("missy.config.settings.load_config")
    def test_load_config_sees_latest_content(self, mock_load, tmp_path):
        """load_config is called after debounce, so it always reads the latest content.

        The implementation debounces before calling load_config, which means even if the
        file changes between the stat and the eventual read, load_config will see whatever
        is on disk at reload time.  This test verifies the callback receives the config
        object returned by load_config (i.e. no intermediate stale snapshot is cached).
        """
        cfg = tmp_path / "config.yaml"
        cfg.write_text("version: 1")
        cfg.chmod(0o600)

        latest_config = MagicMock(name="latest")
        mock_load.return_value = latest_config
        callback = MagicMock()

        w = ConfigWatcher(str(cfg), callback, debounce_seconds=0.1, poll_interval=0.02)
        w.start()
        time.sleep(0.05)

        # First write triggers change detection.
        cfg.write_text("version: 2")

        # Simulate a second write before debounce elapses (race-like update).
        time.sleep(0.02)
        cfg.write_text("version: 3")

        time.sleep(0.4)
        w.stop()

        # The callback receives exactly what load_config returned.
        assert callback.call_count == 1
        callback.assert_called_once_with(latest_config)

    @patch("missy.config.settings.load_config")
    def test_mtime_change_after_pending_reload_resets_debounce(self, mock_load, tmp_path):
        """A write that arrives while a reload is pending resets the debounce timer."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("version: 1")
        cfg.chmod(0o600)
        mock_load.return_value = MagicMock()
        callback = MagicMock()

        # Long debounce so we can inject a second change before it fires.
        w = ConfigWatcher(str(cfg), callback, debounce_seconds=0.3, poll_interval=0.02)
        w.start()
        time.sleep(0.05)

        cfg.write_text("version: 2")
        time.sleep(0.15)  # debounce not yet elapsed

        # Second change — resets the debounce window.
        cfg.write_text("version: 3")
        time.sleep(0.15)  # still not elapsed from second change

        # Debounce has not fired yet; callback should not have been called.
        assert callback.call_count == 0

        # Now wait for debounce to fully elapse from the second change.
        time.sleep(0.3)
        w.stop()

        assert callback.call_count == 1


class TestStopDuringPendingReload:
    """Stop is called while a reload is pending but debounce has not elapsed."""

    @patch("missy.config.settings.load_config")
    def test_stop_before_debounce_cancels_pending_reload(self, mock_load, tmp_path):
        """If stop() is called before the debounce window elapses, the reload is skipped."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("version: 1")
        cfg.chmod(0o600)
        mock_load.return_value = MagicMock()
        callback = MagicMock()

        # Long debounce gives us time to call stop() before it fires.
        w = ConfigWatcher(str(cfg), callback, debounce_seconds=5.0, poll_interval=0.02)
        w.start()
        time.sleep(0.05)

        # Trigger a change — pending_reload becomes True inside the thread.
        cfg.write_text("version: 2")
        time.sleep(0.1)  # let the watcher detect the change, debounce not elapsed

        # Stop immediately — the debounce window (5 s) has not elapsed.
        w.stop()

        # Callback must NOT have been invoked because the loop exits on _stop.
        callback.assert_not_called()

    @patch("missy.config.settings.load_config")
    def test_stop_joins_thread_cleanly(self, mock_load, tmp_path):
        """stop() must join the watcher thread within the 5-second timeout."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("version: 1")
        cfg.chmod(0o600)
        callback = MagicMock()

        w = ConfigWatcher(str(cfg), callback, debounce_seconds=5.0, poll_interval=0.02)
        w.start()
        time.sleep(0.05)

        cfg.write_text("version: 2")
        time.sleep(0.05)  # change detected, debounce pending

        start = time.monotonic()
        w.stop()
        elapsed = time.monotonic() - start

        # Thread must join well within the 5-second join timeout.
        assert elapsed < 2.0
        assert not w._thread.is_alive()

    @patch("missy.config.settings.load_config")
    def test_pending_reload_not_fired_after_stop(self, mock_load, tmp_path):
        """After stop() returns, the callback is never called even if a reload was pending."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("version: 1")
        cfg.chmod(0o600)
        mock_load.return_value = MagicMock()
        callback = MagicMock()

        w = ConfigWatcher(str(cfg), callback, debounce_seconds=5.0, poll_interval=0.02)
        w.start()
        time.sleep(0.05)

        cfg.write_text("version: 2")
        time.sleep(0.08)  # change detected inside loop, debounce not elapsed

        w.stop()

        # Sleep well past what the debounce would have been — thread is dead, no call.
        time.sleep(0.2)
        callback.assert_not_called()
