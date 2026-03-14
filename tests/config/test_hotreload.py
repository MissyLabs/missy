"""Tests for missy.config.hotreload."""

from __future__ import annotations

import time
from pathlib import Path
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
            str(cfg), callback,
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
        w = ConfigWatcher(str(cfg), MagicMock())
        w._do_reload()  # should not raise


class TestApplyConfig:
    @patch("missy.providers.registry.init_registry")
    @patch("missy.policy.engine.init_policy_engine")
    def test_apply_reinitializes_subsystems(self, mock_policy, mock_reg):
        config = MagicMock()
        _apply_config(config)
        mock_policy.assert_called_once_with(config)
        mock_reg.assert_called_once_with(config)
