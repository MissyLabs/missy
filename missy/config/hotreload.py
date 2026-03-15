"""Config hot-reload: watch config.yaml for changes and re-apply policy."""

from __future__ import annotations

import logging
import os
import stat
import threading
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigWatcher:
    """Watches a config file and triggers reload when it changes.

    Args:
        config_path: Path to the YAML config file.
        reload_fn: Callable invoked with the new config on change.
        debounce_seconds: Wait this long after last change before reloading (default 2).
        poll_interval: File stat check interval in seconds (default 1).
    """

    def __init__(
        self,
        config_path: str,
        reload_fn: Callable,
        debounce_seconds: float = 2.0,
        poll_interval: float = 1.0,
    ):
        self._path = Path(config_path).expanduser()
        self._reload_fn = reload_fn
        self._debounce = debounce_seconds
        self._poll = poll_interval
        self._last_mtime: float = 0.0
        self._last_change_time: float = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        """Start the background file watcher."""
        try:
            self._last_mtime = self._path.stat().st_mtime
        except OSError:
            self._last_mtime = 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._watch, daemon=True, name="missy-hotreload")
        self._thread.start()
        logger.info("ConfigWatcher: watching %s", self._path)

    def stop(self) -> None:
        """Stop the background file watcher."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _watch(self) -> None:
        pending_reload = False
        while not self._stop.wait(self._poll):
            try:
                mtime = self._path.stat().st_mtime
            except OSError:
                continue

            if mtime != self._last_mtime:
                self._last_mtime = mtime
                self._last_change_time = time.monotonic()
                pending_reload = True
                logger.debug("ConfigWatcher: change detected in %s", self._path)

            if pending_reload and (time.monotonic() - self._last_change_time) >= self._debounce:
                pending_reload = False
                self._do_reload()

    def _check_file_safety(self) -> bool:
        """Verify config file ownership and permissions before reload.

        Returns True if the file is safe to load, False otherwise.
        Rejects symlinks, files not owned by the current user, and
        files that are group- or world-writable.
        """
        try:
            if self._path.is_symlink():
                logger.warning(
                    "ConfigWatcher: %s is a symlink; refusing to reload", self._path
                )
                return False
            st = self._path.stat()
            if st.st_uid != os.getuid():
                logger.warning(
                    "ConfigWatcher: %s is owned by uid %d, expected %d; refusing to reload",
                    self._path,
                    st.st_uid,
                    os.getuid(),
                )
                return False
            if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                logger.warning(
                    "ConfigWatcher: %s is group- or world-writable (mode %o); refusing to reload",
                    self._path,
                    st.st_mode,
                )
                return False
        except OSError as exc:
            logger.warning("ConfigWatcher: cannot stat %s: %s", self._path, exc)
            return False
        return True

    def _do_reload(self) -> None:
        logger.info("ConfigWatcher: reloading %s", self._path)
        if not self._check_file_safety():
            return
        try:
            from missy.config.settings import load_config

            new_config = load_config(str(self._path))
            self._reload_fn(new_config)
            logger.info("ConfigWatcher: reload complete")
        except Exception as exc:
            logger.error("ConfigWatcher: reload failed: %s", exc)


def _apply_config(new_config) -> None:
    """Re-initialise subsystems with updated config."""
    from missy.policy.engine import init_policy_engine
    from missy.providers.registry import init_registry

    init_policy_engine(new_config)
    init_registry(new_config)
    logger.info("ConfigWatcher: policy engine and provider registry updated")
