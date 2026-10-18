"""Startup and runtime status of gateway subsystems.

Each optional subsystem (voice, screencast, scheduler, web console, Discord,
...) used to fail at startup with only a yellow console line while the
gateway kept running and systemd reported "active". This registry records
what actually happened so ``missy gateway status`` (a separate process, via
the status file) and ``/diagnostics`` (in process) can report it.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

STATUS_PATH = "~/.missy/run/gateway_status.json"

RUNNING = "running"
FAILED = "failed"
DISABLED = "disabled"


class SubsystemStatusRegistry:
    """Thread-safe record of each subsystem's latest state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self.started_at = time.time()

    def record(self, name: str, state: str, detail: str = "") -> None:
        with self._lock:
            self._entries[name] = {"state": state, "detail": detail, "updated_at": time.time()}

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: dict(entry) for name, entry in self._entries.items()}

    def failed(self) -> dict[str, dict[str, Any]]:
        return {n: e for n, e in self.snapshot().items() if e["state"] == FAILED}

    def write(self, path: str = STATUS_PATH) -> None:
        """Atomically write the status file (owner-only)."""
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            "pid": os.getpid(),
            "started_at": self.started_at,
            "written_at": time.time(),
            "subsystems": self.snapshot(),
        }
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".gateway_status.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise


_registry = SubsystemStatusRegistry()


def get_status_registry() -> SubsystemStatusRegistry:
    return _registry


def read_status_file(path: str = STATUS_PATH) -> dict[str, Any] | None:
    """Return the gateway's last written status, or ``None`` if absent/unreadable."""
    try:
        return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def pid_is_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
