"""Heartbeat system: periodic agent invocation from HEARTBEAT.md checklist."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

HEARTBEAT_FILE = "HEARTBEAT.md"
HEARTBEAT_OK_FILE = "HEARTBEAT_OK"


class HeartbeatRunner:
    """Runs a periodic heartbeat by sending HEARTBEAT.md as a synthetic task.

    Args:
        agent_run_fn: Callable(prompt: str) -> str — runs a single agent task.
        interval_seconds: How often to fire (default 1800 = 30 minutes).
        workspace: Path to workspace directory.
        active_hours: Optional "HH:MM-HH:MM" window (empty = always).
        report_fn: Optional callable to deliver the heartbeat result.
    """

    def __init__(
        self,
        agent_run_fn: Callable[[str], str],
        interval_seconds: int = 1800,
        workspace: str = "~/workspace",
        active_hours: str = "",
        report_fn: Callable[[str], None] | None = None,
    ):
        self._run = agent_run_fn
        self._interval = interval_seconds
        self._workspace = Path(workspace).expanduser()
        self._active_hours = active_hours
        self._report = report_fn
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._runs = 0
        self._skips = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="missy-heartbeat")
        self._thread.start()
        logger.info("Heartbeat started (interval=%ds)", self._interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._fire()

    def _fire(self) -> None:
        # Check HEARTBEAT_OK suppression file
        ok_file = self._workspace / HEARTBEAT_OK_FILE
        if ok_file.exists():
            ok_file.unlink(missing_ok=True)
            logger.info("Heartbeat suppressed by HEARTBEAT_OK file")
            self._skips += 1
            return

        # Check active hours
        if self._active_hours and not self._in_active_hours():
            logger.debug("Heartbeat outside active_hours; skipping")
            self._skips += 1
            return

        # Load checklist
        checklist_path = self._workspace / HEARTBEAT_FILE
        if not checklist_path.exists():
            logger.debug("No HEARTBEAT.md found at %s; skipping", checklist_path)
            self._skips += 1
            return

        checklist = checklist_path.read_text(encoding="utf-8", errors="replace")
        prompt = f"[HEARTBEAT CHECK]\n\n{checklist}"

        try:
            result = self._run(prompt)
            self._runs += 1
            logger.info("Heartbeat completed (run #%d)", self._runs)
            if self._report:
                self._report(result)
        except Exception as exc:
            logger.error("Heartbeat run failed: %s", exc)

    def _in_active_hours(self) -> bool:
        from datetime import datetime

        m = re.match(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})", self._active_hours)
        if not m:
            return True
        now = datetime.now()
        start = now.replace(hour=int(m[1]), minute=int(m[2]), second=0, microsecond=0)
        end = now.replace(hour=int(m[3]), minute=int(m[4]), second=0, microsecond=0)
        if end < start:
            return now >= start or now <= end
        return start <= now <= end

    @property
    def metrics(self) -> dict:
        return {"runs": self._runs, "skips": self._skips}
