"""Background watchdog that monitors subsystem health."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class SubsystemHealth:
    name: str
    healthy: bool = True
    consecutive_failures: int = 0
    last_checked: float = 0.0
    last_error: str = ""


class Watchdog:
    """Periodically checks subsystem health and emits audit events.

    Args:
        check_interval: Seconds between health checks (default 60).
        failure_threshold: Failures before ERROR log (default 3).
    """

    def __init__(self, check_interval: float = 60.0, failure_threshold: int = 3):
        self._interval = check_interval
        self._threshold = failure_threshold
        self._checks: dict[str, Callable[[], bool]] = {}
        self._health: dict[str, SubsystemHealth] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def register(self, name: str, check_fn: Callable[[], bool]) -> None:
        """Register a health check function.

        Args:
            name: Subsystem name.
            check_fn: Returns True if healthy, False if not.
        """
        self._checks[name] = check_fn
        self._health[name] = SubsystemHealth(name=name)

    def start(self) -> None:
        """Start background health monitoring."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="missy-watchdog")
        self._thread.start()
        logger.info("Watchdog started (interval=%ds)", self._interval)

    def stop(self) -> None:
        """Stop background health monitoring."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._check_all()

    def _check_all(self) -> None:
        from missy.core.events import AuditEvent, event_bus
        for name, fn in self._checks.items():
            h = self._health[name]
            prev_healthy = h.healthy
            try:
                ok = fn()
                h.last_checked = time.monotonic()
                if ok:
                    if not prev_healthy:
                        logger.info("Watchdog: %s recovered", name)
                    h.healthy = True
                    h.consecutive_failures = 0
                    h.last_error = ""
                else:
                    raise RuntimeError("health check returned False")
            except Exception as exc:
                h.healthy = False
                h.consecutive_failures += 1
                h.last_error = str(exc)
                h.last_checked = time.monotonic()
                level = logging.ERROR if h.consecutive_failures >= self._threshold else logging.WARNING
                logger.log(level, "Watchdog: %s unhealthy (failures=%d): %s", name, h.consecutive_failures, exc)

            try:
                event = AuditEvent.now(
                    session_id="watchdog",
                    task_id=name,
                    event_type="watchdog.health_check",
                    category="plugin",
                    result="allow" if h.healthy else "error",
                    detail={"subsystem": name, "healthy": h.healthy, "failures": h.consecutive_failures},
                )
                event_bus.publish(event)
            except Exception:
                pass

    def get_report(self) -> dict:
        """Return a health report for all subsystems."""
        return {
            name: {
                "healthy": h.healthy,
                "consecutive_failures": h.consecutive_failures,
                "last_error": h.last_error,
            }
            for name, h in self._health.items()
        }
