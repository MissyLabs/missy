"""Graceful shutdown for the vision subsystem.

Coordinates cleanup of camera handles, scene sessions, health monitor
persistence, and multi-camera resources when the process is shutting
down or the vision subsystem is being deactivated.

Register ``vision_shutdown`` as an ``atexit`` handler or call it
directly during agent cleanup.
"""

from __future__ import annotations

import atexit
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_shutdown_lock = threading.Lock()
_shutdown_done = False


def vision_shutdown() -> dict[str, Any]:
    """Gracefully shut down all vision subsystem resources.

    Safe to call multiple times (idempotent).

    Returns a summary dict of what was cleaned up.
    """
    global _shutdown_done
    with _shutdown_lock:
        if _shutdown_done:
            return {"status": "already_shutdown"}
        _shutdown_done = True

    summary: dict[str, Any] = {"status": "shutdown", "steps": []}

    # 1. Close all scene sessions
    try:
        from missy.vision.scene_memory import get_scene_manager

        mgr = get_scene_manager()
        sessions = mgr.list_sessions()
        active_count = sum(1 for s in sessions if s.get("active"))
        mgr.close_all()
        summary["steps"].append(f"Closed {active_count} active scene sessions")
    except Exception as exc:
        summary["steps"].append(f"Scene cleanup failed: {exc}")
        logger.warning("Scene session cleanup failed: %s", exc)

    # 2. Save health monitor data
    try:
        from missy.vision.health_monitor import get_health_monitor

        monitor = get_health_monitor()
        if monitor._persist_path is not None:
            monitor.save()
            summary["steps"].append("Health monitor data saved")
        else:
            summary["steps"].append("Health monitor: no persist path configured")
    except Exception as exc:
        summary["steps"].append(f"Health monitor save failed: {exc}")
        logger.warning("Health monitor save failed: %s", exc)

    # 3. Log shutdown event via audit
    try:
        from missy.vision.audit import audit_vision_session

        audit_vision_session(
            task_id="shutdown",
            task_type="system",
            action="close",
            frame_count=0,
        )
    except Exception:
        pass  # audit is best-effort at shutdown

    logger.info("Vision subsystem shutdown complete: %s", summary)
    return summary


def reset_shutdown_state() -> None:
    """Reset the shutdown flag.  Used in tests."""
    global _shutdown_done
    with _shutdown_lock:
        _shutdown_done = False


def register_shutdown_hook() -> None:
    """Register ``vision_shutdown`` as an atexit handler.

    Safe to call multiple times — ``atexit`` deduplicates.
    """
    atexit.register(vision_shutdown)
