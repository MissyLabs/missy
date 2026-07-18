"""Continuous vision monitoring (F23).

Vision is on-demand today (``vision_capture``/``vision_analyze``). ``VisionMonitor``
turns the camera into a passive sensor: it periodically captures a frame, uses
:class:`~missy.vision.scene_memory.SceneSession`'s change detection to compare it
against the previous frame, and fires an alert callback when the change exceeds a
threshold — with a cooldown so one event doesn't spam. An operator can wire that
callback to a Discord notification, an approval-gated agent run, etc.

The capture source is injected as ``capture_fn() -> np.ndarray | None`` so the
monitor is decoupled from the camera stack and fully unit-testable; a failed or
``None`` capture is skipped, never fatal. Runs as a daemon thread, like the
heartbeat.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class VisionMonitor:
    """Periodically capture frames and alert on significant scene change.

    Args:
        capture_fn: Returns the current frame as a numpy array, or ``None`` to
            skip this tick (e.g. a transient capture failure).
        on_change: Called with the :class:`SceneChange` when change_score meets
            ``threshold`` (subject to the cooldown). Errors are swallowed.
        threshold: Minimum change_score (0..1) to consider "significant".
        interval_seconds: Seconds between captures in the background loop.
        min_alert_gap_seconds: Cooldown between alerts (0 disables the cooldown).
        session: An optional pre-built ``SceneSession``; one is created lazily
            otherwise.
    """

    def __init__(
        self,
        capture_fn: Callable[[], Any],
        on_change: Callable[[Any], None],
        *,
        threshold: float = 0.15,
        interval_seconds: float = 60.0,
        min_alert_gap_seconds: float = 300.0,
        session: Any = None,
    ) -> None:
        self._capture_fn = capture_fn
        self._on_change = on_change
        self.threshold = threshold
        self.interval_seconds = max(1.0, interval_seconds)
        self.min_alert_gap_seconds = max(0.0, min_alert_gap_seconds)
        self._session = session
        self._prev_frame: Any = None
        # None = "never alerted". Must not be 0.0: time.monotonic() can be small
        # on a freshly-booted host, so `now - 0.0 < cooldown` would wrongly
        # suppress the very first alert.
        self._last_alert_at: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.checks = 0
        self.alerts = 0

    def _get_session(self) -> Any:
        if self._session is None:
            from missy.vision.scene_memory import SceneSession

            self._session = SceneSession(task_id="vision-monitor")
        return self._session

    def check_once(self) -> Any:
        """Capture one frame, detect change, and alert if significant.

        Returns the :class:`SceneChange` when an alert fired this tick, else
        ``None``. Never raises.
        """
        self.checks += 1
        try:
            image = self._capture_fn()
        except Exception:
            logger.debug("VisionMonitor: capture_fn raised; skipping tick.", exc_info=True)
            return None
        if image is None:
            return None

        try:
            session = self._get_session()
            frame = session.add_frame(image, source="monitor", deduplicate=False)
        except Exception:
            logger.debug("VisionMonitor: add_frame failed; skipping tick.", exc_info=True)
            return None
        if frame is None:
            return None

        prev = self._prev_frame
        self._prev_frame = frame
        if prev is None:
            return None  # first frame — nothing to compare against

        try:
            change = session.detect_change(prev, frame)
        except Exception:
            logger.debug("VisionMonitor: detect_change failed.", exc_info=True)
            return None

        score = float(getattr(change, "change_score", -1.0))
        if score < self.threshold:
            return None

        now = time.monotonic()
        if (
            self.min_alert_gap_seconds
            and self._last_alert_at is not None
            and (now - self._last_alert_at) < self.min_alert_gap_seconds
        ):
            logger.debug("VisionMonitor: change %.3f within cooldown; suppressing.", score)
            return None

        self._last_alert_at = now
        self.alerts += 1
        try:
            self._on_change(change)
        except Exception:
            logger.debug("VisionMonitor: on_change callback raised.", exc_info=True)
        return change

    # -- background loop ----------------------------------------------------

    def start(self) -> None:
        """Start the monitoring daemon thread (no-op if already running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="missy-vision-monitor", daemon=True)
        self._thread.start()
        logger.info("VisionMonitor started (interval=%.0fs).", self.interval_seconds)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.check_once()

    def stats(self) -> dict[str, Any]:
        return {
            "checks": self.checks,
            "alerts": self.alerts,
            "threshold": self.threshold,
            "interval_seconds": self.interval_seconds,
        }
