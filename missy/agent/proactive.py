"""Proactive task initiation via file-change and threshold triggers.

ProactiveManager runs in a background thread and monitors configured
triggers. When a trigger fires, it submits a synthetic prompt to the
agent (optionally gated by ApprovalGate for confirmation).

Supported trigger types:

- ``file_change``: fires when a watched file/directory is modified.
- ``disk_threshold``: fires when disk usage exceeds a percentage.
- ``load_threshold``: fires when system load average exceeds a value.
- ``schedule``: fires on a repeating interval (seconds).

Example::

    from missy.agent.proactive import ProactiveManager, ProactiveTrigger

    triggers = [
        ProactiveTrigger(
            name="watch-logs",
            trigger_type="file_change",
            watch_path="/var/log/app",
            watch_patterns=["*.log"],
            prompt_template="A log file changed in /var/log/app at {timestamp}.",
            cooldown_seconds=60,
        ),
    ]

    manager = ProactiveManager(triggers=triggers, agent_callback=my_agent_fn)
    manager.start()
    # ... later ...
    manager.stop()
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional watchdog import
# ---------------------------------------------------------------------------

try:
    from watchdog.events import PatternMatchingEventHandler
    from watchdog.observers import Observer

    _WATCHDOG_AVAILABLE = True
except ImportError:
    _WATCHDOG_AVAILABLE = False
    logger.warning(
        "proactive: 'watchdog' package not installed — file_change triggers are disabled. "
        "Install with: pip install watchdog"
    )


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProactiveTrigger:
    """Configuration for a single proactive trigger.

    Attributes:
        name: Unique identifier for this trigger.
        trigger_type: One of ``"file_change"``, ``"disk_threshold"``,
            ``"load_threshold"``, or ``"schedule"``.
        enabled: When ``False`` the trigger is skipped entirely.
        requires_confirmation: When ``True``, the trigger is gated through
            an :class:`~missy.agent.approval.ApprovalGate` before the
            agent callback is invoked.
        prompt_template: Template string for the synthetic prompt.  Supports
            ``{trigger_name}``, ``{trigger_type}``, and ``{timestamp}``
            substitution variables.
        watch_path: Path to watch for ``file_change`` triggers.
        watch_patterns: Glob patterns to match against, e.g. ``["*.log"]``.
            An empty list means all files.
        watch_recursive: Whether to watch subdirectories recursively.
        disk_path: Filesystem path to evaluate for ``disk_threshold`` triggers.
        disk_threshold_pct: Fire when ``disk_usage(disk_path).percent``
            exceeds this value.
        load_threshold: Fire when the 1-minute load average divided by
            ``os.cpu_count()`` exceeds this value (``load_threshold``
            triggers only).
        interval_seconds: Interval for ``schedule`` triggers and the polling
            cadence ceiling for threshold triggers.
        cooldown_seconds: Minimum seconds between two consecutive firings of
            this trigger.
    """

    name: str
    trigger_type: str
    enabled: bool = True
    requires_confirmation: bool = False
    prompt_template: str = ""
    # file_change options
    watch_path: str = ""
    watch_patterns: list = field(default_factory=list)
    watch_recursive: bool = False
    # threshold options
    disk_path: str = "/"
    disk_threshold_pct: float = 90.0
    load_threshold: float = 4.0
    # schedule trigger
    interval_seconds: int = 300
    # cooldown
    cooldown_seconds: int = 300


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ProactiveManager:
    """Monitor configured triggers and submit synthetic prompts to the agent.

    Args:
        triggers: List of :class:`ProactiveTrigger` instances to monitor.
        agent_callback: ``callable(prompt: str, session_id: str) -> str`` —
            called when a trigger fires with a non-empty, cooldown-clear
            prompt.  The return value is ignored.
        approval_gate: Optional :class:`~missy.agent.approval.ApprovalGate`
            instance.  Required for triggers with
            ``requires_confirmation=True``; if ``None`` those triggers are
            silently skipped.
    """

    def __init__(
        self,
        triggers: list[ProactiveTrigger],
        agent_callback: Callable[[str, str], Any],
        approval_gate: Optional[Any] = None,
    ) -> None:
        self._triggers = triggers
        self._agent_callback = agent_callback
        self._approval_gate = approval_gate

        # last_fired maps trigger.name -> unix timestamp of last fire
        self._last_fired: dict[str, float] = {}
        self._lock = threading.Lock()

        # Internal tracking for background threads / observers
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

        # Watchdog observer (created lazily, shared across file_change triggers)
        self._observer: Optional[Any] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all trigger monitors in daemon background threads/observers."""
        self._stop_event.clear()

        file_change_triggers = [
            t for t in self._triggers if t.enabled and t.trigger_type == "file_change"
        ]
        threshold_triggers = [
            t
            for t in self._triggers
            if t.enabled and t.trigger_type in ("disk_threshold", "load_threshold")
        ]
        schedule_triggers = [
            t for t in self._triggers if t.enabled and t.trigger_type == "schedule"
        ]

        # File-change triggers via watchdog
        if file_change_triggers:
            if not _WATCHDOG_AVAILABLE:
                logger.warning(
                    "proactive: skipping %d file_change trigger(s) — watchdog not installed",
                    len(file_change_triggers),
                )
            else:
                self._observer = Observer()
                for trigger in file_change_triggers:
                    if not trigger.watch_path:
                        logger.warning(
                            "proactive: trigger '%s' has no watch_path — skipped", trigger.name
                        )
                        continue
                    patterns = trigger.watch_patterns or ["*"]
                    handler = _ProactiveFileHandler(
                        trigger=trigger,
                        fire_fn=self._fire_trigger,
                        patterns=patterns,
                        ignore_directories=True,
                        case_sensitive=False,
                    )
                    self._observer.schedule(
                        handler,
                        path=trigger.watch_path,
                        recursive=trigger.watch_recursive,
                    )
                    logger.info(
                        "proactive: watching '%s' for trigger '%s'",
                        trigger.watch_path,
                        trigger.name,
                    )
                self._observer.start()

        # Threshold triggers — single polling thread
        if threshold_triggers:
            t = threading.Thread(
                target=self._threshold_loop,
                args=(threshold_triggers,),
                name="proactive-threshold",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        # Schedule triggers — one thread per trigger
        for trigger in schedule_triggers:
            t = threading.Thread(
                target=self._schedule_loop,
                args=(trigger,),
                name=f"proactive-schedule-{trigger.name}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        logger.info(
            "proactive: ProactiveManager started with %d trigger(s)",
            len([t for t in self._triggers if t.enabled]),
        )

    def stop(self) -> None:
        """Stop all monitors gracefully."""
        self._stop_event.set()

        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception as exc:
                logger.debug("proactive: observer stop error: %s", exc)
            self._observer = None

        for thread in self._threads:
            thread.join(timeout=5)
        self._threads.clear()

        logger.info("proactive: ProactiveManager stopped")

    def get_status(self) -> dict[str, Any]:
        """Return a snapshot of trigger names, types, enabled flags, and last_fired times.

        Returns:
            A dict with a ``"triggers"`` key containing a list of per-trigger
            status dicts, and an ``"active"`` boolean.
        """
        with self._lock:
            fired_copy = dict(self._last_fired)

        statuses = []
        for trigger in self._triggers:
            last = fired_copy.get(trigger.name)
            statuses.append(
                {
                    "name": trigger.name,
                    "trigger_type": trigger.trigger_type,
                    "enabled": trigger.enabled,
                    "last_fired": (
                        datetime.fromtimestamp(last, tz=timezone.utc).isoformat()
                        if last
                        else None
                    ),
                }
            )
        return {
            "active": not self._stop_event.is_set(),
            "triggers": statuses,
        }

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    def _threshold_loop(self, triggers: list[ProactiveTrigger]) -> None:
        """Poll disk and load thresholds until the stop event is set."""
        # Use the shortest interval among triggers (floor 5 s, ceiling 30 s).
        intervals = [t.interval_seconds for t in triggers if t.interval_seconds > 0]
        poll_interval = min(max(min(intervals, default=30), 5), 30)

        while not self._stop_event.wait(timeout=poll_interval):
            for trigger in triggers:
                if self._stop_event.is_set():
                    break
                try:
                    if trigger.trigger_type == "disk_threshold":
                        usage = shutil.disk_usage(trigger.disk_path or "/")
                        pct = usage.used / usage.total * 100.0
                        if pct > trigger.disk_threshold_pct:
                            self._fire_trigger(trigger)
                    elif trigger.trigger_type == "load_threshold":
                        load1 = os.getloadavg()[0]
                        cpu_count = os.cpu_count() or 1
                        normalised = load1 / cpu_count
                        if normalised > trigger.load_threshold:
                            self._fire_trigger(trigger)
                except Exception as exc:
                    logger.debug(
                        "proactive: threshold check error for '%s': %s", trigger.name, exc
                    )

    def _schedule_loop(self, trigger: ProactiveTrigger) -> None:
        """Fire *trigger* on a repeating interval until the stop event is set."""
        interval = max(trigger.interval_seconds, 1)
        while not self._stop_event.wait(timeout=interval):
            if self._stop_event.is_set():
                break
            self._fire_trigger(trigger)

    # ------------------------------------------------------------------
    # Trigger firing
    # ------------------------------------------------------------------

    def _fire_trigger(self, trigger: ProactiveTrigger) -> None:
        """Evaluate cooldown and dispatch the trigger to the agent callback.

        Steps:
        1. Check cooldown — return early if within the cooldown window.
        2. Record last_fired.
        3. Build prompt from ``prompt_template``.
        4. If ``requires_confirmation``, gate through ApprovalGate.
        5. Emit audit event.
        6. Call agent_callback, catching all exceptions.
        """
        now = time.time()

        # 1. Cooldown check
        with self._lock:
            last = self._last_fired.get(trigger.name, 0.0)
            if now - last < trigger.cooldown_seconds:
                return
            # 2. Record last_fired inside the lock to prevent double-fire races.
            self._last_fired[trigger.name] = now

        # 3. Build prompt
        ts = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        template = trigger.prompt_template or (
            "Proactive trigger '{trigger_name}' of type '{trigger_type}' fired at {timestamp}."
        )
        prompt = template.format(
            trigger_name=trigger.name,
            trigger_type=trigger.trigger_type,
            timestamp=ts,
        )

        # 4. Confirmation gate
        if trigger.requires_confirmation:
            if self._approval_gate is None:
                logger.warning(
                    "proactive: trigger '%s' requires_confirmation but no ApprovalGate provided"
                    " — skipping",
                    trigger.name,
                )
                self._emit_audit(
                    trigger,
                    "deny",
                    prompt,
                    {"reason": "no_approval_gate"},
                )
                return
            try:
                self._approval_gate.request(
                    action=trigger.name,
                    reason=prompt,
                    risk="medium",
                )
            except Exception as exc:
                logger.info(
                    "proactive: trigger '%s' denied or timed out: %s", trigger.name, exc
                )
                self._emit_audit(
                    trigger,
                    "deny",
                    prompt,
                    {"reason": str(exc)},
                )
                return

        # 5. Emit audit event
        self._emit_audit(trigger, "allow", prompt, {})

        # 6. Invoke agent callback
        session_id = f"proactive-{trigger.name}"
        try:
            self._agent_callback(prompt, session_id)
        except Exception as exc:
            logger.error(
                "proactive: agent_callback error for trigger '%s': %s", trigger.name, exc
            )

    def _emit_audit(
        self,
        trigger: ProactiveTrigger,
        result: str,
        prompt: str,
        extra: dict[str, Any],
    ) -> None:
        """Publish an audit event for a trigger firing attempt."""
        try:
            from missy.core.events import AuditEvent, event_bus

            detail: dict[str, Any] = {
                "trigger_name": trigger.name,
                "trigger_type": trigger.trigger_type,
                "prompt_preview": prompt[:100],
            }
            detail.update(extra)
            event = AuditEvent.now(
                session_id=f"proactive-{trigger.name}",
                task_id="proactive",
                event_type="agent.proactive.trigger_fired",
                category="plugin",
                result=result,  # type: ignore[arg-type]
                detail=detail,
            )
            event_bus.publish(event)
        except Exception as exc:
            logger.debug("proactive: audit emit failed: %s", exc)


# ---------------------------------------------------------------------------
# Watchdog event handler helper
# ---------------------------------------------------------------------------


if _WATCHDOG_AVAILABLE:

    class _ProactiveFileHandler(PatternMatchingEventHandler):  # type: ignore[misc]
        """Watchdog handler that calls *fire_fn* on any filesystem event."""

        def __init__(
            self,
            trigger: ProactiveTrigger,
            fire_fn: Callable[[ProactiveTrigger], None],
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self._trigger = trigger
            self._fire_fn = fire_fn

        def on_any_event(self, event: Any) -> None:  # type: ignore[override]
            self._fire_fn(self._trigger)

else:
    # Provide a stub so imports don't fail when watchdog is absent.
    class _ProactiveFileHandler:  # type: ignore[no-redef]
        pass
