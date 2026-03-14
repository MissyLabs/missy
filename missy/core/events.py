"""Audit event bus for the Missy framework.

All policy decisions and significant runtime actions are published as
:class:`AuditEvent` instances through the module-level :data:`event_bus`
singleton.  Consumers subscribe to specific event types via callbacks.

Example::

    from missy.core.events import event_bus, AuditEvent

    def on_deny(event: AuditEvent) -> None:
        print(f"DENY: {event.category} - {event.detail}")

    event_bus.subscribe("policy.deny", on_deny)
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Allowed values for AuditEvent.category
EventCategory = Literal[
    "network", "filesystem", "shell", "plugin", "scheduler", "provider", "channel"
]

# Allowed values for AuditEvent.result
EventResult = Literal["allow", "deny", "error"]

# Callback signature
EventCallback = Callable[["AuditEvent"], None]


@dataclass
class AuditEvent:
    """An immutable record of a single auditable action.

    Attributes:
        timestamp: UTC time when the event was created.
        session_id: Identifier of the session that generated the event.
        task_id: Identifier of the task within the session.
        event_type: Dotted string describing the event
            (e.g. ``"network.request"``).
        category: Broad category of the action; one of ``network``,
            ``filesystem``, ``shell``, ``plugin``, ``scheduler``,
            ``provider``.
        result: Outcome of the action: ``allow``, ``deny``, or ``error``.
        detail: Arbitrary structured data associated with the event.
        policy_rule: Optional name of the policy rule that produced this
            result.
    """

    timestamp: datetime
    session_id: str
    task_id: str
    event_type: str
    category: EventCategory
    result: EventResult
    detail: dict[str, Any] = field(default_factory=dict)
    policy_rule: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("AuditEvent.timestamp must be timezone-aware.")

    @classmethod
    def now(
        cls,
        *,
        session_id: str,
        task_id: str,
        event_type: str,
        category: EventCategory,
        result: EventResult,
        detail: dict[str, Any] | None = None,
        policy_rule: str | None = None,
    ) -> AuditEvent:
        """Convenience constructor that fills in *timestamp* automatically.

        Args:
            session_id: Session that generated the event.
            task_id: Task within the session.
            event_type: Dotted event-type string.
            category: Broad action category.
            result: Action outcome.
            detail: Optional structured data for the event.
            policy_rule: Optional policy rule name.

        Returns:
            A new :class:`AuditEvent` timestamped to the current UTC time.
        """
        return cls(
            timestamp=datetime.now(tz=UTC),
            session_id=session_id,
            task_id=task_id,
            event_type=event_type,
            category=category,
            result=result,
            detail=detail or {},
            policy_rule=policy_rule,
        )


class EventBus:
    """Thread-safe publish/subscribe bus for :class:`AuditEvent` objects.

    Subscribers register a callback for a specific *event_type* string.
    Wildcards are not supported; use ``get_events`` for post-hoc filtering.

    The bus stores every published event in an in-memory log so that callers
    can replay or filter the audit trail at any time.

    Example::

        bus = EventBus()
        bus.subscribe("network.request", lambda e: print(e))
        bus.publish(AuditEvent.now(event_type="network.request", ...))
        events = bus.get_events(category="network")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[EventCallback]] = defaultdict(list)
        self._log: list[AuditEvent] = []

    def subscribe(self, event_type: str, callback: EventCallback) -> None:
        """Register *callback* to be called whenever *event_type* is published.

        Args:
            event_type: The event type string to listen for.
            callback: Callable that receives a single :class:`AuditEvent`
                argument.  Exceptions raised inside the callback are caught
                and logged; they do not propagate to the publisher.
        """
        with self._lock:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: EventCallback) -> None:
        """Remove a previously registered *callback* for *event_type*.

        No-op if the callback is not currently registered.

        Args:
            event_type: The event type string.
            callback: The callback to remove.
        """
        with self._lock:
            with contextlib.suppress(ValueError):
                self._subscribers[event_type].remove(callback)

    def publish(self, event: AuditEvent) -> None:
        """Append *event* to the internal log and dispatch it to subscribers.

        Subscriber callbacks are invoked synchronously in registration order.
        A callback that raises an exception is silently skipped after the
        exception is logged at ``ERROR`` level.

        Args:
            event: The audit event to publish.
        """
        with self._lock:
            self._log.append(event)
            callbacks = list(self._subscribers.get(event.event_type, []))

        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                logger.exception(
                    "Unhandled exception in EventBus callback %r for event type %r",
                    callback,
                    event.event_type,
                )

    def get_events(
        self,
        *,
        event_type: str | None = None,
        category: EventCategory | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        result: EventResult | None = None,
    ) -> list[AuditEvent]:
        """Return a filtered snapshot of the audit log.

        All provided keyword arguments are ANDed together.  Omitted arguments
        are not filtered on.

        Args:
            event_type: Include only events with this type string.
            category: Include only events in this category.
            session_id: Include only events from this session.
            task_id: Include only events from this task.
            result: Include only events with this result.

        Returns:
            A new list containing matching :class:`AuditEvent` instances in
            chronological order.
        """
        filters: list[tuple[str, Any]] = [
            ("event_type", event_type),
            ("category", category),
            ("session_id", session_id),
            ("task_id", task_id),
            ("result", result),
        ]
        active_filters = [(attr, val) for attr, val in filters if val is not None]

        with self._lock:
            snapshot = list(self._log)

        if not active_filters:
            return snapshot

        return [
            event
            for event in snapshot
            if all(getattr(event, attr) == val for attr, val in active_filters)
        ]

    def clear(self) -> None:
        """Discard all stored events and subscriber registrations.

        Primarily intended for use in tests.
        """
        with self._lock:
            self._log.clear()
            self._subscribers.clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Process-level event bus.  Import and use this instance directly.
event_bus: EventBus = EventBus()
