"""Event-driven async message bus for the Missy framework.

:class:`MessageBus` decouples channels, the agent runtime, and tools via
typed message routing with wildcard topic matching.  It supports both
synchronous dispatch (``publish``) and queued async dispatch
(``publish_async``).

Usage::

    from missy.core.message_bus import init_message_bus, get_message_bus
    from missy.core.bus_topics import AGENT_RUN_START

    bus = init_message_bus()
    bus.subscribe("agent.*", lambda msg: print(msg.topic))
    bus.publish(BusMessage(
        topic=AGENT_RUN_START,
        payload={"user_input_length": 42},
        source="agent",
    ))
"""

from __future__ import annotations

import contextlib
import heapq
import json
import logging
import queue
import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatch

from missy.core.bus_topics import ALL_TOPICS

logger = logging.getLogger(__name__)

# Callback type for subscribers.
BusHandler = Callable[["BusMessage"], None]

_TOPIC_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_ENDPOINT_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")
_MAX_PAYLOAD_BYTES = 256 * 1024
_MAX_PRIORITY_BURST = 8
_REQUIRED_PAYLOAD_FIELDS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "agent.run.start": {"session_id": str, "task_id": str, "user_input_length": int},
    "agent.run.complete": {
        "session_id": str,
        "task_id": str,
        "provider": str,
        "tools_used": list,
        "cost": dict,
    },
    "agent.run.error": {"session_id": str, "task_id": str, "error": str},
    "tool.request": {"tool": str, "session_id": str, "task_id": str},
    "tool.result": {"tool": str, "is_error": bool, "session_id": str, "task_id": str},
    "sleeptime.error": {"detail": str},
}


@dataclass
class BusMessage:
    """A single message routed through the :class:`MessageBus`.

    Attributes:
        topic: Dotted topic string (e.g. ``"channel.inbound"``).
        payload: Arbitrary message-specific data.
        source: Originator identifier (e.g. ``"cli"``, ``"agent"``).
        target: Optional specific recipient identifier.
        message_id: Unique message identifier (auto-generated).
        correlation_id: Links request/response pairs together.
        timestamp: ISO 8601 creation time (auto-generated).
        priority: Dispatch priority — ``0`` = normal, ``1`` = high,
            ``2`` = urgent.  Higher values are dispatched first in the
            async queue.
    """

    topic: str
    payload: dict
    source: str
    target: str | None = None
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    priority: int = 0


class MessageBus:
    """Thread-safe publish/subscribe message bus with wildcard topic matching.

    Topics use ``fnmatch``-style patterns so ``"channel.*"`` matches
    ``"channel.inbound"`` and ``"channel.outbound"``.

    The bus supports two dispatch modes:

    * **Synchronous** — :meth:`publish` calls handlers immediately in the
      calling thread, in registration order.
    * **Asynchronous** — :meth:`publish_async` enqueues the message for
      dispatch by a background worker started via :meth:`start`.

    Exceptions in subscriber handlers are caught and logged; they never
    propagate to the publisher.
    """

    def __init__(self, max_queue_size: int = 1000, *, strict_topics: bool = False) -> None:
        if isinstance(max_queue_size, bool) or not isinstance(max_queue_size, int):
            raise TypeError("max_queue_size must be an integer")
        if not 1 <= max_queue_size <= 100_000:
            raise ValueError("max_queue_size must be between 1 and 100000")
        self._lock = threading.Lock()
        self._dispatch_lock = threading.Lock()
        self._strict_topics = strict_topics
        # Mapping of topic pattern → list of handlers.
        self._subscribers: dict[str, list[BusHandler]] = {}
        # Priority queue for async dispatch.  Items are
        # ``(-priority, sequence, message)`` so that higher priority values
        # sort first and FIFO order is preserved within the same priority.
        self._queue: queue.PriorityQueue[tuple[int, int, BusMessage]] = queue.PriorityQueue(
            maxsize=max_queue_size
        )
        self._seq = 0  # monotonic counter for stable sort within same priority
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._accepting = True
        self._accepted = 0
        self._dropped = 0
        self._delivered = 0
        self._handler_errors = 0
        self._priority_burst = 0

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, topic: str, handler: BusHandler) -> None:
        """Register *handler* for messages matching *topic*.

        *topic* may contain ``fnmatch`` wildcards (e.g. ``"channel.*"``).
        Handlers are called in registration order.

        Args:
            topic: Topic pattern to match.
            handler: Callable receiving a :class:`BusMessage`.
        """
        _validate_subscription_pattern(topic, strict=self._strict_topics)
        if not callable(handler):
            raise TypeError("handler must be callable")
        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)

    def unsubscribe(self, topic: str, handler: BusHandler) -> None:
        """Remove a previously registered *handler* for *topic*.

        No-op if the handler is not currently registered.

        Args:
            topic: The topic pattern the handler was registered with.
            handler: The handler to remove.
        """
        with self._lock:
            handlers = self._subscribers.get(topic)
            if handlers is not None:
                with contextlib.suppress(ValueError):
                    handlers.remove(handler)
                if not handlers:
                    del self._subscribers[topic]

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, message: BusMessage) -> None:
        """Dispatch *message* synchronously to all matching subscribers.

        Handlers are invoked in registration order.  Exceptions in handlers
        are caught and logged — they never propagate to the caller.

        Args:
            message: The message to dispatch.
        """
        self._validate_message(message)
        handlers = self._resolve_handlers(message.topic)
        for handler in handlers:
            try:
                handler(message)
            except Exception:
                with self._lock:
                    self._handler_errors += 1
                logger.exception(
                    "Unhandled exception in MessageBus handler %r for topic %r",
                    handler,
                    message.topic,
                )
        with self._lock:
            self._delivered += 1

    def publish_async(self, message: BusMessage) -> None:
        """Enqueue *message* for asynchronous dispatch.

        The message will be dispatched by the background worker started
        via :meth:`start`, or can be flushed synchronously with
        :meth:`drain`.

        Args:
            message: The message to enqueue.

        Raises:
            queue.Full: If the internal queue has reached ``max_queue_size``.
        """
        self._validate_message(message)
        with self._lock:
            if not self._accepting:
                raise RuntimeError("MessageBus is stopping and no longer accepts messages.")
            seq = self._seq
            self._seq += 1
        # Negate priority so higher values sort first in the min-heap.
        try:
            self._queue.put_nowait((-message.priority, seq, message))
        except queue.Full:
            with self._lock:
                self._dropped += 1
            raise
        with self._lock:
            self._accepted += 1

    # ------------------------------------------------------------------
    # Async worker lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background dispatch worker thread.

        The worker pulls messages from the async queue and dispatches them
        via :meth:`publish`.  Call :meth:`stop` to shut it down.
        """
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        with self._lock:
            self._accepting = True
        self._worker = threading.Thread(
            target=self._dispatch_loop,
            name="missy-message-bus",
            daemon=True,
        )
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background worker and drain remaining messages.

        Args:
            timeout: Maximum seconds to wait for the worker thread to
                finish after the stop signal is set.
        """
        self._stop_event.set()
        with self._lock:
            self._accepting = False
            worker = self._worker
        if worker is not None:
            worker.join(timeout=timeout)
            if worker.is_alive():
                logger.warning("MessageBus worker did not stop within %.3fs", timeout)
                return
        self.drain()
        with self._lock:
            self._worker = None

    def drain(self) -> None:
        """Process all currently queued messages synchronously.

        Useful for testing and for ensuring delivery during shutdown.
        """
        with self._dispatch_lock:
            while True:
                try:
                    _neg_pri, _seq, message = self._next_queued_message()
                except queue.Empty:
                    break
                try:
                    self.publish(message)
                finally:
                    self._queue.task_done()

    def pending_count(self) -> int:
        """Return the number of messages waiting in the async queue."""
        return self._queue.qsize()

    def stats(self) -> dict[str, int | bool]:
        """Return bounded queue/delivery counters for diagnostics."""
        with self._lock:
            return {
                "accepted": self._accepted,
                "dropped": self._dropped,
                "delivered": self._delivered,
                "handler_errors": self._handler_errors,
                "pending": self._queue.qsize(),
                "accepting": self._accepting,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_handlers(self, topic: str) -> list[BusHandler]:
        """Return a snapshot of handlers whose patterns match *topic*."""
        matched: list[BusHandler] = []
        with self._lock:
            for pattern, handlers in self._subscribers.items():
                if fnmatch(topic, pattern):
                    matched.extend(handlers)
        return matched

    def _validate_message(self, message: BusMessage) -> None:
        if not isinstance(message, BusMessage):
            raise TypeError("message must be a BusMessage")
        if (
            not isinstance(message.topic, str)
            or not 0 <= len(message.topic) <= 256
            or (self._strict_topics and not message.topic)
            or any(char in message.topic for char in "*?[]")
            or any(not char.isprintable() or char in "\r\n\x00" for char in message.topic)
        ):
            raise ValueError("Published topic must be a bounded concrete topic name.")
        if self._strict_topics and not _TOPIC_RE.fullmatch(message.topic):
            raise ValueError("Production topics must use the dotted topic grammar.")
        if self._strict_topics and message.topic not in ALL_TOPICS:
            raise ValueError(f"Unknown production message topic {message.topic!r}.")
        if not isinstance(message.source, str) or not _ENDPOINT_RE.fullmatch(message.source):
            raise ValueError("Message source must be a bounded endpoint identifier.")
        if message.target is not None and (
            not isinstance(message.target, str) or not _ENDPOINT_RE.fullmatch(message.target)
        ):
            raise ValueError("Message target must be a bounded endpoint identifier.")
        if (
            isinstance(message.priority, bool)
            or not isinstance(message.priority, int)
            or not 0 <= message.priority <= 2
        ):
            raise ValueError("Message priority must be one of 0, 1, or 2.")
        if not isinstance(message.payload, dict):
            raise TypeError("Message payload must be a dict.")
        try:
            encoded = json.dumps(message.payload, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Message payload must be finite JSON data.") from exc
        if len(encoded.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
            raise ValueError("Message payload exceeds the 256 KiB limit.")
        required = _REQUIRED_PAYLOAD_FIELDS.get(message.topic, {}) if self._strict_topics else {}
        for field_name, field_type in required.items():
            value = message.payload.get(field_name)
            if isinstance(value, bool) and field_type is int:
                valid = False
            else:
                valid = isinstance(value, field_type)
            if not valid:
                raise ValueError(
                    f"Message payload field {field_name!r} has the wrong contract type."
                )

    def _dispatch_loop(self) -> None:
        """Background worker loop: pull from queue and dispatch."""
        while not self._stop_event.is_set():
            with self._dispatch_lock:
                try:
                    _neg_pri, _seq, message = self._next_queued_message(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self.publish(message)
                finally:
                    self._queue.task_done()

    def _next_queued_message(self, timeout: float | None = None) -> tuple[int, int, BusMessage]:
        """Dequeue with FIFO priority and bounded starvation."""
        if self._priority_burst < _MAX_PRIORITY_BURST:
            item = (
                self._queue.get(timeout=timeout)
                if timeout is not None
                else self._queue.get_nowait()
            )
        else:
            # PriorityQueue has no aging primitive. Under its own mutex,
            # select the oldest message from the lowest currently queued
            # priority once per burst, then restore heap order. This keeps
            # urgent traffic preemptive without letting a continuous urgent
            # producer starve normal accepted work forever.
            with self._queue.not_empty:
                if not self._queue.queue:
                    self._priority_burst = 0
                    raise queue.Empty
                index = max(
                    range(len(self._queue.queue)),
                    key=lambda idx: (
                        self._queue.queue[idx][0],
                        -self._queue.queue[idx][1],
                    ),
                )
                item = self._queue.queue.pop(index)
                heapq.heapify(self._queue.queue)
                self._queue.not_full.notify()
        if item[0] < 0:
            self._priority_burst += 1
        else:
            self._priority_burst = 0
        return item

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            n_subs = sum(len(h) for h in self._subscribers.values())
        return f"<MessageBus subscribers={n_subs} pending={self.pending_count()}>"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_bus: MessageBus | None = None
_bus_lock = threading.Lock()


def init_message_bus(max_queue_size: int = 1000) -> MessageBus:
    """Create (or return) the process-level :class:`MessageBus` singleton.

    Args:
        max_queue_size: Maximum number of messages in the async queue.

    Returns:
        The singleton :class:`MessageBus` instance.
    """
    global _bus
    with _bus_lock:
        if _bus is None:
            _bus = MessageBus(max_queue_size=max_queue_size, strict_topics=True)
        return _bus


def get_message_bus() -> MessageBus:
    """Return the process-level :class:`MessageBus` singleton.

    Returns:
        The singleton :class:`MessageBus` instance.

    Raises:
        RuntimeError: If :func:`init_message_bus` has not been called yet.
    """
    if _bus is None:
        raise RuntimeError("MessageBus not initialised. Call init_message_bus() first.")
    return _bus


def reset_message_bus() -> None:
    """Reset the singleton to ``None``.  Intended for tests only."""
    global _bus
    with _bus_lock:
        if _bus is not None:
            _bus.stop()
        _bus = None


def _validate_subscription_pattern(topic: str, *, strict: bool) -> None:
    if (
        not isinstance(topic, str)
        or not 0 <= len(topic) <= 256
        or topic.strip() != topic
        or any(not char.isprintable() or char in "\r\n\x00" for char in topic)
        or (strict and not topic)
    ):
        raise ValueError("Subscription topic must be bounded printable text.")
