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
import logging
import queue
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatch

logger = logging.getLogger(__name__)

# Callback type for subscribers.
BusHandler = Callable[["BusMessage"], None]


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

    def __init__(self, max_queue_size: int = 1000) -> None:
        self._lock = threading.Lock()
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
        handlers = self._resolve_handlers(message.topic)
        for handler in handlers:
            try:
                handler(message)
            except Exception:
                logger.exception(
                    "Unhandled exception in MessageBus handler %r for topic %r",
                    handler,
                    message.topic,
                )

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
        with self._lock:
            seq = self._seq
            self._seq += 1
        # Negate priority so higher values sort first in the min-heap.
        self._queue.put((-message.priority, seq, message))

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
        # Drain remaining messages synchronously.
        self.drain()
        if self._worker is not None:
            self._worker.join(timeout=timeout)
            self._worker = None

    def drain(self) -> None:
        """Process all currently queued messages synchronously.

        Useful for testing and for ensuring delivery during shutdown.
        """
        while True:
            try:
                _neg_pri, _seq, message = self._queue.get_nowait()
            except queue.Empty:
                break
            self.publish(message)

    def pending_count(self) -> int:
        """Return the number of messages waiting in the async queue."""
        return self._queue.qsize()

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

    def _dispatch_loop(self) -> None:
        """Background worker loop: pull from queue and dispatch."""
        while not self._stop_event.is_set():
            try:
                _neg_pri, _seq, message = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self.publish(message)

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
            _bus = MessageBus(max_queue_size=max_queue_size)
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
