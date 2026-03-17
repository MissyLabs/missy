"""Tests for missy.core.message_bus."""

from __future__ import annotations

import time
import uuid

import pytest

from missy.core.bus_topics import (
    AGENT_RUN_COMPLETE,
    AGENT_RUN_ERROR,
    AGENT_RUN_START,
    CHANNEL_INBOUND,
    CHANNEL_OUTBOUND,
    TOOL_REQUEST,
    TOOL_RESULT,
)
from missy.core.message_bus import (
    BusMessage,
    MessageBus,
    get_message_bus,
    init_message_bus,
    reset_message_bus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(
    topic: str = "test.topic",
    source: str = "test",
    priority: int = 0,
    correlation_id: str | None = None,
) -> BusMessage:
    return BusMessage(
        topic=topic,
        payload={"key": "value"},
        source=source,
        priority=priority,
        correlation_id=correlation_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPublishSubscribe:
    """Basic publish/subscribe functionality."""

    def test_publish_subscribe(self) -> None:
        """Subscriber receives a published message."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("test.topic", received.append)

        msg = _make_message()
        bus.publish(msg)

        assert len(received) == 1
        assert received[0] is msg
        assert received[0].topic == "test.topic"
        assert received[0].payload == {"key": "value"}

    def test_wildcard_topic(self) -> None:
        """Wildcard pattern 'channel.*' matches 'channel.inbound'."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("channel.*", received.append)

        bus.publish(_make_message(topic=CHANNEL_INBOUND))
        bus.publish(_make_message(topic=CHANNEL_OUTBOUND))
        bus.publish(_make_message(topic=AGENT_RUN_START))

        assert len(received) == 2
        assert received[0].topic == CHANNEL_INBOUND
        assert received[1].topic == CHANNEL_OUTBOUND

    def test_unsubscribe(self) -> None:
        """Handler is no longer called after unsubscribe."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("test.topic", received.append)

        bus.publish(_make_message())
        assert len(received) == 1

        bus.unsubscribe("test.topic", received.append)
        bus.publish(_make_message())
        assert len(received) == 1  # no new messages

    def test_unsubscribe_noop_for_unknown_handler(self) -> None:
        """Unsubscribing an unknown handler is a no-op."""
        bus = MessageBus()
        bus.unsubscribe("nonexistent", lambda m: None)  # should not raise

    def test_handler_exception_doesnt_propagate(self) -> None:
        """A handler that raises does not crash the bus or block others."""
        bus = MessageBus()
        received: list[BusMessage] = []

        def bad_handler(msg: BusMessage) -> None:
            raise RuntimeError("boom")

        bus.subscribe("test.topic", bad_handler)
        bus.subscribe("test.topic", received.append)

        bus.publish(_make_message())  # should not raise

        # The good handler still received the message.
        assert len(received) == 1

    def test_multiple_subscribers(self) -> None:
        """All handlers for a topic are called."""
        bus = MessageBus()
        a: list[BusMessage] = []
        b: list[BusMessage] = []
        c: list[BusMessage] = []

        bus.subscribe("test.topic", a.append)
        bus.subscribe("test.topic", b.append)
        bus.subscribe("test.*", c.append)

        bus.publish(_make_message())

        assert len(a) == 1
        assert len(b) == 1
        assert len(c) == 1


class TestAsyncDispatch:
    """Async queue and background worker."""

    def test_async_dispatch(self) -> None:
        """publish_async + drain delivers messages."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("test.topic", received.append)

        bus.publish_async(_make_message())
        assert bus.pending_count() == 1

        bus.drain()
        assert bus.pending_count() == 0
        assert len(received) == 1

    def test_priority_ordering(self) -> None:
        """Higher priority messages are dispatched first within async queue."""
        bus = MessageBus()
        order: list[int] = []

        def record_priority(msg: BusMessage) -> None:
            order.append(msg.priority)

        bus.subscribe("test.topic", record_priority)

        # Enqueue low, normal, then urgent.
        bus.publish_async(_make_message(priority=0))
        bus.publish_async(_make_message(priority=2))
        bus.publish_async(_make_message(priority=1))

        bus.drain()

        assert order == [2, 1, 0]

    def test_start_stop_worker(self) -> None:
        """Background worker processes messages."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("test.topic", received.append)

        bus.start()
        bus.publish_async(_make_message())

        # Give the worker a moment to dispatch.
        deadline = time.monotonic() + 2.0
        while len(received) == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        bus.stop()
        assert len(received) == 1

    def test_stop_drains(self) -> None:
        """stop() processes remaining messages in the queue."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("test.topic", received.append)

        # Don't start the worker — messages stay queued.
        bus.publish_async(_make_message())
        bus.publish_async(_make_message())
        assert bus.pending_count() == 2

        bus.stop()
        assert len(received) == 2
        assert bus.pending_count() == 0


class TestCorrelationId:
    """Request/response correlation."""

    def test_correlation_id(self) -> None:
        """Request and response messages are linked by correlation_id."""
        bus = MessageBus()
        requests: list[BusMessage] = []
        results: list[BusMessage] = []

        bus.subscribe(TOOL_REQUEST, requests.append)
        bus.subscribe(TOOL_RESULT, results.append)

        cid = str(uuid.uuid4())
        bus.publish(_make_message(topic=TOOL_REQUEST, correlation_id=cid))
        bus.publish(_make_message(topic=TOOL_RESULT, correlation_id=cid))

        assert len(requests) == 1
        assert len(results) == 1
        assert requests[0].correlation_id == cid
        assert results[0].correlation_id == cid
        assert requests[0].correlation_id == results[0].correlation_id


class TestSingleton:
    """Module-level singleton management."""

    def setup_method(self) -> None:
        reset_message_bus()

    def teardown_method(self) -> None:
        reset_message_bus()

    def test_singleton(self) -> None:
        """init_message_bus / get_message_bus return the same instance."""
        bus1 = init_message_bus()
        bus2 = get_message_bus()
        assert bus1 is bus2

    def test_get_before_init_raises(self) -> None:
        """get_message_bus raises before init_message_bus is called."""
        with pytest.raises(RuntimeError, match="not initialised"):
            get_message_bus()

    def test_init_idempotent(self) -> None:
        """Calling init_message_bus twice returns the same instance."""
        bus1 = init_message_bus()
        bus2 = init_message_bus()
        assert bus1 is bus2


class TestBusMessageDefaults:
    """BusMessage auto-generated fields."""

    def test_defaults(self) -> None:
        """message_id and timestamp are auto-generated."""
        msg = _make_message()
        assert msg.message_id  # non-empty UUID string
        uuid.UUID(msg.message_id)  # valid UUID
        assert msg.timestamp  # non-empty ISO string
        assert msg.target is None
        assert msg.correlation_id is None
        assert msg.priority == 0

    def test_topic_constants_are_strings(self) -> None:
        """Bus topic constants are plain strings."""
        for topic in (
            CHANNEL_INBOUND,
            CHANNEL_OUTBOUND,
            AGENT_RUN_START,
            AGENT_RUN_COMPLETE,
            AGENT_RUN_ERROR,
            TOOL_REQUEST,
            TOOL_RESULT,
        ):
            assert isinstance(topic, str)
            assert "." in topic
