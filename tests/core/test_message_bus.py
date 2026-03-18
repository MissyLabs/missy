"""Tests for missy.core.message_bus."""

from __future__ import annotations

import queue
import threading
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


# ---------------------------------------------------------------------------
# New test classes — edge cases and additional coverage
# ---------------------------------------------------------------------------


class TestWildcardPatternEdgeCases:
    """fnmatch wildcard pattern matching edge cases.

    The bus delegates to Python's :func:`fnmatch.fnmatch` which treats ``*``
    as matching any sequence of characters *including* dots.  This means a
    single ``*`` matches multi-level dotted topics.  The tests below document
    the actual matching semantics of the implementation.
    """

    def test_bare_star_matches_single_level(self) -> None:
        """'*' matches a flat (no-dot) topic."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("*", received.append)
        bus.publish(_make_message(topic="anything"))
        assert len(received) == 1

    def test_bare_star_matches_multi_level(self) -> None:
        """'*' also matches a dotted topic because fnmatch '*' crosses dots."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("*", received.append)
        bus.publish(_make_message(topic="agent.run.start"))
        assert len(received) == 1

    def test_star_dot_star_matches_two_level_topic(self) -> None:
        """'*.*' matches a two-level dotted topic."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("*.*", received.append)
        bus.publish(_make_message(topic="agent.run"))
        assert len(received) == 1

    def test_star_dot_star_matches_three_level_topic(self) -> None:
        """'*.*' also matches a three-level topic; fnmatch '*' is greedy across dots."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("*.*", received.append)
        bus.publish(_make_message(topic="agent.run.start"))
        assert len(received) == 1

    def test_prefix_star_matches_deeper_levels(self) -> None:
        """'agent.*' matches 'agent.run.start' because fnmatch '*' crosses dots."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("agent.*", received.append)
        bus.publish(_make_message(topic="agent.run.start"))
        assert len(received) == 1

    def test_double_star_segment_pattern_matches_three_level(self) -> None:
        """'agent.*.*' matches 'agent.run.start'."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("agent.*.*", received.append)
        bus.publish(_make_message(topic="agent.run.start"))
        assert len(received) == 1

    def test_prefix_star_does_not_match_different_prefix(self) -> None:
        """'agent.*' does not match a topic with a different prefix."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("agent.*", received.append)
        bus.publish(_make_message(topic="channel.inbound"))
        assert len(received) == 0

    def test_empty_topic_matches_empty_pattern(self) -> None:
        """An empty topic string matches an empty pattern exactly."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("", received.append)
        bus.publish(_make_message(topic=""))
        assert len(received) == 1

    def test_empty_topic_does_not_match_nonempty_pattern(self) -> None:
        """An empty topic does not match a non-empty literal pattern."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("something", received.append)
        bus.publish(_make_message(topic=""))
        assert len(received) == 0

    def test_topic_with_dashes_and_underscores(self) -> None:
        """Topics containing hyphens and underscores are matched literally."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("my-service.some_event.*", received.append)
        bus.publish(_make_message(topic="my-service.some_event.fired"))
        assert len(received) == 1

    def test_topic_with_special_chars_no_spurious_match(self) -> None:
        """A topic with special characters only matches patterns that describe it."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("other.*", received.append)
        bus.publish(_make_message(topic="my-service.some_event.fired"))
        assert len(received) == 0

    def test_pattern_is_case_sensitive(self) -> None:
        """fnmatch pattern matching on Linux is case-sensitive."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("UPPER.*", received.append)
        bus.publish(_make_message(topic="upper.run"))
        assert len(received) == 0

    def test_matching_pattern_correct_case(self) -> None:
        """Same pattern and topic with matching case does match."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("UPPER.*", received.append)
        bus.publish(_make_message(topic="UPPER.run"))
        assert len(received) == 1


class TestQueueFullBehavior:
    """Behavior when the async queue is at capacity."""

    def test_queue_reaches_max_size(self) -> None:
        """Messages fill the queue up to max_queue_size."""
        bus = MessageBus(max_queue_size=3)
        for _ in range(3):
            bus.publish_async(_make_message())
        assert bus.pending_count() == 3

    def test_queue_full_blocks_until_space_available(self) -> None:
        """publish_async on a full queue blocks; draining frees space for new messages."""
        bus = MessageBus(max_queue_size=2)
        bus.publish_async(_make_message())
        bus.publish_async(_make_message())
        assert bus.pending_count() == 2

        # Drain creates space; a subsequent publish_async must succeed.
        bus.drain()
        assert bus.pending_count() == 0
        bus.publish_async(_make_message())  # must not raise or block
        assert bus.pending_count() == 1

    def test_put_nowait_raises_full_when_queue_at_capacity(self) -> None:
        """The underlying PriorityQueue raises queue.Full on put_nowait when full.

        This exercises the data structure used by the bus directly, verifying
        that the queue is bounded.  The public publish_async API uses the
        blocking put() and will block rather than raise when full; the docstring
        description of the exception reflects the queue's inherent contract.
        """
        bus = MessageBus(max_queue_size=1)
        bus.publish_async(_make_message())
        assert bus.pending_count() == 1

        with pytest.raises(queue.Full):
            bus._queue.put_nowait((-0, 999, _make_message()))


class TestStartIdempotent:
    """start() must not create duplicate worker threads."""

    def test_start_twice_creates_one_worker(self) -> None:
        """Calling start() a second time when the worker is alive is a no-op."""
        bus = MessageBus()
        bus.start()
        worker_after_first = bus._worker
        bus.start()
        worker_after_second = bus._worker
        bus.stop()

        assert worker_after_first is worker_after_second

    def test_worker_thread_name(self) -> None:
        """The background worker is named 'missy-message-bus' for diagnostics."""
        bus = MessageBus()
        bus.start()
        name = bus._worker.name if bus._worker else None
        bus.stop()
        assert name == "missy-message-bus"


class TestDrainEmptyQueue:
    """drain() on an already-empty queue must be a no-op."""

    def test_drain_empty_does_not_raise(self) -> None:
        """drain() on an empty queue completes without error."""
        bus = MessageBus()
        bus.drain()  # must not raise
        assert bus.pending_count() == 0

    def test_drain_empty_calls_no_handlers(self) -> None:
        """drain() on an empty queue invokes no subscriber handlers."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("test.*", received.append)
        bus.drain()
        assert received == []

    def test_drain_idempotent(self) -> None:
        """Calling drain() twice on an empty queue is safe."""
        bus = MessageBus()
        bus.drain()
        bus.drain()
        assert bus.pending_count() == 0


class TestUnsubscribeCleansUpEmptyPatterns:
    """After the last handler is removed, the pattern key is deleted."""

    def test_pattern_key_deleted_when_last_handler_removed(self) -> None:
        """The pattern entry is absent from _subscribers after last handler removed."""
        bus = MessageBus()
        handler = lambda msg: None  # noqa: E731
        bus.subscribe("test.topic", handler)
        assert "test.topic" in bus._subscribers

        bus.unsubscribe("test.topic", handler)
        assert "test.topic" not in bus._subscribers

    def test_pattern_key_kept_when_other_handlers_remain(self) -> None:
        """The pattern entry is retained when at least one handler is still registered."""
        bus = MessageBus()
        handler_a = lambda msg: None  # noqa: E731
        handler_b = lambda msg: None  # noqa: E731
        bus.subscribe("test.topic", handler_a)
        bus.subscribe("test.topic", handler_b)

        bus.unsubscribe("test.topic", handler_a)
        assert "test.topic" in bus._subscribers
        assert handler_b in bus._subscribers["test.topic"]

    def test_unsubscribe_unknown_pattern_noop(self) -> None:
        """Unsubscribing a pattern that was never registered does not raise."""
        bus = MessageBus()
        bus.unsubscribe("never.registered", lambda msg: None)
        assert "never.registered" not in bus._subscribers

    def test_unsubscribe_same_handler_twice_noop(self) -> None:
        """Removing the same handler a second time is a no-op."""
        bus = MessageBus()
        handler = lambda msg: None  # noqa: E731
        bus.subscribe("test.topic", handler)
        bus.unsubscribe("test.topic", handler)
        bus.unsubscribe("test.topic", handler)  # must not raise


class TestThreadSafety:
    """Multiple threads publishing concurrently must not corrupt state."""

    def test_concurrent_publish_does_not_crash(self) -> None:
        """50 threads each publishing 20 messages — no exceptions, all delivered."""
        bus = MessageBus(max_queue_size=2000)
        received: list[BusMessage] = []
        lock = threading.Lock()

        def safe_append(msg: BusMessage) -> None:
            with lock:
                received.append(msg)

        bus.subscribe("stress.*", safe_append)

        n_threads = 50
        n_messages_per_thread = 20
        barriers: list[threading.Barrier] = [threading.Barrier(n_threads)]

        def publisher() -> None:
            barriers[0].wait()  # all threads start at the same time
            for i in range(n_messages_per_thread):
                bus.publish(_make_message(topic=f"stress.{i}"))

        threads = [threading.Thread(target=publisher) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        expected = n_threads * n_messages_per_thread
        assert len(received) == expected

    def test_concurrent_subscribe_unsubscribe_does_not_crash(self) -> None:
        """Simultaneous subscribe and unsubscribe operations are safe."""
        bus = MessageBus()
        errors: list[Exception] = []

        def subscriber_worker() -> None:
            try:
                handler = lambda msg: None  # noqa: E731
                for _ in range(100):
                    bus.subscribe("concurrent.topic", handler)
                    bus.unsubscribe("concurrent.topic", handler)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=subscriber_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []


class TestMessageOrdering:
    """FIFO ordering within the same priority level."""

    def test_fifo_within_same_priority(self) -> None:
        """Messages at equal priority are dispatched in publication order."""
        bus = MessageBus()
        order: list[str] = []

        def record_id(msg: BusMessage) -> None:
            order.append(msg.message_id)

        bus.subscribe("test.topic", record_id)

        ids = [str(uuid.uuid4()) for _ in range(5)]
        for mid in ids:
            msg = BusMessage(topic="test.topic", payload={}, source="test")
            msg.message_id = mid
            bus.publish_async(msg)

        bus.drain()

        assert order == ids

    def test_fifo_preserved_across_drain_calls(self) -> None:
        """Sequential drain calls do not reorder within same priority."""
        bus = MessageBus()
        order: list[str] = []

        def record_id(msg: BusMessage) -> None:
            order.append(msg.message_id)

        bus.subscribe("test.topic", record_id)

        msg_a = BusMessage(topic="test.topic", payload={}, source="test")
        msg_b = BusMessage(topic="test.topic", payload={}, source="test")
        bus.publish_async(msg_a)
        bus.publish_async(msg_b)

        bus.drain()

        assert order == [msg_a.message_id, msg_b.message_id]


class TestRepr:
    """__repr__ returns a meaningful human-readable string."""

    def test_repr_format(self) -> None:
        """repr includes subscriber count and pending message count."""
        bus = MessageBus()
        r = repr(bus)
        assert "MessageBus" in r
        assert "subscribers=" in r
        assert "pending=" in r

    def test_repr_reflects_subscriber_count(self) -> None:
        """repr subscriber count increases as handlers are added."""
        bus = MessageBus()
        assert "subscribers=0" in repr(bus)

        bus.subscribe("a.topic", lambda msg: None)
        assert "subscribers=1" in repr(bus)

        bus.subscribe("a.topic", lambda msg: None)
        bus.subscribe("b.topic", lambda msg: None)
        assert "subscribers=3" in repr(bus)

    def test_repr_reflects_pending_count(self) -> None:
        """repr pending count matches the number of queued async messages."""
        bus = MessageBus()
        assert "pending=0" in repr(bus)

        bus.publish_async(_make_message())
        bus.publish_async(_make_message())
        assert "pending=2" in repr(bus)

        bus.drain()
        assert "pending=0" in repr(bus)


class TestTargetField:
    """Messages with a target field are dispatched normally."""

    def test_targeted_message_dispatched_to_pattern_subscribers(self) -> None:
        """Setting target does not suppress delivery to pattern-matched subscribers."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("agent.*", received.append)

        msg = BusMessage(
            topic="agent.run",
            payload={"x": 1},
            source="orchestrator",
            target="worker-1",
        )
        bus.publish(msg)

        assert len(received) == 1
        assert received[0].target == "worker-1"

    def test_targeted_message_dispatched_to_exact_topic_subscriber(self) -> None:
        """A subscriber on the exact topic receives the targeted message."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("agent.run", received.append)

        msg = BusMessage(
            topic="agent.run",
            payload={},
            source="test",
            target="specific-handler",
        )
        bus.publish(msg)

        assert len(received) == 1

    def test_target_is_informational_only(self) -> None:
        """target does not filter delivery; all matching subscribers receive the message."""
        bus = MessageBus()
        received_a: list[BusMessage] = []
        received_b: list[BusMessage] = []

        bus.subscribe("agent.run", received_a.append)
        bus.subscribe("agent.*", received_b.append)

        msg = BusMessage(
            topic="agent.run",
            payload={},
            source="test",
            target="worker-a",  # intended for worker-a, but both subscribers get it
        )
        bus.publish(msg)

        assert len(received_a) == 1
        assert len(received_b) == 1


class TestHighPriorityPreemption:
    """Urgent messages published after normal ones are dispatched first."""

    def test_urgent_dispatched_before_normal(self) -> None:
        """priority=2 message enqueued after priority=0 is drained first."""
        bus = MessageBus()
        order: list[int] = []

        def record_priority(msg: BusMessage) -> None:
            order.append(msg.priority)

        bus.subscribe("test.topic", record_priority)

        bus.publish_async(_make_message(priority=0))  # normal — enqueued first
        bus.publish_async(_make_message(priority=2))  # urgent — enqueued second

        bus.drain()

        assert order[0] == 2, "urgent message must be dispatched before normal"
        assert order[1] == 0

    def test_priority_2_before_1_before_0(self) -> None:
        """Three-tier ordering: urgent (2) > high (1) > normal (0)."""
        bus = MessageBus()
        order: list[int] = []

        def record_priority(msg: BusMessage) -> None:
            order.append(msg.priority)

        bus.subscribe("test.topic", record_priority)

        bus.publish_async(_make_message(priority=0))
        bus.publish_async(_make_message(priority=1))
        bus.publish_async(_make_message(priority=2))

        bus.drain()

        assert order == [2, 1, 0]

    def test_multiple_urgent_messages_preserved_fifo(self) -> None:
        """Multiple urgent messages at the same priority are drained in FIFO order."""
        bus = MessageBus()
        ids: list[str] = []

        def record_id(msg: BusMessage) -> None:
            ids.append(msg.message_id)

        bus.subscribe("test.topic", record_id)

        msg_x = BusMessage(topic="test.topic", payload={}, source="test", priority=2)
        msg_y = BusMessage(topic="test.topic", payload={}, source="test", priority=2)
        bus.publish_async(msg_x)
        bus.publish_async(msg_y)

        bus.drain()

        assert ids == [msg_x.message_id, msg_y.message_id]
