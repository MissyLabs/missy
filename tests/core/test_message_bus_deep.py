"""Deep tests for missy.core.message_bus and missy.core.events.

Covers areas not exercised (or only lightly exercised) by the baseline
test_message_bus.py suite:

* Topic wildcard patterns for channel-namespace topics
* Priority ordering interacting with subscriber registration order
* Correlation ID propagation and cross-topic linkage
* AuditEvent category/result/session filtering via EventBus.get_events
* Thread safety under simultaneous publish + subscribe + unsubscribe
* Error isolation with multiple sequential failing subscribers
* Mid-flight unsubscribe (handler removed while bus is receiving messages)
* Active-topic listing via the internal _subscribers mapping
* High-volume (1 000 rapid publishes) correctness
* Synchronous-publish event ordering guarantee
* Module-level singleton lifecycle (init / get / reset)
* Cross-topic correlation on the EventBus (AuditEvent)
* EventBus.clear() removes both log and subscriptions
* Queue-full behaviour with a background worker draining mid-flight
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime

import pytest

from missy.core.bus_topics import (
    AGENT_RUN_COMPLETE,
    AGENT_RUN_ERROR,
    AGENT_RUN_START,
    CHANNEL_INBOUND,
    CHANNEL_OUTBOUND,
    SECURITY_APPROVAL_NEEDED,
    SECURITY_APPROVAL_RESPONSE,
    SECURITY_VIOLATION,
    SYSTEM_SHUTDOWN,
    SYSTEM_STARTUP,
    TOOL_REQUEST,
    TOOL_RESULT,
)
from missy.core.events import AuditEvent, EventBus
from missy.core.message_bus import (
    BusMessage,
    MessageBus,
    get_message_bus,
    init_message_bus,
    reset_message_bus,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _msg(
    topic: str = "test.topic",
    source: str = "test",
    priority: int = 0,
    correlation_id: str | None = None,
    payload: dict | None = None,
) -> BusMessage:
    return BusMessage(
        topic=topic,
        payload=payload or {},
        source=source,
        priority=priority,
        correlation_id=correlation_id,
    )


def _audit(
    event_type: str = "test.event",
    category: str = "network",
    result: str = "allow",
    session_id: str = "sess-1",
    task_id: str = "task-1",
) -> AuditEvent:
    return AuditEvent.now(
        session_id=session_id,
        task_id=task_id,
        event_type=event_type,
        category=category,
        result=result,
    )


# ---------------------------------------------------------------------------
# 1. Topic wildcard patterns — channel namespace
# ---------------------------------------------------------------------------


class TestChannelWildcardPatterns:
    """channel.* resolves correctly for CLI and Discord sub-topics."""

    def test_channel_star_matches_channel_cli(self) -> None:
        """'channel.*' matches the non-standard topic 'channel.cli'."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("channel.*", received.append)
        bus.publish(_msg(topic="channel.cli"))
        assert len(received) == 1
        assert received[0].topic == "channel.cli"

    def test_channel_star_matches_channel_discord(self) -> None:
        """'channel.*' matches 'channel.discord'."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("channel.*", received.append)
        bus.publish(_msg(topic="channel.discord"))
        assert len(received) == 1

    def test_channel_star_matches_both_standard_topics(self) -> None:
        """'channel.*' matches both CHANNEL_INBOUND and CHANNEL_OUTBOUND."""
        bus = MessageBus()
        topics_seen: list[str] = []
        bus.subscribe("channel.*", lambda m: topics_seen.append(m.topic))
        bus.publish(_msg(topic=CHANNEL_INBOUND))
        bus.publish(_msg(topic=CHANNEL_OUTBOUND))
        assert CHANNEL_INBOUND in topics_seen
        assert CHANNEL_OUTBOUND in topics_seen

    def test_channel_star_does_not_match_agent_topics(self) -> None:
        """'channel.*' rejects topics from a different namespace."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("channel.*", received.append)
        bus.publish(_msg(topic=AGENT_RUN_START))
        bus.publish(_msg(topic=TOOL_REQUEST))
        assert received == []

    def test_channel_inbound_matches_exact_and_wildcard(self) -> None:
        """An exact subscription and a wildcard subscription both fire."""
        bus = MessageBus()
        exact: list[BusMessage] = []
        wild: list[BusMessage] = []
        bus.subscribe(CHANNEL_INBOUND, exact.append)
        bus.subscribe("channel.*", wild.append)
        bus.publish(_msg(topic=CHANNEL_INBOUND))
        assert len(exact) == 1
        assert len(wild) == 1

    def test_security_star_matches_all_security_subtopics(self) -> None:
        """'security.*' matches all three security topic constants."""
        bus = MessageBus()
        received: list[str] = []
        bus.subscribe("security.*", lambda m: received.append(m.topic))
        for topic in (SECURITY_VIOLATION, SECURITY_APPROVAL_NEEDED, SECURITY_APPROVAL_RESPONSE):
            bus.publish(_msg(topic=topic))
        assert sorted(received) == sorted(
            [SECURITY_VIOLATION, SECURITY_APPROVAL_NEEDED, SECURITY_APPROVAL_RESPONSE]
        )

    def test_system_star_matches_startup_and_shutdown(self) -> None:
        """'system.*' matches SYSTEM_STARTUP and SYSTEM_SHUTDOWN."""
        bus = MessageBus()
        received: list[str] = []
        bus.subscribe("system.*", lambda m: received.append(m.topic))
        bus.publish(_msg(topic=SYSTEM_STARTUP))
        bus.publish(_msg(topic=SYSTEM_SHUTDOWN))
        assert set(received) == {SYSTEM_STARTUP, SYSTEM_SHUTDOWN}

    def test_tool_star_matches_request_and_result(self) -> None:
        """'tool.*' matches TOOL_REQUEST and TOOL_RESULT."""
        bus = MessageBus()
        received: list[str] = []
        bus.subscribe("tool.*", lambda m: received.append(m.topic))
        bus.publish(_msg(topic=TOOL_REQUEST))
        bus.publish(_msg(topic=TOOL_RESULT))
        assert set(received) == {TOOL_REQUEST, TOOL_RESULT}


# ---------------------------------------------------------------------------
# 2. Priority ordering — interaction with registration order
# ---------------------------------------------------------------------------


class TestPriorityWithRegistrationOrder:
    """Priority ordering and how it interacts with multiple subscribers."""

    def test_same_priority_preserves_registration_order_synchronous(self) -> None:
        """Synchronous publish calls handlers in subscription order, not by priority."""
        bus = MessageBus()
        call_order: list[str] = []
        bus.subscribe("test.topic", lambda m: call_order.append("first"))
        bus.subscribe("test.topic", lambda m: call_order.append("second"))
        bus.subscribe("test.topic", lambda m: call_order.append("third"))
        bus.publish(_msg())
        assert call_order == ["first", "second", "third"]

    def test_async_priority_all_three_tiers_batch(self) -> None:
        """Batching 9 messages across priority tiers — drained order is 2,2,2,1,1,1,0,0,0."""
        bus = MessageBus()
        priorities_seen: list[int] = []
        bus.subscribe("test.topic", lambda m: priorities_seen.append(m.priority))
        for _ in range(3):
            bus.publish_async(_msg(priority=0))
        for _ in range(3):
            bus.publish_async(_msg(priority=1))
        for _ in range(3):
            bus.publish_async(_msg(priority=2))
        bus.drain()
        # All priority-2 messages must precede priority-1, which must precede priority-0.
        assert priorities_seen[:3] == [2, 2, 2]
        assert priorities_seen[3:6] == [1, 1, 1]
        assert priorities_seen[6:] == [0, 0, 0]

    def test_mixed_priority_interleaved_enqueue(self) -> None:
        """Interleaved enqueue of priorities — final drain respects priority heap."""
        bus = MessageBus()
        priorities_seen: list[int] = []
        bus.subscribe("test.topic", lambda m: priorities_seen.append(m.priority))

        # Interleave: 1, 0, 2, 1, 0, 2
        for p in (1, 0, 2, 1, 0, 2):
            bus.publish_async(_msg(priority=p))

        bus.drain()
        # Highest priorities first; within same priority, FIFO.
        assert priorities_seen == [2, 2, 1, 1, 0, 0]

    def test_urgent_message_inserted_after_normal_still_leads(self) -> None:
        """An urgent message enqueued after normals drains before them."""
        bus = MessageBus()
        order: list[int] = []
        bus.subscribe("test.topic", lambda m: order.append(m.priority))
        # Enqueue several normals first, then one urgent.
        for _ in range(5):
            bus.publish_async(_msg(priority=0))
        bus.publish_async(_msg(priority=2))
        bus.drain()
        assert order[0] == 2, "Urgent message must lead the drain despite late enqueue"

    def test_synchronous_publish_priority_field_not_used_for_dispatch_order(self) -> None:
        """Synchronous publish() ignores priority — handler call order = subscription order."""
        bus = MessageBus()
        topics_received: list[str] = []
        bus.subscribe("test.topic", lambda m: topics_received.append(f"A:{m.priority}"))
        bus.subscribe("test.topic", lambda m: topics_received.append(f"B:{m.priority}"))
        bus.publish(_msg(priority=2))
        # Both subscribers called, A before B, despite high priority.
        assert topics_received == ["A:2", "B:2"]


# ---------------------------------------------------------------------------
# 3. Correlation ID propagation
# ---------------------------------------------------------------------------


class TestCorrelationIdPropagation:
    """Correlation IDs survive publish/drain and link request/response pairs."""

    def test_correlation_id_preserved_through_async_drain(self) -> None:
        """Async-enqueued message retains its correlation_id after drain."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("test.topic", received.append)
        cid = str(uuid.uuid4())
        bus.publish_async(_msg(correlation_id=cid))
        bus.drain()
        assert received[0].correlation_id == cid

    def test_messages_without_correlation_id_have_none(self) -> None:
        """Messages that omit correlation_id carry None, not a fabricated value."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("test.topic", received.append)
        bus.publish(_msg())
        assert received[0].correlation_id is None

    def test_different_messages_can_share_correlation_id(self) -> None:
        """Multiple messages may carry the same correlation_id (fan-out scenario)."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("test.*", received.append)
        cid = str(uuid.uuid4())
        bus.publish(_msg(topic="test.a", correlation_id=cid))
        bus.publish(_msg(topic="test.b", correlation_id=cid))
        bus.publish(_msg(topic="test.c", correlation_id=cid))
        assert all(m.correlation_id == cid for m in received)
        assert len(received) == 3

    def test_tool_request_response_correlation(self) -> None:
        """Tool request and result share correlation_id (the standard workflow)."""
        bus = MessageBus()
        requests: list[BusMessage] = []
        results: list[BusMessage] = []
        bus.subscribe(TOOL_REQUEST, requests.append)
        bus.subscribe(TOOL_RESULT, results.append)

        cid = str(uuid.uuid4())
        bus.publish(_msg(topic=TOOL_REQUEST, correlation_id=cid))
        bus.publish(_msg(topic=TOOL_RESULT, correlation_id=cid))

        assert requests[0].correlation_id == results[0].correlation_id == cid

    def test_agent_run_lifecycle_correlation(self) -> None:
        """Start → complete lifecycle messages share correlation_id."""
        bus = MessageBus()
        lifecycle: list[BusMessage] = []
        bus.subscribe("agent.*", lifecycle.append)

        cid = str(uuid.uuid4())
        bus.publish(_msg(topic=AGENT_RUN_START, correlation_id=cid))
        bus.publish(_msg(topic=AGENT_RUN_COMPLETE, correlation_id=cid))

        assert len(lifecycle) == 2
        assert lifecycle[0].correlation_id == lifecycle[1].correlation_id == cid

    def test_each_message_has_unique_message_id_regardless_of_correlation(self) -> None:
        """Even when correlation_id is shared, each message_id is distinct."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("test.*", received.append)

        cid = str(uuid.uuid4())
        for topic in ("test.a", "test.b", "test.c"):
            bus.publish(_msg(topic=topic, correlation_id=cid))

        ids = [m.message_id for m in received]
        assert len(set(ids)) == 3, "Each message must have a distinct message_id"

    def test_subscriber_can_group_messages_by_correlation_id(self) -> None:
        """A subscriber that accumulates messages can index them by correlation_id."""
        bus = MessageBus()
        by_cid: dict[str, list[BusMessage]] = defaultdict(list)

        def collect(msg: BusMessage) -> None:
            if msg.correlation_id:
                by_cid[msg.correlation_id].append(msg)

        bus.subscribe("work.*", collect)

        cid_a = str(uuid.uuid4())
        cid_b = str(uuid.uuid4())
        bus.publish(_msg(topic="work.start", correlation_id=cid_a))
        bus.publish(_msg(topic="work.start", correlation_id=cid_b))
        bus.publish(_msg(topic="work.end", correlation_id=cid_a))

        assert len(by_cid[cid_a]) == 2
        assert len(by_cid[cid_b]) == 1


# ---------------------------------------------------------------------------
# 4. Event filtering by category (EventBus.get_events)
# ---------------------------------------------------------------------------


class TestEventBusFiltering:
    """EventBus.get_events filters by every supported dimension."""

    def test_filter_by_category_network(self) -> None:
        """get_events(category='network') returns only network events."""
        bus = EventBus()
        bus.publish(_audit(event_type="net.req", category="network"))
        bus.publish(_audit(event_type="fs.read", category="filesystem"))
        bus.publish(_audit(event_type="net.resp", category="network"))

        result = bus.get_events(category="network")
        assert len(result) == 2
        assert all(e.category == "network" for e in result)

    def test_filter_by_category_filesystem(self) -> None:
        bus = EventBus()
        bus.publish(_audit(category="filesystem"))
        bus.publish(_audit(category="shell"))
        result = bus.get_events(category="filesystem")
        assert len(result) == 1
        assert result[0].category == "filesystem"

    def test_filter_by_result_allow(self) -> None:
        bus = EventBus()
        bus.publish(_audit(result="allow"))
        bus.publish(_audit(result="deny"))
        bus.publish(_audit(result="allow"))
        result = bus.get_events(result="allow")
        assert len(result) == 2
        assert all(e.result == "allow" for e in result)

    def test_filter_by_result_deny(self) -> None:
        bus = EventBus()
        bus.publish(_audit(result="deny"))
        bus.publish(_audit(result="allow"))
        result = bus.get_events(result="deny")
        assert len(result) == 1
        assert result[0].result == "deny"

    def test_filter_by_result_error(self) -> None:
        bus = EventBus()
        bus.publish(_audit(result="error"))
        bus.publish(_audit(result="allow"))
        result = bus.get_events(result="error")
        assert len(result) == 1
        assert result[0].result == "error"

    def test_filter_by_session_id(self) -> None:
        bus = EventBus()
        bus.publish(_audit(session_id="sess-A"))
        bus.publish(_audit(session_id="sess-B"))
        bus.publish(_audit(session_id="sess-A"))
        result = bus.get_events(session_id="sess-A")
        assert len(result) == 2
        assert all(e.session_id == "sess-A" for e in result)

    def test_filter_by_task_id(self) -> None:
        bus = EventBus()
        bus.publish(_audit(task_id="task-X"))
        bus.publish(_audit(task_id="task-Y"))
        result = bus.get_events(task_id="task-X")
        assert len(result) == 1
        assert result[0].task_id == "task-X"

    def test_filter_by_event_type(self) -> None:
        bus = EventBus()
        bus.publish(_audit(event_type="network.request"))
        bus.publish(_audit(event_type="network.response"))
        bus.publish(_audit(event_type="network.request"))
        result = bus.get_events(event_type="network.request")
        assert len(result) == 2

    def test_combined_filter_category_and_result(self) -> None:
        """Filters ANDed: only network deny events returned."""
        bus = EventBus()
        bus.publish(_audit(category="network", result="deny"))
        bus.publish(_audit(category="network", result="allow"))
        bus.publish(_audit(category="filesystem", result="deny"))
        result = bus.get_events(category="network", result="deny")
        assert len(result) == 1
        assert result[0].category == "network"
        assert result[0].result == "deny"

    def test_combined_filter_session_category(self) -> None:
        bus = EventBus()
        bus.publish(_audit(session_id="s1", category="shell"))
        bus.publish(_audit(session_id="s1", category="network"))
        bus.publish(_audit(session_id="s2", category="shell"))
        result = bus.get_events(session_id="s1", category="shell")
        assert len(result) == 1

    def test_no_filter_returns_all_events(self) -> None:
        bus = EventBus()
        for _ in range(5):
            bus.publish(_audit())
        result = bus.get_events()
        assert len(result) == 5

    def test_filter_returns_empty_list_when_nothing_matches(self) -> None:
        bus = EventBus()
        bus.publish(_audit(category="network"))
        result = bus.get_events(category="shell")
        assert result == []

    def test_get_events_returns_snapshot_not_live_reference(self) -> None:
        """Mutating the returned list does not affect internal log."""
        bus = EventBus()
        bus.publish(_audit())
        snapshot = bus.get_events()
        snapshot.clear()
        # Bus still has the event.
        assert len(bus.get_events()) == 1

    def test_events_returned_in_chronological_order(self) -> None:
        """Events come back in insertion order (oldest first)."""
        bus = EventBus()
        types = ["first.event", "second.event", "third.event"]
        for et in types:
            bus.publish(_audit(event_type=et))
        result = bus.get_events()
        assert [e.event_type for e in result] == types


# ---------------------------------------------------------------------------
# 5. Thread safety — concurrent publish + subscribe + unsubscribe
# ---------------------------------------------------------------------------


class TestThreadSafetyDeep:
    """Races between publishers and subscriber management must not corrupt state."""

    def test_concurrent_publish_and_subscribe(self) -> None:
        """New subscriptions registered while messages are being published."""
        bus = MessageBus()
        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def publisher() -> None:
            barrier.wait()
            for _ in range(200):
                try:
                    bus.publish(_msg(topic="live.topic"))
                except Exception as exc:
                    errors.append(exc)

        def subscriber_registrar() -> None:
            barrier.wait()
            for _ in range(200):
                try:
                    handler = lambda m: None  # noqa: E731
                    bus.subscribe("live.topic", handler)
                except Exception as exc:
                    errors.append(exc)

        t1 = threading.Thread(target=publisher)
        t2 = threading.Thread(target=subscriber_registrar)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert errors == [], f"Unexpected exceptions: {errors}"

    def test_concurrent_publish_and_unsubscribe(self) -> None:
        """Handlers removed while messages are being published."""
        bus = MessageBus()
        errors: list[Exception] = []

        handlers = [lambda m: None for _ in range(50)]  # noqa: E731
        for h in handlers:
            bus.subscribe("churn.topic", h)

        barrier = threading.Barrier(2)

        def publisher() -> None:
            barrier.wait()
            for _ in range(300):
                try:
                    bus.publish(_msg(topic="churn.topic"))
                except Exception as exc:
                    errors.append(exc)

        def remover() -> None:
            barrier.wait()
            for h in handlers:
                try:
                    bus.unsubscribe("churn.topic", h)
                except Exception as exc:
                    errors.append(exc)

        t1 = threading.Thread(target=publisher)
        t2 = threading.Thread(target=remover)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert errors == [], f"Unexpected exceptions: {errors}"

    def test_concurrent_async_publish_and_drain(self) -> None:
        """Multiple threads enqueue; one thread drains — no lost messages."""
        bus = MessageBus(max_queue_size=5000)
        received: list[BusMessage] = []
        lock = threading.Lock()

        def safe_collect(m: BusMessage) -> None:
            with lock:
                received.append(m)

        bus.subscribe("flood.*", safe_collect)

        n_threads = 20
        n_per_thread = 50
        barrier = threading.Barrier(n_threads)

        def enqueue() -> None:
            barrier.wait()
            for i in range(n_per_thread):
                bus.publish_async(_msg(topic=f"flood.{i}"))

        threads = [threading.Thread(target=enqueue) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        bus.drain()
        assert len(received) == n_threads * n_per_thread

    def test_eventbus_concurrent_publish_no_corruption(self) -> None:
        """EventBus handles 20 threads each publishing 50 events — all stored."""
        bus = EventBus()
        n_threads = 20
        n_events = 50
        barrier = threading.Barrier(n_threads)

        def publisher() -> None:
            barrier.wait()
            for _ in range(n_events):
                bus.publish(_audit())

        threads = [threading.Thread(target=publisher) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        all_events = bus.get_events()
        assert len(all_events) == n_threads * n_events


# ---------------------------------------------------------------------------
# 6. Error isolation — multiple failing subscribers in a chain
# ---------------------------------------------------------------------------


class TestErrorIsolationDeep:
    """A chain of failing handlers must not block downstream subscribers."""

    def test_multiple_failing_handlers_do_not_block_final_good_handler(self) -> None:
        """All failing handlers are skipped; the last good handler still runs."""
        bus = MessageBus()
        good_received: list[BusMessage] = []

        def make_bad(label: str):
            def bad(msg: BusMessage) -> None:
                raise RuntimeError(f"bad handler {label}")
            return bad

        bus.subscribe("test.topic", make_bad("A"))
        bus.subscribe("test.topic", make_bad("B"))
        bus.subscribe("test.topic", make_bad("C"))
        bus.subscribe("test.topic", good_received.append)

        bus.publish(_msg())
        assert len(good_received) == 1

    def test_interleaved_good_bad_handlers(self) -> None:
        """Good handlers sandwiching bad ones all receive the message."""
        bus = MessageBus()
        results: list[str] = []

        bus.subscribe("test.topic", lambda m: results.append("good-1"))
        bus.subscribe("test.topic", lambda m: (_ for _ in ()).throw(ValueError("bad-2")))
        bus.subscribe("test.topic", lambda m: results.append("good-3"))
        bus.subscribe("test.topic", lambda m: (_ for _ in ()).throw(ValueError("bad-4")))
        bus.subscribe("test.topic", lambda m: results.append("good-5"))

        bus.publish(_msg())
        assert results == ["good-1", "good-3", "good-5"]

    def test_handler_exception_in_async_path_does_not_stop_queue(self) -> None:
        """A bad handler during drain does not halt subsequent message delivery."""
        bus = MessageBus()
        good_received: list[BusMessage] = []

        def bad(msg: BusMessage) -> None:
            raise RuntimeError("async path failure")

        bus.subscribe("test.topic", bad)
        bus.subscribe("test.topic", good_received.append)

        for _ in range(5):
            bus.publish_async(_msg())
        bus.drain()

        assert len(good_received) == 5

    def test_eventbus_error_isolation(self) -> None:
        """EventBus: a failing callback does not prevent good callbacks from running."""
        bus = EventBus()
        good_received: list[AuditEvent] = []

        bus.subscribe("test.event", lambda e: (_ for _ in ()).throw(RuntimeError("bad")))
        bus.subscribe("test.event", good_received.append)

        bus.publish(_audit())
        assert len(good_received) == 1


# ---------------------------------------------------------------------------
# 7. Unsubscribe — edge cases and mid-flight removal
# ---------------------------------------------------------------------------


class TestUnsubscribeDeep:
    """Unsubscribing in various scenarios."""

    def test_unsubscribe_one_of_many(self) -> None:
        """Removing one of two handlers leaves the other intact and still receiving."""
        bus = MessageBus()
        received_a: list[BusMessage] = []
        received_b: list[BusMessage] = []

        # Keep explicit references so unsubscribe uses the exact same object.
        handler_a = received_a.append
        handler_b = received_b.append
        bus.subscribe("test.topic", handler_a)
        bus.subscribe("test.topic", handler_b)

        # Remove only handler_a.
        bus.unsubscribe("test.topic", handler_a)
        bus.publish(_msg())

        # handler_a removed — receives nothing; handler_b untouched — receives one.
        assert len(received_a) == 0
        assert len(received_b) == 1

    def test_unsubscribe_correct_reference_removes_handler(self) -> None:
        """Using the exact same handler object removes it."""
        bus = MessageBus()
        received: list[BusMessage] = []

        def handler(msg: BusMessage) -> None:
            received.append(msg)

        bus.subscribe("test.topic", handler)
        bus.publish(_msg())
        assert len(received) == 1

        bus.unsubscribe("test.topic", handler)
        bus.publish(_msg())
        assert len(received) == 1  # second publish not delivered

    def test_unsubscribed_handler_not_called_for_wildcard_match(self) -> None:
        """A handler removed from a wildcard pattern is not called."""
        bus = MessageBus()
        received: list[BusMessage] = []

        def handler(msg: BusMessage) -> None:
            received.append(msg)

        bus.subscribe("agent.*", handler)
        bus.publish(_msg(topic=AGENT_RUN_START))
        assert len(received) == 1

        bus.unsubscribe("agent.*", handler)
        bus.publish(_msg(topic=AGENT_RUN_COMPLETE))
        assert len(received) == 1  # not called again

    def test_unsubscribe_clears_pattern_from_subscribers_dict(self) -> None:
        """After the last handler on a pattern is removed the key disappears."""
        bus = MessageBus()

        def h(msg: BusMessage) -> None:
            pass

        bus.subscribe("temp.topic", h)
        assert "temp.topic" in bus._subscribers
        bus.unsubscribe("temp.topic", h)
        assert "temp.topic" not in bus._subscribers

    def test_eventbus_unsubscribe_stops_future_delivery(self) -> None:
        bus = EventBus()
        received: list[AuditEvent] = []

        def cb(e: AuditEvent) -> None:
            received.append(e)

        bus.subscribe("test.event", cb)
        bus.publish(_audit())
        assert len(received) == 1

        bus.unsubscribe("test.event", cb)
        bus.publish(_audit())
        assert len(received) == 1

    def test_eventbus_unsubscribe_wrong_type_is_noop(self) -> None:
        """Removing a callback under the wrong event_type leaves original intact."""
        bus = EventBus()
        received: list[AuditEvent] = []

        def cb(e: AuditEvent) -> None:
            received.append(e)

        bus.subscribe("right.event", cb)
        bus.unsubscribe("wrong.event", cb)  # no-op
        bus.publish(_audit(event_type="right.event"))
        assert len(received) == 1


# ---------------------------------------------------------------------------
# 8. Topic listing — active patterns visible through _subscribers
# ---------------------------------------------------------------------------


class TestTopicListing:
    """Inspect active subscription patterns via _subscribers."""

    def test_subscribe_adds_pattern_to_subscribers_dict(self) -> None:
        bus = MessageBus()
        bus.subscribe("alpha.topic", lambda m: None)
        assert "alpha.topic" in bus._subscribers

    def test_multiple_patterns_all_appear(self) -> None:
        bus = MessageBus()
        patterns = ["a.b", "c.*", "*.d", "e.f.g"]
        for p in patterns:
            bus.subscribe(p, lambda m: None)
        for p in patterns:
            assert p in bus._subscribers

    def test_listing_patterns_via_keys(self) -> None:
        """Caller can inspect active topic patterns via bus._subscribers.keys()."""
        bus = MessageBus()
        bus.subscribe("channel.*", lambda m: None)
        bus.subscribe("agent.*", lambda m: None)
        active = set(bus._subscribers.keys())
        assert {"channel.*", "agent.*"}.issubset(active)

    def test_unsubscribed_pattern_absent_from_listing(self) -> None:
        bus = MessageBus()

        def h(m: BusMessage) -> None:
            pass

        bus.subscribe("transient.*", h)
        assert "transient.*" in bus._subscribers
        bus.unsubscribe("transient.*", h)
        assert "transient.*" not in bus._subscribers

    def test_subscriber_count_correct_after_add_and_remove(self) -> None:
        """Handler count per pattern stays accurate across subscribe/unsubscribe."""
        bus = MessageBus()

        handlers = [lambda m: None for _ in range(4)]  # noqa: E731
        for h in handlers:
            bus.subscribe("count.test", h)
        assert len(bus._subscribers["count.test"]) == 4

        bus.unsubscribe("count.test", handlers[0])
        assert len(bus._subscribers["count.test"]) == 3

    def test_eventbus_listing_via_subscribers(self) -> None:
        """EventBus also exposes its subscription map."""
        bus = EventBus()
        bus.subscribe("net.request", lambda e: None)
        bus.subscribe("fs.read", lambda e: None)
        assert "net.request" in bus._subscribers
        assert "fs.read" in bus._subscribers


# ---------------------------------------------------------------------------
# 9. High-volume events — 1000 rapid publishes
# ---------------------------------------------------------------------------


class TestHighVolumeEvents:
    """1 000 messages processed without loss or exception."""

    def test_synchronous_1000_messages_all_delivered(self) -> None:
        """1 000 synchronous publishes all reach the subscriber."""
        bus = MessageBus()
        counter = {"n": 0}

        def increment(m: BusMessage) -> None:
            counter["n"] += 1

        bus.subscribe("load.*", increment)

        for i in range(1000):
            bus.publish(_msg(topic=f"load.{i % 10}"))

        assert counter["n"] == 1000

    def test_async_1000_messages_all_delivered_after_drain(self) -> None:
        """1 000 async publishes enqueued then drained — all delivered."""
        bus = MessageBus(max_queue_size=1500)
        counter = {"n": 0}

        def increment(m: BusMessage) -> None:
            counter["n"] += 1

        bus.subscribe("load.*", increment)

        for i in range(1000):
            bus.publish_async(_msg(topic=f"load.{i % 10}"))

        assert bus.pending_count() == 1000
        bus.drain()
        assert counter["n"] == 1000
        assert bus.pending_count() == 0

    def test_worker_delivers_1000_messages(self) -> None:
        """Background worker delivers 1 000 messages via start()/stop()."""
        bus = MessageBus(max_queue_size=1500)
        lock = threading.Lock()
        counter = {"n": 0}

        def increment(m: BusMessage) -> None:
            with lock:
                counter["n"] += 1

        bus.subscribe("load.test", increment)
        bus.start()

        for _ in range(1000):
            bus.publish_async(_msg(topic="load.test"))

        deadline = time.monotonic() + 10.0
        while counter["n"] < 1000 and time.monotonic() < deadline:
            time.sleep(0.01)

        bus.stop()
        assert counter["n"] == 1000

    def test_high_volume_unique_message_ids(self) -> None:
        """All 1 000 messages carry distinct message_id values."""
        bus = MessageBus(max_queue_size=1500)
        ids: list[str] = []
        bus.subscribe("uid.*", lambda m: ids.append(m.message_id))

        for _ in range(1000):
            bus.publish(_msg(topic="uid.test"))

        assert len(set(ids)) == 1000

    def test_eventbus_1000_events_stored_and_filterable(self) -> None:
        """EventBus stores 1 000 events and can filter them correctly."""
        bus = EventBus()
        for i in range(500):
            bus.publish(_audit(category="network", result="allow"))
        for i in range(500):
            bus.publish(_audit(category="filesystem", result="deny"))

        all_events = bus.get_events()
        assert len(all_events) == 1000

        network = bus.get_events(category="network")
        assert len(network) == 500

        denied = bus.get_events(result="deny")
        assert len(denied) == 500


# ---------------------------------------------------------------------------
# 10. Event ordering guarantee — synchronous publish order
# ---------------------------------------------------------------------------


class TestEventOrderingGuarantee:
    """Synchronous publish delivers messages in strict publication order."""

    def test_synchronous_order_matches_publish_order(self) -> None:
        """Ten messages published in order 0..9 arrive in the same order."""
        bus = MessageBus()
        order: list[int] = []
        bus.subscribe("ordered.*", lambda m: order.append(m.payload["seq"]))

        for i in range(10):
            bus.publish(BusMessage(topic=f"ordered.{i}", payload={"seq": i}, source="test"))

        assert order == list(range(10))

    def test_synchronous_multiple_topics_order_preserved(self) -> None:
        """Mixed topics published in a fixed order arrive in that same order."""
        bus = MessageBus()
        seen: list[str] = []
        bus.subscribe("topic.*", lambda m: seen.append(m.topic))

        sequence = ["topic.a", "topic.b", "topic.a", "topic.c", "topic.b"]
        for t in sequence:
            bus.publish(_msg(topic=t))

        assert seen == sequence

    def test_async_fifo_within_same_priority_strict(self) -> None:
        """Strict FIFO within priority=0: 100 messages drained in enqueue order."""
        bus = MessageBus(max_queue_size=200)
        received_ids: list[str] = []
        bus.subscribe("fifo.test", lambda m: received_ids.append(m.message_id))

        published_ids: list[str] = []
        for _ in range(100):
            msg = BusMessage(topic="fifo.test", payload={}, source="test", priority=0)
            published_ids.append(msg.message_id)
            bus.publish_async(msg)

        bus.drain()
        assert received_ids == published_ids

    def test_per_subscriber_invocation_order_is_stable(self) -> None:
        """When multiple subscribers match, each sees messages in publish order."""
        bus = MessageBus()
        sub_a: list[int] = []
        sub_b: list[int] = []

        bus.subscribe("order.*", lambda m: sub_a.append(m.payload["n"]))
        bus.subscribe("order.*", lambda m: sub_b.append(m.payload["n"]))

        for n in range(20):
            bus.publish(BusMessage(topic="order.test", payload={"n": n}, source="test"))

        assert sub_a == list(range(20))
        assert sub_b == list(range(20))


# ---------------------------------------------------------------------------
# 11. Module-level singleton lifecycle
# ---------------------------------------------------------------------------


class TestSingletonLifecycle:
    """init_message_bus / get_message_bus / reset_message_bus semantics."""

    def setup_method(self) -> None:
        reset_message_bus()

    def teardown_method(self) -> None:
        reset_message_bus()

    def test_get_before_init_raises_runtime_error(self) -> None:
        with pytest.raises(RuntimeError, match="not initialised"):
            get_message_bus()

    def test_init_returns_message_bus_instance(self) -> None:
        bus = init_message_bus()
        assert isinstance(bus, MessageBus)

    def test_init_and_get_return_same_object(self) -> None:
        bus1 = init_message_bus()
        bus2 = get_message_bus()
        assert bus1 is bus2

    def test_init_idempotent_same_singleton(self) -> None:
        bus1 = init_message_bus()
        bus2 = init_message_bus()
        assert bus1 is bus2

    def test_reset_clears_singleton(self) -> None:
        init_message_bus()
        reset_message_bus()
        with pytest.raises(RuntimeError):
            get_message_bus()

    def test_reinit_after_reset_creates_fresh_bus(self) -> None:
        bus1 = init_message_bus()
        reset_message_bus()
        bus2 = init_message_bus()
        assert bus1 is not bus2

    def test_singleton_messages_persist_between_get_calls(self) -> None:
        """Messages published on the singleton are visible to any get_message_bus() caller."""
        init_message_bus()
        received: list[BusMessage] = []
        get_message_bus().subscribe("singleton.test", received.append)
        get_message_bus().publish(_msg(topic="singleton.test"))
        assert len(received) == 1

    def test_concurrent_init_returns_same_singleton(self) -> None:
        """Concurrent calls to init_message_bus() always return the same instance."""
        instances: list[MessageBus] = []
        lock = threading.Lock()

        def init_and_record() -> None:
            bus = init_message_bus()
            with lock:
                instances.append(bus)

        threads = [threading.Thread(target=init_and_record) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # Every thread must have received the same singleton object.
        first = instances[0]
        assert all(b is first for b in instances), "Concurrent init produced different instances"

    def test_max_queue_size_respected_by_singleton(self) -> None:
        """init_message_bus(max_queue_size=...) configures the singleton correctly."""
        bus = init_message_bus(max_queue_size=42)
        assert bus._queue.maxsize == 42


# ---------------------------------------------------------------------------
# 12. Cross-topic correlation on the EventBus (AuditEvent)
# ---------------------------------------------------------------------------


class TestCrossTopicCorrelationEventBus:
    """AuditEvent does not have correlation_id; simulate cross-event linkage via session_id."""

    def test_same_session_groups_multiple_event_types(self) -> None:
        """Events from the same session_id are retrievable as a coherent group."""
        bus = EventBus()
        bus.publish(_audit(event_type="network.request", session_id="sess-XYZ"))
        bus.publish(_audit(event_type="network.response", session_id="sess-XYZ"))
        bus.publish(_audit(event_type="network.request", session_id="sess-OTHER"))

        result = bus.get_events(session_id="sess-XYZ")
        assert len(result) == 2
        assert all(e.session_id == "sess-XYZ" for e in result)

    def test_task_id_links_events_within_session(self) -> None:
        """task_id further narrows retrieval to a specific task inside a session."""
        bus = EventBus()
        bus.publish(_audit(session_id="sess-1", task_id="task-A", event_type="tool.call"))
        bus.publish(_audit(session_id="sess-1", task_id="task-B", event_type="tool.call"))
        bus.publish(_audit(session_id="sess-1", task_id="task-A", event_type="tool.result"))

        task_a_events = bus.get_events(session_id="sess-1", task_id="task-A")
        assert len(task_a_events) == 2
        assert all(e.task_id == "task-A" for e in task_a_events)

    def test_subscriber_receives_events_from_multiple_tasks(self) -> None:
        """A single subscriber on an event_type receives events from all tasks."""
        bus = EventBus()
        received: list[AuditEvent] = []
        bus.subscribe("tool.call", received.append)

        for task in ("task-1", "task-2", "task-3"):
            bus.publish(_audit(event_type="tool.call", task_id=task))

        assert len(received) == 3
        task_ids_seen = {e.task_id for e in received}
        assert task_ids_seen == {"task-1", "task-2", "task-3"}

    def test_policy_rule_field_preserved(self) -> None:
        """AuditEvent.policy_rule is stored and retrievable."""
        bus = EventBus()
        event = AuditEvent.now(
            session_id="s",
            task_id="t",
            event_type="network.deny",
            category="network",
            result="deny",
            policy_rule="no-external-hosts",
        )
        bus.publish(event)
        stored = bus.get_events(event_type="network.deny")
        assert len(stored) == 1
        assert stored[0].policy_rule == "no-external-hosts"

    def test_detail_dict_preserved_in_stored_event(self) -> None:
        """AuditEvent.detail is stored verbatim."""
        bus = EventBus()
        detail = {"host": "example.com", "port": 443, "method": "GET"}
        event = AuditEvent.now(
            session_id="s",
            task_id="t",
            event_type="network.request",
            category="network",
            result="allow",
            detail=detail,
        )
        bus.publish(event)
        stored = bus.get_events(event_type="network.request")
        assert stored[0].detail == detail


# ---------------------------------------------------------------------------
# 13. EventBus.clear() — removes log and subscriptions
# ---------------------------------------------------------------------------


class TestEventBusClear:
    """EventBus.clear() resets both the event log and all subscribers."""

    def test_clear_empties_event_log(self) -> None:
        bus = EventBus()
        for _ in range(10):
            bus.publish(_audit())
        assert len(bus.get_events()) == 10

        bus.clear()
        assert len(bus.get_events()) == 0

    def test_clear_removes_all_subscribers(self) -> None:
        bus = EventBus()
        received: list[AuditEvent] = []
        bus.subscribe("test.event", received.append)

        bus.clear()
        bus.publish(_audit())

        # clear() removed the subscriber, so nothing received.
        assert received == []

    def test_clear_then_resubscribe_works(self) -> None:
        """After clear(), new subscriptions are accepted and work normally."""
        bus = EventBus()
        bus.subscribe("old.event", lambda e: None)
        bus.clear()

        received: list[AuditEvent] = []
        bus.subscribe("new.event", received.append)
        bus.publish(_audit(event_type="new.event"))

        assert len(received) == 1

    def test_clear_is_idempotent(self) -> None:
        """Calling clear() on an already-empty bus does not raise."""
        bus = EventBus()
        bus.clear()
        bus.clear()
        assert bus.get_events() == []

    def test_events_published_before_clear_not_visible_after(self) -> None:
        bus = EventBus()
        bus.publish(_audit(event_type="before.clear"))
        bus.clear()
        assert bus.get_events(event_type="before.clear") == []

    def test_events_published_after_clear_are_visible(self) -> None:
        bus = EventBus()
        bus.publish(_audit(event_type="before.clear"))
        bus.clear()
        bus.publish(_audit(event_type="after.clear"))

        result = bus.get_events(event_type="after.clear")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 14. Queue full behaviour — background worker interaction
# ---------------------------------------------------------------------------


class TestQueueFullBehavior:
    """MessageBus behaviour at and beyond max_queue_size."""

    def test_queue_fills_to_capacity(self) -> None:
        """Enqueuing exactly max_queue_size messages fills the queue."""
        bus = MessageBus(max_queue_size=10)
        for _ in range(10):
            bus.publish_async(_msg())
        assert bus.pending_count() == 10

    def test_drain_empties_full_queue(self) -> None:
        """Draining a full queue brings pending_count to zero."""
        bus = MessageBus(max_queue_size=10)
        for _ in range(10):
            bus.publish_async(_msg())
        bus.drain()
        assert bus.pending_count() == 0

    def test_put_nowait_raises_full_when_at_capacity(self) -> None:
        """The underlying PriorityQueue raises queue.Full on put_nowait when full."""
        bus = MessageBus(max_queue_size=5)
        for _ in range(5):
            bus.publish_async(_msg())
        with pytest.raises(queue.Full):
            bus._queue.put_nowait((-0, 9999, _msg()))

    def test_partial_drain_frees_space_for_new_messages(self) -> None:
        """After a partial drain, new messages can be enqueued."""
        bus = MessageBus(max_queue_size=4)
        received: list[BusMessage] = []
        bus.subscribe("test.topic", received.append)

        bus.publish_async(_msg())
        bus.publish_async(_msg())
        bus.drain()
        assert bus.pending_count() == 0

        bus.publish_async(_msg())
        bus.publish_async(_msg())
        bus.drain()
        assert len(received) == 4

    def test_worker_drains_full_queue_without_message_loss(self) -> None:
        """Background worker drains a pre-filled queue; all messages delivered."""
        capacity = 50
        bus = MessageBus(max_queue_size=capacity)
        lock = threading.Lock()
        received: list[BusMessage] = []

        def collect(m: BusMessage) -> None:
            with lock:
                received.append(m)

        bus.subscribe("bulk.topic", collect)

        for _ in range(capacity):
            bus.publish_async(_msg(topic="bulk.topic"))

        assert bus.pending_count() == capacity

        bus.start()
        deadline = time.monotonic() + 10.0
        while len(received) < capacity and time.monotonic() < deadline:
            time.sleep(0.01)
        bus.stop()

        assert len(received) == capacity

    def test_stop_drains_remaining_messages_in_full_queue(self) -> None:
        """stop() processes all messages even when worker was never started."""
        bus = MessageBus(max_queue_size=20)
        received: list[BusMessage] = []
        bus.subscribe("stop.test", received.append)

        for _ in range(20):
            bus.publish_async(_msg(topic="stop.test"))

        assert bus.pending_count() == 20
        bus.stop()
        assert bus.pending_count() == 0
        assert len(received) == 20
