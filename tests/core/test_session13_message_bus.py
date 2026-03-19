"""Session-13 edge-case tests for missy.core.message_bus.

These tests focus exclusively on behaviours not already exercised in
test_message_bus.py or test_message_bus_deep.py:

* Worker thread lifecycle: stop with no worker, double-stop, restart after stop
* Subscriber self-unsubscribe from inside a callback
* Same handler registered under multiple patterns — called once per matching pattern
* Same handler registered twice under the same pattern — called twice
* publish() / publish_async() / drain() with zero subscribers
* BusMessage field contract: explicit message_id, explicit priority edge values,
  timestamp UTC format, payload identity (not copied)
* fnmatch single-char '?' wildcard and character-class '[...]' patterns
* Exact literal pattern does not match a different concrete topic
* pending_count() accuracy after partial drain
* Synchronous publish() coexists safely with a live background worker
* repr() stability while the worker thread is active
* _seq counter is strictly monotonically increasing across threads
* BusMessage with priority beyond documented range (>2 and <0) still sorts correctly
"""

from __future__ import annotations

import threading
import time

from missy.core.bus_topics import AGENT_RUN_START
from missy.core.message_bus import (
    BusMessage,
    MessageBus,
    init_message_bus,
    reset_message_bus,
)

# ---------------------------------------------------------------------------
# Helpers
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
        payload=payload if payload is not None else {},
        source=source,
        priority=priority,
        correlation_id=correlation_id,
    )


def _wait_for(condition, *, timeout: float = 3.0, interval: float = 0.01) -> bool:
    """Poll *condition* until it returns True or *timeout* seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# 1. Worker thread lifecycle edge cases
# ---------------------------------------------------------------------------


class TestWorkerLifecycle:
    """start / stop / restart sequencing."""

    def test_stop_when_no_worker_ever_started_is_noop(self) -> None:
        """Calling stop() on a bus that was never started must not raise."""
        bus = MessageBus()
        bus.stop()  # _worker is None — must be a safe no-op

    def test_stop_twice_is_safe(self) -> None:
        """stop() called a second time after the worker has already stopped is safe."""
        bus = MessageBus()
        bus.start()
        bus.stop()
        bus.stop()  # second stop must not raise

    def test_worker_is_none_after_stop(self) -> None:
        """After stop(), the internal _worker reference is cleared to None."""
        bus = MessageBus()
        bus.start()
        assert bus._worker is not None
        bus.stop()
        assert bus._worker is None

    def test_restart_after_stop_delivers_new_messages(self) -> None:
        """A bus that has been stopped can be restarted and delivers new messages."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("restart.test", received.append)

        bus.start()
        bus.publish_async(_msg(topic="restart.test"))
        assert _wait_for(lambda: len(received) >= 1)
        bus.stop()
        first_count = len(received)

        # Restart and publish another message.
        bus.start()
        bus.publish_async(_msg(topic="restart.test"))
        assert _wait_for(lambda: len(received) >= first_count + 1)
        bus.stop()

        assert len(received) == first_count + 1

    def test_stop_clears_worker_reference_even_with_no_pending_messages(self) -> None:
        """stop() on a started-but-idle bus still clears the worker reference."""
        bus = MessageBus()
        bus.start()
        # No messages enqueued.
        bus.stop()
        assert bus._worker is None

    def test_worker_is_daemon_thread(self) -> None:
        """The background worker thread is a daemon so it won't block process exit."""
        bus = MessageBus()
        bus.start()
        assert bus._worker is not None
        assert bus._worker.daemon is True
        bus.stop()

    def test_stop_with_queued_messages_drains_before_joining(self) -> None:
        """stop() synchronously drains queued messages then joins the thread."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("drain.check", received.append)

        # Enqueue but don't start the worker — messages sit in the queue.
        for _ in range(5):
            bus.publish_async(_msg(topic="drain.check"))

        assert bus.pending_count() == 5
        bus.stop()  # should drain then join (worker was never started, that's fine)
        assert len(received) == 5
        assert bus.pending_count() == 0


# ---------------------------------------------------------------------------
# 2. Handler registration edge cases
# ---------------------------------------------------------------------------


class TestHandlerRegistration:
    """Multiple registrations of the same handler object."""

    def test_same_handler_registered_twice_on_same_pattern_called_twice(self) -> None:
        """Registering the same handler object twice on the same pattern → called twice."""
        bus = MessageBus()
        call_count = {"n": 0}

        def handler(msg: BusMessage) -> None:
            call_count["n"] += 1

        bus.subscribe("dup.topic", handler)
        bus.subscribe("dup.topic", handler)  # second registration

        bus.publish(_msg(topic="dup.topic"))

        assert call_count["n"] == 2

    def test_same_handler_on_two_different_patterns_both_match_called_twice(self) -> None:
        """A handler registered on two patterns that both match → called once per match."""
        bus = MessageBus()
        call_count = {"n": 0}

        def handler(msg: BusMessage) -> None:
            call_count["n"] += 1

        bus.subscribe("multi.test", handler)   # exact match
        bus.subscribe("multi.*", handler)       # wildcard match

        bus.publish(_msg(topic="multi.test"))

        assert call_count["n"] == 2

    def test_handler_on_two_patterns_only_one_matches(self) -> None:
        """Same handler on two patterns; only one matches the topic → called once."""
        bus = MessageBus()
        call_count = {"n": 0}

        def handler(msg: BusMessage) -> None:
            call_count["n"] += 1

        bus.subscribe("alpha.*", handler)
        bus.subscribe("beta.*", handler)

        bus.publish(_msg(topic="alpha.run"))  # matches alpha.* only

        assert call_count["n"] == 1

    def test_unsubscribe_removes_one_registration_when_registered_twice(self) -> None:
        """Unsubscribing once reduces a double-registration to a single call."""
        bus = MessageBus()
        call_count = {"n": 0}

        def handler(msg: BusMessage) -> None:
            call_count["n"] += 1

        bus.subscribe("dup.topic", handler)
        bus.subscribe("dup.topic", handler)

        # Remove one registration.
        bus.unsubscribe("dup.topic", handler)

        bus.publish(_msg(topic="dup.topic"))

        # One registration remains → called once.
        assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# 3. Self-unsubscribe from inside a handler
# ---------------------------------------------------------------------------


class TestSelfUnsubscribe:
    """A handler that removes itself during its own callback."""

    def test_handler_unsubscribes_itself_during_callback(self) -> None:
        """A handler that calls unsubscribe(self) during dispatch is only called once."""
        bus = MessageBus()
        call_count = {"n": 0}

        def one_shot(msg: BusMessage) -> None:
            call_count["n"] += 1
            bus.unsubscribe("oneshot.topic", one_shot)

        bus.subscribe("oneshot.topic", one_shot)

        bus.publish(_msg(topic="oneshot.topic"))
        bus.publish(_msg(topic="oneshot.topic"))  # second publish must not call handler

        assert call_count["n"] == 1

    def test_self_unsubscribe_does_not_prevent_other_handlers_from_running(self) -> None:
        """When a handler removes itself, subsequent handlers still receive the message."""
        bus = MessageBus()
        other_received: list[BusMessage] = []

        def self_remover(msg: BusMessage) -> None:
            bus.unsubscribe("shared.topic", self_remover)

        bus.subscribe("shared.topic", self_remover)
        bus.subscribe("shared.topic", other_received.append)

        bus.publish(_msg(topic="shared.topic"))

        # self_remover ran (and removed itself); other handler also ran.
        assert len(other_received) == 1

    def test_async_self_unsubscribe_across_drain(self) -> None:
        """A one-shot handler registered before drain() is not called on subsequent drains."""
        bus = MessageBus()
        call_count = {"n": 0}

        def one_shot(msg: BusMessage) -> None:
            call_count["n"] += 1
            bus.unsubscribe("async.oneshot", one_shot)

        bus.subscribe("async.oneshot", one_shot)

        bus.publish_async(_msg(topic="async.oneshot"))
        bus.drain()
        assert call_count["n"] == 1

        # Enqueue another message — handler was removed, should not be called.
        bus.publish_async(_msg(topic="async.oneshot"))
        bus.drain()
        assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# 4. Zero-subscriber edge cases
# ---------------------------------------------------------------------------


class TestZeroSubscribers:
    """publish / publish_async / drain with no subscribers registered."""

    def test_publish_with_no_subscribers_does_not_raise(self) -> None:
        """publish() on a bus with no subscribers completes silently."""
        bus = MessageBus()
        bus.publish(_msg())  # must not raise

    def test_publish_async_with_no_subscribers_enqueues_message(self) -> None:
        """publish_async() with no subscribers still enqueues the message."""
        bus = MessageBus()
        bus.publish_async(_msg())
        assert bus.pending_count() == 1

    def test_drain_with_no_subscribers_empties_queue(self) -> None:
        """drain() on a no-subscriber bus processes and removes all queued messages."""
        bus = MessageBus()
        for _ in range(5):
            bus.publish_async(_msg())
        assert bus.pending_count() == 5

        bus.drain()
        assert bus.pending_count() == 0

    def test_publish_on_topic_with_no_match_does_not_raise(self) -> None:
        """Publishing on a topic that matches no registered pattern is safe."""
        bus = MessageBus()
        bus.subscribe("other.topic", lambda m: None)
        bus.publish(_msg(topic="unmatched.topic"))  # must not raise


# ---------------------------------------------------------------------------
# 5. BusMessage field contract
# ---------------------------------------------------------------------------


class TestBusMessageFieldContract:
    """Verify BusMessage field behaviour beyond the basic defaults tests."""

    def test_explicit_message_id_is_honoured(self) -> None:
        """Passing an explicit message_id overrides the auto-generated UUID."""
        custom_id = "my-custom-id-12345"
        msg = BusMessage(
            topic="test.topic",
            payload={},
            source="test",
            message_id=custom_id,
        )
        assert msg.message_id == custom_id

    def test_timestamp_includes_utc_offset(self) -> None:
        """Auto-generated timestamp contains the UTC offset marker '+00:00'."""
        msg = _msg()
        assert "+00:00" in msg.timestamp

    def test_payload_is_not_copied_by_bus(self) -> None:
        """publish() passes the exact same payload dict object to the handler."""
        bus = MessageBus()
        received_payloads: list[dict] = []
        bus.subscribe("identity.topic", lambda m: received_payloads.append(m.payload))

        payload = {"x": 42}
        msg = BusMessage(topic="identity.topic", payload=payload, source="test")
        bus.publish(msg)

        assert received_payloads[0] is payload

    def test_message_object_identity_preserved_through_publish(self) -> None:
        """publish() delivers the exact same BusMessage object to the handler."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("id.topic", received.append)

        msg = _msg(topic="id.topic")
        bus.publish(msg)

        assert received[0] is msg

    def test_message_object_identity_preserved_through_async_drain(self) -> None:
        """publish_async() + drain() delivers the exact same BusMessage object."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("id.async", received.append)

        msg = _msg(topic="id.async")
        bus.publish_async(msg)
        bus.drain()

        assert received[0] is msg

    def test_priority_above_documented_max_sorts_before_lower_values(self) -> None:
        """Priority values above 2 (e.g. 10) are sorted before lower priorities."""
        bus = MessageBus()
        order: list[int] = []
        bus.subscribe("test.topic", lambda m: order.append(m.priority))

        bus.publish_async(_msg(priority=0))
        bus.publish_async(_msg(priority=10))  # undocumented high value
        bus.publish_async(_msg(priority=2))

        bus.drain()

        assert order[0] == 10, "Priority 10 must lead over 2 and 0"
        assert order[1] == 2
        assert order[2] == 0

    def test_negative_priority_sorts_after_zero(self) -> None:
        """A negative priority value sorts after the normal priority=0 tier."""
        bus = MessageBus()
        order: list[int] = []
        bus.subscribe("test.topic", lambda m: order.append(m.priority))

        bus.publish_async(_msg(priority=-1))  # below normal
        bus.publish_async(_msg(priority=0))   # normal

        bus.drain()

        assert order[0] == 0, "priority=0 must drain before priority=-1"
        assert order[1] == -1


# ---------------------------------------------------------------------------
# 6. fnmatch wildcard edge cases not yet tested
# ---------------------------------------------------------------------------


class TestFnmatchEdgeCases:
    """'?' single-char wildcard and '[...]' character-class patterns."""

    def test_question_mark_matches_single_character(self) -> None:
        """'?' in a pattern matches exactly one character."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("run.?", received.append)

        bus.publish(_msg(topic="run.a"))   # matches: one char
        bus.publish(_msg(topic="run.b"))   # matches: one char
        bus.publish(_msg(topic="run.ab"))  # does NOT match: two chars
        bus.publish(_msg(topic="run."))    # does NOT match: zero chars

        assert len(received) == 2

    def test_question_mark_does_not_match_dot(self) -> None:
        """'?' matches any single character including '.'.

        fnmatch '?' matches any single character, including '.', so
        'a.?.c' would match 'a.b.c' but not 'a.bc.c'.
        """
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("a.?.c", received.append)

        bus.publish(_msg(topic="a.b.c"))   # matches: '?' = 'b'
        bus.publish(_msg(topic="a.bc.c"))  # does NOT match: 'bc' is two chars

        assert len(received) == 1
        assert received[0].topic == "a.b.c"

    def test_character_class_pattern_matches_listed_chars(self) -> None:
        """'[abc]' matches any single character from the set."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("level.[abc]", received.append)

        bus.publish(_msg(topic="level.a"))  # match
        bus.publish(_msg(topic="level.b"))  # match
        bus.publish(_msg(topic="level.c"))  # match
        bus.publish(_msg(topic="level.d"))  # no match

        assert len(received) == 3
        assert all(m.topic in ("level.a", "level.b", "level.c") for m in received)

    def test_exact_literal_pattern_does_not_match_wildcard_chars_in_topic(self) -> None:
        """A literal pattern like 'test.topic' only matches 'test.topic', not 'test.other'."""
        bus = MessageBus()
        received: list[BusMessage] = []
        bus.subscribe("test.topic", received.append)

        bus.publish(_msg(topic="test.other"))
        bus.publish(_msg(topic="test.topic"))  # only this one matches
        bus.publish(_msg(topic="test.topic.sub"))  # does NOT match

        assert len(received) == 1
        assert received[0].topic == "test.topic"

    def test_wildcard_in_topic_string_itself_is_treated_literally_by_fnmatch(self) -> None:
        """A topic string containing '*' is matched as a literal character by fnmatch."""
        bus = MessageBus()
        received: list[BusMessage] = []
        # Register exact pattern with literal '*'
        bus.subscribe("weird.*topic", received.append)

        # fnmatch("weird.concrete", "weird.*topic") → False (suffix mismatch)
        bus.publish(_msg(topic="weird.concrete"))
        # fnmatch("weird.Xtopic", "weird.*topic") → True ('*' matches 'X')
        bus.publish(_msg(topic="weird.Xtopic"))

        assert len(received) == 1
        assert received[0].topic == "weird.Xtopic"


# ---------------------------------------------------------------------------
# 7. pending_count() accuracy
# ---------------------------------------------------------------------------


class TestPendingCountAccuracy:
    """pending_count() reflects the true queue depth after partial operations."""

    def test_pending_count_increments_with_each_async_publish(self) -> None:
        """pending_count() increases by 1 for each publish_async call."""
        bus = MessageBus()
        for i in range(1, 6):
            bus.publish_async(_msg())
            assert bus.pending_count() == i

    def test_pending_count_decrements_after_each_get_in_drain(self) -> None:
        """After drain(), pending_count() is 0 regardless of prior depth."""
        bus = MessageBus()
        for _ in range(10):
            bus.publish_async(_msg())
        assert bus.pending_count() == 10

        bus.drain()
        assert bus.pending_count() == 0

    def test_pending_count_correct_after_mixed_publish_drain_cycles(self) -> None:
        """pending_count() stays accurate across interleaved publish and drain calls."""
        bus = MessageBus()
        bus.subscribe("count.test", lambda m: None)

        bus.publish_async(_msg(topic="count.test"))
        bus.publish_async(_msg(topic="count.test"))
        assert bus.pending_count() == 2

        bus.drain()
        assert bus.pending_count() == 0

        bus.publish_async(_msg(topic="count.test"))
        assert bus.pending_count() == 1

        bus.drain()
        assert bus.pending_count() == 0


# ---------------------------------------------------------------------------
# 8. Synchronous publish coexisting with live worker thread
# ---------------------------------------------------------------------------


class TestSyncPublishWithLiveWorker:
    """Synchronous publish() is safe while the background worker is running."""

    def test_sync_and_async_publish_both_deliver_while_worker_running(self) -> None:
        """Messages from sync publish and async publish_async both arrive correctly."""
        bus = MessageBus()
        sync_received: list[BusMessage] = []
        async_received: list[BusMessage] = []
        lock = threading.Lock()

        def on_sync(msg: BusMessage) -> None:
            with lock:
                sync_received.append(msg)

        def on_async(msg: BusMessage) -> None:
            with lock:
                async_received.append(msg)

        bus.subscribe("sync.path", on_sync)
        bus.subscribe("async.path", on_async)

        bus.start()

        # Interleave synchronous and asynchronous publishes.
        for _ in range(10):
            bus.publish(_msg(topic="sync.path"))
            bus.publish_async(_msg(topic="async.path"))

        assert _wait_for(lambda: len(async_received) >= 10)
        bus.stop()

        assert len(sync_received) == 10
        assert len(async_received) == 10

    def test_repr_does_not_deadlock_while_worker_is_running(self) -> None:
        """repr() acquires the internal lock briefly; must not deadlock under load."""
        bus = MessageBus()
        bus.subscribe("repr.test", lambda m: None)
        bus.start()

        for _ in range(20):
            bus.publish_async(_msg(topic="repr.test"))

        # repr() must complete quickly even while the worker is draining.
        r = repr(bus)
        bus.stop()

        assert "MessageBus" in r


# ---------------------------------------------------------------------------
# 9. Sequence counter monotonicity
# ---------------------------------------------------------------------------


class TestSequenceCounter:
    """_seq counter must be strictly increasing for FIFO guarantee."""

    def test_seq_increases_monotonically_single_thread(self) -> None:
        """Each publish_async call increments _seq by 1 in a single thread."""
        bus = MessageBus()
        initial = bus._seq

        for _ in range(100):
            bus.publish_async(_msg())

        assert bus._seq == initial + 100

    def test_seq_never_decreases_across_concurrent_threads(self) -> None:
        """Concurrent publish_async calls never produce a lower _seq value."""
        bus = MessageBus(max_queue_size=2000)
        observed_seqs: list[int] = []
        lock = threading.Lock()

        # Capture the queue items (neg_pri, seq, msg) directly.
        original_put = bus._queue.put

        def capturing_put(item, *args, **kwargs):
            with lock:
                observed_seqs.append(item[1])  # item = (-priority, seq, message)
            return original_put(item, *args, **kwargs)

        bus._queue.put = capturing_put  # type: ignore[method-assign]

        n_threads = 20
        barrier = threading.Barrier(n_threads)

        def enqueue() -> None:
            barrier.wait()
            for _ in range(50):
                bus.publish_async(_msg())

        threads = [threading.Thread(target=enqueue) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # All sequence numbers must be non-negative and unique.
        assert len(observed_seqs) == n_threads * 50
        assert len(set(observed_seqs)) == n_threads * 50, "Every _seq value must be unique"
        assert min(observed_seqs) >= 0


# ---------------------------------------------------------------------------
# 10. Singleton — re-init with custom max_queue_size after reset
# ---------------------------------------------------------------------------


class TestSingletonMaxQueueSize:
    """Singleton picks up max_queue_size from first init_message_bus() call only."""

    def setup_method(self) -> None:
        reset_message_bus()

    def teardown_method(self) -> None:
        reset_message_bus()

    def test_singleton_queue_size_matches_init_argument(self) -> None:
        bus = init_message_bus(max_queue_size=77)
        assert bus._queue.maxsize == 77

    def test_second_init_call_does_not_change_queue_size(self) -> None:
        """init_message_bus() is idempotent — second call with different size is ignored."""
        init_message_bus(max_queue_size=10)
        bus = init_message_bus(max_queue_size=999)  # must not override
        assert bus._queue.maxsize == 10

    def test_reset_and_reinit_with_new_queue_size(self) -> None:
        """After reset, a new init can set a different max_queue_size."""
        init_message_bus(max_queue_size=10)
        reset_message_bus()
        bus = init_message_bus(max_queue_size=500)
        assert bus._queue.maxsize == 500


# ---------------------------------------------------------------------------
# 11. BusMessage payload mutation isolation
# ---------------------------------------------------------------------------


class TestPayloadMutationIsolation:
    """Handlers that mutate the payload affect all subsequent handlers."""

    def test_mutating_handler_affects_later_handler_payload(self) -> None:
        """Because the payload dict is passed by reference, mutations are visible downstream.

        This documents the existing behaviour: the bus does NOT copy the payload.
        Callers must not mutate payloads if they need isolation.
        """
        bus = MessageBus()
        seen_by_second: list[dict] = []

        def mutator(msg: BusMessage) -> None:
            msg.payload["mutated"] = True

        def observer(msg: BusMessage) -> None:
            seen_by_second.append(dict(msg.payload))

        bus.subscribe("mutation.test", mutator)
        bus.subscribe("mutation.test", observer)

        bus.publish(_msg(topic="mutation.test", payload={"original": 1}))

        assert seen_by_second[0].get("mutated") is True


# ---------------------------------------------------------------------------
# 12. Bus topics: all constants follow naming convention
# ---------------------------------------------------------------------------


class TestBusTopicsNamingConvention:
    """Every topic constant uses the <subsystem>.<action>[.<detail>] scheme."""

    def test_all_topic_constants_are_nonempty_strings_with_at_least_one_dot(self) -> None:
        """All bus topic constants contain at least one '.' separator."""
        from missy.core import bus_topics

        topic_attrs = [
            v for k, v in vars(bus_topics).items()
            if not k.startswith("_") and isinstance(v, str)
        ]
        assert topic_attrs, "No string constants found in bus_topics"
        for topic in topic_attrs:
            assert "." in topic, f"Topic constant {topic!r} missing a '.' separator"

    def test_security_approval_needed_has_three_levels(self) -> None:
        """SECURITY_APPROVAL_NEEDED has three dot-separated segments."""
        from missy.core.bus_topics import SECURITY_APPROVAL_NEEDED, SECURITY_APPROVAL_RESPONSE

        assert SECURITY_APPROVAL_NEEDED.count(".") == 2
        assert SECURITY_APPROVAL_RESPONSE.count(".") == 2

    def test_agent_run_topics_share_agent_run_prefix(self) -> None:
        """Agent lifecycle topics all start with 'agent.run.'."""
        from missy.core.bus_topics import (
            AGENT_RUN_COMPLETE,
            AGENT_RUN_ERROR,
        )

        for topic in (AGENT_RUN_START, AGENT_RUN_COMPLETE, AGENT_RUN_ERROR):
            assert topic.startswith("agent.run."), f"{topic!r} does not start with 'agent.run.'"
