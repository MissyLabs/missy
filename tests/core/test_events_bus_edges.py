"""Session 14: Edge case tests for AuditEvent, EventBus, and core exceptions.

Covers:
- AuditEvent: timezone validation, now() factory, detail defaults
- EventBus: subscribe/unsubscribe, publish with failing callbacks,
  concurrent publish, get_events filtering, clear
- Exceptions: PolicyViolationError attributes
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timezone

import pytest

from missy.core.events import AuditEvent, EventBus, event_bus

# ---------------------------------------------------------------------------
# AuditEvent tests
# ---------------------------------------------------------------------------


class TestAuditEventEdgeCases:
    """Edge cases for AuditEvent."""

    def test_now_factory(self):
        event = AuditEvent.now(
            session_id="s1", task_id="t1",
            event_type="test.event", category="network",
            result="allow",
        )
        assert event.session_id == "s1"
        assert event.timestamp.tzinfo is not None
        assert event.detail == {}
        assert event.policy_rule is None

    def test_now_with_detail(self):
        event = AuditEvent.now(
            session_id="s1", task_id="t1",
            event_type="test.event", category="network",
            result="deny", detail={"host": "evil.com"},
            policy_rule="block-evil",
        )
        assert event.detail["host"] == "evil.com"
        assert event.policy_rule == "block-evil"

    def test_naive_timestamp_raises(self):
        """Naive (timezone-unaware) timestamp should raise ValueError."""
        with pytest.raises(ValueError, match="timezone-aware"):
            AuditEvent(
                timestamp=datetime(2026, 3, 19, 12, 0, 0),  # no tzinfo
                session_id="s1", task_id="t1",
                event_type="test", category="network",
                result="allow",
            )

    def test_utc_timestamp_ok(self):
        event = AuditEvent(
            timestamp=datetime.now(tz=UTC),
            session_id="s1", task_id="t1",
            event_type="test", category="network",
            result="allow",
        )
        assert event.timestamp.tzinfo is not None

    def test_non_utc_timezone_ok(self):
        """Any timezone-aware datetime should be accepted."""
        eastern = timezone(offset=datetime.now(tz=UTC).utcoffset() or __import__("datetime").timedelta(hours=-5))
        event = AuditEvent(
            timestamp=datetime.now(tz=eastern),
            session_id="s1", task_id="t1",
            event_type="test", category="network",
            result="allow",
        )
        assert event.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# EventBus tests
# ---------------------------------------------------------------------------


class TestEventBusEdgeCases:
    """Edge cases for EventBus."""

    def test_publish_no_subscribers(self):
        """Publishing with no subscribers should not raise."""
        bus = EventBus()
        event = AuditEvent.now(
            session_id="s1", task_id="t1",
            event_type="orphan.event", category="network",
            result="allow",
        )
        bus.publish(event)
        assert len(bus.get_events()) == 1

    def test_subscribe_and_receive(self):
        bus = EventBus()
        received = []
        bus.subscribe("test.event", lambda e: received.append(e))
        event = AuditEvent.now(
            session_id="s1", task_id="t1",
            event_type="test.event", category="network",
            result="allow",
        )
        bus.publish(event)
        assert len(received) == 1
        assert received[0] is event

    def test_unsubscribe(self):
        bus = EventBus()
        received = []

        def callback(e):
            received.append(e)

        bus.subscribe("test.event", callback)
        bus.unsubscribe("test.event", callback)
        bus.publish(AuditEvent.now(
            session_id="s1", task_id="t1",
            event_type="test.event", category="network",
            result="allow",
        ))
        assert len(received) == 0

    def test_unsubscribe_nonexistent(self):
        """Unsubscribing a non-registered callback should be a no-op."""
        bus = EventBus()
        bus.unsubscribe("test.event", lambda e: None)  # Should not raise

    def test_unsubscribe_wrong_event_type(self):
        """Unsubscribing from wrong event type should be a no-op."""
        bus = EventBus()

        def callback(e):
            pass

        bus.subscribe("type_a", callback)
        bus.unsubscribe("type_b", callback)  # Should not raise

    def test_callback_exception_doesnt_break_others(self):
        """A failing callback should not prevent other callbacks from firing."""
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
        bus.subscribe("test", lambda e: received.append(e))
        bus.publish(AuditEvent.now(
            session_id="s1", task_id="t1",
            event_type="test", category="network",
            result="allow",
        ))
        # Second callback should still fire despite first one failing
        # Note: generator throw is not called until iterated, so this won't fail
        # Let's use a proper failing callback instead
        bus2 = EventBus()
        received2 = []

        def bad_callback(e):
            raise RuntimeError("boom")

        bus2.subscribe("test", bad_callback)
        bus2.subscribe("test", lambda e: received2.append(e))
        bus2.publish(AuditEvent.now(
            session_id="s1", task_id="t1",
            event_type="test", category="network",
            result="allow",
        ))
        assert len(received2) == 1

    def test_get_events_no_filter(self):
        bus = EventBus()
        for i in range(5):
            bus.publish(AuditEvent.now(
                session_id="s1", task_id=f"t{i}",
                event_type="test", category="network",
                result="allow",
            ))
        assert len(bus.get_events()) == 5

    def test_get_events_filter_by_category(self):
        bus = EventBus()
        bus.publish(AuditEvent.now(
            session_id="s1", task_id="t1",
            event_type="a", category="network", result="allow",
        ))
        bus.publish(AuditEvent.now(
            session_id="s1", task_id="t2",
            event_type="b", category="filesystem", result="allow",
        ))
        network_events = bus.get_events(category="network")
        assert len(network_events) == 1
        assert network_events[0].task_id == "t1"

    def test_get_events_filter_by_result(self):
        bus = EventBus()
        bus.publish(AuditEvent.now(
            session_id="s1", task_id="t1",
            event_type="a", category="network", result="allow",
        ))
        bus.publish(AuditEvent.now(
            session_id="s1", task_id="t2",
            event_type="b", category="network", result="deny",
        ))
        denied = bus.get_events(result="deny")
        assert len(denied) == 1

    def test_get_events_filter_by_session(self):
        bus = EventBus()
        bus.publish(AuditEvent.now(
            session_id="s1", task_id="t1",
            event_type="a", category="network", result="allow",
        ))
        bus.publish(AuditEvent.now(
            session_id="s2", task_id="t2",
            event_type="b", category="network", result="allow",
        ))
        s1_events = bus.get_events(session_id="s1")
        assert len(s1_events) == 1

    def test_get_events_combined_filters(self):
        bus = EventBus()
        bus.publish(AuditEvent.now(
            session_id="s1", task_id="t1",
            event_type="net.req", category="network", result="deny",
        ))
        bus.publish(AuditEvent.now(
            session_id="s1", task_id="t2",
            event_type="net.req", category="network", result="allow",
        ))
        events = bus.get_events(session_id="s1", result="deny")
        assert len(events) == 1
        assert events[0].task_id == "t1"

    def test_clear(self):
        bus = EventBus()
        bus.subscribe("test", lambda e: None)
        bus.publish(AuditEvent.now(
            session_id="s1", task_id="t1",
            event_type="test", category="network", result="allow",
        ))
        bus.clear()
        assert bus.get_events() == []

    def test_concurrent_publish(self):
        """Concurrent publishes should be thread-safe."""
        bus = EventBus()
        count = {"n": 0}
        lock = threading.Lock()

        def counter(e):
            with lock:
                count["n"] += 1

        bus.subscribe("concurrent", counter)

        def publish_many():
            for _ in range(50):
                bus.publish(AuditEvent.now(
                    session_id="s1", task_id="t1",
                    event_type="concurrent", category="network",
                    result="allow",
                ))

        threads = [threading.Thread(target=publish_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert count["n"] == 250
        assert len(bus.get_events()) == 250

    def test_multiple_subscribers_same_event(self):
        bus = EventBus()
        r1, r2 = [], []
        bus.subscribe("multi", lambda e: r1.append(e))
        bus.subscribe("multi", lambda e: r2.append(e))
        bus.publish(AuditEvent.now(
            session_id="s1", task_id="t1",
            event_type="multi", category="network", result="allow",
        ))
        assert len(r1) == 1
        assert len(r2) == 1


# ---------------------------------------------------------------------------
# Core exceptions tests
# ---------------------------------------------------------------------------


class TestPolicyViolationError:
    """Tests for PolicyViolationError."""

    def test_basic_creation(self):
        from missy.core.exceptions import PolicyViolationError

        err = PolicyViolationError(
            "Network access denied to evil.com",
            category="network", detail="host blocked",
        )
        assert "evil.com" in str(err)

    def test_with_raise(self):
        from missy.core.exceptions import PolicyViolationError

        with pytest.raises(PolicyViolationError):
            raise PolicyViolationError("denied", category="network", detail="blocked")

    def test_attributes(self):
        from missy.core.exceptions import PolicyViolationError

        err = PolicyViolationError("msg", category="filesystem", detail="path denied")
        assert err.category == "filesystem"
        assert err.detail == "path denied"


# ---------------------------------------------------------------------------
# Module singleton tests
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    """Tests for the module-level event_bus singleton."""

    def test_event_bus_is_eventbus(self):
        assert isinstance(event_bus, EventBus)

    def test_event_bus_is_usable(self):
        # Publish to module-level bus
        event = AuditEvent.now(
            session_id="test", task_id="test",
            event_type="test.singleton", category="network",
            result="allow",
        )
        event_bus.publish(event)
        # Should not raise
