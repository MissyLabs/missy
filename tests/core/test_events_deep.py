"""Deep unit tests for missy/core/events.py.

Covers AuditEvent construction, the EventBus publish/subscribe/filter
contract, thread safety, isolation between subscribers, and the
module-level singleton.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from missy.core.events import AuditEvent, EventBus, event_bus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    *,
    event_type: str = "network.request",
    category: str = "network",
    result: str = "allow",
    session_id: str = "sess-default",
    task_id: str = "task-default",
    detail: dict | None = None,
    policy_rule: str | None = None,
) -> AuditEvent:
    """Return an AuditEvent with sensible defaults for tests."""
    return AuditEvent.now(
        session_id=session_id,
        task_id=task_id,
        event_type=event_type,
        category=category,  # type: ignore[arg-type]
        result=result,  # type: ignore[arg-type]
        detail=detail,
        policy_rule=policy_rule,
    )


# ---------------------------------------------------------------------------
# AuditEvent construction
# ---------------------------------------------------------------------------


class TestAuditEventNowFactory:
    """Tests for the AuditEvent.now() classmethod factory."""

    def test_now_returns_audit_event_instance(self) -> None:
        event = _make_event()
        assert isinstance(event, AuditEvent)

    def test_now_timestamp_is_utc_aware(self) -> None:
        event = _make_event()
        assert event.timestamp.tzinfo is not None
        # Normalise: utcoffset() == 0 means UTC-equivalent
        assert event.timestamp.utcoffset() == timedelta(0)

    def test_now_timestamp_is_recent(self) -> None:
        before = datetime.now(tz=UTC)
        event = _make_event()
        after = datetime.now(tz=UTC)
        assert before <= event.timestamp <= after

    def test_now_carries_session_id(self) -> None:
        event = _make_event(session_id="my-session")
        assert event.session_id == "my-session"

    def test_now_carries_task_id(self) -> None:
        event = _make_event(task_id="my-task")
        assert event.task_id == "my-task"

    def test_now_carries_event_type(self) -> None:
        event = _make_event(event_type="shell.execute")
        assert event.event_type == "shell.execute"

    def test_now_carries_category(self) -> None:
        event = _make_event(category="filesystem")
        assert event.category == "filesystem"

    def test_now_carries_result(self) -> None:
        event = _make_event(result="deny")
        assert event.result == "deny"

    def test_now_detail_defaults_to_empty_dict(self) -> None:
        event = _make_event()
        assert event.detail == {}

    def test_now_detail_none_becomes_empty_dict(self) -> None:
        event = _make_event(detail=None)
        assert event.detail == {}

    def test_now_detail_is_preserved_when_provided(self) -> None:
        payload = {"url": "https://example.com", "status": 200}
        event = _make_event(detail=payload)
        assert event.detail == payload

    def test_now_policy_rule_defaults_to_none(self) -> None:
        event = _make_event()
        assert event.policy_rule is None

    def test_now_policy_rule_is_preserved_when_provided(self) -> None:
        event = _make_event(policy_rule="allow-anthropic-preset")
        assert event.policy_rule == "allow-anthropic-preset"

    def test_now_empty_strings_accepted(self) -> None:
        event = _make_event(session_id="", task_id="", event_type="")
        assert event.session_id == ""
        assert event.task_id == ""
        assert event.event_type == ""


class TestAuditEventCategories:
    """Verify all documented EventCategory literals are accepted."""

    @pytest.mark.parametrize(
        "category",
        ["network", "filesystem", "shell", "plugin", "scheduler", "provider", "channel"],
    )
    def test_category_accepted(self, category: str) -> None:
        event = _make_event(category=category)
        assert event.category == category


class TestAuditEventResults:
    """Verify all documented EventResult literals are accepted."""

    @pytest.mark.parametrize("result", ["allow", "deny", "error"])
    def test_result_accepted(self, result: str) -> None:
        event = _make_event(result=result)
        assert event.result == result


class TestAuditEventPostInit:
    """Tests for the __post_init__ validation guard."""

    def test_timezone_aware_timestamp_accepted(self) -> None:
        event = AuditEvent(
            timestamp=datetime.now(tz=UTC),
            session_id="s",
            task_id="t",
            event_type="x",
            category="network",
            result="allow",
        )
        assert event.session_id == "s"

    def test_naive_timestamp_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            AuditEvent(
                timestamp=datetime(2024, 6, 1, 12, 0, 0),  # naive — no tzinfo
                session_id="s",
                task_id="t",
                event_type="x",
                category="network",
                result="allow",
            )

    def test_explicit_utc_timestamp_accepted(self) -> None:
        ts = datetime(2025, 1, 15, 8, 30, 0, tzinfo=UTC)
        event = AuditEvent(
            timestamp=ts,
            session_id="s",
            task_id="t",
            event_type="x",
            category="network",
            result="allow",
        )
        assert event.timestamp == ts

    def test_non_utc_aware_timestamp_is_accepted(self) -> None:
        # Any aware timestamp is accepted — the validator only rejects naive ones.
        eastern = timezone(timedelta(hours=-5))
        ts = datetime(2025, 3, 1, 10, 0, 0, tzinfo=eastern)
        event = AuditEvent(
            timestamp=ts,
            session_id="s",
            task_id="t",
            event_type="x",
            category="network",
            result="allow",
        )
        assert event.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# EventBus — subscribe / unsubscribe
# ---------------------------------------------------------------------------


class TestEventBusSubscribe:
    """Tests for EventBus.subscribe() and basic callback invocation."""

    def test_subscribe_callback_receives_published_event(self) -> None:
        bus = EventBus()
        received: list[AuditEvent] = []
        bus.subscribe("test.event", received.append)
        event = _make_event(event_type="test.event")
        bus.publish(event)
        assert received == [event]

    def test_subscribe_multiple_callbacks_for_same_event_type(self) -> None:
        bus = EventBus()
        calls_a: list[AuditEvent] = []
        calls_b: list[AuditEvent] = []
        bus.subscribe("test.event", calls_a.append)
        bus.subscribe("test.event", calls_b.append)
        event = _make_event(event_type="test.event")
        bus.publish(event)
        assert calls_a == [event]
        assert calls_b == [event]

    def test_subscribe_callbacks_called_in_registration_order(self) -> None:
        bus = EventBus()
        order: list[str] = []
        bus.subscribe("seq.event", lambda _: order.append("first"))
        bus.subscribe("seq.event", lambda _: order.append("second"))
        bus.subscribe("seq.event", lambda _: order.append("third"))
        bus.publish(_make_event(event_type="seq.event"))
        assert order == ["first", "second", "third"]

    def test_callback_not_triggered_for_different_event_type(self) -> None:
        bus = EventBus()
        triggered: list[AuditEvent] = []
        bus.subscribe("network.request", triggered.append)
        bus.publish(_make_event(event_type="shell.execute"))
        assert triggered == []

    def test_callback_only_triggered_for_subscribed_type(self) -> None:
        bus = EventBus()
        network_events: list[AuditEvent] = []
        shell_events: list[AuditEvent] = []
        bus.subscribe("network.request", network_events.append)
        bus.subscribe("shell.execute", shell_events.append)

        bus.publish(_make_event(event_type="network.request"))
        bus.publish(_make_event(event_type="shell.execute"))

        assert len(network_events) == 1
        assert len(shell_events) == 1
        assert network_events[0].event_type == "network.request"
        assert shell_events[0].event_type == "shell.execute"

    def test_subscribing_same_callback_twice_calls_it_twice(self) -> None:
        bus = EventBus()
        counter: list[int] = []
        cb = lambda _: counter.append(1)  # noqa: E731
        bus.subscribe("x.event", cb)
        bus.subscribe("x.event", cb)
        bus.publish(_make_event(event_type="x.event"))
        assert len(counter) == 2


class TestEventBusUnsubscribe:
    """Tests for EventBus.unsubscribe()."""

    def test_unsubscribe_stops_callback_from_receiving_events(self) -> None:
        bus = EventBus()
        received: list[AuditEvent] = []
        bus.subscribe("test.event", received.append)
        bus.unsubscribe("test.event", received.append)
        bus.publish(_make_event(event_type="test.event"))
        assert received == []

    def test_unsubscribe_unknown_callback_is_noop(self) -> None:
        bus = EventBus()
        cb = MagicMock()
        # Never subscribed — must not raise
        bus.unsubscribe("test.event", cb)
        cb.assert_not_called()

    def test_unsubscribe_from_unknown_event_type_is_noop(self) -> None:
        bus = EventBus()
        cb = MagicMock()
        # Event type never subscribed to — must not raise
        bus.unsubscribe("completely.unknown", cb)

    def test_unsubscribe_wrong_event_type_leaves_correct_subscription(self) -> None:
        bus = EventBus()
        received: list[AuditEvent] = []
        bus.subscribe("a.event", received.append)
        bus.unsubscribe("b.event", received.append)  # wrong type — noop
        bus.publish(_make_event(event_type="a.event"))
        assert len(received) == 1

    def test_unsubscribe_only_removes_first_matching_instance(self) -> None:
        """When the same callback is registered twice, one unsubscribe removes one."""
        bus = EventBus()
        counter: list[int] = []
        cb = lambda _: counter.append(1)  # noqa: E731
        bus.subscribe("x.event", cb)
        bus.subscribe("x.event", cb)
        bus.unsubscribe("x.event", cb)
        bus.publish(_make_event(event_type="x.event"))
        # One registration remains
        assert len(counter) == 1

    def test_unsubscribe_after_clear_is_noop(self) -> None:
        bus = EventBus()
        cb = MagicMock()
        bus.subscribe("x.event", cb)
        bus.clear()
        bus.unsubscribe("x.event", cb)  # must not raise


# ---------------------------------------------------------------------------
# EventBus — publish
# ---------------------------------------------------------------------------


class TestEventBusPublish:
    """Tests for EventBus.publish()."""

    def test_publish_stores_event_in_log(self) -> None:
        bus = EventBus()
        event = _make_event(event_type="store.test")
        bus.publish(event)
        assert event in bus.get_events()

    def test_publish_stores_events_in_order(self) -> None:
        bus = EventBus()
        e1 = _make_event(event_type="ordered.event", session_id="s1")
        e2 = _make_event(event_type="ordered.event", session_id="s2")
        e3 = _make_event(event_type="ordered.event", session_id="s3")
        bus.publish(e1)
        bus.publish(e2)
        bus.publish(e3)
        stored = bus.get_events(event_type="ordered.event")
        assert stored == [e1, e2, e3]

    def test_publish_with_no_subscribers_does_not_raise(self) -> None:
        bus = EventBus()
        # Must complete silently even with no subscribers
        bus.publish(_make_event(event_type="orphan.event"))

    def test_raising_callback_does_not_propagate(self) -> None:
        bus = EventBus()
        bus.subscribe("bad.event", MagicMock(side_effect=RuntimeError("boom")))
        # Must not raise from publish
        bus.publish(_make_event(event_type="bad.event"))

    def test_raising_callback_does_not_prevent_later_callbacks(self) -> None:
        bus = EventBus()
        good_results: list[AuditEvent] = []
        bus.subscribe("mixed.event", MagicMock(side_effect=ValueError("oops")))
        bus.subscribe("mixed.event", good_results.append)
        bus.publish(_make_event(event_type="mixed.event"))
        assert len(good_results) == 1

    def test_raising_callback_logged_at_error_level(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        bus = EventBus()
        bus.subscribe("err.event", MagicMock(side_effect=RuntimeError("logged error")))
        with caplog.at_level(logging.ERROR, logger="missy.core.events"):
            bus.publish(_make_event(event_type="err.event"))
        assert any("Unhandled exception" in r.message for r in caplog.records)

    def test_event_stored_even_when_all_callbacks_raise(self) -> None:
        bus = EventBus()
        bus.subscribe("fail.event", MagicMock(side_effect=RuntimeError("x")))
        bus.subscribe("fail.event", MagicMock(side_effect=RuntimeError("y")))
        event = _make_event(event_type="fail.event")
        bus.publish(event)
        assert event in bus.get_events(event_type="fail.event")

    def test_multiple_sequential_publishes_each_stored(self) -> None:
        bus = EventBus()
        for i in range(10):
            bus.publish(_make_event(event_type="bulk.event", session_id=f"s{i}"))
        assert len(bus.get_events(event_type="bulk.event")) == 10


# ---------------------------------------------------------------------------
# EventBus — get_events filtering
# ---------------------------------------------------------------------------


class TestEventBusGetEventsFiltering:
    """Tests for EventBus.get_events() with all filter dimensions."""

    def setup_method(self) -> None:
        self.bus = EventBus()
        # Populate a diverse set of events
        self.bus.publish(
            _make_event(
                event_type="network.request",
                category="network",
                result="allow",
                session_id="sess-A",
                task_id="task-1",
            )
        )
        self.bus.publish(
            _make_event(
                event_type="network.request",
                category="network",
                result="deny",
                session_id="sess-A",
                task_id="task-2",
            )
        )
        self.bus.publish(
            _make_event(
                event_type="shell.execute",
                category="shell",
                result="allow",
                session_id="sess-B",
                task_id="task-1",
            )
        )
        self.bus.publish(
            _make_event(
                event_type="filesystem.read",
                category="filesystem",
                result="error",
                session_id="sess-B",
                task_id="task-3",
            )
        )

    def test_no_filters_returns_all_events(self) -> None:
        all_events = self.bus.get_events()
        assert len(all_events) == 4

    def test_filter_by_event_type(self) -> None:
        events = self.bus.get_events(event_type="network.request")
        assert len(events) == 2
        assert all(e.event_type == "network.request" for e in events)

    def test_filter_by_category_network(self) -> None:
        events = self.bus.get_events(category="network")
        assert len(events) == 2
        assert all(e.category == "network" for e in events)

    def test_filter_by_category_shell(self) -> None:
        events = self.bus.get_events(category="shell")
        assert len(events) == 1
        assert events[0].event_type == "shell.execute"

    def test_filter_by_category_filesystem(self) -> None:
        events = self.bus.get_events(category="filesystem")
        assert len(events) == 1
        assert events[0].result == "error"

    def test_filter_by_session_id(self) -> None:
        events = self.bus.get_events(session_id="sess-A")
        assert len(events) == 2
        assert all(e.session_id == "sess-A" for e in events)

    def test_filter_by_session_id_returns_empty_for_unknown(self) -> None:
        events = self.bus.get_events(session_id="sess-UNKNOWN")
        assert events == []

    def test_filter_by_task_id(self) -> None:
        events = self.bus.get_events(task_id="task-1")
        assert len(events) == 2
        assert all(e.task_id == "task-1" for e in events)

    def test_filter_by_task_id_unique(self) -> None:
        events = self.bus.get_events(task_id="task-3")
        assert len(events) == 1
        assert events[0].event_type == "filesystem.read"

    def test_filter_by_result_allow(self) -> None:
        events = self.bus.get_events(result="allow")
        assert len(events) == 2
        assert all(e.result == "allow" for e in events)

    def test_filter_by_result_deny(self) -> None:
        events = self.bus.get_events(result="deny")
        assert len(events) == 1
        assert events[0].session_id == "sess-A"
        assert events[0].task_id == "task-2"

    def test_filter_by_result_error(self) -> None:
        events = self.bus.get_events(result="error")
        assert len(events) == 1
        assert events[0].event_type == "filesystem.read"

    def test_filter_event_type_and_result(self) -> None:
        events = self.bus.get_events(event_type="network.request", result="deny")
        assert len(events) == 1
        assert events[0].task_id == "task-2"

    def test_filter_session_and_event_type(self) -> None:
        events = self.bus.get_events(session_id="sess-A", event_type="network.request")
        assert len(events) == 2

    def test_filter_session_and_task_and_result(self) -> None:
        events = self.bus.get_events(
            session_id="sess-A",
            task_id="task-1",
            result="allow",
        )
        assert len(events) == 1
        assert events[0].event_type == "network.request"

    def test_filter_all_criteria_returns_single_match(self) -> None:
        events = self.bus.get_events(
            event_type="network.request",
            category="network",
            session_id="sess-A",
            task_id="task-2",
            result="deny",
        )
        assert len(events) == 1

    def test_filter_all_criteria_no_match(self) -> None:
        events = self.bus.get_events(
            event_type="network.request",
            category="network",
            session_id="sess-A",
            task_id="task-2",
            result="allow",  # deny, not allow — no match
        )
        assert events == []

    def test_get_events_returns_new_list_each_call(self) -> None:
        a = self.bus.get_events()
        b = self.bus.get_events()
        assert a is not b
        assert a == b


# ---------------------------------------------------------------------------
# EventBus — clear
# ---------------------------------------------------------------------------


class TestEventBusClear:
    """Tests for EventBus.clear()."""

    def test_clear_removes_all_stored_events(self) -> None:
        bus = EventBus()
        bus.publish(_make_event())
        bus.publish(_make_event())
        bus.clear()
        assert bus.get_events() == []

    def test_clear_removes_all_subscribers(self) -> None:
        bus = EventBus()
        triggered: list[AuditEvent] = []
        bus.subscribe("test.event", triggered.append)
        bus.clear()
        bus.publish(_make_event(event_type="test.event"))
        # Subscriber was cleared, so nothing received
        assert triggered == []

    def test_clear_is_idempotent(self) -> None:
        bus = EventBus()
        bus.clear()
        bus.clear()
        assert bus.get_events() == []

    def test_publish_after_clear_works_normally(self) -> None:
        bus = EventBus()
        bus.publish(_make_event())
        bus.clear()
        new_event = _make_event(event_type="after.clear")
        bus.publish(new_event)
        stored = bus.get_events()
        assert stored == [new_event]

    def test_subscribe_after_clear_works_normally(self) -> None:
        bus = EventBus()
        bus.subscribe("x.event", MagicMock())
        bus.clear()
        received: list[AuditEvent] = []
        bus.subscribe("x.event", received.append)
        bus.publish(_make_event(event_type="x.event"))
        assert len(received) == 1


# ---------------------------------------------------------------------------
# EventBus — thread safety
# ---------------------------------------------------------------------------


class TestEventBusThreadSafety:
    """Tests that EventBus publish/subscribe/get are thread-safe."""

    def test_concurrent_publishes_all_stored(self) -> None:
        bus = EventBus()
        num_threads = 20
        events_per_thread = 50
        barrier = threading.Barrier(num_threads)

        def publish_batch(session_id: str) -> None:
            barrier.wait()  # all threads start simultaneously
            for i in range(events_per_thread):
                bus.publish(
                    _make_event(
                        event_type="concurrent.event",
                        session_id=session_id,
                        task_id=str(i),
                    )
                )

        threads = [
            threading.Thread(target=publish_batch, args=(f"s{n}",)) for n in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = len(bus.get_events(event_type="concurrent.event"))
        assert total == num_threads * events_per_thread

    def test_concurrent_subscribe_and_publish_no_exception(self) -> None:
        bus = EventBus()
        errors: list[Exception] = []

        def subscribe_loop() -> None:
            try:
                for _i in range(100):
                    cb = MagicMock()
                    bus.subscribe("race.event", cb)
            except Exception as exc:
                errors.append(exc)

        def publish_loop() -> None:
            try:
                for _ in range(100):
                    bus.publish(_make_event(event_type="race.event"))
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=subscribe_loop)
        t2 = threading.Thread(target=publish_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_get_events_returns_consistent_snapshot(self) -> None:
        """Concurrent reads must not raise even when writes are in-flight."""
        bus = EventBus()
        errors: list[Exception] = []

        def writer() -> None:
            for _ in range(200):
                bus.publish(_make_event(event_type="snap.event"))

        def reader() -> None:
            try:
                for _ in range(200):
                    result = bus.get_events(event_type="snap.event")
                    # Must be a list (type is guaranteed even if empty mid-write)
                    assert isinstance(result, list)
            except Exception as exc:
                errors.append(exc)

        wt = threading.Thread(target=writer)
        rt = threading.Thread(target=reader)
        wt.start()
        rt.start()
        wt.join()
        rt.join()

        assert errors == [], f"Reader thread errors: {errors}"

    def test_publish_from_multiple_threads_triggers_callbacks_safely(self) -> None:
        bus = EventBus()
        lock = threading.Lock()
        received: list[AuditEvent] = []

        def safe_append(e: AuditEvent) -> None:
            with lock:
                received.append(e)

        bus.subscribe("mt.event", safe_append)
        barrier = threading.Barrier(10)

        def worker() -> None:
            barrier.wait()
            bus.publish(_make_event(event_type="mt.event"))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 10


# ---------------------------------------------------------------------------
# EventBus — module-level singleton
# ---------------------------------------------------------------------------


class TestModuleLevelSingleton:
    """Tests for the module-level event_bus singleton."""

    def setup_method(self) -> None:
        event_bus.clear()

    def teardown_method(self) -> None:
        event_bus.clear()

    def test_singleton_is_event_bus_instance(self) -> None:
        assert isinstance(event_bus, EventBus)

    def test_singleton_persists_events_across_imports(self) -> None:
        from missy.core.events import event_bus as bus2

        event = _make_event(event_type="singleton.test")
        event_bus.publish(event)
        assert event in bus2.get_events(event_type="singleton.test")

    def test_singleton_clear_visible_across_references(self) -> None:
        from missy.core.events import event_bus as bus2

        event_bus.publish(_make_event(event_type="visible.test"))
        assert len(bus2.get_events(event_type="visible.test")) == 1
        event_bus.clear()
        assert bus2.get_events() == []
