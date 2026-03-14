"""Coverage tests for missy/core/events.py.

Targets uncovered lines:
  69        : AuditEvent.__post_init__ — naive timestamp raises ValueError
  152-156   : EventBus.unsubscribe — callback not registered (ValueError swallowed)
  175-176   : EventBus.publish — callback raises exception (logged, not re-raised)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from missy.core.events import AuditEvent, EventBus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(event_type: str = "test.event") -> AuditEvent:
    return AuditEvent.now(
        session_id="sess-1",
        task_id="task-1",
        event_type=event_type,
        category="network",
        result="allow",
    )


# ---------------------------------------------------------------------------
# AuditEvent.__post_init__  (line 69)
# ---------------------------------------------------------------------------


class TestAuditEventPostInit:
    def test_timezone_aware_timestamp_accepted(self):
        """Normal path: aware timestamp does not raise."""
        event = AuditEvent(
            timestamp=datetime.now(tz=UTC),
            session_id="s",
            task_id="t",
            event_type="x",
            category="network",
            result="allow",
        )
        assert event.session_id == "s"

    def test_naive_timestamp_raises_value_error(self):
        """Line 69: naive datetime raises ValueError inside __post_init__."""
        with pytest.raises(ValueError, match="timezone-aware"):
            AuditEvent(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),  # naive — no tzinfo
                session_id="s",
                task_id="t",
                event_type="x",
                category="network",
                result="allow",
            )


# ---------------------------------------------------------------------------
# EventBus.unsubscribe  (lines 152-156)
# ---------------------------------------------------------------------------


class TestEventBusUnsubscribe:
    def test_unsubscribe_registered_callback_removes_it(self):
        bus = EventBus()
        received = []
        def cb(e):
            return received.append(e)
        bus.subscribe("test.event", cb)
        bus.unsubscribe("test.event", cb)

        bus.publish(_make_event("test.event"))
        assert received == []

    def test_unsubscribe_unregistered_callback_is_noop(self):
        """Lines 153-156: ValueError inside unsubscribe is caught and ignored."""
        bus = EventBus()
        def cb(e):
            return None
        # Never registered — should not raise
        bus.unsubscribe("test.event", cb)

    def test_unsubscribe_wrong_event_type_is_noop(self):
        """Callback registered for a different event type — also a noop."""
        bus = EventBus()
        def cb(e):
            return None
        bus.subscribe("other.event", cb)
        # Unsubscribe from a type where cb was never added
        bus.unsubscribe("test.event", cb)

    def test_unsubscribe_for_unknown_event_type_is_noop(self):
        """Event type never subscribed to at all — no KeyError or ValueError."""
        bus = EventBus()
        def cb(e):
            return None
        bus.unsubscribe("completely.unknown", cb)


# ---------------------------------------------------------------------------
# EventBus.publish — callback raises  (lines 175-176)
# ---------------------------------------------------------------------------


class TestEventBusPublishCallbackException:
    def test_raising_callback_does_not_propagate(self):
        """Lines 175-176: exception inside subscriber is caught and logged."""
        bus = EventBus()
        bad_cb = MagicMock(side_effect=RuntimeError("boom"))
        good_results = []
        def good_cb(e):
            return good_results.append(e)

        bus.subscribe("test.event", bad_cb)
        bus.subscribe("test.event", good_cb)

        # Should not raise even though bad_cb raises
        bus.publish(_make_event("test.event"))

        # The good callback still ran
        assert len(good_results) == 1

    def test_raising_callback_is_logged(self, caplog):
        """Exception in callback is logged at ERROR level."""
        bus = EventBus()
        bad_cb = MagicMock(side_effect=ValueError("callback error"))
        bus.subscribe("test.event", bad_cb)

        with caplog.at_level(logging.ERROR, logger="missy.core.events"):
            bus.publish(_make_event("test.event"))

        assert any("Unhandled exception" in r.message for r in caplog.records)

    def test_event_still_appended_to_log_when_callback_raises(self):
        """Event is stored in the internal log even when a callback raises."""
        bus = EventBus()
        bus.subscribe("test.event", MagicMock(side_effect=RuntimeError("bad")))

        event = _make_event("test.event")
        bus.publish(event)

        stored = bus.get_events(event_type="test.event")
        assert event in stored
