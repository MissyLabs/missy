"""Run-18 rollback-safe and bounded operator-session regressions."""

from __future__ import annotations

from unittest.mock import patch

from missy.api.audit_browser import query_audit_events
from missy.api.web_sessions import WebSessionStore
from missy.core.events import AuditEvent, EventBus


def test_coreval_032_ttl_uses_monotonic_time_across_wall_clock_rollback() -> None:
    with (
        patch("missy.api.web_sessions.time.time", return_value=1_000.0),
        patch("missy.api.web_sessions.time.monotonic", return_value=100.0),
    ):
        store = WebSessionStore(ttl_seconds=60)
        session = store.create()

    with (
        patch("missy.api.web_sessions.time.time", return_value=10.0),
        patch("missy.api.web_sessions.time.monotonic", return_value=161.0),
    ):
        assert store.get(session.token) is None


def test_coreval_032_population_is_bounded_and_oldest_session_is_evicted() -> None:
    store = WebSessionStore(ttl_seconds=3600, max_sessions=2)
    first = store.create()
    second = store.create()
    third = store.create()
    assert len(store._sessions) == 2
    assert store.get(first.token) is None
    assert store.get(second.token) is not None
    assert store.get(third.token) is not None


def test_coreval_032_session_repr_never_contains_tokens() -> None:
    session = WebSessionStore(ttl_seconds=60).create()
    representation = repr(session)
    assert session.token not in representation
    assert session.csrf_token not in representation


def test_current_process_audit_event_is_not_hidden_by_stale_file_view() -> None:
    bus = EventBus()
    bus.publish(
        AuditEvent.now(
            session_id="web",
            task_id="-",
            event_type="web.login",
            category="channel",
            result="deny",
            detail={"source": "web_tui", "subsystem": "auth"},
        )
    )
    with patch("missy.observability.audit_logger.get_audit_logger") as get_logger:
        get_logger.return_value.get_recent_events.return_value = []
        result = query_audit_events({"source": "web_tui", "subsystem": "auth"}, bus=bus)
    assert result["source"] == "file+memory"
    assert [event["event_type"] for event in result["events"]] == ["web.login"]
