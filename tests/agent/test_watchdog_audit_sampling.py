"""Healthy watchdog checks are audited on change or hourly (premortem item 12)."""

from __future__ import annotations

from unittest.mock import patch

from missy.agent import watchdog as watchdog_module
from missy.agent.watchdog import Watchdog


def _events_for(checks: list[bool], clock: list[float]) -> list[dict]:
    published: list = []
    results = iter(checks)
    wd = Watchdog()
    wd.register("store", lambda: next(results))
    with (
        patch("missy.core.events.event_bus.publish", side_effect=published.append),
        patch.object(watchdog_module.time, "monotonic", side_effect=lambda: clock[0]),
    ):
        for _ in checks:
            wd._check_all()
            clock[0] += 60.0
    return [event.detail for event in published]


def test_steady_healthy_checks_are_audited_hourly() -> None:
    events = _events_for([True] * 120, [0.0])  # two hours of 60 s checks
    assert len(events) == 2


def test_failures_and_recovery_are_always_audited() -> None:
    events = _events_for([True, False, False, True, True], [0.0])
    assert [e["healthy"] for e in events] == [True, False, False, True]
