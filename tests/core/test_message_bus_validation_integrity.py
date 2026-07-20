"""Run-18 message-bus contract and backpressure regressions."""

from __future__ import annotations

import queue
import time

import pytest

from missy.core.bus_topics import ALL_TOPICS
from missy.core.message_bus import BusMessage, MessageBus


def _payload(topic: str) -> dict:
    return {
        "agent.run.start": {
            "session_id": "session",
            "task_id": "task",
            "user_input_length": 1,
        },
        "agent.run.complete": {
            "session_id": "session",
            "task_id": "task",
            "provider": "provider",
            "tools_used": [],
            "cost": {},
        },
        "agent.run.error": {
            "session_id": "session",
            "task_id": "task",
            "error": "safe error",
        },
        "tool.request": {"tool": "calculator", "session_id": "session", "task_id": "task"},
        "tool.result": {
            "tool": "calculator",
            "is_error": False,
            "session_id": "session",
            "task_id": "task",
        },
        "sleeptime.error": {"detail": "safe error"},
    }.get(topic, {})


def test_coreval_027_all_production_topics_have_validated_contracts() -> None:
    bus = MessageBus(strict_topics=True)
    received = []
    bus.subscribe("*", received.append)
    for topic in sorted(ALL_TOPICS):
        bus.publish(BusMessage(topic=topic, payload=_payload(topic), source="validation"))
    assert [message.topic for message in received] == sorted(ALL_TOPICS)

    invalid = [
        BusMessage(topic="unknown.topic", payload={}, source="validation"),
        BusMessage(topic="tool.*", payload={}, source="validation"),
        BusMessage(topic="tool.request", payload={}, source="validation"),
        BusMessage(topic="tool.request", payload=_payload("tool.request"), source="bad source"),
        BusMessage(
            topic="tool.request",
            payload=_payload("tool.request"),
            source="validation",
            priority=3,
        ),
    ]
    for message in invalid:
        with pytest.raises((TypeError, ValueError)):
            bus.publish(message)
    assert len(received) == len(ALL_TOPICS)


def test_coreval_027_payload_size_and_json_are_bounded() -> None:
    bus = MessageBus(strict_topics=True)
    with pytest.raises(ValueError, match="256 KiB"):
        bus.publish(
            BusMessage(
                topic="system.startup",
                payload={"data": "x" * (257 * 1024)},
                source="validation",
            )
        )
    with pytest.raises(ValueError, match="finite JSON"):
        bus.publish(
            BusMessage(
                topic="system.startup",
                payload={"value": float("nan")},
                source="validation",
            )
        )


def test_coreval_028_queue_full_is_immediate_observable_and_shutdown_bounded() -> None:
    bus = MessageBus(max_queue_size=1)
    first = BusMessage(topic="test.one", payload={}, source="validation")
    bus.publish_async(first)
    started = time.monotonic()
    with pytest.raises(queue.Full):
        bus.publish_async(BusMessage(topic="test.two", payload={}, source="validation"))
    assert time.monotonic() - started < 0.1
    assert bus.stats()["dropped"] == 1
    assert bus.stats()["accepted"] == 1

    received = []
    bus.subscribe("test.*", received.append)
    bus.stop(timeout=0.1)
    assert received == [first]
    assert bus.pending_count() == 0
    assert not bus.stats()["accepting"]
    with pytest.raises(RuntimeError, match="no longer accepts"):
        bus.publish_async(BusMessage(topic="test.three", payload={}, source="validation"))


def test_coreval_028_priority_burst_cannot_starve_normal_work() -> None:
    bus = MessageBus(max_queue_size=32)
    order = []
    bus.subscribe("work.*", lambda message: order.append(message.payload["id"]))
    bus.publish_async(
        BusMessage(topic="work.normal", payload={"id": "normal"}, source="validation")
    )
    for index in range(16):
        bus.publish_async(
            BusMessage(
                topic="work.urgent",
                payload={"id": f"urgent-{index}"},
                source="validation",
                priority=2,
            )
        )
    bus.drain()
    assert order.index("normal") <= 8
    assert len(order) == 17
