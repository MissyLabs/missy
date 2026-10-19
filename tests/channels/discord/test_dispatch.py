"""Per-conversation dispatch: a slow channel must not block other channels.

Live validation: a 10-minute task from another user/channel blocked the
test channel because the gateway drained one shared queue serially.
"""

from __future__ import annotations

import asyncio

import pytest

from missy.channels.discord.config import parse_discord_config
from missy.channels.discord.dispatch import KeyedSerialDispatcher


def _key(item: tuple[str, int]) -> str:
    return item[0]


async def _start(dispatcher, queue, stop):
    return asyncio.create_task(dispatcher.run(queue, lambda: stop.is_set()))


async def _wait_for(predicate, timeout=2.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


def test_slow_channel_does_not_block_other_channel():
    async def main():
        release_a = asyncio.Event()
        done: list[tuple[str, int]] = []

        async def handler(item):
            if item[0] == "A":
                await release_a.wait()
            done.append(item)

        queue: asyncio.Queue = asyncio.Queue(maxsize=16)
        stop = asyncio.Event()
        dispatcher = KeyedSerialDispatcher(handler, _key, max_concurrency=3)
        task = await _start(dispatcher, queue, stop)
        await queue.put(("A", 1))
        await queue.put(("B", 1))
        await _wait_for(lambda: ("B", 1) in done)
        assert ("A", 1) not in done  # still running
        release_a.set()
        await _wait_for(lambda: ("A", 1) in done)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())


def test_order_and_serialization_within_channel():
    async def main():
        running: dict[str, int] = {"A": 0}
        max_running = {"A": 0}
        done: list[tuple[str, int]] = []

        async def handler(item):
            running[item[0]] += 1
            max_running[item[0]] = max(max_running[item[0]], running[item[0]])
            # Later items finish faster: ordering must still hold.
            await asyncio.sleep(0.02 / item[1])
            running[item[0]] -= 1
            done.append(item)

        queue: asyncio.Queue = asyncio.Queue()
        stop = asyncio.Event()
        dispatcher = KeyedSerialDispatcher(handler, _key, max_concurrency=3)
        task = await _start(dispatcher, queue, stop)
        for i in range(1, 6):
            await queue.put(("A", i))
        await _wait_for(lambda: len(done) == 5)
        assert done == [("A", i) for i in range(1, 6)]
        assert max_running["A"] == 1
        stop.set()
        await asyncio.wait_for(task, 3)

    asyncio.run(main())


def test_global_concurrency_is_bounded():
    async def main():
        current = 0
        peak = 0
        gate = asyncio.Event()

        async def handler(item):
            nonlocal current, peak
            current += 1
            peak = max(peak, current)
            await gate.wait()
            current -= 1

        queue: asyncio.Queue = asyncio.Queue()
        stop = asyncio.Event()
        dispatcher = KeyedSerialDispatcher(handler, _key, max_concurrency=2)
        task = await _start(dispatcher, queue, stop)
        for ch in "ABCDE":
            await queue.put((ch, 1))
        await _wait_for(lambda: current == 2)
        await asyncio.sleep(0.05)
        assert peak == 2
        gate.set()
        await _wait_for(lambda: queue.empty() and not dispatcher.active_keys)
        assert peak == 2
        stop.set()
        await asyncio.wait_for(task, 3)

    asyncio.run(main())


def test_backpressure_leaves_items_in_queue():
    async def main():
        gate = asyncio.Event()

        async def handler(item):
            await gate.wait()

        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        stop = asyncio.Event()
        dispatcher = KeyedSerialDispatcher(handler, _key, max_concurrency=1, max_pending=3)
        task = await _start(dispatcher, queue, stop)
        for i in range(8):
            await queue.put(("A", i))
        await asyncio.sleep(0.05)
        # Only max_pending items left the queue; the rest still apply
        # queue_max backpressure to the producer.
        assert queue.qsize() == 5
        gate.set()
        await _wait_for(lambda: queue.empty() and not dispatcher.active_keys)
        stop.set()
        await asyncio.wait_for(task, 3)

    asyncio.run(main())


def test_handler_exception_does_not_stop_channel():
    async def main():
        done: list = []

        async def handler(item):
            if item[1] == 1:
                raise RuntimeError("boom")
            done.append(item)

        queue: asyncio.Queue = asyncio.Queue()
        stop = asyncio.Event()
        task = await _start(KeyedSerialDispatcher(handler, _key), queue, stop)
        await queue.put(("A", 1))
        await queue.put(("A", 2))
        await _wait_for(lambda: done == [("A", 2)])
        stop.set()
        await asyncio.wait_for(task, 3)

    asyncio.run(main())


def test_cancel_cancels_inflight_handlers():
    async def main():
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def handler(item):
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        queue: asyncio.Queue = asyncio.Queue()
        stop = asyncio.Event()
        dispatcher = KeyedSerialDispatcher(handler, _key)
        task = await _start(dispatcher, queue, stop)
        await queue.put(("A", 1))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        assert not dispatcher.active_keys

    asyncio.run(main())


def test_invalid_concurrency_rejected():
    with pytest.raises(ValueError):
        KeyedSerialDispatcher(lambda i: None, _key, max_concurrency=0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 3), (5, 5), ("2", 2), (0, 1), (1000, 16), ("x", 3), (True, 3)],
)
def test_config_max_concurrent_conversations(raw, expected):
    data = {"enabled": True, "accounts": []}
    if raw is not None:
        data["max_concurrent_conversations"] = raw
    assert parse_discord_config(data).max_concurrent_conversations == expected


def test_gateway_conversation_key_is_per_channel():
    from types import SimpleNamespace

    from missy.cli.main import _discord_conversation_key

    a1 = SimpleNamespace(metadata={"discord_channel_id": "A", "discord_author": {"id": "1"}})
    a2 = SimpleNamespace(metadata={"discord_channel_id": "A", "discord_author": {"id": "2"}})
    b = SimpleNamespace(metadata={"discord_channel_id": "B"})
    assert _discord_conversation_key(a1) == _discord_conversation_key(a2) == "A"
    assert _discord_conversation_key(b) == "B"
    assert _discord_conversation_key(SimpleNamespace(metadata=None)) == ""
