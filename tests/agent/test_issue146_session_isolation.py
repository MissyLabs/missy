"""Regressions for issue #146 session ordering and Discord scoping."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from missy.agent.runtime import AgentRuntime
from missy.channels.discord.session_scope import discord_session_id


def _runtime_with_fake_run(fake_run) -> AgentRuntime:
    runtime = object.__new__(AgentRuntime)
    runtime._session_run_locks = {}
    runtime._session_run_locks_guard = threading.Lock()
    runtime._run_once = fake_run
    return runtime


def test_same_session_runs_are_serialized() -> None:
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    order: list[str] = []

    def fake_run(user_input: str, **_kwargs) -> str:
        order.append(f"start:{user_input}")
        if user_input == "first":
            first_entered.set()
            assert release_first.wait(timeout=2)
        else:
            second_entered.set()
        order.append(f"end:{user_input}")
        return user_input

    runtime = _runtime_with_fake_run(fake_run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(runtime.run, "first", "shared")
        assert first_entered.wait(timeout=2)
        second = pool.submit(runtime.run, "second", "shared")
        assert not second_entered.wait(timeout=0.1)
        release_first.set()
        assert first.result(timeout=2) == "first"
        assert second.result(timeout=2) == "second"

    assert order == ["start:first", "end:first", "start:second", "end:second"]
    assert runtime._session_run_locks == {}


def test_different_sessions_remain_concurrent() -> None:
    barrier = threading.Barrier(2)

    def fake_run(user_input: str, **_kwargs) -> str:
        barrier.wait(timeout=2)
        return user_input

    runtime = _runtime_with_fake_run(fake_run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(runtime.run, "one", "session-one")
        two = pool.submit(runtime.run, "two", "session-two")
        assert {one.result(timeout=2), two.result(timeout=2)} == {"one", "two"}


def test_discord_scope_separates_dm_channel_and_thread() -> None:
    dm = discord_session_id("user-1", "", "dm-1")
    channel = discord_session_id("user-1", "guild-1", "channel-1")
    thread = discord_session_id("user-1", "guild-1", "thread-1")

    assert len({dm, channel, thread}) == 3
