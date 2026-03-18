"""Thread-safety tests for MemoryStore (JSON backend).

These tests exercise the threading.Lock introduced in MemoryStore to guard
_turns mutations and _save calls.  They are intentionally distinct from the
existing concurrent tests in test_memory_concurrent.py, which focus on
absence-of-crash and data-completeness.  Here we additionally verify:

- Exact turn counts after high-contention writes (10 threads × 10 turns)
- No crash when clear_session races with add_turn
- No crash or data loss when compact_session races with writes
- _save is never called outside the lock (mock-based lock introspection)
- A fresh MemoryStore reloaded from disk after concurrent writes sees all data
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from missy.memory.store import ConversationTurn, MemoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(store_path=str(tmp_path / "memory.json"))


@pytest.fixture
def store_path(tmp_path: Path) -> str:
    return str(tmp_path / "memory.json")


# ---------------------------------------------------------------------------
# 6. Concurrent add_turn — 10 threads × 10 turns = 100 total
# ---------------------------------------------------------------------------


class TestConcurrentAddTurn:
    def test_all_100_turns_present_after_concurrent_writes(self, store_path: str):
        """10 threads each writing 10 turns must yield exactly 100 in the store."""
        store = MemoryStore(store_path=store_path)
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            for i in range(10):
                try:
                    store.add_turn(
                        session_id=f"t{thread_id}",
                        role="user",
                        content=f"thread={thread_id} msg={i}",
                    )
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Unexpected errors: {errors}"
        assert len(store._turns) == 100

    def test_each_session_has_correct_turn_count(self, store_path: str):
        """Each of 10 sessions must contain exactly 10 turns after concurrent writes."""
        store = MemoryStore(store_path=store_path)

        def worker(tid: int) -> None:
            for i in range(10):
                store.add_turn(
                    session_id=f"sess-{tid}",
                    role="user",
                    content=f"msg {i}",
                )

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        for tid in range(10):
            turns = store.get_session_turns(f"sess-{tid}")
            assert len(turns) == 10, f"Expected 10 turns for sess-{tid}, got {len(turns)}"

    def test_no_duplicate_turn_ids_after_concurrent_writes(self, store_path: str):
        """Every turn must have a unique ID even under concurrent creation."""
        store = MemoryStore(store_path=store_path)

        def worker(tid: int) -> None:
            for i in range(10):
                store.add_turn(session_id="shared", role="user", content=f"{tid}-{i}")

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        ids = [t.id for t in store._turns]
        assert len(ids) == len(set(ids)), "Duplicate turn IDs detected"

    def test_concurrent_writes_to_single_session_no_missing_turns(self, store_path: str):
        """All 50 writes targeting the same session must be present."""
        store = MemoryStore(store_path=store_path)

        def worker(tid: int) -> None:
            for i in range(10):
                store.add_turn(session_id="hotspot", role="user", content=f"{tid}-{i}")

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        turns = store.get_session_turns("hotspot", limit=100)
        assert len(turns) == 50


# ---------------------------------------------------------------------------
# 7. Concurrent add_turn + clear_session — no crash
# ---------------------------------------------------------------------------


class TestConcurrentAddAndClear:
    def test_no_crash_when_clear_races_with_add(self, store_path: str):
        """clear_session during concurrent adds must not raise."""
        store = MemoryStore(store_path=store_path)
        # Pre-seed so clear has something to do
        for i in range(20):
            store.add_turn(session_id="victim", role="user", content=f"seed {i}")

        errors: list[Exception] = []

        def adder() -> None:
            for i in range(30):
                try:
                    store.add_turn(session_id="victim", role="user", content=f"new {i}")
                except Exception as exc:
                    errors.append(exc)

        def clearer() -> None:
            # Stagger slightly to create race overlap
            time.sleep(0.002)
            try:
                store.clear_session("victim")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=adder),
            threading.Thread(target=adder),
            threading.Thread(target=clearer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors during clear+add race: {errors}"

    def test_non_victim_session_intact_after_clear_race(self, store_path: str):
        """Turns for a different session must survive a clear_session race."""
        store = MemoryStore(store_path=store_path)
        errors: list[Exception] = []

        def writer() -> None:
            for i in range(20):
                try:
                    store.add_turn(session_id="safe", role="user", content=f"msg {i}")
                except Exception as exc:
                    errors.append(exc)

        def clearer() -> None:
            for _ in range(5):
                try:
                    store.clear_session("victim")
                except Exception as exc:
                    errors.append(exc)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=clearer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors
        safe_turns = store.get_session_turns("safe")
        assert len(safe_turns) == 20

    def test_multiple_concurrent_clears_do_not_raise(self, store_path: str):
        """Simultaneous clear_session calls for different sessions must not crash."""
        store = MemoryStore(store_path=store_path)
        for sid in ("a", "b", "c"):
            for i in range(10):
                store.add_turn(session_id=sid, role="user", content=f"msg {i}")

        errors: list[Exception] = []

        def clear(sid: str) -> None:
            try:
                store.clear_session(sid)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=clear, args=(sid,)) for sid in ("a", "b", "c")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors


# ---------------------------------------------------------------------------
# 8. Concurrent compact_session — no crash
# ---------------------------------------------------------------------------


class TestConcurrentCompact:
    def test_compact_during_writes_no_crash(self, store_path: str):
        """compact_session running while writers add turns must not raise."""
        store = MemoryStore(store_path=store_path)
        for i in range(30):
            store.add_turn(session_id="s1", role="user", content=f"turn {i}")

        errors: list[Exception] = []

        def writer() -> None:
            for i in range(20):
                try:
                    store.add_turn(session_id="s1", role="user", content=f"new {i}")
                except Exception as exc:
                    errors.append(exc)

        def compactor() -> None:
            try:
                store.compact_session("s1", keep_recent=5)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=compactor),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors during compact race: {errors}"

    def test_compact_result_is_non_negative(self, store_path: str):
        """compact_session must return a non-negative removed count even under contention."""
        store = MemoryStore(store_path=store_path)
        for i in range(20):
            store.add_turn(session_id="s1", role="user", content=f"msg {i}")

        removed_counts: list[int] = []
        errors: list[Exception] = []

        def compactor() -> None:
            try:
                n = store.compact_session("s1", keep_recent=3)
                removed_counts.append(n)
            except Exception as exc:
                errors.append(exc)

        def writer() -> None:
            for i in range(10):
                try:
                    store.add_turn(session_id="s1", role="user", content=f"w{i}")
                except Exception as exc:
                    errors.append(exc)

        threads = [
            threading.Thread(target=compactor),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors
        for count in removed_counts:
            assert count >= 0

    def test_compact_does_not_affect_other_sessions(self, store_path: str):
        """compact_session("s1") must not disturb turns for session "s2"."""
        store = MemoryStore(store_path=store_path)
        for i in range(20):
            store.add_turn(session_id="s1", role="user", content=f"s1-{i}")
        for i in range(5):
            store.add_turn(session_id="s2", role="user", content=f"s2-{i}")

        errors: list[Exception] = []

        def compactor() -> None:
            try:
                store.compact_session("s1", keep_recent=2)
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(30):
                    store.get_session_turns("s2")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=compactor), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors
        # s2 turns must be untouched
        assert len(store.get_session_turns("s2")) == 5


# ---------------------------------------------------------------------------
# 9. Lock prevents corruption — _save called under the lock
# ---------------------------------------------------------------------------


class TestLockPreventsCorruption:
    def test_save_called_under_lock(self, store_path: str):
        """Verify that _save is invoked while _lock is held by inspecting lock state."""
        store = MemoryStore(store_path=store_path)
        lock_held_during_save: list[bool] = []

        original_save = store._save

        def instrumented_save() -> None:
            # If the lock is already acquired by this thread, locked() is True
            lock_held_during_save.append(store._lock.locked())
            original_save()

        store._save = instrumented_save  # type: ignore[method-assign]
        store.add_turn(session_id="s1", role="user", content="test")

        assert lock_held_during_save, "_save was never called"
        assert all(lock_held_during_save), "_save was called outside the lock"

    def test_clear_session_save_called_under_lock(self, store_path: str):
        """clear_session must also call _save under the lock."""
        store = MemoryStore(store_path=store_path)
        store.add_turn(session_id="s1", role="user", content="to be cleared")

        lock_held_during_save: list[bool] = []
        original_save = store._save

        def instrumented_save() -> None:
            lock_held_during_save.append(store._lock.locked())
            original_save()

        store._save = instrumented_save  # type: ignore[method-assign]
        store.clear_session("s1")

        assert lock_held_during_save, "_save was never called by clear_session"
        assert all(lock_held_during_save)

    def test_compact_session_save_called_under_lock(self, store_path: str):
        """compact_session must call _save under the lock."""
        store = MemoryStore(store_path=store_path)
        for i in range(15):
            store.add_turn(session_id="s1", role="user", content=f"msg {i}")

        lock_held_during_save: list[bool] = []
        original_save = store._save

        def instrumented_save() -> None:
            lock_held_during_save.append(store._lock.locked())
            original_save()

        store._save = instrumented_save  # type: ignore[method-assign]
        store.compact_session("s1", keep_recent=3)

        assert lock_held_during_save, "_save was never called by compact_session"
        assert all(lock_held_during_save)

    def test_add_turn_lock_released_after_exception_in_save(self, store_path: str):
        """If _save raises, the context-manager releases the lock; subsequent calls work."""
        store = MemoryStore(store_path=store_path)
        call_count = {"n": 0}

        def failing_save() -> None:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("disk full")

        store._save = failing_save  # type: ignore[method-assign]

        with pytest.raises(OSError, match="disk full"):
            store.add_turn(session_id="s1", role="user", content="first")

        # Lock must be released — second add_turn must not deadlock
        done = threading.Event()

        def second_add() -> None:
            store.add_turn(session_id="s1", role="user", content="second")
            done.set()

        t = threading.Thread(target=second_add)
        t.start()
        t.join(timeout=2)
        assert done.is_set(), "Lock was not released after _save raised"


# ---------------------------------------------------------------------------
# 10. Thread-safe persistence — reload shows all data
# ---------------------------------------------------------------------------


class TestThreadSafePersistence:
    def test_reload_after_concurrent_writes_shows_all_turns(self, store_path: str):
        """A fresh MemoryStore loaded from disk must contain every written turn."""
        store = MemoryStore(store_path=store_path)

        def worker(tid: int) -> None:
            for i in range(10):
                store.add_turn(
                    session_id=f"batch-{tid}",
                    role="user",
                    content=f"item {i}",
                )

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        reloaded = MemoryStore(store_path=store_path)
        assert len(reloaded._turns) == 100

    def test_persisted_json_is_valid_after_concurrent_writes(self, store_path: str):
        """The JSON file written under contention must be parseable."""
        store = MemoryStore(store_path=store_path)

        def worker(tid: int) -> None:
            for i in range(5):
                store.add_turn(session_id=f"s{tid}", role="user", content=f"msg {i}")

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        raw = Path(store_path).read_text(encoding="utf-8")
        records: Any = json.loads(raw)  # must not raise
        assert isinstance(records, list)
        assert len(records) == 40

    def test_reloaded_store_query_returns_correct_session_turns(self, store_path: str):
        """Per-session queries on a reloaded store return complete session history."""
        store = MemoryStore(store_path=store_path)

        def worker(tid: int) -> None:
            for i in range(8):
                store.add_turn(
                    session_id=f"team-{tid}",
                    role="assistant" if i % 2 else "user",
                    content=f"turn {i}",
                )

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        fresh = MemoryStore(store_path=store_path)
        for tid in range(5):
            turns = fresh.get_session_turns(f"team-{tid}", limit=100)
            assert len(turns) == 8, (
                f"Expected 8 turns for team-{tid} after reload, got {len(turns)}"
            )

    def test_disk_file_content_matches_in_memory_state(self, store_path: str):
        """On-disk JSON must reflect the same turns as the live _turns list."""
        store = MemoryStore(store_path=store_path)

        def worker(tid: int) -> None:
            for i in range(5):
                store.add_turn(session_id="sync-check", role="user", content=f"{tid}-{i}")

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        in_memory_ids = {t.id for t in store._turns}
        disk_records = json.loads(Path(store_path).read_text(encoding="utf-8"))
        disk_ids = {r["id"] for r in disk_records}

        assert in_memory_ids == disk_ids

    def test_concurrent_clear_and_add_persistence_consistent(self, store_path: str):
        """After a clear+add race, disk and in-memory state match each other."""
        store = MemoryStore(store_path=store_path)
        for i in range(20):
            store.add_turn(session_id="doomed", role="user", content=f"old {i}")

        def adder() -> None:
            for i in range(10):
                store.add_turn(session_id="survivor", role="user", content=f"new {i}")

        def clearer() -> None:
            time.sleep(0.001)
            store.clear_session("doomed")

        threads = [threading.Thread(target=adder), threading.Thread(target=clearer)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        in_memory_ids = {t.id for t in store._turns}
        disk_records = json.loads(Path(store_path).read_text(encoding="utf-8"))
        disk_ids = {r["id"] for r in disk_records}

        assert in_memory_ids == disk_ids, (
            "In-memory and on-disk state diverged after concurrent clear+add"
        )
