"""Concurrent access tests for memory stores.

Tests that SQLiteMemoryStore and MemoryStore handle concurrent reads and writes
without corruption, data loss, or crashes.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
import time
from pathlib import Path

import pytest

from missy.memory.store import ConversationTurn, MemoryStore


# ---------------------------------------------------------------------------
# MemoryStore (JSON) concurrent tests
# ---------------------------------------------------------------------------


class TestMemoryStoreConcurrentWrites:
    """Test MemoryStore under concurrent write pressure."""

    def test_concurrent_add_turns(self, tmp_path):
        """Multiple threads adding turns simultaneously."""
        store_path = str(tmp_path / "memory.json")
        store = MemoryStore(store_path=store_path)
        errors = []

        def add_turns(thread_id: int, count: int):
            try:
                for i in range(count):
                    store.add_turn(
                        session_id=f"session-{thread_id}",
                        role="user",
                        content=f"Turn {i} from thread {thread_id}",
                    )
            except Exception as e:
                errors.append(e)

        threads = []
        for tid in range(5):
            t = threading.Thread(target=add_turns, args=(tid, 10))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors during concurrent writes: {errors}"
        # All 50 turns should be in the store
        assert len(store._turns) == 50

    def test_concurrent_read_write(self, tmp_path):
        """Reads and writes happening simultaneously."""
        store_path = str(tmp_path / "memory.json")
        store = MemoryStore(store_path=store_path)
        # Pre-seed some data
        for i in range(20):
            store.add_turn(session_id="s1", role="user", content=f"Seed {i}")

        errors = []

        def writer():
            try:
                for i in range(20):
                    store.add_turn(session_id="s1", role="user", content=f"Write {i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(20):
                    turns = store.get_session_turns("s1")
                    assert isinstance(turns, list)
                    recent = store.get_recent_turns(5)
                    assert isinstance(recent, list)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors during concurrent read/write: {errors}"

    def test_concurrent_search(self, tmp_path):
        """Search while writing."""
        store_path = str(tmp_path / "memory.json")
        store = MemoryStore(store_path=store_path)
        for i in range(30):
            store.add_turn(session_id="s1", role="user", content=f"Item {i} alpha beta gamma")

        errors = []

        def searcher():
            try:
                for _ in range(20):
                    results = store.search("alpha")
                    assert isinstance(results, list)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(10):
                    store.add_turn(session_id="s1", role="user", content=f"New item {i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=searcher),
            threading.Thread(target=searcher),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors

    def test_concurrent_clear_session(self, tmp_path):
        """Clear session while adding turns."""
        store_path = str(tmp_path / "memory.json")
        store = MemoryStore(store_path=store_path)
        for i in range(20):
            store.add_turn(session_id="victim", role="user", content=f"V {i}")

        errors = []

        def clearer():
            try:
                time.sleep(0.01)  # Let some writes happen first
                store.clear_session("victim")
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(10):
                    store.add_turn(session_id="survivor", role="user", content=f"S {i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=clearer),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors
        # Survivor turns should still exist
        survivor_turns = store.get_session_turns("survivor")
        assert len(survivor_turns) == 10


class TestMemoryStoreCompaction:
    """Test compaction under concurrent access."""

    def test_compact_concurrent_read(self, tmp_path):
        """Compaction while reads are happening."""
        store_path = str(tmp_path / "memory.json")
        store = MemoryStore(store_path=store_path)
        for i in range(30):
            store.add_turn(session_id="s1", role="user", content=f"Turn {i}")

        errors = []

        def compactor():
            try:
                removed = store.compact_session("s1", keep_recent=5)
                assert removed >= 0
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(20):
                    store.get_session_turns("s1")
                    store.get_recent_turns(10)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=compactor),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors


# ---------------------------------------------------------------------------
# SQLiteMemoryStore concurrent tests
# ---------------------------------------------------------------------------


class TestSQLiteMemoryStoreConcurrent:
    """Test SQLiteMemoryStore under concurrent access."""

    def test_concurrent_add_turns(self, tmp_path):
        from missy.memory.sqlite_store import SQLiteMemoryStore
        from missy.memory.sqlite_store import ConversationTurn as SQLiteTurn
        db_path = str(tmp_path / "memory.db")
        store = SQLiteMemoryStore(db_path=db_path)
        errors = []

        def add_turns(thread_id: int, count: int):
            try:
                for i in range(count):
                    turn = SQLiteTurn.new(
                        session_id=f"session-{thread_id}",
                        role="user",
                        content=f"Turn {i} from thread {thread_id}",
                    )
                    store.add_turn(turn)
            except Exception as e:
                errors.append(e)

        threads = []
        for tid in range(5):
            t = threading.Thread(target=add_turns, args=(tid, 10))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors: {errors}"
        # Verify all turns were persisted
        total = 0
        for tid in range(5):
            turns = store.get_session_turns(f"session-{tid}")
            total += len(turns)
        assert total == 50

    def test_concurrent_search_and_write(self, tmp_path):
        from missy.memory.sqlite_store import SQLiteMemoryStore
        from missy.memory.sqlite_store import ConversationTurn as SQLiteTurn
        db_path = str(tmp_path / "memory.db")
        store = SQLiteMemoryStore(db_path=db_path)

        # Pre-seed
        for i in range(50):
            turn = SQLiteTurn.new(
                session_id="s1", role="user", content=f"Info about topic {i}"
            )
            store.add_turn(turn)

        errors = []

        def searcher():
            try:
                for _ in range(20):
                    results = store.search("topic")
                    assert isinstance(results, list)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(20):
                    turn = SQLiteTurn.new(
                        session_id="s2", role="user", content=f"New data {i}"
                    )
                    store.add_turn(turn)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=searcher),
            threading.Thread(target=searcher),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors

    def test_concurrent_learnings(self, tmp_path):
        from missy.memory.sqlite_store import SQLiteMemoryStore
        db_path = str(tmp_path / "memory.db")
        store = SQLiteMemoryStore(db_path=db_path)
        errors = []

        def writer(tid: int):
            try:
                for i in range(10):
                    learning = {
                        "task_type": f"type-{tid}",
                        "outcome": "success",
                        "lesson": f"Lesson {i} from thread {tid}",
                    }
                    store.save_learning(learning)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(20):
                    results = store.get_learnings(limit=10)
                    assert isinstance(results, list)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(1,)),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors


# ---------------------------------------------------------------------------
# ThreadPoolExecutor stress test
# ---------------------------------------------------------------------------


class TestMemoryStoreStress:
    def test_high_volume_writes(self, tmp_path):
        """Write many turns quickly and verify all are stored."""
        store_path = str(tmp_path / "memory.json")
        store = MemoryStore(store_path=store_path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for i in range(100):
                f = pool.submit(
                    store.add_turn,
                    session_id="stress",
                    role="user",
                    content=f"Message {i}",
                )
                futures.append(f)

            for f in concurrent.futures.as_completed(futures, timeout=30):
                f.result()  # Raise any exceptions

        assert len(store._turns) == 100

    def test_persistence_after_concurrent_writes(self, tmp_path):
        """Verify data is persisted to disk after concurrent writes."""
        store_path = str(tmp_path / "memory.json")
        store = MemoryStore(store_path=store_path)

        def add_batch(batch_id: int):
            for i in range(5):
                store.add_turn(
                    session_id=f"batch-{batch_id}",
                    role="user",
                    content=f"Item {i}",
                )

        threads = [threading.Thread(target=add_batch, args=(b,)) for b in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # Reload from disk and verify
        store2 = MemoryStore(store_path=store_path)
        assert len(store2._turns) == 20
