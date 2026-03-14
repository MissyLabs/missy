"""Tests for missy.memory.store.MemoryStore (JSON-based)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from missy.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    path = str(tmp_path / "memory.json")
    return MemoryStore(store_path=path)


class TestMemoryStoreAddAndGet:
    def test_add_and_get_session_turns(self, store):
        store.add_turn("s1", "user", "first")
        store.add_turn("s1", "assistant", "second")
        turns = store.get_session_turns("s1")
        assert len(turns) == 2

    def test_get_session_turns_limit(self, store):
        for i in range(10):
            store.add_turn("s1", "user", f"msg {i}")
        turns = store.get_session_turns("s1", limit=3)
        assert len(turns) == 3

    def test_get_recent_turns(self, store):
        store.add_turn("s1", "user", "a")
        store.add_turn("s2", "user", "b")
        recent = store.get_recent_turns(limit=10)
        assert len(recent) == 2


class TestMemoryStoreClearSession:
    def test_clear_removes_session(self, store):
        store.add_turn("s1", "user", "x")
        store.add_turn("s2", "user", "y")
        store.clear_session("s1")
        assert len(store.get_session_turns("s1")) == 0
        assert len(store.get_session_turns("s2")) == 1


class TestMemoryStoreSearch:
    def test_search_finds_matching_content(self, store):
        store.add_turn("s1", "user", "hello world")
        store.add_turn("s1", "user", "goodbye")
        results = store.search("hello")
        assert len(results) == 1
        assert results[0].content == "hello world"

    def test_search_case_insensitive(self, store):
        store.add_turn("s1", "user", "Hello World")
        results = store.search("hello")
        assert len(results) == 1

    def test_search_with_session_filter(self, store):
        store.add_turn("s1", "user", "match")
        store.add_turn("s2", "user", "match")
        results = store.search("match", session_id="s1")
        assert len(results) == 1

    def test_search_respects_limit(self, store):
        for i in range(20):
            store.add_turn("s1", "user", f"match {i}")
        results = store.search("match", limit=5)
        assert len(results) == 5


class TestMemoryStoreCompact:
    def test_compact_removes_old_turns(self, store):
        for i in range(10):
            store.add_turn("s1", "user", f"msg {i}")
        removed = store.compact_session("s1", keep_recent=3)
        assert removed == 7
        turns = store.get_session_turns("s1")
        assert len(turns) == 4  # 3 kept + 1 summary

    def test_compact_noop_when_few_turns(self, store):
        store.add_turn("s1", "user", "only one")
        removed = store.compact_session("s1", keep_recent=5)
        assert removed == 0


class TestMemoryStoreLearnings:
    def test_save_learning_noop(self, store):
        store.save_learning({"lesson": "test"})  # should not raise

    def test_get_learnings_empty(self, store):
        assert store.get_learnings() == []


class TestMemoryStorePersistence:
    def test_save_and_reload(self, tmp_path):
        path = str(tmp_path / "memory.json")
        s1 = MemoryStore(store_path=path)
        s1.add_turn("s1", "user", "persisted")

        s2 = MemoryStore(store_path=path)
        turns = s2.get_session_turns("s1")
        assert len(turns) == 1
        assert turns[0].content == "persisted"

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "memory.json"
        path.write_text("NOT JSON", encoding="utf-8")
        store = MemoryStore(store_path=str(path))
        assert store.get_recent_turns() == []

    def test_load_non_array(self, tmp_path):
        path = tmp_path / "memory.json"
        path.write_text('{"not": "array"}', encoding="utf-8")
        store = MemoryStore(store_path=str(path))
        assert store.get_recent_turns() == []

    def test_load_malformed_records(self, tmp_path):
        path = tmp_path / "memory.json"
        path.write_text(json.dumps(["bad", 123, None]), encoding="utf-8")
        store = MemoryStore(store_path=str(path))
        assert store.get_recent_turns() == []
