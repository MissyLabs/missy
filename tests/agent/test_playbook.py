"""Tests for missy.agent.playbook — AI Playbook auto-capture."""

from __future__ import annotations

import threading

import pytest

from missy.agent.playbook import Playbook, _compute_pattern_id


class TestRecordNewPattern:
    def test_record_new_pattern(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "playbook.json"))
        entry = pb.record("shell", "deploy app", ["shell_exec", "file_write"], "use rsync")
        assert entry.success_count == 1
        assert entry.task_type == "shell"
        assert entry.tool_sequence == ["shell_exec", "file_write"]
        assert entry.description == "deploy app"
        assert entry.prompt_template == "use rsync"
        assert entry.promoted is False
        assert entry.created_at  # auto-populated


class TestRecordExistingIncrements:
    def test_record_existing_increments(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "playbook.json"))
        e1 = pb.record("shell", "deploy v1", ["shell_exec", "file_write"], "hint1")
        e2 = pb.record("shell", "deploy v2", ["shell_exec", "file_write"], "hint2")
        assert e2.success_count == 2
        assert e1.pattern_id == e2.pattern_id
        # Description and hint updated to latest
        assert e2.description == "deploy v2"
        assert e2.prompt_template == "hint2"

    def test_different_tool_order_same_pattern(self, tmp_path):
        """Sorted tool sequence hash means order doesn't matter."""
        pb = Playbook(store_path=str(tmp_path / "playbook.json"))
        pb.record("shell", "desc1", ["file_write", "shell_exec"], "hint1")
        e2 = pb.record("shell", "desc2", ["shell_exec", "file_write"], "hint2")
        assert e2.success_count == 2


class TestGetRelevantByTaskType:
    def test_get_relevant_by_task_type(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "playbook.json"))
        pb.record("shell", "shell task", ["shell_exec"], "hint")
        pb.record("file", "file task", ["file_read"], "hint")
        pb.record("shell", "another shell", ["shell_exec", "file_write"], "hint")

        results = pb.get_relevant("shell")
        assert len(results) == 2
        assert all(e.task_type == "shell" for e in results)

    def test_get_relevant_ordered_by_count(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "playbook.json"))
        pb.record("shell", "low", ["shell_exec"], "hint")
        pb.record("shell", "high", ["shell_exec", "file_write"], "hint")
        pb.record("shell", "high", ["shell_exec", "file_write"], "hint")
        pb.record("shell", "high", ["shell_exec", "file_write"], "hint")

        results = pb.get_relevant("shell", top_k=1)
        assert len(results) == 1
        assert results[0].success_count == 3

    def test_get_relevant_empty(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "playbook.json"))
        assert pb.get_relevant("nonexistent") == []


class TestGetPromotableThreshold:
    def test_get_promotable_threshold(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "playbook.json"))
        # Record pattern 3 times
        for _ in range(3):
            pb.record("shell", "deploy", ["shell_exec"], "hint")
        # Record another only once
        pb.record("file", "read", ["file_read"], "hint")

        promotable = pb.get_promotable(threshold=3)
        assert len(promotable) == 1
        assert promotable[0].task_type == "shell"
        assert promotable[0].success_count == 3

    def test_get_promotable_excludes_promoted(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "playbook.json"))
        for _ in range(3):
            pb.record("shell", "deploy", ["shell_exec"], "hint")
        pid = pb.get_promotable()[0].pattern_id
        pb.mark_promoted(pid)
        assert pb.get_promotable(threshold=3) == []

    def test_get_promotable_below_threshold(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "playbook.json"))
        pb.record("shell", "deploy", ["shell_exec"], "hint")
        pb.record("shell", "deploy", ["shell_exec"], "hint")
        assert pb.get_promotable(threshold=3) == []


class TestMarkPromoted:
    def test_mark_promoted(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "playbook.json"))
        entry = pb.record("shell", "deploy", ["shell_exec"], "hint")
        assert entry.promoted is False
        pb.mark_promoted(entry.pattern_id)
        # Re-fetch
        relevant = pb.get_relevant("shell")
        assert relevant[0].promoted is True

    def test_mark_promoted_unknown_raises(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "playbook.json"))
        with pytest.raises(KeyError):
            pb.mark_promoted("nonexistent")


class TestSaveAndLoad:
    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "playbook.json")
        pb1 = Playbook(store_path=path)
        pb1.record("shell", "task1", ["shell_exec"], "hint1")
        pb1.record("file", "task2", ["file_read", "file_write"], "hint2")

        # Load in new instance
        pb2 = Playbook(store_path=path)
        assert len(pb2.get_relevant("shell")) == 1
        assert len(pb2.get_relevant("file")) == 1
        assert pb2.get_relevant("shell")[0].description == "task1"

    def test_load_nonexistent_file(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "does_not_exist.json"))
        assert pb.get_relevant("anything") == []


class TestThreadSafe:
    def test_thread_safe(self, tmp_path):
        pb = Playbook(store_path=str(tmp_path / "playbook.json"))
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                pb.record("shell", f"task{i}", ["shell_exec"], f"hint{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # All 20 records should have incremented the same pattern
        results = pb.get_relevant("shell")
        assert len(results) == 1
        assert results[0].success_count == 20


class TestComputePatternId:
    def test_deterministic(self):
        id1 = _compute_pattern_id("shell", ["a", "b"])
        id2 = _compute_pattern_id("shell", ["b", "a"])
        assert id1 == id2  # sorted

    def test_different_types(self):
        id1 = _compute_pattern_id("shell", ["a"])
        id2 = _compute_pattern_id("file", ["a"])
        assert id1 != id2
