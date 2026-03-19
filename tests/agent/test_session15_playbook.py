"""Session 15 comprehensive tests for missy.agent.playbook.

Covers angles not yet exercised by test_playbook.py and test_playbook_done_edges.py:

- PlaybookEntry dataclass defaults: created_at auto-set, success_count=1, promoted=False
- PlaybookEntry explicit created_at not overwritten by __post_init__
- PlaybookEntry: all fields preserved after round-trip through asdict/from-dict
- _compute_pattern_id: deterministic for identical inputs
- _compute_pattern_id: sorts tool list before hashing (order-independent)
- _compute_pattern_id: different task_type -> different ID
- _compute_pattern_id: different tools -> different ID
- _compute_pattern_id: empty tool list is valid
- _compute_pattern_id: returns exactly 16 hex characters
- _compute_pattern_id: single-tool list
- _compute_pattern_id: large tool list
- _compute_pattern_id: Unicode task_type and tool names
- Playbook constructor: no file -> empty, no crash
- Playbook constructor: non-default path respected
- Playbook constructor: creates no file at construction time when none exists
- record: new entry has pattern_id matching _compute_pattern_id directly
- record: new entry has correct task_type, description, tool_sequence, prompt_template
- record: success_count starts at 1
- record: promoted defaults to False
- record: created_at is populated automatically
- record: returns the PlaybookEntry object
- record: duplicate increments success_count by 1
- record: duplicate updates description to latest value
- record: duplicate updates prompt_template to latest value
- record: duplicate does NOT reset created_at
- record: duplicate does NOT change promoted flag
- record: tool order normalised (different order -> same pattern)
- record: three different patterns -> three entries
- record: multiple increments accumulate correctly
- record: saves to disk after each call
- get_relevant: returns only entries matching task_type
- get_relevant: returns all matches when count < top_k
- get_relevant: respects top_k upper bound
- get_relevant: orders by success_count descending
- get_relevant: ties are stable (both entries returned)
- get_relevant: empty list for unknown task_type
- get_relevant: does not modify internal state
- get_promotable: returns entries >= threshold
- get_promotable: excludes entries below threshold
- get_promotable: excludes already-promoted entries
- get_promotable: custom threshold=1 includes all entries
- get_promotable: default threshold=3 verified
- get_promotable: returns empty list when all promoted
- get_promotable: does not mutate internal state
- mark_promoted: sets promoted=True on the live object
- mark_promoted: persists promoted flag after save/load round-trip
- mark_promoted: raises KeyError for unknown pattern_id
- mark_promoted: KeyError message includes pattern_id
- save/load round-trip: all fields preserved
- save/load round-trip: multiple entries preserved
- load: no-op when file does not exist
- load: handles corrupt JSON gracefully (no crash, entries reset)
- load: handles empty JSON array
- load: handles file that is empty string
- save: creates parent directory if missing
- save: atomic write (replaces file)
- Persistence across instances: second Playbook reads first's records
- Thread safety: concurrent record calls to same pattern
- Thread safety: concurrent record calls to distinct patterns
- Thread safety: concurrent get_relevant calls don't crash
- Thread safety: concurrent mark_promoted + get_promotable don't crash
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict
from datetime import datetime

import pytest

from missy.agent.playbook import Playbook, PlaybookEntry, _compute_pattern_id

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pb(tmp_path, name: str = "playbook.json") -> Playbook:
    """Return a Playbook backed by a temp file."""
    return Playbook(store_path=str(tmp_path / name))


def _record_n(pb: Playbook, n: int, task_type: str = "shell", tools: list[str] | None = None) -> PlaybookEntry:
    """Record the same pattern n times; return final entry."""
    tools = tools or ["shell_exec"]
    entry = None
    for i in range(n):
        entry = pb.record(task_type, f"desc{i}", tools, f"hint{i}")
    return entry


# ---------------------------------------------------------------------------
# PlaybookEntry dataclass
# ---------------------------------------------------------------------------


class TestPlaybookEntryDefaults:
    def test_success_count_defaults_to_one(self, tmp_path):
        entry = PlaybookEntry(
            pattern_id="abc",
            task_type="shell",
            description="test",
            tool_sequence=["shell_exec"],
            prompt_template="hint",
        )
        assert entry.success_count == 1

    def test_promoted_defaults_to_false(self, tmp_path):
        entry = PlaybookEntry(
            pattern_id="abc",
            task_type="shell",
            description="test",
            tool_sequence=["shell_exec"],
            prompt_template="hint",
        )
        assert entry.promoted is False

    def test_created_at_auto_populated(self):
        entry = PlaybookEntry(
            pattern_id="abc",
            task_type="shell",
            description="test",
            tool_sequence=["shell_exec"],
            prompt_template="hint",
        )
        assert entry.created_at != ""

    def test_created_at_is_iso8601(self):
        entry = PlaybookEntry(
            pattern_id="abc",
            task_type="shell",
            description="test",
            tool_sequence=["shell_exec"],
            prompt_template="hint",
        )
        # Must parse without raising
        parsed = datetime.fromisoformat(entry.created_at)
        assert parsed is not None

    def test_explicit_created_at_not_overwritten(self):
        ts = "2025-01-01T00:00:00+00:00"
        entry = PlaybookEntry(
            pattern_id="abc",
            task_type="shell",
            description="test",
            tool_sequence=["shell_exec"],
            prompt_template="hint",
            created_at=ts,
        )
        assert entry.created_at == ts

    def test_all_fields_preserved_in_asdict(self):
        entry = PlaybookEntry(
            pattern_id="deadbeef12345678",
            task_type="file",
            description="read config",
            tool_sequence=["file_read"],
            prompt_template="read /etc/config",
            success_count=5,
            promoted=True,
        )
        d = asdict(entry)
        assert d["pattern_id"] == "deadbeef12345678"
        assert d["task_type"] == "file"
        assert d["description"] == "read config"
        assert d["tool_sequence"] == ["file_read"]
        assert d["prompt_template"] == "read /etc/config"
        assert d["success_count"] == 5
        assert d["promoted"] is True


# ---------------------------------------------------------------------------
# _compute_pattern_id
# ---------------------------------------------------------------------------


class TestComputePatternId:
    def test_returns_16_character_string(self):
        result = _compute_pattern_id("shell", ["shell_exec"])
        assert isinstance(result, str)
        assert len(result) == 16

    def test_result_is_hex(self):
        result = _compute_pattern_id("shell", ["shell_exec"])
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic_same_inputs(self):
        id1 = _compute_pattern_id("shell", ["a", "b"])
        id2 = _compute_pattern_id("shell", ["a", "b"])
        assert id1 == id2

    def test_sorted_tool_sequence_order_independent(self):
        id1 = _compute_pattern_id("shell", ["a", "b", "c"])
        id2 = _compute_pattern_id("shell", ["c", "a", "b"])
        id3 = _compute_pattern_id("shell", ["b", "c", "a"])
        assert id1 == id2 == id3

    def test_different_task_type_different_id(self):
        id1 = _compute_pattern_id("shell", ["tool_a"])
        id2 = _compute_pattern_id("file", ["tool_a"])
        assert id1 != id2

    def test_different_tools_different_id(self):
        id1 = _compute_pattern_id("shell", ["tool_a"])
        id2 = _compute_pattern_id("shell", ["tool_b"])
        assert id1 != id2

    def test_empty_tool_list_is_valid(self):
        result = _compute_pattern_id("shell", [])
        assert len(result) == 16

    def test_empty_tool_list_differs_from_non_empty(self):
        id1 = _compute_pattern_id("shell", [])
        id2 = _compute_pattern_id("shell", ["tool_a"])
        assert id1 != id2

    def test_single_tool(self):
        result = _compute_pattern_id("shell", ["only_one"])
        assert len(result) == 16

    def test_large_tool_list(self):
        tools = [f"tool_{i}" for i in range(100)]
        result = _compute_pattern_id("shell", tools)
        assert len(result) == 16

    def test_unicode_task_type(self):
        result = _compute_pattern_id("シェル", ["tool_a"])
        assert len(result) == 16

    def test_unicode_tool_names(self):
        result = _compute_pattern_id("shell", ["ツール_1", "ツール_2"])
        assert len(result) == 16

    def test_matches_manual_sha256(self):
        task_type = "shell"
        tools = ["file_write", "shell_exec"]
        key = f"{task_type}:{','.join(sorted(tools))}"
        expected = hashlib.sha256(key.encode()).hexdigest()[:16]
        assert _compute_pattern_id(task_type, tools) == expected

    def test_two_tools_same_as_sorted_one_element_each(self):
        """Verify sort semantics match Python sorted() for alphanumeric names."""
        tools_fwd = ["aaa", "zzz"]
        tools_rev = ["zzz", "aaa"]
        assert _compute_pattern_id("x", tools_fwd) == _compute_pattern_id("x", tools_rev)


# ---------------------------------------------------------------------------
# Playbook constructor
# ---------------------------------------------------------------------------


class TestPlaybookConstructor:
    def test_empty_when_no_file(self, tmp_path):
        pb = _make_pb(tmp_path)
        assert pb.get_relevant("anything") == []

    def test_non_default_path_used(self, tmp_path):
        path = str(tmp_path / "custom" / "pb.json")
        # Should not crash; file doesn't exist yet
        pb = Playbook(store_path=path)
        assert pb.get_relevant("anything") == []

    def test_no_file_created_at_construction(self, tmp_path):
        pb_path = tmp_path / "playbook.json"
        _make_pb(tmp_path)
        assert not pb_path.exists()

    def test_file_created_after_first_record(self, tmp_path):
        pb_path = tmp_path / "playbook.json"
        pb = _make_pb(tmp_path)
        pb.record("shell", "task", ["shell_exec"], "hint")
        assert pb_path.exists()


# ---------------------------------------------------------------------------
# record: new entries
# ---------------------------------------------------------------------------


class TestRecordNewEntry:
    def test_returns_playbook_entry(self, tmp_path):
        pb = _make_pb(tmp_path)
        result = pb.record("shell", "task", ["shell_exec"], "hint")
        assert isinstance(result, PlaybookEntry)

    def test_pattern_id_matches_compute_function(self, tmp_path):
        pb = _make_pb(tmp_path)
        tools = ["shell_exec", "file_write"]
        entry = pb.record("shell", "task", tools, "hint")
        assert entry.pattern_id == _compute_pattern_id("shell", tools)

    def test_task_type_stored(self, tmp_path):
        pb = _make_pb(tmp_path)
        entry = pb.record("file", "task", ["file_read"], "hint")
        assert entry.task_type == "file"

    def test_description_stored(self, tmp_path):
        pb = _make_pb(tmp_path)
        entry = pb.record("shell", "my description", ["shell_exec"], "hint")
        assert entry.description == "my description"

    def test_tool_sequence_stored(self, tmp_path):
        pb = _make_pb(tmp_path)
        tools = ["tool_a", "tool_b"]
        entry = pb.record("shell", "task", tools, "hint")
        assert entry.tool_sequence == tools

    def test_prompt_template_stored(self, tmp_path):
        pb = _make_pb(tmp_path)
        entry = pb.record("shell", "task", ["shell_exec"], "my hint")
        assert entry.prompt_template == "my hint"

    def test_success_count_is_one(self, tmp_path):
        pb = _make_pb(tmp_path)
        entry = pb.record("shell", "task", ["shell_exec"], "hint")
        assert entry.success_count == 1

    def test_promoted_is_false(self, tmp_path):
        pb = _make_pb(tmp_path)
        entry = pb.record("shell", "task", ["shell_exec"], "hint")
        assert entry.promoted is False

    def test_created_at_populated(self, tmp_path):
        pb = _make_pb(tmp_path)
        entry = pb.record("shell", "task", ["shell_exec"], "hint")
        assert entry.created_at

    def test_tool_sequence_is_copy(self, tmp_path):
        """Mutations to the original list must not affect stored entry."""
        pb = _make_pb(tmp_path)
        tools = ["tool_a"]
        entry = pb.record("shell", "task", tools, "hint")
        tools.append("tool_b")
        assert entry.tool_sequence == ["tool_a"]


# ---------------------------------------------------------------------------
# record: duplicate / increment
# ---------------------------------------------------------------------------


class TestRecordDuplicate:
    def test_increments_success_count(self, tmp_path):
        pb = _make_pb(tmp_path)
        pb.record("shell", "desc", ["shell_exec"], "hint")
        e2 = pb.record("shell", "desc", ["shell_exec"], "hint")
        assert e2.success_count == 2

    def test_increments_multiple_times(self, tmp_path):
        pb = _make_pb(tmp_path)
        entry = _record_n(pb, 7)
        assert entry.success_count == 7

    def test_updates_description(self, tmp_path):
        pb = _make_pb(tmp_path)
        pb.record("shell", "old description", ["shell_exec"], "hint")
        e2 = pb.record("shell", "new description", ["shell_exec"], "hint")
        assert e2.description == "new description"

    def test_updates_prompt_template(self, tmp_path):
        pb = _make_pb(tmp_path)
        pb.record("shell", "desc", ["shell_exec"], "old hint")
        e2 = pb.record("shell", "desc", ["shell_exec"], "new hint")
        assert e2.prompt_template == "new hint"

    def test_does_not_reset_created_at(self, tmp_path):
        pb = _make_pb(tmp_path)
        e1 = pb.record("shell", "desc", ["shell_exec"], "hint")
        original_ts = e1.created_at
        e2 = pb.record("shell", "desc2", ["shell_exec"], "hint2")
        assert e2.created_at == original_ts

    def test_does_not_change_promoted_flag(self, tmp_path):
        pb = _make_pb(tmp_path)
        e1 = pb.record("shell", "desc", ["shell_exec"], "hint")
        pb.mark_promoted(e1.pattern_id)
        e2 = pb.record("shell", "desc2", ["shell_exec"], "hint2")
        assert e2.promoted is True

    def test_tool_order_normalised(self, tmp_path):
        pb = _make_pb(tmp_path)
        pb.record("shell", "d1", ["b", "a"], "h1")
        e2 = pb.record("shell", "d2", ["a", "b"], "h2")
        assert e2.success_count == 2

    def test_same_pattern_id_on_duplicate(self, tmp_path):
        pb = _make_pb(tmp_path)
        e1 = pb.record("shell", "d1", ["shell_exec"], "h1")
        e2 = pb.record("shell", "d2", ["shell_exec"], "h2")
        assert e1.pattern_id == e2.pattern_id

    def test_three_different_patterns_three_entries(self, tmp_path):
        pb = _make_pb(tmp_path)
        pb.record("shell", "s", ["shell_exec"], "h")
        pb.record("file", "f", ["file_read"], "h")
        pb.record("network", "n", ["http_get"], "h")
        assert len(pb.get_relevant("shell")) == 1
        assert len(pb.get_relevant("file")) == 1
        assert len(pb.get_relevant("network")) == 1

    def test_saves_after_record(self, tmp_path):
        pb_path = tmp_path / "playbook.json"
        pb = _make_pb(tmp_path)
        pb.record("shell", "task", ["shell_exec"], "hint")
        assert pb_path.exists()


# ---------------------------------------------------------------------------
# get_relevant
# ---------------------------------------------------------------------------


class TestGetRelevant:
    def test_filters_by_task_type(self, tmp_path):
        pb = _make_pb(tmp_path)
        pb.record("shell", "shell task", ["shell_exec"], "h")
        pb.record("file", "file task", ["file_read"], "h")
        results = pb.get_relevant("shell")
        assert all(e.task_type == "shell" for e in results)

    def test_returns_all_when_fewer_than_top_k(self, tmp_path):
        pb = _make_pb(tmp_path)
        pb.record("shell", "t1", ["shell_exec"], "h")
        pb.record("shell", "t2", ["shell_exec", "file_write"], "h")
        results = pb.get_relevant("shell", top_k=10)
        assert len(results) == 2

    def test_respects_top_k(self, tmp_path):
        pb = _make_pb(tmp_path)
        pb.record("shell", "t1", ["shell_exec"], "h")
        pb.record("shell", "t2", ["file_write"], "h")
        pb.record("shell", "t3", ["http_get"], "h")
        results = pb.get_relevant("shell", top_k=2)
        assert len(results) == 2

    def test_ordered_by_success_count_descending(self, tmp_path):
        pb = _make_pb(tmp_path)
        # low-count pattern
        pb.record("shell", "low", ["shell_exec"], "h")
        # high-count pattern (record 5 times)
        for _ in range(5):
            pb.record("shell", "high", ["file_write"], "h")
        results = pb.get_relevant("shell")
        assert results[0].success_count == 5
        assert results[1].success_count == 1

    def test_top_k_one_returns_highest_count(self, tmp_path):
        pb = _make_pb(tmp_path)
        pb.record("shell", "low", ["shell_exec"], "h")
        for _ in range(3):
            pb.record("shell", "high", ["file_write"], "h")
        results = pb.get_relevant("shell", top_k=1)
        assert len(results) == 1
        assert results[0].success_count == 3

    def test_empty_for_unknown_task_type(self, tmp_path):
        pb = _make_pb(tmp_path)
        pb.record("shell", "task", ["shell_exec"], "h")
        assert pb.get_relevant("nonexistent") == []

    def test_empty_playbook_returns_empty(self, tmp_path):
        pb = _make_pb(tmp_path)
        assert pb.get_relevant("shell") == []

    def test_does_not_mutate_internal_state(self, tmp_path):
        pb = _make_pb(tmp_path)
        pb.record("shell", "task", ["shell_exec"], "h")
        before = len(pb.get_relevant("shell"))
        pb.get_relevant("shell")  # second call
        after = len(pb.get_relevant("shell"))
        assert before == after

    def test_default_top_k_is_three(self, tmp_path):
        pb = _make_pb(tmp_path)
        for i in range(5):
            pb.record("shell", f"t{i}", [f"tool_{i}"], "h")
        results = pb.get_relevant("shell")
        assert len(results) == 3


# ---------------------------------------------------------------------------
# get_promotable
# ---------------------------------------------------------------------------


class TestGetPromotable:
    def test_returns_entries_at_threshold(self, tmp_path):
        pb = _make_pb(tmp_path)
        _record_n(pb, 3)
        promotable = pb.get_promotable(threshold=3)
        assert len(promotable) == 1
        assert promotable[0].success_count == 3

    def test_excludes_below_threshold(self, tmp_path):
        pb = _make_pb(tmp_path)
        _record_n(pb, 2)
        assert pb.get_promotable(threshold=3) == []

    def test_excludes_already_promoted(self, tmp_path):
        pb = _make_pb(tmp_path)
        entry = _record_n(pb, 3)
        pb.mark_promoted(entry.pattern_id)
        assert pb.get_promotable(threshold=3) == []

    def test_threshold_one_includes_all(self, tmp_path):
        pb = _make_pb(tmp_path)
        pb.record("shell", "t1", ["shell_exec"], "h")
        pb.record("file", "t2", ["file_read"], "h")
        promotable = pb.get_promotable(threshold=1)
        assert len(promotable) == 2

    def test_default_threshold_is_three(self, tmp_path):
        pb = _make_pb(tmp_path)
        _record_n(pb, 2)
        _record_n(pb, 3, tools=["file_read"])
        promotable = pb.get_promotable()
        assert len(promotable) == 1
        assert promotable[0].success_count == 3

    def test_returns_empty_when_all_promoted(self, tmp_path):
        pb = _make_pb(tmp_path)
        entry = _record_n(pb, 3)
        pb.mark_promoted(entry.pattern_id)
        assert pb.get_promotable() == []

    def test_does_not_mutate_state(self, tmp_path):
        pb = _make_pb(tmp_path)
        _record_n(pb, 5)
        before = len(pb.get_promotable())
        pb.get_promotable()
        after = len(pb.get_promotable())
        assert before == after

    def test_multiple_eligible_entries(self, tmp_path):
        pb = _make_pb(tmp_path)
        _record_n(pb, 3, tools=["shell_exec"])
        _record_n(pb, 4, tools=["file_read"])
        promotable = pb.get_promotable(threshold=3)
        assert len(promotable) == 2


# ---------------------------------------------------------------------------
# mark_promoted
# ---------------------------------------------------------------------------


class TestMarkPromoted:
    def test_sets_promoted_flag(self, tmp_path):
        pb = _make_pb(tmp_path)
        entry = pb.record("shell", "task", ["shell_exec"], "hint")
        assert entry.promoted is False
        pb.mark_promoted(entry.pattern_id)
        assert entry.promoted is True

    def test_reflected_in_get_relevant(self, tmp_path):
        pb = _make_pb(tmp_path)
        entry = pb.record("shell", "task", ["shell_exec"], "hint")
        pb.mark_promoted(entry.pattern_id)
        results = pb.get_relevant("shell")
        assert results[0].promoted is True

    def test_raises_key_error_for_unknown_id(self, tmp_path):
        pb = _make_pb(tmp_path)
        with pytest.raises(KeyError):
            pb.mark_promoted("nonexistent_pattern_id")

    def test_key_error_message_contains_pattern_id(self, tmp_path):
        pb = _make_pb(tmp_path)
        with pytest.raises(KeyError, match="bad_pattern"):
            pb.mark_promoted("bad_pattern")

    def test_promoted_flag_persists_after_save_load(self, tmp_path):
        path = str(tmp_path / "playbook.json")
        pb1 = Playbook(store_path=path)
        _record_n(pb1, 3)
        entry = pb1.get_relevant("shell")[0]
        pb1.mark_promoted(entry.pattern_id)

        pb2 = Playbook(store_path=path)
        results = pb2.get_relevant("shell")
        assert results[0].promoted is True

    def test_mark_promoted_saves_immediately(self, tmp_path):
        pb_path = tmp_path / "playbook.json"
        pb = _make_pb(tmp_path)
        entry = pb.record("shell", "task", ["shell_exec"], "hint")
        pb.mark_promoted(entry.pattern_id)
        # Reload from disk and verify
        pb2 = Playbook(store_path=str(pb_path))
        results = pb2.get_relevant("shell")
        assert results[0].promoted is True


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_round_trip_all_fields(self, tmp_path):
        path = str(tmp_path / "playbook.json")
        pb1 = Playbook(store_path=path)
        e = pb1.record("file", "read config", ["file_read", "file_write"], "check perms")

        pb2 = Playbook(store_path=path)
        results = pb2.get_relevant("file")
        assert len(results) == 1
        loaded = results[0]
        assert loaded.pattern_id == e.pattern_id
        assert loaded.task_type == e.task_type
        assert loaded.description == e.description
        assert loaded.tool_sequence == e.tool_sequence
        assert loaded.prompt_template == e.prompt_template
        assert loaded.success_count == e.success_count
        assert loaded.created_at == e.created_at
        assert loaded.promoted == e.promoted

    def test_round_trip_multiple_entries(self, tmp_path):
        path = str(tmp_path / "playbook.json")
        pb1 = Playbook(store_path=path)
        pb1.record("shell", "s task", ["shell_exec"], "sh hint")
        pb1.record("file", "f task", ["file_read"], "file hint")
        pb1.record("network", "n task", ["http_get"], "net hint")

        pb2 = Playbook(store_path=path)
        assert len(pb2.get_relevant("shell")) == 1
        assert len(pb2.get_relevant("file")) == 1
        assert len(pb2.get_relevant("network")) == 1

    def test_load_nonexistent_file_is_noop(self, tmp_path):
        pb = _make_pb(tmp_path, "does_not_exist.json")
        assert pb.get_relevant("anything") == []

    def test_load_corrupt_json_does_not_crash(self, tmp_path):
        pb_path = tmp_path / "playbook.json"
        pb_path.write_text("NOT VALID JSON {{{")
        pb = Playbook(store_path=str(pb_path))
        # Should be empty (graceful degradation)
        assert pb.get_relevant("shell") == []

    def test_load_empty_json_array(self, tmp_path):
        pb_path = tmp_path / "playbook.json"
        pb_path.write_text("[]")
        pb = Playbook(store_path=str(pb_path))
        assert pb.get_relevant("shell") == []

    def test_load_empty_file_does_not_crash(self, tmp_path):
        pb_path = tmp_path / "playbook.json"
        pb_path.write_text("")
        pb = Playbook(store_path=str(pb_path))
        assert pb.get_relevant("shell") == []

    def test_save_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        pb = Playbook(store_path=str(nested / "playbook.json"))
        pb.record("shell", "task", ["shell_exec"], "hint")
        assert (nested / "playbook.json").exists()

    def test_save_is_valid_json(self, tmp_path):
        pb_path = tmp_path / "playbook.json"
        pb = _make_pb(tmp_path)
        pb.record("shell", "task", ["shell_exec"], "hint")
        with open(str(pb_path)) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_save_contains_all_fields(self, tmp_path):
        pb_path = tmp_path / "playbook.json"
        pb = _make_pb(tmp_path)
        pb.record("shell", "task", ["shell_exec"], "use rsync")
        with open(str(pb_path)) as f:
            data = json.load(f)
        item = data[0]
        assert "pattern_id" in item
        assert "task_type" in item
        assert "description" in item
        assert "tool_sequence" in item
        assert "prompt_template" in item
        assert "success_count" in item
        assert "created_at" in item
        assert "promoted" in item

    def test_incremented_count_persists(self, tmp_path):
        path = str(tmp_path / "playbook.json")
        pb1 = Playbook(store_path=path)
        _record_n(pb1, 5)

        pb2 = Playbook(store_path=path)
        results = pb2.get_relevant("shell")
        assert results[0].success_count == 5


# ---------------------------------------------------------------------------
# Persistence across instances
# ---------------------------------------------------------------------------


class TestPersistenceAcrossInstances:
    def test_second_instance_reads_first_records(self, tmp_path):
        path = str(tmp_path / "playbook.json")
        pb1 = Playbook(store_path=path)
        pb1.record("shell", "deploy", ["shell_exec", "file_write"], "use rsync")

        pb2 = Playbook(store_path=path)
        results = pb2.get_relevant("shell")
        assert len(results) == 1
        assert results[0].description == "deploy"
        assert results[0].prompt_template == "use rsync"

    def test_second_instance_can_increment_first_pattern(self, tmp_path):
        path = str(tmp_path / "playbook.json")
        pb1 = Playbook(store_path=path)
        pb1.record("shell", "desc", ["shell_exec"], "hint")

        pb2 = Playbook(store_path=path)
        pb2.record("shell", "desc2", ["shell_exec"], "hint2")
        results = pb2.get_relevant("shell")
        assert results[0].success_count == 2


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_record_same_pattern(self, tmp_path):
        pb = _make_pb(tmp_path)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                pb.record("shell", "task", ["shell_exec"], "hint")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        results = pb.get_relevant("shell")
        assert len(results) == 1
        assert results[0].success_count == 30

    def test_concurrent_record_distinct_patterns(self, tmp_path):
        pb = _make_pb(tmp_path)
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                pb.record(f"type_{i}", f"task{i}", [f"tool_{i}"], f"hint{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Each thread created its own unique pattern
        total = sum(len(pb.get_relevant(f"type_{i}")) for i in range(20))
        assert total == 20

    def test_concurrent_get_relevant_does_not_crash(self, tmp_path):
        pb = _make_pb(tmp_path)
        for i in range(5):
            pb.record("shell", f"task{i}", [f"tool_{i}"], "hint")
        errors: list[Exception] = []

        def reader() -> None:
            try:
                pb.get_relevant("shell", top_k=3)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_mark_promoted_and_get_promotable(self, tmp_path):
        pb = _make_pb(tmp_path)
        # Create 10 promotable patterns
        for i in range(10):
            for _ in range(3):
                pb.record("shell", f"task{i}", [f"tool_{i}"], "hint")

        errors: list[Exception] = []
        promoted_ids: list[str] = []
        lock = threading.Lock()

        def promoter() -> None:
            try:
                promotable = pb.get_promotable()
                for entry in promotable:
                    try:
                        pb.mark_promoted(entry.pattern_id)
                        with lock:
                            promoted_ids.append(entry.pattern_id)
                    except KeyError:
                        pass  # already promoted by another thread
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=promoter) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # No remaining promotable entries
        assert pb.get_promotable() == []
