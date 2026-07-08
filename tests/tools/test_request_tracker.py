"""Tests for missy.tools.intelligence.request_tracker."""

from __future__ import annotations

from pathlib import Path

import pytest

from missy.tools.intelligence.request_tracker import (
    RequestTracker,
    _flatten_tool_lists,
    _hash,
    _normalise,
    _pick_examples,
)

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


class TestNormalise:
    def test_lowercases(self):
        assert _normalise("Hello World") == "hello world"

    def test_strips_url(self):
        result = _normalise("fetch https://example.com/api/v2 please")
        assert "__url__" in result
        assert "example.com" not in result

    def test_strips_email(self):
        result = _normalise("send report to user@domain.org")
        assert "__email__" in result

    def test_strips_path(self):
        result = _normalise("read /home/alice/docs/notes.txt")
        assert "__path__" in result

    def test_strips_long_token(self):
        token = "A" * 25
        result = _normalise(f"use token {token} for auth")
        assert "__token__" in result

    def test_strips_numbers(self):
        result = _normalise("buy 5 items at $12.99 each")
        assert "__num__" in result

    def test_caps_at_500(self):
        long_text = "word " * 200
        result = _normalise(long_text)
        assert len(result) <= 500

    def test_empty_string(self):
        assert _normalise("") == ""


class TestHash:
    def test_deterministic(self):
        assert _hash("hello") == _hash("hello")

    def test_length(self):
        assert len(_hash("some text")) == 16

    def test_differs_for_different_inputs(self):
        assert _hash("abc") != _hash("xyz")


class TestFlattenToolLists:
    def test_single_list(self):
        raw = '["tool_a", "tool_b"]'
        assert _flatten_tool_lists(raw) == ["tool_a", "tool_b"]

    def test_multiple_lists(self):
        raw = '["tool_a"]||["tool_b", "tool_c"]'
        result = _flatten_tool_lists(raw)
        assert set(result) == {"tool_a", "tool_b", "tool_c"}

    def test_empty_string(self):
        assert _flatten_tool_lists("") == []

    def test_invalid_json_skipped(self):
        raw = '["ok"]||not_json'
        assert _flatten_tool_lists(raw) == ["ok"]


class TestPickExamples:
    def test_deduplicates(self):
        samples = ["a", "a", "b", "c"]
        result = _pick_examples(samples, n=3)
        assert len(result) == 3
        assert result.count("a") == 1

    def test_truncates_at_200(self):
        samples = ["x" * 300]
        result = _pick_examples(samples)
        assert len(result[0]) == 200

    def test_returns_up_to_n(self):
        samples = ["a", "b", "c", "d"]
        result = _pick_examples(samples, n=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# RequestTracker
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker(tmp_path: Path) -> RequestTracker:
    return RequestTracker(db_path=tmp_path / "test_tracker.db")


class TestRequestTrackerRecord:
    def test_record_returns_event(self, tracker):
        event = tracker.record("sess1", "please summarise this file")
        assert event.session_id == "sess1"
        assert event.raw_message == "please summarise this file"
        assert event.pattern_key  # non-empty hash

    def test_record_stores_event(self, tracker):
        assert tracker.event_count() == 0
        tracker.record("s", "hello world")
        assert tracker.event_count() == 1

    def test_multiple_events_same_pattern(self, tracker):
        for _ in range(3):
            tracker.record("s", "summarise the quarterly report")
        assert tracker.event_count() == 3
        assert tracker.pattern_count() == 1

    def test_tool_calls_stored(self, tracker):
        event = tracker.record("s", "read a file", tool_calls=["file_read", "calculator"])
        assert "file_read" in event.tool_calls
        assert "calculator" in event.tool_calls

    def test_metadata_stored(self, tracker):
        event = tracker.record("s", "msg", metadata={"channel": "discord"})
        assert event.metadata == {"channel": "discord"}


class TestGetFrequentPatterns:
    def test_no_patterns_below_min(self, tracker):
        tracker.record("s", "unique message abc")
        tracker.record("s", "another unique message xyz")
        patterns = tracker.get_frequent_patterns(min_count=3)
        assert patterns == []

    def test_surfaces_repeated_pattern(self, tracker):
        for _ in range(4):
            tracker.record("s", "convert the pdf file to text")
        patterns = tracker.get_frequent_patterns(min_count=3)
        assert len(patterns) == 1
        p = patterns[0]
        assert p.count == 4
        assert p.frequency_score > 0

    def test_common_tools_computed(self, tracker):
        for _ in range(4):
            tracker.record("s", "search for files matching pattern", tool_calls=["list_files"])
        patterns = tracker.get_frequent_patterns(min_count=3)
        assert "list_files" in patterns[0].common_tools

    def test_score_sorted_desc(self, tracker):
        for _ in range(5):
            tracker.record("s", "do the weekly report")
        for _ in range(3):
            tracker.record("s", "check the daily log please")
        patterns = tracker.get_frequent_patterns(min_count=3)
        scores = [p.frequency_score for p in patterns]
        assert scores == sorted(scores, reverse=True)

    def test_limit_respected(self, tracker):
        for i in range(10):
            for _ in range(4):
                tracker.record("s", f"task number {i} unique name here please")
        # Pattern normalization will group by their hash
        patterns = tracker.get_frequent_patterns(min_count=3, limit=5)
        assert len(patterns) <= 5


class TestPurge:
    def test_purge_removes_old_events(self, tracker):
        tracker.record("s", "old message")
        # Purge with a future date — removes everything
        deleted = tracker.purge_before("2099-01-01T00:00:00+00:00")
        assert deleted == 1
        assert tracker.event_count() == 0

    def test_purge_keeps_recent_events(self, tracker):
        tracker.record("s", "recent message")
        deleted = tracker.purge_before("2000-01-01T00:00:00+00:00")
        assert deleted == 0
        assert tracker.event_count() == 1
