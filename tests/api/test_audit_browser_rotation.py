"""DGAP-03 / PERF-03 / PERF-04: audit browser covers rotated logs with a
bounded scan; the log tail reads backwards."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from missy.api import audit_browser
from missy.api.audit_browser import query_audit_events
from missy.core.events import EventBus
from missy.observability.audit_logger import AuditLogger, _reverse_lines


def _write(path, events):
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


def _event(i, event_type="tool.execute", ts=None):
    return {
        "timestamp": ts or f"2026-09-{(i % 28) + 1:02d}T00:00:{i % 60:02d}+00:00",
        "session_id": f"s{i}",
        "task_id": "",
        "event_type": event_type,
        "category": "tool",
        "result": "allow",
        "detail": {"n": i},
        "policy_rule": None,
    }


def _logger(tmp_path):
    log = tmp_path / "audit.jsonl"
    rotated = tmp_path / "audit.jsonl.20260101_000000"
    _write(rotated, [_event(1, "provider.rotated_only", ts="2026-01-01T00:00:00+00:00")])
    _write(log, [_event(i) for i in range(2, 10)])
    obj = MagicMock()
    real = AuditLogger.__new__(AuditLogger)
    real.log_path = log
    obj.iter_lines_newest_first = real.iter_lines_newest_first
    return obj


def test_filter_finds_event_only_in_rotated_log(tmp_path):
    with patch("missy.observability.audit_logger.get_audit_logger", return_value=_logger(tmp_path)):
        result = query_audit_events({"event_type": "provider.rotated_only"}, bus=EventBus())
    assert [e["event_type"] for e in result["events"]] == ["provider.rotated_only"]
    assert result["total_is_estimate"] is False


def test_scan_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_browser, "MAX_AUDIT_SCAN_LINES", 3)
    with patch("missy.observability.audit_logger.get_audit_logger", return_value=_logger(tmp_path)):
        result = query_audit_events({"event_type": "provider.rotated_only"}, bus=EventBus())
    assert result["events"] == []
    assert result["total_is_estimate"] is True
    assert result["scanned_lines"] == 3


def test_early_stop_once_page_filled(tmp_path):
    with patch("missy.observability.audit_logger.get_audit_logger", return_value=_logger(tmp_path)):
        result = query_audit_events({"limit": 2}, bus=EventBus())
    assert result["count"] == 2
    assert result["has_more"] is True


def test_reverse_lines_across_blocks(tmp_path):
    p = tmp_path / "f.log"
    lines = [f"line-{i}-" + "x" * 50 for i in range(500)]
    p.write_text("\n".join(lines) + "\n")
    got = [ln for ln in _reverse_lines(p, block_size=97) if ln]
    assert got == list(reversed(lines))


def test_logs_tail_reads_last_lines(tmp_path, monkeypatch):
    log = tmp_path / "missy.log"
    log.write_text("".join(f"row {i}\n" for i in range(10_000)))
    monkeypatch.setenv("MISSY_APP_LOG", str(log))
    tail: list[str] = []
    for line in _reverse_lines(log):
        if line:
            tail.insert(0, line)
        if len(tail) == 3:
            break
    assert tail == ["row 9997", "row 9998", "row 9999"]
