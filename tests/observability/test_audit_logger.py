"""Tests for missy.observability.audit_logger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from missy.core.events import AuditEvent, EventBus
from missy.observability.audit_logger import AuditLogger, init_audit_logger, get_audit_logger


@pytest.fixture
def bus() -> EventBus:
    """A fresh EventBus for each test — avoids cross-test state pollution."""
    return EventBus()


@pytest.fixture
def log_path(tmp_path: Path) -> str:
    return str(tmp_path / "audit.jsonl")


@pytest.fixture
def audit_logger(log_path: str, bus: EventBus) -> AuditLogger:
    return AuditLogger(log_path=log_path, bus=bus)


def _publish(bus: EventBus, event_type: str, category: str, result: str, **detail):
    bus.publish(
        AuditEvent.now(
            session_id="s1",
            task_id="t1",
            event_type=event_type,
            category=category,
            result=result,
            detail=detail,
        )
    )


class TestAuditLoggerInit:
    def test_creates_parent_directory(self, tmp_path: Path, bus: EventBus):
        nested = str(tmp_path / "sub" / "dir" / "audit.jsonl")
        al = AuditLogger(log_path=nested, bus=bus)
        assert Path(nested).parent.exists()

    def test_log_path_set_correctly(self, log_path: str, audit_logger: AuditLogger):
        assert str(audit_logger.log_path) == log_path


class TestHandleEvent:
    def test_event_written_to_file(self, audit_logger: AuditLogger, bus: EventBus, log_path: str):
        _publish(bus, "network.request", "network", "allow", host="example.com")
        lines = Path(log_path).read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event_type"] == "network.request"
        assert record["result"] == "allow"
        assert record["category"] == "network"

    def test_multiple_events_appended(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: str
    ):
        _publish(bus, "ev.one", "network", "allow")
        _publish(bus, "ev.two", "shell", "deny")
        _publish(bus, "ev.three", "filesystem", "allow")
        lines = [ln for ln in Path(log_path).read_text().splitlines() if ln.strip()]
        assert len(lines) == 3

    def test_record_contains_all_fields(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: str
    ):
        _publish(bus, "test.event", "scheduler", "error", key="value")
        record = json.loads(Path(log_path).read_text().strip())
        for field in ("timestamp", "session_id", "task_id", "event_type",
                      "category", "result", "detail", "policy_rule"):
            assert field in record, f"Missing field: {field}"

    def test_detail_dict_is_preserved(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: str
    ):
        _publish(bus, "fs.read", "filesystem", "allow", path="/tmp/x.txt")
        record = json.loads(Path(log_path).read_text().strip())
        assert record["detail"]["path"] == "/tmp/x.txt"

    def test_existing_subscribers_still_called(
        self, bus: EventBus, log_path: str
    ):
        received = []
        bus.subscribe("test.event", received.append)
        AuditLogger(log_path=log_path, bus=bus)
        _publish(bus, "test.event", "network", "allow")
        assert len(received) == 1


class TestGetRecentEvents:
    def test_returns_empty_when_file_missing(self, bus: EventBus, tmp_path: Path):
        al = AuditLogger(log_path=str(tmp_path / "missing.jsonl"), bus=bus)
        assert al.get_recent_events() == []

    def test_returns_all_events_within_limit(
        self, audit_logger: AuditLogger, bus: EventBus
    ):
        for i in range(5):
            _publish(bus, f"ev.{i}", "network", "allow")
        events = audit_logger.get_recent_events(limit=10)
        assert len(events) == 5

    def test_limit_is_applied_newest_first(
        self, audit_logger: AuditLogger, bus: EventBus
    ):
        for i in range(10):
            _publish(bus, f"ev.{i}", "network", "allow")
        events = audit_logger.get_recent_events(limit=3)
        assert len(events) == 3
        # The 3 most recent: ev.7, ev.8, ev.9
        assert events[-1]["event_type"] == "ev.9"

    def test_result_is_list_of_dicts(self, audit_logger: AuditLogger, bus: EventBus):
        _publish(bus, "x", "network", "allow")
        events = audit_logger.get_recent_events()
        assert isinstance(events, list)
        assert isinstance(events[0], dict)


class TestGetPolicyViolations:
    def test_returns_only_deny_events(
        self, audit_logger: AuditLogger, bus: EventBus
    ):
        _publish(bus, "net.req", "network", "allow")
        _publish(bus, "shell.exec", "shell", "deny", cmd="rm -rf /")
        _publish(bus, "fs.write", "filesystem", "deny", path="/etc/hosts")
        violations = audit_logger.get_policy_violations()
        assert len(violations) == 2
        for v in violations:
            assert v["result"] == "deny"

    def test_returns_empty_when_no_denies(
        self, audit_logger: AuditLogger, bus: EventBus
    ):
        _publish(bus, "ev.ok", "network", "allow")
        assert audit_logger.get_policy_violations() == []

    def test_limit_is_applied(self, audit_logger: AuditLogger, bus: EventBus):
        for i in range(10):
            _publish(bus, f"ev.{i}", "shell", "deny")
        violations = audit_logger.get_policy_violations(limit=3)
        assert len(violations) == 3

    def test_returns_empty_when_file_missing(self, bus: EventBus, tmp_path: Path):
        al = AuditLogger(log_path=str(tmp_path / "missing.jsonl"), bus=bus)
        assert al.get_policy_violations() == []

    def test_violations_returned_in_chronological_order(
        self, audit_logger: AuditLogger, bus: EventBus
    ):
        for i in range(3):
            _publish(bus, f"ev.{i}", "shell", "deny")
        violations = audit_logger.get_policy_violations()
        types = [v["event_type"] for v in violations]
        assert types == ["ev.0", "ev.1", "ev.2"]


class TestSingleton:
    def test_init_audit_logger_returns_audit_logger(self, tmp_path: Path):
        al = init_audit_logger(str(tmp_path / "a.jsonl"))
        assert isinstance(al, AuditLogger)

    def test_get_audit_logger_returns_same_instance(self, tmp_path: Path):
        al = init_audit_logger(str(tmp_path / "b.jsonl"))
        assert get_audit_logger() is al

    def test_get_audit_logger_raises_before_init(self, monkeypatch):
        import missy.observability.audit_logger as mod
        monkeypatch.setattr(mod, "_audit_logger", None)
        with pytest.raises(RuntimeError, match="AuditLogger has not been initialised"):
            get_audit_logger()
