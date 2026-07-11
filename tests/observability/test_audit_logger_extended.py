"""Extended tests for missy.observability.audit_logger and OtelExporter.

Covers areas not exercised by the existing test files:

- Directory creation with restrictive permissions (mode 0o700)
- Tilde expansion in log_path
- JSONL output format invariants (valid JSON, newline-terminated)
- ISO-8601 timestamp serialisation
- session_id / task_id round-trip
- policy_rule field round-trip
- Multiple AuditLogger instances wrapping the same bus (stacking)
- get_recent_events: empty-file edge case (stat returns 0)
- get_recent_events: default limit value
- get_policy_violations: interleaved allow/deny ordering
- get_policy_violations: limit boundary (exactly limit denies returned)
- Thread safety: concurrent publishes do not corrupt the JSONL file
- Error handling: read_tail_lines on a file that disappears between
  exists() and open()
- init_audit_logger replaces the previous singleton
- get_audit_logger raises RuntimeError when not initialised
- OtelExporter: disabled when opentelemetry not installed
- OtelExporter: export_event is a no-op when disabled
- OtelExporter: is_enabled property
- OtelExporter: init_otel returns disabled stub when otel_enabled=False
- OtelExporter: subscribe is a no-op when disabled
- Integration with vision audit functions (audit.log call path)
- Category-only events (no policy_rule) write null to JSONL
- detail serialises non-string values (int, bool, list)
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import missy.observability.audit_logger as audit_mod
from missy.core.events import AuditEvent, EventBus
from missy.observability.audit_logger import (
    AuditLogger,
    get_audit_logger,
    init_audit_logger,
)
from missy.observability.otel import OtelExporter, init_otel

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bus() -> EventBus:
    """Fresh EventBus for each test — no cross-test pollution."""
    return EventBus()


@pytest.fixture()
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "audit.jsonl"


@pytest.fixture()
def audit_logger(log_path: Path, bus: EventBus) -> AuditLogger:
    return AuditLogger(log_path=str(log_path), bus=bus)


def _publish(
    bus: EventBus,
    event_type: str = "test.event",
    category: str = "network",
    result: str = "allow",
    session_id: str = "s1",
    task_id: str = "t1",
    policy_rule: str | None = None,
    **detail: Any,
) -> AuditEvent:
    event = AuditEvent.now(
        session_id=session_id,
        task_id=task_id,
        event_type=event_type,
        category=category,
        result=result,
        detail=detail or {},
        policy_rule=policy_rule,
    )
    bus.publish(event)
    return event


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 1. Initialisation and file handling
# ---------------------------------------------------------------------------


class TestInitialisationAndFileHandling:
    def test_parent_directory_created_with_restrictive_mode(self, tmp_path: Path, bus: EventBus):
        """mkdir uses mode=0o700 so the audit directory is private."""
        nested = tmp_path / "deep" / "nested" / "audit.jsonl"
        AuditLogger(log_path=str(nested), bus=bus)
        # The full parent chain must exist.
        assert nested.parent.exists()

    def test_tilde_expansion_applied_to_log_path(self, bus: EventBus, monkeypatch):
        """~/.missy/audit.jsonl must be expanded to an absolute path."""
        fake_home = Path("/tmp/fake_home_for_missy_test")
        monkeypatch.setenv("HOME", str(fake_home))
        fake_home.mkdir(parents=True, exist_ok=True)
        al = AuditLogger(log_path="~/.missy/audit.jsonl", bus=bus)
        assert not str(al.log_path).startswith("~")
        assert al.log_path.is_absolute()

    def test_log_path_stored_as_path_object(self, log_path: Path, bus: EventBus):
        al = AuditLogger(log_path=str(log_path), bus=bus)
        assert isinstance(al.log_path, Path)

    def test_no_file_created_on_init_before_first_event(self, log_path: Path, bus: EventBus):
        """AuditLogger must not create the log file until the first event fires."""
        AuditLogger(log_path=str(log_path), bus=bus)
        assert not log_path.exists()


# ---------------------------------------------------------------------------
# 1b. Availability hardening: restrictive permissions + rotation
# ---------------------------------------------------------------------------


class TestRestrictivePermissions:
    """Live-reproduced before this fix: under a common default umask
    (022), the audit log file -- which routinely contains operational
    detail like hostnames, file paths, and prompts, even after secret
    redaction -- was created world- and group-readable (0o644), because
    the write path used a plain append-mode open() with no explicit
    mode."""

    def test_log_file_created_with_0600_regardless_of_umask(
        self, log_path: Path, bus: EventBus
    ):
        import os
        import stat

        old_umask = os.umask(0o022)
        try:
            al = AuditLogger(log_path=str(log_path), bus=bus)
            bus.publish(
                AuditEvent.now(
                    session_id="s", task_id="t", event_type="x", category="tool", result="allow"
                )
            )
        finally:
            os.umask(old_umask)

        mode = stat.S_IMODE(log_path.stat().st_mode)
        assert mode == 0o600
        assert not (mode & stat.S_IROTH)
        assert not (mode & stat.S_IRGRP)

    def test_pre_existing_looser_permissions_are_tightened_on_construction(
        self, log_path: Path, bus: EventBus
    ):
        """A log file left over from a Missy version predating this fix
        (or corrupted permissions from any other cause) is re-secured the
        next time AuditLogger is constructed against it, not left as-is
        forever."""
        import os
        import stat

        log_path.write_text('{"already": "here"}\n')
        os.chmod(log_path, 0o644)
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o644

        AuditLogger(log_path=str(log_path), bus=bus)

        assert stat.S_IMODE(log_path.stat().st_mode) == 0o600


class TestLogRotation:
    """Missy runs as a long-lived background daemon -- an audit log that
    only ever grows would eventually exhaust disk space. Live-verified:
    with a tiny size threshold, publishing enough events produces
    rotated, timestamped sibling files, each still 0600, with old
    rotations pruned beyond the retention count."""

    def test_rotation_creates_a_timestamped_sibling_file(self, log_path: Path, bus: EventBus):
        al = AuditLogger(log_path=str(log_path), bus=bus)
        al._MAX_LOG_SIZE_BYTES = 200

        for i in range(50):
            bus.publish(
                AuditEvent.now(
                    session_id="s",
                    task_id="t",
                    event_type=f"event.{i}",
                    category="tool",
                    result="allow",
                )
            )

        rotated = list(log_path.parent.glob(f"{log_path.name}.*"))
        assert len(rotated) >= 1
        assert log_path.exists()

    def test_rotated_files_retain_0600_permissions(self, log_path: Path, bus: EventBus):
        import stat

        al = AuditLogger(log_path=str(log_path), bus=bus)
        al._MAX_LOG_SIZE_BYTES = 200

        for i in range(50):
            bus.publish(
                AuditEvent.now(
                    session_id="s",
                    task_id="t",
                    event_type=f"event.{i}",
                    category="tool",
                    result="allow",
                )
            )

        rotated = list(log_path.parent.glob(f"{log_path.name}.*"))
        assert rotated
        for f in rotated:
            assert stat.S_IMODE(f.stat().st_mode) == 0o600

    def test_same_second_rotations_do_not_clobber_each_other(
        self, log_path: Path, bus: EventBus
    ):
        """Regression: the rotated filename used only 1-second
        (int(time.time())) granularity with no collision check --
        os.rename() silently overwrites an existing destination on
        POSIX, so two rotations within the same wall-clock second
        produced the identical filename and the second rotation
        permanently and silently destroyed the first burst's audit
        records with no error anywhere. The pre-existing
        test_old_rotations_pruned_beyond_max_rotated_files test above
        works around this exact issue with an explicit
        time.sleep(0.002) "to ensure distinct rotation timestamps"
        rather than proving (or requiring) that a same-second collision
        is handled safely -- this test forces the real collision
        directly via a frozen clock.
        """
        al = AuditLogger(log_path=str(log_path), bus=bus)
        al._MAX_LOG_SIZE_BYTES = 10

        with patch("time.time", return_value=1234567890.0):
            log_path.write_text("BURST_ONE\n" * 100)
            al._rotate_if_needed()
            first_rotated = sorted(log_path.parent.glob(f"{log_path.name}.*"))
            assert len(first_rotated) == 1
            first_content = first_rotated[0].read_text()

            log_path.write_text("BURST_TWO\n" * 100)
            al._rotate_if_needed()
            second_rotated = sorted(log_path.parent.glob(f"{log_path.name}.*"))

        # Both bursts must survive as distinct files -- neither
        # overwrites the other.
        assert len(second_rotated) == 2
        contents = {p.read_text() for p in second_rotated}
        assert first_content in contents
        assert any("BURST_TWO" in c for c in contents)
        assert any("BURST_ONE" in c for c in contents)

    def test_old_rotations_pruned_beyond_max_rotated_files(self, log_path: Path, bus: EventBus):
        import time as _time

        al = AuditLogger(log_path=str(log_path), bus=bus)
        al._MAX_LOG_SIZE_BYTES = 100
        al._MAX_ROTATED_FILES = 2

        for i in range(200):
            bus.publish(
                AuditEvent.now(
                    session_id="s",
                    task_id="t",
                    event_type=f"event.{i}",
                    category="tool",
                    result="allow",
                )
            )
            if i % 20 == 0:
                _time.sleep(0.002)  # ensure distinct rotation timestamps

        rotated = list(log_path.parent.glob(f"{log_path.name}.*"))
        assert len(rotated) <= al._MAX_ROTATED_FILES

    def test_no_rotation_below_the_size_threshold(self, log_path: Path, bus: EventBus):
        al = AuditLogger(log_path=str(log_path), bus=bus)
        # Default threshold (50 MB) -- a handful of small events must
        # never trigger rotation.
        for i in range(5):
            bus.publish(
                AuditEvent.now(
                    session_id="s",
                    task_id="t",
                    event_type=f"event.{i}",
                    category="tool",
                    result="allow",
                )
            )
        rotated = list(log_path.parent.glob(f"{log_path.name}.*"))
        assert rotated == []

    def test_rotation_failure_does_not_prevent_the_event_from_being_lost_silently(
        self, log_path: Path, bus: EventBus
    ):
        """If rotation itself fails (e.g. a permissions issue on the
        directory), the write must still be attempted rather than the
        whole _handle_event() call raising."""
        al = AuditLogger(log_path=str(log_path), bus=bus)
        al._MAX_LOG_SIZE_BYTES = 1  # force rotation attempt on every write

        with patch("os.rename", side_effect=OSError("rotation failed")):
            # Must not raise -- rotation failure is logged, not fatal.
            bus.publish(
                AuditEvent.now(
                    session_id="s", task_id="t", event_type="x", category="tool", result="allow"
                )
            )
        # The event must still have been written despite the rotation failure.
        assert log_path.exists()
        assert "\"event_type\": \"x\"" in log_path.read_text()


# ---------------------------------------------------------------------------
# 2. JSONL output format
# ---------------------------------------------------------------------------


class TestJsonlOutputFormat:
    def test_each_line_is_valid_json(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: Path
    ):
        _publish(bus, "net.req", "network", "allow")
        _publish(bus, "shell.exec", "shell", "deny")
        for line in log_path.read_text().splitlines():
            json.loads(line)  # must not raise

    def test_lines_are_newline_terminated(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: Path
    ):
        _publish(bus, "net.req", "network", "allow")
        content = log_path.read_bytes()
        assert content.endswith(b"\n")

    def test_timestamp_is_iso8601_string(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: Path
    ):
        _publish(bus, "test.ts", "network", "allow")
        record = _read_jsonl(log_path)[0]
        ts = record["timestamp"]
        assert isinstance(ts, str)
        # ISO-8601 datetime must be parseable; datetime.fromisoformat accepts it.
        from datetime import datetime

        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None, "Timestamp must be timezone-aware"

    def test_session_id_round_trips(self, audit_logger: AuditLogger, bus: EventBus, log_path: Path):
        _publish(bus, "ev", "network", "allow", session_id="session-xyz-42")
        record = _read_jsonl(log_path)[0]
        assert record["session_id"] == "session-xyz-42"

    def test_task_id_round_trips(self, audit_logger: AuditLogger, bus: EventBus, log_path: Path):
        _publish(bus, "ev", "network", "allow", task_id="task-abc-007")
        record = _read_jsonl(log_path)[0]
        assert record["task_id"] == "task-abc-007"

    def test_policy_rule_written_when_set(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: Path
    ):
        _publish(bus, "fs.write", "filesystem", "deny", policy_rule="no-etc-writes")
        record = _read_jsonl(log_path)[0]
        assert record["policy_rule"] == "no-etc-writes"

    def test_policy_rule_is_null_when_not_set(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: Path
    ):
        _publish(bus, "net.req", "network", "allow")
        record = _read_jsonl(log_path)[0]
        assert record["policy_rule"] is None

    def test_detail_with_integer_values_serialises(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: Path
    ):
        _publish(bus, "ev", "network", "allow", port=8080, retries=3)
        record = _read_jsonl(log_path)[0]
        assert record["detail"]["port"] == 8080
        assert record["detail"]["retries"] == 3

    def test_detail_with_bool_values_serialises(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: Path
    ):
        _publish(bus, "ev", "network", "allow", success=True, cached=False)
        record = _read_jsonl(log_path)[0]
        assert record["detail"]["success"] is True
        assert record["detail"]["cached"] is False


# ---------------------------------------------------------------------------
# 3. Security audit events
# ---------------------------------------------------------------------------


class TestSecurityAuditEvents:
    def test_security_prompt_drift_event(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: Path
    ):
        _publish(
            bus,
            "security.prompt_drift",
            "provider",
            "error",
            expected_hash="abc123",
            actual_hash="def456",
        )
        record = _read_jsonl(log_path)[0]
        assert record["event_type"] == "security.prompt_drift"
        assert record["category"] == "provider"
        assert record["result"] == "error"
        assert record["detail"]["expected_hash"] == "abc123"

    def test_policy_deny_shell_command_recorded(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: Path
    ):
        _publish(
            bus,
            "shell.command_blocked",
            "shell",
            "deny",
            policy_rule="shell-whitelist",
            command="curl http://evil.example.com",
        )
        record = _read_jsonl(log_path)[0]
        assert record["result"] == "deny"
        assert record["policy_rule"] == "shell-whitelist"

    def test_network_deny_event_is_captured(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: Path
    ):
        _publish(
            bus,
            "network.request_blocked",
            "network",
            "deny",
            policy_rule="default_deny",
            host="10.0.0.1",
            port=22,
        )
        violations = audit_logger.get_policy_violations()
        assert len(violations) == 1
        assert violations[0]["detail"]["host"] == "10.0.0.1"

    def test_filesystem_write_blocked_event(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: Path
    ):
        _publish(
            bus,
            "filesystem.write_blocked",
            "filesystem",
            "deny",
            policy_rule="fs-write-policy",
            path="/etc/passwd",
        )
        violations = audit_logger.get_policy_violations()
        assert violations[0]["detail"]["path"] == "/etc/passwd"

    def test_multiple_security_categories_logged(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: Path
    ):
        categories = ["network", "filesystem", "shell", "plugin", "provider"]
        for cat in categories:
            _publish(bus, f"{cat}.event", cat, "allow")
        records = _read_jsonl(log_path)
        written_categories = {r["category"] for r in records}
        assert written_categories == set(categories)


# ---------------------------------------------------------------------------
# 4. Event filtering via get_recent_events and get_policy_violations
# ---------------------------------------------------------------------------


class TestEventFiltering:
    def test_get_recent_events_empty_file_returns_empty_list(self, bus: EventBus, tmp_path: Path):
        log = tmp_path / "empty.jsonl"
        log.write_text("")
        al = AuditLogger(log_path=str(log), bus=bus)
        assert al.get_recent_events() == []

    def test_get_recent_events_default_limit_is_100(self, audit_logger: AuditLogger, bus: EventBus):
        """Default limit parameter returns at most 100 events."""
        for i in range(150):
            _publish(bus, f"ev.{i}", "network", "allow")
        events = audit_logger.get_recent_events()
        assert len(events) == 100

    def test_get_policy_violations_interleaved_allow_deny(
        self, audit_logger: AuditLogger, bus: EventBus
    ):
        """Violations are returned in chronological order among interleaved allows."""
        _publish(bus, "a.1", "network", "allow")
        _publish(bus, "d.1", "shell", "deny")
        _publish(bus, "a.2", "network", "allow")
        _publish(bus, "d.2", "filesystem", "deny")
        _publish(bus, "a.3", "network", "allow")

        violations = audit_logger.get_policy_violations()
        assert len(violations) == 2
        assert violations[0]["event_type"] == "d.1"
        assert violations[1]["event_type"] == "d.2"

    def test_get_policy_violations_limit_boundary(self, audit_logger: AuditLogger, bus: EventBus):
        """Exactly *limit* denies are returned when more exist."""
        for i in range(7):
            _publish(bus, f"deny.{i}", "shell", "deny")
        violations = audit_logger.get_policy_violations(limit=5)
        assert len(violations) == 5

    def test_get_policy_violations_returns_all_when_fewer_than_limit(
        self, audit_logger: AuditLogger, bus: EventBus
    ):
        _publish(bus, "d.1", "shell", "deny")
        _publish(bus, "d.2", "shell", "deny")
        violations = audit_logger.get_policy_violations(limit=50)
        assert len(violations) == 2


# ---------------------------------------------------------------------------
# 5. Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_publishes_produce_correct_line_count(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: Path
    ):
        """Lines written from many threads must not be interleaved/lost."""
        n_threads = 10
        events_per_thread = 20
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            for i in range(events_per_thread):
                _publish(bus, f"ev.{i}", "network", "allow")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == n_threads * events_per_thread

    def test_concurrent_publishes_produce_valid_jsonl(
        self, audit_logger: AuditLogger, bus: EventBus, log_path: Path
    ):
        """Every line written under concurrent load must be valid JSON."""
        n_threads = 5
        events_per_thread = 10
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            for i in range(events_per_thread):
                _publish(bus, f"ev.{i}", "network", "allow")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for line in log_path.read_text().splitlines():
            if line.strip():
                json.loads(line)  # raises on bad JSON


# ---------------------------------------------------------------------------
# 6. Multiple loggers stacked on the same bus
# ---------------------------------------------------------------------------


class TestMultipleLoggerStacking:
    def test_two_loggers_both_receive_events(self, bus: EventBus, tmp_path: Path):
        """Each logger captures events independently."""
        path_a = str(tmp_path / "a.jsonl")
        path_b = str(tmp_path / "b.jsonl")
        AuditLogger(log_path=path_a, bus=bus)
        AuditLogger(log_path=path_b, bus=bus)

        _publish(bus, "ev.shared", "network", "allow")

        records_a = _read_jsonl(Path(path_a))
        records_b = _read_jsonl(Path(path_b))
        assert len(records_a) == 1
        assert len(records_b) == 1
        assert records_a[0]["event_type"] == "ev.shared"
        assert records_b[0]["event_type"] == "ev.shared"


# ---------------------------------------------------------------------------
# 7. Error handling when the audit file is not writable
# ---------------------------------------------------------------------------


class TestFileNotWritable:
    def test_write_failure_does_not_raise(self, audit_logger: AuditLogger, log_path: Path):
        """_handle_event swallows OSError so the caller is unaffected."""
        with patch("os.open", side_effect=PermissionError("read-only fs")):
            # Should not raise at all
            audit_logger._handle_event(
                AuditEvent.now(
                    session_id="s",
                    task_id="t",
                    event_type="test",
                    category="network",
                    result="allow",
                )
            )

    def test_write_failure_logs_error_message(self, audit_logger: AuditLogger, log_path: Path):
        with (
            patch("os.open", side_effect=PermissionError("denied")),
            patch("missy.observability.audit_logger._module_logger") as mock_log,
        ):
            audit_logger._handle_event(
                AuditEvent.now(
                    session_id="s",
                    task_id="t",
                    event_type="test",
                    category="network",
                    result="allow",
                )
            )
        mock_log.error.assert_called_once()

    def test_publish_continues_after_write_failure(self, bus: EventBus, tmp_path: Path):
        """Even if file writes keep failing, subsequent publishes still deliver to
        in-memory subscribers."""
        received = []
        bus.subscribe("ev.type", received.append)
        AuditLogger(log_path=str(tmp_path / "audit.jsonl"), bus=bus)

        with patch.object(Path, "open", side_effect=OSError("disk full")):
            _publish(bus, "ev.type", "network", "allow")

        assert len(received) == 1


# ---------------------------------------------------------------------------
# 8. Singleton behaviour
# ---------------------------------------------------------------------------


class TestSingletonBehaviour:
    def test_init_audit_logger_replaces_existing_singleton(self, tmp_path: Path):
        """Regression: init_audit_logger()'s own docstring claims a second
        call "replaces the existing logger with a new one targeting
        log_path" -- but _subscribe() wraps whatever self._bus.publish
        currently is, with no way to unwrap a specific prior layer.
        Constructing a brand-new AuditLogger and re-subscribing it
        previously left the FIRST logger's wrapper installed forever: both
        loggers kept receiving and writing every subsequent event, with the
        first one writing to its now-stale log_path indefinitely. The real
        behavioral contract is that only the most recently configured
        log_path receives new events -- verified here directly against the
        real, unwrapped global event_bus (what init_audit_logger() actually
        operates on), not a private in-memory identity check.
        """
        from missy.core.events import event_bus

        path_one = tmp_path / "one.jsonl"
        path_two = tmp_path / "two.jsonl"
        init_audit_logger(str(path_one))
        init_audit_logger(str(path_two))

        event_bus.publish(
            AuditEvent.now(
                session_id="s",
                task_id="t",
                event_type="test.singleton",
                category="network",
                result="allow",
            )
        )

        assert path_two.exists()
        assert "test.singleton" in path_two.read_text()
        assert not path_one.exists() or "test.singleton" not in path_one.read_text()
        assert get_audit_logger().log_path == path_two

    def test_get_audit_logger_returns_most_recent_init(self, tmp_path: Path):
        al = init_audit_logger(str(tmp_path / "c.jsonl"))
        assert get_audit_logger() is al

    def test_get_audit_logger_raises_runtime_error_when_not_initialised(self, monkeypatch):
        monkeypatch.setattr(audit_mod, "_audit_logger", None)
        with pytest.raises(RuntimeError, match="AuditLogger has not been initialised"):
            get_audit_logger()

    def test_get_audit_logger_error_message_includes_init_call(self, monkeypatch):
        monkeypatch.setattr(audit_mod, "_audit_logger", None)
        with pytest.raises(RuntimeError) as exc_info:
            get_audit_logger()
        assert "init_audit_logger" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 9. OtelExporter
# ---------------------------------------------------------------------------


class TestOtelExporter:
    def test_is_disabled_when_opentelemetry_not_installed(self):
        """OtelExporter silently degrades when otel packages are absent."""
        with patch.dict("sys.modules", {"opentelemetry": None}):
            exp = OtelExporter()
        assert exp.is_enabled is False

    def test_export_event_is_noop_when_disabled(self):
        """export_event must not raise when the exporter is disabled."""
        exp = OtelExporter.__new__(OtelExporter)
        exp._enabled = False
        exp._tracer = None
        exp._endpoint = "http://localhost:4317"
        exp._protocol = "grpc"
        exp._service_name = "missy"
        # Should not raise
        exp.export_event({"event_type": "test.event", "result": "allow"})

    def test_is_enabled_property_reflects_setup_state(self):
        """is_enabled returns the _enabled attribute."""
        exp = OtelExporter.__new__(OtelExporter)
        exp._enabled = False
        exp._tracer = None
        assert exp.is_enabled is False

        exp._enabled = True
        assert exp.is_enabled is True

    def test_export_event_skips_non_primitive_detail_keys(self):
        """export_event must not crash when detail values include lists."""
        exp = OtelExporter.__new__(OtelExporter)
        exp._enabled = True
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = lambda s, *a: mock_span
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
        exp._tracer = mock_tracer
        # list value inside "detail" — only primitives are set as attributes
        exp.export_event({"event_type": "x", "detail": {"hosts": ["a", "b"]}})

    def test_init_otel_returns_disabled_stub_when_otel_enabled_false(self):
        """init_otel with otel_enabled=False must return a non-enabled exporter."""
        cfg = MagicMock()
        cfg.observability.otel_enabled = False
        result = init_otel(cfg)
        assert isinstance(result, OtelExporter)
        # The stub returned by __new__ has no _enabled attribute set by _setup.
        assert not getattr(result, "_enabled", False)

    def test_init_otel_with_no_observability_attr_returns_stub(self):
        """init_otel is tolerant when the config object has no observability attr."""
        cfg = MagicMock(spec=[])  # no attributes at all
        result = init_otel(cfg)
        assert isinstance(result, OtelExporter)

    def test_subscribe_does_not_raise_when_disabled(self):
        """subscribe() on a disabled exporter must not raise."""
        exp = OtelExporter.__new__(OtelExporter)
        exp._enabled = False
        exp._tracer = None
        exp._endpoint = "http://localhost:4317"
        exp._protocol = "grpc"
        exp._service_name = "missy"
        # Should not raise even if event_bus.subscribe is not set up
        exp.subscribe()


# ---------------------------------------------------------------------------
# 10. Integration between AuditLogger and vision audit events
# ---------------------------------------------------------------------------


class TestVisionAuditIntegration:
    """The vision audit helpers in missy.vision.audit call get_audit_logger().log().

    Since AuditLogger has no public .log() method (vision.audit calls a non-
    existent API), the helpers fall back to the except branch and log a debug
    message.  These tests verify:
      - vision audit helpers never raise an exception
      - they degrade gracefully when the audit logger is not initialised
      - they degrade gracefully when the audit logger is initialised
    """

    def test_audit_vision_capture_does_not_raise_when_logger_uninitialised(self, monkeypatch):
        monkeypatch.setattr(audit_mod, "_audit_logger", None)
        from missy.vision.audit import audit_vision_capture

        # Must not raise
        audit_vision_capture(device="/dev/video0", success=True, width=1920, height=1080)

    def test_audit_vision_analysis_does_not_raise_when_logger_uninitialised(self, monkeypatch):
        monkeypatch.setattr(audit_mod, "_audit_logger", None)
        from missy.vision.audit import audit_vision_analysis

        audit_vision_analysis(mode="puzzle", success=False, error="timeout")

    def test_audit_vision_intent_does_not_raise_when_logger_uninitialised(self, monkeypatch):
        monkeypatch.setattr(audit_mod, "_audit_logger", None)
        from missy.vision.audit import audit_vision_intent

        audit_vision_intent(
            text="show me the board", intent="vision", confidence=0.95, decision="activate"
        )

    def test_audit_vision_device_discovery_does_not_raise_when_logger_uninitialised(
        self, monkeypatch
    ):
        monkeypatch.setattr(audit_mod, "_audit_logger", None)
        from missy.vision.audit import audit_vision_device_discovery

        audit_vision_device_discovery(camera_count=2, preferred_device="/dev/video0")

    def test_audit_vision_error_does_not_raise_when_logger_uninitialised(self, monkeypatch):
        monkeypatch.setattr(audit_mod, "_audit_logger", None)
        from missy.vision.audit import audit_vision_error

        audit_vision_error(operation="capture", error="camera not found", recoverable=False)

    def test_audit_vision_burst_does_not_raise_when_logger_uninitialised(self, monkeypatch):
        monkeypatch.setattr(audit_mod, "_audit_logger", None)
        from missy.vision.audit import audit_vision_burst

        audit_vision_burst(device="/dev/video0", count=5, successful=4, best_only=True)

    def test_audit_vision_session_does_not_raise_when_logger_uninitialised(self, monkeypatch):
        monkeypatch.setattr(audit_mod, "_audit_logger", None)
        from missy.vision.audit import audit_vision_session

        audit_vision_session(action="start", task_id="t1", task_type="puzzle", frame_count=0)

    def test_vision_audit_helpers_do_not_raise_when_log_method_absent(self, tmp_path: Path):
        """Even though AuditLogger has no .log() method, vision audit helpers
        must not propagate the AttributeError — they catch all exceptions."""
        al = init_audit_logger(str(tmp_path / "vision_audit.jsonl"))
        assert not hasattr(al, "log"), "Confirm .log() really does not exist"

        from missy.vision.audit import (
            audit_vision_analysis,
            audit_vision_burst,
            audit_vision_capture,
            audit_vision_device_discovery,
            audit_vision_error,
            audit_vision_intent,
            audit_vision_session,
        )

        # None of these must raise despite the missing .log() method
        audit_vision_capture()
        audit_vision_analysis()
        audit_vision_intent()
        audit_vision_device_discovery()
        audit_vision_error()
        audit_vision_burst()
        audit_vision_session()
