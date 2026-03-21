"""Tests for session 25: coverage gap improvements."""

from __future__ import annotations

import inspect
import logging
import time
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Incus tools: invalid action paths
# ---------------------------------------------------------------------------


class TestIncusInvalidActions:
    """Cover invalid action error paths in incus tools."""

    def test_snapshot_invalid_action(self):
        from missy.tools.builtin.incus_tools import IncusSnapshotTool

        tool = IncusSnapshotTool()
        result = tool.execute(instance="test", action="invalid_action")
        assert not result.success
        assert "Invalid action" in result.error

    def test_config_invalid_action(self):
        from missy.tools.builtin.incus_tools import IncusConfigTool

        tool = IncusConfigTool()
        result = tool.execute(instance="test", action="invalid_action")
        assert not result.success
        assert "Invalid action" in result.error

    def test_image_invalid_action(self):
        from missy.tools.builtin.incus_tools import IncusImageTool

        tool = IncusImageTool()
        result = tool.execute(action="invalid_action")
        assert not result.success
        assert "Invalid action" in result.error

    def test_network_invalid_action(self):
        from missy.tools.builtin.incus_tools import IncusNetworkTool

        tool = IncusNetworkTool()
        result = tool.execute(action="invalid_action")
        assert not result.success
        assert "Invalid action" in result.error

    def test_storage_invalid_action(self):
        from missy.tools.builtin.incus_tools import IncusStorageTool

        tool = IncusStorageTool()
        result = tool.execute(action="invalid_action")
        assert not result.success
        assert "Invalid action" in result.error

    def test_profile_invalid_action(self):
        from missy.tools.builtin.incus_tools import IncusProfileTool

        tool = IncusProfileTool()
        result = tool.execute(action="invalid_action")
        assert not result.success
        assert "Invalid action" in result.error

    def test_project_invalid_action(self):
        from missy.tools.builtin.incus_tools import IncusProjectTool

        tool = IncusProjectTool()
        result = tool.execute(action="invalid_action")
        assert not result.success
        assert "Invalid action" in result.error

    def test_device_invalid_action(self):
        from missy.tools.builtin.incus_tools import IncusDeviceTool

        tool = IncusDeviceTool()
        result = tool.execute(instance="test", action="invalid_action")
        assert not result.success
        assert "Invalid action" in result.error


# ---------------------------------------------------------------------------
# Incus tools: edge cases for specific actions
# ---------------------------------------------------------------------------


class TestIncusToolEdgeCases:
    """Edge cases in incus tool execute paths."""

    def test_snapshot_restore_without_name(self):
        """Snapshot restore without snapshot_name uses instance-only command."""
        from missy.tools.builtin.incus_tools import IncusSnapshotTool

        tool = IncusSnapshotTool()
        with patch("missy.tools.builtin.incus_tools._run_incus") as mock_run:
            mock_run.return_value = MagicMock(success=True)
            tool.execute(instance="myvm", action="restore")
            args = mock_run.call_args[0][0]
            assert args == ["snapshot", "restore", "myvm"]

    def test_snapshot_restore_with_name(self):
        from missy.tools.builtin.incus_tools import IncusSnapshotTool

        tool = IncusSnapshotTool()
        with patch("missy.tools.builtin.incus_tools._run_incus") as mock_run:
            mock_run.return_value = MagicMock(success=True)
            tool.execute(instance="myvm", action="restore", snapshot_name="snap1")
            args = mock_run.call_args[0][0]
            assert args == ["snapshot", "restore", "myvm", "snap1"]

    def test_snapshot_create_without_name_fails(self):
        from missy.tools.builtin.incus_tools import IncusSnapshotTool

        tool = IncusSnapshotTool()
        result = tool.execute(instance="myvm", action="create")
        assert not result.success
        assert "snapshot_name is required" in result.error

    def test_snapshot_delete_without_name_fails(self):
        from missy.tools.builtin.incus_tools import IncusSnapshotTool

        tool = IncusSnapshotTool()
        result = tool.execute(instance="myvm", action="delete")
        assert not result.success
        assert "snapshot_name is required" in result.error

    def test_config_unset_without_key_fails(self):
        from missy.tools.builtin.incus_tools import IncusConfigTool

        tool = IncusConfigTool()
        result = tool.execute(instance="myvm", action="unset")
        assert not result.success
        assert "key is required" in result.error

    def test_network_detach_without_name_fails(self):
        from missy.tools.builtin.incus_tools import IncusNetworkTool

        tool = IncusNetworkTool()
        result = tool.execute(action="detach")
        assert not result.success
        assert "name is required" in result.error

    def test_network_detach_single_word_fails(self):
        from missy.tools.builtin.incus_tools import IncusNetworkTool

        tool = IncusNetworkTool()
        result = tool.execute(action="detach", name="onlynetwork")
        assert not result.success
        assert "network_name instance_name" in result.error

    def test_image_alias_without_required_fields_fails(self):
        from missy.tools.builtin.incus_tools import IncusImageTool

        tool = IncusImageTool()
        result = tool.execute(action="alias")
        assert not result.success
        assert "image and alias are required" in result.error

    def test_snapshot_create_with_stateful(self):
        from missy.tools.builtin.incus_tools import IncusSnapshotTool

        tool = IncusSnapshotTool()
        with patch("missy.tools.builtin.incus_tools._run_incus") as mock_run:
            mock_run.return_value = MagicMock(success=True)
            tool.execute(instance="myvm", action="create", snapshot_name="s1", stateful=True)
            args = mock_run.call_args[0][0]
            assert "--stateful" in args

    def test_snapshot_with_project(self):
        from missy.tools.builtin.incus_tools import IncusSnapshotTool

        tool = IncusSnapshotTool()
        with patch("missy.tools.builtin.incus_tools._run_incus") as mock_run:
            mock_run.return_value = MagicMock(success=True)
            tool.execute(instance="myvm", action="create", snapshot_name="s1", project="myproj")
            args = mock_run.call_args[0][0]
            assert "--project" in args
            assert "myproj" in args

    def test_storage_create_without_pool_fails(self):
        from missy.tools.builtin.incus_tools import IncusStorageTool

        tool = IncusStorageTool()
        result = tool.execute(action="create")
        assert not result.success
        assert "pool" in result.error.lower() or "required" in result.error.lower()

    def test_storage_delete_without_pool_fails(self):
        from missy.tools.builtin.incus_tools import IncusStorageTool

        tool = IncusStorageTool()
        result = tool.execute(action="delete")
        assert not result.success
        assert "pool" in result.error.lower() or "required" in result.error.lower()


# ---------------------------------------------------------------------------
# Code evolution: _revert_diffs with failure logging
# ---------------------------------------------------------------------------


class TestCodeEvolutionRevertDiffs:
    """_revert_diffs logs warning with exc_info on failure."""

    def test_revert_logs_on_git_failure(self, caplog):
        from missy.agent.code_evolution import CodeEvolutionManager

        # Check the class actually has _revert_diffs
        assert hasattr(CodeEvolutionManager, "_revert_diffs")

        engine = CodeEvolutionManager.__new__(CodeEvolutionManager)
        engine._git = MagicMock(side_effect=RuntimeError("git broken"))

        # Need to create a FileDiff-like object
        diff = MagicMock()
        diff.file_path = "test.py"

        with caplog.at_level(logging.WARNING, logger="missy.agent.code_evolution"):
            engine._revert_diffs([diff])

        assert any("Failed to revert" in msg for msg in caplog.messages)
        assert any(
            record.exc_info for record in caplog.records if "Failed to revert" in record.message
        )


# ---------------------------------------------------------------------------
# Checkpoint: named constants
# ---------------------------------------------------------------------------


class TestCheckpointConstants:
    """Checkpoint module uses named constants instead of magic numbers."""

    def test_constants_defined(self):
        from missy.agent.checkpoint import _RESTART_THRESHOLD_SECS, _RESUME_THRESHOLD_SECS

        assert _RESUME_THRESHOLD_SECS == 3600
        assert _RESTART_THRESHOLD_SECS == 86400

    def test_classify_uses_resume_threshold(self):
        from missy.agent.checkpoint import _RESUME_THRESHOLD_SECS, CheckpointManager

        cm = CheckpointManager.__new__(CheckpointManager)
        checkpoint = {"created_at": time.time() - _RESUME_THRESHOLD_SECS + 10}
        assert cm.classify(checkpoint) == "resume"

    def test_classify_uses_restart_threshold(self):
        from missy.agent.checkpoint import _RESTART_THRESHOLD_SECS, CheckpointManager

        cm = CheckpointManager.__new__(CheckpointManager)
        checkpoint = {"created_at": time.time() - _RESTART_THRESHOLD_SECS + 10}
        assert cm.classify(checkpoint) == "restart"

    def test_classify_abandon_after_restart_threshold(self):
        from missy.agent.checkpoint import _RESTART_THRESHOLD_SECS, CheckpointManager

        cm = CheckpointManager.__new__(CheckpointManager)
        checkpoint = {"created_at": time.time() - _RESTART_THRESHOLD_SECS - 100}
        assert cm.classify(checkpoint) == "abandon"

    def test_abandon_old_default_matches_constant(self):
        from missy.agent.checkpoint import _RESTART_THRESHOLD_SECS, CheckpointManager

        sig = inspect.signature(CheckpointManager.abandon_old)
        default = sig.parameters["max_age_seconds"].default
        assert default == _RESTART_THRESHOLD_SECS


# ---------------------------------------------------------------------------
# Checkpoint: cleanup lifecycle
# ---------------------------------------------------------------------------


class TestCheckpointCleanupLifecycle:
    """Checkpoint cleanup uses constants correctly."""

    def test_cleanup_deletes_old_records(self, tmp_path):
        from missy.agent.checkpoint import CheckpointManager

        db_path = str(tmp_path / "test.db")
        cm = CheckpointManager(db_path=db_path)

        cid = cm.create("sid", "tid", "prompt")
        cm.complete(cid)

        conn = cm._connect()
        conn.execute(
            "UPDATE checkpoints SET updated_at = ? WHERE id = ?",
            (time.time() - 8 * 86400, cid),
        )
        conn.commit()

        deleted = cm.cleanup(older_than_days=7)
        assert deleted == 1

    def test_cleanup_keeps_recent_records(self, tmp_path):
        from missy.agent.checkpoint import CheckpointManager

        db_path = str(tmp_path / "test.db")
        cm = CheckpointManager(db_path=db_path)

        cid = cm.create("sid", "tid", "prompt")
        cm.complete(cid)

        deleted = cm.cleanup(older_than_days=7)
        assert deleted == 0


# ---------------------------------------------------------------------------
# Cost tracker: edge cases (correct API)
# ---------------------------------------------------------------------------


class TestCostTrackerEdgeCases:
    """Cost tracker boundary conditions."""

    def test_record_zero_tokens(self):
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record("test-model", 0, 0)
        assert tracker.total_prompt_tokens == 0
        assert tracker.total_completion_tokens == 0

    def test_no_budget_never_raises(self):
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker(max_spend_usd=0.0)
        tracker.record("claude-sonnet-4-6", 1000, 1000)
        # Should not raise since budget is 0 (disabled)
        tracker.check_budget()

    def test_budget_exceeded_raises(self):
        from missy.agent.cost_tracker import BudgetExceededError, CostTracker

        tracker = CostTracker(max_spend_usd=0.0001)
        tracker.record("claude-sonnet-4-6", 1000000, 1000000)
        with pytest.raises(BudgetExceededError):
            tracker.check_budget()

    def test_summary_keys(self):
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record("claude-sonnet-4-6", 100, 50)
        summary = tracker.get_summary()
        assert "total_prompt_tokens" in summary
        assert "total_completion_tokens" in summary
        assert "total_cost_usd" in summary


# ---------------------------------------------------------------------------
# Circuit breaker: state transitions
# ---------------------------------------------------------------------------


class TestCircuitBreakerTransitions:
    """Circuit breaker state transition tests via call()."""

    def test_success_keeps_closed(self):
        from missy.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test", threshold=2, base_timeout=0.01)
        result = cb.call(lambda: "ok")
        assert result == "ok"
        assert cb.state.value == "closed"

    def test_failures_open_circuit(self):
        from missy.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test", threshold=2, base_timeout=0.01)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        assert cb.state.value == "open"

    def test_open_circuit_rejects_calls(self):
        from missy.agent.circuit_breaker import CircuitBreaker
        from missy.core.exceptions import MissyError

        cb = CircuitBreaker(name="test", threshold=1, base_timeout=60)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        with pytest.raises(MissyError, match="OPEN"):
            cb.call(lambda: "should not reach")

    def test_half_open_success_closes(self):
        from missy.agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test", threshold=1, base_timeout=0.01)
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert cb.state.value == "open"

        time.sleep(0.02)
        # After timeout, state property should return half_open
        assert cb.state.value == "half_open"

        # Successful call should close
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state.value == "closed"


# ---------------------------------------------------------------------------
# Scheduler parser: validation edge cases
# ---------------------------------------------------------------------------


class TestSchedulerParserValidation:
    """Scheduler parser input validation."""

    def test_zero_interval_raises(self):
        from missy.scheduler.parser import parse_schedule

        with pytest.raises((ValueError, Exception)):
            parse_schedule("every 0 minutes")

    def test_invalid_hour_raises(self):
        from missy.scheduler.parser import parse_schedule

        with pytest.raises(ValueError, match="Hour"):
            parse_schedule("daily at 25:00")

    def test_invalid_minute_raises(self):
        from missy.scheduler.parser import parse_schedule

        with pytest.raises(ValueError, match="Minute"):
            parse_schedule("daily at 12:75")

    def test_weekly_with_on_keyword(self):
        from missy.scheduler.parser import parse_schedule

        result = parse_schedule("weekly on Monday at 09:00")
        assert result is not None

    def test_every_5_minutes(self):
        from missy.scheduler.parser import parse_schedule

        result = parse_schedule("every 5 minutes")
        assert result is not None

    def test_unrecognized_format_raises(self):
        from missy.scheduler.parser import parse_schedule

        with pytest.raises(ValueError, match="Unrecognised"):
            parse_schedule("whenever I feel like it")


# ---------------------------------------------------------------------------
# Memory store: add_turn and search roundtrip
# ---------------------------------------------------------------------------


class TestMemoryStoreRoundtrip:
    """MemoryStore add_turn and search."""

    def test_add_and_get_recent(self, tmp_path):
        from missy.memory.store import MemoryStore

        store_path = str(tmp_path / "test_memory.json")
        store = MemoryStore(store_path=store_path)

        store.add_turn("session-1", "user", "Hello world")
        store.add_turn("session-1", "assistant", "Hi!")
        recent = store.get_recent_turns(limit=5)
        assert len(recent) == 2
        assert recent[0].content == "Hello world"

    def test_add_empty_content(self, tmp_path):
        from missy.memory.store import MemoryStore

        store_path = str(tmp_path / "test_memory.json")
        store = MemoryStore(store_path=store_path)

        store.add_turn("session-1", "user", "")
        # Should not raise

    def test_search_returns_matching(self, tmp_path):
        from missy.memory.store import MemoryStore

        store_path = str(tmp_path / "test_memory.json")
        store = MemoryStore(store_path=store_path)

        store.add_turn("session-1", "user", "Hello world")
        results = store.search("Hello")
        assert len(results) >= 1

    def test_search_no_results(self, tmp_path):
        from missy.memory.store import MemoryStore

        store_path = str(tmp_path / "test_memory.json")
        store = MemoryStore(store_path=store_path)

        results = store.search("nonexistent query xyz")
        assert results == []


# ---------------------------------------------------------------------------
# Audit logger: event handling
# ---------------------------------------------------------------------------


class TestAuditLoggerEventHandling:
    """Audit logger write and read events."""

    def test_write_and_read_events(self, tmp_path):
        from missy.core.events import AuditEvent, EventBus
        from missy.observability.audit_logger import AuditLogger

        log_path = str(tmp_path / "audit.jsonl")
        bus = EventBus()
        logger = AuditLogger(log_path=log_path, bus=bus)

        event = AuditEvent.now(
            session_id="sid",
            task_id="tid",
            event_type="test",
            category="test",
            result="allow",
            detail={"msg": "hello"},
        )
        bus.publish(event)

        events = logger.get_recent_events(limit=10)
        assert len(events) >= 1

    def test_unicode_in_events(self, tmp_path):
        from missy.core.events import AuditEvent, EventBus
        from missy.observability.audit_logger import AuditLogger

        log_path = str(tmp_path / "audit.jsonl")
        bus = EventBus()
        logger = AuditLogger(log_path=log_path, bus=bus)

        event = AuditEvent.now(
            session_id="sid",
            task_id="tid",
            event_type="test",
            category="test",
            result="allow",
            detail={"msg": "Unicode: \u00e9\u00e0\u00fc \u4e16\u754c"},
        )
        bus.publish(event)

        events = logger.get_recent_events(limit=1)
        assert len(events) == 1

    def test_empty_audit_log(self, tmp_path):
        from missy.core.events import EventBus
        from missy.observability.audit_logger import AuditLogger

        log_path = str(tmp_path / "audit.jsonl")
        bus = EventBus()
        logger = AuditLogger(log_path=log_path, bus=bus)

        events = logger.get_recent_events(limit=10)
        assert events == []


# ---------------------------------------------------------------------------
# Policy engine: basic smoke tests
# ---------------------------------------------------------------------------


class TestPolicyEngineSmokeTests:
    """Policy engine basic operation with explicit initialization."""

    def _make_config(self):
        from missy.config.settings import (
            FilesystemPolicy,
            MissyConfig,
            NetworkPolicy,
            PluginPolicy,
            ShellPolicy,
        )

        return MissyConfig(
            network=NetworkPolicy(),  # default_deny=True
            filesystem=FilesystemPolicy(),
            shell=ShellPolicy(),  # enabled=False
            plugins=PluginPolicy(),
            providers={},
            workspace_path="/tmp/missy-test",
            audit_log_path="/tmp/missy-test/audit.jsonl",
        )

    def test_default_deny_blocks_unknown_host(self):
        from missy.core.exceptions import PolicyViolationError
        from missy.policy.engine import PolicyEngine

        config = self._make_config()
        engine = PolicyEngine(config)
        with pytest.raises(PolicyViolationError):
            engine.check_network("evil.example.com", "sid", "tid")

    def test_shell_policy_blocks_by_default(self):
        from missy.core.exceptions import PolicyViolationError
        from missy.policy.engine import PolicyEngine

        config = self._make_config()
        engine = PolicyEngine(config)
        with pytest.raises(PolicyViolationError):
            engine.check_shell("rm -rf /", "sid", "tid")

    def test_filesystem_write_denied_by_default(self):
        from missy.core.exceptions import PolicyViolationError
        from missy.policy.engine import PolicyEngine

        config = self._make_config()
        engine = PolicyEngine(config)
        with pytest.raises(PolicyViolationError):
            engine.check_write("/etc/passwd", "sid", "tid")
