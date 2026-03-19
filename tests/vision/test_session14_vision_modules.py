"""Session 14: Edge case tests for vision support modules.

Covers:
- shutdown: idempotency, reset, atexit registration, exception handling
- vision_memory: store/recall edge cases, metadata protection, thread safety
- config_validator: boundary values, type errors, non-standard resolutions
- memory_usage: frame estimation, zero-size frames, tracker limits
- benchmark: throughput calculation, timer context manager, reset, stats edge cases
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shutdown tests
# ---------------------------------------------------------------------------


class TestVisionShutdown:
    """Tests for the vision shutdown module."""

    def setup_method(self):
        from missy.vision.shutdown import reset_shutdown_state
        reset_shutdown_state()

    def test_shutdown_returns_summary(self):
        from missy.vision.shutdown import vision_shutdown
        with patch("missy.vision.scene_memory.get_scene_manager") as mock_sm, \
             patch("missy.vision.health_monitor.get_health_monitor") as mock_hm, \
             patch("missy.vision.audit.audit_vision_session"):
            mock_sm.return_value.list_sessions.return_value = []
            mock_sm.return_value.close_all = MagicMock()
            mock_hm.return_value._persist_path = None
            result = vision_shutdown()
            assert result["status"] == "shutdown"
            assert isinstance(result["steps"], list)

    def test_shutdown_idempotent(self):
        from missy.vision.shutdown import vision_shutdown
        with patch("missy.vision.scene_memory.get_scene_manager") as mock_sm, \
             patch("missy.vision.health_monitor.get_health_monitor") as mock_hm, \
             patch("missy.vision.audit.audit_vision_session"):
            mock_sm.return_value.list_sessions.return_value = []
            mock_sm.return_value.close_all = MagicMock()
            mock_hm.return_value._persist_path = None
            result1 = vision_shutdown()
            result2 = vision_shutdown()
            assert result1["status"] == "shutdown"
            assert result2["status"] == "already_shutdown"

    def test_reset_shutdown_state(self):
        from missy.vision.shutdown import reset_shutdown_state, vision_shutdown
        with patch("missy.vision.scene_memory.get_scene_manager") as mock_sm, \
             patch("missy.vision.health_monitor.get_health_monitor") as mock_hm, \
             patch("missy.vision.audit.audit_vision_session"):
            mock_sm.return_value.list_sessions.return_value = []
            mock_sm.return_value.close_all = MagicMock()
            mock_hm.return_value._persist_path = None
            vision_shutdown()
            reset_shutdown_state()
            result = vision_shutdown()
            assert result["status"] == "shutdown"

    def test_shutdown_scene_manager_exception(self):
        from missy.vision.shutdown import vision_shutdown
        with patch("missy.vision.scene_memory.get_scene_manager", side_effect=RuntimeError("fail")), \
             patch("missy.vision.health_monitor.get_health_monitor") as mock_hm, \
             patch("missy.vision.audit.audit_vision_session"):
            mock_hm.return_value._persist_path = None
            result = vision_shutdown()
            assert any("fail" in s for s in result["steps"])

    def test_shutdown_health_monitor_exception(self):
        from missy.vision.shutdown import vision_shutdown
        with patch("missy.vision.scene_memory.get_scene_manager") as mock_sm, \
             patch("missy.vision.health_monitor.get_health_monitor", side_effect=RuntimeError("hm fail")), \
             patch("missy.vision.audit.audit_vision_session"):
            mock_sm.return_value.list_sessions.return_value = []
            mock_sm.return_value.close_all = MagicMock()
            result = vision_shutdown()
            assert any("hm fail" in s for s in result["steps"])

    def test_shutdown_with_active_sessions(self):
        from missy.vision.shutdown import vision_shutdown
        with patch("missy.vision.scene_memory.get_scene_manager") as mock_sm, \
             patch("missy.vision.health_monitor.get_health_monitor") as mock_hm, \
             patch("missy.vision.audit.audit_vision_session"):
            mock_sm.return_value.list_sessions.return_value = [
                {"active": True}, {"active": True}, {"active": False}
            ]
            mock_sm.return_value.close_all = MagicMock()
            mock_hm.return_value._persist_path = "/tmp/test.db"
            mock_hm.return_value.save = MagicMock()
            result = vision_shutdown()
            assert any("2 active" in s for s in result["steps"])

    def test_register_shutdown_hook(self):
        from missy.vision.shutdown import register_shutdown_hook
        # Should not raise
        register_shutdown_hook()

    def test_shutdown_concurrent_calls(self):
        """Concurrent shutdown calls should not double-execute."""
        from missy.vision.shutdown import vision_shutdown
        results = []

        with patch("missy.vision.scene_memory.get_scene_manager") as mock_sm, \
             patch("missy.vision.health_monitor.get_health_monitor") as mock_hm, \
             patch("missy.vision.audit.audit_vision_session"):
            mock_sm.return_value.list_sessions.return_value = []
            mock_sm.return_value.close_all = MagicMock()
            mock_hm.return_value._persist_path = None

            threads = []
            for _ in range(5):
                t = threading.Thread(target=lambda: results.append(vision_shutdown()))
                threads.append(t)
                t.start()
            for t in threads:
                t.join(timeout=5)

        shutdown_count = sum(1 for r in results if r["status"] == "shutdown")
        already_count = sum(1 for r in results if r["status"] == "already_shutdown")
        assert shutdown_count == 1
        assert already_count == 4


# ---------------------------------------------------------------------------
# VisionMemoryBridge tests
# ---------------------------------------------------------------------------


class TestVisionMemoryBridge:
    """Tests for VisionMemoryBridge."""

    def test_store_observation_returns_uuid(self):
        from missy.vision.vision_memory import VisionMemoryBridge
        memory = MagicMock()
        bridge = VisionMemoryBridge(memory_store=memory)
        obs_id = bridge.store_observation(
            session_id="s1", task_type="puzzle",
            observation="Found 3 edge pieces",
        )
        assert len(obs_id) == 36  # UUID format
        memory.add_turn.assert_called_once()

    def test_store_observation_metadata_protection(self):
        """Reserved keys in metadata should be filtered out."""
        from missy.vision.vision_memory import VisionMemoryBridge
        memory = MagicMock()
        bridge = VisionMemoryBridge(memory_store=memory)
        bridge.store_observation(
            session_id="s1", task_type="puzzle",
            observation="test",
            metadata={"session_id": "EVIL", "custom_key": "safe"},
        )
        call_kwargs = memory.add_turn.call_args[1]
        meta = call_kwargs["metadata"]
        assert meta["session_id"] == "s1"  # Not overridden
        assert meta["custom_key"] == "safe"

    def test_store_observation_memory_failure(self):
        """Memory store failure should not crash."""
        from missy.vision.vision_memory import VisionMemoryBridge
        memory = MagicMock()
        memory.add_turn.side_effect = RuntimeError("DB error")
        bridge = VisionMemoryBridge(memory_store=memory)
        obs_id = bridge.store_observation(
            session_id="s1", task_type="general", observation="test",
        )
        assert obs_id  # Still returns a UUID

    def test_store_observation_with_vector_store(self):
        """Vector store should receive indexed entry."""
        from missy.vision.vision_memory import VisionMemoryBridge
        memory = MagicMock()
        vector = MagicMock()
        bridge = VisionMemoryBridge(memory_store=memory, vector_store=vector)
        bridge.store_observation(
            session_id="s1", task_type="painting",
            observation="Nice brushwork",
        )
        vector.add.assert_called_once()
        text_arg = vector.add.call_args[0][0]
        assert "[painting]" in text_arg

    def test_store_observation_vector_failure(self):
        """Vector store failure should not crash."""
        from missy.vision.vision_memory import VisionMemoryBridge
        memory = MagicMock()
        vector = MagicMock()
        vector.add.side_effect = RuntimeError("FAISS error")
        bridge = VisionMemoryBridge(memory_store=memory, vector_store=vector)
        obs_id = bridge.store_observation(
            session_id="s1", task_type="general", observation="test",
        )
        assert obs_id

    def test_recall_empty_results(self):
        from missy.vision.vision_memory import VisionMemoryBridge
        memory = MagicMock()
        memory.get_recent.return_value = []
        bridge = VisionMemoryBridge(memory_store=memory)
        results = bridge.recall_observations()
        assert results == []

    def test_recall_with_query_vector_search(self):
        from missy.vision.vision_memory import VisionMemoryBridge
        memory = MagicMock()
        vector = MagicMock()
        vector.search.return_value = [
            (0.95, {"task_type": "puzzle", "observation": "found piece"}),
        ]
        bridge = VisionMemoryBridge(memory_store=memory, vector_store=vector)
        results = bridge.recall_observations(query="puzzle piece")
        assert len(results) == 1
        assert results[0]["relevance_score"] == 0.95

    def test_recall_vector_search_with_type_filter(self):
        from missy.vision.vision_memory import VisionMemoryBridge
        memory = MagicMock()
        vector = MagicMock()
        vector.search.return_value = [
            (0.9, {"task_type": "painting", "observation": "nice"}),
            (0.8, {"task_type": "puzzle", "observation": "edge"}),
        ]
        bridge = VisionMemoryBridge(memory_store=memory, vector_store=vector)
        results = bridge.recall_observations(query="test", task_type="puzzle")
        assert len(results) == 1
        assert results[0]["task_type"] == "puzzle"

    def test_recall_vector_failure_falls_back(self):
        """Vector search failure should fall back to SQLite."""
        from missy.vision.vision_memory import VisionMemoryBridge
        memory = MagicMock()
        memory.search.return_value = []
        vector = MagicMock()
        vector.search.side_effect = RuntimeError("FAISS error")
        bridge = VisionMemoryBridge(memory_store=memory, vector_store=vector)
        results = bridge.recall_observations(query="test")
        assert results == []
        memory.search.assert_called_once()

    def test_get_session_context_empty(self):
        from missy.vision.vision_memory import VisionMemoryBridge
        memory = MagicMock()
        memory.get_session_turns.return_value = []
        bridge = VisionMemoryBridge(memory_store=memory)
        context = bridge.get_session_context("s1")
        assert context == ""

    def test_get_session_context_with_observations(self):
        from missy.vision.vision_memory import VisionMemoryBridge
        bridge = VisionMemoryBridge()
        # Mock recall_observations
        bridge.recall_observations = MagicMock(return_value=[
            {"task_type": "puzzle", "observation": "Found edge", "confidence": 0.9, "timestamp": "T1"},
        ])
        context = bridge.get_session_context("s1")
        assert "Visual Observations" in context
        assert "puzzle" in context
        assert "Found edge" in context

    def test_clear_session_no_memory(self):
        from missy.vision.vision_memory import VisionMemoryBridge
        bridge = VisionMemoryBridge(memory_store=None)
        bridge._initialized = True
        count = bridge.clear_session("s1")
        assert count == 0

    def test_store_observation_none_metadata(self):
        """None metadata should be handled."""
        from missy.vision.vision_memory import VisionMemoryBridge
        memory = MagicMock()
        bridge = VisionMemoryBridge(memory_store=memory)
        obs_id = bridge.store_observation(
            session_id="s1", task_type="general",
            observation="test", metadata=None,
        )
        assert obs_id

    def test_store_observation_confidence_clamping(self):
        """Confidence values should be stored as-is (no clamping)."""
        from missy.vision.vision_memory import VisionMemoryBridge
        memory = MagicMock()
        bridge = VisionMemoryBridge(memory_store=memory)
        bridge.store_observation(
            session_id="s1", task_type="general",
            observation="test", confidence=1.5,
        )
        meta = memory.add_turn.call_args[1]["metadata"]
        assert meta["confidence"] == 1.5


# ---------------------------------------------------------------------------
# Config validator tests
# ---------------------------------------------------------------------------


class TestConfigValidatorEdgeCases:
    """Edge cases in vision config validation."""

    def test_empty_config(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({})
        assert result.valid is True
        assert len(result.issues) == 0

    def test_valid_standard_config(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({
            "enabled": True,
            "capture_width": 1920,
            "capture_height": 1080,
            "warmup_frames": 5,
            "max_retries": 3,
            "auto_activate_threshold": 0.80,
        })
        assert result.valid is True

    def test_enabled_non_bool(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"enabled": "yes"})
        assert any(i.field == "enabled" and i.severity == "error" for i in result.issues)

    def test_width_below_minimum(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"capture_width": 50})
        assert any(i.field == "capture_width" and i.severity == "error" for i in result.issues)

    def test_width_above_maximum(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"capture_width": 8000})
        assert any(i.field == "capture_width" and i.severity == "error" for i in result.issues)

    def test_width_non_integer(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"capture_width": "wide"})
        assert any(i.field == "capture_width" and i.severity == "error" for i in result.issues)

    def test_non_standard_resolution_warning(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({
            "capture_width": 1000,
            "capture_height": 750,
        })
        assert any(i.field == "capture_width" and i.severity == "warning" for i in result.issues)

    def test_height_below_minimum(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"capture_height": 50})
        assert any(i.field == "capture_height" and i.severity == "error" for i in result.issues)

    def test_warmup_negative(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"warmup_frames": -1})
        assert any(i.field == "warmup_frames" and i.severity == "error" for i in result.issues)

    def test_warmup_excessive(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"warmup_frames": 50})
        assert any(i.field == "warmup_frames" and i.severity == "warning" for i in result.issues)

    def test_retries_zero(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"max_retries": 0})
        assert any(i.field == "max_retries" and i.severity == "error" for i in result.issues)

    def test_retries_excessive(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"max_retries": 50})
        assert any(i.field == "max_retries" and i.severity == "warning" for i in result.issues)

    def test_threshold_below_zero(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"auto_activate_threshold": -0.1})
        assert any(i.field == "auto_activate_threshold" and i.severity == "error" for i in result.issues)

    def test_threshold_above_one(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"auto_activate_threshold": 1.5})
        assert any(i.field == "auto_activate_threshold" and i.severity == "error" for i in result.issues)

    def test_threshold_low_warning(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"auto_activate_threshold": 0.3})
        assert any(i.field == "auto_activate_threshold" and i.severity == "warning" for i in result.issues)

    def test_scene_frames_zero(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"scene_memory_max_frames": 0})
        assert any(i.field == "scene_memory_max_frames" and i.severity == "error" for i in result.issues)

    def test_scene_frames_excessive(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"scene_memory_max_frames": 200})
        assert any(i.field == "scene_memory_max_frames" and i.severity == "warning" for i in result.issues)

    def test_scene_sessions_zero(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"scene_memory_max_sessions": 0})
        assert any(i.field == "scene_memory_max_sessions" and i.severity == "error" for i in result.issues)

    def test_scene_sessions_excessive(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"scene_memory_max_sessions": 50})
        assert any(i.field == "scene_memory_max_sessions" and i.severity == "warning" for i in result.issues)

    def test_device_path_invalid_format(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"preferred_device": "/dev/sda1"})
        assert any(i.field == "preferred_device" and i.severity == "warning" for i in result.issues)

    def test_device_path_valid(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"preferred_device": "/dev/video0"})
        assert not any(i.field == "preferred_device" for i in result.issues)

    def test_device_path_empty_is_ok(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"preferred_device": ""})
        assert not any(i.field == "preferred_device" for i in result.issues)

    def test_validation_result_to_dict(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"capture_width": -1, "warmup_frames": 100})
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "valid" in d
        assert "issues" in d
        assert d["error_count"] >= 1

    def test_validation_result_errors_property(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"capture_width": -1})
        assert len(result.errors) >= 1
        assert all(i.severity == "error" for i in result.errors)

    def test_validation_result_warnings_property(self):
        from missy.vision.config_validator import validate_vision_config
        result = validate_vision_config({"warmup_frames": 50})
        assert len(result.warnings) >= 1
        assert all(i.severity == "warning" for i in result.warnings)

    def test_all_valid_resolutions(self):
        """All common resolutions should not trigger warning."""
        from missy.vision.config_validator import validate_vision_config
        for w, h in [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]:
            result = validate_vision_config({"capture_width": w, "capture_height": h})
            assert not any(
                i.field == "capture_width" and i.severity == "warning"
                for i in result.issues
            ), f"Resolution {w}x{h} should be valid"


# ---------------------------------------------------------------------------
# Memory usage tracker tests
# ---------------------------------------------------------------------------


class TestMemoryUsageEdgeCases:
    """Edge cases in memory usage tracking."""

    def test_estimate_frame_bytes_none(self):
        from missy.vision.memory_usage import estimate_frame_bytes
        assert estimate_frame_bytes(None) == 0

    def test_estimate_frame_bytes_numpy_like(self):
        from missy.vision.memory_usage import estimate_frame_bytes

        class FakeArray:
            nbytes = 6220800  # 1920*1080*3

        result = estimate_frame_bytes(FakeArray())
        assert result > 6220800

    def test_estimate_frame_bytes_plain_object(self):
        from missy.vision.memory_usage import estimate_frame_bytes
        result = estimate_frame_bytes("hello")
        assert result > 0

    def test_tracker_max_bytes_minimum(self):
        from missy.vision.memory_usage import MemoryTracker
        tracker = MemoryTracker(max_bytes=0)
        assert tracker.max_bytes == 1  # Clamped to 1

    def test_tracker_should_evict_no_report(self):
        from missy.vision.memory_usage import MemoryTracker
        tracker = MemoryTracker()
        assert tracker.should_evict() is False

    def test_tracker_report_none_initially(self):
        from missy.vision.memory_usage import MemoryTracker
        tracker = MemoryTracker()
        assert tracker.report() is None

    def test_session_memory_info_mb(self):
        from missy.vision.memory_usage import SessionMemoryInfo
        info = SessionMemoryInfo(task_id="t1", frame_count=1, estimated_bytes=1048576, active=True)
        assert info.estimated_mb == 1.0

    def test_memory_report_to_dict(self):
        from missy.vision.memory_usage import MemoryReport, SessionMemoryInfo
        info = SessionMemoryInfo(task_id="t1", frame_count=2, estimated_bytes=2000000, active=True)
        report = MemoryReport(
            total_bytes=2000000,
            total_frames=2,
            session_count=1,
            active_sessions=1,
            sessions=[info],
            limit_bytes=500000000,
            usage_fraction=0.004,
            over_limit=False,
        )
        d = report.to_dict()
        assert d["total_frames"] == 2
        assert d["over_limit"] is False
        assert len(d["sessions"]) == 1

    def test_module_singleton(self):
        from missy.vision.memory_usage import get_memory_tracker
        t1 = get_memory_tracker()
        t2 = get_memory_tracker()
        assert t1 is t2


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


class TestBenchmarkEdgeCases:
    """Edge cases in CaptureBenchmark."""

    def test_stats_empty_category(self):
        from missy.vision.benchmark import CaptureBenchmark
        bench = CaptureBenchmark()
        assert bench.get_stats("nonexistent") == {}

    def test_stats_single_sample(self):
        from missy.vision.benchmark import CaptureBenchmark
        bench = CaptureBenchmark()
        bench.record("capture", 50.0)
        stats = bench.get_stats("capture")
        assert stats["count"] == 1
        assert stats["min_ms"] == 50.0
        assert stats["max_ms"] == 50.0
        assert stats["stddev_ms"] == 0.0

    def test_stats_p95_p99_small_sample(self):
        """P95/P99 with <20/<100 samples should use max."""
        from missy.vision.benchmark import CaptureBenchmark
        bench = CaptureBenchmark()
        for i in range(10):
            bench.record("capture", float(i))
        stats = bench.get_stats("capture")
        assert stats["p95_ms"] == stats["max_ms"]
        assert stats["p99_ms"] == stats["max_ms"]

    def test_throughput_no_samples(self):
        from missy.vision.benchmark import CaptureBenchmark
        bench = CaptureBenchmark()
        assert bench.throughput("capture") == 0.0

    def test_throughput_single_sample(self):
        from missy.vision.benchmark import CaptureBenchmark
        bench = CaptureBenchmark()
        bench.record("capture", 10.0)
        assert bench.throughput("capture") == 0.0  # Need >=2

    def test_reset_specific_category(self):
        from missy.vision.benchmark import CaptureBenchmark
        bench = CaptureBenchmark()
        bench.record("capture", 10.0)
        bench.record("pipeline", 5.0)
        bench.reset("capture")
        assert bench.get_stats("capture") == {}
        assert bench.get_stats("pipeline")["count"] == 1

    def test_reset_all(self):
        from missy.vision.benchmark import CaptureBenchmark
        bench = CaptureBenchmark()
        bench.record("capture", 10.0)
        bench.record("pipeline", 5.0)
        bench.reset()
        assert bench.get_stats("capture") == {}
        assert bench.get_stats("pipeline") == {}

    def test_report_structure(self):
        from missy.vision.benchmark import CaptureBenchmark
        bench = CaptureBenchmark()
        bench.record("capture", 10.0)
        report = bench.report()
        assert "uptime_seconds" in report
        assert "categories" in report
        assert "capture" in report["categories"]

    def test_report_empty(self):
        from missy.vision.benchmark import CaptureBenchmark
        bench = CaptureBenchmark()
        report = bench.report()
        assert report["categories"] == {}

    def test_max_samples_clamping(self):
        from missy.vision.benchmark import CaptureBenchmark
        bench = CaptureBenchmark(max_samples=5)
        assert bench._max_samples == 10  # Clamped to minimum 10

    def test_timer_context_manager(self):
        from missy.vision.benchmark import BenchmarkTimer, CaptureBenchmark
        bench = CaptureBenchmark()
        with BenchmarkTimer(bench, "test_op", device="/dev/video0") as timer:
            time.sleep(0.01)
        assert timer.elapsed_ms > 0
        stats = bench.get_stats("test_op")
        assert stats["count"] == 1

    def test_convenience_methods(self):
        from missy.vision.benchmark import CaptureBenchmark
        bench = CaptureBenchmark()
        bench.record_capture(10.0, quality=0.9, device="/dev/video0")
        bench.record_pipeline(5.0, operation="resize")
        bench.record_burst(100.0, frame_count=10)
        bench.record_save(2.0, format="png")
        assert bench.get_stats("capture")["count"] == 1
        assert bench.get_stats("pipeline")["count"] == 1
        assert bench.get_stats("burst")["count"] == 1
        assert bench.get_stats("save")["count"] == 1

    def test_module_singleton(self):
        from missy.vision.benchmark import get_benchmark
        b1 = get_benchmark()
        b2 = get_benchmark()
        assert b1 is b2

    def test_concurrent_recording(self):
        """Concurrent recording should be thread-safe."""
        from missy.vision.benchmark import CaptureBenchmark
        bench = CaptureBenchmark()

        def record_many():
            for i in range(50):
                bench.record("concurrent", float(i))

        threads = [threading.Thread(target=record_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        stats = bench.get_stats("concurrent")
        assert stats["count"] == 250

    def test_throughput_zero_time_span(self):
        """Throughput with samples at same timestamp should return 0."""
        from missy.vision.benchmark import BenchmarkSample, CaptureBenchmark
        from collections import deque

        bench = CaptureBenchmark()
        now = time.time()
        samples = deque([
            BenchmarkSample(timestamp=now, category="test", duration_ms=1.0),
            BenchmarkSample(timestamp=now, category="test", duration_ms=2.0),
        ])
        with bench._lock:
            bench._samples["test"] = samples
        assert bench.throughput("test") == 0.0
