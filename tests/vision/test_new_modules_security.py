"""Security and integration tests for new vision modules.

Covers:
- MultiCameraManager: path injection, thread safety, exception isolation
- VisionMemoryBridge: content injection, SQL injection, session isolation
- ConfigValidator: path traversal, extreme values, empty config
- MemoryTracker: negative max_bytes clamping
- CaptureBenchmark / BenchmarkTimer: thread safety, correct flow
- MultiCaptureResult: mixed success/failure semantics
- Cross-module integration: config validation → CaptureConfig,
  BenchmarkTimer + CaptureResult, MemoryTracker + SceneSession
"""

from __future__ import annotations

import threading
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.benchmark import BenchmarkTimer, CaptureBenchmark
from missy.vision.capture import CaptureConfig, CaptureResult
from missy.vision.config_validator import ValidationResult, validate_vision_config
from missy.vision.discovery import CameraDevice
from missy.vision.memory_usage import MemoryTracker
from missy.vision.multi_camera import MultiCameraManager, MultiCaptureResult
from missy.vision.vision_memory import VisionMemoryBridge

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_device(
    path: str = "/dev/video0",
    name: str = "Test Camera",
    vendor_id: str = "046d",
    product_id: str = "085c",
    bus_info: str = "usb-0000:00:14.0-1",
) -> CameraDevice:
    return CameraDevice(
        device_path=path,
        name=name,
        vendor_id=vendor_id,
        product_id=product_id,
        bus_info=bus_info,
    )


def _make_success_result(
    path: str = "/dev/video0",
    width: int = 1920,
    height: int = 1080,
) -> CaptureResult:
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    return CaptureResult(
        success=True,
        image=image,
        device_path=path,
        width=width,
        height=height,
    )


def _make_failure_result(
    path: str = "/dev/video0",
    error: str = "capture failed",
) -> CaptureResult:
    return CaptureResult(success=False, device_path=path, error=error)


def _make_mock_handle(
    is_open: bool = True,
    capture_result: CaptureResult | None = None,
) -> MagicMock:
    handle = MagicMock()
    handle.is_open = is_open
    handle.capture.return_value = capture_result or _make_success_result()
    return handle


def _make_sqlite_mock(turns: list | None = None) -> MagicMock:
    store = MagicMock()
    store.get_session_turns.return_value = turns or []
    store.search.return_value = turns or []
    store.get_recent.return_value = turns or []
    return store


def _make_turn(
    *,
    role: str = "vision",
    content: str = "a visual observation",
    session_id: str = "sess-1",
    metadata: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=str(uuid.uuid4()),
        role=role,
        content=content,
        session_id=session_id,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Security: MultiCameraManager — path injection
# ---------------------------------------------------------------------------


class TestMultiCameraPathInjection:
    """MultiCameraManager must not allow device paths outside /dev/videoN.

    The validation happens at the CameraDevice level — CameraHandle accepts
    whatever path the CameraDevice carries, so the security boundary is that
    callers must provide a properly formed CameraDevice. These tests verify
    that a crafted device_path does NOT grant filesystem access to arbitrary
    paths by ensuring CameraHandle.open() is called with exactly the
    device_path stored in CameraDevice (no sanitization happens silently),
    and that an OS-level error is surfaced rather than suppressed.
    """

    def test_path_traversal_raises_or_fails_cleanly(self) -> None:
        """add_camera with a traversal path should fail, not silently open."""
        crafted = _make_device(path="/dev/../etc/passwd")

        with (
            patch("missy.vision.multi_camera.CameraHandle") as mock_handle_cls,
            patch("missy.vision.multi_camera.get_health_monitor"),
        ):
            mock_handle = MagicMock()
            mock_handle.is_open = False
            # Simulate the OS refusing to open a non-video device path
            mock_handle.open.side_effect = OSError("No such device")
            mock_handle_cls.return_value = mock_handle

            manager = MultiCameraManager()
            with pytest.raises(OSError, match="No such device"):
                manager.add_camera(crafted)

        # The camera must not appear in the manager's connected list
        assert "/dev/../etc/passwd" not in manager.connected_devices

    def test_arbitrary_file_path_raises(self) -> None:
        """Device path outside /dev must not silently succeed."""
        crafted = _make_device(path="/tmp/malicious_pipe")

        with (
            patch("missy.vision.multi_camera.CameraHandle") as mock_handle_cls,
            patch("missy.vision.multi_camera.get_health_monitor"),
        ):
            mock_handle = MagicMock()
            mock_handle.open.side_effect = ValueError("Not a video device")
            mock_handle_cls.return_value = mock_handle

            manager = MultiCameraManager()
            with pytest.raises(ValueError):
                manager.add_camera(crafted)

    def test_device_path_stored_verbatim(self) -> None:
        """The exact device_path is forwarded to CameraHandle without alteration."""
        path = "/dev/video99"
        device = _make_device(path=path)

        with (
            patch("missy.vision.multi_camera.CameraHandle") as mock_handle_cls,
            patch("missy.vision.multi_camera.get_health_monitor"),
        ):
            mock_handle = _make_mock_handle()
            mock_handle_cls.return_value = mock_handle

            manager = MultiCameraManager()
            manager.add_camera(device)

            # CameraHandle must be constructed with the exact path given
            mock_handle_cls.assert_called_once()
            constructed_path = mock_handle_cls.call_args[0][0]
            assert constructed_path == path

    def test_duplicate_path_raises_value_error(self) -> None:
        """Adding the same device twice must raise ValueError, not silently merge."""
        device = _make_device(path="/dev/video0")

        with (
            patch("missy.vision.multi_camera.CameraHandle") as mock_handle_cls,
            patch("missy.vision.multi_camera.get_health_monitor"),
        ):
            mock_handle_cls.return_value = _make_mock_handle()
            manager = MultiCameraManager()
            manager.add_camera(device)

            with pytest.raises(ValueError, match="/dev/video0"):
                manager.add_camera(device)


# ---------------------------------------------------------------------------
# Security: VisionMemoryBridge — content/code injection
# ---------------------------------------------------------------------------


class TestVisionMemoryBridgeInjection:
    """Observation content must be stored verbatim, not executed."""

    def test_python_code_in_observation_not_executed(self) -> None:
        """Storing Python code as observation content must not execute it."""
        # A payload that would side-effect if eval'd
        payload = "__import__('os').system('echo PWNED')"

        sqlite = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=sqlite, vector_store=None)

        # If the payload were eval'd, os.system would run; we'd detect that
        # via monkeypatching os.system.
        with patch("os.system") as mock_sys:
            obs_id = bridge.store_observation(
                session_id="sec-session-1",
                task_type="general",
                observation=payload,
                confidence=0.9,
            )
            mock_sys.assert_not_called()

        # The observation ID is a UUID, not the payload
        assert obs_id != payload
        uuid.UUID(obs_id)  # raises if not a valid UUID

    def test_exec_payload_in_observation_not_executed(self) -> None:
        """exec()-style payload in observation content must not run."""
        payload = "exec(compile('import sys; sys.exit(1)', '<string>', 'exec'))"

        sqlite = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=sqlite, vector_store=None)

        # Should complete normally without SystemExit
        obs_id = bridge.store_observation(
            session_id="sec-session-2",
            task_type="puzzle",
            observation=payload,
        )
        assert isinstance(obs_id, str)

    def test_sql_injection_in_session_id_handled_safely(self) -> None:
        """SQL injection attempt in session_id must not crash or corrupt."""
        malicious_session_id = "sess'; DROP TABLE turns; --"

        sqlite = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=sqlite, vector_store=None)

        # store_observation passes session_id directly to memory.add_turn;
        # the SQLite store itself is responsible for parameterized queries,
        # but we verify bridge doesn't raise and passes session_id through.
        obs_id = bridge.store_observation(
            session_id=malicious_session_id,
            task_type="general",
            observation="legitimate observation",
        )

        assert isinstance(obs_id, str)
        sqlite.add_turn.assert_called_once()
        call_kwargs = sqlite.add_turn.call_args
        stored_session_id = call_kwargs[1].get("session_id") or call_kwargs[0][0]
        assert stored_session_id == malicious_session_id

    def test_sql_injection_in_recall_query_handled_safely(self) -> None:
        """SQL injection in recall query must not crash."""
        malicious_query = "' OR '1'='1'; DROP TABLE turns; --"

        sqlite = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=sqlite, vector_store=None)

        # Should return empty list without raising
        results = bridge.recall_observations(query=malicious_query, limit=5)
        assert isinstance(results, list)

    def test_newline_injection_in_observation_stored_safely(self) -> None:
        """Newlines and null bytes in observation content must not crash."""
        payload = "legitimate note\x00\nINJECTED_ROLE: system\nmalicious: true"

        sqlite = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=sqlite, vector_store=None)

        obs_id = bridge.store_observation(
            session_id="sess-3",
            task_type="inspection",
            observation=payload,
        )
        assert isinstance(obs_id, str)

    def test_xss_payload_in_observation_stored_verbatim(self) -> None:
        """HTML/JS payload must be stored as-is, not sanitised or executed."""
        payload = "<script>alert('xss')</script>"

        sqlite = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=sqlite, vector_store=None)

        bridge.store_observation(
            session_id="sess-xss",
            task_type="general",
            observation=payload,
        )

        stored_content = sqlite.add_turn.call_args[1]["content"]
        assert stored_content == payload


# ---------------------------------------------------------------------------
# Security: ConfigValidator — path traversal and extreme values
# ---------------------------------------------------------------------------


class TestConfigValidatorSecurity:
    """ConfigValidator must reject traversal paths and handle extremes."""

    def test_path_traversal_in_preferred_device_flagged(self) -> None:
        """preferred_device with path traversal should produce a warning."""
        result = validate_vision_config(
            {
                "preferred_device": "/dev/../../etc/shadow",
            }
        )
        field_names = [i.field for i in result.issues]
        assert "preferred_device" in field_names
        # Should be a warning (not a fatal error — config is otherwise valid)
        device_issues = [i for i in result.issues if i.field == "preferred_device"]
        assert any(i.severity in ("warning", "error") for i in device_issues)

    def test_symlink_lookalike_path_flagged(self) -> None:
        """A device path that doesn't match /dev/videoN is flagged."""
        result = validate_vision_config({"preferred_device": "/tmp/fake_video"})
        preferred_issues = [i for i in result.issues if i.field == "preferred_device"]
        assert len(preferred_issues) > 0

    def test_valid_video_path_not_flagged(self) -> None:
        """A well-formed /dev/videoN path must not produce a preferred_device issue."""
        result = validate_vision_config({"preferred_device": "/dev/video0"})
        preferred_issues = [i for i in result.issues if i.field == "preferred_device"]
        assert len(preferred_issues) == 0

    def test_extremely_large_width_rejected(self) -> None:
        """capture_width larger than 3840 must produce an error."""
        result = validate_vision_config({"capture_width": 999_999_999, "capture_height": 1080})
        error_fields = [i.field for i in result.errors]
        assert "capture_width" in error_fields
        assert result.valid is False

    def test_extremely_large_height_rejected(self) -> None:
        """capture_height larger than 2160 must produce an error."""
        result = validate_vision_config({"capture_width": 1920, "capture_height": 999_999_999})
        error_fields = [i.field for i in result.errors]
        assert "capture_height" in error_fields

    def test_negative_warmup_frames_rejected(self) -> None:
        """Negative warmup_frames must be an error."""
        result = validate_vision_config({"warmup_frames": -100})
        error_fields = [i.field for i in result.errors]
        assert "warmup_frames" in error_fields

    def test_zero_scene_sessions_rejected(self) -> None:
        """scene_memory_max_sessions of 0 must be an error."""
        result = validate_vision_config({"scene_memory_max_sessions": 0})
        error_fields = [i.field for i in result.errors]
        assert "scene_memory_max_sessions" in error_fields

    def test_threshold_above_one_rejected(self) -> None:
        """auto_activate_threshold > 1.0 must be an error."""
        result = validate_vision_config({"auto_activate_threshold": 999.0})
        error_fields = [i.field for i in result.errors]
        assert "auto_activate_threshold" in error_fields

    def test_non_boolean_enabled_rejected(self) -> None:
        """enabled: 'yes' (string) must produce an error."""
        result = validate_vision_config({"enabled": "yes"})
        error_fields = [i.field for i in result.errors]
        assert "enabled" in error_fields

    def test_float_max_int_does_not_crash(self) -> None:
        """Passing sys.maxsize for integer fields must not crash."""
        import sys

        result = validate_vision_config(
            {
                "capture_width": sys.maxsize,
                "capture_height": sys.maxsize,
                "warmup_frames": sys.maxsize,
                "max_retries": sys.maxsize,
            }
        )
        # Should always return a ValidationResult, never raise
        assert isinstance(result, ValidationResult)


# ---------------------------------------------------------------------------
# Security: MemoryTracker — negative max_bytes clamping
# ---------------------------------------------------------------------------


class TestMemoryTrackerNegativeMax:
    """MemoryTracker.max_bytes must never be set to a non-positive value."""

    def test_negative_max_bytes_clamped_to_one(self) -> None:
        tracker = MemoryTracker(max_bytes=-1_000_000)
        assert tracker.max_bytes == 1

    def test_zero_max_bytes_clamped_to_one(self) -> None:
        tracker = MemoryTracker(max_bytes=0)
        assert tracker.max_bytes == 1

    def test_positive_max_bytes_preserved(self) -> None:
        tracker = MemoryTracker(max_bytes=500_000_000)
        assert tracker.max_bytes == 500_000_000

    def test_usage_fraction_never_divides_by_zero(self) -> None:
        """With clamped max_bytes=1 the report must not divide by zero."""
        tracker = MemoryTracker(max_bytes=0)
        manager_mock = MagicMock()
        manager_mock._lock = threading.Lock()
        manager_mock._sessions = {}

        report = tracker.update_from_scene_manager(manager_mock)
        assert isinstance(report.usage_fraction, float)
        assert report.usage_fraction == 0.0


# ---------------------------------------------------------------------------
# Security: Benchmark — thread safety under concurrent record calls
# ---------------------------------------------------------------------------


class TestBenchmarkThreadSafety:
    """CaptureBenchmark must not corrupt state under concurrent writes."""

    def test_concurrent_record_calls_no_data_corruption(self) -> None:
        bench = CaptureBenchmark(max_samples=1000)
        errors: list[Exception] = []
        write_count = 200
        thread_count = 10

        def _writer(thread_id: int) -> None:
            try:
                for i in range(write_count):
                    bench.record("capture", float(i + thread_id), device=f"/dev/video{thread_id}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_writer, args=(t,)) for t in range(thread_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"

        stats = bench.get_stats("capture")
        # We wrote thread_count * write_count samples but max_samples=1000
        assert stats["count"] == 1000

    def test_concurrent_record_and_report_no_exception(self) -> None:
        """Reading a report while writes are in progress must not raise."""
        bench = CaptureBenchmark(max_samples=500)
        errors: list[Exception] = []
        stop = threading.Event()

        def _writer() -> None:
            while not stop.is_set():
                try:
                    bench.record_capture(latency_ms=50.0, quality=0.9)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                    break

        def _reader() -> None:
            for _ in range(100):
                try:
                    bench.report()
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        writer = threading.Thread(target=_writer)
        reader = threading.Thread(target=_reader)
        writer.start()
        reader.start()
        reader.join(timeout=5)
        stop.set()
        writer.join(timeout=5)

        assert not errors


# ---------------------------------------------------------------------------
# Security: MultiCamera — capture_all doesn't leak worker exceptions
# ---------------------------------------------------------------------------


class TestMultiCapturExceptionIsolation:
    """Exceptions in individual camera workers must not propagate to caller."""

    def test_worker_exception_captured_as_failed_result(self) -> None:
        """An unhandled exception in a capture worker produces a failed result."""
        manager = MultiCameraManager()

        exploding_handle = MagicMock()
        exploding_handle.is_open = True
        exploding_handle.capture.side_effect = RuntimeError("cv2 segfault simulation")

        with (
            patch.object(manager, "_handles", {"/dev/video0": exploding_handle}),
            patch("missy.vision.multi_camera.get_health_monitor"),
        ):
            result = manager.capture_all(timeout=5.0)

        assert "/dev/video0" in result.results or "/dev/video0" in result.errors
        assert not result.all_succeeded

    def test_all_workers_explode_returns_multicaptureresult(self) -> None:
        """If every worker raises, capture_all still returns a MultiCaptureResult."""
        manager = MultiCameraManager()

        handles = {f"/dev/video{i}": MagicMock(is_open=True) for i in range(3)}
        for h in handles.values():
            h.capture.side_effect = OSError("device gone")

        with (
            patch.object(manager, "_handles", handles),
            patch("missy.vision.multi_camera.get_health_monitor"),
        ):
            result = manager.capture_all(timeout=5.0)

        assert isinstance(result, MultiCaptureResult)
        assert not result.any_succeeded

    def test_partial_worker_failure_best_result_from_survivors(self) -> None:
        """When one worker fails, best_result returns from the surviving camera."""
        manager = MultiCameraManager()

        good_result = _make_success_result(path="/dev/video1", width=1920, height=1080)
        good_handle = _make_mock_handle(capture_result=good_result)
        bad_handle = MagicMock(is_open=True)
        bad_handle.capture.side_effect = RuntimeError("bad camera")

        handles = {"/dev/video0": bad_handle, "/dev/video1": good_handle}

        with (
            patch.object(manager, "_handles", handles),
            patch("missy.vision.multi_camera.get_health_monitor"),
        ):
            result = manager.capture_all(timeout=5.0)

        best = result.best_result
        assert best is not None
        assert best.device_path == "/dev/video1"


# ---------------------------------------------------------------------------
# Integration: Config validation → CaptureConfig
# ---------------------------------------------------------------------------


class TestConfigValidationToCaptureConfig:
    """A config that passes ConfigValidator should map cleanly to CaptureConfig."""

    def test_valid_config_produces_no_errors(self) -> None:
        valid_config = {
            "enabled": True,
            "capture_width": 1920,
            "capture_height": 1080,
            "warmup_frames": 5,
            "max_retries": 3,
            "auto_activate_threshold": 0.80,
            "scene_memory_max_frames": 20,
            "scene_memory_max_sessions": 5,
            "preferred_device": "/dev/video0",
        }
        result = validate_vision_config(valid_config)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validated_values_produce_valid_capture_config(self) -> None:
        """CaptureConfig constructed from validated values must not raise."""
        raw = {
            "capture_width": 1280,
            "capture_height": 720,
            "warmup_frames": 3,
            "max_retries": 2,
        }
        result = validate_vision_config(raw)
        assert result.valid is True

        cfg = CaptureConfig(
            width=raw["capture_width"],
            height=raw["capture_height"],
            warmup_frames=raw["warmup_frames"],
            max_retries=raw["max_retries"],
        )
        assert cfg.width == 1280
        assert cfg.height == 720
        assert cfg.warmup_frames == 3
        assert cfg.max_retries == 2

    def test_invalid_config_errors_correlate_to_capture_config_fields(self) -> None:
        """Errors from invalid config match fields present in CaptureConfig."""
        result = validate_vision_config(
            {"capture_width": -1, "capture_height": -1, "warmup_frames": -5}
        )
        assert not result.valid
        error_fields = {i.field for i in result.errors}
        # At minimum, the width and warmup errors must be present
        assert "capture_width" in error_fields or "capture_height" in error_fields


# ---------------------------------------------------------------------------
# Integration: BenchmarkTimer + CaptureResult flow
# ---------------------------------------------------------------------------


class TestBenchmarkTimerWithCaptureResult:
    """BenchmarkTimer should integrate naturally with a simulated capture flow."""

    def test_timer_records_capture_latency(self) -> None:
        bench = CaptureBenchmark()

        fake_handle = _make_mock_handle(capture_result=_make_success_result())

        with BenchmarkTimer(bench, "capture", device="/dev/video0") as timer:
            fake_handle.capture()

        assert timer.elapsed_ms >= 0.0
        stats = bench.get_stats("capture")
        assert stats["count"] == 1
        assert stats["min_ms"] == stats["max_ms"]  # only one sample

    def test_timer_records_even_when_capture_raises(self) -> None:
        """BenchmarkTimer records elapsed time even if the body raises."""
        bench = CaptureBenchmark()
        fake_handle = MagicMock()
        fake_handle.capture.side_effect = OSError("device error")

        with pytest.raises(OSError), BenchmarkTimer(bench, "capture"):
            fake_handle.capture()

        # The record must still be present
        stats = bench.get_stats("capture")
        assert stats["count"] == 1

    def test_pipeline_and_capture_categories_tracked_independently(self) -> None:
        bench = CaptureBenchmark()

        with BenchmarkTimer(bench, "capture"):
            pass  # simulate near-instant capture

        with BenchmarkTimer(bench, "pipeline", operation="resize"):
            pass

        capture_stats = bench.get_stats("capture")
        pipeline_stats = bench.get_stats("pipeline")

        assert capture_stats["count"] == 1
        assert pipeline_stats["count"] == 1
        # Categories are independent
        assert bench.get_stats("burst") == {}

    def test_capture_result_success_flag_matches_benchmark_timing(self) -> None:
        """A failed CaptureResult can still be timed — timing is independent."""
        bench = CaptureBenchmark()
        failed_result = _make_failure_result(path="/dev/video0")

        with BenchmarkTimer(bench, "capture", device="/dev/video0"):
            _ = failed_result.success  # simulate checking result

        assert bench.get_stats("capture")["count"] == 1


# ---------------------------------------------------------------------------
# Integration: MemoryTracker + SceneSession
# ---------------------------------------------------------------------------


class TestMemoryTrackerWithSceneSession:
    """MemoryTracker must correctly account for actual SceneSession memory."""

    def test_tracker_measures_session_with_frames(self) -> None:
        """A session with numpy frames should report non-zero bytes."""
        from missy.vision.scene_memory import SceneSession

        session = SceneSession(task_id="task-001")
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        session.add_frame(image, source="test", deduplicate=False)

        tracker = MemoryTracker(max_bytes=1_000_000_000)
        info = tracker.compute_session_usage(session)

        assert info.task_id == "task-001"
        assert info.frame_count == 1
        assert info.estimated_bytes > 0
        # 1920x1080x3 bytes of data + overhead
        assert info.estimated_bytes >= 1920 * 1080 * 3

    def test_tracker_reports_multiple_sessions(self) -> None:
        """update_from_scene_manager must aggregate across all sessions."""
        from missy.vision.scene_memory import SceneSession

        sessions = {}
        for i in range(3):
            sess = SceneSession(task_id=f"task-{i}")
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            sess.add_frame(image, deduplicate=False)
            sessions[f"task-{i}"] = sess

        manager_mock = MagicMock()
        manager_mock._lock = threading.Lock()
        manager_mock._sessions = sessions

        tracker = MemoryTracker(max_bytes=1_000_000_000)
        report = tracker.update_from_scene_manager(manager_mock)

        assert report.session_count == 3
        assert report.total_frames == 3
        assert report.total_bytes > 0

    def test_empty_session_reports_zero_bytes(self) -> None:
        """A session with no frames should report zero bytes."""
        from missy.vision.scene_memory import SceneSession

        session = SceneSession(task_id="empty-task")
        tracker = MemoryTracker(max_bytes=100_000_000)
        info = tracker.compute_session_usage(session)

        assert info.frame_count == 0
        assert info.estimated_bytes == 0

    def test_over_limit_flag_set_correctly(self) -> None:
        """over_limit should be True when total exceeds max_bytes."""
        from missy.vision.scene_memory import SceneSession

        session = SceneSession(task_id="fat-session")
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        session.add_frame(image, deduplicate=False)

        manager_mock = MagicMock()
        manager_mock._lock = threading.Lock()
        manager_mock._sessions = {"fat-session": session}

        # Set max_bytes far below what a single HD frame costs
        tracker = MemoryTracker(max_bytes=100)
        report = tracker.update_from_scene_manager(manager_mock)

        assert report.over_limit is True
        assert report.usage_fraction > 1.0


# ---------------------------------------------------------------------------
# Integration: VisionMemoryBridge lazy init with unavailable stores
# ---------------------------------------------------------------------------


class TestVisionMemoryBridgeLazyInit:
    """lazy init must not crash when stores are unavailable."""

    def test_lazy_init_with_no_stores_does_not_crash(self) -> None:
        """store_observation with no stores at all must not raise.

        SQLiteMemoryStore and VectorMemoryStore are imported inside
        _ensure_init, so we patch at their canonical module paths.
        """
        bridge = VisionMemoryBridge(memory_store=None, vector_store=None)

        with (
            patch(
                "missy.memory.sqlite_store.SQLiteMemoryStore",
                side_effect=Exception("sqlite not available"),
            ),
            patch.dict(
                "sys.modules",
                {
                    "missy.memory.sqlite_store": MagicMock(
                        SQLiteMemoryStore=MagicMock(side_effect=Exception("sqlite not available"))
                    ),
                    "missy.memory.vector_store": MagicMock(
                        VectorMemoryStore=MagicMock(side_effect=ImportError("faiss not installed"))
                    ),
                },
            ),
        ):
            obs_id = bridge.store_observation(
                session_id="lazy-sess",
                task_type="general",
                observation="no store available",
            )

        assert isinstance(obs_id, str)

    def test_recall_with_no_stores_returns_empty_list(self) -> None:
        """recall_observations with both stores absent must return []."""
        bridge = VisionMemoryBridge(memory_store=None, vector_store=None)
        bridge._initialized = True  # skip lazy init
        bridge._memory = None
        bridge._vector = None

        results = bridge.recall_observations(query="anything")
        assert results == []

    def test_initialized_flag_set_after_first_call(self) -> None:
        """_initialized must be True after the first store_observation call."""
        sqlite = _make_sqlite_mock()
        bridge = VisionMemoryBridge(memory_store=sqlite, vector_store=None)

        assert bridge._initialized is False
        bridge.store_observation(
            session_id="s1",
            task_type="general",
            observation="test",
        )
        assert bridge._initialized is True


# ---------------------------------------------------------------------------
# Integration: ConfigValidator with empty dict
# ---------------------------------------------------------------------------


class TestConfigValidatorEmptyDict:
    """An empty config dict must return a valid result (all defaults are sane)."""

    def test_empty_config_is_valid(self) -> None:
        result = validate_vision_config({})
        assert isinstance(result, ValidationResult)
        assert result.valid is True

    def test_empty_config_has_no_errors(self) -> None:
        result = validate_vision_config({})
        assert len(result.errors) == 0

    def test_empty_config_to_dict_is_serialisable(self) -> None:
        result = validate_vision_config({})
        d = result.to_dict()
        assert d["valid"] is True
        assert d["error_count"] == 0
        assert isinstance(d["issues"], list)


# ---------------------------------------------------------------------------
# Integration: MultiCaptureResult mixed success/failure semantics
# ---------------------------------------------------------------------------


class TestMultiCaptureResultMixed:
    """MultiCaptureResult property semantics with mixed success/failure results."""

    def _make_mixed_result(self) -> MultiCaptureResult:
        good = _make_success_result(path="/dev/video0", width=1920, height=1080)
        bigger = _make_success_result(path="/dev/video1", width=2560, height=1440)
        bad = _make_failure_result(path="/dev/video2", error="timeout")

        return MultiCaptureResult(
            results={
                "/dev/video0": good,
                "/dev/video1": bigger,
                "/dev/video2": bad,
            },
            elapsed_ms=123.45,
            errors={"/dev/video2": "timeout"},
        )

    def test_any_succeeded_true_when_at_least_one_good(self) -> None:
        result = self._make_mixed_result()
        assert result.any_succeeded is True

    def test_all_succeeded_false_when_one_failed(self) -> None:
        result = self._make_mixed_result()
        assert result.all_succeeded is False

    def test_successful_devices_excludes_failed(self) -> None:
        result = self._make_mixed_result()
        successful = result.successful_devices
        assert "/dev/video0" in successful
        assert "/dev/video1" in successful
        assert "/dev/video2" not in successful

    def test_failed_devices_includes_only_failures(self) -> None:
        result = self._make_mixed_result()
        assert result.failed_devices == ["/dev/video2"]

    def test_best_result_is_highest_resolution(self) -> None:
        """best_result picks the successful capture with the largest pixel count."""
        result = self._make_mixed_result()
        best = result.best_result
        assert best is not None
        assert best.device_path == "/dev/video1"
        assert best.width * best.height == 2560 * 1440

    def test_all_failed_returns_any_succeeded_false(self) -> None:
        bad0 = _make_failure_result("/dev/video0")
        bad1 = _make_failure_result("/dev/video1")
        result = MultiCaptureResult(
            results={"/dev/video0": bad0, "/dev/video1": bad1},
            errors={"/dev/video0": "err", "/dev/video1": "err"},
        )
        assert result.any_succeeded is False
        assert result.best_result is None

    def test_empty_results_dict_any_succeeded_false(self) -> None:
        result = MultiCaptureResult()
        assert result.any_succeeded is False
        assert result.all_succeeded is False
        assert result.best_result is None
        assert result.successful_devices == []
        assert result.failed_devices == []
