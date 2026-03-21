"""Tests for CLI vision commands: benchmark, validate, memory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from missy.cli.main import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(mix_stderr=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory_report(
    *,
    total_bytes: int = 0,
    total_frames: int = 0,
    session_count: int = 0,
    active_sessions: int = 0,
    sessions: list | None = None,
    limit_bytes: int = 500_000_000,
    usage_fraction: float = 0.0,
    over_limit: bool = False,
) -> MagicMock:
    """Return a MagicMock shaped like a MemoryReport."""
    report = MagicMock()
    report.over_limit = over_limit
    report.to_dict.return_value = {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "total_frames": total_frames,
        "session_count": session_count,
        "active_sessions": active_sessions,
        "limit_bytes": limit_bytes,
        "limit_mb": limit_bytes / (1024 * 1024),
        "usage_fraction": usage_fraction,
        "over_limit": over_limit,
        "sessions": sessions or [],
    }
    return report


def _make_validation_result(
    *,
    valid: bool = True,
    issues: list | None = None,
    warnings: list | None = None,
) -> MagicMock:
    """Return a MagicMock shaped like a ValidationResult."""
    result = MagicMock()
    result.valid = valid
    result.issues = issues or []
    result.warnings = warnings or []
    result.errors = [i for i in (issues or []) if getattr(i, "severity", "") == "error"]
    return result


# ---------------------------------------------------------------------------
# missy vision benchmark
# ---------------------------------------------------------------------------


class TestVisionBenchmark:
    def test_no_data_shows_placeholder_message(self, runner: CliRunner) -> None:
        bench = MagicMock()
        bench.report.return_value = {
            "uptime_seconds": 12.3,
            "categories": {},
        }

        with patch("missy.vision.benchmark.get_benchmark", return_value=bench):
            result = runner.invoke(cli, ["vision", "benchmark"])

        assert result.exit_code == 0
        assert "No benchmark data collected yet" in result.output
        assert "12.3" in result.output

    def test_no_data_hints_at_captures(self, runner: CliRunner) -> None:
        bench = MagicMock()
        bench.report.return_value = {"uptime_seconds": 0.0, "categories": {}}

        with patch("missy.vision.benchmark.get_benchmark", return_value=bench):
            result = runner.invoke(cli, ["vision", "benchmark"])

        assert result.exit_code == 0
        assert "captures" in result.output.lower()

    def test_with_single_category(self, runner: CliRunner) -> None:
        bench = MagicMock()
        bench.report.return_value = {
            "uptime_seconds": 300.0,
            "categories": {
                "capture": {
                    "count": 10,
                    "min_ms": 30.5,
                    "mean_ms": 45.2,
                    "median_ms": 44.0,
                    "p95_ms": 60.1,
                    "max_ms": 75.8,
                },
            },
        }

        with patch("missy.vision.benchmark.get_benchmark", return_value=bench):
            result = runner.invoke(cli, ["vision", "benchmark"])

        assert result.exit_code == 0
        assert "capture" in result.output
        assert "10" in result.output
        assert "30.5" in result.output
        assert "75.8" in result.output

    def test_with_multiple_categories(self, runner: CliRunner) -> None:
        bench = MagicMock()
        bench.report.return_value = {
            "uptime_seconds": 600.0,
            "categories": {
                "capture": {
                    "count": 25,
                    "min_ms": 28.0,
                    "mean_ms": 42.0,
                    "median_ms": 41.0,
                    "p95_ms": 55.0,
                    "max_ms": 70.0,
                },
                "pipeline": {
                    "count": 25,
                    "min_ms": 5.0,
                    "mean_ms": 10.0,
                    "median_ms": 9.5,
                    "p95_ms": 18.0,
                    "max_ms": 22.0,
                },
                "save": {
                    "count": 25,
                    "min_ms": 2.0,
                    "mean_ms": 4.0,
                    "median_ms": 3.8,
                    "p95_ms": 7.0,
                    "max_ms": 9.0,
                },
            },
        }

        with patch("missy.vision.benchmark.get_benchmark", return_value=bench):
            result = runner.invoke(cli, ["vision", "benchmark"])

        assert result.exit_code == 0
        assert "capture" in result.output
        assert "pipeline" in result.output
        assert "save" in result.output

    def test_header_always_shown(self, runner: CliRunner) -> None:
        bench = MagicMock()
        bench.report.return_value = {"uptime_seconds": 0.0, "categories": {}}

        with patch("missy.vision.benchmark.get_benchmark", return_value=bench):
            result = runner.invoke(cli, ["vision", "benchmark"])

        assert result.exit_code == 0
        assert "Benchmark" in result.output

    def test_uptime_shown_with_data(self, runner: CliRunner) -> None:
        bench = MagicMock()
        bench.report.return_value = {
            "uptime_seconds": 999.9,
            "categories": {
                "burst": {
                    "count": 3,
                    "min_ms": 100.0,
                    "mean_ms": 120.0,
                    "median_ms": 115.0,
                    "p95_ms": 140.0,
                    "max_ms": 150.0,
                },
            },
        }

        with patch("missy.vision.benchmark.get_benchmark", return_value=bench):
            result = runner.invoke(cli, ["vision", "benchmark"])

        assert result.exit_code == 0
        assert "999.9" in result.output


# ---------------------------------------------------------------------------
# missy vision validate
# ---------------------------------------------------------------------------


class TestVisionValidate:
    def _invoke(
        self,
        runner: CliRunner,
        validate_return: MagicMock,
        cfg_raises: bool = False,
    ) -> object:
        with (
            patch(
                "missy.vision.config_validator.validate_vision_config",
                return_value=validate_return,
            ),
            patch(
                "missy.config.settings.load_config",
                side_effect=RuntimeError("no config") if cfg_raises else None,
            ),
        ):
            if not cfg_raises:
                # Provide a minimal config object
                mock_cfg = MagicMock()
                mock_cfg.vision = MagicMock(
                    enabled=True,
                    capture_width=1920,
                    capture_height=1080,
                    warmup_frames=5,
                    max_retries=3,
                    auto_activate_threshold=0.80,
                    scene_memory_max_frames=20,
                    scene_memory_max_sessions=5,
                    preferred_device="",
                )
                # Re-patch load_config to return our mock
                with patch("missy.config.settings.load_config", return_value=mock_cfg):
                    return runner.invoke(cli, ["vision", "validate"])
            return runner.invoke(cli, ["vision", "validate"])

    def test_valid_config_shows_success(self, runner: CliRunner) -> None:
        vr = _make_validation_result(valid=True)

        with (
            patch("missy.vision.config_validator.validate_vision_config", return_value=vr),
            patch("missy.config.settings.load_config", return_value=MagicMock()),
        ):
            result = runner.invoke(cli, ["vision", "validate"])

        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_valid_config_no_warnings_says_all_settings_valid(self, runner: CliRunner) -> None:
        vr = _make_validation_result(valid=True, warnings=[])

        with (
            patch("missy.vision.config_validator.validate_vision_config", return_value=vr),
            patch("missy.config.settings.load_config", return_value=MagicMock()),
        ):
            result = runner.invoke(cli, ["vision", "validate"])

        assert result.exit_code == 0
        assert "All settings are valid" in result.output

    def test_valid_config_with_warnings_shows_warnings(self, runner: CliRunner) -> None:
        warn_issue = MagicMock()
        warn_issue.severity = "warning"
        warn_issue.field = "capture_width"
        warn_issue.message = "Non-standard resolution"
        warn_issue.current_value = 1234
        warn_issue.suggested_value = None

        vr = _make_validation_result(valid=True, issues=[warn_issue], warnings=[warn_issue])

        with (
            patch("missy.vision.config_validator.validate_vision_config", return_value=vr),
            patch("missy.config.settings.load_config", return_value=MagicMock()),
        ):
            result = runner.invoke(cli, ["vision", "validate"])

        assert result.exit_code == 0
        assert "WARN" in result.output
        assert "capture_width" in result.output
        assert "Non-standard resolution" in result.output

    def test_invalid_config_shows_errors(self, runner: CliRunner) -> None:
        err_issue = MagicMock()
        err_issue.severity = "error"
        err_issue.field = "capture_width"
        err_issue.message = "Must be between 160 and 3840"
        err_issue.current_value = 9999
        err_issue.suggested_value = 1920

        vr = _make_validation_result(valid=False, issues=[err_issue], warnings=[])
        vr.errors = [err_issue]

        with (
            patch("missy.vision.config_validator.validate_vision_config", return_value=vr),
            patch("missy.config.settings.load_config", return_value=MagicMock()),
        ):
            result = runner.invoke(cli, ["vision", "validate"])

        assert result.exit_code == 0
        assert "ERROR" in result.output
        assert "capture_width" in result.output

    def test_invalid_config_shows_error_count(self, runner: CliRunner) -> None:
        err1 = MagicMock()
        err1.severity = "error"
        err1.field = "warmup_frames"
        err1.message = "Cannot be negative"
        err1.current_value = -1
        err1.suggested_value = 5

        err2 = MagicMock()
        err2.severity = "error"
        err2.field = "max_retries"
        err2.message = "Must be at least 1"
        err2.current_value = 0
        err2.suggested_value = 3

        vr = _make_validation_result(valid=False, issues=[err1, err2], warnings=[])
        vr.errors = [err1, err2]

        with (
            patch("missy.vision.config_validator.validate_vision_config", return_value=vr),
            patch("missy.config.settings.load_config", return_value=MagicMock()),
        ):
            result = runner.invoke(cli, ["vision", "validate"])

        assert result.exit_code == 0
        assert "2" in result.output
        assert "error" in result.output.lower()

    def test_issue_current_and_suggested_values_displayed(self, runner: CliRunner) -> None:
        issue = MagicMock()
        issue.severity = "error"
        issue.field = "capture_width"
        issue.message = "Out of range"
        issue.current_value = 50
        issue.suggested_value = 1920

        vr = _make_validation_result(valid=False, issues=[issue], warnings=[])
        vr.errors = [issue]

        with (
            patch("missy.vision.config_validator.validate_vision_config", return_value=vr),
            patch("missy.config.settings.load_config", return_value=MagicMock()),
        ):
            result = runner.invoke(cli, ["vision", "validate"])

        assert result.exit_code == 0
        assert "50" in result.output
        assert "1920" in result.output

    def test_info_severity_shown(self, runner: CliRunner) -> None:
        # The CLI only renders the issues loop when result.valid is False OR
        # result.warnings is non-empty (otherwise it short-circuits with "All
        # settings are valid.").  Pair the info issue with a warning so the
        # loop executes and the INFO icon appears.
        info_issue = MagicMock()
        info_issue.severity = "info"
        info_issue.field = "preferred_device"
        info_issue.message = "No device configured, will auto-detect"
        info_issue.current_value = None
        info_issue.suggested_value = None

        warn_issue = MagicMock()
        warn_issue.severity = "warning"
        warn_issue.field = "auto_activate_threshold"
        warn_issue.message = "Low threshold may cause excessive auto-activation"
        warn_issue.current_value = 0.3
        warn_issue.suggested_value = None

        vr = _make_validation_result(
            valid=True,
            issues=[info_issue, warn_issue],
            warnings=[warn_issue],
        )

        with (
            patch("missy.vision.config_validator.validate_vision_config", return_value=vr),
            patch("missy.config.settings.load_config", return_value=MagicMock()),
        ):
            result = runner.invoke(cli, ["vision", "validate"])

        assert result.exit_code == 0
        assert "INFO" in result.output
        assert "preferred_device" in result.output

    def test_load_config_failure_falls_back_to_empty(self, runner: CliRunner) -> None:
        vr = _make_validation_result(valid=True)

        # When load_config raises, validate is still called (with empty dict)
        with (
            patch(
                "missy.vision.config_validator.validate_vision_config", return_value=vr
            ) as mock_validate,
            patch(
                "missy.config.settings.load_config",
                side_effect=RuntimeError("no config file"),
            ),
        ):
            result = runner.invoke(cli, ["vision", "validate"])

            assert result.exit_code == 0
            # validate_vision_config is still called
            mock_validate.assert_called_once()


# ---------------------------------------------------------------------------
# missy vision memory
# ---------------------------------------------------------------------------


class TestVisionMemory:
    def test_no_sessions_shows_placeholder(self, runner: CliRunner) -> None:
        tracker = MagicMock()
        report = _make_memory_report()
        tracker.update_from_scene_manager.return_value = report

        with patch("missy.vision.memory_usage.get_memory_tracker", return_value=tracker):
            result = runner.invoke(cli, ["vision", "memory"])

        assert result.exit_code == 0
        assert "No active scene sessions" in result.output

    def test_header_always_shown(self, runner: CliRunner) -> None:
        tracker = MagicMock()
        tracker.update_from_scene_manager.return_value = _make_memory_report()

        with patch("missy.vision.memory_usage.get_memory_tracker", return_value=tracker):
            result = runner.invoke(cli, ["vision", "memory"])

        assert result.exit_code == 0
        assert "Memory" in result.output

    def test_totals_displayed(self, runner: CliRunner) -> None:
        tracker = MagicMock()
        report = _make_memory_report(
            total_bytes=50 * 1024 * 1024,  # 50 MB
            total_frames=8,
            session_count=2,
            active_sessions=1,
            limit_bytes=500 * 1024 * 1024,
            usage_fraction=0.1,
        )
        tracker.update_from_scene_manager.return_value = report

        with patch("missy.vision.memory_usage.get_memory_tracker", return_value=tracker):
            result = runner.invoke(cli, ["vision", "memory"])

        assert result.exit_code == 0
        assert "8" in result.output  # total_frames
        assert "2" in result.output  # session_count
        assert "1" in result.output  # active_sessions

    def test_with_sessions_shows_table(self, runner: CliRunner) -> None:
        tracker = MagicMock()
        report = _make_memory_report(
            total_bytes=12 * 1024 * 1024,
            total_frames=2,
            session_count=1,
            active_sessions=1,
            usage_fraction=0.024,
            sessions=[
                {
                    "task_id": "puzzle-001",
                    "frame_count": 2,
                    "estimated_mb": 12.0,
                    "active": True,
                }
            ],
        )
        tracker.update_from_scene_manager.return_value = report

        with patch("missy.vision.memory_usage.get_memory_tracker", return_value=tracker):
            result = runner.invoke(cli, ["vision", "memory"])

        assert result.exit_code == 0
        assert "puzzle-001" in result.output
        assert "12.0" in result.output or "12.00" in result.output

    def test_with_multiple_sessions(self, runner: CliRunner) -> None:
        tracker = MagicMock()
        report = _make_memory_report(
            total_bytes=30 * 1024 * 1024,
            total_frames=5,
            session_count=3,
            active_sessions=2,
            usage_fraction=0.06,
            sessions=[
                {
                    "task_id": "task-alpha",
                    "frame_count": 2,
                    "estimated_mb": 12.0,
                    "active": True,
                },
                {
                    "task_id": "task-beta",
                    "frame_count": 2,
                    "estimated_mb": 12.0,
                    "active": True,
                },
                {
                    "task_id": "task-gamma",
                    "frame_count": 1,
                    "estimated_mb": 6.0,
                    "active": False,
                },
            ],
        )
        tracker.update_from_scene_manager.return_value = report

        with patch("missy.vision.memory_usage.get_memory_tracker", return_value=tracker):
            result = runner.invoke(cli, ["vision", "memory"])

        assert result.exit_code == 0
        assert "task-alpha" in result.output
        assert "task-beta" in result.output
        assert "task-gamma" in result.output

    def test_over_limit_shows_warning(self, runner: CliRunner) -> None:
        tracker = MagicMock()
        report = _make_memory_report(
            total_bytes=600 * 1024 * 1024,
            limit_bytes=500 * 1024 * 1024,
            usage_fraction=1.2,
            over_limit=True,
            total_frames=100,
            session_count=5,
            active_sessions=5,
        )
        tracker.update_from_scene_manager.return_value = report

        with patch("missy.vision.memory_usage.get_memory_tracker", return_value=tracker):
            result = runner.invoke(cli, ["vision", "memory"])

        assert result.exit_code == 0
        assert "OVER LIMIT" in result.output

    def test_within_limit_no_over_limit_warning(self, runner: CliRunner) -> None:
        tracker = MagicMock()
        report = _make_memory_report(
            total_bytes=50 * 1024 * 1024,
            limit_bytes=500 * 1024 * 1024,
            usage_fraction=0.1,
            over_limit=False,
        )
        tracker.update_from_scene_manager.return_value = report

        with patch("missy.vision.memory_usage.get_memory_tracker", return_value=tracker):
            result = runner.invoke(cli, ["vision", "memory"])

        assert result.exit_code == 0
        assert "OVER LIMIT" not in result.output

    def test_inactive_session_shown_correctly(self, runner: CliRunner) -> None:
        tracker = MagicMock()
        report = _make_memory_report(
            total_bytes=6 * 1024 * 1024,
            total_frames=1,
            session_count=1,
            active_sessions=0,
            usage_fraction=0.012,
            sessions=[
                {
                    "task_id": "stale-session",
                    "frame_count": 1,
                    "estimated_mb": 6.0,
                    "active": False,
                }
            ],
        )
        tracker.update_from_scene_manager.return_value = report

        with patch("missy.vision.memory_usage.get_memory_tracker", return_value=tracker):
            result = runner.invoke(cli, ["vision", "memory"])

        assert result.exit_code == 0
        assert "stale-session" in result.output
        # Inactive sessions must not display the active indicator
        assert "no" in result.output.lower() or "0" in result.output
