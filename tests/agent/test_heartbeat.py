"""Tests for missy.agent.heartbeat.HeartbeatRunner."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from missy.agent.heartbeat import HeartbeatRunner


class TestHeartbeatInit:
    """Initialization and basic attributes."""

    def test_default_values(self):
        runner = HeartbeatRunner(agent_run_fn=lambda p: "ok")
        assert runner._interval == 1800
        assert runner._active_hours == ""
        assert runner._runs == 0
        assert runner._skips == 0

    def test_custom_interval(self):
        runner = HeartbeatRunner(agent_run_fn=lambda p: "ok", interval_seconds=60)
        assert runner._interval == 60

    def test_custom_workspace(self, tmp_path):
        runner = HeartbeatRunner(agent_run_fn=lambda p: "ok", workspace=str(tmp_path))
        assert runner._workspace == tmp_path

    def test_metrics_initially_zero(self):
        runner = HeartbeatRunner(agent_run_fn=lambda p: "ok")
        assert runner.metrics == {"runs": 0, "skips": 0}


class TestHeartbeatFire:
    """Direct calls to _fire()."""

    def test_no_heartbeat_file_skips(self, tmp_path):
        runner = HeartbeatRunner(agent_run_fn=MagicMock(), workspace=str(tmp_path))
        runner._fire()
        assert runner._skips == 1
        assert runner._runs == 0

    def test_heartbeat_ok_suppression(self, tmp_path):
        (tmp_path / "HEARTBEAT_OK").write_text("suppress")
        (tmp_path / "HEARTBEAT.md").write_text("# Check things")
        runner = HeartbeatRunner(agent_run_fn=MagicMock(), workspace=str(tmp_path))
        runner._fire()
        assert runner._skips == 1
        assert runner._runs == 0
        # OK file should be deleted
        assert not (tmp_path / "HEARTBEAT_OK").exists()

    def test_successful_fire(self, tmp_path):
        checklist = "# Daily Check\n- [ ] Check logs"
        (tmp_path / "HEARTBEAT.md").write_text(checklist)
        run_fn = MagicMock(return_value="All clear")
        runner = HeartbeatRunner(agent_run_fn=run_fn, workspace=str(tmp_path))
        runner._fire()
        assert runner._runs == 1
        assert runner._skips == 0
        run_fn.assert_called_once()
        assert "[HEARTBEAT CHECK]" in run_fn.call_args[0][0]

    def test_fire_with_report_fn(self, tmp_path):
        (tmp_path / "HEARTBEAT.md").write_text("# Check")
        run_fn = MagicMock(return_value="result")
        report_fn = MagicMock()
        runner = HeartbeatRunner(
            agent_run_fn=run_fn,
            workspace=str(tmp_path),
            report_fn=report_fn,
        )
        runner._fire()
        report_fn.assert_called_once_with("result")

    def test_fire_exception_handled(self, tmp_path):
        (tmp_path / "HEARTBEAT.md").write_text("# Check")
        run_fn = MagicMock(side_effect=RuntimeError("provider down"))
        runner = HeartbeatRunner(agent_run_fn=run_fn, workspace=str(tmp_path))
        # Should not raise
        runner._fire()
        assert runner._runs == 0
        assert runner._skips == 0

    def test_fire_reads_unicode(self, tmp_path):
        (tmp_path / "HEARTBEAT.md").write_text(
            "# Check \U0001f60a\n- \u2714 Done", encoding="utf-8"
        )
        run_fn = MagicMock(return_value="ok")
        runner = HeartbeatRunner(agent_run_fn=run_fn, workspace=str(tmp_path))
        runner._fire()
        assert runner._runs == 1

    def test_multiple_fires(self, tmp_path):
        (tmp_path / "HEARTBEAT.md").write_text("# Check")
        run_fn = MagicMock(return_value="ok")
        runner = HeartbeatRunner(agent_run_fn=run_fn, workspace=str(tmp_path))
        runner._fire()
        runner._fire()
        runner._fire()
        assert runner._runs == 3
        assert runner.metrics == {"runs": 3, "skips": 0}


class TestActiveHours:
    """Tests for _in_active_hours()."""

    def test_empty_active_hours_always_active(self):
        runner = HeartbeatRunner(agent_run_fn=lambda p: "ok", active_hours="")
        assert runner._in_active_hours() is True

    def test_invalid_format_always_active(self):
        runner = HeartbeatRunner(agent_run_fn=lambda p: "ok", active_hours="badformat")
        assert runner._in_active_hours() is True

    def _make_runner_with_frozen_time(self, hour, minute, active_hours):
        """Create a runner and test _in_active_hours with a frozen time."""
        runner = HeartbeatRunner(agent_run_fn=lambda p: "ok", active_hours=active_hours)
        frozen = datetime(2026, 3, 18, hour, minute, 0)
        # Monkey-patch the method to use frozen time

        def patched():
            import re as _re

            m = _re.match(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})", runner._active_hours)
            if not m:
                return True
            now = frozen
            start = now.replace(hour=int(m[1]), minute=int(m[2]), second=0, microsecond=0)
            end = now.replace(hour=int(m[3]), minute=int(m[4]), second=0, microsecond=0)
            if end < start:
                return now >= start or now <= end
            return start <= now <= end

        runner._in_active_hours = patched
        return runner

    def test_within_active_hours(self):
        runner = self._make_runner_with_frozen_time(14, 30, "08:00-22:00")
        assert runner._in_active_hours() is True

    def test_outside_active_hours(self):
        runner = self._make_runner_with_frozen_time(3, 0, "08:00-22:00")
        assert runner._in_active_hours() is False

    def test_overnight_window_inside(self):
        """22:00-06:00 window, currently 23:00 → active."""
        runner = self._make_runner_with_frozen_time(23, 0, "22:00-06:00")
        assert runner._in_active_hours() is True

    def test_overnight_window_outside(self):
        """22:00-06:00 window, currently 12:00 → not active."""
        runner = self._make_runner_with_frozen_time(12, 0, "22:00-06:00")
        assert runner._in_active_hours() is False

    def test_outside_active_hours_skips_fire(self, tmp_path):
        (tmp_path / "HEARTBEAT.md").write_text("# Check")
        run_fn = MagicMock()
        runner = HeartbeatRunner(
            agent_run_fn=run_fn,
            workspace=str(tmp_path),
            active_hours="08:00-22:00",
        )
        with patch.object(runner, "_in_active_hours", return_value=False):
            runner._fire()
        assert runner._skips == 1
        run_fn.assert_not_called()


class TestStartStop:
    """Start/stop thread management."""

    def test_start_creates_thread(self):
        runner = HeartbeatRunner(agent_run_fn=lambda p: "ok", interval_seconds=3600)
        runner.start()
        assert runner._thread is not None
        assert runner._thread.daemon is True
        assert runner._thread.name == "missy-heartbeat"
        runner.stop()

    def test_stop_without_start(self):
        runner = HeartbeatRunner(agent_run_fn=lambda p: "ok")
        runner.stop()  # Should not raise

    def test_stop_sets_event(self):
        runner = HeartbeatRunner(agent_run_fn=lambda p: "ok", interval_seconds=3600)
        runner.start()
        runner.stop()
        assert runner._stop.is_set()
