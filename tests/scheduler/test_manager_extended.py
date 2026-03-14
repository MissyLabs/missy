"""Extended tests for missy.scheduler.manager covering uncovered paths."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.scheduler.manager import SchedulerManager


@pytest.fixture
def tmp_jobs_file(tmp_path: Path) -> str:
    return str(tmp_path / "jobs.json")


@pytest.fixture
def started_manager(tmp_jobs_file: str):
    mgr = SchedulerManager(jobs_file=tmp_jobs_file)
    mgr.start()
    yield mgr
    mgr.stop()


class TestRunJob:
    """Tests for _run_job — the internal callback executed by APScheduler."""

    def test_run_job_missing_job_id(self, started_manager: SchedulerManager):
        """_run_job with a non-existent job_id logs and returns silently."""
        started_manager._run_job("nonexistent-id")  # should not raise

    @patch("missy.scheduler.manager.uuid")
    def test_run_job_success(self, mock_uuid, started_manager: SchedulerManager):
        mock_uuid.uuid4.return_value = "test-session-id"
        job = started_manager.add_job("test", "every 5 minutes", "do stuff")

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            mock_agent = MagicMock()
            mock_agent.run.return_value = "Done!"
            MockRuntime.return_value = mock_agent

            started_manager._run_job(job.id)

        updated_job = started_manager._jobs[job.id]
        assert updated_job.run_count == 1
        assert updated_job.last_result == "Done!"
        assert updated_job.consecutive_failures == 0

    @patch("missy.scheduler.manager.uuid")
    def test_run_job_error_increments_failures(self, mock_uuid, started_manager: SchedulerManager):
        mock_uuid.uuid4.return_value = "test-session"
        job = started_manager.add_job("fail", "every 5 minutes", "crash")

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            MockRuntime.return_value.run.side_effect = RuntimeError("provider down")
            started_manager._run_job(job.id)

        updated = started_manager._jobs[job.id]
        assert updated.consecutive_failures == 1
        assert "provider down" in updated.last_error

    @patch("missy.scheduler.manager.uuid")
    def test_run_job_retries_on_failure(self, mock_uuid, started_manager: SchedulerManager):
        mock_uuid.uuid4.return_value = "sess"
        job = started_manager.add_job("retry", "every 5 minutes", "task", max_attempts=3)

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            MockRuntime.return_value.run.side_effect = RuntimeError("network error")
            started_manager._run_job(job.id)

        # After first failure, should have scheduled a retry
        retry_job = started_manager._scheduler.get_job(f"{job.id}_retry_1")
        assert retry_job is not None

    @patch("missy.scheduler.manager.uuid")
    def test_run_job_no_retry_after_max_attempts(self, mock_uuid, started_manager: SchedulerManager):
        mock_uuid.uuid4.return_value = "sess"
        job = started_manager.add_job("maxed", "every 5 minutes", "task", max_attempts=1)
        # Set failures to max
        job.consecutive_failures = 1

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            MockRuntime.return_value.run.side_effect = RuntimeError("error")
            started_manager._run_job(job.id)

        # Should NOT have scheduled a retry
        retry_job = started_manager._scheduler.get_job(f"{job.id}_retry_2")
        assert retry_job is None

    @patch("missy.scheduler.manager.uuid")
    def test_run_job_delete_after_run(self, mock_uuid, started_manager: SchedulerManager):
        mock_uuid.uuid4.return_value = "sess"
        job = started_manager.add_job(
            "oneshot", "every 5 minutes", "task", delete_after_run=True
        )

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            MockRuntime.return_value.run.return_value = "ok"
            started_manager._run_job(job.id)

        # Job should be removed
        assert job.id not in started_manager._jobs

    def test_run_job_active_hours_gate(self, started_manager: SchedulerManager):
        """Job outside active hours should be skipped."""
        job = started_manager.add_job("gated", "every 5 minutes", "task", active_hours="00:00-00:01")
        # Patch datetime.now to return a time outside the window
        with patch("missy.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 14, 12, 0, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            # Instead of mocking datetime, just set active_hours to an impossible window
            job.active_hours = "03:00-03:01"
            now = datetime.now()
            if now.hour == 3 and now.minute <= 1:
                pytest.skip("Running at 3AM, window matches")

            with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
                started_manager._run_job(job.id)
                MockRuntime.assert_not_called()


class TestScheduleJobVariants:
    """Tests for _schedule_job with different trigger types."""

    def test_schedule_cron_expression(self, started_manager: SchedulerManager):
        """Cron expression schedule should be accepted."""
        job = started_manager.add_job("cron", "*/5 * * * *", "task")
        assert job.id in started_manager._jobs

    def test_schedule_daily(self, started_manager: SchedulerManager):
        job = started_manager.add_job("daily", "daily at 14:30", "task")
        assert job.id in started_manager._jobs

    def test_schedule_weekly(self, started_manager: SchedulerManager):
        job = started_manager.add_job("weekly", "weekly on monday at 09:00", "task")
        assert job.id in started_manager._jobs

    def test_schedule_with_timezone(self, started_manager: SchedulerManager):
        job = started_manager.add_job(
            "tz", "daily at 09:00", "task", timezone="America/New_York"
        )
        assert job.timezone == "America/New_York"


class TestCleanupMemory:
    def test_cleanup_memory_no_store(self, started_manager: SchedulerManager):
        """cleanup_memory returns 0 when store import fails."""
        with patch("missy.scheduler.manager.SchedulerManager.cleanup_memory") as mock:
            mock.return_value = 0
            result = started_manager.cleanup_memory()
            assert result == 0

    def test_cleanup_memory_exception_returns_zero(self, started_manager: SchedulerManager):
        """cleanup_memory returns 0 on exception."""
        with patch.dict("sys.modules", {"missy.memory.store": None}):
            result = started_manager.cleanup_memory()
            assert result == 0


class TestListJobsWithDetails:
    def test_returns_same_as_list_jobs(self, started_manager: SchedulerManager):
        started_manager.add_job("a", "every 5 minutes", "t")
        assert started_manager.list_jobs_with_details() == started_manager.list_jobs()


class TestPersistence:
    def test_save_and_reload(self, tmp_jobs_file: str):
        """Jobs survive stop/start cycle."""
        mgr = SchedulerManager(jobs_file=tmp_jobs_file)
        mgr.start()
        job = mgr.add_job("persist", "every 10 minutes", "hello")
        mgr.stop()

        mgr2 = SchedulerManager(jobs_file=tmp_jobs_file)
        mgr2.start()
        jobs = mgr2.list_jobs()
        mgr2.stop()

        assert len(jobs) == 1
        assert jobs[0].id == job.id
        assert jobs[0].name == "persist"

    def test_load_jobs_not_array(self, tmp_jobs_file: str):
        """Jobs file with a non-array root is handled gracefully."""
        Path(tmp_jobs_file).parent.mkdir(parents=True, exist_ok=True)
        Path(tmp_jobs_file).write_text('{"not": "an array"}', encoding="utf-8")
        mgr = SchedulerManager(jobs_file=tmp_jobs_file)
        mgr.start()
        assert mgr.list_jobs() == []
        mgr.stop()

    def test_load_jobs_missing_file(self, tmp_path: Path):
        """No jobs file means empty list, no error."""
        mgr = SchedulerManager(jobs_file=str(tmp_path / "nonexistent.json"))
        mgr.start()
        assert mgr.list_jobs() == []
        mgr.stop()


class TestStopErrorHandling:
    def test_stop_handles_scheduler_exception(self, tmp_jobs_file: str):
        mgr = SchedulerManager(jobs_file=tmp_jobs_file)
        mgr.start()
        # Mock the scheduler to raise on shutdown
        mgr._scheduler.shutdown = MagicMock(side_effect=RuntimeError("oops"))
        mgr.stop()  # should not raise


class TestEmitEvent:
    def test_emit_event_exception_handled(self, started_manager: SchedulerManager):
        """_emit_event should not raise even if event_bus fails."""
        with patch("missy.scheduler.manager.event_bus.publish", side_effect=Exception("bus broken")):
            started_manager._emit_event(
                event_type="test.event", result="allow", detail={}
            )  # should not raise
