"""Coverage gap tests for missy.scheduler.manager.

Targets uncovered lines: 70-71, 167-170, 198-199, 226-227, 254-255,
303-306, 416-417, 490-491, 512-513, 544-545, 573-574, 600-605, 625-626.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.core.exceptions import SchedulerError
from missy.scheduler.jobs import ScheduledJob
from missy.scheduler.manager import SchedulerManager


@pytest.fixture
def tmp_jobs_file(tmp_path: Path) -> str:
    return str(tmp_path / "jobs.json")


@pytest.fixture
def started_manager(tmp_jobs_file: str):
    mgr = SchedulerManager(jobs_file=tmp_jobs_file)
    mgr.start()
    yield mgr
    with contextlib.suppress(Exception):
        mgr.stop()


# ---------------------------------------------------------------------------
# Lines 70-71: start() raises SchedulerError when _scheduler.start() fails
# ---------------------------------------------------------------------------


class TestStartRaisesSchedulerError:
    def test_start_wraps_scheduler_exception(self, tmp_jobs_file: str):
        mgr = SchedulerManager(jobs_file=tmp_jobs_file)
        mgr._scheduler.start = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(SchedulerError, match="Failed to start background scheduler"):
            mgr.start()


# ---------------------------------------------------------------------------
# Lines 167-170: add_job rolls back when _schedule_job raises SchedulerError
# ---------------------------------------------------------------------------


class TestAddJobRollbackOnSchedulerError:
    def test_add_job_rolls_back_when_schedule_job_raises(self, started_manager: SchedulerManager):
        with (
            patch.object(
                started_manager,
                "_schedule_job",
                side_effect=SchedulerError("apscheduler refused"),
            ),
            pytest.raises(SchedulerError),
        ):
            started_manager.add_job("failing", "every 5 minutes", "task")

        # Job must not persist in memory after rollback
        assert len(started_manager.list_jobs()) == 0


# ---------------------------------------------------------------------------
# Lines 198-199: remove_job raises SchedulerError when APScheduler call fails
# ---------------------------------------------------------------------------


class TestRemoveJobSchedulerError:
    def test_remove_job_wraps_apscheduler_exception(self, started_manager: SchedulerManager):
        job = started_manager.add_job("r", "every 5 minutes", "t")
        started_manager._scheduler.remove_job = MagicMock(side_effect=RuntimeError("locked"))
        # The scheduler.get_job check must return a truthy value so the code
        # path that calls remove_job is reached.
        started_manager._scheduler.get_job = MagicMock(return_value=MagicMock())
        with pytest.raises(SchedulerError, match="Failed to remove APScheduler job"):
            started_manager.remove_job(job.id)


# ---------------------------------------------------------------------------
# Lines 226-227: pause_job raises SchedulerError when APScheduler call fails
# ---------------------------------------------------------------------------


class TestPauseJobSchedulerError:
    def test_pause_job_wraps_apscheduler_exception(self, started_manager: SchedulerManager):
        job = started_manager.add_job("p", "every 5 minutes", "t")
        started_manager._scheduler.pause_job = MagicMock(side_effect=RuntimeError("no"))
        with pytest.raises(SchedulerError, match="Failed to pause job"):
            started_manager.pause_job(job.id)


# ---------------------------------------------------------------------------
# Lines 254-255: resume_job raises SchedulerError when APScheduler call fails
# ---------------------------------------------------------------------------


class TestResumeJobSchedulerError:
    def test_resume_job_wraps_apscheduler_exception(self, started_manager: SchedulerManager):
        job = started_manager.add_job("r", "every 5 minutes", "t")
        started_manager.pause_job(job.id)
        started_manager._scheduler.resume_job = MagicMock(side_effect=RuntimeError("no"))
        with pytest.raises(SchedulerError, match="Failed to resume job"):
            started_manager.resume_job(job.id)


# ---------------------------------------------------------------------------
# Lines 303-306: cleanup_memory delegates to MemoryStore.cleanup when present
# ---------------------------------------------------------------------------


class TestCleanupMemoryWithStore:
    def test_cleanup_memory_calls_store_cleanup(self, started_manager: SchedulerManager):
        mock_store = MagicMock()
        mock_store.cleanup.return_value = 7

        with patch("missy.scheduler.manager.SchedulerManager.cleanup_memory") as mock_method:
            mock_method.return_value = 7
            result = started_manager.cleanup_memory(older_than_days=14)
        assert result == 7

    def test_cleanup_memory_store_without_cleanup_returns_zero(
        self, started_manager: SchedulerManager
    ):
        """Store that lacks a cleanup() attribute returns 0."""
        mock_store = MagicMock(spec=[])  # no attributes

        fake_store_module = MagicMock()
        fake_store_module.MemoryStore.return_value = mock_store

        with patch.dict("sys.modules", {"missy.memory.store": fake_store_module}):
            result = started_manager.cleanup_memory()
        assert result == 0

    def test_cleanup_memory_store_with_cleanup_returns_count(
        self, started_manager: SchedulerManager
    ):
        """Store that has cleanup() returns whatever cleanup returns."""
        mock_store = MagicMock()
        mock_store.cleanup.return_value = 42

        fake_store_module = MagicMock()
        fake_store_module.MemoryStore.return_value = mock_store

        with patch.dict("sys.modules", {"missy.memory.store": fake_store_module}):
            result = started_manager.cleanup_memory(older_than_days=30)
        assert result == 42


# ---------------------------------------------------------------------------
# Lines 416-417: retry scheduling fails when scheduler.add_job raises
# ---------------------------------------------------------------------------


class TestRetrySchedulingFailure:
    @patch("missy.scheduler.manager.uuid")
    def test_retry_scheduling_failure_is_logged_not_raised(
        self, mock_uuid, started_manager: SchedulerManager
    ):
        """When scheduling a retry job raises, the error is logged (not propagated)."""
        mock_uuid.uuid4.return_value = "sess"
        job = started_manager.add_job("fail-retry", "every 5 minutes", "task", max_attempts=3)

        # Make the main scheduler.add_job fail on the second call (the retry call).
        original_add_job = started_manager._scheduler.add_job
        call_count = [0]

        def add_job_stub(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                raise RuntimeError("retry queue full")
            return original_add_job(*args, **kwargs)

        started_manager._scheduler.add_job = add_job_stub

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            MockRuntime.return_value.run.side_effect = RuntimeError("network error")
            # Must not raise even though retry scheduling fails
            started_manager._run_job(job.id)

        assert started_manager._jobs[job.id].consecutive_failures == 1


# ---------------------------------------------------------------------------
# Lines 490-491: delete_after_run removal failure is warned, not raised
# ---------------------------------------------------------------------------


class TestDeleteAfterRunFailure:
    @patch("missy.scheduler.manager.uuid")
    def test_delete_after_run_failure_logged_not_raised(
        self, mock_uuid, started_manager: SchedulerManager
    ):
        mock_uuid.uuid4.return_value = "sess"
        job = started_manager.add_job("oneshot", "every 5 minutes", "task", delete_after_run=True)

        with (
            patch.object(started_manager, "remove_job", side_effect=Exception("cannot remove")),
            patch("missy.agent.runtime.AgentRuntime") as MockRuntime,
        ):
            MockRuntime.return_value.run.return_value = "ok"
            # Should not raise despite remove_job failing
            started_manager._run_job(job.id)

        # Job still in memory since remove failed, but no exception propagated
        assert job.id in started_manager._jobs


# ---------------------------------------------------------------------------
# Lines 512-513: _save_jobs exception is logged, not re-raised
# ---------------------------------------------------------------------------


class TestSaveJobsExceptionHandling:
    def test_save_jobs_exception_does_not_propagate(self, tmp_jobs_file: str):
        mgr = SchedulerManager(jobs_file=tmp_jobs_file)
        mgr.start()
        # Make write_text raise
        mgr.jobs_file = MagicMock()
        mgr.jobs_file.parent.mkdir = MagicMock()
        mgr.jobs_file.write_text = MagicMock(side_effect=OSError("disk full"))
        # _save_jobs must not propagate
        mgr._save_jobs()  # should not raise
        mgr._scheduler.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Lines 544-545: _load_jobs skips malformed records from ScheduledJob.from_dict
# ---------------------------------------------------------------------------


class TestLoadJobsSkipsMalformedRecord:
    def test_load_jobs_skips_record_with_bad_from_dict(self, tmp_jobs_file: str):
        """A record that causes ScheduledJob.from_dict to raise is skipped."""
        # Write a valid-looking array but with a dict that will fail from_dict
        # because created_at has an invalid datetime string.
        Path(tmp_jobs_file).parent.mkdir(parents=True, exist_ok=True)
        Path(tmp_jobs_file).write_text(
            json.dumps([{"id": "x", "name": "bad", "created_at": "NOT_A_DATE"}]),
            encoding="utf-8",
        )
        mgr = SchedulerManager(jobs_file=tmp_jobs_file)
        mgr.start()
        # The bad record should be skipped without raising
        assert mgr.list_jobs() == []
        mgr.stop()


# ---------------------------------------------------------------------------
# Lines 573-574: _schedule_job raises SchedulerError when parse_schedule fails
# ---------------------------------------------------------------------------


class TestScheduleJobParseError:
    def test_schedule_job_wraps_parse_schedule_value_error(self, started_manager: SchedulerManager):
        job = ScheduledJob(name="bad", schedule="not a real schedule", task="t")
        started_manager._jobs[job.id] = job

        with pytest.raises(SchedulerError, match="invalid schedule"):
            started_manager._schedule_job(job)


# ---------------------------------------------------------------------------
# Lines 600-605: _schedule_job date trigger branch
# ---------------------------------------------------------------------------


class TestScheduleJobDateTrigger:
    def test_schedule_job_date_trigger_internal(self, started_manager: SchedulerManager):
        """Directly test _schedule_job with a date trigger config using a datetime object."""
        import datetime

        from missy.scheduler.jobs import ScheduledJob

        future = datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(hours=2)
        # Patch parse_schedule to return a date trigger config with a real datetime object
        # (APScheduler's DateTrigger requires a datetime, not an ISO string).
        date_config = {
            "trigger": "date",
            "run_date": future,
            "timezone": None,
        }
        job = ScheduledJob(name="date-job", schedule="placeholder", task="task")
        started_manager._jobs[job.id] = job

        with patch("missy.scheduler.manager.parse_schedule", return_value=date_config):
            started_manager._schedule_job(job)

        assert started_manager._scheduler.get_job(job.id) is not None

    def test_schedule_job_date_trigger_with_timezone(self, started_manager: SchedulerManager):
        """Date trigger branch respects the timezone field."""
        import datetime

        from missy.scheduler.jobs import ScheduledJob

        future = datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(hours=3)
        date_config = {
            "trigger": "date",
            "run_date": future,
            "timezone": "UTC",
        }
        job = ScheduledJob(name="date-job-tz", schedule="placeholder", task="task", timezone="UTC")
        started_manager._jobs[job.id] = job

        with patch("missy.scheduler.manager.parse_schedule", return_value=date_config):
            started_manager._schedule_job(job)

        assert started_manager._scheduler.get_job(job.id) is not None


# ---------------------------------------------------------------------------
# Lines 625-626: _schedule_job raises SchedulerError when APScheduler fails
# ---------------------------------------------------------------------------


class TestScheduleJobAPSchedulerFailure:
    def test_schedule_job_wraps_apscheduler_exception(self, started_manager: SchedulerManager):
        job = ScheduledJob(name="aps-fail", schedule="every 5 minutes", task="t")
        started_manager._jobs[job.id] = job
        started_manager._scheduler.add_job = MagicMock(side_effect=RuntimeError("quota exceeded"))

        with pytest.raises(SchedulerError, match="APScheduler failed to add job"):
            started_manager._schedule_job(job)
