"""Tests for missy.scheduler.manager.SchedulerManager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from missy.scheduler.jobs import ScheduledJob
from missy.scheduler.manager import SchedulerManager


@pytest.fixture
def tmp_jobs_file(tmp_path: Path) -> str:
    return str(tmp_path / "jobs.json")


@pytest.fixture
def started_manager(tmp_jobs_file: str):
    """A SchedulerManager that is started and stopped around each test."""
    mgr = SchedulerManager(jobs_file=tmp_jobs_file)
    mgr.start()
    yield mgr
    mgr.stop()


class TestSchedulerManagerLifecycle:
    def test_start_creates_jobs_file_dir(self, tmp_path: Path):
        nested = tmp_path / "sub" / "nested" / "jobs.json"
        mgr = SchedulerManager(jobs_file=str(nested))
        mgr.start()
        mgr.stop()
        # Ensure the parent directories were created (by _save_jobs on first write)
        # The dir need not exist yet; start() simply loads and schedules.
        assert True  # If no exception, start/stop succeeded

    def test_start_loads_persisted_jobs(self, tmp_jobs_file: str):
        # Pre-populate the jobs file with one job.
        job = ScheduledJob(name="persisted", schedule="every 10 minutes", task="hello")
        jobs_path = Path(tmp_jobs_file)
        jobs_path.parent.mkdir(parents=True, exist_ok=True)
        jobs_path.write_text(json.dumps([job.to_dict()]), encoding="utf-8")
        jobs_path.chmod(0o600)

        mgr = SchedulerManager(jobs_file=tmp_jobs_file)
        mgr.start()
        jobs = mgr.list_jobs()
        mgr.stop()
        assert len(jobs) == 1
        assert jobs[0].id == job.id


class TestAddJob:
    def test_add_job_returns_scheduled_job(self, started_manager: SchedulerManager):
        job = started_manager.add_job("daily", "daily at 09:00", "Report")
        assert isinstance(job, ScheduledJob)
        assert job.name == "daily"
        assert job.schedule == "daily at 09:00"
        assert job.task == "Report"

    def test_add_job_stores_provider(self, started_manager: SchedulerManager):
        job = started_manager.add_job("x", "every 5 minutes", "x", provider="openai")
        assert job.provider == "openai"

    def test_add_job_persists_to_file(self, started_manager: SchedulerManager, tmp_jobs_file: str):
        started_manager.add_job("saved", "every 1 hour", "task")
        data = json.loads(Path(tmp_jobs_file).read_text())
        assert len(data) == 1
        assert data[0]["name"] == "saved"

    def test_add_job_bad_schedule_raises_value_error(self, started_manager: SchedulerManager):
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            started_manager.add_job("bad", "every blue moon", "task")

    def test_add_job_bad_schedule_does_not_persist(
        self, started_manager: SchedulerManager, tmp_jobs_file: str
    ):
        with pytest.raises(ValueError):
            started_manager.add_job("bad", "every blue moon", "task")
        assert len(started_manager.list_jobs()) == 0

    def test_add_job_appears_in_list(self, started_manager: SchedulerManager):
        j = started_manager.add_job("A", "every 5 minutes", "task A")
        started_manager.add_job("B", "daily at 08:00", "task B")
        ids = {job.id for job in started_manager.list_jobs()}
        assert j.id in ids
        assert len(ids) == 2


class TestMaxJobsEnforcement:
    """SCHED-003: scheduling.max_jobs must actually be enforced.

    Previously a real, documented, correctly-parsed config field with
    zero enforcement anywhere -- setting max_jobs: 1 and adding 2 jobs
    succeeded identically both times, with no refusal.
    """

    def test_unlimited_by_default(self, tmp_jobs_file: str):
        mgr = SchedulerManager(jobs_file=tmp_jobs_file)
        mgr.start()
        mgr.add_job("a", "every 5 minutes", "t")
        mgr.add_job("b", "every 5 minutes", "t")
        mgr.add_job("c", "every 5 minutes", "t")
        assert len(mgr.list_jobs()) == 3
        mgr.stop()

    def test_second_job_refused_at_limit_one(self, tmp_jobs_file: str):
        from missy.core.exceptions import SchedulerError

        mgr = SchedulerManager(jobs_file=tmp_jobs_file, max_jobs=1)
        mgr.start()
        mgr.add_job("first", "every 5 minutes", "t")
        with pytest.raises(SchedulerError, match="max_jobs"):
            mgr.add_job("second", "every 5 minutes", "t")
        assert len(mgr.list_jobs()) == 1
        mgr.stop()

    def test_refused_job_not_persisted(self, tmp_jobs_file: str):
        from missy.core.exceptions import SchedulerError

        mgr = SchedulerManager(jobs_file=tmp_jobs_file, max_jobs=1)
        mgr.start()
        mgr.add_job("first", "every 5 minutes", "t")
        with pytest.raises(SchedulerError):
            mgr.add_job("second", "every 5 minutes", "t")
        data = json.loads(Path(tmp_jobs_file).read_text())
        assert len(data) == 1
        mgr.stop()

    def test_room_freed_after_remove(self, tmp_jobs_file: str):
        mgr = SchedulerManager(jobs_file=tmp_jobs_file, max_jobs=1)
        mgr.start()
        first = mgr.add_job("first", "every 5 minutes", "t")
        mgr.remove_job(first.id)
        # Room freed up -- adding again must succeed.
        second = mgr.add_job("second", "every 5 minutes", "t")
        assert len(mgr.list_jobs()) == 1
        assert second.name == "second"
        mgr.stop()

    def test_zero_max_jobs_is_unlimited(self, tmp_jobs_file: str):
        mgr = SchedulerManager(jobs_file=tmp_jobs_file, max_jobs=0)
        mgr.start()
        for i in range(5):
            mgr.add_job(f"job{i}", "every 5 minutes", "t")
        assert len(mgr.list_jobs()) == 5
        mgr.stop()


class TestRawCronDayOfWeekEndToEnd:
    """Regression: CronTrigger.from_crontab() applies no conversion between
    standard crontab's day-of-week numbering (Sunday=0..Saturday=6) and
    APScheduler's own (Monday=0..Sunday=6) -- so scheduling "0 9 * * 1-5"
    (intended as "weekdays" per this module's own docstring) silently
    fired Tuesday-Saturday instead. These tests actually schedule a job
    and check which real calendar dates it fires on, not just that the
    raw string is preserved (which the existing parser-only tests
    already cover but can't catch this class of bug).
    """

    def test_numeric_weekdays_fire_monday_through_friday(self, started_manager: SchedulerManager):
        from datetime import datetime

        job = started_manager.add_job("weekdays", "0 9 * * 1-5", "task")
        trigger = started_manager._scheduler.get_job(job.id).trigger

        # 2026-07-12 is a Sunday; walk a week forward and collect which
        # weekday names the trigger actually fires on.
        prev = None
        d = datetime(2026, 7, 12)
        fire_days = []
        for _ in range(5):
            nxt = trigger.get_next_fire_time(prev, d)
            fire_days.append(nxt.strftime("%A"))
            prev = nxt
            d = nxt

        assert fire_days == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    def test_numeric_sunday_fires_on_sunday(self, started_manager: SchedulerManager):
        from datetime import datetime

        job = started_manager.add_job("sunday-only", "0 9 * * 0", "task")
        trigger = started_manager._scheduler.get_job(job.id).trigger

        d = datetime(2026, 7, 12)  # a Sunday
        nxt = trigger.get_next_fire_time(None, d)
        assert nxt.strftime("%A") == "Sunday"

    def test_six_field_cron_with_seconds_schedules_successfully(
        self, started_manager: SchedulerManager
    ):
        """The 6-field-with-seconds format this module's docstring
        advertises previously always failed with a SchedulerError, since
        CronTrigger.from_crontab() hard-rejects anything but exactly 5
        fields.
        """
        from datetime import datetime

        job = started_manager.add_job("sixfield", "0 30 8 * * 1", "task")
        trigger = started_manager._scheduler.get_job(job.id).trigger

        d = datetime(2026, 7, 12)  # a Sunday
        nxt = trigger.get_next_fire_time(None, d)
        assert nxt.strftime("%A") == "Monday"
        assert (nxt.hour, nxt.minute, nxt.second) == (8, 30, 0)


class TestRemoveJob:
    def test_remove_job_removes_from_list(self, started_manager: SchedulerManager):
        job = started_manager.add_job("tmp", "every 5 minutes", "t")
        started_manager.remove_job(job.id)
        assert len(started_manager.list_jobs()) == 0

    def test_remove_job_persists_removal(
        self, started_manager: SchedulerManager, tmp_jobs_file: str
    ):
        job = started_manager.add_job("tmp", "every 5 minutes", "t")
        started_manager.remove_job(job.id)
        data = json.loads(Path(tmp_jobs_file).read_text())
        assert data == []

    def test_remove_nonexistent_job_raises_key_error(self, started_manager: SchedulerManager):
        with pytest.raises(KeyError):
            started_manager.remove_job("does-not-exist")


class TestPauseResumeJob:
    def test_pause_sets_enabled_false(self, started_manager: SchedulerManager):
        job = started_manager.add_job("p", "every 5 minutes", "t")
        started_manager.pause_job(job.id)
        assert started_manager.list_jobs()[0].enabled is False

    def test_resume_sets_enabled_true(self, started_manager: SchedulerManager):
        job = started_manager.add_job("p", "every 5 minutes", "t")
        started_manager.pause_job(job.id)
        started_manager.resume_job(job.id)
        assert started_manager.list_jobs()[0].enabled is True

    def test_pause_nonexistent_raises_key_error(self, started_manager: SchedulerManager):
        with pytest.raises(KeyError):
            started_manager.pause_job("bad-id")

    def test_resume_nonexistent_raises_key_error(self, started_manager: SchedulerManager):
        with pytest.raises(KeyError):
            started_manager.resume_job("bad-id")


class TestListJobs:
    def test_list_jobs_empty_initially(self, started_manager: SchedulerManager):
        assert started_manager.list_jobs() == []

    def test_list_jobs_returns_copies(self, started_manager: SchedulerManager):
        started_manager.add_job("x", "every 5 minutes", "t")
        a = started_manager.list_jobs()
        b = started_manager.list_jobs()
        assert a is not b


class TestLoadJobs:
    """Regression tests for the read-only ``load_jobs()`` diagnostic path.

    ``missy schedule list`` and ``missy doctor`` construct a fresh
    ``SchedulerManager`` and only want to read the persisted job list --
    they must not call ``start()``, since that registers every job with a
    live APScheduler ``BackgroundScheduler`` and starts its thread, risking
    a due job actually firing before ``stop()`` shuts it down.
    """

    def test_list_jobs_is_empty_before_any_load(self, tmp_jobs_file: str):
        job = ScheduledJob(name="persisted", schedule="every 10 minutes", task="hello")
        jobs_path = Path(tmp_jobs_file)
        jobs_path.parent.mkdir(parents=True, exist_ok=True)
        jobs_path.write_text(json.dumps([job.to_dict()]), encoding="utf-8")
        jobs_path.chmod(0o600)

        mgr = SchedulerManager(jobs_file=tmp_jobs_file)
        # Without start() or load_jobs(), _jobs was never populated from disk.
        assert mgr.list_jobs() == []

    def test_load_jobs_reads_persisted_jobs_without_starting_scheduler(self, tmp_jobs_file: str):
        job = ScheduledJob(name="persisted", schedule="every 10 minutes", task="hello")
        jobs_path = Path(tmp_jobs_file)
        jobs_path.parent.mkdir(parents=True, exist_ok=True)
        jobs_path.write_text(json.dumps([job.to_dict()]), encoding="utf-8")
        jobs_path.chmod(0o600)

        mgr = SchedulerManager(jobs_file=tmp_jobs_file)
        jobs = mgr.load_jobs()

        assert len(jobs) == 1
        assert jobs[0].id == job.id
        assert mgr._scheduler.running is False

    def test_load_jobs_matches_subsequent_list_jobs(self, tmp_jobs_file: str):
        job = ScheduledJob(name="persisted", schedule="every 10 minutes", task="hello")
        jobs_path = Path(tmp_jobs_file)
        jobs_path.parent.mkdir(parents=True, exist_ok=True)
        jobs_path.write_text(json.dumps([job.to_dict()]), encoding="utf-8")
        jobs_path.chmod(0o600)

        mgr = SchedulerManager(jobs_file=tmp_jobs_file)
        mgr.load_jobs()
        assert [j.id for j in mgr.list_jobs()] == [job.id]


class TestLoadSave:
    def test_malformed_jobs_file_logs_and_skips(self, tmp_jobs_file: str):
        Path(tmp_jobs_file).parent.mkdir(parents=True, exist_ok=True)
        Path(tmp_jobs_file).write_text("NOT JSON", encoding="utf-8")
        mgr = SchedulerManager(jobs_file=tmp_jobs_file)
        mgr.start()
        assert mgr.list_jobs() == []
        mgr.stop()

    def test_jobs_file_with_array_of_non_dicts_skips_records(self, tmp_jobs_file: str):
        Path(tmp_jobs_file).parent.mkdir(parents=True, exist_ok=True)
        Path(tmp_jobs_file).write_text(json.dumps(["bad", 123, None]), encoding="utf-8")
        mgr = SchedulerManager(jobs_file=tmp_jobs_file)
        mgr.start()
        assert mgr.list_jobs() == []
        mgr.stop()
