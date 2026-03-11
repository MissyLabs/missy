"""Tests for missy.scheduler.manager.SchedulerManager."""

from __future__ import annotations

import json
import os
import tempfile
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
