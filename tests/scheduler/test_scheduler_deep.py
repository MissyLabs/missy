"""Deep tests for the Missy scheduler subsystem.

Covers areas not addressed by the existing test suite:
- Parser: None/whitespace-only input, time boundary validation, tz not
  attached to interval, multiple adjacent whitespace, boundary hours/minutes
- Persistence: multiple-job round-trip, all-removed leaves empty array,
  unsafe file permissions refused, atomic write produces a real file
- Job lifecycle: description/backoff_seconds/retry_on/active_hours
  stored via add_job; whitespace-only name/task rejected
- Active-hours deterministic: exact window boundaries via datetime mock
- Overnight active-hours window logic
- Timezone: propagated from add_job through ScheduledJob and APScheduler
- Concurrent add_job: no data loss under thread contention
- Parser cron passthrough: 5-field and 6-field raw expressions
- Scheduler cleanup: removing every job writes an empty JSON array
"""

from __future__ import annotations

import json
import os
import stat
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.scheduler.jobs import ScheduledJob
from missy.scheduler.manager import SchedulerManager
from missy.scheduler.parser import parse_schedule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def jobs_path(tmp_path: Path) -> str:
    """Return a path string inside tmp_path; the file does not yet exist."""
    return str(tmp_path / "deep_jobs.json")


@pytest.fixture()
def mgr(jobs_path: str) -> SchedulerManager:
    """A running SchedulerManager backed by a temporary file."""
    m = SchedulerManager(jobs_file=jobs_path)
    m.start()
    yield m
    m.stop()


# ---------------------------------------------------------------------------
# 1. Schedule parser — normal forms
# ---------------------------------------------------------------------------


class TestParserNormalForms:
    """Verify every documented schedule format produces the expected config."""

    def test_every_10_seconds(self) -> None:
        result = parse_schedule("every 10 seconds")
        assert result == {"trigger": "interval", "seconds": 10}

    def test_every_1_minute_singular(self) -> None:
        result = parse_schedule("every 1 minute")
        assert result == {"trigger": "interval", "minutes": 1}

    def test_every_48_hours(self) -> None:
        result = parse_schedule("every 48 hours")
        assert result == {"trigger": "interval", "hours": 48}

    def test_daily_midnight(self) -> None:
        result = parse_schedule("daily at 00:00")
        assert result == {"trigger": "cron", "hour": 0, "minute": 0}

    def test_daily_last_minute_of_day(self) -> None:
        result = parse_schedule("daily at 23:59")
        assert result == {"trigger": "cron", "hour": 23, "minute": 59}

    def test_daily_single_digit_hour(self) -> None:
        result = parse_schedule("daily at 7:05")
        assert result == {"trigger": "cron", "hour": 7, "minute": 5}

    def test_weekly_with_optional_at(self) -> None:
        result = parse_schedule("weekly on Wednesday at 06:15")
        assert result["trigger"] == "cron"
        assert result["day_of_week"] == "wed"
        assert result["hour"] == 6
        assert result["minute"] == 15

    def test_weekly_without_optional_at(self) -> None:
        result = parse_schedule("weekly on thursday 18:45")
        assert result["trigger"] == "cron"
        assert result["day_of_week"] == "thu"
        assert result["hour"] == 18
        assert result["minute"] == 45

    def test_at_date_space_separator(self) -> None:
        result = parse_schedule("at 2099-01-01 00:00")
        assert result == {"trigger": "date", "run_date": "2099-01-01T00:00"}

    def test_at_date_t_separator(self) -> None:
        result = parse_schedule("at 2099-06-30T12:59")
        assert result == {"trigger": "date", "run_date": "2099-06-30T12:59"}

    def test_five_field_cron_all_wildcards(self) -> None:
        result = parse_schedule("* * * * *")
        assert result["trigger"] == "cron"
        assert result["_cron_expression"] == "* * * * *"

    def test_five_field_cron_every_five_minutes(self) -> None:
        result = parse_schedule("*/5 * * * *")
        assert result["_cron_expression"] == "*/5 * * * *"

    def test_five_field_cron_weekdays(self) -> None:
        result = parse_schedule("0 9 * * 1-5")
        assert result["_cron_expression"] == "0 9 * * 1-5"

    def test_six_field_cron(self) -> None:
        result = parse_schedule("30 8 * * 1 *")
        assert result["_cron_expression"] == "30 8 * * 1 *"


# ---------------------------------------------------------------------------
# 2. Schedule parser — edge cases and error paths
# ---------------------------------------------------------------------------


class TestParserEdgeCases:
    """Invalid / degenerate inputs must raise ValueError."""

    def test_none_input_raises(self) -> None:
        with pytest.raises((TypeError, AttributeError)):
            parse_schedule(None)  # type: ignore[arg-type]

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("   ")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("")

    def test_daily_invalid_hour_raises(self) -> None:
        # hour 24 is out of range 0-23
        with pytest.raises(ValueError, match="Hour must be 0-23"):
            parse_schedule("daily at 24:00")

    def test_daily_invalid_minute_raises(self) -> None:
        # minute 60 is out of range 0-59
        with pytest.raises(ValueError, match="Minute must be 0-59"):
            parse_schedule("daily at 12:60")

    def test_weekly_invalid_hour_raises(self) -> None:
        with pytest.raises(ValueError, match="Hour must be 0-23"):
            parse_schedule("weekly on monday 25:00")

    def test_weekly_invalid_minute_raises(self) -> None:
        with pytest.raises(ValueError, match="Minute must be 0-59"):
            parse_schedule("weekly on tuesday 10:99")

    def test_unknown_day_of_week_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised day of week"):
            parse_schedule("weekly on Blursday 09:00")

    def test_junk_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("run every morning")

    def test_partial_daily_no_time_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("daily at")

    def test_interval_missing_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("every 5")

    def test_inverted_every_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("5 minutes every")

    def test_leading_whitespace_stripped_before_match(self) -> None:
        result = parse_schedule("  daily at 08:00  ")
        assert result == {"trigger": "cron", "hour": 8, "minute": 0}


# ---------------------------------------------------------------------------
# 3. Timezone handling in the parser
# ---------------------------------------------------------------------------


class TestParserTimezone:
    """Timezone is attached to cron/date but NOT to interval."""

    def test_interval_no_timezone_key(self) -> None:
        result = parse_schedule("every 30 seconds", tz="America/Chicago")
        assert "timezone" not in result

    def test_daily_timezone_attached(self) -> None:
        result = parse_schedule("daily at 09:00", tz="Europe/Berlin")
        assert result["timezone"] == "Europe/Berlin"

    def test_weekly_timezone_attached(self) -> None:
        result = parse_schedule("weekly on friday at 17:00", tz="Asia/Tokyo")
        assert result["timezone"] == "Asia/Tokyo"

    def test_at_timezone_attached(self) -> None:
        result = parse_schedule("at 2099-12-31 23:59", tz="Pacific/Auckland")
        assert result["timezone"] == "Pacific/Auckland"

    def test_cron_expression_timezone_attached(self) -> None:
        result = parse_schedule("0 8 * * 1-5", tz="US/Central")
        assert result["timezone"] == "US/Central"

    def test_no_tz_argument_no_key(self) -> None:
        result = parse_schedule("daily at 09:00")
        assert "timezone" not in result

    def test_tz_none_not_attached(self) -> None:
        result = parse_schedule("daily at 09:00", tz=None)
        assert "timezone" not in result


# ---------------------------------------------------------------------------
# 4. Parser cron-expression passthrough
# ---------------------------------------------------------------------------


class TestParserCronPassthrough:
    """Already-cron strings go straight through without modification."""

    def test_every_minute(self) -> None:
        result = parse_schedule("* * * * *")
        assert result["trigger"] == "cron"
        assert result["_cron_expression"] == "* * * * *"

    def test_first_of_month_noon(self) -> None:
        result = parse_schedule("0 12 1 * *")
        assert result["_cron_expression"] == "0 12 1 * *"

    def test_complex_step(self) -> None:
        result = parse_schedule("*/15 8-18 * * 1-5")
        assert result["_cron_expression"] == "*/15 8-18 * * 1-5"

    def test_six_field_with_seconds(self) -> None:
        result = parse_schedule("0 30 9 * * *")
        assert result["_cron_expression"] == "0 30 9 * * *"

    def test_cron_expression_no_human_key(self) -> None:
        """Raw cron output must not contain human-friendly keys like 'hour'."""
        result = parse_schedule("0 9 * * *")
        assert "hour" not in result
        assert "minute" not in result


# ---------------------------------------------------------------------------
# 5. Job persistence — multiple jobs, empty after remove-all, file created
# ---------------------------------------------------------------------------


class TestPersistenceDeep:
    """Verify the JSON persistence layer in depth."""

    def test_multiple_jobs_survive_restart(self, jobs_path: str) -> None:
        m1 = SchedulerManager(jobs_file=jobs_path)
        m1.start()
        j1 = m1.add_job("alpha", "every 5 minutes", "task alpha")
        j2 = m1.add_job("beta", "daily at 08:00", "task beta")
        j3 = m1.add_job("gamma", "every 1 hour", "task gamma")
        m1.stop()

        m2 = SchedulerManager(jobs_file=jobs_path)
        m2.start()
        jobs = m2.list_jobs()
        m2.stop()

        ids = {j.id for j in jobs}
        assert j1.id in ids
        assert j2.id in ids
        assert j3.id in ids
        assert len(jobs) == 3

    def test_remove_all_jobs_writes_empty_array(self, jobs_path: str) -> None:
        m = SchedulerManager(jobs_file=jobs_path)
        m.start()
        j1 = m.add_job("one", "every 5 minutes", "t1")
        j2 = m.add_job("two", "every 10 minutes", "t2")
        m.remove_job(j1.id)
        m.remove_job(j2.id)
        m.stop()

        data = json.loads(Path(jobs_path).read_text(encoding="utf-8"))
        assert data == []

    def test_jobs_file_is_created_on_first_add(self, jobs_path: str) -> None:
        m = SchedulerManager(jobs_file=jobs_path)
        m.start()
        assert not Path(jobs_path).exists(), "file should not exist before first add"
        m.add_job("created", "every 5 minutes", "t")
        assert Path(jobs_path).exists(), "file must be created after add_job"
        m.stop()

    def test_persisted_file_has_restrictive_permissions(self, jobs_path: str) -> None:
        m = SchedulerManager(jobs_file=jobs_path)
        m.start()
        m.add_job("secure", "every 5 minutes", "t")
        m.stop()

        mode = Path(jobs_path).stat().st_mode
        # Must not be group-writable or world-writable
        assert not (mode & stat.S_IWGRP), "file must not be group-writable"
        assert not (mode & stat.S_IWOTH), "file must not be world-writable"

    def test_load_refuses_world_writable_file(self, jobs_path: str) -> None:
        """_load_jobs silently skips files with unsafe permissions."""
        path = Path(jobs_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        job = ScheduledJob(name="unsafe", schedule="every 5 minutes", task="t")
        path.write_text(json.dumps([job.to_dict()]), encoding="utf-8")
        # Make world-writable — triggers the security check
        path.chmod(0o666)

        m = SchedulerManager(jobs_file=jobs_path)
        m.start()
        assert m.list_jobs() == [], "world-writable jobs file must be rejected"
        m.stop()

    def test_load_refuses_group_writable_file(self, jobs_path: str) -> None:
        """_load_jobs silently skips files that are group-writable."""
        path = Path(jobs_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        job = ScheduledJob(name="grp-unsafe", schedule="every 5 minutes", task="t")
        path.write_text(json.dumps([job.to_dict()]), encoding="utf-8")
        path.chmod(0o620)  # group-writable

        m = SchedulerManager(jobs_file=jobs_path)
        m.start()
        assert m.list_jobs() == [], "group-writable jobs file must be rejected"
        m.stop()

    def test_partial_malformed_records_load_valid_ones(self, jobs_path: str) -> None:
        """A mix of valid and malformed records loads only the valid ones."""
        path = Path(jobs_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        good = ScheduledJob(name="good-job", schedule="every 5 minutes", task="do it")
        bad = {"id": "bad-id", "name": "bad", "created_at": "INVALID_DATE"}

        path.write_text(json.dumps([good.to_dict(), bad]), encoding="utf-8")
        path.chmod(0o600)

        m = SchedulerManager(jobs_file=jobs_path)
        m.start()
        jobs = m.list_jobs()
        m.stop()

        assert len(jobs) == 1
        assert jobs[0].id == good.id

    def test_job_field_values_round_trip_exactly(self, jobs_path: str) -> None:
        """Every significant field survives a stop/start cycle unchanged."""
        m1 = SchedulerManager(jobs_file=jobs_path)
        m1.start()
        orig = m1.add_job(
            name="full-check",
            schedule="weekly on saturday 10:00",
            task="Weekly digest",
            provider="openai",
            description="My weekly task",
            max_attempts=5,
            backoff_seconds=[15, 45, 120],
            retry_on=["timeout", "network"],
            delete_after_run=False,
            active_hours="08:00-20:00",
            timezone="Europe/London",
        )
        m1.stop()

        m2 = SchedulerManager(jobs_file=jobs_path)
        m2.start()
        loaded = m2.list_jobs()[0]
        m2.stop()

        assert loaded.id == orig.id
        assert loaded.name == "full-check"
        assert loaded.schedule == "weekly on saturday 10:00"
        assert loaded.task == "Weekly digest"
        assert loaded.provider == "openai"
        assert loaded.description == "My weekly task"
        assert loaded.max_attempts == 5
        assert loaded.backoff_seconds == [15, 45, 120]
        assert loaded.retry_on == ["timeout", "network"]
        assert loaded.active_hours == "08:00-20:00"
        assert loaded.timezone == "Europe/London"


# ---------------------------------------------------------------------------
# 6. Job lifecycle — add_job input validation
# ---------------------------------------------------------------------------


class TestAddJobValidation:
    """Detailed validation checks for add_job parameters."""

    def test_whitespace_only_name_rejected(self, mgr: SchedulerManager) -> None:
        with pytest.raises(ValueError, match="Job name must not be empty"):
            mgr.add_job("   ", "every 5 minutes", "task")

    def test_whitespace_only_task_rejected(self, mgr: SchedulerManager) -> None:
        with pytest.raises(ValueError, match="Job task must not be empty"):
            mgr.add_job("valid-name", "every 5 minutes", "   ")

    def test_max_attempts_zero_rejected(self, mgr: SchedulerManager) -> None:
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            mgr.add_job("j", "every 5 minutes", "t", max_attempts=0)

    def test_task_exactly_at_limit_accepted(self, mgr: SchedulerManager) -> None:
        task = "x" * 50_000
        job = mgr.add_job("limit-ok", "every 5 minutes", task)
        assert len(job.task) == 50_000

    def test_task_over_limit_rejected(self, mgr: SchedulerManager) -> None:
        task = "x" * 50_001
        with pytest.raises(ValueError, match="Job task is too long"):
            mgr.add_job("limit-fail", "every 5 minutes", task)

    def test_description_stored_correctly(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job("j", "every 5 minutes", "t", description="my description")
        assert job.description == "my description"

    def test_custom_backoff_seconds_stored(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job(
            "j", "every 5 minutes", "t", backoff_seconds=[5, 10, 20]
        )
        assert job.backoff_seconds == [5, 10, 20]

    def test_default_backoff_seconds_when_none(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job("j", "every 5 minutes", "t", backoff_seconds=None)
        assert job.backoff_seconds == [30, 60, 300]

    def test_custom_retry_on_stored(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job("j", "every 5 minutes", "t", retry_on=["timeout"])
        assert job.retry_on == ["timeout"]

    def test_default_retry_on_when_none(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job("j", "every 5 minutes", "t", retry_on=None)
        assert job.retry_on == ["network", "provider_error"]

    def test_delete_after_run_stored(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job("j", "every 5 minutes", "t", delete_after_run=True)
        assert job.delete_after_run is True

    def test_active_hours_stored(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job("j", "every 5 minutes", "t", active_hours="09:00-17:00")
        assert job.active_hours == "09:00-17:00"

    def test_timezone_stored(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job(
            "j", "daily at 09:00", "t", timezone="America/New_York"
        )
        assert job.timezone == "America/New_York"

    def test_add_and_list_preserves_order(self, mgr: SchedulerManager) -> None:
        names = ["first", "second", "third", "fourth"]
        for n in names:
            mgr.add_job(n, "every 5 minutes", "t")
        listed = [j.name for j in mgr.list_jobs()]
        assert listed == names


# ---------------------------------------------------------------------------
# 7. Job lifecycle — pause / resume persistence
# ---------------------------------------------------------------------------


class TestPauseResumePersistence:
    def test_paused_state_survives_restart(self, jobs_path: str) -> None:
        m1 = SchedulerManager(jobs_file=jobs_path)
        m1.start()
        job = m1.add_job("pausable", "every 5 minutes", "t")
        m1.pause_job(job.id)
        m1.stop()

        m2 = SchedulerManager(jobs_file=jobs_path)
        m2.start()
        loaded = m2.list_jobs()[0]
        m2.stop()

        assert loaded.enabled is False

    def test_resumed_state_survives_restart(self, jobs_path: str) -> None:
        m1 = SchedulerManager(jobs_file=jobs_path)
        m1.start()
        job = m1.add_job("toggle", "every 5 minutes", "t")
        m1.pause_job(job.id)
        m1.resume_job(job.id)
        m1.stop()

        m2 = SchedulerManager(jobs_file=jobs_path)
        m2.start()
        loaded = m2.list_jobs()[0]
        m2.stop()

        assert loaded.enabled is True

    def test_paused_job_not_scheduled_on_restart(self, jobs_path: str) -> None:
        """A disabled job must not be registered with APScheduler on reload."""
        m1 = SchedulerManager(jobs_file=jobs_path)
        m1.start()
        job = m1.add_job("noschedule", "every 5 minutes", "t")
        m1.pause_job(job.id)
        m1.stop()

        m2 = SchedulerManager(jobs_file=jobs_path)
        m2.start()
        ap_job = m2._scheduler.get_job(job.id)
        m2.stop()

        assert ap_job is None, "paused job must not be re-scheduled on start"


# ---------------------------------------------------------------------------
# 8. Active-hours window — deterministic tests via datetime.now patching
# ---------------------------------------------------------------------------


class TestActiveHoursDeterministic:
    """Use unittest.mock to freeze time for predictable active-hours tests."""

    @staticmethod
    def _make_job(window: str) -> ScheduledJob:
        return ScheduledJob(active_hours=window, schedule="every 5 minutes", task="t")

    def test_exactly_at_window_start(self) -> None:
        """Time exactly equal to window start is inside the window."""
        job = self._make_job("10:00-18:00")
        with patch("missy.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 10, 0, 0)
            assert job.should_run_now() is True

    def test_exactly_at_window_end(self) -> None:
        """Time exactly equal to window end is still inside (inclusive)."""
        job = self._make_job("10:00-18:00")
        with patch("missy.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 18, 0, 0)
            assert job.should_run_now() is True

    def test_one_minute_before_start(self) -> None:
        job = self._make_job("10:00-18:00")
        with patch("missy.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 9, 59, 0)
            assert job.should_run_now() is False

    def test_one_minute_after_end(self) -> None:
        job = self._make_job("10:00-18:00")
        with patch("missy.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 18, 1, 0)
            assert job.should_run_now() is False

    def test_midday_inside_window(self) -> None:
        job = self._make_job("08:00-20:00")
        with patch("missy.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 14, 30, 0)
            assert job.should_run_now() is True

    def test_midnight_outside_daytime_window(self) -> None:
        job = self._make_job("08:00-20:00")
        with patch("missy.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 0, 0, 0)
            assert job.should_run_now() is False


# ---------------------------------------------------------------------------
# 9. Overnight active-hours window
# ---------------------------------------------------------------------------


class TestOvernightWindow:
    """Windows where end < start span midnight (e.g. 22:00-06:00)."""

    def _make_overnight_job(self) -> ScheduledJob:
        return ScheduledJob(active_hours="22:00-06:00", schedule="every 5 minutes", task="t")

    def test_inside_overnight_window_after_start(self) -> None:
        job = self._make_overnight_job()
        with patch("missy.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 23, 30, 0)
            assert job.should_run_now() is True

    def test_inside_overnight_window_before_end(self) -> None:
        job = self._make_overnight_job()
        with patch("missy.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 2, 5, 0, 0)
            assert job.should_run_now() is True

    def test_outside_overnight_window_midday(self) -> None:
        job = self._make_overnight_job()
        with patch("missy.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 12, 0, 0)
            assert job.should_run_now() is False

    def test_overnight_window_exactly_at_start(self) -> None:
        job = self._make_overnight_job()
        with patch("missy.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 22, 0, 0)
            assert job.should_run_now() is True

    def test_overnight_window_exactly_at_end(self) -> None:
        job = self._make_overnight_job()
        with patch("missy.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 2, 6, 0, 0)
            assert job.should_run_now() is True

    def test_overnight_just_after_end_is_outside(self) -> None:
        job = self._make_overnight_job()
        with patch("missy.scheduler.jobs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 2, 6, 1, 0)
            assert job.should_run_now() is False


# ---------------------------------------------------------------------------
# 10. Timezone handling — end-to-end through manager
# ---------------------------------------------------------------------------


class TestTimezoneHandling:
    def test_cron_expression_with_utc_timezone(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job(
            "tz-cron", "0 9 * * 1-5", "weekday morning", timezone="UTC"
        )
        assert job.timezone == "UTC"

    def test_daily_with_eastern_timezone(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job(
            "tz-daily", "daily at 07:00", "morning", timezone="America/New_York"
        )
        assert job.timezone == "America/New_York"
        listed = mgr.list_jobs()
        assert listed[0].timezone == "America/New_York"

    def test_weekly_with_london_timezone(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job(
            "tz-weekly",
            "weekly on monday at 09:00",
            "monday meeting",
            timezone="Europe/London",
        )
        assert job.timezone == "Europe/London"

    def test_empty_timezone_default(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job("no-tz", "every 5 minutes", "t")
        assert job.timezone == ""

    def test_timezone_survives_persistence(self, jobs_path: str) -> None:
        m1 = SchedulerManager(jobs_file=jobs_path)
        m1.start()
        m1.add_job("persist-tz", "daily at 09:00", "t", timezone="Asia/Tokyo")
        m1.stop()

        m2 = SchedulerManager(jobs_file=jobs_path)
        m2.start()
        loaded = m2.list_jobs()[0]
        m2.stop()

        assert loaded.timezone == "Asia/Tokyo"


# ---------------------------------------------------------------------------
# 11. Job metadata storage
# ---------------------------------------------------------------------------


class TestJobMetadata:
    """Custom fields travel through the full stack correctly."""

    def test_description_empty_by_default(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job("no-desc", "every 5 minutes", "t")
        assert job.description == ""

    def test_description_non_empty(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job("desc-job", "every 5 minutes", "t", description="Do the thing")
        assert job.description == "Do the thing"

    def test_provider_field_stored_and_listed(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job("p", "every 5 minutes", "t", provider="ollama")
        assert mgr.list_jobs()[0].provider == "ollama"

    def test_schedule_string_preserved_verbatim(self, mgr: SchedulerManager) -> None:
        schedule = "weekly on friday 23:59"
        job = mgr.add_job("sched-check", schedule, "t")
        assert job.schedule == schedule
        assert mgr.list_jobs()[0].schedule == schedule

    def test_task_preserved_verbatim(self, mgr: SchedulerManager) -> None:
        task = "Summarise everything that happened today in three bullet points."
        job = mgr.add_job("task-check", "every 5 minutes", task)
        assert job.task == task

    def test_multiple_jobs_distinct_ids(self, mgr: SchedulerManager) -> None:
        ids = [mgr.add_job(f"job{i}", "every 5 minutes", "t").id for i in range(5)]
        assert len(set(ids)) == 5, "each job must receive a unique UUID"

    def test_list_jobs_with_details_returns_same_objects(
        self, mgr: SchedulerManager
    ) -> None:
        mgr.add_job("a", "every 5 minutes", "t")
        by_list = mgr.list_jobs()
        by_details = mgr.list_jobs_with_details()
        assert [j.id for j in by_list] == [j.id for j in by_details]


# ---------------------------------------------------------------------------
# 12. Concurrent job operations
# ---------------------------------------------------------------------------


class TestConcurrentJobOperations:
    """add_job must be safe when called from multiple threads simultaneously."""

    def test_concurrent_add_no_data_loss(self, mgr: SchedulerManager) -> None:
        errors: list[Exception] = []
        job_ids: list[str] = []
        lock = threading.Lock()

        def add_one(idx: int) -> None:
            try:
                job = mgr.add_job(f"concurrent-{idx}", "every 5 minutes", f"task {idx}")
                with lock:
                    job_ids.append(job.id)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=add_one, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Unexpected errors during concurrent add: {errors}"
        assert len(job_ids) == 10, "all 10 concurrent adds must succeed"

    def test_concurrent_add_then_remove(self, mgr: SchedulerManager) -> None:
        """Add N jobs sequentially, then remove them concurrently."""
        N = 8
        jobs = [mgr.add_job(f"seq-{i}", "every 5 minutes", f"t{i}") for i in range(N)]

        errors: list[Exception] = []

        def remove_one(job_id: str) -> None:
            try:
                mgr.remove_job(job_id)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=remove_one, args=(j.id,)) for j in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Unexpected errors during concurrent remove: {errors}"
        assert mgr.list_jobs() == []


# ---------------------------------------------------------------------------
# 13. Scheduler cleanup — removing all jobs
# ---------------------------------------------------------------------------


class TestSchedulerCleanup:
    def test_remove_all_leaves_empty_list(self, mgr: SchedulerManager) -> None:
        jobs = [mgr.add_job(f"c{i}", "every 5 minutes", "t") for i in range(4)]
        for j in jobs:
            mgr.remove_job(j.id)
        assert mgr.list_jobs() == []

    def test_remove_all_then_add_works(self, mgr: SchedulerManager) -> None:
        """After removing everything, adding a new job succeeds."""
        jobs = [mgr.add_job(f"r{i}", "every 5 minutes", "t") for i in range(3)]
        for j in jobs:
            mgr.remove_job(j.id)

        new_job = mgr.add_job("fresh", "daily at 12:00", "noon task")
        assert len(mgr.list_jobs()) == 1
        assert mgr.list_jobs()[0].id == new_job.id

    def test_remove_unknown_id_after_clear(self, mgr: SchedulerManager) -> None:
        """KeyError raised even when the manager has no jobs at all."""
        with pytest.raises(KeyError):
            mgr.remove_job("totally-made-up-id")

    def test_file_empty_array_after_remove_all(self, mgr: SchedulerManager, jobs_path: str) -> None:
        jobs = [mgr.add_job(f"d{i}", "every 5 minutes", "t") for i in range(3)]
        for j in jobs:
            mgr.remove_job(j.id)
        raw = json.loads(Path(jobs_path).read_text(encoding="utf-8"))
        assert raw == []


# ---------------------------------------------------------------------------
# 14. ScheduledJob dataclass — serialisation completeness
# ---------------------------------------------------------------------------


class TestScheduledJobSerialisation:
    """Ensure every field survives to_dict/from_dict without loss."""

    def test_all_fields_in_to_dict(self) -> None:
        job = ScheduledJob(
            name="full",
            description="A full job",
            schedule="*/5 * * * *",
            task="do it",
            provider="openai",
            enabled=False,
            run_count=7,
            last_result="success",
            max_attempts=4,
            backoff_seconds=[10, 20, 40],
            retry_on=["network"],
            consecutive_failures=2,
            last_error="timeout",
            delete_after_run=True,
            active_hours="08:00-18:00",
            timezone="UTC",
        )
        d = job.to_dict()

        assert d["name"] == "full"
        assert d["description"] == "A full job"
        assert d["provider"] == "openai"
        assert d["enabled"] is False
        assert d["run_count"] == 7
        assert d["last_result"] == "success"
        assert d["max_attempts"] == 4
        assert d["backoff_seconds"] == [10, 20, 40]
        assert d["retry_on"] == ["network"]
        assert d["consecutive_failures"] == 2
        assert d["last_error"] == "timeout"
        assert d["delete_after_run"] is True
        assert d["active_hours"] == "08:00-18:00"
        assert d["timezone"] == "UTC"

    def test_from_dict_max_attempts_clamped_to_10(self) -> None:
        """from_dict never allows max_attempts > 10 (security guard)."""
        job = ScheduledJob.from_dict({"max_attempts": 999})
        assert job.max_attempts == 10

    def test_from_dict_legacy_no_timezone_defaults_empty(self) -> None:
        """Old records without 'timezone' key load with empty string."""
        job = ScheduledJob.from_dict({"name": "legacy"})
        assert job.timezone == ""

    def test_from_dict_legacy_no_active_hours_defaults_empty(self) -> None:
        job = ScheduledJob.from_dict({"name": "legacy"})
        assert job.active_hours == ""

    def test_from_dict_legacy_no_delete_after_run_defaults_false(self) -> None:
        job = ScheduledJob.from_dict({"name": "legacy"})
        assert job.delete_after_run is False

    def test_round_trip_with_all_non_default_fields(self) -> None:
        original = ScheduledJob(
            name="roundtrip",
            description="desc",
            schedule="0 8 * * 1",
            task="Monday morning",
            provider="ollama",
            enabled=False,
            run_count=3,
            last_result="ok",
            max_attempts=2,
            backoff_seconds=[5, 10],
            retry_on=["timeout"],
            consecutive_failures=1,
            last_error="conn refused",
            delete_after_run=False,
            active_hours="07:00-19:00",
            timezone="America/Chicago",
        )
        restored = ScheduledJob.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.description == "desc"
        assert restored.provider == "ollama"
        assert restored.enabled is False
        assert restored.run_count == 3
        assert restored.backoff_seconds == [5, 10]
        assert restored.retry_on == ["timeout"]
        assert restored.consecutive_failures == 1
        assert restored.last_error == "conn refused"
        assert restored.active_hours == "07:00-19:00"
        assert restored.timezone == "America/Chicago"


# ---------------------------------------------------------------------------
# 15. _run_job active-hours gate — via manager
# ---------------------------------------------------------------------------


class TestRunJobActiveHoursGate:
    """_run_job must skip execution when outside active_hours."""

    def test_run_job_outside_window_does_not_call_agent(
        self, mgr: SchedulerManager
    ) -> None:
        job = mgr.add_job(
            "gated",
            "every 5 minutes",
            "t",
            # Impossible 1-minute window at 3 AM; almost certainly not now.
            active_hours="03:00-03:01",
        )

        # Guard: if we happen to be running at 3:00-3:01 AM, skip this test.
        now = datetime.now()
        if now.hour == 3 and now.minute <= 1:
            pytest.skip("Running exactly during the narrow active window")

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            mgr._run_job(job.id)
            MockRuntime.assert_not_called()

        # run_count must stay at 0 since we skipped
        assert mgr._jobs[job.id].run_count == 0

    def test_run_job_inside_window_calls_agent(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job("wide-open", "every 5 minutes", "t", active_hours="00:00-23:59")

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            mock_agent = MagicMock()
            mock_agent.run.return_value = "all good"
            MockRuntime.return_value = mock_agent
            mgr._run_job(job.id)

        assert mgr._jobs[job.id].run_count == 1
        assert mgr._jobs[job.id].last_result == "all good"
