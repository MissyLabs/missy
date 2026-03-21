"""Comprehensive extended tests for missy.scheduler parser and manager.

Covers genuine gaps left by the existing 233-test suite:

Parser
------
- "hourly" is not a supported shorthand (raises ValueError)
- "daily at 3pm" 12-hour format is not recognised (raises ValueError)
- "every Monday" bare shorthand raises ValueError
- "weekly" bare string raises ValueError
- None input raises TypeError / AttributeError
- Whitespace-only raises ValueError
- Single-field and two-field strings (not valid cron) raise ValueError
- "every 0 minutes" — zero interval is rejected
- Raw cron with comma-separated values is parsed
- Raw cron with step expressions is parsed
- Timezone None vs omitted produces the same result (no "timezone" key)
- Integer-type argument to parse_schedule raises AttributeError (no .strip)
- at-trigger with T-separator normalised correctly

Manager
-------
- add_job returns a job whose .id is a valid UUID
- list_jobs returns a *list*, not a dict or generator
- list_jobs_with_details returns the same ids as list_jobs
- add_job with empty string name raises ValueError
- add_job with empty string task raises ValueError
- add_job with max_attempts=0 raises ValueError
- task length exactly at 50 000 chars is accepted
- task length 50 001 chars is rejected
- Multiple jobs list length reflects add / remove correctly
- pause / resume round-trip persisted to JSON correctly
- Removing all jobs leaves list empty and file empty array
- Job with delete_after_run=True is cleaned up after successful _run_job
- _run_job with missing job_id is a no-op (no exception)
- _run_job increments consecutive_failures on exception
- _run_job resets consecutive_failures on success
- _run_job schedules a retry job in APScheduler after first failure
- _run_job does NOT schedule a retry when max_attempts exhausted
- Active-hours gate: agent not called when job is outside window
- Active-hours gate: agent IS called when job has no active_hours restriction
- cleanup_memory with no memory store returns 0
- cleanup_memory delegates to store.cleanup when available
- SchedulerManager start raises SchedulerError when APScheduler fails
- add_job rolls back in-memory state when _schedule_job raises SchedulerError
- remove_job raises SchedulerError when APScheduler.remove_job raises
- pause_job raises SchedulerError when APScheduler.pause_job raises
- resume_job raises SchedulerError when APScheduler.resume_job raises
- _schedule_job raises SchedulerError for invalid schedule string
- _schedule_job raises SchedulerError when APScheduler.add_job raises
- _emit_event silently handles a broken event bus
- Jobs persist across stop / start cycle (field-level check)
- Paused jobs are NOT re-scheduled with APScheduler on restart
- Concurrent add_job from multiple threads produces no data loss
"""

from __future__ import annotations

import contextlib
import datetime
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.core.exceptions import SchedulerError
from missy.scheduler.jobs import ScheduledJob
from missy.scheduler.manager import SchedulerManager
from missy.scheduler.parser import parse_schedule

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def jobs_path(tmp_path: Path) -> str:
    return str(tmp_path / "ext_jobs.json")


@pytest.fixture()
def mgr(jobs_path: str) -> SchedulerManager:
    m = SchedulerManager(jobs_file=jobs_path)
    m.start()
    yield m
    with contextlib.suppress(Exception):
        m.stop()


# ===========================================================================
# PARSER TESTS
# ===========================================================================


class TestParserHourlyShorthand:
    """'hourly' is NOT a supported shorthand — it must raise ValueError."""

    def test_hourly_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("hourly")

    def test_hourly_with_trailing_space_still_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("hourly  ")


class TestParserDailyAt3pm:
    """12-hour clock formats like '3pm' are not supported."""

    def test_daily_at_3pm_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("daily at 3pm")

    def test_daily_at_noon_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("daily at noon")

    def test_at_3pm_bare_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("at 3pm")


class TestParserEveryMondayShorthand:
    """'every Monday' bare shorthand is not a recognised format."""

    def test_every_monday_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("every Monday")

    def test_every_day_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("every day")

    def test_every_week_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("every week")


class TestParserWeeklyBare:
    """'weekly' by itself is not parseable — a day and time are required."""

    def test_weekly_bare_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("weekly")

    def test_weekly_on_monday_no_time_raises(self) -> None:
        # Missing the HH:MM portion
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("weekly on monday")

    def test_weekly_missing_day_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("weekly on 09:00")


class TestParserEdgeCases:
    """None, empty string, whitespace, integers."""

    def test_none_raises_attribute_or_type_error(self) -> None:
        with pytest.raises((TypeError, AttributeError)):
            parse_schedule(None)  # type: ignore[arg-type]

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("   \t  ")

    def test_integer_input_raises(self) -> None:
        with pytest.raises((TypeError, AttributeError)):
            parse_schedule(42)  # type: ignore[arg-type]

    def test_single_cron_field_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("*/5")

    def test_two_cron_fields_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("0 9")

    def test_four_cron_fields_raises(self) -> None:
        # Raw cron needs exactly 5 or 6 fields
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule("0 9 * *")


class TestParserZeroInterval:
    """The interval value must be positive (>0)."""

    def test_zero_minutes_raises(self) -> None:
        # The regex itself requires \d+ (at least one digit, no leading zero
        # prohibition), but the handler checks value > 0. However the regex
        # pattern requires \d+ which matches "0", so parse_schedule("every 0
        # minutes") will match the regex and then _parse_interval raises.
        with pytest.raises(ValueError, match="Interval value must be positive"):
            parse_schedule("every 0 minutes")

    def test_zero_seconds_raises(self) -> None:
        with pytest.raises(ValueError, match="Interval value must be positive"):
            parse_schedule("every 0 seconds")

    def test_zero_hours_raises(self) -> None:
        with pytest.raises(ValueError, match="Interval value must be positive"):
            parse_schedule("every 0 hours")


class TestParserRawCronVariants:
    """Raw cron expressions with comma-separated values and steps."""

    def test_comma_separated_dow(self) -> None:
        result = parse_schedule("0 9 * * 1,3,5")
        assert result["trigger"] == "cron"
        assert result["_cron_expression"] == "0 9 * * 1,3,5"

    def test_step_expression(self) -> None:
        result = parse_schedule("*/15 * * * *")
        assert result["_cron_expression"] == "*/15 * * * *"

    def test_range_expression(self) -> None:
        result = parse_schedule("0 8-18 * * 1-5")
        assert result["_cron_expression"] == "0 8-18 * * 1-5"

    def test_six_field_cron_preserved(self) -> None:
        result = parse_schedule("0 0 9 * * 1")
        assert result["_cron_expression"] == "0 0 9 * * 1"
        assert result["trigger"] == "cron"

    def test_cron_no_hour_minute_keys(self) -> None:
        """Raw cron output must not contain 'hour' or 'minute' keys."""
        result = parse_schedule("30 7 * * *")
        assert "hour" not in result
        assert "minute" not in result


class TestParserTimezoneNoneOmitted:
    """Passing tz=None vs omitting tz must produce the same result."""

    def test_daily_tz_none_no_timezone_key(self) -> None:
        result = parse_schedule("daily at 09:00", tz=None)
        assert "timezone" not in result

    def test_daily_omit_tz_no_timezone_key(self) -> None:
        result = parse_schedule("daily at 09:00")
        assert "timezone" not in result

    def test_cron_tz_none_no_timezone_key(self) -> None:
        result = parse_schedule("0 9 * * *", tz=None)
        assert "timezone" not in result

    def test_interval_with_any_tz_no_timezone_key(self) -> None:
        """Interval triggers never include a timezone regardless of tz arg."""
        result = parse_schedule("every 5 minutes", tz="America/Chicago")
        assert "timezone" not in result

    def test_interval_tz_none_no_timezone_key(self) -> None:
        result = parse_schedule("every 10 hours", tz=None)
        assert "timezone" not in result


class TestParserAtTriggerNormalisation:
    """The at-trigger normalises space to 'T' in run_date."""

    def test_space_separator_normalised_to_t(self) -> None:
        result = parse_schedule("at 2030-07-04 18:30")
        assert result["run_date"] == "2030-07-04T18:30"
        assert " " not in result["run_date"]

    def test_t_separator_preserved(self) -> None:
        result = parse_schedule("at 2030-07-04T18:30")
        assert result["run_date"] == "2030-07-04T18:30"

    def test_at_trigger_type_is_date(self) -> None:
        result = parse_schedule("at 2099-12-31 23:59")
        assert result["trigger"] == "date"

    def test_at_trigger_with_tz(self) -> None:
        result = parse_schedule("at 2099-01-01 00:00", tz="UTC")
        assert result["timezone"] == "UTC"
        assert result["trigger"] == "date"


# ===========================================================================
# MANAGER TESTS
# ===========================================================================


class TestManagerAddJobReturnsValidUUID:
    """add_job always returns a ScheduledJob with a proper UUID id."""

    def test_job_id_is_uuid_format(self, mgr: SchedulerManager) -> None:
        import re

        job = mgr.add_job("uuid-test", "every 5 minutes", "task")
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            job.id,
        ), f"job.id is not a valid UUID4: {job.id!r}"

    def test_each_job_has_unique_id(self, mgr: SchedulerManager) -> None:
        ids = [mgr.add_job(f"j{i}", "every 5 minutes", "t").id for i in range(5)]
        assert len(set(ids)) == 5


class TestManagerListJobsType:
    """list_jobs must return a plain list."""

    def test_list_jobs_returns_list(self, mgr: SchedulerManager) -> None:
        assert isinstance(mgr.list_jobs(), list)

    def test_list_jobs_with_details_returns_list(self, mgr: SchedulerManager) -> None:
        assert isinstance(mgr.list_jobs_with_details(), list)

    def test_list_jobs_with_details_ids_match_list_jobs(self, mgr: SchedulerManager) -> None:
        mgr.add_job("a", "every 5 minutes", "t")
        mgr.add_job("b", "daily at 09:00", "t")
        base = [j.id for j in mgr.list_jobs()]
        detailed = [j.id for j in mgr.list_jobs_with_details()]
        assert base == detailed

    def test_list_jobs_length_reflects_adds(self, mgr: SchedulerManager) -> None:
        for i in range(4):
            mgr.add_job(f"job{i}", "every 5 minutes", "t")
        assert len(mgr.list_jobs()) == 4

    def test_list_jobs_length_reflects_remove(self, mgr: SchedulerManager) -> None:
        j = mgr.add_job("removable", "every 5 minutes", "t")
        mgr.add_job("keeper", "daily at 12:00", "t")
        mgr.remove_job(j.id)
        assert len(mgr.list_jobs()) == 1


class TestManagerAddJobValidation:
    """Input validation edge cases for add_job."""

    def test_empty_name_raises(self, mgr: SchedulerManager) -> None:
        with pytest.raises(ValueError, match="Job name must not be empty"):
            mgr.add_job("", "every 5 minutes", "task")

    def test_whitespace_name_raises(self, mgr: SchedulerManager) -> None:
        with pytest.raises(ValueError, match="Job name must not be empty"):
            mgr.add_job("   ", "every 5 minutes", "task")

    def test_empty_task_raises(self, mgr: SchedulerManager) -> None:
        with pytest.raises(ValueError, match="Job task must not be empty"):
            mgr.add_job("valid", "every 5 minutes", "")

    def test_whitespace_task_raises(self, mgr: SchedulerManager) -> None:
        with pytest.raises(ValueError, match="Job task must not be empty"):
            mgr.add_job("valid", "every 5 minutes", "  \n ")

    def test_max_attempts_zero_raises(self, mgr: SchedulerManager) -> None:
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            mgr.add_job("j", "every 5 minutes", "t", max_attempts=0)

    def test_max_attempts_negative_raises(self, mgr: SchedulerManager) -> None:
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            mgr.add_job("j", "every 5 minutes", "t", max_attempts=-1)

    def test_task_at_limit_accepted(self, mgr: SchedulerManager) -> None:
        task = "x" * 50_000
        job = mgr.add_job("at-limit", "every 5 minutes", task)
        assert len(job.task) == 50_000

    def test_task_over_limit_rejected(self, mgr: SchedulerManager) -> None:
        task = "x" * 50_001
        with pytest.raises(ValueError, match="Job task is too long"):
            mgr.add_job("over-limit", "every 5 minutes", task)

    def test_invalid_schedule_raises_value_error(self, mgr: SchedulerManager) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            mgr.add_job("bad-sched", "every blue moon", "task")

    def test_invalid_schedule_does_not_add_job(self, mgr: SchedulerManager) -> None:
        with contextlib.suppress(ValueError):
            mgr.add_job("bad", "hourly", "task")
        assert len(mgr.list_jobs()) == 0


class TestManagerPauseResumePersistence:
    """pause/resume state is correctly written to and reloaded from disk."""

    def test_pause_persisted_to_json(self, jobs_path: str) -> None:
        m = SchedulerManager(jobs_file=jobs_path)
        m.start()
        job = m.add_job("p", "every 5 minutes", "t")
        m.pause_job(job.id)
        m.stop()

        data = json.loads(Path(jobs_path).read_text(encoding="utf-8"))
        assert data[0]["enabled"] is False

    def test_resume_persisted_to_json(self, jobs_path: str) -> None:
        m = SchedulerManager(jobs_file=jobs_path)
        m.start()
        job = m.add_job("r", "every 5 minutes", "t")
        m.pause_job(job.id)
        m.resume_job(job.id)
        m.stop()

        data = json.loads(Path(jobs_path).read_text(encoding="utf-8"))
        assert data[0]["enabled"] is True

    def test_paused_job_not_in_apscheduler_on_restart(self, jobs_path: str) -> None:
        m1 = SchedulerManager(jobs_file=jobs_path)
        m1.start()
        job = m1.add_job("noschedule", "every 5 minutes", "t")
        m1.pause_job(job.id)
        m1.stop()

        m2 = SchedulerManager(jobs_file=jobs_path)
        m2.start()
        ap_job = m2._scheduler.get_job(job.id)
        m2.stop()

        assert ap_job is None, "Paused job must not be registered with APScheduler on reload"


class TestManagerRemoveAllJobs:
    """After removing every job the list is empty and the file holds []."""

    def test_remove_all_list_empty(self, mgr: SchedulerManager) -> None:
        jobs = [mgr.add_job(f"x{i}", "every 5 minutes", "t") for i in range(3)]
        for j in jobs:
            mgr.remove_job(j.id)
        assert mgr.list_jobs() == []

    def test_remove_all_file_empty_array(self, mgr: SchedulerManager, jobs_path: str) -> None:
        jobs = [mgr.add_job(f"y{i}", "every 5 minutes", "t") for i in range(3)]
        for j in jobs:
            mgr.remove_job(j.id)
        data = json.loads(Path(jobs_path).read_text(encoding="utf-8"))
        assert data == []

    def test_remove_then_add_works(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job("first", "every 5 minutes", "t")
        mgr.remove_job(job.id)
        new_job = mgr.add_job("second", "daily at 10:00", "t")
        assert len(mgr.list_jobs()) == 1
        assert mgr.list_jobs()[0].id == new_job.id


class TestManagerRunJobActiveHours:
    """Active-hours gate in _run_job is respected."""

    def test_run_job_outside_window_skips_agent(self, mgr: SchedulerManager) -> None:
        """A narrow 3 AM window is almost certainly not active now."""
        job = mgr.add_job("gated", "every 5 minutes", "t", active_hours="03:00-03:01")
        now = datetime.datetime.now()
        if now.hour == 3 and now.minute <= 1:
            pytest.skip("Running during the narrow active window")

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            mgr._run_job(job.id)
            MockRuntime.assert_not_called()

        assert mgr._jobs[job.id].run_count == 0

    def test_run_job_no_active_hours_calls_agent(self, mgr: SchedulerManager) -> None:
        """Jobs with no active_hours restriction always call the agent."""
        job = mgr.add_job("open", "every 5 minutes", "t")
        assert job.active_hours == ""

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            mock_agent = MagicMock()
            mock_agent.run.return_value = "done"
            MockRuntime.return_value = mock_agent
            mgr._run_job(job.id)

        assert mgr._jobs[job.id].run_count == 1


class TestManagerRunJobFailureTracking:
    """_run_job correctly increments and resets failure counters."""

    @patch("missy.scheduler.manager.uuid")
    def test_failure_increments_consecutive_failures(
        self, mock_uuid, mgr: SchedulerManager
    ) -> None:
        mock_uuid.uuid4.return_value = "sess"
        job = mgr.add_job("fail", "every 5 minutes", "t")

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            MockRuntime.return_value.run.side_effect = RuntimeError("down")
            mgr._run_job(job.id)

        assert mgr._jobs[job.id].consecutive_failures == 1
        assert "down" in mgr._jobs[job.id].last_error

    @patch("missy.scheduler.manager.uuid")
    def test_success_resets_consecutive_failures(self, mock_uuid, mgr: SchedulerManager) -> None:
        mock_uuid.uuid4.return_value = "sess"
        job = mgr.add_job("recover", "every 5 minutes", "t")
        job.consecutive_failures = 2  # simulate prior failures

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            MockRuntime.return_value.run.return_value = "ok"
            mgr._run_job(job.id)

        assert mgr._jobs[job.id].consecutive_failures == 0
        assert mgr._jobs[job.id].run_count == 1

    @patch("missy.scheduler.manager.uuid")
    def test_last_result_updated_on_success(self, mock_uuid, mgr: SchedulerManager) -> None:
        mock_uuid.uuid4.return_value = "sess"
        job = mgr.add_job("result-check", "every 5 minutes", "t")

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            MockRuntime.return_value.run.return_value = "The answer"
            mgr._run_job(job.id)

        assert mgr._jobs[job.id].last_result == "The answer"


class TestManagerRunJobRetryScheduling:
    """_run_job interacts with APScheduler for retry scheduling."""

    @patch("missy.scheduler.manager.uuid")
    def test_retry_job_registered_after_first_failure(
        self, mock_uuid, mgr: SchedulerManager
    ) -> None:
        mock_uuid.uuid4.return_value = "sess"
        job = mgr.add_job("retry-me", "every 5 minutes", "t", max_attempts=3)

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            MockRuntime.return_value.run.side_effect = RuntimeError("error")
            mgr._run_job(job.id)

        retry_job = mgr._scheduler.get_job(f"{job.id}_retry_1")
        assert retry_job is not None, "A retry job must be registered after the first failure"

    @patch("missy.scheduler.manager.uuid")
    def test_no_retry_after_max_attempts_exhausted(self, mock_uuid, mgr: SchedulerManager) -> None:
        mock_uuid.uuid4.return_value = "sess"
        job = mgr.add_job("no-retry", "every 5 minutes", "t", max_attempts=1)
        job.consecutive_failures = 1  # already at max

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            MockRuntime.return_value.run.side_effect = RuntimeError("error")
            mgr._run_job(job.id)

        retry_job = mgr._scheduler.get_job(f"{job.id}_retry_2")
        assert retry_job is None, "No retry must be scheduled when max_attempts is exhausted"


class TestManagerRunJobDeleteAfterRun:
    """delete_after_run removes the job on first success."""

    @patch("missy.scheduler.manager.uuid")
    def test_oneshot_removed_after_success(self, mock_uuid, mgr: SchedulerManager) -> None:
        mock_uuid.uuid4.return_value = "sess"
        job = mgr.add_job("oneshot", "every 5 minutes", "t", delete_after_run=True)

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            MockRuntime.return_value.run.return_value = "done"
            mgr._run_job(job.id)

        assert job.id not in mgr._jobs

    @patch("missy.scheduler.manager.uuid")
    def test_non_oneshot_not_removed_after_success(self, mock_uuid, mgr: SchedulerManager) -> None:
        mock_uuid.uuid4.return_value = "sess"
        job = mgr.add_job("recurring", "every 5 minutes", "t", delete_after_run=False)

        with patch("missy.agent.runtime.AgentRuntime") as MockRuntime:
            MockRuntime.return_value.run.return_value = "done"
            mgr._run_job(job.id)

        assert job.id in mgr._jobs


class TestManagerRunJobMissingId:
    """_run_job with an unknown job_id is a silent no-op."""

    def test_missing_job_id_does_not_raise(self, mgr: SchedulerManager) -> None:
        mgr._run_job("completely-unknown-id")  # must not raise

    def test_missing_job_id_does_not_alter_job_list(self, mgr: SchedulerManager) -> None:
        mgr.add_job("real", "every 5 minutes", "t")
        mgr._run_job("ghost-id")
        assert len(mgr.list_jobs()) == 1


class TestManagerErrorHandling:
    """Error wrapping and propagation in lifecycle methods."""

    def test_start_wraps_apscheduler_exception(self, jobs_path: str) -> None:
        m = SchedulerManager(jobs_file=jobs_path)
        m._scheduler.start = MagicMock(side_effect=RuntimeError("scheduler refused"))
        with pytest.raises(SchedulerError, match="Failed to start background scheduler"):
            m.start()

    def test_add_job_rollback_on_schedule_job_error(self, mgr: SchedulerManager) -> None:
        with (
            patch.object(mgr, "_schedule_job", side_effect=SchedulerError("APScheduler full")),
            pytest.raises(SchedulerError),
        ):
            mgr.add_job("will-fail", "every 5 minutes", "t")
        assert len(mgr.list_jobs()) == 0

    def test_remove_job_wraps_apscheduler_error(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job("r", "every 5 minutes", "t")
        mgr._scheduler.get_job = MagicMock(return_value=MagicMock())
        mgr._scheduler.remove_job = MagicMock(side_effect=RuntimeError("locked"))
        with pytest.raises(SchedulerError, match="Failed to remove APScheduler job"):
            mgr.remove_job(job.id)

    def test_pause_job_wraps_apscheduler_error(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job("p", "every 5 minutes", "t")
        mgr._scheduler.pause_job = MagicMock(side_effect=RuntimeError("no"))
        with pytest.raises(SchedulerError, match="Failed to pause job"):
            mgr.pause_job(job.id)

    def test_resume_job_wraps_apscheduler_error(self, mgr: SchedulerManager) -> None:
        job = mgr.add_job("r", "every 5 minutes", "t")
        mgr.pause_job(job.id)
        mgr._scheduler.resume_job = MagicMock(side_effect=RuntimeError("no"))
        with pytest.raises(SchedulerError, match="Failed to resume job"):
            mgr.resume_job(job.id)

    def test_schedule_job_wraps_parse_error(self, mgr: SchedulerManager) -> None:
        job = ScheduledJob(name="bad-sched", schedule="INVALID SCHEDULE", task="t")
        mgr._jobs[job.id] = job
        with pytest.raises(SchedulerError, match="invalid schedule"):
            mgr._schedule_job(job)

    def test_schedule_job_wraps_apscheduler_add_error(self, mgr: SchedulerManager) -> None:
        job = ScheduledJob(name="aps-fail", schedule="every 5 minutes", task="t")
        mgr._jobs[job.id] = job
        mgr._scheduler.add_job = MagicMock(side_effect=RuntimeError("quota"))
        with pytest.raises(SchedulerError, match="APScheduler failed to add job"):
            mgr._schedule_job(job)

    def test_emit_event_handles_broken_bus(self, mgr: SchedulerManager) -> None:
        with patch(
            "missy.scheduler.manager.event_bus.publish",
            side_effect=Exception("bus exploded"),
        ):
            mgr._emit_event(event_type="test.event", result="allow", detail={})
        # Must not raise


class TestManagerCleanupMemory:
    """cleanup_memory delegates correctly and handles errors gracefully."""

    def test_returns_zero_when_store_has_no_cleanup(self, mgr: SchedulerManager) -> None:
        mock_store = MagicMock(spec=[])  # no attributes
        fake_module = MagicMock()
        fake_module.MemoryStore.return_value = mock_store
        with patch.dict("sys.modules", {"missy.memory.store": fake_module}):
            result = mgr.cleanup_memory()
        assert result == 0

    def test_returns_count_from_store_cleanup(self, mgr: SchedulerManager) -> None:
        mock_store = MagicMock()
        mock_store.cleanup.return_value = 17
        fake_module = MagicMock()
        fake_module.MemoryStore.return_value = mock_store
        with patch.dict("sys.modules", {"missy.memory.store": fake_module}):
            result = mgr.cleanup_memory(older_than_days=7)
        assert result == 17

    def test_returns_zero_on_import_exception(self, mgr: SchedulerManager) -> None:
        with patch.dict("sys.modules", {"missy.memory.store": None}):
            result = mgr.cleanup_memory()
        assert result == 0


class TestManagerPersistenceRoundTrip:
    """Field values survive a full stop/start cycle."""

    def test_all_fields_round_trip(self, jobs_path: str) -> None:
        m1 = SchedulerManager(jobs_file=jobs_path)
        m1.start()
        orig = m1.add_job(
            name="round-trip",
            schedule="daily at 07:30",
            task="Morning brief",
            provider="ollama",
            description="Daily brief",
            max_attempts=2,
            backoff_seconds=[10, 30],
            retry_on=["timeout"],
            delete_after_run=False,
            active_hours="06:00-22:00",
            timezone="America/Chicago",
        )
        m1.stop()

        m2 = SchedulerManager(jobs_file=jobs_path)
        m2.start()
        loaded = m2.list_jobs()[0]
        m2.stop()

        assert loaded.id == orig.id
        assert loaded.name == "round-trip"
        assert loaded.schedule == "daily at 07:30"
        assert loaded.task == "Morning brief"
        assert loaded.provider == "ollama"
        assert loaded.description == "Daily brief"
        assert loaded.max_attempts == 2
        assert loaded.backoff_seconds == [10, 30]
        assert loaded.retry_on == ["timeout"]
        assert loaded.active_hours == "06:00-22:00"
        assert loaded.timezone == "America/Chicago"

    def test_multiple_jobs_round_trip(self, jobs_path: str) -> None:
        m1 = SchedulerManager(jobs_file=jobs_path)
        m1.start()
        j1 = m1.add_job("alpha", "every 10 minutes", "t1")
        j2 = m1.add_job("beta", "daily at 08:00", "t2")
        j3 = m1.add_job("gamma", "weekly on friday 17:00", "t3")
        m1.stop()

        m2 = SchedulerManager(jobs_file=jobs_path)
        m2.start()
        ids = {j.id for j in m2.list_jobs()}
        m2.stop()

        assert j1.id in ids
        assert j2.id in ids
        assert j3.id in ids
        assert len(ids) == 3


class TestManagerConcurrentSafety:
    """add_job must be safe when called from multiple threads simultaneously."""

    def test_concurrent_adds_produce_no_data_loss(self, mgr: SchedulerManager) -> None:
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

        threads = [threading.Thread(target=add_one, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert errors == [], f"Errors during concurrent adds: {errors}"
        assert len(job_ids) == 12
        assert len(set(job_ids)) == 12, "All job IDs must be unique"
