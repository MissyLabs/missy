"""Tests for missy.scheduler.jobs."""

from __future__ import annotations

from datetime import datetime

from missy.scheduler.jobs import VALID_CAPABILITY_MODES, ScheduledJob


class TestScheduledJobDefaults:
    """Tests for ScheduledJob default field values."""

    def test_id_is_uuid_string(self):
        job = ScheduledJob()
        assert isinstance(job.id, str)
        assert len(job.id) == 36  # UUID4 with hyphens

    def test_two_jobs_have_different_ids(self):
        j1 = ScheduledJob()
        j2 = ScheduledJob()
        assert j1.id != j2.id

    def test_default_provider_is_anthropic(self):
        job = ScheduledJob()
        assert job.provider == "anthropic"

    def test_default_enabled_is_true(self):
        job = ScheduledJob()
        assert job.enabled is True

    def test_default_run_count_is_zero(self):
        job = ScheduledJob()
        assert job.run_count == 0

    def test_optional_fields_default_to_none(self):
        job = ScheduledJob()
        assert job.last_run is None
        assert job.next_run is None
        assert job.last_result is None

    def test_default_capability_mode_is_safe_chat(self):
        """SR-2.1: unattended scheduled jobs must default to a restricted
        tool-access mode, not the same "full" access an interactive
        session gets -- an unattended job with a bad/compromised task
        string should have a smaller blast radius by default.
        """
        job = ScheduledJob()
        assert job.capability_mode == "safe-chat"


class TestScheduledJobToDict:
    """Tests for ScheduledJob.to_dict."""

    def test_to_dict_contains_all_keys(self):
        job = ScheduledJob(name="daily", schedule="daily at 09:00", task="Run report")
        d = job.to_dict()
        expected_keys = {
            "id",
            "name",
            "description",
            "schedule",
            "task",
            "provider",
            "enabled",
            "created_at",
            "last_run",
            "next_run",
            "run_count",
            "last_result",
            "max_attempts",
            "backoff_seconds",
            "retry_on",
            "consecutive_failures",
            "last_error",
            "delete_after_run",
            "active_hours",
            "timezone",
            "capability_mode",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_datetimes_are_isoformat(self):
        now = datetime(2025, 6, 15, 12, 0, 0)
        job = ScheduledJob(created_at=now, last_run=now)
        d = job.to_dict()
        assert d["created_at"] == now.isoformat()
        assert d["last_run"] == now.isoformat()

    def test_to_dict_none_datetimes(self):
        job = ScheduledJob()
        d = job.to_dict()
        assert d["last_run"] is None
        assert d["next_run"] is None

    def test_to_dict_values_match_attributes(self):
        job = ScheduledJob(
            name="test",
            task="Do X",
            schedule="every 10 minutes",
            provider="openai",
            enabled=False,
            run_count=5,
            last_result="done",
        )
        d = job.to_dict()
        assert d["name"] == "test"
        assert d["task"] == "Do X"
        assert d["schedule"] == "every 10 minutes"
        assert d["provider"] == "openai"
        assert d["enabled"] is False
        assert d["run_count"] == 5
        assert d["last_result"] == "done"


class TestScheduledJobFromDict:
    """Tests for ScheduledJob.from_dict."""

    def test_round_trip_preserves_all_fields(self):
        now = datetime(2025, 1, 1, 8, 30, 0)
        original = ScheduledJob(
            name="weekly digest",
            description="Weekly summary",
            schedule="weekly on Monday 09:00",
            task="Summarise the week",
            provider="anthropic",
            enabled=True,
            created_at=now,
            run_count=3,
            last_result="All good",
        )
        restored = ScheduledJob.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.schedule == original.schedule
        assert restored.task == original.task
        assert restored.provider == original.provider
        assert restored.enabled == original.enabled
        assert restored.run_count == original.run_count
        assert restored.last_result == original.last_result
        assert restored.created_at == original.created_at

    def test_from_dict_handles_missing_optional_fields(self):
        data = {"name": "minimal", "schedule": "daily at 06:00", "task": "Wake up"}
        job = ScheduledJob.from_dict(data)
        assert job.name == "minimal"
        assert job.last_run is None
        assert job.next_run is None
        assert job.run_count == 0
        assert job.provider == "anthropic"

    def test_from_dict_generates_id_when_absent(self):
        job = ScheduledJob.from_dict({"name": "no id"})
        assert isinstance(job.id, str)
        assert len(job.id) == 36

    def test_from_dict_parses_last_run_datetime(self):
        ts = "2025-03-15T10:30:00"
        job = ScheduledJob.from_dict({"last_run": ts})
        assert job.last_run == datetime(2025, 3, 15, 10, 30, 0)

    def test_from_dict_preserves_capability_mode(self):
        job = ScheduledJob.from_dict({"capability_mode": "full"})
        assert job.capability_mode == "full"

    def test_from_dict_legacy_record_missing_capability_mode_fails_closed(self):
        """SR-2.1 regression: a jobs.json entry written before this field
        existed must not silently resolve to "full" -- absence of an
        explicit capability_mode must not imply the most permissive
        option (fail closed).
        """
        legacy_record = {"name": "old job", "schedule": "daily at 09:00", "task": "do stuff"}
        assert "capability_mode" not in legacy_record
        job = ScheduledJob.from_dict(legacy_record)
        assert job.capability_mode == "safe-chat"

    def test_from_dict_unrecognized_capability_mode_falls_back_to_safe_chat(self):
        """A corrupted or tampered jobs.json with an invalid mode string
        must not be passed through unvalidated.
        """
        job = ScheduledJob.from_dict({"capability_mode": "root-access-please"})
        assert job.capability_mode == "safe-chat"

    def test_valid_capability_modes_excludes_discord(self):
        assert "discord" not in VALID_CAPABILITY_MODES
        assert set(VALID_CAPABILITY_MODES) == {"full", "safe-chat", "no-tools"}

    def test_quoted_string_false_enabled_stays_disabled(self):
        """Regression: bare bool(data.get("enabled", True)) treats any
        non-empty string as truthy, so a hand-edited jobs.json entry with
        "enabled": "false" (a quoted string, not a real YAML/JSON
        boolean) silently produced enabled=True -- the opposite of what
        a user editing the file to disable a job actually wrote. enabled
        is load-bearing for whether SchedulerManager actually fires the
        job (manager.py gates on `if not job.enabled`), so this silently
        re-enabled a job the user believed they'd turned off.
        """
        job = ScheduledJob.from_dict({"id": "x", "name": "test", "enabled": "false"})
        assert job.enabled is False

    def test_quoted_string_true_enabled_stays_enabled(self):
        job = ScheduledJob.from_dict({"id": "x", "name": "test", "enabled": "true"})
        assert job.enabled is True

    def test_quoted_string_false_delete_after_run_stays_false(self):
        """Same bug, same fix, for delete_after_run -- a string "false"
        was misread as True, so a job meant to run repeatedly would be
        silently deleted after its first run.
        """
        job = ScheduledJob.from_dict({"id": "x", "name": "test", "delete_after_run": "false"})
        assert job.delete_after_run is False
