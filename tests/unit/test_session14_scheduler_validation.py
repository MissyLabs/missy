"""Tests for scheduler parser input validation (session 14)."""

from __future__ import annotations

import pytest

from missy.scheduler.parser import parse_schedule


class TestSchedulerTimeValidation:
    """Verify hour/minute range validation in schedule parser."""

    def test_valid_daily_schedule(self):
        result = parse_schedule("daily at 09:00")
        assert result["hour"] == 9
        assert result["minute"] == 0

    def test_valid_daily_23_59(self):
        result = parse_schedule("daily at 23:59")
        assert result["hour"] == 23
        assert result["minute"] == 59

    def test_daily_hour_too_high(self):
        with pytest.raises(ValueError, match="Hour must be 0-23"):
            parse_schedule("daily at 25:00")

    def test_daily_minute_too_high(self):
        with pytest.raises(ValueError, match="Minute must be 0-59"):
            parse_schedule("daily at 09:70")

    def test_weekly_hour_too_high(self):
        with pytest.raises(ValueError, match="Hour must be 0-23"):
            parse_schedule("weekly on Monday at 24:00")

    def test_weekly_minute_too_high(self):
        with pytest.raises(ValueError, match="Minute must be 0-59"):
            parse_schedule("weekly on Friday at 12:60")

    def test_valid_weekly_schedule(self):
        result = parse_schedule("weekly on Monday at 09:00")
        assert result["day_of_week"] == "mon"
        assert result["hour"] == 9

    def test_valid_midnight(self):
        result = parse_schedule("daily at 00:00")
        assert result["hour"] == 0
        assert result["minute"] == 0


class TestSchedulerIntervalValidation:
    """Verify interval value validation."""

    def test_valid_interval(self):
        result = parse_schedule("every 5 minutes")
        assert result["minutes"] == 5

    def test_zero_interval_rejected(self):
        # Zero doesn't match the regex (\d+ requires at least 1 digit > 0)
        # but the regex matches "0", so we need the validation
        with pytest.raises(ValueError, match="positive"):
            parse_schedule("every 0 minutes")

    def test_large_interval_accepted(self):
        result = parse_schedule("every 999 hours")
        assert result["hours"] == 999


class TestSchedulerTimezoneAttachment:
    """Verify timezone is attached to cron/date triggers."""

    def test_daily_with_timezone(self):
        result = parse_schedule("daily at 09:00", tz="America/New_York")
        assert result["timezone"] == "America/New_York"

    def test_weekly_with_timezone(self):
        result = parse_schedule("weekly on Monday at 09:00", tz="Europe/London")
        assert result["timezone"] == "Europe/London"

    def test_interval_no_timezone(self):
        result = parse_schedule("every 5 minutes", tz="UTC")
        assert "timezone" not in result

    def test_at_with_timezone(self):
        result = parse_schedule("at 2026-12-31 23:59", tz="Asia/Tokyo")
        assert result["timezone"] == "Asia/Tokyo"
