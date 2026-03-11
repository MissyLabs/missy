"""Tests for missy.scheduler.parser."""

from __future__ import annotations

import pytest

from missy.scheduler.parser import parse_schedule


class TestParseScheduleInterval:
    """Tests for 'every N unit' interval patterns."""

    def test_every_5_minutes(self):
        result = parse_schedule("every 5 minutes")
        assert result == {"trigger": "interval", "minutes": 5}

    def test_every_1_minute_singular(self):
        result = parse_schedule("every 1 minute")
        assert result == {"trigger": "interval", "minutes": 1}

    def test_every_2_hours(self):
        result = parse_schedule("every 2 hours")
        assert result == {"trigger": "interval", "hours": 2}

    def test_every_1_hour_singular(self):
        result = parse_schedule("every 1 hour")
        assert result == {"trigger": "interval", "hours": 1}

    def test_every_30_seconds(self):
        result = parse_schedule("every 30 seconds")
        assert result == {"trigger": "interval", "seconds": 30}

    def test_every_1_second_singular(self):
        result = parse_schedule("every 1 second")
        assert result == {"trigger": "interval", "seconds": 1}

    def test_every_is_case_insensitive(self):
        result = parse_schedule("EVERY 10 MINUTES")
        assert result == {"trigger": "interval", "minutes": 10}

    def test_leading_trailing_whitespace_stripped(self):
        result = parse_schedule("  every 5 minutes  ")
        assert result == {"trigger": "interval", "minutes": 5}

    def test_large_interval_value(self):
        result = parse_schedule("every 999 hours")
        assert result == {"trigger": "interval", "hours": 999}


class TestParseScheduleDaily:
    """Tests for 'daily at HH:MM' cron patterns."""

    def test_daily_at_zero_zero(self):
        result = parse_schedule("daily at 00:00")
        assert result == {"trigger": "cron", "hour": 0, "minute": 0}

    def test_daily_at_09_00(self):
        result = parse_schedule("daily at 09:00")
        assert result == {"trigger": "cron", "hour": 9, "minute": 0}

    def test_daily_at_23_59(self):
        result = parse_schedule("daily at 23:59")
        assert result == {"trigger": "cron", "hour": 23, "minute": 59}

    def test_daily_case_insensitive(self):
        result = parse_schedule("DAILY AT 12:30")
        assert result == {"trigger": "cron", "hour": 12, "minute": 30}

    def test_daily_single_digit_hour(self):
        result = parse_schedule("daily at 9:05")
        assert result == {"trigger": "cron", "hour": 9, "minute": 5}


class TestParseScheduleWeekly:
    """Tests for 'weekly on <day> HH:MM' cron patterns."""

    @pytest.mark.parametrize(
        "day_input, expected_dow",
        [
            ("Monday", "mon"),
            ("tuesday", "tue"),
            ("WEDNESDAY", "wed"),
            ("thursday", "thu"),
            ("Friday", "fri"),
            ("saturday", "sat"),
            ("sunday", "sun"),
        ],
    )
    def test_all_days_of_week(self, day_input: str, expected_dow: str):
        result = parse_schedule(f"weekly on {day_input} 10:00")
        assert result["trigger"] == "cron"
        assert result["day_of_week"] == expected_dow
        assert result["hour"] == 10
        assert result["minute"] == 0

    def test_weekly_friday_14_30(self):
        result = parse_schedule("weekly on friday 14:30")
        assert result == {"trigger": "cron", "day_of_week": "fri", "hour": 14, "minute": 30}

    def test_weekly_monday_09_00(self):
        result = parse_schedule("weekly on Monday 09:00")
        assert result == {"trigger": "cron", "day_of_week": "mon", "hour": 9, "minute": 0}

    def test_weekly_case_insensitive(self):
        result = parse_schedule("WEEKLY ON FRIDAY 08:00")
        assert result["day_of_week"] == "fri"

    def test_weekly_unknown_day_raises(self):
        with pytest.raises(ValueError, match="Unrecognised day of week"):
            parse_schedule("weekly on Flunday 09:00")


class TestParseScheduleErrors:
    """Tests for unrecognised schedule strings."""

    @pytest.mark.parametrize(
        "bad_schedule",
        [
            "every blue moon",
            "once a day",
            "hourly",
            "",
            "daily",
            "weekly",
            "every minutes 5",
            "at 09:00",
        ],
    )
    def test_unrecognised_raises_value_error(self, bad_schedule: str):
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            parse_schedule(bad_schedule)
