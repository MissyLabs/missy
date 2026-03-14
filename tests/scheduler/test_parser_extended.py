"""Extended parser tests covering raw cron, at (one-shot), timezone, and weekly-with-at."""

from __future__ import annotations

from missy.scheduler.parser import parse_schedule


class TestRawCronExpression:
    def test_five_field_cron(self):
        result = parse_schedule("*/5 * * * *")
        assert result["trigger"] == "cron"
        assert result["_cron_expression"] == "*/5 * * * *"

    def test_five_field_weekdays(self):
        result = parse_schedule("0 9 * * 1-5")
        assert result["trigger"] == "cron"
        assert result["_cron_expression"] == "0 9 * * 1-5"

    def test_six_field_cron(self):
        result = parse_schedule("30 8 * * 1 *")
        assert result["trigger"] == "cron"
        assert result["_cron_expression"] == "30 8 * * 1 *"

    def test_cron_with_timezone(self):
        result = parse_schedule("*/10 * * * *", tz="US/Eastern")
        assert result["timezone"] == "US/Eastern"
        assert result["_cron_expression"] == "*/10 * * * *"


class TestAtOneShot:
    def test_at_with_space_separator(self):
        result = parse_schedule("at 2024-12-31 23:59")
        assert result == {"trigger": "date", "run_date": "2024-12-31T23:59"}

    def test_at_with_t_separator(self):
        result = parse_schedule("at 2024-12-31T23:59")
        assert result == {"trigger": "date", "run_date": "2024-12-31T23:59"}

    def test_at_case_insensitive(self):
        result = parse_schedule("AT 2025-06-15 08:00")
        assert result["trigger"] == "date"
        assert result["run_date"] == "2025-06-15T08:00"

    def test_at_with_timezone(self):
        result = parse_schedule("at 2025-01-01 00:00", tz="America/New_York")
        assert result["timezone"] == "America/New_York"


class TestTimezoneAttachment:
    def test_daily_with_timezone(self):
        result = parse_schedule("daily at 09:00", tz="Europe/London")
        assert result["timezone"] == "Europe/London"
        assert result["hour"] == 9

    def test_weekly_with_timezone(self):
        result = parse_schedule("weekly on friday at 14:30", tz="Asia/Tokyo")
        assert result["timezone"] == "Asia/Tokyo"
        assert result["day_of_week"] == "fri"

    def test_interval_no_timezone(self):
        result = parse_schedule("every 5 minutes", tz="US/Pacific")
        assert "timezone" not in result


class TestWeeklyWithAt:
    def test_weekly_on_day_at_time(self):
        result = parse_schedule("weekly on monday at 09:00")
        assert result["day_of_week"] == "mon"
        assert result["hour"] == 9

    def test_weekly_on_day_without_at(self):
        result = parse_schedule("weekly on tuesday 15:30")
        assert result["day_of_week"] == "tue"
        assert result["hour"] == 15
        assert result["minute"] == 30
