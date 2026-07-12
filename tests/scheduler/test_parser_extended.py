"""Extended parser tests covering raw cron, at (one-shot), timezone, and weekly-with-at."""

from __future__ import annotations

from missy.scheduler.parser import convert_crontab_dow_to_apscheduler, parse_schedule


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


class TestConvertCrontabDowToApscheduler:
    """Regression: standard crontab numbers day-of-week Sunday=0..Saturday=6,
    but APScheduler's day_of_week field follows date.weekday()'s convention
    (Monday=0..Sunday=6). Passing a raw numeric crontab day-of-week field
    straight through to APScheduler (as the manager previously did via
    CronTrigger.from_crontab) silently produces a *different, valid*
    schedule instead of an error -- e.g. crontab's "1-5" ("weekdays")
    actually fired Tuesday-Saturday under APScheduler's own numbering.
    """

    def test_wildcard_passthrough(self):
        assert convert_crontab_dow_to_apscheduler("*") == "*"

    def test_single_digit_monday(self):
        # crontab Monday=1 -> APScheduler Monday=0
        assert convert_crontab_dow_to_apscheduler("1") == "0"

    def test_single_digit_sunday(self):
        # crontab Sunday=0 -> APScheduler Sunday=6
        assert convert_crontab_dow_to_apscheduler("0") == "6"

    def test_single_digit_saturday(self):
        # crontab Saturday=6 -> APScheduler Saturday=5
        assert convert_crontab_dow_to_apscheduler("6") == "5"

    def test_sunday_alias_seven(self):
        # crontab allows 7 as an alias for Sunday
        assert convert_crontab_dow_to_apscheduler("7") == "6"

    def test_weekdays_range(self):
        # crontab "1-5" (Mon-Fri) -> APScheduler "0-4" (Mon-Fri)
        assert convert_crontab_dow_to_apscheduler("1-5") == "0-4"

    def test_wraparound_range_splits_into_two(self):
        # crontab "5-1" (Fri,Sat,Sun,Mon) -> APScheduler "4-6,0-0"
        assert convert_crontab_dow_to_apscheduler("5-1") == "4-6,0-0"

    def test_comma_list(self):
        # crontab "0,6" (Sun,Sat weekend) -> APScheduler "6,5"
        assert convert_crontab_dow_to_apscheduler("0,6") == "6,5"

    def test_step_expression(self):
        assert convert_crontab_dow_to_apscheduler("*/2") == "*/2"

    def test_day_name_range_passthrough(self):
        # Day names carry no numbering ambiguity; left untouched.
        assert convert_crontab_dow_to_apscheduler("mon-fri") == "mon-fri"

    def test_day_name_passthrough(self):
        assert convert_crontab_dow_to_apscheduler("sun") == "sun"


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
