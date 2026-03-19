"""Edge case tests for scheduler parser.


Covers:
- Human-readable schedule parsing: intervals, cron, daily, weekly
- Boundary cases: 0 interval, very large intervals, malformed input
"""

from __future__ import annotations

import pytest

from missy.scheduler.parser import parse_schedule


class TestScheduleParser:
    """Tests for the schedule parser return structures."""

    def test_every_5_minutes(self):
        result = parse_schedule("every 5 minutes")
        assert result["trigger"] == "interval"
        assert result["minutes"] == 5

    def test_every_1_hour_shorthand(self):
        result = parse_schedule("every 1 hours")
        assert result["trigger"] == "interval"

    def test_every_1_day(self):
        result = parse_schedule("every 24 hours")
        assert result["trigger"] == "interval"

    def test_every_30_seconds(self):
        result = parse_schedule("every 30 seconds")
        assert result["trigger"] == "interval"
        assert result["seconds"] == 30

    def test_cron_expression_passthrough(self):
        """Valid cron expressions should return cron trigger."""
        result = parse_schedule("0 */2 * * *")
        assert result["trigger"] == "cron"

    def test_nonsense_input_raises(self):
        with pytest.raises(ValueError):
            parse_schedule("flibbertigibbet")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_schedule("")

    def test_every_1_minute(self):
        result = parse_schedule("every 1 minutes")
        assert result["trigger"] == "interval"
        assert result["minutes"] == 1

    def test_weekly_on_monday(self):
        result = parse_schedule("weekly on monday at 09:00")
        assert result["trigger"] in ("cron", "interval")

    def test_standard_cron_5_fields(self):
        result = parse_schedule("* * * * *")
        assert result["trigger"] == "cron"

    def test_cron_with_ranges(self):
        result = parse_schedule("0 9-17 * * 1-5")
        assert result["trigger"] == "cron"

    def test_numeric_only_raises(self):
        with pytest.raises(ValueError):
            parse_schedule("42")


class TestScheduleParserBoundaries:
    """Boundary tests for schedule parser."""

    def test_every_1_second(self):
        result = parse_schedule("every 1 second")
        assert result["trigger"] == "interval"
        assert result["seconds"] == 1

    def test_every_negative_raises(self):
        with pytest.raises(ValueError):
            parse_schedule("every -5 minutes")

    def test_every_large_interval(self):
        result = parse_schedule("every 1440 minutes")
        assert result["trigger"] == "interval"
        assert result["minutes"] == 1440

    def test_every_1_hour(self):
        result = parse_schedule("every 1 hour")
        assert result["trigger"] == "interval"

    def test_every_24_hours(self):
        result = parse_schedule("every 24 hours")
        assert result["trigger"] == "interval"
