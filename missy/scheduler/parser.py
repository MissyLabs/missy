"""Parse human-readable schedule strings to APScheduler trigger configurations.

Supported formats
-----------------
Interval triggers
    - ``"every 5 minutes"``
    - ``"every 2 hours"``
    - ``"every 30 seconds"``

Cron triggers
    - ``"daily at 09:00"``
    - ``"weekly on Monday 09:00"``
    - ``"weekly on friday 14:30"``
"""

from __future__ import annotations

import re
from typing import Any

# Maps day-of-week full names (and common variants) to APScheduler abbreviations.
_DAY_MAP: dict[str, str] = {
    "monday": "mon",
    "tuesday": "tue",
    "wednesday": "wed",
    "thursday": "thu",
    "friday": "fri",
    "saturday": "sat",
    "sunday": "sun",
}

# ---------------------------------------------------------------------------
# Pattern registry
# Each entry is (compiled_pattern, handler_function).
# The first matching pattern wins.
# ---------------------------------------------------------------------------

_INTERVAL_PATTERN = re.compile(
    r"^every\s+(?P<value>\d+)\s+(?P<unit>seconds?|minutes?|hours?)$",
    re.IGNORECASE,
)

_DAILY_PATTERN = re.compile(
    r"^daily\s+at\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})$",
    re.IGNORECASE,
)

_WEEKLY_PATTERN = re.compile(
    r"^weekly\s+on\s+(?P<day>\w+)\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})$",
    re.IGNORECASE,
)


def _parse_interval(match: re.Match) -> dict[str, Any]:
    """Build an APScheduler interval trigger config from a regex match.

    Args:
        match: A match from :data:`_INTERVAL_PATTERN`.

    Returns:
        A dict suitable for unpacking into APScheduler's ``add_job`` call.
    """
    value = int(match.group("value"))
    raw_unit = match.group("unit").lower().rstrip("s")  # lower first, then strip trailing 's'
    unit_map = {"second": "seconds", "minute": "minutes", "hour": "hours"}
    unit = unit_map.get(raw_unit)
    if unit is None:
        raise ValueError(f"Unrecognised time unit: {match.group('unit')!r}")
    return {"trigger": "interval", unit: value}


def _parse_daily(match: re.Match) -> dict[str, Any]:
    """Build an APScheduler cron trigger config for a daily schedule.

    Args:
        match: A match from :data:`_DAILY_PATTERN`.

    Returns:
        A dict suitable for unpacking into APScheduler's ``add_job`` call.
    """
    return {
        "trigger": "cron",
        "hour": int(match.group("hour")),
        "minute": int(match.group("minute")),
    }


def _parse_weekly(match: re.Match) -> dict[str, Any]:
    """Build an APScheduler cron trigger config for a weekly schedule.

    Args:
        match: A match from :data:`_WEEKLY_PATTERN`.

    Returns:
        A dict suitable for unpacking into APScheduler's ``add_job`` call.

    Raises:
        ValueError: When the day-of-week string is not recognised.
    """
    raw_day = match.group("day").lower()
    day_of_week = _DAY_MAP.get(raw_day)
    if day_of_week is None:
        raise ValueError(
            f"Unrecognised day of week: {match.group('day')!r}. "
            f"Expected one of: {', '.join(_DAY_MAP)}."
        )
    return {
        "trigger": "cron",
        "day_of_week": day_of_week,
        "hour": int(match.group("hour")),
        "minute": int(match.group("minute")),
    }


def parse_schedule(schedule_str: str) -> dict[str, Any]:
    """Parse a human-readable schedule string to an APScheduler trigger config.

    Recognised formats::

        "every 5 minutes"          -> {"trigger": "interval", "minutes": 5}
        "every 2 hours"            -> {"trigger": "interval", "hours": 2}
        "every 30 seconds"         -> {"trigger": "interval", "seconds": 30}
        "daily at 09:00"           -> {"trigger": "cron", "hour": 9, "minute": 0}
        "weekly on Monday 09:00"   -> {"trigger": "cron", "day_of_week": "mon",
                                       "hour": 9, "minute": 0}
        "weekly on friday 14:30"   -> {"trigger": "cron", "day_of_week": "fri",
                                       "hour": 14, "minute": 30}

    The returned dictionary's ``"trigger"`` key should be passed as the
    ``trigger`` argument to APScheduler's ``add_job``, and the remaining
    keys are forwarded as keyword arguments to the trigger constructor.

    Args:
        schedule_str: Human-readable schedule description.

    Returns:
        A dictionary with ``"trigger"`` plus trigger-specific keyword
        arguments suitable for APScheduler's ``add_job``.

    Raises:
        ValueError: When *schedule_str* does not match any known pattern.
    """
    cleaned = schedule_str.strip()

    match = _INTERVAL_PATTERN.match(cleaned)
    if match:
        return _parse_interval(match)

    match = _DAILY_PATTERN.match(cleaned)
    if match:
        return _parse_daily(match)

    match = _WEEKLY_PATTERN.match(cleaned)
    if match:
        return _parse_weekly(match)

    raise ValueError(
        f"Unrecognised schedule string: {schedule_str!r}. "
        "Supported formats: "
        "'every N seconds/minutes/hours', "
        "'daily at HH:MM', "
        "'weekly on <day> HH:MM'."
    )
