"""Built-in skill: date, time, timezone, and system uptime.

Reports the current wall-clock time in ISO-8601 format, the local timezone
name, and the system uptime read from /proc/uptime.  No special permissions
are required because all information is derived from the standard library and
a world-readable kernel file.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from missy.skills.base import BaseSkill, SkillPermissions, SkillResult, reject_unknown_arguments

_UTC_OFFSET_RE = re.compile(r"^([+-])(\d{2}):(\d{2})$")
_MAX_TIMEZONE_CHARS = 128


def _parse_uptime() -> str:
    """Read /proc/uptime and return a human-friendly uptime string.

    Returns:
        A string such as ``"3d 4h 12m 7s"`` or ``"unavailable"`` when
        /proc/uptime cannot be read (e.g. on non-Linux platforms).
    """
    uptime_path = Path("/proc/uptime")
    try:
        raw = uptime_path.read_text(encoding="ascii")
        total_seconds = float(raw.split()[0])
    except Exception:
        return "unavailable"

    total = int(total_seconds)
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


class DateTimeSkill(BaseSkill):
    """Reports current date, time, timezone, and system uptime."""

    name = "datetime_info"
    description = "Report current date, time, timezone, and uptime."
    version = "1.0.0"
    permissions = SkillPermissions(filesystem_read=True)  # Reads /proc/uptime.

    def __init__(self, clock: Callable[[], datetime.datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.datetime.now(tz=datetime.UTC))

    def execute(self, timezone: str = "local", **kwargs: Any) -> SkillResult:
        """Return a formatted date/time and uptime string.

        Returns:
            :class:`~missy.skills.base.SkillResult` whose ``output`` is a
            newline-separated set of key-value pairs.
        """
        if error := reject_unknown_arguments(kwargs):
            return error
        try:
            current = self._clock()
        except Exception:
            return SkillResult(success=False, output=None, error="Current time is unavailable")
        if not isinstance(current, datetime.datetime) or current.tzinfo is None:
            return SkillResult(
                success=False,
                output=None,
                error="Clock must return a timezone-aware datetime",
            )
        target_timezone = _parse_timezone(timezone, current)
        if isinstance(target_timezone, str):
            return SkillResult(success=False, output=None, error=target_timezone)
        now_utc = current.astimezone(datetime.UTC)
        now = now_utc.astimezone(target_timezone)
        tz_name = getattr(target_timezone, "key", None) or now.tzname() or str(now.utcoffset())

        info = {
            "datetime_utc": now_utc.isoformat(timespec="seconds"),
            "datetime_local": now.isoformat(timespec="seconds"),
            "timezone": tz_name,
            "uptime": _parse_uptime(),
        }
        lines = [f"{k}: {v}" for k, v in info.items()]
        return SkillResult(success=True, output="\n".join(lines))


def _parse_timezone(value: object, current: datetime.datetime) -> datetime.tzinfo | str:
    if not isinstance(value, str):
        return "timezone must be a string"
    if (
        not value
        or len(value) > _MAX_TIMEZONE_CHARS
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return "timezone is invalid"
    if value == "local":
        return current.astimezone().tzinfo or datetime.UTC
    if value in {"UTC", "Z", "+00:00", "-00:00"}:
        return datetime.UTC
    if match := _UTC_OFFSET_RE.fullmatch(value):
        hours = int(match[2])
        minutes = int(match[3])
        if hours > 14 or minutes > 59 or (hours == 14 and minutes != 0):
            return "timezone UTC offset is out of range"
        offset = datetime.timedelta(hours=hours, minutes=minutes)
        if match[1] == "-":
            offset = -offset
        return datetime.timezone(offset)
    if "/" not in value or value.startswith(("/", ".")) or ".." in value:
        return "timezone must be local, UTC, an explicit UTC offset, or an IANA zone"
    try:
        return ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return "timezone is not a valid IANA zone"
