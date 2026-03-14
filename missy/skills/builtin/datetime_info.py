"""Built-in skill: date, time, timezone, and system uptime.

Reports the current wall-clock time in ISO-8601 format, the local timezone
name, and the system uptime read from /proc/uptime.  No special permissions
are required because all information is derived from the standard library and
a world-readable kernel file.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from missy.skills.base import BaseSkill, SkillPermissions, SkillResult


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
    permissions = SkillPermissions()  # No special permissions required.

    def execute(self, **kwargs: Any) -> SkillResult:
        """Return a formatted date/time and uptime string.

        Returns:
            :class:`~missy.skills.base.SkillResult` whose ``output`` is a
            newline-separated set of key-value pairs.
        """
        now = datetime.datetime.now(tz=datetime.UTC).astimezone()
        tz_name = now.tzname() or str(now.utcoffset())

        info = {
            "datetime_utc": datetime.datetime.now(tz=datetime.UTC).isoformat(timespec="seconds"),
            "datetime_local": now.isoformat(timespec="seconds"),
            "timezone": tz_name,
            "uptime": _parse_uptime(),
        }
        lines = [f"{k}: {v}" for k, v in info.items()]
        return SkillResult(success=True, output="\n".join(lines))
