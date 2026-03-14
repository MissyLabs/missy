"""Scheduled job dataclass for the Missy scheduler subsystem."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ScheduledJob:
    """Represents a single persisted scheduled task.

    Attributes:
        id: Unique job identifier (UUID string).
        name: Human-readable name for the job.
        description: Optional description of what the job does.
        schedule: Human-readable schedule string, e.g. ``"every 5 minutes"``.
        task: The prompt or task text the agent will receive when the job fires.
        provider: Name of the AI provider to use when running the task.
        enabled: When ``False`` the job will not fire even if scheduled.
        created_at: UTC timestamp of job creation.
        last_run: UTC timestamp of the most recent execution, or ``None``.
        next_run: UTC timestamp of the next scheduled execution, or ``None``.
        run_count: Total number of times the job has been executed.
        last_result: The agent's response from the most recent execution.
        max_attempts: Maximum consecutive failures before giving up (default 3).
        backoff_seconds: Delay in seconds between retry attempts.
        retry_on: Error category tags that trigger a retry.
        consecutive_failures: Number of failures since last success.
        last_error: String representation of the most recent exception.
        delete_after_run: When ``True`` the job is removed after one success.
        active_hours: ``"HH:MM-HH:MM"`` window outside which the job is skipped.
        timezone: IANA timezone string for this job (e.g. ``"America/New_York"``).
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    schedule: str = ""
    task: str = ""
    provider: str = "anthropic"
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    last_run: datetime | None = None
    next_run: datetime | None = None
    run_count: int = 0
    last_result: str | None = None

    # ------------------------------------------------------------------
    # Retry configuration
    # ------------------------------------------------------------------
    max_attempts: int = 3
    backoff_seconds: list = field(default_factory=lambda: [30, 60, 300])
    retry_on: list = field(default_factory=lambda: ["network", "provider_error"])
    consecutive_failures: int = 0
    last_error: str = ""

    # ------------------------------------------------------------------
    # One-shot behaviour
    # ------------------------------------------------------------------
    delete_after_run: bool = False

    # ------------------------------------------------------------------
    # Active-hours window
    # ------------------------------------------------------------------
    active_hours: str = ""  # empty = always active

    # ------------------------------------------------------------------
    # Timezone
    # ------------------------------------------------------------------
    timezone: str = ""  # IANA timezone string, empty = system/UTC

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def should_run_now(self) -> bool:
        """Check whether the current local time falls within :attr:`active_hours`.

        Returns:
            ``True`` when no active-hours restriction is set or when the
            current time is within the configured window (inclusive).
            ``False`` when the current time is outside the window.

        The window format is ``"HH:MM-HH:MM"``.  Overnight windows (where
        end time is earlier than start time) are handled correctly — e.g.
        ``"22:00-06:00"`` means "from 10 PM until 6 AM".
        """
        if not self.active_hours:
            return True

        m = re.match(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})", self.active_hours)
        if not m:
            return True

        now = datetime.now()
        start = now.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
        end = now.replace(hour=int(m.group(3)), minute=int(m.group(4)), second=0, microsecond=0)

        if end < start:  # overnight window (e.g. 22:00-06:00)
            return now >= start or now <= end

        return start <= now <= end

    def should_retry(self, error: str) -> bool:  # noqa: ARG002
        """Return ``True`` if this job should be retried after *error*.

        Retry is allowed when :attr:`consecutive_failures` has not yet reached
        :attr:`max_attempts`.  The *error* string is accepted for API
        compatibility but is not currently used to gate retry decisions —
        all error types are retried up to the maximum.

        Args:
            error: String representation of the exception that caused the failure.

        Returns:
            ``True`` when a retry should be attempted, ``False`` otherwise.
        """
        if self.consecutive_failures >= self.max_attempts:
            return False
        return True

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the job to a JSON-compatible dictionary.

        Returns:
            A dictionary with all fields; datetime values are represented
            as ISO-8601 strings, and ``None`` values are preserved as-is.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "schedule": self.schedule,
            "task": self.task,
            "provider": self.provider,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "last_result": self.last_result,
            # Retry fields
            "max_attempts": self.max_attempts,
            "backoff_seconds": self.backoff_seconds,
            "retry_on": self.retry_on,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            # One-shot
            "delete_after_run": self.delete_after_run,
            # Active hours
            "active_hours": self.active_hours,
            # Timezone
            "timezone": self.timezone,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduledJob:
        """Deserialise a job from a dictionary previously produced by :meth:`to_dict`.

        Missing keys fall back to safe defaults so that existing ``jobs.json``
        files produced by older versions of the software continue to load
        without error.

        Args:
            data: Mapping with job fields.  ISO-8601 datetime strings are
                parsed back to :class:`datetime` objects.

        Returns:
            A new :class:`ScheduledJob` instance.
        """

        def _parse_dt(value: str | None) -> datetime | None:
            if value is None:
                return None
            return datetime.fromisoformat(value)

        return cls(
            id=str(data.get("id", str(uuid.uuid4()))),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            schedule=str(data.get("schedule", "")),
            task=str(data.get("task", "")),
            provider=str(data.get("provider", "anthropic")),
            enabled=bool(data.get("enabled", True)),
            created_at=_parse_dt(data.get("created_at")) or datetime.now(tz=UTC),
            last_run=_parse_dt(data.get("last_run")),
            next_run=_parse_dt(data.get("next_run")),
            run_count=int(data.get("run_count", 0)),
            last_result=data.get("last_result"),
            # Retry fields — safe defaults for legacy records
            max_attempts=int(data.get("max_attempts", 3)),
            backoff_seconds=list(data.get("backoff_seconds", [30, 60, 300])),
            retry_on=list(data.get("retry_on", ["network", "provider_error"])),
            consecutive_failures=int(data.get("consecutive_failures", 0)),
            last_error=str(data.get("last_error", "")),
            # One-shot
            delete_after_run=bool(data.get("delete_after_run", False)),
            # Active hours
            active_hours=str(data.get("active_hours", "")),
            # Timezone
            timezone=str(data.get("timezone", "")),
        )
