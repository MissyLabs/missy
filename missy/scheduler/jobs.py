"""Scheduled job dataclass for the Missy scheduler subsystem."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


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
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    schedule: str = ""
    task: str = ""
    provider: str = "anthropic"
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    last_result: Optional[str] = None

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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduledJob":
        """Deserialise a job from a dictionary previously produced by :meth:`to_dict`.

        Args:
            data: Mapping with job fields.  ISO-8601 datetime strings are
                parsed back to :class:`datetime` objects.

        Returns:
            A new :class:`ScheduledJob` instance.
        """

        def _parse_dt(value: Optional[str]) -> Optional[datetime]:
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
            created_at=_parse_dt(data.get("created_at")) or datetime.now(tz=timezone.utc),
            last_run=_parse_dt(data.get("last_run")),
            next_run=_parse_dt(data.get("next_run")),
            run_count=int(data.get("run_count", 0)),
            last_result=data.get("last_result"),
        )
