"""Scheduler manager using APScheduler.

:class:`SchedulerManager` owns the APScheduler :class:`BackgroundScheduler`
and provides a persistence layer on top of it via a JSON file at
``~/.missy/jobs.json``.  Every significant operation emits an audit event
through :data:`~missy.core.events.event_bus`.

Example::

    from missy.scheduler.manager import SchedulerManager

    mgr = SchedulerManager()
    mgr.start()
    job = mgr.add_job("Morning summary", "daily at 09:00", "Summarise today's agenda")
    mgr.stop()
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler

from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import SchedulerError
from missy.scheduler.jobs import ScheduledJob
from missy.scheduler.parser import parse_schedule

logger = logging.getLogger(__name__)


class SchedulerManager:
    """Manages scheduled jobs backed by APScheduler and a JSON persistence file.

    Jobs are persisted to *jobs_file* so that they survive process restarts.
    On :meth:`start` all previously persisted jobs are re-registered with the
    APScheduler background scheduler.

    Args:
        jobs_file: Path to the JSON file used for job persistence.  Tilde
            expansion is performed automatically.
    """

    def __init__(self, jobs_file: str = "~/.missy/jobs.json") -> None:
        self.jobs_file = Path(jobs_file).expanduser()
        self._scheduler: BackgroundScheduler = BackgroundScheduler()
        self._jobs: dict[str, ScheduledJob] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load persisted jobs, schedule them, and start the background scheduler.

        Raises:
            SchedulerError: When the scheduler fails to start.
        """
        self._load_jobs()
        for job in self._jobs.values():
            if job.enabled:
                self._schedule_job(job)

        try:
            self._scheduler.start()
        except Exception as exc:
            raise SchedulerError(f"Failed to start background scheduler: {exc}") from exc

        logger.info("SchedulerManager started with %d job(s).", len(self._jobs))
        self._emit_event(
            event_type="scheduler.start",
            result="allow",
            detail={"job_count": len(self._jobs)},
        )

    def stop(self) -> None:
        """Shut down the background scheduler gracefully.

        All in-progress jobs are allowed to finish.  This call blocks until
        the scheduler thread has exited.
        """
        try:
            self._scheduler.shutdown(wait=True)
        except Exception as exc:
            logger.warning("Error during scheduler shutdown: %s", exc)

        logger.info("SchedulerManager stopped.")
        self._emit_event(
            event_type="scheduler.stop",
            result="allow",
            detail={},
        )

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def add_job(
        self,
        name: str,
        schedule: str,
        task: str,
        provider: str = "anthropic",
    ) -> ScheduledJob:
        """Create a new job, persist it, and register it with APScheduler.

        Args:
            name: Human-readable job name.
            schedule: Human-readable schedule string (parsed by
                :func:`~missy.scheduler.parser.parse_schedule`).
            task: Prompt or task text to run when the job fires.
            provider: AI provider name to use for the run.

        Returns:
            The newly created :class:`ScheduledJob`.

        Raises:
            ValueError: When *schedule* cannot be parsed.
            SchedulerError: When APScheduler fails to register the job.
        """
        # Validate the schedule string before creating the job.
        try:
            parse_schedule(schedule)
        except ValueError as exc:
            self._emit_event(
                event_type="scheduler.job.add",
                result="error",
                detail={"name": name, "schedule": schedule, "error": str(exc)},
            )
            raise

        job = ScheduledJob(
            name=name,
            schedule=schedule,
            task=task,
            provider=provider,
        )
        self._jobs[job.id] = job

        try:
            self._schedule_job(job)
        except SchedulerError:
            # Roll back the in-memory entry so state stays consistent.
            del self._jobs[job.id]
            raise

        self._save_jobs()
        logger.info("Added job %r (id=%s, schedule=%r).", name, job.id, schedule)
        self._emit_event(
            event_type="scheduler.job.add",
            result="allow",
            detail={"job_id": job.id, "name": name, "schedule": schedule, "provider": provider},
        )
        return job

    def remove_job(self, job_id: str) -> None:
        """Remove a job from the scheduler and the persistence store.

        Args:
            job_id: The :attr:`~ScheduledJob.id` of the job to remove.

        Raises:
            KeyError: When no job with *job_id* exists.
            SchedulerError: When APScheduler fails to remove the job.
        """
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"No job found with id {job_id!r}.")

        try:
            if self._scheduler.get_job(job_id) is not None:
                self._scheduler.remove_job(job_id)
        except Exception as exc:
            raise SchedulerError(f"Failed to remove APScheduler job {job_id!r}: {exc}") from exc

        del self._jobs[job_id]
        self._save_jobs()
        logger.info("Removed job id=%s name=%r.", job_id, job.name)
        self._emit_event(
            event_type="scheduler.job.remove",
            result="allow",
            detail={"job_id": job_id, "name": job.name},
        )

    def pause_job(self, job_id: str) -> None:
        """Pause a job so it no longer fires until resumed.

        Args:
            job_id: The :attr:`~ScheduledJob.id` of the job to pause.

        Raises:
            KeyError: When no job with *job_id* exists.
            SchedulerError: When APScheduler fails to pause the job.
        """
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"No job found with id {job_id!r}.")

        try:
            self._scheduler.pause_job(job_id)
        except Exception as exc:
            raise SchedulerError(f"Failed to pause job {job_id!r}: {exc}") from exc

        job.enabled = False
        self._save_jobs()
        logger.info("Paused job id=%s name=%r.", job_id, job.name)
        self._emit_event(
            event_type="scheduler.job.pause",
            result="allow",
            detail={"job_id": job_id, "name": job.name},
        )

    def resume_job(self, job_id: str) -> None:
        """Resume a previously paused job.

        Args:
            job_id: The :attr:`~ScheduledJob.id` of the job to resume.

        Raises:
            KeyError: When no job with *job_id* exists.
            SchedulerError: When APScheduler fails to resume the job.
        """
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"No job found with id {job_id!r}.")

        try:
            self._scheduler.resume_job(job_id)
        except Exception as exc:
            raise SchedulerError(f"Failed to resume job {job_id!r}: {exc}") from exc

        job.enabled = True
        self._save_jobs()
        logger.info("Resumed job id=%s name=%r.", job_id, job.name)
        self._emit_event(
            event_type="scheduler.job.resume",
            result="allow",
            detail={"job_id": job_id, "name": job.name},
        )

    def list_jobs(self) -> list[ScheduledJob]:
        """Return a snapshot of all registered jobs in creation order.

        Returns:
            A new list of :class:`ScheduledJob` instances.
        """
        return list(self._jobs.values())

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _run_job(self, job_id: str) -> None:
        """Execute a scheduled job by running it through the agent runtime.

        This method is called by APScheduler on the background thread.  It
        resolves the :class:`~missy.agent.runtime.AgentRuntime`, runs the
        job's task string, stores the result, and emits audit events.

        Args:
            job_id: ID of the job to execute.
        """
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning("Scheduled job %r no longer exists; skipping.", job_id)
            return

        session_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())

        self._emit_event(
            event_type="scheduler.job.run.start",
            result="allow",
            detail={
                "job_id": job_id,
                "name": job.name,
                "provider": job.provider,
                "run_count": job.run_count,
            },
            session_id=session_id,
            task_id=task_id,
        )

        try:
            # Import lazily to avoid circular imports at module load time.
            from missy.agent.runtime import AgentConfig, AgentRuntime

            agent = AgentRuntime(AgentConfig(provider=job.provider))
            result_text = agent.run(job.task, session_id=session_id)
        except Exception as exc:
            logger.exception("Error executing scheduled job %r (id=%s).", job.name, job_id)
            job.last_run = datetime.now(tz=timezone.utc)
            job.last_result = f"ERROR: {exc}"
            self._save_jobs()
            self._emit_event(
                event_type="scheduler.job.run.error",
                result="error",
                detail={"job_id": job_id, "name": job.name, "error": str(exc)},
                session_id=session_id,
                task_id=task_id,
            )
            return

        job.last_run = datetime.now(tz=timezone.utc)
        job.run_count += 1
        job.last_result = result_text

        # Update next_run from APScheduler if available.
        ap_job = self._scheduler.get_job(job_id)
        if ap_job is not None and ap_job.next_run_time is not None:
            job.next_run = ap_job.next_run_time.replace(tzinfo=None)

        self._save_jobs()
        logger.info(
            "Completed scheduled job %r (id=%s, run_count=%d).",
            job.name,
            job_id,
            job.run_count,
        )
        self._emit_event(
            event_type="scheduler.job.run.complete",
            result="allow",
            detail={
                "job_id": job_id,
                "name": job.name,
                "run_count": job.run_count,
            },
            session_id=session_id,
            task_id=task_id,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_jobs(self) -> None:
        """Persist the current in-memory job list to the JSON file.

        The parent directory is created if it does not exist.  Errors are
        logged but not re-raised to avoid interrupting the scheduler thread.
        """
        try:
            self.jobs_file.parent.mkdir(parents=True, exist_ok=True)
            payload = [job.to_dict() for job in self._jobs.values()]
            self.jobs_file.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Failed to save jobs to %s: %s", self.jobs_file, exc)

    def _load_jobs(self) -> None:
        """Load jobs from the JSON file into :attr:`_jobs`.

        Malformed records are skipped with a warning.  If the file does not
        exist the method returns without error.
        """
        if not self.jobs_file.exists():
            return

        try:
            raw_text = self.jobs_file.read_text(encoding="utf-8")
            records = json.loads(raw_text)
        except Exception as exc:
            logger.error("Failed to read jobs file %s: %s", self.jobs_file, exc)
            return

        if not isinstance(records, list):
            logger.error("Jobs file %s must contain a JSON array.", self.jobs_file)
            return

        loaded = 0
        for record in records:
            if not isinstance(record, dict):
                logger.warning("Skipping non-dict job record: %r", record)
                continue
            try:
                job = ScheduledJob.from_dict(record)
                self._jobs[job.id] = job
                loaded += 1
            except Exception as exc:
                logger.warning("Skipping malformed job record: %s", exc)

        logger.debug("Loaded %d job(s) from %s.", loaded, self.jobs_file)

    # ------------------------------------------------------------------
    # APScheduler registration
    # ------------------------------------------------------------------

    def _schedule_job(self, job: ScheduledJob) -> None:
        """Register *job* with the APScheduler instance.

        The trigger configuration is derived by
        :func:`~missy.scheduler.parser.parse_schedule`.  The ``"trigger"``
        key is extracted and the remaining keys are passed as trigger kwargs.

        Args:
            job: The job to schedule.

        Raises:
            SchedulerError: When APScheduler refuses to add the job.
        """
        try:
            trigger_config = parse_schedule(job.schedule)
        except ValueError as exc:
            raise SchedulerError(
                f"Cannot schedule job {job.id!r}: invalid schedule {job.schedule!r}: {exc}"
            ) from exc

        trigger = trigger_config.pop("trigger")

        try:
            self._scheduler.add_job(
                func=self._run_job,
                trigger=trigger,
                kwargs={"job_id": job.id},
                id=job.id,
                name=job.name,
                replace_existing=True,
                **trigger_config,
            )
        except Exception as exc:
            raise SchedulerError(
                f"APScheduler failed to add job {job.id!r} ({job.name!r}): {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    def _emit_event(
        self,
        event_type: str,
        result: str,
        detail: dict,
        session_id: str = "",
        task_id: str = "",
    ) -> None:
        """Publish a scheduler audit event to the global event bus.

        Args:
            event_type: Dotted event type string.
            result: One of ``"allow"``, ``"deny"``, or ``"error"``.
            detail: Structured event data dictionary.
            session_id: Optional session identifier.
            task_id: Optional task identifier.
        """
        try:
            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type=event_type,
                category="scheduler",
                result=result,  # type: ignore[arg-type]
                detail=detail,
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit scheduler audit event %r", event_type)
