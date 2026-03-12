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
from datetime import datetime, timedelta, timezone
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
        description: str = "",
        max_attempts: int = 3,
        backoff_seconds: Optional[list] = None,
        retry_on: Optional[list] = None,
        delete_after_run: bool = False,
        active_hours: str = "",
        timezone: str = "",
    ) -> ScheduledJob:
        """Create a new job, persist it, and register it with APScheduler.

        Args:
            name: Human-readable job name.
            schedule: Human-readable schedule string (parsed by
                :func:`~missy.scheduler.parser.parse_schedule`).
            task: Prompt or task text to run when the job fires.
            provider: AI provider name to use for the run.
            description: Optional description of the job.
            max_attempts: Maximum retry attempts on failure.
            backoff_seconds: Delay in seconds between retries.
            retry_on: Error category tags that trigger a retry.
            delete_after_run: Remove the job after one successful execution.
            active_hours: ``"HH:MM-HH:MM"`` window; job is skipped outside it.
            timezone: IANA timezone string for cron/date triggers.

        Returns:
            The newly created :class:`ScheduledJob`.

        Raises:
            ValueError: When *schedule* cannot be parsed.
            SchedulerError: When APScheduler fails to register the job.
        """
        # Validate the schedule string before creating the job.
        try:
            parse_schedule(schedule, tz=timezone or None)
        except ValueError as exc:
            self._emit_event(
                event_type="scheduler.job.add",
                result="error",
                detail={"name": name, "schedule": schedule, "error": str(exc)},
            )
            raise

        job = ScheduledJob(
            name=name,
            description=description,
            schedule=schedule,
            task=task,
            provider=provider,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds if backoff_seconds is not None else [30, 60, 300],
            retry_on=retry_on if retry_on is not None else ["network", "provider_error"],
            delete_after_run=delete_after_run,
            active_hours=active_hours,
            timezone=timezone,
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

    def list_jobs_with_details(self) -> list[ScheduledJob]:
        """Return full :class:`ScheduledJob` objects for all registered jobs.

        This is an alias for :meth:`list_jobs` and always returns the complete
        dataclass objects (never stripped/summarised representations).

        Returns:
            A new list of :class:`ScheduledJob` instances.
        """
        return list(self._jobs.values())

    def cleanup_memory(self, older_than_days: int = 30) -> int:
        """Delete conversation history older than *older_than_days* days.

        Delegates to :class:`~missy.memory.store.MemoryStore` when it exposes
        a ``cleanup`` method.  Errors are logged as warnings and the method
        always returns without raising.

        Args:
            older_than_days: Threshold in days.  Records older than this are
                removed.

        Returns:
            The number of records removed, or ``0`` if the store does not
            support cleanup or an error occurs.
        """
        try:
            from missy.memory.store import MemoryStore

            store = MemoryStore()
            if hasattr(store, "cleanup"):
                return store.cleanup(older_than_days=older_than_days)
            return 0
        except Exception as exc:
            logger.warning("Memory cleanup failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _run_job(self, job_id: str) -> None:
        """Execute a scheduled job by running it through the agent runtime.

        This method is called by APScheduler on the background thread.  It
        resolves the :class:`~missy.agent.runtime.AgentRuntime`, runs the
        job's task string, stores the result, and emits audit events.

        Active-hours gating, retry scheduling, delete-after-run, and
        permanent-failure alerting are all handled here.

        Args:
            job_id: ID of the job to execute.
        """
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning("Scheduled job %r no longer exists; skipping.", job_id)
            return

        # ------------------------------------------------------------------
        # Active-hours gate
        # ------------------------------------------------------------------
        if not job.should_run_now():
            logger.info(
                "Job %s outside active_hours %s; skipping.", job.id, job.active_hours
            )
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
            job.consecutive_failures += 1
            job.last_error = str(exc)

            self._emit_event(
                event_type="scheduler.job.run.error",
                result="error",
                detail={"job_id": job_id, "name": job.name, "error": str(exc)},
                session_id=session_id,
                task_id=task_id,
            )

            if job.should_retry(str(exc)):
                # Calculate backoff delay using the failure index (clamped to
                # the length of the backoff list).
                failures = job.consecutive_failures
                backoff = job.backoff_seconds[
                    min(failures - 1, len(job.backoff_seconds) - 1)
                ]
                retry_run_date = datetime.now(tz=timezone.utc) + timedelta(seconds=backoff)
                try:
                    retry_job_id = f"{job_id}_retry_{failures}"
                    self._scheduler.add_job(
                        func=self._run_job,
                        trigger="date",
                        run_date=retry_run_date,
                        kwargs={"job_id": job_id},
                        id=retry_job_id,
                        name=f"{job.name} (retry {failures})",
                        replace_existing=True,
                    )
                    logger.info(
                        "Retry %d for job %r scheduled in %ds (at %s).",
                        failures,
                        job.name,
                        backoff,
                        retry_run_date.isoformat(),
                    )
                    self._emit_event(
                        event_type="scheduler.job.retry_scheduled",
                        result="allow",
                        detail={
                            "job_id": job_id,
                            "name": job.name,
                            "attempt": failures,
                            "backoff_seconds": backoff,
                            "retry_run_date": retry_run_date.isoformat(),
                        },
                    )
                except Exception as retry_exc:
                    logger.error(
                        "Failed to schedule retry for job %r: %s", job.name, retry_exc
                    )
            else:
                # Maximum attempts exhausted.
                logger.error(
                    "Job %r (id=%s) exceeded max_attempts=%d; not retrying.",
                    job.name,
                    job_id,
                    job.max_attempts,
                )
                self._emit_event(
                    event_type="scheduler.job.failed_permanently",
                    result="error",
                    detail={
                        "job_id": job_id,
                        "name": job.name,
                        "failures": job.consecutive_failures,
                        "last_error": job.last_error,
                    },
                )
                self._emit_event(
                    event_type="scheduler.job.alert",
                    result="error",
                    detail={
                        "job_id": job_id,
                        "job_name": job.name,
                        "failures": job.consecutive_failures,
                        "last_error": job.last_error,
                    },
                )

            self._save_jobs()
            return

        # ------------------------------------------------------------------
        # Successful run
        # ------------------------------------------------------------------
        job.last_run = datetime.now(tz=timezone.utc)
        job.run_count += 1
        job.last_result = result_text
        job.consecutive_failures = 0
        job.last_error = ""

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

        # Delete-after-run: remove the job after a single successful execution.
        if job.delete_after_run:
            logger.info("Job %r has delete_after_run=True; removing.", job.name)
            try:
                self.remove_job(job_id)
            except Exception as exc:
                logger.warning(
                    "Failed to remove one-shot job %r after run: %s", job_id, exc
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
        :func:`~missy.scheduler.parser.parse_schedule`.  Raw cron expressions
        (signalled by the ``"_cron_expression"`` key) are handled via
        ``CronTrigger.from_crontab``; one-shot date triggers are handled via
        ``DateTrigger``; all other triggers are passed directly to
        ``scheduler.add_job``.

        Args:
            job: The job to schedule.

        Raises:
            SchedulerError: When APScheduler refuses to add the job.
        """
        tz = job.timezone or None

        try:
            schedule_config = parse_schedule(job.schedule, tz=tz)
        except ValueError as exc:
            raise SchedulerError(
                f"Cannot schedule job {job.id!r}: invalid schedule {job.schedule!r}: {exc}"
            ) from exc

        trigger_type = schedule_config.pop("trigger")

        try:
            if trigger_type == "cron" and "_cron_expression" in schedule_config:
                # Raw cron expression — use CronTrigger.from_crontab.
                from apscheduler.triggers.cron import CronTrigger

                cron_expr = schedule_config.pop("_cron_expression")
                # Any remaining key after popping _cron_expression is "timezone".
                cron_tz = schedule_config.pop("timezone", None) or tz
                trigger = CronTrigger.from_crontab(cron_expr, timezone=cron_tz)
                self._scheduler.add_job(
                    func=self._run_job,
                    trigger=trigger,
                    kwargs={"job_id": job.id},
                    id=job.id,
                    name=job.name,
                    replace_existing=True,
                )

            elif trigger_type == "date":
                # One-shot future-dated trigger.
                from apscheduler.triggers.date import DateTrigger

                run_date = schedule_config.pop("run_date")
                date_tz = schedule_config.pop("timezone", None) or tz
                trigger = DateTrigger(run_date=run_date, timezone=date_tz)
                self._scheduler.add_job(
                    func=self._run_job,
                    trigger=trigger,
                    kwargs={"job_id": job.id},
                    id=job.id,
                    name=job.name,
                    replace_existing=True,
                )

            else:
                # interval or standard cron (daily/weekly) — pass kwargs directly.
                self._scheduler.add_job(
                    func=self._run_job,
                    trigger=trigger_type,
                    kwargs={"job_id": job.id},
                    id=job.id,
                    name=job.name,
                    replace_existing=True,
                    **schedule_config,
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
