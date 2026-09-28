"""Constrained built-in tools for Missy-owned Nomad workloads."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from missy.config.settings import MissyConfig, load_config
from missy.nomad.errors import NomadError
from missy.nomad.manager import NomadManager
from missy.nomad.models import NomadJobRequest, choose_scope
from missy.tools.base import BaseTool, ToolPermissions, ToolResult

logger = logging.getLogger(__name__)

_READ_PERMISSIONS = ToolPermissions(network=True, filesystem_read=True, shell=True)
_MUTATE_PERMISSIONS = ToolPermissions(
    network=True,
    filesystem_read=True,
    filesystem_write=True,
    shell=True,
)


def _load_config() -> MissyConfig:
    path = str(Path(os.environ.get("MISSY_CONFIG", "~/.missy/config.yaml")).expanduser())
    return load_config(path)


class _NomadToolMixin:
    writes_state = False

    def _config(self) -> MissyConfig:
        config = _load_config()
        if not config.nomad.enabled:
            raise NomadError("Nomad integration is disabled in config.yaml.")
        return config

    def _manager(self) -> NomadManager:
        return NomadManager(self._config().nomad)

    def resolve_shell_command(self, kwargs: dict[str, Any]) -> str:
        del kwargs
        return "nomad"

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        del kwargs
        parsed = urlparse(self._config().nomad.address)
        if not parsed.hostname:
            raise ValueError("Configured Nomad endpoint has no hostname.")
        port = parsed.port or 443
        return [f"{parsed.hostname}:{port}"]

    def resolve_filesystem_targets(self, kwargs: dict[str, Any]) -> tuple[list[str], list[str]]:
        del kwargs
        config = self._config()
        bundle = Path(config.nomad.bundle_dir).expanduser().resolve()
        config_path = Path(os.environ.get("MISSY_CONFIG", "~/.missy/config.yaml")).expanduser()
        reads = [
            str(config_path.resolve()),
            str(bundle / "ca.pem"),
            str(bundle / "missy.pem"),
            str(bundle / "missy-key.pem"),
            str(bundle / "missy.token"),
        ]
        state_dir = str(Path(config.nomad.state_dir).expanduser().resolve())
        if self.writes_state:
            reads.append(state_dir)
        writes = [state_dir] if self.writes_state else []
        return reads, writes

    @staticmethod
    def _error(exc: Exception) -> ToolResult:
        logger.info("Nomad tool operation refused or failed: %s", exc)
        return ToolResult(success=False, output=None, error=str(exc))


class NomadDiscoverTool(_NomadToolMixin, BaseTool):
    """Read current Nomad topology, capacity, and optionally jobs."""

    name = "nomad_discover"
    description = (
        "Read current authorized Nomad namespaces, node pools, detailed node capacity, "
        "driver health, allocation reservations, host pressure, and optional namespace jobs."
    )
    permissions = _READ_PERMISSIONS

    def execute(self, *, namespace: str = "", **_: Any) -> ToolResult:
        try:
            return ToolResult(success=True, output=self._manager().discover(namespace=namespace))
        except (NomadError, KeyError, ValueError) as exc:
            return self._error(exc)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "Optional owner-authorized namespace whose jobs should be listed.",
                    }
                },
                "required": [],
            },
        }


class NomadRecommendPlacementTool(_NomadToolMixin, BaseTool):
    """Recommend placement from a structured workload and live evidence."""

    name = "nomad_recommend_placement"
    description = (
        "Recommend an authorized Nomad namespace, pool, datacenter, and candidate node from "
        "live capacity, allocation, architecture, Docker, and host-pressure evidence."
    )
    permissions = _READ_PERMISSIONS

    def execute(self, *, request: dict[str, Any], **_: Any) -> ToolResult:
        try:
            return ToolResult(success=True, output=self._manager().recommend(request))
        except (NomadError, KeyError, ValueError) as exc:
            return self._error(exc)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {"request": _job_request_schema()},
                "required": ["request"],
            },
        }


class NomadPlanJobTool(_NomadToolMixin, BaseTool):
    """Build, validate, plan, and retain an exact job artifact."""

    name = "nomad_plan_job"
    description = (
        "Build a constrained Docker service or batch job, verify live placement, validate it, "
        "run Nomad's dry-run plan, and retain the exact artifact and check index for submission."
    )
    permissions = _MUTATE_PERMISSIONS
    writes_state = True

    def execute(self, *, request: dict[str, Any], **_: Any) -> ToolResult:
        try:
            return ToolResult(success=True, output=self._manager().plan(request))
        except (NomadError, KeyError, ValueError) as exc:
            return self._error(exc)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {"request": _job_request_schema()},
                "required": ["request"],
            },
        }


class NomadSubmitJobTool(_NomadToolMixin, BaseTool):
    """Submit only the exact artifact retained by a clean plan."""

    name = "nomad_submit_job"
    description = (
        "Submit an unexpired retained Nomad plan using its exact specification and check index. "
        "Returns submitted state, never claims the workload is healthy."
    )
    permissions = _MUTATE_PERMISSIONS
    writes_state = True

    def execute(self, *, plan_id: str, **_: Any) -> ToolResult:
        try:
            return ToolResult(success=True, output=self._manager().submit(plan_id))
        except (NomadError, KeyError, ValueError) as exc:
            return self._error(exc)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {"plan_id": {"type": "string"}},
                "required": ["plan_id"],
            },
        }


class NomadJobStatusTool(_NomadToolMixin, BaseTool):
    """Inspect or briefly observe a Nomad workload."""

    name = "nomad_job_status"
    description = (
        "Inspect a Nomad job's scheduler, evaluation, deployment, allocation, task, and health "
        "state; optionally observe it for a bounded time."
    )
    permissions = _MUTATE_PERMISSIONS
    writes_state = True

    def execute(
        self,
        *,
        namespace: str,
        job_id: str,
        observe: bool = False,
        timeout_seconds: int = 120,
        **_: Any,
    ) -> ToolResult:
        try:
            manager = self._manager()
            output = (
                manager.observe(namespace, job_id, timeout_seconds=timeout_seconds)
                if observe
                else manager.status(namespace, job_id)
            )
            return ToolResult(success=True, output=output)
        except (NomadError, KeyError, ValueError) as exc:
            return self._error(exc)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "job_id": {"type": "string"},
                    "observe": {"type": "boolean", "default": False},
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
                },
                "required": ["namespace", "job_id"],
            },
        }


class NomadJobResultTool(_NomadToolMixin, BaseTool):
    """Collect bounded logs and verify an owned batch result."""

    name = "nomad_job_result"
    description = (
        "Collect bounded redacted stdout/stderr and verify terminal status and an optional "
        "expected-output marker for a Missy-owned Nomad batch job."
    )
    permissions = _MUTATE_PERMISSIONS
    writes_state = True

    def execute(self, *, namespace: str, job_id: str, log_lines: int = 100, **_: Any) -> ToolResult:
        try:
            return ToolResult(
                success=True,
                output=self._manager().result(namespace, job_id, log_lines=log_lines),
            )
        except (NomadError, KeyError, ValueError) as exc:
            return self._error(exc)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "job_id": {"type": "string"},
                    "log_lines": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                "required": ["namespace", "job_id"],
            },
        }


class NomadJobActionTool(_NomadToolMixin, BaseTool):
    """Operate only on jobs proven owned by Missy."""

    name = "nomad_job_action"
    description = (
        "Restart, stop, cancel, or explicitly purge a Nomad job only after protected local and "
        "remote ownership metadata agree. Never deletes application data or runs global cleanup."
    )
    permissions = _MUTATE_PERMISSIONS
    writes_state = True

    def execute(
        self,
        *,
        namespace: str,
        job_id: str,
        action: str,
        confirm_purge: bool = False,
        **_: Any,
    ) -> ToolResult:
        try:
            return ToolResult(
                success=True,
                output=self._manager().action(
                    namespace, job_id, action, confirm_purge=confirm_purge
                ),
            )
        except (NomadError, KeyError, ValueError) as exc:
            return self._error(exc)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "job_id": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["restart", "stop", "cancel", "purge"],
                    },
                    "confirm_purge": {"type": "boolean", "default": False},
                },
                "required": ["namespace", "job_id", "action"],
            },
        }


class NomadScaleTool(_NomadToolMixin, BaseTool):
    """Plan service scaling through the normal exact-spec update path."""

    name = "nomad_plan_scale"
    description = (
        "Plan a scale of a proven Missy-owned service, including fresh discovery, full target "
        "capacity plus rollout overlap, validation, and optimistic concurrency. Submit the "
        "returned plan separately with nomad_submit_job."
    )
    permissions = _MUTATE_PERMISSIONS
    writes_state = True

    def execute(self, *, namespace: str, job_id: str, count: int, **_: Any) -> ToolResult:
        try:
            return ToolResult(
                success=True,
                output=self._manager().plan_scale(namespace, job_id, count),
            )
        except (NomadError, KeyError, ValueError) as exc:
            return self._error(exc)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "job_id": {"type": "string"},
                    "count": {"type": "integer", "minimum": 1},
                },
                "required": ["namespace", "job_id", "count"],
            },
        }


class NomadOffloadTool(_NomadToolMixin, BaseTool):
    """Operate a durable offload task from an operator-approved template."""

    name = "nomad_offload_task"
    description = (
        "Plan, submit, inspect, collect, or cancel a durable high-latency batch task. Planning "
        "accepts only workload templates explicitly configured by the operator; it never "
        "converts arbitrary tool calls or shell text into jobs."
    )
    permissions = _MUTATE_PERMISSIONS
    writes_state = True

    def execute(
        self,
        *,
        action: str,
        task_id: str = "",
        workload_template: str = "",
        parameters: dict[str, str] | None = None,
        input_artifacts: dict[str, str] | None = None,
        output_artifacts: dict[str, str] | None = None,
        idempotency_key: str = "",
        **_: Any,
    ) -> ToolResult:
        try:
            manager = self._manager()
            action = action.lower()
            if action == "plan":
                output = manager.plan_offload(
                    workload_template,
                    parameters=parameters,
                    input_artifacts=input_artifacts,
                    output_artifacts=output_artifacts,
                    idempotency_key=idempotency_key,
                )
            elif action == "submit":
                output = manager.submit_offload(task_id)
            elif action == "status":
                output = manager.offload_status(task_id)
            elif action == "result":
                output = manager.offload_result(task_id)
            elif action == "cancel":
                output = manager.cancel_offload(task_id)
            else:
                raise ValueError("action must be plan, submit, status, result, or cancel.")
            return ToolResult(success=True, output=output)
        except (NomadError, KeyError, ValueError) as exc:
            return self._error(exc)

    def get_schema(self) -> dict[str, Any]:
        mapping = {"type": "object", "additionalProperties": {"type": "string"}}
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["plan", "submit", "status", "result", "cancel"],
                    },
                    "task_id": {"type": "string"},
                    "workload_template": {"type": "string"},
                    "parameters": mapping,
                    "input_artifacts": mapping,
                    "output_artifacts": mapping,
                    "idempotency_key": {"type": "string"},
                },
                "required": ["action"],
            },
        }


class NomadBenchmarkTool(_NomadToolMixin, BaseTool):
    """Plan and operate a bounded reproducible benchmark run set."""

    name = "nomad_benchmark"
    description = (
        "Plan, start, inspect, collect, or cancel a bounded benchmark made from an "
        "operator-approved workload template. Preserves every run and flags incomparable results."
    )
    permissions = _MUTATE_PERMISSIONS
    writes_state = True

    def execute(
        self,
        *,
        action: str,
        benchmark_id: str = "",
        definition: dict[str, Any] | None = None,
        **_: Any,
    ) -> ToolResult:
        try:
            manager = self._manager()
            action = action.lower()
            if action == "plan":
                output = manager.plan_benchmark(definition or {})
            elif action == "start":
                output = manager.start_benchmark(benchmark_id)
            elif action == "status":
                output = manager.benchmark_status(benchmark_id)
            elif action == "results":
                output = manager.benchmark_results(benchmark_id)
            elif action == "cancel":
                output = manager.cancel_benchmark(benchmark_id)
            else:
                raise ValueError("action must be plan, start, status, results, or cancel.")
            return ToolResult(success=True, output=output)
        except (NomadError, KeyError, ValueError) as exc:
            return self._error(exc)

    def get_schema(self) -> dict[str, Any]:
        string_map = {"type": "object", "additionalProperties": {"type": "string"}}
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["plan", "start", "status", "results", "cancel"],
                    },
                    "benchmark_id": {"type": "string"},
                    "definition": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "workload_template": {"type": "string"},
                            "workload_version": {"type": "string"},
                            "input_artifacts": string_map,
                            "parameters": string_map,
                            "output_prefix": {"type": "string"},
                            "result_schema": {"type": "object"},
                            "warmup_runs": {"type": "integer", "minimum": 0},
                            "repetitions": {"type": "integer", "minimum": 1},
                            "parallelism": {"type": "integer", "minimum": 1},
                            "idempotency_key": {"type": "string"},
                        },
                        "required": [
                            "name",
                            "workload_template",
                            "workload_version",
                            "output_prefix",
                        ],
                    },
                },
                "required": ["action"],
            },
        }


class NomadReconcileTool(_NomadToolMixin, BaseTool):
    """Reconcile durable owned-job records after interruption."""

    name = "nomad_reconcile"
    description = (
        "Reconcile Missy's protected Nomad operation journal with current remote jobs after a "
        "restart or connectivity loss without recreating unknown or modified workloads."
    )
    permissions = _MUTATE_PERMISSIONS
    writes_state = True

    def execute(self, **_: Any) -> ToolResult:
        try:
            return ToolResult(success=True, output=self._manager().reconcile())
        except (NomadError, KeyError, ValueError) as exc:
            return self._error(exc)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": {}, "required": []},
        }


class NomadScheduleTool(_NomadToolMixin, BaseTool):
    """Create and control durable schedules that submit fresh batch jobs."""

    name = "nomad_schedule"
    description = (
        "Create, list, pause, resume, or remove durable Missy schedules that perform fresh "
        "Nomad discovery and planning before each finite batch submission."
    )
    permissions = _MUTATE_PERMISSIONS
    writes_state = True

    def resolve_filesystem_targets(self, kwargs: dict[str, Any]) -> tuple[list[str], list[str]]:
        reads, writes = super().resolve_filesystem_targets(kwargs)
        jobs_file = Path("~/.missy/jobs.json").expanduser().resolve()
        return [*reads, str(jobs_file)], [*writes, str(jobs_file)]

    def _scheduler(self):
        from missy.scheduler.manager import SchedulerManager

        config = self._config()
        return SchedulerManager(
            max_jobs=config.scheduling.max_jobs,
            default_active_hours=config.scheduling.active_hours,
            misfire_grace_seconds=config.scheduling.misfire_grace_seconds,
            nomad_config=config.nomad,
        )

    @staticmethod
    def _find_nomad_job(manager: Any, schedule_id: str) -> Any:
        for job in manager.list_jobs():
            if job.id == schedule_id and job.execution_target == "nomad":
                return job
        raise KeyError(f"No Nomad schedule found with id {schedule_id!r}.")

    def execute(
        self,
        *,
        action: str,
        name: str = "",
        schedule: str = "",
        request: dict[str, Any] | None = None,
        schedule_id: str = "",
        overlap_policy: str = "skip",
        timezone: str = "",
        active_hours: str = "",
        **_: Any,
    ) -> ToolResult:
        try:
            manager = self._scheduler()
            manager.open_offline()
            action = action.lower()
            if action == "create":
                config = self._config()
                parsed = NomadJobRequest.from_mapping(request or {})
                parsed.job_type = "batch"
                parsed.validate(config.nomad)
                choose_scope(parsed, config.nomad)
                normalized = dict(parsed.__dict__)
                job = manager.add_job(
                    name=name,
                    schedule=schedule,
                    task="Submit the stored structured Nomad batch workload.",
                    provider="nomad",
                    description="Scheduled Nomad batch offload",
                    max_attempts=3,
                    retry_on=["network", "timeout"],
                    active_hours=active_hours,
                    timezone=timezone,
                    capability_mode="no-tools",
                    execution_target="nomad",
                    nomad_request=normalized,
                    nomad_overlap_policy=overlap_policy,
                )
                output: Any = {
                    "schedule_id": job.id,
                    "name": job.name,
                    "schedule": job.schedule,
                    "next_run": job.next_run.isoformat() if job.next_run else None,
                    "overlap_policy": job.nomad_overlap_policy,
                    "state": "scheduled",
                }
            elif action == "list":
                cluster_manager = NomadManager(self._config().nomad)
                changed = False
                for scheduled_job in manager.list_jobs():
                    if scheduled_job.execution_target != "nomad":
                        continue
                    namespace = str(
                        (scheduled_job.nomad_request or {}).get("namespace")
                        or self._config().nomad.default_namespace
                    )
                    terminal = {
                        "completed",
                        "failed",
                        "cancelled",
                        "timed_out",
                        "result_validation_failed",
                    }
                    for run in scheduled_job.nomad_runs:
                        nomad_job_id = str(run.get("nomad_job_id") or "")
                        if not nomad_job_id or run.get("state") in terminal:
                            continue
                        try:
                            status = cluster_manager.status(namespace, nomad_job_id)
                            state = cluster_manager._state_from_status(status)
                            if state == "completed_unverified":
                                state = str(
                                    cluster_manager.result(namespace, nomad_job_id)["outcome"]
                                )
                        except Exception as exc:
                            state = "observation_unavailable"
                            observation_error = type(exc).__name__
                        else:
                            observation_error = ""
                        run["state"] = state
                        if observation_error:
                            run["observation_error"] = observation_error
                        if state in {
                            "completed_unverified",
                            "completed",
                            "failed",
                            "cancelled",
                            "timed_out",
                            "result_validation_failed",
                        }:
                            run.setdefault("completed_at", datetime.now(tz=UTC).isoformat())
                        changed = True
                if changed:
                    manager._save_jobs()
                output = [
                    {
                        "schedule_id": job.id,
                        "name": job.name,
                        "schedule": job.schedule,
                        "enabled": job.enabled,
                        "overlap_policy": job.nomad_overlap_policy,
                        "run_count": job.run_count,
                        "last_run": job.last_run.isoformat() if job.last_run else None,
                        "next_run": job.next_run.isoformat() if job.next_run else None,
                        "last_nomad_job_id": job.last_nomad_job_id or None,
                        "last_nomad_evaluation_id": job.last_nomad_evaluation_id or None,
                        "queued_runs": job.nomad_queued_runs,
                        "runs": job.nomad_runs,
                        "last_result": job.last_result,
                    }
                    for job in manager.list_jobs()
                    if job.execution_target == "nomad"
                ]
            else:
                self._find_nomad_job(manager, schedule_id)
                if action == "pause":
                    manager.pause_job(schedule_id)
                    state = "paused"
                elif action == "resume":
                    manager.resume_job(schedule_id)
                    state = "scheduled"
                elif action == "remove":
                    manager.remove_job(schedule_id)
                    state = "removed"
                else:
                    raise ValueError("action must be create, list, pause, resume, or remove.")
                output = {
                    "schedule_id": schedule_id,
                    "state": state,
                    "running_nomad_jobs_unchanged": True,
                }
            return ToolResult(success=True, output=output)
        except (NomadError, KeyError, ValueError) as exc:
            return self._error(exc)

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "pause", "resume", "remove"],
                    },
                    "name": {"type": "string"},
                    "schedule": {"type": "string"},
                    "request": _job_request_schema(),
                    "schedule_id": {"type": "string"},
                    "overlap_policy": {
                        "type": "string",
                        "enum": ["skip", "queue", "allow", "replace"],
                    },
                    "timezone": {"type": "string"},
                    "active_hours": {"type": "string"},
                },
                "required": ["action"],
            },
        }


def _job_request_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "purpose": {"type": "string"},
            "image": {
                "type": "string",
                "description": "Docker image, normally pinned with @sha256:<digest>.",
            },
            "job_type": {"type": "string", "enum": ["service", "batch"]},
            "job_id": {"type": "string"},
            "namespace": {"type": "string"},
            "node_pool": {"type": "string"},
            "datacenter": {"type": "string"},
            "command": {"type": "string"},
            "args": {"type": "array", "items": {"type": "string"}, "maxItems": 128},
            "environment": {"type": "object", "additionalProperties": {"type": "string"}},
            "secret_references": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "cpu_mhz": {"type": "integer", "minimum": 1},
            "memory_mb": {"type": "integer", "minimum": 1},
            "disk_mb": {"type": "integer", "minimum": 1},
            "count": {"type": "integer", "minimum": 1},
            "architecture": {"type": "string"},
            "required_node_attributes": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "ports": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string"},
                        "to": {"type": "integer", "minimum": 1, "maximum": 65535},
                        "static": {"type": "integer", "minimum": 1, "maximum": 65535},
                    },
                    "required": ["label"],
                },
            },
            "service": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "port": {"type": "string"},
                    "check_path": {"type": "string"},
                    "check_interval_seconds": {"type": "integer"},
                    "check_timeout_seconds": {"type": "integer"},
                },
                "required": ["name", "port"],
            },
            "max_run_seconds": {"type": "integer", "minimum": 1},
            "retry_attempts": {"type": "integer", "minimum": 0},
            "expected_output": {"type": "string"},
            "input_artifacts": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "output_artifacts": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "parameters": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "result_schema": {"type": "object"},
            "cleanup_policy": {
                "type": "string",
                "enum": ["retain", "stop_after_verified"],
            },
            "disposable": {"type": "boolean"},
            "external_prerequisites": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 16,
            },
            "idempotency_key": {"type": "string"},
            "stateful": {"type": "boolean"},
            "persistence_plan": {"type": "object"},
        },
        "required": ["purpose", "image"],
    }
