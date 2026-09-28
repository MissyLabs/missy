"""High-level safe Nomad workflows used by built-in tools and schedules."""

from __future__ import annotations

import contextlib
import copy
import json
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any

from missy.config.settings import NomadConfig
from missy.core.events import AuditEvent, event_bus
from missy.nomad.benchmark import BenchmarkDefinition, new_benchmark_id
from missy.nomad.client import NomadClient
from missy.nomad.credentials import credential_metadata
from missy.nomad.errors import (
    NomadAuthorizationError,
    NomadCommandError,
    NomadMutationUnknown,
    NomadOwnershipError,
    NomadValidationError,
)
from missy.nomad.models import NomadJobRequest, build_job, choose_scope, spec_hash
from missy.nomad.placement import recommend_placement
from missy.nomad.store import NomadStateStore
from missy.nomad.workloads import build_offload_request, new_task_id
from missy.tools.base import current_tool_context

_TERMINAL_ALLOCATION_STATES = {"complete", "failed", "lost"}


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _data_gap(label: str, exc: Exception) -> str:
    state = "denied" if isinstance(exc, NomadAuthorizationError) else "unavailable"
    return f"{label} {state}: {type(exc).__name__}"


def _sum_allocations(allocations: list[dict[str, Any]]) -> dict[str, Any]:
    cpu = memory = disk = 0
    for allocation in allocations:
        desired = str(allocation.get("DesiredStatus") or "run").lower()
        client = str(allocation.get("ClientStatus") or "pending").lower()
        if desired != "run" or client in _TERMINAL_ALLOCATION_STATES:
            continue
        resources = allocation.get("Resources") or {}
        cpu += int(resources.get("CPU") or 0)
        memory += int(resources.get("MemoryMB") or 0)
        disk += int(resources.get("DiskMB") or 0)
    return {
        "Cpu": {"CpuShares": cpu},
        "Memory": {"MemoryMB": memory},
        "Disk": {"DiskMB": disk},
    }


def _node_summary(node: dict[str, Any]) -> dict[str, Any]:
    attrs = node.get("Attributes") if isinstance(node.get("Attributes"), dict) else {}
    drivers = node.get("Drivers") if isinstance(node.get("Drivers"), dict) else {}
    stats = node.get("HostStats") if isinstance(node.get("HostStats"), dict) else {}
    return {
        "ID": node.get("ID"),
        "Name": node.get("Name"),
        "Datacenter": node.get("Datacenter"),
        "NodePool": node.get("NodePool"),
        "Status": node.get("Status"),
        "SchedulingEligibility": node.get("SchedulingEligibility"),
        "Drain": bool(node.get("Drain")),
        "Attributes": {
            "cpu.arch": attrs.get("cpu.arch"),
            "kernel.arch": attrs.get("kernel.arch"),
        },
        "Drivers": {"docker": drivers.get("docker")},
        "NodeResources": node.get("NodeResources"),
        "ReservedResources": node.get("ReservedResources"),
        "AllocatedResources": node.get("AllocatedResources"),
        "HostStats": {
            "Memory": stats.get("Memory"),
            "DiskStats": stats.get("DiskStats"),
        },
        "data_gaps": list(node.get("_data_gaps") or []),
    }


def _find_int(value: Any, key: str) -> int | None:
    if isinstance(value, dict):
        if key in value and isinstance(value[key], (int, float)):
            return int(value[key])
        for child in value.values():
            found = _find_int(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_int(child, key)
            if found is not None:
                return found
    return None


class NomadManager:
    """Coordinate discovery, exact-plan submission, ownership, and results."""

    def __init__(
        self,
        config: NomadConfig,
        *,
        client: NomadClient | None = None,
        store: NomadStateStore | None = None,
    ) -> None:
        self.config = config
        self.client = client or NomadClient(config)
        self.store = store or NomadStateStore(config.state_dir)

    def _audit(self, event_type: str, result: str, detail: dict[str, Any]) -> None:
        session_id, task_id = current_tool_context()
        with contextlib.suppress(Exception):
            event_bus.publish(
                AuditEvent.now(
                    session_id=session_id,
                    task_id=task_id,
                    event_type=event_type,
                    category="scheduler",
                    result=result,  # type: ignore[arg-type]
                    detail=detail,
                )
            )

    def _authorize_scope(self, namespace: str, node_pool: str = "", datacenter: str = "") -> None:
        for label, value, allowed in (
            ("namespace", namespace, self.config.allowed_namespaces),
            ("node pool", node_pool, self.config.allowed_node_pools),
            ("datacenter", datacenter, self.config.allowed_datacenters),
        ):
            if value and value not in allowed:
                self._audit("nomad.scope.denied", "deny", {"kind": label, "value": value})
                raise NomadValidationError(f"{label.title()} {value!r} is not owner-authorized.")

    def discover_nodes(self) -> list[dict[str, Any]]:
        """Fetch detailed nodes, allocations, and actual host stats independently."""
        nodes: list[dict[str, Any]] = []
        for summary in self.client.nodes():
            node_id = str(summary.get("ID") or "")
            if not node_id:
                continue
            gaps: list[str] = []
            try:
                detail = self.client.node(node_id)
            except Exception as exc:
                detail = copy.deepcopy(summary)
                gaps.append(_data_gap("node detail", exc))
            try:
                allocations = self.client.node_allocations(node_id)
                detail["AllocatedResources"] = _sum_allocations(allocations)
                detail["_allocation_count"] = len(allocations)
            except Exception as exc:
                gaps.append(_data_gap("allocation reservations", exc))
            try:
                detail["HostStats"] = self.client.node_stats(node_id)
            except Exception as exc:
                gaps.append(_data_gap("host pressure", exc))
            detail["_data_gaps"] = gaps
            nodes.append(detail)
        self._audit(
            "nomad.discovery.nodes",
            "allow",
            {
                "node_count": len(nodes),
                "nodes_with_data_gaps": sum(bool(n["_data_gaps"]) for n in nodes),
            },
        )
        return nodes

    def discover(self, *, namespace: str = "") -> dict[str, Any]:
        """Return a timestamped, bounded snapshot of visible cluster state."""
        if namespace:
            self._authorize_scope(namespace)
        gaps: list[str] = []
        try:
            namespaces = [
                {"name": row.get("Name"), "description": row.get("Description", "")}
                for row in self.client.namespaces()
            ]
        except Exception as exc:
            namespaces = []
            gaps.append(_data_gap("namespaces", exc))
        if not namespaces and not any(gap.startswith("namespaces ") for gap in gaps):
            gaps.append("namespace listing empty; absence of namespaces is not established")
        try:
            node_pools = [
                {"name": row.get("Name"), "description": row.get("Description", "")}
                for row in self.client.node_pools()
            ]
        except Exception as exc:
            node_pools = []
            gaps.append(_data_gap("node pools", exc))
        nodes = self.discover_nodes()
        if not node_pools:
            node_pools = [
                {"name": name, "description": "observed from nodes"}
                for name in sorted({str(n.get("NodePool") or "default") for n in nodes})
            ]
            gaps.append("node-pool listing unavailable or empty; pools inferred from visible nodes")
        jobs: list[dict[str, Any]] = []
        if namespace:
            try:
                jobs = [
                    {
                        "id": row.get("ID"),
                        "name": row.get("Name"),
                        "type": row.get("Type"),
                        "status": row.get("Status"),
                        "namespace": row.get("Namespace") or namespace,
                        "node_pool": row.get("NodePool"),
                    }
                    for row in self.client.jobs(namespace)
                ]
            except Exception as exc:
                gaps.append(_data_gap(f"jobs for {namespace!r}", exc))
        return {
            "observed_at": _now().isoformat(),
            "identity": credential_metadata(self.client.credentials),
            "namespaces": namespaces,
            "node_pools": node_pools,
            "nodes": [_node_summary(node) for node in nodes],
            "jobs": jobs,
            "data_gaps": gaps,
        }

    def recommend(self, request_data: dict[str, Any]) -> dict[str, Any]:
        request = NomadJobRequest.from_mapping(request_data)
        request.validate(self.config, allow_unverified_stateful=True)
        choose_scope(request, self.config)
        self._authorize_scope(request.namespace, request.node_pool, request.datacenter)
        result = recommend_placement(self.discover_nodes(), request, self.config)
        self._audit(
            "nomad.placement.recommend",
            "allow",
            {
                "namespace": request.namespace,
                "node_pool": request.node_pool,
                "datacenter": request.datacenter,
                "recommended_node": result["recommended_node"]["name"],
            },
        )
        return result

    @staticmethod
    def _plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
        failed = plan.get("FailedTGAllocs") or {}
        annotations = plan.get("Annotations") or {}
        desired = annotations.get("DesiredTGUpdates", {}) if isinstance(annotations, dict) else {}
        changes = {"create": 0, "replace": 0, "stop": 0, "in_place": 0}
        if isinstance(desired, dict):
            for update in desired.values():
                if not isinstance(update, dict):
                    continue
                changes["create"] += int(update.get("Place") or 0)
                changes["replace"] += int(update.get("DestructiveUpdate") or 0)
                changes["stop"] += int(update.get("Stop") or 0)
                changes["in_place"] += int(update.get("InPlaceUpdate") or 0)
        return {
            "exit_code": plan.get("_exit_code"),
            "job_modify_index": _find_int(plan, "JobModifyIndex") or 0,
            "failed_task_groups": sorted(failed) if isinstance(failed, dict) else [],
            "desired_task_group_updates": desired,
            "allocation_changes": changes,
            "diff_present": bool(plan.get("Diff")),
        }

    def _assert_owned(
        self, namespace: str, job_id: str, remote: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            local = self.store.get_job(namespace, job_id)
        except KeyError as exc:
            raise NomadOwnershipError(
                "No local operation record proves that Missy owns this job."
            ) from exc
        remote = remote or self.client.inspect_job(namespace, job_id)
        meta = remote.get("Meta") if isinstance(remote.get("Meta"), dict) else {}
        expected_tracking = str(local.get("tracking_id") or "")
        if not (
            meta.get("owner") == self.config.owner
            and meta.get("managed-by") == "missy"
            and meta.get("tracking-id") == expected_tracking
        ):
            raise NomadOwnershipError(
                "Remote ownership metadata does not match Missy's protected operation record."
            )
        return local

    def plan(self, request_data: dict[str, Any]) -> dict[str, Any]:
        request = NomadJobRequest.from_mapping(request_data)
        request.validate(self.config)
        choose_scope(request, self.config)
        self._authorize_scope(request.namespace, request.node_pool, request.datacenter)

        if request.idempotency_key:
            for record in self.store.list_jobs():
                if (
                    record.get("idempotency_key") == request.idempotency_key
                    and record.get("job_id") != request.job_id
                    and record.get("status") not in {"complete", "failed", "cancelled", "purged"}
                ):
                    raise NomadValidationError(
                        "An active owned job already uses this idempotency key: "
                        f"{record.get('namespace')}/{record.get('job_id')}."
                    )
        active_batches = sum(
            record.get("job_type") == "batch"
            and record.get("status") not in {"complete", "failed", "cancelled", "purged"}
            for record in self.store.list_jobs()
        )
        if request.job_type == "batch" and active_batches >= self.config.max_parallel_jobs:
            raise NomadValidationError(
                f"Nomad batch concurrency limit of {self.config.max_parallel_jobs} is reached."
            )

        placement = recommend_placement(self.discover_nodes(), request, self.config)
        tracking_id = str(uuid.uuid4())
        spec = build_job(request, self.config, tracking_id=tracking_id)
        existing = [
            row for row in self.client.jobs(request.namespace) if row.get("ID") == request.job_id
        ]
        if existing:
            self._assert_owned(request.namespace, request.job_id)
        validation = self.client.validate_job(request.namespace, spec)
        raw_plan = self.client.plan_job(request.namespace, spec)
        summary = self._plan_summary(raw_plan)
        if summary["failed_task_groups"]:
            self._audit("nomad.job.plan", "deny", {"job_id": request.job_id, "summary": summary})
            raise NomadValidationError(
                "Nomad plan has failed placement for task group(s): "
                + ", ".join(summary["failed_task_groups"])
            )

        plan_id = str(uuid.uuid4())
        created = _now()
        record = {
            "plan_id": plan_id,
            "created_at": created.isoformat(),
            "expires_at": (created + timedelta(seconds=self.config.plan_ttl_seconds)).isoformat(),
            "namespace": request.namespace,
            "job_id": request.job_id,
            "job_type": request.job_type,
            "tracking_id": tracking_id,
            "idempotency_key": request.idempotency_key,
            "expected_output": request.expected_output,
            "request": asdict(request),
            "input_artifacts": request.input_artifacts,
            "output_artifacts": request.output_artifacts,
            "parameters": request.parameters,
            "result_schema": request.result_schema,
            "cleanup_policy": request.cleanup_policy,
            "max_run_seconds": request.max_run_seconds,
            "external_prerequisites": request.external_prerequisites,
            "image": request.image,
            "spec_hash": spec_hash(spec),
            "spec": spec,
            "plan_summary": summary,
            "placement": placement,
            "validation": validation[:2000],
        }
        self.store.put_plan(plan_id, record)
        self._audit(
            "nomad.job.plan",
            "allow",
            {
                "plan_id": plan_id,
                "namespace": request.namespace,
                "job_id": request.job_id,
                "spec_hash": record["spec_hash"],
                "check_index": summary["job_modify_index"],
            },
        )
        return {
            "plan_id": plan_id,
            "expires_at": record["expires_at"],
            "namespace": request.namespace,
            "job_id": request.job_id,
            "job_type": request.job_type,
            "spec_hash": record["spec_hash"],
            "image": request.image,
            "plan": summary,
            "placement": placement,
            "state": "planned_not_submitted",
        }

    def submit(self, plan_id: str) -> dict[str, Any]:
        record = self.store.get_plan(plan_id)
        if record.get("submitted_at"):
            raise NomadValidationError("This exact Nomad plan has already been submitted.")
        expires_at = datetime.fromisoformat(str(record["expires_at"]))
        if expires_at <= _now():
            raise NomadValidationError("Nomad plan expired; discover and plan again.")
        spec = record.get("spec")
        if not isinstance(spec, dict) or spec_hash(spec) != record.get("spec_hash"):
            raise NomadValidationError("Stored Nomad plan artifact failed its integrity check.")
        namespace = str(record["namespace"])
        job_id = str(record["job_id"])
        self._authorize_scope(namespace)
        check_index = int(record["plan_summary"]["job_modify_index"])
        if check_index:
            self._assert_owned(namespace, job_id)
        else:
            active_jobs = [
                item
                for item in self.store.list_jobs()
                if item.get("job_type") == "batch"
                and item.get("status")
                not in {"complete", "failed", "cancelled", "purged", "stopped"}
            ]
            if (
                record.get("job_type") == "batch"
                and len(active_jobs) >= self.config.max_parallel_jobs
            ):
                raise NomadValidationError(
                    f"Nomad batch concurrency limit of {self.config.max_parallel_jobs} is reached; re-plan later."
                )
            idempotency_key = str(record.get("idempotency_key") or "")
            if idempotency_key and any(
                item.get("idempotency_key") == idempotency_key for item in active_jobs
            ):
                raise NomadValidationError(
                    "An active owned job already uses this plan's idempotency key."
                )
        try:
            evaluation_id = self.client.run_job(namespace, spec, check_index=check_index)
        except NomadMutationUnknown:
            self._audit(
                "nomad.job.submit",
                "error",
                {"plan_id": plan_id, "namespace": namespace, "job_id": job_id, "effect": "unknown"},
            )
            raise
        except NomadCommandError as exc:
            text = str(exc).lower()
            if any(marker in text for marker in ("check-index", "modify index", "job index")):
                with contextlib.suppress(NomadCommandError):
                    self.client.inspect_job(namespace, job_id)
                self._audit(
                    "nomad.job.submit",
                    "deny",
                    {
                        "plan_id": plan_id,
                        "namespace": namespace,
                        "job_id": job_id,
                        "reason": "concurrent_job_change",
                    },
                )
                raise NomadValidationError(
                    "The Nomad job changed after planning; current state was inspected and a fresh plan is required."
                ) from exc
            raise
        self.store.mark_plan_submitted(plan_id, evaluation_id=evaluation_id)
        deployment_ids: list[str] = []
        with contextlib.suppress(NomadCommandError):
            deployment_ids = [
                str(item.get("ID"))
                for item in self.client.deployments(namespace, job_id)
                if item.get("ID")
            ]
        job_record = {
            "namespace": namespace,
            "job_id": job_id,
            "job_type": record["job_type"],
            "owner": self.config.owner,
            "tracking_id": record["tracking_id"],
            "plan_id": plan_id,
            "spec_hash": record["spec_hash"],
            "image": record["image"],
            "evaluation_id": evaluation_id,
            "idempotency_key": record.get("idempotency_key", ""),
            "expected_output": record.get("expected_output", ""),
            "request": record.get("request", {}),
            "input_artifacts": record.get("input_artifacts", {}),
            "output_artifacts": record.get("output_artifacts", {}),
            "parameters": record.get("parameters", {}),
            "result_schema": record.get("result_schema"),
            "cleanup_policy": record.get("cleanup_policy", "retain"),
            "max_run_seconds": record.get("max_run_seconds", 0),
            "placement": record.get("placement", {}),
            "check_index": check_index,
            "deployment_ids": deployment_ids,
            "submitted_at": _now().isoformat(),
            "status": "submitted",
        }
        self.store.put_job(namespace, job_id, job_record)
        self._audit(
            "nomad.job.submit",
            "allow",
            {
                "plan_id": plan_id,
                "namespace": namespace,
                "job_id": job_id,
                "evaluation_id": evaluation_id,
                "check_index": check_index,
            },
        )
        return {
            "namespace": namespace,
            "job_id": job_id,
            "evaluation_id": evaluation_id,
            "deployment_ids": deployment_ids,
            "plan_id": plan_id,
            "spec_hash": record["spec_hash"],
            "state": "submitted_not_yet_verified",
        }

    @staticmethod
    def _summarize_allocation(allocation: dict[str, Any]) -> dict[str, Any]:
        task_states = (
            allocation.get("TaskStates") if isinstance(allocation.get("TaskStates"), dict) else {}
        )
        tasks: dict[str, Any] = {}
        for name, state in task_states.items():
            if not isinstance(state, dict):
                continue
            events = state.get("Events") if isinstance(state.get("Events"), list) else []
            latest = events[-1] if events and isinstance(events[-1], dict) else {}
            tasks[str(name)] = {
                "state": state.get("State"),
                "failed": state.get("Failed"),
                "started_at": state.get("StartedAt"),
                "finished_at": state.get("FinishedAt"),
                "latest_event": {
                    "type": latest.get("Type"),
                    "display_message": latest.get("DisplayMessage"),
                    "time": latest.get("Time"),
                },
                "failure_category": NomadManager._failure_category(
                    str(latest.get("Type") or ""),
                    str(latest.get("DisplayMessage") or ""),
                ),
            }
        return {
            "id": allocation.get("ID"),
            "node_id": allocation.get("NodeID"),
            "task_group": allocation.get("TaskGroup"),
            "desired_status": allocation.get("DesiredStatus"),
            "client_status": allocation.get("ClientStatus"),
            "client_description": allocation.get("ClientDescription"),
            "create_time": allocation.get("CreateTime"),
            "modify_time": allocation.get("ModifyTime"),
            "tasks": tasks,
            "network_status": allocation.get("NetworkStatus"),
            "allocated_resources": allocation.get("AllocatedResources"),
        }

    @staticmethod
    def _failure_category(event_type: str, message: str) -> str | None:
        text = f"{event_type} {message}".lower()
        if any(item in text for item in ("download", "pull", "image")):
            return "image_pull"
        if "driver" in text:
            return "driver"
        if any(item in text for item in ("health", "check unhealthy")):
            return "health_check"
        if any(item in text for item in ("network", "address", "port")):
            return "network_reachability"
        if any(item in text for item in ("failed to place", "constraint", "resources exhausted")):
            return "scheduling"
        if any(item in text for item in ("exit", "killing", "terminated", "failed")):
            return "process"
        return None

    def status(self, namespace: str, job_id: str) -> dict[str, Any]:
        self._authorize_scope(namespace)
        job = self.client.job_status(namespace, job_id)
        allocations = self.client.allocations(namespace, job_id)
        deployments = self.client.deployments(namespace, job_id)
        local = None
        with contextlib.suppress(KeyError):
            local = self.store.get_job(namespace, job_id)
        evaluation = None
        if local and local.get("evaluation_id"):
            with contextlib.suppress(NomadCommandError):
                evaluation = self.client.evaluation(namespace, str(local["evaluation_id"]))
        allocation_summaries = [self._summarize_allocation(item) for item in allocations]
        result = {
            "observed_at": _now().isoformat(),
            "namespace": namespace,
            "job_id": job_id,
            "job_type": job.get("Type"),
            "job_status": job.get("Status"),
            "status_description": job.get("StatusDescription"),
            "node_pool": job.get("NodePool"),
            "datacenters": job.get("Datacenters"),
            "version": job.get("Version"),
            "modify_index": job.get("ModifyIndex") or job.get("JobModifyIndex"),
            "owned": local is not None,
            "evaluation": (
                {
                    "id": evaluation.get("ID"),
                    "status": evaluation.get("Status"),
                    "status_description": evaluation.get("StatusDescription"),
                    "failed_task_group_allocations": evaluation.get("FailedTGAllocs"),
                }
                if isinstance(evaluation, dict)
                else None
            ),
            "deployments": [
                {
                    "id": item.get("ID"),
                    "status": item.get("Status"),
                    "status_description": item.get("StatusDescription"),
                    "task_groups": item.get("TaskGroups"),
                }
                for item in deployments
            ],
            "allocations": allocation_summaries,
            "health": {
                "deployment_healthy": any(
                    any(
                        int(group.get("HealthyAllocs") or 0) >= int(group.get("DesiredTotal") or 1)
                        for group in (item.get("TaskGroups") or {}).values()
                        if isinstance(group, dict)
                    )
                    for item in deployments
                ),
                "failed_categories": sorted(
                    {
                        str(task["failure_category"])
                        for allocation in allocation_summaries
                        for task in allocation["tasks"].values()
                        if task.get("failure_category")
                    }
                ),
            },
            "endpoint_availability": {
                "allocation_networks": [
                    item["network_status"]
                    for item in allocation_summaries
                    if item.get("network_status")
                ],
                "externally_reachable": None,
                "note": "Nomad allocation networking is observed; external reachability is not assumed.",
            },
        }
        if local:
            source_request = local.get("request") if isinstance(local.get("request"), dict) else {}
            result["provenance"] = {
                "plan_id": local.get("plan_id"),
                "spec_hash": local.get("spec_hash"),
                "image": local.get("image"),
                "placement": local.get("placement"),
                "resources": {
                    key: source_request.get(key)
                    for key in ("cpu_mhz", "memory_mb", "disk_mb", "count")
                },
                "check_index": local.get("check_index"),
                "evaluation_id": local.get("evaluation_id"),
                "deployment_ids": local.get("deployment_ids", []),
                "rollback_status": [
                    {
                        "deployment_id": item.get("ID"),
                        "auto_revert": item.get("AutoRevert"),
                        "status": item.get("Status"),
                    }
                    for item in deployments
                ],
                "external_prerequisites": source_request.get("external_prerequisites", []),
            }
            statuses = {
                str(item.get("client_status") or "").lower() for item in allocation_summaries
            }
            if (
                local.get("job_type") == "batch"
                and statuses
                and statuses <= _TERMINAL_ALLOCATION_STATES
            ):
                local["status"] = "failed" if statuses & {"failed", "lost"} else "complete"
            elif "running" in statuses:
                local["status"] = "running"
            else:
                local["status"] = str(job.get("Status") or "pending").lower()
            local["last_observed_at"] = result["observed_at"]
            local["allocation_ids"] = [
                item["id"] for item in allocation_summaries if item.get("id")
            ]
            local["deployment_ids"] = [
                item["id"] for item in result["deployments"] if item.get("id")
            ]
            if local["status"] in {"complete", "failed"} and not local.get("completed_at"):
                local["completed_at"] = result["observed_at"]
            self.store.put_job(namespace, job_id, local)
        self._audit("nomad.job.status", "allow", {"namespace": namespace, "job_id": job_id})
        return result

    def observe(self, namespace: str, job_id: str, *, timeout_seconds: int = 120) -> dict[str, Any]:
        deadline = time.monotonic() + min(max(int(timeout_seconds), 1), 300)
        latest: dict[str, Any] = {}
        while True:
            latest = self.status(namespace, job_id)
            allocations = latest.get("allocations") or []
            statuses = {str(item.get("client_status") or "").lower() for item in allocations}
            job_type = latest.get("job_type")
            if job_type == "batch" and statuses and statuses <= _TERMINAL_ALLOCATION_STATES:
                latest["observation"] = "terminal"
                return latest
            if job_type == "service" and "running" in statuses:
                deployments = latest.get("deployments") or []
                if any(dep.get("status") == "failed" for dep in deployments):
                    latest["observation"] = "failed_deployment"
                    return latest
                healthy = any(
                    any(
                        int(group.get("HealthyAllocs") or 0) >= int(group.get("DesiredTotal") or 1)
                        for group in (dep.get("task_groups") or {}).values()
                        if isinstance(group, dict)
                    )
                    for dep in deployments
                )
                if healthy:
                    latest["observation"] = "healthy_deployment"
                    return latest
            if time.monotonic() >= deadline:
                latest["observation"] = "timeout_incomplete"
                return latest
            time.sleep(2)

    def result(self, namespace: str, job_id: str, *, log_lines: int = 100) -> dict[str, Any]:
        local = self._assert_owned(namespace, job_id)
        status = self.status(namespace, job_id)
        logs: list[dict[str, Any]] = []
        for allocation in status["allocations"][-self.config.max_group_count :]:
            allocation_id = str(allocation.get("id") or "")
            if not allocation_id:
                continue
            stdout = self.client.allocation_logs(
                namespace, allocation_id, "workload", lines=log_lines
            )
            stderr = self.client.allocation_logs(
                namespace, allocation_id, "workload", stderr=True, lines=log_lines
            )
            logs.append({"allocation_id": allocation_id, "stdout": stdout, "stderr": stderr})
        expected = str(local.get("expected_output") or "")
        marker_ok = not expected or any(expected in item["stdout"] for item in logs)
        structured_result: dict[str, Any] | None = None
        for item in logs:
            for line in item["stdout"].splitlines():
                if line.startswith("MISSY_RESULT_JSON="):
                    with contextlib.suppress(json.JSONDecodeError):
                        candidate = json.loads(line.removeprefix("MISSY_RESULT_JSON="))
                        if isinstance(candidate, dict):
                            structured_result = candidate
        schema = local.get("result_schema")
        schema_ok, schema_errors = self._validate_structured_result(structured_result, schema)
        declared_outputs = local.get("output_artifacts") or {}
        actual_outputs = (
            structured_result.get("outputs", {}) if isinstance(structured_result, dict) else {}
        )
        artifacts_ok = not declared_outputs or (
            isinstance(actual_outputs, dict)
            and all(actual_outputs.get(name) == value for name, value in declared_outputs.items())
        )
        allocation_states = {
            str(item.get("client_status") or "").lower() for item in status["allocations"]
        }
        terminal_success = bool(allocation_states) and allocation_states <= {"complete"}
        complete = terminal_success and marker_ok and schema_ok and artifacts_ok
        if local.get("status") == "cancelled":
            outcome = "cancelled"
        elif allocation_states & {"failed", "lost"}:
            outcome = "failed"
        elif complete:
            outcome = "completed"
        elif terminal_success:
            outcome = "result_validation_failed"
        else:
            submitted = datetime.fromisoformat(str(local["submitted_at"]))
            max_seconds = int(local.get("max_run_seconds") or 0)
            outcome = (
                "timed_out"
                if max_seconds and (_now() - submitted).total_seconds() > max_seconds
                else "running_or_pending"
            )
        response = {
            "namespace": namespace,
            "job_id": job_id,
            "complete": complete,
            "expected_output": expected or None,
            "expected_output_verified": marker_ok,
            "declared_outputs": declared_outputs,
            "output_artifacts_verified": artifacts_ok,
            "structured_result": structured_result,
            "result_schema_verified": schema_ok,
            "result_schema_errors": schema_errors,
            "outcome": outcome,
            "provenance": {
                "input_artifacts": local.get("input_artifacts", {}),
                "image": local.get("image"),
                "parameters": local.get("parameters", {}),
                "placement": local.get("placement", {}),
                "submitted_at": local.get("submitted_at"),
                "completed_at": local.get("completed_at"),
                "evaluation_id": local.get("evaluation_id"),
                "allocation_ids": local.get("allocation_ids", []),
            },
            "status": status,
            "logs": logs,
            "cleanup_ready": complete,
        }
        if complete:
            local["status"] = "complete"
            local["result_verified_at"] = _now().isoformat()
            self.store.put_job(namespace, job_id, local)
        self._audit(
            "nomad.job.result",
            "allow" if complete else "error",
            {"namespace": namespace, "job_id": job_id, "outcome": outcome},
        )
        return response

    @staticmethod
    def _validate_structured_result(
        value: dict[str, Any] | None, schema: dict[str, Any] | None
    ) -> tuple[bool, list[str]]:
        if not schema:
            return True, []
        if not isinstance(value, dict):
            return False, ["MISSY_RESULT_JSON object was not found"]
        errors: list[str] = []
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"missing required field {name!r}")
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        for name, definition in schema.get("properties", {}).items():
            if name not in value:
                continue
            expected_type = definition["type"]
            actual = value[name]
            valid = isinstance(actual, type_map[expected_type])
            if expected_type in {"number", "integer"} and isinstance(actual, bool):
                valid = False
            if not valid:
                errors.append(f"field {name!r} is not {expected_type}")
        return not errors, errors

    def plan_scale(self, namespace: str, job_id: str, count: int) -> dict[str, Any]:
        """Plan a service scale as a full validated optimistic update."""
        self._authorize_scope(namespace)
        local = self._assert_owned(namespace, job_id)
        if local.get("job_type") != "service":
            raise NomadValidationError("Only an owned service job can be scaled.")
        request = copy.deepcopy(local.get("request"))
        if not isinstance(request, dict):
            raise NomadValidationError("The owned job has no reproducible source request.")
        request["count"] = int(count)
        result = self.plan(request)
        result["operation"] = "scale"
        result["previous_count"] = (
            local.get("request", {}).get("count")
            if isinstance(local.get("request"), dict)
            else None
        )
        return result

    def plan_offload(
        self,
        workload_template: str,
        *,
        parameters: dict[str, str] | None = None,
        input_artifacts: dict[str, str] | None = None,
        output_artifacts: dict[str, str] | None = None,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        """Create a durable task backed by an explicitly approved template."""
        task_id = new_task_id()
        request = build_offload_request(
            self.config,
            workload_template,
            parameters=parameters,
            input_artifacts=input_artifacts,
            output_artifacts=output_artifacts,
            idempotency_key=idempotency_key or task_id,
        )
        planned = self.plan(asdict(request))
        record = {
            "task_id": task_id,
            "workload_template": workload_template,
            "created_at": _now().isoformat(),
            "plan_id": planned["plan_id"],
            "namespace": planned["namespace"],
            "job_id": planned["job_id"],
            "state": "planned_not_submitted",
        }
        self.store.put_task(task_id, record)
        self._audit(
            "nomad.offload.plan",
            "allow",
            {"task_id": task_id, "template": workload_template, "job_id": planned["job_id"]},
        )
        return {"task_id": task_id, **planned}

    def submit_offload(self, task_id: str) -> dict[str, Any]:
        task = self.store.get_task(task_id)
        plan = self.store.get_plan(str(task["plan_id"]))
        if plan.get("submitted_at"):
            job = self.store.get_job(str(task["namespace"]), str(task["job_id"]))
            submitted = {
                "namespace": task["namespace"],
                "job_id": task["job_id"],
                "evaluation_id": job.get("evaluation_id"),
                "plan_id": task["plan_id"],
                "spec_hash": job.get("spec_hash"),
                "state": "submitted_not_yet_verified",
                "recovered_from_journal": True,
            }
        else:
            submitted = self.submit(str(task["plan_id"]))
        task.update(
            {
                "state": submitted["state"],
                "evaluation_id": submitted["evaluation_id"],
                "submitted_at": _now().isoformat(),
            }
        )
        self.store.put_task(task_id, task)
        return {"task_id": task_id, **submitted}

    def offload_status(self, task_id: str) -> dict[str, Any]:
        task = self.store.get_task(task_id)
        status = self.status(str(task["namespace"]), str(task["job_id"]))
        task["state"] = self._state_from_status(status)
        task["last_observed_at"] = _now().isoformat()
        self.store.put_task(task_id, task)
        return {"task_id": task_id, "task": task, "status": status}

    def offload_result(self, task_id: str) -> dict[str, Any]:
        task = self.store.get_task(task_id)
        result = self.result(str(task["namespace"]), str(task["job_id"]))
        task["state"] = str(result["outcome"])
        if result["outcome"] in {
            "completed",
            "failed",
            "cancelled",
            "timed_out",
            "result_validation_failed",
        }:
            task["completed_at"] = _now().isoformat()
        self.store.put_task(task_id, task)
        return {"task_id": task_id, **result}

    def cancel_offload(self, task_id: str) -> dict[str, Any]:
        task = self.store.get_task(task_id)
        result = self.action(str(task["namespace"]), str(task["job_id"]), "cancel")
        task["state"] = "cancelled"
        task["completed_at"] = _now().isoformat()
        self.store.put_task(task_id, task)
        return {"task_id": task_id, **result}

    @staticmethod
    def _state_from_status(status: dict[str, Any]) -> str:
        allocation_states = {
            str(item.get("client_status") or "").lower() for item in status.get("allocations") or []
        }
        if allocation_states & {"failed", "lost"}:
            return "failed"
        if allocation_states and allocation_states <= {"complete"}:
            return "completed_unverified"
        if "running" in allocation_states:
            return "running"
        return str(status.get("job_status") or "pending").lower()

    def plan_benchmark(self, definition_data: dict[str, Any]) -> dict[str, Any]:
        definition = BenchmarkDefinition.from_mapping(definition_data)
        benchmark_id = new_benchmark_id()
        expanded = definition.requests(self.config, benchmark_id)
        capacity_parallelism = self._benchmark_capacity(self.recommend(expanded[0]["request"]))
        if definition.parallelism > capacity_parallelism:
            raise NomadValidationError(
                f"Benchmark parallelism {definition.parallelism} exceeds observed authorized "
                f"capacity for {capacity_parallelism} concurrent run(s)."
            )
        runs: list[dict[str, Any]] = []
        for item in expanded:
            plan = self.plan(item["request"])
            runs.append(
                {
                    "kind": item["kind"],
                    "run_number": item["run_number"],
                    "plan_id": plan["plan_id"],
                    "namespace": plan["namespace"],
                    "job_id": plan["job_id"],
                    "state": "planned_not_submitted",
                    "placement": plan["placement"],
                }
            )
        record = {
            "benchmark_id": benchmark_id,
            "definition": asdict(definition),
            "created_at": _now().isoformat(),
            "parallelism": definition.parallelism,
            "observed_capacity_parallelism": capacity_parallelism,
            "state": "planned_not_submitted",
            "runs": runs,
        }
        self.store.put_benchmark(benchmark_id, record)
        self._audit(
            "nomad.benchmark.plan",
            "allow",
            {"benchmark_id": benchmark_id, "run_count": len(runs)},
        )
        return copy.deepcopy(record)

    def _benchmark_capacity(self, placement: dict[str, Any]) -> int:
        demand = placement.get("demand") if isinstance(placement.get("demand"), dict) else {}
        candidates = (
            placement.get("candidates") if isinstance(placement.get("candidates"), list) else []
        )
        non_protected = [
            item
            for item in candidates
            if isinstance(item, dict) and item.get("fits") and not item.get("protected")
        ]
        usable = non_protected or [
            item for item in candidates if isinstance(item, dict) and item.get("fits")
        ]
        total_slots = 0
        for candidate in usable:
            headroom = candidate.get("scheduler_headroom")
            if not isinstance(headroom, dict):
                continue
            slots = min(
                int(headroom.get(resource) or 0) // max(1, int(demand.get(resource) or 0))
                for resource in ("cpu_mhz", "memory_mb", "disk_mb")
            )
            total_slots += slots
        return max(0, min(total_slots, self.config.max_benchmark_parallelism))

    def start_benchmark(self, benchmark_id: str) -> dict[str, Any]:
        benchmark = self.store.get_benchmark(benchmark_id)
        if benchmark.get("state") not in {"planned_not_submitted", "running"}:
            raise NomadValidationError("Benchmark is not startable in its current state.")
        return self._advance_benchmark(benchmark, submit_pending=True)

    def benchmark_status(self, benchmark_id: str) -> dict[str, Any]:
        benchmark = self.store.get_benchmark(benchmark_id)
        return self._advance_benchmark(benchmark, submit_pending=True)

    def _advance_benchmark(
        self, benchmark: dict[str, Any], *, submit_pending: bool
    ) -> dict[str, Any]:
        terminal = {
            "completed_unverified",
            "completed",
            "failed",
            "cancelled",
            "timed_out",
            "result_validation_failed",
        }
        for run in benchmark["runs"]:
            if run["state"] in {"submitted_not_yet_verified", "running", "pending", "dead"}:
                try:
                    status = self.status(str(run["namespace"]), str(run["job_id"]))
                    run["state"] = self._state_from_status(status)
                    run["allocation_ids"] = [
                        item["id"] for item in status["allocations"] if item.get("id")
                    ]
                    node_ids = sorted(
                        {
                            str(item["node_id"])
                            for item in status["allocations"]
                            if item.get("node_id")
                        }
                    )
                    run["nodes"] = [self._benchmark_node(node_id) for node_id in node_ids]
                except NomadCommandError as exc:
                    run["observation_error"] = str(exc)
        active = sum(
            run["state"] not in terminal | {"planned_not_submitted"} for run in benchmark["runs"]
        )
        if submit_pending:
            for run in benchmark["runs"]:
                if run["state"] != "planned_not_submitted" or active >= int(
                    benchmark["parallelism"]
                ):
                    continue
                submitted = self.submit(str(run["plan_id"]))
                run.update(
                    {
                        "state": submitted["state"],
                        "evaluation_id": submitted["evaluation_id"],
                        "submitted_at": _now().isoformat(),
                    }
                )
                active += 1
                # Persist after every mutation so a later submission failure or
                # process crash cannot make an already-submitted run look pending.
                benchmark["state"] = "running"
                self.store.put_benchmark(str(benchmark["benchmark_id"]), benchmark)
        states = {run["state"] for run in benchmark["runs"]}
        if states <= terminal:
            benchmark["state"] = (
                "complete_with_failures"
                if states - {"completed", "completed_unverified"}
                else "complete"
            )
            benchmark.setdefault("completed_at", _now().isoformat())
        elif any(state not in {"planned_not_submitted"} for state in states):
            benchmark["state"] = "running"
        self.store.put_benchmark(str(benchmark["benchmark_id"]), benchmark)
        return copy.deepcopy(benchmark)

    def _benchmark_node(self, node_id: str) -> dict[str, Any]:
        try:
            node = self.client.node(node_id)
        except NomadCommandError:
            return {"node_id": node_id, "details_available": False}
        attrs = node.get("Attributes") if isinstance(node.get("Attributes"), dict) else {}
        return {
            "node_id": node_id,
            "name": node.get("Name"),
            "datacenter": node.get("Datacenter"),
            "node_pool": node.get("NodePool"),
            "architecture": attrs.get("cpu.arch") or attrs.get("kernel.arch"),
            "resources": node.get("NodeResources"),
            "details_available": True,
        }

    def benchmark_results(self, benchmark_id: str) -> dict[str, Any]:
        benchmark = self._advance_benchmark(
            self.store.get_benchmark(benchmark_id), submit_pending=False
        )
        results: list[dict[str, Any]] = []
        for run in benchmark["runs"]:
            if run["state"] in {"completed_unverified", "completed", "failed"}:
                result = self.result(str(run["namespace"]), str(run["job_id"]))
                run["state"] = result["outcome"]
                run["result"] = result
            results.append(copy.deepcopy(run))
        measured = [run for run in results if run["kind"] == "measured"]
        condition_keys = {
            json.dumps(run.get("nodes", []), sort_keys=True, default=str) for run in measured
        }
        comparison_valid = len(condition_keys) <= 1 and all(
            run["state"] == "completed" for run in measured
        )
        benchmark["runs"] = results
        benchmark["comparison"] = {
            "valid": comparison_valid,
            "reason": (
                "all measured runs completed under matching observed node conditions"
                if comparison_valid
                else "measured runs failed, remain incomplete, or used different observed node conditions"
            ),
        }
        self.store.put_benchmark(benchmark_id, benchmark)
        return benchmark

    def cancel_benchmark(self, benchmark_id: str) -> dict[str, Any]:
        benchmark = self.store.get_benchmark(benchmark_id)
        for run in benchmark["runs"]:
            if run["state"] not in {"planned_not_submitted", "completed", "failed", "cancelled"}:
                self.action(str(run["namespace"]), str(run["job_id"]), "cancel")
                run["state"] = "cancelled"
            elif run["state"] == "planned_not_submitted":
                run["state"] = "cancelled_before_submission"
        benchmark["state"] = "cancelled"
        benchmark["completed_at"] = _now().isoformat()
        self.store.put_benchmark(benchmark_id, benchmark)
        return benchmark

    def action(
        self,
        namespace: str,
        job_id: str,
        action: str,
        *,
        confirm_purge: bool = False,
    ) -> dict[str, Any]:
        self._authorize_scope(namespace)
        local = self._assert_owned(namespace, job_id)
        action = action.lower()
        if action == "restart":
            outcome = self.client.restart_job(namespace, job_id)
            new_status = "restart_requested"
        elif action in {"stop", "cancel"}:
            outcome = self.client.stop_job(namespace, job_id, purge=False)
            new_status = "cancelled" if action == "cancel" else "stopped"
        elif action == "purge":
            if not self.config.allow_purge or not confirm_purge:
                raise NomadValidationError(
                    "Purge requires nomad.allow_purge=true and confirm_purge=true; it removes job history."
                )
            request = local.get("request") if isinstance(local.get("request"), dict) else {}
            if not request.get("disposable"):
                raise NomadValidationError(
                    "Purge is limited to jobs explicitly declared disposable in their reviewed request."
                )
            outcome = self.client.stop_job(namespace, job_id, purge=True)
            new_status = "purged"
        else:
            raise NomadValidationError("action must be restart, stop, cancel, or purge.")
        local["status"] = new_status
        local["last_action"] = action
        local["last_action_at"] = _now().isoformat()
        self.store.put_job(namespace, job_id, local)
        self._audit(
            "nomad.job.action",
            "allow",
            {"namespace": namespace, "job_id": job_id, "action": action},
        )
        return {
            "namespace": namespace,
            "job_id": job_id,
            "action": action,
            "state": new_status,
            "nomad_output": outcome,
            "persistent_data_detected": bool(
                isinstance(local.get("request"), dict) and local["request"].get("stateful")
            ),
            "persistent_data_deleted": False,
        }

    def reconcile(self) -> dict[str, Any]:
        reconciled: list[dict[str, Any]] = []
        for local in self.store.list_jobs():
            namespace = str(local.get("namespace") or "")
            job_id = str(local.get("job_id") or "")
            if not namespace or not job_id:
                continue
            try:
                remote = self.client.inspect_job(namespace, job_id)
                self._assert_owned(namespace, job_id, remote)
                status = self.status(namespace, job_id)
                outcome = {"namespace": namespace, "job_id": job_id, "state": status["job_status"]}
            except NomadOwnershipError:
                outcome = {"namespace": namespace, "job_id": job_id, "state": "ownership_mismatch"}
            except NomadCommandError as exc:
                outcome = {
                    "namespace": namespace,
                    "job_id": job_id,
                    "state": "missing_or_unavailable",
                    "error": str(exc),
                }
            reconciled.append(outcome)
        self._audit("nomad.reconcile", "allow", {"job_count": len(reconciled)})
        return {"observed_at": _now().isoformat(), "jobs": reconciled}
