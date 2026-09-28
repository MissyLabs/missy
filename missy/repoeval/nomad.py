"""Build safe, bounded Nomad job plans. Deliberately has no submit client."""

import re
from collections.abc import Mapping
from dataclasses import dataclass

from .placement import CapacityBudget, ResourceEnvelope, capacity_budget, select_pool


class JobPlanError(ValueError):
    """Input fails the fixed safe worker contract."""


IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:+-]*@sha256:[0-9a-f]{64}$")
ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{7,127}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
ALLOWED_WORKLOADS = frozenset({"repository-orientation", "tool-call", "patch-test-repair"})


@dataclass(frozen=True)
class JobRequest:
    run_id: str
    repository_commit_sha: str
    snapshot_id: str
    image_digest: str
    workload_class: str
    cpu_mhz: int
    memory_mb: int
    disk_mb: int
    timeout_seconds: int
    namespace: str = "default"
    datacenter: str = "dc1"


@dataclass(frozen=True)
class JobPlan:
    job_id: str
    pool: str
    namespace: str
    datacenter: str
    envelope: ResourceEnvelope
    job: Mapping[str, object]
    capacity_budget: CapacityBudget


def _label(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}", value):
        raise JobPlanError(f"invalid {label}")
    return value


def plan_job(request: JobRequest, pools) -> JobPlan:
    """Validate identity/resources and produce a plan only, no cluster effects."""
    if not isinstance(request, JobRequest):
        raise JobPlanError("request must be JobRequest")
    if not isinstance(request.run_id, str) or not ID_RE.fullmatch(request.run_id):
        raise JobPlanError("run_id must be an immutable stable identifier")
    if not isinstance(request.repository_commit_sha, str) or not COMMIT_RE.fullmatch(
        request.repository_commit_sha
    ):
        raise JobPlanError("repository commit must be an exact immutable SHA")
    if not isinstance(request.snapshot_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:-]{0,254}", request.snapshot_id
    ):
        raise JobPlanError("immutable snapshot identity required")
    if not isinstance(request.image_digest, str) or not IMAGE_RE.fullmatch(request.image_digest):
        raise JobPlanError("execution image must be pinned by sha256 digest")
    if request.workload_class not in ALLOWED_WORKLOADS:
        raise JobPlanError("workload class is not an approved fixed template")
    if (
        isinstance(request.timeout_seconds, bool)
        or not isinstance(request.timeout_seconds, int)
        or not 1 <= request.timeout_seconds <= 86400
    ):
        raise JobPlanError("timeout_seconds outside finite limit")
    namespace = _label(request.namespace, "namespace")
    dc = _label(request.datacenter, "datacenter")
    envelope = ResourceEnvelope(request.cpu_mhz, request.memory_mb, request.disk_mb)
    snapshots = tuple(pools)
    budget = capacity_budget(snapshots)
    pool = select_pool(envelope, snapshots)
    job_id = f"foundry-{request.run_id}"
    # Use the Nomad API's JSON job shape, not HCL/lowercase task aliases.
    # MaxRunDuration is enforced by Nomad at the task-group level; the env
    # value is only information for the worker and cannot enforce a deadline.
    task = {
        "Name": "worker",
        "Driver": "docker",
        "Config": {
            "image": request.image_digest,
            "readonly_rootfs": True,
            "network_mode": "none",
            "cap_drop": ["ALL"],
            "privileged": False,
            "force_pull": True,
        },
        "Resources": {"CPU": envelope.cpu_mhz, "MemoryMB": envelope.memory_mb},
        "KillTimeout": 5 * 1_000_000_000,
        "KillSignal": "SIGTERM",
        "Env": {"FOUNDRY_TIMEOUT_SECONDS": str(request.timeout_seconds)},
    }
    job = {
        "ID": job_id,
        "Name": job_id,
        "Type": "batch",
        "Namespace": namespace,
        "Datacenters": [dc],
        "NodePool": pool.name,
        "Priority": 50,
        "TaskGroups": [
            {
                "Name": "worker",
                "Count": 1,
                "RestartPolicy": {
                    "Attempts": 0,
                    "Interval": 60 * 1_000_000_000,
                    "Delay": 5 * 1_000_000_000,
                    "Mode": "fail",
                },
                "ReschedulePolicy": {
                    "Attempts": 0,
                    "Interval": 3600 * 1_000_000_000,
                    "Unlimited": False,
                },
                "MaxRunDuration": request.timeout_seconds * 1_000_000_000,
                "EphemeralDisk": {"SizeMB": envelope.disk_mb, "Sticky": False, "Migrate": False},
                "Tasks": [task],
            }
        ],
        "Meta": {
            "foundry_run_id": request.run_id,
            "foundry_workload_class": request.workload_class,
            "repository_commit_sha": request.repository_commit_sha,
            "snapshot_id": request.snapshot_id,
        },
    }
    return JobPlan(job_id, pool.name, namespace, dc, envelope, job, budget)
