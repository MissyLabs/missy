"""Fail-closed, opt-in RepoEval bridge to the existing authenticated NomadClient.

No client is constructed here. The trusted expected-job resolver must read an
immutable durable reservation, not take jobs from an HTTP request or the model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from missy.nomad.client import NomadClient

from .dispatch import JobObservation
from .nomad import ALLOWED_WORKLOADS, IMAGE_RE, JobRequest, plan_job
from .placement import PoolCapacity


class NomadAdapterError(ValueError):
    """Scope, immutable job, or authoritative scheduler evidence is invalid."""


# This adapter is deliberately narrower than the generic Nomad integration.
NAMESPACE = "sandbox"
DATACENTER = "dc1"
POOL = "staging"
_RUN = re.compile(r"^run-[A-Za-z0-9_-]{8,128}$")
_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_SNAPSHOT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$")
_JOB = re.compile(r"^foundry-run-[A-Za-z0-9_-]{8,128}$")
_SERVER_FIELDS = frozenset({"CreateIndex", "ModifyIndex", "JobModifyIndex"})


def _copy(value: Any) -> Any:
    """Copy finite JSON, rejecting custom objects and non-finite numbers."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


class NomadSchedulerAdapter:
    """Scheduler protocol for one reviewed image and one staging placement.

    expected_job is an operator-owned lookup of the *persisted reservation*.
    Missing evidence blocks submit/lookup/stop. The outbox must still perform
    independent project authorization and commit attempt markers before calls.
    """

    def __init__(
        self,
        client: NomadClient,
        expected_job: Callable[[str, str], Mapping[str, Any] | None],
        *,
        approved_image_digest: str,
        expected_create_index: Callable[[str, str], int | None] | None = None,
        enabled: bool = False,
    ) -> None:
        if not isinstance(client, NomadClient):
            raise NomadAdapterError("authenticated NomadClient required")
        if not callable(expected_job):
            raise NomadAdapterError("trusted immutable job resolver required")
        if not isinstance(approved_image_digest, str) or not IMAGE_RE.fullmatch(
            approved_image_digest
        ):
            raise NomadAdapterError("reviewed immutable image digest required")
        if type(enabled) is not bool:
            raise NomadAdapterError("enabled must be an explicit boolean")
        self.client = client
        self.expected_job = expected_job
        self.expected_create_index = expected_create_index
        self.approved_image_digest = approved_image_digest
        self.enabled = enabled

    def _scope(self, namespace: str) -> None:
        if not self.enabled:
            raise NomadAdapterError("RepoEval Nomad adapter disabled")
        config = self.client.config
        if (
            namespace != NAMESPACE
            or config.enabled is not True
            or NAMESPACE not in config.allowed_namespaces
            or DATACENTER not in config.allowed_datacenters
            or POOL not in config.allowed_node_pools
        ):
            raise NomadAdapterError("Nomad staging scope not authorized")

    def _canonical(self, namespace: str, job: Mapping[str, Any]) -> dict[str, Any]:
        """Regenerate every wire key from a bounded fixed template, then compare."""
        self._scope(namespace)
        try:
            supplied = _copy(job)
            if not isinstance(supplied, dict) or len(json.dumps(supplied)) > 256_000:
                raise ValueError("invalid job")
            meta = supplied["Meta"]
            run_id = meta["foundry_run_id"]
            job_id = supplied["ID"]
            commit = meta["repository_commit_sha"]
            snapshot = meta["snapshot_id"]
            workload = meta["foundry_workload_class"]
            group = supplied["TaskGroups"][0]
            task = group["Tasks"][0]
            resource = task["Resources"]
            timeout_ns = group["MaxRunDuration"]
            disk = group["EphemeralDisk"]["SizeMB"]
            image = task["Config"]["image"]
            if (
                not isinstance(run_id, str)
                or not _RUN.fullmatch(run_id)
                or not isinstance(job_id, str)
                or not _JOB.fullmatch(job_id)
                or job_id != f"foundry-{run_id}"
                or not isinstance(commit, str)
                or not _COMMIT.fullmatch(commit)
                or not isinstance(snapshot, str)
                or not _SNAPSHOT.fullmatch(snapshot)
                or workload not in ALLOWED_WORKLOADS
                or image != self.approved_image_digest
                or type(timeout_ns) is not int
                or timeout_ns % 1_000_000_000
                or supplied["Namespace"] != namespace
                or supplied["Datacenters"] != [DATACENTER]
                or supplied["NodePool"] != POOL
            ):
                raise ValueError("job identity or placement differs")
            request = JobRequest(
                run_id=run_id,
                repository_commit_sha=commit,
                snapshot_id=snapshot,
                image_digest=image,
                workload_class=workload,
                cpu_mhz=resource["CPU"],
                memory_mb=resource["MemoryMB"],
                disk_mb=disk,
                timeout_seconds=timeout_ns // 1_000_000_000,
                namespace=NAMESPACE,
                datacenter=DATACENTER,
            )
            # Pure plan_job needs capacity snapshots only to select staging. A
            # synthetic finite envelope is not a live capacity attestation.
            plan = plan_job(request, [PoolCapacity(POOL, 32_000, 131_072, 262_144)])
            canonical = _copy(plan.job)
            if supplied != canonical:
                raise ValueError("job differs from fixed template")
            return canonical
        except (KeyError, IndexError, TypeError, ValueError, OverflowError):
            # Do not echo job bytes, credentials, or untrusted exception text.
            raise NomadAdapterError("RepoEval job differs from reviewed template") from None

    def _expected(self, namespace: str, job_id: str) -> dict[str, Any]:
        self._scope(namespace)
        if not isinstance(job_id, str) or not _JOB.fullmatch(job_id):
            raise NomadAdapterError("invalid RepoEval job identity")
        try:
            reserved = self.expected_job(namespace, job_id)
        except Exception:
            raise NomadAdapterError("durable job evidence unavailable") from None
        if reserved is None:
            raise NomadAdapterError("durable job evidence missing")
        expected = self._canonical(namespace, reserved)
        if expected["ID"] != job_id:
            raise NomadAdapterError("durable job identity mismatch")
        return expected

    def submit(self, namespace: str, job: Mapping[str, Any]) -> None:
        canonical = self._canonical(namespace, job)
        if canonical != self._expected(namespace, canonical["ID"]):
            raise NomadAdapterError("job differs from durable reservation")
        # There is no sealed snapshot mount, custody receipt, approval receipt,
        # or credential broker in this fixed workload. Even an explicit enable
        # cannot make this execution safe. Keep interface but refuse mutation.
        raise NomadAdapterError("RepoEval Nomad workload custody is not implemented")

    def lookup(self, namespace: str, job_id: str) -> JobObservation | None:
        expected = self._expected(namespace, job_id)
        # The CLI does not expose a typed 404. Treat all inspect failures as
        # unknown rather than guessing absence from error text.
        inspected = self.client.inspect_job(namespace, job_id)
        if not isinstance(inspected, dict):
            raise NomadAdapterError("scheduler job evidence malformed")
        if (
            _copy({key: value for key, value in inspected.items() if key not in _SERVER_FIELDS})
            != expected
        ):
            raise NomadAdapterError("scheduler job differs from durable reservation")
        status = self.client.job_status(namespace, job_id)
        if (
            not isinstance(status, dict)
            or status.get("ID") != job_id
            or status.get("Namespace") != namespace
            or type(status.get("CreateIndex")) is not int
            or status["CreateIndex"] <= 0
            or type(inspected.get("CreateIndex")) is not int
            or inspected["CreateIndex"] != status["CreateIndex"]
        ):
            raise NomadAdapterError("scheduler status identity unavailable")
        # Real Nomad inspect may add server-side defaults; unknown fields
        # deliberately fail closed rather than silently ignoring capabilities.
        state = status.get("Status")
        if status.get("Stop") is True and state == "dead":
            mapped = "stopped"
        elif state == "pending":
            mapped = "pending"
        elif state == "running":
            mapped = "running"
        elif state == "dead":
            allocs = self.client.allocations(namespace, job_id)
            if not allocs:
                raise NomadAdapterError("terminal allocation evidence missing")
            for alloc in allocs:
                if (
                    not isinstance(alloc, dict)
                    or alloc.get("JobID") != job_id
                    or alloc.get("Namespace") != namespace
                    or alloc.get("ClientStatus") not in {"complete", "failed"}
                ):
                    raise NomadAdapterError("terminal allocation evidence invalid")
            mapped = "failed" if any(a["ClientStatus"] == "failed" for a in allocs) else "complete"
        else:
            raise NomadAdapterError("scheduler status unresolved")
        return JobObservation(expected, mapped)

    def stop(self, namespace: str, job_id: str) -> None:
        # A durable dispatcher claims the single stop attempt before calling
        # here. Only a separately persisted incarnation can authorize a stop;
        # the current inspect/status index alone is attacker-replaceable.
        if not callable(self.expected_create_index):
            raise NomadAdapterError("durable scheduler incarnation required for stop")
        try:
            pinned = self.expected_create_index(namespace, job_id)
        except Exception:
            raise NomadAdapterError("durable scheduler incarnation unavailable") from None
        if type(pinned) is not int or pinned <= 0:
            raise NomadAdapterError("durable scheduler incarnation missing")
        observed = self.lookup(namespace, job_id)
        if observed is None:
            raise NomadAdapterError("scheduler job not observed")
        if self.client.job_status(namespace, job_id).get("CreateIndex") != pinned:
            raise NomadAdapterError("scheduler incarnation differs from durable reservation")
        if observed.state != "stopped":
            # NomadClient.stop_job cannot atomically check the incarnation.
            # A replacement can reuse the ID between inspection and stop.
            raise NomadAdapterError("atomic incarnation-fenced stop unavailable")
