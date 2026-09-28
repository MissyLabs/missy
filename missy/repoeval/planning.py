"""Read-only staging placement and operator-owned project policy evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from .placement import PoolCapacity, ResourceEnvelope, capacity_budget, select_pool

CHECKS = ("repository", "image", "providers", "budget", "quota", "egress", "audit")


class PlanningError(ValueError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class CapacitySnapshot:
    revision: str
    observed_at: float
    pools: tuple[PoolCapacity, ...]


@dataclass(frozen=True)
class ProjectPolicySnapshot:
    """Trusted facts, not caller-supplied approval booleans."""

    project_id: str
    revision: str
    observed_at: float
    repositories: frozenset[str]
    images: frozenset[str]
    provider_models: Mapping[str, frozenset[str]]
    quota: ResourceEnvelope
    remaining_budget_units: int  # upper bound in CPU MHz-seconds
    egress_policies: frozenset[str]
    audit_sink_id: str


class CapacityProvider(Protocol):
    def __call__(self) -> CapacitySnapshot: ...


class PolicyProvider(Protocol):
    def __call__(self, project_id: str) -> ProjectPolicySnapshot: ...


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class PlanningAuthority:
    """All providers must be injected by trusted host code, never the API caller.

    A successful recheck does not reserve resources; the scheduler must enforce
    staging-only placement and resource ceilings independently.
    """

    def __init__(
        self,
        capacity_provider: CapacityProvider,
        policy_provider: PolicyProvider,
        *,
        clock: Callable[[], float],
        ttl_seconds: int = 30,
        freshness_seconds: int = 15,
    ) -> None:
        if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= 300:
            raise ValueError("Invalid plan TTL")
        if type(freshness_seconds) is not int or not 1 <= freshness_seconds <= 300:
            raise ValueError("Invalid snapshot freshness window")
        self.capacity_provider = capacity_provider
        self.policy_provider = policy_provider
        self.clock = clock
        self.ttl_seconds = ttl_seconds
        self.freshness_seconds = freshness_seconds

    def attest(self, project_id: str, workload: dict) -> dict:
        try:
            capacity = self.capacity_provider()
            policy = self.policy_provider(project_id)
            now = self.clock()
        except Exception as exc:
            raise PlanningError("unavailable", "Trusted planning evidence unavailable") from exc
        if not isinstance(capacity, CapacitySnapshot) or not isinstance(
            policy, ProjectPolicySnapshot
        ):
            raise PlanningError("evidence", "Trusted planning evidence malformed")
        if policy.project_id != project_id or not isinstance(project_id, str) or not project_id:
            raise PlanningError("authorization", "Project policy scope differs")
        if any(not isinstance(r, str) or not r for r in (capacity.revision, policy.revision)):
            raise PlanningError("evidence", "Snapshot revision missing")
        if any(
            isinstance(t, bool)
            or not isinstance(t, (int, float))
            or not math.isfinite(t)
            or t > now
            or now - t > self.freshness_seconds
            for t in (now, capacity.observed_at, policy.observed_at)
        ):
            raise PlanningError("evidence", "Planning snapshot stale or invalid")
        if not isinstance(capacity.pools, tuple) or not all(
            isinstance(pool, PoolCapacity) and type(pool.eligible) is bool
            for pool in capacity.pools
        ):
            raise PlanningError("evidence", "Capacity pools malformed")
        if (
            not isinstance(policy.repositories, frozenset)
            or not isinstance(policy.images, frozenset)
            or not isinstance(policy.provider_models, Mapping)
            or not all(
                isinstance(key, str)
                and isinstance(models, frozenset)
                and all(isinstance(model, str) for model in models)
                for key, models in policy.provider_models.items()
            )
            or not isinstance(policy.egress_policies, frozenset)
            or not isinstance(policy.quota, ResourceEnvelope)
            or type(policy.remaining_budget_units) is not int
            or not isinstance(policy.audit_sink_id, str)
        ):
            raise PlanningError("evidence", "Project policy malformed")
        try:
            sandbox = workload["sandbox"]
            envelope = ResourceEnvelope(
                sandbox["cpu_mhz"], sandbox["memory_mb"], sandbox["disk_mb"]
            )
            targets = workload["providers"]
            execution = workload["execution"]
            estimated_budget_units = (
                envelope.cpu_mhz
                * execution["timeout_seconds"]
                * (execution["warmups"] + execution["repetitions"])
                * execution["parallelism"]
                * execution["max_attempts"]
                * len(targets)
            )
            checks = {
                "repository": workload["repository"]["repository_id"] in policy.repositories,
                "image": sandbox["image_digest"] in policy.images,
                "providers": bool(targets)
                and all(
                    isinstance(t, dict)
                    and t.get("model") in policy.provider_models.get(t.get("registry_key"), ())
                    for t in targets
                ),
                "budget": type(estimated_budget_units) is int
                and estimated_budget_units > 0
                and policy.remaining_budget_units >= estimated_budget_units,
                "quota": all(
                    getattr(envelope, field) <= getattr(policy.quota, field)
                    for field in ("cpu_mhz", "memory_mb", "disk_mb")
                ),
                "egress": sandbox["network_policy"] in policy.egress_policies,
                "audit": bool(policy.audit_sink_id.strip()),
            }
            if not all(checks.values()):
                raise PlanningError("policy", "Trusted project policy denies workload")
            budget = capacity_budget(capacity.pools)
            selected = select_pool(envelope, capacity.pools)
            policy_hash = _digest(
                {
                    "project_id": policy.project_id,
                    "revision": policy.revision,
                    "observed_at": policy.observed_at,
                    "repositories": sorted(policy.repositories),
                    "images": sorted(policy.images),
                    "provider_models": {k: sorted(v) for k, v in policy.provider_models.items()},
                    "quota": vars(policy.quota),
                    "remaining_budget_units": policy.remaining_budget_units,
                    "egress_policies": sorted(policy.egress_policies),
                    "audit_sink_id": policy.audit_sink_id,
                }
            )
            capacity_hash = _digest(
                {
                    "revision": capacity.revision,
                    "observed_at": capacity.observed_at,
                    "pools": [vars(pool) for pool in capacity.pools],
                }
            )
        except PlanningError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise PlanningError(
                "capacity", "Staging capacity or planning evidence insufficient"
            ) from exc
        return {
            "placement": {"pool": selected.name, "capacity_budget": vars(budget)},
            "policy_checks": checks,
            "policy_snapshot_sha256": policy_hash,
            "capacity_snapshot_sha256": capacity_hash,
            "expires_at": now + self.ttl_seconds,
        }

    def recheck(self, project_id: str, workload: dict, attestation: dict) -> None:
        """Require fresh identical policy/capacity; never silently refresh an old plan."""
        if not isinstance(attestation, dict) or set(attestation) != {
            "placement",
            "policy_checks",
            "policy_snapshot_sha256",
            "capacity_snapshot_sha256",
            "expires_at",
        }:
            raise PlanningError("evidence", "Stored planning attestation malformed")
        try:
            now = self.clock()
            deadline = attestation["expires_at"]
            if isinstance(deadline, bool) or not math.isfinite(deadline):
                raise PlanningError("evidence", "Planning expiry invalid")
        except (TypeError, ValueError) as exc:
            raise PlanningError("evidence", "Planning expiry invalid") from exc
        if now >= deadline:
            raise PlanningError("evidence", "Planning attestation expired")
        renewed = self.attest(project_id, workload)
        try:
            checked_at = self.clock()
        except Exception as exc:
            raise PlanningError("unavailable", "Trusted planning clock unavailable") from exc
        if (
            isinstance(checked_at, bool)
            or not isinstance(checked_at, (int, float))
            or not math.isfinite(checked_at)
        ):
            raise PlanningError("evidence", "Planning clock invalid")
        if checked_at >= deadline:
            raise PlanningError("evidence", "Planning attestation expired")
        if any(
            renewed[field] != attestation[field]
            for field in (
                "placement",
                "policy_checks",
                "policy_snapshot_sha256",
                "capacity_snapshot_sha256",
            )
        ):
            raise PlanningError("evidence", "Planning evidence changed before submission")
