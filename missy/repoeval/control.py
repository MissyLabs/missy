"""Fail-closed, single-process control demo. Not a production execution service."""

from __future__ import annotations

import copy
import hashlib
import importlib.resources
import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import comparability_key, definition_hash, sha256_json
from .contracts import compare_runs as compare_manifest_runs
from .report import build_comparison_report

SHA = re.compile(r"^[0-9a-f]{40,64}$")
IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
KEY = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
STATES = {
    "planned": {"submitted", "failed"},
    "submitted": {"running", "failed", "cancelled"},
    "running": {"collecting", "failed", "cancelled"},
    "collecting": {"verified", "failed", "cancelled"},
    "verified": set(),
    "failed": set(),
    "cancelled": set(),
}


class FoundryError(ValueError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class Principal:
    subject: str
    project_id: str
    permissions: frozenset[str]


class Store(Protocol):
    def put_once(self, key: str, value: dict[str, Any]) -> dict[str, Any]: ...
    def get(self, key: str) -> dict[str, Any] | None: ...
    def update(self, key: str, value: dict[str, Any]) -> None: ...


class Dispatcher(Protocol):
    def submit(self, plan: dict[str, Any], run_id: str) -> str: ...
    def cancel(self, job_id: str) -> None: ...


class MemoryStore:
    """Test/development store only. Not suitable for multiple processes."""

    def __init__(self) -> None:
        self._values: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        # Shared across service instances using this store. Never a distributed lock.
        self.submission_lock = threading.RLock()

    def put_once(self, key: str, value: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._values.setdefault(key, copy.deepcopy(value))
            return copy.deepcopy(self._values[key])

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return copy.deepcopy(self._values.get(key))

    def update(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            if key not in self._values:
                raise FoundryError("missing", "Unknown resource")
            self._values[key] = copy.deepcopy(value)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


class FoundryService:
    """Reject unapproved execution; never derive authority from repository text."""

    def __init__(
        self,
        *,
        repositories: dict[str, set[str]],
        providers: set[str],
        limits: dict[str, int],
        store: Store | None = None,
        dispatcher: Dispatcher | None = None,
        verified_snapshots: dict[str, dict[str, str]] | None = None,
        approved_images: set[str] | None = None,
        approved_models: dict[str, set[str]] | None = None,
        approved_prompts: set[str] | None = None,
        approved_fixtures: dict[str, str] | None = None,
        approved_validators: dict[str, set[tuple[str, str, str]]] | None = None,
        allow_demo_dispatch: bool = False,
    ) -> None:
        self.repositories = {k: frozenset(v) for k, v in repositories.items()}
        self.providers = frozenset(providers)
        self.limits = limits.copy()
        self.store = store or MemoryStore()
        self.dispatcher = dispatcher
        # These must be injected by the trusted operator, not obtained from a
        # workload, snapshot request, API caller, or untrusted repository.
        self.verified_snapshots = copy.deepcopy(verified_snapshots or {})
        self.approved_images = frozenset(approved_images or ())
        self.approved_models = {k: frozenset(v) for k, v in (approved_models or {}).items()}
        self.approved_prompts = frozenset(approved_prompts or ())
        self.approved_fixtures = dict(approved_fixtures or {})
        self.approved_validators = {k: frozenset(v) for k, v in (approved_validators or {}).items()}
        self.allow_demo_dispatch = allow_demo_dispatch

    @staticmethod
    def _require(principal: Principal, permission: str) -> None:
        if (
            not principal.subject
            or not principal.project_id
            or permission not in principal.permissions
        ):
            raise FoundryError("authorization", "Project permission denied")

    def list_repositories(self, principal: Principal) -> list[str]:
        self._require(principal, "read")
        return sorted(self.repositories.get(principal.project_id, ()))

    def snapshot_start(
        self, principal: Principal, repository_id: str, commit_sha: str, idempotency_key: str
    ) -> dict[str, Any]:
        """Create an immutable, idempotent snapshot request; no checkout occurs here."""
        self._require(principal, "snapshot:start")
        if repository_id not in self.repositories.get(
            principal.project_id, ()
        ) or not SHA.fullmatch(str(commit_sha)):
            raise FoundryError(
                "invalid_input", "Registered repository and exact commit SHA required"
            )
        if not KEY.fullmatch(str(idempotency_key)):
            raise FoundryError("invalid_input", "Invalid idempotency key")
        identity = digest([principal.project_id, repository_id, commit_sha])
        snapshot_id = "snapshot-" + identity[:24]
        record = {
            "id": snapshot_id,
            "project_id": principal.project_id,
            "repository_id": repository_id,
            "commit_sha": commit_sha,
            "idempotency_key_sha256": digest(idempotency_key),
            "state": "requested",
        }
        idem_key = "snapshot-idempotency-" + digest([principal.project_id, idempotency_key])
        old = self.store.put_once(idem_key, record)
        if old.get("repository_id") != repository_id or old.get("commit_sha") != commit_sha:
            raise FoundryError("conflict", "Idempotency key already used for another snapshot")
        self.store.put_once(snapshot_id, old)
        return old

    def snapshot_status(self, principal: Principal, snapshot_id: str) -> dict[str, Any]:
        self._require(principal, "read")
        return self._owned(principal, snapshot_id)

    def artifacts(self, principal: Principal, resource_id: str) -> list[dict[str, Any]]:
        self._require(principal, "read")
        record = self._owned(principal, resource_id)
        artifacts = record.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise FoundryError("invalid_state", "Artifact manifest is malformed")
        return artifacts

    def benchmark_plan(self, principal: Principal, workload: dict[str, Any]) -> dict[str, Any]:
        self._require(principal, "read")
        return self._validated_plan(principal, workload)

    def _validated_plan(
        self, principal: Principal, workload: dict[str, Any], *, persist: bool = True
    ) -> dict[str, Any]:
        """Validate against the packaged contract and current operator policy."""
        workload = copy.deepcopy(workload)
        try:
            from jsonschema import Draft202012Validator
            from jsonschema.exceptions import SchemaError
        except ImportError as exc:
            raise FoundryError("unavailable", "Workload schema validation is unavailable") from exc
        try:
            schema = json.loads(
                importlib.resources.files("missy.repoeval")
                .joinpath("schemas/workload.schema.json")
                .read_text(encoding="utf-8")
            )
            Draft202012Validator.check_schema(schema)
        except (OSError, ValueError, SchemaError) as exc:
            raise FoundryError("unavailable", "Packaged workload schema is unavailable") from exc
        error = next(Draft202012Validator(schema).iter_errors(workload), None)
        if error is not None:
            raise FoundryError("invalid_input", "Workload does not satisfy the packaged schema")
        if workload.get("schema_version") != "1.0":
            raise FoundryError("invalid_input", "Unsupported workload version")
        repo = workload.get("repository", {})
        if not isinstance(repo, dict):
            raise FoundryError("invalid_input", "Repository identity is malformed")
        if repo.get("repository_id") not in self.repositories.get(principal.project_id, ()):
            raise FoundryError("authorization", "Repository is not registered for project")
        if not SHA.fullmatch(str(repo.get("commit_sha", ""))):
            raise FoundryError("invalid_input", "An immutable commit SHA is required")
        snapshot = self.verified_snapshots.get(repo.get("snapshot_id"))
        if not isinstance(snapshot, dict) or any(
            snapshot.get(k) != v
            for k, v in (
                ("id", repo.get("snapshot_id")),
                ("project_id", principal.project_id),
                ("repository_id", repo.get("repository_id")),
                ("commit_sha", repo.get("commit_sha")),
                ("state", "verified"),
            )
        ):
            raise FoundryError(
                "authorization",
                "Verified snapshot is not registered for this exact project and commit",
            )
        sandbox = workload.get("sandbox", {})
        if (
            not isinstance(sandbox, dict)
            or not IMAGE.fullmatch(str(sandbox.get("image_digest", "")))
            or sandbox["image_digest"] not in self.approved_images
        ):
            raise FoundryError("authorization", "Approved pinned image digest is required")
        for field in ("cpu_mhz", "memory_mb", "disk_mb"):
            amount = sandbox.get(field)
            if type(amount) is not int or amount <= 0 or amount > self.limits[field]:
                raise FoundryError("quota", "Resource envelope outside project ceiling")
        if sandbox.get("network_policy") not in self.limits.get("network_policies", ("none",)):
            raise FoundryError("policy", "Network policy not approved")
        execution = workload.get("execution", {})
        for field in ("warmups", "repetitions", "parallelism", "timeout_seconds", "max_attempts"):
            amount = execution.get(field)
            minimum = 0 if field == "warmups" else 1
            # Unspecified retry/warmup ceilings grant no extra execution.
            ceiling = self.limits.get(field, minimum)
            if type(amount) is not int or amount < minimum or amount > ceiling:
                raise FoundryError("quota", "Execution envelope outside project ceiling")
        targets = workload.get("providers", [])
        if not 1 <= len(targets) <= self.limits.get("provider_count", 1):
            raise FoundryError("invalid_input", "Provider targets missing or excessive")
        if any(
            not isinstance(t, dict)
            or t.get("registry_key") not in self.providers
            or t.get("model") not in self.approved_models.get(t.get("registry_key"), ())
            for t in targets
        ):
            raise FoundryError("authorization", "Provider/model target is not registered")
        settings = [t.get("settings") for t in targets]
        if any(not isinstance(s, dict) or s != settings[0] for s in settings):
            raise FoundryError("policy", "Provider-independent settings must agree across targets")
        task = workload.get("task", {})
        fixtures = task.get("fixture_digests") if isinstance(task, dict) else None
        if (
            not isinstance(task, dict)
            or task.get("prompt_sha256") not in self.approved_prompts
            or not isinstance(fixtures, dict)
            or not fixtures
            or any(
                not isinstance(name, str)
                or self.approved_fixtures.get(name) != value
                or not isinstance(value, str)
                or not re.fullmatch(r"[0-9a-f]{64}", value)
                for name, value in fixtures.items()
            )
        ):
            raise FoundryError("authorization", "Reviewed prompt and fixture digests are required")
        validation = workload.get("validation", {})
        if (
            not isinstance(validation, dict)
            or not isinstance(validation.get("validators"), list)
            or not validation["validators"]
        ):
            raise FoundryError("authorization", "Reviewed validators are required")
        validators = validation["validators"]
        try:
            reviewed = self.approved_validators.get(validation.get("evaluator_version"), ())
            if any(
                not isinstance(v, dict)
                or v.get("required") is not True
                or (v.get("id"), v.get("version"), sha256_json(v.get("parameters", {})))
                not in reviewed
                for v in validators
            ):
                raise FoundryError(
                    "authorization",
                    "Validator identity, configuration and evaluator must be reviewed",
                )
        except (TypeError, ValueError) as exc:
            raise FoundryError("invalid_input", "Validator configuration is malformed") from exc
        if not workload.get("artifacts", {}).get("required_kinds"):
            raise FoundryError("invalid_input", "Required artifact kinds must be declared")
        definition = definition_hash(workload)
        # Canonical definition excludes providers, but plan identity must retain
        # exact provider selection so a plan ID cannot alias a different target.
        plan_id = "plan-" + digest([principal.project_id, sha256_json(workload)])[:24]
        comparison_manifest = {
            "comparability_rules_version": "1.0",
            "repository": repo,
            "task": task,
            "workload_id": workload.get("id"),
            "workload_version": workload.get("version"),
            "definition_sha256": definition,
            "sandbox": {**sandbox, "timeout_seconds": execution.get("timeout_seconds")},
            "evaluator_version": validation.get("evaluator_version"),
            "validators": [
                {
                    "id": v["id"],
                    "version": v["version"],
                    "required": True,
                    "parameters_sha256": sha256_json(v.get("parameters", {})),
                }
                for v in validators
            ],
            "provider_independent_settings_sha256": sha256_json(settings[0]),
            "tool_schemas_sha256": sha256_json(task.get("tool_schema_uris", [])),
        }
        comparison = comparability_key(comparison_manifest)
        plan = {
            "id": plan_id,
            "project_id": principal.project_id,
            "workload": workload,
            "definition_sha256": definition,
            "comparability_key": comparison,
            "state": "planned",
        }
        if not persist:
            return plan
        old = self.store.put_once(plan_id, plan)
        if old != plan:
            raise FoundryError("conflict", "Plan ID collision")
        return old

    def benchmark_start(
        self, principal: Principal, plan_id: str, idempotency_key: str
    ) -> dict[str, Any]:
        self._require(principal, "execute")
        if not isinstance(idempotency_key, str) or not KEY.fullmatch(idempotency_key):
            raise FoundryError("invalid_input", "Invalid idempotency key")
        plan = self._owned(principal, plan_id)
        if (
            plan.get("id") != plan_id
            or plan.get("definition_sha256") != definition_hash(plan.get("workload", {}))
            or plan_id
            != "plan-" + digest([principal.project_id, sha256_json(plan["workload"])])[:24]
        ):
            raise FoundryError("evidence", "Stored plan identity does not match immutable workload")
        # Hold the store-wide lock across check, reservation, submission and
        # terminal update. A second service sharing the store cannot submit twice.
        with self.store.submission_lock:
            return self._benchmark_start_locked(principal, plan_id, idempotency_key, plan)

    def _benchmark_start_locked(
        self, principal: Principal, plan_id: str, idempotency_key: str, plan: dict[str, Any]
    ) -> dict[str, Any]:
        idem = "idempotency-" + digest([principal.project_id, idempotency_key])
        previous = self.store.get(idem)
        if previous:
            if previous["plan_id"] != plan_id:
                raise FoundryError("conflict", "Idempotency key already used for another plan")
            return self._owned(principal, previous["run_id"])
        if (
            not self.allow_demo_dispatch
            or self.dispatcher is None
            or not isinstance(self.store, MemoryStore)
        ):
            raise FoundryError(
                "unavailable",
                "Live dispatch disabled; only explicitly opted-in single-process demos are supported",
            )
        # A plan is a proposal, not a permanent grant. Re-check every current
        # registry/approval/limit immediately before reserving and dispatching.
        if self._validated_plan(principal, plan["workload"], persist=False) != plan:
            raise FoundryError("evidence", "Stored plan conflicts with current reviewed workload")
        run_id = "run-" + digest([principal.project_id, plan_id, idempotency_key])[:24]
        # A real production dispatcher must implement a transactionally reconciled
        # outbox. A submit failure leaves a failed record rather than a fake job.
        record = {
            "id": run_id,
            "project_id": principal.project_id,
            "plan_id": plan_id,
            "state": "planned",
            "job_id": None,
            "required_artifacts": list(plan["workload"]["artifacts"]["required_kinds"]),
            "idempotency_key_sha256": digest(idempotency_key),
        }
        self.store.put_once(run_id, record)
        reserved = self.store.put_once(idem, {"plan_id": plan_id, "run_id": run_id})
        if reserved != {"plan_id": plan_id, "run_id": run_id}:
            raise FoundryError("conflict", "Idempotency reservation conflict")
        try:
            record["job_id"] = self.dispatcher.submit(plan, run_id)
            if not record["job_id"]:
                raise RuntimeError("No job identity returned")
            record["state"] = "submitted"
        except Exception:
            record["state"] = "failed"
            record["failure_category"] = "dispatch"
        self.store.update(run_id, record)
        return copy.deepcopy(record)

    def _owned(self, principal: Principal, key: str) -> dict[str, Any]:
        record = self.store.get(key)
        if record is None or record.get("project_id") != principal.project_id:
            raise FoundryError("missing", "Resource not found in project")
        return record

    def run_status(self, principal: Principal, run_id: str) -> dict[str, Any]:
        self._require(principal, "read")
        return self._owned(principal, run_id)

    def compare_runs(self, principal: Principal, run_ids: list[str]) -> dict[str, Any]:
        """Compare project-owned run records strictly; missing manifests fail closed."""
        self._require(principal, "read")
        if (
            not isinstance(run_ids, list)
            or not 2 <= len(run_ids) <= 32
            or len(set(run_ids)) != len(run_ids)
        ):
            raise FoundryError("invalid_input", "Two to 32 unique run IDs are required")
        records = [self._owned(principal, run_id) for run_id in run_ids]
        manifests = []
        for record in records:
            manifests.append(self._verified_manifest(principal, record))
        pairwise = []
        for i, left in enumerate(manifests):
            for right in manifests[i + 1 :]:
                result = compare_manifest_runs(left, right)
                pairwise.append(
                    {
                        "left_run_id": left.get("run_id"),
                        "right_run_id": right.get("run_id"),
                        "comparable": result.comparable,
                        "key": result.key,
                        "reasons": list(result.reasons),
                    }
                )
        return {"comparable": all(row["comparable"] for row in pairwise), "pairs": pairwise}

    def report_draft(self, principal: Principal, run_ids: list[str]) -> dict[str, Any]:
        """Build a report from authorized records, preserving incomparable partitions."""
        self._require(principal, "read")
        if (
            not isinstance(run_ids, list)
            or not 1 <= len(run_ids) <= 32
            or len(set(run_ids)) != len(run_ids)
        ):
            raise FoundryError("invalid_input", "One to 32 unique run IDs are required")
        records = [self._owned(principal, run_id) for run_id in run_ids]
        manifests = [self._verified_manifest(principal, record) for record in records]
        return build_comparison_report(manifests)

    def _verified_manifest(self, principal: Principal, record: dict[str, Any]) -> dict[str, Any]:
        manifest = record.get("manifest")
        if record.get("state") != "verified" or not isinstance(manifest, dict):
            raise FoundryError(
                "unavailable", "Only verified runs with persisted manifests can be compared"
            )
        plan = self._owned(principal, record["plan_id"])
        workload = plan["workload"]
        if (
            manifest.get("run_id") != record["id"]
            or manifest.get("project_id") != principal.project_id
            or plan.get("definition_sha256") != definition_hash(workload)
            or plan.get("id")
            != "plan-" + digest([principal.project_id, sha256_json(workload)])[:24]
            or manifest.get("workload_id") != workload.get("id")
            or manifest.get("workload_version") != workload.get("version")
            or manifest.get("definition_sha256") != plan["definition_sha256"]
            or manifest.get("comparability_rules_version") != "1.0"
            or manifest.get("comparability_key") != plan["comparability_key"]
            or comparability_key(manifest) != plan["comparability_key"]
        ):
            raise FoundryError("evidence", "Persisted run manifest conflicts with reviewed plan")
        return manifest

    def reconcile(
        self,
        principal: Principal,
        run_id: str,
        state: str,
        *,
        scheduler_succeeded: bool = False,
        validators_passed: bool = False,
        artifacts_verified: set[str] | None = None,
    ) -> dict[str, Any]:
        self._require(principal, "reconcile")
        record = self._owned(principal, run_id)
        if state not in STATES.get(record["state"], set()):
            raise FoundryError("conflict", "Invalid state transition")
        if state == "verified":
            # The booleans are legacy API parameters, not attestation. No
            # scheduler/validator/artifact verifier is wired in this slice.
            raise FoundryError(
                "unavailable",
                "Independent scheduler, validator and artifact evidence integration is required",
            )
        record["state"] = state
        self.store.update(run_id, record)
        return record

    def run_cancel(self, principal: Principal, run_id: str) -> dict[str, Any]:
        self._require(principal, "execute")
        record = self._owned(principal, run_id)
        if record["state"] in ("cancelled", "failed", "verified"):
            return record
        if not self.dispatcher or not record.get("job_id"):
            raise FoundryError(
                "unavailable", "Scheduler identity unavailable; cancellation needs reconciliation"
            )
        self.dispatcher.cancel(record["job_id"])
        record["state"] = "cancelled"
        self.store.update(run_id, record)
        return record
