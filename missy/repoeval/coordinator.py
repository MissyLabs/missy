"""Opt-in durable RepoEval coordinator. No import-time scheduler activity."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from .artifacts import ArtifactError, safe_to_expose, validate_manifest, verify_object_bytes
from .contracts import comparability_key, definition_hash
from .control import FoundryError, FoundryService, Principal, digest
from .dispatch import DispatchConflict, DispatchError, DurableDispatcher


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class DurableFoundryService:
    """Operator-injected exact-job coordinator, not an implicit deployment."""

    def __init__(
        self,
        foundation: FoundryService,
        dispatcher: DurableDispatcher,
        job_factory: Callable[[dict, str, str, int, int], Mapping[str, Any]],
        *,
        verifier: Callable[[str, str, dict, list[dict], dict], bool] | None = None,
        enabled: bool = False,
    ) -> None:
        if not isinstance(foundation, FoundryService) or not isinstance(
            dispatcher, DurableDispatcher
        ):
            raise TypeError("FoundryService and DurableDispatcher required")
        if not callable(job_factory) or (verifier is not None and not callable(verifier)):
            raise TypeError("Factory/verifier must be callable")
        self.foundation, self.dispatcher, self.job_factory = foundation, dispatcher, job_factory
        self.verifier, self.enabled = verifier, enabled
        with dispatcher.store.transaction() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS coordinator_plans (
                project_id TEXT NOT NULL, plan_id TEXT NOT NULL, plan_json TEXT NOT NULL,
                PRIMARY KEY(project_id,plan_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS coordinator_runs (
                project_id TEXT NOT NULL, run_id TEXT NOT NULL, plan_id TEXT NOT NULL,
                key_hash TEXT NOT NULL, children_json TEXT NOT NULL,
                manifest_json TEXT, artifacts_json TEXT, verified INTEGER NOT NULL DEFAULT 0,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(project_id,run_id), UNIQUE(project_id,key_hash))""")

    @classmethod
    def initialize(cls, path, foundation, dispatcher, job_factory, *, verifier=None, enabled=False):
        if dispatcher.store.path.resolve() != Path(path).resolve():
            raise ValueError("Coordinator must share the outbox database")
        return cls(foundation, dispatcher, job_factory, verifier=verifier, enabled=enabled)

    def _enabled(self) -> None:
        if not self.enabled or not self.dispatcher.enabled:
            raise FoundryError("unavailable", "Durable dispatch disabled")

    def list_repositories(self, principal: Principal) -> list[str]:
        return self.foundation.list_repositories(principal)

    def snapshot_start(
        self, principal: Principal, repository_id: str, commit_sha: str, idempotency_key: str
    ) -> dict:
        raise FoundryError("unavailable", "Durable snapshot capture unavailable")

    def snapshot_status(self, principal: Principal, snapshot_id: str) -> dict:
        raise FoundryError("unavailable", "Durable snapshot capture unavailable")

    def benchmark_plan(self, principal: Principal, workload: dict) -> dict:
        self._enabled()
        self.foundation._require(principal, "read")
        plan = self.foundation._validated_plan(principal, workload, persist=False)
        with self.dispatcher.store.transaction() as db:
            row = db.execute(
                "SELECT plan_json FROM coordinator_plans WHERE project_id=? AND plan_id=?",
                (principal.project_id, plan["id"]),
            ).fetchone()
            if row is not None and row["plan_json"] != _json(plan):
                raise FoundryError("conflict", "Plan identity collision")
            if row is None:
                db.execute(
                    "INSERT INTO coordinator_plans VALUES(?,?,?)",
                    (principal.project_id, plan["id"], _json(plan)),
                )
        return plan

    def _plan(self, principal: Principal, plan_id: str) -> dict:
        with self.dispatcher.store.transaction() as db:
            row = db.execute(
                "SELECT plan_json FROM coordinator_plans WHERE project_id=? AND plan_id=?",
                (principal.project_id, plan_id),
            ).fetchone()
        if row is None:
            raise FoundryError("missing", "Plan not found in project")
        plan = json.loads(row["plan_json"])
        workload = plan.get("workload")
        if (
            plan.get("id") != plan_id
            or plan.get("project_id") != principal.project_id
            or not isinstance(workload, dict)
            or plan.get("definition_sha256") != definition_hash(workload)
            or plan_id
            != self.foundation._plan_identity(
                principal.project_id, workload, plan.get("attestation")
            )
            or any(
                plan.get(k) != plan.get("attestation", {}).get(k)
                for k in ("placement", "policy_checks", "expires_at")
            )
        ):
            raise FoundryError("evidence", "Stored plan conflicts with reviewed identity")
        return plan

    def _recheck(self, principal: Principal, plan: dict) -> None:
        self.foundation._require(principal, "execute")
        checked = self.foundation._validated_plan(
            principal, plan["workload"], persist=False, prior_attestation=plan["attestation"]
        )
        if checked != plan:
            raise FoundryError("evidence", "Stored plan no longer agrees with approval")

    def _run(self, principal: Principal, run_id: str) -> dict:
        with self.dispatcher.store.transaction() as db:
            row = db.execute(
                "SELECT * FROM coordinator_runs WHERE project_id=? AND run_id=?",
                (principal.project_id, run_id),
            ).fetchone()
        if row is None:
            raise FoundryError("missing", "Run not found in project")
        return dict(row)

    @staticmethod
    def _children(row: dict) -> list[dict]:
        return json.loads(row["children_json"])

    def _reserve_children(self, principal: Principal, row: dict, plan: dict) -> None:
        for child in self._children(row):
            if self._run(principal, row["run_id"])["cancel_requested"]:
                return
            try:
                job = self.job_factory(
                    copy.deepcopy(plan),
                    row["run_id"],
                    child["run_id"],
                    child["provider_index"],
                    child["repetition"],
                )
                if not isinstance(job, Mapping):
                    raise ValueError("Job factory returned no job")
                meta = job.get("Meta")
                if not isinstance(meta, Mapping) or any(
                    meta.get(k) != v
                    for k, v in (
                        ("foundry_parent_run_id", row["run_id"]),
                        ("foundry_provider_index", str(child["provider_index"])),
                        ("foundry_repetition", str(child["repetition"])),
                    )
                ):
                    raise FoundryError(
                        "evidence", "Job identity does not bind provider and repetition"
                    )
                # The outbox owns its own transaction. Never nest it inside
                # a coordinator transaction: SQLite would deadlock on itself.
                if self._run(principal, row["run_id"])["cancel_requested"]:
                    return
                self.dispatcher.reserve(principal.project_id, child["run_id"], child["key"], job)
                if self._run(principal, row["run_id"])["cancel_requested"]:
                    self.dispatcher.cancel(principal.project_id, child["run_id"])
            except DispatchConflict as exc:
                raise FoundryError("conflict", "Child reservation conflicts") from exc
            except (DispatchError, ValueError, TypeError) as exc:
                raise FoundryError("unavailable", "Child reservation unavailable") from exc

    def benchmark_start(self, principal: Principal, plan_id: str, idempotency_key: str) -> dict:
        self._enabled()
        self.foundation._require(principal, "execute")
        if not isinstance(idempotency_key, str) or not re.fullmatch(
            r"[A-Za-z0-9._:-]{8,128}", idempotency_key
        ):
            raise FoundryError("invalid_input", "Invalid idempotency key")
        plan = self._plan(principal, plan_id)
        key_hash = hashlib.sha256(idempotency_key.encode()).hexdigest()
        run_id = "run-" + digest([principal.project_id, plan_id, idempotency_key])[:24]
        with self.dispatcher.store.transaction() as db:
            old = db.execute(
                "SELECT * FROM coordinator_runs WHERE project_id=? AND key_hash=?",
                (principal.project_id, key_hash),
            ).fetchone()
            if old is not None:
                if old["plan_id"] != plan_id or old["run_id"] != run_id:
                    raise FoundryError("conflict", "Idempotency key used for another plan")
                row = dict(old)
            else:
                self._recheck(principal, plan)
                children = [
                    {
                        "run_id": "run-" + digest([run_id, i, repetition])[:24],
                        "provider_index": i,
                        "repetition": repetition,
                        "key": "child-" + digest([run_id, i, repetition])[:48],
                    }
                    for i in range(len(plan["workload"]["providers"]))
                    for repetition in range(plan["workload"]["execution"]["repetitions"])
                ]
                if len(children) > 256:
                    raise FoundryError("quota", "Too many child jobs")
                db.execute(
                    "INSERT INTO coordinator_runs(project_id,run_id,plan_id,key_hash,children_json) VALUES(?,?,?,?,?)",
                    (principal.project_id, run_id, plan_id, key_hash, _json(children)),
                )
                row = dict(
                    db.execute(
                        "SELECT * FROM coordinator_runs WHERE project_id=? AND run_id=?",
                        (principal.project_id, run_id),
                    ).fetchone()
                )
        if row["cancel_requested"]:
            return self.run_status(principal, run_id)
        if any(
            self.dispatcher.store.get(principal.project_id, c["run_id"]) is None
            for c in self._children(row)
        ):
            self._recheck(principal, plan)
            self._reserve_children(principal, row, plan)
        return self.run_status(principal, run_id)

    def run_status(self, principal: Principal, run_id: str) -> dict:
        self.foundation._require(principal, "read")
        row = self._run(principal, run_id)
        children = []
        for child in self._children(row):
            stored = self.dispatcher.store.get(principal.project_id, child["run_id"])
            children.append(
                {
                    "run_id": child["run_id"],
                    "provider_index": child["provider_index"],
                    "repetition": child["repetition"],
                    "state": stored["state"] if stored else "reservation_incomplete",
                    "job_id": stored["job_id"] if stored else None,
                }
            )
        states = {child["state"] for child in children}
        if row["verified"]:
            state = "verified"
        elif row["cancel_requested"] and states <= {"cancelled", "reservation_incomplete"}:
            state = "cancelled"
        elif "conflict" in states:
            state = "conflict"
        elif row["cancel_requested"]:
            state = "cancel_pending"
        elif states and states <= {"cancelled"}:
            state = "cancelled"
        elif states & {"stop_uncertain", "cancel_pending"}:
            state = "cancel_pending"
        elif states & {"uncertain", "dispatching"}:
            state = "uncertain"
        elif "failed" in states:
            state = "failed"
        elif states and states <= {"collecting"}:
            state = "collecting"
        elif "running" in states:
            state = "running"
        elif "submitted" in states:
            state = "submitted" if states <= {"submitted", "collecting"} else "partial"
        elif "reservation_incomplete" in states and len(states) > 1:
            state = "partial"
        else:
            state = "reserved"
        return {
            "id": run_id,
            "project_id": principal.project_id,
            "plan_id": row["plan_id"],
            "state": state,
            "job_id": None,
            "children": children,
            "required_artifacts": list(
                self._plan(principal, row["plan_id"])["workload"]["artifacts"]["required_kinds"]
            ),
        }

    def dispatch_pending(self, principal: Principal, run_id: str) -> dict:
        """Explicit opt-in external mutation. Not exposed as an HTTP route."""
        self._enabled()
        self.foundation._require(principal, "execute")
        row = self._run(principal, run_id)
        if row["verified"] or row["cancel_requested"]:
            return self.run_status(principal, run_id)
        plan = self._plan(principal, row["plan_id"])
        self._recheck(principal, plan)
        self._reserve_children(principal, row, plan)
        for child in self._children(row):
            if self._run(principal, run_id)["cancel_requested"]:
                return self.run_status(principal, run_id)
            self._recheck(principal, plan)
            try:
                self.dispatcher.dispatch(principal.project_id, child["run_id"])
            except (DispatchError, DispatchConflict) as exc:
                raise FoundryError("evidence", "Dispatch needs reconciliation") from exc
        return self.run_status(principal, run_id)

    def reconcile_run(self, principal: Principal, run_id: str) -> dict:
        self._enabled()
        self.foundation._require(principal, "read")
        for child in self._children(self._run(principal, run_id)):
            try:
                self.dispatcher.reconcile(principal.project_id, child["run_id"])
            except DispatchConflict:
                pass
            except DispatchError as exc:
                raise FoundryError("unavailable", "Child reservation incomplete") from exc
        return self.run_status(principal, run_id)

    def run_cancel(self, principal: Principal, run_id: str) -> dict:
        self._enabled()
        self.foundation._require(principal, "execute")
        row = self._run(principal, run_id)
        if row["verified"]:
            return self.run_status(principal, run_id)
        with self.dispatcher.store.transaction() as db:
            updated = db.execute(
                "UPDATE coordinator_runs SET cancel_requested=1 WHERE project_id=? AND run_id=? AND verified=0",
                (principal.project_id, run_id),
            )
            cancelled = updated.rowcount == 1
        if not cancelled:
            return self.run_status(principal, run_id)
        for child in self._children(row):
            if self.dispatcher.store.get(principal.project_id, child["run_id"]) is None:
                continue  # parent cancellation tombstone blocks a late reservation
            with suppress(DispatchConflict):
                self.dispatcher.cancel(principal.project_id, child["run_id"])
        return self.run_status(principal, run_id)

    def finalize(
        self, principal: Principal, run_id: str, manifest: dict, artifacts: list[dict]
    ) -> dict:
        """Require independent scheduler, validator and cleared artifact evidence."""
        self._enabled()
        self.foundation._require(principal, "reconcile")
        row = self._run(principal, run_id)
        if row["verified"]:
            if row["manifest_json"] != _json(manifest) or row["artifacts_json"] != _json(artifacts):
                raise FoundryError("conflict", "Verified evidence cannot be replaced")
            return self.run_status(principal, run_id)
        if row["cancel_requested"]:
            raise FoundryError("conflict", "Cancelled run cannot be finalized")
        if (
            self.verifier is None
            or self.foundation.artifact_reader is None
            or self.foundation.artifact_clearance is None
        ):
            raise FoundryError(
                "unavailable", "Independent validator and artifact readers unavailable"
            )
        plan = self._plan(principal, row["plan_id"])
        children = self._children(row)
        if not children or any(
            self.dispatcher.reconcile(principal.project_id, c["run_id"])["state"] != "collecting"
            for c in children
        ):
            raise FoundryError("evidence", "Every exact scheduler job must complete")
        if (
            not isinstance(manifest, dict)
            or manifest.get("run_id") != run_id
            or manifest.get("project_id") != principal.project_id
            or manifest.get("workload_id") != plan["workload"].get("id")
            or manifest.get("workload_version") != plan["workload"].get("version")
            or manifest.get("definition_sha256") != plan["definition_sha256"]
            or manifest.get("comparability_rules_version") != "1.0"
            or manifest.get("comparability_key") != plan["comparability_key"]
            or comparability_key(manifest) != plan["comparability_key"]
        ):
            raise FoundryError("evidence", "Manifest differs from exact reviewed plan")
        if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= 100:
            raise FoundryError("evidence", "Artifact evidence missing")
        try:
            seen, kinds = set(), set()
            for item in artifacts:
                candidate = validate_manifest(item, run_id=run_id)
                if candidate["artifact_id"] in seen or not safe_to_expose(candidate):
                    raise ArtifactError("Duplicate or uncleared artifact")
                seen.add(candidate["artifact_id"])
                kinds.add(candidate["kind"])
                data = self.foundation.artifact_reader(
                    principal.project_id, run_id, candidate["artifact_id"]
                )
                verify_object_bytes(candidate, data)
                if (
                    self.foundation.artifact_clearance(
                        principal.project_id, run_id, candidate, data
                    )
                    is not True
                ):
                    raise ArtifactError("Independent clearance unavailable")
            if not set(plan["workload"]["artifacts"]["required_kinds"]) <= kinds:
                raise ArtifactError("Required artifact kinds unavailable")
            if (
                self.verifier(
                    principal.project_id,
                    run_id,
                    copy.deepcopy(plan),
                    copy.deepcopy(children),
                    copy.deepcopy(manifest),
                )
                is not True
            ):
                raise FoundryError("evidence", "Independent validator has not attested")
        except (ArtifactError, KeyError, TypeError, ValueError, OSError) as exc:
            raise FoundryError("evidence", "Validator or artifact evidence invalid") from exc
        with self.dispatcher.store.transaction() as db:
            current = db.execute(
                "SELECT verified,cancel_requested FROM coordinator_runs WHERE project_id=? AND run_id=?",
                (principal.project_id, run_id),
            ).fetchone()
            if current["verified"] or current["cancel_requested"]:
                raise FoundryError("conflict", "Run changed during finalization")
            for child in children:
                record = db.execute(
                    "SELECT state,cancel_requested FROM dispatch_outbox WHERE project_id=? AND run_id=?",
                    (principal.project_id, child["run_id"]),
                ).fetchone()
                if record is None or record["state"] != "collecting" or record["cancel_requested"]:
                    raise FoundryError("conflict", "Child changed during validation")
            db.execute(
                "UPDATE coordinator_runs SET manifest_json=?,artifacts_json=?,verified=1 WHERE project_id=? AND run_id=?",
                (_json(manifest), _json(artifacts), principal.project_id, run_id),
            )
        return self.run_status(principal, run_id)

    def artifacts(self, principal: Principal, run_id: str) -> dict:
        self.foundation._require(principal, "read")
        row = self._run(principal, run_id)
        if not row["verified"] or row["artifacts_json"] is None:
            raise FoundryError("unavailable", "Only verified artifacts may be exposed")
        results = []
        for item in json.loads(row["artifacts_json"]):
            try:
                manifest = validate_manifest(item, run_id=run_id)
                if (
                    not safe_to_expose(manifest)
                    or self.foundation.artifact_reader is None
                    or self.foundation.artifact_clearance is None
                ):
                    raise ArtifactError("Clearance unavailable")
                data = self.foundation.artifact_reader(
                    principal.project_id, run_id, manifest["artifact_id"]
                )
                verify_object_bytes(manifest, data)
                if (
                    self.foundation.artifact_clearance(principal.project_id, run_id, manifest, data)
                    is not True
                ):
                    raise ArtifactError("Clearance revoked")
            except (ArtifactError, KeyError, TypeError, ValueError, OSError) as exc:
                raise FoundryError(
                    "evidence", "Artifact digest or clearance cannot be verified"
                ) from exc
            results.append(
                {
                    k: manifest[k]
                    for k in ("artifact_id", "run_id", "kind", "sha256", "size_bytes", "media_type")
                }
            )
        return {"project_id": principal.project_id, "run_id": run_id, "artifacts": results}

    def compare_runs(self, principal: Principal, run_ids: list[str]) -> dict:
        raise FoundryError("unavailable", "Durable comparison is not implemented")

    def report_draft(self, principal: Principal, run_ids: list[str]) -> dict:
        raise FoundryError("unavailable", "Durable reporting is not implemented")
