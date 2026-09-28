"""Opt-in coordinator with finite fake scheduler, never a live adapter."""

from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from missy.repoeval.contracts import sha256_json
from missy.repoeval.control import FoundryError, FoundryService, Principal
from missy.repoeval.coordinator import DurableFoundryService
from missy.repoeval.dispatch import DurableDispatcher, JobObservation, SQLiteOutbox
from missy.repoeval.placement import PoolCapacity, ResourceEnvelope
from missy.repoeval.planning import CapacitySnapshot, PlanningAuthority, ProjectPolicySnapshot

HEX = "a" * 64
IMAGE = "registry.example/worker@sha256:" + HEX


class Scheduler:
    def __init__(self):
        self.jobs = {}
        self.submissions = 0
        self.stops = 0
        self.fail_after_submit = False
        self._lock = threading.Lock()

    def submit(self, namespace, job):
        with self._lock:
            self.submissions += 1
            self.jobs[(namespace, job["ID"])] = (copy.deepcopy(job), "running")
        if self.fail_after_submit:
            raise TimeoutError("ack lost after external effect")

    def lookup(self, namespace, job_id):
        observed = self.jobs.get((namespace, job_id))
        return JobObservation(*observed) if observed else None

    def stop(self, namespace, job_id):
        with self._lock:
            self.stops += 1
            job, _ = self.jobs[(namespace, job_id)]
            self.jobs[(namespace, job_id)] = (job, "stopped")


def make_system(tmp_path, *, repetitions=2, providers=2, parallelism=1):
    now = [1000.0]
    allow = [True]

    def policy(project):
        return ProjectPolicySnapshot(
            project,
            "policy",
            now[0],
            frozenset({"MissyLabs/missy"}),
            frozenset({IMAGE}),
            {"one": frozenset({"registered-model"}), "two": frozenset({"registered-model"})}
            if allow[0]
            else {},
            ResourceEnvelope(2000, 2048, 4096),
            1_000_000_000,
            frozenset({"none"}),
            "audit",
        )

    foundation = FoundryService(
        repositories={"project": {"MissyLabs/missy"}},
        providers={"one", "two"},
        limits={
            "cpu_mhz": 2000,
            "memory_mb": 2048,
            "disk_mb": 4096,
            "repetitions": repetitions,
            "parallelism": parallelism,
            "timeout_seconds": 300,
            "network_policies": ("none",),
            "provider_count": providers,
        },
        planning_authority=PlanningAuthority(
            lambda: CapacitySnapshot("cap", now[0], (PoolCapacity("staging", 4000, 8192, 16384),)),
            policy,
            clock=lambda: now[0],
        ),
        verified_snapshots={
            "snapshot-fixed": {
                "id": "snapshot-fixed",
                "state": "verified",
                "project_id": "project",
                "repository_id": "MissyLabs/missy",
                "commit_sha": "b" * 40,
            }
        },
        approved_images={IMAGE},
        approved_models={"one": {"registered-model"}, "two": {"registered-model"}},
        approved_prompts={HEX},
        approved_fixtures={"oracle": "c" * 64},
        approved_validators={"1": {("one", "1", sha256_json({}))}},
    )
    scheduler = Scheduler()
    outbox = SQLiteOutbox.initialize(tmp_path / "outbox.db")
    dispatcher = DurableDispatcher(outbox, scheduler, authorize=lambda *_: True, enabled=True)

    def job_factory(plan, parent, child, index, repetition):
        job_id = "foundry-" + child
        return {
            "ID": job_id,
            "Name": job_id,
            "Namespace": "staging",
            "Type": "batch",
            "Meta": {
                "foundry_run_id": child,
                "foundry_parent_run_id": parent,
                "foundry_provider_index": str(index),
                "foundry_repetition": str(repetition),
                "model": plan["workload"]["providers"][index]["model"],
            },
        }

    def service():
        return DurableFoundryService.initialize(
            outbox.path, foundation, dispatcher, job_factory, enabled=True
        )

    principal = Principal("operator", "project", frozenset({"read", "execute", "reconcile"}))
    spec = {
        "schema_version": "1.0",
        "id": "test-case",
        "version": "1",
        "repository": {
            "repository_id": "MissyLabs/missy",
            "commit_sha": "b" * 40,
            "snapshot_id": "snapshot-fixed",
        },
        "task": {
            "class": "tool-call",
            "prompt_sha256": HEX,
            "fixture_digests": {"oracle": "c" * 64},
        },
        "providers": [
            {"registry_key": k, "model": "registered-model", "settings": {}}
            for k in ("one", "two")[:providers]
        ],
        "sandbox": {
            "image_digest": IMAGE,
            "cpu_mhz": 300,
            "memory_mb": 256,
            "disk_mb": 512,
            "network_policy": "none",
        },
        "execution": {
            "warmups": 0,
            "repetitions": repetitions,
            "parallelism": parallelism,
            "timeout_seconds": 120,
            "max_attempts": 1,
        },
        "validation": {
            "evaluator_version": "1",
            "validators": [{"id": "one", "version": "1", "required": True}],
        },
        "artifacts": {
            "output_prefix": "runs",
            "retention_class": "report",
            "required_kinds": ["result"],
        },
    }
    return service, scheduler, now, allow, principal, spec


def test_reserve_restart_race_and_no_implicit_submit(tmp_path):
    construct, scheduler, _, _, principal, spec = make_system(tmp_path)
    plan = construct().benchmark_plan(principal, spec)
    barrier = threading.Barrier(4)

    def start():
        barrier.wait()
        return construct().benchmark_start(principal, plan["id"], "idempotency-123")

    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(lambda _: start(), range(4)))
    assert len({run["id"] for run in runs}) == 1
    assert all(run["state"] == "reserved" and len(run["children"]) == 4 for run in runs)
    assert len({child["job_id"] for child in runs[0]["children"]}) == 4
    assert scheduler.submissions == 0
    restarted = construct()
    assert restarted.run_status(principal, runs[0]["id"])["state"] == "reserved"
    restarted.dispatch_pending(principal, runs[0]["id"])
    restarted.dispatch_pending(principal, runs[0]["id"])
    assert scheduler.submissions == 1
    assert restarted.run_status(principal, runs[0]["id"])["state"] != "verified"
    with pytest.raises(FoundryError):
        restarted.run_status(Principal("outsider", "other", frozenset({"read"})), runs[0]["id"])


@pytest.mark.parametrize("parallelism", [1, 2])
def test_slots_advance_only_after_reconciled_completion(tmp_path, parallelism):
    construct, scheduler, _, _, principal, spec = make_system(tmp_path, parallelism=parallelism)
    service = construct()
    plan = service.benchmark_plan(principal, spec)
    run = service.benchmark_start(principal, plan["id"], f"parallelism-{parallelism}")
    run_id = run["id"]
    for completed in range(4):
        status = construct().dispatch_pending(principal, run_id)
        assert scheduler.submissions == min(4, completed + parallelism)
        assert (
            sum(
                c["state"] in {"running", "submitted", "dispatching", "uncertain"}
                for c in status["children"]
            )
            <= parallelism
        )
        child = status["children"][completed]
        assert child["state"] == "running"
        job, _ = scheduler.jobs[("staging", child["job_id"])]
        scheduler.jobs[("staging", child["job_id"])] = (job, "complete")
        # Scheduler completion alone is not a verified benchmark result.
        assert (
            construct().reconcile_run(principal, run_id)["children"][completed]["state"]
            == "collecting"
        )
        assert construct().run_status(principal, run_id)["state"] != "verified"
    final = construct().dispatch_pending(principal, run_id)
    assert scheduler.submissions == 4
    assert final["state"] == "collecting"


@pytest.mark.parametrize("parallelism", [1, 2])
def test_concurrent_dispatch_never_exceeds_slots(tmp_path, parallelism):
    construct, scheduler, _, _, principal, spec = make_system(tmp_path, parallelism=parallelism)
    service = construct()
    run = service.benchmark_start(
        principal, service.benchmark_plan(principal, spec)["id"], "parallel-race"
    )
    barrier = threading.Barrier(8)

    def dispatch(_):
        barrier.wait()
        return construct().dispatch_pending(principal, run["id"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(dispatch, range(8)))
    assert scheduler.submissions == parallelism
    assert construct().run_status(principal, run["id"])["state"] == "running"


def test_uncertain_child_holds_slot_until_reconciliation(tmp_path):
    construct, scheduler, _, _, principal, spec = make_system(tmp_path)
    service = construct()
    run = service.benchmark_start(
        principal, service.benchmark_plan(principal, spec)["id"], "uncertain-slots"
    )
    scheduler.fail_after_submit = True
    first = run["children"][0]
    original_lookup = scheduler.lookup
    scheduler.lookup = lambda *_: None
    construct().dispatch_pending(principal, run["id"])
    assert construct().run_status(principal, run["id"])["children"][0]["state"] == "uncertain"
    scheduler.fail_after_submit = False
    construct().dispatch_pending(principal, run["id"])
    assert scheduler.submissions == 1
    scheduler.lookup = original_lookup
    construct().reconcile_run(principal, run["id"])
    construct().dispatch_pending(principal, run["id"])
    assert scheduler.submissions == 1
    job, _ = scheduler.jobs[("staging", first["job_id"])]
    scheduler.jobs[("staging", first["job_id"])] = (job, "complete")
    construct().reconcile_run(principal, run["id"])
    construct().dispatch_pending(principal, run["id"])
    assert scheduler.submissions == 2


def test_failed_child_cancel_restart_is_terminal(tmp_path):
    construct, scheduler, _, _, principal, spec = make_system(tmp_path)
    service = construct()
    run = service.benchmark_start(
        principal, service.benchmark_plan(principal, spec)["id"], "failed-cancel"
    )
    service.dispatch_pending(principal, run["id"])
    child = run["children"][0]
    job, _ = scheduler.jobs[("staging", child["job_id"])]
    scheduler.jobs[("staging", child["job_id"])] = (job, "failed")
    assert service.reconcile_run(principal, run["id"])["state"] == "failed"
    assert service.run_cancel(principal, run["id"])["state"] == "failed"
    assert construct().run_status(principal, run["id"])["state"] == "failed"
    assert construct().reconcile_run(principal, run["id"])["state"] == "failed"
    assert scheduler.submissions == 1


def test_uncertain_submit_never_replays_across_restart(tmp_path):
    construct, scheduler, _, _, principal, spec = make_system(tmp_path, repetitions=1, providers=1)
    service = construct()
    plan = service.benchmark_plan(principal, spec)
    run = service.benchmark_start(principal, plan["id"], "uncertain-key")
    scheduler.fail_after_submit = True
    service.dispatch_pending(principal, run["id"])
    assert scheduler.submissions == 1
    assert construct().run_status(principal, run["id"])["state"] != "verified"
    construct().dispatch_pending(principal, run["id"])
    assert scheduler.submissions == 1
    construct().run_cancel(principal, run["id"])
    assert scheduler.stops == 1
    assert construct().run_status(principal, run["id"])["state"] == "cancelled"


def test_policy_revocation_blocks_dispatch_but_reserved_cancel_is_safe(tmp_path):
    construct, scheduler, _, allow, principal, spec = make_system(
        tmp_path, repetitions=1, providers=1
    )
    service = construct()
    plan = service.benchmark_plan(principal, spec)
    run = service.benchmark_start(principal, plan["id"], "revoked-key")
    allow[0] = False
    with pytest.raises(FoundryError):
        construct().dispatch_pending(principal, run["id"])
    assert scheduler.submissions == 0
    assert construct().run_cancel(principal, run["id"])["state"] == "cancelled"
    assert scheduler.stops == 0


def test_cancel_while_child_factory_is_running_keeps_tombstone(tmp_path):
    construct, scheduler, _, _, principal, spec = make_system(tmp_path, repetitions=1, providers=1)
    service = construct()
    plan = service.benchmark_plan(principal, spec)
    started, resume = threading.Event(), threading.Event()
    real_factory = service.job_factory

    def paused_factory(*args):
        started.set()
        assert resume.wait(timeout=5)
        return real_factory(*args)

    service.job_factory = paused_factory
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(service.benchmark_start, principal, plan["id"], "race-cancel-key")
        assert started.wait(timeout=5)
        from missy.repoeval.control import digest

        run_id = "run-" + digest([principal.project_id, plan["id"], "race-cancel-key"])[:24]
        assert construct().run_cancel(principal, run_id)["state"] == "cancelled"
        resume.set()
        result = future.result(timeout=5)
    assert result["state"] == "cancelled"
    assert construct().dispatch_pending(principal, run_id)["state"] == "cancelled"
    assert scheduler.submissions == 0


def test_scheduler_complete_never_fakes_verified(tmp_path):
    construct, scheduler, _, _, principal, spec = make_system(tmp_path, repetitions=1, providers=1)
    service = construct()
    plan = service.benchmark_plan(principal, spec)
    run = service.benchmark_start(principal, plan["id"], "no-fake-success")
    service.dispatch_pending(principal, run["id"])
    child = run["children"][0]
    job, _ = scheduler.jobs[("staging", child["job_id"])]
    scheduler.jobs[("staging", child["job_id"])] = (job, "complete")
    assert service.reconcile_run(principal, run["id"])["state"] == "collecting"
    with pytest.raises(FoundryError, match="Independent"):
        service.finalize(principal, run["id"], {}, [])
    assert construct().run_status(principal, run["id"])["state"] == "collecting"
