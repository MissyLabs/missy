import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

from missy.repoeval.contracts import definition_hash, sha256_json
from missy.repoeval.control import FoundryError, FoundryService, MemoryStore, Principal
from missy.repoeval.placement import PoolCapacity, ResourceEnvelope
from missy.repoeval.planning import CapacitySnapshot, PlanningAuthority, ProjectPolicySnapshot

HEX = "a" * 64


def workload():
    return {
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
        "providers": [{"registry_key": "one", "model": "registered-model", "settings": {}}],
        "sandbox": {
            "image_digest": "registry.example/worker@sha256:" + HEX,
            "cpu_mhz": 300,
            "memory_mb": 256,
            "disk_mb": 512,
            "network_policy": "none",
        },
        "execution": {
            "warmups": 0,
            "repetitions": 1,
            "parallelism": 1,
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


class FakeDispatcher:
    def __init__(self):
        self.calls = 0
        self.cancelled = []

    def submit(self, plan, run_id):
        self.calls += 1
        return "job-" + run_id

    def cancel(self, job_id):
        self.cancelled.append(job_id)


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.dispatcher = FakeDispatcher()
        self.store = MemoryStore()
        self.now = 1000.0
        self.capacity = CapacitySnapshot(
            "capacity-1",
            self.now,
            (
                PoolCapacity("staging", 3000, 4096, 8192),
                PoolCapacity("production", 5000, 8192, 16384),
            ),
        )
        self.project_policy = ProjectPolicySnapshot(
            "project",
            "policy-1",
            self.now,
            frozenset({"MissyLabs/missy"}),
            frozenset({"registry.example/worker@sha256:" + HEX}),
            {"one": frozenset({"registered-model"})},
            ResourceEnvelope(500, 512, 1024),
            10_000_000,
            frozenset({"none"}),
            "audit-sink-1",
        )
        authority = PlanningAuthority(
            lambda: self.capacity,
            lambda project: self.project_policy,
            clock=lambda: self.now,
        )
        self.service = FoundryService(
            repositories={"project": {"MissyLabs/missy"}},
            providers={"one"},
            limits={
                "cpu_mhz": 500,
                "memory_mb": 512,
                "disk_mb": 1024,
                "repetitions": 3,
                "warmups": 2,
                "max_attempts": 2,
                "parallelism": 2,
                "timeout_seconds": 300,
                "network_policies": ("none",),
                "provider_count": 2,
            },
            store=self.store,
            dispatcher=self.dispatcher,
            allow_demo_dispatch=True,
            planning_authority=authority,
            verified_snapshots={
                "snapshot-fixed": {
                    "id": "snapshot-fixed",
                    "state": "verified",
                    "project_id": "project",
                    "repository_id": "MissyLabs/missy",
                    "commit_sha": "b" * 40,
                }
            },
            approved_images={"registry.example/worker@sha256:" + HEX},
            approved_models={"one": {"registered-model"}},
            approved_prompts={HEX},
            approved_fixtures={"oracle": "c" * 64},
            approved_validators={"1": {("one", "1", sha256_json({}))}},
        )
        self.user = Principal("missy", "project", frozenset({"read", "execute", "reconcile"}))

    def test_plan_and_idempotent_start(self):
        plan = self.service.benchmark_plan(self.user, workload())
        self.assertEqual(plan, self.service.benchmark_plan(self.user, workload()))
        first = self.service.benchmark_start(self.user, plan["id"], "repeat-this-key")
        self.assertEqual(
            first, self.service.benchmark_start(self.user, plan["id"], "repeat-this-key")
        )
        self.assertEqual(self.dispatcher.calls, 1)
        self.service.reconcile(self.user, first["id"], "running")
        self.service.reconcile(self.user, first["id"], "collecting")
        with self.assertRaisesRegex(FoundryError, "Independent"):
            self.service.reconcile(
                self.user, first["id"], "verified", scheduler_succeeded=True, validators_passed=True
            )
        with self.assertRaisesRegex(FoundryError, "Independent"):
            self.service.reconcile(
                self.user,
                first["id"],
                "verified",
                scheduler_succeeded=True,
                validators_passed=True,
                artifacts_verified={"result"},
            )
        self.assertEqual(self.service.run_status(self.user, first["id"])["state"], "collecting")
        self.assertEqual(plan["definition_sha256"], definition_hash(workload()))

    def test_denials_and_isolation(self):
        bad = workload()
        bad["repository"]["commit_sha"] = "master"
        with self.assertRaises(FoundryError):
            self.service.benchmark_plan(self.user, bad)
        bad = workload()
        bad["sandbox"]["cpu_mhz"] = 600
        with self.assertRaises(FoundryError):
            self.service.benchmark_plan(self.user, bad)
        plan = self.service.benchmark_plan(self.user, workload())
        other = Principal("other", "other-project", frozenset({"read", "execute"}))
        with self.assertRaises(FoundryError):
            self.service.benchmark_start(other, plan["id"], "repeat-this-key")
        with self.assertRaises(FoundryError):
            self.service.benchmark_start(
                Principal("viewer", "project", frozenset({"read"})), plan["id"], "repeat-this-key"
            )

    def test_full_packaged_schema_is_required(self):
        changes = (
            lambda w: w["execution"].update(warmups=10**12),
            lambda w: w["execution"].update(max_attempts=10**12),
            lambda w: w["execution"].update(warmups=True),
            lambda w: w["sandbox"].update(unrecognized=1),
            lambda w: w["providers"][0].pop("settings"),
            lambda w: w["validation"]["validators"][0].update(extra="unreviewed"),
            lambda w: w["task"].update(**{"class": "unknown"}),
            lambda w: w["artifacts"].update(required_kinds=["result", "result"]),
        )
        for change in changes:
            with self.subTest(change=change):
                candidate = workload()
                change(candidate)
                with self.assertRaises(FoundryError) as refusal:
                    self.service.benchmark_plan(self.user, candidate)
                self.assertEqual(refusal.exception.category, "invalid_input")
        self.assertEqual(self.dispatcher.calls, 0)

    def test_missing_schema_dependency_denies_planning_and_new_dispatch(self):
        plan = self.service.benchmark_plan(self.user, workload())
        import builtins

        original_import = builtins.__import__

        def missing(name, *args, **kwargs):
            if name == "jsonschema" or name.startswith("jsonschema."):
                raise ImportError("missing optional dependency")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=missing):
            with self.assertRaisesRegex(FoundryError, "schema validation is unavailable"):
                self.service.benchmark_plan(self.user, workload())
            with self.assertRaisesRegex(FoundryError, "schema validation is unavailable"):
                self.service.benchmark_start(self.user, plan["id"], "schema-missing-key")
        self.assertEqual(self.dispatcher.calls, 0)

    def test_project_ceiling_for_each_multiplier(self):
        for field in ("warmups", "repetitions", "parallelism", "timeout_seconds", "max_attempts"):
            with self.subTest(field=field):
                candidate = workload()
                candidate["execution"][field] = self.service.limits[field] + 1
                with self.assertRaises(FoundryError) as refusal:
                    self.service.benchmark_plan(self.user, candidate)
                self.assertEqual(refusal.exception.category, "quota")
        for field in ("warmups", "max_attempts"):
            with self.subTest(missing_ceiling=field):
                limits = self.service.limits.copy()
                limits.pop(field)
                self.service.limits = limits
                candidate = workload()
                candidate["execution"][field] = 1 if field == "warmups" else 2
                with self.assertRaisesRegex(FoundryError, "ceiling"):
                    self.service.benchmark_plan(self.user, candidate)
                self.service.limits[field] = 2

    def test_revalidate_current_approvals_and_limits_at_start_but_allow_replay(self):
        revocations = (
            lambda s: s.repositories.update(project=frozenset()),
            lambda s: s.verified_snapshots.clear(),
            lambda s: setattr(s, "approved_images", frozenset()),
            lambda s: s.approved_models.clear(),
            lambda s: setattr(s, "approved_prompts", frozenset()),
            lambda s: s.approved_fixtures.clear(),
            lambda s: s.approved_validators.clear(),
            lambda s: s.limits.update(cpu_mhz=100),
            lambda s: s.limits.update(warmups=-1),
            lambda s: s.limits.update(max_attempts=0),
            lambda s: s.limits.update(network_policies=()),
        )
        for revoke in revocations:
            with self.subTest(revoke=revoke):
                # Restore the operator policy before each independent revocation.
                self.setUp()
                candidate = workload()
                candidate["execution"].update(warmups=1, max_attempts=2)
                plan = self.service.benchmark_plan(self.user, candidate)
                first = self.service.benchmark_start(self.user, plan["id"], "already-started-key")
                revoke(self.service)
                with self.assertRaises(FoundryError):
                    self.service.benchmark_start(self.user, plan["id"], "new-start-key-123")
                self.assertEqual(self.dispatcher.calls, 1)
                self.assertEqual(
                    first,
                    self.service.benchmark_start(self.user, plan["id"], "already-started-key"),
                )
                self.assertEqual(self.dispatcher.calls, 1)

    def test_dispatch_error_not_reported_as_success(self):
        class Broken(FakeDispatcher):
            def submit(self, plan, run_id):
                raise RuntimeError("should never leak sensitive details")

        self.service.dispatcher = Broken()
        plan = self.service.benchmark_plan(self.user, workload())
        result = self.service.benchmark_start(self.user, plan["id"], "repeat-this-key")
        self.assertEqual(result["state"], "failed")
        self.assertNotIn("sensitive details", str(result))
        self.assertEqual(
            result, self.service.benchmark_start(self.user, plan["id"], "repeat-this-key")
        )

    def test_without_dispatcher_refuses_submission(self):
        self.service.dispatcher = None
        plan = self.service.benchmark_plan(self.user, workload())
        with self.assertRaises(FoundryError):
            self.service.benchmark_start(self.user, plan["id"], "repeat-this-key")

    def test_compare_and_report_fail_closed_without_manifests(self):
        plan = self.service.benchmark_plan(self.user, workload())
        run = self.service.benchmark_start(self.user, plan["id"], "repeat-this-key")
        with self.assertRaises(FoundryError):
            self.service.compare_runs(self.user, [run["id"], run["id"] + "x"])
        with self.assertRaises(FoundryError):
            self.service.report_draft(self.user, [run["id"]])
        second = self.service.benchmark_start(self.user, plan["id"], "another-key-1")
        with self.assertRaises(FoundryError):
            self.service.compare_runs(self.user, [run["id"], second["id"]])
        self.assertNotIn("manifest", self.service.run_status(self.user, run["id"]))

    def test_concurrent_submission_only_once_across_services_sharing_store(self):
        plan = self.service.benchmark_plan(self.user, workload())
        other = FoundryService(
            repositories=self.service.repositories,
            providers=set(self.service.providers),
            limits=self.service.limits,
            store=self.store,
            dispatcher=self.dispatcher,
            allow_demo_dispatch=True,
            verified_snapshots=self.service.verified_snapshots,
            approved_images=set(self.service.approved_images),
            approved_models=self.service.approved_models,
            approved_prompts=set(self.service.approved_prompts),
            approved_fixtures=self.service.approved_fixtures,
            approved_validators=self.service.approved_validators,
        )
        gate = Barrier(16)

        def start(index):
            gate.wait(timeout=5)
            return (self.service if index % 2 else other).benchmark_start(
                self.user, plan["id"], "concurrent-key-123"
            )

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(start, range(16)))
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(self.dispatcher.calls, 1)

    def test_unreviewed_input_denied(self):
        changes = (
            lambda w: w["repository"].update(snapshot_id="fake"),
            lambda w: w["repository"].update(commit_sha="d" * 40),
            lambda w: w["sandbox"].update(image_digest="other/worker@sha256:" + HEX),
            lambda w: w["providers"][0].update(model="unreviewed-model"),
            lambda w: w["task"].update(prompt_sha256="d" * 64),
            lambda w: w["task"].update(fixture_digests={"oracle": "d" * 64}),
            lambda w: w["task"].pop("fixture_digests"),
            lambda w: w["validation"].update(evaluator_version="fake"),
            lambda w: w["validation"]["validators"][0].update(parameters={"flag": True}),
            lambda w: w["validation"]["validators"][0].update(required=False),
        )
        for change in changes:
            with self.subTest(change=change):
                candidate = workload()
                change(candidate)
                with self.assertRaises(FoundryError):
                    self.service.benchmark_plan(self.user, candidate)

    def test_request_is_not_snapshot_verification_and_missing_registries_deny(self):
        requester = Principal("missy", "project", frozenset({"read", "snapshot:start"}))
        requested = self.service.snapshot_start(
            requester, "MissyLabs/missy", "b" * 40, "requested-snapshot-key"
        )
        candidate = workload()
        candidate["repository"]["snapshot_id"] = requested["id"]
        with self.assertRaises(FoundryError):
            self.service.benchmark_plan(self.user, candidate)
        defaults = FoundryService(
            repositories={"project": {"MissyLabs/missy"}},
            providers={"one"},
            limits=self.service.limits,
            dispatcher=self.dispatcher,
        )
        with self.assertRaises(FoundryError):
            defaults.benchmark_plan(self.user, workload())
        self.assertEqual(self.dispatcher.calls, 0)

    def test_no_default_live_dispatch_even_with_dispatcher(self):
        self.service.allow_demo_dispatch = False
        plan = self.service.benchmark_plan(self.user, workload())
        with self.assertRaises(FoundryError):
            self.service.benchmark_start(self.user, plan["id"], "disabled-key")
        self.assertEqual(self.dispatcher.calls, 0)

    def test_provider_selection_affects_plan_not_canonical_definition(self):
        self.service.providers = frozenset({"one", "two"})
        self.service.approved_models["two"] = frozenset({"registered-model"})
        from dataclasses import replace

        self.project_policy = replace(
            self.project_policy,
            provider_models={
                **self.project_policy.provider_models,
                "two": frozenset({"registered-model"}),
            },
        )
        alternate = workload()
        alternate["providers"][0]["registry_key"] = "two"
        left = self.service.benchmark_plan(self.user, workload())
        right = self.service.benchmark_plan(self.user, alternate)
        self.assertNotEqual(left["id"], right["id"])
        self.assertEqual(left["definition_sha256"], right["definition_sha256"])
        self.assertEqual(left["comparability_key"], right["comparability_key"])

    def test_contradictory_persisted_hashes_rejected_without_dispatch(self):
        plan = self.service.benchmark_plan(self.user, workload())
        poisoned = self.store.get(plan["id"])
        poisoned["definition_sha256"] = "d" * 64
        self.store.update(plan["id"], poisoned)
        with self.assertRaisesRegex(FoundryError, "identity"):
            self.service.benchmark_start(self.user, plan["id"], "poisoned-key-123")
        self.assertEqual(self.dispatcher.calls, 0)

    def test_unverified_and_forged_manifests_do_not_become_comparisons(self):
        plan = self.service.benchmark_plan(self.user, workload())
        run = self.service.benchmark_start(self.user, plan["id"], "manifest-key-123")
        forged = self.store.get(run["id"])
        forged["manifest"] = {
            "run_id": run["id"],
            "project_id": "project",
            "definition_sha256": "d" * 64,
            "comparability_key": plan["comparability_key"],
        }
        self.store.update(run["id"], forged)
        with self.assertRaises(FoundryError):
            self.service.report_draft(self.user, [run["id"]])
        forged["state"] = "verified"  # Direct store corruption, not a supported verification path.
        self.store.update(run["id"], forged)
        with self.assertRaisesRegex(FoundryError, "conflicts"):
            self.service.report_draft(self.user, [run["id"]])

    def test_snapshot_request_is_pinned_and_idempotent(self):
        principal = Principal("missy", "project", frozenset({"read", "snapshot:start"}))
        first = self.service.snapshot_start(
            principal, "MissyLabs/missy", "b" * 40, "snapshot-key-123"
        )
        self.assertEqual(
            first,
            self.service.snapshot_start(principal, "MissyLabs/missy", "b" * 40, "snapshot-key-123"),
        )
        self.assertEqual(first["state"], "requested")
        self.assertEqual(self.service.snapshot_status(principal, first["id"]), first)


if __name__ == "__main__":
    unittest.main()
