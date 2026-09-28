"""Opt-in real-wire Foundry loopback, without external providers or a scheduler.

Run explicitly with ``pytest -q tests/repoeval/test_closed_loop.py``.  The
listener is bound only to ephemeral 127.0.0.1, and all persistent state and
credentials live beneath pytest's temporary directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from pathlib import Path

import pytest

from missy.config.settings import (
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ShellPolicy,
)
from missy.gateway.client import PolicyHTTPClient
from missy.policy import engine as engine_module
from missy.policy.engine import init_policy_engine
from missy.repoeval.artifacts import make_manifest
from missy.repoeval.contracts import canonical_json, sha256_json
from missy.repoeval.control import FoundryService, Principal
from missy.repoeval.dispatch import DurableDispatcher, JobObservation, SQLiteOutbox
from missy.repoeval.http_server import CredentialVerifier, FoundryHTTPServer
from missy.repoeval.placement import PoolCapacity, ResourceEnvelope
from missy.repoeval.planning import CapacitySnapshot, PlanningAuthority, ProjectPolicySnapshot
from missy.repoeval.provider import ProviderResult
from missy.repoeval.worker import WorkerRequest, execute_worker
from missy.tools.builtin.repoeval_tools import RepoevalFoundryTool

PROJECT = "e2e-project"
REPO = "MissyLabs/missy"
MODEL = "finite-stub-model"
PROVIDER = "e2e-stub"
IMAGE = "registry.example/approved-offline@sha256:" + "1" * 64
TOKEN = "e2e-test-credential-not-production"
ROOT = Path(__file__).resolve().parents[2] / "missy/repoeval/workloads/missy/repository-orientation"
ANSWER = (ROOT / "fixtures/orientation-facts.json").read_text(encoding="utf-8")


class FiniteBroker:
    def __init__(self):
        self.requests = []

    def execute(self, **kwargs):
        self.requests.append(kwargs)
        assert len(self.requests) <= 1, "finite one-attempt fake provider"
        assert kwargs["project_id"] == PROJECT
        assert kwargs["registry_key"] == PROVIDER
        assert kwargs["model"] == MODEL
        return ProviderResult(
            PROVIDER,
            MODEL,
            MODEL,
            None,
            kwargs["request"].digest,
            ANSWER,
            hashlib.sha256(ANSWER.encode()).hexdigest(),
            False,
            1,
            1,
            1,
            5,
        )


class FiniteScheduler:
    """Exact, in-memory scheduler observation; submits one worker synchronously."""

    def __init__(self, worker):
        self.worker = worker
        self.jobs = {}
        self.submissions = 0

    def submit(self, namespace, job):
        self.submissions += 1
        assert self.submissions == 1, "must never replay a job"
        assert namespace == "e2e"
        identity = (namespace, job["ID"])
        assert identity not in self.jobs
        self.jobs[identity] = (copy.deepcopy(job), "running")
        self.worker()
        self.jobs[identity] = (copy.deepcopy(job), "complete")

    def lookup(self, namespace, job_id):
        entry = self.jobs.get((namespace, job_id))
        return JobObservation(*entry) if entry else None

    def stop(self, namespace, job_id):
        raise AssertionError("test never cancels a worker")


def _fixture():
    workload = json.loads((ROOT / "workload.json").read_text(encoding="utf-8"))
    # Client refuses URI-bearing fields; the immutable prompt/fixture digests
    # and test-owned bytes, not a caller-supplied URI, define worker input.
    workload["task"].pop("prompt_uri")
    workload["providers"] = [{"registry_key": PROVIDER, "model": MODEL, "settings": {}}]
    workload["sandbox"]["image_digest"] = IMAGE
    payloads = {"prompt.md": (ROOT / "prompt.md").read_bytes()}
    for name in workload["task"]["fixture_digests"]:
        payloads[name] = (ROOT / "fixtures" / name).read_bytes()
    return workload, payloads


@pytest.fixture
def loopback_policy():
    previous = engine_module._engine
    init_policy_engine(
        MissyConfig(
            network=NetworkPolicy(default_deny=True, allowed_cidrs=["127.0.0.0/8"]),
            filesystem=FilesystemPolicy(),
            shell=ShellPolicy(),
            plugins=PluginPolicy(),
            providers={},
            workspace_path="/tmp",
            audit_log_path="/tmp/missy-e2e-never-written.log",
        )
    )
    try:
        yield
    finally:
        engine_module._engine = previous


def test_opt_in_real_wire_durable_worker_artifact(tmp_path, loopback_policy):
    # Import only inside this explicit test: no listener or work on collection.
    from missy.repoeval.coordinator import DurableFoundryService

    workload, payloads = _fixture()
    repo = workload["repository"]
    principal = Principal("e2e-operator", PROJECT, frozenset({"read", "execute"}))
    reconciler = Principal("e2e-verifier", PROJECT, frozenset({"read", "reconcile"}))
    broker = FiniteBroker()
    evidence = {}
    objects = {}
    now = 1000.0
    planning = PlanningAuthority(
        lambda: CapacitySnapshot("e2e-capacity", now, (PoolCapacity("staging", 3000, 4096, 8192),)),
        lambda project: ProjectPolicySnapshot(
            PROJECT,
            "e2e-policy",
            now,
            frozenset({REPO}),
            frozenset({IMAGE}),
            {PROVIDER: frozenset({MODEL})},
            ResourceEnvelope(1000, 1024, 2048),
            10_000_000,
            frozenset({"offline"}),
            "e2e-audit",
        ),
        clock=lambda: now,
    )
    foundation = FoundryService(
        repositories={PROJECT: {REPO}},
        providers={PROVIDER},
        limits={
            "cpu_mhz": 1000,
            "memory_mb": 1024,
            "disk_mb": 2048,
            "repetitions": 1,
            "parallelism": 1,
            "timeout_seconds": 120,
            "network_policies": ("offline",),
        },
        verified_snapshots={
            repo["snapshot_id"]: {
                "id": repo["snapshot_id"],
                "project_id": PROJECT,
                "repository_id": REPO,
                "commit_sha": repo["commit_sha"],
                "state": "verified",
            }
        },
        approved_images={IMAGE},
        approved_models={PROVIDER: {MODEL}},
        approved_prompts={workload["task"]["prompt_sha256"]},
        approved_fixtures=workload["task"]["fixture_digests"],
        approved_validators={
            workload["validation"]["evaluator_version"]: {
                (v["id"], v["version"], sha256_json(v["parameters"]))
                for v in workload["validation"]["validators"]
            },
        },
        planning_authority=planning,
        artifact_reader=lambda project, run, artifact: objects[(project, run, artifact)],
        artifact_clearance=lambda project, run, manifest, data: (
            project == PROJECT
            and manifest["run_id"] == run
            and data == objects[(project, run, manifest["artifact_id"])]
            and data
            == (
                canonical_json(evidence["result"].report())
                if manifest["kind"] == "validator-report"
                else ANSWER.encode("utf-8")
            )
        ),
    )

    def run_worker():
        request = WorkerRequest(
            workload,
            sha256_json(workload),
            payloads,
            PROJECT,
            PROVIDER,
            MODEL,
            1000,
            20,
            512,
        )
        evidence["result"] = execute_worker(
            request,
            approve=lambda digest, manifest: (
                digest == sha256_json(workload) and manifest == workload
            ),
            broker=broker,
        )
        assert evidence["result"].status == "passed"

    scheduler = FiniteScheduler(run_worker)
    outbox = SQLiteOutbox.initialize(tmp_path / "outbox.sqlite3")
    dispatcher = DurableDispatcher(
        outbox,
        scheduler,
        authorize=lambda project, run, job: (
            project == PROJECT
            and job.get("Namespace") == "e2e"
            and job.get("Meta", {}).get("foundry_run_id") == run
        ),
        enabled=True,
    )

    def job_factory(plan, parent_run_id, child_run_id, provider_index, repetition):
        assert plan["workload"] == workload
        assert provider_index == repetition == 0
        job_id = f"foundry-{child_run_id}"
        return {
            "ID": job_id,
            "Name": job_id,
            "Type": "batch",
            "Namespace": "e2e",
            "Meta": {
                "foundry_run_id": child_run_id,
                "foundry_parent_run_id": parent_run_id,
                "foundry_provider_index": str(provider_index),
                "foundry_repetition": str(repetition),
            },
            "NodePool": "staging",
            "Image": IMAGE,
        }

    def verifier(project, run, plan, children, manifest):
        result = evidence.get("result")
        return (
            project == PROJECT
            and plan["workload"] == workload
            and len(children) == 1
            and result is not None
            and result.status == "passed"
            and result.report_sha256 == sha256_json(result.report())
            and manifest["run_id"] == run
            and manifest["project_id"] == project
            and manifest["comparability_key"] == plan["comparability_key"]
        )

    service = DurableFoundryService.initialize(
        outbox.path,
        foundation,
        dispatcher,
        job_factory,
        verifier=verifier,
        enabled=True,
    )
    token_path = tmp_path / "credential"
    token_path.write_text(TOKEN, encoding="ascii")
    token_path.chmod(0o600)
    server = FoundryHTTPServer(service, PROJECT, CredentialVerifier({TOKEN: principal}), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    gateway = PolicyHTTPClient(category="tool", timeout=3, max_response_bytes=1024 * 1024)
    thread.start()
    tool = RepoevalFoundryTool(
        base_url=f"http://127.0.0.1:{server.server_port}/api",
        project_id=PROJECT,
        allowed_hosts=["127.0.0.1"],
        http_client=gateway,
        api_available=True,
        token_file=str(token_path),
    )
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert tool.registration_ready
        assert tool.execute(action="list").output["repositories"] == [REPO]
        planned = tool.execute(action="plan", workload=workload)
        assert planned.success, planned.error
        plan = planned.output
        assert plan["state"] == "planned" and plan["placement"]["pool"] == "staging"
        assert scheduler.submissions == 0 and broker.requests == []

        start = tool.execute(
            action="start",
            plan_id=plan["id"],
            idempotency_key="e2e-once-only",
            self_approve=True,
        )
        assert start.success, start.error
        assert start.output["acknowledged"] is True
        assert start.output["execution_complete"] is False
        assert start.output["resource"]["state"] == "reserved"
        assert start.output["resource"]["job_id"] is None
        run = service.benchmark_start(principal, plan["id"], "e2e-once-only")
        assert start.output["resource"]["id"] == run["id"]
        assert run["state"] == "reserved" and run["job_id"] is None
        assert scheduler.submissions == 0 and broker.requests == []
        status = tool.execute(action="status", resource_type="run", resource_id=run["id"])
        assert status.success and status.output["state"] == "reserved"
        assert not tool.execute(action="artifacts", run_id=run["id"]).success

        dispatched = service.dispatch_pending(principal, run["id"])
        assert scheduler.submissions == 1 and len(broker.requests) == 1
        assert evidence["result"].status == "passed"
        assert dispatched["state"] != "verified", "scheduler completion alone is not verification"
        service.reconcile_run(principal, run["id"])

        comparison_manifest = {
            "run_id": run["id"],
            "project_id": PROJECT,
            "comparability_rules_version": "1.0",
            "repository": workload["repository"],
            "task": workload["task"],
            "workload_id": workload["id"],
            "workload_version": workload["version"],
            "definition_sha256": plan["definition_sha256"],
            "sandbox": {
                **workload["sandbox"],
                "timeout_seconds": workload["execution"]["timeout_seconds"],
            },
            "evaluator_version": workload["validation"]["evaluator_version"],
            "validators": [
                {
                    "id": v["id"],
                    "version": v["version"],
                    "required": True,
                    "parameters_sha256": sha256_json(v["parameters"]),
                }
                for v in workload["validation"]["validators"]
            ],
            "provider_independent_settings_sha256": sha256_json({}),
            "tool_schemas_sha256": sha256_json(workload["task"].get("tool_schema_uris", [])),
            "comparability_key": plan["comparability_key"],
        }
        report_bytes = canonical_json(evidence["result"].report())
        assert evidence["result"].report_sha256 == hashlib.sha256(report_bytes).hexdigest()
        report_path = tmp_path / "validator-report.json"
        report_path.write_bytes(report_bytes)
        artifact = make_manifest(
            artifact_id="artifact-e2ereport01",
            run_id=run["id"],
            kind="validator-report",
            uri="objects/validator-report.json",
            sha256=hashlib.sha256(report_bytes).hexdigest(),
            size_bytes=len(report_bytes),
            media_type="application/json",
            producer={"component": "e2e-worker", "version": "1"},
            classification="internal",
            retention_class="benchmark-short",
            scan_state="clean",
            redaction_state="not-required",
        )
        objects[(PROJECT, run["id"], artifact["artifact_id"])] = report_path.read_bytes()
        response_bytes = ANSWER.encode("utf-8")
        response_path = tmp_path / "response.json"
        response_path.write_bytes(response_bytes)
        response_artifact = make_manifest(
            artifact_id="artifact-e2eresponse01",
            run_id=run["id"],
            kind="response",
            uri="objects/response.json",
            sha256=hashlib.sha256(response_bytes).hexdigest(),
            size_bytes=len(response_bytes),
            media_type="application/json",
            producer={"component": "e2e-worker", "version": "1"},
            classification="internal",
            retention_class="benchmark-short",
            scan_state="clean",
            redaction_state="not-required",
        )
        objects[(PROJECT, run["id"], response_artifact["artifact_id"])] = response_path.read_bytes()
        assert response_artifact["sha256"] == evidence["result"].response_sha256
        finished = service.finalize(
            reconciler, run["id"], comparison_manifest, [artifact, response_artifact]
        )
        assert finished["state"] == "verified"
        status = tool.execute(action="status", resource_type="run", resource_id=run["id"])
        assert status.success and status.output["state"] == "verified"
        catalog = tool.execute(action="artifacts", run_id=run["id"])
        assert catalog.success, catalog.error
        assert catalog.output["artifacts"] == [
            {
                "artifact_id": artifact["artifact_id"],
                "run_id": run["id"],
                "kind": "validator-report",
                "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "size_bytes": report_path.stat().st_size,
                "media_type": "application/json",
            },
            {
                "artifact_id": response_artifact["artifact_id"],
                "run_id": run["id"],
                "kind": "response",
                "sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
                "size_bytes": response_path.stat().st_size,
                "media_type": "application/json",
            },
        ]
        # Catalog access must re-verify bytes, not trust a persisted digest.
        objects[(PROJECT, run["id"], artifact["artifact_id"])] = b"tampered"
        assert not tool.execute(action="artifacts", run_id=run["id"]).success
        objects[(PROJECT, run["id"], artifact["artifact_id"])] = report_path.read_bytes()
        assert scheduler.submissions == len(broker.requests) == 1
        assert TOKEN not in repr((status.output, catalog.output, finished))
    finally:
        gateway.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive() and server.socket.fileno() == -1
