"""Exercise Missy's real Foundry client against the in-process core API.

An in-memory HTTP-shaped transport forwards requests to FoundryAPI.handle().
It never opens a socket, contacts a provider, or submits a job. Crucially,
the assertions describe what the existing client actually accepts, not what
we wish the future deployed service would provide.
"""

from __future__ import annotations

import json as json_module
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from missy.repoeval.api import FoundryAPI
from missy.repoeval.artifacts import make_manifest
from missy.repoeval.contracts import sha256_json
from missy.repoeval.control import FoundryService, Principal
from missy.repoeval.placement import PoolCapacity, ResourceEnvelope
from missy.repoeval.planning import CapacitySnapshot, PlanningAuthority, ProjectPolicySnapshot
from missy.tools.builtin.repoeval_tools import RepoevalFoundryTool

PROJECT = "fixture-project"
REPO = "MissyLabs/missy"
COMMIT = "a" * 40
IMAGE = "registry.invalid/offline@sha256:" + "b" * 64
PROMPT = "c" * 64
FIXTURE = "d" * 64
MODEL = "fixture-model"
TOKEN = "fixture-only-not-a-real-credential"


class InMemoryTransport:
    """A closed, deterministic substitute for PolicyHTTPClient in tests only."""

    category = "tool"
    response_limit = 1024 * 1024

    def __init__(self, api: FoundryAPI):
        self.api = api
        self.calls: list[tuple[str, str]] = []
        self.limited_calls: list[tuple[str, int, dict, bool]] = []

    def _forward(self, method, url, max_bytes, *, headers, json=None, follow_redirects):
        # Keep the transport contract under test: callers must request a
        # bounded response and must not delegate redirects to the HTTP layer.
        assert max_bytes == self.response_limit
        assert follow_redirects is False
        assert headers.get("Authorization") == "Bearer " + TOKEN
        assert headers.get("Accept") == "application/json"
        self.limited_calls.append((method, max_bytes, dict(headers), follow_redirects))
        parsed = urlsplit(url)
        assert parsed.scheme == "https" and parsed.netloc == "foundry.invalid"
        assert not parsed.query and not parsed.fragment
        self.calls.append((method, parsed.path))
        response = self.api.handle(method, parsed.path, headers, json)
        assert len(json_module.dumps(response.body).encode("utf-8")) <= max_bytes
        return WireResponse(response.status, response.body)

    def get_limited(self, url, max_bytes, *, headers, follow_redirects):
        return self._forward(
            "GET", url, max_bytes, headers=headers, follow_redirects=follow_redirects
        )

    def post_limited(self, url, max_bytes, *, headers, json, follow_redirects):
        return self._forward(
            "POST", url, max_bytes, headers=headers, json=json, follow_redirects=follow_redirects
        )


class WireResponse:
    def __init__(self, status_code, body):
        self.status_code, self.body = status_code, body

    def json(self):
        return self.body


def _workload(snapshot_id: str) -> dict:
    return {
        "schema_version": "1.0",
        "id": "offline-client-contract",
        "version": "1",
        "repository": {"repository_id": REPO, "commit_sha": COMMIT, "snapshot_id": snapshot_id},
        "task": {
            "class": "tool-call",
            "prompt_sha256": PROMPT,
            "fixture_digests": {"oracle": FIXTURE},
        },
        "providers": [{"registry_key": "offline-fake", "model": MODEL, "settings": {}}],
        "sandbox": {
            "image_digest": IMAGE,
            "cpu_mhz": 300,
            "memory_mb": 256,
            "disk_mb": 512,
            "network_policy": "offline",
        },
        "execution": {
            "warmups": 0,
            "repetitions": 1,
            "parallelism": 1,
            "timeout_seconds": 120,
            "max_attempts": 1,
        },
        "validation": {
            "evaluator_version": "fixture-v1",
            "validators": [{"id": "oracle", "version": "1", "required": True, "parameters": {}}],
        },
        "artifacts": {
            "required_kinds": ["result"],
            "output_prefix": "results/offline",
            "retention_class": "fixture",
        },
    }


def _wired(tmp_path, *, planning=False, artifact_reader=None, artifact_clearance=None):
    token_path = tmp_path / "fixture.token"
    token_path.write_text(TOKEN, encoding="ascii")
    token_path.chmod(0o600)
    principal = Principal("fixture-user", PROJECT, frozenset({"read", "snapshot:start", "execute"}))
    service = FoundryService(
        repositories={PROJECT: {REPO}},
        providers={"offline-fake"},
        limits={
            "cpu_mhz": 500,
            "memory_mb": 512,
            "disk_mb": 1024,
            "repetitions": 1,
            "parallelism": 1,
            "timeout_seconds": 120,
            "network_policies": ("offline",),
        },
        approved_images={IMAGE},
        approved_models={"offline-fake": {MODEL}},
        approved_prompts={PROMPT},
        approved_fixtures={"oracle": FIXTURE},
        approved_validators={"fixture-v1": {("oracle", "1", sha256_json({}))}},
        planning_authority=(
            PlanningAuthority(
                lambda: CapacitySnapshot(
                    "fixture-capacity", 1000.0, (PoolCapacity("staging", 3000, 4096, 8192),)
                ),
                lambda project: ProjectPolicySnapshot(
                    PROJECT,
                    "fixture-policy",
                    1000.0,
                    frozenset({REPO}),
                    frozenset({IMAGE}),
                    {"offline-fake": frozenset({MODEL})},
                    ResourceEnvelope(500, 512, 1024),
                    10_000_000,
                    frozenset({"offline"}),
                    "fixture-audit",
                ),
                clock=lambda: 1000.0,
            )
            if planning
            else None
        ),
        artifact_reader=artifact_reader,
        artifact_clearance=artifact_clearance,
        # No dispatcher. No production snapshot evidence or executable image.
    )
    api = FoundryAPI(
        service,
        lambda headers: principal if headers.get("Authorization") == f"Bearer {TOKEN}" else None,
    )
    wire = InMemoryTransport(api)
    client = RepoevalFoundryTool(
        base_url="https://foundry.invalid/api",
        project_id=PROJECT,
        allowed_hosts=["foundry.invalid"],
        api_available=True,
        token_file=str(token_path),
        http_client=wire,
    )
    assert client.registration_ready
    return client, wire, service


def _verified_test_run(service, run_id, *, project=PROJECT, content=None):
    snapshot_id = "reviewed-fixture-snapshot"
    service.verified_snapshots[snapshot_id] = {
        "id": snapshot_id,
        "project_id": PROJECT,
        "repository_id": REPO,
        "commit_sha": COMMIT,
        "state": "verified",
    }
    plan = service.benchmark_plan(
        Principal("fixture-user", PROJECT, frozenset({"read"})), _workload(snapshot_id)
    )
    w = plan["workload"]
    manifest = {
        "run_id": run_id,
        "project_id": project,
        "comparability_rules_version": "1.0",
        "repository": w["repository"],
        "task": w["task"],
        "workload_id": w["id"],
        "workload_version": w["version"],
        "definition_sha256": plan["definition_sha256"],
        "sandbox": {**w["sandbox"], "timeout_seconds": w["execution"]["timeout_seconds"]},
        "evaluator_version": w["validation"]["evaluator_version"],
        "validators": [
            {"id": "oracle", "version": "1", "required": True, "parameters_sha256": sha256_json({})}
        ],
        "provider_independent_settings_sha256": sha256_json({}),
        "tool_schemas_sha256": sha256_json([]),
        "comparability_key": plan["comparability_key"],
    }
    record = {
        "id": run_id,
        "project_id": project,
        "plan_id": plan["id"],
        "state": "verified",
        "manifest": manifest,
        "manifest_sha256": sha256_json(manifest),
    }
    if content is not None:
        import hashlib

        record["artifacts"] = [
            make_manifest(
                artifact_id="artifact-offline001",
                run_id=run_id,
                kind="result",
                uri="objects/owned-by-server-not-returned",
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                media_type="text/plain",
                producer={"component": "fixture", "version": "1"},
                classification="internal",
                retention_class="fixture",
                scan_state="clean",
                redaction_state="complete",
            )
        ]
    service.store.put_once(run_id, record)
    return record


def test_verified_project_scoped_comparison_draft_and_digest_checked_artifacts(tmp_path):
    payload = b"cleared result"
    reads = []

    def read(project, run, artifact):
        reads.append((project, run, artifact))
        return payload

    client, wire, service = _wired(
        tmp_path,
        planning=True,
        artifact_reader=read,
        artifact_clearance=lambda project, run, manifest, data: (
            project == PROJECT and run == manifest["run_id"] and data == payload
        ),
    )
    first, second = "run-offline0001", "run-offline0002"
    _verified_test_run(service, first, content=payload)
    _verified_test_run(service, second)
    comparison = client.execute(action="compare", run_ids=[first, second])
    assert comparison.success and comparison.output["project_id"] == PROJECT
    assert comparison.output["pairs"][0]["comparable"] is True
    assert comparison.output["comparison_id"].startswith("comparison-")
    artifact_result = client.execute(action="artifacts", run_id=first)
    assert artifact_result.success
    assert reads == [(PROJECT, first, "artifact-offline001")]
    assert set(artifact_result.output["artifacts"][0]) == {
        "artifact_id",
        "run_id",
        "kind",
        "sha256",
        "size_bytes",
        "media_type",
    }
    report = client.execute(action="report", run_ids=[first, second])
    assert report.success and report.output["status"] == "draft"
    assert report.output["published"] is False
    assert report.output["groups"][0]["run_ids"] == [first, second]
    assert wire.calls[-3:] == [
        ("POST", f"/api/projects/{PROJECT}/compare"),
        ("GET", f"/api/projects/{PROJECT}/runs/{first}/artifacts"),
        ("POST", f"/api/projects/{PROJECT}/report"),
    ]


def test_cross_project_and_tampered_artifacts_refused_at_service_wire(tmp_path):
    payload = b"good"
    client, wire, service = _wired(
        tmp_path,
        planning=True,
        artifact_reader=lambda *parts: b"tampered",
        artifact_clearance=lambda *parts: True,
    )
    first, foreign = "run-offline0003", "run-foreign0004"
    _verified_test_run(service, first, content=payload)
    _verified_test_run(service, foreign, project="foreign")
    assert not client.execute(action="compare", run_ids=[first, foreign]).success
    assert not client.execute(action="report", run_ids=[foreign]).success
    assert not client.execute(action="artifacts", run_id=first).success
    same_project = service.store.get(first)
    same_project["manifest"]["task"]["prompt_sha256"] = "f" * 64
    service.store.update(first, same_project)
    assert not client.execute(action="report", run_ids=[first]).success


def test_real_missy_client_reaches_native_core_for_scoped_list_snapshot_and_status(tmp_path):
    client, wire, service = _wired(tmp_path)
    listed = client.execute(action="list")
    assert listed.success
    assert listed.output["repositories"] == [REPO]
    assert listed.output["route_project_id"] == PROJECT

    requested = client.execute(
        action="snapshot",
        repository_id=REPO,
        commit_sha=COMMIT,
        idempotency_key="offline-snapshot-01",
        self_approve=True,
    )
    assert requested.success
    assert requested.output["acknowledged"] is True
    assert requested.output["execution_complete"] is False
    snapshot = requested.output["resource"]
    assert snapshot["state"] == "requested"
    assert snapshot["project_id"] == PROJECT
    status = client.execute(action="status", resource_type="snapshot", resource_id=snapshot["id"])
    assert status.success
    assert status.output["state"] == "requested"
    assert status.output["project_id"] == PROJECT
    assert wire.calls == [
        ("GET", f"/api/projects/{PROJECT}/repositories"),
        ("POST", f"/api/projects/{PROJECT}/snapshots"),
        ("GET", f"/api/projects/{PROJECT}/snapshots/{snapshot['id']}"),
    ]
    assert len(wire.limited_calls) == 3
    assert all(
        limit == wire.response_limit
        and headers["Authorization"] == "Bearer " + TOKEN
        and redirects is False
        for _method, limit, headers, redirects in wire.limited_calls
    )
    assert service.dispatcher is None
    assert service.allow_demo_dispatch is False


def test_core_plan_round_trip_still_cannot_authorize_client_start(tmp_path):
    client, wire, service = _wired(tmp_path)
    assert client.execute(action="list").success
    snapshot_id = "reviewed-fixture-snapshot"
    service.verified_snapshots[snapshot_id] = {
        "id": snapshot_id,
        "project_id": PROJECT,
        "repository_id": REPO,
        "commit_sha": COMMIT,
        "state": "verified",
    }
    workload = _workload(snapshot_id)
    planned = client.execute(action="plan", workload=workload)
    assert (
        not planned.success
    )  # No trusted staging capacity/policy provider was injected into this core.
    assert planned.error
    assert client._plans == {}
    calls_before = len(wire.calls)
    refused = client.execute(
        action="start",
        plan_id="plan-unsupported",
        idempotency_key="offline-start-01",
        self_approve=True,
    )
    assert not refused.success
    assert len(wire.calls) == calls_before  # No start request, even to the in-memory facade.
    assert [method for method, path in wire.calls if path.endswith("/benchmark/start")] == []
    assert service.dispatcher is None


def test_unapproved_snapshot_and_revoked_client_fail_closed(tmp_path):
    client, wire, service = _wired(tmp_path)
    assert client.execute(action="list").success
    # Snapshot request merely marks it requested. A caller cannot use it as a
    # verified snapshot by selecting its ID in the workload.
    requested = client.execute(
        action="snapshot",
        repository_id=REPO,
        commit_sha=COMMIT,
        idempotency_key="offline-snapshot-02",
        self_approve=True,
    )
    snapshot_id = requested.output["resource"]["id"]
    assert not client.execute(action="plan", workload=_workload(snapshot_id)).success
    client.revoke()
    count = len(wire.calls)
    assert not client.execute(action="list").success
    assert len(wire.calls) == count
    assert service.dispatcher is None


def test_bundled_missy_catalog_repository_id_is_the_native_wire_identity(tmp_path):
    catalog = Path(__file__).resolve().parents[2] / (
        "missy/repoeval/workloads/missy/tool-call-correctness/workload.json"
    )
    repository_id = json_module.loads(catalog.read_text(encoding="utf-8"))["repository"][
        "repository_id"
    ]
    assert repository_id == REPO == RepoevalFoundryTool._repository_id(repository_id)
    client, wire, _ = _wired(tmp_path)
    assert client.execute(action="list").output["repositories"] == [repository_id]
    assert client.execute(
        action="snapshot",
        repository_id=repository_id,
        commit_sha=COMMIT,
        idempotency_key="catalog-snapshot-01",
        self_approve=True,
    ).success
    assert wire.calls[-1] == ("POST", f"/api/projects/{PROJECT}/snapshots")


@pytest.mark.parametrize(
    "malicious",
    [
        "MissyLabs/../missy",
        "../missy",
        "MissyLabs/..",
        "./missy",
        "MissyLabs//missy",
        "/MissyLabs/missy",
        "MissyLabs/missy/extra",
        "MissyLabs%2Fmissy",
        "MissyLabs/%2e%2e",
        "MissyLabs\\missy",
        "https://github.com/MissyLabs/missy",
        "MissyLabs/missy?x=1",
        "MissyLabs/missy#fragment",
        "MissyLabs/missy\n",
        ".",
        "..",
    ],
)
def test_repository_identity_rejects_malicious_refs_before_transport(tmp_path, malicious):
    with pytest.raises(ValueError, match="Invalid repository ID"):
        RepoevalFoundryTool._repository_id(malicious)
    client, wire, service = _wired(tmp_path)
    service.repositories[PROJECT] = frozenset({malicious})
    assert not client.execute(action="list").success
    assert wire.calls == [("GET", f"/api/projects/{PROJECT}/repositories")]
