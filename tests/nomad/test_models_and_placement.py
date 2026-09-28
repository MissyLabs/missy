from __future__ import annotations

import pytest

from missy.config.settings import NomadConfig
from missy.nomad.errors import NomadValidationError
from missy.nomad.models import NomadJobRequest, build_job, choose_scope, spec_hash
from missy.nomad.placement import recommend_placement

from .conftest import healthy_node


def request(image: str = "", **overrides) -> NomadJobRequest:
    values = {
        "purpose": "benchmark repository",
        "image": image,
        "job_type": "batch",
        "command": "/bin/sh",
        "args": ["-c", "printf complete"],
        "cpu_mhz": 500,
        "memory_mb": 512,
        "disk_mb": 256,
        "max_run_seconds": 600,
        "architecture": "amd64",
    }
    values.update(overrides)
    return NomadJobRequest(**values)


def test_build_safe_batch_job(nomad_config: NomadConfig, pinned_image: str) -> None:
    req = request(pinned_image)
    spec = build_job(req, nomad_config, tracking_id="tracking-1")
    job = spec["Job"]
    task = job["TaskGroups"][0]["Tasks"][0]
    assert job["Type"] == "batch"
    assert job["Namespace"] == "testing"
    assert job["Meta"]["owner"] == "missy-bot"
    assert job["Meta"]["tracking-id"] == "tracking-1"
    assert task["Driver"] == "docker"
    assert task["User"] == "65534:65534"
    assert task["Config"]["cap_drop"] == ["ALL"]
    assert job["TaskGroups"][0]["MaxRunDuration"] == 600_000_000_000
    assert len(spec_hash(spec)) == 64


def test_service_has_health_check_and_update_policy(
    nomad_config: NomadConfig, pinned_image: str
) -> None:
    req = request(
        pinned_image,
        job_type="service",
        ports=[{"label": "http", "to": 8080}],
        service={"name": "bench-api", "port": "http", "check_path": "/health"},
    )
    spec = build_job(req, nomad_config)
    group = spec["Job"]["TaskGroups"][0]
    assert group["Update"]["MaxParallel"] == 1
    assert group["Tasks"][0]["Services"][0]["Checks"][0]["Path"] == "/health"


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"image": "alpine:latest"}, "sha256"),
        ({"cpu_mhz": 999_999}, "cpu_mhz"),
        ({"environment": {"API_TOKEN": "secret"}}, "Secret-shaped"),
        ({"job_id": "other-job"}, "beginning with 'missy-'"),
        ({"ports": [{"label": "http", "to": 70000}]}, "between 1 and 65535"),
    ],
)
def test_request_validation_rejects_unsafe_input(
    nomad_config: NomadConfig, pinned_image: str, changes: dict, match: str
) -> None:
    req = request(**({"image": pinned_image} | changes))
    with pytest.raises(NomadValidationError, match=match):
        req.validate(nomad_config)


def test_stateful_requires_complete_plan(nomad_config: NomadConfig, pinned_image: str) -> None:
    req = request(pinned_image, stateful=True, persistence_plan={"volume": "scratch"})
    with pytest.raises(NomadValidationError, match="persistence_plan"):
        req.validate(nomad_config)


def test_stateful_submission_stays_blocked_until_volume_discovery(
    nomad_config: NomadConfig, pinned_image: str
) -> None:
    req = request(
        pinned_image,
        stateful=True,
        persistence_plan={
            "volume": "scratch",
            "node_id": "node-1",
            "backup": "daily",
            "restore": "documented",
            "migration": "none",
            "data_loss_boundary": "last backup",
        },
    )
    with pytest.raises(NomadValidationError, match="volume is verified"):
        req.validate(nomad_config)


def test_stateful_placement_honors_volume_locality(
    nomad_config: NomadConfig, pinned_image: str
) -> None:
    req = request(
        pinned_image,
        stateful=True,
        persistence_plan={
            "volume": "scratch",
            "node_id": "node-1",
            "backup": "daily",
            "restore": "documented",
            "migration": "none",
            "data_loss_boundary": "last backup",
        },
    )
    req.validate(nomad_config, allow_unverified_stateful=True)
    choose_scope(req, nomad_config)
    other = healthy_node(node_id="node-2", name="worker-2")
    local = healthy_node(node_id="node-1", name="worker-1")
    result = recommend_placement([other, local], req, nomad_config)
    assert result["recommended_node"]["node_id"] == "node-1"


def test_scope_must_be_authorized(nomad_config: NomadConfig, pinned_image: str) -> None:
    req = request(pinned_image, namespace="production")
    with pytest.raises(NomadValidationError, match="outside"):
        choose_scope(req, nomad_config)


def test_placement_prefers_non_protected_node(nomad_config: NomadConfig, pinned_image: str) -> None:
    req = request(pinned_image)
    req.validate(nomad_config)
    choose_scope(req, nomad_config)
    protected = healthy_node(node_id="g", name="Gato", memory=32_000)
    worker = healthy_node(node_id="w", name="worker", memory=16_000)
    result = recommend_placement([protected, worker], req, nomad_config)
    assert result["recommended_node"]["name"] == "worker"


def test_placement_rejects_missing_headroom(nomad_config: NomadConfig, pinned_image: str) -> None:
    req = request(pinned_image, memory_mb=4096)
    req.validate(nomad_config)
    choose_scope(req, nomad_config)
    node = healthy_node(memory=1024, allocated_memory=800)
    with pytest.raises(NomadValidationError, match="No observed node"):
        recommend_placement([node], req, nomad_config)


def test_placement_reports_missing_host_pressure(
    nomad_config: NomadConfig, pinned_image: str
) -> None:
    req = request(pinned_image)
    req.validate(nomad_config)
    choose_scope(req, nomad_config)
    node = healthy_node()
    node.pop("HostStats")
    result = recommend_placement([node], req, nomad_config)
    assert "scheduler headroom does not prove host headroom" in result["warnings"][0]


def test_placement_rejects_observed_host_pressure(
    nomad_config: NomadConfig, pinned_image: str
) -> None:
    req = request(pinned_image, memory_mb=512, disk_mb=256)
    req.validate(nomad_config)
    choose_scope(req, nomad_config)
    node = healthy_node()
    node["HostStats"]["Memory"]["Available"] = 1
    with pytest.raises(NomadValidationError, match="No observed node"):
        recommend_placement([node], req, nomad_config)


def test_batch_artifact_contract_and_retry_are_rendered(
    nomad_config: NomadConfig, pinned_image: str
) -> None:
    req = request(
        pinned_image,
        retry_attempts=1,
        parameters={"revision": "abc123"},
        input_artifacts={"source": "artifact://missy/inputs/source.tar.zst"},
        output_artifacts={"result": "artifact://missy/results/result.json"},
        result_schema={
            "required": ["score", "outputs"],
            "properties": {"score": {"type": "number"}, "outputs": {"type": "object"}},
        },
    )
    spec = build_job(req, nomad_config)
    group = spec["Job"]["TaskGroups"][0]
    env = group["Tasks"][0]["Env"]
    assert group["RestartPolicy"]["Attempts"] == 1
    assert "artifact://missy/inputs/source.tar.zst" in env["MISSY_INPUT_ARTIFACTS_JSON"]
    assert "artifact://missy/results/result.json" in env["MISSY_OUTPUT_ARTIFACTS_JSON"]


def test_secret_is_runtime_reference_not_embedded_value(
    nomad_config: NomadConfig, pinned_image: str
) -> None:
    req = request(
        pinned_image,
        secret_references={"API_TOKEN": "nomad-var://missy/repository-check#api_token"},
    )
    spec = build_job(req, nomad_config)
    task = spec["Job"]["TaskGroups"][0]["Tasks"][0]
    assert "API_TOKEN" not in (task["Env"] or {})
    assert "nomadVar" in task["Templates"][0]["EmbeddedTmpl"]
    assert "api_token" in task["Templates"][0]["EmbeddedTmpl"]


def test_unapproved_secret_reference_is_rejected(
    nomad_config: NomadConfig, pinned_image: str
) -> None:
    req = request(
        pinned_image,
        secret_references={"API_TOKEN": "nomad-var://someone-else/path#token"},
    )
    with pytest.raises(NomadValidationError, match="approved Nomad Variables"):
        req.validate(nomad_config)


def test_unapproved_container_command_is_rejected(
    nomad_config: NomadConfig, pinned_image: str
) -> None:
    req = request(pinned_image, command="/bin/arbitrary-shell")
    with pytest.raises(NomadValidationError, match="not explicitly allowed"):
        req.validate(nomad_config)


def test_unapproved_artifact_transport_is_rejected(
    nomad_config: NomadConfig, pinned_image: str
) -> None:
    req = request(
        pinned_image,
        input_artifacts={"source": "https://example.test/signed-secret"},
    )
    with pytest.raises(NomadValidationError, match="approved artifact storage"):
        req.validate(nomad_config)


def test_service_capacity_includes_rollout_overlap(
    nomad_config: NomadConfig, pinned_image: str
) -> None:
    req = request(pinned_image, job_type="service", count=2, cpu_mhz=500)
    req.validate(nomad_config)
    choose_scope(req, nomad_config)
    result = recommend_placement([healthy_node()], req, nomad_config)
    assert result["demand"]["cpu_mhz"] == 1500
    assert result["demand"]["temporary_rollout_allocations"] == 1
