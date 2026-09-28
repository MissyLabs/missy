import json

import pytest

from missy.repoeval.nomad import JobPlanError, JobRequest, plan_job
from missy.repoeval.placement import PoolCapacity

IMAGE = "registry.example/worker@sha256:" + "a" * 64
POOLS = [PoolCapacity("staging", 4000, 8192, 10000)]


def req(**kw):
    args = {
        "run_id": "run_12345678",
        "repository_commit_sha": "a" * 40,
        "snapshot_id": "snapshot-123",
        "image_digest": IMAGE,
        "workload_class": "tool-call",
        "cpu_mhz": 1000,
        "memory_mb": 1024,
        "disk_mb": 512,
        "timeout_seconds": 60,
    }
    args.update(kw)
    return JobRequest(**args)


def test_plan_is_finite_isolated_and_digest_pinned():
    plan = plan_job(req(), POOLS)
    job = plan.job
    group = job["TaskGroups"][0]
    task = group["Tasks"][0]
    assert job["Type"] == "batch" and job["NodePool"] == "staging"
    assert job["Meta"]["repository_commit_sha"] == "a" * 40
    assert job["Meta"]["snapshot_id"] == "snapshot-123"
    assert task["Name"] == "worker" and task["Driver"] == "docker"
    assert task["Config"]["image"] == IMAGE and task["Config"]["readonly_rootfs"] is True
    assert task["Config"]["network_mode"] == "none" and task["Config"]["cap_drop"] == ["ALL"]
    assert "command" not in task["Config"] and "args" not in task["Config"]
    assert "mount" not in task["Config"]
    assert (
        job["TaskGroups"][0]["Count"] == 1
        and job["TaskGroups"][0]["RestartPolicy"]["Attempts"] == 0
    )
    assert group["MaxRunDuration"] == 60_000_000_000
    assert group["ReschedulePolicy"]["Attempts"] == 0
    assert group["ReschedulePolicy"]["Unlimited"] is False
    assert task["Env"]["FOUNDRY_TIMEOUT_SECONDS"] == "60"
    assert task["KillTimeout"] == 5_000_000_000
    assert task["Resources"] == {"CPU": 1000, "MemoryMB": 1024}
    assert group["EphemeralDisk"]["SizeMB"] == 512
    assert "ReschedulePolicy" not in job
    assert "Update" not in job
    json.dumps(job)


@pytest.mark.parametrize("seconds", [1, 60, 86400])
def test_deadline_is_enforced_in_group_in_nomad_nanoseconds(seconds):
    job = plan_job(req(timeout_seconds=seconds), POOLS).job
    group = job["TaskGroups"][0]
    assert group["MaxRunDuration"] == seconds * 1_000_000_000
    assert group["Tasks"][0]["Env"]["FOUNDRY_TIMEOUT_SECONDS"] == str(seconds)


def test_resource_values_use_nomad_wire_keys_without_task_disk():
    plan = plan_job(req(cpu_mhz=2100, memory_mb=2048, disk_mb=768), POOLS)
    group = plan.job["TaskGroups"][0]
    assert group["Tasks"][0]["Resources"] == {"CPU": 2100, "MemoryMB": 2048}
    assert group["EphemeralDisk"]["SizeMB"] == 768
    assert "resources" not in group["Tasks"][0]


def test_budget_from_all_eligible_pools_is_advisory_not_spill_or_shrink():
    pools = [
        PoolCapacity("staging", 1000, 1024, 512),
        PoolCapacity("production", 4000, 8192, 10000),
        PoolCapacity("ineligible", 5000, 5000, 5000, False),
    ]
    plan = plan_job(req(), iter(pools))
    assert plan.pool == plan.job["NodePool"] == "staging"
    assert (
        plan.capacity_budget.available_cpu_mhz,
        plan.capacity_budget.available_memory_mb,
        plan.capacity_budget.available_disk_mb,
    ) == (5000, 9216, 10512)
    assert (
        plan.envelope.cpu_mhz == plan.job["TaskGroups"][0]["Tasks"][0]["Resources"]["CPU"] == 1000
    )
    assert plan.envelope.memory_mb == 1024 and plan.envelope.disk_mb == 512


def test_other_pool_cannot_rescue_insufficient_staging():
    pools = [
        PoolCapacity("staging", 999, 8192, 10000),
        PoolCapacity("production", 4000, 8192, 10000),
    ]
    with pytest.raises(ValueError, match="staging"):
        plan_job(req(), pools)


@pytest.mark.parametrize(
    "changes",
    [
        {"image_digest": "worker:latest"},
        {"workload_class": "arbitrary"},
        {"timeout_seconds": 0},
        {"timeout_seconds": 86401},
        {"run_id": "bad id"},
        {"namespace": "../prod"},
        {"cpu_mhz": 999999},
        {"repository_commit_sha": "main"},
        {"snapshot_id": ""},
    ],
)
def test_rejects_untrusted_or_unbounded_inputs(changes):
    with pytest.raises((JobPlanError, ValueError)):
        plan_job(req(**changes), POOLS)


def test_no_capacity_is_refused_without_mutating_snapshot():
    pools = [PoolCapacity("staging", 500, 256, 256)]
    with pytest.raises(ValueError, match="staging"):
        plan_job(req(), pools)
    assert pools[0].available_cpu_mhz == 500
