"""Operator-side placement and policy facts, with no scheduler network access."""

from dataclasses import replace

import pytest
from test_control import ControlTests, workload

from missy.repoeval.control import FoundryError
from missy.repoeval.placement import PoolCapacity, ResourceEnvelope
from missy.repoeval.planning import PlanningAuthority, PlanningError


@pytest.fixture
def context():
    control = ControlTests()
    control.setUp()
    return control


def test_client_contract_only_operator_snapshots_can_attest(context):
    plan = context.service.benchmark_plan(context.user, workload())
    assert plan["placement"]["pool"] == "staging"
    assert all(plan["policy_checks"].values())
    assert plan["placement"]["capacity_budget"]["available_cpu_mhz"] == 8000
    assert plan["attestation"]["policy_snapshot_sha256"]


@pytest.mark.parametrize(
    "change",
    [
        lambda p: replace(p, repositories=frozenset()),
        lambda p: replace(p, images=frozenset()),
        lambda p: replace(p, provider_models={}),
        lambda p: replace(p, remaining_budget_units=0),
        lambda p: replace(p, quota=ResourceEnvelope(100, 128, 256)),
        lambda p: replace(p, egress_policies=frozenset()),
        lambda p: replace(p, audit_sink_id=""),
    ],
)
def test_absent_policy_denies_before_plan(context, change):
    context.project_policy = change(context.project_policy)
    with pytest.raises(FoundryError) as denied:
        context.service.benchmark_plan(context.user, workload())
    assert denied.value.category == "policy"
    assert context.dispatcher.calls == 0


def test_staging_insufficient_even_when_aggregate_capacity_abundant(context):
    context.capacity = replace(
        context.capacity,
        pools=(
            PoolCapacity("staging", 299, 1000, 1000),
            PoolCapacity("production", 10_000, 10_000, 10_000),
        ),
    )
    with pytest.raises(FoundryError) as denied:
        context.service.benchmark_plan(context.user, workload())
    assert denied.value.category == "capacity"
    assert context.dispatcher.calls == 0


def test_capacity_race_denies_start_before_dispatch(context):
    plan = context.service.benchmark_plan(context.user, workload())
    context.capacity = replace(
        context.capacity,
        revision="capacity-2",
        pools=(
            PoolCapacity("staging", 200, 4096, 8192),
            PoolCapacity("production", 99_999, 99_999, 99_999),
        ),
    )
    with pytest.raises(FoundryError):
        context.service.benchmark_start(context.user, plan["id"], "racing-capacity")
    assert context.dispatcher.calls == 0


def test_revocation_and_expiry_reject_original_plan(context):
    plan = context.service.benchmark_plan(context.user, workload())
    context.project_policy = replace(context.project_policy, repositories=frozenset())
    with pytest.raises(FoundryError):
        context.service.benchmark_start(context.user, plan["id"], "revoked-policy")
    assert context.dispatcher.calls == 0
    context.project_policy = replace(
        context.project_policy, repositories=frozenset({"MissyLabs/missy"})
    )
    context.now += 31
    with pytest.raises(FoundryError, match="expired"):
        context.service.benchmark_start(context.user, plan["id"], "expired-plan")
    assert context.dispatcher.calls == 0


def test_snapshot_revision_change_denies_even_unchanged_resources(context):
    plan = context.service.benchmark_plan(context.user, workload())
    context.capacity = replace(context.capacity, revision="changed")
    with pytest.raises(FoundryError, match="changed"):
        context.service.benchmark_start(context.user, plan["id"], "changed-revision")
    assert context.dispatcher.calls == 0


def test_tampered_plan_evidence_denied(context):
    plan = context.service.benchmark_plan(context.user, workload())
    context.store.update(
        plan["id"],
        {
            **plan,
            "policy_checks": {**plan["policy_checks"], "audit": False},
        },
    )
    with pytest.raises(FoundryError, match="identity"):
        context.service.benchmark_start(context.user, plan["id"], "tampered-plan")
    assert context.dispatcher.calls == 0


def test_no_authority_denies(context):
    context.service.planning_authority = None
    with pytest.raises(FoundryError, match="planning authority"):
        context.service.benchmark_plan(context.user, workload())


def test_stale_or_missing_snapshot_denies(context):
    context.capacity = replace(context.capacity, observed_at=context.now - 20)
    with pytest.raises(FoundryError, match="stale"):
        context.service.benchmark_plan(context.user, workload())
    context.capacity = None
    with pytest.raises(FoundryError):
        context.service.benchmark_plan(context.user, workload())


def test_fresh_evidence_acquired_after_provider_latency_is_not_future_dated(context):
    def capacity_provider():
        context.now += 2
        return replace(context.capacity, observed_at=context.now)

    def policy_provider(_project_id):
        context.now += 3
        return replace(context.project_policy, observed_at=context.now)

    authority = PlanningAuthority(
        capacity_provider, policy_provider, clock=lambda: context.now, freshness_seconds=5
    )
    attestation = authority.attest("project", workload())
    assert attestation["expires_at"] == context.now + authority.ttl_seconds
    assert all(attestation["policy_checks"].values())


def test_recheck_expiry_after_evidence_fetch_denies_even_if_snapshots_match(context):
    def capacity_provider():
        context.now += 2
        return context.capacity

    authority = PlanningAuthority(
        capacity_provider,
        lambda _project_id: context.project_policy,
        clock=lambda: context.now,
        freshness_seconds=300,
    )
    # Initial attestation at 1002; the capacity/policy values remain unchanged.
    original = authority.attest("project", workload())
    deadline = original["expires_at"]
    context.now = deadline - 1
    with pytest.raises(PlanningError, match="expired") as denied:
        authority.recheck("project", workload(), original)
    assert denied.value.category == "evidence"
    assert context.now > deadline


@pytest.mark.parametrize(
    "models",
    [
        "prefix-registered-model-suffix",
        ["registered-model"],
        {"registered-model": True},
        frozenset({"registered-model", 42}),
    ],
)
def test_malformed_provider_model_grants_fail_closed(context, models):
    context.project_policy = replace(context.project_policy, provider_models={"one": models})
    with pytest.raises(FoundryError) as denied:
        context.service.benchmark_plan(context.user, workload())
    assert denied.value.category == "evidence"
    assert context.dispatcher.calls == 0


def test_provider_model_registry_keys_must_be_strings(context):
    context.project_policy = replace(
        context.project_policy,
        provider_models={"one": frozenset({"registered-model"}), 1: frozenset()},
    )
    with pytest.raises(PlanningError) as denied:
        context.service.planning_authority.attest("project", workload())
    assert denied.value.category == "evidence"
