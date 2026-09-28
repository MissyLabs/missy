# Foundry planning boundary

`FoundryService` requires an operator-injected `PlanningAuthority` to issue a
plan. Without one, the planning endpoint fails closed. Constructor interface:

```python
PlanningAuthority(
    capacity_provider,  # () -> CapacitySnapshot(revision, observed_at, tuple[PoolCapacity, ...])
    policy_provider,    # (project_id) -> ProjectPolicySnapshot(...)
    clock=time.time,
    ttl_seconds=30,
    freshness_seconds=15,
)
FoundryService(..., planning_authority=authority)
```

Providers must be independently trusted, project scoped, read-only and fresh;
neither repository content nor HTTP parameters supply approval booleans. The
policy snapshot identifies approved repositories, pinned images and model keys,
CPU/memory/disk ceilings, available **CPU MHz-seconds** budget, allowed egress
policies, and an audit sink identity. Estimated budget pessimistically multiplies
CPU by timeout, warmups plus repetitions, parallelism, attempts and provider
count. This budget check is not a spend reservation and is not a monetary
estimate. The provider is responsible for accurate current remaining budget and
atomic accounting at dispatch. Absent/revoked policy fails closed.

Seven client-required checks (`repository`, `image`, `providers`, `budget`,
`quota`, `egress`, `audit`) are computed from these trusted facts. The plan
selects **staging only**, requiring all three staging available-capacity values
to fit the resource envelope. Capacity from other eligible pools is returned
as `placement.capacity_budget` for advisory right-sizing only; it cannot make
an insufficient staging pool eligible. Unknown, stale, or invalid capacity
fails planning. `policy_snapshot_sha256` and `capacity_snapshot_sha256` hash
the snapshots; `expires_at` is bound into the plan identity alongside project
and exact workload. The full attestation is persisted with the plan. Start
rechecks TTL and authoritative snapshots inside the submission lock before
reservation or dispatch; changes require a new plan. Repeated idempotent
requests for already-submitted runs return their prior record, not a new job.

This does **not** contact Nomad, reserve resources, prove a scheduler will
honor placement, verify audit delivery, or authorize deployment. The
downstream scheduler must separately enforce staging-only placement,
effective job configuration, quota, egress and audit obligations. A fresh
capacity snapshot and recheck narrow the race; only scheduler-enforced
limits and transactional budget reservations can eliminate it. Existing
single-process demo dispatch is not a production execution path.
