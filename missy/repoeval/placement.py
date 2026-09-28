"""Read-only, deterministic Nomad node-pool placement planning."""

from collections.abc import Iterable
from dataclasses import dataclass


class PlacementError(ValueError):
    """Requested finite resource envelope cannot be placed safely."""


@dataclass(frozen=True)
class ResourceEnvelope:
    cpu_mhz: int
    memory_mb: int
    disk_mb: int

    def __post_init__(self):
        for name, value, low, high in (
            ("cpu_mhz", self.cpu_mhz, 100, 32000),
            ("memory_mb", self.memory_mb, 128, 131072),
            ("disk_mb", self.disk_mb, 256, 262144),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise PlacementError(f"{name} outside supported finite envelope")


@dataclass(frozen=True)
class PoolCapacity:
    """Read-only snapshot of currently allocatable capacity, never total capacity."""

    name: str
    available_cpu_mhz: int
    available_memory_mb: int
    available_disk_mb: int
    eligible: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or len(self.name) > 128:
            raise PlacementError("invalid pool name")
        for key in ("available_cpu_mhz", "available_memory_mb", "available_disk_mb"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PlacementError(f"invalid read-only capacity: {key}")


@dataclass(frozen=True)
class CapacityBudget:
    """Advisory aggregate available capacity, not a schedulable pool or reservation."""

    available_cpu_mhz: int
    available_memory_mb: int
    available_disk_mb: int


def capacity_budget(pools: Iterable[PoolCapacity]) -> CapacityBudget:
    """Sum read-only available resources across eligible pools for right-sizing."""
    cpu = memory = disk = 0
    names = set()
    for pool in pools:
        if not isinstance(pool, PoolCapacity):
            raise PlacementError("pool capacity must be a PoolCapacity snapshot")
        if pool.name in names:
            raise PlacementError("duplicate pool capacity snapshot")
        names.add(pool.name)
        if pool.eligible:
            cpu += pool.available_cpu_mhz
            memory += pool.available_memory_mb
            disk += pool.available_disk_mb
    return CapacityBudget(cpu, memory, disk)


def select_pool(envelope: ResourceEnvelope, pools: Iterable[PoolCapacity]) -> PoolCapacity:
    """Select staging only; other pools provide capacity information, not targets.

    Capacity snapshots may stale before dispatch; Nomad must revalidate. This
    pure function never reserves resources or changes node/pool configuration.
    """
    staging = None
    for pool in pools:
        if not isinstance(pool, PoolCapacity):
            raise PlacementError("pool capacity must be a PoolCapacity snapshot")
        if pool.name == "staging":
            if staging is not None:
                raise PlacementError("duplicate staging capacity snapshot")
            staging = pool
    if (
        staging is None
        or not staging.eligible
        or any(
            (
                staging.available_cpu_mhz < envelope.cpu_mhz,
                staging.available_memory_mb < envelope.memory_mb,
                staging.available_disk_mb < envelope.disk_mb,
            )
        )
    ):
        raise PlacementError("staging pool unavailable or has insufficient available capacity")
    return staging
