"""Evidence-based Nomad placement recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from missy.config.settings import NomadConfig
from missy.nomad.errors import NomadValidationError
from missy.nomad.models import NomadJobRequest


def _dig_int(value: dict[str, Any], *paths: tuple[str, ...]) -> int | None:
    for path in paths:
        current: Any = value
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, (int, float)):
            return int(current)
    return None


def _driver_healthy(node: dict[str, Any], driver: str) -> bool | None:
    drivers = node.get("Drivers")
    if not isinstance(drivers, dict) or driver not in drivers:
        return None
    detail = drivers[driver]
    if not isinstance(detail, dict):
        return None
    if "Healthy" in detail:
        return bool(detail["Healthy"])
    if "Detected" in detail:
        return bool(detail["Detected"])
    return None


@dataclass(frozen=True)
class PlacementCandidate:
    node_id: str
    name: str
    node_pool: str
    datacenter: str
    architecture: str
    status: str
    eligible: bool
    drained: bool
    docker_healthy: bool | None
    available_cpu_mhz: int | None
    available_memory_mb: int | None
    available_disk_mb: int | None
    observed_memory_available_bytes: int | None
    observed_disk_available_bytes: int | None
    protected: bool
    fits: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_pool": self.node_pool,
            "datacenter": self.datacenter,
            "architecture": self.architecture,
            "status": self.status,
            "eligible": self.eligible,
            "drained": self.drained,
            "docker_healthy": self.docker_healthy,
            "scheduler_headroom": {
                "cpu_mhz": self.available_cpu_mhz,
                "memory_mb": self.available_memory_mb,
                "disk_mb": self.available_disk_mb,
            },
            "observed_host_memory_available_bytes": self.observed_memory_available_bytes,
            "observed_host_disk_available_bytes": self.observed_disk_available_bytes,
            "protected": self.protected,
            "fits": self.fits,
            "reasons": list(self.reasons),
        }


def candidate_from_node(
    node: dict[str, Any], request: NomadJobRequest, config: NomadConfig
) -> PlacementCandidate:
    resources = node.get("NodeResources") or node.get("Resources") or {}
    reserved = node.get("ReservedResources") or node.get("Reserved") or {}
    allocated = node.get("AllocatedResources") or {}

    total_cpu = _dig_int(
        resources,
        ("Cpu", "CpuShares"),
        ("CPU", "CPUShares"),
        ("CPU",),
    )
    reserved_cpu = _dig_int(reserved, ("Cpu", "CpuShares"), ("CPU",), ("CPU", "CPUShares"))
    allocated_cpu = _dig_int(allocated, ("Cpu", "CpuShares"), ("CPU",), ("CPU", "CPUShares"))
    total_memory = _dig_int(resources, ("Memory", "MemoryMB"), ("MemoryMB",))
    reserved_memory = _dig_int(reserved, ("Memory", "MemoryMB"), ("MemoryMB",))
    allocated_memory = _dig_int(allocated, ("Memory", "MemoryMB"), ("MemoryMB",))
    total_disk = _dig_int(resources, ("Disk", "DiskMB"), ("DiskMB",))
    reserved_disk = _dig_int(reserved, ("Disk", "DiskMB"), ("DiskMB",))
    allocated_disk = _dig_int(allocated, ("Disk", "DiskMB"), ("DiskMB",))

    def available(total: int | None, reserved_value: int | None, used: int | None) -> int | None:
        if total is None:
            return None
        return max(0, total - (reserved_value or 0) - (used or 0))

    available_cpu = available(total_cpu, reserved_cpu, allocated_cpu)
    available_memory = available(total_memory, reserved_memory, allocated_memory)
    available_disk = available(total_disk, reserved_disk, allocated_disk)
    attributes = node.get("Attributes") if isinstance(node.get("Attributes"), dict) else {}
    architecture = str(attributes.get("cpu.arch") or attributes.get("kernel.arch") or "")
    node_id = str(node.get("ID") or node.get("NodeID") or "")
    name = str(node.get("Name") or node_id)
    pool = str(node.get("NodePool") or "default")
    dc = str(node.get("Datacenter") or "")
    status = str(node.get("Status") or "unknown").lower()
    eligibility = str(node.get("SchedulingEligibility") or "eligible").lower()
    eligible = eligibility == "eligible"
    drained = bool(node.get("Drain")) or eligibility == "ineligible"
    docker_healthy = _driver_healthy(node, "docker")
    observed_memory = _dig_int(
        node,
        ("HostStats", "Memory", "Available"),
        ("HostStats", "Memory", "AvailableBytes"),
    )
    disk_stats = (node.get("HostStats") or {}).get("DiskStats")
    observed_disk = None
    if isinstance(disk_stats, list):
        available_values = [
            int(item["Available"])
            for item in disk_stats
            if isinstance(item, dict) and isinstance(item.get("Available"), (int, float))
        ]
        if available_values:
            observed_disk = max(available_values)
    protected = name.casefold() in {item.casefold() for item in config.protected_node_names}
    reasons: list[str] = []
    if status != "ready":
        reasons.append(f"node status is {status!r}")
    if not eligible or drained:
        reasons.append("node is drained or scheduling-ineligible")
    if pool != request.node_pool:
        reasons.append(f"node pool {pool!r} does not match {request.node_pool!r}")
    if dc != request.datacenter:
        reasons.append(f"datacenter {dc!r} does not match {request.datacenter!r}")
    if request.architecture and architecture != request.architecture:
        reasons.append(f"architecture {architecture!r} does not match {request.architecture!r}")
    for attribute, expected in request.required_node_attributes.items():
        actual = str(attributes.get(attribute) or "")
        if actual != expected:
            reasons.append(
                f"required node attribute {attribute!r} is {actual!r}, expected {expected!r}"
            )
    if request.stateful and isinstance(request.persistence_plan, dict):
        required_node = str(request.persistence_plan.get("node_id") or "")
        if required_node and node_id != required_node:
            reasons.append(f"persistent volume locality requires node {required_node!r}")
    if docker_healthy is not True:
        reasons.append("Docker driver health is unavailable or unhealthy")
    # A service rollout can temporarily run one additional allocation while
    # the replacement proves healthy. Batch jobs do not use rolling updates.
    placement_count = request.count + (1 if request.job_type == "service" else 0)
    demand_cpu = request.cpu_mhz * placement_count
    demand_memory = request.memory_mb * placement_count
    demand_disk = request.disk_mb * placement_count
    for label, available_value, demand in (
        ("CPU", available_cpu, demand_cpu),
        ("memory", available_memory, demand_memory),
        ("disk", available_disk, demand_disk),
    ):
        if available_value is None:
            reasons.append(f"{label} scheduler headroom is unavailable")
        elif available_value < demand:
            reasons.append(
                f"insufficient {label} scheduler headroom ({available_value} < {demand})"
            )
    if observed_memory is not None and observed_memory < demand_memory * 1024 * 1024:
        reasons.append("insufficient observed host memory headroom")
    if observed_disk is not None and observed_disk < demand_disk * 1024 * 1024:
        reasons.append("insufficient observed host disk headroom")
    fits = not reasons
    return PlacementCandidate(
        node_id=node_id,
        name=name,
        node_pool=pool,
        datacenter=dc,
        architecture=architecture,
        status=status,
        eligible=eligible,
        drained=drained,
        docker_healthy=docker_healthy,
        available_cpu_mhz=available_cpu,
        available_memory_mb=available_memory,
        available_disk_mb=available_disk,
        observed_memory_available_bytes=observed_memory,
        observed_disk_available_bytes=observed_disk,
        protected=protected,
        fits=fits,
        reasons=tuple(reasons),
    )


def recommend_placement(
    nodes: list[dict[str, Any]], request: NomadJobRequest, config: NomadConfig
) -> dict[str, Any]:
    """Return an explainable recommendation without mutating the cluster."""
    candidates = [candidate_from_node(node, request, config) for node in nodes]
    fits = [candidate for candidate in candidates if candidate.fits]
    non_protected = [candidate for candidate in fits if not candidate.protected]
    preferred = non_protected or fits
    preferred.sort(
        key=lambda candidate: (
            candidate.available_memory_mb or -1,
            candidate.available_cpu_mhz or -1,
            candidate.available_disk_mb or -1,
        ),
        reverse=True,
    )
    recommendation = preferred[0] if preferred else None
    if recommendation is None:
        raise NomadValidationError(
            "No observed node safely fits the workload; inspect candidate reasons and capacity."
        )
    warnings: list[str] = []
    if recommendation.protected:
        warnings.append(
            "Only a protected node fit the request; preserve control-plane headroom and review before submission."
        )
    if (
        recommendation.observed_memory_available_bytes is None
        or recommendation.observed_disk_available_bytes is None
    ):
        warnings.append(
            "Observed host memory or disk pressure was unavailable; scheduler headroom does not prove host headroom."
        )
    return {
        "observed_at": datetime.now(tz=UTC).isoformat(),
        "requested_scope": {
            "namespace": request.namespace,
            "node_pool": request.node_pool,
            "datacenter": request.datacenter,
            "architecture": request.architecture or None,
        },
        "demand": {
            "cpu_mhz": request.cpu_mhz
            * (request.count + (1 if request.job_type == "service" else 0)),
            "memory_mb": request.memory_mb
            * (request.count + (1 if request.job_type == "service" else 0)),
            "disk_mb": request.disk_mb
            * (request.count + (1 if request.job_type == "service" else 0)),
            "count": request.count,
            "temporary_rollout_allocations": 1 if request.job_type == "service" else 0,
            "rollout_overlap_note": (
                "service updates may temporarily require an additional allocation"
                if request.job_type == "service"
                else "no rollout overlap requested"
            ),
        },
        "recommended_node": recommendation.to_dict(),
        "warnings": warnings,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
