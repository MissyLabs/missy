from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from missy.config.settings import NomadConfig


def make_bundle(
    path: Path,
    *,
    common_name: str = "missy",
    expired: bool = False,
    mismatched_key: bool = False,
) -> Path:
    path.mkdir(mode=0o700)
    now = datetime.now(tz=UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test Nomad CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=2))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    cert_key = ec.generate_private_key(ec.SECP256R1()) if mismatched_key else key
    leaf_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    not_after = now - timedelta(hours=1) if expired else now + timedelta(days=30)
    cert = (
        x509.CertificateBuilder()
        .subject_name(leaf_name)
        .issuer_name(ca_name)
        .public_key(cert_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(not_after)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    files = {
        "ca.pem": ca.public_bytes(serialization.Encoding.PEM),
        "missy.pem": cert.public_bytes(serialization.Encoding.PEM),
        "missy-key.pem": key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
        "missy.token": b"test-token-value\n",
    }
    for name, content in files.items():
        target = path / name
        target.write_bytes(content)
        os.chmod(target, 0o600)
    return path


@pytest.fixture
def nomad_bundle(tmp_path: Path) -> Path:
    return make_bundle(tmp_path / "bundle")


@pytest.fixture
def nomad_config(nomad_bundle: Path, tmp_path: Path) -> NomadConfig:
    image = "alpine@sha256:" + "a" * 64
    return NomadConfig(
        enabled=True,
        address="https://nomad.example.test",
        bundle_dir=str(nomad_bundle),
        state_dir=str(tmp_path / "state"),
        binary="/bin/echo",
        allowed_namespaces=["testing"],
        allowed_node_pools=["staging"],
        allowed_datacenters=["dc1"],
        default_namespace="testing",
        default_node_pool="staging",
        default_datacenter="dc1",
        approved_registries=["docker.io", "registry.example.test"],
        allowed_job_commands=["/bin/sh", "/bin/check-repository"],
        approved_artifact_prefixes=["artifact://missy/"],
        approved_secret_reference_prefixes=["nomad-var://missy/"],
        workload_templates={
            "repository-check": {
                "description": "Test an immutable repository artifact.",
                "request": {
                    "purpose": "repository check",
                    "image": image,
                    "command": "/bin/check-repository",
                    "cpu_mhz": 500,
                    "memory_mb": 512,
                    "disk_mb": 256,
                    "max_run_seconds": 600,
                    "expected_output": "benchmark complete",
                },
                "allowed_parameters": ["repository", "mode"],
                "required_parameters": ["repository"],
            }
        },
    )


@pytest.fixture
def pinned_image() -> str:
    return "alpine@sha256:" + "a" * 64


def healthy_node(
    *,
    node_id: str = "node-1",
    name: str = "worker-1",
    pool: str = "staging",
    dc: str = "dc1",
    cpu: int = 8000,
    memory: int = 8192,
    disk: int = 100_000,
    allocated_cpu: int = 1000,
    allocated_memory: int = 1024,
    allocated_disk: int = 1000,
) -> dict:
    return {
        "ID": node_id,
        "Name": name,
        "NodePool": pool,
        "Datacenter": dc,
        "Status": "ready",
        "SchedulingEligibility": "eligible",
        "Drain": False,
        "Attributes": {"cpu.arch": "amd64"},
        "Drivers": {"docker": {"Healthy": True}},
        "NodeResources": {
            "Cpu": {"CpuShares": cpu},
            "Memory": {"MemoryMB": memory},
            "Disk": {"DiskMB": disk},
        },
        "ReservedResources": {
            "Cpu": {"CpuShares": 100},
            "Memory": {"MemoryMB": 128},
            "Disk": {"DiskMB": 100},
        },
        "AllocatedResources": {
            "Cpu": {"CpuShares": allocated_cpu},
            "Memory": {"MemoryMB": allocated_memory},
            "Disk": {"DiskMB": allocated_disk},
        },
        "HostStats": {
            "Memory": {"Available": 4_000_000_000},
            "DiskStats": [{"Mountpoint": "/", "Available": 50_000_000_000}],
        },
    }
