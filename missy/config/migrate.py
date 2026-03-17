"""Config file migration for Missy.

Detects old-format config files (missing ``config_version`` or version < current),
backs them up, and rewrites them to use preset-based network policy and a version
stamp.  Designed to run on every startup with zero cost when already migrated.

Migration is **idempotent**: running it on an already-migrated config is a no-op.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CURRENT_CONFIG_VERSION = 2


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def needs_migration(config_path: str) -> bool:
    """Return ``True`` if *config_path* exists and needs migration.

    A config needs migration when it has no ``config_version`` field or
    its version is less than :data:`CURRENT_CONFIG_VERSION`.
    """
    path = Path(config_path).expanduser()
    if not path.exists() or not path.is_file():
        return False
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    version = data.get("config_version", 0)
    try:
        return int(version) < CURRENT_CONFIG_VERSION
    except (ValueError, TypeError):
        return True


def detect_presets(
    network_data: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Detect which presets are covered by the explicit host/domain/CIDR entries.

    A preset is detected if **all** of its ``hosts`` entries are present in
    ``allowed_hosts``.  Domains and CIDRs from detected presets are also
    removed from the remaining lists, but their presence is not required for
    detection.

    Args:
        network_data: The raw ``network:`` dict from the YAML config.

    Returns:
        A 4-tuple of ``(detected_presets, remaining_hosts, remaining_domains,
        remaining_cidrs)``.
    """
    from missy.policy.presets import PRESETS

    hosts = set(network_data.get("allowed_hosts", []) or [])

    # Existing presets already in the config (partial migration / manual edit)
    existing_presets = list(network_data.get("presets", []) or [])

    detected: list[str] = list(existing_presets)
    covered_hosts: set[str] = set()
    covered_domains: set[str] = set()
    covered_cidrs: set[str] = set()

    for name, preset in PRESETS.items():
        if name in detected:
            # Already listed — still collect its entries as covered
            covered_hosts.update(preset.get("hosts", []))
            covered_domains.update(preset.get("domains", []))
            covered_cidrs.update(preset.get("cidrs", []))
            continue

        preset_hosts = set(preset.get("hosts", []))
        if not preset_hosts:
            continue

        # Detect if ALL hosts for this preset are present
        if preset_hosts.issubset(hosts):
            detected.append(name)
            covered_hosts.update(preset_hosts)
            covered_domains.update(preset.get("domains", []))
            covered_cidrs.update(preset.get("cidrs", []))

    remaining_hosts = [
        h for h in (network_data.get("allowed_hosts", []) or []) if h not in covered_hosts
    ]
    remaining_domains = [
        d for d in (network_data.get("allowed_domains", []) or []) if d not in covered_domains
    ]
    remaining_cidrs = [
        c for c in (network_data.get("allowed_cidrs", []) or []) if c not in covered_cidrs
    ]

    return detected, remaining_hosts, remaining_domains, remaining_cidrs


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def migrate_config(
    config_path: str,
    backup_dir: str | None = None,
) -> dict[str, Any]:
    """Migrate a config file to the current version.

    Steps:

    1. Check if migration is needed (version check).
    2. Back up the existing config.
    3. Detect presets from manual host/domain entries.
    4. Rewrite the YAML with presets and ``config_version``.
    5. Atomically replace the config file.

    Args:
        config_path: Path to ``config.yaml``.
        backup_dir: Optional override for the backup directory.

    Returns:
        A summary dict with keys ``migrated``, ``backup_path``,
        ``presets_detected``, and ``version``.
    """
    result: dict[str, Any] = {
        "migrated": False,
        "backup_path": None,
        "presets_detected": [],
        "version": CURRENT_CONFIG_VERSION,
    }

    if not needs_migration(config_path):
        return result

    path = Path(config_path).expanduser()

    # Back up before modifying
    try:
        from missy.config.plan import backup_config

        bkp = backup_config(path, backup_dir)
        result["backup_path"] = str(bkp)
        logger.info("Config backed up to %s before migration", bkp)
    except Exception as exc:
        logger.warning("Could not back up config before migration: %s", exc)

    # Load raw YAML
    raw_text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)
    if not isinstance(data, dict):
        data = {}

    # Migrate network section
    network = data.get("network", {}) or {}
    detected, remaining_hosts, remaining_domains, remaining_cidrs = detect_presets(network)

    if detected:
        network["presets"] = detected
    network["allowed_hosts"] = remaining_hosts
    network["allowed_domains"] = remaining_domains
    network["allowed_cidrs"] = remaining_cidrs
    data["network"] = network

    # Stamp version
    data["config_version"] = CURRENT_CONFIG_VERSION

    result["migrated"] = True
    result["presets_detected"] = detected

    # Write atomically
    _atomic_write_yaml(path, data)

    logger.info(
        "Config migrated to version %d: detected presets=%s",
        CURRENT_CONFIG_VERSION,
        detected,
    )

    return result


def _atomic_write_yaml(path: Path, data: dict) -> None:
    """Write *data* as YAML to *path* atomically via a temp file."""
    content = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".config_migrate_")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, str(path))
    except Exception:
        import contextlib

        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
