"""Built-in skill: basic health checks on Missy subsystems.

Verifies the existence and minimal validity of key runtime artefacts
(config file, memory database, audit log) and whether at least one
provider is configured.  Results are returned as a structured pass/warn/fail
report with no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from missy.skills.base import BaseSkill, SkillPermissions, SkillResult

# Default paths — match the layout documented in CLAUDE.md.
_CONFIG_PATH = Path("~/.missy/config.yaml")
_MEMORY_DB_PATH = Path("~/.missy/memory.db")
_AUDIT_LOG_PATH = Path("~/.missy/audit.jsonl")


@dataclass
class _Check:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str


def _check_config(path: Path) -> _Check:
    if not path.exists():
        return _Check("config_file", "FAIL", f"Not found: {path}")
    if path.stat().st_size == 0:
        return _Check("config_file", "WARN", f"File is empty: {path}")
    return _Check("config_file", "PASS", str(path))


def _check_memory_db(path: Path) -> _Check:
    if not path.exists():
        return _Check("memory_db", "WARN", f"Not found (will be created on first use): {path}")
    return _Check("memory_db", "PASS", str(path))


def _check_audit_log(path: Path) -> _Check:
    if not path.exists():
        return _Check("audit_log", "WARN", f"Not found (will be created on first write): {path}")
    return _Check("audit_log", "PASS", str(path))


def _check_providers(config_path: Path) -> _Check:
    """Check that at least one provider section appears in the config file.

    A very lightweight textual scan avoids pulling in the full config
    stack, keeping the health check self-contained and fast.

    Args:
        config_path: Expanded path to ``config.yaml``.

    Returns:
        A :class:`_Check` describing whether providers look configured.
    """
    if not config_path.exists():
        return _Check(
            "providers_configured", "FAIL", "Config file missing; cannot check providers."
        )

    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        return _Check("providers_configured", "FAIL", f"Could not read config: {exc}")

    if "providers:" not in text:
        return _Check("providers_configured", "WARN", "No 'providers:' section found in config.")

    # Rough check: look for at least one provider sub-key with an api_key or
    # a model value that is not null/empty.
    has_model = any(
        line.strip().startswith("model:") and "null" not in line and line.strip() != "model:"
        for line in text.splitlines()
    )
    if not has_model:
        return _Check(
            "providers_configured",
            "WARN",
            "Providers section present but no model appears to be configured.",
        )

    return _Check("providers_configured", "PASS", "At least one provider model is configured.")


class HealthCheckSkill(BaseSkill):
    """Runs basic health checks on Missy subsystems and reports their status."""

    name = "health_check"
    description = "Run basic health checks on Missy subsystems."
    version = "1.0.0"
    permissions = SkillPermissions(filesystem_read=True)

    def execute(self, **kwargs: Any) -> SkillResult:
        """Perform all subsystem checks and return a formatted report.

        Returns:
            :class:`~missy.skills.base.SkillResult` with ``success=True``
            when all checks pass or warn, ``success=False`` if any check
            fails.  The ``output`` field contains a human-readable table.
        """
        config_path = _CONFIG_PATH.expanduser()
        checks: list[_Check] = [
            _check_config(config_path),
            _check_memory_db(_MEMORY_DB_PATH.expanduser()),
            _check_audit_log(_AUDIT_LOG_PATH.expanduser()),
            _check_providers(config_path),
        ]

        lines: list[str] = ["Missy Health Check", "=" * 40]
        any_fail = False
        for check in checks:
            lines.append(f"[{check.status:<4}] {check.name}: {check.detail}")
            if check.status == "FAIL":
                any_fail = True

        overall = "FAIL" if any_fail else "PASS"
        lines.append("=" * 40)
        lines.append(f"Overall: {overall}")

        return SkillResult(
            success=not any_fail,
            output="\n".join(lines),
        )
