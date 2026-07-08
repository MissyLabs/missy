"""Operator diagnostics view models for the Web TUI and API."""

from __future__ import annotations

import contextlib
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

from missy.api.audit_browser import redact_audit_value
from missy.policy.engine import get_policy_engine

if TYPE_CHECKING:
    from missy.agent.runtime import AgentRuntime
    from missy.api.server import ApiConfig
    from missy.memory.sqlite_store import SQLiteMemoryStore
    from missy.providers.registry import ProviderRegistry
    from missy.tools.registry import ToolRegistry


def build_diagnostics(
    *,
    api_config: ApiConfig,
    session_count: int,
    runtime: AgentRuntime | None = None,
    memory_store: SQLiteMemoryStore | None = None,
    provider_registry: ProviderRegistry | None = None,
    tool_registry: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Build a redacted diagnostics report for local operators."""
    sections = [
        _web_section(api_config),
        _provider_section(provider_registry, runtime),
        _tool_section(tool_registry),
        _memory_section(memory_store),
        _policy_section(),
        _scheduler_section(runtime),
        _runtime_section(runtime, session_count),
    ]
    counts = {"ok": 0, "warn": 0, "error": 0}
    for section in sections:
        counts[section["status"]] += 1
    overall = "error" if counts["error"] else "warn" if counts["warn"] else "ok"
    return {
        "overall": overall,
        "counts": counts,
        "sections": sections,
    }


def _web_section(api_config: ApiConfig) -> dict[str, Any]:
    checks = [
        _check(
            "Bind address",
            "ok" if api_config.host in {"127.0.0.1", "::1", "localhost"} else "warn",
            "loopback" if api_config.host in {"127.0.0.1", "::1", "localhost"} else "non-loopback",
        ),
        _check(
            "API key",
            "ok" if bool(api_config.api_key) else "error",
            "configured" if api_config.api_key else "missing",
        ),
        _check(
            "Browser UI",
            "ok" if api_config.web_ui_enabled else "warn",
            "enabled" if api_config.web_ui_enabled else "disabled",
        ),
        _check("Rate limit", "ok", f"{api_config.rate_limit_rpm} rpm"),
    ]
    return _section("web", "Web entrypoint", checks)


def _provider_section(
    provider_registry: ProviderRegistry | None,
    runtime: AgentRuntime | None,
) -> dict[str, Any]:
    if provider_registry is None:
        return _section(
            "providers",
            "Providers",
            [_check("Registry", "warn", "not attached to API server")],
        )

    checks = []
    names: list[str] = []
    default_name = ""
    with contextlib.suppress(Exception):
        names = list(provider_registry.list_providers())
    with contextlib.suppress(Exception):
        default_name = provider_registry.get_default_name() or ""
    for name in names:
        status = "warn"
        summary = "offline"
        with contextlib.suppress(Exception):
            provider = provider_registry.get(name)
            if provider is not None and provider.is_available():
                status = "ok"
                summary = "available"
        if name == default_name:
            summary += " / default"
        checks.append(_check(name, status, summary))
    if not checks:
        checks.append(_check("Registered providers", "warn", "none"))
    if runtime is not None:
        checks.append(
            _check("Runtime provider", "ok", str(getattr(runtime.config, "provider", "")))
        )
    return _section("providers", "Providers", checks)


def _tool_section(tool_registry: ToolRegistry | None) -> dict[str, Any]:
    if tool_registry is None:
        return _section(
            "tools", "Tools", [_check("Registry", "warn", "not attached to API server")]
        )

    names: list[str] = []
    with contextlib.suppress(Exception):
        names = list(tool_registry.list_tools())

    elevated = {"network": 0, "filesystem_read": 0, "filesystem_write": 0, "shell": 0}
    checks = []
    for name in names:
        tool = None
        with contextlib.suppress(Exception):
            tool = tool_registry.get(name)
        perms = getattr(tool, "permissions", None)
        flags = [flag for flag in elevated if bool(getattr(perms, flag, False))]
        for flag in flags:
            elevated[flag] += 1
        summary = ", ".join(flags) if flags else "no elevated permissions"
        checks.append(_check(name, "warn" if flags else "ok", summary))

    if not checks:
        checks.append(_check("Registered tools", "warn", "none"))
    checks.insert(0, _check("Tool count", "ok" if names else "warn", str(len(names))))
    checks.insert(
        1, _check("Elevated permissions", "warn" if any(elevated.values()) else "ok", elevated)
    )
    return _section("tools", "Tools", checks)


def _memory_section(memory_store: SQLiteMemoryStore | None) -> dict[str, Any]:
    if memory_store is None:
        return _section("memory", "Memory", [_check("Store", "warn", "not attached to API server")])

    checks = [_check("Store", "ok", type(memory_store).__name__)]
    try:
        recent = memory_store.get_recent_turns(limit=1)
        checks.append(_check("Recent turns", "ok", "present" if recent else "empty"))
    except Exception as exc:
        checks.append(_check("Recent turns", "error", _safe_error(exc)))
    with contextlib.suppress(Exception):
        sessions = memory_store.list_sessions(limit=1)
        checks.append(_check("Session index", "ok", "present" if sessions else "empty"))
    return _section("memory", "Memory", checks)


def _policy_section() -> dict[str, Any]:
    try:
        engine = get_policy_engine()
    except RuntimeError:
        return _section("policy", "Policy", [_check("Policy engine", "error", "not initialized")])

    checks = [_check("Policy engine", "ok", "initialized")]
    network = getattr(engine, "network", None)
    network_policy = getattr(network, "_policy", None)
    if network_policy is not None:
        checks.append(
            _check(
                "Network default deny",
                "ok" if bool(getattr(network_policy, "default_deny", True)) else "warn",
                bool(getattr(network_policy, "default_deny", True)),
            )
        )
        checks.append(
            _check(
                "Network allowlists",
                "ok",
                {
                    "hosts": len(getattr(network_policy, "allowed_hosts", []) or []),
                    "domains": len(getattr(network_policy, "allowed_domains", []) or []),
                    "cidrs": len(getattr(network_policy, "allowed_cidrs", []) or []),
                    "provider_hosts": len(
                        getattr(network_policy, "provider_allowed_hosts", []) or []
                    ),
                    "tool_hosts": len(getattr(network_policy, "tool_allowed_hosts", []) or []),
                    "discord_hosts": len(
                        getattr(network_policy, "discord_allowed_hosts", []) or []
                    ),
                },
            )
        )

    shell = getattr(engine, "shell", None)
    shell_policy = getattr(shell, "_policy", None)
    if shell_policy is not None:
        enabled = bool(getattr(shell_policy, "enabled", False))
        allowed = len(getattr(shell_policy, "allowed_commands", []) or [])
        checks.append(
            _check("Shell", "warn" if enabled else "ok", {"enabled": enabled, "allowed": allowed})
        )
    return _section("policy", "Policy", checks)


def _scheduler_section(runtime: AgentRuntime | None) -> dict[str, Any]:
    scheduler = getattr(runtime, "_scheduler", None) if runtime is not None else None
    if scheduler is None:
        return _section("scheduler", "Scheduler", [_check("Scheduler", "warn", "not attached")])
    checks = [_check("Scheduler", "ok", type(scheduler).__name__)]
    with contextlib.suppress(Exception):
        jobs = scheduler.list_jobs()
        checks.append(_check("Jobs", "ok", len(jobs)))
    return _section("scheduler", "Scheduler", checks)


def _runtime_section(runtime: AgentRuntime | None, session_count: int) -> dict[str, Any]:
    checks = [_check("API sessions", "ok", session_count)]
    if runtime is None:
        checks.append(_check("Agent runtime", "warn", "not attached to API server"))
    else:
        checks.append(_check("Agent runtime", "ok", type(runtime).__name__))
        checks.append(
            _check("Capability mode", "ok", str(getattr(runtime.config, "capability_mode", "")))
        )
    return _section("runtime", "Runtime", checks)


def _section(key: str, label: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    status = "ok"
    if any(check["status"] == "error" for check in checks):
        status = "error"
    elif any(check["status"] == "warn" for check in checks):
        status = "warn"
    return {"key": key, "label": label, "status": status, "checks": checks}


def _check(name: str, status: str, summary: Any) -> dict[str, Any]:
    if is_dataclass(summary):
        summary = asdict(summary)
    return {
        "name": str(name),
        "status": status if status in {"ok", "warn", "error"} else "warn",
        "summary": redact_audit_value(summary),
    }


def _safe_error(exc: Exception) -> str:
    return redact_audit_value(f"{type(exc).__name__}: {exc}")
