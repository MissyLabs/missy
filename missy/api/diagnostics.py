"""Operator diagnostics view models for the Web TUI and API."""

from __future__ import annotations

import contextlib
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

from missy.api.audit_browser import redact_audit_value
from missy.policy.engine import get_policy_engine
from missy.policy.tool_policy_pipeline import (
    MISSY_DISCORD_TOOLS,
    build_configured_tool_policy_layers,
    collect_tool_policy_groups,
    resolve_tool_policy,
)

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
        _gateway_section(),
        _discord_section(),
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
            remediation="Bind to 127.0.0.1 or protect non-loopback access with a firewall.",
        ),
        _check(
            "API key",
            "ok" if bool(api_config.api_key) else "error",
            "configured" if api_config.api_key else "missing",
            remediation="Set MISSY_API_KEY or ApiConfig.api_key before enabling the Web TUI.",
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
        return _section(
            "policy",
            "Policy",
            [
                _check(
                    "Policy engine",
                    "error",
                    "not initialized",
                    remediation="Initialize policy before starting operator-facing channels.",
                )
            ],
        )

    checks = [_check("Policy engine", "ok", "initialized")]
    network = getattr(engine, "network", None)
    network_policy = getattr(network, "_policy", None)
    if network_policy is not None:
        checks.append(
            _check(
                "Network default deny",
                "ok" if bool(getattr(network_policy, "default_deny", True)) else "warn",
                bool(getattr(network_policy, "default_deny", True)),
                remediation="Enable network.default_deny for local-first operation.",
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
        checks.append(
            _check(
                "Provider network scope",
                "ok" if getattr(network_policy, "provider_allowed_hosts", []) else "warn",
                _host_scope_summary(network_policy, "provider"),
                remediation=(
                    "Use network.provider_allowed_hosts for model APIs instead of broad "
                    "global allowlists."
                ),
            )
        )
        checks.append(
            _check(
                "Tool network scope",
                "ok" if getattr(network_policy, "tool_allowed_hosts", []) else "warn",
                _host_scope_summary(network_policy, "tool"),
                remediation=(
                    "Use network.tool_allowed_hosts for network-capable tools and keep "
                    "tool destinations narrow."
                ),
            )
        )

    shell = getattr(engine, "shell", None)
    shell_policy = getattr(shell, "_policy", None)
    if shell_policy is not None:
        enabled = bool(getattr(shell_policy, "enabled", False))
        allowed = len(getattr(shell_policy, "allowed_commands", []) or [])
        checks.append(
            _check(
                "Shell",
                "warn" if enabled else "ok",
                {"enabled": enabled, "allowed": allowed},
                remediation="Keep shell disabled unless specific commands are required.",
            )
        )
    rest_policy = getattr(engine, "rest_policy", None)
    rules = getattr(rest_policy, "_rules", []) if rest_policy is not None else []
    checks.append(
        _check(
            "REST method/path rules",
            "ok" if rules else "warn",
            {"rules": len(rules)},
            remediation="Add network.rest_policies for high-risk REST APIs such as GitHub or Discord.",
        )
    )
    return _section("policy", "Policy", checks)


def _gateway_section() -> dict[str, Any]:
    try:
        from missy.gateway.client import PolicyHTTPClient
    except Exception as exc:
        return _section(
            "gateway",
            "Gateway",
            [
                _check(
                    "Policy HTTP client",
                    "error",
                    _safe_error(exc),
                    remediation="Install gateway dependencies and route outbound HTTP through it.",
                )
            ],
        )

    checks = [
        _check("Policy HTTP client", "ok", PolicyHTTPClient.__name__),
        _check(
            "Response size cap",
            "ok",
            {"bytes": getattr(PolicyHTTPClient, "DEFAULT_MAX_RESPONSE_BYTES", 0)},
        ),
    ]
    try:
        get_policy_engine()
        checks.append(_check("Policy binding", "ok", "active"))
    except RuntimeError:
        checks.append(
            _check(
                "Policy binding",
                "error",
                "policy engine not initialized",
                remediation="Call init_policy_engine(config) before network-capable runtime startup.",
            )
        )
    return _section("gateway", "Gateway", checks)


def _discord_section() -> dict[str, Any]:
    try:
        engine = get_policy_engine()
    except RuntimeError:
        return _section(
            "discord",
            "Discord",
            [_check("Configuration", "warn", "policy config unavailable")],
        )

    cfg = getattr(engine, "config", None)
    discord_cfg = getattr(cfg, "discord", None)
    network = getattr(cfg, "network", None)
    if discord_cfg is None or not getattr(discord_cfg, "accounts", []):
        return _section(
            "discord",
            "Discord",
            [_check("Configuration", "warn", "no accounts configured")],
        )

    accounts = list(getattr(discord_cfg, "accounts", []) or [])
    checks = [
        _check(
            "Integration",
            "ok" if bool(getattr(discord_cfg, "enabled", False)) else "warn",
            "enabled" if bool(getattr(discord_cfg, "enabled", False)) else "disabled",
            remediation="Set discord.enabled=true when the Discord channel should run.",
        ),
        _check("Accounts", "ok" if accounts else "warn", len(accounts)),
    ]

    for idx, account in enumerate(accounts):
        token_present = False
        with contextlib.suppress(Exception):
            token_present = bool(account.resolve_token())
        checks.append(
            _check(
                f"Account {idx} token",
                "ok" if token_present else "warn",
                "present"
                if token_present
                else f"missing env:{getattr(account, 'token_env_var', '')}",
                remediation="Set the configured Discord token environment variable or vault reference.",
            )
        )
        checks.append(
            _check(
                f"Account {idx} routing",
                "ok",
                {
                    "application_id": bool(getattr(account, "application_id", "")),
                    "dm_policy": str(getattr(account, "dm_policy", "")),
                    "guilds": len(getattr(account, "guild_policies", {}) or {}),
                    "ignore_bots": bool(getattr(account, "ignore_bots", True)),
                },
            )
        )

    network_values = _network_values(network)
    checks.extend(
        [
            _check(
                "REST host discord.com",
                "ok" if "discord.com" in network_values else "warn",
                "allowed" if "discord.com" in network_values else "missing",
                remediation="Add discord.com to network.allowed_domains or discord_allowed_hosts.",
            ),
            _check(
                "Gateway host gateway.discord.gg",
                "ok" if "gateway.discord.gg" in network_values else "warn",
                "allowed" if "gateway.discord.gg" in network_values else "missing",
                remediation=(
                    "Add gateway.discord.gg to network.allowed_domains or discord_allowed_hosts."
                ),
            ),
        ]
    )

    with contextlib.suppress(Exception):
        layers = build_configured_tool_policy_layers(
            capability_mode="discord",
            global_policy=getattr(cfg, "tools", None),
            agent_policy=None,
            sandbox_policy=getattr(getattr(cfg, "sandbox", None), "tools", None),
        )
        groups = collect_tool_policy_groups(getattr(cfg, "tools", None))
        decision = resolve_tool_policy(MISSY_DISCORD_TOOLS, layers, groups=groups)
        voice_tools = {
            "discord_voice_join",
            "discord_voice_leave",
            "discord_voice_say",
            "discord_voice_status",
        }
        checks.append(
            _check(
                "Discord voice tools",
                "ok" if voice_tools.issubset(set(decision.tools)) else "warn",
                {"visible": sorted(voice_tools.intersection(decision.tools))},
                remediation="Review Discord capability-mode tool policy if voice controls are missing.",
            )
        )
        if decision.warnings:
            checks.append(_check("Tool policy warnings", "warn", decision.warnings[:3]))

    return _section("discord", "Discord", checks)


def _scheduler_section(runtime: AgentRuntime | None) -> dict[str, Any]:
    scheduler = getattr(runtime, "_scheduler", None) if runtime is not None else None
    checks = []
    with contextlib.suppress(Exception):
        cfg = getattr(get_policy_engine(), "config", None)
        scheduling = getattr(cfg, "scheduling", None)
        if scheduling is not None:
            checks.append(
                _check(
                    "Scheduling policy",
                    "ok" if bool(getattr(scheduling, "enabled", True)) else "warn",
                    {
                        "enabled": bool(getattr(scheduling, "enabled", True)),
                        "max_jobs": int(getattr(scheduling, "max_jobs", 0) or 0),
                        "active_hours": str(getattr(scheduling, "active_hours", "")),
                    },
                    remediation="Disable scheduling or set max_jobs/active_hours for tighter posture.",
                )
            )
    if scheduler is None:
        checks.append(_check("Scheduler", "warn", "not attached"))
        return _section("scheduler", "Scheduler", checks)
    checks.append(_check("Scheduler", "ok", type(scheduler).__name__))
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


def _check(
    name: str,
    status: str,
    summary: Any,
    *,
    remediation: str | None = None,
) -> dict[str, Any]:
    if is_dataclass(summary):
        summary = asdict(summary)
    result = {
        "name": str(name),
        "status": status if status in {"ok", "warn", "error"} else "warn",
        "summary": redact_audit_value(summary),
    }
    if remediation and result["status"] != "ok":
        result["remediation"] = redact_audit_value(remediation)
    return result


def _safe_error(exc: Exception) -> str:
    return redact_audit_value(f"{type(exc).__name__}: {exc}")


def _host_scope_summary(network_policy: Any, category: str) -> dict[str, Any]:
    values = {
        "global_hosts": len(getattr(network_policy, "allowed_hosts", []) or []),
        "global_domains": len(getattr(network_policy, "allowed_domains", []) or []),
    }
    key = f"{category}_allowed_hosts"
    values["category_hosts"] = len(getattr(network_policy, key, []) or [])
    return values


def _network_values(network: Any) -> set[str]:
    values: set[str] = set()
    for attr in ("allowed_domains", "allowed_hosts", "discord_allowed_hosts"):
        raw = getattr(network, attr, []) if network is not None else []
        if not isinstance(raw, (list, tuple, set)):
            continue
        for item in raw:
            values.add(str(item).lower().rsplit(":", 1)[0].strip("[]"))
    return values
