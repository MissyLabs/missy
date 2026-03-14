"""Configuration loading and schema definitions for Missy.

All policy dataclasses default to a *secure-by-default* posture: network
access is denied, shell execution is disabled, and plugins are disabled.
Callers must explicitly opt in to each capability in the YAML config file.

Example YAML layout::

    network:
      default_deny: true
      allowed_domains:
        - "api.anthropic.com"
      allowed_cidrs: []
      allowed_hosts: []

    filesystem:
      allowed_read_paths:
        - "/home/user/workspace"
      allowed_write_paths:
        - "/home/user/workspace/output"

    shell:
      enabled: false
      allowed_commands: []

    plugins:
      enabled: false
      allowed_plugins: []

    providers:
      anthropic:
        name: anthropic
        model: claude-sonnet-4-6
        timeout: 30

    workspace_path: "/home/user/workspace"
    audit_log_path: "/home/user/.missy/audit.log"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import yaml

from missy.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from missy.channels.discord.config import DiscordConfig
    from missy.security.sandbox import SandboxConfig


# ---------------------------------------------------------------------------
# Policy dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NetworkPolicy:
    """Controls outbound network access.

    Attributes:
        default_deny: When ``True`` (the default) all network requests are
            blocked unless the destination appears in an allow list.
        allowed_cidrs: List of CIDR blocks that are reachable regardless of
            domain/host rules.
        allowed_domains: Fully-qualified domain names (or suffix patterns)
            that are reachable.
        allowed_hosts: Explicit ``host:port`` strings that are reachable.
    """

    default_deny: bool = True
    allowed_cidrs: list[str] = field(default_factory=list)
    allowed_domains: list[str] = field(default_factory=list)
    allowed_hosts: list[str] = field(default_factory=list)
    # Per-category overrides (union with allowed_domains/hosts)
    provider_allowed_hosts: list[str] = field(default_factory=list)
    tool_allowed_hosts: list[str] = field(default_factory=list)
    discord_allowed_hosts: list[str] = field(default_factory=list)


@dataclass
class FilesystemPolicy:
    """Controls read and write access to the local filesystem.

    Attributes:
        allowed_write_paths: Absolute directory paths the agent may write to.
        allowed_read_paths: Absolute directory paths the agent may read from.
    """

    allowed_write_paths: list[str] = field(default_factory=list)
    allowed_read_paths: list[str] = field(default_factory=list)


@dataclass
class ShellPolicy:
    """Controls shell command execution.

    Attributes:
        enabled: When ``False`` (the default) all shell execution is blocked.
        allowed_commands: Whitelist of command names permitted when the shell
            is enabled.  An empty list means no commands are allowed even
            when ``enabled`` is ``True``.
    """

    enabled: bool = False
    allowed_commands: list[str] = field(default_factory=list)


@dataclass
class PluginPolicy:
    """Controls plugin loading and execution.

    Attributes:
        enabled: When ``False`` (the default) no plugins may be loaded.
        allowed_plugins: Whitelist of plugin identifiers that may be loaded
            when ``enabled`` is ``True``.
    """

    enabled: bool = False
    allowed_plugins: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    """Configuration for a single AI provider.

    Attributes:
        name: Logical name for the provider (e.g. ``"anthropic"``).
        model: Model identifier to use for inference.
        api_key: Optional API key.  When ``None`` the provider implementation
            will fall back to the relevant environment variable.
        base_url: Optional base URL override for the provider's HTTP API.
        timeout: Request timeout in seconds.  Defaults to ``30``.
    """

    name: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    enabled: bool = True
    api_keys: list = field(default_factory=list)  # Multiple API keys for rotation
    fast_model: str = ""      # Model for fast/simple tier (e.g. claude-haiku-4-5)
    premium_model: str = ""   # Model for premium/complex tier (e.g. claude-opus-4-6)


# ---------------------------------------------------------------------------
# Scheduling policy
# ---------------------------------------------------------------------------


@dataclass
class SchedulingPolicy:
    """Controls scheduled job execution.

    Attributes:
        enabled: When False, no new jobs may be added or run.
        max_jobs: Maximum number of concurrent scheduled jobs (0 = unlimited).
        active_hours: Optional time window for job execution, e.g. "08:00-22:00".
    """

    enabled: bool = True
    max_jobs: int = 0
    active_hours: str = ""


@dataclass
class HeartbeatConfig:
    """Heartbeat system configuration."""

    enabled: bool = False
    interval_seconds: int = 1800
    workspace: str = "~/workspace"
    active_hours: str = ""  # e.g. "08:00-22:00"


@dataclass
class ObservabilityConfig:
    """OpenTelemetry and logging configuration."""

    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4317"
    otel_protocol: str = "grpc"  # "grpc" | "http/protobuf"
    otel_service_name: str = "missy"
    log_level: str = "warning"


@dataclass
class VaultConfig:
    """Encrypted secrets vault configuration."""

    enabled: bool = False
    vault_dir: str = "~/.missy/secrets"


# ---------------------------------------------------------------------------
# Proactive config
# ---------------------------------------------------------------------------


@dataclass
class ProactiveTriggerConfig:
    """Configuration for a single proactive trigger loaded from YAML.

    Attributes:
        name: Unique identifier for the trigger.
        trigger_type: One of ``"file_change"``, ``"disk_threshold"``,
            ``"load_threshold"``, or ``"schedule"``.
        enabled: When ``False`` the trigger is skipped entirely.
        requires_confirmation: When ``True``, an :class:`ApprovalGate` must
            confirm the trigger before the agent callback is invoked.
        prompt_template: Template for the synthetic prompt.  Supports
            ``{trigger_name}``, ``{trigger_type}``, and ``{timestamp}``.
        watch_path: Directory or file path to watch (``file_change`` only).
        watch_patterns: Glob patterns, e.g. ``["*.log"]``.
        watch_recursive: Whether to recurse into subdirectories.
        disk_path: Path to evaluate for ``disk_threshold`` triggers.
        disk_threshold_pct: Fire when disk usage exceeds this percentage.
        load_threshold: Fire when normalised 1-minute load average exceeds
            this value (``load_threshold`` triggers only).
        interval_seconds: Repeat interval for ``schedule`` triggers; also
            used as the polling cadence cap for threshold triggers.
        cooldown_seconds: Minimum seconds between consecutive firings.
    """

    name: str
    trigger_type: str
    enabled: bool = True
    requires_confirmation: bool = False
    prompt_template: str = ""
    watch_path: str = ""
    watch_patterns: list = field(default_factory=list)
    watch_recursive: bool = False
    disk_path: str = "/"
    disk_threshold_pct: float = 90.0
    load_threshold: float = 4.0
    interval_seconds: int = 300
    cooldown_seconds: int = 300


@dataclass
class ProactiveConfig:
    """Controls proactive (autonomous) task initiation.

    Attributes:
        enabled: Master switch — when ``False`` no triggers are evaluated.
        triggers: List of :class:`ProactiveTriggerConfig` instances.
    """

    enabled: bool = False
    triggers: list[ProactiveTriggerConfig] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class MissyConfig:
    """Root configuration object for a Missy runtime instance.

    Attributes:
        network: Network access policy.
        filesystem: Filesystem access policy.
        shell: Shell execution policy.
        plugins: Plugin loading policy.
        providers: Mapping of provider name to :class:`ProviderConfig`.
        workspace_path: Absolute path to the agent's working directory.
        audit_log_path: Absolute path where audit events are persisted.
        discord: Optional Discord integration configuration.
        scheduling: Scheduled job execution policy.
    """

    network: NetworkPolicy
    filesystem: FilesystemPolicy
    shell: ShellPolicy
    plugins: PluginPolicy
    providers: dict[str, ProviderConfig]
    workspace_path: str
    audit_log_path: str
    discord: Optional["DiscordConfig"] = None
    scheduling: SchedulingPolicy = field(default_factory=SchedulingPolicy)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    vault: VaultConfig = field(default_factory=VaultConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    sandbox: Optional["SandboxConfig"] = None
    max_spend_usd: float = 0.0  # 0 = unlimited; per-session budget cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_network(data: dict[str, Any]) -> NetworkPolicy:
    return NetworkPolicy(
        default_deny=bool(data.get("default_deny", True)),
        allowed_cidrs=list(data.get("allowed_cidrs", [])),
        allowed_domains=list(data.get("allowed_domains", [])),
        allowed_hosts=list(data.get("allowed_hosts", [])),
        provider_allowed_hosts=list(data.get("provider_allowed_hosts", [])),
        tool_allowed_hosts=list(data.get("tool_allowed_hosts", [])),
        discord_allowed_hosts=list(data.get("discord_allowed_hosts", [])),
    )


def _parse_filesystem(data: dict[str, Any]) -> FilesystemPolicy:
    return FilesystemPolicy(
        allowed_write_paths=list(data.get("allowed_write_paths", [])),
        allowed_read_paths=list(data.get("allowed_read_paths", [])),
    )


def _parse_shell(data: dict[str, Any]) -> ShellPolicy:
    return ShellPolicy(
        enabled=bool(data.get("enabled", False)),
        allowed_commands=list(data.get("allowed_commands", [])),
    )


def _parse_plugins(data: dict[str, Any]) -> PluginPolicy:
    return PluginPolicy(
        enabled=bool(data.get("enabled", False)),
        allowed_plugins=list(data.get("allowed_plugins", [])),
    )


def _parse_providers(data: dict[str, Any]) -> dict[str, ProviderConfig]:
    providers: dict[str, ProviderConfig] = {}
    for key, raw in data.items():
        if not isinstance(raw, dict):
            raise ConfigurationError(
                f"Provider '{key}' must be a mapping, got {type(raw).__name__}."
            )
        if "model" not in raw:
            raise ConfigurationError(f"Provider '{key}' is missing required field 'model'.")
        api_keys = list(raw.get("api_keys", []))
        api_key = raw.get("api_key") or os.environ.get(f"{key.upper()}_API_KEY")
        # If api_key is not set but api_keys has entries, use the first one.
        if not api_key and api_keys:
            api_key = api_keys[0]
        providers[key] = ProviderConfig(
            name=str(raw.get("name", key)),
            model=str(raw["model"]),
            api_key=api_key,
            base_url=raw.get("base_url"),
            timeout=int(raw.get("timeout", 30)),
            enabled=bool(raw.get("enabled", True)),
            api_keys=api_keys,
            fast_model=str(raw.get("fast_model", "")),
            premium_model=str(raw.get("premium_model", "")),
        )
    return providers


def _parse_scheduling(data: dict[str, Any]) -> SchedulingPolicy:
    return SchedulingPolicy(
        enabled=bool(data.get("enabled", True)),
        max_jobs=int(data.get("max_jobs", 0)),
        active_hours=str(data.get("active_hours", "")),
    )


def _parse_heartbeat(data: dict[str, Any]) -> HeartbeatConfig:
    return HeartbeatConfig(
        enabled=bool(data.get("enabled", False)),
        interval_seconds=int(data.get("interval_seconds", 1800)),
        workspace=str(data.get("workspace", "~/workspace")),
        active_hours=str(data.get("active_hours", "")),
    )


def _parse_observability(data: dict[str, Any]) -> ObservabilityConfig:
    return ObservabilityConfig(
        otel_enabled=bool(data.get("otel_enabled", False)),
        otel_endpoint=str(data.get("otel_endpoint", "http://localhost:4317")),
        otel_protocol=str(data.get("otel_protocol", "grpc")),
        otel_service_name=str(data.get("otel_service_name", "missy")),
        log_level=str(data.get("log_level", "warning")),
    )


def _parse_vault(data: dict[str, Any]) -> VaultConfig:
    return VaultConfig(
        enabled=bool(data.get("enabled", False)),
        vault_dir=str(data.get("vault_dir", "~/.missy/secrets")),
    )


def _parse_proactive_trigger(raw: Any) -> ProactiveTriggerConfig:
    """Parse a single proactive trigger dict from YAML."""
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"Each proactive trigger must be a mapping, got {type(raw).__name__}."
        )
    name = raw.get("name")
    if not name:
        raise ConfigurationError("Each proactive trigger must have a 'name' field.")
    trigger_type = raw.get("trigger_type")
    if not trigger_type:
        raise ConfigurationError(
            f"Proactive trigger '{name}' is missing required field 'trigger_type'."
        )
    return ProactiveTriggerConfig(
        name=str(name),
        trigger_type=str(trigger_type),
        enabled=bool(raw.get("enabled", True)),
        requires_confirmation=bool(raw.get("requires_confirmation", False)),
        prompt_template=str(raw.get("prompt_template", "")),
        watch_path=str(raw.get("watch_path", "")),
        watch_patterns=list(raw.get("watch_patterns", [])),
        watch_recursive=bool(raw.get("watch_recursive", False)),
        disk_path=str(raw.get("disk_path", "/")),
        disk_threshold_pct=float(raw.get("disk_threshold_pct", 90.0)),
        load_threshold=float(raw.get("load_threshold", 4.0)),
        interval_seconds=int(raw.get("interval_seconds", 300)),
        cooldown_seconds=int(raw.get("cooldown_seconds", 300)),
    )


def _parse_proactive(data: dict[str, Any]) -> ProactiveConfig:
    """Parse the ``proactive`` section of a Missy config dict."""
    triggers = [
        _parse_proactive_trigger(raw) for raw in data.get("triggers", [])
    ]
    return ProactiveConfig(
        enabled=bool(data.get("enabled", False)),
        triggers=triggers,
    )


def _parse_sandbox(data: dict[str, Any]) -> "SandboxConfig":
    """Parse the ``sandbox`` section of a Missy config dict."""
    from missy.security.sandbox import parse_sandbox_config

    return parse_sandbox_config(data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str) -> MissyConfig:
    """Load a :class:`MissyConfig` from a YAML file at *path*.

    Args:
        path: Absolute or relative path to the YAML configuration file.

    Returns:
        A fully populated :class:`MissyConfig` instance.

    Raises:
        ConfigurationError: If the file cannot be read, is not valid YAML, or
            contains invalid values.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration path is not a file: {path}")

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Cannot read configuration file '{path}': {exc}") from exc

    try:
        data: Any = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in '{path}': {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Top-level YAML value in '{path}' must be a mapping, "
            f"got {type(data).__name__}."
        )

    try:
        discord_raw = data.get("discord")
        discord_cfg: Optional["DiscordConfig"] = None
        if discord_raw is not None:
            from missy.channels.discord.config import parse_discord_config
            discord_cfg = parse_discord_config(discord_raw)

        return MissyConfig(
            network=_parse_network(data.get("network") or {}),
            filesystem=_parse_filesystem(data.get("filesystem") or {}),
            shell=_parse_shell(data.get("shell") or {}),
            plugins=_parse_plugins(data.get("plugins") or {}),
            providers=_parse_providers(data.get("providers") or {}),
            workspace_path=str(data.get("workspace_path", ".")),
            audit_log_path=str(data.get("audit_log_path", "~/.missy/audit.log")),
            discord=discord_cfg,
            scheduling=_parse_scheduling(data.get("scheduling") or {}),
            heartbeat=_parse_heartbeat(data.get("heartbeat") or {}),
            observability=_parse_observability(data.get("observability") or {}),
            vault=_parse_vault(data.get("vault") or {}),
            proactive=_parse_proactive(data.get("proactive") or {}),
            sandbox=_parse_sandbox(data.get("sandbox") or {}),
            max_spend_usd=float(data.get("max_spend_usd", 0.0)),
        )
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Error parsing configuration file '{path}': {exc}") from exc


def get_default_config() -> MissyConfig:
    """Return a :class:`MissyConfig` with secure-by-default settings.

    The returned configuration:

    * Denies all outbound network access (``default_deny=True``).
    * Grants no filesystem read or write permissions.
    * Disables shell execution.
    * Disables plugins.
    * Defines no providers (each provider must be configured explicitly).

    Returns:
        A :class:`MissyConfig` suitable for use as a safe baseline.
    """
    return MissyConfig(
        network=NetworkPolicy(
            default_deny=True,
            allowed_cidrs=[],
            allowed_domains=[],
            allowed_hosts=[],
        ),
        filesystem=FilesystemPolicy(
            allowed_read_paths=[],
            allowed_write_paths=[],
        ),
        shell=ShellPolicy(enabled=False, allowed_commands=[]),
        plugins=PluginPolicy(enabled=False, allowed_plugins=[]),
        providers={},
        workspace_path=str(Path.home() / "missy-workspace"),
        audit_log_path=str(Path.home() / ".missy" / "audit.log"),
        discord=None,
        scheduling=SchedulingPolicy(),
        heartbeat=HeartbeatConfig(),
        observability=ObservabilityConfig(),
        vault=VaultConfig(),
        proactive=ProactiveConfig(),
    )
