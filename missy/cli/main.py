"""Missy CLI - Security-first local agentic assistant.

Entry point registered in ``pyproject.toml`` as ``missy = "missy.cli.main:cli"``.

All commands share a single pattern:

1. Resolve the config path (``--config`` option, default ``~/.missy/config.yaml``).
2. Load :class:`~missy.config.settings.MissyConfig` via :func:`load_config`.
3. Initialise the policy engine and provider registry so that every
   sub-command operates under the configured security policy.
4. Perform the requested operation.

Errors are caught at the command boundary and rendered with rich so that
users see a clear, styled message rather than a raw traceback.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Module-level rich consoles
# ---------------------------------------------------------------------------

console = Console()
err_console = Console(stderr=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = "~/.missy/config.yaml"
DEFAULT_AUDIT_LOG = "~/.missy/audit.jsonl"
DEFAULT_JOBS_FILE = "~/.missy/jobs.json"
DEFAULT_APP_LOG = "~/.missy/missy.log"

# ---------------------------------------------------------------------------
# Default config YAML written by ``missy init``
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_YAML = """\
# Missy configuration — edit to enable capabilities.
# All capabilities are disabled by default (secure-by-default posture).

config_version: 2

network:
  default_deny: true
  presets:
    - anthropic
    - openai
  allowed_cidrs: []
  allowed_domains: []
  allowed_hosts: []

filesystem:
  allowed_write_paths:
    - "~/workspace"
    - "~/.missy"
  allowed_read_paths:
    - "~/workspace"
    - "~/.missy"
    - "/tmp"

shell:
  enabled: false
  allowed_commands: []

plugins:
  enabled: false
  allowed_plugins: []

# Tool visibility policy. Execution is still governed by network/filesystem/shell policy.
tools:
  profile: full
  allow: []
  deny: []
  alsoAllow: []

providers:
  anthropic:
    name: anthropic
    model: "claude-sonnet-4-6"
    timeout: 30

workspace_path: "~/workspace"
audit_log_path: "~/.missy/audit.jsonl"

# Heartbeat: periodic agent task from HEARTBEAT.md
heartbeat:
  enabled: false
  interval_seconds: 1800
  active_hours: ""

# Observability: OpenTelemetry export
observability:
  otel_enabled: false
  otel_endpoint: "http://localhost:4317"
  log_file_path: "~/.missy/missy.log"
  log_level: "warning"

# Vault: encrypted secrets store
vault:
  enabled: false
  vault_dir: "~/.missy/secrets"

# Vision: on-demand visual capabilities (requires pip install -e ".[vision]")
vision:
  enabled: true
  auto_activate_threshold: 0.80
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_subsystems(config_path: str) -> Any:
    """Load config, init policy engine, audit logger, and provider registry.

    Returns:
        The loaded :class:`~missy.config.settings.MissyConfig`.

    Raises:
        SystemExit: On any initialisation failure (message printed to stderr).
    """
    from missy.config.settings import load_config
    from missy.core.exceptions import ConfigurationError
    from missy.core.message_bus import init_message_bus
    from missy.observability.audit_logger import init_audit_logger
    from missy.policy.engine import init_policy_engine
    from missy.providers.registry import init_registry

    expanded = str(Path(config_path).expanduser())

    # Run config migration before loading (backs up + rewrites if needed)
    try:
        from missy.config.migrate import migrate_config

        migration = migrate_config(expanded)
        if migration["migrated"]:
            presets = migration["presets_detected"]
            backup = migration["backup_path"]
            console.print(
                f"[green]Config migrated to v{migration['version']}[/]"
                + (f" — detected presets: {', '.join(presets)}" if presets else "")
                + (f"\n  [dim]Backup: {backup}[/]" if backup else "")
            )
    except Exception as _mig_exc:
        logger.warning("Config migration skipped: %s", _mig_exc)

    try:
        cfg = load_config(expanded)
    except ConfigurationError as exc:
        err_console.print(
            Panel(
                f"[bold red]Configuration error[/]\n\n{exc}\n\n"
                f"Run [bold cyan]missy init[/] to create a default configuration, "
                f"or check that [bold]{config_path}[/] is valid YAML.",
                title="[red]Error[/]",
                border_style="red",
            )
        )
        sys.exit(1)
    except Exception as exc:
        err_console.print(f"[red]Unexpected error loading configuration: {exc}[/]")
        sys.exit(1)

    _configure_file_logging(cfg)

    init_policy_engine(cfg)
    init_audit_logger(cfg.audit_log_path)
    init_registry(cfg)
    # docs/architecture.md documents this as part of the bootstrap sequence
    # (between provider registry and tool registry init), but it was never
    # actually called anywhere in the running app -- AgentRuntime._make_message_bus()
    # and RunRegistry._default_bus() both gracefully degrade to bus=None when
    # get_message_bus() raises "not initialised", so the gap was silent: the
    # Web TUI's live run console never showed tool-call events or
    # provider/tools_used/cost in its completion summary, with no error
    # surfaced anywhere.
    init_message_bus()

    # Register built-in tools so the agent can use them.
    try:
        from missy.tools.builtin import register_builtin_tools
        from missy.tools.registry import init_tool_registry

        tool_registry = init_tool_registry()
        register_builtin_tools(tool_registry)
        # ToolRegistry.disable()/is_enabled() were fully built and
        # tested (execute() refuses a disabled tool outright, and
        # AgentRuntime._get_tools() already filters is_enabled() tools
        # out of what's offered to the model) but had zero callers
        # anywhere in the codebase -- an operator had no way to actually
        # disable a tool via any first-party surface. tools.disabled_tools
        # makes this reachable via config.
        for _disabled_name in getattr(cfg.tools, "disabled_tools", None) or []:
            try:
                tool_registry.disable(_disabled_name)
            except KeyError:
                logger.warning(
                    "tools.disabled_tools: %r is not a registered tool name; ignoring.",
                    _disabled_name,
                )
    except Exception as _tool_exc:
        logger.debug("Tool registry init failed: %s", _tool_exc)

    # Initialize OpenTelemetry if configured.
    try:
        from missy.observability.otel import init_otel

        init_otel(cfg)
    except Exception as _otel_exc:
        logger.debug("OpenTelemetry init failed: %s", _otel_exc)

    return cfg


def _parse_log_level(value: Any, default: int = logging.WARNING) -> int:
    """Return a logging level from a config value."""
    if isinstance(value, int):
        return value
    name = str(value or "").strip().upper()
    return int(getattr(logging, name, default))


def _app_log_path(cfg: Any) -> Path:
    """Resolve the configured application log file path."""
    obs = getattr(cfg, "observability", None)
    raw = getattr(obs, "log_file_path", "") if obs is not None else ""
    if raw is None or raw.__class__.__module__.startswith("unittest.mock"):
        raw = ""
    return Path(raw or DEFAULT_APP_LOG).expanduser()


def _configure_file_logging(cfg: Any) -> Path:
    """Attach a rotating application log file handler to the root logger."""
    log_path = _app_log_path(cfg)
    root = logging.getLogger()
    obs = getattr(cfg, "observability", None)
    configured_level = _parse_log_level(getattr(obs, "log_level", "warning"), logging.WARNING)
    level = logging.DEBUG if root.getEffectiveLevel() <= logging.DEBUG else configured_level

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        logger.warning("Could not create log directory %s: %s", log_path.parent, exc)
        return log_path

    for handler in list(root.handlers):
        if not getattr(handler, "_missy_app_log_handler", False):
            continue
        if getattr(handler, "_missy_app_log_path", "") == str(log_path):
            handler.setLevel(level)
            root.setLevel(min(root.getEffectiveLevel(), level))
            return log_path
        root.removeHandler(handler)
        with contextlib.suppress(Exception):
            handler.close()

    try:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
            delay=True,
        )
    except OSError as exc:
        logger.warning("Could not open application log %s: %s", log_path, exc)
        return log_path

    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    file_handler._missy_app_log_handler = True  # type: ignore[attr-defined]
    file_handler._missy_app_log_path = str(log_path)  # type: ignore[attr-defined]
    root.addHandler(file_handler)
    root.setLevel(min(root.getEffectiveLevel(), level))
    logger.debug("Application log configured at %s", log_path)
    return log_path


def _agent_tool_policy_kwargs(
    cfg: Any,
    *,
    agent_id: str = "default",
) -> dict[str, Any]:
    """Return AgentConfig keyword args for config-backed tool policy layers."""
    agents = getattr(cfg, "agents", {}) or {}
    agent_cfg = (
        (agents.get(agent_id) or agents.get("default")) if isinstance(agents, dict) else None
    )
    sandbox = getattr(cfg, "sandbox", None)

    def _policy(value: Any) -> Any | None:
        if value is None or value.__class__.__module__.startswith("unittest.mock"):
            return None
        if isinstance(value, dict) or hasattr(value, "allow") or hasattr(value, "deny"):
            return value
        return None

    def _intelligence(value: Any) -> Any | None:
        if value is None or value.__class__.__module__.startswith("unittest.mock"):
            return None
        return value

    return {
        "agent_id": agent_id,
        "tool_policy": _policy(getattr(cfg, "tools", None)),
        "agent_tool_policy": _policy(getattr(agent_cfg, "tools", None)),
        "sandbox_tool_policy": _policy(getattr(sandbox, "tools", None)),
        "subagent_tool_policy": _policy(getattr(agent_cfg, "subagent_tools", None)),
        "tool_intelligence": _intelligence(getattr(cfg, "tool_intelligence", None)),
    }


def _load_or_create_web_console_key() -> str:
    """Load the persistent Web TUI operator key, generating one on first run.

    Stored outside config.yaml (like the vault key) so the console has a
    stable login credential across gateway restarts without requiring the
    operator to configure one.
    """
    import secrets

    key_path = Path("~/.missy/secrets/web_console.key").expanduser()
    if key_path.exists():
        existing = key_path.read_text(encoding="utf-8").strip()
        if existing:
            return existing

    key_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = secrets.token_hex(32)
    key_path.write_text(key, encoding="utf-8")
    key_path.chmod(0o600)
    return key


def _print_error(message: str, hint: str | None = None) -> None:
    """Render a styled error panel to stderr."""
    body = message
    if hint:
        body += f"\n\n[dim]{hint}[/]"
    err_console.print(Panel(body, title="[red]Error[/]", border_style="red", expand=False))


def _print_success(message: str) -> None:
    """Render a styled success panel to stdout."""
    console.print(Panel(message, title="[green]Success[/]", border_style="green", expand=False))


def _ensure_tool_registry():
    """Return the process-level ToolRegistry, bootstrapping it if needed.

    Standalone CLI commands (e.g. `missy tools benchmark run`) run in a
    fresh process that never goes through the main agent-construction path
    (`gateway_start`/`ask`/`run`), which is the only place that otherwise
    calls init_tool_registry() + register_builtin_tools(). Without this,
    get_tool_registry() always raises "not initialised".
    """
    from missy.tools.registry import get_tool_registry, init_tool_registry

    try:
        return get_tool_registry()
    except RuntimeError:
        from missy.tools.builtin import register_builtin_tools

        registry = init_tool_registry()
        register_builtin_tools(registry)
        return registry


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.option(
    "--config",
    default=DEFAULT_CONFIG,
    show_default=True,
    help="Path to Missy configuration YAML file.",
    envvar="MISSY_CONFIG",
)
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, config: str, debug: bool) -> None:
    """Missy - Security-first local agentic assistant.

    All network, filesystem, shell, and plugin access is governed by the
    policy defined in your configuration file.  Run [bold]missy init[/] to
    create a default configuration.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# missy init
# ---------------------------------------------------------------------------


@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize Missy configuration and workspace.

    Creates the ``~/.missy/`` directory, writes a default ``config.yaml``
    with a secure-by-default policy, and creates empty placeholder files for
    the audit log and jobs store.
    """
    missy_dir = Path("~/.missy").expanduser()

    try:
        missy_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        _print_error(f"Cannot create directory {missy_dir}: {exc}")
        sys.exit(1)

    config_file = missy_dir / "config.yaml"
    if config_file.exists():
        console.print(f"[yellow]Config already exists at [bold]{config_file}[/] — skipping.[/]")
    else:
        config_file.write_text(_DEFAULT_CONFIG_YAML, encoding="utf-8")
        console.print(f"[green]Created[/] {config_file}")

    audit_file = missy_dir / "audit.jsonl"
    if not audit_file.exists():
        audit_file.touch()
        console.print(f"[green]Created[/] {audit_file}")

    jobs_file = missy_dir / "jobs.json"
    if not jobs_file.exists():
        jobs_file.write_text("[]", encoding="utf-8")
        console.print(f"[green]Created[/] {jobs_file}")

    secrets_dir = missy_dir / "secrets"
    if not secrets_dir.exists():
        secrets_dir.mkdir(mode=0o700)
        console.print(f"[green]Created[/] secrets directory at {secrets_dir} (mode 700)")

    workspace = Path("~/workspace").expanduser()
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Created[/] workspace at {workspace}")
    except OSError:
        console.print(f"[yellow]Could not create workspace at {workspace} — create it manually.[/]")

    _print_success(
        f"Missy initialised.\n\n"
        f"  Config  : [bold]{config_file}[/]\n"
        f"  Audit   : [bold]{audit_file}[/]\n"
        f"  Jobs    : [bold]{jobs_file}[/]\n"
        f"  Workspace: [bold]{workspace}[/]\n\n"
        f"Edit [bold]{config_file}[/] to configure providers and enable capabilities."
    )


# ---------------------------------------------------------------------------
# missy setup (onboarding wizard)
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--provider", "setup_provider", default=None, help="Provider name for non-interactive mode."
)
@click.option("--api-key", "setup_api_key", default=None, help="API key (direct value).")
@click.option(
    "--api-key-env",
    "setup_api_key_env",
    default=None,
    help="Environment variable containing the API key.",
)
@click.option("--model", "setup_model", default=None, help="Model identifier.")
@click.option("--workspace", "setup_workspace", default=None, help="Workspace directory path.")
@click.option("--no-prompt", is_flag=True, default=False, help="Non-interactive mode (no prompts).")
@click.pass_context
def setup(
    ctx: click.Context,
    setup_provider: str | None,
    setup_api_key: str | None,
    setup_api_key_env: str | None,
    setup_model: str | None,
    setup_workspace: str | None,
    no_prompt: bool,
) -> None:
    """Interactive onboarding wizard — configure providers and write config.yaml.

    Walks through workspace setup, AI provider selection (Anthropic / OpenAI /
    Ollama), API key entry with masked preview and optional live verification,
    model tier selection, and atomic config write.

    Safe to re-run: prompts before overwriting an existing config.

    Use --no-prompt for non-interactive mode (requires --provider).
    """
    config_path = (
        ctx.obj.get("config_path", "~/.missy/config.yaml") if ctx.obj else "~/.missy/config.yaml"
    )

    if no_prompt:
        from missy.cli.wizard import run_wizard_noninteractive

        if not setup_provider:
            _print_error("--provider is required in --no-prompt mode.")
            sys.exit(1)
        try:
            run_wizard_noninteractive(
                config_path=config_path,
                provider=setup_provider,
                api_key=setup_api_key,
                api_key_env=setup_api_key_env,
                model=setup_model,
                workspace=setup_workspace,
            )
        except click.ClickException as exc:
            _print_error(str(exc.message))
            sys.exit(1)
        return

    from missy.cli.wizard import run_wizard

    try:
        run_wizard(config_path)
    except (KeyboardInterrupt, click.Abort):
        console.print("\n[dim]Setup aborted. Nothing was written.[/]")
        sys.exit(0)


# ---------------------------------------------------------------------------
# missy ask
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("prompt")
@click.option("--provider", default=None, help="Provider to use (overrides config default).")
@click.option("--session", default=None, help="Session ID for conversation continuity.")
@click.option(
    "--mode",
    "capability_mode",
    default="full",
    type=click.Choice(["full", "safe-chat", "no-tools"], case_sensitive=False),
    show_default=True,
    help="Capability mode: full (all tools), safe-chat (read-only tools), no-tools (pure chat).",
)
@click.pass_context
def ask(
    ctx: click.Context,
    prompt: str,
    provider: str | None,
    session: str | None,
    capability_mode: str,
) -> None:
    """Ask Missy a single question and print the response.

    PROMPT is the question or instruction for Missy.

    \b
    Example:
        missy ask "What is the capital of France?"
        missy ask --provider ollama "Summarise this text: ..."
    """
    from missy.agent.runtime import AgentConfig, AgentRuntime
    from missy.core.exceptions import ProviderError
    from missy.security.sanitizer import sanitizer
    from missy.security.secrets import secrets_detector

    # Check hatching status (non-blocking hint)
    try:
        from missy.agent.hatching import HatchingManager

        if HatchingManager().needs_hatching():
            console.print(
                "[dim]Tip: Run [bold]missy hatch[/bold] to complete initial setup.[/dim]\n"
            )
    except Exception:  # noqa: BLE001
        logger.debug("Hatching check skipped", exc_info=True)

    cfg = _load_subsystems(ctx.obj["config_path"])

    # Security: detect secrets in the prompt before sending.
    if secrets_detector.has_secrets(prompt):
        err_console.print(
            Panel(
                "[yellow]Warning:[/] Your prompt appears to contain credentials or secrets.\n"
                "These will be sent to the configured AI provider.\n"
                "Proceeding — redact sensitive data from your prompt if this is unintentional.",
                title="[yellow]Security Warning[/]",
                border_style="yellow",
            )
        )

    # Sanitize (truncate + log injection warnings).
    clean_prompt = sanitizer.sanitize(prompt)

    injection_matches = sanitizer.check_for_injection(clean_prompt)
    if injection_matches:
        err_console.print(
            Panel(
                "[yellow]Warning:[/] Potential prompt injection patterns detected in your input.\n"
                f"Patterns: {injection_matches}\n"
                "Proceeding with caution.",
                title="[yellow]Security Warning[/]",
                border_style="yellow",
            )
        )

    # Resolve provider.
    provider_name = provider or (
        next(iter(cfg.providers), "anthropic") if cfg.providers else "anthropic"
    )

    agent_cfg = AgentConfig(
        provider=provider_name,
        capability_mode=capability_mode,
        max_spend_usd=getattr(cfg, "max_spend_usd", 0.0),
        **_agent_tool_policy_kwargs(cfg),
    )
    agent = AgentRuntime(agent_cfg)

    with console.status("[bold cyan]Thinking...[/]", spinner="dots"):
        try:
            response = agent.run(clean_prompt, session_id=session)
        except ProviderError as exc:
            _print_error(
                f"Provider error: {exc}",
                hint=(
                    "Check that your API key is set and the provider is configured "
                    "in your config file."
                ),
            )
            sys.exit(1)
        except Exception as exc:
            _print_error(f"Unexpected error: {exc}")
            sys.exit(1)

    console.print(Panel(response, title="[bold cyan]Missy[/]", border_style="cyan"))


# ---------------------------------------------------------------------------
# missy run
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--provider", default=None, help="Provider to use.")
@click.option(
    "--session",
    default="default",
    show_default=True,
    help="Session ID for conversation continuity. Use 'default' to persist memory across runs.",
)
@click.option(
    "--mode",
    "capability_mode",
    default="full",
    type=click.Choice(["full", "safe-chat", "no-tools"], case_sensitive=False),
    show_default=True,
    help="Capability mode: full (all tools), safe-chat (read-only tools), no-tools (pure chat).",
)
@click.pass_context
def run(ctx: click.Context, provider: str | None, session: str, capability_mode: str) -> None:
    """Start an interactive session with Missy.

    Type your messages and press Enter.  Type [bold]quit[/] or [bold]exit[/],
    or press Ctrl-C / Ctrl-D to end the session.
    """
    from missy.agent.runtime import AgentConfig, AgentRuntime
    from missy.channels.cli_channel import CLIChannel
    from missy.core.exceptions import ProviderError
    from missy.security.sanitizer import sanitizer
    from missy.security.secrets import secrets_detector

    cfg = _load_subsystems(ctx.obj["config_path"])

    # Check hatching status (non-blocking hint)
    try:
        from missy.agent.hatching import HatchingManager

        if HatchingManager().needs_hatching():
            console.print(
                "[dim]Tip: Run [bold]missy hatch[/bold] to complete initial setup.[/dim]\n"
            )
    except Exception:  # noqa: BLE001
        logger.debug("Hatching check skipped", exc_info=True)

    provider_name = provider or (
        next(iter(cfg.providers), "anthropic") if cfg.providers else "anthropic"
    )

    agent_cfg = AgentConfig(
        provider=provider_name,
        capability_mode=capability_mode,
        max_spend_usd=getattr(cfg, "max_spend_usd", 0.0),
        **_agent_tool_policy_kwargs(cfg),
    )
    agent = AgentRuntime(agent_cfg)
    channel = CLIChannel()

    mode_label = {
        "full": "full (all tools)",
        "safe-chat": "safe-chat (read-only)",
        "no-tools": "no-tools (pure chat)",
    }
    console.print(
        Panel(
            "[bold cyan]Missy[/] interactive session\n\n"
            f"Provider : [bold]{provider_name}[/]\n"
            f"Mode     : [bold]{mode_label.get(capability_mode, capability_mode)}[/]\n"
            "Type [bold]quit[/] or [bold]exit[/] to end, or press Ctrl-D.",
            border_style="cyan",
        )
    )

    # Notify user about incomplete tasks from previous runs
    recovery = agent.pending_recovery
    if recovery:
        resumable = [r for r in recovery if r.action == "resume"]
        restartable = [r for r in recovery if r.action == "restart"]
        if resumable:
            console.print(
                f"[yellow]Found {len(resumable)} resumable task(s) from previous sessions.[/]"
            )
            for r in resumable[:3]:
                prompt_preview = (r.prompt[:60] + "...") if len(r.prompt) > 60 else r.prompt
                console.print(f"  [dim]• session={r.session_id} — {prompt_preview}[/]")
        if restartable:
            console.print(
                f"[yellow]Found {len(restartable)} restartable task(s) (older, may need restart).[/]"
            )

    session_id: str = session  # stable across turns and re-invocations

    while True:
        # Render our own prompt via rich then delegate to the channel.
        console.print("[bold cyan]You>[/] ", end="")

        msg = channel.receive()
        if msg is None:
            # EOF / Ctrl-D
            console.print("\n[dim]Session ended.[/]")
            break

        user_text = msg.content.strip()

        if not user_text:
            continue

        if user_text.lower() in {"quit", "exit", "q"}:
            console.print("[dim]Goodbye.[/]")
            break

        # Security checks.
        if secrets_detector.has_secrets(user_text):
            console.print(
                "[yellow]Warning:[/] Potential secrets detected in your input. "
                "These will be sent to the provider."
            )

        clean_text = sanitizer.sanitize(user_text)
        injection_matches = sanitizer.check_for_injection(clean_text)
        if injection_matches:
            console.print(f"[yellow]Warning:[/] Injection patterns detected: {injection_matches}")

        with console.status("[bold cyan]Thinking...[/]", spinner="dots"):
            try:
                response = agent.run(clean_text, session_id=session_id)
            except ProviderError as exc:
                console.print(f"[red]Provider error:[/] {exc}")
                continue
            except Exception as exc:
                console.print(f"[red]Error:[/] {exc}")
                continue

        console.print(Panel(response, title="[bold cyan]Missy[/]", border_style="cyan"))


# ---------------------------------------------------------------------------
# missy schedule
# ---------------------------------------------------------------------------


@cli.group()
def schedule() -> None:
    """Manage scheduled jobs (recurring agent tasks)."""


@schedule.command("add")
@click.option("--name", required=True, help="Human-readable job name.")
@click.option(
    "--schedule",
    "schedule_str",
    required=True,
    help='Schedule expression, e.g. "every 5 minutes" or "daily at 09:00".',
)
@click.option("--task", required=True, help="Prompt/task text to run on each firing.")
@click.option("--provider", default="anthropic", show_default=True, help="AI provider to use.")
@click.option(
    "--capability-mode",
    "capability_mode",
    default="safe-chat",
    type=click.Choice(["full", "safe-chat", "no-tools"], case_sensitive=False),
    show_default=True,
    help=(
        "Tool-access mode for this job's unattended run. safe-chat (read-only "
        "tools) is the default so a scheduled job's blast radius is smaller "
        "than an interactive session by default; pass --capability-mode full "
        "to opt this specific job into unrestricted tool access."
    ),
)
@click.option("--description", default="", help="Optional description of the job.")
@click.option(
    "--max-attempts",
    "max_attempts",
    type=int,
    default=3,
    show_default=True,
    help="Maximum retry attempts on failure.",
)
@click.option(
    "--backoff-seconds",
    "backoff_seconds",
    default="",
    help='Comma-separated retry delays in seconds, e.g. "30,60,300" (default: 30,60,300).',
)
@click.option(
    "--retry-on",
    "retry_on",
    default="",
    help='Comma-separated error categories that trigger a retry, e.g. "network,provider_error".',
)
@click.option(
    "--delete-after-run",
    "delete_after_run",
    is_flag=True,
    default=False,
    help="Remove this job after it runs once successfully.",
)
@click.option(
    "--active-hours",
    "active_hours",
    default="",
    help='"HH:MM-HH:MM" window; the job is skipped when it fires outside it.',
)
@click.option(
    "--timezone",
    "job_timezone",
    default="",
    help='IANA timezone string for the schedule (e.g. "America/New_York").',
)
@click.pass_context
def schedule_add(
    ctx: click.Context,
    name: str,
    schedule_str: str,
    task: str,
    provider: str,
    capability_mode: str,
    description: str,
    max_attempts: int,
    backoff_seconds: str,
    retry_on: str,
    delete_after_run: bool,
    active_hours: str,
    job_timezone: str,
) -> None:
    """Add a new scheduled job.

    \b
    Example:
        missy schedule add \\
            --name "Daily digest" \\
            --schedule "daily at 09:00" \\
            --task "Summarise the news"

    By default the job runs in safe-chat mode (read-only tools only).
    Pass --capability-mode full for jobs that need write access, shell,
    or other elevated tools.
    """
    from missy.core.exceptions import SchedulerError
    from missy.scheduler.manager import SchedulerManager

    cfg = _load_subsystems(ctx.obj["config_path"])
    mgr = SchedulerManager(max_jobs=getattr(cfg.scheduling, "max_jobs", 0))

    parsed_backoff = (
        [int(x.strip()) for x in backoff_seconds.split(",") if x.strip()]
        if backoff_seconds
        else None
    )
    parsed_retry_on = [x.strip() for x in retry_on.split(",") if x.strip()] if retry_on else None

    try:
        mgr.start()
        job = mgr.add_job(
            name=name,
            schedule=schedule_str,
            task=task,
            provider=provider,
            capability_mode=capability_mode,
            description=description,
            max_attempts=max_attempts,
            backoff_seconds=parsed_backoff,
            retry_on=parsed_retry_on,
            delete_after_run=delete_after_run,
            active_hours=active_hours,
            timezone=job_timezone,
        )
        mgr.stop()
    except ValueError as exc:
        _print_error(
            f"Invalid schedule expression: {exc}",
            hint='Try "every 5 minutes", "every hour", or "daily at 09:00".',
        )
        sys.exit(1)
    except SchedulerError as exc:
        _print_error(f"Scheduler error: {exc}")
        sys.exit(1)

    _print_success(
        f"Job added.\n\n"
        f"  ID      : [bold]{job.id}[/]\n"
        f"  Name    : {job.name}\n"
        f"  Schedule: {job.schedule}\n"
        f"  Provider: {job.provider}\n"
        f"  Mode    : {job.capability_mode}"
    )


@schedule.command("list")
@click.pass_context
def schedule_list(ctx: click.Context) -> None:
    """List all scheduled jobs."""
    from missy.scheduler.manager import SchedulerManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = SchedulerManager()
    jobs = mgr.load_jobs()

    if not jobs:
        console.print("[dim]No scheduled jobs.[/]")
        return

    table = Table(title="Scheduled Jobs", show_lines=True)
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Name", style="bold")
    table.add_column("Schedule")
    table.add_column("Provider")
    table.add_column("Mode")
    table.add_column("Enabled", justify="center")
    table.add_column("Runs", justify="right")
    table.add_column("Last Run")
    table.add_column("Next Run")

    for job in jobs:
        enabled_text = Text("yes", style="green") if job.enabled else Text("no", style="red")
        last_run = job.last_run.strftime("%Y-%m-%d %H:%M") if job.last_run else "[dim]never[/]"
        next_run = job.next_run.strftime("%Y-%m-%d %H:%M") if job.next_run else "[dim]—[/]"
        table.add_row(
            job.id[:8] + "…",
            job.name,
            job.schedule,
            job.provider,
            job.capability_mode,
            enabled_text,
            str(job.run_count),
            last_run,
            next_run,
        )

    console.print(table)


@schedule.command("pause")
@click.argument("job_id")
@click.pass_context
def schedule_pause(ctx: click.Context, job_id: str) -> None:
    """Pause scheduled job JOB_ID so it no longer fires."""
    from missy.core.exceptions import SchedulerError
    from missy.scheduler.manager import SchedulerManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = SchedulerManager()

    try:
        mgr.start()
        mgr.pause_job(job_id)
        mgr.stop()
    except KeyError:
        _print_error(f"No job found with ID: {job_id!r}")
        sys.exit(1)
    except SchedulerError as exc:
        _print_error(f"Scheduler error: {exc}")
        sys.exit(1)

    _print_success(f"Job [bold]{job_id}[/] paused.")


@schedule.command("resume")
@click.argument("job_id")
@click.pass_context
def schedule_resume(ctx: click.Context, job_id: str) -> None:
    """Resume paused scheduled job JOB_ID."""
    from missy.core.exceptions import SchedulerError
    from missy.scheduler.manager import SchedulerManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = SchedulerManager()

    try:
        mgr.start()
        mgr.resume_job(job_id)
        mgr.stop()
    except KeyError:
        _print_error(f"No job found with ID: {job_id!r}")
        sys.exit(1)
    except SchedulerError as exc:
        _print_error(f"Scheduler error: {exc}")
        sys.exit(1)

    _print_success(f"Job [bold]{job_id}[/] resumed.")


@schedule.command("remove")
@click.argument("job_id")
@click.confirmation_option(prompt="Remove this job?")
@click.pass_context
def schedule_remove(ctx: click.Context, job_id: str) -> None:
    """Permanently remove scheduled job JOB_ID."""
    from missy.core.exceptions import SchedulerError
    from missy.scheduler.manager import SchedulerManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = SchedulerManager()

    try:
        mgr.start()
        mgr.remove_job(job_id)
        mgr.stop()
    except KeyError:
        _print_error(f"No job found with ID: {job_id!r}")
        sys.exit(1)
    except SchedulerError as exc:
        _print_error(f"Scheduler error: {exc}")
        sys.exit(1)

    _print_success(f"Job [bold]{job_id}[/] removed.")


# ---------------------------------------------------------------------------
# missy audit
# ---------------------------------------------------------------------------


@cli.group()
def audit() -> None:
    """Audit log and security event commands."""


@audit.command("security")
@click.option(
    "--limit",
    default=50,
    show_default=True,
    help="Maximum number of events to show.",
)
@click.pass_context
def audit_security(ctx: click.Context, limit: int) -> None:
    """Show recent security audit events (policy violations / denials).

    Reads from the JSONL audit log configured in your config file and renders
    the most recent policy-denial events as a table.
    """
    from missy.observability.audit_logger import AuditLogger

    cfg = _load_subsystems(ctx.obj["config_path"])
    al = AuditLogger(log_path=cfg.audit_log_path)
    violations = al.get_policy_violations(limit=limit)

    if not violations:
        console.print("[dim]No policy violations found in the audit log.[/]")
        return

    table = Table(title=f"Policy Violations (last {limit})", show_lines=True)
    table.add_column("Timestamp", style="dim")
    table.add_column("Event Type")
    table.add_column("Category")
    table.add_column("Result", justify="center")
    table.add_column("Detail")

    for event in violations:
        result = event.get("result", "")
        result_text = Text(result, style="red" if result == "deny" else "yellow")
        detail_raw = event.get("detail", {})
        detail_str = (
            json.dumps(detail_raw, separators=(",", ":"))
            if isinstance(detail_raw, dict)
            else str(detail_raw)
        )
        table.add_row(
            event.get("timestamp", "")[:19],
            event.get("event_type", ""),
            event.get("category", ""),
            result_text,
            detail_str[:80] + ("…" if len(detail_str) > 80 else ""),
        )

    console.print(table)


@audit.command("recent")
@click.option(
    "--limit",
    default=50,
    show_default=True,
    help="Maximum number of events to show.",
)
@click.option(
    "--category",
    default=None,
    help="Filter by event category (e.g. network, filesystem, shell, plugin, scheduler).",
)
@click.pass_context
def audit_recent(ctx: click.Context, limit: int, category: str | None) -> None:
    """Show recent audit events from the log.

    \b
    Example:
        missy audit recent --limit 20 --category network
    """
    from missy.observability.audit_logger import AuditLogger

    cfg = _load_subsystems(ctx.obj["config_path"])
    al = AuditLogger(log_path=cfg.audit_log_path)
    events = al.get_recent_events(limit=limit * 5)  # over-fetch then filter

    if category:
        events = [e for e in events if e.get("category") == category]

    events = events[-limit:]

    if not events:
        msg = "No audit events found"
        msg += f" in category [bold]{category}[/]" if category else ""
        msg += "."
        console.print(f"[dim]{msg}[/]")
        return

    table = Table(
        title=f"Recent Audit Events (last {len(events)})",
        show_lines=True,
    )
    table.add_column("Timestamp", style="dim")
    table.add_column("Event Type")
    table.add_column("Category")
    table.add_column("Result", justify="center")
    table.add_column("Detail")

    result_styles = {"allow": "green", "deny": "red", "error": "yellow"}

    for event in events:
        result = event.get("result", "")
        style = result_styles.get(result, "white")
        result_text = Text(result, style=style)
        detail_raw = event.get("detail", {})
        detail_str = (
            json.dumps(detail_raw, separators=(",", ":"))
            if isinstance(detail_raw, dict)
            else str(detail_raw)
        )
        table.add_row(
            event.get("timestamp", "")[:19],
            event.get("event_type", ""),
            event.get("category", ""),
            result_text,
            detail_str[:80] + ("…" if len(detail_str) > 80 else ""),
        )

    console.print(table)


@audit.command("verify")
@click.option(
    "--limit",
    default=0,
    show_default=True,
    help="Show at most this many non-valid lines (0 = show all).",
)
@click.pass_context
def audit_verify(ctx: click.Context, limit: int) -> None:
    """Verify Ed25519 signatures on every line of the audit log (SR-1.1).

    Recomputes each persisted event's signature against the agent's
    identity public key and reports "valid" / "tampered" / "unsigned" /
    "malformed" per line. A "tampered" line means the recorded
    session_id, task_id, event_type, category, result, detail, or
    policy_rule was changed after the event was written -- this is the
    actual detection mechanism the "every audit event signed" claim
    depends on; signing alone provides no protection without it.
    """
    from missy.observability.audit_logger import verify_audit_log
    from missy.security.identity import AgentIdentity

    cfg = _load_subsystems(ctx.obj["config_path"])
    try:
        identity = AgentIdentity.load_or_generate()
    except Exception as exc:
        _print_error(f"Could not load agent identity: {exc}")
        sys.exit(1)

    results = verify_audit_log(cfg.audit_log_path, identity)
    if not results:
        console.print("[dim]Audit log is empty or does not exist.[/]")
        return

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    style_by_status = {
        "valid": "green",
        "tampered": "bold red",
        "unsigned": "yellow",
        "malformed": "red",
    }
    summary = "  ".join(
        f"[{style_by_status.get(status, 'white')}]{status}: {count}[/]"
        for status, count in sorted(counts.items())
    )
    console.print(f"Verified {len(results)} line(s) — {summary}")

    # Per-line signatures alone catch content tampering but not
    # reordering/deletion -- two validly-signed lines swapped in
    # position would report every line "valid" above. chain_ok surfaces
    # that separately: it's False only when a line's prev_chain_hash
    # doesn't match the actual preceding line's hash.
    broken_chain = [r for r in results if r.chain_ok is False]
    if broken_chain:
        console.print(
            f"[bold red]chain: {len(broken_chain)} line(s) out of sequence "
            "relative to their originally-written order (reordering/deletion "
            "detected despite individually valid signatures)[/]"
        )

    non_valid = [r for r in results if r.status != "valid"]
    if not non_valid and not broken_chain:
        _print_success("Every signed line verified intact. No tampering detected.")
        return

    shown_ids = {r.line_number for r in non_valid} | {r.line_number for r in broken_chain}
    shown_results = [r for r in results if r.line_number in shown_ids]
    shown = shown_results if limit <= 0 else shown_results[:limit]
    table = Table(title="Non-valid or out-of-sequence lines", show_lines=True)
    table.add_column("Line", justify="right")
    table.add_column("Status")
    table.add_column("Chain")
    table.add_column("Event Type")
    for r in shown:
        chain_text = "—" if r.chain_ok is None else ("ok" if r.chain_ok else "broken")
        chain_style = "dim" if r.chain_ok is None else ("green" if r.chain_ok else "bold red")
        table.add_row(
            str(r.line_number),
            Text(r.status, style=style_by_status.get(r.status, "white")),
            Text(chain_text, style=chain_style),
            r.event_type or "",
        )
    console.print(table)

    if any(r.status == "tampered" for r in non_valid) or broken_chain:
        sys.exit(1)


# ---------------------------------------------------------------------------
# missy logs
# ---------------------------------------------------------------------------


@cli.group("logs", invoke_without_command=True)
@click.pass_context
def logs_group(ctx: click.Context) -> None:
    """Application log commands."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(logs_path_cmd)


@logs_group.command("path")
@click.pass_context
def logs_path_cmd(ctx: click.Context) -> None:
    """Print the configured application log path."""
    cfg = _load_subsystems(ctx.obj["config_path"])
    console.print(str(_app_log_path(cfg)))


@logs_group.command("tail")
@click.option("--limit", default=80, show_default=True, help="Maximum log lines to show.")
@click.pass_context
def logs_tail_cmd(ctx: click.Context, limit: int) -> None:
    """Show recent application log lines."""
    cfg = _load_subsystems(ctx.obj["config_path"])
    log_path = _app_log_path(cfg)
    if not log_path.exists():
        console.print(f"[dim]No application log found at {log_path}.[/]")
        return

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        _print_error(f"Could not read application log {log_path}: {exc}")
        sys.exit(1)

    for line in lines[-max(0, limit) :]:
        console.print(line)


# ---------------------------------------------------------------------------
# missy providers
# ---------------------------------------------------------------------------

# Shared by `missy providers switch`, `missy discord pairing ...`, and
# `missy approvals ...` (all talk to the running gateway's Web API from a
# separate CLI process).
_APPROVALS_HOST_OPTION = click.option(
    "--host", default="127.0.0.1", show_default=True, help="Gateway API host."
)
_APPROVALS_PORT_OPTION = click.option(
    "--port", default=8080, type=int, show_default=True, help="Gateway API port."
)
_APPROVALS_API_KEY_OPTION = click.option(
    "--api-key",
    envvar="MISSY_API_KEY",
    default="",
    help="API key for authentication (falls back to ~/.missy/secrets/web_console.key).",
)


def _resolve_approvals_api_key(api_key: str) -> str:
    if api_key:
        return api_key
    try:
        key_path = Path("~/.missy/secrets/web_console.key").expanduser()
        if key_path.exists():
            return key_path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return ""


# Must match `missy.api.operator_controls._CONTROL_PROVIDER_SET_DEFAULT`.
_PROVIDER_SET_DEFAULT_CONTROL_ID = "provider.set_default"


@cli.group("providers", invoke_without_command=True)
@click.pass_context
def providers_group(ctx: click.Context) -> None:
    """Manage AI providers (list, switch)."""
    if ctx.invoked_subcommand is None:
        # Bare `missy providers` → list (backward compatible)
        ctx.invoke(providers_list_cmd)


@providers_group.command("list")
@click.pass_context
def providers_list_cmd(ctx: click.Context) -> None:
    """List configured AI providers and their availability."""
    from missy.providers.registry import get_registry

    cfg = _load_subsystems(ctx.obj["config_path"])

    if not cfg.providers:
        console.print("[dim]No providers configured.[/]")
        return

    registry = get_registry()

    table = Table(title="Configured Providers", show_lines=True)
    table.add_column("Name", style="bold")
    table.add_column("Model")
    table.add_column("Base URL")
    table.add_column("Timeout", justify="right")
    table.add_column("Available", justify="center")
    table.add_column("Balancing")

    for key, provider_cfg in cfg.providers.items():
        provider = registry.get(key)
        if provider is not None:
            try:
                available = provider.is_available()
            except Exception:
                available = False
            avail_text = Text("yes", style="green") if available else Text("no", style="red")
        else:
            avail_text = Text("not loaded", style="dim")

        if getattr(provider, "is_multi_account", False):
            balancing_text = Text(f"round_robin ({provider.account_count} accounts)", style="cyan")
        elif len(provider_cfg.api_keys) > 1:
            balancing_text = Text(f"failover ({len(provider_cfg.api_keys)} keys)", style="dim")
        else:
            balancing_text = Text("—", style="dim")

        table.add_row(
            key,
            provider_cfg.model,
            provider_cfg.base_url or "[dim]—[/]",
            f"{provider_cfg.timeout}s",
            avail_text,
            balancing_text,
        )

    console.print(table)


@providers_group.command("switch")
@click.argument("name")
@_APPROVALS_HOST_OPTION
@_APPROVALS_PORT_OPTION
@_APPROVALS_API_KEY_OPTION
@click.pass_context
def providers_switch(ctx: click.Context, name: str, host: str, port: int, api_key: str) -> None:
    """Switch the active provider to NAME.

    If a `missy gateway start` daemon is reachable at --host/--port, this
    switches *that daemon's* live default provider via its Web API (the only
    process whose provider selection persists across subsequent requests).
    Otherwise there is no running daemon to update, and this command falls
    back to a local, single-process registry mutation with no lasting
    effect (there is no ``default_provider`` config field to persist a
    choice to) -- use ``--provider NAME`` on ``missy ask``/``missy run`` for
    a one-off override instead.
    """
    import httpx

    resolved_key = _resolve_approvals_api_key(api_key)
    url = f"http://{host}:{port}/api/v1/controls/{_PROVIDER_SET_DEFAULT_CONTROL_ID}"
    headers = {"X-API-Key": resolved_key} if resolved_key else {}
    body = {"target": name, "confirm": f"set-default:{name}"}

    resp = None
    try:
        resp = httpx.post(url, json=body, headers=headers, timeout=3.0)
    except httpx.ConnectError:
        resp = None
    except Exception as exc:
        _print_error(f"Could not reach gateway API: {exc}")
        sys.exit(1)

    if resp is not None:
        if resp.status_code == 200:
            _print_success(
                f"Active provider switched to [bold]{name}[/] on the running gateway daemon."
            )
            return
        if resp.status_code == 401:
            _print_error(
                "Authentication required.",
                hint="Pass --api-key or ensure ~/.missy/secrets/web_console.key is readable.",
            )
            sys.exit(1)
        try:
            message = resp.json().get("error", "")
        except Exception:
            message = ""
        _print_error(message or f"Gateway API responded with HTTP {resp.status_code}.")
        sys.exit(1)

    from missy.providers.registry import get_registry

    _load_subsystems(ctx.obj["config_path"])
    registry = get_registry()

    try:
        registry.set_default(name)
    except ValueError as exc:
        _print_error(str(exc))
        sys.exit(1)

    _print_success(
        f"Active provider switched to [bold]{name}[/] for this process only "
        f"(no gateway daemon detected at http://{host}:{port} -- this does not persist; "
        "pass --provider to `missy ask`/`missy run`, or run `missy gateway start` first "
        "for a durable switch)."
    )


@providers_group.command("auth")
@click.argument("name", default="openai", required=False)
@click.option(
    "--method",
    type=click.Choice(["api-key", "oauth"], case_sensitive=False),
    default="api-key",
    show_default=True,
    help="OpenAI auth method to refresh.",
)
@click.option("--api-key", default=None, help="OpenAI API key. Omit to be prompted.")
@click.option(
    "--api-key-env",
    default=None,
    help="Environment variable containing the OpenAI API key; stored as a $ENV reference.",
)
@click.option("--model", default=None, help='Model selector to store, e.g. "auto" or "gpt-5.5".')
@click.option("--no-verify", is_flag=True, default=False, help="Skip the OpenAI verification call.")
@click.pass_context
def providers_auth(
    ctx: click.Context,
    name: str,
    method: str,
    api_key: str | None,
    api_key_env: str | None,
    model: str | None,
    no_verify: bool,
) -> None:
    """Refresh provider credentials in the config file.

    Currently supports OpenAI API-key auth and OpenAI OAuth/Codex auth.
    """
    method = method.lower()
    if name not in {"openai", "openai-codex"}:
        _print_error("Only OpenAI re-auth is currently supported.")
        sys.exit(1)

    import os

    import yaml

    config_file = Path(ctx.obj["config_path"]).expanduser()
    if not config_file.exists():
        _print_error(
            f"Config file not found: {config_file}",
            hint="Run `missy init` or pass --config to the file you want to update.",
        )
        sys.exit(1)

    try:
        raw = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        _print_error(f"Could not read config YAML: {exc}")
        sys.exit(1)

    if not isinstance(raw, dict):
        _print_error("Config root must be a mapping.")
        sys.exit(1)

    providers = raw.setdefault("providers", {})
    if not isinstance(providers, dict):
        _print_error("Config providers section must be a mapping.")
        sys.exit(1)

    provider_key = name
    provider_cfg = providers.get(provider_key)
    if not isinstance(provider_cfg, dict):
        provider_cfg = None
        for key, value in providers.items():
            if isinstance(value, dict) and value.get("name") == name:
                provider_key = key
                provider_cfg = value
                break
    if provider_cfg is None:
        provider_key = "openai-codex" if method == "oauth" else "openai"
        provider_cfg = {}
        providers[provider_key] = provider_cfg

    verify_secret: str | None = None
    if method == "oauth":
        from missy.cli.oauth import run_openai_oauth
        from missy.cli.wizard import _PROVIDERS

        console.print("[dim]Starting OpenAI OAuth flow...[/]")
        token = run_openai_oauth()
        if not token:
            _print_error("OpenAI OAuth did not return a token.")
            sys.exit(1)
        provider_cfg["name"] = "openai-codex"
        # run_openai_oauth writes ~/.missy/secrets/openai-oauth.json with the
        # refresh token. Keep short-lived access tokens out of config.yaml.
        provider_cfg.pop("api_key", None)
        provider_cfg["model"] = (
            model or provider_cfg.get("model") or _PROVIDERS["openai-codex"]["models"]["primary"]
        )
    else:
        provider_cfg["name"] = "openai"
        if api_key_env:
            verify_secret = os.environ.get(api_key_env)
            if not verify_secret and not no_verify:
                _print_error(
                    f"Environment variable {api_key_env!r} is not set.",
                    hint="Set it first, pass --no-verify, or use --api-key.",
                )
                sys.exit(1)
            provider_cfg["api_key"] = f"${api_key_env}"
        else:
            if api_key is None:
                api_key = click.prompt("OpenAI API key", hide_input=True)
            if "=" in api_key:
                api_key = api_key.split("=", 1)[1].strip()
            api_key = api_key.strip()
            if not api_key:
                _print_error("OpenAI API key cannot be empty.")
                sys.exit(1)
            provider_cfg["api_key"] = api_key
            verify_secret = api_key
        provider_cfg["model"] = model or provider_cfg.get("model") or "auto"

    provider_cfg.setdefault("timeout", 30)

    network = raw.setdefault("network", {})
    if isinstance(network, dict):
        presets = network.setdefault("presets", [])
        if isinstance(presets, list) and "openai" not in presets:
            presets.append("openai")

    if method == "api-key" and not no_verify and verify_secret:
        from missy.cli.wizard import _verify_openai

        console.print("[dim]Verifying OpenAI credentials...[/]")
        if not _verify_openai(verify_secret):
            _print_error(
                "OpenAI verification failed; config was not modified.",
                hint="Pass --no-verify to save the credential anyway.",
            )
            sys.exit(1)

    try:
        from missy.cli.wizard import _write_config_atomic
        from missy.config.plan import backup_config

        backup_path = backup_config(config_file)
        content = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
        _write_config_atomic(config_file, content)
    except Exception as exc:
        _print_error(f"Failed to update config: {exc}")
        sys.exit(1)

    _print_success(
        f"Updated [bold]{provider_key}[/] auth in [bold]{config_file}[/].\n"
        f"Backup: [dim]{backup_path}[/]"
    )


# ---------------------------------------------------------------------------
# missy skills
# ---------------------------------------------------------------------------


@cli.group("skills", invoke_without_command=True)
@click.pass_context
def skills_group(ctx: click.Context) -> None:
    """Manage skills (list, scan)."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(skills_list_cmd)


@skills_group.command("list")
@click.pass_context
def skills_list_cmd(ctx: click.Context) -> None:
    """List all registered skills."""
    from missy.skills.registry import SkillRegistry

    _load_subsystems(ctx.obj["config_path"])

    # Build a fresh registry — built-in skills would be registered by the
    # application bootstrap; here we report the current process state.
    registry = SkillRegistry()
    skill_names = registry.list_skills()

    if not skill_names:
        console.print("[dim]No skills are currently registered.[/]")
        console.print("[dim]Skills are registered programmatically at application startup.[/]")
        return

    table = Table(title="Registered Skills")
    table.add_column("Name", style="bold")

    for name in skill_names:
        table.add_row(name)

    console.print(table)


@skills_group.command("scan")
@click.option(
    "--path",
    default="~/.missy/skills",
    help="Directory to scan for SKILL.md files.",
    show_default=True,
)
@click.pass_context
def skills_scan_cmd(ctx: click.Context, path: str) -> None:
    """Scan a directory for SKILL.md files and list discovered skills."""
    from missy.skills.discovery import SkillDiscovery

    discovery = SkillDiscovery()
    manifests = discovery.scan_directory(path)

    if not manifests:
        console.print(f"[dim]No SKILL.md files found in {path}[/]")
        return

    table = Table(title=f"Discovered Skills ({path})")
    table.add_column("Name", style="bold")
    table.add_column("Version")
    table.add_column("Author")
    table.add_column("Description")
    table.add_column("Tools")

    for m in manifests:
        table.add_row(
            m.name,
            m.version or "[dim]-[/]",
            m.author or "[dim]-[/]",
            m.description[:60] + ("..." if len(m.description) > 60 else ""),
            ", ".join(m.tools) if m.tools else "[dim]-[/]",
        )

    console.print(table)


@skills_group.command("promote")
@click.option(
    "--threshold",
    default=3,
    show_default=True,
    help="Minimum success count for a playbook pattern to be promoted.",
)
@click.option(
    "--proposals-dir",
    default=None,
    help="Where to write SKILL.md proposals (default ~/.missy/skills/proposals).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be promoted without writing files.",
)
@click.pass_context
def skills_promote_cmd(
    ctx: click.Context, threshold: int, proposals_dir: str | None, dry_run: bool
) -> None:
    """Materialize promotable playbook patterns into SKILL.md proposals (F20).

    Patterns that have succeeded ``--threshold`` times are written as SKILL.md
    proposal drafts an operator can review and move into the active skills
    directory. Without ``--dry-run`` the promoted patterns are marked so they
    are not re-materialized on the next run.
    """
    from missy.agent.playbook import Playbook

    _load_subsystems(ctx.obj["config_path"])
    playbook = Playbook()
    results = playbook.promote_to_skills(
        threshold=threshold, proposals_dir=proposals_dir, dry_run=dry_run
    )

    if not results:
        console.print(
            f"[dim]No playbook patterns with success_count >= {threshold} await promotion.[/]"
        )
        return

    verb = "Would promote" if dry_run else "Promoted"
    t = Table(title=f"{verb} {len(results)} pattern(s)")
    t.add_column("Task type")
    t.add_column("Successes", justify="right")
    t.add_column("Proposal path" if not dry_run else "Pattern id")
    for r in results:
        t.add_row(
            r["task_type"],
            str(r["success_count"]),
            r["path"] or r["pattern_id"],
        )
    console.print(t)
    if not dry_run:
        console.print(
            "[dim]Review each SKILL.md and move approved ones into "
            "~/.missy/skills to activate them.[/]"
        )


# ---------------------------------------------------------------------------
# missy plugins
# ---------------------------------------------------------------------------


@cli.command("plugins")
@click.pass_context
def plugins_list(ctx: click.Context) -> None:
    """List loaded plugins and their status."""
    from missy.plugins.loader import init_plugin_loader

    cfg = _load_subsystems(ctx.obj["config_path"])
    loader = init_plugin_loader(cfg)
    manifests = loader.list_plugins()

    # Show policy status even if no plugins are loaded.
    enabled_label = (
        Text("enabled", style="green")
        if cfg.plugins.enabled
        else Text("disabled (secure default)", style="red")
    )
    console.print(f"Plugin system: {enabled_label}")

    if cfg.plugins.allowed_plugins:
        console.print(f"Allowed plugins: [bold]{', '.join(cfg.plugins.allowed_plugins)}[/]")
    else:
        console.print("[dim]No plugins on the allow-list.[/]")

    if not manifests:
        console.print("\n[dim]No plugins currently loaded.[/]")
        return

    table = Table(title="Loaded Plugins", show_lines=True)
    table.add_column("Name", style="bold")
    table.add_column("Version")
    table.add_column("Description")
    table.add_column("Enabled", justify="center")

    for manifest in manifests:
        enabled_text = (
            Text("yes", style="green") if manifest.get("enabled") else Text("no", style="red")
        )
        table.add_row(
            manifest.get("name", ""),
            manifest.get("version", "[dim]—[/]"),
            manifest.get("description", "[dim]—[/]"),
            enabled_text,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# missy discord
# ---------------------------------------------------------------------------


@cli.group()
def discord() -> None:
    """Discord channel commands (status, diagnostics, probe, register-commands, audit)."""


@discord.command("status")
@click.pass_context
def discord_status(ctx: click.Context) -> None:
    """Show Discord connection status and bot user info.

    Reads the Discord configuration from the Missy config file and prints
    a summary of each configured account.
    """
    cfg = _load_subsystems(ctx.obj["config_path"])
    discord_cfg = cfg.discord

    if discord_cfg is None or not discord_cfg.accounts:
        console.print("[dim]No Discord accounts configured.[/]")
        return

    enabled_text = "[green]enabled[/]" if discord_cfg.enabled else "[red]disabled[/]"
    console.print(f"Discord integration: {enabled_text}")

    table = Table(title="Discord Accounts", show_lines=True)
    table.add_column("Token Env Var", style="bold")
    table.add_column("Application ID")
    table.add_column("DM Policy")
    table.add_column("Guilds Configured", justify="right")
    table.add_column("Ignore Bots", justify="center")

    for account in discord_cfg.accounts:
        table.add_row(
            account.token_env_var,
            account.application_id or "[dim]—[/]",
            account.dm_policy.value,
            str(len(account.guild_policies)),
            "yes" if account.ignore_bots else "no",
        )

    console.print(table)


@discord.command("diagnostics")
@click.option(
    "--limit",
    default=5,
    show_default=True,
    help="Recent Discord lifecycle events to include.",
)
@click.pass_context
def discord_diagnostics(ctx: click.Context, limit: int) -> None:
    """Show Discord configuration, policy, tool, and voice readiness."""
    from missy.channels.discord.voice_binding import list_voice_bindings
    from missy.observability.audit_logger import AuditLogger
    from missy.policy.tool_policy_pipeline import (
        MISSY_DISCORD_TOOLS,
        build_configured_tool_policy_layers,
        collect_tool_policy_groups,
        resolve_tool_policy,
    )

    cfg = _load_subsystems(ctx.obj["config_path"])
    discord_cfg = cfg.discord

    if discord_cfg is None or not discord_cfg.accounts:
        console.print("[dim]No Discord accounts configured.[/]")
        return

    console.print("[bold]Discord Diagnostics[/]")
    enabled_text = "[green]enabled[/]" if discord_cfg.enabled else "[red]disabled[/]"
    console.print(f"Integration: {enabled_text}")

    account_table = Table(title="Accounts", show_lines=True)
    account_table.add_column("Index", justify="right")
    account_table.add_column("Token")
    account_table.add_column("Application ID")
    account_table.add_column("DM Policy")
    account_table.add_column("Guilds", justify="right")
    account_table.add_column("Routing")

    for idx, account in enumerate(discord_cfg.accounts):
        token_status = "[green]present[/]" if account.resolve_token() else "[yellow]missing[/]"
        guild_policies = getattr(account, "guild_policies", {}) or {}
        mention_required = sum(1 for policy in guild_policies.values() if policy.require_mention)
        routing_bits = [
            "ignore_bots=yes" if account.ignore_bots else "ignore_bots=no",
            f"mention_required={mention_required}",
        ]
        account_table.add_row(
            str(idx),
            f"{account.token_env_var}: {token_status}",
            account.application_id or "[yellow]missing[/]",
            account.dm_policy.value,
            str(len(guild_policies)),
            ", ".join(routing_bits),
        )
    console.print(account_table)

    def _string_set(value: Any) -> set[str]:
        if not isinstance(value, (list, tuple, set)):
            return set()
        return {str(item) for item in value}

    network = getattr(cfg, "network", None)
    allowed_domains = _string_set(getattr(network, "allowed_domains", []))
    allowed_hosts = _string_set(getattr(network, "allowed_hosts", []))
    discord_hosts = _string_set(getattr(network, "discord_allowed_hosts", []))
    network_values = allowed_domains | allowed_hosts | discord_hosts

    policy_table = Table(title="Policy Readiness", show_lines=True)
    policy_table.add_column("Check")
    policy_table.add_column("Status")
    policy_table.add_column("Hint")

    def _policy_row(name: str, ok: bool, hint: str) -> None:
        policy_table.add_row(
            name,
            Text("ok" if ok else "needs attention", style="green" if ok else "yellow"),
            hint,
        )

    _policy_row(
        "REST host discord.com",
        "discord.com" in network_values,
        "Add discord.com to network.allowed_domains or network.discord_allowed_hosts.",
    )
    _policy_row(
        "Gateway host gateway.discord.gg",
        "gateway.discord.gg" in network_values,
        "Add gateway.discord.gg to network.allowed_domains or network.discord_allowed_hosts.",
    )

    layers = build_configured_tool_policy_layers(
        capability_mode="discord",
        global_policy=getattr(cfg, "tools", None),
        agent_policy=None,
        sandbox_policy=getattr(getattr(cfg, "sandbox", None), "tools", None),
    )
    groups = collect_tool_policy_groups(getattr(cfg, "tools", None))
    decision = resolve_tool_policy(MISSY_DISCORD_TOOLS, layers, groups=groups)
    visible_tools = set(decision.tools)
    voice_tools = {
        "discord_voice_join",
        "discord_voice_leave",
        "discord_voice_say",
        "discord_voice_status",
    }
    _policy_row(
        "Discord voice tools visible",
        voice_tools.issubset(visible_tools),
        "Discord capability mode should expose discord_voice_* unless config policy denies them.",
    )
    if decision.warnings:
        _policy_row("Tool policy warnings", False, "; ".join(decision.warnings[:3]))

    console.print(policy_table)

    bindings = list_voice_bindings()
    voice_table = Table(title="Runtime Voice Bindings", show_lines=True)
    voice_table.add_column("Account ID")
    voice_table.add_column("Guild ID")
    voice_table.add_column("Ready", justify="center")
    voice_table.add_column("Listen", justify="center")
    voice_table.add_column("Speak", justify="center")
    if bindings:
        for binding in bindings:
            voice_table.add_row(
                str(binding.get("account_id") or "[dim]unknown[/]"),
                str(binding.get("guild_id") or "[dim]unknown[/]"),
                "yes" if binding.get("ready") else "no",
                "yes" if binding.get("can_listen") else "no",
                "yes" if binding.get("can_speak") else "no",
            )
    else:
        voice_table.add_row("[dim]-[/]", "[dim]-[/]", "no", "no", "no")
    console.print(voice_table)

    with contextlib.suppress(Exception):
        al = AuditLogger(log_path=cfg.audit_log_path)
        all_recent_discord = [
            event
            for event in al.get_recent_events(limit=max(limit * 10, 20))
            if str(event.get("event_type", "")).startswith("discord.")
        ]
        lifecycle_event_types = {
            "discord.gateway.heartbeat_sent",
            "discord.gateway.heartbeat_ack",
            "discord.gateway.reconnect_requested",
            "discord.gateway.invalid_session",
            "discord.gateway.resume_sent",
            "discord.gateway.session_resumed",
            "discord.slash_commands.registered",
            "discord.slash_commands.registration_failed",
        }
        lifecycle_counts: dict[str, int] = {}
        for event in all_recent_discord:
            event_type = str(event.get("event_type", ""))
            if event_type in lifecycle_event_types:
                lifecycle_counts[event_type] = lifecycle_counts.get(event_type, 0) + 1
        lifecycle_table = Table(title="Recent Lifecycle Signals", show_lines=True)
        lifecycle_table.add_column("Signal")
        lifecycle_table.add_column("Count", justify="right")
        lifecycle_table.add_column("Operator Meaning")
        lifecycle_rows = [
            (
                "heartbeat",
                lifecycle_counts.get("discord.gateway.heartbeat_sent", 0)
                + lifecycle_counts.get("discord.gateway.heartbeat_ack", 0),
                "Gateway heartbeat traffic observed.",
            ),
            (
                "reconnect/resume",
                lifecycle_counts.get("discord.gateway.reconnect_requested", 0)
                + lifecycle_counts.get("discord.gateway.resume_sent", 0)
                + lifecycle_counts.get("discord.gateway.session_resumed", 0),
                "Gateway reconnect or resume activity observed.",
            ),
            (
                "invalid-session",
                lifecycle_counts.get("discord.gateway.invalid_session", 0),
                "Discord invalidated a Gateway session.",
            ),
            (
                "slash-registration",
                lifecycle_counts.get("discord.slash_commands.registered", 0)
                + lifecycle_counts.get("discord.slash_commands.registration_failed", 0),
                "Slash command registration attempt observed.",
            ),
        ]
        for name, count, meaning in lifecycle_rows:
            lifecycle_table.add_row(name, str(count), meaning)
        console.print(lifecycle_table)

        recent = all_recent_discord[-limit:]
        if recent:
            audit_table = Table(title=f"Recent Discord Events (last {len(recent)})")
            audit_table.add_column("Timestamp", style="dim")
            audit_table.add_column("Event")
            audit_table.add_column("Result")
            audit_table.add_column("Detail")
            for event in recent:
                detail = event.get("detail", {})
                detail_str = (
                    json.dumps(detail, separators=(",", ":"))
                    if isinstance(detail, dict)
                    else str(detail)
                )
                audit_table.add_row(
                    str(event.get("timestamp", ""))[:19],
                    str(event.get("event_type", "")),
                    str(event.get("result", "")),
                    detail_str[:80] + ("..." if len(detail_str) > 80 else ""),
                )
            console.print(audit_table)


@discord.command("probe")
@click.pass_context
def discord_probe(ctx: click.Context) -> None:
    """Test Discord API connectivity and bot token validity.

    Attempts to call GET /users/@me for each configured account and reports
    whether the token is valid and the network policy allows the request.
    """
    from missy.channels.discord.rest import DiscordRestClient
    from missy.gateway.client import create_client

    cfg = _load_subsystems(ctx.obj["config_path"])
    discord_cfg = cfg.discord

    if discord_cfg is None or not discord_cfg.accounts:
        console.print("[dim]No Discord accounts configured.[/]")
        return

    for idx, account in enumerate(discord_cfg.accounts):
        token = account.resolve_token()
        label = f"Account {idx + 1} ({account.token_env_var})"

        if not token:
            err_console.print(
                f"[yellow]{label}[/]: env var [bold]{account.token_env_var}[/] is not set."
            )
            continue

        try:
            http = create_client(session_id="discord-probe", task_id=f"account-{idx}")
            rest = DiscordRestClient(bot_token=token, http_client=http)
            user = rest.get_current_user()
            console.print(
                f"[green]{label}[/]: connected as [bold]{user.get('username')}[/]"
                f"#{user.get('discriminator', '0')} (id={user.get('id')})"
            )
        except Exception as exc:
            err_console.print(f"[red]{label}[/]: probe failed — {exc}")


@discord.command("register-commands")
@click.option("--guild-id", default=None, help="Guild ID for guild-scoped registration.")
@click.option(
    "--account-index",
    default=0,
    show_default=True,
    help="Index of the account in the config accounts list.",
)
@click.pass_context
def discord_register_commands(
    ctx: click.Context,
    guild_id: str | None,
    account_index: int,
) -> None:
    """Register slash commands with Discord.

    Registers /ask, /status, /model, and /help.  When --guild-id is
    provided the commands are registered as guild-scoped (instant
    propagation); without it they are registered globally.
    """
    from missy.channels.discord.commands import SLASH_COMMANDS
    from missy.channels.discord.rest import DiscordRestClient
    from missy.gateway.client import create_client

    cfg = _load_subsystems(ctx.obj["config_path"])
    discord_cfg = cfg.discord

    if discord_cfg is None or not discord_cfg.accounts:
        _print_error("No Discord accounts configured.")
        sys.exit(1)

    if account_index >= len(discord_cfg.accounts):
        _print_error(
            f"Account index {account_index} out of range "
            f"(only {len(discord_cfg.accounts)} account(s) configured)."
        )
        sys.exit(1)

    account = discord_cfg.accounts[account_index]
    token = account.resolve_token()
    if not token:
        _print_error(
            f"Environment variable {account.token_env_var!r} is not set.",
            hint="Export the bot token before running this command.",
        )
        sys.exit(1)

    if not account.application_id:
        _print_error(
            "No application_id configured for this account.",
            hint="Set application_id in your Discord config.",
        )
        sys.exit(1)

    try:
        http = create_client(session_id="discord-register", task_id="commands")
        rest = DiscordRestClient(bot_token=token, http_client=http)
        registered = rest.register_slash_commands(
            application_id=account.application_id,
            commands=SLASH_COMMANDS,
            guild_id=guild_id,
        )
    except Exception as exc:
        _print_error(f"Failed to register commands: {exc}")
        sys.exit(1)

    scope = f"guild {guild_id}" if guild_id else "global"
    _print_success(
        f"Registered {len(registered)} command(s) [{scope}]:\n"
        + "\n".join(f"  /{cmd.get('name', '?')}" for cmd in registered)
    )


@discord.command("audit")
@click.option(
    "--limit",
    default=50,
    show_default=True,
    help="Maximum number of events to show.",
)
@click.pass_context
def discord_audit(ctx: click.Context, limit: int) -> None:
    """Show recent Discord-related audit events from the log."""
    from missy.observability.audit_logger import AuditLogger

    cfg = _load_subsystems(ctx.obj["config_path"])
    al = AuditLogger(log_path=cfg.audit_log_path)
    all_events = al.get_recent_events(limit=limit * 10)

    discord_events = [e for e in all_events if str(e.get("event_type", "")).startswith("discord.")][
        -limit:
    ]

    if not discord_events:
        console.print("[dim]No Discord audit events found.[/]")
        return

    table = Table(title=f"Discord Audit Events (last {len(discord_events)})", show_lines=True)
    table.add_column("Timestamp", style="dim")
    table.add_column("Event Type")
    table.add_column("Result", justify="center")
    table.add_column("Detail")

    result_styles = {"allow": "green", "deny": "red", "error": "yellow"}

    for event in discord_events:
        result = event.get("result", "")
        style = result_styles.get(result, "white")
        result_text = Text(result, style=style)
        detail_raw = event.get("detail", {})
        detail_str = (
            json.dumps(detail_raw, separators=(",", ":"))
            if isinstance(detail_raw, dict)
            else str(detail_raw)
        )
        table.add_row(
            event.get("timestamp", "")[:19],
            event.get("event_type", ""),
            result_text,
            detail_str[:80] + ("…" if len(detail_str) > 80 else ""),
        )

    console.print(table)


@discord.group("pairing")
def discord_pairing() -> None:
    """Manage pending Discord DM pairing requests (SR-1.12).

    Pairing decisions can never be made from in-band DM content (any
    unpaired stranger could otherwise grant themselves access by
    messaging accept/deny commands to the bot). These commands are the
    real, authenticated approval surface an operator uses instead --
    they call the running gateway's Web API, since a separate `missy`
    CLI invocation cannot see the gateway process's in-memory pairing
    state directly.
    """


@discord_pairing.command("list")
@_APPROVALS_HOST_OPTION
@_APPROVALS_PORT_OPTION
@_APPROVALS_API_KEY_OPTION
def discord_pairing_list(host: str, port: int, api_key: str) -> None:
    """List pending Discord DM pairing requests from a running gateway."""
    import httpx

    resolved_key = _resolve_approvals_api_key(api_key)
    url = f"http://{host}:{port}/api/v1/discord/pairing"
    headers = {"X-API-Key": resolved_key} if resolved_key else {}

    try:
        resp = httpx.get(url, headers=headers, timeout=3.0)
    except httpx.ConnectError:
        console.print(
            f"[dim]No active gateway session at [bold]http://{host}:{port}[/] — "
            "pairing requests are only visible while `missy gateway start` is running.[/]"
        )
        return
    except Exception as exc:
        _print_error(f"Could not reach gateway API: {exc}")
        return

    if resp.status_code == 401:
        _print_error(
            "Authentication required.",
            hint="Pass --api-key or ensure ~/.missy/secrets/web_console.key is readable.",
        )
        return
    if resp.status_code != 200:
        _print_error(f"Gateway API responded with HTTP {resp.status_code}.")
        return

    pending = resp.json().get("data", {}).get("pending", [])
    if not pending:
        console.print("[dim]No pending Discord pairing requests.[/]")
        return

    table = Table(title="Pending Discord Pairing Requests", show_lines=True)
    table.add_column("Account", style="dim")
    table.add_column("User ID", style="bold")
    for item in pending:
        table.add_row(item.get("account", ""), item.get("user_id", ""))
    console.print(table)


def _resolve_discord_pairing(
    host: str, port: int, api_key: str, user_id: str, *, approve: bool
) -> None:
    import httpx

    resolved_key = _resolve_approvals_api_key(api_key)
    verb = "approve" if approve else "deny"
    # APPROVAL-003: "deny" + "d" reads as "denyd", not "denied".
    past_tense = "approved" if approve else "denied"
    url = f"http://{host}:{port}/api/v1/discord/pairing/{user_id}/{verb}"
    headers = {"X-API-Key": resolved_key} if resolved_key else {}

    try:
        resp = httpx.post(url, headers=headers, timeout=3.0)
    except httpx.ConnectError:
        _print_error(
            f"No active gateway session at http://{host}:{port}.",
            hint="Pairing requests are only processed while `missy gateway start` is running.",
        )
        sys.exit(1)
    except Exception as exc:
        _print_error(f"Could not reach gateway API: {exc}")
        sys.exit(1)

    if resp.status_code == 200:
        _print_success(f"Discord user {user_id!r} pairing {past_tense}.")
    elif resp.status_code == 404:
        _print_error(f"No pending pairing request for user {user_id!r}.")
        sys.exit(1)
    elif resp.status_code == 401:
        _print_error(
            "Authentication required.",
            hint="Pass --api-key or ensure ~/.missy/secrets/web_console.key is readable.",
        )
        sys.exit(1)
    else:
        _print_error(f"Gateway API responded with HTTP {resp.status_code}.")
        sys.exit(1)


@discord_pairing.command("approve")
@click.argument("user_id")
@_APPROVALS_HOST_OPTION
@_APPROVALS_PORT_OPTION
@_APPROVALS_API_KEY_OPTION
def discord_pairing_approve(user_id: str, host: str, port: int, api_key: str) -> None:
    """Approve a pending Discord DM pairing request (see `missy discord pairing list`)."""
    _resolve_discord_pairing(host, port, api_key, user_id, approve=True)


@discord_pairing.command("deny")
@click.argument("user_id")
@_APPROVALS_HOST_OPTION
@_APPROVALS_PORT_OPTION
@_APPROVALS_API_KEY_OPTION
def discord_pairing_deny(user_id: str, host: str, port: int, api_key: str) -> None:
    """Deny a pending Discord DM pairing request (see `missy discord pairing list`)."""
    _resolve_discord_pairing(host, port, api_key, user_id, approve=False)


# ---------------------------------------------------------------------------
# missy gateway
# ---------------------------------------------------------------------------


@cli.group()
def gateway() -> None:
    """Gateway / service-mode commands."""


@gateway.command("start")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address.")
@click.option("--port", default=8765, show_default=True, help="Bind port.")
@click.pass_context
def gateway_start(ctx: click.Context, host: str, port: int) -> None:
    """Start Missy in service mode (long-running agent loop).

    Runs the agent as a persistent service, processing tasks from all
    configured channels (Discord, etc.) until interrupted with Ctrl-C.
    """
    import signal

    cfg = _load_subsystems(ctx.obj["config_path"])
    console.print(
        Panel(
            f"[bold cyan]Missy Gateway[/] starting\n\n"
            f"  Host : [bold]{host}[/]\n"
            f"  Port : [bold]{port}[/]\n\n"
            "Press Ctrl-C to stop.",
            border_style="cyan",
        )
    )

    stop_event = False

    def _stop(signum: int, frame: Any) -> None:
        nonlocal stop_event
        stop_event = True
        console.print("\n[dim]Shutting down gateway...[/]")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # SR-2.2: shared ApprovalGate for the whole gateway process. Proactive
    # triggers gate through it (requires_confirmation defaults to True);
    # the Web API's /approvals endpoints (below) let an operator actually
    # respond to a pending request from another process, since this
    # gateway process's in-memory approval state is otherwise unreachable
    # from a separate `missy` CLI invocation.
    from missy.agent.approval import ApprovalGate

    def _approval_send(msg: str) -> None:
        console.print(f"[yellow]{msg}[/]")
        logger.info("Approval request: %s", msg)

    approval_gate = ApprovalGate(send_fn=_approval_send)

    # Start proactive manager if configured.
    proactive_manager = None
    # Populated below only when a proactive-trigger AgentRuntime is actually
    # constructed; used by the config hot-reload callback further down to
    # propagate a changed max_spend_usd to this runtime too.
    _proactive_runtime = None
    try:
        if hasattr(cfg, "proactive") and cfg.proactive.enabled and cfg.proactive.triggers:
            from missy.agent.proactive import ProactiveManager, ProactiveTrigger

            triggers = [
                ProactiveTrigger(
                    name=t.name,
                    trigger_type=t.trigger_type,
                    enabled=t.enabled,
                    requires_confirmation=t.requires_confirmation,
                    prompt_template=t.prompt_template,
                    watch_path=t.watch_path,
                    watch_patterns=t.watch_patterns,
                    watch_recursive=t.watch_recursive,
                    disk_path=t.disk_path,
                    disk_threshold_pct=t.disk_threshold_pct,
                    load_threshold=t.load_threshold,
                    interval_seconds=t.interval_seconds,
                    cooldown_seconds=t.cooldown_seconds,
                )
                for t in cfg.proactive.triggers
            ]

            # Build a lightweight runtime for proactive prompts.
            try:
                from missy.agent.runtime import AgentConfig, AgentRuntime

                _provider_name = (
                    next(iter(cfg.providers), "anthropic") if cfg.providers else "anthropic"
                )
                _agent_cfg = AgentConfig(
                    provider=_provider_name,
                    max_spend_usd=getattr(cfg, "max_spend_usd", 0.0),
                    **_agent_tool_policy_kwargs(cfg),
                )
                _runtime = AgentRuntime(_agent_cfg)
                _proactive_runtime = _runtime

                def _proactive_callback(prompt: str, session_id: str) -> str:
                    return _runtime.run(prompt, session_id=session_id)

            except Exception as _rt_exc:
                logger.warning("proactive: could not create AgentRuntime: %s", _rt_exc)

                def _proactive_callback(prompt: str, session_id: str) -> str:  # type: ignore[misc]
                    logger.info("proactive prompt [%s]: %s", session_id, prompt[:200])
                    return ""

            proactive_manager = ProactiveManager(
                triggers=triggers,
                agent_callback=_proactive_callback,
                approval_gate=approval_gate,
            )
            proactive_manager.start()
            console.print(f"[green]Proactive manager started[/] ({len(triggers)} trigger(s))")
    except Exception as _pe:
        console.print(f"[yellow]Proactive manager failed to start: {_pe}[/]")

    # Build the shared agent runtime for all channels.
    from missy.agent.runtime import DISCORD_SYSTEM_PROMPT, AgentConfig, AgentRuntime

    _provider_name = next(iter(cfg.providers), "anthropic") if cfg.providers else "anthropic"
    # SR-4.7: thread the same real ApprovalGate constructed above (for
    # proactive triggers) into the agent runtimes so destructive/mutating
    # MCP tool calls have real confirmation infrastructure to block on,
    # instead of failing closed for lack of any gate at all.
    _agent_cfg = AgentConfig(
        provider=_provider_name,
        mcp_approval_gate=approval_gate,
        max_spend_usd=getattr(cfg, "max_spend_usd", 0.0),
        **_agent_tool_policy_kwargs(cfg),
    )
    _agent = AgentRuntime(_agent_cfg)

    # Discord-specific agent with filtered tools and appropriate system prompt.
    _discord_agent_cfg = AgentConfig(
        provider=_provider_name,
        system_prompt=DISCORD_SYSTEM_PROMPT,
        capability_mode="discord",
        mcp_approval_gate=approval_gate,
        max_spend_usd=getattr(cfg, "max_spend_usd", 0.0),
        **_agent_tool_policy_kwargs(cfg),
    )
    _discord_agent = AgentRuntime(_discord_agent_cfg)

    # Background subsystem health monitor. Fully built and tested
    # (missy/agent/watchdog.py), but had zero production callers anywhere
    # -- no CLI command or bootstrap path ever called .register()/.start()
    # on it, so the "background subsystem health monitor" this module's own
    # docstring advertises was inert in every real deployment: no operator
    # ever saw a watchdog.health_check audit event or an ERROR-level
    # "unhealthy" log for a real subsystem, silently, with no error
    # anywhere. Registers real, meaningful checks against the objects this
    # function already constructed.
    from missy.agent.watchdog import Watchdog

    def _check_provider_registry() -> bool:
        from missy.providers.registry import get_registry

        return bool(get_registry().get_available())

    def _check_memory_store() -> bool:
        store = getattr(_agent, "_memory_store", None)
        if store is None:
            # AgentRuntime._make_memory_store() always attempts
            # construction -- there is no config path that intentionally
            # disables memory -- so a None store here can only mean
            # construction failed (permissions, disk full, corruption).
            # The 5th tool-specific validation run found this previously
            # returned True ("nothing to monitor"), which silently masked
            # a genuine startup failure for the rest of the process's
            # lifetime: no audit event, no ERROR log, the bot kept
            # responding normally with zero memory persistence and the
            # Watchdog reported healthy indefinitely.
            return False
        store.get_session_turns("watchdog-healthcheck", limit=1)
        return True

    def _check_mcp_servers() -> bool:
        # McpManager.health_check() (restart any dead server, going through
        # the same digest-verification/approval-annotation path as an
        # initial add_server() call) was fully built and tested but had
        # zero production callers anywhere -- once an MCP server subprocess
        # died (crash, OOM-kill), it stayed dead for the rest of the
        # process's life: its tools kept being listed via all_tools() and
        # dispatched via call_tool(), which would simply fail against the
        # dead subprocess forever, with no auto-recovery ever attempted.
        # Piggybacking the restart-attempt onto this periodic check (rather
        # than a bespoke separate thread) reuses the same infrastructure
        # already wired in for provider_registry/memory_store above.
        mgr = getattr(_agent, "_mcp_manager", None)
        if mgr is None:
            return True  # no MCP servers configured; nothing to monitor
        mgr.health_check()
        return all(server["alive"] for server in mgr.list_servers())

    watchdog = Watchdog()
    watchdog.register("provider_registry", _check_provider_registry)
    watchdog.register("memory_store", _check_memory_store)
    watchdog.register("mcp_servers", _check_mcp_servers)
    watchdog.start()

    # Job scheduler. Fully built and tested (missy/scheduler/manager.py,
    # including per-job active-hours gating, retry backoff, and audit
    # events) but had zero production callers in gateway_start() -- the
    # persistent daemon process this systemd unit runs never constructed a
    # SchedulerManager at all, so a job added via `missy schedule add`
    # (itself just a separate CLI invocation that starts a private
    # SchedulerManager, mutates jobs.json, and immediately stops it again)
    # would never actually fire: nothing in the long-running gateway
    # process ever loaded jobs.json into a live, running APScheduler
    # instance. The Web TUI's scheduler pages and operator controls
    # (api/server.py's _handle_list_scheduled_jobs/_handle_create_scheduled_job,
    # api/operator_controls.py's scheduler.pause_job/resume_job/remove_job)
    # were also silently non-functional in every real deployment, since
    # they all resolve their SchedulerManager via
    # getattr(runtime, "_scheduler", None) and nothing ever set that
    # attribute on the AgentRuntime passed to ApiServer as runtime=_agent.
    scheduler_manager = None
    try:
        if cfg.scheduling.enabled:
            from missy.scheduler.manager import SchedulerManager

            scheduler_manager = SchedulerManager(
                default_max_spend_usd=getattr(cfg, "max_spend_usd", 0.0),
                default_tool_policy_kwargs=_agent_tool_policy_kwargs(cfg),
                max_jobs=getattr(cfg.scheduling, "max_jobs", 0),
            )
            scheduler_manager.start()
            _agent._scheduler = scheduler_manager  # noqa: SLF001
            console.print(
                f"[green]Scheduler started[/] ({len(scheduler_manager.list_jobs())} job(s) loaded)"
            )
        else:
            console.print("[dim]Scheduler disabled via config (scheduling.enabled: false).[/]")
    except Exception as _sched_exc:
        console.print(f"[yellow]Scheduler failed to start: {_sched_exc}[/]")
        logger.warning("Scheduler startup error: %s", _sched_exc, exc_info=True)

    # Config hot-reload. Fully built and tested (missy/config/hotreload.py,
    # including its symlink/ownership/permission safety checks before
    # reload), but had zero production callers anywhere -- editing
    # config.yaml while gateway_start() was running (the long-lived service
    # mode where hot-reload matters most) had no effect whatsoever, despite
    # README.md/docs/architecture.md/CLAUDE.md all describing it as an
    # active running control. _apply_config() (this same module) already
    # exists as the ready-made reload callback -- it was simply never
    # wired to an actual ConfigWatcher instance.
    from missy.config.hotreload import ConfigWatcher, _apply_config

    # _apply_config() reinitializes PolicyEngine/ProviderRegistry/
    # OtelExporter/AuditLogger, but each of those is a fresh singleton
    # rebuilt from the new config. _agent/_discord_agent/_proactive_runtime
    # are long-lived AgentRuntime instances already constructed above --
    # their .config (a plain, mutable AgentConfig, not rebuilt on reload)
    # is read fresh only when a *new* per-session CostTracker is first
    # created (AgentRuntime._make_cost_tracker() reads self.config.max_spend_usd
    # at that moment), so editing max_spend_usd in config.yaml while the
    # gateway keeps running had no effect on this process's already-running
    # runtimes at all -- not even for brand-new sessions started after the
    # edit -- until a full restart. Same gap for scheduler_manager's
    # _default_max_spend_usd, read once at construction and threaded into
    # every per-job AgentConfig thereafter. Propagating the new value onto
    # each object in place (mutating existing objects, not rebuilding them)
    # matches the same in-place-repoint approach already used for
    # AuditLogger.reconfigure()/OtelExporter re-init.
    # Populated further down, only if the voice channel is enabled and its
    # safe-chat runtime construction succeeds; referenced by the closure
    # below via ordinary late-binding (no `nonlocal` needed since this
    # function only reads it, and Python closures see later reassignments
    # of an enclosing-scope variable, not a value snapshotted at def-time).
    _voice_safe_chat_agent: Any = None

    def _apply_config_and_refresh_runtimes(new_cfg: Any) -> None:
        _apply_config(new_cfg)
        new_max_spend = getattr(new_cfg, "max_spend_usd", 0.0)
        _agent.config.max_spend_usd = new_max_spend
        _discord_agent.config.max_spend_usd = new_max_spend
        if _proactive_runtime is not None:
            _proactive_runtime.config.max_spend_usd = new_max_spend
        if _voice_safe_chat_agent is not None:
            _voice_safe_chat_agent.config.max_spend_usd = new_max_spend
        if scheduler_manager is not None:
            scheduler_manager._default_max_spend_usd = new_max_spend  # noqa: SLF001
            # SCHED-003: same in-place-repoint treatment for max_jobs --
            # read once at construction otherwise, so editing
            # scheduling.max_jobs in config.yaml while the gateway keeps
            # running would have no effect on this already-constructed
            # scheduler_manager until a full restart.
            scheduler_manager._max_jobs = getattr(  # noqa: SLF001
                new_cfg.scheduling, "max_jobs", 0
            )

    config_watcher = ConfigWatcher(
        ctx.obj["config_path"], reload_fn=_apply_config_and_refresh_runtimes
    )
    config_watcher.start()

    # Start voice channel if configured.
    voice_channel = None
    try:
        import yaml as _yaml

        _cfg_file = Path(ctx.obj["config_path"]).expanduser()
        _raw_cfg = {}
        if _cfg_file.exists():
            with _cfg_file.open() as _fh:
                _raw_cfg = _yaml.safe_load(_fh) or {}
        _voice_cfg = _raw_cfg.get("voice", {})

        if _voice_cfg.get("enabled", True):
            import os as _os

            from missy.channels.voice.channel import VoiceChannel

            # Ensure CUDA libs and Piper libs are available.
            _ld = _os.environ.get("LD_LIBRARY_PATH", "")
            _piper_lib = str(Path.home() / ".local" / "bin")
            _cuda_lib = "/usr/local/lib/ollama/cuda_v12"
            for _p in [_piper_lib, _cuda_lib]:
                if _p not in _ld:
                    _ld = f"{_p}:{_ld}" if _ld else _p
            _os.environ["LD_LIBRARY_PATH"] = _ld

            voice_channel = VoiceChannel(
                host=_voice_cfg.get("host", "0.0.0.0"),
                port=_voice_cfg.get("port", 8765),
                stt_model=_voice_cfg.get("stt", {}).get("model", "base.en"),
                tts_voice=_voice_cfg.get("tts", {}).get("voice", "en_US-lessac-medium"),
                debug_transcripts=_voice_cfg.get("debug_transcripts", False),
            )
            # A dedicated capability_mode="safe-chat" runtime for edge nodes
            # configured via `missy devices policy <id> --mode safe-chat`.
            # Without this, "safe-chat" was read in exactly one place in the
            # whole voice subsystem (the "muted" check) -- a safe-chat node
            # got full, unrestricted tool access identical to "full" mode.
            _voice_safe_chat_agent_cfg = AgentConfig(
                provider=_provider_name,
                capability_mode="safe-chat",
                max_spend_usd=getattr(cfg, "max_spend_usd", 0.0),
                **_agent_tool_policy_kwargs(cfg),
            )
            _voice_safe_chat_agent = AgentRuntime(_voice_safe_chat_agent_cfg)
            voice_channel.start(_agent, safe_chat_agent_runtime=_voice_safe_chat_agent)
            _vc_host = _voice_cfg.get("host", "0.0.0.0")
            _vc_port = _voice_cfg.get("port", 8765)
            console.print(f"[green]Voice channel started[/] on ws://{_vc_host}:{_vc_port}")
    except Exception as _ve:
        console.print(f"[yellow]Voice channel failed to start: {_ve}[/]")
        logger.warning("Voice channel startup error: %s", _ve, exc_info=True)

    # SR-1.12/task #12: shared list the Web API's /api/v1/discord/pairing
    # endpoints read from. DiscordChannel instances are constructed later
    # (inside the async `_run_discord()` below, after this list is passed
    # to ApiServer), so this is the same list object both sides share --
    # appended to once channels exist, read lazily at request time.
    discord_channels: list = []

    # Start screencast channel if configured.
    screencast_channel = None
    try:
        import yaml as _sc_yaml

        _cfg_file_sc = Path(ctx.obj["config_path"]).expanduser()
        _raw_cfg_sc = {}
        if _cfg_file_sc.exists():
            with _cfg_file_sc.open() as _fh_sc:
                _raw_cfg_sc = _sc_yaml.safe_load(_fh_sc) or {}
        _sc_cfg = _raw_cfg_sc.get("screencast", {})

        if _sc_cfg.get("enabled", False):
            from missy.channels.screencast.channel import ScreencastChannel

            screencast_channel = ScreencastChannel(
                host=_sc_cfg.get("host", "127.0.0.1"),
                port=_sc_cfg.get("port", 8780),
                max_sessions=_sc_cfg.get("max_sessions", 20),
                frame_save_dir=_sc_cfg.get("frame_save_dir", ""),
                vision_model=_sc_cfg.get("vision_model", ""),
                analysis_prompt=_sc_cfg.get("analysis_prompt", ""),
                capture_url_base=_sc_cfg.get("capture_url_base", ""),
            )
            screencast_channel.start()
            _sc_host = _sc_cfg.get("host", "127.0.0.1")
            _sc_port = _sc_cfg.get("port", 8780)
            _sc_tls = getattr(screencast_channel._server, "_tls_enabled", False)  # noqa: SLF001
            _sc_scheme = "https" if _sc_tls else "http"
            console.print(
                f"[green]Screencast channel started[/] on {_sc_scheme}://{_sc_host}:{_sc_port}"
            )
    except Exception as _sc_exc:
        console.print(f"[yellow]Screencast channel failed to start: {_sc_exc}[/]")
        logger.warning("Screencast channel startup error: %s", _sc_exc, exc_info=True)

    # Start the Web TUI (operator console + REST API) if configured. Enabled
    # by default so every gateway run exposes the console without extra setup.
    api_server = None
    try:
        import yaml as _api_yaml

        _cfg_file_api = Path(ctx.obj["config_path"]).expanduser()
        _raw_cfg_api: dict[str, Any] = {}
        if _cfg_file_api.exists():
            with _cfg_file_api.open() as _fh_api:
                _raw_cfg_api = _api_yaml.safe_load(_fh_api) or {}
        _api_cfg = _raw_cfg_api.get("api", {})

        if _api_cfg.get("enabled", True):
            from missy.api.server import ApiConfig, ApiServer

            try:
                from missy.providers.registry import get_registry as _get_provider_registry

                _api_provider_registry = _get_provider_registry()
            except Exception:
                _api_provider_registry = None

            try:
                from missy.tools.registry import get_tool_registry as _get_tool_registry

                _api_tool_registry = _get_tool_registry()
            except Exception:
                _api_tool_registry = None

            try:
                from missy.memory.sqlite_store import SQLiteMemoryStore

                _mem_db_path = str(Path(cfg.audit_log_path).expanduser().parent / "memory.db")
                _api_memory_store = SQLiteMemoryStore(db_path=_mem_db_path)
            except Exception as _api_mem_exc:
                logger.debug("Web console memory store unavailable: %s", _api_mem_exc)
                _api_memory_store = None

            _api_key = str(_api_cfg.get("api_key") or "").strip()
            if not _api_key:
                _api_key = _load_or_create_web_console_key()

            _api_host = _api_cfg.get("host", "127.0.0.1")
            _api_port = int(_api_cfg.get("port", 8080))
            _api_server_config = ApiConfig(host=_api_host, port=_api_port, api_key=_api_key)
            api_server = ApiServer(
                config=_api_server_config,
                runtime=_agent,
                memory_store=_api_memory_store,
                provider_registry=_api_provider_registry,
                tool_registry=_api_tool_registry,
                approval_gate=approval_gate,
                discord_channels=discord_channels,
            )
            api_server.start()
            console.print(
                f"[green]Web console started[/] on [bold]{api_server.url}[/]\n"
                f"  [dim]Operator key: {_api_key} (also saved to "
                "~/.missy/secrets/web_console.key)[/]"
            )
        else:
            console.print("[dim]Web console disabled via config (api.enabled: false).[/]")
    except Exception as _api_exc:
        console.print(f"[yellow]Web console failed to start: {_api_exc}[/]")
        logger.warning("Web console startup error: %s", _api_exc, exc_info=True)

    # Start Discord channel if configured.
    try:
        if cfg.discord and cfg.discord.enabled and cfg.discord.accounts:
            import asyncio

            from missy.channels.discord.channel import DiscordChannel
            from missy.core.exceptions import ProviderError

            async def _process_channel(ch: DiscordChannel) -> None:
                """Drain the channel queue and run the agent for each message."""
                while not stop_event:
                    try:
                        msg = await asyncio.wait_for(ch._queue.get(), timeout=1.0)
                    except TimeoutError:
                        continue
                    except Exception:
                        logging.getLogger(__name__).debug("Discord queue error", exc_info=True)
                        break

                    session_id = msg.metadata.get("discord_author", {}).get("id", "discord")
                    channel_id = msg.metadata.get("discord_channel_id", "")

                    # Inject channel context so the agent knows which Discord
                    # channel it is responding in (for discord_upload_file etc.).
                    discord_ctx = (
                        f"[Discord channel {channel_id}] "
                        f"Use discord_upload_file with channel_id='{channel_id}' "
                        f"to share files here."
                    )
                    enriched_prompt = f"{discord_ctx}\n\n{msg.content}"

                    # DiscordChannel._handle_message() already policy-validated any
                    # image/text attachments and attached their metadata to
                    # msg.metadata -- download them now and make the agent
                    # actually aware of them (a local path + vision_capture
                    # instruction for images, sanitized inline content for text).
                    # Previously this metadata was built and forwarded but never
                    # consumed anywhere, so an attached image or spec file was
                    # invisible to the agent regardless of the policy gate.
                    image_atts = msg.metadata.get("discord_image_attachments") or []
                    text_atts = msg.metadata.get("discord_text_attachments") or []
                    if image_atts or text_atts:
                        from missy.channels.discord.attachment_context import (
                            build_inbound_attachment_context,
                        )

                        try:
                            attachment_context = await build_inbound_attachment_context(
                                ch._rest,  # noqa: SLF001
                                image_atts,
                                text_atts,
                                message_id=msg.metadata.get("discord_message_id", ""),
                            )
                        except Exception:
                            logger.exception("Failed to build inbound attachment context")
                            attachment_context = ""
                        if attachment_context:
                            enriched_prompt = f"{enriched_prompt}\n\n{attachment_context}"

                    # Keep "Missy is typing..." visible for the entire agent run.
                    # Discord's typing indicator expires after ~10s, so refresh it.
                    typing_stop = asyncio.Event()

                    async def _typing_keepalive(
                        cid: str = channel_id,
                        stop: asyncio.Event = typing_stop,
                    ) -> None:
                        while not stop.is_set():
                            with contextlib.suppress(Exception):
                                await asyncio.to_thread(ch._rest.trigger_typing, cid)  # noqa: SLF001
                            with contextlib.suppress(asyncio.TimeoutError):
                                await asyncio.wait_for(stop.wait(), timeout=7.0)

                    # Snapshot existing evolution proposal IDs before the
                    # agent runs, so a genuine new proposal created this
                    # turn can be detected reliably afterward (see below)
                    # instead of regex-matching Missy's own final reply
                    # text for a literal "Evolution proposed:" string --
                    # the model paraphrases its own tool-call summaries
                    # (e.g. "Proposed a calculator hardening evolution:"),
                    # so that literal string is not guaranteed to survive
                    # into the user-facing response at all.
                    _existing_proposal_ids: set[str] = set()
                    try:
                        from missy.agent.code_evolution import CodeEvolutionManager

                        _existing_proposal_ids = {p.id for p in CodeEvolutionManager().list_all()}
                    except Exception:
                        logger.debug(
                            "Could not snapshot existing evolution proposals", exc_info=True
                        )

                    typing_task = asyncio.create_task(_typing_keepalive())
                    try:
                        loop = asyncio.get_running_loop()
                        response = await loop.run_in_executor(
                            None, _discord_agent.run, enriched_prompt, session_id
                        )
                    except ProviderError as exc:
                        response = f"Sorry, I encountered a provider error: {exc}"
                    except Exception as exc:
                        logger.exception("Discord agent error: %s", exc)
                        response = f"Sorry, an error occurred: {exc}"
                    finally:
                        typing_stop.set()
                        with contextlib.suppress(Exception):
                            await typing_task

                    try:
                        # Never use Discord reply; for bots, tag them with a mention.
                        mention_ids: list[str] | None = None
                        if msg.metadata.get("discord_author_is_bot"):
                            response = f"<@{msg.sender}> {response}"
                            mention_ids = [msg.sender]
                        sent_id = await ch.send_with_retry(
                            channel_id,
                            response,
                            mention_user_ids=mention_ids,
                        )

                        # Detect evolution proposals and add reaction buttons.
                        # Ground-truth comparison against the pre-run
                        # snapshot above, not text-matching Missy's own
                        # (unreliable, paraphrased) reply -- a proposal
                        # created via code_evolve is real, durable state
                        # in CodeEvolutionManager regardless of how the
                        # model chose to describe it to the user.
                        if sent_id:
                            try:
                                from missy.agent.code_evolution import (
                                    CodeEvolutionManager,
                                    EvolutionStatus,
                                )

                                for _prop in CodeEvolutionManager().list_all():
                                    if (
                                        _prop.id not in _existing_proposal_ids
                                        and _prop.status == EvolutionStatus.PROPOSED
                                    ):
                                        ch.add_evolution_reactions(channel_id, sent_id, _prop.id)
                            except Exception:
                                logger.debug("Evolution proposal detection failed", exc_info=True)
                    except Exception as exc:
                        logger.error(
                            "Discord send failed after retries (channel=%s): %s",
                            channel_id,
                            exc,
                        )
                        # Inform the agent that its response was not delivered.
                        with contextlib.suppress(Exception):
                            _discord_agent.run(
                                f"[SYSTEM] Your previous response to channel "
                                f"{channel_id} FAILED to send after multiple "
                                f"retries. Error: {exc}. The user did NOT see "
                                f"your response. You may want to try again or "
                                f"adjust your response.",
                                session_id,
                            )

            async def _run_discord() -> None:
                channels = []
                tasks = []
                for account in cfg.discord.accounts:
                    ch = DiscordChannel(account_config=account)
                    ch.set_agent_runtime(_discord_agent)
                    if screencast_channel is not None:
                        ch.set_screencast(screencast_channel)
                        screencast_channel.set_discord_rest(ch._rest)  # noqa: SLF001
                    await ch.start()
                    channels.append(ch)
                    discord_channels.append(ch)
                    console.print(f"[green]Discord channel started[/] ({account.token_env_var})")
                    tasks.append(asyncio.create_task(_process_channel(ch)))
                try:
                    while not stop_event:
                        await asyncio.sleep(1)
                finally:
                    for t in tasks:
                        t.cancel()
                    for t in tasks:
                        with contextlib.suppress(asyncio.CancelledError):
                            await t
                    for ch in channels:
                        await ch.stop()
                        with contextlib.suppress(ValueError):
                            discord_channels.remove(ch)

            asyncio.run(_run_discord())
        else:
            console.print("[dim]No Discord channels configured. Running in idle service mode.[/]")
            import time

            while not stop_event:
                time.sleep(1)
    finally:
        if voice_channel is not None:
            try:
                voice_channel.stop()
                console.print("[dim]Voice channel stopped.[/]")
            except Exception as _vs_exc:
                logger.debug("voice: stop error: %s", _vs_exc)
        if screencast_channel is not None:
            try:
                screencast_channel.stop()
                console.print("[dim]Screencast channel stopped.[/]")
            except Exception as _sc_stop_exc:
                logger.debug("screencast: stop error: %s", _sc_stop_exc)
        if api_server is not None:
            try:
                api_server.stop()
                console.print("[dim]Web console stopped.[/]")
            except Exception as _api_stop_exc:
                logger.debug("web console: stop error: %s", _api_stop_exc)
        if proactive_manager is not None:
            try:
                proactive_manager.stop()
            except Exception as _stop_exc:
                logger.debug("proactive: stop error: %s", _stop_exc)
        try:
            watchdog.stop()
        except Exception as _wd_stop_exc:
            logger.debug("watchdog: stop error: %s", _wd_stop_exc)
        if scheduler_manager is not None:
            try:
                scheduler_manager.stop()
                console.print("[dim]Scheduler stopped.[/]")
            except Exception as _sched_stop_exc:
                logger.debug("scheduler: stop error: %s", _sched_stop_exc)
        try:
            config_watcher.stop()
        except Exception as _cw_stop_exc:
            logger.debug("config watcher: stop error: %s", _cw_stop_exc)
        # AgentRuntime.shutdown() stops each runtime's SleeptimeWorker
        # daemon thread cleanly (join with timeout) rather than letting it
        # be killed mid-cycle (possibly mid-LLM-call, mid summary/learning
        # write) whenever the process exits -- gateway_start is exactly the
        # long-running-process case AgentRuntime.shutdown()'s own docstring
        # names as needing this.
        try:
            _agent.shutdown()
        except Exception as _agent_shutdown_exc:
            logger.debug("agent: shutdown error: %s", _agent_shutdown_exc)
        if _discord_agent is not None:
            try:
                _discord_agent.shutdown()
            except Exception as _discord_agent_shutdown_exc:
                logger.debug("discord agent: shutdown error: %s", _discord_agent_shutdown_exc)

    console.print("[dim]Gateway stopped.[/]")


@gateway.command("status")
@click.pass_context
def gateway_status(ctx: click.Context) -> None:
    """Show gateway configuration and channel status."""
    cfg = _load_subsystems(ctx.obj["config_path"])

    table = Table(title="Gateway Status", show_lines=True)
    table.add_column("Channel", style="bold")
    table.add_column("Status")
    table.add_column("Detail")

    # Discord
    if cfg.discord and cfg.discord.enabled and cfg.discord.accounts:
        for i, account in enumerate(cfg.discord.accounts):
            token_set = bool(account.resolve_token())
            status = (
                Text("configured", style="green")
                if token_set
                else Text("token missing", style="red")
            )
            table.add_row(
                f"discord[{i}]",
                status,
                f"env={account.token_env_var} dm_policy={account.dm_policy.value}",
            )
    else:
        table.add_row("discord", Text("disabled", style="dim"), "not configured")

    # CLI channel always available
    table.add_row("cli", Text("available", style="green"), "stdin/stdout")

    # Web TUI (operator console + REST API) — started automatically by
    # `missy gateway start` unless disabled via config.
    import yaml as _status_yaml

    _cfg_file_status = Path(ctx.obj["config_path"]).expanduser()
    _raw_cfg_status: dict[str, Any] = {}
    if _cfg_file_status.exists():
        with _cfg_file_status.open() as _fh_status:
            _raw_cfg_status = _status_yaml.safe_load(_fh_status) or {}
    _api_status_cfg = _raw_cfg_status.get("api", {})
    if _api_status_cfg.get("enabled", True):
        _status_host = _api_status_cfg.get("host", "127.0.0.1")
        _status_port = _api_status_cfg.get("port", 8080)
        table.add_row(
            "web",
            Text("auto-starts with gateway", style="green"),
            f"http://{_status_host}:{_status_port} (missy gateway start)",
        )
    else:
        table.add_row("web", Text("disabled", style="dim"), "api.enabled: false in config")

    # Scheduler — auto-starts with the gateway (missy schedule add/list
    # manage jobs.json directly and don't require the gateway to be running,
    # but jobs only actually *fire* while a `missy gateway start` process is
    # up).
    if cfg.scheduling.enabled:
        try:
            from missy.scheduler.manager import SchedulerManager

            _job_count = len(SchedulerManager().load_jobs())
        except Exception:
            _job_count = None
        _sched_detail = (
            f"{_job_count} job(s) in jobs.json (missy gateway start)"
            if _job_count is not None
            else "jobs.json unreadable"
        )
        table.add_row("scheduler", Text("auto-starts with gateway", style="green"), _sched_detail)
    else:
        table.add_row(
            "scheduler", Text("disabled", style="dim"), "scheduling.enabled: false in config"
        )

    console.print(table)

    # Providers
    from missy.providers.registry import get_registry

    registry = get_registry()
    provider_names = registry.list_providers()
    console.print(
        f"\n[bold]Providers registered:[/] {', '.join(provider_names) if provider_names else '[dim]none[/]'}"
    )


# ---------------------------------------------------------------------------
# missy doctor
# ---------------------------------------------------------------------------


@cli.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Run health diagnostics and print a system status report.

    Checks configuration, provider availability, network policy, filesystem
    policy, audit log, scheduler state, and Discord configuration.
    """
    from missy.providers.registry import get_registry
    from missy.scheduler.manager import SchedulerManager

    cfg = _load_subsystems(ctx.obj["config_path"])

    ok = Text("OK", style="green")
    warn = Text("WARN", style="yellow")
    fail = Text("FAIL", style="red")

    table = Table(title="Missy Doctor", show_lines=True)
    table.add_column("Check", style="bold")
    table.add_column("Result", justify="center")
    table.add_column("Detail")

    # 1. Config loaded
    table.add_row("config loaded", ok, str(Path(ctx.obj["config_path"]).expanduser()))

    # 2. Audit log
    audit_path = Path(cfg.audit_log_path).expanduser()
    if audit_path.exists():
        table.add_row("audit log", ok, str(audit_path))
    else:
        table.add_row("audit log", warn, f"not found: {audit_path}")

    # 2b. Audit signing status (SR-1.1/SR-4.6 residual): a quick,
    # read-only glance at whether the audit trail is actually
    # tamper-evident, without requiring the operator to separately run
    # `missy audit verify`. Signing without verification provides no
    # real tamper detection, so this surfaces the same
    # valid/tampered/unsigned/malformed breakdown that command reports,
    # scoped down to a single doctor row.
    if audit_path.exists():
        try:
            from missy.observability.audit_logger import verify_audit_log
            from missy.security.identity import AgentIdentity

            identity = AgentIdentity.load_or_generate()
            results = verify_audit_log(cfg.audit_log_path, identity)
        except Exception as exc:
            table.add_row("audit signing", warn, f"could not verify: {exc}")
        else:
            counts: dict[str, int] = {}
            for r in results:
                counts[r.status] = counts.get(r.status, 0) + 1
            summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
            # Per-line signatures alone catch content tampering but not
            # reordering/deletion of otherwise validly-signed lines --
            # surfaced separately here since counts (status-keyed) has no
            # slot for it.
            broken_chain_count = sum(1 for r in results if r.chain_ok is False)
            if not results:
                table.add_row("audit signing", warn, "log is empty — nothing to verify yet")
            elif counts.get("tampered") or counts.get("malformed"):
                table.add_row("audit signing", fail, f"integrity issue found: {summary}")
            elif broken_chain_count:
                table.add_row(
                    "audit signing",
                    fail,
                    f"{broken_chain_count} line(s) out of sequence "
                    f"(reordering/deletion detected): {summary}",
                )
            elif counts.get("unsigned"):
                table.add_row(
                    "audit signing",
                    warn,
                    f"some lines predate signing or were written unsigned: {summary}",
                )
            else:
                table.add_row("audit signing", ok, f"all lines verified: {summary}")
    else:
        table.add_row("audit signing", warn, "no audit log to verify yet")

    app_log_path = _app_log_path(cfg)
    if app_log_path.exists():
        table.add_row("application log", ok, str(app_log_path))
    else:
        table.add_row("application log", warn, f"not found yet: {app_log_path}")

    # 3. Workspace
    workspace = Path(cfg.workspace_path).expanduser()
    if workspace.exists():
        table.add_row("workspace", ok, str(workspace))
    else:
        table.add_row("workspace", warn, f"missing: {workspace} — run missy init")

    # 4. Secrets dir
    secrets_dir = Path("~/.missy/secrets").expanduser()
    if secrets_dir.exists():
        table.add_row("secrets dir", ok, str(secrets_dir))
    else:
        table.add_row("secrets dir", warn, f"missing: {secrets_dir} — run missy init")

    # 5. Network policy
    net = cfg.network
    net_detail = (
        f"default_deny={net.default_deny} "
        f"domains={len(net.allowed_domains)} "
        f"cidrs={len(net.allowed_cidrs)} "
        f"hosts={len(net.allowed_hosts)}"
    )
    net_status = ok if net.default_deny else warn
    table.add_row("network policy", net_status, net_detail)

    # 6. Providers
    registry = get_registry()
    provider_names = registry.list_providers()
    if not provider_names:
        table.add_row("providers", warn, "no providers configured")
    else:
        for name in provider_names:
            p = registry.get(name)
            try:
                avail = p.is_available() if p else False
            except Exception:
                logging.getLogger(__name__).debug(
                    "Provider %s availability check failed", name, exc_info=True
                )
                avail = False
            status = ok if avail else fail
            table.add_row(
                f"provider:{name}", status, "api key present" if avail else "not available"
            )
            diagnostics = getattr(p, "diagnostics", None)
            if callable(diagnostics):
                from missy.api.audit_browser import redact_audit_value

                try:
                    report = diagnostics()
                except Exception as exc:
                    logging.getLogger(__name__).debug(
                        "Provider %s diagnostics failed", name, exc_info=True
                    )
                    table.add_row(
                        f"provider:{name}:diagnostics",
                        warn,
                        str(redact_audit_value(f"error: {exc}")),
                    )
                else:
                    if isinstance(report, dict):
                        for item in report.get("checks", []) or []:
                            if not isinstance(item, dict):
                                continue
                            item_status = str(item.get("status", "warn"))
                            status_text = (
                                ok
                                if item_status == "ok"
                                else fail
                                if item_status == "error"
                                else warn
                            )
                            table.add_row(
                                f"provider:{name}:{item.get('name', 'diagnostic')}",
                                status_text,
                                str(redact_audit_value(item.get("summary", ""))),
                            )

    # 7. Shell policy
    shell_status = warn if cfg.shell.enabled else ok
    shell_detail = "ENABLED" if cfg.shell.enabled else "disabled (secure)"
    if cfg.shell.enabled and cfg.shell.allowed_commands:
        shell_detail += f" commands={cfg.shell.allowed_commands}"
    table.add_row("shell policy", shell_status, shell_detail)

    # 8. Plugin policy
    plugin_status = warn if cfg.plugins.enabled else ok
    plugin_detail = "ENABLED" if cfg.plugins.enabled else "disabled (secure)"
    table.add_row("plugin policy", plugin_status, plugin_detail)

    # 9. Scheduler jobs
    mgr = SchedulerManager()
    jobs = mgr.load_jobs()
    table.add_row("scheduled jobs", ok, f"{len(jobs)} job(s) defined")

    # 10. Discord
    if cfg.discord and cfg.discord.enabled:
        accounts = cfg.discord.accounts
        for i, acc in enumerate(accounts):
            has_token = bool(acc.resolve_token())
            ds = ok if has_token else fail
            dd = f"token_env={acc.token_env_var} dm_policy={acc.dm_policy.value}"
            table.add_row(f"discord[{i}]", ds, dd)
    else:
        table.add_row("discord", Text("disabled", style="dim"), "not configured")

    # 11. Memory store
    try:
        from missy.memory.sqlite_store import SQLiteMemoryStore

        mem_path = Path("~/.missy/memory.db").expanduser()
        if mem_path.exists():
            store = SQLiteMemoryStore(str(mem_path))
            # Quick connectivity check: count turns
            store.get_session_turns("__health_check__", limit=1)
            table.add_row("memory store", ok, f"sqlite: {mem_path} (accessible)")
        else:
            table.add_row("memory store", warn, f"not found: {mem_path}")
    except Exception as exc:
        table.add_row("memory store", fail, f"error: {exc}")

    # 12. MCP servers
    try:
        mcp_path = Path("~/.missy/mcp.json").expanduser()
        if mcp_path.exists():
            import json

            mcp_data = json.loads(mcp_path.read_text())
            servers = mcp_data.get("servers", {}) if isinstance(mcp_data, dict) else {}
            if servers:
                table.add_row(
                    "mcp servers",
                    ok,
                    f"{len(servers)} server(s) configured: {', '.join(servers.keys())}",
                )
            else:
                table.add_row("mcp servers", Text("none", style="dim"), "no servers in mcp.json")
        else:
            table.add_row("mcp servers", Text("none", style="dim"), "mcp.json not found")
    except Exception as exc:
        table.add_row("mcp servers", fail, f"error reading mcp.json: {exc}")

    # 13. Config hot-reload (watchdog)
    try:
        import importlib

        watchdog_spec = importlib.util.find_spec("watchdog")
        if watchdog_spec is not None:
            table.add_row("config hot-reload", ok, "watchdog available")
        else:
            table.add_row("config hot-reload", warn, "watchdog not installed — hot-reload disabled")
    except Exception:
        table.add_row("config hot-reload", warn, "could not check watchdog")

    # 14. Voice channel
    try:
        raw_cfg = {}
        config_path = Path(ctx.obj["config_path"]).expanduser()
        if config_path.exists():
            import yaml

            raw_cfg = yaml.safe_load(config_path.read_text()) or {}
        voice_cfg = raw_cfg.get("voice", {})
        if voice_cfg:
            voice_host = voice_cfg.get("host", "0.0.0.0")
            voice_port = voice_cfg.get("port", 8765)
            stt_engine = voice_cfg.get("stt", {}).get("engine", "none")
            tts_engine = voice_cfg.get("tts", {}).get("engine", "none")
            table.add_row(
                "voice channel",
                ok,
                f"{voice_host}:{voice_port} stt={stt_engine} tts={tts_engine}",
            )
        else:
            table.add_row("voice channel", Text("disabled", style="dim"), "not configured")
    except Exception as exc:
        table.add_row("voice channel", warn, f"error checking voice: {exc}")

    # 15. Checkpoint database
    try:
        cp_path = Path("~/.missy/checkpoints.db").expanduser()
        if cp_path.exists():
            table.add_row("checkpoints", ok, f"database: {cp_path}")
        else:
            table.add_row("checkpoints", Text("none", style="dim"), "no checkpoint database")
    except Exception:
        logger.debug("doctor: failed to check checkpoints", exc_info=True)

    console.print(table)


# ---------------------------------------------------------------------------
# missy cost
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--session", default=None, help="Session ID to query (default: show config).")
@click.pass_context
def cost(ctx: click.Context, session: str | None) -> None:
    """Show cost tracking configuration and session cost summary.

    When ``--session`` is given, shows cost data from the memory store for
    that session.  Otherwise shows the current budget configuration.
    """
    cfg = _load_subsystems(ctx.obj["config_path"])

    table = Table(title="Cost Tracking", show_lines=True)
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    budget = getattr(cfg, "max_spend_usd", 0.0)
    table.add_row("Budget limit (max_spend_usd)", f"${budget:.2f}" if budget > 0 else "unlimited")
    table.add_row("Config location", "max_spend_usd in config.yaml")

    if session:
        try:
            from missy.memory.sqlite_store import SQLiteMemoryStore

            store = SQLiteMemoryStore()
            turns = store.get_session_turns(session, limit=1000)
            table.add_row("Session", session)
            table.add_row("Turns", str(len(turns)))

            # Show persisted cost data if available
            cost_rows = store.get_session_costs(session)
            if cost_rows:
                total_cost = sum(r["cost_usd"] for r in cost_rows)
                total_prompt = sum(r["prompt_tokens"] for r in cost_rows)
                total_completion = sum(r["completion_tokens"] for r in cost_rows)
                table.add_row("API calls", str(len(cost_rows)))
                table.add_row("Prompt tokens", f"{total_prompt:,}")
                table.add_row("Completion tokens", f"{total_completion:,}")
                table.add_row("Total cost", f"${total_cost:.6f}")

                # Per-model breakdown
                models: dict[str, float] = {}
                for r in cost_rows:
                    models[r["model"]] = models.get(r["model"], 0.0) + r["cost_usd"]
                for model, cost in sorted(models.items(), key=lambda x: -x[1]):
                    table.add_row(f"  {model}", f"${cost:.6f}")
            else:
                table.add_row("Cost data", "[dim]No cost records for this session[/]")
        except Exception as exc:
            table.add_row("Session lookup", f"[red]Error: {exc}[/]")

    console.print(table)

    console.print(
        "\n[dim]To set a budget, add to config.yaml:[/]\n"
        "  [bold]max_spend_usd: 5.00[/]  # dollars per session\n"
    )


# ---------------------------------------------------------------------------
# missy recover
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--abandon-all", is_flag=True, help="Abandon all incomplete checkpoints.")
@click.option(
    "--resume",
    "resume_id",
    default=None,
    help="Resume the checkpoint with this ID from its saved conversation state.",
)
@click.option(
    "--provider", default=None, help="Provider to use for --resume (overrides config default)."
)
@click.pass_context
def recover(
    ctx: click.Context, abandon_all: bool, resume_id: str | None, provider: str | None
) -> None:
    """List or act on incomplete task checkpoints from previous sessions.

    Scans for tasks that were interrupted by crashes or restarts and shows
    recovery options.  Use --abandon-all to clear all stale checkpoints, or
    --resume ID to continue a specific checkpoint from its saved state.
    """

    try:
        from missy.agent.checkpoint import (
            CheckpointCorruptedError,
            CheckpointManager,
            scan_for_recovery,
        )
    except ImportError:
        _print_error("Checkpoint module not available.")
        sys.exit(1)

    if abandon_all:
        try:
            cm = CheckpointManager()
            count = cm.abandon_old(max_age_seconds=0)
            _print_success(f"Abandoned {count} checkpoint(s).")
        except Exception as exc:
            _print_error(f"Failed to abandon checkpoints: {exc}")
            sys.exit(1)
        return

    if resume_id:
        from missy.agent.runtime import AgentConfig, AgentRuntime
        from missy.core.exceptions import ProviderError

        cfg = _load_subsystems(ctx.obj["config_path"])
        provider_name = provider or (
            next(iter(cfg.providers), "anthropic") if cfg.providers else "anthropic"
        )
        agent_cfg = AgentConfig(
            provider=provider_name,
            max_spend_usd=getattr(cfg, "max_spend_usd", 0.0),
            **_agent_tool_policy_kwargs(cfg),
        )
        agent = AgentRuntime(agent_cfg)
        with console.status(f"[bold cyan]Resuming {resume_id[:12]}...[/]", spinner="dots"):
            try:
                response = agent.resume_checkpoint(resume_id)
            except CheckpointCorruptedError as exc:
                _print_error(f"Checkpoint corrupted, marked FAILED: {exc}")
                sys.exit(1)
            except ValueError as exc:
                _print_error(str(exc))
                sys.exit(1)
            except ProviderError as exc:
                _print_error(
                    f"Provider error: {exc}",
                    hint="Check that your API key is set and the provider is configured.",
                )
                sys.exit(1)
        console.print(Panel(response, title="[bold cyan]Missy (resumed)[/]", border_style="cyan"))
        return

    results = scan_for_recovery()
    if not results:
        console.print("[green]No incomplete checkpoints found.[/]")
        return

    table = Table(title="Incomplete Checkpoints", show_lines=True)
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Session", max_width=12)
    table.add_column("Action", style="bold")
    table.add_column("Prompt", max_width=50)
    table.add_column("Iteration")

    for r in results:
        action_style = {
            "resume": "[green]resume[/]",
            "restart": "[yellow]restart[/]",
            "abandon": "[red]abandon[/]",
        }.get(r.action, r.action)

        table.add_row(
            r.checkpoint_id[:12],
            r.session_id[:12] if r.session_id else "",
            action_style,
            r.prompt[:50] if r.prompt else "",
            str(r.iteration),
        )

    console.print(table)
    console.print(
        f"\n[dim]{len(results)} checkpoint(s) found. "
        "Use [bold]missy recover --resume ID[/bold] to continue one from its "
        "saved state, or [bold]missy recover --abandon-all[/bold] to clear "
        "stale tasks.[/]"
    )


# ---------------------------------------------------------------------------
# missy vault
# ---------------------------------------------------------------------------


@cli.group()
def vault() -> None:
    """Encrypted secrets vault commands."""


@vault.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def vault_set(ctx: click.Context, key: str, value: str) -> None:
    """Store a secret in the encrypted vault."""
    from missy.security.vault import Vault, VaultError

    cfg = _load_subsystems(ctx.obj["config_path"])
    vault_dir = getattr(getattr(cfg, "vault", None), "vault_dir", "~/.missy/secrets")
    try:
        v = Vault(vault_dir)
        v.set(key, value)
        _print_success(f"Secret [bold]{key}[/] stored in vault.")
    except VaultError as exc:
        _print_error(str(exc))
        sys.exit(1)


@vault.command("get")
@click.argument("key")
@click.pass_context
def vault_get(ctx: click.Context, key: str) -> None:
    """Retrieve a secret from the vault (prints to stdout)."""
    from missy.security.vault import Vault, VaultError

    cfg = _load_subsystems(ctx.obj["config_path"])
    vault_dir = getattr(getattr(cfg, "vault", None), "vault_dir", "~/.missy/secrets")
    try:
        v = Vault(vault_dir)
        val = v.get(key)
        if val is None:
            _print_error(f"Key {key!r} not found in vault.")
            sys.exit(1)
        console.print(val)
    except VaultError as exc:
        _print_error(str(exc))
        sys.exit(1)


@vault.command("list")
@click.pass_context
def vault_list(ctx: click.Context) -> None:
    """List all key names in the vault."""
    from missy.security.vault import Vault, VaultError

    cfg = _load_subsystems(ctx.obj["config_path"])
    vault_dir = getattr(getattr(cfg, "vault", None), "vault_dir", "~/.missy/secrets")
    try:
        v = Vault(vault_dir)
        keys = v.list_keys()
        if not keys:
            console.print("[dim]Vault is empty.[/]")
        else:
            for k in keys:
                console.print(f"  {k}")
    except VaultError as exc:
        _print_error(str(exc))
        sys.exit(1)


@vault.command("delete")
@click.argument("key")
@click.pass_context
def vault_delete(ctx: click.Context, key: str) -> None:
    """Delete a secret from the vault."""
    from missy.security.vault import Vault, VaultError

    cfg = _load_subsystems(ctx.obj["config_path"])
    vault_dir = getattr(getattr(cfg, "vault", None), "vault_dir", "~/.missy/secrets")
    try:
        v = Vault(vault_dir)
        removed = v.delete(key)
        if removed:
            _print_success(f"Key [bold]{key}[/] deleted.")
        else:
            _print_error(f"Key {key!r} not found.")
            sys.exit(1)
    except VaultError as exc:
        _print_error(str(exc))
        sys.exit(1)


# ---------------------------------------------------------------------------
# missy sessions
# ---------------------------------------------------------------------------


@cli.group()
def sessions() -> None:
    """Session and conversation history commands."""


@sessions.command("cleanup")
@click.option(
    "--older-than", default=30, show_default=True, help="Delete history older than N days."
)
@click.option(
    "--dry-run", is_flag=True, default=False, help="Show what would be deleted without deleting."
)
@click.pass_context
def sessions_cleanup(ctx: click.Context, older_than: int, dry_run: bool) -> None:
    """Delete old conversation history from the memory store."""
    # SR-3.1/3.5: this previously constructed the legacy JSON MemoryStore,
    # which has no cleanup() method -- the hasattr guard always evaluated
    # False, so this command silently no-op'd on every invocation while
    # printing a message recommending SQLiteMemoryStore, the very store
    # `sessions list` (a few lines below) already uses correctly.
    from missy.memory.sqlite_store import SQLiteMemoryStore

    _load_subsystems(ctx.obj["config_path"])
    store = SQLiteMemoryStore()
    if dry_run:
        # SESSDEEP-002: run the real COUNT query rather than echoing back
        # a hardcoded, generic message -- an operator deciding whether to
        # commit to a real cleanup needs the actual number affected, not
        # just their own --older-than value reflected back at them.
        would_remove = store.cleanup(older_than_days=older_than, dry_run=True)
        console.print(
            f"[dim]Dry run: would delete {would_remove} conversation turn(s) "
            f"older than {older_than} days.[/]"
        )
        return
    removed = store.cleanup(older_than_days=older_than)
    _print_success(f"Removed {removed} conversation turn(s) older than {older_than} days.")


@sessions.command("list")
@click.option("--limit", default=20, show_default=True, help="Max sessions to show.")
@click.pass_context
def sessions_list(ctx: click.Context, limit: int) -> None:
    """List recent sessions with their names and turn counts."""
    from missy.memory.sqlite_store import SQLiteMemoryStore

    _load_subsystems(ctx.obj["config_path"])
    try:
        store = SQLiteMemoryStore()
        items = store.list_sessions(limit=limit)
    except Exception as exc:
        _print_error(f"Cannot read sessions: {exc}")
        return

    if not items:
        console.print("[dim]No sessions found in memory store.[/]")
        return

    table = Table(title="Sessions", show_lines=False)
    table.add_column("Session ID", style="cyan", no_wrap=True, max_width=36)
    table.add_column("Name", style="bold")
    table.add_column("Turns", justify="right")
    table.add_column("Provider")
    table.add_column("Channel")
    table.add_column("Last Updated")

    for s in items:
        sid = s["session_id"]
        name = s["name"] or "[dim]-[/]"
        turns = str(s["turn_count"])
        provider = s["provider"] or "[dim]-[/]"
        channel = s["channel"] or "[dim]-[/]"
        updated = s["updated_at"][:19] if s["updated_at"] else ""
        table.add_row(sid, name, turns, provider, channel, updated)

    console.print(table)


@sessions.command("rename")
@click.argument("session_id")
@click.argument("name")
@click.pass_context
def sessions_rename(ctx: click.Context, session_id: str, name: str) -> None:
    """Set a friendly name for a session."""
    from missy.memory.sqlite_store import SQLiteMemoryStore

    _load_subsystems(ctx.obj["config_path"])
    try:
        store = SQLiteMemoryStore()
        # Try to resolve by name if it doesn't look like a UUID
        if len(session_id) < 32 and "-" not in session_id:
            resolved = store.resolve_session_name(session_id)
            if resolved:
                session_id = resolved

        if store.rename_session(session_id, name):
            _print_success(f"Session {session_id[:12]}... renamed to [bold]{name}[/]")
        else:
            _print_error(f"Session {session_id!r} not found.")
    except Exception as exc:
        _print_error(f"Cannot rename session: {exc}")


@sessions.command("clear")
@click.argument("session_id")
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip the confirmation prompt (for non-interactive use).",
)
@click.pass_context
def sessions_clear(ctx: click.Context, session_id: str, yes: bool) -> None:
    """Fully reset a session: delete its turns AND summaries.

    This is the operator surface for
    :meth:`SQLiteMemoryStore.clear_session_full` (F14) — the supported way
    to recover a session stuck in the "over-refusal spiral," where a plain
    history delete leaves the persisted ``summaries`` rows that keep
    re-injecting the contamination via ``MemorySynthesizer``. Accepts a raw
    session id or a friendly name (resolved like ``sessions rename``).

    Note: this clears the *persisted* store. A gateway already holding the
    session in memory should be restarted to drop in-process state.
    """
    from missy.memory.sqlite_store import SQLiteMemoryStore

    _load_subsystems(ctx.obj["config_path"])
    try:
        store = SQLiteMemoryStore()
        # Resolve a friendly name to its id, matching `sessions rename`.
        if len(session_id) < 32 and "-" not in session_id:
            resolved = store.resolve_session_name(session_id)
            if resolved:
                session_id = resolved

        if not store.session_exists(session_id):
            _print_error(f"Session {session_id!r} not found (no turns or summaries).")
            return

        if not yes:
            confirmed = click.confirm(
                f"Permanently delete ALL turns and summaries for session "
                f"{session_id[:12]}...? This cannot be undone.",
                default=False,
            )
            if not confirmed:
                console.print("[dim]Aborted; nothing was cleared.[/]")
                return

        removed = store.clear_session_full(session_id)
        _print_success(
            f"Cleared session {session_id[:12]}...: removed "
            f"{removed['turns']} turn(s) and {removed['summaries']} summary/summaries. "
            f"Restart the gateway to drop any in-memory copy of this session."
        )
    except Exception as exc:
        _print_error(f"Cannot clear session: {exc}")


# ---------------------------------------------------------------------------
# missy graph (F04) — operator surface for GraphMemoryStore
# ---------------------------------------------------------------------------


@cli.group()
def graph() -> None:
    """Query and seed the entity-relationship knowledge graph.

    Operator surface for ``GraphMemoryStore`` (the same store the agent's
    ``graph_query`` tool reads). Previously the store had no CLI or tool
    entry point at all.
    """


@graph.command("stats")
@click.pass_context
def graph_stats(ctx: click.Context) -> None:
    """Show entity/relationship counts and type breakdowns."""
    from missy.memory.graph_store import GraphMemoryStore

    _load_subsystems(ctx.obj["config_path"])
    try:
        store = GraphMemoryStore()
        s = store.stats()
    except Exception as exc:
        _print_error(f"Cannot read graph: {exc}")
        return

    console.print(
        f"[bold]Knowledge graph[/]: {s.get('entity_count', 0)} entit(y/ies), "
        f"{s.get('relationship_count', 0)} relationship(s)."
    )
    etypes = s.get("entity_types") or {}
    if etypes:
        t = Table(title="Entity types")
        t.add_column("Type")
        t.add_column("Count", justify="right")
        for k, v in sorted(etypes.items(), key=lambda kv: -kv[1]):
            t.add_row(str(k), str(v))
        console.print(t)
    rtypes = s.get("relation_types") or {}
    if rtypes:
        t = Table(title="Relation types")
        t.add_column("Type")
        t.add_column("Count", justify="right")
        for k, v in sorted(rtypes.items(), key=lambda kv: -kv[1]):
            t.add_row(str(k), str(v))
        console.print(t)


@graph.command("query")
@click.argument("text")
@click.option("--limit", default=15, show_default=True, help="Max entities to include.")
@click.pass_context
def graph_query_cmd(ctx: click.Context, text: str, limit: int) -> None:
    """Show entities and relationships related to TEXT."""
    from missy.memory.graph_store import GraphMemoryStore

    _load_subsystems(ctx.obj["config_path"])
    try:
        store = GraphMemoryStore()
        result = store.find_related(text, limit=max(1, limit))
        subgraph = store.get_context_subgraph(text, max_entities=max(1, limit))
    except Exception as exc:
        _print_error(f"Graph query failed: {exc}")
        return

    if not result.entities:
        console.print(f"[dim]No entities related to {text!r} found in the graph.[/]")
        return
    console.print(
        f"[bold]{len(result.entities)}[/] entit(y/ies), "
        f"[bold]{len(result.relationships)}[/] relationship(s) related to {text!r}:"
    )
    console.print(subgraph)


@graph.command("entity")
@click.argument("name")
@click.pass_context
def graph_entity(ctx: click.Context, name: str) -> None:
    """Show a summary of the entity named NAME."""
    from missy.memory.graph_store import GraphMemoryStore

    _load_subsystems(ctx.obj["config_path"])
    try:
        store = GraphMemoryStore()
        summary = store.get_entity_summary(name)
    except Exception as exc:
        _print_error(f"Cannot read entity: {exc}")
        return
    if not summary or not summary.strip():
        console.print(f"[dim]No entity named {name!r} found in the graph.[/]")
        return
    console.print(summary)


@graph.command("add-entity")
@click.argument("name")
@click.option(
    "--type",
    "entity_type",
    default="concept",
    show_default=True,
    help="Entity type (person/tool/file/project/concept/location/organization).",
)
@click.pass_context
def graph_add_entity(ctx: click.Context, name: str, entity_type: str) -> None:
    """Seed a single entity into the graph (operator-only)."""
    from missy.memory.graph_store import Entity, GraphMemoryStore

    _load_subsystems(ctx.obj["config_path"])
    try:
        store = GraphMemoryStore()
        entity = Entity.new(name, entity_type)
        store.add_entity(entity)
    except Exception as exc:
        _print_error(f"Cannot add entity: {exc}")
        return
    _print_success(f"Added entity [bold]{entity.name}[/] ({entity_type}) to the graph.")


# ---------------------------------------------------------------------------
# missy approvals
# ---------------------------------------------------------------------------


@cli.group()
def approvals() -> None:
    """Approval gate management."""


@approvals.command("list")
@_APPROVALS_HOST_OPTION
@_APPROVALS_PORT_OPTION
@_APPROVALS_API_KEY_OPTION
def approvals_list(host: str, port: int, api_key: str) -> None:
    """List pending approval requests from a running `missy gateway start` session.

    SR-2.2: approval state lives in-process inside the running gateway
    (proactive triggers with requires_confirmation=True gate through an
    ApprovalGate there); this CLI command is a separate process and can
    only see that state via the gateway's own Web API.
    """
    import httpx

    resolved_key = _resolve_approvals_api_key(api_key)
    url = f"http://{host}:{port}/api/v1/approvals"
    headers = {"X-API-Key": resolved_key} if resolved_key else {}

    try:
        resp = httpx.get(url, headers=headers, timeout=3.0)
    except httpx.ConnectError:
        console.print(
            f"[dim]No active gateway session at [bold]http://{host}:{port}[/] — "
            "approvals are only processed while `missy gateway start` is running.[/]"
        )
        return
    except Exception as exc:
        _print_error(f"Could not reach gateway API: {exc}")
        return

    if resp.status_code == 401:
        _print_error(
            "Authentication required.",
            hint="Pass --api-key or ensure ~/.missy/secrets/web_console.key is readable.",
        )
        return
    if resp.status_code != 200:
        _print_error(f"Gateway API responded with HTTP {resp.status_code}.")
        return

    approvals_data = resp.json().get("data", {}).get("approvals", [])
    if not approvals_data:
        console.print("[dim]No pending approval requests.[/]")
        return

    table = Table(title="Pending Approvals", show_lines=True)
    table.add_column("ID", style="bold")
    table.add_column("Action")
    table.add_column("Reason")
    for item in approvals_data:
        table.add_row(item.get("id", ""), item.get("action", ""), item.get("reason", ""))
    console.print(table)


def _resolve_approval(
    host: str, port: int, api_key: str, approval_id: str, *, approve: bool
) -> None:
    import httpx

    resolved_key = _resolve_approvals_api_key(api_key)
    verb = "approve" if approve else "deny"
    # APPROVAL-003: "deny" + "d" reads as "denyd", not "denied" -- past
    # tense needs its own mapping, not a naive suffix on the verb used
    # for the URL path.
    past_tense = "approved" if approve else "denied"
    url = f"http://{host}:{port}/api/v1/approvals/{approval_id}/{verb}"
    headers = {"X-API-Key": resolved_key} if resolved_key else {}

    try:
        resp = httpx.post(url, headers=headers, timeout=3.0)
    except httpx.ConnectError:
        _print_error(
            f"No active gateway session at http://{host}:{port}.",
            hint="Approvals are only processed while `missy gateway start` is running.",
        )
        sys.exit(1)
    except Exception as exc:
        _print_error(f"Could not reach gateway API: {exc}")
        sys.exit(1)

    if resp.status_code == 200:
        _print_success(f"Request {approval_id!r} {past_tense}.")
    elif resp.status_code == 404:
        _print_error(f"No pending approval with id {approval_id!r}.")
        sys.exit(1)
    elif resp.status_code == 401:
        _print_error(
            "Authentication required.",
            hint="Pass --api-key or ensure ~/.missy/secrets/web_console.key is readable.",
        )
        sys.exit(1)
    else:
        _print_error(f"Gateway API responded with HTTP {resp.status_code}.")
        sys.exit(1)


@approvals.command("approve")
@click.argument("approval_id")
@_APPROVALS_HOST_OPTION
@_APPROVALS_PORT_OPTION
@_APPROVALS_API_KEY_OPTION
def approvals_approve(approval_id: str, host: str, port: int, api_key: str) -> None:
    """Approve a pending request by ID (see `missy approvals list`)."""
    _resolve_approval(host, port, api_key, approval_id, approve=True)


@approvals.command("deny")
@click.argument("approval_id")
@_APPROVALS_HOST_OPTION
@_APPROVALS_PORT_OPTION
@_APPROVALS_API_KEY_OPTION
def approvals_deny(approval_id: str, host: str, port: int, api_key: str) -> None:
    """Deny a pending request by ID (see `missy approvals list`)."""
    _resolve_approval(host, port, api_key, approval_id, approve=False)


# ---------------------------------------------------------------------------
# missy evolve
# ---------------------------------------------------------------------------


@cli.group()
def evolve() -> None:
    """Code self-evolution management."""


@evolve.command("list")
@click.pass_context
def evolve_list(ctx: click.Context) -> None:
    """List all evolution proposals."""
    from missy.agent.code_evolution import CodeEvolutionManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = CodeEvolutionManager()
    proposals = mgr.list_all()
    if not proposals:
        console.print("[dim]No evolution proposals.[/]")
        return
    table = Table(title="Code Evolution Proposals", show_lines=True)
    table.add_column("ID", style="bold")
    table.add_column("Status")
    table.add_column("Trigger")
    table.add_column("Confidence", justify="right")
    table.add_column("Title")
    table.add_column("Created")
    for p in proposals:
        status_style = {
            "proposed": "yellow",
            "approved": "cyan",
            "applied": "green",
            "rejected": "red",
            "rolled_back": "magenta",
            "failed": "red bold",
        }.get(p.status.value, "")
        table.add_row(
            p.id,
            Text(p.status.value, style=status_style),
            p.trigger.value,
            f"{p.confidence:.0%}",
            p.title[:50],
            p.created_at[:10] if p.created_at else "—",
        )
    console.print(table)


@evolve.command("show")
@click.argument("proposal_id")
@click.pass_context
def evolve_show(ctx: click.Context, proposal_id: str) -> None:
    """Show full details of an evolution proposal."""
    from missy.agent.code_evolution import CodeEvolutionManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = CodeEvolutionManager()
    prop = mgr.get(proposal_id)
    if not prop:
        _print_error(f"Proposal {proposal_id!r} not found.")
        return
    lines = [
        f"[bold]ID:[/] {prop.id}",
        f"[bold]Title:[/] {prop.title}",
        f"[bold]Status:[/] {prop.status.value}",
        f"[bold]Trigger:[/] {prop.trigger.value}",
        f"[bold]Confidence:[/] {prop.confidence:.0%}",
        f"[bold]Created:[/] {prop.created_at}",
        f"[bold]Resolved:[/] {prop.resolved_at or '—'}",
        f"[bold]Commit:[/] {prop.git_commit_sha or '—'}",
        f"\n[bold]Description:[/]\n{prop.description}",
    ]
    if prop.diffs:
        lines.append(f"\n[bold]Diffs ({len(prop.diffs)}):[/]")
        for i, d in enumerate(prop.diffs, 1):
            lines.append(f"\n[cyan]--- Diff {i}: {d.file_path} ---[/]")
            if d.description:
                lines.append(f"  Why: {d.description}")
            lines.append(f"  [red]- {d.original_code[:200]}[/]")
            lines.append(f"  [green]+ {d.proposed_code[:200]}[/]")
    if prop.error_pattern:
        lines.append(f"\n[bold]Error pattern:[/] {prop.error_pattern}")
    if prop.test_output:
        lines.append(f"\n[bold]Test output (last 500 chars):[/]\n{prop.test_output[-500:]}")
    console.print(Panel("\n".join(lines), title="Evolution Proposal", border_style="blue"))


@evolve.command("approve")
@click.argument("proposal_id")
@click.pass_context
def evolve_approve(ctx: click.Context, proposal_id: str) -> None:
    """Approve an evolution proposal for application."""
    from missy.agent.code_evolution import CodeEvolutionManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = CodeEvolutionManager()
    if mgr.approve(proposal_id):
        _print_success(
            f"Proposal [bold]{proposal_id}[/] approved. Use `missy evolve apply {proposal_id}` to apply."
        )
    else:
        _print_error(f"Proposal {proposal_id!r} not found or not in proposed status.")


@evolve.command("reject")
@click.argument("proposal_id")
@click.pass_context
def evolve_reject(ctx: click.Context, proposal_id: str) -> None:
    """Reject an evolution proposal."""
    from missy.agent.code_evolution import CodeEvolutionManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = CodeEvolutionManager()
    if mgr.reject(proposal_id):
        _print_success(f"Proposal [bold]{proposal_id}[/] rejected.")
    else:
        _print_error(f"Proposal {proposal_id!r} not found or not in a rejectable status.")


@evolve.command("apply")
@click.argument("proposal_id")
@click.option(
    "--no-restart",
    is_flag=True,
    default=False,
    help="Skip automatic process restart after successful apply.",
)
@click.pass_context
def evolve_apply(ctx: click.Context, proposal_id: str, no_restart: bool) -> None:
    """Apply an approved evolution (runs tests, commits on success).

    After a successful apply the process restarts itself so the new code
    takes effect immediately.  Pass --no-restart to skip the restart.
    """
    from missy.agent.code_evolution import CodeEvolutionManager, restart_process

    _load_subsystems(ctx.obj["config_path"])
    mgr = CodeEvolutionManager()
    prop = mgr.get(proposal_id)
    if not prop:
        _print_error(f"Proposal {proposal_id!r} not found.")
        return
    if prop.status.value != "approved":
        _print_error(f"Proposal is '{prop.status.value}', not approved. Approve it first.")
        return
    console.print(f"[bold]Applying evolution {proposal_id}...[/] (running tests)")
    result = mgr.apply(proposal_id)
    if result["success"]:
        _print_success(result["message"])
        if no_restart:
            console.print(
                "[dim]Skipping restart (--no-restart). Restart manually to load changes.[/]"
            )
        else:
            console.print("[bold cyan]Restarting to load evolved code...[/]")
            restart_process()
    else:
        _print_error(result["message"])
        if result.get("test_output"):
            console.print(
                Panel(result["test_output"][-1000:], title="Test Output", border_style="red")
            )


@evolve.command("rollback")
@click.argument("proposal_id")
@click.pass_context
def evolve_rollback(ctx: click.Context, proposal_id: str) -> None:
    """Rollback a previously applied evolution via git revert."""
    from missy.agent.code_evolution import CodeEvolutionManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = CodeEvolutionManager()
    result = mgr.rollback(proposal_id)
    if result["success"]:
        _print_success(result["message"])
    else:
        _print_error(result["message"])


# ---------------------------------------------------------------------------
# missy patches
# ---------------------------------------------------------------------------


@cli.group()
def patches() -> None:
    """Prompt patch (self-tuning) management."""


@patches.command("list")
@click.pass_context
def patches_list(ctx: click.Context) -> None:
    """List all prompt patches."""
    from missy.agent.prompt_patches import PromptPatchManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = PromptPatchManager()
    all_patches = mgr.list_all()
    if not all_patches:
        console.print("[dim]No patches.[/]")
        return
    table = Table(title="Prompt Patches", show_lines=True)
    table.add_column("ID")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Success Rate", justify="right")
    table.add_column("Content")
    for p in all_patches:
        rate = f"{p.success_rate:.0%}" if p.applications > 0 else "—"
        table.add_row(p.id, p.patch_type.value, p.status.value, rate, p.content[:60])
    console.print(table)


@patches.command("approve")
@click.argument("patch_id")
@click.pass_context
def patches_approve(ctx: click.Context, patch_id: str) -> None:
    """Approve a proposed patch."""
    from missy.agent.prompt_patches import PromptPatchManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = PromptPatchManager()
    if mgr.approve(patch_id):
        _print_success(f"Patch [bold]{patch_id}[/] approved.")
    else:
        _print_error(f"Patch {patch_id!r} not found or not awaiting review.")


@patches.command("reject")
@click.argument("patch_id")
@click.pass_context
def patches_reject(ctx: click.Context, patch_id: str) -> None:
    """Reject a proposed patch."""
    from missy.agent.prompt_patches import PromptPatchManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = PromptPatchManager()
    if mgr.reject(patch_id):
        _print_success(f"Patch [bold]{patch_id}[/] rejected.")
    else:
        _print_error(f"Patch {patch_id!r} not found or not awaiting review.")


# ---------------------------------------------------------------------------
# missy mcp
# ---------------------------------------------------------------------------


@cli.group()
def mcp() -> None:
    """Model Context Protocol (MCP) server management."""


@mcp.command("list")
@click.pass_context
def mcp_list(ctx: click.Context) -> None:
    """List configured MCP servers."""
    from missy.mcp.manager import McpManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = McpManager()
    # McpManager() starts with an empty in-memory client dict -- list_servers()
    # only reflects self._clients, which stays empty until connect_all() loads
    # and connects every server declared in mcp.json. Without this call, `missy
    # mcp list` always reported "No MCP servers configured" regardless of
    # actual state (matching the pattern already correctly applied by `mcp
    # pin`, below).
    mgr.connect_all()
    servers = mgr.list_servers()
    if not servers:
        console.print("[dim]No MCP servers configured. Add one with missy mcp add.[/]")
        mgr.shutdown()
        return
    table = Table(title="MCP Servers", show_lines=True)
    table.add_column("Name", style="bold")
    table.add_column("Alive", justify="center")
    table.add_column("Tools", justify="right")
    for s in servers:
        alive = Text("yes", style="green") if s["alive"] else Text("no", style="red")
        table.add_row(s["name"], alive, str(s["tools"]))
    console.print(table)
    mgr.shutdown()


@mcp.command("add")
@click.argument("name")
@click.option("--command", default=None, help="Stdio command to launch the MCP server.")
@click.option("--url", default=None, help="HTTP URL for the MCP server.")
@click.pass_context
def mcp_add(ctx: click.Context, name: str, command: str | None, url: str | None) -> None:
    """Connect to a new MCP server."""
    from missy.mcp.manager import McpManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = McpManager()
    # add_server() persists via _save_config(), which rewrites mcp.json from
    # scratch using only the currently in-memory self._clients -- without
    # first loading every already-configured server via connect_all(), adding
    # one new server silently destroyed every previously-configured one.
    mgr.connect_all()
    try:
        client = mgr.add_server(name, command=command, url=url)
        _print_success(f"Connected to MCP server [bold]{name}[/] ({len(client.tools)} tools).")
    except Exception as exc:
        _print_error(f"Failed to connect: {exc}")
        sys.exit(1)
    finally:
        mgr.shutdown()


@mcp.command("remove")
@click.argument("name")
@click.pass_context
def mcp_remove_cmd(ctx: click.Context, name: str) -> None:
    """Disconnect and remove an MCP server."""
    from missy.mcp.manager import McpManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = McpManager()
    # remove_server() is a no-op unless `name` is already in self._clients --
    # without connect_all() first, self._clients is always empty on a fresh
    # CLI process, so this was silently doing nothing to mcp.json.
    mgr.connect_all()
    mgr.remove_server(name)
    mgr.shutdown()
    _print_success(f"MCP server [bold]{name}[/] removed.")


@mcp.command("pin")
@click.argument("name")
@click.pass_context
def mcp_pin_cmd(ctx: click.Context, name: str) -> None:
    """Pin the tool manifest digest for an MCP server.

    Connects to the server, computes the SHA-256 digest of its tool
    manifest, and writes it back to mcp.json.  Future connections will
    refuse to load if the digest changes.
    """
    from missy.mcp.manager import McpManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = McpManager()
    mgr.connect_all()

    try:
        digest = mgr.pin_server_digest(name)
    except KeyError:
        _print_error(f"MCP server {name!r} is not connected.")
        sys.exit(1)
    except Exception as exc:
        _print_error(f"Failed to pin digest: {exc}")
        sys.exit(1)

    _print_success(f"Pinned digest for [bold]{name}[/]:\n  {digest}")
    mgr.shutdown()


# ---------------------------------------------------------------------------
# missy devices
# ---------------------------------------------------------------------------


@cli.group()
def devices() -> None:
    """Edge node device management commands."""


@devices.command("list")
@click.pass_context
def devices_list(ctx: click.Context) -> None:
    """List all registered edge nodes."""
    from datetime import datetime

    from missy.channels.voice.registry import DeviceRegistry

    reg = DeviceRegistry()
    reg.load()
    nodes = reg.list_nodes()
    if not nodes:
        console.print(
            "[dim]No edge nodes registered. Use a node's pair command to initiate pairing.[/]"
        )
        return
    table = Table(title="Edge Nodes", show_lines=True)
    table.add_column("Node ID", style="bold")
    table.add_column("Name")
    table.add_column("Room")
    table.add_column("Status", justify="center")
    table.add_column("Policy")
    table.add_column("Last Seen")
    table.add_column("Paired", justify="center")
    for node in nodes:
        paired = node.paired
        status_text = Text("paired", style="green") if paired else Text("pending", style="yellow")
        paired_text = Text("yes", style="green") if paired else Text("no", style="yellow")
        last_seen_ts = node.last_seen
        if last_seen_ts:
            last_seen = datetime.fromtimestamp(last_seen_ts).strftime("%Y-%m-%d %H:%M")
        else:
            last_seen = "never"
        table.add_row(
            node.node_id[:8],
            node.friendly_name,
            node.room,
            status_text,
            node.policy_mode,
            last_seen,
            paired_text,
        )
    console.print(table)


@devices.command("pair")
@click.option(
    "--node-id", default=None, help="Node ID to approve (omit to list pending and prompt)."
)
@click.pass_context
def devices_pair(ctx: click.Context, node_id: str | None) -> None:
    """Approve a pending edge node pairing request.

    If --node-id is omitted, lists pending nodes and prompts for selection.
    Prints the auth token on approval — the token is shown only once.
    """
    from missy.channels.voice.pairing import PairingManager
    from missy.channels.voice.registry import DeviceRegistry

    reg = DeviceRegistry()
    reg.load()
    mgr = PairingManager(registry=reg)

    if node_id is None:
        pending = reg.list_pending()
        if not pending:
            console.print("[dim]No pending pairing requests.[/]")
            return
        console.print("[bold]Pending nodes:[/]")
        for i, node in enumerate(pending):
            console.print(f"  [{i}] {node.node_id[:8]}  {node.friendly_name}  {node.room}")
        idx = click.prompt("Select index to approve", type=int)
        if idx < 0 or idx >= len(pending):
            _print_error("Invalid selection.")
            sys.exit(1)
        node_id = pending[idx].node_id

    try:
        token = mgr.approve_pairing(node_id)
        _print_success(f"Node [bold]{node_id[:8]}[/] approved.")
        console.print(f"[bold yellow]Auth token (shown once):[/] [green]{token}[/]")
    except Exception as exc:
        _print_error(f"Failed to approve node: {exc}")
        sys.exit(1)


@devices.command("unpair")
@click.argument("node_id")
@click.confirmation_option(prompt="Remove this node?")
@click.pass_context
def devices_unpair(ctx: click.Context, node_id: str) -> None:
    """Remove a paired edge node."""
    from missy.channels.voice.registry import DeviceRegistry

    reg = DeviceRegistry()
    reg.load()
    if reg.get_node(node_id) is None:
        _print_error(f"Node {node_id!r} not found.")
        sys.exit(1)
    reg.remove_node(node_id)
    _print_success(f"Node [bold]{node_id[:8]}[/] removed.")


@devices.command("status")
@click.pass_context
def devices_status(ctx: click.Context) -> None:
    """Show online/offline status of all edge nodes."""
    from datetime import datetime

    from missy.channels.voice.registry import DeviceRegistry

    reg = DeviceRegistry()
    reg.load()
    nodes = reg.list_nodes()
    if not nodes:
        console.print("[dim]No edge nodes registered.[/]")
        return
    table = Table(title="Edge Node Status", show_lines=True)
    table.add_column("Node ID", style="bold")
    table.add_column("Name")
    table.add_column("Room")
    table.add_column("Status", justify="center")
    table.add_column("Last Seen")
    table.add_column("Occupancy", justify="center")
    table.add_column("Noise Level", justify="right")
    for node in nodes:
        online = node.status == "online"
        status_text = Text("online", style="green") if online else Text("offline", style="red")
        last_seen_ts = node.last_seen
        if last_seen_ts:
            last_seen = datetime.fromtimestamp(last_seen_ts).strftime("%Y-%m-%d %H:%M")
        else:
            last_seen = "never"
        occupancy = node.sensor_data.get("occupancy")
        occupancy_str = str(occupancy) if occupancy is not None else "-"
        noise = node.sensor_data.get("noise_level")
        noise_str = f"{noise:.1f} dB" if noise is not None else "-"
        table.add_row(
            node.node_id[:8],
            node.friendly_name,
            node.room,
            status_text,
            last_seen,
            occupancy_str,
            noise_str,
        )
    console.print(table)


@devices.command("policy")
@click.argument("node_id")
@click.option(
    "--mode",
    type=click.Choice(["full", "safe-chat", "muted"]),
    required=True,
    help="Policy mode for this node.",
)
@click.pass_context
def devices_policy(ctx: click.Context, node_id: str, mode: str) -> None:
    """Set the policy mode for an edge node."""
    from missy.channels.voice.registry import DeviceRegistry

    reg = DeviceRegistry()
    reg.load()
    try:
        reg.update_node(node_id, policy_mode=mode)
        _print_success(f"Node [bold]{node_id[:8]}[/] policy set to [bold]{mode}[/].")
    except KeyError:
        _print_error(f"Node {node_id!r} not found.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# missy voice
# ---------------------------------------------------------------------------


@cli.group()
def voice() -> None:
    """Voice channel management commands."""


@voice.command("status")
@click.pass_context
def voice_status(ctx: click.Context) -> None:
    """Show voice channel configuration and STT/TTS engine status."""
    import shutil

    from missy.channels.voice.registry import DeviceRegistry

    # Check faster-whisper availability
    try:
        import faster_whisper  # noqa: F401

        whisper_status = Text("installed", style="green")
    except ImportError:
        whisper_status = Text("not installed", style="red")

    # Check piper binary availability
    piper_bin = shutil.which("piper")
    piper_status = Text(piper_bin or "not found", style="green" if piper_bin else "red")

    # Load config for voice channel settings
    config_path = ctx.obj.get("config_path") if ctx.obj else None
    voice_cfg: dict = {}
    if config_path:
        try:
            import yaml

            cfg_file = Path(config_path).expanduser()
            if cfg_file.exists():
                with cfg_file.open() as fh:
                    raw = yaml.safe_load(fh) or {}
                voice_cfg = raw.get("voice", {})
        except Exception:
            logger.debug("voice status: failed to load voice config", exc_info=True)

    host = voice_cfg.get("host", "0.0.0.0")
    port = voice_cfg.get("port", 8765)
    stt_engine = voice_cfg.get("stt", {}).get("engine", "faster-whisper")
    stt_model = voice_cfg.get("stt", {}).get("model", "base.en")
    tts_engine = voice_cfg.get("tts", {}).get("engine", "piper")
    tts_voice = voice_cfg.get("tts", {}).get("voice", "en_US-lessac-medium")

    reg = DeviceRegistry()
    reg.load()
    paired_count = len(reg.list_paired())

    table = Table(title="Voice Channel Status", show_lines=True)
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    table.add_row("Gateway", f"{host}:{port}")
    table.add_row("STT Engine", stt_engine)
    table.add_row("STT Model", stt_model)
    table.add_row("TTS Engine", tts_engine)
    table.add_row("TTS Voice", tts_voice)
    table.add_row("faster-whisper", whisper_status)
    table.add_row("piper binary", piper_status)
    table.add_row("Paired nodes", str(paired_count))
    console.print(table)


@voice.command("test")
@click.argument("node_id")
@click.option(
    "--text",
    default="Missy voice channel test. Audio is working correctly.",
    help="Text to synthesize and send.",
)
@click.pass_context
def voice_test(ctx: click.Context, node_id: str, text: str) -> None:
    """Send a test TTS phrase to an edge node.

    Synthesizes the text using the configured TTS engine and notes in the
    output what would be sent.  Since this runs outside a live gateway
    session, it validates the TTS engine is functional rather than sending
    to a live node.
    """
    import asyncio
    import time

    from missy.channels.voice.registry import DeviceRegistry
    from missy.channels.voice.tts.base import AudioBuffer  # noqa: F401 (type hint)
    from missy.channels.voice.tts.piper import PiperTTS

    # Validate node exists
    reg = DeviceRegistry()
    reg.load()
    node = next((n for n in reg.list_nodes() if n.node_id.startswith(node_id)), None)
    if node is None:
        _print_error(f"Node {node_id!r} not found in registry.")
        sys.exit(1)

    # Resolve the configured TTS voice so the test uses the same model the
    # running gateway would.
    tts_voice = "en_US-lessac-medium"
    config_path = ctx.obj.get("config_path") if ctx.obj else None
    if config_path:
        try:
            import yaml

            cfg_file = Path(config_path).expanduser()
            if cfg_file.exists():
                raw = yaml.safe_load(cfg_file.read_text()) or {}
                tts_voice = raw.get("voice", {}).get("tts", {}).get("voice", tts_voice)
        except Exception:
            logger.debug("voice test: failed to load voice config", exc_info=True)

    console.print(f"Synthesizing test phrase for node [bold]{node_id[:8]}[/]...")
    try:
        tts = PiperTTS(voice=tts_voice)
        # PiperTTS.synthesize() requires load() to have resolved the binary and
        # model file first; without this the command always failed with
        # "PiperTTS.load() must be called before synthesize()" even when piper
        # was installed.
        tts.load()

        async def _synth() -> AudioBuffer:
            return await tts.synthesize(text)

        start = time.monotonic()
        audio = asyncio.run(_synth())
        elapsed = time.monotonic() - start

        # AudioBuffer.duration_ms is derived from the real PCM length and
        # sample rate, so no manual byte-math estimate is needed.
        _print_success(
            f"TTS synthesis succeeded in {elapsed:.2f}s — "
            f"{len(audio.data):,} bytes, ~{audio.duration_ms / 1000:.1f}s of audio."
        )
        console.print(
            f"[dim]Would send to node {node.friendly_name or node_id[:8]} "
            f"in room {node.room or 'unknown'} via gateway WebSocket.[/]"
        )
    except FileNotFoundError:
        _print_error("piper binary not found in PATH. Install piper and ensure it is on your PATH.")
        sys.exit(1)
    except Exception as exc:
        _print_error(f"TTS synthesis failed: {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# missy presets
# ---------------------------------------------------------------------------


@cli.group("presets")
def presets_group() -> None:
    """Manage network policy presets."""


@presets_group.command("list")
def presets_list() -> None:
    """List all built-in network policy presets and their contents."""
    from missy.policy.presets import PRESETS

    table = Table(title="Network Policy Presets", show_lines=True)
    table.add_column("Name", style="bold")
    table.add_column("Hosts")
    table.add_column("Domains")
    table.add_column("CIDRs")

    for name in sorted(PRESETS):
        preset = PRESETS[name]
        table.add_row(
            name,
            ", ".join(preset.get("hosts", [])) or "[dim]—[/]",
            ", ".join(preset.get("domains", [])) or "[dim]—[/]",
            ", ".join(preset.get("cidrs", [])) or "[dim]—[/]",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# missy mcp pin
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# missy config
# ---------------------------------------------------------------------------


@cli.group("config")
def config_group() -> None:
    """Config backup, diff, and rollback commands."""


@config_group.command("backups")
@click.pass_context
def config_backups(ctx: click.Context) -> None:
    """List all config backups."""
    from missy.config.plan import list_backups

    config_path = ctx.obj.get("config_path", DEFAULT_CONFIG)
    backups = list_backups(config_path=config_path)
    if not backups:
        console.print("[dim]No config backups found.[/]")
        return

    table = Table(title="Config Backups", show_lines=True)
    table.add_column("File", style="bold")
    table.add_column("Size", justify="right")

    for b in backups:
        table.add_row(b.name, f"{b.stat().st_size:,} bytes")

    console.print(table)


@config_group.command("diff")
@click.pass_context
def config_diff(ctx: click.Context) -> None:
    """Show unified diff between current config and latest backup."""
    from missy.config.plan import diff_configs, list_backups

    config_path = ctx.obj.get("config_path", DEFAULT_CONFIG)
    config_file = Path(config_path).expanduser()

    if not config_file.exists():
        _print_error(f"Config file not found: {config_file}")
        sys.exit(1)

    backups = list_backups(config_path=config_file)
    if not backups:
        console.print("[dim]No backups to compare against.[/]")
        return

    latest = backups[-1]
    diff_text = diff_configs(latest, config_file)
    if not diff_text:
        console.print("[dim]No differences between current config and latest backup.[/]")
    else:
        console.print(diff_text)


@config_group.command("rollback")
@click.pass_context
def config_rollback(ctx: click.Context) -> None:
    """Restore config from the latest backup (current config is backed up first)."""
    from missy.config.plan import rollback

    config_path = ctx.obj.get("config_path", DEFAULT_CONFIG)
    config_file = Path(config_path).expanduser()

    restored = rollback(config_file)
    if restored is None:
        console.print("[dim]No backups available to restore from.[/]")
    else:
        _print_success(f"Config restored from [bold]{restored.name}[/].")


@config_group.command("plan")
@click.pass_context
def config_plan(ctx: click.Context) -> None:
    """Show what would change if a new config were applied (diff vs latest backup)."""
    from missy.config.plan import diff_configs, list_backups

    config_path = ctx.obj.get("config_path", DEFAULT_CONFIG)
    config_file = Path(config_path).expanduser()

    if not config_file.exists():
        console.print("[dim]No config file exists yet. Run 'missy setup' to create one.[/]")
        return

    backups = list_backups(config_path=config_file)
    if not backups:
        console.print("[dim]No previous backups. Current config is the baseline.[/]")
        return

    latest = backups[-1]
    diff_text = diff_configs(latest, config_file)
    if not diff_text:
        console.print("[green]Config matches the latest backup — no changes pending.[/]")
    else:
        console.print("[bold]Changes since last backup:[/]\n")
        console.print(diff_text)


# ---------------------------------------------------------------------------
# Security tools
# ---------------------------------------------------------------------------


@cli.group("security")
def security_group() -> None:
    """Security tools and auditing."""


@security_group.command("scan")
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Show full description for each finding."
)
@click.option(
    "--json-output", "json_output", is_flag=True, default=False, help="Output results as JSON."
)
@click.option(
    "--severity",
    "min_severity",
    type=click.Choice(["critical", "high", "medium", "low", "info"]),
    default=None,
    help="Only show findings at or above this severity level.",
)
@click.pass_context
def security_scan(
    ctx: click.Context,
    verbose: bool,
    json_output: bool,
    min_severity: str | None,
) -> None:
    """Scan the Missy installation for security issues.

    Checks configuration, network policy, filesystem policy, shell policy,
    MCP servers, tool permissions, secrets management, agent identity, file
    permissions, and known vulnerability patterns.

    Exits with code 1 when any CRITICAL findings are found.
    """
    import json as _json

    from missy.security.scanner import _SEVERITY_ORDER, SecurityScanner, Severity

    config_path = ctx.obj["config_path"]

    # Let the scanner lazily load the config itself (scan_all() handles a
    # missing/malformed file by emitting a SEC-000 finding rather than
    # raising). Pre-loading it here and passing it in via `config=` would
    # skip SecurityScanner's own _load_raw_provider_keys() step, which is
    # what lets SEC-002/SEC-060 correctly exempt "vault://KEY" references
    # from being flagged as plaintext secrets.
    scanner = SecurityScanner(config_path=config_path)
    result = scanner.scan_all()

    # Apply severity filter
    if min_severity is not None:
        threshold = _SEVERITY_ORDER[Severity(min_severity)]
        result.findings[:] = [
            f for f in result.findings if _SEVERITY_ORDER[f.severity] <= threshold
        ]
        # Recompute summary for filtered set
        for sev in Severity:
            result.summary[sev.value] = sum(1 for f in result.findings if f.severity == sev)

    if json_output:
        console.print_json(_json.dumps(result.to_json(), indent=2))
        if result.has_critical:
            sys.exit(1)
        return

    # Rich-formatted output
    from rich.rule import Rule

    _SEV_COLOUR = {
        "critical": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "cyan",
        "info": "dim",
    }

    console.print(Rule("[bold]Security Scan Results[/]"))
    console.print(
        f"  Scanned at: [dim]{result.scanned_at}[/] ([dim]{result.scan_duration_ms:.0f}ms[/])\n"
    )

    # Severity summary bar
    parts: list[str] = []
    for sev in Severity:
        count = result.summary.get(sev.value, 0)
        colour = _SEV_COLOUR[sev.value]
        parts.append(f"[{colour}]{sev.value.upper()} ({count})[/]")
    console.print("  " + "  |  ".join(parts))
    console.print()

    if not result.findings:
        console.print("  [green]No findings — installation looks secure.[/]\n")
        return

    for finding in result.findings:
        sev_colour = _SEV_COLOUR.get(finding.severity.value, "dim")
        sev_label = f"[{sev_colour}][{finding.severity.value.upper()}][/]"
        console.print(f"{sev_label} [bold]{finding.id}[/]: {finding.title}")
        if verbose and finding.description:
            for line in finding.description.splitlines():
                console.print(f"  [dim]{line}[/]")
        console.print(f"  [cyan]Recommendation:[/] {finding.recommendation}")
        if verbose and finding.details:
            for key, value in finding.details.items():
                console.print(f"  [dim]{key}[/]: {value}")
        console.print()

    # Exit summary
    if result.has_critical:
        console.print(
            f"[bold red]CRITICAL findings: {result.critical_count}.[/] "
            "Address these immediately before running Missy."
        )
        sys.exit(1)
    else:
        high = result.summary.get("high", 0)
        if high:
            console.print(f"[yellow]{high} HIGH finding(s) found.[/] Review and remediate.")


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


@cli.group("sandbox")
def sandbox_group() -> None:
    """Container-per-session sandbox management."""


@sandbox_group.command("status")
@click.pass_context
def sandbox_status(ctx: click.Context) -> None:
    """Check Docker availability and show container sandbox configuration."""
    from missy.security.container import ContainerConfig, ContainerSandbox

    cfg = _load_subsystems(ctx.obj["config_path"])

    container_cfg: ContainerConfig | None = getattr(cfg, "container", None)

    table = Table(title="Container Sandbox Status", show_lines=True)
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    # Docker availability
    docker_ok = ContainerSandbox.is_available()
    docker_status = (
        Text("available", style="green") if docker_ok else Text("not found", style="red")
    )
    table.add_row("Docker", docker_status)

    if container_cfg is not None:
        enabled_style = "green" if container_cfg.enabled else "dim"
        table.add_row("enabled", Text(str(container_cfg.enabled), style=enabled_style))
        table.add_row("image", container_cfg.image)
        table.add_row("memory_limit", container_cfg.memory_limit)
        table.add_row("cpu_quota", str(container_cfg.cpu_quota))
        table.add_row("network_mode", container_cfg.network_mode)
    else:
        table.add_row("enabled", Text("false (not configured)", style="dim"))

    console.print(table)


# ---------------------------------------------------------------------------
# missy hatch
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--non-interactive", is_flag=True, default=False, help="Skip prompts, use defaults.")
@click.pass_context
def hatch(ctx: click.Context, non_interactive: bool) -> None:
    """Run the hatching process — first-run bootstrapping for Missy.

    Validates the environment, initialises configuration, verifies providers,
    sets up security baselines, generates a default persona, and seeds memory.
    Safe to re-run: skips already-completed steps.
    """
    from missy.agent.hatching import HatchingManager, HatchingStatus

    mgr = HatchingManager()

    if mgr.is_hatched():
        console.print(
            "[green]Missy is already hatched![/] Use [bold]missy hatch --non-interactive[/] to re-verify."
        )
        state = mgr.get_state()
        console.print(f"  Hatched at: [bold]{state.completed_at}[/]")
        console.print(f"  Steps: {', '.join(state.steps_completed)}")
        return

    console.print(
        Panel(
            "[bold cyan]Hatching Missy[/]\n\n"
            "This will set up your environment, initialise configuration,\n"
            "verify providers, create a persona, and seed memory.",
            title="[cyan]Hatching[/]",
            border_style="cyan",
        )
    )

    interactive = not non_interactive
    state = mgr.run_hatching(interactive=interactive)

    if state.status == HatchingStatus.HATCHED:
        _print_success(
            f"Missy has hatched successfully!\n\n"
            f"  Steps completed: {len(state.steps_completed)}\n"
            f"  Persona generated: {state.persona_generated}\n"
            f"  Memory seeded: {state.memory_seeded}\n\n"
            f"Run [bold cyan]missy persona show[/] to see your persona.\n"
            f"Run [bold cyan]missy run[/] to start chatting."
        )
    elif state.status == HatchingStatus.FAILED:
        _print_error(
            f"Hatching failed: {state.error}",
            hint="Check the hatching log with: missy hatch --non-interactive",
        )
        sys.exit(1)
    else:
        console.print(f"[yellow]Hatching status: {state.status.value}[/]")


# ---------------------------------------------------------------------------
# missy persona
# ---------------------------------------------------------------------------


@cli.group()
def persona() -> None:
    """View and manage Missy's persona — identity, tone, and response style."""


@persona.command("show")
def persona_show() -> None:
    """Display the current persona configuration."""
    from missy.agent.persona import PersonaManager

    mgr = PersonaManager()
    p = mgr.get_persona()

    table = Table(title=f"Persona: {p.name} (v{p.version})", show_lines=True)
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Name", p.name)
    table.add_row("Tone", ", ".join(p.tone))
    table.add_row("Personality", ", ".join(p.personality_traits))
    table.add_row("Tendencies", "\n".join(f"- {t}" for t in p.behavioral_tendencies))
    table.add_row("Style Rules", "\n".join(f"- {r}" for r in p.response_style_rules))
    table.add_row("Boundaries", "\n".join(f"- {b}" for b in p.boundaries))
    table.add_row("Identity", p.identity_description)

    console.print(table)


@persona.command("edit")
@click.option("--name", default=None, help="Set persona name.")
@click.option("--tone", default=None, help="Comma-separated tone values.")
@click.option("--identity", default=None, help="Identity description text.")
def persona_edit(name: str | None, tone: str | None, identity: str | None) -> None:
    """Edit persona fields.

    \b
    Examples:
        missy persona edit --name "Missy"
        missy persona edit --tone "friendly,casual,technical"
        missy persona edit --identity "A helpful Linux assistant"
    """
    from missy.agent.persona import PersonaManager

    mgr = PersonaManager()
    updates: dict[str, Any] = {}

    if name is not None:
        updates["name"] = name
    if tone is not None:
        updates["tone"] = [t.strip() for t in tone.split(",") if t.strip()]
    if identity is not None:
        updates["identity_description"] = identity

    if not updates:
        console.print("[yellow]No changes specified. Use --name, --tone, or --identity.[/]")
        return

    mgr.update(**updates)
    mgr.save()
    console.print(f"[green]Persona updated (v{mgr.version}).[/]")
    for key, val in updates.items():
        console.print(f"  {key}: {val}")


@persona.command("reset")
def persona_reset() -> None:
    """Reset the persona to factory defaults."""
    from missy.agent.persona import PersonaManager

    mgr = PersonaManager()
    mgr.reset()
    console.print(f"[green]Persona reset to defaults (v{mgr.version}).[/]")


@persona.command("backups")
def persona_backups() -> None:
    """List available persona backups."""
    from missy.agent.persona import PersonaManager

    mgr = PersonaManager()
    backups = mgr.list_backups()
    if not backups:
        console.print("[yellow]No persona backups found.[/]")
        return

    table = Table(title="Persona Backups")
    table.add_column("#", style="dim")
    table.add_column("File")
    table.add_column("Size")
    table.add_column("Modified")

    for idx, bk in enumerate(backups, 1):
        stat = bk.stat()
        mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        table.add_row(str(idx), bk.name, f"{stat.st_size} B", mod_time)

    console.print(table)


@persona.command("diff")
def persona_diff() -> None:
    """Show diff between current persona and latest backup."""
    from missy.agent.persona import PersonaManager

    mgr = PersonaManager()
    diff_text = mgr.diff()
    if not diff_text:
        console.print("[dim]No differences (or no backups to compare against).[/]")
        return
    console.print(diff_text)


@persona.command("rollback")
def persona_rollback() -> None:
    """Restore persona from the latest backup."""
    from missy.agent.persona import PersonaManager

    mgr = PersonaManager()
    restored = mgr.rollback()
    if restored is None:
        console.print("[yellow]No backups available to rollback to.[/]")
        return
    console.print(f"[green]Persona restored from {restored.name} (v{mgr.version}).[/]")


@persona.command("log")
@click.option("--limit", "-n", default=20, help="Number of recent entries to show.")
def persona_log(limit: int) -> None:
    """Show persona change audit log."""
    from missy.agent.persona import PersonaManager

    mgr = PersonaManager()
    entries = mgr.get_audit_log()
    if not entries:
        console.print("[dim]No persona audit log entries.[/]")
        return

    recent = entries[-limit:]
    table = Table(title="Persona Audit Log")
    table.add_column("Time", style="dim")
    table.add_column("Action")
    table.add_column("Version")
    table.add_column("Name")

    for entry in recent:
        ts = entry.get("timestamp", "?")[:19]
        table.add_row(
            ts,
            entry.get("action", "?"),
            str(entry.get("version", "?")),
            entry.get("name", "?"),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# missy vision
# ---------------------------------------------------------------------------


@cli.group()
def vision() -> None:
    """Vision subsystem — camera discovery, capture, and visual analysis."""


@vision.command("health")
def vision_health_cmd() -> None:
    """Show vision subsystem health statistics (includes persisted history)."""
    from pathlib import Path

    from missy.vision.health_monitor import get_health_monitor

    monitor = get_health_monitor()

    # Load persisted history if available
    import contextlib

    persist_path = Path.home() / ".missy" / "vision_health.db"
    if persist_path.exists():
        with contextlib.suppress(Exception):
            monitor.load(persist_path)

    report = monitor.get_health_report()

    status = report["overall_status"]
    status_color = {"healthy": "green", "degraded": "yellow", "unhealthy": "red"}.get(status, "dim")

    console.print(f"[bold]Vision Health:[/] [{status_color}]{status.upper()}[/]")
    console.print(
        f"  Total captures: {report['total_captures']}  "
        f"Failures: {report['total_failures']}  "
        f"Recent success rate: {report['recent_success_rate']:.0%}"
    )
    console.print(f"  Uptime: {report['uptime_seconds']:.0f}s")

    if report["devices"]:
        console.print("\n[bold]Devices:[/]")
        for device, stats in report["devices"].items():
            dev_color = {"healthy": "green", "degraded": "yellow", "unhealthy": "red"}.get(
                stats["status"], "dim"
            )
            console.print(
                f"  [{dev_color}]{stats['status'].upper():>9}[/]  {device}  "
                f"({stats['total_captures']} captures, "
                f"{stats['success_rate']:.0%} success, "
                f"{stats['average_latency_ms']:.0f}ms avg)"
            )
            if stats["consecutive_failures"] > 0:
                console.print(
                    f"           [yellow]{stats['consecutive_failures']} consecutive failures: "
                    f"{stats['last_error']}[/]"
                )

    if report["warnings"]:
        console.print("\n[bold yellow]Warnings:[/]")
        for w in report["warnings"]:
            console.print(f"  [yellow]• {w}[/]")

    recommendations = monitor.get_recommendations()
    if recommendations:
        console.print("\n[bold cyan]Recommendations:[/]")
        for r in recommendations:
            console.print(f"  [cyan]→ {r}[/]")


@vision.command("devices")
def vision_devices() -> None:
    """Enumerate and diagnose available cameras."""
    from missy.vision.discovery import KNOWN_CAMERAS, CameraDiscovery

    disc = CameraDiscovery()
    cameras = disc.discover(force=True)

    if not cameras:
        console.print("[yellow]No cameras detected.[/]")
        console.print(
            "\n[dim]Troubleshooting:[/]\n"
            "  1. Check USB connection\n"
            "  2. Run [bold]lsusb[/] to verify device is detected\n"
            "  3. Ensure user is in 'video' group: [bold]sudo usermod -aG video $USER[/]\n"
            "  4. Run [bold]missy vision doctor[/] for full diagnostics"
        )
        return

    table = Table(title=f"Cameras ({len(cameras)} found)")
    table.add_column("Device", style="bold")
    table.add_column("Name")
    table.add_column("USB ID")
    table.add_column("Bus Info", style="dim")
    table.add_column("Known", style="green")

    for cam in cameras:
        known = KNOWN_CAMERAS.get(cam.usb_id, "")
        table.add_row(
            cam.device_path,
            cam.name,
            cam.usb_id,
            cam.bus_info[:40] if cam.bus_info else "",
            known or "[dim]—[/]",
        )

    console.print(table)

    preferred = disc.find_preferred()
    if preferred:
        console.print(f"\n[green]Preferred camera:[/] {preferred.name} ({preferred.device_path})")


@vision.command("capture")
@click.option("--device", "-d", default=None, help="Device path (e.g. /dev/video0).")
@click.option("--output", "-o", default=None, help="Output file path (default: auto-generated).")
@click.option("--width", default=1920, help="Capture width.")
@click.option("--height", default=1080, help="Capture height.")
@click.option("--count", "-n", default=1, help="Number of frames to capture.")
@click.option("--burst", is_flag=True, help="Burst mode: capture multiple frames rapidly.")
@click.option("--best", is_flag=True, help="Capture a burst and save only the sharpest frame.")
def vision_capture(
    device: str | None,
    output: str | None,
    width: int,
    height: int,
    count: int,
    burst: bool,
    best: bool,
) -> None:
    """Capture one or more frames from a camera."""
    from missy.vision.capture import CameraHandle, CaptureConfig, CaptureError
    from missy.vision.discovery import find_preferred_camera

    if device is None:
        cam = find_preferred_camera()
        if cam is None:
            _print_error("No camera found", hint="Run missy vision devices")
            sys.exit(1)
        device = cam.device_path
        console.print(f"[dim]Using camera: {cam.name} ({device})[/]")

    config = CaptureConfig(width=width, height=height)
    handle = CameraHandle(device, config)

    try:
        handle.open()

        # Best mode: burst + pick sharpest
        if best:
            burst_count = max(count, 3)
            console.print(f"[dim]Burst capturing {burst_count} frames, selecting sharpest...[/]")
            result = handle.capture_best(burst_count=burst_count)
            if result.success:
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path = output or str(Path.home() / f".missy/captures/best_{ts}.jpg")
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                import cv2

                cv2.imwrite(out_path, result.image, [cv2.IMWRITE_JPEG_QUALITY, config.quality])
                console.print(f"[green]Best frame[/] {result.width}x{result.height} → {out_path}")
            else:
                console.print(f"[red]Failed[/]: {result.error}")
            return

        # Burst mode: capture rapid sequence
        if burst:
            console.print(f"[dim]Burst capturing {count} frames...[/]")
            results = handle.capture_burst(count=count, interval=0.3)
            for i, result in enumerate(results):
                if output:
                    base = Path(output)
                    out_path = str(base.parent / f"{base.stem}_{i:03d}{base.suffix}")
                else:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_path = str(Path.home() / f".missy/captures/burst_{ts}_{i:03d}.jpg")
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                if result.success:
                    import cv2

                    cv2.imwrite(out_path, result.image, [cv2.IMWRITE_JPEG_QUALITY, config.quality])
                    console.print(
                        f"[green]Frame {i}[/] {result.width}x{result.height} → {out_path}"
                    )
                else:
                    console.print(f"[red]Frame {i}[/]: {result.error}")
            succeeded = sum(1 for r in results if r.success)
            console.print(f"[dim]Burst complete: {succeeded}/{count} frames captured[/]")
            return

        # Standard capture
        for i in range(count):
            if output and count == 1:
                out_path = output
            elif output:
                base = Path(output)
                out_path = str(base.parent / f"{base.stem}_{i:03d}{base.suffix}")
            else:
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path = str(Path.home() / f".missy/captures/capture_{ts}_{i:03d}.jpg")

            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            result = handle.capture_to_file(out_path)

            if result.success:
                console.print(f"[green]Captured[/] {result.width}x{result.height} → {out_path}")
            else:
                console.print(f"[red]Failed[/] frame {i}: {result.error}")

    except CaptureError as exc:
        _print_error(str(exc), hint="Run missy vision doctor")
        sys.exit(1)
    finally:
        handle.close()


@vision.command("inspect")
@click.option("--device", "-d", default=None, help="Camera device path.")
@click.option("--file", "-f", "file_path", default=None, help="Image file to inspect.")
@click.option("--screenshot", "-s", is_flag=True, help="Capture a screenshot to inspect.")
@click.option("--context", "-c", default="", help="Additional context for analysis.")
def vision_inspect(
    device: str | None,
    file_path: str | None,
    screenshot: bool,
    context: str,
) -> None:
    """Run general visual analysis on an image source."""
    from missy.vision.sources import FileSource, ScreenshotSource, WebcamSource

    console.print("[bold]Visual Inspection[/]\n")

    try:
        if file_path:
            source = FileSource(file_path)
            console.print(f"[dim]Source: file ({file_path})[/]")
        elif screenshot:
            source = ScreenshotSource()
            console.print("[dim]Source: screenshot[/]")
        else:
            if device is None:
                from missy.vision.discovery import find_preferred_camera

                cam = find_preferred_camera()
                if cam is None:
                    _print_error("No camera found", hint="Use --file or --screenshot")
                    sys.exit(1)
                device = cam.device_path
            source = WebcamSource(device)
            console.print(f"[dim]Source: webcam ({device})[/]")

        frame = source.acquire()
        console.print(
            f"[green]Acquired[/] {frame.width}x{frame.height} image from {frame.source_type.value}"
        )

        # Run quality assessment
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        quality = pipeline.assess_quality(frame.image)

        table = Table(title="Image Quality")
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Resolution", f"{quality['width']}x{quality['height']}")
        table.add_row("Brightness", str(quality["brightness"]))
        table.add_row("Contrast", str(quality["contrast"]))
        table.add_row("Sharpness", str(quality["sharpness"]))
        table.add_row(
            "Quality",
            f"[green]{quality['quality']}[/]"
            if quality["quality"] == "good"
            else f"[yellow]{quality['quality']}[/]",
        )
        if quality["issues"]:
            table.add_row("Issues", ", ".join(quality["issues"]))

        console.print(table)
        console.print("\n[dim]For LLM-powered analysis, use missy vision review --mode general[/]")

    except Exception as exc:
        _print_error(f"Inspection failed: {exc}")
        sys.exit(1)


@vision.command("review")
@click.option("--device", "-d", default=None, help="Camera device path.")
@click.option("--file", "-f", "file_path", default=None, help="Image file to review.")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["general", "puzzle", "painting", "inspection"]),
    default="general",
    help="Analysis mode.",
)
@click.option("--context", "-c", default="", help="Additional context for analysis.")
@click.option("--config", "config_path", default=DEFAULT_CONFIG, help="Config file path.")
def vision_review(
    device: str | None,
    file_path: str | None,
    mode: str,
    context: str,
    config_path: str,
) -> None:
    """Run domain-specific visual analysis (puzzle help, painting feedback)."""
    from missy.vision.analysis import AnalysisMode, AnalysisPromptBuilder, AnalysisRequest
    from missy.vision.sources import FileSource, WebcamSource

    console.print(f"[bold]Visual Review[/] — mode: {mode}\n")

    try:
        # Acquire image
        if file_path:
            source = FileSource(file_path)
            console.print(f"[dim]Source: {file_path}[/]")
        else:
            if device is None:
                from missy.vision.discovery import find_preferred_camera

                cam = find_preferred_camera()
                if cam is None:
                    _print_error("No camera found", hint="Use --file to provide an image")
                    sys.exit(1)
                device = cam.device_path
            source = WebcamSource(device)
            console.print(f"[dim]Source: webcam ({device})[/]")

        frame = source.acquire()
        console.print(f"[green]Acquired[/] {frame.width}x{frame.height} image\n")

        # Preprocess
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        processed = pipeline.process(frame.image)

        # Build analysis prompt
        analysis_mode = AnalysisMode(mode)
        request = AnalysisRequest(
            image=processed,
            mode=analysis_mode,
            context=context,
        )
        builder = AnalysisPromptBuilder()
        prompt = builder.build_prompt(request)

        # Encode for LLM
        import base64

        import cv2

        _, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_image = base64.b64encode(buf.tobytes()).decode("ascii")

        # Send to provider with provider-specific formatting
        from missy.providers.registry import get_registry

        registry = get_registry()
        provider = registry.get_provider()
        provider_name = getattr(provider, "name", "anthropic")

        from missy.vision.provider_format import build_vision_message

        msg_dict = build_vision_message(provider_name, b64_image, prompt)

        from missy.providers.base import Message

        messages = [Message(role=msg_dict["role"], content=msg_dict["content"])]

        console.print("[dim]Analyzing...[/]\n")
        response = provider.complete(messages)

        # Log audit event
        try:
            from missy.vision.audit import audit_vision_analysis

            audit_vision_analysis(
                mode=mode,
                source_type=frame.source_type.value,
                trigger_reason="cli_command",
                success=True,
            )
        except Exception:
            pass

        console.print(
            Panel(
                response.text,
                title=f"[bold]{mode.title()} Analysis[/]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    except Exception as exc:
        _print_error(f"Review failed: {exc}")
        sys.exit(1)


@vision.command("doctor")
def vision_doctor_cmd() -> None:
    """Run vision subsystem diagnostics."""
    from missy.vision.doctor import VisionDoctor

    console.print("[bold]Vision Subsystem Diagnostics[/]\n")
    doc = VisionDoctor()
    report = doc.run_all()

    for result in report.results:
        if result.passed:
            icon = "[green]PASS[/]"
        elif result.severity == "warning":
            icon = "[yellow]WARN[/]"
        else:
            icon = "[red]FAIL[/]"

        console.print(f"  {icon}  {result.name}: {result.message}")

        if result.details and not result.passed:
            for key, val in result.details.items():
                if isinstance(val, list) and len(val) > 3:
                    console.print(f"       [dim]{key}: ({len(val)} items)[/]")
                else:
                    console.print(f"       [dim]{key}: {val}[/]")

    console.print()
    console.print(
        f"  [bold]Summary:[/] {report.passed} passed, "
        f"{report.warnings} warnings, {report.errors} errors"
    )

    if report.overall_healthy:
        console.print("\n  [green]Vision subsystem is healthy![/]")
    else:
        console.print("\n  [red]Vision subsystem has issues that need attention.[/]")


@vision.command("benchmark")
def vision_benchmark_cmd() -> None:
    """Show capture performance benchmark statistics."""
    from missy.vision.benchmark import get_benchmark

    bench = get_benchmark()
    report = bench.report()

    console.print("[bold]Vision Capture Benchmarks[/]\n")
    console.print(f"  Uptime: {report['uptime_seconds']:.1f}s\n")

    categories = report.get("categories", {})
    if not categories:
        console.print("  [dim]No benchmark data collected yet.[/]")
        console.print("  [dim]Run some captures to collect latency data.[/]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Category")
    table.add_column("Count", justify="right")
    table.add_column("Min (ms)", justify="right")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("Median (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("Max (ms)", justify="right")

    for cat, stats in categories.items():
        table.add_row(
            cat,
            str(stats["count"]),
            f"{stats['min_ms']:.1f}",
            f"{stats['mean_ms']:.1f}",
            f"{stats['median_ms']:.1f}",
            f"{stats['p95_ms']:.1f}",
            f"{stats['max_ms']:.1f}",
        )

    console.print(table)


@vision.command("validate")
def vision_validate_cmd() -> None:
    """Validate the current vision configuration."""
    from missy.vision.config_validator import validate_vision_config

    # Try to load config
    try:
        from missy.config.settings import load_config

        cfg = load_config()
        vision_cfg = {}
        if hasattr(cfg, "vision"):
            vc = cfg.vision
            for field_name in (
                "enabled",
                "capture_width",
                "capture_height",
                "warmup_frames",
                "max_retries",
                "auto_activate_threshold",
                "scene_memory_max_frames",
                "scene_memory_max_sessions",
                "preferred_device",
            ):
                if hasattr(vc, field_name):
                    vision_cfg[field_name] = getattr(vc, field_name)
    except Exception:
        vision_cfg = {}

    result = validate_vision_config(vision_cfg)

    console.print("[bold]Vision Configuration Validation[/]\n")

    if result.valid and not result.warnings:
        console.print("  [green]All settings are valid.[/]\n")
        return

    for issue in result.issues:
        if issue.severity == "error":
            icon = "[red]ERROR[/]"
        elif issue.severity == "warning":
            icon = "[yellow]WARN [/]"
        else:
            icon = "[blue]INFO [/]"

        console.print(f"  {icon}  {issue.field}: {issue.message}")
        if issue.current_value is not None:
            console.print(f"       [dim]Current: {issue.current_value}[/]")
        if issue.suggested_value is not None:
            console.print(f"       [dim]Suggested: {issue.suggested_value}[/]")

    console.print()
    if result.valid:
        console.print("  [green]Configuration is valid[/] (with warnings)")
    else:
        console.print(f"  [red]Configuration has {len(result.errors)} error(s)[/]")


@vision.command("memory")
def vision_memory_cmd() -> None:
    """Show vision scene memory usage."""
    from missy.vision.memory_usage import get_memory_tracker

    tracker = get_memory_tracker()
    report = tracker.update_from_scene_manager()

    console.print("[bold]Vision Scene Memory Usage[/]\n")

    d = report.to_dict()
    console.print(
        f"  Total: {d['total_mb']:.2f} MB / {d['limit_mb']:.2f} MB ({d['usage_fraction']:.1%})"
    )
    console.print(f"  Frames: {d['total_frames']}")
    console.print(f"  Sessions: {d['session_count']} ({d['active_sessions']} active)")

    if report.over_limit:
        console.print("  [red]OVER LIMIT — consider closing inactive sessions[/]")

    if d["sessions"]:
        console.print()
        table = Table(show_header=True, header_style="bold")
        table.add_column("Task ID")
        table.add_column("Frames", justify="right")
        table.add_column("Memory (MB)", justify="right")
        table.add_column("Active")

        for s in d["sessions"]:
            status = "[green]yes[/]" if s["active"] else "[dim]no[/]"
            table.add_row(
                s["task_id"],
                str(s["frame_count"]),
                f"{s['estimated_mb']:.2f}",
                status,
            )
        console.print(table)
    else:
        console.print("\n  [dim]No active scene sessions.[/]")


# ---------------------------------------------------------------------------
# missy api
# ---------------------------------------------------------------------------


@cli.group()
def api() -> None:
    """REST API server management (Agent-as-a-Service)."""


@api.command("start")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address.")
@click.option("--port", default=8080, type=int, show_default=True, help="TCP port to listen on.")
@click.option(
    "--api-key",
    envvar="MISSY_API_KEY",
    default="",
    help="API key for authentication (required). Can also be set via MISSY_API_KEY env var.",
)
@click.option(
    "--provider",
    default="anthropic",
    show_default=True,
    help="AI provider to use for chat requests.",
)
@click.pass_context
def api_start(
    ctx: click.Context,
    host: str,
    port: int,
    api_key: str,
    provider: str,
) -> None:
    """Start the REST API server.

    \b
    Exposes Missy over HTTP for programmatic use. All requests require
    an API key (Bearer token or X-API-Key header).

    \b
    Example:
        MISSY_API_KEY=secret missy api start --port 8080

    \b
    Endpoints:
        GET  /api/v1/health              Liveness probe
        GET  /api/v1/status              Agent status
        POST /api/v1/chat                Send message to agent
        POST /api/v1/sessions            Create session
        GET  /api/v1/sessions            List sessions
        GET  /api/v1/sessions/{id}       Get session
        GET  /api/v1/sessions/{id}/history  Conversation history
        DELETE /api/v1/sessions/{id}     End session
        GET  /api/v1/memory/search?q=   Full-text memory search
        GET  /api/v1/providers           List providers
        GET  /api/v1/tools               List tools
    """
    if not api_key:
        _print_error(
            "--api-key or MISSY_API_KEY environment variable is required.",
            hint='Generate a key with: python3 -c "import secrets; print(secrets.token_hex(32))"',
        )
        sys.exit(1)

    from missy.agent.runtime import AgentConfig, AgentRuntime
    from missy.api.server import ApiConfig, ApiServer

    cfg = _load_subsystems(ctx.obj["config_path"])

    try:
        from missy.providers.registry import get_registry

        registry = get_registry()
    except Exception:
        registry = None

    try:
        from missy.tools.registry import get_tool_registry

        tool_reg = get_tool_registry()
    except Exception:
        tool_reg = None

    try:
        from missy.tools.benchmark import get_benchmark_store
        from missy.tools.intelligence import get_candidate_store

        candidate_store = get_candidate_store()
        benchmark_store = get_benchmark_store()
    except Exception as _tool_intel_exc:
        logger.debug("Tool intelligence stores unavailable: %s", _tool_intel_exc)
        candidate_store = None
        benchmark_store = None

    try:
        from missy.memory.sqlite_store import SQLiteMemoryStore

        db_path = str(
            Path(getattr(cfg, "audit_log_path", "~/.missy/audit.jsonl")).expanduser().parent
            / "memory.db"
        )
        memory_store = SQLiteMemoryStore(db_path=db_path)
    except Exception as _mem_exc:
        logger.debug("Memory store unavailable: %s", _mem_exc)
        memory_store = None

    agent_config = AgentConfig(
        provider=provider,
        max_spend_usd=getattr(cfg, "max_spend_usd", 0.0),
        **_agent_tool_policy_kwargs(cfg),
    )
    runtime = AgentRuntime(agent_config)

    api_config = ApiConfig(host=host, port=port, api_key=api_key)
    server = ApiServer(
        config=api_config,
        runtime=runtime,
        memory_store=memory_store,
        provider_registry=registry,
        tool_registry=tool_reg,
        candidate_store=candidate_store,
        benchmark_store=benchmark_store,
    )

    try:
        server.start()
        console.print(
            f"[green]API server running on [bold]{server.url}[/][/]\n"
            "  Press [bold]Ctrl+C[/] to stop."
        )
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopping API server...[/]")
        server.stop()
        console.print("[green]API server stopped.[/]")


@api.command("status")
@click.option("--host", default="127.0.0.1", show_default=True, help="API server host.")
@click.option("--port", default=8080, type=int, show_default=True, help="API server port.")
@click.option(
    "--api-key",
    envvar="MISSY_API_KEY",
    default="",
    help="API key for authentication.",
)
def api_status(host: str, port: int, api_key: str) -> None:
    """Check if the API server is running by probing /api/v1/health."""
    import httpx

    url = f"http://{host}:{port}/api/v1/health"
    headers = {"X-API-Key": api_key} if api_key else {}

    try:
        resp = httpx.get(url, headers=headers, timeout=3.0)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            console.print(
                f"[green]API server is running[/] at [bold]http://{host}:{port}[/]\n"
                f"  Status : {data.get('status', '?')}\n"
                f"  Version: {data.get('version', '?')}"
            )
        elif resp.status_code == 401:
            console.print(
                f"[yellow]API server is running[/] at [bold]http://{host}:{port}[/] "
                "(authentication required — supply --api-key)"
            )
        else:
            console.print(
                f"[yellow]API server responded with HTTP {resp.status_code}[/] "
                f"at [bold]http://{host}:{port}[/]"
            )
    except httpx.ConnectError:
        console.print(
            f"[red]API server is not reachable[/] at [bold]http://{host}:{port}[/]\n"
            "  Start it with: [bold]missy api start[/]"
        )
    except Exception as exc:
        _print_error(f"Unexpected error checking API server: {exc}")


# ---------------------------------------------------------------------------
# tools — tool intelligence, candidates, and benchmarks
# ---------------------------------------------------------------------------


@cli.group("tools", invoke_without_command=True)
@click.pass_context
def tools_group(ctx: click.Context) -> None:
    """Tool intelligence: candidates, request patterns, and benchmarks."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@tools_group.command("trust")
@click.argument("name", required=False)
@click.option(
    "--threshold",
    default=200,
    show_default=True,
    help="Score at/below which an entity is flagged as low-trust.",
)
@click.pass_context
def tools_trust(ctx: click.Context, name: str | None, threshold: int) -> None:
    """Show persisted trust scores (0-1000) for tools/providers (F11).

    Reads the persisted score file the running gateway writes on every tool
    call. With no NAME, lists every scored entity (low-trust ones flagged);
    with a NAME, shows just that entity's score. New/unseen entities report
    the default score of 500.
    """
    from missy.security.trust import DEFAULT_SCORE, DEFAULT_TRUST_PATH, TrustScorer

    _load_subsystems(ctx.obj["config_path"])
    scorer = TrustScorer(persist_path=DEFAULT_TRUST_PATH)

    if name:
        s = scorer.score(name)
        trusted = scorer.is_trusted(name, threshold=threshold)
        seen = name in scorer.get_scores()
        note = "" if seen else " [dim](never scored — default)[/]"
        colour = "green" if trusted else "red"
        console.print(f"[bold]{name}[/]: [{colour}]{s}[/]/1000{note}")
        return

    scores = scorer.get_scores()
    if not scores:
        console.print(
            "[dim]No trust scores recorded yet. Scores are written by a running "
            "gateway as tools execute; new entities default to "
            f"{DEFAULT_SCORE}.[/]"
        )
        return

    t = Table(title="Trust scores (0-1000)")
    t.add_column("Entity")
    t.add_column("Score", justify="right")
    t.add_column("Status")
    for entity, s in sorted(scores.items(), key=lambda kv: kv[1]):
        low = s <= threshold
        status = "[red]LOW TRUST[/]" if low else "[green]ok[/]"
        t.add_row(entity, f"[{'red' if low else 'green'}]{s}[/]", status)
    console.print(t)


@tools_group.group("candidates", invoke_without_command=True)
@click.pass_context
def tools_candidates(ctx: click.Context) -> None:
    """Manage tool candidates (proposed → approved → enabled)."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@tools_candidates.command("list")
@click.option(
    "--state",
    default=None,
    help="Filter by lifecycle state: proposed, experimental, benchmarked, approved, enabled, deprecated, disabled.",
)
@click.option("--owner", default=None, help="Filter by owner string.")
@click.option("--limit", default=50, show_default=True, help="Maximum rows to show.")
@click.pass_context
def candidates_list(ctx: click.Context, state: str | None, owner: str | None, limit: int) -> None:
    """List tool candidates."""
    from missy.tools.intelligence import ToolLifecycleState, get_candidate_store

    store = get_candidate_store()
    ls = None
    if state:
        try:
            ls = ToolLifecycleState(state.lower())
        except ValueError:
            _print_error(
                f"Unknown state {state!r}. Valid states: {[s.value for s in ToolLifecycleState]}"
            )
            raise SystemExit(1) from None

    candidates = store.list_all(state=ls, owner=owner, limit=limit)
    if not candidates:
        console.print("[dim]No candidates found.[/]")
        return

    t = Table(title=f"Tool Candidates ({len(candidates)} shown)", show_lines=False)
    t.add_column("ID", style="dim", no_wrap=True, max_width=12)
    t.add_column("Name", style="bold")
    t.add_column("State")
    t.add_column("Owner")
    t.add_column("Score", justify="right")
    t.add_column("Updated")
    for c in candidates:
        best_score = ""
        if c.benchmark_scores:
            best = max(c.benchmark_scores, key=lambda s: s.composite)
            best_score = f"{best.composite:.2f}"
        state_color = {
            "proposed": "yellow",
            "experimental": "cyan",
            "benchmarked": "blue",
            "approved": "green",
            "enabled": "bright_green",
            "deprecated": "dim",
            "disabled": "red",
        }.get(c.state.value, "white")
        t.add_row(
            c.id[:12],
            c.name,
            f"[{state_color}]{c.state.value}[/]",
            c.owner,
            best_score,
            c.updated_at[:10] if c.updated_at else "",
        )
    console.print(t)


@tools_candidates.command("show")
@click.argument("candidate_id")
@click.pass_context
def candidates_show(ctx: click.Context, candidate_id: str) -> None:
    """Show details for a single tool candidate."""
    from missy.tools.intelligence import get_candidate_store

    store = get_candidate_store()
    c = store.get(candidate_id)
    if c is None:
        _print_error(f"Candidate {candidate_id!r} not found.")
        raise SystemExit(1)

    console.print(Panel(f"[bold]{c.name}[/] — {c.state.value}", title="Tool Candidate"))
    console.print(f"[bold]ID:[/] {c.id}")
    console.print(f"[bold]Description:[/] {c.description}")
    console.print(f"[bold]Owner:[/] {c.owner}    [bold]Version:[/] {c.version}")
    console.print(f"[bold]Provenance:[/] {c.provenance}")
    if c.notes:
        console.print(f"[bold]Notes:[/] {c.notes}")
    if c.permissions:
        console.print(f"[bold]Permissions:[/] {c.permissions}")
    if c.tags:
        console.print(f"[bold]Tags:[/] {', '.join(c.tags)}")
    if c.benchmark_scores:
        t = Table(title="Benchmark Scores")
        t.add_column("Provider")
        t.add_column("Composite", justify="right")
        t.add_column("Correctness", justify="right")
        t.add_column("Reliability", justify="right")
        t.add_column("Run At")
        for s in c.benchmark_scores:
            t.add_row(
                s.provider,
                f"{s.composite:.3f}",
                f"{s.correctness:.3f}",
                f"{s.reliability:.3f}",
                s.run_at[:10] if s.run_at else "",
            )
        console.print(t)
    if c.provider_enabled:
        console.print(f"[bold]Provider enabled:[/] {c.provider_enabled}")
    import json as _json

    console.print(f"[bold]Schema:[/]\n{_json.dumps(c.schema, indent=2)}")


@tools_candidates.command("import-benchmarks")
@click.argument("candidate_id")
@click.option(
    "--tool-name",
    default=None,
    help="Benchmark tool name to import. Defaults to the candidate name.",
)
@click.option("--min-samples", default=3, show_default=True, help="Minimum runs per provider.")
@click.option(
    "--min-composite",
    default=0.4,
    show_default=True,
    help="Minimum mean composite score for provider enablement.",
)
@click.option(
    "--min-safety",
    default=1.0,
    show_default=True,
    help="Minimum mean safety score for provider enablement.",
)
@click.option(
    "--min-schema-score",
    default=0.8,
    show_default=True,
    help="Minimum mean schema-adherence score for provider enablement.",
)
@click.pass_context
def candidates_import_benchmarks(
    ctx: click.Context,
    candidate_id: str,
    tool_name: str | None,
    min_samples: int,
    min_composite: float,
    min_safety: float,
    min_schema_score: float,
) -> None:
    """Import stored benchmark summaries into a candidate review record."""
    from missy.tools.benchmark import get_benchmark_store
    from missy.tools.intelligence import CandidateBenchmarkReconciler, get_candidate_store

    reconciler = CandidateBenchmarkReconciler(
        candidate_store=get_candidate_store(),
        benchmark_store=get_benchmark_store(),
        min_samples=min_samples,
        min_composite=min_composite,
        min_safety=min_safety,
        min_schema_score=min_schema_score,
    )
    try:
        result = reconciler.reconcile_candidate(candidate_id, tool_name=tool_name, actor="operator")
    except ValueError as exc:
        _print_error(str(exc))
        raise SystemExit(1) from None
    if result is None:
        _print_error(f"Candidate {candidate_id!r} not found.")
        raise SystemExit(1)

    c = result.candidate
    t = Table(title=f"Imported Benchmarks: {c.name}")
    t.add_column("Provider", style="bold")
    t.add_column("Runs", justify="right")
    t.add_column("Enabled")
    t.add_column("Composite", justify="right")
    t.add_column("Safety", justify="right")
    t.add_column("Schema", justify="right")
    t.add_column("Reason", max_width=54)
    for decision in result.decisions:
        s = decision.summary
        t.add_row(
            decision.provider,
            str(decision.run_count),
            "[green]yes[/]" if decision.enabled else "[red]no[/]",
            f"{s.composite:.3f}",
            f"{s.safety:.3f}",
            f"{s.schema_score:.3f}",
            decision.reason,
        )
    console.print(t)
    console.print(
        f"Candidate [bold]{c.name}[/] is now [bold]{c.state.value}[/]; "
        "approval and enablement remain separate lifecycle actions."
    )


@tools_candidates.command("approve")
@click.argument("candidate_id")
@click.option("--notes", default="", help="Approval notes.")
@click.pass_context
def candidates_approve(ctx: click.Context, candidate_id: str, notes: str) -> None:
    """Approve a benchmarked tool candidate."""
    from missy.tools.intelligence import ToolLifecycleState, get_candidate_store

    store = get_candidate_store()
    try:
        updated = store.transition(
            candidate_id, ToolLifecycleState.APPROVED, notes=notes, actor="operator"
        )
    except ValueError as exc:
        _print_error(str(exc))
        raise SystemExit(1) from None
    if updated is None:
        _print_error(f"Candidate {candidate_id!r} not found.")
        raise SystemExit(1)
    console.print(f"[green]Approved[/] candidate [bold]{updated.name}[/] ({candidate_id[:12]})")


@tools_candidates.command("enable")
@click.argument("candidate_id")
@click.option("--notes", default="", help="Enablement notes.")
@click.pass_context
def candidates_enable(ctx: click.Context, candidate_id: str, notes: str) -> None:
    """Enable an approved tool candidate for agent use."""
    from missy.tools.intelligence import ToolLifecycleState, get_candidate_store

    store = get_candidate_store()
    c = store.get(candidate_id)
    if c is None:
        _print_error(f"Candidate {candidate_id!r} not found.")
        raise SystemExit(1)
    if c.state != ToolLifecycleState.APPROVED:
        _print_error(f"Candidate must be in 'approved' state to enable (current: {c.state.value}).")
        raise SystemExit(1)
    try:
        updated = store.transition(
            candidate_id, ToolLifecycleState.ENABLED, notes=notes, actor="operator"
        )
    except ValueError as exc:
        _print_error(str(exc))
        raise SystemExit(1) from None
    console.print(f"[bright_green]Enabled[/] candidate [bold]{updated.name}[/]")


@tools_candidates.command("deny")
@click.argument("candidate_id")
@click.option("--reason", default="", help="Denial reason (recommended).")
@click.pass_context
def candidates_deny(ctx: click.Context, candidate_id: str, reason: str) -> None:
    """Deny a tool candidate (moves to disabled state)."""
    from missy.tools.intelligence import ToolLifecycleState, get_candidate_store

    store = get_candidate_store()
    try:
        updated = store.transition(
            candidate_id, ToolLifecycleState.DISABLED, notes=reason, actor="operator"
        )
    except ValueError as exc:
        _print_error(str(exc))
        raise SystemExit(1) from None
    if updated is None:
        _print_error(f"Candidate {candidate_id!r} not found.")
        raise SystemExit(1)
    console.print(f"[red]Disabled[/] candidate [bold]{updated.name}[/] ({candidate_id[:12]})")


@tools_group.group("requests", invoke_without_command=True)
@click.pass_context
def tools_requests(ctx: click.Context) -> None:
    """Inspect recorded request patterns."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@tools_requests.command("stats")
@click.option("--min-count", default=2, show_default=True, help="Minimum occurrence count.")
@click.option("--top", default=20, show_default=True, help="Number of patterns to show.")
@click.pass_context
def requests_stats(ctx: click.Context, min_count: int, top: int) -> None:
    """Show the most frequent request patterns."""
    from missy.tools.intelligence import get_request_tracker

    tracker = get_request_tracker()
    total_events = tracker.event_count()
    patterns = tracker.get_frequent_patterns(min_count=min_count, limit=top)
    console.print(
        f"[bold]Request tracker[/]: {total_events} events recorded, "
        f"{tracker.pattern_count()} distinct patterns."
    )
    if not patterns:
        console.print(f"[dim]No patterns with ≥{min_count} occurrences found.[/]")
        return
    t = Table(title=f"Top {len(patterns)} Patterns (min_count={min_count})")
    t.add_column("Key", style="dim", max_width=10)
    t.add_column("Count", justify="right")
    t.add_column("Score", justify="right")
    t.add_column("Common Tools")
    t.add_column("Representative", max_width=50)
    for p in patterns:
        t.add_row(
            p.pattern_key,
            str(p.count),
            f"{p.frequency_score:.3f}",
            ", ".join(p.common_tools) or "—",
            p.representative[:50],
        )
    console.print(t)


@tools_group.group("providers", invoke_without_command=True)
@click.pass_context
def tools_providers(ctx: click.Context) -> None:
    """Per-provider tool enablement based on benchmark results."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@tools_providers.command("status")
@click.argument("tool_name")
@click.option(
    "--provider",
    "provider_names",
    multiple=True,
    help="Provider to check (repeatable). Defaults to every provider with "
    "benchmark data or an override for this tool.",
)
@click.pass_context
def providers_status(ctx: click.Context, tool_name: str, provider_names: tuple[str, ...]) -> None:
    """Show per-provider enablement status for TOOL_NAME.

    Combines operator overrides (see 'enable'/'disable' below) with
    benchmark-derived scores to explain why a provider currently is or is
    not offered TOOL_NAME.
    """
    from missy.tools.benchmark import get_benchmark_store
    from missy.tools.intelligence import ToolProviderGate, get_provider_gate_store

    store = get_benchmark_store()
    overrides = get_provider_gate_store()
    gate = ToolProviderGate(overrides=overrides, benchmark_store=store)

    names = list(provider_names)
    if not names:
        seen = {s.provider for s in store.provider_summary(tool_name)}
        seen.update(overrides.list_overrides().get(tool_name, {}).keys())
        names = sorted(seen)
    if not names:
        console.print(
            f"[dim]No benchmark data or overrides for tool {tool_name!r}. "
            f"Pass --provider to check a specific provider directly.[/]"
        )
        return

    t = Table(title=f"Provider Gate: {tool_name}")
    t.add_column("Provider", style="bold")
    t.add_column("Enabled")
    t.add_column("Source")
    t.add_column("Reason", max_width=60)
    for name in names:
        decision = gate.decide(tool_name, name)
        t.add_row(
            name,
            "[green]yes[/]" if decision.enabled else "[red]no[/]",
            decision.source,
            decision.reason,
        )
    console.print(t)


@tools_providers.command("enable")
@click.argument("tool_name")
@click.argument("provider_name")
@click.pass_context
def providers_enable(ctx: click.Context, tool_name: str, provider_name: str) -> None:
    """Force-enable TOOL_NAME for PROVIDER_NAME, overriding benchmark data."""
    from missy.tools.intelligence import get_provider_gate_store

    get_provider_gate_store().set(tool_name, provider_name, True, actor="operator")
    console.print(f"[green]Enabled[/] [bold]{tool_name}[/] for provider [bold]{provider_name}[/].")


@tools_providers.command("disable")
@click.argument("tool_name")
@click.argument("provider_name")
@click.pass_context
def providers_disable(ctx: click.Context, tool_name: str, provider_name: str) -> None:
    """Force-disable TOOL_NAME for PROVIDER_NAME, overriding benchmark data."""
    from missy.tools.intelligence import get_provider_gate_store

    get_provider_gate_store().set(tool_name, provider_name, False, actor="operator")
    console.print(f"[red]Disabled[/] [bold]{tool_name}[/] for provider [bold]{provider_name}[/].")


@tools_providers.command("clear")
@click.argument("tool_name")
@click.argument("provider_name")
@click.pass_context
def providers_clear(ctx: click.Context, tool_name: str, provider_name: str) -> None:
    """Remove an explicit override, reverting to benchmark-driven gating."""
    from missy.tools.intelligence import get_provider_gate_store

    cleared = get_provider_gate_store().clear(tool_name, provider_name, actor="operator")
    if cleared:
        console.print(f"Cleared override for [bold]{tool_name}[/]/[bold]{provider_name}[/].")
    else:
        console.print("[dim]No override was set for that tool/provider pair.[/]")


@tools_providers.command("recommend")
@click.argument("tool_name")
@click.option(
    "--candidate",
    "candidates",
    multiple=True,
    help="Provider to rank (repeatable). Defaults to every provider with "
    "benchmark data for this tool.",
)
@click.pass_context
def providers_recommend(ctx: click.Context, tool_name: str, candidates: tuple[str, ...]) -> None:
    """Recommend the best-performing enabled provider for TOOL_NAME."""
    from missy.tools.benchmark import get_benchmark_store
    from missy.tools.intelligence import ToolProviderGate

    store = get_benchmark_store()
    gate = ToolProviderGate(benchmark_store=store)
    names = list(candidates) or sorted({s.provider for s in store.provider_summary(tool_name)})
    if not names:
        console.print(f"[dim]No benchmark data for tool {tool_name!r}.[/]")
        return
    best = gate.recommend_provider(tool_name, names)
    if best is None:
        console.print(f"[yellow]No candidate provider is currently enabled for {tool_name!r}.[/]")
        return
    console.print(f"[bold green]{best}[/] is the recommended provider for [bold]{tool_name}[/].")


@tools_group.group("benchmark", invoke_without_command=True)
@click.pass_context
def tools_benchmark(ctx: click.Context) -> None:
    """Run and inspect tool benchmarks."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@tools_benchmark.command("results")
@click.option("--tool", "tool_name", default=None, help="Filter by tool name.")
@click.option("--provider", default=None, help="Filter by provider.")
@click.option("--limit", default=30, show_default=True, help="Maximum rows.")
@click.pass_context
def benchmark_results(
    ctx: click.Context, tool_name: str | None, provider: str | None, limit: int
) -> None:
    """Show stored benchmark results."""
    from missy.tools.benchmark import get_benchmark_store

    store = get_benchmark_store()
    rows = store.query(tool_name=tool_name, provider=provider, limit=limit)
    if not rows:
        console.print("[dim]No benchmark results found.[/]")
        return
    t = Table(title=f"Benchmark Results ({len(rows)} shown)")
    t.add_column("Tool")
    t.add_column("Provider")
    t.add_column("Composite", justify="right")
    t.add_column("Correctness", justify="right")
    t.add_column("Latency ms", justify="right")
    t.add_column("Success")
    t.add_column("Recorded")
    for r in rows:
        t.add_row(
            r["tool_name"],
            r["provider"],
            f"{r['composite']:.3f}",
            f"{r['correctness']:.3f}",
            f"{r['latency_ms']:.0f}",
            "[green]✓[/]" if r["success"] else "[red]✗[/]",
            (r["recorded_at"] or "")[:16],
        )
    console.print(t)


@tools_benchmark.command("compare")
@click.argument("tool_name")
@click.pass_context
def benchmark_compare(ctx: click.Context, tool_name: str) -> None:
    """Compare provider scores for a tool."""
    from missy.tools.benchmark import get_benchmark_store

    store = get_benchmark_store()
    summaries = store.provider_summary(tool_name)
    if not summaries:
        console.print(f"[dim]No benchmark data for tool {tool_name!r}.[/]")
        return
    t = Table(title=f"Provider Comparison: {tool_name}")
    t.add_column("Provider", style="bold")
    t.add_column("Runs", justify="right")
    t.add_column("Composite", justify="right")
    t.add_column("Correctness", justify="right")
    t.add_column("Reliability", justify="right")
    t.add_column("Latency ms", justify="right")
    t.add_column("Cost USD", justify="right")
    for s in summaries:
        t.add_row(
            s.provider,
            str(s.run_count),
            f"{s.mean_composite:.3f}",
            f"{s.mean_correctness:.3f}",
            f"{s.mean_reliability:.3f}",
            f"{s.mean_latency_ms:.0f}",
            f"{s.mean_cost_usd:.6f}",
        )
    console.print(t)


@tools_benchmark.command("run")
@click.argument("tool_name")
@click.option(
    "--provider",
    "provider_label",
    default="direct",
    show_default=True,
    help="Provider label to attach to results (use 'direct' for registry-only execution).",
)
@click.option(
    "--no-persist",
    is_flag=True,
    default=False,
    help="Do not save results to the benchmark store.",
)
@click.pass_context
def benchmark_run(
    ctx: click.Context, tool_name: str, provider_label: str, no_persist: bool
) -> None:
    """Run a benchmark suite for TOOL_NAME against the tool registry.

    Builds a synthetic test suite from the tool's schema examples (or a
    minimal smoke-test if no examples are declared), executes each task
    directly through the registry, scores it, and prints a summary.

    Use --provider to label results for cross-provider comparison.
    Results are persisted by default; pass --no-persist to skip.
    """
    from missy.tools.benchmark.runner import BenchmarkRunner

    registry = _ensure_tool_registry()
    tool = registry.get(tool_name)
    if tool is None:
        _print_error(
            f"Tool {tool_name!r} is not registered. "
            "Run 'missy tools candidates list' to see available candidates, "
            "or check your tool registry."
        )
        raise SystemExit(1)

    # Build suite from tool schema examples, or a minimal smoke task
    suite = _build_suite_from_tool(tool, provider_label)
    if suite.task_count() == 0:
        _print_error(f"Could not build any benchmark tasks for tool {tool_name!r}.")
        raise SystemExit(1)

    console.print(
        f"Running [bold]{suite.task_count()}[/] task(s) for "
        f"[bold]{tool_name}[/] (provider=[cyan]{provider_label}[/]) …"
    )
    runner = BenchmarkRunner(provider=provider_label)
    report = runner.run_suite(suite, registry=registry, persist=not no_persist)

    # Print per-task results
    t = Table(title=f"Benchmark: {tool_name} ({provider_label})", show_lines=False)
    t.add_column("Task")
    t.add_column("Success")
    t.add_column("Composite", justify="right")
    t.add_column("Latency ms", justify="right")
    t.add_column("Correctness", justify="right")
    t.add_column("Error", max_width=40)
    for sr in report.scored_results:
        ok = "[green]✓[/]" if sr.result.success else "[red]✗[/]"
        t.add_row(
            sr.result.task_id[:12],
            ok,
            f"{sr.composite:.3f}",
            f"{sr.result.latency_ms:.0f}",
            f"{sr.correctness:.3f}",
            (sr.result.error or "")[:40],
        )
    console.print(t)

    agg = report.aggregate
    composite = agg.get("composite", 0.0)
    color = "green" if composite >= 0.7 else "yellow" if composite >= 0.4 else "red"
    console.print(
        f"\n[bold]Aggregate[/]: composite=[{color}]{composite:.3f}[/] "
        f"correctness={agg.get('correctness', 0):.3f} "
        f"reliability={agg.get('reliability', 0):.3f} "
        f"latency={agg.get('latency_ms', 0):.0f}ms "
        f"errors={report.error_count}/{suite.task_count()}"
    )
    if not no_persist:
        console.print("[dim]Results saved. Use 'missy tools benchmark results' to review.[/]")


@tools_benchmark.command("run-llm")
@click.argument("tool_name")
@click.option("--prompt", required=True, help="Natural-language prompt asking for the tool.")
@click.option(
    "--provider",
    "provider_name",
    default="mock",
    show_default=True,
    help="Configured provider name (anthropic, openai, ollama, ...) or 'mock' "
    "for the offline deterministic provider (no credentials required).",
)
@click.option(
    "--expect-arg",
    "expect_args",
    multiple=True,
    metavar="KEY=VALUE",
    help="Expected argument the provider should supply (repeatable).",
)
@click.option(
    "--execute",
    is_flag=True,
    default=False,
    help="DANGEROUS: actually run the tool call the provider makes (through "
    "the policy-checked registry) instead of only scoring tool selection "
    "and schema quality.",
)
@click.option(
    "--no-persist",
    is_flag=True,
    default=False,
    help="Do not save results to the benchmark store.",
)
@click.pass_context
def benchmark_run_llm(
    ctx: click.Context,
    tool_name: str,
    prompt: str,
    provider_name: str,
    expect_args: tuple[str, ...],
    execute: bool,
    no_persist: bool,
) -> None:
    """Benchmark a real provider's tool-calling behavior for TOOL_NAME.

    Unlike 'missy tools benchmark run' (which calls the tool directly through
    the registry), this drives an actual provider through its native
    tool-calling API with a natural-language PROMPT and scores whether it
    selected TOOL_NAME and filled in its schema correctly. Use
    --provider mock to exercise the full path offline, with no API
    credentials.

    By default the tool call the provider produces is NOT executed — only
    pass --execute if you specifically want to benchmark end-to-end
    correctness, and understand that policy-checked tool execution really
    will run (shell commands, file writes, etc. if the tool has those
    permissions).
    """
    from missy.tools.benchmark import LLMBenchmarkRunner, LLMBenchmarkTask, MockToolProvider

    registry = _ensure_tool_registry()
    tool = registry.get(tool_name)
    if tool is None:
        _print_error(
            f"Tool {tool_name!r} is not registered. "
            "Run 'missy tools candidates list' to see available candidates, "
            "or check your tool registry."
        )
        raise SystemExit(1)

    if provider_name == "mock":
        provider = MockToolProvider()
    else:
        from missy.providers.registry import get_registry

        # A fresh CLI process has not initialised the provider registry; do it
        # from config here (mirrors _load_subsystems) so run-llm against a real
        # provider works standalone instead of failing "registry not
        # initialised" the way 'benchmark run' used to before its own fix.
        try:
            preg = get_registry()
        except RuntimeError:
            _load_subsystems(ctx.obj["config_path"] if ctx.obj else "~/.missy/config.yaml")
            preg = get_registry()
        provider = preg.get(provider_name)
        if provider is None:
            _print_error(
                f"Provider {provider_name!r} is not configured. "
                "Use --provider mock for an offline run."
            )
            raise SystemExit(1)

    expected_args: dict[str, str] = {}
    for item in expect_args:
        if "=" not in item:
            _print_error(f"--expect-arg must be KEY=VALUE, got {item!r}")
            raise SystemExit(1)
        key, value = item.split("=", 1)
        expected_args[key] = value

    if execute:
        console.print(
            "[yellow bold]Warning:[/] --execute will actually run whatever "
            "tool call the provider produces."
        )

    task = LLMBenchmarkTask.create(tool_name=tool_name, prompt=prompt, expected_args=expected_args)
    runner = LLMBenchmarkRunner(provider=provider, execute_tool=execute)
    scored = runner.run_task(task, tool, registry=registry, persist=not no_persist)

    ok = "[green]✓[/]" if scored.result.success else "[red]✗[/]"
    console.print(
        Panel(
            f"Tool call made: {'yes' if scored.result.tool_call_made else 'no'}   {ok}\n"
            f"Args supplied: {scored.result.tool_call_args or '{}'}\n"
            f"Composite: {scored.composite:.3f}   "
            f"Schema: {scored.schema_score:.3f}   "
            f"Tool-call quality: {scored.tool_call_quality:.3f}\n"
            f"Latency: {scored.result.latency_ms:.0f}ms   "
            f"Error: {scored.result.error or '—'}",
            title=f"LLM Benchmark: {tool_name} ({provider.name})",
        )
    )
    if not no_persist:
        console.print("[dim]Results saved. Use 'missy tools benchmark results' to review.[/]")


def _build_suite_from_tool(tool: Any, provider_label: str) -> Any:
    """Build a :class:`BenchmarkSuite` from *tool* schema examples.

    Looks for ``examples`` in the tool's schema dict.  Each example must
    have an ``input`` key (dict of args) and an optional ``expected_output``
    key.  Falls back to a smoke task built from required parameters when no
    examples are declared.
    """
    from missy.tools.benchmark.runner import BenchmarkSuite, BenchmarkTask

    schema = tool.get_schema() if hasattr(tool, "get_schema") else {}
    suite = BenchmarkSuite(
        name=f"{tool.name}_auto_{provider_label}",
        tool_name=tool.name,
        description=f"Auto-generated suite for {tool.name}",
    )

    examples = schema.get("examples", [])
    for ex in examples:
        if not isinstance(ex, dict) or "input" not in ex:
            continue
        task = BenchmarkTask.create(
            tool_name=tool.name,
            input_args=dict(ex["input"]),
            expected_output=ex.get("expected_output"),
            tags=["auto", "example"],
        )
        suite.add_task(task)

    # If no examples, build a minimal smoke task using required parameters
    if suite.task_count() == 0:
        params = schema.get("parameters", {})
        required = params.get("required", [])
        properties = params.get("properties", {})
        smoke_args: dict = {}
        for req_param in required:
            prop = properties.get(req_param, {})
            ptype = prop.get("type", "string")
            if ptype == "string":
                smoke_args[req_param] = prop.get("example") or prop.get("default") or "test"
            elif ptype in ("integer", "number"):
                smoke_args[req_param] = prop.get("default") or 1
            elif ptype == "boolean":
                smoke_args[req_param] = prop.get("default") or False
            elif ptype == "array":
                smoke_args[req_param] = prop.get("default") or []
            else:
                smoke_args[req_param] = None
        if smoke_args or not required:
            task = BenchmarkTask.create(
                tool_name=tool.name,
                input_args=smoke_args,
                expected_output=None,
                tags=["auto", "smoke"],
            )
            suite.add_task(task)

    return suite


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
