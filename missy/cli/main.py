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

import json
import logging
import sys
from pathlib import Path
from typing import Optional

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

# ---------------------------------------------------------------------------
# Default config YAML written by ``missy init``
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_YAML = """\
# Missy configuration — edit to enable capabilities.
# All capabilities are disabled by default (secure-by-default posture).

network:
  default_deny: true
  allowed_cidrs: []
  allowed_domains: []
  allowed_hosts:
    - "api.anthropic.com"
    - "api.openai.com"

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

providers:
  anthropic:
    name: anthropic
    model: "claude-3-5-sonnet-20241022"
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
  log_level: "warning"

# Vault: encrypted secrets store
vault:
  enabled: false
  vault_dir: "~/.missy/secrets"
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_subsystems(config_path: str):
    """Load config, init policy engine, audit logger, and provider registry.

    Returns:
        The loaded :class:`~missy.config.settings.MissyConfig`.

    Raises:
        SystemExit: On any initialisation failure (message printed to stderr).
    """
    from missy.config.settings import load_config
    from missy.core.exceptions import ConfigurationError
    from missy.observability.audit_logger import init_audit_logger
    from missy.policy.engine import init_policy_engine
    from missy.providers.registry import init_registry

    expanded = str(Path(config_path).expanduser())
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

    init_policy_engine(cfg)
    init_audit_logger(cfg.audit_log_path)
    init_registry(cfg)

    # Initialize OpenTelemetry if configured.
    try:
        from missy.observability.otel import init_otel
        init_otel(cfg)
    except Exception:
        pass

    return cfg


def _print_error(message: str, hint: Optional[str] = None) -> None:
    """Render a styled error panel to stderr."""
    body = message
    if hint:
        body += f"\n\n[dim]{hint}[/]"
    err_console.print(
        Panel(body, title="[red]Error[/]", border_style="red", expand=False)
    )


def _print_success(message: str) -> None:
    """Render a styled success panel to stdout."""
    console.print(
        Panel(message, title="[green]Success[/]", border_style="green", expand=False)
    )


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
        missy_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _print_error(f"Cannot create directory {missy_dir}: {exc}")
        sys.exit(1)

    config_file = missy_dir / "config.yaml"
    if config_file.exists():
        console.print(
            f"[yellow]Config already exists at [bold]{config_file}[/] — skipping.[/]"
        )
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
        console.print(
            f"[yellow]Could not create workspace at {workspace} — create it manually.[/]"
        )

    _print_success(
        f"Missy initialised.\n\n"
        f"  Config  : [bold]{config_file}[/]\n"
        f"  Audit   : [bold]{audit_file}[/]\n"
        f"  Jobs    : [bold]{jobs_file}[/]\n"
        f"  Workspace: [bold]{workspace}[/]\n\n"
        f"Edit [bold]{config_file}[/] to configure providers and enable capabilities."
    )


# ---------------------------------------------------------------------------
# missy ask
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("prompt")
@click.option("--provider", default=None, help="Provider to use (overrides config default).")
@click.option("--session", default=None, help="Session ID for conversation continuity.")
@click.pass_context
def ask(
    ctx: click.Context,
    prompt: str,
    provider: Optional[str],
    session: Optional[str],
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

    agent_cfg = AgentConfig(provider=provider_name)
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
@click.pass_context
def run(ctx: click.Context, provider: Optional[str]) -> None:
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

    provider_name = provider or (
        next(iter(cfg.providers), "anthropic") if cfg.providers else "anthropic"
    )

    agent_cfg = AgentConfig(provider=provider_name)
    agent = AgentRuntime(agent_cfg)
    channel = CLIChannel()

    console.print(
        Panel(
            "[bold cyan]Missy[/] interactive session\n\n"
            f"Provider : [bold]{provider_name}[/]\n"
            "Type [bold]quit[/] or [bold]exit[/] to end, or press Ctrl-D.",
            border_style="cyan",
        )
    )

    session_id: Optional[str] = None

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
            console.print(
                f"[yellow]Warning:[/] Injection patterns detected: {injection_matches}"
            )

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
    pass


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
@click.pass_context
def schedule_add(
    ctx: click.Context,
    name: str,
    schedule_str: str,
    task: str,
    provider: str,
) -> None:
    """Add a new scheduled job.

    \b
    Example:
        missy schedule add \\
            --name "Daily digest" \\
            --schedule "daily at 09:00" \\
            --task "Summarise the news"
    """
    from missy.core.exceptions import SchedulerError
    from missy.scheduler.manager import SchedulerManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = SchedulerManager()

    try:
        mgr.start()
        job = mgr.add_job(name=name, schedule=schedule_str, task=task, provider=provider)
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
        f"  Provider: {job.provider}"
    )


@schedule.command("list")
@click.pass_context
def schedule_list(ctx: click.Context) -> None:
    """List all scheduled jobs."""
    from missy.scheduler.manager import SchedulerManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = SchedulerManager()
    jobs = mgr.list_jobs()

    if not jobs:
        console.print("[dim]No scheduled jobs.[/]")
        return

    table = Table(title="Scheduled Jobs", show_lines=True)
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Name", style="bold")
    table.add_column("Schedule")
    table.add_column("Provider")
    table.add_column("Enabled", justify="center")
    table.add_column("Runs", justify="right")
    table.add_column("Last Run")
    table.add_column("Next Run")

    for job in jobs:
        enabled_text = Text("yes", style="green") if job.enabled else Text("no", style="red")
        last_run = (
            job.last_run.strftime("%Y-%m-%d %H:%M") if job.last_run else "[dim]never[/]"
        )
        next_run = (
            job.next_run.strftime("%Y-%m-%d %H:%M") if job.next_run else "[dim]—[/]"
        )
        table.add_row(
            job.id[:8] + "…",
            job.name,
            job.schedule,
            job.provider,
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
    pass


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
def audit_recent(ctx: click.Context, limit: int, category: Optional[str]) -> None:
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


# ---------------------------------------------------------------------------
# missy providers
# ---------------------------------------------------------------------------


@cli.command("providers")
@click.pass_context
def providers_list(ctx: click.Context) -> None:
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

        table.add_row(
            key,
            provider_cfg.model,
            provider_cfg.base_url or "[dim]—[/]",
            f"{provider_cfg.timeout}s",
            avail_text,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# missy skills
# ---------------------------------------------------------------------------


@cli.command("skills")
@click.pass_context
def skills_list(ctx: click.Context) -> None:
    """List all registered skills."""
    from missy.skills.registry import SkillRegistry

    _load_subsystems(ctx.obj["config_path"])

    # Build a fresh registry — built-in skills would be registered by the
    # application bootstrap; here we report the current process state.
    registry = SkillRegistry()
    skill_names = registry.list_skills()

    if not skill_names:
        console.print("[dim]No skills are currently registered.[/]")
        console.print(
            "[dim]Skills are registered programmatically at application startup.[/]"
        )
        return

    table = Table(title="Registered Skills")
    table.add_column("Name", style="bold")

    for name in skill_names:
        table.add_row(name)

    console.print(table)


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
        console.print(
            f"Allowed plugins: [bold]{', '.join(cfg.plugins.allowed_plugins)}[/]"
        )
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
            Text("yes", style="green")
            if manifest.get("enabled")
            else Text("no", style="red")
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
    """Discord channel commands (status, probe, register-commands, audit)."""
    pass


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

    enabled_text = (
        "[green]enabled[/]" if discord_cfg.enabled else "[red]disabled[/]"
    )
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
    guild_id: Optional[str],
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

    discord_events = [
        e for e in all_events if str(e.get("event_type", "")).startswith("discord.")
    ][-limit:]

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


# ---------------------------------------------------------------------------
# missy gateway
# ---------------------------------------------------------------------------


@cli.group()
def gateway() -> None:
    """Gateway / service-mode commands."""
    pass


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

    def _stop(signum, frame):
        nonlocal stop_event
        stop_event = True
        console.print("\n[dim]Shutting down gateway...[/]")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Start Discord channel if configured.
    if cfg.discord and cfg.discord.enabled and cfg.discord.accounts:
        import asyncio
        from missy.channels.discord.channel import DiscordChannel

        async def _run_discord():
            channels = []
            for account in cfg.discord.accounts:
                ch = DiscordChannel(account_config=account)
                await ch.start()
                channels.append(ch)
                console.print(f"[green]Discord channel started[/] ({account.token_env_var})")
            try:
                while not stop_event:
                    await asyncio.sleep(1)
            finally:
                for ch in channels:
                    await ch.stop()

        asyncio.run(_run_discord())
    else:
        console.print("[dim]No Discord channels configured. Running in idle service mode.[/]")
        import time
        while not stop_event:
            time.sleep(1)

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
            status = Text("configured", style="green") if token_set else Text("token missing", style="red")
            table.add_row(
                f"discord[{i}]",
                status,
                f"env={account.token_env_var} dm_policy={account.dm_policy.value}",
            )
    else:
        table.add_row("discord", Text("disabled", style="dim"), "not configured")

    # CLI channel always available
    table.add_row("cli", Text("available", style="green"), "stdin/stdout")

    console.print(table)

    # Providers
    from missy.providers.registry import get_registry
    registry = get_registry()
    provider_names = registry.list_providers()
    console.print(f"\n[bold]Providers registered:[/] {', '.join(provider_names) if provider_names else '[dim]none[/]'}")


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
                avail = False
            status = ok if avail else fail
            table.add_row(f"provider:{name}", status, "api key present" if avail else "not available")

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
    jobs = mgr.list_jobs()
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

    console.print(table)


# ---------------------------------------------------------------------------
# missy vault
# ---------------------------------------------------------------------------


@cli.group()
def vault() -> None:
    """Encrypted secrets vault commands."""
    pass


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
    except VaultError as exc:
        _print_error(str(exc))
        sys.exit(1)


# ---------------------------------------------------------------------------
# missy sessions
# ---------------------------------------------------------------------------


@cli.group()
def sessions() -> None:
    """Session and conversation history commands."""
    pass


@sessions.command("cleanup")
@click.option("--older-than", default=30, show_default=True, help="Delete history older than N days.")
@click.option("--dry-run", is_flag=True, default=False, help="Show what would be deleted without deleting.")
@click.pass_context
def sessions_cleanup(ctx: click.Context, older_than: int, dry_run: bool) -> None:
    """Delete old conversation history from the memory store."""
    from missy.memory.store import MemoryStore
    _load_subsystems(ctx.obj["config_path"])
    store = MemoryStore()
    if dry_run:
        console.print(f"[dim]Dry run: would delete turns older than {older_than} days.[/]")
        return
    if hasattr(store, "cleanup"):
        removed = store.cleanup(older_than_days=older_than)
        _print_success(f"Removed {removed} conversation turn(s) older than {older_than} days.")
    else:
        console.print("[dim]Memory store does not support cleanup (use SQLiteMemoryStore).[/]")


# ---------------------------------------------------------------------------
# missy approvals
# ---------------------------------------------------------------------------


@cli.group()
def approvals() -> None:
    """Approval gate management."""
    pass


@approvals.command("list")
@click.pass_context
def approvals_list(ctx: click.Context) -> None:
    """List pending approval requests (for the current gateway session)."""
    console.print("[dim]No active gateway session; approvals are processed during missy gateway start.[/]")


# ---------------------------------------------------------------------------
# missy patches
# ---------------------------------------------------------------------------


@cli.group()
def patches() -> None:
    """Prompt patch (self-tuning) management."""
    pass


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
        _print_error(f"Patch {patch_id!r} not found.")


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
        _print_error(f"Patch {patch_id!r} not found.")


# ---------------------------------------------------------------------------
# missy mcp
# ---------------------------------------------------------------------------


@cli.group()
def mcp() -> None:
    """Model Context Protocol (MCP) server management."""
    pass


@mcp.command("list")
@click.pass_context
def mcp_list(ctx: click.Context) -> None:
    """List configured MCP servers."""
    from missy.mcp.manager import McpManager
    _load_subsystems(ctx.obj["config_path"])
    mgr = McpManager()
    servers = mgr.list_servers()
    if not servers:
        console.print("[dim]No MCP servers configured. Add one with missy mcp add.[/]")
        return
    table = Table(title="MCP Servers", show_lines=True)
    table.add_column("Name", style="bold")
    table.add_column("Alive", justify="center")
    table.add_column("Tools", justify="right")
    for s in servers:
        alive = Text("yes", style="green") if s["alive"] else Text("no", style="red")
        table.add_row(s["name"], alive, str(s["tools"]))
    console.print(table)


@mcp.command("add")
@click.argument("name")
@click.option("--command", default=None, help="Stdio command to launch the MCP server.")
@click.option("--url", default=None, help="HTTP URL for the MCP server.")
@click.pass_context
def mcp_add(ctx: click.Context, name: str, command: Optional[str], url: Optional[str]) -> None:
    """Connect to a new MCP server."""
    from missy.mcp.manager import McpManager
    _load_subsystems(ctx.obj["config_path"])
    mgr = McpManager()
    try:
        client = mgr.add_server(name, command=command, url=url)
        _print_success(f"Connected to MCP server [bold]{name}[/] ({len(client.tools)} tools).")
    except Exception as exc:
        _print_error(f"Failed to connect: {exc}")
        sys.exit(1)


@mcp.command("remove")
@click.argument("name")
@click.pass_context
def mcp_remove(ctx: click.Context, name: str) -> None:
    """Disconnect and remove an MCP server."""
    from missy.mcp.manager import McpManager
    _load_subsystems(ctx.obj["config_path"])
    mgr = McpManager()
    mgr.remove_server(name)
    _print_success(f"MCP server [bold]{name}[/] removed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
