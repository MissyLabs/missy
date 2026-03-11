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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
