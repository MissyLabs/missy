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

    init_policy_engine(cfg)
    init_audit_logger(cfg.audit_log_path)
    init_registry(cfg)

    # Register built-in tools so the agent can use them.
    try:
        from missy.tools.builtin import register_builtin_tools
        from missy.tools.registry import init_tool_registry

        tool_registry = init_tool_registry()
        register_builtin_tools(tool_registry)
    except Exception as _tool_exc:
        logger.debug("Tool registry init failed: %s", _tool_exc)

    # Initialize OpenTelemetry if configured.
    try:
        from missy.observability.otel import init_otel

        init_otel(cfg)
    except Exception as _otel_exc:
        logger.debug("OpenTelemetry init failed: %s", _otel_exc)

    return cfg


def _print_error(message: str, hint: str | None = None) -> None:
    """Render a styled error panel to stderr."""
    body = message
    if hint:
        body += f"\n\n[dim]{hint}[/]"
    err_console.print(Panel(body, title="[red]Error[/]", border_style="red", expand=False))


def _print_success(message: str) -> None:
    """Render a styled success panel to stdout."""
    console.print(Panel(message, title="[green]Success[/]", border_style="green", expand=False))


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
@click.option("--api-key-env", "setup_api_key_env", default=None, help="Environment variable containing the API key.")
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
        last_run = job.last_run.strftime("%Y-%m-%d %H:%M") if job.last_run else "[dim]never[/]"
        next_run = job.next_run.strftime("%Y-%m-%d %H:%M") if job.next_run else "[dim]—[/]"
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


# ---------------------------------------------------------------------------
# missy providers
# ---------------------------------------------------------------------------


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


@providers_group.command("switch")
@click.argument("name")
@click.pass_context
def providers_switch(ctx: click.Context, name: str) -> None:
    """Switch the active provider to NAME at runtime."""
    from missy.providers.registry import get_registry

    _load_subsystems(ctx.obj["config_path"])
    registry = get_registry()

    try:
        registry.set_default(name)
    except ValueError as exc:
        _print_error(str(exc))
        sys.exit(1)

    _print_success(f"Active provider switched to [bold]{name}[/].")


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
    """Discord channel commands (status, probe, register-commands, audit)."""


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

    # Start proactive manager if configured.
    proactive_manager = None
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
                _agent_cfg = AgentConfig(provider=_provider_name)
                _runtime = AgentRuntime(_agent_cfg)

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
            )
            proactive_manager.start()
            console.print(f"[green]Proactive manager started[/] ({len(triggers)} trigger(s))")
    except Exception as _pe:
        console.print(f"[yellow]Proactive manager failed to start: {_pe}[/]")

    # Build the shared agent runtime for all channels.
    from missy.agent.runtime import DISCORD_SYSTEM_PROMPT, AgentConfig, AgentRuntime

    _provider_name = next(iter(cfg.providers), "anthropic") if cfg.providers else "anthropic"
    _agent_cfg = AgentConfig(provider=_provider_name)
    _agent = AgentRuntime(_agent_cfg)

    # Discord-specific agent with filtered tools and appropriate system prompt.
    _discord_agent_cfg = AgentConfig(
        provider=_provider_name,
        system_prompt=DISCORD_SYSTEM_PROMPT,
        capability_mode="discord",
    )
    _discord_agent = AgentRuntime(_discord_agent_cfg)

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
            voice_channel.start(_agent)
            _vc_host = _voice_cfg.get("host", "0.0.0.0")
            _vc_port = _voice_cfg.get("port", 8765)
            console.print(f"[green]Voice channel started[/] on ws://{_vc_host}:{_vc_port}")
    except Exception as _ve:
        console.print(f"[yellow]Voice channel failed to start: {_ve}[/]")
        logger.warning("Voice channel startup error: %s", _ve, exc_info=True)

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
                        if sent_id and "Evolution proposed:" in response:
                            import re

                            _evo_match = re.search(r"Evolution proposed:\s*(\S+)", response)
                            if _evo_match:
                                _proposal_id = _evo_match.group(1)
                                ch.add_evolution_reactions(channel_id, sent_id, _proposal_id)
                        elif sent_id and "Multi-file evolution proposed:" in response:
                            import re

                            _evo_match = re.search(
                                r"Multi-file evolution proposed:\s*(\S+)", response
                            )
                            if _evo_match:
                                _proposal_id = _evo_match.group(1)
                                ch.add_evolution_reactions(channel_id, sent_id, _proposal_id)
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
        if proactive_manager is not None:
            try:
                proactive_manager.stop()
            except Exception as _stop_exc:
                logger.debug("proactive: stop error: %s", _stop_exc)

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
@click.pass_context
def recover(ctx: click.Context, abandon_all: bool) -> None:
    """List or act on incomplete task checkpoints from previous sessions.

    Scans for tasks that were interrupted by crashes or restarts and shows
    recovery options.  Use --abandon-all to clear all stale checkpoints.
    """

    try:
        from missy.agent.checkpoint import CheckpointManager, scan_for_recovery
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
        "Use [bold]missy recover --abandon-all[/bold] to clear stale tasks.[/]"
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


# ---------------------------------------------------------------------------
# missy approvals
# ---------------------------------------------------------------------------


@cli.group()
def approvals() -> None:
    """Approval gate management."""


@approvals.command("list")
@click.pass_context
def approvals_list(ctx: click.Context) -> None:
    """List pending approval requests (for the current gateway session)."""
    console.print(
        "[dim]No active gateway session; approvals are processed during missy gateway start.[/]"
    )


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
def mcp_add(ctx: click.Context, name: str, command: str | None, url: str | None) -> None:
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
def mcp_remove_cmd(ctx: click.Context, name: str) -> None:
    """Disconnect and remove an MCP server."""
    from missy.mcp.manager import McpManager

    _load_subsystems(ctx.obj["config_path"])
    mgr = McpManager()
    mgr.remove_server(name)
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
    nodes = reg.all()
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
        paired = node.get("paired", False)
        status_text = Text("paired", style="green") if paired else Text("pending", style="yellow")
        paired_text = Text("yes", style="green") if paired else Text("no", style="yellow")
        last_seen_ts = node.get("last_seen")
        if last_seen_ts:
            last_seen = datetime.fromtimestamp(last_seen_ts).strftime("%Y-%m-%d %H:%M")
        else:
            last_seen = "never"
        table.add_row(
            node.get("node_id", "")[:8],
            node.get("name", ""),
            node.get("room", ""),
            status_text,
            node.get("policy", "full"),
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
        pending = [n for n in reg.all() if not n.get("paired", False)]
        if not pending:
            console.print("[dim]No pending pairing requests.[/]")
            return
        console.print("[bold]Pending nodes:[/]")
        for i, node in enumerate(pending):
            console.print(
                f"  [{i}] {node.get('node_id', '')[:8]}  {node.get('name', '')}  {node.get('room', '')}"
            )
        idx = click.prompt("Select index to approve", type=int)
        if idx < 0 or idx >= len(pending):
            _print_error("Invalid selection.")
            sys.exit(1)
        node_id = pending[idx]["node_id"]

    try:
        token = mgr.approve(node_id)
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
    try:
        reg.remove(node_id)
        reg.save()
        _print_success(f"Node [bold]{node_id[:8]}[/] removed.")
    except KeyError:
        _print_error(f"Node {node_id!r} not found.")
        sys.exit(1)


@devices.command("status")
@click.pass_context
def devices_status(ctx: click.Context) -> None:
    """Show online/offline status of all edge nodes."""
    from datetime import datetime

    from missy.channels.voice.registry import DeviceRegistry

    reg = DeviceRegistry()
    reg.load()
    nodes = reg.all()
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
        online = node.get("online", False)
        status_text = Text("online", style="green") if online else Text("offline", style="red")
        last_seen_ts = node.get("last_seen")
        if last_seen_ts:
            last_seen = datetime.fromtimestamp(last_seen_ts).strftime("%Y-%m-%d %H:%M")
        else:
            last_seen = "never"
        occupancy = node.get("occupancy")
        occupancy_str = str(occupancy) if occupancy is not None else "-"
        noise = node.get("noise_level")
        noise_str = f"{noise:.1f} dB" if noise is not None else "-"
        table.add_row(
            node.get("node_id", "")[:8],
            node.get("name", ""),
            node.get("room", ""),
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
        reg.set_policy(node_id, mode)
        reg.save()
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
    paired_count = sum(1 for n in reg.all() if n.get("paired", False))

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
    from missy.channels.voice.tts.piper import PiperTTS

    # Validate node exists
    reg = DeviceRegistry()
    reg.load()
    node = next((n for n in reg.all() if n.get("node_id", "").startswith(node_id)), None)
    if node is None:
        _print_error(f"Node {node_id!r} not found in registry.")
        sys.exit(1)

    console.print(f"Synthesizing test phrase for node [bold]{node_id[:8]}[/]...")
    try:
        tts = PiperTTS()

        async def _synth() -> bytes:
            return await tts.synthesize(text)

        start = time.monotonic()
        audio_bytes = asyncio.run(_synth())
        elapsed = time.monotonic() - start

        # Estimate duration: PCM 16-bit mono 22050 Hz is piper's default output rate
        sample_rate = 22050
        duration_s = len(audio_bytes) / (sample_rate * 2)
        _print_success(
            f"TTS synthesis succeeded in {elapsed:.2f}s — "
            f"{len(audio_bytes):,} bytes, ~{duration_s:.1f}s of audio."
        )
        console.print(
            f"[dim]Would send to node {node.get('name', node_id[:8])} "
            f"in room {node.get('room', 'unknown')} via gateway WebSocket.[/]"
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

    backups = list_backups()
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

    backups = list_backups()
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

    backups = list_backups()
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
        console.print("[green]Missy is already hatched![/] Use [bold]missy hatch --non-interactive[/] to re-verify.")
        state = mgr.get_state()
        console.print(f"  Hatched at: [bold]{state.completed_at}[/]")
        console.print(f"  Steps: {', '.join(state.steps_completed)}")
        return

    console.print(Panel(
        "[bold cyan]Hatching Missy[/]\n\n"
        "This will set up your environment, initialise configuration,\n"
        "verify providers, create a persona, and seed memory.",
        title="[cyan]Hatching[/]",
        border_style="cyan",
    ))

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
    status_color = {"healthy": "green", "degraded": "yellow", "unhealthy": "red"}.get(
        status, "dim"
    )

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
                out_path = output or str(
                    Path.home() / f".missy/captures/best_{ts}.jpg"
                )
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                import cv2
                cv2.imwrite(out_path, result.image, [cv2.IMWRITE_JPEG_QUALITY, config.quality])
                console.print(
                    f"[green]Best frame[/] {result.width}x{result.height} → {out_path}"
                )
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
                    out_path = str(
                        Path.home() / f".missy/captures/burst_{ts}_{i:03d}.jpg"
                    )
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
                console.print(
                    f"[green]Captured[/] {result.width}x{result.height} → {out_path}"
                )
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
            f"[green]Acquired[/] {frame.width}x{frame.height} image "
            f"from {frame.source_type.value}"
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
        console.print(
            "\n[dim]For LLM-powered analysis, use missy vision review --mode general[/]"
        )

    except Exception as exc:
        _print_error(f"Inspection failed: {exc}")
        sys.exit(1)


@vision.command("review")
@click.option("--device", "-d", default=None, help="Camera device path.")
@click.option("--file", "-f", "file_path", default=None, help="Image file to review.")
@click.option(
    "--mode", "-m",
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

        console.print(Panel(
            response.text,
            title=f"[bold]{mode.title()} Analysis[/]",
            border_style="cyan",
            padding=(1, 2),
        ))

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
