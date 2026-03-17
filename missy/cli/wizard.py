"""Interactive onboarding wizard for Missy.

Invoked by ``missy setup``. Guides the user through:

1. Workspace directory
2. AI provider selection (Anthropic / OpenAI / Ollama)
3. API key entry with masking and env-var detection
4. Model tier selection (fast / primary / premium)
5. Connectivity verification
6. Atomic config.yaml write

Designed to work *before* a config file exists — it does not call
``_load_subsystems`` and has no dependency on an existing config.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
err_console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "anthropic": {
        "label": "Anthropic (Claude)",
        "host": "api.anthropic.com",
        "env_var": "ANTHROPIC_API_KEY",
        "key_prefix": "sk-ant-",
        "models": {
            "primary": "claude-sonnet-4-6",
            "fast": "claude-haiku-4-5-20251001",
            "premium": "claude-opus-4-6",
        },
        "model_choices": [
            ("claude-sonnet-4-6", "Sonnet 4.6 — balanced (recommended)"),
            ("claude-haiku-4-5-20251001", "Haiku 4.5 — fastest / cheapest"),
            ("claude-opus-4-6", "Opus 4.6 — most capable"),
        ],
    },
    "openai": {
        "label": "OpenAI (GPT / Codex)",
        "host": "api.openai.com",
        "env_var": "OPENAI_API_KEY",
        "key_prefix": "sk-",
        "models": {
            "primary": "gpt-4o",
            "fast": "gpt-4o-mini",
            "premium": "gpt-4-turbo",
        },
        "model_choices": [
            ("gpt-4o", "GPT-4o — balanced (recommended)"),
            ("gpt-4o-mini", "GPT-4o Mini — fastest / cheapest"),
            ("gpt-4-turbo", "GPT-4 Turbo — most capable"),
        ],
    },
    # openai-codex is set dynamically when the user chooses OAuth in the openai flow.
    # These models only work with chatgpt.com/backend-api (not api.openai.com).
    "openai-codex": {
        "label": "OpenAI Codex (ChatGPT OAuth)",
        "host": "chatgpt.com",
        "env_var": None,
        "key_prefix": None,
        "models": {
            "primary": "gpt-5.2",
            "fast": "gpt-5.1-codex-mini",
            "premium": "gpt-5.3-codex",
        },
        "model_choices": [
            ("gpt-5.2", "GPT-5.2 — balanced (recommended)"),
            ("gpt-5.1-codex-mini", "GPT-5.1 Codex Mini — fastest"),
            ("gpt-5.1-codex-max", "GPT-5.1 Codex Max — high capacity"),
            ("gpt-5.3-codex", "GPT-5.3 Codex — most capable"),
        ],
    },
    "ollama": {
        "label": "Ollama (local models)",
        "host": None,  # localhost — no network policy entry needed
        "env_var": None,
        "key_prefix": None,
        "models": {
            "primary": "",  # user-supplied
            "fast": "",
            "premium": "",
        },
        "model_choices": [],
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_key(key: str) -> str:
    """Return a masked preview of an API key, e.g. ``sk-ant-...wxyz``."""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:6] + "…" + key[-4:]


def _detect_env_key(env_var: str) -> str | None:
    """Return the value of *env_var* if set and non-empty, else None."""
    return os.environ.get(env_var) or None


def _validate_key_format(key: str, prefix: str | None) -> bool:
    """Basic key format check — non-empty and starts with expected prefix."""
    if not key or not key.strip():
        return False
    return not (prefix and not key.strip().startswith(prefix))


def _prompt_api_key(provider_name: str, info: dict) -> str | None:
    """Interactively prompt for an API key with env-var detection and masking."""
    env_var = info["env_var"]
    prefix = info["key_prefix"]

    existing = _detect_env_key(env_var) if env_var else None
    if existing:
        console.print(
            f"  [green]Detected[/] [bold]{env_var}[/] in environment: [dim]{_mask_key(existing)}[/]"
        )
        if click.confirm("  Use this key?", default=True):
            return None  # None → use env var; don't embed in config

    while True:
        key = click.prompt(
            f"  Enter {info['label']} API key",
            hide_input=True,
            default="",
            show_default=False,
        )
        if not key:
            if click.confirm("  Skip this provider?", default=False):
                return None
            continue
        # Strip common shell assignment patterns (export KEY=value)
        if "=" in key:
            key = key.split("=", 1)[1].strip()
        key = key.strip()
        if not _validate_key_format(key, prefix):
            hint = f" (should start with '{prefix}')" if prefix else ""
            console.print(
                f"  [yellow]Key format looks unusual{hint} — double-check and re-enter.[/]"
            )
            if not click.confirm("  Use this key anyway?", default=False):
                continue
        console.print(f"  [dim]Key accepted: {_mask_key(key)}[/]")
        return key


def _prompt_model(info: dict) -> tuple[str, str, str]:
    """Prompt the user to choose primary / fast / premium models.

    Returns (primary, fast, premium) model names.
    """
    choices = info["model_choices"]
    defaults = info["models"]

    if not choices:
        # Ollama — free-form input
        primary = click.prompt("  Model name (e.g. llama3)", default="llama3")
        return primary, "", ""

    console.print("  [dim]Available models:[/]")
    for i, (model_id, desc) in enumerate(choices, 1):
        console.print(f"    [bold]{i}[/]. {desc}  [dim]({model_id})[/]")

    def _pick(label: str, default_id: str) -> str:
        default_idx = next((i for i, (m, _) in enumerate(choices, 1) if m == default_id), 1)
        raw = click.prompt(
            f"  {label} model [1-{len(choices)}]",
            default=str(default_idx),
        )
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx][0]
        except (ValueError, IndexError):
            pass
        console.print(f"  [yellow]Invalid choice, using default ({default_id}).[/]")
        return default_id

    primary = _pick("Primary", defaults["primary"])
    fast = _pick("Fast (simple tasks)", defaults["fast"])
    premium = _pick("Premium (complex tasks)", defaults["premium"])
    return primary, fast, premium


def _verify_anthropic(api_key: str) -> bool:
    """Send a minimal Anthropic API call to verify the key works."""
    if api_key.startswith("sk-ant-oat"):
        console.print(
            "  [red]Setup-tokens (sk-ant-oat...) are not supported by the Anthropic API.[/]\n"
            "  Get a regular API key from: [cyan]https://console.anthropic.com/settings/keys[/]"
        )
        return False
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True
    except Exception as exc:
        console.print(f"  [red]Anthropic verification failed:[/] {exc}")
        return False


def _verify_openai(api_key: str) -> bool:
    """Send a minimal OpenAI API call to verify the key works."""
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True
    except Exception as exc:
        console.print(f"  [red]OpenAI verification failed:[/] {exc}")
        return False


def _verify_ollama(base_url: str, model: str) -> bool:
    """Check that the Ollama endpoint is reachable."""
    try:
        import httpx

        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        resp.raise_for_status()
        return True
    except Exception as exc:
        console.print(f"  [red]Ollama verification failed:[/] {exc}")
        return False


# ---------------------------------------------------------------------------
# Config writer
# ---------------------------------------------------------------------------


def _build_config_yaml(
    workspace: str,
    providers_cfg: list[dict],
    allowed_hosts: list[str],
    discord_cfg: dict | None = None,
) -> str:
    """Render the config.yaml content from gathered wizard data."""
    # Detect which hosts can be replaced by presets
    from missy.config.migrate import detect_presets

    network_data = {"allowed_hosts": allowed_hosts, "allowed_domains": [], "allowed_cidrs": []}
    detected_presets, remaining_hosts, _, _ = detect_presets(network_data)

    lines: list[str] = [
        "# Missy configuration — generated by 'missy setup'",
        "",
        "config_version: 2",
        "",
        "network:",
        "  default_deny: true",
    ]
    if detected_presets:
        lines.append("  presets:")
        for p in detected_presets:
            lines.append(f"    - {p}")
    lines += [
        "  allowed_cidrs: []",
        "  allowed_domains: []",
        "  allowed_hosts:",
    ]
    for host in remaining_hosts:
        lines.append(f'    - "{host}"')
    if not remaining_hosts:
        lines.append("    []")

    lines += [
        "",
        "filesystem:",
        "  allowed_write_paths:",
        f'    - "{workspace}"',
        '    - "~/.missy"',
        "  allowed_read_paths:",
        f'    - "{workspace}"',
        '    - "~/.missy"',
        '    - "/tmp"',
        "",
        "shell:",
        "  enabled: false",
        "  allowed_commands: []",
        "",
        "plugins:",
        "  enabled: false",
        "  allowed_plugins: []",
        "",
        "providers:",
    ]

    for p in providers_cfg:
        name = p["name"]
        lines.append(f"  {name}:")
        lines.append(f"    name: {name}")
        lines.append(f'    model: "{p["model"]}"')
        if p.get("fast_model"):
            lines.append(f'    fast_model: "{p["fast_model"]}"')
        if p.get("premium_model"):
            lines.append(f'    premium_model: "{p["premium_model"]}"')
        if p.get("api_key"):
            lines.append(f'    api_key: "{p["api_key"]}"')
        if p.get("base_url"):
            lines.append(f'    base_url: "{p["base_url"]}"')
        lines.append("    timeout: 30")

    # Discord section (optional)
    if discord_cfg:
        lines += ["", "discord:", "  enabled: true", "  accounts:"]
        lines.append(f'    - token_env_var: "{discord_cfg["token_env_var"]}"')
        if discord_cfg.get("bot_token"):
            lines.append(f'      token: "{discord_cfg["bot_token"]}"')
        if discord_cfg.get("application_id"):
            lines.append(f'      application_id: "{discord_cfg["application_id"]}"')
        lines.append(f"      dm_policy: {discord_cfg['dm_policy']}")
        if discord_cfg.get("dm_allowlist"):
            lines.append("      dm_allowlist:")
            for uid in discord_cfg["dm_allowlist"]:
                lines.append(f'        - "{uid}"')
        else:
            lines.append("      dm_allowlist: []")
        if discord_cfg.get("ack_reaction"):
            lines.append(f'      ack_reaction: "{discord_cfg["ack_reaction"]}"')
        lines.append(f"      ignore_bots: {str(discord_cfg.get('ignore_bots', True)).lower()}")
        guild_policies = discord_cfg.get("guild_policies", [])
        if guild_policies:
            lines.append("      guild_policies:")
            for gp in guild_policies:
                lines.append(f'        "{gp["guild_id"]}":')
                lines.append("          enabled: true")
                lines.append(f"          require_mention: {str(gp['require_mention']).lower()}")
                lines.append(f"          mode: {gp['mode']}")
                if gp.get("allowed_channels"):
                    lines.append("          allowed_channels:")
                    for ch in gp["allowed_channels"]:
                        lines.append(f'            - "{ch}"')
                else:
                    lines.append("          allowed_channels: []")
        else:
            lines.append("      guild_policies: {}")

    lines += [
        "",
        f'workspace_path: "{workspace}"',
        'audit_log_path: "~/.missy/audit.jsonl"',
        "",
        "heartbeat:",
        "  enabled: false",
        "  interval_seconds: 1800",
        '  active_hours: ""',
        "",
        "observability:",
        "  otel_enabled: false",
        '  otel_endpoint: "http://localhost:4317"',
        '  log_level: "warning"',
        "",
        "vault:",
        "  enabled: false",
        '  vault_dir: "~/.missy/secrets"',
        "",
    ]
    return "\n".join(lines)


def _write_config_atomic(config_path: Path, content: str) -> None:
    """Write *content* to *config_path* atomically via a sibling temp file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=config_path.parent, prefix=".config_tmp_")
    try:
        os.fchmod(fd, 0o600)  # Config may contain API keys; restrict access
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, config_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Main wizard entry point
# ---------------------------------------------------------------------------


def run_wizard(config_path: str) -> None:
    """Run the interactive onboarding wizard and write ``config_path``."""
    config_file = Path(config_path).expanduser()

    # -----------------------------------------------------------------------
    # Welcome
    # -----------------------------------------------------------------------
    console.print(
        Panel(
            "[bold cyan]Welcome to Missy[/]\n\n"
            "This wizard will configure your AI provider(s) and write\n"
            f"[bold]{config_file}[/].\n\n"
            "[dim]Press Ctrl-C at any time to abort without writing anything.[/]",
            title="[bold]Missy Setup Wizard[/]",
            border_style="cyan",
        )
    )

    if config_file.exists():
        console.print(f"\n[yellow]A config file already exists at {config_file}.[/]")
        if not click.confirm("Overwrite it?", default=False):
            console.print("[dim]Aborted. Existing config unchanged.[/]")
            return

    # -----------------------------------------------------------------------
    # Step 1: Workspace
    # -----------------------------------------------------------------------
    console.print("\n[bold]Step 1 of 5 — Workspace[/]")
    workspace_default = str(Path.home() / "workspace")
    workspace = click.prompt("  Workspace directory", default=workspace_default)
    workspace_path = Path(workspace).expanduser()
    workspace_path.mkdir(parents=True, exist_ok=True)
    console.print(f"  [green]Workspace:[/] {workspace_path}")

    # -----------------------------------------------------------------------
    # Step 2: Provider selection
    # -----------------------------------------------------------------------
    console.print("\n[bold]Step 2 of 5 — AI Provider(s)[/]")
    console.print("  Choose which providers to configure:\n")
    console.print("    [bold]1[/]. Anthropic (Claude)")
    console.print("    [bold]2[/]. OpenAI (GPT-4o / Codex)")
    console.print("    [bold]3[/]. Ollama (local models)")
    console.print("    [bold]4[/]. Anthropic + OpenAI (both)")
    console.print("    [bold]5[/]. All three")
    console.print("    [bold]0[/]. Skip (configure manually later)")

    choice = click.prompt("  Selection", default="1")
    provider_keys: list[str] = []
    if choice == "1":
        provider_keys = ["anthropic"]
    elif choice == "2":
        provider_keys = ["openai"]
    elif choice == "3":
        provider_keys = ["ollama"]
    elif choice == "4":
        provider_keys = ["anthropic", "openai"]
    elif choice == "5":
        provider_keys = ["anthropic", "openai", "ollama"]
    else:
        console.print("  [dim]Skipping provider setup.[/]")
        provider_keys = []

    # -----------------------------------------------------------------------
    # Step 3: Configure each provider
    # -----------------------------------------------------------------------
    providers_cfg: list[dict] = []
    allowed_hosts: list[str] = []
    verify_results: list[tuple[str, bool]] = []

    for pkey in provider_keys:
        info = _PROVIDERS[pkey]
        console.print(f"\n[bold]  Configuring {info['label']}[/]")

        if pkey == "ollama":
            base_url = click.prompt("    Ollama base URL", default="http://localhost:11434")
            model_name = click.prompt("    Default model", default="llama3")
            if click.confirm("    Verify Ollama connectivity?", default=True):
                ok = _verify_ollama(base_url, model_name)
                verify_results.append(("ollama", ok))
            providers_cfg.append(
                {
                    "name": "ollama",
                    "model": model_name,
                    "base_url": base_url,
                    "fast_model": "",
                    "premium_model": "",
                    "api_key": None,
                }
            )
            # Ollama is local — no network allowlist entry needed
            continue

        # API-key based providers (Anthropic, OpenAI)
        api_key: str | None = None

        if pkey == "openai":
            # OpenAI supports two auth methods: API key or OAuth (PKCE).
            console.print("    Auth method:")
            console.print("      [bold]1[/]. API key  (sk-…)")
            console.print("      [bold]2[/]. OAuth / Codex CLI  (browser flow, PKCE)")
            auth_choice = click.prompt("    Selection", default="1")

            if auth_choice == "2":
                from missy.cli.oauth import run_openai_oauth

                console.print(
                    "\n    [dim]Starting OAuth flow. A browser window will open.[/]\n"
                    "    [dim]For headless/remote: run  ssh -L 1455:localhost:1455 user@host[/]"
                )
                oauth_token = run_openai_oauth()
                if oauth_token:
                    # OAuth tokens go to chatgpt.com/backend-api, not api.openai.com.
                    # Switch provider name to openai-codex so the right provider is used.
                    pkey = "openai-codex"
                    api_key = oauth_token
                    verify_results.append(("openai-oauth", True))
                    console.print(
                        "    [green]OAuth token acquired.[/] "
                        "Provider set to [bold]openai-codex[/] (chatgpt.com backend)."
                    )
                else:
                    console.print(
                        "    [yellow]OAuth flow failed or was skipped — falling back to API key.[/]"
                    )
                    api_key = _prompt_api_key(pkey, info)
            else:
                api_key = _prompt_api_key(pkey, info)
                if api_key and click.confirm("    Verify API key with a test call?", default=True):
                    console.print("    [dim]Connecting…[/]")
                    ok = _verify_openai(api_key)
                    verify_results.append(("openai", ok))
                    if ok:
                        console.print("    [green]Connection successful.[/]")
                    else:
                        console.print(
                            "    [yellow]Verification failed — key will still be saved.[/]"
                        )
        else:
            # Anthropic: three auth paths.
            console.print("    Auth method:")
            console.print("      [bold]1[/]. API key  (sk-ant-api…)  [green][recommended][/]")
            console.print(
                "      [bold]2[/]. API key + vault  (encrypted, config references vault://)"
            )
            console.print(
                "      [bold]3[/]. Claude Code setup-token  (sk-ant-oat…)  [yellow][ToS risk][/]"
            )
            auth_choice = click.prompt("    Selection", default="1")

            if auth_choice == "3":
                # Setup-token paste flow — ToS warning included inside.
                from missy.cli.anthropic_auth import run_anthropic_setup_token_flow

                setup_tok = run_anthropic_setup_token_flow()
                if setup_tok:
                    api_key = setup_tok
                    verify_results.append(("anthropic-setup-token", True))
                else:
                    console.print("    [yellow]Setup-token skipped — falling back to API key.[/]")
                    api_key = _prompt_api_key(pkey, info)
            else:
                api_key = _prompt_api_key(pkey, info)
                if api_key and auth_choice == "2":
                    from missy.cli.anthropic_auth import run_anthropic_vault_flow

                    vault_dir = str(Path("~/.missy/secrets").expanduser())
                    api_key = run_anthropic_vault_flow(api_key, vault_dir)

            # Live verification (skip for vault refs and setup-tokens).
            if (
                api_key
                and not api_key.startswith("vault://")
                and auth_choice != "3"
                and click.confirm("    Verify API key with a test call?", default=True)
            ):
                console.print("    [dim]Connecting…[/]")
                ok = _verify_anthropic(api_key)
                verify_results.append(("anthropic", ok))
                if ok:
                    console.print("    [green]Connection successful.[/]")
                else:
                    console.print("    [yellow]Verification failed — key will still be saved.[/]")

        # Use updated pkey's info for model choices (e.g. openai-codex after OAuth).
        primary, fast, premium = _prompt_model(_PROVIDERS.get(pkey, info))

        # Add network allowlist entries for this provider.
        provider_host = _PROVIDERS.get(pkey, _PROVIDERS.get("openai", {})).get("host")
        if provider_host and provider_host not in allowed_hosts:
            allowed_hosts.append(provider_host)
        if pkey in ("openai", "openai-codex"):
            for h in ("auth.openai.com", "chatgpt.com"):
                if h not in allowed_hosts:
                    allowed_hosts.append(h)

        providers_cfg.append(
            {
                "name": pkey,
                "model": primary,
                "fast_model": fast,
                "premium_model": premium,
                "api_key": api_key,
                "base_url": None,
            }
        )

    # -----------------------------------------------------------------------
    # Step 4: Discord (optional)
    # -----------------------------------------------------------------------
    console.print("\n[bold]Step 4 of 5 — Discord Integration[/]  [dim](optional)[/]")
    discord_cfg: dict | None = None

    if click.confirm("  Configure Discord bot?", default=False):
        console.print(
            "\n  You'll need:\n"
            "    • Bot token in an environment variable (e.g. DISCORD_BOT_TOKEN)\n"
            "    • Application ID from discord.com/developers/applications\n"
        )

        bot_token = click.prompt(
            "  Bot token (from discord.com/developers → Bot → Reset Token)",
            hide_input=True,
            default="",
        ).strip()
        # Strip "Bot " prefix if user included it — we store the raw token
        if bot_token.lower().startswith("bot "):
            bot_token = bot_token[4:].strip()
        token_env_var = "DISCORD_BOT_TOKEN"  # kept for fallback; direct token takes precedence
        application_id = click.prompt("  Application ID", default="").strip()

        # DM policy
        console.print("\n  DM policy:")
        console.print("    [bold]1[/]. disabled  — ignore all DMs")
        console.print("    [bold]2[/]. allowlist — only listed user IDs can DM")
        console.print("    [bold]3[/]. pairing   — users must send !pair to be approved")
        console.print("    [bold]4[/]. open      — anyone can DM")
        dm_choice = click.prompt("  Selection", default="1")
        dm_policy_map = {"1": "disabled", "2": "allowlist", "3": "pairing", "4": "open"}
        dm_policy = dm_policy_map.get(dm_choice, "disabled")

        dm_allowlist: list[str] = []
        if dm_policy == "allowlist":
            raw = click.prompt("  Allowed user IDs (comma-separated)", default="").strip()
            dm_allowlist = [u.strip() for u in raw.split(",") if u.strip()]

        # Guild (server) policies
        guild_policies: list[dict] = []
        if click.confirm("\n  Add a guild (server) policy?", default=False):
            while True:
                guild_id = click.prompt("    Guild ID", default="").strip()
                if not guild_id:
                    break
                require_mention = click.confirm("    Require @mention to respond?", default=True)
                raw_channels = click.prompt(
                    "    Allowed channels (comma-separated, blank=all)", default=""
                ).strip()
                allowed_channels = [c.strip() for c in raw_channels.split(",") if c.strip()]
                console.print(
                    "    Mode: [bold]1[/]=full  [bold]2[/]=safe-chat  [bold]3[/]=no-tools"
                )
                mode_map = {"1": "full", "2": "safe_chat_only", "3": "no_tools"}
                mode = mode_map.get(click.prompt("    Mode", default="1"), "full")
                guild_policies.append(
                    {
                        "guild_id": guild_id,
                        "require_mention": require_mention,
                        "allowed_channels": allowed_channels,
                        "mode": mode,
                    }
                )
                if not click.confirm("    Add another guild?", default=False):
                    break

        ack_reaction = click.prompt(
            "\n  Acknowledgement reaction emoji (blank to disable)", default="eyes"
        ).strip()
        ignore_bots = click.confirm("  Ignore messages from other bots?", default=True)

        discord_cfg = {
            "bot_token": bot_token,
            "token_env_var": token_env_var,
            "application_id": application_id,
            "dm_policy": dm_policy,
            "dm_allowlist": dm_allowlist,
            "guild_policies": guild_policies,
            "ack_reaction": ack_reaction,
            "ignore_bots": ignore_bots,
        }
        # Discord needs discord.com and gateway.discord.gg in allowed_hosts
        for host in ("discord.com", "gateway.discord.gg"):
            if host not in allowed_hosts:
                allowed_hosts.append(host)
        console.print("  [green]Discord configured.[/]")
    else:
        console.print("  [dim]Skipping Discord — add a discord: section to config.yaml later.[/]")

    # -----------------------------------------------------------------------
    # Step 5: Summary + write
    # -----------------------------------------------------------------------
    console.print("\n[bold]Step 5 of 5 — Write Configuration[/]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("Config path", str(config_file))
    table.add_row("Workspace", str(workspace_path))
    for p in providers_cfg:
        key = p.get("api_key") or ""
        if not key:
            key_display = "(env var)"
        elif key.startswith("vault://"):
            key_display = key
        elif p["name"] == "openai" and any(r[0] == "openai-oauth" for r in verify_results):
            key_display = "(OAuth token)"
        elif p["name"] == "anthropic" and any(
            r[0] == "anthropic-setup-token" for r in verify_results
        ):
            key_display = "(setup-token)"
        else:
            key_display = _mask_key(key)
        table.add_row(f"Provider: {p['name']}", f"{p['model']}  auth={key_display}")
    for pname, ok in verify_results:
        table.add_row(f"Verified: {pname}", "[green]OK[/]" if ok else "[red]FAILED[/]")
    if discord_cfg:
        tok_display = (
            _mask_key(discord_cfg["bot_token"])
            if discord_cfg.get("bot_token")
            else f"env:{discord_cfg['token_env_var']}"
        )
        table.add_row("Discord", f"token={tok_display}  dm={discord_cfg['dm_policy']}")
    console.print(table)

    if not click.confirm("\n  Write this configuration?", default=True):
        console.print("[dim]Aborted. Nothing written.[/]")
        return

    yaml_content = _build_config_yaml(
        workspace=str(workspace_path),
        providers_cfg=providers_cfg,
        allowed_hosts=allowed_hosts,
        discord_cfg=discord_cfg,
    )

    try:
        # Back up existing config before overwriting
        if config_file.exists():
            try:
                from missy.config.plan import backup_config

                backup_path = backup_config(config_file)
                console.print(f"  [dim]Backed up existing config to {backup_path}[/]")
            except Exception as _bkp_exc:
                console.print(f"  [yellow]Could not back up config: {_bkp_exc}[/]")
        _write_config_atomic(config_file, yaml_content)
    except OSError as exc:
        err_console.print(f"[red]Failed to write config: {exc}[/]")
        raise SystemExit(1) from exc

    # Ensure standard directories exist
    missy_dir = config_file.parent
    for subdir, mode in [("secrets", 0o700), ("logs", 0o755)]:
        d = missy_dir / subdir
        if not d.exists():
            d.mkdir(mode=mode)

    jobs_file = missy_dir / "jobs.json"
    if not jobs_file.exists():
        jobs_file.write_text("[]", encoding="utf-8")

    console.print(
        Panel(
            f"[green bold]Configuration written to {config_file}[/]\n\n"
            + (
                "Next steps:\n"
                "  [bold]missy run[/]           — start a conversation\n"
                "  [bold]missy gateway start[/] — run as a persistent service\n"
                "  [bold]missy doctor[/]        — verify all subsystems"
            ),
            title="[green]Setup Complete[/]",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# Non-interactive setup (Feature 2)
# ---------------------------------------------------------------------------


def run_wizard_noninteractive(
    config_path: str,
    provider: str,
    api_key: str | None = None,
    api_key_env: str | None = None,
    model: str | None = None,
    workspace: str | None = None,
) -> None:
    """Write a config file without any interactive prompts.

    Args:
        config_path: Destination path for config.yaml.
        provider: Provider name (must be a key in ``_PROVIDERS``).
        api_key: Direct API key value.
        api_key_env: Environment variable name to read the API key from.
        model: Model identifier (defaults to provider's primary model).
        workspace: Workspace directory (defaults to ``~/workspace``).

    Raises:
        click.ClickException: On validation errors.
    """
    config_file = Path(config_path).expanduser()

    if provider not in _PROVIDERS:
        valid = ", ".join(sorted(_PROVIDERS))
        raise click.ClickException(f"Unknown provider {provider!r}. Valid: {valid}")

    info = _PROVIDERS[provider]

    # Resolve API key
    resolved_key: str | None = api_key
    if not resolved_key and api_key_env:
        resolved_key = os.environ.get(api_key_env)
        if not resolved_key:
            raise click.ClickException(f"Environment variable {api_key_env!r} is not set or empty.")

    # Resolve model
    resolved_model = model or info["models"].get("primary", "")
    if not resolved_model:
        raise click.ClickException(f"No default model for provider {provider!r}; specify --model.")

    # Workspace
    workspace_path = Path(workspace or "~/workspace").expanduser()
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Build allowed hosts
    allowed_hosts: list[str] = []
    host = info.get("host")
    if host:
        allowed_hosts.append(host)

    # Build provider config
    providers_cfg = [
        {
            "name": provider,
            "model": resolved_model,
            "fast_model": info["models"].get("fast", ""),
            "premium_model": info["models"].get("premium", ""),
            "api_key": resolved_key,
            "base_url": None,
        }
    ]

    yaml_content = _build_config_yaml(
        workspace=str(workspace_path),
        providers_cfg=providers_cfg,
        allowed_hosts=allowed_hosts,
    )

    # Back up if overwriting
    if config_file.exists():
        try:
            from missy.config.plan import backup_config

            backup_config(config_file)
        except Exception:
            pass

    _write_config_atomic(config_file, yaml_content)

    # Ensure standard directories exist
    missy_dir = config_file.parent
    for subdir, mode in [("secrets", 0o700), ("logs", 0o755)]:
        d = missy_dir / subdir
        if not d.exists():
            d.mkdir(mode=mode)

    jobs_file = missy_dir / "jobs.json"
    if not jobs_file.exists():
        jobs_file.write_text("[]", encoding="utf-8")

    err_console.print(f"[green]Config written to {config_file}[/]")
