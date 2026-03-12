# Configuration Reference

Missy is configured through a single YAML file, by default located at
`~/.missy/config.yaml`.  The path can be overridden with the `--config` CLI
flag or the `MISSY_CONFIG` environment variable.

Run `missy init` to create a default configuration file with a
secure-by-default posture.

All configuration is loaded by `missy.config.settings.load_config()` and
parsed into a `MissyConfig` dataclass hierarchy.

---

## Table of Contents

1. [network](#network)
2. [filesystem](#filesystem)
3. [shell](#shell)
4. [plugins](#plugins)
5. [scheduling](#scheduling)
6. [providers](#providers)
7. [discord](#discord)
8. [workspace_path](#workspace_path)
9. [audit_log_path](#audit_log_path)
10. [max_spend_usd](#max_spend_usd)
11. [Full Annotated Example](#full-annotated-example)

---

## `network`

Controls all outbound network access.  When `default_deny` is `true` (the
default), every outbound request is blocked unless the destination matches an
entry in one of the allowlists.

| Key | Type | Default | Description |
|---|---|---|---|
| `default_deny` | bool | `true` | When `true`, block all outbound traffic not explicitly allowed. Set to `false` to allow all traffic (not recommended). |
| `allowed_cidrs` | list of strings | `[]` | CIDR blocks that are reachable (e.g. `"10.0.0.0/8"`, `"192.168.1.0/24"`). |
| `allowed_domains` | list of strings | `[]` | Fully-qualified domain names or suffix patterns (e.g. `"*.github.com"`) that are reachable. |
| `allowed_hosts` | list of strings | `[]` | Explicit hostname or `host:port` strings that are reachable (e.g. `"api.anthropic.com"`, `"localhost:11434"`). |
| `provider_allowed_hosts` | list of strings | `[]` | Additional hosts allowed specifically for provider API traffic. Merged (union) with `allowed_hosts`. |
| `tool_allowed_hosts` | list of strings | `[]` | Additional hosts allowed specifically for tool HTTP requests. Merged (union) with `allowed_hosts`. |
| `discord_allowed_hosts` | list of strings | `[]` | Additional hosts allowed specifically for Discord API traffic. Merged (union) with `allowed_hosts`. |

The per-category `*_allowed_hosts` lists are unioned with the global
`allowed_hosts` and `allowed_domains` lists during policy evaluation.  They
exist so that operators can grant narrow access to specific subsystems without
opening the allowlist globally.

---

## `filesystem`

Controls which directories the agent may read from and write to.  Paths
support tilde (`~`) expansion.

| Key | Type | Default | Description |
|---|---|---|---|
| `allowed_read_paths` | list of strings | `[]` | Directories the agent may read from.  Access to any path outside these trees raises `PolicyViolationError`. |
| `allowed_write_paths` | list of strings | `[]` | Directories the agent may write to.  Access to any path outside these trees raises `PolicyViolationError`. |

Paths are matched by prefix after resolution.  For example, if
`allowed_read_paths` contains `"~/workspace"`, the agent may read any file
under `~/workspace/` and its subdirectories.

---

## `shell`

Controls shell command execution.  Disabled by default.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch for shell execution.  When `false`, all shell commands are denied regardless of `allowed_commands`. |
| `allowed_commands` | list of strings | `[]` | Whitelist of command names (not full paths) that may be executed when `enabled` is `true`.  An empty list means no commands are allowed even when the shell is enabled. |

Example:

```yaml
shell:
  enabled: true
  allowed_commands:
    - "git"
    - "python3"
    - "ls"
```

---

## `plugins`

Controls the plugin loading system.  Disabled by default.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch for the plugin system.  When `false`, no plugins may be loaded. |
| `allowed_plugins` | list of strings | `[]` | Whitelist of plugin name strings that may be loaded when `enabled` is `true`.  A plugin whose name does not appear in this list is denied with a `PolicyViolationError`. |

To disable all third-party plugins globally, set `enabled: false` (which is
the default).  Even when `enabled: true`, only plugins explicitly named in
`allowed_plugins` may load.

---

## `scheduling`

Controls the job scheduler subsystem.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Master switch for scheduled job execution.  When `false`, no new jobs may be added or run. |
| `max_jobs` | int | `0` | Maximum number of concurrent scheduled jobs.  `0` means unlimited. |

---

## `providers`

A mapping of provider names to their configuration.  Each key is a logical
name (e.g. `anthropic`, `openai`, `ollama`) and the value is a configuration
block.

| Key | Type | Default | Description |
|---|---|---|---|
| `name` | string | (same as the mapping key) | Canonical provider name.  Must match one of the built-in provider implementations: `anthropic`, `openai`, or `ollama`. |
| `model` | string | (required) | Model identifier to use for inference (e.g. `"claude-3-5-sonnet-20241022"`, `"gpt-4o"`, `"llama3.2"`). |
| `api_key` | string | `null` | API key for the provider.  **Not recommended** -- use environment variables instead.  When `null`, the provider reads `<PROVIDER_NAME>_API_KEY` from the environment (e.g. `ANTHROPIC_API_KEY`). |
| `base_url` | string | `null` | Override the provider's default API endpoint.  Required for Ollama (`"http://localhost:11434"`).  Also useful for OpenAI-compatible third-party services. |
| `timeout` | int | `30` | Request timeout in seconds. |
| `enabled` | bool | `true` | When `false`, the provider is loaded but treated as unavailable by the registry. |

### API Key Resolution Order

For each provider, the API key is resolved in this order:

1. `api_key` field in the YAML config (not recommended).
2. Environment variable `<PROVIDER_NAME>_API_KEY`, where `<PROVIDER_NAME>` is
   the uppercased mapping key (e.g. `ANTHROPIC_API_KEY` for a provider keyed
   as `anthropic`).
3. If neither is set, the provider reports itself as unavailable.

---

## `discord`

Optional Discord bot integration.  Disabled by default.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch for the Discord channel. |
| `accounts` | list | `[]` | List of Discord bot account configurations. |

### Account Configuration

Each entry in `accounts` supports:

| Key | Type | Default | Description |
|---|---|---|---|
| `token_env_var` | string | `"DISCORD_BOT_TOKEN"` | Name of the environment variable holding the bot token.  Tokens are never stored in the config file. |
| `account_id` | string | `null` | Explicit bot user ID.  When omitted, fetched from Discord on startup. |
| `application_id` | string | `""` | Discord application ID, required for slash command registration. |
| `dm_policy` | string | `"disabled"` | How the bot handles direct messages.  One of: `"pairing"`, `"allowlist"`, `"open"`, `"disabled"`. |
| `dm_allowlist` | list of strings | `[]` | User IDs permitted for DM when `dm_policy` is `"allowlist"`. |
| `ack_reaction` | string | `""` | Emoji name used to acknowledge receipt (e.g. `"eyes"`).  Empty string disables. |
| `ignore_bots` | bool | `true` | Ignore messages from other bots. |
| `allow_bots_if_mention_only` | bool | `false` | When `true` and `ignore_bots` is `true`, bot messages that @-mention this bot are not ignored. |
| `guild_policies` | mapping | `{}` | Per-guild access control policies, keyed by guild ID string. |

### Guild Policy

Each entry in `guild_policies` supports:

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | When `false`, the bot ignores all messages from this guild. |
| `require_mention` | bool | `false` | When `true`, the bot only responds if explicitly @-mentioned. |
| `allowed_channels` | list of strings | `[]` | Whitelist of channel names the bot responds in.  Empty means all channels. |
| `allowed_roles` | list of strings | `[]` | Whitelist of role names users must hold.  Empty means all roles. |
| `allowed_users` | list of strings | `[]` | Whitelist of user IDs permitted to interact.  Empty means all users. |
| `mode` | string | `"full"` | Feature mode: `"safe_chat_only"`, `"no_tools"`, or `"full"`. |

---

## `workspace_path`

| Key | Type | Default |
|---|---|---|
| `workspace_path` | string | `"~/workspace"` |

The agent's working directory.  Tilde expansion is supported.  Created by
`missy init`.

---

## `audit_log_path`

| Key | Type | Default |
|---|---|---|
| `audit_log_path` | string | `"~/.missy/audit.jsonl"` |

Path to the JSONL audit log file.  Tilde expansion is supported.  The parent
directory is created automatically.

---

## `max_spend_usd`

| Key | Type | Default |
|---|---|---|
| `max_spend_usd` | float | `0.0` |

Per-session budget cap in USD.  When set to a positive value, the agent loop
will raise `BudgetExceededError` after any provider call that causes the
accumulated session cost to exceed this limit.  An `agent.budget.exceeded`
audit event is emitted before the error propagates.

Set to `0` (the default) for unlimited spending.

```yaml
max_spend_usd: 5.00    # halt after $5 of API calls per session
```

View the current budget with `missy cost`.

---

## Full Annotated Example

The following YAML shows every configuration option with comments.

```yaml
# ---------------------------------------------------------------------------
# Network policy
# Controls all outbound network access.
# ---------------------------------------------------------------------------
network:
  default_deny: true                      # Block all traffic not explicitly allowed

  allowed_cidrs:                          # Reachable IP ranges (CIDR notation)
    - "10.0.0.0/8"
    - "172.16.0.0/12"
    - "192.168.0.0/16"

  allowed_domains:                        # Domain suffix matching
    - "*.github.com"

  allowed_hosts:                          # Explicit host or host:port pairs
    - "api.anthropic.com"
    - "api.openai.com"
    - "localhost:11434"                   # Local Ollama instance

  provider_allowed_hosts: []              # Extra hosts for provider traffic only
  tool_allowed_hosts: []                  # Extra hosts for tool traffic only
  discord_allowed_hosts:                  # Extra hosts for Discord traffic only
    - "discord.com"
    - "gateway.discord.gg"

# ---------------------------------------------------------------------------
# Filesystem policy
# Restricts which directories the agent may read or write.
# ---------------------------------------------------------------------------
filesystem:
  allowed_read_paths:
    - "~/workspace"
    - "~/.missy"
    - "/tmp"

  allowed_write_paths:
    - "~/workspace"
    - "~/.missy"

# ---------------------------------------------------------------------------
# Shell policy
# Disabled by default. Enable only specific commands you trust.
# ---------------------------------------------------------------------------
shell:
  enabled: false
  allowed_commands: []                    # e.g. ["git", "python3", "ls"]

# ---------------------------------------------------------------------------
# Plugin policy
# Disabled by default. Enable only trusted plugins by name.
# ---------------------------------------------------------------------------
plugins:
  enabled: false
  allowed_plugins: []                     # e.g. ["my_trusted_plugin"]

# ---------------------------------------------------------------------------
# Scheduling policy
# Controls the APScheduler-backed job scheduler.
# ---------------------------------------------------------------------------
scheduling:
  enabled: true
  max_jobs: 0                             # 0 = unlimited

# ---------------------------------------------------------------------------
# Provider configuration
# API keys must be set as environment variables, not in this file.
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   export OPENAI_API_KEY="sk-..."
# ---------------------------------------------------------------------------
providers:
  anthropic:
    name: anthropic
    model: "claude-3-5-sonnet-20241022"
    # api_key: null                       # Reads ANTHROPIC_API_KEY from env
    # base_url: null                      # Uses SDK default
    timeout: 30
    enabled: true

  openai:
    name: openai
    model: "gpt-4o"
    # api_key: null                       # Reads OPENAI_API_KEY from env
    # base_url: null                      # Uses SDK default
    timeout: 30
    enabled: true

  ollama:
    name: ollama
    model: "llama3.2"
    base_url: "http://localhost:11434"    # Required for Ollama
    timeout: 60
    enabled: true

# ---------------------------------------------------------------------------
# Discord integration (optional)
# ---------------------------------------------------------------------------
discord:
  enabled: false
  accounts:
    - token_env_var: DISCORD_BOT_TOKEN    # Env var holding the bot token
      application_id: "1234567890"        # Required for slash commands
      dm_policy: disabled                 # pairing | allowlist | open | disabled
      dm_allowlist: []
      ack_reaction: "eyes"
      ignore_bots: true
      allow_bots_if_mention_only: false
      guild_policies:
        "987654321":                      # Guild ID
          enabled: true
          require_mention: true
          allowed_channels:
            - "general"
            - "bot-commands"
          allowed_roles: []
          allowed_users: []
          mode: full                      # safe_chat_only | no_tools | full

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
workspace_path: "~/workspace"
audit_log_path: "~/.missy/audit.jsonl"
```
