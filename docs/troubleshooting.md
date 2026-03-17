# Troubleshooting

This guide covers common errors, diagnostic commands, and solutions for
Missy.

---

## Common Errors and Fixes

### "Configuration file not found"

**Error message**:
```
Configuration error: Configuration file not found: ~/.missy/config.yaml
```

**Cause**: Missy cannot find the configuration file at the expected path.

**Fix**: Run `missy init` to create the default configuration:

```bash
missy init
```

This creates `~/.missy/config.yaml` with secure-by-default settings,
along with `~/.missy/audit.jsonl`, `~/.missy/jobs.json`, and `~/workspace/`.

If your config is at a non-default path, specify it with `--config`:

```bash
missy --config /path/to/my-config.yaml ask "Hello"
```

Or set the `MISSY_CONFIG` environment variable:

```bash
export MISSY_CONFIG=/path/to/my-config.yaml
```

---

### "ProviderRegistry has not been initialised"

**Error message**:
```
RuntimeError: ProviderRegistry has not been initialised.
Call missy.providers.registry.init_registry(config) first.
```

**Cause**: The provider registry singleton was accessed before it was
initialised.  This happens when subsystems are initialised out of order.

**Fix**: Ensure subsystems are initialised in the correct order:

1. `load_config(path)` -- parse the YAML config
2. `init_policy_engine(cfg)` -- initialise the policy engine
3. `init_audit_logger(cfg.audit_log_path)` -- set up audit logging
4. `init_registry(cfg)` -- register providers

The CLI's `_load_subsystems()` function handles this automatically.  If you
see this error when using Missy programmatically, call the init functions in
the order above.

Similar errors exist for other singletons:
- `PolicyEngine has not been initialised`
- `PluginLoader has not been initialised`
- `SkillRegistry has not been initialised`
- `AuditLogger has not been initialised`

The fix is the same: call the corresponding `init_*()` function first.

---

### Provider error / API key not set

**Error message**:
```
Provider error: No providers available. Configured provider was 'anthropic'.
Ensure at least one provider is initialised and its API key is set.
```

**Cause**: The configured provider's `is_available()` method returned `False`,
and no fallback provider is available either.  For Anthropic and OpenAI, this
means the API key is not set.

**Fix**: Set the API key as an environment variable:

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."
```

The environment variable name is derived from the provider config key:
`<KEY>_API_KEY` where `<KEY>` is the uppercased mapping key in your config
file.

Verify the provider is available:

```bash
missy providers
```

This shows each configured provider and whether it reports itself as available.

For Ollama, verify the server is running:

```bash
curl http://localhost:11434/api/tags
```

---

### "Anthropic authentication failed"

**Error message**:
```
Provider error: Anthropic authentication failed: ...
```

**Cause**: The API key is set but is invalid or expired.

**Fix**: Verify your API key is correct and active.  Re-export it:

```bash
export ANTHROPIC_API_KEY="sk-ant-your-actual-key"
```

---

### Network request denied by policy

**Error message**:
```
PolicyViolationError: Network access to 'example.com' denied by policy.
```

**Cause**: The destination host is not in any of the network allowlists and
`network.default_deny` is `true` (the default).

**Fix**: Add the host to the appropriate allowlist in your config:

```yaml
network:
  default_deny: true
  allowed_hosts:
    - "example.com"           # Add the denied host here
```

For domain suffix matching (e.g. any subdomain of `example.com`):

```yaml
network:
  allowed_domains:
    - "*.example.com"
```

For IP ranges:

```yaml
network:
  allowed_cidrs:
    - "10.0.0.0/8"
```

For subsystem-specific access, use the per-category lists:

```yaml
network:
  provider_allowed_hosts:
    - "api.anthropic.com"
  discord_allowed_hosts:
    - "discord.com"
    - "gateway.discord.gg"
  tool_allowed_hosts:
    - "api.github.com"
```

Review recent policy denials:

```bash
missy audit security
```

---

### Filesystem access denied

**Error message**:
```
PolicyViolationError: Filesystem write to '/home/user/documents/file.txt' denied.
```

**Cause**: The path is outside the configured `allowed_read_paths` or
`allowed_write_paths`.

**Fix**: Add the directory to the appropriate list in your config:

```yaml
filesystem:
  allowed_read_paths:
    - "~/workspace"
    - "~/documents"           # Add the parent directory
    - "/tmp"
  allowed_write_paths:
    - "~/workspace"
    - "~/.missy"
```

Paths are matched by prefix.  Adding `"~/documents"` allows access to all
files and subdirectories under that path.

---

### Discord token missing

**Error message**:
```
Account 1 (DISCORD_BOT_TOKEN): env var DISCORD_BOT_TOKEN is not set.
```

**Cause**: The environment variable holding the Discord bot token is not set.

**Fix**: Export the bot token:

```bash
export DISCORD_BOT_TOKEN="your-bot-token-here"
```

The environment variable name is configured in your Discord account config
via the `token_env_var` field (default: `DISCORD_BOT_TOKEN`):

```yaml
discord:
  enabled: true
  accounts:
    - token_env_var: DISCORD_BOT_TOKEN
```

---

### Discord probe fails

**Error message**:
```
Account 1 (DISCORD_BOT_TOKEN): probe failed -- ...
```

**Cause**: The Discord API connectivity test failed.  Common reasons:

1. The bot token is invalid.
2. The Discord API host is not in the network allowlist.
3. Network connectivity issues.

**Fix**:

First, ensure Discord API hosts are in your network allowlist:

```yaml
network:
  allowed_hosts:
    - "discord.com"
  discord_allowed_hosts:
    - "discord.com"
    - "gateway.discord.gg"
```

Then verify the token is valid:

```bash
missy discord probe
```

If the probe still fails, check network connectivity:

```bash
curl -H "Authorization: Bot YOUR_TOKEN" https://discord.com/api/v10/users/@me
```

---

### Schedule parse error

**Error message**:
```
Invalid schedule expression: Unrecognised schedule string: '...'
```

**Cause**: The schedule string does not match any supported format.

**Fix**: Use one of the supported formats:

| Format | Example |
|---|---|
| `every N seconds` | `"every 30 seconds"` |
| `every N minutes` | `"every 5 minutes"` |
| `every N hours` | `"every 2 hours"` |
| `daily at HH:MM` | `"daily at 09:00"` |
| `weekly on <day> at HH:MM` | `"weekly on Monday at 09:00"` |
| `weekly on <day> HH:MM` | `"weekly on friday 14:30"` |

Notes:
- Times are in 24-hour format.
- Day names are case-insensitive but must be spelled out in full (Monday,
  Tuesday, etc.).
- The `at` keyword is required for daily schedules and optional for weekly.

---

### Shell command denied

**Error message**:
```
PolicyViolationError: Shell command 'curl' denied by policy.
```

**Cause**: Shell execution is disabled, or the command is not in the allowlist.

**Fix**: Enable shell execution and add the command to the allowlist:

```yaml
shell:
  enabled: true
  allowed_commands:
    - "curl"
    - "git"
    - "python3"
```

Note: An empty `allowed_commands` list blocks all commands even when
`enabled: true`.

---

### Plugin denied

**Error message**:
```
PolicyViolationError: Plugin 'my_plugin' denied: plugins are disabled in configuration.
```
or
```
PolicyViolationError: Plugin 'my_plugin' denied: not in allowed_plugins list.
```

**Cause**: The plugin system is disabled, or the specific plugin is not
allowlisted.

**Fix**: Enable plugins and add the plugin to the allowlist:

```yaml
plugins:
  enabled: true
  allowed_plugins:
    - "my_plugin"
```

---

## Diagnostic Commands

### Debug Mode

Enable verbose logging with the `--debug` flag:

```bash
missy --debug ask "Hello"
missy --debug run
```

This sets Python logging to `DEBUG` level, which outputs:
- Policy check details (host checked, result)
- Provider selection and fallback decisions
- Audit event publishing
- Job loading and scheduling details
- Plugin loading steps

### Reading the Audit Log

**Recent events** (all categories):

```bash
missy audit recent
missy audit recent --limit 100
```

**Filter by category**:

```bash
missy audit recent --category network
missy audit recent --category filesystem
missy audit recent --category shell
missy audit recent --category plugin
missy audit recent --category scheduler
missy audit recent --category provider
```

**Policy violations only** (events with `result: "deny"`):

```bash
missy audit security
missy audit security --limit 100
```

**Raw log with jq**:

```bash
# All denials
jq 'select(.result == "deny")' ~/.missy/audit.jsonl

# Network denials with host details
jq 'select(.result == "deny" and .category == "network")' ~/.missy/audit.jsonl

# Events from the last hour
jq 'select(.timestamp > "2026-03-11T08:00")' ~/.missy/audit.jsonl
```

### Discord Diagnostics

**Connection probe** -- tests API connectivity and token validity for each
configured account:

```bash
missy discord probe
```

**Discord status** -- shows account configuration without making API calls:

```bash
missy discord status
```

**Discord audit** -- shows Discord-specific audit events:

```bash
missy discord audit
missy discord audit --limit 100
```

### Provider Status

List all configured providers and their availability:

```bash
missy providers
```

Shows: name, model, base URL, timeout, and whether the provider is currently
available (API key set and SDK installed).

### Plugin Status

Show the plugin system status, allowlist, and loaded plugins:

```bash
missy plugins
```

### Skill Status

List all registered skills:

```bash
missy skills
```

### Scheduler Status

List all scheduled jobs with their status:

```bash
missy schedule list
```

---

## General Debugging Tips

1. **Start with `--debug`**: Most issues become clear with debug logging
   enabled.

2. **Check the audit log**: Policy denials are always recorded.  Use
   `missy audit security` to find them.

3. **Verify config loads**: Run `missy providers` or `missy plugins` to
   confirm the config file parses without error.

4. **Test connectivity separately**: For provider issues, test the API
   endpoint directly (e.g. `curl https://api.anthropic.com/`).

5. **Check environment variables**: Many issues stem from missing API keys.
   Verify with `echo $ANTHROPIC_API_KEY`.

6. **Inspect jobs.json manually**: If the scheduler behaves unexpectedly,
   stop Missy and inspect `~/.missy/jobs.json` with `cat` or `jq`.

7. **Reset to defaults**: If the config is corrupted, back it up and re-run
   `missy init`:

   ```bash
   mv ~/.missy/config.yaml ~/.missy/config.yaml.broken
   missy init
   ```
