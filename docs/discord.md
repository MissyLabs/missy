# Missy Discord Integration

Full integration guide for connecting Missy to Discord via the Discord Gateway WebSocket API.

---

## Architecture overview

```
Discord Gateway (WSS)
        |
        v
DiscordGatewayClient          -- websockets library, no discord.py
        |
        | on_message callback
        v
DiscordChannel                -- BaseChannel implementation
        |
        +-- _check_dm_policy()     -- DM access control
        +-- _check_guild_policy()  -- Guild access control
        +-- asyncio.Queue          -- inbound ChannelMessage buffer
        |
        v
DiscordRestClient             -- wraps PolicyHTTPClient
        |
        v
PolicyHTTPClient              -- enforces network policy
        |
        v
discord.com API v10
```

All outbound HTTP to `discord.com` is routed through `PolicyHTTPClient`.
The domain must be listed in `network.allowed_domains` in your Missy config.

Audit events for every allow/deny decision are published to the process-level
`event_bus` and persisted by `AuditLogger`.

---

## Setup guide

### 1. Create a Discord application

1. Visit the [Discord Developer Portal](https://discord.com/developers/applications).
2. Click **New Application** and give it a name.
3. Note the **Application ID** (also called Client ID).

### 2. Create a bot user

1. In your application, go to **Bot**.
2. Click **Add Bot**.
3. Under **Token**, click **Reset Token** and copy the token.
4. Export it as an environment variable:

```bash
export DISCORD_BOT_TOKEN="your-bot-token-here"
```

Never commit the token to source control. The Missy config stores only the
environment variable _name_, not the token value.

### 3. Enable Gateway Intents

In **Bot > Privileged Gateway Intents**, enable:

- **Server Members Intent** (for role-based access control)
- **Message Content Intent** (required to read message text)

### 4. Invite the bot to your server

Generate an invite URL under **OAuth2 > URL Generator**:

- **Scopes**: `bot`, `applications.commands`
- **Bot Permissions**: `Send Messages`, `Add Reactions`, `Read Message History`

### 5. Update network policy

Add Discord domains to your Missy config:

```yaml
network:
  default_deny: true
  allowed_domains:
    - "discord.com"
    - "gateway.discord.gg"
```

### 6. Configure the Discord section

Add a `discord:` section to your Missy config file (see Configuration reference below).

---

## Configuration reference

### Minimal — open DMs, one server

```yaml
discord:
  enabled: true
  accounts:
    - token_env_var: DISCORD_BOT_TOKEN
      application_id: "1234567890123456789"
      dm_policy: open
      guild_policies:
        "987654321098765432":
          enabled: true
          require_mention: false
          mode: full
```

### DM pairing workflow

Users must send `!pair` before the bot will respond to their DMs. An admin
must then approve with `!pair accept <user_id>`.

```yaml
discord:
  enabled: true
  accounts:
    - token_env_var: DISCORD_BOT_TOKEN
      application_id: "1234567890123456789"
      dm_policy: pairing
      guild_policies: {}
```

### Guild — mention-only, restricted channel

```yaml
discord:
  enabled: true
  accounts:
    - token_env_var: DISCORD_BOT_TOKEN
      application_id: "1234567890123456789"
      dm_policy: disabled
      guild_policies:
        "987654321098765432":
          enabled: true
          require_mention: true
          allowed_channels:
            - "bot-commands"
          mode: full
```

### Allowlisted DMs

```yaml
discord:
  enabled: true
  accounts:
    - token_env_var: DISCORD_BOT_TOKEN
      application_id: "1234567890123456789"
      dm_policy: allowlist
      dm_allowlist:
        - "123456789012345678"
        - "234567890123456789"
```

### Multi-account

```yaml
discord:
  enabled: true
  accounts:
    - token_env_var: DISCORD_BOT_TOKEN_A
      application_id: "111111111111111111"
      dm_policy: disabled
      guild_policies:
        "100000000000000001":
          enabled: true
    - token_env_var: DISCORD_BOT_TOKEN_B
      application_id: "222222222222222222"
      dm_policy: open
      ignore_bots: true
```

### Full field reference

| Field | Type | Default | Description |
|---|---|---|---|
| `token_env_var` | string | `DISCORD_BOT_TOKEN` | Environment variable holding the bot token |
| `account_id` | string | (auto) | Bot user ID; auto-resolved from Discord on start |
| `application_id` | string | `""` | Application ID for slash command registration |
| `dm_policy` | enum | `disabled` | One of: `pairing`, `allowlist`, `open`, `disabled` |
| `dm_allowlist` | list[str] | `[]` | User IDs allowed when `dm_policy: allowlist` |
| `ack_reaction` | string | `""` | Emoji to react with on message receipt |
| `ignore_bots` | bool | `true` | Ignore messages from other bots |
| `allow_bots_if_mention_only` | bool | `false` | Exempt bots that @-mention this bot |
| `guild_policies` | dict | `{}` | Per-guild access policies (see below) |

**Guild policy fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Enable/disable this guild entirely |
| `require_mention` | bool | `false` | Only respond if bot is @-mentioned |
| `allowed_channels` | list[str] | `[]` | Channel names allowed (empty = all) |
| `allowed_roles` | list[str] | `[]` | Role names required (empty = all) |
| `allowed_users` | list[str] | `[]` | User IDs allowed (empty = all) |
| `mode` | enum | `full` | `safe_chat_only`, `no_tools`, or `full` |

---

## Security model

### Network policy enforcement

Every Discord REST request uses `PolicyHTTPClient`, which calls
`PolicyEngine.check_network()` before any I/O. If `discord.com` is not
in `allowed_domains`, the request is blocked and a `PolicyViolationError`
is raised — no data leaves the machine.

### Token handling

Bot tokens are read from environment variables at runtime. They are:

- Never stored in the YAML config.
- Never logged.
- Not included in audit event detail payloads.

### Bot loop prevention

The `ignore_bots` flag (default `true`) prevents the bot from responding to
other bots, which guards against infinite message loops. The
`allow_bots_if_mention_only` flag creates a narrow exemption: a bot message
is processed only if it explicitly @-mentions this bot.

Self-messages are always filtered by comparing `author.id` to the bot's own
Discord user ID (`account_id`). This comparison uses the _account's_ ID, not
a process-global variable, so multi-account deployments are safe.

### Attachment policy

The channel does not fetch, download, or forward message attachments. Only
the text `content` field of messages is forwarded to the agent.

### Audit trail

Every access-control decision is published as an `AuditEvent` with one of
the following event types:

| Event type | Outcome | Meaning |
|---|---|---|
| `discord.channel.message_received` | allow | Message accepted and enqueued |
| `discord.channel.message_denied` | deny | Message blocked by DM policy |
| `discord.channel.bot_filtered` | deny | Bot author filtered out |
| `discord.channel.allowlist_denied` | deny | User/channel not on allowlist |
| `discord.channel.require_mention_filtered` | deny | No bot mention in guild message |
| `discord.channel.pairing_wait` | allow | Pairing request recorded |
| `discord.channel.reply_sent` | allow | Reply sent via REST |
| `discord.gateway.connect` | allow | Gateway connected |
| `discord.gateway.disconnect` | allow/error | Gateway disconnected |
| `discord.gateway.heartbeat_sent` | allow | Heartbeat sent |
| `discord.gateway.session_resumed` | allow | Session resumed |

---

## Slash commands reference

| Command | Description |
|---|---|
| `/ask <prompt>` | Ask Missy a question |
| `/status` | Show bot status and configuration summary |
| `/model [name]` | Show active AI model (dynamic switching not yet supported) |
| `/help` | List available commands |

Register commands with:

```bash
missy discord register-commands
# or guild-scoped (instant propagation):
missy discord register-commands --guild-id YOUR_GUILD_ID
```

---

## CLI commands

```bash
# Show configured accounts and their settings
missy discord status

# Test connectivity and token validity
missy discord probe

# Register slash commands globally
missy discord register-commands

# Register slash commands for a specific guild
missy discord register-commands --guild-id 987654321098765432

# Show recent Discord audit events
missy discord audit --limit 20
```

---

## Troubleshooting

### "discord.com is not in allowed_domains"

Add `discord.com` and `gateway.discord.gg` to `network.allowed_domains` in
your Missy config.

### Bot does not respond to DMs

Check `dm_policy` — the default is `disabled`. Set it to `open`, `pairing`,
or `allowlist` as needed.

### Bot does not respond in a guild channel

1. Confirm the guild ID in `guild_policies` matches the actual guild ID.
2. If `require_mention: true`, ensure users are @-mentioning the bot.
3. If `allowed_channels` is non-empty, ensure the channel name is listed.

### Slash commands not appearing

1. Run `missy discord register-commands` (or `--guild-id` for instant propagation).
2. Ensure `application_id` is set correctly in the config.
3. Global commands can take up to an hour to propagate.

### TOKEN environment variable not set

```
missy discord probe
```

This will identify which accounts have missing tokens.

### Audit log shows frequent `discord.gateway.heartbeat_sent` errors

The bot token may be invalid or expired. Reset the token in the Discord
Developer Portal and update the environment variable.

---

## Diagnostics

View recent Discord audit events:

```bash
missy discord audit --limit 50
```

View all policy violations:

```bash
missy audit security --limit 50
```

View network-category events:

```bash
missy audit recent --category network --limit 30
```
