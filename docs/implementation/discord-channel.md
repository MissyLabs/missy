# Discord Channel — Implementation Reference

Internal implementation documentation for `missy.channels.discord`.

---

## Module map

```
missy/channels/discord/
├── __init__.py          Public exports: DiscordChannel, DiscordConfig, etc.
├── config.py            Configuration dataclasses and YAML parser
├── rest.py              Discord REST API v10 client (wraps PolicyHTTPClient)
├── gateway.py           Discord Gateway WebSocket client (websockets library)
├── channel.py           DiscordChannel — BaseChannel implementation
└── commands.py          Slash command definitions and routing
```

---

## Data flow

```
                      Discord Gateway (WSS)
                             |
                    [websockets.connect]
                             |
                    DiscordGatewayClient
                      _receive_loop()
                             |
                    _handle_payload(payload)
                             |
              +----------+----------+
              |          |          |
         HELLO       DISPATCH    RECONNECT
              |          |
         heartbeat  _handle_dispatch(event_name, data)
         loop                 |
                   +----------+----------+
                   |          |          |
             MESSAGE_CREATE  GUILD_CREATE INTERACTION_CREATE
                   |                     |
            DiscordChannel         handle_slash_command()
          ._handle_message()             |
                   |               REST interaction
         access control             callback
                   |
            asyncio.Queue
                   |
            ChannelMessage
```

---

## Gateway connection lifecycle

1. **connect()** — Opens the WSS connection using `websockets.connect()`.
   Uses `_resume_gateway_url` if a previous session exists, otherwise the
   default gateway URL.

2. **HELLO opcode** — Discord sends `{"op": 10, "d": {"heartbeat_interval": N}}`.
   The client:
   - Starts `_heartbeat_loop(interval_seconds)` as an `asyncio.Task`.
   - Sends IDENTIFY (first connect) or RESUME (reconnect with a valid session).

3. **IDENTIFY** — Sends the bot token, intent bitmask, and platform info.
   `_INTENTS = GUILDS | GUILD_MESSAGES | DIRECT_MESSAGES | MESSAGE_CONTENT`

4. **READY event** — Discord responds with the bot user object and session
   metadata. The client stores:
   - `_discord_session_id` — for RESUME
   - `_resume_gateway_url` — server-assigned resume URL
   - `_bot_user_id` — the bot's own Discord user ID

5. **Heartbeat loop** — Sends `{"op": 1, "d": <last_sequence>}` on the
   interval provided in HELLO, with a random initial jitter to avoid
   thundering-herd reconnects.

6. **RECONNECT (op 7)** — Server requests a reconnect. The client closes
   the connection; `run()` catches the exception and reconnects.

7. **INVALID_SESSION (op 9)** — `d: true` means the session is resumable;
   `d: false` means a fresh IDENTIFY is required (session state is cleared).

8. **RESUME (op 6)** — Sent instead of IDENTIFY when `_discord_session_id`
   and `_sequence` are both set. On success Discord sends a RESUMED event.

9. **run()** — The outer loop catches all exceptions, logs them, waits 5
   seconds, and reconnects. Call `disconnect()` to stop cleanly.

---

## Access control flow

```
_handle_message(data)
        |
        v
_is_own_message(author_id)     -- compare author.id to bot_user_id / account_id
   YES -> drop silently
   NO  |
        v
_allow_bot_author(author, content, guild_id)
   NO (filtered) -> emit discord.channel.bot_filtered deny audit event
   YES |
        v
  guild_id is None?
        |               |
       YES             NO
        |               |
_check_dm_policy()  _check_guild_policy()
        |               |
   allowed?        allowed?
        |               |
       YES -> enqueue ChannelMessage + emit message_received
        NO -> (deny audit event already emitted by the policy check)
```

### DM policy checks

| Policy | Logic |
|---|---|
| `DISABLED` | Always deny, emit `message_denied` |
| `OPEN` | Always allow |
| `ALLOWLIST` | Allow if `author_id in dm_allowlist`, else deny + emit `allowlist_denied` |
| `PAIRING` | Allow if in `dm_allowlist`. `!pair` command adds to `pending_pairs` + emits `pairing_wait` |

### Guild policy checks (in order)

1. No policy entry for guild → deny + emit `message_denied` (reason: `no_guild_policy`)
2. `policy.enabled == False` → deny + emit `message_denied` (reason: `guild_disabled`)
3. `allowed_channels` non-empty and channel name not in list → deny + emit `allowlist_denied`
4. `allowed_users` non-empty and author not in list → deny + emit `allowlist_denied`
5. `require_mention == True` and bot not @-mentioned → deny + emit `require_mention_filtered`
6. All checks pass → allow

---

## Audit events schema

All events share the base `AuditEvent` fields:
`timestamp`, `session_id`, `task_id`, `event_type`, `category`, `result`, `detail`, `policy_rule`.

`category` is always `"network"` for Discord events.

### discord.channel.message_received

```json
{
  "event_type": "discord.channel.message_received",
  "result": "allow",
  "detail": {
    "author_id": "<discord user id>",
    "channel_id": "<discord channel id>",
    "guild_id": "<discord guild id or 'dm'>"
  }
}
```

### discord.channel.message_denied

```json
{
  "event_type": "discord.channel.message_denied",
  "result": "deny",
  "detail": {
    "reason": "dm_disabled | pairing_required | no_guild_policy | guild_disabled",
    "author_id": "<discord user id>"
  }
}
```

### discord.channel.bot_filtered

```json
{
  "event_type": "discord.channel.bot_filtered",
  "result": "deny",
  "detail": {
    "author_id": "<discord user id>",
    "is_bot": true
  }
}
```

### discord.channel.allowlist_denied

```json
{
  "event_type": "discord.channel.allowlist_denied",
  "result": "deny",
  "detail": {
    "reason": "dm_allowlist | channel_not_allowed | user_not_in_allowlist",
    "author_id": "<discord user id>",
    "channel_name": "<optional>",
    "guild_id": "<optional>"
  }
}
```

### discord.channel.require_mention_filtered

```json
{
  "event_type": "discord.channel.require_mention_filtered",
  "result": "deny",
  "detail": {
    "reason": "mention_required",
    "guild_id": "<discord guild id>",
    "author_id": "<discord user id>"
  }
}
```

### discord.channel.pairing_wait

```json
{
  "event_type": "discord.channel.pairing_wait",
  "result": "allow",
  "detail": {
    "author_id": "<discord user id>"
  }
}
```

### discord.channel.reply_sent

```json
{
  "event_type": "discord.channel.reply_sent",
  "result": "allow | error",
  "detail": {
    "channel_id": "<discord channel id>",
    "reply_to": "<message id or null>",
    "error": "<only on result=error>"
  }
}
```

### discord.gateway.*

```json
{
  "event_type": "discord.gateway.connect | discord.gateway.disconnect | discord.gateway.heartbeat_sent | discord.gateway.session_resumed",
  "result": "allow | error",
  "detail": {
    "url": "<wss url — only on connect>",
    "seq": "<sequence number — only on heartbeat_sent>",
    "bot_user_id": "<only on READY connect>",
    "error": "<only on result=error>"
  }
}
```

---

## Configuration examples

### settings.py integration

`MissyConfig.discord` is `Optional[DiscordConfig]` (defaults to `None`).
When a `discord:` key is present in the YAML, `load_config()` calls
`parse_discord_config()` via a local import to avoid circular imports:

```
missy.config.settings
  -> (local import inside load_config)
  -> missy.channels.discord.config.parse_discord_config
```

This avoids the circular chain:
`settings -> discord.config -> discord.channel -> discord.rest -> gateway.client -> policy.engine -> settings`

### Instantiating a DiscordChannel

```python
from missy.channels.discord.channel import DiscordChannel
from missy.channels.discord.config import DiscordAccountConfig, DiscordDMPolicy

account = DiscordAccountConfig(
    token_env_var="DISCORD_BOT_TOKEN",
    application_id="123456789",
    dm_policy=DiscordDMPolicy.OPEN,
)

channel = DiscordChannel(account_config=account)

import asyncio

async def main():
    await channel.start()
    while True:
        msg = await channel.areceive()
        if msg:
            await channel.send_to(
                msg.metadata["discord_channel_id"],
                f"You said: {msg.content}",
                reply_to=msg.metadata["discord_message_id"],
            )

asyncio.run(main())
```

---

## Testing notes

- `DiscordGatewayClient` and `DiscordRestClient` are patched out in unit tests
  using `unittest.mock.patch` at construction time.
- `DiscordChannel._emit_audit()` can be monkeypatched to use an isolated
  `EventBus` so tests do not pollute the global bus.
- No real network calls are made in the unit test suite.
- `DiscordRestClient` tests use a mock `PolicyHTTPClient` that records
  call arguments without issuing real HTTP requests.
