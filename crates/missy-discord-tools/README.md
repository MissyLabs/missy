# Native Discord tools

`DiscordTools::new(PolicyConfig, DiscordToolsConfig) -> anyhow::Result<DiscordTools>`
implements `missy_core::ToolExecutor`. Configuration derives serde with unknown
fields denied. Defaults expose no tools, read no credential and perform no I/O.
There is no gateway login, borrowed gateway credential, Python, or direct HTTP
client. All network requests pass through `PolicyHttpClient`.

## Operator configuration

Explicitly enable the executor and each capability, then supply:

- `token_env`: a dedicated trusted-local `MISSY_DISCORD_TOOLS_*` variable, default
  `MISSY_DISCORD_TOOLS_TOKEN`. Tool arguments cannot select or override it.
- `bot_user_id`: exact bot snowflake expected in upload responses.
- `allowed_guilds`, `allowed_users`: exact decimal string snowflakes.
- `allowed_channels`: map from channel ID to its allowed guild ID. The operator
  attests this mapping; arbitrary channel discovery is deliberately absent.
- `max_file_bytes`: positive single-file limit, default 3 MiB. The executor also
  lowers the descriptor-based Policy read bound to this limit.
- Policy `allowed_origins`: must explicitly include `https://discord.com`.
- Policy `read_roots`: explicit upload source directories. Symlinks and traversal
  remain denied by the policy's descriptor-relative reader.

The production API base is fixed at `https://discord.com/api/v10`. Public fixture
configuration exists solely for tests: `fixture_mode=true` and a fixed
`http://127.0.0.1:PORT/api/v10` (or loopback IPv6) base, plus **both** an explicitly
allowed policy origin and `allow_private_network=true`. No arbitrary public
override, credential URL, redirects, query or fragment is accepted.

## Capabilities and safety contract

`discord_lookup_user` takes exactly `guild_id` and `user_id`. It performs one
bounded `GET /guilds/{guild}/members/{user}` and projects only ID, username,
global name, nickname and bot flag. No email, phone, upstream error body,
authentication header or arbitrary field is returned. It is read-only.

`discord_upload_file` takes `guild_id`, `channel_id`, absolute `path`, optional
`filename`, optional `mime_type`, and optional `caption` (maximum 2000 Unicode
characters). It uploads actual binary bytes in one multipart request, not an
instruction or placeholder. A single filename is sanitized to safe ASCII;
PNG/JPEG/GIF/WebP/WAV/PDF/ZIP/MP4 require matching extension and magic bytes.
These are signature checks, **not** full media decoders or malware scanning.
The entire multipart request is capped at 4 MiB. Mentions are disabled.

**Parent integration requirement:** upload's definition has `side_effect=true`.
Always compose it under the runtime's durable, scope/turn/call/tool-argument and
configuration-bound human approval guard. Do not expose the raw executor on an
unguarded caller path. A `ToolContext` alone is not proof of approval. Lookup is
read-only and can remain available when mutation approval is disabled.

The 25-character SHA-256-derived nonce binds length-framed tenant/user/channel/
session/turn/call identifiers and is sent with `enforce_nonce=true`. The runtime
must keep its durable journal: Discord nonce de-duplication is time-limited and
the executor's bounded 65536-entry duplicate barrier is process-local only.
Configuration fingerprints include policy, config and a hash of the current
dedicated credential, so rotation invalidates stale approvals. No token-bearing
Debug implementation or upstream error-body forwarding exists; returned strings
also redact any exact token occurrence.

There is **one POST attempt only**. A failed/invalid POST may trigger one bounded
read-only `GET /channels/{channel}/messages?limit=25` reconciliation. Delivery is
known only when a decimal-u64 message ID, exact channel, configured bot ID,
`bot=true`, nonce, one attachment with exact filename/size and HTTPS Discord CDN
URL all match. The attachment URL is returned as untrusted data and never fetched.
Absence, HTTP 5xx, transport failure, malformed response or ambiguous cancellation
produce typed `UploadOutcomeUnknown`; the runtime must never replay that effect.
HTTP 429 parses bounded `retry_after`, returns a sanitized error, and never sleeps
or automatically reposts. The duplicate barrier remains occupied after 429 too.

Cancellation and policy denial before dispatch use typed `ToolDispatchRefused`.
Cancellation once a request might have been dispatched is conservatively UNKNOWN.
This executor does not implement a durable queue or long-running retry service.

## Verification

`cargo test -p missy-discord-tools` uses native loopback TCP fixture servers with
synthetic tokens only. It verifies actual multipart bytes/headers and nonce,
read-only DTO projection, cross-channel denials before networking, symlink and
size bounds, safe Debug/fingerprints, no repeated POST on 5xx/429, and successful
read-only reconciliation. No real Discord credentials or account login is used.
