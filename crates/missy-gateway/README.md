# Missy Rust gateway

Public interfaces:

* `ApiState::new(runtime: Arc<Runtime>, config: ApiConfig) -> anyhow::Result<ApiState>`
* `router(state: ApiState) -> axum::Router` (no listener; in-process tests)
* `serve(runtime: Arc<Runtime>, config: ApiConfig) -> async anyhow::Result<()>`
* `serve_discord(runtime: Arc<Runtime>, config: DiscordConfig, policy: PolicyConfig) -> async anyhow::Result<()>`
  remains source-compatible but returns an explicit state-directory-required error.
* `serve_discord_with_state_dir(runtime, config, policy, state_dir: impl AsRef<Path>)`
  opts into durable Discord transport in a caller-supplied **existing canonical absolute
  private directory** (owner-only 0700, caller-owned). No implicit state path is opened.
* `probe_discord(config: DiscordConfig, policy: PolicyConfig) -> async Result<DiscordProbe>`
  performs only authenticated GET `/users/@me` and GET `/gateway/bot`. Returns bot
  ID/username and safe session-start counters; no raw JSON/token/session/Gateway URL.
* `require_discord_transport(&DiscordConfig) -> anyhow::Result<()>` validates opt-in channel configuration.

No production startup or live state is used in tests. Scheduler lives in the separate
`missy-automation` crate and is not implicitly started by an HTTP listener.

## HTTP and console

Numeric loopback binds only. Remote addresses are rejected even with a token.
Default `127.0.0.1:8766`; no wildcard, proxy-trust, cookie authentication, or CORS.
Bearer token comes from `ApiConfig.token_env`, 32..256 printable bytes with minimum
character diversity. Generate it using a cryptographic random generator (32 hex
characters supply 128 bits). Token is hashed once and request hashes compared with
constant-time equality. Neither state nor token implements Debug. A single configured
token represents one operator principal, not a multiuser authentication service.
Tenant/user are exclusively server configuration, channel is always `api`, and the
caller chooses a validated session. Session/turn ID knowledge grants no other scope.

Every `/api/v1` route requires authentication:

| Method | Route | Input |
|---|---|---|
| GET | `/health` | none |
| POST | `/chat` | JSON `{session,prompt,request_id}` |
| GET | `/sessions` | query `session`, optional `limit` |
| GET | `/sessions/{session}/history` | optional query `limit` |
| GET | `/turns/{id}` | query `session` |
| POST | `/turns/{id}/cancel` | JSON `{session}` |
| POST | `/turns/{id}/resume` | JSON `{session}` |

`/sessions` is explicitly a scoped turn listing, not an enumeration of all principal
sessions. Store requires a full Scope for every access. No broad memory endpoint or
global promotion exists. Requests reject unknown JSON fields; input is bounded to
64KiB total and 32KiB prompt, with pagination 1..200. An operator principal has a
configured minute rate limit, 64 concurrent HTTP requests and 8 concurrent admission
or resume requests. Actual execution concurrency is bounded by Runtime config.

Chat admission returns an actual durable turn ID and `202` while running. Completed
deduplicated results return `200`; failed runs `422`; cancelled/interrupted `409`.
Runtime owns admitted tasks across HTTP disconnection. Resume waits for completion
and is never converted to a fresh prompt on lookup failure. Cancel returns `202`
only for a cancellation *request*, not proof of a stopped worker, and terminal turns
return `409`. Scope-mismatched and nonexistent turn lookups both return `404`.
Errors never include raw storage/provider internals. JSON extraction errors are
Axum's standard 4xx responses, not necessarily JSON.

`/` is a small static console with same-origin JS/CSS, no CDN, restrictive CSP,
no-store, nosniff, frame deny and referrer denial. Host is checked on all requests
against loopback/localhost plus configured port (DNS rebinding defense). Cross-site
browser API requests are denied. Tokens are explicitly entered into page memory,
cleared from the password field, never persisted, and forgotten on reload/lock.
All output uses `textContent`; no HTML interpretation. The console polls the actual
admitted turn ID until terminal state (bounded to 720 reads, with rate-limit waits),
and cancel remains responsive. Polls never replay execution. There is no SSE.

## Optional Discord transport

Discord startup uses real Gateway discovery, WebSocket identify/resume, heartbeat
ACK deadlines, reconnect with capped backoff, and REST message delivery. Every
outbound path is PolicyHttpClient, including DNS-pinned TLS WebSocket. Configure
explicit HTTPS origins for `https://discord.com` and the discovered gateway host;
REST-to-WebSocket policy origin mapping is enforced by the policy crate. No direct
`connect_async`, bypass proxy client, or live credential is used in tests.

Requires explicit user allowlist, explicit guild and channel allowlists for guild
traffic, mention protection enabled, and DM opt-in (false by default). Bots and
webhooks are rejected. A thread's channel ID must itself be allowlisted, not merely
its parent channel. Guild vs DM, actual channel/thread and source user form disjoint
scopes. Source message ID is the durable runtime request ID and exact content is
bound by the Store idempotency payload hash. Redelivery cannot silently rerun a
changed payload. Outbound mentions are disabled. UTF-16-aware chunks are under 2000
units, with 16-chunk maximum; REST 429 retries/waits and pacing are bounded.

### Durable intake, execution and delivery

`discord.sqlite` uses WAL + FULL synchronous transactions. Intake payload/hash,
persistent principal governor and Gateway resume checkpoint commit atomically before
the sequence advances in memory or in heartbeats. Replayed occurrences are hash-bound
and do not spend another rate allowance. Queue capacity is 256 outstanding occurrences;
saturation disconnects without advancing that event's checkpoint, allowing Gateway
resume instead of `try_send` loss. The 30/minute user governor survives reconnect and
process restart; over-limit admitted occurrences receive a durable rejection instead
of executing. Up to eight asks/resumes run concurrently; control requests, delivery,
heartbeat and ephemeral acknowledgements run independently. Runtime itself enforces
session leases, tool-outcome safety, capacity and scoped access.

Pending intake survives a crash before runtime admission. Runtime's own durable
admission/result closes the completion-to-outbox crash window: completed deduplicated
turns are recovered into the outbox, not discarded. Running/interrupted turns are not
implicitly reexecuted. Explicit slash resume occurrences are persisted as started
before invoking runtime resume, so a restart cannot replay the same resume attempt.
An interrupted result identifies the turn and requires a new explicit resume command.

Each output chunk has a stable occurrence/index-derived 24-character nonce and POST
uses `enforce_nonce: true`, with allowed mentions disabled. Intent is committed before
POST. Receipt IDs are durable. A restart or unknown POST result gets one policy-bound
GET of the same channel's last 100 messages, matched by bot, channel, nonce and exact
content. **No match is not evidence of failure**: a missing receipt, insufficient read
permission, old message beyond that bounded page, or ambiguous response leaves an
`uncertain` outbox row and stops the transport with an actionable error. No blind
re-POST occurs, because Discord's nonce deduplication window is finite. Later chunks
cannot pass an unresolved earlier chunk. Inspect the private journal's `outbox`
state/receipt columns and runtime scoped history before operator reconciliation.
This deliberately favors visible unresolved delivery over duplicate messages; it is
not an exactly-once claim. REST 429 retries are persisted, paced and capped at three.

The directory is bound to authenticated bot ID; a second process is excluded with a
nonblocking advisory lock. Symlink/non-private/multiply-linked database/lock/sidecar
files are rejected. Credentials from interaction events exist only in the private
journal and are never logged. Keep this directory off shared/synchronized paths and
back it up as sensitive state. SQLite has a 256MiB main-db page cap; completed dedup
receipts are retained rather than pruned unsafely. Disk-full/storage failures stop
intake without advancing the uncommitted dispatch. One instance/shard per journal is
supported; persistent conservative identify throttle is one/minute, not a distributed
multi-shard Discord session-budget coordinator. Startup checks remaining budget;
identify consumes a persisted local reservation before network I/O. Exhausting that
snapshot requires an authenticated startup preflight refresh after Discord's reset,
not an unbounded automatic identify loop.

### Slash commands and attachments

Native authenticated Gateway `INTERACTION_CREATE` accepts `/missy ask prompt:...`,
`/missy status turn:...`, `/missy cancel turn:...`, `/missy resume turn:...`. The app ID
must equal the authenticated bot ID, and the same explicit user/guild/channel rules
and exact same Scope as mentioned messages apply. A turn ID is never authorization.
There is no public interaction webhook handler or HTTP signature bypass. Commands
must be registered separately by an authorized operator (guild or global application
commands with `applications.commands` authorization); startup/probe never registers
or mutates them. Subcommands have a single required string option `prompt` for ask,
`turn` for the others. Threads need their own channel allowlist entry.

Interactions get deferred ephemeral acknowledgements on a separate bounded poller,
then idempotent PATCH of the original response. No public fallback or followup POST.
Callback identity is unique; interrupted acknowledgements can be retried safely.
Result is one bounded message, with scoped-history notice on truncation. Discord's
three-second acknowledgement and 15-minute token expiry are external hard limits:
slow network/restart can expire an interaction. Its undelivered result stays durable
and visible as uncertain, never silently discarded or publicly reposted.

Attachments are **inert metadata only**, never fetched or decoded. Maximum ten,
256-byte filenames, 25MiB each and 50MiB total declared sizes, within the overall
prompt bound. Only ID/filename/size are included as untrusted metadata; arbitrary
URL/proxy URL/MIME fields are not forwarded. No voice or message-edit handling.

### Acceptance boundary

Tests use actual local mock WS/REST through PolicyHttpClient and disposable private
SQLite journals, including process-reopen crash windows. They do not log in any live
bot. Live Discord acceptance remains **unverified without a dedicated, currently
authorized Missy bot credential and confirmed bot identity**. Never inherit Odin's
`DISCORD_TOKEN` or reuse another running bot. The parent CLI's read-only preflight is
`cargo run -p missy-cli -- --config /path/to/dedicated-test.yaml discord-probe`;
set `discord.token_env` to a dedicated `MISSY_*` variable supplied securely for that
invocation. Missing dedicated credentials is a blocked acceptance result, not success.
Shared `DISCORD_TOKEN`/`DISCORD_BOT_TOKEN` names are rejected before reading them.
Only after identity approval may the parent wire and invoke durable serving; probe
alone neither sends messages nor opens a Gateway session.
