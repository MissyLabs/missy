# Missy Rust gateway

Public interfaces:

* `ApiState::new(runtime: Arc<Runtime>, config: ApiConfig) -> anyhow::Result<ApiState>`
* `router(state: ApiState) -> axum::Router` (no listener; in-process tests)
* `serve(runtime: Arc<Runtime>, config: ApiConfig) -> async anyhow::Result<()>`
* `serve_discord(runtime: Arc<Runtime>, config: DiscordConfig, policy: PolicyConfig) -> async anyhow::Result<()>`
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

Known limits: Gateway resume state is memory-only across reconnects (not process
restarts), but runtime turn deduplication is durable. A serial message worker and
32-entry queue bound concurrency; overflow/rate-limited messages are dropped rather
than admitted. Per-user 30/minute rate counters reset on reconnect. Response delivery
is at-most-once: a completed deduplicated turn is not reposted, so a crash between
turn completion and REST delivery can lose the reply. There is no durable outbound
delivery journal, distributed shard/session-start governor, slash-command, attachment,
voice, edit-event, or automatic reconciliation support. Fatal auth/intents closes
stop rather than reconnect forever. Reconnect/backoff and heartbeat code is real but
not live-Discord acceptance tested; pure allowlist/chunk tests plus the policy crate's
mock pinned-WebSocket transport tests are not a claim of production Discord parity.
