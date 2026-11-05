# Native browser screencast channel

`missy-screencast` is an HTTP/WebSocket browser channel, not a local screen-capture
tool alias. It replaces the legacy channel's session registry, connection owner,
frame queue/analyzer lifecycle, recent analysis, and callback boundaries. It does
not import Python or read the active desktop. The only Python file is an opt-in
headless browser test fixture that supplies synthetic canvas frames.

## Consent and authentication

The channel is disabled by default. `extensions.screencast_config` points to a
JSON adapter configuration. `channel.bind` must be a fixed literal loopback
socket, such as `127.0.0.1:8780`. Wildcard/LAN binds and proxy exposure are not
supported. The exact authority must be used in the browser, not `localhost` as an
alias when the configured host is `127.0.0.1`.

The local operator invokes `screencast approve --confirm`, with the CLI's exact
tenant/user/channel/session scope and optional `--analyze`, `--deliver-analysis`,
and `--deliver-frame-artifacts`. Approval is durably written to the private
SQLite authority ledger with scope, configuration+policy digest, disclosure
flags and expiry. No model tool can issue an approval. `approve` alone neither
captures a screen nor opens a browser. The running unified gateway consumes the
grant once and writes an exclusive, owner-only pairing JSON file beneath
`<state_dir>/screencast/pairings/<receipt>.json`. CLI output includes the receipt,
private file location and token-free capture URL, **not the pairing token**.

The owner manually pastes the JSON into the capture page, clicks **Start
sharing**, and chooses the surface in the browser's real `getDisplayMedia`
picker. There is no automatic capture, camera, microphone, browser extension,
ambient browser discovery or persisted permission bypass. The token stays in
page memory only, never in a URL, localStorage, sessionStorage or cookie. Token
files contain secrets and must not be posted to a shared Discord channel.

Browser WebSocket APIs cannot set an Authorization header. `/ws` therefore
requires the exact same-origin upgrade and a first text message containing
`{session_id,token,scope}` before any binary image is accepted. The authentication
message is limited to 2048 bytes and a configurable 1–10 second deadline. The
random 256-bit token is kept server-side only as a session-salted SHA-256 digest,
compared in constant time and consumed under the session lock. Another owner,
another socket, a reconnect, cancellation, expiry or restart cannot replay it.

HTTP Host is exact, Origin is exact when present, cross-site Fetch Metadata is
refused, query strings are refused, and there is no CORS grant. WebSocket
upgrades require Origin. Responses include no-store, no-referrer, deny framing,
same-origin resource policy, restrictive external-script-only CSP, and a
camera/microphone-denying Permissions Policy. The capture HTML and JS themselves
are public local resources; session state and pixels are not public HTTP routes.

`screencast status --receipt ID` and `screencast cancel --receipt ID` enforce exact
scope and configuration fingerprint. Cancellation removes the pairing file
immediately and the running supervisor cancels the owner session within its
250ms mailbox poll interval. Stop/disconnect/expiry terminates a token, not just
the transport. A fresh approval is required to restart. Process restart closes
consumed durable grants and removes stale pairing files before readiness.

## Configuration example

Minimal intake-only configuration:

```json
{
  "channel": {
    "enabled": true,
    "bind": "127.0.0.1:8780",
    "allow_analysis": false,
    "allow_analysis_delivery": false,
    "allow_frame_artifacts": false
  }
}
```

All other channel limits use bounded defaults. Unknown fields fail validation.
To enable analysis, provide `vision` with an enabled `missy-vision::VisionConfig`,
an explicit model/backend, policy-allowed endpoint, credential environment
binding where required, prices, and its durable vision ledger. No vision backend
is inferred from an unrelated text provider. `analysis_prompt` is separately
configurable and bounded. Vision requests are NoTools visual observations, never
instructions to an agent. Output is redacted and bounded before status/delivery.

Delivery is separately opted into at **both** operator configuration and session
approval. The shipped adapter delivers to `artifact_root`, an owner-only existing
or newly created directory also covered by application policy write roots.
Files are written through `Policy::write_file`, which enforces path roots, byte
limits, safe atomic replacement and mode 0600. `max_artifact_files` defaults to
1000 and is bounded to 1–10000; filling it fails closed until explicit operator
cleanup. Frames are included only for `deliver_frame_artifacts`; analyses only
for `deliver_analysis`. No source image metadata is retained.

There is **no automatic Discord posting** in the shipped operator adapter.
Embedders can inject `EventSink` to implement an explicitly approved Discord
callback. Its target comes only from the exact approved Scope, not browser JSON;
the sink must disable mentions, preserve the untrusted-observation distinction,
perform durable idempotency on `request_id`, and must not execute observations.
This is an intentional safe difference from legacy implicit Discord callbacks.

## Bounds and provider governance

* Defaults: 20 sessions, 20 sockets (including unauthenticated handshakes), two
  concurrent image decode/analysis slots, 2MiB encoded frame size, four million
  pixels, 4096px maximum dimension, 3600 frame attempts/session, 15 minute TTL.
* Only actual PNG/JPEG decodes are accepted. Dimension headers are checked before
  decode, allocation limits are installed, pixels normalize to RGB, and a new PNG
  strips metadata. Sanitized output is byte-bounded too.
* Rate limiting precedes decode. Admitted invalid images count toward the frame
  limit. 32×32 grayscale change detection skips unchanged/low-difference content;
  a separate cooldown gates analysis/artifact emission.
* Frames are processed in socket order with no application frame queue. Global
  saturation returns `busy`. Browser capture waits for acknowledgment and limits
  transport buffering. Decode slots remain held even if a blocking decoder
  outlives async cancellation. There are no unbounded tasks or queues.
* Each admitted frame has stable `screencast:<session UUID>:<sequence>` request ID,
  exact Scope, parent approval receipt and sanitized-image SHA-256.
* `GovernedVisionAnalyzer` first calls `FrameGovernor::reserve`. The shipped
  SQLite authority binds parent receipt, request ID, image digest and prompt
  digest; duplicate IDs refuse dispatch, including uncertain attempts. It then
  calls the existing `VisionTools::analyze_png` with ToolContext. The existing
  vision policy client and durable pricing ledger remain the only provider and
  financial budget implementation. No second provider loop is introduced.
* Timeout, cancellation or crash can leave a durable reservation unresolved;
  that is deliberately not replayable. Successful responses must settle both the
  existing vision ledger and parent operation receipt. Provider-priced estimates
  are not a prepaid provider-enforced financial hard cap.
* Authority state has hard lifetime caps: 10000 grants and 100000 frame receipts.
  It refuses new work rather than silently discarding replay evidence. Operator
  archival/retention is explicit. Per-session in-memory status keeps one latest
  bounded analysis, not every raw image or unlimited historical results.

## Embedding and lifecycle

`ScreencastChannel::new(config, approval_verifier, analyzer, event_sink)` builds
the channel. `create_session(SessionApproval)` returns the one-time `Pairing`.
It deliberately has no Debug/Serialize implementation to avoid accidental token
logging. `status(owner_scope,id)`, `cancel_session(owner_scope,id)` and
`shutdown()` implement owner/lifecycle control. `router(channel)` exports the
axum router for an exact-authority loopback listener; `start(channel)` owns its
listener and returns `ServerHandle`, with `is_finished()` and async `stop()`.
Dropping the handle cancels sessions and aborts the listener task.

CLI `screencast_adapter::prepare(&Config)` returns optional `Prepared`.
`Prepared::start().await` binds and invalidates stale credentials **before** the
unified supervisor declares readiness. The returned `Running::run(stop)` owns
bounded mailbox polling and shutdown. No OS processes, service deployment or
desktop session are started or changed by these APIs.

## Tests and evidence

`cargo test -p missy-screencast` exercises actual loopback HTTP/WS fixtures:
origin/rebinding/query/CSRF denial, security headers, binary-before-auth refusal,
scope mismatch, token replay/concurrent connection/reconnect denial, owner locks,
capacity, invalid dimension/payload and rate gates, image analysis, duplicate
suppression, exact-scope artifact delivery, cancellation, expiry and restart.

Opt-in headless Chromium fixture:

```sh
MISSY_SCREENCAST_BROWSER_PYTHON=/path/to/playwright/venv/bin/python \
  CARGO_BUILD_JOBS=2 cargo test -p missy-screencast
```

The fixture replaces `getDisplayMedia` with a synthetic canvas stream before
loading the real page. It verifies zero capture calls before the click, one
synthetic call after the click, an actual authenticated WS frame/analysis,
empty browser storage, token input clearing, and stop/track cleanup. It never
selects or records a real screen. This is a protocol/UI fixture, not proof that a
human granted real browser permission or that a real provider was contacted.
CLI unit fixtures cover durable approvals, exact-scope cancellation, owner-only
pairing files, config binding, one-use parent frame receipts and symlink/mode
refusal. No live provider, account, camera, microphone or active desktop is used.
