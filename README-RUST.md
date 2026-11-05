# Missy, native Rust rewrite

This workspace is a new Rust implementation, not Python called through a Rust
launcher. It deliberately leaves the existing Python tree, installer and live
gateway alone while correctness, parity and migration are evaluated.

**This is a reviewable rewrite, not a claim of complete feature parity or an
authorization to replace the running bot.** The exact parity ledger is
[docs/rust/parity.md](docs/rust/parity.md).

## Build and run

Requires Linux and the stable Rust toolchain (locked dependency floor: 1.89).
Final acceptance used 1.99.1; the floor is metadata, not a separate MSRV test.
SQLite is bundled. Rustls handles
HTTPS; no Python environment, OpenSSL SDK, Node runtime or cloud account is
required to build the native executable.

```sh
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo build --release -p missy-cli --locked
./target/release/missy-rs --config ./missy-rust.yaml init
```

The initializer creates a private file and refuses to overwrite an existing
one. Default configuration permits no network origins, filesystem roots, shell
commands or MCP servers. Configure an installed local model using
[examples/rust-config.yaml](examples/rust-config.yaml), or configure a supported
remote provider using an environment-variable credential reference. Never put
credentials in checked-in YAML, arguments or chat messages.

```sh
./target/release/missy-rs --config ./missy-rust.yaml config-check
./target/release/missy-rs --config ./missy-rust.yaml doctor
./target/release/missy-rs --config ./missy-rust.yaml ask 'Hello' --session demo
./target/release/missy-rs --config ./missy-rust.yaml run --session demo
./target/release/missy-rs --config ./missy-rust.yaml history --session demo
./target/release/missy-rs --config ./missy-rust.yaml recover --session demo
```

Relative state and filesystem roots resolve against the config file's directory,
not whichever shell directory happened to launch the bot. The new schema rejects
unknown keys, including legacy Python configuration keys. Migration must be
explicit; accepting an unrecognized policy key as an empty/default value is not
compatibility.

## Crates

| Crate | Responsibility |
|---|---|
| `missy-core` | Typed request scope, protocol, configuration and provider/tool interfaces |
| `missy-policy` | Descriptor-relative filesystem access, exact argv policy, bounded pinned outbound transports |
| `missy-store` | Private SQLite WAL state, sessions, leases/fencing, checkpoints, tool ledger, memory and audit chain |
| `missy-providers` | OpenAI-compatible, Anthropic and Ollama adapters; fallback, validation and circuit breaking |
| `missy-tools` | Files, commands, HTTP, scoped memory and explicitly enabled pinned stdio MCP |
| `missy-runtime` | Durable bounded agent loop, continuation, cancellation, context framing and request deduplication |
| `missy-gateway` | Authenticated loopback HTTP API and browser console; optional channel transport |
| `missy-automation` | Durable scoped scheduling, bounded DAG validation and data-only skill discovery |
| `missy-security` | Private ChaCha20-Poly1305 vault and pinned-key Ed25519 export verification |
| `missy-cli` | Native `missy-rs` executable and scope-preserving extension composition |
| `missy-migrate` | Explicit offline copy-only legacy configuration/state migration |
| `missy-retrieval` | Scoped graph and citation indexing, BM25/hashing-feature hybrid retrieval |
| `missy-governance` | Exact single-use approvals, advisory trust and operator-reviewed prompt revisions |
| `missy-integrations` | Typed policy-bound Nomad, Incus, ComfyUI, OBS and VTube adapters |
| `missy-provider-process` | Explicitly approved chat-only CLI provider; no native provider-tool bypass |
| `missy-voice` | Loopback voice edge protocol, explicit pairing and trusted speech-command adapters |
| `missy-sandbox` | Opt-in child-only Landlock and hardened container command construction |
| `missy-mcp-http` | Pinned Streamable HTTP and persistent SSE MCP through network policy |
| `missy-desktop` | Approval-gated native X11, AT-SPI and isolated CDP controls |
| `missy-media` | Typed pinned FFmpeg editing, image transforms and file-based vision |
| `missy-embedding` | Explicit local neural embedding adapter with CPU model acceptance |
| `missy-mesh` | Signed scope-bound peers, replay protection, memory and budgeted delegation |
| `missy-operations` | Authenticated webhook, diagnostics, metrics, literal workflows and export APIs |
| `missy-schedule-text` | Strict human/cron grammar, IANA timezones and deterministic DST rules |

The optional `scripts/test-rust-console.py` acceptance harness uses Python
Playwright solely to drive Chromium against an isolated compiled Rust server
and a fake provider. It is not required to run Missy. CI runs it separately from
the native tests. An admitted HTTP request returns its real turn ID immediately;
the console polls that scoped ID to completion. Disconnecting the browser does
not cancel admitted work, and cancellation is explicit.

## Session and continuation guarantees

1. Every state lookup is scoped by tenant, principal, channel and session. The
   HTTP server derives tenant and principal from its configuration, not the model
   or request JSON. Local CLI scope flags are operator-level controls, not an
   authentication mechanism for untrusted local users.
2. A scoped request ID is bound to the original payload. Redelivery returns the
   existing turn, not another execution. Reusing the ID for different content is
   rejected. Deduplication survives process restarts.
3. A session has a cross-process fenced lease. Expired or cancelled owners cannot
   publish new state or self-resurrect. Opening another worker does not sweep
   healthy work into an interrupted state.
4. Tool intents are durable before dispatch, and outcomes are persisted before
   subsequent generation. Recovery with an unresolved operation halts generation.
   Merely rewording the same mutation under a fresh tool ID must not evade this.
5. Continuation keeps persisted counters, consumed budgets and configuration
   lineage. A changed system prompt, provider configuration or tool catalog is
   not silently substituted into an old turn.
6. Current instructions remain distinct from inert historical requests. Memory,
   history and tool output retain provenance and are never promoted to system
   instructions. This framing is defense in depth, not a proof that a model can
   never be influenced by malicious text. Policy remains the action boundary.

See [docs/rust/continuation-design.md](docs/rust/continuation-design.md) for the
comparison with the original Python runtime and Odin's durability design.
The focused final comparison and regression evidence are in
[docs/rust/odin-comparison.md](docs/rust/odin-comparison.md).

## Security boundaries and honest limitations

- Exact command approval alone is **not an OS sandbox**. Approved executables can have
  their own network, filesystem and descendant behavior. Do not approve a shell,
  arbitrary interpreter, network client or program that can detach from process
  groups unless those capabilities are genuinely intended. Optional child-only
  Landlock adds kernel filesystem and TCP restrictions on supported kernels;
  TCP restriction is not total denial of UDP/Unix socket communication. The
  container feature constructs hardened argv but does not claim a deployed or
  hardware-accepted container runtime.
- HTTP redirects and ambient proxies are not followed. Origins include ports;
  private networking additionally requires explicit opt-in. DNS answers are
  checked and pinned for the connection rather than checked once then resolved
  again. A private-network opt-in is not itself a host allowlist.
- MCP command execution is separately opt-in and exact-argv approved. A manifest
  digest detects advertised changes; it cannot make a malicious trusted MCP
  implementation truthful. The protocol has no atomic call-if-manifest-digest
  primitive.
- Memory is scoped, provenance-labeled and untrusted. No global fallback,
  cross-user auto-learning or automatic promotion of retrieved instructions.
- Token and spend controls bound locally observable use. Provider billing after
  a timeout can be unknowable. A post-response spend check is not a guaranteed
  prepaid financial ceiling. Unknown costs must not be reported as free.
- The SQLite audit chain is tamper-evident against partial edits, not protection
  against an attacker rewriting the entire database. Export signatures are a
  separate API; verification requires an independently pinned public key.
- Vault encryption protects data at rest, not against the same account reading
  both key and ciphertext. Back up the key securely; losing it does not trigger
  key regeneration over existing ciphertext. The CLI accepts secret values on
  stdin and never offers an accidental plaintext `get` command.
- The API is a local operator service with one configured bearer principal.
  It is not a public multi-tenant identity provider. Use a real authenticated
  reverse proxy and explicit design review before remote exposure.
- Model/provider fixtures validate wire protocols without real credentials.
  Passing those tests is not an assertion that a live provider or Discord bot
  has been exercised.

## Safe rollout

Build and test in a fresh checkout with temporary state. Run a side-by-side
instance with a distinct config, state path, port and bot credential. Review the
parity ledger and recovery behavior before enabling tools. Never point this
build at a production Python `.missy` or reuse a live bot token concurrently.
There is no automatic service restart, installer replacement or destructive
configuration migration in this branch.

## Native extension commands

All optional files are selected under `extensions` in the main config and must
be private operator-owned files. Defaults remain disabled.

```yaml
extensions:
  retrieval_enabled: true
  approvals_config: ./approvals.yaml
  integrations_config: ./integrations.yaml
  voice_config: ./voice.yaml
  sandbox_config: ./sandbox.yaml
  # http_mcp_config: ./http-mcp.yaml
  # desktop_config: ./desktop.yaml
  # media_config: ./media.yaml
  # embedding_config: ./embedding.yaml
  # mesh_config: ./mesh.json
  # operations_config: ./operations.yaml
  # cli_provider_config: ./chat-only-provider.yaml
```

Extension schemas are documented in the individual crate READMEs. Enabling a CLI
provider selects a structurally NoTools runtime. Voice SafeChat also selects a
separate NoTools runtime, never a prompt asking the model not to execute tools.
HTTP MCP, desktop and media mutations require the same durable outer approval
gate; every desktop tool, including reads, must be listed explicitly. Desktop
configuration never discovers the operator's ambient display or accessibility
bus. Media editing requires explicitly pinned FFmpeg/FFprobe, and native
acceptance requires `ffmpeg`, `xvfb`, `xauth` and `dbus` packages on Linux.

The optional local embedding service is separately installed and explicitly
configured. Default retrieval remains deterministic lexical hashing, not a
pretend semantic model. See `crates/missy-embedding/ACCEPTANCE.md` for actual
CPU inference evidence and the documented adversarial-quotation limitation.
Integration mutations require explicit service/action/resource grants **and**
the durable outer approval gate to list every mutation tool. Approval never
widens the filesystem/network/command policy.

```sh
missy-rs approvals list --session demo
missy-rs approvals approve REQUEST_ID --binding-hash EXPECTED_HASH --operator OPERATOR_ID --session demo
missy-rs resume TURN_ID --session demo
missy-rs retrieval ingest handbook /approved/documents/handbook.txt --session demo
missy-rs retrieval query 'repair procedure' --session demo
missy-rs discord-probe
missy-rs voice devices
missy-rs schedule next 'daily at 09:00' --timezone America/New_York
missy-rs schedule add-text 'Check the approved service' 'every monday at 08:00' --timezone America/New_York --session maintenance
missy-rs schedule run-due --session maintenance
missy-rs workflow /private/workflow.yaml --session maintenance
missy-rs mesh status
missy-rs operations
```

`operations` is an explicitly started authenticated loopback webhook/metrics
service. Its trusted scope comes from private configuration, not incoming JSON.
It uses dedicated Missy HMAC/read keys, never another service's credentials.
`mesh serve` requires a private explicit peer configuration and finite stricter
local runtime limits; it never auto-discovers or trusts peers. Neither command
is automatically installed, started or exposed remotely.

Workflows are bounded literal prompt DAGs, not unconstrained autonomous forks.
Each node uses the existing scoped runtime and stable durable request ID.
Aggregate limits gate admission, not externally reserve billing. Scheduler
workers are explicitly invoked, never silently started by opening the database.

`discord-probe` performs only authenticated read requests. It does not open a
Gateway, register commands or send messages. Full live acceptance requires a
dedicated authorized Missy credential and an isolated test guild/channel. Never
reuse another running bot's token just because it happens to be in the process
environment. The current build does not claim live Discord acceptance.

### Offline migration, never an in-place experiment

Create a consistent **backup** of Python Missy's database first, using SQLite's
backup operation. The native migrator refuses live WAL/SHM/journal companions,
implicit paths, `.missy` directories, unsafe source files and missing principal
mapping. Each legacy session must map explicitly to exactly one new Scope.

```sh
missy-rs migrate plan --legacy-config /private/backup/config.yaml --legacy-database /private/backup/memory.db --scope-mapping /private/backup/scopes.json
missy-rs migrate apply --legacy-config /private/backup/config.yaml --legacy-database /private/backup/memory.db --scope-mapping /private/backup/scopes.json --destination /private/missy-rust-import --confirm
```

The plan is credential-redacted and rejects grants it cannot preserve safely.
Apply rechecks source hashes, builds a private staging tree, verifies it and
publishes with atomic no-replace rename. It never overwrites the source or an
existing destination. Historical messages and summaries become scoped inert
memory with provenance, not new instructions. Unscoped learnings and all old
incomplete checkpoints are quarantined rather than promoted globally or replayed.
This is intentionally not a promise that old interrupted external side effects
can be reconstructed from missing receipts.
