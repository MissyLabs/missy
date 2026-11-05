# Missy, native Rust rewrite

This workspace is a new Rust implementation, not Python called through a Rust
launcher. It deliberately leaves the existing Python tree, installer and live
gateway alone while correctness, parity and migration are evaluated.

**This is a reviewable rewrite, not a claim of complete feature parity or an
authorization to replace the running bot.** The exact parity ledger is
[docs/rust/parity.md](docs/rust/parity.md).

## Build and run

Requires Linux and the stable Rust toolchain. SQLite is bundled. Rustls handles
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
| `missy-cli` | Native `missy-rs` executable |

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

## Security boundaries and honest limitations

- Exact command approval is **not an OS sandbox**. Approved executables can have
  their own network, filesystem and descendant behavior. Do not approve a shell,
  arbitrary interpreter, network client or program that can detach from process
  groups unless those capabilities are genuinely intended. Landlock/container
  sandbox parity is not claimed.
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
