# Rust compatibility ledger

This ledger separates implemented native capabilities from retained Python
features. Existing Python tests and its README are not Rust coverage evidence.
Per-task validation results belong in `docs/rust/validation.md`.

| Legacy surface | Rust disposition |
|---|---|
| Agent tool loop and continuation | Native durable loop, strict snapshots, bounded continuation; no Python bridge |
| Sessions/history/recovery | Native SQLite, scope isolation, idempotent admission, expiring fenced leases |
| Personal memory | Native explicit scoped CRUD and FTS; no unsafe global learning fallback |
| OpenAI / Anthropic / Ollama | Native HTTP adapters, strict protocol validation and usage accounting |
| Provider fallback / health | Native bounded failover and circuit breakers; uncertain billing is not free |
| Shell/files/web | Native default-deny builtins, exact argv and descriptor-relative path handling |
| MCP | Native pinned opt-in stdio and policy-bound HTTP Streamable/SSE JSON-RPC; exact initialized session/catalog; HTTP tool calls require durable approval in CLI |
| CLI | Separate `missy-rs`, not a replacement symlink for legacy `missy` |
| HTTP API / browser console | Native authenticated loopback API and isolated console |
| Discord | Native policy-pinned Gateway/REST, durable intake/resume/outbox and nonce reconciliation, scoped interactions; read-only probe command. Live bot acceptance still requires a dedicated authorized credential |
| Scheduler | Native durable scoped interval, daily, weekly, five-field cron and relative one-shot jobs; IANA timezone/DST policy; explicit worker invocation, not all APScheduler syntax |
| Planning DAG | Native bounded topological validation plus literal workflow execution through ordinary scoped Runtime turns; persisted request IDs and aggregate admission bounds |
| SKILL.md | Native bounded data-only discovery; no arbitrary embedded Python execution |
| Vault | Native ChaCha20-Poly1305 at-rest storage, private files and cross-process lock |
| Signed audit exports | Native Ed25519 export API with independently pinned verification key |
| Prompt/persona | Native scoped proposal/review/revision store, immutable prompt snapshots, CLI new-turn loading; hatching/A-B experiments not ported |
| Voice edge / STT / TTS | Native loopback edge protocol, explicit device pairing, mode enforcement, command Whisper/Piper adapters, fake-audio/ELF acceptance. Actual speech models and Discord voice not tested/ported |
| Vision/camera/screencast/native desktop | Native explicitly allowlisted X11, AT-SPI and isolated CDP read/screenshot/input; all CLI desktop tools require exact durable approval. Sysfs camera discovery only, file-based capture. No real camera recording, screencast, active desktop or production browser operated |
| OBS/VTube | Native policy-pinned typed WebSocket adapters, explicit request grants and mutation approvals; only simulated protocol acceptance |
| Image/video workflow | Native approved ComfyUI workflow submission plus pinned FFmpeg trim/concat/overlay/transcode/frame/audio/storyboard and bounded image transforms in private child confinement. Native music composition and hardware recording not ported |
| OAuth/Claude CLI | Native explicitly approved chat-only CLI provider, NoTools runtime, no credential scraping/automatic login. Codex completion deliberately blocked without verified no-tools mode; acpx not ported |
| Automatic prompt patches / code evolution | Native scoped inert patch proposal/review, **no source auto-application or self-modification** |
| Mesh/CRDT/quorum | Native Ed25519 recipient/scope-bound envelopes, durable replay/receipts, explicit peer capability grants, scoped LWW memory and quorum primitives; delegated tasks use stricter locally budgeted Runtime. Only isolated peer acceptance, not a production federation claim |
| Graph/vector retrieval | Native scoped SQLite graph, exact UTF-8 citations, BM25/vector RRF. Default deterministic lexical vectors; explicit local neural embedding adapter, actually tested with pinned MiniLM weights on CPU. No automatic remote model or credential discovery |
| Nomad / Incus | Native bounded typed adapters, resource/action allowlists and exact durable mutation approval. Mock endpoints only; no infrastructure changed. Repo-evaluation tools not ported |
| Landlock/container sandbox | Native child-only Landlock, real filesystem/TCP confinement acceptance on ABI8; hardened container argv builder only, not a deployed container service |
| Approval / trust | Native exact scoped single-use approvals with CLI approve/deny, durable runtime pause/resume. Advisory versioned trust; Rich approval TUI not ported |
| OTel/SIEM | Native authenticated local diagnostics/Prometheus pull, bounded event bus, proper OTLP/HTTP JSON exporter and signed summary export APIs. Export APIs are not automatic runtime instrumentation or an external telemetry deployment |
| Webhook automation | Native dedicated HMAC/timestamp authenticated loopback intake, fixed trusted scope, body limits and durable fail-closed replay; no principal/tool overrides from payload |
| Legacy config/state migration | Native offline plan/apply, immutable backup sources, explicit one-to-one Scope mapping, separate atomic destination. Historical messages imported inert/scoped, unsafe global learning and incomplete checkpoints quarantined, never replayed |

## Promotion gates

- Complete native build, formatting, lint, tests and dependency audit.
- Process-level restart/deduplication and multi-tool acceptance through actual
  provider HTTP, not only in-memory fake trait calls.
- Crash interruption/uncertain outcome tests and expired-owner fencing tests.
- Cross-user/channel/session canary isolation, including direct IDs and memory.
- Provider and Discord live canary with dedicated credentials and no production
  mutation, followed by an operator-approved gradual rollout.
- Port or explicitly retire every required integration before claiming a full
  drop-in replacement. A compiler does not certify feature parity.
