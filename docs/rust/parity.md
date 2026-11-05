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
| MCP | Native pinned opt-in stdio JSON-RPC; HTTP/SSE transport still not ported |
| CLI | Separate `missy-rs`, not a replacement symlink for legacy `missy` |
| HTTP API / browser console | Native authenticated loopback API and isolated console |
| Discord | Native policy-pinned Gateway/REST, durable intake/resume/outbox and nonce reconciliation, scoped interactions; read-only probe command. Live bot acceptance still requires a dedicated authorized credential |
| Scheduler | Native durable explicitly scoped interval jobs, manual worker invocation; not all APScheduler syntax |
| Planning DAG | Native bounded validation/topological compilation; execution does not bypass the durable runtime |
| SKILL.md | Native bounded data-only discovery; no arbitrary embedded Python execution |
| Vault | Native ChaCha20-Poly1305 at-rest storage, private files and cross-process lock |
| Signed audit exports | Native Ed25519 export API with independently pinned verification key |
| Prompt/persona | Native scoped proposal/review/revision store, immutable prompt snapshots, CLI new-turn loading; hatching/A-B experiments not ported |
| Voice edge / STT / TTS | Native loopback edge protocol, explicit device pairing, mode enforcement, command Whisper/Piper adapters, fake-audio/ELF acceptance. Actual speech models and Discord voice not tested/ported |
| Vision/camera/screencast/native desktop | **Not ported**, no desktop session or camera operated by rewrite tests |
| OBS/VTube | Native policy-pinned typed WebSocket adapters, explicit request grants and mutation approvals; only simulated protocol acceptance |
| Image/video workflow | Native operator-workflow ComfyUI submit/history validation; no arbitrary graph or artifact-path grants. Other media/music tools not ported |
| OAuth/Claude CLI | Native explicitly approved chat-only CLI provider, NoTools runtime, no credential scraping/automatic login. Codex completion deliberately blocked without verified no-tools mode; acpx not ported |
| Automatic prompt patches / code evolution | Native scoped inert patch proposal/review, **no source auto-application or self-modification** |
| Mesh/CRDT/quorum | **Not ported**, no federation security claim |
| Graph/vector retrieval | Native scoped SQLite graph, UTF-8 citation indexing, BM25 + deterministic hashing lexical vectors/RRF. Pluggable Embedder, **not a neural semantic model** |
| Nomad / Incus | Native bounded typed adapters, resource/action allowlists and exact durable mutation approval. Mock endpoints only; no infrastructure changed. Repo-evaluation tools not ported |
| Landlock/container sandbox | Native child-only Landlock, real filesystem/TCP confinement acceptance on ABI8; hardened container argv builder only, not a deployed container service |
| Approval / trust | Native exact scoped single-use approvals with CLI approve/deny, durable runtime pause/resume. Advisory versioned trust; Rich approval TUI not ported |
| OTel/SIEM | **Not ported**, durable audit chain is not an external telemetry deployment |
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
