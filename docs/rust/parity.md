# Rust compatibility ledger

This ledger separates implemented native capabilities from retained Python
features. Existing Python tests and its README are not Rust coverage evidence.
Per-task validation results belong in `docs/rust/validation.md`.

| Legacy surface | Rust disposition |
|---|---|
| Agent tool loop and continuation | Native durable loop, strict snapshots, bounded continuation; no Python bridge |
| Sessions/history/recovery | Native SQLite, scope isolation, idempotent admission, expiring fenced leases |
| Personal memory | Separate explicitly enabled tenant+user namespace for cross-session/channel recall; exact-scope session memory unchanged. Principal-local FTS statistics, no global fallback |
| Historical recall | Exact-scope bounded history/checkpoint/tool/raw-message search, durable immutable receipt describe/UTF-8 expand and separately budgeted NoTools context shunt |
| OpenAI / Anthropic / Ollama | Native HTTP adapters, strict protocol validation and usage accounting |
| Provider fallback / health | Native bounded failover and circuit breakers; uncertain billing is not free |
| Shell/files/web | Native default-deny builtins, exact argv and descriptor-relative path handling |
| Native plugins/provider probes | Model-callable pending proposals, exact-hash operator review/activation, new-runtime-only native plugin execution; explicit no-replay bounded provider benchmark |
| MCP | Native pinned opt-in stdio and policy-bound HTTP Streamable/SSE JSON-RPC; exact initialized session/catalog; HTTP tool calls require durable approval in CLI |
| CLI | Separate `missy-rs`, not a replacement symlink for legacy `missy` |
| HTTP API / browser console | Native authenticated loopback API and isolated console |
| Discord | Native policy-pinned Gateway/REST, durable intake/resume/outbox and nonce reconciliation, scoped interactions; read-only probe command. Live bot acceptance still requires a dedicated authorized credential |
| Scheduler | Native durable scoped interval, daily, weekly, five-field cron and relative one-shot jobs; IANA timezone/DST policy; explicit worker invocation, not all APScheduler syntax |
| Planning DAG / subagents | Native literal workflows plus parallel provider-selected agent DAGs, validated parent/depth, private durable root receipts, immutable contexts, cumulative reservations and cancellation. No model-authored capability grants |
| SKILL.md | Native bounded data-only discovery; no arbitrary embedded Python execution |
| Vault | Native ChaCha20-Poly1305 at-rest storage, private files and cross-process lock |
| Signed audit exports | Native Ed25519 export API with independently pinned verification key |
| Prompt/persona | Native reviewed prompt snapshots, inert first-run hatching proposals, deterministic persistent A/B assignments and deduplicated untrusted outcomes; human-selected promotion still requires governance review |
| Voice edge / STT / TTS / Discord voice | Native edge pairing and speech command adapters plus shared authenticated Discord Gateway voice negotiation, policy-bound WS, pinned UDP/AEAD/replay, official native DAVE MLS and Opus. Modes select real separate NoTools runtime. Synthetic/loopback and real native crypto acceptance; no live Discord or physical audio acceptance |
| Vision/camera/screencast/native desktop | Actual sanitized multimodal image payloads and structured scenes with separate reservations; approved V4L2 photo/clip and isolated X11 screencast with bounded capture/full decode. Synthetic providers/fresh Xvfb, no hardware/active desktop. Explicitly allowlisted X11/AT-SPI/CDP |
| OBS/VTube | Native policy-pinned typed WebSocket adapters, explicit request grants and mutation approvals; only simulated protocol acceptance |
| Image/video/music workflow | Approved ComfyUI workflows and authorized artifact publication; native pinned FFmpeg editing including audio mux/mix/loop, captions, speed and crossfade. Actual score-to-WAV/MIDI composition and ACE-Step singing backend adapter, no fabricated singing or playback |
| OAuth/Claude CLI / ACP | Native explicitly approved chat-only CLI and pinned Claude ACP adapter, structural no-tools options, denied permission requests and child cleanup. No credential scraping or automatic login. Codex completion deliberately blocked without verified complete no-tools controls |
| Reviewed code evolution | Native explicit operator apply/rollback with exact plan/policy/HEAD pins, real Git commits, durable recovery journal and no automatic replay after uncertainty. Restricted tracked-text patches and sanitized Git configuration, not unrestricted self-modification or deployment |
| Mesh/CRDT/quorum | Native Ed25519 recipient/scope-bound envelopes, durable replay/receipts, explicit peer capability grants, scoped LWW memory and quorum primitives; delegated tasks use stricter locally budgeted Runtime. Only isolated peer acceptance, not a production federation claim |
| Graph/vector retrieval | Native scoped SQLite graph, exact UTF-8 citations, BM25/vector RRF. Default deterministic lexical vectors; explicit local neural embedding adapter, actually tested with pinned MiniLM weights on CPU. No automatic remote model or credential discovery |
| Nomad / Incus / RepoEval | Native typed resource/operation grants, exact reviewed bodies/argv and durable mutation approvals. Foundry project-scoped list/plan/start/status/compare/artifacts/cancel/report/snapshot with operator workload pins and scoped credential-bound plans. Mock acceptance only; no live infrastructure mutation |
| Landlock/container sandbox | Native child-only Landlock, real filesystem/TCP confinement acceptance on ABI8; hardened container argv builder only, not a deployed container service |
| Approval / trust | Native exact scoped single-use approvals with CLI approve/deny, durable runtime pause/resume. Advisory versioned trust; Rich approval TUI not ported |
| OTel/SIEM | Native payload-free automatic turn/provider/tool/settlement instrumentation, bounded asynchronous policy-bound OTLP worker, explicit drop/failure metrics and awaited CLI final flush. Authenticated diagnostics/Prometheus and signed export APIs; no production collector deployed |
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
