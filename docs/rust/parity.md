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
| MCP | Native pinned opt-in stdio JSON-RPC; HTTP/SSE transport not yet ported |
| CLI | Separate `missy-rs`, not a replacement symlink for legacy `missy` |
| HTTP API / browser console | Native authenticated loopback API and isolated console |
| Discord | Native policy-pinned Gateway/REST, allowlists, heartbeat and reconnect/resume; local wire fixture only, no live bot acceptance |
| Scheduler | Native durable explicitly scoped interval jobs, manual worker invocation; not all APScheduler syntax |
| Planning DAG | Native bounded validation/topological compilation; execution does not bypass the durable runtime |
| SKILL.md | Native bounded data-only discovery; no arbitrary embedded Python execution |
| Vault | Native ChaCha20-Poly1305 at-rest storage, private files and cross-process lock |
| Signed audit exports | Native Ed25519 export API with independently pinned verification key |
| Prompt/persona | Explicit configured system prompt and immutable continuation hash; legacy hatching/persona experiments not ported |
| Voice edge / Discord voice / STT / TTS | **Not ported**, no claim of hardware or protocol compatibility |
| Vision/camera/screencast/desktop/OBS/VTube | **Not ported**, no desktop session or camera operated by rewrite tests |
| Image/video/music media tools | **Not ported**; explicitly approved MCP can provide integrations, not automatic parity |
| OAuth/Codex/Claude CLI/acpx providers | **Not ported**, no credential scraping or inheritance from legacy state |
| Automatic prompt patches / code evolution | **Not ported**, no self-modification or implicit approvals |
| Mesh/CRDT/quorum | **Not ported**, no federation security claim |
| Graph/vector/semantic retrieval | **Not ported**, SQLite FTS is lexical retrieval, not neural embeddings |
| Nomad / Incus / repo evaluation tools | **Not ported**, no deployment credentials or infrastructure mutations |
| Landlock/container sandbox | **Not ported**, exact argv policy is not kernel containment |
| Full approval TUI / trust scoring | **Not ported**, denied capability stays denied rather than silently granting one |
| OTel/SIEM | **Not ported**, durable audit chain is not an external telemetry deployment |
| Legacy config/state migration | Explicitly rejected until reviewed migration exists; Python state remains untouched |

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
