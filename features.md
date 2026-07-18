# Missy — Feature Roadmap (24 candidates)

> **Implementation status (updated):** ✅ **F04**, **F11**, **F14** are fully
> implemented, tested, and documented on branch `feat/features-md-batch-1`
> (see `IMPLEMENTATION_STATUS.md`). The remaining 21 are scoped below; the
> Tier-1 new-core-technologies (F01–F03) are multi-week efforts and are
> intentionally not stubbed.


A grounded feature slate for Missy, derived from a full architecture pass over
`missy/` plus the validation-suite findings. Features are split into three
tiers:

- **New Core Technologies** — net-new capabilities that don't exist in any form
  today and would introduce a genuinely new subsystem/execution model (≥2 required).
- **Activate Implemented-but-Unwired Subsystems** — code that is *already
  written and unit-tested* but has zero production callers (the "known gaps"
  the validation suite documents). Highest ROI: the hard part is done.
- **Subsystem Enhancements** — extensions to subsystems that are already live.

Each item notes its **current state** so the delta is honest, not aspirational
hand-waving.

---

## Summary

| # | Feature | Tier | Current state |
|---|---------|------|---------------|
| F01 | Leyline P2P Agent Mesh | New core tech | Named in docs only; no code |
| F02 | Neuro-Symbolic Planning Kernel | New core tech | Not present (linear tool loop today) |
| F03 | On-Device Retrieval Engine (local embeddings + RAG) | New core tech | FTS5 keyword search only for chat memory |
| F04 | GraphMemoryStore query surface | Activate unwired | ✅ DONE — graph_query tool + `missy graph` CLI + opt-in ingestion |
| F05 | ModelRouter live wiring | Activate unwired | ✅ DONE — routed in single-turn path (opt-in) |
| F06 | HeartbeatRunner production wiring | Activate unwired | ✅ DONE — constructed+started in gateway_start (gated) |
| F07 | LandlockPolicy runtime bootstrap | Activate unwired | ✅ DONE — apply_landlock_if_enabled in gateway_start + `missy security landlock` |
| F08 | ContainerSandbox activation path | Activate unwired | Implemented, no bootstrap path |
| F09 | StructuredOutputRunner adoption | Activate unwired | ✅ DONE — structured benchmark judge (make_structured_llm_judge) |
| F10 | CondenserPipeline/CompactionManager wiring | Activate unwired | ✅ DONE — opt-in condenser pass in _build_context_messages |
| F11 | TrustScorer inspection + gating | Activate unwired | ✅ DONE — persisted + `missy tools trust` CLI |
| F12 | Semantic conversation memory (VectorMemoryStore) | Activate unwired | ✅ DONE — ConversationSemanticIndex + SleeptimeWorker indexing + `missy memory` CLI |
| F13 | PromptPatchManager organic proposal trigger | Activate unwired | ✅ DONE — FailureTracker-driven ERROR_AVOIDANCE proposals (opt-in) |
| F14 | `missy sessions clear` operator CLI | Enhancement | ✅ DONE — CLI + store helpers |
| F15 | Round-robin rotation for all providers | Enhancement | OpenAI-only today |
| F16 | Storyboard video orchestration tool | Enhancement | ✅ DONE — video_storyboard tool (generate→trim→concat→title) |
| F17 | MCP server authentication (OAuth/bearer) | Enhancement | Digest pinning only; no auth |
| F18 | Web TUI live log/audit streaming page | Enhancement | ✅ DONE — `/logs` page + `/api/v1/logs/tail` (redacted, live) |
| F19 | Global + multi-session budget ceilings | Enhancement | ✅ DONE — GlobalBudget + `missy budget` + runtime enforcement |
| F20 | Playbook → Skill auto-promotion (end-to-end) | Enhancement | ✅ DONE — write_skill_proposal + `missy skills promote` |
| F21 | LLM-judge benchmark dimension | Enhancement | ✅ DONE — optional judge_fn in BenchmarkScorer + make_llm_judge |
| F22 | Signed exportable audit bundles + SIEM sink | Enhancement | ✅ DONE (export bundle) — `missy audit export`/`verify-bundle`; SIEM sink deferred |
| F23 | Scheduled/continuous vision monitoring | Enhancement | ✅ DONE — VisionMonitor (capture→change-detect→alert w/ cooldown) |
| F24 | Persona A/B experiments + guardrail tuning | Enhancement | ✅ DONE — PersonaExperiment + `missy persona experiment` + runtime variant selection |

---

## Tier 1 — New Core Technologies

### F01. Leyline P2P Agent Mesh
**What.** A peer-to-peer federation layer letting multiple Missy instances form
a trust-scoped mesh that shares memory, skills, playbooks, and policy verdicts
without a central server. A node can delegate a subtask to a peer better
positioned to run it (e.g. the GPU box for video, a LAN node with a camera),
and query peers' memory/graph stores under per-peer capability policy.
**Why it's a new core.** Introduces a whole networking/identity/consensus
subsystem: peer discovery, an Ed25519-signed gossip protocol (reusing
`AgentIdentity`), CRDT-merged shared memory, and a distributed policy quorum so
one compromised node can't widen the mesh's capabilities. Nothing like this
exists — the docs reference a "Leyline P2P network" but there is no code.
**Approach.** New `missy/mesh/` package: `PeerRegistry` (Ed25519 peer identities
+ capability grants), `GossipTransport` (signed, `PolicyHTTPClient`-gated),
`SharedMemoryCRDT` layered over `SQLiteMemoryStore`, and a `mesh_delegate` tool
that routes through `SubAgentRunner` semantics to a remote peer. Every
cross-node action is a signed audit event chained into the same tamper-evident log.

### F02. Neuro-Symbolic Planning Kernel
**What.** Replace the current linear "call tool → observe → call tool" loop with
a planner that compiles a task into a **typed tool-call DAG**: nodes are tool
invocations with declared pre/post-conditions, edges are data/ordering
dependencies. The kernel then executes the DAG with speculative parallelism
(independent branches run concurrently, like `SubAgentRunner` but at the
tool-call granularity), verifies each node's post-condition before its
dependents run, and rolls back / re-plans on violation.
**Why it's a new core.** It's a genuinely new execution model — a symbolic
planner + verifier sitting above the LLM, turning tool use from an
autoregressive guess into a checked, parallelizable, resumable plan. It
subsumes `DoneCriteria`, `FailureTracker` strategy rotation, and checkpointing
into one principled layer.
**Approach.** New `missy/planning/`: `PlanCompiler` (LLM proposes a DAG in a
constrained schema, enforced by `StructuredOutputRunner` — see F09),
`PlanExecutor` (topological + speculative execution over the existing
`ToolRegistry`, respecting `MAX_CONCURRENT`), `ConditionChecker` (typed
pre/post assertions), and checkpoint integration so a partially-executed plan
resumes via `missy recover`.

### F03. On-Device Retrieval Engine (local embeddings + RAG)
**What.** A first-class local retrieval core: embed the workspace, uploaded
docs, and conversation history with an on-device embedding model, and serve
semantic recall + citation-grounded RAG to the agent — no cloud embedding
calls, matching Missy's secure-by-default, self-hosted posture.
**Why it's a new core.** Today semantic vectors (`VectorMemoryStore`/FAISS)
reach *only* the vision scene memory; general recall is FTS5 keyword match. A
real retrieval engine (chunking, on-device embeddings, hybrid dense+sparse
ranking, citation spans, incremental re-indexing) is a new subsystem, not a
tweak.
**Approach.** New `missy/retrieval/`: a pluggable local `Embedder`
(sentence-transformers / llama.cpp GGUF), a `HybridIndex` (FAISS dense +
existing FTS5 sparse, reciprocal-rank fused), a `rag_query` tool returning
answers with `source_span` citations, and a background re-indexer wired into
`SleeptimeWorker`. Ships behind a `[retrieval]` extra to keep the base install lean.

---

## Tier 2 — Activate Implemented-but-Unwired Subsystems

### F04. GraphMemoryStore query surface
`missy/memory/graph_store.py` is a full entity-relationship graph with
pattern matching — and **zero callers** in runtime/tools/CLI. Add a
`graph_query` agent tool and `missy graph query|add|show` CLI so the
implemented store is actually reachable; wire entity extraction into
`SleeptimeWorker` so the graph populates from conversations.

### F05. ModelRouter live wiring
`ModelRouter` (complexity-scored fast/primary/premium tier selection) is
implemented but **nothing in `AgentRuntime` calls `score_complexity()`/
`select_model()`** — only `SleeptimeWorker` reads `fast_model` directly. Wire
it into `_call_provider_with_fallback()` so simple turns use the cheap tier and
hard turns escalate, with the routing decision in the audit event and a
per-turn override.

### F06. HeartbeatRunner production wiring
`HeartbeatRunner` and its `heartbeat:` config block are fully built and
documented, but there are **zero construction sites** — editing
`heartbeat.enabled` does nothing. Construct/start it in `gateway_start()`
alongside the scheduler, gated on config, with `active_hours` respected and
runs budget-capped like every other `AgentConfig`.

### F07. LandlockPolicy runtime bootstrap
`LandlockPolicy` (Linux LSM FS enforcement) is fully implemented with **no
production caller** — `SEC-094` honestly flags this. Add an
`apply_landlock_from_config()` call in the agent/tool bootstrap gated on a
`security.landlock.enabled` flag, so kernel-enforced filesystem confinement
actually applies on supported kernels.

### F08. ContainerSandbox activation path
Same shape as Landlock (`SEC-090`): per-session Docker isolation is implemented
but has no bootstrap path. Wire `container.enabled: true` into tool execution
so shell/file tools can run inside a `--network=none`, resource-capped
container, with a clean fallback + honest status when Docker is unavailable.

### F09. StructuredOutputRunner adoption
Pydantic schema enforcement with retry-on-invalid is implemented and
unit-tested but has **zero callers**. Adopt it where structured output matters:
the planning kernel (F02), `ModelRouter` decisions, benchmark LLM-judge (F21),
and any tool that expects JSON — turning "hope the model returns valid JSON"
into a validated contract.

### F10. CondenserPipeline / CompactionManager wiring
The 4-stage compression pipeline (observation masking, amortized forgetting,
summarizing, windowing) and its orchestrator are built but **unused** —
`MemoryConsolidator` is the live path. Wire `CompactionManager` in as the
advanced compaction strategy behind a config flag and benchmark it against the
current consolidator on recall quality.

### F11. TrustScorer inspection + gating
`TrustScorer` updates a 0–1000 reliability score on every tool call but is
**not exposed anywhere**. Add `missy tools trust [name]` to inspect scores, a
Web TUI panel, and an opt-in policy that warns/soft-gates a tool whose trust
falls below threshold (mirroring provider benchmark gating).

### F12. Semantic conversation memory
`VectorMemoryStore` (FAISS) is only constructed inside the vision subsystem.
Wire it into the main conversational memory path so `memory_search` and context
injection can do semantic recall (paraphrase-matching) instead of FTS5 keyword
match — reusing F03's embedder.

### F13. PromptPatchManager organic proposal trigger
The patch approve/reject mechanics and `get_active_patches()` are wired, but
`propose()` — the only way a patch is created — has **no organic caller**. Add
a trigger that proposes a `TOOL_USAGE_HINT`/strategy patch when `FailureTracker`
or `Learnings` detect a repeated, correctable failure pattern, feeding the
existing human-approval workflow instead of requiring manual seeding.

---

## Tier 3 — Subsystem Enhancements

### F14. `missy sessions clear <id>` operator CLI
`clear_session_full()` (deletes turns **and** summaries) exists at the library
level, but the only way to invoke it is direct DB manipulation — the documented
over-refusal-spiral recovery still has no operator surface. Add
`missy sessions clear <id>` (and a Web TUI action) that calls it and signals the
gateway to drop in-memory session state.

### F15. Round-robin rotation for all multi-account providers
`key_rotation_strategy: round_robin` (per-account clients + independent rate
limiters) is implemented only for `OpenAIProvider`. Generalize the pattern into
a shared mixin so Anthropic and any OpenAI-compatible provider can balance
across multiple accounts, doubling effective throughput.

### F16. Storyboard video orchestration tool
Multi-scene videos today require the model to manually chain
`video_generate`×N → `video_edit` (trim/concat/caption). Add a `video_storyboard`
tool that takes a scene list (prompt/duration/caption/transition per scene) and
orchestrates generate→trim→crossfade→title→mux in one call, reusing the two
existing tools under the hood with progress events per scene.

### F17. MCP server authentication
MCP support has digest pinning but no auth — remote MCP servers behind
OAuth/bearer can't be used. Add per-server auth config (bearer token via
`vault://`, OAuth client-credentials) injected by `McpManager`, so authenticated
MCP servers are first-class.

### F18. Web TUI live log/audit streaming page
The run console already streams via SSE; extend the pattern to a `/logs` page
that tails the audit log and application log live (with the same redaction and
facet filters as the static `/audit` page), so operators watch activity in real
time without shelling in.

### F19. Global + multi-session budget ceilings
`max_spend_usd` is per-session. Add a global daily/monthly ceiling across all
sessions + scheduled jobs + proactive runs, with threshold alerts (e.g. a
Discord DM at 80%) and a hard stop — so autonomous background activity can't
collectively overspend even when each session is individually under cap.

### F20. Playbook → Skill auto-promotion (end-to-end)
`Playbook.get_promotable(threshold=3)` flags patterns but nothing consumes the
output — promotion stops at "flagged." Close the loop: materialize a promotable
pattern into a real `SKILL.md` / `BaseSkill` proposal routed through the
existing approval workflow, so proven tool sequences become reusable skills.

### F21. LLM-judge benchmark dimension
Benchmark correctness scoring is heuristic (exact/substring/numeric/token
overlap). Add an optional LLM-judge dimension (via `StructuredOutputRunner`)
that scores semantic correctness for open-ended tool outputs, feeding the same
composite + provider-gating pipeline — a stronger signal than string matching.

### F22. Signed exportable audit bundles + SIEM sink
The audit log is signed + hash-chained locally and exports OTLP spans. Add
`missy audit export --since ... --bundle` producing a portable, signature- and
chain-verifiable evidence bundle, plus a syslog/SIEM sink so the tamper-evident
trail integrates with external security tooling.

### F23. Scheduled / continuous vision monitoring
Vision is on-demand. Add a monitoring mode: a scheduled/proactive trigger that
periodically captures from a camera, runs change detection against
`SceneSession`, and raises an approval-gated alert + Discord notification on a
significant change (e.g. "someone at the door", "the print failed") — turning
the vision stack into a passive sensor.

### F24. Persona A/B experiments + guardrail tuning
Persona is a single static YAML. Add support for multiple persona variants with
per-channel/per-session assignment and lightweight A/B measurement (tone
adherence, refusal rate, task success from existing telemetry), so persona/
guardrail changes are evaluated on real outcomes with safe rollback via the
existing persona backup/rollback machinery.

---

## Notes on prioritization

- **Fastest wins:** Tier 2 (F04–F13) — the implementation exists and is tested;
  these are mostly wiring + a CLI/tool surface + an audit event. F04, F05, F06,
  F11, and F14 are each roughly a day.
- **Highest leverage:** F02 (Planning Kernel) and F03 (Retrieval Engine) —
  they upgrade the agent's core competence and several other features (F09, F12,
  F21) compose onto them.
- **Most differentiating:** F01 (Leyline mesh) — turns Missy from a single
  self-hosted agent into a self-hosted agent *fabric*, which nothing else in
  this space does with the same security posture.
