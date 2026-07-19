# features.md — Implementation Status

Tracks which of the 24 candidates in `features.md` are actually implemented in
the tree (with tests + docs, no placeholders) versus scoped-only.

**Done so far: F02, F03, F04, F05, F06, F07, F08, F09, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24** (23 of 24). Remaining: **F01** (Leyline P2P Agent Mesh).

## ✅ Implemented — batch F02 (branch `feat/features-md-f02-planning`)

### F02 — Neuro-Symbolic Planning Kernel
A symbolic planner + verifier above the LLM, in the new `missy/planning/`
package — a genuinely new execution model, not the linear
call-tool→observe→call-tool loop:
- **`plan.py`** — the typed tool-call DAG model. `ToolNode` (tool + args +
  `depends_on` + pre/post `Condition`s), `Plan` with full validation (unique
  ids, resolvable deps, condition targets, **acyclicity** via Kahn's
  algorithm) and topological ordering. `${node.output[.key]}` argument
  references imply dependencies and resolve against upstream results
  (`resolve_args`, type-preserving for whole-value refs).
- **`conditions.py`** — `ConditionChecker`, the verifier half: typed
  assertions (`success`, `output_contains/equals/not_empty/matches/is_number`,
  with `node` targeting and `negate`), each returning `(ok, reason)`.
- **`executor.py`** — `PlanExecutor`: dependency-ordered execution over the
  real `ToolRegistry` with **speculative parallelism** (independent ready
  nodes run concurrently, capped at `MAX_CONCURRENT=3` like `SubAgentRunner`).
  Each node checks pre-conditions → runs the tool → verifies post-conditions
  before dependents become eligible; a failed/raising/violating node fails and
  its dependents are skipped as unreachable rather than run on bad inputs. A
  raising tool is contained. `resume_state` seeds already-completed nodes so an
  interrupted plan resumes without re-running them (the checkpoint hook).
- **`compiler.py`** — `PlanCompiler`: the single validation gate. Compiles a
  plan dict (deterministic path) or an injected `planner` callable's proposal
  (LLM in production, ideally via F09's `StructuredOutputRunner`), verifying
  every node's `tool` against the registered tool set and all DAG invariants
  up front. `plan_schema()` returns the constrained JSON schema a
  structured-output planner should target.
- **Tests:** `tests/planning/` (plan/validation/refs, conditions, executor
  incl. real concurrency + failure propagation + resume, compiler) — 56 tests,
  ~99% package coverage. Added to CI shard 1.

## ✅ Implemented — batch F03 (branch `feat/features-md-f03-retrieval`)

### F03 — On-Device Retrieval Engine (local embeddings + hybrid RAG)
A first-class, fully-offline retrieval core in the new `missy/retrieval/`
package — a genuine new subsystem, not a tweak of the vision-only
`VectorMemoryStore`:
- **Chunking** (`chunking.py`) — span-preserving splitter that biases cuts to
  paragraph/sentence boundaries (hard-splits oversized runs), with overlap, so
  every chunk carries the exact `(start, end)` offsets it occupies in its
  source document. This is what makes citations precise.
- **Embedders** (`embedder.py`) — pluggable `Embedder` protocol. Default
  `HashingEmbedder` is dependency-free (word uni/bi-grams + char 3–5 grams,
  signed feature hashing, sublinear TF, L2-normalized) so the base install
  stays offline and lean; `SentenceTransformerEmbedder` is used automatically
  when the `[retrieval]` extra is installed. `get_default_embedder()` degrades
  gracefully.
- **Hybrid index** (`hybrid_index.py`) — `DenseIndex` (FAISS `IndexFlatIP`
  when available, NumPy brute-force fallback otherwise; cosine via normalized
  inner product) + `BM25SparseIndex` (pure-Python Okapi BM25) fused with
  scale-free **reciprocal-rank fusion**, so dense paraphrase recall and sparse
  rare-keyword matching complement each other.
- **Engine** (`engine.py`) — `RetrievalEngine`: index/re-index (incremental —
  re-indexing a `doc_id` replaces its chunks), remove, query returning
  `RetrievalResult`s with a `source_span` citation, JSON+`.npy` persistence
  (durable across restarts; stale-dimension indexes ignored), and `stats()`.
- **Agent tool** (`missy/tools/builtin/rag_query.py`) — `rag_query` with
  `query`/`index_text`/`index_file`/`stats` actions, citation-grounded output,
  registered into the built-in tool set (`resolve_filesystem_targets` declares
  `index_file` paths to the filesystem policy engine).
- **CLI** — `missy retrieval index|query|stats|remove`.
- **Packaging** — new optional `[retrieval]` extra (FAISS +
  sentence-transformers); the core needs neither.
- **Tests:** `tests/retrieval/` (chunking, embedder, hybrid index, engine,
  rag_query tool) + `tests/cli/test_retrieval_cli.py` — 64 tests, ~95% package
  coverage. `tests/retrieval` added to CI shard 3.

## ✅ Implemented — batch 17 (branch `feat/features-md-batch-17`)

### F17 — MCP server authentication + HTTP(S) transport
- `McpClient` gains a real **Streamable-HTTP transport** alongside the existing
  stdio one: `McpClient(name, url=..., headers=...)` initializes an `httpx.Client`
  and performs the JSON-RPC `initialize`/`tools/list`/`tools/call` handshake over
  HTTP POST. Handles both `application/json` and `text/event-stream` (SSE)
  response bodies, captures and echoes the `MCP-Session-Id` header on subsequent
  requests, and surfaces `401`/`403` as a clear `RuntimeError("MCP HTTP auth
  failed …")`. `is_alive()`/`disconnect()` cover the HTTP client.
- **Authentication:** `manager._resolve_mcp_auth_headers(entry)` builds the auth
  header set from an `mcp.json` entry — `bearer_token` → `Authorization: Bearer …`,
  plus arbitrary `headers` — with every value passed through
  `_resolve_secret()` so `vault://KEY` / `$ENV` references resolve at connect
  time (plain values pass through unchanged; never raises on the hot path).
  Wired into both `connect_all()` and `_connect_new_servers_from_config()`, and
  preserved across `restart_server()` (a restarted HTTP server keeps its auth).
- **Tests:** `tests/mcp/test_http_transport.py` — 16 tests (HTTP connect/list/call,
  session-id capture+echo, auth-header install, 401→error, SSE parsing, disconnect,
  missing-transport error, auth resolution incl. vault, manager wiring). Existing
  MCP suite updated for the new `headers=` constructor/`add_server` contract
  (409 passing total).

## ✅ Implemented — batch 2 (branch `feat/features-md-batch-2`)

### F13 — PromptPatchManager organic proposal trigger
- `AgentRuntime._maybe_propose_error_patch()` proposes an ERROR_AVOIDANCE
  PromptPatch (status PROPOSED → human review via `missy patches`) when a tool
  crosses `FailureTracker`'s consecutive-failure threshold. Opt-in
  (`AgentConfig.prompt_patch_proposals_enabled`, default off), deduped per
  (tool, error-signature), never raises. Wired at the real failure-threshold
  site in the tool loop.
- **Tests:** `tests/agent/test_prompt_patch_proposals.py` — 8 tests.

### F20 — Playbook → Skill auto-promotion (end-to-end)
- `Playbook.write_skill_proposal()` materializes a promotable pattern into a
  real, discoverable SKILL.md draft; `Playbook.promote_to_skills()` promotes all
  eligible patterns and marks them (idempotent; `dry_run` supported).
- New `missy skills promote [--threshold] [--proposals-dir] [--dry-run]` CLI.
- Verified end-to-end: generated SKILL.md is parsed by the real `SkillDiscovery`.
- **Tests:** `tests/agent/test_playbook_promotion.py`,
  `tests/cli/test_skills_promote_cli.py` — 11 tests.

## ✅ Implemented — batch 1 (merged, PR #65)

### F14 — `missy sessions clear` operator CLI
- `SQLiteMemoryStore.clear_session_full()` now returns removed-row counts;
  new `count_session_turns()` / `session_exists()` helpers.
- New `missy sessions clear <id> [--yes]` CLI: name resolution, confirmation
  guard, count feedback, restart reminder.
- **Tests:** `tests/memory/test_sqlite_store_coverage.py` (store helpers),
  `tests/cli/test_sessions_clear.py` (full CLI flow) — 20 new tests.
- Closes the REGR2-004 gap (over-refusal-spiral recovery had no operator surface).

### F04 — GraphMemoryStore query surface + ingestion
- New read-only `graph_query` agent tool (`missy/tools/builtin/graph_tools.py`),
  registered in `_ALL_TOOL_CLASSES`.
- New `missy graph stats|query|entity|add-entity` CLI.
- Ingestion side: `SleeptimeWorker._ingest_graph_entities()` feeds processed
  turns into the graph; gated by `AgentConfig.graph_memory_enabled` (default
  off). End-to-end verified (ingest → `graph_query` retrieval).
- **Tests:** `tests/tools/test_graph_query_tool.py`, `tests/cli/test_graph_cli.py`,
  graph-ingestion tests in `tests/agent/test_sleeptime.py` — 33 new tests.

### F11 — TrustScorer persistence + `missy tools trust`
- `TrustScorer(persist_path=...)`: loads on init, atomic re-save per mutation;
  `persist_path=None` keeps the original pure-in-memory behaviour.
- `AgentRuntime` persists to `~/.missy/trust.json`.
- New `missy tools trust [name] [--threshold]` CLI (lists / flags LOW TRUST).
- **Tests:** `tests/security/test_trust_persistence.py`,
  `tests/cli/test_tools_trust_cli.py` — 22 new tests.

**Totals:** 3 features, ~75 new tests, all green; `ruff check` + `ruff format`
clean; every new CLI/tool smoke-tested against a real store.

## ⏳ Scoped, not yet implemented

Tier 2 (tractable follow-ons): **F05** ModelRouter wiring, **F06** Heartbeat
wiring, **F07** Landlock bootstrap, **F08** ContainerSandbox activation, **F09**
StructuredOutputRunner adoption, **F10** Condenser wiring, **F12** semantic
conversation memory, **F13** PromptPatch organic trigger.

Tier 3 enhancements: **F15–F24**.

Tier 1 new core technologies — **F01** Leyline P2P mesh, **F02** Neuro-Symbolic
Planning Kernel, **F03** On-Device Retrieval Engine — are multi-week subsystems
(new packages, protocols, external model/infra dependencies). They are
deliberately **not** stubbed: a placeholder mesh/planner/retriever would violate
the "no placeholders" bar and give false confidence. They remain fully scoped in
`features.md` for a dedicated effort.

## Why not all 24 at once

The goal was "no placeholders, fully documented, tests to the max." Meeting that
bar means each feature ships real code + real tests + accurate docs. The three
delivered here clear it end-to-end. Shipping 24 shallow stubs would fail the
same bar the earlier validation work in this repo exists to enforce (the suite's
recurring criterion is *"reports actual results rather than hallucinated
success"* — that applies to features as much as to test cases).
