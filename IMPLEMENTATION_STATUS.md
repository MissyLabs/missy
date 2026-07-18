# features.md — Implementation Status

Tracks which of the 24 candidates in `features.md` are actually implemented in
the tree (with tests + docs, no placeholders) versus scoped-only.

## ✅ Implemented this pass (branch `feat/features-md-batch-1`)

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
