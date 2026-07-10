# Build Status

Last updated: 2026-07-10 04:20 UTC

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
