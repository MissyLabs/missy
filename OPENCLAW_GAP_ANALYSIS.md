# OpenClaw Gap Analysis

Last updated: 2026-07-10 08:40 UTC

## Current Focus (2026-07-10 session)

Primary focus shifted to the `~/missy-loops/prompt.md` validation-harness
overhaul: closing the acpx delegate's routing/security gaps (FX-A
through FX-G, all complete this session) and a same-pattern security
sweep across Discord ingress and code-evolution approval that found and
fixed four independent unauthenticated authorization-bypass
vulnerabilities. Full detail in `BUILD_STATUS.md`/`AUDIT_SECURITY.md`.

### Implemented this session

- **acpx delegate routing (FX-A)**: zero native tools enforced via
  verified `--allowed-tools ""` semantics (checked against the pinned
  acpx@0.3.1 source, not just docs), fail-closed
  `--non-interactive-permissions deny`, config-can't-override
  sanitization of security flags, isolated sandbox cwd, a versioned
  delegation envelope replacing bare `[System]:` text, explicit
  current-turn structural boundary, and defensive stripping of leaked
  transcript markers with fail-closed behavior on total fabrication.
- **Production memory backend (FX-B / SR-3.1, substantially)**:
  `AgentRuntime` now uses the real `SQLiteMemoryStore` instead of a JSON
  compatibility store missing required methods — this was silently
  breaking Discord turn persistence, compaction, and large-content
  retrieval simultaneously from one root cause.
- **Code-evolution approval authentication (partial SR-1.2/1.3)**: the
  agent-facing tool and the Discord reaction handler can no longer
  approve/apply/rollback evolutions; only the human-operator `missy
  evolve` CLI can. `CodeEvolutionManager` itself still performs no
  authentication of its own — not a complete fix, see `AUDIT_SECURITY.md`.
- **Discord DM pairing authentication (partial SR-1.12)**: in-band
  `!pair accept`/`!pair deny` DM commands can no longer approve/deny a
  pairing (previously any unpaired stranger could self-approve). No
  authenticated approval surface has been wired to replace it yet —
  pairing currently has no working approval path at all (task #12).
- **Uniform Discord ingress authorization (partial SR-1.13)**: both the
  `MESSAGE_CREATE` special-command path (voice/image/screencast) and
  the `INTERACTION_CREATE` slash-command path (`/ask` etc., which
  previously had *no* authorization check at all) now run the same
  DM/guild policy gate before any dispatch. `/ask` sessions are now
  scoped per invoking user (previously one shared `session_id="discord"`
  for every user). Reactions beyond the two already-fixed flows and
  attachment gates were not independently re-audited.
- **Browser diagnostics (FX-F bullet 1)**: tool-absence vs.
  installation-failure vs. sandbox/kernel-launch-failure vs. real
  interaction error are now distinguishable, with remediation text that
  never suggests weakening sandboxing.
- **Timeout safety (FX-G, partial)**: safe upper bound on acpx timeout
  config; explicit "outcome unknown, verify before retry, make retries
  idempotent" messaging. Process-group cleanup on timeout attempted and
  reverted (needs a test-mock migration, task #17).
- **Memory ID grounding (FX-C)**: `memory_describe`/`memory_expand` now
  distinguish a lookup exception from a genuinely missing record.
  Incus `list`/`network(list)` tools confirmed and locked in with tests
  as deterministic JSON passthroughs (no tool-layer fabrication
  possible); an explicit anti-fabrication rule added to the acpx
  envelope for the model layer, where the harness's observed "invented
  lo network" fabrication actually originates.

### Known gaps / explicitly deferred this session

- FX-A bullet 6 (live end-to-end proof across tool categories with a
  real delegate invocation) — not attempted; this is the single
  biggest remaining gap blocking confident re-validation of the 89-case
  backlog.
- FX-F bullets 2/4 (disposable browser-test environment + WB/XT rerun)
  — this dev sandbox cannot launch a browser at all (no playwright
  installed), confirmed live. Task #16.
- SR-1.1, SR-1.4 through SR-1.11 (except the partial 1.12/1.13 above),
  SR-2.x, SR-3.2 through SR-3.5, SR-4.x — not started this session.
- `allowed_roles` Discord guild-policy field is documented and parsed
  but never enforced (task #15) — found during the SR-1.13 audit, not
  itself one of the four confirmed live vulnerabilities (no live
  exploit path demonstrated, just a silently-no-op config knob).
- Pre-existing, unrelated `CameraDiscovery` cache-TTL test flake (task
  #11).
- Full 89-case tool-specific validation backlog not yet re-run.

## Prior Focus (tool intelligence overhaul, preserved for history)

Primary focus is **tool usage and tool intelligence overhaul**. OpenClaw-style
parity here means Missy should observe repeated work, propose structured tools
when safe, store them with provenance and lifecycle metadata, benchmark them
across providers, reconcile benchmark evidence into reviewable candidate
records, gate provider access based on evidence, expose reviewable operator
diagnostics, and load approved tools only through explicit policy-gated runtime
bindings.

## Tool Intelligence Capability Status

| Capability | Status | Notes |
|---|---|---|
| Frequent-request detection | in place | `RequestTracker` records completed turns and surfaces repeated normalized patterns. |
| Runtime request-tracker wiring | in place | `AgentRuntime._track_request()` records completed user turns with tool calls and provider metadata. |
| Safe tool-candidate generation | in place | `CandidateGenerator` is opt-in and creates proposed candidates only. Shell proposals remain blocked unless explicitly configured. |
| Candidate storage metadata | improved | `CandidateStore` stores schema, permissions, provenance, examples, owner, version, lifecycle state, notes, benchmark summaries, provider flags, implementation metadata, and tags. |
| Candidate lifecycle enforcement | in place | Store-level transition matrix rejects skipped gates and disabled-candidate resurrection, with denied audit events. |
| Benchmark harness | in place | Direct and LLM-mediated benchmark runners score correctness, latency, cost, reliability, safety, schema adherence, tool-call quality, and failure behavior. |
| Benchmark-to-candidate reconciliation | in place | `CandidateBenchmarkReconciler` imports aggregate benchmark data into candidate summaries and provider flags without approving or enabling tools. |
| Provider-aware tool gating | in place | `ToolProviderGate` combines benchmark summaries and operator overrides. Runtime gating is opt-in. |
| Provider fallback recommendation | partial | CLI can recommend the best enabled provider; runtime does not yet surface recommendations when a tool is gated off. |
| CLI diagnostics | in place | Candidate list/show/import-benchmarks/approve/enable/deny, request stats, benchmark run/results/compare, provider status/enable/disable/clear/recommend. |
| Web/API controls | in place | REST candidate list/show endpoints and safe controls for import-benchmarks/approve/enable/deny reuse store-level gates and audit `web.control` events. |
| Runtime loading of enabled candidates | partial | `CandidateRuntimeLoader` can opt-in load enabled candidates with explicit `delegated_tool` implementation metadata after schema/provenance/permission/provider checks. No arbitrary generated code is loaded. |

## OpenClaw Pattern Status

| ID | Pattern | Status | Notes |
|---|---|---|---|
| A1 | Streaming subscription state machine | in place | Existing run streaming remains separate from this session. |
| A2 | Layered tool policy pipeline | hardened | Candidate lifecycle, Web/API controls, provider gates, and candidate runtime loading complement execution policy rather than replacing it. |
| A3 | Mutation fingerprinting + sticky lastToolError | implemented | Unchanged this session. |
| A11 | Raw diagnostics/audit trail | improved | Candidate loader emits explicit loaded/skipped audit events. |
| A12 | Transcript/tool-call repair | partial | Provider schema adapters and benchmark scoring exist; candidate schema compatibility reporting remains future work. |

## Recommended Next Slice

1. Add safe CLI/API/operator controls for candidate implementation metadata,
   starting with `delegated_tool` bindings and typed confirmations.
2. Add candidate lifecycle commands/API for `experimental` and `deprecated`
   transitions where operator review needs those intermediate states.
3. Surface provider fallback recommendations in runtime diagnostics when
   provider gating removes tools from a turn.
4. Add richer schema-compatibility reporting per provider/tool family for
   candidate review.
