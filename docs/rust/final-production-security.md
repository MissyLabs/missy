# Independent production security and wiring audit

2026-11-05 UTC, concurrent-development snapshot of `rewrite/rust-runtime`, PR171.
This is code and disposable fixture evidence, **not live production acceptance**.

## Scope and artifact

Read `CLAUDE.md`, security/production parity/lifecycle notes, actual CLI composition,
runtime/provider projection and dispatch, durable approvals, database-global leases
and budgets, native automation/delegation, mesh dispatch, desktop/vision/media
compound adapters, voice late attachment, Discord intake/attachments/outbox.

The independent artifact is this document plus
`crates/missy-cli/tests/security_production_e2e.rs`. Only these new files were
written by this audit. Other changes in the shared tree belong to their owners.
No commit, deployment, account, real provider, live secret, device or remote
service was used. Child process fixtures used only the native binary, disposable
private config/state and its scheduler; no provider request was issued. Existing
gateway fixtures use loopback servers. Builds were focused, at most two jobs.

## Verdict

**No remaining model-input path bypassing the inspected production durable outer
approval guard was demonstrated.** Exact-binding, reduced-capability execution,
raw-evidence separation and shared admission controls have executable evidence.
This is not proof of absence of bypasses or universal prompt-injection detection.
Three issues found during inspection were reported early and mitigated by the
parent/owners before the final checkpoint below. A newly added owner cancellation
test has a failing observation-count assertion, recorded below rather than hidden.

### Findings and disposition

1. **Parent/child slot starvation at one global slot.** The automation base and
   reasoning-only child runtime intentionally share the parent's Store. The Store
   counts all unexpired leases (`missy-store/src/lib.rs`,
   `acquire_lease_bounded`), including the waiting parent. A one-slot parent plan
   therefore produces an `admitted` child with no tool result or execution.
   `reproduces_plan_child_starvation_at_valid_single_turn_global_capacity`
   proves this through actual outer approval, Runtime, AutomationTools and Store.
   The production builder now rejects orchestration config when
   `max_concurrent_turns < 2`, before execution. A separate real-binary test proves
   failed startup, false readiness and runtime `startup_failed`. Direct library
   callers must enforce this precondition themselves. Even with two or more slots,
   concurrent parents can consume all slots: capacity refusal is backpressure,
   not successful child work. Do not reinterpret an admitted plan receipt as a
   completed workflow. Supporting guaranteed progress under saturated parents
   would need explicit coordinator/child capacity management, not independent
   unbounded stores.

2. **Speech backend identity missing from some adapter fingerprints.**
   `missy-discord-voice/src/tools.rs::configuration_fingerprint` and `attach`
   bind the server config, but not STT/TTS dependency identity.
   `missy-voice/src/audio_device.rs` hashes presence booleans for TTS/STT;
   `missy-integrations/src/speech.rs::with_speech` binds output fingerprint but
   not TTS. The independent fake-backend fixture proves distinct speech backends
   can attach with identical voice fingerprints. Previously CLI approval context
   hashed Config containing extension **paths**, so speech YAML edits at a fixed
   path could preserve approval lineage across reconstruction. The production
   builder now captures `production::fingerprint(cfg)`, including parsed semantic
   content of referenced `*_config` files, uses it in both guard contexts, and
   verifies it again before returning the runtime. Same-path semantic extension
   modification is detected by the actual binary reload fixture. This mitigates
   the inspected CLI path; it does not make arbitrary trusted public-library
   dependency injection immutable. Future reusable composition should add speech
   dependency fingerprints to the traits/pending constructor and validate the
   attachment identity. Trusted same-UID executable replacement is not sandboxed
   by configuration hashing.

3. **Prompt planner cancellation initially detached work.** New prompt-based
   delegation originally selected cancellation against `Runtime::run`, whose
   execution task outlives the dropped caller future. The issue was reported
   during final reinspection. Current `missy-subagents/src/tools.rs::PlannerGuard`
   and `execute_request`
   starts an admitted planner, persists its turn ID, installs PlannerGuard and
   polls the idempotent existing turn; drop/cancellation/deadline cancel the actual
   scoped turn. The independent
   `regression_probe_planner_cancellation_must_not_leave_detached_running_turn`
   blocks a fake provider, cancels the parent and verifies planner cancellation
   requested or terminal status, then drains the disposable runtime. It passes.
   The original bad implementation was found by source inspection; a pre-fix
   executable failure was not captured because the owner fix landed while the
   fixture was being built. This is not evidence of a pre-fix test execution.

## Actual boundary observations

* `extensions.rs::OuterGuardVerified` deliberately returns approval success. It
  is not an authorization implementation. Its safety depends on being unreachable
  except through the assembled outer GuardedTools. The inspected builder collects
  capture, plugins, HTTP MCP, audio, desktop, vision, media, integration, workload,
  evolution and voice mutations; startup rejects missing durable approval config
  or missing sensitive names. Orchestration adds an outer guard around its new
  mutations; its separate base runtime retains the original guard. Inner policy
  preflight still runs before approval consumption and execution.
* SceneAdapter calls the trusted `approved_scene` compound, not the general tool
  dispatcher. Its fingerprint includes IntegrationTools and the operator-owned
  SceneGenerationGrant. Parent `video_generate_storyboard` is effectful and
  guarded; output paths/dimensions/prompts and generated artifacts are checked.
  A compound action can submit external scene jobs before later failure. Its
  uncertainty is not permission to replay or globally cancel another job.
* VisionInterpreter calls `analyze_png` from approved desktop capture. Desktop
  fingerprint includes interpreter configuration; vision shares its durable
  ledger path with separately exposed vision tools. It uses NoTools multimodal
  requests, strict response checks and image bounds. This does not authenticate
  visual model descriptions as facts. Pixels are captured at execution time;
  exact call approval is not a claim that previously reviewed pixels cannot
  change before capture.
* CLI safe voice and configured reasoning-only children use a real NoTools
  executor, not a prompt suggesting tools be hidden. The independent literal
  `file_write` attempt through a NoTools Runtime is refused and creates no file.
  Child scopes preserve tenant/user/channel and derive only session; no model
  arguments grant a different principal. CLI process-provider composition is
  explicitly chat-only and rejects ignored capability extensions.
* Discord late attachment is one-shot and requires identical server config.
  `discord_voice_say` is effectful, uniquely matches authenticated caller scope,
  checks existing joined session, and cannot supply an actor override. Voice
  service reference does not introduce extra tool names or consent after runtime
  catalog construction. Config-content binding mitigation above remains essential.
* ApprovalBinding includes full Scope, turn ID, call ID, tool name, argument hash
  and context identity. The independent test changes arguments, call ID, user
  and turn ID after approval: all refuse before the counter executor runs. The
  original call runs once; reuse refuses. Approval does not override inner policy.
* Runtime repeatedly verifies configuration, stores raw durable requests and tool
  evidence separately, projects input/history/tool JSON before provider calls,
  redacts final/cached/error output, and checks checkpoint system/request hashes.
  Object keys and nested serialized JSON cross projection too. The actual
  BuiltinTools HTML fixture retains exact original source and synthetic credential
  in scope-protected evidence while withholding raw canary/credential/raw envelope
  from delivered content. Deduplication does not mutate evidence.
* Shared full/base/NoTools runtimes use the same runtime.sqlite3. Database-global
  lease limits survive separate Runtime/Store instances. Durable cumulative budget
  policy mismatches and uncertain billing fail closed. Vision and history-summary
  provider calls use their own bounded ledgers: runtime provider usage is not a
  claim of one unified bill including every separately priced extension. Configure
  those extension caps deliberately. Neither ledger proves external billing truth.
* Mesh authenticates and reauthorizes signed exact-scope capability grants, derives
  stable request IDs and dispatches through the shared bounded Runtime. It does
  not directly execute ToolExecutor or return response content to peers. Startup
  rejects mesh Runtime bounds outside its finite local requirements.

## Discord evidence and delivery reinspection

The requested recent fixes were inspected in actual source and relevant existing
fixtures were executed, not merely inferred from documentation:

* Builtin wrapper preserves `raw_evidence` rather than redacting it in place.
  Runtime exposes `model_view`, including recursively nested envelopes.
* Quoted reply content crosses `inspect_untrusted` before mixed prompt framing;
  full reference remains in `Incoming.raw_reference` in the private intake payload.
  Wrong-channel/mismatched referenced-message IDs do not become current instructions.
* Attachment downloads use explicit HTTPS Discord CDN origins and exact
  channel/attachment path binding, no credential forwarding or redirect allowance.
  Actual bytes are retained as Base64 plus hash before text parsing in immutable
  `inbound_evidence`. ZIP members are bounded text read in memory, never extracted;
  path escapes, symlinks, excessive expansion and duplicates are rejected.
  Shared model context bounds and quarantine are separate from private raw storage.
* `Journal::result` binds original complete text hash before outbox chunk creation.
  Divergent later text is refused. Prepared context and inbound source hashes are
  likewise immutable. Uncertain delivery requires exact review-binding operator
  reconciliation and never silently requeues. Existing restart fixtures exercise
  no provider re-execution/no duplicate delivery behavior.

Raw evidence can contain credentials and injections. Scope-protected private
storage is not a redacted user-facing API. Heuristic redaction does not recognize
all possible secret formats or encoded/language-specific prompt attacks. Gateway
and ordinary runtime delivery must not be repurposed as authorized raw retrieval.

## Executed checkpoint

All commands used `CARGO_BUILD_JOBS=2` and `--test-threads=2`:

| Focused command | Result |
| --- | --- |
| `cargo test -p missy-cli --test security_production_e2e` | 8 passed |
| `cargo test -p missy-store --test global_capacity` | 2 passed |
| `cargo test -p missy-store --test operator_budget` | 2 passed |
| `cargo test -p missy-runtime --lib production` | 5 passed |
| `cargo test -p missy-gateway --lib discord::` | 28 passed |
| `cargo test -p missy-automation --test production` | 7 passed |
| `cargo test -p missy-mesh --lib` | 22 passed |
| `cargo test -p missy-discord-voice --lib service` | 10 passed, 2 ignored |
| `cargo test -p missy-subagents --lib` | 20 passed, 1 failed |

**104 focused checks passed, 1 failed, 2 ignored.** The final subagents run's
`explicit_planner_cancellation_and_deadline_cancel_real_provider` failed at
`src/tests.rs:360`: expected provider cancellation-handler count1, observed0,
despite waiting for active provider count to reach0. Runtime cancellation can
drop the provider future instead of executing its cancellation branch; this
assertion needs owner diagnosis. Do not claim the subagents suite is green from
this audit. The independent durable cancellation/terminal test passed, as did the
owner future-drop/no-replan test. Native full voice/DAVE roundtrip and downgrade
fixtures are ignored unless explicit pinned acceptance dependencies are supplied.

Two green tests intentionally reproduce library
limitations documented above; green does not mean those library APIs were fixed.
Earlier concurrent attempts hit a temporarily absent subagents module and one
nonprivate gateway test-directory setup. Final reruns passed after owner changes.
An earlier seven-test run reported subagents unused/dead-code warnings during
ongoing owner work. The subsequent post-format seven-test rerun passed without those
warnings after owner changes; this audit does not claim a workspace clippy result.
No full workspace build, live hardware/media acceptance or production cutover was
attempted. Re-run after owners finish because this is a moving shared checkout.
