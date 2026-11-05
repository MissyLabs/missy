# Rust completion and acceptance handoff

Initial review 2026-11-05, documentation-only. Implementation owners were still editing
the shared tree. This review read source/tests/per-crate reports and ran
`git diff --check`; it did not run tests, use live credentials/devices/services,
deploy, commit, or change code. **Final release verification is pending against
an exact final commit.** The inspected branch is `rewrite/rust-runtime`, based
at `6130532` during review; that base does not identify the uncommitted changes.

**Coordinator final acceptance, 17:45 UTC:** all owners frozen, 35 crates,
521 ordinary passing entries plus all 3 explicitly enabled native cases, strict
format/lint/audit, final release build and 15 Chromium assertions passed.
The earlier review's pending language below is historical. Exact results,
binary hash, native dependencies and limits are in `validation.md`; actual PR:
https://github.com/MissyLabs/missy/pull/171. Hosted final-head CI is checked
separately, not inferred from older green runs.

## Deliverable and native provenance

The end-user artifact must be the **native `missy-rs` executable plus its actual
PR link**, not just an inventory. The coordinator must record the final commit,
binary download/path, SHA-256, target platform, required native libraries/licenses
and final validation results. This reviewer did not look up a PR or execute a
binary. Remote: `https://github.com/MissyLabs/missy`. Do not invent the PR URL or
reuse an earlier build's success as evidence for the final tree.

The runtime does not import the legacy Python package or dispatch through a
Python compatibility server. Python references in Rust test fixtures and the
optional console harness are development tooling. Migration reads legacy data
without executing Python. Explicit CLI providers, STT/TTS programs, FFmpeg,
CDP browser and libdave remain optional real dependencies, not a claim of a
dependency-free static binary. [DAVE provenance](../../crates/missy-dave/README.md)
documents the official C API, pinned sources, trusted-library loading, licenses
and linked-system-library limits; this is not locally invented MLS cryptography.

## Source-observed capability

`crates/missy-cli/src/main.rs::build_runtime` composes native builtins with
optional personal memory, retrieval, history, Discord tools, plugins, HTTP MCP,
integrations, desktop, media, vision, capture and specialty executors. Composite
execution, optional sandboxing and durable governance are application wiring,
not a permissive direct-executor API. CLI providers deliberately take a separate
chat-only NoTools path; that path does not also expose composite tools.

Native application paths include durable turn/recovery/approval handling,
HTTP/browser console, Discord text and consented voice, paired voice edge,
schedules/workflows, mesh, reviewed evolution, operator DAG subagents,
hatching/persona proposals and experiments. Runtime telemetry is wired rather
than merely an exporter library. These are observed implementation claims, not
fresh whole-tree test results or full drop-in parity.

Important distinctions, with configuration and limitations in the linked docs:

* [Personal memory](../../crates/missy-personal/README.md) crosses sessions/channels
  only for the same tenant+user. [History](../../crates/missy-history/README.md)
  uses all four scope components, immutable evidence and optional separately
  budgeted summarization; current source retains receipts durably in SQLite.
* [Vision](../../crates/missy-vision/README.md) sends real sanitized multimodal
  image payloads. Discord lookup/upload tools provide a separate authorized
  delivery path; generation, validation and delivery are different operations.
* [Media](../../crates/missy-media/README.md) includes native pinned FFmpeg edits,
  explicit audio mux/mix/loop, captions, speed and crossfade.
  [Integrations](../../crates/missy-integrations/README.md) include approved Comfy
  graph execution, owned artifact publication/conditioning and typed Nomad/Incus
  grants. [Specialty](../../crates/missy-specialty/README.md) includes actual ACE
  adapter, score-to-WAV/MIDI and project-scoped remote Foundry workflows.
* [Discord voice](../../crates/missy-discord-voice/README.md) uses the shared
  authenticated Gateway, operator consent/scope/mode, real NoTools separation,
  native UDP/Opus and official DAVE. Synthetic/native-crypto acceptance does not
  establish live Discord interoperability or sound quality.
* [Desktop](../../crates/missy-desktop/README.md) is explicitly isolated X11,
  typed AT-SPI and fixed CDP operations; [capture](../../crates/missy-capture/README.md)
  is actual allowlisted V4L2/isolated X11 capture, not the media file-copy operation.

## Explicit configuration and default deny

`missy-rs init` creates a new private native config and refuses overwrite.
`config-check` validates the main schema/policy, **not** extension dependencies,
credentials, inference or remote readiness. Main config is owned, regular,
non-symlink, at most 1 MiB and not group/world writable. Extension config files
must additionally have no group/other permission bits, normally mode 0600.
Relative state, policy roots and extension paths resolve against config directory.

| Capability | Required operator selection |
|---|---|
| Model/network | Explicit provider kind/model/base URL and credential environment name where required; exact `policy.allowed_origins` including ports; separate `allow_private_network` for private destinations. A provider URL grants no network access by itself. |
| Budget | Default runtime: 32 steps, 200000 tokens, 8 continuations, 600 seconds, 8 concurrent turns, `max_cost_microusd=1000000`. A nonzero cost ceiling requires known provider input/output prices on the ordinary HTTP startup path. Explicit zero prices are appropriate only for genuinely uncharged use; a zero ceiling disables monetary enforcement. Unknown consumed cost under a finite ceiling halts in the runtime. |
| Files/processes | Explicit `read_roots`, `write_roots`, file byte cap. Shell/MCP default off. Shell needs `shell_enabled` and exact complete `allowed_commands` argv. Deletion additionally needs `allow_file_delete` and durable approval. |
| Extensions | `extensions.*_config` is opt-in; each subsystem's internal enabled/mutation flags remain authoritative. `personal_memory_enabled` and `retrieval_enabled` are separate false-by-default grants. No automatic personal-memory system-prompt injection. |
| Approval | `approvals_config` is mandatory for summary/context mutations, deletion, Discord uploads, plugins, HTTP MCP, integration mutations, desktop, media, vision, capture and specialty mutations. Startup checks every dynamically collected protected tool against `sensitive_tool_names`. A model-supplied `confirm`/`consent` is not approval. |
| History/vision | `history_config` has `history` plus optional dedicated `summary` provider/policy. CLI fixes evidence/summary/vision ledgers under private state. Their independent reservations/accounting are not implicitly included in the main runtime's provider usage. |
| Native execution | Media/capture require approved absolute executable paths/SHA-256, private staging, narrow runtime roots and supported Landlock. Plugins need reviewed registry/hash, exact scope/IDs, native executable and exact argv grants; no implicit interpreter/compiler or hot reload. |
| Discord | Disabled by default, DMs off, mention required. Explicit token environment binding and guild/channel/user grants. `discord_voice_config` additionally needs consented bindings, STT/TTS, exact `JOIN DISCORD VOICE <guild>/<channel>` approval, voice host/UDP grants and trusted `native_dave` path/hash/version/source commit. Paired edge uses separate `voice_config` and device approval/mode. |
| Desktop/capture | Explicit isolated endpoints/cookies and independent CDP-host network isolation. No ambient active-desktop discovery. Camera paths and screencast consent are separate; capture currently refuses microphone audio. |
| Services/telemetry | Explicit endpoints, credentials, resource/action/workflow grants and mutation opt-in. Comfy artifact downloads default disabled, maximum 4 MiB. Incus requires configured bearer-auth proxy/API, not implicit mTLS setup. `telemetry_config` needs explicit allowed endpoint; export defaults off and installs no collector. |

Use the per-crate README schemas above and `missy-core::ExtensionConfig` rather
than legacy YAML examples. Protect configuration and ledgers from untrusted
writers; configuration is authority, not model input.

## Verification evidence and completion gate

**No tests were run by this reviewer while concurrent modules were incomplete.**
Test-source presence is not a pass result. Existing evidence has narrower scope:

* `validation.md` records an earlier 350-entry/24-crate run, release build and 15
  browser assertions. Those totals predate this expansion and need final results.
* `final-security-review.md` records a reproduced personal FTS ranking side
  channel, failure before repair, then 12 personal tests and focused lint/runtime
  observer passes. It does not certify DAVE or the final whole CLI.
* The DAVE README records actual upstream C++ 62, C API 9 and Rust wrapper 3 passes
  with trusted native integration explicitly enabled. Ordinary Cargo tests leave
  that native-library integration ignored unless deliberately selected.
* CLI acceptance sources cover executable restart/dedup/isolation, exact approval
  resume, DAG budgets, scheduling/workflows, evolution review/apply/rollback,
  telemetry flush, vision HTTP payload/ledger, deletion replay, history receipt
  expansion and missing integration-guard names. Final execution is pending.
* Other fixtures use synthetic media, disposable authenticated Xvfb and loopback
  protocol/provider servers, not live hardware, real inference or infrastructure.

Freeze the finished tree, then record format/lint/locked workspace tests/release
build/dependency audit plus executable/browser acceptance against the **same
commit**. List ignored/skipped tests and separately enabled native crypto tests.
Live credentials, hardware and GPU/voice quality require separately authorized
canaries. Lack of a live test is an acceptance gap, not evidence a tool is absent.

## Residual differences and integration cautions

Exact argv, typed resource grants, approved graphs, reviewed DAGs/diffs, fixed
browser expressions and approved native plugins are deliberately bounded APIs,
not missing arbitrary shell/HTTP/JavaScript/Python tools. Conversely, partial
semantics under a familiar name do not prove complete legacy parity. Codex CLI
completion remains intentionally blocked without verified complete no-tools controls.

Concrete limits include microphone capture, speaker volume/routing and synchronized
VTube speech; capture clips are silent. Voice has energy-gate VAD, no adaptive
jitter buffer/PLC. Foundry artifacts are metadata, storyboard consumes supplied
images, deterministic score synthesis is not model singing, and ACE full mix has
no stems/phoneme timing. OBS source-text setting is outside its mutation catalog.
Comfy MP4 validation is structural, not a full codec decode. These are different
from live interoperability not yet accepted. Do not claim all legacy parity.

The review initially found safe-voice provider-selection/preflight inconsistency.
The parent's subsequent `build_safe_runtime` now selects configured CLI/native
providers, requires known HTTP prices under finite budgets, and uses observed
runtime creation; both Discord and paired voice call sites use it. That repair
was verified by source inspection, not a new test run. Runtime unknown-cost checks
already fail closed; no finite-budget bypass was demonstrated.

Telemetry handles remain retained until awaited process shutdown. Interactive
`run` constructs once, not per prompt; agent pools construct a finite configured
provider set. No unbounded construction loop/leak was demonstrated. Large provider
sets can still multiply queues/workers and sequential shutdown waits. Independent
vision/summary costs and uncertain remote GPU/infra effects also need operational
reconciliation; timeout/cancellation is not rollback or permission to replay.

## Migration, rollout and documentation follow-up

[Migration](../../crates/missy-migrate/README.md): use private consistent offline
SQLite backups, explicit one-to-one full scope mapping, `migrate plan`, inspect
blocking/review issues, then confirmed `migrate apply` into a new absent destination.
Never import live files/WAL/SHM or overwrite an existing installation. Unsafe config
grants block conversion. Historical records become inert scoped memory, not runnable
turns; global learning/checkpoints are quarantined. No activation is performed.

Keep separate config/state, finish isolated acceptance, then authorized dedicated
credential/read-only canaries and capability-by-capability rollout. Preserve the
legacy installation until required parity is accepted. A binary/PR handoff does
not authorize production replacement, restart, or desktop/hardware access.

Coordinator follow-up completed: refreshed validation/feature inventory/PR
summary, persistent history retention README and principal-local ranking README.
Earlier evidence remains explicitly dated. `parity.md` is a ledger, not an
acceptance waiver.
