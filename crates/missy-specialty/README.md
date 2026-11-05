# missy-specialty

Native Rust specialty executor. Public entry point:

```rust
SpecialtyTools::new(policy: PolicyConfig, config: SpecialtyConfig) -> anyhow::Result<SpecialtyTools>
```

Implements `ToolExecutor`. The parent runtime must wire `specialty_config`,
configuration fingerprints, durable approval, identity/scoping, deadlines and
effect journaling. Calling `execute` directly is **not** an approval boundary.
Default config exposes no tools. No subprocesses, shell, Python bridge,
workstation playback, camera, desktop, clone, checkout or local build/test run.

## Actual legacy inventory and implementation

Source inventory: `missy/tools/builtin/calculator.py`, `singing_generate.py`,
`repoeval_tools.py`, plus existing Rust media/providers/integrations crates.

| Exact legacy tool | Native implementation here | Deliberate differences |
| --- | --- | --- |
| `calculator` | Bounded parser for real literals, parentheses, unary signs, `+ - * / // % **`; Python-compatible precedence and negative floor/remainder | Finite f64 only. No arbitrary-precision integers, complex or bitwise operators; rejects instead of substituting expressions. |
| `singing_generate` | Actual ACE-Step 1.5 Comfy graph: fixed UNET, two CLIP encoders, VAE, conditioning, negative, sampler, latent, decode and SaveAudio. Readiness model/node inventories; prompt submission, bounded polling, history descriptor validation, authorized `/view`, native full PCM WAV/FLAC decoding and atomic publication | One operator endpoint, no model host override, CPU/GPU fallback discovery, arbitrary graphs, server `fullpath` file copy or auto-created output directories. GPU availability is explicitly **unverified**. Requires explicit `save_path`. Hard shared HTTP response cap 4 MiB may refuse long songs. Mixed stems/timings remain unavailable, `lip_sync_safe=false`. |
| `repoeval_foundry_read` | Exact legacy project routes and `{ok,data}` envelope: list, plan, status, artifacts, compare, report | Operator allowlists restrict IDs in addition to authenticated project scope; plan additionally requires operator-reviewed workload digest. |
| `repoeval_foundry_mutate` | Snapshot, reviewed staging-plan start, cancel; exact status 202, response identities, idempotency, reserved/submitted acknowledgements | Explicit project acknowledgement for cancel as well; outer runtime approval required. No implication that a reservation executed. |
| `repoeval_foundry` | Compatibility combined surface when mutations enabled | Conservatively marked side-effectful even for reads. |

Foundry is a remote authenticated evaluation coordinator, not a local `git clone`
runner. This crate does not invent an equivalent local evaluation tool or claim
to sandbox hostile build scripts. The coordinator owns remote sandbox/image
enforcement; client validates reviewed pinned image and policy evidence only.
No ACP/acpx provider work here; provider implementation belongs to its executor.

Existing `video_edit`, `video_storyboard`, `image_edit`, capture and provider
implementations are **not duplicated**. Graph/retrieval, personal memory,
voice, infrastructure and runtime composition belong to their existing owners.
This is not a claim of complete repository-wide legacy parity.

## Optional additional capabilities, not legacy substitutes

* `music_compose`: bounded explicit note grid -> deterministic mono 22050 Hz
  signed-16 PCM WAV plus format-0 standard MIDI with tempo and note-on/off.
  Sine/triangle oscillator, attack/release envelope, normalization, no sysex,
  no playback and no fabricated vocals. Max 1024 notes, 32 voices, 120 seconds;
  same-pitch overlap denied so MIDI and synth note lifetime agree. MIDI derives
  the same tempo/grid/pitch/duration/velocity as audio. No claim that a MIDI
  synthesizer's timbre equals this WAV.
* `document_extract`: UTF-8 `.md`, `.markdown`, `.txt` read through Policy;
  original text, ATX headings and fenced code blocks. Never executes blocks or
  embedded instructions. Not a general CommonMark renderer, PDF extractor or OCR.

Each output uses existing Policy descriptor-relative no-symlink atomic private
file replacement and returns byte count, MIME type and SHA-256. Existing approved
files can be replaced. Two music artifacts are individually atomic, **not** a
multi-file transaction; first may exist if second publication fails. Paths and
byte caps are rechecked at execution. Local same-UID/root filesystem attackers
remain outside the shared Policy threat model. CPU synthesis/decoding is bounded
but synchronous; cancellation is checked before local execution, not every sample.

## Configuration

`SpecialtyConfig` fields, all deny unknown fields:

* `enabled=false`: master opt-in, including calculator.
* `singing: Option<SingingConfig>`: `{base_url, max_polls=300,
  poll_interval_ms=1000}`. Exact HTTP(S) origin, no URL credentials/query/path.
  Shared Policy must authorize origin and private networking where necessary.
  Node/model readiness is checked each call; max 600 polls, interval 10..5000ms.
  Cancel/timeout does not pretend the remote GPU job was cancelled.
* `foundry: Option<FoundryConfig>`: `base_url` (origin or `/api`), `project_id`,
  `token_env`, `repository_ids`, `run_ids`, `snapshot_ids`,
  `allow_mutations=false`, `reviewed_workload_sha256`, `idempotency_prefix`.
  Credentials are read at request time and never serialized/fingerprinted.
  Blank prefix means `missy-`. Use a distinctive explicit namespace in production.
* `enable_music=false`, `enable_documents=false`.
* `max_audio_seconds`: zero means 60, hard max 120.
* `max_document_bytes`: zero means 1 MiB, hard max 4 MiB. Policy's own cap also applies.

Foundry `plan` takes the exact workload JSON. Compute its reviewed digest from
canonical `serde_json::Value` serialization (sorted object keys, compact JSON,
UTF-8); no floats recommended. `serde_json` must not enable `preserve_order`.
Workloads include immutable commit/snapshot, `image@sha256:<64 lowercase hex>`,
1..8 providers and bounded repetitions/parallelism/timeout/attempts. They must
match an operator-configured digest and repository allowlist. Call `list` first.
The authenticated returned repository set is intersected with operator scope.
Failed refresh invalidates grants. A new plan invalidates previous plans.

A returned plan must echo the exact workload and have `state=planned`,
`placement.pool=staging`, and all seven true policy checks: repository, image,
providers, budget, quota, egress, audit. `start` requires cached plan ID,
namespaced key and project acknowledgement. Cache is bound to full caller scope
and credential digest; scope/credential rotation invalidates it. Cache is in
memory, intentionally not durable or reusable after process restart. Re-list and
re-plan after restart. Calls are serialized so grants cannot race refresh.
Operator resource allowlists still apply to subsequent status/cancel/artifact
reads; a newly allocated ID must be authorized by the operator before those calls.

Reserved starts require independently derived parent and each child ID, unique
provider/repetition coverage, inert parent job ID and exact required-artifact
policy. Submitted starts require a job ID but are still not execution-complete.
Compare validates complete unique pairs, reason paths and aggregate boolean;
report validates draft/unpublished status and exact run coverage across keyed
groups/exclusions. Artifacts are metadata only, not automatic downloads.

All HTTP uses PolicyHttpClient: exact allowed origins, bounded response/request,
DNS/private-address policy, no redirects/proxies. No network bypass. Error
responses never expose raw remote bodies. Foundry refuses exact bearer echoes
and recursively redacts known secret/log/URL fields from returned evidence.

## Verification

`cargo test -p missy-specialty` and
`cargo clippy -p missy-specialty --all-targets -- -D warnings`.

Tests use synthetic temporary files and loopback fixtures only. They exercise
the actual legacy project-route/envelope plan->reserved-start exchange, ACE
graph->history->view->decoded WAV publication, independent WAV/MIDI decoding,
440 Hz zero crossings, determinism, policy/symlink denials, cancellation, scope
validation and arithmetic resource limits. No real Foundry/Comfy/GPU was contacted.
FLAC decoding is verified with a complete synthetic STREAMINFO/verbatim frame,
including CRC and truncation rejection. Real GPU inference/model quality is not
certified by these synthetic media fixtures.
