# Independent production parity audit

**Integrated workloads update:** the new actual native gateway child test
`crates/missy-cli/tests/workloads_production_e2e.rs` passed model-tool admission,
exact durable approval, explicit resume, shared supervisor due tick, actual
mock Nomad optimistic submission, SIGTERM drain and restart/no duplicate.
The shared-Arc double-open coordinator-lock bug was found and fixed with the
lifecycle owner. This supplies application-wiring evidence for workloads, not
a post-concurrency certificate for all106 features. The original inventory
below remains dated source evidence and must not be read as current failure
status after other agents' changes.

## Follow-up implementation note

The numbered inventory below is the original read-only snapshot, not a claim
that concurrent fixes are still missing. After parent-authorized scope
expansion, this audit agent implemented `crates/missy-workloads` and
`docs/rust/workloads-production.md`: native durable Nomad offload, benchmark,
reconciliation and finite schedules. Targeted actual loopback HTTP acceptance
passes9/9 and strict crate clippy passes. This supersedes original missing
rows49/56/58/59 for **crate implementation**, not application wiring or live
acceptance. Parent/lifecycle owns shared-instance runtime/supervisor integration.
Other teams are concurrently implementing security, lifecycle, runtime,
attachments, audio and media. Their changes require separate final inspection;
this document must not silently relabel source-snapshot observations as a
post-integration certification.

Source inspection on 2026-11-05, approximately 18:13–18:21 UTC. This is a
**concurrent-development snapshot**, not a release certificate. No workspace
build, live credential, account, process, device, provider or infrastructure
connection was used. Only this document was written during the audit. Later
workloads implementation, if present, is separately documented. Test source was
inspected, not executed. Passing test counts do not establish feature parity.

## Method and verdict

Read `CLAUDE.md`, actual `_ALL_TOOL_CLASSES` in
`missy/tools/builtin/__init__.py`, tool implementations, CLI reference and
`missy/cli/main.py`, channel handlers, runtime/security/config source and Rust
executor composition. AST enumeration confirms **106 normally registered
builtins**, plus two conditional Foundry tools. `repoeval_foundry` is a class
but is not one of the 106 normal registrations. There are no normal legacy
`web_search`, PDF/OCR document parser or general `schedule_task` builtins.

At this snapshot the branch has substantial native implementations, but is
**not yet a complete production behavior replacement**. Some earlier final
disposition prose overstates capability: a generic adapter cannot be counted
as every legacy alias, structured scene analysis is not scene memory, and
allocation metadata is not an offload result artifact. Findings were sent
early to the parent and runtime/security owners. Other agents are implementing
these lanes; re-inspect their final changes before using this list as current.

Legend: **I** implemented source path; **R** deliberate replacement API or
security restriction, not automatically a bug; **P** partial capability or
missing semantic behavior; **M** missing runtime/model-callable behavior at
snapshot. Hardware/remote acceptance is separate from implementation.

Rust source abbreviations below: `tools` = `crates/missy-tools/src/lib.rs`;
`desktop` = `crates/missy-desktop/src/`; `integrations` =
`crates/missy-integrations/src/`; `media` = `crates/missy-media/src/`;
`capture` = `crates/missy-capture/src/`; `vision` =
`crates/missy-vision/src/lib.rs`. Every optional executor also needs successful
configuration and wiring through `crates/missy-cli/src/main.rs::build_runtime`.

## Exact 106-tool inventory

Rows follow actual registration order. Legacy files are under
`missy/tools/builtin/`; the row name identifies the class's exact `name`.

| # | Exact legacy name | Native behavior and precise disposition |
|---|---|---|
| 1 | `atspi_click` | I/R `desktop/atspi.rs`, `desktop_accessibility_action`, typed allowlisted actions. |
| 2 | `atspi_get_text` | I/R `desktop_accessibility_get_text`; bounded exact application/window. |
| 3 | `atspi_get_tree` | I/R `desktop_accessibility_tree`; bounded tree, explicit bus. |
| 4 | `atspi_set_value` | I/R `desktop_accessibility_set_value`; exact typed target and approval. |
| 5 | `browser_click` | I `desktop/browser.rs`, fixed DOM click selector; not arbitrary scripting. |
| 6 | `browser_close` | I closes explicit CDP target. |
| 7 | `browser_evaluate` | R only expression IDs `url/title/ready_state`; arbitrary JS deliberately refused, so custom scraping/workflow scripting not equivalent. |
| 8 | `browser_fill` | I/R sets input/textarea and dispatches input/change; not all browser form-control semantics. |
| 9 | `browser_get_content` | I/R `browser_read`, bounded body innerText/title/URL; no general selector/HTML extraction. |
| 10 | `browser_get_url` | I fixed current-target URL read. |
| 11 | `browser_navigate` | I/R explicit target and exact origins, independent isolated-browser network acknowledgement required. |
| 12 | `browser_screenshot` | I/R explicit-target bounded screenshot; file/result shape differs. |
| 13 | `browser_wait` | I/R selector visible/hidden, timeout 1–15000 ms; narrower than general Playwright waits. |
| 14 | `calculator` | I/R `missy-specialty/src/calculator.rs`, bounded finite real arithmetic, not arbitrary Python or complex/bitwise. |
| 15 | `code_evolve` | M model-callable tool absent; `missy-evolution` CLI reviewed propose/show/review/apply/rollback/audit/recovery exists. CLI alone does not let model propose requested changes. |
| 16 | `context_shunt` | I/R `missy-history`, immutable retained items + separately budgeted NoTools summary worker when configured. |
| 17 | `delegate_task` | M model-callable adapter absent; native `missy-subagents` provider-selected DAG and CLI `agents` exist. Legacy accepts prompt decomposition OR explicit agents, diverse specialists/synthesis and three failure policies. |
| 18 | `discord_lookup_user` | I/R `missy-discord-tools`, configured guild/user allowlists, policy-bound lookup. |
| 19 | `discord_upload_file` | I/R same crate, bounded multipart policy-file delivery, exact durable outer approvals; separate from incoming attachments. |
| 20 | `discord_voice_join` | P native service/Gateway bridge + explicit CLI consent join exist; no same-name model-facing ToolExecutor found. |
| 21 | `discord_voice_leave` | P service shutdown/leave exists; model-callable operation not found. |
| 22 | `discord_voice_say` | P voice service/speech backend exists; model-callable say not found. |
| 23 | `discord_voice_status` | P service state exists; model tool not found. |
| 24 | `file_delete` | I/R `tools`, exact expected SHA256 + configured capability + durable approval, descriptor-relative quarantine/recheck. No recursive broad deletion. |
| 25 | `file_read` | I/R UTF-8 bounded policy read; no generic binary decoding. |
| 26 | `incus_copy_move` | I/R `integrations/incus_ops.rs`, configured Copy/Move grants, fixed resource scope. |
| 27 | `incus_config` | I/R pinned typed resource Update grants; model cannot submit arbitrary host config. |
| 28 | `incus_device` | I/R DeviceSet/DeviceDelete exact grants. |
| 29 | `incus_exec` | P/R pinned argv Exec exists; verify actual guest stdout/stderr collection, not just async operation receipt. |
| 30 | `incus_file` | I/R configured `incus_file_read`, FileWrite grant; binary/arbitrary guest paths intentionally restricted. |
| 31 | `incus_image` | I/R Image resource discovery/Create/Delete grants, full fingerprints. |
| 32 | `incus_info` | I/R configured instance info and resource inspect. |
| 33 | `incus_instance_action` | I/R Start/Stop/Restart/Freeze/Unfreeze exact allowlisted instances. |
| 34 | `incus_launch` | I/R Create Instance grant with allowed pinned source image. |
| 35 | `incus_list` | I/R `incus_discover` filters configured resources. |
| 36 | `incus_network` | I/R Network resource typed create/update/delete grants. |
| 37 | `incus_profile` | I/R Profile resource typed grants. |
| 38 | `incus_project` | I/R configured project resource, cannot escape configured project. |
| 39 | `incus_snapshot` | I/R list/create/restore/delete pinned snapshot grants. |
| 40 | `incus_storage` | I/R StoragePool/StorageVolume resource grants. |
| 41 | `file_write` | I/R policy write, strict content/path schema. |
| 42 | `graph_query` | I/R `missy-retrieval`, scoped bounded traversal rather than arbitrary graph query language. |
| 43 | `image_generate` | I/R approved Comfy graphs + `comfyui_submit/history/artifact`, not same-name tool; actual policy-owned download path exists. Variable model/workflow choices are constrained to reviewed graphs. |
| 44 | `list_files` | I bounded policy directory enumeration. |
| 45 | `memory_describe` | I/R `missy-history` retained exact-scope evidence metadata. |
| 46 | `memory_expand` | I/R immutable retained UTF-8 evidence paging; no re-execution. |
| 47 | `memory_search` | I/R legacy conversation recall maps to `history_search`, NOT native same-name explicit-record `memory_search`; exact-scope bounded literal history scan, not full FTS/semantic corpus. |
| 48 | `nomad_discover` | I/R `integrations/infra.rs`, configured nodes/jobs rather than unconstrained cluster. |
| 49 | `nomad_benchmark` | M durable template/runset/warmup/repetition/comparison lifecycle absent from generic adapter. Provider benchmark is unrelated. |
| 50 | `nomad_recommend_placement` | P/R Nomad server plan/placement exists for pinned job; legacy workload scoring/resource reasoning is broader. |
| 51 | `nomad_plan_job` | I/R plan pinned configured job through Nomad, not arbitrary generated specification. |
| 52 | `nomad_submit_job` | P/R deploy pinned job exists; protected owned-job exact-plan concurrency lifecycle not equivalent yet. |
| 53 | `nomad_job_status` | I/R filtered job status/allocation metadata. |
| 54 | `nomad_job_result` | P `NomadRead::Results` returns same allocation metadata as Allocations, not actual output artifact/log collection. |
| 55 | `nomad_job_action` | P/R generic Stop exists; full legacy owned-job action/reconciliation semantics not present. |
| 56 | `nomad_offload_task` | M durable plan/submit/status/result/cancel with template params/artifacts/idempotency missing. |
| 57 | `nomad_plan_scale` | P/R fixed count scale grant exists, not legacy fresh-discovery full-capacity+overlap optimistic scale plan. |
| 58 | `nomad_reconcile` | M remote owned-job journal reconciliation missing. |
| 59 | `nomad_schedule` | M durable fresh discovery/planning per finite batch occurrence missing; normal agent scheduler is not identical. |
| 60 | `provider_benchmark` | I/R `missy-plugins/src/benchmark.rs`, explicit budgeted NoTools runs/no replay; no automatic candidate promotion. |
| 61 | `rag_query` | I/R scoped lexical+hashing hybrid retrieval and optional neural embedding adapter; no global fallback. |
| 62 | `self_create_tool` | P/R model proposes data-only pinned native plugin manifest. No generated-source compiler/automatic executable creation. Exact-hash operator review/activation is deliberate; requiring an already-built native ELF narrows actual creation capability. |
| 63 | `shell_exec` | I/R exact argv allowlist, cleared env, bounded process group; optional Landlock/container. Arbitrary shell strings intentionally not compatible. |
| 64 | `singing_generate` | I/R actual ACE-Step Comfy generation, full mix, no fake TTS substitute; no isolated stems/phoneme timings. |
| 65 | `tts_speak` | P edge/Discord configured speech commands exist; local model-callable output/playback semantics not found at snapshot. |
| 66 | `audio_list_devices` | M local device enumeration tool not found. |
| 67 | `audio_set_volume` | M explicit volume operation not found. |
| 68 | `video_edit` | I/P native FFmpeg file editing trim/concat/transcode/overlay/extract and audio mux/mix/speed/text/crossfade; verify requested op/audio preservation individually. |
| 69 | `video_generate` | I/P approved graph submit/artifact path handles generation, but arbitrary legacy backend/audio_prompt composition choices need explicit workflow equivalents. |
| 70 | `video_storyboard` | P native assembles supplied image files; legacy also orchestrates scene generation. Composition of separate tools may replace it only when model can complete entire workflow. |
| 71 | `web_fetch` | I/P policy HTTP GET, UTF-8 body; safe custom headers/timeout and injection response handling differ. Security owner adding scans. |
| 72 | `x11_click` | I/R `desktop_input(kind=button)`, exact focused allowlisted window. |
| 73 | `x11_key` | I/R `desktop_input(kind=key)`, bounded native key events. |
| 74 | `x11_launch` | M no native launch operation found in x11.rs snapshot, despite older disposition prose. |
| 75 | `x11_read_screen` | P raw pixel screenshot + vision file analysis are separate. Missing integrated screen vision and native OCR coordinate anchors, blank-frame rejection equivalent not established. |
| 76 | `x11_screenshot` | I/R raw native X11 image metadata/base64, not automatically a PNG file vision can consume. |
| 77 | `x11_type` | P/R printable ASCII max256 focused-window typing; legacy broader text/Unicode via tooling. |
| 78 | `x11_window_list` | I/R `desktop_windows`, only explicit allowlisted windows. |
| 79 | `vision_burst` | M no series/burst operation found in capture or media. Multiple calls possible but not exact retained burst workflow. |
| 80 | `vision_capture` | I/P real live photo is `camera_photo`; same-name media `vision_capture` only copies supplied image FILE, never opens camera. |
| 81 | `vision_analyze` | I/R actual sanitized multimodal payload/provider call, 1–4 policy-read images and configured prices/budgets. |
| 82 | `vision_devices` | I/R `media/lib.rs` sysfs discovery, not capture crate. Details/preferred/diagnosis may differ. |
| 83 | `vision_scene` | M legacy scene-session create/add_observation/update_state/summarize/close absent. Native same-name performs one-image structured analysis only. This is not persistent scene memory. |
| 84 | `desktop_status` | P configured desktop windows/enumeration exist, full legacy availability/status aggregator absent. |
| 85 | `desktop_focus_window` | M x11.rs explicitly requires already-focused window, never changes focus. |
| 86 | `desktop_mouse_drag` | I/R `desktop_input(kind=drag)` bounded window-relative native input. |
| 87 | `desktop_mouse_move` | I/R `desktop_input(kind=move)`. |
| 88 | `desktop_launch_app` | M native explicit launch tool not found. |
| 89 | `install_software_confirmed` | M package installation operation not found; arbitrary shell restrictions are deliberate but not replacement capability. |
| 90 | `obs_status` | I/R `obs_read(action=status)` validates authenticated WS, filters stream/record state. |
| 91 | `obs_list_scenes` | I/R filtered scene list. |
| 92 | `obs_switch_scene` | I/R exact allowed scene + action + durable approval. |
| 93 | `obs_set_source_visibility` | I/R fixed scene:item SetSceneItemEnabled grant. |
| 94 | `obs_set_source_text` | M ObsMutate has no SetText/InputSettings variant at snapshot. |
| 95 | `obs_start_recording` | I/R explicit action grant + approval. |
| 96 | `obs_stop_recording` | I/R same. |
| 97 | `obs_start_streaming_confirmed` | I/R actual StartStream, exact approval and configuration, not implicit broadcast. |
| 98 | `obs_stop_streaming_confirmed` | I/R StopStream gated likewise. |
| 99 | `vtube_status` | I/R typed VTube read, preprovisioned token, no unsolicited pairing popup. |
| 100 | `vtube_load_model` | I/R exact model grant. |
| 101 | `vtube_list_models` | I/R configured filtered models. |
| 102 | `vtube_trigger_hotkey` | I/R exact hotkey grant. |
| 103 | `vtube_set_parameter` | I/R parameter allowlist and finite range. |
| 104 | `vtube_speak` | M synchronized playback/parameter speech not present; plain parameter injection is not speech. |
| 105 | `audio_route_tts` | M local sink-routing operation absent. |
| 106 | `audio_test_route` | M local route test/playback absent. |

Conditional extras: native `missy-specialty/src/foundry.rs` implements
`repoeval_foundry_read`/`repoeval_foundry_mutate` and conservative compatibility
`repoeval_foundry` with project/resource grants, staged plan/pins, policy HTTP
and durable outer approvals. No claim of live Foundry acceptance.

## Web, browser and documents

Actual legacy fetch is GET (not web search), capped streamed prefix, disallowed
credential/proxy headers, explicit redirect target without automatic follow,
and fail-closed high-confidence injection scanning (`web_fetch.py:125–136`).
Native `tools::web_fetch` gets a UTF-8 body and headers through the central
policy client; source/output framing alone does not match the scan. Native
`document_extract` in specialty reads UTF-8 text/Markdown headings/fences only;
it explicitly does not support PDF/OCR. Neither absent web search nor PDF
parsing should be invented as a legacy regression. Browser CDP currently
requires operator-provisioned endpoint/target, exact origins and independent
network isolation. This is an intentional security redesign, not automatic
browser launch/personal-profile reuse.

## Discord/channel production gaps

`crates/missy-gateway/src/discord.rs` has genuine policy-bound authenticated
Gateway intake, durable sequence/resume, heartbeat, bot/guild/channel/user
allowlists, request deduplication, scoped controls, bounded concurrent tasks,
nonce-bound outbox and unknown-send reconciliation. Source tests cover actual
mock WebSocket/HTTP restart flows. These are more than stub transports.

Still missing or narrower at snapshot:

* Inbound attachments are **metadata only**. No URL or bytes reach the model.
  Legacy `attachment_context.py` downloads allowed text/image/zip contents,
  sanitizes inline text and safely extracts zip entries (`zip_extract.py`).
  Outgoing multipart file tool does not solve this input gap.
* Native supports `/missy ask/status/cancel/resume`; legacy registers
  `/ask,/status,/model,/help`. Names can change, but connection/provider/help
  reporting differs. Legacy `/model` explicitly refuses switching: do not call
  absent dynamic switching a missing legacy feature.
* Legacy two-stage evolution review/apply reactions are durable in
  `channel.py:1899+`; no native reaction mapping found. Exact approvals through
  another operator interface are a safe alternative only if clearly exposed.
* Outgoing reply message references and typing indicators absent. Do not
  assume reply-to-bot intake without mention was legacy-supported: its guild
  mention gate is explicit. Native also requires mention in DMs, even with
  DM opt-in; that is an operational compatibility decision to document.
* Legacy keyed serial dispatch prevents same-conversation races while other
  channels proceed. Native dispatch was keyed by request ID, not conversation;
  concurrent same-session asks can conflict with runtime leases. Add per-key
  queueing, not global serialization.
* Native principal rate admission is fixed 30/min vs legacy configurable
  token bucket (default10/min). Native text outbox handles429 with persisted
  retries; generic lookup/upload tools do not share a complete proactive
  Discord REST bucket/global governor. Header-aware shared throttling,
  retry-after classification and bounded retry are production requirements,
  not proof supplied by a single429 test.
* Explicit allowlists/always-ignore bots/single account are safer but narrower
  than Python per-account guild policies, role modes and opt-in named bot
  interaction. Preserve explicit safe-chat/full authorization boundaries,
  rather than turning strict admission into accidental full-tool access.

## Security/provider/config/recovery inventory

| Area | Observed native source | Missing or deliberate difference |
|---|---|---|
| Default deny | Strict typed config, optional extensions, descriptor-root filesystem, explicit argv/origins, no implicit legacy HOME import | R strict exact origin/argv and independent state replace flexible shell/domain/CIDR/category preset syntax; migration must report every rejected policy. |
| Input/injection/secrets | Inert data envelopes; configuration fingerprints; `missy-security` vault/signatures | At initial snapshot no sanitizer/secrets detector/censor/quarantine. Security/runtime owners actively adding. Must cover launch/resume/provider inputs and outputs, not only `admit_task`: Discord directly calls Store::admit + Runtime::resume. |
| Prompt drift | Runtime configuration hash checked before execution/resume | I stronger immutable binding to system/config/providers/tools; does not by itself scan malicious recalled content. |
| Context | Immutable current request, whole call/result grouping, durable receipts and bounded verbatim compaction | I/R no claimed factual model summary of omitted blocks; requires retained source tools for details. Legacy per-entry history/memory injection quarantine still needs native equivalent. |
| Response guards | Native step/token/cost/deadline bounds and tool JSON-schema checks | Missing legacy action-only fabrication/promise/unfinished/explicit tool/delivery/verification guards at snapshot. Shared retry budget and preserve answer+notice on exhaustion matter; canned replacement is not parity. |
| Outbound policy | PolicyHttpClient for HTTP and WS, no automatic redirects, central DNS/address checks | I/R configured helper processes and independent browser/container network policy remain trust boundaries; not all subprocess network can be claimed centrally mediated. |
| Provider fallback | Native OpenAI/Anthropic/Ollama payloads, schema validation, fixed-order fallback and per-provider circuits | I fallback; missing weighted account round-robin/API-key pool rotation and managed OAuth refresh/account CLI workflow. Environment references are deliberate, not secret migration. Process provider NoTools must actually disable native tools or refuse. |
| Cost | Per-turn known price/token/cost/deadline limits, conservative uncertain billing after fallback | Verify global/session/subagent/schedule cumulative reservations; per-turn caps alone do not replace legacy global/session budgets. |
| Audit | SQLite journals, explicit signed export vault function, telemetry hooks | Signed export is not continuous signed/hash-chained audit events. Legacy AuditLogger supports chain verify/rotation/security views; native equivalent must be explicit. |
| MCP | Pinned manifest discovery/call checks, process env allowlist; HTTP extension | R no auto-unreviewed dynamic Python tool loading. Test real call-time revalidation/unknown effect behavior, not manifest parsing alone. |
| Recovery | Scoped durable turn/tool journals, leases, immutable checkpoint configuration, unresolved outcome blocks regeneration/replay | I fail-closed is correct. Need operator-visible reconciliation/disposition path and recover diagnostics, not manual SQL as normal runbook. Explicit retry may be appropriate only after evidence. |
| Vault/identity | ChaCha20-Poly1305 vault, private nofollow files, locks, Ed25519 signed export | I isolated native format; importing legacy identity/vault must never silently overwrite or regenerate missing keys. |
| Config migration | Copy-only explicit source/mapping paths and typed native target | Safe deliberate non-drop-in format. Cannot claim complete migration if jobs/devices/persona/patches/OAuth/accounts/raw history/graph/vector state are not either transformed or explicitly unsupported. |

Legacy security reference source: `missy/security/sanitizer.py` normalizes
Unicode/zero-width and deobfuscates base64/hex/ROT13 with second-signal
injection rules; `secrets.py` pattern scan/redact; `censor.py` last-resort
channel output redaction; `agent/context.py` per-entry quarantine;
`agent/response_guards.py` action guards. Counted patterns are not a quality
claim: benign/attack precision fixtures and integration coverage are needed.

## CLI and service inventory beyond tools

The authoritative legacy groups in `docs/reference/cli.md` include init/setup,
ask/run, providers list/switch/auth/oauth-accounts, skills scan, presets,
plugins/benchmarks, doctor/security scan, schedule, audit security/recent/
verify/export, gateway start/status, Discord status/probe/register/audit,
vault, approvals, patches, MCP management, devices/voice, config backups/
diff/rollback/plan, sandbox, API, evolve, sessions rename/cleanup, vision
diagnostics/capture/review/benchmarks, hatch, persona and cost/recover.

Native `Command` source has real init/config-check/migrate, ask/run/resume/
status/cancel/history/recover, serve/discord/discord-probe/discord-voice/voice,
doctor/vault, schedule, operations/workflow/mesh/retrieval/personal-memory,
evolve/agents/hatch/experiments/plugins/benchmark/approvals/persona. These are
not identical command names and need not be; operator auth, runtime wiring and
actual result semantics matter. Missing interactive setup/provider auth/key
pool administration, config backup/diff/rollback/hotreload, session rename/
cleanup and audit/vision diagnostic command equivalents are not replaced by a
larger top-level subcommand count.

Most importantly, legacy gateway startup **actually runs** proactive triggers,
watchdog provider/memory/MCP/DB health probes, scheduler, periodic agent
heartbeat, config hot reload, voice, screencast, console and Discord with
per-subsystem status (`missy/cli/main.py:3127–4198`). Native separate commands
do not automatically make a unified supervised production daemon. Need
explicit lifecycle composition/status/shutdown, scheduler ticking, proactive
and heartbeat policy/budgets, retention scheduling, provider auth refresh,
health reporting and systemd artifact. Protocol/lease heartbeat is not agent
heartbeat. Existing `examples/systemd/missy-gateway.service` is legacy.

## Acceptance needed before a production replacement claim

1. Close exact missing semantic rows, then inspect final application wiring.
2. Tests must exercise model-emitted tool call → actual production composition
   → durable policy/approval/journal → mocked external protocol → delivered
   result; direct crate functions alone miss disabled/unwired capabilities.
3. Security tests must cover initial input, recalled entries, tool/browser/web
   output, attachments, provider text/tool arguments, errors, queued recovery
   and outgoing text/artifacts without exposing secrets.
4. Disposable migration fixtures should represent real legacy schemas and
   protect original bytes/SQLite locks. Rejected fields must be reported.
5. Crash/retry tests need effect outcome unknown, scope conflicts, cancel,
   expired lease, config change, provider bill uncertainty, Discord ack expiry,
  429/global rate governor and no blind remote replay.
6. No live Robby machine/account/process/device acceptance was authorized or
   available. Lack of such acceptance is an honest operational limitation, not
   an excuse to label missing source behavior completed.
