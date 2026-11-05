# Legacy feature inventory and final disposition

Final disposition, 2026-11-05. The dated audit below is retained as discovery
evidence, **not the current missing-feature list**. Final test/build evidence is
in `validation.md`; remaining operational/parity limits are in
`completion-acceptance.md` and `parity.md`.

## Final disposition of discovered gaps

| Discovered area | Current native implementation |
|---|---|
| Vision payloads | Actual sanitized multimodal OpenAI/Anthropic/Ollama requests, strict scenes and persistent independent reservations in `missy-vision`; actual HTTP payload tests. |
| Generation artifacts | Approved Comfy graphs, pinned image conditioning and bounded authorized owned-artifact publication. MP4 validation is structural, not full decoding. |
| Editing semantics | Real FFmpeg audio mux/mix/loop, captions, speed, crossfade and preexisting operations. Storyboard still consumes supplied images. |
| Historical recall | Exact-scope previous-turn/checkpoint/tool/raw-message search, durable immutable SQLite receipts, exact UTF-8 describe/expand and separately budgeted NoTools summary. Executing turn's checkpoint excluded to avoid self-query false hits. Literal bounded search, not full-corpus semantic search. |
| File deletion | Checksum-bound descriptor-relative delete, exact durable outer approval and executable no-replay acceptance. |
| Personal recall | Separate enabled tenant+user cross-session/channel namespace; no global fallback, principal-local ranking and executable isolation/restart acceptance. |
| Tool creation/benchmark | Model-callable pending native-plugin proposals, exact-hash operator review/activation, new-runtime-only sandboxed execution. Explicit bounded NoTools provider benchmark with durable no-replay reservations. No arbitrary code compiler or automatic candidate promotion. |
| Discord upload/lookup | Policy-authorized user lookup and bounded multipart artifact delivery through `missy-discord-tools`. |
| Discord voice | Application-wired shared Gateway, consent/modes, separate actual NoTools safe runtime, native WS/UDP/Opus and official DAVE MLS. Real loopback/native-crypto roundtrip with mocked speech/agent, no live Discord claim. |
| Nomad/Incus | Typed extended operation/resource grants, Nomad nodes/placement/plan/scale/allocation, Incus discovery/operations/pinned exec/guest files. Exact durable approvals; fixture acceptance only. |
| Desktop/browser | Added bounded CDP wait/get-url/close, typed AT-SPI text/value and explicit X11 pointer/focus/launch grants. No active desktop use or arbitrary JavaScript. |
| Other native areas | Actual V4L2/isolated capture, score-to-WAV/MIDI, ACE-Step adapter, scoped Foundry, reviewed Git evolution, provider-selected DAGs and wired runtime telemetry. Per-crate READMEs state exact APIs and acceptance boundaries. |

Microphone capture, speaker routing/volume, synchronized VTube speech, OBS source
text and full deployment-specific drop-in parity remain explicit differences.
Codex completion is intentionally blocked without verified complete NoTools
controls. Hardware/live inference/GPU/infrastructure canaries are unperformed,
not fabricated as successful because local fixtures pass.

## Earlier read-only discovery snapshot, approximately 17:08 UTC

No builds, tests, live devices, credentials or services were used by this audit.
Its missing/in-progress language below is superseded by the final disposition.

## Counting and wiring

`missy/tools/builtin/__init__.py::_ALL_TOOL_CLASSES` registers **106 tools**;
configured Foundry adds `repoeval_foundry_read` and `repoeval_foundry_mutate`.
`repoeval_foundry` exists as a class but is not normally registered there.
Count a generic Rust adapter once, not once per legacy alias. Replacement names
and APIs are acceptable when actual capability exists.

The concrete executor is `BuiltinTools` plus CLI `CompositeTools`, not a type
literally named `RustToolExecutor`. `build_runtime` wires optional personal,
retrieval, HTTP MCP, integrations, desktop, media, capture and specialty tools.
An extension must be configured, and sensitive mutations require durable
approvals. Merely adding a crate is not wiring.

## Top completion priorities

1. **Vision needs actual provider image payloads.** Legacy
   `vision_analyze(image_path,prompt?,provider?,max_tokens?)` sends an actual
   image to a vision model; `vision_scene(action,...)` adds scene memory. Rust
   `Message.content` is a string and native provider adapters have no image
   serialization. No analysis executor is wired. Add bounded policy-authorized
   image ingestion, explicit multimodal messages, provider-specific image API
   serialization and capability checks, then the executor and optional scoped
   scene persistence. A path/base64 string in ordinary text is not vision.
2. **Generation must finish with a usable artifact.** `comfyui_submit(workflow,
   prompt?)` genuinely executes approved image/video graphs, while
   `comfyui_history(prompt_id)` returns validated metadata only. Legacy
   `image_generate(prompt,save_path?,...)` returns PNG paths and
   `video_generate(backend,prompt/image_path,save_path?,audio_prompt/audio_path?,
   ...)` returns MP4. Missing: owned bounded downloads, validation, policy-root
   publication and safe image-conditioning input. Parent integration work is
   **in progress**, not certified complete here.
3. **Shared media names conceal narrower semantics.** Rust `video_edit`
   supports trim, transcode/resize, concat, image overlay, frame/audio extract
   and storyboard, but video outputs are silent. Legacy audio replace/mix/loop,
   captions, speed and crossfade are missing at snapshot. Rust
   `video_storyboard(images,output,seconds_per_image,width,height)` assembles
   stills; legacy `video_storyboard(scenes,...)` also generates scenes. Parent
   media work is **in progress**.

## Concrete mapping

| Legacy exact names / semantics | Rust actual capability and residual gap |
|---|---|
| `file_read`, `file_write`, `list_files`, `shell_exec`, `web_fetch`, `calculator`, `file_delete` | First six implemented: `path`; write also `content`; shell exact `argv`; fetch `url`; calculator `expression`. No dedicated `file_delete`. Exact argv rather than arbitrary shell interpretation is deliberate. |
| `memory_search`, `memory_describe`, `memory_expand`, `context_shunt` | Same-name semantic trap: legacy search reads conversation turns/summaries; Rust search reads explicit `memory_put` records. Personal memory does not fix historical recall. No retained-output describe/expand or `context_shunt(item_ids,question,max_output_tokens?)` worker equivalent found. CLI history is inspection, not searchable model recall. |
| `rag_query`, `graph_query`; personal memory | Implemented `rag_ingest(id,path)`, `rag_query(query,limit?)`, `graph_query(start_id,depth?,limit?)`; scoped hybrid retrieval exists. `personal_memory_put/get/search/delete` implement explicit tenant+user memories across sessions, separate from history. |
| `delegate_task`, `code_evolve`, `self_create_tool`, `provider_benchmark` | `agents --path --request-id` runs bounded provider-selected DAGs; `evolve` supports propose/show/review/apply/rollback/audit/recovery. Underlying DAG/evolution capabilities implemented. Automatic prompt decomposition/diverse synthesis and dynamic tool creation/benchmark candidate lifecycle are not equivalent to operator plans/reviewed diffs. Arbitrary Python loading is deliberately excluded. |
| `vision_capture`, `vision_burst`, `vision_devices`, `vision_analyze`, `vision_scene` | Real `camera_photo(device,output,consent=true)`, `camera_clip(device,output,consent=true,duration_seconds?)` and isolated `screencast(output,consent=true,...)` implemented. Media `vision_capture(input,output)` only copies an existing file; it is not live capture. Discovery exists. Burst series, analysis and scene memory absent at snapshot. Audio recording intentionally refused. Untested hardware is acceptance pending, not implementation absent. |
| `image_generate`, `video_generate`, `video_edit`, `video_storyboard` | Approved graph execution and native editing exist; input/download/composition gaps above. Fixed approved workflow settings are a replacement API constraint, not proof generation is absent. |
| `singing_generate`; music creation | Actual ACE-Step 1.5 full-mix generation implemented and wired: generation needs `lyrics,tags,save_path`; `validate_only=true` probes readiness. No stems/phoneme timings. `music_compose(score,wav_output,midi_output)` renders real WAV/MIDI with deterministic synthesis, not model text-to-music. Legacy has no separate registered `music_generate`; soundtrack generation is inside `video_generate(audio_prompt,...)`. |
| `tts_speak`, `audio_list_devices`, `audio_set_volume`, `audio_route_tts`, `audio_test_route`; `discord_voice_join`, `discord_voice_leave`, `discord_voice_say`, `discord_voice_status` | Command STT/TTS and paired voice edge implemented. Native Discord voice service/protocol exists; main integration is concurrently owned by agent 1d51, so not certified wired. Local playback, volume/routing and synchronized VTube speech are separate semantics not proven by TTS. Live speech/Discord acceptance pending. |
| `discord_lookup_user`, `discord_upload_file` | Text Gateway/intake/outbox implemented, but no equivalent user lookup/binary-upload tool found. Incoming attachments are metadata only. Generated files need an authorized delivery path. |
| `browser_navigate`, `browser_click`, `browser_fill`, `browser_screenshot`, `browser_get_content`, `browser_evaluate`, `browser_wait`, `browser_get_url`, `browser_close` | Native isolated CDP navigate `url`, click `selector`, fill `selector,value`, screenshot and `browser_read()` implemented. Explicit wait/get-url/close absent; arbitrary evaluate deliberately restricted. Bounded fixed operations can close real workflow gaps without arbitrary scripting. |
| `atspi_get_tree`, `atspi_click`, `atspi_get_text`, `atspi_set_value`; `x11_screenshot`, `x11_click`, `x11_type`, `x11_key`, `x11_window_list`, `x11_read_screen`, `x11_launch`; `desktop_status`, `desktop_focus_window`, `desktop_mouse_drag`, `desktop_mouse_move`, `desktop_launch_app`, `install_software_confirmed` | `desktop_windows`, approved `desktop_screenshot(window)`, key/button `desktop_input(window,kind,code,pressed)`, `desktop_accessibility_tree(application,window,...)`, indexed `desktop_accessibility_action(application,window,path,action)` implemented. Not OCR, text/value interfaces, pointer move/drag, launch/install or focus-stealing. Active-desktop restrictions deliberate. |
| `obs_status`, `obs_list_scenes`, `obs_switch_scene`, `obs_set_source_visibility`, `obs_start_recording`, `obs_stop_recording`, `obs_start_streaming_confirmed`, `obs_stop_streaming_confirmed`, `obs_set_source_text` | Two tools `obs_read(action)`, `obs_mutate(action,...)` cover all listed except source text. Visibility needs `scene,item_id,visible`; switch needs `scene`; exact mutation approval required. Live acceptance pending. |
| `vtube_status`, `vtube_list_models`, `vtube_load_model`, `vtube_trigger_hotkey`, `vtube_set_parameter`, `vtube_speak` | Two tools `vtube_read(action)`, `vtube_mutate(action,...)` cover metadata/load `model`/trigger `hotkey`/set `parameter,value`. Synchronized `vtube_speak` absent. Live acceptance pending. |
| `nomad_discover`, `nomad_recommend_placement`, `nomad_plan_job`, `nomad_submit_job`, `nomad_job_status`, `nomad_job_result`, `nomad_job_action`, `nomad_plan_scale`, `nomad_offload_task`, `nomad_benchmark`, `nomad_reconcile`, `nomad_schedule` | Snapshot: two tools `nomad_read(action=list/status,job?)`, `nomad_mutate(action=deploy/stop,job)`. Placement/plan/scale/offload/result/benchmark/reconcile/scheduling orchestration not covered. Parent expansion **in progress**. |
| `incus_list`, `incus_info`, `incus_instance_action`, `incus_launch`, `incus_exec`, `incus_file`, `incus_snapshot`, `incus_config`, `incus_image`, `incus_network`, `incus_storage`, `incus_profile`, `incus_project`, `incus_device`, `incus_copy_move` | Snapshot: two tools `incus_read(action=list/status,instance?)`, `incus_mutate(instance,action=start/stop/restart/freeze/unfreeze)`. Remaining listed capabilities absent, not aliases. Existing no-exec/files/delete boundary explicit; extensions need typed policy and approvals. Parent expansion **in progress**. |
| `repoeval_foundry_read`, `repoeval_foundry_mutate` | Implemented specialty scoped list/plan/status/artifacts/compare/report and allowed snapshot/start/cancel. Start requires reviewed staging plan and project acknowledgement. No local untrusted repository execution. |

## CLI/features and boundaries

Native implementations include ask/interactive run, durable status/resume/
cancel/history/recover, migration, API/browser console, Discord text, voice
edge, vault, scheduling, workflows, mesh, retrieval, personal memory, reviewed
evolution, agent DAGs, hatching, persona proposals and A/B experiments.
Operations includes diagnostics, Prometheus, authenticated webhooks and wired
runtime telemetry configuration, not merely an exporter library. Older claims
in `parity.md` that these new capabilities are unported are stale.

Legacy interactive setup/provider auth administration, logs/cost/budget tools,
session rename/cleanup, config/persona backup-diff-rollback, vision diagnostics
and tool-candidate/benchmark administration do not all have matching Rust
interfaces. This is not evidence the underlying runtime capabilities are absent.

CLI provider mode intentionally uses `NoTools`; Codex completion remains
blocked without verified no-tools mode. Do not bypass that boundary, scrape
credentials, enable ambient hardware, restore arbitrary Python execution or
weaken approvals to chase labels. Authorized live acceptance is still required
after wiring. No full parity is claimed.

Evidence: legacy bootstrap/tool schemas and execute bodies; Rust `Message`,
executor `definitions` plus dispatch, CLI `build_runtime` and handlers,
Discord voice service, operations telemetry. Source presence is observed;
production readiness is not inferred.
