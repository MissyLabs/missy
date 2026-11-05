# Production replacement matrix: exact106 and operator/services

Source reconciliation, 2026-11-05 UTC, `rewrite/rust-runtime`, PR171. Supersedes
missing-feature conclusions in the earlier `production-parity-audit.md`, not its
historical evidence. Only this document was edited by this reviewer. No builds,
tests, accounts, credentials, live devices/services, commits or pushes. Test paths
are inspected evidence locations, not tests this reviewer executed.

**C** concrete complete native behavior/wiring; **R** actual replacement with
stated deliberate API/security restriction; **M** missing/partial behavior;
**PENDING** concurrent owner changes awaiting final inspection. None means live
acceptance or byte-compatible API. Optional means explicitly configured.

## Concrete source/config/test keys

Paths repository-relative. Config fields are `ExtensionConfig` in
`crates/missy-core/src/lib.rs`; actual registration is
`crates/missy-cli/src/main.rs::build_runtime_with_services` (`build_runtime`
delegates). `extensions.rs::CompositeTools` rejects duplicates and routes to owners.
Sensitive names require exact durable scope/turn/call/argument/config approvals;
startup rejects missing guard configuration. Orchestration is guarded after
wrapping; its base retains guarded nonrecursive tools. Referenced YAML content,
not just paths, is fingerprinted. Model consent never creates authority.

| Key | Actual source/config/main wiring | Inspected test paths |
|---|---|---|
| B | `crates/missy-tools/src/lib.rs`; always BuiltinTools, policy roots/origins/argv, optional sandbox_config | tools `tests/{security,file_delete,runtime_security,mcp_protocol,web_document}.rs` |
| D | `crates/missy-desktop/src/{lib,browser,atspi,x11,launch,screen}.rs`; desktop_config isolated display/CDP/bus/windows/origins; vision_config injects VisionInterpreter; specialty_config via CLI desktop_ocr.rs injects ScreenOcr | desktop `tests/{policy,xvfb}.rs`, inline browser/launch tests; CLI `tests/desktop_ocr_e2e.rs` |
| H | `crates/missy-history/src/{lib,sources,retention,summary}.rs`; history_config, private retained SQLite, optional priced NoTools summary | history `src/{tests,sources_tests,retention_tests}.rs` |
| I | `crates/missy-integrations/src/{lib,config,infra,incus_ops,comfy,desktop_ws,speech,scene}.rs`; integrations_config resources/graphs/grants/actions; media_config injects NativeArtifactValidator for external MP4 | integrations `tests/{mock_integrations,extended_integrations,nomad_service}.rs`, inline speech/comfy tests; media decoder test |
| W | `crates/missy-workloads/src/{lib,config,store}.rs`; workloads_config, explicit namespace/node/template grants, same Arc in Runtime/supervisor | workloads `tests/workloads.rs`, CLI `tests/workloads_production_e2e.rs` |
| S | `crates/missy-specialty/src/{lib,calculator,singing,music,foundry,documents}.rs`; specialty_config flags/backend/policy | specialty `src/tests.rs`, inline documents tests |
| V | `crates/missy-vision/src/{lib,scene_memory,monitor}.rs`; vision_config actual multimodal provider/private budget ledger | inline vision lib/monitor tests; providers `tests/http_fixtures.rs` |
| C | `crates/missy-capture/src/{lib,capture,config}.rs`; capture_config pins/device/output grants/consent+approval | capture `src/tests.rs` |
| F | `crates/missy-media/src/{lib,video,process,storyboard,validation}.rs`; media_config; scene_generation_config+I inject SceneAdapter | media `src/tests.rs`, including external_mp4_validation_requires_full_decodable_video; integrations extended scene test |
| A | `crates/missy-voice/src/audio_device.rs`; audio_devices_config Pulse socket/pinned ELF/source/sink/physical consent; voice_config STT/TTS | voice `tests/{audio_devices,commands}.rs` |
| Q | `crates/missy-discord-tools/src/{lib,inbound,governor}.rs`; discord_tools_config exact guild/channel/user/artifact grants | discord-tools `tests/rest.rs` |
| J | `crates/missy-discord-voice/src/{tools,service}.rs`; discord_voice_config, pending catalog then supervisor-retained/attached service/shared Gateway/full caller binding | discord-voice `src/{service_tests,tests,native_acceptance}.rs` |
| P | `crates/missy-plugins/src/{lib,proposal,source,benchmark,registry,sandbox}.rs`; plugins_config exact scope/IDs/private registry/separate source/compiler/ELF review | plugins `tests/{plugins,proposal,source}.rs`, inline benchmark tests |
| G | CLI `extensions.rs::RetrievalTools`, retrieval crate; retrieval_enabled/optional embedding_config/scoped DB/policy ingestion | retrieval `tests/retrieval.rs`, CLI `tests/extension_e2e.rs` |
| E | `crates/missy-evolution/src/tools.rs`; evolution_repo+evolution_config model-proposal wrapper | evolution `tests/workflow.rs`, inline tool tests |
| U | `crates/missy-subagents/src/{tools,pool,lib}.rs`; subagents_config distinct approved providers/limits/NoTools children | subagents `src/tests.rs`, CLI `tests/security_production_e2e.rs` |

Test abbreviations mean `crates/missy-NAME/PATH`, e.g. tools `tests/security.rs`
is `crates/missy-tools/tests/security.rs`. CLI is `crates/missy-cli`.

## Exact normally registered legacy106

Actual `_ALL_TOOL_CLASSES`: `missy/tools/builtin/__init__.py:258–365`, actual
class name attributes, registration order. Read-only Python AST reconciliation
verified106 rows,106 unique names and exact sequential registration order.
Keys supply each row's actual
config/main wiring/test paths. Conditional Foundry is separate, not among106.

| # | Exact legacy name | Actual native name/API; key | Disposition and behavior |
|---|---|---|---|
| 1 | `atspi_click` | `desktop_accessibility_action`; D | R typed allowlisted target/action. |
| 2 | `atspi_get_text` | `desktop_accessibility_get_text`; D | R explicit bounded target text. |
| 3 | `atspi_get_tree` | `desktop_accessibility_tree`; D | R explicit bus/application, bounded tree. |
| 4 | `atspi_set_value` | `desktop_accessibility_set_value`; D | R typed target/value+approval. |
| 5 | `browser_click` | `browser_click`; D | C fixed-selector bound-target click. |
| 6 | `browser_close` | `browser_close`; D | C explicit target close. |
| 7 | `browser_evaluate` | `browser_evaluate`, `browser_extract`, `browser_fill_many`; D | R fixed url/title/ready_state IDs and real typed extraction/form workflows. Arbitrary JS intentionally absent. |
| 8 | `browser_fill` | `browser_fill`, `browser_fill_many`; D | R native input/textarea setters/events, multi-field+submit, not all control types. |
| 9 | `browser_get_content` | `browser_read`, `browser_extract(mode=text/links/table/form,selector)`; D | R bounded DOM extraction, form values/passwords withheld. |
| 10 | `browser_get_url` | `browser_get_url`; D | C actual bound-target URL. |
| 11 | `browser_navigate` | `browser_navigate`; D | R exact origins/target+independent browser network isolation, no personal-profile reuse. |
| 12 | `browser_screenshot` | `browser_screenshot`; D | R bounded decoded PNG, native receipt shape. |
| 13 | `browser_wait` | `browser_wait`; D | R bounded selector visibility, not arbitrary Playwright predicates. |
| 14 | `calculator` | `calculator`; S | R finite real arithmetic, no Python eval/complex/bitwise. |
| 15 | `code_evolve` | `code_evolve`; E | R real title/diff proposal in fixed repo; operator review/apply/rollback separate, no self-deploy. |
| 16 | `context_shunt` | `context_shunt`; H | R separately priced/budgeted NoTools summary over immutable retained evidence, only advertised with backend. |
| 17 | `delegate_task` | `delegate_task`; U | R explicit tasks OR prompt decomposition, diverse specialists/synthesis/distinct allowed providers/three failure policies. Operator-fixed policy, depth1/15 tasks, reasoning-only children. |
| 18 | `discord_lookup_user` | `discord_lookup_user`; Q | R real scoped allowlisted lookup/REST governor. |
| 19 | `discord_upload_file` | `discord_upload_file`; Q | R real bounded multipart artifact delivery/exact approval/nonces. |
| 20 | `discord_voice_join` | `discord_voice_join`; J | R exact preconsented caller binding+durable approval, no model actor. Parent reports focused service suite20 passed/2 ignored. |
| 21 | `discord_voice_leave` | `discord_voice_leave`; J | R actual bound session leave+approval, no implicit retry. |
| 22 | `discord_voice_say` | `discord_voice_say`; J | R actual TTS queue, queued not heard/played, no autojoin. |
| 23 | `discord_voice_status` | `discord_voice_status`; J | R attached service/exact caller state, no capture/join. |
| 24 | `file_delete` | `file_delete`; B, policy.allow_file_delete | R expected SHA256+descriptor quarantine/recheck+durable approval, no recursive broad deletion. |
| 25 | `file_read` | `file_read`; B | R bounded policy UTF-8, raw evidence/model projection separate. |
| 26 | `incus_copy_move` | `incus_operation(grant)` Copy/Move; I | R pinned source/target/project, actual routes. |
| 27 | `incus_config` | `incus_operation` Update; I | R pinned typed resource definition. |
| 28 | `incus_device` | `incus_operation` DeviceSet/DeviceDelete; I | R exact device/resource grants. |
| 29 | `incus_exec` | `incus_operation` Exec; I | R pinned guest argv, real async wait AND stdout/stderr fetch; extended_integrations.rs::incus_exec_wait_recorded_logs_and_descriptor_confinement. |
| 30 | `incus_file` | `incus_file_read`, `incus_operation` FileWrite; I | R exact text paths, not arbitrary binary guest access. |
| 31 | `incus_image` | `incus_discover`, `incus_operation` Image; I | R fingerprints, typed Create/Delete grants. |
| 32 | `incus_info` | `incus_read(status)`, `incus_discover(inspect)`; I | R configured instances/resources. |
| 33 | `incus_instance_action` | `incus_mutate(start/stop/restart/freeze/unfreeze)`; I | R allowed non-forced lifecycle. |
| 34 | `incus_launch` | `incus_operation` Create Instance; I | R pinned definition/image. |
| 35 | `incus_list` | `incus_read(list)`, `incus_discover(list)`; I | R allowed resource discovery, no project escape. |
| 36 | `incus_network` | `incus_discover`, `incus_operation` Network; I | R actual pinned Create/Update/Delete. |
| 37 | `incus_profile` | `incus_discover`, `incus_operation` Profile; I | R actual pinned Create/Update/Delete. |
| 38 | `incus_project` | `incus_discover`, `incus_operation` Project; I | R exact project authority. |
| 39 | `incus_snapshot` | `incus_discover(snapshots)`, `incus_operation` SnapshotCreate/Restore/Delete; I | R actual exact snapshot routes. |
| 40 | `incus_storage` | `incus_discover`, `incus_operation` StoragePool/StorageVolume; I | R exact reviewed definitions. |
| 41 | `file_write` | `file_write`; B | R strict schema/policy atomic replacement, not content CAS. |
| 42 | `graph_query` | `graph_query(start_id,depth,limit)`; G | R bounded scoped traversal, no global arbitrary query. |
| 43 | `image_generate` | `comfyui_submit`→`comfyui_history`→`comfyui_artifact`; I | R actual approved graph+owned descriptor+decoded image publication, named pinned conditioning. No invented alias/arbitrary backend choice. |
| 44 | `list_files` | `list_files`; B | C bounded allowed listing, recovery journals hidden. |
| 45 | `memory_describe` | `memory_describe`; H | R exact-scope immutable retained metadata. |
| 46 | `memory_expand` | `memory_expand`; H | R retained UTF-8 paging, no re-execution. |
| 47 | `memory_search` | `history_search`; H | R actual prior conversation/checkpoint/tool/raw recall. Native builtin memory_search is explicit records, NOT this alias; literal bounded, not semantic corpus search. |
| 48 | `nomad_discover` | `nomad_read(nodes/node/list/status)`; I; W planning discovery | R real configured namespace/node/job HTTP. |
| 49 | `nomad_benchmark` | `nomad_benchmark`; W | R durable version/runset/warmup/repetition/parallel comparison, no replay, not provider benchmark. |
| 50 | `nomad_recommend_placement` | `nomad_read(placement)`, `nomad_offload_task(plan)`; I/W | R actual server planning plus W resource/allocation headroom, not unrestricted scoring. |
| 51 | `nomad_plan_job` | `nomad_mutate(plan)`, `nomad_offload_task(plan)`; I/W | R actual owned pinned-service dry-run, fresh capacity/ownership/index checks and one-use300s plan; arbitrary specs forbidden. |
| 52 | `nomad_submit_job` | `nomad_mutate(deploy,job,plan)`, `nomad_offload_task(submit)`; I/W | R service grant pins owner/expected index; fresh recheck and EnforceIndex CAS, consumed plan/attempted-index latch, no implicit retries. W adds durable unique-batch journals. Service plans/latches process-local, unknown-effect operator reconciliation required across restart. |
| 53 | `nomad_job_status` | `nomad_read(status/allocations)`, `nomad_offload_task(status)`; I/W | R real filtered status and scoped workload lifecycle. |
| 54 | `nomad_job_result` | `nomad_offload_task(result)`; W | R stdout/stderr, typed MISSY_RESULT_JSON, declared output GET+SHA256. I nomad_read(results) is allocation metadata, NOT equivalent. |
| 55 | `nomad_job_action` | `nomad_mutate(plan_stop/stop/plan_restart/restart)`, `nomad_offload_task(cancel)`; I/W | R actual owned CAS Stop:true and rolling replacement restart stamp after exact reviewed plan. Purge deliberately forbidden, not implemented by stop. |
| 56 | `nomad_offload_task` | `nomad_offload_task(plan/submit/status/result/cancel)`; W | R actual durable templates, optimistic unique jobs, result artifacts/logs, uncertain submit never replayed. |
| 57 | `nomad_plan_scale` | `nomad_mutate(plan_scale,grant)` then `scale(grant,plan)`; I | R full pinned graph count change, fresh allowed-node/allocation headroom, all replacement replicas+overlap, real dry-run and CAS. No longer unfenced direct scale. Conservative capacity may reject feasible incremental rollouts. |
| 58 | `nomad_reconcile` | `nomad_reconcile`; W | R remote owned-task comparison, no unknown/missing/tampered adoption, not generic service ownership. |
| 59 | `nomad_schedule` | `nomad_schedule`; W | R finite durable scoped occurrences/fresh plans/actual worker/pause+tombstones/no duplicate restart. |
| 60 | `provider_benchmark` | `provider_benchmark`; P | R actual priced NoTools/durable reservations, no automatic promotion. |
| 61 | `rag_query` | `rag_query`; G | R scoped BM25+hashing hybrid/optional explicit neural backend, no global fallback. |
| 62 | `self_create_tool` | `self_create_tool`; P; plugins source-* CLI | R actual Rust source OR manifest proposal→separate exact-hash source review→pinned offline confined build→ELF review+enable. No automatic Python execution/approval/mid-turn catalog growth. |
| 63 | `shell_exec` | `shell_exec(argv)`; B | R exact argv/clear env/bounded process group/optional sandbox, no arbitrary shell authority. |
| 64 | `singing_generate` | `singing_generate`; S | R real ACE-Step full mix/download/decode, not TTS; no stems/phoneme timing claim. |
| 65 | `tts_speak` | `audio_device(speak)`; A | R real configured TTS+approved sink playback/capped stream volume, requires audio+speech config. |
| 66 | `audio_list_devices` | `audio_device(discover,direction)`; A | R actual granted metadata only, no ambient capture. |
| 67 | `audio_set_volume` | `audio_device(set_volume)`; A | R exact sink/cap1..80%, no default/output-client/implicit unmute changes. |
| 68 | `video_edit` | `video_edit`; F | C actual FFmpeg trim/transcode/resize/concat/overlay/extract/audio replace+mix+loop/captions/speed/crossfade, real decoded fixture outputs. |
| 69 | `video_generate` | `comfyui_submit/history/artifact`; I+media_config F decoder | R actual graph+owned MP4 publication, full bounded sandbox FFmpeg decoder required before publish. Header-only MP4 not sufficient. Without configured validator MP4 download refuses. Arbitrary backend/audio_prompt orchestration not automatic; reviewed graph or separate audio+edit workflow. |
| 70 | `video_storyboard` | `video_storyboard`, `video_generate_storyboard`; F+I | R supplied images OR actual generated scenes through approved injected backend→decoded MP4. Without generator only image assembly. |
| 71 | `web_fetch` | `web_fetch`; B | R actual policy GET/no redirects/bounded UTF-8 raw evidence+safe model projection; native header/schema differences. |
| 72 | `x11_click` | `desktop_input(kind=button)`; D | R native bounded allowed focused-window input. |
| 73 | `x11_key` | `desktop_input(kind=key)`; D | R native bounded keys/chords. |
| 74 | `x11_launch` | `x11_launch(app)`; D | R pinned app key/argv/env, isolated session/supervised children, no PATH search. |
| 75 | `x11_read_screen` | `x11_read_screen`; D+V and/or specialty_config OCR | R actual native pixels→PNG→vision and/or original-dimension Tesseract TSV word boxes/centers. RGB stddev<1.5 blank refusal precedes OCR/billing. main desktop_ocr adapter injects pinned sandbox OCR, errors labelled unavailable not invented anchors. |
| 76 | `x11_screenshot` | `desktop_screenshot`; D | R native explicit-window pixels/metadata, no ambient whole desktop, result differs. |
| 77 | `x11_type` | `desktop_input(kind=type_text)`; D | R1..256 printable ASCII, broader Unicode/delay contract intentionally not reproduced; other typed targets not generic Unicode X11 input. |
| 78 | `x11_window_list` | `desktop_windows`; D | R configured windows only. |
| 79 | `vision_burst` | `vision_burst`, `camera_burst`; C | C actual1..16 unique-output frames/exact burst approval/drift checks/cancel, not old-file copies. |
| 80 | `vision_capture` | `camera_photo`; C | R actual V4L2 photo; F same-name vision_capture copies/normalizes existing FILE, never a camera substitute. |
| 81 | `vision_analyze` | `vision_analyze`; V | C actual sanitized1..4-image provider payloads/prices/reservations/strict responses. |
| 82 | `vision_devices` | `vision_devices`; F | R actual sysfs discovery, not full preferred/health/quality diagnostic CLI. |
| 83 | `vision_scene` | `vision_scene(create/add_observation/update_state/summarize/close)`; V | C real scoped SQLite sessions/append-only observations/state/close enforcement; action-less image analysis alone not counted as memory. |
| 84 | `desktop_status` | `desktop_status`; D | R actual typed configured isolation/capture/input/vision/OCR/launch capabilities, isolated authenticated X11 handshake; browser/AT-SPI explicitly not_probed, no ambient discovery/input. |
| 85 | `desktop_focus_window` | `desktop_focus_window`; D | R actual native raise/set/verify in isolated explicit window. |
| 86 | `desktop_mouse_drag` | `desktop_input(kind=drag)`; D | R bounded window-relative native drag. |
| 87 | `desktop_mouse_move` | `desktop_input(kind=move)`; D | R bounded native motion. |
| 88 | `desktop_launch_app` | `desktop_launch_app(app)`; D | R actual pinned supervised operator-keyed launch. |
| 89 | `install_software_confirmed` | `install_software_confirmed(package)`; D | R actual pinned package command/wait+approval, no auto-sudo/arbitrary package names; tests harmless ELF, not real install. |
| 90 | `obs_status` | `obs_read(status)`; I | C actual authenticated WS recording/stream state. |
| 91 | `obs_list_scenes` | `obs_read(list_scenes)`; I | R configured scene metadata. |
| 92 | `obs_switch_scene` | `obs_mutate(switch_scene)`; I | R exact scene/action approval. |
| 93 | `obs_set_source_visibility` | `obs_mutate(set_visibility)`; I | R exact scene:item SetSceneItemEnabled. |
| 94 | `obs_set_source_text` | `obs_mutate(set_source_text)`; I | R exact text-source/text-only SetInputSettings, no arbitrary settings. |
| 95 | `obs_start_recording` | `obs_mutate(start_recording)`; I | R actual explicit grant/approval, no implicit recording. |
| 96 | `obs_stop_recording` | `obs_mutate(stop_recording)`; I | R actual approved stop. |
| 97 | `obs_start_streaming_confirmed` | `obs_mutate(start_streaming)`; I | R real StartStream/exact approval, no model broadcast authority. |
| 98 | `obs_stop_streaming_confirmed` | `obs_mutate(stop_streaming)`; I | R actual approved StopStream. |
| 99 | `vtube_status` | `vtube_read(status)`; I | R actual typed authenticated read/preprovisioned token/no popup. |
| 100 | `vtube_load_model` | `vtube_mutate(load_model)`; I | R exact model grant/approval. |
| 101 | `vtube_list_models` | `vtube_read(list_models)`; I | R configured filtered metadata. |
| 102 | `vtube_trigger_hotkey` | `vtube_mutate(trigger_hotkey)`; I | R exact hotkey grant/approval. |
| 103 | `vtube_set_parameter` | `vtube_mutate(set_parameter)`; I | R finite allowed parameter range. |
| 104 | `vtube_speak` | `vtube_speak`; I+A+voice_config | R real PCM playback+50ms RMS mouth envelope/process-start sync/cancel reset, advertised only with backend; not bare parameter injection. DAC timing external acceptance. |
| 105 | `audio_route_tts` | `audio_device(create_route)`; A | R actual virtual null sink/capped volume; no default sink change/implicit unmute, monitor setup remains operator. |
| 106 | `audio_test_route` | `audio_device(test_route)` then optional `audio_device(speak)`; A | R metadata health+separately approved real audible test. test_route alone not playback. |

## Non-tool production/operator workflows

Working tools are not absent services. Inspection commands are not model tools.

| Area | Actual source/wiring/test evidence | Disposition and boundary |
|---|---|---|
| Unified daemon | CLI main gateway/discord/discord-voice/voice-serve→production.rs; HTTP/Discord/edge voice/operations/scheduler/mesh/triggers/watchdog/workloads; CLI tests/{production_e2e,watchdog_e2e,workloads_production_e2e}.rs | C owner lock/authenticated readiness/startup deadlines/failure propagation/SIGTERM+SIGINT bounded drain/status. Not installed cutover. |
| Reload/config | production.rs fingerprint/check_changes, gateway check-reload, executable reload fixture | R validated restart-required/false readiness, not in-place reload. Referenced YAML content included; env rotation also restart. |
| Packaging | packaging/rust/{release.sh,missy-native.service}, install.sh; production temp-prefix fixture | R checksummed staging/activate/rollback slots, no automatic live restart. Binary rollback not schema/state downgrade. |
| Scheduler/typed plans | automation src/{scheduler,execution,tools}.rs, automation_config/gateway.scheduler trusted scopes; automation tests/production.rs | C exact calls/JSON refs/conditions/timezone schedules/durable no-replay, explicit approval resume/reconcile, not shell interpolation. |
| Proactive/heartbeat | automation triggers.rs/observations.rs, triggers_config/supervisor polling, production tests; CLI triggers receipts/reset | R consented scoped heartbeat/file/disk/load/idle/watchdog, trusted collectors/cooldown/active hours/runtime budgets. Paused approval/inflight/failed occurrences remain scoped nonfatal receipts, reconcile actual Store completion without replay; explicit reset rejects active turns and unresolved effects. Protocol heartbeat not agent heartbeat. |
| Idle maintenance and relevant memory | automation maintenance.rs; runtime memory_context.rs/prompts.rs; Store memory_candidates; learning-production.md | R once-only extractive digest/tool-outcome-learning with provenance; opt-in namespace-local relevance and maintenance fallback snapshotted at admission as bounded sanitized untrusted DATA. Generated speculative abstraction/automatic graph/entity/embedding indexing is replaced by provenance-bearing retrieval, not claimed as model-generated synthesis or instruction promotion. |
| Watchdog | CLI watchdog.rs/watchdog_config, watchdog+production executable fixtures | R actual DB quick_check/HTTP2xx/durable health+recovery/consented escalation. Dedicated provider/memory/MCP statistics not separately reproduced; no automatic repair. |
| Browser screencast | crates/missy-screencast/src/{lib,server,image_frame,vision}.rs/bundled capture page; CLI screencast_adapter.rs/extensions.screencast_config; production.rs awaits bound server before ready; screencast tests/channel.rs | R actual getDisplayMedia→HTTP/WS frames/private single-use scoped pairing/bounds/expiry/cancel/change threshold/governed vision. Loopback-only fixed origin. CLI approve/status/cancel/durable mailbox. Private JSON/optional PNG delivery, not implicit Discord posting. Ten native tests include opt-in real Chromium synthetic canvas; three durable operator adapter tests. |
| Discord intake | gateway discord.rs/discord/{attachments,journal}.rs, inline tests/Q REST tests | C authenticated sequence/resume/immutable attachment+reply context/FIFO conversation asks/control bypass/exact private raw evidence; opted-in DMs need no mention. |
| Discord delivery/control | gateway+Q governor.rs; discord-outbox/register-commands; production offline outbox fixture | R shared bucket/global429 headers/nonping referenced replies/nonce+result-bound outbox/exact reconcile or abandon without resend. /missy ask/status/cancel/resume/help replaces flat names. Real operator approval replaces reaction authority. |
| Discord authorization | core DiscordConfig/gateway admission | R explicit single-bot numeric allowlists/ignore bots, not legacy multi-account per-guild role-mode/named-bot matrix. Strict full-runtime allowlisting not selective safe-chat/full role routing. |
| Incoming documents | text/ZIP enrichment, Q discord_fetch_attachment exact IDs, V analysis, S document_extract/document_pdf_extract/document_ocr | R private bytes/safe in-memory archive/explicit binary download/real PDF+Tesseract. No model source-URL authority. PDF/OCR additions not106; scanned PDF rasterization not automatic. |
| Provider transport | providers src/{lib,stream}.rs, ProviderConfig/http_fixtures.rs | C OpenAI/Responses/Anthropic/Ollama actual schemas/NoTools/complete SSE+NDJSON+usage/no partial-tool dispatch. Durable completion, not partial-token UI. |
| Accounts/OAuth | providers oauth.rs, CLI provider_ops.rs, oauth_fixtures.rs | R env key pools/round-robin/in-process cooldown/locked vault refresh/persisted rotation; list/check/keypool/oauth-refresh CLI. No interactive login/device/browser provisioning/legacy discovery/CLI subscription conversion. |
| Routing/rates | providers router.rs/core RoutingConfig/CLI provider_ops.rs; main full+safe pools with_routing; routing-production.md/provider+CLI tests | R weighted priorities/task+tier/rules/RPM+TPM/concurrency/circuit/wait bounds. Operator metadata, not model authority. Process-local state; safe switch plans or writes config for restart, no hot swap. Parent reports providers38+CLI3 pass. |
| Process provider | provider-process crate/cli_provider_config/chat-only main/process fixtures | R complete verified NoTools or refusal, no ignored extension powers. Codex CLI blocked; Responses not CLI OAuth parity. |
| Budgets/concurrency | store budget.rs/operator.rs, Runtime/U and W/V/H ledgers; global_capacity/operator_budget/runtime production tests | R shared fenced leases/lifetime cumulative cost+tokens/unknown-billing reservations/per-turn caps. Lifetime stricter than daily/monthly reset; independent legacy session reporting/unified all-extension bill not claimed. |
| Security/context | security content.rs/runtime context.rs+guards.rs/policy; tools runtime_security/CLI security_production_e2e | R recursive input/history/tool/replay projection/raw evidence separation/output censor/prompt binding/shared bounded guard retries. Heuristics not universal injection firewall. |
| Policy/sandbox | policy/sandbox crates/exact roots+argv+origins/outer governance/tests | R default deny/descriptor files/DNS-pinned HTTP+WS/no redirects. Domain/category/preset/arbitrary-shell grammar incompatible by design. Helpers/browser/remote containers need independent isolation. |
| Sessions/recovery | CLI operator.rs/store operator.rs/operator_e2e/history/recover/resume/cancel; admin.rs/store admin.rs/main Admin dispatch/admin_e2e.rs | R exact scope display-name/logical clear/retention dry-run/effect+billing evidence reconciliation/abandon/no replay. Admin inventories exact tenant/user/channel+prefix, bounded, no memory-only sessions. Display-name not scope rekey. Parent reports2 actual admin executable tests pass. |
| Audit/admin scan | Store chain/operator export/vault/operator tests; CLI admin security-scan/admin-production.md | R verified chains/pinned Ed25519 snapshots, not continuous event signatures or consolidated subsystem logs. Real offline installation metadata/capability scan exists, not proof executable integrity/parent ancestry/live safety. |
| Vault/API/devices | native vault+identity/gateway auth/voice pairing/mesh grants/tests | R encrypted native formats/explicit pairing+revocation, no legacy identity/ambient device trust import. Arbitrary token-admin UI not claimed. |
| MCP/skills/plugins | B stdio MCP/http_mcp_config/P reviewed-source+binary; CLI admin mcp-plan/mcp-apply and skills scan | R pinned discovery/call/source creation and explicit manifest-bound config plan/apply/snapshot. Bounded descriptor-relative SKILL.md scan exposes hash/provenance and inert content. No server launch/contact during admin; existing argv grant required. Automatic Python loading/unreviewed dependencies excluded. |
| Evolution/persona | E/governance/evolve+persona+hatch+experiments; runtime ScopedPromptResolver and CLI prompt_resolver | R hash/HEAD-pinned apply+rollback/prompt revisions, no self-deploy; operator CLI replaces reactions. All shared production channels resolve exact-scope revisions at admission; immutable prompt/memory pins survive queueing and continuation. Native daemon HTTP fixture proves pause/review/resume/new-turn and cross-session isolation. |
| Vision diagnostics | C/V/D/S actual tools/general doctor | R partial operator surface, not full legacy vision inspect/doctor/health/benchmark/validate/memory CLI suite. D status is actual isolated-backend capability report, not ambient discovery. |
| Setup/config admin | init/config-check/config_ops.rs validate+backup+diff+plan+rollback/production fixtures | R private atomic main-YAML snapshots/redacted plans/restore/no autorestart. Extension/vault/env/DB excluded; no interactive auth wizard. |
| Telemetry/operations/mesh | telemetry_config/operations_config/mesh_config/hooks+supervised listeners/respective tests | C scoped diagnostics/webhooks/telemetry/signed mesh dispatch through shared Runtime; no live deployment claim. |

Conditional Foundry: S exposes repoeval_foundry_read/repoeval_foundry_mutate and
conservative repoeval_foundry with exact grants/pins/staged plans/outer approvals.
Not among106. Native web_search/read_document/PDF/OCR/schedule_task are additions,
not fictitious missing normally registered legacy tools.

## Config/state conversion, not an alias layer

See migration-production.md/config-conversion.md. Explicit offline sanitized
exports only, no ~/.missy discovery/live config.d merge/writer stop/vault decrypt/
bot activation. Tests: migrate tests/{migration,production,config_production}.rs,
CLI tests/migration_e2e.rs.

| Input | Actual conversion and boundary |
|---|---|
| Config | Supported typed provider order/env refs/bounds/safe roots+argv+origins convert. Unknown active constraints reject or required inert review templates. Unsupported selected default disables pool, never changes accounts. |
| Grants/isolation | No flattening domain/category/REST or cross-product guild grants. Required sandbox clears local grants until reviewed. Templates not executable configs. |
| History | Actual schema/one-to-one legacy session→full Scope/inert untrusted provenance/zero resumable turns. Old system/tool roles never native instructions. |
| Jobs/persona/graph/trust | Originals+disabled reviews; explicit state: mappings add real PAUSED schedules/PENDING persona/scoped nodes+edges/conservative advisory trust. Unsupported fields preserved/quarantined. |
| Effects/cost/checkpoints | Preserve/quarantine, never fabricate receipts. Reconcile remote jobs/reservations/unknown outcomes before activation; old spend cannot become fresh allowance. |
| Credentials/devices | Known secret paths metadata-only, no content read/hash/copy. Independently retain encrypted originals, explicitly reauthenticate/re-pair native formats. |
| Plugins/indexes/media/unknowns | Private inert preservation/review/rebuild, no Python/pickle execution. Full regular-file inventory, nonregular/oversize input fails closed or explicit export resolution. |
| Publish/rollback | Sealed plan/source checks/private staging/integrity/atomic no-replace/offline verify/non-destructive rename rollback, not activation/restart/source deletion/undo of new external effects. |

## Source reconciliation and acceptance

The exact106 inventory was checked against the actual Python registration AST:
unique names, registration order and numbering match. This is a behavior mapping,
not a claim that106 independent live integration scenarios were executed.

Final source reinspection includes durable planner cancellation/distinct worker
providers, reviewed Rust source plugins, locked OAuth, model voice controls,
owned Nomad service CAS/rolling lifecycle, desktop status/blank guard/native OCR,
full bounded external MP4 decode, provider routing/admin, browser screencast,
shared production prompt snapshots, automatic opt-in scoped memory context,
operator skill scanning and nonfatal trigger recovery. Audit findings for newest
history retention and remote installer branch selection were fixed and covered
by focused actual fixtures. Final integrated evidence is recorded in
`production-validation.md`, distinct from earlier concurrent snapshots.

Old transient assertions are not proof of current failure; earlier green counts
are not proof final integrated tree passes. Strict native replacements may be
accepted without pretending byte-identical compatibility.

Actual target permissions/OAuth issuer/audio/devices/Discord+DAVE quality/browser
isolation/GPU graphs/package install/Nomad+Incus+OBS+VTube endpoints are external
deployment acceptance, not code gaps merely because no Robby access occurred.
Conversely Robby access is not needed to implement actual missing source behavior.
