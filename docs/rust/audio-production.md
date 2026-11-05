# Native audio, avatar speech and Discord voice

## Implemented surfaces

- `missy_voice::AudioDeviceTools` exposes the typed `audio_device` tool:
  `discover`, metadata-only `test_route`, `create_route` (null sink only),
  `set_volume`, synthesized `speak`, and bounded `listen` -> STT.
- `AudioDeviceConfig` is default disabled. Enabled configuration requires an
  explicit local `unix:/...` PulseAudio/PipeWire-pulse server and hash-pinned
  native ELF `pactl`; playback/input additionally need pinned `paplay`/`parec`.
  No subprocess inherits DISPLAY, runtime directory, PULSE_SERVER or PATH.
  Exact granted source/sink names are mandatory. Discovery lists only granted
  metadata and does not open any microphone. A configured `Physical` device
  separately requires `physical_input_consent` / `physical_output_consent`.
- Every action consumes durable, scope/turn/call/arguments-bound host approval
  via `AudioApprovalGate`, including metadata requests. Model `confirm` fields
  are rejected. No default sink/source changes, other-client moves or implicit
  unmute. Sink volume is capped at configured 1..80 percent, playback uses its
  own stream volume, and null-route creation refuses a pre-existing non-null
  driver before setting its volume. Capture is bounded to 1..30 seconds, PCM
  16kHz mono, and passed directly to STT, not published as a recording.
- `obs_mutate` action `set_source_text` requires `ObsAction::SetSourceText` and
  an exact `ObsConfig.text_sources` grant. Only `inputSettings.text` with
  `overlay=true` is sent. No arbitrary OBS settings passthrough.
- `IntegrationTools::with_speech(tts, output)` advertises `vtube_speak` only
  with an actual synthesis/output backend. Strict PCM WAV parsing computes a
  50ms RMS envelope, compresses it into 0..0.85, includes final partial chunks
  and preserves silence. It authenticates one policy-pinned VTube socket,
  waits for approved playback's process-start signal, uses monotonic deadlines,
  skips stale animation frames and performs bounded mouth-zero cleanup on
  completion, error or cooperative cancellation. A parameter grant must include
  the complete 0..0.85 range. `SpeechOutput` is a privileged host boundary: its
  one durable exact `vtube_speak` approval covers output and animation together.
  IntegrationTools does not consume that same single-use approval twice.
- `DiscordVoiceService::join_cancellable` cancels negotiation/readiness and the
  resulting session. `shutdown` cancels all sessions first, cooperatively waits
  up to three seconds per task, then aborts uncooperative tasks. Gateway leave
  is bounded. WebSocket resume has three bounded transport-only attempts,
  preserves RTP/DAVE counters/replay state, and never restarts an agent turn or
  automatically rejoins.
- `speak_text` requires an actor binding and an already joined approved session;
  synthesis and a two-item output queue are cancellation/deadline bounded.
  `DiscordVoiceSayTools` exposes protected `discord_voice_say`: actor is resolved
  from one exact consented binding matching the four-component caller scope,
  never accepted from model arguments. Exact durable approval is mandatory.
  Receipts say queued, not played. `pending(config, gate)` plus one-time
  `attach(service)` supports runtime-before-service construction without changing
  the tool fingerprint; this attach stores a weak service reference to avoid a
  runtime/service ownership cycle. The lifecycle owner retains the service.
- `IntegrationTools::approved_scene` supplies the approved ComfyUI storyboard
  backend. This trusted host method is not a tool; call only after media consumes
  exact parent-call approval. Operator-owned `SceneGenerationGrant` pins workflow,
  SaveImage output node and dimensions. One submit, bounded polling and exact
  scope-owned image download/decode; cancellation never invokes global Comfy
  interrupt or retries an uncertain submit. Dimensions are verified against the
  downloaded image, not assumed from server claims.

## Composition

Construct one `Arc<AudioDeviceTools>` with durable gate and explicit STT/TTS.
Register it as a ToolExecutor and pass the same Arc as `SpeechOutput` to the
integrations object. The output policy fingerprint is included in integration
lineage. Add `AudioDeviceConfig` to native application configuration; never infer
it from legacy desktop environment variables. Main/core/extensions composition
is owned separately from these crates.

For Discord, construct pending say tools with configured bindings and the durable
gate, build the runtime, construct/retain the voice service using that runtime,
then attach exactly that configuration-matching service to the tools. Normal and
safe/no-tools voice runtimes remain separately capability-restricted.

## Validation evidence and boundaries

Focused command: `CARGO_BUILD_JOBS=2 cargo test -p missy-voice -p
missy-integrations -p missy-discord-voice --no-default-features`; lint uses the same
three crates and `--all-targets -- -D warnings`.
Final focused run: **68 passed, 0 failed, 2 explicitly ignored native DAVE
external-helper tests**; the same three crates passed all-target clippy with
warnings denied. Evidence process PID 3952814, exit status 0.

New tests compile and execute an isolated synthetic ELF command fixture. Its
server string is `unix:/isolated-test-only`, it does not connect to any audio
daemon, and asserts no DISPLAY. This tests exact argv/environment, filtered
discovery, routing/volume decisions, synthetic PCM playback/input handoff,
denied approvals/physical consent/extra properties and invalid limits.
Loopback WebSocket fixtures test exact OBS text payload, actual VTube envelope
requests and mouth reset after cancellation. Discord synthetic tests cover
binding, no autojoin, queueing, scope/actor override refusal, cancellation and
resume transport retry. A loopback HTTP Comfy fixture tests one submit, scoped
history/download and decoded scene output.

No physical microphone, speaker, camera, active desktop input, real volume,
audio session, live Discord credentials, OBS or VTube instance was exercised.
The two libdave external-helper acceptance tests remain explicitly ignored by
this ordinary command; existing native DAVE acceptance is separate evidence.

## Remaining operator acceptance

- Install/pin native audio utilities and speech models on the destination host.
  Confirm Pulse credentials/socket permissions under the service account without
  borrowing an interactive desktop's environment.
- Operator must accurately classify Physical versus Virtual devices in trusted
  config. A sink name alone cannot prove hardware isolation. Production routing,
  loopback monitor availability and actual microphone/STT/speaker behavior need
  explicit authorized acceptance on the destination host.
- VTube timing is process-start synchronization, not a DAC clock measurement.
  Playback backend buffering/latency remains hardware acceptance. A forced
  process kill or socket failure can defeat best-effort mouth reset.
- Discord queued audio is not durable replayable outbox data. Disconnect/cancel
  may discard queued speech; already transmitted RTP cannot be recalled. Failed
  agent/synthesis/dispatch does not silently repeat uncertain tool execution.
- Gateway readiness/reconnect/live DAVE and Slash/tool application wiring need
  full executable acceptance with dedicated authorized bot credentials.
- Comfy pending job ownership remains in-process; a crash does not auto-recover
  or replay a storyboard submit. Journal reconciliation belongs to the outer
  durable runtime, not an assumption that a failed HTTP request was free.
