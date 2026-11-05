# Native voice edge

This crate receives explicitly sent raw PCM over WebSocket. It never opens a
microphone, camera, speaker, desktop session, or an operating system audio API.
The embedding CLI must load an explicitly selected public config and private
registry. No Python process is involved.

## Public embedding API

* `VoiceConfig`: independent serde configuration with strict unknown-field
  rejection. Default bind `127.0.0.1:8765`; numeric loopback only. Remote bind,
  hostname binds and `allow_insecure_remote` are rejected. Native remote TLS is
  not implemented. Do not expose this listener through an unauthenticated proxy.
* `DeviceRegistry::open(absolute_path)`: containing directory must be owned by
  the current user and 0700, registry file 0600. Parent directory may be created
  if its direct ancestor already exists. Private flock, atomic writes, fresh
  reads ensure operator mode changes are visible to a running gateway.
* `request_pairing(name, room) -> Device`: creates a muted pending UUID with no
  scope, capped to 256 entries and expiring after one hour. A first WebSocket
  `pair_request` does this and closes after `pair_pending`. Hardware metadata
  is accepted as an object for protocol compatibility but not made authority.
* `approve(node_id, scope, mode) -> String`: operator-only approval. Scope
  tenant/user must be explicitly supplied. Channel must be `voice`, session
  must equal the device UUID. Returns a random 256-bit token for **out-of-band
  operator delivery**, never to the unauthenticated pairing socket. Token is
  stored only as salted PBKDF2-HMAC-SHA256 (100,000 iterations). Do not log it.
  Scope cannot be changed by reapproval; revoke and pair again instead.
* `list`, `get`, `set_mode`, `revoke`: trusted local operator operations only.
* `VoiceServer::new(config, Arc<DeviceRegistry>, Arc<dyn Stt>, Arc<dyn Tts>,
  Arc<dyn VoiceAgent>) -> Result<Self>`. Wrap in `Arc` then call
  `bind_and_serve(CancellationToken)` or `serve(TcpListener, CancellationToken)`.
  An externally supplied listener is also checked for loopback.
* `VoiceAgent::respond(&Device, &str, CancellationToken) -> Result<String>`:
  receives only the registry-owned Device. For `SafeChat` the parent MUST use a
  separately constructed runtime with `NoTools` executor and distinct capability
  fingerprint. A prompt or empty tool list layered over a full runtime is not
  isolation. `Muted` never calls STT, agent, or TTS. Clients cannot set modes or
  scope, and there is no global/default user fallback.

## Wire protocol

The first frame must be `auth` with `node_id`/`token`, or `pair_request` with
`friendly_name`, optional `room` and `hardware_profile`. Authentication timeout
includes the WebSocket handshake. No prompt/audio is processed before auth.
Successful authentication yields `auth_ok`; invalid auth yields `auth_fail`.
Unknown auth fields, including scope/mode, are rejected. A second auth frame
or concurrent connection for the same device is rejected. Tokens are static
bearer credentials as in the Python protocol, not nonce-bound challenges:
captured credentials can authenticate again after disconnection until revoked.

An authenticated turn is `audio_start` with integer sample_rate 8000..48000,
channels 1..2 and format `pcm_s16le`, binary PCM frames, then `audio_end`.
No encoded audio or automatic hardware capture is supported. Invalid sequence,
empty/misaligned audio, overflow, rate violations and invalid frames fail
closed. Limits include all preauth connections (default 50), messages/frames
(1 MiB), total buffered PCM (10 MiB), 64 frames/second, ten turns/minute,
idle timeout (120 seconds), and total turn timeout (180 seconds). Limits may be
tightened. Rate windows are connection-local; reconnecting resets them.

STT produces text, optionally emitted as `transcript` with debug_transcripts.
The callback response is emitted as `response_text`. TTS returns a bounded WAV;
the server sends `audio_start` (format `wav`), real binary chunks, `audio_end`.
There is no invented confidence value or fake audio. Synthesis itself is
buffered, while the WAV transport is chunked. No incremental STT/provider/TTS
generation is claimed. Backend failure closes the connection without disclosing
engine errors, prompts, or filesystem paths. Heartbeat validates node identity
and may return `heartbeat_ack` with mode; sensor values never grant authority.
Presence persistence is not implemented.

Scope, revocation and mode are reread on each frame, before/after inference,
before every output chunk, and every 250 ms during long turns. A change cancels
the in-flight future; it cannot undo side effects already dispatched in Full
mode. Disconnect cancels backend futures. Embedding backends must honor future
drop/cancellation; the included command backends kill their own process group.

## Native speech adapters

`CommandConfig { executable: absolute_path, args: Vec<String>, timeout_seconds }`.
`CommandStt::new` / `CommandTts::new` validate native ELF binaries, executable
permissions, bounded trusted argv, and whole-argument placeholders. Shells,
interpreters and scripts are rejected. Arguments are operator configuration,
not a client API. Text is passed to Piper stdin, never substituted into argv.
The child receives a cleared environment, `LC_ALL=C`, null stdout/stderr,
private 0700 working directory and 0600 precreated input/output files. Timeout,
cancellation or dropped futures kill the dedicated process group and remove
temporary files. This is not a sandbox against a malicious configured binary;
models/executable/config must be trusted and protected by the operator.

Whisper.cpp example argv: `-m /absolute/model.bin -f {input} -otxt -of {output}`.
The adapter serializes received PCM to WAV and reads `{output}.txt`. A whisper
build requiring 16 kHz input needs an appropriately configured edge device or
external trusted preprocessing; this crate does not silently resample.

Piper example argv: `--model /absolute/model.onnx --output_file {output}`.
It must write a 16-bit PCM WAV with 8000..48000 Hz and 1..2 channels. WAV chunk
boundaries and file size are checked before returning audio.

## Verification and limits

`cargo test -p missy-voice` exercises actual loopback WebSockets with mock
STT/TTS/agent, private persisted registry, invalid credentials and injected
scope/mode, duplicate/replayed auth frames, muted/live operator policy changes,
audio bounds, frame/connection/auth-timeout limits, real chunk streaming, and
disconnect cancellation. Native CLI adapter tests compile a disposable ELF
fixture, verify argv/stdin/env/private files, and cancel a worker process group.
These fixtures are **not speech engines**. No Whisper/Piper models or binaries
are installed or run by these tests, and actual audio recognition/synthesis
quality/compatibility is not accepted or claimed. CLI/runtime integration and
SafeChat runtime isolation belong to the embedding application.

Legacy Python registries and insecure remote configuration are not silently
imported. Explicit re-pairing or a separately reviewed migration is required.
