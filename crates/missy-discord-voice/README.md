# Native Discord voice

Opt-in native voice session transport, not a second Discord account client.
`DiscordVoiceService` takes existing `Stt`/`Tts` and a `DiscordVoiceAgent`
adapter. The parent selects its real Normal/NoTools/SafeChat runtime using
operator-owned `VoiceBinding.mode` and `scope`. A transcript can never set either.

## Integration

1. Load an explicitly selected `DiscordVoiceConfig`; no implicit default config,
   bot token, environment lookup or audio hardware exists here.
2. Create `missy_gateway::discord_voice_bridge()`. Pass its adapter to
   `DiscordVoiceService::new` and its hook to `serve_discord_with_voice`.
3. `join(guild,channel,authenticated_actor,approval)` requires exact text
   `JOIN DISCORD VOICE <guild>/<channel>`. Leave requires `LEAVE DISCORD VOICE
   <guild>/<channel>`. The actor needs an explicit consenting binding.
4. Cancel via `shutdown()` on application exit. Gateway session changes, bot
   channel moves, credential changes and disconnects invalidate active audio.

The shared text Gateway sends opcode 4 and correlates bot VOICE_STATE_UPDATE
with VOICE_SERVER_UPDATE. Credentials have redacted Debug and zeroized session
tokens, and are never written to the delivery journal. Native voice uses v8:
HELLO, identify/resume, heartbeat+ACK, Ready, UDP discovery, select-protocol,
session description, authenticated SSRC signalling, DAVE MLS transitions.

## Network/security

`endpoint_hosts` is an exact hostname set, not `*.discord.gg`. In addition,
`PolicyHttpClient` must authorize each endpoint's exact HTTPS origin. TLS DNS
pinning remains in the existing policy client. UDP Ready must contain a literal
public IP and operator-approved port range. `allow_private_udp` is an explicit
test/private-server opt-in, never an implicit localhost exemption. A connected
UDP socket pins incoming IP and port; source addresses in packets are not used.

Transport AES-256-GCM authenticates the RTP header, checks a bounded replay window
per permitted SSRC and never wraps nonces. Every non-silence payload then passes
the sender-specific official DAVE ratchet before Opus decoding. Unknown senders,
missing consent, mismatched channels, malformed packets, authentication failures
and replay are dropped. DAVE version zero or transitions to plaintext fail
closed. Crypto is official maintained libdave via `missy-dave`, not a local MLS
implementation. `native_dave` requires explicit absolute library path, SHA256,
version and upstream commit pin. Follow `../missy-dave/README.md` to build it.

Each configured user has an independent Opus decoder, bounded utterance buffer,
two-entry STT queue and cancellable sequential STT/agent/TTS worker. No cross-user
mixdown or shared conversation scope. Output is serialized and paced at 20ms,
with five terminal Opus silence packets and speaking-state cleanup. WAV input
is bounded PCM16 mono/stereo at 8..48kHz, resampled in memory, without ffmpeg.

## Validation and limits

`cargo test -p missy-discord-voice` uses only synthetic PCM and loopback UDP/WS.
It includes ownership tests proving service drop closes its owned session socket
and worker drop destroys an in-flight STT future even when that provider ignores
cooperative cancellation. No live Discord or sound devices are used.

Two ignored native acceptance tests in `src/native_acceptance.rs` require
`MISSY_TEST_LIBDAVE_PATH`, `MISSY_TEST_LIBDAVE_SHA256`, and
`MISSY_TEST_LIBDAVE_HELPER`, matching the upstream library and external-sender
helper built by `../missy-dave/build-native.sh`. Run
`cargo test -p missy-discord-voice -- --include-ignored` with those explicit
variables. Missing native configuration is an error, not a successful skip.
They exercise the real service run loop: PolicyHttpClient WS identify/HELLO/Ready,
actual UDP discovery and select-protocol, mandatory DAVE session description,
two official native MLS sessions and an upstream external sender, commit/welcome
and ratchet activation, SSRC signalling, actual Opus + DAVE + transport AEAD,
mock STT receiving decoded non-silent mono PCM, an agent adapter asserting the
operator-owned NoTools binding/scope, and synthetic TTS WAV returning through
real outbound encryption and peer Opus decoding. They also verify paced audio,
five encrypted terminal silence frames, speaking cleanup, and session teardown
on Gateway invalidation or a signalling downgrade.

Negative media phases send enough packets to form an utterance if broken:
unknown or nonconsenting users, a mapped user whose DAVE identity is wrong,
modified transport nonces, duplicate authenticated packets, plaintext DAVE
downgrades and altered DAVE frames must produce zero STT/agent calls. Invalid
media is **dropped**, not allowed to terminate the entire channel session;
signalling downgrades terminate it. Transition/version integer overflows are
rejected rather than truncating into valid protocol values.

The plaintext WS fixture is reachable only through a private `cfg(test)` adapter
that rejects nonliteral/nonloopback hosts and URL credentials/query/fragment;
it still requires the policy client's exact HTTP origin/private-network grant.
Production callers have no URL override and continue to use verified WSS.
The Gateway credential adapter and STT/agent/TTS are fixture implementations,
not the real Discord Gateway bridge, speech providers, or runtime tool-policy
enforcement. Those boundaries require their own tests; the native fixture does
not establish real NoTools runtime behavior merely by asserting its binding.

Verified locally on 2026-11-05 with the pinned native build: all **19 tests**
passed with `--include-ignored`; `cargo clippy -p missy-discord-voice --all-targets
-- -D warnings` passed. Both full native acceptance tests also passed five
additional consecutive runs (10/10 native cases). This evidence is specifically
local fixture/native acceptance, not a live Discord acceptance claim.

Live interoperability is not claimed without an explicitly authorized live
test. Local fixtures cannot verify Discord deployment behaviour. Voice resumes
preserve transport/DAVE state only for a resumable voice socket interruption;
nonresumable closure, stale heartbeat or Gateway reset requires a fresh join.
No automatic fresh join reuses an old approval. Packet jitter is tolerated by
Opus decoding arrival order, but there is no adaptive jitter buffer/PLC yet.
Speech segmentation uses an energy gate (100 PCM16 RMS, approximately -50dBFS)
plus configured silence timeout. Silent packets do not extend active utterances.
This is conservative energy VAD, not a noise-suppressing speech classifier.

## Dependency selection

Songbird 0.6.1 was inspected and rejected because its private networking cannot
accept the policy WebSocket client, and its DAVE negotiation cannot be enforced
through public APIs. davey 0.2.0 was rejected after audit found incompatible
transitive crypto advisories. No broad Serenity or unmaintained voice runtime is
included. Opus2 0.4 uses bundled maintained libopus; AES-GCM is RustCrypto.
