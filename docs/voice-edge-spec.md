# Edge Node Client Specification

**Version:** 1.1
**Status:** Draft
**Target hardware:** Raspberry Pi 5 + ReSpeaker mic array + speaker
**Companion project:** `missy-edge` (reference implementation)

> **Authoritative protocol reference:** `missy-edge/docs/protocol.md` in
> the companion repository is generated against the actual server
> implementation (`missy/channels/voice/server.py`) and is kept current.
> This document describes the intended design and hardware/deployment
> considerations for edge nodes in general; for exact wire-protocol
> details (message types, fields, close codes) prefer
> `missy-edge/docs/protocol.md`. The terminology below has been updated
> to match the real server protocol.

---

## 1. Overview

Each edge node is a small always-on device deployed in a room of the home. Its sole responsibility is to detect the wake word "Missy", capture the user's spoken utterance, stream it to the Missy gateway over a WebSocket connection, and play back the synthesised audio response through its local speaker. All heavy computation — speech-to-text, language model inference, and text-to-speech — happens on the server. The edge node handles only wake word detection, audio capture, and audio playback.

The reference hardware target is a Raspberry Pi 5 paired with a ReSpeaker 4-mic or 6-mic USB array for voice capture and a local USB or 3.5 mm speaker for playback. The design is intentionally hardware-agnostic: any Linux SBC with a working ALSA audio device and a stable LAN connection is a valid host.

---

## 2. Wake Word Detection

Wake word detection runs entirely on the edge node. Pre-trigger audio is never transmitted to the server; only post-wake-word audio begins streaming.

The preferred wake word engine is **openWakeWord** (Apache 2.0, <https://github.com/dscripka/openWakeWord>). A custom "Missy" model is trained or fine-tuned using openWakeWord's training pipeline and distributed with the `missy-edge` repository. The model is loaded at startup and runs continuously in a background thread. False-positive sensitivity should be configured with a threshold of `0.7` or higher to reduce spurious activations.

**Porcupine** (Picovoice) is an acceptable alternative for deployments where a commercial licence is available or a free-tier access key is sufficient. The interface is compatible; only the initialisation code differs. The remainder of this specification applies equally to both engines.

After wake word detection, the node immediately activates the microphone capture pipeline and begins buffering audio. A short pre-trigger buffer of 0.3 seconds is retained in memory and prepended to the streamed audio so the server receives a clean start of the utterance, eliminating the hard onset typically caused by the detection latency. This buffer is discarded without transmission if no `audio_start` message is sent to the server within 500 milliseconds.

---

## 3. WebSocket Protocol

The edge node maintains a persistent WebSocket connection to the Missy gateway. The gateway listens on the host and port configured in `~/.missy/config.yaml` under the `voice.host` and `voice.port` keys (defaults: `0.0.0.0:8765`). The WebSocket URI is:

```
ws://<host>:<port>/voice
```

or, when TLS is enabled:

```
wss://<host>:<port>/voice
```

All control messages are JSON text frames. Audio data is transmitted as binary frames (see Section 5).

### 3.1 Client → Server messages

#### `auth`

Sent immediately after the WebSocket handshake completes. Must be `auth` or `pair_request` — any other first message, or no message within a 10-second timeout, causes the server to close the connection (close code `1008`). If authentication fails, the server responds with `auth_fail` and closes with code `1008` (see Section 4).

```json
{
  "type": "auth",
  "node_id": "a3f8c2d1-9b4e-4f7a-8e6d-1c2b3d4e5f60",
  "token": "HdK9x2mP..."
}
```

#### `pair_request`

Sent instead of `auth` by a node that has not yet been assigned a token or `node_id`. The **server** generates and assigns the `node_id` — it must not be supplied by the client. The server records the request, responds with `pair_pending` (see below), and closes the connection; the administrator approves the request out-of-band via `missy devices pair --node-id <id>`.

```json
{
  "type": "pair_request",
  "friendly_name": "Living Room Pi",
  "room": "Living Room",
  "hardware_profile": {
    "platform": "aarch64",
    "hostname": "missy-lr",
    "os": "Linux"
  }
}
```

`hardware_profile` is optional freeform device metadata. Do **not** include `node_id` in this message.

#### `audio_start` (client)

Sent immediately before the first binary audio frame of an utterance. Signals the server to open a new STT session. Audio is streamed as raw PCM (not WAV) in the client-to-server direction.

```json
{
  "type": "audio_start",
  "sample_rate": 16000,
  "channels": 1,
  "format": "pcm_s16le"
}
```

#### `audio_end` (client)

Sent after the final binary audio frame of the utterance. The server flushes the remaining audio through the STT pipeline and begins processing the transcript.

```json
{
  "type": "audio_end"
}
```

#### `heartbeat`

Sent every 60 seconds while the connection is alive. Optionally includes environmental sensor readings. The server updates the node's `last_seen` timestamp and `online` status in the registry.

```json
{
  "type": "heartbeat",
  "node_id": "a3f8c2d1-9b4e-4f7a-8e6d-1c2b3d4e5f60",
  "timestamp": 1741737600,
  "occupancy": true,
  "noise_level": 42.3
}
```

`occupancy` is a boolean derived from an mmWave or PIR presence sensor, if available. `noise_level` is the ambient dB SPL estimated from the microphone level. Both fields are optional; omit them entirely if no sensor data is available.

#### `playback_done`

Sent after the node finishes playing the TTS audio response. Allows the gateway to log the round-trip latency and to know the node is ready for the next interaction.

```json
{
  "type": "playback_done",
  "node_id": "a3f8c2d1-9b4e-4f7a-8e6d-1c2b3d4e5f60",
  "session_id": "sess_7f3a2b"
}
```

### 3.2 Server → Client messages

#### `auth_ok`

Sent after a successful `auth` message. Includes the server-confirmed `node_id` and the node's assigned room.

```json
{
  "type": "auth_ok",
  "node_id": "a3f8c2d1-9b4e-4f7a-8e6d-1c2b3d4e5f60",
  "room": "Living Room"
}
```

#### `auth_fail`

Sent when authentication fails. The server closes the connection (close code `1008`) immediately after sending this.

```json
{
  "type": "auth_fail",
  "reason": "invalid credentials"
}
```

Known reasons: `"invalid credentials"`, `"node is not yet approved"`, `"node not found after token verification"`.

#### `pair_pending`

Sent in response to `pair_request`, confirming the request was recorded and returning the **server-assigned** `node_id`. The server closes the connection after sending this; the device must persist the returned `node_id` and wait for out-of-band administrator approval before reconnecting with `auth`.

```json
{
  "type": "pair_pending",
  "node_id": "a3f8c2d1-9b4e-4f7a-8e6d-1c2b3d4e5f60"
}
```

There is no `pair_ack` message and no token is ever sent over the WebSocket connection. The administrator runs `missy devices pair --node-id <id>` on the server host, which prints the token to the CLI. The token must be provisioned to the device out-of-band (e.g. writing it directly to the SD card, or via a one-time pairing QR code shown on a trusted screen) — see Section 6.

#### `transcript` (optional)

Sent only when the server has `debug_transcripts` enabled. Contains the STT result for the just-completed utterance.

```json
{
  "type": "transcript",
  "text": "What's the weather like today?",
  "confidence": 0.95
}
```

#### `response_text`

The agent's text response. Always sent before any TTS audio.

```json
{
  "type": "response_text",
  "text": "It's currently 72 degrees and sunny."
}
```

#### `audio_start` (server)

Sent after the server completes TTS synthesis, signalling the start of the audio response. Binary WAV frames (not raw PCM) follow; the node must buffer all binary frames until `audio_end` is received before beginning playback, since the response is a single self-contained WAV file (with headers) split across frames.

```json
{
  "type": "audio_start",
  "sample_rate": 22050,
  "format": "wav"
}
```

#### `audio_end` (server)

Signals that all binary WAV frames for the current response have been sent. The node should begin playback immediately.

```json
{
  "type": "audio_end"
}
```

#### `error`

Sent when the server encounters an error processing the node's request.

```json
{
  "type": "error",
  "message": "Speech recognition failed"
}
```

#### `muted`

Sent when the administrator changes the node's policy to `muted`. The server closes the connection after sending this. The client should enter a dormant state, display the MUTED LED, and cease sending audio until it reconnects and receives a new `auth_ok`.

```json
{
  "type": "muted"
}
```

---

## 4. Authentication

Every paired edge node holds a unique bearer token issued at pairing time. Tokens are plaintext strings compared verbatim against the registry entry for the given `node_id` — there is no `msy_tok_v1_` prefix convention enforced by the server; treat the token as an opaque secret.

On the edge node, the token must be stored in a file readable only by the process user:

```
/etc/missy-edge/token        # chmod 600, owned by the service user
```

If the Raspberry Pi 5's RP1 chip exposes a TPM 2.0 interface, the token should instead be sealed to the TPM using `tpm2-tools` to provide hardware-backed confidentiality.

The token is included verbatim in every `auth` message. The server validates it against the registry entry for the given `node_id`. If the token is absent, the node is not yet approved, or the token does not match, the server sends an `auth_fail` frame and closes the connection with WebSocket close code `1008` (not `4001`).

Tokens are currently non-expiring. Token rotation is discussed in Section 11.

---

## 5. Audio Format

Audio streamed from the edge node to the server (client → server, via `audio_start`/binary frames/`audio_end`) is raw PCM:

- **Encoding:** PCM signed 16-bit, little-endian (`pcm_s16le`)
- **Sample rate:** 16 000 Hz
- **Channels:** 1 (mono)
- **Frame size:** typically 2 560 bytes per binary WebSocket frame (1 280 samples = 80 ms at 16 kHz). The server buffers up to 10 MB of accumulated audio per connection and will error out an utterance that exceeds this.

Audio received from the server (TTS output) is a **complete WAV file** (including headers), chunked across binary WebSocket frames (default 4096 bytes each) between the server's `audio_start` and `audio_end` messages — not a raw PCM stream. Concatenate all frames to reconstruct the WAV file before playback. Sample rate is indicated by the `sample_rate` field in the server's `audio_start` message; the reference Piper TTS configuration produces 22 050 Hz audio.

**Maximum utterance duration:** If an `audio_end` message has not been sent within a reasonable time of `audio_start`, the server will send an `error` frame and discard the session. The client should handle this gracefully by resetting its microphone capture state.

---

## 6. Pairing Workflow

A factory-fresh edge node has no token and therefore cannot authenticate. Pairing must be completed before normal operation is possible.

1. The node connects to the gateway WebSocket endpoint with no persisted `node_id` or token.
2. Instead of an `auth` message, the node sends a `pair_request` message containing a human-readable `friendly_name`, a `room` identifier, and optional `hardware_profile` metadata. **The node does not choose or send a `node_id`.**
3. The gateway records the request in the device registry with `paired: false`, generates a `node_id`, and responds with `pair_pending` containing that `node_id`. The connection is then closed by the server.
4. The edge node persists the server-assigned `node_id` to `/etc/missy-edge/node_id`.
5. The administrator runs `missy devices pair --node-id <id>` on the server host. The CLI prints the generated token to the terminal — this is the only time it is shown.
6. The administrator provisions the token to the device out-of-band (direct SD card write, secure copy, or a one-time QR code) and it is written to `/etc/missy-edge/token` with permissions `0600`.
7. The node reconnects and sends a normal `auth` message using the new `node_id` and token.

The gateway is idempotent: repeated `pair_request` messages before approval do not create duplicate registry entries, and each such reconnect receives the same pending `node_id`.

**Re-pairing:** If a token is lost (e.g., SD card corruption), the administrator must first remove the stale entry with `missy devices unpair <node_id>` and then run a fresh pairing flow, which will be assigned a new `node_id`.

---

## 7. Heartbeat

The edge node sends a `heartbeat` message every 60 seconds regardless of whether any utterances have been processed. The gateway updates the node's `last_seen` field and marks the node `online: true`. If no heartbeat is received within 120 seconds (two missed intervals), the gateway marks the node `online: false` in the registry; it does not close the connection, because the connection may have silently dropped and the node is in the process of reconnecting.

The heartbeat may include sensor readings:

- `occupancy` (boolean): presence detection via mmWave radar (e.g., LD2410) or PIR sensor. Omit if no sensor is fitted.
- `noise_level` (float, dB): ambient noise level estimated from the RMS of the microphone input over the past second. Omit if not computed.

These values are stored in the registry and surfaced by `missy devices status`.

---

## 8. Reconnection

The edge node must implement exponential backoff with jitter for reconnection attempts. The algorithm is as follows:

- Initial wait: 1 second
- Backoff multiplier: 2×
- Maximum wait: 60 seconds
- Jitter: ±20% of the computed wait value (uniform random)
- After a successful connection and `auth_ok`, reset the backoff counter to zero

Authentication failures (WebSocket close code `1008`) are treated separately. The node must not re-attempt authentication more than 5 times within any 60-second window. If this limit is reached, the node should enter a dormant state and log the failure prominently. This prevents token hammering and log flooding on the server.

Network errors (TCP reset, DNS failure, server unavailable) should trigger normal backoff without the authentication rate limit.

---

## 9. Resource Requirements

The edge node is expected to idle comfortably within the following resource budget on a Raspberry Pi 5 (4 GB RAM):

| Resource | Target (idle) |
|---|---|
| CPU usage | < 5% (single core) |
| Resident memory | < 100 MB |
| Network (idle) | < 2 KB/s (heartbeats only) |
| Network (streaming) | ~256 KB/s during utterance |

The openWakeWord "Missy" ONNX model should be run via the `onnxruntime` inference backend rather than the default TensorFlow Lite backend. On ARM Cortex-A76 (Pi 5), `onnxruntime` with the XNNPACK execution provider provides significantly lower latency and CPU overhead.

Wake word detection runs in a continuous loop with 80 ms audio chunks (1 280 samples at 16 kHz). The inference pass on a Pi 5 with ONNX Runtime typically completes in under 5 ms per chunk, leaving the core largely idle.

---

## 10. Reference Implementation

This specification describes the expected behaviour of any conforming edge node client. It is written for implementors of the companion project `missy-edge`, which is maintained separately from the core `missy` repository.

The recommended technology stack for the reference Python implementation is:

- **Audio capture:** `sounddevice` (PortAudio bindings) or `pyaudio`
- **Wake word:** `openwakeword` with ONNX Runtime backend
- **WebSocket client:** `websockets` >= 12.0
- **Async event loop:** `asyncio` (Python 3.11+)
- **Audio playback:** `sounddevice` or `simpleaudio`
- **Logging:** Python standard library `logging`, structured output to journald

The companion repository should be named `missy-edge` and organised as a `systemd` service installable via a setup script. The service file should set `Restart=always` and `RestartSec=5` so that transient failures are recovered automatically.

---

## 11. Security Considerations

**Network exposure.** The Missy gateway WebSocket server should be bound to the LAN interface only (not `0.0.0.0`) unless the network is fully isolated. Binding to a specific interface (e.g., `192.168.1.10`) prevents accidental exposure over a VPN tunnel or Docker bridge.

**TLS.** Plain WebSocket (`ws://`) is acceptable on a trusted LAN with no guest network. TLS (`wss://`) is strongly encouraged for any deployment where the LAN is shared with untrusted devices. A self-signed certificate is sufficient; the edge node must pin the certificate fingerprint or the CA certificate to prevent MITM attacks on the local network.

**Token confidentiality.** Tokens are never sent over the WebSocket connection except in the client's own `auth` message — pairing approval delivers the token only via the `missy devices pair --node-id <id>` CLI output, not as a protocol message, so it cannot be intercepted mid-pairing on the wire. Tokens must never be logged or included in error messages. The administrator is responsible for conveying the CLI-printed token to the node through a secure out-of-band channel (e.g., provisioning the SD card directly, or using a one-time pairing QR code displayed on a trusted screen).

**Token rotation.** The current implementation issues non-expiring tokens. A future release should add token expiry and a rotation mechanism. Until this is implemented, administrators should rotate tokens manually by unpairing and re-pairing any node suspected of token exposure.

**Microphone privacy.** Pre-trigger audio is never transmitted. The edge node should provide a physical mute button that disconnects microphone power at the hardware level for situations where the user requires a hard privacy guarantee. The gateway's `muted` policy mode provides a software-level mute but does not prevent audio capture on the node itself.
