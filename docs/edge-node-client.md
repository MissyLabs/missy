# Edge Node Client Specification

**Version:** 1.0
**Status:** Draft
**Target hardware:** Raspberry Pi 5 + ReSpeaker mic array + speaker
**Companion project:** `missy-edge` (reference implementation)

---

## 1. Overview

Each edge node is a small always-on device deployed in a room of the home. Its sole responsibility is to detect the wake word "Missy", capture the user's spoken utterance, stream it to the Missy gateway over a WebSocket connection, and play back the synthesised audio response through its local speaker. All heavy computation — speech-to-text, language model inference, and text-to-speech — happens on the server. The edge node handles only wake word detection, audio capture, and audio playback.

The reference hardware target is a Raspberry Pi 5 paired with a ReSpeaker 4-mic or 6-mic USB array for voice capture and a local USB or 3.5 mm speaker for playback. The design is intentionally hardware-agnostic: any Linux SBC with a working ALSA audio device and a stable LAN connection is a valid host.

---

## 2. Wake Word Detection

Wake word detection runs entirely on the edge node. Pre-trigger audio is never transmitted to the server; only post-wake-word audio begins streaming.

The preferred wake word engine is **openWakeWord** (Apache 2.0, <https://github.com/dscripka/openWakeWord>). A custom "Missy" model is trained or fine-tuned using openWakeWord's training pipeline and distributed with the `missy-edge` repository. The model is loaded at startup and runs continuously in a background thread. False-positive sensitivity should be configured with a threshold of `0.7` or higher to reduce spurious activations.

**Porcupine** (Picovoice) is an acceptable alternative for deployments where a commercial licence is available or a free-tier access key is sufficient. The interface is compatible; only the initialisation code differs. The remainder of this specification applies equally to both engines.

After wake word detection, the node immediately activates the microphone capture pipeline and begins buffering audio. A short pre-trigger buffer of 0.3 seconds is retained in memory and prepended to the streamed audio so the server receives a clean start of the utterance, eliminating the hard onset typically caused by the detection latency. This buffer is discarded without transmission if no `stream_start` message is sent to the server within 500 milliseconds.

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

Sent immediately after the WebSocket handshake completes. Must be the first message. If authentication fails, the server closes the connection with code 4001.

```json
{
  "type": "auth",
  "node_id": "a3f8c2d1-9b4e-4f7a-8e6d-1c2b3d4e5f60",
  "token": "msy_tok_v1_..."
}
```

#### `pair_request`

Sent by a node that has not yet been assigned a token. The server records the request and the administrator approves it via `missy devices pair`. After approval, the server sends a `pair_ack` containing the token; the client must persist this token locally and use it for all subsequent `auth` messages.

```json
{
  "type": "pair_request",
  "node_id": "a3f8c2d1-9b4e-4f7a-8e6d-1c2b3d4e5f60",
  "name": "Living Room",
  "room": "living_room"
}
```

#### `stream_start`

Sent immediately before the first binary audio frame of an utterance. Signals the server to open a new STT session.

```json
{
  "type": "stream_start",
  "node_id": "a3f8c2d1-9b4e-4f7a-8e6d-1c2b3d4e5f60",
  "session_id": "sess_7f3a2b"
}
```

#### `stream_end`

Sent after the final binary audio frame of the utterance. The server flushes the remaining audio through the STT pipeline and begins processing the transcript.

```json
{
  "type": "stream_end",
  "node_id": "a3f8c2d1-9b4e-4f7a-8e6d-1c2b3d4e5f60",
  "session_id": "sess_7f3a2b"
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

Sent after a successful `auth` message. Includes the effective policy for this node.

```json
{
  "type": "auth_ok",
  "policy": "full"
}
```

#### `pair_ack`

Sent after an administrator approves a pending pairing request. The token must be stored permanently on the device with file permissions `0600` or in the Pi's TPM if available.

```json
{
  "type": "pair_ack",
  "node_id": "a3f8c2d1-9b4e-4f7a-8e6d-1c2b3d4e5f60",
  "token": "msy_tok_v1_..."
}
```

#### `tts_audio`

Sent after the server completes TTS synthesis. The body is a JSON frame containing metadata; the raw audio binary follows as one or more consecutive binary WebSocket frames. The node must buffer all binary frames until `tts_end` is received before beginning playback.

```json
{
  "type": "tts_audio",
  "session_id": "sess_7f3a2b",
  "sample_rate": 22050,
  "channels": 1,
  "encoding": "pcm_s16le",
  "total_bytes": 176400
}
```

#### `tts_end`

Signals that all binary audio frames for the current session have been sent. The node should begin playback immediately.

```json
{
  "type": "tts_end",
  "session_id": "sess_7f3a2b"
}
```

#### `error`

Sent when the server encounters an error processing the node's request. The `code` field identifies the error class; `message` is human-readable.

```json
{
  "type": "error",
  "code": "stt_failed",
  "message": "Speech recognition produced no transcript."
}
```

Known error codes: `auth_failed`, `policy_denied`, `stt_failed`, `llm_error`, `tts_failed`, `session_not_found`.

#### `muted`

Sent when the administrator changes the node's policy to `muted`. The node should silence its speaker and cease sending `stream_start`/`stream_end` frames until a new `auth_ok` with a non-muted policy is received (which occurs on reconnection after a policy change).

```json
{
  "type": "muted"
}
```

---

## 4. Authentication

Every paired edge node holds a unique bearer token issued by the gateway at pairing time. Tokens are opaque strings with the prefix `msy_tok_v1_` followed by 32 bytes of cryptographically random data encoded in URL-safe base64.

On the edge node, the token must be stored in a file readable only by the process user:

```
/etc/missy-edge/token        # chmod 600, owned by the service user
```

If the Raspberry Pi 5's RP1 chip exposes a TPM 2.0 interface, the token should instead be sealed to the TPM using `tpm2-tools` to provide hardware-backed confidentiality.

The token is included verbatim in every `auth` message. The server validates it against the registry entry for the given `node_id`. If the token is absent, expired (future feature), or mismatched, the server responds with an `error` frame with code `auth_failed` and closes the connection with WebSocket close code `4001`.

Tokens are currently non-expiring. Token rotation is discussed in Section 11.

---

## 5. Audio Format

All audio streamed from the edge node to the server adheres to the following format:

- **Encoding:** PCM signed 16-bit, little-endian (`pcm_s16le`)
- **Sample rate:** 16 000 Hz
- **Channels:** 1 (mono)
- **Frame size:** 512 samples per binary WebSocket frame (32 ms at 16 kHz), yielding 1 024 bytes per frame

Audio received from the server (TTS output) uses the same encoding but may use a different sample rate, as indicated by the `sample_rate` field in the `tts_audio` control message. The reference Piper TTS configuration produces 22 050 Hz audio; the edge node should resample or pass through as appropriate for the attached speaker.

**Maximum utterance duration:** 30 seconds. If a `stream_end` message has not been sent within 30 seconds of `stream_start`, the server will send an `error` frame with code `stt_failed` and discard the session. The client should handle this gracefully by resetting its microphone capture state.

---

## 6. Pairing Workflow

A factory-fresh edge node has no token and therefore cannot authenticate. Pairing must be completed before normal operation is possible.

1. The node generates a random UUID as its `node_id` and persists it to `/etc/missy-edge/node_id` (created on first boot by the setup script).
2. The node connects to the gateway WebSocket endpoint.
3. Instead of an `auth` message, the node sends a `pair_request` message containing its `node_id`, a human-readable `name`, and a `room` identifier.
4. The gateway records the request in the device registry with `paired: false` and holds the connection open.
5. The administrator runs `missy devices pair` on the server host. The CLI lists pending nodes and prompts for confirmation. On approval, the gateway generates a token, updates the registry, and sends a `pair_ack` message to the still-open connection.
6. The edge node receives `pair_ack`, writes the token to `/etc/missy-edge/token` with permissions `0600`, and disconnects cleanly.
7. The node immediately reconnects and sends a normal `auth` message using the new token.

If the connection drops during step 4 before `pair_ack` is received, the node should reconnect and re-send `pair_request`. The gateway is idempotent: repeated `pair_request` messages from the same `node_id` do not create duplicate registry entries. Once the administrator approves, any subsequent reconnecting `pair_request` from that node_id will receive the already-generated token.

**Re-pairing:** If a token is lost (e.g., SD card corruption), the administrator must first remove the stale entry with `missy devices unpair <node_id>` and then run a fresh pairing flow from a node with a new `node_id`.

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

Authentication failures (WebSocket close code `4001`) are treated separately. The node must not re-attempt authentication more than 5 times within any 60-second window. If this limit is reached, the node should enter a dormant state and log the failure prominently. This prevents token hammering and log flooding on the server.

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

**Token confidentiality.** Tokens must never be logged, included in error messages, or transmitted over unencrypted connections. The `missy devices pair` output warns that the token is shown only once; the administrator is responsible for conveying it to the node through a secure channel (e.g., provisioning the SD card directly, or using a one-time pairing QR code displayed on a trusted screen).

**Token rotation.** The current implementation issues non-expiring tokens. A future release should add token expiry and a rotation mechanism: the server issues a new token during a `heartbeat_ack` response and the node replaces the stored token atomically. Until this is implemented, administrators should rotate tokens manually by unpairing and re-pairing any node suspected of token exposure.

**Microphone privacy.** Pre-trigger audio is never transmitted. The edge node should provide a physical mute button that disconnects microphone power at the hardware level for situations where the user requires a hard privacy guarantee. The gateway's `muted` policy mode provides a software-level mute but does not prevent audio capture on the node itself.
