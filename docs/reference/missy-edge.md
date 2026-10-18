# Companion project: missy-edge

Moved verbatim from `CLAUDE.md`.

**Repository:** `/home/missy/missy-edge/` ([GitHub](https://github.com/MissyLabs/missy-edge))

`missy-edge` is the standalone Raspberry Pi voice edge node client. It connects to Missy's VoiceChannel server (`missy/channels/voice/server.py`) over WebSocket and provides always-on wake word detection, audio streaming, and TTS playback.

### Relationship to This Repo

- **Server side** (this repo): `missy/channels/voice/server.py` is the authoritative WebSocket protocol implementation. `missy/channels/voice/registry.py` handles device registration, PBKDF2 token hashing, and node management. `missy/channels/voice/edge_client.py` is the original reference client (PipeWire, manual push-to-talk, local testing only).
- **Client side** (missy-edge): Production edge node client with wake word, sounddevice audio, reconnection, LED/mute hardware support, systemd service.

### missy-edge Architecture

```
missy-edge/
├── missy_edge/
│   ├── client.py        # EdgeClient: reconnect loop, auth, concurrent async loops
│   ├── audio.py         # AudioCapture (sounddevice → asyncio.Queue) + AudioPlayback (WAV)
│   ├── wakeword.py      # WakeWordDetector (openWakeWord + ONNX, pre-trigger buffer)
│   ├── protocol.py      # Message builders matching server.py protocol
│   ├── reconnect.py     # ExponentialBackoff (1s→60s) + AuthRateLimiter (5/60s)
│   ├── config.py        # EdgeConfig, YAML load chain, token permission checks
│   ├── noise.py         # RMS noise estimator (reported in heartbeats)
│   ├── led.py           # ReSpeaker USB HID LED ring (6 states)
│   ├── mute.py          # Physical mute button monitor (HID polling thread)
│   └── __main__.py      # CLI: --pair, --manual, --server, --config
├── systemd/missy-edge.service
├── setup_pi.sh          # Pi provisioning script
├── docs/                # protocol.md, deployment.md, development.md
└── tests/               # 56 tests (protocol, reconnect, config, audio, wakeword, client)
```

### Protocol Alignment (Critical)

The spec doc (`docs/edge-node-client.md`) and actual server differ. Both this repo and missy-edge use the **server implementation**:

| Spec Doc | Actual Server (`server.py`) | missy-edge Uses |
|---|---|---|
| `stream_start`/`stream_end` | `audio_start`/`audio_end` | `audio_start`/`audio_end` |
| `tts_audio` + `tts_end` (raw PCM) | `audio_start` + binary WAV + `audio_end` | WAV over `audio_start`/`audio_end` |
| `pair_request` with `node_id` | Server assigns `node_id`, returns in `pair_pending` | Omit `node_id` from request |
| Token via WebSocket `pair_ack` | Token shown in CLI `missy devices pair` output | Out-of-band token provisioning |

### Cross-Repo Development Notes

- Changing the WebSocket protocol in `missy/channels/voice/server.py` requires corresponding updates in `missy-edge/missy_edge/protocol.py` and `missy-edge/missy_edge/client.py`.
- The device pairing flow spans both repos: `missy devices pair` (this repo, CLI) generates the token; the user manually provisions it to `/etc/missy-edge/token` on the Pi.
- Edge node management commands (`missy devices list/status/pair/unpair/policy`) in this repo interact with the `DeviceRegistry` that missy-edge authenticates against.

### missy-edge Commands

```bash
# Install (on Pi)
sudo bash setup_pi.sh

# Or development install
pip install -e ".[dev]"

# Tests
python -m pytest tests/ -v          # 56 tests

# Run
missy-edge                          # Wake word mode (default)
missy-edge --manual                 # Push-to-talk (Enter key)
missy-edge --pair --name N --room R # Pair new device
```

### missy-edge Config Locations

| Purpose | Path |
|---|---|
| System config | `/etc/missy-edge/config.yaml` |
| User config | `~/.config/missy-edge/config.yaml` |
| Auth token | `/etc/missy-edge/token` (chmod 600, required) |
| Node identity | `/etc/missy-edge/node_id` |
| Wake word model | `/opt/missy-edge/models/*.onnx` |
| systemd service | `/etc/systemd/system/missy-edge.service` |
| Venv | `/opt/missy-edge/venv/` |
