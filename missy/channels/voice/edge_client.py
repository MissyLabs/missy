#!/usr/bin/env python3
"""Local edge-node client for the Missy voice channel.

Captures audio from a local microphone (e.g. Jabra SPEAK 510), streams it
to the Missy voice server over WebSocket, and plays back the TTS response.

This is the same client that runs on Raspberry Pi edge nodes — the only
difference is the hardware (ReSpeaker HAT vs USB speakerphone).

Usage::

    # First time — pair the device:
    python -m missy.channels.voice.edge_client --pair --name "Office Jabra" --room "Office"

    # Then approve on the server:
    missy devices pair --node-id <NODE_ID>

    # Run the voice loop (press Enter to talk, Ctrl+C to quit):
    python -m missy.channels.voice.edge_client --node-id <NODE_ID> --token <TOKEN>

    # Or with config file:
    python -m missy.channels.voice.edge_client --config ~/.missy/edge.json

Prerequisites::

    sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good

"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import websockets
except ImportError:
    print("websockets not installed. Run: pip install websockets", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

# Audio capture defaults.
_DEFAULT_SAMPLE_RATE = 16000
_DEFAULT_CHANNELS = 1
_DEFAULT_RECORD_SECONDS = 5
_DEFAULT_SERVER_URL = "ws://127.0.0.1:8765"
_CONFIG_PATH = Path.home() / ".missy" / "edge.json"


def _ensure_runtime_dir() -> dict[str, str]:
    """Return env dict with XDG_RUNTIME_DIR set for PipeWire."""
    env = {**os.environ}
    if "XDG_RUNTIME_DIR" not in env:
        uid = os.getuid()
        env["XDG_RUNTIME_DIR"] = f"/run/user/{uid}"
    return env


def _record_audio(
    duration: float,
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
    channels: int = _DEFAULT_CHANNELS,
) -> bytes:
    """Record audio from the default mic and return raw PCM-16 bytes.

    Tries pw-record (PipeWire native) first, falls back to GStreamer pipewiresrc.
    """
    env = _ensure_runtime_dir()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        # pw-record captures reliably from PipeWire, including over SSH.
        cmd = [
            "pw-record",
            "--format=s16",
            f"--rate={sample_rate}",
            f"--channels={channels}",
            wav_path,
        ]

        proc = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE)
        time.sleep(duration)
        proc.send_signal(signal.SIGINT)

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        if not Path(wav_path).exists() or Path(wav_path).stat().st_size == 0:
            return b""

        # pw-record writes a WAV file. Extract raw PCM from it.
        import wave

        with wave.open(wav_path, "rb") as wf:
            return wf.readframes(wf.getnframes())
    except FileNotFoundError:
        logger.warning("pw-record not found, trying GStreamer...")
        return _record_audio_gst(duration, sample_rate, channels, env)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(wav_path)


def _record_audio_gst(
    duration: float,
    sample_rate: int,
    channels: int,
    env: dict[str, str],
) -> bytes:
    """Fallback: record audio using GStreamer pipewiresrc."""
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        raw_path = tmp.name

    try:
        cmd = [
            "gst-launch-1.0",
            "-q",
            "pipewiresrc",
            "do-timestamp=true",
            "!",
            f"audio/x-raw,rate={sample_rate},channels={channels},format=S16LE",
            "!",
            "audioconvert",
            "!",
            "audioresample",
            "!",
            f"audio/x-raw,rate={sample_rate},channels={channels},format=S16LE",
            "!",
            "filesink",
            f"location={raw_path}",
        ]

        proc = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE)
        time.sleep(duration)
        proc.send_signal(signal.SIGINT)

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        if Path(raw_path).exists():
            return Path(raw_path).read_bytes()
        return b""
    finally:
        with contextlib.suppress(OSError):
            os.unlink(raw_path)


def _play_wav(wav_data: bytes) -> None:
    """Play WAV audio data through the default audio output via GStreamer."""
    env = _ensure_runtime_dir()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_data)
        wav_path = tmp.name

    try:
        subprocess.run(
            [
                "gst-launch-1.0",
                "-q",
                "filesrc",
                f"location={wav_path}",
                "!",
                "wavparse",
                "!",
                "audioconvert",
                "!",
                "audioresample",
                "!",
                "pipewiresink",
            ],
            env=env,
            timeout=60,
            capture_output=True,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.error("Playback failed: %s", exc)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(wav_path)


async def _pair_device(
    server_url: str,
    friendly_name: str,
    room: str,
) -> str | None:
    """Send a pair_request to the voice server and return the assigned node_id."""
    async with websockets.connect(server_url) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "pair_request",
                    "friendly_name": friendly_name,
                    "room": room,
                    "hardware_profile": {
                        "platform": sys.platform,
                        "hostname": os.uname().nodename,
                    },
                }
            )
        )

        raw = await ws.recv()
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"Malformed pairing response: {exc}", file=sys.stderr)
            return None

        if msg.get("type") == "pair_pending":
            node_id = msg["node_id"]
            print(f"\nPairing request sent. Node ID: {node_id}")
            print("Approve it on the server with:")
            print(f"  missy devices pair --node-id {node_id}")
            print("\nThen run this client with:")
            print(
                f"  python -m missy.channels.voice.edge_client --node-id {node_id} --token <TOKEN>"
            )
            return node_id
        else:
            print(f"Unexpected response: {msg}", file=sys.stderr)
            return None


async def _voice_loop(
    server_url: str,
    node_id: str,
    token: str,
    record_seconds: float,
    sample_rate: int,
    channels: int,
    continuous: bool = False,
) -> None:
    """Main voice interaction loop."""
    async with websockets.connect(server_url) as ws:
        # Authenticate.
        await ws.send(
            json.dumps(
                {
                    "type": "auth",
                    "node_id": node_id,
                    "token": token,
                }
            )
        )

        raw = await ws.recv()
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"Malformed auth response: {exc}", file=sys.stderr)
            return

        if msg.get("type") != "auth_ok":
            reason = msg.get("reason", msg.get("type", "unknown"))
            print(f"Authentication failed: {reason}", file=sys.stderr)
            return

        room = msg.get("room", "unknown")
        print(f"Connected as node {node_id} (room: {room})")
        print(f"Press Enter to speak ({record_seconds}s), Ctrl+C to quit.\n")

        while True:
            try:
                if not continuous:
                    input(">>> Press Enter to speak...")
                else:
                    # In continuous mode, small pause between recordings.
                    await asyncio.sleep(0.5)

                print("  Listening...")
                loop = asyncio.get_running_loop()
                pcm_data = await loop.run_in_executor(
                    None,
                    _record_audio,
                    record_seconds,
                    sample_rate,
                    channels,
                )

                if not pcm_data:
                    print("  No audio captured.")
                    continue

                duration_ms = len(pcm_data) / (sample_rate * channels * 2) * 1000
                print(f"  Captured {len(pcm_data)} bytes ({duration_ms:.0f}ms)")

                # Send audio.
                await ws.send(
                    json.dumps(
                        {
                            "type": "audio_start",
                            "sample_rate": sample_rate,
                            "channels": channels,
                            "format": "pcm_s16le",
                        }
                    )
                )

                # Send in chunks.
                chunk_size = 4096
                offset = 0
                while offset < len(pcm_data):
                    chunk = pcm_data[offset : offset + chunk_size]
                    await ws.send(chunk)
                    offset += chunk_size

                await ws.send(json.dumps({"type": "audio_end"}))
                print("  Processing...")

                # Receive response(s).
                audio_chunks: list[bytes] = []
                in_audio = False
                response_text = ""

                while True:
                    raw_resp = await asyncio.wait_for(ws.recv(), timeout=60)

                    if isinstance(raw_resp, bytes):
                        if in_audio:
                            audio_chunks.append(raw_resp)
                        continue

                    try:
                        resp = json.loads(raw_resp)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    resp_type = resp.get("type", "")

                    if resp_type == "transcript":
                        print(f'  You said: "{resp.get("text", "")}"')
                        conf = resp.get("confidence", -1)
                        if conf >= 0:
                            print(f"  Confidence: {conf:.1%}")

                    elif resp_type == "response_text":
                        response_text = resp.get("text", "")
                        print(f"  Missy: {response_text}")

                    elif resp_type == "audio_start":
                        in_audio = True
                        audio_chunks = []

                    elif resp_type == "audio_end":
                        in_audio = False
                        if audio_chunks:
                            wav_data = b"".join(audio_chunks)
                            print(f"  Playing response ({len(wav_data)} bytes)...")
                            await loop.run_in_executor(None, _play_wav, wav_data)
                        break

                    elif resp_type == "error":
                        print(f"  Error: {resp.get('message', 'unknown')}", file=sys.stderr)
                        break

                    else:
                        # Unknown frame type — may mean no audio is coming.
                        break

                print()

            except KeyboardInterrupt:
                print("\nDisconnecting...")
                break
            except TimeoutError:
                print("  Timeout waiting for response.", file=sys.stderr)
                continue
            except EOFError:
                break


def _load_config(path: Path) -> dict[str, Any]:
    """Load edge client config from JSON file."""
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_config(path: Path, config: dict[str, Any]) -> None:
    """Save edge client config to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    """CLI entry point for the edge client."""
    parser = argparse.ArgumentParser(
        description="Missy voice edge client — captures audio and streams to the voice server.",
    )
    parser.add_argument(
        "--server",
        default=_DEFAULT_SERVER_URL,
        help=f"Voice server WebSocket URL (default: {_DEFAULT_SERVER_URL})",
    )
    parser.add_argument("--node-id", help="Authenticated node ID")
    parser.add_argument("--token", help="Authentication token")
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help=f"Config file path (default: {_CONFIG_PATH})",
    )

    # Recording options.
    parser.add_argument(
        "--duration",
        type=float,
        default=_DEFAULT_RECORD_SECONDS,
        help=f"Recording duration in seconds (default: {_DEFAULT_RECORD_SECONDS})",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=_DEFAULT_SAMPLE_RATE,
        help=f"Sample rate in Hz (default: {_DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continuously listen without waiting for Enter",
    )

    # Pairing mode.
    parser.add_argument("--pair", action="store_true", help="Pair a new device")
    parser.add_argument("--name", default="Edge Node", help="Friendly name for pairing")
    parser.add_argument("--room", default="Unknown", help="Room name for pairing")

    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load config file values as defaults.
    config = _load_config(args.config)
    server_url = args.server or config.get("server", _DEFAULT_SERVER_URL)
    node_id = args.node_id or config.get("node_id")
    token = args.token or config.get("token")

    if args.pair:
        result = asyncio.run(_pair_device(server_url, args.name, args.room))
        if result:
            config["server"] = server_url
            config["node_id"] = result
            _save_config(args.config, config)
            print(f"\nNode ID saved to {args.config}")
        return

    if not node_id or not token:
        print(
            "Error: --node-id and --token are required (or set them in config file).\n"
            "To pair a new device: python -m missy.channels.voice.edge_client --pair",
            file=sys.stderr,
        )
        sys.exit(1)

    # Save working config for next time.
    config.update({"server": server_url, "node_id": node_id, "token": token})
    _save_config(args.config, config)

    asyncio.run(
        _voice_loop(
            server_url=server_url,
            node_id=node_id,
            token=token,
            record_seconds=args.duration,
            sample_rate=args.sample_rate,
            channels=_DEFAULT_CHANNELS,
            continuous=args.continuous,
        )
    )


if __name__ == "__main__":
    main()
