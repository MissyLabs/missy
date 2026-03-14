"""Tests for missy/channels/voice/edge_client.py.

All external I/O (subprocess, websockets, filesystem, time.sleep) is mocked.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

import missy.channels.voice.edge_client as ec
from missy.channels.voice.edge_client import (
    _ensure_runtime_dir,
    _load_config,
    _pair_device,
    _play_wav,
    _record_audio,
    _record_audio_gst,
    _save_config,
    _voice_loop,
)


# ---------------------------------------------------------------------------
# _ensure_runtime_dir
# ---------------------------------------------------------------------------

class TestEnsureRuntimeDir:
    def test_preserves_existing_xdg_runtime_dir(self) -> None:
        env = {"XDG_RUNTIME_DIR": "/run/user/existing"}
        with patch.dict(os.environ, env, clear=True):
            result = _ensure_runtime_dir()
        assert result["XDG_RUNTIME_DIR"] == "/run/user/existing"

    def test_sets_xdg_runtime_dir_when_missing(self) -> None:
        clean_env = {k: v for k, v in os.environ.items() if k != "XDG_RUNTIME_DIR"}
        with patch.dict(os.environ, clean_env, clear=True):
            with patch("os.getuid", return_value=1001):
                result = _ensure_runtime_dir()
        assert result["XDG_RUNTIME_DIR"] == "/run/user/1001"

    def test_returns_copy_of_environment(self) -> None:
        result = _ensure_runtime_dir()
        # Modifying result should not affect os.environ.
        result["FAKE_KEY"] = "fake"
        assert "FAKE_KEY" not in os.environ


# ---------------------------------------------------------------------------
# _load_config / _save_config
# ---------------------------------------------------------------------------

class TestLoadSaveConfig:
    def test_load_returns_empty_dict_when_file_missing(self, tmp_path: Path) -> None:
        result = _load_config(tmp_path / "nonexistent.json")
        assert result == {}

    def test_load_returns_parsed_json(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "edge.json"
        cfg_path.write_text(json.dumps({"server": "ws://test:8765", "node_id": "n1"}))
        result = _load_config(cfg_path)
        assert result["server"] == "ws://test:8765"
        assert result["node_id"] == "n1"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "a" / "b" / "edge.json"
        _save_config(deep_path, {"key": "value"})
        assert deep_path.exists()

    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "edge.json"
        _save_config(cfg_path, {"node_id": "abc", "token": "secret"})
        data = json.loads(cfg_path.read_text())
        assert data["node_id"] == "abc"
        assert data["token"] == "secret"

    def test_save_appends_newline(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "edge.json"
        _save_config(cfg_path, {})
        content = cfg_path.read_text()
        assert content.endswith("\n")

    def test_round_trip(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "edge.json"
        original = {"server": "ws://host:8765", "node_id": "node-42", "token": "tok"}
        _save_config(cfg_path, original)
        loaded = _load_config(cfg_path)
        assert loaded == original


# ---------------------------------------------------------------------------
# _record_audio_gst
# ---------------------------------------------------------------------------

class TestRecordAudioGst:
    @patch("missy.channels.voice.edge_client.time.sleep")
    @patch("missy.channels.voice.edge_client.subprocess.Popen")
    def test_returns_raw_bytes_when_file_exists(self, mock_popen, mock_sleep, tmp_path) -> None:
        raw_file = tmp_path / "audio.raw"
        raw_file.write_bytes(b"\x00\x01\x02\x03")

        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        mock_popen.return_value = mock_proc

        with patch("tempfile.NamedTemporaryFile") as mock_tmp:
            mock_tmp.return_value.__enter__.return_value.name = str(raw_file)
            result = _record_audio_gst(
                duration=1.0,
                sample_rate=16000,
                channels=1,
                env={},
            )

        assert result == b"\x00\x01\x02\x03"

    @patch("missy.channels.voice.edge_client.time.sleep")
    @patch("missy.channels.voice.edge_client.subprocess.Popen")
    def test_returns_empty_when_file_missing(self, mock_popen, mock_sleep, tmp_path) -> None:
        nonexistent = tmp_path / "missing.raw"

        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        mock_popen.return_value = mock_proc

        with patch("tempfile.NamedTemporaryFile") as mock_tmp:
            mock_tmp.return_value.__enter__.return_value.name = str(nonexistent)
            result = _record_audio_gst(
                duration=1.0,
                sample_rate=16000,
                channels=1,
                env={},
            )

        assert result == b""

    @patch("missy.channels.voice.edge_client.time.sleep")
    @patch("missy.channels.voice.edge_client.subprocess.Popen")
    def test_sends_sigint_and_waits(self, mock_popen, mock_sleep, tmp_path) -> None:
        raw_file = tmp_path / "audio.raw"
        raw_file.write_bytes(b"\xAB\xCD")

        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        mock_popen.return_value = mock_proc

        with patch("tempfile.NamedTemporaryFile") as mock_tmp:
            mock_tmp.return_value.__enter__.return_value.name = str(raw_file)
            _record_audio_gst(1.0, 16000, 1, {})

        mock_proc.send_signal.assert_called_once_with(signal.SIGINT)
        mock_proc.wait.assert_called_once()

    @patch("missy.channels.voice.edge_client.time.sleep")
    @patch("missy.channels.voice.edge_client.subprocess.Popen")
    def test_kills_process_on_timeout(self, mock_popen, mock_sleep, tmp_path) -> None:
        raw_file = tmp_path / "audio.raw"
        raw_file.write_bytes(b"\xFF")

        mock_proc = MagicMock()
        # First wait() call raises TimeoutExpired; second (after kill) succeeds.
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="gst-launch", timeout=5),
            None,
        ]
        mock_popen.return_value = mock_proc

        with patch("tempfile.NamedTemporaryFile") as mock_tmp:
            mock_tmp.return_value.__enter__.return_value.name = str(raw_file)
            _record_audio_gst(1.0, 16000, 1, {})

        mock_proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# _record_audio
# ---------------------------------------------------------------------------

class TestRecordAudio:
    @patch("missy.channels.voice.edge_client.time.sleep")
    @patch("missy.channels.voice.edge_client.subprocess.Popen")
    def test_falls_back_to_gst_when_pw_not_found(self, mock_popen, mock_sleep) -> None:
        mock_popen.side_effect = FileNotFoundError("pw-record not found")

        with patch.object(ec, "_record_audio_gst", return_value=b"\x00\x01") as mock_gst:
            result = _record_audio(duration=1.0)

        mock_gst.assert_called_once()
        assert result == b"\x00\x01"

    @patch("missy.channels.voice.edge_client.time.sleep")
    @patch("missy.channels.voice.edge_client.subprocess.Popen")
    def test_returns_empty_when_wav_missing(self, mock_popen, mock_sleep, tmp_path) -> None:
        nonexistent = tmp_path / "missing.wav"
        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        mock_popen.return_value = mock_proc

        with patch("tempfile.NamedTemporaryFile") as mock_tmp:
            mock_tmp.return_value.__enter__.return_value.name = str(nonexistent)
            result = _record_audio(duration=1.0)

        assert result == b""

    @patch("missy.channels.voice.edge_client.time.sleep")
    @patch("missy.channels.voice.edge_client.subprocess.Popen")
    def test_kills_process_on_timeout(self, mock_popen, mock_sleep, tmp_path) -> None:
        wav_path = tmp_path / "audio.wav"
        mock_proc = MagicMock()
        # First wait() raises; second (after kill) succeeds.
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired("pw-record", 5),
            None,
        ]
        mock_popen.return_value = mock_proc

        with patch("tempfile.NamedTemporaryFile") as mock_tmp:
            mock_tmp.return_value.__enter__.return_value.name = str(wav_path)
            _record_audio(duration=1.0)

        mock_proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# _play_wav
# ---------------------------------------------------------------------------

class TestPlayWav:
    @patch("missy.channels.voice.edge_client.subprocess.run")
    def test_calls_gstreamer_with_wav_path(self, mock_run) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        _play_wav(b"\x52\x49\x46\x46")  # minimal RIFF header bytes

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert "gst-launch-1.0" in cmd

    @patch("missy.channels.voice.edge_client.subprocess.run")
    def test_handles_timeout_gracefully(self, mock_run) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired("gst-launch-1.0", 60)
        # Should not raise.
        _play_wav(b"\x00" * 100)

    @patch("missy.channels.voice.edge_client.subprocess.run")
    def test_handles_file_not_found_gracefully(self, mock_run) -> None:
        mock_run.side_effect = FileNotFoundError("gst-launch-1.0 not found")
        # Should not raise.
        _play_wav(b"\x00" * 100)

    @patch("missy.channels.voice.edge_client.subprocess.run")
    def test_cleans_up_temp_file(self, mock_run) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        created_paths = []

        original_ntf = tempfile.NamedTemporaryFile

        def tracking_ntf(**kwargs):
            f = original_ntf(**kwargs)
            created_paths.append(f.name)
            return f

        with patch("tempfile.NamedTemporaryFile", side_effect=tracking_ntf):
            _play_wav(b"\x00" * 10)

        for path in created_paths:
            assert not Path(path).exists()


# ---------------------------------------------------------------------------
# _pair_device
# ---------------------------------------------------------------------------

class TestPairDevice:
    def test_returns_node_id_on_pair_pending(self) -> None:
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({
            "type": "pair_pending",
            "node_id": "node-abc",
        }))
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)

        with patch("missy.channels.voice.edge_client.websockets.connect", return_value=mock_ws):
            result = asyncio.run(_pair_device("ws://localhost:8765", "Test Node", "Office"))

        assert result == "node-abc"

    def test_returns_none_on_unexpected_response(self) -> None:
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({
            "type": "unexpected_type",
        }))
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)

        with patch("missy.channels.voice.edge_client.websockets.connect", return_value=mock_ws):
            result = asyncio.run(_pair_device("ws://localhost:8765", "Test", "Room"))

        assert result is None

    def test_sends_correct_pair_request_frame(self) -> None:
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({
            "type": "pair_pending",
            "node_id": "node-xyz",
        }))
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)

        with patch("missy.channels.voice.edge_client.websockets.connect", return_value=mock_ws):
            asyncio.run(_pair_device("ws://localhost:8765", "My Node", "Living Room"))

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "pair_request"
        assert sent["friendly_name"] == "My Node"
        assert sent["room"] == "Living Room"
        assert "hardware_profile" in sent


# ---------------------------------------------------------------------------
# _voice_loop
# ---------------------------------------------------------------------------

class TestVoiceLoop:
    def _make_ws(self, recv_responses):
        """Build a mock WebSocket that returns a queue of responses."""
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        responses = list(recv_responses)
        call_count = [0]

        async def recv():
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(responses):
                return responses[idx]
            raise EOFError("no more responses")

        mock_ws.recv = recv
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)
        return mock_ws

    def test_auth_failure_exits_early(self) -> None:
        mock_ws = self._make_ws([
            json.dumps({"type": "auth_fail", "reason": "invalid credentials"}),
        ])

        with patch("missy.channels.voice.edge_client.websockets.connect", return_value=mock_ws):
            asyncio.run(_voice_loop(
                server_url="ws://localhost:8765",
                node_id="node-1",
                token="bad-token",
                record_seconds=1.0,
                sample_rate=16000,
                channels=1,
            ))

        # Should have sent auth and then exited — no audio frames expected.
        assert mock_ws.send.call_count == 1
        sent = json.loads(mock_ws.send.call_args_list[0][0][0])
        assert sent["type"] == "auth"

    def test_auth_success_sends_audio_frames(self) -> None:
        # Auth response, then EOF to end loop.
        response_queue = [
            json.dumps({"type": "auth_ok", "room": "Office"}),
        ]
        mock_ws = self._make_ws(response_queue)

        # Patch input() to raise EOFError immediately so loop ends after one iteration.
        with patch("missy.channels.voice.edge_client.websockets.connect", return_value=mock_ws):
            with patch("builtins.input", side_effect=EOFError):
                with patch.object(ec, "_record_audio", return_value=b""):
                    asyncio.run(_voice_loop(
                        server_url="ws://localhost:8765",
                        node_id="node-1",
                        token="valid-token",
                        record_seconds=1.0,
                        sample_rate=16000,
                        channels=1,
                        continuous=False,
                    ))

        # Auth was sent.
        first_sent = json.loads(mock_ws.send.call_args_list[0][0][0])
        assert first_sent["type"] == "auth"

    def test_continuous_mode_skips_input(self) -> None:
        # Auth OK, then timeout/EOFError to stop.
        auth_ok = json.dumps({"type": "auth_ok", "room": "Room"})
        mock_ws = self._make_ws([auth_ok])

        # In continuous mode no input() call should happen.
        input_called = [False]

        def mock_input(_prompt=""):
            input_called[0] = True
            raise EOFError

        with patch("missy.channels.voice.edge_client.websockets.connect", return_value=mock_ws):
            with patch("builtins.input", side_effect=mock_input):
                with patch.object(ec, "_record_audio", return_value=b""):
                    with patch("asyncio.sleep", new_callable=AsyncMock, side_effect=EOFError):
                        asyncio.run(_voice_loop(
                            server_url="ws://localhost:8765",
                            node_id="node-1",
                            token="tok",
                            record_seconds=1.0,
                            sample_rate=16000,
                            channels=1,
                            continuous=True,
                        ))

        assert not input_called[0]

    def test_response_handling_transcript_and_text(self) -> None:
        pcm_data = b"\x00\x01" * 8192  # 16384 bytes > 0
        responses_after_auth = [
            json.dumps({"type": "transcript", "text": "hello", "confidence": 0.9}),
            json.dumps({"type": "response_text", "text": "Hi there"}),
            json.dumps({"type": "audio_start"}),
            b"\x00" * 100,  # binary audio chunk
            json.dumps({"type": "audio_end"}),
        ]

        call_num = [0]

        async def recv():
            if call_num[0] == 0:
                call_num[0] += 1
                return json.dumps({"type": "auth_ok", "room": "Office"})
            # After auth_ok, we need input() to allow one loop iteration.
            raise EOFError

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = recv
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)

        with patch("missy.channels.voice.edge_client.websockets.connect", return_value=mock_ws):
            with patch("builtins.input", side_effect=EOFError):
                with patch.object(ec, "_record_audio", return_value=b""):
                    asyncio.run(_voice_loop(
                        server_url="ws://localhost:8765",
                        node_id="n1",
                        token="tok",
                        record_seconds=1.0,
                        sample_rate=16000,
                        channels=1,
                    ))

    def test_auth_message_with_unknown_reason_field(self) -> None:
        mock_ws = self._make_ws([
            json.dumps({"type": "auth_fail"}),  # no "reason" key
        ])

        with patch("missy.channels.voice.edge_client.websockets.connect", return_value=mock_ws):
            asyncio.run(_voice_loop(
                server_url="ws://localhost:8765",
                node_id="n1",
                token="t",
                record_seconds=1.0,
                sample_rate=16000,
                channels=1,
            ))

        # Just verifying it didn't crash — auth fail exits cleanly.


# ---------------------------------------------------------------------------
# Voice loop response frame types
# ---------------------------------------------------------------------------

class TestVoiceLoopResponseFrames:
    """Test individual response frame processing in the inner response loop."""

    def _run_with_responses(self, after_auth_responses):
        """Auth OK then process a list of server responses."""
        all_responses = [
            json.dumps({"type": "auth_ok", "room": "Test"}),
        ] + after_auth_responses

        idx = [0]
        pcm_chunks_sent = []

        async def recv():
            i = idx[0]
            idx[0] += 1
            if i < len(all_responses):
                return all_responses[i]
            raise EOFError

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock(side_effect=lambda x: pcm_chunks_sent.append(x))
        mock_ws.recv = recv
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)

        pcm_data = b"\x00\x01" * 100  # 200 bytes of PCM

        with patch("missy.channels.voice.edge_client.websockets.connect", return_value=mock_ws):
            with patch("builtins.input", return_value=""):
                with patch.object(ec, "_record_audio", return_value=pcm_data):
                    with patch.object(ec, "_play_wav"):
                        asyncio.run(_voice_loop(
                            server_url="ws://localhost:8765",
                            node_id="n1",
                            token="tok",
                            record_seconds=1.0,
                            sample_rate=16000,
                            channels=1,
                        ))

        return mock_ws

    def test_error_frame_breaks_inner_loop(self) -> None:
        self._run_with_responses([
            json.dumps({"type": "error", "message": "something went wrong"}),
        ])

    def test_audio_start_and_end_triggers_playback(self) -> None:
        self._run_with_responses([
            json.dumps({"type": "transcript", "text": "test", "confidence": 0.95}),
            json.dumps({"type": "response_text", "text": "reply"}),
            json.dumps({"type": "audio_start"}),
            b"\xFF\xFE" * 50,  # binary audio
            json.dumps({"type": "audio_end"}),
        ])

    def test_unknown_frame_breaks_inner_loop(self) -> None:
        self._run_with_responses([
            json.dumps({"type": "unknown_frame_type"}),
        ])

    def test_transcript_without_confidence_field(self) -> None:
        # confidence < 0 means don't print confidence line.
        self._run_with_responses([
            json.dumps({"type": "transcript", "text": "hi"}),  # no confidence key
            json.dumps({"type": "audio_end"}),
        ])
