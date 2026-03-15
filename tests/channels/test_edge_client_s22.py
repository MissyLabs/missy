"""Session 22 edge client coverage tests.

Targets uncovered lines in missy/channels/voice/edge_client.py:
- lines 271-273: malformed auth response JSON
- lines 347-348: malformed response frame in inner loop
- lines 398-399: _load_config with corrupt JSON
- line 501: if __name__ == "__main__" (not testable, skip)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import missy.channels.voice.edge_client as ec
from missy.channels.voice.edge_client import _load_config, _voice_loop


class TestMalformedAuthResponse:
    """Lines 271-273: json.loads fails on auth response."""

    def test_malformed_auth_response_exits_cleanly(self) -> None:
        """When auth response is not valid JSON, _voice_loop returns early."""
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        # Return invalid JSON for auth response
        mock_ws.recv = AsyncMock(return_value="NOT VALID JSON {{{")
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)

        with patch("missy.channels.voice.edge_client.websockets.connect", return_value=mock_ws):
            asyncio.run(
                _voice_loop(
                    server_url="ws://localhost:8765",
                    node_id="n1",
                    token="tok",
                    record_seconds=1.0,
                    sample_rate=16000,
                    channels=1,
                )
            )

        # Only auth frame should have been sent; function returned early.
        assert mock_ws.send.call_count == 1

    def test_auth_response_none_type_error(self) -> None:
        """When auth response is None, TypeError is caught."""
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=None)
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)

        with patch("missy.channels.voice.edge_client.websockets.connect", return_value=mock_ws):
            asyncio.run(
                _voice_loop(
                    server_url="ws://localhost:8765",
                    node_id="n1",
                    token="tok",
                    record_seconds=1.0,
                    sample_rate=16000,
                    channels=1,
                )
            )


class TestMalformedResponseFrame:
    """Lines 347-348: json.loads fails on response frame in inner loop."""

    def test_malformed_response_frame_continues(self) -> None:
        """Invalid JSON in inner response loop is skipped via continue."""
        responses = [
            json.dumps({"type": "auth_ok", "room": "Test"}),
            # After auth, inner loop receives: invalid JSON, then audio_end to break
            "INVALID JSON >>>",
            json.dumps({"type": "audio_end"}),
        ]
        idx = [0]

        async def recv():
            i = idx[0]
            idx[0] += 1
            if i < len(responses):
                return responses[i]
            raise EOFError

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = recv
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)

        pcm_data = b"\x01\x02" * 100

        with (
            patch("missy.channels.voice.edge_client.websockets.connect", return_value=mock_ws),
            patch("builtins.input", return_value=""),
            patch.object(ec, "_record_audio", return_value=pcm_data),
            patch.object(ec, "_play_wav"),
        ):
            asyncio.run(
                _voice_loop(
                    server_url="ws://localhost:8765",
                    node_id="n1",
                    token="tok",
                    record_seconds=1.0,
                    sample_rate=16000,
                    channels=1,
                )
            )


class TestLoadConfigCorrupt:
    """Lines 398-399: _load_config with corrupt JSON file."""

    def test_corrupt_json_returns_empty_dict(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "edge.json"
        cfg_path.write_text("{corrupt json!!!}")
        assert _load_config(cfg_path) == {}

    def test_os_error_returns_empty_dict(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "edge.json"
        cfg_path.write_text("{}")
        with patch.object(Path, "read_text", side_effect=OSError("permission denied")):
            assert _load_config(cfg_path) == {}


class TestPairDeviceMalformedResponse:
    """Lines 227-229 in _pair_device: malformed pairing response."""

    def test_malformed_pairing_response_returns_none(self) -> None:
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(return_value="NOT JSON")
        mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_ws.__aexit__ = AsyncMock(return_value=False)

        with patch("missy.channels.voice.edge_client.websockets.connect", return_value=mock_ws):
            result = asyncio.run(ec._pair_device("ws://localhost:8765", "Test", "Room"))

        assert result is None
