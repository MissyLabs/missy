"""Extended tests for Discord gateway.py and rest.py targeting low-coverage paths.

Coverage targets
----------------
gateway.py (31% -> target 80%+):
  - DiscordGatewayClient.connect
  - DiscordGatewayClient.disconnect
  - DiscordGatewayClient.run (reconnect loop)
  - DiscordGatewayClient._receive_loop (JSON decode error path)
  - DiscordGatewayClient._handle_payload (all opcodes)
  - DiscordGatewayClient._handle_dispatch (READY, RESUMED, forwarded events, callback error)
  - DiscordGatewayClient._start_heartbeat (cancels previous task)
  - DiscordGatewayClient._heartbeat_loop / _send_heartbeat
  - DiscordGatewayClient._identify_or_resume / _send_identify / _send_resume
  - DiscordGatewayClient._emit_audit (exception swallowing)

rest.py (58% -> target 85%+):
  - DiscordRestClient.send_message retry on 429/502/503/504
  - DiscordRestClient.send_message exhausted retries raises RuntimeError
  - DiscordRestClient.send_message exception path retry and re-raise
  - DiscordRestClient.upload_file success and FileNotFoundError
  - DiscordRestClient.add_reaction URL encoding and HTTP PUT
  - DiscordRestClient.trigger_typing exception swallowed
  - DiscordRestClient._headers extra headers merged
  - _mask_mentions helper

All WebSocket and HTTP I/O is mocked.  No real network connections are made.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.discord.gateway import (
    _OP_DISPATCH,
    _OP_HEARTBEAT,
    _OP_HEARTBEAT_ACK,
    _OP_HELLO,
    _OP_IDENTIFY,
    _OP_INVALID_SESSION,
    _OP_RECONNECT,
    _OP_RESUME,
    DiscordGatewayClient,
)
from missy.channels.discord.rest import BASE, DiscordRestClient, _mask_mentions

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_gateway(on_message=None) -> DiscordGatewayClient:
    """Return a DiscordGatewayClient with a no-op on_message callback."""
    if on_message is None:

        async def _noop(payload: dict) -> None:
            pass

        on_message = _noop
    return DiscordGatewayClient(bot_token="testtoken", on_message=on_message)


def _make_rest(mock_http: MagicMock | None = None) -> tuple[DiscordRestClient, MagicMock]:
    if mock_http is None:
        mock_http = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"id": "msg-1"}
        resp.raise_for_status.return_value = None
        mock_http.post.return_value = resp
        mock_http.get.return_value = resp
        mock_http.put.return_value = resp
    client = DiscordRestClient(bot_token="Bot testtoken", http_client=mock_http)
    return client, mock_http


# ===========================================================================
# Gateway tests
# ===========================================================================


class TestGatewayTokenPrefix:
    def test_token_prefix_added_when_missing(self):
        gw = DiscordGatewayClient(bot_token="rawtoken", on_message=AsyncMock())
        assert gw._token == "Bot rawtoken"

    def test_token_prefix_not_doubled(self):
        gw = DiscordGatewayClient(bot_token="Bot rawtoken", on_message=AsyncMock())
        assert gw._token == "Bot rawtoken"


class TestGatewayBotUserIdProperty:
    def test_bot_user_id_initially_none(self):
        gw = _make_gateway()
        assert gw.bot_user_id is None

    def test_bot_user_id_reflects_internal_state(self):
        gw = _make_gateway()
        gw._bot_user_id = "12345"
        assert gw.bot_user_id == "12345"


class TestGatewayConnect:
    @pytest.mark.asyncio
    async def test_connect_opens_websocket(self):
        gw = _make_gateway()
        mock_ws = MagicMock()

        async def _fake_connect(url):
            return mock_ws

        with patch("websockets.connect", side_effect=_fake_connect) as mock_connect:
            await gw.connect()

        mock_connect.assert_called_once_with(gw._gateway_url)
        assert gw._ws is mock_ws

    @pytest.mark.asyncio
    async def test_connect_uses_resume_url_when_available(self):
        gw = _make_gateway()
        gw._resume_gateway_url = "wss://resume.discord.gg"

        async def _fake_connect(url):
            return MagicMock()

        with patch("websockets.connect", side_effect=_fake_connect) as mock_connect:
            await gw.connect()

        mock_connect.assert_called_once_with("wss://resume.discord.gg")

    @pytest.mark.asyncio
    async def test_connect_raises_when_websockets_missing(self):
        gw = _make_gateway()

        with (
            patch.dict("sys.modules", {"websockets": None}),
            pytest.raises(RuntimeError, match="websockets"),
        ):
            await gw.connect()

    @pytest.mark.asyncio
    async def test_connect_emits_audit_event(self):
        gw = _make_gateway()
        audit_calls: list[tuple] = []

        def _capture(event_type, result, detail):
            audit_calls.append((event_type, result, detail))

        gw._emit_audit = _capture  # type: ignore[method-assign]

        async def _fake_connect(url):
            return MagicMock()

        with patch("websockets.connect", side_effect=_fake_connect):
            await gw.connect()

        assert any(et == "discord.gateway.connect" for et, _, _ in audit_calls)


class TestGatewayDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_stops_running_flag(self):
        gw = _make_gateway()
        gw._running = True
        gw._ws = AsyncMock()
        await gw.disconnect()
        assert gw._running is False

    @pytest.mark.asyncio
    async def test_disconnect_cancels_heartbeat_task(self):
        gw = _make_gateway()
        gw._ws = AsyncMock()

        # Create a real task that we can cancel
        async def _sleep_forever():
            await asyncio.sleep(9999)

        task = asyncio.create_task(_sleep_forever())
        gw._heartbeat_task = task

        await gw.disconnect()

        assert task.cancelled()
        assert gw._heartbeat_task is None

    @pytest.mark.asyncio
    async def test_disconnect_closes_websocket(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        gw._ws = mock_ws

        await gw.disconnect()

        mock_ws.close.assert_called_once()
        assert gw._ws is None

    @pytest.mark.asyncio
    async def test_disconnect_tolerates_ws_close_error(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        mock_ws.close.side_effect = OSError("socket gone")
        gw._ws = mock_ws

        # Should not raise
        await gw.disconnect()
        assert gw._ws is None

    @pytest.mark.asyncio
    async def test_disconnect_when_no_ws(self):
        gw = _make_gateway()
        gw._ws = None
        # Should not raise
        await gw.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_emits_audit_event(self):
        gw = _make_gateway()
        gw._ws = AsyncMock()
        audit_calls: list[tuple] = []

        def _capture(et, r, d):
            audit_calls.append((et, r, d))

        gw._emit_audit = _capture  # type: ignore[method-assign]

        await gw.disconnect()

        assert any(et == "discord.gateway.disconnect" for et, _, _ in audit_calls)


class TestGatewayRunReconnect:
    @pytest.mark.asyncio
    async def test_run_calls_connect_then_receive_loop(self):
        gw = _make_gateway()
        connect_calls = []
        loop_calls = []

        async def _fake_connect():
            connect_calls.append(1)

        async def _fake_loop():
            loop_calls.append(1)
            gw._running = False  # Stop after first iteration

        gw.connect = _fake_connect  # type: ignore[method-assign]
        gw._receive_loop = _fake_loop  # type: ignore[method-assign]

        await gw.run()

        assert len(connect_calls) == 1
        assert len(loop_calls) == 1

    @pytest.mark.asyncio
    async def test_run_reconnects_on_exception(self):
        gw = _make_gateway()
        call_count = 0

        async def _fake_connect():
            pass

        async def _fake_loop():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError("connection reset")
            gw._running = False

        gw.connect = _fake_connect  # type: ignore[method-assign]
        gw._receive_loop = _fake_loop  # type: ignore[method-assign]

        audit_calls: list[tuple] = []
        gw._emit_audit = lambda et, r, d: audit_calls.append((et, r, d))  # type: ignore[method-assign]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await gw.run()

        assert call_count == 2
        error_events = [x for x in audit_calls if x[1] == "error"]
        assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_run_stops_cleanly_when_not_running(self):
        gw = _make_gateway()

        async def _fake_connect():
            pass

        async def _fake_loop():
            # Simulate disconnect() setting _running=False then raising
            gw._running = False
            raise OSError("closed")

        gw.connect = _fake_connect  # type: ignore[method-assign]
        gw._receive_loop = _fake_loop  # type: ignore[method-assign]

        await gw.run()
        # No infinite reconnect loop - should exit cleanly


class TestGatewayReceiveLoop:
    @pytest.mark.asyncio
    async def test_receive_loop_calls_handle_payload(self):
        gw = _make_gateway()
        handled: list[dict] = []

        async def _fake_handle(payload):
            handled.append(payload)

        gw._handle_payload = _fake_handle  # type: ignore[method-assign]
        gw._running = True

        payload_obj = {"op": _OP_HEARTBEAT_ACK}

        async def _fake_ws_iter():
            yield json.dumps(payload_obj)

        gw._ws = _fake_ws_iter()

        await gw._receive_loop()

        assert len(handled) == 1
        assert handled[0]["op"] == _OP_HEARTBEAT_ACK

    @pytest.mark.asyncio
    async def test_receive_loop_skips_invalid_json(self):
        gw = _make_gateway()
        handled: list[dict] = []

        async def _fake_handle(payload):
            handled.append(payload)

        gw._handle_payload = _fake_handle  # type: ignore[method-assign]
        gw._running = True

        async def _fake_ws_iter():
            yield "not-valid-json{{{"
            yield json.dumps({"op": _OP_HEARTBEAT_ACK})

        gw._ws = _fake_ws_iter()

        await gw._receive_loop()

        # The invalid JSON message is skipped, the valid one is processed
        assert len(handled) == 1

    @pytest.mark.asyncio
    async def test_receive_loop_stops_when_not_running(self):
        gw = _make_gateway()
        handled: list[dict] = []

        async def _fake_handle(payload):
            handled.append(payload)
            gw._running = False  # Stop after first message

        gw._handle_payload = _fake_handle  # type: ignore[method-assign]
        gw._running = True

        async def _fake_ws_iter():
            yield json.dumps({"op": _OP_HEARTBEAT_ACK})
            yield json.dumps({"op": _OP_HEARTBEAT_ACK})

        gw._ws = _fake_ws_iter()

        await gw._receive_loop()

        # _running became False after first handle, second message should not be processed
        assert len(handled) == 1


class TestGatewayHandlePayloadHello:
    @pytest.mark.asyncio
    async def test_hello_starts_heartbeat_and_identifies(self):
        gw = _make_gateway()
        heartbeat_started = []
        identified = []

        async def _fake_start_heartbeat(interval):
            heartbeat_started.append(interval)

        async def _fake_identify_or_resume():
            identified.append(1)

        gw._start_heartbeat = _fake_start_heartbeat  # type: ignore[method-assign]
        gw._identify_or_resume = _fake_identify_or_resume  # type: ignore[method-assign]

        await gw._handle_payload({"op": _OP_HELLO, "d": {"heartbeat_interval": 45000}})

        assert heartbeat_started == [45.0]  # converted from ms to seconds
        assert identified == [1]

    @pytest.mark.asyncio
    async def test_hello_updates_sequence_number(self):
        gw = _make_gateway()
        gw._start_heartbeat = AsyncMock()  # type: ignore[method-assign]
        gw._identify_or_resume = AsyncMock()  # type: ignore[method-assign]

        await gw._handle_payload({"op": _OP_HELLO, "d": {"heartbeat_interval": 1000}, "s": 42})

        assert gw._sequence == 42


class TestGatewayHandlePayloadDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_routes_to_handle_dispatch(self):
        gw = _make_gateway()
        dispatched: list[tuple] = []

        async def _fake_dispatch(event_name, data):
            dispatched.append((event_name, data))

        gw._handle_dispatch = _fake_dispatch  # type: ignore[method-assign]

        await gw._handle_payload(
            {
                "op": _OP_DISPATCH,
                "t": "MESSAGE_CREATE",
                "d": {"content": "hi"},
                "s": 7,
            }
        )

        assert len(dispatched) == 1
        assert dispatched[0] == ("MESSAGE_CREATE", {"content": "hi"})
        assert gw._sequence == 7


class TestGatewayHandlePayloadHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_request_triggers_immediate_send(self):
        gw = _make_gateway()
        sent: list[int] = []

        async def _fake_send():
            sent.append(1)

        gw._send_heartbeat = _fake_send  # type: ignore[method-assign]

        await gw._handle_payload({"op": _OP_HEARTBEAT, "d": None})

        assert sent == [1]


class TestGatewayHandlePayloadHeartbeatAck:
    @pytest.mark.asyncio
    async def test_heartbeat_ack_is_handled_without_error(self):
        gw = _make_gateway()
        # OP_HEARTBEAT_ACK should be a no-op (just debug log)
        await gw._handle_payload({"op": _OP_HEARTBEAT_ACK, "d": None})


class TestGatewayHandlePayloadReconnect:
    @pytest.mark.asyncio
    async def test_reconnect_closes_websocket(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        gw._ws = mock_ws

        await gw._handle_payload({"op": _OP_RECONNECT, "d": None})

        mock_ws.close.assert_called_once()


class TestGatewayHandlePayloadInvalidSession:
    @pytest.mark.asyncio
    async def test_invalid_session_non_resumable_clears_state(self):
        gw = _make_gateway()
        gw._discord_session_id = "old-session"
        gw._resume_gateway_url = "wss://old.resume"
        gw._sequence = 99
        mock_ws = AsyncMock()
        gw._ws = mock_ws

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await gw._handle_payload({"op": _OP_INVALID_SESSION, "d": False})

        assert gw._discord_session_id is None
        assert gw._resume_gateway_url is None
        assert gw._sequence is None
        mock_ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_session_resumable_keeps_state(self):
        gw = _make_gateway()
        gw._discord_session_id = "keep-session"
        gw._resume_gateway_url = "wss://keep.resume"
        gw._sequence = 55
        mock_ws = AsyncMock()
        gw._ws = mock_ws

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await gw._handle_payload({"op": _OP_INVALID_SESSION, "d": True})

        assert gw._discord_session_id == "keep-session"
        assert gw._resume_gateway_url == "wss://keep.resume"
        assert gw._sequence == 55
        mock_ws.close.assert_called_once()


class TestGatewayHandlePayloadUnknownOpcode:
    @pytest.mark.asyncio
    async def test_unknown_opcode_does_not_raise(self):
        gw = _make_gateway()
        # Opcode 99 is not defined; should log debug and continue
        await gw._handle_payload({"op": 99, "d": None})


class TestGatewayHandleDispatch:
    @pytest.mark.asyncio
    async def test_ready_event_stores_session_info(self):
        gw = _make_gateway()
        audit_calls: list[tuple] = []
        gw._emit_audit = lambda et, r, d: audit_calls.append((et, r, d))  # type: ignore[method-assign]

        ready_data = {
            "session_id": "sess-abc",
            "resume_gateway_url": "wss://resume.gateway.gg",
            "user": {"id": "bot-id-1", "username": "MissyBot", "discriminator": "0001"},
        }
        await gw._handle_dispatch("READY", ready_data)

        assert gw._discord_session_id == "sess-abc"
        assert gw._resume_gateway_url == "wss://resume.gateway.gg"
        assert gw._bot_user_id == "bot-id-1"
        assert any(et == "discord.gateway.connect" for et, _, _ in audit_calls)

    @pytest.mark.asyncio
    async def test_ready_event_emits_audit_with_bot_user_id(self):
        gw = _make_gateway()
        audit_details: list[dict] = []
        gw._emit_audit = lambda et, r, d: audit_details.append(d)  # type: ignore[method-assign]

        await gw._handle_dispatch(
            "READY",
            {
                "session_id": "s",
                "resume_gateway_url": "wss://r",
                "user": {"id": "u-42", "username": "Bot", "discriminator": "0"},
            },
        )

        assert any(d.get("bot_user_id") == "u-42" for d in audit_details)

    @pytest.mark.asyncio
    async def test_resumed_event_emits_audit(self):
        gw = _make_gateway()
        audit_calls: list[tuple] = []
        gw._emit_audit = lambda et, r, d: audit_calls.append((et, r, d))  # type: ignore[method-assign]

        await gw._handle_dispatch("RESUMED", {})

        assert any(et == "discord.gateway.session_resumed" for et, _, _ in audit_calls)

    @pytest.mark.asyncio
    async def test_message_create_forwarded_to_callback(self):
        received: list[dict] = []

        async def _callback(payload: dict) -> None:
            received.append(payload)

        gw = _make_gateway(on_message=_callback)
        await gw._handle_dispatch("MESSAGE_CREATE", {"content": "hello"})

        assert len(received) == 1
        assert received[0]["t"] == "MESSAGE_CREATE"
        assert received[0]["d"]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_guild_create_forwarded_to_callback(self):
        received: list[dict] = []

        async def _callback(payload: dict) -> None:
            received.append(payload)

        gw = _make_gateway(on_message=_callback)
        await gw._handle_dispatch("GUILD_CREATE", {"id": "guild-1"})

        assert len(received) == 1
        assert received[0]["t"] == "GUILD_CREATE"

    @pytest.mark.asyncio
    async def test_interaction_create_forwarded_to_callback(self):
        received: list[dict] = []

        async def _callback(payload: dict) -> None:
            received.append(payload)

        gw = _make_gateway(on_message=_callback)
        await gw._handle_dispatch("INTERACTION_CREATE", {"type": 2})

        assert len(received) == 1
        assert received[0]["t"] == "INTERACTION_CREATE"

    @pytest.mark.asyncio
    async def test_unknown_event_not_forwarded_to_callback(self):
        received: list[dict] = []

        async def _callback(payload: dict) -> None:
            received.append(payload)

        gw = _make_gateway(on_message=_callback)
        await gw._handle_dispatch("PRESENCE_UPDATE", {"status": "online"})

        # PRESENCE_UPDATE is not in the forwarded set
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_callback_exception_is_caught_not_propagated(self):
        async def _bad_callback(payload: dict) -> None:
            raise RuntimeError("callback exploded")

        gw = _make_gateway(on_message=_bad_callback)
        # Should not raise
        await gw._handle_dispatch("MESSAGE_CREATE", {"content": "boom"})


class TestGatewayStartHeartbeat:
    @pytest.mark.asyncio
    async def test_start_heartbeat_creates_task(self):
        gw = _make_gateway()

        async def _short_loop(interval):
            await asyncio.sleep(0)

        with patch.object(gw, "_heartbeat_loop", side_effect=_short_loop):
            await gw._start_heartbeat(30.0)

        assert gw._heartbeat_task is not None

    @pytest.mark.asyncio
    async def test_start_heartbeat_cancels_previous_task(self):
        gw = _make_gateway()

        async def _sleep_forever():
            await asyncio.sleep(9999)

        old_task = asyncio.create_task(_sleep_forever())
        gw._heartbeat_task = old_task

        async def _short_loop(interval):
            await asyncio.sleep(0)

        with patch.object(gw, "_heartbeat_loop", side_effect=_short_loop):
            await gw._start_heartbeat(30.0)

        # Old task should have been cancelled
        assert old_task.cancelled()


class TestGatewaySendHeartbeat:
    @pytest.mark.asyncio
    async def test_send_heartbeat_sends_correct_payload(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        gw._ws = mock_ws
        gw._sequence = 15

        audit_calls: list[tuple] = []
        gw._emit_audit = lambda et, r, d: audit_calls.append((et, r, d))  # type: ignore[method-assign]

        await gw._send_heartbeat()

        mock_ws.send.assert_called_once()
        sent_payload = json.loads(mock_ws.send.call_args[0][0])
        assert sent_payload["op"] == _OP_HEARTBEAT
        assert sent_payload["d"] == 15

    @pytest.mark.asyncio
    async def test_send_heartbeat_emits_audit_event(self):
        gw = _make_gateway()
        gw._ws = AsyncMock()
        gw._sequence = 7
        audit_calls: list[tuple] = []
        gw._emit_audit = lambda et, r, d: audit_calls.append((et, r, d))  # type: ignore[method-assign]

        await gw._send_heartbeat()

        assert any(et == "discord.gateway.heartbeat_sent" for et, _, _ in audit_calls)
        hb_details = [d for et, r, d in audit_calls if et == "discord.gateway.heartbeat_sent"]
        assert hb_details[0]["seq"] == 7

    @pytest.mark.asyncio
    async def test_send_heartbeat_noop_when_ws_is_none(self):
        gw = _make_gateway()
        gw._ws = None
        # Should not raise
        await gw._send_heartbeat()

    @pytest.mark.asyncio
    async def test_send_heartbeat_tolerates_send_error(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        mock_ws.send.side_effect = OSError("broken pipe")
        gw._ws = mock_ws

        # Should not raise
        await gw._send_heartbeat()


class TestGatewayIdentifyOrResume:
    @pytest.mark.asyncio
    async def test_identifies_when_no_session(self):
        gw = _make_gateway()
        gw._discord_session_id = None
        gw._sequence = None

        identified = []
        resumed = []

        gw._send_identify = AsyncMock(side_effect=lambda: identified.append(1))  # type: ignore[method-assign]
        gw._send_resume = AsyncMock(side_effect=lambda: resumed.append(1))  # type: ignore[method-assign]

        await gw._identify_or_resume()

        assert identified == [1]
        assert resumed == []

    @pytest.mark.asyncio
    async def test_resumes_when_session_and_sequence_present(self):
        gw = _make_gateway()
        gw._discord_session_id = "existing-session"
        gw._sequence = 42

        identified = []
        resumed = []

        gw._send_identify = AsyncMock(side_effect=lambda: identified.append(1))  # type: ignore[method-assign]
        gw._send_resume = AsyncMock(side_effect=lambda: resumed.append(1))  # type: ignore[method-assign]

        await gw._identify_or_resume()

        assert identified == []
        assert resumed == [1]

    @pytest.mark.asyncio
    async def test_identifies_when_session_present_but_sequence_none(self):
        gw = _make_gateway()
        gw._discord_session_id = "some-session"
        gw._sequence = None

        gw._send_identify = AsyncMock()  # type: ignore[method-assign]
        gw._send_resume = AsyncMock()  # type: ignore[method-assign]

        await gw._identify_or_resume()

        gw._send_identify.assert_called_once()
        gw._send_resume.assert_not_called()


class TestGatewaySendIdentify:
    @pytest.mark.asyncio
    async def test_send_identify_sends_correct_payload(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        gw._ws = mock_ws

        await gw._send_identify()

        mock_ws.send.assert_called_once()
        payload = json.loads(mock_ws.send.call_args[0][0])
        assert payload["op"] == _OP_IDENTIFY
        assert payload["d"]["token"] == gw._token
        assert "intents" in payload["d"]
        assert payload["d"]["properties"]["os"] == "linux"
        assert payload["d"]["properties"]["browser"] == "missy"

    @pytest.mark.asyncio
    async def test_send_identify_uses_bot_token(self):
        gw = DiscordGatewayClient(bot_token="myrawtoken", on_message=AsyncMock())
        mock_ws = AsyncMock()
        gw._ws = mock_ws

        await gw._send_identify()

        payload = json.loads(mock_ws.send.call_args[0][0])
        assert payload["d"]["token"] == "Bot myrawtoken"


class TestGatewaySendResume:
    @pytest.mark.asyncio
    async def test_send_resume_sends_correct_payload(self):
        gw = _make_gateway()
        gw._discord_session_id = "sess-xyz"
        gw._sequence = 100
        mock_ws = AsyncMock()
        gw._ws = mock_ws

        await gw._send_resume()

        mock_ws.send.assert_called_once()
        payload = json.loads(mock_ws.send.call_args[0][0])
        assert payload["op"] == _OP_RESUME
        assert payload["d"]["token"] == gw._token
        assert payload["d"]["session_id"] == "sess-xyz"
        assert payload["d"]["seq"] == 100


class TestGatewayEmitAudit:
    def test_emit_audit_swallows_exceptions(self):
        gw = _make_gateway()

        with patch("missy.channels.discord.gateway.event_bus") as mock_bus:
            mock_bus.publish.side_effect = RuntimeError("bus exploded")
            # Should not raise
            gw._emit_audit("test.event", "allow", {"key": "val"})


# ===========================================================================
# REST tests
# ===========================================================================


class TestMaskMentions:
    def test_empty_string_returns_empty(self):
        assert _mask_mentions("") == ""

    def test_none_returns_empty(self):
        # The function is called with `s or ""`, so None input maps to ""
        assert _mask_mentions(None) == ""  # type: ignore[arg-type]

    def test_plain_text_unchanged(self):
        text = "Hello, world! No mentions here."
        assert _mask_mentions(text) == text


class TestRestHeaders:
    def test_authorization_header_uses_bot_token(self):
        client, _ = _make_rest()
        headers = client._headers()
        assert headers["Authorization"] == "Bot testtoken"

    def test_content_type_json_by_default(self):
        client, _ = _make_rest()
        headers = client._headers()
        assert headers["Content-Type"] == "application/json"

    def test_extra_headers_merged(self):
        client, _ = _make_rest()
        headers = client._headers(extra={"X-Custom": "value"})
        assert headers["X-Custom"] == "value"
        assert "Authorization" in headers

    def test_extra_headers_can_override_defaults(self):
        client, _ = _make_rest()
        headers = client._headers(extra={"Content-Type": "multipart/form-data"})
        assert headers["Content-Type"] == "multipart/form-data"


class TestRestSendMessageRetry:
    def test_retries_on_429_with_retry_after_header(self):
        client, mock_http = _make_rest()

        rate_limit_resp = MagicMock()
        rate_limit_resp.status_code = 429
        rate_limit_resp.headers = {"Retry-After": "0.01"}
        rate_limit_resp.raise_for_status.return_value = None

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {"id": "msg-after-retry"}
        success_resp.raise_for_status.return_value = None

        mock_http.post.side_effect = [rate_limit_resp, success_resp]

        with patch("time.sleep"):
            result = client.send_message("chan-1", "hello")

        assert result["id"] == "msg-after-retry"
        assert mock_http.post.call_count == 2

    def test_retries_on_429_without_retry_after_uses_backoff(self):
        client, mock_http = _make_rest()

        rate_limit_resp = MagicMock()
        rate_limit_resp.status_code = 429
        rate_limit_resp.headers = {}  # No Retry-After
        rate_limit_resp.raise_for_status.return_value = None

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {"id": "msg-ok"}
        success_resp.raise_for_status.return_value = None

        mock_http.post.side_effect = [rate_limit_resp, success_resp]

        with patch("time.sleep") as mock_sleep:
            result = client.send_message("chan-1", "hello")

        assert result["id"] == "msg-ok"
        mock_sleep.assert_called_once()
        sleep_duration = mock_sleep.call_args[0][0]
        assert sleep_duration >= 1.0  # At least the base backoff

    def test_retries_on_502(self):
        client, mock_http = _make_rest()

        bad_resp = MagicMock()
        bad_resp.status_code = 502
        bad_resp.headers = {}
        bad_resp.raise_for_status.return_value = None

        good_resp = MagicMock()
        good_resp.status_code = 200
        good_resp.json.return_value = {"id": "msg-ok"}
        good_resp.raise_for_status.return_value = None

        mock_http.post.side_effect = [bad_resp, good_resp]

        with patch("time.sleep"):
            result = client.send_message("chan-1", "hello")

        assert result["id"] == "msg-ok"

    def test_retries_on_503(self):
        client, mock_http = _make_rest()

        bad_resp = MagicMock()
        bad_resp.status_code = 503
        bad_resp.headers = {}
        bad_resp.raise_for_status.return_value = None

        good_resp = MagicMock()
        good_resp.status_code = 200
        good_resp.json.return_value = {"id": "msg-ok"}
        good_resp.raise_for_status.return_value = None

        mock_http.post.side_effect = [bad_resp, good_resp]

        with patch("time.sleep"):
            result = client.send_message("chan-1", "hello")

        assert result["id"] == "msg-ok"

    def test_retries_on_504(self):
        client, mock_http = _make_rest()

        bad_resp = MagicMock()
        bad_resp.status_code = 504
        bad_resp.headers = {}
        bad_resp.raise_for_status.return_value = None

        good_resp = MagicMock()
        good_resp.status_code = 200
        good_resp.json.return_value = {"id": "msg-ok"}
        good_resp.raise_for_status.return_value = None

        mock_http.post.side_effect = [bad_resp, good_resp]

        with patch("time.sleep"):
            result = client.send_message("chan-1", "hello")

        assert result["id"] == "msg-ok"

    def test_exhausted_retries_on_persistent_429_raises(self):
        client, mock_http = _make_rest()

        rate_limit_resp = MagicMock()
        rate_limit_resp.status_code = 429
        rate_limit_resp.headers = {}
        # raise_for_status raises on final attempt when backoff list is exhausted
        rate_limit_resp.raise_for_status.side_effect = Exception("429 Too Many Requests")

        mock_http.post.return_value = rate_limit_resp

        with patch("time.sleep"), pytest.raises(Exception, match="429"):
            client.send_message("chan-1", "hello")

    def test_exception_in_send_retries_then_reraises(self):
        client, mock_http = _make_rest()

        mock_http.post.side_effect = OSError("connection refused")

        with patch("time.sleep"), pytest.raises(OSError, match="connection refused"):
            client.send_message("chan-1", "hello")

        # Should have retried: attempt_count = len(backoffs)+1 = 4 total, but
        # exceptions exhaust backoffs array first (3 items) then re-raise
        assert mock_http.post.call_count == 4

    def test_missing_id_in_response_raises_runtime_error(self):
        client, mock_http = _make_rest()

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {}  # No "id" field
        resp.raise_for_status.return_value = None
        mock_http.post.return_value = resp

        with pytest.raises(RuntimeError, match="missing id"):
            client.send_message("chan-1", "hello")

    def test_send_message_with_mention_user_ids(self):
        client, mock_http = _make_rest()

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"id": "msg-with-mention"}
        resp.raise_for_status.return_value = None
        mock_http.post.return_value = resp

        result = client.send_message("chan-1", "hello <@user-42>", mention_user_ids=["user-42"])

        call_kwargs = mock_http.post.call_args[1]
        assert "user-42" in call_kwargs["json"]["allowed_mentions"]["users"]
        assert result["id"] == "msg-with-mention"

    def test_send_message_default_blocks_all_mentions(self):
        client, mock_http = _make_rest()

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"id": "msg-safe"}
        resp.raise_for_status.return_value = None
        mock_http.post.return_value = resp

        client.send_message("chan-1", "hello @everyone")

        call_kwargs = mock_http.post.call_args[1]
        # parse list should be empty to suppress mention parsing
        assert call_kwargs["json"]["allowed_mentions"]["parse"] == []
        assert call_kwargs["json"]["allowed_mentions"]["users"] == []


class TestRestUploadFile:
    def test_upload_file_raises_when_file_not_found(self):
        client, _ = _make_rest()

        with pytest.raises(FileNotFoundError):
            client.upload_file("chan-1", "/nonexistent/path/file.txt")

    def test_upload_file_success(self):
        client, _ = _make_rest()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "upload-msg-1"}
        mock_response.raise_for_status.return_value = None

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"hello file content")
            tmp_path = tmp.name

        try:
            with patch("httpx.post", return_value=mock_response) as mock_post:
                result = client.upload_file("chan-1", tmp_path, caption="My file")

            assert result["id"] == "upload-msg-1"
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert "file" in call_kwargs.get("files", {})
            assert call_kwargs.get("data", {}).get("content") == "My file"
        finally:
            os.unlink(tmp_path)

    def test_upload_file_without_caption(self):
        client, _ = _make_rest()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "upload-msg-2"}
        mock_response.raise_for_status.return_value = None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b"\x89PNG")
            tmp_path = tmp.name

        try:
            with patch("httpx.post", return_value=mock_response) as mock_post:
                client.upload_file("chan-1", tmp_path)

            call_kwargs = mock_post.call_args[1]
            # No caption means no data dict (or empty)
            assert call_kwargs.get("data") == {}
        finally:
            os.unlink(tmp_path)

    def test_upload_file_excludes_content_type_header(self):
        """Content-Type must be omitted so httpx sets multipart boundary correctly."""
        client, _ = _make_rest()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "m"}
        mock_response.raise_for_status.return_value = None

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"data")
            tmp_path = tmp.name

        try:
            with patch("httpx.post", return_value=mock_response) as mock_post:
                client.upload_file("chan-1", tmp_path)

            headers = mock_post.call_args[1]["headers"]
            assert "Content-Type" not in headers
        finally:
            os.unlink(tmp_path)


class TestRestAddReaction:
    def test_add_reaction_uses_put(self):
        client, _ = _make_rest()
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.raise_for_status.return_value = None

        with patch("httpx.put", return_value=mock_response) as mock_put:
            client.add_reaction("chan-1", "msg-1", "\u2705")

        mock_put.assert_called_once()

    def test_add_reaction_url_encodes_emoji(self):
        client, _ = _make_rest()
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.raise_for_status.return_value = None

        with patch("httpx.put", return_value=mock_response) as mock_put:
            client.add_reaction("chan-1", "msg-1", "\u2705")

        call_url = mock_put.call_args[0][0]
        # The checkmark emoji must be percent-encoded in the URL
        assert "%E2%9C%85" in call_url or "✅" not in call_url

    def test_add_reaction_correct_url_structure(self):
        client, _ = _make_rest()
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.raise_for_status.return_value = None

        with patch("httpx.put", return_value=mock_response) as mock_put:
            client.add_reaction("chan-99", "msg-77", "👍")

        call_url = mock_put.call_args[0][0]
        assert "chan-99" in call_url
        assert "msg-77" in call_url
        assert "reactions" in call_url
        assert "@me" in call_url

    def test_add_reaction_excludes_content_type_header(self):
        client, _ = _make_rest()
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.raise_for_status.return_value = None

        with patch("httpx.put", return_value=mock_response) as mock_put:
            client.add_reaction("chan-1", "msg-1", "\u274c")

        headers = mock_put.call_args[1]["headers"]
        assert "Content-Type" not in headers
        assert "Authorization" in headers

    def test_add_reaction_raises_on_http_error(self):
        client, _ = _make_rest()
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = Exception("Forbidden")

        with patch("httpx.put", return_value=mock_response), pytest.raises(Exception, match="Forbidden"):
            client.add_reaction("chan-1", "msg-1", "\u2705")


class TestRestTriggerTyping:
    def test_trigger_typing_posts_to_correct_url(self):
        client, mock_http = _make_rest()
        resp = MagicMock()
        resp.status_code = 204
        mock_http.post.return_value = resp

        client.trigger_typing("chan-42")

        call_url = mock_http.post.call_args[0][0]
        assert "chan-42/typing" in call_url

    def test_trigger_typing_swallows_exceptions(self):
        client, mock_http = _make_rest()
        mock_http.post.side_effect = OSError("connection refused")

        # Should not raise
        client.trigger_typing("chan-1")


class TestRestGetCurrentUser:
    def test_returns_user_object(self):
        client, mock_http = _make_rest()
        mock_http.get.return_value.json.return_value = {
            "id": "bot-123",
            "username": "MissyBot",
        }

        result = client.get_current_user()

        assert result["id"] == "bot-123"
        assert result["username"] == "MissyBot"

    def test_calls_correct_endpoint(self):
        client, mock_http = _make_rest()
        mock_http.get.return_value.json.return_value = {"id": "x"}

        client.get_current_user()

        call_url = mock_http.get.call_args[0][0]
        assert call_url == f"{BASE}/users/@me"


class TestRestGetGatewayBot:
    def test_returns_gateway_info(self):
        client, mock_http = _make_rest()
        mock_http.get.return_value.json.return_value = {
            "url": "wss://gateway.discord.gg",
            "shards": 1,
        }

        result = client.get_gateway_bot()

        assert "url" in result

    def test_calls_correct_endpoint(self):
        client, mock_http = _make_rest()
        mock_http.get.return_value.json.return_value = {"url": "wss://x"}

        client.get_gateway_bot()

        call_url = mock_http.get.call_args[0][0]
        assert call_url == f"{BASE}/gateway/bot"


class TestRestRegisterSlashCommands:
    def test_global_registration_url(self):
        client, mock_http = _make_rest()
        mock_http.put.return_value.json.return_value = []
        mock_http.put.return_value.raise_for_status.return_value = None

        client.register_slash_commands("app-123", [{"name": "ask"}])

        call_url = mock_http.put.call_args[0][0]
        assert "applications/app-123/commands" in call_url
        assert "guilds" not in call_url

    def test_guild_scoped_registration_url(self):
        client, mock_http = _make_rest()
        mock_http.put.return_value.json.return_value = []
        mock_http.put.return_value.raise_for_status.return_value = None

        client.register_slash_commands("app-123", [], guild_id="guild-456")

        call_url = mock_http.put.call_args[0][0]
        assert "guilds/guild-456/commands" in call_url

    def test_sends_commands_as_body(self):
        client, mock_http = _make_rest()
        mock_http.put.return_value.json.return_value = [{"name": "ask"}]
        mock_http.put.return_value.raise_for_status.return_value = None

        commands = [{"name": "ask", "description": "Ask Missy"}]
        result = client.register_slash_commands("app-1", commands)

        call_kwargs = mock_http.put.call_args[1]
        assert call_kwargs["json"] == commands
        assert result == [{"name": "ask"}]

    def test_uses_put_method(self):
        client, mock_http = _make_rest()
        mock_http.put.return_value.json.return_value = []
        mock_http.put.return_value.raise_for_status.return_value = None

        client.register_slash_commands("app-1", [])

        mock_http.put.assert_called_once()
        mock_http.post.assert_not_called()


class TestRestCreateThread:
    def test_standalone_thread_url(self):
        client, mock_http = _make_rest()
        mock_http.post.return_value.json.return_value = {"id": "thread-1"}

        client.create_thread("chan-1", "My Thread")

        call_url = mock_http.post.call_args[0][0]
        assert "/channels/chan-1/threads" in call_url

    def test_message_thread_url(self):
        client, mock_http = _make_rest()
        mock_http.post.return_value.json.return_value = {"id": "thread-2"}

        client.create_thread("chan-1", "My Thread", message_id="msg-99")

        call_url = mock_http.post.call_args[0][0]
        assert "/messages/msg-99/threads" in call_url

    def test_auto_archive_duration_forwarded(self):
        client, mock_http = _make_rest()
        mock_http.post.return_value.json.return_value = {"id": "t"}

        client.create_thread("chan-1", "T", auto_archive_duration=60)

        body = mock_http.post.call_args[1]["json"]
        assert body["auto_archive_duration"] == 60

    def test_standalone_thread_has_public_thread_type(self):
        client, mock_http = _make_rest()
        mock_http.post.return_value.json.return_value = {"id": "t"}

        client.create_thread("chan-1", "T")

        body = mock_http.post.call_args[1]["json"]
        assert body["type"] == 11  # PUBLIC_THREAD

    def test_message_thread_has_no_type_field(self):
        client, mock_http = _make_rest()
        mock_http.post.return_value.json.return_value = {"id": "t"}

        client.create_thread("chan-1", "T", message_id="msg-1")

        body = mock_http.post.call_args[1]["json"]
        assert "type" not in body
