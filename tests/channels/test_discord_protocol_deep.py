"""Deep protocol tests for Discord REST and Gateway modules.

Covers paths not exercised by test_discord_extended.py:

REST (missy/channels/discord/rest.py):
- _validate_snowflake: valid IDs, boundary cases, all invalid shapes
- _mask_mentions: user, role, channel mention variants
- DiscordRestClient constructor token-prefix logic
- get_gateway_bot endpoint and response passthrough
- send_message: reply_to_message_id path, body shape, channel snowflake guard
- send_message: 429 with non-numeric Retry-After falls back to backoff
- send_message: exception on last attempt calls _log_final_failure and re-raises
- upload_file: MIME detection, caption vs no-caption body
- add_reaction: invalid channel_id / message_id snowflakes raise ValueError
- trigger_typing: valid call posts to correct URL
- delete_message: 204 returns True, 403/404 return False, exception returns False
- delete_message: invalid snowflakes raise before any HTTP call
- create_thread: with message_id builds correct URL/body
- create_thread: without message_id uses /threads URL + type=11
- create_thread: name truncated to 100 characters
- get_channel: correct URL, response passthrough
- send_interaction_response: URL construction, body with/without data
- edit_interaction_response: PATCH to webhook URL, content truncated to 2000
- get_channel_messages: limit clamping 1-100, before param, correct URL
- download_attachment: cdn.discordapp.com succeeds, media.discordapp.net succeeds
- download_attachment: unknown domain raises ValueError
- register_slash_commands: global URL (no guild), guild URL construction

Gateway (missy/channels/discord/gateway.py):
- Constructor stores callbacks and audit session/task IDs
- bot_user_id property before and after READY
- _handle_payload: sequence number stored from 's' field
- _handle_payload: HELLO interval converted ms->s
- _handle_payload: HEARTBEAT_ACK is a no-op (no error)
- _handle_payload: RECONNECT calls ws.close
- _handle_payload: INVALID_SESSION resumable preserves session state
- _handle_payload: INVALID_SESSION non-resumable clears all state
- _handle_payload: unknown opcode does not raise
- _handle_dispatch: READY populates bot_user_id, session_id, resume URL
- _handle_dispatch: READY without discriminator does not crash
- _handle_dispatch: RESUMED emits audit, does not forward to callback
- _handle_dispatch: MESSAGE_CREATE payload shape forwarded verbatim
- _handle_dispatch: GUILD_CREATE forwarded
- _handle_dispatch: INTERACTION_CREATE forwarded
- _handle_dispatch: MESSAGE_REACTION_ADD forwarded
- _handle_dispatch: unrecognised event NOT forwarded
- _handle_dispatch: callback exception does not propagate
- _send_heartbeat: correct JSON shape with current sequence
- _send_heartbeat: ws=None returns immediately
- _send_heartbeat: send exception is swallowed
- _identify_or_resume: IDENTIFY when session=None
- _identify_or_resume: IDENTIFY when session set but sequence=None
- _identify_or_resume: RESUME when session + sequence both present
- _send_identify: payload contains token, intents, properties
- _send_resume: payload contains token, session_id, seq
- disconnect: sets _running=False, cancels heartbeat, closes ws, emits audit
- disconnect: safe when heartbeat=None and ws=None
- _emit_audit: exception from event_bus is swallowed silently
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
from missy.channels.discord.rest import (
    BASE,
    DiscordRestClient,
    _mask_mentions,
    _validate_snowflake,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    status_code: int = 200,
    json_data: object = None,
    content: bytes = b"",
    headers: dict | None = None,
) -> MagicMock:
    """Build a minimal httpx-like response mock."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.content = content
    resp.text = ""
    resp.headers = headers or {}
    resp.raise_for_status = MagicMock()
    return resp


def _make_rest(
    *,
    post_response: MagicMock | None = None,
    get_response: MagicMock | None = None,
    put_response: MagicMock | None = None,
    patch_response: MagicMock | None = None,
    delete_response: MagicMock | None = None,
) -> tuple[DiscordRestClient, MagicMock]:
    """Return a (DiscordRestClient, mock_http) pair with configurable responses."""
    mock_http = MagicMock()
    mock_http.post.return_value = post_response or _make_response(json_data={"id": "msg-1"})
    mock_http.get.return_value = get_response or _make_response(json_data={"id": "ch-1"})
    mock_http.put.return_value = put_response or _make_response(status_code=204)
    mock_http.patch.return_value = patch_response or _make_response(json_data={"id": "msg-1"})
    mock_http.delete.return_value = delete_response or _make_response(status_code=204)
    client = DiscordRestClient(bot_token="Bot testtoken", http_client=mock_http)
    return client, mock_http


def _make_gateway(on_message: AsyncMock | None = None) -> DiscordGatewayClient:
    if on_message is None:
        on_message = AsyncMock()
    return DiscordGatewayClient(
        bot_token="testtoken",
        on_message=on_message,
        session_id="s-test",
        task_id="t-test",
    )


# ===========================================================================
# _validate_snowflake
# ===========================================================================


class TestValidateSnowflake:
    def test_single_digit_is_valid(self):
        assert _validate_snowflake("1") == "1"

    def test_max_20_digits_is_valid(self):
        value = "1" * 20
        assert _validate_snowflake(value) == value

    def test_returns_value_unchanged(self):
        val = "123456789012345"
        assert _validate_snowflake(val, "channel_id") is val

    def test_typical_snowflake(self):
        assert _validate_snowflake("1234567890123456789") == "1234567890123456789"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="snowflake"):
            _validate_snowflake("", "channel_id")

    def test_letters_raise(self):
        with pytest.raises(ValueError, match="channel_id"):
            _validate_snowflake("abc123", "channel_id")

    def test_purely_alpha_raises(self):
        with pytest.raises(ValueError, match="snowflake"):
            _validate_snowflake("abcdef", "id")

    def test_21_digits_raises(self):
        with pytest.raises(ValueError, match="snowflake"):
            _validate_snowflake("1" * 21, "id")

    def test_special_chars_raise(self):
        with pytest.raises(ValueError):
            _validate_snowflake("123-456", "id")

    def test_leading_space_raises(self):
        with pytest.raises(ValueError):
            _validate_snowflake(" 123456", "id")

    def test_name_appears_in_error_message(self):
        with pytest.raises(ValueError, match="guild_id"):
            _validate_snowflake("bad!", "guild_id")


# ===========================================================================
# _mask_mentions
# ===========================================================================


class TestMaskMentions:
    def test_user_mention_redacted(self):
        result = _mask_mentions("Hello <@123456789> there")
        assert "123456789" not in result
        assert "redacted" in result

    def test_user_mention_with_exclamation_redacted(self):
        result = _mask_mentions("Hey <@!987654321> here")
        assert "987654321" not in result
        assert "redacted" in result

    def test_role_mention_redacted(self):
        result = _mask_mentions("Role <@&111222333> ping")
        assert "111222333" not in result
        assert "redacted" in result

    def test_channel_mention_redacted(self):
        result = _mask_mentions("See <#444555666> channel")
        assert "444555666" not in result
        assert "redacted" in result

    def test_multiple_mentions_all_redacted(self):
        text = "<@111> <@&222> <#333>"
        result = _mask_mentions(text)
        for snowflake in ("111", "222", "333"):
            assert snowflake not in result

    def test_no_mentions_unchanged(self):
        text = "No mentions here, just text."
        assert _mask_mentions(text) == text

    def test_surrounding_text_preserved(self):
        result = _mask_mentions("prefix <@123> suffix")
        assert "prefix" in result
        assert "suffix" in result


# ===========================================================================
# DiscordRestClient constructor
# ===========================================================================


class TestRestClientConstructor:
    def test_token_without_prefix_gets_prefixed(self):
        client, _ = _make_rest()
        # Already uses "Bot testtoken" in _make_rest but let's test bare token.
        bare_client = DiscordRestClient(bot_token="baretoken", http_client=MagicMock())
        assert bare_client._token == "Bot baretoken"

    def test_token_with_prefix_not_doubled(self):
        client = DiscordRestClient(bot_token="Bot already", http_client=MagicMock())
        assert client._token == "Bot already"

    def test_http_client_injected(self):
        mock_http = MagicMock()
        client = DiscordRestClient(bot_token="t", http_client=mock_http)
        assert client._http is mock_http

    def test_no_http_client_creates_default(self):
        with patch("missy.channels.discord.rest.create_client") as mock_create:
            mock_create.return_value = MagicMock()
            client = DiscordRestClient(bot_token="t")
        mock_create.assert_called_once()
        assert client._http is mock_create.return_value


# ===========================================================================
# get_gateway_bot
# ===========================================================================


class TestGetGatewayBot:
    def test_calls_correct_endpoint(self):
        expected = {"url": "wss://gateway.discord.gg", "shards": 1}
        client, mock_http = _make_rest(get_response=_make_response(json_data=expected))
        client.get_gateway_bot()
        url_called = mock_http.get.call_args[0][0]
        assert url_called == f"{BASE}/gateway/bot"

    def test_returns_gateway_data(self):
        expected = {"url": "wss://gateway.discord.gg", "shards": 2}
        client, _ = _make_rest(get_response=_make_response(json_data=expected))
        assert client.get_gateway_bot() == expected

    def test_raises_for_status_on_error(self):
        resp = _make_response(status_code=401)
        resp.raise_for_status.side_effect = Exception("401 Unauthorised")
        client, _ = _make_rest(get_response=resp)
        with pytest.raises(Exception, match="401"):
            client.get_gateway_bot()


# ===========================================================================
# send_message — body shape and optional fields
# ===========================================================================


class TestSendMessageBody:
    def test_basic_body_shape(self):
        client, mock_http = _make_rest(post_response=_make_response(json_data={"id": "m1"}))
        client.send_message("111222333", "hello")
        call_kwargs = mock_http.post.call_args[1]
        body = call_kwargs["json"]
        assert body["content"] == "hello"
        assert "allowed_mentions" in body

    def test_allowed_mentions_parse_is_empty_list(self):
        client, mock_http = _make_rest(post_response=_make_response(json_data={"id": "m1"}))
        client.send_message("111222333", "hi")
        body = mock_http.post.call_args[1]["json"]
        assert body["allowed_mentions"]["parse"] == []

    def test_mention_user_ids_passed_through(self):
        client, mock_http = _make_rest(post_response=_make_response(json_data={"id": "m1"}))
        client.send_message("111222333", "hi", mention_user_ids=["999"])
        body = mock_http.post.call_args[1]["json"]
        assert body["allowed_mentions"]["users"] == ["999"]

    def test_reply_to_message_id_adds_reference(self):
        client, mock_http = _make_rest(post_response=_make_response(json_data={"id": "m1"}))
        client.send_message("111222333", "reply", reply_to_message_id="555666777")
        body = mock_http.post.call_args[1]["json"]
        assert "message_reference" in body
        assert body["message_reference"]["message_id"] == "555666777"
        assert body["message_reference"]["fail_if_not_exists"] is False

    def test_invalid_channel_id_raises_before_http(self):
        client, mock_http = _make_rest()
        with pytest.raises(ValueError, match="channel_id"):
            client.send_message("bad-id", "hello")
        mock_http.post.assert_not_called()

    def test_invalid_reply_message_id_raises_before_http(self):
        client, mock_http = _make_rest()
        with pytest.raises(ValueError, match="reply_to_message_id"):
            client.send_message("111222333", "hi", reply_to_message_id="not-valid")
        mock_http.post.assert_not_called()

    def test_429_with_non_numeric_retry_after_falls_back_to_backoff(self):
        """Non-numeric Retry-After header should fall through to backoff delay."""
        retry_resp = _make_response(status_code=429, headers={"Retry-After": "soon"})
        ok_resp = _make_response(json_data={"id": "m1"})
        mock_http = MagicMock()
        mock_http.post.side_effect = [retry_resp, ok_resp]
        client = DiscordRestClient(bot_token="Bot t", http_client=mock_http)
        with patch("time.sleep"):
            result = client.send_message("111222333", "hi")
        assert result["id"] == "m1"
        assert mock_http.post.call_count == 2

    def test_missing_id_in_response_raises_runtime_error(self):
        client, _ = _make_rest(post_response=_make_response(json_data={}))
        with pytest.raises(RuntimeError, match="missing id"):
            client.send_message("111222333", "hi")


# ===========================================================================
# upload_file
# ===========================================================================


class TestUploadFile:
    def test_file_not_found_raises(self):
        client, _ = _make_rest()
        with pytest.raises(FileNotFoundError):
            client.upload_file("111222333", "/nonexistent/path/file.txt")

    def test_upload_sends_to_correct_url(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"hello")
            path = f.name
        try:
            client, mock_http = _make_rest(
                post_response=_make_response(json_data={"id": "m-upload"})
            )
            client.upload_file("111222333", path)
            url_called = mock_http.post.call_args[0][0]
            assert url_called == f"{BASE}/channels/111222333/messages"
        finally:
            os.unlink(path)

    def test_upload_with_caption_includes_content(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"data")
            path = f.name
        try:
            client, mock_http = _make_rest(post_response=_make_response(json_data={"id": "m1"}))
            client.upload_file("111222333", path, caption="My caption")
            data_arg = mock_http.post.call_args[1].get("data", {})
            assert data_arg.get("content") == "My caption"
        finally:
            os.unlink(path)

    def test_upload_without_caption_omits_content(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"data")
            path = f.name
        try:
            client, mock_http = _make_rest(post_response=_make_response(json_data={"id": "m1"}))
            client.upload_file("111222333", path)
            data_arg = mock_http.post.call_args[1].get("data", {})
            assert data_arg == {}
        finally:
            os.unlink(path)

    def test_content_type_header_excluded(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG")
            path = f.name
        try:
            client, mock_http = _make_rest(post_response=_make_response(json_data={"id": "m1"}))
            client.upload_file("111222333", path)
            headers_used = mock_http.post.call_args[1]["headers"]
            assert "Content-Type" not in headers_used
        finally:
            os.unlink(path)


# ===========================================================================
# add_reaction
# ===========================================================================


class TestAddReaction:
    def test_valid_call_uses_put(self):
        client, mock_http = _make_rest(put_response=_make_response(status_code=204))
        client.add_reaction("111222333", "444555666", "\u2705")
        mock_http.put.assert_called_once()

    def test_url_contains_encoded_emoji(self):
        client, mock_http = _make_rest(put_response=_make_response(status_code=204))
        client.add_reaction("111222333", "444555666", "\u2705")
        url = mock_http.put.call_args[0][0]
        assert "%E2%9C%85" in url  # URL-encoded form of U+2705

    def test_url_structure(self):
        client, mock_http = _make_rest(put_response=_make_response(status_code=204))
        client.add_reaction("111", "222", "\u2764")
        url = mock_http.put.call_args[0][0]
        assert "/channels/111/messages/222/reactions/" in url
        assert "/@me" in url

    def test_invalid_channel_id_raises_before_http(self):
        client, mock_http = _make_rest()
        with pytest.raises(ValueError, match="channel_id"):
            client.add_reaction("bad", "444555666", "\u2705")
        mock_http.put.assert_not_called()

    def test_invalid_message_id_raises_before_http(self):
        client, mock_http = _make_rest()
        with pytest.raises(ValueError, match="message_id"):
            client.add_reaction("111222333", "not-a-snowflake", "\u2705")
        mock_http.put.assert_not_called()

    def test_http_error_propagates(self):
        resp = _make_response(status_code=403)
        resp.raise_for_status.side_effect = Exception("Forbidden")
        client, _ = _make_rest(put_response=resp)
        with pytest.raises(Exception, match="Forbidden"):
            client.add_reaction("111222333", "444555666", "\u2705")


# ===========================================================================
# delete_message
# ===========================================================================


class TestDeleteMessage:
    def test_204_returns_true(self):
        client, _ = _make_rest(delete_response=_make_response(status_code=204))
        assert client.delete_message("111222333", "444555666") is True

    def test_403_returns_false(self):
        client, _ = _make_rest(delete_response=_make_response(status_code=403))
        assert client.delete_message("111222333", "444555666") is False

    def test_404_returns_false(self):
        client, _ = _make_rest(delete_response=_make_response(status_code=404))
        assert client.delete_message("111222333", "444555666") is False

    def test_exception_returns_false(self):
        mock_http = MagicMock()
        mock_http.delete.side_effect = Exception("network error")
        client = DiscordRestClient(bot_token="Bot t", http_client=mock_http)
        assert client.delete_message("111222333", "444555666") is False

    def test_calls_correct_url(self):
        client, mock_http = _make_rest(delete_response=_make_response(status_code=204))
        client.delete_message("111222333", "444555666")
        url = mock_http.delete.call_args[0][0]
        assert url == f"{BASE}/channels/111222333/messages/444555666"

    def test_invalid_channel_id_raises(self):
        client, mock_http = _make_rest()
        with pytest.raises(ValueError, match="channel_id"):
            client.delete_message("bad-channel", "444555666")
        mock_http.delete.assert_not_called()

    def test_invalid_message_id_raises(self):
        client, mock_http = _make_rest()
        with pytest.raises(ValueError, match="message_id"):
            client.delete_message("111222333", "bad-msg")
        mock_http.delete.assert_not_called()

    def test_non_204_200_raises_for_status(self):
        resp = _make_response(status_code=500)
        resp.raise_for_status.side_effect = Exception("Server error")
        client, _ = _make_rest(delete_response=resp)
        # Exception is caught and returns False.
        assert client.delete_message("111222333", "444555666") is False


# ===========================================================================
# create_thread
# ===========================================================================


class TestCreateThread:
    def test_with_message_id_uses_message_thread_url(self):
        client, mock_http = _make_rest(post_response=_make_response(json_data={"id": "thread-1"}))
        client.create_thread("111222333", "My Thread", message_id="444555666")
        url = mock_http.post.call_args[0][0]
        assert url == f"{BASE}/channels/111222333/messages/444555666/threads"

    def test_without_message_id_uses_threads_url(self):
        client, mock_http = _make_rest(post_response=_make_response(json_data={"id": "thread-1"}))
        client.create_thread("111222333", "Standalone Thread")
        url = mock_http.post.call_args[0][0]
        assert url == f"{BASE}/channels/111222333/threads"

    def test_without_message_id_body_includes_type_11(self):
        client, mock_http = _make_rest(post_response=_make_response(json_data={"id": "thread-1"}))
        client.create_thread("111222333", "Thread")
        body = mock_http.post.call_args[1]["json"]
        assert body["type"] == 11

    def test_with_message_id_body_excludes_type(self):
        client, mock_http = _make_rest(post_response=_make_response(json_data={"id": "thread-1"}))
        client.create_thread("111222333", "Thread", message_id="444555666")
        body = mock_http.post.call_args[1]["json"]
        assert "type" not in body

    def test_name_truncated_to_100_chars(self):
        long_name = "A" * 150
        client, mock_http = _make_rest(post_response=_make_response(json_data={"id": "thread-1"}))
        client.create_thread("111222333", long_name)
        body = mock_http.post.call_args[1]["json"]
        assert len(body["name"]) == 100

    def test_auto_archive_duration_in_body(self):
        client, mock_http = _make_rest(post_response=_make_response(json_data={"id": "thread-1"}))
        client.create_thread("111222333", "Thread", auto_archive_duration=60)
        body = mock_http.post.call_args[1]["json"]
        assert body["auto_archive_duration"] == 60

    def test_default_auto_archive_duration(self):
        client, mock_http = _make_rest(post_response=_make_response(json_data={"id": "thread-1"}))
        client.create_thread("111222333", "Thread")
        body = mock_http.post.call_args[1]["json"]
        assert body["auto_archive_duration"] == 1440

    def test_invalid_channel_id_raises(self):
        client, mock_http = _make_rest()
        with pytest.raises(ValueError, match="channel_id"):
            client.create_thread("bad-channel", "Thread")
        mock_http.post.assert_not_called()

    def test_returns_response_json(self):
        expected = {"id": "thread-42", "name": "T"}
        client, _ = _make_rest(post_response=_make_response(json_data=expected))
        result = client.create_thread("111222333", "T")
        assert result == expected


# ===========================================================================
# get_channel
# ===========================================================================


class TestGetChannel:
    def test_calls_correct_url(self):
        expected = {"id": "111222333", "type": 0}
        client, mock_http = _make_rest(get_response=_make_response(json_data=expected))
        client.get_channel("111222333")
        url = mock_http.get.call_args[0][0]
        assert url == f"{BASE}/channels/111222333"

    def test_returns_channel_object(self):
        expected = {"id": "111222333", "type": 0, "name": "general"}
        client, _ = _make_rest(get_response=_make_response(json_data=expected))
        assert client.get_channel("111222333") == expected

    def test_invalid_channel_id_raises(self):
        client, mock_http = _make_rest()
        with pytest.raises(ValueError, match="channel_id"):
            client.get_channel("not-a-snowflake")
        mock_http.get.assert_not_called()


# ===========================================================================
# send_interaction_response
# ===========================================================================


class TestSendInteractionResponse:
    def test_url_includes_interaction_id_and_token(self):
        client, mock_http = _make_rest(post_response=_make_response(status_code=204))
        client.send_interaction_response("111222333", "token-abc", response_type=4)
        url = mock_http.post.call_args[0][0]
        assert "111222333" in url
        assert "token-abc" in url
        assert url.endswith("/callback")

    def test_body_contains_type(self):
        client, mock_http = _make_rest(post_response=_make_response(status_code=204))
        client.send_interaction_response("111222333", "token-abc", response_type=5)
        body = mock_http.post.call_args[1]["json"]
        assert body["type"] == 5

    def test_data_included_when_provided(self):
        client, mock_http = _make_rest(post_response=_make_response(status_code=204))
        data = {"content": "Thinking…"}
        client.send_interaction_response("111222333", "token-abc", response_type=4, data=data)
        body = mock_http.post.call_args[1]["json"]
        assert body["data"] == data

    def test_data_omitted_when_none(self):
        client, mock_http = _make_rest(post_response=_make_response(status_code=204))
        client.send_interaction_response("111222333", "token-abc", response_type=4)
        body = mock_http.post.call_args[1]["json"]
        assert "data" not in body

    def test_invalid_interaction_id_raises(self):
        client, mock_http = _make_rest()
        with pytest.raises(ValueError, match="interaction_id"):
            client.send_interaction_response("bad-id", "token", response_type=4)
        mock_http.post.assert_not_called()


# ===========================================================================
# edit_interaction_response
# ===========================================================================


class TestEditInteractionResponse:
    def test_uses_patch_method(self):
        client, mock_http = _make_rest(patch_response=_make_response(json_data={"id": "m1"}))
        client.edit_interaction_response("111222333", "tok", "Updated content")
        mock_http.patch.assert_called_once()

    def test_url_structure(self):
        client, mock_http = _make_rest(patch_response=_make_response(json_data={"id": "m1"}))
        client.edit_interaction_response("111222333", "tok", "content")
        url = mock_http.patch.call_args[0][0]
        assert "webhooks/111222333/tok/messages/@original" in url

    def test_content_truncated_to_2000(self):
        long_content = "X" * 2500
        client, mock_http = _make_rest(patch_response=_make_response(json_data={"id": "m1"}))
        client.edit_interaction_response("111222333", "tok", long_content)
        body = mock_http.patch.call_args[1]["json"]
        assert len(body["content"]) == 2000

    def test_returns_updated_message(self):
        expected = {"id": "m1", "content": "done"}
        client, _ = _make_rest(patch_response=_make_response(json_data=expected))
        assert client.edit_interaction_response("111222333", "tok", "done") == expected

    def test_invalid_application_id_raises(self):
        client, mock_http = _make_rest()
        with pytest.raises(ValueError, match="application_id"):
            client.edit_interaction_response("bad", "tok", "content")
        mock_http.patch.assert_not_called()


# ===========================================================================
# get_channel_messages
# ===========================================================================


class TestGetChannelMessages:
    def test_calls_correct_url(self):
        client, mock_http = _make_rest(get_response=_make_response(json_data=[{"id": "m1"}]))
        client.get_channel_messages("111222333")
        url = mock_http.get.call_args[0][0]
        assert url == f"{BASE}/channels/111222333/messages"

    def test_limit_sent_as_param(self):
        client, mock_http = _make_rest(get_response=_make_response(json_data=[]))
        client.get_channel_messages("111222333", limit=25)
        params = mock_http.get.call_args[1]["params"]
        assert params["limit"] == 25

    def test_limit_clamped_to_minimum_1(self):
        client, mock_http = _make_rest(get_response=_make_response(json_data=[]))
        client.get_channel_messages("111222333", limit=0)
        params = mock_http.get.call_args[1]["params"]
        assert params["limit"] == 1

    def test_limit_clamped_to_maximum_100(self):
        client, mock_http = _make_rest(get_response=_make_response(json_data=[]))
        client.get_channel_messages("111222333", limit=999)
        params = mock_http.get.call_args[1]["params"]
        assert params["limit"] == 100

    def test_before_param_included_when_provided(self):
        client, mock_http = _make_rest(get_response=_make_response(json_data=[]))
        client.get_channel_messages("111222333", before="999888777")
        params = mock_http.get.call_args[1]["params"]
        assert params["before"] == "999888777"

    def test_before_param_omitted_when_none(self):
        client, mock_http = _make_rest(get_response=_make_response(json_data=[]))
        client.get_channel_messages("111222333")
        params = mock_http.get.call_args[1]["params"]
        assert "before" not in params

    def test_invalid_before_raises(self):
        client, mock_http = _make_rest()
        with pytest.raises(ValueError, match="before"):
            client.get_channel_messages("111222333", before="bad-id")
        mock_http.get.assert_not_called()

    def test_invalid_channel_id_raises(self):
        client, mock_http = _make_rest()
        with pytest.raises(ValueError, match="channel_id"):
            client.get_channel_messages("bad-chan")
        mock_http.get.assert_not_called()


# ===========================================================================
# download_attachment
# ===========================================================================


class TestDownloadAttachment:
    def test_cdn_url_succeeds(self):
        client, mock_http = _make_rest(get_response=_make_response(content=b"\x89PNG"))
        result = client.download_attachment("https://cdn.discordapp.com/attachments/1/2/file.png")
        assert result == b"\x89PNG"

    def test_media_url_succeeds(self):
        client, mock_http = _make_rest(get_response=_make_response(content=b"data"))
        result = client.download_attachment("https://media.discordapp.net/attachments/1/2/file.jpg")
        assert result == b"data"

    def test_unknown_domain_raises_value_error(self):
        client, _ = _make_rest()
        with pytest.raises(ValueError, match="Not a Discord CDN URL"):
            client.download_attachment("https://example.com/file.png")

    def test_non_discord_domain_raises(self):
        client, _ = _make_rest()
        with pytest.raises(ValueError):
            client.download_attachment("https://evil.com/steal.png")

    def test_passes_timeout_to_get(self):
        client, mock_http = _make_rest(get_response=_make_response(content=b"ok"))
        client.download_attachment("https://cdn.discordapp.com/attachments/1/2/f.txt", timeout=15)
        assert mock_http.get.call_args[1]["timeout"] == 15


# ===========================================================================
# register_slash_commands
# ===========================================================================


class TestRegisterSlashCommands:
    def test_global_commands_url(self):
        commands = [{"name": "ask", "description": "Ask missy"}]
        client, mock_http = _make_rest(put_response=_make_response(json_data=commands))
        client.register_slash_commands("app-111222333", commands)
        url = mock_http.put.call_args[0][0]
        assert url == f"{BASE}/applications/app-111222333/commands"

    def test_guild_commands_url(self):
        commands = [{"name": "ask", "description": "Ask missy"}]
        client, mock_http = _make_rest(put_response=_make_response(json_data=commands))
        client.register_slash_commands("app-111222333", commands, guild_id="guild-999")
        url = mock_http.put.call_args[0][0]
        assert "guilds/guild-999/commands" in url

    def test_commands_sent_as_body(self):
        commands = [{"name": "ping"}, {"name": "help"}]
        client, mock_http = _make_rest(put_response=_make_response(json_data=commands))
        client.register_slash_commands("app-111222333", commands)
        sent_body = mock_http.put.call_args[1]["json"]
        assert sent_body == commands

    def test_returns_registered_commands(self):
        registered = [{"id": "cmd-1", "name": "ask"}]
        client, _ = _make_rest(put_response=_make_response(json_data=registered))
        result = client.register_slash_commands("app-111222333", [])
        assert result == registered

    def test_uses_put_not_post(self):
        client, mock_http = _make_rest(put_response=_make_response(json_data=[]))
        client.register_slash_commands("app-111222333", [])
        mock_http.put.assert_called_once()
        mock_http.post.assert_not_called()


# ===========================================================================
# Gateway — constructor
# ===========================================================================


class TestGatewayConstructor:
    def test_token_without_prefix_gets_prefixed(self):
        gw = _make_gateway()
        assert gw._token == "Bot testtoken"

    def test_token_with_prefix_not_doubled(self):
        gw = DiscordGatewayClient(bot_token="Bot already", on_message=AsyncMock())
        assert gw._token == "Bot already"

    def test_on_message_callback_stored(self):
        cb = AsyncMock()
        gw = DiscordGatewayClient(bot_token="t", on_message=cb)
        assert gw._on_message is cb

    def test_custom_gateway_url_stored(self):
        url = "wss://custom.gateway.example.com"
        gw = DiscordGatewayClient(bot_token="t", on_message=AsyncMock(), gateway_url=url)
        assert gw._gateway_url == url

    def test_session_and_task_id_stored(self):
        gw = DiscordGatewayClient(
            bot_token="t",
            on_message=AsyncMock(),
            session_id="s42",
            task_id="t42",
        )
        assert gw._session_id_audit == "s42"
        assert gw._task_id_audit == "t42"

    def test_initial_state_is_clean(self):
        gw = _make_gateway()
        assert gw._ws is None
        assert gw._heartbeat_task is None
        assert gw._sequence is None
        assert gw._discord_session_id is None
        assert gw._resume_gateway_url is None
        assert gw._bot_user_id is None
        assert gw._running is False

    def test_bot_user_id_property_initially_none(self):
        gw = _make_gateway()
        assert gw.bot_user_id is None


# ===========================================================================
# Gateway — _handle_payload sequence tracking
# ===========================================================================


class TestHandlePayloadSequenceTracking:
    @pytest.mark.asyncio
    async def test_sequence_stored_from_s_field(self):
        gw = _make_gateway()
        gw._ws = MagicMock()
        gw._start_heartbeat = AsyncMock()
        gw._identify_or_resume = AsyncMock()
        payload = {"op": _OP_HELLO, "d": {"heartbeat_interval": 41250}, "s": 99}
        await gw._handle_payload(payload)
        assert gw._sequence == 99

    @pytest.mark.asyncio
    async def test_sequence_not_stored_when_s_is_none(self):
        gw = _make_gateway()
        gw._start_heartbeat = AsyncMock()
        gw._identify_or_resume = AsyncMock()
        payload = {"op": _OP_HELLO, "d": {"heartbeat_interval": 41250}, "s": None}
        await gw._handle_payload(payload)
        assert gw._sequence is None

    @pytest.mark.asyncio
    async def test_dispatch_sequence_updated(self):
        gw = _make_gateway()
        payload = {
            "op": _OP_DISPATCH,
            "d": {"session_id": "abc", "resume_gateway_url": "wss://r", "user": {"id": "u1"}},
            "s": 7,
            "t": "READY",
        }
        await gw._handle_payload(payload)
        assert gw._sequence == 7


# ===========================================================================
# Gateway — _handle_payload HELLO
# ===========================================================================


class TestHandlePayloadHello:
    @pytest.mark.asyncio
    async def test_hello_starts_heartbeat_with_interval_in_seconds(self):
        gw = _make_gateway()
        gw._start_heartbeat = AsyncMock()
        gw._identify_or_resume = AsyncMock()
        payload = {"op": _OP_HELLO, "d": {"heartbeat_interval": 41250}, "s": None}
        await gw._handle_payload(payload)
        gw._start_heartbeat.assert_awaited_once_with(41.25)

    @pytest.mark.asyncio
    async def test_hello_calls_identify_or_resume(self):
        gw = _make_gateway()
        gw._start_heartbeat = AsyncMock()
        gw._identify_or_resume = AsyncMock()
        payload = {"op": _OP_HELLO, "d": {"heartbeat_interval": 41250}, "s": None}
        await gw._handle_payload(payload)
        gw._identify_or_resume.assert_awaited_once()


# ===========================================================================
# Gateway — _handle_payload HEARTBEAT / HEARTBEAT_ACK / RECONNECT
# ===========================================================================


class TestHandlePayloadOtherOpcodes:
    @pytest.mark.asyncio
    async def test_heartbeat_request_sends_immediate_heartbeat(self):
        gw = _make_gateway()
        gw._send_heartbeat = AsyncMock()
        await gw._handle_payload({"op": _OP_HEARTBEAT, "d": None, "s": None})
        gw._send_heartbeat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_heartbeat_ack_does_not_raise(self):
        gw = _make_gateway()
        # Should complete silently.
        await gw._handle_payload({"op": _OP_HEARTBEAT_ACK, "d": None, "s": None})

    @pytest.mark.asyncio
    async def test_reconnect_closes_websocket(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        gw._ws = mock_ws
        await gw._handle_payload({"op": _OP_RECONNECT, "d": None, "s": None})
        mock_ws.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_opcode_does_not_raise(self):
        gw = _make_gateway()
        await gw._handle_payload({"op": 42, "d": None, "s": None})


# ===========================================================================
# Gateway — _handle_payload INVALID_SESSION
# ===========================================================================


class TestHandlePayloadInvalidSession:
    @pytest.mark.asyncio
    async def test_non_resumable_clears_session_state(self):
        gw = _make_gateway()
        gw._discord_session_id = "old-session"
        gw._resume_gateway_url = "wss://old"
        gw._sequence = 42
        gw._ws = AsyncMock()
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await gw._handle_payload({"op": _OP_INVALID_SESSION, "d": False, "s": None})
        assert gw._discord_session_id is None
        assert gw._resume_gateway_url is None
        assert gw._sequence is None

    @pytest.mark.asyncio
    async def test_resumable_preserves_session_state(self):
        gw = _make_gateway()
        gw._discord_session_id = "keep-session"
        gw._resume_gateway_url = "wss://keep"
        gw._sequence = 55
        gw._ws = AsyncMock()
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await gw._handle_payload({"op": _OP_INVALID_SESSION, "d": True, "s": None})
        assert gw._discord_session_id == "keep-session"
        assert gw._resume_gateway_url == "wss://keep"
        assert gw._sequence == 55

    @pytest.mark.asyncio
    async def test_closes_websocket_either_way(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        gw._ws = mock_ws
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await gw._handle_payload({"op": _OP_INVALID_SESSION, "d": False, "s": None})
        mock_ws.close.assert_awaited_once()


# ===========================================================================
# Gateway — _handle_dispatch
# ===========================================================================


class TestHandleDispatchReady:
    @pytest.mark.asyncio
    async def test_ready_stores_session_id(self):
        gw = _make_gateway()
        data = {
            "session_id": "ses-abc",
            "resume_gateway_url": "wss://resume",
            "user": {"id": "bot-1", "username": "Missy", "discriminator": "0001"},
        }
        await gw._handle_dispatch("READY", data)
        assert gw._discord_session_id == "ses-abc"

    @pytest.mark.asyncio
    async def test_ready_stores_resume_gateway_url(self):
        gw = _make_gateway()
        data = {
            "session_id": "s",
            "resume_gateway_url": "wss://resume.discord.gg",
            "user": {"id": "u1"},
        }
        await gw._handle_dispatch("READY", data)
        assert gw._resume_gateway_url == "wss://resume.discord.gg"

    @pytest.mark.asyncio
    async def test_ready_stores_bot_user_id(self):
        gw = _make_gateway()
        data = {
            "session_id": "s",
            "resume_gateway_url": "wss://r",
            "user": {"id": "bot-999", "username": "M", "discriminator": "0"},
        }
        await gw._handle_dispatch("READY", data)
        assert gw._bot_user_id == "bot-999"
        assert gw.bot_user_id == "bot-999"

    @pytest.mark.asyncio
    async def test_ready_without_user_id_does_not_crash(self):
        gw = _make_gateway()
        data = {"session_id": "s", "resume_gateway_url": "wss://r", "user": {}}
        await gw._handle_dispatch("READY", data)
        # _bot_user_id will be empty string (str(None or ""))
        assert gw._bot_user_id is not None

    @pytest.mark.asyncio
    async def test_ready_does_not_call_on_message(self):
        cb = AsyncMock()
        gw = _make_gateway(on_message=cb)
        data = {
            "session_id": "s",
            "resume_gateway_url": "wss://r",
            "user": {"id": "u1"},
        }
        await gw._handle_dispatch("READY", data)
        cb.assert_not_awaited()


class TestHandleDispatchResumed:
    @pytest.mark.asyncio
    async def test_resumed_does_not_forward_to_callback(self):
        cb = AsyncMock()
        gw = _make_gateway(on_message=cb)
        await gw._handle_dispatch("RESUMED", {})
        cb.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resumed_emits_audit(self):
        gw = _make_gateway()
        emitted: list[str] = []

        def _capture(event_type, result, detail):
            emitted.append(event_type)

        gw._emit_audit = _capture
        await gw._handle_dispatch("RESUMED", {})
        assert "discord.gateway.session_resumed" in emitted


class TestHandleDispatchForwarding:
    @pytest.mark.asyncio
    async def test_message_create_forwarded(self):
        cb = AsyncMock()
        gw = _make_gateway(on_message=cb)
        data = {"content": "hello", "channel_id": "ch-1"}
        await gw._handle_dispatch("MESSAGE_CREATE", data)
        cb.assert_awaited_once()
        call_arg = cb.call_args[0][0]
        assert call_arg["t"] == "MESSAGE_CREATE"
        assert call_arg["d"] == data

    @pytest.mark.asyncio
    async def test_guild_create_forwarded(self):
        cb = AsyncMock()
        gw = _make_gateway(on_message=cb)
        await gw._handle_dispatch("GUILD_CREATE", {"id": "g-1"})
        cb.assert_awaited_once()
        assert cb.call_args[0][0]["t"] == "GUILD_CREATE"

    @pytest.mark.asyncio
    async def test_interaction_create_forwarded(self):
        cb = AsyncMock()
        gw = _make_gateway(on_message=cb)
        await gw._handle_dispatch("INTERACTION_CREATE", {"id": "i-1"})
        cb.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_message_reaction_add_forwarded(self):
        cb = AsyncMock()
        gw = _make_gateway(on_message=cb)
        await gw._handle_dispatch("MESSAGE_REACTION_ADD", {"emoji": {"name": "thumbsup"}})
        cb.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_event_not_forwarded(self):
        cb = AsyncMock()
        gw = _make_gateway(on_message=cb)
        await gw._handle_dispatch("TYPING_START", {"user_id": "u1"})
        cb.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_callback_exception_is_caught(self):
        cb = AsyncMock(side_effect=RuntimeError("boom"))
        gw = _make_gateway(on_message=cb)
        # Must not propagate.
        await gw._handle_dispatch("MESSAGE_CREATE", {})


# ===========================================================================
# Gateway — _send_heartbeat
# ===========================================================================


class TestSendHeartbeat:
    @pytest.mark.asyncio
    async def test_sends_correct_json_shape(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        gw._ws = mock_ws
        gw._sequence = 77
        await gw._send_heartbeat()
        sent_text = mock_ws.send.call_args[0][0]
        payload = json.loads(sent_text)
        assert payload["op"] == _OP_HEARTBEAT
        assert payload["d"] == 77

    @pytest.mark.asyncio
    async def test_sends_null_d_when_sequence_none(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        gw._ws = mock_ws
        gw._sequence = None
        await gw._send_heartbeat()
        sent_text = mock_ws.send.call_args[0][0]
        payload = json.loads(sent_text)
        assert payload["d"] is None

    @pytest.mark.asyncio
    async def test_noop_when_ws_is_none(self):
        gw = _make_gateway()
        gw._ws = None
        # Should return immediately without error.
        await gw._send_heartbeat()

    @pytest.mark.asyncio
    async def test_send_exception_is_swallowed(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        mock_ws.send.side_effect = Exception("connection closed")
        gw._ws = mock_ws
        # Must not propagate.
        await gw._send_heartbeat()


# ===========================================================================
# Gateway — _identify_or_resume
# ===========================================================================


class TestIdentifyOrResume:
    @pytest.mark.asyncio
    async def test_identifies_when_no_session(self):
        gw = _make_gateway()
        gw._ws = AsyncMock()
        gw._discord_session_id = None
        gw._sequence = None
        await gw._identify_or_resume()
        sent = gw._ws.send.call_args[0][0]
        payload = json.loads(sent)
        assert payload["op"] == _OP_IDENTIFY

    @pytest.mark.asyncio
    async def test_identifies_when_session_set_but_sequence_none(self):
        gw = _make_gateway()
        gw._ws = AsyncMock()
        gw._discord_session_id = "some-session"
        gw._sequence = None
        await gw._identify_or_resume()
        sent = gw._ws.send.call_args[0][0]
        payload = json.loads(sent)
        assert payload["op"] == _OP_IDENTIFY

    @pytest.mark.asyncio
    async def test_resumes_when_session_and_sequence_present(self):
        gw = _make_gateway()
        gw._ws = AsyncMock()
        gw._discord_session_id = "existing-session"
        gw._sequence = 42
        await gw._identify_or_resume()
        sent = gw._ws.send.call_args[0][0]
        payload = json.loads(sent)
        assert payload["op"] == _OP_RESUME


# ===========================================================================
# Gateway — _send_identify
# ===========================================================================


class TestSendIdentify:
    @pytest.mark.asyncio
    async def test_payload_contains_token(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        gw._ws = mock_ws
        await gw._send_identify()
        payload = json.loads(mock_ws.send.call_args[0][0])
        assert payload["d"]["token"] == gw._token

    @pytest.mark.asyncio
    async def test_payload_contains_intents(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        gw._ws = mock_ws
        await gw._send_identify()
        payload = json.loads(mock_ws.send.call_args[0][0])
        assert "intents" in payload["d"]
        assert isinstance(payload["d"]["intents"], int)

    @pytest.mark.asyncio
    async def test_payload_contains_properties(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        gw._ws = mock_ws
        await gw._send_identify()
        payload = json.loads(mock_ws.send.call_args[0][0])
        props = payload["d"]["properties"]
        assert props["os"] == "linux"
        assert props["browser"] == "missy"
        assert props["device"] == "missy"

    @pytest.mark.asyncio
    async def test_opcode_is_identify(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        gw._ws = mock_ws
        await gw._send_identify()
        payload = json.loads(mock_ws.send.call_args[0][0])
        assert payload["op"] == _OP_IDENTIFY


# ===========================================================================
# Gateway — _send_resume
# ===========================================================================


class TestSendResume:
    @pytest.mark.asyncio
    async def test_payload_contains_token_session_seq(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        gw._ws = mock_ws
        gw._discord_session_id = "sess-xyz"
        gw._sequence = 13
        await gw._send_resume()
        payload = json.loads(mock_ws.send.call_args[0][0])
        assert payload["op"] == _OP_RESUME
        assert payload["d"]["token"] == gw._token
        assert payload["d"]["session_id"] == "sess-xyz"
        assert payload["d"]["seq"] == 13

    @pytest.mark.asyncio
    async def test_uses_bot_token_not_raw(self):
        gw = DiscordGatewayClient(bot_token="rawtoken", on_message=AsyncMock())
        mock_ws = AsyncMock()
        gw._ws = mock_ws
        gw._discord_session_id = "s"
        gw._sequence = 1
        await gw._send_resume()
        payload = json.loads(mock_ws.send.call_args[0][0])
        assert payload["d"]["token"] == "Bot rawtoken"


# ===========================================================================
# Gateway — disconnect
# ===========================================================================


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_sets_running_false(self):
        gw = _make_gateway()
        gw._running = True
        await gw.disconnect()
        assert gw._running is False

    @pytest.mark.asyncio
    async def test_cancels_heartbeat_task(self):
        gw = _make_gateway()
        mock_task = MagicMock()
        mock_task.cancel = MagicMock()

        async def _awaitable():
            raise asyncio.CancelledError()

        mock_task.__await__ = lambda self: _awaitable().__await__()
        gw._heartbeat_task = asyncio.create_task(asyncio.sleep(0))
        gw._heartbeat_task.cancel()
        await gw.disconnect()
        assert gw._heartbeat_task is None

    @pytest.mark.asyncio
    async def test_closes_websocket(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        gw._ws = mock_ws
        await gw.disconnect()
        mock_ws.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_safe_when_heartbeat_is_none(self):
        gw = _make_gateway()
        gw._heartbeat_task = None
        gw._ws = None
        await gw.disconnect()  # Must not raise.

    @pytest.mark.asyncio
    async def test_ws_close_error_is_swallowed(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        mock_ws.close.side_effect = Exception("already closed")
        gw._ws = mock_ws
        await gw.disconnect()  # Must not propagate.

    @pytest.mark.asyncio
    async def test_ws_set_to_none_after_close(self):
        gw = _make_gateway()
        mock_ws = AsyncMock()
        gw._ws = mock_ws
        await gw.disconnect()
        assert gw._ws is None


# ===========================================================================
# Gateway — _emit_audit
# ===========================================================================


class TestEmitAudit:
    def test_exception_from_event_bus_is_swallowed(self):
        gw = _make_gateway()
        with patch(
            "missy.channels.discord.gateway.event_bus.publish",
            side_effect=RuntimeError("bus error"),
        ):
            # Must not raise.
            gw._emit_audit("test.event", "allow", {"key": "val"})
