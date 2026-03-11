"""Tests for missy.channels.cli_channel.CLIChannel."""

from __future__ import annotations

import sys
from io import StringIO
from unittest.mock import patch

import pytest

from missy.channels.base import BaseChannel, ChannelMessage
from missy.channels.cli_channel import CLIChannel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def channel() -> CLIChannel:
    return CLIChannel()


@pytest.fixture()
def channel_with_prompt() -> CLIChannel:
    return CLIChannel(prompt="> ")


# ---------------------------------------------------------------------------
# ChannelMessage dataclass
# ---------------------------------------------------------------------------


class TestChannelMessage:
    def test_create_with_defaults(self):
        msg = ChannelMessage(content="Hello")
        assert msg.content == "Hello"
        assert msg.sender == "user"
        assert msg.channel == "cli"
        assert msg.metadata == {}

    def test_create_with_custom_fields(self):
        msg = ChannelMessage(
            content="Hi",
            sender="bot",
            channel="webhook",
            metadata={"id": "123"},
        )
        assert msg.sender == "bot"
        assert msg.channel == "webhook"
        assert msg.metadata["id"] == "123"


# ---------------------------------------------------------------------------
# BaseChannel – abstract contract
# ---------------------------------------------------------------------------


class TestBaseChannel:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseChannel()  # type: ignore[abstract]

    def test_cli_channel_is_subclass(self):
        assert issubclass(CLIChannel, BaseChannel)


# ---------------------------------------------------------------------------
# CLIChannel.receive
# ---------------------------------------------------------------------------


class TestReceive:
    def test_receive_returns_channel_message(self, channel):
        with patch("builtins.input", return_value="Hello"):
            msg = channel.receive()
        assert isinstance(msg, ChannelMessage)

    def test_receive_content_matches_input(self, channel):
        with patch("builtins.input", return_value="user typed this"):
            msg = channel.receive()
        assert msg is not None
        assert msg.content == "user typed this"

    def test_receive_sender_is_user(self, channel):
        with patch("builtins.input", return_value="anything"):
            msg = channel.receive()
        assert msg is not None
        assert msg.sender == "user"

    def test_receive_channel_is_cli(self, channel):
        with patch("builtins.input", return_value="anything"):
            msg = channel.receive()
        assert msg is not None
        assert msg.channel == "cli"

    def test_receive_metadata_is_empty_dict(self, channel):
        with patch("builtins.input", return_value="hi"):
            msg = channel.receive()
        assert msg is not None
        assert msg.metadata == {}

    def test_receive_returns_none_on_eof(self, channel):
        with patch("builtins.input", side_effect=EOFError):
            msg = channel.receive()
        assert msg is None

    def test_receive_returns_none_on_keyboard_interrupt(self, channel):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            msg = channel.receive()
        assert msg is None

    def test_receive_empty_string_input(self, channel):
        with patch("builtins.input", return_value=""):
            msg = channel.receive()
        assert msg is not None
        assert msg.content == ""

    def test_receive_uses_configured_prompt(self, channel_with_prompt):
        captured_prompts = []

        def fake_input(prompt=""):
            captured_prompts.append(prompt)
            return "hello"

        with patch("builtins.input", side_effect=fake_input):
            channel_with_prompt.receive()

        assert captured_prompts == ["> "]

    def test_receive_no_prompt_by_default(self, channel):
        captured_prompts = []

        def fake_input(prompt=""):
            captured_prompts.append(prompt)
            return "hello"

        with patch("builtins.input", side_effect=fake_input):
            channel.receive()

        assert captured_prompts == [""]


# ---------------------------------------------------------------------------
# CLIChannel.send
# ---------------------------------------------------------------------------


class TestSend:
    def test_send_writes_to_stdout(self, channel):
        captured = StringIO()
        with patch("sys.stdout", captured):
            channel.send("Hello, output!")
        assert "Hello, output!" in captured.getvalue()

    def test_send_appends_newline(self, channel):
        captured = StringIO()
        with patch("sys.stdout", captured):
            channel.send("line")
        # print() adds a newline
        assert captured.getvalue() == "line\n"

    def test_send_empty_string(self, channel):
        captured = StringIO()
        with patch("sys.stdout", captured):
            channel.send("")
        assert captured.getvalue() == "\n"

    def test_send_multiline(self, channel):
        captured = StringIO()
        with patch("sys.stdout", captured):
            channel.send("line1\nline2")
        assert "line1" in captured.getvalue()
        assert "line2" in captured.getvalue()


# ---------------------------------------------------------------------------
# CLIChannel metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_name_is_cli(self, channel):
        assert channel.name == "cli"

    def test_default_prompt_is_empty_string(self, channel):
        assert channel._prompt == ""

    def test_custom_prompt_stored(self, channel_with_prompt):
        assert channel_with_prompt._prompt == "> "
