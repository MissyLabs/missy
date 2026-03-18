"""Tests for missy.providers.acpx_provider.AcpxProvider."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError
from missy.providers.acpx_provider import AcpxProvider
from missy.providers.base import Message


def _make_config(**overrides) -> ProviderConfig:
    defaults = {"name": "acpx", "model": "claude", "api_key": None}
    defaults.update(overrides)
    return ProviderConfig(**defaults)


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------


class TestAcpxInit:
    def test_defaults(self):
        p = AcpxProvider(_make_config())
        assert p._agent == "claude"
        assert p._timeout == 30  # ProviderConfig default
        assert p._extra_flags == []

    def test_custom_agent(self):
        p = AcpxProvider(_make_config(model="codex"))
        assert p._agent == "codex"

    def test_default_agent_when_empty(self):
        p = AcpxProvider(_make_config(model=""))
        assert p._agent == "claude"

    def test_extra_flags_from_base_url(self):
        p = AcpxProvider(_make_config(base_url="--approve-all --cwd /tmp"))
        assert p._extra_flags == ["--approve-all", "--cwd", "/tmp"]

    def test_custom_timeout(self):
        p = AcpxProvider(_make_config(timeout=300))
        assert p._timeout == 300

    def test_provider_name(self):
        p = AcpxProvider(_make_config())
        assert p.name == "acpx"


# ------------------------------------------------------------------
# Availability
# ------------------------------------------------------------------


class TestAcpxAvailability:
    @patch("missy.providers.acpx_provider.subprocess.run")
    @patch("missy.providers.acpx_provider.shutil.which", return_value="/usr/bin/acpx")
    def test_available(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        p = AcpxProvider(_make_config())
        assert p.is_available() is True
        mock_run.assert_called_once()

    @patch("missy.providers.acpx_provider.shutil.which", return_value=None)
    def test_unavailable_no_binary(self, mock_which):
        p = AcpxProvider(_make_config())
        assert p.is_available() is False

    @patch("missy.providers.acpx_provider.subprocess.run")
    @patch("missy.providers.acpx_provider.shutil.which", return_value="/usr/bin/acpx")
    def test_unavailable_nonzero_exit(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        p = AcpxProvider(_make_config())
        assert p.is_available() is False

    @patch("missy.providers.acpx_provider.subprocess.run")
    @patch("missy.providers.acpx_provider.shutil.which", return_value="/usr/bin/acpx")
    def test_unavailable_exception(self, mock_which, mock_run):
        mock_run.side_effect = OSError("broken")
        p = AcpxProvider(_make_config())
        assert p.is_available() is False


# ------------------------------------------------------------------
# Completion
# ------------------------------------------------------------------


class TestAcpxComplete:
    def _ndjson(self, *events: dict) -> str:
        return "\n".join(json.dumps(e) for e in events) + "\n"

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_successful_completion(self, mock_run):
        stdout = self._ndjson(
            {"type": "text_delta", "delta": "Hello "},
            {"type": "text_delta", "delta": "world!"},
        )
        mock_run.return_value = MagicMock(
            returncode=0, stdout=stdout, stderr=""
        )
        p = AcpxProvider(_make_config())
        resp = p.complete([Message(role="user", content="Hi")])

        assert resp.content == "Hello world!"
        assert resp.provider == "acpx"
        assert resp.model == "claude"
        assert resp.finish_reason == "stop"

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_plain_text_fallback(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Just plain text\n", stderr=""
        )
        p = AcpxProvider(_make_config())
        resp = p.complete([Message(role="user", content="Hi")])
        assert resp.content == "Just plain text"

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_nonzero_exit_raises(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="error: auth failed"
        )
        p = AcpxProvider(_make_config())
        with pytest.raises(ProviderError, match="exit.*1"):
            p.complete([Message(role="user", content="Hi")])

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_timeout_raises(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="acpx", timeout=120)
        p = AcpxProvider(_make_config())
        with pytest.raises(ProviderError, match="timed out"):
            p.complete([Message(role="user", content="Hi")])

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_binary_not_found_raises(self, mock_run):
        mock_run.side_effect = FileNotFoundError("acpx")
        p = AcpxProvider(_make_config())
        with pytest.raises(ProviderError, match="not found"):
            p.complete([Message(role="user", content="Hi")])

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_message_event_type(self, mock_run):
        stdout = self._ndjson({"type": "message", "content": "Done!"})
        mock_run.return_value = MagicMock(
            returncode=0, stdout=stdout, stderr=""
        )
        p = AcpxProvider(_make_config())
        resp = p.complete([Message(role="user", content="Hi")])
        assert resp.content == "Done!"

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_result_event_type(self, mock_run):
        stdout = self._ndjson({"type": "result", "text": "Final answer"})
        mock_run.return_value = MagicMock(
            returncode=0, stdout=stdout, stderr=""
        )
        p = AcpxProvider(_make_config())
        resp = p.complete([Message(role="user", content="Hi")])
        assert resp.content == "Final answer"

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_extra_flags_appended(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="ok\n", stderr=""
        )
        p = AcpxProvider(_make_config(base_url="--approve-all"))
        p.complete([Message(role="user", content="Hi")])

        cmd = mock_run.call_args[0][0]
        assert "--approve-all" in cmd
        assert "--format" in cmd
        assert "json" in cmd

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_exec_subcommand_used(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="ok\n", stderr=""
        )
        p = AcpxProvider(_make_config())
        p.complete([Message(role="user", content="Hello")])

        cmd = mock_run.call_args[0][0]
        assert cmd[0].endswith("acpx") or cmd[0] == "acpx"
        assert cmd[1] == "claude"
        assert cmd[2] == "exec"
        assert cmd[3] == "Hello"

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_multi_message_prompt_flattening(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="ok\n", stderr=""
        )
        p = AcpxProvider(_make_config())
        p.complete([
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
            Message(role="user", content="More"),
        ])

        cmd = mock_run.call_args[0][0]
        prompt = cmd[3]
        assert "[System]: Be helpful" in prompt
        assert "[User]: Hi" in prompt
        assert "[Assistant]: Hello!" in prompt
        assert "[User]: More" in prompt

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_single_user_message_no_prefix(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="ok\n", stderr=""
        )
        p = AcpxProvider(_make_config())
        p.complete([Message(role="user", content="just this")])

        cmd = mock_run.call_args[0][0]
        assert cmd[3] == "just this"


# ------------------------------------------------------------------
# Prompt building
# ------------------------------------------------------------------


class TestBuildPrompt:
    def test_single_user_message(self):
        p = AcpxProvider(_make_config())
        result = p._build_prompt([Message(role="user", content="hello")])
        assert result == "hello"

    def test_multi_turn(self):
        p = AcpxProvider(_make_config())
        result = p._build_prompt([
            Message(role="system", content="sys"),
            Message(role="user", content="q"),
        ])
        assert "[System]: sys" in result
        assert "[User]: q" in result


# ------------------------------------------------------------------
# NDJSON parsing
# ------------------------------------------------------------------


class TestNdjsonParsing:
    def test_text_delta_events(self):
        p = AcpxProvider(_make_config())
        stdout = "\n".join([
            json.dumps({"type": "text_delta", "delta": "A"}),
            json.dumps({"type": "text_delta", "delta": "B"}),
        ])
        assert p._parse_ndjson_output(stdout) == "AB"

    def test_response_output_text_delta(self):
        p = AcpxProvider(_make_config())
        stdout = json.dumps({"type": "response.output_text.delta", "delta": "X"})
        assert p._parse_ndjson_output(stdout) == "X"

    def test_mixed_event_types(self):
        p = AcpxProvider(_make_config())
        stdout = "\n".join([
            json.dumps({"type": "text_delta", "delta": "A"}),
            json.dumps({"type": "tool_call", "name": "foo"}),
            json.dumps({"type": "text_delta", "delta": "B"}),
        ])
        assert p._parse_ndjson_output(stdout) == "AB"

    def test_plain_text_fallback(self):
        p = AcpxProvider(_make_config())
        assert p._parse_ndjson_output("just text") == "just text"

    def test_empty_output(self):
        p = AcpxProvider(_make_config())
        assert p._parse_ndjson_output("") == ""

    def test_generic_content_field(self):
        p = AcpxProvider(_make_config())
        stdout = json.dumps({"content": "generic"})
        assert p._parse_ndjson_output(stdout) == "generic"


# ------------------------------------------------------------------
# Event extraction
# ------------------------------------------------------------------


class TestExtractTextFromEvent:
    def test_text_delta(self):
        assert AcpxProvider._extract_text_from_event(
            {"type": "text_delta", "delta": "hi"}
        ) == "hi"

    def test_response_output_text_delta(self):
        assert AcpxProvider._extract_text_from_event(
            {"type": "response.output_text.delta", "delta": "yo"}
        ) == "yo"

    def test_message_type(self):
        assert AcpxProvider._extract_text_from_event(
            {"type": "message", "content": "msg"}
        ) == "msg"

    def test_result_type(self):
        assert AcpxProvider._extract_text_from_event(
            {"type": "result", "text": "done"}
        ) == "done"

    def test_generic_content(self):
        assert AcpxProvider._extract_text_from_event(
            {"content": "fallback"}
        ) == "fallback"

    def test_unknown_type_returns_empty(self):
        assert AcpxProvider._extract_text_from_event(
            {"type": "tool_call", "name": "foo"}
        ) == ""

    def test_empty_event(self):
        assert AcpxProvider._extract_text_from_event({}) == ""


# ------------------------------------------------------------------
# Registry integration
# ------------------------------------------------------------------


class TestRegistryIntegration:
    def test_acpx_in_provider_classes(self):
        from missy.providers.registry import _PROVIDER_CLASSES

        assert "acpx" in _PROVIDER_CLASSES
        assert _PROVIDER_CLASSES["acpx"] is AcpxProvider

    def test_provider_repr(self):
        p = AcpxProvider(_make_config())
        assert "AcpxProvider" in repr(p)
        assert "acpx" in repr(p)
