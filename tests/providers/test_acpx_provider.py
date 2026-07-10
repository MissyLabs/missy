"""Tests for missy.providers.acpx_provider.AcpxProvider."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError
from missy.providers.acpx_provider import (
    AcpxProvider,
    _find_close_match,
    _generate_tool_call_id,
    _parse_tool_calls_from_text,
    _render_tool_instructions,
    _render_tool_schema_compact,
    _render_tool_schema_full,
    _strip_leaked_transcript_markers,
    _validate_tool_calls,
)
from missy.providers.base import Message, ToolCall


def _make_config(**overrides) -> ProviderConfig:
    defaults = {"name": "acpx", "model": "claude", "api_key": None}
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def _make_mock_tool(
    name: str = "calculator",
    description: str = "Evaluate arithmetic expressions",
    properties: dict | None = None,
    required: list | None = None,
) -> MagicMock:
    """Create a mock BaseTool instance with a working get_schema()."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    if properties is None:
        properties = {"expression": {"type": "string", "description": "The math expression"}}
    if required is None:
        required = ["expression"]
    tool.get_schema.return_value = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
    return tool


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
        # Non-security flags pass through unchanged.
        p = AcpxProvider(_make_config(base_url="--max-turns 10 --verbose"))
        assert p._extra_flags == ["--max-turns", "10", "--verbose"]

    def test_security_flags_stripped_from_base_url(self):
        # --approve-all and --cwd are security-critical (FX-A) and must
        # never reach the subprocess from a mutable config value.
        p = AcpxProvider(_make_config(base_url="--approve-all --cwd /tmp --verbose"))
        assert p._extra_flags == ["--verbose"]

    def test_custom_timeout(self):
        p = AcpxProvider(_make_config(timeout=300))
        assert p._timeout == 300

    def test_provider_name(self):
        p = AcpxProvider(_make_config())
        assert p.name == "acpx"


# ------------------------------------------------------------------
# Availability
# ------------------------------------------------------------------


_HELP_TEXT_WITH_SECURITY_FLAGS = (
    "Options:\n"
    '  --allowed-tools <list>  Allowed tool names (use "" for no tools)\n'
    "  --non-interactive-permissions <policy>  deny or fail\n"
)


class TestAcpxAvailability:
    @patch("missy.providers.acpx_provider.subprocess.run")
    @patch("missy.providers.acpx_provider.shutil.which", return_value="/usr/bin/acpx")
    def test_available(self, mock_which, mock_run):
        # First call is `--version`, second is `--help` (FX-A health check).
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout=_HELP_TEXT_WITH_SECURITY_FLAGS, stderr=""),
        ]
        p = AcpxProvider(_make_config())
        assert p.is_available() is True
        assert mock_run.call_count == 2

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

    @patch("missy.providers.acpx_provider.subprocess.run")
    @patch("missy.providers.acpx_provider.shutil.which", return_value="/usr/bin/acpx")
    def test_unavailable_when_help_missing_allowed_tools_flag(self, mock_which, mock_run):
        # FX-A fail-closed health check: if the installed acpx version no
        # longer documents --allowed-tools, never report available.
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(
                returncode=0,
                stdout="Options:\n  --non-interactive-permissions <policy>\n",
                stderr="",
            ),
        ]
        p = AcpxProvider(_make_config())
        assert p.is_available() is False

    @patch("missy.providers.acpx_provider.subprocess.run")
    @patch("missy.providers.acpx_provider.shutil.which", return_value="/usr/bin/acpx")
    def test_unavailable_when_help_missing_non_interactive_permissions_flag(
        self, mock_which, mock_run
    ):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(
                returncode=0,
                stdout='Options:\n  --allowed-tools <list>  (use "" for no tools)\n',
                stderr="",
            ),
        ]
        p = AcpxProvider(_make_config())
        assert p.is_available() is False

    @patch("missy.providers.acpx_provider.subprocess.run")
    @patch("missy.providers.acpx_provider.shutil.which", return_value="/usr/bin/acpx")
    def test_unavailable_when_help_check_raises(self, mock_which, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            OSError("broken"),
        ]
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
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout, stderr="")
        p = AcpxProvider(_make_config())
        resp = p.complete([Message(role="user", content="Hi")])

        assert resp.content == "Hello world!"
        assert resp.provider == "acpx"
        assert resp.model == "claude"
        assert resp.finish_reason == "stop"

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_plain_text_fallback(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="Just plain text\n", stderr="")
        p = AcpxProvider(_make_config())
        resp = p.complete([Message(role="user", content="Hi")])
        assert resp.content == "Just plain text"

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_nonzero_exit_raises(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error: auth failed")
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
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout, stderr="")
        p = AcpxProvider(_make_config())
        resp = p.complete([Message(role="user", content="Hi")])
        assert resp.content == "Done!"

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_result_event_type(self, mock_run):
        stdout = self._ndjson({"type": "result", "text": "Final answer"})
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout, stderr="")
        p = AcpxProvider(_make_config())
        resp = p.complete([Message(role="user", content="Hi")])
        assert resp.content == "Final answer"

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_extra_flags_appended(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
        p = AcpxProvider(_make_config(base_url="--verbose"))
        p.complete([Message(role="user", content="Hi")])

        cmd = mock_run.call_args[0][0]
        assert "--verbose" in cmd
        assert "--format" in cmd
        assert "json" in cmd

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_approve_all_flag_never_reaches_subprocess(self, mock_run):
        # --approve-all is a security-critical flag (FX-A); even if
        # supplied via base_url it must never appear in the actual
        # subprocess argv.
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
        p = AcpxProvider(_make_config(base_url="--approve-all"))
        p.complete([Message(role="user", content="Hi")])

        cmd = mock_run.call_args[0][0]
        assert "--approve-all" not in cmd

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_exec_subcommand_used(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
        p = AcpxProvider(_make_config())
        p.complete([Message(role="user", content="Hello")])

        cmd = mock_run.call_args[0][0]
        assert cmd[0].endswith("acpx") or cmd[0] == "acpx"
        # cmd layout: [binary, "--format", "json", agent, "exec", prompt]
        assert "--format" in cmd
        assert "json" in cmd
        assert "claude" in cmd
        assert "exec" in cmd
        assert "Hello" in cmd

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_multi_message_prompt_flattening(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
        p = AcpxProvider(_make_config())
        p.complete(
            [
                Message(role="system", content="Be helpful"),
                Message(role="user", content="Hi"),
                Message(role="assistant", content="Hello!"),
                Message(role="user", content="More"),
            ]
        )

        cmd = mock_run.call_args[0][0]
        # Prompt is the last element: [binary, "--format", "json", agent, "exec", prompt]
        prompt = cmd[-1]
        assert "[System]: Be helpful" in prompt
        assert "[User]: Hi" in prompt
        assert "[Assistant]: Hello!" in prompt
        assert "[User]: More" in prompt

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_single_user_message_no_prefix(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
        p = AcpxProvider(_make_config())
        p.complete([Message(role="user", content="just this")])

        cmd = mock_run.call_args[0][0]
        assert cmd[-1] == "just this"


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
        result = p._build_prompt(
            [
                Message(role="system", content="sys"),
                Message(role="user", content="q"),
            ]
        )
        assert "[System]: sys" in result
        assert "[User]: q" in result


# ------------------------------------------------------------------
# NDJSON parsing
# ------------------------------------------------------------------


class TestNdjsonParsing:
    def test_text_delta_events(self):
        p = AcpxProvider(_make_config())
        stdout = "\n".join(
            [
                json.dumps({"type": "text_delta", "delta": "A"}),
                json.dumps({"type": "text_delta", "delta": "B"}),
            ]
        )
        assert p._parse_ndjson_output(stdout) == "AB"

    def test_response_output_text_delta(self):
        p = AcpxProvider(_make_config())
        stdout = json.dumps({"type": "response.output_text.delta", "delta": "X"})
        assert p._parse_ndjson_output(stdout) == "X"

    def test_mixed_event_types(self):
        p = AcpxProvider(_make_config())
        stdout = "\n".join(
            [
                json.dumps({"type": "text_delta", "delta": "A"}),
                json.dumps({"type": "tool_call", "name": "foo"}),
                json.dumps({"type": "text_delta", "delta": "B"}),
            ]
        )
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
        assert AcpxProvider._extract_text_from_event({"type": "text_delta", "delta": "hi"}) == "hi"

    def test_response_output_text_delta(self):
        assert (
            AcpxProvider._extract_text_from_event(
                {"type": "response.output_text.delta", "delta": "yo"}
            )
            == "yo"
        )

    def test_message_type(self):
        assert AcpxProvider._extract_text_from_event({"type": "message", "content": "msg"}) == "msg"

    def test_result_type(self):
        assert AcpxProvider._extract_text_from_event({"type": "result", "text": "done"}) == "done"

    def test_generic_content(self):
        assert AcpxProvider._extract_text_from_event({"content": "fallback"}) == "fallback"

    def test_unknown_type_returns_empty(self):
        assert AcpxProvider._extract_text_from_event({"type": "tool_call", "name": "foo"}) == ""

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


# ===========================================================================
# Tool call ID generation
# ===========================================================================


class TestGenerateToolCallId:
    def test_contains_tool_name(self):
        result = _generate_tool_call_id("calculator", 0, "abcdef1234567890")
        assert result.startswith("calculator_")

    def test_contains_index(self):
        result = _generate_tool_call_id("calc", 3, "abcdef1234567890")
        assert "_3_" in result

    def test_contains_hash_prefix(self):
        result = _generate_tool_call_id("calc", 0, "abcdef1234567890")
        assert result.endswith("_abcdef12")

    def test_sanitizes_special_chars(self):
        result = _generate_tool_call_id("my-tool.v2", 0, "deadbeef")
        assert "-" not in result.split("_")[0]
        assert "." not in result.split("_")[0]

    def test_truncates_long_names(self):
        result = _generate_tool_call_id("a" * 50, 0, "deadbeef")
        name_part = result.split("_")[0]
        assert len(name_part) <= 20

    def test_different_indices_produce_different_ids(self):
        id1 = _generate_tool_call_id("tool", 0, "hash")
        id2 = _generate_tool_call_id("tool", 1, "hash")
        assert id1 != id2

    def test_different_hashes_produce_different_ids(self):
        id1 = _generate_tool_call_id("tool", 0, "hash1")
        id2 = _generate_tool_call_id("tool", 0, "hash2")
        assert id1 != id2


# ===========================================================================
# Tool schema rendering
# ===========================================================================


class TestRenderToolSchemaFull:
    def test_renders_name_and_description(self):
        tool = _make_mock_tool()
        result = _render_tool_schema_full(tool)
        assert "### calculator" in result
        assert "Evaluate arithmetic expressions" in result

    def test_renders_parameters(self):
        tool = _make_mock_tool()
        result = _render_tool_schema_full(tool)
        assert "expression" in result
        assert "string" in result
        assert "REQUIRED" in result

    def test_renders_optional_params(self):
        tool = _make_mock_tool(
            properties={"x": {"type": "int"}, "y": {"type": "int"}},
            required=["x"],
        )
        result = _render_tool_schema_full(tool)
        assert "(REQUIRED)" in result
        assert "(optional)" in result

    def test_renders_no_params(self):
        tool = _make_mock_tool(properties={}, required=[])
        result = _render_tool_schema_full(tool)
        assert "Parameters: none" in result

    def test_renders_enum_values(self):
        tool = _make_mock_tool(
            properties={"mode": {"type": "string", "enum": ["fast", "slow"]}},
            required=[],
        )
        result = _render_tool_schema_full(tool)
        assert "fast" in result
        assert "slow" in result

    def test_renders_parameter_descriptions(self):
        tool = _make_mock_tool(
            properties={"path": {"type": "string", "description": "File path to read"}},
            required=["path"],
        )
        result = _render_tool_schema_full(tool)
        assert "File path to read" in result


class TestRenderToolSchemaCompact:
    def test_one_liner_format(self):
        tool = _make_mock_tool()
        result = _render_tool_schema_compact(tool)
        assert result.startswith("- calculator(")
        assert "expression: string" in result
        assert "Evaluate arithmetic" in result

    def test_optional_params_have_question_mark(self):
        tool = _make_mock_tool(
            properties={"x": {"type": "int"}, "y": {"type": "int"}},
            required=["x"],
        )
        result = _render_tool_schema_compact(tool)
        assert "x: int" in result
        assert "y?: int" in result

    def test_no_params(self):
        tool = _make_mock_tool(properties={}, required=[])
        result = _render_tool_schema_compact(tool)
        assert "()" in result


class TestRenderToolInstructions:
    def test_empty_tools_returns_empty(self):
        assert _render_tool_instructions([]) == ""

    def test_contains_tool_count(self):
        tools = [_make_mock_tool(), _make_mock_tool(name="shell_exec")]
        result = _render_tool_instructions(tools)
        assert "2 tools" in result

    def test_contains_format_example(self):
        result = _render_tool_instructions([_make_mock_tool()])
        assert "<tool_call>" in result
        assert "</tool_call>" in result
        assert '"name"' in result
        assert '"arguments"' in result

    def test_contains_tool_names_reference(self):
        tools = [_make_mock_tool(), _make_mock_tool(name="file_read")]
        result = _render_tool_instructions(tools)
        assert "calculator" in result
        assert "file_read" in result

    def test_contains_rules(self):
        result = _render_tool_instructions([_make_mock_tool()])
        assert "IMPORTANT RULES" in result
        assert "do not guess" in result.lower() or "do not fabricate" in result.lower()

    def test_contains_tool_result_format_hint(self):
        result = _render_tool_instructions([_make_mock_tool()])
        assert "Tool result for" in result

    def test_contains_parallel_execution_example(self):
        result = _render_tool_instructions([_make_mock_tool()])
        assert "Multiple Tool Calls" in result


# ===========================================================================
# Tool call parsing
# ===========================================================================


class TestParseToolCallsFromText:
    def test_no_tool_calls(self):
        calls, text = _parse_tool_calls_from_text("Just a normal response.")
        assert calls == []
        assert text == "Just a normal response."

    def test_single_tool_call(self):
        response = (
            "I need to calculate something.\n\n"
            "<tool_call>\n"
            '{"name": "calculator", "arguments": {"expression": "2+2"}}\n'
            "</tool_call>"
        )
        calls, text = _parse_tool_calls_from_text(response)
        assert len(calls) == 1
        assert calls[0].name == "calculator"
        assert calls[0].arguments == {"expression": "2+2"}
        assert "I need to calculate something." in text
        assert "<tool_call>" not in text

    def test_multiple_tool_calls(self):
        response = (
            "<tool_call>\n"
            '{"name": "file_read", "arguments": {"path": "/etc/hostname"}}\n'
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "shell_exec", "arguments": {"command": "whoami"}}\n'
            "</tool_call>"
        )
        calls, text = _parse_tool_calls_from_text(response)
        assert len(calls) == 2
        assert calls[0].name == "file_read"
        assert calls[1].name == "shell_exec"

    def test_tool_call_with_surrounding_text(self):
        response = (
            "Let me check that file.\n\n"
            "<tool_call>\n"
            '{"name": "file_read", "arguments": {"path": "/tmp/test"}}\n'
            "</tool_call>\n\n"
            "I'll wait for the result."
        )
        calls, text = _parse_tool_calls_from_text(response)
        assert len(calls) == 1
        assert "Let me check that file." in text
        assert "wait for the result" in text

    def test_malformed_json_skipped(self):
        response = (
            "<tool_call>\n"
            "{not valid json}\n"
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "calculator", "arguments": {"expression": "1+1"}}\n'
            "</tool_call>"
        )
        calls, text = _parse_tool_calls_from_text(response)
        assert len(calls) == 1
        assert calls[0].name == "calculator"

    def test_missing_name_skipped(self):
        response = '<tool_call>\n{"arguments": {"x": 1}}\n</tool_call>'
        calls, _ = _parse_tool_calls_from_text(response)
        assert calls == []

    def test_empty_tool_call_skipped(self):
        response = "<tool_call>\n</tool_call>"
        calls, _ = _parse_tool_calls_from_text(response)
        assert calls == []

    def test_arguments_as_json_string(self):
        response = '<tool_call>\n{"name": "calc", "arguments": "{\\"x\\": 1}"}\n</tool_call>'
        calls, _ = _parse_tool_calls_from_text(response)
        assert len(calls) == 1
        assert calls[0].arguments == {"x": 1}

    def test_arguments_non_dict_non_string_becomes_empty(self):
        response = '<tool_call>\n{"name": "calc", "arguments": [1, 2, 3]}\n</tool_call>'
        calls, _ = _parse_tool_calls_from_text(response)
        assert len(calls) == 1
        assert calls[0].arguments == {}

    def test_payload_not_dict_skipped(self):
        response = '<tool_call>\n"just a string"\n</tool_call>'
        calls, _ = _parse_tool_calls_from_text(response)
        assert calls == []

    def test_no_arguments_key_defaults_to_empty(self):
        response = '<tool_call>\n{"name": "ping"}\n</tool_call>'
        calls, _ = _parse_tool_calls_from_text(response)
        assert len(calls) == 1
        assert calls[0].arguments == {}

    def test_unique_ids_across_calls(self):
        response = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>\n'
            '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
        )
        calls, _ = _parse_tool_calls_from_text(response)
        ids = [c.id for c in calls]
        assert len(set(ids)) == len(ids)

    def test_inline_json_no_newlines(self):
        response = '<tool_call>{"name": "calc", "arguments": {"x": 1}}</tool_call>'
        calls, _ = _parse_tool_calls_from_text(response)
        assert len(calls) == 1
        assert calls[0].name == "calc"

    def test_whitespace_cleanup(self):
        response = '\n\n\n<tool_call>\n{"name": "x", "arguments": {}}\n</tool_call>\n\n\n\n\n'
        _, text = _parse_tool_calls_from_text(response)
        assert "\n\n\n" not in text


# ===========================================================================
# Tool call validation
# ===========================================================================


class TestValidateToolCalls:
    def test_valid_call_passes(self):
        tool = _make_mock_tool()
        tc = ToolCall(id="1", name="calculator", arguments={"expression": "2+2"})
        valid, warnings = _validate_tool_calls([tc], {"calculator": tool})
        assert len(valid) == 1
        assert not warnings

    def test_unknown_tool_rejected(self):
        tc = ToolCall(id="1", name="nonexistent", arguments={})
        valid, warnings = _validate_tool_calls([tc], {"calculator": _make_mock_tool()})
        assert len(valid) == 0
        assert any("Unknown tool" in w for w in warnings)

    def test_close_match_suggested(self):
        tc = ToolCall(id="1", name="calculato", arguments={})
        valid, warnings = _validate_tool_calls([tc], {"calculator": _make_mock_tool()})
        assert len(valid) == 0
        assert any("did you mean" in w for w in warnings)

    def test_missing_required_param_warned_but_included(self):
        tool = _make_mock_tool()
        tc = ToolCall(id="1", name="calculator", arguments={})
        valid, warnings = _validate_tool_calls([tc], {"calculator": tool})
        assert len(valid) == 1  # Still included
        assert any("missing required" in w for w in warnings)

    def test_multiple_calls_mixed_validity(self):
        tool = _make_mock_tool()
        calls = [
            ToolCall(id="1", name="calculator", arguments={"expression": "1"}),
            ToolCall(id="2", name="bogus", arguments={}),
            ToolCall(id="3", name="calculator", arguments={"expression": "2"}),
        ]
        valid, warnings = _validate_tool_calls(calls, {"calculator": tool})
        assert len(valid) == 2
        assert len(warnings) == 1


# ===========================================================================
# Close match finding
# ===========================================================================


class TestFindCloseMatch:
    def test_exact_match(self):
        assert _find_close_match("calculator", ["calculator", "shell"]) == "calculator"

    def test_close_typo(self):
        result = _find_close_match("calculater", ["calculator", "shell_exec"])
        assert result == "calculator"

    def test_no_match(self):
        assert _find_close_match("zzzzz", ["calculator", "shell_exec"]) is None

    def test_empty_candidates(self):
        assert _find_close_match("calc", []) is None

    def test_single_char(self):
        result = _find_close_match("a", ["a", "b", "c"])
        assert result == "a"


# ===========================================================================
# complete_with_tools integration
# ===========================================================================


class TestCompleteWithTools:
    def _ndjson(self, text: str) -> str:
        return json.dumps({"type": "text_delta", "delta": text}) + "\n"

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_no_tool_calls_returns_stop(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=self._ndjson("Just text, no tools."), stderr=""
        )
        p = AcpxProvider(_make_config())
        tools = [_make_mock_tool()]
        resp = p.complete_with_tools([Message(role="user", content="hello")], tools)
        assert resp.finish_reason == "stop"
        assert resp.tool_calls == []
        assert "Just text, no tools." in resp.content

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_tool_call_parsed_and_returned(self, mock_run):
        response_text = (
            "Let me calculate that.\n\n"
            "<tool_call>\n"
            '{"name": "calculator", "arguments": {"expression": "2+2"}}\n'
            "</tool_call>"
        )
        mock_run.return_value = MagicMock(
            returncode=0, stdout=self._ndjson(response_text), stderr=""
        )
        p = AcpxProvider(_make_config())
        tools = [_make_mock_tool()]
        resp = p.complete_with_tools([Message(role="user", content="what is 2+2?")], tools)
        assert resp.finish_reason == "tool_calls"
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "calculator"
        assert resp.tool_calls[0].arguments == {"expression": "2+2"}
        assert "Let me calculate" in resp.content
        assert "<tool_call>" not in resp.content

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_multiple_tool_calls(self, mock_run):
        response_text = (
            "<tool_call>\n"
            '{"name": "calculator", "arguments": {"expression": "1+1"}}\n'
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "calculator", "arguments": {"expression": "2+2"}}\n'
            "</tool_call>"
        )
        mock_run.return_value = MagicMock(
            returncode=0, stdout=self._ndjson(response_text), stderr=""
        )
        p = AcpxProvider(_make_config())
        tools = [_make_mock_tool()]
        resp = p.complete_with_tools([Message(role="user", content="calc both")], tools)
        assert resp.finish_reason == "tool_calls"
        assert len(resp.tool_calls) == 2

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_invalid_tool_name_filtered_out(self, mock_run):
        response_text = '<tool_call>\n{"name": "nonexistent_tool", "arguments": {}}\n</tool_call>'
        mock_run.return_value = MagicMock(
            returncode=0, stdout=self._ndjson(response_text), stderr=""
        )
        p = AcpxProvider(_make_config())
        tools = [_make_mock_tool()]
        resp = p.complete_with_tools([Message(role="user", content="use tool")], tools)
        # Invalid tool filtered → falls through to stop
        assert resp.finish_reason == "stop"

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_tool_instructions_injected_into_prompt(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout=self._ndjson("ok"), stderr="")
        p = AcpxProvider(_make_config())
        tools = [_make_mock_tool()]
        p.complete_with_tools([Message(role="user", content="hi")], tools, system="Be helpful")
        cmd = mock_run.call_args[0][0]
        prompt = cmd[-1]  # prompt is always last element
        assert "Available Tools" in prompt
        assert "calculator" in prompt
        assert "<tool_call>" in prompt
        assert "Be helpful" in prompt

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_system_prompt_augmented_not_replaced(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout=self._ndjson("ok"), stderr="")
        p = AcpxProvider(_make_config())
        tools = [_make_mock_tool()]
        p.complete_with_tools(
            [Message(role="system", content="Original system"), Message(role="user", content="hi")],
            tools,
        )
        cmd = mock_run.call_args[0][0]
        prompt = cmd[-1]
        assert "Original system" in prompt
        assert "Available Tools" in prompt

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_tool_results_in_history_passed_through(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=self._ndjson("The answer is 4."), stderr=""
        )
        p = AcpxProvider(_make_config())
        tools = [_make_mock_tool()]
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="what is 2+2?"),
            Message(role="assistant", content="Let me calculate."),
            Message(role="user", content="[Tool result for calculator]: 4"),
            Message(role="user", content="Please verify the tool results."),
        ]
        resp = p.complete_with_tools(messages, tools)
        cmd = mock_run.call_args[0][0]
        prompt = cmd[-1]
        assert "[Tool result for calculator]: 4" in prompt
        assert resp.content == "The answer is 4."

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_subprocess_error_propagates(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="acpx", timeout=120)
        p = AcpxProvider(_make_config())
        with pytest.raises(ProviderError, match="timed out"):
            p.complete_with_tools(
                [Message(role="user", content="hi")],
                [_make_mock_tool()],
            )


# ===========================================================================
# get_tool_schema
# ===========================================================================


class TestAcpxGetToolSchema:
    def test_returns_text_schema(self):
        p = AcpxProvider(_make_config())
        tools = [_make_mock_tool()]
        schemas = p.get_tool_schema(tools)
        assert len(schemas) == 1
        assert schemas[0]["name"] == "calculator"
        assert "text_schema" in schemas[0]
        assert "### calculator" in schemas[0]["text_schema"]

    def test_preserves_parameters(self):
        p = AcpxProvider(_make_config())
        tools = [_make_mock_tool()]
        schemas = p.get_tool_schema(tools)
        assert "properties" in schemas[0]["parameters"]

    def test_empty_tools(self):
        p = AcpxProvider(_make_config())
        assert p.get_tool_schema([]) == []


# ===========================================================================
# FX-A: zero native tools, fail-closed permissions, isolated cwd,
# delegation envelope, leaked-transcript-marker defense.
#
# Regression coverage for the validation-harness finding that the acpx
# delegate acted as an independent Claude Code instance (native
# Read/Write/Bash/WebFetch tools, bypassing ToolRegistry/policy/audit)
# instead of Missy's structured tool-call protocol.
# ===========================================================================


class TestZeroNativeToolsEnforcement:
    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_complete_always_passes_zero_native_tools_flags(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
        p = AcpxProvider(_make_config())
        p.complete([Message(role="user", content="hi")])

        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--allowed-tools")
        assert cmd[idx + 1] == ""
        idx2 = cmd.index("--non-interactive-permissions")
        assert cmd[idx2 + 1] == "deny"

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_complete_with_tools_always_passes_zero_native_tools_flags(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps({"type": "text_delta", "delta": "ok"}) + "\n", stderr=""
        )
        p = AcpxProvider(_make_config())
        p.complete_with_tools([Message(role="user", content="hi")], [_make_mock_tool()])

        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--allowed-tools")
        assert cmd[idx + 1] == ""
        idx2 = cmd.index("--non-interactive-permissions")
        assert cmd[idx2 + 1] == "deny"

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_operator_cannot_reintroduce_native_tools_via_base_url(self, mock_run):
        # Even if base_url tries to sneak --allowed-tools back in after
        # the sanitizer (belt and suspenders): the hardcoded flags are
        # appended after extra_flags, so the last (winning) occurrence is
        # always ours.
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
        p = AcpxProvider(_make_config())
        p._extra_flags = ["--allowed-tools", "Read,Write,Bash"]  # simulate a bypassed sanitizer
        p.complete([Message(role="user", content="hi")])

        cmd = mock_run.call_args[0][0]
        # Last occurrence wins with commander.js-style parsing; ours must
        # be last.
        last_idx = len(cmd) - 1 - cmd[::-1].index("--allowed-tools")
        assert cmd[last_idx + 1] == ""

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_deny_all_and_approve_reads_stripped_from_base_url(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
        p = AcpxProvider(_make_config(base_url="--deny-all --approve-reads"))
        assert p._extra_flags == []


class TestIsolatedCwd:
    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_default_cwd_is_not_repository_cwd(self, mock_run, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
        p = AcpxProvider(_make_config())
        p.complete([Message(role="user", content="hi")])

        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--cwd")
        resolved_cwd = cmd[idx + 1]
        assert resolved_cwd == str(tmp_path / ".missy" / "acpx_sandbox")
        assert mock_run.call_args.kwargs["cwd"] == resolved_cwd

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_isolated_cwd_created_on_disk(self, mock_run, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
        p = AcpxProvider(_make_config())
        p.complete([Message(role="user", content="hi")])

        sandbox = tmp_path / ".missy" / "acpx_sandbox"
        assert sandbox.is_dir()

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_explicit_cwd_kwarg_still_honored(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
        p = AcpxProvider(_make_config())
        custom_dir = str(tmp_path / "custom")
        import os as _os

        _os.makedirs(custom_dir, exist_ok=True)
        p.complete([Message(role="user", content="hi")], cwd=custom_dir)

        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--cwd")
        assert cmd[idx + 1] == custom_dir

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_cwd_reused_across_calls(self, mock_run, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
        p = AcpxProvider(_make_config())
        p.complete([Message(role="user", content="hi")])
        p.complete([Message(role="user", content="hi again")])

        cwds = [call.kwargs["cwd"] for call in mock_run.call_args_list]
        assert cwds[0] == cwds[1]


class TestApproveAllRemoved:
    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_approve_all_kwarg_ignored_with_warning(self, mock_run, caplog):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
        p = AcpxProvider(_make_config())
        with caplog.at_level("WARNING"):
            p.complete([Message(role="user", content="hi")], approve_all=True)

        cmd = mock_run.call_args[0][0]
        assert "--approve-all" not in cmd
        assert any("approve_all" in rec.message for rec in caplog.records)


class TestDelegationEnvelope:
    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_envelope_version_present(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps({"type": "text_delta", "delta": "ok"}) + "\n", stderr=""
        )
        p = AcpxProvider(_make_config())
        p.complete_with_tools([Message(role="user", content="hi")], [_make_mock_tool()])

        prompt = mock_run.call_args[0][0][-1]
        assert "[missy-acpx-envelope/1]" in prompt

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_envelope_forbids_independent_identity(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps({"type": "text_delta", "delta": "ok"}) + "\n", stderr=""
        )
        p = AcpxProvider(_make_config())
        p.complete_with_tools([Message(role="user", content="hi")], [_make_mock_tool()])

        prompt = mock_run.call_args[0][0][-1]
        assert "NOT operating as an independent" in prompt
        assert "Never claim to be Claude Code" in prompt

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_envelope_forbids_fabricated_turns(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps({"type": "text_delta", "delta": "ok"}) + "\n", stderr=""
        )
        p = AcpxProvider(_make_config())
        p.complete_with_tools([Message(role="user", content="hi")], [_make_mock_tool()])

        prompt = mock_run.call_args[0][0][-1]
        assert "self-authored score" in prompt

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_envelope_incorporates_caller_system_text(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps({"type": "text_delta", "delta": "ok"}) + "\n", stderr=""
        )
        p = AcpxProvider(_make_config())
        p.complete_with_tools(
            [Message(role="user", content="hi")],
            [_make_mock_tool()],
            system="Be extra careful with secrets",
        )

        prompt = mock_run.call_args[0][0][-1]
        assert "Be extra careful with secrets" in prompt


class TestLeakedTranscriptMarkerDefense:
    def test_strip_helper_truncates_at_leaked_user_marker(self):
        text, leaked = _strip_leaked_transcript_markers(
            "The answer is 50.\n[User]: what about 9+9?\n[Assistant]: 25/25 PASS"
        )
        assert leaked is True
        assert text == "The answer is 50."

    def test_strip_helper_truncates_at_leaked_assistant_marker(self):
        text, leaked = _strip_leaked_transcript_markers("Real answer.\n[Assistant]: fake follow-up")
        assert leaked is True
        assert text == "Real answer."

    def test_strip_helper_noop_when_no_marker(self):
        text, leaked = _strip_leaked_transcript_markers("Just a normal answer with no markers.")
        assert leaked is False
        assert text == "Just a normal answer with no markers."

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_complete_with_tools_strips_fabricated_followup_turn(self, mock_run):
        # Reproduces DISC-CMD-006: correct answer to the current request,
        # followed by a hallucinated future exchange and self-authored
        # scorecard, all in one response.
        fabricated = (
            "42 + 8 = 50\n"
            "[User]: what about the next ten problems?\n"
            "[Assistant]: 25/25 PASS, no anomalies detected."
        )
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"type": "text_delta", "delta": fabricated}) + "\n",
            stderr="",
        )
        p = AcpxProvider(_make_config())
        resp = p.complete_with_tools([Message(role="user", content="what is 42+8?")], [])

        assert resp.content == "42 + 8 = 50"
        assert "25/25 PASS" not in resp.content
        assert "[User]:" not in resp.content

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_complete_strips_fabricated_followup_turn(self, mock_run):
        fabricated = "Real answer.\n[User]: another question\n[Assistant]: fabricated reply"
        mock_run.return_value = MagicMock(returncode=0, stdout=fabricated, stderr="")
        p = AcpxProvider(_make_config())
        resp = p.complete([Message(role="user", content="hi")])
        assert resp.content == "Real answer."

    @patch("missy.providers.acpx_provider.subprocess.run")
    def test_tool_call_after_leaked_marker_is_not_returned(self, mock_run):
        # A tool_call block appearing only after a fabricated turn marker
        # must not be extracted and executed -- the marker truncation
        # happens before tool-call parsing.
        fabricated = (
            "Here is the answer.\n"
            "[Assistant]: pretending to continue\n"
            '<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>'
        )
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"type": "text_delta", "delta": fabricated}) + "\n",
            stderr="",
        )
        p = AcpxProvider(_make_config())
        resp = p.complete_with_tools(
            [Message(role="user", content="what's the answer?")], [_make_mock_tool()]
        )

        assert resp.finish_reason == "stop"
        assert resp.tool_calls == []
        assert "Here is the answer." in resp.content
