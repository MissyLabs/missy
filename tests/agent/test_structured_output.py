"""Tests for missy.agent.structured_output — Structured Output Validation."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from missy.agent.structured_output import (
    ConversationSummary,
    ErrorAnalysis,
    OutputSchema,
    StructuredOutputRunner,
    StructuredResult,
    TaskAnalysis,
    _build_system,
    _format_errors_as_feedback,
)
from missy.providers.base import CompletionResponse, Message

# ---------------------------------------------------------------------------
# Helper models and fixtures
# ---------------------------------------------------------------------------


class SimpleModel(BaseModel):
    """A simple test model."""

    name: str
    value: int


class NestedModel(BaseModel):
    items: list[str]
    meta: dict[str, Any] = Field(default_factory=dict)


class StrictModel(BaseModel):
    required_field: str


def _make_response(content: str) -> CompletionResponse:
    return CompletionResponse(
        content=content,
        model="test-model",
        provider="test",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        raw={},
    )


def _make_provider(*responses: str) -> MagicMock:
    """Return a mock provider that returns each content string in order."""
    provider = MagicMock()
    provider.complete.side_effect = [_make_response(r) for r in responses]
    return provider


# ---------------------------------------------------------------------------
# OutputSchema.to_json_schema
# ---------------------------------------------------------------------------


class TestToJsonSchema:
    def test_returns_dict(self):
        schema = OutputSchema(SimpleModel)
        result = schema.to_json_schema()
        assert isinstance(result, dict)

    def test_contains_properties(self):
        schema = OutputSchema(SimpleModel)
        result = schema.to_json_schema()
        assert "properties" in result
        assert "name" in result["properties"]
        assert "value" in result["properties"]

    def test_nested_model(self):
        schema = OutputSchema(NestedModel)
        result = schema.to_json_schema()
        assert "items" in result["properties"]


# ---------------------------------------------------------------------------
# OutputSchema.to_prompt_instruction
# ---------------------------------------------------------------------------


class TestToPromptInstruction:
    def test_contains_json_keyword(self):
        schema = OutputSchema(SimpleModel)
        instruction = schema.to_prompt_instruction()
        assert "JSON" in instruction

    def test_contains_schema_properties(self):
        schema = OutputSchema(SimpleModel)
        instruction = schema.to_prompt_instruction()
        assert "name" in instruction
        assert "value" in instruction

    def test_contains_no_text_before_after_directive(self):
        schema = OutputSchema(SimpleModel)
        instruction = schema.to_prompt_instruction()
        assert "Do not include any text before or after the JSON" in instruction

    def test_description_included_when_provided(self):
        schema = OutputSchema(SimpleModel, description="My description")
        instruction = schema.to_prompt_instruction()
        assert "My description" in instruction

    def test_docstring_used_as_default_description(self):
        schema = OutputSchema(SimpleModel)
        instruction = schema.to_prompt_instruction()
        assert "simple test model" in instruction

    def test_custom_description_overrides_docstring(self):
        schema = OutputSchema(SimpleModel, description="Custom")
        assert schema.description == "Custom"

    def test_empty_description_falls_back_to_docstring(self):
        schema = OutputSchema(SimpleModel, description="")
        assert "simple test model" in schema.description.lower()

    def test_instruction_contains_json_fence(self):
        schema = OutputSchema(SimpleModel)
        instruction = schema.to_prompt_instruction()
        assert "```json" in instruction


# ---------------------------------------------------------------------------
# OutputSchema._extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def setup_method(self):
        self.schema = OutputSchema(SimpleModel)

    def test_raw_json_object(self):
        result = self.schema._extract_json('{"name": "Alice", "value": 1}')
        assert result == '{"name": "Alice", "value": 1}'

    def test_raw_json_array(self):
        result = self.schema._extract_json("[1, 2, 3]")
        assert result == "[1, 2, 3]"

    def test_json_code_block(self):
        text = '```json\n{"name": "Bob", "value": 2}\n```'
        result = self.schema._extract_json(text)
        assert result == '{"name": "Bob", "value": 2}'

    def test_plain_code_block(self):
        text = '```\n{"name": "Carol", "value": 3}\n```'
        result = self.schema._extract_json(text)
        assert result == '{"name": "Carol", "value": 3}'

    def test_json_embedded_in_prose(self):
        text = 'Here is the result: {"name": "Dave", "value": 4} as requested.'
        result = self.schema._extract_json(text)
        assert result == '{"name": "Dave", "value": 4}'

    def test_empty_string_returns_none(self):
        assert self.schema._extract_json("") is None

    def test_whitespace_only_returns_none(self):
        assert self.schema._extract_json("   \n  ") is None

    def test_no_json_returns_none(self):
        assert self.schema._extract_json("Just plain text here.") is None

    def test_partial_json_no_close_returns_none(self):
        # Opening brace with no closing brace — rfind('}') == -1
        result = self.schema._extract_json("Some text { incomplete")
        assert result is None

    def test_code_block_without_json_tag(self):
        text = '```\n{"x": 1}\n```'
        result = self.schema._extract_json(text)
        assert result == '{"x": 1}'

    def test_strips_surrounding_whitespace(self):
        result = self.schema._extract_json('  {"name": "Eve", "value": 5}  ')
        assert result == '{"name": "Eve", "value": 5}'


# ---------------------------------------------------------------------------
# OutputSchema.parse
# ---------------------------------------------------------------------------


class TestParse:
    def setup_method(self):
        self.schema = OutputSchema(SimpleModel)

    def test_valid_json_success(self):
        result = self.schema.parse('{"name": "Alice", "value": 42}')
        assert result.success is True
        assert result.data is not None
        assert result.data.name == "Alice"
        assert result.data.value == 42

    def test_valid_json_in_code_block(self):
        content = '```json\n{"name": "Bob", "value": 7}\n```'
        result = self.schema.parse(content)
        assert result.success is True
        assert result.data.name == "Bob"

    def test_raw_content_preserved(self):
        raw = '{"name": "Carol", "value": 0}'
        result = self.schema.parse(raw)
        assert result.raw_content == raw

    def test_missing_required_field_fails(self):
        result = self.schema.parse('{"name": "Dave"}')
        assert result.success is False
        assert result.data is None
        assert len(result.validation_errors) > 0

    def test_wrong_type_fails(self):
        result = self.schema.parse('{"name": "Eve", "value": "not-an-int"}')
        assert result.success is False

    def test_empty_response_fails(self):
        result = self.schema.parse("")
        assert result.success is False
        assert "No JSON" in result.validation_errors[0]

    def test_non_json_prose_fails(self):
        result = self.schema.parse("I cannot provide that information.")
        assert result.success is False
        assert result.validation_errors

    def test_invalid_json_fails(self):
        result = self.schema.parse("{bad json}")
        assert result.success is False
        assert any("Invalid JSON" in e or "JSON" in e for e in result.validation_errors)

    def test_attempts_defaults_to_one(self):
        result = self.schema.parse('{"name": "X", "value": 1}')
        assert result.attempts == 1

    def test_partial_json_missing_close_brace_fails(self):
        result = self.schema.parse('{"name": "X", "value"')
        assert result.success is False

    def test_extra_fields_allowed_by_default(self):
        # Pydantic v2 ignores extra fields by default (model_config not set)
        result = self.schema.parse('{"name": "X", "value": 1, "extra": true}')
        assert result.success is True


# ---------------------------------------------------------------------------
# OutputSchema.format_validation_error
# ---------------------------------------------------------------------------


class TestFormatValidationError:
    def test_returns_string(self):
        schema = OutputSchema(SimpleModel)
        try:
            SimpleModel.model_validate({})
        except Exception as exc:
            formatted = schema.format_validation_error(exc)
            assert isinstance(formatted, str)

    def test_contains_please_fix(self):
        schema = OutputSchema(SimpleModel)
        try:
            SimpleModel.model_validate({})
        except Exception as exc:
            formatted = schema.format_validation_error(exc)
            assert "Please fix these errors" in formatted

    def test_contains_field_name(self):
        schema = OutputSchema(SimpleModel)
        try:
            SimpleModel.model_validate({"value": 1})  # missing 'name'
        except Exception as exc:
            formatted = schema.format_validation_error(exc)
            assert "name" in formatted

    def test_lists_multiple_errors(self):
        schema = OutputSchema(SimpleModel)
        try:
            SimpleModel.model_validate({})  # missing both fields
        except Exception as exc:
            formatted = schema.format_validation_error(exc)
            # Should have at least one bullet per missing field
            assert formatted.count("- field") >= 1


# ---------------------------------------------------------------------------
# StructuredResult dataclass
# ---------------------------------------------------------------------------


class TestStructuredResult:
    def test_success_fields(self):
        model = SimpleModel(name="A", value=1)
        result: StructuredResult[SimpleModel] = StructuredResult(
            success=True, data=model, raw_content='{"name":"A","value":1}'
        )
        assert result.success is True
        assert result.data is model
        assert result.validation_errors == []
        assert result.attempts == 1

    def test_failure_fields(self):
        result: StructuredResult[SimpleModel] = StructuredResult(
            success=False,
            data=None,
            raw_content="bad",
            validation_errors=["error 1"],
            attempts=3,
        )
        assert result.success is False
        assert result.data is None
        assert result.attempts == 3
        assert "error 1" in result.validation_errors


# ---------------------------------------------------------------------------
# StructuredOutputRunner — sync path
# ---------------------------------------------------------------------------


class TestStructuredOutputRunnerSync:
    def test_succeeds_on_first_attempt(self):
        provider = _make_provider('{"name": "Alice", "value": 1}')
        runner = StructuredOutputRunner(provider)
        schema = OutputSchema(SimpleModel, max_retries=2)
        result = runner.complete_structured([Message(role="user", content="hello")], schema)
        assert result.success is True
        assert result.data.name == "Alice"
        assert result.attempts == 1
        provider.complete.assert_called_once()

    def test_retries_on_validation_failure(self):
        # First response: invalid JSON; second: valid
        provider = _make_provider(
            "Sorry, here is no JSON.",
            '{"name": "Bob", "value": 2}',
        )
        runner = StructuredOutputRunner(provider)
        schema = OutputSchema(SimpleModel, max_retries=2)
        result = runner.complete_structured([Message(role="user", content="hello")], schema)
        assert result.success is True
        assert result.data.name == "Bob"
        assert result.attempts == 2
        assert provider.complete.call_count == 2

    def test_returns_failure_when_all_retries_exhausted(self):
        provider = _make_provider("nope", "still nope", "nope again")
        runner = StructuredOutputRunner(provider)
        schema = OutputSchema(SimpleModel, max_retries=2)
        result = runner.complete_structured([Message(role="user", content="hello")], schema)
        assert result.success is False
        assert result.attempts == 3
        assert provider.complete.call_count == 3

    def test_error_feedback_appended_to_history(self):
        """Validation errors from attempt N should appear in attempt N+1's messages."""
        provider = _make_provider(
            "not json",
            '{"name": "Carol", "value": 3}',
        )
        runner = StructuredOutputRunner(provider)
        schema = OutputSchema(SimpleModel, max_retries=1)
        runner.complete_structured([Message(role="user", content="go")], schema)

        # Second call receives the extended history (original + assistant + user feedback)
        second_call_messages = provider.complete.call_args_list[1][0][0]
        roles = [m.role for m in second_call_messages]
        assert roles[-2] == "assistant"
        assert roles[-1] == "user"
        last_content = second_call_messages[-1].content
        assert "validation" in last_content.lower() or "error" in last_content.lower()

    def test_system_prompt_includes_schema_instruction(self):
        provider = _make_provider('{"name": "X", "value": 0}')
        runner = StructuredOutputRunner(provider)
        schema = OutputSchema(SimpleModel)
        runner.complete_structured(
            [Message(role="user", content="hi")],
            schema,
            system="Be helpful.",
        )
        call_kwargs = provider.complete.call_args
        system_arg = call_kwargs[1].get("system") or call_kwargs[0][1]
        assert "Be helpful." in system_arg
        assert "JSON" in system_arg

    def test_zero_retries_no_retry(self):
        provider = _make_provider("bad")
        runner = StructuredOutputRunner(provider)
        schema = OutputSchema(SimpleModel, max_retries=0)
        result = runner.complete_structured([Message(role="user", content="hi")], schema)
        assert result.success is False
        assert provider.complete.call_count == 1

    def test_max_retries_one(self):
        provider = _make_provider("bad", '{"name": "Y", "value": 9}')
        runner = StructuredOutputRunner(provider)
        schema = OutputSchema(SimpleModel, max_retries=1)
        result = runner.complete_structured([Message(role="user", content="hi")], schema)
        assert result.success is True
        assert result.attempts == 2


# ---------------------------------------------------------------------------
# StructuredOutputRunner — async path
# ---------------------------------------------------------------------------


class TestStructuredOutputRunnerAsync:
    @pytest.mark.asyncio
    async def test_async_succeeds_first_attempt(self):
        provider = _make_provider('{"name": "Async", "value": 42}')
        runner = StructuredOutputRunner(provider)
        schema = OutputSchema(SimpleModel, max_retries=1)
        result = await runner.acomplete_structured([Message(role="user", content="hi")], schema)
        assert result.success is True
        assert result.data.value == 42

    @pytest.mark.asyncio
    async def test_async_retries_on_failure(self):
        provider = _make_provider("bad", '{"name": "Async2", "value": 5}')
        runner = StructuredOutputRunner(provider)
        schema = OutputSchema(SimpleModel, max_retries=2)
        result = await runner.acomplete_structured([Message(role="user", content="hi")], schema)
        assert result.success is True
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_async_uses_acomplete_when_available(self):
        """When the provider has an acomplete coroutine, it should be awaited."""
        provider = MagicMock()

        async def _acomplete(*args, **kwargs):
            return _make_response('{"name": "Z", "value": 0}')

        provider.acomplete = _acomplete

        runner = StructuredOutputRunner(provider)
        schema = OutputSchema(SimpleModel, max_retries=0)
        result = await runner.acomplete_structured([Message(role="user", content="hi")], schema)
        assert result.success is True
        provider.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Common schemas
# ---------------------------------------------------------------------------


class TestTaskAnalysis:
    def test_valid_instantiation(self):
        obj = TaskAnalysis(
            task_type="code",
            complexity="moderate",
            tools_needed=["shell_exec", "file_read"],
            approach="Write a Python script",
        )
        assert obj.task_type == "code"
        assert "shell_exec" in obj.tools_needed

    def test_schema_parse_success(self):
        schema = OutputSchema(TaskAnalysis)
        payload = json.dumps(
            {
                "task_type": "research",
                "complexity": "simple",
                "tools_needed": [],
                "approach": "Search the web",
            }
        )
        result = schema.parse(payload)
        assert result.success is True
        assert result.data.task_type == "research"

    def test_schema_parse_missing_field(self):
        schema = OutputSchema(TaskAnalysis)
        result = schema.parse('{"task_type": "code"}')
        assert result.success is False


class TestErrorAnalysis:
    def test_valid_instantiation(self):
        obj = ErrorAnalysis(
            error_type="runtime",
            root_cause="Null pointer dereference",
            suggested_fix="Check for None before access",
            can_retry=False,
        )
        assert obj.can_retry is False

    def test_schema_parse_success(self):
        schema = OutputSchema(ErrorAnalysis)
        payload = json.dumps(
            {
                "error_type": "network",
                "root_cause": "DNS failure",
                "suggested_fix": "Check connectivity",
                "can_retry": True,
            }
        )
        result = schema.parse(payload)
        assert result.success is True
        assert result.data.can_retry is True

    def test_bool_field_required(self):
        schema = OutputSchema(ErrorAnalysis)
        result = schema.parse('{"error_type": "logic", "root_cause": "x", "suggested_fix": "y"}')
        assert result.success is False


class TestConversationSummary:
    def test_valid_instantiation(self):
        obj = ConversationSummary(
            key_topics=["deployment"],
            decisions_made=["Use docker"],
            action_items=["Write Dockerfile"],
            entities_mentioned=["Alice", "server1"],
            overall_summary="The team decided to containerise the app.",
        )
        assert obj.key_topics == ["deployment"]

    def test_schema_parse_success(self):
        schema = OutputSchema(ConversationSummary)
        payload = json.dumps(
            {
                "key_topics": ["auth"],
                "decisions_made": ["Use JWT"],
                "action_items": ["Implement refresh tokens"],
                "entities_mentioned": ["Bob"],
                "overall_summary": "Auth discussion.",
            }
        )
        result = schema.parse(payload)
        assert result.success is True

    def test_missing_overall_summary_fails(self):
        schema = OutputSchema(ConversationSummary)
        result = schema.parse(
            json.dumps(
                {
                    "key_topics": [],
                    "decisions_made": [],
                    "action_items": [],
                    "entities_mentioned": [],
                }
            )
        )
        assert result.success is False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class TestBuildSystem:
    def test_appends_instruction_to_system(self):
        schema = OutputSchema(SimpleModel)
        result = _build_system("Be helpful.", schema)
        assert result.startswith("Be helpful.")
        assert "JSON" in result

    def test_empty_system_returns_instruction_only(self):
        schema = OutputSchema(SimpleModel)
        result = _build_system("", schema)
        assert result == schema.to_prompt_instruction()

    def test_separator_between_system_and_instruction(self):
        schema = OutputSchema(SimpleModel)
        result = _build_system("System.", schema)
        assert "\n\n" in result


class TestFormatErrorsAsFeedback:
    def test_returns_string(self):
        assert isinstance(_format_errors_as_feedback(["err"]), str)

    def test_empty_list_returns_generic_message(self):
        result = _format_errors_as_feedback([])
        assert "valid JSON" in result

    def test_errors_appear_in_output(self):
        result = _format_errors_as_feedback(["field x is required"])
        assert "field x is required" in result

    def test_contains_please_fix(self):
        result = _format_errors_as_feedback(["something wrong"])
        assert "Please fix" in result
