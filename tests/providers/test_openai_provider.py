"""Tests for missy.providers.openai_provider.OpenAIProvider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

import missy.policy.engine as engine_module
from missy.agent.structured_output import OutputSchema
from missy.config.settings import ProviderConfig, get_default_config
from missy.core.events import event_bus
from missy.core.exceptions import ProviderError
from missy.policy.engine import init_policy_engine
from missy.providers.base import Message
from missy.providers.openai_provider import OpenAIProvider


class StructuredAnswer(BaseModel):
    answer: str
    confidence: float


def _make_config(**overrides) -> ProviderConfig:
    defaults = {"name": "openai", "model": "gpt-4o", "api_key": "sk-test-key"}
    defaults.update(overrides)
    return ProviderConfig(**defaults)


class TestOpenAIInit:
    def test_defaults(self):
        p = OpenAIProvider(_make_config())
        assert p._model == "gpt-4o"
        assert p._api_key == "sk-test-key"
        assert p._base_url is None

    def test_custom_base_url(self):
        p = OpenAIProvider(_make_config(base_url="http://local:8080/v1"))
        assert p._base_url == "http://local:8080/v1"

    def test_default_model_when_empty(self):
        p = OpenAIProvider(_make_config(model=""))
        assert p._model == "auto"

    def test_api_key_setter_resets_cached_client(self):
        p = OpenAIProvider(_make_config())
        p._client = object()
        p._resolved_model = "gpt-5.4"
        p.api_key = "sk-new-key"
        assert p._client is None
        assert p._resolved_model is None
        assert p.api_key == "sk-new-key"


class TestOpenAIAvailability:
    def test_available(self):
        p = OpenAIProvider(_make_config())
        assert p.is_available() is True

    def test_unavailable_no_key(self):
        p = OpenAIProvider(_make_config(api_key=None))
        assert p.is_available() is False

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env-key"})
    def test_available_with_env_key(self):
        p = OpenAIProvider(_make_config(api_key=None))
        assert p.is_available() is True


class TestOpenAIDiagnostics:
    def setup_method(self):
        self._original_engine = engine_module._engine

    def teardown_method(self):
        engine_module._engine = self._original_engine

    def test_diagnostics_report_redacted_key_source_and_openai_policy(self):
        cfg = get_default_config()
        cfg.network.allowed_hosts = ["api.openai.com"]
        init_policy_engine(cfg)

        secret = "sk-test-diagnostic-secret-abcdefghijklmnopqrstuvwxyz"
        p = OpenAIProvider(_make_config(api_key=secret, model="auto"))

        report = p.diagnostics()
        rendered = str(report)

        assert report["provider"] == "openai"
        assert secret not in rendered
        assert any(
            check["name"] == "credential" and check["summary"] == "configured via config"
            for check in report["checks"]
        )
        assert any(
            check["name"] == "network_policy" and check["status"] == "ok"
            for check in report["checks"]
        )
        assert any(
            check["name"] == "capabilities" and check["summary"]["structured_output"] is True
            for check in report["checks"]
        )

    def test_diagnostics_warn_on_missing_custom_endpoint_allowlist(self):
        cfg = get_default_config()
        cfg.network.allowed_hosts = ["api.openai.com"]
        cfg.network.allowed_domains = []
        cfg.network.provider_allowed_hosts = []
        init_policy_engine(cfg)

        p = OpenAIProvider(
            _make_config(
                api_key=None,
                base_url="https://token:secret@example.invalid/openai/v1?api_key=bad",
            )
        )

        report = p.diagnostics()
        rendered = str(report)

        assert "token:secret" not in rendered
        assert "api_key=bad" not in rendered
        assert any(
            check["name"] == "endpoint"
            and check["summary"]["host"] == "example.invalid"
            and check["summary"]["base_url_override"] is True
            for check in report["checks"]
        )
        assert any(
            check["name"] == "network_policy"
            and check["status"] == "warn"
            and "example.invalid" in check["summary"]
            for check in report["checks"]
        )


class TestOpenAIComplete:
    def _mock_response(self, text="Hello!", model="gpt-4o"):
        choice = SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
            finish_reason="stop",
        )
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        return SimpleNamespace(
            choices=[choice],
            model=model,
            usage=usage,
            model_dump=dict,
        )

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_successful_completion(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        resp = p.complete([Message(role="user", content="Hi")])
        assert resp.content == "Hello!"
        assert resp.provider == "openai"
        assert resp.usage["total_tokens"] == 30

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_timeout_error(self, mock_sdk):
        _APIError = type("APIError", (Exception,), {})
        _APITimeoutError = type("APITimeoutError", (_APIError,), {})
        _AuthenticationError = type("AuthenticationError", (_APIError,), {})
        mock_sdk.APIError = _APIError
        mock_sdk.APITimeoutError = _APITimeoutError
        mock_sdk.AuthenticationError = _AuthenticationError
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _APITimeoutError("timeout")
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        with pytest.raises(ProviderError, match="timed out"):
            p.complete([Message(role="user", content="Hi")])

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_auth_error(self, mock_sdk):
        _APIError = type("APIError", (Exception,), {})
        _APITimeoutError = type("APITimeoutError", (_APIError,), {})
        _AuthenticationError = type("AuthenticationError", (_APIError,), {})
        mock_sdk.APIError = _APIError
        mock_sdk.APITimeoutError = _APITimeoutError
        mock_sdk.AuthenticationError = _AuthenticationError
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _AuthenticationError("bad")
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        with pytest.raises(ProviderError, match="authentication failed"):
            p.complete([Message(role="user", content="Hi")])

    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", False)
    def test_sdk_not_installed(self):
        p = OpenAIProvider(_make_config())
        with pytest.raises(ProviderError, match="not installed"):
            p.complete([Message(role="user", content="Hi")])

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_empty_choices(self, mock_sdk):
        resp = SimpleNamespace(
            choices=[],
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=0, total_tokens=5),
            model_dump=dict,
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        result = p.complete([Message(role="user", content="Hi")])
        assert result.content == ""

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_base_url_passed_to_client(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config(base_url="http://local:8080"))
        p.complete([Message(role="user", content="Hi")])
        call_kwargs = mock_sdk.OpenAI.call_args[1]
        assert call_kwargs["base_url"] == "http://local:8080"

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_auto_model_uses_best_available_model(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(id="gpt-4o"),
                SimpleNamespace(id="gpt-5.4-mini"),
                SimpleNamespace(id="gpt-5.5"),
            ]
        )
        mock_client.chat.completions.create.return_value = self._mock_response(model="gpt-5.5")
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config(model="auto"))
        p.complete([Message(role="user", content="Hi")])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-5.5"

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_gpt5_omits_custom_temperature_and_maps_max_tokens(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(model="gpt-5.5")
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config(model="gpt-5.5"))
        p.complete([Message(role="user", content="Hi")], temperature=0.2, max_tokens=7)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "temperature" not in call_kwargs
        assert call_kwargs["max_completion_tokens"] == 7
        assert "max_tokens" not in call_kwargs

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_native_openai_uses_responses_api_when_available(self, mock_sdk):
        response_create = MagicMock(
            return_value=SimpleNamespace(
                output_text="Hello from Responses",
                output=[],
                model="gpt-5.5",
                usage=SimpleNamespace(input_tokens=11, output_tokens=4, total_tokens=15),
                model_dump=lambda: {"id": "resp_123"},
            )
        )
        mock_client = SimpleNamespace(
            responses=SimpleNamespace(create=response_create),
            chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock())),
        )
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config(model="gpt-5.5"))
        result = p.complete(
            [
                Message(role="system", content="Be brief."),
                Message(role="user", content="Hi"),
            ],
            max_tokens=9,
        )

        assert result.content == "Hello from Responses"
        assert result.usage == {
            "prompt_tokens": 11,
            "completion_tokens": 4,
            "total_tokens": 15,
        }
        response_create.assert_called_once()
        call_kwargs = response_create.call_args[1]
        assert call_kwargs == {
            "model": "gpt-5.5",
            "input": [{"role": "user", "content": "Hi"}],
            "instructions": "Be brief.",
            "max_output_tokens": 9,
        }
        mock_client.chat.completions.create.assert_not_called()

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_responses_api_extracts_text_from_output_parts(self, mock_sdk):
        response_create = MagicMock(
            return_value=SimpleNamespace(
                output_text="",
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[
                            SimpleNamespace(type="output_text", text="Hello"),
                            {"type": "output_text", "text": " world"},
                        ],
                    )
                ],
                model="gpt-5.5",
                usage=SimpleNamespace(input_tokens=3, output_tokens=2),
                model_dump=lambda: {"id": "resp_456"},
            )
        )
        mock_sdk.OpenAI.return_value = SimpleNamespace(
            responses=SimpleNamespace(create=response_create),
            chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock())),
        )

        p = OpenAIProvider(_make_config(model="gpt-5.5"))
        result = p.complete([Message(role="user", content="Hi")])

        assert result.content == "Hello world"
        assert result.usage["total_tokens"] == 5

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_responses_api_converts_vision_parts(self, mock_sdk):
        response_create = MagicMock(
            return_value=SimpleNamespace(
                output_text="vision ok",
                output=[],
                model="gpt-5.5",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
                model_dump=dict,
            )
        )
        mock_sdk.OpenAI.return_value = SimpleNamespace(
            responses=SimpleNamespace(create=response_create),
            chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock())),
        )

        p = OpenAIProvider(_make_config(model="gpt-5.5"))
        p.complete(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/image.png",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ]
        )

        call_kwargs = response_create.call_args[1]
        assert call_kwargs["input"] == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe"},
                    {
                        "type": "input_image",
                        "image_url": "https://example.com/image.png",
                        "detail": "low",
                    },
                ],
            }
        ]

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_base_url_keeps_chat_completions_fallback(self, mock_sdk):
        response_create = MagicMock()
        mock_client = SimpleNamespace(
            responses=SimpleNamespace(create=response_create),
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=MagicMock(return_value=self._mock_response()))
            ),
        )
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config(base_url="https://api.groq.com/openai/v1"))
        result = p.complete([Message(role="user", content="Hi")])

        assert result.content == "Hello!"
        response_create.assert_not_called()
        mock_client.chat.completions.create.assert_called_once()

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_responses_api_uses_native_structured_output_format(self, mock_sdk):
        response_create = MagicMock(
            return_value=SimpleNamespace(
                output_text='{"answer":"yes","confidence":0.9}',
                output=[],
                model="gpt-5.5",
                usage=SimpleNamespace(input_tokens=3, output_tokens=2, total_tokens=5),
                model_dump=dict,
            )
        )
        mock_sdk.OpenAI.return_value = SimpleNamespace(
            responses=SimpleNamespace(create=response_create),
            chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock())),
        )

        p = OpenAIProvider(_make_config(model="gpt-5.5"))
        schema = OutputSchema(StructuredAnswer, strict=True)
        p.complete(
            [Message(role="user", content="answer")],
            **p.structured_output_kwargs(schema),
        )

        call_kwargs = response_create.call_args[1]
        assert call_kwargs["text"]["format"]["type"] == "json_schema"
        assert call_kwargs["text"]["format"]["name"] == "StructuredAnswer"
        assert call_kwargs["text"]["format"]["strict"] is True
        assert call_kwargs["text"]["format"]["schema"]["properties"]["answer"]["type"] == "string"

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_chat_fallback_uses_native_structured_response_format(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(
            text='{"answer":"yes","confidence":0.9}'
        )
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config(base_url="https://api.groq.com/openai/v1"))
        schema = OutputSchema(StructuredAnswer)
        p.complete(
            [Message(role="user", content="answer")],
            **p.structured_output_kwargs(schema),
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["name"] == "StructuredAnswer"
        assert "text" not in call_kwargs


class TestOpenAIToolSchema:
    def test_function_format(self):
        p = OpenAIProvider(_make_config())
        tool = MagicMock()
        tool.name = "calc"
        tool.description = "Calculator"
        tool.get_schema.return_value = {
            "parameters": {
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            }
        }
        schemas = p.get_tool_schema([tool])
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "calc"


class TestOpenAIStream:
    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_native_openai_stream_uses_responses_events(self, mock_sdk):
        response_stream = MagicMock(
            return_value=iter(
                [
                    {"type": "response.output_text.delta", "delta": "Hello"},
                    {"type": "response.output_text.delta", "delta": " world"},
                ]
            )
        )
        mock_client = SimpleNamespace(
            responses=SimpleNamespace(create=MagicMock(), stream=response_stream),
            chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock())),
        )
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config(model="gpt-5.5"))
        chunks = list(
            p.stream(
                [
                    Message(role="system", content="Be brief."),
                    Message(role="user", content="Hi"),
                ],
                system="Runtime system.",
            )
        )

        assert chunks == ["Hello", " world"]
        response_stream.assert_called_once()
        call_kwargs = response_stream.call_args[1]
        assert call_kwargs == {
            "model": "gpt-5.5",
            "input": [{"role": "user", "content": "Hi"}],
            "instructions": "Runtime system.\n\nBe brief.",
        }
        mock_client.chat.completions.create.assert_not_called()

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_responses_stream_reconciles_full_text_snapshots(self, mock_sdk):
        response_stream = MagicMock(
            return_value=iter(
                [
                    {"type": "response.output_text.delta", "delta": "Hel"},
                    {"type": "response.output_text.done", "text": "Hello"},
                    {
                        "type": "response.completed",
                        "response": SimpleNamespace(output_text="Hello world"),
                    },
                ]
            )
        )
        mock_sdk.OpenAI.return_value = SimpleNamespace(
            responses=SimpleNamespace(create=MagicMock(), stream=response_stream),
            chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock())),
        )

        p = OpenAIProvider(_make_config(model="gpt-5.5"))
        chunks = list(p.stream([Message(role="user", content="Hi")]))

        assert chunks == ["Hel", "lo", " world"]

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_responses_stream_failed_event_raises_provider_error(self, mock_sdk):
        response_stream = MagicMock(
            return_value=iter(
                [
                    {
                        "type": "response.failed",
                        "error": {"message": "quota exceeded"},
                    }
                ]
            )
        )
        mock_sdk.OpenAI.return_value = SimpleNamespace(
            responses=SimpleNamespace(create=MagicMock(), stream=response_stream),
            chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock())),
        )

        p = OpenAIProvider(_make_config(model="gpt-5.5"))
        with pytest.raises(ProviderError, match="quota exceeded"):
            list(p.stream([Message(role="user", content="Hi")]))

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_base_url_stream_keeps_chat_completions_fallback(self, mock_sdk):
        response_stream = MagicMock()
        chat_stream = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="chat"))],
            )
        ]
        mock_client = SimpleNamespace(
            responses=SimpleNamespace(create=MagicMock(), stream=response_stream),
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=MagicMock(return_value=chat_stream))
            ),
        )
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config(model="gpt-4o", base_url="https://api.groq.com/openai/v1"))
        chunks = list(p.stream([Message(role="user", content="Hi")]))

        assert chunks == ["chat"]
        response_stream.assert_not_called()
        mock_client.chat.completions.create.assert_called_once()


class TestOpenAIMessagePayload:
    def setup_method(self):
        event_bus.clear()

    def teardown_method(self):
        event_bus.clear()

    def test_runtime_tool_messages_are_converted_to_openai_shape(self):
        p = OpenAIProvider(_make_config())
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "use a tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "name": "calc", "arguments": {"expr": "1+1"}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "calc",
                "content": "2",
                "is_error": False,
            },
        ]
        payload = p._messages_to_chat_payload(messages)
        assert payload[2]["tool_calls"][0] == {
            "id": "call_1",
            "type": "function",
            "function": {"name": "calc", "arguments": '{"expr": "1+1"}'},
        }
        assert payload[3] == {"role": "tool", "tool_call_id": "call_1", "content": "2"}

    def test_vision_message_preserves_safe_openai_image_parts(self):
        p = OpenAIProvider(_make_config())
        payload = p._messages_to_chat_payload(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,abc123",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": "describe"},
                    ],
                }
            ]
        )
        assert payload == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,abc123",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": "describe"},
                ],
            }
        ]

    def test_unsafe_image_url_is_removed_from_openai_payload(self):
        p = OpenAIProvider(_make_config())
        payload = p._messages_to_chat_payload(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "file:///etc/passwd"},
                        },
                        {"type": "text", "text": "describe"},
                    ],
                }
            ]
        )
        assert payload == [{"role": "user", "content": [{"type": "text", "text": "describe"}]}]
        assert p._last_transcript_repairs == [{"reason": "drop_unsafe_image_url"}]

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_orphan_tool_result_is_repaired_and_audited(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="repaired", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model_dump=dict,
        )
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        p.complete(
            [
                {"role": "user", "content": "hi"},
                {"role": "tool", "tool_call_id": "missing", "content": "orphan"},
            ],
            session_id="sess-openai",
            task_id="task-openai",
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]
        events = event_bus.get_events(event_type="provider_transcript_repair")
        assert len(events) == 1
        assert events[0].session_id == "sess-openai"
        assert events[0].detail["repairs"] == [
            {"reason": "drop_orphan_tool_result", "tool_call_id": "missing"}
        ]


class TestOpenAICompleteWithTools:
    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_native_dict_tool_messages_do_not_break_token_estimate(self, mock_sdk):
        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            model_dump=dict,
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        messages = [
            {"role": "user", "content": "use a tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "name": "calc", "arguments": {}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "2"},
        ]

        result = p.complete_with_tools(messages, [])

        assert result.content == "done"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][2] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "2",
        }
        assert "tools" not in call_kwargs

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_tool_call_parsing(self, mock_sdk):
        tc = SimpleNamespace(
            id="tc_1",
            function=SimpleNamespace(name="calc", arguments='{"expr": "1+1"}'),
        )
        choice = SimpleNamespace(
            message=SimpleNamespace(content="Calculating", tool_calls=[tc]),
            finish_reason="tool_calls",
        )
        resp = SimpleNamespace(
            choices=[choice],
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            model_dump=dict,
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        tool = MagicMock()
        tool.name = "calc"
        tool.description = "Calculator"
        tool.get_schema.return_value = {"parameters": {"properties": {}, "required": []}}

        result = p.complete_with_tools([Message(role="user", content="calc")], [tool])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "calc"
        assert result.tool_calls[0].arguments == {"expr": "1+1"}

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_malformed_tool_args(self, mock_sdk):
        tc = SimpleNamespace(
            id="tc_1",
            function=SimpleNamespace(name="calc", arguments="not json"),
        )
        choice = SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=[tc]),
            finish_reason="tool_calls",
        )
        resp = SimpleNamespace(
            choices=[choice],
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            model_dump=dict,
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        result = p.complete_with_tools(
            [Message(role="user", content="x")],
            [
                MagicMock(
                    name="calc",
                    description="c",
                    get_schema=MagicMock(
                        return_value={"parameters": {"properties": {}, "required": []}}
                    ),
                )
            ],
        )
        # Malformed args should be empty dict
        assert result.tool_calls[0].arguments == {}

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_system_prompt_injection(self, mock_sdk):
        mock_client = MagicMock()
        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            model_dump=dict,
        )
        mock_client.chat.completions.create.return_value = resp
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        p.complete_with_tools(
            [Message(role="user", content="hi")],
            [],
            system="Be helpful",
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][0]["content"] == "Be helpful"

    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", False)
    def test_sdk_not_installed(self):
        p = OpenAIProvider(_make_config())
        with pytest.raises(ProviderError, match="not installed"):
            p.complete_with_tools([Message(role="user", content="Hi")], [])


class TestOpenAIMultiAccountHealthTracking:
    """Per-account failure tracking wiring (round_robin.py's RoundRobinAccounts
    record_success/record_failure) for OpenAIProvider's complete()/
    complete_with_tools()/stream(), mirroring CodexProvider's OAuth-account
    coverage."""

    def _multi_config(self) -> ProviderConfig:
        return _make_config(
            api_key=None,
            api_keys=["k1", "k2"],
            key_rotation_strategy="round_robin",
        )

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_successful_complete_records_success(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = TestOpenAIComplete()._mock_response()
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(self._multi_config())
        with patch.object(p._rr, "record_success") as mock_record:
            p.complete([Message(role="user", content="hi")])

        mock_record.assert_called_once()
        assert mock_record.call_args[0][0].index == 0

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_failed_complete_records_failure(self, mock_sdk):
        _APIError = type("APIError", (Exception,), {})
        _APITimeoutError = type("APITimeoutError", (_APIError,), {})
        _AuthenticationError = type("AuthenticationError", (_APIError,), {})
        mock_sdk.APIError = _APIError
        mock_sdk.APITimeoutError = _APITimeoutError
        mock_sdk.AuthenticationError = _AuthenticationError
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _AuthenticationError("bad")
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(self._multi_config())
        with (
            patch.object(p._rr, "record_failure") as mock_record,
            pytest.raises(ProviderError),
        ):
            p.complete([Message(role="user", content="hi")])

        mock_record.assert_called_once()
        assert mock_record.call_args[0][0].index == 0

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_failed_complete_with_tools_records_failure(self, mock_sdk):
        _APIError = type("APIError", (Exception,), {})
        _APITimeoutError = type("APITimeoutError", (_APIError,), {})
        _AuthenticationError = type("AuthenticationError", (_APIError,), {})
        mock_sdk.APIError = _APIError
        mock_sdk.APITimeoutError = _APITimeoutError
        mock_sdk.AuthenticationError = _AuthenticationError
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _APITimeoutError("slow")
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(self._multi_config())
        with (
            patch.object(p._rr, "record_failure") as mock_record,
            pytest.raises(ProviderError),
        ):
            p.complete_with_tools([Message(role="user", content="hi")], [])

        mock_record.assert_called_once()

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_successful_stream_records_success(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(
            [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))])]
        )
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(self._multi_config())
        with patch.object(p._rr, "record_success") as mock_record:
            list(p.stream([Message(role="user", content="hi")]))

        mock_record.assert_called_once()

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_failed_stream_records_failure(self, mock_sdk):
        _APIError = type("APIError", (Exception,), {})
        _APITimeoutError = type("APITimeoutError", (_APIError,), {})
        _AuthenticationError = type("AuthenticationError", (_APIError,), {})
        mock_sdk.APIError = _APIError
        mock_sdk.APITimeoutError = _APITimeoutError
        mock_sdk.AuthenticationError = _AuthenticationError

        def _boom(**_kwargs):
            raise _APITimeoutError("slow")
            yield  # pragma: no cover - make this a generator function

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _boom
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(self._multi_config())
        with (
            patch.object(p._rr, "record_failure") as mock_record,
            pytest.raises(ProviderError),
        ):
            list(p.stream([Message(role="user", content="hi")]))

        mock_record.assert_called_once()

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_single_account_config_never_touches_round_robin_tracking(self, mock_sdk):
        """No account selected -> _record_account_outcome is a no-op, so a
        single-key config's failures/successes never reach record_success/
        record_failure (nothing there to track)."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = TestOpenAIComplete()._mock_response()
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())  # single api_key, no round_robin
        with (
            patch.object(p._rr, "record_success") as mock_success,
            patch.object(p._rr, "record_failure") as mock_failure,
        ):
            p.complete([Message(role="user", content="hi")])

        mock_success.assert_not_called()
        mock_failure.assert_not_called()

    def test_account_skipped_after_repeated_failures(self):
        p = OpenAIProvider(self._multi_config())
        first = p._rr._live_accounts[0]
        for _ in range(5):  # default failure_threshold
            p._rr.record_failure(first)

        picks = [p._select_account().index for _ in range(4)]
        assert picks == [1, 1, 1, 1]
