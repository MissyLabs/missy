"""OpenAI-compatible provider for the Missy framework.

Uses the ``openai`` SDK to call the Chat Completions API.  The ``base_url``
parameter allows this provider to target any OpenAI-compatible endpoint
(e.g. Groq, Together AI, a local vLLM instance).

The SDK is imported lazily so that Missy can start without it installed -
:meth:`is_available` returns ``False`` in that case.

Example::

    from missy.config.settings import ProviderConfig
    from missy.providers.openai_provider import OpenAIProvider

    config = ProviderConfig(name="openai", model="auto", api_key="<REDACTED>")
    provider = OpenAIProvider(config)
    response = provider.complete([Message(role="user", content="Hello")])
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from typing import Any

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError

from .base import BaseProvider, CompletionResponse, Message, ToolCall

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "auto"
_FALLBACK_MODEL = "gpt-5.5"
_AUTO_MODEL_SENTINELS = {"", "auto", "latest", "best"}
_PREFERRED_CHAT_MODELS = (
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
)
_NON_CHAT_MODEL_MARKERS = (
    "audio",
    "dall",
    "embedding",
    "image",
    "moderation",
    "realtime",
    "sora",
    "tts",
    "transcribe",
    "whisper",
)

try:
    import openai as _openai_sdk

    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _openai_sdk = None  # type: ignore[assignment]
    _OPENAI_AVAILABLE = False


class OpenAIProvider(BaseProvider):
    """Provider implementation backed by the OpenAI Chat Completions API.

    Args:
        config: Provider-level configuration.  ``api_key`` is forwarded to
            the SDK; when ``None`` the SDK reads ``OPENAI_API_KEY`` from the
            environment.  ``base_url`` overrides the default API endpoint,
            enabling OpenAI-compatible third-party services.
    """

    name = "openai"
    accepts_message_dicts = True

    def __init__(self, config: ProviderConfig) -> None:
        self._api_key: str | None = config.api_key
        self._base_url: str | None = config.base_url
        self._model: str = config.model or _DEFAULT_MODEL
        self._timeout: int = config.timeout
        self._client: Any | None = None
        self._resolved_model: str | None = None

    @property
    def api_key(self) -> str | None:
        """Return the active API key, if one is configured directly."""
        return self._api_key

    @api_key.setter
    def api_key(self, value: str | None) -> None:
        """Update the API key and force the SDK client to be rebuilt."""
        self._api_key = value
        self._client = None
        self._resolved_model = None

    @property
    def model(self) -> str:
        """Return the configured model selector."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Update the model selector and clear any cached auto resolution."""
        self._model = value or _DEFAULT_MODEL
        self._resolved_model = None

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return ``True`` when the SDK is installed and an API key is set.

        Returns:
            ``True`` if the ``openai`` package is importable and an API key
            is present.
        """
        return _OPENAI_AVAILABLE and bool(self._api_key or os.environ.get("OPENAI_API_KEY"))

    def _make_client(self) -> Any:
        """Return a cached OpenAI client, creating one on first call.

        The SDK is given a policy-aware ``http_client`` so that all provider
        egress transits the network policy check (consistent with the
        gateway), rather than issuing unchecked HTTP directly.
        """
        if self._client is None:
            client_kwargs: dict[str, Any] = {"timeout": float(self._timeout)}
            if self._api_key:
                client_kwargs["api_key"] = self._api_key
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            try:
                from missy.providers.policy_http import build_policy_http_client

                client_kwargs["http_client"] = build_policy_http_client(
                    timeout=float(self._timeout)
                )
            except Exception:  # pragma: no cover - defensive; never block startup
                logger.debug("Could not build policy-aware http client", exc_info=True)
            self._client = _openai_sdk.OpenAI(**client_kwargs)
        return self._client

    def _resolve_model(self, requested_model: str | None = None) -> str:
        """Resolve ``auto`` / ``latest`` to the best available chat model.

        The OpenAI models endpoint is account-aware, so this prefers the
        newest known chat models only when the active credentials can see
        them. If the endpoint is unavailable, fall back to the current
        recommended frontier model instead of a legacy GPT-4-era default.
        """
        model = (requested_model or self._model or _DEFAULT_MODEL).strip()
        if model.lower() not in _AUTO_MODEL_SENTINELS:
            return model
        if self._resolved_model:
            return self._resolved_model

        try:
            response = self._make_client().models.list()
            raw_models = getattr(response, "data", response)
            model_ids: set[str] = set()
            for item in raw_models:
                model_id = getattr(item, "id", None)
                if model_id is None and isinstance(item, dict):
                    model_id = item.get("id")
                if model_id:
                    model_ids.add(str(model_id))

            for preferred in _PREFERRED_CHAT_MODELS:
                if preferred in model_ids:
                    self._resolved_model = preferred
                    return preferred

            chat_like = sorted(
                mid
                for mid in model_ids
                if mid.startswith(("gpt-", "chatgpt-"))
                and not any(marker in mid.lower() for marker in _NON_CHAT_MODEL_MARKERS)
            )
            if chat_like:
                self._resolved_model = chat_like[-1]
                return chat_like[-1]
        except Exception:
            logger.debug("OpenAI model auto-detection failed; using fallback", exc_info=True)

        self._resolved_model = _FALLBACK_MODEL
        return _FALLBACK_MODEL

    @staticmethod
    def _supports_custom_temperature(model: str) -> bool:
        """Return whether custom temperature should be sent for *model*."""
        model_lower = model.lower()
        return not model_lower.startswith(("gpt-5", "o1", "o3", "o4"))

    def _apply_common_generation_kwargs(
        self,
        call_kwargs: dict[str, Any],
        kwargs: dict[str, Any],
        model: str,
    ) -> None:
        """Normalize common generation kwargs for current OpenAI chat models."""
        if "temperature" in kwargs:
            temperature = kwargs.pop("temperature")
            supports_temperature = self._supports_custom_temperature(model)
            try:
                is_default_temperature = float(temperature) == 1.0
            except (TypeError, ValueError):
                is_default_temperature = False
            if temperature is not None and not supports_temperature and not is_default_temperature:
                logger.debug("Omitting unsupported temperature override for OpenAI model %r", model)
            elif temperature is not None:
                call_kwargs["temperature"] = temperature

        if "max_completion_tokens" in kwargs:
            call_kwargs["max_completion_tokens"] = kwargs.pop("max_completion_tokens")
        elif "max_tokens" in kwargs:
            call_kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

        call_kwargs.update(kwargs)

    @staticmethod
    def _message_to_chat_payload(message: Message | dict[str, Any]) -> dict[str, Any] | None:
        """Convert Missy's message shape to OpenAI Chat Completions format."""
        if isinstance(message, Message):
            return {"role": message.role, "content": message.content}

        role = str(message.get("role", ""))
        if role in {"system", "user"}:
            return {"role": role, "content": str(message.get("content", ""))}

        if role == "assistant":
            payload: dict[str, Any] = {
                "role": "assistant",
                "content": str(message.get("content", "") or ""),
            }
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                payload["tool_calls"] = []
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    if "function" in call:
                        payload["tool_calls"].append(call)
                        continue
                    arguments = call.get("arguments", {})
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments)
                    payload["tool_calls"].append(
                        {
                            "id": str(call.get("id", "")),
                            "type": "function",
                            "function": {
                                "name": str(call.get("name", "")),
                                "arguments": arguments,
                            },
                        }
                    )
            return payload

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not tool_call_id:
                return None
            return {
                "role": "tool",
                "tool_call_id": str(tool_call_id),
                "content": str(message.get("content", "")),
            }

        return None

    def _messages_to_chat_payload(
        self,
        messages: list[Message] | list[dict[str, Any]],
        system: str = "",
    ) -> list[dict[str, Any]]:
        """Build a Chat Completions-compatible message list."""
        api_messages: list[dict[str, Any]] = []
        has_system = any(
            (msg.role if isinstance(msg, Message) else msg.get("role")) == "system"
            for msg in messages
        )
        if system and not has_system:
            api_messages.append({"role": "system", "content": system})
        for msg in messages:
            payload = self._message_to_chat_payload(msg)
            if payload is not None:
                api_messages.append(payload)
        return api_messages

    def complete(self, messages: list[Message], **kwargs: Any) -> CompletionResponse:
        """Send *messages* to the OpenAI Chat Completions API.

        Args:
            messages: Ordered conversation turns.  All role values supported
                by the API (``"system"``, ``"user"``, ``"assistant"``) are
                forwarded as-is.
            **kwargs: Optional overrides.  Recognised keys:

                * ``temperature`` (float) - sampling temperature.
                * ``max_tokens`` (int) - maximum completion tokens.
                * ``model`` (str) - override the configured model.

        Returns:
            A :class:`CompletionResponse` with the assistant reply.

        Raises:
            ProviderError: On SDK import failure, authentication error,
                API error, or timeout.
        """
        if not _OPENAI_AVAILABLE:
            raise ProviderError("openai SDK is not installed. Run: pip install openai")

        session_id = kwargs.pop("session_id", "")
        task_id = kwargs.pop("task_id", "")
        model = self._resolve_model(kwargs.pop("model", self._model))
        api_messages = self._messages_to_chat_payload(messages)

        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
        }
        self._apply_common_generation_kwargs(call_kwargs, kwargs, model)

        self._acquire_rate_limit(estimated_tokens=self._estimate_tokens(messages))

        try:
            client = self._make_client()
            raw_response = client.chat.completions.create(**call_kwargs)
        except _openai_sdk.APITimeoutError as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(f"OpenAI request timed out after {self._timeout}s: {exc}") from exc
        except _openai_sdk.AuthenticationError as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(f"OpenAI authentication failed: {exc}") from exc
        except _openai_sdk.APIError as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            if getattr(exc, "status_code", 0) == 429:
                retry_after = float(
                    getattr(getattr(exc, "response", None), "headers", {}).get("retry-after", 5)
                )
                if self.rate_limiter is not None:
                    self.rate_limiter.on_rate_limit_response(retry_after)
                raise ProviderError(f"OpenAI rate limited: {exc}") from exc
            raise ProviderError(f"OpenAI API error: {exc}") from exc
        except Exception as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(f"Unexpected error calling OpenAI: {exc}") from exc

        choice = raw_response.choices[0] if raw_response.choices else None
        content_text = choice.message.content if choice else ""
        usage_obj = raw_response.usage
        usage = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
            "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
            "total_tokens": getattr(usage_obj, "total_tokens", 0),
        }

        self._emit_event(session_id, task_id, "allow", "completion successful")

        response = CompletionResponse(
            content=content_text or "",
            model=raw_response.model,
            provider=self.name,
            usage=usage,
            raw=raw_response.model_dump() if hasattr(raw_response, "model_dump") else {},
        )
        self._record_rate_limit_usage(response)
        return response

    def get_tool_schema(self, tools: list) -> list:
        """Convert BaseTool instances to OpenAI function-calling schema format.

        Args:
            tools: List of :class:`~missy.tools.base.BaseTool` instances.

        Returns:
            A list of OpenAI-format tool dicts with ``type`` and ``function``
            keys.
        """
        schemas = []
        for tool in tools:
            base_schema = tool.get_schema() if hasattr(tool, "get_schema") else {}
            params = base_schema.get("parameters", {})
            function_schema: dict[str, Any] = {
                "type": "object",
                "properties": params.get("properties", {}),
                "required": params.get("required", []),
            }
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": function_schema,
                    },
                }
            )
        return schemas

    def complete_with_tools(
        self,
        messages: list[Message],
        tools: list,
        system: str = "",
    ) -> CompletionResponse:
        """Send *messages* to the OpenAI API with tool calling enabled.

        Args:
            messages: Ordered conversation turns.
            tools: List of :class:`~missy.tools.base.BaseTool` instances.
            system: Optional system prompt string.

        Returns:
            A :class:`CompletionResponse`.  When ``finish_reason`` is
            ``"tool_calls"``, ``tool_calls`` is populated with
            :class:`~missy.providers.base.ToolCall` instances.

        Raises:
            ProviderError: On SDK import failure or API error.
        """
        if not _OPENAI_AVAILABLE:
            raise ProviderError("openai SDK is not installed. Run: pip install openai")

        tool_schemas = self.get_tool_schema(tools)

        # Build message list; inject system prompt if provided
        model = self._resolve_model(self._model)
        api_messages = self._messages_to_chat_payload(messages, system=system)

        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
        }
        if tool_schemas:
            call_kwargs["tools"] = tool_schemas
            call_kwargs["tool_choice"] = "auto"

        self._acquire_rate_limit(estimated_tokens=self._estimate_tokens(messages, system))

        try:
            client = self._make_client()
            raw_response = client.chat.completions.create(**call_kwargs)
        except _openai_sdk.APITimeoutError as exc:
            raise ProviderError(f"OpenAI request timed out after {self._timeout}s: {exc}") from exc
        except _openai_sdk.AuthenticationError as exc:
            raise ProviderError(f"OpenAI authentication failed: {exc}") from exc
        except _openai_sdk.APIError as exc:
            if getattr(exc, "status_code", 0) == 429:
                retry_after = float(
                    getattr(getattr(exc, "response", None), "headers", {}).get("retry-after", 5)
                )
                if self.rate_limiter is not None:
                    self.rate_limiter.on_rate_limit_response(retry_after)
                raise ProviderError(f"OpenAI rate limited: {exc}") from exc
            raise ProviderError(f"OpenAI API error: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"Unexpected error calling OpenAI: {exc}") from exc

        choice = raw_response.choices[0] if raw_response.choices else None
        content_text = ""
        tool_calls: list[ToolCall] = []
        finish_reason = "stop"

        if choice:
            content_text = choice.message.content or ""
            raw_finish = choice.finish_reason or "stop"
            finish_reason = raw_finish if raw_finish in ("stop", "tool_calls", "length") else "stop"

            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, Exception):
                        arguments = {}
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=arguments,
                        )
                    )

        usage_obj = raw_response.usage
        usage = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
            "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
            "total_tokens": getattr(usage_obj, "total_tokens", 0),
        }

        response = CompletionResponse(
            content=content_text,
            model=raw_response.model,
            provider=self.name,
            usage=usage,
            raw=raw_response.model_dump() if hasattr(raw_response, "model_dump") else {},
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
        self._record_rate_limit_usage(response)
        return response

    def stream(self, messages: list[Message], system: str = "") -> Iterator[str]:
        """Stream partial response tokens from the OpenAI API.

        Args:
            messages: Ordered conversation turns.
            system: Optional system prompt string.

        Yields:
            String text delta chunks as they arrive from the API.

        Raises:
            ProviderError: On SDK import failure or API error.
        """
        if not _OPENAI_AVAILABLE:
            raise ProviderError("openai SDK is not installed. Run: pip install openai")

        model = self._resolve_model(self._model)
        api_messages = self._messages_to_chat_payload(messages, system=system)

        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "stream": True,
        }

        try:
            client = self._make_client()
            for chunk in client.chat.completions.create(**call_kwargs):
                delta = chunk.choices[0].delta.content if chunk.choices else None
                yield delta or ""
        except _openai_sdk.APITimeoutError as exc:
            raise ProviderError(f"OpenAI stream timed out after {self._timeout}s: {exc}") from exc
        except _openai_sdk.AuthenticationError as exc:
            raise ProviderError(f"OpenAI authentication failed: {exc}") from exc
        except _openai_sdk.APIError as exc:
            raise ProviderError(f"OpenAI API error during stream: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"Unexpected error streaming from OpenAI: {exc}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit_event(
        self,
        session_id: str,
        task_id: str,
        result: str,
        detail_msg: str,
    ) -> None:
        """Publish a provider audit event including the model name."""
        try:
            from missy.core.events import AuditEvent, event_bus

            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type="provider_invoke",
                category="provider",
                result=result,  # type: ignore[arg-type]
                detail={"provider": self.name, "model": self._model, "message": detail_msg},
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit audit event for provider %r", self.name)
