"""OpenAI-compatible provider for the Missy framework.

Uses the ``openai`` SDK to call the Chat Completions API.  The ``base_url``
parameter allows this provider to target any OpenAI-compatible endpoint
(e.g. Groq, Together AI, a local vLLM instance).

The SDK is imported lazily so that Missy can start without it installed -
:meth:`is_available` returns ``False`` in that case.

Example::

    from missy.config.settings import ProviderConfig
    from missy.providers.openai_provider import OpenAIProvider

    config = ProviderConfig(name="openai", model="gpt-4o", api_key="sk-...")
    provider = OpenAIProvider(config)
    response = provider.complete([Message(role="user", content="Hello")])
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from missy.config.settings import ProviderConfig
from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import ProviderError

from .base import BaseProvider, CompletionResponse, Message, ToolCall

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o"

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

    def __init__(self, config: ProviderConfig) -> None:
        self._api_key: str | None = config.api_key
        self._base_url: str | None = config.base_url
        self._model: str = config.model or _DEFAULT_MODEL
        self._timeout: int = config.timeout

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return ``True`` when the SDK is installed and an API key is set.

        Returns:
            ``True`` if the ``openai`` package is importable and an API key
            is present.
        """
        return _OPENAI_AVAILABLE and bool(self._api_key)

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
        model = kwargs.pop("model", self._model)

        api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
        }
        if "temperature" in kwargs:
            call_kwargs["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            call_kwargs["max_tokens"] = kwargs.pop("max_tokens")
        call_kwargs.update(kwargs)

        try:
            client_kwargs: dict[str, Any] = {"timeout": float(self._timeout)}
            if self._api_key:
                client_kwargs["api_key"] = self._api_key
            if self._base_url:
                client_kwargs["base_url"] = self._base_url

            client = _openai_sdk.OpenAI(**client_kwargs)
            raw_response = client.chat.completions.create(**call_kwargs)
        except _openai_sdk.APITimeoutError as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(f"OpenAI request timed out after {self._timeout}s: {exc}") from exc
        except _openai_sdk.AuthenticationError as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(f"OpenAI authentication failed: {exc}") from exc
        except _openai_sdk.APIError as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
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

        return CompletionResponse(
            content=content_text or "",
            model=raw_response.model,
            provider=self.name,
            usage=usage,
            raw=raw_response.model_dump() if hasattr(raw_response, "model_dump") else {},
        )

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
        api_messages: list[dict] = []
        has_system = any(m.role == "system" for m in messages)
        if system and not has_system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "tools": tool_schemas,
            "tool_choice": "auto",
        }

        try:
            client_kwargs: dict[str, Any] = {"timeout": float(self._timeout)}
            if self._api_key:
                client_kwargs["api_key"] = self._api_key
            if self._base_url:
                client_kwargs["base_url"] = self._base_url

            client = _openai_sdk.OpenAI(**client_kwargs)
            raw_response = client.chat.completions.create(**call_kwargs)
        except _openai_sdk.APITimeoutError as exc:
            raise ProviderError(f"OpenAI request timed out after {self._timeout}s: {exc}") from exc
        except _openai_sdk.AuthenticationError as exc:
            raise ProviderError(f"OpenAI authentication failed: {exc}") from exc
        except _openai_sdk.APIError as exc:
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

        return CompletionResponse(
            content=content_text,
            model=raw_response.model,
            provider=self.name,
            usage=usage,
            raw=raw_response.model_dump() if hasattr(raw_response, "model_dump") else {},
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

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

        api_messages: list[dict] = []
        has_system = any(m.role == "system" for m in messages)
        if system and not has_system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "stream": True,
        }

        try:
            client_kwargs: dict[str, Any] = {"timeout": float(self._timeout)}
            if self._api_key:
                client_kwargs["api_key"] = self._api_key
            if self._base_url:
                client_kwargs["base_url"] = self._base_url

            client = _openai_sdk.OpenAI(**client_kwargs)
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
        """Publish a provider audit event to the global event bus.

        Args:
            session_id: Calling session identifier.
            task_id: Calling task identifier.
            result: One of ``"allow"`` or ``"error"``.
            detail_msg: Human-readable description to include in the event.
        """
        try:
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
