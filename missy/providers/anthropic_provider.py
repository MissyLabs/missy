"""Anthropic Claude provider for the Missy framework.

Uses the ``anthropic`` SDK to call the Messages API.  The SDK is imported
lazily so that Missy can start without it installed - :meth:`is_available`
returns ``False`` in that case.

Example::

    from missy.config.settings import ProviderConfig
    from missy.providers.anthropic_provider import AnthropicProvider

    config = ProviderConfig(name="anthropic", model="claude-3-5-sonnet-20241022",
                            api_key="sk-ant-...")
    provider = AnthropicProvider(config)
    response = provider.complete([Message(role="user", content="Hello")])
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from missy.config.settings import ProviderConfig
from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import ProviderError

from .base import BaseProvider, CompletionResponse, Message, ToolCall

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

try:
    import anthropic as _anthropic_sdk

    _ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _anthropic_sdk = None  # type: ignore[assignment]
    _ANTHROPIC_AVAILABLE = False


class AnthropicProvider(BaseProvider):
    """Provider implementation backed by the Anthropic Messages API.

    Args:
        config: Provider-level configuration.  The ``api_key`` field is
            forwarded directly to :class:`anthropic.Anthropic`; when
            ``None`` the SDK will fall back to the ``ANTHROPIC_API_KEY``
            environment variable.
    """

    name = "anthropic"

    def __init__(self, config: ProviderConfig) -> None:
        key = config.api_key
        if key and key.startswith("sk-ant-oat"):
            logger.error(
                "Setup-tokens (sk-ant-oat...) are not supported by the "
                "Anthropic Messages API. Get an API key from "
                "https://console.anthropic.com/settings/keys"
            )
            key = None  # Mark as unavailable.
        self._api_key: str | None = key
        self._model: str = config.model or _DEFAULT_MODEL
        self._timeout: int = config.timeout

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return ``True`` when the SDK is installed and an API key is set.

        Returns:
            ``True`` if the ``anthropic`` package is importable and an API
            key is present.
        """
        return _ANTHROPIC_AVAILABLE and bool(self._api_key)

    def _make_client(self) -> Any:
        """Construct an Anthropic client."""
        return _anthropic_sdk.Anthropic(
            api_key=self._api_key,
            timeout=float(self._timeout),
        )

    def complete(self, messages: list[Message], **kwargs: Any) -> CompletionResponse:
        """Send *messages* to the Anthropic Messages API.

        System messages are extracted from *messages* and passed separately
        via the ``system`` parameter (Anthropic's API requires this).

        Args:
            messages: Ordered conversation turns.  A single ``"system"``
                role message is supported; it is extracted and forwarded as
                the ``system`` parameter.
            **kwargs: Optional overrides.  Recognised keys:

                * ``temperature`` (float) - sampling temperature.
                * ``max_tokens`` (int) - maximum completion tokens
                  (default 4096).
                * ``model`` (str) - override the configured model.

        Returns:
            A :class:`CompletionResponse` with the assistant reply.

        Raises:
            ProviderError: On SDK import failure, authentication error,
                API error, or timeout.
        """
        if not _ANTHROPIC_AVAILABLE:
            raise ProviderError(
                "anthropic SDK is not installed. Run: pip install anthropic"
            )

        session_id = kwargs.pop("session_id", "")
        task_id = kwargs.pop("task_id", "")
        model = kwargs.pop("model", self._model)

        # Separate system prompt from the message list
        system_content: str | None = None
        api_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        call_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "messages": api_messages,
        }
        if system_content is not None:
            call_kwargs["system"] = system_content
        if "temperature" in kwargs:
            call_kwargs["temperature"] = kwargs.pop("temperature")
        # Forward any remaining provider-specific kwargs
        call_kwargs.update(kwargs)

        try:
            client = self._make_client()
            raw_response = client.messages.create(**call_kwargs)
        except _anthropic_sdk.APITimeoutError as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(
                f"Anthropic request timed out after {self._timeout}s: {exc}"
            ) from exc
        except _anthropic_sdk.AuthenticationError as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(f"Anthropic authentication failed: {exc}") from exc
        except _anthropic_sdk.APIError as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(f"Anthropic API error: {exc}") from exc
        except Exception as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(f"Unexpected error calling Anthropic: {exc}") from exc

        content_text = raw_response.content[0].text if raw_response.content else ""
        usage_obj = raw_response.usage
        usage = {
            "prompt_tokens": getattr(usage_obj, "input_tokens", 0),
            "completion_tokens": getattr(usage_obj, "output_tokens", 0),
            "total_tokens": (
                getattr(usage_obj, "input_tokens", 0)
                + getattr(usage_obj, "output_tokens", 0)
            ),
        }

        self._emit_event(session_id, task_id, "allow", "completion successful")

        return CompletionResponse(
            content=content_text,
            model=raw_response.model,
            provider=self.name,
            usage=usage,
            raw=raw_response.model_dump() if hasattr(raw_response, "model_dump") else {},
        )

    def get_tool_schema(self, tools: list) -> list:
        """Convert BaseTool instances to Anthropic tool schema format.

        Args:
            tools: List of :class:`~missy.tools.base.BaseTool` instances.

        Returns:
            A list of Anthropic-format tool dicts with ``name``,
            ``description``, and ``input_schema`` keys.
        """
        schemas = []
        for tool in tools:
            # Use get_schema() if available for richer parameter info
            base_schema = tool.get_schema() if hasattr(tool, "get_schema") else {}
            params = base_schema.get("parameters", {})
            input_schema: dict[str, Any] = {
                "type": "object",
                "properties": params.get("properties", {}),
                "required": params.get("required", []),
            }
            schemas.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": input_schema,
                }
            )
        return schemas

    def complete_with_tools(
        self,
        messages: list[Message],
        tools: list,
        system: str = "",
    ) -> CompletionResponse:
        """Send *messages* to the Anthropic API with tool calling enabled.

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
        if not _ANTHROPIC_AVAILABLE:
            raise ProviderError(
                "anthropic SDK is not installed. Run: pip install anthropic"
            )

        tool_schemas = self.get_tool_schema(tools)

        # Separate system prompt (prefer explicit arg, fall back to message list)
        system_content: str = system
        api_messages: list[dict] = []
        for msg in messages:
            if msg.role == "system":
                if not system_content:
                    system_content = msg.content
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": api_messages,
            "tools": tool_schemas,
            "tool_choice": {"type": "auto"},
        }
        if system_content:
            call_kwargs["system"] = system_content

        try:
            client = self._make_client()
            raw_response = client.messages.create(**call_kwargs)
        except _anthropic_sdk.APITimeoutError as exc:
            raise ProviderError(
                f"Anthropic request timed out after {self._timeout}s: {exc}"
            ) from exc
        except _anthropic_sdk.AuthenticationError as exc:
            raise ProviderError(f"Anthropic authentication failed: {exc}") from exc
        except _anthropic_sdk.APIError as exc:
            raise ProviderError(f"Anthropic API error: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"Unexpected error calling Anthropic: {exc}") from exc

        # Extract text content and tool use blocks
        content_text = ""
        tool_calls: list[ToolCall] = []
        for block in raw_response.content:
            if getattr(block, "type", None) == "text":
                content_text += block.text
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input),
                    )
                )

        finish_reason = (
            "tool_calls" if raw_response.stop_reason == "tool_use" else "stop"
        )

        usage_obj = raw_response.usage
        usage = {
            "prompt_tokens": getattr(usage_obj, "input_tokens", 0),
            "completion_tokens": getattr(usage_obj, "output_tokens", 0),
            "total_tokens": (
                getattr(usage_obj, "input_tokens", 0)
                + getattr(usage_obj, "output_tokens", 0)
            ),
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
        """Stream partial response tokens from the Anthropic API.

        Args:
            messages: Ordered conversation turns.
            system: Optional system prompt string (overrides any system
                message in *messages*).

        Yields:
            String text delta chunks as they arrive from the API.

        Raises:
            ProviderError: On SDK import failure or API error.
        """
        if not _ANTHROPIC_AVAILABLE:
            raise ProviderError(
                "anthropic SDK is not installed. Run: pip install anthropic"
            )

        system_content: str = system
        api_messages: list[dict] = []
        for msg in messages:
            if msg.role == "system":
                if not system_content:
                    system_content = msg.content
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": api_messages,
        }
        if system_content:
            call_kwargs["system"] = system_content

        try:
            client = self._make_client()
            with client.messages.stream(**call_kwargs) as stream:
                for text in stream.text_stream:
                    yield text
        except _anthropic_sdk.APITimeoutError as exc:
            raise ProviderError(
                f"Anthropic stream timed out after {self._timeout}s: {exc}"
            ) from exc
        except _anthropic_sdk.AuthenticationError as exc:
            raise ProviderError(f"Anthropic authentication failed: {exc}") from exc
        except _anthropic_sdk.APIError as exc:
            raise ProviderError(f"Anthropic API error during stream: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"Unexpected error streaming from Anthropic: {exc}") from exc

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
