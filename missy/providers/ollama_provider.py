"""Ollama local-inference provider for the Missy framework.

Communicates with the Ollama REST API via :class:`~missy.gateway.client.PolicyHTTPClient`
so that all outbound requests pass through the active network policy.

The Ollama SDK is intentionally **not** used here; the raw HTTP API is
simple and avoids an extra dependency.  The ``/api/chat`` endpoint is
used with ``stream=false`` so that the full response is returned in a
single JSON payload.

Example::

    from missy.config.settings import ProviderConfig
    from missy.providers.ollama_provider import OllamaProvider

    config = ProviderConfig(name="ollama", model="llama3.2")
    provider = OllamaProvider(config)
    response = provider.complete([Message(role="user", content="Hello")])
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from typing import Any

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError
from missy.gateway.client import PolicyHTTPClient

from .base import BaseProvider, CompletionResponse, Message, ToolCall

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.2"

_IMAGE_BLOCK_TYPES = frozenset({"image_url", "input_image"})
_TEXT_BLOCK_TYPES = frozenset({"text", "input_text"})


def _data_uri_to_base64(url: str) -> str:
    """Extract the raw base64 payload from a ``data:...;base64,...`` URI.

    Ollama's ``images`` field wants a bare base64 string, not a data URI
    (unlike OpenAI/Anthropic's content-block shapes, which embed it as a
    URL). Falls back to returning *url* unchanged if it isn't a data URI
    (e.g. it's already a bare base64 string), rather than raising.
    """
    if url.startswith("data:") and ";base64," in url:
        return url.split(";base64,", 1)[1]
    return url


def _message_to_ollama_payload(msg: Message) -> dict[str, Any]:
    """Convert a Missy Message into Ollama's ``/api/chat`` message shape.

    ``content`` is normally a plain string, but a caller building a real
    multimodal vision message (see missy/vision/provider_format.py's
    ``build_vision_message``) passes a list of text/image content blocks
    instead. Ollama's own API shape is different from Anthropic/OpenAI's
    content-block lists -- it wants plain-text ``content`` plus a sibling
    ``images`` list of base64 strings -- so list content must be split
    apart rather than forwarded as-is (which previously would have sent
    Ollama a JSON list where it expects a string).
    """
    if not isinstance(msg.content, list):
        return {"role": msg.role, "content": msg.content}

    text_parts: list[str] = []
    images: list[str] = []
    for part in msg.content:
        if isinstance(part, str):
            text_parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type", ""))
        if part_type in _TEXT_BLOCK_TYPES:
            text = part.get("text")
            if text is not None:
                text_parts.append(str(text))
        elif part_type in _IMAGE_BLOCK_TYPES:
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if url:
                images.append(_data_uri_to_base64(str(url)))

    payload: dict[str, Any] = {"role": msg.role, "content": "\n".join(text_parts)}
    if images:
        payload["images"] = images
    return payload


# Matches a ``<tool_call> ... </tool_call>`` block (qwen/Hermes style) OR a bare
# top-level JSON object, so we can recover a tool call a model emitted as plain
# text instead of via Ollama's structured ``message.tool_calls`` field.
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _salvage_tool_calls_from_content(
    content: str, tool_schemas: list[dict[str, Any]]
) -> tuple[list[ToolCall], str]:
    """Recover tool calls a model emitted as text in ``content``.

    Some models/templates on Ollama (seen with several qwen builds) emit a tool
    call as literal JSON in the message content instead of populating the
    structured ``message.tool_calls`` field, so the call leaks to the user
    unexecuted. This conservatively recovers such a call: a candidate is
    promoted **only** when it parses to ``{"name": <name>, "arguments": {...}}``
    and ``<name>`` is one of the tool names actually offered this turn — so an
    ordinary JSON payload the user legitimately asked the model to produce is
    never mistaken for a tool call.

    Args:
        content: The assistant message content text.
        tool_schemas: The Ollama tool schemas sent this turn (``{"type":
            "function", "function": {"name": ...}}``), used to validate names.

    Returns:
        ``(tool_calls, cleaned_content)`` — recovered calls (possibly empty) and
        the content with any promoted tool-call text removed.
    """
    if not content or not tool_schemas:
        return [], content

    valid_names = {
        (s.get("function") or {}).get("name") for s in tool_schemas if isinstance(s, dict)
    }
    valid_names.discard(None)
    valid_names.discard("")
    if not valid_names:
        return [], content

    candidates = _TOOL_CALL_TAG_RE.findall(content)
    if not candidates:
        stripped = content.strip()
        # Only treat the whole message as a call when it *is* a JSON object, to
        # avoid misreading prose that merely contains a brace.
        if stripped.startswith("{") and stripped.endswith("}"):
            m = _JSON_OBJECT_RE.search(stripped)
            if m:
                candidates = [m.group(0)]

    recovered: list[ToolCall] = []
    cleaned = content
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool") or ""
        if name not in valid_names:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        recovered.append(
            ToolCall(id=name[:8], name=name, arguments=args if isinstance(args, dict) else {})
        )
        cleaned = cleaned.replace(f"<tool_call>{raw}</tool_call>", "").replace(raw, "")

    if recovered:
        logger.info(
            "OllamaProvider: recovered %d tool call(s) from message content that the "
            "model emitted as text instead of structured tool_calls",
            len(recovered),
        )
    return recovered, cleaned.strip()


class OllamaProvider(BaseProvider):
    """Provider implementation backed by the Ollama local-inference server.

    All HTTP traffic is routed through :class:`~missy.gateway.client.PolicyHTTPClient`
    so network policy is enforced automatically.

    Args:
        config: Provider-level configuration.  ``base_url`` defaults to
            ``http://localhost:11434``.  ``api_key`` is unused for Ollama.
    """

    name = "ollama"

    def __init__(self, config: ProviderConfig) -> None:
        self._base_url: str = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._model: str = config.model or _DEFAULT_MODEL
        self._timeout: int = config.timeout
        self._client: PolicyHTTPClient | None = None

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def _make_client(
        self,
        session_id: str = "",
        task_id: str = "",
    ) -> PolicyHTTPClient:
        """Return a cached PolicyHTTPClient, creating one on first call."""
        if self._client is None:
            self._client = PolicyHTTPClient(
                session_id=session_id,
                task_id=task_id,
                timeout=self._timeout,
                category="provider",
            )
        return self._client

    def is_available(self) -> bool:
        """Return ``True`` when the Ollama server responds to ``GET /api/tags``.

        Returns:
            ``True`` when the tags endpoint returns HTTP 200.  Returns
            ``False`` on any network or HTTP error without raising.
        """
        try:
            client = PolicyHTTPClient(timeout=self._timeout, category="provider")
            response = client.get(f"{self._base_url}/api/tags")
            return response.status_code == 200
        except Exception as exc:
            logger.debug("Ollama availability check failed: %s", exc)
            return False

    def complete(self, messages: list[Message], **kwargs: Any) -> CompletionResponse:
        """Send *messages* to the Ollama ``/api/chat`` endpoint.

        Args:
            messages: Ordered conversation turns.  All role values supported
                by Ollama (``"system"``, ``"user"``, ``"assistant"``) are
                forwarded as-is.
            **kwargs: Optional overrides.  Recognised keys:

                * ``model`` (str) - override the configured model.
                * ``temperature`` (float) - sampling temperature forwarded
                  inside the ``options`` payload.

        Returns:
            A :class:`CompletionResponse` with the assistant reply.

        Raises:
            ProviderError: On HTTP error, unexpected response format, or
                any transport-level failure.
        """
        session_id = kwargs.pop("session_id", "")
        task_id = kwargs.pop("task_id", "")
        model = kwargs.pop("model", self._model)

        api_messages = [_message_to_ollama_payload(msg) for msg in messages]

        payload: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "stream": False,
        }

        options: dict[str, Any] = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs.pop("temperature")
        if options:
            payload["options"] = options

        # Forward any remaining kwargs directly into the payload
        payload.update(kwargs)

        self._acquire_rate_limit()

        try:
            client = self._make_client(session_id=session_id, task_id=task_id)
            response = client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
        except ProviderError:
            raise
        except Exception as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(f"Ollama request failed: {exc}") from exc

        try:
            data: dict[str, Any] = response.json()
        except Exception as exc:
            self._emit_event(session_id, task_id, "error", "invalid JSON response")
            raise ProviderError(f"Ollama returned invalid JSON: {exc}") from exc

        # Ollama chat response shape:
        # { "model": "...", "message": {"role": "assistant", "content": "..."},
        #   "prompt_eval_count": N, "eval_count": N, ... }
        message_obj = data.get("message") or {}
        content_text: str = message_obj.get("content", "")

        prompt_tokens = int(data.get("prompt_eval_count", 0))
        completion_tokens = int(data.get("eval_count", 0))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        self._emit_event(session_id, task_id, "allow", "completion successful")

        result = CompletionResponse(
            content=content_text,
            model=data.get("model", model),
            provider=self.name,
            usage=usage,
            raw=data,
        )
        self._record_rate_limit_usage(result)
        return result

    def get_tool_schema(self, tools: list) -> list:
        """Convert BaseTool instances to Ollama's native tool schema format.

        Ollama expects tools in the OpenAI-compatible format::

            {"type": "function", "function": {"name": "...", "description": "...",
             "parameters": {"type": "object", "properties": {...}, "required": [...]}}}

        Delegates to :func:`~missy.providers.schema_adapter.normalize_for_provider`
        for canonical → Ollama (OpenAI-compatible) conversion.

        Args:
            tools: List of :class:`~missy.tools.base.BaseTool` instances.

        Returns:
            A list of Ollama-native tool schema dicts.
        """
        try:
            from missy.providers.schema_adapter import normalize_for_provider

            schemas = []
            for tool in tools:
                base = tool.get_schema() if hasattr(tool, "get_schema") else {}
                canonical: dict[str, Any] = {
                    "name": getattr(tool, "name", ""),
                    "description": getattr(tool, "description", ""),
                    **base,
                }
                # Pre-normalise parameters so the adapter receives a proper JSON
                # Schema object.  Ollama historically expected this wrapping.
                params = canonical.get("parameters")
                if params is None or (isinstance(params, dict) and not params):
                    canonical["parameters"] = {"type": "object", "properties": {}}
                elif isinstance(params, dict) and "type" not in params:
                    canonical["parameters"] = {"type": "object", "properties": params}
                schemas.append(normalize_for_provider(canonical, "ollama"))
            return schemas
        except Exception:
            logger.debug("schema_adapter unavailable; falling back to inline schema build")

        schemas = []
        for tool in tools:
            base_schema = tool.get_schema() if hasattr(tool, "get_schema") else {}
            params = base_schema.get("parameters", {})
            if params and "type" not in params:
                params = {"type": "object", "properties": params}
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": params or {"type": "object", "properties": {}},
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
        """Send messages with native tool calling via Ollama's ``tools`` parameter.

        Uses Ollama's native ``/api/chat`` tool calling support. The tool
        schemas are passed as the ``tools`` key in the request payload and
        the model returns ``tool_calls`` in the response message when it
        wants to invoke a tool.

        Args:
            messages: Ordered conversation turns.
            tools: List of :class:`~missy.tools.base.BaseTool` instances.
            system: Optional system prompt string.

        Returns:
            A :class:`CompletionResponse`.  When ``finish_reason`` is
            ``"tool_calls"``, ``tool_calls`` contains the parsed
            :class:`~missy.providers.base.ToolCall` instances.
        """
        tool_schemas = self.get_tool_schema(tools)

        # Build messages, injecting system prompt if provided
        api_messages: list[dict[str, str]] = []
        has_system = any(m.role == "system" for m in messages)
        if system and not has_system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            api_messages.append(_message_to_ollama_payload(msg))

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "tools": tool_schemas,
            "stream": False,
        }

        self._acquire_rate_limit()

        try:
            client = self._make_client()
            response = client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
        except ProviderError:
            raise
        except Exception as exc:
            self._emit_event("", "", "error", str(exc))
            raise ProviderError(f"Ollama request failed: {exc}") from exc

        try:
            data: dict[str, Any] = response.json()
        except Exception as exc:
            self._emit_event("", "", "error", "invalid JSON response")
            raise ProviderError(f"Ollama returned invalid JSON: {exc}") from exc

        message_obj = data.get("message") or {}
        content_text: str = message_obj.get("content", "")

        prompt_tokens = int(data.get("prompt_eval_count", 0))
        completion_tokens = int(data.get("eval_count", 0))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        # Parse native tool_calls from the response
        raw_tool_calls = message_obj.get("tool_calls") or []
        parsed_tool_calls: list[ToolCall] = []
        for tc in raw_tool_calls:
            func = tc.get("function") or {}
            tc_name = func.get("name", "")
            tc_args = func.get("arguments", {})
            if tc_name:
                parsed_tool_calls.append(
                    ToolCall(
                        id=tc.get("id", "") or tc_name[:8],
                        name=tc_name,
                        arguments=tc_args if isinstance(tc_args, dict) else {},
                    )
                )

        # Fallback: some models emit the tool call as JSON text in `content`
        # instead of the structured `tool_calls` field, which would otherwise
        # leak to the user unexecuted. Recover it (validated against the offered
        # tool names) so tool-calling stays reliable across model/template quirks.
        if not parsed_tool_calls:
            salvaged, content_text = _salvage_tool_calls_from_content(content_text, tool_schemas)
            parsed_tool_calls = salvaged

        if parsed_tool_calls:
            self._emit_event("", "", "allow", "tool_calls")
            result = CompletionResponse(
                content=content_text,
                model=data.get("model", self._model),
                provider=self.name,
                usage=usage,
                raw=data,
                tool_calls=parsed_tool_calls,
                finish_reason="tool_calls",
            )
            self._record_rate_limit_usage(result)
            return result

        self._emit_event("", "", "allow", "completion successful")
        result = CompletionResponse(
            content=content_text,
            model=data.get("model", self._model),
            provider=self.name,
            usage=usage,
            raw=data,
            tool_calls=[],
            finish_reason="stop",
        )
        self._record_rate_limit_usage(result)
        return result

    def stream(self, messages: list[Message], system: str = "") -> Iterator[str]:
        """Stream partial response tokens from the Ollama API.

        Uses the ``/api/chat`` endpoint with ``stream=true`` and reads
        newline-delimited JSON chunks.

        Args:
            messages: Ordered conversation turns.
            system: Optional system prompt string (merged into messages if
                no system message is already present).

        Yields:
            String token chunks as they arrive.

        Raises:
            ProviderError: On transport failure or malformed response.
        """
        api_messages: list[dict] = []
        has_system = any(m.role == "system" for m in messages)
        if system and not has_system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            api_messages.append(_message_to_ollama_payload(msg))

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "stream": True,
        }

        self._acquire_rate_limit(estimated_tokens=self._estimate_tokens(messages, system))

        try:
            client = self._make_client()
            response = client.post(
                f"{self._base_url}/api/chat",
                json=payload,
                stream=True,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message_obj = chunk.get("message") or {}
                token = message_obj.get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Ollama stream failed: {exc}") from exc

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
        """Publish a provider audit event including model and base_url."""
        try:
            from missy.core.events import AuditEvent, event_bus

            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type="provider_invoke",
                category="provider",
                result=result,  # type: ignore[arg-type]
                detail={
                    "provider": self.name,
                    "model": self._model,
                    "base_url": self._base_url,
                    "message": detail_msg,
                },
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit audit event for provider %r", self.name)
