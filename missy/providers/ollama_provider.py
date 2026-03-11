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

import logging
from typing import Any

from missy.config.settings import ProviderConfig
from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import ProviderError
from missy.gateway.client import PolicyHTTPClient

from .base import BaseProvider, CompletionResponse, Message

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.2"


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

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return ``True`` when the Ollama server responds to ``GET /api/tags``.

        Returns:
            ``True`` when the tags endpoint returns HTTP 200.  Returns
            ``False`` on any network or HTTP error without raising.
        """
        try:
            client = PolicyHTTPClient(timeout=self._timeout)
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

        api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

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

        try:
            client = PolicyHTTPClient(
                session_id=session_id,
                task_id=task_id,
                timeout=self._timeout,
            )
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

        return CompletionResponse(
            content=content_text,
            model=data.get("model", model),
            provider=self.name,
            usage=usage,
            raw=data,
        )

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
