"""Agent runtime for the Missy framework.

:class:`AgentRuntime` is the top-level orchestrator that binds a provider,
a session, and a tool registry into a single synchronous run loop.  Each
call to :meth:`AgentRuntime.run` creates (or reuses) a session, resolves
the configured provider, builds the message list, calls the provider, and
returns the model's reply as a plain string.

Example::

    from missy.agent.runtime import AgentRuntime, AgentConfig
    from missy.config.settings import load_config
    from missy.policy.engine import init_policy_engine
    from missy.providers.registry import init_registry

    config = load_config("missy.yaml")
    init_policy_engine(config)
    init_registry(config)

    agent = AgentRuntime(AgentConfig(provider="anthropic"))
    reply = agent.run("What is 2 + 2?")
    print(reply)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import ProviderError
from missy.core.session import Session, SessionManager
from missy.providers.base import Message, CompletionResponse
from missy.providers.registry import get_registry
from missy.tools.registry import get_tool_registry

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for a single :class:`AgentRuntime` instance.

    Attributes:
        provider: Registry key of the provider to use.  Defaults to
            ``"anthropic"``.  When the named provider is unavailable the
            runtime falls back to the first available provider.
        model: Optional model override forwarded to the provider's
            :meth:`~missy.providers.base.BaseProvider.complete` call.
            When ``None`` the provider's own default is used.
        system_prompt: System-level instruction prepended to every
            conversation.
        max_iterations: Maximum number of provider calls per
            :meth:`~AgentRuntime.run` invocation (reserved for future
            multi-step agentic use).
        temperature: Sampling temperature forwarded to the provider.
    """

    provider: str = "anthropic"
    model: Optional[str] = None
    system_prompt: str = "You are Missy, a helpful local assistant."
    max_iterations: int = 10
    temperature: float = 0.7


class AgentRuntime:
    """Synchronous agent runtime that resolves a provider and runs completions.

    A :class:`~missy.core.session.SessionManager` is owned by each runtime
    instance.  Sessions are created per :meth:`run` call unless a
    *session_id* is passed in (in which case the existing session is reused
    if it still exists, or a new one is created).

    Args:
        config: Runtime configuration.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._session_mgr = SessionManager()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, user_input: str, session_id: Optional[str] = None) -> str:
        """Run the agent with *user_input* and return the response string.

        Execution steps:

        1. Resolve or create the active :class:`~missy.core.session.Session`.
        2. Emit a ``"agent.run.start"`` audit event.
        3. Resolve the configured provider (with fallback to any available
           provider).
        4. Build the message list: system prompt + user input.
        5. Call :meth:`~missy.providers.base.BaseProvider.complete`.
        6. Emit a ``"agent.run.complete"`` or ``"agent.run.error"`` audit
           event.
        7. Return the completion content string.

        Args:
            user_input: The user's message text.
            session_id: Optional existing session ID.  When provided and a
                session with that ID is not already active on the current
                thread a new session is created (the ID is stored in its
                metadata for traceability).

        Returns:
            The model's reply as a plain string.

        Raises:
            ProviderError: When no provider is available or the provider
                call fails.
        """
        session = self._resolve_session(session_id)
        sid = str(session.id)
        task_id = str(self._session_mgr.generate_task_id())

        self._emit_event(
            session_id=sid,
            task_id=task_id,
            event_type="agent.run.start",
            result="allow",
            detail={"user_input_length": len(user_input)},
        )

        try:
            provider = self._get_provider()
        except ProviderError as exc:
            self._emit_event(
                session_id=sid,
                task_id=task_id,
                event_type="agent.run.error",
                result="error",
                detail={"error": str(exc), "stage": "provider_resolution"},
            )
            raise

        messages = self._build_messages(user_input)

        complete_kwargs: dict = {
            "session_id": sid,
            "task_id": task_id,
            "temperature": self.config.temperature,
        }
        if self.config.model:
            complete_kwargs["model"] = self.config.model

        try:
            response: CompletionResponse = provider.complete(messages, **complete_kwargs)
        except ProviderError as exc:
            self._emit_event(
                session_id=sid,
                task_id=task_id,
                event_type="agent.run.error",
                result="error",
                detail={
                    "error": str(exc),
                    "stage": "completion",
                    "provider": provider.name,
                },
            )
            raise
        except Exception as exc:
            self._emit_event(
                session_id=sid,
                task_id=task_id,
                event_type="agent.run.error",
                result="error",
                detail={
                    "error": str(exc),
                    "stage": "completion",
                    "provider": provider.name,
                },
            )
            raise ProviderError(f"Unexpected error during completion: {exc}") from exc

        self._emit_event(
            session_id=sid,
            task_id=task_id,
            event_type="agent.run.complete",
            result="allow",
            detail={
                "provider": response.provider,
                "model": response.model,
                "usage": response.usage,
            },
        )

        return response.content

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_session(self, session_id: Optional[str]) -> Session:
        """Return the current thread session, creating one if needed.

        Args:
            session_id: Optional caller-supplied identifier.  When provided
                it is stored in the new session's metadata so that audit
                events can be correlated by the caller's own ID scheme.

        Returns:
            The active :class:`~missy.core.session.Session` for this thread.
        """
        existing = self._session_mgr.get_current_session()
        if existing is not None:
            # Reuse the existing session for this thread.
            return existing

        metadata: dict = {}
        if session_id is not None:
            metadata["caller_session_id"] = session_id

        return self._session_mgr.create_session(metadata=metadata)

    def _get_provider(self):
        """Resolve the configured provider with automatic fallback.

        Returns:
            A :class:`~missy.providers.base.BaseProvider` instance.

        Raises:
            ProviderError: When no provider is available.
        """
        registry = get_registry()
        provider = registry.get(self.config.provider)

        if provider is not None and provider.is_available():
            return provider

        if provider is not None:
            logger.warning(
                "Configured provider %r is not available; falling back.",
                self.config.provider,
            )

        available = registry.get_available()
        if available:
            fallback = available[0]
            logger.info("Using fallback provider %r.", fallback.name)
            return fallback

        raise ProviderError(
            f"No providers available. Configured provider was {self.config.provider!r}. "
            "Ensure at least one provider is initialised and its API key is set."
        )

    def _build_messages(self, user_input: str) -> list[Message]:
        """Construct the message list to send to the provider.

        Args:
            user_input: The user's text input for this turn.

        Returns:
            A list of :class:`~missy.providers.base.Message` objects with
            the system prompt first, followed by the user turn.
        """
        return [
            Message(role="system", content=self.config.system_prompt),
            Message(role="user", content=user_input),
        ]

    def _emit_event(
        self,
        session_id: str,
        task_id: str,
        event_type: str,
        result: str,
        detail: dict,
    ) -> None:
        """Publish an agent lifecycle audit event.

        Args:
            session_id: Session identifier.
            task_id: Task identifier.
            event_type: Dotted event type string.
            result: One of ``"allow"`` or ``"error"``.
            detail: Structured event data.
        """
        try:
            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type=event_type,
                category="provider",
                result=result,  # type: ignore[arg-type]
                detail=detail,
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit audit event %r", event_type)
