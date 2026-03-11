"""Abstract base classes for AI provider integrations.

All concrete providers must subclass :class:`BaseProvider` and implement
:meth:`complete` and :meth:`is_available`.  The :class:`Message` and
:class:`CompletionResponse` dataclasses form the canonical interchange format
that is provider-agnostic.

Example::

    from missy.providers.base import Message, CompletionResponse, BaseProvider

    class MyProvider(BaseProvider):
        name = "myprovider"

        def complete(self, messages, **kwargs):
            ...

        def is_available(self):
            return True
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Message:
    """A single turn in a conversation.

    Attributes:
        role: Speaker role - one of ``"user"``, ``"assistant"``, or
            ``"system"``.
        content: The text content of the message.
    """

    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class CompletionResponse:
    """The result of a provider completion call.

    Attributes:
        content: The generated text from the model.
        model: The exact model identifier used for the completion.
        provider: The provider name (e.g. ``"anthropic"``).
        usage: Token usage counters with keys ``prompt_tokens``,
            ``completion_tokens``, and ``total_tokens``.
        raw: The raw deserialized response payload from the provider API,
            useful for accessing provider-specific fields.
    """

    content: str
    model: str
    provider: str
    usage: dict  # {prompt_tokens, completion_tokens, total_tokens}
    raw: dict  # raw provider response


class BaseProvider(ABC):
    """Abstract base for all Missy AI provider implementations.

    Subclasses must set the class-level :attr:`name` attribute and implement
    both :meth:`complete` and :meth:`is_available`.

    Attributes:
        name: Short identifier for this provider (e.g. ``"anthropic"``).
    """

    name: str

    @abstractmethod
    def complete(self, messages: list[Message], **kwargs) -> CompletionResponse:
        """Run a completion against the provider.

        Args:
            messages: Ordered list of conversation turns to send.
            **kwargs: Provider-specific overrides (e.g. ``temperature``,
                ``max_tokens``).

        Returns:
            A :class:`CompletionResponse` with the model's reply.

        Raises:
            ProviderError: On any provider-side failure.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if this provider is ready to accept requests.

        Implementations should check that required credentials are present and
        (optionally) that the upstream service is reachable.

        Returns:
            ``True`` when the provider can service requests.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
