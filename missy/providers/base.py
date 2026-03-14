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
from collections.abc import Iterator
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
class ToolCall:
    """A tool invocation requested by the LLM.

    Attributes:
        id: Unique identifier for this tool call, used to correlate results.
        name: The name of the tool to invoke.
        arguments: Parsed keyword arguments for the tool invocation.
    """

    id: str
    name: str
    arguments: dict  # parsed from JSON


@dataclass
class ToolResult:
    """The result of executing a ToolCall.

    Attributes:
        tool_call_id: The ID of the :class:`ToolCall` this result corresponds to.
        name: The name of the tool that was executed.
        content: The result content as a string.
        is_error: ``True`` when the tool execution failed.
    """

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


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
        tool_calls: Tool invocations requested by the model (empty list for
            plain text completions).
        finish_reason: Why the model stopped generating.  One of
            ``"stop"``, ``"tool_calls"``, or ``"length"``.
    """

    content: str
    model: str
    provider: str
    usage: dict  # {prompt_tokens, completion_tokens, total_tokens}
    raw: dict  # raw provider response
    tool_calls: list = field(default_factory=list)  # list[ToolCall]
    finish_reason: str = "stop"  # "stop" | "tool_calls" | "length"


class BaseProvider(ABC):
    """Abstract base for all Missy AI provider implementations.

    Subclasses must set the class-level :attr:`name` attribute and implement
    :meth:`complete` and :meth:`is_available`.  The optional
    :meth:`get_tool_schema`, :meth:`complete_with_tools`, and :meth:`stream`
    methods have default implementations that work for basic providers.

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

    def get_tool_schema(self, tools: list) -> list:
        """Convert BaseTool instances to provider-native tool schema format.

        The default implementation returns a minimal list of tool description
        dicts using each tool's :meth:`~missy.tools.base.BaseTool.get_schema`
        output.  Concrete providers should override this to match their API
        expectations.

        Args:
            tools: List of :class:`~missy.tools.base.BaseTool` instances.

        Returns:
            A list of provider-specific tool schema dicts ready to pass to
            the provider API.
        """
        schemas = []
        for tool in tools:
            base = tool.get_schema() if hasattr(tool, "get_schema") else {}
            schemas.append(
                {
                    "name": getattr(tool, "name", ""),
                    "description": getattr(tool, "description", ""),
                    "parameters": base.get("parameters", {}),
                }
            )
        return schemas

    def complete_with_tools(
        self,
        messages: list[Message],
        tools: list,  # list of BaseTool instances
        system: str = "",
    ) -> CompletionResponse:
        """Like :meth:`complete` but with tool calling support.

        The default implementation falls back to :meth:`complete` (ignoring
        the tool list) so that providers that have not yet implemented native
        tool calling continue to work.  Override in concrete providers to
        enable full agentic tool calling.

        Args:
            messages: Ordered list of conversation turns to send.
            tools: List of :class:`~missy.tools.base.BaseTool` instances
                that the model may invoke (ignored in default implementation).
            system: Optional system prompt to prepend (ignored in default
                implementation; pass as a ``system`` Message instead).

        Returns:
            A :class:`CompletionResponse`.  When ``finish_reason`` is
            ``"tool_calls"``, ``tool_calls`` contains the requested
            invocations.

        Raises:
            ProviderError: On any provider-side failure.
        """
        return self.complete(messages)

    def stream(self, messages: list[Message], system: str = "") -> Iterator[str]:
        """Stream partial response tokens.

        Default implementation yields the full :meth:`complete` result as a
        single chunk.  Override to emit real streaming tokens.

        Args:
            messages: Ordered list of conversation turns to send.
            system: Optional system prompt (may be ignored if already in
                messages).

        Yields:
            String chunks of the model response.
        """
        response = self.complete(messages, system=system)
        yield response.content

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
