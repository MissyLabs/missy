"""Base channel abstractions for Missy input/output routing.

A *channel* represents a named source of user input and a corresponding
output sink.  Concrete implementations handle the mechanics of the
underlying transport (stdin/stdout, HTTP webhook, messaging platform, etc.)
while the rest of the framework works exclusively with the
:class:`ChannelMessage` data class and the :class:`BaseChannel` interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChannelMessage:
    """A normalised message received from any channel.

    Attributes:
        content: The text body of the message.
        sender: Identifier of the sender (e.g. ``"user"``).
        channel: Name of the channel this message arrived on.
        metadata: Arbitrary key/value pairs for channel-specific data
            (e.g. message IDs, timestamps, thread context).
    """

    content: str
    sender: str = "user"
    channel: str = "cli"
    metadata: dict = field(default_factory=dict)


class BaseChannel(ABC):
    """Abstract base class for all Missy I/O channels.

    Subclasses must declare a :attr:`name` class attribute that uniquely
    identifies the channel type and implement :meth:`receive` and
    :meth:`send`.
    """

    #: Unique identifier for this channel type.
    name: str

    @abstractmethod
    def receive(self) -> ChannelMessage | None:
        """Block until a message is available, then return it.

        Returns:
            A :class:`ChannelMessage` populated with the incoming content,
            or ``None`` when the channel signals end-of-input (e.g. EOF
            on stdin).
        """
        ...

    @abstractmethod
    def send(self, message: str) -> None:
        """Deliver *message* to the channel's output sink.

        Args:
            message: The text to send.
        """
        ...
