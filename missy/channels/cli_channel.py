"""CLI/stdin channel implementation.

:class:`CLIChannel` reads one line at a time from stdin and writes responses
to stdout.  It is the default channel used by ``missy run`` when no other
channel is configured.

Example::

    from missy.channels.cli_channel import CLIChannel

    channel = CLIChannel()
    msg = channel.receive()
    if msg is not None:
        channel.send(f"You said: {msg.content}")
"""

from __future__ import annotations

import sys

from .base import BaseChannel, ChannelMessage


class CLIChannel(BaseChannel):
    """Channel that reads from stdin and writes to stdout.

    End-of-file on stdin causes :meth:`receive` to return ``None``, which
    signals the caller (typically the ``run`` command loop) to exit cleanly.

    Args:
        prompt: Optional prompt string printed before each input request.
            Defaults to an empty string (no prompt) so that callers can
            render their own styled prompt via rich before calling
            :meth:`receive`.
    """

    name = "cli"

    def __init__(self, prompt: str = "") -> None:
        self._prompt = prompt

    def receive(self) -> ChannelMessage | None:
        """Read one line from stdin and return it as a :class:`ChannelMessage`.

        Returns:
            A :class:`ChannelMessage` with ``channel="cli"`` and
            ``sender="user"``, or ``None`` on EOF / ``KeyboardInterrupt``.
        """
        try:
            line = input(self._prompt)
        except EOFError:
            return None
        except KeyboardInterrupt:
            return None

        return ChannelMessage(
            content=line,
            sender="user",
            channel=self.name,
            metadata={},
        )

    def send(self, message: str) -> None:
        """Write *message* to stdout followed by a newline.

        Args:
            message: The text to print.
        """
        print(message, file=sys.stdout)
