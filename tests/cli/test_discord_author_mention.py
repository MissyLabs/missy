"""The bot-author mention prefix must never duplicate a mention the model wrote."""

from __future__ import annotations

import pytest

from missy.cli.main import _prefix_author_mention


@pytest.mark.parametrize(
    "response",
    ["Done.", "<@42> Done.", "<@42> <@42> Done.", "<@!42>Done.", "  <@42>\n<@42> Done."],
)
def test_reply_starts_with_exactly_one_author_mention(response: str) -> None:
    assert _prefix_author_mention(response, "42") == "<@42> Done."


def test_other_mentions_and_body_mentions_are_preserved() -> None:
    assert _prefix_author_mention("<@7> hi <@42>", "42") == "<@42> <@7> hi <@42>"
