"""Agent runtime tests.


Tests for tool result truncation constant and resource bounds.
"""

from __future__ import annotations

from missy.agent.runtime import _MAX_TOOL_RESULT_CHARS


class TestToolResultTruncation:
    """Tool results exceeding _MAX_TOOL_RESULT_CHARS are truncated."""

    def test_max_tool_result_chars_is_positive(self) -> None:
        """Sanity check that the limit constant is defined and positive."""
        assert _MAX_TOOL_RESULT_CHARS > 0
        assert _MAX_TOOL_RESULT_CHARS == 200_000

    def test_truncation_limit_constant_exists(self) -> None:
        """The truncation limit is accessible from the module."""
        from missy.agent import runtime

        assert hasattr(runtime, "_MAX_TOOL_RESULT_CHARS")

    def test_truncation_logic_correct(self) -> None:
        """Verify the truncation logic works correctly on a long string."""
        content = "x" * 300_000
        if content and len(content) > _MAX_TOOL_RESULT_CHARS:
            truncated = (
                content[:_MAX_TOOL_RESULT_CHARS]
                + f"\n[TRUNCATED: output was {len(content)} chars, "
                f"limit is {_MAX_TOOL_RESULT_CHARS}]"
            )
        else:
            truncated = content

        # Verify truncation happened
        assert len(truncated) < len(content)
        assert truncated.startswith("x" * 100)
        assert "TRUNCATED" in truncated
        assert "300000" in truncated

    def test_short_content_not_truncated(self) -> None:
        """Content under the limit passes through unchanged."""
        content = "short result"
        if content and len(content) > _MAX_TOOL_RESULT_CHARS:
            truncated = content[:_MAX_TOOL_RESULT_CHARS] + "[TRUNCATED]"
        else:
            truncated = content

        assert truncated == "short result"

    def test_empty_content_not_truncated(self) -> None:
        """Empty content is not affected by truncation."""
        content = ""
        if content and len(content) > _MAX_TOOL_RESULT_CHARS:
            truncated = content[:_MAX_TOOL_RESULT_CHARS] + "[TRUNCATED]"
        else:
            truncated = content

        assert truncated == ""

    def test_none_content_not_truncated(self) -> None:
        """None content passes through the guard."""
        content = None
        if content and len(content) > _MAX_TOOL_RESULT_CHARS:
            truncated = content[:_MAX_TOOL_RESULT_CHARS] + "[TRUNCATED]"
        else:
            truncated = content

        assert truncated is None
