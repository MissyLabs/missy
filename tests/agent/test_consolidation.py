"""Tests for missy.agent.consolidation — Sleep Mode memory consolidation."""

from __future__ import annotations

from missy.agent.consolidation import MemoryConsolidator


class TestShouldConsolidateBelowThreshold:
    def test_should_consolidate_below_threshold(self):
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=30000)
        # 70% usage -> should NOT consolidate
        assert mc.should_consolidate(21000) is False

    def test_exactly_at_threshold(self):
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=30000)
        # 80% usage -> should consolidate (>= threshold)
        assert mc.should_consolidate(24000) is True

    def test_zero_max_tokens(self):
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=0)
        assert mc.should_consolidate(100) is False


class TestShouldConsolidateAboveThreshold:
    def test_should_consolidate_above_threshold(self):
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=30000)
        # 85% usage -> should consolidate
        assert mc.should_consolidate(25500) is True

    def test_100_percent(self):
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=30000)
        assert mc.should_consolidate(30000) is True


class TestConsolidatePreservesRecent:
    def test_consolidate_preserves_recent(self):
        mc = MemoryConsolidator()
        messages = [{"role": "user", "content": f"message {i}"} for i in range(8)]
        result, summary = mc.consolidate(messages, "system prompt")
        # Last 4 messages should be preserved intact
        assert result[-4:] == messages[-4:]
        # First message should be the summary
        assert "[Session context consolidated]" in result[0]["content"]

    def test_consolidate_total_length(self):
        mc = MemoryConsolidator()
        messages = [{"role": "user", "content": f"message {i}"} for i in range(10)]
        result, summary = mc.consolidate(messages, "system prompt")
        # 1 summary + 4 recent = 5 messages
        assert len(result) == 5


class TestConsolidateCompressesOld:
    def test_consolidate_compresses_old(self):
        mc = MemoryConsolidator()
        messages = [
            {"role": "user", "content": "Please read the config file"},
            {"role": "assistant", "content": "I found: the config has 3 sections"},
            {"role": "user", "content": "Result: deployment succeeded"},
            {"role": "tool", "name": "shell_exec", "content": "output: all tests pass"},
            {"role": "user", "content": "recent 1"},
            {"role": "assistant", "content": "recent 2"},
            {"role": "user", "content": "recent 3"},
            {"role": "assistant", "content": "recent 4"},
        ]
        result, summary = mc.consolidate(messages, "system prompt")
        # Old messages compressed into summary
        assert len(result) == 5  # 1 summary + 4 recent
        # Summary should contain extracted facts
        assert summary  # not empty

    def test_old_messages_not_in_result(self):
        mc = MemoryConsolidator()
        messages = [
            {"role": "user", "content": "old message"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "another old"},
            {"role": "assistant", "content": "another reply"},
            {"role": "user", "content": "recent 1"},
            {"role": "assistant", "content": "recent 2"},
            {"role": "user", "content": "recent 3"},
            {"role": "assistant", "content": "recent 4"},
        ]
        result, _ = mc.consolidate(messages, "system prompt")
        contents = [m["content"] for m in result]
        assert "old message" not in contents
        assert "old reply" not in contents


class TestExtractKeyFacts:
    def test_extract_key_facts(self):
        mc = MemoryConsolidator()
        messages = [
            {"role": "user", "content": "Deploy the app"},
            {
                "role": "assistant",
                "content": "I'll help.\nResult: deployment successful\nHere is some verbose text that goes on and on.",
            },
            {"role": "tool", "name": "shell_exec", "content": "exit code 0"},
            {"role": "user", "content": "Decided: use blue-green deployment"},
        ]
        facts = mc.extract_key_facts(messages)
        assert any("Result:" in f or "result:" in f.lower() for f in facts)
        assert any("shell_exec" in f for f in facts)
        assert any("Decided:" in f or "decided:" in f.lower() for f in facts)

    def test_extract_skips_long_user_messages(self):
        mc = MemoryConsolidator()
        long_msg = "x" * 200
        messages = [
            {"role": "user", "content": long_msg},
        ]
        facts = mc.extract_key_facts(messages)
        # Long user message without keywords should not appear
        assert long_msg not in facts

    def test_extract_deduplicates(self):
        mc = MemoryConsolidator()
        messages = [
            {"role": "user", "content": "Result: done"},
            {"role": "user", "content": "Result: done"},
        ]
        facts = mc.extract_key_facts(messages)
        assert facts.count("Result: done") == 1


class TestEstimateTokens:
    def test_estimate_tokens(self):
        mc = MemoryConsolidator()
        messages = [
            {"role": "user", "content": "a" * 400},
            {"role": "assistant", "content": "b" * 400},
        ]
        tokens = mc.estimate_tokens(messages)
        assert tokens == 200  # 800 chars / 4

    def test_estimate_tokens_empty_content(self):
        mc = MemoryConsolidator()
        messages = [{"role": "user", "content": ""}]
        assert mc.estimate_tokens(messages) == 0


class TestEmptyMessages:
    def test_empty_messages(self):
        mc = MemoryConsolidator()
        result, summary = mc.consolidate([], "system prompt")
        assert result == []
        assert summary == ""

    def test_fewer_than_recent_keep(self):
        mc = MemoryConsolidator()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result, summary = mc.consolidate(messages, "system prompt")
        # Nothing to consolidate — returned as-is
        assert result == messages
        assert summary == ""

    def test_exactly_recent_keep(self):
        mc = MemoryConsolidator()
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(4)]
        result, summary = mc.consolidate(messages, "system prompt")
        assert result == messages
        assert summary == ""
