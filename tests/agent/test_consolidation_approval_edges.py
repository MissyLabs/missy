"""Comprehensive tests for MemoryConsolidator and ApprovalGate.


Covers the full behavioural spec for both modules including boundary conditions,
keyword detection, threading, and exception message content.  All scenarios are
distinct from those already exercised in:
  - tests/agent/test_consolidation.py
  - tests/agent/test_approval_gate.py
  - tests/agent/test_attention_consolidation_edges.py
  - tests/agent/test_approval_subagent_edges.py
"""

from __future__ import annotations

import threading
import time

import pytest

from missy.agent.approval import (
    ApprovalDenied,
    ApprovalGate,
    ApprovalTimeout,
    PendingApproval,
)
from missy.agent.consolidation import (
    _FACT_KEYWORDS,
    _RECENT_KEEP,
    MemoryConsolidator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_pending(gate: ApprovalGate, count: int = 1, retries: int = 150) -> list[dict]:
    """Poll until at least *count* pending entries appear."""
    for _ in range(retries):
        pending = gate.list_pending()
        if len(pending) >= count:
            return pending
        time.sleep(0.01)
    return gate.list_pending()


def _request_in_thread(
    gate: ApprovalGate,
    action: str = "action",
    reason: str = "",
    risk: str = "medium",
) -> tuple[threading.Thread, dict]:
    """Run gate.request() in a daemon thread, capturing the outcome."""
    outcome: dict = {}

    def _run() -> None:
        try:
            gate.request(action, reason=reason, risk=risk)
            outcome["approved"] = True
        except ApprovalDenied as exc:
            outcome["denied"] = True
            outcome["exc_msg"] = str(exc)
        except ApprovalTimeout as exc:
            outcome["timeout"] = True
            outcome["exc_msg"] = str(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, outcome


# ===========================================================================
# MemoryConsolidator — should_consolidate
# ===========================================================================


class TestShouldConsolidateBoundary:
    """Boundary precision at the 80 % threshold with max_tokens=10_000."""

    def test_just_below_threshold_7999(self):
        """79.99 % rounds to strictly below 0.8 — must NOT trigger."""
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=10_000)
        # 7999 / 10000 = 0.7999 < 0.8
        assert mc.should_consolidate(7_999) is False

    def test_exactly_at_threshold_8000(self):
        """8000 / 10000 == 0.8 == threshold — must trigger (>= comparison)."""
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=10_000)
        assert mc.should_consolidate(8_000) is True

    def test_one_over_threshold_8001(self):
        """8001 / 10000 > 0.8 — must trigger."""
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=10_000)
        assert mc.should_consolidate(8_001) is True

    def test_max_tokens_zero_always_false(self):
        """Division-by-zero guard: max_tokens=0 always returns False."""
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=0)
        assert mc.should_consolidate(0) is False
        assert mc.should_consolidate(999_999) is False

    def test_negative_max_tokens_treated_as_disabled(self):
        """Negative max_tokens satisfies the <= 0 guard and returns False."""
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=-1)
        assert mc.should_consolidate(100) is False

    def test_zero_current_tokens_never_triggers(self):
        """0 / max_tokens == 0.0, always below any positive threshold."""
        mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=30_000)
        assert mc.should_consolidate(0) is False

    def test_threshold_pct_one_requires_full_capacity(self):
        """threshold_pct=1.0 triggers only when current_tokens == max_tokens."""
        mc = MemoryConsolidator(threshold_pct=1.0, max_tokens=1_000)
        assert mc.should_consolidate(999) is False
        assert mc.should_consolidate(1_000) is True


# ===========================================================================
# MemoryConsolidator — consolidate
# ===========================================================================


class TestConsolidateEmptyAndSmall:
    def test_empty_messages_returns_empty_list_and_empty_string(self):
        mc = MemoryConsolidator()
        result, summary = mc.consolidate([], "sys")
        assert result == []
        assert summary == ""

    def test_one_message_returned_unchanged(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": "hello"}]
        result, summary = mc.consolidate(msgs, "sys")
        assert result == msgs
        assert summary == ""

    def test_three_messages_returned_unchanged(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(3)]
        result, summary = mc.consolidate(msgs, "sys")
        assert result == msgs
        assert summary == ""

    def test_exactly_four_messages_returned_unchanged(self):
        """_RECENT_KEEP == 4; exactly 4 messages should pass through untouched."""
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(_RECENT_KEEP)]
        result, summary = mc.consolidate(msgs, "sys")
        assert result == msgs
        assert summary == ""


class TestConsolidateFivePlusMessages:
    def test_five_messages_produces_one_summary_plus_four_recent(self):
        """First message becomes the old context; result = 1 summary + 4 recent."""
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        result, _ = mc.consolidate(msgs, "sys")
        assert len(result) == 5
        # First element is the synthesised summary
        assert result[0]["role"] == "user"
        assert "[Session context consolidated]" in result[0]["content"]
        # Last four are the original tail
        assert result[1:] == msgs[1:]

    def test_ten_messages_produces_one_summary_plus_four_recent(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(10)]
        result, _ = mc.consolidate(msgs, "sys")
        assert len(result) == 5
        assert result[1:] == msgs[-4:]

    def test_recent_four_are_identity_copies(self):
        """The last four messages must be exactly the original dict objects (or equal)."""
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(8)]
        result, _ = mc.consolidate(msgs, "sys")
        for original, kept in zip(msgs[-4:], result[-4:], strict=True):
            assert original == kept

    def test_summary_contains_extracted_fact_keyword(self):
        """Summary must include lines matched by fact keywords from old messages."""
        mc = MemoryConsolidator()
        msgs = [
            {"role": "assistant", "content": "Result: the migration finished cleanly"},
            {"role": "user", "content": "ok"},
            {"role": "user", "content": "keep1"},
            {"role": "user", "content": "keep2"},
            {"role": "user", "content": "keep3"},
            {"role": "user", "content": "keep4"},
        ]
        _, summary = mc.consolidate(msgs, "sys")
        assert "Result: the migration finished cleanly" in summary

    def test_no_facts_fallback_text_appears_in_summary(self):
        """When old messages have no extractable facts, the fallback placeholder is used."""
        mc = MemoryConsolidator()
        # All old messages are long prose with no keywords and long user text.
        old = [
            {"role": "assistant", "content": "Certainly, allow me to elaborate at great length about this topic."},
            {"role": "assistant", "content": "There are many perspectives to consider here as well."},
        ]
        recent = [{"role": "user", "content": f"r{i}"} for i in range(4)]
        _, summary = mc.consolidate(old + recent, "sys")
        assert "no key facts extracted" in summary

    def test_summary_text_embedded_in_content_field(self):
        """The summary text from the return value must match what is embedded in the message."""
        mc = MemoryConsolidator()
        msgs = [
            {"role": "user", "content": "decided: use plan B"},
            {"role": "user", "content": "r1"},
            {"role": "user", "content": "r2"},
            {"role": "user", "content": "r3"},
            {"role": "user", "content": "r4"},
        ]
        result, summary = mc.consolidate(msgs, "sys")
        # The summary text should be embedded inside the consolidated message content.
        assert summary in result[0]["content"]

    def test_original_list_not_mutated(self):
        """consolidate() must not alter the original messages list."""
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(8)]
        original_len = len(msgs)
        mc.consolidate(msgs, "sys")
        assert len(msgs) == original_len

    def test_consolidated_list_is_new_object(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(6)]
        result, _ = mc.consolidate(msgs, "sys")
        assert result is not msgs


# ===========================================================================
# MemoryConsolidator — extract_key_facts
# ===========================================================================


class TestExtractKeyFacts:
    def test_tool_message_truncated_to_200_chars(self):
        """Tool content in the fact string must be at most 200 chars."""
        mc = MemoryConsolidator()
        long_output = "Z" * 400
        msgs = [{"role": "tool", "name": "my_tool", "content": long_output}]
        facts = mc.extract_key_facts(msgs)
        assert len(facts) == 1
        # "[my_tool] " is 10 chars; content portion must be exactly 200 chars.
        assert facts[0] == f"[my_tool] {'Z' * 200}"

    def test_tool_message_short_content_not_padded(self):
        """Short tool content should be included verbatim (not padded to 200)."""
        mc = MemoryConsolidator()
        msgs = [{"role": "tool", "name": "checker", "content": "ok"}]
        facts = mc.extract_key_facts(msgs)
        assert facts == ["[checker] ok"]

    def test_tool_message_with_empty_content_not_included(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "tool", "name": "noop", "content": ""}]
        assert mc.extract_key_facts(msgs) == []

    def test_tool_message_with_whitespace_only_content_not_included(self):
        """Whitespace-only tool output stripped to empty should be skipped."""
        mc = MemoryConsolidator()
        msgs = [{"role": "tool", "name": "noop", "content": "   \n\t  "}]
        assert mc.extract_key_facts(msgs) == []

    def test_duplicate_tool_facts_deduplicated(self):
        """Two identical tool messages must only produce one fact entry."""
        mc = MemoryConsolidator()
        msg = {"role": "tool", "name": "cmd", "content": "exit 0"}
        facts = mc.extract_key_facts([msg, msg])
        assert facts.count("[cmd] exit 0") == 1

    def test_duplicate_keyword_lines_deduplicated(self):
        """Identical keyword lines across messages appear only once."""
        mc = MemoryConsolidator()
        msgs = [
            {"role": "assistant", "content": "Result: done"},
            {"role": "assistant", "content": "Result: done"},
        ]
        facts = mc.extract_key_facts(msgs)
        assert facts.count("Result: done") == 1

    def test_short_user_message_at_120_chars_included(self):
        """User message of exactly 120 chars satisfies 0 < len <= 120 and is kept."""
        mc = MemoryConsolidator()
        msg_120 = "A" * 120
        msgs = [{"role": "user", "content": msg_120}]
        facts = mc.extract_key_facts(msgs)
        assert msg_120 in facts

    def test_long_user_message_121_chars_not_included_unless_keyword(self):
        """User message exceeding 120 chars without a keyword must be omitted."""
        mc = MemoryConsolidator()
        msg_121 = "B" * 121
        msgs = [{"role": "user", "content": msg_121}]
        facts = mc.extract_key_facts(msgs)
        assert msg_121 not in facts

    def test_long_user_message_with_keyword_line_still_extracts_the_line(self):
        """A long user message containing a keyword line emits that line as a fact."""
        mc = MemoryConsolidator()
        long_msg = ("filler text " * 20) + "\nResult: the system recovered"
        msgs = [{"role": "user", "content": long_msg}]
        facts = mc.extract_key_facts(msgs)
        assert any("Result: the system recovered" in f for f in facts)

    def test_every_keyword_triggers_extraction(self):
        """Each of the 10 fact keywords must cause the bearing line to be extracted."""
        mc = MemoryConsolidator()
        test_cases = [
            "result: alpha",
            "decided: beta",
            "found: gamma",
            "error: delta",
            "success: epsilon",
            "created: zeta",
            "updated: eta",
            "deleted: theta",
            "confirmed: iota",
            "output: kappa",
        ]
        for line in test_cases:
            msgs = [{"role": "assistant", "content": line}]
            facts = mc.extract_key_facts(msgs)
            assert any(line in f for f in facts), f"Keyword not detected in: {line!r}"

    def test_keyword_count_matches_constant(self):
        """Sanity check: _FACT_KEYWORDS has exactly 10 entries."""
        assert len(_FACT_KEYWORDS) == 10

    def test_keyword_case_insensitive_matching(self):
        """Keywords in ALL CAPS must still be detected via .lower() comparison."""
        mc = MemoryConsolidator()
        msgs = [{"role": "assistant", "content": "ERROR: disk full"}]
        facts = mc.extract_key_facts(msgs)
        assert any("ERROR: disk full" in f for f in facts)

    def test_multiline_message_extracts_all_keyword_lines(self):
        """Multiple keyword lines in a single message are all extracted."""
        mc = MemoryConsolidator()
        content = "result: step 1 ok\nsome prose\nfound: the issue\nmore prose"
        msgs = [{"role": "assistant", "content": content}]
        facts = mc.extract_key_facts(msgs)
        assert any("result: step 1 ok" in f for f in facts)
        assert any("found: the issue" in f for f in facts)

    def test_empty_messages_list_returns_empty_list(self):
        mc = MemoryConsolidator()
        assert mc.extract_key_facts([]) == []

    def test_assistant_long_prose_no_keywords_excluded(self):
        """Long assistant prose without keywords is ignored entirely."""
        mc = MemoryConsolidator()
        prose = "This is a long explanation about how things work in general. " * 5
        msgs = [{"role": "assistant", "content": prose}]
        facts = mc.extract_key_facts(msgs)
        assert facts == []

    def test_tool_name_missing_defaults_to_tool(self):
        """Tool message without a 'name' key uses 'tool' as the label."""
        mc = MemoryConsolidator()
        msgs = [{"role": "tool", "content": "some output"}]
        facts = mc.extract_key_facts(msgs)
        assert len(facts) == 1
        assert facts[0].startswith("[tool]")

    def test_short_user_message_with_keyword_not_double_added(self):
        """A short user message bearing a keyword must not appear twice in facts."""
        mc = MemoryConsolidator()
        # 'result: ok' is 10 chars — short user message AND keyword line.
        # The keyword branch fires first (adds it), then the short-user branch
        # uses the `seen` guard to skip it.
        msgs = [{"role": "user", "content": "result: ok"}]
        facts = mc.extract_key_facts(msgs)
        assert facts.count("result: ok") == 1


# ===========================================================================
# MemoryConsolidator — estimate_tokens
# ===========================================================================


class TestEstimateTokens:
    def test_empty_list_returns_zero(self):
        mc = MemoryConsolidator()
        assert mc.estimate_tokens([]) == 0

    def test_characters_divided_by_four(self):
        """800 characters across two messages → 200 tokens."""
        mc = MemoryConsolidator()
        msgs = [
            {"role": "user", "content": "a" * 400},
            {"role": "assistant", "content": "b" * 400},
        ]
        assert mc.estimate_tokens(msgs) == 200

    def test_three_chars_rounds_down_to_zero(self):
        """3 chars / 4 == 0 via integer division."""
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": "abc"}]
        assert mc.estimate_tokens(msgs) == 0

    def test_four_chars_is_exactly_one_token(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": "abcd"}]
        assert mc.estimate_tokens(msgs) == 1

    def test_single_empty_content_message_returns_zero(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "user", "content": ""}]
        assert mc.estimate_tokens(msgs) == 0

    def test_missing_content_key_treated_as_zero_chars(self):
        mc = MemoryConsolidator()
        msgs = [{"role": "user"}]
        assert mc.estimate_tokens(msgs) == 0

    def test_non_string_content_coerced_via_str(self):
        """Non-string content (e.g. int) must be str()-coerced before measuring."""
        mc = MemoryConsolidator()
        # str(12345678) == "12345678" → 8 chars → 2 tokens
        msgs = [{"role": "tool", "content": 12_345_678}]
        assert mc.estimate_tokens(msgs) == 2

    def test_is_static_method_callable_on_class(self):
        """estimate_tokens is a @staticmethod and can be called without an instance."""
        msgs = [{"role": "user", "content": "a" * 40}]
        assert MemoryConsolidator.estimate_tokens(msgs) == 10


# ===========================================================================
# PendingApproval
# ===========================================================================


class TestPendingApprovalState:
    def test_initial_approved_is_none(self):
        pa = PendingApproval("do x", "reason", timeout=5.0)
        assert pa._approved is None

    def test_approve_sets_approved_to_true(self):
        pa = PendingApproval("do x", "reason", timeout=5.0)
        pa.approve()
        assert pa._approved is True

    def test_deny_sets_approved_to_false(self):
        pa = PendingApproval("do x", "reason", timeout=5.0)
        pa.deny()
        assert pa._approved is False

    def test_wait_returns_true_when_approved(self):
        pa = PendingApproval("do x", "reason", timeout=5.0)
        pa.approve()
        assert pa.wait() is True

    def test_wait_raises_approval_denied_when_denied(self):
        pa = PendingApproval("do x", "reason", timeout=5.0)
        pa.deny()
        with pytest.raises(ApprovalDenied):
            pa.wait()

    def test_wait_raises_approval_timeout_when_no_response(self):
        pa = PendingApproval("do x", "reason", timeout=0.05)
        with pytest.raises(ApprovalTimeout):
            pa.wait()

    def test_approval_timeout_message_contains_action(self):
        action = "launch_missiles"
        pa = PendingApproval(action, "reason", timeout=0.05)
        with pytest.raises(ApprovalTimeout) as exc_info:
            pa.wait()
        assert action in str(exc_info.value)

    def test_approval_denied_message_contains_action(self):
        action = "wipe_database"
        pa = PendingApproval(action, "reason", timeout=5.0)
        pa.deny()
        with pytest.raises(ApprovalDenied) as exc_info:
            pa.wait()
        assert action in str(exc_info.value)

    def test_approval_timeout_message_mentions_seconds(self):
        """The timeout duration in seconds should appear in the exception message."""
        pa = PendingApproval("do x", "reason", timeout=0.12)
        with pytest.raises(ApprovalTimeout) as exc_info:
            pa.wait()
        assert "0.12" in str(exc_info.value)

    def test_approve_from_different_thread(self):
        """Approving from another thread must unblock wait() with True."""
        pa = PendingApproval("do x", "reason", timeout=2.0)
        results: list = []

        def _approve():
            time.sleep(0.02)
            pa.approve()

        t = threading.Thread(target=_approve, daemon=True)
        t.start()
        results.append(pa.wait())
        t.join()
        assert results == [True]

    def test_deny_from_different_thread_raises_denied(self):
        """Denying from another thread must unblock wait() with ApprovalDenied."""
        pa = PendingApproval("do x", "reason", timeout=2.0)
        exc_holder: list = []

        def _deny():
            time.sleep(0.02)
            pa.deny()

        def _wait():
            try:
                pa.wait()
            except ApprovalDenied as exc:
                exc_holder.append(exc)

        t_deny = threading.Thread(target=_deny, daemon=True)
        t_wait = threading.Thread(target=_wait, daemon=True)
        t_deny.start()
        t_wait.start()
        t_deny.join()
        t_wait.join(timeout=2.0)
        assert len(exc_holder) == 1
        assert isinstance(exc_holder[0], ApprovalDenied)


# ===========================================================================
# ApprovalGate
# ===========================================================================


class TestApprovalGateRequest:
    def test_request_resolves_when_approved_from_other_thread(self):
        gate = ApprovalGate(default_timeout=2.0)
        t, outcome = _request_in_thread(gate, action="delete cache")
        pending = _wait_for_pending(gate)
        gate.handle_response(f"approve {pending[0]['id']}")
        t.join(timeout=2.0)
        assert outcome.get("approved") is True

    def test_request_raises_approval_denied_when_denied(self):
        gate = ApprovalGate(default_timeout=2.0)
        t, outcome = _request_in_thread(gate, action="purge data")
        pending = _wait_for_pending(gate)
        gate.handle_response(f"deny {pending[0]['id']}")
        t.join(timeout=2.0)
        assert outcome.get("denied") is True

    def test_request_raises_approval_timeout_when_no_response(self):
        gate = ApprovalGate(default_timeout=0.05)
        with pytest.raises(ApprovalTimeout):
            gate.request("timed-out action")

    def test_approval_timeout_exception_contains_action_name(self):
        gate = ApprovalGate(default_timeout=0.05)
        with pytest.raises(ApprovalTimeout) as exc_info:
            gate.request("reformat_drive")
        assert "reformat_drive" in str(exc_info.value)

    def test_approval_denied_exception_contains_action_name(self):
        gate = ApprovalGate(default_timeout=2.0)
        t, outcome = _request_in_thread(gate, action="drop_table")
        pending = _wait_for_pending(gate)
        gate.handle_response(f"deny {pending[0]['id']}")
        t.join(timeout=2.0)
        assert "drop_table" in outcome.get("exc_msg", "")


class TestApprovalGateHandleResponse:
    def test_handle_response_approve_returns_true(self):
        gate = ApprovalGate(default_timeout=2.0)
        t, _ = _request_in_thread(gate)
        pending = _wait_for_pending(gate)
        result = gate.handle_response(f"approve {pending[0]['id']}")
        t.join(timeout=2.0)
        assert result is True

    def test_handle_response_deny_returns_true(self):
        gate = ApprovalGate(default_timeout=2.0)
        t, _ = _request_in_thread(gate)
        pending = _wait_for_pending(gate)
        result = gate.handle_response(f"deny {pending[0]['id']}")
        t.join(timeout=2.0)
        assert result is True

    def test_handle_response_unmatched_text_returns_false(self):
        gate = ApprovalGate()
        assert gate.handle_response("random garbage") is False

    def test_handle_response_empty_string_returns_false(self):
        gate = ApprovalGate()
        assert gate.handle_response("") is False

    def test_handle_response_is_case_insensitive_approve(self):
        """APPROVE in upper-case must be normalised and still match."""
        gate = ApprovalGate(default_timeout=2.0)
        t, outcome = _request_in_thread(gate)
        pending = _wait_for_pending(gate)
        approval_id = pending[0]["id"]
        result = gate.handle_response(f"APPROVE {approval_id}")
        t.join(timeout=2.0)
        assert result is True
        assert outcome.get("approved") is True

    def test_handle_response_is_case_insensitive_deny(self):
        """DENY in upper-case must be normalised and still match."""
        gate = ApprovalGate(default_timeout=2.0)
        t, outcome = _request_in_thread(gate)
        pending = _wait_for_pending(gate)
        approval_id = pending[0]["id"]
        result = gate.handle_response(f"DENY {approval_id}")
        t.join(timeout=2.0)
        assert result is True
        assert outcome.get("denied") is True

    def test_handle_response_mixed_case_approve(self):
        """Approve with mixed-case ID text must be handled correctly."""
        gate = ApprovalGate(default_timeout=2.0)
        t, outcome = _request_in_thread(gate)
        pending = _wait_for_pending(gate)
        approval_id = pending[0]["id"]
        # IDs are already lowercase hex; mixed-case of "Approve" should still work.
        result = gate.handle_response(f"Approve {approval_id}")
        t.join(timeout=2.0)
        assert result is True

    def test_handle_response_unknown_id_returns_false(self):
        gate = ApprovalGate(default_timeout=2.0)
        t, _ = _request_in_thread(gate)
        _wait_for_pending(gate)
        result = gate.handle_response("approve 00000000")
        # Clean up
        pending = gate.list_pending()
        if pending:
            gate.handle_response(f"approve {pending[0]['id']}")
        t.join(timeout=2.0)
        assert result is False


class TestApprovalGateListPending:
    def test_list_pending_empty_initially(self):
        gate = ApprovalGate()
        assert gate.list_pending() == []

    def test_list_pending_shows_active_request(self):
        gate = ApprovalGate(default_timeout=2.0)
        t, _ = _request_in_thread(gate, action="export data")
        pending = _wait_for_pending(gate)
        assert len(pending) == 1
        assert pending[0]["action"] == "export data"
        # Clean up
        gate.handle_response(f"approve {pending[0]['id']}")
        t.join(timeout=2.0)

    def test_list_pending_cleaned_up_after_approval(self):
        gate = ApprovalGate(default_timeout=2.0)
        t, _ = _request_in_thread(gate)
        pending = _wait_for_pending(gate)
        gate.handle_response(f"approve {pending[0]['id']}")
        t.join(timeout=2.0)
        assert gate.list_pending() == []

    def test_list_pending_cleaned_up_after_denial(self):
        gate = ApprovalGate(default_timeout=2.0)
        t, _ = _request_in_thread(gate)
        pending = _wait_for_pending(gate)
        gate.handle_response(f"deny {pending[0]['id']}")
        t.join(timeout=2.0)
        assert gate.list_pending() == []

    def test_list_pending_cleaned_up_after_timeout(self):
        gate = ApprovalGate(default_timeout=0.05)
        with pytest.raises(ApprovalTimeout):
            gate.request("slow action")
        assert gate.list_pending() == []


class TestApprovalGateSendFn:
    def test_send_fn_none_does_not_raise(self):
        """send_fn=None is explicitly supported; no error should occur."""
        gate = ApprovalGate(send_fn=None, default_timeout=0.05)
        with pytest.raises(ApprovalTimeout):
            gate.request("any action")

    def test_send_fn_called_once_per_request(self):
        messages: list[str] = []
        gate = ApprovalGate(send_fn=messages.append, default_timeout=0.05)
        with pytest.raises(ApprovalTimeout):
            gate.request("action")
        assert len(messages) == 1

    def test_send_fn_message_contains_action(self):
        messages: list[str] = []
        gate = ApprovalGate(send_fn=messages.append, default_timeout=0.05)
        with pytest.raises(ApprovalTimeout):
            gate.request("deploy_to_prod")
        assert "deploy_to_prod" in messages[0]

    def test_send_fn_failure_does_not_abort_approval_flow(self):
        """A crashing send_fn must be swallowed; the approval flow must continue."""
        def _bad_send(msg: str) -> None:
            raise OSError("network error")

        gate = ApprovalGate(send_fn=_bad_send, default_timeout=0.05)
        # Should raise ApprovalTimeout (flow continued), not OSError.
        with pytest.raises(ApprovalTimeout):
            gate.request("action despite bad send")

    def test_send_fn_receives_reason_in_message(self):
        messages: list[str] = []
        gate = ApprovalGate(send_fn=messages.append, default_timeout=0.05)
        with pytest.raises(ApprovalTimeout):
            gate.request("action", reason="because it is dangerous")
        assert "because it is dangerous" in messages[0]


class TestApprovalGateConcurrency:
    def test_two_concurrent_requests_are_independent(self):
        """Approving one request must not affect the other."""
        gate = ApprovalGate(default_timeout=3.0)
        t0, outcome0 = _request_in_thread(gate, action="alpha")
        t1, outcome1 = _request_in_thread(gate, action="beta")
        pending = _wait_for_pending(gate, count=2)
        assert len(pending) == 2

        id_alpha = next(e["id"] for e in pending if e["action"] == "alpha")
        id_beta = next(e["id"] for e in pending if e["action"] == "beta")

        gate.handle_response(f"approve {id_alpha}")
        gate.handle_response(f"deny {id_beta}")

        t0.join(timeout=2.0)
        t1.join(timeout=2.0)
        assert outcome0.get("approved") is True
        assert outcome1.get("denied") is True

    def test_three_concurrent_requests_all_resolved(self):
        gate = ApprovalGate(default_timeout=3.0)
        threads_outcomes = [
            _request_in_thread(gate, action=f"task-{i}") for i in range(3)
        ]
        pending = _wait_for_pending(gate, count=3, retries=200)
        assert len(pending) == 3

        for entry in pending:
            gate.handle_response(f"approve {entry['id']}")

        for t, outcome in threads_outcomes:
            t.join(timeout=2.0)
            assert outcome.get("approved") is True

        assert gate.list_pending() == []

    def test_sequential_requests_each_cleaned_up_between_runs(self):
        """Running two sequential requests leaves list_pending() empty after each."""
        gate = ApprovalGate(default_timeout=2.0)
        for i in range(2):
            t, outcome = _request_in_thread(gate, action=f"seq-{i}")
            pending = _wait_for_pending(gate)
            gate.handle_response(f"approve {pending[0]['id']}")
            t.join(timeout=2.0)
            assert outcome.get("approved") is True
            assert gate.list_pending() == []
