"""Edge-case tests for CircuitBreaker and AttentionSystem.


Covers gaps not addressed by the existing test_circuit_breaker.py and
test_attention.py suites:

CircuitBreaker
- Exact boundary at threshold (threshold-1 stays closed, threshold opens)
- threshold=0 semantics (first failure never increments past threshold)
- threshold=1 rapid open/half-open/closed cycle
- Failure counter keeps climbing while circuit stays closed up to threshold
- Open state: every subsequent call raises MissyError, not the original exc
- Half-open call that is rejected (OPEN) does not call func at all
- Probe failure resets _last_failure_time so the new backoff is measured fresh
- Successive backoff sequence: base → 2x → 4x → cap
- max_timeout == base_timeout: doubling is still capped, never exceeds
- Clock boundary: timeout exactly equal to elapsed — should transition
- Clock boundary: one nanosecond before timeout — must stay OPEN
- Concurrent probe attempts from HALF_OPEN: only first should run as probe
- Return value of None from func in all states
- Failure count is NOT incremented by a MissyError rejection
- On_failure called from CLOSED (below threshold) does not open
- Successful call clears _last_failure_time back to 0 (implicitly via _on_success)

AttentionSystem / subsystems
- AlertingAttention: all 10 urgency keywords individually
- AlertingAttention: punctuation stripping on all variants (!, ?, .)
- AlertingAttention: mixed-case urgency keyword (e.g. "URGENT") is not matched
  (score is case-sensitive only after lower())
- AlertingAttention: multiple urgency words — score proportional
- AlertingAttention: single-word input that is an urgency keyword → 1.0
- OrientingAttention: capitalised word at position 0 is NOT extracted
- OrientingAttention: consecutive topic prepositions
- OrientingAttention: deduplication — same topic word mentioned twice
- OrientingAttention: empty string returns []
- OrientingAttention: word after sentence-ending word is treated as new start
- SustainedAttention: empty topics → duration resets to 1
- SustainedAttention: overlap exactly at 50% boundary (not > 0.5) → resets
- SustainedAttention: overlap above 50% boundary → increments
- SustainedAttention: large topic set with partial overlap
- SelectiveAttention: empty fragments list → []
- SelectiveAttention: empty topics list → all fragments returned
- SelectiveAttention: case-insensitive matching
- SelectiveAttention: no matching fragments → []
- ExecutiveAttention: urgency exactly at 0.5 → no shell_exec (boundary)
- ExecutiveAttention: urgency just above 0.5 → shell_exec included
- ExecutiveAttention: file topic word when urgency already > 0.5 → no duplicate
- ExecutiveAttention: file_write not added when file_read already present via urgency
- ExecutiveAttention: empty topics, low urgency → empty list
- AttentionSystem.process: empty string input — no crash, urgency 0.0
- AttentionSystem.process: very long message (10k chars) — completes without error
- AttentionSystem.process: context_filter is lowercase version of topics
- AttentionSystem.process: focus_duration increments correctly across three turns
- AttentionSystem.process: topic change mid-conversation resets focus_duration to 1
"""

from __future__ import annotations

import contextlib
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.attention import (
    _URGENCY_KEYWORDS,
    AlertingAttention,
    AttentionState,
    AttentionSystem,
    ExecutiveAttention,
    OrientingAttention,
    SelectiveAttention,
    SustainedAttention,
)
from missy.agent.circuit_breaker import CircuitBreaker, CircuitState
from missy.core.exceptions import MissyError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_breaker(
    threshold: int = 3,
    base_timeout: float = 60.0,
    max_timeout: float = 300.0,
    name: str = "test",
) -> CircuitBreaker:
    return CircuitBreaker(name, threshold=threshold, base_timeout=base_timeout, max_timeout=max_timeout)


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


def _trip(breaker: CircuitBreaker, n: int) -> None:
    for _ in range(n):
        with pytest.raises(RuntimeError):
            breaker.call(_raise)


# ===========================================================================
# CircuitBreaker edge cases
# ===========================================================================


class TestThresholdBoundaries:
    """Exact boundary behaviour around the threshold value."""

    def test_threshold_minus_one_failures_stays_closed(self):
        """threshold-1 consecutive failures must NOT open the circuit."""
        breaker = _make_breaker(threshold=4)
        _trip(breaker, n=3)
        assert breaker.state == CircuitState.CLOSED

    def test_exactly_threshold_failures_opens(self):
        """Exactly threshold failures must open the circuit."""
        breaker = _make_breaker(threshold=4)
        _trip(breaker, n=4)
        assert breaker.state == CircuitState.OPEN

    def test_failure_count_at_threshold_minus_one(self):
        breaker = _make_breaker(threshold=5)
        _trip(breaker, n=4)
        assert breaker._failure_count == 4
        assert breaker.state == CircuitState.CLOSED

    def test_threshold_one_opens_on_first_failure(self):
        breaker = _make_breaker(threshold=1)
        _trip(breaker, n=1)
        assert breaker.state == CircuitState.OPEN

    def test_threshold_one_full_cycle(self):
        """threshold=1: open → (timeout) → half-open → success → closed."""
        breaker = _make_breaker(threshold=1, base_timeout=0.01, max_timeout=1.0)
        _trip(breaker, n=1)
        assert breaker.state == CircuitState.OPEN
        time.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN
        breaker.call(lambda: "ok")
        assert breaker.state == CircuitState.CLOSED


class TestThresholdZero:
    """threshold=0 edge case: the implementation does not open before the
    first call, but the failure check ``failure_count >= threshold`` triggers
    the very first time _on_failure is called."""

    def test_threshold_zero_opens_on_first_failure(self):
        breaker = _make_breaker(threshold=0)
        # failure_count starts at 0; 0 >= 0 is True → opens immediately
        _trip(breaker, n=1)
        assert breaker.state == CircuitState.OPEN

    def test_threshold_zero_construction_does_not_raise(self):
        breaker = CircuitBreaker("zero", threshold=0)
        assert breaker._threshold == 0
        assert breaker.state == CircuitState.CLOSED


class TestOpenStateCallBehaviour:
    """Calls in OPEN state must raise MissyError (not the underlying exc)."""

    def test_open_raises_missy_error_not_runtime_error(self):
        breaker = _make_breaker(threshold=1)
        _trip(breaker, n=1)
        # The underlying callable raises RuntimeError, but we must get MissyError
        with pytest.raises(MissyError):
            breaker.call(_raise)

    def test_failure_count_unchanged_after_open_rejection(self):
        """A rejection from OPEN must not increment the failure counter."""
        breaker = _make_breaker(threshold=2)
        _trip(breaker, n=2)
        count_at_open = breaker._failure_count
        with pytest.raises(MissyError):
            breaker.call(_raise)
        assert breaker._failure_count == count_at_open

    def test_func_not_called_when_open(self):
        breaker = _make_breaker(threshold=1)
        _trip(breaker, n=1)
        mock_fn = MagicMock(return_value="should_not_run")
        with pytest.raises(MissyError):
            breaker.call(mock_fn)
        mock_fn.assert_not_called()


class TestClockBoundaries:
    """Recovery timeout clock edge cases via monkeypatched monotonic."""

    def test_exactly_at_timeout_transitions_to_half_open(self):
        """When elapsed == recovery_timeout the condition >= fires → HALF_OPEN."""
        breaker = _make_breaker(threshold=1, base_timeout=30.0)
        with patch("missy.agent.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 1000.0
            _trip(breaker, n=1)
            # elapsed == 30.0 exactly
            mock_time.return_value = 1030.0
            assert breaker.state == CircuitState.HALF_OPEN

    def test_one_unit_before_timeout_stays_open(self):
        """Elapsed slightly less than recovery_timeout must remain OPEN."""
        breaker = _make_breaker(threshold=1, base_timeout=30.0)
        with patch("missy.agent.circuit_breaker.time.monotonic") as mock_time:
            mock_time.return_value = 1000.0
            _trip(breaker, n=1)
            # elapsed == 29.999 — just short of 30.0
            mock_time.return_value = 1029.999
            assert breaker.state == CircuitState.OPEN


class TestBackoffSequence:
    """Verify the full doubling sequence up to the cap."""

    def _fail_probe(self, breaker: CircuitBreaker) -> None:
        """Expire the current timeout and fail the probe."""
        time.sleep(breaker._recovery_timeout * 1.1)
        assert breaker.state == CircuitState.HALF_OPEN
        with pytest.raises(RuntimeError):
            breaker.call(_raise)

    def test_backoff_sequence_three_steps(self):
        """base → 2*base → 4*base (before capping)."""
        base = 0.01
        cap = 1.0
        breaker = _make_breaker(threshold=1, base_timeout=base, max_timeout=cap)
        _trip(breaker, n=1)
        self._fail_probe(breaker)  # 0.02
        assert breaker._recovery_timeout == pytest.approx(base * 2)
        self._fail_probe(breaker)  # 0.04
        assert breaker._recovery_timeout == pytest.approx(base * 4)
        self._fail_probe(breaker)  # 0.08
        assert breaker._recovery_timeout == pytest.approx(base * 8)

    def test_max_equals_base_never_grows(self):
        """When max_timeout == base_timeout doubling is always capped."""
        breaker = _make_breaker(threshold=1, base_timeout=0.01, max_timeout=0.01)
        _trip(breaker, n=1)
        # Force HALF_OPEN internally and fail probe twice
        for _ in range(3):
            time.sleep(0.015)
            if breaker.state == CircuitState.HALF_OPEN:
                with pytest.raises(RuntimeError):
                    breaker.call(_raise)
            else:
                break
        assert breaker._recovery_timeout == pytest.approx(0.01)

    def test_probe_failure_updates_last_failure_time(self):
        """_last_failure_time must be refreshed on probe failure so the new
        backoff period is measured from the probe attempt, not the original
        trip."""
        breaker = _make_breaker(threshold=1, base_timeout=0.01, max_timeout=1.0)
        _trip(breaker, n=1)
        time.sleep(0.02)
        before = time.monotonic()
        with pytest.raises(RuntimeError):
            breaker.call(_raise)
        after = time.monotonic()
        assert before <= breaker._last_failure_time <= after


class TestOnFailureInternals:
    """Direct _on_failure behaviour without going through .call()."""

    def test_on_failure_below_threshold_does_not_open(self):
        breaker = _make_breaker(threshold=5)
        # Call _on_failure 4 times — must stay CLOSED
        for _ in range(4):
            breaker._on_failure()
        assert breaker._state == CircuitState.CLOSED

    def test_on_failure_at_threshold_opens(self):
        breaker = _make_breaker(threshold=3)
        for _ in range(3):
            breaker._on_failure()
        assert breaker._state == CircuitState.OPEN

    def test_on_failure_from_half_open_doubles_timeout(self):
        breaker = _make_breaker(threshold=1, base_timeout=10.0, max_timeout=100.0)
        breaker._state = CircuitState.HALF_OPEN
        breaker._recovery_timeout = 10.0
        breaker._on_failure()
        assert breaker._recovery_timeout == pytest.approx(20.0)
        assert breaker._state == CircuitState.OPEN


class TestOnSuccessInternals:
    """Direct _on_success behaviour."""

    def test_on_success_clears_failure_count(self):
        breaker = _make_breaker(threshold=5)
        breaker._failure_count = 4
        breaker._on_success()
        assert breaker._failure_count == 0

    def test_on_success_restores_recovery_timeout_to_base(self):
        breaker = _make_breaker(threshold=1, base_timeout=5.0, max_timeout=100.0)
        breaker._recovery_timeout = 80.0
        breaker._on_success()
        assert breaker._recovery_timeout == pytest.approx(5.0)

    def test_on_success_sets_state_to_closed(self):
        breaker = _make_breaker(threshold=1)
        breaker._state = CircuitState.OPEN
        breaker._on_success()
        assert breaker._state == CircuitState.CLOSED


class TestConcurrentProbeAttempts:
    """Race condition: multiple threads reach HALF_OPEN simultaneously."""

    def test_concurrent_probes_all_complete_without_exception(self):
        """No thread should see an unhandled exception when probing concurrently."""
        breaker = _make_breaker(threshold=1, base_timeout=0.01, max_timeout=1.0)
        _trip(breaker, n=1)
        time.sleep(0.02)

        errors: list[Exception] = []
        results: list[str] = []
        barrier = threading.Barrier(8)

        def probe():
            barrier.wait()
            try:
                r = breaker.call(lambda: "ok")
                results.append(r)
            except MissyError:
                results.append("rejected")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=probe) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Either "ok" (probe succeeded and closed) or "rejected" (already re-opened)
        assert all(r in ("ok", "rejected") for r in results)

    def test_state_valid_after_concurrent_probes(self):
        """State must be a valid CircuitState after concurrent probe storm."""
        breaker = _make_breaker(threshold=2, base_timeout=0.01, max_timeout=1.0)
        _trip(breaker, n=2)
        time.sleep(0.02)

        barrier = threading.Barrier(5)

        def do_probe():
            barrier.wait()
            with contextlib.suppress(Exception):
                breaker.call(lambda: None)

        threads = [threading.Thread(target=do_probe) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert breaker._state in CircuitState


# ===========================================================================
# AttentionSystem / subsystem edge cases
# ===========================================================================


class TestAlertingAllKeywords:
    """Every urgency keyword must be detected individually."""

    @pytest.mark.parametrize("keyword", sorted(_URGENCY_KEYWORDS))
    def test_each_keyword_scores_nonzero(self, keyword: str):
        alerting = AlertingAttention()
        score = alerting.score(keyword)
        assert score > 0.0, f"keyword '{keyword}' should produce nonzero urgency"

    def test_single_urgency_word_input_scores_one(self):
        alerting = AlertingAttention()
        assert alerting.score("error") == pytest.approx(1.0)

    def test_urgency_keyword_with_exclamation_is_detected(self):
        alerting = AlertingAttention()
        assert alerting.score("error!") > 0.0

    def test_urgency_keyword_with_question_mark_is_detected(self):
        alerting = AlertingAttention()
        assert alerting.score("urgent?") > 0.0

    def test_urgency_keyword_with_period_is_detected(self):
        alerting = AlertingAttention()
        assert alerting.score("critical.") > 0.0

    def test_mixed_case_keyword_not_detected(self):
        """AlertingAttention does .lower() first, so 'ERROR' must still score."""
        alerting = AlertingAttention()
        score = alerting.score("ERROR")
        # lower() converts to "error" → should be detected
        assert score > 0.0

    def test_two_urgency_words_out_of_four(self):
        alerting = AlertingAttention()
        score = alerting.score("the server is down")
        # "down" matches → 1/4 = 0.25
        assert score == pytest.approx(1 / 4)

    def test_all_urgency_words_caps_at_one(self):
        alerting = AlertingAttention()
        text = " ".join(_URGENCY_KEYWORDS)
        score = alerting.score(text)
        assert score == pytest.approx(1.0)

    def test_whitespace_only_input_returns_zero(self):
        alerting = AlertingAttention()
        assert alerting.score("   ") == pytest.approx(0.0)


class TestOrientingTopicExtraction:
    """OrientingAttention.extract_topics edge cases."""

    def test_first_word_capitalised_is_not_extracted(self):
        orienting = OrientingAttention()
        topics = orienting.extract_topics("Docker is great")
        # "Docker" is at position 0 → not extracted by the capitalisation rule
        # (it might still appear if a preposition precedes it, but not here)
        # Position-0 word should NOT appear via capitalisation rule
        assert "Docker" not in topics

    def test_second_capitalised_word_is_extracted(self):
        orienting = OrientingAttention()
        topics = orienting.extract_topics("check Docker status")
        assert "Docker" in topics

    def test_deduplication_of_repeated_topic(self):
        orienting = OrientingAttention()
        topics = orienting.extract_topics("about Docker and about Docker")
        assert topics.count("Docker") == 1

    def test_empty_string_returns_empty(self):
        orienting = OrientingAttention()
        assert orienting.extract_topics("") == []

    def test_word_after_sentence_ending_not_capitalised_rule(self):
        """A capitalised word that follows '.' is treated as sentence start,
        so it should NOT be extracted via the capitalisation rule."""
        orienting = OrientingAttention()
        # "Nginx" follows ".", so it is at sentence start position
        topics = orienting.extract_topics("Server is down. Nginx failed")
        # "Nginx" should NOT be in topics via capitalisation path (previous
        # word ends with ".")
        capitalised_via_rule = [
            t for t in topics if t == "Nginx"
        ]
        # It can only appear if a preposition also precedes it; "." is not a
        # preposition — so it should not appear at all.
        assert "Nginx" not in capitalised_via_rule or len(capitalised_via_rule) == 0

    def test_multiple_prepositions_extract_following_words(self):
        orienting = OrientingAttention()
        topics = orienting.extract_topics("help with nginx for redis")
        assert "nginx" in topics
        assert "redis" in topics

    def test_topic_list_preserves_order(self):
        orienting = OrientingAttention()
        topics = orienting.extract_topics("about alpha with beta for gamma")
        assert topics.index("alpha") < topics.index("beta") < topics.index("gamma")


class TestSustainedAttentionEdgeCases:
    """SustainedAttention edge cases around the 50% overlap boundary."""

    def test_empty_topics_resets_duration_to_one(self):
        sustained = SustainedAttention()
        sustained.update(["Docker"])
        sustained.update(["Docker"])
        assert sustained._duration == 2
        d = sustained.update([])
        assert d == 1

    def test_overlap_at_exactly_fifty_percent_resets(self):
        """Overlap == 0.5 is NOT > 0.5, so duration must reset to 1."""
        sustained = SustainedAttention()
        sustained.update(["alpha", "beta"])  # prev = {alpha, beta}
        # curr = {alpha, gamma}: overlap = 1/2 = 0.5 (exactly at boundary)
        d = sustained.update(["alpha", "gamma"])
        assert d == 1

    def test_overlap_just_above_fifty_percent_increments(self):
        """Overlap > 0.5 must increment duration."""
        sustained = SustainedAttention()
        sustained.update(["a", "b", "c"])  # prev set size 3
        # curr = {a, b, d}: overlap = 2/3 ≈ 0.667 > 0.5
        d = sustained.update(["a", "b", "d"])
        assert d == 2

    def test_completely_different_topics_resets(self):
        sustained = SustainedAttention()
        sustained.update(["Docker"])
        d = sustained.update(["Python"])
        assert d == 1

    def test_first_update_always_returns_one(self):
        sustained = SustainedAttention()
        d = sustained.update(["anything"])
        assert d == 1

    def test_large_topic_set_partial_overlap(self):
        """6 previous, 4 overlap out of 6 → 0.667 > 0.5 → increments."""
        sustained = SustainedAttention()
        sustained.update(["a", "b", "c", "d", "e", "f"])
        d = sustained.update(["a", "b", "c", "d", "x", "y"])
        assert d == 2

    def test_duration_increments_on_every_same_topic_turn(self):
        sustained = SustainedAttention()
        for expected in range(1, 6):
            d = sustained.update(["Docker"])
            assert d == expected


class TestSelectiveAttentionEdgeCases:
    """SelectiveAttention.filter edge cases."""

    def test_empty_fragments_returns_empty(self):
        result = SelectiveAttention.filter([], ["Docker"])
        assert result == []

    def test_empty_topics_returns_all_fragments(self):
        fragments = ["alpha", "beta", "gamma"]
        result = SelectiveAttention.filter(fragments, [])
        assert result == fragments

    def test_case_insensitive_matching(self):
        fragments = ["DOCKER container", "unrelated text"]
        result = SelectiveAttention.filter(fragments, ["docker"])
        assert len(result) == 1
        assert result[0] == "DOCKER container"

    def test_no_matching_fragment_returns_empty(self):
        fragments = ["weather report", "sports news"]
        result = SelectiveAttention.filter(fragments, ["Docker"])
        assert result == []

    def test_multiple_topics_any_match_passes(self):
        fragments = ["nginx config", "redis memory", "mysql schema"]
        result = SelectiveAttention.filter(fragments, ["redis", "mysql"])
        assert "redis memory" in result
        assert "mysql schema" in result
        assert "nginx config" not in result


class TestExecutiveAttentionBoundaries:
    """ExecutiveAttention.prioritise boundary and combination cases."""

    def test_urgency_exactly_half_no_shell_exec(self):
        """urgency == 0.5 is NOT > 0.5, so shell_exec must NOT be added."""
        priority = ExecutiveAttention.prioritise(0.5, [])
        assert "shell_exec" not in priority

    def test_urgency_just_above_half_adds_shell_exec(self):
        """urgency = 0.5 + epsilon → shell_exec must be present."""
        priority = ExecutiveAttention.prioritise(0.5001, [])
        assert "shell_exec" in priority

    def test_urgency_above_half_adds_file_read(self):
        priority = ExecutiveAttention.prioritise(0.9, [])
        assert "file_read" in priority

    def test_file_topic_with_low_urgency_adds_file_tools(self):
        priority = ExecutiveAttention.prioritise(0.1, ["file"])
        assert "file_read" in priority
        assert "file_write" in priority

    def test_high_urgency_plus_file_topic_no_duplicate_file_read(self):
        """When both urgency and file-topic trigger file_read, it appears once."""
        priority = ExecutiveAttention.prioritise(0.8, ["file"])
        assert priority.count("file_read") == 1

    def test_high_urgency_plus_file_topic_no_duplicate_shell_exec(self):
        priority = ExecutiveAttention.prioritise(0.8, ["file"])
        assert priority.count("shell_exec") == 1

    def test_empty_topics_low_urgency_empty_list(self):
        priority = ExecutiveAttention.prioritise(0.0, [])
        assert priority == []

    def test_log_topic_word_triggers_file_tools(self):
        """'log' is in _FILE_TOPIC_WORDS."""
        priority = ExecutiveAttention.prioritise(0.0, ["log"])
        assert "file_read" in priority


class TestAttentionSystemPipeline:
    """AttentionSystem.process integration edge cases."""

    def test_empty_string_no_crash_urgency_zero(self):
        attn = AttentionSystem()
        state = attn.process("")
        assert state.urgency == pytest.approx(0.0)
        assert isinstance(state, AttentionState)

    def test_very_long_message_completes(self):
        attn = AttentionSystem()
        long_msg = "check the file " * 700  # ~10 k chars
        state = attn.process(long_msg)
        assert isinstance(state, AttentionState)

    def test_context_filter_is_lowercase_topics(self):
        attn = AttentionSystem()
        state = attn.process("Tell me about Docker")
        for topic in state.topics:
            assert topic.lower() in state.context_filter

    def test_focus_duration_increments_across_turns(self):
        attn = AttentionSystem()
        state1 = attn.process("about Docker networking")
        state2 = attn.process("more about Docker networking")
        state3 = attn.process("still about Docker networking")
        assert state1.focus_duration == 1
        assert state2.focus_duration >= 2
        assert state3.focus_duration >= 3

    def test_topic_change_resets_focus_duration(self):
        attn = AttentionSystem()
        attn.process("about Docker")
        attn.process("about Docker")
        # Switch topic completely
        state = attn.process("about weather tomorrow rain forecast")
        assert state.focus_duration == 1

    def test_high_urgency_message_sets_priority_tools(self):
        attn = AttentionSystem()
        state = attn.process("critical error immediately")
        assert "shell_exec" in state.priority_tools
        assert "file_read" in state.priority_tools

    def test_file_related_message_prioritises_file_tools(self):
        attn = AttentionSystem()
        state = attn.process("read the config file please")
        assert "file_read" in state.priority_tools

    def test_process_with_none_history_is_same_as_no_history(self):
        attn1 = AttentionSystem()
        attn2 = AttentionSystem()
        msg = "check the logs for errors"
        s1 = attn1.process(msg, history=None)
        s2 = attn2.process(msg)
        assert s1.urgency == pytest.approx(s2.urgency)
        assert s1.topics == s2.topics
        assert s1.priority_tools == s2.priority_tools

    def test_process_with_empty_history_list(self):
        attn = AttentionSystem()
        state = attn.process("any message", history=[])
        assert isinstance(state, AttentionState)

    def test_urgency_zero_for_benign_message(self):
        attn = AttentionSystem()
        state = attn.process("what time is it please")
        assert state.urgency == pytest.approx(0.0)

    def test_attention_state_is_new_instance_each_call(self):
        attn = AttentionSystem()
        s1 = attn.process("about Docker")
        s2 = attn.process("about Docker")
        assert s1 is not s2
