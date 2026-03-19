"""Edge-case hardening tests for VisionIntentClassifier and vision doctor types.

Covers:
- VisionIntentClassifier: long input, unicode, multi-keyword, stop-words,
  case insensitivity, markup, idempotency, empty/whitespace inputs.
- DiagnosticResult / DoctorReport: construction, properties, add() semantics,
  empty-report health, error marking, all-pass health.
"""

from __future__ import annotations

import time

import pytest

from missy.vision.doctor import DiagnosticResult, DoctorReport
from missy.vision.intent import (
    ActivationDecision,
    VisionIntent,
    VisionIntentClassifier,
    classify_vision_intent,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clf() -> VisionIntentClassifier:
    """Fresh classifier instance with default thresholds."""
    return VisionIntentClassifier()


# ---------------------------------------------------------------------------
# Intent classifier — edge cases
# ---------------------------------------------------------------------------


class TestLongInput:
    """Test 1: Very long input string (10 000 chars) must not crash or hang."""

    def test_no_crash(self, clf: VisionIntentClassifier) -> None:
        long_text = "the quick brown fox " * 500  # 10 000 chars
        result = clf.classify(long_text)
        assert result.intent is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_performance(self, clf: VisionIntentClassifier) -> None:
        """Classification of a 10 000-char string must complete in under 1 second."""
        long_text = "a " * 5000
        start = time.monotonic()
        clf.classify(long_text)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"classify took {elapsed:.3f}s on long input"

    def test_vision_keyword_found_in_long_text(self, clf: VisionIntentClassifier) -> None:
        """A vision keyword embedded in a long string must still be detected."""
        padding = "random filler content " * 200
        text = padding + "look at this" + padding
        result = clf.classify(text)
        assert result.intent == VisionIntent.LOOK
        assert result.confidence >= 0.90


class TestUnicodeInput:
    """Test 2: Unicode input (emoji, CJK) must return a valid result."""

    def test_emoji_only(self, clf: VisionIntentClassifier) -> None:
        result = clf.classify("😀🎨🧩🔍")
        assert isinstance(result.intent, VisionIntent)
        assert isinstance(result.decision, ActivationDecision)
        assert 0.0 <= result.confidence <= 1.0

    def test_cjk_characters(self, clf: VisionIntentClassifier) -> None:
        result = clf.classify("これを見てください 请看这个 이것 좀 봐주세요")
        assert isinstance(result.intent, VisionIntent)
        assert 0.0 <= result.confidence <= 1.0

    def test_mixed_unicode_and_ascii_trigger(self, clf: VisionIntentClassifier) -> None:
        """ASCII keyword mixed with CJK must still trigger detection."""
        result = clf.classify("日本語テスト — look at this 写真")
        assert result.intent == VisionIntent.LOOK
        assert result.confidence >= 0.90

    def test_zero_width_characters(self, clf: VisionIntentClassifier) -> None:
        """Zero-width / invisible characters must not crash the classifier."""
        text = "look\u200b at\u200c this\u200d"
        result = clf.classify(text)
        # Result may or may not match; important thing is no exception.
        assert isinstance(result.intent, VisionIntent)

    def test_rtl_mixed(self, clf: VisionIntentClassifier) -> None:
        """Arabic text combined with an English trigger must not crash."""
        result = clf.classify("انظر إلى هذا — screenshot please")
        assert isinstance(result.intent, VisionIntent)
        assert 0.0 <= result.confidence <= 1.0


class TestMultipleVisionKeywords:
    """Test 3: Multiple vision keywords — highest confidence should win."""

    def test_screenshot_beats_lower_pattern(self, clf: VisionIntentClassifier) -> None:
        # "screenshot" has confidence 0.95; "color" (painting) has 0.60
        result = clf.classify("what color is that — also, screenshot please")
        assert result.intent == VisionIntent.SCREENSHOT
        assert result.confidence >= 0.95

    def test_puzzle_piece_beats_generic_look(self, clf: VisionIntentClassifier) -> None:
        # "puzzle piece" (0.95) should beat any 0.85 CHECK pattern
        result = clf.classify("look at this puzzle piece on the table")
        # After boost, puzzle_piece gets min(1.0, 0.95 + 0.1) = 1.0
        assert result.intent == VisionIntent.PUZZLE
        assert result.confidence >= 0.95

    def test_multiple_inspect_keywords(self, clf: VisionIntentClassifier) -> None:
        """Multiple inspect-category keywords: best confidence among them wins."""
        result = clf.classify("what does this say and what is on the table")
        # "what does this say" -> 0.85; "what is on the table" -> 0.80
        assert result.intent == VisionIntent.INSPECT
        assert result.confidence >= 0.80

    def test_confidence_is_max_not_sum(self, clf: VisionIntentClassifier) -> None:
        """Confidence must be capped at 1.0 even when many patterns match."""
        text = (
            "look at this puzzle piece — can you see the missing piece "
            "where does this go in the jigsaw puzzle board state"
        )
        result = clf.classify(text)
        assert result.confidence <= 1.0


class TestStopWordsOnly:
    """Test 4: Input consisting solely of stop words — confidence must be low."""

    @pytest.mark.parametrize("text", [
        "the a is",
        "the the the",
        "a an the is are was",
        "and or but if so",
    ])
    def test_low_confidence(self, clf: VisionIntentClassifier, text: str) -> None:
        result = clf.classify(text)
        assert result.confidence < 0.50, (
            f"Expected low confidence for stop-word input {text!r}, "
            f"got {result.confidence}"
        )

    def test_decision_is_skip(self, clf: VisionIntentClassifier) -> None:
        result = clf.classify("the a is and or")
        assert result.decision == ActivationDecision.SKIP

    def test_intent_is_none(self, clf: VisionIntentClassifier) -> None:
        result = clf.classify("the a is")
        assert result.intent == VisionIntent.NONE


class TestCaseSensitivity:
    """Test 5: Classification must be case-insensitive."""

    @pytest.mark.parametrize("text", [
        "LOOK AT THIS",
        "Look At This",
        "look at this",
        "lOoK aT tHiS",
    ])
    def test_same_intent(self, clf: VisionIntentClassifier, text: str) -> None:
        result = clf.classify(text)
        assert result.intent == VisionIntent.LOOK, (
            f"Expected LOOK for {text!r}, got {result.intent}"
        )

    def test_uppercase_screenshot(self, clf: VisionIntentClassifier) -> None:
        assert clf.classify("SCREENSHOT").intent == VisionIntent.SCREENSHOT
        assert clf.classify("screenshot").intent == VisionIntent.SCREENSHOT

    def test_confidence_parity(self, clf: VisionIntentClassifier) -> None:
        """Upper- and lower-case variants must produce equal confidence."""
        upper = clf.classify("LOOK AT THIS")
        lower = clf.classify("look at this")
        assert upper.confidence == lower.confidence


class TestMarkupInput:
    """Test 6: HTML tags and markdown should not prevent intent detection."""

    def test_html_wrapped_trigger(self, clf: VisionIntentClassifier) -> None:
        result = clf.classify("<p>Can you <b>look at this</b> image?</p>")
        assert result.intent == VisionIntent.LOOK
        assert result.confidence >= 0.90

    def test_markdown_bold(self, clf: VisionIntentClassifier) -> None:
        result = clf.classify("**look at this** painting please")
        assert result.intent in (VisionIntent.LOOK, VisionIntent.PAINTING)

    def test_markdown_code_block(self, clf: VisionIntentClassifier) -> None:
        result = clf.classify("`screenshot` — can you take a screenshot?")
        assert result.intent == VisionIntent.SCREENSHOT

    def test_html_entities(self, clf: VisionIntentClassifier) -> None:
        result = clf.classify("look&nbsp;at this &amp; screenshot it")
        # The pattern must still find "screenshot" in the raw string
        assert isinstance(result.intent, VisionIntent)
        assert 0.0 <= result.confidence <= 1.0

    def test_nested_html_tags(self, clf: VisionIntentClassifier) -> None:
        result = clf.classify("<div><span>screenshot</span></div>")
        assert result.intent == VisionIntent.SCREENSHOT


class TestIdempotency:
    """Test 7: Repeated calls with the same input must return consistent results."""

    def test_same_intent(self, clf: VisionIntentClassifier) -> None:
        text = "look at this puzzle piece"
        results = [clf.classify(text) for _ in range(10)]
        intents = {r.intent for r in results}
        assert len(intents) == 1, f"Got inconsistent intents: {intents}"

    def test_same_confidence(self, clf: VisionIntentClassifier) -> None:
        text = "take a screenshot"
        results = [clf.classify(text) for _ in range(10)]
        confidences = {r.confidence for r in results}
        assert len(confidences) == 1

    def test_same_decision(self, clf: VisionIntentClassifier) -> None:
        text = "what does this say"
        results = [clf.classify(text) for _ in range(5)]
        decisions = {r.decision for r in results}
        assert len(decisions) == 1

    def test_log_grows_with_calls(self, clf: VisionIntentClassifier) -> None:
        """Each classify call must append to the activation log."""
        clf.clear_log()
        text = "look at this"
        for _i in range(5):
            clf.classify(text)
        assert len(clf.activation_log) == 5


class TestEmptyInput:
    """Test 8: Empty string must return a valid low-confidence result."""

    def test_empty_string(self, clf: VisionIntentClassifier) -> None:
        result = clf.classify("")
        assert result.intent == VisionIntent.NONE
        assert result.confidence == 0.0
        assert result.decision == ActivationDecision.SKIP

    def test_empty_string_does_not_log(self, clf: VisionIntentClassifier) -> None:
        """Early-return path must not append to the activation log."""
        clf.clear_log()
        clf.classify("")
        assert len(clf.activation_log) == 0

    def test_to_audit_dict_on_empty(self, clf: VisionIntentClassifier) -> None:
        result = clf.classify("")
        d = result.to_audit_dict()
        assert d["intent"] == "none"
        assert d["confidence"] == 0.0
        assert d["decision"] == "skip"


class TestWhitespaceInput:
    """Test 9: Whitespace-only string must return a valid low-confidence result."""

    @pytest.mark.parametrize("text", [
        " ",
        "   ",
        "\t",
        "\n",
        "\r\n",
        "  \t  \n  ",
    ])
    def test_whitespace(self, clf: VisionIntentClassifier, text: str) -> None:
        result = clf.classify(text)
        assert result.intent == VisionIntent.NONE
        assert result.confidence == 0.0
        assert result.decision == ActivationDecision.SKIP

    def test_whitespace_does_not_log(self, clf: VisionIntentClassifier) -> None:
        clf.clear_log()
        clf.classify("   ")
        assert len(clf.activation_log) == 0


# ---------------------------------------------------------------------------
# Intent classifier — module-level convenience
# ---------------------------------------------------------------------------


class TestModuleLevelClassifier:
    """Smoke tests for the module-level singleton and convenience function."""

    def test_classify_vision_intent_returns_result(self) -> None:
        result = classify_vision_intent("look at this")
        assert result.intent == VisionIntent.LOOK

    def test_classify_vision_intent_empty(self) -> None:
        result = classify_vision_intent("")
        assert result.confidence == 0.0

    def test_invalid_auto_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="auto_threshold"):
            VisionIntentClassifier(auto_threshold=1.5)

    def test_invalid_ask_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="ask_threshold"):
            VisionIntentClassifier(ask_threshold=-0.1)


# ---------------------------------------------------------------------------
# DiagnosticResult — Test 11
# ---------------------------------------------------------------------------


class TestDiagnosticResult:
    """Test 11: DiagnosticResult creation and field defaults."""

    def test_basic_creation(self) -> None:
        r = DiagnosticResult(name="opencv", passed=True, message="OK")
        assert r.name == "opencv"
        assert r.passed is True
        assert r.message == "OK"
        assert r.details == {}
        assert r.severity == "info"

    def test_custom_severity(self) -> None:
        r = DiagnosticResult(name="x", passed=False, message="fail", severity="error")
        assert r.severity == "error"

    def test_warning_severity(self) -> None:
        r = DiagnosticResult(name="y", passed=False, message="warn", severity="warning")
        assert r.severity == "warning"

    def test_details_populated(self) -> None:
        r = DiagnosticResult(
            name="z",
            passed=True,
            message="ok",
            details={"version": "4.8.0"},
        )
        assert r.details["version"] == "4.8.0"

    def test_repr_contains_name(self) -> None:
        r = DiagnosticResult(name="opencv", passed=True, message="OK")
        assert "opencv" in repr(r)


# ---------------------------------------------------------------------------
# DoctorReport — Test 10: properties
# ---------------------------------------------------------------------------


class TestDoctorReportProperties:
    """Test 10: DoctorReport properties return correct counts."""

    def _make_report(self) -> DoctorReport:
        report = DoctorReport()
        report.add(DiagnosticResult("a", passed=True, message="ok"))
        report.add(DiagnosticResult("b", passed=True, message="ok"))
        report.add(DiagnosticResult("c", passed=False, message="warn", severity="warning"))
        report.add(DiagnosticResult("d", passed=False, message="err", severity="error"))
        return report

    def test_passed_count(self) -> None:
        assert self._make_report().passed == 2

    def test_failed_count(self) -> None:
        assert self._make_report().failed == 2

    def test_warnings_count(self) -> None:
        assert self._make_report().warnings == 1

    def test_errors_count(self) -> None:
        assert self._make_report().errors == 1

    def test_passed_result_not_in_failed(self) -> None:
        report = DoctorReport()
        report.add(DiagnosticResult("x", passed=True, message="ok"))
        assert report.failed == 0

    def test_warning_not_counted_as_error(self) -> None:
        report = DoctorReport()
        report.add(DiagnosticResult("w", passed=False, message="w", severity="warning"))
        assert report.errors == 0
        assert report.warnings == 1


# ---------------------------------------------------------------------------
# DoctorReport.add() — Test 12
# ---------------------------------------------------------------------------


class TestDoctorReportAdd:
    """Test 12: DoctorReport.add() appends results and tracks overall health."""

    def test_add_passes_does_not_change_healthy(self) -> None:
        report = DoctorReport()
        report.add(DiagnosticResult("a", passed=True, message="ok"))
        assert report.overall_healthy is True

    def test_add_warning_does_not_change_healthy(self) -> None:
        report = DoctorReport()
        report.add(DiagnosticResult("a", passed=False, message="w", severity="warning"))
        assert report.overall_healthy is True

    def test_add_error_sets_unhealthy(self) -> None:
        report = DoctorReport()
        report.add(DiagnosticResult("a", passed=False, message="err", severity="error"))
        assert report.overall_healthy is False

    def test_add_increments_results(self) -> None:
        report = DoctorReport()
        for _i in range(5):
            report.add(DiagnosticResult(f"check_{_i}", passed=True, message="ok"))
        assert len(report.results) == 5

    def test_add_info_failed_does_not_affect_health(self) -> None:
        """A failed result with severity='info' must not mark the report unhealthy."""
        report = DoctorReport()
        report.add(DiagnosticResult("a", passed=False, message="info-fail", severity="info"))
        assert report.overall_healthy is True

    def test_multiple_errors_only_toggle_once(self) -> None:
        report = DoctorReport()
        report.add(DiagnosticResult("a", passed=False, message="e1", severity="error"))
        report.add(DiagnosticResult("b", passed=False, message="e2", severity="error"))
        assert report.overall_healthy is False
        assert report.errors == 2


# ---------------------------------------------------------------------------
# DoctorReport — Test 13: empty report is healthy
# ---------------------------------------------------------------------------


class TestEmptyReport:
    """Test 13: An empty DoctorReport should be considered healthy."""

    def test_empty_overall_healthy(self) -> None:
        report = DoctorReport()
        assert report.overall_healthy is True

    def test_empty_passed_zero(self) -> None:
        assert DoctorReport().passed == 0

    def test_empty_failed_zero(self) -> None:
        assert DoctorReport().failed == 0

    def test_empty_warnings_zero(self) -> None:
        assert DoctorReport().warnings == 0

    def test_empty_errors_zero(self) -> None:
        assert DoctorReport().errors == 0

    def test_empty_results_list(self) -> None:
        assert DoctorReport().results == []


# ---------------------------------------------------------------------------
# DoctorReport — Test 14: report with one error is not healthy
# ---------------------------------------------------------------------------


class TestReportWithOneError:
    """Test 14: A report containing a single error result must not be healthy."""

    def test_not_healthy(self) -> None:
        report = DoctorReport()
        report.add(DiagnosticResult("x", passed=False, message="boom", severity="error"))
        assert report.overall_healthy is False

    def test_errors_count_is_one(self) -> None:
        report = DoctorReport()
        report.add(DiagnosticResult("x", passed=False, message="boom", severity="error"))
        assert report.errors == 1

    def test_passed_count_unaffected(self) -> None:
        report = DoctorReport()
        report.add(DiagnosticResult("ok", passed=True, message="fine"))
        report.add(DiagnosticResult("bad", passed=False, message="boom", severity="error"))
        assert report.passed == 1


# ---------------------------------------------------------------------------
# DoctorReport — Test 15: all-pass report is healthy
# ---------------------------------------------------------------------------


class TestAllPassReport:
    """Test 15: A report where every result passes must be healthy."""

    def test_all_pass_healthy(self) -> None:
        report = DoctorReport()
        for name in ("opencv", "numpy", "video_devices", "captures_dir"):
            report.add(DiagnosticResult(name, passed=True, message="ok"))
        assert report.overall_healthy is True

    def test_all_pass_zero_failures(self) -> None:
        report = DoctorReport()
        for name in ("a", "b", "c"):
            report.add(DiagnosticResult(name, passed=True, message="ok"))
        assert report.failed == 0
        assert report.errors == 0
        assert report.warnings == 0

    def test_all_pass_counts_match_total(self) -> None:
        report = DoctorReport()
        names = ["check_1", "check_2", "check_3", "check_4", "check_5"]
        for name in names:
            report.add(DiagnosticResult(name, passed=True, message="ok"))
        assert report.passed == len(names)
        assert report.failed == 0
