"""Requirement-focused agent runtime checks from validation run 19."""

from __future__ import annotations

import unicodedata

from missy.agent.runtime import _fingerprint_tc


class TestTdeep029CanonicalFingerprint:
    def test_reordered_nested_json_and_unicode_forms_are_equivalent(self):
        composed = "café"
        decomposed = unicodedata.normalize("NFD", composed)
        first = {"outer": {"b": [1, composed], "a": True}, "value": None}
        second = {"value": None, "outer": {"a": True, "b": [1, decomposed]}}

        assert _fingerprint_tc("tool", first) == _fingerprint_tc("tool", second)

    def test_distinct_numeric_string_and_list_values_do_not_collide(self):
        values = [
            {"value": 1},
            {"value": "1"},
            {"value": 1.5},
            {"value": [1]},
            {"value": ["1"]},
        ]

        fingerprints = {_fingerprint_tc("tool", value) for value in values}

        assert len(fingerprints) == len(values)

    def test_non_finite_non_json_and_huge_values_fail_safely_and_stably(self):
        invalid_values = [
            {"value": float("nan")},
            {"value": float("inf")},
            {"value": {1, 2}},
            {"value": object()},
            {"value": [0] * 20_000},
        ]

        for value in invalid_values:
            first = _fingerprint_tc("tool", value)
            second = _fingerprint_tc("tool", value)
            assert first == second
            assert len(first) == 16

    def test_secret_values_never_appear_in_fingerprint(self):
        secret = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"

        fingerprint = _fingerprint_tc("tool", {"token": secret})

        assert secret not in fingerprint
        assert len(fingerprint) == 16
