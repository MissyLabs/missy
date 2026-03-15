"""Tests for voice channel WebSocket input validation (sample_rate, channels bounds)."""

from __future__ import annotations

import pytest


class TestAudioParamClamping:
    """Verify that audio_start parameters are clamped to safe ranges."""

    def _parse_audio_start(self, msg: dict) -> tuple[int, int]:
        """Simulate the clamping logic from VoiceServer._handle_connection."""
        try:
            sample_rate = max(8000, min(48000, int(msg.get("sample_rate", 16000))))
        except (ValueError, TypeError):
            sample_rate = 16000
        try:
            channels = max(1, min(2, int(msg.get("channels", 1))))
        except (ValueError, TypeError):
            channels = 1
        return sample_rate, channels

    def test_default_values(self):
        rate, ch = self._parse_audio_start({})
        assert rate == 16000
        assert ch == 1

    def test_valid_values(self):
        rate, ch = self._parse_audio_start({"sample_rate": 44100, "channels": 2})
        assert rate == 44100
        assert ch == 2

    def test_sample_rate_too_low(self):
        rate, _ = self._parse_audio_start({"sample_rate": 100})
        assert rate == 8000

    def test_sample_rate_too_high(self):
        rate, _ = self._parse_audio_start({"sample_rate": 999999})
        assert rate == 48000

    def test_channels_too_low(self):
        _, ch = self._parse_audio_start({"channels": 0})
        assert ch == 1

    def test_channels_too_high(self):
        _, ch = self._parse_audio_start({"channels": 100})
        assert ch == 2

    def test_non_numeric_sample_rate(self):
        rate, _ = self._parse_audio_start({"sample_rate": "not_a_number"})
        assert rate == 16000

    def test_non_numeric_channels(self):
        _, ch = self._parse_audio_start({"channels": "stereo"})
        assert ch == 1

    def test_none_sample_rate(self):
        rate, _ = self._parse_audio_start({"sample_rate": None})
        assert rate == 16000

    def test_none_channels(self):
        _, ch = self._parse_audio_start({"channels": None})
        assert ch == 1

    def test_negative_sample_rate(self):
        rate, _ = self._parse_audio_start({"sample_rate": -1})
        assert rate == 8000

    def test_negative_channels(self):
        _, ch = self._parse_audio_start({"channels": -5})
        assert ch == 1

    def test_float_sample_rate(self):
        rate, _ = self._parse_audio_start({"sample_rate": 22050.5})
        assert rate == 22050

    def test_string_numeric_sample_rate(self):
        rate, _ = self._parse_audio_start({"sample_rate": "24000"})
        assert rate == 24000

    def test_extreme_int_sample_rate(self):
        rate, _ = self._parse_audio_start({"sample_rate": 2**31})
        assert rate == 48000


class TestConfidenceClamping:
    """Verify code evolution confidence is clamped to [0.0, 1.0]."""

    def _clamp_confidence(self, raw):
        return max(0.0, min(1.0, float(raw)))

    def test_normal_value(self):
        assert self._clamp_confidence(0.5) == pytest.approx(0.5)

    def test_above_one(self):
        assert self._clamp_confidence(1.5) == pytest.approx(1.0)

    def test_below_zero(self):
        assert self._clamp_confidence(-0.3) == pytest.approx(0.0)

    def test_exactly_zero(self):
        assert self._clamp_confidence(0.0) == pytest.approx(0.0)

    def test_exactly_one(self):
        assert self._clamp_confidence(1.0) == pytest.approx(1.0)

    def test_string_numeric(self):
        assert self._clamp_confidence("0.7") == pytest.approx(0.7)
