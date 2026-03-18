"""Tests for Discord voice utility functions.

Tests _clean_for_speech markdown stripping and _resample_pcm audio
resampling from missy.channels.discord.voice module.
"""

from __future__ import annotations

import struct

from missy.channels.discord.voice import _clean_for_speech, _resample_pcm

# ---------------------------------------------------------------------------
# _clean_for_speech tests
# ---------------------------------------------------------------------------


class TestCleanForSpeech:
    def test_plain_text_unchanged(self):
        assert _clean_for_speech("Hello world") == "Hello world"

    def test_strips_whitespace(self):
        assert _clean_for_speech("  Hello  ") == "Hello"

    def test_removes_code_blocks(self):
        text = "Here is code:\n```python\nprint('hello')\n```\nEnd."
        result = _clean_for_speech(text)
        assert "```" not in result
        assert "print" not in result
        assert "End." in result

    def test_removes_inline_code(self):
        text = "Run `pip install missy` to install."
        result = _clean_for_speech(text)
        assert "`" not in result
        assert "pip install missy" in result

    def test_removes_bold(self):
        text = "This is **bold** text."
        result = _clean_for_speech(text)
        assert "**" not in result
        assert "bold" in result

    def test_removes_italic(self):
        text = "This is *italic* text."
        result = _clean_for_speech(text)
        assert result.count("*") == 0
        assert "italic" in result

    def test_removes_headers(self):
        text = "# Header\nSome text\n## Sub\nMore text"
        result = _clean_for_speech(text)
        assert "#" not in result
        assert "Header" in result

    def test_removes_markdown_links(self):
        text = "Check [this link](https://example.com) out."
        result = _clean_for_speech(text)
        assert "https://example.com" not in result
        assert "this link" in result

    def test_removes_bare_urls(self):
        text = "Visit https://example.com/path for info."
        result = _clean_for_speech(text)
        assert "https://" not in result

    def test_collapses_multiple_newlines(self):
        text = "First.\n\n\n\n\nSecond."
        result = _clean_for_speech(text)
        assert "\n\n\n" not in result

    def test_truncates_long_text(self):
        text = "A" * 700
        result = _clean_for_speech(text)
        assert len(result) <= 603  # 600 + "..."

    def test_truncates_at_sentence_boundary(self):
        sentences = ". ".join([f"Sentence {i}" for i in range(50)])
        result = _clean_for_speech(sentences)
        assert len(result) <= 601
        assert result.endswith(".")

    def test_empty_input(self):
        assert _clean_for_speech("") == ""

    def test_whitespace_only(self):
        assert _clean_for_speech("   ") == ""

    def test_underscore_italic(self):
        text = "This is _italic_ text."
        result = _clean_for_speech(text)
        assert "italic" in result

    def test_mixed_markdown(self):
        text = "# Title\n**Bold** and *italic* with `code` and [link](url).\n```\nblock\n```"
        result = _clean_for_speech(text)
        assert "#" not in result
        assert "**" not in result
        assert "```" not in result
        assert "Bold" in result


# ---------------------------------------------------------------------------
# _resample_pcm tests
# ---------------------------------------------------------------------------


class TestResamplePcm:
    def test_same_rate_returns_input(self):
        pcm = struct.pack("<10h", *([1000] * 10))
        result = _resample_pcm(pcm, 16000, 16000)
        assert result == pcm

    def test_downsample_48k_to_16k(self):
        # 480 stereo samples at 48kHz = 5ms
        samples = [500] * 480
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        result = _resample_pcm(pcm, 48000, 16000)
        # Should produce fewer samples (downsampled)
        assert len(result) < len(pcm)
        # Should be valid 16-bit samples
        assert len(result) % 2 == 0

    def test_empty_audio(self):
        result = _resample_pcm(b"", 48000, 16000)
        # May return empty or very short result
        assert isinstance(result, bytes)

    def test_minimal_samples(self):
        # 2 samples (1 stereo frame)
        pcm = struct.pack("<2h", 100, -100)
        result = _resample_pcm(pcm, 48000, 16000)
        assert isinstance(result, bytes)

    def test_output_is_valid_pcm(self):
        samples = [i * 100 for i in range(100)]
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        result = _resample_pcm(pcm, 48000, 16000)
        if len(result) >= 2:
            out_count = len(result) // 2
            out_samples = struct.unpack(f"<{out_count}h", result)
            for s in out_samples:
                assert -32768 <= s <= 32767

    def test_clamps_to_int16_range(self):
        # Max positive samples
        samples = [32767] * 100
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        result = _resample_pcm(pcm, 48000, 16000)
        if len(result) >= 2:
            out_count = len(result) // 2
            out_samples = struct.unpack(f"<{out_count}h", result)
            for s in out_samples:
                assert s <= 32767
