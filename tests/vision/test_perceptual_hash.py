"""Tests for perceptual hashing (aHash) and Hamming distance."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from missy.vision.scene_memory import (
    SceneSession,
    compute_phash,
    hamming_distance,
)


class TestComputePhash:
    """Tests for the compute_phash function."""

    def test_returns_16_char_hex(self) -> None:
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        h = compute_phash(img)
        assert len(h) == 16
        int(h, 16)  # must be valid hex

    def test_deterministic(self) -> None:
        img = np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8)
        assert compute_phash(img) == compute_phash(img)

    def test_similar_images_low_distance(self) -> None:
        """Slightly modified images should produce similar hashes."""
        rng = np.random.RandomState(99)
        img = rng.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        # Add slight noise
        noisy = img.copy()
        noise = rng.randint(-5, 6, img.shape).astype(np.int16)
        noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        h1 = compute_phash(img)
        h2 = compute_phash(noisy)
        dist = hamming_distance(h1, h2)
        assert dist >= 0
        assert dist <= 10  # very similar

    def test_different_images_high_distance(self) -> None:
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        h1 = compute_phash(black)
        h2 = compute_phash(white)
        # Both uniform → both get special uniform hash, but different values
        assert h1 != h2

    def test_uniform_black(self) -> None:
        """Uniform black image should get intensity-based hash."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        h = compute_phash(img)
        assert len(h) == 16
        # Black = intensity 0 → "00" repeated 8 times
        assert h == "0000000000000000"

    def test_uniform_white(self) -> None:
        """Uniform white image should get intensity-based hash."""
        img = np.full((50, 50, 3), 255, dtype=np.uint8)
        h = compute_phash(img)
        assert len(h) == 16
        assert h == "ffffffffffffffff"

    def test_uniform_gray(self) -> None:
        """Uniform gray image should get intensity-based hash."""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        h = compute_phash(img)
        assert len(h) == 16
        assert h == "8080808080808080"

    def test_grayscale_input(self) -> None:
        """2D grayscale image should still produce valid hash."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        h = compute_phash(img)
        assert len(h) == 16

    def test_fallback_on_cv2_error(self) -> None:
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        with patch("cv2.resize", side_effect=Exception("broken")):
            h = compute_phash(img)
            assert len(h) == 16
            assert h != "unknown_hash"

    def test_ultimate_fallback(self) -> None:
        from unittest.mock import MagicMock

        img = MagicMock()
        img.tobytes.side_effect = Exception("corrupted")
        with patch("cv2.resize", side_effect=Exception("broken")):
            h = compute_phash(img)
            assert h == "unknown_hash"

    def test_small_image(self) -> None:
        """Image smaller than 8x8 should still work (cv2.resize upscales)."""
        img = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        h = compute_phash(img)
        assert len(h) == 16


class TestHammingDistance:
    """Tests for the hamming_distance function."""

    def test_identical_hashes(self) -> None:
        assert hamming_distance("abcdef0123456789", "abcdef0123456789") == 0

    def test_one_bit_differ(self) -> None:
        # 0x0000000000000000 vs 0x0000000000000001 → 1 bit
        assert hamming_distance("0000000000000000", "0000000000000001") == 1

    def test_all_bits_differ(self) -> None:
        # 0x0000000000000000 vs 0xFFFFFFFFFFFFFFFF → 64 bits
        assert hamming_distance("0000000000000000", "ffffffffffffffff") == 64

    def test_partial_difference(self) -> None:
        # 0xFF vs 0x00 in last byte = 8 bits
        assert hamming_distance("00000000000000ff", "0000000000000000") == 8

    def test_different_lengths_returns_neg1(self) -> None:
        assert hamming_distance("abc", "abcdef") == -1

    def test_invalid_hex_returns_neg1(self) -> None:
        assert hamming_distance("not_hex!", "not_hex!") == -1

    def test_empty_strings(self) -> None:
        assert hamming_distance("", "") == -1  # not valid hex

    def test_none_input(self) -> None:
        assert hamming_distance(None, "abc") == -1  # type: ignore[arg-type]


class TestSceneSessionPhashIntegration:
    """Test that SceneSession uses perceptual hash for change detection."""

    def test_detect_change_uses_phash(self) -> None:
        """Change detection should incorporate perceptual hash distance."""
        session = SceneSession("test")
        # Create two noticeably different images
        img_a = np.zeros((64, 64, 3), dtype=np.uint8)
        img_b = np.full((64, 64, 3), 200, dtype=np.uint8)

        f_a = session.add_frame(img_a)
        f_b = session.add_frame(img_b)

        change = session.detect_change(f_a, f_b)
        assert change.change_score > 0.1
        assert change.description != "no change"

    def test_no_change_for_identical_frames(self) -> None:
        session = SceneSession("test")
        rng = np.random.RandomState(42)
        img = rng.randint(50, 200, (64, 64, 3), dtype=np.uint8)

        f_a = session.add_frame(img.copy(), deduplicate=False)
        f_b = session.add_frame(img.copy(), deduplicate=False)

        change = session.detect_change(f_a, f_b)
        assert change.change_score < 0.05
        assert change.description == "no change"

    def test_frame_hash_is_phash_format(self) -> None:
        """Frames should have 16-char perceptual hashes (not old 12-char md5)."""
        rng = np.random.RandomState(7)
        img = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        session = SceneSession("test")
        frame = session.add_frame(img)
        assert len(frame.thumbnail_hash) == 16
