"""Security hardening tests for WebcamSource and FileSource.

Covers:
- WebcamSource device path validation (regex allowlist, injection, traversal)
- FileSource path resolution behavior and acquire() guards (empty, missing, too large)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.sources import FileSource, WebcamSource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_cv2(img: np.ndarray | None = None) -> MagicMock:
    """Return a mock cv2 module whose imread() returns *img*."""
    mock_cv2 = MagicMock()
    mock_cv2.imread.return_value = img
    return mock_cv2


def _normal_image(width: int = 640, height: int = 480) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# WebcamSource — device path validation
# ---------------------------------------------------------------------------


class TestWebcamSourceDevicePathValidation:
    """WebcamSource.__init__ must accept only /dev/videoN paths."""

    # --- valid paths ---------------------------------------------------------

    def test_valid_video0_accepted(self):
        """Standard /dev/video0 path is accepted without error."""
        source = WebcamSource("/dev/video0")
        assert source._device_path == "/dev/video0"

    def test_valid_video12_accepted(self):
        """/dev/video12 (two-digit index) is accepted."""
        source = WebcamSource("/dev/video12")
        assert source._device_path == "/dev/video12"

    def test_valid_video99_accepted(self):
        """/dev/video99 (highest two-digit index) is accepted."""
        source = WebcamSource("/dev/video99")
        assert source._device_path == "/dev/video99"

    # --- invalid paths — must raise ValueError --------------------------------

    def test_rejects_video_without_number(self):
        """/dev/video (no trailing digit) must be rejected."""
        with pytest.raises(ValueError, match="/dev/videoN"):
            WebcamSource("/dev/video")

    def test_rejects_etc_passwd(self):
        """/etc/passwd must be rejected — not a video device."""
        with pytest.raises(ValueError, match="/dev/videoN"):
            WebcamSource("/etc/passwd")

    def test_rejects_path_traversal(self):
        """Relative traversal sequence must be rejected."""
        with pytest.raises(ValueError, match="/dev/videoN"):
            WebcamSource("../../../dev/video0")

    def test_rejects_shell_injection(self):
        """Path with shell metacharacters must be rejected."""
        with pytest.raises(ValueError, match="/dev/videoN"):
            WebcamSource("/dev/video0; rm -rf /")

    def test_rejects_empty_string(self):
        """Empty string must be rejected."""
        with pytest.raises(ValueError, match="/dev/videoN"):
            WebcamSource("")

    def test_rejects_block_device(self):
        """/dev/sda1 (block device, not video) must be rejected."""
        with pytest.raises(ValueError, match="/dev/videoN"):
            WebcamSource("/dev/sda1")


# ---------------------------------------------------------------------------
# FileSource — path resolution
# ---------------------------------------------------------------------------


class TestFileSourcePathResolution:
    """FileSource.__init__ resolves paths to absolute form via Path.resolve()."""

    def test_normal_path_is_stored_as_resolved_absolute(self, tmp_path):
        """A plain file path under tmp_path is resolved to an absolute Path."""
        f = tmp_path / "photo.jpg"
        f.write_bytes(b"fake")

        source = FileSource(str(f))

        assert source._path.is_absolute()
        assert source._path == f.resolve()

    def test_dotdot_components_are_collapsed(self, tmp_path):
        """A path containing '..' components is collapsed to its canonical form."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        f = tmp_path / "image.jpg"
        f.write_bytes(b"fake")

        # Construct path with redundant traversal: subdir/../image.jpg
        dotdot_path = str(subdir / ".." / "image.jpg")

        source = FileSource(dotdot_path)

        assert ".." not in source._path.parts
        assert source._path == f.resolve()

    def test_symlink_resolves_to_real_target(self, tmp_path):
        """A symlink path resolves to the symlink's ultimate target."""
        real = tmp_path / "real.jpg"
        real.write_bytes(b"data")
        link = tmp_path / "link.jpg"
        link.symlink_to(real)

        source = FileSource(str(link))

        assert source._path == real.resolve()


# ---------------------------------------------------------------------------
# FileSource — acquire() guards
# ---------------------------------------------------------------------------


class TestFileSourceAcquireGuards:
    """FileSource.acquire() enforces existence, non-empty, and size limits."""

    def test_empty_file_raises_value_error(self, tmp_path):
        """A zero-byte file must raise ValueError containing 'empty'."""
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")

        source = FileSource(str(empty))
        with pytest.raises(ValueError, match="empty"):
            source.acquire()

    def test_nonexistent_file_raises_file_not_found(self):
        """A path that does not exist must raise FileNotFoundError."""
        source = FileSource("/tmp/missy_test_no_such_file_xyzzy.jpg")
        with pytest.raises(FileNotFoundError):
            source.acquire()

    @patch("missy.vision.sources._get_cv2")
    def test_oversized_file_raises_value_error(self, mock_cv2_fn, tmp_path):
        """A file whose st_size exceeds MAX_FILE_SIZE must raise ValueError 'too large'."""
        mock_cv2_fn.return_value = _make_mock_cv2(_normal_image())

        large = tmp_path / "large.png"
        large.write_bytes(b"x")

        mock_stat = MagicMock()
        mock_stat.st_size = FileSource.MAX_FILE_SIZE + 1

        with patch("pathlib.Path.stat", return_value=mock_stat):
            source = FileSource(str(large))
            with pytest.raises(ValueError, match="too large"):
                source.acquire()
