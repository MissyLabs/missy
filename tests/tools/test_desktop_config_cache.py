"""PERF-01: desktop tool config loads are cached by file identity."""

from __future__ import annotations

import os
from unittest.mock import patch

from missy.tools.builtin import _desktop_shared


def test_cached_until_file_changes(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("providers: {}\n")
    monkeypatch.setenv("MISSY_CONFIG", str(cfg_file))
    with patch("missy.config.settings.load_config", side_effect=lambda p: object()) as load:
        first = _desktop_shared.load_missy_config()
        for _ in range(5):
            assert _desktop_shared.load_missy_config() is first
        assert load.call_count == 1
        cfg_file.write_text("providers: {}\n# edited\n")
        st = cfg_file.stat()
        os.utime(cfg_file, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        assert _desktop_shared.load_missy_config() is not first
        assert load.call_count == 2


def test_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSY_CONFIG", str(tmp_path / "nope.yaml"))
    assert _desktop_shared.load_missy_config() is None
