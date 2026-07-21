"""Tests for missy.tools.builtin.audio_route.

Deep coverage on the two safety guarantees the module docstring promises:
(1) the virtual sink's volume is always capped at creation, never left at
PipeWire's occasionally-100%+ default, and (2) the operator's system
default output is never touched unless ``set_default=True`` is explicit.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from missy.tools.builtin.audio_route import (
    _SAFE_DEFAULT_VOLUME_PCT,
    AudioRouteTtsTool,
    AudioTestRouteTool,
)


def _proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# AudioRouteTtsTool
# ---------------------------------------------------------------------------


class TestAudioRouteTtsTool:
    def test_creates_sink_when_absent(self):
        with (
            patch("shutil.which", return_value="/usr/bin/pactl"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _proc(stdout=""),  # list short sinks: doesn't exist yet
                _proc(),  # load-module
                _proc(),  # set-sink-volume
                _proc(),  # set-sink-mute
            ]
            result = AudioRouteTtsTool().execute()

        assert result.success is True
        assert result.output["created"] is True
        assert result.output["sink_name"] == "missy_tts_out"
        assert result.output["monitor_source"] == "missy_tts_out.monitor"
        load_call = mock_run.call_args_list[1][0][0]
        assert load_call[:3] == ["pactl", "load-module", "module-null-sink"]

    def test_idempotent_when_sink_already_exists(self):
        with (
            patch("shutil.which", return_value="/usr/bin/pactl"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _proc(stdout="1\tmissy_tts_out\tmodule.id\ts16le 2ch 48000Hz\tSUSPENDED"),
                _proc(),  # set-sink-volume
                _proc(),  # set-sink-mute
            ]
            result = AudioRouteTtsTool().execute()

        assert result.success is True
        assert result.output["created"] is False
        # No load-module call was made -- only 3 calls total (list, volume, mute).
        assert mock_run.call_count == 3

    def test_volume_is_always_capped_at_creation(self):
        with (
            patch("shutil.which", return_value="/usr/bin/pactl"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [_proc(stdout=""), _proc(), _proc(), _proc()]
            result = AudioRouteTtsTool().execute()

        assert result.output["volume_pct"] == _SAFE_DEFAULT_VOLUME_PCT
        volume_call = mock_run.call_args_list[2][0][0]
        assert volume_call == [
            "pactl",
            "set-sink-volume",
            "missy_tts_out",
            f"{_SAFE_DEFAULT_VOLUME_PCT}%",
        ]

    def test_does_not_change_system_default_sink_unless_requested(self):
        with (
            patch("shutil.which", return_value="/usr/bin/pactl"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [_proc(stdout=""), _proc(), _proc(), _proc()]
            result = AudioRouteTtsTool().execute(set_default=False)

        assert result.output["set_as_default"] is False
        calls = [c[0][0] for c in mock_run.call_args_list]
        assert not any("set-default-sink" in c for c in calls)

    def test_set_default_true_changes_default_sink(self):
        with (
            patch("shutil.which", return_value="/usr/bin/pactl"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [_proc(stdout=""), _proc(), _proc(), _proc(), _proc()]
            result = AudioRouteTtsTool().execute(set_default=True)

        assert result.output["set_as_default"] is True
        default_call = mock_run.call_args_list[-1][0][0]
        assert default_call == ["pactl", "set-default-sink", "missy_tts_out"]

    def test_pactl_missing_returns_actionable_error(self):
        with patch("shutil.which", return_value=None):
            result = AudioRouteTtsTool().execute()
        assert result.success is False
        assert "pactl" in result.error.lower()

    def test_sink_creation_failure_is_reported(self):
        with (
            patch("shutil.which", return_value="/usr/bin/pactl"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _proc(stdout=""),
                _proc(returncode=1, stderr="module load failed"),
            ]
            result = AudioRouteTtsTool().execute()
        assert result.success is False
        assert "module load failed" in result.error

    def test_custom_sink_name(self):
        with (
            patch("shutil.which", return_value="/usr/bin/pactl"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [_proc(stdout=""), _proc(), _proc(), _proc()]
            result = AudioRouteTtsTool().execute(sink_name="custom_sink")
        assert result.output["sink_name"] == "custom_sink"
        assert result.output["monitor_source"] == "custom_sink.monitor"


# ---------------------------------------------------------------------------
# AudioTestRouteTool
# ---------------------------------------------------------------------------


class TestAudioTestRouteTool:
    def test_missing_sink_reports_error(self):
        with (
            patch("shutil.which", return_value="/usr/bin/pactl"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = _proc(stdout="")  # no matching sink line
            result = AudioTestRouteTool().execute()
        assert result.success is False
        assert "audio_route_tts" in result.error

    def test_healthy_sink_without_playing_test_phrase(self):
        with (
            patch("shutil.which", return_value="/usr/bin/pactl"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _proc(stdout="1\tmissy_tts_out\tmod\tfmt\tSUSPENDED"),  # list short sinks
                _proc(stdout="Volume: 70%"),  # get-sink-volume
                _proc(stdout="Mute: no"),  # get-sink-mute
            ]
            result = AudioTestRouteTool().execute(play_test_phrase=False)

        assert result.success is True
        assert result.output["exists"] is True
        assert result.output["muted"] is False
        assert result.output["test_phrase_played"] is False

    def test_muted_sink_is_reported(self):
        with (
            patch("shutil.which", return_value="/usr/bin/pactl"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _proc(stdout="1\tmissy_tts_out\tmod\tfmt\tSUSPENDED"),
                _proc(stdout="Volume: 70%"),
                _proc(stdout="Mute: yes"),
            ]
            result = AudioTestRouteTool().execute(play_test_phrase=False)

        assert result.output["muted"] is True

    def test_plays_test_phrase_targeted_at_the_sink(self):
        with (
            patch("shutil.which", return_value="/usr/bin/pactl"),
            patch("subprocess.run") as mock_run,
        ):

            def _run_side_effect(*args, **kwargs):
                argv = args[0]
                if argv[:3] == ["pactl", "list", "short"]:
                    return _proc(stdout="1\tmissy_tts_out\tmod\tfmt\tSUSPENDED")
                if argv[:2] == ["pactl", "get-sink-volume"]:
                    return _proc(stdout="Volume: 70%")
                if argv[:2] == ["pactl", "get-sink-mute"]:
                    return _proc(stdout="Mute: no")
                if argv[0] == "espeak-ng":
                    assert kwargs["env"]["PULSE_SINK"] == "missy_tts_out"
                    return _proc()
                raise AssertionError(f"unexpected call: {argv}")

            mock_run.side_effect = _run_side_effect
            result = AudioTestRouteTool().execute(play_test_phrase=True)

        assert result.output["test_phrase_played"] is True

    def test_missing_espeak_skips_gracefully(self):
        with (
            patch(
                "shutil.which",
                side_effect=lambda b: "/usr/bin/pactl" if b == "pactl" else None,
            ),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                _proc(stdout="1\tmissy_tts_out\tmod\tfmt\tSUSPENDED"),
                _proc(stdout="Volume: 70%"),
                _proc(stdout="Mute: no"),
            ]
            result = AudioTestRouteTool().execute(play_test_phrase=True)

        assert result.success is True
        assert result.output["test_phrase_played"] is False
        assert "espeak-ng" in result.output["play_error"]
