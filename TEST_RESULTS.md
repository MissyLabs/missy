# Test Results

Last updated: 2026-07-07

## Latest Runs

| Command | Result | Notes |
| --- | --- | --- |
| `pytest tests/tools/test_discord_voice_tools.py tests/tools/test_builtin_init_coverage.py tests/channels/test_discord_channel_gap_coverage.py -q` | pass | 48 passed in 0.31s. |
| `pytest tests/unit/test_remaining_gaps.py::TestFasterWhisperSTTLoad::test_load_raises_import_error_when_faster_whisper_missing tests/unit/test_remaining_gaps.py::TestFasterWhisperSTTAutoDevice::test_auto_device_falls_back_to_cpu_when_torch_missing tests/vision/test_pipeline_edge_cases.py::TestPipelineEdgeCases::test_single_pixel_image tests/cli/test_vision_cli.py::TestVisionCapture::test_capture_best_mode -q` | pass | 4 passed in 0.35s after OpenCV/runtime dependency correction. |
| `pytest -q` | fail, then pass | First run failed due missing/incompatible `cv2` and host-installed optional STT packages invalidating missing-dependency tests. After installing compatible OpenCV and hardening optional-dependency tests, rerun passed: 20252 passed, 13 skipped in 361.61s. |
| `ruff check .` | pass | Full-repo lint passed. |
| `ruff format --check .` | pass | 708 files already formatted. |

## Environment Notes

- `python3-opencv` was installed through apt to satisfy the declared vision dependency.
- A user-site `opencv-python-headless 5.0.0.93` wheel was installed because the active user-site NumPy is `2.4.3` and the apt OpenCV extension is compiled against NumPy 1.x.

## Not Yet Run

- No known verification gaps for this session.
