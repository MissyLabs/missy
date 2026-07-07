# Build Results

Last updated: 2026-07-07

## Result

- Repository is coherent and verified after the Discord voice tool bridge work.
- No package metadata or dependency declarations were changed.
- Runtime test environment was updated to satisfy existing declared optional vision dependencies:
  - apt: `python3-opencv`
  - user-site pip: `opencv-python-headless 5.0.0.93`

## Commands

| Command | Result |
| --- | --- |
| `ruff check .` | pass |
| `ruff format --check .` | pass |
| `pytest -q` | pass: 20252 passed, 13 skipped |

## Notes

- The first `pytest -q` run exposed environment drift rather than code failures: `cv2` was missing/incompatible, and optional STT tests assumed `faster_whisper`/`ctranslate2` were absent.
- Optional-dependency tests now force missing imports with `patch.dict("sys.modules", {"package": None})`, making them independent of host package state.
