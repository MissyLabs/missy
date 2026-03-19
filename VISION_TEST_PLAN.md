# Vision Subsystem Test Plan

## Test Coverage Summary

| Module | Tests | Status |
|--------|-------|--------|
| discovery.py | 18 | PASS |
| capture.py | 13 + 27 | PASS |
| sources.py | 27 + 24 | PASS |
| pipeline.py | 6 + 30 | PASS |
| scene_memory.py | 25 + 19 | PASS |
| intent.py | 25 | PASS |
| analysis.py | 20 + 25 | PASS |
| doctor.py | 16 | PASS |
| vision_tools.py | 23 | PASS |
| provider_format.py | 12 | PASS |
| audit.py | 7 + 19 | PASS |
| resilient_capture.py | 9 + 15 | PASS |
| edge_cases.py | 30 | PASS |
| hardening.py | 32 | PASS |
| burst_and_diff.py | 14 | PASS |
| security.py | 13 | PASS |
| integration.py | 12 | PASS |
| timeout_and_backoff.py | 10 | PASS |
| CLI vision commands | 14 | PASS |
| Voice-vision integration | 11 | PASS |
| **Total** | **484** | **ALL PASS** |

## Unit Tests

### Camera Discovery (`test_discovery.py`)
- CameraDevice dataclass: USB ID, Logitech detection, immutability
- Fake sysfs scanning: empty, no devices, metadata node filtering
- Cache TTL behavior and force bypass
- Preferred camera selection: Logitech C922x > known > first
- USB ID and name pattern search
- KNOWN_CAMERAS database verification

### Frame Capture (`test_capture.py` + `test_capture_extended.py`)
- CaptureConfig defaults and custom values
- CaptureResult success/failure states, shape property
- CameraHandle open/close lifecycle (mocked OpenCV)
- Warm-up frame discarding (0, 5, exception-stops-early)
- Blank frame detection (all-black, threshold, boundary, single-channel)
- All-retries-exhausted failure path
- Capture-without-open error, double-close safety
- Context manager cleanup
- Device path index parsing (/dev/video0, /dev/video12, non-video, partial)
- Capture-to-file: directory creation, imwrite failure, save exception, PNG params
- Burst capture: count validation, clamp to 20

### Image Sources (`test_sources.py` + `test_sources_extended.py`)
- ImageFrame auto-dimensions, JPEG/PNG encoding, base64 encoding
- FileSource: type, availability, acquire success, missing file, unreadable, directory check
- PhotoSource: type, availability, scan, file count, wrap-around, empty dir
- PhotoSource pattern filtering: default *, glob *.jpg, recursive **/*.png
- WebcamSource: type, unavailable device, timeout protection
- ScreenshotSource: type, tool availability (scrot, none)
- Factory function: all types, missing args, unknown type

### Pipeline (`test_pipeline.py` + `test_pipeline_extended.py`)
- PipelineConfig defaults
- Resize: small (no-op), large (downscale), zero/negative rejection
- Quality assessment: bright, dark, blurry, overexposed, low-contrast, poor (multi-issue)
- Single-channel 3D images (H,W,1): assess_quality, normalize_exposure
- Grayscale 2D images: no cvtColor call
- BGRA 4-channel images: alpha channel handling
- CLAHE: grayscale, BGR (LAB conversion), BGRA (alpha preservation)
- Denoise: fastNlMeansDenoisingColored parameter verification
- Sharpen: unsharp masking weights (1.5 / -0.5)
- Full process(): all steps enabled, all steps disabled
- Input validation: None, empty, 1D array

### Scene Memory (`test_scene_memory.py` + `test_scene_memory_extended.py`)
- SceneFrame: creation, hash stability, hash uniqueness
- Hash fallback: cv2 failure, tobytes failure (ultimate "unknown_hash" fallback)
- SceneSession: create, add frames, eviction at limit, latest/recent frames
- Observations, state updates, close (full cleanup), summarize after close
- Change detection: significant change, no change, insufficient frames
- SceneManager: create, get, active session, close, close_all, eviction, list
- Eviction: inactive preferred, oldest by creation timestamp (not dict order)

### Intent Detection (`test_intent.py`)
- Explicit vision requests: look, show, photo, screenshot
- Puzzle intents: piece placement, edge, sky section, sort, jigsaw
- Painting intents: feedback, improve, canvas
- Inspect intents: read, what's on table
- No-vision intents: general questions, coding requests
- Empty/whitespace input
- Activation thresholds: high, medium, low confidence
- Activation log and clear

### Analysis (`test_analysis.py` + `test_analysis_extended.py`)
- Prompt builder: general, puzzle, painting, inspection modes
- Context injection for all modes
- Follow-up prompts with previous observations
- Painting prompt tone: warm, encouraging, no harsh language
- PuzzlePreprocessor: edge enhancement, color extraction
- Color description: black, white, red, green, blue, yellow, gray, fallback
- AnalysisMode enum values and string construction

### Doctor (`test_doctor.py`)
- DiagnosticResult passed/failed states
- DoctorReport: empty, add passed, add error, add warning
- Individual checks: numpy, video group, video devices, sysfs, screenshot tools, opencv
- Captures directory writable check
- run_all completeness

### Audit (`test_audit.py` + `test_audit_extended.py`)
- All 7 audit functions: capture, analysis, intent, discovery, session, burst, error
- Event emission with correct category/action/details
- No-logger-available graceful handling
- Logger exception graceful handling
- Privacy: text not logged in intent events (only text_length)
- Confidence rounding
- Session lifecycle (create/close)
- Error recoverability tracking

### Resilient Capture (`test_resilient_capture.py` + `test_resilient_extended.py`)
- USB ID connect preference, fallback to preferred
- No camera raises CaptureError
- Successful capture, auto-connect on capture
- Reconnection after capture failure, exhausted attempts
- Force rediscovery during reconnection
- Context manager connect/disconnect
- Disconnect state cleanup, double-disconnect safety

### Timeout and Backoff (`test_timeout_and_backoff.py`)
- WebcamSource timeout parameter (default 15s, custom)
- Successful acquire within timeout
- Frozen camera raises CaptureError after timeout
- Capture failure propagation
- Exponential backoff: increasing delay, max delay cap
- Custom backoff parameters

### CLI Vision Commands (`test_vision_cli.py`)
- `missy vision devices`: no cameras, cameras listed, troubleshooting hints
- `missy vision capture`: no camera error, device capture, best mode, burst mode, failure
- `missy vision doctor`: healthy run, failure display
- `missy vision inspect`: file inspection
- `missy vision review`: invalid mode rejection, all 4 valid modes

### Voice-Vision Integration (`test_voice_vision_integration.py`)
- Puzzle intent detected from voice transcript
- Painting intent detected from voice transcript
- Look-at-this intent detected
- No vision intent on normal speech
- Successful capture populates base64 in metadata
- No camera sets error in metadata
- Capture failure sets error in metadata
- Capture exception handled gracefully
- Vision import error handled gracefully
- Base metadata always present
- ASK decision skips auto-capture

## Integration Test Plan (Manual)

### Camera Hardware Tests
1. Connect Logitech C922x via USB
2. Run `missy vision doctor` — expect all checks PASS
3. Run `missy vision devices` — expect camera listed with correct USB ID
4. Run `missy vision capture -n 3` — expect 3 JPEG files saved
5. Run `missy vision capture --best` — expect sharpest frame saved
6. Run `missy vision capture --burst -n 5` — expect 5 burst frames
7. Disconnect camera, run `missy vision doctor` — expect graceful failure
8. Reconnect camera, run again — expect recovery

### Analysis Tests
1. Place a jigsaw puzzle on table
2. Run `missy vision review --mode puzzle` — expect board state analysis
3. Move pieces, run again — expect change detection (with scene memory)
4. Place a painting, run `missy vision review --mode painting` — expect warm feedback

### Screenshot Tests
1. Install scrot: `sudo apt install scrot`
2. Run `missy vision inspect --screenshot` — expect quality assessment

### Audio-Triggered Tests
1. Say "Missy, look at this" through voice channel
2. Expect vision auto-activation
3. Say "What is the weather?" — expect no vision activation
4. Say "Can you see the puzzle?" — expect auto-activation with puzzle mode

## Edge Cases

- Camera disconnected mid-capture: retry, then fail gracefully
- Frozen camera: timeout after 15s with clear error message
- Multiple cameras: preferred camera selection logic
- Low-light conditions: CLAHE enhancement, quality warning
- Corrupted image file: clear error message
- Very large image file: resize before processing
- Single-channel images: handled without crash
- No screenshot tool installed: clear error message
- Rapid successive captures: warm-up can be skipped
- Reconnection with exponential backoff: capped at 30s
- Session eviction: oldest by creation time, inactive preferred
