# Vision Subsystem Test Plan

## Test Coverage Summary

| Module | Tests | Status |
|--------|-------|--------|
| discovery.py | 16 | PASS |
| capture.py | 10 | PASS |
| sources.py | 18 | PASS |
| pipeline.py | 6 | PASS |
| scene_memory.py | 19 | PASS |
| intent.py | 18 | PASS |
| analysis.py | 15 | PASS |
| doctor.py | 10 | PASS |
| **Total** | **150** | **ALL PASS** |

## Unit Tests

### Camera Discovery (`test_discovery.py`)
- CameraDevice dataclass: USB ID, Logitech detection, immutability
- Fake sysfs scanning: empty, no devices, metadata node filtering
- Cache TTL behavior and force bypass
- Preferred camera selection: Logitech C922x > known > first
- USB ID and name pattern search
- KNOWN_CAMERAS database verification

### Frame Capture (`test_capture.py`)
- CaptureConfig defaults and custom values
- CaptureResult success/failure states
- CameraHandle open/close lifecycle (mocked OpenCV)
- Warm-up frame discarding
- Blank frame detection and retry
- All-retries-exhausted failure path
- Capture-without-open error
- Context manager cleanup
- Device path index parsing
- Capture-to-file with JPEG quality

### Image Sources (`test_sources.py`)
- ImageFrame auto-dimensions, JPEG encoding, base64 encoding
- FileSource: type, availability, acquire success, missing file, unreadable
- PhotoSource: type, availability, scan, file count, wrap-around, empty dir
- WebcamSource: type, unavailable device
- ScreenshotSource: type
- Factory function: all types, missing args, unknown type

### Pipeline (`test_pipeline.py`)
- PipelineConfig defaults
- Resize: small (no-op), large (downscale)
- Quality assessment: bright, dark, blurry conditions

### Scene Memory (`test_scene_memory.py`)
- SceneFrame: creation, hash stability, hash uniqueness
- SceneSession: create, add frames, eviction, latest/recent frames
- Observations, state updates, close, summarize
- Change detection: significant change, no change, insufficient frames
- SceneManager: create, get, active session, close, eviction, list

### Intent Detection (`test_intent.py`)
- Explicit vision requests: look, show, photo, screenshot
- Puzzle intents: piece placement, edge, sky section, sort, jigsaw
- Painting intents: feedback, improve, canvas
- Inspect intents: read, what's on table
- No-vision intents: general questions, coding requests
- Empty/whitespace input
- Activation thresholds: high, medium, low confidence
- Activation log and clear

### Analysis (`test_analysis.py`)
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
- run_all completeness

## Integration Test Plan (Manual)

### Camera Hardware Tests
1. Connect Logitech C922x via USB
2. Run `missy vision doctor` — expect all checks PASS
3. Run `missy vision devices` — expect camera listed with correct USB ID
4. Run `missy vision capture -n 3` — expect 3 JPEG files saved
5. Disconnect camera, run `missy vision doctor` — expect graceful failure
6. Reconnect camera, run again — expect recovery

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
- Multiple cameras: preferred camera selection logic
- Low-light conditions: CLAHE enhancement, quality warning
- Corrupted image file: clear error message
- Very large image file: resize before processing
- No screenshot tool installed: clear error message
- Rapid successive captures: warm-up can be skipped
