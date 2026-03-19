# Missy Vision Subsystem

## Overview

Missy's vision subsystem provides on-demand visual capabilities for a Linux host with USB webcam and additional image sources. It is designed for production use with resilience, diagnostics, and security in mind.

## Architecture

```
missy/vision/
├── __init__.py          # Package documentation
├── discovery.py         # USB camera discovery via sysfs
├── capture.py           # OpenCV-based frame capture with resilience
├── sources.py           # Unified image source abstraction
├── pipeline.py          # Image preprocessing (resize, CLAHE, denoise)
├── scene_memory.py      # Task-scoped scene memory for multi-step tasks
├── analysis.py          # Domain-specific analysis (puzzle, painting)
├── intent.py            # Audio-triggered vision intent classification
└── doctor.py            # Diagnostics and health checks
```

## Image Sources

The vision subsystem supports five image sources through a unified `ImageSource` abstraction:

| Source | Class | Description |
|--------|-------|-------------|
| Webcam | `WebcamSource` | Live USB webcam capture via OpenCV |
| File | `FileSource` | Still image file ingestion |
| Screenshot | `ScreenshotSource` | Desktop screenshot (scrot/gnome-screenshot/grim) |
| Photo | `PhotoSource` | Saved photo directory review |
| Scene Memory | `SceneSession` | Multi-frame task-scoped memory |

## Camera Discovery

Camera discovery uses `/sys/class/video4linux/` sysfs traversal:

1. Enumerate `/sys/class/video4linux/videoN` entries
2. Filter to `VIDEO_CAPTURE` capable devices (index=0)
3. Read USB vendor/product IDs by walking up sysfs tree
4. Match against known camera database
5. Cache results with configurable TTL (default 10s)

### Preferred Camera Selection

1. Logitech C922x (vendor `046d`, product `085c`/`085b`)
2. Any known camera from the built-in database
3. First available device

### Resilience

- Stable identification by USB vendor/product ID, not volatile `/dev/videoN` paths
- Handles re-enumeration when device paths change
- Cache invalidation on forced rediscovery

## Capture Pipeline

```
Camera Device → OpenCV VideoCapture → Warm-up → Frame Read → Blank Detection → Result
                                         ↓            ↓             ↓
                                    Discard N     Retry on      Retry on
                                    frames        failure       blank
```

### Features

- **Warm-up**: Discards configurable number of initial frames (default 5) for auto-exposure/white-balance stabilization
- **Retry**: Configurable retry count (default 3) with delay between attempts
- **Blank detection**: Rejects frames with mean pixel value below threshold (default 5.0)
- **Resolution negotiation**: Requests preferred resolution, accepts what camera provides
- **Graceful failure**: Returns structured `CaptureResult` with error details
- **Burst capture**: `capture_burst(count, interval)` for rapid multi-frame capture (1-20 frames)
- **Best-frame selection**: `capture_best(burst_count)` selects sharpest frame via Laplacian variance
- **Thread safety**: All capture/close operations are protected by `threading.Lock`
- **Input validation**: Device path, frame shape, and configuration validated on entry

## Image Preprocessing

The `ImagePipeline` normalizes images before analysis:

1. **Resize**: Scale to target dimension (default 1280px) preserving aspect ratio
2. **CLAHE**: Contrast Limited Adaptive Histogram Equalization for exposure normalization
3. **Denoise**: Optional fastNlMeansDenoisingColored
4. **Sharpen**: Optional unsharp masking

### Quality Assessment

Returns metrics for:
- Brightness (mean grayscale intensity)
- Contrast (grayscale standard deviation)
- Sharpness (Laplacian variance)
- Overall quality classification (good/fair/poor)
- Specific issues (dark, blurry, overexposed, etc.)

## Analysis Modes

### General

Standard image description and object identification.

### Puzzle Assistance

Board-state tracking, piece identification, edge/corner detection, color clustering, placement guidance, and step-by-step suggestions. Supports follow-up analysis with scene memory context.

### Painting Feedback

Warm, encouraging composition critique with supportive coaching tone. Covers first impressions, color/light, composition, technique, emotional impact, and growth opportunities. Designed to feel like a gentle painting coach.

### Inspection

Detailed condition assessment for general visual inspection tasks.

## Scene Memory

Task-scoped scene memory for multi-step visual tasks:

- Frames stored in-process (not persisted to disk for privacy)
- Configurable max frames per session (default 20, oldest evicted)
- Change detection between frames (normalized pixel difference)
- State tracking (e.g., puzzle completion progress)
- Observation accumulation across frames
- Session summary export for audit logging

## Audio-Triggered Vision

The `VisionIntentClassifier` detects vision-related intent from user utterances:

### Intent Categories

| Intent | Example Phrases | Auto-activate |
|--------|----------------|---------------|
| LOOK | "Look at this", "Show me" | Yes (≥0.80) |
| PUZZLE | "Where does this piece go?" | Yes (≥0.80) |
| PAINTING | "What do you think of this painting?" | Yes (≥0.80) |
| INSPECT | "Read what this says" | Yes (≥0.80) |
| CHECK | "Check what's on the table" | Yes (≥0.80) |
| SCREENSHOT | "Take a screenshot" | Yes (≥0.90) |

### Activation Thresholds

- **≥0.80**: Auto-activate vision (strong signal)
- **0.50–0.79**: Ask user for confirmation (ambiguous)
- **<0.50**: Skip (no vision needed)

### Safety Rules

- Vision activation is scoped to the active task
- Never becomes always-on surveillance
- All activation decisions are logged for audit
- Ambiguous cases require explicit user confirmation

## CLI Commands

```bash
missy vision devices     # Enumerate and diagnose cameras
missy vision capture     # Capture frames (--device, --output, --count)
missy vision inspect     # Quality assessment (--file, --screenshot, --device)
missy vision review      # LLM-powered analysis (--mode puzzle|painting|general)
missy vision doctor      # Full diagnostics check
```

## Configuration

Vision capabilities require the `vision` optional dependency:

```bash
pip install -e ".[vision]"   # opencv-python-headless, numpy
```

## Error Handling

The vision subsystem handles these failure modes:

| Condition | Behavior |
|-----------|----------|
| No camera present | Clear error message, suggest troubleshooting |
| Device busy | Retry with backoff |
| Permission denied | Suggest `usermod -aG video` |
| Disconnect/reconnect | Cache invalidation, rediscovery |
| Bad frame reads | Retry up to max_retries |
| Blank frames | Detect and retry |
| Low light | Quality assessment warns, CLAHE enhances |
| Unsupported resolution | Accept camera's preferred resolution |
| Camera warm-up | Discard initial frames |
| Stale device paths | USB ID-based identification, not path-based |
