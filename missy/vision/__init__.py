"""Vision subsystem for the Missy AI assistant.

Provides camera discovery, image capture, source abstraction, preprocessing,
task-scoped scene memory, health monitoring, and domain-specific visual
analysis (puzzle assistance, painting feedback).

Submodules
----------
discovery
    USB webcam detection via vendor/product ID and device name matching.
capture
    OpenCV-based frame capture with retry, warm-up, and blank-frame detection.
resilient_capture
    Auto-reconnection with exponential backoff and failure tracking.
sources
    Unified ``ImageSource`` abstraction for webcam, file, screenshot, photo.
pipeline
    Image normalization and preprocessing utilities.
scene_memory
    Task-scoped scene memory for multi-step visual tasks.
health_monitor
    Capture statistics, device health tracking, and diagnostic reports.
analysis
    Domain-specific visual analysis (puzzle, painting).
intent
    Audio-triggered vision activation via intent classification.
doctor
    Vision subsystem diagnostics and health checks.
provider_format
    Provider-specific image API formatting.
audit
    Vision audit event logging.
"""
