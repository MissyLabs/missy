"""Vision subsystem for the Missy AI assistant.

Provides camera discovery, image capture, source abstraction, preprocessing,
task-scoped scene memory, and domain-specific visual analysis (puzzle
assistance, painting feedback).

Submodules
----------
discovery
    USB webcam detection via vendor/product ID and device name matching.
capture
    OpenCV-based frame capture with retry, warm-up, and blank-frame detection.
sources
    Unified ``ImageSource`` abstraction for webcam, file, screenshot, photo.
pipeline
    Image normalization and preprocessing utilities.
scene_memory
    Task-scoped scene memory for multi-step visual tasks.
analysis
    Domain-specific visual analysis (puzzle, painting).
intent
    Audio-triggered vision activation via intent classification.
"""
