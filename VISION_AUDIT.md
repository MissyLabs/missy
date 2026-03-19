# Vision Subsystem Audit

## Security & Privacy

### Image Data Handling

- **No persistent storage by default**: Scene memory is in-process only. Images are not written to disk unless explicitly captured via `missy vision capture`.
- **Explicit activation only**: Vision is never always-on. Activation requires either user command or high-confidence intent detection.
- **Audit logging**: All vision activations, captures, and analysis requests are logged with timestamp, source type, trigger reason, and outcome.

### Threat Model

| Threat | Mitigation |
|--------|-----------|
| Accidental always-on surveillance | Vision is strictly on-demand; scoped to active tasks; auto-activation requires ≥0.80 confidence |
| Confused audio trigger | Ambiguous intents (0.50–0.79) require explicit user confirmation |
| Unauthorized camera access | Respects OS-level permissions; user must be in `video` group |
| Data exfiltration via images | Images sent to LLM API are subject to existing network policy enforcement |
| Prompt injection via image content | Input sanitization applies to all user-facing text; image analysis prompts are system-controlled |
| Stale device path exploitation | USB vendor/product ID identification, not volatile `/dev/videoN` paths |
| Device node symlink bypass | `FileSource` verifies `stat.S_ISREG` — rejects device nodes, pipes, sockets |
| Sysfs symlink cycles | `_read_usb_ids` tracks visited paths to detect and break cycles |
| Metadata field injection | `VisionMemoryBridge` filters reserved keys from caller-supplied metadata |
| Resource leak on camera error | `CameraHandle.open()` uses try/finally to release fd on failure |
| Warmup freeze on broken camera | Deadline-based timeout prevents indefinite blocking during warmup |

### Audit Events

Vision operations produce the following audit events:

| Action | Description |
|--------|-------------|
| `capture` | Single-frame capture from any source |
| `burst_capture` | Multi-frame burst capture with count/success tracking |
| `analyze` | Domain-specific analysis request |
| `intent` | Audio intent classification (text NOT logged, only length) |
| `device_discovery` | Camera enumeration |
| `session_create/close` | Scene memory lifecycle |
| `error` | Vision error with operation, device, recoverability |

```json
{
  "category": "vision",
  "action": "capture|burst_capture|analyze|intent|error",
  "device": "/dev/video0",
  "source_type": "webcam|file|screenshot|photo",
  "trigger_reason": "user_command|audio_intent|api",
  "success": true,
  "timestamp": "2026-03-19T00:00:00Z"
}
```

**Privacy note**: Intent audit events log `text_length` instead of the raw user text to prevent sensitive content from appearing in audit logs.

### Intent Classification Audit

All intent classifications are logged in the `VisionIntentClassifier.activation_log`:

- Intent type and confidence score
- Trigger phrase that matched
- Activation decision (activate/ask/skip)
- Timestamp

### Camera Discovery Security

- Discovery reads sysfs (read-only filesystem)
- No privileged operations required for discovery
- Device access requires `video` group membership
- Bus info is logged for forensic identification

## Implementation Review

### OpenCV Justification

OpenCV was chosen over raw V4L2 because:
1. Cross-platform format negotiation
2. Handles Logitech C922x without additional setup
3. Well-tested, widely-used library
4. No need for raw frame format handling

Raw V4L2 is NOT used. If a specific limitation is discovered, it should be documented before switching.

### Dependencies

- `opencv-python-headless` (no GUI dependencies)
- `numpy` (array operations)
- No network dependencies in the vision pipeline itself
- Screenshot tools (`scrot`, `gnome-screenshot`, `grim`) are optional system packages

### Code Quality

- Type annotations throughout
- Structured error handling with specific exception types
- Configurable parameters via dataclasses
- Context manager support for resource cleanup
- Lazy imports for optional dependencies
- ~1,500+ tests covering all modules (including hardening, burst, security, edge cases)
- Thread-safe capture and session management
- Input validation on all public APIs
- Grayscale and BGRA image format support
