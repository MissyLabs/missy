# Missy Edge Case Tests

## Vision Edge Cases

### Camera Discovery
- No USB cameras connected → returns empty list, no crash
- Multiple cameras → preferred selection logic works
- Camera disconnected during enumeration → partial results returned safely
- Stale sysfs cache → TTL-based invalidation, force refresh available
- Non-camera video devices (e.g., capture cards) → filtered by index=0

### Frame Capture
- Camera not opened → CaptureError with clear message
- All frames blank → retry exhausted, structured failure result
- Camera busy (another application) → clear error about device availability
- Permission denied on /dev/video* → message suggests `usermod -aG video`
- Camera disconnected mid-capture → read() returns False, retry logic engages
- Very low light → blank frame detection + quality assessment warning
- USB hub bandwidth limitation → camera negotiates lower resolution silently
- Rapid sequential captures → warm-up skipped on already-warm camera

### Image Sources
- Missing image file → FileNotFoundError with path
- Corrupt image file → ValueError from cv2.imread returning None
- Empty photo directory → FileNotFoundError with clear message
- Photo directory wrap-around → index resets to 0
- Screenshot with no display ($DISPLAY unset) → screenshot tool error
- No screenshot tools installed → RuntimeError listing what was tried

### Scene Memory
- Session at max frames → oldest evicted, new frame stored
- Single frame → no change detection possible (returns None)
- All sessions active at capacity → oldest active session evicted
- Session closed → image data released, metadata preserved
- Identical sequential frames → change score near 0.0
- Completely different frames → change score near 1.0

### Intent Classification
- Empty string → VisionIntent.NONE, confidence 0.0
- Whitespace only → same as empty
- Combined intents ("look at this puzzle piece") → highest confidence wins
- No vision keywords → ActivationDecision.SKIP
- Vision keyword with low context → ActivationDecision.ASK
- Strong explicit request → ActivationDecision.ACTIVATE

### Pipeline
- Very small image (< target) → no resize
- Very large image → resize preserves aspect ratio
- Zero-contrast image → "low contrast" quality issue
- Overexposed image → "overexposed" quality issue

## Security Edge Cases

### Input Sanitization
- Unicode homoglyph injection → normalized before pattern matching
- Base64-encoded injection → decoded and checked
- Multi-language injection → supported patterns

### Secrets Detection
- Partial API key format → still detected
- Key in URL parameter → detected and censored
- Key split across message boundaries → per-message detection

### Policy Enforcement
- Request to non-allowed host → denied at PolicyHTTPClient
- Request with allowed host but denied method → REST policy blocks
- Allow-always in interactive approval → session-scoped only

### Circuit Breaker
- 5 consecutive failures → circuit opens
- Recovery attempt in half-open → single request allowed
- Success in half-open → circuit closes
- Exponential backoff: 60s, 120s, 240s, 300s (capped)

## Agent Edge Cases

### Context Management
- Context at 80% capacity → MemoryConsolidator triggers sleep mode
- Memory fraction overflow → oldest turns pruned first
- Empty conversation → minimal system prompt only

### Tool Execution
- Tool returns > 200k chars → truncated
- Tool raises exception → error captured, not propagated
- Heredoc in shell command → rewritten to temp file

### Provider
- API key expired → ProviderError, circuit breaker counts
- Rate limited (429) → retry with backoff
- Multiple API keys → rotation on failure
