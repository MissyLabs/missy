# Test Edge Cases

## Date: 2026-03-18

## Persona System Edge Cases

| Test | Description | Status |
|---|---|---|
| Corrupt YAML fallback | Invalid YAML syntax → defaults | Covered |
| Non-dict root | YAML list instead of dict → defaults | Covered |
| Empty file | Zero-byte persona.yaml → defaults | Covered |
| Missing file | No persona.yaml → defaults | Covered |
| Invalid field update | `update(bad_field=...)` → ValueError | Covered |
| Version field protection | Cannot update version directly | Covered |
| Copy isolation | `get_persona()` returns independent copy | Covered |
| Shared mutable state | List fields don't alias across instances | Covered |
| Nested dir creation | Save creates parent dirs | Covered |

## Behavior Layer Edge Cases

| Test | Description | Status |
|---|---|---|
| Empty message list | `analyze_user_tone([])` → "casual" | Covered |
| No user messages | Only assistant messages → "casual" | Covered |
| Short messages | < 8 words avg → "brief" | Covered |
| Long messages | > 40 words avg → "verbose" | Covered |
| Frustration override | Frustrated + formal → "frustrated" | Covered |
| None context | `shape_system_prompt(text, None)` → no crash | Covered |
| Empty persona | PersonaConfig with empty lists → no crash | Covered |
| Code block preservation | Robotic text inside ```...``` survives | Covered |
| Inline code preservation | Robotic text inside \`...\` survives | Covered |
| Multiple code blocks | Complex markdown preserved | Covered |
| 3+ blank lines | Collapsed to 2 max | Covered |
| Empty response | `shape_response("")` → "" | Covered |
| Clean input | No robotic patterns → unchanged | Covered |

## Hatching System Edge Cases

| Test | Description | Status |
|---|---|---|
| Corrupt YAML state | Bad hatching.yaml → UNHATCHED default | Covered |
| Already hatched | `run_hatching()` returns immediately | Covered |
| Resume from partial | Completed steps skipped | Covered |
| Recover from FAILED | Error cleared, retry from last step | Covered |
| Missing state file | `reset()` → no-op, no crash | Covered |
| Log file corruption | Malformed JSONL lines skipped | Covered |
| Blank log lines | Empty lines in JSONL → skipped | Covered |
| Step exception | Unrecoverable error → FAILED state | Covered |
| Timestamp population | started_at/completed_at set correctly | Covered |

## Runtime Integration Edge Cases

| Test | Description | Status |
|---|---|---|
| No persona manager | PersonaManager unavailable → graceful | Covered |
| No behavior layer | BehaviorLayer unavailable → graceful | Covered |
| No response shaper | ResponseShaper unavailable → graceful | Covered |
| Behavior exception | Exception during shaping → original prompt | Covered |
| Empty history | No user messages → no tone analysis | Covered |

## Security Edge Cases

| Test | Description | Status |
|---|---|---|
| Oversized tool result | > 200k chars truncated | Covered |
| Secret in prompt | Warning + proceed | Covered |
| Injection patterns | 250+ patterns detected | Covered |
| Unicode normalization | Homoglyph attacks caught | Covered |
| Base64 encoded payloads | Decoded and checked | Covered |
| Nested encoding | Multi-layer encoding caught | Covered |
