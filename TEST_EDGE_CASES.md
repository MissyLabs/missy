# Test Edge Cases

## Date: 2026-03-18 (updated)

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
| Backup on save | Previous version backed up to persona.d/ | Covered |
| Backup pruning | Max 5 backups retained | Covered |
| Rollback restores | Latest backup content restored | Covered |
| Rollback creates backup | Current state backed up before restore | Covered |
| Rollback with no backups | Returns None gracefully | Covered |
| Diff shows changes | Unified diff between backup and current | Covered |
| Diff empty when no backups | Returns empty string | Covered |
| Diff empty when no file | Returns empty string | Covered |
| File permissions | persona.yaml set to 0o600 on save | Covered |

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
| Mixed tone signals | Casual + technical → resolves correctly | Covered |
| All-caps input | Detected as brief or frustrated | Covered |
| Base prompt preserved | System prompt always starts with base | Covered |
| Full context dict | All keys populated → no crash | Covered |
| Very long input | 10k words → no crash | Covered |
| Empty string intent | Returns "question" default | Covered |
| Whitespace-only intent | Returns "question" default | Covered |
| Urgency precedence | High overrides medium | Covered |
| Greeting position | Only matches at start of message | Covered |
| Whitespace-only response | Stripped to empty | Covered |
| Multiple robotic phrases | All removed in single pass | Covered |
| Nested code blocks | Multiple blocks with text between | Covered |

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
| Empty YAML state file | Returns UNHATCHED default | Covered |
| List YAML state file | Non-dict → UNHATCHED default (bug fixed) | Covered |
| Extra keys in state | Unknown keys ignored gracefully | Covered |
| Concurrent log entries | Chronological order preserved | Covered |
| None steps_completed | Coerced to empty list (bug fixed) | Covered |
| Boolean string coercion | String/int booleans handled | Covered |
| IN_PROGRESS state query | Not hatched but not needs_hatching | Covered |
| Unicode log content | Multi-language content preserved | Covered |
| _HatchingStepWarning | Proper Exception subclass | Covered |

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
