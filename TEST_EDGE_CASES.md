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

## Property-Based Testing (Session 5)

| Test | Description | Status |
|---|---|---|
| Sanitizer: arbitrary text | Never crashes on any string input | Covered |
| Sanitizer: binary decode | Never crashes on decoded binary data | Covered |
| Sanitizer: truncation | Always enforces max length | Covered |
| Sanitizer: zero-width strip | All zero-width chars removed | Covered |
| Sanitizer: injection + context | Known injections detected with random prefix/suffix | Covered |
| Sanitizer: case insensitive | Mixed-case injections still caught | Covered |
| Sanitizer: obfuscation | Zero-width obfuscated injections detected | Covered |
| Sanitizer: base64 injection | Base64-encoded injections decoded and caught | Covered |
| Sanitizer: no false positives | Clean alphanumeric text not flagged | Covered |
| Secrets: arbitrary text | scan/redact/has_secrets never crash | Covered |
| Secrets: known credentials | All 17 credential types detected in context | Covered |
| Secrets: redaction hides | Full secret value not in redacted output | Covered |
| Secrets: ordered results | Findings sorted by position | Covered |
| Secrets: DB connections | Postgres/MySQL/MongoDB/Redis URLs detected | Covered |
| Secrets: password patterns | password=... detected | Covered |

## Voice Channel Edge Cases (Session 5)

| Test | Description | Status |
|---|---|---|
| Auth success | Valid token → auth_ok frame | Covered |
| Auth invalid token | Bad token → auth_fail + close | Covered |
| Auth unpaired node | Paired=false → rejected | Covered |
| Auth muted node | Policy_mode=muted → muted frame | Covered |
| Auth node not found | Missing node → auth_fail | Covered |
| Pair request | New node → pair_pending + close | Covered |
| Heartbeat update | Updates presence + marks online | Covered |
| Heartbeat error resilience | Presence error → no crash | Covered |
| Audio full pipeline | STT→agent→TTS→stream works | Covered |
| Audio STT failure | Error frame sent | Covered |
| Audio empty transcript | Agent not called | Covered |
| Audio TTS failure | Text response still sent | Covered |
| Whisper device resolution | Auto→CPU fallback | Covered |
| Whisper compute type | Auto selects int8 for CPU | Covered |
| Piper env sanitization | API keys excluded from subprocess | Covered |
| Piper model resolution | Voice dir scan + fallback | Covered |
| Piper synthesis subprocess | PCM→WAV conversion | Covered |
| Piper timeout | Process killed on timeout | Covered |

## Discord Command Edge Cases (Session 5)

| Test | Description | Status |
|---|---|---|
| Non-command passthrough | No ! prefix → not handled | Covered |
| No guild → error | Voice commands server-only | Covered |
| No voice manager → error | Graceful message | Covered |
| !join by user | Follows user's voice channel | Covered |
| !join by name | Finds channel by name | Covered |
| !join by ID | Joins by snowflake ID | Covered |
| !join shows capabilities | Listen/speak status | Covered |
| !leave not connected | "Not in channel" message | Covered |
| !say no text | Usage hint | Covered |
| Case-insensitive commands | !JOIN, !Leave work | Covered |
| is_image_attachment | Content-type + extension detection | Covered |
| find_latest_image | Skips non-images, handles empty | Covered |
| !analyze no image | "No image found" message | Covered |
| !screenshot bad subcmd | Usage hint | Covered |
| ImageCommandResult frozen | Immutable dataclass | Covered |

## Concurrent Memory Access (Session 5)

| Test | Description | Status |
|---|---|---|
| 5 writers concurrent | All 50 turns stored | Covered |
| Read+write concurrent | No crashes during interleaving | Covered |
| Search+write concurrent | Correct results during writes | Covered |
| Clear+write concurrent | Survivor session preserved | Covered |
| Compact+read concurrent | No crashes | Covered |
| SQLite 5 writers | All 50 turns persisted | Covered |
| SQLite search+write | Concurrent access safe | Covered |
| SQLite learnings concurrent | Save+read no errors | Covered |
| High volume (100 writes) | ThreadPool stress test | Covered |
| Persistence verification | Data survives reload after concurrent writes | Covered |
