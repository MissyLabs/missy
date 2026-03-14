# TEST_EDGE_CASES

- Updated: 2026-03-14
- Total edge-case tests: 400+

## Security Policy Edge Cases (tested)

- Blocked domain enforcement (exact match, wildcard suffix, unknown host)
- Blocked CIDR enforcement (10.0.0.0/8, 192.168.0.0/16, denied ranges)
- Forbidden shell command rejection (command not in allowed list)
- Forbidden filesystem path rejection (write outside sandbox)
- Per-category network policy (provider hosts vs tool hosts vs discord hosts)
- Empty allowed_commands means allow-all (shell policy regression test)

## Provider Edge Cases (tested)

- Provider fallback when primary unavailable
- API key rotation across multiple keys
- Rate limiter token bucket exhaustion and replenishment
- Circuit breaker state transitions (closed → open → half-open → closed)
- Circuit breaker timeout doubling with max cap
- Model tiering (fast_model, premium_model selection)
- Timeout and retry behavior
- Codex provider native tool calling
- Ollama native tool calling (not prompted fallback)

## Agent Runtime Edge Cases (tested)

- Budget enforcement (max_spend_usd exceeded mid-loop)
- Checkpoint recovery scan on init
- Context manager token budget pruning (oldest history first)
- Context manager memory truncation when over budget
- Context manager learnings injection gating
- Done criteria verification prompt injection
- Done criteria pending list computation
- Learning extraction from tool-augmented runs
- Sub-agent spawning and isolation
- Tool execution error paths (KeyError, RuntimeError, generic Exception)
- Tool loop max iterations reached
- Capability mode filtering (no-tools, safe-chat, full)
- Rate limit acquisition failure handling
- Cost recording with subsystem failures
- Event emission with bus failures

## Scheduler Edge Cases (tested)

- Active hours gating (job outside window skipped)
- Overnight active hours window (22:00-06:00)
- Retry with exponential backoff after failure
- Max attempts exhaustion (permanent failure)
- Delete-after-run one-shot jobs
- Malformed jobs.json recovery (non-JSON, non-array, bad records)
- Timezone-aware scheduling (IANA timezone strings)
- APScheduler exception wrapping in SchedulerError
- Job add rollback on scheduling failure
- Date trigger scheduling
- Memory cleanup delegation

## Channel Edge Cases (tested)

- Discord DM policy enforcement (allowlist)
- Discord require-mention filtering
- Discord bot loop prevention (ignore own messages)
- Discord credential message detection and deletion
- Discord send retry with backoff
- Discord guild/channel/user policy enforcement
- Discord slash command routing (ask, model, status, help, unknown)
- Discord interaction handling with HTTP errors
- Discord reaction handling (reject, already-resolved, exception)
- Discord config token resolution (direct, env var, vault)
- Discord ffmpeg audio source creation and cleanup
- Voice device pairing with PBKDF2 hashed tokens
- Voice token regeneration invalidates old tokens
- Voice channel start/stop lifecycle (double-start, double-stop)
- Voice channel server failure propagation
- Voice edge client WebSocket protocol (connect, auth, send audio, receive)
- Voice server handler (registration, audio processing, TTS responses)
- Webhook channel authentication

## Memory Edge Cases (tested)

- SQLite FTS5 search with and without session filter
- Resilient memory store fallback on SQLite failure
- Session cleanup (older_than_days threshold with back-dated rows)
- Cost persistence across sessions
- ConversationTurn serialization/deserialization with metadata
- Clear session verification
- Cross-session recent turns retrieval
- Learning persistence with duck-typed objects
- Row-to-turn NULL metadata handling

## Vault Edge Cases (tested)

- Invalid key length (not 32 bytes)
- Decryption failure (corrupted data, wrong key)
- vault:// reference resolution
- $ENV_VAR reference resolution
- Missing vault key or env var error handling
- Cryptography package unavailable

## Tool Edge Cases (tested)

- File read truncation at max_bytes
- File read Path constructor and stat OS errors
- File write overwrite vs append modes
- File write Path constructor and IOError handling
- File delete directory rejection (only files allowed)
- File delete Path constructor and unlink OS errors
- List files recursive with max_entries truncation
- List files permission errors and stat failures
- Shell exec Docker sandbox routing
- Shell exec sandbox import failure fallback
- Shell exec output truncation at max bytes
- Shell exec error types (FileNotFoundError, PermissionError, OSError)
- Custom tool creation with name validation
- Web fetch response truncation at 64KB
- Tool registry policy violation returns ToolResult(success=False)
- Incus container CRUD, snapshots, exec, file transfer, config, network

## Observability Edge Cases (tested)

- Audit logger event handler exception swallowing
- Audit logger write failure (IOError caught silently)
- Recent events read failure returns empty list
- Malformed JSON lines in audit log skipped
- Policy violations read failure returns empty list
- Empty/whitespace-only lines in audit log ignored
- EventBus unsubscribe for unregistered callback (no-op)
- EventBus publish with raising subscriber (caught, logged, others still run)
- AuditEvent naive timestamp validation

## Code Evolution Edge Cases (tested)

- Restart process OS error fallback to sys.exit(75)
- Apply diff validation failure after approval
- Apply general exception with revert
- Rollback missing git_commit_sha
- Rollback git revert CalledProcessError
- Reject for APPLIED/FAILED proposals returns False
- Error analysis ValueError from relative_to silently skipped
- Revert diffs with git checkout exceptions swallowed
- Event emission with import/publish failures swallowed

## OAuth Edge Cases (tested)

- Callback handler log silence
- Callback server start/OSError fallback
- Wait for callback: code received, timeout, OAuth error, no code
- Refresh token stale-token return for missing credentials
- OAuth port-in-use branch
- Paste thread exception handling
