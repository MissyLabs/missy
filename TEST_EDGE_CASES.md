# TEST_EDGE_CASES

- Updated: 2026-03-14
- Total edge-case tests: 200+

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
- Circuit breaker state transitions (closed → open → half-open)
- Model tiering (fast_model, premium_model selection)
- Timeout and retry behavior
- Codex provider native tool calling
- Ollama native tool calling (not prompted fallback)

## Agent Runtime Edge Cases (tested)

- Budget enforcement (max_spend_usd exceeded mid-loop)
- Checkpoint recovery scan on init
- Context manager token budget pruning (oldest history first)
- Done criteria verification prompt injection
- Learning extraction from tool-augmented runs
- Sub-agent spawning and isolation

## Scheduler Edge Cases (tested)

- Active hours gating (job outside window skipped)
- Overnight active hours window (22:00-06:00)
- Retry with exponential backoff after failure
- Max attempts exhaustion (permanent failure)
- Delete-after-run one-shot jobs
- Malformed jobs.json recovery (non-JSON, non-array, bad records)
- Timezone-aware scheduling (IANA timezone strings)

## Channel Edge Cases (tested)

- Discord DM policy enforcement (allowlist)
- Discord require-mention filtering
- Discord bot loop prevention (ignore own messages)
- Discord credential message detection and deletion
- Discord send retry with backoff
- Voice device pairing with PBKDF2 hashed tokens
- Voice token regeneration invalidates old tokens
- Webhook channel authentication

## Memory Edge Cases (tested)

- SQLite FTS5 search with special characters
- Resilient memory store fallback on SQLite failure
- Session cleanup (older_than_days threshold)
- Cost persistence across sessions

## Vault Edge Cases (tested)

- Invalid key length (not 32 bytes)
- Decryption failure (corrupted data, wrong key)
- vault:// reference resolution
- $ENV_VAR reference resolution
- Missing vault key or env var error handling
- Cryptography package unavailable

## Tool Edge Cases (tested)

- File read truncation at max_bytes
- File write overwrite vs append modes
- File delete directory rejection (only files allowed)
- List files recursive with max_entries truncation
- Shell exec Docker sandbox routing
- Custom tool creation with name validation
- Web fetch response truncation at 64KB
- Tool registry policy violation returns ToolResult(success=False)
