# TEST_EDGE_CASES

- Updated: 2026-03-15 (session 14)
- Total edge-case tests: 1160+

## Security Policy Edge Cases (tested)

- Blocked domain enforcement (exact match, wildcard suffix, unknown host)
- Blocked CIDR enforcement (10.0.0.0/8, 192.168.0.0/16, denied ranges)
- Forbidden shell command rejection (command not in allowed list)
- Forbidden filesystem path rejection (write outside sandbox)
- Per-category network policy (provider hosts vs tool hosts vs discord hosts)
- Empty allowed_commands means allow-all (shell policy regression test)
- Process substitution blocking (<(...), >(...), <<(...))
- Command substitution blocking ($(…), backticks)
- Compound command chain splitting (&&, ||, ;, |, bare &)
- Background execution operator (&) properly split to prevent bypass
- Launcher command warnings (env, bash, sudo, find, xargs)
- DNS rebinding detection (hostname → private IP without explicit CIDR)
- DNS rebinding allowed when private CIDR explicitly configured
- DNS rebinding mixed-record attack (public + private IPs → deny)
- DNS rebinding deduplication (duplicate A records handled correctly)
- DNS rebinding partial CIDR allow (10.x allowed but 192.168.x not → deny)
- IPv6 rebinding protection (::1, fe80::, fd00::)
- Cloud metadata IP blocking (169.254.169.254)
- URL scheme restriction (file://, ftp://, data:// blocked)
- Gateway kwargs sanitization (follow_redirects stripped)
- Redirect following explicitly disabled on httpx clients
- MCP config file permission checks (owner, group/world-writable)

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
- Tool output injection scanning (warning label prepend)
- Response censoring applied in runtime (censor_response on final output)
- MCP server name validation (reject double underscores)
- MCP RPC timeout and response size limits
- MCP config permission verification before loading
- MCP subprocess environment sanitization

## Secrets Detection Edge Cases (tested)

- 20 credential patterns: API keys, AWS, GitHub, Stripe, Slack, JWT, Anthropic, OpenAI, GCP, Discord, GitLab, npm, PyPI, SendGrid, DB connection strings
- Database connection string detection (postgres://, mysql://, mongodb://, redis://)
- Case-insensitive DB scheme matching
- URL without credentials not flagged as secret
- Truncated tokens not flagged (partial matches rejected)
- Vault atomic write failure cleanup (temp file unlinked)
- Vault key symlink rejection (VaultError raised)
- Vault crypto unavailable (VaultError with install hint)
- Config vault:// resolution fallback (exception → original value returned)

## Scheduler Edge Cases (tested)

- Active hours gating (job outside window skipped)
- Overnight active hours window (22:00-06:00) — heartbeat and scheduler
- Heartbeat loop fire on interval (stop.wait returns False)
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
- Discord DM policy fallthrough for unknown policy values
- Discord require-mention filtering
- Discord mention fallback when own_id is falsy (uses mentions list)
- Discord bot loop prevention (ignore own messages)
- Discord credential message detection and deletion
- Discord send retry with backoff
- Discord guild/channel/user policy enforcement
- Discord slash command routing (ask, model, status, help, unknown)
- Discord interaction handling with HTTP errors
- Discord reaction handling (reject, already-resolved, exception)
- Discord config token resolution (direct, env var, vault)
- Discord ffmpeg audio source creation and cleanup
- Discord voice join with channel_id not found
- Discord voice start_listening body (state, watchdog task creation)
- Discord voice watchdog router exception handling
- Discord voice speech handler: empty transcript, no agent callback, empty/cleaned response
- Discord voice resample PCM boundary (last sample without right neighbour)
- Discord gateway heartbeat loop with jitter and interval
- Discord voice agent callback closure (run_in_executor)
- Voice device pairing with PBKDF2 hashed tokens
- Voice token regeneration invalidates old tokens
- Voice channel start/stop lifecycle (double-start, double-stop)
- Voice channel server failure propagation
- Voice edge client WebSocket protocol (connect, auth, send audio, receive)
- Voice server handler (registration, audio processing, TTS responses)
- Webhook channel authentication
- Webhook handler log_message debug output
- Webhook per-IP rate limiting (429 on excess, window expiry, per-IP isolation)
- Webhook payload size limits (413 on oversized, boundary testing)
- Webhook queue overflow (503 on full queue, drain and re-accept)
- Webhook HMAC signature validation (valid, missing, wrong, no-secret bypass)

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
- Vault atomic write with mkstemp+rename (concurrent access safety)
- Vault TOCTOU-safe key creation with O_CREAT|O_EXCL
- Vault symlink rejection on key file read
- Config hotreload rejects symlinks and world-writable files

## Browser Tool Edge Cases (tested)

- Display environment setup (DISPLAY already set, X socket detection, fallback to :0)
- Browser session start (playwright ImportError, successful launch)
- Browser page management (auto-start on None context, reuse open pages, new page on all closed)
- Browser session close (context.close error, pw.stop error, both errors simultaneously)
- Browser page helper (get_or_create delegation, get_page forwarding)
- Browser session registry (has_active_session, screenshot_active, close edge cases)

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

## Config Edge Cases (tested)

- Config hot-reload OSError on stat (file temporarily unavailable)
- Doctor command: watchdog check exception, voice config parsing, checkpoint exception

## Proactive Manager Edge Cases (tested)

- Watchdog import success path (module reimport with mock watchdog)
- Threshold loop inner break (stop event set between triggers)
- Audit event publish and publish exception swallowing
- File handler class with mocked watchdog (init, on_any_event, wiring)
- Proactive callback success (AgentRuntime.run invocation)
- Proactive callback fallback (AgentRuntime creation failure, logger-only stub)

## Tool Security Edge Cases (session 11, tested)

- Path traversal attacks (../../etc/passwd) in file read/write/delete/list
- Null byte injection in file paths (rejected by Python 3.12+ Path)
- Unicode homograph filenames (Cyrillic vs Latin)
- Zero-width space and RTL override in paths
- CJK/emoji filenames round-trip
- NFC vs NFD Unicode normalization
- Extremely long paths (>4096 chars, >255 byte components)
- Special characters in filenames (spaces, quotes, newlines, shell metacharacters)
- SSRF prevention (file://, dict://, gopher://, ftp://, ldap://, AWS IMDS, localhost, IPv6 loopback)
- Empty/missing required parameters for all tools
- Type confusion (int/None/list where string expected, str where int expected)
- File write mode injection (raw open-mode strings, shell injection in mode)
- Encoding injection (fake codec, semicolon in encoding name)
- Special file protection (/dev/zero, /dev/null, directory-as-file)

## Input Validation Edge Cases (session 11, tested)

- Voice WebSocket sample_rate clamping [8000, 48000]
- Voice WebSocket channels clamping [1, 2]
- Non-numeric sample_rate/channels values (safe default fallback)
- None, negative, and extreme int values for audio params
- Code evolution confidence clamping [0.0, 1.0]
- Confidence values above 1.0, below 0.0, string-numeric

## Circuit Breaker Edge Cases (session 11, tested)

- State machine: Closed → Open → HalfOpen → Closed full cycle
- Exponential backoff doubling and max cap enforcement
- Success resets backoff to base timeout
- Alternating success/failure pattern (never opens)
- Open state rejection doesn't bump failure count
- Thread safety: concurrent failures, concurrent reads, deadlock detection
- Lock reentrant safety
- Multiple independent breaker instances
- Custom exception subclass tracking

## MCP Edge Cases (session 11, tested)

- JSON-RPC request/response encoding
- Server process lifecycle (start, health check, restart, shutdown)
- Tool namespacing (server__tool)
- Invalid namespaced tool name format
- Server not connected error
- Config parse failure (malformed JSON)
- Connect failure propagation
- Shutdown with disconnect errors suppressed

## Prompt Patches Edge Cases (session 11, tested)

- Auto-approval: low-risk types (tool hint, domain, style) with confidence >= 0.8
- No auto-approval for high-risk types (error avoidance, workflow) or low confidence
- MAX_PATCHES capacity enforcement (returns None)
- Expiration: >= 5 applications AND < 40% success rate
- Not expired: < 5 applications (even at 0% success)
- Persistence round-trip (all statuses survive reload)
- Malformed store file recovery (returns empty list)

## Sub-Agent Edge Cases (session 11, tested)

- Numbered list parsing (period and parenthesis formats)
- Sequential connective parsing (then, and then, after that, finally)
- Dependency chain: connective tasks have depends_on set
- Single-task fallback for unstructured prompts
- MAX_SUB_AGENTS cap enforcement
- Runtime factory called per subtask (isolation)
- Error in subtask captured in subtask.error

## Approval Gate Edge Cases (session 11, tested)

- Threaded approve/deny via handle_response
- Approval timeout raises ApprovalTimeout
- Pending cleanup after timeout
- Case-insensitive response handling
- send_fn failure doesn't block approval flow
- No send_fn (None) works correctly

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

## Session 14 Hardening Edge Cases (tested)

### Tool Execution Retry
- Transient errors (TimeoutError, ConnectionError, OSError) trigger retry with backoff
- httpx.TimeoutException triggers retry when httpx is available
- Retry exhaustion (3 attempts) returns error result
- Non-transient errors (KeyError, ValueError) fail immediately without retry
- Exponential backoff delays verified (1.0s, 2.0s)

### Webhook Rate Tracker Memory
- Stale IP eviction removes entries with all-expired timestamps
- Empty timestamp lists are cleaned up
- Overflow triggers automatic eviction when >10K IPs tracked
- Active IPs preserved during eviction

### Gateway Connection Pool
- Explicit pool limits verified (max_connections=20, keepalive_expiry=30s)
- DELETE/PATCH methods enforce policy, sanitize kwargs, emit audit events
- Async adelete/apatch methods enforce policy on denied hosts

### Prompt Injection Patterns (session 14)
- Unclosed HTML comments with keywords (system, ignore, inject, override)
- Data URI injection (data:text/html, data:text/javascript)
- Hidden div detection (display:none with embedded instructions)
- Markdown comment pattern ([comment]:)
- Llama 3 model tokens (begin_of_text, start_header_id, end_header_id)
- Reserved special tokens (<|reserved_special_token)
- Chained instruction patterns (new/updated/revised/real instructions:)
- Portuguese injection keywords
- Russian injection keywords
- Benign text false-negative verification

### DNS Rebinding Mixed Records
- Public + private IP → deny (prevents mixed-record TOCTTOU attack)
- Public + loopback → deny
- All-public in CIDR → allow
- Duplicate IPs deduplicated
- Partial CIDR allow + disallowed private → deny
- All-private in allowed CIDRs → allow

### Scheduler Input Validation
- Hour range validation (0-23, rejects 25)
- Minute range validation (0-59, rejects 70)
- Zero interval rejection
- Timezone attachment for cron/date triggers (not intervals)

## OAuth Edge Cases (tested)

- Callback handler log silence
- Callback server start/OSError fallback
- Wait for callback: code received, timeout, OAuth error, no code
- Refresh token stale-token return for missing credentials
- OAuth port-in-use branch
- Paste thread exception handling
