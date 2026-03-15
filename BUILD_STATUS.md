# Missy Build Status

## Status: COMPLETE + HARDENED

All core phases implemented, parity gaps closed, comprehensive hardening applied.

## Completed Steps

1. Core infrastructure (session, events, exceptions)
2. Config system (YAML loading, secure defaults, hot-reload)
3. Policy engine (network CIDR/domain/per-category, filesystem, shell)
4. Network gateway (PolicyHTTPClient wrapping httpx)
5. Providers (Anthropic, OpenAI, Ollama, Codex with policy enforcement, tiering, rotation)
6. Tools framework (registry, BaseTool, 15+ built-in tools)
7. Skills system (registry, BaseSkill, 6 built-in skills)
8. Plugin system (registry, loader with security gates)
9. Scheduler (APScheduler, human schedule parsing, job persistence, retry, timezone)
10. Memory store (SQLite FTS5, resilient fallback, session metadata, cost tracking)
11. Observability (AuditLogger JSONL, OpenTelemetry traces+metrics)
12. Security (InputSanitizer, SecretsDetector, SecretCensor, Vault, Docker Sandbox)
13. Channels (CLI, Discord, Webhook, Voice)
14. Agent runtime (multi-step loop, tool calling, circuit breaker, context management, rate limiting, streaming)
15. Advanced agent (checkpoint/recovery, failure tracker, done criteria, learnings, prompt patches, sub-agents, approval gate, proactive triggers, cost tracking with budget enforcement and persistence)
16. CLI (60+ commands via click + rich, including recover, evolve)
17. Discord (WebSocket gateway, REST API, threads, slash commands, pairing, access control, voice, interactive setup wizard)
18. Code self-evolution engine (propose, test, apply, rollback)
19. Tests (6008 tests, 99%+ coverage)
20. Documentation (SECURITY.md, OPERATIONS.md, ARCHITECTURE.md, CONFIG_REFERENCE.md, DISCORD.md, TESTING.md, TROUBLESHOOTING.md, 10+ implementation docs)
21. Audit artifacts (AUDIT_SECURITY.md, AUDIT_CONNECTIVITY.md)
22. Test artifacts (TEST_RESULTS.md, TEST_EDGE_CASES.md, BUILD_RESULTS.md)
23. OpenClaw gap analysis (OPENCLAW_GAP_ANALYSIS.md)

## Architecture State

```
missy/                          # 123 Python source files
  core/        - session (w/ metadata), events, exceptions
  config/      - settings, YAML loading, hot-reload (watchdog)
  policy/      - network (CIDR/domain/per-category), filesystem, shell engines + facade
  gateway/     - PolicyHTTPClient
  providers/   - base, anthropic, openai, ollama, codex, registry (fallback, tiering, rotation), rate_limiter
  tools/       - base, registry, 15+ builtin tools (shell w/ sandbox, file, web, calculator, browser, tts, atspi, x11, incus, code_evolve, self_create_tool)
  skills/      - base, registry, 6 builtin skills (config_show, datetime, health_check, summarize, system_info, workspace_list)
  plugins/     - base, loader
  scheduler/   - jobs, parser, manager (retry, timezone)
  memory/      - sqlite_store (FTS5, sessions, costs tables), resilient_store, json_store
  observability/ - audit_logger, otel_exporter
  security/    - sanitizer, secrets, censor, vault (ChaCha20), sandbox (Docker)
  channels/    - base, cli, discord (gateway, rest, voice, commands, config, threads), webhook, voice (server, registry, pairing, presence, stt/tts, edge_client)
  agent/       - runtime (w/ rate limiting, streaming, budget enforcement, recovery scan),
                 circuit_breaker, context, checkpoint, failure_tracker, done_criteria,
                 learnings, prompt_patches, sub_agent, approval, proactive,
                 cost_tracker (w/ budget enforcement + SQLite persistence),
                 watchdog, heartbeat, code_evolution
  cli/         - main (60+ click CLI commands), wizard, oauth, anthropic_auth
  mcp/         - manager, client (MCP server integration)
```

## Test Results

- 6556 tests passing across 195 test files
- 99%+ code coverage, zero test warnings
- Unit, integration, policy, Discord, security, memory, agent, tools, skills, CLI, voice, scheduler tests
- 54+ property-based tests (hypothesis) for policy engines, security, and rate limiter
- 116 security fuzz tests (unicode evasion, encoding bypass, vault corruption)
- 48 rate limiter stress tests (concurrent, burst, thread safety)
- 54 concurrency safety tests (checkpoint, cost tracker, registry, memory, circuit breaker)
- 36 resilience tests (corrupted state files, edge case data, recovery paths)
- 135 end-to-end integration tests
- 570+ security edge-case tests (injection, secrets, vault, SSRF, path traversal, tool output injection, webhook hardening, scheme restriction, kwargs allowlist, file policy enforcement, shell brace groups, header filtering, gateway thread safety, cost tracker edge cases, env sanitization, chunked response limits, overlapping redaction, snowflake validation, LShift DoS guard, shell quoting, multimodal token injection, authority claims)

## Session 26 Additions (2026-03-15)

- **6 security vulnerability fixes from audit**:
  - Token file TOCTOU: anthropic_auth and oauth now use os.open(O_CREAT|O_WRONLY|O_TRUNC, 0o600) instead of write_text()+chmod() to eliminate window where token file is briefly world-readable
  - File read tool: O_NOFOLLOW flag atomically prevents symlink swap between resolve() and open()
  - File delete tool: O_NOFOLLOW verification before unlink prevents symlink swap attacks
  - Discord bot token logging: Stopped logging first 8 chars of token at INFO level (leaked bot user ID)
  - AT-SPI tools: 5 silent exception handlers replaced with logger.debug() calls
  - MCP client: Startup check detects immediate process exit before handshake
- **7 new injection detection patterns**: Multimodal token injection (<|image|>, <|audio|>, <|video|>), structural injection (<|separator|>, <|context|>), uppercase override mode, authority claim injection. Total: 98 patterns
- **5 new secret detection patterns**: PlanetScale, Neon, Postmark, Render, Fly.io tokens. Total: 50 patterns
- **Tests (144 new)**: Security fix tests (22), edge case tests (37), security pattern tests (27), integration pipeline tests (58)
- **Total new tests**: 144 (from 6412 to 6556) across 4 new test files
- **7 commits, zero ruff lint errors**

## Session 25 Additions (2026-03-15)

- **5 security vulnerability fixes from code audit**:
  - Self-create tool: expanded blocklist with 10 new patterns (__builtins__, open(, os.exec/fork/spawn/popen, shutil.rmtree/move, os.remove/unlink/rmdir)
  - X11 click/type tools: explicit int() coercion on x, y, delay_ms parameters to prevent shell injection via string params
  - FallbackSandbox: environment sanitized to safe-only vars, preventing API key leakage to arbitrary commands
  - Code evolution: is_relative_to() check blocks path traversal (../../etc/cron.d/backdoor)
  - Code evolution: test subprocess environment sanitized to safe-only vars
- **6 silent exception handler fixes**: Added debug logging with exc_info to runtime (streaming fallback, checkpoint init), code_evolution (_load, _revert_diffs), sanitizer (URL/HTML decode)
- **Magic number extraction**: Checkpoint age thresholds extracted to _RESUME_THRESHOLD_SECS (3600) and _RESTART_THRESHOLD_SECS (86400)
- **Input validation hardening**:
  - Gateway: timeout positivity validation, URL length limit (8192 chars)
  - Runtime: empty user_input validation on run() and run_stream()
  - Browser: session_id length limit (128 chars)
- **Tests (96 new)**: Validation tests (22), coverage gap tests (52), security fix tests (22)
- **Total new tests**: 96 (from 6316 to 6412) across 3 new test files
- **6 commits, zero ruff lint errors**

## Session 24 Additions (2026-03-15)

- **5 security vulnerability fixes from code audit**:
  - TTS env sanitization: `_ensure_runtime_dir()` now filters to safe-only env vars, preventing API key leakage to espeak-ng/piper/gst-launch subprocesses
  - X11 Ollama vision: `_call_ollama_vision()` routed through PolicyHTTPClient instead of raw httpx.post(), enforcing network policy and audit logging
  - Discord REST snowflake validation: All public methods validate channel_id/message_id against `^\d{1,20}$` to prevent URL path traversal
  - X11 shell quoting: Replaced json.dumps() with shlex.quote() in X11ClickTool/X11TypeTool to prevent shell metacharacter injection
  - Calculator LShift guard: Added `_MAX_SHIFT=10000` to prevent memory exhaustion DoS (e.g. `1 << 10000000000`)
- **9 new injection detection patterns**: tool-call/result token, function_calls XML, pad token, urgency-prefixed override, meta-AI instruction, diff_marker, tool_use XML, antThinking. Total: 91 patterns
- **5 new secret detection patterns**: Netlify token, Sentry DSN, Algolia API key, age secret key, Doppler token. Total: 45 patterns
- **5 additional security fixes**: Browser env sanitization, audit log directory permissions (0o700), MCP silent exception logging, Discord gateway max_size (4MB), Discord REST mention regex fix
- **Tests (148 new)**: CLI Discord integration (33), security pattern tests (26), hardening edge cases (35), security fix verification (20), resource leak tests (12), edge case tests (22)
- **Total new tests**: 148 (from 6168 to 6316) across 6 new test files
- **16 commits, zero ruff lint errors**

## Session 23 Additions (2026-03-15)

- **Discord REST security fix**: upload_file, add_reaction, delete_message now use policy-enforced `self._http` instead of raw httpx — previously bypassed network policy enforcement and audit logging
- **Piper TTS timeout**: Added `asyncio.wait_for(timeout=60)` to subprocess communicate() call to prevent indefinite hangs; process killed on timeout
- **Hardening tests (40 new)**: Piper TTS (3), agent runtime tool loop edge cases (6), Discord gateway opcodes (11), Discord REST retry logic (3), network policy (6), MCP lifecycle (6), scheduler parser (5)
- **Security tests (37 new)**: Code evolution path traversal prevention (5), code evolution lifecycle (5), vault operations (6), input sanitizer (6), secrets detector (5), censor pipeline (3), circuit breaker state machine (4), provider config (4)
- **Total new tests**: 77 (from 6091 to 6168) across 2 new test files
- **3 commits, zero ruff lint errors**

## Session 22 Additions (2026-03-15)

- **PEP 561 py.typed marker**: Added `py.typed` marker file and `pyproject.toml` package-data entry for type checker support
- **`__all__` exports**: Added `__all__` to 4 empty `__init__.py` files (tools/, channels/, security/, providers/)
- **Silent exception handler fixes**: Added debug logging to 2 silent exception handlers in CLI (Discord queue error, provider availability check)
- **Type hint improvements**: Added return type annotations to atspi helpers (_get_desktop, _find_application, _get_focused_application) and browser tools (get_page, _page)
- **ReDoS fix (MEDIUM)**: HTML comment pattern changed from `[\s\S]*?` to `(?:(?!-->)[\s\S])*` to prevent catastrophic backtracking
- **ReDoS fix (LOW)**: Prompt extraction pattern bounded `(\w+\s+)*` to `(?:\w+\s+){0,10}` to prevent exponential backtracking
- **WebSocket max_size (HIGH)**: Added `max_size=1MB` to `websockets.serve()` to prevent memory exhaustion from oversized frames
- **Audit log tail-read**: Replaced full-file `read_text()` with seek-from-end `_read_tail_lines()` to prevent memory exhaustion on large audit logs
- **Audio log TOCTOU fix**: Audio log files now created with `os.open(O_CREAT|O_EXCL, 0o600)` instead of write+chmod for atomic permission setting
- **Coverage gap tests (19 new)**: Edge client malformed JSON, voice channel start failure, registry save error, network policy IP parse, Discord voice commands, vault crypto unavailable
- **Security hardening tests (13 new)**: ReDoS resistance, WebSocket max_size verification, audit tail-read round-trips, atomic audio log write verification
- **Edge case tests (36 new)**: Config hot-reload callback, memory store 100KB content, rate limiter refill boundary, tool registry double-register, circuit breaker half-open recovery, provider registry all-unavailable, scheduler parse errors, context manager zero budget, MCP shutdown idempotency, cost tracker negative tokens
- **Property-based tests (9 new)**: Hypothesis invariants for sanitizer (never raises, output ≤ input, returns list), secrets detector (never raises, redact idempotent), censor (never raises), scheduler parser (never crashes)
- **Tool result truncation**: Agent runtime now truncates tool results exceeding 200K chars to prevent memory exhaustion (6 new tests)
- **Total new tests**: 83 (from 6008 to 6091) across 7 new test files
- **10 commits, zero ruff lint errors**

## Session 21 Additions (2026-03-15)

- **New secret detection patterns (6)**: Grafana (glc_), Confluent API, Datadog API/app, New Relic (NRAK-), PagerDuty API, SSH public key content. Total: 40 patterns
- **New injection detection patterns (6)**: Prompt extraction via output/repeat/translate, creative extraction via poem, encoding extraction (base64/hex/rot13), forced behavior change. Total: 82 patterns
- **Concurrency safety tests (54 new)**: Checkpoint concurrent create/update/abandon (2), cost tracker concurrent record/budget/summary (3), provider registry concurrent register/rotate (2), memory store concurrent add/search/cleanup (2), tool registry concurrent register/list (1), circuit breaker concurrent transitions (1), plus scheduler/gateway/vault/rate limiter/sanitizer/policy/secrets/censor/resilient store edge cases (43)
- **Resilience tests (36 new)**: Scheduler corrupted/empty/invalid JSON files (6), checkpoint DB lifecycle (5), memory store unicode/empty/large content/FTS5 special chars (7), config loading edge cases (3), MCP manager corrupted config (3), vault concurrent ops and edge cases (3), cost tracker unknown model/zero/large tokens (4), circuit breaker recovery transitions (3), audit logger unicode/empty events (2)
- **Security hardening tests (30 new)**: New secret patterns positive/negative (10), new injection pattern tests (15), redaction tests (3), combined pipeline tests (2)
- **Debug logging**: Added logging to silent exception handlers in anthropic_auth
- **AUDIT_CONNECTIVITY.md**: Comprehensive update with all security layers, shell/filesystem controls, secret protection
- **Agent subsystem tests (32 new)**: Runtime init edge cases (6), context manager (4), done criteria (4), learnings extraction (5), prompt patches lifecycle (4), sub-agent parsing (3), skills registry (2), event bus (4)
- **Permission hardening**: Vault _save_store and wizard _write_config_atomic now explicitly fchmod(0o600) before writing sensitive data
- **Permission tests (7 new)**: Vault data/key file permissions, config atomic write permissions, scheduler jobs permissions
- **Critical security fix**: user_input now sanitized in AgentRuntime.run() and run_stream() — previously the InputSanitizer was only applied to tool output, not the primary attack surface (user input)
- **Webhook sender validation**: Length capped at 64 chars, control characters stripped, log injection prevented
- **Audit fix tests (20 new)**: Runtime sanitization (4), webhook sender validation (8), sanitizer integration (8)
- **Total new tests**: 179 (from 5829 to 6008) across 6 new test files
- **12 commits, zero ruff lint errors**

## Session 20 Additions (2026-03-15)

- **Shell exec env sanitization**: Subprocess environment filtered to safe-only variables (PATH, HOME, LANG, TERM, etc.) — prevents API key leakage to arbitrary shell commands
- **Gateway chunked response enforcement**: Body size now checked when Content-Length header is absent (chunked transfer encoding), closing bypass of 50MB response size limit
- **New secret detection patterns (6)**: Vercel, Cloudflare, Shopify (4 prefixes), Google OAuth client_secret, HashiCorp Vault (hvs/hvb/hvr), Firebase. Total: 34 patterns
- **Overlapping redaction span merging**: SecretsDetector.redact() now merges overlapping match spans before replacement, preventing partial secret leakage from offset corruption
- **Webhook reverse proxy support**: New `trust_proxy` parameter and X-Forwarded-For parsing for correct rate limiting behind reverse proxies
- **Sanitizer URL/HTML decoding**: check_for_injection() now decodes URL-encoded (%XX) and HTML-entity (&lt; etc.) text before scanning, catching encoded evasion attempts
- **7 new injection patterns**: Few-shot conversation injection, separator + role injection, URL-encoded delimiter detection, code-block disguise (```system), payload concatenation, context override. Total: 76 patterns
- **MCP block_injection mode**: McpManager supports optional `block_injection=True` to reject (not just warn) tool outputs containing injection patterns
- **Edge case tests (20 new)**: Checkpoint _row_to_dict JSON fallback (5), scan_for_recovery DB/event failures (4), scheduler _load_jobs security checks (5), code evolution load/traceback parsing (5), provider registry constructor failure (1)
- **Security hardening tests (43 new)**: Shell env sanitization (5), gateway chunked response (6), new secret patterns (20), overlapping redaction (5), webhook XFF (7)
- **Sanitizer tests (22 new)**: New injection patterns (10), URL-decoding preprocessing (3), HTML-entity decoding (3), pattern count (1), MCP block mode (5)
- **Total new tests**: 85 (from 5744 to 5829) across 3 new test files
- **8 commits, zero ruff lint errors**

## Session 19 Additions (2026-03-15)

- **New secret detection patterns (5)**: HuggingFace (hf_), Databricks (dapi), DigitalOcean (dop_v1_), Linear (lin_api_), Supabase (sbp_). Total: 28 patterns
- **New injection detection patterns (10)**: Korean language, trigger-based injection, conditional override, memory poisoning, future response control, FIM tokens (prefix/middle/suffix), endofprompt token, role confusion attacks. Total: 69 patterns
- **Gateway response size limits**: All HTTP methods (sync + async) now check Content-Length against configurable max_response_bytes (default 50MB). Prevents memory exhaustion from malicious/oversized responses
- **Coverage gap tests (24 new)**: Scheduler _load_jobs error paths (invalid JSON, non-list, non-dict records, malformed records, OSError on stat), voice registry corrupt JSON, voice channel loop error, discord voice resample break branch, voice commands fallthrough, network policy IP parsing, discord channel agent runtime
- **Security hardening tests (32 new)**: All 5 new secret patterns with positive/negative tests, all 10 new injection patterns, redaction verification, combined security pipeline tests, obfuscated/base64 injection detection
- **Gateway hardening tests (16 new)**: Response size limit boundary tests (within/exceeding/exactly at limit, no Content-Length, non-numeric, zero), integration tests for GET/POST/aget/apost, pool limits, allowed schemes, allowed kwargs
- **Edge case tests (45 new)**: Shell policy compound commands (18 tests: newline, pipe, or, background, subshell, backtick, heredoc, herestring, process substitution, brace group), webhook rate limiting (3 tests), circuit breaker state machine (5 tests), memory store FTS5 (5 tests), sanitizer obfuscation resistance (5 tests), cost tracker (5 tests), config hot-reload (1 test), provider errors (3 tests)
- **Tool coverage tests (26 new)**: Full coverage for DiscordUploadTool (no token, success, FileNotFoundError, generic exception, empty caption, missing response ID, metadata) and SelfCreateTool (list empty/populated/corrupt, delete empty/existing/missing, create invalid name/language/script, 5 dangerous pattern categories, successful python/bash creation, unknown action, metadata)
- **Browser session_id validation**: Added regex validation to prevent directory traversal attacks via session_id parameter (e.g., `../../etc/passwd`). 7 new tests
- **Total new tests**: 150 (from 5594 to 5744) across 6 new test files
- **Robustness fixes**: 4 unguarded JSON parse calls in edge_client wrapped in try-except, JWT parse failure logging added to OAuth module, silent `pass` in tool registry replaced with debug log
- **Security hygiene**: Replaced realistic API key placeholders in provider docstrings with `<REDACTED>`
- **12 commits, zero ruff lint errors**

## Session 18 Additions (2026-03-15)

- **Webhook null prompt fix**: Fixed crash on `prompt: null` in webhook handler (AttributeError on NoneType.strip())
- **Gateway async completeness**: Added `aput()` and `ahead()` async methods to PolicyHTTPClient
- **InputSanitizer expansion**: 7 new injection patterns — tool/function abuse, Anthropic delimiters, Human:/Assistant: turn injection, prompt leaking/exfiltration, Japanese injection. Total: 59 patterns
- **SecretsDetector expansion**: 3 new patterns — Azure AccountKey, Twilio SK, Mailgun key. Total: 23 patterns
- **WebFetchTool refactor**: Moved _BLOCKED_HEADERS to class-level frozenset constant
- **AgentRuntime type hint**: Added Iterator[str] return type to run_stream()
- **Edge case tests (101 new)**: Gateway async aput/ahead (6), injection pattern tests (17), secret pattern tests (7), WebFetchTool header tests (3), gateway thread safety (8), cost tracker edge cases (13), shell policy edge cases (12), MCP client edge cases (8), webhook handler edge cases (10), regression tests (6), Piper env sanitization (2), scheduler task validation (2), device registry perms (1)
- **Security audit**: 12 findings identified (2 High, 8 Medium, 2 Low); 7 fixed in code
  - Piper TTS env sanitization (prevents API key leakage to subprocess)
  - Device registry fchmod(0o600) on save
  - CostTracker records capped at 10K (prevents memory exhaustion)
  - Scheduler task length validation (max 50K chars)
  - OAuth state CSRF verification (callback state checked against expected)
  - MCP tool name validation at import time (rejects invalid/ambiguous names)
  - File tool TOCTOU fix (resolve symlinks before I/O to match policy check)
- **Updated reports**: SECURITY.md (v0.3.0), AUDIT_SECURITY.md, all counts updated
- **Zero ruff lint errors**

## Session 16 Additions (2026-03-15)

- **Fresh security audit**: Comprehensive code review identifying 13 findings (2 High, 6 Medium, 5 Low)
- **Shell heredoc bypass fix**: Added `<<` to _SUBSHELL_MARKERS to block heredoc injection
- **Shell brace group fix**: Brace group scanning now checks entire command, not just start position
- **FTS5 query injection fix**: Query input wrapped in double-quotes and escaped; OperationalError caught gracefully
- **Format string injection fix**: Proactive trigger templates use string.Template instead of str.format()
- **Scheduler jobs file permissions**: Atomic write with tempfile+rename, restrictive 0o600 permissions
- **MCP response ID validation**: Response ID checked against request ID, mismatch logged as warning
- **Self-create tool content validation**: Script content scanned for 15+ dangerous patterns before writing (curl, wget, eval, exec, os.system, subprocess, socket, chmod +s, /dev/tcp, etc.)
- **Code evolution logging**: SystemExit now logged as warning instead of silently suppressed
- **Webhook log censoring**: censor_response() applied to log output to prevent secret leakage
- **Device registry permission checks**: File ownership and group/world-writable checks on load
- **Audio log file permissions**: Directory created with 0o700, files with 0o600
- **Total new tests**: 98 (from 5334 to 5432) across 3 new test files
- **All 11 audit findings addressed in code**
- **Zero ruff lint errors**

## Session 15 Additions (2026-03-15)

- **Security audit**: Full codebase security audit identifying 17 findings (3 Critical, 5 High, 6 Medium, 3 Low)
- **H2 fix: File tool policy enforcement**: ToolRegistry._check_permissions now validates the actual `path` kwarg against filesystem policy engine (check_read/check_write), not just static allowed_paths
- **H3 fix: Gateway kwargs allowlist**: _sanitize_kwargs replaced blocklist with explicit allowlist of safe kwargs (headers, params, data, json, content, cookies, timeout, files, extensions); strips verify, base_url, transport, auth, event_hooks
- **H1 fix: Shell heredoc/brace rejection**: Added `<<<` (here-strings) and `{ }` / `{;` (brace groups) to shell policy rejection list
- **H4 fix: Webhook header filtering**: Message metadata now only stores safe headers (Content-Type, User-Agent, X-Request-Id, X-Missy-Signature); Authorization, Cookie, X-Forwarded-* stripped
- **M5 fix: Webhook Content-Type validation**: Requires Content-Type: application/json; rejects with 415 otherwise (CSRF prevention)
- **M6 fix: Webhook Content-Length validation**: Validates as non-negative integer; rejects with 400 for non-integer/negative values
- **L3 fix: Audit event redaction**: Tool audit event detail messages run through censor_response() to prevent secret leakage
- **Runtime coverage**: Tests for tool output injection scanning, httpx ImportError fallback, get_tool_registry error paths
- **Voice command coverage**: Tests for unrecognized commands, voice=None, !say errors
- **Network policy coverage**: Tests for unparseable IP addresses from getaddrinfo
- **L1 fix: Vault hard link detection**: Vault rejects key files with st_nlink > 1 to prevent hard link attacks
- **M1 fix: MCP server name validation**: add_server validates against _SAFE_NAME_RE (alphanumeric, hyphens, underscores only)
- **Total new tests**: 58 (from 5276 to 5334) across 4 new test files
- **Security findings addressed**: 11 of 17 findings from comprehensive security audit
- **Zero ruff lint errors**

## Session 14 Additions (2026-03-15)

- **Tool execution retry**: Transient errors (TimeoutError, ConnectionError, OSError, httpx.TimeoutException) automatically retried up to 2 times with exponential backoff (1s, 2s). Non-transient errors (KeyError, ValueError) fail immediately
- **DNS rebinding fix**: Check ALL resolved IPs before allowing access. If ANY resolved address is private/reserved and not in allowed_cidrs, deny the entire request. Prevents mixed-record attacks where a hostname resolves to both public and private IPs
- **Gateway connection pool limits**: Explicit httpx.Limits(max_connections=20, max_keepalive_connections=10, keepalive_expiry=30) to prevent resource exhaustion
- **Gateway DELETE/PATCH methods**: Added sync (delete/patch) and async (adelete/apatch) HTTP methods with full policy enforcement, kwargs sanitization, and audit events
- **Webhook rate tracker cleanup**: Added _evict_stale_ips() to prevent unbounded memory growth; triggers when tracked IPs exceed 10K, removing IPs with all-expired timestamps
- **Prompt injection sanitizer**: 13 new patterns — unclosed HTML comments with keywords, data: URIs, hidden div detection, Llama 3 tokens (<|begin_of_text|>, <|start_header_id|>, <|end_header_id|>, <|reserved_special_token), chained instruction patterns (new/updated/revised/real instructions:), Portuguese and Russian injection keywords
- **Scheduler input validation**: Hour (0-23) and minute (0-59) range validation for daily/weekly schedules; zero-interval rejection
- **Total new tests**: 66 (from 5166 to 5232) across 3 new test files
- **Zero ruff lint errors**

## Session 13 Additions (2026-03-15)

- **SSRF prevention**: Gateway now rejects non-http/https URL schemes (blocks file://, ftp://, data:// attacks)
- **Redirect bypass prevention**: Gateway strips `follow_redirects` from kwargs, explicitly sets `follow_redirects=False` on httpx clients
- **Shell policy bare `&` fix**: Background execution operator now properly splits compound commands (prevents `allowed_cmd & forbidden_cmd` bypass)
- **Shell launcher command warnings**: Policy engine warns when command-launching programs (env, bash, sudo, find, xargs) are whitelisted
- **MCP config permission checks**: Manager verifies file ownership and permissions (rejects non-owner, group/world-writable) before loading mcp.json
- **New secret patterns**: Added GitLab (glpat-), npm (npm_), PyPI (pypi-), SendGrid (SG.), database connection string detection
- **Websockets deprecation fix**: Updated import to use new `websockets.asyncio.server.ServerConnection` API with fallback
- **Test warning elimination**: Fixed all 4 test warnings (websockets deprecation, unawaited coroutines) — now zero warnings
- **Coverage gap tests** (46 new): vault atomic write, symlink detection, config hotreload owner/stat, settings vault resolution, runtime cost recording, tool output injection, shell empty parts, code evolution malformed paths, voice server pre-auth close, audio param fallback, MCP timeout, discord voice init failure, resampling break branch, vault crypto unavailable
- **Security hardening tests** (70 new): URL scheme restriction, kwargs sanitization, follow_redirects enforcement, bare & split, launcher warnings, MCP config permissions, all 5 new secret patterns
- **Total new tests**: 116 (from 5029 to 5145)
- **Zero test warnings** (was 4)
- **Zero ruff lint errors**

## Session 12 Additions (2026-03-15)

- **Tool output injection scanning**: Agent runtime now scans tool results for prompt injection patterns and prepends security warning labels to suspicious content
- **Response censoring**: `censor_response()` applied to agent runtime output to prevent secret leakage in final responses
- **Shell policy hardening**: Added process substitution markers (`<(`, `>(`, `<<(`) to subshell deny list, blocking bash process substitution attacks
- **MCP security hardening**:
  - Sanitized subprocess environment (only safe vars like PATH, HOME passed)
  - Server name validation (rejects `__` to prevent namespace collision)
  - RPC read timeout (30s) and response size limit (1MB) to prevent DoS
- **OpenAI key detection**: Extended `sk-proj-...` format detection in SecretsDetector
- **Subprocess timeouts**: Added 30s timeout to x11_tools._run(), 10s to xdotool polling in x11_launch
- **Flaky test fix**: Fixed voice channel test race condition (run_forever vs run_until_complete)
- **Test connection leak fix**: Replaced 23 manual sqlite3 conn.open/close patterns with context managers in checkpoint tests
- **New tests**: 19 security hardening tests (shell process substitution, MCP name validation, tool output injection, response censoring)
- **Vault security**: Atomic write with mkstemp+rename, TOCTOU-safe key creation with O_CREAT|O_EXCL, symlink rejection
- **Config hotreload safety**: Rejects symlinks, non-owner files, and group/world-writable config before reload
- **Settings vault:// resolution**: Provider API keys now resolve vault:// and $ENV references
- **Webhook integration tests** (26 new): End-to-end HTTP tests for rate limiting (429), oversized payloads (413), queue overflow (503), HMAC signatures, server lifecycle
- **Hotreload safety tests** (2 new): World-writable rejection, symlink rejection
- **InputSanitizer hardening**: Added patterns for base64-encoded injections, data URIs, unicode homoglyph obfuscation, markdown/HTML hidden instructions, multi-language injection
- **Rate limiter edge case tests** (78 new): Zero-limit unlimited mode, negative max_wait, concurrent acquire, refill calculation, RPM/TPM interaction
- **Sanitizer tests** (300+ lines new): Comprehensive testing of all new injection patterns
- **Total new tests**: 137 (from 4892 to 5029)
- **Warnings reduced**: From 7 to 4 (2 remaining are from websockets library deprecation)

## Session 11 Additions (2026-03-15)

- **Input validation hardening**: Clamped voice WebSocket `sample_rate` [8000,48000] and `channels` [1,2] with safe defaults for non-numeric values; clamped code evolution `confidence` to [0.0, 1.0]
- **Bare except fixes**: Replaced 2 remaining `except: pass` blocks with `logger.debug()` in CLI doctor and voice status
- **Lint fixes**: Fixed raise-from in tool registry fail-closed, removed unused X11 imports in security tests
- **MCP tests** (54 new): Dedicated test files for MCP client (28 tests: connect, RPC, notify, call_tool, disconnect) and manager (26 tests: connect_all, add/remove/restart, health_check, namespacing, shutdown, persistence)
- **Prompt patches tests** (36 new): Full coverage of propose/approve/reject, auto-approval logic, expiration, record_outcome, build_prompt, persistence round-trips
- **Circuit breaker tests** (67 new): State machine transitions, exponential backoff, timeout caps, thread safety (concurrent failures, deadlock detection), edge cases
- **Tool security edge-case tests** (88 new): Path traversal, null byte injection, unicode confusion, long paths, special chars, SSRF prevention, type confusion, mode injection, encoding injection, special file protection
- **Context manager tests** (23 new): Token approximation, budget defaults, history pruning, memory/learnings injection, truncation
- **Sub-agent tests** (19 new): parse_subtasks (numbered, connectives, fallback), SubTask dataclass, runner (context, errors, caps)
- **Approval gate tests** (18 new): PendingApproval lifecycle, threading, handle_response, timeout cleanup, send_fn failure
- **Learnings tests** (25 new): TaskLearning dataclass, extract_task_type priority, extract_outcome keywords
- **Done criteria tests** (20 new): DoneCriteria state tracking, is_compound_task patterns, prompt generators
- **Total new tests**: 402 (from 4489 to 4891)
- **Zero ruff lint errors**

## Session 10 Additions (2026-03-14)

- **Lint cleanup**: Fixed all 46 ruff errors (import sorting, SIM103/105/117, unused variables)
- **CLI coverage tests** (7 new): proactive callback success/fallback paths, doctor watchdog/voice/checkpoint exception handling
- **Browser tools coverage tests** (22 new): display setup, session start/close, page helper, registry edge cases
- **Discord voice coverage tests** (19 new): start full paths, join channel-id-not-found, start_listening body, watchdog router exception, speech handler branches (empty transcript, no callback, empty response), resample PCM boundary
- **Proactive manager coverage tests** (17 new): watchdog import success path, threshold loop inner break, audit publish/exception, file handler with mocked watchdog
- **Remaining gap coverage tests** (21 new): heartbeat loop fire, overnight active hours, webhook log_message, config watcher OSError, Discord voice callback closure, DM policy fallthrough, mention fallback, gateway heartbeat loop
- **Test ordering fix**: Fixed proactive stub test collision with reimport tests
- **Additional coverage tests** (24 new): edge client import/main, wizard prompt/verify/guild/OAuth paths, anthropic auth, config error wrapping, memory malformed record, plugin event failure, network policy CIDR type mismatch, ollama/openai provider paths, scheduler parser/retry, vault crypto unavailable, registry key rotation
- **Total new tests**: 110 (from 4379 to 4489)
- **Coverage**: 98.3% → 99.11% (196 missed → 102 missed)

## Session 9 Additions (2026-03-14)

- **Security fuzz tests** (116 new): unicode homograph evasion, RTL override, combining diacritics, whitespace variants, URL/base64 encoding bypass, large input stress (250K chars), secret near-miss patterns, vault corruption recovery (truncated ciphertext, flipped bits, wrong keys), hypothesis property-based invariants for sanitizer/detector/vault
- **Rate limiter stress tests** (48 new): concurrent acquire with 8-20 threads, token budget exhaustion, refill accuracy verification, burst handling, zero-limit unlimited mode, edge cases (negative tokens, max_wait=0), 429 response handling, thread-safety interleaved acquire+record_usage, hypothesis properties
- **Gateway/watchdog coverage tests** (42 new): sync PUT, async POST, async close, context managers (sync + async), category forwarding, URL validation edge cases, watchdog recovery detection, failure threshold escalation, audit event publish failure handling
- **Proactive manager tests** (37 new): schedule loop stop (line 323), file handler (lines 440-454), watchdog unavailable fallback, disk/load threshold polling, observer stop exception handling, cooldown enforcement, confirmation gate deny path, agent callback error handling
- **Voice registry tests** (18 new): atomic write failure with temp cleanup, purge_audio_logs stat/unlink errors, non-file entry filtering, integration round-trips
- **End-to-end integration tests** (77 new): security pipeline (sanitizer→detector→censor), policy enforcement chain, memory lifecycle, circuit breaker state machine, cost tracker budget enforcement, tool registry with policy, scheduler lifecycle, audit event flow, config mutation, multi-layer security
- **Error handling hardening**: Replaced 8 bare `except: pass` blocks with `logger.debug()` calls in watchdog, CLI init, browser tools, self_create_tool, x11_tools
- **Incus tools coverage tests** (54 new): unreachable fallbacks, network attach/detach parsing, volume operations, profile set/edit, project config, device validation, copy/move flags
- **Targeted coverage gap tests** (35 new): Discord REST error paths, config api_keys fallback, filesystem policy ValueError, skills registry audit errors, sandbox generic exceptions, voice command guards
- **Total new tests**: 412 (from 3967 to 4379)
- **Coverage**: 97% → 98.3%

## Session 8 Additions (2026-03-14)

- **Zero lint errors**: Fixed all 210 ruff errors (was 210, now 0)
- **Security hardening**: 8 new injection patterns, 6 new secret detectors
- **Property-based tests** (54 new): hypothesis-driven tests for all policy engines
- **Security edge-case tests** (92 new): unicode homograph attacks, zero-width injection
- **Total new tests**: 146 (from 3821 to 3967)

## Session 7 Additions (2026-03-14)

- **Total new tests**: 740 (from 3035 to 3775)
- **Coverage**: 86% → 97%
- **Code quality**: ruff format, contextlib.suppress, StrEnum, raise-from
- **17 commits** this session

## Remaining Tasks

- Coverage target of 90% exceeded (98.92% achieved)
- Zero ruff lint errors
- Zero TODOs/FIXMEs in codebase
- Discord multi-account support (P3, low demand)
- Web UI / dashboard (P4, intentionally deferred)

## Remaining Coverage Gaps (102 lines)

Most remaining gaps are in complex async integration code (Discord run loop: 64 lines), platform-dependent tools (atspi: 7, incus: 9), and defensive dead code paths.

## Next Actions

- Project is feature-complete, well-tested, and hardened
- Consider mutation testing to verify test quality
- Consider adding CLI `run` command Discord integration tests
- Consider adding more atspi/incus tool coverage
