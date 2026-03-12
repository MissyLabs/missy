# TEST_EDGE_CASES

- Timestamp: 2026-03-12 (Session 3)

## Policy Enforcement Edge Cases
- Blocked domain enforcement
- Blocked CIDR enforcement
- Forbidden shell command rejection
- Forbidden filesystem path rejection
- Scheduler cannot bypass policy
- Per-category network allowlists (provider, tool, discord)

## Provider Edge Cases
- Provider fallback behavior
- API key rotation
- Model tiering (fast/premium)
- Unavailable provider graceful degradation

## Discord Edge Cases
- DM policy enforcement (disabled, allowlist, pairing, open)
- Require-mention filtering
- Bot loop prevention (ignore own messages, ignore other bots)
- Credential detection and message deletion
- Attachment denial by policy
- Guild policy deny for unconfigured guilds
- Thread-aware routing (PUBLIC_THREAD type 11)
- Thread-scoped session isolation

## Agent Loop Edge Cases
- Circuit breaker state transitions (closed->open->half-open)
- Checkpoint recovery classification (resume/restart/abandon)
- Failure tracker strategy rotation after N failures
- Done criteria compound task detection
- Cost budget exceeded enforcement
- Context manager token budget pruning

## Security Edge Cases
- Docker sandbox bind mount policy enforcement
- Docker sandbox network isolation override
- Read-only root filesystem with writable /tmp
- Fallback sandbox when Docker unavailable
- Vault ChaCha20-Poly1305 encryption/decryption
- Input sanitization for 13+ prompt injection patterns
- Secret detection for 9 credential patterns

## Memory Edge Cases
- Session name resolution (friendly name -> UUID)
- FTS5 search with prefix/phrase/boolean operators
- Resilient store auto-repair on primary failure
- Session turn count recalculation
