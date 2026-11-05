# Scoped learning DATA retrieval

Automatic retrieval is opt-in. Runtime configuration defaults to
`memory_context_enabled: false`, `memory_context_max_records: 6`,
`memory_context_max_bytes: 8192`, and `memory_context_candidate_limit: 128`.
Accepted bounds are respectively 1..20 records, 512..16384 serialized bytes,
and 1..256 candidates. Omitted fields retain backwards-compatible defaults.

`memory_context::snapshot(&Store, &Scope, prompt, &RuntimeConfig)` produces an
`untrusted_scoped_memory` JSON envelope or an empty string. Candidates are the
bounded newest records for the **entire exact tenant/user/channel/session Scope**,
ordered by updated time, created time, then ID. There is no broader user/channel
fallback, cross-session lookup, shared corpus scoring, model call, or write.

Content is projected through the security untrusted JSON projection before
selection. Relevance is deterministic case-folded alphanumeric token overlap
with the sanitized current prompt. Ties retain candidate order. With no match,
recent maintenance-shaped conversation digests and tool outcome records are
eligible, identified by their native-sleeptime provenance and content kind.
These identifiers are selection hints, not authenticated authorship or trust.
Unrelated ordinary memories are not injected as a no-match fallback.

Each record retains sanitized ID, provenance, trust, timestamps and a SHA-256 of
the original serialized source record. Digests establish lineage, not truth.
All stored string values are security-projected, including nested JSON and
metadata. Known secrets are redacted and recognized injections quarantined.
This is defense in depth, not a perfect instruction firewall. Content remains
historical **DATA**, never system/developer instructions, approval, or authority.
Memory writes cannot set instruction trust. The composer must carry this JSON
in a separate user-role DATA message, not concatenate it into a system prompt.

The snapshot must be captured at turn admission in immutable scoped request
metadata and reused on resume/replay; later memory changes must not silently
change an already admitted request. The runtime integration owns that pinning.

The byte ceiling applies to the complete actual UTF-8 JSON serialization,
including escaping and envelope/metadata overhead. Oversized content is clipped
only after sanitization, with `truncated: true`; oversized metadata that cannot
fit is omitted as a whole record. Empty selections produce no message. Retrieval
is bounded by recent candidates, so older relevant records may be missed.

Explicit `memory_search` also uses an exact-scope, newest 256-record shortlist.
It matches literal contiguous alphanumeric token phrases, ranks by local phrase
occurrences, and uses deterministic recency ties. Caller FTS operators are not
interpreted. No global FTS/BM25 statistics are read, preventing another scope's
writes from affecting ranking. Existing FTS storage remains for compatibility;
it is not consulted for retrieval. Search results are capped at 256.

Focused tests cover real-store scope isolation and foreign-corpus rank
invariance, opt-in defaults and bounds, maintenance fallback, nested injection
and credential projection, data-only trust, record limits, and actual JSON byte
caps with Unicode and escaping. No account, deployment, or provider is needed.
