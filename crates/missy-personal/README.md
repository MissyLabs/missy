# Explicit PERSONAL memory

Independent native Rust SQLite memory keyed **only by authenticated tenant + user**.
This crate does not modify `Scope`, exact-scope `Store` memory, runtime defaults, or the
system prompt. A different session or channel belonging to the same principal can recall
these explicit personal records. A different user or tenant cannot get, search, replace,
or delete them, including by guessing an ID. There is no global namespace or fallback.

## Integration API

- `PersonalMemory::open(path) -> Result<PersonalMemory>` takes an explicit absolute database
  path in an existing owned private directory. It creates a new owned private database,
  never imports legacy state, and refuses foreign/corrupt schema or metadata.
- `PersonalTools::new(Arc<PersonalMemory>) -> PersonalTools` implements `ToolExecutor`.
  Construct/register it **only after an explicit operator extension grant**, for example a
  separate `personal_memory_enabled = true` configuration field, disabled by default.
  Construction grants cross-channel and cross-session PERSONAL access for the authenticated
  user. Do not infer this grant from existing exact-session memory or retrieval enablement.
- `put(&Scope, id, content) -> Result<PersonalRecord>` explicitly creates/replaces a record.
- `get(&Scope, id) -> Result<Option<PersonalRecord>>` reads only that principal's ID.
- `search(&Scope, query, limit)` and `recall(&Scope, query, limit)` return
  `Result<Vec<PersonalRecord>>`. Recall is a read-only API, not an automatic prompt hook.
- `delete(&Scope, id) -> Result<bool>` removes only that principal's record and FTS entry.
- `PersonalNamespace::from_scope(&Scope)` creates a typed, non-deserializable authority.
  Its fields are private. A separately serializable output view is not authority.

**Every `Scope` passed here must come from the authenticated adapter/ToolContext.** A type
cannot authenticate a network request. No tool accepts principal, tenant, user, namespace,
channel, session, trust, or provenance arguments; all unknown fields are rejected. Scope
validation remains exactly as in `missy-core`. These APIs are synchronous like `Store`;
callers requiring an async blocking pool should dispatch at their executor boundary.

Tools: `personal_memory_put` (`id`, `content`), `personal_memory_get` (`id`),
`personal_memory_search` (`query`, optional `limit`, default 10), and
`personal_memory_delete` (`id`). Put/delete are marked side effects for the runtime journal;
get/search are read-only. Strict preflight validates input without executing effects.
Cancellation is checked immediately before dispatch, not a rollback guarantee after SQLite
has committed. There are no hidden network services, providers, embeddings, or background jobs.

## Provenance and prompt boundary

Each returned record includes its current content, a full `source_scope` of the most recent
explicit put, `namespace` output label, fixed `provenance = explicit_user_memory`,
`trust = untrusted_data`, SHA-256 of the content, and whole-content UTF-8 `start_byte` /
`end_byte` citation offsets. `source_scope` is **not** relabeled to the current caller's
session/channel at recall time. The tool result is additionally framed as untrusted data.
An instruction saved in memory remains data, never authority. Content cannot forge the
outer JSON framing because serde escapes it.

There is deliberately **no automatic context injection** in this crate. A future separate
read-only context-provider trait may call `recall` with authenticated scope, then append the
bounded records after the system-prompt boundary as untrusted source material. Do not append
them to the system prompt or replay their text as new user instructions.

## Storage and limits

Uses a distinct SQLite application ID, exact versioned schema/metadata checks, full
synchronous transactions, checksums binding principal/ID/content/source scope, FTS5 lexical
Unicode search, and explicit tenant+user SQL predicates on every record query and mutation.
Terms are split into alphanumeric words and individually quoted; model text is never FTS
syntax or SQL. Empty/punctuation-only searches return no records. This is lexical search,
not semantic recall. Candidate filtering and ranking statistics are principal-local;
foreign-principal documents cannot perturb the authenticated user's ranking. The
regression test and reproduced pre-fix side channel are in
`docs/rust/final-security-review.md`.

Limits are enforced in UTF-8 bytes at runtime: 256-byte IDs without control characters,
8192-byte nonempty content, 1024-byte queries, 1..20 results, 512 records / 2 MiB content per
principal, 64 KiB serialized output, 64 KiB serialized tool arguments, plus a 256 MiB SQLite
page cap. Search returns only complete records within the output budget, possibly fewer
than requested. No silent text truncation or malformed UTF-8 citations.

The database is private, owner-checked, regular, singly linked and opened with SQLite
`NOFOLLOW`; ancestor symlinks/dot components and unsafe sidecars are rejected. The immediate
directory must be owned and 0700-equivalent; files must have no group/other permissions.
File identity and permission/sidecar checks repeat on each operation. This follows the
repository's secure-opening pattern, **not a sandbox against a malicious process running
as the same OS user** or an attacker able to rename ancestor directories. Protect the
state directory and its ancestors from untrusted writers. SQLite secure-delete is enabled
but this is not a forensic secure-erasure guarantee for SSDs, snapshots, or backups.

The executor's opaque configuration fingerprint binds database path, explicit namespace
grant, immutable capability/limit policy and tool definitions. Path/capability changes must
invalidate continuation, rather than silently continuing against another personal store.

Validation: `cargo test -p missy-personal`; no live state required. Tests use disposable
private directories and cover restart/session/channel recall, tenant/user isolation,
update/delete isolation, unknown principal arguments, framing, bounds, FTS quoting/Unicode,
metadata/content tampering, symlinks/hardlinks, permissions/ownership, and replaced files.
