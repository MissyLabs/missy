# Native exact-scope history recall

`HistoryTools::new(Arc<Store>, HistoryConfig) -> Result<HistoryTools>` is disabled
by default. It uses only the trusted executor `ToolContext.scope`. Every lookup
requires the exact tenant, user, channel **and** session. Arguments cannot override
scope and there is no global, other-session or personal-memory fallback.

## Tools

* `history_search {query, limit?}` searches a **literal phrase**, ASCII case
  insensitive and Unicode byte-exact. Not regex, SQL, FTS syntax or semantic
  similarity. It scans prompts, responses, errors, checkpoint transcripts and
  actual retained tool result JSON/tool errors from the latest configured
  at-most-1000 turns, plus the latest at-most-10000 raw history messages. Scores are
  occurrence count capped at 100 then recency. The current executing turn is
  excluded from turn/checkpoint sources so its own query does not become a hit.
  Match and excerpt intervals are
  explicit UTF-8 byte ranges.
* `memory_describe {item_ids}` returns source count, actual turn/tool identifiers,
  content length, SHA-256 and trust labels for previously issued receipts.
* `memory_expand {item_id, content_sha256, start_byte, end_byte, turn_id?, tool_id?}`
  returns an exact interval from immutable evidence. Intervals are half-open and
  must be valid UTF-8 boundaries. Supplying source IDs additionally verifies them.
  A wrong scope, forged receipt/checksum or source ID is denied. No tool execution
  or network request is involved.
* `context_shunt {query, item_ids}` is advertised only after attaching an explicit
  `SummaryBackend`. It makes exactly **one** completion with an empty tool catalog,
  no continuation and no child worker/runtime. It rejects tool calls, unexpected
  roles, incomplete finish reasons, excessive usage and malformed citations.
  Result citations must name selected receipt IDs and exact UTF-8 intervals.
  Citation validity does not prove the summary's semantic entailment. Summary is
  explicitly labeled untrusted model output.

The existing builtin `memory_search` continues to mean explicit memory records;
this crate intentionally does not register that name.

## Truthful limits and retention

The public Store API supports bounded turn listing and per-turn tool journals,
not full-corpus indexed search. Search reports `full_corpus: false`, older-turn
omission, scan byte limits, oversized source omissions and scanned-source counts.
Checkpoint transcripts are retrieved through public Store.checkpoint using the
expected context hash from that same exact-scope turn's durable checkpoint event.
Existing context-binding/checksum checks remain intact. Transcript content,
assistant tool-call JSON and tool output messages are included; system/developer
policy messages are excluded rather than exposing runtime instructions.

Message-table-only records are read through public Store.history, including user,
assistant and tool messages and assistant tool-call JSON. This API does not expose
source turn/row IDs, so those receipts honestly return `turn_id: null`, use the
full serialized message hash as source identity and preserve actual tool-call IDs
when present. Supplying a turn ID for such an expansion is rejected, not guessed.
Messages outside the turn window can therefore still be recalled from the
separate latest-10000-message window. Window-full metadata means older raw
messages *may* be omitted. Empty search means no match **within the reported
scan**, not proof that evidence never existed.

Source size and scan limits bound matching work after Store deserialization;
the Store API itself materializes complete turn and tool-journal records before
these checks. This is not a streaming database memory bound.

Enabled history requires an operator-configured `retention_path: Some(PathBuf)`.
The direct parent must already exist, be current-user owned and exactly 0700.
All ancestors must be real directories; database and SQLite sidecars must be owned,
0600, regular and single-linked. Symlinks, hardlinks, changed inodes, unknown schema,
SQLite corruption and snapshot checksum corruption fail closed. Disabled history
does not open or create this database. The application places it at
`state_dir/history-evidence.sqlite3`; tool arguments cannot override the path.

Receipts are random opaque IDs persisted in private SQLite, stable for identical
exact-scope/turn/tool/source/content-hash snapshots across executor and process
restarts. Metadata labels retention `durable_immutable_snapshot`. Stored strings
are immutable and checksummed, independent of later source Store changes. Repeated
lookup does not rerun source operations. Item/byte capacity is global to this
database and refuses new receipts rather than silently expiring existing ones.
There is no automatic expiry or global lookup path. Treat recalled history as
untrusted. Hashes detect accidental corruption, not malicious edits by its owner.

## Optional summary backend and accounting

Attach with
`with_summary_backend(SummaryBackend::new(Arc<dyn Provider>, SummaryPolicy)?)?`.
Both config types use strict serde field validation. Policy defaults have zero
calls/cost and are unusable until explicitly configured. The trusted application
must construct a **dedicated** provider whose configured output limit is at most
`policy.max_output_tokens`, and known provider input/output rates equal the policy
rates. `Provider::complete` cannot pass a per-request output cap, so the crate
documents this operator contract and checks returned usage; it cannot prevent a
malicious/misconfigured provider from billing beyond its declared configuration.

SummaryPolicy fields: `ledger_path`, `max_calls`, `max_cost_microusd`,
`max_input_bytes`, `max_output_tokens`, `input_microusd_per_million`,
`output_microusd_per_million`, `timeout_seconds`. Positive known prices are
mandatory. `ledger_path` must be absolute, the direct parent an existing owned
0700 directory, and the file owned, regular, private, single-linked and not a
symlink. The application should place it in its private configured state dir.

Before dispatch, a durable locked append-only ledger reserves worst-case input
byte-derived tokens plus protocol overhead and configured output-token costs.
It records parent turn/call lineage, exact-scope hash and provider fingerprint.
Costs and call counts are lifetime totals for that operator ledger. Concurrent
executors share its file lock. Same scope/turn/call replay is refused. Successful
usage and calculated operator-price costs are recorded separately. Unknown
provider cost remains explicitly unknown, never represented as free. Failure,
timeout, cancellation and invalid responses retain their full reservation; no
automatic retries or unaccounted continuation are performed. Reservations are
not released, intentionally conservative even after successes. The ledger is
not protected against malicious edits by its owning operator.

`context_shunt` has `side_effect: true`: application policy must require durable
approval before dispatch. Read tools have `side_effect: false`. The crate has no
authority to grant outer application approvals itself.

## Verification

`cargo test -p missy-history --offline` uses only disposable Store fixtures and
mock Providers. It neither reads production history nor contacts external APIs.
