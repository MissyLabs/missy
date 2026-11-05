# Native scoped operator workflows

The executable entry point is `missy-rs operator`. These are **local operator
commands**, not model tools, Discord commands, or an authentication boundary
against other local users. They never initialize a provider, dispatch a tool,
connect a gateway, or import Python Missy's state. Configuration uses the same
native validation and relative-state resolution as the rest of the executable.
The Rust conversion is repository work, not authorization to install/restart a
bot on someone else's machine.

Place all four scope options **after `operator` and before the nested command**.
The examples below use one exact scope:

```sh
missy-rs --config ./missy-rust.yaml operator \
  --tenant local --user operator --channel cli --session example \
  session show --limit 20
```

Every lookup and mutation uses the full tenant/user/channel/session tuple. There
is no implicit global fallback, session glob, bulk abandon, or manual SQL step.
The local flags are operator controls; an untrusted remote user must never be
allowed to choose these flags through a shell wrapper.

## Session administration

Append the following suffixes to the scoped command above:

| Suffix | Actual behavior |
|---|---|
| `session show --limit 20` | Exact-scope display name and recent durable turn dispositions; limit 1..1000. |
| `session display-name` | Read presentation metadata. |
| `session display-name 'Project notes'` | Set a validated display name. Does not rename the scope, rekey requests or change authorization. |
| `session clear --confirm` | Logical history cutoff. Retains turns, request bindings, tool receipts, checkpoints and audit evidence. |
| `session cleanup --before-unix-ms 1793491200000 --dry-run` | Preview eligible terminal turn/memory content count without deleting or adding an audit event. |
| `session cleanup --before-unix-ms 1793491200000 --confirm` | Purge old terminal content using the Store's transactional guarded retention API. Keep deduplication tombstones and the audit chain. |

The cutoff is an **absolute positive UTC Unix timestamp in milliseconds**,
not a duration or local date. `1793491200000` is 2026-11-01 00:00:00 UTC.
Clear and cleanup reject a live session lease, nonterminal work or unresolved
tool effects. Cleanup is explicit content destruction, not session deletion or
permission to rerun an old request. Retention preserves the original request's
payload binding even after the visible prompt has been removed. The audit chain
also retains its original event payloads; cleanup is **not** a secure erasure
claim, physical SQLite vacuum, or audit-redaction command.

## Audit inspection and signed export

```sh
missy-rs --config ./missy-rust.yaml operator --session example audit verify
missy-rs --config ./missy-rust.yaml operator --session example audit recent --limit 50
missy-rs --config ./missy-rust.yaml operator --session example audit recent --kind operator_tool_reconciliation
missy-rs --config ./missy-rust.yaml operator --session example audit security --limit 50
missy-rs --config ./missy-rust.yaml operator --session example audit export ./example-audit.json
missy-rs --config ./missy-rust.yaml operator --session example audit verify-export ./example-audit.json \
  --public-key 'INDEPENDENTLY_PINNED_BASE64_ED25519_PUBLIC_KEY'
```

The shorter commands above intentionally use the local/operator/cli defaults;
set all four scope flags for a different scope. Recent/security reads verify the
**complete captured scoped chain** before selecting a newest-first tail.
`--kind` is an exact stored event kind, not an invented legacy category.
`security` selects stored security/approval/denial/cancellation/uncertainty and
operator events. It is **not an installation scanner**, an amalgamation of
separate governance/plugin/evolution logs, or a claim every security event is
written into the runtime Store. The selection definition is explicit in
`operator.rs::security_event`.

Export captures and verifies one complete scoped snapshot, serializes a typed
canonical payload, and signs it with the native vault's Ed25519 identity. On
first use this explicitly creates the private state vault/identity. The output
is a new mode-0600 file; existing files and symlink outputs are never overwritten.
The signed payload limit is 4 MiB and the complete pretty JSON envelope limit is
8 MiB. Larger exports fail explicitly rather than silently truncating history.
An I/O failure can leave an incomplete new output, which must be investigated
and removed or renamed explicitly before retrying. The vault and database remain
separate; an export is not inserted back into its own audit snapshot.

Export prints the **public** verification key and payload SHA-256, never the
private signing key. Pin that public key through an independent trusted channel
or read-only trusted configuration. Accepting a key merely because it appears in
the supplied export proves no identity. `verify-export` requires `--public-key`,
checks the configured exact scope, signature, digest, format, increasing sequence
and every scope-bound chain hash. It is offline: it never opens/creates the local
database, vault or identity. JSON whitespace may change without invalidating a
signature because the typed payload is canonically reserialized for verification.

The live SQLite chain is tamper-evident, not independently signed per event.
A malicious database owner who can rewrite a full chain can produce another
self-consistent chain. Independently pinned signed exports anchor only the
snapshots actually exported, not all later events or completeness beyond the
signed snapshot. Private audit payloads can contain sensitive operational data;
public keys are public, exports generally are not.

## Unknown-effect and provider-billing reconciliation

`operator budget` presents the existing Store's database-global reserved/used
cost, token and uncertainty totals without contacting a provider. Its output
explicitly says `scope_kind: database_global` and `scope_filter_applied: false`;
it must not be mistaken for the selected session's cost report. There is no
automatic reset or release of unresolved reservations.

Inspect first. No inspection command contacts the external system or assumes
that an interrupted operation failed:

```sh
missy-rs --config ./missy-rust.yaml operator --session example effects inspect TURN_ID
```

The JSON includes the original prompt SHA-256, exact tool argument SHA-256,
current tool dispositions and scoped events. Argument hashes are over the
Store's serialized JSON representation, not file bytes or a rewritten request.
Do not place secrets in operator evidence files, results, or terminal transcripts.
These are retained durable records, not secret storage.

After independently checking external receipts/state, supply a bounded UTF-8
JSON result file and a nonempty UTF-8 evidence file (at most 16 KiB):

```sh
missy-rs --config ./missy-rust.yaml operator --session example effects reconcile-tool \
  TURN_ID CALL_ID --arguments-sha256 ORIGINAL_ARGUMENT_HASH \
  --result ./verified-result.json --evidence ./independent-investigation.txt \
  --succeeded --confirm
```

Use exactly one of `--succeeded` or `--failed`. A failed attestation is a final
disposition, **not authorization to retry the same call ID**. The Store checks
the exact scope, original argument hash, mutable disposition and absence of a
live lease transactionally. It records result/evidence hashes and evidence in
the audit chain. Repeated or stale reconciliation fails. The CLI does not verify
that the operator's external investigation is truthful.

To resolve unknown provider billing and **abandon**, not replay, the turn:

```sh
missy-rs --config ./missy-rust.yaml operator --session example effects reconcile-billing \
  TURN_ID --prompt-sha256 ORIGINAL_PROMPT_HASH \
  --usage ./verified-usage.json --evidence ./billing-receipt.txt --confirm
```

Usage JSON supplies `input_tokens`, `output_tokens`, `cost_microusd` and
`cost_known: true`. Values must come from independent provider/account evidence,
not from guessing that a disconnected completion cost nothing. Known cumulative
usage cannot decrease. The Store refuses unresolved tool effects, a live lease,
a wrong prompt hash or an already terminal turn, settles an existing global
reservation if present, and marks the turn failed. This command never asks the
provider again, fabricates a completion, resets a budget, deletes a receipt, or
resurrects the request. There is deliberately no bulk unknown-to-success switch.

Operator file input rejects symlinks, nonregular files and oversized data using
a single bounded no-follow descriptor. Reconciliation is local-only and requires
`--confirm`. `effects inspect` is read-only; normal inspection does not recover
or sweep leases. It may show private prompts/results to the local operator.

## Executable acceptance and remaining gaps

`crates/missy-cli/tests/operator_e2e.rs` drives the actual compiled `missy-rs`
subprocess with cleared environment, temporary HOME/config/state and fixtures
created through public Store APIs. It checks four-component isolation, logical
clear, confirmation gates, dry-run/retention tombstones, live/nonterminal refusal,
signed export/pin/scope/tamper/chain verification, database-independent offline
verification, bounded no-follow inputs, exact-hash unknown-tool reconciliation,
externally attested billing settlement, final dispositions and no replay. No
live provider, accounts, shell tools, bot process or network connection is needed.

Verified 2026-11-05: `CARGO_BUILD_JOBS=2 cargo test -p missy-cli --test
operator_e2e --locked -- --nocapture` passed **6/6 actual executable tests**.
The focused pass is evidence for these workflows only, not complete workspace
or live deployment acceptance. No installed bot was contacted or replaced.
Focused test-target lint also passed with `cargo clippy -p missy-cli --test
operator_e2e --no-deps --locked -- -D warnings`; this does not substitute for
workspace or executable-wide lint acceptance.

These workflows close substantive native administration gaps. They do **not**
claim complete legacy CLI parity. Broader cross-session inventory/age scanning,
installation security scan, legacy cost-report presentation, arbitrary API-token
management, OAuth account/device flows, and consolidation of separate subsystem
audit journals require their own real implementations and acceptance. Existing
provider/recovery/history/config commands should not be counted as multiple
new capabilities just by adding aliases.
