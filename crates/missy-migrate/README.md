# missy-migrate: explicit OFFLINE migration

Native Rust, copy-only migration from **operator-supplied offline files**. This
crate does not locate a home directory, read environment credentials, resolve a
vault reference, call Python, contact a provider, execute tools, replay a task,
restart a service or activate the result.

## API

`plan(MigrationInput) -> Result<MigrationPlan>` accepts optional `config_path`,
`database_path`, `mapping_path` and bounded `limits`. At least a config or database
must be supplied. Every path must be absolute. Planning only reads source files;
it does not accept or create a destination. The serializable/Debug plan reports
`offline`, `converted_config`, `issues`, `counts`, `sources`. Source-row content
and raw credentials are excluded. `is_applicable()` means no blocking error;
review issues still require operator review. `fingerprint()` binds the complete
captured plan, private row data, sources and bounds.

`apply(&plan, destination, confirm) -> Result<MigrationManifest>` requires
`confirm == true`, a still-matching plan and source hashes, and an **absent**
destination. It never merges into or replaces an existing Rust installation.
The caller should display the complete plan before requesting confirmation.
An apply retry against an existing destination is refused, rather than importing
duplicates. Native memory IDs are deterministic hashes of backup/table/row ID.

Output is a private, separate directory containing:

* `config.yaml`, if a config was supplied, with `state_dir: state`.
* `state/runtime.sqlite3`, the native Store database.
* `quarantine.json`, inert global/session metadata and checkpoints.
* `plan.json`, the credential-redacted report.
* `manifest.json`, source SHA-256 hashes, artifact SHA-256 hashes, counts and
  limitations. The manifest does not hash itself.

## Prepare source backups first

Do not `cp` a live SQLite database. Make a consistent snapshot using SQLite's
online backup API or the SQLite CLI `.backup` operation into a separate private
directory. The importer **refuses any `-wal`, `-shm` or `-journal` companion**, even
an empty file, and opens the checked snapshot read-only with `immutable=1`.
All `.missy` path components are refused, including explicit ones. Copy/export
the operator-reviewed config and database to a separate backup directory first.
No real home config or live database is inferred or inspected.

Source files must be regular single-link files owned by the current operator or
root, with private permissions (normally 0600 or 0400), and no execute bits.
Ancestors must be root/operator-owned and not group/world-writable; root-owned
sticky temporary directories are permitted. Every source path component is opened
descriptor-relative with `O_NOFOLLOW`; symlinks are refused. The importer never
chmods/chowns sources. Same-UID adversaries able to rename opened ancestors or
change mounts remain outside this userspace boundary, as with native policy.

Default hard bounds: 256 MiB per source DB, 1 MiB config/mapping, 50,000 records,
256 KiB per serialized row, 64 MiB total imported content. Callers may lower but
not raise these limits. UTF-8, ordinary legacy tables and expected column types
are verified; unknown data tables/columns fail closed. Derived FTS state is not
imported. Optional historical cost/checkpoint columns are accepted.

## Configuration conversion is a plan, not blind compatibility

Supported providers: enabled OpenAI, Anthropic and Ollama, model/base URL/timeout,
and default-provider order. Credentials become **environment names only**.
`api_key_env: MY_KEY` or an exact `${MY_KEY}` reference is accepted. Missing keys
use the conventional provider environment name without reading it. Raw API keys,
vault references, multi-key rotation, unsupported provider fields and OAuth/CLI
providers produce blocking errors; values are never printed or copied. Disabled
providers remain disabled by omission, with a review issue.

Absolute read/write roots convert without home expansion. Protected-path
exceptions cannot silently disappear. Legacy CIDRs, domain wildcards, bare hosts,
per-category lists, network presets and REST rules cannot be safely inferred as
native origins: nonempty grants are blocking errors. The operator can edit only
the offline copy to supply `network.allowed_origins` (exact HTTP(S) origins) and
explicit `network.allow_private_network`. Provider endpoints do **not**
automatically receive network grants. No private-network access is inferred.

Legacy unrestricted shell and executable-name allowlists cannot convert to exact
argv. Remove them in the offline copy and supply `shell.exact_argv`, an array of
complete argv arrays with absolute executables. `shell.enabled` remains off unless
explicitly enabled in that source. Exact argv remains userspace approval, not an
OS sandbox. Python plugins are unsupported and never become native MCP/tool grants.

Unknown keys, including optional legacy subsystems, are blocking issues rather
than silently discarded. Unknown key names are hash-labeled in reports to avoid
leaking credentials placed in key names. Config version stamps and disabled
subsystems produce explicit review notes. Root Python personas/system prompts
are not carried over. Native capability defaults remain off.

## Scope and history: deliberate limitations

The mapping file is JSON/YAML with **every legacy session ID as a key and a full
native `Scope` as its value**, for example:

```json
{"legacy-session-1":{"tenant":"local","user":"alice","channel":"cli","session":"import-1"}}
```

All four scope fields must validate. Many-to-one mappings are refused to prevent
accidental history merges. Missing mappings block application; there is no global
scope fallback, inferred user/channel or default session. Unscoped rows are
quarantined, never granted runtime scope.

**This imports historical data, not resumable sessions or native conversation
turns.** Raw turns, summaries and stored large tool output become one **untrusted
scoped memory** per source row with explicit source-table, row identity hash and
backup-hash provenance. Every original role, metadata object and tool-call payload
is retained as JSON data inside an `inert_imported_legacy_data` wrapper, never
promoted into a system/assistant/tool protocol message. Native `history()` remains
empty immediately after import. Retrieval requires the mapped scope. Historical
timestamps remain source data, not fabricated native activity timestamps.

Legacy learnings have no reliable session field, so **all global learnings are
quarantined**, not assigned to an arbitrary operator. Session names and cost
metadata also remain quarantined rather than modifying native billing/accounting.
Every checkpoint, including incomplete/running checkpoints, is quarantined outside
the runtime database, never admitted, resumed or replayed. Separate Python
`checkpoints.db`, scheduler/jobs, vault, identities, plugins and other state files
are not discovered or imported by this memory-database API. Do not interpret a
successful import as parity for these subsystems.

Private quarantine and native memory necessarily preserve source data, which may
contain sensitive historical content. They are not public reports or instruction
sources. Review them under the same access controls as the original backup.

## Atomicity and verification

Apply writes only into a private sibling `.missy-import-*` staging directory,
imports through native scoped `Store::memory_put`, verifies content/trust/audit
chains and absence of runnable turns, checkpoints SQLite WAL, checks database
integrity, writes/hashes/fsyncs artifacts and rechecks sources before publication.
Linux `renameat2(RENAME_NOREPLACE)` publishes the whole destination atomically.
Ordinary validation errors remove the stage. A process crash may leave an inert
private staging directory, but never activates partial state. Such remnants are
not resumed or merged; review and remove them manually. Existing unrelated
staging directories are ignored. No automatic startup or rollout is performed.

Tests use only temporary private fixtures matching the Python schema. They check
scope canaries, quarantine, source hash stability, raw credential redaction,
unknown/grant errors, symlink/unsafe permission rejection, bounds, existing-target
refusal, changed-source/plan refusal and inert partial-stage behavior.
