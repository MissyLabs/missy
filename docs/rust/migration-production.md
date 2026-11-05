# Offline production-state migration

`missy-migrate` is copy-only preservation and explicitly scoped history import,
not authorization to activate a replacement gateway. It never discovers a home
directory or reads `~/.missy`. Tests use source-derived schemas and private
temporary fixture exports. No deployment, accounts or credentials are needed.

## Explicit input and consistent export

`MigrationInput.backup_root` accepts an absolute exported directory. Existing
`config_path`, `database_path` and `mapping_path` remain supported. A root-only
import gets a secure default-deny config. Root `config.yaml` and `config.d/**`
are metadata-only references, NOT automatically parsed or copied, because legacy
configuration can contain plaintext credentials. To convert configuration supply
a separately reviewed sanitized `config_path`. Merge `config.d` precedence
offline first; migration never guesses include order or interpolates secrets.

Stop all legacy writers before exporting, including gateway, CLI and extensions.
Use SQLite's backup API for DB snapshots, not file copies of an active WAL DB.
The library neither stops processes nor creates live snapshots. Refusing paths
with `.missy` components prevents accidental live reads but cannot prove that a
differently named path is offline. Cross-file consistency requires a coordinated
operator export. External configured paths are not discovered; export workspace,
custom stores, audit/retrieval paths and extension/integration state deliberately.
Inventory completeness is relative to the supplied root, not the entire machine.

Files must be private regular single-link operator/root-owned non-executable
data. Ancestors are descriptor-relative ownership/writability checked; private
backup directories, O_NOFOLLOW and no writable ancestors are required (root-owned
sticky `/tmp` is allowed). Symlinks, sockets, FIFOs and devices are refused, not
silently dropped. Record their role separately during export. Export executable
scripts as non-executable data. Sources are never chmodded or modified.

Read stability checks include size, mtime and ctime. SQLite uses pinned
`mode=ro&immutable=1`, query-only, trusted-schema OFF, no WAL/SHM/journal companions
and integrity checks. No SQL extensions/source hooks are loaded. Graph foreign
keys are checked. Recognized DB schemas are strict. Unknown tables block rather
than disappear. Unknown operational DBs are validated and archived whole.

## Full disposition inventory

Every exported regular file, including unknown files, is inventoried; directory
listings including empty directories are fingerprinted. Content-addressed private
`archive/<sha256>` files are inert, never discovery roots, served media or native
attachments. A public plan contains paths/counts/digests/instructions, never row
bodies. Private archives and quarantine may contain personal data or embedded
credentials outside known credential paths: treat them as sensitive offline
backups, never publish them.

| Legacy state | Preserved result and required resolution |
|---|---|
| `config.yaml`, `config.d/**` | Metadata references only by default. Explicit sanitized conversion preserves supported provider priority, environment-only credentials and bounded limits. Unsupported grant-only settings produce explicit reviewed default-deny templates; dropping active constraints, mixed incompatible policy schemas and unknown fields block at hashed key paths. Required legacy isolation clears local grants instead of silently weakening them. See [config-conversion.md](config-conversion.md). |
| `memory.db` turns/summaries/large_content | Validate actual schema, require explicit one-to-one session to full tenant/user/channel/session mapping, import as inert untrusted JSON memories with per-source DB SHA provenance. System/tool roles never become native instructions/protocol turns. Root import also archives original DB. |
| Learnings, sessions, costs | Quarantine exact rows outside retrieval. Global learning ownership is not invented; historical costs/session metadata are not native budget/effect receipts. |
| Entities/relationships in memory DB or `graph_memory.db` | Validate actual global graph schema, quarantine exact rows, retain raw DB. Explicit authenticated ownership/scopes required before rebuilding native retrieval. |
| `checkpoints.db` or checkpoint table in memory DB | Validate current and older optional lease columns, quarantine every row. Never resume/replay loop messages. Reconcile external effects and tool idempotency receipts, then explicitly abandon or recreate a reconciled task. A prompt/history is not an effect receipt. |
| `jobs.json` | Preserve entire actual array plus every field in durable disabled review records. No registration, overdue execution or Nomad dispatch. Recreate PAUSED with authenticated Scope, cron/timezone, active hours, provider, capability/retry/backoff, one-shot and overlap semantics. Reconcile last/next run, queued/running evaluations, costs and effects before enabling. |
| `persona.yaml`, persona experiments, hatching files/log | Preserve original bytes; main persona gets disabled review record including boundaries, rules and traits. Compose reviewed native system prompt with lineage. No automatic promotion of experimental variants or historical feedback. |
| `memory.json`, `personal_memories.json`, `memories.json` | Preserve raw JSON and disabled review record. Actual legacy source has no authenticated tenant+user personal-memory DB schema. Map/review ownership before inserting native personal memories. Never fan out global records across users. |
| `skills/**`, custom tools/plugins, `mcp.json`, provider overrides | Archive only. No Python imports/execution, plugin activation or MCP starts. Port/review exact argv, capabilities, network permissions and digest pins. Proposals remain unapproved. |
| `trust.json`, `devices.json` | Exact review records/bytes. Trust scores are advisory history, not grants. Rebuild authenticated principal mappings and re-pair devices. |
| Patches, playbook, evolution/reaction feedback, `tool_candidates.db` | Preserve proposals and ledgers; inspect revision, hash, approval and apply/rollback evidence. Never auto-apply. |
| `request_tracker.db`, `global_budget.json`, `nomad-state/**` | Preserve operational bytes. Reconcile admitted requests, deduplication keys, reservation/spend and external workloads before replacement admits work. Unknown outcome is not success. |
| Audit/log files and rotations | Preserve originals independently. Verify original signature/chains offline with original identity, never splice into native audit chain. Unknown rotation names take explicit unknown-state review, not omission. |
| Retrieval/FAISS/semantic sidecars | Preserve bytes/source documents/citations without pickle/model execution. Rebuild derived indexes only from reviewed scoped sources. Unknown sidecars remain preserved. |
| Captures, videos, quarantine, backups | Preserve privately, unserved. Review owner, consent, retention and original quarantine reason. Nested backups may contain secrets outside recognized credential paths. |
| Run state and CLI sandboxes | Archive diagnostics only; never restore PIDs, sockets, leases, status or processes. Export non-regular runtime objects separately as diagnostic descriptions. |
| `secrets/**`, `identity.pem`, browser sessions, `edge.json` | Metadata references only: no content read, decrypted, hashed or copied. Includes encrypted vault and its key, OAuth refresh tokens, console/TLS/pairing keys and cookies. Keep original encrypted backup independently; reauthenticate, rotate, re-pair and initialize native vault. |
| Any unknown file | Retain exact bytes and require subsystem/secret/scope/effect review. Explicitly port or retire. No silent successful semantic-migration claim. |

Malformed structured JSON/YAML remains exact archived bytes with a repair/replan
instruction, never silently empty state. `operator-resolution.json` provides code,
category, `required_before_activation` and concrete instructions. Preservation
publication is allowed with unresolved reviews so data can be safely retained;
that does NOT mean semantic conversion or production readiness. There is no
blanket override or automatic-enable step. Known credential-reference fingerprints
hash metadata, not secret content; they are not low-entropy password hashes and
do not prove that an adversarial owner has not changed unread contents.

## Transaction, verification and rollback

### Explicit native portability mappings

The mapping YAML supports reserved `state:` keys alongside ordinary legacy
session IDs. Ordinary session mappings remain one-to-one; state mappings may
intentionally share a destination. Every value is a full validated Scope:

```yaml
legacy-session: {tenant: t, user: alice, channel: discord, session: imported}
state:job:backup-id: {tenant: t, user: alice, channel: discord, session: imported}
state:persona: {tenant: t, user: alice, channel: discord, session: imported}
state:graph:graph_memory.db: {tenant: t, user: alice, channel: discord, session: imported}
state:trust:tool:shell: {tenant: t, user: alice, channel: discord, session: imported}
```

Explicit mappings create real native assets in addition to preserved originals:

- `state/automation.sqlite3`: native scheduler schema/version, mapped jobs
  inserted disabled. Native ScheduleSpec validates schedule and IANA timezone;
  RFC3339 persisted next_run is retained. Interval anchor is next_run, not current
  time; one-shot retains persisted due date. Original provider/retry/active-hour/
  Nomad/capability settings remain quarantined and receipts require equivalent
  runtime policy and effect reconciliation before resume. Unsupported/missing
  schedules/timezones remain unported with concrete receipts. Opening Scheduler
  creates its other empty tables without dispatching paused rows. Runtime config
  is not automatically changed.
- `state/governance.sqlite3`: actual pending native Persona proposal retaining
  every source field/boundary, with plan-fingerprint context. No active persona
  revision exists. Review hash/context, then approve through native governance at
  a new-turn boundary.
- `state/retrieval.sqlite3`: actual nodes/edges through Retrieval APIs, exact
  scopes/endpoints/native types validated, stable IDs bound to source DB hash,
  row-source citation provenance. Properties/weights/timestamps remain in raw
  originals because native graph has no equivalent fields. Invalid type/capacity/
  endpoints remain unported with receipts. Mapping a global graph is explicit
  operator authority, never user-name inference.
- Mapped trust normalizes `(score-500)/500`. Negative evidence goes through native
  advisory API with Unknown outcome; positive scores remain reviewable advice
  with neutral native score. No fabricated successful effect elevates trust.
  Multiple negative entities mapped to one native scope fail rather than overwrite.
  Neither trust nor persona proposals create approvals or tool grants.

`portable-state.json` lists actual converted assets, proposal IDs/hashes/scopes,
unported records and review/enable requirements. This private artifact and all
DBs are checksummed. Without reserved mappings, behavior is preservation-only.
The matrix above describes default handling; mappings are explicit opt-in
semantic portability. They never make checkpoints resumable. DB integrity is
checked before publication and by verify; runtime config remains operator-owned.

Plans seal captured data, mappings, inventory and limits. Apply requires explicit
confirmation and the unchanged sealed plan. Re-read full source inventory before
staging and immediately before publication: additions/removals/content changes
invalidate approval. SQLite sources are also rehashed after inspection.

All writes target a new private sibling staging tree. Verify expected imported
rows, trust, audit chain and zero resumable turns; checkpoint WAL and check DB
integrity. Fsync artifacts/directories; format-2 manifest records SHA-256 and byte
length; publish atomically with RENAME_NOREPLACE. Never overwrite/merge existing
destinations. Failures clean staging; a crash may leave an inert private
`.missy-import-*` sibling for manual inspection/deletion, never automatic resume.
A post-rename fsync failure can leave a published destination despite an error;
inspect and verify before retrying.

`verify(destination)` checks format/required artifacts, every digest and private
regular-file safety, rejects unexpected files/directories/journals, and checks
standalone runtime DB integrity. It does not read original credential references
or require the original export. Run only on stopped/offline output, not a live
mutating gateway. Manifest is not signed: keep its trusted hash/copy independently.
Checksums detect corruption, not malicious replacement of both manifest/data.

`rollback(destination, fresh_sibling, confirm)` verifies the unchanged offline
import then atomically renames it without replacement to a fresh sibling. It
never deletes data, edits legacy sources, stops services, changes accounts or
switches processes. This is NOT rollback of an activated gateway with new side
effects. Preserve/reconcile new effects before any independent handover back.

Hard bounds: 256 MiB/file, 50,000 records, 256 KiB/record, 64 MiB aggregate archive
bytes, nesting 16. Limits may only be tightened. Exceeding bounds fails closed,
not partial export. Raw archive and decoded-row bytes have separate bounds so
peak memory can exceed either. Large deployments require reviewed batching or a
future streaming migration design, not silently unbounded allocations.

## Fixture validation

`tests/migration.rs` covers actual memory schemas, inert scoped import, source
immutability, no-overwrite, mappings, credential redaction, unsafe grants, unknown
tables, WAL/SHM, bounds and plan/source tampering. `tests/production.rs` adds actual
graph/checkpoint/job/persona/trust fixtures, separate DBs, disabled durable review
records, vault/key/OAuth references, directory additions, malformed state retention,
symlink/permission/integrity failures, verification and non-destructive rollback.
These are repository fixture tests, not evidence of a live machine migration,
complete semantic operational-state port, authentication or deployment.
