//! Complete explicit backup inventory. Nothing here is an activation authority.
use super::*;
use std::{collections::BTreeSet, os::fd::AsRawFd};

#[derive(Clone, Debug, Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StateDisposition {
    ScopedInertMemory,
    QuarantinedArchive,
    SecretReferenceOnly,
    DirectoryInventory,
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct InventoryEntry {
    pub relative_path: String,
    pub category: String,
    pub disposition: StateDisposition,
    /// Content SHA-256 for archives; metadata fingerprint ONLY for secrets.
    pub fingerprint: String,
    pub bytes: u64,
    pub archive: Option<String>,
    /// Records preserved in the durable, never-executed review queue.
    pub review_records: usize,
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct Resolution {
    pub code: String,
    pub category: String,
    pub required_before_activation: bool,
    pub instruction: String,
}
impl Resolution {
    pub(crate) fn new(code: &str, category: &str, instruction: &str, required: bool) -> Self {
        Self {
            code: code.into(),
            category: category.into(),
            required_before_activation: required,
            instruction: instruction.into(),
        }
    }
}

fn classify(path: &str) -> (&'static str, bool, &'static str) {
    let top = path.split('/').next().unwrap_or(path);
    match top {
        "secrets" | "identity.pem" | "browser_sessions" | "edge.json" => (
            "credentials_identity_pairing",
            true,
            "Do not copy legacy keys, cookies, tokens or pairing authority. Reauthenticate providers, re-pair devices, create a native vault and rotate keys. Keep the encrypted original backup offline; references are metadata-only, not portable secrets.",
        ),
        "config.yaml" | "config.d" => (
            "configuration",
            true,
            "Merge legacy config.d precedence offline, replace raw/vault secrets with environment references, and run explicit config conversion. Unsupported settings require equivalent native policy or explicit denial, never silently discard category-specific network rules, filesystem exclusions, approvals, budgets or channel allowlists.",
        ),
        "memory.db" => (
            "memory",
            false,
            "Review scope mapping and quarantine. Imported history is inert untrusted memory, not native protocol turns.",
        ),
        "checkpoints.db" => (
            "checkpoints",
            false,
            "Reconcile tool-side effects and idempotency receipts before abandoning or recreating unfinished tasks. Legacy checkpoints are never resumable in Rust.",
        ),
        "graph_memory.db" => (
            "graph",
            false,
            "Assign authenticated tenant/user/session scopes before reconstructing graph retrieval. Legacy graph has no authority-bearing scope.",
        ),
        "jobs.json" => (
            "jobs",
            false,
            "Recreate each job PAUSED with an authenticated Scope, timezone, schedule, provider, retry policy, active hours and capability grants. Reconcile last_run, run_count, next_run, Nomad queued/running evaluations and side effects before enabling. Archive retains every legacy field; no overdue occurrence is dispatched.",
        ),
        "persona.yaml" | "persona_experiment.json" | "hatching.yaml" | "hatching_log.jsonl" => (
            "persona",
            false,
            "Review identity, tone, traits, behavioral tendencies, response rules and hard boundaries. Compose a native system prompt preserving boundaries and record its lineage; do not promote experimental variants or historical feedback automatically.",
        ),
        "memory.json" | "personal_memories.json" | "memories.json" => (
            "unscoped_memory",
            false,
            "Legacy JSON/global memories lack authenticated tenant+user ownership. Explicitly map owners and review content before personal-memory import; global-to-user fanout is forbidden.",
        ),
        "skills" | "custom-tools" | "plugins" | "mcp.json" | "tool_provider_overrides.json" => (
            "executable_extensions",
            false,
            "Port Python/scripts to reviewed native tools or data-only skills. Recheck exact argv, network policy, capabilities and MCP digest pins. Archived code/manifests are never loaded or executed.",
        ),
        "trust.json" | "devices.json" => (
            "trust_devices",
            false,
            "Trust scores are advisory history, not authorization. Rebuild authenticated principal/device mappings and explicit grants; do not elevate legacy scores, device records or cookies into native approvals.",
        ),
        "patches.json"
        | "playbook.json"
        | "evolutions.json"
        | "discord_evolution_reactions.json"
        | "tool_candidates.db" => (
            "evolution",
            false,
            "Review proposal hashes, approvals, original source revisions and apply/rollback evidence. Never auto-apply archived patches, playbooks or tool proposals.",
        ),
        "request_tracker.db" | "global_budget.json" | "nomad-state" => (
            "operational_ledger",
            false,
            "Reconcile admitted requests, external workload status, spent/reserved budgets and deduplication keys. Resolve unknown outcomes before admitting replacement work; archive is not a native effect receipt.",
        ),
        "audit.jsonl" | "audit.log" | "missy.log" | "logs" => (
            "audit_logs",
            false,
            "Retain immutable historical logs and verify their original signature/hash chain with the original identity offline. Do not splice them into the new native audit chain.",
        ),
        "retrieval" | "memory.faiss" | "memory_semantic.faiss" | "memory.faiss.meta.json" => (
            "retrieval_indexes",
            false,
            "Preserve source documents, citations and ownership. Rebuild derived vector/FTS indexes only from reviewed scoped sources; do not deserialize pickle/FAISS metadata as executable objects.",
        ),
        "captures" | "videos" | "quarantine" | "backups" => (
            "retained_artifacts",
            false,
            "Review artifact ownership, consent, retention and original quarantine reasons. Archives remain private and unserved; never clear quarantine merely because migration succeeded.",
        ),
        "run" | "acpx_sandbox" | "claude_cli_sandbox" => (
            "ephemeral_runtime",
            false,
            "Archive diagnostics for reconciliation only. Never restore PIDs, sockets, locks, leases, gateway status, CLI sandboxes or active browser processes.",
        ),
        _ => (
            "unknown_state",
            false,
            "Unrecognized state is preserved, not silently dropped. Identify its owning subsystem, determine whether it includes secrets or outstanding effects, then document a scoped port or explicit retirement before activation.",
        ),
    }
}

pub(crate) fn capture(
    root: &Path,
    mappings: &BTreeMap<String, Scope>,
    p: &mut MigrationPlan,
) -> Result<()> {
    let mut seen = BTreeSet::new();
    let mut total = 0usize;
    walk(root, root, 0, mappings, p, &mut seen, &mut total)?;
    p.review("offline_inventory_requires_resolution", "backup_root",
        "Archive publication is not production readiness. Resolve every required operator-resolution item before enabling the replacement; secrets remain external references.");
    Ok(())
}

fn walk(
    root: &Path,
    dir: &Path,
    depth: usize,
    mappings: &BTreeMap<String, Scope>,
    p: &mut MigrationPlan,
    seen: &mut BTreeSet<String>,
    total: &mut usize,
) -> Result<()> {
    ensure!(depth <= 16, "backup directory nesting bound exceeded");
    let pinned = source::directory(dir)?;
    let mut entries = fs::read_dir(format!("/proc/self/fd/{}", pinned.as_raw_fd()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    ensure!(
        entries.len() <= p.input.limits.max_records,
        "backup directory entry bound exceeded"
    );
    let mut listing = vec![];
    for entry in entries {
        let ty = entry.file_type()?;
        ensure!(
            !ty.is_symlink() && (ty.is_dir() || ty.is_file()),
            "backup contains a symlink or non-regular object; export only private regular snapshots"
        );
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow!("non-UTF8 backup filename"))?;
        listing.push((name.clone(), ty.is_dir()));
        let path = dir.join(&name);
        let relative = path
            .strip_prefix(root)?
            .to_str()
            .ok_or_else(|| anyhow!("invalid backup path"))?
            .to_owned();
        ensure!(
            p.inventory.len() < p.input.limits.max_records,
            "backup inventory bound exceeded"
        );
        if ty.is_dir() {
            walk(root, &path, depth + 1, mappings, p, seen, total)?;
            continue;
        }
        let (category, secret, instruction) = classify(&relative);
        if seen.insert(category.into()) {
            p.resolutions.push(Resolution::new(
                "review_legacy_state",
                category,
                instruction,
                true,
            ));
        }
        // Known credentials are not read, decrypted, hashed by content or copied.
        if secret {
            let reference = source::reference(&path)?;
            p.inventory.push(InventoryEntry {
                relative_path: relative,
                category: category.into(),
                disposition: StateDisposition::SecretReferenceOnly,
                fingerprint: reference.sha256.clone(),
                bytes: reference.bytes,
                archive: None,
                review_records: 0,
            });
            p.sources.push(reference);
            continue;
        }
        let snapshot = source::pin(&path, "backup_inventory", p.input.limits.max_file_bytes)?;
        *total = total
            .checked_add(snapshot.bytes.len())
            .ok_or_else(|| anyhow!("inventory byte overflow"))?;
        ensure!(
            *total <= p.input.limits.max_total_content_bytes,
            "backup aggregate byte bound exceeded"
        );
        let hash = snapshot.source.sha256.clone();
        let sqlite = snapshot.bytes.starts_with(b"SQLite format 3\0");
        if sqlite {
            snapshot.no_journals()?;
        }
        let supported_db = matches!(
            relative.as_str(),
            "memory.db" | "checkpoints.db" | "graph_memory.db"
        );
        if supported_db {
            ensure!(
                sqlite,
                "recognized legacy database is not a standalone SQLite backup"
            );
            // An explicit --database and a root may designate the same source.
            if !p
                .sources
                .iter()
                .any(|s| s.kind == "database" && s.path == path)
            {
                sqlite::read_backup(&path, mappings, p)?;
            }
        } else if sqlite {
            sqlite::validate_backup(&path, p.input.limits.max_file_bytes)?;
        }
        let review_records = structured_records(&relative, category, &snapshot.bytes, &hash, p)?;
        p.inventory.push(InventoryEntry {
            relative_path: relative,
            category: category.into(),
            disposition: if supported_db && category == "memory" {
                StateDisposition::ScopedInertMemory
            } else {
                StateDisposition::QuarantinedArchive
            },
            fingerprint: hash.clone(),
            bytes: snapshot.source.bytes,
            archive: Some(format!("archive/{hash}")),
            review_records,
        });
        p.sources.push(snapshot.source);
        p.archives.insert(hash, snapshot.bytes);
    }
    let bytes = serde_json::to_vec(&listing)?;
    let relative = dir
        .strip_prefix(root)?
        .to_str()
        .ok_or_else(|| anyhow!("invalid backup directory"))?;
    p.inventory.push(InventoryEntry {
        relative_path: if relative.is_empty() {
            ".".into()
        } else {
            relative.into()
        },
        category: "directory".into(),
        disposition: StateDisposition::DirectoryInventory,
        fingerprint: digest(&bytes),
        bytes: 0,
        archive: None,
        review_records: 0,
    });
    Ok(())
}

/// Preserve ALL fields and explicit disabled state, not a lossy best-effort job
/// translation. This is a durable review queue, never a scheduler/approval DB.
fn structured_records(
    relative: &str,
    category: &str,
    bytes: &[u8],
    hash: &str,
    p: &mut MigrationPlan,
) -> Result<usize> {
    if !matches!(
        relative,
        "jobs.json"
            | "trust.json"
            | "devices.json"
            | "persona.yaml"
            | "playbook.json"
            | "personal_memories.json"
            | "memories.json"
            | "memory.json"
    ) {
        return Ok(0);
    }
    let parsed: Result<Value> = if relative.ends_with(".yaml") {
        serde_yaml_ng::from_slice(bytes).map_err(Into::into)
    } else {
        serde_json::from_slice(bytes).map_err(Into::into)
    };
    let value = match parsed {
        Ok(value) => value,
        Err(_) => {
            p.resolutions.push(Resolution::new("malformed_legacy_state", category,
                "Structured legacy data could not be parsed. Exact original bytes remain archived. Repair a separate copy and create a new plan; no parser source details are exposed.", true));
            return Ok(0);
        }
    };
    let records = if relative == "jobs.json" {
        match value {
            Value::Array(records) => records,
            other => {
                p.resolutions.push(Resolution::new("unknown_jobs_envelope", category,
                    "Expected actual legacy jobs.json top-level array. Unknown envelope preserved whole; inspect before porting jobs.", true));
                vec![other]
            }
        }
    } else {
        vec![value]
    };
    let count = records.len();
    ensure!(
        p.rows.len() + count <= p.input.limits.max_records,
        "structured review record bound exceeded"
    );
    for (index, record) in records.into_iter().enumerate() {
        let row = json!({"_migration_source_sha256":hash, "legacy_record":record,
            "migration_status":"requires_operator_resolution", "enabled":false,
            "never_execute":true, "authenticated_scope":null,
            "effect_receipts_verified":false});
        ensure!(
            serde_json::to_vec(&row)?.len() <= p.input.limits.max_record_bytes,
            "structured legacy review record exceeds byte bound"
        );
        p.rows.push(ImportedRow {
            table: format!("legacy_{category}"),
            source_id: index.to_string(),
            scope: None,
            row,
        });
        p.counts.quarantined_records += 1;
    }
    Ok(count)
}
