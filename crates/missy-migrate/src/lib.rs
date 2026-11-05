//! Explicit OFFLINE, copy-only migration. No home discovery, providers, tools or replay.
//! Source files must be private operator-owned (or root-owned) backups, with no
//! SQLite WAL/SHM companions. Use SQLite's backup API first, not `cp` on a live DB.
#![cfg(target_os = "linux")]

use anyhow::{Result, anyhow, ensure};
use missy_core::{Config, Scope};
use missy_store::Store;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fmt,
    fs::{self, File, OpenOptions},
    io::Write,
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};

mod config;
mod inventory;
mod portable;
mod source;
mod sqlite;

pub use inventory::{InventoryEntry, Resolution, StateDisposition};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MigrationLimits {
    pub max_file_bytes: usize,
    pub max_records: usize,
    pub max_record_bytes: usize,
    pub max_total_content_bytes: usize,
}
impl Default for MigrationLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: 256 * 1024 * 1024,
            max_records: 50_000,
            max_record_bytes: 256 * 1024,
            max_total_content_bytes: 64 * 1024 * 1024,
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct MigrationInput {
    /// Explicit exported backup directory, never a live .missy tree.
    pub backup_root: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
    pub database_path: Option<PathBuf>,
    pub mapping_path: Option<PathBuf>,
    pub limits: MigrationLimits,
}
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IssueSeverity {
    Error,
    Review,
}
#[derive(Clone, Debug, Serialize)]
pub struct MigrationIssue {
    pub severity: IssueSeverity,
    pub code: String,
    pub location: String,
    pub message: String,
}
#[derive(Clone, Debug, Default, Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct MigrationCounts {
    pub imported_memories: usize,
    pub quarantined_records: usize,
    pub sessions: usize,
}
#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct SourceDigest {
    pub kind: String,
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}
#[derive(Clone, Serialize)]
struct ImportedRow {
    table: String,
    source_id: String,
    scope: Option<Scope>,
    row: Value,
}
/// Serializable public report never includes source row content or raw credentials.
/// Private captured data cannot be modified by deserializing a forged plan.
#[derive(Clone, Serialize)]
pub struct MigrationPlan {
    pub offline: bool,
    pub converted_config: Option<Config>,
    /// Inert metadata-only templates. Never read by the native runtime.
    pub config_templates: Vec<Value>,
    pub issues: Vec<MigrationIssue>,
    pub counts: MigrationCounts,
    pub sources: Vec<SourceDigest>,
    pub inventory: Vec<InventoryEntry>,
    pub resolutions: Vec<Resolution>,
    #[serde(skip)]
    input: MigrationInput,
    #[serde(skip)]
    rows: Vec<ImportedRow>,
    #[serde(skip)]
    archives: BTreeMap<String, Vec<u8>>,
    #[serde(skip)]
    mappings: BTreeMap<String, Scope>,
    #[serde(skip)]
    sealed_fingerprint: String,
}
impl fmt::Debug for MigrationPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match serde_json::to_string(self) {
            Ok(v) => f.write_str(&v),
            Err(_) => f.write_str("MigrationPlan(redacted)"),
        }
    }
}
impl MigrationPlan {
    pub fn is_applicable(&self) -> bool {
        self.offline
            && !self
                .issues
                .iter()
                .any(|i| i.severity == IssueSeverity::Error)
    }
    pub fn fingerprint(&self) -> String {
        // Include captured data, mappings and bounds without publishing them.
        let data = serde_json::to_vec(&(
            self,
            &self.rows,
            &self.archives,
            &self.mappings,
            &self.input.limits,
        ))
        .expect("serializable migration plan");
        digest(&data)
    }
    fn issue(&mut self, severity: IssueSeverity, code: &str, location: &str, message: &str) {
        self.issues.push(MigrationIssue {
            severity,
            code: code.into(),
            location: location.into(),
            message: message.into(),
        });
    }
    fn error(&mut self, code: &str, location: &str, message: &str) {
        self.issue(IssueSeverity::Error, code, location, message);
    }
    fn review(&mut self, code: &str, location: &str, message: &str) {
        self.issue(IssueSeverity::Review, code, location, message);
    }
}
pub(crate) fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Read explicit files only. Planning never creates a destination or modifies sources.
pub fn plan(input: MigrationInput) -> Result<MigrationPlan> {
    let hard = MigrationLimits::default();
    ensure!(
        input.backup_root.is_some() || input.config_path.is_some() || input.database_path.is_some(),
        "explicit offline config and/or backup database required"
    );
    ensure!(
        input.limits.max_file_bytes > 0
            && input.limits.max_file_bytes <= hard.max_file_bytes
            && input.limits.max_records > 0
            && input.limits.max_records <= hard.max_records
            && input.limits.max_record_bytes > 0
            && input.limits.max_record_bytes <= hard.max_record_bytes
            && input.limits.max_total_content_bytes > 0
            && input.limits.max_total_content_bytes <= hard.max_total_content_bytes,
        "migration limits must be positive and cannot exceed hard bounds"
    );
    let mut p = MigrationPlan {
        offline: true,
        converted_config: None,
        config_templates: vec![],
        issues: vec![],
        counts: MigrationCounts::default(),
        sources: vec![],
        inventory: vec![],
        resolutions: vec![],
        input: input.clone(),
        rows: vec![],
        archives: BTreeMap::new(),
        mappings: BTreeMap::new(),
        sealed_fingerprint: String::new(),
    };
    if let Some(path) = &input.config_path {
        let (bytes, source) =
            source::read(path, "config", input.limits.max_file_bytes.min(1024 * 1024))?;
        p.sources.push(source);
        config::convert(&bytes, &mut p)?;
    }
    let mut mappings = BTreeMap::<String, Scope>::new();
    if let Some(path) = &input.mapping_path {
        let (bytes, source) = source::read(path, "scope_mapping", 1024 * 1024)?;
        p.sources.push(source);
        mappings = serde_yaml_ng::from_slice(&bytes).map_err(|_| {
            anyhow!("invalid scope mapping; expected legacy session ID -> full Scope")
        })?;
        for scope in mappings.values() {
            scope.validate()?;
        }
        // Accidental many-to-one mappings merge user histories, so reject them.
        let mut targets = std::collections::HashSet::new();
        ensure!(
            mappings
                .iter()
                .filter(|(name, _)| !name.starts_with("state:"))
                .all(|(_, s)| targets.insert(serde_json::to_string(s).unwrap())),
            "scope mapping must be one-to-one; refusing merged legacy sessions"
        );
    }
    if let Some(path) = &input.database_path {
        sqlite::read_backup(path, &mappings, &mut p)?;
    }
    if let Some(root) = &input.backup_root {
        inventory::capture(root, &mappings, &mut p)?;
    }
    p.mappings = mappings;
    p.sealed_fingerprint = p.fingerprint();
    Ok(p)
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct ArtifactDigest {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}
#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct MigrationManifest {
    pub format_version: u32,
    pub offline: bool,
    pub plan_sha256: String,
    pub sources: Vec<SourceDigest>,
    pub counts: MigrationCounts,
    pub inventory: Vec<InventoryEntry>,
    pub resolutions: Vec<Resolution>,
    pub artifacts: Vec<ArtifactDigest>,
    pub notes: Vec<String>,
}

/// Explicitly confirmed application into a NEW directory only. Failures remove
/// staging data; a crash can leave an inert sibling `.missy-import-*` directory,
/// never an activated destination. Existing destinations are always refused.
pub fn apply(
    plan: &MigrationPlan,
    destination: impl AsRef<Path>,
    confirm: bool,
) -> Result<MigrationManifest> {
    ensure!(
        confirm,
        "offline migration apply requires explicit confirmation"
    );
    ensure!(
        plan.is_applicable(),
        "migration plan has blocking errors; review the plan before applying"
    );
    ensure!(
        plan.fingerprint() == plan.sealed_fingerprint,
        "migration plan was modified after review"
    );
    let fresh = self::plan(plan.input.clone())?;
    ensure!(
        fresh.fingerprint() == plan.sealed_fingerprint,
        "source files changed after planning; create and review a new plan"
    );
    let destination = destination.as_ref();
    source::destination(destination)?;
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow!("destination parent required"))?;
    let stage = tempfile::Builder::new()
        .prefix(".missy-import-")
        .tempdir_in(parent)?;
    fs::set_permissions(stage.path(), fs::Permissions::from_mode(0o700))?;
    let store_path = stage.path().join("state/runtime.sqlite3");
    fs::create_dir(stage.path().join("state"))?;
    fs::set_permissions(
        stage.path().join("state"),
        fs::Permissions::from_mode(0o700),
    )?;
    let store = Store::open(&store_path)?;
    let mut quarantine = vec![];
    let mut scopes = BTreeMap::new();
    let mut expected = vec![];
    for row in &plan.rows {
        let database_hash = row
            .row
            .get("_migration_source_sha256")
            .and_then(Value::as_str)
            .unwrap_or("none");
        if let Some(scope) = &row.scope {
            // All original roles, tool calls, metadata and instructions remain JSON
            // data. No legacy role becomes a system/user/assistant protocol message.
            let content = serde_json::to_string(
                &json!({"kind":"inert_imported_legacy_data", "never_execute":true,
                "source_table":row.table,"source_row":row.row}),
            )?;
            let id = format!(
                "legacy-{}",
                digest(format!("{}:{}:{}", database_hash, row.table, row.source_id).as_bytes())
            );
            let provenance = format!(
                "offline_python_import:sha256:{database_hash}:table:{}:row_sha256:{}",
                row.table,
                digest(row.source_id.as_bytes())
            );
            store.memory_put(scope, &id, &content, &provenance, "untrusted")?;
            scopes.insert(serde_json::to_string(scope)?, scope.clone());
            expected.push((scope, id, content));
        } else {
            quarantine.push(json!({"reason":"unscoped_global_or_checkpoint_never_replay", "source_table":row.table,
                "source_row":row.row,"source_sha256":database_hash}));
        }
    }
    for (scope, id, content) in expected {
        let memory = store
            .memory_get(scope, &id)?
            .ok_or_else(|| anyhow!("import verification failed"))?;
        ensure!(
            memory.content == content && memory.trust == "untrusted",
            "import verification failed"
        );
    }
    for scope in scopes.values() {
        store.verify_audit(scope)?;
        ensure!(
            store.list_turns(scope, 1)?.is_empty(),
            "migration must not create resumable turns"
        );
    }
    drop(store);
    let check = rusqlite::Connection::open(&store_path)?;
    check.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
    let integrity: String = check.query_row("PRAGMA integrity_check", [], |r| r.get(0))?;
    ensure!(integrity == "ok", "destination database validation failed");
    drop(check);
    // WAL and SHM are disposable only after every store connection has closed.
    for suffix in ["-wal", "-shm"] {
        let path = PathBuf::from(format!("{}{suffix}", store_path.display()));
        if path.exists() {
            fs::remove_file(path)?;
        }
    }
    {
        // A state-only import is still a usable isolated native tree. With no
        // source config, publish secure default-deny config, never implicit grants.
        let mut cfg = plan.converted_config.clone().unwrap_or_default();
        cfg.state_dir = PathBuf::from("state");
        cfg.validate()?;
        private_write(
            &stage.path().join("config.yaml"),
            serde_yaml_ng::to_string(&cfg)?.as_bytes(),
        )?;
    }
    private_write(
        &stage.path().join("quarantine.json"),
        &serde_json::to_vec_pretty(&quarantine)?,
    )?;
    private_write(
        &stage.path().join("plan.json"),
        &serde_json::to_vec_pretty(plan)?,
    )?;
    private_write(
        &stage.path().join("operator-resolution.json"),
        &serde_json::to_vec_pretty(&plan.resolutions)?,
    )?;
    private_write(
        &stage.path().join("config-review.json"),
        &serde_json::to_vec_pretty(&plan.config_templates)?,
    )?;
    fs::create_dir(stage.path().join("archive"))?;
    fs::set_permissions(
        stage.path().join("archive"),
        fs::Permissions::from_mode(0o700),
    )?;
    for (hash, bytes) in &plan.archives {
        private_write(&stage.path().join("archive").join(hash), bytes)?;
    }
    let portable_artifacts = portable::write(plan, stage.path())?;
    let mut artifacts = vec![];
    let mut names = vec![
        "state/runtime.sqlite3".to_owned(),
        "config.yaml".into(),
        "quarantine.json".into(),
        "plan.json".into(),
        "operator-resolution.json".into(),
        "config-review.json".into(),
    ];
    names.extend(plan.archives.keys().map(|hash| format!("archive/{hash}")));
    names.extend(portable_artifacts);
    for name in names {
        let path = stage.path().join(&name);
        if path.exists() {
            let bytes = fs::read(&path)?;
            File::open(&path)?.sync_all()?;
            artifacts.push(ArtifactDigest {
                path: name,
                sha256: digest(&bytes),
                bytes: bytes.len() as u64,
            });
        }
    }
    let manifest = MigrationManifest { format_version: 2, offline: true, plan_sha256: plan.fingerprint(), sources: plan.sources.clone(),
        inventory: plan.inventory.clone(), resolutions: plan.resolutions.clone(),
        counts: plan.counts.clone(), artifacts,
        notes: vec!["No source writes, provider calls, tool calls, runnable turns, automatic activation or live-state discovery.".into(),
            "Raw turns/summaries/large content are untrusted scoped memories, not native conversation turns.".into(),
            "Global learnings, session/cost metadata and every checkpoint stay quarantined outside runtime retrieval.".into(),
            "Crash remnants .missy-import-* are inert; never resume them. Review then delete manually.".into()] };
    private_write(
        &stage.path().join("manifest.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    File::open(stage.path().join("state"))?.sync_all()?;
    File::open(stage.path().join("archive"))?.sync_all()?;
    File::open(stage.path())?.sync_all()?;
    // Check hashes once more immediately before publication.
    ensure!(
        self::plan(plan.input.clone())?.fingerprint() == plan.sealed_fingerprint,
        "source changed during migration; destination not published"
    );
    source::publish(stage.path(), destination)?;
    // rename succeeded; TempDir cleanup sees its former name absent.
    File::open(parent)?.sync_all()?;
    Ok(manifest)
}
/// Verify an OFFLINE published import, not an activated/mutating service tree.
/// Checksums detect corruption, not malicious replacement of both manifest/data.
pub fn verify(destination: impl AsRef<Path>) -> Result<MigrationManifest> {
    let destination = destination.as_ref();
    source::directory(destination)?;
    let (bytes, _) = source::read(
        &destination.join("manifest.json"),
        "manifest",
        16 * 1024 * 1024,
    )?;
    let manifest: MigrationManifest = serde_json::from_slice(&bytes)
        .map_err(|_| anyhow!("invalid migration manifest; details suppressed"))?;
    ensure!(
        manifest.offline && manifest.format_version == 2 && manifest.artifacts.len() <= 50_010,
        "unsupported migration manifest"
    );
    let mut paths = std::collections::BTreeSet::new();
    for artifact in &manifest.artifacts {
        let path = Path::new(&artifact.path);
        ensure!(
            !path.is_absolute()
                && path
                    .components()
                    .all(|c| matches!(c, std::path::Component::Normal(_)))
                && paths.insert(artifact.path.clone()),
            "unsafe or duplicate manifest artifact path"
        );
        let (_, actual) = source::read(
            &destination.join(path),
            "artifact",
            MigrationLimits::default().max_file_bytes,
        )?;
        ensure!(
            actual.bytes == artifact.bytes && actual.sha256 == artifact.sha256,
            "migration artifact checksum verification failed"
        );
    }
    for required in [
        "state/runtime.sqlite3",
        "config.yaml",
        "quarantine.json",
        "plan.json",
        "operator-resolution.json",
    ] {
        ensure!(
            paths.contains(required),
            "required migration artifact missing from manifest"
        );
    }
    paths.insert("manifest.json".into());
    verify_tree(destination, destination, &paths)?;
    for artifact in &manifest.artifacts {
        if artifact.path.starts_with("state/") && artifact.path.ends_with(".sqlite3") {
            sqlite::validate_backup(
                &destination.join(&artifact.path),
                MigrationLimits::default().max_file_bytes,
            )?;
        }
    }
    Ok(manifest)
}

fn verify_tree(
    root: &Path,
    dir: &Path,
    expected: &std::collections::BTreeSet<String>,
) -> Result<()> {
    source::directory(dir)?;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ty = entry.file_type()?;
        let relative = path
            .strip_prefix(root)?
            .to_str()
            .ok_or_else(|| anyhow!("invalid artifact filename"))?;
        if ty.is_dir() {
            ensure!(
                matches!(relative, "state" | "archive"),
                "unexpected artifact directory"
            );
            verify_tree(root, &path, expected)?;
        } else {
            ensure!(
                ty.is_file() && expected.contains(relative),
                "unexpected artifact, symlink or active runtime journal"
            );
        }
    }
    Ok(())
}

/// Offline non-destructive rollback: atomically retire the entire new import to
/// a fresh sibling path. Legacy source files are never touched or restored over.
pub fn rollback(
    destination: impl AsRef<Path>,
    retired: impl AsRef<Path>,
    confirm: bool,
) -> Result<()> {
    ensure!(
        confirm,
        "offline rollback requires explicit confirmation and stopped native runtime"
    );
    let destination = destination.as_ref();
    verify(destination)?;
    source::publish(destination, retired.as_ref())?;
    File::open(
        retired
            .as_ref()
            .parent()
            .ok_or_else(|| anyhow!("parent required"))?,
    )?
    .sync_all()?;
    Ok(())
}

fn private_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}
