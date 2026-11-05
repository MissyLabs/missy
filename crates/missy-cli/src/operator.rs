//! Local operator workflows. No model, provider, gateway or legacy state access.
use anyhow::{Context, Result, ensure};
use clap::Subcommand;
use missy_core::{Config, Scope, Usage};
use missy_security::{SignedExport, Vault};
use missy_store::{AuditEntry, Store, hash_text};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs::OpenOptions,
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
};

#[derive(Subcommand)]
pub enum OperatorCommand {
    /// Read database-global reservations/known usage. Explicitly not a scoped total.
    Budget,
    /// Exact-scope presentation, logical clear and content retention.
    Session {
        #[command(subcommand)]
        command: SessionCommand,
    },
    /// Verify/read the scoped hash chain or sign an independently verifiable export.
    Audit {
        #[command(subcommand)]
        command: AuditCommand,
    },
    /// Inspect and attest externally verified unknown outcomes. Never replay effects.
    Effects {
        #[command(subcommand)]
        command: EffectsCommand,
    },
}

#[derive(Subcommand)]
pub enum SessionCommand {
    /// Show this exact scope's display name and recent turn dispositions.
    Show {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Read the presentation name, or set it without changing authorization scope.
    DisplayName { name: Option<String> },
    /// Hide prior chat history, retaining requests, turns, effects and audit evidence.
    Clear {
        #[arg(long)]
        confirm: bool,
    },
    /// Purge terminal content older than an absolute UTC Unix millisecond cutoff.
    Cleanup {
        #[arg(long)]
        before_unix_ms: i64,
        #[arg(long, conflicts_with = "confirm")]
        dry_run: bool,
        #[arg(long, conflicts_with = "dry_run")]
        confirm: bool,
    },
}

#[derive(Subcommand)]
pub enum AuditCommand {
    /// Verify this scope's complete chain, not merely a recent tail.
    Verify,
    Recent {
        #[arg(long, default_value_t = 50)]
        limit: usize,
        /// Exact stored event kind, not a guessed legacy category.
        #[arg(long)]
        kind: Option<String>,
    },
    /// View stored approval, cancellation, uncertainty and operator security events.
    Security {
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    /// Write a private, non-overwriting signed complete scoped audit snapshot.
    Export { output: PathBuf },
    /// Offline signature AND chain verification using a separately trusted public key.
    VerifyExport {
        input: PathBuf,
        #[arg(long)]
        public_key: String,
    },
}

#[derive(Subcommand)]
pub enum EffectsCommand {
    /// Inspect one exact turn, original hashes and tool dispositions without dispatch.
    Inspect { turn_id: String },
    /// Attest a tool's externally checked outcome against its ORIGINAL argument hash.
    ReconcileTool {
        turn_id: String,
        call_id: String,
        #[arg(long)]
        arguments_sha256: String,
        /// UTF-8 JSON result from the operator's independent external investigation.
        #[arg(long)]
        result: PathBuf,
        /// UTF-8 evidence; never passed to a model or executed.
        #[arg(long)]
        evidence: PathBuf,
        /// Exactly one of succeeded or failed must be supplied.
        #[arg(long, conflicts_with = "failed")]
        succeeded: bool,
        #[arg(long, conflicts_with = "succeeded")]
        failed: bool,
        #[arg(long)]
        confirm: bool,
    },
    /// Set externally verified billing and fail the turn. Never regenerate completion.
    ReconcileBilling {
        turn_id: String,
        #[arg(long)]
        prompt_sha256: String,
        /// JSON Usage: input_tokens, output_tokens, cost_microusd, cost_known=true.
        #[arg(long)]
        usage: PathBuf,
        #[arg(long)]
        evidence: PathBuf,
        #[arg(long)]
        confirm: bool,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuditPayload {
    format: String,
    scope: Scope,
    entries: Vec<AuditEntry>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuditExport {
    payload: AuditPayload,
    signature: SignedExport,
}

const EXPORT_FORMAT: &str = "missy-rust-scoped-audit-v1";
const EXPORT_LIMIT: usize = 8 * 1024 * 1024;

fn print(value: impl Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
fn limit(value: usize) -> Result<()> {
    ensure!((1..=1000).contains(&value), "limit must be 1..1000");
    Ok(())
}
fn confirm(value: bool) -> Result<()> {
    ensure!(value, "operator mutation requires --confirm");
    Ok(())
}
fn digest(value: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "expected a lowercase SHA-256 hex digest"
    );
    Ok(())
}

// One descriptor, bounded reads, no symlink/FIFO/device acceptance or open race.
fn read(path: &Path, max: usize) -> Result<Vec<u8>> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .with_context(|| format!("cannot open operator input {}", path.display()))?;
    let meta = file.metadata()?;
    ensure!(
        meta.is_file() && meta.len() <= max as u64,
        "operator input must be a bounded regular file"
    );
    let mut bytes = Vec::new();
    file.take(max as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= max, "operator input exceeds bound");
    Ok(bytes)
}
fn evidence(path: &Path) -> Result<String> {
    let evidence = String::from_utf8(read(path, 16384)?).context("evidence must be UTF-8")?;
    ensure!(
        !evidence.trim().is_empty(),
        "operator evidence must not be empty"
    );
    Ok(evidence)
}

// Same explicit scope-bound hash protocol as Store::verify_audit. Also used on
// a single captured snapshot, avoiding a verify-then-reload concurrent writer gap.
fn verify_entries(scope: &Scope, entries: &[AuditEntry]) -> Result<()> {
    scope.validate()?;
    let key = serde_json::to_string(scope)?;
    let mut previous = String::new();
    let mut sequence = 0;
    for entry in entries {
        ensure!(
            entry.sequence > sequence,
            "audit sequence is not increasing"
        );
        ensure!(
            entry.previous_hash == previous,
            "audit previous hash mismatch"
        );
        let payload = serde_json::to_string(&entry.payload)?;
        let computed = hash_text(&serde_json::to_string(&(
            &key,
            entry.created_at,
            &entry.kind,
            &payload,
            &previous,
        ))?);
        ensure!(computed == entry.hash, "audit hash mismatch");
        previous = entry.hash.clone();
        sequence = entry.sequence;
    }
    Ok(())
}

fn security_event(entry: &AuditEntry) -> bool {
    [
        "security",
        "approval",
        "denied",
        "blocked",
        "rejected",
        "quarantin",
        "injection",
        "redact",
        "cancellation",
        "outcome_unknown",
        "operator_",
    ]
    .iter()
    .any(|marker| entry.kind.contains(marker))
}

pub async fn execute(config: &Config, cmd: OperatorCommand, scope: Scope) -> Result<()> {
    scope.validate()?;
    // Offline verification must not create/open a database, vault or identity.
    if let OperatorCommand::Audit {
        command: AuditCommand::VerifyExport { input, public_key },
    } = &cmd
    {
        let export: AuditExport = serde_json::from_slice(&read(input, EXPORT_LIMIT)?)?;
        ensure!(
            export.payload.format == EXPORT_FORMAT,
            "unsupported audit export format"
        );
        ensure!(export.payload.scope == scope, "audit export scope mismatch");
        let canonical = serde_json::to_vec(&export.payload)?;
        ensure!(
            canonical.len() <= 4 * 1024 * 1024,
            "signed payload exceeds 4MiB"
        );
        export.signature.verify(&canonical, public_key)?;
        verify_entries(&scope, &export.payload.entries)?;
        return print(
            json!({"verified":true,"scope":scope,"entries":export.payload.entries.len(),"sha256":export.signature.sha256,"public_key":public_key}),
        );
    }

    let store = Store::open(config.state_dir.join("runtime.sqlite3"))?;
    match cmd {
        OperatorCommand::Budget => print(
            json!({"scope_kind":"database_global","scope_filter_applied":false,"totals":store.budget_totals()?,"note":"Reservations include unsettled/uncertain provider usage. No reset or release is performed."}),
        ),
        OperatorCommand::Session { command } => match command {
            SessionCommand::Show { limit: count } => {
                limit(count)?;
                print(
                    json!({"scope":scope,"display_name":store.session_display_name(&scope)?,"turns":store.list_turns(&scope,count)?}),
                )
            }
            SessionCommand::DisplayName { name } => {
                if let Some(name) = name {
                    store.set_session_display_name(&scope, &name)?;
                }
                print(json!({"scope":scope,"display_name":store.session_display_name(&scope)?}))
            }
            SessionCommand::Clear { confirm: yes } => {
                confirm(yes)?;
                let cutoff = store.session_clear(&scope)?;
                print(
                    json!({"scope":scope,"history_cutoff":cutoff,"logical_clear":true,"evidence_preserved":true}),
                )
            }
            SessionCommand::Cleanup {
                before_unix_ms,
                dry_run,
                confirm: yes,
            } => {
                ensure!(
                    before_unix_ms > 0,
                    "cutoff must be a positive Unix millisecond timestamp"
                );
                ensure!(dry_run || yes, "cleanup requires --dry-run or --confirm");
                let removed = store.session_cleanup(&scope, before_unix_ms, dry_run)?;
                print(
                    json!({"scope":scope,"before_unix_ms":before_unix_ms,"dry_run":dry_run,"content_records":removed,"tombstones_and_audit_preserved":true}),
                )
            }
        },
        OperatorCommand::Audit { command } => {
            let mut entries = store.audit_entries(&scope)?;
            verify_entries(&scope, &entries)?;
            match command {
                AuditCommand::Verify => print(
                    json!({"verified":true,"scope":scope,"entries":entries.len(),"head":entries.last().map(|e|&e.hash)}),
                ),
                AuditCommand::Recent { limit: count, kind } => {
                    limit(count)?;
                    if let Some(kind) = kind {
                        entries.retain(|e| e.kind == kind);
                    }
                    entries.reverse();
                    entries.truncate(count);
                    print(json!({"scope":scope,"newest_first":true,"entries":entries}))
                }
                AuditCommand::Security { limit: count } => {
                    limit(count)?;
                    entries.retain(security_event);
                    entries.reverse();
                    entries.truncate(count);
                    print(
                        json!({"scope":scope,"newest_first":true,"selection":"stored security and operator event kinds; not a host security scan","entries":entries}),
                    )
                }
                AuditCommand::Export { output } => {
                    let payload = AuditPayload {
                        format: EXPORT_FORMAT.into(),
                        scope: scope.clone(),
                        entries,
                    };
                    let canonical = serde_json::to_vec(&payload)?;
                    ensure!(
                        canonical.len() <= 4 * 1024 * 1024,
                        "signed payload exceeds 4MiB"
                    );
                    let signature =
                        Vault::open(config.state_dir.join("vault"))?.sign_export(&canonical)?;
                    let public_key = signature.public_key.clone();
                    let sha256 = signature.sha256.clone();
                    let encoded = serde_json::to_vec_pretty(&AuditExport { payload, signature })?;
                    ensure!(encoded.len() <= EXPORT_LIMIT, "encoded export exceeds 8MiB");
                    let mut file = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .mode(0o600)
                        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
                        .open(&output)
                        .context("export output must not already exist")?;
                    file.write_all(&encoded)?;
                    file.sync_all()?;
                    print(
                        json!({"scope":scope,"output":output,"sha256":sha256,"public_key":public_key,"note":"Pin this public key through an independent trusted channel before verifying later exports."}),
                    )
                }
                AuditCommand::VerifyExport { .. } => unreachable!("handled before opening store"),
            }
        }
        OperatorCommand::Effects { command } => match command {
            EffectsCommand::Inspect { turn_id } => {
                let turn = store.get_turn(&scope, &turn_id)?;
                let tools: Vec<_> = store
                    .tool_journal(&scope, &turn_id)?
                    .into_iter()
                    .map(|record| {
                        let arguments_sha256 = hash_text(
                            &serde_json::to_string(&record.call.arguments).expect("JSON value"),
                        );
                        json!({"record":record,"arguments_sha256":arguments_sha256})
                    })
                    .collect();
                print(
                    json!({"scope":scope,"prompt_sha256":hash_text(&turn.prompt),"turn":turn,"tools":tools,"events":store.events(&scope,&turn_id)?,"automatic_replay":false}),
                )
            }
            EffectsCommand::ReconcileTool {
                turn_id,
                call_id,
                arguments_sha256,
                result,
                evidence: path,
                succeeded,
                failed,
                confirm: yes,
            } => {
                confirm(yes)?;
                ensure!(
                    succeeded != failed,
                    "specify exactly one of --succeeded or --failed"
                );
                digest(&arguments_sha256)?;
                let result: Value = serde_json::from_slice(&read(&result, 4 * 1024 * 1024)?)?;
                store.reconcile_tool(
                    &scope,
                    &turn_id,
                    &call_id,
                    &arguments_sha256,
                    &result,
                    succeeded,
                    &evidence(&path)?,
                )?;
                print(
                    json!({"scope":scope,"turn_id":turn_id,"call_id":call_id,"reconciled":true,"succeeded":succeeded,"automatic_replay":false}),
                )
            }
            EffectsCommand::ReconcileBilling {
                turn_id,
                prompt_sha256,
                usage,
                evidence: path,
                confirm: yes,
            } => {
                confirm(yes)?;
                digest(&prompt_sha256)?;
                let usage: Usage = serde_json::from_slice(&read(&usage, 65536)?)?;
                ensure!(
                    usage.cost_known,
                    "externally verified known billing required"
                );
                store.reconcile_billing_and_abandon(
                    &scope,
                    &turn_id,
                    &prompt_sha256,
                    usage,
                    &evidence(&path)?,
                )?;
                print(
                    json!({"scope":scope,"turn_id":turn_id,"billing_reconciled":true,"turn_status":"failed","automatic_replay":false}),
                )
            }
        },
    }
}
