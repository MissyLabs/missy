//! Explicit, offline operator administration. No network or process dispatch.
use anyhow::{Context, Result, ensure};
use clap::Subcommand;
use missy_core::{Config, McpConfig, Scope};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
};

#[derive(Subcommand)]
pub enum AdminCommand {
    /// Offline permission and configuration posture report. Values are redacted.
    SecurityScan,
    /// Sessions from turns and metadata in exactly this tenant/user/channel.
    Sessions {
        #[arg(long, default_value = "")]
        prefix: String,
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },
    /// Validate an explicit registration/removal/pin instruction and display its binding hashes.
    McpPlan { instruction: PathBuf },
    /// Apply the exact planned instruction against the exact original config bytes.
    McpApply {
        instruction: PathBuf,
        #[arg(long)]
        config_sha256: String,
        #[arg(long)]
        instruction_sha256: String,
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        confirm: bool,
    },
}

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum Instruction {
    Add { entry: McpConfig, manifest: PathBuf },
    Pin { name: String, manifest: PathBuf },
    Remove { name: String },
}
fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn private_read(path: &Path) -> Result<Vec<u8>> {
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .context("private file unavailable")?;
    let m = f.metadata()?;
    ensure!(
        m.is_file()
            && m.uid() == unsafe { libc::geteuid() }
            && m.nlink() == 1
            && m.mode() & 0o077 == 0
            && m.len() <= 2_097_152,
        "private file must be owned, private, regular, single-link and <=2 MiB"
    );
    let mut bytes = Vec::new();
    f.take(2_097_153).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 2_097_152, "private file too large");
    Ok(bytes)
}
fn safe_parent(path: &Path) -> Result<()> {
    ensure!(path.is_absolute(), "absolute path required");
    let m = std::fs::symlink_metadata(path.parent().context("parent required")?)?;
    ensure!(
        m.is_dir()
            && !m.file_type().is_symlink()
            && m.uid() == unsafe { libc::geteuid() }
            && m.mode() & 0o022 == 0,
        "unsafe parent directory"
    );
    Ok(())
}
fn create(path: &Path, bytes: &[u8]) -> Result<()> {
    safe_parent(path)?;
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    File::open(path.parent().unwrap())?.sync_all()?;
    Ok(())
}
fn manifest(path: &Path) -> Result<(String, String)> {
    let bytes = private_read(path)?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|_| anyhow::anyhow!("invalid tools manifest JSON"))?;
    let obj = v
        .as_object()
        .context("manifest must be a tools/list object")?;
    ensure!(
        obj.len() == 1 && obj.contains_key("tools"),
        "manifest must contain only tools"
    );
    let tools = v["tools"]
        .as_array()
        .context("manifest tools must be an array")?;
    let mut names = std::collections::BTreeSet::new();
    for tool in tools {
        let item = tool.as_object().context("invalid manifest tool")?;
        ensure!(
            item.keys().all(
                |k| ["name", "description", "inputSchema", "annotations"].contains(&k.as_str())
            ),
            "unknown manifest tool field"
        );
        let name = item
            .get("name")
            .and_then(|v| v.as_str())
            .context("manifest tool name required")?;
        ensure!(
            !name.is_empty()
                && name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
                && names.insert(name),
            "invalid or duplicate manifest tool name"
        );
        ensure!(
            item.get("inputSchema").is_some_and(|v| v.is_object()),
            "manifest tool inputSchema must be an object"
        );
        ensure!(
            item.get("description").is_none_or(|v| v.is_string()),
            "manifest tool description must be a string"
        );
    }
    Ok((missy_tools::manifest_digest(&v), hash(&bytes)))
}
fn candidate(cfg: &Config, instruction: &Instruction) -> Result<(Config, Option<String>)> {
    let mut next = cfg.clone();
    let (entry, digest) = match instruction {
        Instruction::Add {
            entry,
            manifest: path,
        } => {
            ensure!(
                !next.mcp.iter().any(|x| x.name == entry.name),
                "MCP name already exists"
            );
            let (pin, raw) = manifest(path)?;
            let mut e = entry.clone();
            e.manifest_sha256 = pin;
            next.mcp.push(e.clone());
            (Some(e), Some(raw))
        }
        Instruction::Pin {
            name,
            manifest: path,
        } => {
            let (pin, raw) = manifest(path)?;
            let e = next
                .mcp
                .iter_mut()
                .find(|x| x.name == *name)
                .context("MCP name not found")?;
            e.manifest_sha256 = pin;
            (Some(e.clone()), Some(raw))
        }
        Instruction::Remove { name } => {
            let n = next.mcp.len();
            next.mcp.retain(|x| x.name != *name);
            ensure!(next.mcp.len() + 1 == n, "MCP name missing or duplicated");
            (None, None)
        }
    };
    if let Some(e) = entry {
        ensure!(
            !e.name.is_empty()
                && e.name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
                && e.timeout_seconds > 0,
            "invalid MCP name or timeout"
        );
        ensure!(
            !e.command.is_empty()
                && next.policy.mcp_enabled
                && next.policy.allowed_commands.iter().any(|a| a == &e.command),
            "MCP requires enabled policy and explicitly allowed exact argv; no authority was granted"
        );
        ensure!(
            !e.env
                .iter()
                .any(|(k, v)| k.is_empty() || k.contains(['=', '\0']) || v.contains('\0')),
            "invalid MCP environment"
        );
    }
    next.validate()?;
    Ok((next, digest))
}
fn posture(path: &Path, directory: bool) -> serde_json::Value {
    match std::fs::symlink_metadata(path) {
        Ok(m) => {
            json!({"exists":true,"symlink":m.file_type().is_symlink(),"expected_type":if directory {m.is_dir()} else {m.is_file()},"owned":m.uid()==unsafe{libc::geteuid()},"private":m.mode()&0o077==0,"single_link":directory || m.nlink()==1})
        }
        Err(e) => json!({"exists":false,"error_kind":format!("{:?}",e.kind())}),
    }
}
fn plan(config_path: &Path, instruction: &Path) -> Result<(Vec<u8>, Config, String)> {
    let before = private_read(config_path)?;
    let cfg: Config =
        serde_yaml_ng::from_slice(&before).map_err(|_| anyhow::anyhow!("invalid configuration"))?;
    cfg.validate()?;
    let bytes = private_read(instruction)?;
    let request: Instruction =
        serde_json::from_slice(&bytes).map_err(|_| anyhow::anyhow!("invalid MCP instruction"))?;
    let (next, manifest_hash) = candidate(&cfg, &request)?;
    let binding = hash(&serde_json::to_vec(&(hash(&bytes), manifest_hash))?);
    Ok((before, next, binding))
}
pub fn execute(
    config: &Config,
    config_path: &Path,
    command: AdminCommand,
    scope: Scope,
) -> Result<()> {
    match command {
        AdminCommand::SecurityScan => {
            let deps:Vec<_>=config.mcp.iter().map(|e| {
                let found=e.command.first().map(|cmd| {
                    if cmd.contains('/') { Path::new(cmd).is_file() } else { std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()).any(|p|p.join(cmd).is_file()) }
                }).unwrap_or(false);
                json!({"command_present":found,"argv_allowed":config.policy.allowed_commands.iter().any(|a|a==&e.command)})
            }).collect();
            println!(
                "{}",
                json!({"config":posture(config_path,false),"state_dir":posture(&config.state_dir,true),"shell_enabled":config.policy.shell_enabled,"mcp_enabled":config.policy.mcp_enabled,"allow_private_network":config.policy.allow_private_network,"allow_file_delete":config.policy.allow_file_delete,"mcp_dependencies":deps,"values_redacted":true,"network_checked":false})
            );
        }
        AdminCommand::Sessions { prefix, limit } => {
            scope.validate()?;
            let db = config.state_dir.join("runtime.sqlite3");
            ensure!(db.exists(), "state database does not exist");
            let sessions = missy_store::Store::open(db)?.scoped_sessions(&scope, &prefix, limit)?;
            println!(
                "{}",
                json!({"sessions":sessions,"scope":{"tenant":scope.tenant,"user":scope.user,"channel":scope.channel},"sources":["turns","session_metadata"],"memory_fallback":false})
            );
        }
        AdminCommand::McpPlan { instruction } => {
            let (before, _, binding) = plan(config_path, &instruction)?;
            println!(
                "{}",
                json!({"config_sha256":hash(&before),"instruction_sha256":binding,"applied":false,"values_redacted":true,"running_services_changed":false})
            );
        }
        AdminCommand::McpApply {
            instruction,
            config_sha256,
            instruction_sha256,
            snapshot,
            confirm,
        } => {
            ensure!(
                confirm,
                "MCP configuration apply requires explicit confirmation"
            );
            safe_parent(config_path)?;
            let lock_path = config_path.with_extension("yaml.lock");
            let lock = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
                .open(lock_path)?;
            let m = lock.metadata()?;
            ensure!(
                m.is_file()
                    && m.nlink() == 1
                    && m.uid() == unsafe { libc::geteuid() }
                    && m.mode() & 0o077 == 0,
                "invalid config operation lock"
            );
            ensure!(
                unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
                "another config mutation is active"
            );
            let (before, next, binding) = plan(config_path, &instruction)?;
            ensure!(
                hash(&before) == config_sha256 && binding == instruction_sha256,
                "plan binding mismatch"
            );
            let canonical = std::fs::canonicalize(config_path)?;
            let backup = json!({"format":"missy-native-config-snapshot-v1","config_path":canonical,"sha256":hash(&before),"contents":std::str::from_utf8(&before)?});
            let snapshot_bytes = zeroize::Zeroizing::new(serde_json::to_vec_pretty(&backup)?);
            create(&snapshot, &snapshot_bytes)?;
            let temp = canonical
                .parent()
                .unwrap()
                .join(format!(".config-admin-{}", uuid::Uuid::new_v4()));
            let result = (|| -> Result<()> {
                let output = zeroize::Zeroizing::new(serde_yaml_ng::to_string(&next)?);
                create(&temp, output.as_bytes())?;
                ensure!(
                    private_read(config_path)? == before,
                    "configuration changed during apply"
                );
                std::fs::rename(&temp, &canonical)?;
                File::open(canonical.parent().unwrap())?.sync_all()?;
                Ok(())
            })();
            if result.is_err() {
                let _ = std::fs::remove_file(&temp);
            }
            result?;
            println!(
                "{}",
                json!({"applied":true,"snapshot_created":true,"restart_required":true,"running_services_changed":false,"hot_reload_supported":false})
            );
        }
    }
    Ok(())
}
