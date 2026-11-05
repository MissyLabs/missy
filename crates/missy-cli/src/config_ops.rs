//! Offline configuration operations. Outputs contain field names, never values.
use anyhow::{Context, Result, ensure};
use clap::Subcommand;
use serde::{Deserialize, Serialize};
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
pub enum ConfigCommand {
    Validate,
    Backup {
        destination: PathBuf,
    },
    /// Redacted semantic diff. No files are changed.
    Diff {
        candidate: PathBuf,
    },
    /// Validate candidate and print a redacted restart-only change plan.
    Plan {
        candidate: PathBuf,
    },
    /// Restore a snapshot bound to this exact config pathname, without signaling any process.
    Rollback {
        snapshot: PathBuf,
        #[arg(long)]
        confirm: bool,
    },
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    format: String,
    config_path: PathBuf,
    sha256: String,
    contents: String,
}
fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn read_private(path: &Path) -> Result<Vec<u8>> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .context("private configuration/snapshot unavailable")?;
    let m = file.metadata()?;
    ensure!(
        m.is_file()
            && m.uid() == unsafe { libc::geteuid() }
            && m.nlink() == 1
            && m.mode() & 0o077 == 0
            && m.len() <= 2_097_152,
        "configuration/snapshot must be an owned private single-link regular file <=2MiB"
    );
    let mut bytes = vec![];
    file.take(2_097_153).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 2_097_152, "configuration/snapshot too large");
    Ok(bytes)
}
fn private_parent(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .context("absolute destination with parent required")?;
    let m = std::fs::symlink_metadata(parent)?;
    ensure!(
        m.is_dir()
            && !m.file_type().is_symlink()
            && m.uid() == unsafe { libc::geteuid() }
            && m.mode() & 0o022 == 0,
        "configuration/snapshot parent must be owned and not writable by others"
    );
    Ok(())
}
fn create(path: &Path, bytes: &[u8]) -> Result<()> {
    ensure!(path.is_absolute(), "absolute snapshot destination required");
    private_parent(path)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .context("snapshot destination already exists or is unavailable")?;
    file.write_all(bytes)?;
    file.sync_all()?;
    File::open(path.parent().unwrap())?.sync_all()?;
    Ok(())
}
fn changes(prefix: &str, a: &serde_json::Value, b: &serde_json::Value, out: &mut Vec<String>) {
    if a == b {
        return;
    }
    if let (Some(a), Some(b)) = (a.as_object(), b.as_object()) {
        let keys: std::collections::BTreeSet<_> = a.keys().chain(b.keys()).collect();
        for k in keys {
            let path = if prefix.is_empty() {
                k.clone()
            } else {
                format!("{prefix}.{k}")
            };
            changes(
                &path,
                a.get(k).unwrap_or(&serde_json::Value::Null),
                b.get(k).unwrap_or(&serde_json::Value::Null),
                out,
            );
        }
    } else {
        out.push(prefix.to_owned());
    }
}
pub fn execute(path: &Path, command: ConfigCommand) -> Result<()> {
    // Reject symlink/hardlink/special file even for rollback of malformed YAML.
    read_private(path)?;
    let canonical = std::fs::canonicalize(path)?;
    match command {
        ConfigCommand::Validate => {
            crate::production::fingerprint(&crate::load_config(path)?)?;
            println!(
                "{}",
                serde_json::json!({"valid":true,"hot_reload_supported":false})
            );
        }
        ConfigCommand::Backup { destination } => {
            crate::load_config(path)?;
            let bytes = read_private(path)?;
            let snapshot = Snapshot {
                format: "missy-native-config-snapshot-v1".into(),
                config_path: canonical,
                sha256: hash(&bytes),
                contents: String::from_utf8(bytes).context("configuration must be UTF-8")?,
            };
            let mut bytes = zeroize::Zeroizing::new(serde_json::to_vec_pretty(&snapshot)?);
            create(&destination, &bytes)?;
            use zeroize::Zeroize;
            bytes.zeroize();
            println!(
                "{}",
                serde_json::json!({"backed_up":true,"contains_private_configuration":true,"running_services_changed":false})
            );
        }
        ConfigCommand::Diff { candidate } | ConfigCommand::Plan { candidate } => {
            let cfg = crate::load_config(path)?;
            let next = crate::load_config(&candidate)?;
            let current_hash = crate::production::fingerprint(&cfg)?;
            let next_hash = crate::production::fingerprint(&next)?;
            let mut fields = vec![];
            changes(
                "",
                &serde_json::to_value(cfg)?,
                &serde_json::to_value(next)?,
                &mut fields,
            );
            println!(
                "{}",
                serde_json::json!({"valid":true,"changed_fields":fields,"restart_required":current_hash!=next_hash,
                "values_redacted":true,"running_services_changed":false,"applied":false})
            );
        }
        ConfigCommand::Rollback { snapshot, confirm } => {
            ensure!(
                confirm,
                "configuration rollback requires explicit confirmation"
            );
            private_parent(&canonical)?;
            let record: Snapshot = serde_json::from_slice(&read_private(&snapshot)?)
                .context("invalid configuration snapshot")?;
            ensure!(
                record.format == "missy-native-config-snapshot-v1"
                    && record.config_path == canonical
                    && record.sha256 == hash(record.contents.as_bytes()),
                "snapshot format/path/hash mismatch"
            );
            let lock_path = canonical.with_extension("yaml.lock");
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
            let before = read_private(path)?;
            let temp = canonical
                .parent()
                .unwrap()
                .join(format!(".config-restore-{}", uuid::Uuid::new_v4()));
            let restore = (|| -> Result<()> {
                create(&temp, record.contents.as_bytes())?;
                crate::production::fingerprint(&crate::load_config(&temp)?)?;
                ensure!(
                    read_private(path)? == before,
                    "configuration changed during rollback"
                );
                std::fs::rename(&temp, &canonical)?;
                File::open(canonical.parent().unwrap())?.sync_all()?;
                Ok(())
            })();
            if restore.is_err() {
                let _ = std::fs::remove_file(temp);
            }
            restore?;
            println!(
                "{}",
                serde_json::json!({"restored":true,"restart_required":true,"running_services_changed":false,"hot_reload_supported":false})
            );
        }
    }
    Ok(())
}
