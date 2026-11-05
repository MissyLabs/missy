use anyhow::{Result, ensure};
use rusqlite::Connection;
use std::{
    fs,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Component, Path},
    sync::Mutex,
};

pub(crate) static OPEN_LOCK: Mutex<()> = Mutex::new(());
pub(crate) const APP_ID: i64 = 0x4d535952;

pub(crate) fn private_database(path: &Path) -> Result<()> {
    ensure!(
        path.is_absolute(),
        "explicit absolute persistent retrieval database path required"
    );
    ensure!(
        !path
            .as_os_str()
            .as_encoded_bytes()
            .split(|b| *b == b'/')
            .any(|s| s == b"." || s == b".."),
        "dot path components denied"
    );
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("database parent required"))?;
    let mut current = std::path::PathBuf::from("/");
    for part in parent.components() {
        if let Component::Normal(part) = part {
            current.push(part);
            let meta = fs::symlink_metadata(&current)?;
            ensure!(
                meta.is_dir() && !meta.file_type().is_symlink(),
                "database ancestor must be a real directory"
            );
        }
    }
    let meta = fs::symlink_metadata(parent)?;
    ensure!(
        meta.uid() == unsafe { libc::geteuid() } && meta.mode() & 0o077 == 0,
        "database parent must be current-user owned and private (0700)"
    );
    match fs::symlink_metadata(path) {
        Ok(meta) => validate_file(&meta)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Never open/close an existing database here: that drops process SQLite locks.
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW)
                .open(path)?;
        }
        Err(e) => return Err(e.into()),
    }
    for suffix in ["-journal", "-wal", "-shm"] {
        let mut sidecar = path.as_os_str().to_os_string();
        sidecar.push(suffix);
        match fs::symlink_metadata(sidecar) {
            Ok(meta) => validate_file(&meta)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
fn validate_file(meta: &fs::Metadata) -> Result<()> {
    ensure!(
        meta.is_file()
            && !meta.file_type().is_symlink()
            && meta.nlink() == 1
            && meta.uid() == unsafe { libc::geteuid() }
            && meta.mode() & 0o077 == 0,
        "database/sidecar must be private, owned, regular and singly linked"
    );
    Ok(())
}
pub(crate) fn migrate(c: &mut Connection, model: &str, dimension: usize) -> Result<()> {
    let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    let app: i64 = tx.pragma_query_value(None, "application_id", |r| r.get(0))?;
    let version: i64 = tx.pragma_query_value(None, "user_version", |r| r.get(0))?;
    ensure!(
        (app == APP_ID && version == 1) || (app == 0 && version == 0),
        "unrecognized retrieval application or schema version"
    );
    if version == 0 {
        let count: i64 = tx.query_row(
            "SELECT count(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'",
            [],
            |r| r.get(0),
        )?;
        ensure!(count == 0, "refusing nonempty unrecognized database");
        tx.execute_batch(include_str!("schema.sql"))?;
        tx.execute(
            "INSERT INTO metadata VALUES('embedder',?1),('dimension',?2)",
            rusqlite::params![model, dimension.to_string()],
        )?;
        tx.pragma_update(None, "application_id", APP_ID)?;
        tx.pragma_update(None, "user_version", 1)?;
    }
    let stored: String =
        tx.query_row("SELECT value FROM metadata WHERE key='embedder'", [], |r| {
            r.get(0)
        })?;
    let dim: String = tx.query_row(
        "SELECT value FROM metadata WHERE key='dimension'",
        [],
        |r| r.get(0),
    )?;
    ensure!(
        stored == model && dim == dimension.to_string(),
        "embedder mismatch: explicitly reindex into a new database"
    );
    tx.commit()?;
    let integrity: String = c.pragma_query_value(None, "quick_check", |r| r.get(0))?;
    ensure!(
        integrity == "ok",
        "retrieval database integrity check failed"
    );
    ensure!(
        !c.prepare("PRAGMA foreign_key_check")?.exists([])?,
        "retrieval foreign key corruption"
    );
    Ok(())
}
