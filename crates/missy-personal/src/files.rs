//! This database deliberately owns its schema and path. Never opens legacy Store state.
use anyhow::{Result, ensure};
use rusqlite::Connection;
use std::{
    fs,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Component, Path},
    sync::Mutex,
};

pub(crate) static OPEN_LOCK: Mutex<()> = Mutex::new(());
const APP_ID: i64 = 0x4d535950;
const METADATA: [(&str, &str); 3] = [
    ("namespace", "personal:tenant+user:v1"),
    ("source", "explicit_user_memory"),
    ("trust", "untrusted_data"),
];

pub(crate) fn private_database(path: &Path, create: bool) -> Result<(u64, u64)> {
    ensure!(
        path.is_absolute(),
        "absolute personal database path required"
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
            let m = fs::symlink_metadata(&current)?;
            ensure!(
                m.is_dir() && !m.file_type().is_symlink(),
                "database ancestor must be a real directory"
            );
        }
    }
    let m = fs::symlink_metadata(parent)?;
    ensure!(
        m.uid() == unsafe { libc::geteuid() } && m.mode() & 0o077 == 0,
        "database parent must be current-user owned and private (0700)"
    );
    match fs::symlink_metadata(path) {
        Ok(m) => validate_file(&m)?,
        Err(e) if create && e.kind() == std::io::ErrorKind::NotFound => {
            // Closing a descriptor on an EXISTING SQLite inode drops process locks. Only
            // create a new file here; never reopen an existing file with std::fs.
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW)
                .open(path)?;
        }
        Err(e) => return Err(e.into()),
    }
    let m = fs::symlink_metadata(path)?;
    validate_file(&m)?;
    for suffix in ["-journal", "-wal", "-shm"] {
        let mut sidecar = path.as_os_str().to_os_string();
        sidecar.push(suffix);
        match fs::symlink_metadata(sidecar) {
            Ok(m) => validate_file(&m)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            Err(e) => return Err(e.into()),
        }
    }
    Ok((m.dev(), m.ino()))
}

fn validate_file(m: &fs::Metadata) -> Result<()> {
    ensure!(
        m.is_file()
            && !m.file_type().is_symlink()
            && m.nlink() == 1
            && m.uid() == unsafe { libc::geteuid() }
            && m.mode() & 0o077 == 0,
        "personal database/sidecar must be private, owned, regular and singly linked"
    );
    Ok(())
}

type SchemaEntry = (String, String, String, Option<String>);

fn schema(c: &Connection) -> Result<Vec<SchemaEntry>> {
    Ok(
        c.prepare("SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name")?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?,
    )
}

pub(crate) fn migrate(c: &mut Connection) -> Result<()> {
    let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    let app: i64 = tx.pragma_query_value(None, "application_id", |r| r.get(0))?;
    let version: i64 = tx.pragma_query_value(None, "user_version", |r| r.get(0))?;
    ensure!(
        (app == APP_ID && version == 1) || (app == 0 && version == 0),
        "unrecognized personal memory application or schema version"
    );
    if version == 0 {
        ensure!(
            schema(&tx)?.is_empty(),
            "refusing nonempty unrecognized database"
        );
        tx.execute_batch(include_str!("schema.sql"))?;
        for (key, value) in METADATA {
            tx.execute("INSERT INTO metadata VALUES(?1,?2)", [key, value])?;
        }
        tx.pragma_update(None, "application_id", APP_ID)?;
        tx.pragma_update(None, "user_version", 1)?;
    }
    // Reject altered tables, triggers, indexes or views, including unsupported additions.
    let expected = Connection::open_in_memory()?;
    expected.execute_batch(include_str!("schema.sql"))?;
    ensure!(
        schema(&tx)? == schema(&expected)?,
        "personal memory schema mismatch"
    );
    let actual = tx
        .prepare("SELECT key,value FROM metadata ORDER BY key")?
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let expected: Vec<_> = METADATA
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    ensure!(actual == expected, "corrupt personal memory metadata");
    tx.commit()?;
    let integrity: String = c.pragma_query_value(None, "quick_check", |r| r.get(0))?;
    ensure!(integrity == "ok", "personal memory integrity check failed");
    // Verify FTS internal consistency too. Search content is verified separately on read.
    c.execute(
        "INSERT INTO personal_fts(personal_fts) VALUES('integrity-check')",
        [],
    )?;
    Ok(())
}
