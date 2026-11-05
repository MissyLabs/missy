use super::*;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
pub(super) static OPEN_LOCK: Mutex<()> = Mutex::new(());
pub(super) fn private_database(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !parent.exists() {
        let mut b = fs::DirBuilder::new();
        b.recursive(true);
        #[cfg(unix)]
        b.mode(0o700);
        b.create(parent)?;
    }
    let pmeta = fs::symlink_metadata(parent)?;
    ensure!(
        pmeta.is_dir() && !pmeta.file_type().is_symlink(),
        "database parent must be a real private directory"
    );
    #[cfg(unix)]
    ensure!(
        pmeta.uid() == unsafe { libc::geteuid() } && pmeta.mode() & 0o022 == 0,
        "database parent must be current-user owned and not group/world writable"
    );
    match fs::symlink_metadata(path) {
        Ok(meta) => {
            ensure!(
                meta.is_file() && !meta.file_type().is_symlink(),
                "database must be a regular non-symlink file"
            );
            #[cfg(unix)]
            {
                ensure!(
                    meta.uid() == unsafe { libc::geteuid() } && meta.nlink() == 1,
                    "database owner or hardlink rejected"
                );
                fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
            }
            // Never open/close existing DB outside SQLite: process locks lost.
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let mut o = fs::OpenOptions::new();
            o.write(true).create_new(true);
            #[cfg(unix)]
            o.mode(0o600).custom_flags(libc::O_NOFOLLOW);
            match o.open(path) {
                Ok(f) => drop(f),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    return private_database(path);
                }
                Err(e) => return Err(e.into()),
            }
        }
        Err(e) => return Err(e.into()),
    }
    for suffix in ["-wal", "-shm"] {
        let mut name = path.as_os_str().to_os_string();
        name.push(suffix);
        let sidecar = std::path::PathBuf::from(name);
        match fs::symlink_metadata(&sidecar) {
            Ok(m) => {
                ensure!(
                    m.is_file() && !m.file_type().is_symlink(),
                    "unsafe SQLite sidecar"
                );
                #[cfg(unix)]
                {
                    ensure!(
                        m.uid() == unsafe { libc::geteuid() } && m.nlink() == 1,
                        "unsafe sidecar owner/links"
                    );
                    fs::set_permissions(sidecar, fs::Permissions::from_mode(0o600))?;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
pub(super) fn migrate(c: &mut Connection) -> Result<()> {
    let app: i64 = c.pragma_query_value(None, "application_id", |r| r.get(0))?;
    ensure!(
        app == 0 || app == 0x4d535953,
        "database belongs to another application"
    );
    let tx = c.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let v: i64 = tx.pragma_query_value(None, "user_version", |r| r.get(0))?;
    ensure!(v <= 4, "database schema newer than binary");
    if v == 0 {
        let tables: i64 = tx.query_row(
            "SELECT count(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'",
            [],
            |r| r.get(0),
        )?;
        ensure!(
            tables == 0,
            "refusing to initialize over an unrecognized nonempty database"
        );
    }
    if v < 1 {
        tx.execute_batch(include_str!("schema-v1.sql"))?;
    }
    if v < 2 {
        tx.execute_batch("CREATE VIRTUAL TABLE memory_fts USING fts5(content, tokenize='unicode61'); INSERT INTO memory_fts(rowid,content) SELECT rowid,content FROM memories;")?;
    }
    if v < 3 {
        tx.execute_batch("ALTER TABLE checkpoints ADD COLUMN format_version INTEGER NOT NULL DEFAULT 1; ALTER TABLE checkpoints ADD COLUMN digest TEXT NOT NULL DEFAULT '';")?;
        let rows: Vec<(String, String, String)> = {
            let mut q = tx.prepare("SELECT scope,turn_id,data FROM checkpoints")?;
            q.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
                .collect::<rusqlite::Result<_>>()?
        };
        for (key, id, data) in rows {
            let cp: Checkpoint = parse_json(data.clone())?;
            checkpoint::validate(&cp)?;
            let digest = checkpoint::digest(&key, &id, &data);
            tx.execute(
                "UPDATE checkpoints SET digest=?1 WHERE scope=?2 AND turn_id=?3",
                params![digest, key, id],
            )?;
        }
    }
    tx.execute_batch("CREATE TABLE IF NOT EXISTS exact_tool_requests(scope TEXT NOT NULL,turn_id TEXT NOT NULL,calls TEXT NOT NULL,PRIMARY KEY(scope,turn_id),FOREIGN KEY(scope,turn_id) REFERENCES turns(scope,id) ON DELETE CASCADE);
    CREATE TABLE IF NOT EXISTS raw_tool_evidence(scope TEXT NOT NULL,turn_id TEXT NOT NULL,call_id TEXT NOT NULL,data TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(scope,turn_id,call_id),FOREIGN KEY(scope,turn_id,call_id) REFERENCES tools(scope,turn_id,call_id) ON DELETE CASCADE);")?;
    tx.execute_batch("CREATE TABLE IF NOT EXISTS session_metadata(scope TEXT PRIMARY KEY,display_name TEXT NOT NULL DEFAULT '',history_cutoff INTEGER NOT NULL DEFAULT 0);
    CREATE TABLE IF NOT EXISTS budget_policy(id INTEGER PRIMARY KEY CHECK(id=1),max_cost INTEGER NOT NULL,max_tokens INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS budget_reservations(scope TEXT NOT NULL,turn_id TEXT NOT NULL,cost INTEGER NOT NULL,tokens INTEGER NOT NULL,used_cost INTEGER NOT NULL DEFAULT 0,used_tokens INTEGER NOT NULL DEFAULT 0,uncertain INTEGER NOT NULL DEFAULT 0,settled INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(scope,turn_id),FOREIGN KEY(scope,turn_id) REFERENCES turns(scope,id));")?;
    tx.pragma_update(None, "application_id", 0x4d535953)?;
    tx.pragma_update(None, "user_version", 4)?;
    tx.commit()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn subprocess_lock_probe() {
        let Some(path) = std::env::var_os("MISSY_STORE_LOCK_PROBE") else {
            return;
        };
        let c = Connection::open(std::path::PathBuf::from(path)).unwrap();
        c.busy_timeout(Duration::ZERO).unwrap();
        let error = c
            .execute_batch("BEGIN IMMEDIATE;")
            .expect_err("existing process write lock was lost");
        assert_eq!(
            error.sqlite_error_code(),
            Some(rusqlite::ErrorCode::DatabaseBusy)
        );
    }
    #[test]
    fn securing_existing_file_does_not_release_sqlite_process_locks() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("lock.db");
        private_database(&p).unwrap();
        let c = Connection::open(&p).unwrap();
        c.execute_batch(
            "CREATE TABLE fixture(id); BEGIN IMMEDIATE; INSERT INTO fixture VALUES(1);",
        )
        .unwrap();
        private_database(&p).unwrap();
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "files::tests::subprocess_lock_probe",
                "--nocapture",
            ])
            .env("MISSY_STORE_LOCK_PROBE", &p)
            .output()
            .unwrap();
        assert!(
            child.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        c.execute_batch("ROLLBACK;").unwrap();
    }
}
