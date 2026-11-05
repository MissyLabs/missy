use anyhow::{Result, ensure};
use missy_core::Scope;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::{File, OpenOptions},
    os::{
        fd::AsRawFd,
        unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
    sync::Mutex,
};

pub(crate) struct Journal {
    conn: Mutex<Connection>,
    _lock: File,
    path: PathBuf,
    identity: (u64, u64),
    max: usize,
}
impl Journal {
    pub fn open(dir: &Path, max: usize) -> Result<Self> {
        ensure!(
            dir.is_absolute()
                && !dir
                    .components()
                    .any(|c| matches!(c, std::path::Component::ParentDir)),
            "absolute state directory required"
        );
        if !dir.exists() {
            let mut b = std::fs::DirBuilder::new();
            b.mode(0o700);
            b.create(dir)?;
        }
        for a in dir.ancestors() {
            let m = std::fs::symlink_metadata(a)?;
            ensure!(
                m.is_dir() && !m.file_type().is_symlink(),
                "symlink state ancestor denied"
            );
        }
        let m = std::fs::symlink_metadata(dir)?;
        ensure!(
            m.uid() == unsafe { libc::geteuid() } && m.mode() & 0o077 == 0,
            "state directory must be owner-private"
        );
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(dir.join("workloads.lock"))?;
        let m = lock.metadata()?;
        ensure!(
            m.is_file()
                && m.nlink() == 1
                && m.uid() == unsafe { libc::geteuid() }
                && m.mode() & 0o077 == 0,
            "unsafe workload lock"
        );
        ensure!(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "workload coordinator already running"
        );
        let path = dir.join("workloads.sqlite3");
        private_db(&path)?;
        let m = std::fs::symlink_metadata(&path)?;
        let identity = (m.dev(), m.ino());
        for suffix in ["-wal", "-shm"] {
            let p = PathBuf::from(format!("{}{suffix}", path.display()));
            if p.symlink_metadata().is_ok() {
                private_db(&p)?;
            }
        }
        let mut conn = Connection::open_with_flags(
            &path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let app: i64 = tx.pragma_query_value(None, "application_id", |r| r.get(0))?;
        let ver: i64 = tx.pragma_query_value(None, "user_version", |r| r.get(0))?;
        ensure!(
            (app == 0 && ver == 0) || (app == 0x4d57524b && ver == 1),
            "foreign workload database refused"
        );
        if ver == 0 {
            let n: i64 = tx.query_row(
                "SELECT count(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'",
                [],
                |r| r.get(0),
            )?;
            ensure!(n == 0, "nonempty foreign database refused");
            tx.execute_batch("CREATE TABLE records(kind TEXT NOT NULL,id TEXT NOT NULL,scope TEXT NOT NULL,data TEXT NOT NULL,PRIMARY KEY(kind,id));")?;
            tx.pragma_update(None, "application_id", 0x4d57524b)?;
            tx.pragma_update(None, "user_version", 1)?;
        }
        tx.commit()?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "FULL")?;
        Ok(Self {
            conn: Mutex::new(conn),
            _lock: lock,
            path,
            identity,
            max,
        })
    }
    fn check(&self) -> Result<()> {
        let m = std::fs::symlink_metadata(&self.path)?;
        ensure!(
            m.is_file()
                && !m.file_type().is_symlink()
                && m.nlink() == 1
                && m.uid() == unsafe { libc::geteuid() }
                && m.mode() & 0o077 == 0
                && (m.dev(), m.ino()) == self.identity,
            "workload database identity changed"
        );
        Ok(())
    }
    pub fn get<T: DeserializeOwned>(
        &self,
        kind: &str,
        id: &str,
        scope: &Scope,
    ) -> Result<Option<T>> {
        self.check()?;
        scope.validate()?;
        let c = self
            .conn
            .lock()
            .map_err(|_| anyhow::anyhow!("journal poisoned"))?;
        let row: Option<(String, String)> = c
            .query_row(
                "SELECT scope,data FROM records WHERE kind=?1 AND id=?2",
                params![kind, id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        row.map(|(s, d)| {
            ensure!(
                s == serde_json::to_string(scope)?,
                "foreign workload scope denied"
            );
            Ok(serde_json::from_str(&d)?)
        })
        .transpose()
    }
    pub fn put(&self, kind: &str, id: &str, scope: &Scope, data: &impl Serialize) -> Result<()> {
        self.check()?;
        scope.validate()?;
        let raw = serde_json::to_string(data)?;
        ensure!(
            raw.len() <= 4 * 1024 * 1024,
            "workload record exceeds bound"
        );
        let mut c = self
            .conn
            .lock()
            .map_err(|_| anyhow::anyhow!("journal poisoned"))?;
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let old: Option<String> = tx
            .query_row(
                "SELECT scope FROM records WHERE kind=?1 AND id=?2",
                params![kind, id],
                |r| r.get(0),
            )
            .optional()?;
        let s = serde_json::to_string(scope)?;
        if let Some(old) = old {
            ensure!(old == s, "foreign record replacement denied");
            tx.execute(
                "UPDATE records SET data=?3 WHERE kind=?1 AND id=?2",
                params![kind, id, raw],
            )?;
        } else {
            let n: i64 = tx.query_row("SELECT count(*) FROM records", [], |r| r.get(0))?;
            ensure!(n < self.max as i64, "workload record capacity exceeded");
            tx.execute(
                "INSERT INTO records(kind,id,scope,data)VALUES(?1,?2,?3,?4)",
                params![kind, id, s, raw],
            )?;
        }
        tx.commit()?;
        Ok(())
    }
    pub fn list<T: DeserializeOwned>(&self, kind: &str, scope: Option<&Scope>) -> Result<Vec<T>> {
        self.check()?;
        let c = self
            .conn
            .lock()
            .map_err(|_| anyhow::anyhow!("journal poisoned"))?;
        let mut q = c.prepare("SELECT scope,data FROM records WHERE kind=?1 ORDER BY id")?;
        let rows = q.query_map([kind], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })?;
        let key = scope.map(serde_json::to_string).transpose()?;
        let mut out = Vec::new();
        for row in rows {
            let (s, d) = row?;
            if key.as_ref().is_none_or(|k| k == &s) {
                out.push(serde_json::from_str(&d)?);
            }
        }
        Ok(out)
    }
}
fn private_db(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(m) => ensure!(
            m.is_file()
                && !m.file_type().is_symlink()
                && m.nlink() == 1
                && m.uid() == unsafe { libc::geteuid() }
                && m.mode() & 0o077 == 0,
            "unsafe workload SQLite file"
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW)
                .open(path)?;
            drop(f);
        }
        Err(e) => return Err(e.into()),
    }
    Ok(())
}
