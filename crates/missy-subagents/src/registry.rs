use super::*;
use anyhow::anyhow;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
#[cfg(unix)]
use std::os::unix::{
    fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
    io::AsRawFd,
};
use std::{
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

/// Private SQLite receipts; no inherited service handles or model supplied paths.
pub struct AgentRegistry {
    conn: Mutex<Connection>,
    path: PathBuf,
    #[cfg(unix)]
    identity: (u64, u64),
}
static OPEN_LOCK: Mutex<()> = Mutex::new(());
impl AgentRegistry {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let absolute = if path.as_ref().is_absolute() {
            path.as_ref().to_owned()
        } else {
            std::env::current_dir()?.join(path)
        };
        ensure!(
            !absolute
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir)),
            "registry path traversal denied"
        );
        let path = absolute.as_path();
        ensure!(
            path != Path::new(":memory:"),
            "persistent registry required"
        );
        let _lock = OPEN_LOCK
            .lock()
            .map_err(|_| anyhow!("registry open lock poisoned"))?;
        private_file(path)?;
        #[cfg(unix)]
        let identity = {
            let metadata = fs::symlink_metadata(path)?;
            (metadata.dev(), metadata.ino())
        };
        for suffix in ["-wal", "-shm"] {
            let sidecar = PathBuf::from(format!("{}{suffix}", path.display()));
            if sidecar.symlink_metadata().is_ok() {
                private_file(&sidecar)?;
            }
        }
        let mut conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )?;
        conn.busy_timeout(Duration::from_secs(5))?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let app: i64 = tx.pragma_query_value(None, "application_id", |r| r.get(0))?;
        let version: i64 = tx.pragma_query_value(None, "user_version", |r| r.get(0))?;
        ensure!(
            (app == 0 && version == 0) || (app == 0x4d534147 && version == 1),
            "unrecognized agent registry"
        );
        if version == 0 {
            let count: i64 = tx.query_row(
                "SELECT count(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'",
                [],
                |r| r.get(0),
            )?;
            ensure!(count == 0, "refusing nonempty foreign registry");
            tx.execute_batch("CREATE TABLE roots(root_id TEXT PRIMARY KEY, scope TEXT NOT NULL, immutable_hash TEXT NOT NULL, receipt TEXT NOT NULL);")?;
            tx.pragma_update(None, "application_id", 0x4d534147)?;
            tx.pragma_update(None, "user_version", 1)?;
        }
        tx.commit()?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "FULL")?;
        let registry = Self {
            conn: Mutex::new(conn),
            path: path.to_owned(),
            #[cfg(unix)]
            identity,
        };
        registry.check_identity()?;
        Ok(registry)
    }
    fn check_identity(&self) -> Result<()> {
        let metadata = fs::symlink_metadata(&self.path)?;
        ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "registry path replaced"
        );
        #[cfg(unix)]
        ensure!(
            (metadata.dev(), metadata.ino()) == self.identity
                && metadata.uid() == unsafe { libc::geteuid() }
                && metadata.nlink() == 1
                && metadata.mode() & 0o077 == 0,
            "registry inode, owner, links or permissions changed"
        );
        Ok(())
    }
    pub fn receipt(&self, scope: &Scope, root_id: &str) -> Result<Option<RootReceipt>> {
        self.check_identity()?;
        scope.validate()?;
        let conn = self.conn.lock().map_err(|_| anyhow!("registry poisoned"))?;
        let row: Option<(String, String)> = conn
            .query_row(
                "SELECT scope,receipt FROM roots WHERE root_id=?1",
                [root_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        row.map(|(stored, data)| {
            ensure!(
                stored == serde_json::to_string(scope)?,
                "foreign root scope denied"
            );
            let receipt: RootReceipt = serde_json::from_str(&data)?;
            ensure!(
                receipt.scope == *scope && receipt.root_id == root_id,
                "registry receipt identity mismatch"
            );
            Ok(receipt)
        })
        .transpose()
    }
    pub(crate) fn insert(&self, receipt: &RootReceipt) -> Result<()> {
        self.check_identity()?;
        self.conn
            .lock()
            .map_err(|_| anyhow!("registry poisoned"))?
            .execute(
                "INSERT INTO roots(root_id,scope,immutable_hash,receipt) VALUES(?1,?2,?3,?4)",
                params![
                    receipt.root_id,
                    serde_json::to_string(&receipt.scope)?,
                    receipt.immutable_hash,
                    serde_json::to_string(receipt)?
                ],
            )?;
        Ok(())
    }
    pub(crate) fn save(&self, receipt: &RootReceipt) -> Result<()> {
        self.check_identity()?;
        let n = self
            .conn
            .lock()
            .map_err(|_| anyhow!("registry poisoned"))?
            .execute(
                "UPDATE roots SET receipt=?4 WHERE root_id=?1 AND scope=?2 AND immutable_hash=?3",
                params![
                    receipt.root_id,
                    serde_json::to_string(&receipt.scope)?,
                    receipt.immutable_hash,
                    serde_json::to_string(receipt)?
                ],
            )?;
        ensure!(n == 1, "immutable root identity mismatch");
        Ok(())
    }
    pub(crate) fn lock_root(&self, root_id: &str) -> Result<File> {
        self.check_identity()?;
        ensure!(
            root_id.len() == 64 && root_id.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid root identity"
        );
        let path = self.path.with_file_name(format!(".agent-{root_id}.lock"));
        private_file(&path)?;
        let mut o = OpenOptions::new();
        o.read(true).write(true);
        #[cfg(unix)]
        o.custom_flags(libc::O_NOFOLLOW);
        let f = o.open(path)?;
        #[cfg(unix)]
        {
            let m = f.metadata()?;
            ensure!(
                m.is_file()
                    && m.uid() == unsafe { libc::geteuid() }
                    && m.nlink() == 1
                    && m.mode() & 0o077 == 0,
                "unsafe opened root lock"
            );
        }
        #[cfg(unix)]
        ensure!(
            unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "root coordinator already active"
        );
        #[cfg(not(unix))]
        anyhow::bail!("native subagent coordination requires Unix file locking");
        Ok(f)
    }
}
fn private_file(path: &Path) -> Result<()> {
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
    for ancestor in parent.ancestors() {
        let m = fs::symlink_metadata(ancestor)?;
        ensure!(
            m.is_dir() && !m.file_type().is_symlink(),
            "symlink registry ancestor denied"
        );
    }
    let m = fs::symlink_metadata(parent)?;
    ensure!(
        m.is_dir() && !m.file_type().is_symlink(),
        "registry parent must be a real directory"
    );
    #[cfg(unix)]
    ensure!(
        m.uid() == unsafe { libc::geteuid() } && m.mode() & 0o077 == 0,
        "registry parent must be current-user private (0700)"
    );
    match fs::symlink_metadata(path) {
        Ok(m) => {
            ensure!(
                m.is_file() && !m.file_type().is_symlink(),
                "registry must be a regular non-symlink file"
            );
            #[cfg(unix)]
            {
                ensure!(
                    m.uid() == unsafe { libc::geteuid() }
                        && m.nlink() == 1
                        && m.mode() & 0o077 == 0,
                    "unsafe registry ownership, permissions or links"
                );
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let mut o = OpenOptions::new();
            o.write(true).create_new(true);
            #[cfg(unix)]
            o.mode(0o600).custom_flags(libc::O_NOFOLLOW);
            match o.open(path) {
                Ok(f) => drop(f),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    return private_file(path);
                }
                Err(e) => return Err(e.into()),
            }
        }
        Err(e) => return Err(e.into()),
    }
    Ok(())
}
