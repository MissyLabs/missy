use super::*;
use std::{
    ffi::{CString, OsStr},
    io::Read,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::{ffi::OsStrExt, fs::MetadataExt},
    },
    path::Component,
};

fn name(s: &OsStr) -> Result<CString> {
    CString::new(s.as_bytes()).map_err(|_| anyhow!("invalid path component"))
}
fn openat(dir: &File, component: &OsStr, flags: i32) -> Result<File> {
    let component = name(component)?;
    // SAFETY: CString is terminated and returned descriptor becomes uniquely owned.
    let fd = unsafe {
        libc::openat(
            dir.as_raw_fd(),
            component.as_ptr(),
            flags | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    ensure!(
        fd >= 0,
        "offline source/destination component unavailable or symlink refused"
    );
    Ok(unsafe { File::from_raw_fd(fd) })
}
fn check_path(path: &Path) -> Result<()> {
    ensure!(
        path.is_absolute(),
        "explicit absolute offline path required"
    );
    ensure!(
        !path
            .as_os_str()
            .as_bytes()
            .split(|b| *b == b'/')
            .any(|s| s == b"." || s == b".." || s == b".missy"),
        "dot components and Python .missy state paths are refused; export backups to a separate private directory first"
    );
    Ok(())
}
fn trusted_dir(dir: &File) -> Result<()> {
    let m = dir.metadata()?;
    let uid = unsafe { libc::geteuid() };
    ensure!(
        m.is_dir() && (m.uid() == uid || m.uid() == 0),
        "unsafe ancestor ownership"
    );
    // Root-owned sticky /tmp is safe when every following component is opened
    // O_NOFOLLOW and verified. Non-sticky writable ancestors are never accepted.
    ensure!(
        m.mode() & 0o022 == 0 || (m.uid() == 0 && m.mode() & libc::S_ISVTX != 0),
        "unsafe writable ancestor"
    );
    Ok(())
}
fn parent(path: &Path) -> Result<File> {
    check_path(path)?;
    let mut dir = File::open("/")?;
    trusted_dir(&dir)?;
    for c in path
        .parent()
        .ok_or_else(|| anyhow!("parent directory required"))?
        .components()
    {
        if let Component::Normal(c) = c {
            dir = openat(&dir, c, libc::O_RDONLY | libc::O_DIRECTORY)?;
            trusted_dir(&dir)?;
        }
    }
    Ok(dir)
}
pub(crate) struct PinnedSource {
    pub file: File,
    dir: File,
    basename: std::ffi::OsString,
    pub bytes: Vec<u8>,
    pub source: SourceDigest,
}
impl PinnedSource {
    pub fn no_journals(&self) -> Result<()> {
        for suffix in ["-wal", "-shm", "-journal"] {
            let mut n = self.basename.clone();
            n.push(suffix);
            let c = name(&n)?;
            let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
            let rc = unsafe {
                libc::fstatat(
                    self.dir.as_raw_fd(),
                    c.as_ptr(),
                    stat.as_mut_ptr(),
                    libc::AT_SYMLINK_NOFOLLOW,
                )
            };
            ensure!(
                rc != 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ENOENT),
                "offline backup has WAL/SHM/journal companion; create a standalone snapshot with SQLite backup API first, not cp of a live database"
            );
        }
        Ok(())
    }
}
pub(crate) fn pin(path: &Path, kind: &str, max: usize) -> Result<PinnedSource> {
    let dir = parent(path)?;
    let basename = path
        .file_name()
        .ok_or_else(|| anyhow!("source filename required"))?;
    let mut file = openat(&dir, basename, libc::O_RDONLY | libc::O_NONBLOCK)?;
    let before = file.metadata()?;
    let uid = unsafe { libc::geteuid() };
    ensure!(
        before.is_file() && before.nlink() == 1 && (before.uid() == uid || before.uid() == 0),
        "source must be a regular single-link operator/root-owned offline file"
    );
    ensure!(
        before.mode() & 0o177 == 0,
        "source permissions must be private, without group/other access or execute bits; migration will not chmod sources"
    );
    ensure!(
        before.len() <= max as u64,
        "source exceeds offline migration byte bound"
    );
    let mut bytes = Vec::with_capacity(before.len() as usize);
    std::io::Read::by_ref(&mut file)
        .take(max as u64 + 1)
        .read_to_end(&mut bytes)?;
    let after = file.metadata()?;
    ensure!(
        bytes.len() <= max
            && before.len() == after.len()
            && before.mtime() == after.mtime()
            && before.mtime_nsec() == after.mtime_nsec(),
        "source changed during read"
    );
    let source = SourceDigest {
        kind: kind.into(),
        path: path.to_owned(),
        sha256: digest(&bytes),
        bytes: bytes.len() as u64,
    };
    Ok(PinnedSource {
        file,
        dir,
        basename: basename.to_owned(),
        bytes,
        source,
    })
}
pub(crate) fn read(path: &Path, kind: &str, max: usize) -> Result<(Vec<u8>, SourceDigest)> {
    let p = pin(path, kind, max)?;
    Ok((p.bytes, p.source))
}
pub(crate) fn destination(path: &Path) -> Result<()> {
    let dir = parent(path)?;
    let basename = name(
        path.file_name()
            .ok_or_else(|| anyhow!("destination filename required"))?,
    )?;
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    let rc = unsafe {
        libc::fstatat(
            dir.as_raw_fd(),
            basename.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    ensure!(
        rc != 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ENOENT),
        "destination already exists or unavailable; never merge or overwrite"
    );
    Ok(())
}
pub(crate) fn publish(stage: &Path, destination: &Path) -> Result<()> {
    destination_check(stage, destination)?;
    let dir = parent(destination)?;
    let src = name(
        stage
            .file_name()
            .ok_or_else(|| anyhow!("stage filename required"))?,
    )?;
    let dst = name(
        destination
            .file_name()
            .ok_or_else(|| anyhow!("destination filename required"))?,
    )?;
    // Atomic no-replace publication, including refusal of concurrent symlink/file.
    let rc = unsafe {
        libc::renameat2(
            dir.as_raw_fd(),
            src.as_ptr(),
            dir.as_raw_fd(),
            dst.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    ensure!(
        rc == 0,
        "atomic publication failed; destination not overwritten"
    );
    Ok(())
}
fn destination_check(stage: &Path, dst: &Path) -> Result<()> {
    ensure!(
        stage.parent() == dst.parent(),
        "staging must share destination parent"
    );
    destination(dst)
}
