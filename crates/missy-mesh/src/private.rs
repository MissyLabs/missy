//! Descriptor-relative private persistence. No symlink traversal, implicit key
//! regeneration, permissive file repair, or writes to legacy/live state.
use anyhow::{Context, Result, bail, ensure};
use rand::RngCore;
use std::{
    ffi::CString,
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::{MetadataExt, OpenOptionsExt},
    },
    path::{Component, Path},
};

pub(crate) const STATE_LIMIT: usize = 4 * 1024 * 1024;

pub(crate) struct PrivateDir(pub File);
impl PrivateDir {
    pub fn open(path: &Path, create: bool) -> Result<Self> {
        ensure!(path.is_absolute(), "private path must be absolute");
        let mut dir = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open("/")?;
        let components: Vec<_> = path.components().collect();
        ensure!(components.len() > 1, "root is not private storage");
        for (i, part) in components.iter().enumerate().skip(1) {
            let Component::Normal(name) = part else {
                bail!("invalid private path component")
            };
            let name = CString::new(name.as_encoded_bytes())?;
            let final_part = i == components.len() - 1;
            if final_part && create {
                let rc = unsafe { libc::mkdirat(dir.as_raw_fd(), name.as_ptr(), 0o700) };
                if rc != 0
                    && std::io::Error::last_os_error().kind() != std::io::ErrorKind::AlreadyExists
                {
                    return Err(std::io::Error::last_os_error().into());
                }
            }
            let fd = unsafe {
                libc::openat(
                    dir.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                )
            };
            ensure!(fd >= 0, "private directory unavailable or symlinked");
            dir = unsafe { File::from_raw_fd(fd) };
        }
        check(&dir, true)?;
        Ok(Self(dir))
    }
    fn open_file(&self, name: &str, flags: i32) -> Result<File> {
        ensure!(
            !name.is_empty() && !name.contains('/') && name != "." && name != "..",
            "invalid storage name"
        );
        let name = CString::new(name)?;
        let fd = unsafe {
            libc::openat(
                self.0.as_raw_fd(),
                name.as_ptr(),
                flags | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK,
                0o600,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let file = unsafe { File::from_raw_fd(fd) };
        check(&file, false)?;
        Ok(file)
    }
    pub fn read(&self, name: &str, limit: usize) -> Result<Option<Vec<u8>>> {
        let file = match self.open_file(name, libc::O_RDONLY) {
            Ok(f) => f,
            Err(e)
                if e.downcast_ref::<std::io::Error>()
                    .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound) =>
            {
                return Ok(None);
            }
            Err(e) => return Err(e),
        };
        let mut out = Vec::new();
        file.take(limit as u64 + 1).read_to_end(&mut out)?;
        ensure!(out.len() <= limit, "private file exceeds size limit");
        Ok(Some(out))
    }
    pub fn create(&self, name: &str, bytes: &[u8]) -> Result<()> {
        let mut file = self.open_file(name, libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        self.0.sync_all()?;
        Ok(())
    }
    pub fn lock(&self) -> Result<File> {
        let file = self.open_file("mesh.lock", libc::O_RDWR | libc::O_CREAT)?;
        ensure!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "mesh state already in use"
        );
        Ok(file)
    }
    pub fn replace(&self, name: &str, bytes: &[u8]) -> Result<()> {
        ensure!(bytes.len() <= STATE_LIMIT, "mesh state capacity exhausted");
        // Validate the existing entry as well. Rename must never hide unsafe state.
        self.read(name, STATE_LIMIT)?;
        let mut random = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut random);
        let temp = format!("mesh-{}.tmp", hex::encode(random));
        self.create(&temp, bytes)?;
        let old = CString::new(temp)?;
        let new = CString::new(name)?;
        let rc = unsafe {
            libc::renameat(
                self.0.as_raw_fd(),
                old.as_ptr(),
                self.0.as_raw_fd(),
                new.as_ptr(),
            )
        };
        if rc != 0 {
            unsafe {
                libc::unlinkat(self.0.as_raw_fd(), old.as_ptr(), 0);
            }
            return Err(std::io::Error::last_os_error().into());
        }
        self.0.sync_all().context("mesh state directory sync")
    }
}
fn check(file: &File, directory: bool) -> Result<()> {
    let m = file.metadata()?;
    ensure!(
        if directory {
            m.is_dir()
        } else {
            m.is_file() && m.nlink() == 1
        },
        "private entry has unsafe type or hard links"
    );
    ensure!(
        m.uid() == unsafe { libc::geteuid() } && m.mode() & 0o077 == 0,
        "private entry must be owned by current uid with no group/other permissions"
    );
    Ok(())
}

pub(crate) fn split(path: &Path) -> Result<(PrivateDir, String)> {
    let parent = path.parent().context("explicit private parent required")?;
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .context("invalid private file name")?;
    Ok((PrivateDir::open(parent, false)?, name.into()))
}
