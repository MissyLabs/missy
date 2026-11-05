//! Private database path checks, matching the personal store's owned-path policy.
use anyhow::{Result, ensure};
use std::{
    fs,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Component, Path},
    sync::Mutex,
};
pub(crate) static OPEN_LOCK: Mutex<()> = Mutex::new(());
pub(crate) fn private_database(path: &Path, create: bool) -> Result<(u64, u64)> {
    ensure!(
        path.is_absolute(),
        "absolute history retention path required"
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
        m.uid() == unsafe { libc::geteuid() } && m.mode() & 0o777 == 0o700,
        "history database parent must be current-user owned and 0700"
    );
    match fs::symlink_metadata(path) {
        Ok(m) => validate_file(&m)?,
        Err(e) if create && e.kind() == std::io::ErrorKind::NotFound => {
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
            && m.mode() & 0o777 == 0o600,
        "history database/sidecar must be owned, 0600, regular and singly linked"
    );
    Ok(())
}
