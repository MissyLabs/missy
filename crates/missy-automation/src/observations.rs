//! Trusted native collectors. Operator paths are bounded by explicit roots;
//! no model tool exposes these collectors or expands their authorization.
use crate::{TriggerConfig, TriggerKind, TriggerObservation, execution::fingerprint};
use anyhow::{Result, ensure};
use chrono::{DateTime, Utc};
use missy_core::Scope;
use missy_store::{Store, TurnStatus};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case", deny_unknown_fields)]
pub enum ObservationSource {
    /// Read exactly one bounded regular file, never a directory recursion.
    File {
        path: PathBuf,
        allowed_root: PathBuf,
        max_bytes: usize,
    },
    Disk {
        path: PathBuf,
        allowed_root: PathBuf,
    },
    Load,
    /// Activity means durable turns in this exact scope, not desktop events.
    Idle,
}
fn authorized_path(path: &Path, root: &Path) -> Result<PathBuf> {
    ensure!(
        path.is_absolute() && root.is_absolute(),
        "collector paths must be absolute"
    );
    crate::skills::reject_symlink_components(root)?;
    crate::skills::reject_symlink_components(path)?;
    let root = fs::canonicalize(root)?;
    let actual = fs::canonicalize(path)?;
    ensure!(
        actual == path && actual.starts_with(&root),
        "collector path escapes explicit canonical root"
    );
    Ok(actual)
}
#[cfg(unix)]
fn open_nofollow(path: &Path) -> Result<fs::File> {
    use std::os::{
        fd::{AsRawFd, FromRawFd},
        unix::{ffi::OsStrExt, fs::OpenOptionsExt},
    };
    let mut current = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC)
        .open("/")?;
    let components: Vec<_> = path
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => Some(s),
            _ => None,
        })
        .collect();
    for (index, component) in components.iter().enumerate() {
        let name = std::ffi::CString::new(component.as_bytes())?;
        let flags = libc::O_RDONLY
            | libc::O_NOFOLLOW
            | libc::O_NONBLOCK
            | libc::O_CLOEXEC
            | if index + 1 < components.len() {
                libc::O_DIRECTORY
            } else {
                0
            };
        let fd = unsafe { libc::openat(current.as_raw_fd(), name.as_ptr(), flags) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        current = unsafe { fs::File::from_raw_fd(fd) };
    }
    Ok(current)
}
pub fn observe(
    source: &ObservationSource,
    config: &TriggerConfig,
    store: &Store,
    scope: &Scope,
    now: DateTime<Utc>,
) -> Result<TriggerObservation> {
    config.validate()?;
    scope.validate()?;
    let mut o = TriggerObservation::default();
    match (source, &config.kind) {
        (
            ObservationSource::File {
                path,
                allowed_root,
                max_bytes,
            },
            TriggerKind::FileChange,
        ) => {
            ensure!(
                (1..=1_048_576).contains(max_bytes),
                "file collector bound must be 1..1048576"
            );
            let path = authorized_path(path, allowed_root)?;
            #[cfg(unix)]
            let file = open_nofollow(&path)?;
            #[cfg(not(unix))]
            let file = fs::File::open(&path)?;
            let m = file.metadata()?;
            ensure!(
                m.is_file() && m.len() <= *max_bytes as u64,
                "watched file must be bounded regular file"
            );
            let mut data = Vec::new();
            file.take(*max_bytes as u64 + 1).read_to_end(&mut data)?;
            ensure!(data.len() <= *max_bytes, "watched file grew beyond bound");
            o.file_revision = Some(fingerprint(&data)?);
        }
        (ObservationSource::Disk { path, allowed_root }, TriggerKind::DiskAbove { .. }) => {
            let path = authorized_path(path, allowed_root)?;
            #[cfg(unix)]
            {
                use std::os::fd::AsRawFd;
                let file = open_nofollow(&path)?;
                let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
                ensure!(
                    unsafe { libc::fstatvfs(file.as_raw_fd(), stat.as_mut_ptr()) } == 0,
                    "disk usage observation failed"
                );
                let stat = unsafe { stat.assume_init() };
                ensure!(stat.f_blocks > 0, "disk has no measurable blocks");
                o.disk_percent = Some(
                    (stat.f_blocks.saturating_sub(stat.f_bavail)) as f64 / stat.f_blocks as f64
                        * 100.0,
                );
            }
            #[cfg(not(unix))]
            anyhow::bail!("native disk collector unsupported on this platform");
        }
        (ObservationSource::Load, TriggerKind::LoadAbove { .. }) => {
            #[cfg(unix)]
            {
                let mut load = 0.0;
                ensure!(
                    unsafe { libc::getloadavg(&mut load, 1) } == 1,
                    "load observation unavailable"
                );
                let cores = std::thread::available_parallelism()?.get();
                ensure!(load.is_finite() && load >= 0.0, "invalid load observation");
                o.normalized_load = Some(load / cores as f64);
            }
            #[cfg(not(unix))]
            anyhow::bail!("native load collector unsupported on this platform");
        }
        (ObservationSource::Idle, TriggerKind::Idle { .. }) => {
            let turns = store.list_turns(scope, 1000)?;
            if turns.iter().any(|t| {
                matches!(
                    t.status,
                    TurnStatus::Admitted | TurnStatus::Running | TurnStatus::Interrupted
                )
            }) {
                o.idle_seconds = Some(0);
            } else if let Some(updated) = turns.iter().map(|t| t.updated_at.max(t.created_at)).max()
            {
                o.idle_seconds =
                    Some((now.timestamp_millis().saturating_sub(updated).max(0) as u64) / 1000);
            }
            // No history is not evidence of an idle user. Do not spend quota.
        }
        _ => anyhow::bail!("observation source does not match trigger kind"),
    }
    Ok(o)
}
/// Validate source configuration without collecting or reading file contents.
pub fn validate_source(source: &ObservationSource, config: &TriggerConfig) -> Result<()> {
    config.validate()?;
    match (source, &config.kind) {
        (
            ObservationSource::File {
                path,
                allowed_root,
                max_bytes,
            },
            TriggerKind::FileChange,
        ) => {
            ensure!(
                (1..=1_048_576).contains(max_bytes),
                "invalid file observation limit"
            );
            authorized_path(path, allowed_root)?;
        }
        (ObservationSource::Disk { path, allowed_root }, TriggerKind::DiskAbove { .. }) => {
            authorized_path(path, allowed_root)?;
        }
        (ObservationSource::Load, TriggerKind::LoadAbove { .. })
        | (ObservationSource::Idle, TriggerKind::Idle { .. }) => (),
        _ => anyhow::bail!("observation source does not match trigger kind"),
    }
    Ok(())
}
