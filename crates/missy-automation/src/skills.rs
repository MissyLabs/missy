use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::{
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
};

/// Untrusted documentation only. Discovery never executes code or imports instructions.
#[derive(Clone, Debug, Serialize)]
pub struct DiscoveredSkill {
    pub root: PathBuf,
    pub path: PathBuf,
    pub markdown: String,
    pub trusted: bool,
}

pub(crate) fn reject_symlink_components(path: &Path) -> Result<()> {
    if !path.is_absolute() {
        bail!("absolute canonical path required");
    }
    let mut current = PathBuf::new();
    for c in path.components() {
        if matches!(c, Component::ParentDir | Component::CurDir) {
            bail!("noncanonical path");
        }
        current.push(c.as_os_str());
        let m = fs::symlink_metadata(&current)?;
        if m.file_type().is_symlink() {
            bail!("symlink paths are refused");
        }
    }
    Ok(())
}

// Open each component relative to an open directory, refusing ancestor link swaps.
#[cfg(unix)]
fn open_regular_without_links(path: &Path) -> Result<fs::File> {
    use std::{
        ffi::CString,
        os::{
            fd::{AsRawFd, FromRawFd},
            unix::ffi::OsStrExt,
        },
    };
    let mut parent = fs::File::open("/")?;
    let components: Vec<_> = path.components().collect();
    if !path.is_absolute() {
        bail!("absolute skill path required");
    }
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(name) = component else {
            if matches!(component, Component::RootDir) {
                continue;
            }
            bail!("canonical skill path required");
        };
        let name = CString::new(name.as_bytes())?;
        let is_last = index + 1 == components.len();
        let flags = libc::O_RDONLY
            | libc::O_CLOEXEC
            | libc::O_NOFOLLOW
            | if is_last {
                libc::O_NONBLOCK
            } else {
                libc::O_DIRECTORY
            };
        let fd = unsafe { libc::openat(parent.as_raw_fd(), name.as_ptr(), flags) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        parent = unsafe { fs::File::from_raw_fd(fd) };
    }
    if !parent.metadata()?.is_file() {
        bail!("skill must be a regular file");
    }
    Ok(parent)
}

#[cfg(not(unix))]
fn open_regular_without_links(path: &Path) -> Result<fs::File> {
    let _ = path;
    bail!("secure no-symlink skill discovery unsupported on this platform")
}

/// Scan only explicitly supplied canonical roots. Limits apply across all roots:
/// 16 roots, 512 directories, 8192 entries, depth 8, 128 skills,
/// 64 KiB/file, 1 MiB total.
/// Concurrent mutation of trusted roots is not supported; roots should be read-only.
pub fn discover_skills(roots: &[PathBuf]) -> Result<Vec<DiscoveredSkill>> {
    if roots.len() > 16 {
        bail!("too many skill roots");
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut found = Vec::new();
    let mut dirs = 0;
    let mut total = 0;
    let mut total_entries = 0;
    for root in roots {
        reject_symlink_components(root)?;
        if fs::canonicalize(root)? != *root || !root.is_dir() || !seen.insert(root.clone()) {
            bail!("skill roots must be distinct canonical directories");
        }
        let mut pending = vec![(root.clone(), 0)];
        while let Some((dir, depth)) = pending.pop() {
            dirs += 1;
            if dirs > 512 {
                bail!("skill directory bound exceeded");
            }
            // Reject, rather than silently follow, links including non-SKILL entries.
            let mut entries = fs::read_dir(&dir)?
                .take(1025)
                .collect::<std::io::Result<Vec<_>>>()?;
            if entries.len() > 1024 {
                bail!("skill directory entry bound exceeded");
            }
            total_entries += entries.len();
            if total_entries > 8192 {
                bail!("skill total entry bound exceeded");
            }
            entries.sort_by_key(|e| e.file_name());
            for entry in entries {
                let ty = entry.file_type()?;
                if ty.is_symlink() {
                    bail!("symlink in skill root");
                }
                if ty.is_dir() {
                    if depth >= 8 {
                        bail!("skill depth bound exceeded");
                    }
                    if pending.len() + dirs >= 512 {
                        bail!("skill directory bound exceeded");
                    }
                    pending.push((entry.path(), depth + 1));
                } else if ty.is_file() && entry.file_name() == "SKILL.md" {
                    if found.len() >= 128 {
                        bail!("skill count bound exceeded");
                    }
                    let path = entry.path();
                    reject_symlink_components(&path)?;
                    let file = open_regular_without_links(&path)?;
                    let mut bytes = Vec::new();
                    file.take(65_537).read_to_end(&mut bytes)?;
                    total += bytes.len();
                    if bytes.len() > 65_536 || total > 1_048_576 {
                        bail!("skill byte bound exceeded");
                    }
                    found.push(DiscoveredSkill {
                        root: root.clone(),
                        path,
                        markdown: String::from_utf8(bytes).context("skill must be UTF-8")?,
                        trusted: false,
                    });
                }
            }
        }
    }
    found.sort_by(|a, b| a.path.cmp(&b.path));
    if found.windows(2).any(|pair| pair[0].path == pair[1].path) {
        bail!("overlapping skill roots produced duplicate paths");
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn discovery_is_data_only_bounded_and_refuses_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        fs::write(root.join("SKILL.md"), "# Skill\nExecute nothing.").unwrap();
        let found = discover_skills(std::slice::from_ref(&root)).unwrap();
        assert_eq!(found.len(), 1);
        assert!(!found[0].trusted);
        fs::write(root.join("SKILL.md"), vec![b'x'; 65_537]).unwrap();
        assert!(discover_skills(std::slice::from_ref(&root)).is_err());
        fs::remove_file(root.join("SKILL.md")).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("/etc/passwd", root.join("SKILL.md")).unwrap();
            assert!(discover_skills(&[root]).is_err());
        }
    }
}
