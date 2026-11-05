//! Linux UAPI: Documentation/userspace-api/landlock.rst and linux/landlock.h.
//! Raw syscalls keep child-only placement auditable without an extra dependency.
//! ABI 1: fs bits 0..12; ABI 2: REFER (13); ABI 3: TRUNCATE (14);
//! ABI 4: TCP bind/connect and second ruleset u64; ABI 5: IOCTL_DEV (15).
//! Later ABIs are treated as ABI 5; no required feature silently degrades.
use crate::{SandboxConfig, validate_path};
use anyhow::{Context, Result, bail};
use std::{
    ffi::{CString, OsStr},
    fs::File,
    io,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::ffi::OsStrExt,
    },
    path::{Component, Path},
};

const READ: u64 = (1 << 0) | (1 << 2) | (1 << 3);
// No device node creation or IOCTL_DEV permission, even in writable trees.
const WRITE: u64 = (1 << 1)
    | (1 << 4)
    | (1 << 5)
    | (1 << 7)
    | (1 << 8)
    | (1 << 9)
    | (1 << 10)
    | (1 << 12)
    | (1 << 13)
    | (1 << 14);

pub fn landlock_abi() -> Result<u32> {
    // SAFETY: query uses no pointers and changes no confinement state.
    let abi = unsafe {
        libc::syscall(
            libc::SYS_landlock_create_ruleset,
            std::ptr::null::<u8>(),
            0usize,
            1u32,
        )
    };
    if abi < 0 {
        return Err(io::Error::last_os_error())
            .context("Landlock unavailable; refusing unsandboxed fallback");
    }
    Ok(abi as u32)
}
pub(crate) fn require_abi(abi: u32, deny_tcp: bool) -> Result<()> {
    if abi < 3 {
        bail!("Landlock ABI >= 3 required for filesystem truncate protection (found {abi})");
    }
    if deny_tcp && abi < 4 {
        bail!("Landlock ABI >= 4 required for TCP denial (found {abi})");
    }
    Ok(())
}
#[repr(C)]
struct RulesetAttr {
    handled_access_fs: u64,
    handled_access_net: u64,
}
// UAPI struct is explicitly packed: sizeof == 12, not Rust's default 16.
#[repr(C, packed)]
struct PathBeneath {
    allowed_access: u64,
    parent_fd: i32,
}

/// Walk all components O_NOFOLLOW. Canonicalizing first would accept symlink roots.
pub(crate) fn open_root(path: &Path) -> Result<File> {
    validate_path(path)?;
    fn open(fd: i32, part: &OsStr) -> Result<File> {
        let part = CString::new(part.as_bytes())?;
        let raw = unsafe {
            libc::openat(
                fd,
                part.as_ptr(),
                libc::O_PATH | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if raw < 0 {
            return Err(io::Error::last_os_error())
                .context("sandbox root unavailable or symlink denied");
        }
        Ok(unsafe { File::from_raw_fd(raw) })
    }
    let mut dir = open(libc::AT_FDCWD, OsStr::new("/"))?;
    for part in path.components() {
        if let Component::Normal(part) = part {
            dir = open(dir.as_raw_fd(), part)?;
        }
    }
    Ok(dir)
}
pub(crate) struct Prepared {
    fd: File,
    pub(crate) abi: u32,
}
impl Prepared {
    /// All allocation/path opens/add_rule happen before fork; none restrict parent.
    pub(crate) fn new(config: &SandboxConfig) -> Result<Self> {
        let abi = landlock_abi()?;
        require_abi(abi, config.deny_tcp)?;
        let attr = RulesetAttr {
            handled_access_fs: (1 << if abi >= 5 { 16 } else { 15 }) - 1,
            handled_access_net: if config.deny_tcp { 3 } else { 0 },
        };
        let size = if abi >= 4 { 16 } else { 8 };
        let raw = unsafe {
            libc::syscall(
                libc::SYS_landlock_create_ruleset,
                &attr as *const RulesetAttr,
                size,
                0u32,
            )
        };
        if raw < 0 {
            return Err(io::Error::last_os_error()).context("Landlock ruleset creation failed");
        }
        let fd = unsafe { File::from_raw_fd(raw as i32) };
        for (roots, access) in [(&config.read_roots, READ), (&config.write_roots, WRITE)] {
            for path in roots {
                let root = open_root(path)?;
                let rule = PathBeneath {
                    allowed_access: access,
                    parent_fd: root.as_raw_fd(),
                };
                if unsafe {
                    libc::syscall(
                        libc::SYS_landlock_add_rule,
                        fd.as_raw_fd(),
                        1u32,
                        &rule as *const PathBeneath,
                        0u32,
                    )
                } < 0
                {
                    return Err(io::Error::last_os_error())
                        .context("Landlock path rule creation failed");
                }
                // Kernel pins rule inode; descriptor closes in unrestricted parent.
            }
        }
        Ok(Self { fd, abi })
    }
    /// # Safety
    /// ONLY from Command::pre_exec in dedicated forked child. Never call in parent.
    /// No locks, allocation, formatting or path lookup. Every failure aborts exec.
    pub(crate) unsafe fn restrict_child(&self) -> io::Result<()> {
        if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } < 0 {
            return Err(io::Error::last_os_error());
        }
        if unsafe { libc::syscall(libc::SYS_landlock_restrict_self, self.fd.as_raw_fd(), 0u32) } < 0
        {
            return Err(io::Error::last_os_error());
        }
        // Preserve Rust's exec-error pipe until exec. ALL non-stdio descriptors
        // become CLOEXEC including third-party inherited descriptors and ruleset.
        if unsafe { libc::syscall(libc::SYS_close_range, 3u32, u32::MAX, 4u32) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn rejects_unsupported_required_features() {
        assert!(super::require_abi(2, false).is_err());
        assert!(super::require_abi(3, true).is_err());
        assert!(super::require_abi(3, false).is_ok());
        assert!(super::require_abi(4, true).is_ok());
        assert_eq!(std::mem::size_of::<super::PathBeneath>(), 12);
    }
}
