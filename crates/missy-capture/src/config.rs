use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::Read,
    os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
};

/// Operator-only pin. Never resolved through PATH. Executed through the verified
/// open descriptor, not a second pathname lookup after hash verification.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutablePin {
    pub path: PathBuf,
    pub sha256: String,
}
impl ExecutablePin {
    pub(crate) fn open(&self) -> Result<File> {
        canonical(&self.path)?;
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&self.path)?;
        let meta = file.metadata()?;
        ensure!(
            meta.is_file()
                && meta.mode() & 0o111 != 0
                && meta.mode() & 0o022 == 0
                && meta.len() <= 128 * 1024 * 1024,
            "unsafe executable"
        );
        let expected = hex::decode(&self.sha256)?;
        ensure!(expected.len() == 32, "invalid executable SHA-256");
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        ensure!(
            hash.finalize().as_slice() == expected,
            "executable pin mismatch"
        );
        Ok(file)
    }
}

/// A session created/owned by the trusted operator, never an inherited DISPLAY.
/// `isolated_session` is an operator attestation, NOT model/user tool arguments.
/// The application must not configure an existing interactive desktop here.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustedX11 {
    pub display: String,
    pub xauthority: PathBuf,
    pub isolated_session: bool,
}
impl TrustedX11 {
    pub(crate) fn number(&self) -> Result<u16> {
        ensure!(
            self.isolated_session,
            "X11 session must be explicitly trusted and isolated"
        );
        let digits = self.display.strip_prefix(':').unwrap_or("");
        ensure!(
            !digits.is_empty() && digits.len() <= 5 && digits.bytes().all(|b| b.is_ascii_digit()),
            "only local :NUMBER displays are accepted"
        );
        let n: u16 = digits.parse()?;
        ensure!(
            n >= 100 && self.display == format!(":{n}"),
            "isolated display number must be canonical and >=100"
        );
        Ok(n)
    }
    pub(crate) fn socket(&self) -> Result<PathBuf> {
        Ok(PathBuf::from(format!("/tmp/.X11-unix/X{}", self.number()?)))
    }
    pub(crate) fn cookie(&self) -> Result<Vec<u8>> {
        canonical(&self.xauthority)?;
        let f = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&self.xauthority)?;
        let m = f.metadata()?;
        ensure!(
            m.is_file()
                && m.mode() & 0o077 == 0
                && m.uid() == unsafe { libc::geteuid() }
                && (1..=16384).contains(&m.len()),
            "Xauthority must be private, owned, bounded regular file"
        );
        let mut data = Vec::new();
        f.take(16385).read_to_end(&mut data)?;
        ensure!(data.len() <= 16384, "Xauthority grew");
        Ok(data)
    }
    pub(crate) fn validate_live(&self) -> Result<()> {
        let m = std::fs::symlink_metadata(self.socket()?)?;
        ensure!(
            m.file_type().is_socket() && m.uid() == unsafe { libc::geteuid() },
            "X11 socket must be an owned Unix socket"
        );
        self.cookie()?;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureConfig {
    #[serde(default)]
    pub enable_camera: bool,
    #[serde(default)]
    pub enable_screencast: bool,
    #[serde(default)]
    pub cameras: Vec<PathBuf>,
    #[serde(default)]
    pub x11: Option<TrustedX11>,
    pub ffmpeg: ExecutablePin,
    pub ffprobe: ExecutablePin,
    pub staging_root: PathBuf,
    /// Explicit canonical runtime library/binary trees below /usr. May include
    /// private runtime trees outside /usr only when strictly owned and mode0700.
    pub runtime_read_roots: Vec<PathBuf>,
    #[serde(default = "timeout")]
    pub timeout_ms: u64,
    #[serde(default = "budget")]
    pub max_output_bytes: usize,
    #[serde(default = "pixels")]
    pub max_pixels: u32,
    #[serde(default = "fps")]
    pub max_fps: u32,
}
fn timeout() -> u64 {
    120_000
}
fn budget() -> usize {
    32 * 1024 * 1024
}
fn pixels() -> u32 {
    1920 * 1080
}
fn fps() -> u32 {
    30
}
impl CaptureConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        self.ffmpeg.open()?;
        self.ffprobe.open()?;
        ensure!(
            (1..=180_000).contains(&self.timeout_ms),
            "timeout outside 1..180000ms"
        );
        ensure!(
            (1024..=64 * 1024 * 1024).contains(&self.max_output_bytes),
            "output budget outside hard bounds"
        );
        ensure!(
            (1..=1920 * 1080).contains(&self.max_pixels) && (1..=30).contains(&self.max_fps),
            "pixel/fps budget invalid"
        );
        private_directory(&self.staging_root)?;
        ensure!(
            !self.runtime_read_roots.is_empty() && self.runtime_read_roots.len() <= 16,
            "explicit bounded runtime read roots required"
        );
        for root in &self.runtime_read_roots {
            canonical(root)?;
            ensure!(root.is_dir(), "runtime root is not directory");
            if !(root.starts_with("/usr/lib")
                || root.starts_with("/usr/bin")
                || root.starts_with("/usr/local/lib")
                || root.starts_with("/usr/local/bin"))
            {
                private_directory(root)?;
            }
            ensure!(
                !self.staging_root.starts_with(root) && !root.starts_with(&self.staging_root),
                "staging and runtime roots must be disjoint"
            );
        }
        ensure!(self.cameras.len() <= 16, "too many camera devices");
        for path in &self.cameras {
            camera_path(path)?;
        }
        if self.enable_camera {
            ensure!(
                !self.cameras.is_empty(),
                "camera enable requires exact device allowlist"
            );
        }
        if let Some(x) = &self.x11 {
            x.number()?;
            canonical(&x.xauthority)?;
        }
        ensure!(
            !self.enable_screencast || self.x11.is_some(),
            "screencast requires explicit isolated X11 config"
        );
        ensure!(
            missy_sandbox::landlock_abi()? >= 5,
            "capture requires Landlock ABI>=5 (device IOCTL isolation), no fallback"
        );
        Ok(())
    }
}
pub(crate) fn safe_path(path: &Path) -> Result<()> {
    use std::os::unix::ffi::OsStrExt;
    ensure!(
        path.is_absolute()
            && path.as_os_str().len() <= 4096
            && path
                .as_os_str()
                .as_bytes()
                .iter()
                .all(|b| *b >= 32 && *b != 127)
            && !path
                .as_os_str()
                .as_bytes()
                .split(|b| *b == b'/')
                .any(|p| p == b"." || p == b".."),
        "unsafe absolute path"
    );
    Ok(())
}
pub(crate) fn canonical(path: &Path) -> Result<()> {
    safe_path(path)?;
    ensure!(
        path.canonicalize()? == path,
        "symlink or noncanonical path refused"
    );
    // canonicalize alone would accept intermediate symlinks resolving back to
    // the same tree. Check each component too.
    let mut current = PathBuf::from("/");
    for c in path.components().skip(1) {
        current.push(c);
        ensure!(
            !std::fs::symlink_metadata(&current)?
                .file_type()
                .is_symlink(),
            "symlink component refused"
        );
    }
    Ok(())
}
pub(crate) fn private_directory(path: &Path) -> Result<()> {
    canonical(path)?;
    let m = std::fs::metadata(path)?;
    ensure!(
        m.is_dir() && m.mode() & 0o077 == 0 && m.uid() == unsafe { libc::geteuid() },
        "directory must be private and owned"
    );
    Ok(())
}
pub(crate) fn camera_path(path: &Path) -> Result<()> {
    safe_path(path)?;
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let digits = name.strip_prefix("video").unwrap_or("");
    ensure!(
        path.parent() == Some(Path::new("/dev"))
            && !digits.is_empty()
            && digits.len() <= 5
            && digits.bytes().all(|b| b.is_ascii_digit()),
        "only exact /dev/videoNUMBER camera paths allowed"
    );
    Ok(())
}
pub(crate) fn camera_live(path: &Path) -> Result<()> {
    camera_path(path)?;
    canonical(path)?;
    let m = std::fs::symlink_metadata(path)?;
    ensure!(
        m.file_type().is_char_device() && libc::major(m.rdev()) == 81,
        "not a Linux V4L2 character device"
    );
    Ok(())
}
