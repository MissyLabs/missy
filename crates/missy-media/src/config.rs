use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, os::unix::fs::MetadataExt, path::PathBuf};

/// Operator configuration, never model-supplied arguments. No PATH search or
/// automatic executable discovery. Both executables must be explicitly pinned.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutablePin {
    pub path: PathBuf,
    pub sha256: String,
}
impl ExecutablePin {
    pub(crate) fn verify(&self) -> Result<()> {
        ensure!(self.path.is_absolute(), "executable must be absolute");
        ensure!(
            self.path.canonicalize()? == self.path,
            "executable path must be canonical"
        );
        let expected = hex::decode(&self.sha256).context("invalid executable SHA-256")?;
        ensure!(expected.len() == 32, "SHA-256 must contain 32 bytes");
        let mut file = File::open(&self.path)?;
        let stat = file.metadata()?;
        ensure!(
            stat.is_file() && stat.mode() & 0o022 == 0 && stat.mode() & 0o111 != 0,
            "executable must be regular, executable, and not group/world writable"
        );
        ensure!(stat.len() <= 128 * 1024 * 1024, "executable too large");
        let mut hash = Sha256::new();
        let mut buf = [0u8; 65536];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hash.update(&buf[..n]);
        }
        ensure!(
            hash.finalize().as_slice() == expected,
            "executable hash mismatch"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediaConfig {
    pub ffmpeg: ExecutablePin,
    pub ffprobe: ExecutablePin,
    /// Canonical directory containing private per-call temporary directories.
    pub staging_root: PathBuf,
    /// Explicit canonical runtime directories for binaries/shared libraries.
    /// These must NOT contain user documents/secrets. Normally /usr/bin and
    /// /usr/lib suffice. No implicit /home, /etc, /proc, /dev, or network grants.
    pub runtime_read_roots: Vec<PathBuf>,
    pub timeout_ms: u64,
    pub max_duration_seconds: u32,
    pub max_pixels: u32,
    pub max_output_bytes: usize,
}
impl MediaConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        self.ffmpeg.verify()?;
        self.ffprobe.verify()?;
        ensure!(
            (1..=300_000).contains(&self.timeout_ms),
            "timeout outside 1..300000 ms"
        );
        ensure!(
            (1..=600).contains(&self.max_duration_seconds),
            "duration outside 1..600 seconds"
        );
        ensure!(
            (1..=16_777_216).contains(&self.max_pixels),
            "pixel limit outside hard bound"
        );
        ensure!(
            (1024..=64 * 1024 * 1024).contains(&self.max_output_bytes),
            "output size outside hard bound"
        );
        ensure!(
            self.staging_root.is_absolute()
                && self.staging_root.canonicalize()? == self.staging_root,
            "staging root must be canonical absolute directory"
        );
        ensure!(self.staging_root.is_dir(), "staging root unavailable");
        ensure!(
            !self.runtime_read_roots.is_empty() && self.runtime_read_roots.len() <= 8,
            "explicit bounded runtime read roots required"
        );
        for root in &self.runtime_read_roots {
            ensure!(
                root.is_absolute() && root.canonicalize()? == *root && root.is_dir(),
                "runtime read roots must be canonical absolute directories"
            );
            // Deliberately narrow: this is not a general sandbox permission API.
            ensure!(
                root.starts_with("/usr") && root != std::path::Path::new("/usr"),
                "runtime roots must be below /usr, never an entire filesystem"
            );
        }
        for executable in [&self.ffmpeg, &self.ffprobe] {
            ensure!(
                self.runtime_read_roots
                    .iter()
                    .any(|r| executable.path.starts_with(r)),
                "pinned executable outside runtime read roots"
            );
        }
        Ok(())
    }
}
