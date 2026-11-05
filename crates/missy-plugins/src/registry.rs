use crate::{PluginConfig, PluginManifest, validate_id};
use anyhow::{Result, ensure};
use missy_core::Scope;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt},
    },
    path::{Path, PathBuf},
};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginProposal {
    pub id: String,
    pub hash: String,
    pub scope: Scope,
    pub manifest: PluginManifest,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    proposal: PluginProposal,
    reviewed: bool,
    enabled: bool,
}
/// Owner-only control-plane registry. Review/activation are not model tools.
pub struct ToolRegistry {
    dir: File,
    path: PathBuf,
}
impl ToolRegistry {
    fn lock(&self) -> Result<File> {
        // A fresh open-file description is essential: flock on a duplicate of
        // self.dir would not serialize concurrent calls on the same registry.
        let lock = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&self.path)?;
        let expected = self.dir.metadata()?;
        let actual = lock.metadata()?;
        ensure!(
            expected.dev() == actual.dev() && expected.ino() == actual.ino(),
            "registry directory moved"
        );
        ensure!(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } == 0,
            "registry lock failed"
        );
        Ok(lock)
    }
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        ensure!(path.is_absolute(), "registry path must be absolute");
        crate::sandbox::validate_root(
            path.parent()
                .ok_or_else(|| anyhow::anyhow!("registry parent required"))?,
        )?;
        match std::fs::DirBuilder::new().mode(0o700).create(path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e.into()),
        }
        let dir = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)?;
        let m = dir.metadata()?;
        ensure!(
            m.uid() == unsafe { libc::geteuid() } && m.mode() & 0o077 == 0,
            "registry directory must be private 0700 and owned by current uid"
        );
        Ok(Self {
            dir,
            path: path.to_owned(),
        })
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    fn filename(scope: &Scope, id: &str) -> Result<String> {
        scope.validate()?;
        validate_id(id)?;
        Ok(format!(
            "{}-{id}.json",
            hex::encode(Sha256::digest(serde_json::to_vec(scope)?))
        ))
    }
    fn load(&self, scope: &Scope, id: &str) -> Result<Record> {
        let name = std::ffi::CString::new(Self::filename(scope, id)?)?;
        let fd = unsafe {
            libc::openat(
                self.dir.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK,
            )
        };
        ensure!(fd >= 0, "plugin record unavailable");
        let file = unsafe { File::from_raw_fd(fd) };
        let m = file.metadata()?;
        ensure!(
            m.is_file()
                && m.uid() == unsafe { libc::geteuid() }
                && m.mode() & 0o077 == 0
                && m.nlink() == 1
                && m.len() <= 131072,
            "plugin record must be bounded private regular file"
        );
        let mut bytes = Vec::new();
        file.take(131073).read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= 131072, "plugin record too large");
        let r: Record = serde_json::from_slice(&bytes)?;
        ensure!(
            r.proposal.scope == *scope && r.proposal.id == id,
            "registry principal/id mismatch"
        );
        r.proposal.manifest.validate()?;
        ensure!(
            r.proposal.hash == proposal_hash(scope, &r.proposal.manifest)?,
            "plugin manifest drift"
        );
        Ok(r)
    }
    fn save(&self, record: &Record, new: bool) -> Result<()> {
        let bytes = serde_json::to_vec(record)?;
        ensure!(bytes.len() <= 131072, "plugin proposal too large");
        let mut tmp = tempfile::NamedTempFile::new_in(&self.path)?;
        tmp.as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o600))?;
        tmp.write_all(&bytes)?;
        tmp.as_file().sync_all()?;
        let target = self
            .path
            .join(Self::filename(&record.proposal.scope, &record.proposal.id)?);
        if new {
            tmp.persist_noclobber(target)?;
        } else {
            tmp.persist(target)?;
        }
        self.dir.sync_all()?;
        Ok(())
    }
    pub fn propose(&self, scope: &Scope, manifest: PluginManifest) -> Result<PluginProposal> {
        let _lock = self.lock()?;
        scope.validate()?;
        manifest.validate()?;
        let proposal = PluginProposal {
            id: manifest.id.clone(),
            hash: proposal_hash(scope, &manifest)?,
            scope: scope.clone(),
            manifest,
        };
        self.save(
            &Record {
                proposal: proposal.clone(),
                reviewed: false,
                enabled: false,
            },
            true,
        )?;
        Ok(proposal)
    }
    pub fn get(&self, scope: &Scope, id: &str) -> Result<PluginProposal> {
        let _lock = self.lock()?;
        Ok(self.load(scope, id)?.proposal)
    }
    pub fn review(&self, scope: &Scope, id: &str, exact_hash: &str, approve: bool) -> Result<()> {
        let _lock = self.lock()?;
        let mut r = self.load(scope, id)?;
        ensure!(
            exact_hash == r.proposal.hash,
            "exact proposal hash approval mismatch"
        );
        r.reviewed = approve;
        r.enabled = false;
        self.save(&r, false)
    }
    pub fn enable(
        &self,
        scope: &Scope,
        id: &str,
        exact_hash: &str,
        config: &PluginConfig,
    ) -> Result<()> {
        let _lock = self.lock()?;
        config.validate()?;
        ensure!(
            config.enabled && config.scope == *scope && config.allowed_ids.iter().any(|v| v == id),
            "plugin disabled or scope/id not operator allowed"
        );
        let mut r = self.load(scope, id)?;
        ensure!(
            r.reviewed && r.proposal.hash == exact_hash,
            "exact-hash human approval required"
        );
        r.enabled = true;
        self.save(&r, false)
    }
    pub(crate) fn enabled(&self, config: &PluginConfig) -> Result<Vec<PluginProposal>> {
        let _lock = self.lock()?;
        let mut result = Vec::new();
        for id in &config.allowed_ids {
            let r = self.load(&config.scope, id)?;
            ensure!(
                r.enabled && r.reviewed,
                "configured plugin not approved and enabled"
            );
            result.push(r.proposal);
        }
        Ok(result)
    }
}
fn proposal_hash(scope: &Scope, manifest: &PluginManifest) -> Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(&(
        "missy-plugin-proposal-v1",
        scope,
        manifest,
    ))?)))
}
