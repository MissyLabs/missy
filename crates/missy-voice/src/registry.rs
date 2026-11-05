use anyhow::{Context, Result, bail, ensure};
use missy_core::Scope;
use pbkdf2::pbkdf2_hmac;
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    },
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use subtle::ConstantTimeEq;

const ITERATIONS: u32 = 100_000;
const MAX_REGISTRY: u64 = 1024 * 1024;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PolicyMode {
    Full,
    SafeChat,
    Muted,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Device {
    pub node_id: String,
    pub friendly_name: String,
    pub room: String,
    pub scope: Option<Scope>,
    pub mode: PolicyMode,
    pub paired: bool,
    pub requested_at: u64,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    device: Device,
    salt: String,
    token_hash: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryFile {
    version: u32,
    nodes: BTreeMap<String, Entry>,
}
impl Default for RegistryFile {
    fn default() -> Self {
        Self {
            version: 1,
            nodes: BTreeMap::new(),
        }
    }
}

/// Private JSON state with a separate flock, atomic 0600 writes and fresh reads
/// on every operation, including when the CLI changes mode in another process.
/// An existing insecure, corrupt, legacy or foreign-owned registry fails closed.
pub struct DeviceRegistry {
    path: PathBuf,
}
impl DeviceRegistry {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let registry = Self { path: path.into() };
        ensure!(
            registry.path.is_absolute(),
            "registry path must be absolute"
        );
        let parent = registry.path.parent().context("registry parent required")?;
        if !parent.exists() {
            // Only the final directory may be created; do not chmod caller directories.
            fs::DirBuilder::new().mode(0o700).create(parent)?;
        }
        private_metadata(parent, true)?;
        registry.read(|_| Ok(()))?;
        Ok(registry)
    }

    pub fn request_pairing(&self, friendly_name: &str, room: &str) -> Result<Device> {
        validate_name(friendly_name)?;
        ensure!(
            room.len() <= 128 && !room.chars().any(char::is_control),
            "invalid room"
        );
        self.mutate(|state| {
            let now = now();
            // Expired unapproved requests cannot exhaust persistent state.
            state
                .nodes
                .retain(|_, e| e.device.paired || now.saturating_sub(e.device.requested_at) < 3600);
            ensure!(state.nodes.len() < 256, "device registry full");
            let node = Device {
                node_id: uuid::Uuid::new_v4().to_string(),
                friendly_name: friendly_name.into(),
                room: room.into(),
                scope: None,
                mode: PolicyMode::Muted,
                paired: false,
                requested_at: now,
            };
            state.nodes.insert(
                node.node_id.clone(),
                Entry {
                    device: node.clone(),
                    salt: String::new(),
                    token_hash: String::new(),
                },
            );
            Ok(node)
        })
    }

    /// Operator-only. Token is returned exactly here, never persisted in plaintext
    /// or sent to an unauthenticated pairing socket. Deliver it out of band.
    pub fn approve(&self, node_id: &str, scope: Scope, mode: PolicyMode) -> Result<String> {
        scope.validate()?;
        ensure!(
            scope.channel == "voice" && scope.session == node_id,
            "voice scope must bind channel=voice and session=node_id"
        );
        self.mutate(|state| {
            let entry = state
                .nodes
                .get_mut(node_id)
                .context("unknown pairing request")?;
            ensure!(
                !entry.device.paired,
                "already paired; revoke and pair again to change scope"
            );
            ensure!(
                now().saturating_sub(entry.device.requested_at) < 3600,
                "pairing request expired"
            );
            let mut token = [0; 32];
            let mut salt = [0; 16];
            OsRng.fill_bytes(&mut token);
            OsRng.fill_bytes(&mut salt);
            let token = hex::encode(token);
            entry.salt = hex::encode(salt);
            entry.token_hash = hex::encode(derive(&token, &salt));
            entry.device.scope = Some(scope);
            entry.device.mode = mode;
            entry.device.paired = true;
            Ok(token)
        })
    }

    pub fn authenticate(&self, node_id: &str, token: &str) -> Result<Device> {
        ensure!(
            node_id.len() <= 128 && token.len() <= 256,
            "invalid credentials"
        );
        self.read(|state| {
            let entry = state.nodes.get(node_id);
            let salt = entry
                .map(|e| hex::decode(&e.salt))
                .transpose()?
                .unwrap_or(vec![0; 16]);
            let expected = entry
                .map(|e| hex::decode(&e.token_hash))
                .transpose()?
                .unwrap_or(vec![0; 32]);
            let actual = derive(token, &salt);
            ensure!(
                bool::from(actual.as_slice().ct_eq(expected.as_slice())),
                "invalid credentials"
            );
            let device = entry.context("invalid credentials")?.device.clone();
            validate_device(&device)?;
            ensure!(device.paired, "invalid credentials");
            Ok(device)
        })
    }

    pub fn get(&self, node_id: &str) -> Result<Device> {
        self.read(|state| {
            state
                .nodes
                .get(node_id)
                .map(|e| e.device.clone())
                .context("unknown device")
        })
    }
    pub fn list(&self) -> Result<Vec<Device>> {
        self.read(|state| Ok(state.nodes.values().map(|e| e.device.clone()).collect()))
    }
    pub fn set_mode(&self, node_id: &str, mode: PolicyMode) -> Result<()> {
        self.mutate(|state| {
            let e = state.nodes.get_mut(node_id).context("unknown device")?;
            ensure!(e.device.paired, "not paired");
            e.device.mode = mode;
            Ok(())
        })
    }
    pub fn revoke(&self, node_id: &str) -> Result<()> {
        self.mutate(|state| {
            ensure!(state.nodes.remove(node_id).is_some(), "unknown device");
            Ok(())
        })
    }

    fn read<T>(&self, f: impl FnOnce(&RegistryFile) -> Result<T>) -> Result<T> {
        let _lock = self.lock()?;
        f(&self.load()?)
    }
    fn mutate<T>(&self, f: impl FnOnce(&mut RegistryFile) -> Result<T>) -> Result<T> {
        let _lock = self.lock()?;
        let mut state = self.load()?;
        let out = f(&mut state)?;
        let parent = self.path.parent().unwrap();
        let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
        tmp.as_file()
            .set_permissions(fs::Permissions::from_mode(0o600))?;
        serde_json::to_writer(&mut tmp, &state)?;
        tmp.flush()?;
        tmp.as_file().sync_all()?;
        tmp.persist(&self.path)?;
        File::open(parent)?.sync_all()?;
        Ok(out)
    }
    fn lock(&self) -> Result<File> {
        private_metadata(self.path.parent().unwrap(), true)?;
        let path = self.path.with_extension("lock");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&path)?;
        private_metadata(&path, false)?;
        // SAFETY: owned valid descriptor; blocking advisory lock, released on drop.
        ensure!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } == 0,
            "registry lock failed"
        );
        Ok(file)
    }
    fn load(&self) -> Result<RegistryFile> {
        match fs::symlink_metadata(&self.path) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(RegistryFile::default());
            }
            Err(e) => return Err(e.into()),
        }
        private_metadata(&self.path, false)?;
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&self.path)?;
        ensure!(file.metadata()?.len() <= MAX_REGISTRY, "registry too large");
        let mut bytes = Vec::new();
        file.take(MAX_REGISTRY + 1).read_to_end(&mut bytes)?;
        ensure!(bytes.len() as u64 <= MAX_REGISTRY, "registry too large");
        let state: RegistryFile = serde_json::from_slice(&bytes)
            .context("invalid/legacy voice registry; explicit migration required")?;
        ensure!(
            state.version == 1 && state.nodes.len() <= 256,
            "unsupported registry"
        );
        for (id, e) in &state.nodes {
            ensure!(id == &e.device.node_id, "registry identity mismatch");
            validate_device(&e.device)?;
            if e.device.paired {
                ensure!(
                    hex::decode(&e.salt)?.len() == 16 && hex::decode(&e.token_hash)?.len() == 32,
                    "invalid credential hash"
                );
            }
        }
        Ok(state)
    }
}
use std::os::unix::fs::DirBuilderExt;

fn private_metadata(path: &Path, directory: bool) -> Result<()> {
    let m = fs::symlink_metadata(path)?;
    ensure!(
        !m.file_type().is_symlink() && (if directory { m.is_dir() } else { m.is_file() }),
        "private state must not be a symlink or special file"
    );
    ensure!(
        m.uid() == unsafe { libc::geteuid() } && m.mode() & 0o077 == 0,
        "state must be owned by current user and private (directory 0700/file 0600)"
    );
    Ok(())
}
fn validate_name(name: &str) -> Result<()> {
    ensure!(
        !name.trim().is_empty() && name.len() <= 128 && !name.chars().any(char::is_control),
        "friendly name must be 1..128 printable bytes"
    );
    Ok(())
}
fn validate_device(d: &Device) -> Result<()> {
    uuid::Uuid::parse_str(&d.node_id)?;
    validate_name(&d.friendly_name)?;
    if d.paired {
        let s = d.scope.as_ref().context("paired device missing scope")?;
        s.validate()?;
        ensure!(
            s.channel == "voice" && s.session == d.node_id,
            "invalid bound device scope"
        );
    } else if d.scope.is_some() || d.mode != PolicyMode::Muted {
        bail!("pending device may not have scope or capabilities");
    }
    Ok(())
}
fn derive(token: &str, salt: &[u8]) -> [u8; 32] {
    let mut out = [0; 32];
    pbkdf2_hmac::<Sha256>(token.as_bytes(), salt, ITERATIONS, &mut out);
    out
}
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
