//! Explicit private vault and export signatures. Does not read Python Missy's secrets.
#![forbid(unsafe_op_in_unsafe_fn)]
use anyhow::{Context, Result, bail};
use base64::{Engine, engine::general_purpose::STANDARD};
use chacha20poly1305::{
    ChaCha20Poly1305, KeyInit, Nonce,
    aead::{Aead, Payload},
};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::CString,
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    },
    path::Path,
};
use zeroize::{Zeroize, Zeroizing};

const LIMIT: u64 = 4 * 1024 * 1024;
const FORMAT: &str = "missy-rust-vault-v1";

#[derive(Default, Serialize, Deserialize)]
#[serde(transparent)]
struct SecretMap(BTreeMap<String, String>);
impl std::ops::Deref for SecretMap {
    type Target = BTreeMap<String, String>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for SecretMap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl Zeroize for SecretMap {
    fn zeroize(&mut self) {
        for value in self.0.values_mut() {
            value.zeroize();
        }
        self.0.clear();
    }
}

/// All file access is descriptor relative to a private directory, never via a symlink.
pub struct Vault {
    dir: File,
    _lock: File,
    key: Zeroizing<[u8; 32]>,
    mutation: std::sync::Mutex<()>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    format: String,
    nonce: String,
    ciphertext: String,
}

impl Vault {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        match std::fs::symlink_metadata(path) {
            Ok(m) => {
                if m.file_type().is_symlink() || !m.is_dir() {
                    bail!("vault directory must not be a symlink");
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Parent is operator-selected state; never recursively chmod an existing tree.
                std::fs::create_dir(path).context("vault parent must exist")?;
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
            }
            Err(_) => bail!("vault directory unavailable"),
        }
        let dir = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)?;
        check_private(&dir, true)?;
        let lock = openat(&dir, "vault.lock", libc::O_RDWR | libc::O_CREAT)?;
        check_private(&lock, false)?;
        // Lock held throughout decrypt/read/modify/encrypt/write, not merely on final rename.
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } != 0 {
            bail!("vault lock unavailable");
        }
        let key = match readat(&dir, "vault.key", 32) {
            Ok(bytes) => {
                if bytes.len() != 32 {
                    bail!("vault key has invalid length");
                }
                let mut key = Zeroizing::new([0u8; 32]);
                key.copy_from_slice(&bytes);
                key
            }
            Err(e)
                if e.downcast_ref::<std::io::Error>()
                    .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound) =>
            {
                // Never regenerate a missing key over an existing encrypted vault.
                if entry_exists(&dir, "vault.json")? {
                    bail!("vault key is missing; restore it, do not regenerate");
                }
                let mut key = Zeroizing::new([0u8; 32]);
                OsRng.fill_bytes(key.as_mut());
                let mut keyfile = openat(
                    &dir,
                    "vault.key",
                    libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
                )?;
                keyfile.write_all(key.as_ref())?;
                keyfile.sync_all()?;
                dir.sync_all()?;
                key
            }
            Err(_) => bail!("vault key unavailable or insecure"),
        };
        Ok(Self {
            dir,
            _lock: lock,
            key,
            mutation: std::sync::Mutex::new(()),
        })
    }
    pub fn set(&self, name: &str, secret: &[u8]) -> Result<()> {
        let _guard = self
            .mutation
            .lock()
            .map_err(|_| anyhow::anyhow!("vault mutation lock unavailable"))?;
        validate_name(name)?;
        if secret.len() > 65536 {
            bail!("secret exceeds 64KiB");
        }
        let mut values = self.decrypt()?;
        if let Some(mut previous) = values.insert(name.to_owned(), STANDARD.encode(secret)) {
            previous.zeroize();
        }
        self.encrypt(&values)
    }
    pub fn get(&self, name: &str) -> Result<Option<Zeroizing<Vec<u8>>>> {
        validate_name(name)?;
        self.decrypt()?
            .get(name)
            .map(|v| STANDARD.decode(v).map(Zeroizing::new).map_err(Into::into))
            .transpose()
    }
    pub fn list(&self) -> Result<Vec<String>> {
        Ok(self.decrypt()?.keys().cloned().collect())
    }
    pub fn delete(&self, name: &str) -> Result<bool> {
        let _guard = self
            .mutation
            .lock()
            .map_err(|_| anyhow::anyhow!("vault mutation lock unavailable"))?;
        validate_name(name)?;
        let mut values = self.decrypt()?;
        let removed = if let Some(mut previous) = values.remove(name) {
            previous.zeroize();
            true
        } else {
            false
        };
        if removed {
            self.encrypt(&values)?;
        }
        Ok(removed)
    }
    fn decrypt(&self) -> Result<Zeroizing<SecretMap>> {
        let bytes = match readat(&self.dir, "vault.json", LIMIT) {
            Ok(b) => b,
            Err(e)
                if e.downcast_ref::<std::io::Error>()
                    .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound) =>
            {
                return Ok(Zeroizing::new(SecretMap::default()));
            }
            Err(_) => bail!("vault ciphertext unavailable or insecure"),
        };
        let envelope: Envelope =
            serde_json::from_slice(&bytes).context("invalid vault envelope")?;
        if envelope.format != FORMAT {
            bail!("unsupported vault format");
        }
        let nonce = STANDARD.decode(envelope.nonce)?;
        if nonce.len() != 12 {
            bail!("invalid vault nonce");
        }
        let ciphertext = STANDARD.decode(envelope.ciphertext)?;
        let cipher = ChaCha20Poly1305::new_from_slice(self.key.as_ref())
            .map_err(|_| anyhow::anyhow!("vault key invalid"))?;
        let plaintext = Zeroizing::new(
            cipher
                .decrypt(
                    Nonce::from_slice(&nonce),
                    Payload {
                        msg: &ciphertext,
                        aad: FORMAT.as_bytes(),
                    },
                )
                .map_err(|_| anyhow::anyhow!("vault authentication failed"))?,
        );
        serde_json::from_slice(&plaintext)
            .map(Zeroizing::new)
            .context("invalid authenticated vault contents")
    }
    fn encrypt(&self, values: &BTreeMap<String, String>) -> Result<()> {
        let plaintext = Zeroizing::new(serde_json::to_vec(values)?);
        if plaintext.len() > LIMIT as usize / 2 {
            bail!("vault capacity exceeded");
        }
        let mut nonce = [0u8; 12];
        OsRng.fill_bytes(&mut nonce);
        let cipher = ChaCha20Poly1305::new_from_slice(self.key.as_ref())
            .map_err(|_| anyhow::anyhow!("vault key invalid"))?;
        let ciphertext = cipher
            .encrypt(
                Nonce::from_slice(&nonce),
                Payload {
                    msg: &plaintext,
                    aad: FORMAT.as_bytes(),
                },
            )
            .map_err(|_| anyhow::anyhow!("vault encryption failed"))?;
        let envelope = Envelope {
            format: FORMAT.into(),
            nonce: STANDARD.encode(nonce),
            ciphertext: STANDARD.encode(ciphertext),
        };
        atomic_write(&self.dir, "vault.json", &serde_json::to_vec(&envelope)?)
    }
    /// Sign a supplied audit export. Signatures authenticate this export, not every live event.
    pub fn sign_export(&self, data: &[u8]) -> Result<SignedExport> {
        if data.len() > LIMIT as usize {
            bail!("export exceeds 4MiB");
        }
        let key = match readat(&self.dir, "identity.key", 32) {
            Ok(k) => {
                if k.len() != 32 {
                    bail!("identity key invalid");
                }
                let mut b = Zeroizing::new([0u8; 32]);
                b.copy_from_slice(&k);
                b
            }
            Err(e)
                if e.downcast_ref::<std::io::Error>()
                    .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound) =>
            {
                let mut b = Zeroizing::new([0u8; 32]);
                OsRng.fill_bytes(b.as_mut());
                let mut f = openat(
                    &self.dir,
                    "identity.key",
                    libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
                )?;
                f.write_all(b.as_ref())?;
                f.sync_all()?;
                self.dir.sync_all()?;
                b
            }
            Err(_) => bail!("identity key unavailable"),
        };
        let signing = SigningKey::from_bytes(&key);
        let digest = Sha256::digest(data);
        Ok(SignedExport {
            version: 1,
            public_key: STANDARD.encode(signing.verifying_key().as_bytes()),
            sha256: hex::encode(digest),
            signature: STANDARD.encode(signing.sign(data).to_bytes()),
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedExport {
    pub version: u32,
    pub public_key: String,
    pub sha256: String,
    pub signature: String,
}
impl SignedExport {
    /// Caller must independently pin `expected_public_key`; a self-supplied key proves no identity.
    pub fn verify(&self, data: &[u8], expected_public_key: &str) -> Result<()> {
        if self.version != 1
            || self.public_key != expected_public_key
            || self.sha256 != hex::encode(Sha256::digest(data))
        {
            bail!("export identity or digest mismatch");
        }
        let public: [u8; 32] = STANDARD
            .decode(&self.public_key)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("invalid public key"))?;
        let sig = Signature::from_slice(&STANDARD.decode(&self.signature)?)?;
        VerifyingKey::from_bytes(&public)?
            .verify(data, &sig)
            .context("export signature invalid")
    }
}
fn validate_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 128
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-.:".contains(&b))
    {
        bail!("invalid secret name");
    }
    Ok(())
}
fn check_private(file: &File, directory: bool) -> Result<()> {
    let m = file.metadata()?;
    if m.uid() != unsafe { libc::geteuid() }
        || m.mode() & 0o077 != 0
        || (directory && !m.is_dir())
        || (!directory && (!m.is_file() || m.nlink() != 1))
    {
        bail!("vault path must be private, owned and not hard-linked");
    }
    Ok(())
}
fn openat(dir: &File, name: &str, flags: i32) -> Result<File> {
    let name = CString::new(name)?;
    let fd = unsafe {
        libc::openat(
            dir.as_raw_fd(),
            name.as_ptr(),
            flags | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK,
            0o600,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}
fn readat(dir: &File, name: &str, limit: u64) -> Result<Zeroizing<Vec<u8>>> {
    let file = openat(dir, name, libc::O_RDONLY)?;
    check_private(&file, false)?;
    if file.metadata()?.len() > limit {
        bail!("vault file exceeds limit");
    }
    let mut data = Zeroizing::new(Vec::new());
    file.take(limit + 1).read_to_end(&mut data)?;
    if data.len() as u64 > limit {
        bail!("vault file exceeds limit");
    }
    Ok(data)
}
fn entry_exists(dir: &File, name: &str) -> Result<bool> {
    let name = CString::new(name)?;
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    if unsafe {
        libc::fstatat(
            dir.as_raw_fd(),
            name.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    } == 0
    {
        return Ok(true);
    }
    let e = std::io::Error::last_os_error();
    if e.kind() == std::io::ErrorKind::NotFound {
        Ok(false)
    } else {
        Err(e.into())
    }
}
fn atomic_write(dir: &File, name: &str, data: &[u8]) -> Result<()> {
    let mut random = [0u8; 16];
    OsRng.fill_bytes(&mut random);
    let temp = format!(".vault-{}", hex::encode(random));
    let result = (|| {
        let mut f = openat(dir, &temp, libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL)?;
        f.write_all(data)?;
        f.sync_all()?;
        let a = CString::new(temp.as_str())?;
        let b = CString::new(name)?;
        if unsafe { libc::renameat(dir.as_raw_fd(), a.as_ptr(), dir.as_raw_fd(), b.as_ptr()) } != 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
        dir.sync_all()?;
        Ok(())
    })();
    if let Ok(temp) = CString::new(temp) {
        unsafe {
            libc::unlinkat(dir.as_raw_fd(), temp.as_ptr(), 0);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_private_ciphertext_and_persistence() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("vault");
        {
            let v = Vault::open(&path).unwrap();
            v.set("test", b"sensitive-canary").unwrap();
            assert_eq!(
                v.get("test").unwrap().unwrap().as_slice(),
                b"sensitive-canary"
            );
            assert_eq!(v.list().unwrap(), vec!["test"]);
        }
        assert!(
            !String::from_utf8_lossy(&std::fs::read(path.join("vault.json")).unwrap())
                .contains("sensitive-canary")
        );
        let v = Vault::open(&path).unwrap();
        assert!(v.delete("test").unwrap());
        assert!(v.get("test").unwrap().is_none());
    }
    #[test]
    fn corrupted_ciphertext_is_not_empty_vault() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("vault");
        {
            let v = Vault::open(&path).unwrap();
            v.set("test", b"secret").unwrap();
        }
        std::fs::write(path.join("vault.json"), b"{}").unwrap();
        assert!(Vault::open(&path).unwrap().list().is_err());
    }
    #[test]
    fn deleted_key_never_regenerates() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("vault");
        {
            let v = Vault::open(&path).unwrap();
            v.set("test", b"secret").unwrap();
        }
        std::fs::remove_file(path.join("vault.key")).unwrap();
        assert!(Vault::open(&path).is_err());
        assert!(!path.join("vault.key").exists());
    }
    #[test]
    fn missing_key_with_symlink_ciphertext_never_regenerates() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("vault");
        {
            let _v = Vault::open(&path).unwrap();
        }
        std::fs::remove_file(path.join("vault.key")).unwrap();
        std::os::unix::fs::symlink("does-not-exist", path.join("vault.json")).unwrap();
        assert!(Vault::open(&path).is_err());
        assert!(!path.join("vault.key").exists());
    }
    #[test]
    fn signed_export_requires_pinned_key_and_original_bytes() {
        let temp = tempfile::tempdir().unwrap();
        let v = Vault::open(temp.path().join("vault")).unwrap();
        let s = v.sign_export(b"audit").unwrap();
        s.verify(b"audit", &s.public_key).unwrap();
        assert!(s.verify(b"changed", &s.public_key).is_err());
        assert!(s.verify(b"audit", "other").is_err());
    }
    #[test]
    fn symlink_and_insecure_modes_refused() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("vault");
        {
            let _v = Vault::open(&path).unwrap();
        }
        let link = temp.path().join("link");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(Vault::open(link).is_err());
        std::fs::set_permissions(
            path.join("vault.key"),
            std::fs::Permissions::from_mode(0o644),
        )
        .unwrap();
        assert!(Vault::open(&path).is_err());
    }
    #[test]
    fn parallel_mutations_reload_under_lock() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("vault");
        {
            let _v = Vault::open(&path).unwrap();
        }
        std::thread::scope(|s| {
            for i in 0..8 {
                let path = &path;
                s.spawn(move || {
                    Vault::open(path)
                        .unwrap()
                        .set(&format!("k{i}"), b"v")
                        .unwrap();
                });
            }
        });
        assert_eq!(Vault::open(&path).unwrap().list().unwrap().len(), 8);
    }
    #[test]
    fn same_vault_parallel_mutations_are_serialized() {
        let temp = tempfile::tempdir().unwrap();
        let vault = Vault::open(temp.path().join("vault")).unwrap();
        std::thread::scope(|s| {
            for i in 0..32 {
                let v = &vault;
                s.spawn(move || {
                    v.set(&format!("k{i}"), b"value").unwrap();
                });
            }
        });
        assert_eq!(vault.list().unwrap().len(), 32);
    }
}
