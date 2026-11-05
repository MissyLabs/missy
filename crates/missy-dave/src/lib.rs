//! Thin, fail-closed bindings to the official MIT-licensed Discord libdave C API.
//! No cryptographic primitives, automatic library discovery, subprocesses, or network.
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::{CString, c_char, c_void},
    fs::File,
    io::{Read, Write},
    num::NonZeroU16,
    path::PathBuf,
    ptr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub const DAVE_PROTOCOL_VERSION: u16 = 1;
pub const LIBDAVE_VERSION: &str = "1.2.2/cpp";
pub const LIBDAVE_SOURCE_COMMIT: &str = "6396c90aedbb9a6ee117ea9ce1ce9832af6a3375";
const LIMIT: usize = 1_048_576;
type Handle = *mut c_void;
type FailureCallback = Option<unsafe extern "C" fn(*const c_char, *const c_char, Handle)>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeDaveConfig {
    pub library_path: PathBuf,
    pub sha256: String,
    pub version: String,
    pub source_commit: String,
}
impl NativeDaveConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.library_path.is_absolute(),
            "libdave requires an explicit absolute library path"
        );
        ensure!(
            self.version == LIBDAVE_VERSION && self.source_commit == LIBDAVE_SOURCE_COMMIT,
            "unsupported libdave source/version pin"
        );
        ensure!(
            self.sha256.len() == 64 && self.sha256.bytes().all(|v| v.is_ascii_hexdigit()),
            "explicit libdave SHA-256 required"
        );
        Ok(())
    }
}

macro_rules! api {
    ($($name:ident: $ty:ty),* $(,)?) => {
        pub(super) struct Api { $(pub(super) $name: $ty,)* _library: libloading::Library, _image: File }
        impl Api {
            pub(super) unsafe fn load(library: libloading::Library, image: File) -> Result<Self> {
                // SAFETY: exact signatures from the pinned official dave.h; lifetime owned below.
                unsafe { Ok(Self { $($name: *library.get::<$ty>(concat!(stringify!($name), "\0").as_bytes())?,)* _library: library, _image: image }) }
            }
        }
    }
}
#[allow(non_snake_case)]
mod ffi {
    use super::*;
    api! {
        daveMaxSupportedProtocolVersion: unsafe extern "C" fn() -> u16,
        daveFree: unsafe extern "C" fn(Handle),
        daveSessionCreate: unsafe extern "C" fn(Handle, *const c_char, FailureCallback, Handle) -> Handle,
        daveSessionDestroy: unsafe extern "C" fn(Handle),
        daveSessionInit: unsafe extern "C" fn(Handle, u16, u64, *const c_char),
        daveSessionReset: unsafe extern "C" fn(Handle),
        daveSessionGetProtocolVersion: unsafe extern "C" fn(Handle) -> u16,
        daveSessionSetExternalSender: unsafe extern "C" fn(Handle, *const u8, usize),
        daveSessionGetMarshalledKeyPackage: unsafe extern "C" fn(Handle, *mut *mut u8, *mut usize),
        daveSessionProcessProposals: unsafe extern "C" fn(Handle, *const u8, usize, *const *const c_char, usize, *mut *mut u8, *mut usize),
        daveSessionProcessCommit: unsafe extern "C" fn(Handle, *const u8, usize) -> Handle,
        daveSessionProcessWelcome: unsafe extern "C" fn(Handle, *const u8, usize, *const *const c_char, usize) -> Handle,
        daveSessionGetKeyRatchet: unsafe extern "C" fn(Handle, *const c_char) -> Handle,
        daveKeyRatchetDestroy: unsafe extern "C" fn(Handle),
        daveCommitResultIsFailed: unsafe extern "C" fn(Handle) -> bool,
        daveCommitResultIsIgnored: unsafe extern "C" fn(Handle) -> bool,
        daveCommitResultGetRosterMemberIds: unsafe extern "C" fn(Handle, *mut *mut u64, *mut usize),
        daveCommitResultDestroy: unsafe extern "C" fn(Handle),
        daveWelcomeResultGetRosterMemberIds: unsafe extern "C" fn(Handle, *mut *mut u64, *mut usize),
        daveWelcomeResultDestroy: unsafe extern "C" fn(Handle),
        daveEncryptorCreate: unsafe extern "C" fn() -> Handle,
        daveEncryptorDestroy: unsafe extern "C" fn(Handle),
        daveEncryptorSetKeyRatchet: unsafe extern "C" fn(Handle, Handle),
        daveEncryptorSetPassthroughMode: unsafe extern "C" fn(Handle, bool),
        daveEncryptorAssignSsrcToCodec: unsafe extern "C" fn(Handle, u32, i32),
        daveEncryptorGetMaxCiphertextByteSize: unsafe extern "C" fn(Handle, i32, usize) -> usize,
        daveEncryptorEncrypt: unsafe extern "C" fn(Handle, i32, u32, *const u8, usize, *mut u8, usize, *mut usize) -> i32,
        daveDecryptorCreate: unsafe extern "C" fn() -> Handle,
        daveDecryptorDestroy: unsafe extern "C" fn(Handle),
        daveDecryptorTransitionToKeyRatchet: unsafe extern "C" fn(Handle, Handle),
        daveDecryptorTransitionToPassthroughMode: unsafe extern "C" fn(Handle, bool),
        daveDecryptorGetMaxPlaintextByteSize: unsafe extern "C" fn(Handle, i32, usize) -> usize,
        daveDecryptorDecrypt: unsafe extern "C" fn(Handle, i32, *const u8, usize, *mut u8, usize, *mut usize) -> i32,
        daveSetLogSinkCallback: unsafe extern "C" fn(Option<unsafe extern "C" fn(i32, *const c_char, i32, *const c_char)>),
    }
    pub(super) use self::Api as NativeApi;
}
use ffi::NativeApi as Api;

// The bytes loaded are the bytes hashed, not a subsequently reopened mutable pathname.
// A sealed Linux memfd prevents a path-swap or in-place replacement between hash and dlopen.
#[cfg(target_os = "linux")]
fn load(config: &NativeDaveConfig) -> Result<Arc<Api>> {
    use std::os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::OpenOptionsExt,
    };
    config.validate()?;
    let mut input = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&config.library_path)
        .context("open pinned libdave")?;
    ensure!(
        input.metadata()?.is_file() && input.metadata()?.len() <= 64 * 1024 * 1024,
        "invalid native library file"
    );
    let mut bytes = Vec::new();
    Read::by_ref(&mut input)
        .take(64 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 64 * 1024 * 1024,
        "native library exceeds size bound"
    );
    ensure!(
        hex::encode(Sha256::digest(&bytes)).eq_ignore_ascii_case(&config.sha256),
        "libdave SHA-256 mismatch"
    );
    // SAFETY: constant NUL-terminated name, returned FD checked then uniquely owned.
    let fd = unsafe {
        libc::memfd_create(
            c"missy-libdave".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        )
    };
    ensure!(fd >= 0, "cannot create sealed native library image");
    let mut file = unsafe { File::from_raw_fd(fd) };
    file.write_all(&bytes)?;
    let seals = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
    ensure!(
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, seals) } == 0,
        "cannot seal native library image"
    );
    // SAFETY: explicit operator-approved executable bytes pinned to digest and source version.
    let api = unsafe {
        Api::load(
            libloading::Library::new(format!("/proc/self/fd/{}", file.as_raw_fd()))?,
            file,
        )?
    };
    ensure!(
        unsafe { (api.daveMaxSupportedProtocolVersion)() } == DAVE_PROTOCOL_VERSION,
        "unsupported native DAVE protocol"
    );
    unsafe {
        (api.daveSetLogSinkCallback)(Some(discard_log));
    }
    Ok(Arc::new(api))
}
#[cfg(not(target_os = "linux"))]
fn load(_: &NativeDaveConfig) -> Result<Arc<Api>> {
    bail!("pinned libdave loader currently requires Linux sealed memfd")
}
unsafe extern "C" fn discard_log(_: i32, _: *const c_char, _: i32, _: *const c_char) {}
unsafe extern "C" fn failure(_: *const c_char, _: *const c_char, data: Handle) {
    if !data.is_null() {
        unsafe {
            (&*(data.cast::<AtomicBool>())).store(true, Ordering::SeqCst);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ProposalsOperationType {
    APPEND,
    REVOKE,
}
pub struct CommitWelcome {
    pub commit: Vec<u8>,
    pub welcome: Option<Vec<u8>>,
}
struct Ratchet {
    api: Arc<Api>,
    handle: Handle,
}
impl Drop for Ratchet {
    fn drop(&mut self) {
        unsafe {
            (self.api.daveKeyRatchetDestroy)(self.handle);
        }
    }
}

pub struct DaveSession {
    api: Arc<Api>,
    session: Handle,
    encryptor: Handle,
    decryptors: BTreeMap<u64, Handle>,
    pending: BTreeMap<u16, BTreeMap<u64, Ratchet>>,
    pending_order: BTreeMap<u16, u64>,
    transition_serial: u64,
    executed_serial: u64,
    roster: Vec<u64>,
    user: u64,
    failed: Box<AtomicBool>,
    active: bool,
}
// SAFETY: session and media handles are exclusively accessed through &mut self;
// official libdave has no caller-thread affinity and callback data is heap-stable/atomic.
// No Sync implementation: concurrent use of a session is not supported.
unsafe impl Send for DaveSession {}
impl DaveSession {
    /// Media remains gated until a validated MLS transition installs ratchets.
    pub fn is_ready(&self) -> bool {
        self.active
    }
    pub fn new(
        config: &NativeDaveConfig,
        version: NonZeroU16,
        user: u64,
        channel: u64,
        _: Option<()>,
    ) -> Result<Self> {
        let api = load(config)?;
        let mut failed = Box::new(AtomicBool::new(false));
        let session = unsafe {
            (api.daveSessionCreate)(
                ptr::null_mut(),
                ptr::null(),
                Some(failure),
                (&mut *failed as *mut AtomicBool).cast(),
            )
        };
        ensure!(!session.is_null(), "native DAVE session allocation failed");
        let encryptor = unsafe { (api.daveEncryptorCreate)() };
        if encryptor.is_null() {
            unsafe {
                (api.daveSessionDestroy)(session);
            }
            bail!("native DAVE encryptor allocation failed");
        }
        let mut value = Self {
            api,
            session,
            encryptor,
            decryptors: BTreeMap::new(),
            pending: BTreeMap::new(),
            pending_order: BTreeMap::new(),
            transition_serial: 0,
            executed_serial: 0,
            roster: vec![user],
            user,
            failed,
            active: false,
        };
        unsafe {
            (value.api.daveEncryptorSetPassthroughMode)(encryptor, false);
        }
        value.reinit(version, user, channel, None)?;
        Ok(value)
    }
    fn check(&self) -> Result<()> {
        ensure!(
            !self.failed.swap(false, Ordering::SeqCst),
            "native MLS operation rejected"
        );
        Ok(())
    }
    fn bound(data: &[u8]) -> Result<()> {
        ensure!(
            !data.is_empty() && data.len() <= LIMIT,
            "invalid DAVE message size"
        );
        Ok(())
    }
    fn users(users: Option<&[u64]>) -> Result<(Vec<CString>, Vec<*const c_char>)> {
        let ids = users.unwrap_or_default();
        ensure!(
            !ids.is_empty() && ids.len() <= 2048 && ids.iter().all(|v| *v != 0),
            "recognized MLS roster required"
        );
        let strings: Vec<CString> = ids
            .iter()
            .map(|v| CString::new(v.to_string()).unwrap())
            .collect();
        let pointers = strings.iter().map(|v| v.as_ptr()).collect();
        Ok((strings, pointers))
    }
    unsafe fn take_bytes(&self, data: *mut u8, len: usize) -> Result<Vec<u8>> {
        if data.is_null() {
            ensure!(len == 0, "invalid native output");
            return Ok(Vec::new());
        }
        let result = if len <= LIMIT {
            Ok(unsafe { std::slice::from_raw_parts(data, len) }.to_vec())
        } else {
            Err(anyhow::anyhow!("native output exceeds bound"))
        };
        unsafe {
            (self.api.daveFree)(data.cast());
        }
        result
    }
    unsafe fn take_roster(&self, data: *mut u64, len: usize) -> Result<Vec<u64>> {
        if data.is_null() {
            ensure!(len == 0, "invalid native roster");
            return Ok(Vec::new());
        }
        let result = if len <= 2048 {
            Ok(unsafe { std::slice::from_raw_parts(data, len) }.to_vec())
        } else {
            Err(anyhow::anyhow!("native roster exceeds bound"))
        };
        unsafe {
            (self.api.daveFree)(data.cast());
        }
        result
    }
    pub fn reinit(
        &mut self,
        version: NonZeroU16,
        user: u64,
        channel: u64,
        _: Option<()>,
    ) -> Result<()> {
        ensure!(
            version.get() == DAVE_PROTOCOL_VERSION && user != 0 && channel != 0,
            "invalid DAVE initialization"
        );
        self.active = false;
        self.pending.clear();
        self.pending_order.clear();
        self.transition_serial = 0;
        self.executed_serial = 0;
        for (_, handle) in std::mem::take(&mut self.decryptors) {
            unsafe {
                (self.api.daveDecryptorDestroy)(handle);
            }
        }
        self.user = user;
        self.roster = vec![user];
        self.failed.store(false, Ordering::SeqCst);
        let user = CString::new(user.to_string())?;
        unsafe {
            (self.api.daveSessionReset)(self.session);
            (self.api.daveSessionInit)(self.session, version.get(), channel, user.as_ptr());
        }
        self.check()?;
        ensure!(
            unsafe { (self.api.daveSessionGetProtocolVersion)(self.session) } == version.get(),
            "native DAVE initialization failed"
        );
        Ok(())
    }
    pub fn create_key_package(&mut self) -> Result<Vec<u8>> {
        let (mut data, mut len) = (ptr::null_mut(), 0);
        unsafe {
            (self.api.daveSessionGetMarshalledKeyPackage)(self.session, &mut data, &mut len);
        }
        let bytes = unsafe { self.take_bytes(data, len)? };
        self.check()?;
        ensure!(!bytes.is_empty(), "empty MLS key package");
        Ok(bytes)
    }
    pub fn set_external_sender(&mut self, bytes: &[u8]) -> Result<()> {
        Self::bound(bytes)?;
        unsafe {
            (self.api.daveSessionSetExternalSender)(self.session, bytes.as_ptr(), bytes.len());
        }
        self.check()
    }
    pub fn process_proposals(
        &mut self,
        op: ProposalsOperationType,
        bytes: &[u8],
        users: Option<&[u64]>,
    ) -> Result<Option<CommitWelcome>> {
        Self::bound(bytes)?;
        let (_strings, ids) = Self::users(users)?;
        let mut wire = vec![match op {
            ProposalsOperationType::APPEND => 0,
            ProposalsOperationType::REVOKE => 1,
        }];
        wire.extend_from_slice(bytes);
        let (mut data, mut len) = (ptr::null_mut(), 0);
        unsafe {
            (self.api.daveSessionProcessProposals)(
                self.session,
                wire.as_ptr(),
                wire.len(),
                ids.as_ptr(),
                ids.len(),
                &mut data,
                &mut len,
            );
        }
        let combined = unsafe { self.take_bytes(data, len)? };
        self.check()?;
        // Official C API returns already-concatenated MLS commit + optional welcome.
        Ok((!combined.is_empty()).then_some(CommitWelcome {
            commit: combined,
            welcome: None,
        }))
    }
    pub fn process_commit(&mut self, bytes: &[u8]) -> Result<bool> {
        Self::bound(bytes)?;
        let result = unsafe {
            (self.api.daveSessionProcessCommit)(self.session, bytes.as_ptr(), bytes.len())
        };
        ensure!(!result.is_null(), "missing MLS commit result");
        let failed = unsafe { (self.api.daveCommitResultIsFailed)(result) };
        let ignored = unsafe { (self.api.daveCommitResultIsIgnored)(result) };
        let (mut ids, mut len) = (ptr::null_mut(), 0);
        if !failed && !ignored {
            unsafe {
                (self.api.daveCommitResultGetRosterMemberIds)(result, &mut ids, &mut len);
            }
        }
        unsafe {
            (self.api.daveCommitResultDestroy)(result);
        }
        let roster = unsafe { self.take_roster(ids, len)? };
        self.check()?;
        ensure!(!failed, "MLS commit rejected");
        if !ignored {
            self.roster = roster;
        }
        Ok(!ignored)
    }
    pub fn process_welcome(&mut self, bytes: &[u8], users: Option<&[u64]>) -> Result<()> {
        Self::bound(bytes)?;
        let (_strings, ids) = Self::users(users)?;
        let result = unsafe {
            (self.api.daveSessionProcessWelcome)(
                self.session,
                bytes.as_ptr(),
                bytes.len(),
                ids.as_ptr(),
                ids.len(),
            )
        };
        if result.is_null() {
            self.check()?;
            bail!("MLS welcome rejected");
        }
        let (mut roster, mut len) = (ptr::null_mut(), 0);
        unsafe {
            (self.api.daveWelcomeResultGetRosterMemberIds)(result, &mut roster, &mut len);
            (self.api.daveWelcomeResultDestroy)(result);
        }
        self.roster = unsafe { self.take_roster(roster, len)? };
        self.check()
    }
    pub fn prepare_transition(&mut self, id: u16, version: u16) -> Result<()> {
        ensure!(version == DAVE_PROTOCOL_VERSION, "DAVE downgrade denied");
        ensure!(
            self.pending.len() < 16 && !self.pending.contains_key(&id),
            "too many/duplicate DAVE transitions"
        );
        let mut ratchets = BTreeMap::new();
        for user in &self.roster {
            let text = CString::new(user.to_string())?;
            let handle =
                unsafe { (self.api.daveSessionGetKeyRatchet)(self.session, text.as_ptr()) };
            ensure!(!handle.is_null(), "MLS epoch key not available");
            ratchets.insert(
                *user,
                Ratchet {
                    api: self.api.clone(),
                    handle,
                },
            );
        }
        ensure!(ratchets.contains_key(&self.user), "local MLS member absent");
        self.check()?;
        self.pending.insert(id, ratchets);
        self.transition_serial = self
            .transition_serial
            .checked_add(1)
            .context("DAVE transition serial exhausted")?;
        self.pending_order.insert(id, self.transition_serial);
        Ok(())
    }
    pub fn execute_transition(&mut self, id: u16) -> Result<()> {
        let serial = *self
            .pending_order
            .get(&id)
            .context("unprepared DAVE transition")?;
        ensure!(
            serial > self.executed_serial,
            "stale DAVE transition rejected"
        );
        let ratchets = self
            .pending
            .remove(&id)
            .context("unprepared DAVE transition")?;
        for (user, ratchet) in &ratchets {
            if *user == self.user {
                unsafe {
                    (self.api.daveEncryptorSetKeyRatchet)(self.encryptor, ratchet.handle);
                }
            }
            let decryptor = if let Some(v) = self.decryptors.get(user) {
                *v
            } else {
                let handle = unsafe { (self.api.daveDecryptorCreate)() };
                ensure!(!handle.is_null(), "native decryptor allocation failed");
                unsafe {
                    (self.api.daveDecryptorTransitionToPassthroughMode)(handle, false);
                }
                self.decryptors.insert(*user, handle);
                handle
            };
            unsafe {
                (self.api.daveDecryptorTransitionToKeyRatchet)(decryptor, ratchet.handle);
            }
        }
        let removed: Vec<u64> = self
            .decryptors
            .keys()
            .filter(|v| !ratchets.contains_key(v))
            .copied()
            .collect();
        for user in removed {
            if let Some(handle) = self.decryptors.remove(&user) {
                unsafe {
                    (self.api.daveDecryptorDestroy)(handle);
                }
            }
        }
        self.active = true;
        self.executed_serial = serial;
        self.pending_order.retain(|_, order| *order > serial);
        self.pending
            .retain(|id, _| self.pending_order.contains_key(id));
        Ok(())
    }
    pub fn encrypt_opus(&mut self, ssrc: u32, frame: &[u8]) -> Result<Vec<u8>> {
        Self::bound(frame)?;
        ensure!(self.active, "DAVE epoch not active");
        unsafe {
            (self.api.daveEncryptorAssignSsrcToCodec)(self.encryptor, ssrc, 1);
        }
        let capacity = unsafe {
            (self.api.daveEncryptorGetMaxCiphertextByteSize)(self.encryptor, 0, frame.len())
        };
        ensure!(capacity <= LIMIT, "native ciphertext exceeds bound");
        let mut output = vec![0; capacity];
        let mut written = 0;
        let code = unsafe {
            (self.api.daveEncryptorEncrypt)(
                self.encryptor,
                0,
                ssrc,
                frame.as_ptr(),
                frame.len(),
                output.as_mut_ptr(),
                capacity,
                &mut written,
            )
        };
        ensure!(code == 0 && written <= capacity, "DAVE encryption failed");
        output.truncate(written);
        Ok(output)
    }
    pub fn decrypt_opus(&mut self, user: u64, frame: &[u8]) -> Result<Vec<u8>> {
        Self::bound(frame)?;
        ensure!(self.active, "DAVE epoch not active");
        let handle = *self.decryptors.get(&user).context("unknown DAVE sender")?;
        let capacity =
            unsafe { (self.api.daveDecryptorGetMaxPlaintextByteSize)(handle, 0, frame.len()) };
        ensure!(capacity <= LIMIT, "native plaintext exceeds bound");
        let mut output = vec![0; capacity];
        let mut written = 0;
        let code = unsafe {
            (self.api.daveDecryptorDecrypt)(
                handle,
                0,
                frame.as_ptr(),
                frame.len(),
                output.as_mut_ptr(),
                capacity,
                &mut written,
            )
        };
        ensure!(
            code == 0 && written <= capacity,
            "DAVE authentication/decryption failed"
        );
        output.truncate(written);
        Ok(output)
    }
}
impl Drop for DaveSession {
    fn drop(&mut self) {
        self.pending.clear();
        unsafe {
            for (_, handle) in std::mem::take(&mut self.decryptors) {
                (self.api.daveDecryptorDestroy)(handle);
            }
            (self.api.daveEncryptorDestroy)(self.encryptor);
            (self.api.daveSessionDestroy)(self.session);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> NativeDaveConfig {
        NativeDaveConfig {
            library_path: "/not/configured/libdave.so".into(),
            sha256: "0".repeat(64),
            version: LIBDAVE_VERSION.into(),
            source_commit: LIBDAVE_SOURCE_COMMIT.into(),
        }
    }
    #[test]
    fn unconfigured_and_unpinned_libraries_are_rejected() {
        let mut cfg = config();
        assert!(load(&cfg).is_err());
        cfg.library_path = "libdave.so".into();
        assert!(cfg.validate().is_err());
        cfg.library_path = "/native/libdave.so".into();
        cfg.sha256.clear();
        assert!(cfg.validate().is_err());
        cfg.sha256 = "0".repeat(64);
        cfg.source_commit = "HEAD".into();
        assert!(cfg.validate().is_err());
    }
    #[test]
    fn hash_is_checked_before_loading_executable_bytes() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"not a native library").unwrap();
        let mut cfg = config();
        cfg.library_path = file.path().into();
        let error = match load(&cfg) {
            Ok(_) => panic!("unexpected successful load"),
            Err(e) => e,
        };
        assert!(error.to_string().contains("SHA-256 mismatch"));
    }
    /// Real official MLS sessions and cryptography, not a fake cryptographic fixture.
    /// Requires the pinned native build and its upstream external-sender test helper.
    #[test]
    #[ignore = "explicit native libdave + official test-helper paths required"]
    fn official_native_two_session_roundtrip() -> Result<()> {
        let mut cfg = config();
        cfg.library_path = std::env::var("MISSY_TEST_LIBDAVE_PATH")?.into();
        cfg.sha256 = std::env::var("MISSY_TEST_LIBDAVE_SHA256")?;
        let helper_path = std::env::var("MISSY_TEST_LIBDAVE_HELPER")?;
        ensure!(
            PathBuf::from(&helper_path).is_absolute(),
            "absolute official test helper required"
        );
        type Create = unsafe extern "C" fn(u64) -> Handle;
        type Destroy = unsafe extern "C" fn(Handle);
        type Sender = unsafe extern "C" fn(Handle, *mut *mut u8, *mut usize);
        type Add = unsafe extern "C" fn(Handle, u32, *mut u8, usize, *mut *mut u8, *mut usize);
        type Split = unsafe extern "C" fn(
            Handle,
            *mut u8,
            usize,
            *mut *mut u8,
            *mut usize,
            *mut *mut u8,
            *mut usize,
        );
        // SAFETY: explicit test-only upstream helper, exact signatures in its pinned header.
        unsafe {
            let library = libloading::Library::new(helper_path)?;
            let create = *library.get::<Create>(b"daveExternalSenderCreate\0")?;
            let destroy = *library.get::<Destroy>(b"daveExternalSenderDestroy\0")?;
            let sender =
                *library.get::<Sender>(b"daveExternalSenderGetMarshalledExternalSender\0")?;
            let add = *library.get::<Add>(b"daveExternalSenderProposeAdd\0")?;
            let split = *library.get::<Split>(b"daveExternalSenderSplitCommitWelcome\0")?;
            let mut alice = DaveSession::new(&cfg, NonZeroU16::new(1).unwrap(), 1001, 9999, None)?;
            let mut bob = DaveSession::new(&cfg, NonZeroU16::new(1).unwrap(), 1002, 9999, None)?;
            let external = create(9999);
            ensure!(!external.is_null(), "external sender allocation failed");
            // Helper RAII is deliberately tied to its still-live native library.
            struct External(Handle, Destroy);
            impl Drop for External {
                fn drop(&mut self) {
                    unsafe {
                        (self.1)(self.0);
                    }
                }
            }
            let _external = External(external, destroy);
            let (mut data, mut len) = (ptr::null_mut(), 0);
            sender(external, &mut data, &mut len);
            let external_bytes = alice.take_bytes(data, len)?;
            alice.set_external_sender(&external_bytes)?;
            bob.set_external_sender(&external_bytes)?;
            let _alice_package = alice.create_key_package()?;
            let mut bob_package = bob.create_key_package()?;
            add(
                external,
                0,
                bob_package.as_mut_ptr(),
                bob_package.len(),
                &mut data,
                &mut len,
            );
            let proposal = alice.take_bytes(data, len)?;
            // Official helper already serializes the operation byte.
            ensure!(
                proposal.first() == Some(&0),
                "unexpected upstream proposal operation"
            );
            let mut combined = alice
                .process_proposals(
                    ProposalsOperationType::APPEND,
                    &proposal[1..],
                    Some(&[1001, 1002]),
                )?
                .context("missing native commit/welcome")?
                .commit;
            let (mut commit, mut commit_len, mut welcome, mut welcome_len) =
                (ptr::null_mut(), 0, ptr::null_mut(), 0);
            split(
                external,
                combined.as_mut_ptr(),
                combined.len(),
                &mut commit,
                &mut commit_len,
                &mut welcome,
                &mut welcome_len,
            );
            let commit = alice.take_bytes(commit, commit_len)?;
            let welcome = alice.take_bytes(welcome, welcome_len)?;
            assert!(alice.process_commit(&commit)?);
            bob.process_welcome(&welcome, Some(&[1001, 1002]))?;
            alice.prepare_transition(7, 1)?;
            bob.prepare_transition(7, 1)?;
            alice.prepare_transition(8, 1)?;
            bob.prepare_transition(8, 1)?;
            assert!(alice.encrypt_opus(42, b"pre-transition").is_err());
            alice.execute_transition(8)?;
            bob.execute_transition(8)?;
            assert!(
                alice.execute_transition(7).is_err(),
                "older transition must not reinstall old key domains"
            );
            assert!(bob.execute_transition(7).is_err());
            let frame = b"real opus-format-independent DAVE test media";
            let ciphertext = alice.encrypt_opus(42, frame)?;
            assert_ne!(&ciphertext, frame);
            assert_eq!(bob.decrypt_opus(1001, &ciphertext)?, frame);
            assert!(
                bob.decrypt_opus(1001, &ciphertext).is_err(),
                "replay must be rejected"
            );
            let mut tampered = alice.encrypt_opus(42, frame)?;
            tampered[0] ^= 1;
            assert!(
                bob.decrypt_opus(1001, &tampered).is_err(),
                "tampering must be rejected"
            );
            assert!(
                bob.decrypt_opus(1001, frame).is_err(),
                "plaintext downgrade must be rejected"
            );
            assert!(
                bob.decrypt_opus(999, &ciphertext).is_err(),
                "unrecognized sender must be rejected"
            );
            let reverse = bob.encrypt_opus(43, frame)?;
            assert_eq!(alice.decrypt_opus(1002, &reverse)?, frame);
            assert!(alice.prepare_transition(8, 0).is_err());
            assert!(alice.set_external_sender(b"malformed").is_err());
        }
        Ok(())
    }
}
