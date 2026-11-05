//! Capability checks, descriptor-relative filesystem access and the sole outbound
//! HTTP gateway. This is userspace policy, NOT an OS sandbox. An allowed program
//! runs with the service user's authority; configure exact argv accordingly.
//! Linux directory descriptors anchor roots and O_NOFOLLOW is used on every
//! component. Root administrators and same-UID adversaries able to move already
//! opened directories or alter mounts are outside this userspace threat boundary.
#![cfg(target_os = "linux")]

use anyhow::{Result, anyhow, bail};
use missy_core::PolicyConfig;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    ffi::{CString, OsStr},
    fs::File,
    io::{Read, Write},
    net::{IpAddr, Ipv4Addr, SocketAddr},
    os::{
        fd::{AsRawFd, FromRawFd, RawFd},
        unix::{ffi::OsStrExt, fs::MetadataExt},
    },
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use url::{Host, Url};

pub const MAX_HTTP_REQUEST_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_HTTP_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
pub const HTTP_TIMEOUT_SECONDS: u64 = 30;
const MAX_FILE_BYTES: usize = 64 * 1024 * 1024;

struct Root {
    path: PathBuf,
    dir: File,
}
struct Inner {
    config: PolicyConfig,
    reads: Vec<Root>,
    writes: Vec<Root>,
    origins: HashSet<String>,
}
#[derive(Clone)]
pub struct Policy {
    inner: Arc<Inner>,
}

fn cstr(s: &OsStr) -> Result<CString> {
    CString::new(s.as_bytes()).map_err(|_| anyhow!("invalid filesystem component"))
}
fn open_at(fd: RawFd, name: &OsStr, flags: i32, mode: libc::mode_t) -> Result<File> {
    let name = cstr(name)?;
    // SAFETY: name is NUL-terminated; the returned fd is uniquely owned by File.
    let raw = unsafe {
        libc::openat(
            fd,
            name.as_ptr(),
            flags | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            mode,
        )
    };
    if raw < 0 {
        bail!("filesystem access denied or unavailable");
    }
    Ok(unsafe { File::from_raw_fd(raw) })
}
fn check_path(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path
            .as_os_str()
            .as_bytes()
            .split(|b| *b == b'/')
            .any(|p| p == b".." || p == b".")
    {
        bail!("filesystem path must be absolute without dot components");
    }
    cstr(path.as_os_str())?;
    Ok(())
}
fn roots(paths: &[PathBuf]) -> Result<Vec<Root>> {
    paths
        .iter()
        .map(|p| {
            check_path(p)?;
            let canonical = p
                .canonicalize()
                .map_err(|_| anyhow!("configured filesystem root unavailable"))?;
            // Walk root setup without intermediate symlinks as well.
            let mut dir = open_at(
                libc::AT_FDCWD,
                OsStr::new("/"),
                libc::O_RDONLY | libc::O_DIRECTORY,
                0,
            )?;
            for component in canonical.components() {
                if let Component::Normal(part) = component {
                    dir = open_at(dir.as_raw_fd(), part, libc::O_RDONLY | libc::O_DIRECTORY, 0)?;
                }
            }
            Ok(Root {
                path: canonical,
                dir,
            })
        })
        .collect()
}

fn delete_hash(expected: &str) -> Result<[u8; 32]> {
    if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("expected_sha256 must be exactly 64 hexadecimal characters");
    }
    let mut hash = [0; 32];
    for (out, pair) in hash.iter_mut().zip(expected.as_bytes().as_chunks::<2>().0) {
        let digit = |b: u8| {
            if b <= b'9' {
                b - b'0'
            } else {
                b.to_ascii_lowercase() - b'a' + 10
            }
        };
        *out = digit(pair[0]) * 16 + digit(pair[1]);
    }
    Ok(hash)
}
fn same_delete_metadata(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    a.dev() == b.dev()
        && a.ino() == b.ino()
        && a.mode() == b.mode()
        && a.uid() == b.uid()
        && a.gid() == b.gid()
        && a.nlink() == b.nlink()
        && a.len() == b.len()
        && a.mtime() == b.mtime()
        && a.mtime_nsec() == b.mtime_nsec()
}
fn delete_snapshot(
    parent: &File,
    name: &OsStr,
    expected: &[u8; 32],
    limit: usize,
) -> Result<std::fs::Metadata> {
    let file = open_at(
        parent.as_raw_fd(),
        name,
        libc::O_RDONLY | libc::O_NONBLOCK,
        0,
    )?;
    let before = file
        .metadata()
        .map_err(|_| anyhow!("delete metadata unavailable"))?;
    if !before.is_file()
        || before.uid() != unsafe { libc::geteuid() }
        || before.nlink() != 1
        || before.len() > limit as u64
    {
        bail!("delete file type, ownership, links or size denied");
    }
    let mut hash = Sha256::new();
    let mut reader = (&file).take(limit as u64 + 1);
    let mut buf = [0u8; 8192];
    let mut count = 0usize;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|_| anyhow!("delete bounded read failed"))?;
        if n == 0 {
            break;
        }
        count += n;
        if count > limit {
            bail!("delete file exceeds configured limit");
        }
        hash.update(&buf[..n]);
    }
    let actual = hash.finalize();
    // No digest-dependent early exit. Hash values and file contents are never returned.
    let mut difference = 0u8;
    for i in 0..32 {
        difference |= actual[i] ^ expected[i];
    }
    let after = file
        .metadata()
        .map_err(|_| anyhow!("delete metadata unavailable"))?;
    if difference != 0
        || count as u64 != after.len()
        || !same_delete_metadata(&before, &after)
        || before.ctime() != after.ctime()
        || before.ctime_nsec() != after.ctime_nsec()
    {
        bail!("delete content precondition failed");
    }
    // Ensure the pathname still names the descriptor that was hashed.
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    let name = cstr(name)?;
    if unsafe {
        libc::fstatat(
            parent.as_raw_fd(),
            name.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    } != 0
    {
        bail!("delete entry unavailable");
    }
    let stat = unsafe { stat.assume_init() };
    if stat.st_dev != after.dev()
        || stat.st_ino != after.ino()
        || stat.st_mode != after.mode()
        || stat.st_nlink != 1
        || stat.st_uid != after.uid()
    {
        bail!("delete entry changed during validation");
    }
    Ok(after)
}
fn rename_noreplace(from: &File, source: &CString, to: &File, target: &CString) -> Result<()> {
    // Linux-only: never fall back to an overwriting rename or check-then-rename.
    if unsafe {
        libc::renameat2(
            from.as_raw_fd(),
            source.as_ptr(),
            to.as_raw_fd(),
            target.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    } != 0
    {
        bail!("delete non-overwriting rename unavailable");
    }
    Ok(())
}

impl Policy {
    pub fn new(config: PolicyConfig) -> Result<Self> {
        if config.max_file_bytes > MAX_FILE_BYTES {
            bail!("max_file_bytes exceeds hard limit");
        }
        let reads = roots(&config.read_roots)?;
        let writes = roots(&config.write_roots)?;
        let mut origins = HashSet::new();
        for origin in &config.allowed_origins {
            let url = parse_url(origin)?;
            let suffix = origin
                .split_once("://")
                .map(|(_, s)| s.find('/').map(|i| &s[i..]).unwrap_or(""))
                .unwrap_or("invalid");
            if !matches!(suffix, "" | "/") || url.path() != "/" || url.query().is_some() {
                bail!("allowlist entries must be exact HTTP origins");
            }
            origins.insert(origin_key(&url)?);
        }
        for argv in &config.allowed_commands {
            if argv.is_empty()
                || !Path::new(&argv[0]).is_absolute()
                || argv.iter().any(|a| a.as_bytes().contains(&0))
            {
                bail!("command allowlist requires absolute executable and valid exact argv");
            }
        }
        Ok(Self {
            inner: Arc::new(Inner {
                config,
                reads,
                writes,
                origins,
            }),
        })
    }
    pub fn config(&self) -> &PolicyConfig {
        &self.inner.config
    }
    pub fn check_command(&self, argv: &[String]) -> Result<()> {
        if !self.inner.config.shell_enabled {
            bail!("shell capability disabled");
        }
        self.check_argv(argv)
    }
    pub fn check_mcp_command(&self, argv: &[String]) -> Result<()> {
        if !self.inner.config.mcp_enabled {
            bail!("MCP capability disabled");
        }
        self.check_argv(argv)
    }
    fn check_argv(&self, argv: &[String]) -> Result<()> {
        if argv.is_empty()
            || !self
                .inner
                .config
                .allowed_commands
                .iter()
                .any(|allowed| allowed == argv)
        {
            bail!("command denied by exact argv policy");
        }
        Ok(())
    }
    fn parent(&self, path: &Path, writing: bool) -> Result<(File, std::ffi::OsString)> {
        check_path(path)?;
        let roots = if writing {
            &self.inner.writes
        } else {
            &self.inner.reads
        };
        let root = roots
            .iter()
            .filter(|r| path.starts_with(&r.path))
            .max_by_key(|r| r.path.components().count())
            .ok_or_else(|| anyhow!("filesystem path outside allowed roots"))?;
        let relative = path
            .strip_prefix(&root.path)
            .map_err(|_| anyhow!("filesystem path denied"))?;
        let parts: Vec<_> = relative
            .components()
            .map(|c| match c {
                Component::Normal(s) => Ok(s),
                _ => Err(anyhow!("invalid filesystem component")),
            })
            .collect::<Result<_>>()?;
        let (name, parents) = parts
            .split_last()
            .ok_or_else(|| anyhow!("root itself is not a file"))?;
        let mut dir = root
            .dir
            .try_clone()
            .map_err(|_| anyhow!("filesystem descriptor unavailable"))?;
        for part in parents {
            dir = open_at(dir.as_raw_fd(), part, libc::O_RDONLY | libc::O_DIRECTORY, 0)?;
        }
        Ok((dir, name.to_os_string()))
    }
    /// Bounded immediate entry names, not recursive. Symlink directories are denied.
    pub fn list_files(&self, path: &Path) -> Result<Vec<String>> {
        check_path(path)?;
        let root = self.inner.reads.iter().find(|r| r.path == path);
        let dir = if let Some(root) = root {
            open_at(
                root.dir.as_raw_fd(),
                OsStr::new("."),
                libc::O_RDONLY | libc::O_DIRECTORY,
                0,
            )?
        } else {
            let (parent, name) = self.parent(path, false)?;
            open_at(
                parent.as_raw_fd(),
                &name,
                libc::O_RDONLY | libc::O_DIRECTORY,
                0,
            )?
        };
        use std::os::fd::IntoRawFd;
        let fd = dir.into_raw_fd();
        let raw = unsafe { libc::fdopendir(fd) };
        if raw.is_null() {
            unsafe {
                libc::close(fd);
            }
            bail!("directory listing unavailable");
        }
        struct DirGuard(*mut libc::DIR);
        impl Drop for DirGuard {
            fn drop(&mut self) {
                unsafe {
                    libc::closedir(self.0);
                }
            }
        }
        let guard = DirGuard(raw);
        let mut entries = Vec::new();
        let mut bytes = 0;
        loop {
            unsafe {
                *libc::__errno_location() = 0;
            }
            let entry = unsafe { libc::readdir(guard.0) };
            if entry.is_null() {
                if std::io::Error::last_os_error().raw_os_error().unwrap_or(0) != 0 {
                    bail!("directory listing failed");
                }
                break;
            }
            let name = unsafe { std::ffi::CStr::from_ptr((*entry).d_name.as_ptr()) };
            if matches!(name.to_bytes(), b"." | b"..") {
                continue;
            }
            let name = name
                .to_str()
                .map_err(|_| anyhow!("directory contains non-UTF8 name"))?
                .to_owned();
            bytes += name.len();
            if entries.len() >= 4096 || bytes > 256 * 1024 {
                bail!("directory listing exceeds limit");
            }
            entries.push(name);
        }
        entries.sort();
        Ok(entries)
    }
    pub fn read_file(&self, path: impl AsRef<Path>) -> Result<Vec<u8>> {
        let (parent, name) = self.parent(path.as_ref(), false)?;
        let file = open_at(
            parent.as_raw_fd(),
            &name,
            libc::O_RDONLY | libc::O_NONBLOCK,
            0,
        )?;
        let meta = file
            .metadata()
            .map_err(|_| anyhow!("filesystem metadata unavailable"))?;
        if !meta.is_file()
            || meta.nlink() != 1
            || meta.len() > self.inner.config.max_file_bytes as u64
        {
            bail!("file type or size denied");
        }
        let mut bytes = Vec::new();
        file.take(self.inner.config.max_file_bytes as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| anyhow!("file read failed"))?;
        if bytes.len() > self.inner.config.max_file_bytes {
            bail!("file exceeds configured limit");
        }
        Ok(bytes)
    }
    /// Delete one exact, owned, single-link regular file within write roots.
    /// The digest is mandatory and checked before any namespace mutation and again
    /// after quarantining the entry in a private directory. No recursive deletion.
    /// This is not an atomic compare-and-unlink against malicious same-UID actors:
    /// they can enter the private journal or retain a writable file descriptor.
    /// Any error after quarantine may mean an uncertain outcome, never "no effect".
    pub fn delete_file(&self, path: impl AsRef<Path>, expected_sha256: &str) -> Result<()> {
        self.delete_file_inner(path.as_ref(), expected_sha256, || {}, || {})
    }
    fn delete_file_inner(
        &self,
        path: &Path,
        expected_sha256: &str,
        before_move: impl FnOnce(),
        after_move: impl FnOnce(),
    ) -> Result<()> {
        let expected = delete_hash(expected_sha256)?;
        if path.components().any(
            |p| matches!(p, Component::Normal(n) if n.as_bytes().starts_with(b".missy-delete-")),
        ) {
            bail!("private delete journals are protected");
        }
        let (parent, name) = self.parent(path, true)?;
        let parent_meta = parent
            .metadata()
            .map_err(|_| anyhow!("delete parent unavailable"))?;
        if parent_meta.uid() != unsafe { libc::geteuid() } || parent_meta.mode() & 0o022 != 0 {
            bail!("delete requires an owned parent without group or world write permission");
        }
        let original =
            delete_snapshot(&parent, &name, &expected, self.inner.config.max_file_bytes)?;
        let journal_name = CString::new(format!(".missy-delete-{}", uuid::Uuid::new_v4()))?;
        if unsafe { libc::mkdirat(parent.as_raw_fd(), journal_name.as_ptr(), 0o700) } != 0 {
            bail!("private delete journal unavailable");
        }
        let cleanup = || unsafe {
            libc::unlinkat(
                parent.as_raw_fd(),
                journal_name.as_ptr(),
                libc::AT_REMOVEDIR,
            )
        };
        let journal = match open_at(
            parent.as_raw_fd(),
            OsStr::from_bytes(journal_name.as_bytes()),
            libc::O_RDONLY | libc::O_DIRECTORY,
            0,
        ) {
            Ok(file) => file,
            Err(error) => {
                cleanup();
                return Err(error);
            }
        };
        if unsafe { libc::fchmod(journal.as_raw_fd(), 0o700) } != 0 {
            cleanup();
            bail!("private delete journal permissions unavailable");
        }
        let source = cstr(&name)?;
        let entry = CString::new("entry")?;
        before_move();
        if let Err(error) = rename_noreplace(&parent, &source, &journal, &entry) {
            cleanup();
            return Err(error);
        }
        after_move();
        let checked = delete_snapshot(
            &journal,
            OsStr::new("entry"),
            &expected,
            self.inner.config.max_file_bytes,
        )
        .and_then(|moved| {
            if !same_delete_metadata(&original, &moved) {
                bail!("delete entry identity changed");
            }
            Ok(())
        });
        if checked.is_err() {
            // Restore the moved entry, even a symlink, without following it and
            // without replacing anything created at the original name meanwhile.
            if rename_noreplace(&journal, &entry, &parent, &source).is_err() {
                let _ = journal.sync_all();
                let _ = parent.sync_all();
                bail!(
                    "delete outcome uncertain; quarantined entry preserved in {} for recovery",
                    journal_name.to_string_lossy()
                );
            }
            cleanup();
            parent
                .sync_all()
                .map_err(|_| anyhow!("delete restore sync failed; outcome uncertain"))?;
            bail!("delete validation failed; moved entry restored without overwrite");
        }
        if unsafe { libc::unlinkat(journal.as_raw_fd(), entry.as_ptr(), 0) } != 0 {
            bail!(
                "delete outcome uncertain; quarantined entry retained in {}",
                journal_name.to_string_lossy()
            );
        }
        journal
            .sync_all()
            .map_err(|_| anyhow!("delete journal sync failed; outcome uncertain"))?;
        if cleanup() != 0 {
            bail!("delete journal cleanup failed; outcome uncertain");
        }
        parent
            .sync_all()
            .map_err(|_| anyhow!("directory sync failed after delete; outcome uncertain"))?;
        Ok(())
    }
    /// Atomic replace, always mode 0600 and current effective user ownership.
    /// Parent directories must already exist. Symlink destinations are rejected.
    pub fn write_file(&self, path: impl AsRef<Path>, data: &[u8]) -> Result<()> {
        if data.len() > self.inner.config.max_file_bytes {
            bail!("file exceeds configured limit");
        }
        let (parent, name) = self.parent(path.as_ref(), true)?;
        let destination = cstr(&name)?;
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        // renameat replaces an entry, never follows it, even if swapped after this check.
        let rc = unsafe {
            libc::fstatat(
                parent.as_raw_fd(),
                destination.as_ptr(),
                stat.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        if rc == 0 {
            let stat = unsafe { stat.assume_init() };
            if stat.st_mode & libc::S_IFMT != libc::S_IFREG
                || stat.st_nlink != 1
                || stat.st_uid != unsafe { libc::geteuid() }
            {
                bail!("write destination type or owner denied");
            }
        } else if std::io::Error::last_os_error().raw_os_error() != Some(libc::ENOENT) {
            bail!("write destination unavailable");
        }
        let temp = std::ffi::OsString::from(format!(".missy-write-{}", uuid::Uuid::new_v4()));
        let temp_c = cstr(&temp)?;
        let mut file = open_at(
            parent.as_raw_fd(),
            &temp,
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
            0o600,
        )?;
        let result = (|| -> Result<()> {
            // Do not let an unusually restrictive inherited umask change the
            // promised owner-only mode of the replacement inode.
            if unsafe { libc::fchmod(file.as_raw_fd(), 0o600) } != 0 {
                bail!("private file mode setup failed");
            }
            file.write_all(data)
                .map_err(|_| anyhow!("atomic file write failed"))?;
            file.sync_all()
                .map_err(|_| anyhow!("atomic file sync failed"))?;
            if unsafe {
                libc::renameat(
                    parent.as_raw_fd(),
                    temp_c.as_ptr(),
                    parent.as_raw_fd(),
                    destination.as_ptr(),
                )
            } != 0
            {
                bail!("atomic file replace failed");
            }
            parent
                .sync_all()
                .map_err(|_| anyhow!("directory sync failed after write"))?;
            Ok(())
        })();
        if result.is_err() {
            unsafe {
                libc::unlinkat(parent.as_raw_fd(), temp_c.as_ptr(), 0);
            }
        }
        result
    }
}

fn parse_url(raw: &str) -> Result<Url> {
    if raw.len() > 8192
        || !raw.contains("://")
        || raw
            .chars()
            .any(|c| c.is_whitespace() || c.is_control() || c == '\\')
    {
        bail!("invalid HTTP URL");
    }
    let url = Url::parse(raw).map_err(|_| anyhow!("invalid HTTP URL"))?;
    let authority = raw
        .split_once("://")
        .map(|(_, rest)| rest.split(['/', '?', '#']).next().unwrap_or(""))
        .unwrap_or("");
    if authority.is_empty()
        || !matches!(url.scheme(), "http" | "https")
        || url.host().is_none()
        || url.port() == Some(0)
        || !url.username().is_empty()
        || url.password().is_some()
        || authority.contains('@')
        || url.fragment().is_some()
    {
        bail!("HTTP URL scheme, authority or fragment denied");
    }
    if let Some(Host::Domain(host)) = url.host()
        && (host.ends_with('.')
            || host.split('.').any(|label| {
                label.is_empty()
                    || label.starts_with('-')
                    || label.ends_with('-')
                    || !label
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-')
            }))
    {
        bail!("invalid exact HTTP hostname");
    }
    Ok(url)
}
fn origin_key(url: &Url) -> Result<String> {
    Ok(format!(
        "{}://{}:{}",
        url.scheme(),
        url.host_str().ok_or_else(|| anyhow!("HTTP host missing"))?,
        url.port_or_known_default()
            .ok_or_else(|| anyhow!("HTTP port missing"))?
    ))
}
fn public_v4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(a == 0
        || a == 10
        || a == 127
        || a >= 224
        || (a == 100 && (64..=127).contains(&b))
        || (a == 169 && b == 254)
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && (b == 168 || (b == 0 && (c == 0 || c == 2)) || (b == 88 && c == 99)))
        || (a == 198 && (b == 18 || b == 19 || (b == 51 && c == 100)))
        || (a == 203 && b == 0 && c == 113))
}
fn public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => public_v4(ip),
        IpAddr::V6(ip) => {
            let s = ip.segments();
            // Conservative global unicast, excluding transition/documentation ranges.
            s[0] & 0xe000 == 0x2000
                && !(s[0] == 0x2001 && (s[1] < 0x0200 || s[1] == 0x0db8))
                && s[0] != 0x2002
                && !(s[0] == 0x3fff && s[1] < 0x1000)
        }
    }
}
fn validate_answers(addrs: &[SocketAddr], private: bool) -> Result<()> {
    if addrs.is_empty() || (!private && addrs.iter().any(|a| !public_ip(a.ip()))) {
        bail!("HTTP address denied by network policy");
    }
    if addrs.iter().any(|a| match a.ip() {
        IpAddr::V4(ip) => ip.is_unspecified() || ip.is_multicast() || ip.is_broadcast(),
        IpAddr::V6(ip) => ip.is_unspecified() || ip.is_multicast(),
    }) {
        bail!("non-unicast HTTP address denied");
    }
    if addrs.iter().any(|a| public_ip(a.ip())) && addrs.iter().any(|a| !public_ip(a.ip())) {
        bail!("mixed public and private HTTP addresses denied");
    }
    if addrs
        .iter()
        .any(|a| matches!(a.ip(), IpAddr::V6(ip) if ip.to_ipv4_mapped().is_some()))
    {
        bail!("mapped IPv6 HTTP address denied");
    }
    Ok(())
}

#[derive(Clone)]
pub struct PolicyHttpClient {
    policy: Policy,
    timeout: Duration,
    chunk_timeout: Duration,
}
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}
impl std::fmt::Debug for HttpResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpResponse")
            .field("status", &self.status)
            .field("header_count", &self.headers.len())
            .field("body_bytes", &self.body.len())
            .finish()
    }
}
/// Bounded streaming response from the policy gateway, never a raw transport.
/// Drop closes the response. Cancelling a pending `chunk()` also closes it.
pub struct PolicyHttpStream {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    response: Option<reqwest::Response>,
    deadline: tokio::time::Instant,
    chunk_timeout: Duration,
    received: usize,
}
impl std::fmt::Debug for PolicyHttpStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyHttpStream")
            .field("status", &self.status)
            .field("header_count", &self.headers.len())
            .field("received_bytes", &self.received)
            .field("closed", &self.response.is_none())
            .finish()
    }
}
impl PolicyHttpStream {
    fn new(
        response: reqwest::Response,
        deadline: tokio::time::Instant,
        chunk_timeout: Duration,
    ) -> Result<Self> {
        if response
            .content_length()
            .is_some_and(|n| n > MAX_HTTP_RESPONSE_BYTES as u64)
        {
            bail!("HTTP response exceeds limit");
        }
        let status = response.status().as_u16();
        let mut headers = Vec::new();
        let mut header_bytes = 0;
        for (index, (k, v)) in response.headers().iter().enumerate() {
            header_bytes += k.as_str().len() + v.len();
            if header_bytes > 32 * 1024 || index >= 128 {
                bail!("HTTP response headers exceed limit");
            }
            if matches!(
                k.as_str(),
                "set-cookie" | "www-authenticate" | "proxy-authenticate"
            ) {
                continue;
            }
            if let Ok(value) = v.to_str() {
                headers.push((k.to_string(), value.to_owned()));
            }
        }
        Ok(Self {
            status,
            headers,
            response: Some(response),
            deadline,
            chunk_timeout,
            received: 0,
        })
    }
    /// Read one transport chunk, with cumulative 4 MiB and absolute lifetime bounds.
    /// An error or cancellation is terminal; subsequent calls return `None`.
    /// The total deadline includes DNS, connection setup and time between reads.
    pub async fn chunk(&mut self) -> Result<Option<Vec<u8>>> {
        // Move ownership into this future: cancelling a pending read must not leave
        // an unattended live connection in the outer stream.
        let Some(mut response) = self.response.take() else {
            return Ok(None);
        };
        let now = tokio::time::Instant::now();
        if now >= self.deadline {
            bail!("HTTP response stream timed out");
        }
        let read_deadline = self.deadline.min(now + self.chunk_timeout);
        let next = tokio::time::timeout_at(read_deadline, response.chunk())
            .await
            .map_err(|_| anyhow!("HTTP response stream timed out"))?
            .map_err(|_| anyhow!("HTTP response transfer failed"))?;
        // A ready transport may win a timeout poll at the boundary. Never deliver
        // bytes after the absolute policy lifetime, including buffered bytes.
        if tokio::time::Instant::now() >= self.deadline {
            bail!("HTTP response stream timed out");
        }
        let Some(chunk) = next else {
            return Ok(None);
        };
        if self.received.saturating_add(chunk.len()) > MAX_HTTP_RESPONSE_BYTES {
            bail!("HTTP response exceeds limit");
        }
        self.received += chunk.len();
        self.response = Some(response);
        Ok(Some(chunk.to_vec()))
    }
}
pub type PolicyWebSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
impl PolicyHttpClient {
    pub fn new(config: PolicyConfig) -> Result<Self> {
        Ok(Self::from_policy(Policy::new(config)?))
    }
    pub fn from_policy(policy: Policy) -> Self {
        Self {
            policy,
            timeout: Duration::from_secs(HTTP_TIMEOUT_SECONDS),
            chunk_timeout: Duration::from_secs(HTTP_TIMEOUT_SECONDS),
        }
    }
    pub fn config(&self) -> &PolicyConfig {
        self.policy.config()
    }
    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self> {
        if timeout < Duration::from_secs(1) || timeout > Duration::from_secs(600) {
            bail!("HTTP timeout must be 1..600 seconds");
        }
        self.timeout = timeout;
        Ok(self)
    }
    /// Streaming idle-read bound, independent of the absolute `with_timeout` bound.
    /// Legacy SSE long polls may opt into up to 600 seconds; default is 30 seconds.
    pub fn with_chunk_timeout(mut self, timeout: Duration) -> Result<Self> {
        if timeout < Duration::from_secs(1) || timeout > Duration::from_secs(600) {
            bail!("HTTP chunk timeout must be 1..600 seconds");
        }
        self.chunk_timeout = timeout;
        Ok(self)
    }
    pub async fn get(&self, url: &str) -> Result<HttpResponse> {
        self.request("GET", url, vec![], None).await
    }
    /// WebSocket handshake on an explicitly allowed equivalent HTTP(S) origin.
    /// Once returned the caller owns cancellation/heartbeats and stream lifetime.
    /// TLS uses the original hostname; TCP uses only the validated DNS snapshot.
    pub async fn connect_websocket(
        &self,
        raw: &str,
        headers: Vec<(String, String)>,
    ) -> Result<PolicyWebSocket> {
        tokio::time::timeout(self.timeout, self.connect_websocket_inner(raw, headers))
            .await
            .map_err(|_| anyhow!("WebSocket handshake timed out"))?
    }
    async fn connect_websocket_inner(
        &self,
        raw: &str,
        headers: Vec<(String, String)>,
    ) -> Result<PolicyWebSocket> {
        use tokio_tungstenite::tungstenite::{
            client::IntoClientRequest,
            http::header::{HeaderName, HeaderValue},
            protocol::WebSocketConfig,
        };
        let http_url = if let Some(rest) = raw.strip_prefix("wss://") {
            format!("https://{rest}")
        } else if let Some(rest) = raw.strip_prefix("ws://") {
            format!("http://{rest}")
        } else {
            bail!("WebSocket scheme denied");
        };
        let url = parse_url(&http_url)?;
        let addrs = self.approved_addresses(&url).await?;
        if headers.len() > 64
            || headers
                .iter()
                .map(|(k, v)| k.len().saturating_add(v.len()))
                .sum::<usize>()
                > 32 * 1024
        {
            bail!("WebSocket headers exceed limit");
        }
        let mut request = raw
            .into_client_request()
            .map_err(|_| anyhow!("invalid WebSocket request"))?;
        let mut seen = HashSet::new();
        for (name, value) in headers {
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|_| anyhow!("invalid WebSocket header"))?;
            if name.as_str().starts_with("sec-websocket-")
                || matches!(
                    name.as_str(),
                    "host"
                        | "connection"
                        | "upgrade"
                        | "transfer-encoding"
                        | "content-length"
                        | "proxy-authorization"
                        | "proxy-connection"
                        | "te"
                        | "trailer"
                )
                || !seen.insert(name.clone())
            {
                bail!("WebSocket routing or framing header denied");
            }
            let mut value =
                HeaderValue::from_str(&value).map_err(|_| anyhow!("invalid WebSocket header"))?;
            value.set_sensitive(true);
            request.headers_mut().insert(name, value);
        }
        let mut stream = None;
        for address in addrs {
            if let Ok(Ok(tcp)) = tokio::time::timeout(
                Duration::from_secs(10),
                tokio::net::TcpStream::connect(address),
            )
            .await
            {
                stream = Some(tcp);
                break;
            }
        }
        let stream = stream.ok_or_else(|| anyhow!("WebSocket TCP connection failed"))?;
        let config = WebSocketConfig::default()
            .max_message_size(Some(MAX_HTTP_RESPONSE_BYTES))
            .max_frame_size(Some(MAX_HTTP_RESPONSE_BYTES))
            .max_write_buffer_size(MAX_HTTP_REQUEST_BYTES + 128 * 1024);
        let (socket, _) =
            tokio_tungstenite::client_async_tls_with_config(request, stream, Some(config), None)
                .await
                .map_err(|_| anyhow!("WebSocket TLS or handshake failed"))?;
        Ok(socket)
    }
    async fn approved_addresses(&self, url: &Url) -> Result<Vec<SocketAddr>> {
        if !self.policy.inner.origins.contains(&origin_key(url)?) {
            bail!("HTTP origin denied by exact allowlist");
        }
        let port = url
            .port_or_known_default()
            .ok_or_else(|| anyhow!("HTTP port missing"))?;
        let addrs: Vec<SocketAddr> = match url.host().ok_or_else(|| anyhow!("HTTP host missing"))? {
            Host::Ipv4(ip) => vec![SocketAddr::new(ip.into(), port)],
            Host::Ipv6(ip) => vec![SocketAddr::new(ip.into(), port)],
            Host::Domain(name) => tokio::net::lookup_host((name, port))
                .await
                .map_err(|_| anyhow!("HTTP DNS resolution failed"))?
                .collect(),
        };
        validate_answers(&addrs, self.policy.inner.config.allow_private_network)?;
        Ok(addrs)
    }
    pub async fn request_json(
        &self,
        method: &str,
        url: &str,
        mut headers: Vec<(String, String)>,
        body: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let body = body
            .map(|v| serde_json::to_vec(&v))
            .transpose()
            .map_err(|_| anyhow!("HTTP JSON request encoding failed"))?;
        if body.is_some()
            && !headers
                .iter()
                .any(|(k, _)| k.eq_ignore_ascii_case("content-type"))
        {
            headers.push(("content-type".into(), "application/json".into()));
        }
        let response = self.request(method, url, headers, body).await?;
        if !(200..300).contains(&response.status) {
            bail!("HTTP upstream returned status {}", response.status);
        }
        serde_json::from_slice(&response.body)
            .map_err(|_| anyhow!("HTTP response was not valid JSON"))
    }
    pub async fn request(
        &self,
        method: &str,
        raw: &str,
        headers: Vec<(String, String)>,
        body: Option<Vec<u8>>,
    ) -> Result<HttpResponse> {
        tokio::time::timeout(self.timeout, self.request_inner(method, raw, headers, body))
            .await
            .map_err(|_| anyhow!("HTTP request timed out"))?
    }
    /// Start a streaming request through the same origin, DNS and header policy
    /// as buffered requests. No redirects, proxies or unvalidated DNS fallback.
    pub async fn request_stream(
        &self,
        method: &str,
        raw: &str,
        headers: Vec<(String, String)>,
        body: Option<Vec<u8>>,
    ) -> Result<PolicyHttpStream> {
        let deadline = tokio::time::Instant::now() + self.timeout;
        let response =
            tokio::time::timeout_at(deadline, self.send_inner(method, raw, headers, body, true))
                .await
                .map_err(|_| anyhow!("HTTP request timed out"))??;
        PolicyHttpStream::new(response, deadline, self.chunk_timeout)
    }
    async fn request_inner(
        &self,
        method: &str,
        raw: &str,
        headers: Vec<(String, String)>,
        body: Option<Vec<u8>>,
    ) -> Result<HttpResponse> {
        let deadline = tokio::time::Instant::now() + self.timeout;
        let response = self.send_inner(method, raw, headers, body, false).await?;
        // Buffered HTTP keeps its existing total timeout, not the streaming idle bound.
        let mut stream = PolicyHttpStream::new(response, deadline, self.timeout)?;
        let mut body = Vec::new();
        while let Some(chunk) = stream.chunk().await? {
            body.extend_from_slice(&chunk);
        }
        Ok(HttpResponse {
            status: stream.status,
            headers: stream.headers,
            body,
        })
    }
    async fn send_inner(
        &self,
        method: &str,
        raw: &str,
        headers: Vec<(String, String)>,
        body: Option<Vec<u8>>,
        streaming: bool,
    ) -> Result<reqwest::Response> {
        let url = parse_url(raw)?;
        if !self.policy.inner.origins.contains(&origin_key(&url)?) {
            bail!("HTTP origin denied by exact allowlist");
        }
        if body
            .as_ref()
            .is_some_and(|b| b.len() > MAX_HTTP_REQUEST_BYTES)
            || headers.len() > 64
            || headers
                .iter()
                .map(|(k, v)| k.len().saturating_add(v.len()))
                .sum::<usize>()
                > 32 * 1024
        {
            bail!("HTTP request exceeds limits");
        }
        let method = reqwest::Method::from_bytes(method.as_bytes())
            .map_err(|_| anyhow!("invalid HTTP method"))?;
        if !matches!(
            method.as_str(),
            "GET" | "HEAD" | "POST" | "PUT" | "PATCH" | "DELETE" | "OPTIONS"
        ) {
            bail!("HTTP method denied");
        }
        let mut header_map = reqwest::header::HeaderMap::new();
        for (name, value) in headers {
            let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
                .map_err(|_| anyhow!("invalid HTTP header"))?;
            if matches!(
                name.as_str(),
                "host"
                    | "connection"
                    | "transfer-encoding"
                    | "content-length"
                    | "proxy-authorization"
                    | "proxy-connection"
                    | "upgrade"
                    | "te"
                    | "trailer"
            ) {
                bail!("HTTP routing or framing header denied");
            }
            let mut value = reqwest::header::HeaderValue::from_str(&value)
                .map_err(|_| anyhow!("invalid HTTP header"))?;
            value.set_sensitive(true);
            if header_map.insert(name, value).is_some() {
                bail!("duplicate HTTP header denied");
            }
        }
        let host = url.host().ok_or_else(|| anyhow!("HTTP host missing"))?;
        let addrs = self.approved_addresses(&url).await?;
        let mut builder = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(10))
            .referer(false)
            .pool_max_idle_per_host(0);
        // Streaming uses the absolute policy deadline and per-read timeout rather
        // than reqwest's body-wide timer, which is unsuitable for long-lived SSE.
        if !streaming {
            builder = builder.timeout(self.timeout);
        }
        if let Host::Domain(name) = host {
            builder = builder.resolve_to_addrs(name, &addrs);
        }
        let client = builder
            .build()
            .map_err(|_| anyhow!("HTTP transport initialization failed"))?;
        let mut request = client.request(method, url).headers(header_map);
        if let Some(body) = body {
            request = request.body(body);
        }
        request
            .send()
            .await
            .map_err(|_| anyhow!("HTTP transport failed"))
    }
}

#[cfg(test)]
mod delete_tests;
#[cfg(test)]
mod tests;
