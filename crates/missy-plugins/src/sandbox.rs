//! Dedicated child confinement. Never restricts the service calling thread.
use crate::{CommandTemplate, PluginConfig, PluginManifest};
use anyhow::{Context, Result, ensure};
use missy_core::ToolContext;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    ffi::CString,
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::{
            ffi::OsStrExt,
            fs::{MetadataExt, OpenOptionsExt},
            process::CommandExt,
        },
    },
    path::{Component, Path},
    process::{Child, Command, Stdio},
    time::Duration,
};

pub(crate) fn validate_root(p: &Path) -> Result<()> {
    ensure!(
        p.is_absolute() && p != Path::new("/") && !p.as_os_str().as_bytes().contains(&0),
        "root must be bounded absolute path"
    );
    ensure!(
        !p.as_os_str()
            .as_bytes()
            .split(|b| *b == b'/')
            .any(|s| s == b"." || s == b".."),
        "dot path denied"
    );
    let mut current = std::path::PathBuf::from("/");
    for c in p.components() {
        if let Component::Normal(s) = c {
            current.push(s);
            let m = std::fs::symlink_metadata(&current)?;
            ensure!(!m.file_type().is_symlink(), "symlink root component denied");
        }
    }
    ensure!(p.is_dir(), "root must be directory");
    Ok(())
}
pub(crate) fn validate_executable_path(p: &Path) -> Result<()> {
    ensure!(
        p.is_absolute() && p.file_name().is_some() && !p.as_os_str().as_bytes().contains(&0),
        "absolute executable required"
    );
    validate_root(p.parent().unwrap())?;
    let name = p
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_ascii_lowercase();
    ensure!(
        ![
            "sh", "bash", "dash", "zsh", "fish", "env", "busybox", "python", "python3", "perl",
            "ruby", "node", "lua", "php", "java", "rustc", "cargo", "cc", "gcc", "clang", "ld",
            "ld.so"
        ]
        .contains(&name.as_str())
            && !name.starts_with("python")
            && !name.starts_with("ld-linux"),
        "interpreters, shells and compilers are not plugins"
    );
    Ok(())
}
/// Hash into sealed memfd, then execute that very image. No hash/exec path race.
pub(crate) fn pinned_executable(c: &CommandTemplate) -> Result<File> {
    validate_executable_path(&c.executable)?;
    pinned_native(c, 32 * 1024 * 1024)
}
pub(crate) fn pinned_native(c: &CommandTemplate, max: usize) -> Result<File> {
    validate_root(
        c.executable
            .parent()
            .ok_or_else(|| anyhow::anyhow!("executable parent required"))?,
    )?;
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(&c.executable)?;
    let meta = f.metadata()?;
    ensure!(
        meta.is_file() && meta.len() <= max as u64 && meta.mode() & 0o111 != 0,
        "plugin must be bounded executable regular file"
    );
    let mut bytes = Vec::new();
    f.take(max as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= max
            && bytes.starts_with(b"\x7fELF")
            && bytes.get(4) == Some(&2)
            && bytes.get(5) == Some(&1),
        "native 64-bit little-endian ELF required; scripts rejected"
    );
    ensure!(
        hex::encode(Sha256::digest(&bytes)) == c.sha256,
        "executable SHA256 drift"
    );
    let name = CString::new("missy-approved-plugin")?;
    let fd =
        unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
    ensure!(fd >= 0, "memfd pin unavailable");
    let mut pinned = unsafe { File::from_raw_fd(fd) };
    pinned.write_all(&bytes)?;
    let seals = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
    ensure!(
        unsafe { libc::fcntl(fd, libc::F_ADD_SEALS, seals) } == 0,
        "ELF seal failed"
    );
    Ok(pinned)
}
pub(crate) fn check_available() -> Result<()> {
    let abi = unsafe {
        libc::syscall(
            libc::SYS_landlock_create_ruleset,
            std::ptr::null::<u8>(),
            0usize,
            1u32,
        )
    };
    ensure!(
        abi >= 3,
        "Landlock ABI >=3 required; no unconfined fallback"
    );
    Ok(())
}
#[repr(C)]
struct Attr {
    fs: u64,
}
#[repr(C, packed)]
struct Rule {
    access: u64,
    fd: i32,
}
fn landlock(reads: &[std::path::PathBuf], temp: &Path) -> Result<File> {
    landlock_mode(reads, temp, false)
}
pub(crate) fn build_landlock(reads: &[std::path::PathBuf], temp: &Path) -> Result<File> {
    landlock_mode(reads, temp, true)
}
fn landlock_mode(reads: &[std::path::PathBuf], temp: &Path, build: bool) -> Result<File> {
    check_available()?;
    let attr = Attr { fs: (1 << 15) - 1 };
    let raw = unsafe {
        libc::syscall(
            libc::SYS_landlock_create_ruleset,
            &attr as *const Attr,
            8usize,
            0u32,
        )
    };
    ensure!(raw >= 0, "Landlock creation failed");
    let rules = unsafe { File::from_raw_fd(raw as i32) };
    for (p, access) in reads
        .iter()
        .map(|p| (p.as_path(), (1 << 0) | (1 << 2) | (1 << 3)))
        .chain(std::iter::once((
            temp,
            (if build {
                (1 << 0) | (1 << 2) | (1 << 3)
            } else {
                0
            }) | (1 << 1)
                | (1 << 4)
                | (1 << 5)
                | (1 << 7)
                | (1 << 8)
                | (1 << 9)
                | (1 << 10)
                | (1 << 12)
                | (1 << 13)
                | (1 << 14),
        )))
    {
        validate_root(p)?;
        let c = CString::new(p.as_os_str().as_bytes())?;
        let fd = unsafe {
            libc::open(
                c.as_ptr(),
                libc::O_PATH | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        ensure!(fd >= 0, "sandbox root unavailable");
        let root = unsafe { File::from_raw_fd(fd) };
        let rule = Rule {
            access,
            fd: root.as_raw_fd(),
        };
        ensure!(
            unsafe {
                libc::syscall(
                    libc::SYS_landlock_add_rule,
                    rules.as_raw_fd(),
                    1u32,
                    &rule as *const Rule,
                    0u32,
                )
            } == 0,
            "Landlock root rule failed"
        );
    }
    Ok(rules)
}
pub(crate) fn filter() -> Vec<libc::sock_filter> {
    #[cfg(target_arch = "x86_64")]
    const ARCH: u32 = 0xc000003e;
    #[cfg(target_arch = "aarch64")]
    const ARCH: u32 = 0xc00000b7;
    let stmt = |code, k| libc::sock_filter {
        code,
        jt: 0,
        jf: 0,
        k,
    };
    let jump = |k, jt, jf| libc::sock_filter {
        code: 0x15,
        jt,
        jf,
        k,
    };
    let mut f = vec![
        stmt(0x20, 4),
        jump(ARCH, 1, 0),
        stmt(0x06, 0x80000000),
        stmt(0x20, 0),
    ];
    #[cfg(target_arch = "x86_64")]
    {
        f.push(libc::sock_filter {
            code: 0x45,
            jt: 0,
            jf: 1,
            k: 0x40000000,
        });
        f.push(stmt(0x06, 0x80000000));
    }
    let denied = [
        libc::SYS_socket,
        libc::SYS_socketpair,
        libc::SYS_connect,
        libc::SYS_bind,
        libc::SYS_listen,
        libc::SYS_accept,
        libc::SYS_accept4,
        libc::SYS_sendto,
        libc::SYS_sendmsg,
        libc::SYS_sendmmsg,
        libc::SYS_recvfrom,
        libc::SYS_recvmsg,
        libc::SYS_recvmmsg,
        libc::SYS_setsockopt,
        libc::SYS_getsockopt,
        libc::SYS_shutdown,
        libc::SYS_ptrace,
        libc::SYS_process_vm_readv,
        libc::SYS_process_vm_writev,
        libc::SYS_mount,
        libc::SYS_umount2,
        libc::SYS_unshare,
        libc::SYS_setns,
        libc::SYS_bpf,
        libc::SYS_keyctl,
        libc::SYS_open_by_handle_at,
        libc::SYS_io_uring_setup,
        libc::SYS_io_uring_enter,
        libc::SYS_io_uring_register,
        libc::SYS_pidfd_getfd,
        libc::SYS_setsid,
        libc::SYS_setpgid,
        libc::SYS_kill,
        libc::SYS_tkill,
        libc::SYS_tgkill,
    ];
    for n in denied {
        f.push(jump(n as u32, 0, 1));
        f.push(stmt(0x06, 0x00050000 | libc::EPERM as u32));
    }
    f.push(stmt(0x06, 0x7fff0000));
    f
}
struct Owned {
    child: Child,
    killed: bool,
}
impl Owned {
    fn kill(&mut self) {
        if !self.killed {
            unsafe {
                libc::kill(-(self.child.id() as i32), libc::SIGKILL);
            }
            self.killed = true;
        }
    }
}
impl Drop for Owned {
    fn drop(&mut self) {
        self.kill();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
pub(crate) async fn execute(
    m: &PluginManifest,
    argv: &[String],
    cfg: &PluginConfig,
    ctx: &ToolContext,
) -> Result<Value> {
    let pinned = pinned_executable(&m.command)?;
    let temp = tempfile::Builder::new().prefix("missy-plugin-").tempdir()?;
    let mut roots = m.capabilities.read_roots.clone();
    roots.extend(cfg.runtime_read_roots.clone());
    let rules = landlock(&roots, temp.path())?;
    let filters = filter();
    let mut command = Command::new(format!("/proc/self/fd/{}", pinned.as_raw_fd()));
    command
        .args(&argv[1..])
        .arg0(&argv[0])
        .env_clear()
        .current_dir(temp.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    unsafe {
        command.pre_exec(move || {
            let zero = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            if libc::setrlimit(libc::RLIMIT_CORE, &zero) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::syscall(libc::SYS_landlock_restrict_self, rules.as_raw_fd(), 0u32) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            let prog = libc::sock_fprog {
                len: filters.len() as u16,
                filter: filters.as_ptr() as *mut _,
            };
            if libc::prctl(libc::PR_SET_SECCOMP, 2, &prog) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::syscall(libc::SYS_close_range, 3u32, u32::MAX, 4u32) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    ensure!(
        !ctx.cancellation.is_cancelled(),
        "plugin cancelled before spawn"
    );
    let child = command
        .spawn()
        .context("plugin child sandbox/exec failed; no fallback")?;
    drop(pinned);
    let mut owned = Owned {
        child,
        killed: false,
    };
    let out = owned.child.stdout.take().unwrap();
    let err = owned.child.stderr.take().unwrap();
    use std::os::fd::IntoRawFd;
    let out = unsafe { File::from_raw_fd(out.into_raw_fd()) };
    let err = unsafe { File::from_raw_fd(err.into_raw_fd()) };
    let remaining = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(cfg.max_output_bytes));
    let work = async {
        tokio::try_join!(
            async {
                loop {
                    let done = {
                        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
                        ensure!(
                            unsafe {
                                libc::waitid(
                                    libc::P_PID,
                                    owned.child.id(),
                                    &mut info,
                                    libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                                )
                            } == 0,
                            "plugin waitid failed"
                        );
                        (unsafe { info.si_pid() }) != 0
                    };
                    if done {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                owned.kill();
                Ok::<_, anyhow::Error>(owned.child.wait()?)
            },
            capture(out, remaining.clone()),
            capture(err, remaining)
        )
    };
    let result = tokio::select! {biased;_ = ctx.cancellation.cancelled()=>Err(anyhow::anyhow!("plugin cancelled")),_ = tokio::time::sleep(Duration::from_millis(cfg.timeout_ms))=>Err(anyhow::anyhow!("plugin timed out")),r=work=>r};
    drop(owned);
    let (status, stdout, stderr) = result?;
    Ok(
        json!({"trust":"untrusted","status":status.code(),"success":status.success(),"stdout":String::from_utf8_lossy(&stdout),"stderr":String::from_utf8_lossy(&stderr),"sandbox":{"backend":"landlock+seccomp","network_denied":true,"sealed_elf":true}}),
    )
}
pub(crate) async fn compile_rust(
    compiler: File,
    cfg: &crate::RustBuildConfig,
    temp: &Path,
    source: &Path,
    linker: &Path,
    output: &Path,
    cancellation: tokio_util::sync::CancellationToken,
) -> Result<()> {
    let rules = build_landlock(&cfg.read_roots, temp)?;
    let filters = filter();
    let mut command = Command::new(format!("/proc/self/fd/{}", compiler.as_raw_fd()));
    command
        .args([
            "--edition=2024",
            "--crate-name",
            "missy_generated_plugin",
            "-C",
            "linker-flavor=ld.lld",
            "-C",
            "opt-level=0",
            "-C",
            "debuginfo=0",
            "-C",
            "codegen-units=1",
            "-C",
            "panic=abort",
        ])
        .arg("--sysroot")
        .arg(&cfg.sysroot)
        .arg("-C")
        .arg(format!("linker={}", linker.display()))
        .arg(source)
        .arg("-o")
        .arg(output)
        .env_clear()
        .env("TMPDIR", temp)
        .env("HOME", temp)
        .env("LD_LIBRARY_PATH", cfg.sysroot.join("lib"))
        .current_dir(temp)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    for root in &cfg.native_library_roots {
        command.arg("-L").arg(format!("native={}", root.display()));
    }
    for argument in &cfg.linker_args {
        command.arg("-C").arg(format!("link-arg={argument}"));
    }
    unsafe {
        command.pre_exec(move || {
            for (resource, size) in [
                (libc::RLIMIT_CORE, 0),
                (libc::RLIMIT_FSIZE, 64 * 1024 * 1024),
                (libc::RLIMIT_AS, 2 * 1024 * 1024 * 1024),
                (libc::RLIMIT_CPU, 120),
                (libc::RLIMIT_NOFILE, 256),
            ] {
                let limit = libc::rlimit {
                    rlim_cur: size,
                    rlim_max: size,
                };
                if libc::setrlimit(resource, &limit) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0
                || libc::syscall(libc::SYS_landlock_restrict_self, rules.as_raw_fd(), 0u32) < 0
            {
                return Err(std::io::Error::last_os_error());
            }
            let prog = libc::sock_fprog {
                len: filters.len() as u16,
                filter: filters.as_ptr() as *mut _,
            };
            if libc::prctl(libc::PR_SET_SECCOMP, 2, &prog) < 0
                || libc::syscall(libc::SYS_close_range, 3u32, u32::MAX, 4u32) < 0
            {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let child = command
        .spawn()
        .context("reviewed Rust compiler confinement/exec failed; no fallback")?;
    drop(compiler);
    let mut owned = Owned {
        child,
        killed: false,
    };
    use std::os::fd::IntoRawFd;
    let stdout = unsafe { File::from_raw_fd(owned.child.stdout.take().unwrap().into_raw_fd()) };
    let stderr = unsafe { File::from_raw_fd(owned.child.stderr.take().unwrap().into_raw_fd()) };
    let remaining = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(cfg.max_output_bytes));
    let work = async {
        tokio::try_join!(
            async {
                loop {
                    // Keep leader unreaped until group cleanup, preventing a reused
                    // numeric PID/process group from receiving our timeout kill.
                    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
                    ensure!(
                        unsafe {
                            libc::waitid(
                                libc::P_PID,
                                owned.child.id(),
                                &mut info,
                                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                            )
                        } == 0,
                        "compiler waitid failed"
                    );
                    if unsafe { info.si_pid() } != 0 {
                        owned.kill();
                        return Ok::<_, anyhow::Error>(owned.child.wait()?);
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            },
            capture(stdout, remaining.clone()),
            capture(stderr, remaining)
        )
    };
    let result = tokio::select! {biased;
        _=cancellation.cancelled()=>Err(anyhow::anyhow!("source build cancelled")),
        _=tokio::time::sleep(Duration::from_millis(cfg.timeout_ms))=>Err(anyhow::anyhow!("source build timed out")),
        result=work=>result
    };
    drop(owned);
    let (status, _stdout, _stderr) = result?;
    // Compiler diagnostics can contain source strings/include_str! secrets.
    // Never publish them as a trusted/operator-visible build error.
    ensure!(
        status.success(),
        "reviewed Rust compile failed; compiler diagnostics withheld (untrusted source)"
    );
    Ok(())
}

async fn capture(
    file: File,
    remaining: std::sync::Arc<std::sync::atomic::AtomicUsize>,
) -> Result<Vec<u8>> {
    use std::sync::atomic::Ordering;
    use tokio::io::unix::AsyncFd;
    ensure!(
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFL, libc::O_NONBLOCK) } == 0,
        "nonblocking pipe setup failed"
    );
    let pipe = unsafe { AsyncFd::register(file) }?;
    let mut result = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        let mut ready = pipe.readable().await?;
        let n = match ready.try_io(|p| {
            let n =
                unsafe { libc::read(p.get_ref().as_raw_fd(), buf.as_mut_ptr().cast(), buf.len()) };
            if n < 0 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(n as usize)
            }
        }) {
            Ok(n) => n?,
            Err(_) => continue,
        };
        if n == 0 {
            return Ok(result);
        }
        let mut left = remaining.load(Ordering::Relaxed);
        loop {
            let next = left
                .checked_sub(n)
                .ok_or_else(|| anyhow::anyhow!("plugin output limit exceeded"))?;
            match remaining.compare_exchange_weak(left, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(current) => left = current,
            }
        }
        result.extend_from_slice(&buf[..n]);
    }
}
