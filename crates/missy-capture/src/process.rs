//! Fork-child-only Landlock with exact device IOCTL grant. No parent confinement.
//! Process leader remains unreaped until group cleanup, avoiding recycled-PID
//! group kills. Dropping the future synchronously kills AND reaps its own group.
use crate::{CaptureConfig, ExecutablePin};
use anyhow::{Context, Result, ensure};
use missy_core::ToolContext;
use std::{
    ffi::CString,
    fs::{File, OpenOptions},
    io,
    os::{
        fd::{AsRawFd, FromRawFd, IntoRawFd},
        unix::{ffi::OsStrExt, fs::OpenOptionsExt, process::CommandExt},
    },
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::io::unix::AsyncFd;

#[repr(C)]
struct Ruleset {
    fs: u64,
    net: u64,
}
#[repr(C, packed)]
struct Beneath {
    access: u64,
    fd: i32,
}
const READ: u64 = (1 << 0) | (1 << 2) | (1 << 3);
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
struct Prepared(File);
impl Prepared {
    fn new(reads: &[PathBuf], stage: &Path, device: Option<&Path>) -> Result<Self> {
        ensure!(
            missy_sandbox::landlock_abi()? >= 5,
            "Landlock ABI>=5 required"
        );
        let attr = Ruleset {
            fs: (1 << 16) - 1,
            net: 3,
        };
        let fd = unsafe {
            libc::syscall(
                libc::SYS_landlock_create_ruleset,
                &attr,
                std::mem::size_of::<Ruleset>(),
                0u32,
            )
        };
        ensure!(
            fd >= 0,
            "Landlock creation failed: {}",
            io::Error::last_os_error()
        );
        let owned = unsafe { File::from_raw_fd(fd as i32) };
        let add = |path: &Path, access: u64| -> Result<()> {
            crate::config::canonical(path)?;
            let path_c = CString::new(path.as_os_str().as_bytes())?;
            let raw = unsafe {
                libc::open(
                    path_c.as_ptr(),
                    libc::O_PATH | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                )
            };
            ensure!(raw >= 0, "Landlock path unavailable");
            let file = unsafe { File::from_raw_fd(raw) };
            let is_dir = file.metadata()?.is_dir();
            let access = if is_dir {
                access
            } else {
                access & ((1 << 0) | (1 << 1) | (1 << 2) | (1 << 14) | (1 << 15))
            };
            let rule = Beneath {
                access,
                fd: file.as_raw_fd(),
            };
            ensure!(
                unsafe {
                    libc::syscall(
                        libc::SYS_landlock_add_rule,
                        owned.as_raw_fd(),
                        1u32,
                        &rule,
                        0u32,
                    )
                } == 0,
                "Landlock rule failed: {}",
                io::Error::last_os_error()
            );
            Ok(())
        };
        for root in reads {
            add(root, READ)?;
        }
        add(stage, READ | WRITE)?;
        if let Some(device) = device {
            // FFmpeg's v4l2 driver requires O_RDWR plus VIDIOC_* configuration.
            // Write/IOCTL rights exist ONLY for this approved exact node, never
            // /dev, microphones, other devices, or writable staging files.
            add(device, (1 << 1) | (1 << 2) | (1 << 15))?;
        }
        Ok(Self(owned))
    }
}
struct OwnedChild {
    child: Child,
    killed: bool,
}
impl OwnedChild {
    fn kill_group(&mut self) {
        if !self.killed {
            unsafe {
                libc::kill(-(self.child.id() as i32), libc::SIGKILL);
            };
            self.killed = true;
        }
    }
}
impl Drop for OwnedChild {
    fn drop(&mut self) {
        self.kill_group();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
fn exited(pid: u32) -> Result<bool> {
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    ensure!(
        unsafe {
            libc::waitid(
                libc::P_PID,
                pid,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        } == 0,
        "waitid failed"
    );
    Ok(unsafe { info.si_pid() } != 0)
}
fn pipe(raw: impl IntoRawFd) -> Result<AsyncFd<File>> {
    let file = unsafe { File::from_raw_fd(raw.into_raw_fd()) };
    ensure!(
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFL, libc::O_NONBLOCK) } >= 0,
        "nonblocking pipe failed"
    );
    Ok(unsafe { AsyncFd::register(file) }?)
}
async fn drain(pipe: AsyncFd<File>, remaining: Arc<AtomicUsize>) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = [0u8; 8192];
    loop {
        let mut ready = pipe.readable().await?;
        let n = match ready.try_io(|fd| {
            let n = unsafe {
                libc::read(
                    fd.get_ref().as_raw_fd(),
                    buffer.as_mut_ptr().cast(),
                    buffer.len(),
                )
            };
            if n < 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(n as usize)
            }
        }) {
            Ok(r) => r?,
            Err(_) => continue,
        };
        if n == 0 {
            return Ok(output);
        }
        let mut left = remaining.load(Ordering::Relaxed);
        loop {
            let next = left.checked_sub(n).ok_or_else(|| {
                anyhow::anyhow!("capture process combined stdout/stderr budget exceeded")
            })?;
            match remaining.compare_exchange_weak(left, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(current) => left = current,
            }
        }
        output.extend_from_slice(&buffer[..n]);
    }
}
pub(crate) async fn run(
    config: &CaptureConfig,
    pin: &ExecutablePin,
    args: Vec<String>,
    stage: &Path,
    device: Option<&Path>,
    xauthority: Option<&Path>,
    ctx: &ToolContext,
) -> Result<Vec<u8>> {
    ensure!(
        !ctx.cancellation.is_cancelled(),
        "capture process cancelled"
    );
    let binary = pin.open()?;
    let mut reads = config.runtime_read_roots.clone();
    reads.push(pin.path.clone());
    let ruleset = Prepared::new(&reads, stage, device)?;
    let mut command = Command::new(format!("/proc/self/fd/{}", binary.as_raw_fd()));
    command
        .arg0(&pin.path)
        .args(args)
        .env_clear()
        .env("LC_ALL", "C")
        .current_dir(stage)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    if let Some(cookie) = xauthority {
        command.env("XAUTHORITY", cookie);
    }
    let max_file = config.max_output_bytes as u64;
    unsafe {
        command.pre_exec(move || {
            for (resource, value) in [
                (libc::RLIMIT_CORE, 0),
                (libc::RLIMIT_FSIZE, max_file),
                (libc::RLIMIT_CPU, 100),
            ] {
                let limit = libc::rlimit {
                    rlim_cur: value,
                    rlim_max: value,
                };
                if libc::setrlimit(resource, &limit) < 0 {
                    return Err(io::Error::last_os_error());
                }
            }
            if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0 {
                return Err(io::Error::last_os_error());
            }
            if libc::syscall(
                libc::SYS_landlock_restrict_self,
                ruleset.0.as_raw_fd(),
                0u32,
            ) < 0
            {
                return Err(io::Error::last_os_error());
            }
            if libc::syscall(libc::SYS_close_range, 3u32, u32::MAX, 4u32) < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut owned = OwnedChild {
        child: command
            .spawn()
            .context("capture spawn/restriction/exec failed; no fallback")?,
        killed: false,
    };
    drop(binary);
    let stdout = pipe(owned.child.stdout.take().context("stdout unavailable")?)?;
    let stderr = pipe(owned.child.stderr.take().context("stderr unavailable")?)?;
    let budget = Arc::new(AtomicUsize::new(128 * 1024));
    let work = async {
        tokio::try_join!(
            async {
                while !exited(owned.child.id())? {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                owned.kill_group();
                Ok::<_, anyhow::Error>(owned.child.wait()?)
            },
            drain(stdout, budget.clone()),
            drain(stderr, budget)
        )
    };
    let result = tokio::select! {biased; _=ctx.cancellation.cancelled()=>Err(anyhow::anyhow!("capture cancelled; outcome may be unknown; never automatically retry")), _=tokio::time::sleep(Duration::from_millis(config.timeout_ms))=>Err(anyhow::anyhow!("capture timed out; outcome may be unknown; never automatically retry")), result=work=>result};
    drop(owned);
    let (status, stdout, stderr) = result?;
    ensure!(
        status.success(),
        "capture process failed: {}",
        String::from_utf8_lossy(&stderr)
    );
    Ok(stdout)
}
pub(crate) fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    ensure!(
        file.metadata()?.is_file() && file.metadata()?.len() <= limit as u64,
        "capture output exceeds budget or is not regular"
    );
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(
        !bytes.is_empty() && bytes.len() <= limit,
        "capture output empty or exceeds budget"
    );
    Ok(bytes)
}
