use crate::{SandboxConfig, landlock::Prepared};
use anyhow::{Context, Result, bail};
use missy_core::ToolContext;
use serde_json::{Value, json};
use std::{
    fs::File,
    io,
    os::{
        fd::{AsRawFd, FromRawFd, IntoRawFd},
        unix::process::CommandExt,
    },
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::io::unix::AsyncFd;

/// Std Child ownership guarantees synchronous kill AND reap on future drop.
/// WNOWAIT reserves the leader PID until group cleanup; no reused-PID kills.
struct OwnedChild {
    child: Child,
    killed: bool,
}
impl OwnedChild {
    fn kill_group(&mut self) {
        if !self.killed {
            unsafe {
                libc::kill(-(self.child.id() as i32), libc::SIGKILL);
            }
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
    if unsafe {
        libc::waitid(
            libc::P_PID,
            pid,
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    } < 0
    {
        return Err(io::Error::last_os_error()).context("sandbox waitid failed");
    }
    Ok(unsafe { info.si_pid() } != 0)
}
fn pipe(raw: impl IntoRawFd) -> Result<AsyncFd<File>> {
    let file = unsafe { File::from_raw_fd(raw.into_raw_fd()) };
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFL, libc::O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error()).context("sandbox nonblocking pipe failed");
    }
    // SAFETY: exclusive File ownership and stable fd until deregistration/drop;
    // capture never closes/replaces this fd.
    Ok(unsafe { AsyncFd::register(file) }?)
}
async fn capture(pipe: AsyncFd<File>, remaining: Arc<AtomicUsize>) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        let mut ready = pipe.readable().await?;
        let n = match ready.try_io(|fd| {
            let n =
                unsafe { libc::read(fd.get_ref().as_raw_fd(), buf.as_mut_ptr().cast(), buf.len()) };
            if n < 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(n as usize)
            }
        }) {
            Ok(n) => n?,
            Err(_) => continue,
        };
        if n == 0 {
            return Ok(output);
        }
        if remaining
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |left| {
                left.checked_sub(n)
            })
            .is_err()
        {
            bail!("sandbox combined output byte limit exceeded");
        }
        output.extend_from_slice(&buf[..n]);
    }
}
pub(crate) async fn execute(
    argv: &[String],
    config: &SandboxConfig,
    ruleset: Prepared,
    ctx: &ToolContext,
) -> Result<Value> {
    let abi = ruleset.abi;
    let mut command = Command::new(&argv[0]);
    command
        .args(&argv[1..])
        .env_clear()
        .current_dir("/")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    // SAFETY: closure only runs in forked child. Parent preparation allocated and
    // opened paths. Here: setrlimit, prctl, syscall and errno conversion only.
    unsafe {
        command.pre_exec(move || {
            let limit = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            if libc::setrlimit(libc::RLIMIT_CORE, &limit) < 0 {
                return Err(io::Error::last_os_error());
            }
            ruleset.restrict_child()
        });
    }
    let child = command
        .spawn()
        .context("sandbox child spawn/restriction/exec failed; no fallback")?;
    let mut owned = OwnedChild {
        child,
        killed: false,
    };
    let stdout = pipe(
        owned
            .child
            .stdout
            .take()
            .context("sandbox stdout unavailable")?,
    )?;
    let stderr = pipe(
        owned
            .child
            .stderr
            .take()
            .context("sandbox stderr unavailable")?,
    )?;
    let remaining = Arc::new(AtomicUsize::new(config.max_output_bytes));
    let work = async {
        tokio::try_join!(
            async {
                while !exited(owned.child.id())? {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                owned.kill_group();
                owned.child.wait().map_err(anyhow::Error::from)
            },
            capture(stdout, remaining.clone()),
            capture(stderr, remaining)
        )
    };
    let result = tokio::select! {
        biased;
        _ = ctx.cancellation.cancelled() => Err(anyhow::anyhow!("sandbox command cancelled")),
        _ = tokio::time::sleep(Duration::from_millis(config.timeout_ms)) => Err(anyhow::anyhow!("sandbox command timed out")),
        result = work => result,
    };
    drop(owned); // Exact owned group killed and child reaped on EVERY path.
    let (status, stdout, stderr) = result?;
    Ok(
        json!({"trust":"untrusted", "sandbox":{"backend":"linux-landlock", "abi":abi,"tcp_denied":config.deny_tcp},
        "status":status.code(), "success":status.success(), "stdout":String::from_utf8_lossy(&stdout), "stderr":String::from_utf8_lossy(&stderr)}),
    )
}
