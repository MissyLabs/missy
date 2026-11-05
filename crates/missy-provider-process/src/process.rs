//! Nonblocking Unix pipes keep cancellation bounded without detached readers.
//! The guard synchronously kills the entire child process group and waits on
//! every exit path, including dropping/aborting the completion future.
use crate::CliProviderConfig;
use anyhow::{Result, bail};
use std::{
    io::{Read, Write},
    os::{fd::AsRawFd, unix::process::CommandExt},
    path::Path,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
use tokio_util::sync::CancellationToken;

pub(crate) struct Output {
    pub success: bool,
    pub stdout: Vec<u8>,
}

struct Guard {
    child: Child,
    group: i32,
    killed: bool,
}
impl Guard {
    fn kill_group(&mut self) {
        if self.killed {
            return;
        }
        // SAFETY: child was started in a new group whose id is its positive PID.
        // No shell or unrelated session group is targeted.
        unsafe {
            libc::kill(-self.group, libc::SIGKILL);
        }
        self.killed = true;
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        self.kill_group();
        let _ = self.child.kill();
        // Blocking wait after SIGKILL deliberately reaps the immediate child on
        // future drop. No asynchronous reaper task can be abandoned at shutdown.
        let _ = self.child.wait();
    }
}

fn nonblocking(fd: &impl AsRawFd) -> Result<()> {
    // SAFETY: fcntl only accesses the valid descriptor held by the borrowed pipe.
    let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFL) };
    if flags < 0
        || unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0
    {
        bail!("cannot configure nonblocking CLI pipes");
    }
    Ok(())
}

fn drain(
    pipe: &mut impl Read,
    sink: &mut Vec<u8>,
    total: &mut usize,
    limit: usize,
    retain: bool,
) -> Result<bool> {
    let mut buffer = [0u8; 8192];
    // Bound each scheduling slice even if an adversarial subprocess never stops.
    for _ in 0..8 {
        match pipe.read(&mut buffer) {
            Ok(0) => return Ok(true),
            Ok(n) => {
                *total = total
                    .checked_add(n)
                    .ok_or_else(|| anyhow::anyhow!("CLI output limit exceeded"))?;
                if *total > limit {
                    bail!("CLI output limit exceeded; diagnostics withheld");
                }
                if retain {
                    sink.extend_from_slice(&buffer[..n]);
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return Ok(false),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => bail!("CLI pipe read failed; diagnostics withheld"),
        }
    }
    Ok(false)
}

pub(crate) async fn run(
    config: &CliProviderConfig,
    args: &[String],
    cwd: &Path,
    input: &[u8],
    cancellation: CancellationToken,
) -> Result<Output> {
    if cancellation.is_cancelled() {
        bail!("CLI completion cancelled");
    }
    let mut command = Command::new(&config.executable);
    command
        .args(args)
        .current_dir(cwd)
        .env_clear()
        // No PATH, API keys, proxy endpoints, nested-agent flags, LD_PRELOAD,
        // NODE_OPTIONS or inherited shell/config variables. These are the entire
        // whitelist. Operators must supply a self-contained absolute executable.
        .env("HOME", &config.auth_home)
        .env("CLAUDE_CONFIG_DIR", config.auth_home.join(".claude"))
        .env("CODEX_HOME", config.auth_home.join(".codex"))
        .env("LANG", "C.UTF-8")
        .env("NO_COLOR", "1")
        .env("DISABLE_AUTOUPDATER", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    let child = command
        .spawn()
        .map_err(|_| anyhow::anyhow!("cannot start approved CLI executable"))?;
    let group =
        i32::try_from(child.id()).map_err(|_| anyhow::anyhow!("invalid CLI process group"))?;
    let mut guard = Guard {
        child,
        group,
        killed: false,
    };
    let mut stdin = guard.child.stdin.take();
    let mut stdout = guard.child.stdout.take().expect("piped stdout");
    let mut stderr = guard.child.stderr.take().expect("piped stderr");
    nonblocking(stdin.as_ref().expect("piped stdin"))?;
    nonblocking(&stdout)?;
    nonblocking(&stderr)?;
    let mut written = 0;
    let mut out = Vec::new();
    let (mut out_total, mut err_total) = (0, 0);
    let (mut out_done, mut err_done) = (false, false);
    let mut status = None;
    let started = Instant::now();
    loop {
        if cancellation.is_cancelled() {
            bail!("CLI completion cancelled");
        }
        if started.elapsed() >= Duration::from_secs(config.timeout_seconds) {
            bail!("CLI completion timed out");
        }
        if let Some(pipe) = stdin.as_mut() {
            if written < input.len() {
                match pipe.write(&input[written..input.len().min(written + 8192)]) {
                    Ok(0) => bail!("CLI closed prompt input"),
                    Ok(n) => written += n,
                    Err(e)
                        if matches!(
                            e.kind(),
                            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                        ) => {}
                    Err(_) => bail!("CLI prompt input failed; diagnostics withheld"),
                }
            }
            if written == input.len() {
                stdin = None;
            }
        }
        if !out_done {
            out_done = drain(
                &mut stdout,
                &mut out,
                &mut out_total,
                config.max_stdout_bytes,
                true,
            )?;
        }
        // Stderr is counted, never retained or included in errors/logs.
        if !err_done {
            err_done = drain(
                &mut stderr,
                &mut Vec::new(),
                &mut err_total,
                config.max_stderr_bytes,
                false,
            )?;
        }
        if status.is_none() {
            status = guard
                .child
                .try_wait()
                .map_err(|_| anyhow::anyhow!("CLI wait failed"))?;
            if status.is_some() {
                guard.kill_group();
                stdin = None;
            }
        }
        if let Some(status) = status
            && out_done
            && err_done
        {
            return Ok(Output {
                success: status.success(),
                stdout: out,
            });
        }
        tokio::select! {
            _ = cancellation.cancelled() => bail!("CLI completion cancelled"),
            _ = tokio::time::sleep(Duration::from_millis(5)) => {},
        }
    }
}
