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

fn command(config: &CliProviderConfig, args: &[String], cwd: &Path) -> Command {
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
    command
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
    let mut command = command(config, args, cwd);
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

/// Interactive bounded stdio driver. Shares the one-shot driver's cleared env,
/// trusted executable and RAII process-group guard. No interpreter discovery.
pub(crate) async fn run_acp(
    config: &CliProviderConfig,
    cwd: &Path,
    session: &mut crate::acp::Session<'_>,
    cancellation: CancellationToken,
) -> Result<()> {
    if cancellation.is_cancelled() {
        bail!("ACP completion cancelled");
    }
    let pin = config.acp_adapter.as_ref().expect("validated ACP pin");
    let args = vec![pin.entrypoint.to_string_lossy().into_owned()];
    let child = command(config, &args, cwd)
        .spawn()
        .map_err(|_| anyhow::anyhow!("cannot start approved ACP executable"))?;
    let group =
        i32::try_from(child.id()).map_err(|_| anyhow::anyhow!("invalid ACP process group"))?;
    let mut guard = Guard {
        child,
        group,
        killed: false,
    };
    let mut stdin = guard.child.stdin.take().expect("piped stdin");
    let mut stdout = guard.child.stdout.take().expect("piped stdout");
    let mut stderr = guard.child.stderr.take().expect("piped stderr");
    nonblocking(&stdin)?;
    nonblocking(&stdout)?;
    nonblocking(&stderr)?;
    let mut pending = crate::acp::Session::initialize()?;
    let mut written = 0;
    let mut incoming = Vec::new();
    let (mut out_total, mut err_total) = (0, 0);
    let started = Instant::now();
    loop {
        if cancellation.is_cancelled()
            || started.elapsed() >= Duration::from_secs(config.timeout_seconds)
        {
            // Best-effort protocol cancellation followed immediately by mandatory
            // group kill/reap. Never wait indefinitely for a server acknowledgement.
            let _ = stdin.write(&session.cancel());
            if cancellation.is_cancelled() {
                bail!("ACP completion cancelled");
            }
            bail!("ACP completion timed out");
        }
        if written < pending.len() {
            match stdin.write(&pending[written..pending.len().min(written + 8192)]) {
                Ok(0) => bail!("ACP closed input"),
                Ok(n) => written += n,
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                    ) => {}
                Err(_) => bail!("ACP input failed; diagnostics withheld"),
            }
        }
        let eof = drain(
            &mut stdout,
            &mut incoming,
            &mut out_total,
            config.max_stdout_bytes,
            true,
        )?;
        drain(
            &mut stderr,
            &mut Vec::new(),
            &mut err_total,
            config.max_stderr_bytes,
            false,
        )?;
        // A peer cannot cause unbounded response queuing: process at most 64 KiB
        // per drain and reject a request while its predecessor is still pending.
        while let Some(end) = incoming.iter().position(|b| *b == b'\n') {
            let line: Vec<_> = incoming.drain(..=end).collect();
            match session.receive(&line)? {
                Some(reply) if reply.is_empty() => {}
                Some(reply) => {
                    if written != pending.len() {
                        bail!("ACP invalid request ordering");
                    }
                    pending = reply;
                    written = 0;
                }
                None => {
                    if !incoming.is_empty() {
                        bail!("ACP trailing protocol data");
                    }
                    return Ok(()); // guard kills/reaps the long-lived adapter
                }
            }
        }
        if eof {
            bail!("ACP closed before terminal response; diagnostics withheld");
        }
        if guard
            .child
            .try_wait()
            .map_err(|_| anyhow::anyhow!("ACP wait failed"))?
            .is_some()
        {
            bail!("ACP exited before terminal response; diagnostics withheld");
        }
        // Poll delay is bounded; cancellation is checked before the next slice.
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}
