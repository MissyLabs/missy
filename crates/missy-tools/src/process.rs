//! Owned process groups. Dropping an abandoned async future cancels and joins
//! its worker, whose child guard kills the complete group and waits the leader.
use anyhow::{Context, Result, bail};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    os::{fd::AsRawFd, unix::process::CommandExt},
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use tokio_util::sync::CancellationToken;

pub(crate) struct Process {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: ChildStdout,
    stderr: ChildStderr,
    pending: Vec<u8>,
    stderr_bytes: Vec<u8>,
    read_bytes: usize,
    stdout_eof: bool,
    stderr_eof: bool,
    reaped: bool,
    limit: usize,
    deadline: Instant,
    stop: Arc<AtomicBool>,
    cancellation: CancellationToken,
}
impl Drop for Process {
    fn drop(&mut self) {
        if self.reaped {
            return;
        }
        // Also kill descendants when the leader has already exited successfully.
        unsafe {
            libc::kill(-(self.child.id() as i32), libc::SIGKILL);
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
fn nonblocking(fd: i32) -> Result<()> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}
impl Process {
    pub(crate) fn spawn(
        argv: &[String],
        env: &BTreeMap<String, String>,
        timeout: Duration,
        limit: usize,
        stop: Arc<AtomicBool>,
        cancellation: CancellationToken,
    ) -> Result<Self> {
        if stop.load(Ordering::Acquire) || cancellation.is_cancelled() {
            bail!("process cancelled before dispatch");
        }
        let executable = argv.first().context("empty command")?;
        if !std::path::Path::new(executable).is_absolute() {
            bail!("executable must be absolute");
        }
        let mut cmd = Command::new(executable);
        cmd.args(&argv[1..])
            .env_clear()
            .envs(env)
            .current_dir("/")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0);
        let mut child = cmd.spawn().context("spawn approved executable")?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let p = Self {
            child,
            stdin,
            stdout,
            stderr,
            pending: vec![],
            stderr_bytes: vec![],
            read_bytes: 0,
            stdout_eof: false,
            stderr_eof: false,
            reaped: false,
            limit,
            deadline: Instant::now() + timeout,
            stop,
            cancellation,
        };
        nonblocking(p.stdin.as_ref().expect("piped stdin").as_raw_fd())?;
        nonblocking(p.stdout.as_raw_fd())?;
        nonblocking(p.stderr.as_raw_fd())?;
        Ok(p)
    }
    fn check(&self) -> Result<()> {
        if self.stop.load(Ordering::Acquire) || self.cancellation.is_cancelled() {
            bail!("process cancelled");
        }
        if Instant::now() >= self.deadline {
            bail!("process deadline exceeded");
        }
        Ok(())
    }
    fn drain(&mut self) -> Result<bool> {
        self.check()?;
        for is_err in [false, true] {
            let mut buf = [0u8; 4096];
            let result = if is_err {
                self.stderr.read(&mut buf)
            } else {
                self.stdout.read(&mut buf)
            };
            match result {
                Ok(0) => {
                    if is_err {
                        self.stderr_eof = true;
                    } else {
                        self.stdout_eof = true;
                    }
                }
                Ok(n) => {
                    self.read_bytes = self.read_bytes.saturating_add(n);
                    if self.read_bytes > self.limit {
                        bail!("process output exceeds byte limit");
                    }
                    if is_err {
                        self.stderr_bytes.extend_from_slice(&buf[..n]);
                    } else {
                        self.pending.extend_from_slice(&buf[..n]);
                    }
                }
                Err(e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e.into()),
            }
        }
        Ok(self.stdout_eof)
    }
    pub(crate) fn write(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > self.limit {
            bail!("process input exceeds byte limit");
        }
        let mut at = 0;
        while at < data.len() {
            self.check()?;
            match self
                .stdin
                .as_mut()
                .context("closed stdin")?
                .write(&data[at..])
            {
                Ok(0) => bail!("closed process stdin"),
                Ok(n) => at += n,
                Err(e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::Interrupted =>
                {
                    self.drain()?;
                    thread::sleep(Duration::from_millis(3));
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }
    pub(crate) fn line(&mut self) -> Result<Vec<u8>> {
        loop {
            self.check()?;
            if let Some(n) = self.pending.iter().position(|&b| b == b'\n') {
                return Ok(self.pending.drain(..=n).collect());
            }
            if self.drain()? {
                bail!("MCP process closed stdout before complete message");
            }
            thread::sleep(Duration::from_millis(3));
        }
    }
    pub(crate) fn shell(mut self) -> Result<serde_json::Value> {
        self.stdin.take();
        loop {
            self.drain()?;
            if self.stdout_eof && self.stderr_eof {
                // WNOWAIT keeps PID reserved until the group has been killed,
                // preventing group-ID reuse between status detection and cleanup.
                let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
                let rc = unsafe {
                    libc::waitid(
                        libc::P_PID,
                        self.child.id(),
                        &mut info,
                        libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                    )
                };
                if rc < 0 {
                    return Err(std::io::Error::last_os_error().into());
                }
                if unsafe { info.si_pid() } != 0 {
                    unsafe {
                        libc::kill(-(self.child.id() as i32), libc::SIGKILL);
                    }
                    let status = self.child.wait()?;
                    self.reaped = true;
                    return Ok(
                        serde_json::json!({"exit_code":status.code(),"stdout":String::from_utf8_lossy(&self.pending),"stderr":String::from_utf8_lossy(&self.stderr_bytes)}),
                    );
                }
            }
            thread::sleep(Duration::from_millis(3));
        }
    }
}

struct Worker<T> {
    stop: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<()>>,
    rx: tokio::sync::oneshot::Receiver<Result<T>>,
}
impl<T> Drop for Worker<T> {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}
pub(crate) async fn worker<T: Send + 'static>(
    f: impl FnOnce(Arc<AtomicBool>) -> Result<T> + Send + 'static,
) -> Result<T> {
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();
    let join = thread::Builder::new()
        .name("missy-owned-tool".into())
        .spawn(move || {
            let _ = tx.send(f(flag));
        })?;
    let mut worker = Worker {
        stop,
        join: Some(join),
        rx,
    };
    (&mut worker.rx).await.context("tool worker failed")?
}
