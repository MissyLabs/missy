//! Restrictions only inside dedicated exec children, NEVER this test runner.
use missy_core::{PolicyConfig, Scope, ToolContext};
use missy_sandbox::{SandboxConfig, SandboxedCommand, landlock_abi};
use std::{fs, net::TcpListener, path::PathBuf, time::Duration};
use tokio_util::sync::CancellationToken;
fn supported() -> bool {
    match landlock_abi() {
        Ok(abi) if abi >= 4 => {
            eprintln!("REAL Landlock subprocess tests: kernel ABI {abi}");
            true
        }
        result => {
            eprintln!("SKIP real Landlock tests: require ABI >= 4, got {result:?}");
            false
        }
    }
}
fn context() -> ToolContext {
    ToolContext {
        scope: Scope {
            tenant: "test".into(),
            user: "test".into(),
            channel: "test".into(),
            session: "test".into(),
        },
        turn_id: "sandbox-test".into(),
        cancellation: CancellationToken::new(),
    }
}
fn python(script: String) -> Vec<String> {
    vec![
        "/usr/bin/python3".into(),
        "-I".into(),
        "-S".into(),
        "-c".into(),
        script,
    ]
}
fn executor(
    argv: &[String],
    reads: Vec<PathBuf>,
    writes: Vec<PathBuf>,
    timeout_ms: u64,
    limit: usize,
) -> SandboxedCommand {
    let config = SandboxConfig {
        read_roots: reads.clone(),
        write_roots: writes.clone(),
        deny_tcp: true,
        timeout_ms,
        max_output_bytes: limit,
    };
    SandboxedCommand::new(
        PolicyConfig {
            shell_enabled: true,
            mcp_enabled: true,
            allowed_commands: vec![argv.to_vec()],
            read_roots: reads,
            write_roots: writes,
            ..Default::default()
        },
        config,
    )
    .unwrap()
}
fn runtime_roots() -> Vec<PathBuf> {
    ["/usr", "/lib", "/lib64"]
        .iter()
        .filter_map(|p| fs::canonicalize(p).ok())
        .collect()
}
#[tokio::test]
async fn real_fs_tcp_environment_descriptors_parent_unchanged() {
    if !supported() {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let read = temp.path().join("read");
    let write = temp.path().join("write");
    let forbidden = temp.path().join("forbidden");
    for p in [&read, &write, &forbidden] {
        fs::create_dir(p).unwrap();
    }
    fs::write(read.join("allowed"), "allowed").unwrap();
    fs::write(forbidden.join("secret"), "secret").unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let secret_file = fs::File::open(forbidden.join("secret")).unwrap();
    use std::os::fd::AsRawFd;
    let inherited = unsafe { libc::fcntl(secret_file.as_raw_fd(), libc::F_DUPFD, 200) };
    assert!(inherited >= 200);
    struct Close(i32);
    impl Drop for Close {
        fn drop(&mut self) {
            unsafe {
                libc::close(self.0);
            }
        }
    }
    let inherited = Close(inherited);
    let script = format!(
        r#"
import os, socket, errno
assert open({allowed:?}).read() == 'allowed'
open({written:?}, 'w').write('written')
def denied(f):
    try: f()
    except PermissionError: return
    raise AssertionError('operation unexpectedly permitted')
denied(lambda: open({secret:?}).read())
denied(lambda: open({forbidden_write:?}, 'w'))
denied(lambda: open({allowed:?}, 'w'))
denied(lambda: os.truncate({allowed:?}, 0))
denied(lambda: open({link:?}).read())
s=socket.socket()
denied(lambda: s.connect(('127.0.0.1', {port})))
s.close()
s=socket.socket()
denied(lambda: s.bind(('127.0.0.1', 0)))
s.close()
try: os.fstat({fd})
except OSError as e: assert e.errno == errno.EBADF
else: raise AssertionError('inherited descriptor leaked')
assert set(os.environ) <= {{'LC_CTYPE'}}
print('ISOLATED_OK')
"#,
        allowed = read.join("allowed").to_str().unwrap(),
        written = write.join("written").to_str().unwrap(),
        secret = forbidden.join("secret").to_str().unwrap(),
        forbidden_write = forbidden.join("new").to_str().unwrap(),
        link = read.join("escape").to_str().unwrap(),
        fd = inherited.0
    );
    std::os::unix::fs::symlink(forbidden.join("secret"), read.join("escape")).unwrap();
    let argv = python(script);
    let mut roots = runtime_roots();
    roots.push(read.clone());
    let result = executor(&argv, roots, vec![write.clone()], 3000, 4096)
        .execute(&argv, &context())
        .await
        .unwrap();
    assert_eq!(result["success"], true, "{result}");
    assert_eq!(result["stdout"], "ISOLATED_OK\n");
    assert_eq!(
        fs::read_to_string(write.join("written")).unwrap(),
        "written"
    );
    assert_eq!(
        fs::read_to_string(forbidden.join("secret")).unwrap(),
        "secret"
    );
    fs::write(forbidden.join("parent"), "parent unrestricted").unwrap();
    let _connection = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    assert!(!forbidden.join("new").exists());
}
#[tokio::test]
async fn exact_policy_symlink_roots_mcp_gate() {
    if !supported() {
        return;
    }
    let argv = python("print('ok')".into());
    let sandbox = executor(&argv, runtime_roots(), vec![], 3000, 4096);
    let mut other = argv.clone();
    other.push("unexpected".into());
    assert!(sandbox.execute(&other, &context()).await.is_err());
    assert_eq!(
        sandbox.execute_mcp(&argv, &context()).await.unwrap()["success"],
        true
    );
    let temp = tempfile::tempdir().unwrap();
    let real = temp.path().join("real");
    fs::create_dir(&real).unwrap();
    let link = temp.path().join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let config = SandboxConfig {
        read_roots: vec![link.clone()],
        ..Default::default()
    };
    let sandbox = SandboxedCommand::new(
        PolicyConfig {
            read_roots: vec![link],
            shell_enabled: true,
            allowed_commands: vec![argv.clone()],
            ..Default::default()
        },
        config,
    )
    .unwrap();
    assert!(sandbox.execute(&argv, &context()).await.is_err());
    assert!(
        SandboxedCommand::new(
            PolicyConfig::default(),
            SandboxConfig {
                write_roots: vec![real],
                ..Default::default()
            }
        )
        .is_err()
    );
}
#[tokio::test]
async fn overflow_timeout_cancel_and_abort_cleanup() {
    if !supported() {
        return;
    }
    let argv = python("import os; os.write(1,b'x'*100000); os.write(2,b'y'*100000)".into());
    assert!(
        executor(&argv, runtime_roots(), vec![], 3000, 100)
            .execute(&argv, &context())
            .await
            .unwrap_err()
            .to_string()
            .contains("output byte limit")
    );
    let temp = tempfile::tempdir().unwrap();
    let pidfile = temp.path().join("pid");
    let make_argv = || {
        python(format!(
            "import os,time; open({:?},'w').write(str(os.getpid())); time.sleep(60)",
            pidfile.to_str().unwrap()
        ))
    };
    let argv = make_argv();
    assert!(
        executor(
            &argv,
            runtime_roots(),
            vec![temp.path().to_owned()],
            100,
            4096
        )
        .execute(&argv, &context())
        .await
        .unwrap_err()
        .to_string()
        .contains("timed out")
    );
    if pidfile.exists() {
        assert_dead(fs::read_to_string(&pidfile).unwrap().parse().unwrap());
    }
    fs::remove_file(&pidfile).ok();
    let argv = make_argv();
    let sandbox = executor(
        &argv,
        runtime_roots(),
        vec![temp.path().to_owned()],
        3000,
        4096,
    );
    let ctx = context();
    let cancellation = ctx.cancellation.clone();
    let task = tokio::spawn(async move { sandbox.execute(&argv, &ctx).await });
    let pid = wait_pid(&pidfile).await;
    cancellation.cancel();
    assert!(
        task.await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("cancelled")
    );
    assert_dead(pid);
    fs::remove_file(&pidfile).unwrap();
    let argv = make_argv();
    let sandbox = executor(
        &argv,
        runtime_roots(),
        vec![temp.path().to_owned()],
        3000,
        4096,
    );
    let task = tokio::spawn(async move { sandbox.execute(&argv, &context()).await });
    let pid = wait_pid(&pidfile).await;
    task.abort();
    let _ = task.await;
    assert_dead(pid);
}
async fn wait_pid(path: &std::path::Path) -> i32 {
    for _ in 0..200 {
        if let Ok(content) = fs::read_to_string(path)
            && let Ok(pid) = content.parse()
        {
            return pid;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("child failed to write isolated pid file")
}
fn assert_dead(pid: i32) {
    assert!(
        unsafe { libc::kill(pid, 0) } < 0,
        "sandbox child {pid} survived cleanup"
    );
}
