//! Disposable real supervisor and loopback fixture; no external services.
use missy_core::{Config, ProviderConfig};
use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    os::unix::fs::PermissionsExt,
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn watchdog_process_records_failures_recovery_and_stops() {
    let temp = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let healthy = Arc::new(AtomicBool::new(false));
    let stop = Arc::new(AtomicBool::new(false));
    let (h, s) = (healthy.clone(), stop.clone());
    let fixture = thread::spawn(move || {
        while !s.load(Ordering::SeqCst) {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0; 1024];
                let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                let _ = stream.read(&mut buf);
                let code = if h.load(Ordering::SeqCst) {
                    "200 OK"
                } else {
                    "503 Unavailable"
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 {code}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
            } else {
                thread::sleep(Duration::from_millis(10));
            }
        }
    });
    let mut cfg = Config {
        state_dir: temp.path().join("state"),
        ..Default::default()
    };
    cfg.providers.push(ProviderConfig {
        model: "never-contacted".into(),
        input_microusd_per_million: Some(0),
        output_microusd_per_million: Some(0),
        ..Default::default()
    });
    cfg.policy.allow_private_network = true;
    cfg.policy.allowed_origins.push(format!("http://{address}"));
    let watchdog = temp.path().join("watchdog.yaml");
    fs::write(&watchdog, format!("enabled: true\nscope: {{tenant: t, user: u, channel: test, session: test}}\npoll_seconds: 1\nfailure_threshold: 2\nprobes:\n  - {{kind: database, name: db}}\n  - {{kind: http, name: mock, url: 'http://{address}/health'}}\n")).unwrap();
    fs::set_permissions(&watchdog, fs::Permissions::from_mode(0o600)).unwrap();
    cfg.extensions.watchdog_config = Some(watchdog);
    let path = temp.path().join("config.yaml");
    fs::write(&path, serde_yaml_ng::to_string(&cfg).unwrap()).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    let mut child = Process(
        Command::new(env!("CARGO_BIN_EXE_missy-rs"))
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .arg("--config")
            .arg(&path)
            .args(["gateway", "start"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let db = cfg.state_dir.join("scheduler.sqlite3");
    let query = || -> Option<(i64, i64)> {
        let db =
            rusqlite::Connection::open_with_flags(&db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
                .ok()?;
        db.query_row(
            "SELECT healthy, failures FROM automation_health WHERE name='mock'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .ok()
    };
    let deadline = Instant::now() + Duration::from_secs(30);
    while !query().is_some_and(|(h, f)| h == 0 && f >= 2) {
        assert!(
            Instant::now() < deadline && child.0.try_wait().unwrap().is_none(),
            "two failed checks not recorded"
        );
        thread::sleep(Duration::from_millis(50));
    }
    healthy.store(true, Ordering::SeqCst);
    while query() != Some((1, 0)) {
        assert!(
            Instant::now() < deadline && child.0.try_wait().unwrap().is_none(),
            "recovery not recorded"
        );
        thread::sleep(Duration::from_millis(50));
    }
    let conn =
        rusqlite::Connection::open_with_flags(&db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .unwrap();
    let db_ok: i64 = conn
        .query_row(
            "SELECT healthy FROM automation_health WHERE name='db'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(db_ok, 1);
    let events: i64 = conn
        .query_row(
            "SELECT count(*) FROM automation_health_events WHERE name='mock'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(events >= 3);
    unsafe {
        libc::kill(child.0.id() as i32, libc::SIGTERM);
    }
    assert!(child.0.wait().unwrap().success());
    thread::sleep(Duration::from_millis(1200));
    let after: i64 = conn
        .query_row(
            "SELECT count(*) FROM automation_health_events WHERE name='mock'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(events, after);
    stop.store(true, Ordering::SeqCst);
    fixture.join().unwrap();
}
