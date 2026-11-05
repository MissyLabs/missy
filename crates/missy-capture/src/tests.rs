use super::*;
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};
use tokio_util::sync::CancellationToken;

fn context() -> ToolContext {
    ToolContext {
        scope: missy_core::Scope {
            tenant: "capture-tests".into(),
            user: "synthetic".into(),
            channel: "isolated".into(),
            session: "fixture".into(),
        },
        turn_id: "capture-test".into(),
        cancellation: CancellationToken::new(),
    }
}
fn pin(path: &str) -> ExecutablePin {
    let p = PathBuf::from(path).canonicalize().unwrap();
    ExecutablePin {
        sha256: hex::encode(Sha256::digest(fs::read(&p).unwrap())),
        path: p,
    }
}
fn config(root: &std::path::Path) -> CaptureConfig {
    CaptureConfig {
        enable_camera: true,
        enable_screencast: false,
        cameras: vec!["/dev/video65000".into()],
        x11: None,
        ffmpeg: pin("/usr/bin/ffmpeg"),
        ffprobe: pin("/usr/bin/ffprobe"),
        staging_root: root.join("stage"),
        runtime_read_roots: vec!["/usr/bin".into(), "/usr/lib".into()],
        timeout_ms: 10_000,
        max_output_bytes: 4 * 1024 * 1024,
        max_pixels: 1920 * 1080,
        max_fps: 30,
    }
}
struct Fixture {
    root: tempfile::TempDir,
    c: CaptureConfig,
    p: PolicyConfig,
}
impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("stage")).unwrap();
        fs::set_permissions(root.path().join("stage"), fs::Permissions::from_mode(0o700)).unwrap();
        let c = config(root.path());
        let p = PolicyConfig {
            write_roots: vec![root.path().to_owned()],
            max_file_bytes: c.max_output_bytes,
            ..Default::default()
        };
        Self { root, c, p }
    }
    fn call(&self, name: &str) -> ToolCall {
        ToolCall {
            id: "capture-1".into(),
            name: name.into(),
            arguments: json!({"device":"/dev/video65000","output":self.root.path().join(if name=="camera_photo" {"photo.png"} else {"clip.mp4"}),"consent":true,"width":160,"height":120,"fps":5,"duration_seconds":1}),
        }
    }
}
struct Deny(AtomicUsize);
#[async_trait]
impl CaptureApproval for Deny {
    async fn authorize(&self, _: &ToolContext, _: &ToolCall) -> Result<()> {
        self.0.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("human approval denied")
    }
    fn configuration_fingerprint(&self) -> String {
        "deny-fixture".into()
    }
}

#[tokio::test]
async fn synthetic_photo_and_clip_full_decode_hash_and_atomic_publication() {
    let f = Fixture::new();
    f.c.validate().unwrap();
    let p = Policy::new(f.p.clone()).unwrap();
    for name in ["camera_photo", "camera_clip"] {
        let call = f.call(name);
        let r: Request = serde_json::from_value(call.arguments).unwrap();
        r.validate(&f.c, name).unwrap();
        let result = capture::execute(
            &p,
            &f.c,
            &context(),
            &r,
            name,
            Some(capture::InternalSource::Synthetic),
        )
        .await
        .unwrap();
        let bytes = fs::read(&r.output).unwrap();
        assert_eq!(result["sha256"], hex::encode(Sha256::digest(&bytes)));
        assert_eq!(result["bytes"], bytes.len());
        assert_eq!(result["audio"], false);
        assert_eq!(
            fs::metadata(&r.output).unwrap().permissions().mode() & 0o777,
            0o600
        );
        if name == "camera_photo" {
            assert_eq!(result["frames"], 1);
        } else {
            assert_eq!(result["duration_seconds"], 1.0);
        }
        assert_eq!(fs::read_dir(&f.c.staging_root).unwrap().count(), 0);
    }
}
#[tokio::test]
async fn malformed_and_unsafe_arguments_never_reach_approval() {
    let f = Fixture::new();
    let deny = Arc::new(Deny(AtomicUsize::new(0)));
    let tools = CaptureTools::new(f.p.clone(), f.c.clone(), deny.clone()).unwrap();
    for (key, value) in [
        ("consent", json!(false)),
        ("duration_seconds", json!(0)),
        ("duration_seconds", json!(91)),
        ("fps", json!(31)),
        ("width", json!(159)),
        ("audio", json!(true)),
        ("audio_consent", json!(true)),
        ("device", json!("/dev/video0;echo bad")),
        ("output", json!("/tmp/bad\n.mp4")),
        ("output", json!("/tmp/../bad.mp4")),
        ("unknown", json!(1)),
    ] {
        let mut call = f.call("camera_clip");
        call.arguments[key] = value;
        assert!(
            tools
                .execute(&context(), &call)
                .await
                .unwrap_err()
                .downcast_ref::<ToolDispatchRefused>()
                .is_some()
        );
    }
    assert_eq!(deny.0.load(Ordering::SeqCst), 0);
    assert!(tools.definitions().iter().all(|d| d.side_effect));
    let mut call = f.call("camera_photo");
    call.arguments["output"] = json!(f.root.path().join("linked.png"));
    std::os::unix::fs::symlink(
        f.root.path().join("outside.png"),
        f.root.path().join("linked.png"),
    )
    .unwrap();
    assert!(tools.preflight(&context(), &call).await.is_err());
    let ctx = context();
    ctx.cancellation.cancel();
    assert!(
        tools
            .preflight(&ctx, &f.call("camera_photo"))
            .await
            .is_err()
    );
}
#[test]
fn strict_config_rejects_ambient_and_nonisolated_x11_and_pin_drift() {
    let f = Fixture::new();
    for display in [
        ":0",
        ":99",
        ":100.0",
        "localhost:100",
        ":100;bad",
        ":0100",
        ":100\n",
    ] {
        assert!(
            TrustedX11 {
                display: display.into(),
                xauthority: f.root.path().join("cookie"),
                isolated_session: true
            }
            .number()
            .is_err()
        );
    }
    assert!(
        TrustedX11 {
            display: ":100".into(),
            xauthority: f.root.path().join("cookie"),
            isolated_session: false
        }
        .number()
        .is_err()
    );
    let mut c = f.c.clone();
    c.ffmpeg.sha256 = "00".repeat(32);
    assert!(c.validate().is_err());
    c = f.c.clone();
    c.runtime_read_roots = vec!["/dev".into()];
    assert!(c.validate().is_err());
    c = f.c.clone();
    c.staging_root = f.root.path().join("alias");
    std::os::unix::fs::symlink(&f.c.staging_root, &c.staging_root).unwrap();
    assert!(c.validate().is_err());
}
#[tokio::test]
async fn synthetic_timeout_cancellation_and_drop_remove_private_staging() {
    let mut f = Fixture::new();
    let p = Policy::new(f.p.clone()).unwrap();
    let call = f.call("camera_clip");
    let r: Request = serde_json::from_value(call.arguments).unwrap();
    f.c.timeout_ms = 1;
    assert!(
        capture::execute(
            &p,
            &f.c,
            &context(),
            &r,
            "camera_clip",
            Some(capture::InternalSource::Synthetic)
        )
        .await
        .is_err()
    );
    assert!(!r.output.exists());
    assert_eq!(fs::read_dir(&f.c.staging_root).unwrap().count(), 0);
    f.c.timeout_ms = 10_000;
    let ctx = context();
    ctx.cancellation.cancel();
    assert!(
        capture::execute(
            &p,
            &f.c,
            &ctx,
            &r,
            "camera_clip",
            Some(capture::InternalSource::Synthetic)
        )
        .await
        .is_err()
    );
    assert!(!r.output.exists());
    // A realtime, synthetic-only source keeps a process alive long enough to
    // abandon its future. No devices or DISPLAY are involved.
    let stage = tempfile::tempdir_in(&f.c.staging_root).unwrap();
    let args = [
        "-nostdin",
        "-v",
        "error",
        "-re",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=5",
        "-t",
        "60",
        "-f",
        "null",
        "-",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let run = process::run(&f.c, &f.c.ffmpeg, args, stage.path(), None, None, &ctx);
    assert!(run.await.is_err());
    let ctx = context();
    let stage_path = stage.path().to_owned();
    let c = f.c.clone();
    let task = tokio::spawn(async move {
        let args = [
            "-nostdin",
            "-v",
            "error",
            "-re",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=5",
            "-t",
            "60",
            "-f",
            "null",
            "-",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        process::run(&c, &c.ffmpeg, args, &stage_path, None, None, &ctx).await
    });
    tokio::time::sleep(Duration::from_millis(150)).await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    let ctx = context();
    let cancellation = ctx.cancellation.clone();
    let args = [
        "-nostdin",
        "-v",
        "error",
        "-re",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=5",
        "-t",
        "60",
        "-f",
        "null",
        "-",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let cancel = async {
        tokio::time::sleep(Duration::from_millis(150)).await;
        cancellation.cancel();
    };
    let (_, result) = tokio::join!(
        cancel,
        process::run(&f.c, &f.c.ffmpeg, args, stage.path(), None, None, &ctx)
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("never automatically retry")
    );
}

#[tokio::test]
async fn byte_limits_and_stdout_drain_fail_closed() {
    let mut f = Fixture::new();
    let policy = Policy::new(f.p.clone()).unwrap();
    let r: Request = serde_json::from_value(f.call("camera_clip").arguments).unwrap();
    f.c.max_output_bytes = 1024;
    assert!(
        capture::execute(
            &policy,
            &f.c,
            &context(),
            &r,
            "camera_clip",
            Some(capture::InternalSource::Synthetic)
        )
        .await
        .is_err()
    );
    assert!(!r.output.exists());
    assert_eq!(fs::read_dir(&f.c.staging_root).unwrap().count(), 0);
    f.c.max_output_bytes = 4 * 1024 * 1024;
    let stage = tempfile::tempdir_in(&f.c.staging_root).unwrap();
    let args = [
        "-nostdin",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=16x16:rate=30",
        "-t",
        "90",
        "-threads",
        "1",
        "-f",
        "framemd5",
        "-",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    assert!(
        process::run(
            &f.c,
            &f.c.ffmpeg,
            args,
            stage.path(),
            None,
            None,
            &context()
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("budget exceeded")
    );
}

struct PrivateXvfb {
    child: Child,
    display: String,
    cookie: PathBuf,
}
impl Drop for PrivateXvfb {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
impl PrivateXvfb {
    async fn start(root: &std::path::Path) -> Self {
        // Probe only socket path occupancy, NEVER connect to an existing server.
        let seed = std::process::id() % 20_000 + 20_000;
        let number = (seed..seed + 100)
            .find(|n| {
                !std::path::Path::new(&format!("/tmp/.X11-unix/X{n}")).exists()
                    && !std::path::Path::new(&format!("/tmp/.X{n}-lock")).exists()
            })
            .unwrap();
        let display = format!(":{number}");
        let cookie = root.join("fresh-Xauthority");
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(&cookie)
            .unwrap();
        use std::io::{Read, Write};
        let mut secret = [0u8; 16];
        fs::File::open("/dev/urandom")
            .unwrap()
            .read_exact(&mut secret)
            .unwrap();
        // Wildcard-family Xauthority record, scoped to our fresh display number.
        file.write_all(&65535u16.to_be_bytes()).unwrap();
        for data in [
            b"".as_slice(),
            number.to_string().as_bytes(),
            b"MIT-MAGIC-COOKIE-1".as_slice(),
            secret.as_slice(),
        ] {
            file.write_all(&(data.len() as u16).to_be_bytes()).unwrap();
            file.write_all(data).unwrap();
        }
        drop(file);
        let mut child = Command::new("/usr/bin/Xvfb")
            .env_clear()
            .args([
                &display,
                "-screen",
                "0",
                "320x240x24",
                "-nolisten",
                "tcp",
                "-noreset",
                "-auth",
            ])
            .arg(&cookie)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let socket = PathBuf::from(format!("/tmp/.X11-unix/X{number}"));
        for _ in 0..100 {
            assert!(
                child.try_wait().unwrap().is_none(),
                "fresh Xvfb failed to start; refusing existing display"
            );
            if socket.exists() {
                return Self {
                    child,
                    display,
                    cookie,
                };
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let _ = child.kill();
        let _ = child.wait();
        panic!("fresh Xvfb readiness timed out");
    }
    fn configure(&self, c: &mut CaptureConfig) {
        c.enable_screencast = true;
        c.x11 = Some(TrustedX11 {
            display: self.display.clone(),
            xauthority: self.cookie.clone(),
            isolated_session: true,
        });
    }
}
use std::fs::OpenOptions;
use std::os::unix::fs::OpenOptionsExt;
struct ExactApproval {
    expected: ToolCall,
    ctx: ToolContext,
    mutate_cookie: Option<PathBuf>,
    deny: bool,
    hits: AtomicUsize,
}
#[async_trait]
impl CaptureApproval for ExactApproval {
    async fn authorize(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        ensure!(
            *call == self.expected
                && ctx.scope == self.ctx.scope
                && ctx.turn_id == self.ctx.turn_id,
            "not exact approval"
        );
        self.hits.fetch_add(1, Ordering::SeqCst);
        if self.deny {
            anyhow::bail!("durable human denied fixture");
        }
        if let Some(p) = &self.mutate_cookie {
            use std::io::Write;
            OpenOptions::new()
                .append(true)
                .open(p)?
                .write_all(b"changed")?;
        }
        Ok(())
    }
    fn configuration_fingerprint(&self) -> String {
        "exact-test-human-approval".into()
    }
}
#[tokio::test]
async fn fresh_authenticated_xvfb_clip_exact_approval_and_postapproval_drift() {
    let mut f = Fixture::new();
    let x = PrivateXvfb::start(f.root.path()).await;
    x.configure(&mut f.c);
    let ctx = context();
    let mut call = f.call("screencast");
    call.arguments.as_object_mut().unwrap().remove("device");
    let approval = Arc::new(ExactApproval {
        expected: call.clone(),
        ctx: ctx.clone(),
        mutate_cookie: None,
        deny: false,
        hits: AtomicUsize::new(0),
    });
    let tools = CaptureTools::new(f.p.clone(), f.c.clone(), approval.clone()).unwrap();
    tools.preflight(&ctx, &call).await.unwrap();
    let value = tools.capture_run(&ctx, &call).await.unwrap();
    assert_eq!(value["capture"], true);
    assert_eq!(value["audio"], false);
    assert_eq!(approval.hits.load(Ordering::SeqCst), 1);
    fs::remove_file(f.root.path().join("clip.mp4")).unwrap();
    let denied = Arc::new(ExactApproval {
        expected: call.clone(),
        ctx: ctx.clone(),
        mutate_cookie: None,
        deny: true,
        hits: AtomicUsize::new(0),
    });
    let tools = CaptureTools::new(f.p.clone(), f.c.clone(), denied).unwrap();
    assert!(
        tools
            .execute(&ctx, &call)
            .await
            .unwrap_err()
            .to_string()
            .contains("denied")
    );
    assert!(!f.root.path().join("clip.mp4").exists());
    let drift = Arc::new(ExactApproval {
        expected: call.clone(),
        ctx: ctx.clone(),
        mutate_cookie: Some(x.cookie.clone()),
        deny: false,
        hits: AtomicUsize::new(0),
    });
    let tools = CaptureTools::new(f.p.clone(), f.c.clone(), drift).unwrap();
    assert!(
        tools
            .execute(&ctx, &call)
            .await
            .unwrap_err()
            .downcast_ref::<ToolDispatchRefused>()
            .is_some()
    );
    assert!(!f.root.path().join("clip.mp4").exists());
}
