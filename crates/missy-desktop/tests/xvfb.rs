//! This test starts ONLY a new authenticated Xserver. It never reads DISPLAY,
//! XAUTHORITY or an existing session. XTest input goes to its new test window only.
use anyhow::{Result, bail};
use async_trait::async_trait;
use missy_core::{PolicyConfig, Scope, ToolCall, ToolContext, ToolExecutor};
use missy_desktop::{
    DesktopApproval, DesktopConfig, DesktopTools, ScreenInterpreter, ScreenOcr, X11Config,
};
use missy_policy::PolicyHttpClient;
use serde_json::json;
use std::{
    collections::BTreeSet,
    io::{BufRead, BufReader, Write},
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use x11rb::{
    connection::Connection,
    protocol::xproto::*,
    rust_connection::{DefaultStream, RustConnection},
    wrapper::ConnectionExt as _,
};

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
struct Approval(AtomicUsize);
struct FixtureInterpreter(AtomicUsize);
struct FixtureOcr(AtomicUsize);
#[async_trait]
impl ScreenOcr for FixtureOcr {
    async fn recognize_tsv(&self, _: &ToolContext, png: Vec<u8>) -> Result<String> {
        if self.0.fetch_add(1, Ordering::SeqCst) > 0 {
            bail!("{}", "sensitive-unbounded-backend-error".repeat(10000));
        }
        assert_eq!(
            image::load_from_memory(&png)?.to_rgb8().dimensions(),
            (32, 24)
        );
        Ok("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n5\t1\t1\t1\t1\t1\t2\t3\t10\t8\t95\tOpen\n".into())
    }
    fn configuration_fingerprint(&self) -> String {
        "fixture-literal-tsv".into()
    }
}
#[async_trait]
impl ScreenInterpreter for FixtureInterpreter {
    async fn interpret(
        &self,
        _: &ToolContext,
        png: Vec<u8>,
        question: &str,
    ) -> Result<serde_json::Value> {
        self.0.fetch_add(1, Ordering::SeqCst);
        assert_eq!(question, "pixel color?");
        let img = image::load_from_memory(&png)?.to_rgb8();
        assert_eq!(img.dimensions(), (32, 24));
        assert_eq!(img.get_pixel(1, 1).0, [0x33, 0x66, 0x99]);
        Ok(json!({"description":"synthetic blue test window","untrusted":true}))
    }
    fn configuration_fingerprint(&self) -> String {
        "fixture-no-model".into()
    }
}
#[async_trait]
impl DesktopApproval for Approval {
    async fn authorize(&self, _: &ToolContext, _: &ToolCall) -> Result<()> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn configuration_fingerprint(&self) -> String {
        "isolated-test".into()
    }
}
#[tokio::test]
async fn authenticated_isolated_xvfb_reads_and_inputs_only_focused_allowed_window() -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    if !std::path::Path::new("/usr/bin/Xvfb").exists() {
        bail!("Xvfb unavailable: isolated acceptance not run");
    }
    let temp = tempfile::tempdir()?;
    let cookie = [42u8; 16];
    let cookie_path = temp.path().join("cookie");
    std::fs::write(&cookie_path, cookie)?;
    std::fs::set_permissions(&cookie_path, std::fs::Permissions::from_mode(0o600))?;
    // Xauth wildcard record for this fresh server only. Never load user auth.
    let authority = temp.path().join("authority");
    let mut file = std::fs::File::create(&authority)?;
    file.write_all(&65535u16.to_be_bytes())?;
    for value in [&b""[..], &b""[..], &b"MIT-MAGIC-COOKIE-1"[..], &cookie[..]] {
        file.write_all(&(value.len() as u16).to_be_bytes())?;
        file.write_all(value)?;
    }
    std::fs::set_permissions(&authority, std::fs::Permissions::from_mode(0o600))?;
    let child = Command::new("/usr/bin/Xvfb")
        .env_clear()
        .args([
            "-displayfd",
            "1",
            "-screen",
            "0",
            "128x96x24",
            "-nolisten",
            "tcp",
            "-auth",
        ])
        .arg(&authority)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut guard = ChildGuard(child);
    let stdout = guard.0.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let result = BufReader::new(stdout).read_line(&mut line).map(|_| line);
        let _ = tx.send(result);
    });
    let display: u16 = rx
        .recv_timeout(std::time::Duration::from_secs(10))??
        .trim()
        .parse()?;
    let socket = std::path::PathBuf::from(format!("/tmp/.X11-unix/X{display}"));
    let (stream, _) =
        DefaultStream::from_unix_stream(std::os::unix::net::UnixStream::connect(&socket)?)?;
    let c = RustConnection::connect_to_stream_with_auth_info(
        stream,
        0,
        b"MIT-MAGIC-COOKIE-1".to_vec(),
        cookie.to_vec(),
    )?;
    let root = c.setup().roots[0].root;
    let window = c.generate_id()?;
    c.create_window(
        c.setup().roots[0].root_depth,
        window,
        root,
        0,
        0,
        32,
        24,
        0,
        WindowClass::INPUT_OUTPUT,
        0,
        &CreateWindowAux::new()
            .background_pixel(0x336699)
            .event_mask(
                EventMask::KEY_PRESS
                    | EventMask::KEY_RELEASE
                    | EventMask::BUTTON_PRESS
                    | EventMask::BUTTON_RELEASE
                    | EventMask::POINTER_MOTION,
            ),
    )?
    .check()?;
    c.change_property8(
        PropMode::REPLACE,
        window,
        AtomEnum::WM_NAME,
        AtomEnum::STRING,
        b"Isolated Missy Test",
    )?
    .check()?;
    c.change_property8(
        PropMode::REPLACE,
        window,
        AtomEnum::WM_CLASS,
        AtomEnum::STRING,
        b"missy\0MissyTest\0",
    )?
    .check()?;
    c.map_window(window)?.check()?;
    c.flush()?;
    let config = DesktopConfig {
        x11: Some(X11Config {
            socket_path: socket,
            screen: 0,
            cookie_path,
            allowed_classes: BTreeSet::from(["MissyTest".into()]),
            allowed_titles: BTreeSet::from(["Isolated Missy Test".into()]),
            isolated_session: true,
        }),
        enable_screenshots: true,
        enable_input: true,
        approve_all: false,
        ..Default::default()
    };
    let approval = Arc::new(Approval(AtomicUsize::new(0)));
    let interpreter = Arc::new(FixtureInterpreter(AtomicUsize::new(0)));
    let ocr = Arc::new(FixtureOcr(AtomicUsize::new(0)));
    let tools = DesktopTools::new(
        config.clone(),
        PolicyHttpClient::new(PolicyConfig::default())?,
        approval.clone(),
    )?
    .with_screen_interpreter(interpreter.clone())
    .with_screen_ocr(ocr.clone());
    let ctx = ToolContext {
        scope: Scope {
            tenant: "t".into(),
            user: "u".into(),
            channel: "c".into(),
            session: "s".into(),
        },
        turn_id: "test".into(),
        cancellation: Default::default(),
    };
    let windows = tools
        .execute(
            &ctx,
            &ToolCall {
                id: "list".into(),
                name: "desktop_windows".into(),
                arguments: json!({}),
            },
        )
        .await?;
    assert_eq!(windows["data"]["windows"][0]["window"], window);
    assert_eq!(windows["trust"], "untrusted");
    let image = tools
        .execute(
            &ctx,
            &ToolCall {
                id: "image".into(),
                name: "desktop_screenshot".into(),
                arguments: json!({"window":window}),
            },
        )
        .await?;
    assert_eq!(
        image["data"]["artifact"]["media_type"],
        "application/x-x11-zpixmap"
    );
    assert_eq!(image["data"]["window"]["width"], 32);
    assert!(image["data"]["artifact"]["data"].as_str().unwrap().len() > 100);
    assert_eq!(approval.0.load(Ordering::SeqCst), 1);
    let blank = tools
        .execute(
            &ctx,
            &ToolCall {
                id: "blank".into(),
                name: "x11_read_screen".into(),
                arguments: json!({"window":window,"question":"pixel color?"}),
            },
        )
        .await
        .unwrap_err();
    assert!(blank.to_string().contains("blank_capture"));
    assert_eq!(interpreter.0.load(Ordering::SeqCst), 0);
    assert_eq!(ocr.0.load(Ordering::SeqCst), 0);
    // Draw variation in this fresh isolated window only. No existing session.
    let gc = c.generate_id()?;
    c.create_gc(gc, window, &CreateGCAux::new().foreground(0xffffff))?
        .check()?;
    c.poly_fill_rectangle(
        window,
        gc,
        &[Rectangle {
            x: 16,
            y: 0,
            width: 16,
            height: 24,
        }],
    )?
    .check()?;
    c.flush()?;
    let read = tools
        .execute(
            &ctx,
            &ToolCall {
                id: "read-screen".into(),
                name: "x11_read_screen".into(),
                arguments: json!({"window":window,"question":"pixel color?"}),
            },
        )
        .await?;
    assert_eq!(
        read["data"]["analysis"]["description"],
        "synthetic blue test window"
    );
    assert_eq!(interpreter.0.load(Ordering::SeqCst), 1);
    assert_eq!(ocr.0.load(Ordering::SeqCst), 1);
    assert_eq!(read["data"]["ocr"]["availability"], "available");
    assert_eq!(
        read["data"]["ocr"]["coordinate_space"],
        "native_window_pixels"
    );
    assert_eq!(read["data"]["ocr"]["anchors"][0]["text"], "Open");
    assert_eq!(read["data"]["ocr"]["anchors"][0]["center_x"], 7);
    assert_eq!(read["data"]["ocr"]["anchors"][0]["center_y"], 7);
    let status = tools
        .execute(
            &ctx,
            &ToolCall {
                id: "status".into(),
                name: "desktop_status".into(),
                arguments: json!({}),
            },
        )
        .await?;
    assert_eq!(status["data"]["x11"]["endpoint_health"], "reachable");
    assert_eq!(status["data"]["x11"]["isolation_attested"], true);
    assert_eq!(status["data"]["ambient_discovery"], false);
    assert_eq!(status["data"]["input_dispatched"], false);
    // OCR-only errors remain unknown, not made-up labels or raw exception text.
    let ocr_only = DesktopTools::new(
        config,
        PolicyHttpClient::new(PolicyConfig::default())?,
        approval.clone(),
    )?
    .with_screen_ocr(ocr.clone());
    let unknown = ocr_only
        .execute(
            &ctx,
            &ToolCall {
                id: "unknown-ocr".into(),
                name: "x11_read_screen".into(),
                arguments: json!({"window":window,"question":"labels?"}),
            },
        )
        .await?;
    assert!(unknown["data"]["analysis"].is_null());
    assert_eq!(unknown["data"]["ocr"]["availability"], "unavailable");
    assert_eq!(unknown["data"]["ocr"]["anchors"], json!([]));
    assert_eq!(
        unknown["data"]["ocr"]["diagnostic"],
        "ocr_backend_failed_or_invalid_tsv"
    );
    assert!(!unknown.to_string().contains("sensitive-unbounded"));
    tools
        .execute(
            &ctx,
            &ToolCall {
                id: "focus".into(),
                name: "desktop_focus_window".into(),
                arguments: json!({"window":window}),
            },
        )
        .await?;
    assert_eq!(c.get_input_focus()?.reply()?.focus, window);
    let input = |args| ToolCall {
        id: "input".into(),
        name: "desktop_input".into(),
        arguments: args,
    };
    // No focus is stolen by the executor. Only this test's connection sets it.
    c.set_input_focus(InputFocus::POINTER_ROOT, root, x11rb::CURRENT_TIME)?
        .check()?;
    assert!(
        tools
            .execute(
                &ctx,
                &input(json!({"window":window,"kind":"move","x":5,"y":6}))
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("not already focused")
    );
    c.set_input_focus(InputFocus::PARENT, window, x11rb::CURRENT_TIME)?
        .check()?;
    assert_eq!(c.get_input_focus()?.reply()?.focus, window);
    tools
        .execute(
            &ctx,
            &input(json!({"window":window,"kind":"move","x":5,"y":6})),
        )
        .await?;
    let pointer = c.query_pointer(window)?.reply()?;
    assert_eq!((pointer.win_x, pointer.win_y), (5, 6));
    tools
        .execute(
            &ctx,
            &input(json!({"window":window,"kind":"drag","code":1,"x":5,"y":6,"to_x":20,"to_y":15})),
        )
        .await?;
    let pointer = c.query_pointer(window)?.reply()?;
    assert_eq!((pointer.win_x, pointer.win_y), (20, 15));
    assert_eq!(u16::from(pointer.mask), 0);
    tools
        .execute(
            &ctx,
            &input(json!({"window":window,"kind":"type_text","text":"Az 9!"})),
        )
        .await?;
    c.get_input_focus()?.reply()?;
    let mut key_presses = 0;
    let mut key_releases = 0;
    let mut buttons = 0;
    while let Some(event) = c.poll_for_event()? {
        match event {
            x11rb::protocol::Event::KeyPress(_) => key_presses += 1,
            x11rb::protocol::Event::KeyRelease(_) => key_releases += 1,
            x11rb::protocol::Event::ButtonPress(_) => buttons += 1,
            _ => {}
        }
    }
    assert!(key_presses >= 5);
    assert_eq!(key_presses, key_releases);
    assert_eq!(buttons, 1);
    assert_eq!(u16::from(c.query_pointer(window)?.reply()?.mask), 0);
    for args in [
        json!({"window":window,"kind":"move","x":32,"y":0}),
        json!({"window":window,"kind":"drag","code":1,"x":0,"y":0,"to_x":32,"to_y":0}),
        json!({"window":window,"kind":"type_text","text":"newline\n"}),
        json!({"window":root,"kind":"move","x":0,"y":0}),
    ] {
        assert!(tools.execute(&ctx, &input(args)).await.is_err());
    }
    c.change_property8(
        PropMode::REPLACE,
        window,
        AtomEnum::WM_NAME,
        AtomEnum::STRING,
        b"Not allowed",
    )?
    .check()?;
    let call = ToolCall {
        id: "deny".into(),
        name: "desktop_screenshot".into(),
        arguments: json!({"window":window}),
    };
    assert!(
        tools
            .execute(&ctx, &call)
            .await
            .unwrap_err()
            .to_string()
            .contains("denied")
    );
    let root_call = ToolCall {
        id: "root".into(),
        name: "desktop_screenshot".into(),
        arguments: json!({"window":root}),
    };
    assert!(tools.execute(&ctx, &root_call).await.is_err());
    Ok(())
}
