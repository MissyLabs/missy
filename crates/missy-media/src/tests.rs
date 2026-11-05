use super::*;
use missy_core::Scope;
use std::{io::Cursor, path::Path, process::Command};
use tokio_util::sync::CancellationToken;

fn ctx() -> ToolContext {
    ToolContext {
        scope: Scope {
            tenant: "test".into(),
            user: "test".into(),
            channel: "test".into(),
            session: "test".into(),
        },
        turn_id: "media-test".into(),
        cancellation: CancellationToken::new(),
    }
}
fn pin(path: &str) -> ExecutablePin {
    ExecutablePin {
        path: PathBuf::from(path),
        sha256: hex::encode(Sha256::digest(std::fs::read(path).unwrap())),
    }
}
fn config(dir: &Path) -> MediaConfig {
    MediaConfig {
        ffmpeg: pin("/usr/bin/ffmpeg"),
        ffprobe: pin("/usr/bin/ffprobe"),
        staging_root: dir.to_owned(),
        runtime_read_roots: vec!["/usr/bin".into(), "/usr/lib".into()],
        timeout_ms: 15000,
        max_duration_seconds: 10,
        max_pixels: 512 * 512,
        max_output_bytes: 4 * 1024 * 1024,
    }
}
fn policy(dir: &Path) -> PolicyConfig {
    PolicyConfig {
        read_roots: vec![dir.to_owned()],
        write_roots: vec![dir.to_owned()],
        max_file_bytes: 4 * 1024 * 1024,
        ..Default::default()
    }
}
fn fixture(dir: &Path, name: &str, color: &str) -> PathBuf {
    // TEST ONLY trusted constant lavfi source. No capture, device or playback.
    let path = dir.join(name);
    let status = Command::new("/usr/bin/ffmpeg")
        .args([
            "-v",
            "error",
            "-nostdin",
            "-n",
            "-f",
            "lavfi",
            "-i",
            &format!("color=c={color}:s=64x48:r=15:d=1"),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=1",
            "-c:v",
            "mpeg4",
            "-q:v",
            "4",
            "-threads",
            "1",
            "-c:a",
            "aac",
            "-shortest",
        ])
        .arg(&path)
        .status()
        .unwrap();
    assert!(status.success());
    path
}
fn image_fixture(dir: &Path, name: &str, color: [u8; 4]) -> PathBuf {
    let path = dir.join(name);
    image::RgbaImage::from_pixel(16, 16, image::Rgba(color))
        .save(&path)
        .unwrap();
    path
}
fn call(name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: "test".into(),
        name: name.into(),
        arguments,
    }
}
fn assert_artifact(result: &Value, codec: &str) {
    let bytes = std::fs::read(result["path"].as_str().unwrap()).unwrap();
    assert_eq!(result["bytes"].as_u64().unwrap(), bytes.len() as u64);
    assert_eq!(
        result["sha256"].as_str().unwrap(),
        hex::encode(Sha256::digest(&bytes))
    );
    assert_eq!(result["media"]["streams"][0]["codec_name"], codec);
}

#[tokio::test]
async fn actual_ffmpeg_all_edit_operations_decode_and_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let tools = MediaTools::new(policy(dir.path()), config(dir.path())).unwrap();
    let red = fixture(dir.path(), "red.mp4", "red");
    let blue = fixture(dir.path(), "blue.mp4", "blue");
    let green = image_fixture(dir.path(), "green.png", [0, 255, 0, 255]);
    let cases = vec![
        (
            "trim.mp4",
            "mpeg4",
            VideoOperation::Trim {
                input: red.clone(),
                start_ms: 0,
                duration_ms: 500,
            },
        ),
        (
            "transcode.mp4",
            "mpeg4",
            VideoOperation::Transcode {
                input: red.clone(),
                width: 32,
                height: 32,
            },
        ),
        (
            "concat.mp4",
            "mpeg4",
            VideoOperation::Concat {
                inputs: vec![red.clone(), blue.clone()],
                width: 64,
                height: 48,
            },
        ),
        (
            "overlay.mp4",
            "mpeg4",
            VideoOperation::Overlay {
                input: red.clone(),
                image: green.clone(),
                x: 0,
                y: 0,
            },
        ),
        (
            "frame.png",
            "png",
            VideoOperation::ExtractFrame {
                input: red.clone(),
                at_ms: 100,
            },
        ),
        (
            "audio.wav",
            "pcm_s16le",
            VideoOperation::ExtractAudio { input: red.clone() },
        ),
        (
            "story.mp4",
            "mpeg4",
            VideoOperation::Storyboard {
                images: vec![green.clone(), green],
                seconds_per_image: 1,
                width: 64,
                height: 48,
            },
        ),
    ];
    for (name, codec, operation) in cases {
        let result = tools
            .edit_video(
                &ctx(),
                VideoRequest {
                    output: dir.path().join(name),
                    operation,
                },
            )
            .await
            .unwrap_or_else(|e| panic!("{name}: {e:#}"));
        assert_artifact(&result, codec);
        if codec != "png" {
            assert!(result["media"]["duration_seconds"].as_f64().unwrap() > 0.0);
        }
    }
    let frame = image::open(dir.path().join("frame.png")).unwrap().to_rgb8();
    let px = frame.get_pixel(32, 24);
    assert!(
        px[0] > 200 && px[1] < 50 && px[2] < 50,
        "real red fixture decoded"
    );
}

#[tokio::test]
async fn image_file_capture_native_edits_and_policy_denial() {
    let dir = tempfile::tempdir().unwrap();
    let tools = MediaTools::new(policy(dir.path()), config(dir.path())).unwrap();
    let input = image_fixture(dir.path(), "source.png", [255, 0, 0, 255]);
    let result = tools
        .execute(
            &ctx(),
            &call(
                "vision_capture",
                json!({"input":input,"output":dir.path().join("copy.jpg")}),
            ),
        )
        .await
        .unwrap();
    assert_artifact(&result, "jpeg");
    for (name, operation) in [
        (
            "resize.png",
            ImageOperation::Resize {
                width: 8,
                height: 8,
            },
        ),
        (
            "crop.png",
            ImageOperation::Crop {
                x: 2,
                y: 2,
                width: 4,
                height: 4,
            },
        ),
        (
            "rotate.png",
            ImageOperation::Rotate {
                degrees: Rotation::R90,
            },
        ),
        (
            "overlay.png",
            ImageOperation::Overlay {
                overlay: input.clone(),
                x: 0,
                y: 0,
            },
        ),
    ] {
        let result = tools
            .edit_image(ImageRequest {
                input: input.clone(),
                output: dir.path().join(name),
                operation,
            })
            .unwrap();
        assert_artifact(&result, "png");
    }
    let external = tempfile::tempdir().unwrap();
    let outside = image_fixture(external.path(), "private.png", [0, 0, 255, 255]);
    assert!(
        tools
            .edit_image(ImageRequest {
                input: outside.clone(),
                output: dir.path().join("bad.png"),
                operation: ImageOperation::Copy
            })
            .is_err()
    );
    assert!(
        tools
            .edit_image(ImageRequest {
                input: input.clone(),
                output: outside.clone(),
                operation: ImageOperation::Copy
            })
            .is_err()
    );
    let link = dir.path().join("link.png");
    std::os::unix::fs::symlink(&outside, &link).unwrap();
    assert!(
        tools
            .edit_image(ImageRequest {
                input: link.clone(),
                output: dir.path().join("bad.png"),
                operation: ImageOperation::Copy
            })
            .is_err()
    );
    assert!(
        tools
            .edit_image(ImageRequest {
                input,
                output: link,
                operation: ImageOperation::Copy
            })
            .is_err()
    );
}

#[tokio::test]
async fn strict_typed_parameters_and_cancellation() {
    let dir = tempfile::tempdir().unwrap();
    let tools = MediaTools::new(policy(dir.path()), config(dir.path())).unwrap();
    for arguments in [
        json!({"operation":"trim","input":"/tmp/a.mp4","output":"/tmp/o.mp4","start_ms":0,"duration_ms":100,"vf":"movie=/etc/passwd"}),
        json!({"operation":"trim","input":"https://evil/a.mp4","output":"/tmp/o.mp4","start_ms":0,"duration_ms":100}),
        json!({"operation":"transcode","input":"/tmp/a.m3u8","output":"/tmp/o.mp4","width":64,"height":48}),
        json!({"operation":"transcode","input":"/tmp/a.mp4","output":"/tmp/o.mp4","width":999999,"height":48}),
        json!({"operation":"trim","input":"/tmp/a.mp4","output":"/tmp/o.mp4","start_ms":0,"duration_ms":-1}),
    ] {
        assert!(
            tools
                .preflight(&ctx(), &call("video_edit", arguments))
                .await
                .is_err()
        );
    }
    assert!(
        tools
            .preflight(
                &ctx(),
                &call(
                    "vision_capture",
                    json!({"input":"/tmp/a.png","output":"/tmp/o.png","device":"/dev/video0"})
                )
            )
            .await
            .is_err()
    );
    assert!(tools.preflight(&ctx(),&call("image_edit",json!({"input":"/tmp/a.png","output":"/tmp/o.png","operation":"copy","filter":"movie=/etc/passwd"}))).await.is_err());
    let c = ctx();
    c.cancellation.cancel();
    assert!(
        tools
            .execute(&c, &call("vision_devices", json!({})))
            .await
            .is_err()
    );
    for d in tools.definitions() {
        assert_eq!(d.side_effect, d.name != "vision_devices");
    }
}

#[tokio::test]
async fn disguised_playlist_and_external_reference_cannot_read_secrets() {
    let dir = tempfile::tempdir().unwrap();
    let tools = MediaTools::new(policy(dir.path()), config(dir.path())).unwrap();
    let malicious = dir.path().join("playlist.mp4");
    std::fs::write(&malicious, b"#EXTM3U\n#EXTINF:1,\nfile:///etc/passwd\n").unwrap();
    let output = dir.path().join("bad.mp4");
    assert!(
        tools
            .edit_video(
                &ctx(),
                VideoRequest {
                    output: output.clone(),
                    operation: VideoOperation::Trim {
                        input: malicious,
                        start_ms: 0,
                        duration_ms: 100
                    }
                }
            )
            .await
            .is_err()
    );
    assert!(!output.exists());
    assert!(std::fs::read_dir(dir.path()).unwrap().all(|e| {
        !e.unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with("missy-media-")
    }));
}

#[tokio::test]
async fn adapter_really_confines_child_and_uses_exact_argv() {
    let dir = tempfile::tempdir().unwrap();
    let c = config(dir.path());
    // Only test code supplies arbitrary argv. Production adapter is private and
    // only typed builders call it. External file is inaccessible to the child.
    let outside = tempfile::tempdir().unwrap();
    let video = fixture(outside.path(), "private.mp4", "red");
    let args = vec![
        "-v".into(),
        "error".into(),
        "-show_format".into(),
        video.to_string_lossy().into_owned(),
    ];
    let error = process::run(&c, &c.ffprobe, args, dir.path(), &ctx())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("Permission denied"), "{error:#}");
}

#[test]
fn config_pin_and_decompression_bounds_fail_closed() {
    let dir = tempfile::tempdir().unwrap();
    let mut c = config(dir.path());
    c.ffmpeg.sha256 = "00".repeat(32);
    assert!(MediaTools::new(policy(dir.path()), c).is_err());
    let mut c = config(dir.path());
    c.runtime_read_roots.push("/".into());
    assert!(c.validate().is_err());
    let c = config(dir.path());
    let mut png = Cursor::new(Vec::new());
    image::RgbaImage::new(513, 513)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    assert!(image_ops::decode(png.get_ref(), &c).is_err());
    assert!(image_ops::decode(b"not an image", &c).is_err());
}

#[tokio::test]
async fn actual_native_process_timeout_and_cancellation() {
    let dir = tempfile::tempdir().unwrap();
    let mut c = config(dir.path());
    // TEST ONLY unbounded lavfi. Never exposed by a production typed builder.
    let args = || {
        vec![
            "-v".into(),
            "error".into(),
            "-nostdin".into(),
            "-re".into(),
            "-f".into(),
            "lavfi".into(),
            "-i".into(),
            "color=c=red:s=16x16:r=15".into(),
            "-f".into(),
            "null".into(),
            "pipe:1".into(),
        ]
    };
    c.timeout_ms = 100;
    let start = std::time::Instant::now();
    let error = process::run(&c, &c.ffmpeg, args(), dir.path(), &ctx())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("timed out"), "{error:#}");
    assert!(start.elapsed() < std::time::Duration::from_secs(3));
    c.timeout_ms = 15000;
    let context = ctx();
    let token = context.cancellation.clone();
    let cancel = tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        token.cancel();
    });
    let error = process::run(&c, &c.ffmpeg, args(), dir.path(), &context)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("cancelled"), "{error:#}");
    cancel.await.unwrap();
    // Dropped future uses the same exact-group kill/reap ownership path.
    let context = ctx();
    let job = process::run(&c, &c.ffmpeg, args(), dir.path(), &context);
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(100), job)
            .await
            .is_err()
    );
}
