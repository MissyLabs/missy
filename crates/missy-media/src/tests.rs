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
fn frame_at(path: &Path, seconds: &str) -> image::RgbImage {
    let output = Command::new("/usr/bin/ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args([
            "-ss",
            seconds,
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-an",
            "-threads",
            "1",
            "-c:v",
            "png",
            "-f",
            "image2pipe",
            "pipe:1",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    image::load_from_memory(&output.stdout).unwrap().to_rgb8()
}

fn decoded_hash(path: &Path, audio: bool) -> String {
    let mut cmd = Command::new("/usr/bin/ffmpeg");
    cmd.args(["-v", "error", "-xerror", "-nostdin", "-i"])
        .arg(path);
    if audio {
        cmd.args([
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le",
        ]);
    } else {
        cmd.args([
            "-map", "0:v:0", "-an", "-pix_fmt", "rgb24", "-threads", "1", "-f", "rawvideo",
        ]);
    }
    let output = cmd.arg("pipe:1").output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!output.stdout.is_empty());
    hex::encode(Sha256::digest(output.stdout))
}

#[tokio::test]
async fn actual_ffmpeg_explicit_audio_speed_text_crossfade_decode_hashes() {
    let dir = tempfile::tempdir().unwrap();
    let mut p = policy(dir.path());
    let font = PathBuf::from("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
    p.read_roots.push(font.parent().unwrap().to_owned());
    let tools = MediaTools::new(p, config(dir.path())).unwrap();
    let red = fixture(dir.path(), "red.mp4", "red");
    let blue = fixture(dir.path(), "blue.mp4", "blue");
    let audio = dir.path().join("short.wav");
    assert!(
        Command::new("/usr/bin/ffmpeg")
            .args([
                "-v",
                "error",
                "-nostdin",
                "-n",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=880:sample_rate=16000:duration=0.2",
                "-c:a",
                "pcm_s16le"
            ])
            .arg(&audio)
            .status()
            .unwrap()
            .success()
    );
    let silent = dir.path().join("silent.mp4");
    tools
        .edit_video(
            &ctx(),
            VideoRequest {
                output: silent.clone(),
                operation: VideoOperation::Transcode {
                    input: red.clone(),
                    width: 64,
                    height: 48,
                },
            },
        )
        .await
        .unwrap();
    let text = |text: &str, preserve_audio| VideoOperation::Text {
        input: red.clone(),
        text: text.into(),
        font_file: font.clone(),
        font_size: 8,
        position: TextPosition::Center,
        font_color: CaptionColor::White,
        box_color: CaptionColor::Black,
        box_opacity: 0.5,
        boxed: true,
        start_ms: 200,
        end_ms: 800,
        preserve_audio,
    };
    let mut results = std::collections::BTreeMap::new();
    for (name, operation, expect_audio, duration) in [
        (
            "mux",
            VideoOperation::AudioMux {
                input: silent.clone(),
                audio_file: audio.clone(),
                gain: 0.5,
                duration_ms: None,
                loop_audio: false,
            },
            true,
            1.0,
        ),
        (
            "loop",
            VideoOperation::AudioMux {
                input: silent.clone(),
                audio_file: audio.clone(),
                gain: 0.5,
                duration_ms: Some(1000),
                loop_audio: true,
            },
            true,
            1.0,
        ),
        (
            "mix",
            VideoOperation::AudioMix {
                input: red.clone(),
                audio_file: audio.clone(),
                gain: 0.5,
                original_gain: 0.25,
                duration_ms: None,
                loop_audio: true,
            },
            true,
            1.0,
        ),
        (
            "mix_silent",
            VideoOperation::AudioMix {
                input: silent.clone(),
                audio_file: audio.clone(),
                gain: 0.5,
                original_gain: 0.25,
                duration_ms: Some(500),
                loop_audio: false,
            },
            true,
            0.5,
        ),
        (
            "speed_quarter",
            VideoOperation::Speed {
                input: red.clone(),
                factor: 0.25,
                preserve_audio: true,
            },
            true,
            4.0,
        ),
        (
            "speed_four",
            VideoOperation::Speed {
                input: red.clone(),
                factor: 4.0,
                preserve_audio: true,
            },
            true,
            0.25,
        ),
        (
            "speed_silent",
            VideoOperation::Speed {
                input: red.clone(),
                factor: 2.0,
                preserve_audio: false,
            },
            false,
            0.5,
        ),
        (
            "speed_no_track",
            VideoOperation::Speed {
                input: silent.clone(),
                factor: 1.0,
                preserve_audio: true,
            },
            false,
            1.0,
        ),
        ("text", text("100% : ' \\ ; [x] %{eif:1}", true), true, 1.0),
        ("text_silent", text("caption", false), false, 1.0),
        (
            "fade",
            VideoOperation::Crossfade {
                inputs: vec![red.clone(), blue.clone()],
                width: 64,
                height: 48,
                duration_ms: 400,
                preserve_audio: true,
            },
            true,
            1.6,
        ),
        (
            "fade_mixed",
            VideoOperation::Crossfade {
                inputs: vec![silent.clone(), blue.clone()],
                width: 64,
                height: 48,
                duration_ms: 400,
                preserve_audio: true,
            },
            true,
            1.6,
        ),
        (
            "fade_silent",
            VideoOperation::Crossfade {
                inputs: vec![red.clone(), blue.clone()],
                width: 64,
                height: 48,
                duration_ms: 1000,
                preserve_audio: false,
            },
            false,
            1.0,
        ),
    ] {
        let result = tools
            .execute(
                &ctx(),
                &call(
                    "video_edit",
                    serde_json::to_value(VideoRequest {
                        output: dir.path().join(format!("{name}.mp4")),
                        operation,
                    })
                    .unwrap(),
                ),
            )
            .await
            .unwrap_or_else(|e| panic!("{name}: {e:#}"));
        assert_artifact(&result, "mpeg4");
        assert_eq!(
            result["media"]["streams"]
                .as_array()
                .unwrap()
                .iter()
                .any(|s| s["codec_type"] == "audio"),
            expect_audio,
            "{name}"
        );
        assert!(
            (result["media"]["duration_seconds"].as_f64().unwrap() - duration).abs() < 0.2,
            "{name}"
        );
        let path = Path::new(result["path"].as_str().unwrap());
        let video_hash = decoded_hash(path, false);
        let audio_hash = expect_audio.then(|| decoded_hash(path, true));
        results.insert(name, (video_hash, audio_hash));
    }
    assert_eq!(
        results["mux"].0, results["loop"].0,
        "mux must not reencode video"
    );
    assert_ne!(
        results["mux"].1, results["loop"].1,
        "loop has real decoded audio beyond short track"
    );
    assert_ne!(
        results["mix"].1, results["loop"].1,
        "mix includes original audio"
    );
    assert_ne!(
        results["text"].0,
        decoded_hash(&red, false),
        "text changes decoded pixels"
    );
    assert_ne!(
        results["fade"].0,
        decoded_hash(&red, false),
        "crossfade renders both clips"
    );
    let text_output = dir.path().join("text.mp4");
    let before = frame_at(&text_output, "0.0");
    let during = frame_at(&text_output, "0.4");
    let after = frame_at(&text_output, "0.9");
    assert_eq!(
        before.as_raw(),
        after.as_raw(),
        "caption disabled outside its interval"
    );
    assert_ne!(
        before.as_raw(),
        during.as_raw(),
        "caption enabled within its interval"
    );
    let fade_output = dir.path().join("fade.mp4");
    let midpoint = frame_at(&fade_output, "0.8");
    let pixel = midpoint.get_pixel(32, 24);
    assert!(
        pixel[0] > 50 && pixel[2] > 50 && pixel[1] < 40,
        "crossfade really blends red and blue: {pixel:?}"
    );
}

#[tokio::test]
async fn new_operations_reject_unbounded_values_and_policy_bypasses() {
    let dir = tempfile::tempdir().unwrap();
    let tools = MediaTools::new(policy(dir.path()), config(dir.path())).unwrap();
    let red = fixture(dir.path(), "red.mp4", "red");
    let outside = tempfile::tempdir().unwrap();
    let external = fixture(outside.path(), "private.mp4", "blue");
    let output = dir.path().join("denied.mp4");
    for arguments in [
        json!({"operation":"speed", "input":red,"output":output,"factor":2.0}),
        json!({"operation":"speed", "input":red,"output":output,"factor":4.1,"preserve_audio":true}),
        json!({"operation":"speed", "input":red,"output":output,"factor":0.0,"preserve_audio":true}),
        json!({"operation":"speed", "input":red,"output":output,"factor":1.0,"preserve_audio":true,"filter":"movie=/etc/passwd"}),
        json!({"operation":"audio_mux", "input":red,"output":output,"audio_file":red,"gain":5,"loop":false}),
        json!({"operation":"crossfade", "inputs":[red],"output":output,"width":64,"height":48,"duration_ms":400,"preserve_audio":false}),
    ] {
        assert!(
            tools
                .preflight(&ctx(), &call("video_edit", arguments))
                .await
                .is_err()
        );
    }
    for operation in [
        VideoOperation::AudioMux {
            input: red.clone(),
            audio_file: external,
            gain: 1.0,
            duration_ms: None,
            loop_audio: false,
        },
        VideoOperation::AudioMix {
            input: red.clone(),
            audio_file: red.clone(),
            gain: 1.0,
            original_gain: f64::NAN,
            duration_ms: None,
            loop_audio: false,
        },
        VideoOperation::Speed {
            input: red.clone(),
            factor: f64::INFINITY,
            preserve_audio: true,
        },
        VideoOperation::Crossfade {
            inputs: vec![red.clone(), red.clone()],
            width: 64,
            height: 48,
            duration_ms: 1100,
            preserve_audio: true,
        },
        VideoOperation::Text {
            input: red.clone(),
            text: "literal:movie=/etc/passwd".into(),
            font_file: "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf".into(),
            font_size: 8,
            position: TextPosition::Center,
            font_color: CaptionColor::White,
            box_color: CaptionColor::Black,
            box_opacity: 0.5,
            boxed: false,
            start_ms: 0,
            end_ms: 800,
            preserve_audio: false,
        },
    ] {
        assert!(
            tools
                .edit_video(
                    &ctx(),
                    VideoRequest {
                        output: output.clone(),
                        operation
                    }
                )
                .await
                .is_err()
        );
        assert!(!output.exists());
    }
    let bad_font = dir.path().join("bad.ttf");
    std::fs::write(&bad_font, b"not a font").unwrap();
    let linked_audio = dir.path().join("link.mp4");
    std::os::unix::fs::symlink(&red, &linked_audio).unwrap();
    for operation in [
        VideoOperation::AudioMux {
            input: red.clone(),
            audio_file: linked_audio,
            gain: 1.0,
            duration_ms: None,
            loop_audio: true,
        },
        VideoOperation::AudioMux {
            input: red.clone(),
            audio_file: red.clone(),
            gain: 1.0,
            duration_ms: Some(1200),
            loop_audio: true,
        },
        VideoOperation::Text {
            input: red,
            text: "caption".into(),
            font_file: bad_font,
            font_size: 8,
            position: TextPosition::Center,
            font_color: CaptionColor::White,
            box_color: CaptionColor::Black,
            box_opacity: 0.5,
            boxed: false,
            start_ms: 0,
            end_ms: 800,
            preserve_audio: false,
        },
    ] {
        assert!(
            tools
                .edit_video(
                    &ctx(),
                    VideoRequest {
                        output: output.clone(),
                        operation
                    }
                )
                .await
                .is_err()
        );
        assert!(!output.exists());
    }
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
    assert!(
        error
            .to_string()
            .contains("outcome unknown; no automatic retry"),
        "{error:#}"
    );
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
    assert!(
        error
            .to_string()
            .contains("outcome unknown; no automatic retry"),
        "{error:#}"
    );
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
