use crate::{CaptureConfig, Request, process};
use anyhow::{Result, ensure};
use missy_core::ToolContext;
use missy_policy::Policy;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs::OpenOptions, io::Write, os::unix::fs::OpenOptionsExt};

// Only unit tests can construct the synthetic source. Production has neither a
// lavfi argument nor a file/device bypass in its tool schema.
pub(crate) enum InternalSource {
    #[cfg(test)]
    Synthetic,
}
fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}
pub(crate) async fn execute(
    policy: &Policy,
    c: &CaptureConfig,
    ctx: &ToolContext,
    r: &Request,
    name: &str,
    source: Option<InternalSource>,
) -> Result<Value> {
    let stage = tempfile::Builder::new()
        .prefix("capture-")
        .tempdir_in(&c.staging_root)?;
    let photo = name == "camera_photo";
    let destination = stage
        .path()
        .join(if photo { "capture.png" } else { "capture.mp4" });
    let mut args = strings(&[
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-xerror",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-filter_complex_threads",
        "1",
    ]);
    let mut device = None;
    let mut cookie_path = None;
    match source {
        #[cfg(test)]
        Some(InternalSource::Synthetic) => {
            args.extend(strings(&["-f", "lavfi", "-i"]));
            args.push(format!(
                "testsrc=size={}x{}:rate={}",
                r.width, r.height, r.fps
            ));
        }
        None => {
            if name == "screencast" {
                let x = c.x11.as_ref().expect("validated X11");
                let cookie = stage.path().join("xauthority");
                let mut f = OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .mode(0o600)
                    .open(&cookie)?;
                f.write_all(&x.cookie()?)?;
                f.sync_all()?;
                cookie_path = Some(cookie);
                args.extend(strings(&[
                    "-f",
                    "x11grab",
                    "-draw_mouse",
                    "0",
                    "-framerate",
                ]));
                args.push(r.fps.to_string());
                args.push("-video_size".into());
                args.push(format!("{}x{}", r.width, r.height));
                args.push("-i".into());
                args.push(format!("unix{}+0,0", x.display));
            } else {
                let path = r.device.as_ref().expect("validated camera");
                crate::config::camera_live(path)?;
                device = Some(path.as_path());
                args.extend(strings(&["-f", "video4linux2", "-framerate"]));
                args.push(r.fps.to_string());
                args.push("-video_size".into());
                args.push(format!("{}x{}", r.width, r.height));
                args.push("-i".into());
                args.push(path.to_string_lossy().into_owned());
            }
        }
    }
    args.extend(strings(&[
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-threads",
        "1",
    ]));
    if photo {
        args.extend(strings(&[
            "-frames:v",
            "1",
            "-c:v",
            "png",
            "-f",
            "image2",
            "-update",
            "1",
        ]));
    } else {
        args.push("-t".into());
        args.push(r.duration_seconds.to_string());
        args.extend(strings(&[
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
        ]));
    }
    args.push("-fs".into());
    args.push(c.max_output_bytes.to_string());
    args.push(destination.to_string_lossy().into_owned());
    process::run(
        c,
        &c.ffmpeg,
        args,
        stage.path(),
        device,
        cookie_path.as_deref(),
        ctx,
    )
    .await?;
    let probe = process::run(
        c,
        &c.ffprobe,
        strings(&[
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            destination.to_str().expect("UTF8 staging"),
        ]),
        stage.path(),
        None,
        None,
        ctx,
    )
    .await?;
    let metadata: Value = serde_json::from_slice(&probe)?;
    let streams = metadata["streams"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("ffprobe streams missing"))?;
    ensure!(
        streams.len() == 1 && streams[0]["codec_type"] == "video",
        "capture must contain exactly one video stream and NO audio/data"
    );
    let stream = &streams[0];
    ensure!(
        stream["width"].as_u64() == Some(r.width as u64)
            && stream["height"].as_u64() == Some(r.height as u64),
        "capture dimensions mismatch"
    );
    ensure!(
        stream["codec_name"] == if photo { "png" } else { "mpeg4" },
        "capture codec mismatch"
    );
    let frames: u64 = stream["nb_read_frames"].as_str().unwrap_or("").parse()?;
    ensure!(
        frames > 0 && frames <= r.duration_seconds as u64 * r.fps as u64 + 2,
        "capture frame budget violated"
    );
    let duration = if photo {
        ensure!(frames == 1, "photo must contain exactly one frame");
        0.0
    } else {
        let d: f64 = metadata["format"]["duration"]
            .as_str()
            .unwrap_or("")
            .parse()?;
        ensure!(
            d.is_finite()
                && d > 0.0
                && d <= r.duration_seconds as f64 + 0.15
                && d >= r.duration_seconds as f64 - 0.25,
            "capture incomplete or duration outside requested bounds"
        );
        d
    };
    // ffprobe -count_frames traverses the complete stream, then a separate strict
    // full decode rejects corrupt/truncated frames rather than trusting headers.
    process::run(
        c,
        &c.ffmpeg,
        strings(&[
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-threads",
            "1",
            "-i",
            destination.to_str().expect("UTF8 staging"),
            "-map",
            "0:v:0",
            "-an",
            "-f",
            "null",
            "-",
        ]),
        stage.path(),
        None,
        None,
        ctx,
    )
    .await?;
    let bytes = process::read_bounded(&destination, c.max_output_bytes)?;
    ensure!(
        !ctx.cancellation.is_cancelled(),
        "capture cancelled before publication; capture may have occurred; never retry"
    );
    policy.write_file(&r.output, &bytes)?;
    Ok(
        json!({"trust":"untrusted","path":r.output,"bytes":bytes.len(),"sha256":hex::encode(Sha256::digest(&bytes)),"duration_seconds":duration,"frames":frames,"audio":false,"media":metadata,"sandbox":{"backend":"linux-landlock","tcp_denied":true},"capture":true}),
    )
}
