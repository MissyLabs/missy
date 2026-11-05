use crate::{MediaConfig, extension, image_ops, object_schema, path_schema, process, publish};
use anyhow::{Result, bail, ensure};
use missy_core::ToolContext;
use missy_policy::Policy;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize)]
pub struct VideoRequest {
    pub output: PathBuf,
    #[serde(flatten)]
    pub operation: VideoOperation,
}
impl<'de> Deserialize<'de> for VideoRequest {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let mut value = Value::deserialize(deserializer)?;
        let map = value
            .as_object_mut()
            .ok_or_else(|| serde::de::Error::custom("object required"))?;
        let allowed: &[&str] = match map.get("operation").and_then(Value::as_str) {
            Some("trim") => &["output", "operation", "input", "start_ms", "duration_ms"],
            Some("transcode") => &["output", "operation", "input", "width", "height"],
            Some("concat") => &["output", "operation", "inputs", "width", "height"],
            Some("overlay") => &["output", "operation", "input", "image", "x", "y"],
            Some("extract_frame") => &["output", "operation", "input", "at_ms"],
            Some("extract_audio") => &["output", "operation", "input"],
            Some("storyboard") => &[
                "output",
                "operation",
                "images",
                "seconds_per_image",
                "width",
                "height",
            ],
            _ => return Err(serde::de::Error::custom("unknown video operation")),
        };
        if map.keys().any(|k| !allowed.contains(&k.as_str())) {
            return Err(serde::de::Error::custom("unknown video operation field"));
        }
        let output = serde_json::from_value(map.remove("output").unwrap_or(Value::Null))
            .map_err(serde::de::Error::custom)?;
        let operation = serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        Ok(Self { output, operation })
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum VideoOperation {
    Trim {
        input: PathBuf,
        start_ms: u32,
        duration_ms: u32,
    },
    Transcode {
        input: PathBuf,
        width: u32,
        height: u32,
    },
    Concat {
        inputs: Vec<PathBuf>,
        width: u32,
        height: u32,
    },
    Overlay {
        input: PathBuf,
        image: PathBuf,
        x: u32,
        y: u32,
    },
    ExtractFrame {
        input: PathBuf,
        at_ms: u32,
    },
    ExtractAudio {
        input: PathBuf,
    },
    Storyboard {
        images: Vec<PathBuf>,
        seconds_per_image: u32,
        width: u32,
        height: u32,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamMetadata {
    pub codec_type: String,
    pub codec_name: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MediaMetadata {
    pub duration_seconds: Option<f64>,
    pub format: String,
    pub streams: Vec<StreamMetadata>,
}

fn input_format(path: &Path) -> Result<&'static str> {
    match extension(path)? {
        "mp4" | "mov" => Ok("mov"),
        "mkv" | "webm" => Ok("matroska"),
        "wav" => Ok("wav"),
        _ => bail!("only MP4/MOV/MKV/WebM/WAV media inputs allowed; no playlists"),
    }
}
fn video_dimensions(w: u32, h: u32, c: &MediaConfig) -> Result<()> {
    image_ops::dimensions(w, h, c)?;
    ensure!(
        w >= 2 && h >= 2 && w <= 1920 && h <= 1920 && w.is_multiple_of(2) && h.is_multiple_of(2),
        "video dimensions must be even, 2..1920"
    );
    Ok(())
}
impl VideoRequest {
    pub fn validate(&self, c: &MediaConfig) -> Result<()> {
        let expected = match self.operation {
            VideoOperation::ExtractAudio { .. } => "wav",
            VideoOperation::ExtractFrame { .. } => "png",
            _ => "mp4",
        };
        ensure!(
            extension(&self.output)? == expected,
            "output extension must be {expected}"
        );
        match &self.operation {
            VideoOperation::Trim {
                input,
                start_ms,
                duration_ms,
            } => {
                input_format(input)?;
                ensure!(
                    *duration_ms > 0
                        && *duration_ms <= c.max_duration_seconds * 1000
                        && start_ms
                            .checked_add(*duration_ms)
                            .is_some_and(|v| v <= c.max_duration_seconds * 1000),
                    "trim range outside bounds"
                );
            }
            VideoOperation::Transcode {
                input,
                width,
                height,
            } => {
                input_format(input)?;
                video_dimensions(*width, *height, c)?;
            }
            VideoOperation::Concat {
                inputs,
                width,
                height,
            } => {
                ensure!((2..=8).contains(&inputs.len()), "concat needs 2..8 inputs");
                for input in inputs {
                    input_format(input)?;
                }
                video_dimensions(*width, *height, c)?;
            }
            VideoOperation::Overlay { input, image, x, y } => {
                input_format(input)?;
                ensure!(
                    matches!(extension(image)?, "png" | "jpg" | "jpeg"),
                    "overlay PNG/JPEG required"
                );
                ensure!(
                    *x <= 8192 && *y <= 8192,
                    "overlay coordinate outside bounds"
                );
            }
            VideoOperation::ExtractFrame { input, at_ms } => {
                input_format(input)?;
                ensure!(
                    *at_ms <= c.max_duration_seconds * 1000,
                    "frame timestamp outside bounds"
                );
            }
            VideoOperation::ExtractAudio { input } => {
                input_format(input)?;
            }
            VideoOperation::Storyboard {
                images,
                seconds_per_image,
                width,
                height,
            } => {
                ensure!(
                    (1..=8).contains(&images.len())
                        && (1..=30).contains(seconds_per_image)
                        && images.len() as u32 * seconds_per_image <= c.max_duration_seconds,
                    "storyboard bounds exceeded"
                );
                for image in images {
                    ensure!(
                        matches!(extension(image)?, "png" | "jpg" | "jpeg"),
                        "storyboard requires PNG/JPEG files"
                    );
                }
                video_dimensions(*width, *height, c)?;
            }
        }
        Ok(())
    }
}
fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).into()).collect()
}
fn common() -> Vec<String> {
    strings(&[
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-xerror",
        "-max_alloc",
        "67108864",
        "-filter_threads",
        "1",
        "-filter_complex_threads",
        "1",
    ])
}
fn input_args(args: &mut Vec<String>, path: &Path, format: &str) {
    args.extend(strings(&[
        "-protocol_whitelist",
        "file,pipe",
        "-format_whitelist",
        "mov,matroska,wav,png_pipe,jpeg_pipe",
        "-probesize",
        "1048576",
        "-analyzeduration",
        "1000000",
        "-threads",
        "1",
        "-f",
        format,
    ]));
    // Explicitly deny QuickTime external data references. No concat/image2/HLS
    // demuxer and no autodetection: suffix changes cannot activate a playlist.
    if format == "mov" {
        args.extend(strings(&["-enable_drefs", "0", "-use_absolute_path", "0"]));
    }
    args.push("-i".into());
    args.push(path.to_string_lossy().into_owned());
}
fn scale(w: u32, h: u32) -> String {
    format!(
        "scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=15,format=yuv420p,setpts=PTS-STARTPTS"
    )
}
async fn probe(
    c: &MediaConfig,
    staging: &Path,
    path: &Path,
    format: &str,
    ctx: &ToolContext,
) -> Result<MediaMetadata> {
    let mut args = strings(&["-v", "error", "-max_alloc", "67108864"]);
    input_args(&mut args, path, format);
    args.extend(strings(&["-show_entries","format=duration,format_name:stream=codec_type,codec_name,width,height,sample_rate,channels","-of","json"]));
    let result = process::run(c, &c.ffprobe, args, staging, ctx).await?;
    let value: Value = serde_json::from_str(result["stdout"].as_str().unwrap_or(""))?;
    let streams = value["streams"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing media streams"))?;
    ensure!(
        !streams.is_empty() && streams.len() <= 16,
        "media stream count outside bounds"
    );
    let mut parsed = Vec::new();
    for s in streams {
        let width = s["width"].as_u64().map(u32::try_from).transpose()?;
        let height = s["height"].as_u64().map(u32::try_from).transpose()?;
        if let (Some(w), Some(h)) = (width, height) {
            image_ops::dimensions(w, h, c)?;
        }
        let sample_rate = s["sample_rate"]
            .as_str()
            .map(str::parse::<u32>)
            .transpose()?;
        let channels = s["channels"].as_u64().map(u32::try_from).transpose()?;
        ensure!(
            sample_rate.is_none_or(|s| s <= 192000) && channels.is_none_or(|n| n <= 8),
            "audio format outside bounds"
        );
        parsed.push(StreamMetadata {
            codec_type: s["codec_type"].as_str().unwrap_or("").into(),
            codec_name: s["codec_name"].as_str().unwrap_or("").into(),
            width,
            height,
            sample_rate,
            channels,
        });
    }
    let duration = value["format"]["duration"]
        .as_str()
        .map(str::parse::<f64>)
        .transpose()?;
    if let Some(d) = duration {
        ensure!(
            d.is_finite() && d > 0.0 && d <= f64::from(c.max_duration_seconds) + 0.1,
            "media duration outside bounds"
        );
    }
    Ok(MediaMetadata {
        duration_seconds: duration,
        format: value["format"]["format_name"].as_str().unwrap_or("").into(),
        streams: parsed,
    })
}

async fn stage_video(
    policy: &Policy,
    c: &MediaConfig,
    dir: &Path,
    input: &Path,
    index: usize,
    ctx: &ToolContext,
) -> Result<(PathBuf, MediaMetadata)> {
    let format = input_format(input)?;
    let path = dir.join(format!("input-{index}.bin"));
    // Policy opens once through O_NOFOLLOW descriptors and caps the bytes. The
    // decoder receives only this private copy, never the original pathname.
    std::fs::write(&path, policy.read_file(input)?)?;
    let media = probe(c, dir, &path, format, ctx).await?;
    ensure!(
        media.duration_seconds.is_some(),
        "bounded finite media duration required"
    );
    Ok((path, media))
}
fn require_video(media: &MediaMetadata) -> Result<&StreamMetadata> {
    media
        .streams
        .iter()
        .find(|s| s.codec_type == "video")
        .ok_or_else(|| anyhow::anyhow!("video stream required"))
}
fn video_encoding(args: &mut Vec<String>) {
    args.extend(strings(&[
        "-an", "-sn", "-dn", "-c:v", "mpeg4", "-q:v", "4", "-pix_fmt", "yuv420p", "-threads", "1",
        "-r", "15", "-f", "mp4",
    ]));
}

pub(crate) async fn execute(
    policy: &Policy,
    c: &MediaConfig,
    ctx: &ToolContext,
    r: VideoRequest,
) -> Result<Value> {
    ensure!(!ctx.cancellation.is_cancelled(), "media cancelled");
    r.validate(c)?;
    let staging = tempfile::Builder::new()
        .prefix("missy-media-")
        .tempdir_in(&c.staging_root)?;
    let dir = staging.path();
    let output = dir.join("result.bin");
    let mut args = common();
    let mut output_format = "mov";
    let mut expected_codec = "mpeg4";
    let expected_duration;
    match &r.operation {
        VideoOperation::Trim {
            input,
            start_ms,
            duration_ms,
        } => {
            let (p, m) = stage_video(policy, c, dir, input, 0, ctx).await?;
            let v = require_video(&m)?;
            video_dimensions(v.width.unwrap_or(0), v.height.unwrap_or(0), c)?;
            ensure!(
                f64::from(*start_ms + *duration_ms) / 1000.0 <= m.duration_seconds.unwrap() + 0.01,
                "trim beyond media duration"
            );
            expected_duration = Some(f64::from(*duration_ms) / 1000.0);
            input_args(&mut args, &p, input_format(input)?);
            args.extend(vec![
                "-ss".into(),
                format!("{:.3}", f64::from(*start_ms) / 1000.0),
                "-t".into(),
                format!("{:.3}", f64::from(*duration_ms) / 1000.0),
                "-map".into(),
                "0:v:0".into(),
            ]);
            video_encoding(&mut args);
        }
        VideoOperation::Transcode {
            input,
            width,
            height,
        } => {
            let (p, m) = stage_video(policy, c, dir, input, 0, ctx).await?;
            require_video(&m)?;
            expected_duration = m.duration_seconds;
            input_args(&mut args, &p, input_format(input)?);
            args.extend(vec![
                "-map".into(),
                "0:v:0".into(),
                "-vf".into(),
                scale(*width, *height),
            ]);
            video_encoding(&mut args);
        }
        VideoOperation::Concat {
            inputs,
            width,
            height,
        } => {
            let mut graph = String::new();
            let mut total = 0.0;
            for (i, input) in inputs.iter().enumerate() {
                let (p, m) = stage_video(policy, c, dir, input, i, ctx).await?;
                require_video(&m)?;
                total += m.duration_seconds.unwrap();
                input_args(&mut args, &p, input_format(input)?);
                graph.push_str(&format!("[{i}:v:0]{}[v{i}];", scale(*width, *height)));
            }
            ensure!(
                total <= f64::from(c.max_duration_seconds),
                "concatenated duration exceeds limit"
            );
            expected_duration = Some(total);
            for i in 0..inputs.len() {
                graph.push_str(&format!("[v{i}]"));
            }
            graph.push_str(&format!("concat=n={}:v=1:a=0[out]", inputs.len()));
            args.extend(vec![
                "-filter_complex".into(),
                graph,
                "-map".into(),
                "[out]".into(),
            ]);
            video_encoding(&mut args);
        }
        VideoOperation::Overlay { input, image, x, y } => {
            let (p, m) = stage_video(policy, c, dir, input, 0, ctx).await?;
            let v = require_video(&m)?;
            expected_duration = m.duration_seconds;
            let (w, h) = (v.width.unwrap_or(0), v.height.unwrap_or(0));
            video_dimensions(w, h, c)?;
            let bytes = image_ops::normalized_png(policy, image, c)?;
            let decoded = image_ops::decode(&bytes, c)?;
            ensure!(
                x.checked_add(decoded.width()).is_some_and(|n| n <= w)
                    && y.checked_add(decoded.height()).is_some_and(|n| n <= h),
                "overlay outside video"
            );
            let img = dir.join("overlay.png");
            std::fs::write(&img, bytes)?;
            input_args(&mut args, &p, input_format(input)?);
            input_args(&mut args, &img, "png_pipe");
            args.extend(vec![
                "-filter_complex".into(),
                format!("[0:v:0][1:v:0]overlay=x={x}:y={y}:eof_action=repeat:shortest=0[out]"),
                "-map".into(),
                "[out]".into(),
            ]);
            video_encoding(&mut args);
        }
        VideoOperation::ExtractFrame { input, at_ms } => {
            let (p, m) = stage_video(policy, c, dir, input, 0, ctx).await?;
            require_video(&m)?;
            ensure!(
                f64::from(*at_ms) / 1000.0 < m.duration_seconds.unwrap(),
                "frame timestamp beyond media duration"
            );
            expected_duration = None;
            input_args(&mut args, &p, input_format(input)?);
            args.extend(vec![
                "-ss".into(),
                format!("{:.3}", f64::from(*at_ms) / 1000.0),
            ]);
            args.extend(strings(&[
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-an",
                "-sn",
                "-dn",
                "-c:v",
                "png",
                "-threads",
                "1",
                "-f",
                "image2pipe",
            ]));
            output_format = "png_pipe";
            expected_codec = "png";
        }
        VideoOperation::ExtractAudio { input } => {
            let (p, m) = stage_video(policy, c, dir, input, 0, ctx).await?;
            ensure!(
                m.streams.iter().any(|s| s.codec_type == "audio"),
                "audio stream required"
            );
            expected_duration = m.duration_seconds;
            input_args(&mut args, &p, input_format(input)?);
            args.extend(strings(&[
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
            ]));
            output_format = "wav";
            expected_codec = "pcm_s16le";
        }
        VideoOperation::Storyboard {
            images,
            seconds_per_image,
            width,
            height,
        } => {
            expected_duration = Some(images.len() as f64 * f64::from(*seconds_per_image));
            let mut graph = String::new();
            for (i, image) in images.iter().enumerate() {
                let p = dir.join(format!("image-{i}.png"));
                std::fs::write(&p, image_ops::normalized_png(policy, image, c)?)?;
                // Loop is generated filter logic over a normalized single image,
                // not image2 patterns, filesystem globbing or a user expression.
                input_args(&mut args, &p, "png_pipe");
                graph.push_str(&format!("[{i}:v:0]loop=loop=-1:size=1:start=0,trim=duration={seconds_per_image},{}[v{i}];",scale(*width,*height)));
            }
            for i in 0..images.len() {
                graph.push_str(&format!("[v{i}]"));
            }
            graph.push_str(&format!("concat=n={}:v=1:a=0[out]", images.len()));
            args.extend(vec![
                "-filter_complex".into(),
                graph,
                "-map".into(),
                "[out]".into(),
            ]);
            video_encoding(&mut args);
        }
    }
    args.extend(vec![
        "-t".into(),
        expected_duration
            .unwrap_or(f64::from(c.max_duration_seconds))
            .to_string(),
        "-fs".into(),
        c.max_output_bytes.to_string(),
        "-map_metadata".into(),
        "-1".into(),
        "-map_chapters".into(),
        "-1".into(),
        output.to_string_lossy().into_owned(),
    ]);
    process::run(c, &c.ffmpeg, args, dir, ctx).await?;
    let mut bytes = Vec::new();
    std::fs::File::open(&output)?
        .take(c.max_output_bytes as u64 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        !bytes.is_empty() && bytes.len() <= c.max_output_bytes,
        "output file outside byte bounds"
    );
    let media = probe(c, dir, &output, output_format, ctx).await?;
    ensure!(
        media.streams.len() == 1 && media.streams[0].codec_name == expected_codec,
        "unexpected output codec or stream count"
    );
    if output_format != "png_pipe" {
        ensure!(media.duration_seconds.is_some(), "output duration missing");
    }
    if let Some(expected) = expected_duration {
        ensure!(
            media
                .duration_seconds
                .is_some_and(|actual| (actual - expected).abs() <= 0.2),
            "output duration differs from requested duration; possible size-limit truncation"
        );
    }
    // Full bounded decode catches malformed/truncated outputs, not just a valid
    // container header. Discard muxer writes to stdout, never speakers/devices.
    let mut check = common();
    input_args(&mut check, &output, output_format);
    check.extend(strings(&[
        "-map", "0:0", "-threads", "1", "-f", "null", "pipe:1",
    ]));
    process::run(c, &c.ffmpeg, check, dir, ctx).await?;
    ensure!(
        !ctx.cancellation.is_cancelled(),
        "media cancelled before publication"
    );
    publish(policy, r.output, &bytes, media)
}
pub(crate) fn schema() -> Value {
    object_schema(
        json!({"output":path_schema(),"operation":{"type":"string","enum":["trim","transcode","concat","overlay","extract_frame","extract_audio","storyboard"]},"input":path_schema(),"inputs":{"type":"array","items":path_schema(),"minItems":2,"maxItems":8},"image":path_schema(),"images":{"type":"array","items":path_schema(),"minItems":1,"maxItems":8},"start_ms":{"type":"integer","minimum":0,"maximum":600000},"duration_ms":{"type":"integer","minimum":1,"maximum":600000},"at_ms":{"type":"integer","minimum":0,"maximum":600000},"width":{"type":"integer","minimum":2,"maximum":1920},"height":{"type":"integer","minimum":2,"maximum":1920},"x":{"type":"integer","minimum":0,"maximum":8192},"y":{"type":"integer","minimum":0,"maximum":8192},"seconds_per_image":{"type":"integer","minimum":1,"maximum":30}}),
        &["output", "operation"],
    )
}
