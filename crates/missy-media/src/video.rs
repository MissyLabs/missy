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
            Some("audio_mux") => &[
                "output",
                "operation",
                "input",
                "audio_file",
                "gain",
                "duration_ms",
                "loop",
            ],
            Some("audio_mix") => &[
                "output",
                "operation",
                "input",
                "audio_file",
                "gain",
                "original_gain",
                "duration_ms",
                "loop",
            ],
            Some("speed") => &["output", "operation", "input", "factor", "preserve_audio"],
            Some("crossfade") => &[
                "output",
                "operation",
                "inputs",
                "width",
                "height",
                "duration_ms",
                "preserve_audio",
            ],
            Some("text") => &[
                "output",
                "operation",
                "input",
                "text",
                "font_file",
                "font_size",
                "position",
                "font_color",
                "box_color",
                "box_opacity",
                "box",
                "start_ms",
                "end_ms",
                "preserve_audio",
            ],
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
    AudioMux {
        input: PathBuf,
        audio_file: PathBuf,
        gain: f64,
        duration_ms: Option<u32>,
        #[serde(rename = "loop")]
        loop_audio: bool,
    },
    AudioMix {
        input: PathBuf,
        audio_file: PathBuf,
        gain: f64,
        original_gain: f64,
        duration_ms: Option<u32>,
        #[serde(rename = "loop")]
        loop_audio: bool,
    },
    Speed {
        input: PathBuf,
        factor: f64,
        preserve_audio: bool,
    },
    Text {
        input: PathBuf,
        text: String,
        font_file: PathBuf,
        font_size: u32,
        position: TextPosition,
        font_color: CaptionColor,
        box_color: CaptionColor,
        box_opacity: f64,
        #[serde(rename = "box")]
        boxed: bool,
        start_ms: u32,
        end_ms: u32,
        preserve_audio: bool,
    },
    Crossfade {
        inputs: Vec<PathBuf>,
        width: u32,
        height: u32,
        duration_ms: u32,
        preserve_audio: bool,
    },
    Storyboard {
        images: Vec<PathBuf>,
        seconds_per_image: u32,
        width: u32,
        height: u32,
    },
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TextPosition {
    TopLeft,
    Top,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    Bottom,
    BottomRight,
}
impl TextPosition {
    fn coordinates(&self) -> (&'static str, &'static str) {
        match self {
            Self::TopLeft => ("4", "4"),
            Self::Top => ("(w-text_w)/2", "4"),
            Self::TopRight => ("w-text_w-4", "4"),
            Self::CenterLeft => ("4", "(h-text_h)/2"),
            Self::Center => ("(w-text_w)/2", "(h-text_h)/2"),
            Self::CenterRight => ("w-text_w-4", "(h-text_h)/2"),
            Self::BottomLeft => ("4", "h-text_h-4"),
            Self::Bottom => ("(w-text_w)/2", "h-text_h-4"),
            Self::BottomRight => ("w-text_w-4", "h-text_h-4"),
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptionColor {
    White,
    Black,
    Red,
    Blue,
    Yellow,
}
impl CaptionColor {
    fn filter_value(&self) -> &'static str {
        match self {
            Self::White => "white",
            Self::Black => "black",
            Self::Red => "red",
            Self::Blue => "blue",
            Self::Yellow => "yellow",
        }
    }
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
            VideoOperation::AudioMux {
                input,
                audio_file,
                gain,
                duration_ms,
                ..
            }
            | VideoOperation::AudioMix {
                input,
                audio_file,
                gain,
                duration_ms,
                ..
            } => {
                input_format(input)?;
                input_format(audio_file)?;
                ensure!(
                    gain.is_finite() && (0.0..=4.0).contains(gain),
                    "gain must be finite 0..4"
                );
                ensure!(
                    duration_ms.is_none_or(|d| d > 0 && d <= c.max_duration_seconds * 1000),
                    "audio duration outside bounds"
                );
                if let VideoOperation::AudioMix { original_gain, .. } = &self.operation {
                    ensure!(
                        original_gain.is_finite() && (0.0..=4.0).contains(original_gain),
                        "original gain must be finite 0..4"
                    );
                }
            }
            VideoOperation::Speed { input, factor, .. } => {
                input_format(input)?;
                ensure!(
                    factor.is_finite() && (0.25..=4.0).contains(factor),
                    "speed factor must be finite 0.25..4"
                );
            }
            VideoOperation::Text {
                input,
                text,
                font_file,
                font_size,
                box_opacity,
                start_ms,
                end_ms,
                ..
            } => {
                input_format(input)?;
                ensure!(
                    matches!(extension(font_file)?, "ttf" | "otf"),
                    "explicit TTF/OTF required"
                );
                ensure!(
                    !text.is_empty() && text.len() <= 4096 && !text.contains('\0'),
                    "caption text outside bounds"
                );
                ensure!((8..=256).contains(font_size), "font size outside 8..256");
                ensure!(
                    box_opacity.is_finite() && (0.0..=1.0).contains(box_opacity),
                    "box opacity outside 0..1"
                );
                ensure!(
                    start_ms < end_ms && *end_ms <= c.max_duration_seconds * 1000,
                    "caption time range outside bounds"
                );
            }
            VideoOperation::Crossfade {
                inputs,
                width,
                height,
                duration_ms,
                ..
            } => {
                ensure!(inputs.len() == 2, "crossfade requires exactly two clips");
                for input in inputs {
                    input_format(input)?;
                }
                video_dimensions(*width, *height, c)?;
                ensure!(
                    *duration_ms > 0
                        && *duration_ms <= 5000
                        && *duration_ms <= c.max_duration_seconds * 1000,
                    "crossfade duration outside bounds"
                );
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
pub(crate) fn common() -> Vec<String> {
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
pub(crate) fn input_args(args: &mut Vec<String>, path: &Path, format: &str) {
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
pub(crate) async fn probe(
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
    video_encoding_with_audio(args, false);
}
fn video_encoding_with_audio(args: &mut Vec<String>, audio: bool) {
    args.extend(strings(&[
        "-sn", "-dn", "-c:v", "mpeg4", "-q:v", "4", "-pix_fmt", "yuv420p", "-threads", "1", "-r",
        "15", "-f", "mp4",
    ]));
    if audio {
        args.extend(strings(&[
            "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-ar", "48000",
        ]));
    } else {
        args.push("-an".into());
    }
}
fn has_audio(media: &MediaMetadata) -> bool {
    media.streams.iter().any(|s| s.codec_type == "audio")
}
fn atempo(mut factor: f64) -> String {
    let mut stages = Vec::new();
    while factor > 2.0 {
        stages.push("atempo=2".to_owned());
        factor /= 2.0;
    }
    while factor < 0.5 {
        stages.push("atempo=0.5".to_owned());
        factor /= 0.5;
    }
    stages.push(format!("atempo={factor:.9}"));
    stages.join(",")
}
fn filter_path(path: &Path) -> Result<String> {
    let value = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("UTF-8 staging path required"))?;
    // Only private generated filenames are used. Reject unusual operator staging
    // roots rather than attempt ambiguous multi-layer FFmpeg graph escaping.
    ensure!(
        value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"/._-".contains(&b)),
        "caption staging path contains filter metacharacters"
    );
    Ok(value.to_owned())
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
    let mut expected_audio = false;
    let mut copied_video_codec = None;
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
        VideoOperation::AudioMux {
            input,
            audio_file,
            gain,
            duration_ms,
            loop_audio,
        }
        | VideoOperation::AudioMix {
            input,
            audio_file,
            gain,
            duration_ms,
            loop_audio,
            ..
        } => {
            let (p, m) = stage_video(policy, c, dir, input, 0, ctx).await?;
            let video = require_video(&m)?;
            video_dimensions(video.width.unwrap_or(0), video.height.unwrap_or(0), c)?;
            copied_video_codec = Some(video.codec_name.clone());
            let (a, am) = stage_video(policy, c, dir, audio_file, 1, ctx).await?;
            ensure!(has_audio(&am), "audio_file needs an audio stream");
            let duration = duration_ms
                .map(|d| f64::from(d) / 1000.0)
                .unwrap_or(m.duration_seconds.unwrap());
            ensure!(
                duration <= m.duration_seconds.unwrap() + 0.001,
                "audio duration exceeds video"
            );
            expected_duration = Some(duration);
            expected_audio = true;
            input_args(&mut args, &p, input_format(input)?);
            if *loop_audio {
                args.extend(strings(&["-stream_loop", "-1"]));
            }
            input_args(&mut args, &a, input_format(audio_file)?);
            let incoming = format!(
                "[1:a:0]asetpts=PTS-STARTPTS,volume={gain:.9},apad,atrim=duration={duration:.9}[incoming]"
            );
            let graph = if let VideoOperation::AudioMix { original_gain, .. } = &r.operation {
                if has_audio(&m) {
                    format!(
                        "{incoming};[0:a:0]asetpts=PTS-STARTPTS,volume={original_gain:.9},apad,atrim=duration={duration:.9}[original];[original][incoming]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
                    )
                } else {
                    incoming.replace("[incoming]", "[aout]")
                }
            } else {
                incoming.replace("[incoming]", "[aout]")
            };
            args.extend(vec![
                "-filter_complex".into(),
                graph,
                "-map".into(),
                "0:v:0".into(),
                "-map".into(),
                "[aout]".into(),
            ]);
            args.extend(strings(&[
                "-sn", "-dn", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-ar",
                "48000", "-f", "mp4",
            ]));
        }
        VideoOperation::Speed {
            input,
            factor,
            preserve_audio,
        } => {
            let (p, m) = stage_video(policy, c, dir, input, 0, ctx).await?;
            let video = require_video(&m)?;
            video_dimensions(video.width.unwrap_or(0), video.height.unwrap_or(0), c)?;
            let duration = m.duration_seconds.unwrap() / factor;
            ensure!(
                duration <= f64::from(c.max_duration_seconds),
                "slowed duration exceeds limit"
            );
            expected_duration = Some(duration);
            expected_audio = *preserve_audio && has_audio(&m);
            input_args(&mut args, &p, input_format(input)?);
            let mut graph = format!("[0:v:0]setpts=(PTS-STARTPTS)/{factor:.9}[vout]");
            if expected_audio {
                graph.push_str(&format!(
                    ";[0:a:0]asetpts=PTS-STARTPTS,{},apad,atrim=duration={duration:.9}[aout]",
                    atempo(*factor)
                ));
            }
            args.extend(vec![
                "-filter_complex".into(),
                graph,
                "-map".into(),
                "[vout]".into(),
            ]);
            if expected_audio {
                args.extend(strings(&["-map", "[aout]"]));
            }
            video_encoding_with_audio(&mut args, expected_audio);
        }
        VideoOperation::Text {
            input,
            text,
            font_file,
            font_size,
            position,
            font_color,
            box_color,
            box_opacity,
            boxed,
            start_ms,
            end_ms,
            preserve_audio,
        } => {
            let (p, m) = stage_video(policy, c, dir, input, 0, ctx).await?;
            let video = require_video(&m)?;
            video_dimensions(video.width.unwrap_or(0), video.height.unwrap_or(0), c)?;
            ensure!(
                f64::from(*end_ms) / 1000.0 <= m.duration_seconds.unwrap() + 0.001,
                "caption beyond video duration"
            );
            let font = policy.read_file(font_file)?;
            ensure!(
                font.len() >= 12
                    && font.len() <= 8 * 1024 * 1024
                    && matches!(&font[..4], b"\x00\x01\x00\x00" | b"OTTO" | b"true"),
                "bounded TrueType/OpenType font required"
            );
            let font_path = dir.join("caption-font.ttf");
            let text_path = dir.join("caption.txt");
            std::fs::write(&font_path, font)?;
            std::fs::write(&text_path, text.as_bytes())?;
            let (x, y) = position.coordinates();
            let vf = format!(
                "drawtext=fontfile={}:textfile={}:expansion=none:fontsize={font_size}:fontcolor={}:x={x}:y={y}:box={}:boxcolor={}@{box_opacity:.9}:boxborderw=2:enable=between(t\\,{:.3}\\,{:.3})",
                filter_path(&font_path)?,
                filter_path(&text_path)?,
                font_color.filter_value(),
                u8::from(*boxed),
                box_color.filter_value(),
                f64::from(*start_ms) / 1000.0,
                f64::from(*end_ms) / 1000.0
            );
            expected_duration = m.duration_seconds;
            expected_audio = *preserve_audio && has_audio(&m);
            input_args(&mut args, &p, input_format(input)?);
            args.extend(vec!["-map".into(), "0:v:0".into(), "-vf".into(), vf]);
            if expected_audio {
                args.extend(strings(&["-map", "0:a:0"]));
            }
            video_encoding_with_audio(&mut args, expected_audio);
        }
        VideoOperation::Crossfade {
            inputs,
            width,
            height,
            duration_ms,
            preserve_audio,
        } => {
            let mut media = Vec::new();
            let mut graph = String::new();
            let fade = f64::from(*duration_ms) / 1000.0;
            for (i, input) in inputs.iter().enumerate() {
                let (p, m) = stage_video(policy, c, dir, input, i, ctx).await?;
                require_video(&m)?;
                ensure!(
                    fade <= m.duration_seconds.unwrap(),
                    "crossfade exceeds shorter clip"
                );
                input_args(&mut args, &p, input_format(input)?);
                graph.push_str(&format!(
                    "[{i}:v:0]{},settb=AVTB[v{i}];",
                    scale(*width, *height)
                ));
                media.push(m);
            }
            let first = media[0].duration_seconds.unwrap();
            let duration = first + media[1].duration_seconds.unwrap() - fade;
            ensure!(
                duration <= f64::from(c.max_duration_seconds),
                "crossfade output exceeds duration limit"
            );
            expected_duration = Some(duration);
            graph.push_str(&format!(
                "[v0][v1]xfade=transition=fade:duration={fade:.9}:offset={:.9}[vout]",
                first - fade
            ));
            expected_audio = *preserve_audio && media.iter().any(has_audio);
            if expected_audio {
                for (i, m) in media.iter().enumerate() {
                    let d = m.duration_seconds.unwrap();
                    if has_audio(m) {
                        graph.push_str(&format!(";[{i}:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,asetpts=PTS-STARTPTS,apad,atrim=duration={d:.9}[a{i}]"));
                    } else {
                        graph.push_str(&format!(
                            ";anullsrc=r=48000:cl=stereo,atrim=duration={d:.9}[a{i}]"
                        ));
                    }
                }
                graph.push_str(&format!(
                    ";[a0][a1]acrossfade=d={fade:.9}:c1=tri:c2=tri[aout]"
                ));
            }
            args.extend(vec![
                "-filter_complex".into(),
                graph,
                "-map".into(),
                "[vout]".into(),
            ]);
            if expected_audio {
                args.extend(strings(&["-map", "[aout]"]));
            }
            video_encoding_with_audio(&mut args, expected_audio);
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
        media.streams.len() == if expected_audio { 2 } else { 1 }
            && media.streams[0].codec_name
                == copied_video_codec.as_deref().unwrap_or(expected_codec)
            && (!expected_audio
                || (media.streams[1].codec_type == "audio"
                    && media.streams[1].codec_name == "aac")),
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
        "-map", "0", "-threads", "1", "-f", "null", "pipe:1",
    ]));
    process::run(c, &c.ffmpeg, check, dir, ctx).await?;
    ensure!(
        !ctx.cancellation.is_cancelled(),
        "media cancelled before publication"
    );
    publish(policy, r.output, &bytes, media)
}
pub(crate) fn schema() -> Value {
    let mut schema = object_schema(
        json!({"output":path_schema(),"operation":{"type":"string","enum":["trim","transcode","concat","overlay","extract_frame","extract_audio","storyboard"]},"input":path_schema(),"inputs":{"type":"array","items":path_schema(),"minItems":2,"maxItems":8},"image":path_schema(),"images":{"type":"array","items":path_schema(),"minItems":1,"maxItems":8},"start_ms":{"type":"integer","minimum":0,"maximum":600000},"duration_ms":{"type":"integer","minimum":1,"maximum":600000},"at_ms":{"type":"integer","minimum":0,"maximum":600000},"width":{"type":"integer","minimum":2,"maximum":1920},"height":{"type":"integer","minimum":2,"maximum":1920},"x":{"type":"integer","minimum":0,"maximum":8192},"y":{"type":"integer","minimum":0,"maximum":8192},"seconds_per_image":{"type":"integer","minimum":1,"maximum":30}}),
        &["output", "operation"],
    );
    schema["properties"]["operation"]["enum"]
        .as_array_mut()
        .unwrap()
        .extend(["audio_mux", "audio_mix", "speed", "text", "crossfade"].map(|s| json!(s)));
    let extra = json!({
        "audio_file":path_schema(), "gain":{"type":"number","minimum":0,"maximum":4},
        "original_gain":{"type":"number","minimum":0,"maximum":4}, "loop":{"type":"boolean"},
        "factor":{"type":"number","minimum":0.25,"maximum":4}, "preserve_audio":{"type":"boolean"},
        "text":{"type":"string","minLength":1,"maxLength":4096}, "font_file":path_schema(),
        "font_size":{"type":"integer","minimum":8,"maximum":256},
        "position":{"type":"string","enum":["top-left","top","top-right","center-left","center","center-right","bottom-left","bottom","bottom-right"]},
        "font_color":{"type":"string","enum":["white","black","red","blue","yellow"]},
        "box_color":{"type":"string","enum":["white","black","red","blue","yellow"]},
        "box_opacity":{"type":"number","minimum":0,"maximum":1}, "box":{"type":"boolean"},
        "end_ms":{"type":"integer","minimum":1,"maximum":600000}
    });
    schema["properties"]
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    // Audio duration can be omitted/null to use the video's bounded duration.
    schema["properties"]["duration_ms"]["type"] = json!(["integer", "null"]);
    schema
}
