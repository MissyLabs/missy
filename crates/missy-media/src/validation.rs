//! Full decoding for externally generated media, without publishing or playing it.
use crate::{MediaConfig, MediaMetadata, process, video};
use anyhow::{Result, ensure};
use missy_core::ToolContext;
pub(crate) async fn validate_mp4(
    c: &MediaConfig,
    ctx: &ToolContext,
    bytes: &[u8],
) -> Result<MediaMetadata> {
    ensure!(
        !ctx.cancellation.is_cancelled(),
        "media validation cancelled"
    );
    ensure!(
        !bytes.is_empty() && bytes.len() <= c.max_output_bytes,
        "media validation byte bound exceeded"
    );
    ensure!(bytes.get(4..8) == Some(b"ftyp"), "MP4 header required");
    let staging = tempfile::Builder::new()
        .prefix("missy-validate-")
        .tempdir_in(&c.staging_root)?;
    let path = staging.path().join("input.mp4");
    std::fs::write(&path, bytes)?;
    let metadata = video::probe(c, staging.path(), &path, "mov", ctx).await?;
    ensure!(
        metadata.duration_seconds.is_some(),
        "finite media duration required"
    );
    ensure!(
        metadata.streams.iter().any(|s| s.codec_type == "video"),
        "video stream required"
    );
    let mut args = video::common();
    video::input_args(&mut args, &path, "mov");
    args.extend(
        ["-map", "0", "-threads", "1", "-f", "null", "pipe:1"]
            .iter()
            .map(|s| s.to_string()),
    );
    process::run(c, &c.ffmpeg, args, staging.path(), ctx).await?;
    Ok(metadata)
}
