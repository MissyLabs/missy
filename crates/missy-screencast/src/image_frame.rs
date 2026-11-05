use crate::ScreencastConfig;
use anyhow::{Result, bail};
use image::{ImageFormat, ImageReader, Limits};
use std::io::Cursor;

pub(crate) struct CleanFrame {
    pub png: Vec<u8>,
    pub sample: Vec<u8>,
    pub width: u32,
    pub height: u32,
}
pub(crate) fn sanitize(bytes: Vec<u8>, config: &ScreencastConfig) -> Result<CleanFrame> {
    if bytes.len() > config.max_frame_bytes {
        bail!("frame too large");
    }
    let format = image::guess_format(&bytes)?;
    if !matches!(format, ImageFormat::Png | ImageFormat::Jpeg) {
        bail!("only PNG/JPEG frames are permitted");
    }
    let (width, height) =
        ImageReader::with_format(Cursor::new(&bytes), format).into_dimensions()?;
    if width == 0
        || height == 0
        || width > config.max_dimension
        || height > config.max_dimension
        || u64::from(width) * u64::from(height) > config.max_pixels
    {
        bail!("frame dimensions exceed policy");
    }
    let mut reader = ImageReader::with_format(Cursor::new(bytes), format);
    let mut limits = Limits::default();
    limits.max_image_width = Some(config.max_dimension);
    limits.max_image_height = Some(config.max_dimension);
    limits.max_alloc = Some(config.max_pixels.saturating_mul(16));
    reader.limits(limits);
    // Decode, strip all source metadata, normalize alpha and encode a fresh PNG.
    let image = image::DynamicImage::ImageRgb8(reader.decode()?.to_rgb8());
    let sample = image
        .resize_exact(32, 32, image::imageops::FilterType::Triangle)
        .to_luma8()
        .into_raw();
    let mut out = Cursor::new(Vec::new());
    image.write_to(&mut out, ImageFormat::Png)?;
    let png = out.into_inner();
    if png.len() > config.max_frame_bytes {
        bail!("sanitized PNG exceeds frame policy");
    }
    Ok(CleanFrame {
        png,
        sample,
        width,
        height,
    })
}
pub(crate) fn changed(old: &[u8], new: &[u8], threshold: u8) -> bool {
    if old.len() != new.len() {
        return true;
    }
    let total: u64 = old
        .iter()
        .zip(new)
        .map(|(a, b)| u64::from(a.abs_diff(*b)))
        .sum();
    total != 0 && total >= u64::from(threshold) * old.len() as u64
}
