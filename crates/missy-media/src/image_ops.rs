use crate::{
    MediaConfig, MediaMetadata, StreamMetadata, extension, object_schema, path_schema, publish,
};
use anyhow::{Result, ensure};
use image::{DynamicImage, ImageFormat, ImageReader, Limits};
use missy_policy::Policy;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{io::Cursor, path::PathBuf};

#[derive(Clone, Debug, Serialize)]
pub struct ImageRequest {
    pub input: PathBuf,
    pub output: PathBuf,
    #[serde(flatten)]
    pub operation: ImageOperation,
}
impl<'de> Deserialize<'de> for ImageRequest {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let mut value = Value::deserialize(deserializer)?;
        let map = value
            .as_object_mut()
            .ok_or_else(|| serde::de::Error::custom("object required"))?;
        let allowed: &[&str] = match map.get("operation").and_then(Value::as_str) {
            Some("copy") => &["input", "output", "operation"],
            Some("resize") => &["input", "output", "operation", "width", "height"],
            Some("crop") => &["input", "output", "operation", "x", "y", "width", "height"],
            Some("rotate") => &["input", "output", "operation", "degrees"],
            Some("overlay") => &["input", "output", "operation", "overlay", "x", "y"],
            _ => return Err(serde::de::Error::custom("unknown image operation")),
        };
        if map.keys().any(|k| !allowed.contains(&k.as_str())) {
            return Err(serde::de::Error::custom("unknown image operation field"));
        }
        let input = serde_json::from_value(map.remove("input").unwrap_or(Value::Null))
            .map_err(serde::de::Error::custom)?;
        let output = serde_json::from_value(map.remove("output").unwrap_or(Value::Null))
            .map_err(serde::de::Error::custom)?;
        let operation = serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        Ok(Self {
            input,
            output,
            operation,
        })
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum ImageOperation {
    Copy,
    Resize {
        width: u32,
        height: u32,
    },
    Crop {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    Rotate {
        degrees: Rotation,
    },
    Overlay {
        overlay: PathBuf,
        x: u32,
        y: u32,
    },
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum Rotation {
    #[serde(rename = "90")]
    R90,
    #[serde(rename = "180")]
    R180,
    #[serde(rename = "270")]
    R270,
}
pub(crate) fn dimensions(w: u32, h: u32, config: &MediaConfig) -> Result<()> {
    ensure!(
        w > 0
            && h > 0
            && w <= 8192
            && h <= 8192
            && u64::from(w) * u64::from(h) <= u64::from(config.max_pixels),
        "pixel/dimension limit exceeded"
    );
    Ok(())
}
fn format(path: &std::path::Path) -> Result<ImageFormat> {
    match extension(path)? {
        "png" => Ok(ImageFormat::Png),
        "jpg" | "jpeg" => Ok(ImageFormat::Jpeg),
        _ => anyhow::bail!("only PNG/JPEG allowed"),
    }
}
impl ImageRequest {
    pub fn validate(&self, config: &MediaConfig) -> Result<()> {
        format(&self.input)?;
        format(&self.output)?;
        match &self.operation {
            ImageOperation::Resize { width, height }
            | ImageOperation::Crop { width, height, .. } => dimensions(*width, *height, config)?,
            ImageOperation::Overlay { overlay, .. } => {
                format(overlay)?;
            }
            _ => (),
        }
        Ok(())
    }
}
pub(crate) fn decode(bytes: &[u8], config: &MediaConfig) -> Result<DynamicImage> {
    let detected = image::guess_format(bytes)?;
    ensure!(
        matches!(detected, ImageFormat::Png | ImageFormat::Jpeg),
        "PNG/JPEG bytes required"
    );
    let (w, h) = ImageReader::with_format(Cursor::new(bytes), detected).into_dimensions()?;
    dimensions(w, h, config)?;
    let mut reader = ImageReader::with_format(Cursor::new(bytes), detected);
    let mut limits = Limits::default();
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    limits.max_alloc = Some(u64::from(config.max_pixels) * 8);
    reader.limits(limits);
    Ok(reader.decode()?)
}
pub(crate) fn normalized_png(
    policy: &Policy,
    input: &std::path::Path,
    config: &MediaConfig,
) -> Result<Vec<u8>> {
    format(input)?;
    let image = decode(&policy.read_file(input)?, config)?;
    let mut out = Cursor::new(Vec::new());
    image.write_to(&mut out, ImageFormat::Png)?;
    ensure!(
        out.get_ref().len() <= config.max_output_bytes,
        "normalized image exceeds limit"
    );
    Ok(out.into_inner())
}
pub(crate) fn execute(policy: &Policy, config: &MediaConfig, r: ImageRequest) -> Result<Value> {
    r.validate(config)?;
    let mut image = decode(&policy.read_file(&r.input)?, config)?;
    image = match r.operation {
        ImageOperation::Copy => image,
        ImageOperation::Resize { width, height } => {
            image.resize_exact(width, height, image::imageops::FilterType::Triangle)
        }
        ImageOperation::Crop {
            x,
            y,
            width,
            height,
        } => {
            ensure!(
                x.checked_add(width).is_some_and(|v| v <= image.width())
                    && y.checked_add(height).is_some_and(|v| v <= image.height()),
                "crop outside image"
            );
            image.crop_imm(x, y, width, height)
        }
        ImageOperation::Rotate { degrees } => match degrees {
            Rotation::R90 => image.rotate90(),
            Rotation::R180 => image.rotate180(),
            Rotation::R270 => image.rotate270(),
        },
        ImageOperation::Overlay { overlay, x, y } => {
            let overlay = decode(&policy.read_file(overlay)?, config)?;
            ensure!(
                x.checked_add(overlay.width())
                    .is_some_and(|v| v <= image.width())
                    && y.checked_add(overlay.height())
                        .is_some_and(|v| v <= image.height()),
                "overlay outside image"
            );
            image::imageops::overlay(&mut image, &overlay, i64::from(x), i64::from(y));
            image
        }
    };
    let out_format = format(&r.output)?;
    if out_format == ImageFormat::Jpeg {
        image = DynamicImage::ImageRgb8(image.to_rgb8());
    }
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, out_format)?;
    ensure!(
        bytes.get_ref().len() <= config.max_output_bytes,
        "image output exceeds limit"
    );
    let checked = decode(bytes.get_ref(), config)?;
    let codec = if out_format == ImageFormat::Png {
        "png"
    } else {
        "jpeg"
    };
    let media = MediaMetadata {
        duration_seconds: None,
        format: codec.into(),
        streams: vec![StreamMetadata {
            codec_type: "image".into(),
            codec_name: codec.into(),
            width: Some(checked.width()),
            height: Some(checked.height()),
            sample_rate: None,
            channels: None,
        }],
    };
    publish(policy, r.output, bytes.get_ref(), media)
}
pub(crate) fn schema() -> Value {
    object_schema(
        json!({"input":path_schema(),"output":path_schema(),"operation":{"type":"string","enum":["copy","resize","crop","rotate","overlay"]},"width":{"type":"integer","minimum":1,"maximum":8192},"height":{"type":"integer","minimum":1,"maximum":8192},"x":{"type":"integer","minimum":0},"y":{"type":"integer","minimum":0},"degrees":{"type":"string","enum":["90","180","270"]},"overlay":path_schema()}),
        &["input", "output", "operation"],
    )
}
