//! Native-pixel checks and Tesseract TSV anchors. No ambient discovery.
use anyhow::{Result, bail, ensure};
use async_trait::async_trait;
use missy_core::ToolContext;
use serde::{Deserialize, Serialize};

/// Inject only an explicitly configured executable-pinned sandboxed backend.
/// The embedder owns policy, subprocess bounds, cancellation and censoring.
/// Use the original PNG without resizing; return actual word-level TSV.
#[async_trait]
pub trait ScreenOcr: Send + Sync {
    async fn recognize_tsv(&self, ctx: &ToolContext, png: Vec<u8>) -> Result<String>;
    fn configuration_fingerprint(&self) -> String;
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OcrAnchor {
    pub text: String,
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
    pub center_x: u32,
    pub center_y: u32,
    pub confidence: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OcrAvailability {
    NotConfigured,
    Available,
    Unavailable,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScreenOcrResult {
    pub availability: OcrAvailability,
    pub coordinate_space: String,
    pub width: u32,
    pub height: u32,
    pub anchors: Vec<OcrAnchor>,
    pub truncated: bool,
    /// Fixed bounded code; no arbitrary subprocess exception/stderr.
    pub diagnostic: Option<String>,
}
impl ScreenOcrResult {
    pub(crate) fn absent(width: u32, height: u32, unavailable: bool) -> Self {
        Self {
            availability: if unavailable {
                OcrAvailability::Unavailable
            } else {
                OcrAvailability::NotConfigured
            },
            coordinate_space: "native_window_pixels".into(),
            width,
            height,
            anchors: Vec::new(),
            truncated: false,
            diagnostic: unavailable.then(|| "ocr_backend_failed_or_invalid_tsv".into()),
        }
    }
}
/// Strict original-image Tesseract word TSV. Bounds independent of backend.
pub fn parse_tesseract_tsv(tsv: &str, width: u32, height: u32) -> Result<ScreenOcrResult> {
    ensure!(
        width > 0 && height > 0 && width as u64 * height as u64 <= 4_000_000,
        "invalid OCR dimensions"
    );
    ensure!(tsv.len() <= 1024 * 1024, "OCR TSV exceeds byte budget");
    let mut lines = tsv.lines();
    ensure!(
        lines.next()
            == Some(
                "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
            ),
        "invalid OCR TSV header"
    );
    let mut result = ScreenOcrResult::absent(width, height, false);
    result.availability = OcrAvailability::Available;
    for (index, line) in lines.enumerate() {
        ensure!(index < 4096, "OCR TSV exceeds row budget");
        let fields: Vec<_> = line.splitn(12, '\t').collect();
        ensure!(fields.len() == 12, "invalid OCR TSV row");
        let level: u8 = fields[0].parse()?;
        ensure!((1..=5).contains(&level), "invalid OCR TSV level");
        if level != 5 {
            continue;
        }
        let left: u32 = fields[6].parse()?;
        let top: u32 = fields[7].parse()?;
        let w: u32 = fields[8].parse()?;
        let h: u32 = fields[9].parse()?;
        let confidence: f32 = fields[10].parse()?;
        ensure!(
            confidence.is_finite() && (-1.0..=100.0).contains(&confidence),
            "invalid OCR confidence"
        );
        let right = left
            .checked_add(w)
            .ok_or_else(|| anyhow::anyhow!("OCR coordinate overflow"))?;
        let bottom = top
            .checked_add(h)
            .ok_or_else(|| anyhow::anyhow!("OCR coordinate overflow"))?;
        ensure!(
            right <= width && bottom <= height,
            "OCR box outside captured window"
        );
        let text = fields[11].trim();
        ensure!(!text.chars().any(char::is_control), "invalid OCR text");
        if confidence < 40.0 || w == 0 || h == 0 || text.is_empty() {
            continue;
        }
        if result.anchors.len() == 128 {
            result.truncated = true;
            continue;
        }
        result.anchors.push(OcrAnchor {
            text: text.chars().take(128).collect(),
            left,
            top,
            right,
            bottom,
            center_x: left + w / 2,
            center_y: top + h / 2,
            confidence,
        });
    }
    Ok(result)
}
/// Every RGB channel stddev below 1.5 is near-uniform (legacy threshold).
pub(crate) fn reject_blank_rgb(pixels: &[u8]) -> Result<()> {
    ensure!(
        !pixels.is_empty() && pixels.len().is_multiple_of(3),
        "invalid RGB frame"
    );
    let count = (pixels.len() / 3) as f64;
    let mut sums = [0f64; 3];
    let mut squares = [0f64; 3];
    for pixel in pixels.as_chunks::<3>().0 {
        for i in 0..3 {
            let v = pixel[i] as f64;
            sums[i] += v;
            squares[i] += v * v;
        }
    }
    if (0..3).all(|i| (squares[i] / count - (sums[i] / count).powi(2)).max(0.0) < 2.25) {
        bail!(
            "blank_capture: uniform or near-uniform frame; no OCR or vision invoked; screen content is unknown"
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    const HEADER: &str = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n";
    #[test]
    fn synthetic_matrices() {
        for color in [[0, 0, 0], [255, 255, 255], [51, 102, 153]] {
            assert!(
                reject_blank_rgb(&color.repeat(32 * 24))
                    .unwrap_err()
                    .to_string()
                    .contains("blank_capture")
            );
        }
        assert!(reject_blank_rgb(&[0, 0, 0, 1, 1, 1].repeat(64)).is_err());
        assert!(reject_blank_rgb(&[]).is_err());
        assert!(reject_blank_rgb(&[0, 1]).is_err());
        assert!(reject_blank_rgb(&[0, 0, 0, 255, 0, 0].repeat(64)).is_ok());
    }
    #[test]
    fn literal_bounded_tsv() {
        let row = "5\t1\t1\t1\t1\t1\t12\t8\t40\t20\t96.5\tOpen\n";
        let r = parse_tesseract_tsv(&format!("{HEADER}{row}"), 100, 50).unwrap();
        let a = &r.anchors[0];
        assert_eq!(
            (
                &*a.text, a.left, a.top, a.right, a.bottom, a.center_x, a.center_y
            ),
            ("Open", 12, 8, 52, 28, 32, 18)
        );
        assert_eq!(
            parse_tesseract_tsv(HEADER, 100, 50).unwrap().availability,
            OcrAvailability::Available
        );
        for row in [
            "5\t1\t1\t1\t1\t1\t99\t8\t40\t20\t96\tOpen",
            "5\t1\t1\t1\t1\t1\t-1\t8\t40\t20\t96\tOpen",
            "5\t1\t1\t1\t1\t1\t12\t8\t40\t20\tNaN\tOpen",
            "junk",
        ] {
            assert!(parse_tesseract_tsv(&format!("{HEADER}{row}\n"), 100, 50).is_err());
        }
        let r = parse_tesseract_tsv(&format!("{HEADER}{}", row.repeat(129)), 100, 50).unwrap();
        assert_eq!(r.anchors.len(), 128);
        assert!(r.truncated);
        assert!(parse_tesseract_tsv(&"x".repeat(1024 * 1024 + 1), 100, 50).is_err());
    }
}
