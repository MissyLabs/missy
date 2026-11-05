//! Explicit native OCR composition. No ambient capture/executable discovery.
use anyhow::Result;
use async_trait::async_trait;
use missy_core::{PolicyConfig, ToolContext, ToolExecutor};
use missy_desktop::ScreenOcr;
use missy_specialty::{SpecialtyConfig, SpecialtyTools};
use std::sync::Arc;
struct NativeScreenOcr(SpecialtyTools);
#[async_trait]
impl ScreenOcr for NativeScreenOcr {
    async fn recognize_tsv(&self, ctx: &ToolContext, png: Vec<u8>) -> Result<String> {
        let tsv = self.0.ocr_png(ctx, &png).await?;
        Ok(redact_tsv(tsv))
    }
    fn configuration_fingerprint(&self) -> String {
        format!(
            "native-screen-ocr-tsv-v1:{}",
            self.0.configuration_fingerprint()
        )
    }
}
fn redact_tsv(tsv: String) -> String {
    // Joined words catch credential assignments spanning adjacent OCR rows.
    // Conservatively withhold all labels if any secret shape is found.
    let text = tsv
        .lines()
        .skip(1)
        .filter_map(|l| l.splitn(12, '\t').nth(11))
        .collect::<Vec<_>>()
        .join(" ");
    if missy_security::redact_output(&text) == text {
        return tsv;
    }
    let mut safe = String::new();
    for (index, line) in tsv.lines().enumerate() {
        if index == 0 {
            safe.push_str(line);
        } else if let Some((prefix, _)) = line.rsplit_once('\t') {
            safe.push_str(prefix);
            safe.push_str("\t[REDACTED]");
        }
        safe.push('\n');
    }
    safe
}
/// No adapter unless document OCR and pinned Tesseract explicitly configured.
pub fn build(policy: PolicyConfig, config: SpecialtyConfig) -> Result<Option<Arc<dyn ScreenOcr>>> {
    let tools = SpecialtyTools::new(policy, config)?;
    Ok(tools
        .screen_ocr_configured()
        .then(|| Arc::new(NativeScreenOcr(tools)) as Arc<dyn ScreenOcr>))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn joined_credential_assignment_is_redacted_without_coordinate_changes() {
        let header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n";
        let raw = format!(
            "{header}5\t1\t1\t1\t1\t1\t2\t3\t10\t8\t95\tpassword:\n5\t1\t1\t1\t1\t2\t20\t3\t10\t8\t95\tsynthetic-secret\n"
        );
        let safe = redact_tsv(raw);
        assert!(!safe.contains("synthetic-secret"));
        let parsed = missy_desktop::parse_tesseract_tsv(&safe, 100, 50).unwrap();
        assert_eq!(parsed.anchors[0].text, "[REDACTED]");
        assert_eq!((parsed.anchors[1].left, parsed.anchors[1].top), (20, 3));
        let ordinary = format!("{header}5\t1\t1\t1\t1\t1\t2\t3\t10\t8\t95\tOpen\n");
        assert_eq!(redact_tsv(ordinary.clone()), ordinary);
    }
}
