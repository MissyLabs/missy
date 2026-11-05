//! Bounded scheduled monitoring of an explicit file refreshed by a separately
//! approved producer. Never discovers or opens a camera/display.
use super::*;
use anyhow::ensure;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Monitor {
    imagepath: PathBuf,
    question: String,
    samples: u32,
    interval_ms: u64,
    cooldown_ms: u64,
    change_threshold: f64,
    #[serde(default)]
    analyze_first: bool,
}
pub(super) fn parse(tools: &VisionTools, call: &ToolCall) -> Result<Monitor> {
    ensure!(tools.config.enabled, "vision disabled");
    let r: Monitor = serde_json::from_value(call.arguments.clone())?;
    validate_paths(std::slice::from_ref(&r.imagepath))?;
    ensure!(
        !r.question.trim().is_empty() && r.question.len() <= tools.config.max_question_bytes,
        "invalid monitor question"
    );
    ensure!(
        (1..=60).contains(&r.samples) && (100..=60_000).contains(&r.interval_ms),
        "monitor bounds exceeded"
    );
    ensure!(
        (r.samples as u64 - 1) * r.interval_ms <= 300_000 && r.cooldown_ms <= 300_000,
        "monitor run/cooldown exceeds five minutes"
    );
    ensure!(
        r.change_threshold.is_finite() && (0.0..=1.0).contains(&r.change_threshold),
        "change threshold must be finite 0..1"
    );
    Ok(r)
}
pub(super) fn definition() -> ToolDefinition {
    ToolDefinition { name:"vision_monitor".into(), description:"Approval-gated bounded scheduled vision monitoring of an explicit image FILE refreshed by a separately authorized producer. Pixel-change threshold and cooldown suppress redundant billable analyses. No camera/display access; maximum five minutes. Use scheduler for recurring runs.".into(),side_effect:true,
        parameters:json!({"type":"object","additionalProperties":false,"properties":{"imagepath":{"type":"string"},"question":{"type":"string"},"samples":{"type":"integer","minimum":1,"maximum":60},"interval_ms":{"type":"integer","minimum":100,"maximum":60000},"cooldown_ms":{"type":"integer","minimum":0,"maximum":300000},"change_threshold":{"type":"number","minimum":0,"maximum":1},"analyze_first":{"type":"boolean","default":false}},"required":["imagepath","question","samples","interval_ms","cooldown_ms","change_threshold"]}) }
}
fn pixels(bytes: &[u8], config: &VisionConfig) -> Result<(Vec<u8>, CleanImage)> {
    let clean = sanitize_image(bytes, config)?;
    let png = STANDARD.decode(&clean.b64)?;
    let small = image::load_from_memory(&png)?
        .resize_exact(32, 32, image::imageops::FilterType::Triangle)
        .to_rgb8()
        .into_raw();
    Ok((small, clean))
}
fn difference(a: &[u8], b: &[u8]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| a.abs_diff(*b) as u64)
        .sum::<u64>() as f64
        / (a.len().max(1) as f64 * 255.0)
}
pub(super) async fn execute(
    tools: &VisionTools,
    ctx: &ToolContext,
    call: &ToolCall,
) -> Result<Value> {
    let r = parse(tools, call)?;
    let mut baseline: Option<Vec<u8>> = None;
    let mut last_analysis: Option<tokio::time::Instant> = None;
    let mut observations = Vec::new();
    let mut suppressed = 0;
    for sample in 0..r.samples {
        ensure!(!ctx.cancellation.is_cancelled(), "vision monitor cancelled");
        if sample > 0 {
            tokio::select! { _ = ctx.cancellation.cancelled() => bail!("vision monitor cancelled"), _ = tokio::time::sleep(Duration::from_millis(r.interval_ms)) => {} }
        }
        let (current, clean) = pixels(&tools.policy.read_file(&r.imagepath)?, &tools.config)?;
        let delta = baseline.as_ref().map(|b| difference(b, &current));
        let changed = delta
            .map(|d| d > r.change_threshold)
            .unwrap_or(r.analyze_first);
        let cooled =
            last_analysis.is_none_or(|t| t.elapsed() >= Duration::from_millis(r.cooldown_ms));
        // Compare to last analyzed frame so gradual changes accumulate.
        if changed && cooled {
            ensure!(
                observations.len() < 32,
                "monitor analysis output limit reached"
            );
            let analysis = tools
                .run_images(
                    ctx,
                    Request {
                        paths: vec![r.imagepath.clone()],
                        prompt: r.question.clone(),
                        max_objects: None,
                    },
                    vec![clean],
                )
                .await?;
            observations.push(json!({"sample":sample,"change":delta,"analysis":analysis}));
            baseline = Some(current);
            last_analysis = Some(tokio::time::Instant::now());
        } else {
            suppressed += 1;
            if baseline.is_none() {
                baseline = Some(current);
            }
        }
    }
    Ok(
        json!({"untrusted":true,"samples":r.samples,"suppressed":suppressed,"observations":observations,"capture":false,"recurring":false}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn threshold_is_pixel_based() {
        assert_eq!(difference(&[0, 0, 0], &[255, 255, 255]), 1.0);
        assert_eq!(difference(&[7, 7, 7], &[7, 7, 7]), 0.0);
    }
    #[test]
    fn sanitized_pixels_ignore_metadata() {
        let mut data = Vec::new();
        image::DynamicImage::new_rgb8(8, 8)
            .write_to(&mut Cursor::new(&mut data), ImageFormat::Png)
            .unwrap();
        let (small, clean) = pixels(&data, &VisionConfig::default()).unwrap();
        assert_eq!(small.len(), 32 * 32 * 3);
        assert_eq!(clean.width, 8);
        assert!(pixels(b"not an image", &VisionConfig::default()).is_err());
    }
}
