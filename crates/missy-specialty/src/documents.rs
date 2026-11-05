//! Actual PDF text extraction / OCR using pinned native binaries in child-only
//! Landlock, with TCP denied and no arbitrary executable/flag/language inputs.
use super::*;
use std::{
    io::{Cursor, Read},
    os::unix::fs::MetadataExt,
};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentExecutablePin {
    pub path: PathBuf,
    pub sha256: String,
}
impl DocumentExecutablePin {
    fn verify(&self) -> Result<()> {
        ensure!(
            self.path.is_absolute() && self.path.canonicalize()? == self.path,
            "document executable must be canonical"
        );
        let f = std::fs::File::open(&self.path)?;
        let m = f.metadata()?;
        ensure!(
            m.is_file()
                && m.mode() & 0o022 == 0
                && m.mode() & 0o111 != 0
                && m.len() <= 128 * 1024 * 1024,
            "document executable unsafe"
        );
        let mut bytes = Vec::new();
        f.take(128 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
        ensure!(
            hex::encode(Sha256::digest(&bytes)) == self.sha256,
            "document executable pin mismatch"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentBackendConfig {
    pub pdftotext: Option<DocumentExecutablePin>,
    pub tesseract: Option<DocumentExecutablePin>,
    pub staging_root: PathBuf,
    /// Exact canonical /usr subdirectories for binaries, libraries and (OCR)
    /// trained data. Never grant /home, /etc, /proc or an entire filesystem.
    pub runtime_read_roots: Vec<PathBuf>,
    pub timeout_ms: u64,
}
impl DocumentBackendConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            missy_sandbox::landlock_abi()? >= 4,
            "documents require Landlock ABI >=4"
        );
        ensure!(
            (1..=60000).contains(&self.timeout_ms),
            "document timeout invalid"
        );
        ensure!(
            self.staging_root.is_dir() && self.staging_root.canonicalize()? == self.staging_root,
            "document staging must be canonical"
        );
        ensure!(
            !self.runtime_read_roots.is_empty() && self.runtime_read_roots.len() <= 12,
            "document runtime roots invalid"
        );
        for p in &self.runtime_read_roots {
            ensure!(
                p.is_dir()
                    && p.canonicalize()? == *p
                    && p.starts_with("/usr")
                    && p != std::path::Path::new("/usr"),
                "document runtime root unsafe"
            );
        }
        for pin in [&self.pdftotext, &self.tesseract].into_iter().flatten() {
            pin.verify()?;
            ensure!(
                self.runtime_read_roots
                    .iter()
                    .any(|p| pin.path.starts_with(p)),
                "document binary outside runtime roots"
            );
        }
        Ok(())
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DocumentRequest {
    input: PathBuf,
    #[serde(default = "pages")]
    max_pages: u32,
}
fn pages() -> u32 {
    20
}
impl DocumentRequest {
    pub fn validate(&self, name: &str, c: &DocumentBackendConfig) -> Result<()> {
        ensure!(
            self.input.is_absolute() && (1..=100).contains(&self.max_pages),
            "invalid document path/page count"
        );
        ensure!(
            if name == "document_ocr" {
                c.tesseract.is_some()
            } else {
                c.pdftotext.is_some()
            },
            "document backend not configured"
        );
        if name == "document_ocr" {
            ensure!(
                self.max_pages == 20,
                "max_pages applies only to PDF extraction"
            );
        }
        Ok(())
    }
}
pub(crate) fn definitions(c: &DocumentBackendConfig) -> Vec<ToolDefinition> {
    let mut out = Vec::new();
    if c.pdftotext.is_some() {
        out.push(def("document_pdf_extract","Extract bounded text from a policy-readable PDF using pinned pdftotext inside a no-network file sandbox. No scripts or links executed. Image-only PDFs need separate page rasterization/OCR.",schema(json!({"input":{"type":"string"},"max_pages":{"type":"integer","minimum":1,"maximum":100,"default":20}}), &["input"]),false));
    }
    if c.tesseract.is_some() {
        out.push(def("document_ocr","Decode and sanitize an explicit policy-readable PNG/JPEG, then recognize English text through pinned sandboxed Tesseract. No camera/screen capture, playback or arbitrary executable flags.",schema(json!({"input":{"type":"string"}}), &["input"]),false));
    }
    out
}
pub(crate) async fn execute(
    policy: &Policy,
    c: &DocumentBackendConfig,
    limit: usize,
    ctx: &ToolContext,
    name: &str,
    r: DocumentRequest,
) -> Result<Value> {
    r.validate(name, c)?;
    ensure!(!ctx.cancellation.is_cancelled(), "document cancelled");
    let bytes = policy.read_file(&r.input)?;
    ensure!(
        bytes.len() <= limit,
        "document input exceeds configured cap"
    );
    let stage = tempfile::Builder::new()
        .prefix("document-")
        .tempdir_in(&c.staging_root)?;
    let (pin, args) = if name == "document_pdf_extract" {
        ensure!(bytes.starts_with(b"%PDF-"), "input is not PDF");
        let input = stage.path().join("input.pdf");
        std::fs::write(&input, bytes)?;
        (
            c.pdftotext.as_ref().unwrap(),
            vec![
                "-f".into(),
                "1".into(),
                "-l".into(),
                r.max_pages.to_string(),
                "-enc".into(),
                "UTF-8".into(),
                "-nopgbrk".into(),
                input.to_string_lossy().into_owned(),
                "-".into(),
            ],
        )
    } else {
        let format = image::guess_format(&bytes)?;
        ensure!(
            matches!(format, image::ImageFormat::Png | image::ImageFormat::Jpeg),
            "OCR requires PNG/JPEG"
        );
        let mut reader = image::ImageReader::with_format(Cursor::new(bytes), format);
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(8192);
        limits.max_image_height = Some(8192);
        limits.max_alloc = Some(64 * 1024 * 1024);
        reader.limits(limits);
        let decoded = reader.decode()?;
        ensure!(
            decoded.width() as u64 * decoded.height() as u64 <= 16_000_000,
            "OCR pixel limit exceeded"
        );
        let input = stage.path().join("input.png");
        decoded.to_rgb8().save(&input)?;
        (
            c.tesseract.as_ref().unwrap(),
            vec![
                input.to_string_lossy().into_owned(),
                "stdout".into(),
                "-l".into(),
                "eng".into(),
                "--psm".into(),
                "3".into(),
            ],
        )
    };
    pin.verify()?;
    let mut argv = vec![pin.path.to_string_lossy().into_owned()];
    argv.extend(args);
    let mut reads = c.runtime_read_roots.clone();
    reads.push(stage.path().to_owned());
    let sandbox = missy_sandbox::SandboxedCommand::new(
        PolicyConfig {
            shell_enabled: true,
            allowed_commands: vec![argv.clone()],
            read_roots: reads.clone(),
            write_roots: vec![stage.path().to_owned()],
            ..Default::default()
        },
        missy_sandbox::SandboxConfig {
            read_roots: reads,
            write_roots: vec![stage.path().to_owned()],
            deny_tcp: true,
            timeout_ms: c.timeout_ms,
            max_output_bytes: limit,
        },
    )?;
    let result = sandbox.execute(&argv, ctx).await?;
    ensure!(
        result["success"] == true,
        "native document extraction failed"
    );
    let text = result["stdout"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("document output missing"))?;
    ensure!(text.len() <= limit, "document output exceeded cap");
    Ok(
        json!({"untrusted":true,"text":text,"operation":name,"max_pages":if name=="document_pdf_extract" {Some(r.max_pages)} else {None},"sandbox":"landlock-no-tcp","capture":false}),
    )
}

/// Recognize approved PNG bytes; no ambient screen/file discovery or arbitrary
/// flags. Child gets only explicit runtime roots plus a private per-call stage.
pub(crate) async fn ocr_png(
    policy: &Policy,
    c: &DocumentBackendConfig,
    limit: usize,
    ctx: &ToolContext,
    png: &[u8],
) -> Result<String> {
    ensure!(!ctx.cancellation.is_cancelled(), "OCR cancelled");
    ensure!(
        png.len() <= limit && png.starts_with(b"\x89PNG\r\n\x1a\n"),
        "OCR requires bounded PNG"
    );
    c.validate()?;
    let pin = c
        .tesseract
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Tesseract not configured"))?;
    check_output(policy, &c.staging_root.join("approved-ocr-stage"))?;
    ensure!(
        policy
            .config()
            .read_roots
            .iter()
            .any(|r| c.staging_root.starts_with(r)),
        "OCR staging outside read policy"
    );
    ensure!(
        std::fs::metadata(&c.staging_root)?.mode() & 0o077 == 0,
        "OCR staging root must be private"
    );
    let mut reader = image::ImageReader::with_format(Cursor::new(png), image::ImageFormat::Png);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(4096);
    limits.max_image_height = Some(4096);
    limits.max_alloc = Some(32 * 1024 * 1024);
    reader.limits(limits);
    let decoded = reader.decode()?;
    ensure!(
        decoded.width() as u64 * decoded.height() as u64 <= 4_000_000,
        "OCR pixel budget exceeded"
    );
    let mut sanitized = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(decoded.to_rgb8())
        .write_to(&mut sanitized, image::ImageFormat::Png)?;
    ensure!(
        sanitized.get_ref().len() <= limit,
        "sanitized OCR image exceeds byte budget"
    );
    ensure!(!ctx.cancellation.is_cancelled(), "OCR cancelled");
    let stage = tempfile::Builder::new()
        .prefix("screen-ocr-")
        .tempdir_in(&c.staging_root)?;
    let input = stage.path().join("input.png");
    policy.write_file(&input, sanitized.get_ref())?;
    let argv = vec![
        pin.path.to_string_lossy().into_owned(),
        input.to_string_lossy().into_owned(),
        "stdout".into(),
        "-l".into(),
        "eng".into(),
        "--psm".into(),
        "3".into(),
        "tsv".into(),
    ];
    pin.verify()?;
    let mut reads = c.runtime_read_roots.clone();
    reads.push(stage.path().to_owned());
    let sandbox = missy_sandbox::SandboxedCommand::new(
        PolicyConfig {
            shell_enabled: true,
            allowed_commands: vec![argv.clone()],
            read_roots: reads.clone(),
            write_roots: vec![stage.path().to_owned()],
            ..Default::default()
        },
        missy_sandbox::SandboxConfig {
            read_roots: reads,
            write_roots: vec![stage.path().to_owned()],
            deny_tcp: true,
            timeout_ms: c.timeout_ms,
            max_output_bytes: limit.min(1024 * 1024),
        },
    )?;
    let result = sandbox.execute(&argv, ctx).await?;
    ensure!(result["success"] == true, "screen OCR backend failed");
    let tsv = result["stdout"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("OCR TSV missing"))?;
    ensure!(
        tsv.len() <= limit.min(1024 * 1024) && !tsv.contains('\u{fffd}'),
        "OCR TSV invalid or oversized"
    );
    Ok(tsv.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pin(path: &str) -> DocumentExecutablePin {
        let path = std::fs::canonicalize(path).unwrap();
        DocumentExecutablePin {
            sha256: hex::encode(Sha256::digest(std::fs::read(&path).unwrap())),
            path,
        }
    }
    fn pdf() -> Vec<u8> {
        let stream = "BT /F1 24 Tf 50 100 Td (MISSY FIXTURE DOCUMENT) Tj ET";
        let objects=["<< /Type /Catalog /Pages 2 0 R >>".to_owned(),"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".into(),"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 200] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".into(),"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".into(),format!("<< /Length {} >>\nstream\n{}\nendstream",stream.len(),stream)];
        let mut text = "%PDF-1.4\n".to_owned();
        let mut offsets = vec![0];
        for (i, obj) in objects.iter().enumerate() {
            offsets.push(text.len());
            text.push_str(&format!("{} 0 obj\n{}\nendobj\n", i + 1, obj));
        }
        let start = text.len();
        text.push_str("xref\n0 6\n0000000000 65535 f \n");
        for n in offsets.iter().skip(1) {
            text.push_str(&format!("{n:010} 00000 n \n"));
        }
        text.push_str(&format!(
            "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n"
        ));
        text.into_bytes()
    }
    #[tokio::test]
    async fn actual_sandboxed_pdf_and_ocr_on_synthetic_fixtures() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o700)).unwrap();
        let c = DocumentBackendConfig {
            pdftotext: Some(pin("/usr/bin/pdftotext")),
            tesseract: Some(pin("/usr/bin/tesseract")),
            staging_root: root.to_owned(),
            runtime_read_roots: vec![
                "/usr/bin".into(),
                "/usr/lib".into(),
                "/usr/share/tesseract-ocr".into(),
            ],
            timeout_ms: 10000,
        };
        c.validate().unwrap();
        let p = Policy::new(PolicyConfig {
            read_roots: vec![root.to_owned()],
            max_file_bytes: 1024 * 1024,
            ..Default::default()
        })
        .unwrap();
        let ctx = ToolContext {
            scope: missy_core::Scope {
                tenant: "test".into(),
                user: "test".into(),
                channel: "test".into(),
                session: "test".into(),
            },
            turn_id: "fixture".into(),
            cancellation: Default::default(),
        };
        let input = root.join("fixture.pdf");
        std::fs::write(&input, pdf()).unwrap();
        let out = execute(
            &p,
            &c,
            1024 * 1024,
            &ctx,
            "document_pdf_extract",
            DocumentRequest {
                input,
                max_pages: 1,
            },
        )
        .await
        .unwrap();
        assert!(
            out["text"]
                .as_str()
                .unwrap()
                .contains("MISSY FIXTURE DOCUMENT")
        );
        // Synthetic text image only; no real display/camera ever opened.
        let input = root.join("fixture.png");
        let status=std::process::Command::new("/usr/bin/ffmpeg").args(["-nostdin","-v","error","-f","lavfi","-i","color=white:s=1000x200","-vf","drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='MISSY FIXTURE TEXT':fontsize=64:fontcolor=black:x=20:y=60","-frames:v","1","-threads","1"]).arg(&input).status().unwrap();
        assert!(status.success());
        let out = execute(
            &p,
            &c,
            1024 * 1024,
            &ctx,
            "document_ocr",
            DocumentRequest {
                input: input.clone(),
                max_pages: 20,
            },
        )
        .await
        .unwrap();
        assert!(out["text"].as_str().unwrap().contains("MISSY FIXTURE TEXT"));
        let policy = Policy::new(PolicyConfig {
            read_roots: vec![root.to_owned()],
            write_roots: vec![root.to_owned()],
            max_file_bytes: 1024 * 1024,
            ..Default::default()
        })
        .unwrap();
        let png = std::fs::read(&input).unwrap();
        let tsv = ocr_png(&policy, &c, 1024 * 1024, &ctx, &png).await.unwrap();
        let words: Vec<_> = tsv
            .lines()
            .skip(1)
            .filter_map(|line| {
                let f: Vec<_> = line.splitn(12, '\t').collect();
                (f[0] == "5").then(|| (f[11].to_owned(), f[6..10].join(",")))
            })
            .collect();
        assert_eq!(
            words,
            vec![
                ("MISSY".into(), "26,60,188,49".into()),
                ("FIXTURE".into(), "241,61,259,48".into()),
                ("TEXT".into(), "524,61,162,47".into())
            ]
        );
        assert!(!std::fs::read_dir(root).unwrap().any(|e| {
            e.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("screen-ocr-")
        }));
        assert!(
            ocr_png(&p, &c, 1024 * 1024, &ctx, &png)
                .await
                .unwrap_err()
                .to_string()
                .contains("write policy")
        );
        assert!(
            ocr_png(&policy, &c, 1024 * 1024, &ctx, b"not png")
                .await
                .is_err()
        );
        let mut bad_pin = c.clone();
        bad_pin.tesseract.as_mut().unwrap().sha256 = "0".repeat(64);
        assert!(
            ocr_png(&policy, &bad_pin, 1024 * 1024, &ctx, &png)
                .await
                .unwrap_err()
                .to_string()
                .contains("pin mismatch")
        );
        ctx.cancellation.cancel();
        assert!(
            ocr_png(&policy, &c, 1024 * 1024, &ctx, &png)
                .await
                .unwrap_err()
                .to_string()
                .contains("cancelled")
        );
    }
}
