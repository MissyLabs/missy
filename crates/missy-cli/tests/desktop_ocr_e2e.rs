//! Native OCR on a synthetic text PNG only, never a display or account.
#[path = "../src/desktop_ocr.rs"]
mod desktop_ocr;
use anyhow::Result;
use missy_core::{PolicyConfig, Scope, ToolContext};
use missy_specialty::{DocumentBackendConfig, DocumentExecutablePin, SpecialtyConfig};
use sha2::{Digest, Sha256};
#[tokio::test]
async fn real_pinned_tesseract_adapter_preserves_native_word_boxes() -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir()?;
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700))?;
    let image = dir.path().join("fixture.png");
    let status = std::process::Command::new("/usr/bin/ffmpeg").env_clear().args(["-nostdin","-v","error","-f","lavfi","-i","color=white:s=1000x200","-vf","drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='MISSY FIXTURE TEXT':fontsize=64:fontcolor=black:x=20:y=60","-frames:v","1","-threads","1"]).arg(&image).status()?;
    assert!(status.success());
    let path = std::fs::canonicalize("/usr/bin/tesseract")?;
    let config = SpecialtyConfig {
        enabled: true,
        enable_documents: true,
        document_backend: Some(DocumentBackendConfig {
            pdftotext: None,
            tesseract: Some(DocumentExecutablePin {
                sha256: hex::encode(Sha256::digest(std::fs::read(&path)?)),
                path,
            }),
            staging_root: dir.path().to_owned(),
            runtime_read_roots: vec![
                "/usr/bin".into(),
                "/usr/lib".into(),
                "/usr/share/tesseract-ocr".into(),
            ],
            timeout_ms: 10000,
        }),
        ..Default::default()
    };
    let policy = PolicyConfig {
        read_roots: vec![dir.path().to_owned()],
        write_roots: vec![dir.path().to_owned()],
        max_file_bytes: 1024 * 1024,
        ..Default::default()
    };
    let backend = desktop_ocr::build(policy.clone(), config.clone())?.unwrap();
    let ctx = ToolContext {
        scope: Scope {
            tenant: "t".into(),
            user: "u".into(),
            channel: "c".into(),
            session: "s".into(),
        },
        turn_id: "synthetic".into(),
        cancellation: Default::default(),
    };
    let png = std::fs::read(&image)?;
    let tsv = backend.recognize_tsv(&ctx, png.clone()).await?;
    let r = missy_desktop::parse_tesseract_tsv(&tsv, 1000, 200)?;
    assert_eq!(
        r.anchors
            .iter()
            .map(|a| a.text.as_str())
            .collect::<Vec<_>>(),
        vec!["MISSY", "FIXTURE", "TEXT"]
    );
    let a = &r.anchors[0];
    assert_eq!(
        (a.left, a.top, a.right, a.bottom, a.center_x, a.center_y),
        (26, 60, 214, 109, 120, 84)
    );
    let a = &r.anchors[1];
    assert_eq!((a.left, a.top, a.right, a.bottom), (241, 61, 500, 109));
    assert_eq!(r.coordinate_space, "native_window_pixels");
    let denied = desktop_ocr::build(PolicyConfig::default(), config.clone())?.unwrap();
    assert!(denied.recognize_tsv(&ctx, png.clone()).await.is_err());
    let mut disabled = config.clone();
    disabled.enabled = false;
    assert!(desktop_ocr::build(policy.clone(), disabled)?.is_none());
    let mut bad = config;
    bad.document_backend
        .as_mut()
        .unwrap()
        .tesseract
        .as_mut()
        .unwrap()
        .sha256 = "0".repeat(64);
    assert!(desktop_ocr::build(policy, bad).is_err());
    ctx.cancellation.cancel();
    assert!(backend.recognize_tsv(&ctx, png).await.is_err());
    assert!(!std::fs::read_dir(dir.path())?.any(|e| {
        e.unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with("screen-ocr-")
    }));
    Ok(())
}
