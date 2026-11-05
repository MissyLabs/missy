//! Explicit offline acceptance only. Never downloads or contacts an existing model service.
//! Usage: cargo run -p missy-embedding --example neural_acceptance -- /path/to/llama-server /path/to/model.gguf
use anyhow::{Result, ensure};
use missy_core::Scope;
use missy_embedding::{LocalEmbeddingConfig, LocalHttpEmbedder};
use missy_retrieval::{Embedder, HashingEmbedder, Retrieval};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::Read,
    net::TcpListener,
    os::unix::fs::PermissionsExt,
    process::{Child, Command, Stdio},
    sync::Arc,
    time::{Duration, Instant},
};

const MODEL_SHA: &str = "797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662";
struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum()
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    ensure!(
        args.len() == 3,
        "expected explicit llama-server and model paths"
    );
    let mut input = File::open(&args[2])?;
    ensure!(
        input.metadata()?.len() == 45_949_216,
        "unexpected model size"
    );
    let mut sha = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        sha.update(&buffer[..n]);
    }
    ensure!(
        hex::encode(sha.finalize()) == MODEL_SHA,
        "model SHA256 mismatch"
    );
    let reservation = TcpListener::bind("127.0.0.1:0")?;
    let port = reservation.local_addr()?.port();
    drop(reservation);
    let log = tempfile::tempfile()?;
    let mut server = Server(
        Command::new(&args[1])
            .args(["--model"])
            .arg(&args[2])
            .args([
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--alias",
                "acceptance-minilm",
                "--embedding",
                "--pooling",
                "mean",
                "--gpu-layers",
                "0",
                "--threads",
                "2",
                "--ctx-size",
                "512",
                "--batch-size",
                "512",
                "--ubatch-size",
                "512",
                "--parallel",
                "1",
            ])
            .env("CUDA_VISIBLE_DEVICES", "")
            .stdin(Stdio::null())
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log))
            .spawn()?,
    );
    let origin = format!("http://127.0.0.1:{port}");
    let embedder = Arc::new(LocalHttpEmbedder::new(LocalEmbeddingConfig {
        endpoint: format!("{origin}/v1/embeddings"),
        allowed_origin: origin,
        allow_private_network: true,
        model: "acceptance-minilm".into(),
        model_sha256: MODEL_SHA.into(),
        dimensions: 384,
        timeout_seconds: 5,
    })?);
    let start = Instant::now();
    let q = "How do I repair my automobile?";
    let qv = loop {
        if let Some(status) = server.0.try_wait()? {
            anyhow::bail!("dedicated model server exited: {status}");
        }
        match embedder.embed(q) {
            Ok(vector) => break vector,
            Err(error) if start.elapsed() < Duration::from_secs(30) => {
                let _ = error;
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(error) => return Err(error),
        }
    };
    let lexical = HashingEmbedder;
    let repeated = embedder.embed(q)?;
    ensure!(
        cosine(&qv, &repeated) > 0.99999,
        "repeat embedding mismatch"
    );
    // Explicit smoke corpus, not a general semantic-retrieval benchmark. Retain the
    // difficult quotation as a diagnostic: this small model barely separates it.
    let cases = [
        (
            "vehicle paraphrase",
            q,
            "A mechanic can fix your broken car.",
            "Automobile and repair are vocabulary words printed in an alphabet puzzle.",
            true,
        ),
        (
            "sleep paraphrase",
            "I cannot sleep at night.",
            "Insomnia keeps me awake until morning.",
            "Sleep and night are words printed on a grammar worksheet.",
            true,
        ),
        (
            "medical paraphrase",
            "Where can I get medicine for a headache?",
            "The pharmacy sells painkillers for migraines.",
            "Medicine and headache are vocabulary words in an alphabet puzzle.",
            true,
        ),
        (
            "adversarial exact quotation",
            q,
            "A mechanic can fix your broken car.",
            "How do I repair my automobile is the title of a poem about astronomy.",
            false,
        ),
    ];
    let mut results = vec![];
    let mut all_pass = true;
    for (label, query, relevant, decoy, required) in cases {
        let query_vector = embedder.embed(query)?;
        let semantic = cosine(&query_vector, &embedder.embed(relevant)?);
        let lexical_decoy = cosine(&query_vector, &embedder.embed(decoy)?);
        let lexical_q = lexical.embed(query)?;
        let control_relevant = cosine(&lexical_q, &lexical.embed(relevant)?);
        let control_decoy = cosine(&lexical_q, &lexical.embed(decoy)?);
        let pass = semantic > lexical_decoy + 0.03 && control_decoy > control_relevant;
        if required {
            all_pass &= pass;
        }
        results.push(serde_json::json!({
            "case": label, "query": query, "semantic_document": relevant,
            "lexical_decoy_document": decoy, "neural_cosine_relevant": semantic,
            "neural_cosine_decoy": lexical_decoy, "hashing_cosine_relevant": control_relevant,
            "hashing_cosine_decoy": control_decoy, "required": required,
            "semantic_margin_0_03_and_lexical_decoy_control_pass": pass,
        }));
    }
    let temp = tempfile::tempdir()?;
    std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o700))?;
    let scope = Scope {
        tenant: "acceptance".into(),
        user: "fixture".into(),
        channel: "isolated".into(),
        session: "test".into(),
    };
    let retrieval = Retrieval::open_with_embedder(temp.path().join("neural.db"), embedder.clone())?;
    retrieval.ingest(
        &scope,
        "mechanic",
        "A mechanic can fix your broken car.",
        "acceptance:paraphrase",
    )?;
    retrieval.ingest(
        &scope,
        "astronomy",
        "Planets orbit the sun in the solar system.",
        "acceptance:unrelated",
    )?;
    let hits = retrieval.search(&scope, q, 2)?;
    let retrieval_pass = hits.first().is_some_and(|hit| hit.doc_id == "mechanic");
    drop(retrieval);
    let evidence = serde_json::json!({
        "backend": "dedicated llama.cpp CPU-only GGUF BERT mean pooling",
        "model_repository": "second-state/All-MiniLM-L6-v2-Embedding-GGUF",
        "model_revision": "544f204f2eaa2d71361ffc74d6df7170285b286a",
        "model_sha256_verified": MODEL_SHA, "dimensions": qv.len(),
        "cases": results, "all_required_cases_pass": all_pass,
        "persistent_retrieval_top_result": hits.first().map(|hit| &hit.doc_id),
        "persistent_retrieval_pass": retrieval_pass,
        "elapsed_seconds": start.elapsed().as_secs_f64(),
        "identity": embedder.name(),
    });
    println!("{}", serde_json::to_string_pretty(&evidence)?);
    ensure!(
        all_pass && retrieval_pass,
        "neural semantic ranking acceptance failed"
    );
    drop(embedder);
    drop(server);
    println!("Dedicated model server reaped; no existing service used or modified.");
    Ok(())
}
