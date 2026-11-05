//! Protocol fixtures are NOT evidence of neural quality; see examples/neural_acceptance.rs.
use missy_embedding::{LocalEmbeddingConfig, LocalHttpEmbedder, MAX_INPUT_BYTES};
use missy_retrieval::Embedder;
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

fn config(origin: &str) -> LocalEmbeddingConfig {
    LocalEmbeddingConfig {
        endpoint: format!("{origin}/v1/embeddings"),
        allowed_origin: origin.into(),
        allow_private_network: true,
        model: "fixture".into(),
        model_sha256: "a".repeat(64),
        dimensions: 3,
        timeout_seconds: 2,
    }
}
fn fixture(body: &'static str, delay: Duration) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let handle = thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut bytes = vec![];
        let mut buf = [0; 2048];
        loop {
            let count = socket.read(&mut buf).unwrap();
            if count == 0 {
                break;
            }
            bytes.extend_from_slice(&buf[..count]);
            if let Some(pos) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                let headers = std::str::from_utf8(&bytes[..pos])
                    .unwrap()
                    .to_ascii_lowercase();
                let size: usize = headers
                    .lines()
                    .find_map(|v| v.strip_prefix("content-length: "))
                    .unwrap()
                    .parse()
                    .unwrap();
                if bytes.len() >= pos + 4 + size {
                    break;
                }
            }
        }
        let request = String::from_utf8(bytes).unwrap();
        assert!(request.starts_with("POST /v1/embeddings HTTP/1.1"));
        assert!(request.contains("\"encoding_format\":\"float\""));
        assert!(!request.to_lowercase().contains("authorization:"));
        thread::sleep(delay);
        let _ = write!(
            socket,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
    });
    (origin, handle)
}

#[test]
fn normalizes_real_protocol_and_works_inside_runtime() {
    let (origin, server) = fixture(
        r#"{"model":"fixture","data":[{"index":0,"embedding":[3,4,0]}]}"#,
        Duration::ZERO,
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let embedder = LocalHttpEmbedder::new(config(&origin)).unwrap();
        assert_eq!(embedder.dimension(), 3);
        assert_eq!(embedder.embed("test").unwrap(), vec![0.6, 0.8, 0.0]);
        assert!(embedder.name().len() < 256);
    });
    server.join().unwrap();
}

#[test]
fn rejects_wrong_model_dimension_empty_zero_index_and_multiple_results() {
    let bodies = [
        r#"{"model":"other","data":[{"index":0,"embedding":[3,4,0]}]}"#,
        r#"{"model":"fixture","data":[{"index":0,"embedding":[3,4]}]}"#,
        r#"{"model":"fixture","data":[]}"#,
        r#"{"model":"fixture","data":[{"index":0,"embedding":[0,0,0]}]}"#,
        r#"{"model":"fixture","data":[{"index":1,"embedding":[3,4,0]}]}"#,
        r#"{"model":"fixture","data":[{"index":0,"embedding":[3,4,0]},{"index":1,"embedding":[3,4,0]}]}"#,
        r#"{"model":"fixture","data":[{"index":0,"embedding":[1e1000,0,0]}]}"#,
        r#"{"model":"fixture","data":[{"index":0,"embedding":"AA=="}]}"#,
    ];
    for body in bodies {
        let (origin, server) = fixture(body, Duration::ZERO);
        let embedder = LocalHttpEmbedder::new(config(&origin)).unwrap();
        assert!(embedder.embed("test").is_err());
        server.join().unwrap();
    }
}

#[test]
fn configuration_is_explicit_local_credential_free_and_redacted() {
    let original = config("http://127.0.0.1:9999");
    assert!(!format!("{original:?}").contains("9999"));
    let serialized = serde_json::to_string(&original).unwrap();
    let _: LocalEmbeddingConfig = serde_json::from_str(&serialized).unwrap();
    let mut variants = vec![];
    for endpoint in [
        "https://example.org/v1/embeddings",
        "http://localhost:9999/v1/embeddings",
        "http://127.0.0.1:9999/v1/embeddings?key=secret",
        "http://secret@127.0.0.1:9999/v1/embeddings",
        "http://127.0.0.1:9999/other",
    ] {
        let mut c = original.clone();
        c.endpoint = endpoint.into();
        variants.push(c);
    }
    let mut c = original.clone();
    c.allow_private_network = false;
    variants.push(c);
    let mut c = original.clone();
    c.allowed_origin = "http://127.0.0.1:8888".into();
    variants.push(c);
    let mut c = original.clone();
    c.model_sha256.clear();
    variants.push(c);
    let mut c = original.clone();
    c.dimensions = 4097;
    variants.push(c);
    let mut c = original.clone();
    c.timeout_seconds = 0;
    variants.push(c);
    let mut c = original.clone();
    c.model = "\nsecret".into();
    variants.push(c);
    for c in variants {
        assert!(LocalHttpEmbedder::new(c).is_err());
    }
    let embedder = LocalHttpEmbedder::new(original).unwrap();
    assert!(embedder.embed("").is_err());
    assert!(embedder.embed("  ").is_err());
    assert!(embedder.embed(&"a".repeat(MAX_INPUT_BYTES + 1)).is_err());
    assert!(!format!("{embedder:?}").contains("9999"));
}

#[test]
fn identity_binds_endpoint_model_weights_and_dimension() {
    let c = config("http://127.0.0.1:9999");
    let original = LocalHttpEmbedder::new(c.clone()).unwrap().name().to_owned();
    let mut changes = vec![];
    let mut v = c.clone();
    v.model = "another-model".into();
    changes.push(v);
    let mut v = c.clone();
    v.model_sha256 = "b".repeat(64);
    changes.push(v);
    let mut v = c.clone();
    v.dimensions = 4;
    changes.push(v);
    let mut v = c.clone();
    v.endpoint = "http://127.0.0.1:9999/embeddings".into();
    changes.push(v);
    for change in changes {
        assert_ne!(LocalHttpEmbedder::new(change).unwrap().name(), original);
    }
    let mut uppercase = c;
    uppercase.model_sha256.make_ascii_uppercase();
    assert_eq!(LocalHttpEmbedder::new(uppercase).unwrap().name(), original);
}

#[test]
fn timeout_cancels_and_drop_reaps_worker() {
    let (origin, server) = fixture(
        r#"{"model":"fixture","data":[{"index":0,"embedding":[3,4,0]}]}"#,
        Duration::from_millis(1800),
    );
    let mut c = config(&origin);
    c.timeout_seconds = 1;
    let embedder = LocalHttpEmbedder::new(c).unwrap();
    let start = Instant::now();
    assert!(embedder.embed("test").is_err());
    assert!(start.elapsed() < Duration::from_millis(1600));
    let dropped = Instant::now();
    drop(embedder);
    assert!(dropped.elapsed() < Duration::from_millis(300));
    server.join().unwrap();
}

#[test]
fn concurrent_overload_is_bounded() {
    let (origin, server) = fixture(
        r#"{"model":"fixture","data":[{"index":0,"embedding":[3,4,0]}]}"#,
        Duration::from_millis(300),
    );
    let adapter = Arc::new(LocalHttpEmbedder::new(config(&origin)).unwrap());
    let mut callers = vec![];
    for _ in 0..12 {
        let a = adapter.clone();
        callers.push(thread::spawn(move || a.embed("test")));
    }
    let mut successes = 0;
    let mut busy = 0;
    for caller in callers {
        match caller.join().unwrap() {
            Ok(_) => successes += 1,
            Err(e) => {
                if e.to_string().contains("busy") {
                    busy += 1
                }
            }
        }
    }
    assert_eq!(successes, 1);
    assert!(busy >= 9);
    server.join().unwrap();
}
