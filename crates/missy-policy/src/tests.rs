use super::*;
use std::{
    os::unix::fs::{PermissionsExt, symlink},
    sync::atomic::{AtomicBool, Ordering},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

fn fs_policy(root: &Path) -> Policy {
    Policy::new(PolicyConfig {
        read_roots: vec![root.into()],
        write_roots: vec![root.into()],
        max_file_bytes: 64,
        ..Default::default()
    })
    .unwrap()
}
fn network_policy(origin: &str, private: bool) -> PolicyConfig {
    PolicyConfig {
        allowed_origins: vec![origin.into()],
        allow_private_network: private,
        ..Default::default()
    }
}
async fn mock(response: Vec<u8>) -> (String, tokio::task::JoinHandle<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = vec![0; 8192];
        let count = stream.read(&mut request).await.unwrap();
        request.truncate(count);
        let _ = stream.write_all(&response).await;
        request
    });
    (origin, task)
}
#[test]
fn default_denies_everything() {
    let p = Policy::new(PolicyConfig::default()).unwrap();
    assert!(p.read_file("/etc/passwd").is_err());
    assert!(p.write_file("/tmp/missy-policy-no", b"x").is_err());
    assert!(p.check_command(&["/bin/true".into()]).is_err());
    assert!(p.check_mcp_command(&["/bin/true".into()]).is_err());
}
#[test]
fn exact_argv_capabilities_independent() {
    let mut c = PolicyConfig {
        allowed_commands: vec![vec!["/usr/bin/printf".into(), "hello".into()]],
        mcp_enabled: true,
        ..Default::default()
    };
    let p = Policy::new(c.clone()).unwrap();
    assert!(p.check_mcp_command(&c.allowed_commands[0]).is_ok());
    assert!(p.check_command(&c.allowed_commands[0]).is_err());
    c.shell_enabled = true;
    let p = Policy::new(c).unwrap();
    assert!(
        p.check_command(&["/usr/bin/printf".into(), "hello".into()])
            .is_ok()
    );
    assert!(
        p.check_command(&["/usr/bin/printf".into(), "hello; id".into()])
            .is_err()
    );
    assert!(p.check_command(&["printf".into(), "hello".into()]).is_err());
    assert!(
        Policy::new(PolicyConfig {
            allowed_commands: vec![vec!["sh".into()]],
            ..Default::default()
        })
        .is_err()
    );
}
#[test]
fn private_atomic_files_and_bounds() {
    let tmp = tempfile::tempdir().unwrap();
    let p = fs_policy(tmp.path());
    let file = tmp.path().join("test");
    p.write_file(&file, b"first").unwrap();
    assert_eq!(p.read_file(&file).unwrap(), b"first");
    assert_eq!(file.metadata().unwrap().permissions().mode() & 0o777, 0o600);
    assert_eq!(file.metadata().unwrap().uid(), unsafe { libc::geteuid() });
    p.write_file(&file, b"replacement").unwrap();
    assert_eq!(p.read_file(&file).unwrap(), b"replacement");
    assert!(p.write_file(&file, &[0; 65]).is_err());
    std::fs::write(&file, [0; 65]).unwrap();
    assert!(p.read_file(&file).is_err());
    assert_eq!(p.list_files(tmp.path()).unwrap(), vec!["test"]);
}
#[test]
fn path_symlink_and_hardlink_escapes_denied() {
    let tmp = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("secret"), b"secret").unwrap();
    let p = fs_policy(tmp.path());
    symlink(outside.path(), tmp.path().join("escape")).unwrap();
    symlink(outside.path().join("secret"), tmp.path().join("file")).unwrap();
    std::fs::hard_link(outside.path().join("secret"), tmp.path().join("hard")).unwrap();
    for path in [
        tmp.path().join("escape/secret"),
        tmp.path().join("file"),
        tmp.path().join("hard"),
        tmp.path().join("../secret"),
        tmp.path().join("./file"),
    ] {
        assert!(p.read_file(&path).is_err(), "{path:?}");
        assert!(p.write_file(&path, b"corrupted").is_err(), "{path:?}");
    }
    assert!(p.list_files(&tmp.path().join("escape")).is_err());
    assert!(p.read_file("relative").is_err());
    assert!(p.read_file(outside.path().join("secret")).is_err());
    assert_eq!(
        std::fs::read(outside.path().join("secret")).unwrap(),
        b"secret"
    );
}
#[test]
fn symlink_substitution_race_never_reads_or_writes_outside() {
    let tmp = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let inside = tmp.path().join("real");
    std::fs::create_dir(&inside).unwrap();
    std::fs::write(inside.join("file"), b"inside").unwrap();
    std::fs::write(outside.path().join("file"), b"OUTSIDE").unwrap();
    let p = fs_policy(tmp.path());
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    let path = tmp.path().join("raced");
    let outside_path = outside.path().to_owned();
    let race_path = path.clone();
    let racer = std::thread::spawn(move || {
        while !stop2.load(Ordering::Relaxed) {
            let _ = symlink(&outside_path, &race_path);
            let _ = std::fs::remove_file(&race_path);
            let _ = std::fs::rename(&inside, &race_path);
            let _ = std::fs::rename(&race_path, &inside);
        }
    });
    for _ in 0..1000 {
        if let Ok(bytes) = p.read_file(path.join("file")) {
            assert_ne!(bytes, b"OUTSIDE");
        }
        let _ = p.write_file(path.join("file"), b"inside");
    }
    stop.store(true, Ordering::Relaxed);
    racer.join().unwrap();
    assert_eq!(
        std::fs::read(outside.path().join("file")).unwrap(),
        b"OUTSIDE"
    );
}

#[test]
fn url_validation_and_exact_origins() {
    for origin in [
        "http://*.example.com",
        "https://example.com/path",
        "https://example.com/a/..",
        "https://example.com?x=1",
        "http://u:p@example.com",
        "http://@example.com",
        "http://example.com#frag",
        "ftp://example.com",
        "http://example.com.",
        "http://example.com:0",
    ] {
        assert!(
            Policy::new(network_policy(origin, false)).is_err(),
            "{origin}"
        );
    }
    let p = Policy::new(network_policy("https://EXAMPLE.com:443", false)).unwrap();
    assert!(p.inner.origins.contains("https://example.com:443"));
    for raw in [
        "http:example.com",
        "http:/example.com",
        "http://@example.com",
        "http://example.com\\evil",
        "http://a:b@example.com",
        "http://example.com/#x",
        "http://example.com\n",
    ] {
        assert!(parse_url(raw).is_err(), "{raw}");
    }
}
#[test]
fn all_answers_checked_including_mapped_and_special_ranges() {
    for ip in [
        "0.1.2.3",
        "10.1.2.3",
        "100.64.0.1",
        "127.0.0.1",
        "169.254.169.254",
        "172.16.0.1",
        "192.168.0.1",
        "192.0.0.1",
        "198.18.0.1",
        "224.0.0.1",
        "255.255.255.255",
        "::1",
        "::",
        "::ffff:127.0.0.1",
        "::ffff:8.8.8.8",
        "fe80::1",
        "fd00::1",
        "2001:db8::1",
        "2002:7f00:1::",
        "64:ff9b::7f00:1",
    ] {
        let ip: IpAddr = ip.parse().unwrap();
        assert!(!public_ip(ip), "{ip}");
    }
    let public: SocketAddr = "8.8.8.8:443".parse().unwrap();
    let private: SocketAddr = "127.0.0.1:443".parse().unwrap();
    assert!(validate_answers(&[public], false).is_ok());
    assert!(validate_answers(&[public, private], false).is_err());
    assert!(validate_answers(&[public, private], true).is_err());
    assert!(validate_answers(&[private], true).is_ok());
    assert!(validate_answers(&[], true).is_err());
    assert!(validate_answers(&["[::ffff:127.0.0.1]:80".parse().unwrap()], true).is_err());
    assert!(validate_answers(&["0.0.0.0:80".parse().unwrap()], true).is_err());
    assert!(validate_answers(&["224.0.0.1:80".parse().unwrap()], true).is_err());
}
#[tokio::test]
async fn local_json_requires_private_and_exact_port_and_sends_auth() {
    let (origin, task) = mock(
        b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}".to_vec(),
    )
    .await;
    assert!(
        PolicyHttpClient::new(network_policy(&origin, false))
            .unwrap()
            .get(&origin)
            .await
            .is_err()
    );
    assert!(
        PolicyHttpClient::new(PolicyConfig {
            allow_private_network: true,
            ..Default::default()
        })
        .unwrap()
        .get(&origin)
        .await
        .is_err()
    );
    let client = PolicyHttpClient::new(network_policy(&origin, true)).unwrap();
    let value = client
        .request_json(
            "POST",
            &origin,
            vec![("Authorization".into(), "Bearer test-only".into())],
            Some(serde_json::json!({"input":"hello"})),
        )
        .await
        .unwrap();
    assert_eq!(value["ok"], true);
    let request = String::from_utf8_lossy(&task.await.unwrap()).to_string();
    assert!(request.contains("authorization: Bearer test-only"));
    let wrong_port = format!(
        "http://127.0.0.1:{}",
        if origin.ends_with(":1") { 2 } else { 1 }
    );
    assert!(
        client
            .get(&wrong_port)
            .await
            .unwrap_err()
            .to_string()
            .contains("origin denied")
    );
}
#[tokio::test]
async fn hostname_resolution_is_pinned_with_explicit_private_opt_in() {
    let (origin, task) =
        mock(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}".to_vec()).await;
    let localhost = origin.replace("127.0.0.1", "localhost");
    let client = PolicyHttpClient::new(network_policy(&localhost, true)).unwrap();
    assert_eq!(client.get(&localhost).await.unwrap().body, b"{}");
    task.await.unwrap();
}
#[tokio::test]
async fn redirects_not_followed_and_json_errors_sanitized() {
    let (origin,task) = mock(b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/secret-token\r\nSet-Cookie: credential=secret\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()).await;
    let client = PolicyHttpClient::new(network_policy(&origin, true)).unwrap();
    let response = client.get(&origin).await.unwrap();
    assert_eq!(response.status, 302);
    assert!(!response.headers.iter().any(|(k, _)| k == "set-cookie"));
    task.await.unwrap();
    let (origin, task) = mock(
        b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 12\r\nConnection: close\r\n\r\nSECRET-TOKEN"
            .to_vec(),
    )
    .await;
    let client = PolicyHttpClient::new(network_policy(&origin, true)).unwrap();
    let error = client
        .request_json("GET", &origin, vec![], None)
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "HTTP upstream returned status 401");
    task.await.unwrap();
}
#[tokio::test]
async fn bounded_http_and_routing_headers() {
    let (origin, task) = mock(
        format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            MAX_HTTP_RESPONSE_BYTES + 1
        )
        .into_bytes(),
    )
    .await;
    let client = PolicyHttpClient::new(network_policy(&origin, true)).unwrap();
    for name in [
        "Host",
        "connection",
        "transfer-encoding",
        "Content-Length",
        "Proxy-Authorization",
    ] {
        assert!(
            client
                .request("GET", &origin, vec![(name.into(), "secret".into())], None)
                .await
                .is_err()
        );
    }
    assert!(
        client
            .request(
                "POST",
                &origin,
                vec![],
                Some(vec![0; MAX_HTTP_REQUEST_BYTES + 1])
            )
            .await
            .is_err()
    );
    assert!(
        client
            .get(&origin)
            .await
            .unwrap_err()
            .to_string()
            .contains("exceeds limit")
    );
    task.await.unwrap();
    assert!(client.clone().with_timeout(Duration::ZERO).is_err());
    assert!(client.with_timeout(Duration::from_secs(601)).is_err());
}

#[tokio::test]
async fn chunked_overflow_is_bounded_without_content_length() {
    let mut response =
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n".to_vec();
    response.extend_from_slice(format!("{:x}\r\n", MAX_HTTP_RESPONSE_BYTES + 1).as_bytes());
    response.extend(vec![b'x'; MAX_HTTP_RESPONSE_BYTES + 1]);
    response.extend_from_slice(b"\r\n0\r\n\r\n");
    let (origin, task) = mock(response).await;
    let client = PolicyHttpClient::new(network_policy(&origin, true)).unwrap();
    assert!(
        client
            .get(&origin)
            .await
            .unwrap_err()
            .to_string()
            .contains("exceeds limit")
    );
    task.await.unwrap();
}

#[tokio::test]
async fn total_timeout_sanitized_and_future_drop_cancels() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let client = PolicyHttpClient::new(network_policy(&origin, true))
        .unwrap()
        .with_timeout(Duration::from_secs(1))
        .unwrap();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut bytes = [0; 8192];
        assert!(socket.read(&mut bytes).await.unwrap() > 0);
        // Client timeout drops the active transport, not just its result.
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(3), socket.read(&mut bytes))
                .await
                .unwrap()
                .unwrap(),
            0
        );
    });
    let error = client
        .get(&format!("{origin}/SECRET-TOKEN"))
        .await
        .unwrap_err()
        .to_string();
    assert!(!error.contains("SECRET-TOKEN"));
    assert!(error.contains("timed out") || error.contains("transport failed"));
    task.await.unwrap();
}

#[tokio::test]
async fn websocket_default_denial_and_redirect_rejection() {
    let client = PolicyHttpClient::new(PolicyConfig::default()).unwrap();
    assert!(
        client
            .connect_websocket("wss://example.com/", vec![])
            .await
            .is_err()
    );
    for raw in [
        "wss://u:p@example.com",
        "wss://@example.com",
        "wss://example.com/#fragment",
        "https://example.com",
        "ws://[::ffff:127.0.0.1]/",
    ] {
        assert!(client.connect_websocket(raw, vec![]).await.is_err());
    }
    let (origin,task) = mock(b"HTTP/1.1 302 Found\r\nLocation: ws://127.0.0.1:1/SECRET-TOKEN\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()).await;
    let client = PolicyHttpClient::new(network_policy(&origin, true)).unwrap();
    let error = client
        .connect_websocket(&origin.replacen("http://", "ws://", 1), vec![])
        .await
        .unwrap_err()
        .to_string();
    assert_eq!(error, "WebSocket TLS or handshake failed");
    task.await.unwrap();
}

#[tokio::test]
async fn websocket_pinned_local_handshake_private_optin() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://localhost:{}", listener.local_addr().unwrap().port());
    let ws_url = origin.replacen("http://", "ws://", 1);
    let client = PolicyHttpClient::new(network_policy(&origin, true)).unwrap();
    let task = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
        socket.close(None).await.unwrap();
    });
    let socket = client.connect_websocket(&ws_url, vec![]).await.unwrap();
    assert_eq!(
        socket.get_config().max_message_size,
        Some(MAX_HTTP_RESPONSE_BYTES)
    );
    drop(socket);
    task.await.unwrap();
}

#[tokio::test]
async fn origins_never_widen_to_subdomains_ports_schemes_or_suffixes() {
    let client = PolicyHttpClient::new(network_policy("https://example.com", true)).unwrap();
    for raw in [
        "http://example.com",
        "https://example.com:444",
        "https://sub.example.com",
        "https://example.com.evil.test",
        "https://example.com@evil.test",
        "https://example.com.",
    ] {
        let error = client.get(raw).await.unwrap_err().to_string();
        assert!(
            error.contains("origin denied")
                || error.contains("authority")
                || error.contains("hostname"),
            "{raw}: {error}"
        );
    }
    let client =
        PolicyHttpClient::new(network_policy("http://[::ffff:127.0.0.1]:80", true)).unwrap();
    assert!(
        client
            .get("http://[::ffff:127.0.0.1]:80")
            .await
            .unwrap_err()
            .to_string()
            .contains("mapped IPv6")
    );
}

#[test]
fn debug_response_does_not_leak_contents() {
    let response = HttpResponse {
        status: 200,
        headers: vec![("x-key".into(), "SECRET".into())],
        body: b"SECRET".to_vec(),
    };
    let debug = format!("{response:?}");
    assert!(!debug.contains("SECRET"));
    assert!(debug.contains("body_bytes"));
}

// Real chunked SSE without a terminating chunk. Drop/cancellation must close TCP.
async fn open_sse(heartbeat: bool) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://localhost:{}", listener.local_addr().unwrap().port());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut bytes = [0; 8192];
        while !request.windows(4).any(|w| w == b"\r\n\r\n") {
            let n = socket.read(&mut bytes).await.unwrap();
            assert!(n > 0);
            request.extend_from_slice(&bytes[..n]);
            assert!(request.len() <= 32 * 1024);
        }
        socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nX-Secret: STREAM-SECRET\r\nSet-Cookie: SECRET-COOKIE\r\n\r\n").await.unwrap();
        socket.write_all(b"b\r\ndata: one\n\n\r\n").await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                tokio::select! {
                    n = socket.read(&mut bytes) => {
                        match n {
                            Ok(0) => return,
                            Err(e) if e.kind() == std::io::ErrorKind::ConnectionReset => return,
                            other => panic!("expected closed streaming connection: {other:?}"),
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_millis(25)), if heartbeat => {
                        if socket.write_all(b"3\r\n:\n\n\r\n").await.is_err() {
                            return;
                        }
                    }
                }
            }
        })
        .await
        .expect("stream connection was not closed");
    });
    (origin, task)
}

async fn first_event(stream: &mut PolicyHttpStream) {
    let mut body = Vec::new();
    while !body.windows(11).any(|w| w == b"data: one\n\n") {
        body.extend(stream.chunk().await.unwrap().expect("SSE ended early"));
    }
}

#[tokio::test]
async fn streaming_sse_is_incremental_pinned_redacted_and_drop_closes() {
    let (origin, task) = open_sse(false).await;
    let client = PolicyHttpClient::new(network_policy(&origin, true))
        .unwrap()
        .with_timeout(Duration::from_secs(600))
        .unwrap()
        .with_chunk_timeout(Duration::from_secs(600))
        .unwrap();
    let mut stream = tokio::time::timeout(
        Duration::from_secs(2),
        client.request_stream("GET", &format!("{origin}/SECRET-PATH"), vec![], None),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(stream.status, 200);
    assert!(
        stream
            .headers
            .iter()
            .any(|(k, v)| k == "content-type" && v == "text/event-stream")
    );
    assert!(!stream.headers.iter().any(|(k, _)| k == "set-cookie"));
    assert!(!format!("{stream:?}").contains("SECRET"));
    tokio::time::timeout(Duration::from_secs(2), first_event(&mut stream))
        .await
        .unwrap();
    assert_eq!(stream.received, 11);
    drop(stream);
    task.await.unwrap();
}

#[tokio::test]
async fn cancelling_pending_stream_chunk_closes_and_is_terminal() {
    let (origin, task) = open_sse(false).await;
    let client = PolicyHttpClient::new(network_policy(&origin, true)).unwrap();
    let mut stream = client
        .request_stream("GET", &origin, vec![], None)
        .await
        .unwrap();
    first_event(&mut stream).await;
    assert!(
        tokio::time::timeout(Duration::from_millis(50), stream.chunk())
            .await
            .is_err()
    );
    // The outer object remains in scope, but the cancelled read dropped TCP.
    assert!(stream.chunk().await.unwrap().is_none());
    task.await.unwrap();
}

#[tokio::test]
async fn streaming_continuous_heartbeats_cannot_extend_total_deadline() {
    let (origin, task) = open_sse(true).await;
    let client = PolicyHttpClient::new(network_policy(&origin, true))
        .unwrap()
        .with_timeout(Duration::from_secs(1))
        .unwrap()
        .with_chunk_timeout(Duration::from_secs(600))
        .unwrap();
    let started = tokio::time::Instant::now();
    let mut stream = client
        .request_stream("GET", &origin, vec![], None)
        .await
        .unwrap();
    first_event(&mut stream).await;
    let error = loop {
        match stream.chunk().await {
            Ok(Some(_)) => {}
            Ok(None) => panic!("indefinite fixture unexpectedly ended"),
            Err(error) => break error,
        }
    };
    assert_eq!(error.to_string(), "HTTP response stream timed out");
    assert!(started.elapsed() < Duration::from_secs(3));
    assert!(stream.chunk().await.unwrap().is_none());
    task.await.unwrap();
}

#[tokio::test]
async fn streaming_idle_timeout_is_additional_to_total_deadline() {
    let (origin, task) = open_sse(false).await;
    let client = PolicyHttpClient::new(network_policy(&origin, true))
        .unwrap()
        .with_timeout(Duration::from_secs(600))
        .unwrap()
        .with_chunk_timeout(Duration::from_secs(1))
        .unwrap();
    let mut stream = client
        .request_stream("GET", &origin, vec![], None)
        .await
        .unwrap();
    first_event(&mut stream).await;
    let started = tokio::time::Instant::now();
    let error = stream.chunk().await.unwrap_err();
    assert_eq!(error.to_string(), "HTTP response stream timed out");
    assert!(started.elapsed() < Duration::from_secs(3));
    assert!(stream.chunk().await.unwrap().is_none());
    task.await.unwrap();
}

#[tokio::test]
async fn streaming_request_cancel_before_headers_closes_transport() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let (accepted_tx, accepted_rx) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut bytes = [0; 8192];
        assert!(socket.read(&mut bytes).await.unwrap() > 0);
        accepted_tx.send(()).unwrap();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(3), socket.read(&mut bytes))
                .await
                .unwrap()
                .unwrap(),
            0
        );
    });
    let client = PolicyHttpClient::new(network_policy(&origin, true)).unwrap();
    let mut request = Box::pin(client.request_stream("GET", &origin, vec![], None));
    tokio::select! {
        _ = accepted_rx => {},
        response = &mut request => panic!("fixture must not send headers: {response:?}"),
    }
    drop(request);
    task.await.unwrap();
}

#[tokio::test]
async fn streaming_redirects_not_followed_and_policy_validation_shared() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let target = format!("http://{}", listener.local_addr().unwrap());
    let (origin, task) = mock(
        format!(
            "HTTP/1.1 302 Found\r\nLocation: {target}/SECRET-REDIRECT\r\nContent-Length: 0\r\n\r\n"
        )
        .into_bytes(),
    )
    .await;
    let mut config = network_policy(&origin, true);
    config.allowed_origins.push(target);
    let client = PolicyHttpClient::new(config).unwrap();
    for name in [
        "Host",
        "Connection",
        "Transfer-Encoding",
        "Content-Length",
        "Proxy-Authorization",
        "Proxy-Connection",
        "Upgrade",
        "TE",
        "Trailer",
    ] {
        let error = client
            .request_stream("GET", &origin, vec![(name.into(), "SECRET".into())], None)
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "HTTP routing or framing header denied");
    }
    assert!(
        client
            .request_stream("TRACE", &origin, vec![], None)
            .await
            .is_err()
    );
    assert!(
        client
            .request_stream(
                "POST",
                &origin,
                vec![],
                Some(vec![0; MAX_HTTP_REQUEST_BYTES + 1])
            )
            .await
            .is_err()
    );
    assert!(
        client
            .request_stream(
                "GET",
                &origin,
                vec![
                    ("X-Test".into(), "one".into()),
                    ("x-test".into(), "two".into())
                ],
                None
            )
            .await
            .is_err()
    );
    let mut response = client
        .request_stream("GET", &origin, vec![], None)
        .await
        .unwrap();
    assert_eq!(response.status, 302);
    assert!(response.chunk().await.unwrap().is_none());
    assert!(
        tokio::time::timeout(Duration::from_millis(50), listener.accept())
            .await
            .is_err()
    );
    task.await.unwrap();
    assert!(client.clone().with_chunk_timeout(Duration::ZERO).is_err());
    assert!(client.with_chunk_timeout(Duration::from_secs(601)).is_err());
}

#[tokio::test]
async fn streaming_chunked_cumulative_limit_is_hard_and_terminal() {
    let mut response = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
    for _ in 0..=(MAX_HTTP_RESPONSE_BYTES / 1024) {
        response.extend_from_slice(b"400\r\n");
        response.extend_from_slice(&[b'x'; 1024]);
        response.extend_from_slice(b"\r\n");
    }
    response.extend_from_slice(b"0\r\n\r\n");
    let (origin, task) = mock(response).await;
    let client = PolicyHttpClient::new(network_policy(&origin, true)).unwrap();
    let mut stream = client
        .request_stream("GET", &origin, vec![], None)
        .await
        .unwrap();
    let mut received = 0;
    loop {
        match stream.chunk().await {
            Ok(Some(chunk)) => received += chunk.len(),
            Ok(None) => panic!("oversized stream accepted"),
            Err(error) => {
                assert_eq!(error.to_string(), "HTTP response exceeds limit");
                break;
            }
        }
    }
    assert!(received <= MAX_HTTP_RESPONSE_BYTES);
    assert!(stream.chunk().await.unwrap().is_none());
    task.await.unwrap();
}

#[tokio::test]
async fn streaming_validation_before_connect_and_declared_limits() {
    let (origin, task) = mock(
        format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
            MAX_HTTP_RESPONSE_BYTES + 1
        )
        .into_bytes(),
    )
    .await;
    let private_denied = PolicyHttpClient::new(network_policy(&origin, false)).unwrap();
    assert_eq!(
        private_denied
            .request_stream("GET", &origin, vec![], None)
            .await
            .unwrap_err()
            .to_string(),
        "HTTP address denied by network policy"
    );
    let denied = PolicyHttpClient::new(PolicyConfig::default()).unwrap();
    assert_eq!(
        denied
            .request_stream("GET", &origin, vec![], None)
            .await
            .unwrap_err()
            .to_string(),
        "HTTP origin denied by exact allowlist"
    );
    let client = PolicyHttpClient::new(network_policy(&origin, true)).unwrap();
    assert_eq!(
        client
            .request_stream("GET", &origin, vec![], None)
            .await
            .unwrap_err()
            .to_string(),
        "HTTP response exceeds limit"
    );
    task.await.unwrap();
}

#[tokio::test]
async fn streaming_total_deadline_includes_time_between_reads() {
    let (origin, task) = open_sse(false).await;
    let client = PolicyHttpClient::new(network_policy(&origin, true))
        .unwrap()
        .with_timeout(Duration::from_secs(1))
        .unwrap();
    let mut stream = client
        .request_stream("GET", &origin, vec![], None)
        .await
        .unwrap();
    first_event(&mut stream).await;
    tokio::time::sleep(Duration::from_millis(1100)).await;
    assert_eq!(
        stream.chunk().await.unwrap_err().to_string(),
        "HTTP response stream timed out"
    );
    assert!(stream.chunk().await.unwrap().is_none());
    task.await.unwrap();
}

#[tokio::test]
async fn streaming_exact_limit_and_eof_are_accepted() {
    let mut response = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {MAX_HTTP_RESPONSE_BYTES}\r\nConnection: close\r\n\r\n"
    )
    .into_bytes();
    response.resize(response.len() + MAX_HTTP_RESPONSE_BYTES, b'x');
    let (origin, task) = mock(response).await;
    let client = PolicyHttpClient::new(network_policy(&origin, true)).unwrap();
    let mut stream = client
        .request_stream("GET", &origin, vec![], None)
        .await
        .unwrap();
    let mut received = 0;
    while let Some(bytes) = stream.chunk().await.unwrap() {
        received += bytes.len();
    }
    assert_eq!(received, MAX_HTTP_RESPONSE_BYTES);
    assert!(stream.chunk().await.unwrap().is_none());
    task.await.unwrap();
}

#[tokio::test]
async fn streaming_transfer_errors_do_not_expose_upstream_or_url() {
    let (origin, task) =
        mock(b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\nConnection: close\r\n\r\nSECRET".to_vec())
            .await;
    let client = PolicyHttpClient::new(network_policy(&origin, true)).unwrap();
    let mut stream = client
        .request_stream("GET", &format!("{origin}/SECRET-URL"), vec![], None)
        .await
        .unwrap();
    let error = loop {
        match stream.chunk().await {
            Ok(Some(_)) => {}
            Ok(None) => panic!("truncated body accepted"),
            Err(error) => break error,
        }
    };
    assert_eq!(format!("{error:#}"), "HTTP response transfer failed");
    assert!(stream.chunk().await.unwrap().is_none());
    task.await.unwrap();
}

#[tokio::test]
async fn websocket_tls_never_falls_back_to_plaintext() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!(
        "https://127.0.0.1:{}",
        listener.local_addr().unwrap().port()
    );
    let ws_url = origin.replacen("https://", "wss://", 1);
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut bytes = [0; 8192];
        let count = socket.read(&mut bytes).await.unwrap();
        assert!(count > 0);
        assert_ne!(&bytes[..3], b"GET");
        // A hostile plaintext server cannot downgrade the approved wss route.
        let _ = socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\nSECRET-TOKEN")
            .await;
    });
    let client = PolicyHttpClient::new(network_policy(&origin, true)).unwrap();
    let error = client
        .connect_websocket(&ws_url, vec![])
        .await
        .unwrap_err()
        .to_string();
    assert_eq!(error, "WebSocket TLS or handshake failed");
    task.await.unwrap();
}

#[test]
fn fifo_denied_without_blocking_and_directory_bounds_enforced() {
    let tmp = tempfile::tempdir().unwrap();
    let p = fs_policy(tmp.path());
    let fifo = tmp.path().join("fifo");
    let name = cstr(fifo.as_os_str()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    assert!(p.read_file(&fifo).is_err());
    assert!(p.write_file(&fifo, b"x").is_err());
    for n in 0..4096 {
        std::fs::write(tmp.path().join(format!("{n}")), b"").unwrap();
    }
    assert!(
        p.list_files(tmp.path())
            .unwrap_err()
            .to_string()
            .contains("exceeds limit")
    );
}
