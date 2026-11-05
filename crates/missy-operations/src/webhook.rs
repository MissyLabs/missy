use crate::{EventBus, OperationalKind, SharedRuntime, diagnostics};
use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use hmac::{Hmac, Mac};
use missy_core::Scope;
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};
use subtle::ConstantTimeEq;
pub const MAX_WEBHOOK_BODY: usize = 65_536;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OperationsConfig {
    pub bind: SocketAddr,
    pub scope: Scope,
    pub webhook_key_env: String,
    pub read_key_env: String,
    pub replay_db: PathBuf,
    pub max_clock_skew_secs: u64,
    pub max_replay_entries: usize,
}
impl Default for OperationsConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8091".parse().unwrap(),
            scope: Scope {
                tenant: "local".into(),
                user: "webhook".into(),
                channel: "webhook".into(),
                session: "default".into(),
            },
            webhook_key_env: "MISSY_WEBHOOK_HMAC_KEY".into(),
            read_key_env: "MISSY_OPERATIONS_READ_KEY".into(),
            replay_db: "operations-replay.sqlite3".into(),
            max_clock_skew_secs: 300,
            max_replay_entries: 100_000,
        }
    }
}
impl OperationsConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.bind.ip().is_loopback(),
            "operations listener must be loopback"
        );
        self.scope.validate()?;
        ensure!(
            (1..=900).contains(&self.max_clock_skew_secs),
            "clock skew bound must be 1..900 seconds"
        );
        ensure!(
            (1..=1_000_000).contains(&self.max_replay_entries),
            "invalid replay capacity"
        );
        for name in [&self.webhook_key_env, &self.read_key_env] {
            ensure!(
                name.starts_with("MISSY_")
                    && name.len() <= 128
                    && name
                        .bytes()
                        .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_'),
                "operations secrets require dedicated MISSY environment references"
            );
        }
        ensure!(
            self.webhook_key_env != self.read_key_env,
            "read and webhook keys must be distinct"
        );
        Ok(())
    }
}
struct Secrets {
    webhook: Vec<u8>,
    read: Vec<u8>,
}
impl Secrets {
    fn from_env(config: &OperationsConfig) -> Result<Self> {
        let read = |name: &str| -> Result<Vec<u8>> {
            let value = std::env::var(name)
                .map_err(|_| anyhow::anyhow!("required operations credential unavailable"))?;
            ensure!(
                (32..=4096).contains(&value.len()) && !value.bytes().any(|b| b.is_ascii_control()),
                "operations credentials require 32..4096 non-control bytes"
            );
            Ok(value.into_bytes())
        };
        let keys = Self {
            webhook: read(&config.webhook_key_env)?,
            read: read(&config.read_key_env)?,
        };
        ensure!(
            keys.webhook != keys.read,
            "operations credentials must be distinct"
        );
        Ok(keys)
    }
}
#[derive(Clone)]
struct App {
    runtime: SharedRuntime,
    config: OperationsConfig,
    keys: Arc<Secrets>,
    replay: Arc<Mutex<Connection>>,
    events: EventBus,
}
/// Does not listen or spawn work. Loads dedicated env credentials once.
pub fn build_router(runtime: SharedRuntime, config: OperationsConfig) -> Result<Router> {
    config.validate()?;
    let keys = Secrets::from_env(&config)?;
    build_with_keys(runtime, config, keys)
}
/// Attach an independently owned bounded subscription bus to webhook admission.
pub fn build_router_with_events(
    runtime: SharedRuntime,
    config: OperationsConfig,
    events: EventBus,
) -> Result<Router> {
    config.validate()?;
    let keys = Secrets::from_env(&config)?;
    build_with_events(runtime, config, keys, events)
}
fn build_with_keys(
    runtime: SharedRuntime,
    config: OperationsConfig,
    keys: Secrets,
) -> Result<Router> {
    build_with_events(runtime, config, keys, EventBus::new(256)?)
}
fn build_with_events(
    runtime: SharedRuntime,
    config: OperationsConfig,
    keys: Secrets,
    events: EventBus,
) -> Result<Router> {
    config.validate()?;
    let replay = open_replay(&config.replay_db)?;
    let state = App {
        runtime,
        config,
        keys: Arc::new(keys),
        replay: Arc::new(Mutex::new(replay)),
        events,
    };
    Ok(Router::new()
        .route("/webhook", post(webhook))
        .route("/diagnostics", get(diagnostics_handler))
        .route("/metrics", get(metrics_handler))
        .layer(DefaultBodyLimit::max(MAX_WEBHOOK_BODY))
        .with_state(state))
}
pub async fn serve(runtime: SharedRuntime, config: OperationsConfig) -> Result<()> {
    config.validate()?;
    let address = config.bind;
    let router = build_router(runtime, config)?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
fn open_replay(path: &std::path::Path) -> Result<Connection> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    ensure!(
        path != std::path::Path::new(":memory:"),
        "durable replay database required"
    );
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    let parent_meta = std::fs::metadata(parent)?;
    ensure!(
        parent_meta.is_dir()
            && parent_meta.mode() & 0o022 == 0
            && parent_meta.uid() == unsafe { libc::geteuid() },
        "replay parent directory must be owned and not writable by others"
    );
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .context("open private replay database")?;
    let meta = file.metadata()?;
    ensure!(
        meta.is_file()
            && meta.mode() & 0o077 == 0
            && meta.nlink() == 1
            && meta.uid() == unsafe { libc::geteuid() },
        "replay database must be private, owned, regular and not hard-linked"
    );
    let conn = Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE | rusqlite::OpenFlags::SQLITE_OPEN_NOFOLLOW,
    )?;
    conn.busy_timeout(Duration::from_secs(5))?;
    let app_id: i64 = conn.pragma_query_value(None, "application_id", |r| r.get(0))?;
    ensure!(
        app_id == 0 || app_id == 0x4d53594f,
        "replay path belongs to another application"
    );
    if app_id == 0 {
        let tables: usize = conn.query_row(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            [],
            |r| r.get(0),
        )?;
        ensure!(
            tables == 0,
            "refusing to repurpose nonempty database as replay cache"
        );
    }
    conn.execute_batch("PRAGMA application_id=1297307983; PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA trusted_schema=OFF; CREATE TABLE IF NOT EXISTS webhook_replay(scope_hash TEXT NOT NULL, event_hash TEXT NOT NULL, body_hash TEXT NOT NULL, request_id TEXT NOT NULL, turn_id TEXT, created_at INTEGER NOT NULL, PRIMARY KEY(scope_hash,event_hash));")?;
    Ok(conn)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WebhookRequest {
    event_id: String,
    prompt: String,
}
fn header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    let mut values = headers.get_all(name).iter();
    let first = values.next()?.to_str().ok()?;
    if values.next().is_some() {
        return None;
    }
    Some(first)
}
fn authenticate(headers: &HeaderMap, body: &[u8], key: &[u8], skew: u64) -> bool {
    let Some(timestamp) = header(headers, "x-missy-timestamp") else {
        return false;
    };
    if timestamp.is_empty()
        || timestamp.len() > 12
        || !timestamp.bytes().all(|b| b.is_ascii_digit())
    {
        return false;
    }
    let Ok(seconds) = timestamp.parse::<i64>() else {
        return false;
    };
    if chrono::Utc::now().timestamp().abs_diff(seconds) > skew {
        return false;
    }
    let Some(signature) = header(headers, "x-missy-signature") else {
        return false;
    };
    if signature.len() != 64 {
        return false;
    }
    let Ok(signature) = hex::decode(signature) else {
        return false;
    };
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(timestamp.as_bytes());
    mac.update(b".");
    mac.update(body);
    mac.verify_slice(&signature).is_ok()
}
fn read_authorized(headers: &HeaderMap, key: &[u8]) -> bool {
    header(headers, "authorization")
        .and_then(|s| s.strip_prefix("Bearer "))
        .is_some_and(|s| bool::from(s.as_bytes().ct_eq(key)))
}
fn failure(status: StatusCode, code: &'static str) -> Response {
    (status, Json(serde_json::json!({"error":code}))).into_response()
}
async fn webhook(State(app): State<App>, headers: HeaderMap, body: Bytes) -> Response {
    if !authenticate(
        &headers,
        &body,
        &app.keys.webhook,
        app.config.max_clock_skew_secs,
    ) {
        app.events.publish(OperationalKind::WebhookRejected);
        return failure(StatusCode::UNAUTHORIZED, "authentication_failed");
    }
    if header(&headers, "content-type")
        .is_none_or(|v| v.split(';').next() != Some("application/json"))
    {
        return failure(StatusCode::UNSUPPORTED_MEDIA_TYPE, "json_required");
    }
    let Ok(request) = serde_json::from_slice::<WebhookRequest>(&body) else {
        return failure(StatusCode::BAD_REQUEST, "invalid_request");
    };
    if request.event_id.is_empty()
        || request.event_id.len() > 128
        || !request
            .event_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b))
        || request.prompt.trim().is_empty()
        || request.prompt.len() > 64_000
    {
        return failure(StatusCode::BAD_REQUEST, "invalid_request");
    }
    let hash = |data: &[u8]| hex::encode(Sha256::digest(data));
    let scope_hash = hash(&serde_json::to_vec(&app.config.scope).expect("serializable scope"));
    let event_hash = hash(request.event_id.as_bytes());
    let body_hash = hash(&body);
    // Claim first, then Runtime scoped idempotency. Cross-database atomicity is
    // impossible here: a crash before receipt persistence is explicitly uncertain
    // and fails closed on retry, never silently dispatching effects a second time.
    let request_id = format!("webhook:{event_hash}");
    let claim = (|| -> Result<(bool, Option<String>)> {
        let mut conn = app
            .replay
            .lock()
            .map_err(|_| anyhow::anyhow!("replay lock poisoned"))?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let previous: Option<(String,Option<String>)> = tx.query_row("SELECT body_hash,turn_id FROM webhook_replay WHERE scope_hash=?1 AND event_hash=?2", params![scope_hash, event_hash], |r| Ok((r.get(0)?,r.get(1)?))).optional()?;
        if let Some((previous, turn_id)) = previous {
            ensure!(previous == body_hash, "replay_conflict");
            tx.commit()?;
            return Ok((false, turn_id));
        }
        let count: usize = tx.query_row("SELECT count(*) FROM webhook_replay", [], |r| r.get(0))?;
        ensure!(count < app.config.max_replay_entries, "replay_capacity");
        tx.execute("INSERT INTO webhook_replay(scope_hash,event_hash,body_hash,request_id,created_at) VALUES(?1,?2,?3,?4,?5)", params![scope_hash, event_hash, body_hash, request_id, chrono::Utc::now().timestamp()])?;
        tx.commit()?;
        Ok((true, None))
    })();
    let (created, prior_turn) = match claim {
        Ok(v) => v,
        Err(e) if e.to_string() == "replay_conflict" => {
            return failure(StatusCode::CONFLICT, "event_payload_conflict");
        }
        Err(_) => return failure(StatusCode::SERVICE_UNAVAILABLE, "replay_unavailable"),
    };
    // Runtime retention must not turn an old delivery into a fresh execution.
    if !created && prior_turn.is_none() {
        return failure(
            StatusCode::CONFLICT,
            "admission_receipt_uncertain_requires_reconciliation",
        );
    }
    if let Some(id) = prior_turn
        && app.runtime.status(&app.config.scope, &id).is_err()
    {
        return failure(StatusCode::CONFLICT, "original_turn_no_longer_available");
    }
    let result = match app
        .runtime
        .start(app.config.scope.clone(), request.prompt, request_id)
        .await
    {
        Ok(result) => result,
        Err(_) => return failure(StatusCode::SERVICE_UNAVAILABLE, "runtime_unavailable"),
    };
    let receipt_saved = app.replay.lock().ok().is_some_and(|conn| {
        conn.execute(
            "UPDATE webhook_replay SET turn_id=?1 WHERE scope_hash=?2 AND event_hash=?3",
            params![result.turn_id, scope_hash, event_hash],
        )
        .is_ok()
    });
    if !receipt_saved {
        return failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "admitted_but_receipt_uncertain_requires_reconciliation",
        );
    }
    app.events.publish(OperationalKind::WebhookAccepted);
    (if created { StatusCode::ACCEPTED } else { StatusCode::OK }, Json(serde_json::json!({"turn_id":result.turn_id,"status":result.status,"deduplicated":!created || result.deduplicated}))).into_response()
}
async fn diagnostics_handler(State(app): State<App>, headers: HeaderMap) -> Response {
    if !read_authorized(&headers, &app.keys.read) {
        return failure(StatusCode::UNAUTHORIZED, "authentication_failed");
    }
    match diagnostics::snapshot(app.runtime.store(), &app.config.scope) {
        Ok(report) => Json(report).into_response(),
        Err(_) => failure(StatusCode::SERVICE_UNAVAILABLE, "store_unavailable"),
    }
}
async fn metrics_handler(State(app): State<App>, headers: HeaderMap) -> Response {
    if !read_authorized(&headers, &app.keys.read) {
        return failure(StatusCode::UNAUTHORIZED, "authentication_failed");
    }
    match diagnostics::snapshot(app.runtime.store(), &app.config.scope) {
        Ok(report) => (
            [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
            report.prometheus(),
        )
            .into_response(),
        Err(_) => failure(StatusCode::SERVICE_UNAVAILABLE, "store_unavailable"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use tower::ServiceExt;
    fn keys() -> Secrets {
        Secrets {
            webhook: vec![b'w'; 32],
            read: vec![b'r'; 32],
        }
    }
    fn config(dir: &std::path::Path) -> OperationsConfig {
        OperationsConfig {
            replay_db: dir.join("replay.sqlite3"),
            scope: crate::tests::scope(),
            ..Default::default()
        }
    }
    fn signed(body: Vec<u8>, timestamp: &str) -> Request<Body> {
        let mut mac = Hmac::<Sha256>::new_from_slice(&keys().webhook).unwrap();
        mac.update(timestamp.as_bytes());
        mac.update(b".");
        mac.update(&body);
        Request::builder()
            .method("POST")
            .uri("/webhook")
            .header("content-type", "application/json")
            .header("x-missy-timestamp", timestamp)
            .header(
                "x-missy-signature",
                hex::encode(mac.finalize().into_bytes()),
            )
            .body(Body::from(body))
            .unwrap()
    }
    async fn send(router: &Router, value: serde_json::Value) -> Response {
        router
            .clone()
            .oneshot(signed(
                serde_json::to_vec(&value).unwrap(),
                &chrono::Utc::now().timestamp().to_string(),
            ))
            .await
            .unwrap()
    }
    #[tokio::test]
    async fn authenticated_real_turn_scoped_and_durable_replay_conflict() {
        let dir = tempfile::tempdir().unwrap();
        let runtime = crate::tests::runtime(dir.path());
        let config = config(dir.path());
        let router = build_with_keys(runtime.clone(), config.clone(), keys()).unwrap();
        let value = serde_json::json!({"event_id":"evt1","prompt":"test private prompt"});
        let response = send(&router, value.clone()).await;
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let first: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
        let turn = runtime
            .status(&config.scope, first["turn_id"].as_str().unwrap())
            .unwrap();
        assert_eq!(turn.prompt, "test private prompt");
        drop(router);
        let router = build_with_keys(runtime.clone(), config.clone(), keys()).unwrap();
        let response = send(&router, value.clone()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let replay: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
        assert_eq!(first["turn_id"], replay["turn_id"]);
        assert_eq!(
            send(
                &router,
                serde_json::json!({"event_id":"evt1","prompt":"changed"})
            )
            .await
            .status(),
            StatusCode::CONFLICT
        );
        let mut other = config.scope;
        other.user = "other".into();
        assert!(
            runtime
                .status(&other, first["turn_id"].as_str().unwrap())
                .is_err()
        );
        assert_eq!(
            runtime
                .store()
                .list_turns(&crate::tests::scope(), 100)
                .unwrap()
                .len(),
            1
        );
    }
    #[tokio::test]
    async fn rejects_principal_tools_bad_auth_timestamp_and_big_body() {
        let dir = tempfile::tempdir().unwrap();
        let runtime = crate::tests::runtime(dir.path());
        let router = build_with_keys(runtime.clone(), config(dir.path()), keys()).unwrap();
        for field in ["scope", "principal", "tools", "request_id", "runtime"] {
            let mut value = serde_json::json!({"event_id":"e","prompt":"prompt"});
            value[field] = serde_json::json!({"user":"admin"});
            assert_eq!(send(&router, value).await.status(), StatusCode::BAD_REQUEST);
        }
        for timestamp in ["0", "-1", "999999999999", "1e9", "invalid"] {
            assert_eq!(
                router
                    .clone()
                    .oneshot(signed(
                        br#"{"event_id":"e","prompt":"prompt"}"#.to_vec(),
                        timestamp
                    ))
                    .await
                    .unwrap()
                    .status(),
                StatusCode::UNAUTHORIZED
            );
        }
        let mut request = signed(
            br#"{"event_id":"e","prompt":"prompt"}"#.to_vec(),
            &chrono::Utc::now().timestamp().to_string(),
        );
        request
            .headers_mut()
            .insert("x-missy-signature", "00".repeat(32).parse().unwrap());
        assert_eq!(
            router.clone().oneshot(request).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
        let response = router
            .clone()
            .oneshot(signed(
                vec![b'a'; MAX_WEBHOOK_BODY + 1],
                &chrono::Utc::now().timestamp().to_string(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert!(
            runtime
                .store()
                .list_turns(&crate::tests::scope(), 100)
                .unwrap()
                .is_empty()
        );
    }
    #[tokio::test]
    async fn read_endpoints_require_separate_key_and_never_return_content() {
        let dir = tempfile::tempdir().unwrap();
        let runtime = crate::tests::runtime(dir.path());
        runtime
            .run(
                crate::tests::scope(),
                "SECRET_PROMPT".into(),
                "read-test".into(),
            )
            .await
            .unwrap();
        let router = build_with_keys(runtime, config(dir.path()), keys()).unwrap();
        for path in ["/metrics", "/diagnostics"] {
            let unauth = Request::builder().uri(path).body(Body::empty()).unwrap();
            assert_eq!(
                router.clone().oneshot(unauth).await.unwrap().status(),
                StatusCode::UNAUTHORIZED
            );
            let wrong = Request::builder()
                .uri(path)
                .header("authorization", format!("Bearer {}", "w".repeat(32)))
                .body(Body::empty())
                .unwrap();
            assert_eq!(
                router.clone().oneshot(wrong).await.unwrap().status(),
                StatusCode::UNAUTHORIZED
            );
            let request = Request::builder()
                .uri(path)
                .header("authorization", format!("Bearer {}", "r".repeat(32)))
                .body(Body::empty())
                .unwrap();
            let response = router.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let text = String::from_utf8(
                to_bytes(response.into_body(), 65_536)
                    .await
                    .unwrap()
                    .to_vec(),
            )
            .unwrap();
            assert!(!text.contains("SECRET_PROMPT"));
            assert!(!text.contains("private model response"));
        }
    }
    #[test]
    fn config_refuses_nonloopback_generic_or_shared_keys_and_insecure_db() {
        let mut cfg = OperationsConfig {
            bind: "0.0.0.0:1234".parse().unwrap(),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
        cfg.bind = "127.0.0.1:0".parse().unwrap();
        cfg.webhook_key_env = "API_KEY".into();
        assert!(cfg.validate().is_err());
        cfg.webhook_key_env = cfg.read_key_env.clone();
        assert!(cfg.validate().is_err());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        std::fs::write(&path, []).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(open_replay(&path).is_err());
    }
    #[tokio::test]
    async fn durable_cache_capacity_fails_closed_but_allows_existing_retries() {
        let dir = tempfile::tempdir().unwrap();
        let runtime = crate::tests::runtime(dir.path());
        let mut cfg = config(dir.path());
        cfg.max_replay_entries = 1;
        let router = build_with_keys(runtime, cfg, keys()).unwrap();
        let value = serde_json::json!({"event_id":"e1","prompt":"one"});
        assert_eq!(
            send(&router, value.clone()).await.status(),
            StatusCode::ACCEPTED
        );
        assert_eq!(
            send(&router, serde_json::json!({"event_id":"e2","prompt":"two"}))
                .await
                .status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(send(&router, value).await.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn uncertain_claim_never_dispatches_new_turn() {
        let dir = tempfile::tempdir().unwrap();
        let runtime = crate::tests::runtime(dir.path());
        let cfg = config(dir.path());
        let value = serde_json::json!({"event_id":"uncertain","prompt":"prompt"});
        let hash = |bytes: &[u8]| hex::encode(Sha256::digest(bytes));
        let conn = open_replay(&cfg.replay_db).unwrap();
        conn.execute("INSERT INTO webhook_replay(scope_hash,event_hash,body_hash,request_id,created_at) VALUES(?1,?2,?3,?4,0)", params![hash(&serde_json::to_vec(&cfg.scope).unwrap()),hash(b"uncertain"),hash(&serde_json::to_vec(&value).unwrap()),format!("webhook:{}",hash(b"uncertain"))]).unwrap();
        drop(conn);
        let router = build_with_keys(runtime.clone(), cfg.clone(), keys()).unwrap();
        assert_eq!(send(&router, value).await.status(), StatusCode::CONFLICT);
        assert!(
            runtime
                .store()
                .list_turns(&cfg.scope, 100)
                .unwrap()
                .is_empty()
        );
    }
}
