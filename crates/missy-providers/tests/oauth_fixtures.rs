//! Native OAuth against isolated TCP fixtures and fresh OS processes only.
use axum::{
    Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::post,
};
use missy_core::{Message, OAuthConfig, PolicyConfig, Provider, ProviderConfig};
use missy_policy::PolicyHttpClient;
use missy_providers::{CredentialResolver, HttpProvider};
use serde_json::json;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio_util::sync::CancellationToken;

struct Credentials;
impl CredentialResolver for Credentials {
    fn resolve(&self, reference: &str) -> anyhow::Result<String> {
        anyhow::ensure!(reference == "FIXTURE_REFRESH", "unexpected reference");
        Ok("fixture-refresh-zero".into())
    }
}
#[derive(Clone, Default)]
struct Data {
    refreshes: Arc<Mutex<Vec<String>>>,
    accesses: Arc<Mutex<Vec<String>>>,
    reject: Arc<AtomicBool>,
    delay: Arc<AtomicBool>,
    malformed: Arc<AtomicBool>,
}
struct Fixture {
    origin: String,
    data: Data,
    task: tokio::task::JoinHandle<()>,
    root: tempfile::TempDir,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Fixture {
    async fn start() -> Self {
        async fn token(State(data): State<Data>, body: Bytes) -> impl IntoResponse {
            let pairs: Vec<_> = url::form_urlencoded::parse(&body).into_owned().collect();
            assert!(pairs.contains(&("grant_type".into(), "refresh_token".into())));
            assert!(pairs.contains(&("client_id".into(), "fixture-client".into())));
            let refresh = pairs
                .iter()
                .find(|(k, _)| k == "refresh_token")
                .unwrap()
                .1
                .clone();
            let generation = {
                let mut seen = data.refreshes.lock().unwrap();
                seen.push(refresh);
                seen.len()
            };
            if data.delay.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_secs(3)).await;
            }
            let body = if data.malformed.load(Ordering::SeqCst) {
                "fixture-access-secret malformed token response".to_owned()
            } else {
                json!({"token_type":"Bearer", "access_token":format!("fixture-access-{generation}"),
                    "refresh_token":format!("fixture-refresh-{generation}"), "expires_in":3600})
                .to_string()
            };
            (StatusCode::OK, body)
        }
        async fn inference(State(data): State<Data>, headers: HeaderMap) -> impl IntoResponse {
            data.accesses
                .lock()
                .unwrap()
                .push(headers["authorization"].to_str().unwrap().to_owned());
            if data.reject.swap(false, Ordering::SeqCst) {
                return (
                    StatusCode::UNAUTHORIZED,
                    "fixture-access-secret rejected".to_owned(),
                );
            }
            (StatusCode::OK, json!({"choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":1,"completion_tokens":1}}).to_string())
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let data = Data::default();
        let app = Router::new()
            .route("/token", post(token))
            .route("/v1/chat/completions", post(inference))
            .with_state(data.clone());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            origin,
            data,
            task,
            root: tempfile::tempdir().unwrap(),
        }
    }
    fn config(&self) -> ProviderConfig {
        ProviderConfig {
            name: "fixture".into(),
            kind: "openai".into(),
            model: "fixture".into(),
            base_url: format!("{}/v1", self.origin),
            timeout_seconds: 5,
            oauth: Some(OAuthConfig {
                token_vault: self.root.path().join("vault"),
                token_url: format!("{}/token", self.origin),
                client_id: "fixture-client".into(),
                refresh_token_env: "FIXTURE_REFRESH".into(),
                access_token_env: None,
                scopes: vec!["inference".into()],
                timeout_seconds: 1,
            }),
            ..Default::default()
        }
    }
    fn policy(&self) -> PolicyConfig {
        PolicyConfig {
            allowed_origins: vec![self.origin.clone()],
            allow_private_network: true,
            ..Default::default()
        }
    }
    fn provider(&self) -> HttpProvider {
        HttpProvider::with_credentials(
            self.config(),
            PolicyHttpClient::new(self.policy()).unwrap(),
            Arc::new(Credentials),
        )
        .unwrap()
    }
}
async fn complete(p: &HttpProvider, cancellation: CancellationToken) -> anyhow::Result<()> {
    p.complete(&[Message::text("user", "fixture")], &[], cancellation)
        .await?;
    Ok(())
}

// Only invoked by the parent fixture with explicit serialized fixture configs.
#[tokio::test]
#[ignore]
async fn oauth_child_process() {
    let config: ProviderConfig =
        serde_json::from_str(&std::env::var("MISSY_FIXTURE_PROFILE").unwrap()).unwrap();
    let policy: PolicyConfig =
        serde_json::from_str(&std::env::var("MISSY_FIXTURE_POLICY").unwrap()).unwrap();
    let p = HttpProvider::with_credentials(
        config,
        PolicyHttpClient::new(policy).unwrap(),
        Arc::new(Credentials),
    )
    .unwrap();
    let result = complete(&p, CancellationToken::new()).await;
    if std::env::var("MISSY_FIXTURE_REJECT").is_ok() {
        let error = format!("{:#}", result.unwrap_err());
        assert!(!error.contains("fixture-access") && !error.contains("fixture-refresh"));
    } else {
        result.unwrap();
    }
}
async fn child(config: ProviderConfig, policy: PolicyConfig, reject: bool) {
    let output = tokio::task::spawn_blocking(move || {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", "oauth_child_process", "--ignored", "--nocapture"])
            .env(
                "MISSY_FIXTURE_PROFILE",
                serde_json::to_string(&config).unwrap(),
            )
            .env(
                "MISSY_FIXTURE_POLICY",
                serde_json::to_string(&policy).unwrap(),
            );
        if reject {
            cmd.env("MISSY_FIXTURE_REJECT", "1");
        } else {
            cmd.env_remove("MISSY_FIXTURE_REJECT");
        }
        cmd.output().unwrap()
    })
    .await
    .unwrap();
    assert!(
        output.status.success(),
        "fixture child failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cross_process_rotation_restart_and_401_invalidation() {
    let f = Fixture::start().await;
    // Concurrent independent processes must serialize the refresh and share it.
    tokio::join!(
        child(f.config(), f.policy(), false),
        child(f.config(), f.policy(), false)
    );
    assert_eq!(
        *f.data.refreshes.lock().unwrap(),
        vec!["fixture-refresh-zero"]
    );
    f.data.reject.store(true, Ordering::SeqCst);
    child(f.config(), f.policy(), true).await;
    // 401 does not replay inference. A fresh process uses the rotated refresh.
    assert_eq!(f.data.accesses.lock().unwrap().len(), 3);
    child(f.config(), f.policy(), false).await;
    child(f.config(), f.policy(), false).await;
    assert_eq!(
        *f.data.refreshes.lock().unwrap(),
        vec!["fixture-refresh-zero", "fixture-refresh-1"]
    );
    let seen = f.data.accesses.lock().unwrap();
    assert_eq!(
        &seen[3..],
        &["Bearer fixture-access-2", "Bearer fixture-access-2"]
    );
    drop(seen);
    // Ciphertext never contains access or refresh plaintext.
    let bytes = std::fs::read(f.root.path().join("vault/vault.json")).unwrap();
    let encoded = String::from_utf8(bytes).unwrap();
    assert!(!encoded.contains("fixture-access") && !encoded.contains("fixture-refresh"));
}

#[tokio::test]
async fn refresh_timeout_cancellation_policy_and_error_redaction() {
    let f = Fixture::start().await;
    f.data.delay.store(true, Ordering::SeqCst);
    let p = f.provider();
    let cancellation = CancellationToken::new();
    let cancel = cancellation.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel();
    });
    let error = complete(&p, cancellation).await.unwrap_err().to_string();
    assert!(error.contains("cancelled"));
    let before = std::time::Instant::now();
    let error = p
        .refresh_oauth(CancellationToken::new())
        .await
        .unwrap_err()
        .to_string();
    assert!(before.elapsed() < Duration::from_secs(2));
    assert!(!error.contains("fixture-refresh"));
    f.data.delay.store(false, Ordering::SeqCst);
    f.data.malformed.store(true, Ordering::SeqCst);
    let error = p
        .refresh_oauth(CancellationToken::new())
        .await
        .unwrap_err()
        .to_string();
    assert!(!error.contains("fixture-access"));
    f.data.malformed.store(false, Ordering::SeqCst);
    let blocked = HttpProvider::with_credentials(
        f.config(),
        PolicyHttpClient::new(PolicyConfig::default()).unwrap(),
        Arc::new(Credentials),
    )
    .unwrap();
    let count = f.data.refreshes.lock().unwrap().len();
    assert!(
        blocked
            .refresh_oauth(CancellationToken::new())
            .await
            .is_err()
    );
    assert_eq!(f.data.refreshes.lock().unwrap().len(), count);
    assert!(f.data.accesses.lock().unwrap().is_empty());
}

#[tokio::test]
async fn vault_contention_is_cancelable_and_does_not_leave_blocked_workers() {
    let f = Fixture::start().await;
    let vault = missy_security::Vault::open(&f.config().oauth.unwrap().token_vault).unwrap();
    let p = f.provider();
    let cancellation = CancellationToken::new();
    let cancel = cancellation.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(60)).await;
        cancel.cancel();
    });
    let error = complete(&p, cancellation).await.unwrap_err().to_string();
    assert!(error.contains("cancelled"));
    assert!(f.data.refreshes.lock().unwrap().is_empty());
    drop(vault);
    complete(&p, CancellationToken::new()).await.unwrap();
}
