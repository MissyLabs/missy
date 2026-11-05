use super::*;
use async_trait::async_trait;
use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use missy_core::{
    Completion, Message, Provider, RuntimeConfig, ToolCall, ToolContext, ToolDefinition,
    ToolExecutor, Usage,
};
use missy_store::Store;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;
const TOKEN: &str = "ad83f926c07b1542e6d9a30f8b4715ce";
struct Mock;
#[async_trait]
impl Provider for Mock {
    fn name(&self) -> &str {
        "test"
    }
    async fn complete(
        &self,
        _: &[Message],
        _: &[ToolDefinition],
        _: CancellationToken,
    ) -> Result<Completion> {
        Ok(Completion {
            message: Message::text("assistant", "mock <script>not executable</script>"),
            usage: Usage::default(),
            finish_reason: "stop".into(),
        })
    }
}
#[async_trait]
impl ToolExecutor for Mock {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![]
    }
    async fn execute(&self, _: &ToolContext, _: &ToolCall) -> Result<Value> {
        bail!("no tools")
    }
}
fn setup(rate: u32) -> (tempfile::TempDir, ApiState) {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(Store::open(dir.path().join("state.db")).unwrap());
    let runtime = Arc::new(
        Runtime::new(
            store,
            Arc::new(Mock),
            Arc::new(Mock),
            "test system".into(),
            RuntimeConfig::default(),
        )
        .unwrap(),
    );
    let cfg = ApiConfig {
        requests_per_minute: rate,
        ..ApiConfig::default()
    };
    let state = ApiState::with_token(runtime, cfg, TOKEN).unwrap();
    (dir, state)
}
fn req(method: &str, path: &str, body: Option<Value>) -> Request<Body> {
    let mut r = Request::builder()
        .method(method)
        .uri(path)
        .header("host", "127.0.0.1:8766")
        .header("authorization", format!("Bearer {TOKEN}"));
    if body.is_some() {
        r = r.header("content-type", "application/json");
    }
    r.body(Body::from(body.map(|v| v.to_string()).unwrap_or_default()))
        .unwrap()
}
async fn value(r: Response) -> Value {
    serde_json::from_slice(&to_bytes(r.into_body(), 1024 * 1024).await.unwrap()).unwrap()
}
#[tokio::test]
async fn authentication_configuration_and_console() {
    let (_dir, s) = setup(60);
    let mut r = req("GET", "/api/v1/health", None);
    r.headers_mut().remove(header::AUTHORIZATION);
    assert_eq!(
        router(s.clone()).oneshot(r).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );
    let mut r = req("GET", "/api/v1/health", None);
    r.headers_mut().insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer invalid"),
    );
    assert_eq!(
        router(s.clone()).oneshot(r).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );
    let mut r = req("GET", "/api/v1/health", None);
    r.headers_mut().append(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer duplicate"),
    );
    assert_eq!(
        router(s.clone()).oneshot(r).await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );
    let cfg = ApiConfig {
        bind: "0.0.0.0:8766".into(),
        ..ApiConfig::default()
    };
    assert!(ApiState::with_token(s.0.runtime.clone(), cfg, TOKEN).is_err());
    assert!(
        ApiState::with_token(
            s.0.runtime.clone(),
            ApiConfig::default(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
        .is_err()
    );
    let response = router(s.clone())
        .oneshot(req("GET", "/", None))
        .await
        .unwrap();
    assert!(
        response.headers()[header::CONTENT_SECURITY_POLICY]
            .to_str()
            .unwrap()
            .contains("frame-ancestors 'none'")
    );
    assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
    for bad in ["innerHTML", "localStorage", "sessionStorage"] {
        assert!(!include_str!("console.js").contains(bad));
    }
    let mut r = req("GET", "/", None);
    r.headers_mut()
        .insert(header::HOST, HeaderValue::from_static("attacker.test:8766"));
    assert_eq!(
        router(s.clone()).oneshot(r).await.unwrap().status(),
        StatusCode::BAD_REQUEST
    );
    let mut r = req("GET", "/api/v1/health", None);
    r.headers_mut().insert(
        header::ORIGIN,
        HeaderValue::from_static("https://attacker.test"),
    );
    assert_eq!(
        router(s).oneshot(r).await.unwrap().status(),
        StatusCode::FORBIDDEN
    );
}
#[tokio::test]
async fn chat_dedup_and_scope() {
    let (_dir, s) = setup(60);
    let app = router(s.clone());
    let input = json!({"session":"mine","prompt":"hello","request_id":"stable-id"});
    let r = app
        .clone()
        .oneshot(req("POST", "/api/v1/chat", Some(input.clone())))
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::ACCEPTED);
    let result = value(r).await;
    let turn = result["turn_id"].as_str().unwrap();
    let scope = s.scope("mine".into()).unwrap();
    for _ in 0..100 {
        if s.0.runtime.status(&scope, turn).unwrap().status == missy_store::TurnStatus::Completed {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        s.0.runtime.status(&scope, turn).unwrap().status,
        missy_store::TurnStatus::Completed
    );
    let repeat = value(
        app.clone()
            .oneshot(req("POST", "/api/v1/chat", Some(input)))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(result["turn_id"], repeat["turn_id"]);
    let other_cfg = ApiConfig {
        user: "different-user".into(),
        ..ApiConfig::default()
    };
    let other = router(ApiState::with_token(s.0.runtime.clone(), other_cfg, TOKEN).unwrap());
    assert_eq!(
        other
            .oneshot(req(
                "GET",
                &format!("/api/v1/turns/{turn}?session=mine"),
                None
            ))
            .await
            .unwrap()
            .status(),
        StatusCode::NOT_FOUND
    );
    let r = app
        .clone()
        .oneshot(req(
            "POST",
            &format!("/api/v1/turns/{turn}/cancel"),
            Some(json!({"session":"mine"})),
        ))
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::CONFLICT);
    assert_eq!(
        app.clone()
            .oneshot(req(
                "GET",
                &format!("/api/v1/turns/{turn}?session=foreign"),
                None
            ))
            .await
            .unwrap()
            .status(),
        StatusCode::NOT_FOUND
    );
    for action in ["cancel", "resume"] {
        assert_eq!(
            app.clone()
                .oneshot(req(
                    "POST",
                    &format!("/api/v1/turns/{turn}/{action}"),
                    Some(json!({"session":"foreign"}))
                ))
                .await
                .unwrap()
                .status(),
            StatusCode::NOT_FOUND
        );
    }
    assert_eq!(
        value(
            app.clone()
                .oneshot(req("GET", "/api/v1/sessions/foreign/history", None))
                .await
                .unwrap()
        )
        .await["messages"],
        json!([])
    );
    let r = app
        .clone()
        .oneshot(req(
            "POST",
            "/api/v1/chat",
            Some(json!({"session":"a","prompt":"x","request_id":"1","tenant":"evil"})),
        ))
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        app.oneshot(req(
            "POST",
            "/api/v1/chat",
            Some(json!({"session":"../x","prompt":"x","request_id":"2"}))
        ))
        .await
        .unwrap()
        .status(),
        StatusCode::BAD_REQUEST
    );
}
#[tokio::test]
async fn rate_body_and_concurrency_bounds() {
    let (_dir, s) = setup(1);
    let app = router(s);
    assert_eq!(
        app.clone()
            .oneshot(req("GET", "/api/v1/health", None))
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        app.oneshot(req("GET", "/api/v1/health", None))
            .await
            .unwrap()
            .status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    let (_dir, s) = setup(60);
    assert_eq!(
        router(s.clone())
            .oneshot(req(
                "POST",
                "/api/v1/chat",
                Some(json!({"session":"a","prompt":"x".repeat(MAX_BODY_BYTES),"request_id":"big"}))
            ))
            .await
            .unwrap()
            .status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
    let _permits =
        s.0.runs
            .clone()
            .acquire_many_owned(MAX_RUNS as u32)
            .await
            .unwrap();
    assert_eq!(
        router(s)
            .oneshot(req(
                "POST",
                "/api/v1/chat",
                Some(json!({"session":"a","prompt":"hello","request_id":"blocked"}))
            ))
            .await
            .unwrap()
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
}

struct Waiting;
#[async_trait]
impl Provider for Waiting {
    fn name(&self) -> &str {
        "waiting"
    }
    async fn complete(
        &self,
        _: &[Message],
        _: &[ToolDefinition],
        token: CancellationToken,
    ) -> Result<Completion> {
        token.cancelled().await;
        bail!("cancelled")
    }
}
#[tokio::test]
async fn admission_cancel_and_status_are_truthful() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = Arc::new(
        Runtime::new(
            Arc::new(Store::open(dir.path().join("state.db")).unwrap()),
            Arc::new(Waiting),
            Arc::new(Mock),
            "test".into(),
            RuntimeConfig::default(),
        )
        .unwrap(),
    );
    let state = ApiState::with_token(runtime, ApiConfig::default(), TOKEN).unwrap();
    let app = router(state.clone());
    let response = app
        .clone()
        .oneshot(req(
            "POST",
            "/api/v1/chat",
            Some(json!({"session":"test","prompt":"wait","request_id":"cancel-me"})),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let response = value(response).await;
    let id = response["turn_id"].as_str().unwrap();
    let response = app
        .clone()
        .oneshot(req(
            "POST",
            &format!("/api/v1/turns/{id}/cancel"),
            Some(json!({"session":"test"})),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    assert_eq!(value(response).await["cancellation_requested"], true);
    let scope = state.scope("test".into()).unwrap();
    for _ in 0..100 {
        if state.0.runtime.status(&scope, id).unwrap().status == missy_store::TurnStatus::Cancelled
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let response = app
        .clone()
        .oneshot(req(
            "GET",
            &format!("/api/v1/turns/{id}?session=test"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(value(response).await["turn"]["status"], "cancelled");
    assert_eq!(
        app.oneshot(req(
            "POST",
            &format!("/api/v1/turns/{id}/resume"),
            Some(json!({"session":"test"}))
        ))
        .await
        .unwrap()
        .status(),
        StatusCode::CONFLICT
    );
}
