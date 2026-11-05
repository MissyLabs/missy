//! Loopback-only authenticated HTTP transport. No transport-provided scope escalation.
use anyhow::{Context, Result, bail};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, Query, Request, State},
    http::{HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use missy_core::{ApiConfig, Scope};
use missy_runtime::Runtime;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use subtle::ConstantTimeEq;
use tokio::sync::Semaphore;
mod discord;
pub use discord::{DiscordProbe, probe_discord, serve_discord, serve_discord_with_state_dir};

pub const MAX_BODY_BYTES: usize = 64 * 1024;
pub const MAX_PROMPT_BYTES: usize = 32 * 1024;
const MAX_RUNS: usize = 8;

/// Does not implement Debug: authentication material never enters diagnostics.
#[derive(Clone)]
pub struct ApiState(Arc<Inner>);
struct Inner {
    runtime: Arc<Runtime>,
    config: ApiConfig,
    bind: SocketAddr,
    token_hash: [u8; 32],
    rate: Mutex<RateWindow>,
    requests: Semaphore,
    runs: Arc<Semaphore>,
}
struct RateWindow {
    started: Instant,
    count: u32,
}

impl ApiState {
    /// Read the environment once. There is no development/anonymous auth mode.
    pub fn new(runtime: Arc<Runtime>, config: ApiConfig) -> Result<Self> {
        let token = std::env::var(&config.token_env)
            .context("API bearer token environment variable is missing")?;
        Self::with_token(runtime, config, &token)
    }
    fn with_token(runtime: Arc<Runtime>, config: ApiConfig, token: &str) -> Result<Self> {
        let bind: SocketAddr = config
            .bind
            .parse()
            .context("API bind must be a numeric loopback IP:port")?;
        if !bind.ip().is_loopback() {
            bail!("remote API binding is unsupported; use an authenticated loopback tunnel");
        }
        if config.requests_per_minute == 0 || config.requests_per_minute > 100_000 {
            bail!("invalid API rate limit");
        }
        Scope {
            tenant: config.tenant.clone(),
            user: config.user.clone(),
            channel: "api".into(),
            session: "validation".into(),
        }
        .validate()?;
        let distinct: HashSet<_> = token.bytes().collect();
        if !(32..=256).contains(&token.len())
            || !token.bytes().all(|c| c.is_ascii_graphic())
            || distinct.len() < 12
        {
            bail!(
                "API token must contain 32..256 printable ASCII bytes and at least 12 distinct characters; use a cryptographically random token"
            );
        }
        Ok(Self(Arc::new(Inner {
            runtime,
            config,
            bind,
            token_hash: Sha256::digest(token.as_bytes()).into(),
            rate: Mutex::new(RateWindow {
                started: Instant::now(),
                count: 0,
            }),
            requests: Semaphore::new(64),
            runs: Arc::new(Semaphore::new(MAX_RUNS)),
        })))
    }
    fn scope(&self, session: String) -> Result<Scope, ApiError> {
        let scope = Scope {
            tenant: self.0.config.tenant.clone(),
            user: self.0.config.user.clone(),
            channel: "api".into(),
            session,
        };
        scope
            .validate()
            .map_err(|_| ApiError(StatusCode::BAD_REQUEST, "invalid session"))?;
        Ok(scope)
    }
}

/// Build without opening sockets, suitable for isolated tower::ServiceExt tests.
pub fn router(state: ApiState) -> Router {
    let api = Router::new()
        .route("/health", get(health))
        .route("/chat", post(chat))
        .route("/sessions", get(sessions))
        .route("/sessions/{session}/history", get(history))
        .route("/turns/{id}", get(status))
        .route("/turns/{id}/cancel", post(cancel))
        .route("/turns/{id}/resume", post(resume))
        .layer(middleware::from_fn_with_state(state.clone(), authenticate));
    Router::new()
        .nest("/api/v1", api)
        .route("/", get(|| async { Html(include_str!("console.html")) }))
        .route(
            "/console.js",
            get(|| async {
                (
                    [(
                        header::CONTENT_TYPE,
                        "application/javascript; charset=utf-8",
                    )],
                    include_str!("console.js"),
                )
            }),
        )
        .route(
            "/console.css",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
                    include_str!("console.css"),
                )
            }),
        )
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            security_headers,
        ))
        .with_state(state)
}

pub async fn serve(runtime: Arc<Runtime>, config: ApiConfig) -> Result<()> {
    let state = ApiState::new(runtime, config)?;
    let listener = tokio::net::TcpListener::bind(state.0.bind)
        .await
        .context("bind loopback API")?;
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

async fn security_headers(State(state): State<ApiState>, req: Request, next: Next) -> Response {
    // Defeat DNS rebinding and do not serve a token entry page through an arbitrary Host.
    let valid_host = req
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|host| trusted_host(host, state.0.bind.port()));
    let mut response = if valid_host {
        next.run(req).await
    } else {
        ApiError(StatusCode::BAD_REQUEST, "loopback Host header required").into_response()
    };
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_SECURITY_POLICY, HeaderValue::from_static("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'; object-src 'none'"));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    response
}

async fn authenticate(State(state): State<ApiState>, req: Request, next: Next) -> Response {
    // No cookies or CORS. Reject browser cross-origin requests even if a token was supplied.
    if req
        .headers()
        .get("sec-fetch-site")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|s| s == "cross-site")
    {
        return ApiError(StatusCode::FORBIDDEN, "cross-origin API request denied").into_response();
    }
    if let Some(origin) = req.headers().get(header::ORIGIN) {
        let host = req
            .headers()
            .get(header::HOST)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if origin.to_str().ok() != Some(format!("http://{host}").as_str())
            || !trusted_host(host, state.0.bind.port())
        {
            return ApiError(StatusCode::FORBIDDEN, "cross-origin API request denied")
                .into_response();
        }
    }
    let auth = req.headers().get_all(header::AUTHORIZATION);
    let mut values = auth.iter();
    let value = values
        .next()
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "));
    let authorized = value.filter(|s| s.len() <= 256).is_some_and(|s| {
        let hash: [u8; 32] = Sha256::digest(s.as_bytes()).into();
        bool::from(hash.ct_eq(&state.0.token_hash))
    });
    if !authorized || values.next().is_some() {
        let mut response =
            ApiError(StatusCode::UNAUTHORIZED, "bearer authentication required").into_response();
        response
            .headers_mut()
            .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        return response;
    }
    // One token maps to exactly one configured principal. No client-controlled rate key.
    {
        let Ok(mut rate) = state.0.rate.lock() else {
            return ApiError(StatusCode::SERVICE_UNAVAILABLE, "rate limiter unavailable")
                .into_response();
        };
        if rate.started.elapsed() >= Duration::from_secs(60) {
            rate.started = Instant::now();
            rate.count = 0;
        }
        if rate.count >= state.0.config.requests_per_minute {
            let mut response = ApiError(
                StatusCode::TOO_MANY_REQUESTS,
                "principal rate limit exceeded",
            )
            .into_response();
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("60"));
            return response;
        }
        rate.count += 1;
    }
    let Ok(_permit) = state.0.requests.try_acquire() else {
        return ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "API concurrency limit reached",
        )
        .into_response();
    };
    next.run(req).await
}
fn trusted_host(host: &str, port: u16) -> bool {
    host == format!("localhost:{port}")
        || host
            .parse::<SocketAddr>()
            .is_ok_and(|a| a.ip().is_loopback() && a.port() == port)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatRequest {
    session: String,
    prompt: String,
    request_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionRequest {
    session: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionQuery {
    session: String,
    #[serde(default = "default_limit")]
    limit: usize,
}
fn default_limit() -> usize {
    100
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HistoryQuery {
    #[serde(default = "default_limit")]
    limit: usize,
}
fn bounded_limit(limit: usize) -> Result<usize, ApiError> {
    if limit == 0 || limit > 200 {
        Err(ApiError(StatusCode::BAD_REQUEST, "limit must be 1..200"))
    } else {
        Ok(limit)
    }
}
fn validate_id(id: &str) -> Result<(), ApiError> {
    if id.is_empty()
        || id.len() > 128
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_: .".contains(&b) && b != b' ')
    {
        Err(ApiError(StatusCode::BAD_REQUEST, "invalid identifier"))
    } else {
        Ok(())
    }
}
async fn health() -> Json<Value> {
    Json(json!({"status":"ok","api_version":1,"discord":"separate_opt_in_process"}))
}
async fn chat(
    State(state): State<ApiState>,
    Json(body): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    let scope = state.scope(body.session)?;
    validate_id(&body.request_id)?;
    if body.prompt.trim().is_empty() || body.prompt.len() > MAX_PROMPT_BYTES {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "prompt must be nonempty and at most 32768 bytes",
        ));
    }
    let _permit = state.0.runs.clone().try_acquire_owned().map_err(|_| {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "admission concurrency limit reached",
        )
    })?;
    // Runtime owns a bounded task and durable identity before returning 202.
    // Disconnecting the HTTP request cannot abort an admitted turn.
    let result = state
        .0
        .runtime
        .start(scope, body.prompt, body.request_id)
        .await
        .map_err(runtime_error)?;
    outcome_response(result)
}
async fn sessions(
    State(state): State<ApiState>,
    Query(query): Query<SessionQuery>,
) -> Result<Json<Value>, ApiError> {
    let scope = state.scope(query.session)?;
    let limit = bounded_limit(query.limit)?;
    let turns = state
        .0
        .runtime
        .store()
        .list_turns(&scope, limit)
        .map_err(store_error)?;
    Ok(Json(json!({"session":scope.session,"turns":turns})))
}
async fn history(
    State(state): State<ApiState>,
    Path(session): Path<String>,
    Query(query): Query<HistoryQuery>,
) -> Result<Json<Value>, ApiError> {
    let scope = state.scope(session)?;
    let messages = state
        .0
        .runtime
        .store()
        .history(&scope, bounded_limit(query.limit)?)
        .map_err(store_error)?;
    Ok(Json(json!({"session":scope.session,"messages":messages})))
}
async fn status(
    State(state): State<ApiState>,
    Path(id): Path<String>,
    Query(query): Query<SessionQuery>,
) -> Result<Json<Value>, ApiError> {
    validate_id(&id)?;
    let scope = state.scope(query.session)?;
    let turn = state
        .0
        .runtime
        .status(&scope, &id)
        .map_err(|_| ApiError(StatusCode::NOT_FOUND, "turn not found"))?;
    Ok(Json(json!({"turn":turn})))
}
async fn cancel(
    State(state): State<ApiState>,
    Path(id): Path<String>,
    Json(body): Json<SessionRequest>,
) -> Result<Response, ApiError> {
    validate_id(&id)?;
    let scope = state.scope(body.session)?;
    let turn = state
        .0
        .runtime
        .status(&scope, &id)
        .map_err(|_| ApiError(StatusCode::NOT_FOUND, "turn not found"))?;
    if matches!(
        turn.status,
        missy_store::TurnStatus::Completed
            | missy_store::TurnStatus::Failed
            | missy_store::TurnStatus::Cancelled
    ) {
        return Err(ApiError(StatusCode::CONFLICT, "turn is already terminal"));
    }
    state.0.runtime.cancel(&scope, &id).map_err(runtime_error)?;
    // Cancellation is a request, not a claim that the worker has already stopped.
    Ok((
        StatusCode::ACCEPTED,
        Json(json!({"turn_id":id,"cancellation_requested":true})),
    )
        .into_response())
}
async fn resume(
    State(state): State<ApiState>,
    Path(id): Path<String>,
    Json(body): Json<SessionRequest>,
) -> Result<Response, ApiError> {
    validate_id(&id)?;
    let scope = state.scope(body.session)?;
    state
        .0
        .runtime
        .status(&scope, &id)
        .map_err(|_| ApiError(StatusCode::NOT_FOUND, "turn not found"))?;
    let permit = state.0.runs.clone().try_acquire_owned().map_err(|_| {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "run concurrency limit reached",
        )
    })?;
    let runtime = state.0.runtime.clone();
    let task = tokio::spawn(async move {
        let _permit = permit;
        runtime.resume(scope, &id).await
    });
    let result = task
        .await
        .map_err(|_| {
            ApiError(
                StatusCode::INTERNAL_SERVER_ERROR,
                "resume task failed; inspect scoped status",
            )
        })?
        .map_err(runtime_error)?;
    outcome_response(result)
}

fn outcome_response(result: impl serde::Serialize) -> Result<Response, ApiError> {
    let value = serde_json::to_value(result)
        .map_err(|_| ApiError(StatusCode::INTERNAL_SERVER_ERROR, "invalid runtime result"))?;
    let code = match value.get("status").and_then(Value::as_str) {
        Some("completed") => StatusCode::OK,
        Some("admitted" | "running") => StatusCode::ACCEPTED,
        Some("cancelled" | "interrupted") => StatusCode::CONFLICT,
        Some("failed") => StatusCode::UNPROCESSABLE_ENTITY,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    Ok((code, Json(value)).into_response())
}

#[derive(Debug)]
struct ApiError(StatusCode, &'static str);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error":self.1}))).into_response()
    }
}
fn runtime_error(_: anyhow::Error) -> ApiError {
    ApiError(
        StatusCode::CONFLICT,
        "turn could not execute; inspect scoped status before retrying",
    )
}
fn store_error(_: anyhow::Error) -> ApiError {
    ApiError(StatusCode::INTERNAL_SERVER_ERROR, "storage request failed")
}

/// Validate opt-in configuration; all connections use policy transport only.
pub fn require_discord_transport(config: &missy_core::DiscordConfig) -> Result<()> {
    if config.enabled {
        discord::validate_config(config)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
