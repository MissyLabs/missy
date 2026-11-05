//! Real TCP fixtures, using explicit policy opt-in for local HTTP.
use axum::{
    Router, body::Bytes, extract::State, http::StatusCode, response::IntoResponse, routing::post,
};
use missy_core::{Message, PolicyConfig, Provider, ProviderConfig, ToolDefinition};
use missy_policy::PolicyHttpClient;
use missy_providers::{CredentialResolver, HttpProvider, ProviderPool, cost_microusd};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
struct Reply {
    status: StatusCode,
    body: String,
    delay: Duration,
    stream: bool,
}
impl Reply {
    fn json(v: Value) -> Self {
        Self {
            status: StatusCode::OK,
            body: v.to_string(),
            delay: Duration::ZERO,
            stream: false,
        }
    }
    fn raw(status: StatusCode, body: &str) -> Self {
        Self {
            status,
            body: body.into(),
            delay: Duration::ZERO,
            stream: false,
        }
    }
}
#[derive(Clone)]
struct StateData {
    replies: Arc<Mutex<VecDeque<Reply>>>,
    requests: Arc<Mutex<Vec<Value>>>,
}
struct Fixture {
    origin: String,
    state: StateData,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Fixture {
    async fn start(replies: Vec<Reply>) -> Self {
        async fn handler(State(state): State<StateData>, body: Bytes) -> impl IntoResponse {
            state
                .requests
                .lock()
                .unwrap()
                .push(serde_json::from_slice(&body).unwrap());
            let reply = state
                .replies
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| {
                    Reply::raw(StatusCode::INTERNAL_SERVER_ERROR, "fixture exhausted")
                });
            tokio::time::sleep(reply.delay).await;
            (
                reply.status,
                [(
                    "content-type",
                    if reply.stream {
                        "text/event-stream"
                    } else {
                        "application/json"
                    },
                )],
                reply.body,
            )
        }
        let state = StateData {
            replies: Arc::new(Mutex::new(replies.into())),
            requests: Arc::default(),
        };
        let app = Router::new()
            .route("/v1/chat/completions", post(handler))
            .route("/v1/responses", post(handler))
            .route("/v1/messages", post(handler))
            .route("/api/chat", post(handler))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            origin,
            state,
            task,
        }
    }
    fn policy(&self) -> PolicyConfig {
        PolicyConfig {
            allowed_origins: vec![self.origin.clone()],
            allow_private_network: true,
            ..Default::default()
        }
    }
    fn config(&self, kind: &str) -> ProviderConfig {
        ProviderConfig {
            name: kind.into(),
            kind: kind.into(),
            base_url: format!(
                "{}{}",
                self.origin,
                if kind == "ollama" { "" } else { "/v1" }
            ),
            model: "fixture".into(),
            max_output_tokens: 128,
            timeout_seconds: 3,
            input_microusd_per_million: Some(1_000_000),
            output_microusd_per_million: Some(2_000_000),
            ..Default::default()
        }
    }
    fn provider(&self, kind: &str) -> HttpProvider {
        HttpProvider::new(
            self.config(kind),
            PolicyHttpClient::new(self.policy()).unwrap(),
        )
        .unwrap()
    }
    fn requests(&self) -> Vec<Value> {
        self.state.requests.lock().unwrap().clone()
    }
}
fn tools() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        name: "lookup".into(),
        description: "lookup".into(),
        parameters: json!({"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}),
        side_effect: false,
    }]
}
fn user() -> Vec<Message> {
    vec![
        Message::text("system", "boundary"),
        Message::text("user", "find"),
    ]
}
fn text() -> Value {
    json!({"choices":[{"message":{"role":"assistant","content":"finished"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":3}})
}
fn calls(c: Value) -> Value {
    json!({"choices":[{"message":{"role":"assistant","content":null,"tool_calls":c},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":3}})
}
fn call(id: &str, args: &str) -> Value {
    json!({"id":id,"type":"function","function":{"name":"lookup","arguments":args}})
}

#[derive(Default)]
struct FixtureCredentials(Mutex<Vec<String>>);
impl CredentialResolver for FixtureCredentials {
    fn resolve(&self, reference: &str) -> anyhow::Result<String> {
        self.0.lock().unwrap().push(reference.into());
        if reference == "MISSING" {
            anyhow::bail!("secret diagnostic must not escape");
        }
        Ok(format!("fixture-{reference}"))
    }
}
#[tokio::test]
async fn account_round_robin_and_rejection_cooldown_are_bounded() {
    let f = Fixture::start(vec![
        Reply::raw(StatusCode::UNAUTHORIZED, "secret"),
        Reply::json(text()),
        Reply::json(text()),
        Reply::json(text()),
    ])
    .await;
    let mut config = f.config("openai");
    config.api_key_envs = vec!["ACCOUNT_A".into(), "ACCOUNT_B".into()];
    let credentials = Arc::new(FixtureCredentials::default());
    let p = HttpProvider::with_credentials(
        config,
        PolicyHttpClient::new(f.policy()).unwrap(),
        credentials.clone(),
    )
    .unwrap();
    let first = p
        .complete(&user(), &[], CancellationToken::new())
        .await
        .unwrap();
    assert!(!first.usage.cost_known);
    for _ in 0..2 {
        p.complete(&user(), &[], CancellationToken::new())
            .await
            .unwrap();
    }
    assert_eq!(
        *credentials.0.lock().unwrap(),
        vec!["ACCOUNT_A", "ACCOUNT_B", "ACCOUNT_B", "ACCOUNT_B"]
    );
    assert_eq!(f.requests().len(), 4);
}
#[tokio::test]
async fn healthy_accounts_rotate_and_missing_reference_never_dispatches() {
    let f = Fixture::start(vec![
        Reply::json(text()),
        Reply::json(text()),
        Reply::json(text()),
    ])
    .await;
    let mut config = f.config("openai");
    config.api_key_envs = vec!["MISSING".into(), "ACCOUNT_B".into(), "ACCOUNT_C".into()];
    let credentials = Arc::new(FixtureCredentials::default());
    let p = HttpProvider::with_credentials(
        config,
        PolicyHttpClient::new(f.policy()).unwrap(),
        credentials.clone(),
    )
    .unwrap();
    for _ in 0..3 {
        assert!(
            p.complete(&user(), &[], CancellationToken::new())
                .await
                .unwrap()
                .usage
                .cost_known
        );
    }
    assert_eq!(
        *credentials.0.lock().unwrap(),
        vec!["MISSING", "ACCOUNT_B", "ACCOUNT_B", "ACCOUNT_C"]
    );
    assert_eq!(f.requests().len(), 3);
}
fn responses(output: Value) -> Value {
    json!({"status":"completed","output":output,"usage":{"input_tokens":10,"output_tokens":3}})
}

#[tokio::test]
async fn openai_sse_tools_and_final_usage() {
    let events = [
        json!({"choices":[{"index":0,"delta":{"role":"assistant","content":"é","tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"look","arguments":"{\"q\":"}}]},"finish_reason":null}]}),
        json!({"choices":[{"index":0,"delta":{"content":"𐐀","tool_calls":[{"index":0,"function":{"name":"up","arguments":"\"rust\"}"}}]},"finish_reason":"tool_calls"}]}),
        json!({"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3}}),
    ];
    let body = events
        .iter()
        .map(|v| format!("data: {v}\r\n\r\n"))
        .collect::<String>()
        + "data: [DONE]\r\n\r\n";
    let mut reply = Reply::raw(StatusCode::OK, &body);
    reply.stream = true;
    let f = Fixture::start(vec![reply]).await;
    let mut config = f.config("openai");
    config.stream = true;
    let p = HttpProvider::new(config, PolicyHttpClient::new(f.policy()).unwrap()).unwrap();
    let c = p
        .complete(&user(), &tools(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(c.message.content, "é𐐀");
    assert_eq!(c.message.tool_calls[0].arguments, json!({"q":"rust"}));
    assert_eq!(c.usage.input_tokens, 10);
    assert_eq!(f.requests()[0]["stream_options"]["include_usage"], true);
}
#[tokio::test]
async fn stream_partial_missing_usage_and_error_fail_closed() {
    for body in [
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"text\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        "data: {\"error\":{\"message\":\"secret-never-echo\"}}\n\n",
    ] {
        let mut reply = Reply::raw(StatusCode::OK, body);
        reply.stream = true;
        let f = Fixture::start(vec![reply]).await;
        let mut config = f.config("openai");
        config.stream = true;
        let p = HttpProvider::new(config, PolicyHttpClient::new(f.policy()).unwrap()).unwrap();
        let e = p
            .complete(&user(), &[], CancellationToken::new())
            .await
            .unwrap_err();
        assert!(!format!("{e:#}").contains("secret-never-echo"));
    }
}
#[tokio::test]
async fn native_responses_no_tools_and_gateway_functions_roundtrip() {
    let f = Fixture::start(vec![
        Reply::json(responses(json!([{"type":"message","role":"assistant","content":[{"type":"output_text","text":"native text"}]}]))),
        Reply::json(responses(json!([{"type":"function_call","call_id":"r1","name":"lookup","arguments":"{\"q\":\"rust\"}"}]))),
        Reply::json(responses(json!([{"type":"message","role":"assistant","content":[{"type":"output_text","text":"finished"}]}]))),
    ]).await;
    let p = f.provider("openai_responses");
    let c = p
        .complete(&user(), &[], CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(c.message.content, "native text");
    let c = p
        .complete(&user(), &tools(), CancellationToken::new())
        .await
        .unwrap();
    let mut history = user();
    history.push(c.message);
    history.push(Message::tool("r1", "evidence"));
    p.complete(&history, &tools(), CancellationToken::new())
        .await
        .unwrap();
    let requests = f.requests();
    assert_eq!(requests[0]["tool_choice"], "none");
    assert_eq!(requests[0]["tools"], json!([]));
    assert_eq!(requests[0]["store"], false);
    assert_eq!(requests[1]["tools"][0]["type"], "function");
    assert!(
        requests[2]["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v["type"] == "function_call_output" && v["call_id"] == "r1")
    );
}
#[tokio::test]
async fn responses_rejects_native_tools_unadvertised_calls_and_ambiguous_arguments() {
    for output in [
        json!([{"type":"web_search_call","id":"native"}]),
        json!([{"type":"function_call","call_id":"r1","name":"lookup","arguments":"{\"q\":\"a\",\"q\":\"b\"}"}]),
        json!([{"type":"function_call","call_id":"r1","name":"lookup","arguments":"{}"}]),
    ] {
        let f = Fixture::start(vec![Reply::json(responses(output))]).await;
        assert!(
            f.provider("openai_responses")
                .complete(&user(), &[], CancellationToken::new())
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn openai_tool_roundtrip_usage() {
    let f = Fixture::start(vec![
        Reply::json(calls(json!([call("c1", r#"{"q":"rust"}"#)]))),
        Reply::json(text()),
    ])
    .await;
    let p = f.provider("openai");
    let mut history = user();
    let c = p
        .complete(&history, &tools(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(c.message.tool_calls[0].arguments, json!({"q":"rust"}));
    assert_eq!(c.usage.cost_microusd, 16);
    assert!(c.usage.cost_known);
    history.push(c.message);
    history.push(Message::tool("c1", "answer"));
    p.complete(&history, &tools(), CancellationToken::new())
        .await
        .unwrap();
    let r = f.requests();
    assert_eq!(r[0]["messages"][0]["role"], "system");
    assert_eq!(r[0]["max_tokens"], 128);
    assert_eq!(
        r[1]["messages"][2]["tool_calls"][0]["function"]["arguments"],
        r#"{"q":"rust"}"#
    );
    assert_eq!(r[1]["messages"][3]["tool_call_id"], "c1");
}

#[tokio::test]
async fn anthropic_content_use_result() {
    let f=Fixture::start(vec![Reply::json(json!({"type":"message","role":"assistant","content":[{"type":"text","text":"checking"},{"type":"tool_use","id":"tu1","name":"lookup","input":{"q":"rust"}}],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":3}})),Reply::json(json!({"type":"message","role":"assistant","content":[{"type":"text","text":"finished"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":3}}))]).await;
    let p = f.provider("anthropic");
    let mut history = user();
    let c = p
        .complete(&history, &tools(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(c.message.content, "checking");
    history.push(c.message);
    history.push(Message::tool("tu1", "answer"));
    p.complete(&history, &tools(), CancellationToken::new())
        .await
        .unwrap();
    let r = f.requests();
    assert_eq!(r[0]["system"][0]["text"], "boundary");
    assert_eq!(r[0]["messages"][0]["role"], "user");
    assert_eq!(r[1]["messages"][1]["content"][1]["type"], "tool_use");
    assert_eq!(r[1]["messages"][2]["content"][0]["tool_use_id"], "tu1");
}

#[tokio::test]
async fn ollama_tools_generated_ids_and_results() {
    let f=Fixture::start(vec![Reply::json(json!({"done":true,"done_reason":"stop","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"lookup","arguments":{"q":"rust"}}}]},"prompt_eval_count":10,"eval_count":3})),Reply::json(json!({"done":true,"done_reason":"stop","message":{"role":"assistant","content":"done"},"prompt_eval_count":10,"eval_count":3}))]).await;
    let p = f.provider("ollama");
    let mut history = user();
    let c = p
        .complete(&history, &tools(), CancellationToken::new())
        .await
        .unwrap();
    let id = c.message.tool_calls[0].id.clone();
    assert!(id.starts_with("ollama_"));
    history.push(c.message);
    history.push(Message::tool(&id, "answer"));
    p.complete(&history, &tools(), CancellationToken::new())
        .await
        .unwrap();
    let r = f.requests();
    assert_eq!(r[1]["messages"][3]["tool_name"], "lookup");
    assert_eq!(r[0]["options"]["num_predict"], 128);
}

#[tokio::test]
async fn fallback_and_circuit_mark_uncertain_billing() {
    let bad = Fixture::start(vec![Reply::raw(
        StatusCode::SERVICE_UNAVAILABLE,
        "secret-never-echo",
    )])
    .await;
    let good = Fixture::start(vec![Reply::json(text()), Reply::json(text())]).await;
    let mut a = bad.config("openai");
    a.name = "bad".into();
    let mut b = good.config("openai");
    b.name = "good".into();
    let policy = PolicyConfig {
        allowed_origins: vec![bad.origin.clone(), good.origin.clone()],
        allow_private_network: true,
        ..Default::default()
    };
    let p = ProviderPool::new(vec![a, b], policy).unwrap();
    let c = p
        .complete(&user(), &[], CancellationToken::new())
        .await
        .unwrap();
    assert!(!c.usage.cost_known);
    assert_eq!(c.message.content, "finished");
    p.complete(&user(), &[], CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(bad.requests().len(), 1);
    assert_eq!(good.requests().len(), 2);
}

#[tokio::test]
async fn weighted_task_routing_and_capacity_fallback() {
    use missy_core::{RoutingConfig, RoutingMode};
    let a = Fixture::start((0..8).map(|_| Reply::json(text())).collect()).await;
    let b = Fixture::start((0..8).map(|_| Reply::json(text())).collect()).await;
    let mut first = a.config("openai");
    first.name = "a".into();
    let mut second = b.config("openai");
    second.name = "b".into();
    second.routing.weight = 3;
    let policy = PolicyConfig {
        allowed_origins: vec![a.origin.clone(), b.origin.clone()],
        allow_private_network: true,
        ..Default::default()
    };
    let weighted = ProviderPool::with_routing(
        vec![first.clone(), second.clone()],
        policy.clone(),
        RoutingConfig {
            mode: RoutingMode::Weighted,
            ..Default::default()
        },
    )
    .unwrap();
    for _ in 0..8 {
        weighted
            .complete(&user(), &[], CancellationToken::new())
            .await
            .unwrap();
    }
    assert_eq!(a.requests().len(), 2);
    assert_eq!(b.requests().len(), 6);
    first.routing.tiers = vec!["fast".into()];
    second.routing.tiers = vec!["premium".into()];
    let routed = ProviderPool::with_routing(
        vec![first.clone(), second.clone()],
        policy.clone(),
        RoutingConfig {
            default_tier: Some("fast".into()),
            task_tiers: [("worker".into(), "premium".into())].into(),
            ..Default::default()
        },
    )
    .unwrap();
    routed
        .complete_for(Some("worker"), None, &user(), &[], CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(b.requests().len(), 7);
    first.routing.tiers.clear();
    second.routing.tiers.clear();
    first.routing.tokens_per_minute = Some(1);
    let capped = ProviderPool::new(vec![first, second], policy).unwrap();
    let result = capped
        .complete(&user(), &[], CancellationToken::new())
        .await
        .unwrap();
    assert!(result.usage.cost_known);
    assert_eq!(a.requests().len(), 2);
    assert_eq!(b.requests().len(), 8);
}

#[tokio::test]
async fn last_eligible_fallback_preserves_unknown_prices_and_fingerprint() {
    use missy_core::{RoutingConfig, RoutingMode};
    let bad = Fixture::start(vec![Reply::raw(StatusCode::SERVICE_UNAVAILABLE, "hidden")]).await;
    let good = Fixture::start(vec![Reply::json(text()), Reply::json(text())]).await;
    let mut first = bad.config("openai");
    first.name = "bad".into();
    let mut last = good.config("openai");
    last.name = "last".into();
    last.input_microusd_per_million = None;
    let policy = PolicyConfig {
        allowed_origins: vec![bad.origin.clone(), good.origin.clone()],
        allow_private_network: true,
        ..Default::default()
    };
    let pool = ProviderPool::new(vec![first.clone(), last.clone()], policy.clone()).unwrap();
    assert!(pool.require_known_prices().is_err());
    assert!(
        !pool
            .complete(&user(), &[], CancellationToken::new())
            .await
            .unwrap()
            .usage
            .cost_known
    );
    assert!(
        !pool
            .complete(&user(), &[], CancellationToken::new())
            .await
            .unwrap()
            .usage
            .cost_known
    );
    assert_eq!(bad.requests().len(), 1);
    assert_eq!(good.requests().len(), 2);
    let weighted = ProviderPool::with_routing(
        vec![first, last],
        policy,
        RoutingConfig {
            mode: RoutingMode::Weighted,
            ..Default::default()
        },
    )
    .unwrap();
    assert_ne!(
        pool.configuration_fingerprint(),
        weighted.configuration_fingerprint()
    );
}

#[tokio::test]
async fn all_errors_and_no_body_leak() {
    let f = Fixture::start(vec![Reply::raw(
        StatusCode::UNAUTHORIZED,
        "secret-never-echo",
    )])
    .await;
    let p = ProviderPool::new(vec![f.config("openai")], f.policy()).unwrap();
    let e = p
        .complete(&user(), &[], CancellationToken::new())
        .await
        .unwrap_err();
    assert!(e.to_string().contains("all configured providers failed"));
    assert!(!format!("{e:#}").contains("secret-never-echo"));
    assert!(
        p.complete(&user(), &[], CancellationToken::new())
            .await
            .is_err()
    );
    assert_eq!(f.requests().len(), 1);
}

#[tokio::test]
async fn cancellation_during_request_no_fallback() {
    let mut slow = Reply::json(text());
    slow.delay = Duration::from_secs(30);
    let f = Fixture::start(vec![slow]).await;
    let mut b = f.config("openai");
    b.name = "fallback".into();
    let p = ProviderPool::new(vec![f.config("openai"), b], f.policy()).unwrap();
    let token = CancellationToken::new();
    let cancel = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel();
    });
    let result = tokio::time::timeout(Duration::from_secs(1), p.complete(&user(), &[], token))
        .await
        .unwrap();
    assert!(result.unwrap_err().to_string().contains("cancelled"));
    assert_eq!(f.requests().len(), 1);
}

#[tokio::test]
async fn timeout_and_precancel() {
    let mut slow = Reply::json(text());
    slow.delay = Duration::from_secs(30);
    let f = Fixture::start(vec![slow]).await;
    let mut config = f.config("openai");
    config.timeout_seconds = 1;
    let p = HttpProvider::new(config, PolicyHttpClient::new(f.policy()).unwrap()).unwrap();
    let token = CancellationToken::new();
    token.cancel();
    assert!(p.complete(&user(), &[], token).await.is_err());
    assert!(f.requests().is_empty());
    assert!(
        tokio::time::timeout(
            Duration::from_secs(2),
            p.complete(&user(), &[], CancellationToken::new())
        )
        .await
        .unwrap()
        .is_err()
    );
}

#[tokio::test]
async fn bad_json_roles_usage_and_truncation() {
    let mut variants = vec![json!({}), json!({"choices":[]})];
    for role in ["system", "user", "tool", "alien"] {
        let mut v = text();
        v["choices"][0]["message"]["role"] = json!(role);
        variants.push(v);
    }
    let mut v = text();
    v["choices"][0]["finish_reason"] = json!("length");
    variants.push(v);
    let mut v = text();
    v.as_object_mut().unwrap().remove("usage");
    variants.push(v);
    let mut v = text();
    v["usage"]["completion_tokens"] = json!(129);
    variants.push(v);
    let mut v = text();
    v["usage"]["prompt_tokens"] = json!(-1);
    variants.push(v);
    let replies = std::iter::once(Reply::raw(StatusCode::OK, "{bad"))
        .chain(variants.into_iter().map(Reply::json))
        .collect::<Vec<_>>();
    let n = replies.len();
    let f = Fixture::start(replies).await;
    let p = f.provider("openai");
    for _ in 0..n {
        assert!(
            p.complete(&user(), &[], CancellationToken::new())
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn invalid_tool_ids_args_and_schema() {
    let cases = vec![
        json!([call("dup", r#"{"q":"a"}"#), call("dup", r#"{"q":"b"}"#)]),
        json!([call("", "{}")]),
        json!([call("bad id", "{}")]),
        json!([call("ok", "{bad")]),
        json!([call("ok", "[]")]),
        json!([call("ok", r#"{"q":42}"#)]),
        json!([call("ok", "{}")]),
        json!([call("ok", r#"{"q":"a","extra":1}"#)]),
        json!([{"type":"function","function":{"name":"lookup","arguments":"{}"}}]),
        json!([{"id":"ok","type":"function","function":{"name":"unknown","arguments":"{}"}}]),
    ];
    let n = cases.len();
    let f = Fixture::start(cases.into_iter().map(|c| Reply::json(calls(c))).collect()).await;
    let p = f.provider("openai");
    for _ in 0..n {
        assert!(
            p.complete(&user(), &tools(), CancellationToken::new())
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn deny_policy_and_bad_history_before_network() {
    let f = Fixture::start(vec![]).await;
    let p = HttpProvider::new(
        f.config("openai"),
        PolicyHttpClient::new(PolicyConfig::default()).unwrap(),
    )
    .unwrap();
    assert!(
        p.complete(&user(), &[], CancellationToken::new())
            .await
            .is_err()
    );
    let p = f.provider("openai");
    assert!(
        p.complete(
            &[Message::text("alien", "x")],
            &[],
            CancellationToken::new()
        )
        .await
        .is_err()
    );
    assert!(
        p.complete(
            &[Message::tool("unknown", "x")],
            &[],
            CancellationToken::new()
        )
        .await
        .is_err()
    );
    assert!(
        p.complete(
            &[
                Message::text("user", "x"),
                Message::text("system", "elevate")
            ],
            &[],
            CancellationToken::new()
        )
        .await
        .is_err()
    );
    assert!(f.requests().is_empty());
}

#[tokio::test]
async fn unknown_price_and_fingerprint() {
    let f = Fixture::start(vec![Reply::json(text())]).await;
    let mut config = f.config("openai");
    config.input_microusd_per_million = None;
    let p = ProviderPool::new(vec![config.clone()], f.policy()).unwrap();
    assert!(!p.costs_known());
    assert!(p.require_known_prices().is_err());
    assert!(
        !p.complete(&user(), &[], CancellationToken::new())
            .await
            .unwrap()
            .usage
            .cost_known
    );
    let fp = p.configuration_fingerprint();
    config.model = "other".into();
    let p = ProviderPool::new(vec![config], f.policy()).unwrap();
    assert_ne!(fp, p.configuration_fingerprint());
}

#[test]
fn cost_arithmetic_and_env_refs_only() {
    assert_eq!(cost_microusd(1, 0, 1, 0).unwrap(), 1);
    assert_eq!(cost_microusd(3, 4, 1_000_000, 2_000_000).unwrap(), 11);
    assert!(cost_microusd(u64::MAX, u64::MAX, u64::MAX, u64::MAX).is_err());
    let c = ProviderConfig {
        model: "x".into(),
        api_key_env: Some("sk-secret-literal".into()),
        ..Default::default()
    };
    assert!(HttpProvider::new(c, PolicyHttpClient::new(PolicyConfig::default()).unwrap()).is_err());
}

#[tokio::test]
async fn oversize_response_rejected() {
    let f = Fixture::start(vec![Reply::raw(
        StatusCode::OK,
        &" ".repeat(4 * 1024 * 1024 + 1),
    )])
    .await;
    assert!(
        f.provider("openai")
            .complete(&user(), &[], CancellationToken::new())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn anthropic_and_ollama_reject_roles_ids_args_truncation() {
    let anthropic = json!({"type":"message","role":"assistant","content":[{"type":"tool_use","id":"tu1","name":"lookup","input":{"q":"rust"}}],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":3}});
    let ollama = json!({"done":true,"done_reason":"stop","message":{"role":"assistant","content":"","tool_calls":[{"id":"o1","function":{"name":"lookup","arguments":{"q":"rust"}}}]},"prompt_eval_count":10,"eval_count":3});
    for (kind, base) in [("anthropic", anthropic), ("ollama", ollama)] {
        let mut cases = vec![];
        for role in ["system", "user", "tool", "alien"] {
            let mut v = base.clone();
            if kind == "anthropic" {
                v["role"] = json!(role);
            } else {
                v["message"]["role"] = json!(role);
            }
            cases.push(v);
        }
        let mut v = base.clone();
        if kind == "anthropic" {
            v["stop_reason"] = json!("max_tokens");
        } else {
            v["done_reason"] = json!("length");
        }
        cases.push(v);
        let mut v = base.clone();
        if kind == "anthropic" {
            v["content"][0]["id"] = json!("");
        } else {
            v["message"]["tool_calls"][0]["id"] = json!("");
        }
        cases.push(v);
        let mut v = base.clone();
        if kind == "anthropic" {
            v["content"][0]["input"] = json!([]);
        } else {
            v["message"]["tool_calls"][0]["function"]["arguments"] = json!([]);
        }
        cases.push(v);
        let mut v = base.clone();
        if kind == "anthropic" {
            v["content"]
                .as_array_mut()
                .unwrap()
                .push(base["content"][0].clone());
        } else {
            v["message"]["tool_calls"]
                .as_array_mut()
                .unwrap()
                .push(base["message"]["tool_calls"][0].clone());
        }
        cases.push(v);
        let n = cases.len();
        let f = Fixture::start(cases.into_iter().map(Reply::json).collect()).await;
        let p = f.provider(kind);
        for _ in 0..n {
            assert!(
                p.complete(&user(), &tools(), CancellationToken::new())
                    .await
                    .is_err()
            );
        }
    }
}

#[tokio::test]
async fn duplicate_json_keys_are_not_last_wins() {
    let f=Fixture::start(vec![Reply::raw(StatusCode::OK,r#"{"choices":[{"message":{"role":"system","role":"assistant","content":"x"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#),Reply::json(calls(json!([call("id",r#"{"q":"one","q":"two"}"#)])))]).await;
    let p = f.provider("openai");
    for _ in 0..2 {
        assert!(
            p.complete(&user(), &tools(), CancellationToken::new())
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn full_schema_constraints_and_no_remote_refs() {
    let f = Fixture::start(vec![Reply::json(calls(json!([call(
        "id",
        r#"{"q":"short"}"#
    )])))])
    .await;
    let p = f.provider("openai");
    let mut t = tools();
    t[0].parameters["properties"]["q"]["minLength"] = json!(50);
    assert!(
        p.complete(&user(), &t, CancellationToken::new())
            .await
            .is_err()
    );
    t[0].parameters = json!({"type":"object","$ref":"http://127.0.0.1:12345/forbidden"});
    assert!(
        p.complete(&user(), &t, CancellationToken::new())
            .await
            .is_err()
    );
    assert_eq!(f.requests().len(), 1);
}

#[tokio::test]
async fn cached_pricing_not_silently_assumed() {
    let mut v = text();
    v["usage"]["prompt_tokens_details"] = json!({"cached_tokens":5});
    let f = Fixture::start(vec![Reply::json(v)]).await;
    assert!(
        f.provider("openai")
            .complete(&user(), &[], CancellationToken::new())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn native_anthropic_ollama_and_responses_stream_http_roundtrip() {
    let cases = [
        (
            "anthropic",
            vec![
                json!({"type":"message_start","message":{"role":"assistant","usage":{"input_tokens":10,"output_tokens":0}}}),
                json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"a1","name":"lookup","input":{}}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}),
                json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"rust\"}"}}),
                json!({"type":"content_block_stop","index":0}),
                json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}),
                json!({"type":"message_stop"}),
            ],
        ),
        (
            "ollama",
            vec![
                json!({"done":false,"message":{"role":"assistant","content":""}}),
                json!({"done":true,"done_reason":"stop","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"lookup","arguments":{"q":"rust"}}}]},"prompt_eval_count":10,"eval_count":3}),
            ],
        ),
        (
            "openai_responses",
            vec![
                json!({"type":"response.function_call_arguments.delta","delta":"{\"q\":"}),
                json!({"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call","call_id":"r1","name":"lookup","arguments":"{\"q\":\"rust\"}"}],"usage":{"input_tokens":10,"output_tokens":3}}}),
            ],
        ),
    ];
    for (kind, events) in cases {
        let body = events
            .iter()
            .map(|v| {
                if kind == "ollama" {
                    format!("{v}\n")
                } else {
                    format!("data: {v}\n\n")
                }
            })
            .collect::<String>();
        let reply = Reply {
            status: StatusCode::OK,
            body,
            delay: Duration::ZERO,
            stream: kind != "ollama",
        };
        let f = Fixture::start(vec![reply]).await;
        let mut config = f.config(kind);
        config.stream = true;
        let p = HttpProvider::new(config, PolicyHttpClient::new(f.policy()).unwrap()).unwrap();
        let completion = p
            .complete(&user(), &tools(), CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(completion.message.tool_calls.len(), 1, "{kind}");
        assert_eq!(
            completion.message.tool_calls[0].arguments,
            json!({"q":"rust"})
        );
        assert_eq!(completion.usage.input_tokens, 10);
        assert_eq!(completion.usage.output_tokens, 3);
        assert_eq!(f.requests()[0]["stream"], true);
    }
}

#[tokio::test]
async fn historical_call_id_cannot_be_reused() {
    let f = Fixture::start(vec![
        Reply::json(calls(json!([call("c1", r#"{"q":"rust"}"#)]))),
        Reply::json(calls(json!([call("c1", r#"{"q":"rust"}"#)]))),
    ])
    .await;
    let p = f.provider("openai");
    let mut h = user();
    let c = p
        .complete(&h, &tools(), CancellationToken::new())
        .await
        .unwrap();
    h.push(c.message);
    h.push(Message::tool("c1", "done"));
    assert!(
        p.complete(&h, &tools(), CancellationToken::new())
            .await
            .is_err()
    );
}
