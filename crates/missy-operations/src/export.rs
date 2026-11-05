use crate::{Diagnostics, OperationalKind};
use anyhow::{Result, ensure};
use hmac::{Hmac, Mac};
use missy_policy::PolicyHttpClient;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::Sha256;

/// Fixed instrumentation names and enum attributes only. No arbitrary attributes,
/// prompts, payloads or URLs can be smuggled into exported spans.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationalSpan {
    pub trace_id: String,
    pub span_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    pub start_time_unix_nano: u64,
    pub end_time_unix_nano: u64,
    pub kind: OperationalKind,
    pub succeeded: bool,
}
impl OperationalSpan {
    pub fn new(kind: OperationalKind, start: u64, end: u64, succeeded: bool) -> Result<Self> {
        ensure!(start > 0 && end >= start, "invalid span timestamps");
        let trace_id = uuid::Uuid::new_v4().simple().to_string();
        let span_id = uuid::Uuid::new_v4().simple().to_string()[..16].to_owned();
        Ok(Self {
            trace_id,
            span_id,
            parent_span_id: None,
            start_time_unix_nano: start,
            end_time_unix_nano: end,
            kind,
            succeeded,
        })
    }
    fn validate(&self) -> Result<()> {
        for (id, len) in [(&self.trace_id, 32), (&self.span_id, 16)] {
            ensure!(
                id.len() == len
                    && id.bytes().all(|b| b.is_ascii_hexdigit())
                    && id.bytes().any(|b| b != b'0'),
                "invalid OTLP span identity"
            );
        }
        if let Some(id) = &self.parent_span_id {
            ensure!(
                id.len() == 16
                    && id.bytes().all(|b| b.is_ascii_hexdigit())
                    && id.bytes().any(|b| b != b'0'),
                "invalid OTLP parent identity"
            );
        }
        ensure!(
            self.start_time_unix_nano > 0 && self.end_time_unix_nano >= self.start_time_unix_nano,
            "invalid span timestamps"
        );
        Ok(())
    }
}
/// Proper OTLP/HTTP JSON ExportTraceServiceRequest. Caller supplies a *policy
/// client*, not reqwest. All destinations use origin/DNS/redirect/body policy.
pub struct OtlpExporter {
    http: PolicyHttpClient,
    endpoint: String,
    max_batch: usize,
    max_bytes: usize,
}
impl OtlpExporter {
    pub fn new(
        http: PolicyHttpClient,
        endpoint: String,
        max_batch: usize,
        max_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=1024).contains(&max_batch) && (512..=1_048_576).contains(&max_bytes),
            "invalid OTLP collector limits"
        );
        ensure!(
            endpoint.ends_with("/v1/traces") && endpoint.len() <= 2048,
            "OTLP endpoint must end in /v1/traces"
        );
        Ok(Self {
            http,
            endpoint,
            max_batch,
            max_bytes,
        })
    }
    pub fn payload(&self, spans: &[OperationalSpan]) -> Result<Vec<u8>> {
        ensure!(
            !spans.is_empty() && spans.len() <= self.max_batch,
            "OTLP batch bound exceeded"
        );
        let spans: Vec<Value> = spans.iter().map(|s| -> Result<Value> {
            s.validate()?;
            let mut span = json!({"traceId":s.trace_id,"spanId":s.span_id,"name":"missy.operation","kind":1,"startTimeUnixNano":s.start_time_unix_nano.to_string(),"endTimeUnixNano":s.end_time_unix_nano.to_string(),"attributes":[{"key":"missy.operation.kind","value":{"stringValue":serde_json::to_value(&s.kind)?}},{"key":"missy.operation.succeeded","value":{"boolValue":s.succeeded}}],"status":{"code":if s.succeeded {1} else {2}}});
            if let Some(parent) = &s.parent_span_id { span["parentSpanId"] = json!(parent); }
            Ok(span)
        }).collect::<Result<_>>()?;
        let body = serde_json::to_vec(
            &json!({"resourceSpans":[{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"missy"}}]},"scopeSpans":[{"scope":{"name":"missy-operations","version":env!("CARGO_PKG_VERSION")},"spans":spans}]}]}),
        )?;
        ensure!(
            body.len() <= self.max_bytes,
            "OTLP encoded byte bound exceeded"
        );
        Ok(body)
    }
    pub async fn export(&self, spans: &[OperationalSpan]) -> Result<()> {
        let body = self.payload(spans)?;
        let response = self
            .http
            .request(
                "POST",
                &self.endpoint,
                vec![("content-type".into(), "application/json".into())],
                Some(body),
            )
            .await?;
        ensure!(
            (200..300).contains(&response.status),
            "OTLP collector rejected export"
        );
        if !response.body.is_empty() {
            let response: Value = serde_json::from_slice(&response.body)
                .map_err(|_| anyhow::anyhow!("invalid OTLP collector response"))?;
            if let Some(partial) = response.get("partialSuccess") {
                let rejected = partial
                    .get("rejectedSpans")
                    .and_then(|v| {
                        v.as_str()
                            .and_then(|s| s.parse::<u64>().ok())
                            .or_else(|| v.as_u64())
                    })
                    .unwrap_or(0);
                ensure!(
                    rejected == 0
                        && partial
                            .get("errorMessage")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .is_empty(),
                    "OTLP collector reported partial rejection"
                );
            }
        }
        Ok(())
    }
}
/// Explicit one-shot, signed *redacted operational summary* for SIEM ingestion.
/// This is not a full audit journal and does not claim to export one.
pub async fn export_signed_summary(
    http: &PolicyHttpClient,
    endpoint: &str,
    key_env: &str,
    report: &Diagnostics,
) -> Result<()> {
    ensure!(
        key_env.starts_with("MISSY_")
            && key_env.len() <= 128
            && key_env
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_'),
        "dedicated MISSY signing key reference required"
    );
    let key = std::env::var(key_env)
        .map_err(|_| anyhow::anyhow!("summary signing credential unavailable"))?;
    ensure!(
        (32..=4096).contains(&key.len()),
        "invalid signing credential length"
    );
    let body = serde_json::to_vec(report)?;
    ensure!(body.len() <= 65_536, "summary byte bound exceeded");
    let timestamp = chrono::Utc::now().timestamp().to_string();
    let mut mac = Hmac::<Sha256>::new_from_slice(key.as_bytes()).expect("HMAC accepts key length");
    mac.update(timestamp.as_bytes());
    mac.update(b".");
    mac.update(&body);
    let response = http
        .request(
            "POST",
            endpoint,
            vec![
                ("content-type".into(), "application/json".into()),
                ("x-missy-timestamp".into(), timestamp),
                (
                    "x-missy-signature".into(),
                    hex::encode(mac.finalize().into_bytes()),
                ),
            ],
            Some(body),
        )
        .await?;
    ensure!(
        (200..300).contains(&response.status),
        "summary collector rejected export"
    );
    Ok(())
}
