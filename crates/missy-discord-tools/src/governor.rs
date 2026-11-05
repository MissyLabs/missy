//! Shared header-aware Discord REST throttling. No blind mutation retries.
//! One conservative bucket per bot/origin is intentionally stricter than the
//! platform's route buckets and covers concurrent adapters in this process.
use anyhow::{Result, anyhow, ensure};
use missy_policy::{HttpResponse, PolicyHttpClient};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};
use tokio::{sync::Mutex as AsyncMutex, time::Instant};
use tokio_util::sync::CancellationToken;
pub struct DiscordRestGovernor {
    inner: Arc<AsyncMutex<Instant>>,
}
impl Clone for DiscordRestGovernor {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
impl DiscordRestGovernor {
    pub fn shared(origin: &str, credential: &str) -> Result<Self> {
        type Registry = Mutex<HashMap<[u8; 32], Arc<AsyncMutex<Instant>>>>;
        static REGISTRY: OnceLock<Registry> = OnceLock::new();
        let origin = url::Url::parse(origin)
            .map_err(|_| anyhow!("Discord REST origin invalid"))?
            .origin()
            .ascii_serialization();
        let key: [u8; 32] = Sha256::digest(format!("{origin}\0{credential}").as_bytes()).into();
        let mut map = REGISTRY
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .map_err(|_| anyhow!("Discord REST governor unavailable"))?;
        if let Some(inner) = map.get(&key) {
            return Ok(Self {
                inner: inner.clone(),
            });
        }
        ensure!(
            map.len() < 1024,
            "Discord REST governor account limit exceeded"
        );
        let inner = Arc::new(AsyncMutex::new(Instant::now()));
        map.insert(key, inner.clone());
        Ok(Self { inner })
    }
    pub async fn request(
        &self,
        http: &PolicyHttpClient,
        method: &str,
        url: &str,
        headers: Vec<(String, String)>,
        body: Option<Vec<u8>>,
        cancel: &CancellationToken,
    ) -> Result<HttpResponse> {
        let mut next = tokio::select! {_=cancel.cancelled()=>return Err(anyhow!("Discord request cancelled before dispatch")),guard=self.inner.lock()=>guard};
        tokio::select! {_=cancel.cancelled()=>return Err(anyhow!("Discord request cancelled before dispatch")),_=tokio::time::sleep_until(*next)=>{}};
        let response = tokio::select! {_=cancel.cancelled()=>return Err(anyhow!("Discord request interrupted; outcome may be unknown")),r=http.request(method,url,headers,body)=>r.map_err(|_|anyhow!("Discord REST transport failed; outcome may be unknown"))?};
        let delay = delay(&response);
        *next = Instant::now() + delay;
        Ok(response)
    }
}
fn header<'a>(response: &'a HttpResponse, name: &str) -> Option<&'a str> {
    response
        .headers
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}
fn seconds(raw: Option<&str>) -> Option<f64> {
    raw.and_then(|s| s.parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0)
}
fn delay(response: &HttpResponse) -> Duration {
    let mut seconds_delay = 0.05f64; // <=20 requests/sec, below Discord's global50/s.
    if header(response, "x-ratelimit-remaining") == Some("0") {
        seconds_delay =
            seconds_delay.max(seconds(header(response, "x-ratelimit-reset-after")).unwrap_or(1.0));
    }
    if response.status == 429 {
        let json = serde_json::from_slice::<Value>(&response.body).ok();
        let retry = json
            .as_ref()
            .and_then(|v| v["retry_after"].as_f64())
            .filter(|n| n.is_finite() && *n >= 0.0);
        seconds_delay = seconds_delay.max(
            retry
                .or_else(|| seconds(header(response, "retry-after")))
                .unwrap_or(1.0),
        );
    }
    // Very long server limits cannot hang an agent indefinitely. Caller sees429
    // and must stop or explicitly retry later; never replay a write automatically.
    Duration::from_secs_f64(seconds_delay.clamp(0.05, 120.0))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn global_and_bucket_headers_are_bounded() {
        let r = HttpResponse {
            status: 200,
            headers: vec![
                ("X-RateLimit-Remaining".into(), "0".into()),
                ("X-RateLimit-Reset-After".into(), "3.2".into()),
            ],
            body: vec![],
        };
        assert_eq!(delay(&r), Duration::from_secs_f64(3.2));
        let r = HttpResponse {
            status: 429,
            headers: vec![],
            body: br#"{"global":true,"retry_after":1.7}"#.to_vec(),
        };
        assert_eq!(delay(&r), Duration::from_secs_f64(1.7));
        let r = HttpResponse {
            status: 429,
            headers: vec![],
            body: br#"{"retry_after":9999999999}"#.to_vec(),
        };
        assert_eq!(delay(&r), Duration::from_secs(120));
    }
    #[tokio::test]
    async fn adapters_share_bot_bucket_and_cancellation_never_dispatches() {
        let a = DiscordRestGovernor::shared("https://discord.com", "synthetic-a").unwrap();
        let b = DiscordRestGovernor::shared("https://discord.com", "synthetic-a").unwrap();
        assert!(Arc::ptr_eq(&a.inner, &b.inner));
        let c = DiscordRestGovernor::shared("https://discord.com", "synthetic-b").unwrap();
        assert!(!Arc::ptr_eq(&a.inner, &c.inner));
        let cancel = CancellationToken::new();
        cancel.cancel();
        let client = PolicyHttpClient::new(Default::default()).unwrap();
        assert!(
            a.request(
                &client,
                "POST",
                "https://discord.com/never",
                vec![],
                None,
                &cancel
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("before dispatch")
        );
    }
}
