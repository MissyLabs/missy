//! Bounded document/search parsing. All network access uses PolicyHttpClient.
//! These helpers do not bypass the runtime's immutable receipt/model projection boundary.
use anyhow::{Context, Result, bail};
use missy_policy::{Policy, PolicyHttpClient};
use scraper::{Html, Selector};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;
use url::Url;

const PARSE_LIMIT: usize = 1024 * 1024;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SearchArgs {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
}
fn default_limit() -> usize {
    5
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DocumentArgs {
    path: Option<String>,
    url: Option<String>,
    #[serde(default = "default_format")]
    format: String,
    #[serde(default = "default_chars")]
    max_chars: usize,
}
fn default_format() -> String {
    "text".into()
}
fn default_chars() -> usize {
    16000
}
pub(crate) async fn search(http: &PolicyHttpClient, a: SearchArgs) -> Result<Value> {
    if a.query.trim().is_empty() || a.query.len() > 2048 || !(1..=10).contains(&a.limit) {
        bail!("search query/limit outside bounds");
    }
    let mut url = Url::parse("https://html.duckduckgo.com/html/")?;
    url.query_pairs_mut().append_pair("q", &a.query);
    let response = http.get(url.as_str()).await?;
    if !(200..300).contains(&response.status) {
        bail!("search upstream returned status {}", response.status);
    }
    let text = bounded_utf8(response.body)?;
    // Scan the original source, including hidden comments, before DOM extraction.
    let scan = missy_security::inspect_untrusted(&text);
    if scan.quarantined {
        return Ok(evidence(
            &text,
            json!({"results":[],"security":scan,"status":"quarantined"}),
        ));
    }
    let view = match parse_search(&text, a.limit) {
        Ok(view) => view,
        Err(_) => {
            json!({"results":[],"status":"unrecognized","error":"search response not recognized (possible upstream challenge or markup change)"})
        }
    };
    Ok(evidence(&text, view))
}
fn evidence(source: &str, view: Value) -> Value {
    json!({"raw_evidence":{"source_utf8":source},"model_view":view})
}
fn selector(value: &str) -> Selector {
    Selector::parse(value).expect("static HTML selector")
}
fn visible_text(element: scraper::ElementRef<'_>) -> String {
    let mut text = String::new();
    for node in element.descendants() {
        if let Some(value) = node.value().as_text() {
            let hidden = node
                .ancestors()
                .take(128)
                .filter_map(scraper::ElementRef::wrap)
                .any(|e| {
                    matches!(
                        e.value().name(),
                        "script" | "style" | "noscript" | "template" | "svg"
                    ) || e.value().attr("hidden").is_some()
                        || e.value().attr("aria-hidden") == Some("true")
                });
            // Pathological deep HTML must not cause quadratic ancestor traversal.
            // Deep text is withheld rather than skipping potentially hidden ancestors.
            if !hidden && node.ancestors().nth(128).is_none() {
                text.push_str(value);
                text.push(' ');
            }
        }
    }
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
fn parse_search(text: &str, limit: usize) -> Result<Value> {
    let document = Html::parse_document(text);
    let mut results = Vec::new();
    for element in document.select(&selector(".result")).take(100) {
        let Some(link) = element.select(&selector("a.result__a")).next() else {
            continue;
        };
        let Some(href) = link.value().attr("href") else {
            continue;
        };
        let Some(url) = result_url(href) else {
            continue;
        };
        let title = visible_text(link).chars().take(1000).collect::<String>();
        let snippet = element
            .select(&selector(".result__snippet"))
            .next()
            .map(visible_text)
            .unwrap_or_default()
            .chars()
            .take(4000)
            .collect::<String>();
        results.push(json!({"title":title,"url":url,"snippet":snippet}));
        if results.len() >= limit {
            break;
        }
    }
    // A challenge page is not an authoritative empty result set.
    if results.is_empty() && !text.contains("No results") && !text.contains("no-results") {
        bail!("search response not recognized (possible upstream challenge or markup change)");
    }
    Ok(json!({"results":results,"source":"duckduckgo_html","fetched_pages":false}))
}
fn result_url(href: &str) -> Option<String> {
    let parsed = Url::parse(href)
        .or_else(|_| {
            Url::parse("https://html.duckduckgo.com")
                .unwrap()
                .join(href)
        })
        .ok()?;
    let url = if parsed
        .host_str()
        .is_some_and(|h| matches!(h, "duckduckgo.com" | "html.duckduckgo.com"))
        && parsed.path() == "/l/"
    {
        Url::parse(&parsed.query_pairs().find(|(k, _)| k == "uddg")?.1).ok()?
    } else {
        parsed
    };
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return None;
    }
    Some(url.to_string())
}
fn bounded_utf8(bytes: Vec<u8>) -> Result<String> {
    if bytes.len() > PARSE_LIMIT {
        bail!("document parser input exceeds 1MiB");
    }
    if bytes.starts_with(b"%PDF-") || bytes.starts_with(b"PK\x03\x04") || bytes.contains(&0) {
        bail!("binary/PDF/archive document parsing is unsupported");
    }
    String::from_utf8(bytes).context("document must be UTF-8; binary/PDF/OCR unsupported")
}
pub(crate) async fn read_document(
    policy: &Policy,
    http: &PolicyHttpClient,
    a: DocumentArgs,
) -> Result<Value> {
    if !(1..=64000).contains(&a.max_chars) || !matches!(a.format.as_str(), "text" | "html" | "json")
    {
        bail!("invalid document format or character limit");
    }
    let bytes = match (&a.path, &a.url) {
        (Some(path), None) => policy.read_file(Path::new(path))?,
        (None, Some(url)) => {
            let response = http.get(url).await?;
            if !(200..300).contains(&response.status) {
                bail!("document upstream returned status {}", response.status);
            }
            response.body
        }
        _ => bail!("provide exactly one absolute path or URL"),
    };
    let text = bounded_utf8(bytes)?;
    let scan = missy_security::inspect_untrusted(&text);
    if scan.quarantined {
        return Ok(evidence(
            &text,
            json!({"content":scan.text,"security":scan}),
        ));
    }
    let content = match a.format.as_str() {
        "html" => visible_text(Html::parse_document(&text).root_element()),
        "json" => {
            let value: Value =
                serde_json::from_str(&text).context("document is not bounded JSON")?;
            serde_json::to_string_pretty(&value)?
        }
        _ => text.clone(),
    };
    let truncated = content.chars().count() > a.max_chars;
    let content = content.chars().take(a.max_chars).collect::<String>();
    Ok(evidence(
        &text,
        json!({"content":content,"format":a.format,"truncated":truncated,"source_sha256":scan.sha256}),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binary_pdf_archive_and_parser_limit_fail_explicitly() {
        for bytes in [
            b"%PDF-1.7\nsynthetic".to_vec(),
            b"PK\x03\x04synthetic".to_vec(),
            b"a\0b".to_vec(),
            vec![b'x'; PARSE_LIMIT + 1],
            vec![0xff],
        ] {
            assert!(bounded_utf8(bytes).is_err());
        }
    }
    #[test]
    fn result_links_cannot_be_executable_or_authenticated() {
        for href in [
            "javascript:alert(1)",
            "file:///etc/passwd",
            "https://user:pass@example.com/",
            "//duckduckgo.com/l/?uddg=file%3A%2F%2F%2Fetc%2Fpasswd",
        ] {
            assert!(result_url(href).is_none(), "{href}");
        }
        assert_eq!(
            result_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F"),
            Some("https://example.com/".into())
        );
    }
    #[test]
    fn html_is_data_not_execution() {
        let doc = Html::parse_document(
            "<h1>Title &amp; text</h1><script>secret script</script><style>hidden</style><div hidden>invisible</div><p>Body</p>",
        );
        assert_eq!(visible_text(doc.root_element()), "Title & text Body");
        let parsed=parse_search(r#"<div class="result"><a class="result__a" href="https://example.com/">Example</a><div class="result__snippet">A &amp; B</div></div>"#,1).unwrap();
        assert_eq!(parsed["results"][0]["snippet"], "A & B");
        assert!(parse_search("<html>prove you are human</html>", 5).is_err());
    }
}
