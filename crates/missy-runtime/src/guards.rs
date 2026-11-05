//! Advisory action-only response guards. These never authorize execution.
use regex::Regex;
use std::sync::LazyLock;
fn pattern(value: &str) -> Regex {
    Regex::new(value).expect("static guard regex")
}
static ACTION: LazyLock<Regex> = LazyLock::new(|| {
    pattern(
        r"(?i)\b(run|execute|check|inspect|fetch|search|download|upload|install|deploy|restart|start|stop|update|change|edit|modify|fix|test|verify|create|delete|remove|send|post|save|generate|produce|render|export|compile|take (a |the )?(photo|screenshot))\b",
    )
});
static ADVICE: LazyLock<Regex> = LazyLock::new(|| {
    pattern(
        r"(?i)^\s*(please\s+)?(how (would|can|do|should)|what (would|does)|explain|describe|draft|write (me )?(a |the )?(shell |bash )?(script|command|instructions))\b|\b(don't (run|execute)|do not (run|execute)|without (running|executing))\b",
    )
});
static THEN: LazyLock<Regex> = LazyLock::new(|| {
    pattern(
        r"(?i)\b(and|then|also)\s+(please\s+)?(run|execute|deploy|install|create|save|send|post|start|restart)\b",
    )
});
static CLAIM: LazyLock<Regex> = LazyLock::new(|| {
    pattern(
        r"(?i)\bI('ve| have)?\s+(ran|executed|checked|performed|looked at|reviewed|inspected|verified|confirmed|tested|scanned|queried)\b[^.!?\n]*|\b(generated|posted|created|saved|uploaded|deployed|installed|deleted|removed|wrote|downloaded)\b[^.!?\n]{0,60}\b(image|file|script|server|container|process|document|report|screenshot)\b|\b(according to (the )?(logs?|output|results?|data|metrics)|based on (the )?(output|logs?|results?|metrics))\b[^.!?\n]*|[├└]──",
    )
});
static PROMISE: LazyLock<Regex> = LazyLock::new(|| {
    pattern(
        r"(?i)\b(I'll|I will|let me|I'm going to|I am going to)\s+(run|execute|check|inspect|fetch|search|download|upload|install|deploy|restart|start|stop|update|change|edit|modify|fix|test|verify|create|delete|remove|send|post|save|generate|produce|render|export|compile)\b[^.!?\n]*|^\s*(Working on it|Running now|Starting now|Generating now)\b",
    )
});
static SELF_CONTAINED: LazyLock<Regex> = LazyLock::new(|| {
    pattern(
        r"(?i)\b(you('ve| have)? (pasted|shared|provided|sent|gave|attached|included|quoted)|above|below|(the|my) (math|arithmetic|calculation|reasoning|logic|answer))\b",
    )
});
static FAILURE: LazyLock<Regex> = LazyLock::new(|| {
    pattern(
        r"(?i)\b(could(n't| not)|can(not|'t)|unable to)\s+(complete|finish|generate|create|produce|perform)|\b(task|request) remains? incomplete",
    )
});
fn relevant(claim: &str, successful: &[&str]) -> bool {
    let claim = claim.to_ascii_lowercase();
    let groups: &[(&[&str], &[&str])] = &[
        (
            &["shell", "exec", "command"],
            &[
                "ran",
                "executed",
                "command",
                "script",
                "log",
                "output",
                "server",
                "container",
                "process",
                "deployed",
                "installed",
            ],
        ),
        (
            &["file", "read", "write", "list"],
            &[
                "file",
                "directory",
                "script",
                "document",
                "report",
                "config",
                "reviewed",
                "saved",
                "written",
            ],
        ),
        (
            &["image_generate", "screenshot", "webcam"],
            &["image", "picture", "photo", "screenshot", "render"],
        ),
        (
            &["discord", "upload", "message", "post"],
            &[
                "discord",
                "uploaded",
                "posted",
                "sent",
                "message",
                "attachment",
            ],
        ),
        (
            &["web", "http", "browser", "search", "fetch"],
            &[
                "website",
                "page",
                "url",
                "searched",
                "downloaded",
                "fetched",
                "source",
                "dashboard",
                "metrics",
                "data",
                "results",
            ],
        ),
    ];
    successful.iter().any(|name| {
        let name = name.to_ascii_lowercase();
        groups.iter().any(|(tools, words)| {
            tools.iter().any(|t| name.contains(t)) && words.iter().any(|w| claim.contains(w))
        }) || name
            .split(['_', '-'])
            .any(|part| part.len() >= 4 && claim.contains(part))
    })
}
pub(crate) fn check(
    request: &str,
    response: &str,
    successful: &[&str],
    failed: bool,
) -> Option<&'static str> {
    if !ACTION.is_match(request)
        || (ADVICE.is_match(request) && !THEN.is_match(request))
        || FAILURE.is_match(response)
    {
        return None;
    }
    if failed
        && ["failed", "timed out", "error"]
            .iter()
            .any(|s| response.to_lowercase().contains(s))
    {
        return None;
    }
    for sentence in response.split(['.', '!', '?', '\n']) {
        if SELF_CONTAINED.is_match(sentence) {
            continue;
        }
        if CLAIM
            .find_iter(sentence)
            .any(|m| !relevant(m.as_str(), successful))
        {
            return Some("action claim lacks relevant successful tool evidence");
        }
        if PROMISE
            .find_iter(sentence)
            .any(|m| !relevant(m.as_str(), successful))
        {
            return Some("promise of action without relevant successful tool evidence");
        }
    }
    None
}
