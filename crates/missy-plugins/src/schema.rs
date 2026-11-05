use anyhow::{Result, ensure};
use serde_json::Value;
use std::collections::BTreeSet;

/// Intentionally small closed JSON-schema subset. Reject unsupported vocabulary,
/// all references/URIs, nested objects, arrays, regex engines and remote loading.
pub fn validate(s: &Value) -> Result<()> {
    let obj = s
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("schema must be object"))?;
    ensure!(
        obj.keys().all(
            |k| ["type", "properties", "required", "additionalProperties"].contains(&k.as_str())
        ),
        "unsupported schema keyword or reference"
    );
    ensure!(
        s["type"] == "object" && s["additionalProperties"] == false,
        "schema root must be closed object"
    );
    let props = s["properties"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("schema properties required"))?;
    ensure!(props.len() <= 32, "too many properties");
    let required = s["required"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("schema required array required"))?;
    let mut seen = BTreeSet::new();
    for r in required {
        let key = r
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("required must be strings"))?;
        ensure!(
            props.contains_key(key) && seen.insert(key),
            "invalid/duplicate required property"
        );
    }
    for (key, p) in props {
        ensure!(
            !key.is_empty()
                && key.len() <= 48
                && key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'),
            "invalid property name"
        );
        let p = p
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("property schema required"))?;
        ensure!(
            p.keys().all(|k| [
                "type",
                "description",
                "enum",
                "minLength",
                "maxLength",
                "minimum",
                "maximum"
            ]
            .contains(&k.as_str())),
            "unsupported property keyword/reference"
        );
        let t = p.get("type").and_then(Value::as_str).unwrap_or("");
        ensure!(
            ["string", "number", "integer"].contains(&t),
            "only scalar string/number/integer properties"
        );
        if let Some(d) = p.get("description") {
            ensure!(
                d.as_str().is_some_and(|s| s.len() <= 4096),
                "invalid description"
            );
        }
        for k in ["minLength", "maxLength"] {
            if let Some(v) = p.get(k) {
                ensure!(
                    t == "string" && v.as_u64().is_some_and(|n| n <= 8192),
                    "invalid length bound"
                );
            }
        }
        for k in ["minimum", "maximum"] {
            if let Some(v) = p.get(k) {
                ensure!(
                    t != "string" && v.as_f64().is_some_and(f64::is_finite),
                    "invalid numeric bound"
                );
            }
        }
        if let Some(v) = p.get("enum") {
            let a = v
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("enum must be array"))?;
            ensure!(!a.is_empty() && a.len() <= 64, "enum outside bounds");
            for v in a {
                scalar(t, v)?;
            }
        }
    }
    Ok(())
}
fn scalar(t: &str, v: &Value) -> Result<()> {
    ensure!(
        match t {
            "string" => v
                .as_str()
                .is_some_and(|s| s.len() <= 8192 && !s.contains('\0')),
            "integer" => v.is_i64() || v.is_u64(),
            "number" => v.as_f64().is_some_and(f64::is_finite),
            _ => false,
        },
        "argument type/size mismatch"
    );
    Ok(())
}
pub fn arguments(s: &Value, args: &Value) -> Result<()> {
    let a = args
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("arguments must be closed object"))?;
    let props = s["properties"].as_object().unwrap();
    ensure!(a.keys().all(|k| props.contains_key(k)), "unknown argument");
    for key in s["required"].as_array().unwrap() {
        ensure!(
            a.contains_key(key.as_str().unwrap()),
            "required argument missing"
        );
    }
    for (key, v) in a {
        let p = &props[key];
        scalar(p["type"].as_str().unwrap(), v)?;
        if let Some(en) = p.get("enum") {
            ensure!(en.as_array().unwrap().contains(v), "argument not in enum");
        }
        if let Some(text) = v.as_str() {
            let n = text.chars().count() as u64;
            ensure!(
                n >= p["minLength"].as_u64().unwrap_or(0)
                    && n <= p["maxLength"].as_u64().unwrap_or(8192),
                "string length outside bounds"
            );
        } else {
            let n = v.as_f64().unwrap();
            ensure!(
                n >= p["minimum"].as_f64().unwrap_or(f64::NEG_INFINITY)
                    && n <= p["maximum"].as_f64().unwrap_or(f64::INFINITY),
                "number outside bounds"
            );
        }
    }
    Ok(())
}
