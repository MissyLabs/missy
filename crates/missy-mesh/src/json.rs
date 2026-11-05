//! Deliberately restricted canonical JSON: sorted keys, integer numbers only,
//! duplicate-key rejection, depth/node/byte limits. Not an RFC8785 claim.
use anyhow::{Result, ensure};
use serde::{
    Deserialize,
    de::{self, MapAccess, SeqAccess, Visitor},
};
use serde_json::Value;
use std::{collections::BTreeMap, fmt};

struct Strict(Value);
impl<'de> Deserialize<'de> for Strict {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Strict;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("bounded integer JSON")
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Strict, E> {
                Ok(Strict(Value::Bool(v)))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Strict, E> {
                Ok(Strict(v.into()))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Strict, E> {
                Ok(Strict(v.into()))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Strict, E> {
                Ok(Strict(v.into()))
            }
            fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Strict, E> {
                Ok(Strict(v.into()))
            }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Strict, E> {
                Ok(Strict(Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Strict, A::Error> {
                let mut out = Vec::new();
                while let Some(Strict(v)) = a.next_element()? {
                    if out.len() >= 65536 {
                        return Err(de::Error::custom("too many elements"));
                    }
                    out.push(v);
                }
                Ok(Strict(Value::Array(out)))
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Strict, A::Error> {
                let mut out = serde_json::Map::new();
                while let Some(k) = a.next_key::<String>()? {
                    if out.len() >= 65536 || out.contains_key(&k) {
                        return Err(de::Error::custom("duplicate key or too many members"));
                    }
                    let Strict(v) = a.next_value()?;
                    out.insert(k, v);
                }
                Ok(Strict(Value::Object(out)))
            }
        }
        d.deserialize_any(V)
    }
}
pub(crate) fn parse(bytes: &[u8]) -> Result<Value> {
    let Strict(value) = serde_json::from_slice(bytes)?;
    canonical(&value)?;
    Ok(value)
}
pub(crate) fn parse_private(bytes: &[u8]) -> Result<Value> {
    ensure!(
        bytes.len() <= crate::private::STATE_LIMIT,
        "private mesh JSON too large"
    );
    let Strict(value) = serde_json::from_slice(bytes)?;
    // Snapshots contain many individually bounded records. The serde_json
    // recursion limit still applies; wire data always uses parse instead.
    Ok(value)
}
pub(crate) fn canonical(value: &Value) -> Result<Vec<u8>> {
    fn normalize(v: &Value, depth: usize, nodes: &mut usize) -> Result<Value> {
        *nodes += 1;
        ensure!(
            depth <= 16 && *nodes <= 4096,
            "JSON depth/node bound exceeded"
        );
        Ok(match v {
            Value::Number(n) => {
                ensure!(
                    n.is_i64() || n.is_u64(),
                    "floating point JSON is not canonical mesh data"
                );
                v.clone()
            }
            Value::Object(map) => {
                let sorted: BTreeMap<_, _> = map.iter().collect();
                let mut out = serde_json::Map::new();
                for (k, v) in sorted {
                    out.insert(k.clone(), normalize(v, depth + 1, nodes)?);
                }
                Value::Object(out)
            }
            Value::Array(a) => Value::Array(
                a.iter()
                    .map(|v| normalize(v, depth + 1, nodes))
                    .collect::<Result<_>>()?,
            ),
            _ => v.clone(),
        })
    }
    Ok(serde_json::to_vec(&normalize(value, 0, &mut 0)?)?)
}
