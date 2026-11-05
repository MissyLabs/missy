use aes_gcm::{
    Aes256Gcm, KeyInit,
    aead::{Aead, Payload},
};
use anyhow::{Result, bail, ensure};
use missy_dave::{self as davey, DaveSession, NativeDaveConfig, ProposalsOperationType};
use serde_json::{Value, json};
use std::{
    collections::{BTreeSet, HashMap},
    num::NonZeroU16,
};
use tokio::time::Instant;
use tokio_tungstenite::tungstenite::Message;

pub(crate) const MODE: &str = "aead_aes256_gcm_rtpsize";
pub(crate) const SILENCE: &[u8] = &[0xf8, 0xff, 0xfe];
pub(crate) fn message(op: u8, d: Value) -> Message {
    Message::Text(json!({"op":op,"d":d}).to_string().into())
}
pub(crate) fn binary(op: u8, payload: &[u8]) -> Message {
    let mut bytes = vec![op];
    bytes.extend_from_slice(payload);
    Message::Binary(bytes.into())
}

#[derive(Default)]
pub(crate) struct Heartbeat {
    pub interval_ms: Option<u64>,
    pending: Option<u64>,
    counter: u64,
    last: Option<Instant>,
}
impl Heartbeat {
    pub fn hello(&mut self, interval: u64) -> Result<()> {
        ensure!(
            (100..=120_000).contains(&interval),
            "invalid voice heartbeat interval"
        );
        self.interval_ms = Some(interval);
        self.pending = None;
        self.last = None;
        Ok(())
    }
    pub fn tick(&mut self, seq: i64) -> Result<Option<Message>> {
        let Some(ms) = self.interval_ms else {
            return Ok(None);
        };
        if self
            .last
            .is_some_and(|t| t.elapsed().as_millis() < ms as u128)
        {
            return Ok(None);
        }
        ensure!(self.pending.is_none(), "voice heartbeat stale");
        self.counter += 1;
        self.pending = Some(self.counter);
        self.last = Some(Instant::now());
        Ok(Some(message(3, json!({"t":self.counter,"seq_ack":seq}))))
    }
    pub fn ack(&mut self, d: &Value) -> Result<()> {
        let nonce = d.as_u64().or_else(|| d["t"].as_u64());
        ensure!(
            nonce.is_some() && nonce == self.pending,
            "invalid voice heartbeat ACK"
        );
        self.pending = None;
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct Replay {
    high: Option<u32>,
    bits: u64,
}
impl Replay {
    pub fn accept(&mut self, value: u32) -> bool {
        match self.high {
            None => {
                self.high = Some(value);
                self.bits = 1;
                true
            }
            Some(high) if value > high => {
                self.bits = if value - high >= 64 {
                    1
                } else {
                    (self.bits << (value - high)) | 1
                };
                self.high = Some(value);
                true
            }
            Some(high) => {
                let delta = high - value;
                if delta >= 64 || self.bits & (1 << delta) != 0 {
                    false
                } else {
                    self.bits |= 1 << delta;
                    true
                }
            }
        }
    }
}

pub(crate) struct RtpCipher {
    cipher: Aes256Gcm,
    pub ssrc: u32,
    sequence: u16,
    timestamp: u32,
    nonce: u32,
    replay: HashMap<u32, Replay>,
}
impl RtpCipher {
    pub fn new(key: &[u8], ssrc: u32) -> Result<Self> {
        Ok(Self {
            cipher: Aes256Gcm::new_from_slice(key)
                .map_err(|_| anyhow::anyhow!("invalid transport key"))?,
            ssrc,
            sequence: 0,
            timestamp: 0,
            nonce: 0,
            replay: HashMap::new(),
        })
    }
    pub fn encode(&mut self, opus: &[u8]) -> Result<Vec<u8>> {
        ensure!(
            self.nonce < u32::MAX,
            "voice nonce exhausted; new session required"
        );
        let mut header = vec![0x80, 0x78];
        header.extend_from_slice(&self.sequence.to_be_bytes());
        header.extend_from_slice(&self.timestamp.to_be_bytes());
        header.extend_from_slice(&self.ssrc.to_be_bytes());
        self.nonce += 1;
        let mut nonce = [0; 12];
        nonce[..4].copy_from_slice(&self.nonce.to_be_bytes());
        let encrypted = self
            .cipher
            .encrypt(
                (&nonce).into(),
                Payload {
                    msg: opus,
                    aad: &header,
                },
            )
            .map_err(|_| anyhow::anyhow!("voice encryption failed"))?;
        header.extend_from_slice(&encrypted);
        header.extend_from_slice(&nonce[..4]);
        self.sequence = self.sequence.wrapping_add(1);
        self.timestamp = self.timestamp.wrapping_add(960);
        Ok(header)
    }
    pub fn decode(&mut self, packet: &[u8]) -> Result<(u32, Vec<u8>)> {
        ensure!(
            (32..=4096).contains(&packet.len()) && packet[0] >> 6 == 2 && packet[1] & 0x7f == 0x78,
            "invalid voice RTP"
        );
        let ssrc = u32::from_be_bytes(packet[8..12].try_into()?);
        let mut header_len = 12 + (packet[0] as usize & 15) * 4;
        ensure!(packet.len() >= header_len + 20, "truncated RTP header");
        let extension = packet[0] & 0x10 != 0;
        if extension {
            header_len += 4;
            ensure!(packet.len() >= header_len + 20, "truncated RTP extension");
        }
        let mut nonce = [0; 12];
        nonce[..4].copy_from_slice(&packet[packet.len() - 4..]);
        let counter = u32::from_be_bytes(nonce[..4].try_into()?);
        let mut plain = self
            .cipher
            .decrypt(
                (&nonce).into(),
                Payload {
                    msg: &packet[header_len..packet.len() - 4],
                    aad: &packet[..header_len],
                },
            )
            .map_err(|_| anyhow::anyhow!("voice authentication failed"))?;
        ensure!(
            self.replay.contains_key(&ssrc) || self.replay.len() < 64,
            "too many RTP senders"
        );
        ensure!(
            self.replay.entry(ssrc).or_default().accept(counter),
            "replayed voice packet"
        );
        if extension {
            let words = u16::from_be_bytes(packet[header_len - 2..header_len].try_into()?) as usize;
            ensure!(words * 4 <= plain.len(), "invalid RTP extension length");
            plain.drain(..words * 4);
        }
        if packet[0] & 0x20 != 0 {
            let padding = *plain
                .last()
                .ok_or_else(|| anyhow::anyhow!("missing RTP padding"))?
                as usize;
            ensure!(padding > 0 && padding <= plain.len(), "invalid RTP padding");
            plain.truncate(plain.len() - padding);
        }
        Ok((ssrc, plain))
    }
    pub fn forget(&mut self, ssrc: u32) {
        self.replay.remove(&ssrc);
    }
}

pub(crate) struct Dave {
    pub session: DaveSession,
    pub recognized: BTreeSet<u64>,
    pub sequence: i64,
    user: u64,
    channel: u64,
}
impl Dave {
    pub fn new(config: &NativeDaveConfig, version: u16, user: u64, channel: u64) -> Result<Self> {
        ensure!(
            version == davey::DAVE_PROTOCOL_VERSION,
            "DAVE required; unsupported negotiated version"
        );
        let session = DaveSession::new(
            config,
            NonZeroU16::new(version).unwrap(),
            user,
            channel,
            None,
        )
        .map_err(|_| anyhow::anyhow!("DAVE initialization failed"))?;
        Ok(Self {
            session,
            recognized: BTreeSet::from([user]),
            sequence: -1,
            user,
            channel,
        })
    }
    pub fn key_package(&mut self) -> Result<Message> {
        let key = self
            .session
            .create_key_package()
            .map_err(|_| anyhow::anyhow!("DAVE key package failed"))?;
        Ok(binary(26, &key))
    }
    pub fn json(&mut self, op: u64, d: &Value) -> Result<Vec<Message>> {
        match op {
            11 => {
                let users = d["user_ids"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("invalid voice clients"))?;
                ensure!(
                    users.len() <= 1024 && self.recognized.len() + users.len() <= 2048,
                    "voice roster exceeds bound"
                );
                for u in users {
                    self.recognized.insert(id(u)?);
                }
            }
            13 => {
                self.recognized.remove(&id(&d["user_id"])?);
            }
            21 => {
                let version = u16::try_from(number(d, "protocol_version")?)?;
                ensure!(
                    version == davey::DAVE_PROTOCOL_VERSION,
                    "DAVE downgrade denied"
                );
                let transition = u16::try_from(number(d, "transition_id")?)?;
                self.session
                    .prepare_transition(transition, version)
                    .map_err(|_| anyhow::anyhow!("DAVE prepare failed"))?;
                if transition == 0 {
                    self.session
                        .execute_transition(0)
                        .map_err(|_| anyhow::anyhow!("DAVE execute failed"))?;
                } else {
                    return Ok(vec![message(
                        23,
                        json!({"transition_id":transition,"protocol_version":version}),
                    )]);
                }
            }
            22 => {
                self.session
                    .execute_transition(u16::try_from(number(d, "transition_id")?)?)
                    .map_err(|_| anyhow::anyhow!("DAVE transition failed"))?;
            }
            24 => {
                ensure!(
                    number(d, "protocol_version")? == davey::DAVE_PROTOCOL_VERSION as u64,
                    "DAVE epoch downgrade denied"
                );
                if number(d, "epoch")? == 1 {
                    self.session
                        .reinit(
                            NonZeroU16::new(davey::DAVE_PROTOCOL_VERSION).unwrap(),
                            self.user,
                            self.channel,
                            None,
                        )
                        .map_err(|_| anyhow::anyhow!("DAVE epoch initialization failed"))?;
                    return Ok(vec![self.key_package()?]);
                }
            }
            _ => {}
        }
        Ok(Vec::new())
    }
    pub fn binary(&mut self, bytes: &[u8]) -> Result<Vec<Message>> {
        // v8 server messages: sequence:u16, opcode:u8. Client messages: opcode:u8.
        ensure!(
            (3..=1_048_576).contains(&bytes.len()),
            "invalid DAVE message length"
        );
        self.sequence = u16::from_be_bytes(bytes[..2].try_into()?) as i64;
        let op = bytes[2];
        let body = &bytes[3..];
        let roster: Vec<u64> = self.recognized.iter().copied().collect();
        match op {
            25 => {
                self.session
                    .set_external_sender(body)
                    .map_err(|_| anyhow::anyhow!("invalid MLS external sender"))?;
            }
            27 => {
                ensure!(!body.is_empty(), "missing MLS proposal operation");
                let operation = match body[0] {
                    0 => ProposalsOperationType::APPEND,
                    1 => ProposalsOperationType::REVOKE,
                    _ => bail!("invalid MLS proposal operation"),
                };
                if let Some(cw) = self
                    .session
                    .process_proposals(operation, &body[1..], Some(&roster))
                    .map_err(|_| anyhow::anyhow!("MLS proposals rejected"))?
                {
                    let mut payload = cw.commit;
                    if let Some(welcome) = cw.welcome {
                        payload.extend_from_slice(&welcome);
                    }
                    return Ok(vec![binary(28, &payload)]);
                }
            }
            29 | 30 => {
                ensure!(body.len() >= 2, "missing MLS transition ID");
                let transition = u16::from_be_bytes(body[..2].try_into()?);
                let processed = if op == 29 {
                    self.session.process_commit(&body[2..]).map_err(|_| ())
                } else {
                    self.session
                        .process_welcome(&body[2..], Some(&roster))
                        .map(|_| true)
                        .map_err(|_| ())
                };
                match processed {
                    Ok(true) => {
                        self.session
                            .prepare_transition(transition, davey::DAVE_PROTOCOL_VERSION)
                            .map_err(|_| anyhow::anyhow!("DAVE ratchet prepare failed"))?;
                        if transition == 0 {
                            self.session
                                .execute_transition(0)
                                .map_err(|_| anyhow::anyhow!("DAVE ratchet execute failed"))?;
                        } else {
                            return Ok(vec![message(
                                23,
                                json!({"transition_id":transition,"protocol_version":davey::DAVE_PROTOCOL_VERSION}),
                            )]);
                        }
                    }
                    Ok(false) => {}
                    Err(()) => {
                        self.session
                            .reinit(
                                NonZeroU16::new(davey::DAVE_PROTOCOL_VERSION).unwrap(),
                                self.user,
                                self.channel,
                                None,
                            )
                            .map_err(|_| anyhow::anyhow!("DAVE reset failed"))?;
                        return Ok(vec![
                            message(31, json!({"transition_id":transition})),
                            self.key_package()?,
                        ]);
                    }
                }
            }
            _ => {}
        }
        Ok(Vec::new())
    }
}
pub(crate) fn number(d: &Value, key: &str) -> Result<u64> {
    d[key]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("invalid voice numeric field"))
}
pub(crate) fn id(d: &Value) -> Result<u64> {
    let v = d
        .as_str()
        .and_then(|v| v.parse().ok())
        .or_else(|| d.as_u64())
        .ok_or_else(|| anyhow::anyhow!("invalid voice user ID"))?;
    ensure!(v != 0, "zero voice ID");
    Ok(v)
}
pub(crate) fn discovery_request(ssrc: u32) -> [u8; 74] {
    let mut bytes = [0; 74];
    bytes[..2].copy_from_slice(&1u16.to_be_bytes());
    bytes[2..4].copy_from_slice(&70u16.to_be_bytes());
    bytes[4..8].copy_from_slice(&ssrc.to_be_bytes());
    bytes
}
pub(crate) fn discovery_response(bytes: &[u8], ssrc: u32) -> Result<(String, u16)> {
    ensure!(
        bytes.len() == 74
            && bytes[..2] == 2u16.to_be_bytes()
            && bytes[2..4] == 70u16.to_be_bytes()
            && bytes[4..8] == ssrc.to_be_bytes(),
        "invalid UDP discovery response"
    );
    let end = bytes[8..72]
        .iter()
        .position(|b| *b == 0)
        .ok_or_else(|| anyhow::anyhow!("unterminated discovery IP"))?;
    let ip = std::str::from_utf8(&bytes[8..8 + end])?;
    let _: std::net::IpAddr = ip.parse()?;
    let port = u16::from_be_bytes(bytes[72..74].try_into()?);
    ensure!(port != 0, "invalid discovery port");
    Ok((ip.into(), port))
}
