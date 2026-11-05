//! Opt-in Discord voice, sharing the application's authenticated text Gateway.
//! No bot login, hardware, subprocess, file recording or implicit network access.
//! Wire DAVE/MLS uses pinned official libdave, Opus uses maintained libopus bindings.
mod audio;
mod protocol;
mod service;
mod tools;
pub use tools::{DiscordVoiceSayTools, VoiceApprovalGate};
#[cfg(test)]
mod tests;
pub use service::*;

use anyhow::{Result, ensure};
use missy_core::Scope;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    net::{IpAddr, SocketAddr},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VoiceMode {
    Normal,
    NoTools,
    SafeChat,
}

/// Operator-owned mapping, never constructed from transcripts or audio metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceBinding {
    pub guild_id: u64,
    pub channel_id: u64,
    pub user_id: u64,
    pub consent: bool,
    pub mode: VoiceMode,
    pub scope: Scope,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscordVoiceConfig {
    pub native_dave: missy_dave::NativeDaveConfig,
    pub enabled: bool,
    pub bot_user_id: u64,
    pub bindings: Vec<VoiceBinding>,
    /// Exact Gateway-provided hostnames. PolicyHttpClient additionally requires
    /// their exact HTTPS origins. Wildcards and arbitrary voice hostnames denied.
    pub endpoint_hosts: BTreeSet<String>,
    pub udp_port_min: u16,
    pub udp_port_max: u16,
    /// For a private Discord-compatible test server ONLY. Default must be false.
    pub allow_private_udp: bool,
    pub max_utterance_seconds: u16,
    pub silence_ms: u16,
    pub max_users: u16,
    pub max_reply_seconds: u16,
}
impl DiscordVoiceConfig {
    pub fn validate(&self) -> Result<()> {
        self.native_dave.validate()?;
        ensure!(
            self.enabled && self.bot_user_id != 0,
            "Discord voice disabled or missing bot identity"
        );
        ensure!(
            !self.endpoint_hosts.is_empty() && self.endpoint_hosts.len() <= 64,
            "explicit voice endpoint allowlist required"
        );
        for host in &self.endpoint_hosts {
            ensure!(
                !host.is_empty()
                    && host.len() <= 253
                    && host
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b".-".contains(&b)),
                "invalid voice host"
            );
        }
        ensure!(
            self.udp_port_min > 0 && self.udp_port_min <= self.udp_port_max,
            "invalid UDP range"
        );
        ensure!(
            (1..=60).contains(&self.max_utterance_seconds)
                && (200..=5000).contains(&self.silence_ms),
            "invalid speech bounds"
        );
        ensure!(
            (1..=32).contains(&self.max_users) && (1..=120).contains(&self.max_reply_seconds),
            "invalid voice resource bounds"
        );
        ensure!(
            !self.bindings.is_empty() && self.bindings.len() <= 1024,
            "explicit voice bindings required"
        );
        let mut seen = BTreeSet::new();
        for b in &self.bindings {
            b.scope.validate()?;
            ensure!(
                b.guild_id != 0
                    && b.channel_id != 0
                    && b.user_id != 0
                    && b.user_id != self.bot_user_id,
                "invalid voice identity"
            );
            ensure!(
                seen.insert((b.guild_id, b.channel_id, b.user_id)),
                "duplicate voice binding"
            );
        }
        Ok(())
    }
    pub(crate) fn binding(&self, guild: u64, channel: u64, user: u64) -> Option<&VoiceBinding> {
        self.bindings.iter().find(|b| {
            b.guild_id == guild && b.channel_id == channel && b.user_id == user && b.consent
        })
    }
    pub(crate) fn udp_address(&self, ip: &str, port: u16) -> Result<SocketAddr> {
        ensure!(
            (self.udp_port_min..=self.udp_port_max).contains(&port),
            "voice UDP port denied"
        );
        let ip: IpAddr = ip
            .parse()
            .map_err(|_| anyhow::anyhow!("voice Ready requires literal IP"))?;
        let public = match ip {
            IpAddr::V4(v) => {
                !v.is_private()
                    && !v.is_loopback()
                    && !v.is_link_local()
                    && !v.is_unspecified()
                    && !v.is_multicast()
                    && !v.is_broadcast()
                    && !v.is_documentation()
                    && v.octets()[0] != 0
                    && v.octets()[0] < 224
                    && !(v.octets()[0] == 100 && (64..128).contains(&v.octets()[1]))
                    && !(v.octets()[0] == 198 && (18..20).contains(&v.octets()[1]))
            }
            IpAddr::V6(v) => {
                v.segments()[0] & 0xe000 == 0x2000
                    && !v.is_loopback()
                    && !v.is_multicast()
                    && v.to_ipv4_mapped().is_none()
                    && !(v.segments()[0] == 0x2001 && v.segments()[1] == 0x0db8)
            }
        };
        ensure!(
            public || self.allow_private_udp,
            "private/reserved voice UDP IP denied"
        );
        Ok(SocketAddr::new(ip, port))
    }
}
