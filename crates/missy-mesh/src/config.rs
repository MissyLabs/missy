use crate::{
    Capability,
    identity::{peer_id, public_key},
    private::{STATE_LIMIT, split},
};
use anyhow::{Context, Result, ensure};
use missy_core::{PolicyConfig, Scope};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    net::SocketAddr,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Grant {
    pub scope: Scope,
    pub capabilities: BTreeSet<Capability>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Peer {
    /// Canonical lowercase hex Ed25519 public key, never secret key material.
    pub public_key: String,
    /// Fixed ingress URL. No endpoint is ever accepted from a signed payload.
    pub endpoint: Option<String>,
    pub grants: Vec<Grant>,
}
impl Peer {
    pub fn id(&self) -> Result<String> {
        Ok(peer_id(&public_key(&self.public_key)?))
    }
    pub fn allows(&self, scope: &Scope, capability: Capability) -> bool {
        self.grants
            .iter()
            .any(|g| &g.scope == scope && g.capabilities.contains(&capability))
    }
    pub(crate) fn validate(&self) -> Result<()> {
        self.id()?;
        ensure!(self.grants.len() <= 128, "peer scope limit exceeded");
        let mut seen = BTreeSet::new();
        for grant in &self.grants {
            grant.scope.validate()?;
            ensure!(
                seen.insert(serde_json::to_string(&grant.scope)?),
                "duplicate peer scope"
            );
        }
        if let Some(endpoint) = &self.endpoint {
            let url = url::Url::parse(endpoint)?;
            ensure!(
                endpoint.len() <= 2048
                    && url.host_str().is_some()
                    && url.username().is_empty()
                    && url.password().is_none()
                    && url.query().is_none()
                    && url.fragment().is_none()
                    && url.path() == "/mesh/envelopes",
                "invalid fixed mesh endpoint"
            );
            ensure!(
                url.scheme() == "https"
                    || (url.scheme() == "http"
                        && url
                            .host_str()
                            .and_then(|h| h.parse::<std::net::IpAddr>().ok())
                            .is_some_and(|ip| ip.is_loopback())),
                "mesh transport requires HTTPS except numeric loopback fixtures"
            );
        }
        Ok(())
    }
}

/// Separate, private configuration. No discovery, default state directory,
/// implicit identity creation, wildcard Scope, or trust-on-first-use.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshConfig {
    pub version: u32,
    pub identity_path: PathBuf,
    pub state_dir: PathBuf,
    /// Operator-authorized upper bound, applied even to persisted peers.
    pub authorized_peers: Vec<Peer>,
    pub listen: SocketAddr,
    pub http_policy: PolicyConfig,
}
impl MeshConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let (dir, name) = split(path.as_ref())?;
        let bytes = dir
            .read(&name, STATE_LIMIT)?
            .context("private mesh config missing")?;
        let config: Self = serde_json::from_value(crate::json::parse_private(&bytes)?)?;
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(self.version == 1, "unsupported mesh configuration");
        ensure!(
            self.identity_path.is_absolute() && self.state_dir.is_absolute(),
            "explicit absolute mesh paths required"
        );
        ensure!(
            self.listen.ip().is_loopback(),
            "mesh inbound bind must be loopback"
        );
        ensure!(self.authorized_peers.len() <= 256, "peer limit exceeded");
        let mut ids = BTreeSet::new();
        for peer in &self.authorized_peers {
            peer.validate()?;
            ensure!(ids.insert(peer.id()?), "duplicate authorized public key");
        }
        missy_policy::PolicyHttpClient::new(self.http_policy.clone())?;
        Ok(())
    }
    pub(crate) fn permits(&self, peer: &Peer) -> Result<bool> {
        let id = peer.id()?;
        Ok(self.authorized_peers.iter().any(|allowed| {
            allowed.id().ok().as_ref() == Some(&id)
                && peer.endpoint == allowed.endpoint
                && peer.grants.iter().all(|grant| {
                    grant
                        .capabilities
                        .iter()
                        .all(|cap| allowed.allows(&grant.scope, *cap))
                })
        }))
    }
}
