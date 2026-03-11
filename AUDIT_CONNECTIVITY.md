# Connectivity Audit

**Project:** Missy AI Agent Framework
**Audit Date:** 2026-03-11
**Scope:** All outbound network connections the agent may initiate, policy
enforcement verification, and recommended allow-list configurations per
provider.

---

## Overview

Missy's default network policy is `default_deny=True` with empty allow-lists.
No outbound connection can be made until an operator explicitly adds entries to
`allowed_domains`, `allowed_hosts`, or `allowed_cidrs` in the YAML
configuration.

All outbound HTTP requests flow through `missy.gateway.client.PolicyHttpClient`,
which extracts the hostname from the URL and calls
`NetworkPolicyEngine.check_host` before the request is sent.  A
`PolicyViolationError` aborts the request; the underlying HTTP client is never
opened.

---

## Outbound Network Connections

### AI Provider Endpoints

| Provider | Hostname | Protocol | Port | Purpose |
|----------|----------|----------|------|---------|
| Anthropic | `api.anthropic.com` | HTTPS | 443 | Chat completions, streaming |
| OpenAI | `api.openai.com` | HTTPS | 443 | Chat completions, streaming |
| Ollama (local) | `127.0.0.1` or `localhost` | HTTP | 11434 | Local model inference |

### Optional / Plugin-Driven Connections

| When | Hostname | Notes |
|------|----------|-------|
| Plugin with `network=True` | Plugin-declared `allowed_hosts` | Only reachable if plugin policy passes |
| Tool making external calls | Tool-specific, operator-configured | No built-in tools make network calls (calculator is offline) |
| Scheduler jobs | Job-specific | If a scheduled job calls the agent, it follows the same policy |

### Connections Missy Does NOT Make

- No telemetry or analytics beacons.
- No package auto-update checks.
- No license validation to an external server.
- No DNS lookups other than those triggered by `NetworkPolicyEngine` step 5
  (DNS fallback) when a hostname is not matched by any allow-list entry.

---

## Policy Enforcement Verification

The enforcement chain for every outbound request is:

```
Agent / Tool code
      |
      v
PolicyHttpClient.get() / post()
      |
      |-- URL parsed -> hostname extracted
      |
      v
NetworkPolicyEngine.check_host(hostname)
      |
      |-- default_deny=True? -> proceed to checks
      |
      |-- IP address? -> CIDR list only (no DNS)
      |-- Exact host in allowed_hosts? -> allow
      |-- Domain match in allowed_domains? -> allow
      |-- DNS resolution -> CIDR re-check
      |-- None matched -> PolicyViolationError raised
      |
      v
AuditEvent emitted (allow or deny)
      |
      v
HTTP request sent (only on allow)
```

Every step is covered by unit and integration tests.  The integration test
`TestNetworkPolicyEnforcement` in
`tests/integration/test_policy_enforcement.py` runs against real
`NetworkPolicyEngine` instances and verifies:

- `evil.com` is denied when `default_deny=True` and no allow-list entries exist.
- `8.8.8.8` is denied when only `10.0.0.0/8` is in `allowed_cidrs`.
- `10.0.0.1` is allowed when `10.0.0.0/8` is in `allowed_cidrs`.
- `api.github.com` is allowed when `*.github.com` is in `allowed_domains`.

---

## Example Allow-List Configurations

### Anthropic Claude Only

```yaml
network:
  default_deny: true
  allowed_domains:
    - "api.anthropic.com"
  allowed_cidrs: []
  allowed_hosts: []
```

This configuration permits only the Anthropic API endpoint.  No other outbound
connection is possible.

---

### OpenAI Only

```yaml
network:
  default_deny: true
  allowed_domains:
    - "api.openai.com"
  allowed_cidrs: []
  allowed_hosts: []
```

---

### Ollama Local (Loopback Only)

```yaml
network:
  default_deny: true
  allowed_cidrs:
    - "127.0.0.0/8"
  allowed_domains: []
  allowed_hosts: []
```

Using a CIDR block for loopback (`127.0.0.0/8`) covers both `127.0.0.1` and
`::1` (if additionally adding `::1/128`).  Because Ollama runs on the local
machine, no external network traffic is possible under this configuration.

---

### Anthropic + Ollama Fallback

```yaml
network:
  default_deny: true
  allowed_domains:
    - "api.anthropic.com"
  allowed_cidrs:
    - "127.0.0.0/8"
  allowed_hosts: []
```

---

### Multi-Provider (Anthropic + OpenAI)

```yaml
network:
  default_deny: true
  allowed_domains:
    - "api.anthropic.com"
    - "api.openai.com"
  allowed_cidrs: []
  allowed_hosts: []
```

---

### Private Kubernetes Cluster (Internal CIDR)

```yaml
network:
  default_deny: true
  allowed_cidrs:
    - "10.0.0.0/8"
    - "172.16.0.0/12"
  allowed_domains: []
  allowed_hosts: []
```

This permits connections to RFC-1918 private address space only.  Use this
when the AI provider is deployed on a private internal network.

---

## Network Flow Diagram

```
  +-----------+       HTTP POST       +-----------------------+
  |           |  ------------------>  |                       |
  |  Agent    |                       |  PolicyHttpClient     |
  |  Runtime  |                       |                       |
  |           |                       |  1. Parse URL         |
  +-----------+                       |  2. Extract hostname  |
                                      |  3. check_host()      |
                                      |        |              |
                                      |        v              |
                                      |  NetworkPolicyEngine  |
                                      |  +-----------------+  |
                                      |  | default_deny?   |  |
                                      |  | CIDR check?     |  |
                                      |  | host check?     |  |
                                      |  | domain check?   |  |
                                      |  | DNS + CIDR?     |  |
                                      |  +-----------------+  |
                                      |        |              |
                          [DENY] <----+--------+--------> [ALLOW]
                             |                               |
                    PolicyViolationError           HTTP request sent
                    AuditEvent(deny)               AuditEvent(allow)
                                                        |
                                                        v
                                          +-----------------------+
                                          |  AI Provider          |
                                          |  api.anthropic.com    |
                                          |  api.openai.com       |
                                          |  localhost:11434      |
                                          +-----------------------+
```

---

## Default Deny Verification

The following sequence confirms that the default configuration blocks all
network access:

```python
from missy.config.settings import get_default_config
from missy.policy.network import NetworkPolicyEngine
from missy.core.exceptions import PolicyViolationError

config = get_default_config()
engine = NetworkPolicyEngine(config.network)

# Every host is denied under the default configuration.
for host in ["api.anthropic.com", "api.openai.com", "8.8.8.8", "evil.com"]:
    try:
        engine.check_host(host)
        print(f"FAIL: {host} was allowed — expected denial")
    except PolicyViolationError:
        print(f"OK: {host} denied by default policy")
```

Expected output:

```
OK: api.anthropic.com denied by default policy
OK: api.openai.com denied by default policy
OK: 8.8.8.8 denied by default policy
OK: evil.com denied by default policy
```

This can be run as a manual smoke test or added to a deployment verification
script.
