# Network Client (PolicyHTTPClient)

`PolicyHTTPClient` is the sole sanctioned path for outbound HTTP in the
Missy framework. It wraps `httpx` and enforces the active network policy
before any request leaves the process. If the destination host is not
permitted, a `PolicyViolationError` is raised before any network I/O
occurs.

**Source file:** `missy/gateway/client.py`

---

## Why All Outbound HTTP Must Go Through This Client

Missy's threat model assumes that arbitrary code (plugins, tools, provider
implementations) will attempt to contact external services. Network policy
enforcement is useless if it can be bypassed by importing `httpx` or
`requests` directly. By centralising all HTTP through `PolicyHTTPClient`:

- Every outbound request is checked against `NetworkPolicyEngine`
- Every request is audit-logged (both the policy decision and the
  completed request)
- Session and task identifiers propagate through to audit events
  automatically
- Connection pooling is shared across calls (via lazy `httpx.Client`
  creation)

Providers, the Discord REST client, the Discord Gateway client, and all
tools/plugins are expected to use `create_client()` rather than creating
their own HTTP clients.

---

## Creating a Client

```python
from missy.gateway.client import create_client

client = create_client(session_id="s1", task_id="t1")
response = client.get("https://api.github.com/zen")
```

The factory function `create_client(session_id, task_id, timeout=30)` is
the recommended constructor. It returns a `PolicyHTTPClient` instance. The
`session_id` and `task_id` are forwarded to all audit events and policy
checks.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `str` | `""` | Session identifier for audit/policy. |
| `task_id` | `str` | `""` | Task identifier for audit/policy. |
| `timeout` | `int` | `30` | Default request timeout in seconds. |

---

## Request Methods

### Synchronous

| Method | Signature |
|--------|-----------|
| `get(url, **kwargs)` | `-> httpx.Response` |
| `post(url, **kwargs)` | `-> httpx.Response` |

### Asynchronous

| Method | Signature |
|--------|-----------|
| `aget(url, **kwargs)` | `-> httpx.Response` |
| `apost(url, **kwargs)` | `-> httpx.Response` |

All four methods follow the same pattern:

1. Call `_check_url(url)` -- extract host, run policy check
2. Forward the request to the underlying `httpx` client
3. Emit a `network_request` audit event with the HTTP method, URL, and
   status code

Extra `**kwargs` are forwarded verbatim to the corresponding `httpx`
method (e.g. `headers`, `json`, `data`, `params`).

---

## Policy Enforcement: `_check_url(url)`

```python
def _check_url(self, url: str) -> None:
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise ValueError(...)
    get_policy_engine().check_network(host, self.session_id, self.task_id)
```

1. The URL is parsed with `urllib.parse.urlparse`.
2. The hostname is extracted via `parsed.hostname` (which strips brackets
   from IPv6 literals and lowercases).
3. If the hostname is empty or `None`, a `ValueError` is raised.
4. `get_policy_engine().check_network(host, ...)` is called, which
   delegates to `NetworkPolicyEngine.check_host()`.
5. On deny, `PolicyViolationError` is raised before any network I/O.

### CIDR Check Logic

When the host is a bare IP address (or resolves to one via DNS),
`NetworkPolicyEngine._check_cidr()` iterates pre-parsed
`ipaddress.IPv4Network` / `IPv6Network` objects and tests whether the
address falls within any CIDR block using the `in` operator.

Example: if `allowed_cidrs` contains `"10.0.0.0/8"`, then
`10.42.1.5` is allowed but `192.168.1.1` is denied.

### Domain Suffix Matching Logic

`NetworkPolicyEngine._check_domain()` implements two semantics:

- **Exact:** `"github.com"` matches only `"github.com"`.
- **Wildcard:** `"*.github.com"` matches `"api.github.com"`,
  `"github.com"` itself, and any subdomain ending in `.github.com`.

The wildcard prefix `*.` is stripped to produce a suffix. The host matches
if it equals the suffix or ends with `"." + suffix`.

### Explicit Host:Port Matching

`NetworkPolicyEngine._check_exact_host()` compares the normalised hostname
against each `allowed_hosts` entry. The entry may include a port suffix
(e.g. `"api.github.com:443"`). The port is stripped with `rsplit(":", 1)`
before comparison. Comparison is case-insensitive.

---

## Audit Events

### Policy Check Events

These are emitted by `NetworkPolicyEngine.check_host()`, not by the
client itself:

```json
{
  "event_type": "network_check",
  "category": "network",
  "result": "allow" | "deny",
  "policy_rule": "domain:*.github.com",
  "detail": {"host": "api.github.com"}
}
```

### Request Completion Events

After a successful HTTP request, the client emits:

```json
{
  "event_type": "network_request",
  "category": "network",
  "result": "allow",
  "detail": {"method": "GET", "url": "https://api.github.com/zen", "status_code": 200}
}
```

This event is only emitted when the request completes (no exception from
`httpx`). If the request raises an `httpx.HTTPError`, no request event is
emitted (but the earlier `network_check` allow event will be present in
the audit trail).

---

## Context Manager Support

```python
with create_client(session_id="s1", task_id="t1") as client:
    response = client.get("https://example.com")
# Underlying httpx.Client is closed on exit
```

Both synchronous (`__enter__` / `__exit__` / `close()`) and asynchronous
(`__aenter__` / `__aexit__` / `aclose()`) context managers are supported.
`close()` and `aclose()` are safe to call even if the underlying client
was never created.

---

## Lazy Client Creation

The underlying `httpx.Client` and `httpx.AsyncClient` are created lazily
on first use:

- `_get_sync_client()` creates an `httpx.Client(timeout=self.timeout)` on
  the first synchronous request and reuses it for all subsequent sync
  calls.
- `_get_async_client()` does the same for `httpx.AsyncClient`.

This means that constructing a `PolicyHTTPClient` is cheap -- no
connections are opened until a request is actually made.

---

## How Providers Use the Client

Providers do not currently use `PolicyHTTPClient` directly. The Anthropic
provider uses the `anthropic` SDK which manages its own HTTP client.
Network policy enforcement for provider API calls happens at the network
level through the policy engine's allowed-domain rules (the operator must
add `api.anthropic.com` to `allowed_domains` or `allowed_hosts`).

The Discord channel integration uses `PolicyHTTPClient` extensively:
- `DiscordRestClient` uses it for sending messages, registering commands,
  and triggering typing indicators
- `DiscordGatewayClient` uses it for WebSocket-adjacent REST calls
- Interaction responses in `DiscordChannel._handle_interaction()` create
  a client via `create_client()` to POST to the Discord API
