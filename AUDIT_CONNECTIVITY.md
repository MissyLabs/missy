# Connectivity Audit Report

## Date: 2026-03-18

## Network Architecture

All outbound HTTP traffic passes through `PolicyHTTPClient` (`missy/gateway/client.py`), which enforces:

1. **Network policy**: CIDR blocks, domain suffixes, host:port pairs
2. **REST policy**: HTTP method + URL path matching per host
3. **Interactive approval**: TUI prompt for denied operations (y/n/a)

## Default-Deny Posture

By default, **all network access is blocked**. Users must explicitly enable:

```yaml
network:
  default_deny: true
  presets:
    - anthropic    # api.anthropic.com:443
    - openai       # api.openai.com:443
    - github       # api.github.com:443, github.com:443
    - discord      # discord.com, gateway.discord.gg, cdn.discordapp.com
    - ollama       # localhost:11434
```

## Provider Connectivity

| Provider | Default Host | Preset | Protocol |
|---|---|---|---|
| Anthropic | api.anthropic.com:443 | `anthropic` | HTTPS |
| OpenAI | api.openai.com:443 | `openai` | HTTPS |
| Ollama | localhost:11434 | `ollama` | HTTP |
| Codex | api.openai.com:443 | `openai` | HTTPS |

## Channel Connectivity

| Channel | Direction | Protocol | Auth |
|---|---|---|---|
| CLI | Local | stdin/stdout | None |
| Discord | Outbound | WebSocket + HTTPS | Bot token |
| Webhook | Inbound | HTTP | Configurable |
| Voice | Both | WebSocket (port 8765) | PBKDF2 token |

## REST Policy (L7 Controls)

```yaml
rest_policies:
  - host: "api.github.com"
    method: "GET"
    path: "/repos/**"
    action: "allow"
  - host: "api.github.com"
    method: "DELETE"
    path: "/**"
    action: "deny"
```

## MCP Server Connectivity

MCP servers connect via stdio or SSE. Digest pinning (`missy mcp pin`) verifies tool manifests with SHA-256 hashes. Mismatches refuse to load.

## Audit Trail

Every network request is logged to `~/.missy/audit.jsonl`:
- Timestamp, session ID, task ID
- URL, method, status code
- Policy rule that allowed/denied
- Duration
