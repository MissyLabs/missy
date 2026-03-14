# AUDIT_CONNECTIVITY

- Timestamp: 2026-03-14
- Auditor: Automated build analysis

## Network Architecture

Missy enforces **default-deny** outbound networking. All HTTP traffic passes through `PolicyHTTPClient` (`missy/gateway/client.py`) which checks every request against the network policy engine before dispatch.

## Allowed Connectivity Patterns

### Provider Endpoints (per-category: provider_allowed_hosts)

| Provider | Required Hosts |
|---|---|
| Anthropic | api.anthropic.com |
| OpenAI | api.openai.com |
| Ollama | localhost:11434 (configurable) |

### Tool Endpoints (per-category: tool_allowed_hosts)

| Tool | Required Hosts |
|---|---|
| WebFetchTool | Operator-configured domains |
| BrowserTools | Operator-configured domains |

### Discord Endpoints (per-category: discord_allowed_hosts)

| Service | Required Hosts |
|---|---|
| Discord API | discord.com, gateway.discord.gg |
| Discord CDN | cdn.discordapp.com |

### Infrastructure Endpoints

| Service | Required Hosts |
|---|---|
| OpenTelemetry | Operator-configured OTLP endpoint (default: localhost:4317) |
| MCP servers | Operator-configured per-server |

## Secure Configuration Example

```yaml
network:
  default_deny: true
  allowed_cidrs:
    - "10.0.0.0/8"          # Internal network
  allowed_domains:
    - "*.github.com"         # GitHub API access
  allowed_hosts: []          # No general host access
  provider_allowed_hosts:
    - "api.anthropic.com"
    - "api.openai.com"
  tool_allowed_hosts: []     # No tool network access by default
  discord_allowed_hosts:
    - "discord.com"
    - "gateway.discord.gg"
    - "cdn.discordapp.com"
```

## Policy Enforcement Points

1. **PolicyHTTPClient.request()** — Every outbound HTTP request checked
2. **ToolRegistry.execute()** — Tools checked against policy before execution
3. **ShellExecTool.execute()** — Commands checked against shell policy
4. **FileReadTool/FileWriteTool** — Paths checked against filesystem policy
5. **SchedulerManager._run_job()** — Jobs run through policy enforcement

## Network Policy Test Coverage

| Test Category | Tests |
|---|---|
| CIDR allowlist matching | 15+ |
| Domain suffix matching | 10+ |
| Exact hostname matching | 10+ |
| Per-category host lists | 10+ |
| Default deny enforcement | 5+ |
| PolicyHTTPClient integration | 20+ |
| **Total** | **70+** |

## Connectivity Verification

All outbound connections are auditable via:
- `missy audit security` — Shows network allow/deny events
- `~/.missy/audit.jsonl` — Structured JSONL with every network decision
- OpenTelemetry traces — When OTEL is enabled, traces include HTTP spans
