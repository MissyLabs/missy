# Missy Connectivity Audit

## Network Architecture

All outbound HTTP is routed through `PolicyHTTPClient` (`missy/gateway/client.py`), which enforces network policy before dispatching any request.

## Policy Layers

### 1. Network Policy (`NetworkPolicyEngine`)
- CIDR block matching
- Domain suffix matching
- Per-category host allowlists (provider, tool, discord)
- Default deny when `network.default_deny: true`

### 2. REST Policy (`RestPolicy`)
- Per-host HTTP method + path glob rules
- Example: Allow `GET /repos/**` on `api.github.com`, deny `DELETE /**`

### 3. Interactive Approval (`InteractiveApproval`)
- TUI prompt for policy-denied operations
- Session-scoped memory (allow-always)
- Non-TTY auto-denies

## Default Presets

| Preset | Hosts | Purpose |
|--------|-------|---------|
| `anthropic` | `api.anthropic.com` | Anthropic Claude API |
| `openai` | `api.openai.com` | OpenAI API |
| `github` | `api.github.com`, `github.com` | GitHub API and web |

## Provider Connectivity

| Provider | Endpoint | Auth |
|----------|----------|------|
| Anthropic | `api.anthropic.com/v1/messages` | API key or vault reference |
| OpenAI | `api.openai.com/v1/chat/completions` | API key or vault reference |
| Ollama | `localhost:11434` | None (local) |

### API Key Management
- Direct config: `api_key` field
- Environment variable: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
- Vault reference: `vault://KEY_NAME`
- Key rotation: `api_keys` list with automatic rotation

## Channel Connectivity

| Channel | Protocol | Port | Auth |
|---------|----------|------|------|
| CLI | stdin/stdout | — | Local user |
| Discord | WebSocket Gateway | 443 | Bot token |
| Webhook | HTTP POST | Configurable | Optional header |
| Voice | WebSocket | 8765 | PBKDF2-hashed tokens |

## Vision Connectivity

The vision subsystem itself has **no network dependencies**:
- Camera discovery: local sysfs reads
- Frame capture: local OpenCV
- Screenshot: local CLI tools (scrot, gnome-screenshot)
- Image preprocessing: local NumPy/OpenCV

Network is only used when sending processed images to a vision-capable LLM API (through the existing `PolicyHTTPClient` enforcement).

## Outbound Request Audit

All outbound HTTP requests are logged with:
- Timestamp
- Target host and URL
- HTTP method
- Policy rule that allowed/denied
- Response status code
- Request duration

## DNS Resolution

- System resolver used for all DNS
- No custom DNS configuration
- Domain matching in policy is suffix-based (e.g., `.anthropic.com`)
