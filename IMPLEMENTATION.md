# Missy — Full Implementation Reference

**Version**: Current (March 2026)
**Platform**: Linux, Python 3.11+
**License**: Self-hosted, security-first local agentic AI assistant

---

## Table of Contents

1. [Overview & Philosophy](#1-overview--philosophy)
2. [Architecture](#2-architecture)
3. [Agent Runtime & Agentic Loop](#3-agent-runtime--agentic-loop)
4. [Provider System](#4-provider-system)
5. [Tool System](#5-tool-system)
6. [Security Layer](#6-security-layer)
7. [Policy Engine](#7-policy-engine)
8. [Memory & Learning](#8-memory--learning)
9. [Context Management](#9-context-management)
10. [Channel System](#10-channel-system)
11. [MCP Integration](#11-mcp-integration)
12. [Scheduler](#12-scheduler)
13. [Observability & Audit](#13-observability--audit)
14. [Configuration System](#14-configuration-system)
15. [CLI Interface](#15-cli-interface)
16. [Desktop Automation](#16-desktop-automation)
17. [Voice Channel](#17-voice-channel)
18. [Sub-Agent System](#18-sub-agent-system)
19. [Self-Improvement Systems](#19-self-improvement-systems)
20. [Fault Tolerance](#20-fault-tolerance)
21. [Dependencies & Installation](#21-dependencies--installation)
22. [File Layout & Module Tree](#22-file-layout--module-tree)
23. [Test Suite](#23-test-suite)
24. [Comparison Matrix](#24-comparison-matrix)

---

## 1. Overview & Philosophy

Missy is a **security-first, self-hosted, local agentic AI assistant** for Linux. Every capability — shell access, network, filesystem, plugins — is **disabled by default** and must be explicitly enabled in configuration. The system is designed for full auditability: every decision, tool call, policy check, and provider invocation emits a structured audit event.

### Core Design Principles

| Principle | Implementation |
|---|---|
| **Secure by default** | All capabilities disabled until explicitly enabled in `config.yaml` |
| **Policy enforcement at the gateway** | Single `PolicyHTTPClient` wraps all outbound HTTP; network/filesystem/shell checks happen before any I/O |
| **Full auditability** | Every action emits structured `AuditEvent` via event bus → JSONL file + optional OTLP |
| **Provider agnostic** | Anthropic, OpenAI, Ollama (local), Codex — with fallback, rotation, and tiering |
| **Multi-channel** | CLI, Discord, Webhook, Voice — same agent runtime, different I/O |
| **Self-improving** | Cross-task learning, prompt patches, done-criteria verification |
| **Fault tolerant** | Circuit breaker, checkpointing, failure tracking, watchdog, retry with strategy rotation |

### Codebase Metrics

- **99 Python source files** across 15 packages
- **48 test files**, **948+ tests**, ~86% coverage
- **20+ built-in tools** (filesystem, shell, web, browser, X11, accessibility, Discord)
- **4 providers** (Anthropic, OpenAI, Ollama, Codex)
- **4 channels** (CLI, Discord, Webhook, Voice)

---

## 2. Architecture

### System Diagram

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Channel Layer                                               │
│  ┌─────────┐ ┌───────────┐ ┌───────────┐ ┌──────────────┐  │
│  │   CLI   │ │  Discord  │ │  Webhook  │ │    Voice     │  │
│  │ (stdin/ │ │ (Gateway  │ │  (HTTP    │ │ (WebSocket   │  │
│  │  stdout)│ │  WebSocket│ │   POST)   │ │  + STT/TTS)  │  │
│  └────┬────┘ └─────┬─────┘ └─────┬─────┘ └──────┬───────┘  │
│       └─────────────┴─────────────┴──────────────┘           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Security Pipeline                                           │
│  InputSanitizer → SecretsDetector → SecretCensor (output)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Agent Runtime                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Agentic Tool Loop (max_iterations configurable)      │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │  │
│  │  │  Provider    │  │  Tool        │  │  Context     │  │  │
│  │  │  (via        │  │  Registry    │  │  Manager     │  │  │
│  │  │  Circuit     │  │  (policy-    │  │  (token      │  │  │
│  │  │  Breaker)    │  │  gated)      │  │  budget)     │  │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │  │
│  │         │                 │                  │          │  │
│  │  ┌──────┴─────────────────┴──────────────────┴───────┐  │  │
│  │  │  DoneCriteria · Learnings · PromptPatches          │  │  │
│  │  │  FailureTracker · Checkpoint · SubAgentRunner      │  │  │
│  │  │  ApprovalGate · Watchdog                           │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Policy   │ │ Memory   │ │ Audit    │
        │ Engine   │ │ (SQLite  │ │ Logger   │
        │ (net/fs/ │ │  FTS5 +  │ │ (JSONL + │
        │  shell)  │ │  resilient│ │  OTLP)  │
        └──────────┘ └──────────┘ └──────────┘
```

### Data Flow

1. **Input arrives** via a channel (CLI stdin, Discord message, webhook POST, voice transcription)
2. **Security pipeline** scans for prompt injection (13 patterns) and secrets (9 credential types)
3. **Agent runtime** resolves provider, loads session history, builds context within token budget
4. **Agentic loop** sends messages to LLM provider (via circuit breaker), executes tool calls (via policy-gated registry), repeats until `finish_reason == "stop"` or max iterations reached
5. **Post-loop**: saves conversation to memory, extracts learnings, censors secrets from output
6. **Response** delivered back through the originating channel

---

## 3. Agent Runtime & Agentic Loop

### AgentConfig

```python
@dataclass
class AgentConfig:
    provider: str = "anthropic"      # Provider name to resolve
    model: Optional[str] = None      # Model override (uses provider default if None)
    system_prompt: str = "..."       # Injected as system message
    max_iterations: int = 10         # Tool loop iteration cap
    temperature: float = 0.7         # Sampling temperature
```

### Runtime Lifecycle (`AgentRuntime.run()`)

```
run(user_input, session_id)
│
├─ 1. Resolve/create session (thread-local SessionManager)
├─ 2. Emit "agent.run.start" audit event
├─ 3. Resolve provider with fallback (ProviderRegistry)
├─ 4. Load session history from memory store (up to 50 turns)
├─ 5. Build context-managed messages (ContextManager)
│     ├─ Enrich system prompt with relevant memories (15% token budget)
│     └─ Inject learnings from past runs (5% token budget)
├─ 6. Enter agentic tool loop (_tool_loop)
│     ├─ Initialize FailureTracker (threshold=3)
│     ├─ Initialize Checkpoint
│     └─ Loop up to max_iterations:
│           ├─ Call provider.complete_with_tools() via CircuitBreaker
│           ├─ If finish_reason == "tool_calls":
│           │     ├─ Execute each tool via ToolRegistry
│           │     ├─ Track failures in FailureTracker
│           │     ├─ If failure threshold exceeded → inject strategy rotation prompt
│           │     ├─ Update checkpoint
│           │     ├─ Inject DoneCriteria verification prompt
│           │     └─ Continue loop
│           ├─ If finish_reason == "stop":
│           │     ├─ Mark checkpoint complete
│           │     └─ Return (response_text, tools_used)
│           └─ If max_iterations reached:
│                 └─ Make one final plain completion (no tools)
├─ 7. Persist conversation turn to memory
├─ 8. Extract and save learnings (task_type, outcome, lesson)
├─ 9. Censor secrets from response
├─ 10. Emit "agent.run.complete" audit event
└─ Return response string
```

### Tool Execution Flow

```python
def _execute_tool(tool_call, session_id, task_id):
    # Strip session_id/task_id from tool args (prevent kwarg collision)
    tool_args = {k: v for k, v in tool_call.arguments.items()
                 if k not in ("session_id", "task_id")}

    result = registry.execute(
        tool_call.name,
        session_id=session_id,
        task_id=task_id,
        **tool_args,
    )
```

The `ToolRegistry.execute()` method:
1. Looks up tool by name
2. Checks permissions against `PolicyEngine` (network → filesystem_read → filesystem_write → shell)
3. Strips internal kwargs (`session_id`, `task_id`) before forwarding to `tool.execute()`
4. Catches all exceptions → returns `ToolResult(success=False)` instead of propagating
5. Emits audit event with allow/deny/error result

### System Prompt

The default system prompt instructs the agent to:
- Use tools immediately without narrating intent
- Never say "I will now..." — just call the tool
- Prefer browser tools for web tasks
- Use shell_exec for system commands
- Use file_read/file_write for filesystem operations

---

## 4. Provider System

### Provider Interface

```python
class BaseProvider:
    name: str

    def is_available(self) -> bool: ...
    def complete(self, messages: list[Message], **kwargs) -> CompletionResponse: ...
    def complete_with_tools(self, messages, tools, system) -> CompletionResponse: ...
    def get_tool_schema(self, tools) -> list: ...
    def stream(self, messages, system) -> Iterator[str]: ...
```

### Data Types

```python
@dataclass
class Message:
    role: str       # "user" | "assistant" | "system"
    content: str

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class CompletionResponse:
    content: str
    model: str
    provider: str
    usage: dict          # {prompt_tokens, completion_tokens, total_tokens}
    raw: dict
    tool_calls: list     # List[ToolCall]
    finish_reason: str   # "stop" | "tool_calls" | "length"
```

### Implemented Providers

| Provider | Native Tool Calling | Streaming | Model Tiering | API Key Rotation |
|---|---|---|---|---|
| **Anthropic** | Yes (Messages API) | Yes | fast/premium model | Yes (api_keys list) |
| **OpenAI** | Yes (Chat Completions) | Yes | fast/premium model | Yes |
| **Ollama** | No (prompted fallback) | Yes | N/A (local) | N/A |
| **Codex** | No | No | N/A | OAuth token |

### Anthropic Provider

- Uses `anthropic` SDK (lazy import for graceful degradation)
- Extracts system messages into the `system` parameter
- Converts tool schemas to Anthropic's `input_schema` format
- Handles `tool_use` content blocks in responses
- Error handling: `APITimeoutError`, `AuthenticationError`, `APIError` → `ProviderError`

### OpenAI Provider

- Uses `openai` SDK (lazy import)
- Standard Chat Completions API with `tools` parameter
- Supports function calling natively
- Handles `tool_calls` in response choices

### Ollama Provider

- **No SDK dependency** — raw HTTP to `/api/chat` via `PolicyHTTPClient`
- Default base URL: `http://localhost:11434`
- Tool calling via **prompted fallback**: injects tool schemas into system prompt, parses JSON response
- Expected tool-call format: `{"tool_call": {"name": "...", "arguments": {...}}}`
- Regex extraction: `r'\{\s*"tool_call"\s*:\s*\{.*?\}\s*\}'`
- Streaming via newline-delimited JSON chunks

### Codex Provider

- Uses ChatGPT's Codex backend API (`chatgpt.com/backend-api/codex/responses`)
- OAuth token authentication (from `missy setup` wizard)
- Account ID extracted from JWT payload

### Provider Resolution

```python
# ProviderRegistry resolves with fallback
registry = get_registry()
provider = registry.get(config.provider)  # Try requested
if not provider or not provider.is_available():
    for fallback in registry.get_available():
        provider = fallback
        break
```

### ProviderConfig

```python
@dataclass
class ProviderConfig:
    name: str
    model: str = "claude-sonnet-4-6"
    timeout: int = 30
    api_key: Optional[str] = None
    api_keys: list[str] = field(default_factory=list)  # Rotation pool
    base_url: Optional[str] = None
    enabled: bool = True
    fast_model: str = ""       # e.g. claude-haiku-4-5
    premium_model: str = ""    # e.g. claude-opus-4-6
```

---

## 5. Tool System

### Tool Base Classes

```python
@dataclass
class ToolPermissions:
    network: bool = False
    filesystem_read: bool = False
    filesystem_write: bool = False
    shell: bool = False

@dataclass
class ToolResult:
    success: bool
    output: Any
    error: Optional[str] = None

class BaseTool:
    name: str
    description: str
    permissions: ToolPermissions
    parameters: dict[str, Any]

    def execute(self, **kwargs) -> ToolResult: ...
    def get_schema(self) -> dict: ...
```

### Built-in Tools (20+)

#### Filesystem Tools

| Tool | Permissions | Key Parameters |
|---|---|---|
| `file_read` | filesystem_read | `path`, `encoding="utf-8"`, `max_bytes=65536` |
| `file_write` | filesystem_write | `path`, `content`, `mode="overwrite"\|"append"` |
| `file_delete` | filesystem_write | `path` (regular files only, not directories) |
| `list_files` | filesystem_read | `path`, `recursive=False`, `max_entries=200` |

#### Shell

| Tool | Permissions | Key Parameters |
|---|---|---|
| `shell_exec` | shell | `command`, `cwd`, `timeout=30` (max 300s). Uses `/bin/bash`, output truncated to 32KB |

#### Network

| Tool | Permissions | Key Parameters |
|---|---|---|
| `web_fetch` | network | `url`, `timeout=30`, `headers`. Uses `PolicyHTTPClient`, truncated to 64KB |

#### Computation

| Tool | Permissions | Key Parameters |
|---|---|---|
| `calculator` | none | `expression`. AST-based safe eval, max exponent 1000 |

#### Browser Automation (Playwright)

| Tool | Key Parameters | Notes |
|---|---|---|
| `browser_navigate` | `url`, `headless=False`, `wait_until`, `session_id` | Launches Firefox via `launch_persistent_context()` |
| `browser_click` | `text`, `selector`, `role`, `name`, `timeout_ms=5000` | Supports text, CSS, ARIA selectors |
| `browser_fill` | `value`, `selector`, `label`, `placeholder`, `press_enter` | Input field filling |
| `browser_screenshot` | `path`, `full_page`, `selector` | Page or element capture |
| `browser_get_content` | `selector="body"`, `content_type="text"\|"html"`, `max_length=5000` | Extract page content |
| `browser_evaluate` | `script` | Execute JavaScript |
| `browser_wait` | `for_selector`, `for_url`, `for_text`, `seconds` | Wait for conditions |
| `browser_get_url` | — | Returns current URL and title |
| `browser_close` | — | Closes browser session |

**Browser Session Architecture**:
- `BrowserSession`: Manages Playwright context per session ID
- `_SessionRegistry`: Thread-safe session pool
- Persistent profiles at `~/.missy/browser_sessions/<session_id>/`
- `get_page()` always returns the latest live page (prevents stale tab references)
- Firefox prefs disable session restore, crash dialogs, tab close warnings

#### X11 Desktop Automation

| Tool | Permissions | Key Parameters |
|---|---|---|
| `x11_screenshot` | shell, filesystem_write | `path`, `region="x,y,w,h"`. Uses `scrot` |
| `x11_read_screen` | shell, filesystem_write, network | `question`, `path`, `region`. Screenshots then sends to Ollama vision (`minicpm-v`). Auto-uses Playwright page if browser session is active |
| `x11_click` | shell | `x`, `y`, `button="left"`, `window_name`. Uses `xdotool` |
| `x11_type` | shell | `text`, `window_name`, `delay_ms=12`. Uses `xdotool` |
| `x11_key` | shell | `key` (e.g. "Return", "ctrl+c"), `window_name` |
| `x11_window_list` | shell | Lists windows via `wmctrl -l` (fallback: `xdotool`) |

**Vision Pipeline** (`x11_read_screen`):
1. If Playwright browser session active → capture from browser page directly
2. Otherwise → capture desktop via `scrot`
3. Base64-encode the screenshot
4. Send to Ollama's `/api/chat` endpoint with `images` field
5. Model: `minicpm-v` (configurable), timeout: 120s
6. Returns AI description of what's on screen

#### AT-SPI Accessibility Tools

| Tool | Key Parameters |
|---|---|
| `atspi_get_tree` | `app_name`, `max_depth=10` — inspect GTK accessibility tree |
| `atspi_click` | `name`, `role`, `app_name` — click accessible element |
| `atspi_get_text` | `name`, `app_name` — extract text from element |
| `atspi_set_value` | `name`, `value`, `app_name` — fill accessible input |

#### Discord

| Tool | Permissions | Key Parameters |
|---|---|---|
| `discord_upload_file` | filesystem_read, network | `channel_id`, `path`, `description` |

#### Meta

| Tool | Key Parameters |
|---|---|
| `self_create_tool` | `name`, `description`, `code` — dynamically register Python tools at runtime |

---

## 6. Security Layer

### Input Sanitization

**InputSanitizer** detects 13 prompt injection patterns:

```
ignore (all) previous instructions
disregard (all) previous instructions
forget (all) previous instructions
you are now (a) different
pretend you are
act as (if you are) a
system:
<system>
[INST]
### (System|Instruction)
<|im_start|>
<|system|>
override (your) (previous) instructions
```

- Max input length: 10,000 characters (truncated with warning)
- Detection is non-blocking (logs warning, returns text as-is for now)

### Secrets Detection

**SecretsDetector** scans for 9 credential patterns:

| Type | Pattern |
|---|---|
| API Key | `(api[_-]?key\|apikey)` followed by 20+ chars |
| AWS Key | `AKIA[0-9A-Z]{16}` |
| Private Key | `-----BEGIN (RSA\|EC )?PRIVATE KEY-----` |
| GitHub Token | `ghp_[A-Za-z0-9]{36}` |
| Password | `(password\|passwd\|pwd)` followed by 8+ chars |
| Token/Secret | `(token\|secret)` followed by 20+ chars |
| Stripe Key | `sk_(live\|test)_[A-Za-z0-9]{24,}` |
| Slack Token | `xox[baprs]-[A-Za-z0-9\-]{10,}` |
| JWT | `eyJ...` three-part base64 |

### Output Censoring

**SecretCensor** redacts detected secrets from agent responses before delivery to the channel. Replaces matches with `[REDACTED]`, processing right-to-left to preserve positions.

### Vault

**ChaCha20-Poly1305 encrypted key-value store**:

- Key file: `~/.missy/secrets/vault.key` (32 bytes, mode 0600)
- Data file: `~/.missy/secrets/vault.enc` (mode 0600)
- Directory: `~/.missy/secrets/` (mode 0700)
- Reference resolution: `vault://KEY_NAME`, `$ENV_VAR`, or plain value
- Operations: `set`, `get`, `delete`, `list_keys`, `resolve`

---

## 7. Policy Engine

### Three-Layer Enforcement

```python
class PolicyEngine:
    network: NetworkPolicyEngine
    filesystem: FilesystemPolicyEngine
    shell: ShellPolicyEngine
```

All policy checks emit audit events with allow/deny results.

### Network Policy

```yaml
network:
  default_deny: true           # Deny all unless matched
  allowed_cidrs: []            # CIDR blocks (e.g. "10.0.0.0/8")
  allowed_domains: []          # Wildcard suffixes (e.g. "*.github.com")
  allowed_hosts: []            # Exact host:port pairs
  provider_allowed_hosts: []   # Per-category overrides
  tool_allowed_hosts: []
  discord_allowed_hosts: []
```

**Check order**: default_deny → CIDR → exact host → domain suffix → DNS resolve + re-check CIDR → deny

### Filesystem Policy

```yaml
filesystem:
  allowed_read_paths: []       # e.g. ["~/workspace", "/tmp"]
  allowed_write_paths: []      # e.g. ["~/workspace"]
```

- Resolves symlinks via `Path.resolve(strict=False)` before checking
- Uses `Path.is_relative_to()` for containment check

### Shell Policy

```yaml
shell:
  enabled: false               # Master switch
  allowed_commands: []          # e.g. ["ls", "git", "docker"]
```

- `enabled=false` → deny all
- Empty `allowed_commands` + `enabled=true` → allow all
- Extracts command basename via `shlex.split()`

### Gateway Client

**`PolicyHTTPClient`** is the single enforcement point for ALL outbound HTTP:

```python
class PolicyHTTPClient:
    def get(self, url, **kwargs) -> httpx.Response:
        self._check_url(url)       # Raises PolicyViolationError
        return self._client.get(url, **kwargs)

    def post(self, url, **kwargs) -> httpx.Response:
        self._check_url(url)
        return self._client.post(url, **kwargs)
```

Every provider, tool, and internal HTTP call goes through this client.

---

## 8. Memory & Learning

### Memory Store (SQLite FTS5)

```
~/.missy/memory.db
├── turns (id, session_id, timestamp, role, content, provider, metadata)
├── turns_fts (FTS5 virtual table for full-text search)
└── Triggers: auto-populate FTS on insert/delete
```

**Pragmas**: `journal_mode=WAL`, `synchronous=NORMAL`

**Operations**:
- `add_turn(session_id, role, content, provider)` — persist conversation
- `get_session_turns(session_id, limit=50)` — retrieve history
- `search(query, limit=10)` — FTS5 ranked search
- `save_learning(TaskLearning)` — persist cross-task learning
- `get_learnings(task_type, limit=5)` — retrieve relevant learnings
- `cleanup(older_than_days)` — remove old turns

### Resilient Memory Layer

**`ResilientMemoryStore`** wraps SQLiteMemoryStore with fallback:
- If SQLite fails → falls back to JSON file store
- Logs warnings but never crashes the agent

### Cross-Task Learning

```python
@dataclass
class TaskLearning:
    task_type: str       # "shell", "web", "file", "shell+web", etc.
    approach: list[str]  # Tool names used (capped at 5)
    outcome: str         # "success" | "failure" | "partial"
    lesson: str          # Extracted insight
    timestamp: str       # ISO-8601 UTC
```

**Extraction**: After each tool-augmented run, the system:
1. Determines `task_type` from tools used (priority: shell+web > shell+file > shell > web > file > chat)
2. Determines `outcome` from response keywords (success indicators vs failure indicators)
3. Stores the learning in SQLite
4. Injects relevant past learnings into future context (5% token budget)

---

## 9. Context Management

### Token Budget

```python
@dataclass
class TokenBudget:
    total: int = 30_000              # Total context window budget
    system_reserve: int = 2_000      # Reserved for system prompt
    tool_definitions_reserve: int = 2_000  # Reserved for tool schemas
    memory_fraction: float = 0.15    # 15% of available for memories
    learnings_fraction: float = 0.05 # 5% of available for learnings
```

### Context Assembly

`ContextManager.build_messages()`:

1. Calculate available tokens: `total - system_reserve - tool_definitions_reserve`
2. Allocate memory budget: `available * memory_fraction`
3. Allocate learnings budget: `available * learnings_fraction`
4. Enrich system prompt with memory results and learnings (up to 5)
5. Add new user message
6. Add history turns from most recent backward
7. **Prune from oldest** when over budget
8. Return `(enriched_system_prompt, messages_list)`

Token estimation: `max(1, len(text) // 4)` (4 characters per token heuristic)

---

## 10. Channel System

### Channel Interface

```python
class BaseChannel:
    name: str
    def send(self, message: str) -> None: ...
    def receive(self) -> Optional[ChannelMessage]: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...

@dataclass
class ChannelMessage:
    content: str
    sender: str
    channel: str
    metadata: dict[str, Any]
```

### CLI Channel

- Interactive stdin/stdout REPL
- Supports `--provider` and `--session` flags
- Rich terminal output via `rich` library

### Discord Channel

**Full WebSocket Gateway API implementation**:

- `DiscordGatewayClient`: WebSocket connection with heartbeat, resume, reconnect
- `DiscordRestClient`: HTTP API for sending messages, managing reactions, registering commands
- Access control pipeline:
  1. Filter own-bot messages
  2. Credential detection → delete message + warn user
  3. Bot filtering (configurable)
  4. Attachment policy gate
  5. DM policy check (OPEN / ALLOWLIST_ONLY / DISABLED)
  6. Guild policy check (channel allowlist, user allowlist, mention requirement)

**Slash Commands**: `/ask`, `/status`, `/model`, `/help`
- `/ask` runs agent via `run_in_executor` (prevents async conflicts with sync tools like Playwright)

**Configuration**:
```python
@dataclass
class DiscordAccountConfig:
    token_env_var: str
    dm_policy: DiscordDMPolicy = DiscordDMPolicy.OPEN
    dm_allowlist: list[str] = field(default_factory=list)
    ignore_bots: bool = True
    guild_policies: list[DiscordGuildPolicy] = field(default_factory=list)
```

### Webhook Channel

- HTTP POST ingress endpoint
- Configurable via gateway

### Voice Channel

See [Voice Channel](#17-voice-channel) section.

---

## 11. MCP Integration

### Model Context Protocol

**`McpManager`** manages connections to external MCP servers:

```json
// ~/.missy/mcp.json
[
  {"name": "filesystem", "command": "npx @modelcontextprotocol/server-filesystem /tmp"},
  {"name": "postgres", "command": "npx @modelcontextprotocol/server-postgres postgresql://..."},
  {"name": "web", "url": "http://localhost:3000"}
]
```

**Capabilities**:
- Connect to MCP servers via stdio (command) or HTTP (url)
- Tools are namespaced: `server__tool` (e.g. `filesystem__read_file`)
- Auto-restart dead servers via `health_check()`
- Hot-add/remove servers at runtime
- CLI: `missy mcp list`, `missy mcp add NAME`, `missy mcp remove NAME`

---

## 12. Scheduler

### APScheduler-Backed Job Management

```python
@dataclass
class ScheduledJob:
    id: str
    name: str
    schedule: str                    # Human-readable or cron expression
    task: str                        # Prompt to execute
    provider: str
    enabled: bool
    max_attempts: int                # Retry count
    backoff_seconds: list[int]       # Exponential backoff schedule
    retry_on: list[str]              # Error patterns to retry
    delete_after_run: bool           # One-shot jobs
    active_hours: str                # e.g. "08:00-22:00"
    timezone: str                    # e.g. "America/New_York"
    run_count: int
    success_count: int
    error_count: int
```

**Schedule Parser** — human-readable to APScheduler triggers:
- `"daily at 09:00"` → `CronTrigger`
- `"every 5 minutes"` → `IntervalTrigger`
- `"Monday at 14:30"` → `CronTrigger(day_of_week=0)`
- Supports timezone-aware scheduling

**Persistence**: JSON at `~/.missy/jobs.json`

**CLI**: `missy schedule add/list/pause/resume/remove`

---

## 13. Observability & Audit

### Audit Event System

```python
@dataclass
class AuditEvent:
    event_type: str        # e.g. "network_check", "tool_execution", "agent.run.start"
    category: str          # e.g. "security", "provider", "agent"
    result: str            # "allow" | "deny" | "error"
    session_id: str
    task_id: str
    timestamp: str         # ISO-8601
    detail: dict
```

**Event Bus**: Thread-safe publish/subscribe with in-memory log
- `subscribe(event_type, callback)`
- `publish(event)`
- `get_events(category, result, event_type, limit)` — query log

### Audit Logger

- JSONL file at `~/.missy/audit.jsonl`
- One JSON object per line
- CLI: `missy audit security`, `missy audit recent`

### OpenTelemetry (Optional)

```yaml
observability:
  otel_enabled: true
  otel_endpoint: "http://localhost:4317"
  otel_protocol: "grpc"          # or "http/protobuf"
  otel_service_name: "missy"
```

- Traces and metrics sent to OTLP endpoint
- Requires `pip install -e ".[otel]"`

---

## 14. Configuration System

### Config File

`~/.missy/config.yaml` — YAML with dataclass validation

### Hot Reload

**`ConfigWatcher`** uses `watchdog` to monitor config file changes:
- Reloads policy engine on config change
- No restart required for policy updates
- Channels notified via callback

### Initialization

```bash
missy init                    # Create default secure config
missy setup                   # Interactive wizard (API keys, OAuth)
missy doctor                  # Health check all subsystems
```

### Setup Wizard

Interactive onboarding:
1. Anthropic API key (direct entry or setup-token flow)
2. OpenAI OAuth PKCE flow (browser-based)
3. Ollama local server detection
4. Discord bot token configuration
5. Policy defaults

---

## 15. CLI Interface

**Entry point**: `missy` → `missy.cli.main:cli` (Click)

### Command Tree

```
missy
├── init                              # Create default config
├── setup                             # Interactive setup wizard
├── doctor                            # System health check
├── ask PROMPT                        # Single-turn query
├── run                               # Interactive REPL
├── providers                         # List providers
├── skills                            # List skills
├── plugins                           # List plugins
├── schedule
│   ├── add                           # Add job
│   ├── list                          # List jobs
│   ├── pause JOB_ID                  # Pause job
│   ├── resume JOB_ID                 # Resume job
│   └── remove JOB_ID                 # Remove job
├── audit
│   ├── security                      # Security events
│   └── recent                        # Recent events
├── gateway
│   ├── start                         # Start gateway server
│   └── status                        # Gateway status
├── discord
│   ├── status                        # Discord config
│   ├── probe                         # Test bot token
│   ├── register-commands             # Register slash commands
│   └── audit                         # Discord events
├── vault
│   ├── set KEY VALUE                 # Store secret
│   ├── get KEY                       # Retrieve secret
│   ├── list                          # List keys
│   └── delete KEY                    # Delete secret
├── sessions
│   └── cleanup                       # Delete old history
├── approvals
│   └── list                          # Pending approvals
├── patches
│   ├── list                          # Prompt patches
│   ├── approve PATCH_ID
│   └── reject PATCH_ID
├── mcp
│   ├── list                          # MCP servers
│   ├── add NAME                      # Connect server
│   └── remove NAME                   # Disconnect server
├── devices
│   ├── list                          # Edge nodes
│   ├── pair                          # Approve pairing
│   ├── unpair NODE_ID
│   ├── status                        # Online/offline
│   └── policy NODE_ID               # Set policy mode
└── voice
    ├── status                        # Voice config
    └── test NODE_ID                  # Test TTS
```

---

## 16. Desktop Automation

### Three Layers

1. **Playwright Browser** — headless or visible Firefox, persistent sessions, full DOM access
2. **X11 Tools** — `scrot` for screenshots, `xdotool` for mouse/keyboard, `wmctrl` for window management
3. **AT-SPI Accessibility** — GTK widget tree inspection, accessible element interaction (requires `python3-pyatspi` system package)

### Vision Pipeline

`x11_read_screen` intelligently sources screenshots:
1. Check if a Playwright browser session is active → capture from browser page directly
2. Otherwise → capture desktop via `scrot`
3. Send to **Ollama** vision model (`minicpm-v`) at `localhost:11434`
4. No external API key required — fully local inference on GPU

---

## 17. Voice Channel

### Architecture

```
Edge Node (ReSpeaker/Raspberry Pi)
    │
    │ WebSocket (JSON control + binary PCM audio)
    │
    ▼
VoiceServer (port 8765)
    ├── DeviceRegistry (~/.missy/devices.json)
    ├── PairingManager (PBKDF2-hashed tokens)
    ├── PresenceStore (occupancy/sensor data)
    ├── STTEngine → FasterWhisperSTT (faster-whisper)
    └── TTSEngine → PiperTTS (piper binary)
```

### Device Management

- **Pairing**: PBKDF2-hashed token exchange
- **Policy modes**: `full` (all capabilities), `safe-chat` (text only), `muted` (no output)
- **Presence**: Occupancy and sensor data from edge nodes
- **CLI**: `missy devices list/pair/unpair/status/policy`

### STT/TTS

- **STT**: faster-whisper (CTranslate2), model: `base.en` (configurable)
- **TTS**: Piper binary (not pip), voice: `en_US-lessac-medium` (configurable)
- Requires `pip install -e ".[voice]"` for faster-whisper, numpy, soundfile

---

## 18. Sub-Agent System

### Task Decomposition

```python
@dataclass
class SubTask:
    id: int
    description: str
    tool_hints: list[str]           # Suggested tools
    depends_on: list[int]           # Dependency graph
    result: Optional[str] = None
    error: Optional[str] = None
```

### SubAgentRunner

- Parses user prompt into subtasks (numbered lists → sequential connectives → single task)
- Respects dependency graph
- Max concurrent subtasks: 3 (semaphore)
- Max total sub-agents: configurable
- Accumulates context from completed subtasks for dependent ones
- Each subtask runs its own `AgentRuntime` instance

---

## 19. Self-Improvement Systems

### Done Criteria

- Detects compound/multi-step tasks via regex (numbered lists, bullet lists, sequential connectives, ordinals)
- Generates verification prompts injected after each tool-call round
- Tracks condition completion (verified/pending)

### Prompt Patches

```python
@dataclass
class PromptPatch:
    id: str
    patch_type: PatchType    # TOOL_USAGE_HINT, ERROR_AVOIDANCE, WORKFLOW_PATTERN,
                              # DOMAIN_KNOWLEDGE, STYLE_PREFERENCE
    content: str
    confidence: float
    status: PatchStatus      # PROPOSED, APPROVED, REJECTED, EXPIRED
    applications: int
    successes: int
```

- Auto-approves low-risk patches with confidence ≥ 0.8
- Expires patches with ≥ 5 applications and < 40% success rate
- Max 20 active patches
- CLI: `missy patches list/approve/reject`
- Patches persisted at `~/.missy/patches.json`

### Learning Extraction

After each tool-augmented run:
1. Classify task type from tools used
2. Determine outcome (success/failure/partial) from response keywords
3. Store `TaskLearning` in SQLite
4. Inject relevant learnings into future system prompts (5% of token budget)

---

## 20. Fault Tolerance

### Circuit Breaker

```
State Machine:
  CLOSED ──(consecutive failures ≥ 5)──→ OPEN
  OPEN ──(timeout elapsed)──→ HALF_OPEN
  HALF_OPEN ──(success)──→ CLOSED
  HALF_OPEN ──(failure)──→ OPEN (timeout *= 2, capped at 300s)
```

- Default: threshold=5, base_timeout=60s, max_timeout=300s
- Wraps all provider calls
- Exponential backoff on repeated failures

### Failure Tracker

- Tracks consecutive tool failures per agent run
- Threshold: 3 failures
- On threshold exceeded: injects strategy rotation prompt ("try a different approach")
- Prevents infinite loops of failing tool calls

### Checkpoint

- Saves agent state (messages, tool results) after each tool-call round
- Enables recovery on crash (marks complete/failed)

### Watchdog

- Monitors agent loop for deadlocks
- Configurable timeout

### Approval Gate

- Human-in-the-loop for sensitive operations
- Thread-blocking with timeout (default 60s)
- Supports approve/deny via any channel
- CLI: `missy approvals list`

---

## 21. Dependencies & Installation

### Core Dependencies

| Package | Purpose |
|---|---|
| `click>=8.1` | CLI framework |
| `pyyaml>=6.0` | YAML config parsing |
| `anthropic>=0.25` | Anthropic Claude SDK |
| `openai>=1.25` | OpenAI SDK |
| `httpx>=0.27` | HTTP client (sync + async) |
| `apscheduler>=3.10` | Job scheduling |
| `pydantic>=2.6` | Config validation |
| `rich>=13.7` | Terminal UI |
| `cryptography>=42.0` | Vault encryption (ChaCha20-Poly1305) |
| `python-dotenv>=1.0` | Environment variables |
| `websockets>=12.0` | WebSocket server (voice, Discord) |
| `watchdog>=4.0` | Filesystem watching (hot-reload) |

### Optional Extras

```bash
pip install -e ".[dev]"       # pytest, pytest-asyncio, pytest-cov, ruff, mypy
pip install -e ".[voice]"     # faster-whisper, numpy, soundfile
pip install -e ".[otel]"      # opentelemetry-sdk, OTLP exporters
pip install -e ".[desktop]"   # playwright
```

### External Binaries

| Binary | Purpose | Install |
|---|---|---|
| `scrot` | X11 screenshots | `apt install scrot` |
| `xdotool` | X11 mouse/keyboard | `apt install xdotool` |
| `wmctrl` | X11 window management | `apt install wmctrl` |
| `piper` | TTS synthesis | github.com/rhasspy/piper |
| `ollama` | Local LLM inference | ollama.com |
| `python3-pyatspi` | AT-SPI accessibility | `apt install python3-pyatspi` |

---

## 22. File Layout & Module Tree

### Source Structure

```
missy/                          # 99 Python files
├── agent/                      # Agent runtime and subsystems
│   ├── runtime.py              # Main agentic loop
│   ├── circuit_breaker.py      # Failure isolation (Closed/Open/HalfOpen)
│   ├── context.py              # Token budget management
│   ├── done_criteria.py        # Task verification prompts
│   ├── learnings.py            # Cross-task learning extraction
│   ├── prompt_patches.py       # Self-tuning prompt patches
│   ├── sub_agent.py            # Task decomposition + parallel execution
│   ├── approval.py             # Human-in-the-loop gate
│   ├── heartbeat.py            # Periodic health checks
│   ├── failure_tracker.py      # Strategy rotation on repeated failures
│   ├── checkpoint.py           # Run state persistence
│   ├── watchdog.py             # Deadlock detection
│   └── proactive.py            # Proactive task initiation
├── channels/                   # I/O interfaces
│   ├── base.py                 # BaseChannel, ChannelMessage
│   ├── cli_channel.py          # Interactive terminal
│   ├── webhook.py              # HTTP POST ingress
│   ├── discord/                # Full Discord integration
│   │   ├── config.py           # DM/guild/role policies
│   │   ├── channel.py          # Access control + message handling
│   │   ├── gateway.py          # WebSocket client (heartbeat, resume)
│   │   ├── commands.py         # Slash commands (/ask, /status, etc.)
│   │   └── rest.py             # HTTP API client
│   └── voice/                  # Voice assistant
│       ├── channel.py          # Orchestrator
│       ├── server.py           # WebSocket server
│       ├── registry.py         # Device management
│       ├── pairing.py          # PBKDF2 token auth
│       ├── presence.py         # Occupancy tracking
│       ├── stt/                # Speech-to-text
│       │   ├── base.py
│       │   └── whisper.py      # faster-whisper
│       └── tts/                # Text-to-speech
│           ├── base.py
│           └── piper.py        # piper binary
├── cli/                        # CLI entry points
│   ├── main.py                 # Click command tree
│   ├── wizard.py               # Interactive setup
│   ├── oauth.py                # OpenAI PKCE flow
│   └── anthropic_auth.py       # Anthropic setup-token
├── config/                     # Configuration
│   ├── settings.py             # Dataclass schemas, load_config()
│   └── hotreload.py            # ConfigWatcher (watchdog)
├── core/                       # Shared infrastructure
│   ├── session.py              # Session/task ID management
│   ├── events.py               # AuditEvent, EventBus
│   └── exceptions.py           # Exception hierarchy
├── gateway/                    # HTTP gateway
│   └── client.py               # PolicyHTTPClient
├── memory/                     # Persistence
│   ├── store.py                # JSON memory store
│   ├── sqlite_store.py         # SQLite FTS5 store
│   └── resilient.py            # Fallback wrapper
├── mcp/                        # Model Context Protocol
│   ├── manager.py              # Server lifecycle
│   └── client.py               # Tool calling
├── observability/              # Monitoring
│   ├── audit_logger.py         # JSONL writer
│   └── otel.py                 # OpenTelemetry exporter
├── plugins/                    # Plugin system
│   ├── base.py                 # Plugin interface
│   └── loader.py               # Dynamic loading
├── policy/                     # Security policies
│   ├── engine.py               # PolicyEngine facade
│   ├── network.py              # CIDR/domain/host checking
│   ├── filesystem.py           # Symlink-aware path control
│   └── shell.py                # Command whitelisting
├── providers/                  # LLM providers
│   ├── base.py                 # Message, CompletionResponse, BaseProvider
│   ├── registry.py             # ProviderRegistry with fallback
│   ├── anthropic_provider.py   # Claude (native tool calling)
│   ├── openai_provider.py      # GPT (native tool calling)
│   ├── ollama_provider.py      # Local (prompted tool calling)
│   └── codex_provider.py       # ChatGPT Codex backend
├── scheduler/                  # Job scheduling
│   ├── manager.py              # APScheduler wrapper
│   ├── parser.py               # Human-to-cron conversion
│   └── jobs.py                 # ScheduledJob dataclass
├── security/                   # Security subsystem
│   ├── sanitizer.py            # Prompt injection detection (13 patterns)
│   ├── secrets.py              # Credential scanning (9 types)
│   ├── censor.py               # Output redaction
│   └── vault.py                # ChaCha20-Poly1305 encrypted store
├── skills/                     # Skill system
│   ├── base.py                 # Skill interface
│   ├── registry.py             # Skill registry
│   └── builtin/
│       ├── system_info.py
│       └── workspace_list.py
└── tools/                      # Tool system
    ├── base.py                 # BaseTool, ToolPermissions, ToolResult
    ├── registry.py             # ToolRegistry (policy-gated execution)
    └── builtin/                # 14 tool files, 20+ tools
```

### Default File Locations

| File | Path | Purpose |
|---|---|---|
| Config | `~/.missy/config.yaml` | Main configuration |
| Audit log | `~/.missy/audit.jsonl` | Structured event log |
| Memory DB | `~/.missy/memory.db` | SQLite FTS5 conversation store |
| Jobs | `~/.missy/jobs.json` | Scheduler persistence |
| MCP | `~/.missy/mcp.json` | MCP server config |
| Devices | `~/.missy/devices.json` | Voice edge node registry |
| Vault key | `~/.missy/secrets/vault.key` | 32-byte encryption key |
| Vault data | `~/.missy/secrets/vault.enc` | Encrypted secrets |
| Patches | `~/.missy/patches.json` | Prompt patch store |
| Browser | `~/.missy/browser_sessions/` | Playwright profiles |
| Workspace | `~/workspace` | Default working directory |

---

## 23. Test Suite

- **48 test files** across 16 subdirectories
- **948+ tests**, ~86% coverage
- Coverage threshold: 85% (configured in `pyproject.toml`)
- Framework: pytest + pytest-asyncio + pytest-cov

### Test Organization

```
tests/
├── agent/          # Runtime, circuit breaker, context, learnings, patches, sub-agent
├── channels/       # Discord, webhook, voice
├── cli/            # CLI commands
├── config/         # Settings, hot-reload
├── core/           # Sessions, events, exceptions
├── integration/    # End-to-end flows
├── memory/         # JSON store, SQLite store, resilient layer
├── observability/  # Audit logger, OTLP
├── plugins/        # Plugin loading
├── policy/         # Network, filesystem, shell policies
├── providers/      # Anthropic, OpenAI, Ollama, registry
├── scheduler/      # Job management, parser
├── security/       # Sanitizer, secrets, censor, vault
├── skills/         # Skill registry
├── tools/          # Calculator, file ops, shell, web fetch, browser, X11
└── unit/           # Additional unit tests
```

---

## 24. Comparison Matrix

Use this matrix to compare Missy against other agentic systems:

| Capability | Missy | Other System |
|---|---|---|
| **Core** | | |
| Multi-step agentic loop | Yes (configurable max_iterations) | |
| Native tool calling | Yes (Anthropic, OpenAI) | |
| Prompted tool calling fallback | Yes (Ollama) | |
| Streaming responses | Yes | |
| Multi-provider support | 4 (Anthropic, OpenAI, Ollama, Codex) | |
| Provider fallback | Yes (automatic) | |
| API key rotation | Yes (api_keys list) | |
| Model tiering | Yes (fast/default/premium) | |
| | | |
| **Security** | | |
| Default-deny policy | Yes (network, filesystem, shell) | |
| Prompt injection detection | Yes (13 patterns) | |
| Credential scanning | Yes (9 types) | |
| Output secret redaction | Yes | |
| Encrypted vault | Yes (ChaCha20-Poly1305) | |
| Symlink-aware filesystem policy | Yes | |
| Network CIDR + domain policy | Yes | |
| Shell command whitelisting | Yes | |
| Full audit trail | Yes (JSONL + OTLP) | |
| | | |
| **Tools** | | |
| Filesystem (read/write/delete/list) | Yes (4 tools) | |
| Shell execution | Yes (bash, timeout, truncation) | |
| Web fetch | Yes (policy-gated) | |
| Browser automation | Yes (Playwright Firefox, persistent sessions) | |
| Desktop automation (X11) | Yes (screenshot, click, type, key, window list) | |
| AI vision (screen reading) | Yes (Ollama local, no API key) | |
| Accessibility (AT-SPI) | Yes (GTK widget tree) | |
| Calculator | Yes (safe AST eval) | |
| Dynamic tool creation | Yes (self_create_tool) | |
| MCP tool integration | Yes (namespaced, auto-restart) | |
| | | |
| **Channels** | | |
| CLI (interactive REPL) | Yes | |
| Discord (full Gateway API) | Yes (access control, slash commands) | |
| Webhook (HTTP ingress) | Yes | |
| Voice (WebSocket + STT/TTS) | Yes (faster-whisper + Piper) | |
| Edge device management | Yes (pairing, policy modes) | |
| | | |
| **Agent Intelligence** | | |
| Context management (token budget) | Yes (7-tier: system, tools, memory, learnings, history) | |
| Conversation memory | Yes (SQLite FTS5) | |
| Cross-task learning | Yes (task_type, outcome, lesson extraction) | |
| Self-tuning prompt patches | Yes (propose/approve/reject/expire cycle) | |
| Done criteria verification | Yes (compound task detection + verification prompts) | |
| Sub-agent task decomposition | Yes (dependency graph, max 3 concurrent) | |
| Strategy rotation on failure | Yes (failure tracker, threshold=3) | |
| | | |
| **Reliability** | | |
| Circuit breaker | Yes (Closed/Open/HalfOpen, exponential backoff) | |
| Checkpointing | Yes (per tool-call round) | |
| Watchdog | Yes (deadlock detection) | |
| Human-in-the-loop approval | Yes (timeout, approve/deny) | |
| Resilient memory (fallback) | Yes (SQLite → JSON fallback) | |
| | | |
| **Operations** | | |
| Job scheduling | Yes (APScheduler, cron + human-readable, retry, backoff) | |
| Config hot-reload | Yes (watchdog-based, no restart) | |
| OpenTelemetry export | Yes (gRPC + HTTP/protobuf) | |
| Health check (`doctor`) | Yes | |
| Setup wizard | Yes (interactive, OAuth) | |
| | | |
| **Extensibility** | | |
| Plugin system | Yes (dynamic loading) | |
| Skill system | Yes (registry) | |
| MCP servers | Yes (stdio + HTTP, namespaced) | |
| Custom providers | Yes (BaseProvider interface) | |
| Runtime tool creation | Yes (self_create_tool) | |
