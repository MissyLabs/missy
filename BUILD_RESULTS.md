# Build Results

## Build Date: 2026-03-18

## Summary

Missy is a security-first, self-hosted local agentic AI assistant for Linux. It provides a production-grade agent platform with strict security controls, policy enforcement, and full auditability.

## Feature Parity with OpenClaw

| Feature | Status | Notes |
|---|---|---|
| Multi-provider support | Done | Anthropic, OpenAI, Ollama, Codex |
| Tool calling loop | Done | Multi-step with iteration limits |
| Memory persistence | Done | SQLite FTS5 + optional FAISS vector |
| Policy enforcement | Done | Network, filesystem, shell, REST L7 |
| Audit logging | Done | JSONL + OpenTelemetry |
| Hatching/bootstrap | Done | 7-step first-run flow |
| Persona system | Done | YAML-backed, runtime-loaded |
| Behavior layer | Done | Tone analysis, intent classification, response shaping |
| Channel support | Done | CLI, Discord, webhook, voice |
| MCP integration | Done | With digest pinning |
| Skill discovery | Done | SKILL.md format |
| Plugin system | Done | With permissions |
| Scheduling | Done | APScheduler, cron, timezone |
| Secret management | Done | ChaCha20-Poly1305 vault |
| Container sandbox | Done | Docker isolation |

## Module Count

- Source files: 157
- Test files: 270+
- Total tests: 7830 (0 failures, 17 skipped)

## Key Metrics

- Security patterns detected: 250+ injection, 37+ credential patterns
- Provider support: 4 (Anthropic, OpenAI, Ollama, Codex)
- Channels: 4 (CLI, Discord, Webhook, Voice)
- CLI commands: 40+
