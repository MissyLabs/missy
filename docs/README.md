# Missy Documentation

## User Guides

| Guide | Description |
|---|---|
| [Operations](operations.md) | Installation, setup, running, monitoring, backup |
| [Configuration](configuration.md) | Complete YAML reference with annotated examples |
| [Providers](providers.md) | Anthropic, OpenAI, Ollama setup and API key management |
| [Discord](discord.md) | Discord bot integration, access control, slash commands |
| [Scheduler](scheduler.md) | Job scheduling with human-friendly syntax |
| [Skills & Plugins](skills-and-plugins.md) | Extension system: tools, skills, plugins |
| [Troubleshooting](troubleshooting.md) | Common errors and diagnostic procedures |

## Architecture & Security

| Guide | Description |
|---|---|
| [Architecture](architecture.md) | System design, data flow, module dependencies |
| [Security](security.md) | Security policy, hardening guide, vulnerability reporting |
| [Threat Model](threat-model.md) | Attack vectors and mitigations |
| [Memory & Persistence](memory-and-persistence.md) | Conversation memory, learnings, storage schema |
| [Testing](testing.md) | Test suite layout, coverage, writing tests |

## Voice Edge Nodes

| Guide | Description |
|---|---|
| [Edge Node Spec](voice-edge-spec.md) | Hardware target, protocol specification, pairing workflow |
| [missy-edge repo](https://github.com/MissyLabs/missy-edge) | Raspberry Pi client implementation |

## Implementation Deep-Dives

| Reference | Source |
|---|---|
| [Agent Loop](implementation/agent-loop.md) | `missy/agent/runtime.py` |
| [Policy Engine](implementation/policy-engine.md) | `missy/policy/` |
| [Provider Abstraction](implementation/provider-abstraction.md) | `missy/providers/` |
| [Network Client](implementation/network-client.md) | `missy/gateway/client.py` |
| [Discord Channel](implementation/discord-channel.md) | `missy/channels/discord/` |
| [Audit Events](implementation/audit-events.md) | `missy/observability/` |
| [Persistence Schema](implementation/persistence-schema.md) | `missy/memory/`, `missy/scheduler/` |
| [Scheduler Execution](implementation/scheduler-execution.md) | `missy/scheduler/` |
| [Module Map](implementation/module-map.md) | Full import dependency graph |
| [Manifest Schema](implementation/manifest-schema.md) | Plugin/skill manifests |
