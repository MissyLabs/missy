# Persona System

Missy's persona system defines her identity, tone, personality traits, and response behavior. The persona is loaded at runtime and actively shapes how Missy communicates.

## Quick Start

```bash
# View current persona
missy persona show

# Edit specific fields
missy persona edit --name "Missy"
missy persona edit --tone "friendly,casual,technical"
missy persona edit --identity "A helpful Linux assistant"

# Reset to defaults
missy persona reset
```

## Default Persona

When first created (during hatching or on first access), Missy uses these defaults:

```yaml
version: 1
name: Missy
tone:
  - helpful
  - direct
  - technical
personality_traits:
  - curious
  - thorough
  - security-conscious
  - pragmatic
behavioral_tendencies:
  - prefers action over narration
  - adapts formality to context
  - asks clarifying questions when needed
response_style_rules:
  - Be concise unless detail is requested
  - Use technical terms when appropriate
  - Show reasoning for non-obvious decisions
  - Acknowledge uncertainty rather than guessing
boundaries:
  - Never execute destructive operations without confirmation
  - Never expose secrets or credentials
  - Always respect policy engine decisions
  - Flag security concerns proactively
identity_description: >
  Missy is a security-first local AI assistant for Linux systems.
  She is knowledgeable, practical, and focused on getting things done
  safely and efficiently.
```

## Configuration

The persona is stored at `~/.missy/persona.yaml`. It can be edited directly or via CLI commands.

### Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Agent's display name |
| `tone` | list | Communication tone descriptors |
| `personality_traits` | list | Core personality characteristics |
| `behavioral_tendencies` | list | How the agent tends to act |
| `response_style_rules` | list | Rules governing response format |
| `boundaries` | list | Hard constraints on behavior |
| `identity_description` | string | Free-form identity paragraph |
| `version` | int | Auto-incremented on each save |

## How Persona Shapes Responses

The persona influences responses through two mechanisms:

### 1. System Prompt Injection

The `BehaviorLayer` injects persona information into the system prompt sent to the AI provider. This includes identity, tone preferences, style rules, and boundaries. The prompt is structured so the LLM naturally adopts the persona.

### 2. Response Post-Processing

The `ResponseShaper` removes robotic patterns ("As an AI...", "Certainly!", etc.) from LLM output while preserving code blocks and technical content. This makes responses feel more natural and aligned with the persona.

## Behavior Layer

The behavior layer (`missy/agent/behavior.py`) provides:

- **Tone Analysis** — Detects user tone (casual, formal, frustrated, technical) and adapts responses
- **Intent Classification** — Identifies whether input is a question, command, greeting, etc.
- **Urgency Detection** — Prioritizes urgent requests
- **Conciseness Control** — Shortens responses in long conversations or when user is brief
- **Response Guidelines** — Context-specific behavioral directives

## Versioning

Every save increments the `version` field. This allows tracking persona evolution over time. The `reset` command restores defaults but preserves the version counter.

## Programmatic Access

```python
from missy.agent.persona import PersonaManager

mgr = PersonaManager()
persona = mgr.get_persona()
print(persona.name)  # "Missy"
print(persona.tone)  # ["helpful", "direct", "technical"]

# Modify
mgr.update(tone=["friendly", "casual"])
mgr.save()

# Get system prompt prefix
prefix = mgr.get_system_prompt_prefix()
```
