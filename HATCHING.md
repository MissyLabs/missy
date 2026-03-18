# Hatching System

Hatching is Missy's first-run bootstrapping process. It establishes the agent's identity, validates the environment, and ensures all subsystems are ready.

## Quick Start

```bash
missy hatch
```

## What Hatching Does

The hatching process runs 7 steps in order:

1. **Validate Environment** — Checks Python >= 3.11, creates `~/.missy/` directory, verifies filesystem permissions
2. **Initialize Config** — Creates `~/.missy/config.yaml` with secure defaults if it doesn't exist
3. **Verify Providers** — Checks for API keys (env vars or config) for at least one AI provider
4. **Initialize Security** — Creates `~/.missy/secrets/` directory (mode 700), notes agent identity status
5. **Generate Persona** — Creates `~/.missy/persona.yaml` with default personality configuration
6. **Seed Memory** — Initializes the SQLite memory database with a welcome entry
7. **Finalize** — Marks hatching as complete, records timestamp

## Recovery

Hatching is resumable. If interrupted, running `missy hatch` again will skip already-completed steps and resume from where it left off. A `FAILED` state is automatically retried.

## Non-Interactive Mode

```bash
missy hatch --non-interactive
```

Skips all user prompts and uses defaults. Useful for automated deployments.

## State File

Hatching state is stored at `~/.missy/hatching.yaml`:

```yaml
status: hatched
started_at: "2025-01-15T10:30:00"
completed_at: "2025-01-15T10:30:02"
steps_completed:
  - validate_environment
  - initialize_config
  - verify_providers
  - initialize_security
  - generate_persona
  - seed_memory
  - finalize
persona_generated: true
environment_validated: true
provider_verified: true
security_initialized: true
memory_seeded: true
error: null
```

## Hatching Log

All hatching events are logged to `~/.missy/hatching_log.jsonl` in structured JSONL format. Each entry includes timestamp, step name, status, and message.

## Re-Hatching

To start fresh:

```bash
# In Python or via future CLI command
from missy.agent.hatching import HatchingManager
mgr = HatchingManager()
mgr.reset()
```

Then run `missy hatch` again.

## Integration

The hatching system is checked during `missy run` and `missy ask`. If Missy has not been hatched, users are prompted to run `missy hatch` first. The persona generated during hatching is loaded by the agent runtime to shape all subsequent responses.
