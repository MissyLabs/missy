# Architecture decision records

Short records of decisions that are easy to undo by accident. Each states the
context, the decision, and its consequences. Add a new numbered file rather
than editing an accepted one; mark superseded records as such.

| ADR | Decision | Status |
|---|---|---|
| [0000](0000-subsystem-design-history.md) | Subsystem design history moved from CLAUDE.md | Historical |
| [0001](0001-injection-patterns-need-a-second-signal.md) | Prompt-injection patterns need a second signal | Accepted |
| [0002](0002-never-open-live-sqlite-files.md) | Never open/close a live SQLite file outside SQLite | Accepted |
| [0003](0003-one-shared-guard-retry-budget.md) | One shared guard-retry budget per turn | Accepted |

Template: `# ADR-NNNN: Title` / **Status** / **Context** / **Decision** / **Consequences**.
