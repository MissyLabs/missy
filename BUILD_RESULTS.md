# BUILD_RESULTS

- Timestamp: 2026-07-08
- Branch: overhaul/tools-20260708-020326
- Base: origin/master

## Repository Snapshot

This rebase replayed the tool-intelligence overhaul commits on top of the Discord diagnostics work in `origin/master`.

## Code Areas Present After Rebase

- Tool intelligence runtime wiring in `missy/agent/runtime.py`.
- Provider schema adapter wiring in `missy/providers/anthropic_provider.py`, `missy/providers/openai_provider.py`, and `missy/providers/ollama_provider.py`.
- `missy tools benchmark run` in `missy/cli/main.py`.
- Discord Gateway diagnostics, Discord voice binding, Discord voice tools, and related tests from `origin/master`.

## Verification To Rerun

- Tool-intelligence focused tests.
- Discord-focused tests touched by `origin/master`.
- CLI command tests where `missy/cli/main.py` auto-merged.
- Ruff check and format check.
