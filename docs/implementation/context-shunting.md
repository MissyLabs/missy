# Provider-neutral context shunting

Missy implements the core idea from Spotify's [Portal/AiKA token-usage
case study](https://engineering.atspotify.com/2026/9/portal-by-spotify-cut-my-claude-code-token-usage-by-90): large, mechanical input should be
processed by an ephemeral worker while only a compact answer enters the
frontier model's context. Spotify's implementation calls these paths
`bulk-reader` and `code-writer`; its published examples report 82–94% less
frontier-model context usage, with a 90% mean. Those figures are workload
results, not a guarantee for Missy.

## Missy's equivalent

The provider-neutral flow uses existing runtime boundaries rather than a
Claude-specific hook:

1. Tool output above 16,000 characters is stored as a session-scoped `ref_*`
   record. The parent sees only size metadata, a short preview, and the ID.
2. The parent calls `context_shunt(item_ids=[...], question=...)` for a focused
   summary, search, comparison, or extraction.
3. `context_shunt` loads the records and makes one stateless call through
   `BaseProvider.complete()` to the configured worker.
4. Only the bounded, scanned worker answer and usage metrics return to the
   parent context. The worker still consumed the corpus, so the reported
   percentage is explicitly **parent/frontier context avoided**, not total
   tokens eliminated.

This common path works with the built-in `anthropic`, `openai`,
`openai-codex`, `ollama`, and `acpx` providers. Anthropic and OpenAI receive a
per-call output-token limit; Ollama receives `num_predict`; Codex and ACPX are
bounded after completion because their current adapters do not expose a
per-call token limit. All five are subject to the same hard returned-output
cap.

## Configuration and egress

Worker routing is operator configuration, never a model argument:

```yaml
providers:
  anthropic:
    name: anthropic
    model: claude-sonnet-4-6
    fast_model: claude-haiku-4-5
    context_worker_provider: ollama
    context_worker_model: qwen3:8b

  ollama:
    name: ollama
    model: qwen3:8b
    base_url: http://localhost:11434
```

An empty `context_worker_provider` uses the parent provider, which introduces
no new provider boundary. Naming another provider is explicit authorization to
send the stored corpus there. Missy does not silently fall back to another
provider if the configured worker fails: that would violate the operator's
data-egress and cost choice.

If `context_worker_model` is empty, the worker provider's `fast_model` is used
when configured, followed by its primary `model`. Per-call model overrides are
best-effort for Codex and ACPX because those adapters currently select their
model internally.

## Security properties

- IDs must be `ref_*` records owned by the current session. Runtime-injected
  session, runtime, task, and parent-provider fields overwrite any hidden
  arguments invented by a model.
- The question and **entire corpus** are prompt-injection scanned in bounded,
  overlapping chunks immediately before worker egress. A detection or scanner
  error fails closed and makes no provider call.
- Worker input is JSON-encoded and system-labeled as untrusted data. Worker
  output is scanned again and omitted on detection or scan failure.
- Calls are capped at eight references, 400,000 combined source characters,
  and 4,096 returned tokens. Oversized input is refused rather than silently
  truncated.
- Worker usage is charged to the same session budget and audited without
  logging source or answer content.

`memory_expand` remains available for a small exact excerpt, but feeding a
large expansion back into the parent defeats the context boundary.

## The `code-writer` analogue

Use `delegate_task` for independent boilerplate generation that can be written
directly to disk, selecting an explicit provider on the delegated agent when
desired. Keep architecture, debugging, security-sensitive changes, and final
review with the parent model. This follows the Portal article's own limitation:
the worker is useful for mechanical transformations, but its example missed a
thread-safety defect when given work that required deeper engineering
judgment.

The reference implementation described by Spotify is available in the
[Shunt plugin source](https://github.com/sorantis/portal-ai-plugins/tree/add-shunt-claude/plugins/shunt).
