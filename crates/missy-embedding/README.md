# Explicit local neural embeddings

`LocalHttpEmbedder::new(LocalEmbeddingConfig)` implements the synchronous
`missy_retrieval::Embedder` trait. Supply it as an `Arc` to
`Retrieval::open_with_embedder`. There is no Python/native-process invocation in the
adapter, no hash-feature fallback, no model auto-download, and no network call at
construction. The operator supplies a local OpenAI-compatible embedding server.

## Configuration

`LocalEmbeddingConfig` is strict serde JSON (`deny_unknown_fields`), with all
fields required. The following describes the CPU MiniLM acceptance setup; choose
a stable dedicated port for a persistent retrieval database.

```json
{
  "endpoint": "http://127.0.0.1:18081/v1/embeddings",
  "allowed_origin": "http://127.0.0.1:18081",
  "allow_private_network": true,
  "model": "acceptance-minilm",
  "model_sha256": "797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662",
  "dimensions": 384,
  "timeout_seconds": 5
}
```

Only literal loopback IPv4/IPv6 is accepted, not remote origins, LAN IPs, DNS
names, URL credentials, queries, fragments, or arbitrary paths. The origin must
exactly match the normalized origin, and the private-network flag must explicitly
be true. Requests always pass through `PolicyHttpClient`, inheriting its DNS/IP
policy, response/request size bounds, no proxy/redirect bypass, and transport
timeouts. No authentication headers or environment secrets are supported.
`Debug` redacts endpoint and model. Do not put secrets in model identifiers.

The response must have a matching `model`, exactly one `data` item at index zero,
the exact configured dimension, and nonzero finite float values. Vectors are
L2-normalized. Base64 embeddings and missing/mismatched model identifiers fail
closed, even if a looser server accepts them.

The stable embedding identity includes length-framed endpoint, exact model name,
dimension, and normalized operator-supplied weights SHA256. It changes if any
of those change. **An HTTP model echo does not attest weights.** The operator
must ensure the supplied digest belongs to the actual loaded weights. Endpoint
changes require rebuilding the retrieval index because the endpoint is part of
the identity. This crate deliberately cannot label arbitrary server output as
proven neural inference. The independent acceptance below verifies real weights
before launching its own dedicated server.

## Bounds and sync/async behavior

- One dedicated OS thread owns a current-thread Tokio runtime, so use inside an
  existing Tokio runtime never attempts nested `block_on`.
- One active request and one queued job; excess callers get a busy error.
- Nonblank UTF-8 input at most 16 KiB; dimensions 1..4096; timeout 1..120 seconds.
- The deadline includes queue time and applies to the entire async request.
  The blocking caller cancels the future on timeout; timed-out jobs cannot grow
  an unbounded orphan queue.
- Drop signals cancellation, joins the worker and shuts its runtime down.
- The synchronous trait necessarily blocks its calling thread. Async callers
  should put complete retrieval operations in `tokio::task::spawn_blocking`.
  This adapter does not pretend a synchronous trait is nonblocking.

## Native neural acceptance

`cargo test -p missy-embedding` uses only synthetic local HTTP fixtures. Those
tests verify protocol and policy, **not** neural inference or semantic quality.

`examples/neural_acceptance.rs` separately performs genuine native model
inference against a server it launches and kills/reaps itself. It accepts only
the exact SHA256/size of the vetted Apache-2.0 MiniLM F16 model below. It never
downloads anything and never probes an existing model service.

- Public model: `second-state/All-MiniLM-L6-v2-Embedding-GGUF`
- Pinned revision: `544f204f2eaa2d71361ffc74d6df7170285b286a`
- File: `all-MiniLM-L6-v2-ggml-model-f16.gguf`, 45,949,216 bytes
- SHA256: `797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662`
- Download URL: `https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/544f204f2eaa2d71361ffc74d6df7170285b286a/all-MiniLM-L6-v2-ggml-model-f16.gguf`

After explicitly obtaining that public file, run:

```sh
cargo run -p missy-embedding --example neural_acceptance -- \
  /absolute/path/to/llama-server /absolute/path/to/model.gguf
```

The acceptance server uses its own ephemeral loopback port and model alias,
two CPU threads, no GPU layers, empty CUDA device visibility, 512 token context,
one slot and mean pooling. Startup is bounded to 30 seconds, each embedding to
5 seconds, and every error path reaps the child through RAII. No live service
is restarted or used. It tests repeat-vector stability, three semantic
paraphrase rankings against lexical decoys with a lexical hashing control, and
persistent retrieval ingest/search. The tiny hand-authored corpus is a smoke
test, not a general semantic benchmark. The difficult exact-quotation case is
reported as a nonrequired diagnostic, including its failed 0.03 margin.

Observed evidence is in `ACCEPTANCE.md`. Model binaries and run logs are ignored
under `acceptance-cache/`; do not commit weights.
