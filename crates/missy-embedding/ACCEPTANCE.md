# Local neural acceptance evidence

Observed 2026-11-05 on localhost, isolated project test only.

- Native adapter and acceptance binary, no Python.
- Existing llama.cpp executable: `/home/odin/projects/llama.cpp/build-cuda/bin/llama-server`
- `--version`: `1 (74ce157)`, GNU 13.3.0, Linux x86_64.
- Dedicated ephemeral CPU-only server, two threads, GPU layers zero, mean pooling.
- Exact weights size: **45,949,216 bytes**.
- Exact model SHA256 verified both after HTTPS download and inside native example:
  `797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662`.
- Model source/revision documented in README; no credentials used.
- All output vectors exactly **384 dimensions** and L2-normalized by the adapter.
- Repeated query cosine >0.99999.
- Acceptance process exit **0**, child server killed and waited/reaped.
- Native example's startup plus inference elapsed **0.278470252 seconds**.
- Independent repeat under `/usr/bin/time -v`: exit **0**, total wall time
  **1.28 seconds** including SHA256 verification, maximum resident set size
  **170,820 KiB**, **0 swaps**. Inference/startup subtotal **0.297051325 seconds**.
  These are observed bounds for this fixed model smoke test, not an enforced
  resource limit on arbitrary operator-managed external model servers.

## Actual measured cosine rankings

| Case | Neural relevant | Neural lexical decoy | Hashing relevant | Hashing decoy | Required margin >0.03 |
|---|---:|---:|---:|---:|---|
| Vehicle paraphrase | 0.6244148811 | 0.4903519881 | 0 | 0.2461829931 | PASS |
| Sleep paraphrase | 0.5509142825 | 0.3858860655 | 0 | 0.2828427072 | PASS |
| Medical paraphrase | 0.7244829208 | 0.5074647482 | 0.1443375701 | 0.2236067924 | PASS |
| Adversarial exact quotation diagnostic | 0.6244148811 | 0.6239420539 | 0 | 0.6546536823 | FAIL, nonrequired diagnostic |

The source example includes every query, relevant text and decoy. All three
required cases rank the semantic paraphrase above the lexical decoy, while the
existing hashing embedder control ranks the decoy above the paraphrase.

Persistent `Retrieval::open_with_embedder` used an isolated owner-only temporary
database. Ingested mechanic and astronomy documents; query
`How do I repair my automobile?` returned **mechanic first** despite its text
`A mechanic can fix your broken car.` sharing no lexical tokens with the query.

## Limitations preserved, not hidden

The original exact-query-quotation decoy nearly ties the genuine paraphrase.
An initial acceptance demanding a 0.03 margin for that case failed. It remains
visible in the final example as a diagnostic with `required: false`; three
separate ordinary lexical keyword decoys constitute the required smoke corpus.
This is real neural inference and a small semantic smoke test, not evidence of
robust adversarial retrieval, general quality, or authenticated HTTP weights.

The adapter validates the response model echo and binds the operator-declared
weights fingerprint to its identity. Only this isolated acceptance verifies the
actual GGUF file; a deployment must separately establish which weights its
server loaded. Public model revision and hash pinning establish repeatability,
not immunity from model supply-chain risks.

## Protocol/code checks

`cargo test -p missy-embedding`: **6 passed**. Covers strict explicit local
configuration, serde roundtrip, redacted Debug, input limits, identity binding,
working inside Tokio current-thread runtime, normalization, malformed/wrong
model/dimension/index/count/nonfinite/base64/zero responses, cancellation,
bounded overload, and joining the worker on drop.

`cargo clippy -p missy-embedding --all-targets -- -D warnings`: **PASS**.

Fixture HTTP tests are deliberately not presented as neural semantic evidence.
