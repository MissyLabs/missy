# Scoped native retrieval and graph memory

`Retrieval::open(absolute_database_file)` opens only an explicitly supplied path in
an existing current-user-owned private directory (0700). The caller creates that
directory. Database and SQLite sidecars must be owner-only regular files with no
hardlinks or symlinks; nonprivate existing files are rejected, not silently fixed.
Symlink ancestors are rejected. As with the policy crate, same-UID directory swaps,
mount manipulation and a malicious database owner are outside this userspace
boundary. Use a separate service user for stronger isolation.

All operations require a validated `missy_core::Scope` containing tenant, user,
channel and session. Even direct document/node IDs and every graph hop include the
whole scope. There is no default scope, live legacy-state import, provider, model
download or GPU access. Retrieved text/provenance are **untrusted data**, never
system instructions. The runtime must preserve this boundary when rendering them.

## API

- `ingest(scope, id, text, provenance) -> IngestReport`
- `ingest_file(scope, id, path, &Policy) -> IngestReport`, using only descriptor-based
  policy reads with explicit roots and no symlink traversal. Text must be valid UTF-8.
- `search(scope, query, limit) -> Vec<Hit>`
- `get_document(scope, id) -> Option<Document>`
- `drop_document(scope, id) -> bool`
- `put_node(scope, &GraphNode)`, `get_node(scope,id)`, `drop_node(scope,id)`
- `put_edge(scope, &GraphEdge)`, `drop_edge(scope,source,target,relationship)`
- `traverse(scope, start_id, depth, limit) -> GraphTraversal`

Search uses local Okapi BM25 (k1=1.5,b=.75) plus cosine similarity of SHA-256 hashed
token-frequency lexical features, fused using reciprocal rank fusion (k=60,
200 candidates per ranking). **Hashing vectors are not neural semantic embeddings**
and do not supply model-level paraphrase understanding. Hash collisions can yield
false positive candidates. Ties are stable by document ID and chunk ordinal.
`Embedder` is a pluggable deterministic local interface; `open_with_embedder` rejects
stored identity/dimension mismatches rather than silently mixing embedding spaces.
Changing a model requires explicit reindexing into a fresh database.

Documents use deterministic 2048-byte windows, 256-byte overlap rounded to UTF-8
boundaries. `Hit.start_byte..end_byte` is an exact half-open **UTF-8 byte** slice of
the original document, not Python character offsets. No whitespace is normalized.
SHA-256 document hashes, chunk checksums, source provenance and SQLite transactions
make unchanged ingestion idempotent and replacement atomic, removing all stale
chunks. Checksums detect accidental corruption, not a malicious DB owner rewriting
data and checksums together. Schema version/application ID, quick/foreign-key checks
and strict tables reject incompatible or structurally corrupted databases.

Limits: 1 MiB/document, 8 MiB text/scope, 256 documents/scope, 8192 chunks/scope,
4096 query bytes, 1..50 hits, ~128 KiB serialized search/graph output. Full document
reads return up to the separate 1 MiB document limit. The physical SQLite page cap
is 256 MiB (65536 default 4096-byte pages), including all scopes. Corpus-wide token
statistics and ranking are computed on bounded scoped chunks at query time rather
than using FTS expressions, eliminating the FTS-query-language injection surface.

Graph has explicit nodes and directed edges, no automatic entity extraction,
Cypher, arbitrary SQL or heuristic relationship claims. Entity types and relation
names use the legacy allowlists. Each scope permits 2048 nodes and 4096 edges.
Traversal is cycle-safe breadth-first, depth 0..8, 1..100 nodes and edges each,
bounded SQL row reads and output. `truncated` signals row/output exhaustion, not
the intentional depth horizon. Node deletion cascades to edges within that scope.

Test with `cargo test -p missy-retrieval`; lint with
`cargo clippy -p missy-retrieval --all-targets -- -D warnings`.
