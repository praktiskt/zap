# zap — in-memory full-text search engine

**Blazingly fast, single-binary text search in Rust.** Everything lives in RAM — the index,
the docs, the HTTP API and the query loop share one process. Built for scale: millions of
documents, texts 100k+ characters long.

```
top-10 query (10k docs):  single hot term 48 µs • AND 94 µs • phrase 290 µs
1M docs (8 lanes):        ingest 16 s  •  query 0.36 ms
skewed tf (block-max WAND): 90 ns @ 200k docs
ingest: 10k × 200w docs in 320 ms  •  1M × 50w docs in 16 s (60k docs/s)
```

## What's inside
- **BM25 ranking** (Lucene-style idf) + phrase-window bonus + per-field boosts
- **Embedding search**: schema-gated vector fields, exact flat scan (AVX2/FMA), and
  hybrid fusion of keyword + embedding (weighted or RRF) — you supply the embeddings
- **External doc ids + metadata**: user-assigned unique string ids (duplicate ingest
  replaces), arbitrary stored-only JSON metadata, get/delete-by-id over lock-free
  snapshots; tombstones mask deleted docs and are physically purged by merge
- **Full query language**: boolean ops with parens, phrases with slop, `field:` scoping,
  prefix/wildcard globs, fuzzy `~N` (Levenshtein DFA over the term FST)
- **Segments**: immutable sealed segments (FST term dict + delta-varint postings +
  skip anchors), parallel hot-ingest lanes, log-tier merge
- **block-max WAND**: live top-k thresholds skip whole 128-doc blocks
- **arc-swap snapshots**: single writer, lock-free readers; commits/merges never block queries
- **charabia tokenization**: Unicode segmentation, NFC folding, CJK segmentation, snowball
  stemming for Latin tokens
- **doc store**: lz4 blocks per doc (raw for tiny docs) → snippets + ANSI highlighting
- **HTTP serve mode** (tiny_http threads, no async runtime) and an interactive query loop

## Query syntax

| input | meaning |
|---|---|
| `fox dog` | implicit AND of both terms |
| `a AND b` / `a OR b` | boolean operators (uppercase) |
| `a NOT b` | exclusion matching (subtract docs) |
| `"quick brown fox"` | exact phrase (consecutive positions) |
| `"ran and fast"~1` | phrase with slop (1 token gap allowed, order-preserving) |
| `field:term` / `field:(a OR b)` | restrict a subtree to one field |
| `prefix*` | prefix expansion from the term FST (range walk) |
| `wild*car?` | glob wildcard (`*` = any run, `?` = one char) |
| `typo~1` | fuzzy match, distance 1 or 2 (DFA over term dictionary) |
| `( a OR b ) AND NOT c` | grouping, operators |
| `:stats` / `:help` / `:q` | query-loop meta commands |

## Usage

```sh
# index files/dirs, search interactively (every stdin line = one query)
zap dir/ file.md
# per-line documents; blank line ends docs, the rest is queries
{ echo "doc one"; echo; echo "some query"; } | zap --stdin-lines --lanes 8
# block-max WAND needs BM25 params; defaults are k1=1.2 b=0.75
zap --k1 1.2 --b 0.75 dir/
# opt-in persistence: state loads from DIR at boot and every commit is
# persisted there (segments + MANIFEST.json, atomic rewrite) — reboot-safe
zap serve --bind 127.0.0.1:3279 --disk ./data
```

Input docs: files/dirs (`.txt .md .rst .json .toml .csv .log .xml`; unknown extensions tried
as text); `--stdin` = whole stdin is one doc; `--stdin-lines` = one doc per line until a
blank line. Common flags: `-k N`, `--lanes N`, `--no-stem`, `--plain`, `--snippet N`,
`--commit-every N`, `--disk DIR`.

## Embedding search

zap stores and searches **embeddings you compute**. The model stays on your side: you
embed the text, zap takes the numbers. Declare a vector field, send the vector with
the document, send the query vector with the search.

```sh
# declare the field: name:dim[:metric] (metric = cosine | dot | l2, default cosine)
zap serve --bind 127.0.0.1:3279 --vector emb:768

# ingest: a number array is an embedding (a string is a text field)
curl -X POST localhost:3279/ingest \
  -d '{"id":"doc-1","text":"the quick brown fox","emb":[0.012, -0.44, ...]}'

# search by embedding only
curl -X POST localhost:3279/search \
  -d '{"vector":[0.012, -0.44, ...], "k":5}'

# hybrid: keyword AND embedding, fused (alpha = keyword weight)
curl -X POST localhost:3279/search \
  -d '{"q":"quick brown","vector":[0.012, -0.44, ...], "k":5, "mode":"hybrid", "alpha":0.5}'

# rank the keyword matches by the embedding (the keyword selects, the vector ranks)
curl -X POST localhost:3279/search \
  -d '{"q":"quick brown","vector":[0.012, -0.44, ...], "mode":"filter"}'
```

| mode | meaning | fusion |
|---|---|---|
| `keyword` | BM25 only (the default when there is no embedding) | — |
| `vector` | embedding similarity only (cosine/dot/L2 per the field) | — |
| `hybrid` | both, mixed into one score | `weighted` (default): `alpha*bm25 + (1-alpha)*cosine` after min-max normalization; `rrf`: reciprocal rank fusion |
| `filter` | the keyword picks the candidates, the embedding ranks them | — |

The mode defaults to `hybrid` when a query **and** an embedding are present. Result
rows carry the components: `score` (the fused score), `keyword_score`, `vector_score`.

- **Validation is schema-gated**: an unknown field, a wrong dimension or a non-finite
  value is rejected at ingest (400).
- **Storage**: the embeddings live in the segment next to the postings (merge and the
  disk format carry them), 768 dims = 3 KB/doc as f32 — zap stores exactly what you
  send, no quantization layer.
- **Cluster mode**: the shards return candidate rows with their raw per-modality
  scores and the coordinator fuses them globally — a hybrid search ranks identically
  on one node and on a ring.
- **Cost**: the scan is a flat exact pass over the corpus (10k docs x 768 dims =
  ~0.5 ms, 100k x 768 = ~8 ms). The `VectorIndex` trait is the seam for an HNSW
  index when the corpus outgrows the flat scan.

## Repo

```
src/lib.rs        engine crate
src/main.rs       CLI + serve dispatch
src/index.rs      writer: hot lanes, commit, log-tier merge, snapshots
src/query.rs      query AST + recursive-descent parser
src/search/       scoring: DocScores merges, phrase windows, block-max WAND, exec tree
src/segment.rs    segment freeze (FSTs per field + doc store: v2 blocks with id/metadata)
src/posting.rs    postings: interleave [delta][tf], skip anchors, galloping seeks
src/analyzer.rs   charabia + stemmer, one analyzer pipeline shared by docs and queries
src/server.rs     tiny_http serve mode
tests/            property + integration suite (57 tests)
benches/search.rs criterion scale + doc-length matrix
examples/big.rs   1M-doc probe   example: wand.rs: skew probe   example: smoke.rs: quick timings
```

## Distributed mode

Three+ shards can form a **token ring** (every member owns 1024 virtual
tokens; a key lands on the first token at or above its hash). Search fans out
to every shard and merges top-k (scores stay comparable because each shard
gossips its BM25 stats deltas at commit); ingest ring-routes each doc by hash
of its external id (unid'd docs: content hash), and `GET/DELETE /doc/{id}`
route to the owning shard. Ids stay unique + replaceable. Every doc lives on
the slot owner **plus one ring successor** (`REPLICA_FACTOR = 2`), so one
shard may die without losing the doc.

```sh
zap serve --bind 127.0.0.1:3279 --announce 127.0.0.1:3279 --peers addr1:3279,addr2:3279
```

Membership: a booting node announces itself to its configured peers (join
merge, idempotent) and boots `--resync` to survey its holder share: it pulls
the rows it owns from the peers' paginated handoff streams and tells the
sources to drop the copies they no longer hold — a scale-out **moves** the
documents, it does not copy them. Adding a member re-homes ~1/(N+1) of the
keyspace (the joiner's share), not the whole corpus. A delete is tombstoned
on every member, so a stale copy can never resurrect it.

Ring repartition of pre-existing nodes: `POST /__zap/peers {"peers":"a,b"}`
per node (or `{"join":"ip:port"}` to widen one member). `GET
/__zap/handoff?after=<seg>:<docid>` streams a shard's live docs for
movement/recovery; `POST /__zap/resync` re-runs the survey by hand. Shards
can persist state with `--disk DIR` (segments + MANIFEST.json; uncommitted
hot lanes are lost on a crash). A lost disk-backed shard reboots onto its
PV; a lost RAM-only shard loses its copy unless another node holds its
range.

Any node acts as the query coordinator (`/search` fan-out); a dead peer shows
up as `"degraded": true`. Cluster-internal routes live under `/__zap/…`
(gossip, shard_search, shard_ingest, shard_stats, peers, handoff, resync).
Helm: `cluster.enabled=true` plus a **headless** service (`zap-headless`)
wires the ring automatically; `cluster.disk.enabled=true` renders a
StatefulSet with a PVC per pod (`--disk`), so pod restarts keep their data.

## Packaging

```sh
make build              # release binary
make docker-build       # rust:alpine musl static -> FROM scratch (non-root 65534)
make docker-run         # serve mode on :3279
make helm-lint          # helm lint + template the chart
helm install zap charts/zap                 # empty index, populate via POST /ingest
kubectl create configmap corpus --from-file=docs/ -o yaml --dry-run=client | kubectl apply -f -
helm install zap charts/zap --set corpus.configMap=corpus   # bootstrap from static text
```

The index lives in RAM only: chart defaults to `Recreate` strategy (a rolling
update would serve from an empty pod), `replicaCount > 1` runs independent
indexes. Size memory resources to your corpus.

## HTTP serve mode

```sh
zap serve --bind 127.0.0.1:3279 dir/            # load corpus, then serve
zap serve --bind 127.0.0.1:3279 --stdin-lines   # empty index, ingest via HTTP
```

| endpoint | method | meaning |
|---|---|---|
| `/health` | GET | liveness |
| `/stats` | GET | live docs / segments / generation + per-shard chips |
| `/search?q=..&k=N` | GET | query string search (percent-encoded) |
| `/search` | POST | `{"q":"..","k":N}` body search |
| `/search` | POST | `{"vector":[..],"k":N}` embedding search (`mode`, `field`, `alpha`, `fusion` optional) |
| `/dashboard` | GET | live stats page |
| `/ingest` | POST | NDJSON `{"text":..}` lines or a JSON array of them (auto-commit) |
| `/commit` | POST | explicit segment commit |
| `/doc/{id}` | GET | stored fields + metadata for one external id |
| `/doc/{id}` | DELETE | tombstone one doc (visible immediately, purged at merge) |

Ingest docs accept optional `"id"` (string, unique — duplicates replace) and
`"metadata"` (arbitrary JSON, stored only, never tokenized).

Search result shape: `{"results":[{"score":0.71,"id":"my-doc","docid":42,"doc":"text",
"metadata":{...}}],"count":N}` — `"id"` is the external id (`null` when absent),
`"docid"` is the internal segment-local address, `"metadata"` only when present.
Ingest accepts a 64 MB body per call; docs commit per request. Sockets are plain
threads (no async runtime) — the engine is CPU-bound sync.

## Benchmarks (release, criterion)

Query latency (top-10, 10k doc corpus, merged single segment, block-max WAND):
- single hot term `fox`: **48 µs** (was 131 µs at the first working version)
- `fox AND dog`: 94 µs
- 3-term OR: 113 µs
- `"quick brown fox"` phrase: 290 µs
- wildcard `f*zz*`: 56 µs (was 1000 µs — FST range walk win)
- fuzzy `fox~1`: ~150 µs
- exact rare term: 0.65 µs

Scale (term / phrase / AND at 100k docs): 1.3 ms / 4.5 ms / 1.3 ms.
1M docs (`examples/big.rs`): ingest ≈ 16 s (8 lanes), rare-term top-10 = **0.36 ms**,
skew probe (`examples/wand.rs`): 90 ns @200k docs (outlier blocks only).

Doc-length sweep (1k docs/photo): rare-term query flat ~5 µs regardless of doc size
(100 w → 20 kw / 100k chars); phrase scales with doc length (23 µs @100w → 2.2 ms @20kw).
Ingest: 10k × 200w in 320 ms; 1M × 50w (8 lanes) in ~16 s.

Embedding search (flat scan, 768 dims, cosine, 8 lanes):
- 10k docs: **0.50 ms** vector-only, 0.57 ms hybrid
- 100k docs: **8.3 ms** vector-only, 8.9 ms hybrid

## Correctness

57-test suite: `cargo test`
- WAND-vs-dense evaluator equivalence (proptest, skewed corpora)
- skip-anchor invariants (seek = first doc ≥ t; positions source-identical)
- merge-heavy lane builds match single-lane reference (tie-tolerant compare)
- long-doc (100k chars) roundtrip; degenerate-input edge cases
- docstore decode fuzz (never panics on garbage; v2 blocks roundtrip raw + lz4)
- ids: roundtrip, replace across commit boundaries, delete masking (WAND/dense/
  fuzzy/prefix paths), tombstone purge at merge, duplicate batches, HTTP lifecycle
- concurrent commit + reader stress (incl. replace/delete storm); top-k outlier
  canary; CLI e2e over real sockets

Benchmarks: `cargo bench --bench search -- --noplot <filter>`.
Quick iteration: `cargo run --release --example smoke` (~1 s wall, coarse timings).

## Roadmap

- Embedding search: the flat SIMD scan + hybrid fusion shipped — next round is HNSW
  behind the `VectorIndex` trait seam and a quantized (f16) column as the memory lever
- Per-field bitpacked postings (128-doc blocks à la tantivy) if profiling shows decode-bound
