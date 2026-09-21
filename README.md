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
```

Input docs: files/dirs (`.txt .md .rst .json .toml .csv .log .xml`; unknown extensions tried
as text); `--stdin` = whole stdin is one doc; `--stdin-lines` = one doc per line until a
blank line. Common flags: `-k N`, `--lanes N`, `--no-stem`, `--plain`, `--snippet N`,
`--commit-every N`.

## Repo

```
src/lib.rs        engine crate
src/main.rs       CLI + serve dispatch
src/index.rs      writer: hot lanes, commit, log-tier merge, snapshots
src/query.rs      query AST + recursive-descent parser
src/search/       scoring: DocScores merges, phrase windows, block-max WAND, exec tree
src/segment.rs    segment freeze (FSTs per field + doc store)
src/posting.rs    postings: interleave [delta][tf], skip anchors, galloping seeks
src/analyzer.rs   charabia + stemmer, one analyzer pipeline shared by docs and queries
src/server.rs     tiny_http serve mode
tests/            property + integration suite (41 tests)
benches/search.rs criterion scale + doc-length matrix
examples/big.rs   1M-doc probe   example: wand.rs: skew probe
```

## HTTP serve mode

```sh
zap serve --bind 127.0.0.1:3279 dir/            # load corpus, then serve
zap serve --bind 127.0.0.1:3279 --stdin-lines   # empty index, ingest via HTTP
```

| endpoint | method | meaning |
|---|---|---|
| `/health` | GET | liveness |
| `/stats` | GET | docs / segments / generation |
| `/search?q=..&k=N` | GET | query string search (percent-encoded) |
| `/search` | POST | `{"q":"..","k":N}` body search |
| `/ingest` | POST | NDJSON `{"text":..}` lines or a JSON array of them (auto-commit) |
| `/commit` | POST | explicit segment commit |

Search result shape: `{"results":[{"score":0.71,"id":0,"doc":"text"}],"count":N}`.
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

## Correctness

41-test suite: `cargo test`
- WAND-vs-dense evaluator equivalence (proptest, skewed corpora)
- skip-anchor invariants (seek = first doc ≥ t; positions source-identical)
- merge-heavy lane builds match single-lane reference (tie-tolerant compare)
- long-doc (100k chars) roundtrip; degenerate-input edge cases
- docstore decode fuzz (never panics on garbage)
- concurrent commit + reader stress; top-k outlier canary; CLI e2e over real sockets

Benchmarks: `cargo bench --bench search -- --noplot <filter>`.

## Roadmap

- WAND fan-out into AND/OR chains (currently only single-Term streaming prunes)
- Embedding search: schema-gated vector fields + flat SIMD scan → HNSW via `VectorIndex` seam
- Per-field bitpacked postings (128-doc blocks à la tantivy) if profiling shows decode-bound
