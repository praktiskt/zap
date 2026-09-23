# AGENTS.md — zap

In-memory full-text search engine. Performance is the product: search-first, ingest second.
Single binary, everything lives in the process; no mmap, no persistence, no server dependency
(HTTP mode serves straight out of RAM over tiny_http threads).

## What zap is
- Rust FTS: BM25 (+ phrase window bonus, field boosts), boolean/phrase/slop/prefix/wildcard/fuzzy queries
- Segment architecture: immutable sealed segments (FST term dict, delta-varint postings, skip anchors) + a parallel hot-ingest lane builder; log-tier merge
- block-max WAND top-k with live thresholds; galloping seeks power both phase cursors and block pruning
- arccos-swap snapshots: single writer, lock-free readers
- HTTP serve subcommand (tiny_http): /ingest /search /stats /commit /health
- multilingual tokenization via charabia (+ snowball stemming on latin tokens)

Future extension is pre-plumbed: `schema.rs` declares `vector` fields (dim + metric) with
validation gates, so embedding search can be added without re-shaping the documents model.

## Repo layout
- `src/lib.rs` crate; `src/main.rs` CLI (interactive + `serve` subcommand)
- core modules: `analyzer/`, `index.rs` (writer/snapshots/merge), `query.rs` (AST+parser),
  `search/` (term/phrase/exec: scorers + WAND), `segment.rs`, `posting.rs`, `schema.rs`, `server.rs`
- `tests/`: property + integration tests; `benches/search.rs`: criterion matrix;
  `examples/`: scale probes (`big.rs` 1M-doc, `wand.rs` skew probe, `postprobe.rs` decode A/B, `smoke.rs` quick timings, `doclen_probe.rs` doc-length phases)
- packaging: `Dockerfile` (rust:alpine musl static -> FROM scratch, non-root), `Makefile`
  (build/test/bench/build-all/docker-*/helm-lint), `charts/zap/` (serve-mode Deployment
  Recreate strategy, corpus ConfigMap mount, /health probes)

## Perf-optimization history
- Every optimization attempt (successful or failed) gets its own git commit.
- Successful ones: commit messages start with `perf:` and MUST include the benchmark numbers used (before → after), the criterion filter runnable, and the date.
- Reverted/failed attempts: commit message starts with `perf-reject:` and states why + numbers.
- Check `git log --oneline --grep='perf'` before proposing new optimization: avoid repeating tried ideas.
- Benchmarks: `cargo bench --bench search -- --noplot <filter>`; bench log copies land in `/tmp/opencode/bench.log`.
- After each perf change: `cargo test`, `cargo fmt`, `cargo clippy --all-targets`, then benchmark with an identical filter as the previous commit.

## Non-perf rules
- always cargo fmt + clippy clean before commit
- integration tests in `tests/`; crate tests in `src/`
- test suite map: `properties.rs` (T1-T4: skip/WAND/merge equivalence props), `longdoc.rs` (T5), `edges.rs` (T6), `docstore_fuzz.rs` (T9), `concurrent.rs` (T7), `wand_canary.rs` (T8), `cli.rs` (T10), `ids.rs` (T11: external ids/metadata/replace/delete), `serve.rs` (HTTP incl. /doc/{id} lifecycle), `search_it.rs` (boolean/phrase integration)
- benchmarking: use `examples/smoke.rs` (`cargo run --release --example smoke`, ~1 s) for iteration feedback; run rigorous criterion benches (`cargo bench --bench search -- --noplot <filter>`) sparsely — at meaningful checkpoints only, not per change
- do not use ad-hoc sed/python for edits unless trivial; prefer dedicated tools when editing files
- top-k tie ordering across different index builds is NOT stable (heap tie-break by docid/addr): property tests must compare as unordered (score,text) multisets, or filter entries strictly above the kth-score threshold

## Known perf sensitive spots (from history)
- prefix/fuzzy: FST range/automaton paths (NOT dict scans — `stream_terms` full scans are perf-reject)
- AND eval: streaming gallop-intersection with the smallest-df term as driver (R12/R15); bitmap-first double-scan is perf-reject (R3)
- phrase cursors: must use galloping `seek_to`, never linear re-decode; hoist position buffers out of per-doc loops (R6); min_span early-exits at minimal span n-1 (R17, 2x on huge docs); phrase eval is DUAL-PATH (R18): small-df = fused single-pass gallop merge + inline verify (do NOT split — two-pass cost +24-77% on small corpora), large-df (min_df >= 65536) = docid_vec decode + sorted-vec intersect + rayon-parallel per-doc verify (1M-doc phrase 43.3->24.6ms); roaring bitmap AND tried for intersection: no win (decode-bound, not seek-bound) — do not re-add; root-level phrase gets live-threshold segment skip, nested phrases must evaluate unconditionally (score-sum safety)
- ingestion cost lands in `analyzer.tokenize` (charabia); parallelize per document lanes when possible
- group-varint postings decode: perf-reject (R4, +57% term scan; branchy GV loses to plain varint at tested sizes 10k docs) — retry only with true fixed-width SIMD blocks
- fixed-width bitpacked blocks (R19, perf-reject): 128-doc byte-aligned blocks, per-block widths, combined (delta<<w_tf|tf) 32-bit fields + skip-anchor reuse — scalar bit-unpack LOSES to byte-varint: advance 8.1-8.4 vs 7.6-7.9 ns/doc, docid_vec −18% dense-only, 1M-doc term query +70% end-to-end; varint is already near the decode floor (7.6ns, WAND skips blocks); remaining path = per-width AVX2 unpack kernels (pshufb), ceiling small on the dominant term path — do NOT retry without real SIMD kernels; examples/postprobe.rs isolates decode costs (works for any future format A/B via git stash)
- block-max WAND: implemented (streaming single-Term path + live threshold); skip anchors (16B: prev_docid/off/tf_max/len_min) power both galloping seek and block pruning; length tables are NOT term-local (field-level)
- inline `[delta][tf]` posting layout (R16): single cursor; do NOT split docid/tf streams again
- external ids: snapshot `ids` map + per-segment `id_fwd` rebuilt ONLY at segment change; tombstone-only publishes (delete) are O(1) via cached Arc'd df stats + id map, readers mask via `deleted`; deleting from hot lanes commits first — do NOT add hot-lane deletion machinery
- doc store v2: [tag][len][flags][n][TLV fields][id][metadata]; merge copies blocks verbatim (ids/metadata survive free); purge at merge uses per-segment docid remap (u32::MAX = purged)

## Future rounds, if we resume WAND/general top-k work
- WAND fan-out into AND/OR walkers (R20, perf-reject): classic pivot WAND-OR (static-max bound, docid-gap seeking, live per-segment collector threshold via direct pushes) + AND driver block skipping both implemented and LOSE: uniform 10k-corpus OR +8% (139.0 vs 128.71us), skew 200k 'fox OR dog' +23% (4.68 vs 3.81ms, env-gated same-binary A/B); per-iteration pivot refresh (docs+order+sort) outweighs threshold pruning because dense union is already efficient and uniform score distributions keep thresholds low; dense AND gained nothing either (102.6 vs 102.1us) — reattempt only with block-max-refined pivot bounds AND push-free cursor granularity; reverted to dense-union OR + return-based galloping AND
- per-field postings bitpack (128-doc blocks à la tantivy) if profiling shows decode-bound
- embedding search: flat SIMD scan first; HNSW (hnsw_rs / instant-distance) behind the `VectorIndex` trait seam
