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
- distributed mode (cluster.rs): consistent-hash slot ring (SLOTS=1024, owner = ring[fnv64a(id)%SLOTS % N] — stepped layout: membership change re-homes exactly SLOTS/N slots, unit-tested for churn + skew; ring list = [self+peers...] sorted by addr), coordinator-less scatter-gather (any pod fans out /search via /__zap/shard_search), BM25-stats gossip (per-peer ABSOLUTE df envelopes pushed as deltas at commit; first contact = full envelope + resync pull), ring-routed ingest and /doc/{id}; empty --peers = zero-overhead single-node path (engine untouched; scores need the global view via Index::search_with_stats)
- shard movement / recovery (handoff): GET /__zap/handoff = paginated live-doc stream of a shard (Index::docs_page cursor "seg:docid", tombstoned skipped); pull_own_range() pages every peer's stream, re-ingests only rows the puller OWNS in its ring view (id key, unid = content-hash key), and ack-deletes the id-carrying ones on the source via POST /__zap/handoff_delete; triggers: boot --resync (only when the local index is EMPTY) or POST /__zap/resync; POST /__zap/peers replaces a node's peer list IN PLACE (ring repartition — every node must apply the same membership or views diverge), {"join":"ip:port"} = idempotent membership merge and every joining node auto-announces to its configured peers at boot (800ms after serve, then the pull at 1.8s) — move procedure = singleton ingest -> joiner --resync (announce widens both views automatically) -> /doc/{id} routes ring-correct everywhere; RAM-only shards CANNOT rehydrate lost ranges (nothing holds a copy) — this moves docs, it does not replicate them; scripts/handoff-probe.sh = live manual probe, tests/handoff.rs = integration; chart: cluster.disk.enabled renders a StatefulSet (stable ordinals, serviceName peersService, PVC zap-data mounted -disk /zap-data) so rebooted pods reload their own state
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
- test suite map: `properties.rs` (T1-T4: skip/WAND/merge equivalence props), `longdoc.rs` (T5), `edges.rs` (T6), `docstore_fuzz.rs` (T9), `concurrent.rs` (T7), `wand_canary.rs` (T8), `cli.rs` (T10), `ids.rs` (T11: external ids/metadata/replace/delete), `serve.rs` (HTTP incl. /doc/{id} lifecycle), `cluster.rs` (3-process ring roundtrip), `handoff.rs` (singleton -> --resync joiner ring-share move + /__zap/peers repartition), `search_it.rs` (boolean/phrase integration)
- benchmarking: use `examples/smoke.rs` (`cargo run --release --example smoke`, ~1 s) for iteration feedback; run rigorous criterion benches (`cargo bench --bench search -- --noplot <filter>`) sparsely — at meaningful checkpoints only, not per change
- do not use ad-hoc sed/python for edits unless trivial; prefer dedicated tools when editing files
- top-k tie ordering across different index builds is NOT stable (heap tie-break by docid/addr): property tests must compare as unordered (score,text) multisets, or filter entries strictly above the kth-score threshold

## Known perf sensitive spots (from history)
- prefix/fuzzy: FST range/automaton paths (NOT dict scans — `stream_terms` full scans are perf-reject)
- AND eval: streaming gallop-intersection with the smallest-df term as driver (R12/R15); bitmap-first double-scan is perf-reject (R3)
- phrase cursors: must use galloping `seek_to`, never linear re-decode; hoist position buffers out of per-doc loops (R6); min_span early-exits at minimal span n-1 (R17, 2x on huge docs); phrase eval is DUAL-PATH (R18): small-df = fused single-pass gallop merge + inline verify (do NOT split — two-pass cost +24-77% on small corpora), large-df (min_df >= 65536) = docid_vec decode + sorted-vec intersect + rayon-parallel per-doc verify (1M-doc phrase 43.3->24.6ms); roaring bitmap AND tried for intersection: no win (decode-bound, not seek-bound) — do not re-add; root-level phrase gets live-threshold segment skip, nested phrases must evaluate unconditionally (score-sum safety)
- ingest cost lands in `analyzer.tokenize` (charabia); parallelize per document lanes when possible
- merge policy (R21): trigger raised MAX_SEGMENTS 4->8 (bench corpora 10 commits x 8 lanes = 80 builder segments keep their shape: trigger 64 -> one merge -> 17 final segments at 8 lanes, identical to the old 4-budget); merge internals: segment-range parallel posting union (single-field corpora used to merge on ONE thread), get-mut-first dict probes, reused position buffer + mem::take; tiered/oldest-first merge fan-in ATTEMPTED and perf-rejected (min-fan math sabotaged steady state: 25.0s vs 16.4s) — proper tiering needs explicit tier lists in IndexInner, do not hack it with a trigger-side fan-in min; 1M-doc ingest 18.2 -> 13.0s (min-of-3, examples/big.rs)
- merge parallelization round 2 (R22): freeze_parts parallel per-term encode (per-chunk sub-blobs, fold rebases section ranges, FST inserts stay sequential) + parallel per-segment scan of norms/store/ids (order-preserving collect fold); full-corpus merge instrumentation: freeze_parts 714->561ms, scan_store 189->127ms; 1M-doc ingest 13.0 -> 11.2s within-window (commit phase 4.3 -> 3.9s, -11%); remaining merge walls: postings_union ~1.0s (decode+re-encode floor per R19)
- big-doc phrase parallelism (R23): work-aware routing replaces the df-only gate — 1000x1M-char phrase was single-threaded (per-seg df 125 << 65536) at ~2360us; VEC gate now keys on positions-blob bytes (32KB floor) alongside df, pipe to the two-pass rayon-parallel verify with chunk=8; min_span gains a downstream-list-exhaustion break (failing docs stop scanning). Within-window: phrase 2360 -> 1790us (-24%); 1M-doc x 50w corpus: phrase 24.6 -> 9.9ms (-60%); query matrix flat in drift bands (small q1seg phrase ~+8%, mixed with ambient noise)
- group-varint postings decode: perf-reject (R4, +57% term scan; branchy GV loses to plain varint at tested sizes 10k docs) — retry only with true fixed-width SIMD blocks
- fixed-width bitpacked blocks (R19, perf-reject): 128-doc byte-aligned blocks, per-block widths, combined (delta<<w_tf|tf) 32-bit fields + skip-anchor reuse — scalar bit-unpack LOSES to byte-varint: advance 8.1-8.4 vs 7.6-7.9 ns/doc, docid_vec −18% dense-only, 1M-doc term query +70% end-to-end; varint is already near the decode floor (7.6ns, WAND skips blocks); remaining path = per-width AVX2 unpack kernels (pshufb), ceiling small on the dominant term path — do NOT retry without real SIMD kernels; examples/postprobe.rs isolates decode costs (works for any future format A/B via git stash)
- block-max WAND: implemented (streaming single-Term path + live threshold); skip anchors (16B: prev_docid/off/tf_max/len_min) power both galloping seek and block pruning; length tables are NOT term-local (field-level); same-block seek fast path in PostingReader::seek_to (skip the anchor reset when target is inside the cursor's own block) ATTEMPTED and perf-rejected (R25): strictly-gated row-pair boundary test — 10k term 120->131us +9%, phrase 562->588us +5%, and2 320->305us -4%, q10seg and2 flat; skip never fires on dense-uniform distributions (bound==thr-band) so the two extra skip-row reads per seek pay for nothing; PER-SEEK anchor-bounce cost is bounded by the block layout (<=128 docs decode), and gallop callers are monotone so amortized re-decode is already short — revisit only if the block layout changes again (R19/R24 rejects remain the floor)
- OR eval interval: dense-union is NOT the last wall — segment-count parallelism hides a skew problem (R26): merged target segments are often 300-600k docs, so the per_segment rayon split leaves ONE task at ~13ms (dr+seek 21ns/doc) while the other 32 finish in us — the biggest segment owns the wall; and_two_infos (dense AND) now splits internally at min(df) >= 65536 into docid_vec pass + thread::scope chunks gallop with its own reader pair per chunk (identical scoring, chunk-local Ctx with idle threshold, shared-nothing monotone ranges); results concat in ascending docid (ties preserved). 1M and2 13.0-> 4.65ms; query matrix flat (10k utf8 tier min(df) < 65536 keeps the fused gallop); phrase already had the same treatment via R18/R23 VEC two-pass
- inline `[delta][tf]` posting layout (R16): single cursor; do NOT split docid/tf streams again
- external ids: snapshot `ids` map (persistent im::HashMap behind Arc) + per-segment `id_fwd` rebuilt incrementally at segment change: commit extends the map via O(1) clone + O(log n) update per appended doc (`segs.extend` appends so old addresses stay valid), merge/boot reset to a full O(live) rescan; delete/replace rewrite the shared map in place (O(log n)); deletes keep this O(1)-per-publish via cached Arc'd df stats; deleting from hot lanes commits first — do NOT add hot-lane deletion machinery
- doc store v2: [tag][len][flags][n][TLV fields][id][metadata]; merge copies blocks verbatim (ids/metadata survive free); purge at merge uses per-segment docid remap (u32::MAX = purged)
- disk persistence (opt-in --disk DIR): SegmentReader::to_bytes/from_bytes (binary: magic zapseg1, FST raw bytes via the fst_raw shadow field populated at freeze — fst::Map cannot expose them post-construction), content-addressed seg-<fnv64>.zap files, atomically-rewritten MANIFEST.json (tmp+fsync+rename) incl. tombstone bitmaps; boot = with_lanes_disk loads synchronously; hooks on commit_locked + index.delete; UNCOMMITTED hot lanes are lost on crash (bounded by commit-every, HTTP /ingest auto-commits per request)

## Future rounds, if we resume WAND/general top-k work
- WAND fan-out into AND/OR walkers (R20, perf-reject): classic pivot WAND-OR (static-max bound, docid-gap seeking, live per-segment collector threshold via direct pushes) + AND driver block skipping both implemented and LOSE: uniform 10k-corpus OR +8% (139.0 vs 128.71us), skew 200k 'fox OR dog' +23% (4.68 vs 3.81ms, env-gated same-binary A/B); per-iteration pivot refresh (docs+order+sort) outweighs threshold pruning because dense union is already efficient and uniform score distributions keep thresholds low; dense AND gained nothing either (102.6 vs 102.1us) — reattempt only with block-max-refined pivot bounds AND push-free cursor granularity; reverted to dense-union OR + return-based galloping AND
- per-field postings bitpack (128-doc blocks à la tantivy) if profiling shows decode-bound
- embedding search: flat SIMD scan first; HNSW (hnsw_rs / instant-distance) behind the `VectorIndex` trait seam
