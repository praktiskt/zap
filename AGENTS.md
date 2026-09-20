# AGENTS.md — zap

In-memory full-text search engine. Performance is the product: search-first, ingest second.

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
- do not use ad-hoc sed/python for edits unless trivial; prefer dedicated tools when editing files

## Known perf sensitive spots (from history)
- prefix/fuzzy: FST range/automaton paths (NOT dict scans — `stream_terms` full scans are perf-reject)
- AND eval: smallest-df positive seeds + stream intersect; bitmap-first double-scan is perf-reject (R3)
- phrase cursors: must use galloping `seek_to`, never linear re-decode
- ingestion cost lands in `analyzer.tokenize` (charabia); parallelize per document lanes when possible
- group-varint postings decode: perf-reject (R4, +57% term scan; branchy GV loses to plain varint at tested sizes 10k docs) — retry only with true fixed-width SIMD blocks
- block-max WAND: implemented (streaming single-Term path + live threshold); skip anchors (16B: prev_docid/off/tf_max/len_min) power both galloping seek and block pruning; length tables are NOT term-local (field-level)
