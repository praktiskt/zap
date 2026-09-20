# zap — in-memory full-text search engine

Single-binary Rust FTS: BM25 ranking, boolean/phrase/prefix/wildcard/fuzzy queries,
multilingual tokenization (charabia), segmented index with log-tier merge, snapshot
concurrency (arc-swap), lambda4 doc store for snippets + highlighting. Nothing leaves the
process: all index data lives in memory.

## Query syntax

| input | meaning |
|---|---|
| `fox dog` | implicit AND of both terms |
| `a AND b` / `a OR b` | boolean operators (uppercase) |
| `a NOT b` | exclusion matching (subtract docs) |
| `"quick brown fox"` | exact phrase (consecutive positions) |
| `"fast the"~1` | phrase with slop (1 token gaps allowed) |
| `field:term` | restrict to a field; `field:(a OR b)` scopes a subtree |
| `prefix*` | prefix expansion from term dict |
| `wild*car?` | expr wildcard (`*` any run, `?` one char) |
| `typo~1` | fuzzy match (Levenshtein/Damerau DFA over term FST, up to distance 2) |
| `( a OR b ) AND NOT c` | grouping, operators |
| `:stats` / `:help` / `:q` | meta commands in the CLI query loop |

## Usage

```sh
# index files/dirs, search interactively (every stdin line = one query)
zap dir/ file.md
# whole stdin as ONE doc; per-line documents with a blank-line terminator
echo anthology | zap --stdin-lines
```

Docs: files/dirs (`.txt .md .rst .json .toml .csv .log .xml`, unknown extensions tried as text).
Query loop command set: terms, flags `-k` top-k, `--k1/--b` BM25 params, `--no-stem`.

## Architecture

- `schema`: field declarative text fields (vectors declared as future extension hook)
- segments: immutable, sealed, FST term dict; hot-built segments + committing
- posting list: delta-varint docids/tfs/positions packed with per-doc offset table
- Evaluators: materialized `DocScores` (doc ids + scores arrays) merged by boolean vectors
-ór ranking: BM25 with Lucene-style idf, norm-provider fieldnorms (bytes/doc), field boosts
- Termdict: fst FST: exact/prefix/range via automaton, configure fst-levenshtein for fuzzy
- doc store: lz4 blocks per doc + metadata for snippet/highlight reconstruction (highlighting via re-tokenization)
- multi-lang: charabia detectors (Latin uax word segmentation + NFKC + CJK segmentation), stemmers

## Tests + benchmarks

- `tests/search_it.rs`: boolean/phrase/slop/prefix/fuzzy/field/merge/CJK (integration)
- unit tests: varint, posting roundtrip, tokenizer behavior, parser grammar
- `cargo bench` matrıx: scale (100/1k/10k/100k docs × term/phrase/AND), doc-length sweep
 (100/1k/5k words), ingest throughput (~8 MB/s, release)

## Roadmap

- Segment-level roaring bitmaps + block-parallel terms for faster scale brittleness
- Embedding search: vectors declared in schema + brute-force SIMD matrix scan → HNSW
- Docid-tier compression for postings (bitpacked blocks 128 like tantivy) + block-max WAND
