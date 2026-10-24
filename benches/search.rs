//! criterion benchmarks: ingest + query throughput

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

/// ZAP_BENCH_QUICK=1 shaves every group to a ~1s loop: times still rank
/// reliably (min-of-20 on hot paths), 5-8x faster wall per criterion pass.
/// Real numbers for perf: commits still use the default run.
fn cfg_quick(g: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>) {
    if std::env::var("ZAP_BENCH_QUICK").is_ok() {
        g.warm_up_time(Duration::from_millis(200))
            .measurement_time(Duration::from_millis(400))
            .sample_size(10);
    }
}

use std::sync::Arc;
use zap::analyzer::{Analyzer, AnalyzerConf};
use zap::index::Index;
use zap::schema::{Document, Metric, Schema};
use zap::search::SearchRequest;

/// synthetic english-ish corpus: `lorem`-ish words with zipf skew
fn corpus(n_docs: usize, words_per_doc: usize) -> Vec<String> {
    const WORDS: [&str; 116] = [
        "fox", "quick", "brown", "jumps", "lazy", "dog", "forest", "river", "mountain", "stone",
        "winter", "summer", "spring", "autumn", "night", "day", "sun", "moon", "star", "cloud",
        "wind", "rain", "snow", "storm", "field", "path", "road", "bridge", "house", "garden",
        "tree", "leaf", "branch", "root", "flower", "grass", "stone", "sand", "wave", "ocean",
        "fish", "bird", "nest", "feather", "wing", "sky", "dawn", "dusk", "shadow", "light",
        "fire", "water", "earth", "air", "metal", "wood", "glass", "paper", "ink", "word",
        "sentence", "story", "song", "dance", "music", "artist", "player", "game", "match", "team",
        "goal", "score", "point", "line", "curve", "angle", "circle", "square", "box", "code",
        "rust", "engine", "search", "index", "token", "query", "term", "field", "doc", "rank",
        "score", "boost", "merge", "segment", "posting", "bitmap", "stem", "stop", "fuzzy",
        "prefix", "phrase", "window", "position", "length", "norm", "field", "body", "title",
        "text", "plain", "fast", "slow", "deep", "wide", "open", "close",
    ];
    let mut out = Vec::with_capacity(n_docs);
    let mut state: u64 = 0x2545F4914F6CDD1D;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for i in 0..n_docs {
        let mut s = String::with_capacity(words_per_doc * 7);
        // zipf-ish: some terms repeat within docs and across docs
        for w in 0..words_per_doc {
            let pick = rnd();
            let word_i = if w % 16 == 0 {
                (pick % 8) as usize // hot terms
            } else {
                (pick % WORDS.len() as u64) as usize
            };
            s.push_str(WORDS[word_i]);
            s.push(' ');
        }
        s.push_str(&format!(" doc{i}"));
        out.push(s);
    }
    out
}

fn ingest_lanes(c: &mut Criterion) {
    let mut g = c.benchmark_group("ingest");
    g.warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .sample_size(10);
    cfg_quick(&mut g);
    // quick mode: 1000 docs only — the 10k tier rebuilds a full index per iteration
    let quick = std::env::var("ZAP_BENCH_QUICK").is_ok();
    let tiers: &[usize] = if quick { &[1000] } else { &[1000, 10000] };
    for n in tiers.iter().copied() {
        let docs = corpus(n, 200);
        g.throughput(Throughput::Bytes(
            docs.iter().map(|s| s.len() as u64).sum::<u64>(),
        ));
        g.bench_with_input(BenchmarkId::from_parameter(n), &docs, |b, docs| {
            b.iter(|| {
                let mut schema = Schema::default();
                schema.add_text("text", 1.0);
                let analyzer = Analyzer::new(AnalyzerConf::default());
                let index = Index::with_lanes(schema, analyzer, 1.2, 0.75, 8);
                let owned: Vec<Document> = docs
                    .iter()
                    .map(|d| Document::new().text("text", d.clone()))
                    .collect();
                index.add_documents(owned).unwrap();
                index.commit();
            });
        });
    }
    g.finish();
}

fn build_multi_segment(n: usize, commits: usize, words: Option<usize>) -> Arc<Index> {
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    let analyzer = Analyzer::new(AnalyzerConf::default());
    let lanes = 8;
    let index = Index::with_lanes(schema, analyzer, 1.2, 0.75, lanes);
    // doc-length floors: 50w keeps the 1M tier's memory budget; default 200w
    let words = words.unwrap_or(if n >= 1_000_000 { 50 } else { 200 });
    let docs = corpus(n, words);
    let per = (n / commits.max(1)).max(1);
    let mut chunk: Vec<Document> = Vec::with_capacity(per);
    for d in docs {
        chunk.push(Document::new().text("text", d));
        if chunk.len() == per {
            index.add_documents(std::mem::take(&mut chunk)).unwrap();
            index.commit();
        }
    }
    if !chunk.is_empty() {
        index.add_documents(chunk).unwrap();
    }
    index.commit();
    index
}

fn queries(c: &mut Criterion) {
    let index = build_multi_segment(10000, 10, None);
    let single = build_multi_segment(10000, 1, None);
    for (name, ix) in [("q1seg", single), ("q10seg", index)] {
        let mut g = c.benchmark_group("queries");
        g.warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(2))
            .sample_size(10);
        cfg_quick(&mut g);
        g.throughput(Throughput::Elements(10_000));
        for q in [
            "fox",
            "fox AND dog",
            "fox OR dog OR tree",
            "\"quick brown fox\"",
            "f*zz*",
            "fox~1",
            "doc99",
        ] {
            g.bench_with_input(BenchmarkId::new(name, q), q, |b, q| {
                b.iter(|| ix.search(q, 10).unwrap())
            });
        }
        g.finish();
    }
}

fn topk_collect(c: &mut Criterion) {
    let index = build_multi_segment(10000, 10, None);
    {
        let mut g = c.benchmark_group("topk");
        g.warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(2))
            .sample_size(10);
        cfg_quick(&mut g);
        g.bench_function("topk_union_broad", |b| b.iter(|| index.search("shot", 10)));
    }
}

fn query_scale(c: &mut Criterion) {
    // relevance query latency as the corpus grows: 100 / 1k / 10k / 100k docs
    let mut g = c.benchmark_group("query_scale");
    g.warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(10);
    cfg_quick(&mut g);
    // quick mode drops the 100k tier: the corpus BUILD (not the sampling) has
    // dominated this group's wall; quick runs target the 10k tier only.
    // 1M tier runs via examples/big.rs (criterion OOM-safe), see bench logs
    let quick = std::env::var("ZAP_BENCH_QUICK").is_ok();
    let tiers: &[(usize, usize)] = if quick {
        &[(100, 100), (10000, 1000)]
    } else {
        &[(100, 100), (1000, 250), (10000, 1000), (100000, 5000)]
    };
    for &(n, per_commit) in tiers {
        let ix = build_multi_segment(n, per_commit.max(1), None);
        for (tag, q) in [
            ("term", "fox"),
            ("phrase", "\"winter storm\""),
            ("and2", "fox AND dog"),
        ] {
            g.bench_with_input(BenchmarkId::new(format!("{n}docs"), tag), tag, |b, _| {
                b.iter(|| ix.search(q, 10).unwrap())
            });
        }
    }
    g.finish();
}

fn doc_length(c: &mut Criterion) {
    // short vs long documents at a fixed corpus: ingest throughput + query time.
    // tiers: ~600B, ~1KB, ~6KB, ~30KB, ~120KB, ~1MB per doc (big tiers use fewer
    // docs to keep per-iteration cost bounded)
    let mut g = c.benchmark_group("doc_length");
    g.warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(10);
    cfg_quick(&mut g);
    // big tiers take the wall; quick mode keeps the 100w..5w brackets
    let quick = std::env::var("ZAP_BENCH_QUICK").is_ok();
    let tiers = if quick {
        vec![100usize, 170, 1000, 5000]
    } else {
        vec![100usize, 170, 1000, 5000, 20000, 160000]
    };
    for wl in tiers {
        let n_docs = if wl >= 50000 { 16 } else { 1000 };
        let docs = corpus(n_docs, wl);
        g.throughput(Throughput::Bytes(
            docs.iter().map(|s| s.len() as u64).sum::<u64>(),
        ));
        g.bench_with_input(
            BenchmarkId::new("ingest", format!("{wl}w")),
            &docs,
            |b, docs| {
                b.iter(|| {
                    let mut schema = Schema::default();
                    schema.add_text("text", 1.0);
                    let analyzer = Analyzer::new(AnalyzerConf::default());
                    let index = Index::with_lanes(schema, analyzer, 1.2, 0.75, 8);
                    let owned: Vec<Document> = docs
                        .iter()
                        .map(|d| Document::new().text("text", d.clone()))
                        .collect();
                    let per = owned.len().div_ceil(10);
                    for ch in owned.chunks(per) {
                        index.add_documents(ch.to_vec()).unwrap();
                        index.commit();
                    }
                });
            },
        );
        let ix = build_multi_segment(n_docs, 10, Some(wl));
        for (tag, q) in [
            ("hot_term", "fox"),
            ("phrase", "\"winter storm\""),
            ("rare_term", "quixotic"),
        ] {
            g.bench_with_input(
                BenchmarkId::new(format!("query_{wl}w"), tag),
                tag,
                |b, _| b.iter(|| ix.search(q, 10).unwrap()),
            );
        }
    }
    g.finish();
}

/// embedding search: the flat scan cost is O(rows x dim) — this is the
/// number that decides when the HNSW round behind the `VectorIndex` seam
/// becomes mandatory.
fn embedding_scan(c: &mut Criterion) {
    let dim = 768usize;
    // deterministic pseudo-random embeddings
    let mut state = 0x2545F491_4F6CDD1Du64;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut build = |n: usize| -> (Arc<Index>, Vec<f32>) {
        let mut schema = Schema::default();
        schema.add_text("text", 1.0);
        schema.add_vector("emb", dim as u32, Metric::Cosine);
        let analyzer = Analyzer::new(AnalyzerConf::default());
        let index = Index::with_lanes(schema, analyzer, 1.2, 0.75, 8);
        let mut chunk: Vec<Document> = Vec::with_capacity(n);
        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|_| ((rnd() % 2000) as f32 / 2000.0) - 0.5)
                .collect();
            chunk.push(
                Document::new()
                    .id(format!("d{i}"))
                    .text("text", "embedding corpus")
                    .vector("emb", v),
            );
        }
        index.add_documents(chunk).unwrap();
        index.commit();
        let query: Vec<f32> = (0..dim)
            .map(|_| ((rnd() % 2000) as f32 / 2000.0) - 0.5)
            .collect();
        (index, query)
    };

    let quick = std::env::var("ZAP_BENCH_QUICK").is_ok();
    let tiers: Vec<usize> = if quick {
        vec![10_000]
    } else {
        vec![10_000, 100_000]
    };
    for n in tiers {
        let (index, query) = build(n);
        let mut g = c.benchmark_group("vector");
        cfg_quick(&mut g);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(
            BenchmarkId::new("scan", format!("{n}x{dim}")),
            &query,
            |b, q| {
                b.iter(|| {
                    index
                        .search_request(&SearchRequest::vector(q.clone(), 10), index.pub_stats())
                        .unwrap()
                })
            },
        );
        // the hybrid path: the keyword side plus the embedding side, fused
        g.bench_with_input(
            BenchmarkId::new("hybrid", format!("{n}x{dim}")),
            &query,
            |b, q| {
                b.iter(|| {
                    index
                        .search_request(
                            &SearchRequest::hybrid("fox", q.clone(), 10),
                            index.pub_stats(),
                        )
                        .unwrap()
                })
            },
        );
        g.finish();
    }
}

criterion_group!(
    benches,
    ingest_lanes,
    queries,
    topk_collect,
    query_scale,
    doc_length,
    embedding_scan
);
criterion_main!(benches);
