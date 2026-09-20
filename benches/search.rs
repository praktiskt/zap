//! criterion benchmarks: ingest + query throughput

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

use std::sync::Arc;
use zap::analyzer::{Analyzer, AnalyzerConf};
use zap::index::Index;
use zap::schema::{Document, Schema};

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

fn ingest(c: &mut Criterion) {
    let mut g = c.benchmark_group("ingest");
    g.warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .sample_size(10);
    for n in [1000_usize, 10000] {
        let docs = corpus(n, 200);
        g.throughput(Throughput::Bytes(
            docs.iter().map(|s| s.len() as u64).sum::<u64>(),
        ));
        g.bench_with_input(BenchmarkId::from_parameter(n), &docs, |b, docs| {
            b.iter(|| {
                let mut schema = Schema::default();
                schema.add_text("text", 1.0);
                let analyzer = Analyzer::new(AnalyzerConf::default());
                let index = Index::new(schema, analyzer, 1.2, 0.75);
                for d in docs {
                    index
                        .add_document(&Document::new().text("text", d.clone()))
                        .unwrap();
                }
                index.commit();
            });
        });
    }
    g.finish();
}

fn build_multi_segment(n: usize, commits: usize) -> Arc<Index> {
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    let analyzer = Analyzer::new(AnalyzerConf::default());
    let index = Index::new(schema, analyzer, 1.2, 0.75);
    let docs = corpus(n, 200);
    let per = n / commits.max(1);
    for (i, d) in docs.iter().enumerate() {
        index
            .add_document(&Document::new().text("text", d.clone()))
            .unwrap();
        if i % per == per - 1 {
            index.commit();
        }
    }
    index.commit();
    index
}

fn queries(c: &mut Criterion) {
    let index = build_multi_segment(10000, 10);
    let single = build_multi_segment(10000, 1);
    for (name, ix) in [("q1seg", single), ("q10seg", index)] {
        let mut g = c.benchmark_group("queries");
        g.warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(2))
            .sample_size(10);
        g.throughput(Throughput::Elements(n_docs_const() as u64));
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

fn n_docs_const() -> usize {
    10
}

fn topk_collect(c: &mut Criterion) {
    let index = build_multi_segment(10000, 10);
    {
        let mut g = c.benchmark_group("topk");
        g.warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(2))
            .sample_size(10);
        g.bench_function("topk_union_broad", |b| b.iter(|| index.search("shot", 10)));
    }
}

fn query_scale(c: &mut Criterion) {
    // relevance query latency as the corpus grows: 100 / 1k / 10k / 100k docs
    let mut g = c.benchmark_group("query_scale");
    g.warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(10);
    for (n, per_commit) in [
        (100usize, 100usize),
        (1000, 250),
        (10000, 1000),
        (100000, 5000),
    ] {
        let ix = build_multi_segment(n, per_commit.max(1));
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
    // short vs long documents at a fixed corpus: ingest throughput + query time
    let mut g = c.benchmark_group("doc_length");
    g.warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(10);
    for wl in [100usize, 1000, 5000] {
        let docs = corpus(1000, wl);
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
                    let index = Index::new(schema, analyzer, 1.2, 0.75);
                    for d in docs {
                        index
                            .add_document(&Document::new().text("text", d.clone()))
                            .unwrap();
                    }
                    index.commit();
                });
            },
        );
        let ix = build_multi_segment_len(1000, 10, wl);
        for (tag, q) in [("hot_term", "fox"), ("phrase", "\"winter storm\"")] {
            g.bench_with_input(
                BenchmarkId::new(format!("query_{wl}w"), tag),
                tag,
                |b, _| b.iter(|| ix.search(q, 10).unwrap()),
            );
        }
    }
    g.finish();
}

fn build_multi_segment_len(n: usize, commits: usize, words: usize) -> Arc<Index> {
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    let analyzer = Analyzer::new(AnalyzerConf::default());
    let index = Index::new(schema, analyzer, 1.2, 0.75);
    let docs = corpus(n, words);
    let per = (n / commits.max(1)).max(1);
    for (i, d) in docs.iter().enumerate() {
        index
            .add_document(&Document::new().text("text", d.clone()))
            .unwrap();
        if i % per == per - 1 {
            index.commit();
        }
    }
    index.commit();
    index
}

criterion_group!(
    benches,
    ingest,
    queries,
    topk_collect,
    query_scale,
    doc_length
);
criterion_main!(benches);
