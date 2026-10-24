//! smoke bench: quick relative timings for the perf-sensitive paths.
//! NOT a criterion replacement — coarse numbers in seconds, used during iteration.
//! Rigorous runs stay in `cargo bench --bench search -- --noplot <filter>`.
//!
//!   cargo run --release --example smoke

use std::sync::Arc;
use std::time::Instant;

use zap::analyzer::{Analyzer, AnalyzerConf};
use zap::index::Index;
use zap::schema::{Document, Metric, Schema};
use zap::search::SearchRequest;

const N: usize = 10_000;

fn corpus() -> Vec<String> {
    (0..N)
        .map(|i| {
            let w = i % 97;
            format!(
                "the quick brown fox jumps over the lazy dog number {i} \
                 forest {w} river {w} shot {w} whisper of birds {w} doc{w}"
            )
        })
        .collect()
}

fn build(lanes: usize) -> Arc<Index> {
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    Index::with_lanes(
        schema,
        Analyzer::new(AnalyzerConf::default()),
        1.2,
        0.75,
        lanes,
    )
}

fn bench_index(name: &str) -> Arc<Index> {
    let index = build(8);
    let docs = corpus();
    // ingest: chunked to force a few commits like the criterion bench
    let t = Instant::now();
    for chunk in docs.chunks(N / 10) {
        index
            .add_documents(
                chunk
                    .iter()
                    .map(|d| Document::new().text("text", d.clone()))
                    .collect(),
            )
            .unwrap();
        index.commit();
    }
    let ingest_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("{name:>10}  ingest {N:6} docs: {ingest_ms:8.1} ms");
    index
}

fn time_q<F: FnMut()>(name: &str, iters: usize, mut f: F) {
    // warmup
    for _ in 0..iters / 10 {
        f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("{name:>38}: {us:8.1} us");
}

fn main() {
    let t0 = Instant::now();
    println!("== single segment (merged) ==");
    let single = {
        let index = build(1);
        let docs = corpus();
        index
            .add_documents(
                docs.into_iter()
                    .map(|d| Document::new().text("text", d))
                    .collect(),
            )
            .unwrap();
        index.commit();
        index
    };
    time_q("fox", 500, || {
        let _ = single.search("fox", 10).unwrap();
    });
    time_q("fox AND dog", 500, || {
        let _ = single.search("fox AND dog", 10).unwrap();
    });
    time_q("fox OR dog OR tree", 300, || {
        let _ = single.search("fox OR dog OR tree", 10).unwrap();
    });
    time_q("\"quick brown fox\"", 200, || {
        let _ = single.search("\"quick brown fox\"", 10).unwrap();
    });
    time_q("fore* (prefix)", 300, || {
        let _ = single.search("fore*", 10).unwrap();
    });
    time_q("fox~1", 200, || {
        let _ = single.search("fox~1", 10).unwrap();
    });
    time_q("doc77 (rare)", 2000, || {
        let _ = single.search("doc77", 10).unwrap();
    });

    println!("== 10 segments (merge-heavy) ==");
    let multi = bench_index("lanes=8");
    time_q("fox", 300, || {
        let _ = multi.search("fox", 10).unwrap();
    });
    time_q("fox AND dog", 200, || {
        let _ = multi.search("fox AND dog", 10).unwrap();
    });

    println!("== ids + tombstones (masked search overhead) ==");
    let index = {
        let index = build(1);
        let docs: Vec<Document> = corpus()
            .into_iter()
            .enumerate()
            .map(|(i, d)| Document::new().id(format!("d{i}")).text("text", d))
            .collect();
        index.add_documents(docs).unwrap();
        index.commit();
        index
    };
    // delete 1% to arm the tombstone masks
    let t = Instant::now();
    for i in (0..N).step_by(100) {
        assert!(index.delete(&format!("d{i}")).unwrap());
    }
    println!(
        "{:>38}: {:8.3} us/delete",
        "delete 1% tombstones",
        t.elapsed().as_secs_f64() * 1e6 / (N / 100) as f64
    );
    time_q("fox (with tombstones)", 300, || {
        let _ = index.search("fox", 10).unwrap();
    });
    time_q("get-by-id d5001", 20000, || {
        let _ = index.get("d5001").unwrap();
    });

    println!("== embedding search (flat scan) ==");
    {
        let mut schema = Schema::default();
        schema.add_text("text", 1.0);
        schema.add_vector("emb", 768, Metric::Cosine);
        let index = Index::with_lanes(schema, Analyzer::new(AnalyzerConf::default()), 1.2, 0.75, 8);
        let mut state = 0x2545F491_4F6CDD1Du64;
        let mut rnd = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let docs: Vec<Document> = (0..N)
            .map(|i| {
                let v: Vec<f32> = (0..768)
                    .map(|_| ((rnd() % 2000) as f32 / 2000.0) - 0.5)
                    .collect();
                Document::new()
                    .id(format!("d{i}"))
                    .text("text", "embedding corpus")
                    .vector("emb", v)
            })
            .collect();
        index.add_documents(docs).unwrap();
        index.commit();
        let q: Vec<f32> = (0..768)
            .map(|_| ((rnd() % 2000) as f32 / 2000.0) - 0.5)
            .collect();
        // kernel-level: the AVX2/FMA dot vs the scalar reference over 768 dims
        let a: Vec<f32> = (0..768)
            .map(|_| ((rnd() % 2000) as f32 / 2000.0) - 0.5)
            .collect();
        let mut sink = 0.0f32;
        let iters = 200_000;
        let q768 = std::hint::black_box(&q);
        let a768 = std::hint::black_box(&a);
        let t = Instant::now();
        for _ in 0..iters {
            // the accumulator flows through a black box: without it LLVM
            // folds the whole loop and the timing reads 0.1ns
            sink = std::hint::black_box(sink) + zap::vector::kernels::dot(a768, q768);
        }
        let avx = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let t = Instant::now();
        for _ in 0..iters {
            sink = std::hint::black_box(sink) + zap::vector::kernels::dot_scalar(a768, q768);
        }
        let scalar = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        std::hint::black_box(sink);
        println!(
            "{:>38}: dot {avx:.3} us | scalar {scalar:.3} us (768 dims)",
            "similarity kernel"
        );
        time_q("vector top-10 (10k x 768)", 50, || {
            let _ = index
                .search_request(&SearchRequest::vector(q.clone(), 10), index.pub_stats())
                .unwrap();
        });
        time_q("hybrid top-10 (10k x 768)", 50, || {
            let _ = index
                .search_request(
                    &SearchRequest::hybrid("fox", q.clone(), 10),
                    index.pub_stats(),
                )
                .unwrap();
        });
    }

    println!("total wall: {:.1}s", t0.elapsed().as_secs_f64());
}
