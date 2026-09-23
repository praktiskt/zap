//! posting decode microprobe at 1M-doc scale: isolates decode cost from query
//! machinery. A/B against the varint format via `git stash`.
//!
//!   cargo run --release --example postprobe

use std::hint::black_box;
use std::time::Instant;

use zap::analyzer::{Analyzer, AnalyzerConf};
use zap::index::Index;
use zap::schema::{Document, Schema};

fn main() {
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    let index = Index::with_lanes(schema, Analyzer::new(AnalyzerConf::default()), 1.2, 0.75, 8);
    let words = [
        "fox", "quick", "brown", "jumps", "lazy", "dog", "forest", "river", "mountain", "stone",
        "winter", "summer", "spring", "autumn", "night", "day", "sun", "moon", "star", "cloud",
        "wind", "rain", "snow", "storm", "field", "path", "road", "bridge", "house", "garden",
        "tree", "leaf", "branch", "root", "flower", "grass", "ceremony", "sand", "wave", "ocean",
        "fish", "bird", "nest", "feather", "wing", "sky", "dawn", "dusk", "shadow", "light",
    ];
    let n = 1_000_000usize;
    let per_doc = 50usize;
    let mut chunk: Vec<Document> = Vec::with_capacity(50_000);
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    for i in 0..n {
        let mut s = String::with_capacity(per_doc * 7);
        for _w in 0..per_doc {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let word = if i % 17 == 0 {
                "quixotic"
            } else {
                words[(seed as usize) % words.len()]
            };
            s.push_str(word);
            s.push(' ');
        }
        chunk.push(Document::new().text("text", s));
        if chunk.len() == 50_000 {
            index.add_documents(std::mem::take(&mut chunk)).unwrap();
            index.commit();
        }
    }
    if !chunk.is_empty() {
        index.add_documents(chunk).unwrap();
        index.commit();
    }
    let segs = index.segments_snapshot();
    let seg = &segs[0];
    // snowball stems quixotic -> quixot at index time; probe both a sparse-delta
    // (df 59k) and a dense-delta (df ~650k) posting
    for term in ["quixot", "storm"] {
        let Some(info) = seg.term(0, term) else {
            panic!("{term} missing");
        };
        println!("{term} df={}", info.df);
        probe(seg, &info);
    }
}

fn probe(seg: &zap::segment::SegmentReader, info: &zap::posting::TermInfo) {
    let iters = 200usize;
    // (a) docid_vec: pure docid decode
    let t = Instant::now();
    let mut sink = 0usize;
    for _ in 0..iters {
        let v = seg.reader(info).docid_vec();
        sink += v.len();
    }
    println!(
        "docid_vec   {:9.1} us  ({:.2} ns/doc)",
        t.elapsed().as_secs_f64() * 1e6 / iters as f64,
        t.elapsed().as_secs_f64() * 1e9 / (iters * info.df as usize) as f64
    );
    // (b) full advance loop (docid + tf + peek machinery)
    let t = Instant::now();
    for _ in 0..iters {
        let mut r = seg.reader(info);
        while let Some((d, tf)) = r.advance() {
            black_box((d, tf));
        }
    }
    println!(
        "advance     {:9.1} us  ({:.2} ns/doc)",
        t.elapsed().as_secs_f64() * 1e6 / iters as f64,
        t.elapsed().as_secs_f64() * 1e9 / (iters * info.df as usize) as f64
    );
    black_box(sink);
}
