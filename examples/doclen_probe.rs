//! doc-length profiling probe: separates tokenize / index / store costs for big
//! (1M chars) and small (1k chars) documents, plus phrase + term query cost.
//! Pair with perf:  perf record -g -- cargo run --release --example doclen_probe -- <phase>
//!
//!   cargo run --release --example doclen_probe -- ingest1m
//!   cargo run --release --example doclen_probe -- query1m
//!   cargo run --release --example doclen_probe -- ingest1k

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use zap::analyzer::{Analyzer, AnalyzerConf};
use zap::index::Index;
use zap::schema::{Document, Schema};

const WORDS: [&str; 116] = [
    "fox", "quick", "brown", "jumps", "lazy", "dog", "forest", "river", "mountain", "stone",
    "winter", "summer", "spring", "autumn", "night", "day", "sun", "moon", "star", "cloud", "wind",
    "rain", "snow", "storm", "field", "path", "road", "bridge", "house", "garden", "tree", "leaf",
    "branch", "root", "flower", "grass", "stone", "sand", "wave", "ocean", "fish", "bird", "nest",
    "feather", "wing", "sky", "dawn", "dusk", "shadow", "light", "fire", "water", "earth", "air",
    "metal", "wood", "glass", "paper", "ink", "word", "sentence", "story", "song", "dance",
    "music", "artist", "player", "game", "match", "team", "goal", "score", "point", "line",
    "curve", "angle", "circle", "square", "box", "code", "rust", "engine", "search", "index",
    "token", "query", "term", "field", "doc", "rank", "score", "boost", "merge", "segment",
    "posting", "bitmap", "stem", "stop", "fuzzy", "prefix", "phrase", "window", "position",
    "length", "norm", "field", "body", "title", "text", "plain", "fast", "slow", "deep", "wide",
    "open", "close",
];

/// deterministic corpus, same shape as the criterion bench
fn corpus(n_docs: usize, words_per_doc: usize) -> Vec<String> {
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
        for w in 0..words_per_doc {
            let pick = rnd();
            let word_i = if w % 16 == 0 {
                (pick % 8) as usize
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

fn index1(lanes: usize) -> Arc<Index> {
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

fn main() {
    let phase = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    match phase.as_str() {
        "tokenize1m" => {
            // raw analyzer cost on 1M-char docs (no indexing)
            let docs = corpus(4, 160_000);
            let an = Analyzer::new(AnalyzerConf::default());
            let t = Instant::now();
            let mut n = 0u64;
            for d in &docs {
                n += an.tokenize(d).len() as u64;
            }
            let dt = t.elapsed();
            println!(
                "tokenize 4x1M-char: {dt:?}  ({:.1} MB/s, {} tokens)",
                (docs.iter().map(|d| d.len()).sum::<usize>() as f64 / 1e6) / dt.as_secs_f64(),
                n
            );
        }
        "ingest1m" => {
            let docs = corpus(16, 160_000);
            let ix = index1(1);
            let t = Instant::now();
            for d in &docs {
                ix.add_document(&Document::new().text("text", d.clone()))
                    .unwrap();
            }
            let t_add = t.elapsed();
            ix.commit();
            let t_commit = t.elapsed() - t_add;
            println!(
                "ingest 16x1M-char: add {t_add:?} + commit {t_commit:?}  ({:.1} MB/s)",
                (docs.iter().map(|d| d.len()).sum::<usize>() as f64 / 1e6)
                    / t.elapsed().as_secs_f64()
            );
        }
        "ingest1k" => {
            let docs = corpus(1000, 170);
            let ix = index1(1);
            let t = Instant::now();
            for d in &docs {
                ix.add_document(&Document::new().text("text", d.clone()))
                    .unwrap();
            }
            let t_add = t.elapsed();
            ix.commit();
            let t_commit = t.elapsed() - t_add;
            println!(
                "ingest 1000x1k-char: add {t_add:?} + commit {t_commit:?}  ({:.1} MB/s)",
                (docs.iter().map(|d| d.len()).sum::<usize>() as f64 / 1e6)
                    / t.elapsed().as_secs_f64()
            );
        }
        "query1m" | "all" => {
            let docs = corpus(16, 160_000);
            let ix = index1(1);
            for d in &docs {
                ix.add_document(&Document::new().text("text", d.clone()))
                    .unwrap();
            }
            ix.commit();
            for (tag, q, iters) in [
                ("term fox (hot)", "fox", 50),
                ("phrase \"winter storm\"", "\"winter storm\"", 50),
                ("rare doc15", "doc15", 1000),
            ] {
                // warmup
                for _ in 0..2 {
                    black_box(ix.search(q, 10).unwrap());
                }
                let t = Instant::now();
                for _ in 0..iters {
                    black_box(ix.search(q, 10).unwrap());
                }
                println!(
                    "query1m {tag:>26}: {:9.1} us",
                    t.elapsed().as_secs_f64() * 1e6 / iters as f64
                );
            }
        }
        "querybigscale" => {
            // the big-docs-at-scale scenario: 1000 docs x 1M chars (~1GB of
            // text, 160M tokens). Uses the parallel lane ingest.
            let docs = corpus(1000, 160_000);
            let mut schema = Schema::default();
            schema.add_text("text", 1.0);
            let index =
                Index::with_lanes(schema, Analyzer::new(AnalyzerConf::default()), 1.2, 0.75, 8);
            let t = Instant::now();
            for d in &docs {
                index
                    .add_document(&Document::new().text("text", d.clone()))
                    .unwrap();
            }
            index.commit();
            println!(
                "ingest 1000x1M-char: {t:?} ({:.1} MB/s, segs={})",
                (docs.iter().map(|d| d.len()).sum::<usize>() as f64 / 1e6)
                    / t.elapsed().as_secs_f64(),
                index.num_segments()
            );
            for (tag, q, iters) in [
                ("term fox (hot)", "fox", 50),
                ("phrase \"winter storm\"", "\"winter storm\"", 20),
                ("phrase \"storm bridge\"", "\"storm bridge\"", 20),
                ("rare doc999", "doc999", 500),
            ] {
                for _ in 0..2 {
                    black_box(index.search(q, 10).unwrap());
                }
                let t = Instant::now();
                for _ in 0..iters {
                    black_box(index.search(q, 10).unwrap());
                }
                println!(
                    "bigscale {tag:>26}: {:9.1} us",
                    t.elapsed().as_secs_f64() * 1e6 / iters as f64
                );
            }
        }
        "phrase_breakdown" => {
            // where does 1M-char phrase time go: docid intersect vs position
            // decode vs min-window verify
            let docs = corpus(16, 160_000);
            let ix = index1(1);
            for d in &docs {
                ix.add_document(&Document::new().text("text", d.clone()))
                    .unwrap();
            }
            ix.commit();
            let segs = ix.segments_snapshot();
            let seg = &segs[0];
            let a = Analyzer::new(AnalyzerConf::default());
            let stem = |w: &str| a.tokenize(w).first().map(|t| t.text.to_string()).unwrap();
            let iters = 200usize;
            let mut out = 0u64;
            // (0) direct phrase_scores call (bypasses query parse/exec tree)
            {
                let t1 = stem("winter");
                let t2 = stem("storm");
                let i1 = seg.term(0, &t1).expect("winter");
                let i2 = seg.term(0, &t2).expect("storm");
                let t = Instant::now();
                for _ in 0..iters {
                    let ds = zap::search::phrase::phrase_scores(
                        &[i1.clone(), i2.clone()],
                        seg.blob(),
                        1.0,
                        0,
                        1.5,
                    );
                    out += ds.ids.len() as u64;
                }
                println!(
                    "  phrase_scores direct:  {:8.1} us",
                    t.elapsed().as_secs_f64() * 1e6 / iters as f64
                );
                // manual replica of the cursor loop, real min_span, no clones
                use zap::search::phrase::TermCursor;
                let t = Instant::now();
                for _ in 0..iters {
                    let mut ca = TermCursor::new(&i1, seg.blob());
                    let mut cb = TermCursor::new(&i2, seg.blob());
                    let mut lists: Vec<Vec<u32>> = vec![Vec::new(), Vec::new()];
                    let mut n = 0u64;
                    loop {
                        let (Some(da), Some(db)) = (ca.doc(), cb.doc()) else {
                            break;
                        };
                        let hi = da.max(db);
                        ca.seek_to(hi);
                        cb.seek_to(hi);
                        if ca.doc() == Some(hi) && cb.doc() == Some(hi) {
                            ca.drain_positions(&mut lists[0]);
                            cb.drain_positions(&mut lists[1]);
                            if let Some(span) = zap::search::phrase::min_span(&lists, 0) {
                                n += span as u64;
                            } else {
                                n += lists[0].len() as u64;
                            }
                        }
                    }
                    out += n;
                }
                println!(
                    "  manual cursor loop:    {:8.1} us",
                    t.elapsed().as_secs_f64() * 1e6 / iters as f64
                );
            }
            for word in ["winter", "storm"] {
                let term = stem(word);
                let Some(info) = seg.term(0, &term) else {
                    continue;
                };
                // (1) docid gallop + peek cost
                let t = Instant::now();
                for _ in 0..iters {
                    let mut r = seg.reader(&info);
                    while let Some((d, _)) = r.advance() {
                        out += d as u64;
                    }
                }
                println!(
                    "  {word}: docid walk+tf  {:8.1} us",
                    t.elapsed().as_secs_f64() * 1e6 / iters as f64
                );
                // (2) full position decode for every doc
                let t = Instant::now();
                for _ in 0..iters {
                    let mut r = seg.reader(&info);
                    let mut buf = Vec::new();
                    let mut n = 0usize;
                    while r.advance().is_some() {
                        r.peek_positions_into(&mut buf);
                        n += buf.len();
                    }
                    out += n as u64;
                }
                println!(
                    "  {word}: +positions    {:8.1} us",
                    t.elapsed().as_secs_f64() * 1e6 / iters as f64
                );
            }
            // (3) min_span on two 1400-element lists (window verify shape)
            let mk = |seed: u64| -> Vec<u32> {
                let mut v = Vec::with_capacity(1400);
                let mut x = seed;
                for _ in 0..1400 {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                    v.push((x % 4000) as u32);
                }
                v.sort_unstable();
                v.dedup();
                v
            };
            let l1 = mk(1);
            let l2 = mk(2);
            let t = Instant::now();
            for _ in 0..iters {
                if let Some(s) = zap::search::phrase::min_span(&[l1.clone(), l2.clone()], 0) {
                    out += s as u64;
                }
            }
            println!(
                "  min_span 2x{:.0}:        {:8.1} us",
                l1.len(),
                t.elapsed().as_secs_f64() * 1e6 / iters as f64
            );
            black_box(out);
        }
        _ => eprintln!("phases: tokenize1m ingest1m ingest1k query1m phrase_breakdown all"),
    }
}
