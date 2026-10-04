//! T8: WAND block skipping must never reorder the true top-k:
//! skewed tf outliers must always outrank tf-1 docs on the same term.

use zap::analyzer::{Analyzer, AnalyzerConf};
use zap::index::Index;
use zap::schema::{Document, Schema};

#[test]
fn outliers_rank_on_top() {
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    let analyzer = Analyzer::new(AnalyzerConf::default());
    let index = Index::with_lanes(schema, analyzer, 1.2, 0.75, 4);
    let words = [
        "fox", "quick", "brown", "jumps", "lazy", "dog", "forest", "river",
    ];
    let n = 100_000u32;
    let mut chunk: Vec<Document> = Vec::with_capacity(20_000);
    let mut seed: u64 = 0x9E3779B97F4A7C15u64;
    for i in 0..n {
        let mut s = String::with_capacity(96);
        // outlier docs: high tf for 'fox' every 1000th doc
        if i % 1000 == 500 {
            s.push_str(&"fox ".repeat(100));
        }
        for _w in 0..8 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let word = words[((seed >> 8) as usize) % words.len()];
            s.push_str(word);
            s.push(' ');
        }
        chunk.push(Document::new().text("text", s));
        if chunk.len() == 20_000 {
            index.add_documents(std::mem::take(&mut chunk)).unwrap();
            index.commit();
        }
    }
    if !chunk.is_empty() {
        index.add_documents(chunk).unwrap();
        index.commit();
    }
    let hits = index.search("fox", 10).unwrap();
    assert_eq!(hits.len(), 10);
    // every returned doc must be an outlier: at least the 100 planted
    // 'fox' occurrences ("fox" is also in the random word list, so true
    // outliers usually carry a few extra). The docid % 1000 == 500 pattern
    // is NOT a valid invariant: subset tier merges re-index docids
    // mid-corpus (the merged segment's row 0 starts wherever its first
    // source started), which shifts the modulo pattern without changing
    // which docs are outliers. The tf itself is the truth.
    for h in &hits {
        let text = index.doc_payload(h.addr).expect("hit payload");
        let fcount = text
            .fields
            .iter()
            .map(|f| f.1.as_str())
            .collect::<String>()
            .matches("fox ")
            .count();
        assert!(
            fcount >= 100,
            "non-outlier doc leaked into top-k: {:?} (fcount={})",
            h.addr.docid,
            fcount
        );
    }
    // scores ordered high -> low
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score, "descending order violated");
    }
}
