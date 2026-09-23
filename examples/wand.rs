//! block-max WAND probe: skewed hot term + live threshold pruning
fn main() {
    use zap::analyzer::{Analyzer, AnalyzerConf};
    use zap::index::Index;
    use zap::schema::{Document, Schema};
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    let analyzer = Analyzer::new(AnalyzerConf::default());
    let index = Index::with_lanes(schema, analyzer, 1.2, 0.75, 8);
    // docs 0..100k: 'fox' appears in every doc; tf=1 normally, tf=1000 when i%1000==0 and quirky tf
    let words = [
        "fox", "quick", "brown", "jumps", "lazy", "dog", "forest", "river", "mountain", "stone",
    ];
    let n = 200_000usize;
    let mut chunk: Vec<Document> = Vec::with_capacity(20_000);
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    for i in 0..n {
        let mut s = String::with_capacity(80);
        if i % 1000 == 500 {
            // super-hot outliers: tf ~ 80 on 'fox'
            s.push_str(&"fox ".repeat(100));
        }
        for _w in 0..8 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let w = words[(seed as usize) % words.len()];
            s.push_str(w);
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
    println!("docs={} segs={}", index.num_docs(), index.num_segments());
    // top-10 for 'fox': the 200 tf-100 outliers should win; others tf~1
    for q in ["fox", "fox OR dog", "fox AND dog", "fox OR dog OR quick"] {
        let t = std::time::Instant::now();
        let hits = index.search(q, 10).unwrap();
        println!(
            "{}: {} hits in {:?} (~{:.2} us)",
            q,
            hits.len(),
            t.elapsed(),
            t.elapsed().as_secs_f64() * 1e6
        );
    }
    // top-10 for 'fox': the 200 tf-100 outliers should win; others tf~1
    {
        let hits = index.search("fox", 10).unwrap();
        println!(
            "scores: {:?}",
            hits.iter()
                .map(|h| (h.addr.docid, h.score))
                .collect::<Vec<_>>()
        );
    }
}
