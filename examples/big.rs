fn main() {
    use zap::analyzer::{Analyzer, AnalyzerConf};
    use zap::index::Index;
    use zap::schema::{Document, Schema};
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    let analyzer = Analyzer::new(AnalyzerConf::default());
    let index = Index::with_lanes(schema, analyzer, 1.2, 0.75, 8);
    let words = [
        "fox", "quick", "brown", "jumps", "lazy", "dog", "forest", "river", "mountain", "stone",
        "winter", "summer", "spring", "autumn", "night", "day", "sun", "moon", "star", "cloud",
        "wind", "rain", "snow", "storm", "field", "path", "road", "bridge", "house", "garden",
        "tree", "leaf", "branch", "root", "flower", "grass", "ceremony", "sand", "wave", "ocean",
        "fish", "bird", "nest", "feather", "wing", "sky", "dawn", "dusk", "shadow", "light",
    ];
    let n = 1_000_000usize;
    let per_doc = 50usize;
    let t0 = std::time::Instant::now();
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
    println!(
        "ingest {} docs in {:.1}s, segs={}",
        n,
        t0.elapsed().as_secs_f32(),
        index.num_segments()
    );
    let t1 = std::time::Instant::now();
    let hits = index.search("quixotic", 10).unwrap();
    println!(
        "quixotic hits={} in {:?} (~{:.2} ms) docs={}",
        hits.len(),
        t1.elapsed(),
        t1.elapsed().as_millis() as f32,
        index.num_docs()
    );
    println!(
        "doc110 score {:?}",
        index
            .search("quixotic", 3)
            .unwrap()
            .iter()
            .map(|h| h.addr.docid)
            .collect::<Vec<_>>()
    );
    // phrase at scale: candidate df spans ~1M (zipf hot pairs), mid, rare
    let t2 = std::time::Instant::now();
    let hits = index.search("\"winter storm\"", 10).unwrap();
    println!(
        "phrase \"winter storm\" hits={} in {:.2} ms",
        hits.len(),
        t2.elapsed().as_secs_f32() * 1e3
    );
    let t3 = std::time::Instant::now();
    let hits = index.search("\"quixotic fox\"", 10).unwrap();
    println!(
        "phrase \"quixotic fox\" (df~59k) hits={} in {:.2} ms",
        hits.len(),
        t3.elapsed().as_secs_f32() * 1e3
    );
    let t4 = std::time::Instant::now();
    let hits = index.search("\"storm bridge stone\"", 10).unwrap();
    println!(
        "phrase 3-term \"storm bridge stone\" hits={} in {:.2} ms",
        hits.len(),
        t4.elapsed().as_secs_f32() * 1e3
    );
}
