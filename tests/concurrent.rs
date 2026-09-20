//! T7: writer commit loop vs concurrent readers (arc-swap snapshot smoke)

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use zap::analyzer::{Analyzer, AnalyzerConf};
use zap::index::Index;
use zap::schema::{Document, Schema};

#[test]
fn commit_and_concurrent_search() {
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    let analyzer = Analyzer::new(AnalyzerConf::default());
    let index = Index::with_lanes(schema, analyzer, 1.2, 0.75, 2);

    let stop = Arc::new(AtomicUsize::new(0));
    let errs = Arc::new(AtomicUsize::new(0));
    for reader_i in 0..3usize {
        let idx = index.clone();
        let stop = stop.clone();
        let errs = errs.clone();
        std::thread::spawn(move || {
            let mut queries = 0;
            while stop.load(Ordering::SeqCst) == 0 && queries < 40 {
                for q in ["fox", "river", "fox AND river", "«quick» fox~1", "forest*"] {
                    if idx.search(q, 5).is_err() {
                        errs.fetch_add(1, Ordering::SeqCst);
                    }
                    queries += 1;
                }
            }
            let _ = (reader_i, queries);
        });
    }

    // writer: many commits with growing corpus; no errors on the reader path
    for c in 0..30 {
        let docs: Vec<Document> = (0..50)
            .map(|i| {
                Document::new().text(
                    "text",
                    format!("fox river field doc{} wooded {}", c * 50 + i, c),
                )
            })
            .collect();
        index.add_documents(docs).unwrap();
        index.commit();
    }
    stop.store(1, Ordering::SeqCst);
    assert_eq!(
        errs.fetch_add(0, Ordering::SeqCst),
        0,
        "no reader errors while committing"
    );
    assert_eq!(index.num_docs(), 30 * 50);
    // every committed doc survived all merges: spot check the last batch
    assert!(index.search("doc1499", 1).unwrap().len() == 1);
}
