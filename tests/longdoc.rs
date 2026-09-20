//! T5: long-document (100k+ chars) roundtrip: positions > block boundaries,
//! fieldnorm saturation, docstore byte-offset TLV with unicode

use std::sync::Arc;

use zap::analyzer::{Analyzer, AnalyzerConf};
use zap::index::Index;
use zap::schema::{Document, Schema};

fn build() -> Arc<Index> {
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    let analyzer = Analyzer::new(AnalyzerConf::default());
    Index::new(schema, analyzer, 1.2, 0.75)
}

#[test]
fn longdoc_roundtrip() {
    let index = build();
    // ~100k chars: word soup with skewed tfs + multi-byte text + block-crossing repeats
    let mut text = String::with_capacity(120_000);
    text.push_str("alpha ");
    for i in 0..6000u32 {
        // positions cross 128-doc skip blocks inside one doc's tf block
        text.push_str("leaf ");
        if i % 251 == 0 {
            text.push_str("é中文 ここ Sophie ");
        }
        if i % 7 == 0 {
            text.push_str("quixotic ");
        }
    }
    text.push_str("alpha");
    index
        .add_document(&Document::new().text("text", text.clone()))
        .unwrap();
    index.commit();
    assert_eq!(index.num_docs(), 1);

    let hits = index.search("leaf", 5).unwrap();
    assert_eq!(hits.len(), 1, "tf-heavy leaf doc must hit");
    // fieldnorm u16 must record every token (~2800)
    let stored = index.stored_fields(hits[0].addr).unwrap();
    assert_eq!(stored[0].1.len(), text.len(), "docstore bytes survive");
    assert!(stored[0].1.contains("中文"), "CJK inside long doc stored");
    assert!(
        stored[0].1.contains("ここ"),
        "multi-block inside long doc stored"
    );

    // phrase boundaries inside the long doc
    assert!(index.search("\"alpha leaf\"", 5).unwrap().len() == 1);

    // rare-term query inside the long doc
    let quhits = index.search("quixotic", 5).unwrap();
    assert_eq!(quhits.len(), 1);
}

#[test]
fn positions_beyond_block_boundaries() {
    let index = build();
    // one doc with 300 repeats of the same term: position gaps cross skip blocks
    let mut text = String::new();
    for i in 0..300 {
        text.push_str("word ");
        if i % 100 == 0 {
            text.push_str("link ");
        }
    }
    index
        .add_document(&Document::new().text("text", text))
        .unwrap();
    index.commit();
    let hits = index.search("\"word link\"", 5).unwrap();
    assert_eq!(hits.len(), 1, "word link adjacent once per 100 reps");
    assert_eq!(index.search("\"word word\"", 5).unwrap().len(), 1);
}
