//! T6: degenerate inputs: empty docs/queries/segments, boundary tfs, unicode offsets

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
fn empty_index_commits() {
    let index = build();
    index.commit();
    assert_eq!(index.num_docs(), 0);
    assert_eq!(index.num_segments(), 0);
    assert!(index.search("anything", 5).unwrap().is_empty());
    // multiple empty commits fine
    index.commit();
    index.commit();
    assert_eq!(index.num_segments(), 0);
}

#[test]
fn analyzer_only_tokens() {
    let index = build();
    // doc whose text is entirely stopwords → zero tokens: stored but unindexed
    index
        .add_document(&Document::new().text("text", "! ! !! ? ??"))
        .unwrap();
    // empty-field candidate
    index
        .add_document(&Document::new().text("text", ""))
        .unwrap();
    index.commit();
    assert_eq!(index.num_docs(), 2);
    assert!(index.search("zz", 5).unwrap().is_empty());
}

#[test]
fn exactly_boundary_tf() {
    let index = build();
    // tf straddling the 128-doc skip boundary: 128+300 same-doc repeats
    let mut text = String::new();
    for _ in 0..128 {
        text.push_str("uniq ");
    }
    for _ in 0..300 {
        text.push_str("core ");
    }
    index
        .add_document(&Document::new().text("text", text))
        .unwrap();
    index.commit();
    assert_eq!(index.search("uniq", 3).unwrap().len(), 1);
    assert_eq!(index.search("uniq*", 3).unwrap().len(), 1);
    // phrase near the very end of the doc
    assert_eq!(index.search("\"uniq uniq\"", 3).unwrap().len(), 1);
}

#[test]
fn unicode_byte_offsets() {
    let index = build();
    // multi-byte content: byte offsets must be char-safe in snippets
    index
        .add_document(&Document::new().text("text", "café 中文 naive café 中"))
        .unwrap();
    index.commit();
    let hits = index.search("中文", 3).unwrap();
    assert_eq!(hits.len(), 1);
}

#[test]
fn unknown_field_rejected() {
    let index = build();
    let mut schema = Schema::empty();
    schema.add_text("text", 1.0);
    let r = index.add_document(&Document::new().text("nope", "val"));
    assert!(r.is_err(), "unknown field rejected");
}
