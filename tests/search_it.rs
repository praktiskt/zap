//! end-to-end integration tests

use std::sync::Arc;

use zap::analyzer::{Analyzer, AnalyzerConf};
use zap::index::Index;
use zap::schema::{Document, Schema};

fn mk_index() -> Arc<Index> {
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    let analyzer = Analyzer::new(AnalyzerConf::default());
    Index::new(schema, analyzer, 1.2, 0.75)
}

const DOCS: [&str; 6] = [
    "the quick brown fox jumps over the lazy dog",
    "A fox ran fast and the dog chased it",
    "foxes are quick and clever in the forest",
    "the lazy dog sleeps all day",
    "matching fuzzy is fast fun",
    "中文搜索测试 the fox again",
];

fn feed(index: &Index) {
    for text in DOCS {
        index
            .add_document(&Document::new().text("text", text.to_string()))
            .unwrap();
    }
    index.commit();
}

fn texts(index: &Index, q: &str) -> Vec<String> {
    index
        .search(q, 10)
        .unwrap()
        .into_iter()
        .filter_map(|h| index.doc_payload(h.addr))
        .map(|f| f.fields[0].1.clone())
        .collect()
}

#[test]
fn boolean_ops() {
    let index = mk_index();
    feed(&index);
    let and = texts(&index, "fox AND dog");
    assert_eq!(and.len(), 2, "{and:?}");
    let or = texts(&index, "forest OR fuzzy");
    assert_eq!(or.len(), 2, "{or:?}");
    let not = texts(&index, "fox NOT forest");
    assert_eq!(not.len(), 3, "{not:?}");
    assert!(not.iter().all(|t| !t.contains("forest")));
}

/// the DOC pipeline normalizes (charabia folds the accents: "café" -> "cafe"),
/// so the QUERY term has to be normalized the same way — otherwise a query
/// for "café" misses the indexed term "cafe" (caught by the real-data run:
/// "hembränt" matched nothing while "hembrant" hit the same documents)
#[test]
fn query_terms_are_normalized_like_docs() {
    let index = mk_index();
    index
        .add_documents(vec![
            Document::new().text("text", "café latte i Stockholm"),
            Document::new().text("text", "naïve approach to hembränt"),
            Document::new().text("text", "unrelated plain text"),
        ])
        .unwrap();
    index.commit();
    for q in ["café", "cafe", "CAFÉ", "naïve", "hembränt", "hembrant"] {
        let hits = texts(&index, q);
        assert!(!hits.is_empty(), "query {q} matched nothing: {hits:?}");
    }
    // the folded form and the accented form find the SAME documents
    assert_eq!(
        texts(&index, "hembränt"),
        texts(&index, "hembrant"),
        "accented and folded queries must agree"
    );
}

#[test]
fn phrase_tests() {
    let index = mk_index();
    feed(&index);
    assert!(texts(&index, "\"quick brown fox\"").len() == 1);
    assert!(texts(&index, "\"brown quick fox\"").is_empty());
    assert_eq!(texts(&index, "\"ran fast\"").len(), 1);
    assert_eq!(texts(&index, "\"fast the\"").len(), 0);
    assert_eq!(texts(&index, "\"fast the\"~1").len(), 1);
}

#[test]
fn prefix_wild_fuzzy() {
    let index = mk_index();
    feed(&index);
    assert_eq!(texts(&index, "fuzz*").len(), 1);
    assert_eq!(texts(&index, "f*zz*").len(), 1);
    assert_eq!(texts(&index, "fuzzy~1").len(), 1);
    // fuzzy of a non-existing word via edit distance
}

#[test]
fn field_scope() {
    let index = mk_index();
    feed(&index);
    assert_eq!(texts(&index, "text:lazy").len(), 2);
    // unknown field returns nothing
    assert_eq!(texts(&index, "title:lazy").len(), 0);
}

#[test]
fn commit_merge() {
    let index = mk_index();
    for (i, text) in DOCS.iter().enumerate() {
        index
            .add_document(&Document::new().text("text", text.to_string()))
            .unwrap();
        if i % 2 == 1 {
            index.commit();
        }
    }
    index.commit();
    assert_eq!(index.num_docs(), 6);
    assert_eq!(texts(&index, "fox AND dog").len(), 2);
    let hits = index.search("fox", 5).unwrap();
    assert_eq!(hits.len(), 4);
    // every hit contains the stem `fox`
    for h in &hits {
        let f = index.doc_payload(h.addr).unwrap();
        assert!(
            f.fields[0].1.contains("fox") || f.fields[0].1.contains("foxes"),
            "{f:?}"
        );
    }
}

#[test]
fn cjk() {
    let index = mk_index();
    feed(&index);
    let hits = texts(&index, "搜索");
    assert_eq!(hits.len(), 1);
    assert!(hits[0].contains("搜索"));
}
