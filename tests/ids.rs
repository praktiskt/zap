//! external ids + stored metadata + replace/delete semantics (T11)

use std::sync::Arc;
use zap::analyzer::{Analyzer, AnalyzerConf};
use zap::index::Index;
use zap::schema::{Document, Schema};
use zap::segment::SegmentReader;

fn index_with_lanes(lanes: usize) -> Arc<Index> {
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

fn doc(id: &str, text: &str) -> Document {
    Document::new().id(id).text("text", text)
}

#[test]
fn id_roundtrip_and_get() {
    let index = index_with_lanes(4);
    index
        .add_documents(vec![
            doc("a1", "the quick brown fox"),
            Document::new().text("text", "lazy dog sleeps"),
            doc("b2", "foxes in the winter forest"),
        ])
        .unwrap();
    index.commit();

    let p = index.get("a1").expect("id a1 resolves");
    assert_eq!(p.id.as_deref(), Some("a1"));
    assert!(p.fields.iter().any(|(_, t)| t.contains("quick brown fox")));

    let p = index.get("b2").expect("id b2 resolves");
    assert!(p.fields.iter().any(|(_, t)| t.contains("winter forest")));

    assert!(index.get("missing").is_none());
    // doc without id is not reachable through get
    assert!(index.get("lazy dog sleeps").is_none());
}

#[test]
fn id_lookup_before_commit_invisible_after_commit_visible() {
    let index = index_with_lanes(2);
    index.add_document(&doc("x", "alpha beta")).unwrap();
    // uncommitted: invisible to get (consistent with search visibility)
    assert!(index.get("x").is_none());
    index.commit();
    assert!(index.get("x").is_some());
}

#[test]
fn replace_tombstones_old_doc() {
    let index = index_with_lanes(4);
    index
        .add_documents(vec![
            doc("k1", "original alpha content"),
            doc("k2", "unrelated beta content"),
        ])
        .unwrap();
    index.commit();
    assert_eq!(index.num_docs(), 2);

    // replace k1 across a commit boundary
    index
        .add_document(&doc("k1", "replacement gamma content"))
        .unwrap();
    index.commit();
    assert_eq!(index.num_docs(), 2, "replace keeps doc count stable");

    let p = index.get("k1").unwrap();
    assert!(p.fields[0].1.contains("replacement"));
    let texts: Vec<String> = index
        .search("content", 10)
        .unwrap()
        .into_iter()
        .filter_map(|h| index.doc_payload(h.addr))
        .map(|p| p.fields[0].1.clone())
        .collect();
    assert_eq!(texts.len(), 2, "only live docs hit: {texts:?}");
    assert!(
        !texts.iter().any(|t| t.contains("original")),
        "old version must be gone"
    );
    assert!(texts.iter().any(|t| t.contains("replacement")));
    assert!(texts.iter().any(|t| t.contains("unrelated")));
}

#[test]
fn replace_uncommitted_hot_doc_commits_first() {
    let index = index_with_lanes(2);
    index.add_document(&doc("h1", "hot version one")).unwrap();
    // no commit: hot doc is replaced -> commit fires internally
    index.add_document(&doc("h1", "hot version two")).unwrap();
    index.commit();
    assert_eq!(index.num_docs(), 1, "old hot version flushed+replaced");
    let p = index.get("h1").unwrap();
    assert!(p.fields[0].1.contains("two"));
    assert!(index.search("version", 10).unwrap().len() == 1);
}

#[test]
fn delete_hides_doc_and_stats_drop() {
    let index = index_with_lanes(4);
    index
        .add_documents(vec![
            doc("d1", "delta target here"),
            doc("d2", "epsilon other"),
        ])
        .unwrap();
    index.commit();
    assert!(index.delete("d1").unwrap());
    assert_eq!(index.num_docs(), 1, "live count drops after delete");
    assert!(index.get("d1").is_none());
    let texts: Vec<String> = index
        .search("target OR other", 10)
        .unwrap()
        .into_iter()
        .filter_map(|h| index.doc_payload(h.addr))
        .map(|p| p.fields[0].1.clone())
        .collect();
    assert!(!texts.iter().any(|t| t.contains("target")), "{texts:?}");
    assert_eq!(texts.len(), 1);
    // double delete reports miss
    assert!(!index.delete("d1").unwrap());
}

#[test]
fn delete_uncommitted_hot_doc() {
    let index = index_with_lanes(2);
    index.add_document(&doc("hd", "hot target doc")).unwrap();
    assert!(index.delete("hd").unwrap());
    index.commit();
    assert_eq!(index.num_docs(), 0);
    assert!(index.get("hd").is_none());
}

#[test]
fn merge_purges_tombstones_and_keeps_ids() {
    let index = index_with_lanes(1);
    // force merges: MAX_SEGMENTS * lanes = 4 segments triggers merge at commit
    for i in 0..12 {
        index
            .add_document(&doc(&format!("m{i}"), &format!("merge doc number {i}")))
            .unwrap();
        index.commit();
    }
    // tombstone half, then commit enough to force a merge
    for i in 0..6 {
        assert!(index.delete(&format!("m{i}")).unwrap());
    }
    for i in 12..16 {
        index
            .add_document(&doc(&format!("m{i}"), &format!("merge doc number {i}")))
            .unwrap();
        index.commit();
    }
    assert_eq!(index.num_docs(), 10, "purged docs leave the live count");
    for i in 0..6 {
        assert!(index.get(&format!("m{i}")).is_none(), "m{i} must be purged");
    }
    for i in 6..16 {
        let p = index.get(&format!("m{i}")).expect("live id resolves");
        assert!(p.fields[0].1.contains(&format!("number {i}")));
    }
    // replaced id lands correctly after purge
    index.add_document(&doc("m7", "brand new seven")).unwrap();
    index.commit();
    let p = index.get("m7").unwrap();
    assert!(p.fields[0].1.contains("brand new seven"));
}

#[test]
fn search_results_carry_external_id_and_metadata() {
    let index = index_with_lanes(2);
    index
        .add_documents(vec![
            Document::new()
                .id("meta-1")
                .text("text", "metadata fox doc")
                .metadata(serde_json::json!({"lang": "en", "tags": ["a", "b"]})),
            doc("meta-2", "another fox doc"),
            Document::new().text("text", "no id fox doc"),
        ])
        .unwrap();
    index.commit();
    let hits = index.search("fox", 10).unwrap();
    assert_eq!(hits.len(), 3);
    let mut seen_meta = 0;
    for h in &hits {
        let p = index.doc_payload(h.addr).unwrap();
        if p.id.as_deref() == Some("meta-1") {
            seen_meta += 1;
            assert_eq!(
                p.metadata,
                Some(serde_json::json!({"lang": "en", "tags": ["a", "b"]}))
            );
        } else if p.id.is_none() {
            assert!(p.metadata.is_none());
        }
    }
    assert_eq!(seen_meta, 1);
}

#[test]
fn metadata_survives_merge() {
    let index = index_with_lanes(1);
    for i in 0..8 {
        index
            .add_document(
                &Document::new()
                    .id(format!("s{i}"))
                    .text("text", format!("surv doc {i}"))
                    .metadata(serde_json::json!({"n": i})),
            )
            .unwrap();
        index.commit();
    }
    for i in 0..8 {
        let p = index.get(&format!("s{i}")).expect("id survives merges");
        assert_eq!(p.metadata, Some(serde_json::json!({"n": i})));
    }
}

#[test]
fn duplicate_ids_within_one_batch_last_wins() {
    let index = index_with_lanes(4);
    index
        .add_documents(vec![
            doc("dup", "first version"),
            doc("dup", "second version"),
            doc("dup", "third version"),
        ])
        .unwrap();
    index.commit();
    assert_eq!(index.num_docs(), 1, "batch dedup keeps last doc only");
    let p = index.get("dup").unwrap();
    assert!(p.fields[0].1.contains("third"));
}

#[test]
fn empty_id_rejected() {
    let index = index_with_lanes(1);
    let r = index.add_document(&Document::new().id("").text("text", "x"));
    assert!(r.is_err(), "empty id must be rejected");
}

#[test]
fn wand_and_dense_paths_mask_deleted() {
    // Term query exercises the WAND streaming path; OR/NOT exercise dense paths
    let index = index_with_lanes(2);
    index
        .add_documents(vec![
            doc("w1", "alpha bravo charlie"),
            doc("w2", "alpha delta echo"),
            doc("w3", "bravo charlie delta"),
        ])
        .unwrap();
    index.commit();
    index.delete("w2").unwrap();

    let hits = index.search("alpha", 10).unwrap();
    assert_eq!(hits.len(), 1, "WAND path must mask tombstones");
    assert_eq!(
        index.doc_payload(hits[0].addr).unwrap().id.as_deref(),
        Some("w1")
    );

    let hits = index.search("alpha OR delta", 10).unwrap();
    let ids: Vec<Option<String>> = hits
        .iter()
        .map(|h| index.doc_payload(h.addr).unwrap().id)
        .collect();
    assert_eq!(ids.len(), 2, "{ids:?}");
    assert!(ids.contains(&Some("w1".into())));
    assert!(ids.contains(&Some("w3".into())));

    // NOT over a deleted doc: w2 must not resurface through the complement
    let hits = index.search("bravo NOT charlie", 10).unwrap();
    let ids: Vec<String> = hits
        .iter()
        .filter_map(|h| index.doc_payload(h.addr).unwrap().id)
        .collect();
    assert!(!ids.contains(&"w2".to_string()), "{ids:?}");
}

#[test]
fn fuzzy_prefix_paths_mask_deleted() {
    let index = index_with_lanes(1);
    index
        .add_documents(vec![
            doc("f1", "foxes run fast"),
            doc("f2", "foxes run slow"),
        ])
        .unwrap();
    index.commit();
    index.delete("f1").unwrap();
    let hits = index.search("fox~1", 10).unwrap();
    assert_eq!(hits.len(), 1, "fuzzy path must mask tombstones");
    assert_eq!(
        index.doc_payload(hits[0].addr).unwrap().id.as_deref(),
        Some("f2")
    );
    let hits = index.search("slo*", 10).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(
        index.doc_payload(hits[0].addr).unwrap().id.as_deref(),
        Some("f2")
    );
}

#[test]
fn segment_reader_id_lookup_direct() {
    let index = index_with_lanes(1);
    index
        .add_documents(vec![
            doc("zz-last", "a"),
            doc("aa-first", "b"),
            doc("mm-mid", "c"),
        ])
        .unwrap();
    index.commit();
    let segs = index.segments_snapshot();
    assert_eq!(segs.len(), 1);
    let seg: Arc<SegmentReader> = segs[0].clone();
    assert_eq!(seg.lookup_id("aa-first"), Some(1));
    assert_eq!(seg.lookup_id("mm-mid"), Some(2));
    assert_eq!(seg.lookup_id("zz-last"), Some(0));
    assert_eq!(seg.lookup_id("nope"), None);
    assert_eq!(seg.id_of(1), Some("aa-first"));
    assert_eq!(seg.id_of(0), Some("zz-last"));
}
