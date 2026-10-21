//! embedding search: ranking, validation, hybrid fusion, filter mode,
//! deletes/replaces, merges, disk round-trip and the HTTP surface

use std::sync::Arc;

use zap::analyzer::{Analyzer, AnalyzerConf};
use zap::index::Index;
use zap::schema::{Document, Metric, Schema};
use zap::search::{Fusion, SearchHit, SearchMode, SearchRequest};

fn schema(dim: u32) -> Schema {
    let mut s = Schema::default();
    s.add_text("text", 1.0);
    s.add_vector("emb", dim, Metric::Cosine);
    s
}

fn build(dim: u32) -> Arc<Index> {
    Index::with_lanes(
        schema(dim),
        Analyzer::new(AnalyzerConf::default()),
        1.2,
        0.75,
        2,
    )
}

fn build_disk(dim: u32, dir: std::path::PathBuf) -> Arc<Index> {
    Index::with_lanes_disk(
        schema(dim),
        Analyzer::new(AnalyzerConf::default()),
        1.2,
        0.75,
        2,
        Some(dir),
    )
}

/// the external ids of the hits, in rank order
fn ids(index: &Index, hits: &[SearchHit]) -> Vec<String> {
    hits.iter()
        .filter_map(|h| index.doc_payload(h.addr).and_then(|p| p.id))
        .collect()
}

fn doc(id: &str, text: &str, emb: &[f32]) -> Document {
    Document::new()
        .id(id)
        .text("text", text)
        .vector("emb", emb.to_vec())
}

#[test]
fn vector_search_ranks_by_cosine() {
    let index = build(3);
    index
        .add_documents(vec![
            doc("a", "alpha", &[1.0, 0.0, 0.0]),
            doc("b", "beta", &[0.0, 1.0, 0.0]),
            doc("c", "gamma", &[0.9, 0.1, 0.0]),
        ])
        .unwrap();
    index.commit();

    let req = SearchRequest::vector(vec![1.0, 0.0, 0.0], 3);
    let hits = index.search_request(&req, index.pub_stats()).unwrap();
    assert_eq!(ids(&index, &hits), vec!["a", "c", "b"]);
    // the reported score IS the cosine similarity
    assert!((hits[0].score - 1.0).abs() < 1e-5, "{hits:?}");
    let cos_c = 0.9f32 / (0.9f32 * 0.9f32 + 0.1f32 * 0.1f32).sqrt();
    assert!((hits[1].score - cos_c).abs() < 1e-5, "{hits:?}");
    assert!((hits[2].score - 0.0).abs() < 1e-5, "{hits:?}");
}

#[test]
fn vector_search_finds_the_nearest_neighbour() {
    let index = build(4);
    let mut docs = Vec::new();
    for i in 0..50 {
        let ang = i as f32 * 0.1;
        docs.push(doc(
            &format!("d{i}"),
            "text",
            &[ang.cos(), ang.sin(), 0.0, 0.0],
        ));
    }
    index.add_documents(docs).unwrap();
    index.commit();
    let q = [0.0f32, 1.0, 0.0, 0.0];
    let hits = index
        .search_request(&SearchRequest::vector(q.to_vec(), 1), index.pub_stats())
        .unwrap();
    assert_eq!(hits.len(), 1);
    // the angle closest to 90 degrees is i*0.1 = 1.6 rad (i = 16)
    assert_eq!(ids(&index, &hits), vec!["d16"]);
}

#[test]
fn embedding_validation_is_schema_gated() {
    let index = build(3);
    // wrong dimension
    let err = index
        .add_documents(vec![Document::new()
            .text("text", "x")
            .vector("emb", vec![1.0, 2.0])])
        .unwrap_err();
    assert!(err.to_string().contains("expects 3 dimensions"), "{err}");
    // unknown field
    let err = index
        .add_documents(vec![Document::new()
            .text("text", "x")
            .vector("nope", vec![1.0])])
        .unwrap_err();
    assert!(err.to_string().contains("unknown field"), "{err}");
    // a text field cannot carry an embedding
    let err = index
        .add_documents(vec![Document::new()
            .text("text", "x")
            .vector("text", vec![1.0])])
        .unwrap_err();
    assert!(err.to_string().contains("text field"), "{err}");
    // a vector field cannot carry text
    let err = index
        .add_documents(vec![Document::new().text("emb", "x")])
        .unwrap_err();
    assert!(err.to_string().contains("vector field"), "{err}");
    // non-finite values poison every score: rejected
    let err = index
        .add_documents(vec![Document::new()
            .text("text", "x")
            .vector("emb", vec![f32::NAN, 0.0, 0.0])])
        .unwrap_err();
    assert!(err.to_string().contains("non-finite"), "{err}");
}

#[test]
fn query_vector_dimension_must_match_the_field() {
    let index = build(3);
    index
        .add_documents(vec![doc("a", "x", &[1.0, 0.0, 0.0])])
        .unwrap();
    index.commit();
    let req = SearchRequest::vector(vec![1.0, 0.0], 5);
    let err = index.search_request(&req, index.pub_stats()).unwrap_err();
    assert!(err.to_string().contains("expects 3 dimensions"), "{err}");
}

#[test]
fn delete_hides_the_document_from_vector_results() {
    let index = build(2);
    index
        .add_documents(vec![
            doc("a", "alpha", &[1.0, 0.0]),
            doc("b", "beta", &[0.0, 1.0]),
        ])
        .unwrap();
    index.commit();
    assert!(index.delete("a").unwrap());
    let hits = index
        .search_request(&SearchRequest::vector(vec![1.0, 0.0], 5), index.pub_stats())
        .unwrap();
    assert_eq!(ids(&index, &hits), vec!["b"]);
}

#[test]
fn replace_swaps_the_embedding() {
    let index = build(2);
    index
        .add_documents(vec![doc("a", "alpha", &[1.0, 0.0])])
        .unwrap();
    index.commit();
    index
        .add_documents(vec![doc("a", "alpha", &[0.0, 1.0])])
        .unwrap();
    index.commit();
    let hits = index
        .search_request(&SearchRequest::vector(vec![1.0, 0.0], 5), index.pub_stats())
        .unwrap();
    // the old embedding is gone: the query no longer matches a
    assert!(hits.is_empty() || hits[0].score < 0.99, "{hits:?}");
}

#[test]
fn merge_keeps_the_embeddings() {
    let index = build(2);
    // enough docs to force several seals and a merge (2 lanes x 8 segments)
    let mut docs = Vec::new();
    for i in 0..400 {
        docs.push(doc(
            &format!("d{i}"),
            "text",
            &[
                if i == 399 { 1.0 } else { 0.0 },
                if i == 399 { 0.0 } else { 1.0 },
            ],
        ));
    }
    index.add_documents(docs).unwrap();
    index.commit();
    assert!(
        index.num_segments() <= 16,
        "segments {}",
        index.num_segments()
    );
    let hits = index
        .search_request(&SearchRequest::vector(vec![1.0, 0.0], 3), index.pub_stats())
        .unwrap();
    // d399 carries the query vector (cosine 1.0), the others are orthogonal
    assert_eq!(ids(&index, &hits).first().map(|s| s.as_str()), Some("d399"));
    assert!((hits[0].score - 1.0).abs() < 1e-5, "{hits:?}");
}

#[test]
fn hybrid_alpha_switches_between_the_modalities() {
    let index = build(2);
    index
        .add_documents(vec![
            // keyword-best, vector-orthogonal
            doc("kw", "fox", &[0.0, 1.0]),
            // vector-best, weaker keyword match
            doc("vec", "fox dog fox dog", &[1.0, 0.0]),
        ])
        .unwrap();
    index.commit();
    let base = SearchRequest {
        text: Some("fox".into()),
        vector: Some(vec![1.0, 0.0]),
        field: None,
        k: 5,
        mode: SearchMode::Hybrid,
        alpha: 1.0,
        fusion: Fusion::Weighted,
    };
    // alpha=1 -> keyword only
    let hits = index.search_request(&base, index.pub_stats()).unwrap();
    assert_eq!(ids(&index, &hits).first().map(|s| s.as_str()), Some("kw"));
    // alpha=0 -> vector only
    let hits = index
        .search_request(
            &SearchRequest {
                alpha: 0.0,
                ..base.clone()
            },
            index.pub_stats(),
        )
        .unwrap();
    assert_eq!(ids(&index, &hits).first().map(|s| s.as_str()), Some("vec"));
    // both modalities contribute: the fused rows carry both components
    let hits = index
        .search_request(
            &SearchRequest {
                alpha: 0.5,
                ..base.clone()
            },
            index.pub_stats(),
        )
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert!(
        hits.iter().all(|h| h.kw.is_some() && h.vec.is_some()),
        "{hits:?}"
    );
}

#[test]
fn rrf_mode_ranks_by_both_ranks() {
    let index = build(2);
    index
        .add_documents(vec![
            doc("best", "fox", &[1.0, 0.0]),
            doc("other", "fox", &[0.0, 1.0]),
        ])
        .unwrap();
    index.commit();
    let hits = index
        .search_request(
            &SearchRequest {
                text: Some("fox".into()),
                vector: Some(vec![1.0, 0.0]),
                field: None,
                k: 5,
                mode: SearchMode::Hybrid,
                alpha: 0.5,
                fusion: Fusion::Rrf,
            },
            index.pub_stats(),
        )
        .unwrap();
    assert_eq!(ids(&index, &hits).first().map(|s| s.as_str()), Some("best"));
}

#[test]
fn filter_mode_lets_the_keyword_select_and_the_vector_rank() {
    let index = build(2);
    index
        .add_documents(vec![
            doc("a", "fox", &[0.0, 1.0]),
            doc("b", "fox", &[1.0, 0.0]),
            doc("c", "dog", &[1.0, 0.0]),
        ])
        .unwrap();
    index.commit();
    let hits = index
        .search_request(
            &SearchRequest {
                text: Some("fox".into()),
                vector: Some(vec![1.0, 0.0]),
                field: None,
                k: 5,
                mode: SearchMode::Filter,
                alpha: 0.5,
                fusion: Fusion::Weighted,
            },
            index.pub_stats(),
        )
        .unwrap();
    // "c" does not match the keyword -> it is filtered out; the vector ranks
    let got = ids(&index, &hits);
    assert_eq!(got, vec!["b", "a"]);
    // the reported score is the similarity
    assert!((hits[0].score - 1.0).abs() < 1e-5, "{hits:?}");
}

#[test]
fn documents_without_an_embedding_are_skipped() {
    let index = build(2);
    index
        .add_documents(vec![
            Document::new().id("plain").text("text", "no vector here"),
            doc("vec", "with vector", &[1.0, 0.0]),
        ])
        .unwrap();
    index.commit();
    let hits = index
        .search_request(&SearchRequest::vector(vec![1.0, 0.0], 5), index.pub_stats())
        .unwrap();
    assert_eq!(ids(&index, &hits), vec!["vec"]);
}

#[test]
fn disk_roundtrip_keeps_the_embeddings() {
    let dir = std::env::temp_dir().join(format!("zap-vector-disk-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    {
        let index = build_disk(2, dir.clone());
        index
            .add_documents(vec![
                doc("a", "alpha", &[1.0, 0.0]),
                doc("b", "beta", &[0.0, 1.0]),
            ])
            .unwrap();
        index.commit();
    }
    {
        let index = build_disk(2, dir.clone());
        let hits = index
            .search_request(&SearchRequest::vector(vec![1.0, 0.0], 5), index.pub_stats())
            .unwrap();
        // "a" carries the query vector (cosine 1.0); "b" is orthogonal
        assert_eq!(ids(&index, &hits).first().map(|s| s.as_str()), Some("a"));
        assert!((hits[0].score - 1.0).abs() < 1e-5, "{hits:?}");
        // the keyword side still works after the reboot
        let hits = index.search("alpha", 5).unwrap();
        assert_eq!(hits.len(), 1);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// the engine's vector search must match a brute-force reference exactly
/// (unordered comparison: top-k ties are not stable across builds)
#[test]
fn vector_search_matches_brute_force() {
    // deterministic pseudo-random corpus
    let mut state = 0x2545F491_4F6CDD1Du64;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let dim = 8usize;
    let n = 200usize;
    let index = build(dim as u32);
    let mut docs = Vec::new();
    let mut corpus: Vec<Vec<f32>> = Vec::new();
    for i in 0..n {
        let v: Vec<f32> = (0..dim)
            .map(|_| ((rnd() % 1000) as f32 / 1000.0) - 0.5)
            .collect();
        corpus.push(v.clone());
        docs.push(doc(&format!("d{i}"), "text", &v));
    }
    index.add_documents(docs).unwrap();
    index.commit();

    for _ in 0..20 {
        let q: Vec<f32> = (0..dim)
            .map(|_| ((rnd() % 1000) as f32 / 1000.0) - 0.5)
            .collect();
        let qn = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut scored: Vec<(usize, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let dot: f32 = v.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
                let vn = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                (i, dot / (vn * qn))
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let want: Vec<String> = scored
            .iter()
            .take(10)
            .map(|(i, _)| format!("d{i}"))
            .collect();

        let hits = index
            .search_request(&SearchRequest::vector(q.clone(), 10), index.pub_stats())
            .unwrap();
        let got = ids(&index, &hits);
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        // same set (order may differ on ties)
        let mut g = got.clone();
        let mut w = want.clone();
        g.sort();
        w.sort();
        assert_eq!(g, w, "{got:?} vs {want:?}");
    }
}
