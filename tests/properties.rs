//! property tests:
//! T3: posting skip-anchor/seek/position invariants
//! T1/T2: block-max WAND streaming == dense evaluation top-k equivalence

use std::sync::Arc;

use proptest::prelude::*;

use zap::analyzer::{Analyzer, AnalyzerConf};
use zap::index::Index;
use zap::posting::PostingData;
use zap::query::parse;
use zap::schema::{Document, Schema};
use zap::search::exec::{eval_allowed, eval_term_stream, Ctx, Stats};
use zap::search::{Bm25Params, DocAddr, SearchHit, TopK};

const WORDS: [&str; 24] = [
    "fox", "quick", "brown", "jumps", "lazy", "dog", "forest", "river", "mountain", "stone",
    "winter", "summer", "night", "day", "sun", "cloud", "wind", "rain", "field", "road", "tree",
    "leaf", "root", "grass",
];

// ---------- T3: posting skip-anchor invariants ----------

// random posting data: ascending docids (with small gaps), small tfs, sparse positions
fn post_data_strategy2() -> impl Strategy<Value = (Vec<u32>, Vec<u32>)> {
    (0usize..64).prop_flat_map(|len| {
        (
            proptest::collection::vec(1u32..400, len),
            proptest::collection::vec(1u32..4, len),
        )
            .prop_map(|(gaps, tfs)| {
                let mut docids = Vec::with_capacity(gaps.len());
                let mut acc = 0u32;
                for g in gaps {
                    acc += g;
                    docids.push(acc);
                }
                (docids, tfs)
            })
    })
}

proptest! {
    #[test]
    fn seek_matches_first_ge(
        (docids, tfs) in post_data_strategy2(),
        target in 0u32..100_000,
    ) {
        prop_assume!(!docids.is_empty());
        let mut data = PostingData::default();
        for (pos, (&d, &tf)) in docids.iter().zip(&tfs).enumerate() {
            data.add(d, tf, vec![pos as u32; tf as usize]);
        }
        let mut blob = Vec::new();
        let info = data.freeze(None).encode_to(&mut blob);
        let mut r = zap::posting::PostingReader::new(&info, &blob);

        let expected = docids.iter().copied().find(|d| *d >= target);
        r.seek_to(target);
        assert_eq!(r.peek().map(|(d, _)| d), expected, "seek_to({target})");
        // repeated seek to the same target keeps the state
        r.seek_to(target);
        assert_eq!(r.peek().map(|(d, _)| d), expected, "repeated seek_to({target})");
    }

    #[test]
    fn advanced_positions_match_source(
        (docids, tfs) in post_data_strategy2(),
        advance_n in 0usize..64,
    ) {
        prop_assume!(!docids.is_empty());
        let src_positions: Vec<Vec<u32>> = docids
            .iter()
            .zip(tfs.iter())
            .enumerate()
            .map(|(i, (_, tf))| vec![i as u32; *tf as usize])
            .collect();
        let mut data = PostingData::default();
        for (i, (d, tf)) in docids.iter().zip(tfs.iter()).enumerate() {
            data.add(*d, *tf, src_positions[i].clone());
        }
        let mut blob = Vec::new();
        let info = data.freeze(None).encode_to(&mut blob);
        let mut r = zap::posting::PostingReader::new(&info, &blob);
        let n = advance_n.min(docids.len());
        for _ in 0..n {
            r.advance();
        }
        let idx = n;
        if idx < src_positions.len() {
            let mut buf = Vec::new();
            // prime the peek (fresh readers have nothing decoded yet)
            assert_eq!(r.peek().map(|(d, _)| d), Some(docids[idx]));
            assert!(r.peek_positions_into(&mut buf));
            assert_eq!(buf, src_positions[idx], "positions at idx {idx}");
        } else {
            // exhausted reader
            let mut buf = Vec::new();
            assert!(!r.peek_positions_into(&mut buf));
        }
    }
}

// ---------- T1/T2 corpus builders ----------

fn build_index(docs: &[String]) -> (Arc<Index>, Stats, Arc<zap::segment::SegmentReader>) {
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    let analyzer = Analyzer::new(AnalyzerConf::default());
    let index = Index::with_lanes(schema, analyzer, 1.2, 0.75, 2);
    let owned: Vec<Document> = docs
        .iter()
        .map(|d| Document::new().text("text", d.clone()))
        .collect();
    index.add_documents(owned).unwrap();
    index.commit();
    let seg = index.segments_snapshot()[0].clone();
    // stats rebuilt the same way index.publish_snapshot does
    let mut stats = Stats::empty();
    let mut df: std::collections::HashMap<u32, std::collections::HashMap<String, u32>> =
        std::collections::HashMap::new();
    let mut total: u64 = 0;
    let mut sums: Vec<u64> = vec![0; 1];
    for s in &index.segments_snapshot() {
        total += s.num_docs as u64;
        sums[0] += s.sum_field_len(0);
        for (term, info) in s.stream_terms(0) {
            *df.entry(0u32).or_default().entry(term).or_insert(0) += info.df;
        }
    }
    stats.total_docs = total as u32;
    stats.avg_len = vec![sums[0] as f32 / total.max(1) as f32];
    stats.df = std::sync::Arc::new(df);
    (index, stats, seg)
}

fn ctx_of<'a>(index: &'a Arc<Index>, stats: &'a Stats) -> Ctx<'a> {
    Ctx {
        schema: index.schema(),
        stats,
        params: &Bm25Params { k1: 1.2, b: 0.75 },
        analyzer: index.analyzer(),
        threshold: &(|| 0.0),
        stems: &EMPTY_STEMS,
    }
}

fn ctx_prune_of<'a>(index: &'a Arc<Index>, stats: &'a Stats, thr: &'a dyn Fn() -> f32) -> Ctx<'a> {
    Ctx {
        schema: index.schema(),
        stats,
        params: &Bm25Params { k1: 1.2, b: 0.75 },
        analyzer: index.analyzer(),
        threshold: thr,
        stems: &EMPTY_STEMS,
    }
}

static EMPTY_STEMS: std::sync::LazyLock<std::collections::HashMap<String, [Option<Arc<str>>; 2]>> =
    std::sync::LazyLock::new(std::collections::HashMap::new);

/// (doc text, score bits, docid)
fn dense_topk(
    node: &zap::query::Node,
    seg: &zap::segment::SegmentReader,
    ctx: &Ctx,
    k: usize,
) -> Vec<SearchHit> {
    let ds = eval_allowed(node, None, seg, ctx, None);
    let mut col = TopK::new(k);
    for (i, doc) in ds.ids.iter().enumerate() {
        col.push(
            DocAddr {
                segment: 0,
                docid: *doc,
            },
            ds.scores[i],
        );
    }
    col.into_sorted()
}

fn corpus_strategy() -> impl Strategy<Value = (Vec<String>, String, usize)> {
    (2usize..20, 1usize..6, 1usize..12).prop_flat_map(|(ndocs, hot_every, k)| {
        proptest::collection::vec(0u64..u64::MAX, ndocs * 8).prop_map(move |seeds| {
            let mut docs = Vec::with_capacity(ndocs);
            for i in 0..ndocs {
                let mut s = String::new();
                if i % hot_every.max(1) == 0 {
                    // hot outliers with repeated terms
                    s.push_str("fox fox ");
                }
                for w in 0..8 {
                    let idx = ((seeds[i * 8 + w] >> (w * 7)) as usize) % WORDS.len();
                    let word = if w == 0 { "fox" } else { WORDS[idx] };
                    s.push_str(word);
                    s.push(' ');
                }
                docs.push(s);
            }
            let query = WORDS[(seeds[ndocs] as usize) % WORDS.len()].to_string();
            (docs, query, k)
        })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// T1/T2: streaming WAND walk must produce the identical top-k as dense eval
    #[test]
    fn wand_matches_dense_topk(docs in corpus_strategy()) {
        let (docs, query, _k) = docs;
        let (index, stats, seg) = build_index(&docs);
        let node = parse(&query).unwrap();
        prop_assume!(matches!(node, zap::query::Node::Term(_)));
        let ctx = ctx_of(&index, &stats);
        for k in [1usize, 3, 10] {
            let dense = dense_topk(&node, &seg, &ctx, k);
            let mut wcol = TopK::new(k);
            eval_term_stream(&node, None, &seg, &ctx, None, &mut wcol);
            let wand = wcol.into_sorted();
            assert_eq!(dense.len(), wand.len(), "k={k} hit counts differ");
            for (dh, wh) in dense.iter().zip(wand.iter()) {
                assert_eq!(dh.addr, wh.addr, "k={k} dense vs wand");
                assert!((dh.score - wh.score).abs() < 1e-4, "k={k} scores differ");
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// T1: with a live threshold (the dense k-th score), WAND pruning keeps the same top-k
    #[test]
    fn wand_threshold_prunes_to_same_topk(docs in corpus_strategy()) {
        let (docs, query, _k) = docs;
        let (index, stats, seg) = build_index(&docs);
        let node = parse(&query).unwrap();
        prop_assume!(matches!(node, zap::query::Node::Term(_)));
        let ctx = ctx_of(&index, &stats);
        for kk in [1usize, 3, 10] {
            let dense = dense_topk(&node, &seg, &ctx, kk);
            let Some(kth) = dense.get(kk - 1) else { continue; };
            let thr_val = kth.score;
            let thr = move || thr_val;
            let ctx_prune = ctx_prune_of(&index, &stats, &thr);
            let mut wcol = TopK::new(kk);
            eval_term_stream(&node, None, &seg, &ctx_prune, None, &mut wcol);
            let wand = wcol.into_sorted();
            let d_res = dense.iter().map(|h| h.addr).collect::<Vec<_>>();
            let w_res = wand.iter().map(|h| h.addr).collect::<Vec<_>>();
            if d_res != w_res {
                eprintln!("DIVG kk={kk} dense={d_res:?} wand={w_res:?}");
                eprintln!("docs={docs:?} query={query:?}");
                eprintln!("scores={:?}", dense.iter().map(|h| (h.addr.docid, h.score)).collect::<Vec<_>>());
                panic!("kk={kk} wand-pruned topk diverged");
            }
            assert_eq!(
                dense.iter().map(|h| h.addr).collect::<Vec<_>>(),
                wand.iter().map(|h| h.addr).collect::<Vec<_>>(),
                "kk={kk} wand-pruned topk diverged"
            );
            for (dh, wh) in dense.iter().zip(wand.iter()) {
                assert!((dh.score - wh.score).abs() < 1e-4, "kk={kk} scores differ");
            }
        }
    }
}

// ---------- T4: merge invariants ----------

/// the corpus used to force merges; queries are single terms drawn from the vocab
fn merge_corpus_strategy() -> impl Strategy<Value = (Vec<String>, Vec<String>)> {
    (2usize..40, 2usize..30)
        .prop_flat_map(|(ndocs, nqueries)| {
            (
                proptest::collection::vec(0u64..u64::MAX, ndocs * 8),
                proptest::collection::vec(0usize..WORDS.len(), nqueries),
            )
        })
        .prop_map(|(seeds, qidx)| {
            let mut docs = Vec::with_capacity(seeds.len() / 8);
            for i in 0..seeds.len() / 8 {
                let mut s = String::new();
                for w in 0..8 {
                    if w == 0 {
                        // always one shared term so queries hit many docs
                        s.push_str("hop ");
                        continue;
                    }
                    let word = WORDS[((seeds[i * 8 + w] >> (w * 7)) as usize) % WORDS.len()];
                    s.push_str(word);
                    s.push(' ');
                }
                docs.push(s);
            }
            let queries = qidx
                .iter()
                .map(|i| WORDS[*i].to_string())
                .collect::<Vec<_>>();
            (docs, queries)
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// T4: a lane-parallel, merge-heavy build must agree with a single-lane
    /// reference build on ranked top-k output (texts + scores)
    #[test]
    fn merged_index_matches_single_lane_reference(
        (docs, queries) in merge_corpus_strategy(),
        k in 1usize..8,
        lane in 1usize..8usize,
    ) {
        prop_assume!(!docs.is_empty());
        // reference: one lane, one commit
        let mut schema = Schema::default();
        schema.add_text("text", 1.0);
        let analyzer = Analyzer::new(AnalyzerConf::default());
        let ref_index = Index::with_lanes(schema, analyzer, 1.2, 0.75, 1);
        let ref_docs: Vec<Document> = docs
            .iter()
            .map(|d| Document::new().text("text", d.clone()))
            .collect();
        ref_index
            .add_documents(ref_docs)
            .unwrap();
        ref_index.commit();
        let ref_hits: Vec<Vec<(String, String, u32)>> = queries
            .iter()
            .map(|q| {
                ref_index
                    .search(q, k)
                    .unwrap()
                    .into_iter()
                    .map(|h| {
                        let fields = ref_index.doc_payload(h.addr).unwrap();
                        (fields.fields[0].1.clone(), h.score.to_bits().to_string(), h.addr.docid)
                    })
                    .collect()
            })
            .collect();

        // multi-lane, multi-commit: forces merges
        let mut schema = Schema::default();
        schema.add_text("text", 1.0);
        let analyzer = Analyzer::new(AnalyzerConf::default());
        let merged_index = Index::with_lanes(schema, analyzer, 1.2, 0.75, lane);
        let chunk = docs.len().div_ceil(lane);
        let mut i = 0;
        while i < docs.len() {
            let end = (i + chunk).min(docs.len());
            let chunk_docs: Vec<Document> = docs[i..end]
                .iter()
                .map(|d| Document::new().text("text", d.clone()))
                .collect();
            merged_index.add_documents(chunk_docs).unwrap();
            merged_index.commit();
            i = end;
        }
        // tie ordering can differ between builds; compare as an unordered
        // multiset of (score, text) pairs at the same size
        for (qi, q) in queries.iter().enumerate() {
            let hits = merged_index.search(q, k).unwrap();
            let mut got: Vec<(String, String)> = hits
                .iter()
                .map(|h| {
                    let fields = merged_index.doc_payload(h.addr).unwrap();
                    (fields.fields[0].1.clone(), h.score.to_bits().to_string())
                })
                .collect();
            let mut refs: Vec<(String, String)> = ref_hits[qi]
                .iter()
                .map(|(t, s, _)| (t.clone(), s.clone()))
                .collect();
            assert_eq!(refs.len(), got.len(), "query {q} hit counts differ");
            if refs.len() < k {
                // fewer hits than k: no threshold ties were cut: exact match
                got.sort();
                refs.sort();
                assert_eq!(refs, got, "query {q}: (text, score) multiset differs");
            } else {
                // ties at the cut: entries above the min score must match exactly;
                // the min-level ties may pick different (equally-scored) docs
                let min_score = got
                    .iter()
                    .map(|(_, s)| s.clone())
                    .min()
                    .expect("nonempty");
                let ref_min = refs.iter().map(|(_, s)| s.clone()).min().expect("nonempty");
                assert_eq!(ref_min, min_score, "query {q}: min score differs");
                let got_above: Vec<(String, String)> = got
                    .iter()
                    .filter(|(_, s)| *s != min_score)
                    .cloned()
                    .collect();
                let ref_above: Vec<(String, String)> = refs
                    .iter()
                    .filter(|(_, s)| *s != min_score)
                    .cloned()
                    .collect();
                assert_eq!(ref_above, got_above, "query {q}: above-tie docs differ");
            }
        }
    }
}
