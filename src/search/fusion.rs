//! hybrid fusion: combine keyword (BM25) and vector (similarity) scores.
//!
//! The engine and the cluster coordinator run the SAME fusion over the same
//! candidate shape, so a hybrid search ranks identically whether it is
//! answered by one node or by a ring of shards.
//!
//! Candidates carry the two modalities separately (`kw` / `vec`); the fusion
//! decides the final score:
//! - `Weighted`: min-max normalize each modality over the candidate set and
//!   mix them with `alpha` (0 = embedding only, 1 = keyword only)
//! - `Rrf`: reciprocal rank fusion — scale-free, no normalization needed
//!
//! A candidate that only one modality produced contributes 0 for the other
//! one (weighted) or just its own rank term (rrf).

use std::collections::HashMap;

use crate::search::{DocAddr, Fusion, SearchHit, SearchMode};

/// one candidate in a fused result set
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub addr: DocAddr,
    /// external id, when the row carries one. Cluster rows do (a replica
    /// copy of a document shares the id, so replicas fuse into one row);
    /// local candidates do not and fall back to the segment address.
    pub id: Option<String>,
    /// BM25 component (None = the document did not match the keyword query)
    pub kw: Option<f32>,
    /// similarity component (None = the document carries no embedding)
    pub vec: Option<f32>,
}

impl Candidate {
    pub fn new(addr: DocAddr) -> Self {
        Self {
            addr,
            id: None,
            kw: None,
            vec: None,
        }
    }
}

/// RRF damping constant (the usual default)
pub const RRF_K: f32 = 60.0;

/// identity of a candidate: the external id when known, else the address
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    Addr(DocAddr),
    Id(String),
}

/// fuse candidates into a ranked top-k
pub fn fuse(
    cands: Vec<Candidate>,
    mode: SearchMode,
    alpha: f32,
    fusion: Fusion,
    k: usize,
) -> Vec<SearchHit> {
    // 1. group by identity (a document replicated on several shards is one
    //    candidate; keep the best score of each modality)
    let mut merged: Vec<Candidate> = Vec::with_capacity(cands.len());
    let mut index: HashMap<Key, usize> = HashMap::with_capacity(cands.len());
    for c in cands {
        let key = match &c.id {
            Some(id) => Key::Id(id.clone()),
            None => Key::Addr(c.addr),
        };
        match index.get(&key) {
            Some(&i) => {
                let m = &mut merged[i];
                if let Some(v) = c.kw {
                    m.kw = Some(m.kw.unwrap_or(f32::NEG_INFINITY).max(v));
                }
                if let Some(v) = c.vec {
                    m.vec = Some(m.vec.unwrap_or(f32::NEG_INFINITY).max(v));
                }
            }
            None => {
                index.insert(key, merged.len());
                merged.push(c);
            }
        }
    }
    // 2. score
    let alpha = alpha.clamp(0.0, 1.0);
    let mut hits: Vec<SearchHit> = match mode {
        SearchMode::Keyword => merged
            .iter()
            .filter_map(|c| c.kw.map(|s| hit(c, s)))
            .collect(),
        SearchMode::Vector | SearchMode::Filter => merged
            .iter()
            .filter_map(|c| c.vec.map(|s| hit(c, s)))
            .collect(),
        SearchMode::Hybrid => match fusion {
            Fusion::Weighted => weighted(&merged, alpha),
            Fusion::Rrf => rrf(&merged),
        },
    };
    // 3. rank + truncate (ties by address: stable across builds)
    hits.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.addr.cmp(&b.addr))
    });
    hits.truncate(k);
    hits
}

fn hit(c: &Candidate, score: f32) -> SearchHit {
    SearchHit {
        addr: c.addr,
        score,
        kw: c.kw,
        vec: c.vec,
    }
}

/// weighted fusion: alpha * normalized(BM25) + (1 - alpha) * normalized(sim)
fn weighted(cands: &[Candidate], alpha: f32) -> Vec<SearchHit> {
    let kw_range = range(cands.iter().filter_map(|c| c.kw));
    let vec_range = range(cands.iter().filter_map(|c| c.vec));
    cands
        .iter()
        .map(|c| {
            let kw_n = c.kw.map(|s| normalize(s, kw_range)).unwrap_or(0.0);
            let vec_n = c.vec.map(|s| normalize(s, vec_range)).unwrap_or(0.0);
            hit(c, alpha * kw_n + (1.0 - alpha) * vec_n)
        })
        .collect()
}

/// reciprocal rank fusion: sum of 1/(K + rank) over the modalities a
/// candidate appears in
fn rrf(cands: &[Candidate]) -> Vec<SearchHit> {
    let mut scores = vec![0.0f32; cands.len()];
    // keyword ranks
    let mut kw_order: Vec<usize> = cands
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.kw.map(|_| i))
        .collect();
    kw_order.sort_by(|&a, &b| {
        let sa = cands[a].kw.expect("filtered");
        let sb = cands[b].kw.expect("filtered");
        sb.total_cmp(&sa)
            .then_with(|| identity_cmp(&cands[a], &cands[b]))
    });
    for (rank, &i) in kw_order.iter().enumerate() {
        scores[i] += 1.0 / (RRF_K + rank as f32);
    }
    // similarity ranks
    let mut vec_order: Vec<usize> = cands
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.vec.map(|_| i))
        .collect();
    vec_order.sort_by(|&a, &b| {
        let sa = cands[a].vec.expect("filtered");
        let sb = cands[b].vec.expect("filtered");
        sb.total_cmp(&sa)
            .then_with(|| identity_cmp(&cands[a], &cands[b]))
    });
    for (rank, &i) in vec_order.iter().enumerate() {
        scores[i] += 1.0 / (RRF_K + rank as f32);
    }
    cands.iter().zip(scores).map(|(c, s)| hit(c, s)).collect()
}

/// min/max of a score series
fn range(scores: impl Iterator<Item = f32>) -> Option<(f32, f32)> {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut any = false;
    for s in scores {
        any = true;
        if s < min {
            min = s;
        }
        if s > max {
            max = s;
        }
    }
    any.then_some((min, max))
}

/// min-max normalize; a flat series carries no information -> 0
fn normalize(s: f32, r: Option<(f32, f32)>) -> f32 {
    let Some((min, max)) = r else {
        return 0.0;
    };
    let span = max - min;
    if span <= f32::EPSILON {
        0.0
    } else {
        (s - min) / span
    }
}

/// deterministic identity order for equal scores
fn identity_cmp(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    a.id.cmp(&b.id).then_with(|| a.addr.cmp(&b.addr))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(seg: u32, doc: u32) -> DocAddr {
        DocAddr {
            segment: seg,
            docid: doc,
        }
    }

    fn cand(seg: u32, doc: u32, kw: Option<f32>, vec: Option<f32>) -> Candidate {
        Candidate {
            addr: addr(seg, doc),
            id: None,
            kw,
            vec,
        }
    }

    #[test]
    fn weighted_mixes_the_two_modalities() {
        let cands = vec![
            cand(0, 0, Some(10.0), Some(0.0)), // best keyword, worst vector
            cand(0, 1, Some(0.0), Some(1.0)),  // worst keyword, best vector
            cand(0, 2, Some(5.0), Some(0.9)),  // good on both
        ];
        let hits = fuse(cands.clone(), SearchMode::Hybrid, 1.0, Fusion::Weighted, 10);
        assert_eq!(hits[0].addr.docid, 0); // alpha=1 -> keyword order
        let hits = fuse(cands.clone(), SearchMode::Hybrid, 0.0, Fusion::Weighted, 10);
        assert_eq!(hits[0].addr.docid, 1); // alpha=0 -> vector order
        let hits = fuse(cands, SearchMode::Hybrid, 0.5, Fusion::Weighted, 10);
        assert_eq!(hits[0].addr.docid, 2); // the middle doc wins the mix
    }

    #[test]
    fn rrf_ranks_by_both_ranks() {
        let cands = vec![
            cand(0, 0, Some(5.0), Some(0.2)),
            cand(0, 1, Some(10.0), Some(1.0)), // best on both
            cand(0, 2, Some(1.0), Some(0.5)),
        ];
        let hits = fuse(cands, SearchMode::Hybrid, 0.5, Fusion::Rrf, 10);
        assert_eq!(hits[0].addr.docid, 1);
    }

    #[test]
    fn vector_mode_reports_the_raw_similarity() {
        let cands = vec![
            cand(0, 0, Some(10.0), Some(0.25)),
            cand(0, 1, None, Some(0.75)),
        ];
        let hits = fuse(cands, SearchMode::Vector, 0.5, Fusion::Weighted, 10);
        assert_eq!(hits[0].addr.docid, 1);
        assert!((hits[0].score - 0.75).abs() < 1e-6);
    }

    #[test]
    fn same_id_fuses_into_one_row() {
        let mut a = cand(0, 0, Some(1.0), Some(0.2));
        a.id = Some("doc-1".into());
        let mut b = cand(1, 7, Some(2.0), Some(0.4));
        b.id = Some("doc-1".into());
        let hits = fuse(vec![a, b], SearchMode::Hybrid, 0.5, Fusion::Weighted, 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].kw, Some(2.0));
        assert_eq!(hits[0].vec, Some(0.4));
    }
}
