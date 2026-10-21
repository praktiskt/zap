//! vector search core: embedding columns, similarity metrics, flat scan.
//!
//! Dependency-free by design (no `crate::` types): the engine wraps the
//! segment-local hits into its own addresses, so this module can be lifted
//! into its own crate later without touching the engine. `VectorIndex` is the
//! seam an HNSW implementation plugs into later; `FlatVectorIndex` is the
//! reference (exact, brute-force) implementation.

pub mod kernels;

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use roaring::RoaringBitmap;

/// similarity metric for a vector field
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    /// cosine similarity: dot / (|a| * |b|) — the default for embeddings
    Cosine,
    /// raw dot product
    Dot,
    /// negative squared euclidean distance (higher score = closer)
    L2,
}

impl Metric {
    /// parse a metric name ("cosine" | "dot" | "l2"); None = unknown
    pub fn parse(s: &str) -> Option<Metric> {
        match s.to_ascii_lowercase().as_str() {
            "cosine" | "cos" => Some(Metric::Cosine),
            "dot" | "inner" => Some(Metric::Dot),
            "l2" | "euclidean" => Some(Metric::L2),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::Dot => "dot",
            Metric::L2 => "l2",
        }
    }
}

/// one vector field's column inside a segment.
///
/// Rows exist only for documents that actually carry the field (a doc may
/// omit it), so the column is sparse: `docids` are ascending and index the
/// row-major `data` (row r = `data[r*dim .. (r+1)*dim]`).
#[derive(Debug, Clone, Default)]
pub struct VectorColumn {
    pub dim: u32,
    /// docids that carry a vector (ascending)
    pub docids: Box<[u32]>,
    /// row-major embedding data: `docids.len() * dim` f32
    pub data: Box<[f32]>,
    /// per-row L2 norm (cosine metric)
    pub norms: Box<[f32]>,
}

impl VectorColumn {
    pub fn is_empty(&self) -> bool {
        self.docids.is_empty()
    }

    /// number of rows (docs carrying the field)
    pub fn rows(&self) -> usize {
        self.docids.len()
    }

    /// row index of a docid (binary search over the ascending docids)
    pub fn row_of(&self, docid: u32) -> Option<usize> {
        self.docids.binary_search(&docid).ok()
    }

    /// embedding of one doc
    pub fn vector_of(&self, docid: u32) -> Option<&[f32]> {
        let r = self.row_of(docid)?;
        let d = self.dim as usize;
        Some(&self.data[r * d..(r + 1) * d])
    }
}

/// ingest-side accumulator for one vector field
#[derive(Debug, Clone, Default)]
pub struct VectorBuilder {
    pub dim: u32,
    pub docids: Vec<u32>,
    pub data: Vec<f32>,
    pub norms: Vec<f32>,
}

impl VectorBuilder {
    pub fn new(dim: u32) -> Self {
        Self {
            dim,
            docids: Vec::new(),
            data: Vec::new(),
            norms: Vec::new(),
        }
    }

    /// append one row (the caller validated the dimension)
    pub fn push(&mut self, docid: u32, v: &[f32]) {
        self.docids.push(docid);
        self.data.extend_from_slice(v);
        self.norms.push(kernels::norm(v));
    }

    pub fn freeze(self) -> VectorColumn {
        VectorColumn {
            dim: self.dim,
            docids: self.docids.into_boxed_slice(),
            data: self.data.into_boxed_slice(),
            norms: self.norms.into_boxed_slice(),
        }
    }
}

/// one nearest-neighbour hit: segment-local docid + similarity score
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorHit {
    pub docid: u32,
    pub score: f32,
}

/// a searchable embedding collection.
///
/// `FlatVectorIndex` scans every row (exact, O(rows*dim)); a future HNSW
/// implementation implements the same trait, so the engine's candidate
/// collection does not change.
pub trait VectorIndex {
    /// top-`k` hits by descending score
    fn search(&self, query: &[f32], k: usize) -> Vec<VectorHit>;
}

/// exact brute-force scan over one column.
///
/// `allowed` masks tombstoned docs (the segment's complement bitmap);
/// `restrict` narrows the scan to an ascending docid list (keyword-filter
/// mode). Both are optional: no mask and no restriction = scan everything.
pub struct FlatVectorIndex<'a> {
    column: &'a VectorColumn,
    metric: Metric,
    allowed: Option<&'a RoaringBitmap>,
    restrict: Option<&'a [u32]>,
}

impl<'a> FlatVectorIndex<'a> {
    pub fn new(column: &'a VectorColumn, metric: Metric) -> Self {
        Self {
            column,
            metric,
            allowed: None,
            restrict: None,
        }
    }

    /// live-doc mask (tombstone complement); None = every row is live
    pub fn with_allowed(mut self, allowed: Option<&'a RoaringBitmap>) -> Self {
        self.allowed = allowed;
        self
    }

    /// restrict the scan to these docids (ascending)
    pub fn with_restrict(mut self, restrict: Option<&'a [u32]>) -> Self {
        self.restrict = restrict;
        self
    }

    /// scan one row range `[from, to)` of the column
    pub fn search_rows(&self, query: &[f32], k: usize, from: usize, to: usize) -> Vec<VectorHit> {
        let col = self.column;
        let dim = col.dim as usize;
        if query.len() != dim || dim == 0 {
            return Vec::new();
        }
        let qnorm = match self.metric {
            Metric::Cosine => kernels::norm(query),
            _ => 1.0,
        };
        let mut top = TopHits::new(k);
        // the restrict list ascends with the column's docids -> one cursor
        let mut rp = 0usize;
        let restrict = self.restrict.unwrap_or(&[]);
        let restricted = self.restrict.is_some();
        let to = to.min(col.rows());
        for r in from..to {
            let docid = col.docids[r];
            if restricted {
                while rp < restrict.len() && restrict[rp] < docid {
                    rp += 1;
                }
                if rp >= restrict.len() || restrict[rp] != docid {
                    continue;
                }
            }
            if let Some(al) = self.allowed {
                if !al.contains(docid) {
                    continue;
                }
            }
            let row = &col.data[r * dim..(r + 1) * dim];
            let score = match self.metric {
                Metric::Cosine => {
                    let dn = col.norms[r];
                    if dn <= 0.0 || qnorm <= 0.0 {
                        // degenerate (zero) embedding: no meaningful direction
                        continue;
                    }
                    kernels::dot(query, row) / (qnorm * dn)
                }
                Metric::Dot => kernels::dot(query, row),
                Metric::L2 => -kernels::sq_dist(query, row),
            };
            top.push(docid, score);
        }
        top.into_sorted()
    }
}

impl VectorIndex for FlatVectorIndex<'_> {
    fn search(&self, query: &[f32], k: usize) -> Vec<VectorHit> {
        self.search_rows(query, k, 0, self.column.rows())
    }
}

/// bounded top-k over (score, docid): binary min-heap, ties by smaller docid
struct TopHits {
    heap: BinaryHeap<Reverse<HitKey>>,
    k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HitKey {
    /// total-order f32 bits (score < 0 flips the sign bit)
    score_bits: u32,
    docid: u32,
}

impl HitKey {
    fn new(score: f32, docid: u32) -> Self {
        let bits = if score < 0.0 {
            !score.to_bits()
        } else {
            score.to_bits() ^ 0x8000_0000
        };
        Self {
            score_bits: bits,
            docid,
        }
    }

    fn score(&self) -> f32 {
        f32::from_bits(if self.score_bits < 0x8000_0000 {
            !self.score_bits
        } else {
            self.score_bits ^ 0x8000_0000
        })
    }
}

impl Ord for HitKey {
    fn cmp(&self, other: &Self) -> Ordering {
        // ASCENDING by score: the BinaryHeap<Reverse<HitKey>> is then a
        // MIN-heap over the score, so `peek()` is the worst kept hit (the one
        // to evict). Ties break towards the LARGER docid, so on equal scores
        // the smaller docid survives (same tie rule as the engine's TopK).
        self.score_bits
            .cmp(&other.score_bits)
            .then_with(|| other.docid.cmp(&self.docid))
    }
}

impl PartialOrd for HitKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

use std::cmp::Reverse;

impl TopHits {
    fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k.min(1024) + 1),
            k,
        }
    }

    fn push(&mut self, docid: u32, score: f32) {
        if self.k == 0 {
            return;
        }
        // -0.0 must order like +0.0: the bit trick below maps negative floats
        // below positive ones, and -0.0 would fall on the wrong side of it
        let score = if score == 0.0 { 0.0 } else { score };
        let key = HitKey::new(score, docid);
        if self.heap.len() < self.k {
            self.heap.push(Reverse(key));
            return;
        }
        let Some(Reverse(min)) = self.heap.peek() else {
            return;
        };
        if key.score_bits > min.score_bits
            || (key.score_bits == min.score_bits && docid < min.docid)
        {
            self.heap.pop();
            self.heap.push(Reverse(key));
        }
    }

    fn into_sorted(mut self) -> Vec<VectorHit> {
        let mut hits: Vec<VectorHit> = self
            .heap
            .drain()
            .map(|Reverse(h)| VectorHit {
                docid: h.docid,
                score: h.score(),
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.docid.cmp(&b.docid))
        });
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(rows: &[(u32, &[f32])]) -> VectorColumn {
        let mut b = VectorBuilder::new(rows[0].1.len() as u32);
        for (id, v) in rows {
            b.push(*id, v);
        }
        b.freeze()
    }

    #[test]
    fn cosine_ranks_by_angle() {
        let c = col(&[(0, &[1.0, 0.0]), (1, &[0.9, 0.1]), (2, &[0.0, 1.0])]);
        let idx = FlatVectorIndex::new(&c, Metric::Cosine);
        let hits = idx.search(&[1.0, 0.0], 3);
        assert_eq!(hits[0].docid, 0);
        assert_eq!(hits[1].docid, 1);
        assert_eq!(hits[2].docid, 2);
        assert!((hits[0].score - 1.0).abs() < 1e-6);
        assert!(hits[0].score > hits[1].score && hits[1].score > hits[2].score);
    }

    #[test]
    fn restrict_narrows_the_scan() {
        let c = col(&[(0, &[1.0, 0.0]), (1, &[0.0, 1.0]), (2, &[1.0, 1.0])]);
        let idx = FlatVectorIndex::new(&c, Metric::Dot).with_restrict(Some(&[2u32, 5]));
        let hits = idx.search(&[1.0, 0.0], 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].docid, 2);
    }

    #[test]
    fn l2_scores_inverse_distance() {
        let c = col(&[(0, &[0.0, 0.0]), (1, &[3.0, 4.0])]);
        let idx = FlatVectorIndex::new(&c, Metric::L2);
        let hits = idx.search(&[0.0, 0.0], 2);
        assert_eq!(hits[0].docid, 0);
        assert_eq!(hits[1].docid, 1);
        assert!((hits[0].score - 0.0).abs() < 1e-6);
        assert!((hits[1].score + 25.0).abs() < 1e-6);
    }

    #[test]
    fn dim_mismatch_yields_nothing() {
        let c = col(&[(0, &[1.0, 0.0])]);
        let idx = FlatVectorIndex::new(&c, Metric::Cosine);
        assert!(idx.search(&[1.0, 0.0, 0.0], 5).is_empty());
    }
}
