//! scoring combinators: doc-materialized scores (BM25), boolean vector merges, phrase windows

pub mod exec;
pub mod fusion;
pub mod phrase;
pub mod term;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DocAddr {
    pub segment: u32,
    pub docid: u32,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub addr: DocAddr,
    /// the score the ranking used (fused score in hybrid mode)
    pub score: f32,
    /// BM25 component (None = the document did not match the keyword query)
    pub kw: Option<f32>,
    /// similarity component (None = the document carries no embedding)
    pub vec: Option<f32>,
}

/// which modalities a search uses
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// BM25 only
    Keyword,
    /// embedding similarity only
    Vector,
    /// BM25 + embedding, fused into one score
    Hybrid,
    /// BM25 selects the candidates, the embedding ranks them
    Filter,
}

/// how the two modalities are combined in `Hybrid` mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fusion {
    /// alpha * normalized(BM25) + (1 - alpha) * normalized(similarity)
    Weighted,
    /// reciprocal rank fusion: sum of 1/(K + rank)
    Rrf,
}

/// one search request: keyword text and/or an embedding
#[derive(Debug, Clone)]
pub struct SearchRequest {
    /// the keyword query (BM25 side)
    pub text: Option<String>,
    /// the query embedding (vector side)
    pub vector: Option<Vec<f32>>,
    /// which vector field to search (None = the schema's first vector field)
    pub field: Option<String>,
    /// number of results
    pub k: usize,
    pub mode: SearchMode,
    /// keyword weight in `Weighted` fusion (0 = vector only, 1 = keyword only)
    pub alpha: f32,
    pub fusion: Fusion,
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            text: None,
            vector: None,
            field: None,
            k: 10,
            mode: SearchMode::Keyword,
            alpha: 0.5,
            fusion: Fusion::Weighted,
        }
    }
}

impl SearchRequest {
    /// keyword-only search
    pub fn keyword(text: impl Into<String>, k: usize) -> Self {
        Self {
            text: Some(text.into()),
            k,
            ..Default::default()
        }
    }

    /// embedding-only search
    pub fn vector(vector: Vec<f32>, k: usize) -> Self {
        Self {
            vector: Some(vector),
            k,
            mode: SearchMode::Vector,
            ..Default::default()
        }
    }

    /// keyword + embedding, fused
    pub fn hybrid(text: impl Into<String>, vector: Vec<f32>, k: usize) -> Self {
        Self {
            text: Some(text.into()),
            vector: Some(vector),
            k,
            mode: SearchMode::Hybrid,
            ..Default::default()
        }
    }

    /// keyword selects the candidates, the embedding ranks them
    pub fn filter(text: impl Into<String>, vector: Vec<f32>, k: usize) -> Self {
        Self {
            text: Some(text.into()),
            vector: Some(vector),
            k,
            mode: SearchMode::Filter,
            ..Default::default()
        }
    }
}

pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
}

/// aligned docid / score series, ascending docid
#[derive(Debug, Default, Clone)]
pub struct DocScores {
    pub ids: Vec<u32>,
    pub scores: Vec<f32>,
}

impl DocScores {
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn all(ids: Vec<u32>) -> Self {
        let n = ids.len();
        Self {
            ids,
            scores: vec![0.0; n],
        }
    }
}

/// intersect two ascending series, summing scores
pub fn intersect(a: &DocScores, b: &DocScores) -> DocScores {
    let mut out = DocScores::default();
    out.ids.reserve(a.len().min(b.len()));
    out.scores.reserve(a.len().min(b.len()));
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let (di, dj) = (a.ids[i], b.ids[j]);
        match di.cmp(&dj) {
            std::cmp::Ordering::Equal => {
                out.ids.push(di);
                out.scores.push(a.scores[i] + b.scores[j]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    out
}

/// union two ascending series, summing scores on shared docs
pub fn union(a: &DocScores, b: &DocScores) -> DocScores {
    // empty side: the other wins outright (skip alloc + merge)
    if a.is_empty() {
        let mut out = DocScores::default();
        out.ids.extend_from_slice(&b.ids);
        out.scores.extend_from_slice(&b.scores);
        return out;
    }
    if b.is_empty() {
        let mut out = DocScores::default();
        out.ids.extend_from_slice(&a.ids);
        out.scores.extend_from_slice(&a.scores);
        return out;
    }
    let mut out = DocScores::default();
    out.ids.reserve(a.len() + b.len());
    out.scores.reserve(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let (di, dj) = (a.ids[i], b.ids[j]);
        match di.cmp(&dj) {
            std::cmp::Ordering::Equal => {
                out.ids.push(di);
                out.scores.push(a.scores[i] + b.scores[j]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                out.ids.push(di);
                out.scores.push(a.scores[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.ids.push(dj);
                out.scores.push(b.scores[j]);
                j += 1;
            }
        }
    }
    out.ids.extend_from_slice(&a.ids[i..]);
    out.scores.extend_from_slice(&a.scores[i..]);
    out.ids.extend_from_slice(&b.ids[j..]);
    out.scores.extend_from_slice(&b.scores[j..]);
    out
}

use std::cmp::Reverse;

/// threshold + bounded top-k (binary min-heap)
pub struct TopK {
    heap: std::collections::BinaryHeap<Reverse<HitKey>>,
    k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HitKey {
    score_bits: u32,
    addr: DocAddr,
}

impl HitKey {
    fn from(score: f32, addr: DocAddr) -> Self {
        // total-order mapping of f32 bits for heap ordering; ties by addr
        let bits = if score < 0.0 {
            !score.to_bits()
        } else {
            score.to_bits() ^ 0x8000_0000
        };
        Self {
            score_bits: bits,
            addr,
        }
    }
}

impl PartialOrd for HitKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HitKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score_bits
            .cmp(&other.score_bits)
            .then_with(|| self.addr.cmp(&other.addr))
    }
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self {
            heap: std::collections::BinaryHeap::with_capacity(k + 1),
            k,
        }
    }

    pub fn threshold(&self) -> f32 {
        match self.heap.peek() {
            Some(Reverse(h)) if self.heap.len() >= self.k => bit_reverse(h.score_bits),
            _ => f32::MIN,
        }
    }

    pub fn push(&mut self, addr: DocAddr, score: f32) {
        if self.k == 0 {
            return;
        }
        let entry = HitKey::from(score, addr);
        if self.heap.len() < self.k {
            self.heap.push(Reverse(entry));
            return;
        }
        let Some(Reverse(min)) = self.heap.peek() else {
            return;
        };
        if entry.score_bits > min.score_bits
            || (entry.score_bits == min.score_bits && addr < min.addr)
        {
            self.heap.pop();
            self.heap.push(Reverse(entry));
        }
    }

    pub fn into_sorted(mut self) -> Vec<SearchHit> {
        let mut hits: Vec<SearchHit> = self
            .heap
            .drain()
            .map(|Reverse(h)| SearchHit {
                addr: h.addr,
                score: f32::from_bits(if h.score_bits < 0x8000_0000 {
                    !h.score_bits
                } else {
                    h.score_bits ^ 0x8000_0000
                }),
                kw: None,
                vec: None,
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.addr.cmp(&b.addr))
        });
        hits
    }
}

fn bit_reverse(bits: u32) -> f32 {
    if bits < 0x8000_0000 {
        f32::from_bits(!bits)
    } else {
        f32::from_bits(bits ^ 0x8000_0000)
    }
}
