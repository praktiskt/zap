//! scoring combinators: doc-materialized scores (BM25), boolean vector merges, phrase windows

pub mod exec;
pub mod phrase;
pub mod term;

use roaring::RoaringBitmap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DocAddr {
    pub segment: u32,
    pub docid: u32,
}

pub struct SearchHit {
    pub addr: DocAddr,
    pub score: f32,
}

pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
}

pub fn bm25_idf(df: u32, total_docs: u32) -> f32 {
    let n = total_docs.max(1) as f32;
    let d = df.max(1) as f32;
    (1.0 + (n - d + 0.5) / (d + 0.5)).ln()
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

/// zero-score series from a bitmap
pub fn from_bitmap(rb: &RoaringBitmap) -> DocScores {
    DocScores::all(rb.iter().collect())
}

/// bitmap complement over [0, cap)
pub fn complement(rb: &RoaringBitmap, cap: u32) -> RoaringBitmap {
    let full = RoaringBitmap::from_iter(0..cap);
    &full - rb
}

/// threshold + bounded top-k collection
pub struct TopK {
    hits: Vec<(f32, DocAddr)>,
    k: usize,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self {
            hits: Vec::new(),
            k,
        }
    }

    pub fn threshold(&self) -> f32 {
        if self.hits.len() < self.k {
            f32::MIN
        } else {
            self.hit_min().0
        }
    }

    fn hit_min(&self) -> (f32, DocAddr) {
        *self
            .hits
            .iter()
            .min_by(|x, y| x.0.total_cmp(&y.0).then_with(|| x.1.cmp(&y.1)))
            .unwrap()
    }

    pub fn push(&mut self, addr: DocAddr, score: f32) {
        if self.k == 0 {
            return;
        }
        if self.hits.len() < self.k {
            self.hits.push((score, addr));
            return;
        }
        let m = self.hit_min();
        if score > m.0 || (score == m.0 && addr < m.1) {
            let i = self
                .hits
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            self.hits[i] = (score, addr);
        }
    }

    pub fn into_sorted(mut self) -> Vec<SearchHit> {
        self.hits
            .sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        self.hits
            .into_iter()
            .map(|(score, addr)| SearchHit { addr, score })
            .collect()
    }
}
