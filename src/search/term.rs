//! term -> DocScores (BM25)

use crate::posting::{PostingReader, TermInfo};
use crate::search::{Bm25Params, DocScores};

pub fn bm25_idf(df: u32, total_docs: u32) -> f32 {
    let n = total_docs.max(1) as f32;
    let d = df.max(1) as f32;
    (1.0 + (n - d + 0.5) / (d + 0.5)).ln()
}

/// norm ratio: (1-b) + b*len/avg
pub fn norm_ratio(norms: Option<&[u16]>, doc: u32, avg_len: f32, b: f32) -> f32 {
    let avg = avg_len.max(1.0);
    match norms.and_then(|n| n.get(doc as usize)) {
        Some(len) => (1.0 - b) + b * (*len as f32 / avg),
        None => 1.0,
    }
}

/// per-(term,field) BM25 constants hoisted OUT of the per-doc loops: the old
/// inner expressions recomputed `idf*(k1+1)`, `k1*(1-b)` and a per-doc
/// division `b*len/avg` for every posting. With the folding below a doc
/// scores with 2 mults + 1 div instead of 2 divs + 3 muls.
#[derive(Clone, Copy)]
pub struct Bm25Fold {
    /// idf * (k1 + 1)
    pub pa: f32,
    /// plain k1 (fallback divisor + block-bound normless case)
    pub k1: f32,
    /// k1 * (1 - b)
    pub p1: f32,
    /// k1 * b / avg
    pub p2: f32,
}

impl Bm25Fold {
    pub fn new(idf: f32, params: &Bm25Params, avg_len: f32) -> Self {
        let avg = avg_len.max(1.0);
        let b = params.b;
        let k1 = params.k1;
        Self {
            pa: idf * (k1 + 1.0),
            k1,
            p1: k1 * (1.0 - b),
            p2: k1 * b / avg,
        }
    }

    /// BM25(doc) for term tf, doc length `len`:
    /// idf*(k1+1)*tf / (tf + k1*((1-b) + b*len/avg))
    #[inline]
    pub fn score(&self, tf: f32, len: u16) -> f32 {
        // k1*norm = k1*((1-b) + b*len/avg) = p1 + p2*len
        self.pa * tf / (tf + self.p1 + self.p2 * len as f32)
    }

    /// block-max bound over the block's (tf_max, len_min) — SAME evaluation
    /// shape as `score` (identical f32 associativity) so a doc whose score
    /// IS exactly the threshold equals its block bound bit-for-bit and can
    /// never be pruned by `< thr` (the T1 invariant that ties the pruner to
    /// the dense collector).
    #[inline]
    pub fn bound(&self, tfm: f32, len_min: u16) -> f32 {
        self.pa * tfm / (tfm + self.p1 + self.p2 * len_min as f32)
    }

    /// caller-held idf*(k1+1) fallback for norm-less docs (norm 1.0):
    /// exactly the old `idf*(tf*(k1+1))/(tf + k1*1.0)` value
    #[inline]
    pub fn score_fallback(&self, idf_k1p1: f32, tf: f32) -> f32 {
        idf_k1p1 * tf / (tf + self.k1)
    }
}

pub fn term_scores(
    info: &TermInfo,
    blob: &[u8],
    idf: f32,
    params: &Bm25Params,
    avg_len: f32,
    boost: f32,
    norms: Option<&[u16]>,
) -> DocScores {
    // `allowed`: docs outside the probe bitmap are decode-skipped (varints still walked)
    term_scores_allowed(info, blob, idf, params, avg_len, boost, norms, None)
}

#[allow(clippy::too_many_arguments)]
pub fn term_scores_allowed(
    info: &TermInfo,
    blob: &[u8],
    idf: f32,
    params: &Bm25Params,
    avg_len: f32,
    boost: f32,
    norms: Option<&[u16]>,
    allowed: Option<&roaring::RoaringBitmap>,
) -> DocScores {
    let fold = Bm25Fold::new(idf, params, avg_len);
    let mut r = PostingReader::new(info, blob);
    let mut out = DocScores::default();
    out.ids.reserve(info.df as usize);
    while let Some((doc, tf)) = r.advance() {
        if let Some(a) = allowed {
            if !a.contains(doc) {
                continue;
            }
        }
        let tf = tf as f32;
        // norm slice with the doc present = hoisted fold; missing doc/norm
        // slice => norm 1.0 (old norm_ratio fallback, exact same value)
        let w = match norms.and_then(|n| n.get(doc as usize)) {
            Some(&len) => fold.score(tf, len),
            None => idf * (tf * (params.k1 + 1.0)) / (tf + params.k1),
        };
        out.ids.push(doc);
        out.scores.push(w * boost);
    }
    out
}
