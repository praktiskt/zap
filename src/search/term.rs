//! term -> DocScores (BM25)

use crate::posting::{PostingReader, TermInfo};
use crate::search::{Bm25Params, DocScores};

pub fn bm25_idf(df: u32, total_docs: u32) -> f32 {
    let n = total_docs.max(1) as f32;
    let d = df.max(1) as f32;
    (1.0 + (n - d + 0.5) / (d + 0.5)).ln()
}

pub fn norm_ratio(norms: Option<&[u16]>, doc: u32, avg_len: f32, b: f32) -> f32 {
    let avg = avg_len.max(1.0);
    match norms.and_then(|n| n.get(doc as usize)) {
        Some(len) => (1.0 - b) + b * (*len as f32 / avg),
        None => 1.0,
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
        let norm = norm_ratio(norms, doc, avg_len, params.b);
        let w = idf * (tf * (params.k1 + 1.0)) / (tf + params.k1 * norm);
        out.ids.push(doc);
        out.scores.push(w * boost);
    }
    out
}

/// score ONLY the docs in `probe` by galloping seeks along the posting list
#[allow(clippy::too_many_arguments)]
pub fn term_scores_probe(
    info: &TermInfo,
    blob: &[u8],
    idf: f32,
    params: &Bm25Params,
    avg_len: f32,
    boost: f32,
    norms: Option<&[u16]>,
    probe: &roaring::RoaringBitmap,
) -> DocScores {
    let mut r = PostingReader::new(info, blob);
    let mut out = DocScores::default();
    out.ids.reserve(probe.len().min(info.df as u64) as usize);
    for doc in probe.iter() {
        r.seek_to(doc);
        if let Some((d, tf)) = r.peek() {
            if d == doc {
                let norm = norm_ratio(norms, doc, avg_len, params.b);
                let tf = tf as f32;
                let w = idf * (tf * (params.k1 + 1.0)) / (tf + params.k1 * norm);
                out.ids.push(doc);
                out.scores.push(w * boost);
            }
        }
    }
    out
}
