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
    let mut r = PostingReader::new(info, blob);
    let mut out = DocScores::default();
    while let Some((doc, tf)) = r.advance() {
        let tf = tf as f32;
        let norm = norm_ratio(norms, doc, avg_len, params.b);
        let w = idf * (tf * (params.k1 + 1.0)) / (tf + params.k1 * norm);
        out.ids.push(doc);
        out.scores.push(w * boost);
    }
    out
}
