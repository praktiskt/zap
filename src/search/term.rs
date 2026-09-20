//! term -> DocScores (BM25)

use crate::posting::{PostingReader, TermInfo};
use crate::search::{Bm25Params, DocScores};

pub fn bm25_idf(df: u32, total_docs: u32) -> f32 {
    let n = total_docs.max(1) as f32;
    let d = df.max(1) as f32;
    (1.0 + (n - d + 0.5) / (d + 0.5)).ln()
}

/// bm25 evaluation with per-block upper-bound skip (block-max WAND lite)
/// `threshold_fn() -> f32` returns the current top-k heap minimum; blocks whose
/// theoretical max score can't beat it are jumped over (docs in them are missing
/// from the output by design: retained top-k only)
#[allow(clippy::too_many_arguments)]
pub fn term_scores_wand(
    info: &TermInfo,
    blob: &[u8],
    idf: f32,
    params: &Bm25Params,
    avg_len: f32,
    boost: f32,
    norms: Option<&[u16]>,
    threshold_fn: impl Fn() -> f32,
) -> DocScores {
    let mut r = PostingReader::new(info, blob);
    let mut out = DocScores::default();
    out.ids.reserve(info.df as usize);
    let mut seen_block = usize::MAX;
    let mut done = false;
    while !done {
        match r.advance() {
            Some((doc, tf)) => {
                let blk = r.cur_block();
                if blk != seen_block {
                    seen_block = blk;
                    if let Some((_, _, tf_max, len_min)) = r.block_bounds() {
                        let tfm = tf_max.max(tf) as f32;
                        let norm_min = if len_min > 0 {
                            (1.0 - params.b) + params.b * (len_min as f32 / avg_len.max(1.0))
                        } else {
                            1.0
                        };
                        let bound = idf * (tfm * (params.k1 + 1.0)) / (tfm + params.k1 * norm_min);
                        if bound * boost < threshold_fn() {
                            match r.next_block_start() {
                                Some(ns) => {
                                    r.seek_to(ns);
                                    continue;
                                }
                                None => break,
                            }
                        }
                    }
                }
                let tf = tf as f32;
                let norm = norm_ratio(norms, doc, avg_len, params.b);
                let w = idf * (tf * (params.k1 + 1.0)) / (tf + params.k1 * norm);
                out.ids.push(doc);
                out.scores.push(w * boost);
            }
            None => {
                done = true;
            }
        }
    }
    out
}

/// norm ratio: (1-b) + b*len/avg
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
