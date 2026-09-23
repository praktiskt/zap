//! phrase evaluation: docid intersection + minimum-window position verification.
//! Big candidate lists verify in parallel (per-doc `min_span` is independent).

use rayon::prelude::*;

use crate::posting::PostingReader;
use crate::search::DocScores;

/// linear merge of two ascending docid vecs (branch-light, L1/L2 resident)
fn intersect_sorted(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let (x, y) = (a[i], b[j]);
        if x == y {
            out.push(x);
            i += 1;
            j += 1;
        } else if x < y {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// parallel verification kicks in above this many candidate docs (serial below);
/// heavy docs make per-doc verify expensive, so parallel runs at low counts too
const PAR_MIN_CANDIDATES: usize = 64;
/// granularity floor for rayon chunking (candidates/threads wins below this)
const PAR_CHUNK: usize = 8;

/// track one posting stream positionally
pub struct TermCursor<'a> {
    r: PostingReader<'a>,
}

impl<'a> TermCursor<'a> {
    pub fn new(info: &crate::posting::TermInfo, blob: &'a [u8]) -> Self {
        Self {
            r: PostingReader::new(info, blob),
        }
    }

    /// lazy peek of the cursor's current doc
    pub fn doc(&mut self) -> Option<u32> {
        self.r.peek().map(|(d, _)| d)
    }

    /// gallop forward until doc >= target
    pub fn seek_to(&mut self, target: u32) {
        self.r.seek_to(target);
    }

    /// positions of current doc into `buf` (reused across docs); consumes the doc
    pub fn drain_positions_pub(&mut self, buf: &mut Vec<u32>) {
        buf.clear();
        self.r.peek_positions_into(buf);
        self.r.advance();
    }
}

/// smallest span of an increasing chain with gaps <= slop+1; None when no chain fits.
/// Early-exits at the minimal possible span (n-1: each step advances >= 1).
pub fn min_span(lists: &[Vec<u32>], slop: u32) -> Option<u32> {
    if lists.is_empty() {
        return None;
    }
    let n = lists.len();
    if n == 1 {
        return lists[0].first().map(|_| 0);
    }
    let best_possible = (n - 1) as u32;
    let mut best: Option<u32> = None;
    let mut ptrs = vec![0usize; n];
    for &start in &lists[0] {
        let mut prev = start;
        let mut ok = true;
        for i in 1..n {
            let list = &lists[i];
            while ptrs[i] < list.len() && list[ptrs[i]] < prev + 1 {
                ptrs[i] += 1;
            }
            match list.get(ptrs[i]) {
                Some(&p) if p <= prev + 1 + slop => prev = p,
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            let span = prev - start;
            best = Some(best.map_or(span, |b: u32| b.min(span)));
            if span <= best_possible {
                break;
            }
        }
        // downstream list exhausted: no later start can form a chain, stop
        if (1..n).any(|i| ptrs[i] >= lists[i].len()) {
            break;
        }
    }
    best
}

/// verify one candidate doc: decode positions for every term, min-window span
fn verify_doc(
    cursors: &mut [TermCursor<'_>],
    doc: u32,
    lists: &mut [Vec<u32>],
    slop: u32,
) -> Option<u32> {
    for c in cursors.iter_mut() {
        c.seek_to(doc);
    }
    for (c, l) in cursors.iter_mut().zip(lists.iter_mut()) {
        c.drain_positions_pub(l);
    }
    min_span(lists, slop)
}

/// df gate for the two-pass large-df phrase path (vec intersection + parallel verify)
const VEC_MIN_DF: u64 = 65_536;
/// position-bytes gate: documents whose positional payload dominates get the
/// two-pass parallel verify even when their df is modest (huge-doc corpora).
/// The old df-only gate left 1000x1M-char corpora single-threaded (R18-era
/// threshold measured work by docid count, which says nothing about doc size).
const VEC_MIN_POS_BYTES: u64 = 32_768;
/// intersect all phrase cursors; bonus score scales with window tightness.
/// Small-df phrases use the fused single-pass walk (gallop merge + inline
/// window verify — no re-walk). Large-df phrases decode docid vecs, fold
/// sorted intersections, and verify candidates in rayon-parallel chunks
/// (per-doc independence); below the parallel gate they verify serially.
pub fn phrase_scores(
    infos: &[crate::posting::TermInfo],
    blob: &[u8],
    idf_sum: f32,
    slop: u32,
    phrase_boost: f32,
) -> DocScores {
    if infos.is_empty() {
        return DocScores::default();
    }
    let n_terms = infos.len();
    // bonus ∈ (0, phrase_boost]; span >= n_terms-1 always
    let score = |span: u32| idf_sum * (1.0 + phrase_boost / (1.0 + span as f32));
    let min_df = infos.iter().map(|i| i.df as u64).min().unwrap_or(0);
    // work estimate: either many docs (df) or big positional payload (bytes)
    let pos_bytes = infos
        .iter()
        .map(|i| (i.pos.end - i.pos.start) as u64)
        .min()
        .unwrap_or(0);
    let large_df = n_terms >= 2 && (min_df >= VEC_MIN_DF || pos_bytes >= VEC_MIN_POS_BYTES);

    if !large_df {
        // fused single pass: gallop merge + verify in one walk over the postings
        let mut cursors: Vec<TermCursor> = infos.iter().map(|i| TermCursor::new(i, blob)).collect();
        let mut lists: Vec<Vec<u32>> = (0..n_terms).map(|_| Vec::new()).collect();
        let mut out = DocScores::default();
        loop {
            let mut any_none = false;
            let mut hi = 0u32;
            for c in cursors.iter_mut() {
                match c.doc() {
                    Some(d) => {
                        if d > hi {
                            hi = d;
                        }
                    }
                    None => {
                        any_none = true;
                    }
                }
            }
            if any_none {
                return out;
            }
            for c in cursors.iter_mut() {
                c.seek_to(hi);
            }
            let mut all_eq = true;
            for c in cursors.iter_mut() {
                if c.doc() != Some(hi) {
                    all_eq = false;
                    break;
                }
            }
            if all_eq {
                for (c, l) in cursors.iter_mut().zip(lists.iter_mut()) {
                    c.drain_positions_pub(l);
                }
                if let Some(span) = min_span(&lists, slop) {
                    out.ids.push(hi);
                    out.scores.push(score(span));
                }
            }
        }
    }
    // two-pass large-df path: cheap docid intersection, then (parallel) verify
    let candidates = {
        let mut acc: Option<Vec<u32>> = None;
        for info in infos {
            let v = PostingReader::new(info, blob).docid_vec();
            acc = Some(match acc {
                None => v,
                Some(a) => intersect_sorted(&a, &v),
            });
        }
        acc.unwrap_or_default()
    };
    if candidates.is_empty() {
        return DocScores::default();
    }
    let mut out = DocScores::default();
    if candidates.len() < PAR_MIN_CANDIDATES {
        let mut cursors: Vec<TermCursor> = infos.iter().map(|i| TermCursor::new(i, blob)).collect();
        let mut lists: Vec<Vec<u32>> = (0..n_terms).map(|_| Vec::new()).collect();
        for doc in candidates {
            if let Some(span) = verify_doc(&mut cursors, doc, &mut lists, slop) {
                out.ids.push(doc);
                out.scores.push(score(span));
            }
        }
        return out;
    }
    // parallel: chunk the ascending candidate list; each chunk owns fresh cursors
    // and reuses its position buffers across its docs
    let chunk = candidates
        .len()
        .div_ceil(rayon::current_num_threads())
        .max(PAR_CHUNK);
    let results: Vec<Vec<(u32, u32)>> = candidates
        .par_chunks(chunk)
        .map(|docs| {
            let mut cs: Vec<TermCursor> = infos.iter().map(|i| TermCursor::new(i, blob)).collect();
            let mut lists: Vec<Vec<u32>> = (0..n_terms).map(|_| Vec::new()).collect();
            let mut out: Vec<(u32, u32)> = Vec::with_capacity(docs.len() / 4 + 1);
            for &doc in docs {
                if let Some(span) = verify_doc(&mut cs, doc, &mut lists, slop) {
                    out.push((doc, span));
                }
            }
            out
        })
        .collect();
    for chunk_res in results {
        for (doc, span) in chunk_res {
            out.ids.push(doc);
            out.scores.push(score(span));
        }
    }
    out
}
