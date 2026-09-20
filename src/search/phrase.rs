//! phrase evaluation: docid intersection + minimum-window position verification

use crate::posting::PostingReader;
use crate::search::DocScores;

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

    /// positions of current doc; consumes the doc (cursor moves past it)
    fn take_positions(&mut self) -> Vec<u32> {
        let pos = self.r.peek_positions().unwrap_or_default();
        self.r.advance();
        pos
    }
}

/// smallest span of an increasing chain with gaps <= slop+1; None when no chain fits
pub fn min_span(lists: &[Vec<u32>], slop: u32) -> Option<u32> {
    if lists.is_empty() {
        return None;
    }
    let n = lists.len();
    if n == 1 {
        return lists[0].first().map(|_| 0);
    }
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
            if span == 0 {
                break;
            }
        }
    }
    best
}

/// intersect all phrase cursors; bonus score scales with window tightness
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
    let mut cursors: Vec<TermCursor> = infos.iter().map(|i| TermCursor::new(i, blob)).collect();
    let mut out = DocScores::default();
    loop {
        // current max doc across cursors; exhaustion ends intersection
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
            let lists: Vec<Vec<u32>> = cursors.iter_mut().map(|c| c.take_positions()).collect();
            if let Some(span) = min_span(&lists, slop) {
                let bonus = phrase_boost / (1.0 + span as f32);
                out.ids.push(hi);
                out.scores.push(idf_sum * (1.0 + bonus));
            }
        }
    }
}
