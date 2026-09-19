//! BM25 term scorer

use crate::posting::{PostingReader, TermInfo};
use crate::search::{Bm25Params, Scorer};

pub struct TermScorer<'a> {
    r: PostingReader<'a>,
    peeked: Option<(u32, u32)>,
    idf: f32,
    k1: f32,
    b: f32,
    avg_len: f32,
    boost: f32,
    norms: Option<&'a [u16]>,
}

impl<'a> TermScorer<'a> {
    pub fn new(
        info: &crate::posting::TermInfo,
        blob: &'a [u8],
        idf: f32,
        params: &Bm25Params,
        avg_len: f32,
        boost: f32,
        norms: Option<&'a [u16]>,
    ) -> Self {
        let mut r = PostingReader::new(info, blob);
        let peeked = r.advance();
        Self {
            r,
            peeked,
            idf,
            k1: params.k1,
            b: params.b,
            avg_len: avg_len.max(1.0),
            boost,
            norms,
        }
    }

    fn score(&self, doc: u32, tf: u32) -> f32 {
        let len = self
            .norms
            .and_then(|n| n.get(doc as usize))
            .map(|x| *x as f32);
        let tf = tf as f32;
        let norm = match len {
            Some(len) => (1.0 - self.b) + self.b * (len / self.avg_len),
            None => 1.0,
        };
        self.boost * self.idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
    }
}

impl Scorer for TermScorer<'_> {
    fn peek(&self) -> Option<(u32, f32)> {
        self.peeked.map(|(d, tf)| (d, self.score(d, tf)))
    }

    fn next(&mut self) -> Option<(u32, f32)> {
        let cur = self.peeked.take()?;
        let out = self.score(cur.0, cur.1);
        self.peeked = self.r.advance();
        Some((cur.0, out))
    }

    fn skip_to(&mut self, target: u32) -> Option<(u32, f32)> {
        while let Some((d, tf)) = self.peeked {
            if d >= target {
                break;
            }
            self.peeked = self.r.advance();
            let _ = tf;
        }
        self.peek()
    }
}
