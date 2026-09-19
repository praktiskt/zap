//! document addresses + scoring combinators (BM25 term / phrase / boolean)

use crate::posting::{PostingReader, TermInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    ((n - d + 0.5) / (d + 0.5) + 1.0).max(1e-9).ln()
}

/// one term over one segment
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
            idf: idf,
            k1: params.k1,
            b: params.b,
            avg_len: avg_len.max(1.0),
            boost,
            norms,
        }
    }

    fn score(&self, doc: u32, tf: u32) -> f32 {
        let len = self.norms.and_then(|n| n.get(doc as usize)).map(|x| *x as f32);
        let tf = tf as f32;
        let k1 = self.k1;
        let b = self.b;
        let norm = match len {
            Some(len) => (1.0 - b) + b * (len / self.avg_len),
            None => 1.0,
        };
        self.boost * self.idf * (tf * (k1 + 1.0)) / (tf + k1 * norm)
    }
}

impl Scorer for TermScorer<'_> {
    fn peek(&self) -> Option<(u32, f32)> {
        self.peeked.map(|(d, tf)| (d, self.score(d, tf)))
    }

    fn exhaust(&mut self) {
        self.peeked = None;
    }

    fn next(&mut self) -> Option<(u32, f32)> {
        let cur = self.peeked.take()?;
        let out = self.score(cur.0, cur.1);
        self.peeked = self.r.advance();
        Some((cur.0, out))
    }

    fn skip_to(&mut self, target: u32) -> Option<(u32, f32)> {
        self.r.seek_to(target, &mut self.peeked);
        self.peek().take()
    }
}

pub trait Scorer {
    fn peek(&self) -> Option<(u32, f32)>;
    /// retrieve and move forward
    fn next(&mut self) -> Option<(u32, f32)>;
    /// skip docs < target
    fn skip_to(&mut self, target: u32) -> Option<(u32, f32)>;
    fn exhaust(&mut self) -> Option<()> {
        self.next().clear()?;
        None
    }
}

pub trait DocSource {
    /// force full docid set (filters & NOT lowering); may be a scan
    fn doc_ids(&self) -> RoaringBitmap;
}

/// phrase with position verification + window bonus
pub struct PhraseScorer<'a> {
    readers: Vec<PostingReader<'a>>,
    peeked: Vec<Option<(u32, u32)>>,
    /// per-reader current docidx counter, aligned with `readers`
    docidx: Vec<usize>,
    idf_sum: f32,
    slop: u32,
    phrase_boost: f32,
    /// number of phrase terms
    n: usize,
}

impl<'a> PhraseScorer<'a> {
    pub fn new(
        infos: &[crate::posting::TermInfo],
        blob: &'a [u8],
        idf_sum: f32,
        slop: u32,
        phrase_boost: f32,
    ) -> Option<Self> {
        if infos.is_empty() {
            return None;
        }
        let readers: Vec<PostingReader<'a>> = infos.iter().map(|i| PostingReader::new(i, blob)).collect();
        let mut peeked = Vec::with_capacity(readers.len());
        let mut docidx = Vec::with_capacity(readers.len());
        for r in &readers {
            peeked.push(unsafe { std::mem::transmute(r.peek_error()) });
        }
        let _ = docidx;
        None
    }
}
