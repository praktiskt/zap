//! compact posting lists: delta-varint docids, varint tfs, varint position runs

use roaring::RoaringBitmap;
use std::ops::Range;

pub fn write_varint(out: &mut Vec<u8>, mut v: u32) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            break;
        }
        out.push(b | 0x80);
    }
}

/// byte length of a value's varint encoding
pub fn u32_len(v: u32) -> usize {
    let bits = u32::BITS - v.leading_zeros();
    (((bits + 6).max(1)) / 7) as usize
}

/// decode one varint at `pos`, returns (value, bytes consumed)
pub fn read_varint(bytes: &[u8], pos: usize) -> (u32, usize) {
    let mut v = 0u32;
    let mut shift = 0;
    let mut p = pos;
    loop {
        let b = bytes[p];
        p += 1;
        v |= ((b & 0x7f) as u32) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (v, p - pos)
}

/// in-memory posting data accumulated while documents are added
#[derive(Default)]
pub struct PostingData {
    pub docids: Vec<u32>,
    pub tfs: Vec<u32>,
    pub positions: Vec<Vec<u32>>,
}

impl PostingData {
    pub fn add(&mut self, doc: u32, tf: u32, positions: Vec<u32>) {
        debug_assert!(self.docids.is_empty() || self.docids.last().unwrap() < &doc);
        self.docids.push(doc);
        self.tfs.push(tf);
        self.positions.push(positions);
    }

    pub fn len(&self) -> usize {
        self.docids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docids.is_empty()
    }

    pub fn freeze(&self, norms: Option<&[u16]>) -> FrozenPosting {
        // R16: interleaved [delta_i][tf_i] per doc, one cursor.
        // WAND: anchors carry per-block max tf + min doc len for score-bound pruning.
        let n = self.len();
        let mut docids = Vec::with_capacity(n * 6);
        let mut byte_offs: Vec<u32> = Vec::with_capacity(n + 1); // doc cursor byte offsets
        let mut prev = 0u32;
        for (docidx, &d) in self.docids.iter().enumerate() {
            byte_offs.push(docids.len() as u32);
            let delta = d - prev;
            prev = d;
            write_varint(&mut docids, delta);
            write_varint(&mut docids, self.tfs[docidx]);
        }
        // skip anchors: per 128 docs, block [bstart, bend) stats
        let mut skips = Vec::new();
        for bstart in (0..n).step_by(SKIP_STEP) {
            let bend = (bstart + SKIP_STEP).min(n);
            let prev_docid = if bstart == 0 {
                0u32
            } else {
                self.docids[bstart - 1]
            };
            let tf_max = self.tfs[bstart..bend].iter().max().copied().unwrap_or(1);
            let len_min = match norms {
                Some(ns) => {
                    let mut lo = u16::MAX;
                    for &docid in &self.docids[bstart..bend] {
                        if let Some(len) = ns.get(docid as usize) {
                            if *len < lo {
                                lo = *len;
                            }
                        }
                    }
                    if lo == u16::MAX {
                        0
                    } else {
                        lo as u32
                    }
                }
                None => 0,
            };
            skips.extend_from_slice(&prev_docid.to_le_bytes());
            skips.extend_from_slice(&byte_offs[bstart].to_le_bytes());
            skips.extend_from_slice(&tf_max.to_le_bytes());
            skips.extend_from_slice(&len_min.to_le_bytes());
        }
        let tfs: Vec<u8> = Vec::new();
        let mut pos = Vec::new();
        let mut pos_offsets = Vec::with_capacity(self.len() + 1);
        pos_offsets.push(0u32);
        for plist in &self.positions {
            write_varint(&mut pos, plist.len() as u32);
            let mut prev_p = 0u32;
            for &p in plist {
                write_varint(&mut pos, p - prev_p);
                prev_p = p;
            }
            pos_offsets.push(pos.len() as u32);
        }
        FrozenPosting {
            docids,
            tfs,
            pos,
            pos_offsets,
            skips,
        }
    }
}

/// docs between skip anchors
const SKIP_STEP: usize = 128;
/// anchors store u32 prev_docid + u32 cursor_off + u32 block_max_tf + u32 block_min_len
const SKIP_ENTRY_BYTES: usize = 16;

/// serialized form of one posting list
pub struct FrozenPosting {
    /// varint deltas between consecutive docids
    docids: Vec<u8>,
    /// varint tf per doc, doc order
    tfs: Vec<u8>,
    /// concatenated per-doc position runs (varint count then varint gaps)
    pos: Vec<u8>,
    /// (len+1) byte offsets into `pos`, plain u32 LE
    pos_offsets: Vec<u32>,
    /// per-block anchors: block_start_docid(-base!), start offset(s) etc u32 LE
    skips: Vec<u8>,
}

impl FrozenPosting {
    /// append to `blob`; returns term info with ranges into it
    pub fn encode_to(mut self, blob: &mut Vec<u8>) -> TermInfo {
        let df = self.pos_offsets.len() as u32 - 1;
        let docids = write_section(blob, std::mem::take(&mut self.docids));
        let tfs = write_section(blob, std::mem::take(&mut self.tfs));
        let pos = write_section(blob, std::mem::take(&mut self.pos));
        let mut offs = Vec::with_capacity(self.pos_offsets.len() * 4);
        for &o in &self.pos_offsets {
            offs.extend_from_slice(&o.to_le_bytes());
        }
        let pos_offsets = write_section(blob, offs);
        // skip entries are relative to the section starts: store docid offsets raw
        let skips = write_section(blob, std::mem::take(&mut self.skips));
        TermInfo {
            df,
            docids,
            tfs,
            pos,
            pos_offsets,
            skips,
        }
    }
}

fn write_section(blob: &mut Vec<u8>, data: Vec<u8>) -> Range<u32> {
    let s = blob.len() as u32;
    blob.extend_from_slice(&data);
    s..blob.len() as u32
}

#[derive(Clone, Debug)]
pub struct TermInfo {
    /// number of docs in the list
    pub df: u32,
    pub docids: Range<u32>,
    pub tfs: Range<u32>,
    pub pos: Range<u32>,
    pub pos_offsets: Range<u32>,
    /// skip anchors: every SKIP_STEP docs -> (docid, delta-blob byte, tf byte), u32 LE
    pub skips: Range<u32>,
}

/// reader over a frozen posting list: sequential doc iteration, galloping seeks, positions
pub struct PostingReader<'a> {
    blob: &'a [u8],
    /// docids section base: skip anchors are offsets relative to it
    docids_base: usize,
    docid_c: usize,
    docids_end: usize,
    /// cached lower bound for forward-only anchor searches
    anchor_lo: usize,
    posoff_start: usize,
    pos_start: usize,
    skips_start: usize,
    skips_end: usize,
    doc_count: usize,
    /// number of docs decoded so far (peeked doc included)
    docidx: usize,
    prev_docid: u32,
    /// decoded but not yet handed out by advance()
    peeked: Option<(u32, u32)>,
}

impl<'a> PostingReader<'a> {
    pub fn new(info: &TermInfo, blob: &'a [u8]) -> Self {
        let doc_count = info.df as usize;
        Self {
            blob,
            docids_base: info.docids.start as usize,
            docid_c: info.docids.start as usize,
            docids_end: info.docids.end as usize,
            posoff_start: info.pos_offsets.start as usize,
            pos_start: info.pos.start as usize,
            skips_start: info.skips.start as usize,
            skips_end: info.skips.end as usize,
            anchor_lo: 0,
            doc_count,
            docidx: 0,
            prev_docid: 0,
            peeked: None,
        }
    }

    pub fn len(&self) -> usize {
        self.doc_count
    }

    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }

    /// docidx of the last doc returned by advance() (for positions()); None before the first advance
    pub fn cur_docidx(&self) -> Option<usize> {
        // decoded minus any still-peeked (not consumed) doc
        let consumed = self
            .docidx
            .saturating_sub(usize::from(self.peeked.is_some()));
        if consumed == 0 {
            None
        } else {
            Some(consumed - 1)
        }
    }

    /// the peeked docid, if any
    pub fn peek_docid(&self) -> Option<u32> {
        self.peeked.map(|(d, _)| d)
    }

    /// docidx of the doc whose positions peek_positions() would fetch (the peeked one)
    pub fn peek_docidx(&self) -> Option<usize> {
        self.peeked.as_ref().map(|_| self.docidx - 1)
    }

    /// block-max stats for the block containing the current decode position
    /// returns (tf_max, len_min, next_block_first_docid); None when no anchors
    pub fn block_bounds(&self) -> Option<(u32, u32, u32, u32)> {
        let skips = &self.blob[self.skips_start..self.skips_end];
        let n = skips.len() / SKIP_ENTRY_BYTES;
        if n == 0 {
            return None;
        }
        let b = self.docidx / SKIP_STEP;
        let b = b.min(n - 1);
        let row = |field: usize| {
            u32::from_le_bytes(
                skips[b * 16 + field * 4..b * 16 + field * 4 + 4]
                    .try_into()
                    .unwrap(),
            )
        };
        let prev_docid = row(0);
        let _off = row(1);
        let tf_max = row(2);
        let len_min = row(3);
        // first doc of the next block: prev_docid of the following anchor + 1
        let _next_start = if b + 1 < n {
            let next_row = |field: usize| {
                u32::from_le_bytes(
                    skips[(b + 1) * 16 + field * 4..(b + 1) * 16 + field * 4 + 4]
                        .try_into()
                        .unwrap(),
                )
            };
            Some(next_row(0) + 1)
        } else {
            None
        };
        Some((prev_docid, _off, tf_max, len_min))
    }

    /// block index containing the current decode position
    pub fn cur_block(&self) -> usize {
        self.docidx / SKIP_STEP
    }

    /// first docid of the block following the current one (for block skipping)
    pub fn next_block_start(&self) -> Option<u32> {
        let skips = &self.blob[self.skips_start..self.skips_end];
        let n = skips.len() / SKIP_ENTRY_BYTES;
        let b = (self.docidx / SKIP_STEP).min(n.saturating_sub(1));
        if b + 1 < n {
            let next_row =
                u32::from_le_bytes(skips[(b + 1) * 16..(b + 1) * 16 + 4].try_into().unwrap());
            Some(next_row + 1)
        } else {
            None
        }
    }

    /// positions encoded per doc run (varint count then gaps)
    pub fn positions_into(&self, docidx: usize, buf: &mut Vec<u32>) {
        buf.clear();
        let pos = self.posoff_start + docidx * 4;
        let r = |i: usize| -> u32 {
            u32::from_le_bytes(self.blob[pos + i * 4..pos + i * 4 + 4].try_into().unwrap())
        };
        let s = r(0) as usize;
        let e = r(1) as usize;
        let run = &self.blob[self.pos_start + s..self.pos_start + e];
        let (count, mut p) = read_varint(run, 0);
        buf.reserve(count as usize);
        let mut prev = 0u32;
        for _ in 0..count {
            let (d, n) = read_varint(run, p);
            p += n;
            prev += d;
            buf.push(prev);
        }
    }

    /// positions of the peeked doc into `buf` reused buffer (does not consume)
    pub fn peek_positions_into(&self, buf: &mut Vec<u32>) -> bool {
        if self.peeked.is_none() {
            return false;
        }
        let docidx = self.docidx - 1;
        self.positions_into(docidx, buf);
        true
    }

    /// gallop to the first doc >= target; `peek()` then yields that doc
    pub fn seek_to(&mut self, target: u32) {
        if self.peeked.is_some_and(|p| p.0 >= target) {
            return;
        }
        self.peeked = None;
        self.reset_to_anchor(target);
        loop {
            match self.decode_next() {
                Some(p) if p.0 < target => {
                    self.peeked = None;
                    continue;
                }
                None => break,
                Some(_) => break,
            }
        }
    }

    /// jump cursors to the best skip anchor <= target (leave decode state consistent)
    fn reset_to_anchor(&mut self, target: u32) {
        if self.docidx >= self.doc_count {
            return;
        }
        let skips = &self.blob[self.skips_start..self.skips_end];
        let n = skips.len() / SKIP_ENTRY_BYTES;
        if n == 0 {
            return;
        }
        // binary search: last anchor index with anchor docid <= target; forward-only hint
        let anchor = |i: usize| u32::from_le_bytes(skips[i * 16..i * 16 + 4].try_into().unwrap());
        let (mut lo, mut hi) = (self.anchor_lo.min(n.saturating_sub(1)), n);
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if anchor(mid) <= target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let use_anchor = anchor(lo) <= target && lo * SKIP_STEP >= self.docidx;
        if !use_anchor {
            return;
        }
        // anchor stores absolute byte offsets within each section (measured from section start)
        let a_docid = anchor(lo);
        let a_docid_off =
            u32::from_le_bytes(skips[lo * 16 + 4..lo * 16 + 8].try_into().unwrap()) as usize;
        let a_tf_off =
            u32::from_le_bytes(skips[lo * 16 + 8..lo * 16 + 12].try_into().unwrap()) as usize;
        // rebuild cursor offsets from the section bases stored in ctor
        self.docid_c = self.docids_base + a_docid_off;
        let _ = a_tf_off; // single interleaved cursor; tf offset folds into docid cursor
        self.docidx = lo * SKIP_STEP;
        self.prev_docid = a_docid;
        self.anchor_lo = lo;
    }

    /// collect all matching docids (used by filter-style evaluation); skips inline tf varints
    pub fn docids(&self) -> RoaringBitmap {
        let mut rb = RoaringBitmap::new();
        let mut prev = 0u32;
        let mut c = self.docids_base;
        while c < self.docids_end {
            let (delta, n) = read_varint(self.blob, c);
            c += n;
            let (_, n2) = read_varint(self.blob, c);
            c += n2;
            prev += delta;
            rb.insert(prev);
        }
        rb
    }

    /// decode next interleaved [delta][tf] pair; caches into `peeked`
    fn decode_next(&mut self) -> Option<(u32, u32)> {
        if self.docidx >= self.doc_count {
            return None;
        }
        let (delta, n) = read_varint(self.blob, self.docid_c);
        self.docid_c += n;
        let (tf, n2) = read_varint(self.blob, self.docid_c);
        self.docid_c += n2;
        self.prev_docid += delta;
        self.docidx += 1;
        self.peeked = Some((self.prev_docid, tf));
        self.peeked
    }

    /// peek next un-consumed doc (fills the cache without consuming)
    pub fn peek(&mut self) -> Option<(u32, u32)> {
        if self.peeked.is_none() && self.docidx < self.doc_count {
            self.decode_next();
        }
        self.peeked
    }

    /// advance to next doc, returns (docid, tf)
    pub fn advance(&mut self) -> Option<(u32, u32)> {
        if self.peeked.is_none() && self.docidx < self.doc_count {
            self.decode_next();
        }
        self.peeked.take()
    }

    /// positions of the doc at 0-based order `docidx`
    pub fn positions(&self, docidx: usize) -> Option<Vec<u32>> {
        let offs = self.posoff_start + docidx * 4;
        let get = |i: usize| -> Option<u32> {
            Some(u32::from_le_bytes(
                self.blob[offs + i * 4..offs + i * 4 + 4].try_into().ok()?,
            ))
        };
        let s = get(0)? as usize;
        let e = get(1)? as usize;
        let run = &self.blob[self.pos_start + s..self.pos_start + e];
        let (count, mut p) = read_varint(run, 0);
        let mut out = Vec::with_capacity(count as usize);
        let mut prev = 0u32;
        for _ in 0..count {
            let (d, n) = read_varint(run, p);
            p += n;
            prev += d;
            out.push(prev);
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip() {
        let vals = [0u32, 1, 127, 128, 130, 3_123_456, u32::MAX];
        let mut out = Vec::new();
        for &v in &vals {
            write_varint(&mut out, v);
        }
        let mut pos = 0;
        for &v in &vals {
            let (got, n) = read_varint(&out, pos);
            assert_eq!(got, v);
            pos += n;
        }
        assert_eq!(pos, out.len());
    }

    #[test]
    fn posting_roundtrip() {
        let mut data = PostingData::default();
        data.add(2, 2, vec![0, 7]);
        data.add(9, 1, vec![3]);
        data.add(1000, 3, vec![1, 4, 9]);
        let mut blob = Vec::new();
        let info = data.freeze(None).encode_to(&mut blob);
        assert_eq!(info.df, 3);

        let mut r = PostingReader::new(&info, &blob);
        assert_eq!(r.docids().iter().collect::<Vec<_>>(), vec![2, 9, 1000]);
        assert_eq!(r.advance(), Some((2, 2)));
        assert_eq!(r.positions(0), Some(vec![0, 7]));
        assert_eq!(r.advance(), Some((9, 1)));
        assert_eq!(r.positions(1), Some(vec![3]));
        assert_eq!(r.advance(), Some((1000, 3)));
        assert_eq!(r.positions(2), Some(vec![1, 4, 9]));
        assert_eq!(r.advance(), None);
    }

    #[test]
    fn seek_gallops() {
        let mut data = PostingData::default();
        for i in 0..600u32 {
            data.add(i * 7, 1, vec![i]);
        }
        let mut blob = Vec::new();
        let info = data.freeze(None).encode_to(&mut blob);
        let want = |target: u32| -> u32 { target.div_ceil(7) * 7 };
        // docs are i*7 for i in 0..600 => last doc = 4193
        let mut r = PostingReader::new(&info, &blob);
        for step in [0u32, 900, 3000] {
            r.seek_to(step);
            let d = r.advance();
            assert_eq!(d, Some((want(step), 1)), "after seek {step}");
        }
        // past-the-end seeks exhaust the reader
        let mut r = PostingReader::new(&info, &blob);
        r.seek_to(4200);
        assert_eq!(r.advance(), None);
        // forward-only invariant: seek backwards keeps the latest state
        let mut r2 = PostingReader::new(&info, &blob);
        r2.seek_to(350);
        assert_eq!(r2.advance(), Some((350, 1)));
        r2.seek_to(100);
        assert_eq!(r2.advance(), Some((357, 1)));
    }
}
