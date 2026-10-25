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

    /// fold a partial-merge posting list into this one; docids must already be
    /// globally sorted across the two (per-chunk docid ranges concatenate cleanly)
    pub fn absorb(&mut self, other: PostingData) {
        self.docids.extend_from_slice(&other.docids);
        self.tfs.extend_from_slice(&other.tfs);
        self.positions.extend(other.positions);
    }

    pub fn freeze(&self, norms: Option<&[u16]>) -> FrozenPosting {
        // SIMD bitpacked planes (zapseg4): per 128-doc block, docid deltas and
        // tfs each get their own bitpacked plane (BitPacker4x, real SIMD
        // kernels). Positions stay varint (not the hot path). Anchors keep the
        // block-max WAND stats and now also the block length.
        use bitpacking::{BitPacker, BitPacker4x};
        let n = self.docids.len();
        let packer = BitPacker4x::new();
        let mut docids = Vec::new();
        // byte_offs[bstart] = offset of that block's header in `docids`
        let mut byte_offs = vec![0u32; n + 1];
        for bstart in (0..n).step_by(SKIP_STEP) {
            let bend = (bstart + SKIP_STEP).min(n);
            let block_len = bend - bstart;
            byte_offs[bstart] = docids.len() as u32;
            // deltas are relative to the previous docid (global), tfs raw
            let mut deltas = [0u32; SKIP_STEP];
            let mut tfs_block = [0u32; SKIP_STEP];
            let mut prev = if bstart == 0 {
                0u32
            } else {
                self.docids[bstart - 1]
            };
            for i in bstart..bend {
                deltas[i - bstart] = self.docids[i] - prev;
                prev = self.docids[i];
                tfs_block[i - bstart] = self.tfs[i];
            }
            let bd = packer.num_bits(&deltas);
            let bt = packer.num_bits(&tfs_block);
            // block header: n_docs, bits_delta, bits_tf, pad
            docids.push(block_len as u8);
            docids.push(bd);
            docids.push(bt);
            docids.push(0);
            if bd > 0 {
                let mut buf = vec![0u8; SKIP_STEP * bd as usize / 8];
                packer.compress(&deltas, &mut buf, bd);
                docids.extend_from_slice(&buf);
            }
            if bt > 0 {
                let mut buf = vec![0u8; SKIP_STEP * bt as usize / 8];
                packer.compress(&tfs_block, &mut buf, bt);
                docids.extend_from_slice(&buf);
            }
        }
        // skip anchors: per 128 docs, block [bstart, bend) stats
        let mut skips = Vec::new();
        for bstart in (0..n).step_by(SKIP_STEP) {
            let bend = (bstart + SKIP_STEP).min(n);
            let block_len = bend - bstart;
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
            skips.push(block_len as u8);
            skips.extend_from_slice(&[0u8; 7]);
        }
        let mut pos = Vec::new();
        let mut pos_offsets = Vec::with_capacity(self.docids.len() + 1);
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
            pos,
            pos_offsets,
            skips,
        }
    }
}

/// docs between skip anchors
const SKIP_STEP: usize = 128;
/// anchors store u32 prev_docid + u32 cursor_off + u32 block_max_tf + u32 block_min_len
/// + u8 block_len + 7 bytes padding = 24 (packed format needs the block length)
const SKIP_ENTRY_BYTES: usize = 24;

/// serialized form of one posting list
pub struct FrozenPosting {
    /// varint deltas between consecutive docids
    docids: Vec<u8>,
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
            pos,
            pos_offsets,
            skips,
            packed: true,
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
    pub pos: Range<u32>,
    pub pos_offsets: Range<u32>,
    /// skip anchors: every SKIP_STEP docs -> (docid, delta-blob byte, tf byte), u32 LE
    pub skips: Range<u32>,
    /// true when the docids section uses SIMD bitpacked planes (zapseg4)
    pub packed: bool,
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
    /// packed (zapseg4) posting format
    packed: bool,
    /// current 128-doc block's decoded deltas/tfs (packed format only)
    block_buf_doc: [u32; SKIP_STEP],
    block_buf_tf: [u32; SKIP_STEP],
    block_len: usize,
    block_pos: usize,
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
            packed: info.packed,
            block_buf_doc: [0; SKIP_STEP],
            block_buf_tf: [0; SKIP_STEP],
            block_len: 0,
            block_pos: 0,
        }
    }

    /// block-max stats for the block containing the current decode position
    /// returns (tf_max, len_min); None when no anchors
    pub fn block_bounds(&self) -> Option<(u32, u32)> {
        let skips = &self.blob[self.skips_start..self.skips_end];
        let n = skips.len() / SKIP_ENTRY_BYTES;
        if n == 0 {
            return None;
        }
        let b = self.docidx / SKIP_STEP;
        let b = b.min(n - 1);
        let row = |field: usize| {
            u32::from_le_bytes(
                skips[b * SKIP_ENTRY_BYTES + field * 4..b * SKIP_ENTRY_BYTES + field * 4 + 4]
                    .try_into()
                    .unwrap(),
            )
        };
        Some((row(2), row(3)))
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
            let next_row = u32::from_le_bytes(
                skips[(b + 1) * SKIP_ENTRY_BYTES..(b + 1) * SKIP_ENTRY_BYTES + 4]
                    .try_into()
                    .unwrap(),
            );
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
        let anchor = |i: usize| {
            u32::from_le_bytes(
                skips[i * SKIP_ENTRY_BYTES..i * SKIP_ENTRY_BYTES + 4]
                    .try_into()
                    .unwrap(),
            )
        };
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
        let a_docid_off = u32::from_le_bytes(
            skips[lo * SKIP_ENTRY_BYTES + 4..lo * SKIP_ENTRY_BYTES + 8]
                .try_into()
                .unwrap(),
        ) as usize;
        let a_tf_off = u32::from_le_bytes(
            skips[lo * SKIP_ENTRY_BYTES + 8..lo * SKIP_ENTRY_BYTES + 12]
                .try_into()
                .unwrap(),
        ) as usize;
        // rebuild cursor offsets from the section bases stored in ctor
        self.docid_c = self.docids_base + a_docid_off;
        let _ = a_tf_off; // single interleaved cursor; tf offset folds into docid cursor
        self.docidx = lo * SKIP_STEP;
        self.prev_docid = a_docid;
        self.block_len = 0;
        self.block_pos = 0;
        self.anchor_lo = lo;
    }

    /// collect all matching docids (used by filter-style evaluation); skips inline tf varints
    pub fn docids(&self) -> RoaringBitmap {
        let mut rb = RoaringBitmap::new();
        if self.packed {
            use bitpacking::{BitPacker, BitPacker4x};
            let packer = BitPacker4x::new();
            let mut docid_c = self.docids_base;
            let mut docidx = 0usize;
            let mut prev = 0u32;
            let mut block_len = 0usize;
            let mut block_pos = 0usize;
            let mut buf = [0u32; SKIP_STEP];
            while docidx < self.doc_count {
                if block_pos >= block_len {
                    if docid_c >= self.docids_end {
                        break;
                    }
                    let n = self.blob[docid_c] as usize;
                    if n == 0 {
                        break;
                    }
                    let bd = self.blob[docid_c + 1];
                    let bt = self.blob[docid_c + 2];
                    docid_c += 4;
                    let delta_bytes = SKIP_STEP * bd as usize / 8;
                    if bd > 0 {
                        packer.decompress(&self.blob[docid_c..docid_c + delta_bytes], &mut buf, bd);
                    } else {
                        buf.fill(0);
                    }
                    docid_c += delta_bytes;
                    // skip tf plane
                    let tf_bytes = SKIP_STEP * bt as usize / 8;
                    docid_c += tf_bytes;
                    block_len = n;
                    block_pos = 0;
                }
                if block_pos >= block_len {
                    break;
                }
                let delta = buf[block_pos];
                prev += delta;
                rb.insert(prev);
                docidx += 1;
                block_pos += 1;
            }
        } else {
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
        }
        rb
    }

    /// decode the full docid series into an ascending Vec (no bitmap container
    /// overhead); caller intersects sorted vecs directly for large-df phrases
    pub fn docid_vec(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.doc_count);
        if self.packed {
            use bitpacking::{BitPacker, BitPacker4x};
            let packer = BitPacker4x::new();
            let mut docid_c = self.docids_base;
            let mut docidx = 0usize;
            let mut prev = 0u32;
            let mut block_len = 0usize;
            let mut block_pos = 0usize;
            let mut buf = [0u32; SKIP_STEP];
            while docidx < self.doc_count {
                if block_pos >= block_len {
                    if docid_c >= self.docids_end {
                        break;
                    }
                    let n = self.blob[docid_c] as usize;
                    if n == 0 {
                        break;
                    }
                    let bd = self.blob[docid_c + 1];
                    let bt = self.blob[docid_c + 2];
                    docid_c += 4;
                    let delta_bytes = SKIP_STEP * bd as usize / 8;
                    if bd > 0 {
                        packer.decompress(&self.blob[docid_c..docid_c + delta_bytes], &mut buf, bd);
                    } else {
                        buf.fill(0);
                    }
                    docid_c += delta_bytes;
                    let tf_bytes = SKIP_STEP * bt as usize / 8;
                    docid_c += tf_bytes;
                    block_len = n;
                    block_pos = 0;
                }
                if block_pos >= block_len {
                    break;
                }
                let delta = buf[block_pos];
                prev += delta;
                out.push(prev);
                docidx += 1;
                block_pos += 1;
            }
        } else {
            let mut prev = 0u32;
            let mut c = self.docids_base;
            while c < self.docids_end {
                let (delta, n) = read_varint(self.blob, c);
                c += n;
                let (_, n2) = read_varint(self.blob, c);
                c += n2;
                prev += delta;
                out.push(prev);
            }
        }
        out
    }

    /// decode next interleaved [delta][tf] pair; caches into `peeked`
    #[inline(always)]
    fn decode_next(&mut self) -> Option<(u32, u32)> {
        if self.docidx >= self.doc_count {
            return None;
        }
        if self.packed {
            if self.block_pos >= self.block_len {
                if self.docid_c >= self.docids_end {
                    return None;
                }
                use bitpacking::{BitPacker, BitPacker4x};
                let packer = BitPacker4x::new();
                let n = self.blob[self.docid_c] as usize;
                if n == 0 {
                    return None;
                }
                let bd = self.blob[self.docid_c + 1];
                let bt = self.blob[self.docid_c + 2];
                self.docid_c += 4;
                let delta_bytes = SKIP_STEP * bd as usize / 8;
                let tf_bytes = SKIP_STEP * bt as usize / 8;
                if bd > 0 {
                    packer.decompress(
                        &self.blob[self.docid_c..self.docid_c + delta_bytes],
                        &mut self.block_buf_doc,
                        bd,
                    );
                } else {
                    self.block_buf_doc.fill(0);
                }
                self.docid_c += delta_bytes;
                if bt > 0 {
                    packer.decompress(
                        &self.blob[self.docid_c..self.docid_c + tf_bytes],
                        &mut self.block_buf_tf,
                        bt,
                    );
                } else {
                    self.block_buf_tf.fill(0);
                }
                self.docid_c += tf_bytes;
                self.block_len = n;
                self.block_pos = 0;
            }
            if self.block_pos >= self.block_len {
                return None;
            }
            // SAFETY: block_pos < block_len <= SKIP_STEP, and both buffers
            // are [u32; SKIP_STEP], so the index is in bounds.
            let delta = unsafe { *self.block_buf_doc.get_unchecked(self.block_pos) };
            let tf = unsafe { *self.block_buf_tf.get_unchecked(self.block_pos) };
            self.prev_docid += delta;
            self.docidx += 1;
            self.block_pos += 1;
            self.peeked = Some((self.prev_docid, tf));
            return self.peeked;
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
        let out = self.peeked.take();
        // keep the peek invariant: next unread doc always cached (positions access)
        if self.peeked.is_none() && self.docidx < self.doc_count {
            self.decode_next();
        }
        out
    }

    /// advance without the peek prefetch: one row decode per call. For pure
    /// docid/tf walks (term scoring, AND gallop drivers) that never touch
    /// peek()/peek_positions_into; phrase cursors keep advance().
    #[inline(always)]
    pub fn advance_lazy(&mut self) -> Option<(u32, u32)> {
        if self.peeked.is_none() {
            if self.docidx >= self.doc_count {
                return None;
            }
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
