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

    pub fn len(&self) -> usize {
        self.docids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docids.is_empty()
    }

    pub fn freeze(&self) -> FrozenPosting {
        let mut docids = Vec::with_capacity(self.len() * 4);
        let mut prev = 0u32;
        for &d in &self.docids {
            write_varint(&mut docids, d - prev);
            prev = d;
        }
        let mut tfs = Vec::with_capacity(self.len());
        for &tf in &self.tfs {
            write_varint(&mut tfs, tf);
        }
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
        }
    }
}

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
        TermInfo {
            df,
            docids,
            tfs,
            pos,
            pos_offsets,
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
}

/// reader over a frozen posting list: sequential doc iteration plus random-access positions
pub struct PostingReader<'a> {
    blob: &'a [u8],
    docid_c: usize,
    tf_c: usize,
    docids_end: usize,
    posoff_start: usize,
    pos_start: usize,
    doc_count: usize,
    docidx: usize,
    prev_docid: u32,
}

impl<'a> PostingReader<'a> {
    pub fn new(info: &TermInfo, blob: &'a [u8]) -> Self {
        let doc_count = info.df as usize;
        let docid_c = info.docids.start as usize;
        let docids_end = info.docids.end as usize;
        Self {
            blob,
            docid_c,
            docids_end,
            tf_c: info.tfs.start as usize,
            posoff_start: info.pos_offsets.start as usize,
            pos_start: info.pos.start as usize,
            doc_count,
            docidx: 0,
            prev_docid: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.doc_count
    }

    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }

    /// collect all matching docids (used by filter-style evaluation)
    pub fn docids(&self) -> RoaringBitmap {
        let mut rb = RoaringBitmap::new();
        let mut prev = 0u32;
        let mut c = self.docid_c;
        while c < self.docids_end {
            let (delta, n) = read_varint(self.blob, c);
            c += n;
            prev += delta;
            rb.insert(prev);
        }
        rb
    }

    /// advance to next doc, returns (docid, tf)
    pub fn advance(&mut self) -> Option<(u32, u32)> {
        if self.docidx >= self.doc_count {
            return None;
        }
        let (delta, n) = read_varint(self.blob, self.docid_c);
        self.docid_c += n;
        let (tf, n2) = read_varint(self.blob, self.tf_c);
        self.tf_c += n2;
        self.prev_docid += delta;
        self.docidx += 1;
        Some((self.prev_docid, tf))
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
        let info = data.freeze().encode_to(&mut blob);
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
}
