//! segment: immutable per-segment inverted index + compressed doc store
//!
//! term dictionary: one FST per field (term -> global-info ordinal), enabling
//! DFA/range queries (prefix, fuzzy, glob) without dict scans.

use std::collections::HashMap;

use fst::automaton::Automaton;
use fst::{IntoStreamer, Map, MapBuilder, Streamer};
use lz4_flex::{compress_prepend_size, decompress_size_prepended};

use crate::analyzer::Analyzer;
use crate::posting::{PostingData, PostingReader, TermInfo};
use crate::schema::{Document, Schema, StoredDoc};
use std::sync::Arc;

/// doc-store size gate: below this, store raw instead of lz4 (frequent tiny docs)
const RAW_STORE_THRESHOLD: usize = 4096;
const STORE_RAW_TAG: u8 = 0x00;
const STORE_COMP_TAG: u8 = 0x01;
/// block flags: bit0 = external id present, bit1 = metadata present
const BLOCK_FLAG_ID: u8 = 0b0000_0001;
const BLOCK_FLAG_META: u8 = 0b0000_0010;
/// block layout: [tag][len u32][flags u8][n u8][TLV(fid u32, len u32, text)][id len u32 + utf8][meta len u32 + json]
fn encode_store(doc: &Document, schema: &Schema) -> Result<Vec<u8>, String> {
    let mut pairs: Vec<(u32, &String)> = doc
        .fields
        .iter()
        .map(|(name, val)| {
            let fid = schema
                .field_id(name)
                .ok_or_else(|| format!("unknown field `{name}`"))?;
            Ok((fid, val))
        })
        .collect::<Result<_, String>>()?;
    pairs.sort_by_key(|p| p.0);
    let mut block = Vec::new();
    let mut flags = 0u8;
    if doc.id.is_some() {
        flags |= BLOCK_FLAG_ID;
    }
    if doc.metadata.is_some() {
        flags |= BLOCK_FLAG_META;
    }
    block.push(flags);
    block.push(pairs.len() as u8);
    for (fid, val) in pairs {
        block.extend_from_slice(&fid.to_le_bytes());
        block.extend_from_slice(&(val.len() as u32).to_le_bytes());
        block.extend_from_slice(val.as_bytes());
    }
    if let Some(id) = &doc.id {
        block.extend_from_slice(&(id.len() as u32).to_le_bytes());
        block.extend_from_slice(id.as_bytes());
    }
    if let Some(meta) = &doc.metadata {
        let json = serde_json::to_vec(meta).map_err(|e| format!("metadata: {e}"))?;
        block.extend_from_slice(&(json.len() as u32).to_le_bytes());
        block.extend_from_slice(&json);
    }
    Ok(block)
}

pub struct SegmentBuilder {
    schema: Arc<Schema>,
    analyzer: Arc<Analyzer>,
    pub num_docs: u32,
    /// field_id -> term -> postings (textarea feed of sorted terms into each field FST)
    postings: Vec<HashMap<String, PostingData>>,
    /// field_id -> doc -> token count
    field_norms: Vec<Vec<u16>>,
    /// lz4-compressed blocks, one per doc
    store_blocks: Vec<Box<[u8]>>,
    /// per-doc external id (None = implicit internal id only)
    ids: Vec<Option<String>>,
}

impl SegmentBuilder {
    pub fn new(schema: Arc<Schema>, analyzer: Arc<Analyzer>) -> Self {
        let nf = schema.num_fields();
        let mut postings = Vec::with_capacity(nf);
        let mut field_norms = Vec::with_capacity(nf);
        for _ in 0..nf {
            postings.push(HashMap::new());
            field_norms.push(Vec::new());
        }
        Self {
            schema,
            analyzer,
            num_docs: 0,
            postings,
            field_norms,
            store_blocks: Vec::new(),
            ids: Vec::new(),
        }
    }

    /// index + store a document; returns its segment-local docid
    pub fn add_document(&mut self, doc: &Document) -> Result<u32, String> {
        let docid = self.num_docs;
        let block = encode_store(doc, &self.schema)?;
        for (name, val) in &doc.fields {
            let fid = self
                .schema
                .field_id(name)
                .ok_or_else(|| format!("unknown field `{name}`"))?;
            let mut pos = 0u32;
            for tok in self.analyzer.tokenize(val) {
                // R21: get-mut-first dict probe — token allocation only when the
                // term is new to this builder (hot terms skip the per-occurrence
                // String clone entirely)
                let e = match self.postings[fid as usize].get_mut(tok.text.as_ref()) {
                    Some(e) => e,
                    None => self.postings[fid as usize]
                        .entry(tok.text.into_owned())
                        .or_default(),
                };
                if e.docids.last() == Some(&docid) {
                    // repeated token in the same doc: bump tf, extend positions
                    let last = e.tfs.len() - 1;
                    e.tfs[last] += 1;
                    e.positions
                        .last_mut()
                        .expect("tf>0 implies positions")
                        .push(tok.pos);
                } else {
                    e.add(docid, 1, vec![tok.pos]);
                }
                pos += 1;
            }
            self.field_norms[fid as usize].push(pos as u16);
        }
        // small docs stay raw (marker 0x00) to skip lz4 costs; big docs compress (0x01)
        let stored: Box<[u8]> = if block.len() <= RAW_STORE_THRESHOLD {
            let mut raw = Vec::with_capacity(block.len() + 5);
            raw.push(STORE_RAW_TAG);
            raw.extend_from_slice(&(block.len() as u32).to_le_bytes());
            raw.extend_from_slice(&block);
            raw.into_boxed_slice()
        } else {
            let mut c = compress_prepend_size(&block);
            c.insert(0, STORE_COMP_TAG);
            c.into_boxed_slice()
        };
        self.store_blocks.push(stored);
        self.ids.push(doc.id.clone());
        self.num_docs += 1;
        Ok(docid)
    }

    pub fn freeze(self) -> SegmentReader {
        freeze_parts(
            self.postings,
            self.field_norms,
            self.store_blocks,
            self.ids,
            self.num_docs,
        )
    }
}

pub struct SegmentReader {
    /// stable identity across publishes: tombstones, id-map addresses and
    /// snapshot lookups key on this instead of the snapshot's positional
    /// index (which renumbers at merges). 0 = unassigned (assigned by Index
    /// when the segment is installed).
    pub segid: u32,
    pub num_docs: u32,
    /// per-field term dictionaries; global infos offset base is tracked per field
    fsts: Vec<Map<Vec<u8>>>,
    /// shadow copy of the FST bytes (populated at freeze; to_bytes uses this)
    fst_raw: Box<[Vec<u8>]>,
    /// memoized content hash (fnv64 of to_bytes): segments are immutable, so
    /// the disk layer can address files without re-serializing on every call
    content_hash: std::sync::OnceLock<u64>,
    infos: Vec<TermInfo>,
    blob: Box<[u8]>,
    pub field_norms: Vec<Box<[u16]>>,
    store_blocks: Vec<Box<[u8]>>,
    /// docid-indexed external ids; empty when the segment has no ids at all
    ids: Vec<Option<String>>,
    /// docids sorted by their id (binary-search forward index); empty iff ids is empty
    id_fwd: Vec<u32>,
}

impl SegmentReader {
    /// shallow re-identity: every field clones (all small metadata vecs/blobs
    /// copied) so the old Arc can stay shared. Only used on zapseg1/2 boot
    /// segments, whose bytes carried no segid (assigned fresh at boot).
    pub(crate) fn reidentify(&self, segid: u32) -> Self {
        Self {
            segid,
            num_docs: self.num_docs,
            fsts: self.fsts.clone(),
            fst_raw: self.fst_raw.clone(),
            content_hash: std::sync::OnceLock::new(),
            infos: self.infos.clone(),
            blob: self.blob.clone(),
            field_norms: self.field_norms.clone(),
            store_blocks: self.store_blocks.clone(),
            ids: self.ids.clone(),
            id_fwd: self.id_fwd.clone(),
        }
    }
}

// -- disk serialization (opt-in --disk) --------------------------------------

const SEG_MAGIC: &[u8; 8] = b"zapseg2\0";
/// v1 kept bootable: 5 info ranges (the vestigial always-empty tf section)
const SEG_MAGIC_V1: &[u8; 8] = b"zapseg1\0";

impl SegmentReader {
    /// full on-disk serialization (little-endian sections; immutable snapshot).
    /// Layout: [magic][num_docs u32][nf u32] per-field [fst len u32][fst]
    /// [n_infos u32] per-info 36B [df u32 + 4 ranges x (start,end)]
    /// [blob len u32][blob] per fid [norm len u32][u16*n]
    /// [n_store u32]([len u32][bytes]) [n_ids u32]([has u8][len u32][bytes])
    pub fn to_bytes(&self) -> Vec<u8> {
        fn push_u32(out: &mut Vec<u8>, v: u32) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        fn push_opt_string(out: &mut Vec<u8>, s: Option<&str>) {
            match s {
                Some(t) => {
                    push_u32(out, 1);
                    push_u32(out, t.len() as u32);
                    out.extend_from_slice(t.as_bytes());
                }
                None => push_u32(out, 0),
            }
        }
        let mut out = Vec::new();
        out.extend_from_slice(SEG_MAGIC);
        // v3: stable segment identity survives disk round-trips (subset tier
        // merges must not renumber live segments; tombstones/id addresses key on it)
        push_u32(&mut out, self.segid);
        push_u32(&mut out, self.num_docs);
        push_u32(&mut out, self.fsts.len() as u32);
        for bytes in self.fst_raw.iter() {
            push_u32(&mut out, bytes.len() as u32);
            out.extend_from_slice(bytes);
        }
        push_u32(&mut out, self.infos.len() as u32);
        for i in &self.infos {
            push_u32(&mut out, i.df);
            for r in [&i.docids, &i.pos, &i.pos_offsets, &i.skips] {
                push_u32(&mut out, r.start);
                push_u32(&mut out, r.end);
            }
        }
        push_u32(&mut out, self.blob.len() as u32);
        out.extend_from_slice(&self.blob);
        for norms in &self.field_norms {
            push_u32(&mut out, norms.len() as u32);
            for n in norms.iter() {
                out.extend_from_slice(&n.to_le_bytes());
            }
        }
        push_u32(&mut out, self.store_blocks.len() as u32);
        for b in &self.store_blocks {
            push_u32(&mut out, b.len() as u32);
            out.extend_from_slice(b);
        }
        push_u32(&mut out, self.ids.len() as u32);
        for id in &self.ids {
            push_opt_string(&mut out, id.as_deref());
        }
        self.content_hash.set(crate::cluster::fnv64a(&out)).ok();
        out
    }

    /// memoized content hash; None until the first to_bytes (or a priming
    /// call at load time — segments are immutable so the value is stable)
    pub fn content_hash(&self) -> Option<u64> {
        self.content_hash.get().copied()
    }

    /// prime the memoized hash from already-known serialized bytes (load path:
    /// the file bytes ARE the to_bytes output, free hash without re-serializing)
    pub fn prime_content_hash(&self, hash: u64) {
        self.content_hash.set(hash).ok();
    }

    /// mirror image of `to_bytes`; id_fwd is rebuilt on load
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut p = 0usize;
        if bytes.len() < 8 || &bytes[..8] != SEG_MAGIC {
            return Err("bad segment magic".into());
        }
        p += 8;
        let is_v3 = &bytes[..8] == SEG_MAGIC;
        let mut segid = 0u32;
        if is_v3 {
            if bytes.len() < p + 4 {
                return Err("segment truncated (segid)".into());
            }
            segid = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
            p += 4;
        }
        if bytes.len() < p + 4 {
            return Err("segment truncated (num_docs)".into());
        }
        let num_docs = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
        p += 4;
        if bytes.len() < p + 4 {
            return Err("segment truncated (nf)".into());
        }
        let nf = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        let mut fsts = Vec::with_capacity(nf);
        let mut fst_raws: Vec<Vec<u8>> = Vec::with_capacity(nf);
        for _ in 0..nf {
            if bytes.len() < p + 4 {
                return Err("segment truncated (fst len)".into());
            }
            let len = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            if bytes.len() < p + len {
                return Err("segment truncated (fst)".into());
            }
            let raw = bytes[p..p + len].to_vec();
            fsts.push(Map::new(raw.clone()).map_err(|e| format!("fst: {e}"))?);
            fst_raws.push(raw);
            p += len;
        }
        if bytes.len() < p + 4 {
            return Err("segment truncated (n_infos)".into());
        }
        let n_infos = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        let mut infos = Vec::with_capacity(n_infos);
        for _ in 0..n_infos {
            if bytes.len() < p + 4 {
                return Err("segment truncated (info df)".into());
            }
            let df = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
            p += 4;
            let nranges = if &bytes[..8] == SEG_MAGIC_V1 { 5 } else { 4 };
            let mut starts = [0u32; 5];
            let mut ends = [0u32; 5];
            for s in 0..nranges {
                if bytes.len() < p + 8 {
                    return Err("segment truncated (info ranges)".into());
                }
                starts[s] = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
                ends[s] = u32::from_le_bytes(bytes[p + 4..p + 8].try_into().unwrap());
                p += 8;
            }
            // v1 index 1 is the vestigial empty tf-stream range: skipped
            let sh = |k: usize| -> usize {
                if &bytes[..8] == SEG_MAGIC_V1 && k >= 1 {
                    k + 1
                } else {
                    k
                }
            };
            infos.push(TermInfo {
                df,
                docids: starts[sh(0)]..ends[sh(0)],
                pos: starts[sh(1)]..ends[sh(1)],
                pos_offsets: starts[sh(2)]..ends[sh(2)],
                skips: starts[sh(3)]..ends[sh(3)],
            });
        }
        if bytes.len() < p + 4 {
            return Err("segment truncated (blob len)".into());
        }
        let blob_len = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        if bytes.len() < p + blob_len {
            return Err("segment truncated (blob)".into());
        }
        let blob = bytes[p..p + blob_len].to_vec().into_boxed_slice();
        p += blob_len;
        let mut field_norms = Vec::with_capacity(nf);
        for _ in 0..nf {
            if bytes.len() < p + 4 {
                return Err("segment truncated (norm len)".into());
            }
            let len = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            if bytes.len() < p + len * 2 {
                return Err("segment truncated (norms)".into());
            }
            let mut v = Vec::with_capacity(len);
            for i in 0..len {
                v.push(u16::from_le_bytes(
                    bytes[p + i * 2..p + i * 2 + 2].try_into().unwrap(),
                ));
            }
            field_norms.push(v.into_boxed_slice());
            p += len * 2;
        }
        if bytes.len() < p + 4 {
            return Err("segment truncated (n_store)".into());
        }
        let n_store = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        let mut store_blocks = Vec::with_capacity(n_store);
        for _ in 0..n_store {
            if bytes.len() < p + 4 {
                return Err("segment truncated (store len)".into());
            }
            let len = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            if bytes.len() < p + len {
                return Err("segment truncated (store)".into());
            }
            store_blocks.push(bytes[p..p + len].to_vec().into_boxed_slice());
            p += len;
        }
        if bytes.len() < p + 4 {
            return Err("segment truncated (n_ids)".into());
        }
        let n_ids = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        let mut ids: Vec<Option<String>> = Vec::with_capacity(n_ids);
        for _ in 0..n_ids {
            if bytes.len() < p + 4 {
                return Err("segment truncated (id tag)".into());
            }
            let has = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
            p += 4;
            if has == 1 {
                if bytes.len() < p + 4 {
                    return Err("segment truncated (id len)".into());
                }
                let len = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
                p += 4;
                if bytes.len() < p + len {
                    return Err("segment truncated (id bytes)".into());
                }
                let t = std::str::from_utf8(&bytes[p..p + len])
                    .map_err(|e| e.to_string())?
                    .to_owned();
                ids.push(Some(t));
                p += len;
            } else {
                ids.push(None);
            }
        }
        let mut reader = SegmentReader {
            segid,
            num_docs,
            fsts,
            fst_raw: fst_raws.into_boxed_slice(),
            infos,
            blob,
            field_norms,
            store_blocks,
            ids,
            id_fwd: Vec::new(),
            content_hash: std::sync::OnceLock::new(),
        };
        let fwd = rebuild_id_fwd(&reader.ids, reader.num_docs);
        reader.id_fwd = fwd.1;
        reader.ids = fwd.0;
        Ok(reader)
    }
}

/// shared by freeze_parts and from_bytes: an (ids, id_fwd) pair (ids empty = no ids)
pub(crate) fn rebuild_id_fwd(
    ids: &[Option<String>],
    num_docs: u32,
) -> (Vec<Option<String>>, Vec<u32>) {
    let any = ids.iter().any(Option::is_some);
    if !any {
        return (Vec::new(), Vec::new());
    }
    let mut fwd: Vec<u32> = (0..num_docs)
        .filter(|d| ids[*d as usize].is_some())
        .collect();
    fwd.sort_unstable_by(|a, b| {
        ids[*a as usize]
            .as_deref()
            .unwrap()
            .cmp(ids[*b as usize].as_deref().unwrap())
    });
    (ids.to_vec(), fwd)
}

/// shared by SegmentBuilder::freeze and segment merge
pub fn freeze_parts(
    postings: Vec<HashMap<String, PostingData>>,
    field_norms: Vec<Vec<u16>>,
    store_blocks: Vec<Box<[u8]>>,
    ids: Vec<Option<String>>,
    num_docs: u32,
) -> SegmentReader {
    let mut blob: Vec<u8> = Vec::new();
    let mut infos: Vec<TermInfo> = Vec::new();
    let mut fsts = Vec::with_capacity(postings.len());
    let mut fst_raws: Vec<Vec<u8>> = Vec::with_capacity(postings.len());
    for (fid, terms) in postings.iter().enumerate() {
        let norms = field_norms.get(fid).map(|v| v.as_slice());
        // HashMap iteration order is arbitrary -> sort by term for the FST
        let mut sorted: Vec<(&String, &PostingData)> = terms.iter().collect();
        sorted.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        // R22: parallel per-term encoding. Each worker chunk freezes+encodes
        // its own sub-blob (sections are self-contained), the serial fold
        // appends the bytes and rebases section ranges; FST inserts stay
        // sequential (ordinals are the global infos order).
        let encoded: Vec<(Vec<u8>, Vec<TermInfo>, Vec<&String>)> = {
            use rayon::prelude::*;
            let workers = rayon::current_num_threads().max(1);
            let chunk = sorted.len().div_ceil(workers).max(1);
            sorted
                .par_chunks(chunk)
                .map(|part| {
                    let mut sub = Vec::new();
                    let mut local: Vec<TermInfo> = Vec::with_capacity(part.len());
                    for (_, data) in part {
                        let info = data.freeze(norms).encode_to(&mut sub);
                        local.push(info);
                    }
                    let part_terms: Vec<&String> = part.iter().map(|(t, _)| *t).collect();
                    (sub, local, part_terms)
                })
                .collect()
        };
        let mut map_builder = MapBuilder::new(Vec::new()).unwrap();
        for (sub, mut local, part_terms) in encoded {
            let base = blob.len() as u32;
            for info in local.iter_mut() {
                info.docids.start += base;
                info.docids.end += base;
                info.pos.start += base;
                info.pos.end += base;
                info.pos_offsets.start += base;
                info.pos_offsets.end += base;
                info.skips.start += base;
                info.skips.end += base;
            }
            blob.extend_from_slice(&sub);
            for (i, info) in local.iter().enumerate() {
                let term = part_terms[i];
                let ordinal = infos.len() as u64;
                map_builder.insert(term.as_bytes(), ordinal).unwrap();
                infos.push(info.clone());
            }
        }
        let fst_bytes = map_builder.into_inner().unwrap();
        fst_raws.push(fst_bytes.clone());
        fsts.push(Map::new(fst_bytes).unwrap());
    }
    // forward id index: only materialized when at least one doc carries an id
    let (ids, id_fwd) = rebuild_id_fwd(&ids, num_docs);
    SegmentReader {
        segid: 0,
        num_docs,
        fsts,
        fst_raw: fst_raws.into_boxed_slice(),
        infos,
        blob: blob.into_boxed_slice(),
        field_norms: field_norms
            .into_iter()
            .map(|v| v.into_boxed_slice())
            .collect(),
        store_blocks,
        ids,
        id_fwd,
        content_hash: std::sync::OnceLock::new(),
    }
}

impl SegmentReader {
    /// raw (still compressed) store block for one doc
    pub fn stored_raw(&self, docid: u32) -> Option<&[u8]> {
        self.store_blocks.get(docid as usize).map(|b| b.as_ref())
    }

    /// external id of a segment-local docid (cheap: one Vec index)
    pub fn id_of(&self, docid: u32) -> Option<&str> {
        self.ids.get(docid as usize).and_then(|o| o.as_deref())
    }

    /// true when at least one doc in this segment carries an external id
    pub fn has_ids(&self) -> bool {
        !self.ids.is_empty()
    }

    /// forward id lookup: binary search over docids sorted by id (O(log n) compares)
    pub fn lookup_id(&self, id: &str) -> Option<u32> {
        if self.id_fwd.is_empty() {
            return None;
        }
        self.id_fwd
            .binary_search_by(|d| self.ids[*d as usize].as_deref().unwrap_or("").cmp(id))
            .ok()
            .map(|i| self.id_fwd[i])
    }

    pub fn term(&self, fid: u32, term: &str) -> Option<TermInfo> {
        let fst = self.fsts.get(fid as usize)?;
        let ord = fst.get(term)?;
        Some(self.infos[ord as usize].clone())
    }

    pub fn blob(&self) -> &[u8] {
        &self.blob
    }

    pub fn reader(&self, info: &TermInfo) -> PostingReader<'_> {
        PostingReader::new(info, &self.blob)
    }

    /// exact + fuzzy: DFA-pruned walk over the field's sorted dict
    pub fn stream_automaton<A>(&self, aut: A, fid: u32, limit: usize) -> Vec<(String, TermInfo)>
    where
        A: Automaton,
        A::State: Clone,
    {
        let mut out = Vec::new();
        let fst = match self.fsts.get(fid as usize) {
            Some(f) => f,
            None => return out,
        };
        let mut st = fst.search_with_state(aut).into_stream();
        while let Some((key, ord, _state)) = st.next() {
            if out.len() >= limit {
                break;
            }
            if let Ok(t) = std::str::from_utf8(key) {
                out.push((t.to_owned(), self.infos[ord as usize].clone()));
            }
        }
        out
    }

    /// prefix terms via FST range query [prefix, prefix+1)
    pub fn stream_prefix(&self, fid: u32, p: &str, limit: usize) -> Vec<(String, TermInfo)> {
        let mut out = Vec::new();
        let fst = match self.fsts.get(fid as usize) {
            Some(f) => f,
            None => return out,
        };
        // upper bound: first string after every key with prefix `p`
        let mut upper = p.as_bytes().to_vec();
        while let Some(last) = upper.last_mut() {
            if *last == u8::MAX {
                upper.pop();
            } else {
                *last += 1;
                break;
            }
        }
        let rb = fst.range().ge(p);
        let mut st = if upper.is_empty() {
            rb.into_stream()
        } else {
            rb.lt(&upper).into_stream()
        };
        while let Some((key, ord)) = st.next() {
            if out.len() >= limit {
                break;
            }
            if let Ok(t) = std::str::from_utf8(key) {
                out.push((t.to_owned(), self.infos[ord as usize].clone()));
            }
        }
        out
    }

    /// plain sorted iteration over one field's dict (merge paths)
    pub fn stream_terms(&self, fid: u32) -> Vec<(String, TermInfo)> {
        let mut out = Vec::new();
        let fst = match self.fsts.get(fid as usize) {
            Some(f) => f,
            None => return out,
        };
        let mut stream = fst.stream();
        while let Some((key, ord)) = stream.next() {
            if let Ok(t) = std::str::from_utf8(key) {
                out.push((t.to_owned(), self.infos[ord as usize].clone()));
            }
        }
        out
    }

    pub fn norms(&self, fid: u32) -> Option<&[u16]> {
        self.field_norms.get(fid as usize).map(|v| v.as_ref())
    }

    pub fn sum_field_len(&self, fid: u32) -> u64 {
        self.norms(fid)
            .map(|v| v.iter().map(|x| *x as u64).sum())
            .unwrap_or(0)
    }

    pub fn stored_doc(&self, docid: u32) -> Option<StoredDoc> {
        let block = self.store_blocks.get(docid as usize)?;
        read_stored(block)
    }
}

/// decompress + decode one store block:
/// [tag][len u32][flags u8][n u8][TLV(fid u32, len u32, text)][id len u32 + utf8][meta len u32 + json]
pub fn read_stored(block: &[u8]) -> Option<StoredDoc> {
    let raw: std::borrow::Cow<[u8]> = match block.first() {
        None => return None,
        Some(&STORE_RAW_TAG) => {
            // raw layout: tag + le-u32 length + payload
            let len = u32::from_le_bytes(block.get(1..5)?.try_into().ok()?) as usize;
            std::borrow::Cow::Owned(block.get(5..5 + len)?.to_vec())
        }
        Some(&STORE_COMP_TAG) => {
            std::borrow::Cow::Owned(decompress_size_prepended(&block[1..]).ok()?)
        }
        _ => return None,
    };
    let flags = *raw.first()?;
    let mut p = 1;
    let n = *raw.get(p)? as usize;
    p += 1;
    let mut fields = Vec::with_capacity(n);
    for _ in 0..n {
        let fid = u32::from_le_bytes(raw.get(p..p + 4)?.try_into().ok()?);
        p += 4;
        let l = u32::from_le_bytes(raw.get(p..p + 4)?.try_into().ok()?) as usize;
        p += 4;
        let val = String::from_utf8(raw.get(p..p + l)?.to_vec()).ok()?;
        p += l;
        fields.push((fid, val));
    }
    let id = if flags & BLOCK_FLAG_ID != 0 {
        let l = u32::from_le_bytes(raw.get(p..p + 4)?.try_into().ok()?) as usize;
        p += 4;
        let val = String::from_utf8(raw.get(p..p + l)?.to_vec()).ok()?;
        p += l;
        Some(val)
    } else {
        None
    };
    let metadata = if flags & BLOCK_FLAG_META != 0 {
        let l = u32::from_le_bytes(raw.get(p..p + 4)?.try_into().ok()?) as usize;
        p += 4;
        let val: serde_json::Value = serde_json::from_slice(raw.get(p..p + l)?).ok()?;
        Some(val)
    } else {
        None
    };
    Some(StoredDoc {
        id,
        fields,
        metadata,
    })
}
