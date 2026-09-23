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
                let e = match self.postings[fid as usize].get_mut(tok.text.as_str()) {
                    Some(e) => e,
                    None => self.postings[fid as usize]
                        .entry(tok.text.clone())
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
    pub num_docs: u32,
    /// per-field term dictionaries; global infos offset base is tracked per field
    fsts: Vec<Map<Vec<u8>>>,
    infos: Vec<TermInfo>,
    blob: Box<[u8]>,
    pub field_norms: Vec<Box<[u16]>>,
    store_blocks: Vec<Box<[u8]>>,
    /// docid-indexed external ids; empty when the segment has no ids at all
    ids: Vec<Option<String>>,
    /// docids sorted by their id (binary-search forward index); empty iff ids is empty
    id_fwd: Vec<u32>,
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
    for (fid, terms) in postings.iter().enumerate() {
        let norms = field_norms.get(fid).map(|v| v.as_slice());
        let mut map_builder = MapBuilder::new(Vec::new()).unwrap();
        // sorted by term for the FST (HashMap keeps insertion order)
        let mut sorted: Vec<(&String, &PostingData)> = terms.iter().collect();
        sorted.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        for (term, data) in sorted {
            let info = data.freeze(norms).encode_to(&mut blob);
            let ordinal = infos.len() as u64;
            map_builder.insert(term.as_bytes(), ordinal).unwrap();
            infos.push(info);
        }
        fsts.push(Map::new(map_builder.into_inner().unwrap()).unwrap());
    }
    // forward id index: only materialized when at least one doc carries an id
    let any_ids = ids.iter().any(Option::is_some);
    let (ids, id_fwd) = if any_ids {
        let mut fwd: Vec<u32> = (0..num_docs)
            .filter(|d| ids[*d as usize].is_some())
            .collect();
        fwd.sort_unstable_by(|a, b| {
            ids[*a as usize]
                .as_deref()
                .unwrap()
                .cmp(ids[*b as usize].as_deref().unwrap())
        });
        (ids, fwd)
    } else {
        (Vec::new(), Vec::new())
    };
    SegmentReader {
        num_docs,
        fsts,
        infos,
        blob: blob.into_boxed_slice(),
        field_norms: field_norms
            .into_iter()
            .map(|v| v.into_boxed_slice())
            .collect(),
        store_blocks,
        ids,
        id_fwd,
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

    pub fn num_fields(&self) -> usize {
        self.fsts.len()
    }

    pub fn num_terms(&self, fid: u32) -> usize {
        self.fsts.get(fid as usize).map_or(0, |f| f.len())
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
