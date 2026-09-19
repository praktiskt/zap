//! segment: immutable per-segment inverted index + compressed doc store

use std::collections::BTreeMap;

use fst::{Map, MapBuilder, Streamer};
use lz4_flex::{compress_prepend_size, decompress_size_prepended};

use crate::analyzer::Analyzer;
use crate::posting::{PostingData, PostingReader, TermInfo};
use crate::schema::{Document, Schema};
use std::sync::Arc;

/// single doc store block: count + TLV(field_id, len, text bytes)
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
    block.push(pairs.len() as u8);
    for (fid, val) in pairs {
        block.extend_from_slice(&fid.to_le_bytes());
        block.extend_from_slice(&(val.len() as u32).to_le_bytes());
        block.extend_from_slice(val.as_bytes());
    }
    Ok(block)
}

pub struct SegmentBuilder<'a> {
    schema: &'a Schema,
    analyzer: Arc<Analyzer>,
    pub num_docs: u32,
    /// field_id -> term -> postings; sorted terms feed the FST directly
    postings: Vec<BTreeMap<String, PostingData>>,
    /// field_id -> doc -> token count
    field_norms: Vec<Vec<u16>>,
    /// lz4-compressed blocks, one per doc
    store_blocks: Vec<Box<[u8]>>,
}

impl<'a> SegmentBuilder<'a> {
    pub fn new(schema: &'a Schema, analyzer: Arc<Analyzer>) -> Self {
        let nf = schema.num_fields();
        let mut postings = Vec::with_capacity(nf);
        let mut field_norms = Vec::with_capacity(nf);
        for _ in 0..nf {
            postings.push(BTreeMap::new());
            field_norms.push(Vec::new());
        }
        Self {
            schema,
            analyzer,
            num_docs: 0,
            postings,
            field_norms,
            store_blocks: Vec::new(),
        }
    }

    pub fn sum_field_len(&self, fid: u32) -> u64 {
        self.field_norms
            .get(fid as usize)
            .map(|v| v.iter().map(|x| *x as u64).sum())
            .unwrap_or(0)
    }

    /// index + store a document; allocates the segment-local docid
    pub fn add_document(&mut self, doc: &Document) -> Result<u32, String> {
        let docid = self.num_docs;
        let block = encode_store(doc, self.schema)?;
        for (name, val) in &doc.fields {
            let fid = self
                .schema
                .field_id(name)
                .ok_or_else(|| format!("unknown field `{name}`"))?;
            let mut pos = 0u32;
            for tok in self.analyzer.tokenize(val) {
                let e = self.postings[fid as usize]
                    .entry(tok.text.clone())
                    .or_default();
                if e.docids.last() == Some(&docid) {
                    // repeated token within the same doc: bump tf, extend positions
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
        self.store_blocks.push(compress_prepend_size(&block).into());
        self.num_docs += 1;
        Ok(docid)
    }

    pub fn freeze(self) -> SegmentReader {
        let mut map_builder = MapBuilder::new(Vec::new()).unwrap();
        let mut blob: Vec<u8> = Vec::new();
        let mut infos: Vec<TermInfo> = Vec::new();
        for (fid, terms) in self.postings.iter().enumerate() {
            for (term, data) in terms {
                let mut key = (fid as u32).to_be_bytes().to_vec();
                key.extend_from_slice(term.as_bytes());
                let info = data.freeze().encode_to(&mut blob);
                let ordinal = infos.len() as u64;
                map_builder.insert(key, ordinal).unwrap();
                infos.push(info);
            }
        }
        let field_norms: Vec<Box<[u16]>> = self
            .field_norms
            .into_iter()
            .map(|v| v.into_boxed_slice())
            .collect();
        SegmentReader {
            num_docs: self.num_docs,
            fst: Map::new(map_builder.into_inner().unwrap()).unwrap(),
            infos,
            blob: blob.into_boxed_slice(),
            field_norms,
            store_blocks: self.store_blocks,
        }
    }
}

pub struct SegmentReader {
    pub num_docs: u32,
    fst: Map<Vec<u8>>,
    infos: Vec<TermInfo>,
    blob: Box<[u8]>,
    pub field_norms: Vec<Box<[u16]>>,
    store_blocks: Vec<Box<[u8]>>,
}

impl SegmentReader {
    pub fn term(&self, fid: u32, term: &str) -> Option<TermInfo> {
        let mut key = fid.to_be_bytes().to_vec();
        key.extend_from_slice(term.as_bytes());
        let ord = self.fst.get(&key)?;
        Some(self.infos[ord as usize].clone())
    }

    pub fn blob(&self) -> &[u8] {
        &self.blob
    }

    pub fn reader(&self, info: &TermInfo) -> PostingReader<'_> {
        PostingReader::new(info, &self.blob)
    }

    pub fn norms(&self, fid: u32) -> Option<&[u16]> {
        self.field_norms.get(fid as usize).map(|v| v.as_ref())
    }

    pub fn sum_field_len(&self, fid: u32) -> u64 {
        self.norms(fid)
            .map(|v| v.iter().map(|x| *x as u64).sum())
            .unwrap_or(0)
    }

    /// all (term, info) pairs of one field (merge + term expansion)
    pub fn stream_terms(&self, fid: u32) -> Vec<(String, TermInfo)> {
        let mut out = Vec::new();
        let mut stream = self.fst.stream();
        while let Some((key, ord)) = stream.next() {
            if key.len() >= 4 && u32::from_be_bytes(key[..4].try_into().unwrap()) == fid {
                out.push((
                    String::from_utf8(key[4..].to_vec()).unwrap(),
                    self.infos[ord as usize].clone(),
                ));
            }
        }
        out
    }

    pub fn num_terms(&self) -> usize {
        self.infos.len()
    }

    pub fn stored_doc(&self, docid: u32) -> Option<Vec<(u32, String)>> {
        let block = self.store_blocks.get(docid as usize)?;
        read_stored(block)
    }
}

/// decompress one stored doc block
pub fn read_stored(block: &[u8]) -> Option<Vec<(u32, String)>> {
    let raw = decompress_size_prepended(block).ok()?;
    let n = *raw.first()? as usize;
    let mut p = 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let fid = u32::from_le_bytes(raw.get(p..p + 4)?.try_into().ok()?);
        p += 4;
        let l = u32::from_le_bytes(raw.get(p..p + 4)?.try_into().ok()?) as usize;
        p += 4;
        let val = String::from_utf8(raw.get(p..p + l)?.to_vec()).ok()?;
        p += l;
        out.push((fid, val));
    }
    Some(out)
}
