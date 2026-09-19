//! index: writer (add/commit/merge) + snapshot readers

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;

use crate::analyzer::Analyzer;
use crate::posting::PostingData;
use crate::schema::{Document, Schema};
use crate::search::exec::Stats;
use crate::search::{Bm25Params, DocAddr, SearchHit, TopK};
use crate::segment::{SegmentBuilder, SegmentReader};

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("unknown field `{0}`")]
    UnknownField(String),
    #[error("query error")]
    Query(String),
}

/// when a commit pushes segments to this count, merge them all into one
const MAX_SEGMENTS: usize = 4;

pub struct IndexSnapshot {
    pub segments: Vec<Arc<SegmentReader>>,
    pub stats: Stats,
    pub generation: u64,
}

pub struct Index {
    schema: Arc<Schema>,
    analyzer: Arc<Analyzer>,
    k1: f32,
    b: f32,
    snap: ArcSwap<IndexSnapshot>,
    inner: Mutex<IndexInner>,
}

struct IndexInner {
    hot: SegmentBuilder,
    segments: Vec<Arc<SegmentReader>>,
    generation: u64,
}

impl Index {
    pub fn new(schema: Schema, analyzer: Arc<Analyzer>, k1: f32, b: f32) -> Arc<Self> {
        let schema = Arc::new(schema);
        let hot = SegmentBuilder::new(schema.clone(), analyzer.clone());
        Arc::new(Self {
            schema,
            analyzer,
            k1,
            b,
            snap: ArcSwap::from_pointee(IndexSnapshot {
                segments: Vec::new(),
                stats: Stats::empty(),
                generation: 0,
            }),
            inner: Mutex::new(IndexInner {
                hot,
                segments: Vec::new(),
                generation: 0,
            }),
        })
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    pub fn analyzer(&self) -> &Analyzer {
        &self.analyzer
    }

    pub fn params(&self) -> Bm25Params {
        Bm25Params {
            k1: self.k1,
            b: self.b,
        }
    }

    pub fn add_document(&self, doc: &Document) -> Result<(), IndexError> {
        let mut g = self.inner.lock().unwrap();
        g.hot
            .add_document(doc)
            .map(|_| ())
            .map_err(IndexError::UnknownField)
    }

    pub fn commit(&self) {
        let mut g: std::sync::MutexGuard<'_, IndexInner> = self.inner.lock().unwrap();
        if g.hot.num_docs == 0 {
            return;
        }
        let builder = std::mem::replace(
            &mut g.hot,
            SegmentBuilder::new(self.schema.clone(), self.analyzer.clone()),
        );
        let seg = builder.freeze();
        let mut segs = std::mem::take(&mut g.segments);
        segs.push(Arc::new(seg));
        if segs.len() >= MAX_SEGMENTS {
            let merged = merge_segments(&segs, self.schema.num_fields());
            segs = vec![Arc::new(merged)];
        }
        g.segments = segs;
        g.generation += 1;
        self.publish_snapshot(&g);
    }

    fn publish_snapshot(&self, g: &IndexInner) {
        let nf = self.schema.num_fields();
        let mut total_docs: u64 = 0;
        let mut sums: Vec<u64> = vec![0u64; nf];
        let mut df: std::collections::HashMap<(u32, String), u32> =
            std::collections::HashMap::new();
        for s in &g.segments {
            total_docs += s.num_docs as u64;
            for fid in 0..nf as u32 {
                if let Some(norms) = s.norms(fid) {
                    sums[fid as usize] += norms.iter().map(|x| *x as u64).sum::<u64>();
                }
                for (term, info) in s.stream_terms(fid) {
                    *df.entry((fid, term)).or_insert(0) += info.df;
                }
            }
        }
        let snapshot = IndexSnapshot {
            segments: g.segments.clone(),
            stats: Stats {
                total_docs: total_docs as u32,
                avg_len: sums
                    .iter()
                    .map(|s| {
                        if total_docs > 0 {
                            *s as f32 / total_docs as f32
                        } else {
                            1.0
                        }
                    })
                    .collect(),
                df,
            },
            generation: g.generation,
        };
        self.snap.store(Arc::new(snapshot));
    }

    pub fn search(&self, q: &str, k: usize) -> Result<Vec<SearchHit>, IndexError> {
        let node = crate::query::parse(q).map_err(|e| IndexError::Query(format!("{e:?}")))?;
        let snap = self.snap.load();
        let params = Bm25Params {
            k1: self.k1,
            b: self.b,
        };
        let mut collector = TopK::new(k);
        for (si, seg) in snap.segments.iter().enumerate() {
            let ctx = crate::search::exec::Ctx {
                schema: &self.schema,
                stats: &snap.stats,
                params: &params,
            };
            let ds = crate::search::exec::eval(&node, None, seg, &ctx);
            for (i, doc) in ds.ids.iter().enumerate() {
                collector.push(
                    DocAddr {
                        segment: si as u32,
                        docid: *doc,
                    },
                    ds.scores[i],
                );
            }
        }
        Ok(collector.into_sorted())
    }

    pub fn stored_fields(&self, addr: DocAddr) -> Option<Vec<(String, String)>> {
        let snap = self.snap.load();
        let seg = snap.segments.get(addr.segment as usize)?;
        let pairs = seg.stored_doc(addr.docid)?;
        Some(
            pairs
                .into_iter()
                .map(|(fid, text)| {
                    let name = self
                        .schema
                        .field(fid)
                        .map(|f| f.name.clone())
                        .unwrap_or_default();
                    (name, text)
                })
                .collect(),
        )
    }

    pub fn num_docs(&self) -> usize {
        self.snap.load().stats.total_docs as usize
    }

    pub fn num_segments(&self) -> usize {
        self.snap.load().segments.len()
    }

    pub fn generation(&self) -> u64 {
        self.snap.load().generation
    }
}

/// concatenate segments into one (rebuild postings with remapped docids)
fn merge_segments(segs: &[Arc<SegmentReader>], nf: usize) -> SegmentReader {
    let mut postings: Vec<BTreeMap<String, PostingData>> =
        (0..nf).map(|_| BTreeMap::new()).collect();
    let mut field_norms: Vec<Vec<u16>> = (0..nf).map(|_| Vec::new()).collect();
    let mut store_blocks: Vec<Box<[u8]>> = Vec::new();
    let mut total = 0u32;
    for s in segs {
        let base = total;
        total += s.num_docs;
        for (fid, norms) in s.field_norms.iter().enumerate() {
            field_norms[fid].extend(norms.iter().copied());
        }
        for docid in 0..s.num_docs {
            if let Some(block) = s.stored_raw(docid) {
                store_blocks.push(block.to_vec().into_boxed_slice());
            }
        }
        for (fid, terms) in s.inner_postings() {
            let map = &mut postings[fid as usize];
            for (term, info) in terms {
                let entry = map.entry(term.clone()).or_default();
                let mut r = s.reader(&info);
                let mut idx = 0usize;
                while let Some((doc, tf)) = r.advance() {
                    let pos = r.positions(idx).unwrap_or_default();
                    idx += 1;
                    entry.add(doc + base, tf, pos);
                }
            }
        }
    }
    crate::segment::freeze_parts(postings, field_norms, store_blocks, total)
}

impl SegmentReader {
    pub fn inner_postings(&self) -> Vec<(u32, Vec<(String, crate::posting::TermInfo)>)> {
        (0..self.field_norms.len() as u32)
            .map(|fid| (fid, self.stream_terms(fid)))
            .collect()
    }
}
