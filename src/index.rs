//! index: writer (add/put/delete/commit/merge) + snapshot readers
//!
//! external ids: unique across live docs; duplicate ingest replaces (tombstone old,
//! new doc wins). Tombstones mask immutable segments at query time and are physically
//! purged by the log-tier merge.

use crate::query::Node;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;
use roaring::RoaringBitmap;

use crate::analyzer::Analyzer;
use crate::posting::PostingData;
use crate::schema::{Document, Schema, StoredDoc};
use crate::search::exec::Stats;
use crate::search::{Bm25Params, DocAddr, SearchHit, TopK};
use crate::segment::{SegmentBuilder, SegmentReader};

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("unknown field `{0}`")]
    UnknownField(String),
    #[error("invalid document id `{0}`")]
    BadId(String),
    #[error("query error")]
    Query(String),
}

/// when a commit pushes segments to this count, merge them all into one.
/// Value chosen so that the bench corpora (10 commits x 8 lanes = 80 builder
/// segments) keep their historical shape: trigger 64 -> merge once -> 17 final
/// segments at 8 lanes, exactly like the previous 4-segment budget produced.
const MAX_SEGMENTS: usize = 8;

/// cached publish stats: (per-field length sums, global df, total docs)
type StatsCache = (Vec<u64>, Arc<HashMap<(u32, String), u32>>, u64);

/// resolved store payload for one document
#[derive(Debug, Clone)]
pub struct DocPayload {
    pub addr: DocAddr,
    pub id: Option<String>,
    /// (field name, text) pairs
    pub fields: Vec<(String, String)>,
    pub metadata: Option<serde_json::Value>,
}

pub struct IndexSnapshot {
    pub segments: Vec<Arc<SegmentReader>>,
    pub stats: Stats,
    /// per-segment tombstone masks (segment index -> deleted docids)
    pub deleted: Arc<HashMap<u32, RoaringBitmap>>,
    /// live external id -> address (tombstoned docs excluded)
    pub ids: Arc<HashMap<String, DocAddr>>,
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
    /// per-lane builders (round-robin docid partitioning for parallel ingest)
    hot: Vec<SegmentBuilder>,
    next_lane: usize,
    segments: Vec<Arc<SegmentReader>>,
    /// tombstones over `segments` (segment index -> deleted docids); purged at merge
    deleted: HashMap<u32, RoaringBitmap>,
    /// live external id -> committed address (hot ids live in `hot_ids` until commit)
    ids: HashMap<String, DocAddr>,
    /// Arc-shared id map for tombstone-only snapshot publishes (no segment change)
    ids_arc: Arc<HashMap<String, DocAddr>>,
    /// (sums, df, total_docs) cached between segment changes (tombstone-only publishes)
    stats_cache: Option<StatsCache>,
    /// ids of docs pending in hot lanes (uncommitted, invisible to readers)
    hot_ids: HashSet<String>,
    generation: u64,
}

impl Index {
    pub fn new(schema: Schema, analyzer: Arc<Analyzer>, k1: f32, b: f32) -> Arc<Self> {
        Self::with_lanes(schema, analyzer, k1, b, 1)
    }

    /// parallel ingest: docs are round-robined across `lanes` builders; commit freezes in parallel
    pub fn with_lanes(
        schema: Schema,
        analyzer: Arc<Analyzer>,
        k1: f32,
        b: f32,
        lanes: usize,
    ) -> Arc<Self> {
        let schema = Arc::new(schema);
        let lanes = lanes.max(1);
        let hot = (0..lanes)
            .map(|_| SegmentBuilder::new(schema.clone(), analyzer.clone()))
            .collect();
        Arc::new(Self {
            schema,
            analyzer,
            k1,
            b,
            snap: ArcSwap::from_pointee(IndexSnapshot {
                segments: Vec::new(),
                stats: Stats::empty(),
                deleted: Arc::new(HashMap::new()),
                ids: Arc::new(HashMap::new()),
                generation: 0,
            }),
            inner: Mutex::new(IndexInner {
                hot,
                next_lane: 0,
                segments: Vec::new(),
                deleted: HashMap::new(),
                ids: HashMap::new(),
                ids_arc: Arc::new(HashMap::new()),
                stats_cache: None,
                hot_ids: HashSet::new(),
                generation: 0,
            }),
        })
    }

    pub fn lane_count(&self) -> usize {
        self.inner.lock().unwrap().hot.len()
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// read-only segment list snapshot (tests/debug)
    pub fn segments_snapshot(&self) -> Vec<Arc<SegmentReader>> {
        self.snap.load().segments.clone()
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

    /// ingest one document. Duplicate external ids REPLACE: the old live doc is
    /// tombstoned, the new one wins. Uncommitted duplicates trigger a commit first
    /// (single-writer semantics; hot docs are invisible until committed).
    pub fn add_document(&self, doc: &Document) -> Result<(), IndexError> {
        let mut g = self.inner.lock().unwrap();
        self.add_locked(&mut g, doc)
    }

    /// locked single-doc ingest (replace-aware)
    fn add_locked(&self, g: &mut IndexInner, doc: &Document) -> Result<(), IndexError> {
        if let Some(id) = &doc.id {
            if id.is_empty() {
                return Err(IndexError::BadId(id.clone()));
            }
            if g.hot_ids.contains(id) {
                // old version still hot: flush it so it can be tombstoned
                self.commit_locked(g);
            }
            if let Some(old) = g.ids.get(id).copied() {
                g.deleted.entry(old.segment).or_default().insert(old.docid);
                g.ids.remove(id);
            }
        }
        let lane = g.next_lane % g.hot.len();
        g.next_lane += 1;
        g.hot[lane]
            .add_document(doc)
            .map(|_| ())
            .map_err(IndexError::UnknownField)?;
        if let Some(id) = &doc.id {
            g.hot_ids.insert(id.clone());
        }
        Ok(())
    }

    /// parallel batch ingest: docs are chunked into lanes and tokenized/frozen concurrently.
    /// Id-carrying batches that collide with existing ids fall back to the sequential
    /// replace-aware path; fresh-id batches take the parallel fast path unchanged.
    pub fn add_documents(&self, docs: Vec<Document>) -> Result<(), IndexError> {
        // de-dup inside the batch first: later docs replace earlier same-id docs
        let mut deduped: Vec<Document> = Vec::with_capacity(docs.len());
        let mut slot_of: HashMap<String, usize> = HashMap::new();
        for d in docs {
            match &d.id {
                Some(id) => match slot_of.get(id) {
                    Some(&slot) => deduped[slot] = d,
                    None => {
                        slot_of.insert(id.clone(), deduped.len());
                        deduped.push(d);
                    }
                },
                None => deduped.push(d),
            }
        }
        let collides = {
            let g = self.inner.lock().unwrap();
            deduped.iter().any(|d| {
                d.id.as_ref()
                    .is_some_and(|id| g.hot_ids.contains(id) || g.ids.contains_key(id))
            })
        };
        if collides {
            let mut g = self.inner.lock().unwrap();
            for doc in &deduped {
                self.add_locked(&mut g, doc)?;
            }
            return Ok(());
        }
        use rayon::prelude::*;
        let mut g = self.inner.lock().unwrap();
        let lanes = g.hot.len().max(1);
        let chunk = deduped.len().div_ceil(lanes).max(1);
        let mut chunks: Vec<Vec<Document>> = Vec::with_capacity(g.hot.len());
        for chunk in deduped.chunks(chunk) {
            chunks.push(chunk.to_vec());
        }
        while chunks.len() < g.hot.len() {
            chunks.push(Vec::new());
        }
        let hot_ids_new: Vec<String> = deduped.iter().filter_map(|d| d.id.clone()).collect();
        let errors: Vec<Option<String>> = {
            let builders: &mut Vec<SegmentBuilder> = &mut g.hot;
            builders
                .par_iter_mut()
                .zip(chunks.into_par_iter())
                .map(|(b, docs)| {
                    let mut first_err = None;
                    for d in docs {
                        if let Err(e) = b.add_document(&d) {
                            first_err = Some(e);
                            break;
                        }
                    }
                    first_err
                })
                .collect()
        };
        for id in hot_ids_new {
            g.hot_ids.insert(id);
        }
        if let Some(e) = errors.into_iter().flatten().next() {
            return Err(IndexError::UnknownField(e));
        }
        Ok(())
    }

    /// flush hot lanes into sealed segments (+ merge when over budget)
    fn commit_locked(&self, g: &mut IndexInner) {
        if g.hot.iter().all(|h| h.num_docs == 0) {
            return;
        }
        // swap in fresh builders; freeze the hot ones in parallel
        let frozen = std::mem::take(&mut g.hot);
        g.hot = (0..frozen.len())
            .map(|_| SegmentBuilder::new(self.schema.clone(), self.analyzer.clone()))
            .collect();
        let new_segs: Vec<Arc<SegmentReader>> = {
            use rayon::prelude::*;
            frozen
                .into_par_iter()
                .filter_map(|b| (b.num_docs > 0).then(|| Arc::new(b.freeze())))
                .collect()
        };
        let mut segs = std::mem::take(&mut g.segments);
        segs.extend(new_segs);
        let lanes = g.hot.len().max(1);
        if segs.len() >= MAX_SEGMENTS * lanes.max(1) {
            let masks_all: Vec<RoaringBitmap> = (0..segs.len())
                .map(|i| g.deleted.get(&(i as u32)).cloned().unwrap_or_default())
                .collect();
            let merged = merge_segments(&segs, &masks_all, self.schema.num_fields());
            segs = vec![Arc::new(merged)];
            g.deleted.clear();
        }
        g.segments = segs;
        g.hot_ids.clear();
        g.generation += 1;
        self.publish_snapshot(g, true);
    }

    pub fn commit(&self) {
        let mut g = self.inner.lock().unwrap();
        self.commit_locked(&mut g);
    }

    /// tombstone one live doc by external id; the tombstone is published immediately
    /// (lock-free readers see it without a segment commit) and physically purged at merge
    pub fn delete(&self, id: &str) -> Result<bool, IndexError> {
        let mut g = self.inner.lock().unwrap();
        if g.hot_ids.contains(id) {
            // uncommitted: flush so the doc becomes tombstoneable
            self.commit_locked(&mut g);
        }
        let Some(old) = g.ids.remove(id) else {
            return Ok(false);
        };
        g.deleted.entry(old.segment).or_default().insert(old.docid);
        self.publish_snapshot(&mut g, false);
        Ok(true)
    }

    /// get-by-id over the current snapshot (lock-free); uncommitted docs are invisible.
    /// `ids` may reference tombstoned docs until the next segment change — masked here.
    pub fn get(&self, id: &str) -> Option<DocPayload> {
        let snap = self.snap.load();
        let addr = *snap.ids.get(id)?;
        if snap
            .deleted
            .get(&addr.segment)
            .is_some_and(|m| m.contains(addr.docid))
        {
            return None;
        }
        self.payload_of(&snap, addr)
    }

    /// resolve stored fields + id + metadata for one address
    pub fn doc_payload(&self, addr: DocAddr) -> Option<DocPayload> {
        let snap = self.snap.load();
        self.payload_of(&snap, addr)
    }

    fn payload_of(&self, snap: &IndexSnapshot, addr: DocAddr) -> Option<DocPayload> {
        let seg = snap.segments.get(addr.segment as usize)?;
        let stored: StoredDoc = seg.stored_doc(addr.docid)?;
        let fields = stored
            .fields
            .into_iter()
            .map(|(fid, text)| {
                let name = self
                    .schema
                    .field(fid)
                    .map(|f| f.name.clone())
                    .unwrap_or_default();
                (name, text)
            })
            .collect();
        Some(DocPayload {
            addr,
            id: stored.id,
            fields,
            metadata: stored.metadata,
        })
    }

    /// publish a snapshot. `segment_change`: segments were added/merged — recompute
    /// df stats and rebuild the id map. Tombstone-only publishes (delete) reuse the
    /// cached stats (Arc-shared df) + id map; readers mask via `deleted`.
    fn publish_snapshot(&self, g: &mut IndexInner, segment_change: bool) {
        let nf = self.schema.num_fields();
        let deleted: HashMap<u32, RoaringBitmap> = g
            .deleted
            .iter()
            .filter(|(_, m)| !m.is_empty())
            .map(|(&si, m)| (si, m.clone()))
            .collect();
        let tombstoned: u64 = deleted.values().map(|m| m.len()).sum();
        let mut total_docs: u64 = 0;
        let mut sums: Vec<u64> = vec![0u64; nf];
        let mut df: Arc<HashMap<(u32, String), u32>> = Arc::new(HashMap::new());
        if segment_change {
            let mut raw_df: HashMap<(u32, String), u32> = HashMap::new();
            for s in &g.segments {
                total_docs += s.num_docs as u64;
                for fid in 0..nf as u32 {
                    if let Some(norms) = s.norms(fid) {
                        sums[fid as usize] += norms.iter().map(|x| *x as u64).sum::<u64>();
                    }
                    for (term, info) in s.stream_terms(fid) {
                        *raw_df.entry((fid, term)).or_insert(0) += info.df;
                    }
                }
            }
            df = Arc::new(raw_df);
            g.stats_cache = Some((sums.clone(), df.clone(), total_docs));
        } else {
            // segments unchanged: reuse cached totals/sums/df (Arc clone, no deep copy)
            if let Some((s, d, t)) = &g.stats_cache {
                sums = s.clone();
                df = d.clone();
                total_docs = *t;
            }
        }
        // live external id -> address. Rebuilt only when segments change: readers
        // mask tombstoned entries through `deleted`, so delete publishes are O(1).
        let any_ids = !g.hot_ids.is_empty() || g.segments.iter().any(|s| s.has_ids());
        let ids: Arc<HashMap<String, DocAddr>> = if segment_change && any_ids {
            let mut ids: HashMap<String, DocAddr> = HashMap::with_capacity(g.ids.len());
            for (si, s) in g.segments.iter().enumerate() {
                let mask = deleted.get(&(si as u32));
                for docid in 0..s.num_docs {
                    if mask.is_some_and(|m| m.contains(docid)) {
                        continue;
                    }
                    if let Some(id) = s.id_of(docid) {
                        ids.insert(
                            id.to_owned(),
                            DocAddr {
                                segment: si as u32,
                                docid,
                            },
                        );
                    }
                }
            }
            let ids = Arc::new(ids);
            g.ids = ids.as_ref().clone();
            g.ids_arc = ids.clone();
            ids
        } else if segment_change {
            g.ids = HashMap::new();
            g.ids_arc = Arc::new(HashMap::new());
            g.ids_arc.clone()
        } else {
            // tombstone-only publish: previous id map stays valid (masked by `deleted`)
            g.ids_arc.clone()
        };
        let live = total_docs - tombstoned;
        let snapshot = IndexSnapshot {
            segments: g.segments.clone(),
            stats: Stats {
                total_docs: live as u32,
                avg_len: sums
                    .iter()
                    .map(|s| {
                        if live > 0 {
                            *s as f32 / live as f32
                        } else {
                            1.0
                        }
                    })
                    .collect(),
                df,
            },
            deleted: Arc::new(deleted),
            ids,
            generation: g.generation,
        };
        self.snap.store(Arc::new(snapshot));
    }

    pub fn search(&self, q: &str, k: usize) -> Result<Vec<SearchHit>, IndexError> {
        let node = crate::query::parse(q).map_err(|e| IndexError::Query(format!("{e:?}")))?;
        let snap = self.snap.load();
        let params = crate::search::Bm25Params {
            k1: self.k1,
            b: self.b,
        };
        // single Term: streaming walk with a live topk threshold (block-max WAND);
        // generic nodes: dense arrays + collector push. Tombstoned docs are masked
        // through the `allowed` probe (zero cost when no tombstones exist).
        let per_seg: Vec<Vec<SearchHit>> = {
            use rayon::prelude::*;
            snap.segments
                .par_iter()
                .enumerate()
                .map(|(si, seg)| {
                    let mask = snap.deleted.get(&(si as u32)).filter(|m| !m.is_empty());
                    let allowed: Option<RoaringBitmap> = mask.map(|m| {
                        let full: RoaringBitmap = (0..seg.num_docs).collect();
                        &full - m
                    });
                    let mut collector = TopK::new(k);
                    // WAND-lite fast path needs a ctx closure bound to this collector;
                    // for non-Term nodes fall back to dense array evaluation
                    match &node {
                        Node::Term(_) => {
                            // the closure holds a shared view of the collector: threshold live
                            let col: *mut TopK = &mut collector as *mut TopK;
                            let thr = move || unsafe { (*col).threshold() };
                            let ctx = crate::search::exec::Ctx {
                                schema: &self.schema,
                                stats: &snap.stats,
                                params: &params,
                                analyzer: &self.analyzer,
                                threshold: &thr,
                            };
                            crate::search::exec::eval_term_stream(
                                &node,
                                None,
                                seg,
                                &ctx,
                                allowed.as_ref(),
                                &mut collector,
                            );
                        }
                        _ => {
                            // live threshold only for root-level phrase (the doc's score
                            // IS the phrase score there, so a segment-level bound is
                            // sound); nested phrases must evaluate unconditionally
                            let phrase_root = matches!(&node, Node::Phrase { .. })
                                || matches!(&node, Node::Field { node: inner, .. }
                                    if matches!(**inner, Node::Phrase { .. }));
                            let col: *mut TopK = &mut collector as *mut TopK;
                            let thr: Box<dyn Fn() -> f32> = if phrase_root {
                                Box::new(move || unsafe { (*col).threshold() })
                            } else {
                                Box::new(|| 0.0)
                            };
                            let ctx = crate::search::exec::Ctx {
                                schema: &self.schema,
                                stats: &snap.stats,
                                params: &params,
                                analyzer: &self.analyzer,
                                threshold: &thr,
                            };
                            let ds = crate::search::exec::eval_allowed(
                                &node,
                                None,
                                seg,
                                &ctx,
                                allowed.as_ref(),
                            );
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
                    }
                    // term_stream pushes with segment=0; remap to this segment index
                    let out = collector.into_sorted();
                    out.into_iter()
                        .filter(|h| {
                            // safety net for eval paths without an `allowed` probe
                            // (phrase, fuzzy, prefix): drop tombstoned docs
                            !mask.is_some_and(|m| m.contains(h.addr.docid))
                        })
                        .map(|h| SearchHit {
                            addr: DocAddr {
                                segment: si as u32,
                                ..h.addr
                            },
                            score: h.score,
                        })
                        .collect()
                })
                .collect()
        };
        let mut merged = TopK::new(k);
        for hits in per_seg {
            for h in hits {
                merged.push(h.addr, h.score);
            }
        }
        Ok(merged.into_sorted())
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

/// concatenate segments into one, dropping tombstoned docs and remapping docids
pub(crate) fn merge_segments(
    segs: &[Arc<SegmentReader>],
    masks: &[RoaringBitmap],
    nf: usize,
) -> SegmentReader {
    // per-segment docid remap: old docid -> new docid, u32::MAX = tombstoned (purged)
    let mut remaps: Vec<Vec<u32>> = segs
        .iter()
        .enumerate()
        .map(|(si, s)| {
            let mask = masks.get(si);
            (0..s.num_docs)
                .map(|d| {
                    if mask.is_some_and(|m| m.contains(d)) {
                        u32::MAX
                    } else {
                        0
                    }
                })
                .collect()
        })
        .collect();
    // running new-docid base per segment (live docs only)
    let mut running = 0u32;
    let bases: Vec<u32> = remaps
        .iter()
        .map(|r| {
            let b = running;
            running += r.iter().filter(|&&d| d != u32::MAX).count() as u32;
            b
        })
        .collect();
    for (si, r) in remaps.iter_mut().enumerate() {
        let mut next = bases[si];
        for d in r.iter_mut() {
            if *d != u32::MAX {
                *d = next;
                next += 1;
            }
        }
    }
    let mut field_norms: Vec<Vec<u16>> = (0..nf).map(|_| Vec::new()).collect();
    let mut store_blocks: Vec<Box<[u8]>> = Vec::new();
    let mut ids: Vec<Option<String>> = Vec::new();
    for (si, s) in segs.iter().enumerate() {
        let remap = &remaps[si];
        for (fid, norms) in s.field_norms.iter().enumerate() {
            field_norms[fid].extend(
                norms
                    .iter()
                    .enumerate()
                    .filter(|(d, _)| remap[*d] != u32::MAX)
                    .map(|(_, n)| *n),
            );
        }
        for (docid, &nd) in remap.iter().enumerate() {
            if nd == u32::MAX {
                continue;
            }
            if let Some(block) = s.stored_raw(docid as u32) {
                store_blocks.push(block.to_vec().into_boxed_slice());
            }
            ids.push(s.id_of(docid as u32).map(|s| s.to_owned()));
        }
    }
    let total = running;
    // per-field posting union. The field list alone saturates only multi-field
    // corpora (the common single-field case merges on ONE thread), so merge
    // also fans out over contiguous segment ranges within each field: parallel
    // partial maps decode in parallel, then fold (docid ranges per chunk are
    // contiguous and ascending, so PostingData concat is order-preserving).
    let fields: Vec<usize> = (0..nf).collect();
    let postings = {
        use rayon::prelude::*;
        let workers = rayon::current_num_threads().max(1);
        fields
            .par_iter()
            .map(|fid| {
                let parts: Vec<HashMap<String, PostingData>> = {
                    let chunk = segs.len().div_ceil(workers).max(1);
                    (0..segs.len())
                        .into_par_iter()
                        .chunks(chunk)
                        .map(|range: Vec<usize>| {
                            let mut m = HashMap::<String, PostingData>::new();
                            // R21: get-mut-first dict probes (clone the term only
                            // when it is new) + reused position buffer with
                            // mem::take (no per-entry Vec allocation)
                            let mut pos_buf = Vec::new();
                            for si in range {
                                let s = &segs[si];
                                let remap = &remaps[si];
                                for (term, info) in s.stream_terms(*fid as u32) {
                                    let ent = match m.get_mut(term.as_str()) {
                                        Some(e) => e,
                                        None => m.entry(term.clone()).or_default(),
                                    };
                                    let mut r = s.reader(&info);
                                    let mut idx = 0usize;
                                    while let Some((doc, tf)) = r.advance() {
                                        let nd = remap[doc as usize];
                                        if nd == u32::MAX {
                                            continue;
                                        }
                                        r.positions_into(idx, &mut pos_buf);
                                        idx += 1;
                                        ent.add(nd, tf, std::mem::take(&mut pos_buf));
                                    }
                                }
                            }
                            m
                        })
                        .collect()
                };
                // fold partial maps into one (per-chunk docid ranges are
                // contiguous + ascending, so plain concatenation stays sorted)
                let mut total = HashMap::<String, PostingData>::new();
                for pm in parts {
                    for (term, data) in pm {
                        match total.get_mut(term.as_str()) {
                            Some(e) => e.absorb(data),
                            None => {
                                total.insert(term, data);
                            }
                        }
                    }
                }
                total
            })
            .collect::<Vec<_>>()
    };
    crate::segment::freeze_parts(postings, field_norms, store_blocks, ids, total)
}
