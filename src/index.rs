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
/// cached publish stats: (global df, total docs). Per-field LIVE sums live in
/// `IndexInner.live_sums` (exact, tombstone-masked) instead — the gossip path
/// needs them tombstone-exact and the cache's all-doc sums never were.
type StatsCache = (Arc<HashMap<u32, HashMap<String, u32>>>, u64);

/// per-segment merge scan: (per-field norm slices, store blocks, external ids)
type SegScan = (Vec<Vec<u16>>, Vec<Box<[u8]>>, Vec<Option<String>>);

/// opt-in disk persistence: immutable segments under DIR/segments/, plus an
/// atomically-rewritten MANIFEST.json (tmp file + fsync + rename). Boot:
/// read the manifest, load named segments, publish the snapshot. Uncommitted
/// hot lanes are bounded by commit-every and ARE LOST on crash (documented).
pub struct DiskHandle {
    dir: std::path::PathBuf,
}

impl DiskHandle {
    pub fn open(dir: std::path::PathBuf) -> Result<Self, String> {
        std::fs::create_dir_all(dir.join("segments"))
            .map_err(|e| format!("disk {:?}: {e}", dir))?;
        Ok(Self { dir })
    }

    /// persist every live segment whose file does not exist yet + rewrite manifest.
    /// files are CONTENT-ADDRESSED (hash of the serialized bytes) so manifest
    /// entries are unambiguous across generations without any migration step.
    pub fn persist(
        &self,
        segments: &[Arc<SegmentReader>],
        deleted: &HashMap<u32, RoaringBitmap>,
    ) -> Result<(), String> {
        let mut files = Vec::with_capacity(segments.len());
        let mut seen = std::collections::HashSet::new();
        for seg in segments {
            // memoized hash: known segments never re-serialize on delete/merge
            let hash = match seg.content_hash() {
                Some(h) => h,
                None => {
                    let bytes = seg.to_bytes(); // primes the memo inside the reader
                    let h = crate::cluster::fnv64a(&bytes);
                    let name = format!("seg-{:016x}.zap", h);
                    if !self.dir.join("segments").join(&name).exists() {
                        atomic_write(&self.dir.join("segments").join(&name), &bytes)?;
                    }
                    seen.insert(name.clone());
                    files.push(name);
                    continue;
                }
            };
            let name = format!("seg-{:016x}.zap", hash);
            if !seen.contains(&name) && !self.dir.join("segments").join(&name).exists() {
                let bytes = seg.to_bytes();
                atomic_write(&self.dir.join("segments").join(&name), &bytes)?;
            }
            seen.insert(name.clone());
            files.push(name);
        }
        let mut doc = serde_json::json!({
            "version": 1,
            "segments": segments.len(),
            "files": files,
        });
        if !deleted.is_empty() {
            let mut tomb = serde_json::Map::new();
            for (si, mask) in deleted {
                if mask.is_empty() {
                    continue;
                }
                tomb.insert(
                    si.to_string(),
                    serde_json::json!(mask.iter().collect::<Vec<u32>>()),
                );
            }
            if !tomb.is_empty() {
                doc.as_object_mut()
                    .expect("object")
                    .insert("tombstones".to_string(), serde_json::Value::Object(tomb));
            }
        }
        let bytes = serde_json::to_vec(&doc).map_err(|e| e.to_string())?;
        atomic_write(&self.manifest_path(), &bytes)
    }

    /// load back: files referenced by the manifest, in order. Reads + decodes
    /// in parallel (segments are independent); primes the content-hash memo
    /// from the file bytes so the first delete pays zero serialization.
    pub fn load(&self) -> Result<Vec<Arc<SegmentReader>>, String> {
        let manifest = match std::fs::read(self.manifest_path()) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(format!("manifest: {e}")),
        };
        let doc: serde_json::Value =
            serde_json::from_slice(&manifest).map_err(|e| format!("manifest parse: {e}"))?;
        let files = doc["files"].as_array().ok_or("manifest missing files")?;
        let names: Vec<&str> = files.iter().filter_map(|f| f.as_str()).collect();
        let loaded: Vec<Result<Arc<SegmentReader>, String>> = {
            use rayon::prelude::*;
            names
                .par_iter()
                .map(|name| {
                    let bytes = std::fs::read(self.dir.join("segments").join(name))
                        .map_err(|e| format!("segment {name}: {e}"))?;
                    let hash = crate::cluster::fnv64a(&bytes);
                    let seg = SegmentReader::from_bytes(&bytes)?;
                    seg.prime_content_hash(hash);
                    Ok(Arc::new(seg))
                })
                .collect()
        };
        let mut out = Vec::with_capacity(names.len());
        for r in loaded {
            out.push(r?);
        }
        Ok(out)
    }

    /// rebuild tombstone map from the manifest
    pub fn load_tombstones(&self) -> Result<HashMap<u32, RoaringBitmap>, String> {
        let manifest = match std::fs::read(self.manifest_path()) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(HashMap::new()),
            Err(e) => return Err(format!("manifest: {e}")),
        };
        let doc: serde_json::Value =
            serde_json::from_slice(&manifest).map_err(|e| format!("manifest parse: {e}"))?;
        let mut out = HashMap::new();
        if let Some(t) = doc["tombstones"].as_object() {
            for (k, v) in t {
                let si: u32 = k.parse().map_err(|e| format!("tombstone key: {e}"))?;
                let ids: Vec<u32> = serde_json::from_value(v.clone()).map_err(|e| e.to_string())?;
                let mut bm = RoaringBitmap::new();
                for id in ids {
                    bm.insert(id);
                }
                out.insert(si, bm);
            }
        }
        Ok(out)
    }

    fn manifest_path(&self) -> std::path::PathBuf {
        self.dir.join("MANIFEST.json")
    }
}

fn atomic_write(path: &std::path::Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp).map_err(|e| format!("open {:?}: {e}", tmp))?;
        f.write_all(bytes)
            .map_err(|e| format!("write {tmp:?}: {e}"))?;
        f.sync_all().map_err(|e| format!("sync {tmp:?}: {e}"))?;
    }
    std::fs::rename(&tmp, path).map_err(|e| format!("rename {tmp:?}: {e}"))
}

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
    /// exact per-field token sums over LIVE docs (tombstoned excluded) —
    /// maintained incrementally so gossip envelopes cost O(fields), not a
    /// full norms rescan per push
    pub sums: Vec<u64>,
    /// per-segment tombstone masks (segment index -> deleted docids)
    pub deleted: Arc<HashMap<u32, RoaringBitmap>>,
    /// live external id -> address (tombstoned docs excluded)
    pub ids: Arc<im::HashMap<String, DocAddr>>,
    pub generation: u64,
    /// tombstone complement per segment (allowed probe), built ONCE per publish:
    /// readers extend the lifetime via Arc instead of rebuilding the `1 - mask`
    /// probe live (O(num_docs/8) bytes per segment per query whenever tombstones
    /// exist). Empty masks stay absent.
    pub allowed: Arc<HashMap<u32, RoaringBitmap>>,
}

pub struct Index {
    disk: Option<DiskHandle>,
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
    /// live external id -> committed address, PERSISTENT im::HashMap Arc-shared
    /// with the published snapshot: deletes rewrite in place (O(log n)), commits
    /// extend via O(1) clone + O(log n) update per appended doc — no copy of the
    /// map per commit. Hot ids live in `hot_ids` until commit.
    ids_arc: Arc<im::HashMap<String, DocAddr>>,
    /// exact per-field token sums over LIVE docs (tombstoned excluded).
    /// boot: full rescan; commit fold: += new segments; delete: -= deleted doc;
    /// merge: recomputed from the merged (purged) segment.
    live_sums: Vec<u64>,
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
        Self::with_lanes_disk(schema, analyzer, k1, b, lanes, None)
    }

    /// opt-in persistence: `disk_dir` set -> segments load at boot, and every
    /// commit/merge/delete atomically rewrites the manifest under DIR
    pub fn with_lanes_disk(
        schema: Schema,
        analyzer: Arc<Analyzer>,
        k1: f32,
        b: f32,
        lanes: usize,
        disk_dir: Option<std::path::PathBuf>,
    ) -> Arc<Self> {
        let schema = Arc::new(schema);
        let lanes = lanes.max(1);
        let hot = (0..lanes)
            .map(|_| SegmentBuilder::new(schema.clone(), analyzer.clone()))
            .collect();
        // opt-in persistence: load existing state synchronously at boot
        let disk = match disk_dir {
            Some(dir) => {
                let h = DiskHandle::open(dir).expect("open disk dir");
                let segments = h.load().expect("load disk segments");
                let tombstones = h.load_tombstones().expect("load disk tombstones");
                Some((h, segments, tombstones))
            }
            None => None,
        };
        let nf = schema.num_fields();
        let (handle, segs, tomb) = match disk {
            Some((h, segs, t)) => (Some(h), segs, t),
            None => (None, Vec::new(), HashMap::new()),
        };
        // boot live-sum rescan: tombstone-masked (the cache-less path in
        // publish_snapshot counts tombstoned docs — these sums must not)
        let live_sums: Vec<u64> = {
            let mut sums = vec![0u64; nf];
            for (si, s) in segs.iter().enumerate() {
                let mask = tomb.get(&(si as u32));
                for (fid, norms) in s.field_norms.iter().enumerate() {
                    sums[fid] += norms
                        .iter()
                        .enumerate()
                        .filter(|(d, _)| !mask.is_some_and(|m| m.contains(*d as u32)))
                        .map(|(_, v)| *v as u64)
                        .sum::<u64>();
                }
            }
            sums
        };
        let index = Arc::new(Self {
            disk: handle,
            schema,
            analyzer,
            k1,
            b,
            snap: ArcSwap::from_pointee(IndexSnapshot {
                segments: Vec::new(),
                stats: Stats::empty(),
                sums: Vec::new(),
                deleted: Arc::new(HashMap::new()),
                allowed: Arc::new(HashMap::new()),
                ids: Arc::new(im::HashMap::new()),
                generation: 0,
            }),
            inner: Mutex::new(IndexInner {
                hot,
                next_lane: 0,
                segments: segs,
                deleted: tomb,
                ids_arc: Arc::new(im::HashMap::new()),
                live_sums,
                stats_cache: None,
                hot_ids: HashSet::new(),
                generation: 0,
            }),
        });
        // publish the loaded snapshot + rebuild the id index on top of it
        {
            let mut g = index.inner.lock().unwrap();
            if std::env::var("ZAP_DISK_TRACE").is_ok() {
                eprintln!(
                    "[disk] segs={} tomb={} tomblen={:?}",
                    g.segments.len(),
                    g.deleted.len(),
                    g.deleted.values().map(|m| m.len()).collect::<Vec<_>>()
                );
            }
            index.publish_snapshot(&mut g, true, None);
            if std::env::var("ZAP_DISK_TRACE").is_ok() {
                eprintln!(
                    "[disk] post-publish snap total_docs={}",
                    index.pub_stats().total_docs
                );
            }
        }
        index
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
            if let Some(old) = g.ids_arc.get(id).copied() {
                g.deleted.entry(old.segment).or_default().insert(old.docid);
                g.ids_arc = Arc::new(g.ids_arc.as_ref().without(id));
            }
        }
        let lane = g.next_lane % g.hot.len();
        g.next_lane += 1;
        g.hot[lane]
            .add_document(doc)
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
                    .is_some_and(|id| g.hot_ids.contains(id) || g.ids_arc.contains_key(id))
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
        // split into per-lane vecs WITHOUT cloning the documents (the old
        // `chunk.to_vec()` deep-copied every field String per ingest)
        let mut chunks: Vec<Vec<Document>> = (0..g.hot.len()).map(|_| Vec::new()).collect();
        for (i, d) in deduped.into_iter().enumerate() {
            chunks[i / chunk].push(d);
        }
        let hot_ids_new: Vec<String> = chunks
            .iter()
            .flat_map(|c| c.iter().filter_map(|d| d.id.clone()))
            .collect();
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
        let base_len = segs.len();
        segs.extend(new_segs);
        let lanes = g.hot.len().max(1);
        let mut merged_now = false;
        if segs.len() >= MAX_SEGMENTS * lanes.max(1) {
            let masks_all: Vec<RoaringBitmap> = (0..segs.len())
                .map(|i| g.deleted.get(&(i as u32)).cloned().unwrap_or_default())
                .collect();
            let merged = merge_segments(&segs, &masks_all, self.schema.num_fields());
            segs = vec![Arc::new(merged)];
            g.deleted.clear();
            merged_now = true;
        }
        g.segments = segs;
        g.hot_ids.clear();
        g.generation += 1;
        // stats maintenance happens HERE so publishes stay O(new segments):
        // appends stream only the frozen lanes and fold their df/sums/docs into
        // the cached envelope; merges/boot fall back to one full rescan
        let nf = self.schema.num_fields();
        let (df, total_docs_check) = if merged_now {
            // merge purged docids: fold from the merged segment alone
            let mut raw_df: HashMap<u32, HashMap<String, u32>> = HashMap::new();
            let mut total = 0u64;
            for s in &g.segments {
                total += s.num_docs as u64;
                for fid in 0..nf as u32 {
                    if let Some(nm) = s.norms(fid) {
                        g.live_sums[fid as usize] = nm.iter().map(|x| *x as u64).sum::<u64>();
                    }
                    for (term, info) in s.stream_terms(fid) {
                        *raw_df.entry(fid).or_default().entry(term).or_insert(0) += info.df;
                    }
                }
            }
            (Arc::new(raw_df), total)
        } else {
            // take the mutable map back when no live snapshot still holds it;
            // else deep-clone (a query was executing mid-commit)
            let (mut df_mut, mut total) = match g.stats_cache.take() {
                Some((d, t)) => {
                    let d = match Arc::try_unwrap(d) {
                        Ok(m) => m,
                        Err(arc) => (*arc).clone(),
                    };
                    (d, t)
                }
                None => (HashMap::<u32, HashMap<String, u32>>::new(), 0u64),
            };
            for s in &g.segments[base_len..] {
                total += s.num_docs as u64;
                for fid in 0..nf as u32 {
                    if let Some(nm) = s.norms(fid) {
                        g.live_sums[fid as usize] += nm.iter().map(|x| *x as u64).sum::<u64>();
                    }
                    for (term, info) in s.stream_terms(fid) {
                        *df_mut.entry(fid).or_default().entry(term).or_insert(0) += info.df;
                    }
                }
            }
            (Arc::new(df_mut), total)
        };
        let total_docs = total_docs_check;
        g.stats_cache = Some((df, total_docs));
        let new_from = if merged_now { None } else { Some(base_len) };
        self.publish_snapshot(g, true, new_from);
        if let Some(d) = &self.disk {
            d.persist(&g.segments, &g.deleted).expect("disk persist");
        }
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
        let Some(old) = g.ids_arc.get(id).copied() else {
            return Ok(false);
        };
        if std::env::var("ZAP_DISK_TRACE").is_ok() {
            eprintln!("[delete] id={id} old={:?}", old);
        }
        g.deleted.entry(old.segment).or_default().insert(old.docid);
        // exact live-sum maintenance: subtract the tombstoned doc's norms
        // (copy the small norm slice out first: seg is an Arc from the shared list)
        if let Some(seg) = g.segments.get(old.segment as usize).cloned() {
            let norms: Vec<Option<u16>> = (0..g.live_sums.len())
                .map(|fid| seg.norms(fid as u32).map(|nm| nm[old.docid as usize]))
                .collect();
            for (fid, sums) in g.live_sums.iter_mut().enumerate() {
                if let Some(n) = norms[fid] {
                    *sums -= n as u64;
                }
            }
        }
        g.ids_arc = Arc::new(g.ids_arc.as_ref().without(id));
        self.publish_snapshot(&mut g, false, None);
        if let Some(d) = &self.disk {
            d.persist(&g.segments, &g.deleted)
                .expect("disk persist tombstone");
        }
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

    /// live-doc scan page for handoff: documents strictly after the cursor
    /// address `(segment, docid)` in segment/docid order, literal docs up to
    /// `limit` rows as the FLAT ingest wire shape, plus the next cursor
    /// (None = the snapshot is exhausted). Tombstoned docs are skipped.
    pub fn docs_page(
        &self,
        after: Option<(u32, u32)>,
        limit: usize,
    ) -> (Vec<serde_json::Value>, Option<(u32, u32)>) {
        let snap = self.snap.load();
        let mut out = Vec::with_capacity(limit.min(64));
        let mut cursor = after;
        for (si, seg) in snap.segments.iter().enumerate() {
            let mask = snap.deleted.get(&(si as u32));
            // advance the inner loop to the cursor instead of filtering from
            // docid 0 (the old `addr <= cur continue` walk made a full handoff
            // O(pages × corpus): page i re-scanned all i×limit prior docids)
            let start_docid = match cursor {
                Some((cur_si, cur_d)) if (cur_si as usize) == si => cur_d + 1,
                Some((cur_si, _)) if (cur_si as usize) > si => continue,
                _ => 0,
            };
            for docid in start_docid..seg.num_docs {
                let addr = (si as u32, docid);
                if let Some(cur) = cursor {
                    if addr <= cur {
                        continue;
                    }
                }
                if mask.is_some_and(|m| m.contains(docid)) {
                    continue;
                }
                let Some(stored) = seg.stored_doc(docid) else {
                    continue;
                };
                let mut row = serde_json::Map::new();
                if let Some(id) = stored.id {
                    row.insert("id".to_string(), serde_json::Value::String(id));
                }
                if let Some(meta) = stored.metadata {
                    row.insert("metadata".to_string(), meta);
                }
                for (fid, text) in stored.fields {
                    if let Some(f) = self.schema.field(fid) {
                        row.insert(f.name.clone(), serde_json::Value::String(text));
                    }
                }
                out.push(serde_json::Value::Object(row));
                cursor = Some(addr);
                if out.len() >= limit {
                    return (out, cursor);
                }
            }
        }
        (out, None)
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
    fn publish_snapshot(&self, g: &mut IndexInner, segment_change: bool, new_from: Option<usize>) {
        let nf = self.schema.num_fields();
        let deleted: HashMap<u32, RoaringBitmap> = g
            .deleted
            .iter()
            .filter(|(_, m)| !m.is_empty())
            .map(|(&si, m)| (si, m.clone()))
            .collect();
        // complement bitmaps for the allowed probe: one build per publish
        // (O(num_docs/8) per tombstoned segment) instead of one per query
        let allowed: HashMap<u32, RoaringBitmap> = if deleted.is_empty() {
            HashMap::new()
        } else {
            deleted
                .iter()
                .filter_map(|(&si, mask)| {
                    let n = g.segments.get(si as usize)?.num_docs;
                    let full: RoaringBitmap = (0..n).collect();
                    Some((si, &full - mask))
                })
                .collect()
        };
        let tombstoned: u64 = deleted.values().map(|m| m.len()).sum();
        let mut total_docs: u64 = 0;
        let mut df: Arc<HashMap<u32, HashMap<String, u32>>> = Arc::new(HashMap::new());
        if segment_change {
            // commit path: the caller refreshed the envelope in commit_locked
            // (cache stays in g — later commits fold into it); boot has none
            if let Some((d, t)) = g.stats_cache.as_ref() {
                df = d.clone();
                total_docs = *t;
            } else {
                let mut raw_df: HashMap<u32, HashMap<String, u32>> = HashMap::new();
                for s in &g.segments {
                    total_docs += s.num_docs as u64;
                    for fid in 0..nf as u32 {
                        for (term, info) in s.stream_terms(fid) {
                            *raw_df.entry(fid).or_default().entry(term).or_insert(0) += info.df;
                        }
                    }
                }
                df = Arc::new(raw_df);
                g.stats_cache = Some((df.clone(), total_docs));
            }
        } else {
            // segments unchanged: reuse cached totals/sums/df (Arc clone, no deep copy)
            if let Some((d, t)) = &g.stats_cache {
                df = d.clone();
                total_docs = *t;
            }
        }
        // live external id -> address over a persistent im::HashMap: deletes
        // remove in place (delete/replace); segment-change publishes extend via
        // O(1) clone + O(log n) update per appended doc (`segs.extend` appends,
        // so old addresses stay valid). Merges/boot reset segment indices ->
        // full rescan (O(live), amortized over the rare merge).
        let any_ids = !g.hot_ids.is_empty() || g.segments.iter().any(|s| s.has_ids());
        let ids: Arc<im::HashMap<String, DocAddr>> = if segment_change && any_ids {
            let from = new_from.unwrap_or(0);
            let base: im::HashMap<String, DocAddr> = if from == 0 {
                im::HashMap::new()
            } else {
                (*g.ids_arc).clone()
            };
            let mut ids = base;
            for si in from..g.segments.len() {
                let s = &g.segments[si];
                let mask = deleted.get(&(si as u32));
                for docid in 0..s.num_docs {
                    if mask.is_some_and(|m| m.contains(docid)) {
                        continue;
                    }
                    if let Some(id) = s.id_of(docid) {
                        ids = ids.update(
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
            g.ids_arc = ids.clone();
            ids
        } else if segment_change {
            g.ids_arc = Arc::new(im::HashMap::new());
            g.ids_arc.clone()
        } else {
            // tombstone-only publish: deletes already rewrote the shared map
            g.ids_arc.clone()
        };
        let live = total_docs - tombstoned;
        let snapshot = IndexSnapshot {
            segments: g.segments.clone(),
            stats: Stats {
                total_docs: live as u32,
                avg_len: g
                    .live_sums
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
            sums: g.live_sums.clone(),
            deleted: Arc::new(deleted),
            allowed: Arc::new(allowed),
            ids,
            generation: g.generation,
        };
        self.snap.store(Arc::new(snapshot));
    }

    pub fn search(&self, q: &str, k: usize) -> Result<Vec<SearchHit>, IndexError> {
        let snap = self.snap.load();
        self.search_with_stats(q, k, snap.stats.clone())
    }

    /// evaluate with an externally supplied Stats (cluster global df/totals);
    /// identical eval tree, identical tie-order, one source of stats influence
    pub fn search_with_stats(
        &self,
        q: &str,
        k: usize,
        stats: crate::search::exec::Stats,
    ) -> Result<Vec<SearchHit>, IndexError> {
        let node = crate::query::parse(q).map_err(|e| IndexError::Query(format!("{e:?}")))?;
        let snap = self.snap.load();
        let params = self.params();
        // resolve query-term stems once (snowball + alloc per DISTINCT term),
        // shared read-only across the per-segment rayon tasks
        let stems = crate::search::exec::stem_cache(&self.analyzer, &node);
        // single Term: streaming walk with a live topk threshold (block-max WAND);
        // generic nodes: dense arrays + collector push. Tombstoned docs are masked
        // through the `allowed` probe (zero cost when no tombstones exist).
        let per_seg: Vec<Vec<SearchHit>> = {
            use rayon::prelude::*;
            snap.segments
                .par_iter()
                .enumerate()
                .map(|(si, seg)| {
                    // complement bitmap is precomputed on the snapshot: when
                    // tombstones exist this used to rebuild `(0..num_docs)-mask`
                    // per segment per query; now it is an Arc clone
                    let allowed: Option<&RoaringBitmap> = snap.allowed.get(&(si as u32));
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
                                stats: &stats,
                                params: &params,
                                analyzer: &self.analyzer,
                                threshold: &thr,
                                stems: &stems,
                            };
                            crate::search::exec::eval_term_stream(
                                &node,
                                None,
                                seg,
                                &ctx,
                                allowed,
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
                                // same shared-view contract as the Term path above:
                                // col outlives every read because it only escapes
                                // inside this scope
                                Box::new(move || unsafe { (*col).threshold() })
                            } else {
                                Box::new(|| 0.0)
                            };
                            let ctx = crate::search::exec::Ctx {
                                schema: &self.schema,
                                stats: &stats,
                                params: &params,
                                analyzer: &self.analyzer,
                                threshold: &thr,
                                stems: &stems,
                            };
                            let ds =
                                crate::search::exec::eval_allowed(&node, None, seg, &ctx, allowed);
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
                            !snap
                                .deleted
                                .get(&(si as u32))
                                .is_some_and(|m| m.contains(h.addr.docid))
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

    /// current snapshot stats copy (Arc df + avg lens); cheap for callers
    pub fn pub_stats(&self) -> crate::search::exec::Stats {
        self.snap.load().stats.clone()
    }

    /// live per-field token sums (tombstoned docs excluded) — O(fields): the
    /// exact sums are maintained incrementally on the snapshot (this used to
    /// re-scan every norm of every segment per call, i.e. O(corpus) per gossip push)
    pub fn field_sums(&self) -> Vec<u64> {
        self.snap.load().sums.clone()
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
    // R22: parallel per-segment scan of norms + store blocks + external ids;
    // which-parallel collect keeps segment order, the fold is order-preserving
    let per_seg: Vec<SegScan> = {
        use rayon::prelude::*;
        (0..segs.len())
            .into_par_iter()
            .map(|si| {
                let s = &segs[si];
                let remap = &remaps[si];
                let mut norms: Vec<Vec<u16>> = Vec::with_capacity(s.field_norms.len());
                for n in s.field_norms.iter() {
                    norms.push(
                        n.iter()
                            .enumerate()
                            .filter(|(d, _)| remap[*d] != u32::MAX)
                            .map(|(_, v)| *v)
                            .collect(),
                    );
                }
                let mut blocks = Vec::new();
                let mut seg_ids = Vec::new();
                for (docid, &nd) in remap.iter().enumerate() {
                    if nd == u32::MAX {
                        continue;
                    }
                    if let Some(block) = s.stored_raw(docid as u32) {
                        blocks.push(block.to_vec().into_boxed_slice());
                    }
                    seg_ids.push(s.id_of(docid as u32).map(|x| x.to_owned()));
                }
                (norms, blocks, seg_ids)
            })
            .collect()
    };
    let mut field_norms: Vec<Vec<u16>> = (0..nf).map(|_| Vec::new()).collect();
    let mut store_blocks: Vec<Box<[u8]>> = Vec::new();
    let mut ids: Vec<Option<String>> = Vec::new();
    for (seg_norms, blocks, seg_ids) in per_seg {
        for (fid, n) in seg_norms.into_iter().enumerate() {
            field_norms[fid].extend(n);
        }
        store_blocks.extend(blocks);
        ids.extend(seg_ids);
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
