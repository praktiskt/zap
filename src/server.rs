//! HTTP serve mode: NDJSON ingest + search + cluster fan-out endpoints.
//!
//! The engine is CPU-bound and sync, so the HTTP layer stays threaded
//! (`tiny_http`); no async runtime is involved. Public endpoints:
//! - `GET  /health`            liveness (local shard)
//! - `GET  /stats`             docs / segments / generation (cluster-summed)
//! - `GET/POST /search` {q,k}  fan-out to all shards, merged top-k
//! - `POST /ingest`   NDJSON   {"id"?, "text", "metadata"?} — routed by ring
//! - `POST /commit`            local shard commit (auto-commits on ingest too)
//! - `GET/DELETE /doc/{id}`    get / tombstone one external id (ring-routed)
//!
//! Cluster-internal endpoints (never routed onward; always local shard work):
//! - `POST   /__zap/gossip`       StatsEnvelope merge (delta or full)
//! - `GET    /__zap/gossip_full`  this shard's absolute envelope
//! - `POST   /__zap/shard_search` local search with merged global stats
//! - `POST   /__zap/shard_ingest` local ingest (owner already chose us)
//! - `GET    /__zap/shard_stats`  live counts for the aggregate /stats
//! - `GET    /dashboard`          live HTML stats page

use std::io::Read;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tiny_http::{Method, Request, Response, Server};

use crate::cluster::{
    encode_df_key, http_delete, http_get, http_post, parse_df_key, LocalCluster, PeerCache,
    PeerConfig, StatsEnvelope,
};
use crate::index::Index;
use crate::schema::Document;
use crate::search::exec::Stats;

const MAX_BODY: usize = 64 << 20; // 64 MB ingest ceiling per request

// dashboard counters (per process; the cluster chips merge them per shard)
static SEARCHES: AtomicU64 = AtomicU64::new(0);
static INGESTED: AtomicU64 = AtomicU64::new(0);
static SERVE_STARTED: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

/// owned cluster runtime; None = single node (routes unchanged)
pub type ClusterOpt = Option<Arc<LocalCluster>>;
/// borrowed cluster handle (internal plumbing)
pub type ClusterRef<'a> = Option<&'a Arc<LocalCluster>>;

/// per-shard ingest batch: owner address -> docs routed to that shard
type ShardBatch = (SocketAddr, Vec<Document>);

/// build Stats from an envelope
fn stats_from_envelope(env: StatsEnvelope) -> Stats {
    let nf = env.sums.len();
    let mut avg = Vec::with_capacity(nf);
    for s in &env.sums {
        avg.push(if env.docs > 0 {
            *s as f32 / env.docs as f32
        } else {
            1.0
        });
    }
    let mut df = std::collections::HashMap::new();
    for (k, v) in env.df {
        if let Some((fid, term)) = parse_df_key(&k) {
            df.insert((fid, term), v);
        }
    }
    Stats {
        total_docs: env.docs,
        avg_len: avg,
        df: Arc::new(df),
    }
}

/// absolute envelope from engine stats; sums recomputed from live norms
fn envelope_from_stats(index: &Index, stats: &Stats) -> StatsEnvelope {
    let sums = index.field_sums();
    let mut df = std::collections::HashMap::new();
    for ((fid, term), v) in stats.df.iter() {
        df.insert(encode_df_key(*fid, term), *v);
    }
    StatsEnvelope {
        docs: stats.total_docs,
        sums,
        df,
        full: false,
    }
}

/// this shard's absolute envelope
fn own_envelope(index: &Index) -> StatsEnvelope {
    envelope_from_stats(index, &index.pub_stats())
}

/// run the HTTP listener until `shutdown` flips; returns the bound address
pub fn serve(
    index: Arc<Index>,
    bind: &str,
    cluster: Option<PeerConfig>,
    shutdown: &'static AtomicBool,
) -> Result<SocketAddr, String> {
    let server = Server::http(bind).map_err(|e| format!("bind {bind}: {e}"))?;
    SERVE_STARTED.set(std::time::Instant::now()).ok();
    let addr: SocketAddr = server
        .server_addr()
        .to_ip()
        .ok_or_else(|| format!("unsupported bind address: {bind}"))?;
    let served: ClusterOpt = cluster.map(|c| {
        Arc::new(LocalCluster {
            cache: Mutex::new(PeerCache::new(c)),
            view: Mutex::new(crate::cluster::ClusterView::default()),
            last_pushed: Mutex::new(None),
        })
    });
    // join: pull every peer's absolute envelope once so this node starts from
    // a converged global view (one roundtrip per peer, at startup only)
    if let Some(cl) = &served {
        let peers = cl.cache.lock().unwrap().peers();
        for peer in peers {
            if let Some((_, body)) = http_get(peer, "/__zap/gossip_full") {
                if let Ok(env) = serde_json::from_str::<StatsEnvelope>(&body) {
                    cl.view.lock().unwrap().merge_peer(peer, env);
                }
            }
        }
    }
    std::thread::spawn(move || loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        match server.recv() {
            Ok(req) => {
                let index = index.clone();
                handle(req, &index, served.as_ref());
            }
            Err(e) => {
                eprintln!("[zap serve] recv error: {e}");
                break;
            }
        }
    });
    Ok(addr)
}

fn handle(mut req: Request, index: &Index, cluster: ClusterRef<'_>) {
    let url = req.url().to_string();
    let (path, query) = match url.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (url.clone(), String::new()),
    };
    let method = req.method().clone();
    match (&method, path.as_str()) {
        (Method::Get, "/health") => {
            reply(req, 200, r##"{"status":"ok"}"##);
        }
        (Method::Get, "/dashboard") => match dashboard_dir().get_file("index.html") {
            Some(f) => reply_html(req, 200, f.contents_utf8().unwrap_or("")),
            None => reply(req, 500, r#"{"error":"dashboard assets missing"}"#),
        },
        (Method::Get, "/dashboard/events") => {
            // SSE: stream stats frames to the page; each request handled on
            // its own thread so the serve loop (and the server) stay free
            std::thread::spawn(move || sse_stream(req));
        }
        (Method::Get, p) if p.starts_with("/dashboard/") => {
            let rel = &p["/dashboard/".len()..];
            match dashboard_dir().get_file(url_decode(rel)) {
                Some(f) => {
                    let mime = if rel.ends_with(".css") {
                        "text/css; charset=utf-8"
                    } else if rel.ends_with(".js") {
                        "application/javascript"
                    } else {
                        "text/plain; charset=utf-8"
                    };
                    let mut r = Response::from_string(f.contents_utf8().unwrap_or("").to_string());
                    r.add_header(
                        tiny_http::Header::from_bytes(&b"Content-Type"[..], mime.as_bytes())
                            .unwrap(),
                    );
                    let _ = req.respond(r.with_status_code(200));
                }
                None => reply(req, 404, r#"{"error":"not found"}"#),
            }
        }
        (Method::Get, "/stats") => stats_reply(req, index, cluster),
        (Method::Get, "/search") => match query_param(&query, "q") {
            Some(q) => {
                let k = query_param(&query, "k")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(10);
                SEARCHES.fetch_add(1, Ordering::Relaxed);
                search_reply(req, index, cluster, q, k);
            }
            None => {
                reply(req, 400, r#"{"error":"missing `q`"}"#);
            }
        },
        (Method::Post, "/search") => match read_body(&mut req).and_then(|b| json_search(&b)) {
            Ok((q, k)) => {
                SEARCHES.fetch_add(1, Ordering::Relaxed);
                search_reply(req, index, cluster, q, k);
            }
            Err(msg) => reply(req, 400, &msg),
        },
        (Method::Post, "/ingest") => match ingest(&mut req, index, cluster) {
            Ok(n) => {
                INGESTED.fetch_add(n as u64, Ordering::Relaxed);
                let body = serde_json::json!({
                    "accepted": n,
                    "docs": docs_total(index, cluster),
                    "committed": true,
                })
                .to_string();
                reply(req, 200, &body);
            }
            Err(msg) => reply(req, 400, &serde_json::json!({"error": msg}).to_string()),
        },
        (Method::Post, "/commit") => {
            index.commit();
            crate::bus::notify_stats();
            if let Some(cl) = cluster {
                push_own_stats(cl, index);
            }
            let body = serde_json::json!({ "committed": true, "docs": docs_total(index, cluster) })
                .to_string();
            reply(req, 200, &body);
        }
        (Method::Get, p) if p.starts_with("/doc/") => {
            let id = url_decode(&p["/doc/".len()..]);
            doc_get_reply(req, index, cluster, &id);
        }
        (Method::Delete, p) if p.starts_with("/doc/") => {
            let id = url_decode(&p["/doc/".len()..]);
            doc_delete_reply(req, index, cluster, &id);
        }

        // ---- cluster internals (never forwarded; always local shard) ----
        (Method::Post, "/__zap/gossip") => {
            let peer = query_param(&query, "peer").and_then(|p| p.parse::<SocketAddr>().ok());
            match (read_body(&mut req), peer) {
                (Ok(b), Some(peer)) => match serde_json::from_str::<StatsEnvelope>(&b) {
                    Ok(env) => {
                        gossip_handle(cluster, peer, env);
                        reply(req, 200, r#"{"ok":true}"#);
                    }
                    Err(e) => reply(
                        req,
                        400,
                        &serde_json::json!({"error": format!("bad envelope: {e}")}).to_string(),
                    ),
                },
                _ => reply(
                    req,
                    400,
                    &serde_json::json!({"error": "missing ?peer or bad body"}).to_string(),
                ),
            }
        }
        (Method::Get, "/__zap/gossip_full") => {
            let env = own_envelope(index);
            match serde_json::to_string(&env) {
                Ok(s) => reply(req, 200, &s),
                Err(e) => reply(req, 500, &format!("serialize: {e}")),
            }
        }
        (Method::Post, "/__zap/shard_search") => {
            match read_body(&mut req).and_then(|b| json_search(&b)) {
                Ok((q, k)) => shard_search_reply(req, index, cluster, q, k),
                Err(msg) => reply(req, 400, &msg),
            }
        }
        (Method::Post, "/__zap/shard_ingest") => match shard_ingest(&mut req, index, cluster) {
            Ok(n) => {
                let body = serde_json::json!({
                    "accepted": n,
                    "docs": index.num_docs(),
                    "committed": true,
                })
                .to_string();
                reply(req, 200, &body);
            }
            Err(msg) => reply(req, 400, &serde_json::json!({"error": msg}).to_string()),
        },
        (Method::Get, "/__zap/shard_stats") => {
            let body = serde_json::json!({
                "docs": index.num_docs(),
                "segments": index.num_segments(),
                "generation": index.generation(),
                "searches": SEARCHES.load(Ordering::Relaxed),
                "ingested": INGESTED.load(Ordering::Relaxed),
                "uptime_s": uptime_secs(),
            })
            .to_string();
            reply(req, 200, &body);
        }

        _ => {
            reply(req, 404, r#"{"error":"not found"}"#);
        }
    }
}

fn search_reply(req: Request, index: &Index, cluster: ClusterRef<'_>, q: String, k: usize) {
    let Some(cl) = cluster else {
        return match index.search(&q, k) {
            Ok(hits) => {
                let results: Vec<serde_json::Value> =
                    hits.iter().filter_map(|h| hit_row(index, h)).collect();
                reply(
                    req,
                    200,
                    &serde_json::json!({"results": results, "count": results.len()}).to_string(),
                );
            }
            Err(e) => reply(req, 400, &e.to_string()),
        };
    };
    let (code, body) = cluster_search(index, cl, &q, k);
    reply(req, code, &body);
}

// ---- search fan-out helpers -----------------------------------------------

/// one merged result row on the wire
#[derive(Debug, Clone)]
struct MergedRow {
    score: f32,
    id: Option<String>,
    docid: u32,
    doc: String,
    metadata: Option<serde_json::Value>,
}

fn merged_cmp(a: &MergedRow, b: &MergedRow) -> std::cmp::Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| a.docid.cmp(&b.docid))
        .then_with(|| a.id.cmp(&b.id))
}

/// wire row for one search hit (payload resolved through the doc store)
fn hit_row(index: &Index, h: &crate::search::SearchHit) -> Option<serde_json::Value> {
    let p = index.doc_payload(h.addr)?;
    let doc = p
        .fields
        .iter()
        .map(|(_name, text)| text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let mut v = serde_json::json!({
        "score": h.score,
        "id": p.id,
        "docid": h.addr.docid,
        "doc": doc,
    });
    if let Some(meta) = p.metadata {
        v["metadata"] = meta;
    }
    Some(v)
}

fn row_to_json(r: MergedRow) -> serde_json::Value {
    let mut v = serde_json::json!({
        "score": r.score,
        "id": r.id,
        "docid": r.docid,
        "doc": r.doc,
    });
    if let Some(m) = r.metadata {
        v["metadata"] = m;
    }
    v
}

fn row_from_json(v: &serde_json::Value) -> MergedRow {
    MergedRow {
        score: v["score"].as_f64().unwrap_or(0.0) as f32,
        id: v["id"].as_str().map(|s| s.to_owned()),
        docid: v["docid"].as_u64().unwrap_or(0) as u32,
        doc: v["doc"].as_str().unwrap_or("").to_owned(),
        metadata: v.get("metadata").cloned().filter(|m| !m.is_null()),
    }
}

/// coordinator: score the local shard and fan out to every peer CONCURRENTLY
/// (wall time = slowest peer + local eval, not the sum of all peers);
/// merge rows, truncate to k
fn cluster_search(index: &Index, cl: &LocalCluster, q: &str, k: usize) -> (u16, String) {
    let stats: Stats = {
        let mut view = cl.view.lock().unwrap();
        let env = global_to_envelope(view.stats());
        stats_from_envelope(env)
    };
    let peers = cl.cache.lock().unwrap().peers();
    let q_owned = q.to_owned();
    let (mut rows, degraded) = std::thread::scope(|scope| {
        let mut jobs: Vec<std::thread::ScopedJoinHandle<'_, Result<Vec<MergedRow>, String>>> =
            Vec::new();
        for peer in &peers {
            let q2 = q_owned.clone();
            jobs.push(scope.spawn(move || fanout_rows(*peer, &q2, k)));
        }
        // own shard runs on this thread while peers fan out
        let mut rows = match index.search_with_stats(q, k, stats) {
            Ok(hits) => hits
                .into_iter()
                .filter_map(|h| row_of_hit(index, &h))
                .collect(),
            Err(_) => Vec::new(),
        };
        let mut degraded = false;
        for jh in jobs {
            match jh.join() {
                Ok(Ok(mut part)) => rows.append(&mut part),
                Ok(Err(_)) | Err(_) => degraded = true,
            }
        }
        (rows, degraded)
    });
    rows.sort_by(merged_cmp);
    rows.truncate(k);
    let results: Vec<serde_json::Value> = rows.into_iter().map(row_to_json).collect();
    let n = results.len();
    (
        200,
        serde_json::json!({"results": results, "count": n, "degraded": degraded}).to_string(),
    )
}

/// direct row from a hit (no JSON roundtrip for the local shard)
fn row_of_hit(index: &Index, h: &crate::search::SearchHit) -> Option<MergedRow> {
    let p = index.doc_payload(h.addr)?;
    Some(MergedRow {
        score: h.score,
        id: p.id,
        docid: h.addr.docid,
        doc: p
            .fields
            .iter()
            .map(|(_name, text)| text.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        metadata: p.metadata,
    })
}

/// merged global view on the wire
fn global_to_envelope(g: crate::cluster::GlobalStats) -> StatsEnvelope {
    let mut df = std::collections::HashMap::new();
    for ((fid, term), v) in g.df {
        df.insert(encode_df_key(fid, &term), v);
    }
    StatsEnvelope {
        docs: g.total_docs,
        sums: g.sums,
        df,
        full: false,
    }
}

/// fan-out one query to a shard; returns its top-k rows
fn fanout_rows(peer: SocketAddr, q: &str, k: usize) -> Result<Vec<MergedRow>, String> {
    let body = serde_json::json!({"q": q, "k": k}).to_string();
    let (_, resp) = http_post(peer, "/__zap/shard_search", &body)
        .ok_or_else(|| "peer unreachable".to_string())?;
    let v: serde_json::Value =
        serde_json::from_str(&resp).map_err(|e| format!("peer bad json: {e}"))?;
    let items = v
        .as_object()
        .and_then(|o| o.get("results"))
        .and_then(|r| r.as_array())
        .ok_or("peer response missing results")?;
    Ok(items.iter().map(row_from_json).collect())
}

/// shard-side handler: score ONLY our shard with OUR global view
fn shard_search_reply(req: Request, index: &Index, cluster: ClusterRef<'_>, q: String, k: usize) {
    let env: StatsEnvelope = match cluster {
        Some(cl) => global_to_envelope(cl.view.lock().unwrap().stats()),
        None => {
            let snap = index.pub_stats();
            envelope_from_stats(index, &snap)
        }
    };
    let stats = stats_from_envelope(env);
    match index.search_with_stats(&q, k, stats) {
        Ok(hits) => {
            let results: Vec<serde_json::Value> =
                hits.iter().filter_map(|h| hit_row(index, h)).collect();
            reply(
                req,
                200,
                &serde_json::json!({"results": results, "count": results.len()}).to_string(),
            );
        }
        Err(e) => reply(req, 200, &e.to_string()),
    }
}

fn json_search(body: &str) -> Result<(String, usize), String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("bad json: {e}"))?;
    let q = v
        .as_object()
        .and_then(|o| o.get("q"))
        .and_then(|x| x.as_str())
        .ok_or("missing `q`")?
        .to_string();
    let k = v
        .as_object()
        .and_then(|o| o.get("k"))
        .and_then(|x| x.as_u64())
        .unwrap_or(10) as usize;
    Ok((q, k))
}

/// NDJSON (one `{"text": ...}` per line) or a JSON array. Optional per doc:
/// `"id"` (string, unique — duplicates replace) and `"metadata"`.
/// Cluster mode routes every doc by ring slot to its owning shard.
fn ingest(req: &mut Request, index: &Index, cluster: ClusterRef<'_>) -> Result<usize, String> {
    let body = read_body(req)?;
    if body.is_empty() {
        return Err("empty ingest body".into());
    }
    let mut docs = Vec::new();
    if body.trim_start().starts_with('[') {
        let v: serde_json::Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
        let items = v.as_array().ok_or("expected document array")?;
        for it in items {
            docs.push(parse_doc(it)?);
        }
    } else {
        for line in body.lines() {
            let trim = line.trim();
            if trim.is_empty() {
                continue;
            }
            let v: serde_json::Value =
                serde_json::from_str(trim).map_err(|e| format!("bad ndjson line: {e}"))?;
            docs.push(parse_doc(&v)?);
        }
    }
    if docs.is_empty() {
        return Err("no documents supplied".into());
    }
    let Some(cl) = cluster else {
        let n = docs.len();
        index.add_documents(docs).map_err(|e| e.to_string())?;
        index.commit();
        crate::bus::notify_stats();
        return Ok(n);
    };
    // bucket by ring owner; unid'd docs key on content hash for stability
    let ring = cl.cache.lock().unwrap().ring();
    let self_addr = cl.cache.lock().unwrap().self_addr();
    let mut buckets: std::collections::HashMap<SocketAddr, Vec<Document>> =
        std::collections::HashMap::new();
    for doc in docs {
        let key = doc.id.clone().unwrap_or_else(|| {
            let text = doc.fields.get("text").map(|t| t.as_str()).unwrap_or("");
            format!("text:{}", crate::cluster::fnv64a(text.as_bytes()))
        });
        let slot = crate::cluster::ring_owner(&ring, &key).unwrap_or(0);
        buckets.entry(ring[slot]).or_default().push(doc);
    }
    // split: this shard's docs commit locally (in-thread), remote owner
    // batches ingest concurrently so wall time = slowest peer + local commit
    let (local_batch, remote): (Option<Vec<Document>>, Vec<ShardBatch>) = {
        let (l, r): (Vec<(SocketAddr, Vec<Document>)>, Vec<_>) = buckets
            .into_iter()
            .partition(|(owner, _)| *owner == self_addr);
        let local = l.into_iter().next().map(|(_, b)| b);
        (local, r)
    };
    // remote fan-out uses the same thread-scope pattern as /search; the local
    // commit overlaps with the peer POSTs inside the scope
    let (total, errors) = std::thread::scope(|scope| {
        let mut jobs = Vec::new();
        for (owner, batch) in &remote {
            let (owner, batch) = (*owner, batch.clone());
            jobs.push(scope.spawn(move || {
                let n = batch.len();
                let body = serde_json::json!({"docs": batch}).to_string();
                match http_post(owner, "/__zap/shard_ingest", &body) {
                    Some((200, _)) => Ok(n),
                    Some((code, b)) => Err(format!("shard {owner}: {code} {b}")),
                    None => Err(format!("shard {owner} unreachable")),
                }
            }));
        }
        // local commit runs on this thread while peers ingest
        let mut ok_total = 0usize;
        let mut errors = Vec::new();
        if let Some(batch) = &local_batch {
            if let Err(e) = index.add_documents(batch.clone()) {
                errors.push(e.to_string());
            } else {
                index.commit();
                crate::bus::notify_stats();
                ok_total += batch.len();
            }
        }
        for jh in jobs {
            match jh.join() {
                Ok(Ok(n)) => ok_total += n,
                Ok(Err(e)) => errors.push(e),
                Err(_) => errors.push("fanout thread panicked".to_string()),
            }
        }
        (ok_total, errors)
    });
    push_own_stats(cl, index);
    if errors.is_empty() {
        Ok(total)
    } else {
        Err(errors.join("; "))
    }
}

/// internal shard ingest (ring already chose this shard; never re-routed)
fn shard_ingest(
    req: &mut Request,
    index: &Index,
    cluster: ClusterRef<'_>,
) -> Result<usize, String> {
    let body = read_body(req)?;
    if body.is_empty() {
        return Err("empty ingest body".into());
    }
    let v: serde_json::Value = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    let arr = v
        .get("docs")
        .and_then(|d| d.as_array())
        .ok_or("missing docs array")?;
    let mut docs = Vec::with_capacity(arr.len());
    for it in arr {
        docs.push(parse_doc(it)?);
    }
    if docs.is_empty() {
        return Err("no documents supplied".into());
    }
    let n = docs.len();
    index.add_documents(docs).map_err(|e| e.to_string())?;
    index.commit();
    crate::bus::notify_stats();
    if let Some(cl) = cluster {
        push_own_stats(cl, index);
    }
    Ok(n)
}

fn parse_doc(v: &serde_json::Value) -> Result<Document, String> {
    let obj = v.as_object().ok_or("each doc must be a JSON object")?;
    let text = obj
        .get("text")
        .and_then(|x| x.as_str())
        .ok_or("each doc requires a string `text` field")?;
    let mut doc = Document::new().text("text", text);
    if let Some(id) = obj.get("id") {
        let id = id.as_str().ok_or("`id` must be a string")?;
        doc = doc.id(id);
    }
    if let Some(meta) = obj.get("metadata") {
        if !meta.is_null() {
            doc = doc.metadata(meta.clone());
        }
    }
    Ok(doc)
}

// ---- id routing (ring owners own /doc/{id}) -------------------------------

fn doc_get_reply(req: Request, index: &Index, cluster: ClusterRef<'_>, id: &str) {
    let Some(cl) = cluster else {
        return doc_get(req, index, id);
    };
    let (_, owner) = cl.cache.lock().unwrap().owner_of(id);
    if owner == cl.cache.lock().unwrap().self_addr() {
        return doc_get(req, index, id);
    }
    match http_get(owner, &format!("/doc/{}", url_escape(id))) {
        Some((code, body)) => reply(req, code, &body),
        None => reply(
            req,
            502,
            &serde_json::json!({"error": "owner shard unreachable"}).to_string(),
        ),
    }
}

fn doc_get(req: Request, index: &Index, id: &str) {
    match index.get(id) {
        Some(p) => {
            let mut v = serde_json::json!({
                "id": p.id,
                "fields": p.fields,
            });
            if let Some(meta) = p.metadata {
                v["metadata"] = meta;
            }
            reply(req, 200, &v.to_string());
        }
        None => reply(req, 404, r#"{"error":"not found"}"#),
    }
}

fn doc_delete_reply(req: Request, index: &Index, cluster: ClusterRef<'_>, id: &str) {
    let Some(cl) = cluster else {
        return doc_delete(req, index, id);
    };
    let (_, owner) = cl.cache.lock().unwrap().owner_of(id);
    if owner == cl.cache.lock().unwrap().self_addr() {
        return doc_delete(req, index, id);
    }
    match http_delete(owner, &format!("/doc/{}", url_escape(id))) {
        Some((code, body)) => reply(req, code, &body),
        None => reply(
            req,
            502,
            &serde_json::json!({"error": "owner shard unreachable"}).to_string(),
        ),
    }
}

fn doc_delete(req: Request, index: &Index, id: &str) {
    match index.delete(id) {
        Ok(true) => reply(
            req,
            200,
            &serde_json::json!({"deleted": true, "docs": index.num_docs()}).to_string(),
        ),
        Ok(false) => reply(req, 404, r#"{"error":"not found"}"#),
        Err(e) => reply(req, 400, &e.to_string()),
    }
}

/// minimal percent-escape for a path component
fn url_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

// ---- gossip ---------------------------------------------------------------

/// merge a peer's envelope; a previously unseen peer triggers a one-time full
/// resync pull from that peer so deltas land on a converged base
fn gossip_handle(cluster: ClusterRef<'_>, peer: SocketAddr, env: StatsEnvelope) {
    let Some(cl) = cluster else {
        return;
    };
    let mut view = cl.view.lock().unwrap();
    let fresh = view.is_new_peer(&peer);
    view.merge_peer(peer, env);
    if fresh {
        if let Some((_, body)) = http_get(peer, "/__zap/gossip_full") {
            if let Ok(full) = serde_json::from_str::<StatsEnvelope>(&body) {
                view.merge_peer(peer, full);
            }
        }
    }
}

/// cluster gossip push after a local commit: df-diff to every peer (spawned;
/// failures tolerated — the delta exchange is eventually-consistent)
fn push_own_stats(cl: &LocalCluster, index: &Index) {
    let env = own_envelope(index);
    let first = cl.last_pushed.lock().unwrap().is_none();
    let send = StatsEnvelope {
        docs: env.docs,
        sums: env.sums.clone(),
        df: if first {
            env.df.clone()
        } else {
            let last = cl.last_pushed.lock().unwrap();
            let base = last.as_ref().expect("just checked");
            let mut diff = std::collections::HashMap::new();
            for (k, v) in &env.df {
                if base.df.get(k) != Some(v) {
                    diff.insert(k.clone(), *v);
                }
            }
            diff
        },
        full: first,
    };
    *cl.last_pushed.lock().unwrap() = Some(env);
    let self_addr = cl.cache.lock().unwrap().self_addr();
    let payload = serde_json::to_string(&send).unwrap_or_default();
    for peer in cl.cache.lock().unwrap().peers() {
        let payload = payload.clone();
        std::thread::spawn(move || {
            let _ = http_post(peer, &format!("/__zap/gossip?peer={self_addr}"), &payload);
        });
    }
}

// ---- /stats ---------------------------------------------------------------

/// ONE concurrent fan-out per peer-contact moment: (peer, parsed stats-or-None)
fn fetch_peer_stats(cl: &LocalCluster) -> Vec<(SocketAddr, Option<serde_json::Value>)> {
    let peers = cl.cache.lock().unwrap().peers();
    std::thread::scope(|scope| {
        let jobs: Vec<_> = peers
            .iter()
            .map(|peer| scope.spawn(move || (*peer, http_get(*peer, "/__zap/shard_stats"))))
            .collect();
        jobs.into_iter()
            .map(|j| j.join().expect("peer stats fetch thread"))
            .map(|(peer, resp)| (peer, resp.and_then(|(_, b)| serde_json::from_str(&b).ok())))
            .collect()
    })
}

/// totals over self + live peers (docs, segments, max generation)
fn aggregate_stats(index: &Index, cluster: ClusterRef<'_>) -> (usize, usize, u64) {
    let mut docs = index.num_docs();
    let mut segments = index.num_segments();
    let mut generation = index.generation();
    if let Some(cl) = cluster {
        for (_, v) in fetch_peer_stats(cl)
            .into_iter()
            .filter_map(|(a, v)| v.map(|v| (a, v)))
        {
            docs += v["docs"].as_u64().unwrap_or(0) as usize;
            segments += v["segments"].as_u64().unwrap_or(0) as usize;
            generation = generation.max(v["generation"].as_u64().unwrap_or(0));
        }
    }
    (docs, segments, generation)
}

fn stats_reply(req: Request, index: &Index, cluster: ClusterRef<'_>) {
    // per-shard chips (self first, then peers that answered in time);
    // totals reuse the same single fan-out — one fetch per peer per request
    let (docs, segments, generation) = (index.num_docs(), index.num_segments(), index.generation());
    let mut docs_tot = docs as u64;
    let mut segments_tot = segments as u64;
    let mut generation_tot = generation;
    let mut shards: Vec<serde_json::Value> = vec![serde_json::json!({
        "addr": local_addr_of(cluster).unwrap_or_else(|| "self".to_string()),
        "docs": docs as u64,
        "segments": segments as u64,
        "generation": generation,
        "searches": SEARCHES.load(Ordering::Relaxed),
        "ingested": INGESTED.load(Ordering::Relaxed),
        "uptime_s": uptime_secs(),
    })];
    let mut degraded = false;
    if let Some(cl) = cluster {
        for (peer, v) in fetch_peer_stats(cl) {
            match v {
                Some(v) => {
                    docs_tot += v["docs"].as_u64().unwrap_or(0);
                    segments_tot += v["segments"].as_u64().unwrap_or(0);
                    generation_tot = generation_tot.max(v["generation"].as_u64().unwrap_or(0));
                    shards.push(serde_json::json!({
                        "addr": peer.to_string(),
                        "docs": v["docs"].as_u64().unwrap_or(0),
                        "segments": v["segments"].as_u64().unwrap_or(0),
                        "generation": v["generation"].as_u64().unwrap_or(0),
                        "searches": v["searches"].as_u64().unwrap_or(0),
                        "ingested": v["ingested"].as_u64().unwrap_or(0),
                        "uptime_s": v["uptime_s"].as_u64().unwrap_or(0),
                    }));
                }
                None => {
                    degraded = true;
                    shards.push(serde_json::json!({
                        "addr": peer.to_string(),
                        "docs": null, "segments": null,
                        "generation": null, "searches": null,
                        "ingested": null, "uptime_s": null,
                        "down": true,
                    }));
                }
            }
        }
    }
    let body = serde_json::json!({
        "docs": docs_tot as usize,
        "segments": segments_tot as usize,
        "generation": generation_tot,
        "shards": shards,
        "degraded": degraded,
    })
    .to_string();
    reply(req, 200, &body);
}

fn uptime_secs() -> u64 {
    SERVE_STARTED
        .get()
        .map(|t| t.elapsed().as_secs())
        .unwrap_or(0)
}

fn local_addr_of(cluster: ClusterRef<'_>) -> Option<String> {
    cluster.map(|c| c.cache.lock().unwrap().self_addr().to_string())
}

fn docs_total(index: &Index, cluster: ClusterRef<'_>) -> usize {
    aggregate_stats(index, cluster).0
}

fn read_body(req: &mut Request) -> Result<String, String> {
    let mut buf = Vec::new();
    let mut reader = req.as_reader().take(MAX_BODY as u64 + 1);
    reader.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    if buf.len() > MAX_BODY {
        return Err("body too large".into());
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn reply(req: Request, code: u16, body: &str) {
    let mut r = Response::from_string(body.to_string());
    r.add_header(
        tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
    );
    let _ = req.respond(r.with_status_code(code));
}

fn reply_html(req: Request, code: u16, body: &str) {
    let mut r = Response::from_string(body.to_string());
    r.add_header(
        tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..])
            .unwrap(),
    );
    let _ = req.respond(r.with_status_code(code));
}

// ---- /dashboard (compile-time embedded assets) -----------------------------

/// embedded at compile time: src/server/dashboard/{index.html, style.css, app.js}
static DASHBOARD: include_dir::Dir =
    include_dir::include_dir!("$CARGO_MANIFEST_DIR/src/server/dashboard");

fn dashboard_dir() -> &'static include_dir::Dir<'static> {
    &DASHBOARD
}

// ---- dashboard push (SSE over the event bus) -------------------------------

const SSE_HEARTBEAT: std::time::Duration = std::time::Duration::from_secs(2);

/// SSE via connection upgrade: the response carries the SSE headers, then
/// `upgrade` hijacks the raw socket and we write frames with explicit flushes
/// (tiny_http's internal buffering never sees frames smaller than the write
/// threshold, so direct socket writes are required for real-time push)
fn sse_stream(req: Request) {
    use std::io::Write;
    use tiny_http::Response;
    // hard cap: refuse beyond MAX_CLIENTS (thread-per-client bound)
    let Some((id, rx)) = crate::bus::subscribe() else {
        let mut r: Response<std::io::Empty> = Response::empty(503);
        if let Ok(h) = tiny_http::Header::from_bytes("Retry-After", "5") {
            r.add_header(h);
        }
        let _ = req.respond(r);
        return;
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3600);
    let mut r: Response<std::io::Empty> = Response::empty(200);
    for (k, v) in [
        ("Content-Type", "text/event-stream"),
        ("Cache-Control", "no-cache"),
    ] {
        if let Ok(h) = tiny_http::Header::from_bytes(k, v) {
            r.add_header(h);
        }
    }
    let mut stream = req.upgrade("zap-sse", r);
    loop {
        // force-cycle after 1h: browsers auto-reconnect (no zombie threads)
        if std::time::Instant::now() >= deadline {
            break;
        }
        let ev = rx
            .recv_timeout(SSE_HEARTBEAT)
            .unwrap_or(crate::bus::EVENT_STATS);
        let frame = format!("event: {ev}\ndata: {{}}\n\n");
        if stream.write_all(frame.as_bytes()).is_err() || stream.flush().is_err() {
            break;
        }
    }
    crate::bus::unsubscribe(id);
}

/// percent-decode ('+' = space); invalid escapes pass through raw
fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                if let (Some(h), Some(l)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                    out.push(h << 4 | l);
                    i += 3;
                } else {
                    out.push(b'%');
                    i += 1;
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    Some(match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'f' => b - b'A' + 10,
        _ => return None,
    })
}

fn query_param(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == key).then(|| url_decode(v))
    })
}
