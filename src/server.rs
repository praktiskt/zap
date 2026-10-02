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
    http_delete, http_get, http_post, LocalCluster, PeerCache, PeerConfig, StatsEnvelope,
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
    Stats {
        total_docs: env.docs,
        avg_len: avg,
        df: env.df,
    }
}

/// absolute envelope from engine stats; sums come from the snapshot's exact
/// live sums (maintained incrementally), df is an Arc move — both O(fields)
fn envelope_from_stats(index: &Index, stats: &Stats) -> StatsEnvelope {
    let sums = index.field_sums();
    StatsEnvelope {
        docs: stats.total_docs,
        sums,
        df: stats.df.clone(),
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
    resync: bool,
) -> Result<SocketAddr, String> {
    let server = Server::http(bind).map_err(|e| format!("bind {bind}: {e}"))?;
    SERVE_STARTED.set(std::time::Instant::now()).ok();
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
    Ok(served_maps(server, index, served, shutdown, resync))
}

fn served_maps(
    server: Server,
    index: Arc<Index>,
    served: ClusterOpt,
    shutdown: &'static AtomicBool,
    resync: bool,
) -> SocketAddr {
    let addr: SocketAddr = server
        .server_addr()
        .to_ip()
        .expect("server addr already resolved");
    // membership announcement + boot survey: a joining node tells every
    // configured peer its announce address (join merge, idempotent), then
    // under --resync runs its OWN owned-range pull. The pull is unconditional
    // (NOT gated on an empty index): ring-routed ingests can land rows on a
    // joining node BEFORE its own pull runs, and rows ingested before the
    // join may still sit on the pre-join owner's stale copy — the survey
    // fetches owned rows from the peers and ack-deletes them there; the
    // nodes that already hold their share just re-ingest (replace semantics)
    if let Some(s) = &served {
        let scar = s.clone();
        let idx2 = index.clone();
        std::thread::spawn(move || {
            // yield the listener a moment so peers can talk back
            std::thread::sleep(std::time::Duration::from_millis(800));
            let me = scar.cache.lock().unwrap().self_addr();
            let peers = scar.cache.lock().unwrap().peers();
            for peer in peers {
                let body = serde_json::json!({ "join": me.to_string() }).to_string();
                if let Some((code, resp)) = http_post(peer, "/__zap/peers", &body) {
                    if code != 200 {
                        eprintln!("[join] peer {peer} rejected join: {resp}");
                    }
                } else {
                    eprintln!("[join] peer {peer} unreachable at join announce");
                }
            }
            if resync {
                // the peers must know us (and our slot ownership) before we
                // can survey our owned range. The survey is unconditional:
                // rows may already have arrived via ring-routed ingest AND
                // rows from before the join may sit on a pre-join owner's
                // stale copy; pull_own_range replaces owned rows (no-op) and
                // ack-deletes the stale source copies
                std::thread::sleep(std::time::Duration::from_millis(600));
                let n = pull_own_range(&idx2, &scar, true);
                if n > 0 {
                    eprintln!("[handoff] boot resync moved {n} docs");
                } else {
                    eprintln!("[handoff] boot resync: nothing to move");
                }
            }
        });
    }
    std::thread::spawn(move || {
        while !shutdown.load(Ordering::SeqCst) {
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
        }
    });
    addr
}

/// pull this shard's SEGMENT of the ring back from the peers: pages through
/// each peer's `/__zap/handoff` stream, re-ingests rows this node holds
/// (owner OR replica slots) and ack-deletes the id-carrying ones on the
/// source ONLY when this node owns the row outright (the replica copy must
/// stay at the source: replication requires R holders, the ack is a MOVE).
/// For shard movement / empty-node recovery; unid rows re-ingest without ack.
pub fn pull_own_range(index: &Index, cluster: &LocalCluster, verbose: bool) -> usize {
    // one survey at a time per node: overlapping membership-change pulls and
    // boot surveys would double-commit their batches AND storm the peers'
    // single-threaded tiny_http accept loop (that storm showed as transient
    // connection refusals on the receiving shards during repartitions)
    static SURVEYING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if SURVEYING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return 0;
    }
    let peers = cluster.cache.lock().unwrap().peers();
    let ring = cluster.cache.lock().unwrap().ring();
    let self_addr = cluster.cache.lock().unwrap().self_addr();
    let self_idx = ring.iter().position(|r| r == &self_addr);
    let mut moved = 0usize;
    for peer in peers {
        let mut cursor: Option<String> = None;
        loop {
            let path = match &cursor {
                Some(c) => format!("/__zap/handoff?after={c}&limit=200"),
                None => "/__zap/handoff?limit=200".to_string(),
            };
            let Some((_, body)) = http_get(peer, &path) else {
                if verbose {
                    eprintln!("[handoff] peer {peer} unreachable, stopping this source");
                }
                break;
            };
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) else {
                break;
            };
            let docs: Vec<serde_json::Value> = v
                .get("docs")
                .and_then(|d| d.as_array())
                .cloned()
                .unwrap_or_default();
            let next = v
                .get("next")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string());
            if docs.is_empty() {
                break;
            }
            let mut batch: Vec<Document> = Vec::with_capacity(docs.len());
            for d in docs {
                // re-home rows this node holds: owner slot OR replica slot
                // (slot+k rotation). NO ACK-DELETE here: with replication the
                // source copy may be the legitimate replica (R holders stay),
                // not a dup-purge-safety stale copy; stale rows left behind by
                // membership moves are removed by the search's id-dedup and
                // the /doc holder-first routing until an explicit purge pass
                // (search + /doc both resolve correctly regardless)
                match serde_json::from_value::<Document>(d.clone()) {
                    Ok(doc) => {
                        let key = doc.id.clone().unwrap_or_else(|| {
                            let text = doc.fields.get("text").map(|t| t.as_str()).unwrap_or("");
                            format!("text:{}", crate::cluster::fnv64a(text.as_bytes()))
                        });
                        let holders = crate::cluster::ring_holder_indices(&ring, &key);
                        let holds = self_idx.is_some_and(|si| holders.contains(&si));
                        if !holds {
                            continue;
                        }
                        batch.push(doc);
                    }
                    Err(e) if verbose => eprintln!("[handoff] row skipped: {e}"),
                    Err(_) => {}
                }
            }
            if batch.is_empty() {
                break;
            }
            let pulled = batch.len();
            // the engine's add_documents holds the inner mutex for the WHOLE
            // batch (tokenization happens under the lock). A 150-doc page at
            // once stalled the single-threaded tiny_http accept loop for
            // seconds and starved every handler including /health undo. Feed
            // the survey's page into the engine in small groups so concurrent
            // /ingest + /search handlers interleave between groups.
            for chunk_slice in batch.chunks(12) {
                if index.add_documents(chunk_slice.to_vec()).is_err() {
                    SURVEYING.store(false, std::sync::atomic::Ordering::SeqCst);
                    crate::bus::notify_stats();
                    return moved;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            moved += pulled;
            if verbose {
                eprintln!("[handoff] pulled {pulled} docs from {peer}");
            }
            cursor = next;
            if cursor.is_none() {
                break;
            }
        }
    }
    // ONE commit per survey (not per page): publish_snapshot re-partitions
    // caller-visible snapshots; per-page commits under the survey's own
    // add_documents contention blocked the single-threaded accept loop long
    // enough for connection refusals on the receiving shards
    index.commit();
    SURVEYING.store(false, std::sync::atomic::Ordering::SeqCst);
    crate::bus::notify_stats();
    moved
}

fn handle(mut req: Request, index: &Arc<Index>, cluster: ClusterRef<'_>) {
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
            // holder-walks carry a hop depth: forwarded requests are answered
            // locally only, so a replica miss can never loop A->B->A forever
            doc_get_reply(req, index, cluster, &id, query_hop(&url));
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
        // replica write path: the OWNER pushed this batch; store locally and
        // commit, DO NOT re-push (replication terminates here)
        (Method::Post, "/__zap/replica_push") => match read_body(&mut req) {
            Ok(body) => {
                // typed parse: one pass, no Value tree + per-row clones
                let wire: DocsWire = serde_json::from_str(&body).unwrap_or(DocsWire {
                    docs: Vec::new(),
                    replica: true,
                });
                let docs: Vec<Document> = wire
                    .docs
                    .into_iter()
                    .filter(|d| !d.fields.is_empty())
                    .collect();
                let n = docs.len();
                if n > 0 && index.add_documents(docs).is_ok() {
                    index.commit();
                    crate::bus::notify_stats();
                    if let Some(cl) = cluster {
                        push_own_stats(cl, index);
                    }
                }
                reply(req, 200, &serde_json::json!({"replicated": n}).to_string());
            }
            Err(e) => reply(req, 400, &serde_json::json!({"error": e}).to_string()),
        },
        // replica delete path: the OWNER pushed the tombstone id list; apply
        // locally (no re-propagation)
        (Method::Post, "/__zap/replica_delete") => match read_body(&mut req) {
            Ok(body) => {
                let v = serde_json::from_str::<serde_json::Value>(&body)
                    .map_err(|e| e.to_string())
                    .unwrap_or_default();
                let ids: Vec<&str> = v
                    .get("ids")
                    .and_then(|i| i.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
                    .unwrap_or_default();
                let mut n = 0u32;
                for id in ids {
                    if index.delete(id).is_ok_and(|r| r) {
                        n += 1;
                    }
                }
                crate::bus::notify_stats();
                reply(req, 200, &serde_json::json!({"deleted": n}).to_string());
            }
            Err(e) => reply(req, 400, &serde_json::json!({"error": e}).to_string()),
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
        // handoff: paginated live-doc stream of this shard (shard movement +
        // recovery pull). `after` = "seg:docid" cursor, `limit` max 500 rows.
        (Method::Get, "/__zap/handoff") => {
            let url = req.url().to_string();
            let mut after = None;
            let mut limit = 200usize;
            for kv in url
                .split('?')
                .nth(1)
                .map(|q| q.split('&'))
                .into_iter()
                .flatten()
            {
                let (k, val) = kv.split_once('=').unwrap_or((kv, ""));
                match k {
                    "after" if !val.is_empty() => {
                        after = val
                            .split_once(':')
                            .and_then(|(s, d)| Some((s.parse().ok()?, d.parse().ok()?)));
                    }
                    "limit" => {
                        if let Ok(n) = val.parse::<usize>() {
                            limit = n.clamp(1, 500);
                        }
                    }
                    _ => {}
                }
            }
            let (docs, next) = index.docs_page(after, limit);
            let body = serde_json::json!({
                "docs": docs,
                "next": next.map(|(s, d)| format!("{s}:{d}")),
            })
            .to_string();
            reply(req, 200, &body);
        }
        // delete this shard's copies of the migrated ids once the puller acked
        (Method::Post, "/__zap/handoff_delete") => match read_body(&mut req) {
            Ok(body) => {
                let v = match serde_json::from_str::<serde_json::Value>(&body) {
                    Ok(v) => v,
                    Err(e) => {
                        reply(
                            req,
                            400,
                            &serde_json::json!({"error": e.to_string()}).to_string(),
                        );
                        return;
                    }
                };
                let ids: Vec<&str> = v
                    .get("ids")
                    .and_then(|i| i.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
                    .unwrap_or_default();
                let mut n = 0u32;
                for id in ids {
                    if index.delete(id).is_ok_and(|r| r) {
                        n += 1;
                    }
                }
                crate::bus::notify_stats();
                reply(req, 200, &serde_json::json!({"deleted": n}).to_string());
            }
            Err(e) => reply(req, 400, &serde_json::json!({"error": e}).to_string()),
        },
        // trigger a resync pull of this node's ring range from the peers
        // (post-boot shard movement / recovery; runs in the background)
        (Method::Post, "/__zap/resync") => match cluster.map(Arc::clone) {
            Some(cl) => {
                let idx = index.clone();
                let handle = std::thread::Builder::new()
                    .name("handoff-resync".into())
                    .spawn(move || {
                        let n = pull_own_range(&idx, &cl, true);
                        eprintln!("[handoff] resync moved {n} docs");
                    });
                if handle.is_err() {
                    reply(req, 500, r#"{"error":"thread spawn failed"}"#);
                    return;
                }
                reply(req, 202, r#"{"resync":"started"}"#);
            }
            None => reply(req, 400, r#"{"error":"single-node (no peers)"}"#),
        },
        // ring repartition: replace this node's peer list in place. Every node
        // must apply the SAME membership or the ring views diverge; the caller
        // (operator or the new member's boot) fans this out before a resync.
        (Method::Get, "/__zap/peers") => match cluster {
            Some(cl) => {
                let cache = cl.cache.lock().unwrap();
                let cfg = serde_json::json!({
                    "self": cache.self_addr().to_string(),
                    "peers": cache.peers_csv(),
                });
                reply(req, 200, &cfg.to_string());
            }
            None => reply(req, 200, r#"{"peers":""}"#),
        },
        (Method::Post, "/__zap/peers") => match cluster {
            Some(cl) => match read_body(&mut req) {
                Ok(body) => {
                    let v = serde_json::from_str::<serde_json::Value>(&body).unwrap_or_default();
                    let mut cache = cl.cache.lock().unwrap();
                    // did the membership actually change? joins/replacements
                    // that don't alter the ring skip the re-pull below
                    let (peer_count, _self_addr) = (cache.peers_csv().len(), cache.self_addr());
                    let outcome = if let Some(join) = v.get("join").and_then(|p| p.as_str()) {
                        // membership merge: one announced address joins the
                        // ring (new node boot fan-out / manual widen)
                        match join.parse::<SocketAddr>() {
                            Ok(addr) => {
                                let added = cache.join_peer(addr);
                                Some(Ok(("joined", added)))
                            }
                            Err(e) => Some(Err(e.to_string())),
                        }
                    } else {
                        // full repartition: replace the peer list in place
                        let csv = v
                            .get("peers")
                            .and_then(|p| p.as_str())
                            .unwrap_or("")
                            .to_string();
                        match cache.set_peers(&csv) {
                            Ok(()) => Some(Ok(("replaced", true))),
                            Err(e) => Some(Err(e)),
                        }
                    };
                    let changed = match &outcome {
                        Some(Ok((_, _added))) => cache.peers_csv().len() != peer_count,
                        _ => false,
                    };
                    drop(cache);
                    let body = match outcome {
                        None => {
                            let cache = cl.cache.lock().unwrap();
                            serde_json::json!({
                                "self": cache.self_addr().to_string(),
                                "peers": cache.peers_csv(),
                            })
                        }
                        Some(Ok((kind, _))) => {
                            let cache = cl.cache.lock().unwrap();
                            serde_json::json!({
                                "self": cache.self_addr().to_string(),
                                "peers": cache.peers_csv(),
                                "result": kind,
                            })
                        }
                        Some(Err(e)) => {
                            reply(req, 400, &serde_json::json!({"error": e}).to_string());
                            return;
                        }
                    };
                    // a real membership change re-partitions the ring: rows
                    // this node newly owns may sit on any peer's stale copy —
                    // re-pull the owned range until the ownership settles
                    if changed {
                        let idx = index.clone();
                        let cl2 = Arc::clone(cl);
                        let _ = std::thread::Builder::new()
                            .name("membership-resync".into())
                            .spawn(move || {
                                std::thread::sleep(std::time::Duration::from_millis(1000));
                                let n = pull_own_range(&idx, &cl2, false);
                                if n > 0 {
                                    eprintln!("[handoff] membership change pulled {n} docs");
                                }
                            });
                    }
                    reply(req, 200, &body.to_string());
                }
                Err(e) => reply(req, 400, &serde_json::json!({"error": e}).to_string()),
            },
            None => reply(req, 400, r#"{"error":"single-node (no cluster runtime)"}"#),
        },

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
/// concatenated text of a payload's stored fields (the search surface's "doc")
fn joined_fields(p: &crate::index::DocPayload) -> String {
    p.fields
        .iter()
        .map(|(_name, text)| text.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn hit_row(index: &Index, h: &crate::search::SearchHit) -> Option<serde_json::Value> {
    let p = index.doc_payload(h.addr)?;
    let doc = joined_fields(&p);
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
        let g = view.stats();
        Stats {
            total_docs: g.total_docs,
            avg_len: g
                .sums
                .iter()
                .map(|s| {
                    if g.total_docs > 0 {
                        *s as f32 / g.total_docs as f32
                    } else {
                        1.0
                    }
                })
                .collect(),
            df: g.df,
        }
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
    // replication means the same doc can surface on R shards — and its replica
    // copy may be a STALE pre-replace version: sort by score, then keep only
    // the FIRST (highest-scored) row per id
    rows.sort_by(merged_cmp);
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    rows.retain(|r| match &r.id {
        Some(id) => seen.insert(id.clone()),
        None => true,
    });
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
    let doc = joined_fields(&p);
    Some(MergedRow {
        score: h.score,
        id: p.id,
        docid: h.addr.docid,
        doc,
        metadata: p.metadata,
    })
}

/// merged global view on the wire
fn global_to_envelope(g: crate::cluster::GlobalStats) -> StatsEnvelope {
    StatsEnvelope {
        docs: g.total_docs,
        sums: g.sums,
        df: g.df,
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
    // remote fan-out uses the same thread-scope pattern as /search; remote
    // owners ingest concurrently while the LOCAL batch commits + performs its
    // replica duty INLINE on this thread (no self-POST loopback: the tiny_http
    // coordinator thread is busy serving this request)
    let mut errors = Vec::new();
    let mut ok_total = 0usize;
    std::thread::scope(|scope| {
        let mut jobs = Vec::new();
        for (owner, batch) in &remote {
            let (owner, batch) = (*owner, batch.clone());
            jobs.push(scope.spawn(move || {
                let n = batch.len();
                let body = serde_json::to_string(&DocsWire {
                    docs: batch,
                    replica: false,
                })
                .unwrap_or_default();
                match http_post(owner, "/__zap/shard_ingest", &body) {
                    Some((200, _)) => Ok(n),
                    Some((code, b)) => Err(format!("shard {owner}: {code} {b}")),
                    None => Err(format!("shard {owner} unreachable")),
                }
            }));
        }
        if let Some(batch) = local_batch {
            match owner_commit_and_replicate(index, Some(cl), batch) {
                Ok(n) => ok_total += n,
                Err(e) => errors.push(e),
            }
        }
        for jh in jobs {
            match jh.join() {
                Ok(Ok(n)) => ok_total += n,
                Ok(Err(e)) => errors.push(e),
                Err(_) => errors.push("fanout thread panicked".to_string()),
            }
        }
    });
    push_own_stats(cl, index);
    if errors.is_empty() {
        Ok(ok_total)
    } else {
        Err(errors.join("; "))
    }
}

/// ingest a batch on THIS shard (owner duty), then push its own replica
/// copies to the (slot+k)%N rotation holders. Direct helper: both the
/// coordinator's local-lane and the /__zap/shard_ingest receipt call it
fn owner_commit_and_replicate(
    index: &Index,
    cl: Option<&LocalCluster>,
    batch: Vec<Document>,
) -> Result<usize, String> {
    let n = batch.len();
    if let Some(cl) = cl {
        // replica re-bucketing happens BEFORE the consume: the docs move into
        // the engine without any defensive deep clone (old code ran
        // batch.to_vec()/clone() chains for a double copy of every field)
        let ring = cl.cache.lock().unwrap().ring();
        let self_addr = cl.cache.lock().unwrap().self_addr();
        let mut reps: std::collections::HashMap<SocketAddr, Vec<Document>> =
            std::collections::HashMap::new();
        for doc in &batch {
            let key = doc.id.clone().unwrap_or_else(|| {
                let text = doc.fields.get("text").map(|t| t.as_str()).unwrap_or("");
                format!("text:{}", crate::cluster::fnv64a(text.as_bytes()))
            });
            let holders = crate::cluster::ring_holder_indices(&ring, &key);
            for (k, idx) in holders.iter().enumerate() {
                let holder = ring[*idx];
                if k > 0 && holder != self_addr {
                    reps.entry(holder).or_default().push(doc.clone());
                }
            }
        }
        for (peer, rep_batch) in reps {
            // replica copies MUST be cloned (they ship to another shard);
            // build the wire body from a typed struct, not a Value tree
            let body = serde_json::to_string(&DocsWire {
                docs: rep_batch,
                replica: true,
            })
            .unwrap_or_default();
            let _ = std::thread::Builder::new()
                .name("replica-push".into())
                .spawn(move || {
                    let sent = http_post(peer, "/__zap/replica_push", &body);
                    if sent.is_none() {
                        std::thread::sleep(std::time::Duration::from_millis(150));
                        let _retry = http_post(peer, "/__zap/replica_push", &body);
                    }
                });
        }
    }
    // local commit — the replica push must never serializer-wedge the
    // ingest handler (a synchronous push can deadlock against the
    // coordinator's own busy tiny_http loop)
    index.add_documents(batch).map_err(|e| e.to_string())?;
    index.commit();
    crate::bus::notify_stats();
    if let Some(cl) = cl {
        push_own_stats(cl, index);
    }
    Ok(n)
}

/// typed ingest wire shape: avoids the serde_json::Value tree roundtrip
#[derive(serde::Serialize, serde::Deserialize)]
struct DocsWire {
    docs: Vec<Document>,
    #[serde(default)]
    replica: bool,
}
/// internal shard ingest (ring already chose this shard; never re-routed).
/// The owner performs its replication duty via owner_commit_and_replicate
fn shard_ingest(
    req: &mut Request,
    index: &Index,
    cluster: ClusterRef<'_>,
) -> Result<usize, String> {
    let body = read_body(req)?;
    if body.is_empty() {
        return Err("empty ingest body".into());
    }
    let wire: DocsWire = serde_json::from_str(&body).map_err(|e| e.to_string())?;
    let docs: Vec<Document> = wire
        .docs
        .into_iter()
        .filter(|d| !d.fields.is_empty())
        .collect();
    if docs.is_empty() {
        return Err("no documents supplied".into());
    }
    owner_commit_and_replicate(index, cluster.map(|c| &**c), docs)
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

fn doc_get_reply(req: Request, index: &Index, cluster: ClusterRef<'_>, id: &str, hop: u8) {
    let Some(cl) = cluster else {
        return doc_get(req, index, id);
    };
    // walk the holder set [owner, replicas...]: the first 200 wins; a dead
    // owner must not hide rows the replica holds (that's the whole point of
    // REPLICA_FACTOR). Forwarded requests (hop > 0) answer LOCALLY ONLY so
    // two miss-holders can never redirect forever.
    if hop > 0 {
        return doc_get(req, index, id);
    }
    let ring = cl.cache.lock().unwrap().ring();
    let self_addr = cl.cache.lock().unwrap().self_addr();
    let holders = crate::cluster::ring_holder_indices(&ring, id);
    let path = format!("/doc/{}?hop={}", url_escape(id), hop + 1);
    for idx in holders {
        let addr = ring[idx];
        if addr == self_addr {
            let exists = index.get(id).is_some();
            if exists {
                return doc_get(req, index, id);
            }
            continue; // local miss: a replica may hold it
        }
        // forward retries: a single-threaded tiny_http owner can transiently
        // refuse under load (pull threads, ingest fan-outs, hammering).
        // bounded 3× — persistent refusal surfaces as a walk-on to the replica
        let mut attempt = http_get(addr, &path);
        for _ in 0..2 {
            if attempt.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(120));
            attempt = http_get(addr, &path);
        }
        match attempt {
            Some((200, body)) => reply(req, 200, &body),
            Some((404, _)) => continue,
            Some((code, body)) => reply(req, code, &body),
            None => continue,
        };
        return;
    }
    reply(req, 404, r#"{"error":"not found"}"#);
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
        return doc_delete(req, index, id, None);
    };
    // route to the OWNER (the write authority); the owner pushes the delete
    // to the replicas (id carries the whole list, so every copy hides)
    let ring = cl.cache.lock().unwrap().ring();
    let self_addr = cl.cache.lock().unwrap().self_addr();
    let holders = crate::cluster::ring_holder_indices(&ring, id);
    let Some(&first) = holders.first() else {
        return reply(req, 400, r#"{"error":"empty ring"}"#);
    };
    let owner = ring[first];
    if owner != self_addr {
        let path = format!("/doc/{}", url_escape(id));
        let mut attempt = http_delete(owner, &path);
        if attempt.is_none() {
            std::thread::sleep(std::time::Duration::from_millis(120));
            attempt = http_delete(owner, &path);
        }
        match attempt {
            Some((code, body)) => reply(req, code, &body),
            None => reply(
                req,
                502,
                &serde_json::json!({"error": "owner shard unreachable"}).to_string(),
            ),
        }
    } else {
        doc_delete(req, index, id, Some(cl));
    }
}

fn doc_delete(req: Request, index: &Index, id: &str, cluster: ClusterRef<'_>) {
    let res = index.delete(id);
    // the owner propagates the tombstone to the replica holders; the same
    // holder-set derivation keeps every copy in lockstep
    if let Ok(true) = res {
        if let Some(cl) = cluster {
            let ring = cl.cache.lock().unwrap().ring();
            let self_addr = cl.cache.lock().unwrap().self_addr();
            let holders = crate::cluster::ring_holder_indices(&ring, id);
            let mut reps: Vec<SocketAddr> = Vec::new();
            for (k, idx) in holders.iter().enumerate() {
                let addr = ring[*idx];
                if k > 0 && addr != self_addr {
                    reps.push(addr);
                }
            }
            if !reps.is_empty() {
                let body = serde_json::json!({ "ids": [id] }).to_string();
                for peer in reps {
                    let body = body.clone();
                    let _ = std::thread::Builder::new()
                        .name("replica-delete".into())
                        .spawn(move || {
                            let sent = http_post(peer, "/__zap/replica_delete", &body);
                            if sent.is_none() {
                                std::thread::sleep(std::time::Duration::from_millis(150));
                                let _retry = http_post(peer, "/__zap/replica_delete", &body);
                            }
                        });
                }
            }
        }
    }
    match res {
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
            // fast path: same df map (no vocabulary change since last push)
            // -> pointer-identical Arc, empty diff without an O(vocab) scan
            if Arc::ptr_eq(&env.df, &base.df) {
                Arc::new(std::collections::HashMap::new())
            } else {
                let mut diff: std::collections::HashMap<
                    u32,
                    std::collections::HashMap<String, u32>,
                > = std::collections::HashMap::new();
                for (fid, m) in env.df.iter() {
                    let Some(base_m) = base.df.get(fid) else {
                        diff.insert(*fid, m.clone());
                        continue;
                    };
                    for (term, v) in m.iter() {
                        if base_m.get(term) != Some(v) {
                            diff.entry(*fid).or_default().insert(term.clone(), *v);
                        }
                    }
                }
                Arc::new(diff)
            }
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
    respond_body(req, code, body, "application/json");
}

fn reply_html(req: Request, code: u16, body: &str) {
    respond_body(req, code, body, "text/html; charset=utf-8");
}

fn respond_body(req: Request, code: u16, body: &str, content_type: &str) {
    let mut r = Response::from_string(body.to_string());
    r.add_header(
        tiny_http::Header::from_bytes(&b"Content-Type"[..], content_type.as_bytes()).unwrap(),
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

/// /doc/{id}?hop=N: forwarded holder walks increment; hop>0 = answer locally
fn query_hop(url: &str) -> u8 {
    url.split('?')
        .nth(1)
        .and_then(|q| query_param(q, "hop"))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}
