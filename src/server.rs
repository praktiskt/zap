//! HTTP serve mode: NDJSON ingest + /search endpoints over the in-memory index
//!
//! The engine is CPU-bound and sync, so the HTTP layer stays threaded
//! (`tiny_http`); no async runtime is involved. Endpoints:
//! - `GET  /health`            liveness
//! - `GET  /stats`             docs / segments / generation
//! - `GET  /search?q=..&k=..`  query string search
//! - `POST /search`  `{"q":..,"k":..}` body search
//! - `POST /ingest`   NDJSON `{"text":..}` lines or a JSON array of them
//! - `POST /commit`   explicit segment commit

use std::io::Read;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tiny_http::{Method, Request, Response, Server};

use crate::index::Index;
use crate::schema::Document;

const MAX_BODY: usize = 64 << 20; // 64 MB ingest ceiling per request

/// run the HTTP listener until `shutdown` flips; returns the bound address
pub fn serve(
    index: Arc<Index>,
    bind: &str,
    shutdown: &'static AtomicBool,
) -> Result<SocketAddr, String> {
    let server = Server::http(bind).map_err(|e| format!("bind {bind}: {e}"))?;
    let addr: SocketAddr = server
        .server_addr()
        .to_ip()
        .ok_or_else(|| format!("unsupported bind address: {bind}"))?;
    std::thread::spawn(move || loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        match server.recv() {
            Ok(req) => {
                let index = index.clone();
                handle(req, &index);
            }
            Err(e) => {
                eprintln!("[zap serve] recv error: {e}");
                break;
            }
        }
    });
    Ok(addr)
}

fn handle(mut req: Request, index: &Index) {
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
        (Method::Get, "/stats") => {
            let body = serde_json::json!({
                "docs": index.num_docs(),
                "segments": index.num_segments(),
                "generation": index.generation(),
            })
            .to_string();
            reply(req, 200, &body);
        }
        (Method::Get, "/search") => match query_param(&query, "q") {
            Some(q) => {
                let k = query_param(&query, "k")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(10);
                search_reply(req, index, q, k);
            }
            None => {
                reply(req, 400, r#"{"error":"missing `q`"}"#);
            }
        },
        (Method::Post, "/search") => match read_body(&mut req).and_then(|b| json_search(&b)) {
            Ok((q, k)) => search_reply(req, index, q, k),
            Err(msg) => reply(req, 400, &msg),
        },
        (Method::Post, "/ingest") => match ingest(&mut req, index) {
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
        (Method::Post, "/commit") => {
            index.commit();
            let body =
                serde_json::json!({ "committed": true, "docs": index.num_docs() }).to_string();
            reply(req, 200, &body);
        }
        (Method::Get, p) if p.starts_with("/doc/") => {
            let id = url_decode(&p["/doc/".len()..]);
            match index.get(&id) {
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
        (Method::Delete, p) if p.starts_with("/doc/") => {
            let id = url_decode(&p["/doc/".len()..]);
            match index.delete(&id) {
                Ok(true) => reply(
                    req,
                    200,
                    &serde_json::json!({ "deleted": true, "docs": index.num_docs() }).to_string(),
                ),
                Ok(false) => reply(req, 404, r#"{"error":"not found"}"#),
                Err(e) => reply(req, 400, &e.to_string()),
            }
        }
        _ => {
            reply(req, 404, r#"{"error":"not found"}"#);
        }
    }
}

fn search_reply(req: Request, index: &Index, q: String, k: usize) {
    match index.search(&q, k) {
        Ok(hits) => {
            let results: Vec<serde_json::Value> = hits
                .iter()
                .filter_map(|h| {
                    let p = index.doc_payload(h.addr)?;
                    let doc = p
                        .fields
                        .iter()
                        .map(|(_name, text)| text)
                        .cloned()
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
                })
                .collect();
            reply(
                req,
                200,
                &serde_json::json!({ "results": results, "count": results.len() }).to_string(),
            );
        }
        Err(e) => reply(req, 400, &e.to_string()),
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

/// NDJSON (one `{"text": ...}` per line) or a JSON array of such objects.
/// Optional per doc: `"id"` (string, unique — duplicates replace) and
/// `"metadata"` (arbitrary JSON, stored-only).
fn ingest(req: &mut Request, index: &Index) -> Result<usize, String> {
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
    let n = docs.len();
    index.add_documents(docs).map_err(|e| e.to_string())?;
    index.commit();
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
