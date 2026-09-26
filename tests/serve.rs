//! HTTP serve mode: ingest + search + stats roundtrip over a real TCP socket

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::atomic::AtomicBool;

use serde_json::{json, Value};
use zap::index::Index;
use zap::schema::Schema;

fn atomic_leak() -> &'static AtomicBool {
    Box::leak(Box::new(AtomicBool::new(false)))
}

/// tiny sync HTTP client: returns (status_code, body)
fn request_raw(addr: SocketAddr, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut s = TcpStream::connect(addr).expect("connect");
    let mut raw = format!("{method} {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n");
    if let Some(b) = body {
        raw += &format!("Content-Length: {}\r\n", b.len());
    }
    raw += "\r\n";
    if let Some(b) = body {
        raw += b;
    }
    s.write_all(raw.as_bytes()).expect("write request");
    let mut resp = Vec::new();
    s.read_to_end(&mut resp).expect("read response");
    let text = String::from_utf8_lossy(&resp).into_owned();
    // status line + headers + body
    let status: u16 = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_owned())
        .unwrap_or_default();
    (status, body)
}

fn get_json(addr: SocketAddr, path: &str) -> Value {
    let (code, body) = request_raw(addr, "GET", path, None);
    assert_eq!(code, 200, "GET {path}: {body}");
    serde_json::from_str(&body).expect("json body")
}

fn post_json(addr: SocketAddr, path: &str, body: &str) -> Value {
    let (code, b) = request_raw(addr, "POST", path, Some(body));
    let v: Value = serde_json::from_str(&b).expect("json body");
    assert_eq!(code, 200, "POST {path}: body {b}");
    v
}

fn start_serve() -> SocketAddr {
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    let analyzer = zap::analyzer::Analyzer::new(zap::analyzer::AnalyzerConf::default());
    let index = Index::with_lanes(schema, analyzer, 1.2, 0.75, 2);
    zap::server::serve(index, "127.0.0.1:0", None, atomic_leak()).expect("bind :0")
}

#[test]
fn serve_ingest_search_stats() {
    let addr = start_serve();
    // NDJSON ingest (two lines)
    let docs = "{\"text\":\"the quick brown fox\"}\n{\"text\":\"foxes are quick\"}\n";
    let v = post_json(addr, "/ingest", docs);
    assert_eq!(v["accepted"], json!(2), "{v}");
    assert_eq!(v["docs"], json!(2));
    // GET search with percent escape
    let v = get_json(addr, "/search?q=quick+fox&k=5");
    let results = v["results"].as_array().expect("results array");
    assert!(!results.is_empty(), "{v}");
    // JSON body search path
    let v = post_json(addr, "/search", r#"{"q":"fox","k":2}"#);
    let hits = v["results"].as_array().unwrap();
    assert!(hits
        .iter()
        .any(|r| r["doc"].as_str().unwrap().contains("fox")));
    // stats + health + commit
    let v = get_json(addr, "/stats");
    assert_eq!(v["docs"], json!(2));
    assert_eq!(get_json(addr, "/health")["status"], "ok");
    let v = post_json(addr, "/commit", "");
    assert_eq!(v["committed"], json!(true));
}

#[test]
fn serve_rejects_bad_ingest() {
    let addr = start_serve();
    let (code, b) = request_raw(addr, "POST", "/ingest", Some("{not json}"));
    assert_eq!(code, 400, "{b}");
    let v: Value = serde_json::from_str(&b).unwrap();
    assert!(!v["error"].as_str().unwrap_or_default().is_empty(), "{v}");
    // unknown path
    let (code, b) = request_raw(addr, "GET", "/nope", None);
    assert_eq!(code, 404, "BAD: {b:?}");
    assert_eq!(
        serde_json::from_str::<Value>(&b).unwrap()["error"],
        "not found"
    );
}

#[test]
fn serve_doc_id_metadata_lifecycle() {
    let addr = start_serve();
    // ingest with ids + metadata (NDJSON)
    let docs = concat!(
        "{\"id\":\"http-1\",\"text\":\"alpha bravo charlie\",\"metadata\":{\"src\":\"api\",\"n\":1}}\n",
        "{\"id\":\"http-2\",\"text\":\"bravo delta echo\"}\n",
        "{\"text\":\"no id here\"}\n"
    );
    let v = post_json(addr, "/ingest", docs);
    assert_eq!(v["accepted"], json!(3), "{v}");

    // get-by-id roundtrips id + metadata
    let v = get_json(addr, "/doc/http-1");
    assert_eq!(v["id"], json!("http-1"), "{v}");
    assert_eq!(v["metadata"]["src"], json!("api"));
    let v = get_json(addr, "/doc/http-2");
    assert_eq!(v["id"], json!("http-2"));
    assert!(v.get("metadata").is_none_or(|m| m.is_null()));
    // missing id -> 404
    let (code, b) = request_raw(addr, "GET", "/doc/nope", None);
    assert_eq!(code, 404, "{b}");

    // search results carry external id + metadata; internal docid present
    let v = post_json(addr, "/search", r#"{"q":"bravo","k":10}"#);
    let hits = v["results"].as_array().unwrap();
    assert_eq!(hits.len(), 2, "{v}");
    let ids: Vec<&str> = hits.iter().filter_map(|r| r["id"].as_str()).collect();
    assert!(ids.contains(&"http-1") && ids.contains(&"http-2"));
    assert!(hits[0].get("docid").is_some());

    // replace over HTTP: old text gone
    let v = post_json(
        addr,
        "/ingest",
        "{\"id\":\"http-1\",\"text\":\"zulu uniform victor\"}\n",
    );
    assert_eq!(v["accepted"], json!(1));
    assert_eq!(
        get_json(addr, "/stats")["docs"],
        json!(3),
        "replace keeps count"
    );
    let v = get_json(addr, "/doc/http-1");
    assert_eq!(v["fields"][0][1], json!("zulu uniform victor"));
    let v = post_json(addr, "/search", r#"{"q":"alpha","k":10}"#);
    assert_eq!(v["count"], json!(0), "old version must be invisible: {v}");

    // delete over HTTP
    let (code, b) = request_raw(addr, "DELETE", "/doc/http-2", None);
    assert_eq!(code, 200, "{b}");
    let v: Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["deleted"], json!(true));
    assert_eq!(get_json(addr, "/stats")["docs"], json!(2));
    let (code, _) = request_raw(addr, "GET", "/doc/http-2", None);
    assert_eq!(code, 404);
    let (code, _) = request_raw(addr, "DELETE", "/doc/http-2", None);
    assert_eq!(code, 404, "double delete misses");
    // deleted doc invisible to search
    let v = post_json(addr, "/search", r#"{"q":"delta","k":10}"#);
    assert_eq!(v["count"], json!(0), "{v}");
}

#[test]
fn dashboard_served_with_mocha() {
    // /dashboard = live HTML stats page in catppuccin mocha colors, assets
    // embedded at compile time (src/server/dashboard/*)
    let addr = start_serve();
    let (code, body) = request_raw(addr, "GET", "/dashboard", None);
    assert_eq!(code, 200, "{body}");
    assert!(body.contains("catppuccin"), "palette attribution missing");
    assert!(body.contains("/dashboard/style.css"), "css link missing");
    assert!(body.contains("/dashboard/app.js"), "js link missing");
    let (code, css) = request_raw(addr, "GET", "/dashboard/style.css", None);
    assert_eq!(code, 200, "{css}");
    assert!(css.contains("--base: #1e1e2e"), "mocha base color missing");
    assert!(css.contains("--mauve: #cba6f7"), "mocha accent missing");
    let (code, js) = request_raw(addr, "GET", "/dashboard/app.js", None);
    assert_eq!(code, 200, "{js}");
    assert!(js.contains("fetch(\"/stats\")"), "stats polling missing");
    let (code, _) = request_raw(addr, "GET", "/dashboard/nope.css", None);
    assert_eq!(code, 404, "unknown asset should miss");
    // /stats carries the shard chip + counters the dashboard renders
    let v: Value = get_json(addr, "/stats");
    assert!(v["shards"].as_array().is_some(), "{v}");
    assert!(v["shards"][0]["searches"].as_u64().is_some(), "{v}");
}
