//! embedding search over HTTP: ingest with embeddings, vector / hybrid /
//! filter queries, validation errors, the ops surface and a CLI disk reboot

use std::net::SocketAddr;
use std::sync::atomic::AtomicBool;

use serde_json::{json, Value};
use zap::analyzer::AnalyzerConf;
use zap::index::Index;
use zap::schema::{Metric, Schema};

mod common;
use common::request_raw;

fn atomic_leak() -> &'static AtomicBool {
    Box::leak(Box::new(AtomicBool::new(false)))
}

fn start_serve() -> SocketAddr {
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    schema.add_vector("emb", 3, Metric::Cosine);
    let analyzer = zap::analyzer::Analyzer::new(AnalyzerConf::default());
    let index = Index::with_lanes(schema, analyzer, 1.2, 0.75, 2);
    zap::server::serve(index, "127.0.0.1:0", None, atomic_leak(), false).expect("bind :0")
}

fn post(addr: SocketAddr, path: &str, body: &str) -> (u16, String) {
    request_raw(addr, "POST", path, Some(body))
}

fn get(addr: SocketAddr, path: &str) -> (u16, String) {
    request_raw(addr, "GET", path, None)
}

fn json_post(addr: SocketAddr, path: &str, body: &str) -> (u16, Value) {
    let (code, b) = post(addr, path, body);
    let v = serde_json::from_str(&b).unwrap_or(Value::Null);
    (code, v)
}

#[test]
fn ingest_and_search_by_embedding() {
    let addr = start_serve();
    // ingest with embeddings
    let (code, body) = post(
        addr,
        "/ingest",
        r#"{"id":"a","text":"the quick brown fox","emb":[1.0,0.0,0.0]}
{"id":"b","text":"a lazy dog","emb":[0.0,1.0,0.0]}
{"id":"c","text":"fox and dog","emb":[0.9,0.1,0.0]}"#,
    );
    assert_eq!(code, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["accepted"], json!(3), "{body}");

    // vector-only search
    let (code, v) = json_post(addr, "/search", r#"{"vector":[1.0,0.0,0.0],"k":3}"#);
    assert_eq!(code, 200, "{v}");
    let results = v["results"].as_array().expect("results");
    assert_eq!(results.len(), 3, "{v}");
    assert_eq!(results[0]["id"], json!("a"), "{v}");
    assert!(
        (results[0]["vector_score"].as_f64().unwrap() - 1.0).abs() < 1e-5,
        "{v}"
    );
    assert_eq!(results[1]["id"], json!("c"), "{v}");

    // GET form: comma-separated embedding
    let (code, body) = get(addr, "/search?vector=1,0,0&k=2");
    assert_eq!(code, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["results"][0]["id"], json!("a"), "{v}");

    // hybrid: keyword + embedding (the mode defaults to hybrid when both are
    // present)
    let (code, v) = json_post(
        addr,
        "/search",
        r#"{"q":"fox","vector":[0.0,1.0,0.0],"k":5,"mode":"hybrid","alpha":0.5}"#,
    );
    assert_eq!(code, 200, "{v}");
    let results = v["results"].as_array().expect("results");
    // the keyword component is present for the documents that matched the
    // query ("b" does not contain "fox" -> no keyword score)
    assert!(
        results
            .iter()
            .filter(|r| r["id"] != json!("b"))
            .all(|r| r["keyword_score"].is_number()),
        "{v}"
    );
    assert!(results.iter().all(|r| r["vector_score"].is_number()), "{v}");
    // "b" is the vector match, "a"/"c" are the keyword matches
    let ids: Vec<&str> = results
        .iter()
        .map(|r| r["id"].as_str().unwrap_or(""))
        .collect();
    assert!(ids.contains(&"b"), "{ids:?}");

    // filter mode: the keyword selects, the embedding ranks
    let (code, v) = json_post(
        addr,
        "/search",
        r#"{"q":"fox","vector":[0.9,0.1,0.0],"k":5,"mode":"filter"}"#,
    );
    assert_eq!(code, 200, "{v}");
    let ids: Vec<&str> = v["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap_or(""))
        .collect();
    // "b" does not contain "fox" -> filtered out
    assert_eq!(ids, vec!["c", "a"], "{ids:?}");

    // rrf fusion
    let (code, v) = json_post(
        addr,
        "/search",
        r#"{"q":"fox","vector":[1.0,0.0,0.0],"k":5,"mode":"hybrid","fusion":"rrf"}"#,
    );
    assert_eq!(code, 200, "{v}");
    assert_eq!(v["results"][0]["id"], json!("a"), "{v}");
}

#[test]
fn vector_queries_are_validated() {
    let addr = start_serve();
    let (code, body) = post(
        addr,
        "/ingest",
        r#"{"id":"a","text":"the fox runs","emb":[1.0,0.0,0.0]}"#,
    );
    assert_eq!(code, 200, "{body}");
    // wrong dimension
    let (code, v) = json_post(addr, "/search", r#"{"vector":[1.0,0.0]}"#);
    assert_eq!(code, 400, "{v}");
    assert!(
        v["error"].as_str().unwrap_or("").contains("dimensions"),
        "{v}"
    );
    // unknown vector field
    let (code, v) = json_post(
        addr,
        "/search",
        r#"{"vector":[1.0,0.0,0.0],"field":"nope"}"#,
    );
    assert_eq!(code, 400, "{v}");
    // vector mode without an embedding
    let (code, v) = json_post(addr, "/search", r#"{"q":"fox","mode":"vector"}"#);
    assert_eq!(code, 400, "{v}");
    // unknown mode
    let (code, v) = json_post(addr, "/search", r#"{"q":"fox","mode":"magic"}"#);
    assert_eq!(code, 400, "{v}");
    // keyword search still works
    let (code, v) = json_post(addr, "/search", r#"{"q":"fox"}"#);
    assert_eq!(code, 200, "{v}");
    assert_eq!(v["results"][0]["id"], json!("a"), "{v}");
}

#[test]
fn ingest_rejects_bad_embeddings() {
    let addr = start_serve();
    // wrong dimension
    let (code, body) = post(addr, "/ingest", r#"{"text":"x","emb":[1.0,2.0]}"#);
    assert_eq!(code, 400, "{body}");
    assert!(body.contains("dimensions"), "{body}");
    // unknown vector field
    let (code, body) = post(addr, "/ingest", r#"{"text":"x","nope":[1.0,2.0,3.0]}"#);
    assert_eq!(code, 400, "{body}");
    assert!(body.contains("unknown field"), "{body}");
    // non-numeric embedding
    let (code, body) = post(addr, "/ingest", r#"{"text":"x","emb":["a","b","c"]}"#);
    assert_eq!(code, 400, "{body}");
    // a good one still lands
    let (code, body) = post(addr, "/ingest", r#"{"text":"x","emb":[1.0,0.0,0.0]}"#);
    assert_eq!(code, 200, "{body}");
}

#[test]
fn stats_report_the_vector_fields() {
    let addr = start_serve();
    let (code, body) = post(
        addr,
        "/ingest",
        r#"{"id":"a","text":"x","emb":[1.0,0.0,0.0]}
{"id":"b","text":"y","emb":[0.0,1.0,0.0]}"#,
    );
    assert_eq!(code, 200, "{body}");
    let (code, body) = get(addr, "/stats");
    assert_eq!(code, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let vecs = v["vectors"].as_array().expect("vectors");
    assert_eq!(vecs.len(), 1, "{v}");
    assert_eq!(vecs[0]["field"], json!("emb"), "{v}");
    assert_eq!(vecs[0]["dim"], json!(3), "{v}");
    assert_eq!(vecs[0]["metric"], json!("cosine"), "{v}");
    assert_eq!(vecs[0]["docs"], json!(2), "{v}");
}

#[test]
fn cli_disk_reboot_keeps_the_embeddings() {
    // the CLI declares the vector field (--vector name:dim[:metric])
    let dir = std::env::temp_dir().join(format!("zap-vector-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let port = 47931u16;
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    fn spawn(port: u16, dir: &std::path::Path) -> std::process::Child {
        let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_zap"));
        cmd.args([
            "serve",
            "--bind",
            &format!("127.0.0.1:{port}"),
            "--disk",
            dir.to_str().unwrap(),
            "--vector",
            "emb:3",
        ]);
        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::null());
        cmd.spawn().expect("spawn zap serve")
    }

    // readiness probe that tolerates "not up yet"
    fn wait_up(addr: SocketAddr) {
        for _ in 0..200 {
            if std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(100))
                .is_ok()
            {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        panic!("server did not come up");
    }

    let mut child = spawn(port, &dir);
    wait_up(addr);
    let (code, body) = post(
        addr,
        "/ingest",
        r#"{"id":"a","text":"alpha","emb":[1.0,0.0,0.0]}"#,
    );
    assert_eq!(code, 200, "{body}");
    let _ = child.kill();
    let _ = child.wait();

    // reboot on the same directory: the embeddings must survive
    let mut child = spawn(port, &dir);
    wait_up(addr);
    let (code, v) = json_post(addr, "/search", r#"{"vector":[1.0,0.0,0.0],"k":3}"#);
    assert_eq!(code, 200, "{v}");
    assert_eq!(v["results"][0]["id"], json!("a"), "{v}");
    assert!(
        (v["results"][0]["vector_score"].as_f64().unwrap() - 1.0).abs() < 1e-5,
        "{v}"
    );
    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&dir);
}
