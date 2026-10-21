//! embedding search across a ring of shards: the vector query fans out, the
//! modalities fuse globally, and the embeddings survive a shard move

use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde_json::Value;

mod common;
use common::{get, post};

fn spawn_serve(bind: u16, announce: u16, peers: &str) -> Child {
    Command::new(env!("CARGO_BIN_EXE_zap"))
        .args([
            "serve",
            "--bind",
            &format!("127.0.0.1:{bind}"),
            "--announce",
            &format!("127.0.0.1:{announce}"),
            "--peers",
            peers,
            "--lanes",
            "2",
            // every shard declares the SAME embedding field
            "--vector",
            "emb:3",
        ])
        .stdout(Stdio::null())
        .stderr(std::fs::File::create(format!("/tmp/zvec-{bind}.stderr")).expect("log"))
        .spawn()
        .expect("spawn zap serve")
}

fn wait_ready(ports: &[u16]) {
    for &port in ports {
        let mut ok = false;
        for _ in 0..100 {
            let (code, body) = get(port, "/health");
            if code == 200 && body.contains("ok") {
                ok = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(ok, "shard {port} did not come up");
    }
}

struct Guard(Vec<Child>);
impl std::ops::Drop for Guard {
    fn drop(&mut self) {
        for k in &mut self.0 {
            let _ = k.kill();
            let _ = k.wait();
        }
    }
}

fn spawn_cluster(ports: [u16; 3]) -> Guard {
    let mut kids = Vec::new();
    for &p in ports.iter() {
        let others: Vec<String> = ports
            .iter()
            .filter(|x| **x != p)
            .map(|x| format!("127.0.0.1:{x}"))
            .collect();
        kids.push(spawn_serve(p, p, &others.join(",")));
        std::thread::sleep(Duration::from_millis(150));
    }
    Guard(kids)
}

fn vector_search(port: u16, vector: &str, k: usize) -> Vec<Value> {
    let body = format!(r#"{{"vector":[{vector}],"k":{k}}}"#);
    let (code, b) = post(port, "/search", &body);
    assert_eq!(code, 200, "{b}");
    let v: Value = serde_json::from_str(&b).unwrap();
    v["results"].as_array().cloned().unwrap_or_default()
}

#[test]
fn cluster_vector_search_fuses_globally() {
    let ports = [47461u16, 47462, 47463];
    let _guard = spawn_cluster(ports);
    wait_ready(&ports);

    // one doc per direction; the ring spreads them over the shards
    let docs = r#"{"id":"x","text":"east wind","emb":[1.0,0.0,0.0]}
{"id":"y","text":"north wind","emb":[0.0,1.0,0.0]}
{"id":"z","text":"south wind","emb":[0.0,0.0,1.0]}
{"id":"t","text":"west wind","emb":[-1.0,0.0,0.0]}
"#;
    let (code, body) = post(ports[0], "/ingest", docs);
    assert_eq!(code, 200, "{body}");

    // the nearest neighbour is the same from every node
    for &port in ports.iter() {
        let hits = vector_search(port, "1.0,0.0,0.0", 4);
        assert!(!hits.is_empty(), "port {port}: {hits:?}");
        assert_eq!(
            hits[0]["id"],
            Value::String("x".into()),
            "port {port}: {hits:?}"
        );
        assert!(
            (hits[0]["vector_score"].as_f64().unwrap() - 1.0).abs() < 1e-5,
            "port {port}: {hits:?}"
        );
    }

    // the diagonal direction: "z" is the closest
    let hits = vector_search(ports[1], "0.1,0.0,1.0", 4);
    assert_eq!(hits[0]["id"], Value::String("z".into()), "{hits:?}");

    // hybrid across the ring: the keyword and the embedding both contribute
    let body = r#"{"q":"wind","vector":[1.0,0.0,0.0],"k":4,"mode":"hybrid","alpha":0.5}"#;
    let (code, b) = post(ports[2], "/search", body);
    assert_eq!(code, 200, "{b}");
    let v: Value = serde_json::from_str(&b).unwrap();
    let results = v["results"].as_array().expect("results");
    assert!(!results.is_empty(), "{v}");
    assert_eq!(results[0]["id"], Value::String("x".into()), "{v}");
    assert!(results[0]["keyword_score"].is_number(), "{v}");
    assert!(results[0]["vector_score"].is_number(), "{v}");

    // filter mode over the ring: only the keyword matches survive
    let body = r#"{"q":"east","vector":[0.0,0.0,1.0],"k":4,"mode":"filter"}"#;
    let (code, b) = post(ports[0], "/search", body);
    assert_eq!(code, 200, "{b}");
    let v: Value = serde_json::from_str(&b).unwrap();
    let results = v["results"].as_array().expect("results");
    assert_eq!(results.len(), 1, "{v}");
    assert_eq!(results[0]["id"], Value::String("x".into()), "{v}");
}

#[test]
fn handoff_carries_the_embeddings() {
    // a fresh node joins the ring and pulls its share: the embeddings must
    // travel with the documents
    let ports = [47471u16, 47472, 47473];
    let _guard = spawn_cluster(ports);
    wait_ready(&ports);

    let docs = r#"{"id":"x","text":"east wind","emb":[1.0,0.0,0.0]}
{"id":"y","text":"north wind","emb":[0.0,1.0,0.0]}
"#;
    let (code, body) = post(ports[0], "/ingest", docs);
    assert_eq!(code, 200, "{body}");

    // the handoff stream carries the embedding arrays (the shards that own
    // the documents stream them)
    let mut with_vec = 0usize;
    for &p in ports.iter() {
        let (code, body) = get(p, "/__zap/handoff?limit=50");
        assert_eq!(code, 200, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        let rows = v["docs"].as_array().expect("docs");
        with_vec += rows
            .iter()
            .filter(|r| r.get("emb").and_then(|e| e.as_array()).is_some())
            .count();
    }
    assert!(with_vec > 0, "no embedding in the handoff streams");
}
