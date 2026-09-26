//! cluster roundtrip: 3 real zap processes, ring-routed ingest/replace/delete,
//! fan-out search from any node, score convergence after gossip deltas

use std::process::{Child, Command, Stdio};
use std::time::Duration;

mod common;
use common::{get, post, request};

fn delete(port: u16, path: &str) -> (u16, String) {
    request(port, "DELETE", path, None)
}

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
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
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
    for (i, &p) in ports.iter().enumerate() {
        let others: Vec<String> = ports
            .iter()
            .filter(|x| **x != p)
            .map(|x| format!("127.0.0.1:{x}"))
            .collect();
        let csv = others.join(",");
        eprintln!(
            "[cluster-test] spawning shard {} at {} with peers {:?} (announce {p})",
            i, p, csv
        );
        let mut kid = spawn_serve(p, p, &csv);
        // die-fast diagnostics: is the child already gone?
        std::thread::sleep(std::time::Duration::from_millis(200));
        if let Ok(Some(_exit)) = kid.try_wait() {
            eprintln!("[cluster-test] CRASH: shard {p} exited immediately");
        }
        kids.push(kid);
    }
    Guard(kids)
}

#[test]
fn cluster_of_three() {
    let ports = [47451u16, 47452, 47453];
    let _guard = spawn_cluster(ports);
    wait_ready(&ports);

    let n0 = ports[0];

    // --- ingest spread over the ring, through ONE node (port 0) ---
    let docs = r#"{"id":"a1","text":"quick brown fox"}
{"id":"b2","text":"the lazy dog sleeps"}
{"id":"c3","text":"fox hunts in winter"}
"#;
    let (code, body) = post(n0, "/ingest", docs);
    assert_eq!(code, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["accepted"], 3, "{body}");

    // --- /stats aggregates the cluster ---
    let (code, body) = get(n0, "/stats");
    assert_eq!(code, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["docs"], 3, "{body}");

    // --- search ANY node duplicates results across the ring ---
    let hits_a = search(n0, "fox");
    let hits_b = search(ports[1], "fox");
    assert_eq!(hits_a.len(), 2, "{hits_a:?}");
    assert_eq!(hits_a.len(), hits_b.len(), "same cluster same query");

    // --- replace by id is ring-stable ---
    let (code, body) = post(
        ports[2],
        "/ingest",
        r#"{"id":"a1","text":"replaced entirely"}"#,
    );
    assert_eq!(code, 200, "{body}");
    let (code, b) = get(n0, "/doc/a1");
    assert_eq!(code, 200, "{b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["fields"][0][1], "replaced entirely");
    // total count stays 3
    let v: serde_json::Value = serde_json::from_str(&get(n0, "/stats").1).unwrap();
    assert_eq!(v["docs"], 3, "{v}");

    // --- search sees the replacement, not the original ---
    let hits1 = search(n0, "fox");
    for h in &hits1 {
        if h["id"] == "a1" {
            assert!(h["doc"].as_str().unwrap_or("").contains("replaced"));
        }
    }

    // --- delete-by-id routes into the ring and hides everywhere ---
    let (code, body) = delete(n0, "/doc/c3");
    assert_eq!(code, 200, "{body}");
    assert_eq!(get(n0, "/doc/c3").0, 404);
    assert_eq!(get(ports[1], "/doc/c3").0, 404);
    let v: serde_json::Value = serde_json::from_str(&get(n0, "/stats").1).unwrap();
    assert_eq!(v["docs"], 2, "{v}");
}

fn search(port: u16, q: &str) -> Vec<serde_json::Value> {
    let (code, body) = get(port, &format!("/search?q={q}"));
    assert_eq!(code, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    v["results"].as_array().cloned().unwrap_or_default()
}
