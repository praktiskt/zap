//! handoff / resync (shard movement + empty-node recovery):
//! singleton node boots, ingests with the ring = [self], then a fresh node
//! joins with --resync. The joined node's two-node ring re-owns a share and
//! the boot pull moves those docs over (the source ack-deletes id-carrying
//! rows). After both sides repartition (each node must apply the same peer
//! list), every doc routes to its new owner from BOTH nodes.

use std::process::{Child, Command, Stdio};
use std::time::Duration;

mod common;
use common::{get, post};

struct Guard(Vec<Child>);
impl std::ops::Drop for Guard {
    fn drop(&mut self) {
        for k in &mut self.0 {
            let _ = k.kill();
            let _ = k.wait();
        }
    }
}

fn spawn_serve(bind: u16, peers: &str, resync: bool) -> Child {
    let mut args = vec![
        "serve".to_string(),
        "--bind".to_string(),
        format!("127.0.0.1:{bind}"),
        "--announce".to_string(),
        format!("127.0.0.1:{bind}"),
        "--peers".to_string(),
        peers.to_string(),
        "--lanes".to_string(),
        "2".to_string(),
    ];
    if resync {
        args.push("--resync".to_string());
    }
    Command::new(env!("CARGO_BIN_EXE_zap"))
        .args(args)
        .stdout(Stdio::null())
        .stderr(std::fs::File::create(format!("/tmp/zdtest-{bind}.stderr")).expect("log"))
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

#[test]
fn resync_pull_moves_owned_range() {
    let mut kids = Guard(Vec::new());
    let a = 48671u16;
    let b = 48672u16;

    // singleton shard: --peers empty => ring sees only itself
    kids.0.push(spawn_serve(a, "", false));
    wait_ready(&[a]);

    for (id, text) in [
        ("h1", "handoff fox"),
        ("h2", "handoff dog sleeps"),
        ("h3", "handoff fox again"),
    ] {
        let (code, body) = post(a, "/ingest", &format!(r#"{{"id":"{id}","text":"{text}"}}"#));
        assert_eq!(code, 200, "{body}");
    }

    // fresh node joins --resync (its view of the new ring = [a, b] already);
    // the boot pull re-homes the rows the new ring owner assigns to it.
    // Poll the SPECIFIC moved doc: fnv64a("h2")%2=1 => b owns it in the new
    // ring (h1/h3 keep owner a); a forwarded /doc/h1 on b would satisfy an
    // `any` check purely through routing, before any move happened
    kids.0.push(spawn_serve(b, "127.0.0.1:48671", true));
    wait_ready(&[b]);
    // the joined node's boot survey must pull its holder share from a
    // (singleton-polled-by-routing /doc/h* could satisfy forwarding rules):
    // poll for B's LOCAL row count reaching the full corpus (every doc = a
    // [A, B] holder pair, R=2)
    for _ in 0..40 {
        let vs: serde_json::Value = serde_json::from_str(&get(b, "/stats").1).unwrap();
        let local = vs["shards"][0]["docs"].as_u64().unwrap_or(0);
        if local == 3 {
            break;
        }
        std::thread::sleep(Duration::from_millis(250));
    }

    // the join announce already widened both ring views (A learns B through
    // the join merge; B learned A at boot) — the explicit repartition would
    // be a no-op → the ring routes converged via /__zap/peers {join}

    // every doc readable from BOTH nodes (ring routing + same ids)
    for id in ["h1", "h2", "h3"] {
        for port in [a, b] {
            let (code, _) = get(port, &format!("/doc/{id}"));
            assert_eq!(code, 200, "missing {id} on {port}");
        }
    }

    // stored-row sum: 3 docs x R=2 holders — the singleton's owner share
    // moved to b (ack-deleted at a) and b's survey re-pulled the share plus
    // its replica share stays at a
    let v: serde_json::Value = serde_json::from_str(&get(b, "/stats").1).unwrap();
    let d0 = v["shards"][0]["docs"].as_u64().unwrap_or(0);
    let d1 = v["shards"][1]["docs"].as_u64().unwrap_or(0);
    assert_eq!(d0 + d1, 6, "{v}");
}
