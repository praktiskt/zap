//! scale-out (N -> N+1) rebalance: the ring moves the minimum, the old
//! owner drops its stale copies, and a delete does not resurrect.
//!
//! Two bugs are pinned here:
//! - the placement was `slot % N` (owner = ring[slot % N]): every N -> N+1
//!   membership change re-homed ~N/(N+1) of the keyspace (67% at 2->3, 75%
//!   at 3->4) instead of the new member's ~1/(N+1) share. The token ring
//!   (zap::cluster::Ring) fixes the movement.
//! - the handoff survey only COPIED the rows: the source kept its stale
//!   copy, so the cluster held more rows than docs x R and a deleted doc
//!   came back from the stale copy (the search fan-out still returned it).

use std::net::SocketAddr;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

mod common;
use common::{get, post};

use zap::cluster::{Ring, REPLICA_FACTOR};

const N_DOCS: usize = 12;

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

/// the ring view the cluster converges on: [self, peers...] sorted
fn ring_of(ports: &[u16]) -> Ring {
    let members: Vec<SocketAddr> = ports
        .iter()
        .map(|p| format!("127.0.0.1:{p}").parse().unwrap())
        .collect();
    Ring::new(members)
}

/// the ids a shard holds locally (the handoff stream is the local live set)
fn local_ids(port: u16) -> Vec<String> {
    let (_, body) = get(port, "/__zap/handoff?limit=500");
    let v: serde_json::Value = serde_json::from_str(&body).expect("handoff json");
    v["docs"]
        .as_array()
        .expect("docs array")
        .iter()
        .filter_map(|d| d.get("id").and_then(|i| i.as_str()).map(|s| s.to_string()))
        .collect()
}

/// local row count of a shard
fn local_docs(port: u16) -> usize {
    let (_, body) = get(port, "/__zap/shard_stats");
    let v: serde_json::Value = serde_json::from_str(&body).expect("shard stats json");
    v["docs"].as_u64().unwrap_or(0) as usize
}

/// boot a 2-node cluster, then scale it out to 3; returns the ports
fn scale_out_cluster(guard: &mut Guard, ports: [u16; 3]) {
    // node A alone: every doc lands on A
    guard.0.push(spawn_serve(ports[0], "", false));
    wait_ready(&[ports[0]]);
    for i in 0..N_DOCS {
        let (code, body) = post(
            ports[0],
            "/ingest",
            &format!(r#"{{"id":"d{i}","text":"rebalance probe doc {i} zebrafish{i} storm"}}"#),
        );
        assert_eq!(code, 200, "{body}");
    }
    // node B joins: R=2 with two members -> both hold every doc
    guard.0.push(spawn_serve(
        ports[1],
        &format!("127.0.0.1:{}", ports[0]),
        true,
    ));
    wait_ready(&[ports[1]]);
    // node C joins: the ring re-partitions, C pulls its holder share and
    // the members that lost the slot drop their stale copies
    guard.0.push(spawn_serve(
        ports[2],
        &format!("127.0.0.1:{},127.0.0.1:{}", ports[0], ports[1]),
        true,
    ));
    wait_ready(&[ports[2]]);
}

/// wait until every doc sits on exactly its holder set (the stale copies are
/// gone) — the row sum settles at docs x R
fn wait_converged(ports: &[u16]) -> usize {
    let want = N_DOCS * REPLICA_FACTOR;
    for _ in 0..160 {
        let total: usize = ports.iter().map(|&p| local_docs(p)).sum();
        if total == want {
            return total;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    let total: usize = ports.iter().map(|&p| local_docs(p)).sum();
    panic!("cluster did not converge: {total} rows, want {want}");
}

/// a scale-out MOVES the documents: every doc ends up on exactly its holder
/// set and the cluster holds docs x R rows (no stale copies left behind)
#[test]
fn scale_out_moves_and_purges_stale_copies() {
    let ports = [48711u16, 48712, 48713];
    let mut guard = Guard(Vec::new());
    scale_out_cluster(&mut guard, ports);
    let ring = ring_of(&ports);
    wait_converged(&ports);

    // every doc lives on exactly the members its key hashes to
    for i in 0..N_DOCS {
        let id = format!("d{i}");
        let want: Vec<String> = ring
            .holders_of_id(&id)
            .iter()
            .map(|&idx| ring.members()[idx].to_string())
            .collect();
        for &port in &ports {
            let has = local_ids(port).contains(&id);
            assert_eq!(
                has,
                want.contains(&format!("127.0.0.1:{port}")),
                "doc {id} presence on {port} is wrong (holders {want:?})"
            );
        }
    }
}

/// a delete AFTER a scale-out stays deleted: the tombstone reaches the stale
/// copies too, so the search fan-out cannot resurrect the row
#[test]
fn delete_after_scale_out_does_not_resurrect() {
    let ports = [48721u16, 48722, 48723];
    let mut guard = Guard(Vec::new());
    scale_out_cluster(&mut guard, ports);

    // delete through the ring (the owner applies it)
    let (code, body) = common::request(ports[0], "DELETE", "/doc/d3", None);
    assert_eq!(code, 200, "{body}");

    // gone from every shard
    for &port in &ports {
        assert!(
            !local_ids(port).contains(&"d3".to_string()),
            "d3 survived on {port}"
        );
        let (code, _) = get(port, "/doc/d3");
        assert_eq!(code, 404, "d3 still readable on {port}");
    }

    // and gone from the cluster search (the stale copy used to bring it back)
    let (_, body) = get(ports[0], "/search?q=zebrafish3&k=30");
    let v: serde_json::Value = serde_json::from_str(&body).expect("search json");
    let ids: Vec<String> = v["results"]
        .as_array()
        .expect("results array")
        .iter()
        .filter_map(|r| r.get("id").and_then(|i| i.as_str()).map(|s| s.to_string()))
        .collect();
    assert!(!ids.contains(&"d3".to_string()), "d3 resurrected: {ids:?}");
}

/// the scale-out moves the MINIMUM: the joining node takes over its holder
/// share only (~2/(N+1) of the corpus with R=2), it does not re-home the
/// whole keyspace the way the old `slot % N` layout did (~N/(N+1))
#[test]
fn scale_out_movement_is_minimal() {
    let ports = [48731u16, 48732, 48733];
    let mut guard = Guard(Vec::new());
    scale_out_cluster(&mut guard, ports);
    let ring = ring_of(&ports);

    // C holds the docs its key hashes to (owner or replica) and nothing else
    let c = format!("127.0.0.1:{}", ports[2]);
    let want_c: Vec<String> = (0..N_DOCS)
        .map(|i| format!("d{i}"))
        .filter(|id| {
            ring.holders_of_id(id)
                .iter()
                .any(|&idx| ring.members()[idx].to_string() == c)
        })
        .collect();
    // wait for C's survey to land
    for _ in 0..80 {
        if local_docs(ports[2]) == want_c.len() {
            break;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    let got_c = local_ids(ports[2]);
    assert_eq!(
        got_c.len(),
        want_c.len(),
        "C holds {got_c:?}, want {want_c:?}"
    );
    for id in &want_c {
        assert!(got_c.contains(id), "C is missing {id}");
    }
}
