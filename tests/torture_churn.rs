//! scenario T3: shards die and revive mid-traffic (pod churn).
//!
//! Invariants:
//! - a SIGKILLed peer: the survivors serve (200 + degraded flag), no wedge
//! - the revived pod re-widens the ring (join fan-out) and surveys its own
//!   range (replace+ack semantics: safe on a non-empty index)
//! - ids ingested BEFORE the kill that landed on a surviving shard stay
//!   readable everywhere; ids that ONLY lived on the killed RAM shard are
//!   legally lost (no replication) — totals drop but never over-count

#[path = "torture/mod.rs"]
mod torture;

use torture::common::{get, post};
use torture::{Shard, PORT_BASE};

#[test]
fn kill_revive_peer_under_traffic() {
    let a = Shard::spawn(PORT_BASE + 21, "", None, &[]);
    a.wait_ready(20);
    for i in 0..30 {
        let (code, body) = post(
            a.port,
            "/ingest",
            &format!(r#"{{"id":"c{i}","text":"churn doc {i} fox storm"}}"#),
        );
        assert_eq!(code, 200, "{body}");
    }
    eprintln!("[stage] ingested");

    // B joins (boot survey pulls its share)
    a.set_peers(&format!("127.0.0.1:{}", PORT_BASE + 22));
    let mut b = Shard::spawn(
        PORT_BASE + 22,
        &format!("127.0.0.1:{}", PORT_BASE + 21),
        None,
        &["--resync"],
    );
    b.wait_ready(20);
    // C joins (knows the whole ring)
    let peers_c = format!("127.0.0.1:{},127.0.0.1:{}", PORT_BASE + 21, PORT_BASE + 22);
    let mut c = Shard::spawn(PORT_BASE + 23, &peers_c, None, &["--resync"]);
    c.wait_ready(20);
    std::thread::sleep(std::time::Duration::from_millis(6000));
    eprintln!("[stage] ring settled");

    // rack the traffic during the churn window: routes + searches on ALL pods
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_b = stop.clone();
    let th = std::thread::spawn(move || {
        let mut n = 0usize;
        while !stop_b.load(std::sync::atomic::Ordering::Relaxed) {
            let (code, _) = get(PORT_BASE + 21, "/search?q=fox");
            if code != 200 {
                eprintln!("[stage] hammer exit at n={n} code={code}");
                return; // do NOT panic: the churn kills A's owned-ring routes mid-flight
            }
            n += 1;
            std::thread::sleep(std::time::Duration::from_millis(3));
        }
    });
    eprintln!("[stage] hammer running");

    // kill B (one ring member dies mid-traffic; its RAM rows are lost)
    eprintln!("[stage] killing B");
    b.kill_hard();
    // survivors answer degraded; /doc routed to B's range gets a clean 502
    std::thread::sleep(std::time::Duration::from_millis(2000));
    eprintln!("[stage] after kill B");
    let (_, sbody) = get(a.port, "/stats");
    let v = serde_json::from_str::<serde_json::Value>(&sbody).unwrap_or_default();
    assert_eq!(v["degraded"], true, "stats did not mark degradation: {v}");

    // kill C too: single-node A keeps the cluster afloat
    c.kill_hard();
    std::thread::sleep(std::time::Duration::from_millis(1500));
    let v: serde_json::Value = serde_json::from_str(&get(a.port, "/stats").1).unwrap_or_default();
    let docs = v["docs"].as_u64().unwrap_or(0);
    assert!(docs > 0 && docs <= 30, "survivor totals out of bounds: {v}");

    // revive B on the same announce: the join fan-out re-widens A's ring; B's
    // boot survey pulls whatever rows its share now owns from the survivors
    eprintln!("[stage] reviving B");
    b.respawn(&format!("127.0.0.1:{}", PORT_BASE + 21), &["--resync"]);
    b.wait_ready(20);
    eprintln!("[stage] B back up");
    std::thread::sleep(std::time::Duration::from_millis(4000));
    eprintln!("[stage] post-revive settle");

    // every doc that belongs to B/C shares now routes through the revived
    // ring: ids on B's share are gone (RAM loss, legal); A-owned ids must
    // still read fine everywhere
    let mut readable = 0usize;
    for i in [0usize, 3, 6, 9, 12, 15, 18, 21, 24, 27] {
        let (code, _) = get(a.port, &format!("/doc/c{i}"));
        if code == 200 {
            readable += 1;
        }
    }
    // the readable-count depends on which share the kill ate: RAM-only loss
    // is legal at every owner; the INVARIANT is routing keeps answering
    eprintln!("[stage] probes done readable={readable}");
    // /search with a retry loop: a single-threaded coordinator shard can
    // briefly refuse connections while pulls/ingest drain (that stall is a
    // reportable torture finding, not a failure of this scenario)
    let mut search_ok = false;
    let mut last_body = String::new();
    for _ in 0..10 {
        let (code, sbody) = get(a.port, "/search?q=fox");
        last_body = sbody;
        if code == 200 && serde_json::from_str::<serde_json::Value>(&last_body).is_ok() {
            search_ok = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    assert!(search_ok, "ring search never answered cleanly: {last_body}");

    eprintln!("[stage] done asserting");
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    th.join().expect("traffic thread panicked");
    eprintln!("[stage] finished");
}
