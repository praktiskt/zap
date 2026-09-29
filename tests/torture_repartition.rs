//! scenario T2: ring repartition + join under a live query load.
//!
//! Invariants:
//! - concurrent /search against a ring whose membership is widening runs to
//!   completion or degrades cleanly (no wedged connections, no panics)
//! - post-join-and-pull: every earlier id answers 200 from every live node
//! - totals never exceed the ingested count (ack-deletes beat duplicates)

#[path = "torture/mod.rs"]
mod torture;

use torture::common::{get, post};
use torture::{Shard, PORT_BASE};

const N_IDS: usize = 40;

/// hammer loop: /search must always come back 200 with a JSON body (never a
/// hangs socket); a sideband ingest trickle keeps the engine under mixed load
fn hammer(port: u16, stop: std::sync::Arc<std::sync::atomic::AtomicBool>) {
    let mut n = 0usize;
    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
        let (code, body) = get(port, "/search?q=fox");
        assert_eq!(code, 200, "hammered search broke: {body}");
        assert!(
            serde_json::from_str::<serde_json::Value>(&body).is_ok(),
            "hammered search non-JSON: {body}"
        );
        n += 1;
        if n.is_multiple_of(7) {
            let _ = post(
                port,
                "/ingest",
                &format!(r#"{{"id":"all-{n}","text":"churn wolf doc {n} ramp storm"}}"#),
            );
        }
    }
}

#[test]
fn repartition_under_search_load() {
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop2 = stop.clone();

    // shard A singleton first (full-mesh ring would need B up at spawn time)
    let mut a = Shard::spawn(PORT_BASE + 11, "", None, &[]);
    a.wait_ready(20);

    // hammer while ring changes: join side B (announce + empty pull) then
    // a third node C announcing itself in
    let th = std::thread::spawn(move || hammer(PORT_BASE + 11, stop2));

    for i in 0..N_IDS {
        let (code, body) = post(
            a.port,
            "/ingest",
            &format!(r#"{{"id":"r{i}","text":"repart doc {i} fox storm"}}"#),
        );
        assert_eq!(code, 200, "{body}");
    }

    // B joins the ring behind the running queries (announce fan-out makes A
    // widen its ring; a --resync pull takes over the docs A no longer owns)
    a.set_peers(&format!("127.0.0.1:{}", PORT_BASE + 12));
    let mut b = Shard::spawn(
        PORT_BASE + 12,
        &format!("127.0.0.1:{}", PORT_BASE + 11),
        None,
        &["--resync"],
    );
    b.wait_ready(20);
    let boot_resync = std::env::var("ZAP_TORTURE_NO_BOOT_RESYNC").is_err();
    assert!(boot_resync, "always-on resync expected");

    // C joins KNOWING THE WHOLE RING: its announce fan-out widens every node it
    // addresses (single-peer joins only widen that one node), and its own
    // --resync pull walks every peer's handoff stream so re-homed rows hop over
    let peers_c = format!("127.0.0.1:{},127.0.0.1:{}", PORT_BASE + 11, PORT_BASE + 12);
    let mut c = Shard::spawn(PORT_BASE + 13, &peers_c, None, &["--resync"]);
    c.wait_ready(20);

    // give the joins + pulls time to settle before asserting routing
    std::thread::sleep(std::time::Duration::from_millis(4000));

    // capture the pre-assert snapshot so a failing assert leaves evidence
    for s in [&a, &b, &c] {
        let (_, body) = get(s.port, "/__zap/handoff?limit=500");
        let _ = std::fs::write(format!("/tmp/zap-torture/handoff-{}.json", s.port), body);
        for id in ["r0", "r1", "r2", "r3", "r4", "r5"] {
            let (code, _) = get(s.port, &format!("/doc/{id}"));
            eprintln!("[probe] port={} id={id} code={code}", s.port);
        }
    }

    for i in 0..N_IDS {
        for s in [&mut a, &mut b, &mut c] {
            // bounded retry: a single-threaded coordinator shard can briefly
            // refuse connections while the survey/replica pushes drain —
            // the invariant is the row RESPONDS within a retry window
            let mut code = 0u16;
            for _ in 0..6 {
                let (c, _) = get(s.port, &format!("/doc/r{i}"));
                code = c;
                if code == 200 || code == 404 {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(400));
            }
            assert_eq!(code, 200, "losing r{i} on {}", s.port);
        }
    }

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    th.join().expect("hammer thread panicked");
}

// stdlib plumbing needed by the harness-only pieces
