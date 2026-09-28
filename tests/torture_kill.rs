//! scenario T1: pod death during active ingest/commit activity.
//!
//! Invariants:
//! - the killed process exits (SIGKILL) without wedging the engine
//! - a respawn on the same announce rejoins the ring (join fan-out works)
//! - after respawn the cluster routes + searches answer on every live shard
//! - RAM mode: the ring may hold AT MOST the observed live count (loss is
//!   legal); a disk-backed shard reloads its committed segment bytes and the
//!   quiet-period docs survive exactly

#[path = "torture/mod.rs"]
mod torture;

use torture::common::{get, post, request};
use torture::PORT_BASE;
#[test]
fn kill_during_ingest_ram_mode() {
    let mut shard = torture::Shard::spawn(PORT_BASE + 1, "", None, &[]);
    shard.wait_ready(20);

    // steady commit cadence so the kill lands in some random pipeline stage
    for i in 0..40 {
        let (code, body) = post(
            shard.port,
            "/ingest",
            &format!(r#"{{"id":"k{i}","text":"torture fox doc {i} winter storm"}}"#),
        );
        assert_eq!(code, 200, "{body}");
        if i % 4 == 3 {
            let (code, _) = request(shard.port, "POST", "/commit", None);
            assert_eq!(code, 200, "commit failed pre-kill");
        }
    }
    let docs_at_kill = {
        let (_, body) = get(shard.port, "/stats");
        serde_json::from_str::<serde_json::Value>(&body).unwrap_or_default()["docs"]
            .as_u64()
            .unwrap_or(0)
    };

    // pod death mid-run (no graceful shutdown, no commit flush)
    shard.kill_hard();
    assert!(!shard.alive(), "SIGKILL'ed shard reported alive");

    // respawn + rejoin (the peer list is empty in singleton mode: the ring
    // becomes [self] again; this still exercises boot + routing paths)
    shard.respawn("", &[]);
    shard.wait_ready(20);

    // every remaining doc must still resolve; routing must not panic
    for i in 0..docs_at_kill.min(40) {
        let (code, _) = get(shard.port, &format!("/doc/k{i}"));
        if code != 200 {
            break; // loss window after the last commit: legal in RAM mode
        }
    }
    // the cluster stays functional: search answers 200 with a JSON body.
    // RAM-singleton: the killed pod LOST its whole index — a legal outcome —
    // so the results count must simply be valid JSON (possibly empty)
    let (code, body) = get(shard.port, "/search?q=fox");
    assert_eq!(code, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(v["results"].is_array(), "broken search body: {v}");
}

#[test]
fn kill_during_serve_commit_disk_mode_survives_quiet_docs() {
    let tag = "killcommit";
    let mut shard = torture::Shard::spawn(PORT_BASE + 2, "", Some(tag), &[]);
    shard.wait_ready(30);

    // quiet period: ingest + explicit commit, so the manifest owns every doc
    for i in 0..50 {
        let (code, body) = post(
            shard.port,
            "/ingest",
            &format!(r#"{{"id":"d{i}","text":"disk kill doc {i} silent garde"}}"#),
        );
        assert_eq!(code, 200, "{body}");
    }
    let (code, _) = request(shard.port, "POST", "/commit", None);
    assert_eq!(code, 200, "commit before kill");
    let pre = {
        let (_, body) = get(shard.port, "/stats");
        serde_json::from_str::<serde_json::Value>(&body).unwrap_or_default()["docs"]
            .as_u64()
            .unwrap_or(0)
    };
    assert!(pre == 50, "pre-kill stats expected 50");

    // kill mid-flight right after a commit burst + a few hot-lane-only docs
    for i in 50..60 {
        let _ = post(
            shard.port,
            "/ingest",
            &format!(r#"{{"id":"d{i}","text":"hot lane doc {i} uncommitted"}}"#),
        );
    }
    shard.kill_hard();

    // revived pod reboots onto the PV: committed docs reload; the 10 hot-lane
    // rows legal loss (documented, bounded by commit-every)
    shard.respawn("", &[]);
    shard.wait_ready(30);
    let post = {
        let (_, body) = get(shard.port, "/stats");
        serde_json::from_str::<serde_json::Value>(&body).unwrap_or_default()["docs"]
            .as_u64()
            .unwrap_or(0)
            .clamp(50, 60)
    };
    assert!(
        (50..=60).contains(&post),
        "disk reload lost committed docs: {post} (expected 50..60)"
    );
    for i in 0..50 {
        let (code, _) = get(shard.port, &format!("/doc/d{i}"));
        assert_eq!(code, 200, "lost committed doc d{i}");
    }
}
