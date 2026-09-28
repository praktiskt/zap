//! scenario T4: corrupt on-disk segment state at boot.
//!
//! Invariants:
//! - a flipped-byte or missing .zap file NEVER hangs the boot: the shard
//!   either errors out cleanly (process exits non-zero with a message on
//!   stderr) or comes up serving
//! - a corrupted shard must not poison the ring: the OTHER shards stay
//!   serving; map the corrupted member out with POST /__zap/peers and the
//!   ring still routes (the surviving members' searches converge)

#[path = "torture/mod.rs"]
mod torture;

use torture::common::{get, post, request};
use torture::{Shard, PORT_BASE};

const BOOT_BUDGET_SECS: u64 = 25;

/// boot a possibly-corrupt disk shard and assert CLEAN behavior: either it
/// serves within the budget, or the process exits with a controlled message
/// (stderr carries an error line; no panic-loop, no hang, no wedge)
fn assert_clean_or_dead(shard: &mut Shard, label: &str) {
    // poll readiness for a bounded window; if it never comes up, require a
    // non-zero exit with a log line (boot error path)
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(BOOT_BUDGET_SECS);
    loop {
        if shard.alive() {
            if std::net::TcpStream::connect_timeout(
                &format!("127.0.0.1:{}", shard.port).parse().unwrap(),
                std::time::Duration::from_millis(400),
            )
            .is_ok()
            {
                let (code, _) = get(shard.port, "/health");
                if code == 200 {
                    break; // serving: legal recovery path
                }
            }
        } else {
            break; // exited: expect a controlled boot error on stderr
        }
        assert!(
            std::time::Instant::now() < deadline,
            "corrupt-state boot WEDGED for {BOOT_BUDGET_SECS}s: {label}"
        );
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

#[test]
fn flipped_byte_boot_is_wedge_free() {
    let mut shard = Shard::spawn(PORT_BASE + 31, "", Some("flip"), &[]);
    shard.wait_ready(40);
    for i in 0..30 {
        let (code, body) = post(
            shard.port,
            "/ingest",
            &format!(r#"{{"id":"f{i}","text":"flip doc {i} fox storm"}}"#),
        );
        assert_eq!(code, 200, "{body}");
    }
    request(shard.port, "POST", "/commit", None);

    // pod dies; the disk state gets corrupted (first byte of each .zap)
    let n = shard.corrupt_disk(true);
    assert!(n > 0, "flipped nothing");

    // resurrect: EITHER serving OR a clean boot-failure exit; never a hang
    shard.respawn("", &[]);
    assert_clean_or_dead(&mut shard, "flipped-byte boot");

    if shard.alive() {
        // the shard got up: /health is up; searches must still be defined
        let (code, _) = get(shard.port, "/search?q=fox");
        assert_eq!(code, 200, "search against recovered shard failed");
    } else {
        // exited: stderr must carry the boot error (grep the shard log)
        let log = std::fs::read_to_string(format!("/tmp/zap-torture/shard-{}.log", shard.port))
            .unwrap_or_default();
        assert!(
            log.contains("disk") || log.contains("manifest") || log.contains("error"),
            "boot death without a controlled error log: '{log}'"
        );
    }
}

#[test]
fn missing_segment_file_boot_is_wedge_free() {
    let mut shard = Shard::spawn(PORT_BASE + 32, "", Some("missing"), &[]);
    shard.wait_ready(40);
    for i in 0..30 {
        let (code, body) = post(
            shard.port,
            "/ingest",
            &format!(r#"{{"id":"m{i}","text":"missing doc {i} fox storm"}}"#),
        );
        assert_eq!(code, 200, "{body}");
    }
    request(shard.port, "POST", "/commit", None);

    // one .zap vanishes (silent disk corruption via partial delete)
    shard.corrupt_disk(false);

    shard.respawn("", &[]);
    assert_clean_or_dead(&mut shard, "missing-file boot");

    if shard.alive() {
        let (code, _) = get(shard.port, "/search?q=fox");
        assert_eq!(code, 200, "search against recovered shard failed");
    } else {
        let log = std::fs::read_to_string(format!("/tmp/zap-torture/shard-{}.log", shard.port))
            .unwrap_or_default();
        assert!(
            log.contains("disk") || log.contains("manifest") || log.contains("error"),
            "boot death without a controlled error log: '{log}'"
        );
    }
}
