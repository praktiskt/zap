// each scenario binary compiles this module independently: not every helper
// is used in every scenario
#![allow(dead_code)]

//! torture harness: multi-shard zap process scaffolding for destructive
//! scenarios (kills, corrupt state, churn, repartition under load).
//!
//! Included by the `tests/torture_*.rs` scenario files. A scenario is a TEST;
//! the harness provides:
//!
//! - shards with per-shard stderr logs under /tmp/zap-torture (postmortem)
//! - `kill_hard` = SIGKILL (the "pod dies" event), `respawn` = same announce
//! - disk-dir helpers: delete/flip on-disk segment files for corrupt state
//! - polled API calls with generous timeouts (children listen late)
//!
//! Scenarios assert INVARIANTS (no panics, routing stays consistent, live
//! totals bounded), not exact data: RAM-only loss + hot-lane loss are legal
//! crash outcomes, never PASS/FAIL material.

#[path = "../common/mod.rs"]
pub mod common;
// scenario files pull what they use; not every helper is used in every binary
#[allow(unused_imports)]
pub use common::{get, post, request};

use std::io::Write;
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// bind/announce base: sits clear of the other test suites (47xxx, 48xxx)
pub const PORT_BASE: u16 = 51001;

pub struct Shard {
    pub port: u16,
    pub disk: Option<std::path::PathBuf>,
    child: Option<Child>,
}

impl Drop for Shard {
    fn drop(&mut self) {
        self.kill_hard();
    }
}

fn addr_of(port: u16) -> std::net::SocketAddr {
    format!("127.0.0.1:{port}").parse().expect("addr")
}

fn spawn_cmd(port: u16, peers: &str, disk: Option<&std::path::Path>, extra: &[&str]) -> Child {
    let _ = std::fs::create_dir_all("/tmp/zap-torture");
    let mut args = vec![
        "serve".to_string(),
        "--bind".to_string(),
        format!("127.0.0.1:{port}"),
        "--announce".to_string(),
        format!("127.0.0.1:{port}"),
        "--peers".to_string(),
        peers.to_string(),
        "--lanes".to_string(),
        "2".to_string(),
    ];
    if let Some(d) = disk {
        args.push("--disk".to_string());
        args.push(d.display().to_string());
    }
    for a in extra {
        args.push((*a).to_string());
    }
    let log = format!("/tmp/zap-torture/shard-{port}.log");
    Command::new(env!("CARGO_BIN_EXE_zap"))
        .args(args)
        .stdout(Stdio::null())
        .stderr(std::fs::File::create(&log).expect("shard log"))
        .spawn()
        .expect("spawn zap serve")
}

impl Shard {
    /// spawn; `peers` = ring CSV (empty = singleton);
    /// `disk_tag` = Some(tag) => fresh --disk dir inside /tmp/zap-torture
    pub fn spawn(port: u16, peers: &str, disk_tag: Option<&str>, extra: &[&str]) -> Shard {
        let dir = disk_tag.map(|tag| {
            let root = std::path::PathBuf::from(format!("/tmp/zap-torture/{tag}.{port}"));
            if root.exists() {
                std::fs::remove_dir_all(&root).expect("wipe disk dir");
            }
            std::fs::create_dir_all(&root).expect("disk dir");
            root
        });
        let child = spawn_cmd(port, peers, dir.as_deref(), extra);
        Shard {
            port,
            disk: dir,
            child: Some(child),
        }
    }

    /// re-spawn on the SAME announce/ports; a disk-backed shard boots onto
    /// its existing --disk directory contents (no wipe: recovery semantics)
    pub fn respawn(&mut self, peers: &str, extra: &[&str]) {
        self.kill_hard();
        let child = spawn_cmd(self.port, peers, self.disk.as_deref(), extra);
        self.child = Some(child);
    }

    pub fn wait_ready(&self, secs: u64) {
        let deadline = Instant::now() + Duration::from_secs(secs);
        loop {
            if let Ok(mut s) =
                TcpStream::connect_timeout(&addr_of(self.port), Duration::from_millis(500))
            {
                let _ =
                    s.write_all(b"GET /health HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n");
                let mut resp = String::new();
                use std::io::Read;
                if s.read_to_string(&mut resp).is_ok() && resp.starts_with("HTTP/1.1 200") {
                    return;
                }
            }
            assert!(
                Instant::now() < deadline,
                "shard {} not ready in {secs}s",
                self.port
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// pod death: SIGKILL the child process owned by the test
    pub fn kill_hard(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
            self.child = None;
        }
    }

    pub fn alive(&mut self) -> bool {
        match &mut self.child {
            Some(child) => child.try_wait().ok().flatten().is_none(),
            None => false,
        }
    }

    /// point this shard's peer list at a CSV in place (ring repartition)
    pub fn set_peers(&self, peers_csv: &str) {
        let (code, body) = request(
            self.port,
            "POST",
            "/__zap/peers",
            Some(&format!(r#"{{"peers":"{peers_csv}"}}"#)),
        );
        assert!(code == 200, "set_peers {body}");
    }

    /// stop the shard and corrupt on-disk state: `flip` XORs the first byte
    /// of every .zap segment file, false = DELETE one such file
    pub fn corrupt_disk(&mut self, flip: bool) -> usize {
        self.kill_hard();
        let n = {
            let d = self.disk.as_ref().expect("no disk for corrupt_disk");
            let segdir = d.join("segments");
            let mut n = 0usize;
            for entry in std::fs::read_dir(segdir).expect("no segments dir") {
                let Ok(e) = entry else { continue };
                let path = e.path();
                if path.extension().and_then(|x| x.to_str()) != Some("zap") {
                    continue;
                }
                if flip {
                    let mut bytes = std::fs::read(&path).expect("segment read");
                    if let Some(b) = bytes.first_mut() {
                        *b ^= 0xFF;
                    }
                    std::fs::write(&path, &bytes).expect("corrupt write");
                } else {
                    std::fs::remove_file(&path).expect("remove");
                }
                n += 1;
            }
            n
        };
        assert!(n > 0, "corrupted nothing at {:?}", self.disk);
        n
    }
}

/// fixed-size ring spawn: every shard peers with every other (full mesh
/// CSV); callers can post-spawn add members with joins/repartitions
pub fn spawn_ring(base: u16, n: usize, disk_tag: Option<&str>) -> Vec<Shard> {
    let ports: Vec<String> = (0..n)
        .map(|i| format!("127.0.0.1:{}", base + i as u16))
        .collect();
    (0..n)
        .map(|i| {
            let peers = ports
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, p)| p.clone())
                .collect::<Vec<_>>()
                .join(",");
            Shard::spawn(
                base + i as u16,
                &peers,
                disk_tag,
                if disk_tag.is_some() {
                    &["--resync"]
                } else {
                    &[]
                },
            )
        })
        .collect()
}
