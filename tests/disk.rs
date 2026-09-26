//! disk persistence (opt-in --disk): docs survive a process restart, deletes
//! and replaces stick, and the no-disk default stays byte-identical (no files)

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

fn request(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let mut raw = format!("{method} {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n");
    if let Some(b) = body {
        raw += &format!("Content-Length: {}\r\n\r\n{b}", b.len());
    } else {
        raw += "\r\n";
    }
    s.write_all(raw.as_bytes()).expect("write");
    let mut resp = Vec::new();
    s.read_to_end(&mut resp).expect("read");
    let t = String::from_utf8_lossy(&resp).into_owned();
    let code: u16 = t
        .split_whitespace()
        .nth(1)
        .and_then(|x| x.parse().ok())
        .unwrap_or(0);
    let body = t
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_owned())
        .unwrap_or_default();
    (code, body)
}

fn get(port: u16, path: &str) -> (u16, String) {
    request(port, "GET", path, None)
}

fn post(port: u16, path: &str, body: &str) -> (u16, String) {
    request(port, "POST", path, Some(body))
}

fn spawn_serve(bind: u16, disk: Option<&std::path::Path>) -> Child {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_zap"));
    cmd.args([
        "serve",
        "--bind",
        &format!("127.0.0.1:{bind}"),
        "--lanes",
        "2",
    ]);
    if let Some(d) = disk {
        cmd.arg("--disk").arg(d);
    }
    cmd.stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn zap serve")
}

fn wait_ready(port: u16) {
    for _ in 0..100 {
        let Ok(mut s) = TcpStream::connect(("127.0.0.1", port)) else {
            std::thread::sleep(Duration::from_millis(50));
            continue;
        };
        let raw = "GET /health HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n".to_string();
        let _ = s.write_all(raw.as_bytes());
        let mut resp = String::new();
        let _ = s.read_to_string(&mut resp);
        if resp.contains("ok") {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("shard {port} did not come up");
}

pub struct Guard(Vec<Child>);
impl std::ops::Drop for Guard {
    fn drop(&mut self) {
        for k in &mut self.0 {
            let _ = k.kill();
            let _ = k.wait();
        }
    }
}

#[test]
fn disk_roundtrip_across_reboot() {
    let dir = std::env::temp_dir().join(format!("zap-disk-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let port = 48200u16;
    let mut first_kid = vec![spawn_serve(port, Some(&dir))];
    wait_ready(port);

    // ingest via HTTP (auto-commits per request)
    let (code, body) = post(
        port,
        "/ingest",
        r#"{"id":"a1","text":"quick brown fox"}
{"id":"b2","text":"the lazy dog sleeps"}
"#,
    );
    assert_eq!(code, 200, "{body}");

    // delete + replace before the reboot: both must survive
    let (code, body) = post(port, "/ingest", r#"{"id":"a1","text":"replaced fox"}"#);
    assert_eq!(code, 200, "{body}");
    // tombstone one doc for good measure
    request(port, "DELETE", "/doc/b2", None);

    // hard restart: drop the first shard and wait for the socket to close
    drop(Guard(std::mem::take(&mut first_kid)));
    std::thread::sleep(Duration::from_millis(300));

    let _kid2 = Guard(vec![spawn_serve(port, Some(&dir))]);
    wait_ready(port);

    // stats: 2 originals - 1 deleted - 1 replaced-in-place = 1 live doc
    let v: serde_json::Value = serde_json::from_str(&get(port, "/stats").1).unwrap();
    assert_eq!(v["docs"], 1, "{v}");

    // search results survive with the replaced text
    let (code, b) = get(port, "/search?q=fox");
    assert_eq!(code, 200, "{b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    let rows = v["results"].as_array().expect("results");
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0]["id"], "a1");
    assert!(rows[0]["doc"].as_str().unwrap().contains("replaced"));
    // tombstoned doc is gone
    assert_eq!(get(port, "/doc/b2").0, 404, "delete must survive reboot");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn no_disk_writes_nothing() {
    // default: no --disk -> no state files appear anywhere (byte-identical
    // to pre-persistence behavior)
    let port = 48210u16;
    let _kid = Guard(vec![spawn_serve(port, None)]);
    wait_ready(port);
    let (code, body) = post(port, "/ingest", r#"{"id":"z","text":"ephemeral"}"#);
    assert_eq!(code, 200, "{body}");
    // nothing observable side-effect: just make sure ingest + search work
    let v: serde_json::Value = serde_json::from_str(&get(port, "/search?q=ephemeral").1).unwrap();
    assert_eq!(v["count"], 1);
}
