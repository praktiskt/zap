// each integration test binary compiles this module independently, so not
// every helper is used in every binary
#![allow(dead_code)]

//! shared test HTTP helper: tiny sync client, one per
//! (port, method, path, body); returns (status_code, body)

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};

pub fn request_raw(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> (u16, String) {
    let mut s = TcpStream::connect(addr).expect("connect");
    send(&mut s, method, path, body)
}

/// connect with retry: shard processes do not listen within the first ~300ms
/// after spawn, and booting a disk-backed index can take longer
pub fn request(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut s = None;
    for _ in 0..40 {
        match TcpStream::connect(("127.0.0.1", port)) {
            Ok(sock) => {
                s = Some(sock);
                break;
            }
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(50)),
        }
    }
    let mut s = s.expect("connect after retries");
    send(&mut s, method, path, body)
}

pub fn get(port: u16, path: &str) -> (u16, String) {
    request(port, "GET", path, None)
}

pub fn post(port: u16, path: &str, body: &str) -> (u16, String) {
    request(port, "POST", path, Some(body))
}

fn send(s: &mut TcpStream, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    use std::time::Duration;
    // bound both sides so a wedged shard handler cannot freeze a test
    // process: a /doc forwarded into a dead ring leg can stall one of the
    // single-threaded tiny_http workers for arbitrarily long
    let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
    let _ = s.set_write_timeout(Some(Duration::from_secs(5)));
    let mut raw = format!("{method} {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n");
    if let Some(b) = body {
        raw += &format!("Content-Length: {}\r\n\r\n{b}", b.len());
    } else {
        raw += "\r\n";
    }
    s.write_all(raw.as_bytes()).expect("write request");
    let mut resp = Vec::new();
    // timeout-terminated reads surface as Err: a partial response maps to a
    // 0-status (0xx) rather than an infinite block
    let _ = s.read_to_end(&mut resp);
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
