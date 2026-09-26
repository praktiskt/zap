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

pub fn request(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    send(&mut s, method, path, body)
}

pub fn get(port: u16, path: &str) -> (u16, String) {
    request(port, "GET", path, None)
}

pub fn post(port: u16, path: &str, body: &str) -> (u16, String) {
    request(port, "POST", path, Some(body))
}

fn send(s: &mut TcpStream, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut raw = format!("{method} {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n");
    if let Some(b) = body {
        raw += &format!("Content-Length: {}\r\n\r\n{b}", b.len());
    } else {
        raw += "\r\n";
    }
    s.write_all(raw.as_bytes()).expect("write request");
    let mut resp = Vec::new();
    s.read_to_end(&mut resp).expect("read response");
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
