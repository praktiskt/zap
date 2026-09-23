//! T10: binary end-to-end: ingestion modes, query loop, meta commands

use assert_cmd::Command;

fn zap() -> Command {
    Command::cargo_bin("zap").unwrap()
}

#[test]
fn stdin_lines_blank_separates_docs_and_queries() {
    let input = "the quick brown fox\nA the dog jumped quick\n\nquick\n:stats\n:quit\n";
    zap()
        .arg("--stdin-lines")
        .write_stdin(input)
        .assert()
        .success()
        .stdout(predicates::str::contains("score="));
}

#[test]
fn stdin_single_doc_mode() {
    // --stdin consumes the entire input as ONE doc; the query loop gets EOF
    zap()
        .arg("--stdin")
        .write_stdin("alpha beta gamma delta alpha\n")
        .assert()
        .success()
        .stderr(predicates::str::contains("indexed 1 docs"));
}

#[test]
fn meta_commands() {
    let input = "alpha brown\n\n:help\n:stats\nunexistingtermzz\n:quit\n";
    zap()
        .arg("--stdin-lines")
        .write_stdin(input)
        .assert()
        .success()
        .stdout(predicates::str::contains("no hits"))
        .stderr(predicates::str::contains("[zap] docs=1"));
}

#[test]
fn stdin_lines_with_unknown_query() {
    // malformed query syntaxes are reported, not fatal
    let input = "alpha beta\n\n(unclosed\n:quit\n";
    zap()
        .arg("--stdin-lines")
        .write_stdin(input)
        .assert()
        .success()
        .stderr(predicates::str::contains("query error"));
}

#[test]
fn serve_subcommand_starts_http_api() {
    // fixed high port for the test; the serve subcommand serves /search etc.
    use std::io::{Read, Write};
    use std::net::TcpStream;
    let port = 47412u16;
    let docs_path = std::env::temp_dir().join(format!("zap-serve-cli-{port}.txt"));
    std::fs::write(
        &docs_path,
        "the quick brown fox jumps\nfoxes in the winter forest\n",
    )
    .expect("corpus file");
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_zap"))
        .args([
            "serve",
            "--bind",
            &format!("127.0.0.1:{port}"),
            &docs_path.to_string_lossy(),
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn zap serve");

    // wait for the server to listen, then hit /search over a raw socket
    let mut bound = false;
    for _ in 0..50 {
        if TcpStream::connect(format!("127.0.0.1:{port}")).is_ok() {
            bound = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert!(bound, "serve did not start listening");
    let mut s = std::net::TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
    s.write_all(b"GET /search?q=fox HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n")
        .unwrap();
    let mut resp_buf = [0u8; 65536];
    let mut resp = Vec::new();
    loop {
        let n = s.read(&mut resp_buf).unwrap();
        resp.extend_from_slice(&resp_buf[..n]);
        if n == 0 {
            break;
        }
    }
    let resp = String::from_utf8_lossy(&resp).into_owned();
    let body = resp
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_owned())
        .unwrap_or_default();
    assert!(body.contains("fox"), "response: {resp}");
    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_file(&docs_path);
}
