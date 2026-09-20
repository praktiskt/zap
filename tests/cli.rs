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
