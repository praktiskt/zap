//! zap CLI: load corpus into memory, search interactively

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{bail, Context, Result};
use clap::Parser;
use zap::analyzer::{Analyzer, AnalyzerConf};
use zap::index::Index;
use zap::schema::{Document, Schema};

#[derive(Parser, Debug)]
#[command(name = "zap", about = "in-memory full-text search engine")]
struct Args {
    /// files or directories to index into the `text` field
    inputs: Vec<PathBuf>,
    /// read one document from stdin instead of files
    #[arg(long)]
    stdin: bool,
    /// each stdin line becomes one document
    #[arg(long)]
    stdin_lines: bool,
    /// number of results per query
    #[arg(short, long, default_value = "10")]
    top_k: usize,
    /// BM25 k1
    #[arg(long, default_value = "1.2")]
    k1: f32,
    /// BM25 b
    #[arg(long, default_value = "0.75")]
    b: f32,
    /// disable stemming
    #[arg(long)]
    no_stem: bool,
    /// plain output (no ANSI)
    #[arg(long)]
    plain: bool,
    /// snippet width in chars
    #[arg(long, default_value = "240")]
    snippet: usize,
}

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

const KNOWN_TEXT_EXTS: [&str; 8] = ["txt", "md", "rst", "json", "toml", "csv", "log", "xml"];

fn read_doc(p: &Path) -> Result<String> {
    let known = p
        .extension()
        .and_then(|x| x.to_str())
        .map(|x| KNOWN_TEXT_EXTS.contains(&x))
        .unwrap_or(false);
    if known {
        return std::fs::read_to_string(p).context(p.display().to_string());
    }
    match std::fs::read_to_string(p) {
        Ok(s) => Ok(s),
        Err(_) => bail!("skip {}: not text", p.display()),
    }
}

fn collect_docs(args: &Args) -> Result<Vec<String>> {
    let mut docs = Vec::new();
    if args.stdin {
        let mut s = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut s).context("read stdin")?;
        if !s.trim().is_empty() {
            docs.push(s);
        }
        return Ok(docs);
    }
    if args.stdin_lines {
        for line in std::io::stdin().lock().lines() {
            let l = line?;
            if !l.trim().is_empty() {
                docs.push(l);
            }
        }
        return Ok(docs);
    }
    for p in &args.inputs {
        let md = std::fs::metadata(p).with_context(|| p.display().to_string())?;
        if md.is_file() {
            docs.push(read_doc(p)?);
        } else if md.is_dir() {
            let mut stack = vec![p.clone()];
            while let Some(d) = stack.pop() {
                for e in std::fs::read_dir(&d).with_context(|| d.display().to_string())?.flatten() {
                    let ep = e.path();
                    if ep.is_dir() {
                        stack.push(ep);
                    } else if ep.is_file() {
                        docs.push(read_doc(&ep)?);
                    }
                    if SHUTDOWN.load(Ordering::SeqCst) {
                        break;
                    }
                }
            }
        } else {
            bail!("skip {}: not a file or dir", p.display());
        }
    }
    Ok(docs)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ctrlc::set_handler(|| SHUTDOWN.store(true, Ordering::SeqCst)).ok();
    let docs = collect_docs(&args)?;
    if docs.is_empty() {
        eprintln!("no documents; pass files/dirs or --stdin / --stdin-lines");
        std::process::exit(2);
    }
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    let analyzer = Analyzer::new(AnalyzerConf {
        stem: !args.no_stem,
        max_token_len: 64,
    });
    let index = Index::new(schema, analyzer.clone(), args.k1, args.b);
    for text in &docs {
        index.add_document(&Document::new().text("text", text.clone()))?;
    }
    index.commit();
    eprintln!(
        "[zap] indexed {} docs, segments {}",
        index.num_docs(),
        index.num_segments()
    );
    for line in std::io::stdin().lock().lines() {
        if SHUTDOWN.load(Ordering::SeqCst) {
            break;
        }
        let line = line?;
        let line = line.trim();
        match line {
            "" => continue,
            ":q" | ":quit" => break,
            ":help" => eprintln!(
                "queries: terms | a AND b | a OR c | NOT x | \"phrase\" | field:term | prefix* | wild*card | typo~1 | :stats | :help | :quit"
            ),
            ":stats" => eprintln!(
                "[zap] docs={} segments={} gen={}",
                index.num_docs(),
                index.num_segments(),
                index.generation()
            ),
            q => {
                let hits = match index.search(q, args.top_k) {
                    Ok(h) => h,
                    Err(e) => {
                        eprintln!("query error: {e}");
                        continue;
                    }
                };
                let terms: Vec<String> = analyzer.tokenize_query(q).iter().map(|t| t.text.clone()).collect();
                let stems: Vec<String> = analyzer.tokenize(q).iter().map(|t| t.text.clone()).collect();
                let highlight: Vec<String> = terms
                    .iter()
                    .chain(stems.iter())
                    .cloned()
                    .collect::<HashSet<String>>()
                    .into_iter()
                    .collect();
                if hits.is_empty() {
                    println!("no hits");
                }
                for hit in &hits {
                    if let Some(fields) = index.stored_fields(hit.addr) {
                        for (name, text) in &fields {
                            let snippet = snippet_of(text, &highlight, args.snippet, args.plain);
                            println!("score={:.4} {} | {}", hit.score, name, snippet);
                        }
                    }
                }
                let _ = std::io::stdout().flush();
            }
        }
    }
    Ok(())
}

use std::collections::HashSet;

/// snippet centered on the first matched token cluster, terms highlighted
fn snippet_of(text: &str, terms: &[String], width: usize, plain: bool) -> String {
    if plain {
        return trim_char(text, 0, width);
    }
    let analyzer = Analyzer::new(AnalyzerConf::default());
    let toks = analyzer.tokenize(text);
    let tset: HashSet<&String> = terms.iter().collect();
    let highlight: Vec<(usize, usize)> = toks
        .iter()
        .filter(|t| tset.contains(&t.text))
        .map(|t| (t.byte_start, t.byte_end))
        .collect();
    // window start: beginning of text or a bit before the first match
    let char_start = highlight
        .first()
        .map(|(s, _)| s.saturating_sub(3 * width))
        .unwrap_or(0);
    let start = floor_char(text, char_start);
    let end = floor_char(text, (start + 6 * width).min(text.len()));
    let win = &text[start..end];
    let base = start;
    let mut out = String::new();
    let mut last = 0usize;
    for (s, e) in highlight {
        let (s, e) = (s.saturating_sub(base), e.saturating_sub(base));
        if s < last || s > win.len() {
            continue;
        }
        out.push_str(&win[last..s]);
        out.push_str("\u{1b}[1m");
        out.push_str(&win[s..e.min(win.len())]);
        out.push_str("\u{1b}[0m");
        last = e.min(win.len());
    }
    out.push_str(&win[last..]);
    if start > 0 {
        out = format!("…{out}");
    }
    if end < text.len() {
        out.push('…');
    }
    out
}

fn trim_char(text: &str, start: usize, width: usize) -> String {
    let s = floor_char(text, start);
    let e = floor_char(text, (s + width).min(text.len()));
    text[s..e].to_string()
}

/// snap a byte index back to a char boundary (or 0)
fn floor_char(text: &str, i: usize) -> usize {
    if text.is_char_boundary(i) || i == 0 {
        i
    } else {
        let mut i = i;
        while i > 0 && !text.is_char_boundary(i) {
            i -= 1;
        }
        i
    }
}
