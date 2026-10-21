//! zap CLI: load corpus into memory, search interactively

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use zap::analyzer::{Analyzer, AnalyzerConf};
use zap::index::Index;
use zap::schema::{Document, Metric, Schema};
use zap::search::{Fusion, SearchMode, SearchRequest};

#[derive(Parser, Debug)]
#[command(name = "zap", about = "in-memory full-text search engine")]
struct Args {
    /// files or directories to index into the `text` field
    inputs: Vec<PathBuf>,
    /// read one document from stdin instead of files
    #[arg(long, global = true)]
    stdin: bool,
    /// each stdin line becomes one document
    #[arg(long, global = true)]
    stdin_lines: bool,
    /// opt-in persistence: load existing state from DIR at boot; every
    /// commit atomically persists segments + manifest there (reboot-safe)
    #[arg(long, global = true, value_name = "DIR")]
    disk: Option<PathBuf>,
    /// number of results per query
    #[arg(short, long, default_value = "10", global = true)]
    top_k: usize,
    /// BM25 k1 (0 disables length normalization variance; default 1.2)
    #[arg(long, global = true, default_value = "1.2")]
    k1: f32,
    /// BM25 b (0 = no length normalization; default 0.75)
    #[arg(long, global = true, default_value = "0.75")]
    b: f32,
    /// disable stemming
    #[arg(long, global = true)]
    no_stem: bool,
    /// plain output (no ANSI)
    #[arg(long, global = true)]
    plain: bool,
    /// snippet width in chars
    #[arg(long, global = true, default_value = "240")]
    snippet: usize,
    /// parallel ingest lanes (default: cores on this machine)
    #[arg(long, global = true, default_value = "16")]
    lanes: usize,
    /// declare a vector (embedding) field: name:dim[:metric] (repeatable).
    /// The metric defaults to cosine; the caller supplies the embeddings.
    #[arg(long = "vector", global = true, value_name = "NAME:DIM[:METRIC]")]
    vectors: Vec<String>,
    /// auto-commit every N documents (bounds segment memory)
    #[arg(long, global = true, default_value = "50000")]
    commit_every: usize,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// serve the HTTP API (POST /ingest, GET /search, POST /commit)
    Serve {
        /// host:port to bind
        #[arg(long, default_value = "127.0.0.1:3279")]
        bind: String,
        /// cluster peers as CSV name:port (empty = single-node, zero overhead)
        #[arg(long, global = true, default_value = "")]
        peers: String,
        /// this node's ring address (defaults to the bind address)
        #[arg(long, global = true, default_value = "")]
        announce: String,
        /// files or directories to index before serving (also HTTP /ingest)
        inputs: Vec<PathBuf>,
        /// boot empty and pull this node's ring share from the peers once
        /// serving (shard movement / empty-node recovery)
        #[arg(long)]
        resync: bool,
    },
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

fn collect_docs(args: &Args, extra_inputs: Vec<PathBuf>) -> Result<Vec<String>> {
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
        // first blank line separates the doc block from the query block
        let stdin = std::io::stdin();
        let mut done = false;
        for line in stdin.lock().lines() {
            let l = line?;
            done |= l.trim().is_empty();
            if done {
                break;
            }
            docs.push(l);
        }
        return Ok(docs);
    }
    let mut inputs = args.inputs.clone();
    inputs.extend(extra_inputs);
    for p in &inputs {
        let md = std::fs::metadata(p).with_context(|| p.display().to_string())?;
        if md.is_file() {
            docs.push(read_doc(p)?);
        } else if md.is_dir() {
            let mut stack = vec![p.clone()];
            while let Some(d) = stack.pop() {
                for e in std::fs::read_dir(&d)
                    .with_context(|| d.display().to_string())?
                    .flatten()
                {
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
    let extra = match &args.command {
        Some(Command::Serve { inputs, .. }) => inputs.clone(),
        None => Vec::new(),
    };
    let docs = collect_docs(&args, extra)?;
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    for spec in &args.vectors {
        let mut parts = spec.split(':');
        let name = parts.next().unwrap_or("").trim().to_string();
        let dim: u32 = parts
            .next()
            .and_then(|s| s.trim().parse().ok())
            .ok_or_else(|| anyhow::anyhow!("bad --vector `{spec}` (want name:dim[:metric])"))?;
        if dim == 0 {
            bail!("--vector `{spec}`: dim must be > 0");
        }
        let metric = match parts.next().map(|s| s.trim()) {
            None | Some("") => Metric::Cosine,
            Some(m) => Metric::parse(m)
                .ok_or_else(|| anyhow::anyhow!("unknown metric `{m}` in --vector `{spec}`"))?,
        };
        schema.add_vector(&name, dim, metric);
        eprintln!(
            "[zap] vector field `{name}` dim={dim} metric={}",
            metric.name()
        );
    }
    let analyzer = Analyzer::new(AnalyzerConf {
        stem: !args.no_stem,
        max_token_len: 64,
    });
    let lanes = args.lanes.min(docs.len().max(1)).max(1);
    let index = Index::with_lanes_disk(
        schema,
        analyzer.clone(),
        args.k1,
        args.b,
        lanes,
        args.disk.clone(),
    );
    let mut chunk: Vec<Document> = Vec::with_capacity(args.commit_every);
    for text in &docs {
        chunk.push(Document::new().text("text", text.clone()));
        if chunk.len() >= args.commit_every {
            index.add_documents(std::mem::take(&mut chunk))?;
            index.commit();
        }
    }
    if !chunk.is_empty() {
        index.add_documents(chunk)?;
    }
    index.commit();
    eprintln!(
        "[zap] indexed {} docs, segments {}",
        index.num_docs(),
        index.num_segments()
    );
    if let Some(Command::Serve {
        bind,
        peers,
        announce,
        resync,
        ..
    }) = &args.command
    {
        let idx = index.clone();
        let cluster =
            zap::cluster::build_peer_config(bind, peers, announce).map_err(anyhow::Error::msg)?;
        match zap::server::serve(idx, bind, cluster, &SHUTDOWN, *resync) {
            Ok(addr) => eprintln!("[zap serving] http://{addr}"),
            Err(e) => {
                eprintln!("[zap serve] {e}");
                std::process::exit(2);
            }
        }
        // block until shutdown
        while !SHUTDOWN.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        return Ok(());
    }
    // sticky query state for the embedding side of the search
    let mut sticky_vector: Option<Vec<f32>> = None;
    let mut sticky_mode = SearchMode::Keyword;
    let mut sticky_alpha = 0.5f32;
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
            // ---- embedding controls ----
            cmd if cmd.starts_with(":vec") => {
                let arg = cmd[4..].trim();
                if arg.is_empty() || arg == "off" || arg == "clear" {
                    sticky_vector = None;
                    eprintln!("[zap] query embedding cleared");
                } else {
                    let mut v = Vec::new();
                    for part in arg.split(',') {
                        let t = part.trim();
                        if t.is_empty() {
                            continue;
                        }
                        match t.parse::<f32>() {
                            Ok(x) => v.push(x),
                            Err(_) => {
                                eprintln!("[zap] bad embedding element `{t}`");
                                break;
                            }
                        }
                    }
                    if !v.is_empty() {
                        sticky_vector = Some(v);
                        eprintln!("[zap] query embedding set ({} dims)", sticky_vector.as_ref().map(|v| v.len()).unwrap_or(0));
                    }
                }
            }
            cmd if cmd.starts_with(":mode") => {
                let arg = cmd[5..].trim();
                sticky_mode = match arg {
                    "keyword" => SearchMode::Keyword,
                    "vector" => SearchMode::Vector,
                    "hybrid" => SearchMode::Hybrid,
                    "filter" => SearchMode::Filter,
                    _ => {
                        eprintln!("[zap] mode: keyword | vector | hybrid | filter");
                        sticky_mode
                    }
                };
                eprintln!("[zap] search mode: {:?}", sticky_mode);
            }
            cmd if cmd.starts_with(":alpha") => {
                let arg = cmd[5..].trim();
                match arg.parse::<f32>() {
                    Ok(a) => {
                        sticky_alpha = a.clamp(0.0, 1.0);
                        eprintln!("[zap] alpha (keyword weight) = {}", sticky_alpha);
                    }
                    Err(_) => eprintln!("[zap] alpha: 0.0 .. 1.0"),
                }
            }
            q => {
                // the embedding side is optional: with a sticky vector the
                // query becomes vector / hybrid / filter (per :mode)
                let use_vector = sticky_vector.is_some() && sticky_mode != SearchMode::Keyword;
                let hits = match use_vector {
                    false => match index.search(q, args.top_k) {
                        Ok(h) => h,
                        Err(e) => {
                            eprintln!("query error: {e}");
                            continue;
                        }
                    },
                    true => {
                        let mut req = SearchRequest {
                            text: Some(q.to_string()),
                            vector: sticky_vector.clone(),
                            field: None,
                            k: args.top_k,
                            mode: sticky_mode,
                            alpha: sticky_alpha,
                            fusion: Fusion::Weighted,
                        };
                        if sticky_mode == SearchMode::Vector {
                            req.text = None;
                        }
                        match index.search_request(&req, index.pub_stats()) {
                            Ok(h) => h,
                            Err(e) => {
                                eprintln!("query error: {e}");
                                continue;
                            }
                        }
                    }
                };
                let terms: Vec<String> = analyzer.tokenize_query(q).iter().map(|t| t.text.to_string()).collect();
                let stems: Vec<String> = analyzer.tokenize(q).iter().map(|t| t.text.to_string()).collect();
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
                    if let Some(p) = index.doc_payload(hit.addr) {
                        if let Some(id) = &p.id {
                            println!("score={:.4} id={} |", hit.score, id);
                        }
                        for (name, text) in &p.fields {
                            let snippet = snippet_of(&analyzer, text, &highlight, args.snippet, args.plain);
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
fn snippet_of(
    analyzer: &Analyzer,
    text: &str,
    terms: &[String],
    width: usize,
    plain: bool,
) -> String {
    if plain {
        return trim_char(text, width);
    }
    let toks = analyzer.tokenize(text);
    let tset: HashSet<&str> = terms.iter().map(String::as_str).collect();
    let highlight: Vec<(usize, usize)> = toks
        .iter()
        .filter(|t| tset.contains(t.text.as_ref()))
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

fn trim_char(text: &str, width: usize) -> String {
    let e = floor_char(text, width.min(text.len()));
    text[..e].to_string()
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
