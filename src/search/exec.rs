//! term expansion + query tree evaluation against one segment

use std::collections::HashMap;

use fst::automaton::Levenshtein;
use roaring::RoaringBitmap;

use crate::analyzer::Analyzer;
use crate::posting::TermInfo;
use crate::query::Node;
use crate::schema::Schema;
use crate::search::phrase::phrase_scores;
use crate::search::term::{bm25_idf, term_scores};
use crate::search::{Bm25Params, DocScores};

pub const MAX_EXPANSIONS: usize = 512;

/// global corpus stats for ranking
pub struct Stats {
    pub total_docs: u32,
    /// field id -> average doc length
    pub avg_len: Vec<f32>,
    /// (field id, term) -> global doc frequency
    pub df: HashMap<(u32, String), u32>,
}

impl Stats {
    pub fn empty() -> Self {
        Self {
            total_docs: 0,
            avg_len: Vec::new(),
            df: HashMap::new(),
        }
    }

    pub fn df_of(&self, fid: u32, term: &str, fallback: u32) -> u32 {
        self.df
            .get(&(fid, term.to_string()))
            .copied()
            .unwrap_or(fallback)
    }

    pub fn avg_of(&self, fid: u32) -> f32 {
        self.avg_len.get(fid as usize).copied().unwrap_or(1.0)
    }
}

pub struct Ctx<'a> {
    pub schema: &'a Schema,
    pub stats: &'a Stats,
    pub params: &'a Bm25Params,
    pub analyzer: &'a Analyzer,
}

fn glob_match(term: &str, pat: &[char]) -> bool {
    let t: Vec<char> = term.chars().collect();
    let n = t.len();
    let m = pat.len();
    let (mut ti, mut pi, mut star, mut back) = (0usize, 0usize, None, 0usize);
    while ti < n {
        if pi < m && (pat[pi] == '?' || pat[pi] == t[ti]) {
            ti += 1;
            pi += 1;
        } else if pi < m && pat[pi] == '*' {
            star = Some(pi);
            pi += 1;
            back = ti;
        } else if let Some(sp) = star {
            back += 1;
            ti = back;
            pi = sp + 1;
        } else {
            return false;
        }
    }
    while pi < m && pat[pi] == '*' {
        pi += 1;
    }
    pi == m
}

pub fn eval(
    node: &Node,
    scope: Option<u32>,
    seg: &crate::segment::SegmentReader,
    ctx: &Ctx,
) -> DocScores {
    match node {
        Node::Term(t) => {
            let mut acc = DocScores::default();
            let stems = stem_variants(ctx, t);
            for fid in scope_ids(scope, ctx) {
                for st in &stems {
                    if let Some(info) = seg.term(fid, st) {
                        let idf = bm25_idf(ctx.stats.df_of(fid, st, info.df), ctx.stats.total_docs);
                        let ts = term_scores(
                            &info,
                            seg.blob(),
                            idf,
                            ctx.params,
                            ctx.stats.avg_of(fid),
                            1.0,
                            seg.norms(fid),
                        );
                        acc = crate::search::union(&acc, &ts);
                    }
                }
            }
            acc
        }
        Node::Fuzzy { term, distance } => {
            let dist = (*distance as u32).min(2);
            let mut acc = DocScores::default();
            for fid in scope_ids(scope, ctx) {
                // fresh automaton ownership per field (moved into the fst walk)
                let lev = match Levenshtein::new(term, dist) {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                let pairs: Vec<(String, TermInfo)> = seg.stream_automaton(lev, fid, MAX_EXPANSIONS);
                let ls = leaf_score(seg, ctx, fid, &pairs, dist);
                acc = crate::search::union(&acc, &ls);
            }
            acc
        }
        Node::Prefix(p) | Node::Wildcard(p) => {
            let prefix = matches!(node, Node::Prefix(_));
            let pattern: Vec<char> = p.chars().collect();
            let mut acc = DocScores::default();
            for fid in scope_ids(scope, ctx) {
                let pairs = if prefix {
                    seg.stream_prefix(fid, p, MAX_EXPANSIONS)
                } else {
                    // wildcards in the middle of the term: keep the glob sweep (rare)
                    let pat = pattern.clone();
                    into_pairs(seg.stream_terms(fid), move |t| glob_match(t, &pat))
                };
                let ls = leaf_score(seg, ctx, fid, &pairs, 0);
                acc = crate::search::union(&acc, &ls);
            }
            acc
        }
        Node::Phrase { terms, slop } => {
            if terms.len() == 1 {
                return eval(&Node::Term(terms[0].clone()), scope, seg, ctx);
            }
            let mut acc = DocScores::default();
            for fid in scope_ids(scope, ctx) {
                let infos: Vec<Option<TermInfo>> = terms.iter().map(|t| seg.term(fid, t)).collect();
                if infos.iter().any(|i| i.is_none()) {
                    continue;
                }
                let infos: Vec<TermInfo> = infos.into_iter().flatten().collect();
                let idf_sum: f32 = infos
                    .iter()
                    .zip(terms)
                    .map(|(info, t)| {
                        bm25_idf(ctx.stats.df_of(fid, t, info.df), ctx.stats.total_docs)
                    })
                    .sum();
                let ps = phrase_scores(&infos, seg.blob(), idf_sum, *slop, 1.5);
                acc = crate::search::union(&acc, &ps);
            }
            acc
        }
        Node::Field { field, node } => {
            match field.as_deref().and_then(|f| ctx.schema.field_id(f)) {
                Some(fid) => eval(node, Some(fid), seg, ctx),
                None => DocScores::default(),
            }
        }
        Node::Not(child) => {
            let ds = eval(child, scope, seg, ctx);
            let docs: RoaringBitmap = ds.ids.iter().collect();
            DocScores::all((0..seg.num_docs).filter(|d| !docs.contains(*d)).collect())
        }
        Node::Or(items) => {
            let mut acc = DocScores::default();
            for it in items {
                let ds = eval(it, scope, seg, ctx);
                acc = crate::search::union(&acc, &ds);
            }
            acc
        }
        Node::And(items) => {
            let mut positives: Vec<&Node> = Vec::new();
            let mut excluded: Vec<RoaringBitmap> = Vec::new();
            for it in items {
                match it {
                    Node::Not(child) => {
                        let ds = eval(child, scope, seg, ctx);
                        excluded.push(ds.ids.iter().collect());
                    }
                    other => positives.push(other),
                }
            }
            match positives.pop() {
                Some(first) => {
                    let mut acc = eval(first, scope, seg, ctx);
                    for p in positives.iter() {
                        let ds = eval(p, scope, seg, ctx);
                        acc = crate::search::intersect(&acc, &ds);
                    }
                    if excluded.is_empty() {
                        acc
                    } else {
                        filter_out(&acc, &excluded)
                    }
                }
                None => DocScores::default(),
            }
        }
    }
}

fn stem_variants(ctx: &Ctx, t: &str) -> Vec<String> {
    let mut v = vec![t.to_string()];
    let stemmed = ctx.analyzer.stem_word(t);
    if !v.contains(&stemmed) {
        v.insert(0, stemmed);
    }
    v
}

fn scope_ids(scope: Option<u32>, ctx: &Ctx) -> Vec<u32> {
    match scope {
        Some(f) => vec![f],
        None => (0..ctx.schema.num_fields() as u32).collect(),
    }
}

fn leaf_score(
    seg: &crate::segment::SegmentReader,
    ctx: &Ctx,
    fid: u32,
    pairs: &[(String, TermInfo)],
    dist: u32,
) -> DocScores {
    let mut acc = DocScores::default();
    for (term, info) in pairs {
        let idf = bm25_idf(ctx.stats.df_of(fid, term, info.df), ctx.stats.total_docs);
        let decay = if dist > 0 {
            1.0 / (1.0 + dist as f32)
        } else {
            1.0
        };
        let ts = term_scores(
            info,
            seg.blob(),
            idf,
            ctx.params,
            ctx.stats.avg_of(fid),
            decay,
            seg.norms(fid),
        );
        acc = crate::search::union(&acc, &ts);
    }
    acc
}

/// remove excluded docs from a score series
fn filter_out(ds: &DocScores, excl: &[RoaringBitmap]) -> DocScores {
    let mut out = DocScores::default();
    for (i, d) in ds.ids.iter().enumerate() {
        if !excl.iter().any(|p| p.contains(*d)) {
            out.ids.push(*d);
            out.scores.push(ds.scores[i]);
        }
    }
    out
}

fn into_pairs(
    terms: Vec<(String, TermInfo)>,
    pred: impl Fn(&str) -> bool,
) -> Vec<(String, TermInfo)> {
    terms
        .into_iter()
        .filter(|(t, _)| pred(t))
        .take(MAX_EXPANSIONS)
        .collect()
}
