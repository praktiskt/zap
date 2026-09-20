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

/// [stemmed?, raw] variants (1 alloc max); used by stem_variants()
/// (stemmed, raw) iteration alternates; at most 1 alloc; replaces stem_variants()
fn stem_pair(ctx: &Ctx, t: &str) -> [Option<String>; 2] {
    let stem = ctx.analyzer.stem_word(t);
    if stem == t {
        [Some(stem), None]
    } else {
        [Some(stem), Some(t.to_string())]
    }
}

/// convenience: eval without bitmap probe
pub fn eval(
    node: &Node,
    scope: Option<u32>,
    seg: &crate::segment::SegmentReader,
    ctx: &Ctx,
) -> DocScores {
    eval_allowed(node, scope, seg, ctx, None)
}

pub fn eval_allowed(
    node: &Node,
    scope: Option<u32>,
    seg: &crate::segment::SegmentReader,
    ctx: &Ctx,
    allowed: Option<&roaring::RoaringBitmap>,
) -> DocScores {
    // `allowed`: bitmap probe pre-applied at term level (bitmap-first query execution)
    use crate::search::term::term_scores_allowed;
    match node {
        Node::Term(t) => {
            let mut acc = DocScores::default();
            for fid in scope_ids(scope, ctx) {
                for st in stem_pair(ctx, t).into_iter().flatten() {
                    if let Some(info) = seg.term(fid, &st) {
                        let idf = bm25_idf(
                            ctx.stats.df_of(fid, st.as_str(), info.df),
                            ctx.stats.total_docs,
                        );
                        let ts = term_scores_allowed(
                            &info,
                            seg.blob(),
                            idf,
                            ctx.params,
                            ctx.stats.avg_of(fid),
                            1.0,
                            seg.norms(fid),
                            allowed,
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
            let mut acc = DocScores::default();
            for fid in scope_ids(scope, ctx) {
                // leading literal segment bounds the FST range; remaining glob chars filter
                let head_end = if prefix {
                    p.len()
                } else {
                    p.find('*').unwrap_or(p.len())
                };
                let head = &p[..head_end];
                let pattern: Vec<char> = p.chars().collect();
                let pat = pattern.clone();
                let mut pairs = seg.stream_prefix(fid, head, MAX_EXPANSIONS);
                if !prefix {
                    pairs.retain(|(t, _)| glob_match(t, &pat));
                }
                let ls = leaf_score(seg, ctx, fid, &pairs, 0);
                acc = crate::search::union(&acc, &ls);
            }
            acc
        }
        Node::Phrase { terms, slop } => {
            if terms.len() == 1 {
                return eval_allowed(&Node::Term(terms[0].clone()), scope, seg, ctx, allowed);
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
                Some(fid) => eval_allowed(node, Some(fid), seg, ctx, allowed),
                None => DocScores::default(),
            }
        }
        Node::Not(child) => {
            let ds = eval_allowed(child, scope, seg, ctx, allowed);
            let docs: RoaringBitmap = ds.ids.iter().collect();
            DocScores::all((0..seg.num_docs).filter(|d| !docs.contains(*d)).collect())
        }
        Node::Or(items) => {
            let mut acc = DocScores::default();
            for it in items {
                let ds = eval_allowed(it, scope, seg, ctx, allowed);
                acc = crate::search::union(&acc, &ds);
            }
            acc
        }
        Node::And(items) => {
            // R12/R15: streaming term-AND: two smallest-df positives drive
            // galloping walks; only matches get scored; longer chains intersect
            let mut positives: Vec<&Node> = Vec::new();
            let mut excluded: Vec<RoaringBitmap> = Vec::new();
            for it in items {
                match it {
                    Node::Not(child) => {
                        let ds = eval_allowed(child, scope, seg, ctx, allowed);
                        excluded.push(ds.ids.iter().collect());
                    }
                    other => positives.push(other),
                }
            }
            let all_terms =
                positives.iter().all(|p| matches!(p, Node::Term(_))) && excluded.is_empty();
            if all_terms {
                let term_infos: Vec<Vec<(u32, String, TermInfo)>> = positives
                    .iter()
                    .map(|p| resolve_term_positives(p, scope, seg, ctx))
                    .collect();
                let dfs: Vec<u32> = term_infos
                    .iter()
                    .map(|infos| infos.iter().map(|i| i.2.df).sum())
                    .collect();
                let mut order: Vec<usize> = (0..positives.len()).collect();
                order.sort_by_key(|i| dfs[*i]);
                let pair = [order[0], order[1]];
                if let Some(mut acc) = and_two_infos(
                    &term_infos[pair[0]],
                    &term_infos[pair[1]],
                    seg,
                    ctx,
                    allowed,
                ) {
                    for &oi in &order[2..] {
                        let ds = term_infos_doc_scores(&term_infos[oi], seg, ctx, None);
                        acc = crate::search::intersect(&acc, &ds);
                    }
                    return acc;
                }
            }
            // legacy: stream intersect of per-positive scored arrays
            let mut iter = positives.iter();
            let Some(first) = iter.next() else {
                return DocScores::default();
            };
            let mut acc = eval_allowed(first, scope, seg, ctx, allowed);
            for p in iter {
                let ds = eval_allowed(p, scope, seg, ctx, allowed);
                acc = crate::search::intersect(&acc, &ds);
            }
            if excluded.is_empty() {
                acc
            } else {
                let mask: RoaringBitmap = excluded.iter().flat_map(|m| m.iter()).collect();
                let full: RoaringBitmap = (0..seg.num_docs).collect();
                filter_by(&acc, &(&full - &mask))
            }
        }
    }
}

/// resolve a Term node to all matching (fid, stem, info) entries
fn resolve_term_positives(
    p: &Node,
    scope: Option<u32>,
    seg: &crate::segment::SegmentReader,
    ctx: &Ctx,
) -> Vec<(u32, String, TermInfo)> {
    let Node::Term(t) = p else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for fid in scope_ids(scope, ctx) {
        for st in stem_pair(ctx, t).into_iter().flatten() {
            if let Some(info) = seg.term(fid, &st) {
                out.push((fid, st, info));
            }
        }
    }
    out
}

/// bm25 score series for a resolved term set (plain walk)
fn term_infos_doc_scores(
    infos: &[(u32, String, TermInfo)],
    seg: &crate::segment::SegmentReader,
    ctx: &Ctx,
    allowed: Option<&roaring::RoaringBitmap>,
) -> DocScores {
    let mut acc = DocScores::default();
    for (fid, st, info) in infos {
        let idf = bm25_idf(ctx.stats.df_of(*fid, st, info.df), ctx.stats.total_docs);
        let sc = crate::search::term::term_scores_allowed(
            info,
            seg.blob(),
            idf,
            ctx.params,
            ctx.stats.avg_of(*fid),
            1.0,
            seg.norms(*fid),
            allowed,
        );
        acc = crate::search::union(&acc, &sc);
    }
    acc
}

/// galloping driver-walk over two resolved single-variant postings; smallest df drives
fn and_two_infos(
    a_infos: &[(u32, String, TermInfo)],
    b_infos: &[(u32, String, TermInfo)],
    seg: &crate::segment::SegmentReader,
    ctx: &Ctx,
    allowed: Option<&roaring::RoaringBitmap>,
) -> Option<DocScores> {
    if a_infos.len() != 1 || b_infos.len() != 1 {
        return None;
    }
    let (a, b) = (&a_infos[0], &b_infos[0]);
    let idf_a = bm25_idf(ctx.stats.df_of(a.0, &a.1, a.2.df), ctx.stats.total_docs);
    let idf_b = bm25_idf(ctx.stats.df_of(b.0, &b.1, b.2.df), ctx.stats.total_docs);
    let drive = if a.2.df <= b.2.df { (a, b) } else { (b, a) };
    let (drv, oth) = drive;
    let mut r_drv = seg.reader(&drv.2);
    let mut r_oth = seg.reader(&oth.2);
    let norms_drv = seg.norms(drv.0);
    let norms_oth = seg.norms(oth.0);
    let mut out = DocScores::default();
    while let Some((doc, tf_d)) = r_drv.advance() {
        if let Some(al) = allowed {
            if !al.contains(doc) {
                continue;
            }
        }
        r_oth.seek_to(doc);
        let Some((d2, tf_o)) = r_oth.peek() else {
            continue;
        };
        if d2 != doc {
            continue;
        }
        let w_d = idf_a * (tf_d as f32 * (ctx.params.k1 + 1.0))
            / (tf_d as f32
                + ctx.params.k1
                    * crate::search::term::norm_ratio(
                        norms_drv,
                        doc,
                        ctx.stats.avg_of(drv.0),
                        ctx.params.b,
                    ));
        let w_o = idf_b * (tf_o as f32 * (ctx.params.k1 + 1.0))
            / (tf_o as f32
                + ctx.params.k1
                    * crate::search::term::norm_ratio(
                        norms_oth,
                        doc,
                        ctx.stats.avg_of(oth.0),
                        ctx.params.b,
                    ));
        out.ids.push(doc);
        out.scores.push(w_d + w_o);
    }
    Some(out)
}

/// score only docs inside the bitmap
fn filter_by(ds: &DocScores, probe: &RoaringBitmap) -> DocScores {
    let mut out = DocScores::default();
    for (i, d) in ds.ids.iter().enumerate() {
        if probe.contains(*d) {
            out.ids.push(*d);
            out.scores.push(ds.scores[i]);
        }
    }
    out
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
