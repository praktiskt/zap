//! SIMD bit-packing A/B probe (postprobe: decode formats)
//!
//! Takes the engine's REAL postings for a dense term at 1M-doc scale and
//! times the same work three ways:
//!   (a) engine varint [delta][tf] walk + fold-score   (the current floor)
//!   (b) engine docid_vec only (docid decode floor)
//!   (c) bitpacking::BitPacker4x blocks (tantivy's simdcomp port, real SIMD
//!       kernels), per-block uniform widths for the two planes, forward
//!       decompress + score
//!   (d) per-block decompress cost only (models block-max WAND skipping)
//!
//!   cargo run --release --example probe_bp

use std::hint::black_box;
use std::time::Instant;

use bitpacking::{BitPacker, BitPacker4x};
use zap::analyzer::{Analyzer, AnalyzerConf};
use zap::index::Index;
use zap::schema::{Document, Schema};

const BLOCK: usize = 128;

fn main() {
    unsafe { run() }
}

unsafe fn run() {
    let mut schema = Schema::default();
    schema.add_text("text", 1.0);
    let index = Index::with_lanes(schema, Analyzer::new(AnalyzerConf::default()), 1.2, 0.75, 8);
    let words = ["fox", "quick", "brown", "jumps", "lazy", "dog", "forest"];
    let n_docs = 1_000_000usize;
    let mut chunk: Vec<Document> = Vec::with_capacity(50_000);
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    for _i in 0..n_docs {
        let mut s = String::with_capacity(50 * 7);
        for _w in 0..50 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            s.push_str(words[(seed as usize) % words.len()]);
            s.push(' ');
        }
        chunk.push(Document::new().text("text", s));
        if chunk.len() == 50_000 {
            index.add_documents(std::mem::take(&mut chunk)).unwrap();
            index.commit();
        }
    }
    if !chunk.is_empty() {
        index.add_documents(chunk).unwrap();
        index.commit();
    }
    let snap = index.segments_snapshot();
    let big = snap.iter().max_by_key(|s| s.num_docs).unwrap().clone();
    let info = big
        .stream_terms(0)
        .into_iter()
        .find(|(t, _)| *t == "fox")
        .unwrap()
        .1;
    let blob = big.blob();

    // extract engine postings: deltas + tfs (plain walk)
    let df = info.df as usize;
    let mut deltas: Vec<u32> = Vec::with_capacity(df);
    let mut tfs: Vec<u32> = Vec::with_capacity(df);
    {
        let mut r = zap::posting::PostingReader::new(&info, blob);
        let mut prev = 0u32;
        while let Some((d, t)) = r.advance() {
            deltas.push(d - prev);
            tfs.push(t);
            prev = d;
        }
    }
    let usable = df - df % BLOCK;
    println!(
        "docs={} seg_docs={} df={df}",
        index.num_docs(),
        big.num_docs
    );

    // pack both planes (per-128 uniform width; widths ride like anchors)
    let packer = BitPacker4x::new();
    let mut pk_delta: Vec<u8> = Vec::with_capacity(df * 6 / 8);
    let mut pk_tf: Vec<u8> = Vec::new();
    let mut w_delta: Vec<u8> = Vec::with_capacity(df / BLOCK + 1);
    let mut w_tf: Vec<u8> = Vec::with_capacity(df / BLOCK + 1);
    let mut scratch: Vec<u32> = vec![0u32; BLOCK];
    for blk in deltas.chunks_exact(BLOCK) {
        scratch.copy_from_slice(blk);
        let bits = packer.num_bits(&scratch);
        w_delta.push(bits);
        let mut out = vec![0u8; BLOCK * bits as usize / 8];
        let n = packer.compress(&scratch, &mut out, bits);
        out.truncate(n);
        pk_delta.extend_from_slice(&out);
    }
    for blk in tfs.chunks_exact(BLOCK) {
        scratch.copy_from_slice(blk);
        let bits = packer.num_bits(&scratch);
        w_tf.push(bits);
        let mut out = vec![0u8; BLOCK * bits as usize / 8];
        let n = packer.compress(&scratch, &mut out, bits);
        out.truncate(n);
        pk_tf.extend_from_slice(&out);
    }
    println!(
        "bytes: engine(delta+tf)={} simd(delta+tf)={} (w max d/t = {}/{}); blocks={}",
        df * 8,
        pk_delta.len() + pk_tf.len(),
        w_delta.iter().max().unwrap_or(&0),
        w_tf.iter().max().unwrap_or(&0),
        usable / BLOCK
    );

    let params = zap::search::Bm25Params { k1: 1.2, b: 0.75 };
    let norms: Vec<u16> = (0..df).map(|i| (10 + i % 30) as u16).collect();
    let fold = zap::search::term::Bm25Fold::new(0.5, &params, 50.0);
    let nblocks = usable / BLOCK;
    let packed_total: usize = w_delta.iter().map(|w| BLOCK * *w as usize / 8).sum();
    println!(
        "debug: nblocks={} pk_len={} sum={}",
        nblocks,
        pk_delta.len(),
        packed_total
    );
    let rounds = 20usize;

    // (a) engine varint walk + score
    {
        let t0 = Instant::now();
        let mut sum = 0f32;
        for _ in 0..rounds {
            let mut r = zap::posting::PostingReader::new(&info, blob);
            let mut di = 0usize;
            while let Some((d, tf)) = r.advance() {
                if di >= usable {
                    break;
                }
                let len = norms[di];
                di += 1;
                sum += fold.score(tf as f32, len);
                black_box(d);
            }
        }
        let per = t0.elapsed().as_nanos() as f64 / (rounds * usable) as f64;
        println!("(a) varint advance+score : {per:.2} ns/doc (sum={sum:e})");
    }

    // (b) engine docid_vec (decode floor reference)
    {
        let t0 = Instant::now();
        for _ in 0..rounds {
            let v = zap::posting::PostingReader::new(&info, blob).docid_vec();
            black_box(&v);
        }
        let per = t0.elapsed().as_nanos() as f64 / (rounds * usable) as f64;
        println!("(b) varint docid_vec     : {per:.2} ns/doc");
    }

    // (c) SIMD bitpacked forward decompress + score
    {
        let t0 = Instant::now();
        let mut sum = 0f32;
        let mut docbuf: [u32; BLOCK] = [0; BLOCK];
        let mut tfbuf: [u32; BLOCK] = [0; BLOCK];
        let mut next_doc = 0u32;
        for _ in 0..rounds {
            let mut di = 0usize;
            next_doc = 0;
            let mut dbase = 0usize;
            let mut tbase = 0usize;
            for b in 0..nblocks {
                let wd = w_delta[b] as usize;
                packer.decompress(
                    &pk_delta[dbase..dbase + BLOCK * wd / 8],
                    &mut docbuf,
                    wd as u8,
                );
                dbase += BLOCK * wd / 8;
                let wt = w_tf[b] as usize;
                packer.decompress(&pk_tf[tbase..tbase + BLOCK * wt / 8], &mut tfbuf, wt as u8);
                tbase += BLOCK * wt / 8;
                for l in 0..BLOCK {
                    next_doc = next_doc.wrapping_add(docbuf[l]);
                    black_box(next_doc);
                    let len = norms[di];
                    sum += fold.score(tfbuf[l] as f32, len);
                    di += 1;
                    if di >= usable {
                        break;
                    }
                }
                if di >= usable {
                    break;
                }
            }
        }
        let per = t0.elapsed().as_nanos() as f64 / (rounds * usable) as f64;
        println!("(c) SIMD unpack+score    : {per:.2} ns/doc (sum={sum:e})");
        let _ = next_doc;
    }

    // (d) per-block decompress cost only (models block-max WAND skipping:
    // every Nth block, others never touched)
    {
        let t0 = Instant::now();
        let step = 8usize;
        let mut docbuf: [u32; BLOCK] = [0; BLOCK];
        for _ in 0..rounds {
            let mut skipbase = 0usize;
            for (b, w) in w_delta.iter().enumerate().take(nblocks) {
                let _ = (b, w);
                if b % step == 0 {
                    let wd = w_delta[b] as usize;
                    packer.decompress(
                        &pk_delta[skipbase..skipbase + BLOCK * wd / 8],
                        &mut docbuf,
                        wd as u8,
                    );
                    black_box(&docbuf);
                }
                skipbase += BLOCK * w_delta[b] as usize / 8;
            }
        }
        let per_block_visits = rounds * (nblocks / step);
        let per = t0.elapsed().as_nanos() as f64 / (per_block_visits * BLOCK) as f64;
        println!("(d) SIMD per-block cost  : {per:.2} ns/doc-unpacked");
    }
}
