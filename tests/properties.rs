//! property tests for posting-list invariants (T3: skip anchors, seek, positions)
//!
//! random posting data is frozen at SKIP_STEP boundaries; the galloping reader
//! must agree with the in-memory PostingData for every seek and position lookup.

use proptest::prelude::*;

use zap::posting::{PostingData, PostingReader};

/// random posting data: ascending docids (with small gaps), small tfs, sparse positions
fn post_data_strategy() -> impl Strategy<Value = (Vec<u32>, Vec<u32>)> {
    (0usize..64).prop_flat_map(|len| {
        (
            proptest::collection::vec(1u32..400, len),
            proptest::collection::vec(1u32..4, len),
        )
            .prop_map(|(gaps, tfs)| {
                let mut docids = Vec::with_capacity(gaps.len());
                let mut acc = 0u32;
                for g in gaps {
                    acc += g;
                    docids.push(acc);
                }
                (docids, tfs)
            })
    })
}

proptest! {
    #[test]
    fn seek_matches_first_ge(
        (docids, tfs) in post_data_strategy(),
        target in 0u32..100_000,
    ) {
        prop_assume!(!docids.is_empty());
        let mut data = PostingData::default();
        for (pos, (&d, &tf)) in docids.iter().zip(&tfs).enumerate() {
            data.add(d, tf, vec![pos as u32; tf as usize]);
        }
        let mut blob = Vec::new();
        let info = data.freeze(None).encode_to(&mut blob);
        let mut r = PostingReader::new(&info, &blob);

        let expected = docids.iter().copied().find(|d| *d >= target);
        r.seek_to(target);
        assert_eq!(r.peek_docid(), expected, "seek_to({target})");

        // repeated seek at the same target is stable
        r.seek_to(target);
        assert_eq!(r.peek_docid(), expected, "repeated seek_to({target})");
    }

    #[test]
    fn advanced_positions_match_source(
        (docids, tfs) in post_data_strategy(),
        advance_n in 0usize..64,
    ) {
        prop_assume!(!docids.is_empty());
        let mut data = PostingData::default();
        let src_positions: Vec<Vec<u32>> = docids
            .iter()
            .zip(tfs.iter())
            .enumerate()
            .map(|(i, (_, tf))| vec![i as u32; *tf as usize])
            .collect();
        for (i, (d, tf)) in docids.iter().zip(tfs.iter()).enumerate() {
            data.add(*d, *tf, src_positions[i].clone());
        }
        let mut blob = Vec::new();
        let info = data.freeze(None).encode_to(&mut blob);
        let mut r = PostingReader::new(&info, &blob);
        let n = advance_n.min(docids.len());
        for _ in 0..n {
            r.advance();
        }
        // the peeked doc (if any) carries positions equal to the source at its docidx
        let idx = n;
        if idx < src_positions.len() {
            let mut buf = Vec::new();
            // prime the peek (fresh readers have nothing decoded yet)
            assert_eq!(r.peek().map(|(d, _)| d), Some(docids[idx]));
            assert!(r.peek_positions_into(&mut buf));
            assert_eq!(buf, src_positions[idx], "positions at idx {idx}");
        } else {
            // exhausted reader
            let mut buf = Vec::new();
            assert!(!r.peek_positions_into(&mut buf));
        }
    }
}
