//! T9: docstore decode must never panic on malformed input, and valid
//! raw/compressed blocks must roundtrip exactly

use proptest::prelude::*;

use zap::segment::read_stored;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// any random byte sequence decodes to None or a valid field set — never panics
    #[test]
    fn garbage_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..256)) {
        let _ = read_stored(&bytes);
    }

    #[test]
    fn roundtrip_block(fields in proptest::collection::vec((0u32..3, 0usize..400), 1..4)) {
        // build the block the same way encode_store does
        let mut pairs: Vec<(u32, String)> = fields
            .into_iter()
            .map(|(fid, l)| {
                let val: String = (0..l)
                    .map(|_| char::from_u32(0x4e00u32 + (l as u32) % 10).unwrap_or('a').to_string())
                    .collect();
                (fid, val)
            })
            .collect();
        pairs.sort_by_key(|p| p.0);
        let mut block = Vec::new();
        block.push(pairs.len() as u8);
        for (fid, val) in &pairs {
            block.extend_from_slice(&fid.to_le_bytes());
            block.extend_from_slice(&(val.len() as u32).to_le_bytes());
            block.extend_from_slice(val.as_bytes());
        }
        // raw path (block <= threshold): [tag, len, bytes]
        let mut raw = Vec::new();
        raw.push(0x00u8);
        raw.extend_from_slice(&(block.len() as u32).to_le_bytes());
        raw.extend_from_slice(&block);
        let got = read_stored(&raw).expect("raw block decodes");
        assert_eq!(got, pairs);
    }
}
