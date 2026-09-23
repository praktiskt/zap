//! T9: docstore decode must never panic on malformed input, and valid
//! raw/compressed v2 blocks (flags + fields + optional id/metadata) must
//! roundtrip exactly

use proptest::prelude::*;

use zap::schema::StoredDoc;
use zap::segment::read_stored;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// any random byte sequence decodes to None or a valid doc — never panics
    #[test]
    fn garbage_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..256)) {
        let _ = read_stored(&bytes);
    }

    #[test]
    fn roundtrip_block(
        fields in proptest::collection::vec((0u32..3, 0usize..400), 1..4),
        use_id in proptest::bool::ANY,
        use_meta in proptest::bool::ANY,
    ) {
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
        let id = if use_id {
            Some("doc-å-id-42".to_string())
        } else {
            None
        };
        let metadata = if use_meta {
            Some(serde_json::json!({"k": [1, 2, 3], "nested": {"s": "ä"}}))
        } else {
            None
        };
        let mut flags = 0u8;
        if id.is_some() {
            flags |= 0b0000_0001;
        }
        if metadata.is_some() {
            flags |= 0b0000_0010;
        }
        let mut block = Vec::new();
        block.push(flags);
        block.push(pairs.len() as u8);
        for (fid, val) in &pairs {
            block.extend_from_slice(&fid.to_le_bytes());
            block.extend_from_slice(&(val.len() as u32).to_le_bytes());
            block.extend_from_slice(val.as_bytes());
        }
        if let Some(id) = &id {
            block.extend_from_slice(&(id.len() as u32).to_le_bytes());
            block.extend_from_slice(id.as_bytes());
        }
        if let Some(meta) = &metadata {
            let json = serde_json::to_vec(meta).unwrap();
            block.extend_from_slice(&(json.len() as u32).to_le_bytes());
            block.extend_from_slice(&json);
        }
        // raw path (block <= threshold): [tag, len, bytes]
        let mut raw = Vec::new();
        raw.push(0x00u8);
        raw.extend_from_slice(&(block.len() as u32).to_le_bytes());
        raw.extend_from_slice(&block);
        let got: StoredDoc = read_stored(&raw).expect("raw block decodes");
        assert_eq!(got.fields, pairs);
        assert_eq!(got.id, id);
        assert_eq!(got.metadata, metadata);

        // compressed path: same payload through lz4
        let mut comp = lz4_flex::compress_prepend_size(&block);
        comp.insert(0, 0x01u8);
        let got: StoredDoc = read_stored(&comp).expect("compressed block decodes");
        assert_eq!(got.fields, pairs);
        assert_eq!(got.id, id);
        assert_eq!(got.metadata, metadata);
    }

    /// unknown tag byte -> None (no panic), unknown flags must not confuse decode
    #[test]
    fn bad_tags_rejected(tag in 2u8..=255, bytes in proptest::collection::vec(any::<u8>(), 0..64)) {
        let mut raw = vec![tag];
        raw.extend_from_slice(&4u32.to_le_bytes());
        raw.extend(bytes);
        assert!(read_stored(&raw).is_none());
    }
}
