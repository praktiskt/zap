fn main() {
    use charabia::{Token, TokenizerBuilder};
    let text = "The quick brown foxes. Hyphen-word, café naïve; 中文搜索 日本語テスト";
    let mut b = TokenizerBuilder::default();
    let tokenizer = b.build();
    for tok in tokenizer.tokenize(text) {
        let Token {
            lemma,
            byte_start,
            byte_end,
            kind,
            script,
            ..
        } = tok;
        println!("kind={kind:?} script={script:?} [{byte_start}..{byte_end}] lemma={lemma}");
    }
}
