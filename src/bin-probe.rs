fn main() {
    let mut b = charabia::TokenizerBuilder::default();
    let t = b.build();
    for tok in t.tokenize("café naïve 中文搜索") {
        println!("{:?} |{:?}|", tok.lemma(), tok.script());
    }
}
