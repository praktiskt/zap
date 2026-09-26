//! analyzer pipeline: charabia segmentation/normalization + stopword filter + optional snowball stemming
//!
//! Tokens borrow their text from the input blob (`Cow`): passthrough lemmas (the
//! bulk of English text) carry a zero-allocation `&str`; rewrites (charabia
//! normalization, snowball stemming) keep the same one owned allocation as before.

use std::borrow::Cow;
use std::sync::Arc;

use charabia::{Tokenizer, TokenizerBuilder};
use rust_stemmers::{Algorithm, Stemmer};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalyzerConf {
    /// apply a snowball stemmer to latin-script tokens
    pub stem: bool,
    /// drop tokens longer than this (chars)
    pub max_token_len: usize,
}

impl Default for AnalyzerConf {
    fn default() -> Self {
        Self {
            stem: true,
            max_token_len: 64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Token<'o> {
    pub text: Cow<'o, str>,
    /// position index among word tokens (per analyzed blob)
    pub pos: u32,
    /// byte offsets into the original text
    pub byte_start: usize,
    pub byte_end: usize,
}

pub struct Analyzer {
    conf: AnalyzerConf,
    tokenizer: Tokenizer<'static>,
    stemmer: Option<Stemmer>,
}

impl Analyzer {
    pub fn new(conf: AnalyzerConf) -> Arc<Self> {
        let tokenizer = TokenizerBuilder::default().into_tokenizer();
        let stemmer = conf.stem.then(|| Stemmer::create(Algorithm::English));
        Arc::new(Self {
            conf,
            tokenizer,
            stemmer,
        })
    }

    /// stem one raw word through the same pipeline as docs
    pub fn stem_word(&self, s: &str) -> String {
        match (&self.stemmer, s.is_ascii()) {
            (Some(st), true) => st.stem(s).into_owned(),
            _ => s.to_string(),
        }
    }

    pub fn conf(&self) -> &AnalyzerConf {
        &self.conf
    }

    /// tokenize a full text (long-token filtering included)
    pub fn tokenize<'a>(&self, text: &'a str) -> Vec<Token<'a>> {
        let mut out = Vec::new();
        let mut pos = 0u32;
        for tok in self.tokenizer.tokenize(text) {
            if !tok.is_word() {
                continue;
            }
            // move the lemma cow out of the token: borrowed cows point into
            // `text` (zero alloc), owned cows are normalization rewrites and
            // keep their allocation
            let lemma_anow: Cow<'_, str> = tok.lemma;
            if lemma_anow.chars().count() > self.conf.max_token_len {
                continue;
            }
            let text_out = match lemma_anow {
                Cow::Borrowed(lemma) => match (&self.stemmer, lemma.is_ascii()) {
                    (Some(s), true) => s.stem(lemma),
                    _ => Cow::Borrowed(lemma),
                },
                owned_cow => {
                    let owned = owned_cow.into_owned();
                    match (&self.stemmer, owned.is_ascii()) {
                        (Some(s), true) => match s.stem(owned.as_str()) {
                            // identical stem: reuse the owned string, no new alloc
                            Cow::Borrowed(_) => Cow::Owned(owned),
                            Cow::Owned(st) => Cow::Owned(st),
                        },
                        _ => Cow::Owned(owned),
                    }
                }
            };
            if text_out.is_empty() {
                continue;
            }
            out.push(Token {
                text: text_out,
                pos,
                byte_start: tok.byte_start,
                byte_end: tok.byte_end,
            });
            pos += 1;
        }
        out
    }

    /// tokenize a query fragment: same pipeline without stemming (stem expansion handled by term dict)
    pub fn tokenize_query<'a>(&self, text: &'a str) -> Vec<Token<'a>> {
        let mut out = Vec::new();
        let mut pos = 0u32;
        for tok in self.tokenizer.tokenize(text) {
            if !tok.is_word() {
                continue;
            }
            let lemma_anow: Cow<'_, str> = tok.lemma;
            out.push(Token {
                text: lemma_anow,
                pos,
                byte_start: tok.byte_start,
                byte_end: tok.byte_end,
            });
            pos += 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_basics() {
        let a = Analyzer::new(AnalyzerConf::default());
        let toks = a.tokenize("The quick brown foxes, running!");
        let texts: Vec<&str> = toks.iter().map(|t| t.text.as_ref()).collect();
        assert!(texts.contains(&"quick"));
        // stemmer folds plurals and gerunds
        assert!(texts.iter().any(|t| t.starts_with("fox")));
        assert!(texts.iter().any(|t| t.starts_with("run")));
        // positions are compacted over word tokens
        for (i, t) in toks.iter().enumerate() {
            assert_eq!(t.pos, i as u32);
            assert_eq!(toks.len(), 5);
        }
    }

    #[test]
    fn unicode_and_cjk() {
        let a = Analyzer::new(AnalyzerConf::default());
        let toks = a.tokenize("café naïve 中文搜索");
        let texts: Vec<&str> = toks.iter().map(|t| t.text.as_ref()).collect();
        assert!(texts.contains(&"cafe"));
        assert!(texts.contains(&"naiv"));
        assert!(texts.contains(&"中文") || texts.contains(&"搜索"));
    }

    #[test]
    fn long_token_dropped() {
        let a = Analyzer::new(AnalyzerConf {
            stem: false,
            max_token_len: 5,
        });
        let toks = a.tokenize("short verylongtoken");
        let texts: Vec<&str> = toks.iter().map(|t| t.text.as_ref()).collect();
        assert_eq!(texts, vec!["short"]);
    }
}
