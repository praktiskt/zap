//! query AST + recursive-descent parser
//!
//! grammar (lowest → highest): `Or` ; `And` (explicit `AND` or adjacency) / `NOT` ; atoms:
//! `word`, `"phrase"`, `field:term`, `prefix*`, `wil*dcard?`, `fuzzy~N`, `( full query )`

#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Term(String),
    Phrase {
        terms: Vec<String>,
        slop: u32,
    },
    Prefix(String),
    /// glob with embedded `*` / `?`
    Wildcard(String),
    Fuzzy {
        term: String,
        distance: u8,
    },
    /// restrict a subtree to one field (None = all text fields)
    Field {
        field: Option<String>,
        node: Box<Node>,
    },
    Not(Box<Node>),
    And(Vec<Node>),
    Or(Vec<Node>),
}

#[derive(Debug, Clone, PartialEq)]
enum Tk {
    LParen,
    RParen,
    And,
    Or,
    Not,
    /// field name if the token contained `field:...`
    Fielded(Option<String>, String),
    Phrase(String, u32),
}

#[derive(Debug, PartialEq)]
pub enum ParseError {
    UnmatchedParen { open: usize },
    UnexpectedEnd,
    Empty,
}

fn lex(q: &str) -> Result<Vec<Tk>, ParseError> {
    let b: Vec<char> = q.chars().collect();
    let n = b.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        match b[i] {
            c if c.is_whitespace() => i += 1,
            '(' => {
                out.push(Tk::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tk::RParen);
                i += 1;
            }
            '"' => {
                // quoted phrase up to closing quote (or end); optional `slop` suffix
                let mut j = i + 1;
                while j < n && b[j] != '"' {
                    j += 1;
                }
                let inner: String = b[i + 1..j].iter().collect();
                let mut after = if j < n { j + 1 } else { j };
                let mut slop = 0;
                if j < n && b[j] == '"' && j + 1 < n && b[j + 1] == '~' {
                    let mut k = j + 2;
                    let mut digits = String::new();
                    while k < n && b[k].is_ascii_digit() {
                        digits.push(b[k]);
                        k += 1;
                    }
                    if !digits.is_empty() {
                        if let Ok(v) = digits.parse::<u32>() {
                            slop = v;
                        }
                        after = k;
                    }
                }
                if inner.trim().is_empty() {
                    return Err(ParseError::UnexpectedEnd);
                }
                out.push(Tk::Phrase(inner, slop));
                i = after;
            }
            _ => {
                // plain word; may contain field: prefix
                let mut j = i;
                let mut raw = String::new();
                while j < n && !b[j].is_whitespace() && b[j] != '(' && b[j] != ')' && b[j] != '"' {
                    raw.push(b[j]);
                    j += 1;
                }
                if raw.is_empty() {
                    i += 1;
                    continue;
                }
                let tk = match raw.split_once(':') {
                    Some((field, rest)) if is_word(field) => {
                        Tk::Fielded(Some(field.to_lowercase()), rest.to_string())
                    }
                    _ => match raw.as_str() {
                        "AND" => Tk::And,
                        "OR" => Tk::Or,
                        "NOT" => Tk::Not,
                        _ => Tk::Fielded(None, raw),
                    },
                };
                out.push(tk);
                i = j;
            }
        }
    }
    Ok(out)
}

fn is_word(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

fn word_node(rest: &str) -> Node {
    if rest.contains('*') || rest.contains('?') {
        if rest.ends_with('*') && !rest[..rest.len() - 1].contains('*') {
            Node::Prefix(rest.trim_end_matches('*').to_string())
        } else {
            Node::Wildcard(rest.to_string())
        }
    } else if let Some((term, n)) = split_fuzzy(rest) {
        Node::Fuzzy { term, distance: n }
    } else {
        Node::Term(rest.to_lowercase())
    }
}

/// `term~2` → ("term", 2); `term~` → auto distance 1
fn split_fuzzy(s: &str) -> Option<(String, u8)> {
    if !s.contains('~') {
        return None;
    }
    let (term, suff) = s.rsplit_once('~')?;
    if term.is_empty() {
        return None;
    }
    let d = if suff.is_empty() {
        1
    } else {
        suff.parse::<u8>().ok()?.max(1)
    };
    Some((term.to_lowercase(), d))
}

struct P {
    toks: Vec<Tk>,
    i: usize,
}

impl P {
    fn peek(&self) -> Option<&Tk> {
        self.toks.get(self.i)
    }

    fn term_start(&self) -> bool {
        if matches!(self.peek(), Some(Tk::LParen)) {
            return true;
        }
        match self.peek() {
            Some(Tk::Fielded(f, w)) => f.is_some() || !w.is_empty(),
            _ => false,
        }
    }

    fn parse_or(&mut self) -> Result<Node, ParseError> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Some(Tk::Or)) {
            self.i += 1;
            let right = self.parse_and()?;
            left = Node::Or(vec![left, right]);
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Node, ParseError> {
        let mut items = vec![self.parse_not()?];
        loop {
            // boolean reads before mutation to dodge borrows
            let has_and = matches!(self.peek(), Some(Tk::And));
            let starts = self.term_start();
            if has_and {
                self.i += 1;
                items.push(self.parse_not()?);
                continue;
            }
            if starts {
                items.push(self.parse_not()?);
                continue;
            }
            break;
        }
        Ok(if items.len() == 1 {
            items.pop().unwrap()
        } else {
            Node::And(items)
        })
    }

    fn parse_not(&mut self) -> Result<Node, ParseError> {
        if matches!(self.peek(), Some(Tk::Not)) {
            self.i += 1;
            return Ok(Node::Not(Box::new(self.parse_not()?)));
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Result<Node, ParseError> {
        match self.peek() {
            Some(Tk::LParen) => {
                self.i += 1;
                let node = self.parse_or()?;
                if !matches!(self.peek(), Some(Tk::RParen)) {
                    return Err(ParseError::UnmatchedParen { open: self.i });
                }
                self.i += 1;
                Ok(node)
            }
            Some(Tk::Phrase(raw, slop)) => {
                let terms: Vec<String> = raw.split_whitespace().map(|t| t.to_lowercase()).collect();
                let slop = *slop;
                if terms.is_empty() {
                    return Err(ParseError::UnexpectedEnd);
                }
                self.i += 1;
                Ok(Node::Phrase { terms, slop })
            }
            Some(Tk::Fielded(field, rest)) => {
                let field = field.clone();
                let rest = rest.clone();
                self.i += 1;
                let node = if rest.is_empty() {
                    // `field:( a OR b )`
                    match self.peek() {
                        Some(Tk::LParen) => self.parse_atom()?,
                        _ => return Err(ParseError::UnexpectedEnd),
                    }
                } else {
                    word_node(&rest)
                };
                match field {
                    Some(f) => Ok(Node::Field {
                        field: Some(f),
                        node: Box::new(node),
                    }),
                    None => Ok(node),
                }
            }
            _ => Err(ParseError::UnexpectedEnd),
        }
    }
}

pub fn parse(q: &str) -> Result<Node, ParseError> {
    let toks = lex(q)?;
    if toks.is_empty() {
        return Err(ParseError::Empty);
    }
    let mut p = P { toks, i: 0 };
    let node = p.parse_or()?;
    if p.i < p.toks.len() {
        return Err(ParseError::UnmatchedParen { open: p.i });
    }
    Ok(node)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bools() {
        assert_eq!(
            parse("a b"),
            Ok(Node::And(vec![
                Node::Term("a".into()),
                Node::Term("b".into())
            ]))
        );
        assert_eq!(
            parse("a OR b"),
            Ok(Node::Or(vec![
                Node::Term("a".into()),
                Node::Term("b".into())
            ]))
        );
        assert_eq!(
            parse("a AND NOT b").unwrap(),
            Node::And(vec![
                Node::Term("a".into()),
                Node::Not(Box::new(Node::Term("b".into())))
            ])
        );
    }

    #[test]
    fn atoms() {
        assert_eq!(parse("fox*"), Ok(Node::Prefix("fox".into())));
        assert_eq!(parse("fo*x"), Ok(Node::Wildcard("fo*x".into())));
        assert_eq!(
            parse("run~2"),
            Ok(Node::Fuzzy {
                term: "run".into(),
                distance: 2
            })
        );
        assert_eq!(
            parse("foo~"),
            Ok(Node::Fuzzy {
                term: "foo".into(),
                distance: 1
            })
        );
        assert_eq!(
            parse("\"quick brown fox\""),
            Ok(Node::Phrase {
                terms: vec!["quick".into(), "brown".into(), "fox".into()],
                slop: 0
            })
        );
    }

    #[test]
    fn field_scoping() {
        assert_eq!(
            parse("title:hello"),
            Ok(Node::Field {
                field: Some("title".into()),
                node: Box::new(Node::Term("hello".into()))
            })
        );
        assert_eq!(
            parse("title:(a OR b)"),
            Ok(Node::Field {
                field: Some("title".into()),
                node: Box::new(Node::Or(vec![
                    Node::Term("a".into()),
                    Node::Term("b".into())
                ]))
            })
        );
        assert_eq!(
            parse("title:world*"),
            Ok(Node::Field {
                field: Some("title".into()),
                node: Box::new(Node::Prefix("world".into()))
            })
        );
    }

    #[test]
    fn precedence_and_parens() {
        // OR binds looser AND: a b OR c => (a AND b) OR c
        assert_eq!(
            parse("a b OR c").unwrap(),
            Node::Or(vec![
                Node::And(vec![Node::Term("a".into()), Node::Term("b".into())]),
                Node::Term("c".into())
            ])
        );
        assert_eq!(
            parse("(a b) OR c").unwrap(),
            Node::Or(vec![
                Node::And(vec![Node::Term("a".into()), Node::Term("b".into())]),
                Node::Term("c".into())
            ])
        );
        assert_eq!(
            parse("NOT a").unwrap(),
            Node::Not(Box::new(Node::Term("a".into())))
        );
    }

    #[test]
    fn errors() {
        assert!(matches!(
            parse("(a"),
            Err(ParseError::UnmatchedParen { .. })
        ));
        assert!(parse("").is_err());
    }
}
