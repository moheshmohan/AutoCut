//! IndicF5's character vocabulary (`vocab.txt`) and the reference's text-to-token step.
//!
//! The vocabulary is one token per line (F5-TTS's pinyin syllables, ASCII, and the letters of
//! the eleven Indian scripts); a token's id is its line number. The reference reads it as
//! Python iterates a text file (universal newlines) and drops each line's last character (the
//! newline), and later lines win a duplicate.
//!
//! Text becomes tokens the way the reference's `convert_char_to_pinyin` does for text without
//! Chinese characters: `;` → `,`, curly quotes straightened, every character its own token,
//! and a space put in front of an ASCII word (jieba's segmentation: a run of letters and digits)
//! that does not follow a space, colon or quote. Then `list_str_to_idx`: unknown tokens → 0
//! (the space). [`Vocab::encode`] also says which characters were unknown, so a front end can
//! report them instead of letting them become silent spaces.

use std::collections::HashMap;
use std::path::Path;

use crate::TtsError;

/// Largest vocabulary file accepted (the released one is 11 KB).
const MAX_BYTES: u64 = 1 << 20;
/// Most tokens.
const MAX_TOKENS: usize = 100_000;
/// Longest token (the pinyin syllables are at most 7 characters).
const MAX_TOKEN_CHARS: usize = 16;

/// The token table.
#[derive(Clone, Debug, PartialEq)]
pub struct Vocab {
    map: HashMap<String, u32>,
    len: usize,
}

/// Text as token ids, and the characters the vocabulary does not have (they became 0).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Encoded {
    pub ids: Vec<u32>,
    pub unknown: Vec<char>,
}

impl Vocab {
    pub fn read(path: &Path) -> Result<Vocab, TtsError> {
        let meta = std::fs::metadata(path).map_err(|e| TtsError::Model(format!("{}: {e}", path.display())))?;
        if meta.len() > MAX_BYTES {
            return Err(TtsError::Model(format!("{}: too large for a vocabulary", path.display())));
        }
        let bytes = std::fs::read(path).map_err(|e| TtsError::Model(format!("{}: {e}", path.display())))?;
        let text = String::from_utf8(bytes).map_err(|_| TtsError::Model(format!("{}: not UTF-8", path.display())))?;
        Vocab::parse(&text)
    }

    /// The tokens of a `vocab.txt`'s text.
    pub fn parse(text: &str) -> Result<Vocab, TtsError> {
        // universal newlines: \r\n and \r end a line like \n
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        let mut map = HashMap::new();
        let mut len = 0usize;
        for line in text.split_inclusive('\n') {
            if len >= MAX_TOKENS {
                return Err(TtsError::Model("the vocabulary has too many tokens".into()));
            }
            // the reference drops the last character, newline or not
            let mut chars = line.chars();
            chars.next_back();
            let token: String = chars.collect();
            if token.chars().count() > MAX_TOKEN_CHARS {
                return Err(TtsError::Model(format!("vocabulary line {}: token too long", len + 1)));
            }
            map.insert(token, len as u32);
            len += 1;
        }
        if len < 2 {
            return Err(TtsError::Model("the vocabulary is empty".into()));
        }
        Ok(Vocab { map, len })
    }

    /// Number of tokens (lines).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn id(&self, token: &str) -> Option<u32> {
        self.map.get(token).copied()
    }

    /// Whether `c` is a token.
    pub fn has(&self, c: char) -> bool {
        let mut b = [0u8; 4];
        self.map.contains_key(c.encode_utf8(&mut b) as &str)
    }

    /// `text` as the reference tokenizes it (see the module docs).
    pub fn encode(&self, text: &str) -> Encoded {
        let mut out = Encoded::default();
        for tok in tokens(text) {
            match self.map.get(&tok) {
                Some(id) => out.ids.push(*id),
                None => {
                    out.ids.push(0);
                    for c in tok.chars() {
                        if !out.unknown.contains(&c) {
                            out.unknown.push(c);
                        }
                    }
                }
            }
        }
        out
    }
}

/// Characters of an ASCII "word" in jieba's segmentation.
fn wordish(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '+' | '#' | '&' | '.' | '_' | '%' | '-')
}

/// The reference's tokens for `text` (one string per character, spaces inserted).
pub(crate) fn tokens(text: &str) -> Vec<String> {
    let text: String = text
        .chars()
        .map(|c| match c {
            ';' => ',',
            '\u{201c}' | '\u{201d}' => '"',
            '\u{2018}' | '\u{2019}' => '\'',
            c => c,
        })
        .collect();
    let mut out: Vec<String> = Vec::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        if wordish(c) {
            // a block of word characters; jieba then splits it into alphanumeric words
            // (`[a-zA-Z0-9]+(\.\d+)?%?`) and the pieces between them
            let start = i;
            while chars.get(i).is_some_and(|c| wordish(*c)) {
                i += 1;
            }
            for seg in split_block(chars.get(start..i).unwrap_or(&[])) {
                if seg.len() > 1 && out.last().is_some_and(|l| !matches!(l.as_str(), " " | ":" | "'" | "\"")) {
                    out.push(" ".into());
                }
                out.extend(seg.iter().map(|c| c.to_string()));
            }
        } else {
            out.push(c.to_string());
            i += 1;
        }
    }
    out
}

/// jieba's `finalseg` split of a block of word characters: alphanumeric runs (with one
/// optional `.digits` and `%`) and the pieces between them.
fn split_block(b: &[char]) -> Vec<&[char]> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut other = 0;
    while i < b.len() {
        if b.get(i).is_some_and(|c| c.is_ascii_alphanumeric()) {
            if other < i {
                out.push(b.get(other..i).unwrap_or(&[]));
            }
            let s = i;
            while b.get(i).is_some_and(|c| c.is_ascii_alphanumeric()) {
                i += 1;
            }
            if b.get(i) == Some(&'.') && b.get(i + 1).is_some_and(|c| c.is_ascii_digit()) {
                i += 1;
                while b.get(i).is_some_and(|c| c.is_ascii_digit()) {
                    i += 1;
                }
            }
            if b.get(i) == Some(&'%') {
                i += 1;
            }
            out.push(b.get(s..i).unwrap_or(&[]));
            other = i;
        } else {
            i += 1;
        }
    }
    if other < b.len() {
        out.push(b.get(other..).unwrap_or(&[]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v() -> Vocab {
        Vocab::parse(" \n!\n,\n.\na\nb\nc\nH\ni\nഇ\nത\n്\n'\n").unwrap()
    }

    #[test]
    fn lines_are_tokens() {
        let v = v();
        assert_eq!(v.len(), 13);
        assert_eq!(v.id(" "), Some(0));
        assert_eq!(v.id("ത"), Some(10));
        assert!(v.has('്') && !v.has('x'));
        // universal newlines, the last line without one loses its last character as in Python
        let w = Vocab::parse(" \r\na\rbc").unwrap();
        assert_eq!((w.id(" "), w.id("a"), w.id("b"), w.id("bc")), (Some(0), Some(1), Some(2), None));
        assert!(Vocab::parse("").is_err());
        assert!(Vocab::parse(&format!("{}\n", "x".repeat(100))).is_err());
    }

    #[test]
    fn text_tokenizes_like_the_reference() {
        // a space before an ASCII word that does not follow a space
        assert_eq!(tokens("Hi,there"), ["H", "i", ",", " ", "t", "h", "e", "r", "e"]);
        assert_eq!(tokens("Hi there"), ["H", "i", " ", "t", "h", "e", "r", "e"]);
        // one-letter words and words after quotes or colons get none
        assert_eq!(tokens("a.b"), ["a", ".", "b"]);
        assert_eq!(tokens("'ok"), ["'", "o", "k"]);
        assert_eq!(tokens("x:ab"), ["x", ":", "a", "b"]);
        // curly quotes and semicolons
        assert_eq!(tokens("“a”;"), ["\"", "a", "\"", ","]);
        // Malayalam: one token per code point, no spaces added
        assert_eq!(tokens("ഇത്"), ["ഇ", "ത", "്"]);
        assert_eq!(tokens("ഇത്ab"), ["ഇ", "ത", "്", " ", "a", "b"]);
        assert_eq!(tokens("3.5%x"), ["3", ".", "5", "%", "x"]);
    }

    #[test]
    fn unknown_characters_are_reported() {
        let e = v().encode("Hi! ഇത് xyz");
        assert_eq!(e.ids.len(), 11);
        assert_eq!(e.unknown, vec!['x', 'y', 'z']);
        assert_eq!(e.ids.get(8..), Some(&[0u32, 0, 0][..]));
        assert!(v().encode("").ids.is_empty());
    }
}
