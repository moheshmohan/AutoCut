//! The text encoder's tokenizer, read from the release's `tokenizer.json` (Qwen3-Embedding-0.6B):
//! a GPT-2 style **byte-level BPE**.
//!
//! 1. Added tokens (`<|endoftext|>`, `<|im_start|>`, …) are cut out of the raw text first,
//!    longest match first.
//! 2. The rest is NFC-normalised and split by the Qwen2 pre-tokenizer pattern
//!    `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!    (written out as a scanner: the lookahead has no `regex` equivalent; leftmost alternative
//!    wins, as in Oniguruma).
//! 3. Each piece's UTF-8 bytes map to the byte-level alphabet and are merged lowest rank first
//!    (leftmost on ties).
//! 4. The post-processor template appends `<|endoftext|>` (the embedding model's pooling token);
//!    truncation keeps room for it.
//!
//! Any other tokenizer configuration (another model type, normalizer or pre-tokenizer) is refused
//! when loading. Token ids equal the Hugging Face library's (oracle test).

use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

use unicode_general_category::{GeneralCategory, get_general_category};

use super::{R, err};

/// Longest text tokenized at once, bytes.
pub const MAX_TEXT_BYTES: usize = 64 * 1024;
const MAX_FILE_BYTES: u64 = 64 << 20;
/// The exact pre-tokenizer pattern this scanner implements.
const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    /// `(left, right)` → `(rank, merged)`.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Byte → its token (every byte has one in a byte-level vocabulary).
    bytes: [u32; 256],
    /// Added tokens, longest first: `(text, id)`.
    added: Vec<(String, u32)>,
    /// Ids the post-processor appends to every sequence.
    suffix: Vec<u32>,
}

/// GPT-2's byte → printable character table.
fn byte_chars() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let c = if printable {
            b
        } else {
            n += 1;
            255 + n
        };
        if let (Some(slot), Some(ch)) = (out.get_mut(b as usize), char::from_u32(c)) {
            *slot = ch;
        }
    }
    out
}

impl Tokenizer {
    pub fn load(path: &Path) -> R<Tokenizer> {
        let len = std::fs::metadata(path).map_err(|e| err(format!("{}: {e}", path.display())))?.len();
        if len > MAX_FILE_BYTES {
            return Err(err("tokenizer.json is too large"));
        }
        let bytes = std::fs::read(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
        Tokenizer::parse(&bytes)
    }

    pub fn parse(bytes: &[u8]) -> R<Tokenizer> {
        let v: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| err(format!("tokenizer.json: {e}")))?;
        let model = v.get("model").ok_or_else(|| err("tokenizer.json: no model"))?;
        if model.get("type").and_then(|t| t.as_str()) != Some("BPE") {
            return Err(err("tokenizer.json: only BPE tokenizers are supported"));
        }
        if model.get("byte_fallback").and_then(|b| b.as_bool()) == Some(true) {
            return Err(err("tokenizer.json: byte fallback is not supported (a byte-level vocabulary is)"));
        }
        for k in ["continuing_subword_prefix", "end_of_word_suffix"] {
            if model.get(k).and_then(|p| p.as_str()).is_some_and(|p| !p.is_empty()) {
                return Err(err("tokenizer.json: subword prefixes and suffixes are not supported"));
            }
        }
        if v.get("normalizer").and_then(|n| n.get("type")).and_then(|t| t.as_str()) != Some("NFC") {
            return Err(err("tokenizer.json: only the NFC normalizer is supported"));
        }
        check_pre_tokenizer(v.get("pre_tokenizer"))?;
        let table = model.get("vocab").and_then(|t| t.as_object()).ok_or_else(|| err("tokenizer.json: no vocabulary"))?;
        if table.len() > 2_000_000 {
            return Err(err("tokenizer.json: the vocabulary is too large"));
        }
        let mut vocab = HashMap::with_capacity(table.len());
        for (piece, id) in table {
            let id = id.as_u64().and_then(|i| u32::try_from(i).ok()).ok_or_else(|| err(format!("tokenizer.json: bad id for `{piece}`")))?;
            vocab.insert(piece.clone(), id);
        }
        let chars = byte_chars();
        let mut byte_ids = [0u32; 256];
        for (b, slot) in byte_ids.iter_mut().enumerate() {
            let c = chars.get(b).copied().unwrap_or('\0');
            *slot = *vocab.get(c.encode_utf8(&mut [0; 4]) as &str).ok_or_else(|| err("tokenizer.json: the vocabulary is not byte-level"))?;
        }
        let mut added = Vec::new();
        if let Some(list) = v.get("added_tokens").and_then(|a| a.as_array()) {
            if list.len() > 4096 {
                return Err(err("tokenizer.json: too many added tokens"));
            }
            for a in list {
                let (Some(content), Some(id)) = (a.get("content").and_then(|c| c.as_str()), a.get("id").and_then(|i| i.as_u64())) else {
                    return Err(err("tokenizer.json: bad added token"));
                };
                let id = u32::try_from(id).map_err(|_| err("tokenizer.json: bad added token id"))?;
                if a.get("lstrip").and_then(|x| x.as_bool()) == Some(true) || a.get("rstrip").and_then(|x| x.as_bool()) == Some(true) {
                    return Err(err("tokenizer.json: stripping added tokens are not supported"));
                }
                if !content.is_empty() {
                    added.push((content.to_string(), id));
                }
            }
        }
        added.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        let list = model.get("merges").and_then(|m| m.as_array()).ok_or_else(|| err("tokenizer.json: no merges"))?;
        if list.len() > 4_000_000 {
            return Err(err("tokenizer.json: too many merges"));
        }
        let mut merges = HashMap::with_capacity(list.len());
        for (rank, m) in list.iter().enumerate() {
            // ["a", "b"] (current files) or "a b" (older ones)
            let (a, b) = match m {
                serde_json::Value::String(s) => s.split_once(' ').ok_or_else(|| err(format!("tokenizer.json: bad merge `{s}`")))?,
                serde_json::Value::Array(p) => match (p.first().and_then(|x| x.as_str()), p.get(1).and_then(|x| x.as_str())) {
                    (Some(a), Some(b)) => (a, b),
                    _ => return Err(err("tokenizer.json: bad merge")),
                },
                _ => return Err(err("tokenizer.json: bad merge")),
            };
            let (Some(ia), Some(ib)) = (vocab.get(a), vocab.get(b)) else { continue };
            let Some(im) = vocab.get(&format!("{a}{b}")) else { continue };
            let rank = u32::try_from(rank).map_err(|_| err("tokenizer.json: too many merges"))?;
            merges.entry((*ia, *ib)).or_insert((rank, *im));
        }
        let suffix = post_suffix(v.get("post_processor"), &added)?;
        Ok(Tokenizer { vocab, merges, bytes: byte_ids, added, suffix })
    }

    /// The id of a vocabulary entry or added token.
    pub fn id(&self, piece: &str) -> Option<u32> {
        self.vocab.get(piece).copied().or_else(|| self.added.iter().find(|(s, _)| s == piece).map(|(_, i)| *i))
    }

    /// Token ids of `text` with the post-processor's suffix, at most `max_len` of them (the text
    /// is truncated, the suffix kept), like the reference's `truncation=True, max_length=…`.
    pub fn encode(&self, text: &str, max_len: usize) -> R<Vec<u32>> {
        if text.len() > MAX_TEXT_BYTES {
            return Err(err(format!("text to tokenize is longer than {} KB", MAX_TEXT_BYTES / 1024)));
        }
        let mut ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // the next added token (earliest; longest at the same place)
            let mut found: Option<(usize, usize, u32)> = None;
            for (s, id) in &self.added {
                if let Some(p) = rest.find(s.as_str())
                    && found.is_none_or(|(fp, fl, _)| p < fp || (p == fp && s.len() > fl))
                {
                    found = Some((p, s.len(), *id));
                }
            }
            rest = match found {
                Some((p, l, id)) => {
                    self.encode_plain(rest.get(..p).unwrap_or(""), &mut ids);
                    ids.push(id);
                    rest.get(p + l..).unwrap_or("")
                }
                None => {
                    self.encode_plain(rest, &mut ids);
                    ""
                }
            };
        }
        let keep = max_len.saturating_sub(self.suffix.len());
        ids.truncate(keep);
        ids.extend_from_slice(&self.suffix);
        Ok(ids)
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let nfc = icu_normalizer::ComposingNormalizerBorrowed::new_nfc().normalize(text);
        let chars: Vec<char> = nfc.chars().collect();
        let mut i = 0;
        let mut buf = String::new();
        while i < chars.len() {
            let end = next_piece(&chars, i).clamp(i + 1, chars.len());
            buf.clear();
            buf.extend(chars.get(i..end).unwrap_or(&[]).iter());
            self.bpe(buf.as_bytes(), out);
            i = end;
        }
    }

    /// Byte-level BPE of one piece.
    fn bpe(&self, piece: &[u8], out: &mut Vec<u32>) {
        struct Sym {
            id: u32,
            prev: Option<usize>,
            next: Option<usize>,
            alive: bool,
        }
        let n = piece.len();
        let mut syms: Vec<Sym> = piece
            .iter()
            .enumerate()
            .map(|(i, b)| Sym {
                id: self.bytes.get(usize::from(*b)).copied().unwrap_or(0),
                prev: i.checked_sub(1),
                next: (i + 1 < n).then_some(i + 1),
                alive: true,
            })
            .collect();
        // (rank, left index, left id, right id): min-heap by rank, then position
        let mut heap: BinaryHeap<std::cmp::Reverse<(u32, usize, u32, u32)>> = BinaryHeap::new();
        let candidate = |syms: &Vec<Sym>, i: usize| -> Option<(u32, usize, u32, u32)> {
            let a = syms.get(i)?;
            let b = syms.get(a.next?)?;
            let (rank, _) = self.merges.get(&(a.id, b.id))?;
            Some((*rank, i, a.id, b.id))
        };
        for i in 0..syms.len() {
            if let Some(c) = candidate(&syms, i) {
                heap.push(std::cmp::Reverse(c));
            }
        }
        while let Some(std::cmp::Reverse((_, i, ida, idb))) = heap.pop() {
            // stale entries: the pair changed since it was queued
            let Some(a) = syms.get(i) else { continue };
            if !a.alive || a.id != ida {
                continue;
            }
            let Some(j) = a.next else { continue };
            let Some(b) = syms.get(j) else { continue };
            if !b.alive || b.id != idb {
                continue;
            }
            let Some((_, merged)) = self.merges.get(&(ida, idb)).copied() else { continue };
            let b_next = b.next;
            if let Some(a) = syms.get_mut(i) {
                a.id = merged;
                a.next = b_next;
            }
            if let Some(b) = syms.get_mut(j) {
                b.alive = false;
            }
            if let Some(nx) = b_next
                && let Some(ns) = syms.get_mut(nx)
            {
                ns.prev = Some(i);
            }
            if let Some(p) = syms.get(i).and_then(|a| a.prev)
                && let Some(c) = candidate(&syms, p)
            {
                heap.push(std::cmp::Reverse(c));
            }
            if let Some(c) = candidate(&syms, i) {
                heap.push(std::cmp::Reverse(c));
            }
        }
        out.extend(syms.iter().filter(|s| s.alive).map(|s| s.id));
    }
}

fn is_letter(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

fn is_number(c: char) -> bool {
    matches!(get_general_category(c), GeneralCategory::DecimalNumber | GeneralCategory::LetterNumber | GeneralCategory::OtherNumber)
}

fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// Neither whitespace, letter nor number.
fn is_other(c: char) -> bool {
    !c.is_whitespace() && !is_letter(c) && !is_number(c)
}

/// Case-insensitive match of an ASCII letter (with the case folds Oniguruma applies).
fn ci(c: Option<&char>, lower: char) -> bool {
    match c {
        Some(c) => c.to_ascii_lowercase() == lower || (lower == 's' && *c == '\u{17f}'),
        None => false,
    }
}

/// End of the pre-tokenizer piece starting at `i` (the leftmost alternative that matches).
fn next_piece(s: &[char], i: usize) -> usize {
    let n = s.len();
    let at = |k: usize| s.get(k).copied();
    let Some(c) = at(i) else { return n };
    // 1. contractions
    if c == '\'' {
        let (a, b) = (s.get(i + 1), s.get(i + 2));
        if ci(a, 's') || ci(a, 't') {
            return i + 2;
        }
        if (ci(a, 'r') || ci(a, 'v')) && ci(b, 'e') {
            return i + 3;
        }
        if ci(a, 'm') {
            return i + 2;
        }
        if ci(a, 'l') && ci(b, 'l') {
            return i + 3;
        }
        if ci(a, 'd') {
            return i + 2;
        }
    }
    let letters_from = |k: usize| -> usize {
        let mut e = k;
        while at(e).is_some_and(is_letter) {
            e += 1;
        }
        e
    };
    // 2. an optional non-letter, non-number, non-newline character, then letters
    if !is_newline(c) && !is_letter(c) && !is_number(c) && at(i + 1).is_some_and(is_letter) {
        return letters_from(i + 1);
    }
    if is_letter(c) {
        return letters_from(i);
    }
    // 3. one number character
    if is_number(c) {
        return i + 1;
    }
    // 4. an optional space, then other characters, then newlines
    let start = if c == ' ' && at(i + 1).is_some_and(is_other) {
        Some(i + 1)
    } else if is_other(c) {
        Some(i)
    } else {
        None
    };
    if let Some(mut e) = start {
        while at(e).is_some_and(is_other) {
            e += 1;
        }
        while at(e).is_some_and(is_newline) {
            e += 1;
        }
        return e;
    }
    // c is whitespace from here on
    let mut run = i;
    while at(run).is_some_and(char::is_whitespace) {
        run += 1;
    }
    // 5. whitespace ending in newlines (up to the last newline of the run)
    if let Some(last) = (i..run).rev().find(|k| at(*k).is_some_and(is_newline)) {
        return last + 1;
    }
    // 6. whitespace not followed by a non-space: all but the last before a word
    if run == n {
        return run;
    }
    if run - i >= 2 {
        return run - 1;
    }
    // 7. whitespace
    run
}

fn check_pre_tokenizer(p: Option<&serde_json::Value>) -> R<()> {
    let bad = || err("tokenizer.json: this pre-tokenizer is not supported (Qwen2 split + byte level only)");
    let p = p.ok_or_else(bad)?;
    if p.get("type").and_then(|t| t.as_str()) != Some("Sequence") {
        return Err(bad());
    }
    let list = p.get("pretokenizers").and_then(|l| l.as_array()).ok_or_else(bad)?;
    let [split, bytes] = list.as_slice() else { return Err(bad()) };
    let ok_split = split.get("type").and_then(|t| t.as_str()) == Some("Split")
        && split.get("pattern").and_then(|p| p.get("Regex")).and_then(|r| r.as_str()) == Some(PATTERN)
        && split.get("behavior").and_then(|b| b.as_str()) == Some("Isolated")
        && split.get("invert").and_then(|b| b.as_bool()) == Some(false);
    let ok_bytes = bytes.get("type").and_then(|t| t.as_str()) == Some("ByteLevel")
        && bytes.get("add_prefix_space").and_then(|b| b.as_bool()) == Some(false)
        && bytes.get("use_regex").and_then(|b| b.as_bool()) == Some(false);
    if ok_split && ok_bytes { Ok(()) } else { Err(bad()) }
}

/// The ids a `TemplateProcessing` post-processor puts after a single sequence (nothing before it
/// is supported).
fn post_suffix(p: Option<&serde_json::Value>, added: &[(String, u32)]) -> R<Vec<u32>> {
    let Some(p) = p.filter(|p| !p.is_null()) else { return Ok(Vec::new()) };
    let procs: Vec<&serde_json::Value> = match p.get("type").and_then(|t| t.as_str()) {
        Some("Sequence") => p.get("processors").and_then(|l| l.as_array()).map(|l| l.iter().collect()).unwrap_or_default(),
        _ => vec![p],
    };
    let mut out = Vec::new();
    for q in procs {
        match q.get("type").and_then(|t| t.as_str()) {
            Some("ByteLevel") => {}
            Some("TemplateProcessing") => {
                let single = q.get("single").and_then(|s| s.as_array()).ok_or_else(|| err("tokenizer.json: bad template"))?;
                let mut seen_seq = false;
                for part in single {
                    if part.get("Sequence").is_some() {
                        seen_seq = true;
                    } else if let Some(sp) = part.get("SpecialToken").and_then(|s| s.get("id")).and_then(|i| i.as_str()) {
                        if !seen_seq {
                            return Err(err("tokenizer.json: tokens before the sequence are not supported"));
                        }
                        let ids = q
                            .get("special_tokens")
                            .and_then(|m| m.get(sp))
                            .and_then(|m| m.get("ids"))
                            .and_then(|l| l.as_array())
                            .map(|l| l.iter().filter_map(|i| i.as_u64().and_then(|i| u32::try_from(i).ok())).collect::<Vec<_>>())
                            .or_else(|| added.iter().find(|(s, _)| s == sp).map(|(_, i)| vec![*i]))
                            .ok_or_else(|| err(format!("tokenizer.json: unknown template token `{sp}`")))?;
                        out.extend(ids.into_iter().take(16));
                    }
                }
            }
            _ => return Err(err("tokenizer.json: this post-processor is not supported")),
        }
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A tiny byte-level tokenizer.json with merges for "he", "ll", "hell", "o", " w" and the
    /// Qwen2 pre-tokenizer, NFC and the `<|endoftext|>` template.
    pub(crate) fn tiny_json() -> String {
        let chars = byte_chars();
        let mut vocab = serde_json::Map::new();
        for (i, c) in chars.iter().enumerate() {
            vocab.insert(c.to_string(), serde_json::json!(i));
        }
        let sp = chars[b' ' as usize];
        let extra = ["he".to_string(), "ll".to_string(), "hell".to_string(), "hello".to_string(), format!("{sp}w")];
        for (k, e) in extra.iter().enumerate() {
            vocab.insert(e.clone(), serde_json::json!(256 + k));
        }
        let merges = serde_json::json!([["h", "e"], ["l", "l"], ["he", "ll"], ["hell", "o"], [sp.to_string(), "w"]]);
        serde_json::json!({
            "version": "1.0",
            "added_tokens": [
                {"id": 300, "content": "<|endoftext|>", "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 301, "content": "<|im_start|>", "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": PATTERN}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
            ]},
            "post_processor": {"type": "Sequence", "processors": [
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
                {"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}}],
                 "pair": [], "special_tokens": {"<|endoftext|>": {"id": "<|endoftext|>", "ids": [300], "tokens": ["<|endoftext|>"]}}}
            ]},
            "decoder": {"type": "ByteLevel"},
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "", "end_of_word_suffix": "",
                      "fuse_unk": false, "byte_fallback": false, "vocab": vocab, "merges": merges}
        })
        .to_string()
    }

    fn pieces(s: &str) -> Vec<String> {
        let c: Vec<char> = s.chars().collect();
        let mut i = 0;
        let mut out = Vec::new();
        while i < c.len() {
            let e = next_piece(&c, i).clamp(i + 1, c.len());
            out.push(c[i..e].iter().collect());
            i = e;
        }
        out
    }

    #[test]
    fn pre_tokenizer_follows_the_pattern() {
        assert_eq!(pieces("Hello world"), ["Hello", " world"]);
        assert_eq!(pieces("  two  spaces\n\nand"), [" ", " two", " ", " spaces", "\n\n", "and"]);
        assert_eq!(pieces("It's we'll I'M"), ["It", "'s", " we", "'ll", " I", "'M"]);
        assert_eq!(pieces("a 123!!! ok"), ["a", " ", "1", "2", "3", "!!!", " ok"]);
        assert_eq!(pieces("x  \n  y   "), ["x", "  \n", " ", " y", "   "]);
        assert_eq!(pieces("(word) end."), ["(word", ")", " end", "."]);
        assert_eq!(pieces("tab\there"), ["tab", "\there"]);
        assert_eq!(pieces("!?\n\nnext"), ["!?\n\n", "next"]);
        assert_eq!(pieces("日本語 テキスト"), ["日本語", " テキスト"]);
        assert_eq!(pieces("'x"), ["'x"]);
        assert_eq!(pieces(" "), [" "]);
        assert_eq!(pieces(""), Vec::<String>::new());
    }

    #[test]
    fn encodes_with_merges_specials_and_the_template() {
        let t = Tokenizer::parse(tiny_json().as_bytes()).unwrap();
        // "hello" merges fully; " w" merges; "orld" stays bytes
        assert_eq!(t.encode("hello world", 100).unwrap(), vec![259, 260, u32::from(b'o'), u32::from(b'r'), u32::from(b'l'), u32::from(b'd'), 300]);
        // specials are cut out of the text and never split
        assert_eq!(t.encode("hello<|endoftext|>he", 100).unwrap(), vec![259, 300, 256, 300]);
        assert_eq!(t.encode("<|im_start|><|endoftext|>", 100).unwrap(), vec![301, 300, 300]);
        // truncation keeps the suffix
        let ids = t.encode("hello hello hello hello", 3).unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(ids.last(), Some(&300));
        assert_eq!(t.encode("", 10).unwrap(), vec![300]);
        // NFC: a decomposed é is encoded like the composed one
        assert_eq!(t.encode("e\u{301}", 10).unwrap(), t.encode("\u{e9}", 10).unwrap());
        assert!(t.encode(&"x".repeat(MAX_TEXT_BYTES + 1), 10).is_err());
        assert_eq!(t.id("<|endoftext|>"), Some(300));
    }

    #[test]
    fn hostile_text_never_panics() {
        let t = Tokenizer::parse(tiny_json().as_bytes()).unwrap();
        let samples = ["\u{0}\u{1}\u{7f}", "'", "''''", "<|", "|>", "<|endoftext", "\r\r\n\n \u{a0}\u{2028}", "🎵🎶", "\u{301}\u{301}", "ſ's"];
        for s in samples {
            let ids = t.encode(s, 64).unwrap();
            assert!(!ids.is_empty());
        }
        // a long word of one letter: the heap merge stays fast
        let long = "l".repeat(20_000);
        let ids = t.encode(&long, 100_000).unwrap();
        assert_eq!(ids.len(), 10_001);
    }

    #[test]
    fn other_tokenizers_are_refused() {
        let good: serde_json::Value = serde_json::from_str(&tiny_json()).unwrap();
        let edits: Vec<Box<dyn Fn(&mut serde_json::Value)>> = vec![
            Box::new(|v| v["model"]["type"] = serde_json::json!("Unigram")),
            Box::new(|v| v["model"]["byte_fallback"] = serde_json::json!(true)),
            Box::new(|v| v["model"]["continuing_subword_prefix"] = serde_json::json!("##")),
            Box::new(|v| v["normalizer"] = serde_json::json!({"type": "NFKC"})),
            Box::new(|v| v["pre_tokenizer"] = serde_json::Value::Null),
            Box::new(|v| v["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = serde_json::json!("\\s+")),
            Box::new(|v| v["post_processor"] = serde_json::json!({"type": "BertProcessing"})),
            Box::new(|v| v["added_tokens"][0]["lstrip"] = serde_json::json!(true)),
            Box::new(|v| v["model"]["merges"] = serde_json::json!([5])),
            Box::new(|v| v["model"]["vocab"] = serde_json::json!({"a": 0})),
            Box::new(|v| v["model"]["vocab"]["h"] = serde_json::json!(-1)),
        ];
        for (i, e) in edits.iter().enumerate() {
            let mut v = good.clone();
            e(&mut v);
            assert!(Tokenizer::parse(v.to_string().as_bytes()).is_err(), "edit {i}");
        }
        for junk in ["", "{", "[]", "null", "{\"model\":5}"] {
            assert!(Tokenizer::parse(junk.as_bytes()).is_err(), "{junk}");
        }
        // byte flips of a valid file: errors or a tokenizer, never a panic
        let bytes = tiny_json().into_bytes();
        for i in (0..bytes.len()).step_by(37) {
            let mut b = bytes.clone();
            b[i] ^= 0x21;
            std::panic::catch_unwind(|| {
                if let Ok(t) = Tokenizer::parse(&b) {
                    let _ = t.encode("hello world it's 42", 16);
                }
            })
            .expect("no panic");
        }
    }
}
