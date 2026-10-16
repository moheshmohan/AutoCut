//! VoxCPM's text tokenizer, read from the release's `tokenizer.json`: a SentencePiece-style BPE
//! (the MiniCPM-4 vocabulary) — the text gets a leading `▁` and every space becomes `▁`, there is
//! no pre-tokenizer (the whole text is one word), merges are applied lowest rank first (leftmost
//! on ties) and characters outside the vocabulary fall back to `<0xXX>` byte tokens. As in the
//! reference, no begin-of-sequence token is added, and a token made of two or more CJK ideographs
//! is split into its single characters (the models were trained that way).
//!
//! This is a minimal reimplementation of exactly that configuration; any other tokenizer.json
//! (another model type, a pre-tokenizer, other normalizers) is refused when loading.

use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

use super::{R, err};

/// Longest text tokenized at once, in bytes (callers chunk scripts into sentences first).
pub const MAX_TEXT_BYTES: usize = 16 * 1024;
const MAX_FILE_BYTES: u64 = 64 << 20;
const META: char = '\u{2581}';

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    /// `(left, right)` → `(rank, merged)`.
    merges: HashMap<(u32, u32), (u32, u32)>,
    bytes: [Option<u32>; 256],
    unk: u32,
    /// Control tokens (`<s>`, `<|im_end|>`…): removed from input text, never produced.
    specials: Vec<String>,
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
        if !v.get("pre_tokenizer").is_none_or(|p| p.is_null()) {
            return Err(err("tokenizer.json: pre-tokenizers are not supported"));
        }
        if model.get("continuing_subword_prefix").is_some_and(|p| !p.is_null()) || model.get("end_of_word_suffix").is_some_and(|p| !p.is_null()) {
            return Err(err("tokenizer.json: subword prefixes and suffixes are not supported"));
        }
        if model.get("byte_fallback").and_then(|b| b.as_bool()) != Some(true) {
            return Err(err("tokenizer.json: the BPE model must use byte fallback"));
        }
        check_normalizer(v.get("normalizer"))?;
        let table = model.get("vocab").and_then(|t| t.as_object()).ok_or_else(|| err("tokenizer.json: no vocabulary"))?;
        if table.len() > 2_000_000 {
            return Err(err("tokenizer.json: the vocabulary is too large"));
        }
        let mut vocab = HashMap::with_capacity(table.len());
        for (piece, id) in table {
            let id = id.as_u64().and_then(|i| u32::try_from(i).ok()).ok_or_else(|| err(format!("tokenizer.json: bad id for `{piece}`")))?;
            vocab.insert(piece.clone(), id);
        }
        let mut specials = Vec::new();
        if let Some(added) = v.get("added_tokens").and_then(|a| a.as_array()) {
            for a in added.iter().take(10_000) {
                if let (Some(content), Some(id)) = (a.get("content").and_then(|c| c.as_str()), a.get("id").and_then(|i| i.as_u64())) {
                    let id = u32::try_from(id).map_err(|_| err("tokenizer.json: bad added token id"))?;
                    vocab.entry(content.to_string()).or_insert(id);
                    if !content.is_empty() {
                        specials.push(content.to_string());
                    }
                }
            }
        }
        // longest first, so `<|im_end|>` is removed before a shorter special inside it
        specials.sort_by_key(|s| std::cmp::Reverse(s.len()));
        let unk_name = model.get("unk_token").and_then(|u| u.as_str()).unwrap_or("<unk>");
        let unk = vocab.get(unk_name).copied().ok_or_else(|| err("tokenizer.json: the unknown token is not in the vocabulary"))?;
        let list = model.get("merges").and_then(|m| m.as_array()).ok_or_else(|| err("tokenizer.json: no merges"))?;
        if list.len() > 4_000_000 {
            return Err(err("tokenizer.json: too many merges"));
        }
        let mut merges = HashMap::with_capacity(list.len());
        for (rank, m) in list.iter().enumerate() {
            // "a b" (older files) or ["a", "b"]
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
            merges.entry((*ia, *ib)).or_insert((rank as u32, *im));
        }
        let mut bytes = [None; 256];
        for (i, slot) in bytes.iter_mut().enumerate() {
            *slot = vocab.get(&format!("<0x{i:02X}>")).copied();
        }
        Ok(Tokenizer { vocab, merges, bytes, unk, specials })
    }

    /// The id of a vocabulary entry.
    pub fn id(&self, piece: &str) -> Option<u32> {
        self.vocab.get(piece).copied()
    }

    /// Token ids of `text` (no begin-of-sequence token). Control-token text is dropped first.
    pub fn encode(&self, text: &str) -> R<Vec<u32>> {
        if text.len() > MAX_TEXT_BYTES {
            return Err(err(format!("text to tokenize is longer than {} KB", MAX_TEXT_BYTES / 1024)));
        }
        let mut clean = text.to_string();
        for s in &self.specials {
            if clean.contains(s.as_str()) {
                clean = clean.replace(s.as_str(), "");
            }
        }
        if clean.is_empty() {
            return Ok(Vec::new());
        }
        let normalized: String = std::iter::once(META).chain(clean.chars().map(|c| if c == ' ' { META } else { c })).collect();
        let symbols = self.bpe(&normalized);
        let mut out = Vec::with_capacity(symbols.len());
        for (id, piece) in symbols {
            match piece {
                Some(p) if is_multi_cjk(&p) => {
                    out.extend(p.chars().filter(|c| *c != META).map(|c| self.vocab.get(c.encode_utf8(&mut [0; 4]) as &str).copied().unwrap_or(self.unk)))
                }
                _ => out.push(id),
            }
        }
        Ok(out)
    }

    /// BPE over the whole normalized text. Returns each token's id and, for tokens that may need
    /// the CJK split, its text.
    fn bpe(&self, s: &str) -> Vec<(u32, Option<String>)> {
        // symbols: (id, text span) in a doubly linked list over a vector
        struct Sym {
            id: u32,
            start: usize,
            end: usize,
            prev: Option<usize>,
            next: Option<usize>,
            alive: bool,
        }
        let mut syms: Vec<Sym> = Vec::with_capacity(s.len());
        let push = |syms: &mut Vec<Sym>, id: u32, start: usize, end: usize| {
            let i = syms.len();
            let prev = i.checked_sub(1);
            syms.push(Sym { id, start, end, prev, next: None, alive: true });
            if let Some(p) = prev
                && let Some(ps) = syms.get_mut(p)
            {
                ps.next = Some(i);
            }
        };
        let mut buf = [0u8; 4];
        for (pos, c) in s.char_indices() {
            let end = pos + c.len_utf8();
            if let Some(id) = self.vocab.get(c.encode_utf8(&mut buf) as &str) {
                push(&mut syms, *id, pos, end);
                continue;
            }
            let utf8 = c.encode_utf8(&mut buf);
            let ids: Option<Vec<u32>> = utf8.bytes().map(|b| self.bytes.get(usize::from(b)).copied().flatten()).collect();
            match ids {
                Some(ids) => {
                    for id in ids {
                        // byte tokens carry no text span of their own (never merged with text)
                        push(&mut syms, id, pos, pos);
                    }
                }
                None => {
                    // unknown, fused with a preceding unknown
                    if syms.last().is_some_and(|l| l.id == self.unk) {
                        if let Some(l) = syms.last_mut() {
                            l.end = end;
                        }
                    } else {
                        push(&mut syms, self.unk, pos, end);
                    }
                }
            }
        }
        // (rank, left index, left id, right id): min-heap by rank then position
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
            let (b_end, b_next) = (b.end, b.next);
            if let Some(a) = syms.get_mut(i) {
                a.id = merged;
                a.end = b_end;
                a.next = b_next;
            }
            if let Some(b) = syms.get_mut(j) {
                b.alive = false;
            }
            if let Some(n) = b_next
                && let Some(ns) = syms.get_mut(n)
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
        syms.iter()
            .filter(|x| x.alive)
            .map(|x| {
                let text = s.get(x.start..x.end).filter(|t| t.chars().any(is_cjk)).map(str::to_string);
                (x.id, text)
            })
            .collect()
    }
}

fn check_normalizer(n: Option<&serde_json::Value>) -> R<()> {
    // exactly: Sequence[Prepend("▁"), Replace(" " → "▁")]
    let ok = n.and_then(|n| n.get("normalizers")).and_then(|l| l.as_array()).is_some_and(|l| {
        l.len() == 2
            && l.first()
                .is_some_and(|p| p.get("type").and_then(|t| t.as_str()) == Some("Prepend") && p.get("prepend").and_then(|t| t.as_str()) == Some("\u{2581}"))
            && l.get(1).is_some_and(|r| {
                r.get("type").and_then(|t| t.as_str()) == Some("Replace")
                    && r.get("pattern").and_then(|p| p.get("String")).and_then(|t| t.as_str()) == Some(" ")
                    && r.get("content").and_then(|t| t.as_str()) == Some("\u{2581}")
            })
    });
    if ok { Ok(()) } else { Err(err("tokenizer.json: the normalizer is not the SentencePiece one this code implements")) }
}

fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

/// Two or more characters, all CJK ideographs, once `▁` is removed.
fn is_multi_cjk(piece: &str) -> bool {
    let mut n = 0;
    for c in piece.chars().filter(|c| *c != META) {
        if !is_cjk(c) {
            return false;
        }
        n += 1;
    }
    n >= 2
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A tiny tokenizer.json: letters, `▁`, a few merges, byte tokens for `é`'s bytes and two CJK
    /// characters plus their pair.
    pub(crate) fn tiny_json() -> String {
        let mut vocab: Vec<(String, u32)> = vec![("<unk>".into(), 0), ("<s>".into(), 1), ("</s>".into(), 2)];
        let mut next = 3u32;
        let mut add = |s: &str, vocab: &mut Vec<(String, u32)>| {
            vocab.push((s.to_string(), next));
            next += 1;
        };
        for c in "abcdefghijklmnopqrstuvwxyzHTW.,!?'".chars() {
            add(&c.to_string(), &mut vocab);
        }
        for s in [
            "\u{2581}",
            "<0xC3>",
            "<0xA9>",
            "你",
            "好",
            "你好",
            "\u{2581}h",
            "\u{2581}he",
            "ll",
            "llo",
            "\u{2581}hello",
            "\u{2581}w",
            "or",
            "\u{2581}wor",
            "ld",
            "\u{2581}world",
        ] {
            add(s, &mut vocab);
        }
        let merges = ["\u{2581} h", "\u{2581}h e", "l l", "ll o", "\u{2581}he llo", "\u{2581} w", "o r", "\u{2581}w or", "l d", "\u{2581}wor ld", "你 好"];
        let v: serde_json::Map<String, serde_json::Value> = vocab.into_iter().map(|(k, i)| (k, serde_json::json!(i))).collect();
        serde_json::json!({
            "version": "1.0",
            "added_tokens": [{"id": 0, "content": "<unk>", "special": true}, {"id": 1, "content": "<s>", "special": true}, {"id": 2, "content": "</s>", "special": true}],
            "normalizer": {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "\u{2581}"}, {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"}]},
            "pre_tokenizer": null,
            "model": {"type": "BPE", "unk_token": "<unk>", "fuse_unk": true, "byte_fallback": true, "vocab": v, "merges": merges}
        })
        .to_string()
    }

    #[test]
    fn merges_by_rank_and_falls_back_to_bytes() {
        let t = Tokenizer::parse(tiny_json().as_bytes()).unwrap();
        let id = |s: &str| t.id(s).unwrap();
        assert_eq!(t.encode("hello world").unwrap(), vec![id("\u{2581}hello"), id("\u{2581}world")]);
        assert_eq!(t.encode("hello").unwrap(), vec![id("\u{2581}hello")]);
        // é is not in the vocabulary: its UTF-8 bytes
        assert_eq!(t.encode("é").unwrap(), vec![id("\u{2581}"), id("<0xC3>"), id("<0xA9>")]);
        // ☃ has no byte tokens either: one fused unknown for the run
        assert_eq!(t.encode("☃☃").unwrap(), vec![id("\u{2581}"), 0]);
        // control tokens in the text are dropped
        assert_eq!(t.encode("<s>hello</s>").unwrap(), vec![id("\u{2581}hello")]);
        assert!(t.encode("").unwrap().is_empty());
        assert!(t.encode(&"a".repeat(MAX_TEXT_BYTES + 1)).is_err());
    }

    #[test]
    fn multi_character_cjk_tokens_are_split() {
        let t = Tokenizer::parse(tiny_json().as_bytes()).unwrap();
        let id = |s: &str| t.id(s).unwrap();
        // "▁" + "你好": merged to 你好 by BPE, then split back to single characters
        assert_eq!(t.encode("你好").unwrap(), vec![id("\u{2581}"), id("你"), id("好")]);
        assert!(is_multi_cjk("\u{2581}你好"));
        assert!(!is_multi_cjk("你"));
        assert!(!is_multi_cjk("你a"));
    }

    #[test]
    fn hostile_tokenizer_files_are_refused() {
        let good = tiny_json();
        for (from, to) in [
            ("\"BPE\"", "\"Unigram\""),
            ("\"pre_tokenizer\":null", "\"pre_tokenizer\":{\"type\":\"Whitespace\"}"),
            ("\"Prepend\"", "\"Lowercase\""),
            ("\"byte_fallback\":true", "\"byte_fallback\":false"),
            ("\"unk_token\":\"<unk>\"", "\"unk_token\":\"<nope>\""),
        ] {
            let bad = good.replacen(from, to, 1);
            assert_ne!(bad, good, "{from}");
            assert!(Tokenizer::parse(bad.as_bytes()).is_err(), "{from}");
        }
        for junk in ["", "{}", "[]", "{\"model\":{\"type\":\"BPE\"}}"] {
            assert!(Tokenizer::parse(junk.as_bytes()).is_err(), "{junk}");
        }
        // any byte flipped: an error or a tokenizer, never a panic
        let b = good.as_bytes();
        for i in (0..b.len()).step_by(7) {
            let mut x = b.to_vec();
            x[i] ^= 0x21;
            if let Ok(t) = Tokenizer::parse(&x) {
                let _ = t.encode("hello world 你好 é");
            }
        }
    }

    /// Against the reference tokenizer, gated: FILMCRAFT_VOXCPM_DIR=<a VoxCPM release>
    /// (ids from the Hugging Face `tokenizers` library for these sentences).
    #[test]
    fn live_matches_the_reference_ids() {
        let Some(dir) = std::env::var_os("FILMCRAFT_VOXCPM_DIR") else {
            eprintln!("skipped (FILMCRAFT_VOXCPM_DIR unset)");
            return;
        };
        let t = Tokenizer::load(&Path::new(&dir).join("tokenizer.json")).unwrap();
        // reference: tokenizers.Tokenizer.from_file(...).encode(text, add_special_tokens=False).ids,
        // with 你好 and 世界 split into 你 好 世 界 as VoxCPM's wrapper does
        let cases: &[(&str, &[u32])] = &[
            ("Hello, world!", &[21045, 59342, 2809, 73]),
            (
                "In October twenty twenty-two I met Spot, the yellow robot dog from Boston Dynamics.",
                &[1646, 5405, 17691, 17691, 63, 15438, 1417, 2510, 28609, 59342, 1358, 13548, 15898, 6595, 1525, 12923, 36317, 72],
            ),
            ("Naïve café — 你好世界 ☃ 😀", &[11367, 62949, 1412, 33903, 60025, 3934, 59320, 59496, 59495, 59792, 59868, 59320, 72997, 59320, 70571]),
            ("  double  spaces ", &[1345, 5371, 1345, 29786, 59320]),
            ("Go1 at GITEX in Dubai.", &[4395, 59345, 1510, 1527, 2268, 6979, 1377, 31755, 72]),
        ];
        for (text, want) in cases {
            assert_eq!(t.encode(text).unwrap(), *want, "{text}");
        }
    }
}
