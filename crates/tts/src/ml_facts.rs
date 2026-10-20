//! The fact lock of a Malayalam narration line: every number, name, unit and month of the
//! English line it says must be in the Malayalam, and no number may be added. A translation
//! (a person's or a model's) is checked against it before it is voiced.
//!
//! - **Numbers** are compared by value, however each side writes them: "six hundred", "600",
//!   "അറുന്നൂറ്", "സിക്സ് ഹണ്ട്രഡ്" ([`crate::ml_script::numbers`]). A year said in halves
//!   ("twenty twenty"), "two and a half thousand", digits inside codes ("M02") count too; "one"
//!   and "a" are words as often as numbers, so they are left out, and a Malayalam "ഒരു" is never
//!   "added". A decimal is kept when its parts are ("1.8 metres" as "ഒരു മീറ്റർ എൺപത്").
//! - **Names** (capitalised words; at a sentence's start only when the plan uses them as names
//!   elsewhere) are compared by sound: both sides become a skeleton of consonant classes
//!   ("Neolix" N-L-K-S, "നിയോലിക്സിന്റെ" N-L-K-S-N-T) and the English one must be in a window
//!   of up to three Malayalam words with at most one difference in four (two from seven).
//!   A plan's respellings ("GITEX" → "Jai-tex") give the sound to look for.
//! - **Acronyms** and single capitals are their letters' names ("UAE" യു എ ഇ), or, for one said
//!   as a word ("NEMO" നീമോ), its sound.
//! - **Units** (metres, kilograms, hours, dollars, percent, …) after a number, and **months**,
//!   by their Malayalam words.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

use std::collections::{BTreeMap, HashSet};

use serde::Serialize;

use crate::{malayalam, ml_script};

/// Longest line read (bytes).
const MAX_TEXT_BYTES: usize = 64 * 1024;
/// Most facts one line has.
const MAX_FACTS: usize = 200;
/// Most respellings used, and the longest word respelled.
const MAX_RESPELLINGS: usize = 2000;
const MAX_RESPELLED: usize = 200;

/// What a fact is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum FactKind {
    Number,
    Name,
    Letters,
    Unit,
    Month,
}

/// One fact of an English line, as the English writes it.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Fact {
    pub kind: FactKind,
    pub text: String,
    #[serde(skip)]
    want: Want,
}

#[derive(Clone, Debug, PartialEq)]
enum Want {
    Value(f64),
    Skeleton(Vec<String>),
    /// The letters' names (loose spelling), else the sound of the word (an acronym said as one).
    Letters(String, Vec<String>),
    Stems(&'static [&'static str]),
}

/// What a Malayalam line lost or added.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FactCheck {
    /// Facts of the English the Malayalam does not say.
    pub missing: Vec<Fact>,
    /// Numbers the Malayalam says that the English does not.
    pub added: Vec<String>,
}

impl FactCheck {
    pub fn is_clean(&self) -> bool {
        self.missing.is_empty() && self.added.is_empty()
    }

    /// One line for people: "missing: 2013, Kerala; added: 5".
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if !self.missing.is_empty() {
            parts.push(format!("missing: {}", self.missing.iter().map(|f| f.text.as_str()).collect::<Vec<_>>().join(", ")));
        }
        if !self.added.is_empty() {
            parts.push(format!("added: {}", self.added.join(", ")));
        }
        parts.join("; ")
    }
}

/// The fact lock of a plan: its respellings, and the words it uses as names.
#[derive(Clone, Debug, Default)]
pub struct FactLock {
    respell: Vec<(String, String)>,
    known: HashSet<String>,
}

fn clip(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text.get(..end).unwrap_or("")
}

// ---------------------------------------------------------------------------------------------
// English words

struct Word {
    text: String,
    /// It starts a sentence.
    initial: bool,
    /// It was written with a possessive ("Promobot's"): a name, even at a sentence's start.
    possessive: bool,
    start: usize,
    end: usize,
}

fn edge(c: char) -> bool {
    matches!(c, '.' | ',' | '?' | '!' | ':' | ';' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | '‘' | '’' | '“' | '”' | '—' | '–' | '…' | '-' | '«' | '»')
}

fn words(text: &str) -> Vec<Word> {
    let mut out = Vec::new();
    let mut initial = true;
    let push = |s: usize, e: usize, out: &mut Vec<Word>, initial: &mut bool| {
        let raw = text.get(s..e).unwrap_or("");
        let lead = raw.len() - raw.trim_start_matches(edge).len();
        let trimmed = raw.trim_matches(edge);
        let mut core = trimmed;
        for p in ["'s", "’s"] {
            if let Some(c) = core.strip_suffix(p) {
                core = c;
            }
        }
        let core = core.trim_end_matches(['\'', '’']);
        let bare = raw.trim_end_matches([',', '.', '!', '?', ';', ':']);
        let possessive = core.len() < trimmed.len() || bare.ends_with("s'") || bare.ends_with("s’");
        if !core.is_empty() {
            out.push(Word { text: core.to_string(), initial: *initial, possessive, start: s + lead, end: s + lead + core.len() });
        }
        *initial = raw.trim_end_matches(['"', '\'', '’', '”', ')', ']']).ends_with(['.', '!', '?']);
    };
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                push(s, i, &mut out, &mut initial);
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        push(s, text.len(), &mut out, &mut initial);
    }
    out
}

/// `text` with the respellings made (whole words, the longest first, never inside a word), and
/// where each respelling went (start, end, the word it replaced).
fn respell(text: &str, pairs: &[(String, String)]) -> (String, Vec<(usize, usize, String)>) {
    if pairs.is_empty() {
        return (text.to_string(), Vec::new());
    }
    let word = |c: char| c.is_alphanumeric();
    let mut out = String::with_capacity(text.len());
    let mut spans = Vec::new();
    let mut prev: Option<char> = None;
    let mut i = 0;
    while let Some(rest) = text.get(i..) {
        let Some(c) = rest.chars().next() else { break };
        let hit = pairs.iter().find(|(k, _)| {
            rest.starts_with(k.as_str())
                && !(k.starts_with(word) && prev.is_some_and(word))
                && !(k.ends_with(word) && rest.get(k.len()..).and_then(|r| r.chars().next()).is_some_and(word))
        });
        match hit {
            Some((k, v)) => {
                let s = out.len();
                out.push_str(v);
                spans.push((s, out.len(), k.clone()));
                prev = k.chars().next_back();
                i += k.len();
            }
            None => {
                out.push(c);
                prev = Some(c);
                i += c.len_utf8();
            }
        }
    }
    (out, spans)
}

/// Capitalised words that are not names.
const STOP: &[&str] = &[
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "it's",
    "i",
    "i'm",
    "i've",
    "i'd",
    "i'll",
    "my",
    "me",
    "we",
    "we're",
    "our",
    "us",
    "you",
    "you're",
    "you'd",
    "your",
    "they",
    "they're",
    "their",
    "them",
    "he",
    "she",
    "his",
    "her",
    "and",
    "but",
    "or",
    "so",
    "then",
    "there",
    "there's",
    "here",
    "here's",
    "next",
    "in",
    "on",
    "at",
    "by",
    "for",
    "from",
    "with",
    "of",
    "to",
    "as",
    "after",
    "before",
    "back",
    "not",
    "just",
    "still",
    "every",
    "each",
    "some",
    "most",
    "all",
    "when",
    "while",
    "where",
    "what",
    "which",
    "who",
    "how",
    "why",
    "thanks",
    "thank",
    "tell",
    "look",
    "over",
    "under",
    "about",
    "around",
    "than",
    "like",
    "also",
    "even",
    "only",
    "very",
    "yes",
    "no",
    "today",
    "later",
    "little",
    "big",
    "behind",
    "inside",
    "outside",
    "let's",
    "meet",
    "say",
    "once",
    "again",
    "first",
    "last",
    "new",
    "old",
    "if",
    "is",
    "was",
    "were",
    "be",
    "been",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "will",
    "has",
    "have",
    "had",
    "up",
    "down",
    "out",
    "off",
    "into",
    "onto",
    "near",
    "honestly",
    "finally",
    "lastly",
    "now",
    "well",
    "oh",
    "wow",
    "yeah",
    "okay",
    "ok",
    "hey",
    "hi",
    "hello",
    "welcome",
    "both",
    "many",
    "much",
    "more",
    "less",
    "same",
    "other",
    "another",
    "such",
    "one",
    "ones",
    "robots",
    "robot",
    "that's",
    "what's",
    "who's",
    "it'll",
    "don't",
    "didn't",
    "isn't",
    "wasn't",
    "can't",
    "won't",
    "during",
    "since",
    "until",
    "through",
    "across",
    "along",
    "between",
    "among",
    "without",
    "within",
    "right",
    "left",
    "inside",
    "everyone",
    "everything",
    "someone",
    "something",
    "nothing",
    "nobody",
    "always",
    "never",
    "sometimes",
    "maybe",
    "perhaps",
];

fn has_inner_capital(w: &str) -> bool {
    w.chars().skip(1).any(|c| c.is_ascii_uppercase()) && w.chars().any(|c| c.is_ascii_lowercase())
}

fn capitalised(w: &str) -> bool {
    w.chars().next().is_some_and(|c| c.is_uppercase())
}

// ---------------------------------------------------------------------------------------------
// English numbers

#[derive(Clone, Copy, Debug, PartialEq)]
enum Ew {
    Unit(u64),
    Ten(u64),
    Scale(u64),
    Half,
    A,
    And,
}

const EN_UNITS: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const EN_TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

fn ew(w: &str) -> Option<Ew> {
    let w = w.to_ascii_lowercase();
    if let Some(i) = EN_UNITS.iter().position(|u| *u == w) {
        return Some(Ew::Unit(i as u64));
    }
    if let Some(i) = EN_TENS.iter().position(|t| !t.is_empty() && *t == w) {
        return Some(Ew::Ten(i as u64 * 10));
    }
    Some(match w.as_str() {
        "hundred" | "hundreds" => Ew::Scale(100),
        "thousand" | "thousands" => Ew::Scale(1000),
        "lakh" | "lakhs" => Ew::Scale(100_000),
        "million" | "millions" => Ew::Scale(1_000_000),
        "crore" | "crores" => Ew::Scale(10_000_000),
        "billion" | "billions" => Ew::Scale(1_000_000_000),
        "half" => Ew::Half,
        "a" | "an" => Ew::A,
        "and" => Ew::And,
        _ => return None,
    })
}

/// The number words at the start of `ws`: the values and the words taken (None: no number).
fn number_run(ws: &[Word]) -> Option<(Vec<f64>, usize)> {
    let mut run: Vec<Ew> = Vec::new();
    let mut used = 0;
    for (k, w) in ws.iter().enumerate() {
        let mut parts: Vec<Ew> = Vec::new();
        let mut whole = true;
        for p in w.text.split('-') {
            match ew(p) {
                Some(e) => parts.push(e),
                None => {
                    whole = false;
                    break;
                }
            }
        }
        if parts.is_empty() {
            break;
        }
        // "a"/"an" only before a scale or "half", "and" only between number words
        let next = ws.get(k + 1).and_then(|n| ew(n.text.split('-').next().unwrap_or("")));
        if let [only] = parts.as_slice() {
            match only {
                Ew::A if !matches!(next, Some(Ew::Scale(_)) | Some(Ew::Half)) => break,
                Ew::And if run.is_empty() || !matches!(next, Some(Ew::Unit(_) | Ew::Ten(_) | Ew::A | Ew::Half)) => break,
                _ => {}
            }
        }
        run.extend(parts);
        used = k + 1;
        // "six-axis": the number ends inside the word
        if !whole {
            break;
        }
    }
    if used == 0 {
        return None;
    }
    let numeric = run.iter().filter(|e| matches!(e, Ew::Unit(_) | Ew::Ten(_) | Ew::Scale(_) | Ew::Half)).count();
    // "one" or "a" alone is as often a word as a number
    if numeric == 0 || (numeric == 1 && run.iter().any(|e| matches!(e, Ew::Unit(1))) && !run.iter().any(|e| matches!(e, Ew::Scale(_) | Ew::Half))) {
        return Some((Vec::new(), used));
    }
    Some((values(&run), used))
}

fn values(run: &[Ew]) -> Vec<f64> {
    let (mut total, mut cur) = (0f64, 0f64);
    let mut groups: Vec<f64> = Vec::new();
    let mut scaled = false;
    let mut last: Option<Ew> = None;
    for e in run {
        match *e {
            Ew::Unit(u) => {
                let u = u as f64;
                if cur == 0.0 || (matches!(last, Some(Ew::Ten(_))) && u < 10.0) || matches!(last, Some(Ew::Scale(_)) | Some(Ew::And)) {
                    cur += u;
                } else {
                    groups.push(cur);
                    cur = u;
                }
            }
            Ew::Ten(t) => {
                if cur % 100.0 == 0.0 {
                    cur += t as f64;
                } else {
                    groups.push(cur);
                    cur = t as f64;
                }
            }
            Ew::Scale(100) => {
                cur = cur.max(1.0) * 100.0;
                scaled = true;
            }
            Ew::Scale(s) => {
                total += cur.max(1.0) * s as f64;
                cur = 0.0;
                scaled = true;
            }
            Ew::Half => cur += 0.5,
            Ew::A => {}
            Ew::And => {}
        }
        if *e != Ew::A {
            last = Some(*e);
        }
    }
    if !groups.is_empty() && !scaled {
        if let [hi] = groups.as_slice()
            && (10.0..100.0).contains(hi)
            && cur < 100.0
        {
            return vec![hi * 100.0 + cur];
        }
        let mut all = groups;
        all.push(cur);
        return all;
    }
    vec![total + cur]
}

// ---------------------------------------------------------------------------------------------
// Units and months

struct Unit {
    id: &'static str,
    words: &'static [&'static str],
    /// Short forms, only right after digits ("20 km", "20km").
    abbr: &'static [&'static str],
    ml: &'static [&'static str],
}

const UNITS: &[Unit] = &[
    Unit {
        id: "kilometres", words: &["kilometre", "kilometres", "kilometer", "kilometers"], abbr: &["km"], ml: &["കിലോമീറ്റ", "കി.മീ"]
    },
    Unit {
        id: "centimetres", words: &["centimetre", "centimetres", "centimeter", "centimeters"], abbr: &["cm"], ml: &["സെന്റിമീറ്റ", "സെന്റീമീറ്റ", "സെ.മീ"]
    },
    Unit {
        id: "millimetres", words: &["millimetre", "millimetres", "millimeter", "millimeters"], abbr: &["mm"], ml: &["മില്ലിമീറ്റ", "മി.മീ"]
    },
    Unit { id: "metres", words: &["metre", "metres", "meter", "meters"], abbr: &["m"], ml: &["മീറ്റ"] },
    Unit { id: "kilograms", words: &["kilogram", "kilograms", "kilo", "kilos"], abbr: &["kg"], ml: &["കിലോ"] },
    Unit { id: "grams", words: &["gram", "grams"], abbr: &["g"], ml: &["ഗ്രാം", "ഗ്രാമ"] },
    Unit { id: "tonnes", words: &["tonne", "tonnes", "ton", "tons"], abbr: &["t"], ml: &["ടൺ", "ടണ്ണ"] },
    Unit { id: "hours", words: &["hour", "hours"], abbr: &["h", "hr", "hrs"], ml: &["മണിക്കൂ", "അവർ"] },
    Unit {
        id: "minutes", words: &["minute", "minutes"], abbr: &["min", "mins"], ml: &["മിനിറ്റ", "മിനുട്ട", "മിനിട്ട"]
    },
    Unit {
        id: "seconds", words: &["second", "seconds"], abbr: &["s", "sec", "secs"], ml: &["സെക്കന്റ", "സെക്കൻഡ", "സെക്കൻറ", "സെക്കണ്ട"]
    },
    Unit { id: "degrees", words: &["degree", "degrees"], abbr: &["°", "°c"], ml: &["ഡിഗ്രി"] },
    Unit {
        id: "percent", words: &["percent", "per cent"], abbr: &["%"], ml: &["ശതമാന", "പേഴ്സന്റ", "പെർസന്റ", "പെർസെന്റ", "%"]
    },
    Unit { id: "dollars", words: &["dollar", "dollars"], abbr: &["$"], ml: &["ഡോളർ", "ഡോളറ", "$"] },
    Unit { id: "rupees", words: &["rupee", "rupees"], abbr: &["₹"], ml: &["രൂപ", "₹"] },
    Unit { id: "dirhams", words: &["dirham", "dirhams"], abbr: &["aed"], ml: &["ദിർഹ"] },
    Unit { id: "euros", words: &["euro", "euros"], abbr: &["€"], ml: &["യൂറോ", "€"] },
    Unit { id: "pounds", words: &["pound", "pounds"], abbr: &["£"], ml: &["പൗണ്ട", "£"] },
    Unit { id: "gallons", words: &["gallon", "gallons"], abbr: &["gal"], ml: &["ഗാലൻ", "ഗാലണ"] },
    Unit { id: "litres", words: &["litre", "litres", "liter", "liters"], abbr: &["l"], ml: &["ലിറ്റ"] },
    Unit { id: "miles", words: &["mile", "miles"], abbr: &["mi"], ml: &["മൈൽ", "മൈല"] },
    Unit { id: "inches", words: &["inch", "inches"], abbr: &["in"], ml: &["ഇഞ്ച"] },
    Unit { id: "feet", words: &["foot", "feet"], abbr: &["ft"], ml: &["അടി", "ഫീറ്റ", "ഫൂട്ട"] },
    Unit { id: "years", words: &["year", "years"], abbr: &[], ml: &["വർഷ", "കൊല്ല", "ഇയർ"] },
    Unit { id: "months", words: &["month", "months"], abbr: &[], ml: &["മാസ", "മന്ത"] },
    Unit { id: "weeks", words: &["week", "weeks"], abbr: &[], ml: &["ആഴ്ച", "വീക്ക"] },
    Unit { id: "days", words: &["day", "days"], abbr: &[], ml: &["ദിവസ", "ദിന", "ഡേ"] },
    Unit { id: "square", words: &["square"], abbr: &["sq"], ml: &["സ്ക്വയർ", "ചതുരശ്ര"] },
    Unit { id: "volts", words: &["volt", "volts"], abbr: &["v"], ml: &["വോൾട്ട"] },
    Unit { id: "watts", words: &["watt", "watts"], abbr: &["w"], ml: &["വാട്ട"] },
];

fn unit_word(w: &str) -> Option<&'static Unit> {
    let w = w.to_ascii_lowercase();
    UNITS.iter().find(|u| u.words.contains(&w.as_str()))
}

fn unit_abbr(w: &str) -> Option<&'static Unit> {
    let w = w.to_lowercase();
    UNITS.iter().find(|u| u.abbr.contains(&w.as_str()))
}

const MONTHS: [(&str, &[&str]); 12] = [
    ("January", &["ജനുവരി"]),
    ("February", &["ഫെബ്രുവരി"]),
    ("March", &["മാർച്ച"]),
    ("April", &["ഏപ്രി"]),
    ("May", &["മേയ", "മെയ"]),
    ("June", &["ജൂൺ", "ജൂണ"]),
    ("July", &["ജൂലൈ"]),
    ("August", &["ഓഗസ്റ്റ", "ആഗസ്റ്റ"]),
    ("September", &["സെപ്റ്റംബ"]),
    ("October", &["ഒക്ടോബ"]),
    ("November", &["നവംബ"]),
    ("December", &["ഡിസംബ"]),
];

// ---------------------------------------------------------------------------------------------
// Sounds

/// The consonant classes of an English word, in every likely reading (at most a dozen).
fn skeleton_en(word: &str) -> Vec<String> {
    let s: Vec<char> = word.chars().filter(char::is_ascii_alphabetic).map(|c| c.to_ascii_lowercase()).collect();
    let vowel = |c: Option<&char>| c.is_some_and(|c| "aeiou".contains(*c));
    let soft = |c: Option<&char>| c.is_some_and(|c| "eiy".contains(*c));
    // each position: its alternatives (a string of classes each)
    let mut pos: Vec<Vec<&'static str>> = Vec::new();
    let mut i = 0;
    while let Some(&c) = s.get(i) {
        let nx = s.get(i + 1);
        let two = nx.map(|n| [c, *n]);
        let digraph: Option<Vec<&'static str>> = match two {
            Some(['c', 'h']) => Some(vec!["C", "K"]),
            Some(['s', 'h']) | Some(['z', 'h']) => Some(vec!["S"]),
            Some(['t', 'h']) | Some(['d', 'h']) => Some(vec!["T"]),
            Some(['p', 'h']) => Some(vec!["F"]),
            Some(['c', 'k']) | Some(['k', 'h']) => Some(vec!["K"]),
            Some(['n', 'g']) if soft(s.get(i + 2)) => Some(vec!["N", "NK", "NC"]),
            Some(['n', 'g']) => Some(vec!["N", "NK"]),
            Some(['q', 'u']) => Some(vec!["KV", "K"]),
            Some(['g', 'h']) => Some(vec!["", "K"]),
            Some(['w', 'h']) => Some(vec!["V"]),
            Some(['b', 'h']) => Some(vec!["P"]),
            Some(['j', 'h']) => Some(vec!["C"]),
            _ => None,
        };
        if let Some(d) = digraph {
            pos.push(d);
            i += 2;
            continue;
        }
        let one: Vec<&'static str> = match c {
            'b' | 'p' => vec!["P"],
            'c' if soft(nx) => vec!["S"],
            'c' | 'k' | 'q' => vec!["K"],
            'd' => vec!["T"],
            't' if nx == Some(&'u') => vec!["T", "C"],
            't' => vec!["T"],
            'f' => vec!["F"],
            'v' | 'w' => vec!["V"],
            'g' if soft(nx) => vec!["K", "C"],
            'g' => vec!["K"],
            'j' => vec!["C"],
            'x' => vec!["KS"],
            'l' => vec!["L"],
            'm' => vec!["M"],
            'n' => vec!["N"],
            // an "r" before a consonant or at the end is often not said ("arm" ആം)
            'r' if vowel(nx) => vec!["R"],
            'r' => vec!["R", ""],
            's' | 'z' => vec!["S"],
            'h' if vowel(nx) => vec!["H"],
            _ => vec![""],
        };
        pos.push(one);
        i += 1;
    }
    let mut variants: Vec<String> = vec![String::new()];
    for alts in pos {
        let mut next = Vec::new();
        for v in &variants {
            for a in &alts {
                if next.len() < 12 {
                    next.push(format!("{v}{a}"));
                }
            }
        }
        variants = next;
    }
    let mut out: Vec<String> = Vec::new();
    for v in variants {
        let c = collapse(&v);
        if !c.is_empty() && !out.contains(&c) {
            out.push(c);
        }
    }
    out
}

fn collapse(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last: Option<char> = None;
    for c in s.chars() {
        if last != Some(c) {
            out.push(c);
        }
        last = Some(c);
    }
    out
}

/// The consonant classes of a Malayalam word; `~` is the anusvara (ം), an m or an n. A doubled
/// "റ്റ" and the "റ" of "ന്റ" are said as a t.
fn skeleton_ml(word: &str) -> String {
    let mut out = String::new();
    let chars: Vec<char> = word.chars().collect();
    for (i, c) in chars.iter().copied().enumerate() {
        let before = |k: usize| i.checked_sub(k).and_then(|j| chars.get(j)).copied();
        if c == 'റ' && before(1) == Some('\u{0D4D}') && matches!(before(2), Some('റ') | Some('ന')) {
            if before(2) == Some('റ') && out.ends_with('R') {
                out.pop();
            }
            out.push('T');
            continue;
        }
        if c == 'റ' && before(1) == Some('ൻ') {
            out.push('T');
            continue;
        }
        let class = match c {
            'ക' | 'ഖ' | 'ഗ' | 'ഘ' | 'ൿ' => "K",
            'ങ' | 'ഞ' | 'ണ' | 'ന' | 'ൻ' | 'ൺ' => "N",
            'ച' | 'ഛ' | 'ജ' | 'ഝ' => "C",
            'ട' | 'ഠ' | 'ഡ' | 'ഢ' | 'ത' | 'ഥ' | 'ദ' | 'ധ' => "T",
            'പ' | 'ബ' | 'ഭ' => "P",
            'ഫ' => "F",
            'മ' => "M",
            'ര' | 'റ' | 'ർ' => "R",
            'ല' | 'ള' | 'ഴ' | 'ൽ' | 'ൾ' => "L",
            'വ' => "V",
            'ശ' | 'ഷ' | 'സ' => "S",
            'ഹ' => "H",
            'ം' => "~",
            c if c.is_ascii_alphabetic() => {
                out.push_str(skeleton_en(&c.to_string()).first().map_or("", String::as_str));
                continue;
            }
            _ => "",
        };
        out.push_str(class);
    }
    collapse(&out)
}

fn same(a: char, b: char) -> bool {
    a == b || (a == '~' && matches!(b, 'M' | 'N')) || (b == '~' && matches!(a, 'M' | 'N'))
}

/// `needle` is in `hay` with at most `edits` differences (an alignment that may start and end
/// anywhere in `hay`).
fn approx_contains(hay: &str, needle: &str, edits: usize) -> bool {
    let n: Vec<char> = needle.chars().collect();
    let h: Vec<char> = hay.chars().collect();
    if n.is_empty() {
        return true;
    }
    let mut prev: Vec<usize> = vec![0; h.len() + 1];
    for (i, nc) in n.iter().enumerate() {
        let mut cur = vec![i + 1; h.len() + 1];
        for (j, hc) in h.iter().enumerate() {
            let cost = usize::from(!same(*nc, *hc));
            let diag = prev.get(j).copied().unwrap_or(usize::MAX / 2) + cost;
            let up = prev.get(j + 1).copied().unwrap_or(usize::MAX / 2) + 1;
            let left = cur.get(j).copied().unwrap_or(usize::MAX / 2) + 1;
            if let Some(slot) = cur.get_mut(j + 1) {
                *slot = diag.min(up).min(left);
            }
        }
        prev = cur;
    }
    prev.iter().copied().min().unwrap_or(usize::MAX) <= edits
}

fn allowed_edits(len: usize) -> usize {
    match len {
        0..=3 => 0,
        4..=6 => 1,
        _ => 2,
    }
}

/// A variant of a name's sound is in a window of up to three words.
fn sound_found(skel: &[String], variants: &[String]) -> bool {
    variants.iter().filter(|v| !v.is_empty()).any(|v| {
        let edits = allowed_edits(v.chars().count());
        (0..skel.len()).any(|i| {
            let mut window = String::new();
            (i..skel.len().min(i + 3)).any(|j| {
                window.push_str(skel.get(j).map_or("", String::as_str));
                approx_contains(&window, v, edits)
            })
        })
    })
}

/// A run of capitals spelled with the letters' names, loosely ("UAE" → "യുഎഇ").
fn letters_loose(letters: &str) -> String {
    ml_script::loose(&letters.chars().filter_map(ml_script::letter_name).collect::<String>())
}

// ---------------------------------------------------------------------------------------------
// The lock

impl FactLock {
    /// The lock of a plan: its respellings (`pronounce`: what the voice says for a written word,
    /// "GITEX" → "Jai-tex"), its English `texts` (lines, intro, outro) and entry `labels`, whose
    /// capitalised words are names wherever they appear.
    pub fn new(pronounce: &BTreeMap<String, String>, texts: &[&str], labels: &[&str]) -> Self {
        let mut respell: Vec<(String, String)> = pronounce
            .iter()
            .filter(|(k, v)| !k.trim().is_empty() && k.len() <= MAX_RESPELLED && v.len() <= MAX_RESPELLED)
            .take(MAX_RESPELLINGS)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        respell.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(&b.0)));
        let mut known = HashSet::new();
        for t in texts {
            let (r, _) = respell_with(clip(t, MAX_TEXT_BYTES), &respell);
            for w in words(&r) {
                if capitalised(&w.text) && (!w.initial || has_inner_capital(&w.text)) {
                    known.insert(w.text.to_lowercase());
                }
            }
        }
        for l in labels {
            let (r, _) = respell_with(clip(l, MAX_TEXT_BYTES), &respell);
            let ws = words(&r);
            let second_capital = ws.get(1).is_some_and(|w| capitalised(&w.text));
            for (i, w) in ws.iter().enumerate() {
                if capitalised(&w.text) && (i > 0 || ws.len() == 1 || second_capital) {
                    known.insert(w.text.to_lowercase());
                }
            }
        }
        Self { respell, known }
    }

    /// The facts of an English line, in order.
    pub fn facts(&self, english: &str) -> Vec<Fact> {
        let (text, spans) = respell_with(clip(english, MAX_TEXT_BYTES), &self.respell);
        let ws = words(&text);
        let shown = |w: &Word| spans.iter().find(|(s, e, _)| w.start >= *s && w.end <= *e).map_or_else(|| w.text.clone(), |(_, _, k)| k.clone());
        let mut out: Vec<Fact> = Vec::new();
        let push = |f: Fact, out: &mut Vec<Fact>| {
            let dup = out.iter().any(|o| o.want == f.want || (o.kind == f.kind && o.kind != FactKind::Number && o.text.eq_ignore_ascii_case(&f.text)));
            if !dup && out.len() < MAX_FACTS {
                out.push(f);
            }
        };
        let numberish = |w: Option<&Word>| {
            w.is_some_and(|w| {
                w.text.chars().any(|c| c.is_ascii_digit())
                    || ew(w.text.split('-').next_back().unwrap_or("")).is_some()
                    || ["per", "an", "a"].contains(&w.text.to_ascii_lowercase().as_str())
            })
        };
        let mut i = 0;
        while let Some(w) = ws.get(i) {
            let text_shown = shown(w);
            if w.text.chars().any(|c| c.is_ascii_digit()) {
                for f in digit_facts(&w.text, &text_shown) {
                    push(f, &mut out);
                }
                i += 1;
                continue;
            }
            if let Some((vals, used)) = number_run(ws.get(i..).unwrap_or(&[])) {
                let last = ws.get(i + used - 1).map_or(w.end, |l| l.end);
                let respelled = spans.iter().any(|(s, e, _)| w.start >= *s && w.end <= *e);
                let said = if respelled { text_shown.clone() } else { text.get(w.start..last.max(w.start)).unwrap_or("").to_string() };
                for v in vals {
                    push(Fact { kind: FactKind::Number, text: said.clone(), want: Want::Value(v) }, &mut out);
                }
                i += used;
                continue;
            }
            if let Some((m, stems)) = MONTHS.iter().find(|(m, _)| *m == w.text)
                && (*m != "May" || numberish(ws.get(i + 1)) || numberish(i.checked_sub(1).and_then(|p| ws.get(p))))
            {
                push(Fact { kind: FactKind::Month, text: m.to_string(), want: Want::Stems(stems) }, &mut out);
                i += 1;
                continue;
            }
            let lw = w.text.to_ascii_lowercase();
            if let Some(u) = unit_word(&lw)
                && numberish(i.checked_sub(1).and_then(|p| ws.get(p)))
            {
                push(Fact { kind: FactKind::Unit, text: w.text.clone(), want: Want::Stems(u.ml) }, &mut out);
                i += 1;
                continue;
            }
            if !capitalised(&w.text) && !has_inner_capital(&w.text) {
                i += 1;
                continue;
            }
            // single capitals in a row: letters ("Dilly S", "M C U"); "I" and an opening "A" are words
            let single =
                |w: &Word| w.text.chars().count() == 1 && w.text.chars().all(|c| c.is_ascii_uppercase()) && w.text != "I" && !(w.text == "A" && w.initial);
            if single(w) {
                let mut j = i;
                let mut letters = String::new();
                while let Some(x) = ws.get(j).filter(|x| single(x)) {
                    letters.push_str(&x.text);
                    j += 1;
                }
                // a respelling's letters show as the word they spell ("Node M C U": NodeMCU)
                let label = if text_shown != w.text { text_shown } else { letters.clone() };
                push(Fact { kind: FactKind::Letters, text: label, want: Want::Letters(letters_loose(&letters), Vec::new()) }, &mut out);
                i = j;
                continue;
            }
            let letters_only: String = w.text.chars().filter(|c| *c != '-' && *c != '.').collect();
            let all_caps = letters_only.chars().count() >= 2 && letters_only.chars().all(|c| c.is_ascii_uppercase());
            // a respelt word is a name; at a sentence's start a word is a name when the plan uses
            // it as one elsewhere, writes it with a possessive or in capitals
            let respelt = text_shown != w.text;
            let named = respelt || !w.initial || self.known.contains(&lw) || has_inner_capital(&w.text) || all_caps || w.possessive;
            if (STOP.contains(&lw.as_str()) && !respelt) || !named {
                i += 1;
                continue;
            }
            if letters_only.chars().count() >= 2 && letters_only.chars().count() <= 8 && letters_only.chars().all(|c| c.is_ascii_uppercase()) {
                push(
                    Fact { kind: FactKind::Letters, text: text_shown, want: Want::Letters(letters_loose(&letters_only), skeleton_en(&letters_only)) },
                    &mut out,
                );
            } else {
                let v = skeleton_en(&w.text);
                if !v.is_empty() {
                    push(Fact { kind: FactKind::Name, text: text_shown, want: Want::Skeleton(v) }, &mut out);
                }
            }
            i += 1;
        }
        out
    }

    /// What `malayalam` loses or adds of `english`'s facts.
    pub fn check(&self, english: &str, malayalam: &str) -> FactCheck {
        let facts = self.facts(english);
        let ml = malayalam::canonical(clip(malayalam, MAX_TEXT_BYTES));
        let mlw: Vec<String> = ml.split_whitespace().map(|w| w.trim_matches(edge).to_string()).filter(|w| !w.is_empty()).collect();
        let skel: Vec<String> = mlw.iter().map(|w| skeleton_ml(w)).collect();
        let loose_all = ml_script::loose(&ml);
        let nums: Vec<f64> = ml_script::numbers(&ml).into_iter().map(|n| n.value).collect();
        let has = |list: &[f64], v: f64| list.iter().any(|x| (x - v).abs() < 1e-6);
        let english_values: Vec<f64> = facts.iter().filter_map(|f| if let Want::Value(v) = f.want { Some(v) } else { None }).collect();
        let mut covered: Vec<f64> = Vec::new();
        let mut missing = Vec::new();
        for f in facts {
            // a name left in Latin letters is still there (the lint says to write it in Malayalam)
            let in_latin = f.text.chars().any(|c| c.is_ascii_alphabetic()) && ml.contains(f.text.as_str());
            let ok = match &f.want {
                Want::Value(v) => has(&nums, *v) || decimal_parts(*v, &nums, &mut covered),
                Want::Skeleton(vars) => in_latin || sound_found(&skel, vars),
                Want::Letters(spelled, vars) => {
                    in_latin || (!spelled.is_empty() && loose_all.contains(spelled.as_str())) || (!vars.is_empty() && sound_found(&skel, vars))
                }
                Want::Stems(stems) => stems.iter().any(|s| ml.contains(s)),
            };
            if !ok {
                missing.push(f);
            }
        }
        let mut added: Vec<String> = Vec::new();
        for v in nums {
            if v != 1.0 && v != 0.0 && !has(&english_values, v) && !has(&covered, v) {
                let s = ml_script::digits_of(v);
                if !added.contains(&s) {
                    added.push(s);
                }
            }
        }
        FactCheck { missing, added }
    }
}

fn respell_with(text: &str, pairs: &[(String, String)]) -> (String, Vec<(usize, usize, String)>) {
    respell(text, pairs)
}

/// A decimal said as its parts ("1.8 metres" as "ഒരു മീറ്റർ എൺപത് സെന്റിമീറ്റർ").
fn decimal_parts(v: f64, nums: &[f64], covered: &mut Vec<f64>) -> bool {
    if v.fract() == 0.0 {
        return false;
    }
    let s = format!("{v}");
    let Some((int, frac)) = s.split_once('.') else { return false };
    let (Ok(int), Ok(f1)) = (int.parse::<f64>(), frac.parse::<f64>()) else { return false };
    let f2 = format!("{frac:0<2}").parse::<f64>().unwrap_or(f1);
    let has = |x: f64| nums.iter().any(|n| (n - x).abs() < 1e-6);
    let part = [f1, f2].into_iter().find(|x| has(*x));
    match part {
        Some(p) if has(int) || int == 0.0 || int == 1.0 => {
            covered.extend([int, p]);
            true
        }
        _ => false,
    }
}

/// The facts of a word with digits: its numbers, a currency, percent or unit with them, the
/// letters or name of a code ("M02", "4K", "Go1", "V.4", "$1,600", "85%", "20km").
fn digit_facts(token: &str, shown: &str) -> Vec<Fact> {
    let mut out = Vec::new();
    let mut chars = token.chars().peekable();
    let mut groups: Vec<(bool, String)> = Vec::new(); // (digits, text)
    let mut symbols: Vec<char> = Vec::new();
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() || ((c == ',' || c == '.') && groups.last().is_some_and(|(d, _)| *d) && chars.peek().is_some_and(char::is_ascii_digit)) {
            match groups.last_mut() {
                Some((true, g)) => g.push(c),
                _ => groups.push((true, c.to_string())),
            }
        } else if c.is_alphabetic() {
            match groups.last_mut() {
                Some((false, g)) => g.push(c),
                _ => groups.push((false, c.to_string())),
            }
        } else {
            symbols.push(c);
            // a separator ends a group
            groups.push((false, String::new()));
        }
    }
    let groups: Vec<(bool, String)> = groups.into_iter().filter(|(_, g)| !g.is_empty()).collect();
    for s in symbols {
        if let Some(u) = unit_abbr(&s.to_string()) {
            out.push(Fact { kind: FactKind::Unit, text: u.id.to_string(), want: Want::Stems(u.ml) });
        }
    }
    let mut k = 0;
    while let Some((digits, g)) = groups.get(k) {
        if *digits {
            let mut v: f64 = g.replace(',', "").parse().unwrap_or(f64::NAN);
            // "10k": thousands
            if let Some((false, next)) = groups.get(k + 1)
                && next == "k"
            {
                v *= 1000.0;
                k += 1;
            }
            if v.is_finite() {
                out.push(Fact { kind: FactKind::Number, text: shown.to_string(), want: Want::Value(v) });
            }
        } else {
            let lower = g.to_lowercase();
            if ["st", "nd", "rd", "th"].contains(&lower.as_str()) && k > 0 {
                // an ordinal: the number is the fact
            } else if let Some(u) = unit_abbr(&lower).filter(|_| k > 0 && g.chars().any(char::is_lowercase)) {
                out.push(Fact { kind: FactKind::Unit, text: u.id.to_string(), want: Want::Stems(u.ml) });
            } else if g.chars().all(|c| c.is_ascii_uppercase()) {
                out.push(Fact { kind: FactKind::Letters, text: shown.to_string(), want: Want::Letters(letters_loose(g), Vec::new()) });
            } else {
                let v = skeleton_en(g);
                if !v.is_empty() {
                    out.push(Fact { kind: FactKind::Name, text: shown.to_string(), want: Want::Skeleton(v) });
                }
            }
        }
        k += 1;
    }
    out
}

/// What `malayalam` loses or adds of `english`'s facts, with no plan to go by.
pub fn check(english: &str, malayalam: &str) -> FactCheck {
    FactLock::default().check(english, malayalam)
}

#[cfg(test)]
#[path = "ml_facts_tests.rs"]
mod tests;
