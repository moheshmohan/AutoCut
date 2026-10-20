//! Natural spoken Malayalam for narration scripts: a lint that finds what makes a Malayalam
//! script sound written (read out, it sounds robotic) and says what a Malayali tech creator
//! would say instead, and the numbers a script says.
//!
//! The style is `docs/malayalam-style.md`. [`lint`] checks each sentence for:
//!
//! - [`Kind::LongSentence`]: more than [`DEFAULT_MAX_WORDS`] spoken words. A number said the
//!   English way ("ട്വന്റി ട്വന്റി വൺ") or an acronym spelled letter by letter ("ആർ ടി എ")
//!   counts as one word, as it is said in one breath.
//! - [`Kind::Formal`]: a formal or literary word with a common spoken form, from a curated table
//!   (സ്വാഗതം ചെയ്യുന്നു → വെൽക്കം ചെയ്യുന്നു, സന്ദർശകർ → വിസിറ്റേഴ്സ്, വികസിപ്പിക്കാൻ →
//!   ഡെവലപ്പ് ചെയ്യാൻ, ലഭിക്കും → കിട്ടും, -ക്കായി → -ക്ക് വേണ്ടി). The table is conservative:
//!   a word is in it only when the spoken form is what people say in a tech video, and the
//!   weaker cases are hints.
//! - [`Kind::NumberWords`]: a number from 100 up in Malayalam words ("അറുന്നൂറിലധികം"): write
//!   digits ("600-ലധികം"), which the voice says the English way ([`crate::malayalam`]).
//! - [`Kind::Latin`]: Latin letters. An acronym or a code is spelled with the letters' names
//!   ("UAE" → "യു എ ഇ"); any other word is written in Malayalam script as it sounds.
//! - [`Kind::NoVerb`] (a hint): a sentence with no verb ending at all. A deliberate fragment
//!   ("പന്ത്രണ്ട് വർഷം, നാൽപ്പത്തിയെട്ട് റോബോട്ടുകൾ.") is fine; an accidental one reads as a list.
//!
//! Every finding says what to say instead; [`fix`] applies the replacements that keep the
//! meaning and the word's suffix exactly (the safe part of the table, digits for number words,
//! letter names for acronyms). [`numbers`] reads every number a script says (digits, Malayalam
//! words, or English words in Malayalam script), for the fact checks of [`crate::ml_facts`].
//!
//! Hostile input is fine: a text over [`MAX_TEXT_BYTES`] is read up to there, findings stop at
//! [`MAX_FINDINGS`], and nothing indexes or slices outside a char boundary.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

use std::sync::OnceLock;

use serde::Serialize;

use crate::malayalam::{self, Numbers};

/// Spoken words a sentence may have before it is too long to say naturally.
pub const DEFAULT_MAX_WORDS: usize = 14;
/// The range `max_words` may be set to.
pub const MIN_MAX_WORDS: usize = 4;
pub const MAX_MAX_WORDS: usize = 60;
/// Longest text read (bytes); the rest is ignored.
pub const MAX_TEXT_BYTES: usize = 1 << 20;
/// Most findings one text gets.
pub const MAX_FINDINGS: usize = 2000;
/// Most words one number may span.
const MAX_NUMBER_WORDS: usize = 8;

/// What a finding is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Kind {
    LongSentence,
    Formal,
    NumberWords,
    Latin,
    NoVerb,
}

impl Kind {
    pub const ALL: [Kind; 5] = [Kind::LongSentence, Kind::Formal, Kind::NumberWords, Kind::Latin, Kind::NoVerb];

    /// The id reports use (`longSentence`, `formal`, `numberWords`, `latin`, `noVerb`).
    pub fn id(self) -> &'static str {
        match self {
            Kind::LongSentence => "longSentence",
            Kind::Formal => "formal",
            Kind::NumberWords => "numberWords",
            Kind::Latin => "latin",
            Kind::NoVerb => "noVerb",
        }
    }
}

/// How much a finding matters: warnings make a script sound written (`--strict-script` fails on
/// them); hints are worth a look.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Level {
    Hint,
    Warning,
}

/// One thing in a script that a Malayali creator would say differently.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Finding {
    pub kind: Kind,
    pub level: Level,
    /// The sentence it is in (from 0).
    pub sentence: usize,
    /// The words found, as written (a whole sentence for [`Kind::LongSentence`] and
    /// [`Kind::NoVerb`]).
    pub text: String,
    pub message: String,
    /// What to say instead.
    pub suggestion: String,
    /// A safe replacement of `text`: same meaning, same suffix ([`fix`] applies it).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
    /// Where `text` is in the text linted (bytes).
    #[serde(skip)]
    pub span: (usize, usize),
}

/// Lint settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// Spoken words a sentence may have ([`MIN_MAX_WORDS`]..=[`MAX_MAX_WORDS`]).
    pub max_words: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { max_words: DEFAULT_MAX_WORDS }
    }
}

/// `text` cut to [`MAX_TEXT_BYTES`] at a char boundary.
fn clip(text: &str) -> &str {
    if text.len() <= MAX_TEXT_BYTES {
        return text;
    }
    let mut end = MAX_TEXT_BYTES;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text.get(..end).unwrap_or("")
}

// ---------------------------------------------------------------------------------------------
// Words and sentences

/// A word of a text, and its core: the word without the punctuation around it (bytes).
#[derive(Clone, Copy, Debug)]
struct Tok {
    start: usize,
    end: usize,
    cs: usize,
    ce: usize,
}

fn edge_punct(c: char) -> bool {
    matches!(c, '.' | ',' | '?' | '!' | ':' | ';' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | '‘' | '’' | '“' | '”' | '—' | '–' | '…' | '।' | '«' | '»')
}

impl Tok {
    fn core<'a>(&self, text: &'a str) -> &'a str {
        text.get(self.cs..self.ce).unwrap_or("")
    }

    /// The text after the core (its trailing punctuation).
    fn tail<'a>(&self, text: &'a str) -> &'a str {
        text.get(self.ce..self.end).unwrap_or("")
    }

    /// The word ends a sentence.
    fn ends_sentence(&self, text: &str) -> bool {
        let t = self.tail(text).trim_end_matches(['"', '\'', '’', '”', ')', ']', '»']);
        t.ends_with(['.', '?', '!', '।', '…'])
    }

    /// Punctuation follows the word (a clause ends).
    fn ends_clause(&self, text: &str) -> bool {
        !self.tail(text).is_empty()
    }
}

fn tokens(text: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    let push = |s: usize, e: usize, out: &mut Vec<Tok>| {
        let w = text.get(s..e).unwrap_or("");
        let lead = w.len() - w.trim_start_matches(edge_punct).len();
        let trail = w.len() - w.trim_end_matches(edge_punct).len();
        let cs = s + lead;
        let ce = e.saturating_sub(trail).max(cs);
        out.push(Tok { start: s, end: e, cs, ce });
    };
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                push(s, i, &mut out);
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        push(s, text.len(), &mut out);
    }
    out
}

/// Sentences as token ranges (first, last), inclusive.
fn sentence_ranges(text: &str, toks: &[Tok]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut first = 0;
    for (i, t) in toks.iter().enumerate() {
        if t.ends_sentence(text) {
            out.push((first, i));
            first = i + 1;
        }
    }
    if first < toks.len() {
        out.push((first, toks.len() - 1));
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Malayalam letters

const VIRAMA: char = '\u{0D4D}';

fn is_vowel_sign(c: char) -> bool {
    ('\u{0D3E}'..='\u{0D4C}').contains(&c) || c == '\u{0D57}' || c == '\u{0D62}' || c == '\u{0D63}'
}

fn is_consonant(c: char) -> bool {
    ('ക'..='ഹ').contains(&c)
}

/// The vowel sign an independent vowel takes after a consonant (`അ`: none).
fn vowel_sign(c: char) -> Option<&'static str> {
    Some(match c {
        'അ' => "",
        'ആ' => "ാ",
        'ഇ' => "ി",
        'ഈ' => "ീ",
        'ഉ' => "ു",
        'ഊ' => "ൂ",
        'ഋ' => "ൃ",
        'എ' => "െ",
        'ഏ' => "േ",
        'ഐ' => "ൈ",
        'ഒ' => "ൊ",
        'ഓ' => "ോ",
        'ഔ' => "ൌ",
        _ => return None,
    })
}

/// The consonant a chillu stands for.
fn chillu_base(c: char) -> Option<char> {
    Some(match c {
        'ൻ' => 'ന',
        'ൺ' => 'ണ',
        'ർ' => 'റ',
        'ൽ' => 'ല',
        'ൾ' => 'ള',
        'ൿ' => 'ക',
        _ => return None,
    })
}

/// The Latin letters' names in Malayalam script.
pub const LETTERS: [&str; 26] = [
    "എ",
    "ബി",
    "സി",
    "ഡി",
    "ഇ",
    "എഫ്",
    "ജി",
    "എച്ച്",
    "ഐ",
    "ജെ",
    "കെ",
    "എൽ",
    "എം",
    "എൻ",
    "ഒ",
    "പി",
    "ക്യൂ",
    "ആർ",
    "എസ്",
    "ടി",
    "യു",
    "വി",
    "ഡബ്ല്യു",
    "എക്സ്",
    "വൈ",
    "സെഡ്",
];

/// The name of Latin letter `c` (any case) in Malayalam script.
pub fn letter_name(c: char) -> Option<&'static str> {
    let c = c.to_ascii_uppercase();
    if !c.is_ascii_uppercase() {
        return None;
    }
    LETTERS.get(usize::from(c as u8 - b'A')).copied()
}

/// English number names in Malayalam script, 0–19, and the tens.
pub const EN_UNITS: [&str; 20] = [
    "സീറോ",
    "വൺ",
    "ടു",
    "ത്രീ",
    "ഫോർ",
    "ഫൈവ്",
    "സിക്സ്",
    "സെവൻ",
    "എയ്റ്റ്",
    "നൈൻ",
    "ടെൻ",
    "ഇലവൻ",
    "ട്വെൽവ്",
    "തേർട്ടീൻ",
    "ഫോർട്ടീൻ",
    "ഫിഫ്റ്റീൻ",
    "സിക്സ്റ്റീൻ",
    "സെവന്റീൻ",
    "എയ്റ്റീൻ",
    "നൈന്റീൻ",
];
pub const EN_TENS: [&str; 10] = ["", "", "ട്വന്റി", "തേർട്ടി", "ഫോർട്ടി", "ഫിഫ്റ്റി", "സിക്സ്റ്റി", "സെവന്റി", "എയ്റ്റി", "നൈന്റി"];

/// `d` (0–9) said in English in Malayalam script.
pub fn english_digit(d: u32) -> &'static str {
    EN_UNITS.get(d as usize).copied().unwrap_or("")
}

/// `text` with long vowels made short, independent vowels' long forms short and everything but
/// letters dropped: spellings of one sound compare equal ("ക്യൂ" and "ക്യു", "എ ഐ" and "എഐ").
pub fn loose(text: &str) -> String {
    malayalam::canonical(text)
        .chars()
        .filter_map(|c| match c {
            'ൂ' => Some('ു'),
            'ീ' => Some('ി'),
            'േ' => Some('െ'),
            'ോ' => Some('ൊ'),
            'ഈ' => Some('ഇ'),
            'ഊ' => Some('ഉ'),
            'ഏ' => Some('എ'),
            'ഓ' => Some('ഒ'),
            c if c.is_alphanumeric() || is_vowel_sign(c) || c == VIRAMA || ('\u{0D00}'..='\u{0D7F}').contains(&c) => Some(c),
            _ => None,
        })
        .collect()
}

/// The stem a suffix attaches to: a final virama dropped, a final chillu its consonant.
fn stem_keys(w: &str) -> Vec<String> {
    let mut chars: Vec<char> = w.chars().collect();
    match chars.last().copied() {
        Some(VIRAMA) => {
            chars.pop();
            vec![chars.into_iter().collect()]
        }
        Some(c) => match chillu_base(c) {
            Some(base) => {
                chars.pop();
                let stem: String = chars.into_iter().collect();
                let mut keys = vec![format!("{stem}{base}")];
                if c == 'ർ' {
                    keys.push(format!("{stem}ര"));
                }
                keys
            }
            None => vec![w.to_string()],
        },
        None => Vec::new(),
    }
}

/// `rest` can be a suffix after the stem `key`: a vowel sign, a glide, a hyphen, the stem's
/// last consonant doubled ("വൺ" → "വണ്ണിന്റെ", never "വണ്ടി"), or after a vowel a case ending.
fn suffix_start(rest: &str, key: &str) -> bool {
    let mut it = rest.chars();
    let Some(first) = it.next() else { return true };
    let last = key.chars().last();
    if is_vowel_sign(first) || first == 'യ' || first == 'വ' || first == '-' {
        return true;
    }
    if first == VIRAMA {
        return it.next().is_some_and(|c| Some(c) == last && is_consonant(c));
    }
    let vowel_final = last.is_some_and(|c| is_vowel_sign(c) || vowel_sign(c).is_some());
    vowel_final && (rest.starts_with("ക്ക") || first == 'ല' || first == 'ന' || first == 'ം' || first == 'ൽ')
}

// ---------------------------------------------------------------------------------------------
// English number words in Malayalam script

#[derive(Clone, Copy, Debug, PartialEq)]
enum En {
    Unit(u64),
    Ten(u64),
    Scale(u64),
    Point,
    Oh,
}

/// (stem key, the word, what it is), longest keys first.
fn en_table() -> &'static [(String, String, En)] {
    static TABLE: OnceLock<Vec<(String, String, En)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut words: Vec<(&str, En)> = Vec::new();
        for (i, w) in EN_UNITS.iter().enumerate() {
            words.push((w, En::Unit(i as u64)));
        }
        for (i, w) in EN_TENS.iter().enumerate().filter(|(_, w)| !w.is_empty()) {
            words.push((w, En::Ten(i as u64 * 10)));
        }
        words.extend([
            ("ഹണ്ട്രഡ്", En::Scale(100)),
            ("തൗസന്റ്", En::Scale(1000)),
            ("തൗസൻഡ്", En::Scale(1000)),
            ("ലാഖ്", En::Scale(100_000)),
            ("ക്രോർ", En::Scale(10_000_000)),
            ("മില്യൺ", En::Scale(1_000_000)),
            ("ബില്യൺ", En::Scale(1_000_000_000)),
            ("പോയിന്റ്", En::Point),
            // spellings people use
            ("ത്രി", En::Unit(3)),
            ("തെർട്ടീൻ", En::Unit(13)),
            ("തെർട്ടി", En::Ten(30)),
        ]);
        let mut out: Vec<(String, String, En)> = Vec::new();
        for (w, e) in words {
            for k in stem_keys(w) {
                out.push((k, w.to_string(), e));
            }
        }
        out.push(("ഓ".into(), "ഓ".into(), En::Oh));
        out.sort_by_key(|e| std::cmp::Reverse(e.0.len()));
        out
    })
}

/// The English number word `core` is (canonical Malayalam), and whether a suffix follows it.
fn en_word(core: &str) -> Option<(En, bool)> {
    if core.is_empty() {
        return None;
    }
    let table = en_table();
    if let Some((_, _, e)) = table.iter().find(|(_, w, _)| w == core) {
        return Some((*e, false));
    }
    for (key, _, e) in table {
        if *e == En::Oh {
            continue;
        }
        if let Some(rest) = core.strip_prefix(key.as_str())
            && !rest.is_empty()
            && suffix_start(rest, key)
        {
            return Some((*e, true));
        }
    }
    None
}

/// The values a run of English number words says: one number, or a year said in two halves
/// ("ട്വന്റി തേർട്ടീൻ" 2013, "നൈന്റീൻ സിക്സ്റ്റി നൈൻ" 1969), digits one by one ("സീറോ ടു"),
/// a decimal ("ടു പോയിന്റ് ഫൈവ്").
fn en_values(run: &[En]) -> Vec<f64> {
    if run.is_empty() {
        return Vec::new();
    }
    if let Some(p) = run.iter().position(|e| *e == En::Point) {
        let int = en_values(run.get(..p).unwrap_or(&[])).last().copied().unwrap_or(0.0);
        let digits: String =
            run.get(p + 1..).unwrap_or(&[]).iter().map_while(|e| if let En::Unit(u) = e { (*u < 10).then(|| u.to_string()) } else { None }).collect();
        let frac = format!("0.{digits}").parse::<f64>().unwrap_or(0.0);
        return vec![int + frac];
    }
    // digits one by one: units below ten only, more than one
    if run.len() > 1 && run.iter().all(|e| matches!(e, En::Unit(u) if *u < 10)) {
        let digits: String = run.iter().filter_map(|e| if let En::Unit(u) = e { Some(u.to_string()) } else { None }).collect();
        return digits.parse::<f64>().map(|v| vec![v]).unwrap_or_default();
    }
    let (mut total, mut cur) = (0u64, 0u64);
    let mut groups: Vec<u64> = Vec::new();
    let mut scaled = false;
    let mut last: Option<En> = None;
    for e in run {
        match *e {
            En::Unit(u) => {
                let joins = cur == 0 || (matches!(last, Some(En::Ten(_))) && u < 10) || matches!(last, Some(En::Scale(_)) | Some(En::Oh));
                if joins {
                    cur = cur.saturating_add(u);
                } else {
                    groups.push(cur);
                    cur = u;
                }
            }
            En::Ten(t) => {
                if cur.is_multiple_of(100) {
                    cur = cur.saturating_add(t);
                } else {
                    groups.push(cur);
                    cur = t;
                }
            }
            En::Oh => {
                groups.push(cur);
                cur = 0;
            }
            En::Scale(100) => {
                cur = cur.max(1).saturating_mul(100);
                scaled = true;
            }
            En::Scale(s) => {
                total = total.saturating_add(cur.max(1).saturating_mul(s));
                cur = 0;
                scaled = true;
            }
            En::Point => {}
        }
        last = Some(*e);
    }
    if !groups.is_empty() && !scaled {
        // a year said in halves ("ട്വന്റി ട്വന്റി ഫൈവ്")
        if let [hi] = groups.as_slice()
            && (10..=99).contains(hi)
            && cur < 100
        {
            return vec![(hi * 100 + cur) as f64];
        }
        let mut all: Vec<f64> = groups.iter().map(|g| *g as f64).collect();
        all.push(cur as f64);
        return all;
    }
    vec![total.saturating_add(cur) as f64]
}

// ---------------------------------------------------------------------------------------------
// Malayalam number words

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Atom {
    /// 1–99.
    Small(u64),
    /// 100–900.
    Hundred(u64),
    /// ആയിരം on its own (or after a number word: "രണ്ട് ആയിരം").
    Thousand,
    /// A stem only "-ായിരം" follows: പതി(നായിരം) 10, മൂവ(ായിരം) 3, അയ്യ(ായിരം) 5, എണ്ണ(ായിരം) 8.
    ThousandStem(u64),
    Lakh,
    Crore,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    Plain,
    /// After a joining "ി": the vowel of a vowel-initial word glides ("യേഴ്" for ഏഴ്).
    Glided,
    /// After a geminate (നൂറ്റ്, ആയിരത്ത്): the initial vowel joins it ("നൂറ്റമ്പത്").
    Elided,
}

/// How a number word joins the next one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Join {
    None,
    /// The final consonant doubled and "ി" ("ഇരുപത്" → "ഇരുപത്തി", "നൂറ്" → "നൂറ്റി").
    Geminate(char),
    /// "ത്തി" ("ആയിരം" → "ആയിരത്തി", "ലക്ഷം" → "ലക്ഷത്തി").
    Tti,
}

struct AtomForm {
    surface: String,
    atom: Atom,
    form: Form,
    join: Join,
    /// Its surface ends in "ത" (a "-തിനായിരം" can follow).
    ends_ta: bool,
}

const ML_SMALL: [(&str, u64); 32] = [
    ("ഒന്ന", 1),
    ("രണ്ട", 2),
    ("മൂന്ന", 3),
    ("നാല", 4),
    ("അഞ്ച", 5),
    ("ആറ", 6),
    ("ഏഴ", 7),
    ("എട്ട", 8),
    ("ഒൻപത", 9),
    ("ഒമ്പത", 9),
    ("പത്ത", 10),
    ("പതിനൊന്ന", 11),
    ("പന്ത്രണ്ട", 12),
    ("പതിമൂന്ന", 13),
    ("പതിനാല", 14),
    ("പതിനഞ്ച", 15),
    ("പതിനാറ", 16),
    ("പതിനേഴ", 17),
    ("പതിനെട്ട", 18),
    ("പത്തൊൻപത", 19),
    ("പത്തൊമ്പത", 19),
    ("ഇരുപത", 20),
    ("മുപ്പത", 30),
    ("നാൽപ്പത", 40),
    ("നാല്പത", 40),
    ("നാൽപത", 40),
    ("അമ്പത", 50),
    ("അൻപത", 50),
    ("അറുപത", 60),
    ("എഴുപത", 70),
    ("എൺപത", 80),
    ("തൊണ്ണൂറ", 90),
];

const ML_HUNDREDS: [(&str, u64); 13] = [
    ("നൂറ", 100),
    ("ഇരുനൂറ", 200),
    ("ഇരുന്നൂറ", 200),
    ("മുന്നൂറ", 300),
    ("നാനൂറ", 400),
    ("നാന്നൂറ", 400),
    ("അഞ്ഞൂറ", 500),
    ("അറുനൂറ", 600),
    ("അറുന്നൂറ", 600),
    ("എഴുനൂറ", 700),
    ("എഴുന്നൂറ", 700),
    ("എണ്ണൂറ", 800),
    ("തൊള്ളായിര", 900),
];

fn atoms() -> &'static [AtomForm] {
    static ATOMS: OnceLock<Vec<AtomForm>> = OnceLock::new();
    ATOMS.get_or_init(|| {
        let mut out: Vec<AtomForm> = Vec::new();
        let add = |surface: &str, atom: Atom, join: Join, out: &mut Vec<AtomForm>| {
            let ends_ta = surface.ends_with('ത');
            out.push(AtomForm { surface: surface.to_string(), atom, form: Form::Plain, join, ends_ta });
            let mut chars = surface.chars();
            if let Some(first) = chars.next()
                && let Some(sign) = vowel_sign(first)
            {
                let rest: String = chars.collect();
                out.push(AtomForm { surface: format!("യ{sign}{rest}"), atom, form: Form::Glided, join, ends_ta });
                out.push(AtomForm { surface: format!("{sign}{rest}"), atom, form: Form::Elided, join, ends_ta });
            }
        };
        for (s, v) in ML_SMALL {
            let join = if v >= 20 {
                match s.chars().last() {
                    Some('റ') => Join::Geminate('റ'),
                    _ => Join::Geminate('ത'),
                }
            } else {
                Join::None
            };
            add(s, Atom::Small(v), join, &mut out);
        }
        for (s, v) in ML_HUNDREDS {
            let join = if v == 900 { Join::Tti } else { Join::Geminate('റ') };
            add(s, Atom::Hundred(v), join, &mut out);
        }
        add("ആയിര", Atom::Thousand, Join::Tti, &mut out);
        for (s, v) in [("പതിന", 10), ("മൂവ", 3), ("അയ്യ", 5), ("എണ്ണ", 8)] {
            out.push(AtomForm { surface: s.to_string(), atom: Atom::ThousandStem(v), form: Form::Plain, join: Join::None, ends_ta: false });
        }
        out.push(AtomForm { surface: "ലക്ഷ".into(), atom: Atom::Lakh, form: Form::Plain, join: Join::Tti, ends_ta: false });
        out.push(AtomForm { surface: "കോടി".into(), atom: Atom::Crore, form: Form::Plain, join: Join::None, ends_ta: false });
        out.sort_by_key(|a| std::cmp::Reverse(a.surface.len()));
        out
    })
}

/// What may come at a position of a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Want {
    /// The start of a number.
    Start,
    /// After a joining "ി": plain or glided.
    Joined,
    /// After a geminate: an elided form.
    Elided,
    /// A new word after a whole number word: a multiplier only (ലക്ഷം, കോടി, ആയിരം).
    Multiplier,
    /// A new word after a multiplier word (കോടി, ലക്ഷം, ആയിരം): any number word.
    AfterMultiplier,
}

fn atom_at(s: &str, want: Want) -> Option<&'static AtomForm> {
    atoms().iter().find(|a| {
        let form_ok = match want {
            Want::Start | Want::AfterMultiplier => a.form == Form::Plain,
            Want::Joined => a.form != Form::Elided,
            Want::Elided => a.form == Form::Elided,
            Want::Multiplier => a.form == Form::Plain && matches!(a.atom, Atom::Thousand | Atom::Lakh | Atom::Crore),
        };
        form_ok && s.starts_with(a.surface.as_str())
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MlState {
    total: u64,
    cur: u64,
}

impl MlState {
    fn value(self) -> u64 {
        self.total.saturating_add(self.cur)
    }
}

/// How a parsed number word ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum End {
    /// With a joining form: the number goes on in the next word ("നൂറ്റി ഒന്ന്").
    Joining,
    /// A whole number word: the next word may multiply it ("രണ്ട് ലക്ഷം").
    Whole { multiplier: bool },
    /// A suffix follows ("അറുന്നൂറിലധികം"): the number ends in this word.
    Suffix,
}

/// Add `atom` to the state.
fn apply(st: &mut MlState, atom: Atom) {
    match atom {
        Atom::Small(v) | Atom::Hundred(v) => st.cur = st.cur.saturating_add(v),
        Atom::ThousandStem(v) => st.cur = st.cur.saturating_add(v),
        Atom::Thousand => {
            st.total = st.total.saturating_add(st.cur.max(1).saturating_mul(1000));
            st.cur = 0;
        }
        Atom::Lakh => {
            st.total = st.total.saturating_add(st.cur.max(1).saturating_mul(100_000));
            st.cur = 0;
        }
        Atom::Crore => {
            st.total = st.total.saturating_add(st.cur.max(1)).saturating_mul(10_000_000);
            st.cur = 0;
        }
    }
}

/// The number words at the start of `w` (canonical, one word), going on from `st` with `want`:
/// the bytes of `w` they take and how they end.
fn ml_word(w: &str, mut want: Want, st: &mut MlState) -> Option<(usize, End)> {
    let mut pos = 0usize;
    // the end of the number so far, and the state there (a link that leads nowhere is undone)
    let mut good: Option<(usize, MlState, End)> = None;
    let mut multiplier_last;
    loop {
        let rest = w.get(pos..)?;
        let Some(a) = atom_at(rest, want) else { break };
        let mut next = *st;
        let mut p = pos + a.surface.len();
        let mut join = a.join;
        match a.atom {
            Atom::Small(_) | Atom::ThousandStem(_) => {
                apply(&mut next, a.atom);
                let after = w.get(p..).unwrap_or("");
                let thousand = if after.starts_with("ായിര") {
                    Some("ായിര".len())
                } else if a.ends_ta && after.starts_with("ിനായിര") {
                    Some("ിനായിര".len())
                } else {
                    None
                };
                let thousand = thousand.filter(|l| ends_a_big_number(w.get(p + l..).unwrap_or("")));
                match thousand {
                    Some(l) => {
                        apply(&mut next, Atom::Thousand);
                        p += l;
                        join = Join::Tti;
                        multiplier_last = true;
                    }
                    // a stem of a thousand alone is no number ("എണ്ണ")
                    None if matches!(a.atom, Atom::ThousandStem(_)) => break,
                    None => multiplier_last = false,
                }
            }
            atom => {
                // "ആയിരുന്നു" (was), "ലക്ഷ്യം" (aim): words that start like ആയിരം or ലക്ഷം
                let big = matches!(atom, Atom::Thousand | Atom::Lakh | Atom::Hundred(900));
                if big && !ends_a_big_number(w.get(p..).unwrap_or("")) {
                    break;
                }
                apply(&mut next, atom);
                multiplier_last = matches!(atom, Atom::Thousand | Atom::Lakh | Atom::Crore);
            }
        }
        *st = next;
        pos = p;
        let after = w.get(pos..).unwrap_or("");
        if after.is_empty() || after == "്" || after == "ം" {
            return Some((w.len(), End::Whole { multiplier: multiplier_last }));
        }
        good = Some((pos, *st, End::Suffix));
        let (link, elide): (String, String) = match join {
            Join::Geminate(c) => (format!("{VIRAMA}{c}ി"), format!("{VIRAMA}{c}")),
            Join::Tti => ("ത്തി".to_string(), "ത്ത".to_string()),
            Join::None => break,
        };
        if let Some(r) = after.strip_prefix(link.as_str()) {
            if r.is_empty() {
                return Some((w.len(), End::Joining));
            }
            pos += link.len();
            // a doubled consonant after the joining "ി" ("നൂറ്റിപ്പത്ത്")
            let mut cs = r.chars();
            if let (Some(c1), Some(VIRAMA), Some(c2)) = (cs.next(), cs.next(), cs.next())
                && c1 == c2
                && is_consonant(c1)
            {
                pos += c1.len_utf8() + VIRAMA.len_utf8();
            }
            want = Want::Joined;
            continue;
        }
        if let Some(r) = after.strip_prefix(elide.as_str())
            && r.chars().next().is_some_and(|c| c != VIRAMA)
        {
            pos += elide.len();
            want = Want::Elided;
            continue;
        }
        break;
    }
    let (end, state, how) = good?;
    *st = state;
    Some((end, how))
}

/// What may follow "ആയിര", "ലക്ഷ" or "തൊള്ളായിര" in a number word: its "ം", its joining
/// "ത്ത(ി)", a plural "ങ്ങൾ", or nothing.
fn ends_a_big_number(after: &str) -> bool {
    after.is_empty() || after.starts_with('ം') || after.starts_with("ത്ത") || after.starts_with("ങ്ങ")
}

/// Words that start like a number but are not one.
fn not_a_number(w: &str) -> bool {
    w.starts_with("നൂറ്റാണ്ട") || w.contains("കണക്ക")
}

/// A number in Malayalam words at the start of `cores`: its value, the words it spans and the
/// bytes of the last of them that belong to it.
fn ml_run(cores: &[&str]) -> Option<(u64, usize, usize)> {
    let mut st = MlState::default();
    let mut want = Want::Start;
    let (mut used, mut last_bytes) = (0usize, 0usize);
    for (k, w) in cores.iter().enumerate().take(MAX_NUMBER_WORDS) {
        if not_a_number(w) {
            break;
        }
        // "ഒരു" (one, a) only starts a number or follows a multiplier ("ഒരു കോടി ഒരു ലക്ഷം")
        if *w == "ഒരു" && matches!(want, Want::Start | Want::AfterMultiplier) {
            st.cur = st.cur.saturating_add(1);
            used = k + 1;
            last_bytes = w.len();
            want = Want::Multiplier;
            continue;
        }
        let mut trial = st;
        let Some((bytes, end)) = ml_word(w, want, &mut trial) else { break };
        st = trial;
        used = k + 1;
        last_bytes = bytes;
        match end {
            End::Joining => want = Want::Joined,
            End::Whole { multiplier: true } => want = Want::AfterMultiplier,
            End::Whole { multiplier: false } => want = Want::Multiplier,
            End::Suffix => break,
        }
    }
    (used > 0).then(|| (st.value(), used, last_bytes))
}

// ---------------------------------------------------------------------------------------------
// Numbers a script says

/// How a number is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Said {
    Digits,
    /// Malayalam words ("അറുന്നൂറ്").
    Malayalam,
    /// English words in Malayalam script ("സിക്സ് ഹണ്ട്രഡ്").
    English,
}

/// A number a script says.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Number {
    pub value: f64,
    /// The words or digits, as written.
    pub text: String,
    pub said: Said,
}

/// A number found among the words of a text: its value, the tokens it spans and the suffix of
/// its last word (canonical).
struct Hit {
    value: f64,
    first: usize,
    last: usize,
    said: Said,
    suffix: String,
    /// For digits: the bytes of the number within the core of `first` (canonical core).
    digits: Option<(usize, usize)>,
}

/// Numbers written in digits within `core` (canonical): value and byte range.
fn digit_numbers(core: &str) -> Vec<(f64, usize, usize)> {
    let chars: Vec<(usize, char)> = core.char_indices().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(&(start, c)) = chars.get(i) {
        if !c.is_ascii_digit() {
            i += 1;
            continue;
        }
        let mut j = i;
        let mut dot = false;
        while let Some(&(_, d)) = chars.get(j) {
            let next_digit = chars.get(j + 1).is_some_and(|(_, n)| n.is_ascii_digit());
            if d.is_ascii_digit() || (d == ',' && next_digit) || (d == '.' && !dot && next_digit) {
                dot |= d == '.';
                j += 1;
            } else {
                break;
            }
        }
        let end = chars.get(j).map_or(core.len(), |(b, _)| *b);
        let digits: String = core.get(start..end).unwrap_or("").chars().filter(|c| *c != ',').collect();
        if let Ok(v) = digits.parse::<f64>()
            && v.is_finite()
        {
            out.push((v, start, end));
        }
        i = j.max(i + 1);
    }
    out
}

/// Every number among `cores` (canonical cores of `toks` in `text`).
fn scan_numbers(text: &str, toks: &[Tok], cores: &[String]) -> Vec<Hit> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(core) = cores.get(i) {
        if core.chars().any(|c| c.is_ascii_digit()) {
            for (v, s, e) in digit_numbers(core) {
                out.push(Hit { value: v, first: i, last: i, said: Said::Digits, suffix: String::new(), digits: Some((s, e)) });
            }
            i += 1;
            continue;
        }
        if en_word(core).is_some() {
            let mut run: Vec<En> = Vec::new();
            let mut j = i;
            while let Some((e, suffixed)) = cores.get(j).and_then(|c| en_word(c)) {
                // "ഓ" only inside a year ("നൈന്റീൻ ഓ ഫൈവ്")
                if e == En::Oh && (run.is_empty() || !cores.get(j + 1).and_then(|c| en_word(c)).is_some_and(|(n, _)| matches!(n, En::Unit(_)))) {
                    break;
                }
                run.push(e);
                j += 1;
                if suffixed || toks.get(j - 1).is_some_and(|t| t.ends_clause(text)) {
                    break;
                }
            }
            if run.is_empty() {
                i += 1;
                continue;
            }
            // a lone "പോയിന്റ്" or "ഓ" is a word, not a number
            if !(run.len() == 1 && matches!(run.first(), Some(En::Point | En::Oh))) {
                for v in en_values(&run) {
                    out.push(Hit { value: v, first: i, last: j - 1, said: Said::English, suffix: String::new(), digits: None });
                }
            }
            i = j;
            continue;
        }
        // Malayalam words, never across a comma
        let mut stop = i;
        while stop < cores.len() && stop < i + MAX_NUMBER_WORDS {
            stop += 1;
            if toks.get(stop - 1).is_some_and(|t| t.ends_clause(text)) {
                break;
            }
        }
        let words: Vec<&str> = cores.get(i..stop).unwrap_or(&[]).iter().map(String::as_str).collect();
        match ml_run(&words) {
            Some((v, used, bytes)) => {
                let last = i + used - 1;
                let suffix = cores.get(last).and_then(|c| c.get(bytes..)).unwrap_or("").to_string();
                out.push(Hit { value: v as f64, first: i, last, said: Said::Malayalam, suffix, digits: None });
                i += used;
            }
            None => i += 1,
        }
    }
    out
}

/// Every number `text` says: digits, Malayalam words, English words in Malayalam script.
pub fn numbers(text: &str) -> Vec<Number> {
    let text = clip(text);
    let toks = tokens(text);
    let cores: Vec<String> = toks.iter().map(|t| malayalam::canonical(t.core(text))).collect();
    scan_numbers(text, &toks, &cores)
        .into_iter()
        .map(|h| {
            let s = match h.digits {
                Some((a, b)) => cores.get(h.first).and_then(|c| c.get(a..b)).unwrap_or("").to_string(),
                None => {
                    let (a, b) = (toks.get(h.first).map_or(0, |t| t.cs), toks.get(h.last).map_or(0, |t| t.ce));
                    text.get(a..b.max(a)).unwrap_or("").to_string()
                }
            };
            Number { value: h.value, text: s, said: h.said }
        })
        .collect()
}

/// `v` as digits: years (1900–2099) plain, other numbers from 1,000 with group commas.
pub fn digits_of(v: f64) -> String {
    if v.fract() != 0.0 {
        return format!("{v}");
    }
    let n = v.max(0.0) as u64;
    if n < 1000 || (1900..=2099).contains(&n) {
        return n.to_string();
    }
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Spoken words

/// The word is a Latin letter's name ("എ", "ആർ", "എ-യുടെ").
fn letter_word(core: &str) -> bool {
    let head = core.split('-').next().unwrap_or(core);
    !head.is_empty() && (LETTERS.contains(&head) || head == "ക്യു" || head == "ഓ")
}

/// Said in one breath with its neighbours: an English number word or a letter's name.
fn spelled(core: &str) -> bool {
    en_word(core).is_some() || letter_word(core)
}

fn spoken_count(cores: &[String]) -> usize {
    let mut n = 0;
    let mut prev_spelled = false;
    for c in cores.iter().filter(|c| c.chars().any(char::is_alphanumeric)) {
        let s = spelled(c);
        if !(s && prev_spelled) {
            n += 1;
        }
        prev_spelled = s;
    }
    n
}

/// The spoken words of `sentence`: a number said the English way or an acronym spelled
/// letter by letter counts as one.
pub fn spoken_words(sentence: &str) -> usize {
    let text = clip(sentence);
    let cores: Vec<String> = tokens(text).iter().map(|t| malayalam::canonical(t.core(text))).collect();
    spoken_count(&cores)
}

// ---------------------------------------------------------------------------------------------
// Formal words and their spoken forms

/// How a table form matches a word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum How {
    /// The whole word.
    Word,
    /// The start of the word (inflected forms).
    Stem,
    /// The end of the word.
    Suffix,
}

/// What [`fix`] may replace a match with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fix {
    No,
    /// This, then the rest of the word.
    Prefix(&'static str),
    /// This then the rest, when the rest is a present, future or infinitive ending ("-ുന്നു",
    /// "-ും", "-ാൻ", "-ണം"): "-ിക്ക" and "ചെയ്യ" take the same endings.
    Present(&'static str),
    /// A past stem ending in "ി" ("ഉണ്ടാക്കി"), for a "-ിച്ച" stem: "-ിച്ചു" → "ഉണ്ടാക്കി",
    /// "-ിച്ചത്" → "ഉണ്ടാക്കിയത്".
    PastI(&'static str),
    /// For a suffix: the word before it, then this.
    Suffix(&'static str),
}

struct Formal {
    form: &'static str,
    how: How,
    say: &'static str,
    fix: Fix,
    /// A warning (else a hint).
    strong: bool,
}

const fn f(form: &'static str, how: How, say: &'static str, fix: Fix, strong: bool) -> Formal {
    Formal { form, how, say, fix, strong }
}

use Fix::{No, PastI, Prefix, Present};
use How::{Stem, Suffix, Word};

/// Formal and literary forms with a common spoken one, the specific before the general (the
/// first that matches a word is the one reported).
const FORMAL: &[Formal] = &[
    // welcome, visitors
    f("സ്വാഗതം ചെയ്", Stem, "വെൽക്കം ചെയ്യുന്നു", Prefix("വെൽക്കം ചെയ്"), true),
    f("സ്വാഗതം", Word, "വെൽക്കം", Prefix("വെൽക്കം"), true),
    f("സന്ദർശകരെ", Word, "വിസിറ്റേഴ്സിനെ", Prefix("വിസിറ്റേഴ്സിനെ"), true),
    f("സന്ദർശകരോട്", Word, "വിസിറ്റേഴ്സിനോട്", Prefix("വിസിറ്റേഴ്സിനോട്"), true),
    f("സന്ദർശകർക്ക്", Word, "വിസിറ്റേഴ്സിന്", Prefix("വിസിറ്റേഴ്സിന്"), true),
    f("സന്ദർശകരുടെ", Word, "വിസിറ്റേഴ്സിന്റെ", Prefix("വിസിറ്റേഴ്സിന്റെ"), true),
    f("സന്ദർശകർ", Word, "വിസിറ്റേഴ്സ്", Prefix("വിസിറ്റേഴ്സ്"), true),
    f("സന്ദർശക", Stem, "വിസിറ്റേഴ്സ്", No, true),
    f("സന്ദർശിക്ക", Stem, "വിസിറ്റ് ചെയ്യുക", Present("വിസിറ്റ് ചെയ്യ"), true),
    f("സന്ദർശിച്ച", Stem, "വിസിറ്റ് ചെയ്തു", Prefix("വിസിറ്റ് ചെയ്ത"), true),
    // making, developing, designing, launching
    f("വികസിപ്പിക്ക", Stem, "ഡെവലപ്പ് ചെയ്യുക", Present("ഡെവലപ്പ് ചെയ്യ"), true),
    f("വികസിപ്പിച്ച", Stem, "ഡെവലപ്പ് ചെയ്തു", Prefix("ഡെവലപ്പ് ചെയ്ത"), true),
    f("നിർമ്മിക്ക", Stem, "ഉണ്ടാക്കുക", Present("ഉണ്ടാക്ക"), true),
    f("നിർമിക്ക", Stem, "ഉണ്ടാക്കുക", Present("ഉണ്ടാക്ക"), true),
    f("നിർമ്മിച്ച", Stem, "ഉണ്ടാക്കി", PastI("ഉണ്ടാക്കി"), true),
    f("നിർമിച്ച", Stem, "ഉണ്ടാക്കി", PastI("ഉണ്ടാക്കി"), true),
    f("നിർമ്മാണം", Word, "പ്രൊഡക്ഷൻ", Prefix("പ്രൊഡക്ഷൻ"), true),
    f("നിർമ്മാണ", Stem, "പ്രൊഡക്ഷൻ / ഉണ്ടാക്കുന്നത്", No, true),
    f("രൂപകൽപ്പന ചെയ്", Stem, "ഡിസൈൻ ചെയ്തു", Prefix("ഡിസൈൻ ചെയ്"), true),
    f("രൂപകല്പന ചെയ്", Stem, "ഡിസൈൻ ചെയ്തു", Prefix("ഡിസൈൻ ചെയ്"), true),
    f("രൂപകൽപ്പന", Word, "ഡിസൈൻ", Prefix("ഡിസൈൻ"), true),
    f("രൂപകൽപ്പന", Stem, "ഡിസൈൻ", No, true),
    f("അവതരിപ്പിക്ക", Stem, "ലോഞ്ച് ചെയ്യുക", Present("ലോഞ്ച് ചെയ്യ"), true),
    f("അവതരിപ്പിച്ച", Stem, "ലോഞ്ച് ചെയ്തു", Prefix("ലോഞ്ച് ചെയ്ത"), true),
    f("പുറത്തിറക്കിയ", Stem, "ലോഞ്ച് ചെയ്ത", Prefix("ലോഞ്ച് ചെയ്ത"), true),
    f("പുറത്തിറക്കി", Word, "ലോഞ്ച് ചെയ്തു / ഇറക്കി", No, true),
    f("പുറത്തിറക്ക", Stem, "ലോഞ്ച് ചെയ്യുക", Present("ലോഞ്ച് ചെയ്യ"), true),
    f("ഉൽപ്പാദനം", Word, "പ്രൊഡക്ഷൻ", Prefix("പ്രൊഡക്ഷൻ"), true),
    f("ഉത്പാദനം", Word, "പ്രൊഡക്ഷൻ", Prefix("പ്രൊഡക്ഷൻ"), true),
    f("ഉല്പാദനം", Word, "പ്രൊഡക്ഷൻ", Prefix("പ്രൊഡക്ഷൻ"), true),
    f("ഉൽപ്പാദന", Stem, "പ്രൊഡക്ഷൻ", No, true),
    f("ഉത്പാദന", Stem, "പ്രൊഡക്ഷൻ", No, true),
    // getting, being able, starting
    f("ലഭിക്ക", Stem, "കിട്ടുക", Present("കിട്ട"), true),
    f("ലഭിച്ച", Stem, "കിട്ടി", PastI("കിട്ടി"), true),
    f("ലഭ്യമാണ്", Word, "കിട്ടും", Prefix("കിട്ടും"), true),
    f("സാധിക്ക", Stem, "പറ്റുക", Prefix("പറ്റ"), true),
    f("സാധിച്ച", Stem, "പറ്റി", PastI("പറ്റി"), true),
    f("സാധ്യമാണ്", Word, "പറ്റും", Prefix("പറ്റും"), true),
    f("ആരംഭിക്ക", Stem, "തുടങ്ങുക", Present("തുടങ്ങ"), true),
    f("ആരംഭിച്ച", Stem, "തുടങ്ങി", PastI("തുടങ്ങി"), true),
    f("കഴിയും", Word, "പറ്റും", Prefix("പറ്റും"), false),
    f("കഴിയില്ല", Word, "പറ്റില്ല", Prefix("പറ്റില്ല"), false),
    f("കഴിയുന്ന", Word, "പറ്റുന്ന", Prefix("പറ്റുന്ന"), false),
    f("ആവശ്യമാണ്", Word, "വേണം", Prefix("വേണം"), true),
    f("ആവശ്യമുണ്ട്", Word, "വേണം", Prefix("വേണം"), false),
    // controlling, using, working, showing, testing, sorting, preparing
    f("നിയന്ത്രിക്ക", Stem, "കൺട്രോൾ ചെയ്യുക", Present("കൺട്രോൾ ചെയ്യ"), true),
    f("നിയന്ത്രിച്ച", Stem, "കൺട്രോൾ ചെയ്തു", Prefix("കൺട്രോൾ ചെയ്ത"), true),
    f("ഉപയോഗിച്ച്", Word, "വെച്ച് / യൂസ് ചെയ്ത്", No, false),
    f("ഉപയോഗിക്ക", Stem, "യൂസ് ചെയ്യുക", Present("യൂസ് ചെയ്യ"), false),
    f("ഉപയോഗിച്ച", Stem, "യൂസ് ചെയ്തു", Prefix("യൂസ് ചെയ്ത"), false),
    f("ഉപയോക്താക്കൾ", Word, "യൂസേഴ്സ്", Prefix("യൂസേഴ്സ്"), true),
    f("ഉപയോക്ത", Stem, "യൂസർ", No, true),
    f("പ്രവർത്തിക്ക", Stem, "വർക്ക് ചെയ്യുക", Present("വർക്ക് ചെയ്യ"), false),
    f("പ്രവർത്തിച്ച", Stem, "വർക്ക് ചെയ്തു", Prefix("വർക്ക് ചെയ്ത"), false),
    f("പ്രദർശിപ്പിക്ക", Stem, "കാണിക്കുക", Present("കാണിക്ക"), true),
    f("പ്രദർശിപ്പിച്ച", Stem, "കാണിച്ചു", Prefix("കാണിച്ച"), true),
    f("പ്രദർശനം", Word, "എക്സിബിഷൻ / ഷോ", No, false),
    f("പരീക്ഷിക്ക", Stem, "ടെസ്റ്റ് ചെയ്യുക", Present("ടെസ്റ്റ് ചെയ്യ"), false),
    f("പരീക്ഷിച്ച", Stem, "ടെസ്റ്റ് ചെയ്തു / ട്രയൽ ചെയ്തു", Prefix("ടെസ്റ്റ് ചെയ്ത"), false),
    f("തരംതിരിക്ക", Stem, "സോർട്ട് ചെയ്യുക", Present("സോർട്ട് ചെയ്യ"), true),
    f("തരംതിരിച്ച", Stem, "സോർട്ട് ചെയ്തു", Prefix("സോർട്ട് ചെയ്ത"), true),
    f("തയ്യാറാക്ക", Stem, "റെഡിയാക്കുക", Prefix("റെഡിയാക്ക"), false),
    // finding, going around, carrying
    f("കണ്ടെത്തിയ", Stem, "കണ്ടുപിടിച്ച", Prefix("കണ്ടുപിടിച്ച"), false),
    f("കണ്ടെത്തി", Word, "കണ്ടുപിടിച്ചു / കണ്ടുപിടിച്ച്", No, false),
    f("കണ്ടെത്ത", Stem, "കണ്ടുപിടിക്കുക", Present("കണ്ടുപിടിക്ക"), false),
    f("ചുറ്റിനട", Stem, "കറങ്ങി നടക്കുക", Prefix("കറങ്ങി നട"), false),
    f("ചുമക്ക", Stem, "കൊണ്ടുപോകുക", Present("കൊണ്ടുപോക"), false),
    // things
    f("മുഖഭാവങ്ങൾ", Word, "എക്സ്പ്രഷൻസ്", Prefix("എക്സ്പ്രഷൻസ്"), true),
    f("മുഖഭാവ", Stem, "എക്സ്പ്രഷൻ", No, true),
    f("വികാരങ്ങൾ", Word, "ഇമോഷൻസ്", Prefix("ഇമോഷൻസ്"), true),
    f("തലച്ചോർ", Word, "ബ്രെയിൻ", Prefix("ബ്രെയിൻ"), false),
    f("തലച്ചോറ്", Word, "ബ്രെയിൻ", Prefix("ബ്രെയിൻ"), false),
    f("തലച്ചോറ", Stem, "ബ്രെയിൻ", No, false),
    f("വേഗത്തിൽ", Word, "സ്പീഡിൽ / വേഗം", No, false),
    f("വേഗത", Word, "സ്പീഡ്", Prefix("സ്പീഡ്"), false),
    f("വേഗത", Stem, "സ്പീഡ്", No, false),
    f("പരമാവധി", Word, "മാക്സിമം", Prefix("മാക്സിമം"), false),
    f("ഭക്ഷണം", Word, "ഫുഡ്", Prefix("ഫുഡ്"), false),
    f("ഭക്ഷണ", Stem, "ഫുഡ്", No, false),
    f("വിഭവങ്ങൾ", Word, "ഫുഡ് / ഡിഷസ്", No, false),
    f("യഥാർത്ഥത്തിൽ", Word, "ശരിക്കും", Prefix("ശരിക്കും"), true),
    f("യഥാർത്ഥ", Word, "ശരിക്കുള്ള", Prefix("ശരിക്കുള്ള"), true),
    f("യഥാർത്ഥ", Stem, "ശരിക്കുള്ള / ശരിക്കും", No, true),
    f("ഗവേഷകരെ", Word, "റിസർച്ചേഴ്സിനെ", Prefix("റിസർച്ചേഴ്സിനെ"), true),
    f("ഗവേഷകർ", Word, "റിസർച്ചേഴ്സ്", Prefix("റിസർച്ചേഴ്സ്"), true),
    f("ഗവേഷക", Stem, "റിസർച്ചേഴ്സ്", No, true),
    f("സ്വദേശി", Word, "സ്വന്തം", No, false),
    f("പടികൾ", Word, "സ്റ്റെപ്പുകൾ", Prefix("സ്റ്റെപ്പുകൾ"), false),
    f("പടികള", Stem, "സ്റ്റെപ്പുകൾ", Prefix("സ്റ്റെപ്പുകള"), false),
    f("റോബോട്ട് നായക", Stem, "റോബോട്ട് ഡോഗുകൾ", Prefix("റോബോട്ട് ഡോഗുക"), true),
    f("റോബോട്ട് നായ", Word, "റോബോട്ട് ഡോഗ്", Prefix("റോബോട്ട് ഡോഗ്"), true),
    f("സാങ്കേതികവിദ്യ", Stem, "ടെക്നോളജി", Prefix("ടെക്നോളജി"), true),
    f("കൃത്രിമ ബുദ്ധി", Stem, "എ ഐ", No, true),
    f("നിർമ്മിത ബുദ്ധി", Stem, "എ ഐ", No, true),
    f("യന്ത്രമനുഷ്യ", Stem, "റോബോട്ട്", No, true),
    f("ഉപകരണം", Word, "ഡിവൈസ്", Prefix("ഡിവൈസ്"), false),
    f("ഉപകരണ", Stem, "ഡിവൈസ്", No, false),
    f("സവിശേഷതകൾ", Word, "ഫീച്ചേഴ്സ്", Prefix("ഫീച്ചേഴ്സ്"), true),
    f("സവിശേഷത", Stem, "ഫീച്ചർ", No, true),
    f("സംവിധാനം", Word, "സിസ്റ്റം", Prefix("സിസ്റ്റം"), false),
    f("ഗുണനിലവാരം", Word, "ക്വാളിറ്റി", Prefix("ക്വാളിറ്റി"), true),
    f("ഗുണനിലവാര", Stem, "ക്വാളിറ്റി", No, true),
    f("ബാറ്ററി ആയുസ്സ്", Word, "ബാറ്ററി ലൈഫ്", Prefix("ബാറ്ററി ലൈഫ്"), true),
    f("അത്യാധുനിക", Stem, "ലേറ്റസ്റ്റ് / അഡ്വാൻസ്ഡ്", No, true),
    f("വളരെയധികം", Word, "ഒരുപാട്", Prefix("ഒരുപാട്"), true),
    f("വാഹന", Stem, "വണ്ടി", No, false),
    f("തെരുവ", Stem, "റോഡ് / സ്ട്രീറ്റ്", No, false),
    f("അണുനാശിനി", Stem, "ഡിസ്ഇൻഫെക്റ്റന്റ് / സാനിറ്റൈസർ", No, false),
    f("സമ്മേളന", Stem, "കോൺഫറൻസ്", No, false),
    // thanks, asking the viewer
    f("നന്ദി", Word, "താങ്ക്സ്", Prefix("താങ്ക്സ്"), false),
    f("പറയൂ", Word, "പറയണേ", Prefix("പറയണേ"), true),
    f("ചെയ്യൂ", Word, "ചെയ്യണേ", Prefix("ചെയ്യണേ"), true),
    f("കാണൂ", Word, "കാണണേ", Prefix("കാണണേ"), false),
    // joining words and written endings
    f("എന്നിരുന്നാലും", Word, "എന്നാലും", Prefix("എന്നാലും"), true),
    f("എന്നാൽ", Word, "പക്ഷേ", Prefix("പക്ഷേ"), true),
    f("കൂടാതെ", Word, "പിന്നെ", No, false),
    f("കൊണ്ടിരിക്കുന്നു", Suffix, "…കൊണ്ടിരിക്കുവാ / …കൊണ്ടേയിരിക്കും", No, false),
    f("ക്കായി", Suffix, "…ക്ക് വേണ്ടി", Fix::Suffix("ക്ക് വേണ്ടി"), true),
    f("ിനായി", Suffix, "…ിന് വേണ്ടി", Fix::Suffix("ിന് വേണ്ടി"), true),
    f("തിനാൽ", Suffix, "…തുകൊണ്ട്", Fix::Suffix("തുകൊണ്ട്"), true),
];

/// Endings a present-tense replacement may take over ([`Fix::Present`]).
const PRESENT_RESTS: [&str; 8] = ["ാ", "ു", "ൂ", "ണ", "ൽ", "പ്പെ", "േ", "ില്ല"];

/// The replacement `fix` makes of a match whose last word goes on with `rest` (a suffix
/// match: `word` is the whole last word).
fn replacement(fix: Fix, form_last: &str, word: &str, rest: &str) -> Option<String> {
    match fix {
        Fix::No => None,
        Fix::Prefix(to) => Some(format!("{to}{rest}")),
        Fix::Present(to) => (rest.is_empty() || PRESENT_RESTS.iter().any(|r| rest.starts_with(r))).then(|| format!("{to}{rest}")),
        Fix::PastI(base) => {
            let mut it = rest.chars();
            Some(match it.next() {
                Some('ു') | Some(VIRAMA) => format!("{base}{}", it.as_str()),
                _ => format!("{base}യ{rest}"),
            })
        }
        Fix::Suffix(to) => word.strip_suffix(form_last).map(|before| format!("{before}{to}")),
    }
}

/// A formal form at word `i` (no further than word `last`): the words it takes, the entry and
/// the replacement.
fn formal_at(text: &str, toks: &[Tok], cores: &[String], i: usize, last: usize) -> Option<(usize, &'static Formal, Option<String>)> {
    for e in FORMAL {
        let words: Vec<&str> = e.form.split(' ').collect();
        let k = words.len();
        if i + k - 1 > last {
            continue;
        }
        let Some((form_last, inner)) = words.split_last() else { continue };
        let inner_ok = inner.iter().enumerate().all(|(j, w)| cores.get(i + j).is_some_and(|c| c == w) && toks.get(i + j).is_some_and(|t| !t.ends_clause(text)));
        if !inner_ok {
            continue;
        }
        let Some(word) = cores.get(i + k - 1) else { continue };
        let rest = match e.how {
            How::Word => (word == form_last).then_some(""),
            How::Stem => word.strip_prefix(form_last),
            How::Suffix => (k == 1 && word.len() > form_last.len() && word.ends_with(form_last)).then_some(""),
        };
        let Some(rest) = rest else { continue };
        // a phrase's replacement holds its first words too; a suffix form is one word
        let fixed = replacement(e.fix, form_last, word, rest);
        return Some((k, e, fixed));
    }
    None
}

// ---------------------------------------------------------------------------------------------
// Verbs

/// Endings of a finite verb, a copula or a spoken sentence end.
const VERB_ENDINGS: [&str; 19] = ["ു", "ും", "ാം", "ണം", "ണേ", "ൂ", "ാ", "ി", "ഉണ്ട്", "ുണ്ട്", "ാണ്", "ആണ്", "ല്ല", "തി", "ോ", "േ", "ട്ടെ", "യത്", "ന്നത്"];

fn has_verb(cores: &[String]) -> bool {
    cores.iter().any(|c| {
        let w = c.split('-').next_back().unwrap_or(c);
        VERB_ENDINGS.iter().any(|e| w.ends_with(e))
    })
}

// ---------------------------------------------------------------------------------------------
// Latin letters

/// Runs of ASCII letters and digits with a letter in them, within `core` (byte ranges).
fn latin_runs(core: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    let mut has_letter = false;
    let close = |s: usize, e: usize, letter: bool, out: &mut Vec<(usize, usize)>| {
        if letter {
            out.push((s, e));
        }
    };
    for (i, c) in core.char_indices() {
        let part = c.is_ascii_alphanumeric() || (c == '.' && start.is_some());
        if part {
            if start.is_none() {
                start = Some(i);
                has_letter = false;
            }
            has_letter |= c.is_ascii_alphabetic();
        } else if let Some(s) = start.take() {
            close(s, i, has_letter, &mut out);
        }
    }
    if let Some(s) = start {
        close(s, core.len(), has_letter, &mut out);
    }
    out.into_iter()
        .map(|(s, e)| {
            let t = core.get(s..e).unwrap_or("");
            (s, s + t.trim_end_matches('.').len())
        })
        .collect()
}

/// An acronym or code spelled the way the voice reads it: letters' names, digits' English names
/// ("UAE" → "യു എ ഇ", "M02" → "എം സീറോ ടു"). None for a word (it has lower-case letters) and for
/// four or more capitals without a digit, which are often said as a word ("GITEX" ജൈടെക്സ്).
pub fn spell_code(code: &str) -> Option<String> {
    let letters: Vec<char> = code.chars().filter(char::is_ascii_alphabetic).collect();
    let digits = code.chars().any(|c| c.is_ascii_digit());
    if letters.is_empty() || letters.len() > 8 || letters.iter().any(char::is_ascii_lowercase) || (letters.len() > 3 && !digits) {
        return None;
    }
    let parts: Vec<&str> =
        code.chars().filter_map(|c| if c.is_ascii_digit() { Some(english_digit(c.to_digit(10).unwrap_or(0))) } else { letter_name(c) }).collect();
    Some(parts.join(" "))
}

// ---------------------------------------------------------------------------------------------
// The lint

/// `text` linted with the default options.
pub fn lint(text: &str) -> Vec<Finding> {
    lint_with(text, &Options::default())
}

/// `text` linted (see the module docs).
pub fn lint_with(text: &str, opts: &Options) -> Vec<Finding> {
    let text = clip(text);
    let max_words = opts.max_words.clamp(MIN_MAX_WORDS, MAX_MAX_WORDS);
    let toks = tokens(text);
    let cores: Vec<String> = toks.iter().map(|t| malayalam::canonical(t.core(text))).collect();
    let hits = scan_numbers(text, &toks, &cores);
    let mut out: Vec<Finding> = Vec::new();
    for (si, &(a, b)) in sentence_ranges(text, &toks).iter().enumerate() {
        if out.len() >= MAX_FINDINGS {
            break;
        }
        let (Some(first), Some(last_tok)) = (toks.get(a), toks.get(b)) else { continue };
        let sentence_span = (first.cs, last_tok.end);
        let sentence_text = text.get(first.start..last_tok.end).unwrap_or("").trim().to_string();
        let words = cores.get(a..=b).unwrap_or(&[]);
        let n = spoken_count(words);
        if n > max_words {
            out.push(Finding {
                kind: Kind::LongSentence,
                level: Level::Warning,
                sentence: si,
                text: sentence_text.clone(),
                message: format!("{n} spoken words: too long to say in one go (at most {max_words})"),
                suggestion: split_hint(text, &toks, &cores, a, b, max_words),
                fix: None,
                span: sentence_span,
            });
        }
        // word findings, in word order
        let mut word_findings: Vec<(usize, Finding)> = Vec::new();
        let mut i = a;
        while i <= b {
            if let Some((k, e, fixed)) = formal_at(text, &toks, &cores, i, b) {
                let (s, en) = (toks.get(i).map_or(0, |t| t.cs), toks.get(i + k - 1).map_or(0, |t| t.ce));
                let found = text.get(s..en.max(s)).unwrap_or("").to_string();
                word_findings.push((
                    i,
                    Finding {
                        kind: Kind::Formal,
                        level: if e.strong { Level::Warning } else { Level::Hint },
                        sentence: si,
                        message: format!("«{found}» is written Malayalam; people say «{}»", e.say),
                        suggestion: fixed.clone().unwrap_or_else(|| e.say.to_string()),
                        text: found,
                        fix: fixed,
                        span: (s, en.max(s)),
                    },
                ));
                i += k;
            } else {
                i += 1;
            }
        }
        for h in hits.iter().filter(|h| h.first >= a && h.first <= b && h.said == Said::Malayalam && h.value >= 100.0) {
            let (s, en) = (toks.get(h.first).map_or(0, |t| t.cs), toks.get(h.last).map_or(0, |t| t.ce));
            let found = text.get(s..en.max(s)).unwrap_or("").to_string();
            let digits = digits_of(h.value);
            let tail = digit_suffix(&h.suffix);
            let written = match &tail {
                Some(t) => format!("{digits}{t}"),
                None => format!("{digits} ({})", h.suffix),
            };
            let said = malayalam::spell_numbers(if tail.is_some() { &written } else { &digits }, Numbers::English);
            word_findings.push((
                h.first,
                Finding {
                    kind: Kind::NumberWords,
                    level: Level::Warning,
                    sentence: si,
                    message: format!(
                        "{} in Malayalam words: long number words blur the start of a line, and the English way is how it is said",
                        digits_of(h.value)
                    ),
                    suggestion: format!("{written} (said «{said}»)"),
                    text: found,
                    fix: tail.map(|t| format!("{digits}{t}")),
                    span: (s, en.max(s)),
                },
            ));
        }
        for j in a..=b {
            let Some(t) = toks.get(j) else { continue };
            let raw_core = t.core(text);
            for (rs, re) in latin_runs(raw_core) {
                let run = raw_core.get(rs..re).unwrap_or("");
                let spelled = spell_code(run);
                let fixed = spelled.as_ref().map(|sp| format!("{}{sp}{}", raw_core.get(..rs).unwrap_or(""), raw_core.get(re..).unwrap_or("")));
                let suggestion = match &spelled {
                    Some(sp) => format!("{sp} (spelled with the letters' names)"),
                    None => "write it in Malayalam script as it is said: a name by its sound (GITEX → ജൈടെക്സ്), an acronym letter by letter (എ ഐ)".to_string(),
                };
                word_findings.push((
                    j,
                    Finding {
                        kind: Kind::Latin,
                        level: Level::Warning,
                        sentence: si,
                        text: run.to_string(),
                        message: format!("«{run}» is in Latin letters: the voice reads Malayalam script"),
                        suggestion,
                        fix: fixed,
                        span: (t.cs, t.ce),
                    },
                ));
            }
        }
        word_findings.sort_by_key(|(i, _)| *i);
        out.extend(word_findings.into_iter().map(|(_, f)| f));
        let exclaims = last_tok.tail(text).contains('!');
        if !exclaims && words.iter().any(|w| w.chars().any(char::is_alphabetic)) && !has_verb(words) {
            out.push(Finding {
                kind: Kind::NoVerb,
                level: Level::Hint,
                sentence: si,
                text: sentence_text,
                message: "no verb ending: fine as a deliberate fragment, otherwise it reads like a list".into(),
                suggestion: "end it with a verb (…ആണ്, …ഉണ്ട്, …കണ്ടു), or keep it short on purpose".into(),
                fix: None,
                span: sentence_span,
            });
        }
    }
    out.truncate(MAX_FINDINGS);
    out
}

/// The suffix of a number word as it is written after digits ("ിലധികം" → "-ലധികം", "ത്തിൽ" →
/// "-ൽ", "ോളം" → "-ഓളം"); None when it does not come apart cleanly.
fn digit_suffix(suffix: &str) -> Option<String> {
    if suffix.is_empty() || suffix == "്" || suffix == "ം" {
        return Some(String::new());
    }
    // the joining consonant of a thousand or a hundred ("ആയിരത്തിൽ", "നൂറ്റിൽ")
    let s = ["ത്ത", "്റ", "്ത"].iter().find_map(|g| suffix.strip_prefix(g)).unwrap_or(suffix);
    let mut it = s.chars();
    let first = it.next()?;
    let rest = it.as_str();
    if first == 'ി' && rest.chars().next().is_some_and(|c| is_consonant(c) || chillu_base(c).is_some()) {
        return Some(format!("-{rest}"));
    }
    let vowel = match first {
        'ാ' => 'ആ',
        'െ' => 'എ',
        'േ' => 'ഏ',
        'ൊ' => 'ഒ',
        'ോ' => 'ഓ',
        'ു' => 'ഉ',
        _ => return None,
    };
    (!rest.is_empty()).then(|| format!("-{vowel}{rest}"))
}

/// Where to split a long sentence: at the comma that leaves the shorter longest half, or at
/// every comma when one split is not enough.
fn split_hint(text: &str, toks: &[Tok], cores: &[String], a: usize, b: usize, max: usize) -> String {
    let total = spoken_count(cores.get(a..=b).unwrap_or(&[]));
    let commas: Vec<(usize, usize)> = (a..b)
        .filter(|j| toks.get(*j).is_some_and(|t| t.tail(text).contains([',', ';', ':'])))
        .map(|j| (j, spoken_count(cores.get(a..=j).unwrap_or(&[]))))
        .collect();
    let word = |j: usize| toks.get(j).map(|t| t.core(text)).unwrap_or("");
    match commas.iter().min_by_key(|(_, left)| (*left).max(total.saturating_sub(*left))) {
        Some(&(j, left)) if left.max(total.saturating_sub(left)) <= max || commas.len() == 1 => {
            format!("split it after «{}» into two sentences ({left} and {} words)", word(j), total.saturating_sub(left))
        }
        Some(_) => format!("split it at its commas (after {})", commas.iter().map(|(j, _)| format!("«{}»", word(*j))).collect::<Vec<_>>().join(", ")),
        None => format!("say it as two sentences of at most {max} words"),
    }
}

/// A replacement [`fix`] made.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Applied {
    pub from: String,
    pub to: String,
}

/// A text with the safe replacements made.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Fixed {
    pub text: String,
    pub applied: Vec<Applied>,
}

/// `text` with every finding's safe replacement made (see [`Finding::fix`]); the rest of the
/// text is kept byte for byte.
pub fn fix(text: &str) -> Fixed {
    fix_with(text, &Options::default())
}

/// [`fix`] with these options.
pub fn fix_with(text: &str, opts: &Options) -> Fixed {
    let text = clip(text);
    let mut edits: Vec<((usize, usize), String, String)> =
        lint_with(text, opts).into_iter().filter_map(|f| f.fix.map(|to| (f.span, text.get(f.span.0..f.span.1).unwrap_or("").to_string(), to))).collect();
    edits.sort_by_key(|(s, _, _)| *s);
    let mut out = String::with_capacity(text.len());
    let mut at = 0usize;
    let mut applied = Vec::new();
    for ((s, e), from, to) in edits {
        if s < at || e < s || from == to {
            continue;
        }
        out.push_str(text.get(at..s).unwrap_or(""));
        out.push_str(&to);
        applied.push(Applied { from, to });
        at = e;
    }
    out.push_str(text.get(at..).unwrap_or(""));
    Fixed { text: out, applied }
}

#[cfg(test)]
#[path = "ml_script_tests.rs"]
mod tests;
