//! The Malayalam text front end: a script written in Malayalam (English loanwords written in
//! Malayalam script too) → utterances a Malayalam voice reads, and notes about anything it could
//! not read as written.
//!
//! - **Spelling** is made canonical: the two-part vowel signs composed (`െ` + `ാ` → `ൊ`, as NFC
//!   does), old-style chillus (consonant + virama + ZWJ) made the atomic chillu letters, other
//!   zero-width joiners dropped, Malayalam digits made ASCII digits.
//! - **Numbers** become Malayalam words in the Indian system (ആയിരം, ലക്ഷം, കോടി): "2013" →
//!   "രണ്ടായിരത്തി പതിമൂന്ന്". Years are read as numbers, as Malayalam does. A suffix written
//!   after a number joins it ("2013-ൽ" → "…പതിമൂന്നിൽ", "2-ാം" → "രണ്ടാം"); a range
//!   "2013–2025" → "… മുതൽ … വരെ"; "50%" → "… ശതമാനം"; "₹500" → "… രൂപ"; decimals with
//!   "പോയിന്റ്" and their digits one by one; digit strings longer than twelve, or with a
//!   leading zero, digit by digit.
//! - **Latin letters**: an acronym (all capitals, at most eight) is read letter by letter with
//!   the letters' Malayalam names ("AI" → "എഐ"), digits inside a code with their English names
//!   in Malayalam script ("G1" → "ജി വൺ"); any other Latin word is kept, and reported (the model
//!   was trained on Indian-language text: write such words in Malayalam script).
//! - **Symbols** with a spoken form are spoken (`&` ആൻഡ്, `+` പ്ലസ്, `@` അറ്റ്); dashes become
//!   commas; emoji and other symbols are dropped and reported.
//! - [`pieces`]: pause markers (`[pause 1s]`) become exact silence; sentences (`.` `?` `!` `।`)
//!   are grouped into utterances of at most the given length, a long sentence split at commas,
//!   else between words.

use crate::script::{self, Segment};
use crate::{TtsError, check_text};

/// The language tag of Malayalam narrations.
pub const TAG: &str = "ml-IN";
/// Most utterances one script makes.
const MAX_PIECES: usize = 2000;
/// Longest digit string read as a number (more is read digit by digit).
const MAX_NUMBER_DIGITS: usize = 12;

/// `tag` names Malayalam (`ml`, `ml-IN`, any case, `_` or `-`).
pub fn is_tag(tag: &str) -> bool {
    let t = tag.trim().to_ascii_lowercase().replace('_', "-");
    t == "ml" || t.starts_with("ml-")
}

/// Whether `text` is written in Malayalam: it has Malayalam letters and at least as many of
/// them as Latin ones.
pub fn detect(text: &str) -> bool {
    let ml = text.chars().filter(|c| is_malayalam(*c) && c.is_alphabetic()).count();
    let latin = text.chars().filter(char::is_ascii_alphabetic).count();
    ml > 0 && ml >= latin
}

fn is_malayalam(c: char) -> bool {
    ('\u{0D00}'..='\u{0D7F}').contains(&c)
}

const UNITS: [&str; 10] = ["പൂജ്യം", "ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്", "ഏഴ്", "എട്ട്", "ഒൻപത്"];
/// Units after a joining form ending in `ി`: a vowel-initial unit takes the glide `യ`.
const UNITS_JOINED: [&str; 10] = ["", "യൊന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "യഞ്ച്", "യാറ്", "യേഴ്", "യെട്ട്", "യൊൻപത്"];
const TEENS: [&str; 10] = ["പത്ത്", "പതിനൊന്ന്", "പന്ത്രണ്ട്", "പതിമൂന്ന്", "പതിനാല്", "പതിനഞ്ച്", "പതിനാറ്", "പതിനേഴ്", "പതിനെട്ട്", "പത്തൊൻപത്"];
const TENS: [&str; 10] = ["", "", "ഇരുപത്", "മുപ്പത്", "നാൽപ്പത്", "അമ്പത്", "അറുപത്", "എഴുപത്", "എൺപത്", "തൊണ്ണൂറ്"];
const TENS_JOINING: [&str; 10] = ["", "", "ഇരുപത്തി", "മുപ്പത്തി", "നാൽപ്പത്തി", "അമ്പത്തി", "അറുപത്തി", "എഴുപത്തി", "എൺപത്തി", "തൊണ്ണൂറ്റി"];
const HUNDREDS: [&str; 10] = ["", "നൂറ്", "ഇരുനൂറ്", "മുന്നൂറ്", "നാനൂറ്", "അഞ്ഞൂറ്", "അറുനൂറ്", "എഴുനൂറ്", "എണ്ണൂറ്", "തൊള്ളായിരം"];
const HUNDREDS_JOINING: [&str; 10] = ["", "നൂറ്റി", "ഇരുനൂറ്റി", "മുന്നൂറ്റി", "നാനൂറ്റി", "അഞ്ഞൂറ്റി", "അറുനൂറ്റി", "എഴുനൂറ്റി", "എണ്ണൂറ്റി", "തൊള്ളായിരത്തി"];
const THOUSANDS: [&str; 10] = ["", "ആയിരം", "രണ്ടായിരം", "മൂവായിരം", "നാലായിരം", "അയ്യായിരം", "ആറായിരം", "ഏഴായിരം", "എണ്ണായിരം", "ഒൻപതിനായിരം"];
/// English digit names in Malayalam script (digits inside codes like "G1").
const ENGLISH_DIGITS: [&str; 10] = ["സീറോ", "വൺ", "ടു", "ത്രീ", "ഫോർ", "ഫൈവ്", "സിക്സ്", "സെവൻ", "എയ്റ്റ്", "നൈൻ"];
/// The Latin letters' names in Malayalam script.
const LETTERS: [&str; 26] = [
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

const VIRAMA: char = '\u{0D4D}';

fn below_100(n: u64) -> String {
    let n = n % 100;
    let (t, u) = ((n / 10) as usize, (n % 10) as usize);
    match t {
        0 => UNITS.get(u).copied().unwrap_or("").to_string(),
        1 => TEENS.get(u).copied().unwrap_or("").to_string(),
        _ if u == 0 => TENS.get(t).copied().unwrap_or("").to_string(),
        _ => format!("{}{}", TENS_JOINING.get(t).copied().unwrap_or(""), UNITS_JOINED.get(u).copied().unwrap_or("")),
    }
}

fn below_1000(n: u64) -> String {
    let n = n % 1000;
    let (h, r) = ((n / 100) as usize, n % 100);
    match (h, r) {
        (0, r) => below_100(r),
        (h, 0) => HUNDREDS.get(h).copied().unwrap_or("").to_string(),
        (h, r) => format!("{} {}", HUNDREDS_JOINING.get(h).copied().unwrap_or(""), below_100(r)),
    }
}

/// `x` (1–99) thousand.
fn thousands(x: u64) -> String {
    if let Some(t) = THOUSANDS.get(x as usize).filter(|_| x < 10) {
        return t.to_string();
    }
    let w = below_100(x);
    if let Some(stem) = w.strip_suffix("ത്ത്").or_else(|| w.strip_suffix("ത്")) {
        return format!("{stem}തിനായിരം");
    }
    match w.strip_suffix(VIRAMA) {
        Some(stem) => format!("{stem}ായിരം"),
        None => format!("{w} ആയിരം"),
    }
}

/// The joining form of a word ending in `ം` (`ആയിരം` → `ആയിരത്തി`).
fn joining(w: &str) -> String {
    match w.strip_suffix('ം') {
        Some(stem) => format!("{stem}ത്തി"),
        None => w.to_string(),
    }
}

/// `n` in Malayalam words (the Indian system: ആയിരം, ലക്ഷം, കോടി).
pub fn number(n: u64) -> String {
    if n == 0 {
        return UNITS[0].to_string();
    }
    let crore = n / 10_000_000;
    let lakh = (n / 100_000) % 100;
    let thousand = (n / 1000) % 100;
    let rest = n % 1000;
    let mut parts: Vec<String> = Vec::new();
    if crore > 0 {
        let w = if crore == 1 { "ഒരു".to_string() } else { number(crore) };
        parts.push(format!("{w} കോടി"));
    }
    if lakh > 0 {
        let w = if lakh == 1 { "ഒരു".to_string() } else { below_100(lakh) };
        let last = thousand == 0 && rest == 0;
        parts.push(if last { format!("{w} ലക്ഷം") } else { format!("{w} ലക്ഷത്തി") });
    }
    if thousand > 0 {
        let t = thousands(thousand);
        parts.push(if rest == 0 { t } else { joining(&t) });
    }
    if rest > 0 {
        parts.push(below_1000(rest));
    }
    parts.join(" ")
}

/// The ordinal of `n` ("രണ്ടാം").
pub fn ordinal(n: u64) -> String {
    attach(&number(n), "ാം")
}

/// Join a suffix written after a number (`-ൽ`, `-ാം`, `-ഓടെ`…) to its word, with the sandhi
/// Malayalam writes.
fn attach(word: &str, suffix: &str) -> String {
    let Some(first) = suffix.chars().next() else { return word.to_string() };
    let sign = vowel_sign(first);
    if let Some(stem) = word.strip_suffix(VIRAMA) {
        return match (first, sign) {
            // a vowel sign replaces the virama's half-vowel
            (c, _) if is_vowel_sign(c) => format!("{stem}{suffix}"),
            // an independent vowel becomes its sign
            (_, Some(s)) => format!("{stem}{s}{}", suffix.chars().skip(1).collect::<String>()),
            // a consonant or chillu: the half-vowel becomes ി ("…മൂന്നിൽ")
            _ => format!("{stem}ി{suffix}"),
        };
    }
    if let Some(stem) = word.strip_suffix('ം') {
        // ആയിരം + ൽ → ആയിരത്തിൽ
        return match (first, sign) {
            (c, _) if is_vowel_sign(c) => format!("{stem}ത്ത{suffix}"),
            (_, Some(s)) => format!("{stem}ത്ത{s}{}", suffix.chars().skip(1).collect::<String>()),
            _ => format!("{stem}ത്തി{suffix}"),
        };
    }
    format!("{word}{suffix}")
}

fn is_vowel_sign(c: char) -> bool {
    ('\u{0D3E}'..='\u{0D4C}').contains(&c) || c == '\u{0D57}'
}

/// The vowel sign of an independent vowel (`ഓ` → `ോ`; `അ` → nothing).
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

/// Canonical Malayalam spelling (see the module docs).
pub fn canonical(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        let next = chars.get(i + 1).copied();
        let after = chars.get(i + 2).copied();
        // old-style chillu: consonant + virama + ZWJ
        if next == Some(VIRAMA) && after == Some('\u{200D}') {
            let chillu = match c {
                'ണ' => Some('ൺ'),
                'ന' => Some('ൻ'),
                'ര' | 'റ' => Some('ർ'),
                'ല' => Some('ൽ'),
                'ള' => Some('ൾ'),
                'ക' => Some('ൿ'),
                _ => None,
            };
            if let Some(ch) = chillu {
                out.push(ch);
                i += 3;
                continue;
            }
        }
        // two-part vowel signs (canonical composition)
        let composed = match (c, next) {
            ('\u{0D46}', Some('\u{0D3E}')) => Some('\u{0D4A}'),
            ('\u{0D47}', Some('\u{0D3E}')) => Some('\u{0D4B}'),
            ('\u{0D46}', Some('\u{0D57}')) => Some('\u{0D4C}'),
            _ => None,
        };
        if let Some(v) = composed {
            out.push(v);
            i += 2;
            continue;
        }
        match c {
            '\u{200C}' | '\u{200D}' | '\u{200B}' | '\u{FEFF}' => {}
            '\u{0D66}'..='\u{0D6F}' => out.push(char::from(b'0' + (c as u32 - 0x0D66) as u8)),
            c => out.push(c),
        }
        i += 1;
    }
    out
}

/// Text normalised for a Malayalam voice, and what could not be read as written.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Normalized {
    pub text: String,
    pub notes: Vec<String>,
}

fn push_word(out: &mut String, w: &str) {
    if !out.is_empty() && !out.ends_with([' ', '(', '"', '\'']) {
        out.push(' ');
    }
    out.push_str(w);
}

/// Digits read one by one with their Malayalam names.
fn digit_words(d: &str) -> String {
    d.chars().filter_map(|c| c.to_digit(10)).filter_map(|v| UNITS.get(v as usize).copied()).collect::<Vec<_>>().join(" ")
}

/// A run of digits (commas between groups allowed) as a number, if it reads as one.
fn parse_number(run: &str) -> Option<u64> {
    let plain: String = run.chars().filter(|c| *c != ',').collect();
    if plain.is_empty() || plain.len() > MAX_NUMBER_DIGITS || (plain.len() > 1 && plain.starts_with('0')) {
        return None;
    }
    // commas only between groups of two or three digits (Indian or Western grouping)
    let groups: Vec<&str> = run.split(',').collect();
    if groups.len() > 1 && (groups.first().is_none_or(|g| g.is_empty() || g.len() > 3) || groups.iter().skip(1).any(|g| !(2..=3).contains(&g.len()))) {
        return None;
    }
    plain.parse().ok()
}

/// Normalise one stretch of speech (no pause markers) for a Malayalam voice.
pub fn normalize(text: &str) -> Normalized {
    let text = canonical(text);
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() * 2);
    let mut notes: Vec<String> = Vec::new();
    let mut note = |n: String| {
        if !notes.contains(&n) && notes.len() < 50 {
            notes.push(n);
        }
    };
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        if c.is_ascii_digit() {
            // a number: digits with group commas and one decimal point
            let s = i;
            while chars.get(i).is_some_and(|c| c.is_ascii_digit() || (*c == ',' && chars.get(i + 1).is_some_and(char::is_ascii_digit))) {
                i += 1;
            }
            let int: String = chars.get(s..i).unwrap_or(&[]).iter().collect();
            let mut frac = String::new();
            if chars.get(i) == Some(&'.') && chars.get(i + 1).is_some_and(char::is_ascii_digit) {
                i += 1;
                while chars.get(i).is_some_and(char::is_ascii_digit) {
                    frac.push(chars.get(i).copied().unwrap_or('0'));
                    i += 1;
                }
            }
            let mut words = match parse_number(&int) {
                Some(n) => number(n),
                None => digit_words(&int),
            };
            if !frac.is_empty() {
                words = format!("{words} പോയിന്റ് {}", digit_words(&frac));
            }
            // a range: 2013–2025
            if matches!(chars.get(i), Some('-' | '–')) && chars.get(i + 1).is_some_and(char::is_ascii_digit) && frac.is_empty() {
                let s2 = i + 1;
                let mut j = s2;
                while chars.get(j).is_some_and(char::is_ascii_digit) {
                    j += 1;
                }
                let second: String = chars.get(s2..j).unwrap_or(&[]).iter().collect();
                let w2 = parse_number(&second).map(number).unwrap_or_else(|| digit_words(&second));
                push_word(&mut out, &format!("{words} മുതൽ {w2} വരെ"));
                i = j;
                continue;
            }
            // a percentage
            if chars.get(i) == Some(&'%') {
                push_word(&mut out, &format!("{words} ശതമാനം"));
                i += 1;
                continue;
            }
            // a suffix written after the number: "2013-ൽ", "2-ാം", "2013ൽ"
            let hyphen = usize::from(matches!(chars.get(i), Some('-')) && chars.get(i + 1).is_some_and(|c| is_malayalam(*c)));
            if chars.get(i + hyphen).is_some_and(|c| is_malayalam(*c)) {
                let s3 = i + hyphen;
                let mut j = s3;
                while chars.get(j).is_some_and(|c| is_malayalam(*c)) {
                    j += 1;
                }
                let suffix: String = chars.get(s3..j).unwrap_or(&[]).iter().collect();
                // the suffix joins the number's last word
                let (head, last) = words.rsplit_once(' ').map_or((String::new(), words.clone()), |(h, l)| (h.to_string(), l.to_string()));
                let joined = attach(&last, &suffix);
                push_word(&mut out, &if head.is_empty() { joined } else { format!("{head} {joined}") });
                i = j;
                continue;
            }
            push_word(&mut out, &words);
            continue;
        }
        if c.is_ascii_alphabetic() {
            let s = i;
            while chars.get(i).is_some_and(char::is_ascii_alphanumeric) {
                i += 1;
            }
            let word: String = chars.get(s..i).unwrap_or(&[]).iter().collect();
            push_word(&mut out, &latin(&word, &mut note));
            continue;
        }
        i += 1;
        match c {
            c if is_malayalam(c) => {
                // a Malayalam word continues the current one
                out.push(c);
            }
            c if c.is_whitespace() => {
                if !out.is_empty() && !out.ends_with(' ') {
                    out.push(' ');
                }
            }
            '.' | ',' | '?' | '!' | ':' | '"' | '\'' | '(' | ')' => {
                // punctuation sits on its word
                while out.ends_with(' ') && !matches!(c, '(' | '"' | '\'') {
                    out.pop();
                }
                out.push(c);
            }
            ';' => out.push(','),
            '।' | '॥' => out.push('.'),
            '“' | '”' | '«' | '»' => out.push('"'),
            '‘' | '’' => out.push('\''),
            '—' | '–' | '-' => {
                // a dash between words is a short pause
                if out.trim_end().ends_with(|ch: char| ch.is_alphanumeric() || is_malayalam(ch)) && chars.get(i).is_some_and(|n| n.is_whitespace()) {
                    while out.ends_with(' ') {
                        out.pop();
                    }
                    out.push(',');
                } else if c == '-' && out.ends_with(|ch: char| is_malayalam(ch)) && chars.get(i).is_some_and(|n| is_malayalam(*n)) {
                    // a hyphenated Malayalam compound reads as one word
                }
            }
            '…' => out.push_str("..."),
            '&' => push_word(&mut out, "ആൻഡ്"),
            '+' => push_word(&mut out, "പ്ലസ്"),
            '@' => push_word(&mut out, "അറ്റ്"),
            '₹' => push_word(&mut out, "രൂപ"),
            '$' => push_word(&mut out, "ഡോളർ"),
            '°' => push_word(&mut out, "ഡിഗ്രി"),
            '=' => push_word(&mut out, "സമം"),
            '/' | '|' | '_' | '*' | '#' | '~' | '^' | '`' | '<' | '>' | '[' | ']' | '{' | '}' | '\\' => {
                if !out.is_empty() && !out.ends_with(' ') {
                    out.push(' ');
                }
            }
            c if c.is_control() => {}
            c => note(format!("left out `{c}` (not readable as written)")),
        }
    }
    // a currency word read before its amount goes after it, as it is spoken ("₹500" → "500 രൂപ")
    let text = move_currency(&out.split_whitespace().collect::<Vec<_>>().join(" "));
    Normalized { text, notes }
}

/// "രൂപ അഞ്ഞൂറ്" (from "₹500") → "അഞ്ഞൂറ് രൂപ"; the same for dollars.
fn move_currency(text: &str) -> String {
    let words: Vec<&str> = text.split(' ').collect();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    let mut i = 0;
    while let Some(w) = words.get(i) {
        if matches!(*w, "രൂപ" | "ഡോളർ") && words.get(i + 1).is_some_and(|n| is_number_word(n)) {
            let mut j = i + 1;
            while words.get(j).is_some_and(|n| is_number_word(n)) {
                j += 1;
            }
            out.extend(words.get(i + 1..j).unwrap_or(&[]).iter().map(|s| s.to_string()));
            out.push((*w).to_string());
            i = j;
            continue;
        }
        out.push((*w).to_string());
        i += 1;
    }
    out.join(" ")
}

fn is_number_word(w: &str) -> bool {
    let w = w.trim_end_matches(['.', ',', '?', '!']);
    UNITS.iter().chain(&TEENS).chain(&TENS).chain(&HUNDREDS).chain(&THOUSANDS).any(|u| !u.is_empty() && w == *u)
        || ["ലക്ഷം", "കോടി", "ഒരു", "പോയിന്റ്"].contains(&w)
        || [TENS_JOINING, HUNDREDS_JOINING].iter().flatten().any(|j| !j.is_empty() && w.starts_with(j))
        || w.ends_with("ായിരം")
        || w.ends_with("ായിരത്തി")
        || w.ends_with("ലക്ഷത്തി")
}

/// A Latin word for a Malayalam voice: acronyms and codes spelled, other words kept (noted).
fn latin(word: &str, note: &mut dyn FnMut(String)) -> String {
    let letters = word.chars().filter(char::is_ascii_alphabetic).count();
    let caps = word.chars().filter(char::is_ascii_uppercase).count();
    let has_digit = word.chars().any(|c| c.is_ascii_digit());
    let acronym = caps == letters && letters <= 8;
    if !acronym && !has_digit {
        note(format!("kept the Latin word `{word}`: the voice may not say it well; write it in Malayalam script"));
        return word.to_string();
    }
    // letters and digits in groups: "R2D2" → "ആർ ടു ഡി ടു"
    let mut groups: Vec<String> = Vec::new();
    let chars: Vec<char> = word.chars().collect();
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        let s = i;
        let digit = c.is_ascii_digit();
        while chars.get(i).is_some_and(|x| x.is_ascii_digit() == digit) {
            i += 1;
        }
        let part: String = chars.get(s..i).unwrap_or(&[]).iter().collect();
        if digit {
            groups.push(part.chars().filter_map(|d| d.to_digit(10)).filter_map(|v| ENGLISH_DIGITS.get(v as usize).copied()).collect::<Vec<_>>().join(" "));
        } else if part.chars().all(|c| c.is_ascii_uppercase()) && part.len() <= 8 {
            groups.push(part.chars().filter_map(|c| LETTERS.get((c as u8 - b'A') as usize).copied()).collect::<String>());
        } else {
            note(format!("kept the Latin word `{part}`: the voice may not say it well; write it in Malayalam script"));
            groups.push(part);
        }
    }
    groups.join(" ")
}

/// Syllables of Malayalam text: every vowel sound — an independent vowel, a consonant that is
/// not silenced by a virama inside a cluster (its inherent or written vowel), and a word-final
/// virama (spoken as a short "ŭ", as in "ആണ്"). Chillus and conjunct halves add none; other
/// scripts count their letter groups roughly (a vowel letter run each).
pub fn syllables(text: &str) -> usize {
    let chars: Vec<char> = text.chars().collect();
    let mut n = 0usize;
    for (i, c) in chars.iter().enumerate() {
        let next = chars.get(i + 1).copied();
        match *c {
            'അ'..='ഔ' => n += 1,
            'ക'..='ഹ' => {
                // silenced only by a virama that a consonant follows (a cluster)
                let clustered = next == Some(VIRAMA) && chars.get(i + 2).is_some_and(|x| ('ക'..='ഹ').contains(x));
                if !clustered {
                    n += 1;
                }
            }
            _ => {}
        }
        // a Latin word: its vowel groups
        if c.is_ascii_alphabetic() && "aeiouyAEIOUY".contains(*c) && !chars.get(i.wrapping_sub(1)).is_some_and(|p| "aeiouyAEIOUY".contains(*p)) {
            n += 1;
        }
    }
    n
}

/// A piece of a script.
#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    Say(String),
    Pause(u32),
}

/// Sentences of normalised text (ending at `.`, `?`, `!`; the mark kept).
pub fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        cur.push(*c);
        let end = matches!(c, '.' | '?' | '!') && chars.get(i + 1).is_none_or(|n| n.is_whitespace()) && !matches!(chars.get(i + 1), Some('.'));
        if end {
            let t = cur.trim().to_string();
            if t.chars().any(char::is_alphanumeric) {
                out.push(t);
            }
            cur.clear();
        }
    }
    let t = cur.trim().trim_end_matches([',', ':', '-']).to_string();
    if t.chars().any(char::is_alphanumeric) {
        out.push(if t.ends_with(['.', '?', '!']) { t } else { format!("{t}.") });
    }
    out
}

/// Split one long sentence at commas (else between words) into parts of at most `max`.
fn split_long(s: &str, max: usize) -> Vec<String> {
    if s.chars().count() <= max {
        return vec![s.to_string()];
    }
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for word in s.split_whitespace() {
        let would = cur.chars().count() + 1 + word.chars().count();
        let at_comma = cur.ends_with([',', ':']);
        if !cur.is_empty() && (would > max || (at_comma && cur.chars().count() > max / 2)) {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(&word.chars().take(max).collect::<String>());
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// A script → utterances of at most `max` characters and pauses, and the notes of its
/// normalisation (see the module docs).
pub fn pieces(text: &str, max: usize) -> Result<(Vec<Piece>, Vec<String>), TtsError> {
    check_text(text)?;
    let max = max.max(20);
    let mut out: Vec<Piece> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    for seg in script::parse(text) {
        match seg {
            Segment::Pause { millis } => out.push(Piece::Pause(millis)),
            Segment::Speech(t) => {
                let n = normalize(&t);
                for x in n.notes {
                    if !notes.contains(&x) {
                        notes.push(x);
                    }
                }
                let mut cur = String::new();
                for s in sentences(&n.text) {
                    for part in split_long(&s, max) {
                        if !cur.is_empty() && cur.chars().count() + 1 + part.chars().count() > max {
                            out.push(Piece::Say(std::mem::take(&mut cur)));
                        }
                        if !cur.is_empty() {
                            cur.push(' ');
                        }
                        cur.push_str(&part);
                    }
                }
                if !cur.is_empty() {
                    out.push(Piece::Say(cur));
                }
            }
        }
        if out.len() > MAX_PIECES {
            return Err(TtsError::TextTooLong);
        }
    }
    if !out.iter().any(|p| matches!(p, Piece::Say(_))) {
        return Err(TtsError::Empty);
    }
    Ok((out, notes))
}

/// What a Malayalam voice would not read as written in `text` (empty: everything reads).
pub fn report(text: &str) -> Vec<String> {
    pieces(text, usize::MAX / 2).map(|(_, n)| n).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_and_detection() {
        assert!(is_tag("ml") && is_tag("ml-IN") && is_tag("ML_in") && !is_tag("en-US") && !is_tag("mlx"));
        assert!(detect("ഇത് ഞാൻ കണ്ടുമുട്ടിയ എല്ലാ റോബോട്ടുകളുമാണ്, 2013 മുതൽ 2025 വരെ."));
        assert!(detect("GITEX-ൽ ഈ ആൻഡ്രോയിഡ്"));
        assert!(!detect("Hello there, robots."));
        assert!(!detect("12345"));
        assert!(!detect("This is English with one word ഇത്"));
    }

    #[test]
    fn numbers_are_malayalam_words() {
        let cases: &[(u64, &str)] = &[
            (0, "പൂജ്യം"),
            (1, "ഒന്ന്"),
            (9, "ഒൻപത്"),
            (10, "പത്ത്"),
            (13, "പതിമൂന്ന്"),
            (19, "പത്തൊൻപത്"),
            (20, "ഇരുപത്"),
            (21, "ഇരുപത്തിയൊന്ന്"),
            (22, "ഇരുപത്തിരണ്ട്"),
            (25, "ഇരുപത്തിയഞ്ച്"),
            (48, "നാൽപ്പത്തിയെട്ട്"),
            (99, "തൊണ്ണൂറ്റിയൊൻപത്"),
            (100, "നൂറ്"),
            (101, "നൂറ്റി ഒന്ന്"),
            (150, "നൂറ്റി അമ്പത്"),
            (500, "അഞ്ഞൂറ്"),
            (999, "തൊള്ളായിരത്തി തൊണ്ണൂറ്റിയൊൻപത്"),
            (1000, "ആയിരം"),
            (1998, "ആയിരത്തി തൊള്ളായിരത്തി തൊണ്ണൂറ്റിയെട്ട്"),
            (2000, "രണ്ടായിരം"),
            (2013, "രണ്ടായിരത്തി പതിമൂന്ന്"),
            (2025, "രണ്ടായിരത്തി ഇരുപത്തിയഞ്ച്"),
            (5000, "അയ്യായിരം"),
            (10_000, "പതിനായിരം"),
            (11_000, "പതിനൊന്നായിരം"),
            (20_000, "ഇരുപതിനായിരം"),
            (25_000, "ഇരുപത്തിയഞ്ചായിരം"),
            (100_000, "ഒരു ലക്ഷം"),
            (250_000, "രണ്ട് ലക്ഷത്തി അമ്പതിനായിരം"),
            (10_000_000, "ഒരു കോടി"),
            (35_000_000, "മൂന്ന് കോടി അമ്പത് ലക്ഷം"),
        ];
        for (n, w) in cases {
            assert_eq!(number(*n), *w, "{n}");
        }
        assert_eq!(ordinal(1), "ഒന്നാം");
        assert_eq!(ordinal(25), "ഇരുപത്തിയഞ്ചാം");
        // never panics, whatever the size
        assert!(!number(u64::MAX).is_empty());
    }

    #[test]
    fn the_test_lines_normalise() {
        let n = normalize("ഇത് ഞാൻ കണ്ടുമുട്ടിയ എല്ലാ റോബോട്ടുകളുമാണ്, 2013 മുതൽ 2025 വരെ.");
        assert_eq!(n.text, "ഇത് ഞാൻ കണ്ടുമുട്ടിയ എല്ലാ റോബോട്ടുകളുമാണ്, രണ്ടായിരത്തി പതിമൂന്ന് മുതൽ രണ്ടായിരത്തി ഇരുപത്തിയഞ്ച് വരെ.");
        assert!(n.notes.is_empty());
        let n = normalize("ദുബായിൽ നടന്ന ജൈടെക്സിൽ ഈ ആൻഡ്രോയിഡ് ഒരു മനുഷ്യനെപ്പോലെ തന്നെ തോന്നി.");
        assert_eq!(n.text, "ദുബായിൽ നടന്ന ജൈടെക്സിൽ ഈ ആൻഡ്രോയിഡ് ഒരു മനുഷ്യനെപ്പോലെ തന്നെ തോന്നി.");
    }

    #[test]
    fn suffixes_ranges_and_symbols() {
        assert_eq!(normalize("2013-ൽ").text, "രണ്ടായിരത്തി പതിമൂന്നിൽ");
        assert_eq!(normalize("2013ൽ").text, "രണ്ടായിരത്തി പതിമൂന്നിൽ");
        assert_eq!(normalize("2-ാം").text, "രണ്ടാം");
        assert_eq!(normalize("1000-ൽ").text, "ആയിരത്തിൽ");
        assert_eq!(normalize("2013-ഓടെ").text, "രണ്ടായിരത്തി പതിമൂന്നോടെ");
        assert_eq!(normalize("2013–2025").text, "രണ്ടായിരത്തി പതിമൂന്ന് മുതൽ രണ്ടായിരത്തി ഇരുപത്തിയഞ്ച് വരെ");
        assert_eq!(normalize("50%").text, "അമ്പത് ശതമാനം");
        assert_eq!(normalize("₹500").text, "അഞ്ഞൂറ് രൂപ");
        assert_eq!(normalize("2.5").text, "രണ്ട് പോയിന്റ് അഞ്ച്");
        assert_eq!(normalize("1,00,000 ആളുകൾ").text, "ഒരു ലക്ഷം ആളുകൾ");
        assert_eq!(normalize("1,000,000").text, "പത്ത് ലക്ഷം");
        assert_eq!(normalize("007").text, "പൂജ്യം പൂജ്യം ഏഴ്");
        assert_eq!(normalize("൨൦൨൫").text, "രണ്ടായിരത്തി ഇരുപത്തിയഞ്ച്");
        assert_eq!(normalize("ഇതും — അതും").text, "ഇതും, അതും");
        assert_eq!(normalize("A & B").text, "എ ആൻഡ് ബി");
    }

    #[test]
    fn latin_acronyms_are_spelled_other_words_reported() {
        assert_eq!(normalize("AI റോബോട്ട്").text, "എഐ റോബോട്ട്");
        assert_eq!(normalize("GITEX").text, "ജിഐടിഇഎക്സ്");
        assert_eq!(normalize("Unitree G1").text, "Unitree ജി വൺ");
        assert_eq!(normalize("R2D2").text, "ആർ ടു ഡി ടു");
        let n = normalize("ഇത് iPhone ആണ്");
        assert_eq!(n.text, "ഇത് iPhone ആണ്");
        assert_eq!(n.notes.len(), 1);
        assert!(n.notes[0].contains("iPhone"));
        let n = normalize("ഹായ് 🙂!");
        assert_eq!(n.text, "ഹായ്!");
        assert!(n.notes.iter().any(|x| x.contains('🙂')));
    }

    #[test]
    fn syllables_are_counted() {
        assert_eq!(syllables("ഇത്"), 2);
        assert_eq!(syllables("കണ്ടുമുട്ടിയ"), 5);
        assert_eq!(syllables("ഞാൻ"), 1);
        assert_eq!(syllables("ആൻഡ്രോയിഡ്"), 4, "aan-ḍroy-i-ḍŭ");
        assert_eq!(syllables("ദുബായിൽ നടന്ന ജൈടെക്സിൽ ഈ ആൻഡ്രോയിഡ് ഒരു മനുഷ്യനെപ്പോലെ തന്നെ തോന്നി."), 26);
        assert_eq!(syllables("robot"), 2);
        assert_eq!(syllables(""), 0);
    }

    #[test]
    fn spelling_is_canonical() {
        // decomposed ൊ, old chillu ന്‍, zero-width joiners
        assert_eq!(canonical("കൊ"), "കൊ");
        assert_eq!(canonical("\u{0D15}\u{0D46}\u{0D3E}"), "\u{0D15}\u{0D4A}");
        assert_eq!(canonical("അവന\u{0D4D}\u{200D}"), "അവൻ");
        assert_eq!(canonical("ക\u{200C}ട"), "കട");
    }

    #[test]
    fn scripts_become_utterances() {
        let (p, notes) = pieces("ഇത് ഒന്നാണ്. [pause 1s] ഇത് രണ്ടാണ്? അതെ!", 200).unwrap();
        assert_eq!(p, vec![Piece::Say("ഇത് ഒന്നാണ്.".into()), Piece::Pause(1000), Piece::Say("ഇത് രണ്ടാണ്? അതെ!".into())]);
        assert!(notes.is_empty());
        // short sentences share an utterance; long ones are split; nothing is lost
        let long = "ഇതൊരു ചെറിയ വാചകമാണ്. ".repeat(30);
        let (p, _) = pieces(&long, 60).unwrap();
        let says: Vec<&String> = p.iter().filter_map(|x| if let Piece::Say(s) = x { Some(s) } else { None }).collect();
        assert!(says.len() > 1 && says.iter().all(|s| s.chars().count() <= 60), "{says:?}");
        assert_eq!(says.iter().map(|s| s.matches("വാചകമാണ്.").count()).sum::<usize>(), 30);
        let one = format!("{}.", (0..60).map(|_| "വാക്ക്,").collect::<Vec<_>>().join(" "));
        let (p, _) = pieces(&one, 50).unwrap();
        assert!(p.iter().all(|x| matches!(x, Piece::Say(s) if s.chars().count() <= 50)));
        // a sentence without its full stop gets one
        assert_eq!(sentences("ഇത് ശരിയാണ്"), vec!["ഇത് ശരിയാണ്."]);
        assert_eq!(sentences("ശരി... ശരി."), vec!["ശരി...", "ശരി."]);
    }

    #[test]
    fn empty_and_hostile_scripts() {
        assert_eq!(pieces("", 100), Err(TtsError::Empty));
        assert_eq!(pieces("[pause 1s] ... !!", 100), Err(TtsError::Empty));
        assert_eq!(pieces(&"അ ".repeat(40_000), 100), Err(TtsError::TextTooLong));
        for t in [
            "\u{0}\u{1}abc",
            "🙂🙂 hi",
            "日本語",
            "[[[pause",
            "1e999999",
            "-0.0.0.0",
            "₹₹₹",
            "99999999999999999999",
            "--",
            "1,2,3,4",
            "2013-",
            "-ൽ",
            "്്്",
            "\u{200D}\u{0D4D}",
        ] {
            let _ = pieces(t, 100);
            let _ = normalize(t);
            let _ = report(t);
        }
    }
}
