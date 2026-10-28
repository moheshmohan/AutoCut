//! The Malayalam text front end: a script written in Malayalam (English loanwords written in
//! Malayalam script too) → utterances a Malayalam voice reads, and notes about anything it could
//! not read as written.
//!
//! - **Spelling** is made canonical: the two-part vowel signs composed (`െ` + `ാ` → `ൊ`, as NFC
//!   does), old-style chillus (consonant + virama + ZWJ) made the atomic chillu letters, other
//!   zero-width joiners dropped, Malayalam digits made ASCII digits.
//! - **Numbers** written in digits are read in one of two styles ([`Numbers`]):
//!   - [`Numbers::English`], the default for narration: years, money and numbers from 100 up
//!     are said the English way in Malayalam script, as Malayali tech creators speak ("2021" →
//!     "ട്വന്റി ട്വന്റി വൺ", "2008" → "ടു തൗസന്റ് എയ്റ്റ്", "$1,600" → "സിക്സ്റ്റീൻ ഹണ്ട്രഡ്
//!     ഡോളർ", "170,000" → "വൺ ലാഖ് സെവന്റി തൗസന്റ്", the Indian system with lakh and crore);
//!     decimals too ("2.5" → "ടു പോയിന്റ് ഫൈവ്"); small counts stay Malayalam words ("2
//!     ദിവസം" → "രണ്ട് ദിവസം"). A four-digit number from 1100 to 2099 without a group comma is
//!     a year ("1969" → "നൈന്റീൻ സിക്സ്റ്റി നൈൻ"): write a count with its comma ("1,234").
//!   - [`Numbers::Malayalam`]: Malayalam words in the Indian system (ആയിരം, ലക്ഷം, കോടി):
//!     "2013" → "രണ്ടായിരത്തി പതിമൂന്ന്"; years are read as numbers, as written Malayalam does.
//!
//!   A suffix written after a number joins its last word with the sandhi Malayalam writes
//!   ("2013-ൽ" → "…പതിമൂന്നിൽ" or "ട്വന്റി തേർട്ടീനിൽ", "2021-ൽ" → "ട്വന്റി ട്വന്റി വണ്ണിൽ",
//!   "2020-ലും" → "ട്വന്റി ട്വന്റിയിലും", "2-ാം" → "രണ്ടാം"); a range "2013–2025" → "… മുതൽ …
//!   വരെ"; "50%" → "… ശതമാനം"; "₹500" → "… രൂപ" (a currency symbol or word makes it money);
//!   decimals with "പോയിന്റ്" and their digits one by one; digit strings longer than twelve, or
//!   with a leading zero, digit by digit; digits joined to capitals are a code ("4K" → "ഫോർ
//!   കെ"). [`spell_numbers`] rewrites only the numbers of a script (pause markers kept), so a
//!   caller can choose the style before the voice reads it.
//! - **Latin letters**: an acronym (all capitals, at most eight) is read letter by letter with
//!   the letters' Malayalam names ("AI" → "എഐ"), digits inside a code with their English names
//!   in Malayalam script ("G1" → "ജി വൺ"); any other Latin word is kept, and reported (the model
//!   was trained on Indian-language text: write such words in Malayalam script).
//! - **Symbols** with a spoken form are spoken (`&` ആൻഡ്, `+` പ്ലസ്, `@` അറ്റ്); dashes become
//!   commas; emoji and other symbols are dropped and reported.
//! - [`pieces`]: pause markers (`[pause 1s]`) become exact silence; sentences (`.` `?` `!` `।`)
//!   are grouped into utterances of at most the given length, a long sentence split at commas,
//!   else between words.
//! - [`parts`]: how a cloned voice reads a script — as [`pieces`], and also style tags
//!   (`[excited] … [calm] …`, never read aloud: each utterance carries the style in force),
//!   `[breath]`, and line breaks (an utterance never spans one; the pause there is longer).

use crate::script::{self, Mark, Segment};
use crate::styles::{Asked, Boundary, Tag};
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

/// How numbers written in digits are read in a Malayalam script (see the module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Numbers {
    /// Years, money, decimals and numbers from 100 up said in English, in Malayalam script, the
    /// Indian way (lakh, crore); small counts in Malayalam words. The default for narration.
    #[default]
    English,
    /// Every number in Malayalam words.
    Malayalam,
}

impl Numbers {
    pub const ALL: [Numbers; 2] = [Numbers::English, Numbers::Malayalam];

    /// The parameter id (`english`, `malayalam`).
    pub fn id(self) -> &'static str {
        match self {
            Numbers::English => "english",
            Numbers::Malayalam => "malayalam",
        }
    }

    pub fn from_id(s: &str) -> Option<Numbers> {
        let s = s.trim().to_ascii_lowercase();
        Self::ALL.into_iter().find(|n| n.id() == s)
    }
}

/// English number names in Malayalam script, 0–19.
const EN_UNITS: [&str; 20] = [
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
/// English tens in Malayalam script.
const EN_TENS: [&str; 10] = ["", "", "ട്വന്റി", "തേർട്ടി", "ഫോർട്ടി", "ഫിഫ്റ്റി", "സിക്സ്റ്റി", "സെവന്റി", "എയ്റ്റി", "നൈന്റി"];
const EN_HUNDRED: &str = "ഹണ്ട്രഡ്";
const EN_THOUSAND: &str = "തൗസന്റ്";
const EN_LAKH: &str = "ലാഖ്";
const EN_CRORE: &str = "ക്രോർ";
const EN_POINT: &str = "പോയിന്റ്";
/// "Oh", as in "nineteen oh five".
const EN_OH: &str = "ഓ";

fn en_unit(n: u64) -> &'static str {
    EN_UNITS.get(n as usize).copied().unwrap_or("")
}

fn english_below_100(n: u64) -> String {
    let n = n % 100;
    if n < 20 {
        return en_unit(n).to_string();
    }
    let tens = EN_TENS.get((n / 10) as usize).copied().unwrap_or("");
    if n.is_multiple_of(10) { tens.to_string() } else { format!("{tens} {}", en_unit(n % 10)) }
}

fn english_below_1000(n: u64) -> String {
    let n = n % 1000;
    let (h, r) = (n / 100, n % 100);
    match (h, r) {
        (0, r) => english_below_100(r),
        (h, 0) => format!("{} {EN_HUNDRED}", en_unit(h)),
        (h, r) => format!("{} {EN_HUNDRED} {}", en_unit(h), english_below_100(r)),
    }
}

/// `n` said in English, in Malayalam script, the Indian way ("170000" → "വൺ ലാഖ് സെവന്റി
/// തൗസന്റ്"); round hundreds from 1100 to 1900 as hundreds ("1600" → "സിക്സ്റ്റീൻ ഹണ്ട്രഡ്").
pub fn english_number(n: u64) -> String {
    if n == 0 {
        return en_unit(0).to_string();
    }
    if (1100..=1999).contains(&n) && n.is_multiple_of(100) {
        return format!("{} {EN_HUNDRED}", en_unit(n / 100));
    }
    let crore = n / 10_000_000;
    let lakh = (n / 100_000) % 100;
    let thousand = (n / 1000) % 100;
    let rest = n % 1000;
    let mut parts: Vec<String> = Vec::new();
    if crore > 0 {
        // crore < n: the recursion ends
        parts.push(format!("{} {EN_CRORE}", english_number(crore)));
    }
    if lakh > 0 {
        parts.push(format!("{} {EN_LAKH}", english_below_100(lakh)));
    }
    if thousand > 0 {
        parts.push(format!("{} {EN_THOUSAND}", english_below_100(thousand)));
    }
    if rest > 0 {
        parts.push(english_below_1000(rest));
    }
    parts.join(" ")
}

/// A year said in English, in Malayalam script: "2021" → "ട്വന്റി ട്വന്റി വൺ", "2008" → "ടു
/// തൗസന്റ് എയ്റ്റ്", "1969" → "നൈന്റീൻ സിക്സ്റ്റി നൈൻ", "1905" → "നൈന്റീൻ ഓ ഫൈവ്", "1900" →
/// "നൈന്റീൻ ഹണ്ട്രഡ്"; other numbers as [`english_number`].
pub fn english_year(n: u64) -> String {
    match n {
        2010..=2099 => format!("{} {}", EN_TENS[2], english_below_100(n % 100)),
        1100..=1999 => {
            let (hi, lo) = (en_unit(n / 100), n % 100);
            match lo {
                0 => format!("{hi} {EN_HUNDRED}"),
                1..=9 => format!("{hi} {EN_OH} {}", en_unit(lo)),
                _ => format!("{hi} {}", english_below_100(lo)),
            }
        }
        _ => english_number(n),
    }
}

/// Digits read one by one with their English names, in Malayalam script.
fn english_digits(d: &str) -> String {
    d.chars().filter_map(|c| c.to_digit(10)).map(|v| en_unit(u64::from(v))).collect::<Vec<_>>().join(" ")
}

/// A four-digit number without a group comma, from 1100 to 2099: read as a year.
fn is_year(int: &str, n: u64) -> bool {
    int.len() == 4 && !int.contains(',') && (1100..=2099).contains(&n)
}

/// The words of a number written `int` (digits, group commas) and `frac` (decimals, may be
/// empty) in `style`; `money` when a currency goes with it.
fn number_words(int: &str, frac: &str, money: bool, style: Numbers) -> String {
    let n = parse_number(int);
    match style {
        Numbers::Malayalam => {
            let words = n.map(number).unwrap_or_else(|| digit_words(int));
            if frac.is_empty() { words } else { format!("{words} {EN_POINT} {}", digit_words(frac)) }
        }
        Numbers::English => {
            let small = n.is_some_and(|v| v < 100);
            let words = match n {
                // small counts stay Malayalam ("രണ്ട് ദിവസം"), unless money or a decimal
                Some(v) if small && !money && frac.is_empty() => return number(v),
                Some(v) if !money && frac.is_empty() && is_year(int, v) => english_year(v),
                Some(v) => english_number(v),
                None => english_digits(int),
            };
            if frac.is_empty() { words } else { format!("{words} {EN_POINT} {}", english_digits(frac)) }
        }
    }
}

/// Join a suffix written after a number (`-ൽ`, `-ാം`, `-ഓടെ`…) to its word, with the sandhi
/// Malayalam writes.
fn attach(word: &str, suffix: &str) -> String {
    let Some(first) = suffix.chars().next() else { return word.to_string() };
    let sign = vowel_sign(first);
    let rest: String = suffix.chars().skip(1).collect();
    if let Some(stem) = word.strip_suffix(VIRAMA) {
        return match (first, sign) {
            // a vowel sign replaces the virama's half-vowel
            (c, _) if is_vowel_sign(c) => format!("{stem}{suffix}"),
            // an independent vowel becomes its sign
            (_, Some(s)) => format!("{stem}{s}{rest}"),
            // a consonant or chillu: the half-vowel becomes ി ("…മൂന്നിൽ")
            _ => format!("{stem}ി{suffix}"),
        };
    }
    if let Some(stem) = word.strip_suffix('ം') {
        // ആയിരം + ൽ → ആയിരത്തിൽ
        return match (first, sign) {
            (c, _) if is_vowel_sign(c) => format!("{stem}ത്ത{suffix}"),
            (_, Some(s)) => format!("{stem}ത്ത{s}{rest}"),
            _ => format!("{stem}ത്തി{suffix}"),
        };
    }
    let Some(last) = word.chars().last() else { return suffix.to_string() };
    let stem = word.get(..word.len().saturating_sub(last.len_utf8())).unwrap_or("");
    if let Some(base) = chillu_base(last) {
        // a chillu takes its consonant back: "ഫോർ" + "ൽ" → "ഫോറിൽ"; after the short vowel of a
        // one-syllable word it doubles: "വൺ" + "ൽ" → "വണ്ണിൽ", "ടെൻ" → "ടെന്നിൽ"
        let cons = if base != 'റ' && short_syllable(stem) { format!("{base}{VIRAMA}{base}") } else { base.to_string() };
        return match (first, sign) {
            (c, _) if is_vowel_sign(c) => format!("{stem}{cons}{suffix}"),
            (_, Some(s)) => format!("{stem}{cons}{s}{rest}"),
            _ => format!("{stem}{cons}ി{suffix}"),
        };
    }
    if is_vowel_sign(last) || vowel_sign(last).is_some() {
        // a vowel ending takes a glide: "ട്വന്റി" + "ൽ" → "ട്വന്റിയിൽ", "ടു" + "ൽ" → "ടുവിൽ"
        let glide = if matches!(last, 'ു' | 'ൂ' | 'ഉ' | 'ഊ') { 'വ' } else { 'യ' };
        // the genitive after a vowel is -ുടെ ("ട്വന്റിയുടെ"), not -ന്റെ
        if let Some(after) = suffix.strip_prefix("ന്റെ") {
            return format!("{word}{glide}ുടെ{after}");
        }
        return match (first, sign) {
            (c, _) if is_vowel_sign(c) => format!("{word}{glide}{suffix}"),
            (_, Some(s)) => format!("{word}{glide}{s}{rest}"),
            _ => format!("{word}{glide}ി{suffix}"),
        };
    }
    format!("{word}{suffix}")
}

/// The consonant a chillu stands for (`ൻ` → `ന`; `ർ` → `റ`, as loanwords take it).
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

/// `stem` (a word without its final chillu) is one syllable with a short vowel ("വ", "ടെ").
fn short_syllable(stem: &str) -> bool {
    let long = ['ാ', 'ീ', 'ൂ', 'േ', 'ൈ', 'ോ', 'ൌ', 'ൗ', 'ആ', 'ഈ', 'ഊ', 'ഏ', 'ഐ', 'ഓ', 'ഔ'];
    syllables(stem) == 1 && !stem.chars().any(|c| long.contains(&c))
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

/// Normalise one stretch of speech (no pause markers) for a Malayalam voice, numbers in the
/// narration default ([`Numbers::English`]).
pub fn normalize(text: &str) -> Normalized {
    normalize_with(text, Numbers::default())
}

/// Normalise one stretch of speech (no pause markers) for a Malayalam voice, numbers in `style`.
pub fn normalize_with(text: &str, style: Numbers) -> Normalized {
    rewrite(text, style, false)
}

/// `text` with only its numbers spelled out in `style` (and its spelling made canonical):
/// everything else, pause markers and style tags included, stays as written, for the voice's
/// front end to read. Lets a caller choose how numbers are said before the voice reads the script.
pub fn spell_numbers(text: &str, style: Numbers) -> String {
    let marks = script::parse_marked(text);
    // line breaks between markers matter only to a script with style tags (and a script without
    // keeps the text it always had: its narrations keep their cache keys)
    let tagged = marks.iter().any(|m| matches!(m, Mark::Tag(_)));
    let mut out = String::with_capacity(text.len() * 2);
    let marker = |out: &mut String, m: &str| {
        if !out.is_empty() && !out.ends_with(char::is_whitespace) {
            out.push(' ');
        }
        out.push_str(m);
    };
    for m in marks {
        match m {
            Mark::Speech(t) if !tagged && t.trim().is_empty() => {}
            Mark::Speech(t) => out.push_str(&rewrite(&t, style, true).text),
            Mark::Pause { millis } => marker(&mut out, &format!("[pause {millis}ms]")),
            Mark::Tag(Tag::Style(s)) => marker(&mut out, &format!("[{s}]")),
            Mark::Tag(Tag::Plain) => marker(&mut out, "[plain]"),
            Mark::Tag(Tag::Breath) => marker(&mut out, "[breath]"),
        }
    }
    out
}

/// The currency word of a symbol written before an amount.
fn currency_word(c: char) -> Option<&'static str> {
    Some(match c {
        '₹' => "രൂപ",
        '$' => "ഡോളർ",
        '€' => "യൂറോ",
        '£' => "പൗണ്ട്",
        _ => return None,
    })
}

/// Words after an amount that make it money ("50 ഡോളർ"): read in English with
/// [`Numbers::English`] whatever its size.
const CURRENCY_WORDS: [&str; 8] = ["ഡോളർ", "ഡോളറ", "രൂപ", "ദിർഹം", "ദിർഹ", "യൂറോ", "പൗണ്ട", "ദിനാർ"];

/// Whether the word after `i` (past spaces) is a currency.
fn money_follows(chars: &[char], mut i: usize) -> bool {
    while chars.get(i).is_some_and(|c| *c == ' ') {
        i += 1;
    }
    let word: String = chars.get(i..).unwrap_or(&[]).iter().take_while(|c| is_malayalam(**c)).collect();
    CURRENCY_WORDS.iter().any(|w| word.starts_with(w))
}

/// The number written at `chars[i]` (a digit) read in `style`, with what goes with it: a
/// currency (`currency`, its symbol written before), a decimal part, a range, a percent sign, a
/// suffix. Returns the words and where the number ends.
fn read_number(chars: &[char], mut i: usize, currency: Option<&str>, style: Numbers) -> (String, usize) {
    // digits with group commas and one decimal point
    let s = i;
    while chars.get(i).is_some_and(|c| c.is_ascii_digit() || (*c == ',' && chars.get(i + 1).is_some_and(char::is_ascii_digit))) {
        i += 1;
    }
    let int: String = chars.get(s..i).unwrap_or(&[]).iter().collect();
    let mut frac = String::new();
    if chars.get(i) == Some(&'.') && chars.get(i + 1).is_some_and(char::is_ascii_digit) {
        i += 1;
        while let Some(c) = chars.get(i).filter(|c| c.is_ascii_digit()) {
            frac.push(*c);
            i += 1;
        }
    }
    let money = currency.is_some() || money_follows(chars, i);
    let mut words = number_words(&int, &frac, money, style);
    // a range: 2013–2025
    if currency.is_none() && frac.is_empty() && matches!(chars.get(i), Some('-' | '–')) && chars.get(i + 1).is_some_and(char::is_ascii_digit) {
        let s2 = i + 1;
        let mut j = s2;
        while chars.get(j).is_some_and(char::is_ascii_digit) {
            j += 1;
        }
        let second: String = chars.get(s2..j).unwrap_or(&[]).iter().collect();
        let w2 = number_words(&second, "", money, style);
        return (format!("{words} മുതൽ {w2} വരെ"), j);
    }
    // a percentage
    if chars.get(i) == Some(&'%') {
        return (format!("{words} ശതമാനം"), i + 1);
    }
    if let Some(c) = currency {
        words = format!("{words} {c}");
    }
    // a suffix written after the number: "2013-ൽ", "2-ാം", "2013ൽ", "$1,600-നാണ്"
    let hyphen = usize::from(matches!(chars.get(i), Some('-')) && chars.get(i + 1).is_some_and(|c| is_malayalam(*c)));
    if chars.get(i + hyphen).is_some_and(|c| is_malayalam(*c)) {
        let s3 = i + hyphen;
        let mut j = s3;
        while chars.get(j).is_some_and(|c| is_malayalam(*c)) {
            j += 1;
        }
        let suffix: String = chars.get(s3..j).unwrap_or(&[]).iter().collect();
        // the suffix joins the number's last word
        let joined = match words.rsplit_once(' ') {
            Some((head, last)) => format!("{head} {}", attach(last, &suffix)),
            None => attach(&words, &suffix),
        };
        return (joined, j);
    }
    (words, i)
}

/// Normalise `text` for a Malayalam voice (`numbers_only`: spell its numbers, keep the rest as
/// written).
fn rewrite(text: &str, style: Numbers, numbers_only: bool) -> Normalized {
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
        // an amount after a currency symbol: "$1,600", "₹ 500"
        if let Some(word) = currency_word(c) {
            let mut j = i + 1;
            while chars.get(j).is_some_and(|c| *c == ' ') {
                j += 1;
            }
            if chars.get(j).is_some_and(char::is_ascii_digit) {
                let (words, next) = read_number(&chars, j, Some(word), style);
                push_word(&mut out, &words);
                i = next;
                continue;
            }
        }
        if c.is_ascii_digit() {
            // digits joined to capitals are a code ("4K", "3D"): read as one
            let mut j = i;
            while chars.get(j).is_some_and(char::is_ascii_digit) {
                j += 1;
            }
            if chars.get(j).is_some_and(char::is_ascii_uppercase) {
                let s = i;
                while chars.get(i).is_some_and(char::is_ascii_alphanumeric) {
                    i += 1;
                }
                let word: String = chars.get(s..i).unwrap_or(&[]).iter().collect();
                if numbers_only {
                    out.push_str(&word);
                } else {
                    push_word(&mut out, &latin(&word, &mut note));
                }
                continue;
            }
            let (words, next) = read_number(&chars, i, None, style);
            push_word(&mut out, &words);
            i = next;
            continue;
        }
        if c.is_ascii_alphabetic() {
            let s = i;
            while chars.get(i).is_some_and(char::is_ascii_alphanumeric) {
                i += 1;
            }
            let word: String = chars.get(s..i).unwrap_or(&[]).iter().collect();
            if numbers_only {
                out.push_str(&word);
            } else {
                push_word(&mut out, &latin(&word, &mut note));
            }
            continue;
        }
        i += 1;
        if numbers_only {
            out.push(c);
            continue;
        }
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
            '°' => push_word(&mut out, "ഡിഗ്രി"),
            '=' => push_word(&mut out, "സമം"),
            c if currency_word(c).is_some() => push_word(&mut out, currency_word(c).unwrap_or("")),
            '/' | '|' | '_' | '*' | '#' | '~' | '^' | '`' | '<' | '>' | '[' | ']' | '{' | '}' | '\\' => {
                if !out.is_empty() && !out.ends_with(' ') {
                    out.push(' ');
                }
            }
            c if c.is_control() => {}
            c => note(format!("left out `{c}` (not readable as written)")),
        }
    }
    if numbers_only {
        return Normalized { text: out, notes };
    }
    // a currency word read before its amount goes after it, as it is spoken
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
        || EN_UNITS.iter().chain(&EN_TENS).any(|u| !u.is_empty() && w == *u)
        || [EN_HUNDRED, EN_THOUSAND, EN_LAKH, EN_CRORE, EN_OH].contains(&w)
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
    pieces_with(text, max, Numbers::default())
}

/// [`pieces`] with numbers read in `style`.
pub fn pieces_with(text: &str, max: usize, style: Numbers) -> Result<(Vec<Piece>, Vec<String>), TtsError> {
    split_script(text, max, style, true)
}

/// [`pieces`], every sentence an utterance of its own (a long one still split at commas).
pub fn sentence_pieces(text: &str, max: usize) -> Result<(Vec<Piece>, Vec<String>), TtsError> {
    split_script(text, max, Numbers::default(), false)
}

/// The utterances of normalised `text`: sentences packed into utterances of at most `max`
/// characters (`pack`), or each its own; a sentence longer than `max` split at commas, else
/// between words.
fn utterances(text: &str, max: usize, pack: bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for s in sentences(text) {
        if !pack && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        for part in split_long(&s, max) {
            if !cur.is_empty() && cur.chars().count() + 1 + part.chars().count() > max {
                out.push(std::mem::take(&mut cur));
            }
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(&part);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Add `new` notes to `notes`, each once.
fn merge_notes(notes: &mut Vec<String>, new: Vec<String>) {
    for x in new {
        if !notes.contains(&x) {
            notes.push(x);
        }
    }
}

/// A script → utterances and pauses; `pack`: sentences share an utterance while they fit.
fn split_script(text: &str, max: usize, style: Numbers, pack: bool) -> Result<(Vec<Piece>, Vec<String>), TtsError> {
    check_text(text)?;
    let max = max.max(20);
    let mut out: Vec<Piece> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    for seg in script::parse(text) {
        match seg {
            Segment::Pause { millis } => out.push(Piece::Pause(millis)),
            Segment::Speech(t) => {
                let n = normalize_with(&t, style);
                merge_notes(&mut notes, n.notes);
                out.extend(utterances(&n.text, max, pack).into_iter().map(Piece::Say));
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

/// A piece of a script read with voice styles ([`parts`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Part {
    /// An utterance, and the style its tags put in force.
    Say { text: String, asked: Asked },
    /// Exact silence (a pause marker), ms.
    Pause(u32),
    /// A breath-length pause (`[breath]`).
    Breath,
    /// A line break ([`Boundary::Line`]) or a blank line ([`Boundary::Paragraph`]) between the
    /// utterances around it.
    Break(Boundary),
}

/// A script → utterances, each with the style its tags put in force ([`crate::styles`]: tags
/// are never read aloud), pauses, breaths and line breaks; and the notes of its normalisation.
/// An utterance never spans a style tag or a line break; within them, sentences are packed into
/// utterances of at most `max(style)` characters (`pack`: the reference each style reads after
/// sets how much one generation holds) or each sentence is its own.
pub fn parts(text: &str, max: &dyn Fn(Asked) -> usize, style: Numbers, pack: bool) -> Result<(Vec<Part>, Vec<String>), TtsError> {
    check_text(text)?;
    let mut out: Vec<Part> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    let mut asked = Asked::Line;
    // line breaks since the last words
    let mut breaks = 0usize;
    for m in script::parse_marked(text) {
        match m {
            Mark::Pause { millis } => {
                out.push(Part::Pause(millis));
                breaks = 0;
            }
            Mark::Tag(Tag::Breath) => {
                out.push(Part::Breath);
                breaks = 0;
            }
            Mark::Tag(Tag::Plain) => asked = Asked::Plain,
            Mark::Tag(Tag::Style(s)) => asked = Asked::Style(s),
            Mark::Speech(t) => {
                let limit = max(asked).max(20);
                for (k, line) in t.split('\n').enumerate() {
                    if k > 0 {
                        breaks = breaks.saturating_add(1);
                    }
                    if line.trim().is_empty() {
                        continue;
                    }
                    let n = normalize_with(line, style);
                    merge_notes(&mut notes, n.notes);
                    let says = utterances(&n.text, limit, pack);
                    if says.is_empty() {
                        continue;
                    }
                    if breaks > 0 && matches!(out.last(), Some(Part::Say { .. })) {
                        out.push(Part::Break(if breaks >= 2 { Boundary::Paragraph } else { Boundary::Line }));
                    }
                    breaks = 0;
                    out.extend(says.into_iter().map(|text| Part::Say { text, asked }));
                    if out.len() > MAX_PIECES {
                        return Err(TtsError::TextTooLong);
                    }
                }
            }
        }
        if out.len() > MAX_PIECES {
            return Err(TtsError::TextTooLong);
        }
    }
    if !out.iter().any(|p| matches!(p, Part::Say { .. })) {
        return Err(TtsError::Empty);
    }
    Ok((out, notes))
}

/// What a Malayalam voice would not read as written in `text` (empty: everything reads; style
/// tags are not words).
pub fn report(text: &str) -> Vec<String> {
    parts(text, &|_| usize::MAX / 2, Numbers::default(), true).map(|(_, n)| n).unwrap_or_default()
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
        let ml = |t: &str| normalize_with(t, Numbers::Malayalam);
        let n = ml("ഇത് ഞാൻ കണ്ടുമുട്ടിയ എല്ലാ റോബോട്ടുകളുമാണ്, 2013 മുതൽ 2025 വരെ.");
        assert_eq!(n.text, "ഇത് ഞാൻ കണ്ടുമുട്ടിയ എല്ലാ റോബോട്ടുകളുമാണ്, രണ്ടായിരത്തി പതിമൂന്ന് മുതൽ രണ്ടായിരത്തി ഇരുപത്തിയഞ്ച് വരെ.");
        assert!(n.notes.is_empty());
        // the narration default says the years in English
        let n = normalize("ഇത് ഞാൻ കണ്ടുമുട്ടിയ എല്ലാ റോബോട്ടുകളുമാണ്, 2013 മുതൽ 2025 വരെ.");
        assert_eq!(n.text, "ഇത് ഞാൻ കണ്ടുമുട്ടിയ എല്ലാ റോബോട്ടുകളുമാണ്, ട്വന്റി തേർട്ടീൻ മുതൽ ട്വന്റി ട്വന്റി ഫൈവ് വരെ.");
        let n = normalize("ദുബായിൽ നടന്ന ജൈടെക്സിൽ ഈ ആൻഡ്രോയിഡ് ഒരു മനുഷ്യനെപ്പോലെ തന്നെ തോന്നി.");
        assert_eq!(n.text, "ദുബായിൽ നടന്ന ജൈടെക്സിൽ ഈ ആൻഡ്രോയിഡ് ഒരു മനുഷ്യനെപ്പോലെ തന്നെ തോന്നി.");
    }

    #[test]
    fn suffixes_ranges_and_symbols() {
        let ml = |t: &str| normalize_with(t, Numbers::Malayalam).text;
        assert_eq!(ml("2013-ൽ"), "രണ്ടായിരത്തി പതിമൂന്നിൽ");
        assert_eq!(ml("2013ൽ"), "രണ്ടായിരത്തി പതിമൂന്നിൽ");
        assert_eq!(ml("2-ാം"), "രണ്ടാം");
        assert_eq!(ml("1000-ൽ"), "ആയിരത്തിൽ");
        assert_eq!(ml("2013-ഓടെ"), "രണ്ടായിരത്തി പതിമൂന്നോടെ");
        assert_eq!(ml("2013–2025"), "രണ്ടായിരത്തി പതിമൂന്ന് മുതൽ രണ്ടായിരത്തി ഇരുപത്തിയഞ്ച് വരെ");
        assert_eq!(ml("50%"), "അമ്പത് ശതമാനം");
        assert_eq!(ml("₹500"), "അഞ്ഞൂറ് രൂപ");
        assert_eq!(ml("₹ 500"), "അഞ്ഞൂറ് രൂപ");
        assert_eq!(ml("2.5"), "രണ്ട് പോയിന്റ് അഞ്ച്");
        assert_eq!(ml("1,00,000 ആളുകൾ"), "ഒരു ലക്ഷം ആളുകൾ");
        assert_eq!(ml("1,000,000"), "പത്ത് ലക്ഷം");
        assert_eq!(ml("007"), "പൂജ്യം പൂജ്യം ഏഴ്");
        assert_eq!(ml("൨൦൨൫"), "രണ്ടായിരത്തി ഇരുപത്തിയഞ്ച്");
        // a word ending in a vowel takes a glide before the suffix
        assert_eq!(ml("1,00,00,000-ൽ"), "ഒരു കോടിയിൽ");
        assert_eq!(normalize("ഇതും — അതും").text, "ഇതും, അതും");
        assert_eq!(normalize("A & B").text, "എ ആൻഡ് ബി");
    }

    #[test]
    fn english_numbers_in_malayalam_script() {
        let cases: &[(u64, &str)] = &[
            (0, "സീറോ"),
            (7, "സെവൻ"),
            (12, "ട്വെൽവ്"),
            (19, "നൈന്റീൻ"),
            (20, "ട്വന്റി"),
            (85, "എയ്റ്റി ഫൈവ്"),
            (100, "വൺ ഹണ്ട്രഡ്"),
            (600, "സിക്സ് ഹണ്ട്രഡ്"),
            (999, "നൈൻ ഹണ്ട്രഡ് നൈന്റി നൈൻ"),
            (1000, "വൺ തൗസന്റ്"),
            (1234, "വൺ തൗസന്റ് ടു ഹണ്ട്രഡ് തേർട്ടി ഫോർ"),
            (1600, "സിക്സ്റ്റീൻ ഹണ്ട്രഡ്"),
            (2500, "ടു തൗസന്റ് ഫൈവ് ഹണ്ട്രഡ്"),
            (2700, "ടു തൗസന്റ് സെവൻ ഹണ്ട്രഡ്"),
            (27_000, "ട്വന്റി സെവൻ തൗസന്റ്"),
            (100_000, "വൺ ലാഖ്"),
            (170_000, "വൺ ലാഖ് സെവന്റി തൗസന്റ്"),
            (10_000_000, "വൺ ക്രോർ"),
            (35_000_000, "ത്രീ ക്രോർ ഫിഫ്റ്റി ലാഖ്"),
        ];
        for (n, w) in cases {
            assert_eq!(english_number(*n), *w, "{n}");
        }
        let years: &[(u64, &str)] = &[
            (1969, "നൈന്റീൻ സിക്സ്റ്റി നൈൻ"),
            (1900, "നൈന്റീൻ ഹണ്ട്രഡ്"),
            (1905, "നൈന്റീൻ ഓ ഫൈവ്"),
            (2000, "ടു തൗസന്റ്"),
            (2008, "ടു തൗസന്റ് എയ്റ്റ്"),
            (2010, "ട്വന്റി ടെൻ"),
            (2013, "ട്വന്റി തേർട്ടീൻ"),
            (2016, "ട്വന്റി സിക്സ്റ്റീൻ"),
            (2020, "ട്വന്റി ട്വന്റി"),
            (2021, "ട്വന്റി ട്വന്റി വൺ"),
            (2025, "ട്വന്റി ട്വന്റി ഫൈവ്"),
        ];
        for (n, w) in years {
            assert_eq!(english_year(*n), *w, "{n}");
        }
        // never panics, whatever the size
        assert!(!english_number(u64::MAX).is_empty());
        assert!(!english_year(u64::MAX).is_empty());
    }

    #[test]
    fn english_is_the_narration_default() {
        let en = |t: &str| normalize(t).text;
        // the user's examples
        assert_eq!(en("2021"), "ട്വന്റി ട്വന്റി വൺ");
        assert_eq!(en("2013"), "ട്വന്റി തേർട്ടീൻ");
        assert_eq!(en("2008"), "ടു തൗസന്റ് എയ്റ്റ്");
        assert_eq!(en("1969"), "നൈന്റീൻ സിക്സ്റ്റി നൈൻ");
        assert_eq!(en("$1,600"), "സിക്സ്റ്റീൻ ഹണ്ട്രഡ് ഡോളർ");
        assert_eq!(en("2,500"), "ടു തൗസന്റ് ഫൈവ് ഹണ്ട്രഡ്");
        assert_eq!(en("170,000"), "വൺ ലാഖ് സെവന്റി തൗസന്റ്");
        assert_eq!(en("2021-ൽ"), "ട്വന്റി ട്വന്റി വണ്ണിൽ");
        // suffixes after English words: chillus, glides, the genitive
        assert_eq!(en("2013-ൽ"), "ട്വന്റി തേർട്ടീനിൽ");
        assert_eq!(en("2024-ൽ"), "ട്വന്റി ട്വന്റി ഫോറിൽ");
        assert_eq!(en("2023-ൽ"), "ട്വന്റി ട്വന്റി ത്രീയിൽ");
        assert_eq!(en("2022-ൽ"), "ട്വന്റി ട്വന്റി ടുവിൽ");
        assert_eq!(en("2020-ലും"), "ട്വന്റി ട്വന്റിയിലും");
        assert_eq!(en("2020-ന്റെ"), "ട്വന്റി ട്വന്റിയുടെ");
        assert_eq!(en("2021-ന്റെ"), "ട്വന്റി ട്വന്റി വണ്ണിന്റെ");
        assert_eq!(en("2013-ഓടെ"), "ട്വന്റി തേർട്ടീനോടെ");
        assert_eq!(en("2025-ൽ"), "ട്വന്റി ട്വന്റി ഫൈവിൽ");
        assert_eq!(en("2007-ൽ"), "ടു തൗസന്റ് സെവനിൽ");
        assert_eq!(en("170,000-ലധികം"), "വൺ ലാഖ് സെവന്റി തൗസന്റിലധികം");
        assert_eq!(en("$2,700-നാണ്"), "ടു തൗസന്റ് സെവൻ ഹണ്ട്രഡ് ഡോളറിനാണ്");
        // money is English at any size, small counts stay Malayalam
        assert_eq!(en("$5"), "ഫൈവ് ഡോളർ");
        assert_eq!(en("₹500"), "ഫൈവ് ഹണ്ട്രഡ് രൂപ");
        assert_eq!(en("50 ഡോളർ"), "ഫിഫ്റ്റി ഡോളർ");
        assert_eq!(en("2 ദിവസം"), "രണ്ട് ദിവസം");
        assert_eq!(en("85%"), "എൺപത്തിയഞ്ച് ശതമാനം");
        assert_eq!(en("2-ാം"), "രണ്ടാം");
        // decimals, codes, digit strings
        assert_eq!(en("2.5"), "ടു പോയിന്റ് ഫൈവ്");
        assert_eq!(en("007"), "സീറോ സീറോ സെവൻ");
        assert_eq!(en("4K ക്യാമറ"), "ഫോർ കെ ക്യാമറ");
        assert_eq!(en("3D"), "ത്രീ ഡി");
        assert_eq!(en("1,234"), "വൺ തൗസന്റ് ടു ഹണ്ട്രഡ് തേർട്ടി ഫോർ");
        assert_eq!(en("1,00,00,000"), "വൺ ക്രോർ");
        assert_eq!(en("2100"), "ടു തൗസന്റ് വൺ ഹണ്ട്രഡ്");
        assert_eq!(en("2013–2025"), "ട്വന്റി തേർട്ടീൻ മുതൽ ട്വന്റി ട്വന്റി ഫൈവ് വരെ");
        assert_eq!(en("൨൦൨൫"), "ട്വന്റി ട്വന്റി ഫൈവ്");
        assert_eq!(Numbers::from_id(" English "), Some(Numbers::English));
        assert_eq!(Numbers::from_id("malayalam"), Some(Numbers::Malayalam));
        assert_eq!(Numbers::from_id("roman"), None);
        assert_eq!(Numbers::default(), Numbers::English);
    }

    /// The Malayalam script of "Every Robot I've Met" (v5), whose numbers were converted to
    /// English words by hand, with digits put back where they were: the front end must say them
    /// as the hand conversion does.
    const HAND_CONVERTED: &[(&str, &str)] = &[
        (
            "2013 മുതൽ 2025 വരെ ഞാൻ കണ്ടുമുട്ടിയ എല്ലാ റോബോട്ടുകളും ആണ് ഇത്. ഇതിൽ മിക്കതിനെയും ദുബായിലെ എക്സ്പോ 2020-ലും ജൈടെക്സിലും ആണ് കണ്ടത്, ആദ്യത്തെ രണ്ടെണ്ണം ഞാൻ തന്നെ ഉണ്ടാക്കിയതാണ്.",
            "ട്വന്റി തേർട്ടീൻ മുതൽ ട്വന്റി ട്വന്റി ഫൈവ് വരെ ഞാൻ കണ്ടുമുട്ടിയ എല്ലാ റോബോട്ടുകളും ആണ് ഇത്. ഇതിൽ മിക്കതിനെയും ദുബായിലെ എക്സ്പോ ട്വന്റി ട്വന്റിയിലും ജൈടെക്സിലും ആണ് കണ്ടത്, ആദ്യത്തെ രണ്ടെണ്ണം ഞാൻ തന്നെ ഉണ്ടാക്കിയതാണ്.",
        ),
        (
            "എല്ലാം തുടങ്ങിയത് ഇവിടെ നിന്നാണ്. 2013-ൽ, നാട്ടിൽ കേരളത്തിൽ വെച്ച്, ഒരു കിറ്റിൽ നിന്ന് ഞാൻ എന്റെ ആദ്യത്തെ റോബോട്ട് ഉണ്ടാക്കി.",
            "എല്ലാം തുടങ്ങിയത് ഇവിടെ നിന്നാണ്. ട്വന്റി തേർട്ടീനിൽ, നാട്ടിൽ കേരളത്തിൽ വെച്ച്, ഒരു കിറ്റിൽ നിന്ന് ഞാൻ എന്റെ ആദ്യത്തെ റോബോട്ട് ഉണ്ടാക്കി.",
        ),
        (
            "2021 ഒക്ടോബറിൽ ജൈടെക്സിൽ, ഈ ആൻഡ്രോയിഡിനെ കണ്ടാൽ ശരിക്കും ഒരു മനുഷ്യനാണെന്ന് തോന്നും.",
            "ട്വന്റി ട്വന്റി വൺ ഒക്ടോബറിൽ ജൈടെക്സിൽ, ഈ ആൻഡ്രോയിഡിനെ കണ്ടാൽ ശരിക്കും ഒരു മനുഷ്യനാണെന്ന് തോന്നും.",
        ),
        ("2022 ഒക്ടോബർ, വീണ്ടും ജൈടെക്സിൽ, 170,000-ലധികം സന്ദർശകർ.", "ട്വന്റി ട്വന്റി ടു ഒക്ടോബർ, വീണ്ടും ജൈടെക്സിൽ, വൺ ലാഖ് സെവന്റി തൗസന്റിലധികം സന്ദർശകർ."),
        (
            "സോഫ്റ്റ്ബാങ്ക് ഉണ്ടാക്കിയ റോബോട്ട് ആണ്, 2020-ൽ നിർമ്മാണം നിർത്തുന്നതിന് മുമ്പ് ഏകദേശം ഇരുപത്തിയേഴായിരം എണ്ണം ഉണ്ടാക്കി.",
            "സോഫ്റ്റ്ബാങ്ക് ഉണ്ടാക്കിയ റോബോട്ട് ആണ്, ട്വന്റി ട്വന്റിയിൽ നിർമ്മാണം നിർത്തുന്നതിന് മുമ്പ് ഏകദേശം ഇരുപത്തിയേഴായിരം എണ്ണം ഉണ്ടാക്കി.",
        ),
        ("മിനിറ്റിൽ 2,500 ഗാലൻ വെള്ളം പമ്പ് ചെയ്യും, ഡീസലിൽ ഇരുപത് മണിക്കൂർ ഓടും.", "മിനിറ്റിൽ ടു തൗസന്റ് ഫൈവ് ഹണ്ട്രഡ് ഗാലൻ വെള്ളം പമ്പ് ചെയ്യും, ഡീസലിൽ ഇരുപത് മണിക്കൂർ ഓടും."),
        ("2008 മുതൽ കുട്ടികളെയും ഗവേഷകരെയും പഠിപ്പിക്കുന്നു.", "ടു തൗസന്റ് എയ്റ്റ് മുതൽ കുട്ടികളെയും ഗവേഷകരെയും പഠിപ്പിക്കുന്നു."),
        (
            "ജപ്പാനിലെ ആദ്യത്തെ സ്വദേശി ഇൻഡസ്ട്രിയൽ റോബോട്ട് ഉണ്ടാക്കിയത് കവാസാക്കി ആണ്, 1969-ൽ.",
            "ജപ്പാനിലെ ആദ്യത്തെ സ്വദേശി ഇൻഡസ്ട്രിയൽ റോബോട്ട് ഉണ്ടാക്കിയത് കവാസാക്കി ആണ്, നൈന്റീൻ സിക്സ്റ്റി നൈനിൽ.",
        ),
        ("ബ്രിട്ടനിലെ എഞ്ചിനീയേർഡ് ആർട്സ് 2021-ന്റെ അവസാനത്തിലാണ് ഇതിനെ അവതരിപ്പിച്ചത്.", "ബ്രിട്ടനിലെ എഞ്ചിനീയേർഡ് ആർട്സ് ട്വന്റി ട്വന്റി വണ്ണിന്റെ അവസാനത്തിലാണ് ഇതിനെ അവതരിപ്പിച്ചത്."),
        (
            "പന്ത്രണ്ട് കിലോ ഭാരം, ഏകദേശം $2,700-നാണ് ഇറങ്ങിയത്, പഴയ റോബോട്ട് നായകളുടെ വിലയുടെ ഒരു ചെറിയ ഭാഗം മാത്രം.",
            "പന്ത്രണ്ട് കിലോ ഭാരം, ഏകദേശം ടു തൗസന്റ് സെവൻ ഹണ്ട്രഡ് ഡോളറിനാണ് ഇറങ്ങിയത്, പഴയ റോബോട്ട് നായകളുടെ വിലയുടെ ഒരു ചെറിയ ഭാഗം മാത്രം.",
        ),
        ("2023 ഒക്ടോബറിൽ, ബുസാനിൽ, എന്റെ ഭക്ഷണം കൊണ്ടുവന്നത് ഡില്ലി എസ് ആണ്.", "ട്വന്റി ട്വന്റി ത്രീ ഒക്ടോബറിൽ, ബുസാനിൽ, എന്റെ ഭക്ഷണം കൊണ്ടുവന്നത് ഡില്ലി എസ് ആണ്."),
        (
            "വീണ്ടും പെപ്പർ, സോഫ്റ്റ്ബാങ്ക് നിർമ്മാണം നിർത്തി വർഷങ്ങൾ കഴിഞ്ഞിട്ടും, 2024-ലും ഒരു ഗ്ലാസ് ബോക്സിൽ നിന്ന് ആളുകളെ സ്വാഗതം ചെയ്യുന്നു.",
            "വീണ്ടും പെപ്പർ, സോഫ്റ്റ്ബാങ്ക് നിർമ്മാണം നിർത്തി വർഷങ്ങൾ കഴിഞ്ഞിട്ടും, ട്വന്റി ട്വന്റി ഫോറിലും ഒരു ഗ്ലാസ് ബോക്സിൽ നിന്ന് ആളുകളെ സ്വാഗതം ചെയ്യുന്നു.",
        ),
        ("ചൈനയിലെ ഹാങ്ഷൗവിൽ 2016-ലാണ് യൂണിട്രീ തുടങ്ങിയത്.", "ചൈനയിലെ ഹാങ്ഷൗവിൽ ട്വന്റി സിക്സ്റ്റീനിലാണ് യൂണിട്രീ തുടങ്ങിയത്."),
        ("ബേസിക് മോഡൽ 2023-ൽ $1,600-നാണ് ഇറങ്ങിയത്.", "ബേസിക് മോഡൽ ട്വന്റി ട്വന്റി ത്രീയിൽ സിക്സ്റ്റീൻ ഹണ്ട്രഡ് ഡോളറിനാണ് ഇറങ്ങിയത്."),
        (
            "ഫുട്പാത്തിന്റെ അരികുകൾ കയറും, 2024-ൽ യു കെ-യിലെ ആദ്യത്തെ ക്യാമ്പസ് റോബോട്ട് ഡെലിവറി തുടങ്ങിയത് ഇതാണ്.",
            "ഫുട്പാത്തിന്റെ അരികുകൾ കയറും, ട്വന്റി ട്വന്റി ഫോറിൽ യു കെ-യിലെ ആദ്യത്തെ ക്യാമ്പസ് റോബോട്ട് ഡെലിവറി തുടങ്ങിയത് ഇതാണ്.",
        ),
        (
            "പിറ്റേ ദിവസം മ്യൂസിയം ഓഫ് ദ ഫ്യൂച്ചറിൽ ഞാൻ വീണ്ടും അമേക്കയെ കണ്ടു, 2022 മുതൽ അവിടെ സന്ദർശകരെ സ്വാഗതം ചെയ്യുന്നു.",
            "പിറ്റേ ദിവസം മ്യൂസിയം ഓഫ് ദ ഫ്യൂച്ചറിൽ ഞാൻ വീണ്ടും അമേക്കയെ കണ്ടു, ട്വന്റി ട്വന്റി ടു മുതൽ അവിടെ സന്ദർശകരെ സ്വാഗതം ചെയ്യുന്നു.",
        ),
        ("ഏറ്റവും പുതിയത്, 2025 നവംബറിൽ, ജുമൈറയിലെ സൺസെറ്റ് മാളിന് പിന്നിലുള്ള ബീച്ചിൽ.", "ഏറ്റവും പുതിയത്, ട്വന്റി ട്വന്റി ഫൈവ് നവംബറിൽ, ജുമൈറയിലെ സൺസെറ്റ് മാളിന് പിന്നിലുള്ള ബീച്ചിൽ."),
    ];

    #[test]
    fn numbers_read_as_the_hand_converted_script() {
        for (digits, hand) in HAND_CONVERTED {
            // only the numbers change, as the hand conversion changed them
            assert_eq!(spell_numbers(digits, Numbers::English), *hand, "{digits}");
            // and the voice reads both alike
            assert_eq!(normalize(digits).text, normalize(hand).text, "{digits}");
            // the Malayalam style says them in Malayalam words, and no digit survives either
            assert!(!spell_numbers(digits, Numbers::Malayalam).chars().any(|c| c.is_ascii_digit()), "{digits}");
        }
        assert_eq!(
            spell_numbers(HAND_CONVERTED[1].0, Numbers::Malayalam),
            "എല്ലാം തുടങ്ങിയത് ഇവിടെ നിന്നാണ്. രണ്ടായിരത്തി പതിമൂന്നിൽ, നാട്ടിൽ കേരളത്തിൽ വെച്ച്, ഒരു കിറ്റിൽ നിന്ന് ഞാൻ എന്റെ ആദ്യത്തെ റോബോട്ട് ഉണ്ടാക്കി."
        );
    }

    #[test]
    fn spelling_numbers_keeps_pauses_and_codes() {
        let s = spell_numbers("2013-ൽ ഞാൻ. [pause 1s] Go1, 4K, 2025.", Numbers::English);
        assert_eq!(s, "ട്വന്റി തേർട്ടീനിൽ ഞാൻ. [pause 1000ms] Go1, 4K, ട്വന്റി ട്വന്റി ഫൈവ്.");
        let (p, _) = pieces(&s, 200).unwrap();
        assert_eq!(p.get(1), Some(&Piece::Pause(1000)), "{p:?}");
        assert_eq!(p.get(2), Some(&Piece::Say("Go വൺ, ഫോർ കെ, ട്വന്റി ട്വന്റി ഫൈവ്.".into())), "{p:?}");
        // a script with no numbers is unchanged
        let plain = "ഇത് ശരിയാണ്. [pause 500ms] അതെ!";
        assert_eq!(spell_numbers(plain, Numbers::English), "ഇത് ശരിയാണ്. [pause 500ms] അതെ!");
        // pieces in either style
        let (p, _) = pieces_with("2013.", 100, Numbers::Malayalam).unwrap();
        assert_eq!(p, vec![Piece::Say("രണ്ടായിരത്തി പതിമൂന്ന്.".into())]);
        let (p, _) = pieces("2013.", 100).unwrap();
        assert_eq!(p, vec![Piece::Say("ട്വന്റി തേർട്ടീൻ.".into())]);
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

    fn say(text: &str, asked: Asked) -> Part {
        Part::Say { text: text.into(), asked }
    }

    #[test]
    fn styled_scripts_become_parts() {
        let all = |_: Asked| 200usize;
        // tags switch the style for what follows and are never read; a tag mid-sentence splits it
        let (p, notes) = parts("ഇത് ഒന്നാണ്. [excited] ഇത് രണ്ടാണ്! അതെ! [calm] പിന്നെ [plain] ശരി.", &all, Numbers::English, true).unwrap();
        assert_eq!(
            p,
            vec![say("ഇത് ഒന്നാണ്.", Asked::Line), say("ഇത് രണ്ടാണ്! അതെ!", Asked::Style("excited")), say("പിന്നെ.", Asked::Style("calm")), say("ശരി.", Asked::Plain),]
        );
        assert!(notes.is_empty(), "tags are not reported as words: {notes:?}");
        assert!(report("[excited] ഇത് ശരിയാണ്.").is_empty());
        // pauses, breaths and line breaks: an utterance never spans a line break
        let (p, _) = parts("ഒന്ന്.\nരണ്ട്.\n\n[breath]മൂന്ന്. [pause 1s] നാല്.\n\n\nഅഞ്ച്.", &all, Numbers::English, true).unwrap();
        assert_eq!(
            p,
            vec![
                say("ഒന്ന്.", Asked::Line),
                Part::Break(Boundary::Line),
                say("രണ്ട്.", Asked::Line),
                Part::Breath,
                say("മൂന്ന്.", Asked::Line),
                Part::Pause(1000),
                say("നാല്.", Asked::Line),
                Part::Break(Boundary::Paragraph),
                say("അഞ്ച്.", Asked::Line),
            ]
        );
        // a line break before a tag still breaks; packing and per-sentence reading as `pieces`
        let (p, _) = parts("ഒന്ന്.\n[calm] രണ്ട്. മൂന്ന്.", &all, Numbers::English, false).unwrap();
        assert_eq!(p, vec![say("ഒന്ന്.", Asked::Line), Part::Break(Boundary::Line), say("രണ്ട്.", Asked::Style("calm")), say("മൂന്ന്.", Asked::Style("calm"))]);
        // each style packs to its own reference's length
        let long = "ഇതൊരു ചെറിയ വാചകമാണ്. ".repeat(10);
        let max = |a: Asked| if a == Asked::Style("calm") { 60 } else { 1000 };
        let (p, _) = parts(&format!("{long}[calm]{long}"), &max, Numbers::English, true).unwrap();
        let (line, calm): (Vec<&Part>, Vec<&Part>) = p.iter().partition(|x| matches!(x, Part::Say { asked: Asked::Line, .. }));
        assert_eq!(line.len(), 1);
        assert!(calm.len() > 1 && calm.iter().all(|x| matches!(x, Part::Say { text, .. } if text.chars().count() <= 60)));
        // a script without tags or line breaks reads as `pieces` reads it
        let plain = "ഇത് ഒന്നാണ്. [pause 1s] ഇത് രണ്ടാണ്? അതെ!";
        let (p, _) = parts(plain, &all, Numbers::English, true).unwrap();
        let (q, _) = pieces(plain, 200).unwrap();
        let as_pieces: Vec<Piece> = p
            .into_iter()
            .map(|x| match x {
                Part::Say { text, .. } => Piece::Say(text),
                Part::Pause(ms) => Piece::Pause(ms),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(as_pieces, q);
        // nothing to say
        assert_eq!(parts("[excited] [breath]", &all, Numbers::English, true), Err(TtsError::Empty));
        assert_eq!(parts("", &all, Numbers::English, true), Err(TtsError::Empty));
        assert_eq!(parts(&"അ.\n".repeat(20_000), &all, Numbers::English, true), Err(TtsError::TextTooLong));
        for t in ["[", "\n\n\n", "[calm]\n", "\r\n[breath]\r\n", "[excited", "🙂\n🙂", "[pause 1s]\n[pause 1s]"] {
            let _ = parts(t, &all, Numbers::English, true);
            let _ = parts(t, &|_| 0, Numbers::Malayalam, false);
        }
    }

    #[test]
    fn spelling_numbers_keeps_style_tags() {
        let s = spell_numbers("[excited] 2013-ൽ! [Calm]ശരി. [breath] [neutral] അതെ.", Numbers::English);
        assert_eq!(s, "[excited] ട്വന്റി തേർട്ടീനിൽ! [calm]ശരി. [breath] [plain] അതെ.");
        // line breaks between tags are kept for the reading
        assert_eq!(spell_numbers("ഒന്ന്.[calm]\n\n[warm]രണ്ട്.", Numbers::English), "ഒന്ന്. [calm]\n\n[warm]രണ്ട്.");
        // a script without tags spells as before (its cache keys stay)
        assert_eq!(spell_numbers("ഒന്ന്. [pause 1s]\n[pause 1s] രണ്ട്.", Numbers::English), "ഒന്ന്. [pause 1000ms] [pause 1000ms] രണ്ട്.");
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
            for style in Numbers::ALL {
                let _ = pieces_with(t, 100, style);
                let _ = normalize_with(t, style);
                let _ = spell_numbers(t, style);
            }
        }
        // money, ranges and suffixes cut off anywhere
        for t in ["$", "$ ", "₹1,", "$1.", "2013–", "2013-ൽ-", "1,00,", "€9999999999999999999", "0.0000000000000000001%", "1-2-3-4", "$1-2"] {
            for style in Numbers::ALL {
                let _ = normalize_with(t, style);
                let _ = spell_numbers(t, style);
            }
        }
    }
}
