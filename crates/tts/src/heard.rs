//! The strict heard-back check of Malayalam narration: what a speech recogniser heard a voice say,
//! scored against what it was asked to say, and the retake policy built on it.
//!
//! - [`comparable`]: text as it is compared — canonical Malayalam spelling (the two-part vowel
//!   signs composed, as Unicode NFC does; old-style chillus made atomic; zero-width joiners and
//!   non-joiners dropped), then each chillu written as its consonant and virama (`ൻ` = `ന്`: a
//!   recogniser writes either), `ൌ` as `ൗ`, Latin letters in lower case; punctuation, symbols
//!   and spaces dropped (Malayalam joins and splits words freely: "ഉണ്ടല്ലോ" and "ഉണ്ട്
//!   അല്ലോ" are the same speech).
//! - [`score`]: the character error rate (edits over the said characters) and, for every
//!   sentence, the error rate of its first [`START_SYLLABLES`] syllables — where IndicF5's
//!   garble is: a reading can say most of a line and still mumble its first word. Edits are
//!   read off one minimum-edit alignment of the two texts; characters heard just before a
//!   sentence's first one count double against its start (speech in front of a sentence is the
//!   garble: IndicF5 saying its reference's last word again, "ഇല്ല ഇവിടെ നിന്നാണ്…").
//! - [`signal`]: an advisory scan of a reading's samples (empty, invalid, clipped, a dropout
//!   mid-reading, a burst at the start); each finding fails the reading.
//! - [`hear_in_pieces`]: a long reading heard in stretches cut at pauses (a recogniser's decoder
//!   holds about 12 s of Malayalam).
//! - [`Policy`] and [`read_until_clear`]: an utterance is read, heard and scored; one that fails
//!   is read again (the caller varies the seed and the settings per attempt) up to
//!   [`Policy::max_takes`], the best kept ([`Score::badness`]), every try reported with its seed.
//! - **A second opinion and a voice gate** ([`read_until_clear_judged`], [`Strict::second`],
//!   [`Strict::voice`]): a second recogniser hears a reading the first rejects, and under
//!   [`Agreement::Both`] (the default) the reading fails only when both hear it wrong — one
//!   recogniser's own mistakes (a loanword, a number written as digits) no longer cost a retake.
//!   The voice gate is a speaker model's similarity of the reading to the voice's own sample: a
//!   reading under [`Strict::min_similarity`] has drifted from the speaker's sound and is read
//!   again.
//! - **English** ([`english_words`], [`Policy::english`]): numbers in digits are spelled as words
//!   on both sides, `%` and `&` said, so "10 to 80%" and "ten to eighty percent" compare equal.
//!
//! Pure functions: no model here. The caller hears (a speech model fine-tuned for Malayalam: the
//! engine's `check: "strict"`).

use serde::Serialize;

use crate::TtsError;

/// Syllables at the start of each sentence scored on their own ([`Score::start_cer`]).
pub const START_SYLLABLES: usize = 6;
/// What a character heard in front of a sentence counts in its start score (see [`score`]).
const LEAD_WEIGHT: usize = 2;
/// Longest text compared, characters (the edit alignment is quadratic).
const MAX_COMPARED: usize = 1500;

/// The atomic chillus and the consonant each is a final form of.
fn chillu_consonant(c: char) -> Option<char> {
    Some(match c {
        'ൺ' => 'ണ',
        'ൻ' => 'ന',
        'ർ' => 'ര',
        'ൽ' => 'ല',
        'ൾ' => 'ള',
        'ൿ' => 'ക',
        // the dot reph is a chillu ര too
        'ൎ' => 'ര',
        _ => return None,
    })
}

/// `text` as compared (see the module docs).
pub fn comparable(text: &str) -> Vec<char> {
    let mut out = Vec::new();
    for c in crate::malayalam::canonical(text).chars() {
        if let Some(base) = chillu_consonant(c) {
            out.extend([base, '\u{0D4D}']);
        } else if c == '\u{0D4C}' {
            out.push('\u{0D57}');
        } else if c.is_alphanumeric() || is_mark(c) {
            out.extend(c.to_lowercase());
        }
        if out.len() >= MAX_COMPARED {
            break;
        }
    }
    out.truncate(MAX_COMPARED);
    out
}

/// A vowel sign, virama, anusvara or visarga of an Indian script (letters' marks: kept).
fn is_mark(c: char) -> bool {
    let cp = c as u32;
    (0x0900..0x0E00).contains(&cp) && matches!(cp & 0x7F, 0x00..=0x03 | 0x3A..=0x4F | 0x55..=0x57 | 0x62 | 0x63)
}

/// One step of an alignment of said against heard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    /// Said character `i` heard as heard character `j` (equal or not).
    Pair(usize, usize, bool),
    /// Said character `i` not heard.
    Missed(usize),
    /// Heard character `j` that was not said, before said character `i` (`i` = len: at the end).
    Extra(usize),
}

/// A minimum-edit alignment of `a` (said) and `b` (heard), in order.
fn align(a: &[char], b: &[char]) -> Vec<Step> {
    let (n, m) = (a.len(), b.len());
    let w = m + 1;
    let mut d = vec![0u32; (n + 1) * w];
    for j in 0..=m {
        d[j] = j as u32;
    }
    for i in 1..=n {
        d[i * w] = i as u32;
        for j in 1..=m {
            let sub = d[(i - 1) * w + j - 1] + u32::from(a[i - 1] != b[j - 1]);
            let del = d[(i - 1) * w + j] + 1;
            let ins = d[i * w + j - 1] + 1;
            d[i * w + j] = sub.min(del).min(ins);
        }
    }
    // back from the end: pairs first, then missed, then extra (extras land early: noise in
    // front of a word is aligned in front of it)
    let mut steps = Vec::with_capacity(n + m);
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        let here = d[i * w + j];
        if i > 0 && j > 0 && d[(i - 1) * w + j - 1] + u32::from(a[i - 1] != b[j - 1]) == here {
            steps.push(Step::Pair(i - 1, j - 1, a[i - 1] == b[j - 1]));
            i -= 1;
            j -= 1;
        } else if i > 0 && d[(i - 1) * w + j] + 1 == here {
            steps.push(Step::Missed(i - 1));
            i -= 1;
        } else {
            steps.push(Step::Extra(i));
            j = j.saturating_sub(1);
        }
    }
    steps.reverse();
    steps
}

/// How well a reading was heard.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Score {
    /// Character error rate over the whole text: edits / said characters.
    pub cer: f64,
    /// The worst sentence start: edits in a sentence's first [`START_SYLLABLES`] syllables (and
    /// noise just before them) / their characters, at most 1.
    pub start_cer: f64,
    /// Each sentence's start error rate, in order.
    pub starts: Vec<f64>,
}

impl Score {
    /// What ranks readings when none passes: a garbled start weighs double.
    pub fn badness(&self) -> f64 {
        2.0 * self.start_cer + self.cer.min(2.0)
    }
}

/// Where each sentence of `said` starts and how long its scored start is, in comparable
/// characters: `(first, len)`.
fn sentence_starts(said: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    for s in crate::malayalam::sentences(said) {
        let chars = comparable(&s);
        if chars.is_empty() {
            continue;
        }
        // the characters of the first syllables: grow until START_SYLLABLES are complete (the
        // syllable after them begins)
        let mut len = chars.len();
        for k in 1..=chars.len() {
            let head: String = chars.iter().take(k).collect();
            if crate::malayalam::syllables(&head) > START_SYLLABLES {
                len = k - 1;
                break;
            }
        }
        out.push((offset, len.max(1)));
        offset += chars.len();
    }
    out
}

/// `heard` scored against `said` (see the module docs). Nothing to say scores 0; something
/// said and nothing heard scores 1.
pub fn score(said: &str, heard: &str) -> Score {
    let (a, b) = (comparable(said), comparable(heard));
    if a.is_empty() {
        return Score::default();
    }
    let steps = align(&a, &b);
    let edits = steps.iter().filter(|s| !matches!(s, Step::Pair(_, _, true))).count();
    let starts: Vec<f64> = sentence_starts(said)
        .into_iter()
        .filter(|(first, _)| *first < a.len())
        .map(|(first, len)| {
            let end = (first + len).min(a.len());
            let bad: usize = steps
                .iter()
                .map(|s| match s {
                    Step::Pair(i, _, ok) => usize::from((first..end).contains(i) && !ok),
                    Step::Missed(i) => usize::from((first..end).contains(i)),
                    // speech in front of the sentence counts double: a word said that was not
                    // asked for (IndicF5 repeating its reference's last word) is the garble
                    Step::Extra(i) if *i == first => LEAD_WEIGHT,
                    Step::Extra(i) => usize::from((first..end).contains(i)),
                })
                .sum();
            (bad as f64 / (end - first).max(1) as f64).min(1.0)
        })
        .collect();
    let start_cer = starts.iter().copied().fold(0.0, f64::max);
    Score { cer: edits as f64 / a.len() as f64, start_cer, starts }
}

/// When a reading is clear enough, and how often to try.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Policy {
    /// Readings of one utterance at most (the first and the retakes).
    pub max_takes: u32,
    /// Highest character error rate that passes.
    pub max_cer: f64,
    /// Highest sentence-start error rate that passes.
    pub max_start_cer: f64,
}

/// Version of the scoring and the default policy: the narration cache keys checked readings on
/// it, so a stricter check reads them again.
pub const VERSION: u32 = 1;

impl Default for Policy {
    /// The measured thresholds (crate README, "Malayalam: sentence starts and the strict
    /// check"): readings with intact reference words score at most 0.23 overall (the
    /// recogniser's own errors, mostly loanwords) and no first sentence start over 0.3; the old
    /// readings' leaked words score 0.4–1 there.
    fn default() -> Self {
        Policy { max_takes: 4, max_cer: 0.25, max_start_cer: 0.3 }
    }
}

impl Policy {
    pub fn passes(&self, s: &Score) -> bool {
        s.cer <= self.max_cer && s.start_cer <= self.max_start_cer
    }

    /// English narration (cloned voices): the thresholds measured on the owner's cloned English
    /// readings heard by Whisper large-v3-turbo (`crates/speech/README.md`, "English narration:
    /// the strict check"): good readings stay under 0.09 overall; a sentence start may lose a
    /// short word to the recogniser.
    pub fn english() -> Self {
        Policy { max_takes: 3, max_cer: 0.15, max_start_cer: 0.5 }
    }

    pub fn check(&self) -> Result<(), TtsError> {
        if !(1..=MAX_TAKES).contains(&self.max_takes)
            || !(self.max_cer.is_finite() && self.max_cer >= 0.0)
            || !(self.max_start_cer.is_finite() && self.max_start_cer >= 0.0)
        {
            return Err(TtsError::Invalid(format!("a strict check takes 1–{MAX_TAKES} readings and finite thresholds")));
        }
        Ok(())
    }
}

/// Most readings of one utterance a policy may ask for.
pub const MAX_TAKES: u32 = 8;

/// One reading of an utterance and what was heard.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Try {
    pub attempt: u32,
    /// The seed it was read with: the same seed and settings read it again exactly.
    pub seed: u64,
    pub heard: String,
    #[serde(flatten)]
    pub score: Score,
    /// What the signal scan found ([`signal`]): each fails the reading.
    pub signal: Vec<String>,
    /// The second recogniser's hearing, when it was asked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second: Option<Opinion>,
    /// How alike the reading sounds to the voice (the voice gate), when measured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity: Option<f64>,
    /// What failed the reading: `first`, `second` (the recognisers), `voice`, `signal`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub flagged: Vec<String>,
}

/// A second recogniser's hearing of a reading.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Opinion {
    pub heard: String,
    pub cer: f64,
    pub start_cer: f64,
    pub passed: bool,
}

impl Try {
    fn badness(&self, agreement: Agreement, min_similarity: f64) -> f64 {
        let first = self.score.badness();
        let text = match (&self.second, agreement) {
            (Some(o), Agreement::Both) => first.min(2.0 * o.start_cer + o.cer.min(2.0)),
            (Some(o), Agreement::Either) => first.max(2.0 * o.start_cer + o.cer.min(2.0)),
            _ => first,
        };
        let voice = self.similarity.filter(|s| *s < min_similarity).map_or(0.0, |s| 0.5 + (min_similarity - s));
        text + 0.25 * self.signal.len() as f64 + voice
    }
}

/// A reading of an utterance: its samples and the seed it was read with.
pub struct Read {
    pub samples: Vec<f32>,
    pub seed: u64,
}

/// What hears a reading: a speech recogniser (the engine's Malayalam speech model).
pub trait Hearer: Send + Sync {
    /// The words a recogniser hears in mono `samples` at `rate`.
    fn hear(&self, samples: &[f32], rate: u32) -> Result<String, String>;
    /// The recogniser and its settings (the narration cache keys checked readings on it).
    fn id(&self) -> String;
}

/// How alike a reading sounds to the voice it should be (the voice gate).
pub trait VoiceMatch: Send + Sync {
    /// A speaker model's cosine similarity of mono `samples` at `rate` to the voice's own sample.
    fn similarity(&self, samples: &[f32], rate: u32) -> Result<f64, String>;
    /// The speaker model and the voice (the narration cache keys checked readings on it).
    fn id(&self) -> String;
}

/// How two recognisers' opinions of a reading combine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Agreement {
    /// A reading fails only when both recognisers hear it wrong (the second is asked only when
    /// the first rejects it).
    #[default]
    Both,
    /// A reading fails when either hears it wrong (both always asked).
    Either,
    /// Only the first recogniser counts.
    First,
}

impl Agreement {
    pub fn id(self) -> &'static str {
        match self {
            Agreement::Both => "both",
            Agreement::Either => "either",
            Agreement::First => "first",
        }
    }

    pub fn from_id(s: &str) -> Option<Self> {
        Some(match s {
            "both" => Agreement::Both,
            "either" => Agreement::Either,
            "first" => Agreement::First,
            _ => return None,
        })
    }
}

/// The strict check a voice reads under ([`crate::Voice::synthesize_checked`]).
pub struct Strict<'a> {
    pub hearer: &'a dyn Hearer,
    pub policy: Policy,
    /// A second recogniser (Malayalam: IndicWhisper), combined by `agreement`.
    pub second: Option<&'a dyn Hearer>,
    pub agreement: Agreement,
    /// The voice gate, and the least similarity that passes.
    pub voice: Option<&'a dyn VoiceMatch>,
    pub min_similarity: f64,
}

impl<'a> Strict<'a> {
    /// One recogniser, no voice gate.
    pub fn new(hearer: &'a dyn Hearer, policy: Policy) -> Self {
        Strict { hearer, policy, second: None, agreement: Agreement::Both, voice: None, min_similarity: 0.0 }
    }

    /// What the check is, for cache keys and reports: the recogniser, then the second opinion
    /// and the voice gate when there are.
    pub fn id(&self) -> String {
        let mut id = self.hearer.id();
        if let Some(s) = self.second.filter(|_| self.agreement != Agreement::First) {
            id = format!("{id} + {} ({})", s.id(), self.agreement.id());
        }
        if let Some(v) = self.voice {
            id = format!("{id}, voice {} >= {:.2}", v.id(), self.min_similarity);
        }
        id
    }

    /// Read `text` until clear under this check ([`read_until_clear_judged`]): `hear_rate` is the
    /// rate of the readings `read` makes.
    pub fn read_until_clear(&self, text: &str, rate: u32, read: &mut dyn FnMut(u32) -> Result<Read, TtsError>) -> Result<(Vec<f32>, Checked), TtsError> {
        let mut hear = |x: &[f32]| self.hearer.hear(x, rate).map_err(|e| TtsError::Model(format!("the strict check could not hear the reading: {e}")));
        let mut second = self.second.map(|h| move |x: &[f32]| h.hear(x, rate).map_err(TtsError::Model));
        let mut voice = self.voice.map(|v| move |x: &[f32]| v.similarity(x, rate).map_err(TtsError::Model));
        let mut judges = Judges {
            second: second.as_mut().map(|f| f as &mut dyn FnMut(&[f32]) -> Result<String, TtsError>),
            agreement: self.agreement,
            voice: voice.as_mut().map(|f| f as &mut dyn FnMut(&[f32]) -> Result<f64, TtsError>),
            min_similarity: self.min_similarity,
        };
        read_until_clear_judged(text, &self.policy, rate, read, &mut hear, &mut judges)
    }
}

/// The second opinion and the voice gate of [`read_until_clear_judged`].
#[derive(Default)]
pub struct Judges<'a> {
    /// A second recogniser: what it hears in a reading (an error: no second opinion this time).
    pub second: Option<&'a mut dyn FnMut(&[f32]) -> Result<String, TtsError>>,
    pub agreement: Agreement,
    /// The voice gate: a reading's similarity to the voice (an error: not measured).
    pub voice: Option<&'a mut dyn FnMut(&[f32]) -> Result<f64, TtsError>>,
    pub min_similarity: f64,
}

/// What the strict check found for one utterance.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Checked {
    /// What the voice was asked to say.
    pub text: String,
    /// What was heard in the reading kept.
    pub heard: String,
    pub cer: f64,
    pub start_cer: f64,
    /// Readings made.
    pub takes: u32,
    /// The reading kept (0: the first).
    pub kept: u32,
    /// The kept reading's seed.
    pub seed: u64,
    /// The kept reading passed the policy.
    pub passed: bool,
    /// What the signal scan found in the kept reading.
    pub signal: Vec<String>,
    /// The second recogniser's hearing of the kept reading, when it was asked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second: Option<Opinion>,
    /// The kept reading's similarity to the voice (the voice gate), when measured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity: Option<f64>,
    /// What failed the kept reading (empty when it passed).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub flagged: Vec<String>,
    /// The voice style it was read in and the style reference it was read after (a voice that
    /// keeps references per style, [`crate::styles`]), when not plain.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    /// Every reading, in order.
    pub tries: Vec<Try>,
}

/// Read `text` until a reading is heard clearly: `read(attempt)` makes reading `attempt` (0, 1,
/// …), `hear(samples)` says what a recogniser heard in it (`rate`: the readings' rate). Stops at
/// the first reading that passes `policy` with a clean signal ([`signal`]), else after
/// [`Policy::max_takes`] keeps the best ([`Score::badness`], a quarter more per signal finding),
/// every try reported with its seed. A failed hearing is an error (the check could not run),
/// never a pass.
pub fn read_until_clear(
    text: &str,
    policy: &Policy,
    rate: u32,
    read: &mut dyn FnMut(u32) -> Result<Read, TtsError>,
    hear: &mut dyn FnMut(&[f32]) -> Result<String, TtsError>,
) -> Result<(Vec<f32>, Checked), TtsError> {
    read_until_clear_judged(text, policy, rate, read, hear, &mut Judges::default())
}

/// [`read_until_clear`] with a second opinion and a voice gate (`judges`): a reading passes when
/// its text passes (the first recogniser, or as `judges.agreement` combines it with the second),
/// its similarity to the voice is at least `judges.min_similarity` (when measured) and its signal
/// is clean. Each try reports what failed it (`flagged`).
pub fn read_until_clear_judged(
    text: &str,
    policy: &Policy,
    rate: u32,
    read: &mut dyn FnMut(u32) -> Result<Read, TtsError>,
    hear: &mut dyn FnMut(&[f32]) -> Result<String, TtsError>,
    judges: &mut Judges,
) -> Result<(Vec<f32>, Checked), TtsError> {
    policy.check()?;
    let min_similarity = if judges.min_similarity.is_finite() { judges.min_similarity.clamp(-1.0, 1.0) } else { 0.0 };
    let mut best: Option<(f64, Vec<f32>, Try, bool)> = None;
    let mut tries = Vec::new();
    for attempt in 0..policy.max_takes {
        let r = read(attempt)?;
        let heard = hear(&r.samples)?;
        let sc = score(text, &heard);
        let first_ok = policy.passes(&sc);
        let ask = match judges.agreement {
            Agreement::Both => !first_ok,
            Agreement::Either => true,
            Agreement::First => false,
        };
        let second = match (ask, judges.second.as_mut()) {
            (true, Some(h)) => h(&r.samples).ok().map(|heard| {
                let s = score(text, &heard);
                Opinion { passed: policy.passes(&s), heard, cer: s.cer, start_cer: s.start_cer }
            }),
            _ => None,
        };
        let text_ok = match (judges.agreement, &second) {
            (Agreement::Both, Some(o)) => first_ok || o.passed,
            (Agreement::Either, Some(o)) => first_ok && o.passed,
            _ => first_ok,
        };
        let similarity = judges.voice.as_mut().and_then(|v| v(&r.samples).ok()).filter(|s| s.is_finite());
        let voice_ok = similarity.is_none_or(|s| s >= min_similarity);
        let sig = signal(&r.samples, rate);
        let mut flagged = Vec::new();
        if !first_ok {
            flagged.push("first".to_string());
        }
        if second.as_ref().is_some_and(|o| !o.passed) {
            flagged.push("second".to_string());
        }
        if !voice_ok {
            flagged.push("voice".to_string());
        }
        if !sig.is_empty() {
            flagged.push("signal".to_string());
        }
        let passed = text_ok && voice_ok && sig.is_empty();
        let t = Try { attempt, seed: r.seed, score: sc, heard, signal: sig, second, similarity, flagged: if passed { Vec::new() } else { flagged } };
        tries.push(t.clone());
        let bad = t.badness(judges.agreement, min_similarity);
        if passed || best.as_ref().is_none_or(|(b, _, _, _)| bad < *b) {
            best = Some((bad, r.samples, t, passed));
        }
        if passed {
            break;
        }
    }
    let (_, audio, t, passed) = best.ok_or_else(|| TtsError::Invalid("nothing was read".into()))?;
    let checked = Checked {
        text: text.to_string(),
        heard: t.heard.clone(),
        cer: t.score.cer,
        start_cer: t.score.start_cer,
        takes: tries.len() as u32,
        kept: t.attempt,
        seed: t.seed,
        passed,
        signal: t.signal.clone(),
        second: t.second.clone(),
        similarity: t.similarity,
        flagged: t.flagged.clone(),
        style: None,
        tries,
    };
    Ok((audio, checked))
}

/// English text as it is compared: numbers written in digits spelled as words ("1,600" → "one
/// thousand six hundred", "5.3" → "five point three"), `%` as "percent", `&` as "and". Both the
/// script and what a recogniser heard go through it.
pub fn english_words(text: &str) -> String {
    const ONES: [&str; 20] = [
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
    const TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
    fn below_1000(n: u64, out: &mut Vec<String>) {
        let (h, r) = (n / 100, n % 100);
        if h > 0 {
            out.push(ONES[h as usize].to_string());
            out.push("hundred".into());
        }
        if r >= 20 {
            out.push(TENS[(r / 10) as usize].to_string());
            if r % 10 > 0 {
                out.push(ONES[(r % 10) as usize].to_string());
            }
        } else if r > 0 || h == 0 {
            out.push(ONES[r as usize].to_string());
        }
    }
    fn cardinal(n: u64) -> String {
        if n == 0 {
            return "zero".into();
        }
        let mut out = Vec::new();
        for (scale, name) in [(1_000_000_000_000u64, "trillion"), (1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")] {
            if !(n / scale).is_multiple_of(1000) {
                below_1000(n / scale % 1000, &mut out);
                out.push(name.into());
            }
        }
        if !n.is_multiple_of(1000) {
            below_1000(n % 1000, &mut out);
        }
        out.join(" ")
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + 16);
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_digit() {
            // digits, with thousands commas and one decimal point
            let mut int = String::new();
            while i < chars.len() && (chars[i].is_ascii_digit() || (chars[i] == ',' && chars.get(i + 1).is_some_and(char::is_ascii_digit) && !int.is_empty())) {
                if chars[i] != ',' {
                    int.push(chars[i]);
                }
                i += 1;
            }
            let mut frac = String::new();
            if chars.get(i) == Some(&'.') && chars.get(i + 1).is_some_and(char::is_ascii_digit) {
                i += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    frac.push(chars[i]);
                    i += 1;
                }
            }
            out.push(' ');
            match int.parse::<u64>() {
                Ok(n) if int.len() <= 15 => out.push_str(&cardinal(n)),
                // too long to say as one number: digit by digit
                _ => out.push_str(&int.chars().filter_map(|d| d.to_digit(10)).map(|d| ONES[d as usize]).collect::<Vec<_>>().join(" ")),
            }
            if !frac.is_empty() {
                out.push_str(" point ");
                out.push_str(&frac.chars().filter_map(|d| d.to_digit(10)).map(|d| ONES[d as usize]).collect::<Vec<_>>().join(" "));
            }
            out.push(' ');
            continue;
        }
        match c {
            '%' => out.push_str(" percent "),
            '&' => out.push_str(" and "),
            _ => out.push(c),
        }
        i += 1;
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The longest stretch a recogniser hears at once, seconds: Whisper's decoder holds about 440
/// tokens, and Malayalam takes about two a character (a 9 s clip: 273 tokens).
pub const HEARD_SECONDS: f64 = 12.0;

/// What `hear` hears in `x`, in stretches of at most `max_secs` cut at the longest pause near
/// each middle (a long reading would overflow the recogniser's decoder), the texts joined.
pub fn hear_in_pieces(x: &[f32], rate: u32, max_secs: f64, hear: &mut dyn FnMut(&[f32]) -> Result<String, TtsError>) -> Result<String, TtsError> {
    let mut out = Vec::new();
    split_at_pauses(x, rate, max_secs, 0, &mut |piece| {
        let t = hear(piece)?;
        out.push(t.trim().to_string());
        Ok(())
    })?;
    Ok(out.into_iter().filter(|t| !t.is_empty()).collect::<Vec<_>>().join(" "))
}

/// Calls `each` with consecutive stretches of `x` of at most `max_secs` (bounded depth).
fn split_at_pauses(x: &[f32], rate: u32, max_secs: f64, depth: u32, each: &mut dyn FnMut(&[f32]) -> Result<(), TtsError>) -> Result<(), TtsError> {
    let max = (max_secs.max(1.0) * f64::from(rate.max(1))) as usize;
    if x.len() <= max || depth >= 6 {
        return each(x);
    }
    let n = ((f64::from(rate) * SCAN_FRAME) as usize).max(1);
    let levels: Vec<f64> =
        x.chunks(n).map(|c| 10.0 * (c.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / c.len().max(1) as f64 + 1e-12).log10()).collect();
    // the quietest point in the middle half, preferring long pauses: a frame's score is its
    // level averaged over ±100 ms
    let (lo, hi) = (levels.len() / 4, (levels.len() * 3 / 4).max(levels.len() / 4 + 1));
    let smooth = |f: usize| {
        let w = levels.get(f.saturating_sub(5)..(f + 6).min(levels.len())).unwrap_or(&[]);
        w.iter().sum::<f64>() / w.len().max(1) as f64
    };
    let cut = (lo..hi).min_by(|a, b| smooth(*a).total_cmp(&smooth(*b))).unwrap_or(levels.len() / 2);
    let at = (cut * n + n / 2).min(x.len());
    let (a, b) = x.split_at(at);
    split_at_pauses(a, rate, max_secs, depth + 1, each)?;
    split_at_pauses(b, rate, max_secs, depth + 1, each)
}

/// Frames of the signal scan, seconds.
const SCAN_FRAME: f64 = 0.02;
/// Shortest reading that is not empty, seconds.
const MIN_READING: f64 = 0.15;
/// A silence this long between words is a dropout, seconds (a pause at a comma, or between two
/// sentences of one utterance, is shorter: see the crate README).
pub const DROPOUT_SECONDS: f64 = 0.9;
/// Frames this far below the loud speech are silence, dB.
const SILENCE_DB: f64 = 35.0;
/// The first speech this much louder than the reading's typical speech is a burst, dB.
pub const BURST_DB: f64 = 9.0;
/// Samples at full scale over this share of the reading are clipping.
const CLIP_SHARE: f64 = 0.001;

/// An advisory scan of a reading's signal (not a measure of naturalness): empty audio, invalid
/// samples, clipping, a long silence between words (a dropout), and a burst of level at the
/// start — each often comes with a garbled reading. Empty: nothing found.
pub fn signal(x: &[f32], rate: u32) -> Vec<String> {
    let mut out = Vec::new();
    if rate == 0 || (x.len() as f64) < MIN_READING * f64::from(rate) {
        out.push("empty".to_string());
        return out;
    }
    if x.iter().any(|v| !v.is_finite()) {
        out.push("invalid samples".to_string());
        return out;
    }
    let clipped = x.iter().filter(|v| v.abs() >= 0.999).count();
    if clipped as f64 > CLIP_SHARE * x.len() as f64 {
        out.push(format!("clipping ({clipped} samples)"));
    }
    let n = ((f64::from(rate) * SCAN_FRAME) as usize).max(1);
    let levels: Vec<f64> =
        x.chunks(n).map(|c| 10.0 * (c.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / c.len().max(1) as f64 + 1e-12).log10()).collect();
    let mut sorted = levels.clone();
    sorted.sort_by(f64::total_cmp);
    // the loud speech: the 95th percentile frame
    let loud = sorted.get(sorted.len() * 95 / 100).copied().unwrap_or(-120.0);
    if loud < -60.0 {
        out.push("silent".to_string());
        return out;
    }
    let speech = |l: &f64| *l > loud - SILENCE_DB;
    let (Some(first), Some(last)) = (levels.iter().position(speech), levels.iter().rposition(speech)) else { return out };
    // the longest silence between the first and the last speech
    let (mut run, mut longest) = (0usize, 0usize);
    for l in levels.get(first..=last).unwrap_or(&[]) {
        run = if speech(l) { 0 } else { run + 1 };
        longest = longest.max(run);
    }
    let gap = longest as f64 * SCAN_FRAME;
    if gap > DROPOUT_SECONDS {
        out.push(format!("a {gap:.1} s silence mid-reading"));
    }
    // the first 200 ms of speech against the reading's median speech level
    let mut voiced: Vec<f64> = levels.iter().copied().filter(|l| speech(l)).collect();
    voiced.sort_by(f64::total_cmp);
    let median = voiced.get(voiced.len() / 2).copied().unwrap_or(loud);
    let head: Vec<f64> = levels.iter().skip(first).take((0.2 / SCAN_FRAME) as usize).copied().collect();
    let start = 10.0 * (head.iter().map(|l| 10f64.powf(l / 10.0)).sum::<f64>() / head.len().max(1) as f64 + 1e-12).log10();
    if start > median + BURST_DB {
        out.push(format!("a burst at the start ({:.0} dB over the speech)", start - median));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spelling_variants_compare_equal() {
        // chillu or consonant + virama, old ZWJ chillus, two-part vowels, ZWNJ, spaces, punctuation
        assert_eq!(comparable("അവൻ"), comparable("അവന്"));
        assert_eq!(comparable("അവ\u{0D28}\u{0D4D}\u{200D}"), comparable("അവൻ"));
        assert_eq!(comparable("കൊ"), comparable("ക\u{0D46}\u{0D3E}"));
        assert_eq!(comparable("കൌ"), comparable("കൗ"));
        assert_eq!(comparable("ടാബ്\u{200C}ലെറ്റ്"), comparable("ടാബ്ലെറ്റ്"));
        assert_eq!(comparable("ഉണ്ട് അല്ലോ."), comparable("ഉണ്ട്അല്ലോ"));
        assert_eq!(comparable("Robot, AI!"), comparable("robotai"));
        assert!(comparable("").is_empty() && comparable(" ,.!? ").is_empty());
        // marks are kept
        assert_eq!(comparable("ഇതാണ്").len(), 5);
        assert!(comparable(&"ക".repeat(10_000)).len() <= MAX_COMPARED);
    }

    #[test]
    fn character_error_rates() {
        let said = "ഇതാണ് സ്പോട്ട്.";
        assert_eq!(score(said, said), Score { cer: 0.0, start_cer: 0.0, starts: vec![0.0] });
        assert_eq!(score(said, "ഇതാണ് സ്പോട്ട്"), score(said, said), "punctuation and spaces do not count");
        assert_eq!(score(said, "").cer, 1.0);
        assert_eq!(score(said, "").start_cer, 1.0);
        assert_eq!(score("", "anything"), Score::default());
        // one character of twelve substituted
        let n = comparable(said).len() as f64;
        let s = score(said, "ഇതാണ് സ്പോട്ട");
        assert!((s.cer - 1.0 / n).abs() < 1e-9, "{s:?}");
        // insertions count: more heard than said can pass 1
        assert!(score("ഇത്", "ഇത് ഇത് ഇത് ഇത്").cer > 1.0);
        // bounded on hostile sizes
        assert!(score(&"ക ".repeat(5000), &"ഖ ".repeat(5000)).cer.is_finite());
    }

    #[test]
    fn a_garbled_start_is_scored_on_its_own() {
        let said = "ട്വന്റി ട്വന്റി വൺ ഒക്ടോബറിൽ ജൈടെക്സിൽ ഞാൻ ഈ ആൻഡ്രോയിഡിനെ കണ്ടു. കണ്ടാൽ ശരിക്കും ഒരു മനുഷ്യനാണെന്നേ തോന്നൂ.";
        // the first word mumbled, the rest clear
        let garbled = "ഡ്യൂ ഒക്ടോബറിൽ ജൈടെക്സിൽ ഞാൻ ഈ ആൻഡ്രോയിഡിനെ കണ്ടു. കണ്ടാൽ ശരിക്കും ഒരു മനുഷ്യനാണെന്നേ തോന്നൂ.";
        let s = score(said, garbled);
        assert_eq!(s.starts.len(), 2);
        assert!(s.cer < 0.25, "{s:?}");
        assert!(s.starts[0] > 0.5 && s.starts[1] == 0.0, "{s:?}");
        assert_eq!(s.start_cer, s.starts[0]);
        // noise in front of a clear second sentence
        let noisy = "ട്വന്റി ട്വന്റി വൺ ഒക്ടോബറിൽ ജൈടെക്സിൽ ഞാൻ ഈ ആൻഡ്രോയിഡിനെ കണ്ടു. ആഹ് ഹ കണ്ടാൽ ശരിക്കും ഒരു മനുഷ്യനാണെന്നേ തോന്നൂ.";
        let s = score(said, noisy);
        assert!(s.starts[0] == 0.0 && s.starts[1] > 0.2, "{s:?}");
        // a change late in a sentence does not touch its start
        let late = "ട്വന്റി ട്വന്റി വൺ ഒക്ടോബറിൽ ജൈടെക്സിൽ ഞാൻ ഈ ആൻഡ്രോയിഡിനെ കണ്ടി. കണ്ടാൽ ശരിക്കും ഒരു മനുഷ്യനാണെന്നേ തോന്നി.";
        let s = score(said, late);
        assert_eq!(s.start_cer, 0.0, "{s:?}");
        assert!(s.cer > 0.0);
    }

    #[test]
    fn sentence_starts_cover_six_syllables() {
        let st = sentence_starts("ഇതാണ് ബോസ്റ്റൺ ഡൈനാമിക്സിന്റെ സ്പോട്ട്. ശരി.");
        assert_eq!(st.len(), 2);
        // ഇ താ ണ് ബോ സ്റ്റ ൺ… : six syllables, then the seventh is left out
        let first: String = comparable("ഇതാണ് ബോസ്റ്റൺ ഡൈനാമിക്സിന്റെ സ്പോട്ട്.").iter().take(st[0].1).collect();
        assert_eq!(crate::malayalam::syllables(&first), START_SYLLABLES, "{first}");
        // a short sentence is all start
        assert_eq!(st[1].1, comparable("ശരി").len());
        assert!(sentence_starts("").is_empty());
    }

    const RATE: u32 = 24_000;

    /// A fake reading: a second of tone whose level says which attempt it is.
    fn tone(attempt: u32) -> Read {
        let a = 0.1 + 0.01 * attempt as f32;
        Read { samples: (0..RATE).map(|i| a * (i as f32 * 0.07).sin()).collect(), seed: 100 + u64::from(attempt) }
    }

    fn attempt_of(x: &[f32]) -> u32 {
        let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        ((peak - 0.1) / 0.01).round() as u32
    }

    #[test]
    fn retakes_stop_at_the_first_clear_reading() {
        let text = "ഇതാണ് ബോസ്റ്റൺ ഡൈനാമിക്സിന്റെ സ്പോട്ട്.";
        // a garbled reading: its first word mumbled, the rest said
        let rest: String = text.split_whitespace().skip(1).collect::<Vec<_>>().join(" ");
        let p = Policy { max_takes: 4, max_cer: 0.15, max_start_cer: 0.2 };
        for (bad, kept) in [(0, 0), (2, 2)] {
            let mut read = |attempt| Ok(tone(attempt));
            let mut hear = |x: &[f32]| Ok(if attempt_of(x) < bad { format!("ഹ്മ് {rest}") } else { text.to_string() });
            let (audio, c) = read_until_clear(text, &p, RATE, &mut read, &mut hear).unwrap();
            assert_eq!((c.takes, c.kept, c.passed, attempt_of(&audio)), (kept + 1, kept, true, kept));
            // every try reported with its seed; the kept one's seed reads it again
            assert_eq!(c.tries.len() as u32, kept + 1);
            assert_eq!(c.seed, 100 + u64::from(kept));
            assert!(c.tries.iter().take(kept as usize).all(|t| t.score.start_cer > p.max_start_cer));
            assert_eq!((c.heard.as_str(), c.cer, c.signal.len()), (text, 0.0, 0));
        }
    }

    #[test]
    fn never_clear_keeps_the_best_and_says_so() {
        let text = "ഇതാണ് ബോസ്റ്റൺ ഡൈനാമിക്സിന്റെ സ്പോട്ട്.";
        let p = Policy { max_takes: 3, max_cer: 0.15, max_start_cer: 0.2 };
        // every reading garbled; the second least
        let mut read = |attempt| Ok(tone(attempt));
        let mut hear = |x: &[f32]| {
            Ok(match attempt_of(x) {
                1 => "ഇതാ ബോസ്റ്റൺ ഡൈനാമിക്സിന്റെ".to_string(),
                _ => "ഹ്മ്".to_string(),
            })
        };
        let (audio, c) = read_until_clear(text, &p, RATE, &mut read, &mut hear).unwrap();
        assert_eq!((c.takes, c.kept, c.passed, attempt_of(&audio)), (3, 1, false, 1));
        // a clear hearing of a broken signal does not pass
        let mut read = |_| Ok(Read { samples: vec![0.0; 100], seed: 1 });
        let mut hear = |_: &[f32]| Ok(text.to_string());
        let (_, c) = read_until_clear(text, &p, RATE, &mut read, &mut hear).unwrap();
        assert!(!c.passed && c.takes == 3 && c.signal == vec!["empty".to_string()], "{c:?}");
        // a recogniser that fails is an error, not a pass
        let mut read = |attempt| Ok(tone(attempt));
        let mut deaf = |_: &[f32]| Err(TtsError::Model("no speech model".into()));
        assert!(read_until_clear(text, &p, RATE, &mut read, &mut deaf).is_err());
        // a reading that fails is an error too
        let mut broken = |_| Err(TtsError::Model("GPU".into()));
        let mut hear = |_: &[f32]| Ok(text.to_string());
        assert!(read_until_clear(text, &p, RATE, &mut broken, &mut hear).is_err());
        // hostile policies are refused
        for bad in [Policy { max_takes: 0, ..p }, Policy { max_takes: 99, ..p }, Policy { max_cer: f64::NAN, ..p }, Policy { max_start_cer: -1.0, ..p }] {
            assert!(read_until_clear(text, &bad, RATE, &mut |a| Ok(tone(a)), &mut |_| Ok(String::new())).is_err(), "{bad:?}");
        }
    }

    /// The second opinion: under `both` a reading the first recogniser rejects passes when the
    /// second hears it right; under `either` it needs both; the second is asked only when needed.
    #[test]
    fn a_second_opinion_decides_with_the_first() {
        let text = "ഇതാണ് ബോസ്റ്റൺ ഡൈനാമിക്സിന്റെ സ്പോട്ട്.";
        let p = Policy { max_takes: 3, max_cer: 0.15, max_start_cer: 0.2 };
        // the first recogniser mishears every reading; the second hears the second reading right
        let mut asked = 0;
        let mut read = |attempt| Ok(tone(attempt));
        let mut first = |_: &[f32]| Ok("ഹ്മ് ബോസ്റ്റൺ".to_string());
        let mut second = |x: &[f32]| {
            asked += 1;
            Ok(if attempt_of(x) >= 1 { text.to_string() } else { "ഹ്മ്".to_string() })
        };
        let mut j = Judges { second: Some(&mut second), ..Judges::default() };
        let (audio, c) = read_until_clear_judged(text, &p, RATE, &mut read, &mut first, &mut j).unwrap();
        assert_eq!((c.takes, c.kept, c.passed, attempt_of(&audio)), (2, 1, true, 1), "{c:?}");
        assert_eq!(c.second.as_ref().map(|o| o.passed), Some(true));
        assert_eq!(c.tries[0].flagged, vec!["first", "second"]);
        assert!(c.flagged.is_empty());
        assert_eq!(asked, 2);
        // both clear on the first reading: the second is never asked
        let mut asked = 0;
        let mut clear = |_: &[f32]| Ok(text.to_string());
        let mut second = |_: &[f32]| {
            asked += 1;
            Ok(String::new())
        };
        let mut j = Judges { second: Some(&mut second), ..Judges::default() };
        let (_, c) = read_until_clear_judged(text, &p, RATE, &mut |a| Ok(tone(a)), &mut clear, &mut j).unwrap();
        assert!(c.passed && c.second.is_none());
        assert_eq!(asked, 0);
        // `either`: the second's rejection fails a reading the first passes
        let mut deaf_second = |_: &[f32]| Ok("ഹ്മ്".to_string());
        let mut j = Judges { second: Some(&mut deaf_second), agreement: Agreement::Either, ..Judges::default() };
        let (_, c) = read_until_clear_judged(text, &p, RATE, &mut |a| Ok(tone(a)), &mut clear, &mut j).unwrap();
        assert!(!c.passed && c.takes == 3 && c.flagged == vec!["second"], "{c:?}");
        // a second recogniser that fails gives no opinion: the first decides
        let mut broken = |_: &[f32]| Err(TtsError::Model("out of memory".into()));
        let mut j = Judges { second: Some(&mut broken), ..Judges::default() };
        let (_, c) = read_until_clear_judged(text, &p, RATE, &mut |a| Ok(tone(a)), &mut first, &mut j).unwrap();
        assert!(!c.passed && c.second.is_none());
    }

    /// The voice gate with stub similarities: a reading that drifted from the voice is read again
    /// even when every word was heard, and the closest is kept when none is close enough.
    #[test]
    fn the_voice_gate_retakes_readings_that_drift() {
        let text = "ഇതാണ് ബോസ്റ്റൺ ഡൈനാമിക്സിന്റെ സ്പോട്ട്.";
        let p = Policy { max_takes: 4, max_cer: 0.15, max_start_cer: 0.2 };
        let mut clear = |_: &[f32]| Ok(text.to_string());
        // similarity rises with the attempt: 0.2, 0.4, 0.6…
        let mut sim = |x: &[f32]| Ok(0.2 + 0.2 * f64::from(attempt_of(x)));
        let mut j = Judges { voice: Some(&mut sim), min_similarity: 0.5, ..Judges::default() };
        let (audio, c) = read_until_clear_judged(text, &p, RATE, &mut |a| Ok(tone(a)), &mut clear, &mut j).unwrap();
        assert_eq!((c.takes, c.kept, c.passed, attempt_of(&audio)), (3, 2, true, 2), "{c:?}");
        assert_eq!(c.tries[0].flagged, vec!["voice"]);
        assert!((c.similarity.unwrap() - 0.6).abs() < 1e-9);
        // never close enough: the closest kept, flagged
        let mut low = |x: &[f32]| Ok(0.1 * f64::from(attempt_of(x)));
        let mut j = Judges { voice: Some(&mut low), min_similarity: 0.9, ..Judges::default() };
        let (_, c) = read_until_clear_judged(text, &p, RATE, &mut |a| Ok(tone(a)), &mut clear, &mut j).unwrap();
        assert!(!c.passed && c.kept == 3 && c.flagged == vec!["voice"], "{c:?}");
        // NaN similarity and a NaN threshold are no verdict, never a crash
        let mut nan = |_: &[f32]| Ok(f64::NAN);
        let mut j = Judges { voice: Some(&mut nan), min_similarity: f64::NAN, ..Judges::default() };
        let (_, c) = read_until_clear_judged(text, &p, RATE, &mut |a| Ok(tone(a)), &mut clear, &mut j).unwrap();
        assert!(c.passed && c.similarity.is_none());
    }

    /// The check's name changes with its judges, so a reading checked one way is not served for
    /// another; one recogniser alone keeps the name it always had.
    #[test]
    fn the_check_is_named_by_its_judges() {
        struct H(&'static str);
        impl Hearer for H {
            fn hear(&self, _: &[f32], _: u32) -> Result<String, String> {
                Ok(String::new())
            }
            fn id(&self) -> String {
                self.0.into()
            }
        }
        struct V;
        impl VoiceMatch for V {
            fn similarity(&self, _: &[f32], _: u32) -> Result<f64, String> {
                Ok(1.0)
            }
            fn id(&self) -> String {
                "ecapa me".into()
            }
        }
        let (a, b, v) = (H("a v1"), H("b v1"), V);
        let one = Strict::new(&a, Policy::default());
        assert_eq!(one.id(), "a v1");
        let two = Strict { second: Some(&b), ..Strict::new(&a, Policy::default()) };
        assert_eq!(two.id(), "a v1 + b v1 (both)");
        let gated = Strict { voice: Some(&v), min_similarity: 0.3, ..Strict::new(&a, Policy::default()) };
        assert_eq!(gated.id(), "a v1, voice ecapa me >= 0.30");
        let first_only = Strict { second: Some(&b), agreement: Agreement::First, ..Strict::new(&a, Policy::default()) };
        assert_eq!(first_only.id(), "a v1");
        assert_eq!(Agreement::from_id("either"), Some(Agreement::Either));
        assert_eq!(Agreement::from_id("most"), None);
    }

    #[test]
    fn english_numbers_compare_as_words() {
        assert_eq!(english_words("From 10 to 80% in 30 minutes."), "From ten to eighty percent in thirty minutes.");
        assert_eq!(english_words("It costs $1,600 & ships in 2021."), "It costs $ one thousand six hundred and ships in two thousand twenty one .");
        assert_eq!(english_words("version 5.3"), "version five point three");
        assert_eq!(english_words("0"), "zero");
        assert_eq!(english_words("100"), "one hundred");
        assert_eq!(english_words("1000000"), "one million");
        // hostile: a number too long to say is read digit by digit; nothing panics
        assert_eq!(english_words("12345678901234567890").split(' ').count(), 20);
        assert_eq!(english_words(""), "");
        assert_eq!(english_words("1,"), "one ,");
        let s = score(&english_words("Ten to 80% in half an hour."), &english_words("10 to eighty percent in half an hour"));
        assert_eq!(s.cer, 0.0, "{s:?}");
        let e = Policy::english();
        assert!(e.check().is_ok());
    }

    /// Syllable-like bursts of tone, `on` seconds each with `off` seconds of near-silence after.
    fn syllables(n: usize, on: f64, off: f64, level: f32) -> Vec<f32> {
        let mut x = Vec::new();
        for _ in 0..n {
            x.extend((0..(on * f64::from(RATE)) as usize).map(|i| level * (i as f32 * 0.05).sin()));
            x.extend(std::iter::repeat_n(0.0001, (off * f64::from(RATE)) as usize));
        }
        x
    }

    #[test]
    fn the_signal_scan_finds_what_breaks_a_reading() {
        // speech with short gaps: clean
        let clean = syllables(12, 0.18, 0.06, 0.3);
        assert!(signal(&clean, RATE).is_empty(), "{:?}", signal(&clean, RATE));
        // empty, invalid, silent
        assert_eq!(signal(&[], RATE), vec!["empty"]);
        assert_eq!(signal(&clean, 0), vec!["empty"]);
        let mut nan = clean.clone();
        nan[100] = f32::NAN;
        assert_eq!(signal(&nan, RATE), vec!["invalid samples"]);
        assert_eq!(signal(&vec![0.0; RATE as usize], RATE), vec!["silent"]);
        // a dropout: a second and a half of silence between words
        let mut gap = syllables(6, 0.18, 0.06, 0.3);
        gap.extend(std::iter::repeat_n(0.0, (1.5 * f64::from(RATE)) as usize));
        gap.extend(syllables(6, 0.18, 0.06, 0.3));
        let s = signal(&gap, RATE);
        assert!(s.len() == 1 && s[0].contains("silence mid-reading"), "{s:?}");
        // a burst at the start
        let mut burst = syllables(1, 0.2, 0.0, 0.99);
        burst.extend(syllables(12, 0.18, 0.06, 0.08));
        let s = signal(&burst, RATE);
        assert!(s.iter().any(|f| f.contains("burst")), "{s:?}");
        // clipping
        let clipped: Vec<f32> = clean.iter().map(|v| (v * 8.0).clamp(-1.0, 1.0)).collect();
        assert!(signal(&clipped, RATE).iter().any(|f| f.starts_with("clipping")));
    }

    /// A long reading is heard in stretches the recogniser can decode, cut in its pauses, every
    /// sample heard once and in order.
    #[test]
    fn long_readings_are_heard_in_pieces_cut_at_pauses() {
        // 30 s: words of 0.5 s with 0.1 s gaps, and a 0.4 s pause every 4 s
        let mut x = Vec::new();
        let mut pauses = Vec::new();
        while x.len() < 30 * RATE as usize {
            for _ in 0..6 {
                x.extend(syllables(1, 0.5, 0.1, 0.3));
            }
            pauses.push((x.len(), x.len() + (0.4 * f64::from(RATE)) as usize));
            x.extend(std::iter::repeat_n(0.0, (0.4 * f64::from(RATE)) as usize));
        }
        let mut pieces: Vec<usize> = Vec::new();
        let text = hear_in_pieces(&x, RATE, HEARD_SECONDS, &mut |p| {
            pieces.push(p.len());
            Ok(format!("ഭാഗം{}", pieces.len()))
        })
        .unwrap();
        assert!(pieces.len() >= 3 && pieces.iter().all(|n| *n as f64 <= HEARD_SECONDS * f64::from(RATE)), "{pieces:?}");
        assert_eq!(pieces.iter().sum::<usize>(), x.len(), "every sample heard once");
        assert_eq!(text, (1..=pieces.len()).map(|i| format!("ഭാഗം{i}")).collect::<Vec<_>>().join(" "));
        // every cut falls in a pause
        let mut at = 0;
        for n in &pieces[..pieces.len() - 1] {
            at += n;
            assert!(pauses.iter().any(|(a, b)| (*a..=*b).contains(&at)), "a cut at {:.2} s", at as f64 / f64::from(RATE));
        }
        // a short reading is heard whole; a failing recogniser fails the hearing
        let mut calls = 0;
        hear_in_pieces(&x[..RATE as usize], RATE, HEARD_SECONDS, &mut |_| {
            calls += 1;
            Ok(String::new())
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert!(hear_in_pieces(&x, RATE, HEARD_SECONDS, &mut |_| Err(TtsError::Model("deaf".into()))).is_err());
        // hostile: no rate, nothing
        assert!(hear_in_pieces(&x, 0, HEARD_SECONDS, &mut |_| Ok("x".into())).is_ok());
        assert_eq!(hear_in_pieces(&[], RATE, HEARD_SECONDS, &mut |_| Ok(String::new())).unwrap(), "");
    }
}
