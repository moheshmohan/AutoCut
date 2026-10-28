//! Voice styles of Malayalam narration (IndicF5): the feeling a line is read with.
//!
//! IndicF5 continues a reference clip, and the clip dictates the emotion and prosody of what it
//! reads. A cloned voice therefore keeps one reference per style (its main reference is the plain
//! one; `crate::clone`'s style references are the others), as F5-TTS's multi-style generation
//! keeps one reference audio and text per style and switches with `[style]` tags.
//!
//! - **Styles** ([`STYLES`]): the narration deliveries but `plain` (`conversational`, `warm`,
//!   `excited`, `amazed`, `curious`, `calm`, `dramatic`).
//! - **Fallbacks** ([`fallbacks`], [`resolve`]): a style the voice has no reference for reads
//!   after the nearest one it has, else after the main reference:
//!
//!   | asked | then | then |
//!   |---|---|---|
//!   | `excited` | `amazed` | `conversational` |
//!   | `amazed` | `excited` | `conversational` |
//!   | `curious` | `conversational` | |
//!   | `conversational` | `warm` | |
//!   | `warm` | `conversational` | |
//!   | `calm` | `warm` | |
//!   | `dramatic` | `calm` | `warm` |
//!
//! - **Sampler presets** ([`preset`]): each style's temperature (the scale of the flow-matching
//!   sampler's initial noise), emphasis (classifier-free guidance) and speed. Explicit settings
//!   override them; `plain` has none (the voice's own reading, bit-identical to before styles).
//! - **Inline tags** ([`tag`]): `[excited] … [calm] …` switch the style for what follows in the
//!   same script (`[plain]` / `[neutral]` go back to the main reference); `[breath]` is a
//!   breath-length pause of room tone. Tags are never read aloud.
//! - **Natural pauses** ([`gap_ms`]): the pause between two utterances follows how the first one
//!   ends (a full stop, `?`, `!`, `…`, a comma, a line or paragraph break), scaled by the style
//!   (excited shorter, calm longer), with a small variation that is deterministic per seed.

use serde::Serialize;

/// The styles a cloned voice may hold a reference for.
pub const STYLES: [&str; 7] = ["conversational", "warm", "excited", "amazed", "curious", "calm", "dramatic"];

/// Version of the presets and pauses below: the narration cache keys styled readings, and
/// readings spaced by natural pauses, on it (bump it when they change, so those read again).
pub const VERSION: u32 = 1;

/// The style id of `id` (trimmed, any case), if it is one of [`STYLES`].
pub fn find(id: &str) -> Option<&'static str> {
    let n = id.trim().to_ascii_lowercase();
    STYLES.iter().copied().find(|s| *s == n)
}

/// The styles tried after `style` when the voice has no reference for it (see the module docs),
/// nearest first.
pub fn fallbacks(style: &str) -> &'static [&'static str] {
    match style {
        "excited" => &["amazed", "conversational"],
        "amazed" => &["excited", "conversational"],
        "curious" => &["conversational"],
        "conversational" => &["warm"],
        "warm" => &["conversational"],
        "calm" => &["warm"],
        "dramatic" => &["calm", "warm"],
        _ => &[],
    }
}

/// The reference `style` reads after, given the styles the voice `has`: the style itself, else
/// its nearest fallback, else `None` (the main reference). `None` asks for the main reference.
pub fn resolve(style: Option<&str>, has: &dyn Fn(&str) -> bool) -> Option<&'static str> {
    let s = find(style?)?;
    std::iter::once(s).chain(fallbacks(s).iter().copied()).find(|c| has(c))
}

/// What a tag in a script asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tag {
    /// Read what follows in this style.
    Style(&'static str),
    /// Read what follows after the main reference, plainly (`[plain]`, `[neutral]`).
    Plain,
    /// A breath-length pause of room tone (`[breath]`).
    Breath,
}

/// The tag `[inner]` is (`inner` without its brackets, trimmed, any case), if it is one.
pub fn tag(inner: &str) -> Option<Tag> {
    let n = inner.trim().to_ascii_lowercase();
    match n.as_str() {
        "plain" | "neutral" => Some(Tag::Plain),
        "breath" => Some(Tag::Breath),
        _ => find(&n).map(Tag::Style),
    }
}

/// The style in force for a part of a script: the line's own (no tag yet), plain, or a tag's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Asked {
    Line,
    Plain,
    Style(&'static str),
}

impl Asked {
    /// The style asked for, given the line's own (`None`: plain).
    pub fn style(self, line: Option<&'static str>) -> Option<&'static str> {
        match self {
            Asked::Line => line,
            Asked::Plain => None,
            Asked::Style(s) => Some(s),
        }
    }
}

/// How the flow-matching sampler reads a style.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Sampler {
    /// Scale of the initial noise ([`TEMPERATURE`]; 1 = the model's own).
    pub temperature: f32,
    /// Classifier-free guidance strength ([`EMPHASIS`]; 2 = the model's own).
    pub emphasis: f32,
    /// Speaking speed on top of the natural rate and the narration's pace ([`SPEED`]).
    pub speed: f32,
    /// The reading's pitch range after synthesis ([`EXPRESSIVENESS`]; 1 = as synthesized, the
    /// audio untouched): deviations from the median pitch scaled by it ([`reshaping`]).
    pub expressiveness: f32,
}

impl Sampler {
    /// The model's own reading (plain).
    pub const PLAIN: Sampler = Sampler { temperature: 1.0, emphasis: 2.0, speed: 1.0, expressiveness: 1.0 };
}

/// Temperature, emphasis, speed and expressiveness a narration may ask for (inclusive).
pub const TEMPERATURE: (f32, f32) = (0.3, 1.5);
pub const EMPHASIS: (f32, f32) = (1.0, 4.0);
pub const SPEED: (f32, f32) = (0.7, 1.4);
pub const EXPRESSIVENESS: (f32, f32) = (0.5, 2.0);

/// The sampler preset of `style` (`None` for plain: the reading's own settings). Measured on the
/// owner's voice (`docs/malayalam-narration.md`, "Voice styles"): livelier styles get a little
/// more guidance and noise and speak a touch faster; calm ones less, slower. Expressiveness:
/// IndicF5's readings of the owner span 3.0 semitones (10th–90th percentile of the pitch)
/// against 4.0 in his own speech, so the livelier styles widen it.
pub fn preset(style: Option<&str>) -> Option<Sampler> {
    let (temperature, emphasis, speed, expressiveness) = match style? {
        "conversational" => (1.0, 2.0, 1.02, 1.1),
        "warm" => (0.95, 2.0, 0.98, 1.15),
        "excited" => (1.1, 2.4, 1.06, 1.3),
        "amazed" => (1.1, 2.3, 1.0, 1.35),
        "curious" => (1.05, 2.1, 1.0, 1.2),
        "calm" => (0.85, 1.8, 0.95, 1.0),
        "dramatic" => (0.9, 2.2, 0.92, 1.2),
        _ => return None,
    };
    Some(Sampler { temperature, emphasis, speed, expressiveness })
}

/// The rise a reading's question gets at its end when its intonation is reshaped, semitones.
pub const QUESTION_RISE_ST: f64 = 2.0;
/// The lift of stressed syllables per unit of expressiveness over 1, dB (1.3: 0.9 dB).
pub const EMPHASIS_DB_PER_UNIT: f64 = 3.0;

/// How an utterance's intonation is reshaped after synthesis at `expressiveness`: the pitch range
/// (the deviations from its median scaled), the rise at its end (a question: `question`) and the
/// lift of its stressed syllables, dB. `None` at 1 (and for a value that is not finite): the
/// audio is left exactly as synthesized.
pub fn reshaping(expressiveness: f32, question: bool) -> Option<(f64, f64, f64)> {
    let e = f64::from(expressiveness);
    if !e.is_finite() || e == 1.0 {
        return None;
    }
    let e = e.clamp(f64::from(EXPRESSIVENESS.0), f64::from(EXPRESSIVENESS.1));
    let rise = if question { QUESTION_RISE_ST } else { 0.0 };
    Some((e, rise, ((e - 1.0) * EMPHASIS_DB_PER_UNIT).clamp(0.0, 6.0)))
}

/// `v` within `range`, else a message saying so (`what`: the parameter's name).
pub fn check(what: &str, v: f32, range: (f32, f32)) -> Result<(), String> {
    if v.is_finite() && (range.0..=range.1).contains(&v) { Ok(()) } else { Err(format!("`{what}` must be a number from {} to {} (not {v})", range.0, range.1)) }
}

/// How an utterance ends, which sets the pause after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Boundary {
    /// Split between words (a sentence too long for one utterance).
    Word,
    /// At a comma, colon or semicolon.
    Comma,
    Sentence,
    Question,
    Exclaim,
    /// `…` or `...`: a trailing pause.
    Ellipsis,
    /// A line break in the script.
    Line,
    /// A blank line.
    Paragraph,
}

/// How `text` (an utterance) ends.
pub fn boundary_after(text: &str) -> Boundary {
    let t = text.trim_end_matches(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | ')' | '”' | '’'));
    if t.ends_with("...") || t.ends_with('…') {
        Boundary::Ellipsis
    } else if t.ends_with('?') {
        Boundary::Question
    } else if t.ends_with('!') {
        Boundary::Exclaim
    } else if t.ends_with(['.', '।']) {
        Boundary::Sentence
    } else if t.ends_with([',', ':', ';']) {
        Boundary::Comma
    } else {
        Boundary::Word
    }
}

/// The pause after `b` in a plain reading, ms (each utterance also keeps 40 ms of its own silence
/// at either end).
pub fn base_gap_ms(b: Boundary) -> f64 {
    match b {
        Boundary::Word => 110.0,
        Boundary::Comma => 180.0,
        Boundary::Sentence => 320.0,
        Boundary::Question => 400.0,
        Boundary::Exclaim => 340.0,
        Boundary::Ellipsis => 560.0,
        Boundary::Line => 480.0,
        Boundary::Paragraph => 760.0,
    }
}

/// How a style stretches its pauses (excited shorter, calm and dramatic longer).
pub fn pause_scale(style: Option<&str>) -> f64 {
    match style {
        Some("excited") => 0.8,
        Some("amazed") => 0.85,
        Some("conversational") => 0.9,
        Some("curious") => 0.95,
        Some("calm") => 1.25,
        Some("dramatic") => 1.35,
        _ => 1.0,
    }
}

/// A `[breath]`: room tone this long, ms (before the style's scale).
pub const BREATH_MS: f64 = 420.0;
/// The pause before an utterance read before natural pauses (and by readings that keep it).
pub const FIXED_GAP_MS: u32 = 250;
/// How much a pause varies, either way (a share of it).
const JITTER: f64 = 0.12;

/// A number in −1..1 from `seed` (SplitMix64).
fn jitter(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// The pause of `base` ms read in `style` with the variation of `seed`, ms.
fn scaled(base: f64, style: Option<&str>, seed: u64) -> u32 {
    let ms = base * pause_scale(style) * (1.0 + JITTER * jitter(seed));
    if ms.is_finite() { ms.round().clamp(0.0, 5000.0) as u32 } else { 0 }
}

/// The pause after boundary `b`, read in `style`, with the variation of `seed`, ms.
pub fn gap_ms(b: Boundary, style: Option<&str>, seed: u64) -> u32 {
    scaled(base_gap_ms(b), style, seed)
}

/// A `[breath]` read in `style` with the variation of `seed`, ms.
pub fn breath_ms(style: Option<&str>, seed: u64) -> u32 {
    scaled(BREATH_MS, style, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn styles_tags_and_fallbacks() {
        assert_eq!(find(" Excited "), Some("excited"));
        assert_eq!(find("plain"), None, "plain is the main reference");
        assert_eq!(find("angry"), None);
        assert_eq!(tag("Breath"), Some(Tag::Breath));
        assert_eq!((tag("plain"), tag(" neutral")), (Some(Tag::Plain), Some(Tag::Plain)));
        assert_eq!(tag("calm"), Some(Tag::Style("calm")));
        assert_eq!(tag("pause 1s"), None);
        // every fallback is a style, never itself, and every chain ends
        for s in STYLES {
            assert!(fallbacks(s).iter().all(|f| STYLES.contains(f) && *f != s), "{s}");
        }
        let has = |set: &'static [&'static str]| move |s: &str| set.contains(&s);
        assert_eq!(resolve(Some("amazed"), &has(&["excited", "calm"])), Some("excited"));
        assert_eq!(resolve(Some("amazed"), &has(&["conversational"])), Some("conversational"));
        assert_eq!(resolve(Some("amazed"), &has(&["amazed", "excited"])), Some("amazed"));
        assert_eq!(resolve(Some("dramatic"), &has(&["calm"])), Some("calm"));
        assert_eq!(resolve(Some("curious"), &has(&["calm"])), None, "the main reference");
        assert_eq!(resolve(None, &has(&["calm"])), None);
        assert_eq!(resolve(Some("nope"), &has(&["calm"])), None);
        assert_eq!(Asked::Line.style(Some("warm")), Some("warm"));
        assert_eq!(Asked::Plain.style(Some("warm")), None);
        assert_eq!(Asked::Style("calm").style(None), Some("calm"));
    }

    #[test]
    fn presets_are_within_the_ranges() {
        assert_eq!(preset(None), None);
        assert_eq!(preset(Some("plain")), None);
        for s in STYLES {
            let p = preset(Some(s)).unwrap();
            assert!(check("temperature", p.temperature, TEMPERATURE).is_ok(), "{s}");
            assert!(check("emphasis", p.emphasis, EMPHASIS).is_ok(), "{s}");
            assert!(check("speed", p.speed, SPEED).is_ok(), "{s}");
            assert!(check("expressiveness", p.expressiveness, EXPRESSIVENESS).is_ok(), "{s}");
        }
        let excited = preset(Some("excited")).unwrap();
        let calm = preset(Some("calm")).unwrap();
        assert!(excited.emphasis > calm.emphasis && excited.temperature > calm.temperature && excited.speed > calm.speed);
        assert!(excited.expressiveness > calm.expressiveness && calm.expressiveness == 1.0);
        // reshaping: none at 1 (bit-identical), a range, a question's rise and a small lift otherwise
        assert_eq!(reshaping(1.0, true), None);
        assert_eq!(reshaping(f32::NAN, false), None);
        assert_eq!(reshaping(1.3, false), Some((f64::from(1.3f32), 0.0, (f64::from(1.3f32) - 1.0) * 3.0)));
        assert_eq!(reshaping(1.3, true).map(|r| r.1), Some(QUESTION_RISE_ST));
        assert_eq!(reshaping(0.7, false).map(|r| r.2), Some(0.0), "a narrower range lifts nothing");
        assert_eq!(reshaping(9.0, false).map(|r| (r.0, r.2)), Some((2.0, 3.0)), "bounded");
        for bad in [f32::NAN, f32::INFINITY, 0.29, 1.51] {
            assert!(check("temperature", bad, TEMPERATURE).unwrap_err().contains("temperature"));
        }
    }

    #[test]
    fn pauses_follow_the_punctuation_and_the_style() {
        assert_eq!(boundary_after("ഇത് ശരിയാണ്."), Boundary::Sentence);
        assert_eq!(boundary_after("ശരിയാണോ? "), Boundary::Question);
        assert_eq!(boundary_after("കൊള്ളാം!\""), Boundary::Exclaim);
        assert_eq!(boundary_after("പിന്നെ..."), Boundary::Ellipsis);
        assert_eq!(boundary_after("പിന്നെ…"), Boundary::Ellipsis);
        assert_eq!(boundary_after("ആദ്യം,"), Boundary::Comma);
        assert_eq!(boundary_after("ഇത്"), Boundary::Word);
        assert_eq!(boundary_after(""), Boundary::Word);
        // ordered: words < commas < sentences < questions < ellipses < paragraphs
        let g = |b| gap_ms(b, None, 7);
        assert!(g(Boundary::Word) < g(Boundary::Comma) && g(Boundary::Comma) < g(Boundary::Sentence));
        assert!(g(Boundary::Sentence) < g(Boundary::Question) && g(Boundary::Question) < g(Boundary::Ellipsis));
        assert!(g(Boundary::Ellipsis) < g(Boundary::Paragraph) && g(Boundary::Line) < g(Boundary::Paragraph));
        // excited is quicker than plain, calm slower
        for seed in 0..50 {
            assert!(gap_ms(Boundary::Sentence, Some("excited"), seed) < gap_ms(Boundary::Sentence, Some("calm"), seed));
        }
        // deterministic per seed, varying within ±12 %
        assert_eq!(gap_ms(Boundary::Sentence, None, 3), gap_ms(Boundary::Sentence, None, 3));
        let all: Vec<u32> = (0..200).map(|s| gap_ms(Boundary::Sentence, None, s)).collect();
        assert!(all.iter().all(|ms| (281..=359).contains(ms)), "{all:?}");
        assert!(all.iter().any(|ms| *ms != all[0]), "it varies");
        assert!((0..100).all(|s| jitter(s).abs() <= 1.0));
        assert!((369..=471).contains(&breath_ms(None, 1)));
    }
}
