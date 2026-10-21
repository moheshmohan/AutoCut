//! Auditioning references: the fixed test set each candidate speaks, how a reading is scored
//! against held-out genuine speech of the speaker, and the ranking.

use serde::Serialize;

use super::{Target, cosine};

/// English test lines (VoxCPM2): showcase-style, with starts that are hard to get right (a vowel,
/// a name, a cluster after a pause) and a long line.
pub const ENGLISH: &[&str] = &[
    "Honestly, I did not expect a robot to remember my name, but this one did.",
    "Every robot at the expo had its own personality, and some of them were surprisingly funny.",
    "Ameca looked straight at me, raised an eyebrow, and asked me where I was from.",
    "Under the bright lights of the hall, a tiny delivery robot waited patiently at the crossing.",
];

/// Malayalam test lines (IndicF5): spoken Malayalam as the showcase scripts are written, English
/// loanwords in Malayalam script, sentences that start on a vowel and on a loanword, and a line
/// of two sentences (a start inside a generation).
pub const MALAYALAM: &[&str] = &[
    "ഇതാണ് ബോസ്റ്റൺ ഡൈനാമിക്സിന്റെ സ്പോട്ട്. ഒരു റോബോട്ട് ഡോഗ് ആണ് ഇത്.",
    "എക്സ്പോയിൽ വെച്ച് ഈ റോബോട്ട് നേരെ എന്റെ അടുത്തേക്ക് നടന്നു വന്നു.",
    "ഓരോ റോബോട്ടിനും അതിന്റേതായ ഒരു പേഴ്സണാലിറ്റി ഉണ്ട്, അതാണ് ശരിക്കും ഇന്ററസ്റ്റിംഗ്.",
    "അമേക്ക എന്നെ നോക്കി ചിരിച്ചു, എന്നിട്ട് ഞാൻ എവിടുന്നാണെന്ന് ചോദിച്ചു.",
];

/// The test lines of an engine.
pub fn lines(t: Target) -> &'static [&'static str] {
    match t {
        Target::Voxcpm => ENGLISH,
        Target::Indicf5 => MALAYALAM,
    }
}

/// Genuine speech of the speaker the readings are compared with (never the reference itself).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Pool {
    #[serde(skip)]
    pub colour: Vec<f32>,
    /// Median pitch of the speech, Hz.
    pub pitch_hz: Option<f64>,
    pub seconds: f64,
    /// Where it came from (clip and seconds), for the report.
    pub from: Vec<String>,
}

/// One reading of a test line, scored.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Take {
    pub line: usize,
    pub text: String,
    pub seconds: f64,
    /// Tone-colour cosine to the held-out speech.
    pub similarity: f64,
    pub pitch_hz: Option<f64>,
    pub heard: String,
    pub cer: f64,
    /// The worst sentence start ([`crate::heard::Score::start_cer`]).
    pub start_cer: f64,
    /// The signal scan's findings ([`crate::heard::signal`]): clipping, dropouts, bursts.
    pub signal: Vec<String>,
    pub lufs: f64,
    pub peak_db: f64,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub file: String,
}

/// Score a reading `x` (mono at `rate`) of `text`, which the speech model heard as `heard`;
/// `colour` is its embedding.
pub fn take(line: usize, text: &str, heard: &str, x: &[f32], rate: u32, colour: &[f32], pool: &Pool) -> Take {
    let s = crate::heard::score(text, heard);
    let peak = x.iter().filter(|v| v.is_finite()).fold(0f32, |m, v| m.max(v.abs()));
    let lufs = filmcraft_audio_dsp::loudness::measure(&[x], f64::from(rate.max(1))).integrated_lufs;
    Take {
        line,
        text: text.to_string(),
        seconds: x.len() as f64 / f64::from(rate.max(1)),
        similarity: cosine(colour, &pool.colour),
        pitch_hz: crate::clone::median_pitch(x, rate),
        heard: heard.to_string(),
        cer: s.cer,
        start_cer: s.start_cer,
        signal: crate::heard::signal(x, rate),
        lufs: if lufs.is_finite() { lufs } else { -120.0 },
        peak_db: 20.0 * f64::from(peak.max(1e-6)).log10(),
        file: String::new(),
    }
}

/// A reference auditioned: its readings and what they add up to.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Scored {
    /// The candidate's id (`voxcpm-1`), `current`, or a variant (`voxcpm-1+voxcpm-2`,
    /// `voxcpm-1 cleaned`).
    pub id: String,
    pub target: Option<Target>,
    /// `current`, `candidate`, `multi` (two references joined) or `cleaned` (a light clean-up).
    pub kind: String,
    /// Where the reference is from.
    pub source: String,
    pub reference_seconds: f64,
    /// The reference itself against the held-out speech: how close genuine speech of that
    /// recording is (the ceiling a reading of it can reach).
    pub reference_similarity: f64,
    pub takes: Vec<Take>,
    pub similarity: f64,
    /// Median pitch of the readings against the held-out speech's, semitones.
    pub pitch_delta_st: Option<f64>,
    pub cer: f64,
    pub start_cer: f64,
    pub worst_start_cer: f64,
    /// Signal findings over every reading.
    pub artefacts: usize,
    /// [`score`]: similarity, less what was misheard and the artefacts.
    pub score: f64,
    /// 1 = best of its engine.
    pub rank: usize,
    /// Why it could not be auditioned, when it could not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// What ranks references: the readings' similarity to the speaker, less half their character
/// error rate and half their mean sentence-start error rate (a garbled start is what a listener
/// hears first), less 0.02 a signal finding and 0.005 a semitone of pitch away from the speaker.
pub fn score(similarity: f64, cer: f64, start_cer: f64, artefacts: usize, pitch_delta_st: Option<f64>) -> f64 {
    let s = similarity - 0.5 * cer.min(2.0) - 0.5 * start_cer.min(1.0) - 0.02 * artefacts as f64 - 0.005 * pitch_delta_st.map_or(0.0, f64::abs).min(12.0);
    if s.is_finite() { s } else { f64::NEG_INFINITY }
}

/// The readings of one reference added up.
pub fn summarise(mut s: Scored, takes: Vec<Take>, pool: &Pool) -> Scored {
    let n = takes.len().max(1) as f64;
    let mean = |f: &dyn Fn(&Take) -> f64| takes.iter().map(f).sum::<f64>() / n;
    s.similarity = mean(&|t| t.similarity);
    s.cer = mean(&|t| t.cer);
    s.start_cer = mean(&|t| t.start_cer);
    s.worst_start_cer = takes.iter().map(|t| t.start_cer).fold(0.0, f64::max);
    s.artefacts = takes.iter().map(|t| t.signal.len()).sum();
    let mut pitches: Vec<f64> = takes.iter().filter_map(|t| t.pitch_hz).collect();
    pitches.sort_by(f64::total_cmp);
    s.pitch_delta_st = match (pitches.get(pitches.len() / 2), pool.pitch_hz) {
        (Some(p), Some(q)) if *p > 0.0 && q > 0.0 => Some(12.0 * (p / q).log2()),
        _ => None,
    };
    s.takes = takes;
    s.score = if s.takes.is_empty() { f64::NEG_INFINITY } else { score(s.similarity, s.cer, s.start_cer, s.artefacts, s.pitch_delta_st) };
    s
}

/// Rank the references of each engine (best first; ones that failed last).
pub fn rank(v: &mut [Scored]) {
    v.sort_by(|a, b| a.target.map(Target::id).cmp(&b.target.map(Target::id)).then(b.score.total_cmp(&a.score)));
    let mut last: Option<Target> = None;
    let mut r = 0;
    for s in v.iter_mut() {
        if s.target != last {
            last = s.target;
            r = 0;
        }
        r += 1;
        s.rank = r;
    }
}
