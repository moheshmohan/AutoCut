//! The voice lab (feature `kokoro`): better reference recordings for a cloned voice, found in
//! the speaker's own clips and auditioned before one replaces the voice's reference.
//!
//! - **Finding** ([`scan`], [`rank`]): a clip's speech is cut into phrases at its pauses
//!   ([`signal`]); chunks of 6–12 s are checked against the speaker (the cosine of their speaker
//!   embedding, OpenVoice's tone colour, with the voice's own reference); the stretches by the
//!   speaker are heard (words with times and the spoken language) and every run of whole phrases
//!   that fits an engine ([`Target::bounds`]: VoxCPM 10–30 s, IndicF5 6–12 s) is scored on
//!   speaker similarity, signal quality ([`signal::Quality`]: signal to noise, clipping,
//!   reverberation, clarity, music under it, bandwidth), whether it is whole sentences (cut at
//!   pauses where a sentence ends), its length and its language (English for VoxCPM, Malayalam
//!   for IndicF5).
//! - **Auditioning** ([`audition`]): each candidate speaks a fixed test set, and the readings are
//!   scored on speaker similarity against held-out genuine speech of the speaker (never the
//!   reference itself), on what a speech model hears (character error rate, and the first
//!   syllables of each sentence on their own: [`crate::heard::score`]) and on artefacts.
//!
//! - **Styles** ([`styles`]): IndicF5 candidates sorted by delivery — calm, plain,
//!   conversational, warm, excited, amazed, curious — from their prosody (pitch range and
//!   movement, energy variation, syllable rate, pitch level, the final contour) against the
//!   speaker's own spread over all his candidates.
//!
//! The models are the caller's ([`Models`]): pure functions here, tested on synthetic audio.

pub mod audition;
pub mod signal;
pub mod styles;

use serde::{Deserialize, Serialize};

use crate::TtsError;
pub use signal::{Quality, Span};

/// What a reference is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Target {
    /// VoxCPM2 speaking English: the voice sample, an isolated reference.
    Voxcpm,
    /// IndicF5 speaking Malayalam: a clip it continues, with its exact words.
    Indicf5,
}

/// How long a reference is (seconds): accepted, and best.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub min: f64,
    pub max: f64,
    pub ideal: (f64, f64),
}

impl Target {
    pub const ALL: [Target; 2] = [Target::Voxcpm, Target::Indicf5];

    pub fn id(self) -> &'static str {
        match self {
            Target::Voxcpm => "voxcpm",
            Target::Indicf5 => "indicf5",
        }
    }

    pub fn from_id(s: &str) -> Option<Target> {
        match s {
            "voxcpm" => Some(Target::Voxcpm),
            "indicf5" => Some(Target::Indicf5),
            _ => None,
        }
    }

    /// VoxCPM2 carries a voice best from 10–30 s (the voice sample keeps 10–25 s); IndicF5
    /// generates its reference again with every utterance, so 6–12 s (8 s ending at a sentence
    /// end was the author's preferred cut).
    pub fn bounds(self) -> Bounds {
        match self {
            Target::Voxcpm => Bounds { min: 10.0, max: 30.0, ideal: (15.0, 25.0) },
            Target::Indicf5 => Bounds { min: 6.0, max: 12.0, ideal: (8.0, 11.0) },
        }
    }

    /// The language the engine speaks (ISO 639-1).
    pub fn language(self) -> &'static str {
        match self {
            Target::Voxcpm => "en",
            Target::Indicf5 => "ml",
        }
    }

    /// How well speech in `language` suits the engine (0…1): the speaker's accent and rhythm in
    /// the language spoken carry over best; IndicF5 is prompted across languages too.
    pub fn language_fit(self, language: &str) -> f64 {
        match (self, language) {
            (Target::Voxcpm, "en") | (Target::Indicf5, "ml") => 1.0,
            (Target::Indicf5, "en") => 0.7,
            (Target::Voxcpm, "ml") => 0.6,
            _ => 0.4,
        }
    }
}

/// 1 within the ideal length, falling to 0.5 at the accepted bounds, 0 outside them.
pub fn length_fit(secs: f64, b: &Bounds) -> f64 {
    if !secs.is_finite() || secs < b.min || secs > b.max {
        return 0.0;
    }
    if secs < b.ideal.0 {
        return 0.5 + 0.5 * (secs - b.min) / (b.ideal.0 - b.min).max(1e-9);
    }
    if secs > b.ideal.1 {
        return 0.5 + 0.5 * (b.max - secs) / (b.max - b.ideal.1).max(1e-9);
    }
    1.0
}

/// A word heard, with its times (seconds) and attached punctuation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// What a speech model heard: the language (ISO 639-1) and the words.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Heard {
    pub language: String,
    pub words: Vec<Word>,
}

/// The models the lab listens with (the caller's: the engine's OpenVoice converter and Whisper).
pub trait Models {
    /// A speaker embedding of speech `x` (mono at `rate`): OpenVoice's tone colour.
    fn colour(&mut self, x: &[f32], rate: u32) -> Result<Vec<f32>, TtsError>;
    /// The words of `x` with their times (seconds from its start) and the language spoken.
    fn transcribe(&mut self, x: &[f32], rate: u32) -> Result<Heard, TtsError>;
}

/// The cosine of two embeddings (0 when either is empty or zero).
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let n = |v: &[f32]| v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    let d = n(a) * n(b);
    if d < 1e-12 || !dot.is_finite() { 0.0 } else { dot / d }
}

/// Languages a multilingual Whisper reports for Malayalam speech: its base model writes
/// Malayalam in Tamil script and often says so (crate README).
pub fn normalise_language(l: &str) -> String {
    match l {
        "ml" | "ta" => "ml".into(),
        "" => "unknown".into(),
        other => other.to_string(),
    }
}

/// How the finder works.
#[derive(Clone, Debug, PartialEq)]
pub struct FindSettings {
    /// Speech whose embedding is at least this close to the speaker's is the speaker's.
    pub min_similarity: f64,
    /// Candidates kept per clip and engine (best first, never overlapping).
    pub per_clip: usize,
}

impl Default for FindSettings {
    fn default() -> Self {
        FindSettings { min_similarity: MIN_SIMILARITY, per_clip: 3 }
    }
}

/// The default speaker check: tone-colour cosine to the voice's reference. Measured on the
/// author's own recordings (crate README, "Better references"): other stretches of the same
/// speaker score 0.9 and more, other speakers less.
pub const MIN_SIMILARITY: f64 = 0.85;
/// Speaker-check chunks: 6–12 s of phrases, gaps of at most 1 s, at least 2.5 s of speech.
const CHUNK: (f64, f64) = (6.0, 12.0);
const CHUNK_GAP: f64 = 1.0;
const CHUNK_SPEECH: f64 = 2.5;
/// Chunks by the speaker this close together are one run of their speech.
const RUN_GAP: f64 = 1.5;
/// A pause this long ends a run of phrases inside a reference.
const MAX_INNER_PAUSE: f64 = 1.2;
/// A pause this long marks a sentence end even where the words do not.
const SENTENCE_PAUSE: f64 = 0.7;
/// Longest piece heard at once (Whisper's window is 30 s).
const HEAR_SECONDS: f64 = 28.0;
/// Windows looked at per run of the speaker's speech and engine (whole-phrase runs within the
/// bounds), and how many of the best of them get the signal measures.
const MAX_WINDOWS: usize = 4000;
const MEASURED_PER_RUN: usize = 60;

/// A chunk of a clip checked against the speaker.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Checked {
    pub span: Span,
    pub similarity: f64,
}

/// A stretch of a clip that could be a voice's reference, and why.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    /// Set by [`rank`] (`voxcpm-1`, `indicf5-2`, …).
    pub id: String,
    /// The clip (its path).
    pub clip: String,
    pub span: Span,
    pub target: Target,
    /// The language heard (ISO 639-1, `ml` for Whisper's Malayalam).
    pub language: String,
    /// Tone-colour cosine to the speaker's reference.
    pub similarity: f64,
    pub quality: Quality,
    /// Whole sentences: 1 when it starts and ends where sentences do (by the words heard), less
    /// when only by its pauses.
    pub sentence: f64,
    /// It ends where a sentence does (by the words heard or a long pause after it).
    pub ends_sentence: bool,
    pub length_fit: f64,
    pub language_fit: f64,
    /// All of it ([`candidate_score`]).
    pub score: f64,
    /// What the speech model heard in it.
    pub words: String,
}

/// The weights of a candidate's score: speaker similarity first, then signal quality, whole
/// sentences, length and language.
pub fn candidate_score(similarity: f64, min_similarity: f64, quality: f64, sentence: f64, length: f64, language: f64) -> f64 {
    let sim = ((similarity - min_similarity) / (0.98 - min_similarity).max(1e-6)).clamp(0.0, 1.0);
    let s = 0.35 * sim + 0.30 * quality + 0.15 * sentence + 0.10 * length + 0.10 * language;
    if s.is_finite() { s } else { 0.0 }
}

/// What one clip held.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipScan {
    pub clip: String,
    pub seconds: f64,
    /// Every chunk checked, and how close it is to the speaker.
    pub chunks: Vec<Checked>,
    /// Seconds of the clip by the speaker (chunks at or above the threshold).
    pub speaker_seconds: f64,
    /// The language of most of the speaker's speech.
    pub language: Option<String>,
    pub candidates: Vec<Candidate>,
}

/// Whether the words make `w` whole sentences: 0.5 for each edge where a sentence ends by the
/// words heard (the word before ends a sentence, or there is none; its last word ends one), 0.35
/// where only a long pause says so, 0 where a word is cut or the sentence runs on.
pub fn sentence_fit(words: &[Word], w: Span, pause_before: f64, pause_after: f64) -> f64 {
    let (a, b) = sentence_edges(words, w, pause_before, pause_after);
    a + b
}

/// [`sentence_fit`]'s two edges, the start's and the end's: each 0.5 where a sentence ends by
/// the words heard, 0.35 where only a long pause says so, 0 where a word is cut or the sentence
/// runs on.
pub fn sentence_edges(words: &[Word], w: Span, pause_before: f64, pause_after: f64) -> (f64, f64) {
    let ends = |t: &str| t.trim_end_matches(['"', '\'', ')', '”', '’']).ends_with(['.', '?', '!', '।']);
    let straddles = |t: f64| words.iter().any(|x| x.start + 0.05 < t && t < x.end - 0.05);
    let edge = |cut: bool, by_words: bool, pause: f64| {
        if cut {
            0.0
        } else if by_words {
            0.5
        } else if pause >= SENTENCE_PAUSE {
            0.35
        } else {
            0.0
        }
    };
    let before = words.iter().rfind(|x| x.end <= w.start + 0.05);
    let inside: Vec<&Word> = words.iter().filter(|x| x.start >= w.start - 0.05 && x.end <= w.end + 0.05).collect();
    let start_by_words = !words.is_empty() && before.is_none_or(|b| ends(&b.text));
    let end_by_words = inside.last().is_some_and(|l| ends(&l.text));
    (edge(straddles(w.start), start_by_words, pause_before), edge(straddles(w.end), end_by_words, pause_after))
}

/// Runs of whole phrases that fit `b`: each from a phrase's start to a later phrase's end, with
/// no pause longer than [`MAX_INNER_PAUSE`] inside, edges padded into the pauses by up to 0.15 s.
/// Also returns the pauses before and after each.
pub fn windows(phrases: &[Span], b: &Bounds, limit: usize) -> Vec<(Span, f64, f64)> {
    let mut out = Vec::new();
    for i in 0..phrases.len() {
        let Some(first) = phrases.get(i) else { continue };
        let before = i.checked_sub(1).and_then(|k| phrases.get(k)).map_or(f64::INFINITY, |p| first.start - p.end);
        for j in i..phrases.len() {
            let Some(last) = phrases.get(j) else { break };
            if j > i && phrases.get(j - 1).is_some_and(|p| last.start - p.end > MAX_INNER_PAUSE) {
                break;
            }
            let after = phrases.get(j + 1).map_or(f64::INFINITY, |n| n.start - last.end);
            let pad = |gap: f64| (gap / 2.0).min(0.15);
            let w = Span::new((first.start - pad(before)).max(0.0), last.end + pad(after));
            let secs = w.seconds();
            if secs > b.max {
                break;
            }
            if secs >= b.min {
                out.push((w, before, after));
                if out.len() >= limit {
                    return out;
                }
            }
        }
    }
    out
}

/// Look for references in one clip (`x`, mono at `rate`) by the speaker whose embedding is
/// `speaker` (see the module docs). Errors from the models fail the clip.
pub fn scan(
    clip: &str,
    x: &[f32],
    rate: u32,
    speaker: &[f32],
    s: &FindSettings,
    m: &mut dyn Models,
    progress: &mut dyn FnMut(&str),
) -> Result<ClipScan, TtsError> {
    if rate == 0 {
        return Err(TtsError::Invalid("the clip has no sample rate".into()));
    }
    let f = signal::frames(x, rate);
    let a = signal::activity(&f);
    let phrases = signal::phrases(&f, &a);
    let mut out = ClipScan { clip: clip.to_string(), seconds: x.len() as f64 / f64::from(rate), ..Default::default() };
    // the speaker check, chunk by chunk
    for c in signal::chunks(&phrases, CHUNK, CHUNK_GAP, CHUNK_SPEECH) {
        let speech = signal::speech_of(x, &f, &a, c);
        if (speech.len() as f64) < 1.5 * f64::from(rate) {
            continue;
        }
        let sim = cosine(&m.colour(&speech, rate)?, speaker);
        out.chunks.push(Checked { span: c, similarity: sim });
    }
    let mine: Vec<Span> = out.chunks.iter().filter(|c| c.similarity >= s.min_similarity).map(|c| c.span).collect();
    out.speaker_seconds = mine.iter().map(Span::seconds).sum();
    if mine.is_empty() {
        return Ok(out);
    }
    // runs of the speaker's speech
    let mut runs: Vec<Span> = Vec::new();
    for c in mine {
        match runs.last_mut() {
            Some(r) if c.start - r.end <= RUN_GAP => r.end = r.end.max(c.end),
            _ => runs.push(c),
        }
    }
    progress(&format!("{clip}: {:.0} s by the speaker in {} runs", out.speaker_seconds, runs.len()));
    let mut by_language: Vec<(String, f64)> = Vec::new();
    let mut found: Vec<Candidate> = Vec::new();
    for run in &runs {
        let inside: Vec<Span> = phrases.iter().copied().filter(|p| p.start >= run.start - 0.01 && p.end <= run.end + 0.01).collect();
        // heard in pieces of at most HEAR_SECONDS, cut between phrases
        let mut words: Vec<Word> = Vec::new();
        let mut langs: Vec<(String, f64)> = Vec::new();
        let mut piece: Option<Span> = None;
        let mut pieces: Vec<Span> = Vec::new();
        for p in &inside {
            piece = match piece {
                Some(c) if p.end - c.start <= HEAR_SECONDS => Some(Span::new(c.start, p.end)),
                Some(c) => {
                    pieces.push(c);
                    Some(*p)
                }
                None => Some(*p),
            };
        }
        pieces.extend(piece);
        for pc in pieces {
            let w = pc.widened(0.1);
            let heard = m.transcribe(signal::cut(x, rate, w), rate)?;
            langs.push((normalise_language(&heard.language), pc.seconds()));
            words.extend(heard.words.into_iter().map(|x| Word { text: x.text, start: x.start + w.start, end: x.end + w.start }));
        }
        let language = majority(&langs).unwrap_or_else(|| "unknown".into());
        by_language.push((language.clone(), run.seconds()));
        let sim_of = |w: &Span| {
            let (mut num, mut den) = (0.0, 0.0);
            for c in out.chunks.iter().filter(|c| c.span.overlaps(w)) {
                let o = c.span.end.min(w.end) - c.span.start.max(w.start);
                num += o * c.similarity;
                den += o;
            }
            if den > 0.0 { num / den } else { 0.0 }
        };
        for t in Target::ALL {
            let b = t.bounds();
            let lang_fit = t.language_fit(&language);
            // scored without the signal measures first (an average quality), the best measured
            let mut pre: Vec<(f64, Span, f64, f64)> = windows(&inside, &b, MAX_WINDOWS)
                .into_iter()
                .map(|(w, before, after)| {
                    let sentence = sentence_fit(&words, w, before, after);
                    (candidate_score(sim_of(&w), s.min_similarity, 0.5, sentence, length_fit(w.seconds(), &b), lang_fit), w, before, after)
                })
                .collect();
            pre.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (_, w, before, after) in pre.into_iter().take(MEASURED_PER_RUN) {
                let q = signal::quality(x, &f, &a, w, w.widened(3.0));
                let (start_edge, end_edge) = sentence_edges(&words, w, before, after);
                let sentence = start_edge + end_edge;
                let length = length_fit(w.seconds(), &b);
                let sim = sim_of(&w);
                let said: Vec<&str> = words.iter().filter(|x| x.start >= w.start - 0.05 && x.end <= w.end + 0.05).map(|x| x.text.as_str()).collect();
                found.push(Candidate {
                    id: String::new(),
                    clip: clip.to_string(),
                    span: w,
                    target: t,
                    language: language.clone(),
                    similarity: sim,
                    score: candidate_score(sim, s.min_similarity, q.score, sentence, length, lang_fit),
                    quality: q,
                    sentence,
                    ends_sentence: end_edge > 0.0,
                    length_fit: length,
                    language_fit: lang_fit,
                    words: said.join(" "),
                });
            }
        }
    }
    out.language = majority(&by_language);
    // the best few per engine, never overlapping, with their own embedding
    for t in Target::ALL {
        let mut of: Vec<Candidate> = found.iter().filter(|c| c.target == t).cloned().collect();
        of.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut kept: Vec<Candidate> = Vec::new();
        for mut c in of {
            if kept.len() >= s.per_clip {
                break;
            }
            if kept.iter().any(|k| k.span.overlaps(&c.span)) {
                continue;
            }
            let speech = signal::speech_of(x, &f, &a, c.span);
            if (speech.len() as f64) >= 1.5 * f64::from(rate) {
                c.similarity = cosine(&m.colour(&speech, rate)?, speaker);
                c.score = candidate_score(c.similarity, s.min_similarity, c.quality.score, c.sentence, c.length_fit, c.language_fit);
            }
            if c.similarity >= s.min_similarity {
                kept.push(c);
            }
        }
        out.candidates.extend(kept);
    }
    Ok(out)
}

/// The language with the most seconds.
fn majority(v: &[(String, f64)]) -> Option<String> {
    let mut tally: Vec<(String, f64)> = Vec::new();
    for (l, s) in v {
        match tally.iter_mut().find(|(k, _)| k == l) {
            Some((_, t)) => *t += s,
            None => tally.push((l.clone(), *s)),
        }
    }
    tally.into_iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|(l, _)| l)
}

/// The best `top` candidates per engine over every clip, at most `per_clip` from one clip (the
/// audition compares recordings, not neighbouring cuts of one), never overlapping; ids given in
/// rank order (`voxcpm-1`, …).
pub fn rank(all: &[Candidate], top: usize, per_clip: usize) -> Vec<Candidate> {
    let mut out = Vec::new();
    for t in Target::ALL {
        let mut of: Vec<&Candidate> = all.iter().filter(|c| c.target == t).collect();
        of.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut kept: Vec<Candidate> = Vec::new();
        for c in of {
            if kept.len() >= top {
                break;
            }
            let from_clip = kept.iter().filter(|k| k.clip == c.clip).count();
            if from_clip >= per_clip || kept.iter().any(|k| k.clip == c.clip && k.span.overlaps(&c.span)) {
                continue;
            }
            kept.push(c.clone());
        }
        for (i, c) in kept.iter_mut().enumerate() {
            c.id = format!("{}-{}", t.id(), i + 1);
        }
        out.extend(kept);
    }
    out
}

/// Held-out genuine speech of the speaker: chunks at or above `min_similarity` that overlap none
/// of `exclude` (each widened by a second), taken a clip at a time in turn so many recordings are
/// in it, up to `max_seconds`.
pub fn held_out(scans: &[ClipScan], exclude: &[(String, Span)], min_similarity: f64, max_seconds: f64) -> Vec<(String, Span)> {
    let mut queues: Vec<(String, Vec<Span>)> = scans
        .iter()
        .map(|sc| {
            let mut c: Vec<&Checked> = sc
                .chunks
                .iter()
                .filter(|c| c.similarity >= min_similarity)
                .filter(|c| !exclude.iter().any(|(clip, sp)| *clip == sc.clip && sp.widened(1.0).overlaps(&c.span)))
                .collect();
            // the clearest first
            c.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
            (sc.clip.clone(), c.into_iter().map(|c| c.span).rev().collect::<Vec<Span>>())
        })
        .filter(|(_, v)| !v.is_empty())
        .collect();
    let mut out = Vec::new();
    let mut total = 0.0;
    while total < max_seconds && queues.iter().any(|(_, v)| !v.is_empty()) {
        for (clip, v) in queues.iter_mut() {
            if let Some(sp) = v.pop() {
                total += sp.seconds();
                out.push((clip.clone(), sp));
                if total >= max_seconds {
                    break;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "styles_tests.rs"]
mod styles_tests;
