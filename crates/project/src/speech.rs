//! What speech understanding found in a media clip (`media.understandSpeech`): speech segments
//! with their speaker, text, words and confidence; pauses; filler words; retakes (the same line
//! said again) with the take to keep; and whether the clip looks like a talking head.
//!
//! An analysis belongs to one media item (`Project::speech`, keyed by the item) and is in media
//! time, like the item's transcript (`Project::transcripts`), whose word indices it refers to: it
//! stays valid however the clip is trimmed, moved or reused. It is derived data (running the
//! command again rebuilds it), so older builds that drop it lose nothing that cannot be remade.

use filmcraft_time::{Tick, TimeRange};
use serde::{Deserialize, Serialize};

/// A stretch of media time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub start: Tick,
    pub end: Tick,
}

impl Span {
    pub fn new(start: Tick, end: Tick) -> Self {
        Self { start, end: end.max(start) }
    }
    pub fn range(&self) -> TimeRange {
        TimeRange::from_bounds(self.start, self.end.max(self.start))
    }
    pub fn duration(&self) -> Tick {
        (self.end - self.start).max(Tick::ZERO)
    }
}

/// A sentence (or a run of words between pauses) of one speaker.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SpeechSegment {
    pub start: Tick,
    pub end: Tick,
    /// Index into the transcript's speakers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<u32>,
    pub text: String,
    /// First and last word (inclusive) in the item's transcript.
    pub from: usize,
    pub to: usize,
    /// Mean recogniser confidence of its words, 0..1.
    pub confidence: f32,
    /// How cleanly it was said, 0..1 (fewer fillers and restarts, confident words, no clipping):
    /// what picks the take to keep.
    pub fluency: f32,
    /// Filler words in it.
    pub fillers: u32,
    /// Words said twice in a row ("the the"): restarts.
    pub restarts: u32,
    /// The audio clips (full-scale samples) in it.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub clipped: bool,
}

/// A filler: a filler word or phrase of the transcript, or a hesitation (speech the recogniser
/// wrote no word for, between words).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Filler {
    pub start: Tick,
    pub end: Tick,
    pub text: String,
    /// The words (inclusive) when it is in the transcript; `None` for a hesitation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<usize>,
}

impl Filler {
    pub fn is_hesitation(&self) -> bool {
        self.from.is_none()
    }
}

/// Takes of one line: segments saying (nearly) the same thing, in order, and the one to keep.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Retake {
    /// Segment indices, in order.
    pub takes: Vec<usize>,
    /// The segment kept (one of `takes`).
    pub keep: usize,
    /// How alike the takes are, 0..1 (the lowest of the pairs).
    pub similarity: f32,
    /// Why: `repeat` (the line said again) or `falseStart` (a start abandoned and said again).
    pub reason: String,
}

impl Retake {
    /// The takes that go.
    pub fn dropped(&self) -> impl Iterator<Item = usize> + '_ {
        self.takes.iter().copied().filter(move |t| *t != self.keep)
    }
}

/// What the clip looks like.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ShotHint {
    /// Speech and a face on screen.
    TalkingHead,
    /// Speech without a face (an off-screen voice, a screen recording).
    Voice,
    /// Little or no speech: cutaway material.
    BRoll,
    /// Not known (no picture, or faces could not be looked for).
    #[default]
    Unknown,
}

/// What speech understanding found in one media item.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SpeechAnalysis {
    /// ISO 639-1 code of the spoken language.
    pub language: String,
    /// The recogniser (`"whisper-medium-ml"`).
    pub source: String,
    /// How speech was found: `silero` or `energy`.
    pub vad: String,
    /// How speakers were told apart: `ecapa`, `mfcc` or `off`.
    pub diarization: String,
    /// The item's length.
    pub duration: Tick,
    /// Where someone speaks (voice activity).
    pub speech: Vec<Span>,
    pub segments: Vec<SpeechSegment>,
    /// Silences between words (at least the analysis' shortest pause).
    pub pauses: Vec<Span>,
    pub fillers: Vec<Filler>,
    pub retakes: Vec<Retake>,
    pub shot: ShotHint,
    /// Whether a face was seen (`None`: not looked for).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub face: Option<bool>,
}

impl SpeechAnalysis {
    /// Segment indices dropped by retakes (every take but the kept one), sorted.
    pub fn dropped_takes(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.retakes.iter().flat_map(|r| r.dropped().collect::<Vec<_>>()).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The share of the item that is speech, 0..1.
    pub fn speech_share(&self) -> f64 {
        if self.duration <= Tick::ZERO {
            return 0.0;
        }
        let s: i64 = self.speech.iter().map(|s| s.duration().0).sum();
        (s as f64 / self.duration.0 as f64).clamp(0.0, 1.0)
    }

    /// Check the invariants the editing commands rely on against a transcript of `words` words:
    /// spans ordered, word ranges in bounds, retakes pointing at segments.
    pub fn check(&self, words: usize) -> Result<(), String> {
        for (i, s) in self.segments.iter().enumerate() {
            if s.end < s.start || s.from > s.to || s.to >= words {
                return Err(format!("segment {i} is malformed (words {}–{} of {words})", s.from, s.to));
            }
        }
        for f in &self.fillers {
            if f.end < f.start || f.to.is_some_and(|t| t >= words) || f.from.zip(f.to).is_some_and(|(a, b)| a > b) {
                return Err("a filler is malformed".into());
            }
        }
        for r in &self.retakes {
            if r.takes.is_empty() || !r.takes.contains(&r.keep) || r.takes.iter().any(|t| *t >= self.segments.len()) {
                return Err("a retake is malformed".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropped_takes_and_checks() {
        let seg = |from, to| SpeechSegment { from, to, end: Tick(10), ..Default::default() };
        let mut a = SpeechAnalysis { segments: vec![seg(0, 1), seg(2, 3), seg(4, 5)], ..Default::default() };
        a.retakes.push(Retake { takes: vec![0, 1, 2], keep: 2, similarity: 0.9, reason: "repeat".into() });
        assert_eq!(a.dropped_takes(), vec![0, 1]);
        a.check(6).unwrap();
        assert!(a.check(5).is_err());
        a.retakes[0].keep = 7;
        assert!(a.check(6).is_err());
        // round trip
        let j = serde_json::to_string(&a).unwrap();
        let b: SpeechAnalysis = serde_json::from_str(&j).unwrap();
        assert_eq!(a, b);
        // an empty object reads
        let e: SpeechAnalysis = serde_json::from_str("{}").unwrap();
        assert_eq!(e.shot, ShotHint::Unknown);
        assert_eq!(e.speech_share(), 0.0);
    }
}
