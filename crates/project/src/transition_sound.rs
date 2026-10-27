//! Sounds coupled to video transitions (roadmap C4): a whoosh, riser, impact, swish, chime, camera
//! shutter, bubble pop or thump that belongs to a transition, synthesised when the sequence is mixed (`filmcraft-render`'s mixer, with
//! `filmcraft_audio_dsp::synth`), so it moves, resizes and goes away with its transition.
//!
//! Where it plays is fixed relative to the transition's midpoint (the moment the picture swaps):
//!
//! | kind | default length | placement |
//! |---|---|---|
//! | whoosh | the transition + ½ s (0.6–4 s) | centred on the midpoint (it swells in its middle) |
//! | riser | 2 s | ends on the midpoint (it builds to the swap and stops) |
//! | impact | 2.5 s | starts on the midpoint (it hits on the swap and decays) |
//! | swish | the transition + ⅕ s (0.35–1 s) | centred on the midpoint (a soft, short brush) |
//! | chime | 1.8 s | starts on the midpoint (it rings from the swap) |
//! | shutter | ¼ s | starts [`SHUTTER_LEAD`] (2 ms) before the midpoint (its first click lands on the swap) |
//! | pop | ¼ s | starts on the midpoint (a soft bubble pop on the swap) |
//! | thump | 0.8 s | starts on the midpoint (a soft, warm low hit on the swap) |
//!
//! The span is in exact ticks and is turned into whole samples at the sequence's rate by
//! [`TransitionSound::samples`] (rounded to the nearest sample), the same numbers the mixer and
//! the commands report, so a sound is placed to the sample.

use filmcraft_time::{TICKS_PER_SECOND, Tick};
use serde::{Deserialize, Serialize};

/// Longest coupled sound (seconds), the synth's own limit for sound effects.
pub const MAX_SECONDS: f64 = 30.0;
/// Quietest and loudest gain (dB).
pub const GAIN_RANGE: (f64, f64) = (-60.0, 12.0);
/// How far into a shutter its first click is (`filmcraft_audio_dsp::synth::SHUTTER_LEAD`, 2 ms):
/// it starts this much before the swap.
pub const SHUTTER_LEAD: Tick = Tick(TICKS_PER_SECOND / 500);

/// What a transition sounds like.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SoundKind {
    /// Air rushing past, centred on the swap.
    #[default]
    Whoosh,
    /// Tension building up to the swap.
    Riser,
    /// A hit on the swap.
    Impact,
    /// A soft, short brush of air on the swap (gentler than the whoosh).
    Swish,
    /// A soft bell from the swap.
    Chime,
    /// A camera shutter: two quick mechanical clicks, the first on the swap.
    Shutter,
    /// A soft bubble pop on the swap.
    Pop,
    /// A soft, warm low hit on the swap (shorter than the impact, no sub-bass).
    Thump,
}

impl SoundKind {
    pub const ALL: [SoundKind; 8] =
        [SoundKind::Whoosh, SoundKind::Riser, SoundKind::Impact, SoundKind::Swish, SoundKind::Chime, SoundKind::Shutter, SoundKind::Pop, SoundKind::Thump];

    pub fn id(self) -> &'static str {
        match self {
            SoundKind::Whoosh => "whoosh",
            SoundKind::Riser => "riser",
            SoundKind::Impact => "impact",
            SoundKind::Swish => "swish",
            SoundKind::Chime => "chime",
            SoundKind::Shutter => "shutter",
            SoundKind::Pop => "pop",
            SoundKind::Thump => "thump",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SoundKind::Whoosh => "Whoosh",
            SoundKind::Riser => "Riser",
            SoundKind::Impact => "Impact",
            SoundKind::Swish => "Swish",
            SoundKind::Chime => "Chime",
            SoundKind::Shutter => "Shutter",
            SoundKind::Pop => "Pop",
            SoundKind::Thump => "Thump",
        }
    }

    pub fn from_id(s: &str) -> Option<SoundKind> {
        SoundKind::ALL.into_iter().find(|k| k.id().eq_ignore_ascii_case(s.trim()) || k.label().eq_ignore_ascii_case(s.trim()))
    }
}

fn default_gain() -> f64 {
    -6.0
}
fn default_intensity() -> f64 {
    0.7
}

/// A sound coupled to a transition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionSound {
    pub kind: SoundKind,
    /// Level in dB (clamped to [`GAIN_RANGE`] when mixed).
    #[serde(default = "default_gain")]
    pub gain_db: f64,
    /// The synth's intensity, 0..1 (brightness, density and level).
    #[serde(default = "default_intensity")]
    pub intensity: f64,
    /// The synth's seed: the same seed always gives the same sound.
    #[serde(default)]
    pub seed: u64,
    /// Length; None = the kind's default (see the module docs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length: Option<Tick>,
}

impl TransitionSound {
    pub fn new(kind: SoundKind, seed: u64) -> Self {
        Self { kind, gain_db: default_gain(), intensity: default_intensity(), seed, length: None }
    }

    /// The sound's length for a transition of `duration` (at least one sample's worth of ticks
    /// at any rate, at most [`MAX_SECONDS`]).
    pub fn length_for(&self, duration: Tick) -> Tick {
        let most = Tick::from_seconds_f64(MAX_SECONDS);
        let least = Tick::from_seconds_f64(0.05);
        let len = match self.length {
            Some(l) => l,
            None => match self.kind {
                SoundKind::Whoosh => (duration.max(Tick::ZERO) + Tick(TICKS_PER_SECOND / 2)).clamp(Tick::from_seconds_f64(0.6), Tick::from_seconds_f64(4.0)),
                SoundKind::Riser => Tick(TICKS_PER_SECOND * 2),
                SoundKind::Impact => Tick(TICKS_PER_SECOND * 5 / 2),
                SoundKind::Swish => (duration.max(Tick::ZERO) + Tick(TICKS_PER_SECOND / 5)).clamp(Tick::from_seconds_f64(0.35), Tick::from_seconds_f64(1.0)),
                SoundKind::Chime => Tick(TICKS_PER_SECOND * 9 / 5),
                SoundKind::Shutter | SoundKind::Pop => Tick(TICKS_PER_SECOND / 4),
                SoundKind::Thump => Tick(TICKS_PER_SECOND * 4 / 5),
            },
        };
        len.clamp(least, most)
    }

    /// Where the sound plays on the timeline for a transition at `start` lasting `duration`:
    /// (start, length) in ticks.
    pub fn span(&self, start: Tick, duration: Tick) -> (Tick, Tick) {
        let mid = start + duration.max(Tick::ZERO).mul_ratio(1, 2);
        let len = self.length_for(duration);
        let at = match self.kind {
            SoundKind::Whoosh | SoundKind::Swish => mid - len.mul_ratio(1, 2),
            SoundKind::Riser => mid - len,
            SoundKind::Impact | SoundKind::Chime | SoundKind::Pop | SoundKind::Thump => mid,
            SoundKind::Shutter => mid - SHUTTER_LEAD,
        };
        (at, len)
    }

    /// The span in whole samples at `rate` (the nearest sample to each end): (first sample,
    /// sample count). None for a rate of zero.
    pub fn samples(&self, start: Tick, duration: Tick, rate: u32) -> Option<(i64, usize)> {
        if rate == 0 {
            return None;
        }
        let (at, len) = self.span(start, duration);
        let first = at.to_rational_round(1, i64::from(rate));
        let last = (at + len).to_rational_round(1, i64::from(rate));
        let n = usize::try_from(last.saturating_sub(first)).ok()?.max(1);
        Some((first, n))
    }

    /// The gain as a linear factor (clamped to [`GAIN_RANGE`]; NaN is silence).
    pub fn gain(&self) -> f32 {
        if !self.gain_db.is_finite() {
            return 0.0;
        }
        let db = self.gain_db.clamp(GAIN_RANGE.0, GAIN_RANGE.1);
        10f64.powf(db / 20.0) as f32
    }

    /// The intensity, clamped to 0..1 (NaN = the default).
    pub fn intensity(&self) -> f64 {
        if self.intensity.is_finite() { self.intensity.clamp(0.0, 1.0) } else { default_intensity() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_follows_the_midpoint() {
        let s = |k| TransitionSound::new(k, 1);
        let start = Tick::from_seconds_f64(10.0);
        let dur = Tick::from_seconds_f64(1.0);
        let mid = Tick::from_seconds_f64(10.5);
        let (at, len) = s(SoundKind::Whoosh).span(start, dur);
        assert_eq!(len, Tick::from_seconds_f64(1.5));
        assert_eq!(at + len.mul_ratio(1, 2), mid);
        let (at, len) = s(SoundKind::Riser).span(start, dur);
        assert_eq!(at + len, mid);
        let (at, _) = s(SoundKind::Impact).span(start, dur);
        assert_eq!(at, mid);
        // to the sample at 48 kHz: 10.5 s is sample 504 000
        assert_eq!(s(SoundKind::Impact).samples(start, dur, 48_000), Some((504_000, 120_000)));
        assert_eq!(s(SoundKind::Riser).samples(start, dur, 48_000), Some((504_000 - 96_000, 96_000)));
        assert_eq!(s(SoundKind::Whoosh).samples(start, dur, 48_000), Some((504_000 - 36_000, 72_000)));
        assert_eq!(s(SoundKind::Whoosh).samples(start, dur, 0), None);
        // the soft pair: a short swish centred on the swap, a chime ringing from it
        let (at, len) = s(SoundKind::Swish).span(start, dur);
        assert_eq!(len, Tick::from_seconds_f64(1.0));
        assert_eq!(at + len.mul_ratio(1, 2), mid);
        assert_eq!(s(SoundKind::Swish).span(start, Tick::ZERO).1, Tick::from_seconds_f64(0.35));
        assert_eq!(s(SoundKind::Chime).samples(start, dur, 48_000), Some((504_000, 86_400)));
        // the shutter's first click on the swap (2 ms in), the pop and the thump from it; their
        // length is their own, whatever the transition's
        assert_eq!(s(SoundKind::Shutter).samples(start, dur, 48_000), Some((504_000 - 96, 12_000)));
        assert_eq!(s(SoundKind::Pop).samples(start, dur, 48_000), Some((504_000, 12_000)));
        assert_eq!(s(SoundKind::Thump).samples(start, dur, 48_000), Some((504_000, 38_400)));
        assert_eq!(s(SoundKind::Thump).span(start, Tick::from_seconds_f64(4.0)).1, Tick::from_seconds_f64(0.8));
        for k in SoundKind::ALL {
            assert_eq!(SoundKind::from_id(k.id()), Some(k));
            assert_eq!(SoundKind::from_id(k.label()), Some(k));
        }
    }

    #[test]
    fn hostile_values_stay_bounded() {
        let mut s = TransitionSound::new(SoundKind::Whoosh, 0);
        for len in [Tick(i64::MIN), Tick(-1), Tick(0), Tick(i64::MAX)] {
            s.length = Some(len);
            let l = s.length_for(Tick(i64::MAX));
            assert!(l >= Tick::from_seconds_f64(0.05) && l <= Tick::from_seconds_f64(MAX_SECONDS), "{l:?}");
            let (_, n) = s.samples(Tick(i64::MAX / 2), Tick(i64::MAX), 48_000).unwrap();
            assert!(n <= (MAX_SECONDS * 48_000.0) as usize + 1);
            assert!(s.samples(Tick(i64::MIN), Tick(i64::MIN), u32::MAX).is_some());
        }
        for g in [f64::NAN, f64::INFINITY, -1e300, 1e300] {
            s.gain_db = g;
            assert!(s.gain().is_finite() && s.gain() <= 4.0);
        }
        s.intensity = f64::NAN;
        assert_eq!(s.intensity(), 0.7);
        assert_eq!(SoundKind::from_id(" Riser "), Some(SoundKind::Riser));
        assert_eq!(SoundKind::from_id("drone"), None);
        // serde: lower-case kinds, defaults for missing fields
        let t: TransitionSound = serde_json::from_str(r#"{"kind":"impact"}"#).unwrap();
        assert_eq!((t.kind, t.gain_db, t.intensity, t.seed, t.length), (SoundKind::Impact, -6.0, 0.7, 0, None));
    }
}
